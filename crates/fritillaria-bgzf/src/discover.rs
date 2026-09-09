//! Stage 1: block discovery.
//!
//! A sequential walk of block headers producing one [`BlockSpan`] per block.
//! This is cheap and I/O-bound — it touches only the ~18-byte header and the
//! 8-byte trailer of each block, never the payload — and it is what makes the
//! *next* stage parallel: with every `ISIZE` known up front, output offsets are
//! a prefix sum and all blocks can be inflated at once.

use fritillaria_core::{BlockSpan, Error, MAX_BLOCK_SIZE, Result};

use crate::block::{TRAILER_SIZE, header_len, parse_header, read_u32};

/// Iterator over the blocks in a contiguous BGZF byte range.
///
/// `base_offset` is the absolute file position of `buf[0]`, so spans carry true
/// file offsets even when walking a slice of a larger file.
#[derive(Debug)]
pub struct BlockDiscovery<'a> {
    buf: &'a [u8],
    pos: usize,
    base_offset: u64,
    done: bool,
}

impl<'a> BlockDiscovery<'a> {
    /// Starts a walk over `buf`, whose first byte sits at `base_offset`.
    #[must_use]
    pub fn new(buf: &'a [u8], base_offset: u64) -> Self {
        Self {
            buf,
            pos: 0,
            base_offset,
            done: false,
        }
    }

    /// Byte offset just past the last successfully parsed block.
    ///
    /// When a walk stops early because the buffer ends mid-block, this is where
    /// the next read should resume — the caller keeps the remainder and
    /// prepends it to the next chunk.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl Iterator for BlockDiscovery<'_> {
    type Item = Result<BlockSpan>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.buf.len() {
            return None;
        }

        let offset = self.base_offset + self.pos as u64;
        let rest = &self.buf[self.pos..];

        // Too few bytes to even measure the header: the buffer ended
        // mid-header. That is a resume point, not corruption — only the caller
        // knows whether more bytes are coming.
        match header_len(rest) {
            Some(len) if len <= rest.len() => {}
            _ => {
                self.done = true;
                return None;
            }
        }

        let header = match parse_header(rest, offset) {
            Ok(header) => header,
            Err(err) => {
                self.done = true;
                return Some(Err(err));
            }
        };

        // A block whose tail is past the end of the buffer is not an error —
        // the caller may simply have read a partial chunk. Stop and let
        // `position()` drive the resume.
        if header.block_size > rest.len() {
            self.done = true;
            return None;
        }

        let block = &rest[..header.block_size];
        let crc32 = read_u32(block, header.block_size - TRAILER_SIZE)?;
        let isize = read_u32(block, header.block_size - 4)?;

        if isize as usize > MAX_BLOCK_SIZE {
            self.done = true;
            return Some(Err(Error::InvalidBlock {
                offset,
                reason: "ISIZE exceeds 64 KiB",
            }));
        }

        let span = BlockSpan {
            compressed_offset: offset,
            payload_start: self.pos + header.header_size,
            payload_len: header.payload_size(),
            isize,
            crc32,
        };

        self.pos += header.block_size;
        Some(Ok(span))
    }
}

/// Collects every block in `buf` into a vector of spans.
///
/// Convenience over [`BlockDiscovery`] for whole-file, in-memory use. Streaming
/// callers should drive the iterator so they can act on `position()`.
pub fn discover_blocks(buf: &[u8], base_offset: u64) -> Result<Vec<BlockSpan>> {
    BlockDiscovery::new(buf, base_offset).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::EOF_BLOCK;
    use crate::write::BgzfWriter;

    fn bgzf(payloads: &[&[u8]]) -> Vec<u8> {
        let mut writer = BgzfWriter::new(Vec::new());
        for payload in payloads {
            writer.write_block(payload).unwrap();
        }
        writer.finish().unwrap()
    }

    #[test]
    fn discovers_every_block_including_eof() {
        let data = bgzf(&[b"alpha", b"beta", b"gamma"]);
        let spans = discover_blocks(&data, 0).unwrap();

        // Three payload blocks plus the terminating EOF block.
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[0].isize, 5);
        assert_eq!(spans[1].isize, 4);
        assert_eq!(spans[2].isize, 5);
        assert_eq!(spans[3].isize, 0, "EOF block carries no payload");
    }

    #[test]
    fn spans_tile_the_input_without_gaps() {
        // Every byte of the file must belong to exactly one block; a gap means
        // a block size was misread.
        let data = bgzf(&[b"one", b"two"]);
        let spans = discover_blocks(&data, 0).unwrap();

        let mut cursor = 0u64;
        for span in &spans {
            assert_eq!(
                span.compressed_offset, cursor,
                "gap or overlap between blocks"
            );
            cursor += (span.payload_start as u64 - span.compressed_offset)
                + span.payload_len as u64
                + TRAILER_SIZE as u64;
        }
        assert_eq!(cursor, data.len() as u64, "spans must cover the whole file");
    }

    #[test]
    fn base_offset_shifts_reported_positions() {
        let data = bgzf(&[b"payload"]);
        let spans = discover_blocks(&data, 1_000_000).unwrap();
        assert_eq!(spans[0].compressed_offset, 1_000_000);
    }

    #[test]
    fn stops_cleanly_on_a_partial_trailing_block() {
        // Simulates a chunked read landing mid-block: the complete blocks are
        // yielded, the partial one is not, and position() says where to resume.
        let data = bgzf(&[b"first", b"second"]);
        let first_len = parse_header(&data, 0).unwrap().block_size;
        let truncated = &data[..first_len + 10];

        let mut discovery = BlockDiscovery::new(truncated, 0);
        let spans: Vec<_> = discovery.by_ref().collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(spans.len(), 1, "partial block must not be yielded");
        assert_eq!(
            discovery.position(),
            first_len,
            "resume point must be the start of the partial block"
        );
    }

    #[test]
    fn any_truncation_point_stops_cleanly_and_resumes_correctly() {
        // Chunk boundaries fall wherever the reader happens to stop, including
        // mid-header and mid-trailer. Every one of those must be a clean stop
        // with an accurate resume point, never an error.
        let data = bgzf(&[b"first", b"second"]);

        // End offset of every block in the complete file.
        let mut ends = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            pos += parse_header(&data[pos..], 0).unwrap().block_size;
            ends.push(pos);
        }

        for cut in 0..=data.len() {
            let mut discovery = BlockDiscovery::new(&data[..cut], 0);
            let spans: Vec<_> = discovery
                .by_ref()
                .collect::<Result<Vec<_>>>()
                .unwrap_or_else(|e| panic!("truncation at {cut} must not error: {e:?}"));

            // Exactly those blocks that fit entirely within the cut.
            let complete = ends.iter().filter(|&&end| end <= cut).count();
            assert_eq!(spans.len(), complete, "wrong span count at cut {cut}");

            let resume = if complete == 0 { 0 } else { ends[complete - 1] };
            assert_eq!(
                discovery.position(),
                resume,
                "wrong resume point at cut {cut}"
            );
        }
    }

    #[test]
    fn empty_input_yields_no_blocks() {
        assert!(discover_blocks(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn eof_only_file_is_valid() {
        // A legal, empty BGZF file — not an error, and a common edge case.
        let spans = discover_blocks(&EOF_BLOCK, 0).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].isize, 0);
    }

    #[test]
    fn propagates_corruption_as_an_error() {
        let mut data = bgzf(&[b"payload"]);
        data[0] = 0x00; // break the magic
        assert!(discover_blocks(&data, 0).is_err());
    }
}
