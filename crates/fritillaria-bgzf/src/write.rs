//! A CPU BGZF writer.
//!
//! Its jobs are to generate test fixtures and to be the ratio/throughput
//! baseline the GPU compressor is measured against. Output must be readable by
//! `samtools`; that is the acceptance bar, not byte-equality with `bgzip`.

use std::io::{self, Write};

use fritillaria_core::{Error, MAX_BLOCK_SIZE, Result};

use crate::block::{EOF_BLOCK, FIXED_HEADER_SIZE, TRAILER_SIZE};

/// Default uncompressed bytes per block.
///
/// 65280 (`0xff00`), matching htslib. The 256-byte gap below 64 KiB is
/// headroom: deflate can expand incompressible input slightly, and the result
/// still has to fit the hard 64 KiB cap once framing is added.
pub const DEFAULT_PAYLOAD_SIZE: usize = 0xff00;

/// Size of the BGZF extra field: the `BC` subfield and nothing else.
const XLEN: usize = 6;
/// Header size for the headers this writer emits.
const HEADER_SIZE: usize = FIXED_HEADER_SIZE + XLEN;
/// Largest deflate payload that still leaves room for framing.
const MAX_PAYLOAD: usize = MAX_BLOCK_SIZE - HEADER_SIZE - TRAILER_SIZE;

/// Writes BGZF blocks to an underlying sink.
///
/// The EOF block is written by [`BgzfWriter::finish`]. Dropping the writer
/// without calling it produces a file every downstream tool reports as
/// truncated, so `finish` is not optional.
#[derive(Debug)]
pub struct BgzfWriter<W: Write> {
    inner: W,
    payload_size: usize,
    level: u8,
    buf: Vec<u8>,
}

impl<W: Write> BgzfWriter<W> {
    /// Creates a writer with default block size and compression level.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            payload_size: DEFAULT_PAYLOAD_SIZE,
            level: 6,
            buf: Vec::new(),
        }
    }

    /// Sets the deflate level (0-10 as interpreted by `miniz_oxide`).
    #[must_use]
    pub fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// Sets the uncompressed bytes per block.
    ///
    /// Clamped to the format maximum. Smaller blocks mean more parallelism and
    /// finer seek granularity at the cost of ratio.
    #[must_use]
    pub fn with_payload_size(mut self, size: usize) -> Self {
        self.payload_size = size.clamp(1, MAX_PAYLOAD);
        self
    }

    /// Compresses `payload` as exactly one BGZF block.
    ///
    /// An empty payload is legal and produces a valid empty block.
    pub fn write_block(&mut self, payload: &[u8]) -> Result<()> {
        if payload.len() > MAX_PAYLOAD {
            return Err(Error::Malformed {
                format: "bgzf",
                position: 0,
                reason: format!(
                    "block payload {} exceeds maximum {MAX_PAYLOAD}",
                    payload.len()
                ),
            });
        }

        let compressed = miniz_oxide::deflate::compress_to_vec(payload, self.level);
        let block_size = HEADER_SIZE + compressed.len() + TRAILER_SIZE;

        // Deflate can expand pathologically incompressible input past the cap.
        // Rather than emit an invalid block, split and retry.
        if block_size > MAX_BLOCK_SIZE {
            let mid = payload.len() / 2;
            self.write_block(&payload[..mid])?;
            return self.write_block(&payload[mid..]);
        }

        self.buf.clear();
        self.buf.extend_from_slice(&[
            0x1f, 0x8b, 0x08, 0x04, // magic, deflate, FEXTRA
            0x00, 0x00, 0x00, 0x00, // MTIME: zero, so output is reproducible
            0x00, 0xff, // XFL, OS = unknown
        ]);
        self.buf.extend_from_slice(&(XLEN as u16).to_le_bytes());
        self.buf.extend_from_slice(&[b'B', b'C', 0x02, 0x00]);
        // BC holds size MINUS ONE.
        self.buf
            .extend_from_slice(&((block_size - 1) as u16).to_le_bytes());
        self.buf.extend_from_slice(&compressed);
        self.buf
            .extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
        self.buf
            .extend_from_slice(&(payload.len() as u32).to_le_bytes());

        debug_assert_eq!(self.buf.len(), block_size);
        self.inner.write_all(&self.buf)?;
        Ok(())
    }

    /// Splits `data` into blocks of the configured payload size and writes them.
    pub fn write_data(&mut self, data: &[u8]) -> Result<()> {
        for chunk in data.chunks(self.payload_size) {
            self.write_block(chunk)?;
        }
        Ok(())
    }

    /// Writes the EOF block and returns the underlying sink.
    pub fn finish(mut self) -> Result<W> {
        self.inner.write_all(&EOF_BLOCK)?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for BgzfWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_data(buf)
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{is_eof_block, parse_header};
    use crate::discover::discover_blocks;

    #[test]
    fn output_ends_with_the_eof_block() {
        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(b"data").unwrap();
        let out = writer.finish().unwrap();

        assert!(
            is_eof_block(&out[out.len() - EOF_BLOCK.len()..]),
            "missing EOF block: samtools would report truncation"
        );
    }

    #[test]
    fn empty_file_is_just_the_eof_block() {
        let out = BgzfWriter::new(Vec::new()).finish().unwrap();
        assert_eq!(out, EOF_BLOCK);
    }

    #[test]
    fn declared_block_size_matches_bytes_written() {
        // Guards the BC minus-one encoding from the writer side.
        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(b"some payload here").unwrap();
        let out = writer.finish().unwrap();

        let header = parse_header(&out, 0).unwrap();
        assert!(is_eof_block(&out[header.block_size..]));
    }

    #[test]
    fn empty_block_is_legal() {
        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(b"").unwrap();
        let out = writer.finish().unwrap();

        let spans = discover_blocks(&out, 0).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].isize, 0);
        assert_eq!(spans[0].crc32, 0);
    }

    #[test]
    fn splits_data_across_blocks() {
        let data = vec![b'A'; 1000];
        let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(100);
        writer.write_data(&data).unwrap();
        let out = writer.finish().unwrap();

        let spans = discover_blocks(&out, 0).unwrap();
        assert_eq!(spans.len(), 11, "10 payload blocks + EOF");
        assert!(spans[..10].iter().all(|s| s.isize == 100));
    }

    #[test]
    fn every_written_block_stays_within_the_64kib_cap() {
        // Incompressible input is the case that can push a block over the cap.
        let mut data = vec![0u8; MAX_PAYLOAD];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
        }

        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(&data).unwrap();
        let out = writer.finish().unwrap();

        for span in discover_blocks(&out, 0).unwrap() {
            let total = (span.payload_start as u64 - span.compressed_offset) as usize
                + span.payload_len
                + TRAILER_SIZE;
            assert!(
                total <= MAX_BLOCK_SIZE,
                "block of {total} bytes exceeds cap"
            );
        }
    }

    #[test]
    fn rejects_oversized_single_block() {
        let mut writer = BgzfWriter::new(Vec::new());
        assert!(writer.write_block(&vec![0u8; MAX_PAYLOAD + 1]).is_err());
    }
}
