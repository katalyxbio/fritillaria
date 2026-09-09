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
pub const HEADER_SIZE: usize = FIXED_HEADER_SIZE + XLEN;
/// Largest deflate payload that still leaves room for framing.
const MAX_PAYLOAD: usize = MAX_BLOCK_SIZE - HEADER_SIZE - TRAILER_SIZE;

/// Largest **deflate stream** that still fits a block once framed.
///
/// The cap BGZF imposes is on the whole block, so this is what a compressor
/// must come in under — not [`MAX_BLOCK_SIZE`], and not the payload size. It is
/// the number [`frame_block`] rejects against.
pub const MAX_DEFLATE_STREAM: usize = MAX_BLOCK_SIZE - HEADER_SIZE - TRAILER_SIZE;

/// Wraps an already-compressed deflate stream as one BGZF block.
///
/// Split out of [`BgzfWriter::write_block`] because framing is the half that
/// has nothing to do with *where* the deflate stream came from: a device codec
/// produces the stream on the GPU and still needs exactly these 18 header bytes
/// and 8 trailer bytes around it. Keeping one implementation is what stops the
/// two paths drifting on the `BC` off-by-one, which is the classic BGZF mistake.
///
/// `crc32` and `uncompressed_len` describe the **uncompressed** payload, not the
/// stream — the gzip trailer is a checksum of the original bytes, so a device
/// path must checksum before compressing, not after.
///
/// The block is appended to `out`, which is *not* cleared first, so blocks can
/// be concatenated into one buffer.
///
/// # Errors
///
/// Returns [`Error::Malformed`] if the stream is too large to frame within
/// BGZF's 64 KiB block cap. That is a real possibility rather than a formality:
/// deflate can expand incompressible input, and nvCOMP's own worst-case output
/// for a full block is 2.26x the input. The caller decides what to do about it —
/// [`BgzfWriter::write_block`] splits the payload and retries, which a device
/// batch cannot do mid-launch and must handle by re-compressing the offenders.
pub fn frame_block(
    out: &mut Vec<u8>,
    deflate_stream: &[u8],
    crc32: u32,
    uncompressed_len: u32,
) -> Result<()> {
    let block_size = HEADER_SIZE + deflate_stream.len() + TRAILER_SIZE;
    if block_size > MAX_BLOCK_SIZE {
        return Err(Error::Malformed {
            format: "bgzf",
            position: 0,
            reason: format!(
                "framed block {block_size} exceeds maximum {MAX_BLOCK_SIZE} \
                 (deflate stream {})",
                deflate_stream.len()
            ),
        });
    }

    out.extend_from_slice(&[
        0x1f, 0x8b, 0x08, 0x04, // magic, deflate, FEXTRA
        0x00, 0x00, 0x00, 0x00, // MTIME: zero, so output is reproducible
        0x00, 0xff, // XFL, OS = unknown
    ]);
    out.extend_from_slice(&(XLEN as u16).to_le_bytes());
    out.extend_from_slice(&[b'B', b'C', 0x02, 0x00]);
    // BC holds size MINUS ONE.
    out.extend_from_slice(&((block_size - 1) as u16).to_le_bytes());
    out.extend_from_slice(deflate_stream);
    out.extend_from_slice(&crc32.to_le_bytes());
    out.extend_from_slice(&uncompressed_len.to_le_bytes());

    Ok(())
}

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

        // Deflate can expand pathologically incompressible input past the cap.
        // Rather than emit an invalid block, split and retry. An empty payload
        // still compresses to a few bytes, so this always terminates.
        if HEADER_SIZE + compressed.len() + TRAILER_SIZE > MAX_BLOCK_SIZE {
            let mid = payload.len() / 2;
            self.write_block(&payload[..mid])?;
            return self.write_block(&payload[mid..]);
        }

        self.buf.clear();
        frame_block(
            &mut self.buf,
            &compressed,
            crc32fast::hash(payload),
            payload.len() as u32,
        )?;

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

    /// The extracted seam frames a stream someone else compressed.
    ///
    /// This is the path a device codec takes — it hands over a deflate stream
    /// and the CRC of the bytes *before* compression — so it is checked by
    /// round-tripping through the ordinary reader rather than by inspecting
    /// bytes, which would only prove the framing agrees with itself.
    #[test]
    fn a_separately_compressed_stream_frames_into_a_readable_block() {
        let payload = b"framed by one path, compressed by another".repeat(20);
        let stream = miniz_oxide::deflate::compress_to_vec(&payload, 6);

        let mut out = Vec::new();
        frame_block(
            &mut out,
            &stream,
            crc32fast::hash(&payload),
            payload.len() as u32,
        )
        .unwrap();
        out.extend_from_slice(&EOF_BLOCK);

        let blocks = discover_blocks(&out, 0).unwrap();
        assert_eq!(blocks.len(), 2, "one data block and the EOF block");
        assert_eq!(blocks[0].isize as usize, payload.len());
        assert_eq!(blocks[0].crc32, crc32fast::hash(&payload));

        // And it decompresses to the original, which is the actual claim.
        let mut reader = crate::BgzfReader::new(&out[..]);
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
        assert_eq!(got, payload);
    }

    /// Both paths must agree byte-for-byte, or the `BC` off-by-one has two
    /// chances to be wrong instead of one — which is the whole reason framing
    /// was extracted rather than duplicated.
    #[test]
    fn the_seam_produces_exactly_what_the_writer_produces() {
        let payload = b"identical bytes from either route".repeat(7);

        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(&payload).unwrap();
        let via_writer = writer.finish().unwrap();

        let mut via_seam = Vec::new();
        frame_block(
            &mut via_seam,
            &miniz_oxide::deflate::compress_to_vec(&payload, 6),
            crc32fast::hash(&payload),
            payload.len() as u32,
        )
        .unwrap();
        via_seam.extend_from_slice(&EOF_BLOCK);

        assert_eq!(via_seam, via_writer);
    }

    /// A deflate stream too large to frame is refused, not truncated.
    ///
    /// Unreachable from `write_block`, which splits and retries, but squarely
    /// reachable from a device batch: nvCOMP's worst-case output for a full
    /// block is 2.26x the input, well past the 64 KiB whole-block cap, and a
    /// kernel cannot split mid-launch. So the error is the device path's
    /// signal to re-compress the offending blocks, and silently emitting a
    /// block whose `BC` field had wrapped would be the worst outcome available.
    #[test]
    fn a_stream_too_large_to_frame_is_an_error() {
        let mut out = Vec::new();
        let err = frame_block(&mut out, &vec![0u8; MAX_DEFLATE_STREAM + 1], 0, 0)
            .expect_err("an unframeable stream must not produce a block");
        assert!(matches!(err, Error::Malformed { format: "bgzf", .. }));

        // Exactly at the limit is legal — this is a cap, not a margin.
        out.clear();
        frame_block(&mut out, &vec![0u8; MAX_DEFLATE_STREAM], 0, 0)
            .expect("a stream at exactly the limit must frame");
        assert_eq!(out.len(), MAX_BLOCK_SIZE);
    }

    /// `frame_block` appends, so a caller can build a stream in one buffer.
    #[test]
    fn framing_appends_rather_than_replacing() {
        let mut out = Vec::new();
        for payload in [&b"first"[..], b"second", b"third"] {
            frame_block(
                &mut out,
                &miniz_oxide::deflate::compress_to_vec(payload, 6),
                crc32fast::hash(payload),
                payload.len() as u32,
            )
            .unwrap();
        }
        out.extend_from_slice(&EOF_BLOCK);

        assert_eq!(discover_blocks(&out, 0).unwrap().len(), 4);

        let mut reader = crate::BgzfReader::new(&out[..]);
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
        assert_eq!(got, b"firstsecondthird");
    }
}
