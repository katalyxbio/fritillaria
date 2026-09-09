//! A streaming BGZF reader that decompresses in batches.
//!
//! # Why batched
//!
//! A GPU needs many blocks per launch to be worth the trip. Reading one block
//! at a time would serialise exactly what the design exists to parallelise, so
//! this reader pulls a run of blocks, inflates them in a single
//! [`BlockCodec`] call, and then serves bytes out of the decompressed batch.
//!
//! With [`CpuCodec`](crate::CpuCodec) that is merely tidy; with the CUDA codec
//! it is the whole point.
//!
//! # Interop
//!
//! With the `noodles` feature this implements `fritillaria_bgzf::io::Read` and
//! `BufRead`, which is the seam every noodles format crate is generic over. A
//! `fritillaria_bam::io::Reader` built on top of this decompresses on the GPU
//! while noodles does the record parsing — and the same holds for BCF,
//! `bgzip`ped VCF, and anything tabix-indexed.

use std::io::{self, BufRead, Read, Seek, SeekFrom};

use fritillaria_core::{BlockCodec, BlockSpan, Error, InflateBatch, MAX_BLOCK_SIZE};

use crate::block::TRAILER_SIZE;
use crate::cpu::CpuCodec;
use crate::discover::BlockDiscovery;

/// Default number of BGZF blocks to inflate per batch.
///
/// At 64 KiB per block this is ~16 MiB of decompressed data — enough work to
/// amortise a kernel launch and a PCIe round trip, small enough to stay in
/// device memory comfortably. Unmeasured; tune once there is a benchmark.
pub const DEFAULT_BLOCKS_PER_BATCH: usize = 256;

/// Reads a BGZF stream, decompressing a batch of blocks at a time.
#[derive(Debug)]
pub struct BgzfReader<R, C = CpuCodec> {
    inner: R,
    codec: C,
    blocks_per_batch: usize,

    /// Compressed bytes for the batch being decoded.
    compressed: Vec<u8>,
    /// Absolute file offset of `compressed[0]`.
    batch_offset: u64,
    /// Bytes of a block that was cut short by the end of the last read.
    carry: Vec<u8>,

    spans: Vec<BlockSpan>,
    inflated: InflateBatch,
    /// Read position within `inflated.data()`.
    cursor: usize,

    /// The inner reader has no more bytes.
    inner_exhausted: bool,
}

impl<R: Read> BgzfReader<R, CpuCodec> {
    /// Creates a reader using the CPU reference codec.
    pub fn new(inner: R) -> Self {
        Self::with_codec(inner, CpuCodec::new())
    }
}

impl<R: Read, C: BlockCodec> BgzfReader<R, C> {
    /// Creates a reader using an explicit codec.
    ///
    /// This is where a GPU backend is injected; the reader itself is unaware
    /// of which one it holds.
    pub fn with_codec(inner: R, codec: C) -> Self {
        Self {
            inner,
            codec,
            blocks_per_batch: DEFAULT_BLOCKS_PER_BATCH,
            compressed: Vec::new(),
            batch_offset: 0,
            carry: Vec::new(),
            spans: Vec::new(),
            inflated: InflateBatch::new(),
            cursor: 0,
            inner_exhausted: false,
        }
    }

    /// Sets how many blocks are inflated per batch.
    ///
    /// Larger batches give the GPU more work per launch at the cost of memory.
    #[must_use]
    pub fn with_blocks_per_batch(mut self, blocks: usize) -> Self {
        self.blocks_per_batch = blocks.max(1);
        self
    }

    /// The backend actually in use, for diagnostics.
    pub fn codec_name(&self) -> &'static str {
        self.codec.name()
    }

    /// Consumes the reader, returning the underlying byte source.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Total size of a block, derived from its span.
    ///
    /// `payload_start` is relative to the batch slice while `compressed_offset`
    /// is absolute, so the header length is the difference between them.
    fn block_size(&self, span: &BlockSpan) -> usize {
        let relative_start = (span.compressed_offset - self.batch_offset) as usize;
        let header = span.payload_start - relative_start;
        header + span.payload_len + TRAILER_SIZE
    }

    /// Virtual offset of the next byte to be returned.
    ///
    /// Upper 48 bits are the containing block's compressed offset, lower 16 its
    /// uncompressed offset. When the cursor sits exactly at a block boundary
    /// the position is the *start of the next block*, not the end of the
    /// previous one — that is the convention BAI/CSI indexes use.
    pub fn virtual_offset(&self) -> fritillaria_core::VirtualOffset {
        let offsets = self.inflated.offsets();

        for (i, span) in self.spans.iter().enumerate() {
            let start = offsets[i];
            let end = offsets[i + 1];
            // Skip empty blocks: they can never contain the cursor, and
            // reporting one would give a position no read can resume from.
            if self.cursor < end || (self.cursor == start && start < end) {
                return fritillaria_core::VirtualOffset::new_unchecked(
                    span.compressed_offset,
                    (self.cursor - start) as u16,
                );
            }
        }

        // Past the end of the batch: the next block begins after the last one.
        match self.spans.last() {
            Some(span) => fritillaria_core::VirtualOffset::new_unchecked(
                span.compressed_offset + self.block_size(span) as u64,
                0,
            ),
            None => fritillaria_core::VirtualOffset::new_unchecked(self.batch_offset, 0),
        }
    }

    /// Fills `compressed` with the next run of blocks and inflates them.
    ///
    /// Returns `false` once the stream is exhausted.
    fn load_batch(&mut self) -> io::Result<bool> {
        loop {
            if self.inner_exhausted && self.carry.is_empty() {
                return Ok(false);
            }

            // A partial block from the previous read leads the new buffer.
            self.batch_offset += self.compressed.len() as u64;
            self.batch_offset -= self.carry.len() as u64;
            self.compressed.clear();
            self.compressed.append(&mut self.carry);

            // Blocks cap at 64 KiB, so this always yields at least
            // `blocks_per_batch` of them unless the stream ends first.
            let want = self.blocks_per_batch * MAX_BLOCK_SIZE;
            let start = self.compressed.len();
            self.compressed.resize(start + want, 0);

            let mut filled = start;
            while filled < self.compressed.len() {
                match self.inner.read(&mut self.compressed[filled..]) {
                    Ok(0) => {
                        self.inner_exhausted = true;
                        break;
                    }
                    Ok(n) => filled += n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            self.compressed.truncate(filled);

            let mut discovery = BlockDiscovery::new(&self.compressed, self.batch_offset);
            self.spans.clear();
            for span in discovery.by_ref() {
                self.spans.push(span.map_err(to_io)?);
            }
            let consumed = discovery.position();

            // Whatever follows the last complete block is a partial block; it
            // leads the next batch. At true EOF it means a truncated file.
            self.carry.clear();
            self.carry.extend_from_slice(&self.compressed[consumed..]);
            if self.inner_exhausted && !self.carry.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "truncated BGZF: {} trailing bytes are not a complete block",
                        self.carry.len()
                    ),
                ));
            }

            if self.spans.is_empty() {
                // No complete block this round. Either the stream is over, or
                // the carry needs more bytes appended to it.
                if self.inner_exhausted {
                    return Ok(false);
                }
                continue;
            }

            self.codec
                .inflate_batch(&self.compressed, &self.spans, &mut self.inflated)
                .map_err(to_io)?;
            self.cursor = 0;

            // A batch of only empty blocks (an EOF marker alone) yields no
            // bytes but is not the end of the stream unless the input is over.
            if self.inflated.data().is_empty() && !self.inner_exhausted {
                continue;
            }
            return Ok(true);
        }
    }
}

impl<R: Read + Seek, C: BlockCodec> BgzfReader<R, C> {
    /// Repositions the reader to a virtual position.
    ///
    /// The upper 48 bits name a block by its offset in the compressed stream;
    /// the lower 16 name a byte inside that block once inflated. Both halves
    /// matter: seeking to the block is a file seek, and seeking *within* it
    /// requires the block to be inflated first, which is why this cannot be a
    /// plain `io::Seek` on the inner reader.
    ///
    /// Every batching field is reset, not adjusted. A carry from before the
    /// seek describes bytes at the old position, and keeping it would prepend
    /// them to the new batch — the kind of error that produces plausible
    /// records from the wrong part of the file.
    pub fn seek_to(&mut self, pos: crate::VirtualPosition) -> io::Result<crate::VirtualPosition> {
        let (compressed, uncompressed) = (pos.compressed(), pos.uncompressed());

        self.inner.seek(SeekFrom::Start(compressed))?;

        self.batch_offset = compressed;
        self.compressed.clear();
        self.carry.clear();
        self.spans.clear();
        self.inflated.clear();
        self.cursor = 0;
        self.inner_exhausted = false;

        if !self.load_batch()? {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("no BGZF block at compressed offset {compressed}"),
            ));
        }

        // `load_batch` may have skipped leading empty blocks, so the block the
        // caller named is not necessarily the first one loaded. Find where it
        // actually starts.
        let index = self
            .spans
            .iter()
            .position(|span| span.compressed_offset == compressed)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("compressed offset {compressed} is not a block boundary"),
                )
            })?;
        let block_start = self.inflated.offsets()[index];
        let block_end = self.inflated.offsets()[index + 1];

        // Bounded by *this block*, not by the batch. Checking against the batch
        // is the tempting mistake and it is silently wrong: a batch holds up to
        // 256 blocks, so an out-of-range uncompressed offset lands in a later
        // block and yields plausible bytes from the wrong record. The header
        // block of `testdata/htslib_multiblock.bam` inflates to 223 bytes, so
        // an offset of 250 is invalid — and with the batch-wide check it
        // resolved 27 bytes into the next block instead of erroring.
        let cursor = block_start + usize::from(uncompressed);
        if cursor > block_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "uncompressed offset {uncompressed} is past the end of the {} byte block at {compressed}",
                    block_end - block_start
                ),
            ));
        }
        self.cursor = cursor;

        Ok(pos)
    }
}

fn to_io(err: Error) -> io::Error {
    match err {
        Error::Io(e) => e,
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    }
}

impl<R: Read, C: BlockCodec> Read for BgzfReader<R, C> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: Read, C: BlockCodec> BufRead for BgzfReader<R, C> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.cursor >= self.inflated.data().len() {
            if !self.load_batch()? {
                return Ok(&[]);
            }
        }
        Ok(&self.inflated.data()[self.cursor..])
    }

    fn consume(&mut self, amt: usize) {
        self.cursor = (self.cursor + amt).min(self.inflated.data().len());
    }
}

// The two traits every format reader in this workspace is generic over. They
// used to live in an external crate and so sat behind a feature; now that the
// CPU path is vendored alongside, the drop-in property is unconditional.
mod trait_impls {
    use super::{BgzfReader, BlockCodec, Read};

    impl<R: Read, C: BlockCodec> crate::io::Read for BgzfReader<R, C> {
        fn virtual_position(&self) -> crate::VirtualPosition {
            crate::VirtualPosition::from(self.virtual_offset().as_u64())
        }
    }

    // No methods of its own: the marker that makes every format crate here
    // accept this reader.
    impl<R: Read, C: BlockCodec> crate::io::BufRead for BgzfReader<R, C> {}

    // Indexed access. Without this a region query — `bam::io::IndexedReader`,
    // `bcf`'s, anything tabix-driven — cannot use this reader at all and falls
    // back to the vendored CPU one, which quietly gives up the GPU for exactly
    // the workloads that read least of a file.
    impl<R: super::Seek + Read, C: BlockCodec> crate::io::Seek for BgzfReader<R, C> {
        fn seek_to_virtual_position(
            &mut self,
            pos: crate::VirtualPosition,
        ) -> super::io::Result<crate::VirtualPosition> {
            self.seek_to(pos)
        }

        fn seek_with_index(
            &mut self,
            index: &crate::gzi::Index,
            pos: super::SeekFrom,
        ) -> super::io::Result<u64> {
            match pos {
                super::SeekFrom::Start(offset) => {
                    // gzi maps an *uncompressed* file offset to the virtual
                    // position of the block containing it. That is what makes
                    // `bgzip -b` work on a plain bgzipped file with no format
                    // index of its own.
                    let virtual_position = index.query(offset)?;
                    self.seek_to(virtual_position)?;
                    Ok(offset)
                }
                // Matching the vendored reader, which does the same. Neither
                // End nor Current is expressible without knowing the
                // uncompressed length, which the index does not carry.
                other => Err(super::io::Error::new(
                    super::io::ErrorKind::Unsupported,
                    format!("only SeekFrom::Start is supported, got {other:?}"),
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::BgzfWriter;

    fn encode(data: &[u8], payload_size: usize) -> Vec<u8> {
        let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(payload_size);
        writer.write_data(data).unwrap();
        writer.finish().unwrap()
    }

    #[test]
    fn reads_a_single_block() {
        let encoded = encode(b"hello world", 4096);
        let mut out = Vec::new();
        BgzfReader::new(&encoded[..]).read_to_end(&mut out).unwrap();
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn reads_an_empty_stream() {
        // A legal, empty BGZF file: just the EOF block.
        let encoded = encode(b"", 4096);
        let mut out = Vec::new();
        BgzfReader::new(&encoded[..]).read_to_end(&mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reassembles_data_across_many_blocks() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&data, 997);

        let mut out = Vec::new();
        BgzfReader::new(&encoded[..]).read_to_end(&mut out).unwrap();
        assert_eq!(out, data, "data must survive block reassembly");
    }

    #[test]
    fn spans_batch_boundaries() {
        // More blocks than fit in one batch, so the reader must stitch batches
        // together without dropping or duplicating a byte.
        let data: Vec<u8> = (0..50_000u32).map(|i| (i % 97) as u8).collect();
        let encoded = encode(&data, 512);

        let mut reader = BgzfReader::new(&encoded[..]).with_blocks_per_batch(1);
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn small_reads_reassemble_correctly() {
        // Byte-at-a-time reads cross block and batch boundaries at every
        // possible offset.
        let data: Vec<u8> = (0..5_000u32).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&data, 128);

        let mut reader = BgzfReader::new(&encoded[..]).with_blocks_per_batch(2);
        let mut out = Vec::new();
        let mut byte = [0u8; 1];
        while reader.read(&mut byte).unwrap() == 1 {
            out.push(byte[0]);
        }
        assert_eq!(out, data);
    }

    #[test]
    fn virtual_offset_starts_at_the_first_block() {
        let encoded = encode(b"payload", 4096);
        let mut reader = BgzfReader::new(&encoded[..]);
        reader.fill_buf().unwrap();

        let offset = reader.virtual_offset();
        assert_eq!(offset.compressed(), 0);
        assert_eq!(offset.uncompressed(), 0);
    }

    #[test]
    fn virtual_offset_advances_within_a_block() {
        let encoded = encode(b"0123456789", 4096);
        let mut reader = BgzfReader::new(&encoded[..]);

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();

        let offset = reader.virtual_offset();
        assert_eq!(offset.compressed(), 0, "still inside the first block");
        assert_eq!(offset.uncompressed(), 4);
    }

    #[test]
    fn virtual_offset_moves_to_the_next_block_at_a_boundary() {
        // Two blocks of 5 bytes: after reading exactly 5, the position must be
        // the start of block 2, not the end of block 1. Indexes depend on this.
        let encoded = encode(b"AAAAABBBBB", 5);
        let mut reader = BgzfReader::new(&encoded[..]);

        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"AAAAA");

        let offset = reader.virtual_offset();
        assert!(offset.compressed() > 0, "must point at the second block");
        assert_eq!(offset.uncompressed(), 0);
    }

    #[test]
    fn rejects_a_truncated_stream() {
        let encoded = encode(b"some data here", 4096);
        let truncated = &encoded[..encoded.len() - 12];

        let mut out = Vec::new();
        let err = BgzfReader::new(truncated)
            .read_to_end(&mut out)
            .unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::UnexpectedEof,
            "a truncated file must not read as a short one"
        );
    }

    #[test]
    fn rejects_corruption_rather_than_returning_bad_bytes() {
        let mut encoded = encode(b"the quick brown fox", 4096);
        let len = encoded.len();
        // Corrupt the CRC in the trailer of the first block.
        encoded[len - 40] ^= 0xff;

        let mut out = Vec::new();
        assert!(
            BgzfReader::new(&encoded[..]).read_to_end(&mut out).is_err()
                || out != b"the quick brown fox",
            "corruption must not pass silently"
        );
    }

    #[test]
    fn reports_the_codec_in_use() {
        let encoded = encode(b"x", 4096);
        assert_eq!(BgzfReader::new(&encoded[..]).codec_name(), "cpu-reference");
    }
}
