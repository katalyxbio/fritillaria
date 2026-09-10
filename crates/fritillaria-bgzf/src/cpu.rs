//! The CPU reference codec, in both directions.
//!
//! This is the correctness oracle. It is not tuned for speed — it allocates per
//! block and makes no attempt at parallelism — because its job is to be
//! obviously correct and to be the thing GPU output is diffed against. Optimise
//! it only if it becomes the bottleneck of the *fallback* path, and never at
//! the cost of legibility.
//!
//! [`CpuCodec`] inflates and [`CpuCompressor`] deflates, and what "oracle"
//! means differs between them. A GPU inflater must be byte-identical to
//! `CpuCodec`; a GPU compressor cannot be, because compression level and
//! algorithm legitimately change the output. So `CpuCompressor` is the oracle
//! for the *round trip* and the floor for the *ratio*, not a byte reference —
//! see [`BlockCompressor`].

use fritillaria_core::compress::{EMPTY_DEFLATE_STREAM, Framing, choose_framing};
use fritillaria_core::{
    BlockCodec, BlockCompressor, BlockSpan, CompressedBatch, Error, InflateBatch, MAX_BLOCK_SIZE,
    MAX_COMPRESSIBLE_PAYLOAD, Result,
};

use crate::block::TRAILER_SIZE;
use crate::write::{HEADER_SIZE, STORED_BLOCK_HEADER, frame_block, store_block};

/// Single-threaded reference implementation of [`BlockCodec`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuCodec;

impl CpuCodec {
    /// Creates the codec.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Inflates one block and verifies it against its trailer.
    ///
    /// Verification is not optional here — see the [`BlockCodec`] contract.
    fn inflate_one(batch: &[u8], span: &BlockSpan) -> Result<Vec<u8>> {
        let payload = span.payload(batch).ok_or(Error::InvalidBlock {
            offset: span.compressed_offset,
            reason: "payload extends past the end of the batch",
        })?;

        let inflated = miniz_oxide::inflate::decompress_to_vec_with_limit(payload, MAX_BLOCK_SIZE)
            .map_err(|err| Error::Inflate {
                offset: span.compressed_offset,
                reason: format!("{:?}", err.status),
            })?;

        if inflated.len() != span.isize as usize {
            return Err(Error::SizeMismatch {
                offset: span.compressed_offset,
                expected: span.isize,
                actual: inflated.len(),
            });
        }

        let actual = crc32fast::hash(&inflated);
        if actual != span.crc32 {
            return Err(Error::ChecksumMismatch {
                offset: span.compressed_offset,
                expected: span.crc32,
                actual,
            });
        }

        Ok(inflated)
    }
}

impl BlockCodec for CpuCodec {
    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()> {
        out.clear();

        // The prefix sum the GPU path needs anyway; here it just avoids
        // repeated reallocation.
        let total: usize = spans.iter().map(|s| s.isize as usize).sum();
        out.reserve(total);

        for span in spans {
            let inflated = Self::inflate_one(batch, span)?;
            out.push_block(&inflated);
        }

        Ok(())
    }
}

/// Single-threaded reference implementation of [`BlockCompressor`].
///
/// Wraps `miniz_oxide`, with one addition that is not a detail: when deflate
/// would produce *more* bytes than the input, this stores the payload verbatim
/// instead. That is what makes the one-block-per-block guarantee hold for any
/// input, and it also stops the pathological case — incompressible data — from
/// costing ratio rather than merely failing to gain it.
#[derive(Clone, Copy, Debug)]
pub struct CpuCompressor {
    level: u8,
}

impl Default for CpuCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuCompressor {
    /// Creates the compressor at the default level.
    ///
    /// Level 6 to match htslib, which is the bar every ratio measurement in
    /// `docs/compression.md` is stated against.
    #[must_use]
    pub const fn new() -> Self {
        Self { level: 6 }
    }

    /// Sets the deflate level (0-10 as interpreted by `miniz_oxide`).
    #[must_use]
    pub const fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// Compresses one payload and appends the framed BGZF block to `out`.
    ///
    /// # Errors
    ///
    /// If `payload` exceeds [`MAX_COMPRESSIBLE_PAYLOAD`].
    pub fn compress_one(&self, payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
        if payload.len() > MAX_COMPRESSIBLE_PAYLOAD {
            return Err(Error::Malformed {
                format: "bgzf",
                position: 0,
                reason: format!(
                    "chunk of {} bytes exceeds the compressible maximum \
                     {MAX_COMPRESSIBLE_PAYLOAD}; splitting it would move a block \
                     boundary the caller chose",
                    payload.len()
                ),
            });
        }

        let deflated = miniz_oxide::deflate::compress_to_vec(payload, self.level);
        let crc = crc32fast::hash(payload);
        let isize = payload.len() as u32;

        // The rule lives in core because the nvCOMP path has to apply the same
        // one while laying its output out, before any of the bytes exist.
        match choose_framing(payload.len(), deflated.len()) {
            Framing::Deflated => frame_block(out, &deflated, crc, isize),
            Framing::Stored => {
                let mut stored = Vec::with_capacity(payload.len() + STORED_BLOCK_HEADER);
                store_block(payload, &mut stored);
                frame_block(out, &stored, crc, isize)
            }
            Framing::Empty => frame_block(out, &EMPTY_DEFLATE_STREAM, crc, isize),
        }
    }
}

impl BlockCompressor for CpuCompressor {
    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn compress_batch(
        &self,
        data: &[u8],
        bounds: &[usize],
        out: &mut CompressedBatch,
    ) -> Result<()> {
        let malformed = |reason: String| Error::Malformed {
            format: "bgzf",
            position: 0,
            reason,
        };

        // `bounds` names ranges and need not cover `data` — see the
        // `BlockCompressor` docs. What is checked is what could read past the
        // end.
        let Some(&last) = bounds.last() else {
            return Err(malformed("bounds must have at least one entry".to_string()));
        };
        if last > data.len() {
            return Err(malformed(format!(
                "bounds reach byte {last} of {} bytes",
                data.len()
            )));
        }
        if let Some([prev, next, ..]) = bounds.windows(2).find(|w| w[0] > w[1]) {
            return Err(malformed(format!(
                "bounds must be non-decreasing, found {prev} then {next}"
            )));
        }

        out.clear();
        // A guess, not a bound: real WGS BAM compresses 3.37x and BCF far more,
        // so this over-reserves on genotype data and under-reserves on nothing
        // that matters.
        out.reserve(data.len() / 3 + bounds.len() * (HEADER_SIZE + TRAILER_SIZE));

        let (buf, offsets) = out.parts_mut();
        for chunk in bounds.windows(2) {
            self.compress_one(&data[chunk[0]..chunk[1]], buf)?;
            offsets.push(buf.len());
        }

        debug_assert!(out.is_consistent());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::EOF_BLOCK;
    use crate::discover::discover_blocks;
    use crate::write::{BgzfWriter, DEFAULT_PAYLOAD_SIZE};

    fn round_trip(payloads: &[&[u8]]) -> InflateBatch {
        let mut writer = BgzfWriter::new(Vec::new());
        for payload in payloads {
            writer.write_block(payload).unwrap();
        }
        let data = writer.finish().unwrap();

        let spans = discover_blocks(&data, 0).unwrap();
        let mut out = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&data, &spans, &mut out)
            .unwrap();
        out
    }

    #[test]
    fn round_trips_payloads_in_order() {
        let out = round_trip(&[b"alpha", b"beta", b"gamma"]);
        assert_eq!(out.len(), 4, "3 payload blocks + EOF");
        assert_eq!(out.data(), b"alphabetagamma");
        assert_eq!(out.block(0), Some(&b"alpha"[..]));
        assert_eq!(out.block(3), Some(&b""[..]));
    }

    #[test]
    fn concatenation_is_seamless_across_blocks() {
        // A record spanning a block boundary is the failure mode this guards:
        // the concatenated buffer must contain the original bytes contiguously,
        // with no per-block padding or gap.
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(997);
        writer.write_data(&data).unwrap();
        let encoded = writer.finish().unwrap();

        let spans = discover_blocks(&encoded, 0).unwrap();
        let mut out = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&encoded, &spans, &mut out)
            .unwrap();

        assert_eq!(out.data(), data.as_slice());
        assert!(spans.len() > 10, "test needs many blocks to be meaningful");
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let mut out = InflateBatch::new();
        CpuCodec::new().inflate_batch(&[], &[], &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn detects_corrupted_payload() {
        let mut writer = BgzfWriter::new(Vec::new());
        writer
            .write_block(b"the quick brown fox jumps over the lazy dog")
            .unwrap();
        let mut data = writer.finish().unwrap();

        let spans = discover_blocks(&data, 0).unwrap();
        // Flip a bit inside the deflate payload, leaving the trailer intact.
        data[spans[0].payload_start + 2] ^= 0x01;

        let mut out = InflateBatch::new();
        let err = CpuCodec::new()
            .inflate_batch(&data, &spans, &mut out)
            .unwrap_err();

        assert!(
            matches!(
                err,
                Error::ChecksumMismatch { .. } | Error::SizeMismatch { .. } | Error::Inflate { .. }
            ),
            "corruption must never pass silently, got {err:?}"
        );
    }

    #[test]
    fn detects_tampered_crc() {
        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(b"payload").unwrap();
        let data = writer.finish().unwrap();

        let mut spans = discover_blocks(&data, 0).unwrap();
        spans[0].crc32 ^= 0xffff_ffff;

        let mut out = InflateBatch::new();
        assert!(matches!(
            CpuCodec::new()
                .inflate_batch(&data, &spans, &mut out)
                .unwrap_err(),
            Error::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn detects_isize_disagreement() {
        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_block(b"payload").unwrap();
        let data = writer.finish().unwrap();

        let mut spans = discover_blocks(&data, 0).unwrap();
        spans[0].isize += 1;

        let mut out = InflateBatch::new();
        assert!(matches!(
            CpuCodec::new()
                .inflate_batch(&data, &spans, &mut out)
                .unwrap_err(),
            Error::SizeMismatch {
                expected: 8,
                actual: 7,
                ..
            }
        ));
    }

    #[test]
    fn reuses_the_output_buffer_across_batches() {
        let codec = CpuCodec::new();
        let mut out = InflateBatch::new();

        for payload in [&b"first"[..], &b"second"[..]] {
            let mut writer = BgzfWriter::new(Vec::new());
            writer.write_block(payload).unwrap();
            let data = writer.finish().unwrap();
            let spans = discover_blocks(&data, 0).unwrap();

            codec.inflate_batch(&data, &spans, &mut out).unwrap();
            assert_eq!(out.block(0), Some(payload), "stale data from prior batch");
        }
    }

    // --- compression ---------------------------------------------------------

    /// Bytes that genuinely do not compress.
    ///
    /// A patterned sequence is not good enough: the obvious multiply-shift
    /// filler compresses ~28x at level 6, so a test built on it would prove
    /// nothing about the case that matters.
    fn incompressible(len: usize) -> Vec<u8> {
        let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
        (0..len)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng as u8
            })
            .collect()
    }

    /// Compresses, then inflates with the codec, and returns what came back.
    ///
    /// The round trip *is* the oracle here — there is no byte reference to diff
    /// against, so anything weaker would only prove the compressor agrees with
    /// itself.
    fn round_trip_compress(data: &[u8], bounds: &[usize]) -> (CompressedBatch, InflateBatch) {
        let mut compressed = CompressedBatch::new();
        CpuCompressor::new()
            .compress_batch(data, bounds, &mut compressed)
            .unwrap();

        let mut stream = compressed.data().to_vec();
        stream.extend_from_slice(&EOF_BLOCK);

        let spans = discover_blocks(&stream, 0).unwrap();
        let mut inflated = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&stream, &spans, &mut inflated)
            .unwrap();

        (compressed, inflated)
    }

    #[test]
    fn compressed_blocks_inflate_back_to_the_input() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let bounds = vec![0, 1000, 1000, 4321, 10_000];

        let (compressed, inflated) = round_trip_compress(&data, &bounds);

        assert_eq!(compressed.len(), 4);
        assert_eq!(inflated.data(), data.as_slice());
        assert_eq!(
            inflated.offsets(),
            [&bounds[..], &[10_000]].concat(),
            "block boundaries must survive the round trip, plus the appended EOF"
        );
    }

    /// The contract that protects the *reader's* parallelism, not the writer's
    /// tidiness: a block start is a record start only if nobody moved it.
    #[test]
    fn incompressible_input_still_yields_one_block_per_chunk() {
        let data = incompressible(MAX_COMPRESSIBLE_PAYLOAD * 3);
        let bounds = vec![
            0,
            MAX_COMPRESSIBLE_PAYLOAD,
            MAX_COMPRESSIBLE_PAYLOAD * 2,
            MAX_COMPRESSIBLE_PAYLOAD * 3,
        ];

        let (compressed, inflated) = round_trip_compress(&data, &bounds);

        assert_eq!(compressed.len(), 3, "a chunk was split to make it fit");
        assert_eq!(inflated.data(), data.as_slice());
        for i in 0..3 {
            assert_eq!(
                compressed.block(i).unwrap().len(),
                MAX_BLOCK_SIZE,
                "a maximal incompressible chunk frames to exactly the cap"
            );
        }
    }

    /// miniz expands random input by 15 bytes; a stored block costs 5. Taking
    /// the smaller is free ratio, and on the maximal chunk it is the difference
    /// between framing and not.
    #[test]
    fn storing_beats_deflating_when_deflating_would_grow_the_payload() {
        let payload = incompressible(1024);

        let mut stored = Vec::new();
        CpuCompressor::new()
            .compress_one(&payload, &mut stored)
            .unwrap();

        let deflated_len = miniz_oxide::deflate::compress_to_vec(&payload, 6).len();
        assert!(
            deflated_len > payload.len(),
            "fixture is compressible, so this test proves nothing"
        );
        assert_eq!(
            stored.len(),
            HEADER_SIZE + payload.len() + STORED_BLOCK_HEADER + TRAILER_SIZE
        );
    }

    #[test]
    fn a_chunk_too_large_to_guarantee_is_refused_rather_than_split() {
        let data = incompressible(MAX_COMPRESSIBLE_PAYLOAD + 1);
        let mut out = CompressedBatch::new();
        let err = CpuCompressor::new()
            .compress_batch(&data, &[0, data.len()], &mut out)
            .unwrap_err();
        assert!(matches!(err, Error::Malformed { format: "bgzf", .. }));

        // One byte less is fine, which is what makes the limit a limit and not
        // a margin.
        let data = incompressible(MAX_COMPRESSIBLE_PAYLOAD);
        CpuCompressor::new()
            .compress_batch(&data, &[0, data.len()], &mut out)
            .expect("a chunk at exactly the limit must compress");
    }

    #[test]
    fn rejects_bounds_that_would_read_past_the_data() {
        let data = b"abcdef";
        let cases: &[(&str, &[usize])] = &[
            ("reaches past the end", &[0, 7]),
            ("starts past the end", &[9, 9]),
            ("not monotonic", &[0, 6, 2, 6]),
            ("no entries at all", &[]),
        ];

        for (why, bounds) in cases {
            let mut out = CompressedBatch::new();
            assert!(
                CpuCompressor::new()
                    .compress_batch(data, bounds, &mut out)
                    .is_err(),
                "{why}: must not be allowed to read outside the buffer"
            );
        }

        // A sub-range is *legal*, and deliberately so: it is what lets a writer
        // compress a large file a window at a time out of one buffer that
        // cannot be sub-sliced. The cost — that "I passed the wrong bounds and
        // silently compressed part of my data" is no longer caught here — is
        // recorded in the `BlockCompressor` docs.
        let mut out = CompressedBatch::new();
        CpuCompressor::new()
            .compress_batch(data, &[2, 4], &mut out)
            .expect("a window of the buffer must compress");
        assert_eq!(out.len(), 1);

        let mut stream = out.data().to_vec();
        stream.extend_from_slice(&EOF_BLOCK);
        let spans = discover_blocks(&stream, 0).unwrap();
        let mut inflated = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&stream, &spans, &mut inflated)
            .unwrap();
        assert_eq!(inflated.block(0), Some(&b"cd"[..]));
    }

    #[test]
    fn an_empty_chunk_is_a_legal_block() {
        // Not a curiosity: the EOF block is exactly an empty block, and every
        // file the writer produces ends with one.
        let (compressed, inflated) = round_trip_compress(b"", &[0, 0]);
        assert_eq!(compressed.len(), 1);
        assert_eq!(inflated.len(), 2, "the empty block plus the appended EOF");
        assert!(inflated.data().is_empty());
    }

    /// Re-framing an empty block reproduces the canonical EOF marker byte for
    /// byte, which is worth knowing: it means a caller can terminate a file
    /// through this path instead of special-casing 28 literal bytes.
    #[test]
    fn an_empty_block_is_the_eof_marker() {
        let mut out = Vec::new();
        CpuCompressor::new().compress_one(b"", &mut out).unwrap();
        assert_eq!(out, EOF_BLOCK);
    }

    #[test]
    fn compressing_reuses_the_output_batch() {
        let compressor = CpuCompressor::new();
        let mut out = CompressedBatch::new();

        for payload in [&b"first"[..], &b"second"[..]] {
            compressor
                .compress_batch(payload, &[0, payload.len()], &mut out)
                .unwrap();
            assert_eq!(out.len(), 1, "stale blocks from the prior batch");
            assert!(out.is_consistent());
        }
    }

    /// Ratio is not a free variable: level has to reach the compressor.
    #[test]
    fn the_level_changes_the_output_size() {
        let data: Vec<u8> = (0..40_000u32).map(|i| (i % 97) as u8).collect();
        let sizes: Vec<usize> = [1u8, 9]
            .iter()
            .map(|&level| {
                let mut out = CompressedBatch::new();
                CpuCompressor::new()
                    .with_level(level)
                    .compress_batch(&data, &[0, data.len()], &mut out)
                    .unwrap();
                out.byte_len()
            })
            .collect();

        assert!(
            sizes[1] < sizes[0],
            "level 9 produced {} bytes against level 1's {}",
            sizes[1],
            sizes[0]
        );
    }

    /// The writer and the compressor must not drift: they are the same
    /// operation, and the `BC` off-by-one is the classic place to disagree.
    #[test]
    fn the_writer_and_the_compressor_produce_the_same_bytes() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 253) as u8).collect();

        let mut writer = BgzfWriter::new(Vec::new());
        writer.write_data(&data).unwrap();
        let via_writer = writer.finish().unwrap();

        let bounds: Vec<usize> = (0..=data.len())
            .step_by(DEFAULT_PAYLOAD_SIZE)
            .chain([data.len()])
            .collect();
        let mut out = CompressedBatch::new();
        CpuCompressor::new()
            .compress_batch(&data, &bounds, &mut out)
            .unwrap();
        let mut via_compressor = out.data().to_vec();
        via_compressor.extend_from_slice(&EOF_BLOCK);

        assert_eq!(via_compressor, via_writer);
    }
}
