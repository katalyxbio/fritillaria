//! The CPU reference codec.
//!
//! This is the correctness oracle. It is not tuned for speed — it allocates per
//! block and makes no attempt at parallelism — because its job is to be
//! obviously correct and to be the thing GPU output is diffed against. Optimise
//! it only if it becomes the bottleneck of the *fallback* path, and never at
//! the cost of legibility.

use fritillaria_core::{BlockCodec, BlockSpan, Error, InflateBatch, MAX_BLOCK_SIZE, Result};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::discover_blocks;
    use crate::write::BgzfWriter;

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
}
