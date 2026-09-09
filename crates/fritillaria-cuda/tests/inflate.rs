//! Differential tests for the inflate kernel.
//!
//! Every assertion compares GPU output against the CPU reference codec on the
//! same input. That is the strategy CLAUDE.md prescribes: the CPU path is the
//! oracle, and the GPU path is correct exactly when it is byte-identical to it.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure, so
//! skipping cannot quietly hide a broken kernel in CI.

#![cfg(feature = "cuda")]

use fritillaria_bgzf::{BgzfWriter, CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, Error, InflateBatch};
use fritillaria_cuda::CudaCodec;

/// Builds a CUDA codec, or explains why the test is being skipped.
fn codec() -> Option<CudaCodec> {
    match CudaCodec::new() {
        Ok(codec) => Some(codec),
        Err(err) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

/// Encodes payloads as one BGZF file, one block each.
fn bgzf(payloads: &[&[u8]], level: u8) -> Vec<u8> {
    let mut writer = BgzfWriter::new(Vec::new()).with_level(level);
    for payload in payloads {
        writer.write_block(payload).unwrap();
    }
    writer.finish().unwrap()
}

/// Asserts the GPU reproduces the CPU reference exactly for `data`.
fn assert_matches_cpu(codec: &CudaCodec, data: &[u8], what: &str) {
    let spans = discover_blocks(data, 0).unwrap();

    let mut expected = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(data, &spans, &mut expected)
        .unwrap_or_else(|e| panic!("CPU reference failed on {what}: {e:?}"));

    let mut actual = InflateBatch::new();
    codec
        .inflate_batch(data, &spans, &mut actual)
        .unwrap_or_else(|e| panic!("GPU inflate failed on {what}: {e:?}"));

    assert_eq!(actual.len(), expected.len(), "block count differs ({what})");
    assert_eq!(
        actual.offsets(),
        expected.offsets(),
        "block boundaries differ ({what})"
    );
    assert_eq!(
        actual.data(),
        expected.data(),
        "decompressed bytes differ ({what})"
    );
}

#[test]
fn matches_cpu_on_dynamic_huffman() {
    let Some(codec) = codec() else { return };
    // Ordinary compressible text: what a real BAM's blocks look like.
    let text: Vec<u8> = std::iter::repeat_n(
        b"the quick brown fox jumps over the lazy dog. ".as_slice(),
        400,
    )
    .flatten()
    .copied()
    .collect();
    assert_matches_cpu(&codec, &bgzf(&[&text], 6), "dynamic huffman");
}

#[test]
fn matches_cpu_on_stored_blocks() {
    let Some(codec) = codec() else { return };
    // Level 0 emits stored (uncompressed) DEFLATE blocks — a separate code
    // path in the kernel that compressible input never exercises.
    let data: Vec<u8> = (0..30_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    assert_matches_cpu(&codec, &bgzf(&[&data], 0), "stored blocks");
}

#[test]
fn matches_cpu_on_tiny_payloads() {
    let Some(codec) = codec() else { return };
    // Very short inputs tend to produce fixed-Huffman blocks.
    for payload in [&b"a"[..], b"ab", b"abc", b"hello world"] {
        assert_matches_cpu(&codec, &bgzf(&[payload], 6), "tiny payload");
    }
}

#[test]
fn matches_cpu_on_overlapping_back_references() {
    let Some(codec) = codec() else { return };
    // A long run encodes as a back-reference with distance < length, so the
    // copy overlaps itself. Vectorising that copy would silently corrupt it.
    let runs: Vec<u8> = std::iter::repeat_n(b'A', 5000)
        .chain(
            std::iter::repeat_n(b"ABAB".as_slice(), 500)
                .flatten()
                .copied(),
        )
        .collect();
    assert_matches_cpu(&codec, &bgzf(&[&runs], 6), "overlapping back-references");
}

#[test]
fn matches_cpu_on_empty_and_eof_blocks() {
    let Some(codec) = codec() else { return };
    // An empty block is legal, and every file ends with the empty EOF block.
    assert_matches_cpu(&codec, &bgzf(&[b""], 6), "empty block");
    assert_matches_cpu(&codec, &bgzf(&[], 6), "EOF block only");
}

#[test]
fn matches_cpu_across_many_blocks() {
    let Some(codec) = codec() else { return };
    // Many blocks in one batch: the case the whole design exists for, and
    // where a per-block offset error would show up.
    let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(997);
    writer.write_data(&data).unwrap();
    let encoded = writer.finish().unwrap();

    let spans = discover_blocks(&encoded, 0).unwrap();
    assert!(spans.len() > 100, "test needs many blocks to be meaningful");

    assert_matches_cpu(&codec, &encoded, "many blocks");

    // And the concatenated result must equal the original bytes, which is what
    // record parsing across block boundaries depends on.
    let mut out = InflateBatch::new();
    codec.inflate_batch(&encoded, &spans, &mut out).unwrap();
    assert_eq!(out.data(), data.as_slice());
}

#[test]
fn matches_cpu_on_mixed_block_types_in_one_batch() {
    let Some(codec) = codec() else { return };
    // Different blocks in the same launch take different kernel paths, so
    // threads within a warp diverge. Correctness must not depend on that.
    let incompressible: Vec<u8> = (0..4000u32)
        .map(|i| (i.wrapping_mul(2_246_822_519) >> 11) as u8)
        .collect();
    let compressible = vec![b'Z'; 4000];

    let mut writer = BgzfWriter::new(Vec::new());
    writer.write_block(&compressible).unwrap();
    writer.write_block(b"").unwrap();
    writer.write_block(&incompressible).unwrap();
    writer.write_block(b"short").unwrap();
    let encoded = writer.finish().unwrap();

    assert_matches_cpu(&codec, &encoded, "mixed block types");
}

#[test]
fn rejects_a_tampered_checksum() {
    let Some(codec) = codec() else { return };
    // The BlockCodec contract requires verification; a backend that returned
    // unverified bytes would be worse than a slow one.
    let encoded = bgzf(&[b"payload that will be checksummed"], 6);
    let mut spans = discover_blocks(&encoded, 0).unwrap();
    spans[0].crc32 ^= 0xffff_ffff;

    let mut out = InflateBatch::new();
    let err = codec.inflate_batch(&encoded, &spans, &mut out).unwrap_err();
    assert!(
        matches!(err, Error::ChecksumMismatch { .. }),
        "expected a checksum mismatch, got {err:?}"
    );
}

#[test]
fn rejects_a_tampered_isize() {
    let Some(codec) = codec() else { return };
    let encoded = bgzf(&[b"payload"], 6);
    let mut spans = discover_blocks(&encoded, 0).unwrap();
    spans[0].isize += 1;

    let mut out = InflateBatch::new();
    let err = codec.inflate_batch(&encoded, &spans, &mut out).unwrap_err();
    assert!(
        matches!(err, Error::SizeMismatch { .. } | Error::Inflate { .. }),
        "expected a size mismatch, got {err:?}"
    );
}

#[test]
fn rejects_a_corrupted_payload() {
    let Some(codec) = codec() else { return };
    // Corrupt the deflate stream itself and leave the trailer intact: the
    // kernel must either fail to decode or produce bytes that fail the CRC.
    let mut encoded = bgzf(&[b"the quick brown fox jumps over the lazy dog"], 6);
    let spans = discover_blocks(&encoded, 0).unwrap();
    encoded[spans[0].payload_start + 2] ^= 0x5a;

    let mut out = InflateBatch::new();
    let err = codec.inflate_batch(&encoded, &spans, &mut out).unwrap_err();
    assert!(
        matches!(
            err,
            Error::ChecksumMismatch { .. } | Error::SizeMismatch { .. } | Error::Inflate { .. }
        ),
        "corruption must never pass silently, got {err:?}"
    );
}

#[test]
fn empty_batch_is_not_an_error() {
    let Some(codec) = codec() else { return };
    let mut out = InflateBatch::new();
    codec.inflate_batch(&[], &[], &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn output_buffer_is_reusable_across_batches() {
    let Some(codec) = codec() else { return };
    let mut out = InflateBatch::new();
    for payload in [&b"first batch"[..], b"second", b""] {
        let encoded = bgzf(&[payload], 6);
        let spans = discover_blocks(&encoded, 0).unwrap();
        codec.inflate_batch(&encoded, &spans, &mut out).unwrap();
        assert_eq!(out.block(0), Some(payload), "stale data from a prior batch");
    }
}
