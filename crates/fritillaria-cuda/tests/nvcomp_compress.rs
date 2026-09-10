//! Device BGZF compression on nvCOMP, against the host reference and htslib.
//!
//! # What is left to check here
//!
//! Most of this path is already checked without a GPU, deliberately:
//! `nvcomp::compress`'s unit tests cover the batch sizing and the dense layout,
//! and `tests/frame_kernel.rs` runs the framing kernel on the CPU and diffs it
//! against `frame_block` byte for byte. What needs hardware is the part that
//! genuinely cannot be faked — that NVRTC accepts the kernel, that the launch
//! configuration is right, and that nvCOMP produces a deflate stream at all.
//!
//! # There is no byte oracle, and that is not a gap in these tests
//!
//! Two valid DEFLATE streams of the same input differ legitimately, so unlike
//! every other differential test in this workspace these cannot pin output to
//! the CPU reference. The net is the same three checks the host compressor uses:
//! round trip, htslib acceptance, and an asserted ratio floor. The floor is the
//! one worth stating out loud — nvCOMP quietly falling back to entropy-only
//! output would pass the other two.
//!
//! # Skipping
//!
//! Same discipline as `tests/nvcomp.rs`: an absent device or library is a
//! `NOTE:`, but a `FRITILLARIA_NVCOMP_LIB` that is set and will not load with a
//! device present is a hard failure. A green run must never mean "nvCOMP was
//! never exercised".

#![cfg(feature = "nvcomp")]
// Ratios are compared in floating point; the sizes involved are megabytes,
// nowhere near a f64 mantissa.
#![allow(clippy::cast_precision_loss)]

use std::path::PathBuf;
use std::process::Command;

use fritillaria_bgzf::{
    CpuCodec, CpuCompressor, DeviceBgzfWriter, EOF_BLOCK, discover_blocks, is_eof_block,
};
use fritillaria_core::{
    BlockCodec, BlockCompressor, CompressedBatch, DeviceBlockCodec, DeviceBlockCompressor,
    DeviceInflateBatch, InflateBatch,
};
use fritillaria_cuda::nvcomp::ffi::{DeflateAlgorithm, LIB_PATH_ENV};
use fritillaria_cuda::{NvcompCodec, NvcompCompressor};

fn compressor() -> Option<NvcompCompressor> {
    match NvcompCompressor::new(0) {
        Ok(c) => Some(c),
        Err(err) => {
            assert!(
                std::env::var_os(LIB_PATH_ENV).is_none()
                    || !fritillaria_cuda::device_is_available(),
                "{LIB_PATH_ENV} is set and a device is present, but nvCOMP \
                 would not load: {err}"
            );
            eprintln!("NOTE: nvCOMP or CUDA unavailable on this machine ({err})");
            None
        }
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// A fixture's payloads, with the block boundaries htslib chose.
fn inflate(raw: &[u8]) -> InflateBatch {
    let spans = discover_blocks(raw, 0).unwrap();
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .unwrap();
    out
}

/// Inflates a compressed batch back to payloads, appending the EOF marker so the
/// stream is a valid file.
fn round_trip(compressed: &CompressedBatch) -> InflateBatch {
    let mut stream = compressed.data().to_vec();
    stream.extend_from_slice(&EOF_BLOCK);
    let spans = discover_blocks(&stream, 0).unwrap();
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&stream, &spans, &mut out)
        .unwrap();
    out
}

/// Compresses through the host entry point, which uploads and then runs the
/// device path — the same kernels, reachable without building a device batch.
fn compress_host(c: &NvcompCompressor, batch: &InflateBatch) -> CompressedBatch {
    let mut out = CompressedBatch::new();
    BlockCompressor::compress_batch(c, batch.data(), batch.offsets(), &mut out).unwrap();
    out
}

#[test]
fn compressed_blocks_inflate_back_to_the_input() {
    let Some(c) = compressor() else { return };

    for name in ["pacbio_hifi.bam", "ont_ultralong.bam", "kg_phase3.bcf"] {
        let inflated = inflate(&fixture(name));
        let compressed = compress_host(&c, &inflated);

        assert_eq!(
            compressed.len(),
            inflated.len(),
            "{name}: block count changed, so a boundary moved — and in a BAM \
             that costs the record scan its per-block parallelism"
        );
        assert!(
            compressed.is_consistent(),
            "{name}: batch does not describe itself"
        );

        let back = round_trip(&compressed);
        assert_eq!(back.data(), inflated.data(), "{name}: payload changed");
        assert_eq!(
            &back.offsets()[..inflated.offsets().len()],
            inflated.offsets(),
            "{name}: block boundaries moved"
        );
    }
}

/// The device entry point proper: input already in VRAM, nothing uploaded.
///
/// This is the path the design exists for, so it is checked against the host
/// entry point rather than only against the reference — if the upload wrapper
/// and the device call disagreed, only this would say so.
#[test]
fn compressing_a_device_batch_matches_compressing_the_same_bytes_from_the_host() {
    let Some(c) = compressor() else { return };
    let Ok(codec) = NvcompCodec::new() else {
        return;
    };

    let raw = fixture("pacbio_hifi.bam");
    let spans = discover_blocks(&raw, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut device)
        .unwrap();

    let mut from_device = CompressedBatch::new();
    c.compress_batch_device(device.data().unwrap(), device.offsets(), &mut from_device)
        .unwrap();

    let from_host = compress_host(&c, &inflate(&raw));

    // These *can* be byte-equal, because it is the same compressor on the same
    // bytes — unlike a CPU-vs-GPU comparison, where they legitimately differ.
    assert_eq!(from_device.data(), from_host.data());
    assert_eq!(from_device.offsets(), from_host.offsets());
}

/// The acceptance bar, and it is binary: does htslib read what we wrote?
#[test]
fn samtools_reads_a_gpu_compressed_bam() {
    let Some(c) = compressor() else { return };

    let raw = fixture("pacbio_hifi.bam");
    let compressed = compress_host(&c, &inflate(&raw));

    let mut stream = compressed.data().to_vec();
    assert!(
        is_eof_block(&stream[stream.len() - EOF_BLOCK.len()..]),
        "re-framing the fixture's trailing empty block must reproduce the EOF \
         marker, or every tool reports truncation"
    );

    let dir = std::env::temp_dir().join("fritillaria-nvcomp-compress");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gpu.bam");
    std::fs::write(&path, &stream).unwrap();

    let expected = samtools_count(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/pacbio_hifi.bam"),
    );
    let Some(expected) = expected else {
        eprintln!("NOTE: samtools not installed, acceptance bar unchecked");
        return;
    };

    assert_eq!(
        samtools_count(&path),
        Some(expected),
        "samtools read a different number of records from our BAM"
    );

    stream.clear();
    let _ = std::fs::remove_dir_all(&dir);
}

fn samtools_count(path: &std::path::Path) -> Option<usize> {
    let out = Command::new("samtools")
        .args(["view", "-c"])
        .arg(path)
        .output()
        .ok()?;
    assert!(
        out.status.success(),
        "samtools rejected {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let count: usize = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    assert!(
        count > 0,
        "samtools found no records, so this proves nothing"
    );
    Some(count)
}

/// A ratio floor, asserted rather than printed.
///
/// The comparison that matters and the whole reason the default is
/// [`DeflateAlgorithm::HighRatio`] rather than Parabricks' entropy-only choice:
/// nvCOMP claims level 4 "beats Zlib level 6", but libdeflate is better than
/// zlib and htslib is what our files are measured against. Level 0 would fail
/// this by a wide margin, which is the point.
#[test]
fn the_ratio_stays_within_reach_of_htslib() {
    /// Largest tolerated excess over htslib's own output.
    ///
    /// Measured on an L4 2026-09-10: +5.4% on HiFi, +0.8% on ONT, +2.5% on the
    /// 2504-sample BCF — so 10% is the same floor the host reference uses, with
    /// real headroom. It was provisionally 20% before that run, because the run
    /// *was* the measurement.
    ///
    /// The margin is what makes this a check rather than a formality:
    /// entropy-only output measured **+75.1%** on the same fixture, so a
    /// silently downgraded level misses this by 65 points.
    const TOLERANCE: f64 = 0.10;

    let Some(c) = compressor() else { return };

    for name in ["pacbio_hifi.bam", "ont_ultralong.bam", "kg_phase3.bcf"] {
        let raw = fixture(name);
        let inflated = inflate(&raw);
        let ours = compress_host(&c, &inflated).byte_len();

        let excess = ours as f64 / raw.len() as f64 - 1.0;
        println!(
            "{name:24} htslib {:>9} nvcomp {ours:>9} ({:+.1}%)",
            raw.len(),
            excess * 100.0
        );
        assert!(
            excess <= TOLERANCE,
            "{name}: {:.1}% larger than htslib, past the {:.0}% floor",
            excess * 100.0,
            TOLERANCE * 100.0
        );
    }
}

/// The ladder is real in both directions, and the measurement the default rests
/// on: higher levels must actually produce smaller files.
///
/// If this ever fails, the default is costing 17x the scratch for nothing.
#[test]
fn a_higher_algorithm_produces_a_smaller_file() {
    let Some(_) = compressor() else { return };

    let inflated = inflate(&fixture("pacbio_hifi.bam"));
    let mut sizes = Vec::new();

    for algorithm in [DeflateAlgorithm::EntropyOnly, DeflateAlgorithm::HighRatio] {
        let c = NvcompCompressor::with_algorithm(0, algorithm).unwrap();
        let size = compress_host(&c, &inflated).byte_len();
        println!(
            "{algorithm:?}: {size} bytes, scratch {} per chunk",
            c.budget().scratch_per_chunk
        );
        sizes.push(size);
    }

    assert!(
        sizes[1] < sizes[0],
        "HighRatio produced {} bytes against EntropyOnly's {} — the ratio \
         default is paying 17x the scratch for nothing",
        sizes[1],
        sizes[0]
    );
}

/// Incompressible input is where the stored fallback and the block cap meet.
///
/// nvCOMP's worst case is 2.26x a full chunk, so this is the case that would
/// otherwise emit a block whose `BC` field had wrapped.
#[test]
fn incompressible_input_still_yields_one_block_per_chunk() {
    let Some(c) = compressor() else { return };

    let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
    let data: Vec<u8> = (0..fritillaria_core::MAX_COMPRESSIBLE_PAYLOAD * 3)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u8
        })
        .collect();
    let bounds: Vec<usize> = (0..=3)
        .map(|i| i * fritillaria_core::MAX_COMPRESSIBLE_PAYLOAD)
        .collect();

    let mut out = CompressedBatch::new();
    BlockCompressor::compress_batch(&c, &data, &bounds, &mut out).unwrap();

    assert_eq!(out.len(), 3, "a chunk was split to make it fit");
    for i in 0..3 {
        assert!(
            out.block(i).unwrap().len() <= fritillaria_core::MAX_BLOCK_SIZE,
            "block {i} exceeds the 64 KiB cap"
        );
    }
    assert_eq!(round_trip(&out).data(), data.as_slice());
}

/// The whole loop, on real hardware: read a BAM to device columns, write it
/// straight back out, and hand the result to htslib.
///
/// Every other test here compresses one batch. This one drives
/// `DeviceBgzfWriter` over a file at **one chunk per batch**, so the fixture's
/// 13 blocks span 13 compression batches — the hardest split available, and the
/// seam a realistic batch size would never reach. Nothing touches host memory
/// between the inflate and the compress.
#[test]
fn a_file_round_trips_through_the_device_writer() {
    let Some(c) = compressor() else { return };
    let Ok(codec) = NvcompCodec::new() else {
        return;
    };

    let raw = fixture("pacbio_hifi.bam");
    let spans = discover_blocks(&raw, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut device)
        .unwrap();

    let mut w = DeviceBgzfWriter::new(Vec::new(), &c).with_chunks_per_batch(1);
    w.write_batch(device.data().unwrap(), device.offsets())
        .unwrap();
    assert_eq!(
        w.batches_run(),
        device.len() as u64,
        "one chunk per batch should mean one batch per block; without this the \
         split being tested may not have happened at all"
    );
    let stream = w.finish().unwrap();

    let inflated = inflate(&raw);
    let back = {
        let spans = discover_blocks(&stream, 0).unwrap();
        let mut out = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&stream, &spans, &mut out)
            .unwrap();
        out
    };
    assert_eq!(back.data(), inflated.data());
    assert_eq!(
        &back.offsets()[..inflated.offsets().len()],
        inflated.offsets(),
        "block boundaries moved across a batch split"
    );

    let dir = std::env::temp_dir().join("fritillaria-device-writer");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.bam");
    std::fs::write(&path, &stream).unwrap();
    if let Some(count) = samtools_count(&path) {
        assert_eq!(count, 20, "the fixture holds 20 HiFi reads");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The host and device compressors must agree on *content*, never on bytes.
///
/// Stating it as a test rather than a comment: they compress differently on
/// purpose, and a future change that accidentally made them byte-equal would
/// mean the device path had stopped using nvCOMP.
#[test]
fn the_gpu_and_cpu_compressors_agree_on_content_not_on_bytes() {
    let Some(c) = compressor() else { return };

    let inflated = inflate(&fixture("pacbio_hifi.bam"));
    let gpu = compress_host(&c, &inflated);

    let mut cpu = CompressedBatch::new();
    CpuCompressor::new()
        .compress_batch(inflated.data(), inflated.offsets(), &mut cpu)
        .unwrap();

    assert_eq!(round_trip(&gpu).data(), round_trip(&cpu).data());
    assert_eq!(gpu.len(), cpu.len());
    assert_ne!(
        gpu.data(),
        cpu.data(),
        "byte-identical output would mean the GPU path is not running nvCOMP"
    );
}
