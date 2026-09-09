//! The CPU compressor against real files, and against htslib.
//!
//! # Why this cannot be a byte comparison
//!
//! Every other differential test in this workspace pins GPU output to the CPU
//! reference byte for byte. Compression has no such oracle: two valid DEFLATE
//! streams of the same input differ legitimately, and htslib's own output would
//! change with its libdeflate version. So the net here is three weaker checks
//! that together are still binding:
//!
//! 1. **Round trip.** Inflating our output reproduces the input exactly, block
//!    for block — not merely byte for byte in aggregate, because block
//!    boundaries are information (see below).
//! 2. **htslib accepts it.** `samtools`/`bcftools` read the rewritten file and
//!    agree on the record count. This is the acceptance bar and it is binary.
//! 3. **A ratio floor.** Asserted, not reported. `docs/compression.md` warns
//!    that a number in a benchmark report is not a test, and a compressor that
//!    quietly fell back to storing everything would pass (1) and (2).
//!
//! # Block boundaries are the thing being protected
//!
//! htslib starts a new BGZF block rather than splitting a BAM record, which is
//! what makes a block start almost always a record start — the property this
//! library's GPU record scan depends on to avoid one serial chain per batch. So
//! a compressor that split a chunk to make it fit would produce a file that is
//! valid, reads correctly, and is *slower for us to read*. These tests assert
//! the boundaries survive, not just the bytes.

// Ratios are reported and compared in floating point; the sizes involved are
// megabytes, nowhere near a f64 mantissa.
#![allow(clippy::cast_precision_loss)]

use std::path::PathBuf;
use std::process::Command;

use fritillaria_bgzf::{CpuCodec, CpuCompressor, discover_blocks, is_eof_block};
use fritillaria_core::{BlockCodec, BlockCompressor, CompressedBatch, InflateBatch};

/// Fixtures written by htslib, with the record count `samtools`/`bcftools`
/// reports and the tool that reads them.
const FIXTURES: &[(&str, &str)] = &[
    ("pacbio_hifi.bam", "samtools"),
    ("ont_ultralong.bam", "samtools"),
    ("htslib_multiblock.bam", "samtools"),
    ("kg_phase3.bcf", "bcftools"),
    ("giab_hg002.bcf", "bcftools"),
];

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn inflate(raw: &[u8]) -> InflateBatch {
    let spans = discover_blocks(raw, 0).expect("fixture is not valid BGZF");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .expect("fixture failed to inflate");
    out
}

fn recompress(inflated: &InflateBatch) -> CompressedBatch {
    let mut out = CompressedBatch::new();
    CpuCompressor::new()
        .compress_batch(inflated.data(), inflated.offsets(), &mut out)
        .expect("real BGZF payloads must all be compressible");
    out
}

/// The record count a tool reports, or `None` if the tool is not installed.
///
/// `samtools view -c` counts records; the same flag on `bcftools` is
/// `--min-ac`, so that path prints the headerless body and counts lines
/// instead. Getting this wrong is not silent — `bcftools` refuses to parse the
/// path as an allele count — but it is worth naming, because a *quieter*
/// mistake here would have this test comparing two zeroes and passing.
fn count_records(tool: &str, file: &std::path::Path) -> Option<usize> {
    let args: &[&str] = if tool == "bcftools" {
        &["view", "-H", "--no-version"]
    } else {
        &["view", "-c"]
    };

    let out = Command::new(tool).args(args).arg(file).output().ok()?;
    assert!(
        out.status.success(),
        "{tool} rejected {}: {}",
        file.display(),
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let count = if tool == "bcftools" {
        stdout.lines().count()
    } else {
        stdout.trim().parse().expect("record count")
    };

    assert!(
        count > 0,
        "{tool} found no records in {}, so a comparison against it proves \
         nothing",
        file.display()
    );
    Some(count)
}

/// Inflating our output must reproduce the fixture's payload exactly, and with
/// the same block boundaries it went in with.
#[test]
fn recompressed_fixtures_round_trip_with_their_boundaries_intact() {
    for (name, _) in FIXTURES {
        let raw = std::fs::read(path(name)).expect("fixture missing; see testdata/README.md");
        let inflated = inflate(&raw);
        let compressed = recompress(&inflated);

        assert_eq!(
            compressed.len(),
            inflated.len(),
            "{name}: block count changed, so a boundary moved"
        );
        assert!(
            compressed.is_consistent(),
            "{name}: batch does not describe itself"
        );
        assert!(
            is_eof_block(&compressed.data()[compressed.data().len() - 28..]),
            "{name}: re-framing the fixture's trailing empty block must \
             reproduce the EOF marker, or every tool reports truncation"
        );

        let back = inflate(compressed.data());
        assert_eq!(back.data(), inflated.data(), "{name}: payload changed");
        assert_eq!(
            back.offsets(),
            inflated.offsets(),
            "{name}: block boundaries moved, which silently costs the GPU \
             record scan its per-block parallelism"
        );
    }
}

/// The acceptance bar: htslib reads what we wrote and finds the same records.
#[test]
fn htslib_reads_our_output_and_agrees_on_the_record_count() {
    let dir = std::env::temp_dir().join("fritillaria-compress-test");
    std::fs::create_dir_all(&dir).unwrap();

    let mut checked = 0;
    for (name, tool) in FIXTURES {
        let src = path(name);
        let Some(expected) = count_records(tool, &src) else {
            continue;
        };

        let compressed = recompress(&inflate(&std::fs::read(&src).unwrap()));
        let dst = dir.join(name);
        std::fs::write(&dst, compressed.data()).unwrap();

        assert_eq!(
            count_records(tool, &dst),
            Some(expected),
            "{tool} read a different number of records from our {name}"
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "neither samtools nor bcftools is installed, so the acceptance bar was \
         not checked at all — this must fail rather than pass quietly"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A ratio floor, asserted rather than printed.
///
/// The floor is deliberately loose — this is level-6 miniz against level-6
/// libdeflate, and libdeflate genuinely wins — but it is tight enough that a
/// compressor which stopped compressing would fail it. `docs/compression.md`
/// records the exact measured figures.
#[test]
fn the_ratio_stays_within_reach_of_htslib() {
    /// Largest tolerated excess over htslib's own output, per fixture.
    const TOLERANCE: f64 = 0.10;

    for (name, _) in FIXTURES {
        let raw = std::fs::read(path(name)).expect("fixture missing");
        let inflated = inflate(&raw);
        let compressed = recompress(&inflated);

        let ours = compressed.byte_len() as f64;
        let htslib = raw.len() as f64;
        let excess = ours / htslib - 1.0;

        println!(
            "{name:24} htslib {:>9} ours {:>9} ({:+.1}%)  ratio {:.2}x",
            raw.len(),
            compressed.byte_len(),
            excess * 100.0,
            inflated.data().len() as f64 / ours,
        );

        assert!(
            excess <= TOLERANCE,
            "{name}: {:.1}% larger than htslib, past the {:.0}% floor",
            excess * 100.0,
            TOLERANCE * 100.0
        );
    }
}
