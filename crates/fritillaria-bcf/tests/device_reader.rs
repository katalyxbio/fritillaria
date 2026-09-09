//! The reader-driving loop a BCF consumer writes, without a GPU.
//!
//! `fritillaria-cuda/tests/bcf_scan.rs` runs this same loop on real hardware.
//! This runs it through [`HostDeviceCodec`], the host-memory stand-in for a
//! device allocation, so the *driver* logic — batching, the carry, where a
//! batch resumes — is testable on a machine with no CUDA at all.
//!
//! It exists because it was needed. The GPU version of this test was written
//! first and failed on a rented VM for a reason that had nothing to do with the
//! GPU: an assertion about how many batches the reader would produce, which was
//! wrong. That is a slow and expensive way to find out something a host test
//! answers in a second.
//!
//! # What this pins down
//!
//! **A BCF batch almost always ends mid-record.** bcftools packs BGZF blocks
//! full, so unlike BAM the partial-tail path is the common case rather than the
//! edge case. If the speculative scan fell back to the serial walk once per
//! batch it would still be correct, and would perform exactly like the design
//! it replaces — so the fallback count is asserted, not assumed.

use std::path::PathBuf;

use fritillaria_bcf::{Proof, header::parse_header, scan_records, scan_records_speculative};
use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, HostDeviceCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

const FIXTURES: &[&str] = &["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"];

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(path).expect("fixture missing; see testdata/README.md")
}

fn inflate(raw: &[u8]) -> InflateBatch {
    let spans = discover_blocks(raw, 0).expect("BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .expect("must inflate");
    out
}

/// What one pass over a file through the reader found.
struct Pass {
    /// Every record's length, in file order. Position-independent, so it
    /// survives the rebasing the reader's carry does — and still catches a
    /// boundary landing one byte off.
    lengths: Vec<usize>,
    batches: usize,
    fallbacks: usize,
}

/// Drives a whole BCF through the reader, scanning each batch speculatively.
///
/// This is the loop a consumer writes. It is short, which is the point: the
/// carry and the window growth are the reader's problem.
fn drive(raw: &[u8], blocks_per_batch: usize) -> Pass {
    let codec = HostDeviceCodec::new();
    let mut reader = DeviceBgzfReader::new(raw, &codec).with_blocks_per_batch(blocks_per_batch);

    let mut pass = Pass {
        lengths: Vec::new(),
        batches: 0,
        fallbacks: 0,
    };
    let (mut samples, mut contigs) = (0u32, 0u32);
    let mut header_parsed = false;

    while let Some(batch) = reader.next_batch().expect("batch") {
        let staged = batch.data.to_host().expect("stage");
        let begin = if header_parsed {
            batch.start
        } else {
            // The one unavoidable host read: the header says where records
            // begin, and carries the sample and contig counts the sieve keys
            // on.
            let Ok(header) = parse_header(staged.data()) else {
                drop(batch);
                reader.carry_from(0).expect("carry");
                continue;
            };
            samples = u32::try_from(header.sample_count()).unwrap();
            contigs = u32::try_from(header.dictionary.contigs.len()).unwrap();
            header_parsed = true;
            header.records_start
        };

        let scan = scan_records_speculative(staged.data(), begin, samples, contigs).expect("scan");
        pass.batches += 1;
        if matches!(scan.proof, Proof::Walked { .. }) {
            pass.fallbacks += 1;
        }
        for (i, &start) in scan.offsets.iter().enumerate() {
            let end = scan.offsets.get(i + 1).copied().unwrap_or(scan.tail);
            pass.lengths.push(end - start);
        }

        let tail = scan.tail;
        drop(batch);
        reader.carry_from(tail).expect("carry");
    }

    pass
}

/// Ground truth: whole file, one serial scan.
fn whole_file(raw: &[u8]) -> Vec<usize> {
    let host = inflate(raw);
    let buf = host.data();
    let header = parse_header(buf).expect("header");
    let (offsets, _) = scan_records(buf, header.records_start).expect("scan");
    offsets
        .iter()
        .enumerate()
        .map(|(i, &start)| offsets.get(i + 1).copied().unwrap_or(buf.len()) - start)
        .collect()
}

#[test]
fn batching_does_not_change_the_answer() {
    for name in FIXTURES {
        let raw = fixture(name);
        let expected = whole_file(&raw);
        assert!(!expected.is_empty(), "{name}: fixture has no records");

        for blocks_per_batch in [1usize, 2, 4] {
            let pass = drive(&raw, blocks_per_batch);
            let context = format!("{name} at {blocks_per_batch} blocks/batch");
            assert_eq!(
                pass.lengths.len(),
                expected.len(),
                "{context}: record count"
            );
            assert_eq!(pass.lengths, expected, "{context}: record lengths");
            assert_eq!(
                pass.fallbacks, 0,
                "{context}: the tiling must hold at a seam, or the scan \
                 degrades to the serial walk once per batch"
            );
        }
    }
}

#[test]
fn a_seam_is_actually_crossed() {
    // Without this, the test above could be asserting "0 fallbacks" over runs
    // that never produced a second batch — which is exactly what happened on a
    // GPU VM before this test existed.
    //
    // `blocks_per_batch` budgets *compressed* bytes at 64 KiB per block, and
    // genotype data compresses about 20x, so a small number still covers a lot
    // of file. The panel fixture at one block per batch is the configuration
    // that actually splits; pinning the number here means a change to the
    // reader's batching fails loudly instead of quietly stopping the seam from
    // being tested.
    let raw = fixture("kg_phase3.bcf");
    let pass = drive(&raw, 1);
    assert!(
        pass.batches >= 3,
        "expected the panel fixture to split into at least 3 batches, got {}",
        pass.batches
    );
    assert_eq!(pass.fallbacks, 0);

    // And the small fixtures do not split at all, which is worth recording so
    // the zero-fallback claim above is not read as covering them.
    assert_eq!(
        drive(&fixture("giab_hg002.bcf"), 1).batches,
        1,
        "the single-sample fixture fits one batch; its seams are untested here"
    );
}
