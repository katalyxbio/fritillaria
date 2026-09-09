//! The reader-driving loop a FASTQ consumer writes, without a GPU.
//!
//! `fritillaria-cuda/tests/fastq_decode.rs` runs this same loop on real
//! hardware. This runs it through [`HostDeviceCodec`], the host-memory stand-in
//! for a device allocation, so the *driver* logic — batching, the carry, where a
//! batch resumes — is testable on a machine with no CUDA at all.
//!
//! It exists because the equivalent BCF test was written after a GPU run failed
//! on a rented VM for a reason that had nothing to do with the GPU: an assertion
//! about how many batches the reader would produce, which was wrong. That is a
//! slow and expensive way to find out something a host test answers in a second.
//!
//! # What this pins down
//!
//! **FASTQ has no header.** Records begin at offset 0, so unlike BAM and BCF
//! there is no host round trip to find `records_start` — the loop below is the
//! whole contract.
//!
//! **A long read can exceed a whole batch.** The ONT fixture's largest record is
//! bigger than several BGZF blocks, so at one block per batch the reader has to
//! carry compressed blocks forward and grow its window until the record fits.
//! That path is driven here rather than assumed.

use std::path::PathBuf;

use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, HostDeviceCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};
use fritillaria_fastq::columnar::{Proof, RecordBatch, scan_records_speculative};

const FIXTURES: &[&str] = &[
    "ont_ultralong.fastq.gz",
    "pacbio_hifi.fastq.gz",
    "illumina.fastq.gz",
];

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
    /// Read lengths, in file order. Position-independent, so they survive the
    /// rebasing the reader's carry does, and still catch a boundary landing one
    /// byte off.
    lengths: Vec<u32>,
    batches: usize,
    fallbacks: usize,
}

/// Drives a whole FASTQ through the reader, scanning each batch speculatively.
fn drive(raw: &[u8], blocks_per_batch: usize) -> Pass {
    let codec = HostDeviceCodec::new();
    let mut reader = DeviceBgzfReader::new(raw, &codec).with_blocks_per_batch(blocks_per_batch);

    let mut pass = Pass {
        lengths: Vec::new(),
        batches: 0,
        fallbacks: 0,
    };

    while let Some(batch) = reader.next_batch().expect("batch") {
        let staged = batch.data.to_host().expect("stage");
        let scan = scan_records_speculative(staged.data(), batch.start).expect("scan");
        pass.batches += 1;
        if matches!(scan.proof, Proof::Walked { .. }) {
            pass.fallbacks += 1;
        }

        for &offset in &scan.offsets {
            let record =
                fritillaria_fastq::columnar::Record::new(&staged.data()[offset..]).expect("record");
            pass.lengths.push(record.bounds().sequence_len);
        }

        let tail = scan.tail;
        drop(batch);
        reader.carry_from(tail).expect("carry");
    }

    pass
}

/// Ground truth: whole file, one scan.
fn whole_file(raw: &[u8]) -> Vec<u32> {
    let host = inflate(raw);
    let mut batch = RecordBatch::new();
    batch.decode(host.data(), 0).expect("decode");
    batch.sequence_lengths()
}

#[test]
fn batching_does_not_change_the_answer() {
    for name in FIXTURES {
        let raw = fixture(name);
        let expected = whole_file(&raw);
        assert!(!expected.is_empty(), "{name}: fixture has no records");

        for blocks_per_batch in [1usize, 2, 8] {
            let pass = drive(&raw, blocks_per_batch);
            let context = format!("{name} at {blocks_per_batch} blocks/batch");
            assert_eq!(
                pass.lengths.len(),
                expected.len(),
                "{context}: record count"
            );
            assert_eq!(pass.lengths, expected, "{context}: read lengths");
            assert_eq!(
                pass.fallbacks, 0,
                "{context}: the tiling must hold at a seam, or the scan degrades \
                 to the serial walk once per batch"
            );
        }
    }
}

#[test]
fn a_seam_is_actually_crossed() {
    // Without this, the test above could be asserting "0 fallbacks" over runs
    // that never produced a second batch — which is exactly what happened on a
    // GPU VM before the BCF version of this test existed.
    //
    // `blocks_per_batch` budgets *compressed* bytes at 64 KiB per block, and
    // FASTQ compresses well, so a small number still covers a lot of file.
    let raw = fixture("illumina.fastq.gz");
    let pass = drive(&raw, 1);
    assert!(
        pass.batches >= 3,
        "expected the Illumina fixture to split into at least 3 batches, got {}",
        pass.batches
    );
    assert_eq!(pass.fallbacks, 0);
}

#[test]
fn a_record_larger_than_a_batch_is_carried_until_it_fits() {
    // The ONT case. At one block per batch the reader's window starts far
    // smaller than the largest record, so it must carry compressed blocks
    // forward and grow until the record fits — the loop the caller would
    // otherwise have to write, and get wrong.
    let raw = fixture("ont_ultralong.fastq.gz");
    let expected = whole_file(&raw);
    let longest = *expected.iter().max().expect("records");
    assert!(
        longest as usize > 65536,
        "this fixture is supposed to hold a record larger than a BGZF block; \
         longest read is {longest} bases"
    );

    let pass = drive(&raw, 1);
    assert_eq!(pass.lengths, expected, "read lengths across the carry");
    assert_eq!(pass.fallbacks, 0);
}

#[test]
fn every_batch_resumes_exactly_where_the_last_one_stopped() {
    // The seam contract itself: no record skipped, none counted twice. Compared
    // against the whole-file scan, which cannot have a seam.
    for name in FIXTURES {
        let raw = fixture(name);
        let expected = whole_file(&raw);
        let pass = drive(&raw, 2);

        assert_eq!(
            pass.lengths.len(),
            expected.len(),
            "{name}: a skipped or duplicated record changes the count"
        );
        assert_eq!(pass.lengths, expected, "{name}: read lengths");
    }
}
