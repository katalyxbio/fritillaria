//! The blocked boundary scan against real BGZF block layouts.
//!
//! `blocked.rs`'s unit tests use synthetic records and evenly-spaced blocks.
//! This runs the same algorithm over the block boundaries htslib actually
//! produced, which is the only place the assumption it speculates on — that a
//! block start is usually a record start — is either true or false for real.
//!
//! Both cases are represented on purpose:
//!
//! - `htslib.bam`, `htslib_multiblock.bam`, `pacbio_hifi.bam` — every block
//!   starts on a record boundary, so every speculative walk is right and the
//!   fallback never runs.
//! - `ont_ultralong.bam` — a 254 KB record covers three block boundaries, so
//!   three blocks have no record starting in them and their speculative walks
//!   are junk. This is the fixture that decides whether the fallback works.

use std::path::PathBuf;

use fritillaria_bam::columnar::{
    blocked::scan_records_blocked, header::parse_header, scan_records,
};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

const FIXTURES: &[&str] = &[
    "htslib.bam",
    "htslib_multiblock.bam",
    "pacbio_hifi.bam",
    "ont_ultralong.bam",
];

fn inflate(name: &str) -> InflateBatch {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    let raw = std::fs::read(path).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("htslib BGZF must inflate and verify");
    out
}

#[test]
fn matches_the_sequential_scan_on_every_fixture() {
    for name in FIXTURES {
        let batch = inflate(name);
        let header = parse_header(batch.data()).expect("header must parse");
        let (expected, expected_tail) =
            scan_records(batch.data(), header.records_start).expect("sequential scan");

        let scan = scan_records_blocked(batch.data(), batch.offsets(), header.records_start)
            .unwrap_or_else(|e| panic!("{name}: blocked scan failed: {e}"));

        assert_eq!(scan.offsets, expected, "{name}: record offsets");
        assert_eq!(scan.tail, expected_tail, "{name}: tail");
        assert_eq!(
            scan.counts.iter().sum::<usize>(),
            expected.len(),
            "{name}: counts must account for every record"
        );
    }
}

#[test]
fn the_ont_fixture_exercises_the_fallback_and_the_others_do_not() {
    // Asserted rather than assumed. If every fixture took the fast path, the
    // test above would pass while the fallback — the whole reason the algorithm
    // is correct rather than merely usually-right — went unexecuted.
    let mut measured = Vec::new();

    for name in FIXTURES {
        let batch = inflate(name);
        let header = parse_header(batch.data()).expect("header must parse");
        let scan = scan_records_blocked(batch.data(), batch.offsets(), header.records_start)
            .expect("blocked scan");
        let entryless = scan.entries[1..].iter().filter(|e| e.is_none()).count();
        measured.push((*name, scan.fallbacks, entryless));
    }

    for (name, fallbacks, entryless) in &measured {
        if *name == "ont_ultralong.bam" {
            assert!(
                *fallbacks >= 1,
                "{name} must exercise the fallback, got {fallbacks}"
            );
            // Three blocks are covered by the 254 KB record, plus the EOF
            // block: the speculation has nothing useful to say about any of
            // them, which is exactly the case the fallback exists for.
            assert!(
                *entryless >= 4,
                "{name}: expected several blocks with no record starting in \
                 them, got {entryless}"
            );
        } else {
            // Zero: htslib ends each block on a record boundary, and in these
            // files the header fills whole blocks, so every block start the
            // chain reaches is a record start and the fast path covers all of
            // them.
            assert_eq!(
                *fallbacks, 0,
                "{name}: every block start should already be a record start"
            );
            // Only the EOF block, which is empty by definition.
            assert_eq!(
                *entryless, 1,
                "{name}: only the EOF block should be entryless"
            );
        }
    }
}

#[test]
fn every_starting_offset_is_a_real_record_boundary() {
    // Independent of the sequential scan: walk the emitted offsets and check
    // each record's declared length lands exactly on the next offset. A scan
    // that drifted by a few bytes would still produce the right *count*.
    for name in FIXTURES {
        let batch = inflate(name);
        let buf = batch.data();
        let header = parse_header(buf).expect("header must parse");
        let scan =
            scan_records_blocked(buf, batch.offsets(), header.records_start).expect("blocked scan");

        for (i, &offset) in scan.offsets.iter().enumerate() {
            let block_size =
                u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            let end = offset + 4 + block_size;
            let next = scan.offsets.get(i + 1).copied().unwrap_or(scan.tail);
            assert_eq!(
                end,
                next,
                "{name}: record {i} does not end where {} begins",
                i + 1
            );
        }
    }
}
