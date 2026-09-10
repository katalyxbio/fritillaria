//! Ultra-long ONT reads: records that span BGZF block and batch boundaries.
//!
//! # Why this fixture exists
//!
//! Until it was added, **no fixture in this workspace contained a record
//! spanning a BGZF block boundary**, despite `testdata/README.md` listing that
//! as a covered case. htslib calls `bgzf_flush_try` before writing each record
//! and starts a new block rather than splitting one, so an htslib-written BAM
//! only splits records *larger than 65280 bytes*. Illumina records are ~300
//! bytes and PacBio HiFi records top out around 56 KB, so neither can produce
//! the case at all.
//!
//! Ultra-long ONT can: the largest record here is 254 KB and crosses three
//! block boundaries. That makes this the only fixture that tests the
//! concatenation seam against a real file rather than against one this crate
//! wrote itself.
//!
//! The second thing it tests is the *batch* seam, which is the one that
//! actually breaks drivers: a record can be larger than a whole batch of
//! blocks, so a caller must carry a partial record forward and must not assume
//! a record fits in one batch. `scan_records` reports a tail offset for exactly
//! this, and `driver_carries_a_record_across_batch_edges` is the first test to
//! drive it the way a real consumer would.

use std::path::PathBuf;

use fritillaria_bam::columnar::{Record, header::parse_header, scan_records};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, BlockSpan, InflateBatch};

const FIXTURE: &str = "ont_ultralong.bam";

/// htslib's BGZF payload cap. A record longer than this is the only way an
/// htslib-written BAM ever splits one across blocks.
const BGZF_PAYLOAD_CAP: usize = 65280;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn raw_bytes() -> Vec<u8> {
    std::fs::read(fixture(FIXTURE)).expect("fixture missing; see testdata/README.md")
}

fn inflate_spans(raw: &[u8], spans: &[BlockSpan]) -> InflateBatch {
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, spans, &mut out)
        .expect("htslib BGZF must inflate and verify");
    out
}

fn samtools(args: &[&str]) -> Option<std::process::Output> {
    match std::process::Command::new("samtools").args(args).output() {
        Ok(output) => Some(output),
        Err(err) => {
            eprintln!("SKIP: samtools unavailable ({err})");
            None
        }
    }
}

/// `samtools view` output, one line per record.
fn ground_truth() -> Option<Vec<String>> {
    let path = fixture(FIXTURE);
    let out = samtools(&["view", path.to_str().unwrap()])?;
    assert!(
        out.status.success(),
        "samtools view failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(
        String::from_utf8(out.stdout)
            .expect("samtools emits UTF-8 here")
            .lines()
            .map(str::to_owned)
            .collect(),
    )
}

/// Cumulative uncompressed end offset of each block, so a record's extent can
/// be tested against the boundaries it crosses.
fn block_bounds(spans: &[BlockSpan]) -> Vec<usize> {
    let mut acc = 0usize;
    spans
        .iter()
        .map(|s| {
            acc += s.isize as usize;
            acc
        })
        .collect()
}

/// How many block boundaries fall strictly inside `[start, end)`.
fn boundaries_crossed(bounds: &[usize], start: usize, end: usize) -> usize {
    bounds.partition_point(|&b| b < end) - bounds.partition_point(|&b| b <= start)
}

#[test]
fn the_fixture_really_spans_block_boundaries() {
    // Asserted rather than assumed: this property is the entire reason the
    // fixture is committed, and it was absent from every earlier fixture
    // without anyone noticing. If htslib's blocking ever changes, or the file
    // is regenerated from shorter reads, every other test in this file quietly
    // stops testing the seam — so this one fails loudly instead.
    let raw = raw_bytes();
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let batch = inflate_spans(&raw, &spans);
    let bounds = block_bounds(&spans);

    let header = parse_header(batch.data()).expect("header must parse");
    let (offsets, tail) = scan_records(batch.data(), header.records_start).expect("scan");
    assert_eq!(tail, batch.data().len(), "whole file is buffered");

    let mut crossing = 0;
    let mut max_crossed = 0;
    let mut largest = 0;
    for (i, &start) in offsets.iter().enumerate() {
        let end = offsets.get(i + 1).copied().unwrap_or(batch.data().len());
        let crossed = boundaries_crossed(&bounds, start, end);
        if crossed > 0 {
            crossing += 1;
        }
        max_crossed = max_crossed.max(crossed);
        largest = largest.max(end - start);
    }

    assert!(
        largest > BGZF_PAYLOAD_CAP,
        "a record must exceed {BGZF_PAYLOAD_CAP} bytes or htslib will never split one; \
         largest is {largest}"
    );
    assert!(
        crossing >= 2,
        "expected at least two records to cross a block boundary, got {crossing}"
    );
    assert!(
        max_crossed >= 3,
        "expected a record spanning at least four blocks, got {max_crossed} boundaries"
    );
}

#[test]
fn records_spanning_blocks_match_samtools() {
    let Some(truth) = ground_truth() else { return };
    let raw = raw_bytes();
    let spans = discover_blocks(&raw, 0).expect("parse");
    let batch = inflate_spans(&raw, &spans);
    let header = parse_header(batch.data()).expect("header must parse");
    let (offsets, _) = scan_records(batch.data(), header.records_start).expect("scan");

    assert_eq!(
        offsets.len(),
        truth.len(),
        "record count must match samtools"
    );

    for (i, &offset) in offsets.iter().enumerate() {
        let record = Record::new(&batch.data()[offset..]).expect("record must parse");
        let cols: Vec<&str> = truth[i].split('\t').collect();

        assert_eq!(
            std::str::from_utf8(record.name()).unwrap(),
            cols[0],
            "record {i} name"
        );
        // A sequence read across three block boundaries is the point of the
        // fixture: if concatenation were padded or gapped, this is where it
        // would show up as garbage bases rather than as a length mismatch.
        let sequence = record.sequence();
        let expected_seq = if cols[9] == "*" { "" } else { cols[9] };
        assert_eq!(
            std::str::from_utf8(&sequence).unwrap(),
            expected_seq,
            "record {i} sequence ({} bp)",
            sequence.len()
        );

        let quals = record.qualities();
        if cols[10] == "*" {
            assert!(
                quals.is_empty() || quals.iter().all(|&q| q == 0xff),
                "record {i} should have absent qualities"
            );
        } else {
            let ours: String = quals.iter().map(|q| (q + 33) as char).collect();
            assert_eq!(ours, cols[10], "record {i} qualities");
        }
    }
}

#[test]
fn a_record_with_no_sequence_still_parses() {
    // The fixture includes a secondary alignment: l_seq == 0, no qualities, but
    // a real multi-op CIGAR. Deriving the aux offset from l_seq makes this the
    // record most likely to expose an offset bug.
    let raw = raw_bytes();
    let spans = discover_blocks(&raw, 0).expect("parse");
    let batch = inflate_spans(&raw, &spans);
    let header = parse_header(batch.data()).expect("header must parse");
    let (offsets, _) = scan_records(batch.data(), header.records_start).expect("scan");

    let empty = offsets
        .iter()
        .map(|&o| Record::new(&batch.data()[o..]).expect("record must parse"))
        .find(|r| r.sequence_len() == 0)
        .expect("fixture must contain a record with no stored sequence");

    assert!(empty.sequence().is_empty());
    assert!(empty.qualities().is_empty());
    assert!(
        empty.cigar_op_count() > 100,
        "this record should still carry a real CIGAR, got {} ops",
        empty.cigar_op_count()
    );
    // Aux must still be found, which it will not be if the offset arithmetic
    // treats an absent sequence as anything other than zero bytes.
    for field in empty.aux() {
        field.expect("aux of a sequence-less record must decode");
    }
}

#[test]
fn driver_carries_a_record_across_batch_edges() {
    // What a real consumer has to do, and what nothing in this repo does yet:
    // inflate a few blocks at a time, scan what is buffered, and carry the
    // trailing partial record into the next batch. With a 254 KB record and
    // batches of one 64 KB block, a single record spans four batches — so a
    // driver that assumed one record per batch, or that dropped the tail, would
    // lose records rather than merely mis-order them.
    let Some(truth) = ground_truth() else { return };
    let raw = raw_bytes();
    let spans = discover_blocks(&raw, 0).expect("parse");

    for blocks_per_batch in [1usize, 2, 3] {
        let mut carried: Vec<u8> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let mut started = false;
        let mut skip = 0usize;

        for chunk in spans.chunks(blocks_per_batch) {
            let batch = inflate_spans(&raw, chunk);
            carried.extend_from_slice(batch.data());

            if !started {
                // The header only becomes parseable once enough blocks have
                // arrived; with one block per batch that is not the first.
                match parse_header(&carried) {
                    Ok(header) => {
                        skip = header.records_start;
                        started = true;
                    }
                    Err(_) => continue,
                }
            }

            let (offsets, tail) = scan_records(&carried, skip).expect("scan must not fail");
            for &off in &offsets {
                let record = Record::new(&carried[off..]).expect("record must parse");
                names.push(String::from_utf8(record.name().to_vec()).expect("ASCII name"));
            }
            carried.drain(..tail);
            skip = 0;
        }

        assert!(carried.is_empty(), "no bytes may be left unconsumed");

        let expected: Vec<String> = truth
            .iter()
            .map(|l| l.split('\t').next().unwrap().to_owned())
            .collect();
        assert_eq!(
            names, expected,
            "with {blocks_per_batch} block(s) per batch, every record must be \
             recovered exactly once and in order"
        );
    }
}
