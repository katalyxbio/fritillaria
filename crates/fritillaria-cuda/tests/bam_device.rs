//! Differential tests for columnar BAM decode on the device.
//!
//! The claim under test: **decoding records on the device must produce exactly
//! what decoding them on the host produces.** Every assertion here copies the
//! device columns back and compares them against `RecordBatch::decode` and
//! `Record` views over the same buffer. That round trip is what the device path
//! exists to avoid in production; here it is the measuring instrument.
//!
//! The fixtures are chosen to cover both halves of the boundary-scan algorithm
//! (see `fritillaria_bam::blocked`):
//!
//! - `htslib.bam`, `htslib_multiblock.bam`, `pacbio_hifi.bam` — every block
//!   starts on a record boundary, so every speculative walk is right.
//! - `ont_ultralong.bam` — a 254 KB record covers three block boundaries, so
//!   several blocks have no record starting in them and their speculative
//!   walks are junk. This is the fixture that exercises the fallback, and the
//!   one most likely to catch a kernel that only handles the easy case.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure, so
//! skipping cannot quietly hide a broken path.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bam::{Record, RecordBatch, header::parse_header};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, DeviceBlockCodec, DeviceInflateBatch, InflateBatch};
use fritillaria_cuda::{BamDecoder, CudaCodec};

const FIXTURES: &[&str] = &[
    "htslib.bam",
    "htslib_multiblock.bam",
    "pacbio_hifi.bam",
    "ont_ultralong.bam",
];

fn parts() -> Option<(CudaCodec, BamDecoder)> {
    match (CudaCodec::new(), BamDecoder::new()) {
        (Ok(codec), Ok(decoder)) => Some((codec, decoder)),
        (Err(err), _) | (_, Err(err)) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

fn fixture_bytes(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(path).expect("fixture missing; see testdata/README.md")
}

/// The host oracle: inflated bytes, and the columns decoded from them.
fn host_side(raw: &[u8]) -> (InflateBatch, usize, RecordBatch, usize) {
    let spans = discover_blocks(raw, 0).expect("htslib BGZF must parse");
    let mut inflated = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut inflated)
        .expect("CPU reference must inflate");
    let header = parse_header(inflated.data()).expect("header must parse");
    let start = header.records_start;

    let mut batch = RecordBatch::new();
    let tail = batch
        .decode(inflated.data(), start)
        .expect("host decode must succeed");
    (inflated, start, batch, tail)
}

#[test]
fn columns_match_the_host_decode_on_every_fixture() {
    let Some((codec, decoder)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture_bytes(name);
        let (_, start, expected, expected_tail) = host_side(&raw);

        let spans = discover_blocks(&raw, 0).unwrap();
        let mut device = DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut device)
            .unwrap_or_else(|e| panic!("{name}: device inflate failed: {e:?}"));

        let records = decoder
            .decode(&device, start)
            .unwrap_or_else(|e| panic!("{name}: device decode failed: {e:?}"));
        let actual = records
            .to_host()
            .unwrap_or_else(|e| panic!("{name}: column download failed: {e:?}"));

        assert_eq!(records.len(), expected.len(), "{name}: record count");
        assert_eq!(records.tail(), expected_tail, "{name}: tail");
        assert_eq!(
            actual.record_offsets(),
            expected.record_offsets(),
            "{name}: record offsets"
        );
        assert_eq!(
            actual.reference_sequence_id(),
            expected.reference_sequence_id(),
            "{name}: reference_sequence_id"
        );
        assert_eq!(actual.position(), expected.position(), "{name}: position");
        assert_eq!(actual.flags(), expected.flags(), "{name}: flags");
        assert_eq!(
            actual.mapping_quality(),
            expected.mapping_quality(),
            "{name}: mapping_quality"
        );
        assert_eq!(
            actual.sequence_len(),
            expected.sequence_len(),
            "{name}: sequence_len"
        );
        assert_eq!(
            actual.mate_reference_sequence_id(),
            expected.mate_reference_sequence_id(),
            "{name}: mate_reference_sequence_id"
        );
        assert_eq!(
            actual.mate_position(),
            expected.mate_position(),
            "{name}: mate_position"
        );
        assert_eq!(
            actual.template_length(),
            expected.template_length(),
            "{name}: template_length"
        );
    }
}

#[test]
fn field_boundaries_match_the_host_record_views() {
    // The columns the host path has no counterpart for. Getting these wrong
    // points a consumer at the middle of a sequence rather than at its start,
    // which is silent: the bytes are all there, just misattributed.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture_bytes(name);
        let (inflated, start, expected, _) = host_side(&raw);

        let spans = discover_blocks(&raw, 0).unwrap();
        let mut device = DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut device)
            .unwrap();
        let records = decoder.decode(&device, start).unwrap();
        let bounds = records.field_bounds_to_host().unwrap();

        for (i, &offset) in expected.record_offsets().iter().enumerate() {
            let record = Record::new(&inflated.data()[offset..]).expect("record must parse");
            let name_len = record.name().len() + 1; // l_read_name counts the NUL
            let cigar = 36 + name_len;
            let sequence = cigar + 4 * record.cigar_op_count();
            let quality = sequence + record.sequence_packed().len();
            let aux = quality + record.qualities().len();
            let end = aux + record.aux_raw().len();

            assert_eq!(
                bounds.cigar_start[i] as usize, cigar,
                "{name} rec {i} cigar"
            );
            assert_eq!(
                bounds.sequence_start[i] as usize, sequence,
                "{name} rec {i} sequence"
            );
            assert_eq!(
                bounds.quality_start[i] as usize, quality,
                "{name} rec {i} quality"
            );
            assert_eq!(bounds.aux_start[i] as usize, aux, "{name} rec {i} aux");
            assert_eq!(bounds.record_end[i] as usize, end, "{name} rec {i} end");
        }
    }
}

#[test]
fn a_record_spanning_blocks_decodes_correctly() {
    // The ONT case in isolation: the fixture's largest record crosses three
    // block boundaries, so the block start it runs through is not a record
    // start and the speculative walk from there is meaningless.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let raw = fixture_bytes("ont_ultralong.bam");
    let (inflated, start, expected, _) = host_side(&raw);

    let spans = discover_blocks(&raw, 0).unwrap();
    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut device)
        .unwrap();
    let records = decoder.decode(&device, start).unwrap();
    let actual = records.to_host().unwrap();

    assert_eq!(actual.record_offsets(), expected.record_offsets());

    // Prove the case is really present rather than trusting the fixture name.
    let block_bounds = inflated.offsets();
    let largest = expected
        .record_offsets()
        .iter()
        .enumerate()
        .map(|(i, &o)| {
            let end = expected
                .record_offsets()
                .get(i + 1)
                .copied()
                .unwrap_or(inflated.data().len());
            (end - o, o, end)
        })
        .max()
        .expect("fixture has records");
    let crossed = block_bounds
        .iter()
        .filter(|&&b| b > largest.1 && b < largest.2)
        .count();
    assert!(
        crossed >= 3,
        "the largest record should cross at least three block boundaries, crossed {crossed}"
    );
}

#[test]
fn a_partial_trailing_record_reports_the_tail() {
    // A batch that ends mid-record is the seam a driver has to handle.
    //
    // It cannot be reproduced by truncating an ordinary htslib BAM at a block:
    // htslib ends a block on a record boundary rather than splitting a record,
    // so *every* prefix of blocks ends exactly on a record boundary and the
    // tail is never partial. Only a record larger than a block splits, which is
    // why this uses the ONT fixture and searches for a prefix that lands inside
    // its 254 KB record instead of assuming one.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let raw = fixture_bytes("ont_ultralong.bam");
    let spans = discover_blocks(&raw, 0).unwrap();

    let mut found = None;
    for k in 2..spans.len() {
        let prefix = &spans[..k];
        let mut inflated = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&raw, prefix, &mut inflated)
            .unwrap();
        let Ok(header) = parse_header(inflated.data()) else {
            continue;
        };
        let mut expected = RecordBatch::new();
        let tail = expected
            .decode(inflated.data(), header.records_start)
            .unwrap();
        if tail < inflated.data().len() {
            found = Some((k, header.records_start, expected, tail, inflated));
            break;
        }
    }

    let (k, start, expected, expected_tail, inflated) =
        found.expect("the ONT fixture must have a block prefix ending inside a record");

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans[..k], &mut device)
        .unwrap();
    let records = decoder.decode(&device, start).unwrap();

    assert_eq!(records.len(), expected.len(), "record count at {k} blocks");
    assert_eq!(records.tail(), expected_tail, "tail at {k} blocks");
    assert_eq!(
        records.to_host().unwrap().record_offsets(),
        expected.record_offsets()
    );
    assert!(
        expected_tail < inflated.data().len(),
        "the prefix search must have found a genuinely partial record"
    );
}

#[test]
fn an_empty_batch_decodes_to_no_records() {
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let mut device = DeviceInflateBatch::new();
    codec.inflate_batch_device(&[], &[], &mut device).unwrap();

    let records = decoder.decode(&device, 0).unwrap();
    assert!(records.is_empty());
    assert_eq!(records.tail(), 0);
    assert!(records.to_host().unwrap().is_empty());
}

#[test]
fn columns_report_the_device_they_live_on() {
    // A consumer holding its own context must be able to check this before
    // launching, rather than finding out inside a kernel.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let raw = fixture_bytes("htslib.bam");
    let (_, start, _, _) = host_side(&raw);
    let spans = discover_blocks(&raw, 0).unwrap();
    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut device)
        .unwrap();

    let records = decoder.decode(&device, start).unwrap();
    assert_eq!(records.device_ordinal(), device.device_ordinal().unwrap());
    assert_eq!(
        records.position().device_ordinal(),
        records.device_ordinal()
    );
}
