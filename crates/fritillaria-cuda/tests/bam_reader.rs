//! The whole stack: a BAM file in, records in device memory out.
//!
//! Everything below this has been tested in pieces — the codec inflates to the
//! device, the decoder turns a batch into columns, the reader batches and
//! carries. This is the first test that puts a *file* in one end and gets
//! device-resident records out of the other, which is the thing the project
//! claims to do.
//!
//! Host memory is touched exactly twice, both deliberately:
//!
//! - the BAM header, to learn where records start. It is tens of kilobytes and
//!   there is no way to know `records_start` without reading it.
//! - `to_host()` on the columns, purely as the measuring instrument — that is
//!   the transfer the design exists to delete, and no production path does it.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bam::{RecordBatch, header::parse_header};
use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};
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

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(path).expect("fixture missing; see testdata/README.md")
}

/// Ground truth: whole file, host codec, one decode pass.
fn whole_file(raw: &[u8]) -> (Vec<i32>, Vec<i32>, Vec<u16>) {
    let spans = discover_blocks(raw, 0).expect("BGZF must parse");
    let mut inflated = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut inflated)
        .expect("must inflate");
    let header = parse_header(inflated.data()).expect("header must parse");

    let mut columns = RecordBatch::new();
    columns
        .decode(inflated.data(), header.records_start)
        .expect("must decode");
    (
        columns.reference_sequence_id().to_vec(),
        columns.position().to_vec(),
        columns.flags().to_vec(),
    )
}

/// Reads a whole BAM through the device stack, accumulating columns.
///
/// This is the loop a GPU consumer writes today. It is short, which is the
/// point: the carry and the growth are the reader's problem, not theirs.
fn drive(
    raw: &[u8],
    codec: &CudaCodec,
    decoder: &BamDecoder,
    blocks_per_batch: usize,
) -> (Vec<i32>, Vec<i32>, Vec<u16>) {
    let mut reader = DeviceBgzfReader::new(raw, codec).with_blocks_per_batch(blocks_per_batch);
    let (mut refs, mut positions, mut flags) = (Vec::new(), Vec::new(), Vec::new());
    let mut header_parsed = false;

    while let Some(batch) = reader.next_batch().expect("batch") {
        let begin = if header_parsed {
            batch.start
        } else {
            // The one unavoidable host read: the header names the references
            // and says where records begin.
            let host = batch.data.to_host().expect("header download");
            let Ok(header) = parse_header(host.data()) else {
                reader.carry_from(0).expect("carry");
                continue;
            };
            header_parsed = true;
            header.records_start
        };

        let records = decoder.decode(&batch.data, begin).expect("device decode");
        let tail = records.tail();
        let host = records.to_host().expect("column download");
        refs.extend_from_slice(host.reference_sequence_id());
        positions.extend_from_slice(host.position());
        flags.extend_from_slice(host.flags());

        // `records` borrows `batch.data` in every sense but the type system's,
        // so it must go first. Declared after `batch`, so it does.
        drop(records);
        reader.carry_from(tail).expect("carry");
    }

    (refs, positions, flags)
}

#[test]
fn a_whole_bam_decodes_to_device_columns() {
    let Some((codec, decoder)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture(name);
        let expected = whole_file(&raw);

        for blocks in [1usize, 4, 256] {
            let actual = drive(&raw, &codec, &decoder, blocks);
            assert_eq!(
                actual.1.len(),
                expected.1.len(),
                "{name} at {blocks} block(s) per batch: record count"
            );
            assert_eq!(
                actual, expected,
                "{name} at {blocks} block(s) per batch: columns"
            );
        }
    }
}

#[test]
fn batching_does_not_change_the_answer() {
    // The seam most likely to break: a record straddling a batch edge is
    // decoded once, in one piece, whatever the batch size. Fixed batching
    // would hide it; varying it is the test.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let raw = fixture("ont_ultralong.bam");
    let baseline = drive(&raw, &codec, &decoder, 256);
    assert!(!baseline.1.is_empty(), "fixture must have records");

    for blocks in [1usize, 2, 3, 5, 8] {
        assert_eq!(
            drive(&raw, &codec, &decoder, blocks),
            baseline,
            "{blocks} block(s) per batch disagreed with a single-batch read"
        );
    }
}

#[test]
fn every_record_is_emitted_exactly_once() {
    // The carry re-inflates blocks, so a reader that ignored `start` would
    // emit the leading records of a carried block twice. Positions in this
    // fixture are unique and ascending, which makes that visible.
    let Some((codec, decoder)) = parts() else {
        return;
    };

    let raw = fixture("htslib_multiblock.bam");
    let (_, positions, _) = drive(&raw, &codec, &decoder, 1);

    assert_eq!(positions.len(), 4000, "samtools view -c reports 4000");
    assert!(
        positions.windows(2).all(|w| w[0] < w[1]),
        "positions must stay strictly ascending; a repeat means a carried \
         block was re-emitted"
    );
}
