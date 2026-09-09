//! FASTQ boundaries and columns on device, diffed against the CPU reference.
//!
//! The oracle chain is the strongest of the three formats: our columns are
//! compared against `fritillaria_fastq::columnar::RecordBatch`, and
//! `fritillaria-fastq/tests/columns.rs` compares *that* against noodles' own
//! parser — independent code we neither wrote nor modified.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, discover_blocks};
use fritillaria_core::{BlockCodec, DeviceBlockCodec, DeviceInflateBatch, InflateBatch};
use fritillaria_cuda::{CudaCodec, FastqScanner};
use fritillaria_fastq::columnar::{Proof, RecordBatch};

/// One huge record, a few medium ones, and thousands of tiny ones — the three
/// shapes that stress different parts of the sieve.
const FIXTURES: &[&str] = &[
    "ont_ultralong.fastq.gz",
    "pacbio_hifi.fastq.gz",
    "illumina.fastq.gz",
];

fn parts() -> Option<(CudaCodec, FastqScanner)> {
    match (CudaCodec::new(), FastqScanner::new()) {
        (Ok(codec), Ok(scanner)) => Some((codec, scanner)),
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

fn inflate_host(raw: &[u8]) -> InflateBatch {
    let spans = discover_blocks(raw, 0).expect("htslib BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .expect("must inflate");
    out
}

fn assert_same(got: &RecordBatch, expected: &RecordBatch, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: record count");
    assert_eq!(
        got.record_offsets(),
        expected.record_offsets(),
        "{context}: record offsets"
    );
    assert_eq!(got.bounds(), expected.bounds(), "{context}: line bounds");
    assert_eq!(
        got.sequence_lengths(),
        expected.sequence_lengths(),
        "{context}: read lengths"
    );
}

#[test]
fn device_columns_match_the_host_reference() {
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture(name);
        let spans = discover_blocks(&raw, 0).unwrap();
        let mut batch = DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut batch)
            .expect("device inflate");

        let host = inflate_host(&raw);
        let mut expected = RecordBatch::new();
        let expected_tail = expected.decode(host.data(), 0).unwrap();

        let device = scanner.decode(&batch, 0).expect("device decode");
        assert_eq!(device.tail(), expected_tail, "{name}: tail");
        assert_same(&device.to_host().unwrap(), &expected, name);
    }
}

#[test]
fn the_sieve_takes_the_fast_path_on_real_data() {
    // A silent fall back to the single-thread walk is correct and performs
    // exactly like the design this replaces, so "it worked" is not the same as
    // "it worked the fast way".
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture(name);
        let spans = discover_blocks(&raw, 0).unwrap();
        let mut batch = DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut batch)
            .expect("device inflate");

        let scan = scanner.scan(&batch, 0).expect("device scan");
        assert_eq!(
            scan.proof,
            Proof::Tiled,
            "{name}: the tiling must hold, or the kernel bought nothing"
        );

        let host = inflate_host(&raw);
        let mut expected = RecordBatch::new();
        expected.decode(host.data(), 0).unwrap();
        assert_eq!(
            scan.survivors,
            expected.len(),
            "{name}: the sieve must admit exactly the records"
        );
    }
}

#[test]
fn device_columns_match_across_batch_seams() {
    // Driving one block per batch puts a seam in the hardest place available,
    // and the ONT fixture has a record spanning many blocks, so the reader has
    // to grow its window before that record fits at all.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let mut seams = 0usize;
    for name in FIXTURES {
        let raw = fixture(name);
        let host = inflate_host(&raw);

        let mut whole = RecordBatch::new();
        whole.decode(host.data(), 0).unwrap();
        let expected_lengths: Vec<u32> = whole.sequence_lengths();

        for blocks_per_batch in [1usize, 4] {
            let mut reader =
                DeviceBgzfReader::new(&raw[..], &codec).with_blocks_per_batch(blocks_per_batch);
            let mut lengths: Vec<u32> = Vec::new();
            let mut batches = 0usize;

            while let Some(batch) = reader.next_batch().expect("batch") {
                let decoded = scanner.decode(&batch.data, batch.start).expect("decode");
                batches += 1;

                let columns = decoded.to_host().expect("download columns");
                lengths.extend(columns.sequence_lengths());

                let tail = decoded.tail();
                drop(batch);
                reader.carry_from(tail).expect("carry");
            }

            let context = format!("{name} at {blocks_per_batch} blocks/batch");
            assert_eq!(lengths, expected_lengths, "{context}: read lengths");
            seams += batches.saturating_sub(1);
        }
    }

    assert!(
        seams >= 3,
        "no batch seam was crossed in any configuration, so nothing above \
         tested the seam behaviour it claims to ({seams} seams)"
    );
}

#[test]
fn an_empty_batch_decodes_to_an_empty_column_set() {
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let raw = fixture("pacbio_hifi.fastq.gz");
    let spans = discover_blocks(&raw, 0).unwrap();
    let mut batch = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut batch)
        .expect("device inflate");

    let len = batch
        .data()
        .map_or(0, fritillaria_core::DeviceBuffer::byte_len);
    let decoded = scanner
        .decode(&batch, len)
        .expect("decoding past the end must succeed, not error");

    assert!(decoded.is_empty());
    assert_eq!(decoded.tail(), len);
    assert!(decoded.to_host().unwrap().is_empty());
}
