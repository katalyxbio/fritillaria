//! FASTA compaction on device, diffed against the CPU reference.
//!
//! The oracle chain reaches htslib: `fritillaria-fasta/tests/faidx.rs` checks
//! the host path against a `.fai` that `samtools faidx` wrote and against
//! the vendored reader, and this checks the device against that host path.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_core::{DeviceBlockCodec, DeviceInflateBatch};
use fritillaria_cuda::{CudaCodec, FastaCompactor};
use fritillaria_fasta::columnar::RecordBatch;

fn parts() -> Option<(CudaCodec, FastaCompactor)> {
    match (CudaCodec::new(), FastaCompactor::new()) {
        (Ok(codec), Ok(compactor)) => Some((codec, compactor)),
        (Err(err), _) | (_, Err(err)) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

fn fixture() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join("controls.fa");
    std::fs::read(path).expect("fixture missing; see testdata/README.md")
}

/// Wraps a plain buffer as a device batch.
///
/// FASTA is not usually BGZF — a reference is normally a plain `.fa` — so this
/// stages the bytes directly rather than going through the container. The
/// compactor takes an inflated batch either way.
fn staged(codec: &CudaCodec, bytes: &[u8]) -> DeviceInflateBatch {
    let mut writer = fritillaria_bgzf::BgzfWriter::new(Vec::new());
    writer.write_data(bytes).expect("bgzip the fixture");
    let raw = writer.finish().expect("finish");
    let spans = fritillaria_bgzf::discover_blocks(&raw, 0).expect("blocks");
    let mut batch = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut batch)
        .expect("device inflate");
    batch
}

#[test]
fn the_device_reference_matches_the_host_compaction() {
    let Some((codec, compactor)) = parts() else {
        return;
    };
    let bytes = fixture();

    let mut host = RecordBatch::new();
    host.decode(&bytes, 0, true).expect("host decode");
    let expected = host.compact(&bytes).expect("host compact");

    let batch = staged(&codec, &bytes);
    let reference = compactor.compact(&batch).expect("device compact");

    assert_eq!(reference.len(), host.len(), "contig count");
    assert_eq!(reference.total_bases(), host.total_bases(), "total bases");

    let got = reference.sequence_to_host().expect("download");
    assert_eq!(got.len(), expected.len(), "compacted length");
    assert_eq!(got, expected, "compacted bases");
    assert!(!got.contains(&b'\n'), "compaction must remove newlines");
}

#[test]
fn the_contig_index_survives_the_round_trip() {
    let Some((codec, compactor)) = parts() else {
        return;
    };
    let bytes = fixture();

    let mut host = RecordBatch::new();
    host.decode(&bytes, 0, true).expect("host decode");

    let batch = staged(&codec, &bytes);
    let reference = compactor.compact(&batch).expect("device compact");
    let index = reference.index_to_host().expect("download index");

    assert_eq!(index.sequence_offsets, host.sequence_offsets(), "offsets");
    assert_eq!(index.sequence_lengths, host.sequence_lengths(), "lengths");
    assert_eq!(
        index.record_offsets,
        host.record_offsets()
            .iter()
            .map(|&o| o as u64)
            .collect::<Vec<_>>(),
        "record offsets"
    );
}

#[test]
fn each_contig_slices_back_out_of_the_compacted_buffer() {
    // The contract a consumer actually relies on: offset plus length must cut
    // the reference back into the original sequences. An index that is
    // self-consistent but off by one contig would pass the equality above.
    let Some((codec, compactor)) = parts() else {
        return;
    };
    let bytes = fixture();

    let mut host = RecordBatch::new();
    host.decode(&bytes, 0, true).expect("host decode");

    let batch = staged(&codec, &bytes);
    let reference = compactor.compact(&batch).expect("device compact");
    let got = reference.sequence_to_host().expect("download");
    let index = reference.index_to_host().expect("download index");

    for i in 0..host.len() {
        let record = host.record(&bytes, i).expect("record view");
        let want = record.to_sequence().expect("host sequence");
        let at = index.sequence_offsets[i] as usize;
        let len = index.sequence_lengths[i] as usize;
        assert_eq!(&got[at..at + len], &want[..], "contig {i}");
    }
}

#[test]
fn an_empty_input_yields_an_empty_reference() {
    let Some((codec, compactor)) = parts() else {
        return;
    };
    let batch = staged(&codec, b"");
    let reference = compactor.compact(&batch).expect("compact");
    assert!(reference.is_empty());
    assert_eq!(reference.total_bases(), 0);
    assert!(reference.sequence_to_host().unwrap().is_empty());
}
