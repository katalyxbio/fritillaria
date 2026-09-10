//! The text scan on device, diffed against the CPU reference.
//!
//! The oracle chain reaches four independent parsers: this compares against
//! `fritillaria_text::columnar::RecordBatch`, and
//! `fritillaria-text/tests/real_files.rs` compares *that* against the vendored
//! sam, vcf, bed, gff and gtf readers — noodles' code we neither wrote nor
//! modified.
//!
//! All five fixtures run through the same kernel, which is the point: their
//! record framing is identical and only the header marker differs.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_core::{DeviceBlockCodec, DeviceInflateBatch};
use fritillaria_cuda::{CudaCodec, TextScanner};
use fritillaria_text::columnar::{Dialect, RecordBatch};

/// Every text fixture, with the dialect its format uses.
fn fixtures() -> Vec<(&'static str, Dialect)> {
    vec![
        ("reads.sam", Dialect::SAM),
        ("calls.vcf", Dialect::VCF),
        ("lambda.gff3", Dialect::GFF),
        ("lambda.gtf", Dialect::GFF),
        ("cytoband.bed", Dialect::BED),
    ]
}

fn parts() -> Option<(CudaCodec, TextScanner)> {
    match (CudaCodec::new(), TextScanner::new()) {
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

/// Stages a plain buffer on the device through the BGZF container.
///
/// These formats are usually plain text; the container is just the transport
/// this workspace already has for getting bytes onto a device.
fn staged(codec: &CudaCodec, bytes: &[u8]) -> DeviceInflateBatch {
    let mut writer = fritillaria_bgzf::BgzfWriter::new(Vec::new());
    writer.write_data(bytes).expect("bgzip");
    let raw = writer.finish().expect("finish");
    let spans = fritillaria_bgzf::discover_blocks(&raw, 0).expect("blocks");
    let mut batch = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut batch)
        .expect("device inflate");
    batch
}

#[test]
fn device_columns_match_the_host_reference() {
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for (name, dialect) in fixtures() {
        let bytes = fixture(name);

        let mut expected = RecordBatch::new();
        expected.decode(&bytes, 0, dialect).expect("host decode");

        let batch = staged(&codec, &bytes);
        let device = scanner.scan(&batch, 0, dialect).expect("device scan");
        let got = device.to_host().expect("download");

        assert_eq!(got.len(), expected.len(), "{name}: record count");
        assert_eq!(
            device.header_count(),
            expected.header_offsets().len(),
            "{name}: header count"
        );
        assert_eq!(
            got.record_offsets(),
            expected.record_offsets(),
            "{name}: record offsets"
        );
        assert_eq!(
            got.record_ends(),
            expected.record_ends(),
            "{name}: record ends"
        );
        assert_eq!(
            got.header_offsets(),
            expected.header_offsets(),
            "{name}: header offsets"
        );
        assert_eq!(
            got.fields().starts(),
            expected.fields().starts(),
            "{name}: field table boundaries"
        );
        assert_eq!(
            got.fields().flat_tabs(),
            expected.fields().flat_tabs(),
            "{name}: tab offsets"
        );
    }
}

#[test]
fn every_field_slices_the_same_bytes_on_both_paths() {
    // Offsets agreeing is necessary but not sufficient: this checks that
    // slicing at them yields the same field contents, which is what a consumer
    // actually reads.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for (name, dialect) in fixtures() {
        let bytes = fixture(name);
        let mut expected = RecordBatch::new();
        expected.decode(&bytes, 0, dialect).expect("host decode");

        let batch = staged(&codec, &bytes);
        let got = scanner
            .scan(&batch, 0, dialect)
            .expect("device scan")
            .to_host()
            .expect("download");

        for i in 0..expected.len() {
            let mine: Vec<_> = got.record_fields(&bytes, i).collect();
            let want: Vec<_> = expected.record_fields(&bytes, i).collect();
            assert_eq!(mine, want, "{name} record {i}: fields");
        }
    }
}

#[test]
fn the_ragged_field_table_survives_the_round_trip() {
    // BED is the ragged case by design, and cytoband.bed is uniform at 5, so
    // the raggedness worth checking here is *between formats*: one kernel
    // produced 12-field SAM rows and 5-field BED rows, and the boundary table
    // is what keeps them apart.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let mut seen = Vec::new();
    for (name, dialect) in fixtures() {
        let bytes = fixture(name);
        let batch = staged(&codec, &bytes);
        let got = scanner
            .scan(&batch, 0, dialect)
            .expect("device scan")
            .to_host()
            .expect("download");

        let counts = got.field_counts();
        assert!(!counts.is_empty(), "{name}: no records");
        let first = counts[0];
        assert!(
            counts.iter().all(|&c| c == first),
            "{name}: this fixture should be uniform"
        );
        seen.push(first);
    }

    assert!(
        seen.iter().collect::<std::collections::HashSet<_>>().len() > 1,
        "the fixtures should span several field counts, got {seen:?}"
    );
}

#[test]
fn an_empty_input_scans_to_nothing() {
    let Some((codec, scanner)) = parts() else {
        return;
    };
    let batch = staged(&codec, b"");
    let got = scanner.scan(&batch, 0, Dialect::SAM).expect("scan");
    assert!(got.is_empty());
    assert_eq!(got.header_count(), 0);
    assert!(got.to_host().unwrap().is_empty());
}

#[test]
fn a_header_only_input_yields_headers_and_no_records() {
    // The case that would divide by zero or index an empty table if the field
    // machinery assumed at least one record.
    let Some((codec, scanner)) = parts() else {
        return;
    };
    let batch = staged(&codec, b"@HD\tVN:1.6\n@SQ\tSN:c0\tLN:100\n");
    let got = scanner.scan(&batch, 0, Dialect::SAM).expect("scan");
    assert_eq!(got.len(), 0, "no records");
    assert_eq!(got.header_count(), 2, "two header lines");
    assert!(got.to_host().unwrap().is_empty());
}
