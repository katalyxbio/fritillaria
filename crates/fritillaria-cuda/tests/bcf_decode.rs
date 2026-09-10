//! BCF site cores decoded into device columns, diffed against the CPU path.
//!
//! The oracle is `fritillaria_bcf::columnar::RecordBatch::decode`, which
//! `fritillaria-bcf/tests/columns.rs` checks against the record view, which
//! `tests/bcftools.rs` checks against `bcftools` itself. So a pass here chains
//! back to htslib rather than to our own opinion of the format.
//!
//! Two things are compared, and the second is where the arithmetic actually
//! lives:
//!
//! 1. **The eight fixed-width columns.** Reads at known offsets; cheap to get
//!    right, and cheap to be wrong about in a way that looks plausible — every
//!    one of `position`, `reference_span` and `chromosome_id` is an `i32` at a
//!    4-byte stride, so a mis-set offset yields numbers, not a crash.
//!
//! 2. **The bounds.** ID, each allele and FILTER are typed values whose lengths
//!    live in their own descriptors, so the kernel has to *walk* them. That walk
//!    is the only data-dependent loop in the decode and the only place a thread
//!    can desynchronise.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bcf::columnar::{RecordBatch, parse_header};
use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, discover_blocks};
use fritillaria_core::{BlockCodec, DeviceBlockCodec, DeviceInflateBatch, InflateBatch};
use fritillaria_cuda::{BcfScanner, CudaCodec};

const FIXTURES: &[&str] = &["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"];

fn parts() -> Option<(CudaCodec, BcfScanner)> {
    match (CudaCodec::new(), BcfScanner::new()) {
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
    let spans = discover_blocks(raw, 0).expect("bcftools BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .expect("must inflate");
    out
}

/// The header's sample and contig counts, which the sieve keys on.
fn header_of(raw: &[u8]) -> (usize, u32, u32) {
    let host = inflate_host(raw);
    let header = parse_header(host.data()).expect("bad BCF header");
    (
        header.records_start,
        u32::try_from(header.sample_count()).unwrap(),
        u32::try_from(header.dictionary.contigs.len()).unwrap(),
    )
}

/// Asserts two batches agree column for column, row for row.
fn assert_same(got: &RecordBatch, expected: &RecordBatch, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: record count");

    assert_eq!(
        got.chromosome_id(),
        expected.chromosome_id(),
        "{context}: CHROM"
    );
    assert_eq!(got.position(), expected.position(), "{context}: POS");
    assert_eq!(
        got.reference_span(),
        expected.reference_span(),
        "{context}: rlen"
    );
    // Bits, not floats: missing QUAL is a NaN and NaN != NaN, so comparing
    // decoded values would pass vacuously on every missing entry.
    assert_eq!(
        got.quality_bits(),
        expected.quality_bits(),
        "{context}: QUAL"
    );
    assert_eq!(got.info_count(), expected.info_count(), "{context}: n_info");
    assert_eq!(
        got.allele_count(),
        expected.allele_count(),
        "{context}: n_allele"
    );
    assert_eq!(
        got.sample_count(),
        expected.sample_count(),
        "{context}: n_sample"
    );
    assert_eq!(
        got.format_count(),
        expected.format_count(),
        "{context}: n_fmt"
    );
    assert_eq!(got.bounds(), expected.bounds(), "{context}: field bounds");
}

#[test]
fn device_columns_match_the_host_reference() {
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture(name);
        let (records_start, samples, contigs) = header_of(&raw);

        // Whole file as one batch, so the answer is directly comparable to the
        // host oracle over the same bytes.
        let spans = discover_blocks(&raw, 0).unwrap();
        let mut batch = DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut batch)
            .expect("device inflate");

        let host = inflate_host(&raw);
        let mut expected = RecordBatch::new();
        let expected_tail = expected.decode(host.data(), records_start).unwrap();

        let device = scanner
            .decode(&batch, records_start, samples, contigs)
            .expect("device decode");

        assert_eq!(device.tail(), expected_tail, "{name}: tail");
        assert_same(&device.to_host().unwrap(), &expected, name);
    }
}

#[test]
fn device_columns_match_across_batch_seams() {
    // A BCF batch almost always ends mid-record, because bcftools packs BGZF
    // blocks full. Driving one block per batch puts a seam in the hardest place
    // available.
    //
    // Batch-local offsets are not comparable to whole-file ones — the reader
    // hands over a window and re-inflates carried blocks at its front — so the
    // comparison is on the values that are position-independent. `position` and
    // the record *lengths* would both catch a boundary landing one byte off.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let mut seams = 0usize;
    for name in FIXTURES {
        let raw = fixture(name);
        let (records_start, samples, contigs) = header_of(&raw);
        let host = inflate_host(&raw);

        let mut whole = RecordBatch::new();
        whole.decode(host.data(), records_start).unwrap();
        let expected_lengths: Vec<u32> = whole.bounds().iter().map(|b| b.record_end).collect();

        for blocks_per_batch in [1usize, 2] {
            let mut reader =
                DeviceBgzfReader::new(&raw[..], &codec).with_blocks_per_batch(blocks_per_batch);
            let mut positions: Vec<i32> = Vec::new();
            let mut lengths: Vec<u32> = Vec::new();
            let mut batches = 0usize;
            let mut header_parsed = false;

            while let Some(batch) = reader.next_batch().expect("batch") {
                let begin = if header_parsed {
                    batch.start
                } else {
                    let staged = batch.data.to_host().expect("header download");
                    let Ok(header) = parse_header(staged.data()) else {
                        reader.carry_from(0).expect("carry");
                        continue;
                    };
                    header_parsed = true;
                    header.records_start
                };

                let decoded = scanner
                    .decode(&batch.data, begin, samples, contigs)
                    .expect("device decode");
                batches += 1;

                let columns = decoded.to_host().expect("download columns");
                positions.extend_from_slice(columns.position());
                lengths.extend(columns.bounds().iter().map(|b| b.record_end));

                let tail = decoded.tail();
                drop(batch);
                reader.carry_from(tail).expect("carry");
            }

            let context = format!("{name} at {blocks_per_batch} blocks/batch");
            assert_eq!(positions, whole.position(), "{context}: POS across seams");
            assert_eq!(
                lengths, expected_lengths,
                "{context}: record lengths across seams"
            );
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
    // A batch holding only a partial record is normal for BCF, and must not be
    // an error — the driver carries it forward and tries again.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let raw = fixture("giab_hg002.bcf");
    let (records_start, samples, contigs) = header_of(&raw);
    let spans = discover_blocks(&raw, 0).unwrap();
    let mut batch = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&raw, &spans, &mut batch)
        .expect("device inflate");

    // Starting past the end: no record can begin here.
    let len = batch
        .data()
        .map_or(0, fritillaria_core::DeviceBuffer::byte_len);
    let decoded = scanner
        .decode(&batch, len, samples, contigs)
        .expect("decode past the end must succeed, not error");

    assert!(decoded.is_empty());
    assert_eq!(decoded.tail(), len);
    assert!(decoded.to_host().unwrap().is_empty());
    let _ = records_start;
}
