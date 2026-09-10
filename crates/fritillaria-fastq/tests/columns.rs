//! The FASTQ columnar path against real bgzipped fixtures.
//!
//! The oracle here is stronger than the one BAM and BCF get. Those compare our
//! columns against our own record view, and reach ground truth only through a
//! separate `samtools`/`bcftools` test. Here the vendored
//! [`io::Reader`](fritillaria_fastq::io::Reader) is *independent* code — noodles'
//! parser, which we did not write and did not modify — so every assertion below
//! is a genuine differential test on its own.
//!
//! The fixtures come from the committed BAMs via `samtools fastq`, so the reads
//! are real and the container is htslib's. See `testdata/README.md`.

use std::io::BufReader;
use std::path::PathBuf;

use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};
use fritillaria_fastq::columnar::{Proof, RecordBatch, scan_records, scan_records_speculative};

/// Every bgzipped fixture, spanning the three read types that behave
/// differently: one huge record, a few medium ones, and thousands of tiny ones.
const FIXTURES: &[&str] = &[
    "ont_ultralong.fastq.gz",
    "pacbio_hifi.fastq.gz",
    "illumina.fastq.gz",
];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// Inflates a whole fixture through the BGZF container.
fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("must inflate");
    out
}

/// Ground truth: the vendored reader, over the same inflated bytes.
fn vendored_records(data: &[u8]) -> Vec<fritillaria_fastq::Record> {
    let mut reader = fritillaria_fastq::io::Reader::new(BufReader::new(data));
    let mut out = Vec::new();
    let mut record = fritillaria_fastq::Record::default();
    while reader.read_record(&mut record).expect("vendored read") != 0 {
        out.push(record.clone());
    }
    out
}

#[test]
fn the_columnar_scan_finds_what_the_vendored_reader_finds() {
    for name in FIXTURES {
        let host = inflate(name);
        let data = host.data();

        let expected = vendored_records(data);
        assert!(
            !expected.is_empty(),
            "{name}: vendored reader found nothing"
        );

        let mut batch = RecordBatch::new();
        let tail = batch.decode(data, 0).expect("decode");

        assert_eq!(batch.len(), expected.len(), "{name}: record count");
        assert_eq!(tail, data.len(), "{name}: a whole file must end cleanly");

        for (i, want) in expected.iter().enumerate() {
            let got = batch.record(data, i).expect("record view");
            let at = format!("{name} record {i}");
            assert_eq!(got.name(), want.name().as_ref() as &[u8], "{at}: name");
            assert_eq!(got.sequence(), want.sequence(), "{at}: sequence");
            assert_eq!(
                got.quality_scores(),
                want.quality_scores(),
                "{at}: quality scores"
            );
            assert_eq!(
                got.description(),
                want.description().as_ref() as &[u8],
                "{at}: description"
            );
        }
    }
}

#[test]
fn sequence_and_quality_are_always_the_same_length() {
    // The invariant the validator leans on, checked against real data rather
    // than assumed from the spec.
    for name in FIXTURES {
        let host = inflate(name);
        let data = host.data();
        let mut batch = RecordBatch::new();
        batch.decode(data, 0).unwrap();

        for (i, record) in batch.records(data).enumerate() {
            assert_eq!(
                record.sequence().len(),
                record.quality_scores().len(),
                "{name} record {i}"
            );
            assert_eq!(record.len(), batch.bounds()[i].sequence_len as usize);
        }
    }
}

#[test]
fn the_speculative_scan_agrees_with_the_serial_one_and_takes_the_fast_path() {
    for name in FIXTURES {
        let host = inflate(name);
        let data = host.data();

        let (expected, expected_tail) = scan_records(data, 0).unwrap();
        let scan = scan_records_speculative(data, 0).unwrap();

        assert_eq!(scan.offsets, expected, "{name}: offsets");
        assert_eq!(scan.tail, expected_tail, "{name}: tail");
        assert_eq!(
            scan.proof,
            Proof::Tiled,
            "{name}: real data must tile, or the kernel buys nothing"
        );
    }
}

#[test]
fn the_sieve_admits_exactly_the_records_on_real_data() {
    // The measurement docs/fastq-boundaries.md is built on, as an assertion.
    // BCF's sieve needs a second stage because it lets ~12.7 candidates per
    // block through; FASTQ's admits nothing but true record starts, and if that
    // ever stops being true the kernel's design should be revisited.
    for name in FIXTURES {
        let host = inflate(name);
        let data = host.data();
        let scan = scan_records_speculative(data, 0).unwrap();

        assert_eq!(
            scan.survivors,
            scan.offsets.len(),
            "{name}: sieve admitted a non-record"
        );
        assert!(
            scan.candidates > scan.survivors,
            "{name}: candidates ({}) must outnumber records ({}), or the sieve \
             is not being exercised",
            scan.candidates,
            scan.survivors
        );
    }
}

#[test]
fn the_illumina_fixture_still_carries_the_decoys_it_exists_for() {
    // A fixture that is supposed to exercise a case should assert that it does.
    // If this file is ever regenerated from different reads, the zero-false-
    // positive measurement stops meaning anything and this fails loudly.
    let host = inflate("illumina.fastq.gz");
    let data = host.data();
    let mut batch = RecordBatch::new();
    batch.decode(data, 0).unwrap();

    let mut at_line_start = 0usize;
    let mut at_anywhere = 0usize;
    for record in batch.records(data) {
        let quality = record.quality_scores();
        if quality.first() == Some(&b'@') {
            at_line_start += 1;
        }
        at_anywhere += quality.iter().filter(|&&b| b == b'@').count();
    }

    assert_eq!(
        at_line_start, 83,
        "quality lines starting with '@' — the decoy the '+' check rejects"
    );
    assert_eq!(at_anywhere, 9660, "'@' bytes inside quality lines");
}

#[test]
fn no_sequence_line_starts_with_a_plus() {
    // Why the validator works at all. If real data ever violated this the '+'
    // check would stop being decisive and the sieve would need a second stage.
    for name in FIXTURES {
        let host = inflate(name);
        let data = host.data();
        let mut batch = RecordBatch::new();
        batch.decode(data, 0).unwrap();

        for (i, record) in batch.records(data).enumerate() {
            assert_ne!(
                record.sequence().first(),
                Some(&b'+'),
                "{name} record {i}: a sequence starting with '+' breaks the sieve"
            );
        }
    }
}

#[test]
fn the_ont_fixture_has_a_record_larger_than_a_bgzf_block() {
    // The seam property, asserted rather than assumed from the filename — the
    // mistake `htslib_multiblock.bam` made for months.
    let host = inflate("ont_ultralong.fastq.gz");
    let mut batch = RecordBatch::new();
    batch.decode(host.data(), 0).unwrap();

    let longest = batch
        .bounds()
        .iter()
        .map(|b| b.record_end)
        .max()
        .expect("records");
    assert!(
        longest as usize > 65536,
        "longest record is {longest} bytes, which fits in one BGZF block — \
         this fixture is supposed to span several"
    );
}

#[test]
fn batching_at_arbitrary_offsets_does_not_change_the_answer() {
    // The driver contract, without a GPU: scan a prefix, carry the tail, scan
    // on. Record lengths are position-independent, so they compare across the
    // seam.
    let host = inflate("pacbio_hifi.fastq.gz");
    let data = host.data();

    let mut whole = RecordBatch::new();
    whole.decode(data, 0).unwrap();
    let expected: Vec<u32> = whole.bounds().iter().map(|b| b.record_end).collect();

    for window in [4096usize, 65536, 200_000] {
        let mut lengths: Vec<u32> = Vec::new();
        let mut start = 0usize;
        let mut end = window.min(data.len());

        loop {
            let mut batch = RecordBatch::new();
            let tail = batch.decode(&data[..end], start).expect("decode");
            lengths.extend(batch.bounds().iter().map(|b| b.record_end));
            if end == data.len() {
                assert_eq!(tail, data.len(), "the final batch must end cleanly");
                break;
            }
            start = tail;
            end = (end + window).min(data.len());
        }

        assert_eq!(lengths, expected, "window {window}: record lengths");
    }
}
