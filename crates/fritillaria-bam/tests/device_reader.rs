//! Driving `DeviceBgzfReader` over real BAMs, one batch at a time.
//!
//! `device_read.rs`'s own tests use a synthetic length-prefixed format, because
//! the reader knows nothing about BAM and neither should they. This is the
//! other half: the same reader driven by an actual BAM consumer, over the block
//! layouts htslib really produced.
//!
//! It runs on [`HostDeviceCodec`], so it needs no GPU. Only the transfer is
//! fictional — the batching, the carry, and the record decode are the real
//! ones, and `fritillaria-cuda/tests/bam_reader.rs` runs the identical loop
//! with a CUDA codec and the on-device decoder.
//!
//! The fixture that matters is `ont_ultralong.bam`: its 254 KB record is larger
//! than several batches at small block counts, so the reader has to carry
//! blocks forward *and* grow the window until the record fits. At one block per
//! batch nothing else in `testdata/` exercises that.

use std::path::PathBuf;

use fritillaria_bam::columnar::{RecordBatch, header::parse_header};
use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, HostDeviceCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

const FIXTURES: &[&str] = &[
    "htslib.bam",
    "htslib_multiblock.bam",
    "pacbio_hifi.bam",
    "ont_ultralong.bam",
];

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    std::fs::read(path).expect("fixture missing; see testdata/README.md")
}

/// Ground truth: the whole file inflated at once and decoded in one pass.
fn whole_file(raw: &[u8]) -> Vec<(Vec<u8>, i32)> {
    let spans = discover_blocks(raw, 0).expect("htslib BGZF must parse");
    let mut inflated = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut inflated)
        .expect("must inflate");
    let header = parse_header(inflated.data()).expect("header must parse");

    let mut columns = RecordBatch::new();
    columns
        .decode(inflated.data(), header.records_start)
        .expect("must decode");
    columns
        .records(inflated.data())
        .zip(columns.position())
        .map(|(r, &pos)| (r.name().to_vec(), pos))
        .collect()
}

/// The loop a BAM consumer has to write, and the one `DeviceBamReader` will
/// encapsulate: decode what is buffered, report where it stopped, repeat.
///
/// The header is handled the same way a long record is — if it does not fit in
/// the first batch, carry everything and let the window grow.
fn drive(raw: &[u8], blocks_per_batch: usize) -> Vec<(Vec<u8>, i32)> {
    let mut reader =
        DeviceBgzfReader::new(raw, HostDeviceCodec::new()).with_blocks_per_batch(blocks_per_batch);
    let mut out = Vec::new();
    let mut header_parsed = false;

    while let Some(batch) = reader.next_batch().expect("batch") {
        let host = batch.data.to_host().expect("download");

        let begin = if header_parsed {
            batch.start
        } else {
            let Ok(header) = parse_header(host.data()) else {
                // The header itself spans more than this batch. Same fix as an
                // over-long record: keep everything, read more.
                reader.carry_from(0).expect("carry");
                continue;
            };
            header_parsed = true;
            header.records_start
        };

        let mut columns = RecordBatch::new();
        let tail = columns.decode(host.data(), begin).expect("decode");
        out.extend(
            columns
                .records(host.data())
                .zip(columns.position())
                .map(|(r, &pos)| (r.name().to_vec(), pos)),
        );
        reader.carry_from(tail).expect("carry");
    }

    out
}

#[test]
fn recovers_every_record_of_every_fixture_at_every_batch_size() {
    for name in FIXTURES {
        let raw = fixture(name);
        let expected = whole_file(&raw);
        assert!(!expected.is_empty(), "{name} has no records");

        for blocks in [1usize, 2, 3, 7, 256] {
            let actual = drive(&raw, blocks);
            assert_eq!(
                actual.len(),
                expected.len(),
                "{name} at {blocks} block(s) per batch: record count"
            );
            assert_eq!(
                actual, expected,
                "{name} at {blocks} block(s) per batch: names and positions"
            );
        }
    }
}

#[test]
fn the_ont_record_forces_the_window_to_grow() {
    // Asserted rather than assumed. The 254 KB record cannot fit in a
    // one-block batch, so the reader must carry and grow; if it ever could,
    // this test would be passing over the easy path instead.
    let raw = fixture("ont_ultralong.bam");
    let spans = discover_blocks(&raw, 0).unwrap();
    let largest_block = spans.iter().map(|s| s.isize as usize).max().unwrap();

    let mut inflated = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut inflated)
        .unwrap();
    let header = parse_header(inflated.data()).unwrap();
    let mut columns = RecordBatch::new();
    columns
        .decode(inflated.data(), header.records_start)
        .unwrap();

    let offsets = columns.record_offsets();
    let largest_record = offsets
        .iter()
        .enumerate()
        .map(|(i, &o)| offsets.get(i + 1).copied().unwrap_or(inflated.data().len()) - o)
        .max()
        .unwrap();

    assert!(
        largest_record > largest_block,
        "the fixture's largest record ({largest_record} B) must exceed its largest \
         block ({largest_block} B) or the growth path is never taken"
    );
    assert_eq!(drive(&raw, 1), whole_file(&raw));
}

#[test]
fn a_batch_size_of_one_still_sees_every_record_exactly_once() {
    // Duplication is the failure mode the carry can produce: re-inflating a
    // block means re-seeing records that were already emitted unless `start`
    // is honoured exactly.
    let raw = fixture("htslib_multiblock.bam");
    let names: Vec<Vec<u8>> = drive(&raw, 1).into_iter().map(|(n, _)| n).collect();
    let unique: std::collections::HashSet<_> = names.iter().collect();

    assert_eq!(names.len(), 4000, "samtools view -c reports 4000 records");
    assert_eq!(
        unique.len(),
        names.len(),
        "a record was emitted twice: the carry re-read blocks without honouring `start`"
    );
}
