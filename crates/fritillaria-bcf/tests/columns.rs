//! The BCF columnar decode, checked against real bcftools-written files.
//!
//! Two separate claims are tested here, and the second is the one that could be
//! subtly wrong:
//!
//! 1. **The columns carry what the record view carries.** Cheap to get right,
//!    cheap to check, and it chains back to `bcftools` because
//!    `tests/bcftools.rs` validates the record view itself.
//!
//! 2. **The bounds actually delimit the fields they name.** A bounds table is
//!    the sort of thing that can be self-consistently wrong — monotonic,
//!    in-range, and pointing one typed value too far. So it is not checked for
//!    plausibility; it is checked by *slicing at the offsets and decoding what
//!    is there*, and requiring that to equal what the record view returns.
//!
//! These are also the oracle for `kernels/bcf_decode.cu`. The kernel is a
//! direct translation of `RecordBatch::decode`, so the two must agree on real
//! data before either is believed.

use std::path::PathBuf;

use fritillaria_bcf::columnar::{RecordBatch, SITE_CORE_SIZE, parse_header, typed};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

const FIXTURES: &[&str] = &["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("bcftools BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("must inflate");
    out
}

/// Decodes a whole fixture into columns.
fn columns(name: &str) -> (InflateBatch, RecordBatch) {
    let host = inflate(name);
    let start = parse_header(host.data()).expect("header").records_start;
    let mut batch = RecordBatch::new();
    batch.decode(host.data(), start).expect("decode");
    (host, batch)
}

#[test]
fn columns_stay_in_lockstep_across_every_fixture() {
    for name in FIXTURES {
        let (_, batch) = columns(name);
        let n = batch.len();
        assert!(n > 0, "{name}: fixture decoded to no records");

        for (label, len) in [
            ("chromosome_id", batch.chromosome_id().len()),
            ("position", batch.position().len()),
            ("reference_span", batch.reference_span().len()),
            ("quality_bits", batch.quality_bits().len()),
            ("info_count", batch.info_count().len()),
            ("allele_count", batch.allele_count().len()),
            ("sample_count", batch.sample_count().len()),
            ("format_count", batch.format_count().len()),
            ("bounds", batch.bounds().len()),
        ] {
            assert_eq!(len, n, "{name}: column {label} out of lockstep");
        }
    }
}

#[test]
fn every_column_matches_the_record_view() {
    for name in FIXTURES {
        let (host, batch) = columns(name);
        let buf = host.data();

        for i in 0..batch.len() {
            let record = batch.record(buf, i).expect("record view");
            let at = format!("{name} record {i}");

            assert_eq!(
                record.chromosome_id(),
                batch.chromosome_id()[i],
                "{at}: CHROM"
            );
            assert_eq!(record.position(), batch.position()[i], "{at}: POS");
            assert_eq!(
                record.reference_span(),
                batch.reference_span()[i],
                "{at}: rlen"
            );
            assert_eq!(
                record.quality_bits(),
                batch.quality_bits()[i],
                "{at}: QUAL bits"
            );
            // Against the *independently implemented* accessor, not against
            // quality_at — which reads this very column and so would agree with
            // it however wrong it is. A mutation pointing quality_bits at rlen
            // survived the whole suite until this line existed.
            assert_eq!(
                batch.quality_at(i),
                record.quality(),
                "{at}: decoded QUAL must match the record view"
            );
            assert_eq!(
                record.info_count() as u16,
                batch.info_count()[i],
                "{at}: n_info"
            );
            assert_eq!(
                record.allele_count() as u16,
                batch.allele_count()[i],
                "{at}: n_allele"
            );
            assert_eq!(
                record.sample_count() as u32,
                batch.sample_count()[i],
                "{at}: n_sample"
            );
            assert_eq!(
                record.format_count() as u8,
                batch.format_count()[i],
                "{at}: n_fmt"
            );
        }
    }
}

#[test]
fn quality_keeps_missing_and_nan_distinguishable() {
    // The reason the column is `u32` and not `f32`. Missing QUAL is the
    // signalling NaN 0x7F800001; a genuine NaN QUAL is legal BCF. Stored as
    // f32 both would be NaN and `==` would call neither equal to itself, so the
    // distinction would be gone in a way no test comparing floats would catch.
    for name in FIXTURES {
        let (host, batch) = columns(name);
        let buf = host.data();
        for i in 0..batch.len() {
            let bits = batch.quality_bits()[i];
            // Anchored to the record view, which reads QUAL's offset
            // independently. Comparing the column only against itself proves
            // nothing — see the note in every_column_matches_the_record_view.
            let record = batch.record(buf, i).expect("record view");
            assert_eq!(bits, record.quality_bits(), "{name} record {i}: QUAL bits");

            match batch.quality_at(i) {
                None => assert_eq!(
                    bits, 0x7F80_0001,
                    "{name} record {i}: missing must be exact"
                ),
                Some(q) => assert_eq!(
                    q.to_bits(),
                    bits,
                    "{name} record {i}: decoded QUAL must round-trip its bits"
                ),
            }
        }
    }
}

#[test]
fn bounds_are_ordered_and_inside_the_record() {
    for name in FIXTURES {
        let (_, batch) = columns(name);
        for (i, b) in batch.bounds().iter().enumerate() {
            let at = format!("{name} record {i}");
            let id_start = (SITE_CORE_SIZE + 8) as u32;
            assert!(b.alleles_start >= id_start, "{at}: alleles before ID");
            assert!(
                b.filters_start >= b.alleles_start,
                "{at}: FILTER before alleles"
            );
            assert!(b.info_start >= b.filters_start, "{at}: INFO before FILTER");
            assert!(
                b.genotypes_start >= b.info_start,
                "{at}: genotypes before INFO"
            );
            assert!(
                b.record_end >= b.genotypes_start,
                "{at}: record ends before its genotypes"
            );
        }
    }
}

#[test]
fn slicing_at_the_bounds_yields_the_fields_they_name() {
    // The real test of the bounds, and the reason the one above is not enough:
    // an offset table can be monotonic, in range, and still point one typed
    // value too far. Decoding *at* each offset and requiring the result to
    // equal the record view's is what makes it impossible to be plausibly
    // wrong.
    for name in FIXTURES {
        let (host, batch) = columns(name);
        let buf = host.data();

        for i in 0..batch.len() {
            let offset = batch.record_offsets()[i];
            let b = batch.bounds()[i];
            let record = batch.record(buf, i).expect("record view");
            let bytes = &buf[offset..offset + b.record_end as usize];
            let at = format!("{name} record {i}");

            // ID occupies SITE_CORE_SIZE + 8 .. alleles_start.
            let id = typed::read(bytes, SITE_CORE_SIZE + 8).expect("ID");
            assert_eq!(
                (SITE_CORE_SIZE + 8 + id.encoded_len()) as u32,
                b.alleles_start,
                "{at}: ID does not end where alleles_start says"
            );

            // Every allele lies in alleles_start .. filters_start.
            let mut pos = b.alleles_start as usize;
            let mut alleles = Vec::new();
            for _ in 0..batch.allele_count()[i] {
                let value = typed::read(bytes, pos).expect("allele");
                alleles.push(value.as_str().expect("allele is a string").to_vec());
                pos += value.encoded_len();
            }
            assert_eq!(pos as u32, b.filters_start, "{at}: alleles overrun FILTER");
            let expected: Vec<Vec<u8>> = record
                .alleles()
                .expect("alleles")
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect();
            assert_eq!(
                alleles, expected,
                "{at}: alleles decoded from bounds differ"
            );

            // FILTER occupies filters_start .. info_start.
            let filter = typed::read(bytes, b.filters_start as usize).expect("FILTER");
            assert_eq!(
                b.filters_start as usize + filter.encoded_len(),
                b.info_start as usize,
                "{at}: FILTER does not end where info_start says"
            );

            // INFO fills info_start .. genotypes_start exactly.
            let mut pos = b.info_start as usize;
            for _ in 0..batch.info_count()[i] {
                pos = typed::skip(bytes, pos).expect("INFO key");
                pos = typed::skip(bytes, pos).expect("INFO value");
            }
            assert_eq!(
                pos as u32, b.genotypes_start,
                "{at}: INFO does not fill the shared block"
            );
        }
    }
}

#[test]
fn a_record_view_rebuilt_from_columns_still_validates() {
    // record() slices using record_end, so a bounds error would hand back a
    // mis-framed record. `validate()` walks every typed value in both blocks
    // and requires them to consume l_shared and l_indiv exactly, so it fails
    // loudly on exactly that.
    for name in FIXTURES {
        let (host, batch) = columns(name);
        let buf = host.data();
        for i in 0..batch.len() {
            let record = batch.record(buf, i).expect("record view");
            record
                .validate()
                .unwrap_or_else(|e| panic!("{name} record {i}: {e}"));
        }
    }
}

#[test]
fn decoding_twice_does_not_accumulate() {
    let host = inflate("giab_hg002.bcf");
    let start = parse_header(host.data()).expect("header").records_start;
    let mut batch = RecordBatch::new();
    batch.decode(host.data(), start).expect("decode");
    let first = batch.len();
    batch.decode(host.data(), start).expect("decode");
    assert_eq!(batch.len(), first, "stale records from the previous decode");
}

#[test]
fn the_tail_matches_the_scan() {
    // The batch must report the same resumption point the boundary scan does,
    // or a driver carrying it forward would skip or repeat a record at a seam.
    for name in FIXTURES {
        let host = inflate(name);
        let buf = host.data();
        let start = parse_header(buf).expect("header").records_start;

        let (offsets, scan_tail) = fritillaria_bcf::columnar::scan_records(buf, start).unwrap();
        let mut batch = RecordBatch::new();
        let batch_tail = batch.decode(buf, start).unwrap();

        assert_eq!(batch_tail, scan_tail, "{name}: tail");
        assert_eq!(batch.record_offsets(), offsets, "{name}: offsets");
    }
}

#[test]
fn a_truncated_record_is_left_for_the_next_batch() {
    let host = inflate("giab_hg002.bcf");
    let buf = host.data();
    let start = parse_header(buf).expect("header").records_start;

    let mut whole = RecordBatch::new();
    whole.decode(buf, start).unwrap();
    assert!(whole.len() >= 2, "need at least two records to cut between");

    // Cut three bytes into the last record: everything before it still decodes,
    // and the tail points at the record that was cut.
    let last = whole.record_offsets()[whole.len() - 1];
    let truncated = &buf[..last + 3];

    let mut partial = RecordBatch::new();
    let tail = partial.decode(truncated, start).unwrap();

    assert_eq!(partial.len(), whole.len() - 1, "records before the cut");
    assert_eq!(tail, last, "tail must point at the incomplete record");
}
