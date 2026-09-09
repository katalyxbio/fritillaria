//! The text scanner against real SAM, VCF and GFF3, written by three different
//! tools — samtools, bcftools and NCBI.
//!
//! The oracles are the vendored parsers, which are noodles' code we neither
//! wrote nor modified. If our field boundaries disagree with theirs, one of us
//! is wrong about the format.
//!
//! Three writers matters. A scanner tested against one tool's output tests that
//! tool's habits as much as the format — the lesson `htslib_multiblock.bam`
//! taught when its blocking turned out to be htslib's preference rather than a
//! property of BAM.

use std::io::BufReader;
use std::path::PathBuf;

use fritillaria_text::columnar::{Dialect, RecordBatch};

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(path(name)).expect("fixture missing; see testdata/README.md")
}

fn decoded(name: &str, dialect: Dialect) -> (Vec<u8>, RecordBatch) {
    let buf = read(name);
    let mut batch = RecordBatch::new();
    batch.decode(&buf, 0, dialect).expect("decode");
    (buf, batch)
}

#[test]
fn sam_records_match_the_vendored_reader() {
    use fritillaria_sam as sam;

    let (buf, batch) = decoded("reads.sam", Dialect::SAM);

    let mut reader = sam::io::Reader::new(BufReader::new(&buf[..]));
    let header = reader.read_header().expect("vendored header");
    let expected: Vec<_> = reader
        .records()
        .map(|r| r.expect("vendored record"))
        .collect();

    assert!(!expected.is_empty());
    assert_eq!(batch.len(), expected.len(), "record count");
    assert!(
        !header.reference_sequences().is_empty(),
        "the fixture has @SQ lines, so the header must not be empty"
    );

    // SAM's first two fields are QNAME and FLAG; comparing them field-by-field
    // checks our tab boundaries against a parser that found them independently.
    for (i, want) in expected.iter().enumerate() {
        let name = batch.field(&buf, i, 0).expect("QNAME");
        assert_eq!(
            name,
            want.name().expect("vendored name"),
            "record {i}: QNAME"
        );

        let flag: u16 = std::str::from_utf8(batch.field(&buf, i, 1).expect("FLAG"))
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            flag,
            u16::from(want.flags().expect("vendored flags")),
            "record {i}: FLAG"
        );
    }
}

#[test]
fn vcf_records_match_the_vendored_reader() {
    use fritillaria_vcf as vcf;

    let (buf, batch) = decoded("calls.vcf", Dialect::VCF);

    let mut reader = vcf::io::Reader::new(BufReader::new(&buf[..]));
    let _header = reader.read_header().expect("vendored header");
    let expected: Vec<_> = reader
        .records()
        .map(|r| r.expect("vendored record"))
        .collect();

    assert!(!expected.is_empty());
    assert_eq!(batch.len(), expected.len(), "record count");

    for (i, want) in expected.iter().enumerate() {
        let chrom = batch.field(&buf, i, 0).expect("CHROM");
        assert_eq!(
            chrom,
            want.reference_sequence_name().as_bytes(),
            "{i}: CHROM"
        );

        let pos = batch.field(&buf, i, 1).expect("POS");
        assert_eq!(
            std::str::from_utf8(pos).unwrap(),
            want.variant_start()
                .expect("vendored position")
                .expect("valid position")
                .to_string(),
            "record {i}: POS"
        );
    }
}

#[test]
fn gff3_records_match_the_vendored_reader() {
    use fritillaria_gff as gff;

    let (buf, batch) = decoded("lambda.gff3", Dialect::GFF);

    let mut reader = gff::io::Reader::new(BufReader::new(&buf[..]));
    let expected: Vec<_> = reader
        .record_bufs()
        .map(|r| r.expect("vendored record"))
        .collect();

    assert!(!expected.is_empty());
    assert_eq!(batch.len(), expected.len(), "record count");

    for (i, want) in expected.iter().enumerate() {
        let seqid = batch.field(&buf, i, 0).expect("seqid");
        assert_eq!(seqid, want.reference_sequence_name(), "record {i}: seqid");

        let source = batch.field(&buf, i, 1).expect("source");
        assert_eq!(source, want.source(), "record {i}: source");
    }
}

#[test]
fn every_format_has_the_field_count_its_spec_demands() {
    // SAM has 11 mandatory fields plus optional tags; VCF has 8 plus FORMAT and
    // one column per sample; GFF3 has exactly 9. A wrong tab boundary would
    // shift these, and they are checked against the spec rather than against
    // ourselves.
    for (name, dialect, expected) in [
        ("reads.sam", Dialect::SAM, 12usize),
        ("calls.vcf", Dialect::VCF, 10),
        ("lambda.gff3", Dialect::GFF, 9),
    ] {
        let (_, batch) = decoded(name, dialect);
        let counts = batch.field_counts();
        let ragged: Vec<_> = counts.iter().filter(|&&c| c != expected).collect();
        assert!(
            ragged.is_empty(),
            "{name}: {} records do not have {expected} fields",
            ragged.len()
        );
    }
}

#[test]
fn header_lines_are_counted_and_excluded() {
    // Getting this wrong would parse a header as a record — plausible-looking
    // and completely wrong, since a `##INFO=<...>` line has no tabs at all.
    for (name, dialect, headers) in [
        ("reads.sam", Dialect::SAM, 4usize),
        ("calls.vcf", Dialect::VCF, 235),
        ("lambda.gff3", Dialect::GFF, 5),
    ] {
        let (buf, batch) = decoded(name, dialect);
        assert_eq!(
            batch.header_offsets().len(),
            headers,
            "{name}: header lines"
        );
        for &offset in batch.header_offsets() {
            assert!(
                dialect.is_header(&buf, offset),
                "{name}: a non-header line was classified as one"
            );
        }
        for &offset in batch.record_offsets() {
            assert!(
                !dialect.is_header(&buf, offset),
                "{name}: a header line was classified as a record"
            );
        }
    }
}

#[test]
fn slicing_at_the_tabs_reassembles_the_record() {
    // The strongest structural check available without interpreting anything:
    // joining a record's fields with tabs must give the record back exactly.
    for (name, dialect) in [
        ("reads.sam", Dialect::SAM),
        ("calls.vcf", Dialect::VCF),
        ("lambda.gff3", Dialect::GFF),
    ] {
        let (buf, batch) = decoded(name, dialect);
        for i in 0..batch.len() {
            let joined = batch
                .record_fields(&buf, i)
                .collect::<Vec<_>>()
                .join(&b'\t');
            assert_eq!(
                joined,
                batch.record(&buf, i).expect("record"),
                "{name} record {i}: fields do not reassemble"
            );
        }
    }
}

#[test]
fn no_field_contains_a_tab_or_a_newline() {
    // The assumption the whole scanner rests on, asserted against real data
    // from three writers rather than taken from the specs.
    for (name, dialect) in [
        ("reads.sam", Dialect::SAM),
        ("calls.vcf", Dialect::VCF),
        ("lambda.gff3", Dialect::GFF),
    ] {
        let (buf, batch) = decoded(name, dialect);
        for i in 0..batch.len() {
            for (f, field) in batch.record_fields(&buf, i).enumerate() {
                assert!(
                    !field.contains(&b'\t') && !field.contains(&b'\n'),
                    "{name} record {i} field {f} contains a delimiter"
                );
            }
        }
    }
}

#[test]
fn the_sam_fixture_still_contains_the_quote_bytes_that_misled_the_first_scan() {
    // `"` is Phred+33 Q1, ordinary quality data. A scanner that treated it as a
    // delimiter reported 1,956 phantom "quoted tabs" here. If this fixture is
    // ever regenerated without them, the note in the crate docs stops being
    // demonstrable.
    let (buf, batch) = decoded("reads.sam", Dialect::SAM);
    let quotes: usize = (0..batch.len())
        .map(|i| {
            batch
                .record(&buf, i)
                .map_or(0, |r| r.iter().filter(|&&b| b == b'"').count())
        })
        .sum();
    assert!(
        quotes > 100,
        "expected many '\"' bytes in the quality strings, found {quotes}"
    );
}

#[test]
fn batching_at_a_line_boundary_does_not_change_the_answer() {
    // A batched caller carries the unterminated trailing line forward.
    let buf = read("lambda.gff3");
    let mut whole = RecordBatch::new();
    whole.decode(&buf, 0, Dialect::GFF).unwrap();
    let expected = whole.field_counts();

    // Cut mid-line, well inside the records.
    let cut = whole.record_offsets()[100] + 20;
    let mut first = RecordBatch::new();
    let tail = first.decode(&buf[..cut], 0, Dialect::GFF).unwrap();
    assert_eq!(tail, whole.record_offsets()[100], "tail at the cut line");

    let mut second = RecordBatch::new();
    second.decode(&buf, tail, Dialect::GFF).unwrap();

    let mut counts = first.field_counts();
    counts.extend(second.field_counts());
    assert_eq!(counts, expected, "field counts across the seam");
}
