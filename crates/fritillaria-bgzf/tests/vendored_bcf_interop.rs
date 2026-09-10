//! The "container, not format" claim, tested rather than argued.
//!
//! `tests/vendored_interop.rs` shows the vendored BAM reader sitting on top of
//! this one. The design makes a stronger claim from that:
//! because every format reader is generic over `fritillaria_bgzf::io::BufRead`,
//! a *second* BGZF-contained format should need no new code from us at all.
//!
//! BCF is the first chance to check that, and it is worth checking rather than
//! assuming — the claim is what the project's roadmap ordering rests on, and
//! "should compose" and "does compose" are not the same statement. Nothing in
//! this file touches our BCF code: it opens a bcftools-written file through
//! [`BgzfReader`], hands it to the vendored reader, and reads variants out.
//!
//! Swapping [`fritillaria_bgzf::CpuCodec`] for the CUDA codec accelerates this
//! path exactly as it does BAM's, which is the whole point.

use std::path::PathBuf;

use fritillaria_bcf as bcf;
use fritillaria_bgzf::BgzfReader;
use fritillaria_vcf::header::StringMaps;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// Opens a bcftools-written BCF through our reader, wrapped in the vendored parser.
fn open(name: &str) -> bcf::io::Reader<BgzfReader<std::fs::File>> {
    let file = std::fs::File::open(fixture(name)).expect("fixture missing");
    bcf::io::Reader::from(BgzfReader::new(file))
}

#[test]
fn the_vendored_reader_reads_a_bcf_header_through_our_decompression() {
    let mut reader = open("kg_phase3.bcf");
    let header = reader
        .read_header()
        .expect("the vendored reader must parse the header");

    assert_eq!(header.sample_names().len(), 2504);
    assert!(header.sample_names().contains("HG00096"));
    assert_eq!(header.contigs().len(), 86);
}

#[test]
fn the_vendored_reader_reads_bcf_records_through_our_decompression() {
    let mut reader = open("giab_hg002.bcf");
    let header = reader.read_header().unwrap();
    let string_maps = StringMaps::try_from(&header).expect("header must yield string maps");

    let mut count = 0;
    let mut first = None;
    for result in reader.records() {
        let record = result.expect("the vendored reader must decode every record");
        if first.is_none() {
            let position = record
                .variant_start()
                .expect("record must be placed")
                .expect("position must parse");
            first = Some((
                record
                    .reference_sequence_name(&string_maps)
                    .unwrap()
                    .to_string(),
                usize::from(position),
            ));
        }
        count += 1;
    }

    // Ground truth from `bcftools view -H | wc -l` and `bcftools query`.
    assert_eq!(count, 275);
    assert_eq!(first, Some(("chr20".to_string(), 1_000_474)));
}

#[test]
fn the_vendored_reader_reads_bcf_records_that_span_block_and_batch_boundaries() {
    // The seam that matters, and it matters more for BCF than for BAM: htslib
    // starts a new block rather than splitting an alignment, but `bcf_write`
    // packs blocks full, so essentially every interior block boundary falls
    // inside a record. Driving the reader at one block per batch puts a seam
    // in the hardest place available.
    let file = std::fs::File::open(fixture("kg_phase3.bcf")).unwrap();
    let mut reader = bcf::io::Reader::from(BgzfReader::new(file).with_blocks_per_batch(1));

    let header = reader.read_header().unwrap();
    assert_eq!(header.sample_names().len(), 2504);

    let mut count = 0;
    for result in reader.records() {
        result.expect("every record must decode across the seam");
        count += 1;
    }
    assert_eq!(count, 715, "no record may be lost or split at a seam");
}
