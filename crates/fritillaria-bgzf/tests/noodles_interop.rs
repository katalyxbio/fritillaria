//! Proof that noodles format crates compose on top of this reader.
//!
//! This is the load-bearing test for the whole integration strategy. noodles'
//! format readers are generic over `noodles_bgzf::io::BufRead` rather than over
//! a concrete type, so implementing that trait means every noodles format —
//! BAM here, but equally BCF, `bgzip`ped VCF, and anything tabix-indexed — can
//! be layered on our decompression without forking a line of noodles.
//!
//! Swapping [`CpuCodec`] for the CUDA codec accelerates all of them at once.
//! If this test ever stops compiling, the strategy has broken, not just a test.

#![cfg(feature = "noodles")]

use std::path::PathBuf;

use fritillaria_bgzf::BgzfReader;
use noodles_bam as bam;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// Opens an htslib-written BAM through our reader, wrapped in noodles' parser.
fn open(name: &str) -> bam::io::Reader<BgzfReader<std::fs::File>> {
    let file = std::fs::File::open(fixture(name)).expect("fixture missing");
    bam::io::Reader::from(BgzfReader::new(file))
}

#[test]
fn noodles_reads_a_header_through_our_decompression() {
    let mut reader = open("htslib.bam");
    let header = reader.read_header().expect("noodles must parse the header");

    // Ground truth from `samtools view -H`.
    let refs = header.reference_sequences();
    assert_eq!(refs.len(), 2);
    assert!(refs.contains_key(&b"chr1"[..]));
    assert!(refs.contains_key(&b"chr2"[..]));
}

#[test]
fn noodles_reads_records_through_our_decompression() {
    let mut reader = open("htslib.bam");
    let _header = reader.read_header().unwrap();

    let records: Vec<_> = reader
        .records()
        .collect::<std::io::Result<Vec<_>>>()
        .expect("noodles must decode every record");

    assert_eq!(records.len(), 8, "samtools view -c reports 8 records");

    // BAM positions are 0-based; noodles' Position is 1-based, matching SAM.
    let first = &records[0];
    assert_eq!(
        first.alignment_start().unwrap().unwrap().get(),
        7,
        "SAM-space position of r001"
    );
    assert_eq!(first.flags().bits(), 99);
    assert_eq!(first.mapping_quality().unwrap().get(), 60);
    assert_eq!(first.sequence().len(), 17);

    // The unmapped record has no reference or position.
    let unmapped = &records[6];
    assert!(unmapped.alignment_start().is_none());
    assert!(unmapped.flags().is_unmapped());
}

#[test]
fn noodles_reads_records_spanning_block_and_batch_boundaries() {
    // 4000 records over 15 BGZF blocks, read with a batch size of one block so
    // the reader must also stitch across batches. Both seams at once.
    let file = std::fs::File::open(fixture("htslib_multiblock.bam")).unwrap();
    let inner = BgzfReader::new(file).with_blocks_per_batch(1);
    let mut reader = bam::io::Reader::from(inner);

    let _header = reader.read_header().unwrap();
    let count = reader.records().count();

    assert_eq!(count, 4000, "records lost or duplicated across a seam");
}

#[test]
fn record_contents_survive_the_seams() {
    let file = std::fs::File::open(fixture("htslib_multiblock.bam")).unwrap();
    let inner = BgzfReader::new(file).with_blocks_per_batch(1);
    let mut reader = bam::io::Reader::from(inner);
    let _header = reader.read_header().unwrap();

    // Positions were generated at 1, 101, 201, ... in SAM space.
    for (i, record) in reader.records().enumerate() {
        let record = record.unwrap();
        let start = record.alignment_start().unwrap().unwrap().get();
        assert_eq!(start, 1 + i * 100, "position drift at record {i}");
        assert_eq!(record.sequence().len(), 100, "sequence truncated at {i}");
    }
}
