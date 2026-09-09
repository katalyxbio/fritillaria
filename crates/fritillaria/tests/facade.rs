//! The migration claim, compiled rather than asserted in prose.
//!
//! The README says switching from noodles is a rename: `noodles_bam` becomes
//! `fritillaria::bam` and everything else stays put. That claim is only worth
//! making if the facade actually re-exports the whole surface, and a doc table
//! cannot fail. This file can.
//!
//! Each test below names a concrete type through the facade path a noodles user
//! would reach for. If a re-export is dropped, renamed, or accidentally left
//! behind a feature that is off by default, this stops compiling — which is the
//! point. The assertions are almost incidental; the `use` lines are the test.

/// Every format crate is reachable, and under the name noodles uses.
///
/// The default feature set covers everything that touches a *file*. htsget and
/// refget are deliberately excluded — they pull in a TLS stack — so their
/// absence here is the specification, not an oversight.
#[test]
fn every_file_format_is_reachable_by_its_noodles_name() {
    use fritillaria::{
        bam, bcf, bed, bgzf, core, cram, csi, fasta, fastq, gff, gtf, sam, tabix, vcf,
    };

    // Naming one type per crate is what forces the re-export to resolve; the
    // values are just something to compare.
    let _ = bgzf::VirtualPosition::default();
    let _ = core::Position::MIN;
    assert_eq!(bam::Record::default().flags().bits(), 4); // UNMAPPED
    let _ = bcf::Record::default();
    let _ = sam::Header::default();
    let _ = vcf::Header::default();
    let _ = csi::Index::default();
    let _ = tabix::Index::default();
    let _ = fasta::Record::new(
        fasta::record::Definition::new("sq0", None),
        b"ACGT".to_vec().into(),
    );
    let _ = fastq::Record::default();
    let _: Option<bed::Record<3>> = None;
    let _ = gff::Line::default();
    let _: Option<gtf::Record> = None;
    let _ = cram::Record::default();

    // Ours, and the only crate here that serves five formats at once.
    let _ = fritillaria::text::Dialect::SAM;
}

/// The GPU half is additive: it sits *next to* the vendored API, not over it.
///
/// Both `Record` types resolve from the same crate, which is the arrangement
/// `VENDORED.md` describes — the vendored owning record keeps the short name a
/// migrating caller expects, and ours lives under `columnar`.
#[test]
fn the_columnar_path_sits_alongside_the_vendored_api() {
    use fritillaria::bam;

    let vendored = bam::Record::default();
    assert!(vendored.name().is_none());

    // Ours: a zero-copy view over an inflated buffer, and a different type.
    assert_eq!(bam::columnar::RECORD_CORE_SIZE, 32);
    assert_eq!(bam::columnar::MAGIC, *b"BAM\x01");
}

/// The one-line migration from the README, compiled.
///
/// If this signature ever stops resolving, the drop-in property is gone and the
/// adoption story with it — so it is checked here rather than trusted.
#[test]
fn a_bgzf_reader_still_drops_into_a_vendored_format_reader() {
    use fritillaria::{bam, bgzf::BgzfReader};

    let empty: &[u8] = &[];
    let _reader: bam::io::Reader<BgzfReader<&[u8]>> = bam::io::Reader::from(BgzfReader::new(empty));
}
