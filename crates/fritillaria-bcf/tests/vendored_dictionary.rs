//! The dictionary, checked against a second independent implementation.
//!
//! `tests/bcftools.rs` already checks INFO and FILTER values by *name* against
//! bcftools, which catches a dictionary that is wrong in a way that changes a
//! value. This catches a narrower and nastier case: numbering that happens to
//! agree with ours on the keys the fixtures use, and diverges elsewhere.
//!
//! The vendored `fritillaria-vcf` builds the same two maps from the same header
//! text and calls them `StringMaps`. It was written from the spec by someone
//! else — it is noodles' code, not ours — so agreement between it, bcftools and
//! this crate is three implementations rather than two, the same standard the
//! CUDA codecs are held to.
//!
//! The IDX rules are the reason this is worth a test of its own. `IDX=`
//! overrides the implicit position, an ID declared as both INFO and FORMAT
//! takes one slot rather than two, and PASS occupies slot 0 whether declared or
//! not. Each is easy to implement in a way that is self-consistent and wrong,
//! and none of them errors when it is.

use std::path::PathBuf;

use fritillaria_bcf::columnar::header::parse_header;
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};
use fritillaria_vcf::{self as vcf, header::StringMaps};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing");
    let spans = discover_blocks(&raw, 0).unwrap();
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .unwrap();
    out
}

#[test]
fn our_dictionary_matches_the_one_the_vendored_crate_builds() {
    for name in ["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"] {
        let batch = inflate(name);
        let header = parse_header(batch.data()).unwrap();

        // Same input text, independently interpreted.
        let text = String::from_utf8(header.text.clone()).expect("VCF headers are ASCII");
        let parsed: vcf::Header = text
            .parse()
            .expect("the vendored parser must accept the header text");
        let maps =
            StringMaps::try_from(&parsed).expect("the vendored parser must build string maps");

        // Compared by index in both directions, and past the end of each
        // table: agreeing on the entries we happen to have is weaker than
        // agreeing on which indices exist at all. `IDX` can leave gaps, so
        // "both say nothing is here" is a real assertion, not a vacuous one.
        let probe = header
            .dictionary
            .contigs
            .len()
            .max(header.dictionary.strings.len())
            + 32;

        let mut contigs = 0;
        for index in 0..probe {
            let ours = header.dictionary.contig(i32::try_from(index).unwrap());
            let theirs = maps.contigs().get_index(index);
            match (ours, theirs) {
                (Some(a), Some(b)) => {
                    assert_eq!(String::from_utf8_lossy(a), b, "{name}: contig {index}");
                    contigs += 1;
                }
                (None, None) => {}
                _ => panic!(
                    "{name}: contig {index} — ours {:?}, vendored {theirs:?}",
                    ours.map(String::from_utf8_lossy)
                ),
            }
        }
        assert_eq!(
            contigs,
            header.dictionary.contigs.len(),
            "{name}: every contig must have been compared"
        );

        // FILTER/INFO/FORMAT keys share one namespace, so a disagreement here
        // is exactly the silent misattribution this test exists to catch.
        let mut strings = 0;
        for index in 0..probe {
            let ours = header.dictionary.string(i32::try_from(index).unwrap());
            let theirs = maps.strings().get_index(index);
            match (ours, theirs) {
                (Some(a), Some(b)) => {
                    assert_eq!(String::from_utf8_lossy(a), b, "{name}: key {index}");
                    strings += 1;
                }
                (None, None) => {}
                _ => panic!(
                    "{name}: key {index} — ours {:?}, vendored {theirs:?}",
                    ours.map(String::from_utf8_lossy)
                ),
            }
        }
        assert!(
            strings > 10,
            "{name}: only {strings} keys compared; the fixture got thinner"
        );
    }
}

#[test]
fn the_idx_gap_fixture_actually_has_a_gap() {
    // Without this, the test above proves nothing about IDX. bcftools writes
    // IDX on every dictionary line, but on a freshly converted file those
    // numbers happen to equal the declaration order — so ignoring IDX
    // entirely still passes. `giab_hg002_idx_gap.bcf` had one INFO tag removed
    // with `bcftools annotate -x`, which is the exact situation IDX exists
    // for: the numbers survive the deletion and the order no longer matches.
    //
    // Asserting the gap here means that if the fixture is ever regenerated
    // from a file without one, this fails loudly instead of the IDX path
    // quietly going untested again.
    let batch = inflate("giab_hg002_idx_gap.bcf");
    let header = parse_header(batch.data()).unwrap();
    let text = String::from_utf8(header.text.clone()).unwrap();

    let declared: Vec<(usize, usize)> = text
        .lines()
        .filter(|l| {
            l.starts_with("##FILTER=") || l.starts_with("##INFO=") || l.starts_with("##FORMAT=")
        })
        .enumerate()
        .map(|(order, line)| {
            let idx: usize = line
                .split("IDX=")
                .nth(1)
                .expect("bcftools writes IDX on every dictionary line")
                .trim_end_matches('>')
                .trim_matches('"')
                .parse()
                .expect("IDX must be a number");
            (order, idx)
        })
        .collect();

    let mismatched = declared.iter().filter(|(order, idx)| idx != order).count();
    assert!(
        mismatched > 0,
        "the fixture's IDX values still match declaration order, so it does \
         not exercise IDX at all"
    );

    // And the resolved dictionary must follow IDX, not position.
    let platformnames = header
        .dictionary
        .strings
        .iter()
        .position(|e| e == b"platformnames")
        .expect("platformnames must resolve");
    assert_eq!(
        platformnames, 12,
        "platformnames keeps the number it had before `platforms` was removed"
    );
    assert_eq!(
        header.dictionary.string(11),
        None,
        "slot 11 was vacated by the removed tag and must stay empty"
    );
}

#[test]
fn sample_names_match_the_ones_the_vendored_crate_parses() {
    for name in ["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"] {
        let batch = inflate(name);
        let header = parse_header(batch.data()).unwrap();
        let text = String::from_utf8(header.text.clone()).unwrap();
        let parsed: vcf::Header = text.parse().unwrap();

        let ours: Vec<String> = header
            .samples
            .iter()
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        let theirs: Vec<String> = parsed.sample_names().iter().cloned().collect();
        assert_eq!(ours, theirs, "{name}: sample names and order");
    }
}
