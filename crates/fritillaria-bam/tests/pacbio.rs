//! Aux tag validation against a real PacBio HiFi BAM.
//!
//! # Why a long-read fixture specifically
//!
//! Aux tags are optional trailing detail in aligned Illumina data and the
//! *payload* in long-read data: PacBio and ONT ship unaligned BAM as their raw
//! delivery format, and the basecaller's output lives in tags (`MM`/`ML` base
//! modifications, per-base kinetics, and so on). On these records the tags are
//! a large fraction of each record, so a decoder that is wrong about them is
//! wrong about the interesting data.
//!
//! The fixture is cut from GIAB HG002; see `testdata/README.md` for the exact
//! provenance and reproduction command. Ground truth is `samtools view`, so
//! these tests are not circular — nothing here was produced by this crate.
//!
//! # What this fixture cannot test
//!
//! - **The `CG` long-CIGAR overflow.** HiFi reads are far too accurate to reach
//!   65535 CIGAR operations (the longest record here uses 790), so that path is
//!   covered by hand-built records in `record.rs` instead.
//! - **A record spanning a BGZF block boundary.** htslib never splits a record
//!   under 65280 bytes across blocks, and the largest record here is 56 KB.

use std::collections::BTreeSet;
use std::path::PathBuf;

use fritillaria_bam::{Array, Record, Value, header::parse_header, scan_records, seq};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

const FIXTURE: &str = "pacbio_hifi.bam";

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("htslib BGZF must inflate and verify");
    out
}

/// Runs a samtools subcommand, or `None` if samtools is unavailable.
///
/// Absent on the Colab image, so remote runs skip rather than fail — but the
/// skip is loud, because a silently skipped validation is worthless.
fn samtools(args: &[&str]) -> Option<std::process::Output> {
    match std::process::Command::new("samtools").args(args).output() {
        Ok(output) => Some(output),
        Err(err) => {
            eprintln!("SKIP: samtools unavailable ({err})");
            None
        }
    }
}

/// `samtools view` output, one line per record.
fn ground_truth() -> Option<Vec<String>> {
    let path = fixture(FIXTURE);
    let out = samtools(&["view", path.to_str().unwrap()])?;
    assert!(
        out.status.success(),
        "samtools view failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("samtools emits UTF-8 here");
    Some(text.lines().map(str::to_owned).collect())
}

/// Offsets of every record in the inflated buffer.
fn records(batch: &InflateBatch) -> (usize, Vec<usize>) {
    let header = parse_header(batch.data()).expect("header must parse");
    let (offsets, tail) = scan_records(batch.data(), header.records_start).expect("scan");
    assert_eq!(
        tail,
        batch.data().len(),
        "the whole file is buffered, so no record may be partial"
    );
    (header.records_start, offsets)
}

/// Renders one decoded aux value the way `samtools view` prints it.
///
/// Integer widths all collapse to `i` in SAM, which is exactly why the decoder
/// keeps the width internally and this test cannot check it — see
/// `aux::tests::decodes_every_scalar_type` for that half.
fn render(value: &Value<'_>) -> String {
    fn join<T: std::fmt::Display>(sub: char, it: impl Iterator<Item = T>) -> String {
        let mut s = format!("B:{sub}");
        for v in it {
            s.push(',');
            s.push_str(&v.to_string());
        }
        s
    }

    match value {
        Value::Character(c) => format!("A:{}", *c as char),
        Value::Float(f) => format!("f:{f}"),
        Value::String(s) => format!("Z:{}", String::from_utf8_lossy(s)),
        Value::Hex(s) => format!("H:{}", String::from_utf8_lossy(s)),
        Value::Array(a) => match a {
            Array::Int8(v) => join('c', v.iter()),
            Array::UInt8(v) => join('C', v.iter()),
            Array::Int16(v) => join('s', v.iter()),
            Array::UInt16(v) => join('S', v.iter()),
            Array::Int32(v) => join('i', v.iter()),
            Array::UInt32(v) => join('I', v.iter()),
            Array::Float(v) => join('f', v.iter()),
        },
        other => format!("i:{}", other.as_int().expect("remaining variants are ints")),
    }
}

/// Compares one rendered field against samtools' text.
///
/// Floats are compared numerically: htslib prints them with `%g`, so the text
/// is lossy and a string compare would fail on values that are bit-identical.
fn fields_agree(ours: &str, theirs: &str) -> bool {
    if ours == theirs {
        return true;
    }
    let (Some(a), Some(b)) = (ours.strip_prefix("f:"), theirs.strip_prefix("f:")) else {
        // Float arrays: compare element-wise under the same rule.
        if let (Some(a), Some(b)) = (ours.strip_prefix("B:f,"), theirs.strip_prefix("B:f,")) {
            let (a, b): (Vec<&str>, Vec<&str>) = (a.split(',').collect(), b.split(',').collect());
            return a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| floats_agree(x, y));
        }
        return false;
    };
    floats_agree(a, b)
}

fn floats_agree(a: &str, b: &str) -> bool {
    let (Ok(a), Ok(b)) = (a.parse::<f32>(), b.parse::<f32>()) else {
        return false;
    };
    // Exact equality first, deliberately: it is the common case, and it is also
    // the only branch that handles infinities, where the relative test below
    // computes inf - inf = NaN and would report a mismatch.
    #[allow(clippy::float_cmp, reason = "exact hit is the fast path; see above")]
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
    (a - b).abs() / scale < 1e-5
}

#[test]
fn every_aux_tag_of_every_record_matches_samtools() {
    let Some(truth) = ground_truth() else { return };
    let batch = inflate(FIXTURE);
    let (_, offsets) = records(&batch);

    assert_eq!(
        offsets.len(),
        truth.len(),
        "record count must match samtools"
    );
    assert!(!offsets.is_empty(), "fixture must not be empty");

    // Proof the interesting paths actually ran: a test that decoded no arrays
    // would pass this file just as happily.
    let mut scalar_types = BTreeSet::new();
    let mut array_subtypes = BTreeSet::new();
    let mut total_fields = 0usize;

    for (i, &offset) in offsets.iter().enumerate() {
        let record = Record::new(&batch.data()[offset..]).expect("record must parse");
        let expected: Vec<&str> = truth[i].split('\t').skip(11).collect();

        let mut got = Vec::new();
        for field in record.aux() {
            let (tag, value) = field.unwrap_or_else(|e| panic!("record {i} aux: {e}"));
            scalar_types.insert(value.ty());
            if let Some(a) = value.as_array() {
                array_subtypes.insert(a.subtype());
            }
            total_fields += 1;
            got.push(format!(
                "{}:{}",
                std::str::from_utf8(&tag).expect("tags are ASCII"),
                render(&value)
            ));
        }

        assert_eq!(
            got.len(),
            expected.len(),
            "record {i}: tag count differs\n  ours:   {got:?}\n  theirs: {expected:?}"
        );
        for (ours, theirs) in got.iter().zip(&expected) {
            assert!(
                fields_agree(ours, theirs),
                "record {i}: {ours}\n           != {theirs}"
            );
        }
    }

    // Recorded rather than asserted loosely: these are the types this real file
    // happens to contain, and the rest are covered synthetically in `aux.rs`.
    assert_eq!(
        scalar_types,
        BTreeSet::from([b'B', b'C', b'I', b'S', b'Z', b'f']),
        "binary scalar types exercised by the fixture"
    );
    assert_eq!(
        array_subtypes,
        BTreeSet::from([b'C', b'S', b'f', b'i']),
        "B subtypes exercised by the fixture"
    );
    assert!(
        total_fields > 500,
        "expected a tag-heavy file, got {total_fields}"
    );
}

#[test]
fn core_fields_of_long_reads_match_samtools() {
    let Some(truth) = ground_truth() else { return };
    let batch = inflate(FIXTURE);
    let (_, offsets) = records(&batch);

    let mut longest = 0usize;
    for (i, &offset) in offsets.iter().enumerate() {
        let record = Record::new(&batch.data()[offset..]).expect("record must parse");
        let cols: Vec<&str> = truth[i].split('\t').collect();

        assert_eq!(
            std::str::from_utf8(record.name()).unwrap(),
            cols[0],
            "record {i} name"
        );
        assert_eq!(record.flags().to_string(), cols[1], "record {i} flag");
        // BAM is 0-based, SAM is 1-based. Getting this wrong shifts every
        // coordinate by one and still looks plausible.
        assert_eq!(
            (record.position() + 1).to_string(),
            cols[3],
            "record {i} pos"
        );
        assert_eq!(
            record.mapping_quality().to_string(),
            cols[4],
            "record {i} mapq"
        );

        let cigar =
            record
                .cigar_resolved()
                .expect("cigar")
                .iter()
                .fold(String::new(), |mut s, op| {
                    use std::fmt::Write as _;
                    write!(
                        s,
                        "{}{}",
                        seq::cigar_op_len(op),
                        seq::cigar_op_kind(op).expect("valid op") as char
                    )
                    .expect("writing to a String cannot fail");
                    s
                });
        assert_eq!(cigar, cols[5], "record {i} cigar");

        let sequence = record.sequence();
        assert_eq!(
            std::str::from_utf8(&sequence).unwrap(),
            cols[9],
            "record {i} sequence"
        );

        let quals: String = record
            .qualities()
            .iter()
            .map(|q| (q + 33) as char)
            .collect();
        assert_eq!(quals, cols[10], "record {i} qualities");

        longest = longest.max(sequence.len());
    }

    assert!(
        longest > 10_000,
        "fixture must actually contain long reads; longest is {longest} bp"
    );
}

#[test]
fn the_base_modification_tags_are_present_and_decode() {
    // MM/ML are the reason long-read uBAM support needs aux decoding at all:
    // they carry the basecaller's methylation calls.
    let Some(_) = ground_truth() else { return };
    let batch = inflate(FIXTURE);
    let (_, offsets) = records(&batch);

    let mut seen = 0usize;
    for &offset in &offsets {
        let record = Record::new(&batch.data()[offset..]).expect("record must parse");

        let Some(mm) = record.aux_get(*b"MM").expect("aux must decode") else {
            continue;
        };
        let ml = record
            .aux_get(*b"ML")
            .expect("aux must decode")
            .expect("ML accompanies MM");

        let mm = mm.as_bytes().expect("MM is a Z string");
        assert!(
            mm.starts_with(b"C+m") || mm.starts_with(b"C+h"),
            "MM should describe a cytosine modification, got {:?}",
            String::from_utf8_lossy(&mm[..mm.len().min(16)])
        );

        let Some(Array::UInt8(probs)) = ml.as_array() else {
            panic!("ML must be a B:C array of probabilities");
        };
        assert!(!probs.is_empty(), "ML must carry probabilities");
        seen += 1;
    }

    assert!(seen > 0, "fixture must contain MM/ML tags");
}

#[test]
fn no_record_in_this_fixture_uses_the_long_cigar_placeholder() {
    // Recorded as a fact about the fixture, not as a property of BAM: HiFi is
    // too accurate to overflow n_cigar_op. If this ever fails, a genuine
    // long-CIGAR fixture has arrived and `record.rs`'s synthetic tests can be
    // backed by real data.
    let batch = inflate(FIXTURE);
    let (_, offsets) = records(&batch);

    let max_ops = offsets
        .iter()
        .map(|&o| {
            Record::new(&batch.data()[o..])
                .expect("record must parse")
                .cigar_op_count()
        })
        .max()
        .expect("fixture is non-empty");

    assert!(max_ops < 65_535, "unexpectedly long CIGAR: {max_ops} ops");
    assert!(
        offsets.iter().all(|&o| !Record::new(&batch.data()[o..])
            .expect("record must parse")
            .has_long_cigar_placeholder()),
        "no HiFi record should look like a long-CIGAR placeholder"
    );
}
