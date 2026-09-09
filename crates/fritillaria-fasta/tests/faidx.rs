//! The FASTA columnar path against `samtools faidx` and the vendored reader.
//!
//! Two independent oracles, and neither is ours:
//!
//! 1. **`samtools faidx`** produces a `.fai` whose five columns are exactly what
//!    [`RecordBounds`] holds — name, length, offset, linebases, linewidth. If
//!    our measurement of a contig disagrees with htslib's, one of us is wrong
//!    about the format.
//! 2. **The vendored reader**, which is noodles' code we neither wrote nor
//!    modified, gives the sequences themselves.
//!
//! `testdata/controls.fa` is real: phiX174 and lambda, fetched from NCBI, which
//! wrote the 70-column wrapping. Both are standard sequencing controls.

use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;

use fritillaria_fasta::columnar::RecordBatch;

const FIXTURE: &str = "controls.fa";

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn fixture() -> Vec<u8> {
    std::fs::read(path(FIXTURE)).expect("fixture missing; see testdata/README.md")
}

fn decoded() -> (Vec<u8>, RecordBatch) {
    let buf = fixture();
    let mut batch = RecordBatch::new();
    batch.decode(&buf, 0, true).expect("decode");
    (buf, batch)
}

/// One `.fai` row: NAME, LENGTH, OFFSET, LINEBASES, LINEWIDTH.
#[derive(Debug, PartialEq, Eq)]
struct FaiRow {
    name: String,
    length: u64,
    offset: u64,
    line_bases: u32,
    line_width: u32,
}

/// The committed `.fai`, which `samtools faidx` generated.
fn committed_fai() -> Vec<FaiRow> {
    let text = std::fs::read_to_string(path("controls.fa.fai")).expect("committed .fai missing");
    parse_fai(&text)
}

fn parse_fai(text: &str) -> Vec<FaiRow> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            FaiRow {
                name: f[0].to_string(),
                length: f[1].parse().unwrap(),
                offset: f[2].parse().unwrap(),
                line_bases: f[3].parse().unwrap(),
                line_width: f[4].parse().unwrap(),
            }
        })
        .collect()
}

/// Regenerates the `.fai` with samtools, or `None` when it is not installed.
fn samtools_fai() -> Option<Vec<FaiRow>> {
    let dir = tempdir()?;
    let copy = dir.join(FIXTURE);
    std::fs::copy(path(FIXTURE), &copy).ok()?;
    let out = Command::new("samtools")
        .arg("faidx")
        .arg(&copy)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("NOTE: samtools faidx failed; skipping the live htslib check");
        return None;
    }
    let text = std::fs::read_to_string(copy.with_extension("fa.fai")).ok()?;
    let rows = parse_fai(&text);
    std::fs::remove_dir_all(&dir).ok();
    Some(rows)
}

fn tempdir() -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("fritillaria-faidx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

#[test]
fn our_contig_index_matches_the_fai_samtools_wrote() {
    let (buf, batch) = decoded();
    let expected = committed_fai();

    assert_eq!(batch.len(), expected.len(), "contig count");

    for (i, want) in expected.iter().enumerate() {
        let record = batch.record(&buf, i).expect("record view");
        let bounds = batch.bounds()[i];
        let at = format!("contig {i} ({})", want.name);

        assert_eq!(
            std::str::from_utf8(record.name()).unwrap(),
            want.name,
            "{at}: NAME"
        );
        assert_eq!(bounds.sequence_len, want.length, "{at}: LENGTH");
        assert_eq!(bounds.line_bases, want.line_bases, "{at}: LINEBASES");
        assert_eq!(bounds.line_width, want.line_width, "{at}: LINEWIDTH");

        // The .fai OFFSET is absolute in the file; ours is relative to the
        // record, so it is the record offset plus the sequence start.
        let absolute = batch.record_offsets()[i] as u64 + u64::from(bounds.sequence_start);
        assert_eq!(absolute, want.offset, "{at}: OFFSET");
    }
}

#[test]
fn the_committed_fai_is_what_samtools_produces_today() {
    // Guards the oracle itself: the .fai is committed so the test above runs
    // without samtools, but a stale one would silently stop being ground truth.
    let Some(live) = samtools_fai() else {
        eprintln!("NOTE: samtools not installed; the committed .fai is unverified here");
        return;
    };
    assert_eq!(live, committed_fai(), "the committed .fai has gone stale");
}

#[test]
fn compaction_reproduces_what_the_vendored_reader_reads() {
    // The strongest available check on the compaction: noodles' own parser
    // reads the sequences independently, and the compacted buffer must contain
    // exactly those bases, back to back.
    let (buf, batch) = decoded();
    let reference = batch.compact(&buf).expect("compact");

    let mut reader = fritillaria_fasta::io::Reader::new(BufReader::new(&buf[..]));
    let mut expected: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for result in reader.records() {
        let record = result.expect("vendored read");
        expected.push((record.name().to_vec(), record.sequence().as_ref().to_vec()));
    }

    assert_eq!(batch.len(), expected.len(), "contig count");
    assert!(
        !reference.contains(&b'\n'),
        "compaction must remove newlines"
    );

    for (i, (name, sequence)) in expected.iter().enumerate() {
        let record = batch.record(&buf, i).expect("record view");
        assert_eq!(record.name(), &name[..], "contig {i}: name");

        let at = batch.sequence_offsets()[i] as usize;
        let len = batch.bounds()[i].sequence_len as usize;
        assert_eq!(len, sequence.len(), "contig {i}: length");
        assert_eq!(&reference[at..at + len], &sequence[..], "contig {i}: bases");
    }
}

#[test]
fn the_fixture_is_actually_wrapped() {
    // A fixture that is supposed to exercise wrapping should assert that it
    // does. Unwrapped FASTA would make the compaction a no-op and every test
    // above would pass while testing nothing.
    let (buf, batch) = decoded();
    for (i, bounds) in batch.bounds().iter().enumerate() {
        assert!(
            bounds.sequence_len > u64::from(bounds.line_bases),
            "contig {i} fits on one line, so it is not wrapped"
        );
        let record = batch.record(&buf, i).expect("record view");
        assert!(
            record.wrapped_sequence().contains(&b'\n'),
            "contig {i}: no interior newline"
        );
    }
    assert!(batch.is_uniform(), "faidx would refuse a non-uniform file");
}

#[test]
fn the_reference_is_smaller_than_the_file_by_exactly_the_newlines() {
    let (buf, batch) = decoded();
    let reference = batch.compact(&buf).expect("compact");

    // Every newline in the sequence spans, and no others: the definition lines'
    // terminators are not part of any sequence span.
    let sequence_newlines: usize = batch
        .records(&buf)
        .map(|r| r.wrapped_sequence().iter().filter(|&&b| b == b'\n').count())
        .sum();
    let sequence_span: usize = batch
        .bounds()
        .iter()
        .map(|b| b.sequence_span() as usize)
        .sum();

    assert_eq!(
        reference.len(),
        sequence_span - sequence_newlines,
        "compaction must drop the newlines and nothing else"
    );
    assert_eq!(reference.len() as u64, batch.total_bases());
}

#[test]
fn every_base_is_a_nucleotide_code() {
    // Not a format check but a fixture check, and it bounds what may be claimed
    // elsewhere: these two phage genomes are 100% ACGT, so this file cannot
    // test N runs or IUPAC ambiguity codes. 2-bit packing is not implemented
    // partly because nothing here would catch it going wrong.
    let (buf, batch) = decoded();
    let reference = batch.compact(&buf).expect("compact");
    let unusual: Vec<u8> = reference
        .iter()
        .copied()
        .filter(|b| !matches!(b, b'A' | b'C' | b'G' | b'T'))
        .collect();
    assert!(
        unusual.is_empty(),
        "fixture gained non-ACGT bases; 2-bit packing claims may now be testable"
    );
}

#[test]
fn a_truncated_record_read_as_final_is_silently_short() {
    // The reason `at_eof` exists, stated as a test rather than a doc comment.
    // Told the buffer is the whole file when it is not, the scan cannot detect
    // the truncation -- it produces a complete-looking contig with fewer bases.
    // Nothing in FASTA makes this catchable, which is why the caller must say.
    let buf = fixture();
    let mut whole = RecordBatch::new();
    whole.decode(&buf, 0, true).unwrap();
    let cut = whole.record_offsets()[1] + 500;

    let mut wrong = RecordBatch::new();
    wrong.decode(&buf[..cut], 0, true).unwrap();

    assert_eq!(wrong.len(), 2, "it looks like two complete contigs");
    assert!(
        wrong.sequence_lengths()[1] < whole.sequence_lengths()[1],
        "and the second is short: {} vs {}",
        wrong.sequence_lengths()[1],
        whole.sequence_lengths()[1]
    );
}

#[test]
fn batching_at_contig_boundaries_does_not_change_the_answer() {
    // The driver contract. A FASTA record ends where the next '>' begins, so a
    // batch cut mid-record must report that record's start as its tail.
    let buf = fixture();
    let mut whole = RecordBatch::new();
    whole.decode(&buf, 0, true).unwrap();
    let expected = whole.sequence_lengths();

    // Cut inside the second contig. `at_eof: false` is what tells the scan the
    // trailing record may be truncated -- nothing in the bytes says so, because
    // a FASTA record does not announce its length.
    let cut = whole.record_offsets()[1] + 500;
    let mut first = RecordBatch::new();
    let tail = first.decode(&buf[..cut], 0, false).unwrap();

    assert_eq!(first.len(), 1, "only the first contig is complete");
    assert_eq!(
        tail,
        whole.record_offsets()[1],
        "the tail must point at the incomplete record's start"
    );

    let mut second = RecordBatch::new();
    second.decode(&buf, tail, true).unwrap();
    let mut lengths = first.sequence_lengths();
    lengths.extend(second.sequence_lengths());
    assert_eq!(lengths, expected, "lengths across the seam");
}
