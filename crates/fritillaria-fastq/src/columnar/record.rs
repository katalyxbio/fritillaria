//! Record boundary discovery, and a zero-copy record view.
//!
//! ```text
//! @name description      line 1, the definition
//! ACGTACGT...            line 2, the sequence
//! +[name]                line 3, the separator
//! IIIIIIII...            line 4, the quality scores
//! ```
//!
//! # Delimited, not length-prefixed — which changes everything
//!
//! BAM and BCF put a length in front of each record, so finding record *n+1*
//! requires having read record *n*. That is a serial chain, and both formats
//! needed a trick to break it for a device: BAM speculates per BGZF block,
//! BCF per candidate offset.
//!
//! FASTQ has no such chain. Records are delimited by newlines, and newlines are
//! findable in parallel with no dependency whatsoever. The scan below is
//! sequential only because it is the reference implementation; nothing about the
//! format requires it to be.
//!
//! # The difficulty moves to deciding what starts a record
//!
//! `@` does **not** identify a record start. It is Phred+33 Q31, a thoroughly
//! ordinary quality score, so it appears inside quality lines constantly — 13,159
//! times across the committed fixtures, and at the *start* of 83 quality lines in
//! `testdata/illumina.fastq.gz`.
//!
//! [`looks_like_a_record`] is what separates them, and it needs **both** of its
//! checks. Consider a decoy `@` inside a quality line: line 1 is the rest of that
//! quality line, line 2 is the next record's `@name` line, line 3 is the next
//! record's *sequence*, and line 4 is its `+` line.
//!
//! - `len(line 2) == len(line 4)` rejects every decoy in the committed fixtures,
//!   because a name line and a `+` line rarely have the same length.
//! - Line 3 starting with `+` rejects the ones where those lengths coincide. A
//!   sequence line never starts with `+`.
//!
//! **Deleting the second check passed every test in this crate** until
//! `only_the_plus_check_rejects_a_length_matched_decoy` existed — real data does
//! not supply that case, so it is built by hand. Measured over 1,626,353 byte
//! offsets of three fixtures: **zero false positives**. See
//! `docs/fastq-boundaries.md`.

use fritillaria_core::{Error, Result};

/// Bytes from a record's start to its name: past the `@`.
pub const NAME_START: u32 = 1;

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "fastq",
        position: position as u64,
        reason: reason.into(),
    }
}

/// Offset just past the next newline at or after `pos`, or `None`.
fn line_end(buf: &[u8], pos: usize) -> Option<usize> {
    memchr::memchr(b'\n', buf.get(pos..)?).map(|i| pos + i)
}

/// Whether the bytes at `pos` plausibly begin a record.
///
/// Reads three newlines and two single bytes, with no data-dependent loop
/// beyond scanning for those newlines — cheap enough to run at every candidate
/// offset in parallel, which is the whole reason it exists.
///
/// **A `true` is a verdict for well-formed FASTQ and a hypothesis in general.**
/// It was measured to admit nothing but true record starts on real ONT, HiFi and
/// Illumina data, but the guarantee comes from the tiling proof in
/// [`speculative`](crate::columnar::speculative), not from this.
#[must_use]
pub fn looks_like_a_record(buf: &[u8], pos: usize) -> bool {
    bounds_at(buf, pos).is_some()
}

/// The four line boundaries of a record at `pos`, or `None` if it is not one.
///
/// Shared by the validator and the scan so the two cannot disagree about what
/// counts as a record — the kind of drift that produced a differential oracle
/// disagreeing with itself in `fritillaria-bcf`.
pub(crate) fn bounds_at(buf: &[u8], pos: usize) -> Option<RecordBounds> {
    if buf.get(pos)? != &b'@' {
        return None;
    }

    let definition_end = line_end(buf, pos)?;
    let sequence_start = definition_end + 1;
    let sequence_end = line_end(buf, sequence_start)?;
    let plus_start = sequence_end + 1;

    if buf.get(plus_start)? != &b'+' {
        return None;
    }

    let plus_end = line_end(buf, plus_start)?;
    let quality_start = plus_end + 1;
    let sequence_len = sequence_end - sequence_start;

    // The quality *line*, measured the same way the sequence line was: to the
    // next newline, or to the end of the buffer for a file that does not end in
    // one. Then require the two lengths to agree.
    //
    // Measuring it as `quality_start + sequence_len` instead is the obvious
    // shortcut and is wrong: on the last record it lets the span run over the
    // trailing newline, so a quality line one byte short reads as complete.
    // Found by a test that expected a short quality line to be rejected.
    let quality_end = line_end(buf, quality_start).unwrap_or(buf.len());
    if quality_end - quality_start != sequence_len {
        return None;
    }

    // A record ends past its trailing newline when it has one. The final record
    // of a file may not, which is legal.
    let record_end = if quality_end < buf.len() {
        quality_end + 1
    } else {
        quality_end
    };

    Some(RecordBounds {
        sequence_start: (sequence_start - pos) as u32,
        plus_start: (plus_start - pos) as u32,
        quality_start: (quality_start - pos) as u32,
        sequence_len: sequence_len as u32,
        record_end: (record_end - pos) as u32,
    })
}

/// Offsets of a record's four lines, relative to its start.
///
/// The definition occupies `NAME_START .. sequence_start - 1`; the trailing
/// `- 1` drops the newline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecordBounds {
    /// Start of the sequence line.
    pub sequence_start: u32,
    /// Start of the `+` separator line.
    pub plus_start: u32,
    /// Start of the quality line.
    pub quality_start: u32,
    /// Bases in the sequence, which is also the length of the quality line.
    pub sequence_len: u32,
    /// One past the record's last byte, its trailing newline included.
    pub record_end: u32,
}

/// Walks record boundaries, returning the start offset of each record.
///
/// A trailing partial record is **not** an error — the batch ended mid-record.
/// Those bytes are reported through the returned tail so the caller can carry
/// them forward. For long reads that is the common case: one ONT record can be
/// larger than several BGZF blocks.
///
/// Returns `(offsets, tail)` where `tail` is the offset of the first
/// incompletely-buffered record, or `buf.len()` if the buffer ends cleanly.
pub fn scan_records(buf: &[u8], start: usize) -> Result<(Vec<usize>, usize)> {
    let mut offsets = Vec::new();
    let mut pos = start;

    while pos < buf.len() {
        // Skip a leading newline before trying to read a record.
        //
        // Not cosmetic tolerance — it closes a seam hazard. A record whose
        // quality line ends exactly at a batch boundary is complete, so the
        // scan reports a tail past it and the *next* batch begins with that
        // record's trailing newline. Without this the next scan would find '\n'
        // where it expects '@' and fail the batch. Blank lines between records
        // are also legal in some FASTQ dialects, so skipping costs nothing.
        if buf[pos] == b'\n' {
            pos += 1;
            continue;
        }

        let Some(bounds) = bounds_at(buf, pos) else {
            // Either a partial record at the end of the batch, or corruption.
            // Told apart by whether a `@` even starts here: anything else means
            // the caller resumed at an offset that is not a record start.
            if buf.get(pos) == Some(&b'@') {
                break;
            }
            return Err(malformed(
                pos,
                format!(
                    "expected a record to start with '@', found {:?}",
                    buf.get(pos).map(|b| *b as char)
                ),
            ));
        };
        offsets.push(pos);
        pos += bounds.record_end as usize;
    }

    Ok((offsets, pos))
}

/// A zero-copy view over one FASTQ record.
///
/// Borrows the inflated buffer; nothing is copied. A different type from the
/// vendored [`Record`](crate::Record), which owns its fields — see the
/// `columnar` module docs for why both exist.
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    buf: &'a [u8],
    bounds: RecordBounds,
}

impl<'a> Record<'a> {
    /// Wraps the bytes of a single record.
    ///
    /// `buf` may extend past the record; the bounds decide where it ends.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        let bounds =
            bounds_at(buf, 0).ok_or_else(|| malformed(0, "not a well-formed FASTQ record"))?;
        Ok(Self { buf, bounds })
    }

    /// The line boundaries, relative to the record start.
    #[must_use]
    pub fn bounds(&self) -> RecordBounds {
        self.bounds
    }

    /// The whole definition line, `@` excluded: name and description together.
    #[must_use]
    pub fn definition(&self) -> &'a [u8] {
        // -1 drops the newline that ends the line.
        &self.buf[NAME_START as usize..self.bounds.sequence_start as usize - 1]
    }

    /// The read name: the definition up to the first space.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        let definition = self.definition();
        match memchr::memchr(b' ', definition) {
            Some(i) => &definition[..i],
            None => definition,
        }
    }

    /// Everything after the first space of the definition, or empty.
    #[must_use]
    pub fn description(&self) -> &'a [u8] {
        let definition = self.definition();
        match memchr::memchr(b' ', definition) {
            Some(i) => &definition[i + 1..],
            None => &[],
        }
    }

    /// The sequence, unpacked — FASTQ stores one base per byte.
    #[must_use]
    pub fn sequence(&self) -> &'a [u8] {
        let start = self.bounds.sequence_start as usize;
        &self.buf[start..start + self.bounds.sequence_len as usize]
    }

    /// The quality scores, Phred+33, one byte per base.
    #[must_use]
    pub fn quality_scores(&self) -> &'a [u8] {
        let start = self.bounds.quality_start as usize;
        &self.buf[start..start + self.bounds.sequence_len as usize]
    }

    /// Bases in the read.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bounds.sequence_len as usize
    }

    /// Whether the read is empty, which is legal but unusual.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bounds.sequence_len == 0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds one record. `plus` is the text after the `+`, usually empty.
    pub(crate) fn build_record(name: &[u8], sequence: &[u8], plus: &[u8]) -> Vec<u8> {
        let mut out = vec![b'@'];
        out.extend_from_slice(name);
        out.push(b'\n');
        out.extend_from_slice(sequence);
        out.push(b'\n');
        out.push(b'+');
        out.extend_from_slice(plus);
        out.push(b'\n');
        out.extend(std::iter::repeat_n(b'I', sequence.len()));
        out.push(b'\n');
        out
    }

    #[test]
    fn parses_a_minimal_record() {
        let buf = build_record(b"r0", b"ACGT", b"");
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.name(), b"r0");
        assert_eq!(record.sequence(), b"ACGT");
        assert_eq!(record.quality_scores(), b"IIII");
        assert_eq!(record.len(), 4);
    }

    #[test]
    fn splits_name_from_description() {
        let buf = build_record(b"r0 length=4 ch=1", b"ACGT", b"");
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.name(), b"r0");
        assert_eq!(record.description(), b"length=4 ch=1");
        assert_eq!(record.definition(), b"r0 length=4 ch=1");
    }

    #[test]
    fn tolerates_a_repeated_name_after_the_plus() {
        // Legal and still written by some tools: the '+' line may repeat the
        // definition. Its length is unrelated to the sequence length, which is
        // why the validator reads the *quality* line for that comparison.
        let buf = build_record(b"r0", b"ACGTACGT", b"r0");
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.sequence(), b"ACGTACGT");
        assert_eq!(record.quality_scores(), b"IIIIIIII");
    }

    #[test]
    fn accepts_a_final_record_with_no_trailing_newline() {
        let mut buf = build_record(b"r0", b"ACGT", b"");
        buf.pop();
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.quality_scores(), b"IIII");
        assert_eq!(record.bounds().record_end as usize, buf.len());
    }

    #[test]
    fn rejects_a_quality_line_of_the_wrong_length() {
        // The second line of defence after the '+' check. A quality line that
        // disagrees with the sequence means the framing is wrong.
        let mut buf = build_record(b"r0", b"ACGT", b"");
        buf.remove(buf.len() - 2); // one quality byte short
        assert!(Record::new(&buf).is_err());
    }

    #[test]
    fn an_at_inside_a_quality_line_is_not_a_record_start() {
        // The decoy the whole validator exists for. '@' is Phred+33 Q31.
        let mut buf = build_record(b"r0", b"ACGT", b"");
        let quality_start = buf.len() - 5;
        buf[quality_start] = b'@';
        buf.extend_from_slice(&build_record(b"r1", b"TTTT", b""));

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets.len(), 2, "the decoy must not become a third record");
        assert_eq!(tail, buf.len());
        assert!(
            !looks_like_a_record(&buf, quality_start),
            "'@' at a quality-line start must be rejected"
        );
    }

    #[test]
    fn only_the_plus_check_rejects_a_length_matched_decoy() {
        // Earns the `+` check, which nothing else did.
        //
        // A mutation deleting it passed every test in this crate, including the
        // differential run over all three real fixtures. The reason: on real
        // data the *length* check happens to catch quality-line decoys too,
        // because the next record's name line and its `+` line rarely have the
        // same length. So the `+` check was never being exercised.
        //
        // Here they are contrived to match. `@XY` and `+ab` are both 3 bytes,
        // so a decoy at the `@` starting record 1's quality line sees a
        // consistent-looking record and only the `+` on line 3 -- which is
        // really record 2's *sequence* -- gives it away.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"@A\nACG\n+\n@BC\n");
        buf.extend_from_slice(b"@XY\nACGT\n+ab\nIIII\n");

        let decoy = 9;
        assert_eq!(&buf[decoy..decoy + 4], b"@BC\n", "decoy is where we think");

        assert!(
            !looks_like_a_record(&buf, decoy),
            "a length-matched decoy must still be rejected"
        );

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets, vec![0, 13], "exactly the two real records");
        assert_eq!(tail, buf.len());
    }

    #[test]
    fn scans_several_records() {
        let mut buf = Vec::new();
        for name in [&b"a"[..], b"b", b"c"] {
            buf.extend_from_slice(&build_record(name, b"ACGTAC", b""));
        }
        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets.len(), 3);
        assert_eq!(tail, buf.len());
    }

    #[test]
    fn reports_the_tail_for_a_partial_record() {
        let mut buf = build_record(b"whole", b"ACGT", b"");
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"partial", b"ACGTACGT", b""));
        buf.truncate(buf.len() - 4);

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets.len(), 1);
        assert_eq!(tail, boundary, "the partial record must be carried forward");
    }

    #[test]
    fn resuming_at_a_non_record_offset_is_an_error_not_a_silent_skip() {
        let buf = build_record(b"r0", b"ACGT", b"");
        assert!(scan_records(&buf, 3).is_err());
    }

    #[test]
    fn a_record_ending_at_a_batch_edge_resumes_cleanly() {
        // The seam this format has and the binary ones do not. A record whose
        // quality line ends exactly where the batch does is *complete* -- the
        // sequence and quality agree -- so the scan reports a tail past it, and
        // the next batch opens with that record's trailing newline.
        let whole = build_record(b"r0", b"ACGT", b"");
        let cut = whole.len() - 1; // everything but the trailing newline

        let first = &whole[..cut];
        let (offsets, tail) = scan_records(first, 0).unwrap();
        assert_eq!(offsets, vec![0], "a complete record, newline or not");
        assert_eq!(tail, cut);

        // The next batch begins with the orphaned newline, then a real record.
        let mut second = vec![b'\n'];
        second.extend_from_slice(&build_record(b"r1", b"TTTT", b""));
        let (offsets, tail) = scan_records(&second, 0).unwrap();
        assert_eq!(
            offsets,
            vec![1],
            "the leading newline must not fail the batch"
        );
        assert_eq!(tail, second.len());
    }

    #[test]
    fn a_short_quality_line_is_rejected_even_at_the_end_of_a_buffer() {
        // The bug the fix above closes: measuring quality as
        // `quality_start + sequence_len` lets the span swallow the trailing
        // newline on the final record, so one byte short reads as complete.
        let mut buf = build_record(b"r0", b"ACGT", b"");
        buf.pop(); // drop the trailing newline
        buf.pop(); // and one quality byte: 3 scores for 4 bases
        assert!(Record::new(&buf).is_err());
        assert!(!looks_like_a_record(&buf, 0));
    }

    #[test]
    fn an_empty_read_is_legal() {
        let buf = build_record(b"r0", b"", b"");
        let record = Record::new(&buf).unwrap();
        assert!(record.is_empty());
        assert_eq!(record.sequence(), b"");
        assert_eq!(scan_records(&buf, 0).unwrap().0.len(), 1);
    }
}
