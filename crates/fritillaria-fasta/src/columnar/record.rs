//! Record boundary discovery, a contig index, and the compaction formula.
//!
//! ```text
//! >name description      the definition line
//! ACGTACGT...            sequence, WRAPPED across lines
//! ACGTACGT...
//! ACGT
//! ```
//!
//! # Wrapping is the whole problem
//!
//! FASTQ's sequence is one line. FASTA's is wrapped — conventionally at 60, 70
//! or 80 columns — so the bases a consumer wants are not contiguous in the file.
//! Every other format in this workspace hands a GPU consumer a span it can read
//! directly; FASTA hands it bases interrupted by a newline every 70 bytes.
//!
//! # Finding records is trivial here, and that is worth saying
//!
//! Unlike FASTQ, where `@` is an ordinary quality score and the validator needs
//! two independent checks, `>` at the start of a line is unambiguous: sequence
//! lines are nucleotide or amino-acid codes and never begin with `>`. There is
//! no decoy case to defend against, no speculative scan, and no tiling proof —
//! which is why this module is much shorter than
//! [`fritillaria_fastq::columnar`](https://docs.rs/fritillaria-fastq).
//!
//! # The compaction formula
//!
//! With **uniform** wrapping, removing the newlines needs no scan and no prefix
//! sum. For a source index `i` measured from the start of a record's sequence
//! span, with `line_width = line_bases + 1`:
//!
//! ```text
//! skip byte i        if  src[i] == b'\n'
//! destination of i    =  i - i / line_width
//! ```
//!
//! O(1) per byte, one thread per byte, no cooperation. That only works because
//! the wrapping is uniform — which is exactly the condition `samtools faidx`
//! requires before it will index a FASTA at all, so the restriction costs
//! nothing that htslib does not also refuse.
//!
//! **Newlines are found by looking, not by arithmetic**, and the arithmetic
//! version is a trap worth naming. `i % line_width == line_bases` identifies
//! every *interior* newline and misses the last one of each record, because the
//! final line is short and its terminator therefore sits at an irregular
//! offset. That is not an edge case — it is every record with more than one
//! line. A test on a three-line record caught it immediately; the formula had
//! looked obviously right.
//!
//! Non-uniform wrapping is *detected*, not mis-parsed: [`RecordBounds`] reports
//! it and the caller falls back to a byte-wise copy.

use fritillaria_core::{Error, Result};

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "fasta",
        position: position as u64,
        reason: reason.into(),
    }
}

/// Offset just past the next newline at or after `pos`, or `None`.
fn line_end(buf: &[u8], pos: usize) -> Option<usize> {
    memchr::memchr(b'\n', buf.get(pos..)?).map(|i| pos + i)
}

/// Whether the bytes at `pos` begin a record.
///
/// `>` and nothing else. Deliberately not called `looks_like_a_record` as in
/// the other formats: this is a verdict rather than a hypothesis, because a
/// sequence line cannot start with `>`.
#[must_use]
pub fn is_record_start(buf: &[u8], pos: usize) -> bool {
    buf.get(pos) == Some(&b'>')
}

/// A contig's layout, mirroring the five columns of a `.fai` index.
///
/// Named to match `samtools faidx` output because that is the oracle this is
/// tested against, and because a caller who knows `.fai` already knows this.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecordBounds {
    /// Start of the sequence, relative to the record start. The definition
    /// occupies `1 .. sequence_start - 1`, the leading `>` excluded.
    pub sequence_start: u32,
    /// Bases in the sequence, **newlines excluded**. The `.fai` LENGTH column.
    pub sequence_len: u64,
    /// Bases per line, ignoring the final short line. The `.fai` LINEBASES.
    pub line_bases: u32,
    /// Bytes per line, i.e. `line_bases + 1`. The `.fai` LINEWIDTH.
    pub line_width: u32,
    /// One past the record's last byte, relative to its start.
    pub record_end: u64,
    /// Whether every interior line is exactly `line_bases` long.
    ///
    /// `false` means the compaction formula does not apply and a caller must
    /// copy byte-wise. `samtools faidx` refuses to index such a file at all, so
    /// this is the case htslib calls unindexable rather than an exotic one.
    pub uniform: bool,
}

impl RecordBounds {
    /// Bytes the sequence occupies in the file, interior newlines included.
    #[must_use]
    pub fn sequence_span(&self) -> u64 {
        self.record_end - u64::from(self.sequence_start)
    }

    /// Where a **non-newline** source byte at `i` lands once compacted.
    ///
    /// `i` counts from the start of the sequence span. Only valid when
    /// [`uniform`](Self::uniform); the caller is expected to have checked, and
    /// to have established that `i` is not a newline.
    ///
    /// Newlines are found by looking at the byte, not by arithmetic. The
    /// tempting test — `i % line_width == line_bases` — identifies every
    /// *interior* newline and misses the last one, because the final line of a
    /// record is short and its terminator therefore sits at an irregular
    /// offset. That is not a rare case: it is every record. Checking the byte
    /// costs nothing, since a compaction has to load it to copy it anyway.
    #[must_use]
    pub fn compacted_index(&self, i: u64) -> u64 {
        i - i / u64::from(self.line_width)
    }
}

/// Measures the record beginning at `pos`, or `None` if one does not.
pub(crate) fn bounds_at(buf: &[u8], pos: usize) -> Option<RecordBounds> {
    if !is_record_start(buf, pos) {
        return None;
    }

    let definition_end = line_end(buf, pos)?;
    let sequence_start = definition_end + 1;

    // The record runs to the next '>' at a line start, or to the end.
    let mut at = sequence_start;
    let mut lengths: Vec<usize> = Vec::new();
    let mut sequence_len: u64 = 0;

    while at < buf.len() {
        if buf[at] == b'>' {
            break;
        }
        let end = line_end(buf, at).unwrap_or(buf.len());
        let len = end - at;
        lengths.push(len);
        sequence_len += len as u64;
        at = if end < buf.len() { end + 1 } else { end };
    }

    // Blank lines are legal separators and real files use them: NCBI writes
    // `\n\n>` between records, and `samtools faidx` indexes such a file without
    // complaint. They contribute no bases, so they must not count toward the
    // wrapping either — counting one made the whole fixture look non-uniform
    // and would have sent every contig down the byte-wise fallback.
    lengths.retain(|&l| l > 0);

    // htslib's convention: LINEBASES is the first line's length, and every line
    // but the last must match it. A single-line record is trivially uniform.
    let line_bases = lengths.first().copied().unwrap_or(0);
    let uniform = lengths.len() < 2
        || lengths[..lengths.len() - 1]
            .iter()
            .all(|&l| l == line_bases);

    Some(RecordBounds {
        sequence_start: (sequence_start - pos) as u32,
        sequence_len,
        line_bases: line_bases as u32,
        line_width: (line_bases + 1) as u32,
        record_end: (at - pos) as u64,
        uniform,
    })
}

/// Walks record boundaries, returning the start offset of each.
///
/// # Why this takes `at_eof`, when no other format's scan does
///
/// A BAM, BCF or FASTQ record announces its own length, so a scan can tell a
/// complete record from a truncated one by looking. **A FASTA record cannot.**
/// It ends where the next `>` begins, or at the end of the data — and those are
/// indistinguishable from inside the buffer. `>c1\nACGT` is either a complete
/// four-base contig or the first four bases of a longer one, and nothing in the
/// bytes says which.
///
/// So the caller has to say. `at_eof` means "this buffer ends the file":
///
/// - `true` — every record is complete and the tail is `buf.len()`.
/// - `false` — the final record is assumed truncated. It is left out of the
///   offsets and its start is returned as the tail, for the caller to carry.
///
/// Guessing either way would be wrong somewhere: assume complete and a batched
/// reader silently truncates a contig at every seam; assume truncated and a
/// whole-file read drops its last contig.
pub fn scan_records(buf: &[u8], start: usize, at_eof: bool) -> Result<(Vec<usize>, usize)> {
    let mut offsets = Vec::new();
    let mut pos = start;

    while pos < buf.len() {
        if buf[pos] == b'\n' {
            pos += 1;
            continue;
        }
        if !is_record_start(buf, pos) {
            return Err(malformed(
                pos,
                format!(
                    "expected a record to start with '>', found {:?}",
                    buf[pos] as char
                ),
            ));
        }
        let Some(bounds) = bounds_at(buf, pos) else {
            break;
        };
        offsets.push(pos);
        pos += bounds.record_end as usize;
    }

    if !at_eof && let Some(&last) = offsets.last() {
        // The final record may be cut off; carry it rather than truncate it.
        offsets.pop();
        return Ok((offsets, last));
    }

    Ok((offsets, pos))
}

/// A zero-copy view over one FASTA record.
///
/// The sequence is **not** contiguous: use [`Record::compact_into`] to get the
/// bases without newlines, or iterate the lines.
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    buf: &'a [u8],
    bounds: RecordBounds,
}

impl<'a> Record<'a> {
    /// Wraps the bytes of a single record.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        let bounds = bounds_at(buf, 0).ok_or_else(|| malformed(0, "not a FASTA record"))?;
        Ok(Self { buf, bounds })
    }

    /// The contig layout.
    #[must_use]
    pub fn bounds(&self) -> RecordBounds {
        self.bounds
    }

    /// The whole definition line, `>` excluded.
    #[must_use]
    pub fn definition(&self) -> &'a [u8] {
        &self.buf[1..self.bounds.sequence_start as usize - 1]
    }

    /// The contig name: the definition up to the first whitespace.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        let definition = self.definition();
        let end = definition
            .iter()
            .position(u8::is_ascii_whitespace)
            .unwrap_or(definition.len());
        &definition[..end]
    }

    /// Everything after the first whitespace of the definition, or empty.
    #[must_use]
    pub fn description(&self) -> &'a [u8] {
        let definition = self.definition();
        match definition.iter().position(u8::is_ascii_whitespace) {
            Some(i) => &definition[i + 1..],
            None => &[],
        }
    }

    /// Bases in the sequence, newlines excluded.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.bounds.sequence_len
    }

    /// Whether the record carries no sequence.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bounds.sequence_len == 0
    }

    /// The sequence span as it sits in the file, newlines included.
    #[must_use]
    pub fn wrapped_sequence(&self) -> &'a [u8] {
        &self.buf[self.bounds.sequence_start as usize..self.bounds.record_end as usize]
    }

    /// Copies the bases into `dst`, dropping newlines.
    ///
    /// The CPU reference for `fasta_compact`. Uses the arithmetic formula when
    /// the wrapping is uniform and falls back to a byte-wise filter when it is
    /// not — the same two paths the kernel takes.
    ///
    /// # Errors
    ///
    /// If `dst` is shorter than [`len`](Self::len).
    pub fn compact_into(&self, dst: &mut [u8]) -> Result<()> {
        let n = self.bounds.sequence_len as usize;
        if dst.len() < n {
            return Err(malformed(
                0,
                format!("destination holds {} bytes, need {n}", dst.len()),
            ));
        }
        let src = self.wrapped_sequence();

        if self.bounds.uniform {
            for (i, &byte) in src.iter().enumerate() {
                if byte == b'\n' {
                    continue;
                }
                dst[self.bounds.compacted_index(i as u64) as usize] = byte;
            }
        } else {
            let mut at = 0;
            for &byte in src {
                if byte != b'\n' {
                    dst[at] = byte;
                    at += 1;
                }
            }
        }
        Ok(())
    }

    /// The bases as an owned contiguous buffer.
    ///
    /// Convenience over [`compact_into`](Self::compact_into); allocates.
    pub fn to_sequence(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.bounds.sequence_len as usize];
        self.compact_into(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds a record with `sequence` wrapped at `width` bases per line.
    pub(crate) fn build_record(name: &[u8], sequence: &[u8], width: usize) -> Vec<u8> {
        let mut out = vec![b'>'];
        out.extend_from_slice(name);
        out.push(b'\n');
        for chunk in sequence.chunks(width) {
            out.extend_from_slice(chunk);
            out.push(b'\n');
        }
        out
    }

    #[test]
    fn measures_a_wrapped_record_the_way_faidx_does() {
        let buf = build_record(b"c0", b"ACGTACGTAC", 4);
        let record = Record::new(&buf).unwrap();
        let b = record.bounds();

        assert_eq!(record.name(), b"c0");
        assert_eq!(b.sequence_len, 10, "newlines must not count as bases");
        assert_eq!(b.line_bases, 4);
        assert_eq!(b.line_width, 5);
        assert!(
            b.uniform,
            "4,4,2 is uniform: only the last line may be short"
        );
    }

    #[test]
    fn compaction_drops_the_newlines() {
        let buf = build_record(b"c0", b"ACGTACGTAC", 4);
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.to_sequence().unwrap(), b"ACGTACGTAC");
        assert!(
            record.wrapped_sequence().contains(&b'\n'),
            "source is wrapped"
        );
    }

    #[test]
    fn the_arithmetic_and_bytewise_paths_agree() {
        // The formula only applies to uniform wrapping, so the two paths must
        // produce the same bytes wherever both are valid — otherwise the kernel
        // and its fallback would disagree.
        let sequence: Vec<u8> = (0..250).map(|i| b"ACGT"[i % 4]).collect();
        for width in [1usize, 2, 7, 60, 70, 249, 250, 300] {
            let buf = build_record(b"c0", &sequence, width);
            let record = Record::new(&buf).unwrap();
            assert!(record.bounds().uniform, "width {width}");

            let mut bytewise = Vec::with_capacity(sequence.len());
            bytewise.extend(record.wrapped_sequence().iter().filter(|&&b| b != b'\n'));

            assert_eq!(record.to_sequence().unwrap(), bytewise, "width {width}");
            assert_eq!(record.to_sequence().unwrap(), sequence, "width {width}");
        }
    }

    #[test]
    fn non_uniform_wrapping_is_detected_not_mis_parsed() {
        // htslib refuses to index such a file. We must not silently apply the
        // arithmetic formula to it, which would interleave the bases wrongly.
        let mut buf = Vec::from(&b">c0\n"[..]);
        buf.extend_from_slice(b"ACGTACGT\n"); // 8
        buf.extend_from_slice(b"ACG\n"); // 3 — short, and not the last line
        buf.extend_from_slice(b"ACGTACGT\n"); // 8
        let record = Record::new(&buf).unwrap();

        assert!(!record.bounds().uniform);
        assert_eq!(record.bounds().sequence_len, 19);
        // The byte-wise path still gets it right.
        assert_eq!(record.to_sequence().unwrap(), b"ACGTACGTACGACGTACGT");
    }

    #[test]
    fn a_blank_line_between_records_does_not_break_the_wrapping() {
        // Real NCBI FASTA separates records with a blank line -- the committed
        // `testdata/controls.fa` has `\n\n>` -- and `samtools faidx` indexes
        // such a file happily. Counting that empty line as a sequence line made
        // every contig look non-uniform and would have sent the whole file down
        // the byte-wise fallback. Found by the real fixture; pinned here too, so
        // it is covered without needing a 55 KB file.
        let mut buf = Vec::from(&b">a\nACGT\nACGT\nAC\n\n"[..]);
        buf.extend_from_slice(b">b\nTTTT\n");

        let (offsets, _) = scan_records(&buf, 0, true).unwrap();
        assert_eq!(offsets.len(), 2);

        let first = Record::new(&buf[offsets[0]..]).unwrap();
        assert_eq!(first.bounds().sequence_len, 10, "blank line adds no bases");
        assert!(
            first.bounds().uniform,
            "4,4,2 is uniform; the blank is not a line"
        );
        assert_eq!(first.to_sequence().unwrap(), b"ACGTACGTAC");
    }

    #[test]
    fn splits_name_from_description() {
        let buf = build_record(b"NC_001422.1 phage phiX174, complete genome", b"ACGT", 70);
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.name(), b"NC_001422.1");
        assert_eq!(record.description(), b"phage phiX174, complete genome");
    }

    #[test]
    fn scans_several_contigs() {
        let mut buf = build_record(b"c0", b"ACGTACGT", 4);
        buf.extend_from_slice(&build_record(b"c1", b"TTTT", 4));
        buf.extend_from_slice(&build_record(b"c2", b"GG", 4));

        let (offsets, tail) = scan_records(&buf, 0, true).unwrap();
        assert_eq!(offsets.len(), 3);
        assert_eq!(tail, buf.len());

        let names: Vec<_> = offsets
            .iter()
            .map(|&o| Record::new(&buf[o..]).unwrap().name().to_vec())
            .collect();
        assert_eq!(names, vec![b"c0".to_vec(), b"c1".to_vec(), b"c2".to_vec()]);
    }

    #[test]
    fn a_record_ends_where_the_next_begins() {
        let mut buf = build_record(b"c0", b"ACGTACGT", 4);
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"c1", b"TTTT", 4));

        let record = Record::new(&buf).unwrap();
        assert_eq!(
            record.bounds().record_end as usize,
            boundary,
            "a record must not swallow the next one's definition"
        );
        assert_eq!(record.to_sequence().unwrap(), b"ACGTACGT");
    }

    #[test]
    fn a_final_line_without_a_newline_is_legal() {
        let mut buf = Vec::from(&b">c0\nACGT\nAC"[..]);
        let record = Record::new(&buf).unwrap();
        assert_eq!(record.bounds().sequence_len, 6);
        assert_eq!(record.to_sequence().unwrap(), b"ACGTAC");
        buf.clear();
    }

    #[test]
    fn an_empty_record_is_legal() {
        let buf = Vec::from(&b">c0\n"[..]);
        let record = Record::new(&buf).unwrap();
        assert!(record.is_empty());
        assert_eq!(record.to_sequence().unwrap(), b"");
    }

    #[test]
    fn resuming_at_a_non_record_offset_is_an_error_not_a_silent_skip() {
        let buf = build_record(b"c0", b"ACGT", 4);
        assert!(scan_records(&buf, 2, true).is_err());
    }
}
