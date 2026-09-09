//! Line and field discovery over tab-delimited genomic text.
//!
//! Three phases, and the middle one is why this is not a single pass:
//!
//! 1. **Line starts** — one thread per byte, testing `buf[i - 1] == b'\n'`.
//! 2. **Field counts** — one thread per line, counting its tabs. A prefix sum
//!    over those gives each line a slice of the field table.
//! 3. **Field offsets** — one thread per line again, writing its tab positions
//!    into the slice reserved for it.
//!
//! The alternative — appending every tab atomically and sorting — was rejected
//! on a scale argument. Line starts scale with *records*, which every other
//! format in this workspace already round-trips to the host. Tabs scale with
//! *fields*, roughly an order of magnitude more, and on a many-sample VCF far
//! more than that. The prefix sum keeps the host transfer at the record scale.
//!
//! A fixed stride of `max_fields` per line would avoid the scan entirely and is
//! wrong for exactly the same reason: a 2504-sample VCF has thousands of fields
//! per record, so the table would be sized for the worst line and mostly empty.

use fritillaria_core::{Error, Result};

/// Which leading characters mark a header line, per format.
///
/// The only thing that differs between the five formats at this level. Field
/// *meanings* differ enormously; field *framing* does not differ at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dialect {
    /// A line starting with this byte is a header line.
    pub comment: u8,
    /// A second header marker, for formats that have one.
    ///
    /// BED allows `track` and `browser` lines, which start with letters and so
    /// cannot be told apart by a single byte; those are left to the format
    /// crate. This covers SAM's `@` versus VCF's `#`, which is the common case.
    pub secondary: Option<u8>,
}

impl Dialect {
    /// SAM: header lines begin with `@`.
    pub const SAM: Self = Self {
        comment: b'@',
        secondary: None,
    };
    /// VCF: `##` meta lines and the `#CHROM` column header.
    pub const VCF: Self = Self {
        comment: b'#',
        secondary: None,
    };
    /// GFF3 and GTF: `#` directives and comments.
    pub const GFF: Self = Self {
        comment: b'#',
        secondary: None,
    };
    /// BED: `#` comments. `track`/`browser` lines are the format crate's problem.
    pub const BED: Self = Self {
        comment: b'#',
        secondary: None,
    };

    /// Whether a line beginning at `pos` is a header line.
    #[must_use]
    pub fn is_header(&self, buf: &[u8], pos: usize) -> bool {
        match buf.get(pos) {
            Some(&b) => b == self.comment || Some(b) == self.secondary,
            None => false,
        }
    }
}

/// What a line is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// A header or comment line, not a record.
    Header,
    /// A data record.
    Record,
}

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "text",
        position: position as u64,
        reason: reason.into(),
    }
}

/// Finds every line in `buf` from `start`.
///
/// Returns `(offsets, kinds, tail)`. `tail` is the start of a trailing line
/// with no terminator, which a batched caller must carry forward — a line
/// without a newline is indistinguishable from one cut off by the batch edge,
/// the same ambiguity FASTA has.
///
/// Blank lines are skipped rather than reported: they carry no record and no
/// header, and every one of these formats tolerates them.
pub fn scan_lines(
    buf: &[u8],
    start: usize,
    dialect: Dialect,
) -> (Vec<usize>, Vec<LineKind>, usize) {
    let mut offsets = Vec::new();
    let mut kinds = Vec::new();
    let mut pos = start;

    while pos < buf.len() {
        if buf[pos] == b'\n' {
            pos += 1;
            continue;
        }
        let Some(end) = memchr::memchr(b'\n', &buf[pos..]).map(|i| pos + i) else {
            // No terminator: either the final line of the file or a line the
            // batch cut in half. The caller decides, via the tail.
            break;
        };
        offsets.push(pos);
        kinds.push(if dialect.is_header(buf, pos) {
            LineKind::Header
        } else {
            LineKind::Record
        });
        pos = end + 1;
    }

    (offsets, kinds, pos)
}

/// Tab positions for a run of records, in one flat array.
///
/// `starts[i]..starts[i + 1]` is record `i`'s slice of `tabs`, so the table is
/// ragged without being a `Vec<Vec<_>>` — the same layout the device uses, and
/// the reason the two can be compared directly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldTable {
    /// Absolute offset of every tab, grouped by record.
    tabs: Vec<usize>,
    /// Exclusive prefix sum of per-record tab counts; `len() + 1` entries.
    starts: Vec<usize>,
}

impl FieldTable {
    /// Number of records described.
    #[must_use]
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether the table describes no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tab offsets for record `index`.
    #[must_use]
    pub fn tabs(&self, index: usize) -> &[usize] {
        let (a, b) = (self.starts[index], self.starts[index + 1]);
        &self.tabs[a..b]
    }

    /// Fields in record `index`: one more than its tabs.
    #[must_use]
    pub fn field_count(&self, index: usize) -> usize {
        self.tabs(index).len() + 1
    }

    /// The flat tab array, for handing to a device.
    #[must_use]
    pub fn flat_tabs(&self) -> &[usize] {
        &self.tabs
    }

    /// The per-record slice boundaries.
    #[must_use]
    pub fn starts(&self) -> &[usize] {
        &self.starts
    }
}

/// Records and their fields, decoded into columns.
#[derive(Clone, Debug, Default)]
pub struct RecordBatch {
    record_offsets: Vec<usize>,
    record_ends: Vec<usize>,
    header_offsets: Vec<usize>,
    fields: FieldTable,
}

impl RecordBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears every column, retaining capacity.
    pub fn clear(&mut self) {
        self.record_offsets.clear();
        self.record_ends.clear();
        self.header_offsets.clear();
        self.fields = FieldTable::default();
    }

    /// Number of records, header lines excluded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.record_offsets.len()
    }

    /// Whether the batch holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.record_offsets.is_empty()
    }

    /// Start offset of each record.
    #[must_use]
    pub fn record_offsets(&self) -> &[usize] {
        &self.record_offsets
    }

    /// One past each record's last byte, its newline excluded.
    #[must_use]
    pub fn record_ends(&self) -> &[usize] {
        &self.record_ends
    }

    /// Start offset of each header line, in file order.
    #[must_use]
    pub fn header_offsets(&self) -> &[usize] {
        &self.header_offsets
    }

    /// The tab table.
    #[must_use]
    pub fn fields(&self) -> &FieldTable {
        &self.fields
    }

    /// Fields per record.
    #[must_use]
    pub fn field_counts(&self) -> Vec<usize> {
        (0..self.len())
            .map(|i| self.fields.field_count(i))
            .collect()
    }

    /// Builds a batch from columns decoded elsewhere, for the device path.
    ///
    /// # Panics
    ///
    /// If the columns are not parallel arrays of equal length.
    #[must_use]
    pub fn from_columns(
        record_offsets: Vec<usize>,
        record_ends: Vec<usize>,
        header_offsets: Vec<usize>,
        tabs: Vec<usize>,
        starts: Vec<usize>,
    ) -> Self {
        assert_eq!(
            record_offsets.len(),
            record_ends.len(),
            "columns must be parallel arrays of equal length"
        );
        assert_eq!(
            starts.len(),
            record_offsets.len() + 1,
            "the field table needs one more boundary than there are records"
        );
        Self {
            record_offsets,
            record_ends,
            header_offsets,
            fields: FieldTable { tabs, starts },
        }
    }

    /// Scans `buf` from `start`, filling every column.
    ///
    /// Returns the offset of the first line without a terminator, which a
    /// batched caller carries forward.
    pub fn decode(&mut self, buf: &[u8], start: usize, dialect: Dialect) -> Result<usize> {
        self.clear();

        let (offsets, kinds, tail) = scan_lines(buf, start, dialect);
        let mut tabs = Vec::new();
        let mut starts = vec![0usize];

        for (&offset, &kind) in offsets.iter().zip(&kinds) {
            let end = memchr::memchr(b'\n', &buf[offset..]).map_or(buf.len(), |i| offset + i);
            match kind {
                LineKind::Header => self.header_offsets.push(offset),
                LineKind::Record => {
                    // The one data-dependent loop, and the one a kernel runs per
                    // line rather than per byte.
                    tabs.extend(memchr::memchr_iter(b'\t', &buf[offset..end]).map(|i| offset + i));
                    starts.push(tabs.len());
                    self.record_offsets.push(offset);
                    self.record_ends.push(end);
                }
            }
        }

        self.fields = FieldTable { tabs, starts };
        Ok(tail)
    }

    /// Record `index` as raw bytes, its newline excluded.
    #[must_use]
    pub fn record<'a>(&self, buf: &'a [u8], index: usize) -> Option<&'a [u8]> {
        let (start, end) = (
            *self.record_offsets.get(index)?,
            *self.record_ends.get(index)?,
        );
        buf.get(start..end)
    }

    /// Field `field` of record `index`, as raw bytes.
    ///
    /// The tab table gives the interior boundaries; the record's own start and
    /// end give the outer two.
    #[must_use]
    pub fn field<'a>(&self, buf: &'a [u8], index: usize, field: usize) -> Option<&'a [u8]> {
        let tabs = self.fields.tabs(index);
        if field > tabs.len() {
            return None;
        }
        let start = if field == 0 {
            *self.record_offsets.get(index)?
        } else {
            tabs[field - 1] + 1
        };
        let end = if field == tabs.len() {
            *self.record_ends.get(index)?
        } else {
            tabs[field]
        };
        buf.get(start..end)
    }

    /// Every field of record `index`.
    pub fn record_fields<'a>(
        &'a self,
        buf: &'a [u8],
        index: usize,
    ) -> impl Iterator<Item = &'a [u8]> + 'a {
        (0..self.fields.field_count(index)).filter_map(move |f| self.field(buf, index, f))
    }

    /// Checks every record has the same field count, returning it.
    ///
    /// SAM, VCF, GFF and GTF are fixed-arity; BED is not, which is why this
    /// reports rather than enforces. A ragged file is usually a truncated one.
    pub fn uniform_field_count(&self) -> Result<Option<usize>> {
        let mut counts = (0..self.len()).map(|i| self.fields.field_count(i));
        let Some(first) = counts.next() else {
            return Ok(None);
        };
        for (i, count) in counts.enumerate() {
            if count != first {
                return Err(malformed(
                    self.record_offsets[i + 1],
                    format!("record {} has {count} fields, expected {first}", i + 1),
                ));
            }
        }
        Ok(Some(first))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] =
        b"@HD\tVN:1.6\n@SQ\tSN:c0\tLN:100\nr0\t0\tc0\t1\t60\t4M\nr1\t16\tc0\t9\t60\t4M\n";

    #[test]
    fn header_lines_are_separated_from_records() {
        let mut batch = RecordBatch::new();
        let tail = batch.decode(SAMPLE, 0, Dialect::SAM).unwrap();

        assert_eq!(batch.header_offsets().len(), 2, "@HD and @SQ");
        assert_eq!(batch.len(), 2, "two alignment records");
        assert_eq!(tail, SAMPLE.len());
    }

    #[test]
    fn fields_are_sliced_at_the_tabs() {
        let mut batch = RecordBatch::new();
        batch.decode(SAMPLE, 0, Dialect::SAM).unwrap();

        let fields: Vec<_> = batch.record_fields(SAMPLE, 0).collect();
        assert_eq!(fields, vec![&b"r0"[..], b"0", b"c0", b"1", b"60", b"4M"]);
        assert_eq!(batch.field_counts(), vec![6, 6]);
    }

    #[test]
    fn the_first_and_last_fields_have_no_tab_on_one_side() {
        // The off-by-one that a tab table invites: field 0 starts at the record,
        // not after a tab, and the last field ends at the record end, not at one.
        let mut batch = RecordBatch::new();
        batch.decode(SAMPLE, 0, Dialect::SAM).unwrap();

        assert_eq!(batch.field(SAMPLE, 1, 0), Some(&b"r1"[..]), "first field");
        assert_eq!(batch.field(SAMPLE, 1, 5), Some(&b"4M"[..]), "last field");
        assert_eq!(batch.field(SAMPLE, 1, 6), None, "past the last field");
    }

    #[test]
    fn a_record_excludes_its_newline() {
        let mut batch = RecordBatch::new();
        batch.decode(SAMPLE, 0, Dialect::SAM).unwrap();
        let record = batch.record(SAMPLE, 0).unwrap();
        assert!(!record.contains(&b'\n'));
        assert_eq!(record, b"r0\t0\tc0\t1\t60\t4M");
    }

    #[test]
    fn the_field_table_is_ragged_without_being_nested() {
        let buf = b"a\tb\nc\td\te\nf\n";
        let mut batch = RecordBatch::new();
        batch.decode(buf, 0, Dialect::SAM).unwrap();

        assert_eq!(batch.field_counts(), vec![2, 3, 1]);
        assert_eq!(batch.fields().starts(), &[0, 1, 3, 3]);
        assert_eq!(
            batch.fields().tabs(2),
            &[] as &[usize],
            "one field, no tabs"
        );
    }

    #[test]
    fn an_unterminated_final_line_is_left_in_the_tail() {
        // Same ambiguity FASTA has: a line without a newline is either the last
        // one or one the batch cut in half, and the bytes do not say which.
        let buf = b"a\tb\nc\td";
        let mut batch = RecordBatch::new();
        let tail = batch.decode(buf, 0, Dialect::SAM).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(tail, 4, "the unterminated line is carried");
    }

    #[test]
    fn blank_lines_are_skipped() {
        let buf = b"a\tb\n\n\nc\td\n";
        let mut batch = RecordBatch::new();
        batch.decode(buf, 0, Dialect::SAM).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn a_quote_is_ordinary_data() {
        // The mistake a scanner borrowed from CSV would make. In SAM `"` is
        // Phred+33 Q1 and appears in quality strings; treating it as a delimiter
        // reported 1,956 phantom quoted tabs in testdata/reads.sam.
        let buf = b"r0\t\"\"\"\t\"a\tb\"\n";
        let mut batch = RecordBatch::new();
        batch.decode(buf, 0, Dialect::SAM).unwrap();
        let fields: Vec<_> = batch.record_fields(buf, 0).collect();
        assert_eq!(
            fields,
            vec![&b"r0"[..], b"\"\"\"", b"\"a", b"b\""],
            "tabs split fields even inside what looks like a quoted string"
        );
    }

    #[test]
    fn the_dialect_decides_what_is_a_header() {
        // The same bytes are a record in one format and a header in another.
        let buf = b"#comment\n@notacomment\n";
        let mut sam = RecordBatch::new();
        sam.decode(buf, 0, Dialect::SAM).unwrap();
        assert_eq!(sam.header_offsets().len(), 1, "SAM: only @ is a header");
        assert_eq!(sam.len(), 1);

        let mut vcf = RecordBatch::new();
        vcf.decode(buf, 0, Dialect::VCF).unwrap();
        assert_eq!(vcf.header_offsets().len(), 1, "VCF: only # is a header");
        assert_eq!(vcf.len(), 1);
    }

    #[test]
    fn a_ragged_file_is_reported_not_accepted() {
        let buf = b"a\tb\nc\td\te\n";
        let mut batch = RecordBatch::new();
        batch.decode(buf, 0, Dialect::SAM).unwrap();
        assert!(batch.uniform_field_count().is_err());

        let buf = b"a\tb\nc\td\n";
        let mut batch = RecordBatch::new();
        batch.decode(buf, 0, Dialect::SAM).unwrap();
        assert_eq!(batch.uniform_field_count().unwrap(), Some(2));
    }

    #[test]
    fn an_empty_buffer_decodes_to_nothing() {
        let mut batch = RecordBatch::new();
        assert_eq!(batch.decode(&[], 0, Dialect::SAM).unwrap(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.uniform_field_count().unwrap(), None);
    }
}
