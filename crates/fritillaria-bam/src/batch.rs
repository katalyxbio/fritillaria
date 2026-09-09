//! Stage 5: columnar field decode.
//!
//! This is the native API and the reason to use this library: fixed-width
//! fields land in dense, separately-addressable columns rather than in
//! per-record structs. A GPU consumer wants `&[i32]` of positions, not a
//! `Vec<Record>` it has to re-scatter.
//!
//! Variable-length fields (name, CIGAR, sequence, qualities, aux) stay in the
//! source buffer and are reached through [`Record`] views built from
//! [`RecordBatch::record_offsets`]. Copying them into columns would undo the
//! zero-copy property for no benefit until there is a device-side consumer.

use fritillaria_core::Result;

use crate::record::{Record, scan_records};

/// Fixed-width BAM fields, decoded into columns.
///
/// Every column has the same length: one entry per record, in file order.
#[derive(Clone, Debug, Default)]
pub struct RecordBatch {
    reference_sequence_id: Vec<i32>,
    position: Vec<i32>,
    mapping_quality: Vec<u8>,
    flags: Vec<u16>,
    sequence_len: Vec<u32>,
    mate_reference_sequence_id: Vec<i32>,
    mate_position: Vec<i32>,
    template_length: Vec<i32>,
    record_offsets: Vec<usize>,
}

macro_rules! column {
    ($name:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        #[must_use]
        pub fn $name(&self) -> &[$ty] {
            &self.$name
        }
    };
}

impl RecordBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears every column, retaining capacity for reuse.
    pub fn clear(&mut self) {
        self.reference_sequence_id.clear();
        self.position.clear();
        self.mapping_quality.clear();
        self.flags.clear();
        self.sequence_len.clear();
        self.mate_reference_sequence_id.clear();
        self.mate_position.clear();
        self.template_length.clear();
        self.record_offsets.clear();
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.record_offsets.len()
    }

    /// Whether the batch holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.record_offsets.is_empty()
    }

    column!(
        reference_sequence_id,
        i32,
        "Reference IDs; `-1` is unplaced."
    );
    column!(position, i32, "**0-based** positions; `-1` is unplaced.");
    column!(mapping_quality, u8, "Mapping qualities.");
    column!(flags, u16, "Bitwise FLAGs.");
    column!(sequence_len, u32, "Read lengths in bases.");
    column!(mate_reference_sequence_id, i32, "Mate reference IDs.");
    column!(mate_position, i32, "Mate 0-based positions.");
    column!(template_length, i32, "Observed template lengths.");
    column!(
        record_offsets,
        usize,
        "Start offset of each record in the source buffer."
    );

    /// Builds a batch from columns decoded elsewhere.
    ///
    /// Exists for the device path: `DeviceRecordBatch::to_host` needs to
    /// rebuild exactly this shape so the two can be compared field by field.
    ///
    /// # Panics
    ///
    /// If the columns are not all the same length. They are parallel arrays by
    /// definition, and a caller that gets that wrong has already lost the
    /// record-to-value correspondence this type exists to maintain.
    #[must_use]
    #[allow(clippy::too_many_arguments, reason = "one parameter per column")]
    pub fn from_columns(
        reference_sequence_id: Vec<i32>,
        position: Vec<i32>,
        mapping_quality: Vec<u8>,
        flags: Vec<u16>,
        sequence_len: Vec<u32>,
        mate_reference_sequence_id: Vec<i32>,
        mate_position: Vec<i32>,
        template_length: Vec<i32>,
        record_offsets: Vec<usize>,
    ) -> Self {
        let n = record_offsets.len();
        assert!(
            reference_sequence_id.len() == n
                && position.len() == n
                && mapping_quality.len() == n
                && flags.len() == n
                && sequence_len.len() == n
                && mate_reference_sequence_id.len() == n
                && mate_position.len() == n
                && template_length.len() == n,
            "columns must be parallel arrays of equal length"
        );

        Self {
            reference_sequence_id,
            position,
            mapping_quality,
            flags,
            sequence_len,
            mate_reference_sequence_id,
            mate_position,
            template_length,
            record_offsets,
        }
    }

    /// Appends one record's fields.
    fn push(&mut self, record: &Record<'_>, offset: usize) {
        self.reference_sequence_id
            .push(record.reference_sequence_id());
        self.position.push(record.position());
        self.mapping_quality.push(record.mapping_quality());
        self.flags.push(record.flags());
        self.sequence_len.push(record.sequence_len() as u32);
        self.mate_reference_sequence_id
            .push(record.mate_reference_sequence_id());
        self.mate_position.push(record.mate_position());
        self.template_length.push(record.template_length());
        self.record_offsets.push(offset);
    }

    /// Decodes every complete record in `buf` starting at `start`.
    ///
    /// Returns the offset of the first incomplete record, which the caller
    /// carries into the next batch. See [`scan_records`].
    ///
    /// This is the stage that is embarrassingly parallel once boundaries are
    /// known — the loop below is the CPU reference for a kernel that assigns
    /// one thread per record.
    pub fn decode(&mut self, buf: &[u8], start: usize) -> Result<usize> {
        self.clear();

        let (offsets, tail) = scan_records(buf, start)?;
        self.reserve(offsets.len());

        for offset in offsets {
            let record = Record::new(&buf[offset..])?;
            self.push(&record, offset);
        }

        Ok(tail)
    }

    fn reserve(&mut self, n: usize) {
        self.reference_sequence_id.reserve(n);
        self.position.reserve(n);
        self.mapping_quality.reserve(n);
        self.flags.reserve(n);
        self.sequence_len.reserve(n);
        self.mate_reference_sequence_id.reserve(n);
        self.mate_position.reserve(n);
        self.template_length.reserve(n);
        self.record_offsets.reserve(n);
    }

    /// A [`Record`] view over record `index`, backed by `buf`.
    ///
    /// This is the record-at-a-time adapter. Prefer the columns for bulk work.
    #[must_use]
    pub fn record<'a>(&self, buf: &'a [u8], index: usize) -> Option<Record<'a>> {
        let offset = *self.record_offsets.get(index)?;
        Record::new(buf.get(offset..)?).ok()
    }

    /// Iterates record views over `buf`.
    pub fn records<'a>(&'a self, buf: &'a [u8]) -> impl Iterator<Item = Record<'a>> + 'a {
        (0..self.len()).filter_map(move |i| self.record(buf, i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::tests::build_record;

    fn buffer(names: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        for name in names {
            buf.extend_from_slice(&build_record(name, b"ACGT", None));
        }
        buf
    }

    #[test]
    fn columns_are_parallel_and_equal_length() {
        let buf = buffer(&[b"a", b"b", b"c"]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        assert_eq!(batch.len(), 3);
        for len in [
            batch.reference_sequence_id().len(),
            batch.position().len(),
            batch.mapping_quality().len(),
            batch.flags().len(),
            batch.sequence_len().len(),
            batch.mate_reference_sequence_id().len(),
            batch.mate_position().len(),
            batch.template_length().len(),
            batch.record_offsets().len(),
        ] {
            assert_eq!(len, 3, "columns must stay in lockstep");
        }
    }

    #[test]
    fn columns_carry_the_right_values() {
        let buf = buffer(&[b"a", b"b"]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        assert_eq!(batch.position(), &[100, 100]);
        assert_eq!(batch.mapping_quality(), &[60, 60]);
        assert_eq!(batch.flags(), &[99, 99]);
        assert_eq!(batch.sequence_len(), &[4, 4]);
        assert_eq!(batch.template_length(), &[150, 150]);
    }

    #[test]
    fn column_values_match_the_record_view() {
        // The columnar path and the adapter must never disagree; this is the
        // in-crate version of the differential test.
        let buf = buffer(&[b"first", b"second", b"third"]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        for (i, record) in batch.records(&buf).enumerate() {
            assert_eq!(record.position(), batch.position()[i]);
            assert_eq!(record.flags(), batch.flags()[i]);
            assert_eq!(record.mapping_quality(), batch.mapping_quality()[i]);
            assert_eq!(record.sequence_len() as u32, batch.sequence_len()[i]);
        }
    }

    #[test]
    fn record_views_resolve_variable_length_fields() {
        let buf = buffer(&[b"alpha", b"beta"]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        let names: Vec<_> = batch.records(&buf).map(|r| r.name().to_vec()).collect();
        assert_eq!(names, vec![b"alpha".to_vec(), b"beta".to_vec()]);
    }

    #[test]
    fn reports_the_tail_for_a_split_record() {
        let mut buf = buffer(&[b"complete"]);
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"partial", b"ACGT", None));
        buf.truncate(buf.len() - 3);

        let mut batch = RecordBatch::new();
        let tail = batch.decode(&buf, 0).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(tail, boundary);
    }

    #[test]
    fn decode_clears_previous_contents() {
        let mut batch = RecordBatch::new();
        batch.decode(&buffer(&[b"a", b"b", b"c"]), 0).unwrap();
        batch.decode(&buffer(&[b"x"]), 0).unwrap();

        assert_eq!(batch.len(), 1, "stale records from the previous batch");
    }

    #[test]
    fn empty_buffer_decodes_to_an_empty_batch() {
        let mut batch = RecordBatch::new();
        assert_eq!(batch.decode(&[], 0).unwrap(), 0);
        assert!(batch.is_empty());
    }
}
