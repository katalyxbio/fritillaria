//! Columnar field decode.
//!
//! FASTQ has almost no fixed-width fields — no flags, no positions, no mapping
//! quality. What a consumer actually wants in a column is **read length**, and
//! then offsets to the three variable-length spans.
//!
//! That makes this batch narrower than BAM's or BCF's, and it is worth being
//! explicit that the narrowness is the format's, not an omission. A GPU aligner
//! consuming raw reads needs lengths to size its work and offsets to find the
//! bases; there is nothing else in a FASTQ record to columnise.
//!
//! Sequence and quality stay in the inflated buffer, as everywhere else in this
//! workspace. For FASTQ they are essentially the entire file, so copying them
//! into columns would double VRAM to gain nothing — the bases are already
//! one byte each and already contiguous.

use fritillaria_core::Result;

use crate::columnar::record::{Record, RecordBounds, scan_records};

/// FASTQ records decoded into columns.
///
/// Every column has the same length: one entry per record, in file order.
#[derive(Clone, Debug, Default)]
pub struct RecordBatch {
    record_offsets: Vec<usize>,
    bounds: Vec<RecordBounds>,
}

impl RecordBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears every column, retaining capacity for reuse.
    pub fn clear(&mut self) {
        self.record_offsets.clear();
        self.bounds.clear();
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

    /// Start offset of each record in the source buffer.
    #[must_use]
    pub fn record_offsets(&self) -> &[usize] {
        &self.record_offsets
    }

    /// Where each record's lines begin, relative to the record.
    #[must_use]
    pub fn bounds(&self) -> &[RecordBounds] {
        &self.bounds
    }

    /// Read lengths in bases — the column a consumer sizing work actually wants.
    #[must_use]
    pub fn sequence_lengths(&self) -> Vec<u32> {
        self.bounds.iter().map(|b| b.sequence_len).collect()
    }

    /// Builds a batch from columns decoded elsewhere.
    ///
    /// Exists for the device path: `DeviceRecordBatch::to_host` rebuilds exactly
    /// this shape so the two can be compared field by field.
    ///
    /// # Panics
    ///
    /// If the columns are not the same length. They are parallel arrays by
    /// definition, and a caller that gets that wrong has already lost the
    /// record-to-value correspondence this type maintains.
    #[must_use]
    pub fn from_columns(record_offsets: Vec<usize>, bounds: Vec<RecordBounds>) -> Self {
        assert_eq!(
            record_offsets.len(),
            bounds.len(),
            "columns must be parallel arrays of equal length"
        );
        Self {
            record_offsets,
            bounds,
        }
    }

    /// Decodes every complete record in `buf` starting at `start`.
    ///
    /// Returns the offset of the first incomplete record, which the caller
    /// carries into the next batch. See [`scan_records`].
    ///
    /// The loop is the CPU reference for a kernel that assigns one thread per
    /// record — and unlike BAM's and BCF's, the *scan* it depends on is not
    /// inherently serial either. See the module docs in
    /// [`record`](crate::columnar::record).
    pub fn decode(&mut self, buf: &[u8], start: usize) -> Result<usize> {
        self.clear();

        let (offsets, tail) = scan_records(buf, start)?;
        self.record_offsets.reserve(offsets.len());
        self.bounds.reserve(offsets.len());

        for &offset in &offsets {
            let record = Record::new(&buf[offset..])?;
            self.record_offsets.push(offset);
            self.bounds.push(record.bounds());
        }

        Ok(tail)
    }

    /// A [`Record`] view over record `index`, backed by `buf`.
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
    use crate::columnar::record::tests::build_record;

    fn buffer(specs: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (name, sequence) in specs {
            buf.extend_from_slice(&build_record(name, sequence, b""));
        }
        buf
    }

    #[test]
    fn columns_are_parallel_and_equal_length() {
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGTAC"), (b"c", b"AC")]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.record_offsets().len(), 3);
        assert_eq!(batch.bounds().len(), 3);
    }

    #[test]
    fn sequence_lengths_track_the_reads() {
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGTAC"), (b"c", b"AC")]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();
        assert_eq!(batch.sequence_lengths(), vec![4, 6, 2]);
    }

    #[test]
    fn bounds_locate_the_actual_bytes() {
        // The bounds are checked by slicing at them, not by looking plausible.
        let buf = buffer(&[(b"first", b"ACGT"), (b"second", b"TTTTTT")]);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0).unwrap();

        for (i, record) in batch.records(&buf).enumerate() {
            let offset = batch.record_offsets()[i];
            let b = batch.bounds()[i];
            let seq = &buf[offset + b.sequence_start as usize..][..b.sequence_len as usize];
            let qual = &buf[offset + b.quality_start as usize..][..b.sequence_len as usize];
            assert_eq!(seq, record.sequence());
            assert_eq!(qual, record.quality_scores());
            assert_eq!(buf[offset + b.plus_start as usize], b'+');
        }
    }

    #[test]
    fn record_offsets_and_ends_tile_the_buffer() {
        // Each record must end exactly where the next begins, or a consumer
        // walking them would read a boundary's worth of the wrong record.
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGTAC"), (b"c", b"AC")]);
        let mut batch = RecordBatch::new();
        let tail = batch.decode(&buf, 0).unwrap();

        let mut at = 0usize;
        for i in 0..batch.len() {
            assert_eq!(batch.record_offsets()[i], at);
            at += batch.bounds()[i].record_end as usize;
        }
        assert_eq!(at, tail);
        assert_eq!(tail, buf.len());
    }

    #[test]
    fn decode_clears_previous_contents() {
        let mut batch = RecordBatch::new();
        batch
            .decode(&buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT")]), 0)
            .unwrap();
        batch.decode(&buffer(&[(b"x", b"AC")]), 0).unwrap();
        assert_eq!(batch.len(), 1, "stale records from the previous batch");
    }

    #[test]
    fn empty_buffer_decodes_to_an_empty_batch() {
        let mut batch = RecordBatch::new();
        assert_eq!(batch.decode(&[], 0).unwrap(), 0);
        assert!(batch.is_empty());
    }
}
