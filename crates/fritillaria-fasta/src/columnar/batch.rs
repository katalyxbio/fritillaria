//! The contig index, and the compacted reference it describes.
//!
//! Two things come out of a FASTA, and they have very different sizes: a
//! handful of numbers per contig, and the entire genome. This type holds the
//! first and describes where the second lives.

use fritillaria_core::Result;

use crate::columnar::record::{Record, RecordBounds, scan_records};

/// Contigs decoded into columns, with their offsets into a compacted reference.
///
/// Every column has the same length: one entry per contig, in file order.
#[derive(Clone, Debug, Default)]
pub struct RecordBatch {
    record_offsets: Vec<usize>,
    bounds: Vec<RecordBounds>,
    /// Where each contig's bases begin in the compacted buffer — the exclusive
    /// prefix sum of the lengths.
    sequence_offsets: Vec<u64>,
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
        self.sequence_offsets.clear();
    }

    /// Number of contigs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.record_offsets.len()
    }

    /// Whether the batch holds no contigs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.record_offsets.is_empty()
    }

    /// Start offset of each contig's record in the source buffer.
    #[must_use]
    pub fn record_offsets(&self) -> &[usize] {
        &self.record_offsets
    }

    /// Each contig's layout, mirroring a `.fai` row.
    #[must_use]
    pub fn bounds(&self) -> &[RecordBounds] {
        &self.bounds
    }

    /// Where each contig's bases begin in the compacted reference.
    #[must_use]
    pub fn sequence_offsets(&self) -> &[u64] {
        &self.sequence_offsets
    }

    /// Bases per contig, newlines excluded — the `.fai` LENGTH column.
    #[must_use]
    pub fn sequence_lengths(&self) -> Vec<u64> {
        self.bounds.iter().map(|b| b.sequence_len).collect()
    }

    /// Total bases across every contig: the size of the compacted reference.
    #[must_use]
    pub fn total_bases(&self) -> u64 {
        self.sequence_offsets
            .last()
            .zip(self.bounds.last())
            .map_or(0, |(offset, bounds)| offset + bounds.sequence_len)
    }

    /// Whether every contig wraps uniformly, so the arithmetic compaction
    /// applies to all of them.
    ///
    /// A `false` here is what `samtools faidx` would refuse to index.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.bounds.iter().all(|b| b.uniform)
    }

    /// Builds a batch from columns decoded elsewhere.
    ///
    /// # Panics
    ///
    /// If the columns are not the same length; they are parallel arrays by
    /// definition.
    #[must_use]
    pub fn from_columns(record_offsets: Vec<usize>, bounds: Vec<RecordBounds>) -> Self {
        assert_eq!(
            record_offsets.len(),
            bounds.len(),
            "columns must be parallel arrays of equal length"
        );
        let mut batch = Self {
            record_offsets,
            bounds,
            sequence_offsets: Vec::new(),
        };
        batch.rebuild_offsets();
        batch
    }

    fn rebuild_offsets(&mut self) {
        self.sequence_offsets.clear();
        self.sequence_offsets.reserve(self.bounds.len());
        let mut at = 0u64;
        for bounds in &self.bounds {
            self.sequence_offsets.push(at);
            at += bounds.sequence_len;
        }
    }

    /// Decodes every contig in `buf` starting at `start`.
    ///
    /// Returns the offset of the first incomplete record, which the caller
    /// carries into the next batch.
    ///
    /// `at_eof` says whether this buffer ends the file. It is required because
    /// a FASTA record does not announce its length, so a truncated final record
    /// is indistinguishable from a complete one — see [`scan_records`].
    pub fn decode(&mut self, buf: &[u8], start: usize, at_eof: bool) -> Result<usize> {
        self.clear();

        let (offsets, tail) = scan_records(buf, start, at_eof)?;
        self.record_offsets.reserve(offsets.len());
        self.bounds.reserve(offsets.len());

        for &offset in &offsets {
            let record = Record::new(&buf[offset..])?;
            self.record_offsets.push(offset);
            self.bounds.push(record.bounds());
        }
        self.rebuild_offsets();

        Ok(tail)
    }

    /// A [`Record`] view over contig `index`, backed by `buf`.
    #[must_use]
    pub fn record<'a>(&self, buf: &'a [u8], index: usize) -> Option<Record<'a>> {
        let offset = *self.record_offsets.get(index)?;
        Record::new(buf.get(offset..)?).ok()
    }

    /// Iterates contig views over `buf`.
    pub fn records<'a>(&'a self, buf: &'a [u8]) -> impl Iterator<Item = Record<'a>> + 'a {
        (0..self.len()).filter_map(move |i| self.record(buf, i))
    }

    /// Compacts every contig into one contiguous buffer.
    ///
    /// The CPU reference for the device path, and the thing a GPU aligner
    /// actually wants: the whole reference with no newlines in it, plus
    /// [`sequence_offsets`](Self::sequence_offsets) saying where each contig
    /// starts.
    pub fn compact(&self, buf: &[u8]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.total_bases() as usize];
        for i in 0..self.len() {
            let Some(record) = self.record(buf, i) else {
                continue;
            };
            let at = self.sequence_offsets[i] as usize;
            let len = self.bounds[i].sequence_len as usize;
            record.compact_into(&mut out[at..at + len])?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columnar::record::tests::build_record;

    fn buffer(specs: &[(&[u8], &[u8])], width: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        for (name, sequence) in specs {
            buf.extend_from_slice(&build_record(name, sequence, width));
        }
        buf
    }

    #[test]
    fn columns_are_parallel_and_equal_length() {
        let buf = buffer(&[(b"a", b"ACGTACGT"), (b"b", b"TTTT"), (b"c", b"GG")], 4);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0, true).unwrap();

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.bounds().len(), 3);
        assert_eq!(batch.sequence_offsets().len(), 3);
    }

    #[test]
    fn sequence_offsets_are_the_exclusive_prefix_sum() {
        let buf = buffer(&[(b"a", b"ACGTACGT"), (b"b", b"TTTT"), (b"c", b"GG")], 4);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0, true).unwrap();

        assert_eq!(batch.sequence_lengths(), vec![8, 4, 2]);
        assert_eq!(batch.sequence_offsets(), &[0, 8, 12]);
        assert_eq!(batch.total_bases(), 14);
    }

    #[test]
    fn compaction_concatenates_the_contigs_with_no_newlines() {
        let buf = buffer(&[(b"a", b"ACGTACGT"), (b"b", b"TTTT"), (b"c", b"GG")], 4);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0, true).unwrap();

        let reference = batch.compact(&buf).unwrap();
        assert_eq!(reference, b"ACGTACGTTTTTGG");
        assert!(!reference.contains(&b'\n'));
        assert_eq!(reference.len(), batch.total_bases() as usize);
    }

    #[test]
    fn each_contig_is_findable_by_its_offset_and_length() {
        // The contract a consumer relies on: offsets and lengths must slice the
        // compacted buffer back into exactly the original sequences.
        let specs: &[(&[u8], &[u8])] = &[(b"a", b"ACGTACGT"), (b"b", b"TTTT"), (b"c", b"GG")];
        let buf = buffer(specs, 4);
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0, true).unwrap();
        let reference = batch.compact(&buf).unwrap();

        for (i, (_, expected)) in specs.iter().enumerate() {
            let at = batch.sequence_offsets()[i] as usize;
            let len = batch.bounds()[i].sequence_len as usize;
            assert_eq!(&reference[at..at + len], *expected, "contig {i}");
        }
    }

    #[test]
    fn a_non_uniform_contig_is_reported_and_still_compacts() {
        let mut buf = Vec::from(&b">a\nACGTACGT\nACG\nACGTACGT\n"[..]);
        buf.extend_from_slice(&build_record(b"b", b"TTTT", 4));
        let mut batch = RecordBatch::new();
        batch.decode(&buf, 0, true).unwrap();

        assert!(!batch.is_uniform(), "the first contig wraps irregularly");
        assert_eq!(batch.compact(&buf).unwrap(), b"ACGTACGTACGACGTACGTTTTT");
    }

    #[test]
    fn decode_clears_previous_contents() {
        let mut batch = RecordBatch::new();
        batch
            .decode(&buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT")], 4), 0, true)
            .unwrap();
        batch.decode(&buffer(&[(b"x", b"AC")], 4), 0, true).unwrap();
        assert_eq!(batch.len(), 1, "stale contigs from the previous batch");
        assert_eq!(batch.total_bases(), 2);
    }

    #[test]
    fn empty_buffer_decodes_to_an_empty_batch() {
        let mut batch = RecordBatch::new();
        assert_eq!(batch.decode(&[], 0, true).unwrap(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.total_bases(), 0);
        assert!(batch.compact(&[]).unwrap().is_empty());
    }
}
