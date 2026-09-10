//! Columnar field decode: the BCF counterpart of `fritillaria-bam`'s.
//!
//! Fixed-width site-core fields land in dense, separately-addressable columns
//! rather than in per-record structs. A GPU consumer wants `&[i32]` of
//! positions, not a `Vec<Record>` it has to re-scatter.
//!
//! # What is and is not a column
//!
//! The 24-byte site core is eight fixed-width fields, and those become columns.
//! Everything after it — ID, alleles, FILTER, INFO, and the genotype block —
//! is variable-length, stays in the source buffer, and is reached through
//! [`RecordBounds`] offsets.
//!
//! That is the same decision BAM's path made, for the same reason: copying the
//! variable fields into columns would roughly double memory and add a pass over
//! the largest thing in the pipeline. It matters more here. A BCF genotype
//! block is `n_sample × n_fmt` values and is usually the overwhelming majority
//! of the record — a 2504-sample panel record is tens of kilobytes of genotypes
//! behind a 24-byte core — so materialising it is not a modest overhead but the
//! whole file again.
//!
//! # Why the bounds cost anything
//!
//! In BAM the variable fields are found by arithmetic on the core: the name is
//! `l_read_name` bytes, the CIGAR `n_cigar_op` u32s, and so on. In BCF they are
//! not. ID, each allele and FILTER are BCF2 typed values whose lengths live in
//! their own descriptors, so finding where INFO starts means *decoding*
//! `2 + n_allele` values in sequence. That walk is what a decode kernel thread
//! actually spends its time on; see [`Record::bounds`].

use fritillaria_core::Result;

use crate::columnar::record::{Record, RecordBounds, scan_records};

/// Fixed-width BCF site-core fields, decoded into columns.
///
/// Every column has the same length: one entry per record, in file order.
#[derive(Clone, Debug, Default)]
pub struct RecordBatch {
    chromosome_id: Vec<i32>,
    position: Vec<i32>,
    reference_span: Vec<i32>,
    quality: Vec<u32>,
    info_count: Vec<u16>,
    allele_count: Vec<u16>,
    sample_count: Vec<u32>,
    format_count: Vec<u8>,
    record_offsets: Vec<usize>,
    bounds: Vec<RecordBounds>,
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
        self.chromosome_id.clear();
        self.position.clear();
        self.reference_span.clear();
        self.quality.clear();
        self.info_count.clear();
        self.allele_count.clear();
        self.sample_count.clear();
        self.format_count.clear();
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

    column!(
        chromosome_id,
        i32,
        "Contig indices into the header dictionary, not names."
    );
    column!(
        position,
        i32,
        "**0-based** leftmost positions. VCF text shows these 1-based."
    );
    column!(reference_span, i32, "Spans on the reference (`rlen`).");
    column!(info_count, u16, "Number of INFO fields per record.");
    column!(
        allele_count,
        u16,
        "Number of alleles, REF included, so always at least 1."
    );
    column!(
        sample_count,
        u32,
        "Number of samples; must equal the header's."
    );
    column!(format_count, u8, "Number of FORMAT keys.");
    column!(
        record_offsets,
        usize,
        "Start offset of each record in the source buffer."
    );
    column!(
        bounds,
        RecordBounds,
        "Where each record's variable-length sections begin."
    );

    /// QUAL as raw bits, one per record.
    ///
    /// Deliberately not `f32`. Missing QUAL is the signalling NaN `0x7F800001`
    /// and BCF permits a genuine NaN QUAL, so the two are only distinguishable
    /// as bit patterns — storing `f32` would let a `==` comparison anywhere
    /// downstream quietly conflate them. Use [`Self::quality_at`] for the
    /// decoded value.
    #[must_use]
    pub fn quality_bits(&self) -> &[u32] {
        &self.quality
    }

    /// QUAL for one record, or `None` when it is missing.
    #[must_use]
    pub fn quality_at(&self, index: usize) -> Option<f32> {
        let bits = *self.quality.get(index)?;
        crate::columnar::typed::Float::classify(bits).value()
    }

    /// Builds a batch from columns decoded elsewhere.
    ///
    /// Exists for the device path: `DeviceRecordBatch::to_host` rebuilds exactly
    /// this shape so the two can be compared field by field.
    ///
    /// # Panics
    ///
    /// If the columns are not all the same length. They are parallel arrays by
    /// definition, and a caller that gets that wrong has already lost the
    /// record-to-value correspondence this type exists to maintain.
    #[must_use]
    #[allow(clippy::too_many_arguments, reason = "one parameter per column")]
    pub fn from_columns(
        chromosome_id: Vec<i32>,
        position: Vec<i32>,
        reference_span: Vec<i32>,
        quality: Vec<u32>,
        info_count: Vec<u16>,
        allele_count: Vec<u16>,
        sample_count: Vec<u32>,
        format_count: Vec<u8>,
        record_offsets: Vec<usize>,
        bounds: Vec<RecordBounds>,
    ) -> Self {
        let n = record_offsets.len();
        assert!(
            chromosome_id.len() == n
                && position.len() == n
                && reference_span.len() == n
                && quality.len() == n
                && info_count.len() == n
                && allele_count.len() == n
                && sample_count.len() == n
                && format_count.len() == n
                && bounds.len() == n,
            "columns must be parallel arrays of equal length"
        );

        Self {
            chromosome_id,
            position,
            reference_span,
            quality,
            info_count,
            allele_count,
            sample_count,
            format_count,
            record_offsets,
            bounds,
        }
    }

    /// Appends one record's fields.
    fn push(&mut self, record: &Record<'_>, offset: usize, bounds: RecordBounds) {
        self.chromosome_id.push(record.chromosome_id());
        self.position.push(record.position());
        self.reference_span.push(record.reference_span());
        self.quality.push(record.quality_bits());
        self.info_count.push(record.info_count() as u16);
        self.allele_count.push(record.allele_count() as u16);
        self.sample_count.push(record.sample_count() as u32);
        self.format_count.push(record.format_count() as u8);
        self.record_offsets.push(offset);
        self.bounds.push(bounds);
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

        for (i, &offset) in offsets.iter().enumerate() {
            let end = offsets.get(i + 1).copied().unwrap_or(tail);
            let record = Record::new(&buf[offset..end])?;
            let bounds = record.bounds()?;
            self.push(&record, offset, bounds);
        }

        Ok(tail)
    }

    fn reserve(&mut self, n: usize) {
        self.chromosome_id.reserve(n);
        self.position.reserve(n);
        self.reference_span.reserve(n);
        self.quality.reserve(n);
        self.info_count.reserve(n);
        self.allele_count.reserve(n);
        self.sample_count.reserve(n);
        self.format_count.reserve(n);
        self.record_offsets.reserve(n);
        self.bounds.reserve(n);
    }

    /// A [`Record`] view over record `index`, backed by `buf`.
    ///
    /// This is the record-at-a-time adapter. Prefer the columns for bulk work.
    #[must_use]
    pub fn record<'a>(&self, buf: &'a [u8], index: usize) -> Option<Record<'a>> {
        let offset = *self.record_offsets.get(index)?;
        let end = offset + self.bounds.get(index)?.record_end as usize;
        Record::new(buf.get(offset..end)?).ok()
    }

    /// Iterates record views over `buf`.
    pub fn records<'a>(&'a self, buf: &'a [u8]) -> impl Iterator<Item = Record<'a>> + 'a {
        (0..self.len()).filter_map(move |i| self.record(buf, i))
    }
}
