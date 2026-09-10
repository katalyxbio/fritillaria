//! BCF records in device memory: the columnar batch a GPU consumer receives.
//!
//! The variant-calling counterpart of `fritillaria_bam::columnar::device`, and
//! deliberately the same shape — a consumer that has written a kernel against
//! BAM's columns should recognise these without re-learning anything.
//!
//! # Why this type names no CUDA
//!
//! Every column is a [`DeviceBuffer`], core's opaque owning handle. This crate
//! can hold, move and drop one without depending on any GPU crate; only
//! `fritillaria-cuda` can recover the pointer behind it. That is what keeps
//! the workspace rule true — device memory and streams live in one crate — while
//! still letting a format crate hand back device-resident columns.
//!
//! # Layout
//!
//! The 24-byte site core becomes eight dense arrays, one element per record.
//! Everything after it stays in the inflated buffer and is described by offsets:
//!
//! ```text
//! record_offsets[i]                          where record i starts
//!   + 32                                     ID (a typed string)
//!   + alleles_start[i]                       REF then ALT, n_allele typed strings
//!   + filters_start[i]                       FILTER, a typed integer vector
//!   + info_start[i]                          INFO, n_info (key, value) pairs
//!   + genotypes_start[i] .. record_end[i]    the genotype block
//! ```
//!
//! ID needs no column: it always begins 32 bytes in, because the two length
//! prefixes and the site core are fixed-width.
//!
//! Boundaries are relative to the record start and `u32`, because a record is
//! bounded by its own `l_shared + l_indiv`; only `record_offsets` needs the full
//! 64-bit range of the batch.
//!
//! # Why the genotype block is not a column
//!
//! Same reason BAM leaves sequence and qualities in place, only more so. A BCF
//! genotype block is `n_sample × n_fmt` values and is normally the overwhelming
//! majority of the record — on a 2504-sample panel it is over 99% of the file.
//! Copying it into columns would not be a modest overhead; it would be the whole
//! file again, plus a pass over the largest thing in the pipeline.
//!
//! The consequence, which callers must know: **the
//! [`DeviceInflateBatch`](fritillaria_core::DeviceInflateBatch) these offsets
//! index has to outlive this batch.**
//!
//! # Ownership
//!
//! Dropping frees every column. A consumer whose kernel is still reading them
//! must keep the batch alive — see [`fritillaria_core::device`].

use fritillaria_core::{DeviceBuffer, Error, Result};

use crate::columnar::batch::RecordBatch;
use crate::columnar::record::RecordBounds;

/// Bytes from a record's start to its ID: the two length prefixes plus the core.
pub const ID_START: u32 = 32;

/// The columns of a [`DeviceRecordBatch`], as a backend hands them over.
///
/// A struct rather than fourteen positional arguments, so a backend cannot
/// silently transpose two columns of the same width — `position` and
/// `reference_span` are both `i32` and swapping them would produce a batch that
/// validates perfectly and is wrong.
#[derive(Debug)]
pub struct DeviceColumns {
    /// `i32` — contig indices into the header dictionary.
    pub chromosome_id: DeviceBuffer,
    /// `i32` — **0-based** positions.
    pub position: DeviceBuffer,
    /// `i32` — spans on the reference (`rlen`).
    pub reference_span: DeviceBuffer,
    /// `u32` — QUAL as raw bits; `0x7F80_0001` is missing.
    pub quality: DeviceBuffer,
    /// `u16` — number of INFO fields.
    pub info_count: DeviceBuffer,
    /// `u16` — number of alleles, REF included.
    pub allele_count: DeviceBuffer,
    /// `u32` — number of samples.
    pub sample_count: DeviceBuffer,
    /// `u8` — number of FORMAT keys.
    pub format_count: DeviceBuffer,
    /// `u64` — where each record starts in the inflated buffer.
    pub record_offsets: DeviceBuffer,
    /// `u32` — first allele, relative to the record start.
    pub alleles_start: DeviceBuffer,
    /// `u32` — FILTER, relative to the record start.
    pub filters_start: DeviceBuffer,
    /// `u32` — first INFO key, relative to the record start.
    pub info_start: DeviceBuffer,
    /// `u32` — genotype block, relative to the record start.
    pub genotypes_start: DeviceBuffer,
    /// `u32` — one past the record's last byte, relative to its start.
    pub record_end: DeviceBuffer,
}

/// Element width of each column, in the order [`DeviceColumns`] declares them.
const WIDTHS: [(&str, usize); 14] = [
    ("chromosome_id", 4),
    ("position", 4),
    ("reference_span", 4),
    ("quality", 4),
    ("info_count", 2),
    ("allele_count", 2),
    ("sample_count", 4),
    ("format_count", 1),
    ("record_offsets", 8),
    ("alleles_start", 4),
    ("filters_start", 4),
    ("info_start", 4),
    ("genotypes_start", 4),
    ("record_end", 4),
];

impl DeviceColumns {
    fn buffers(&self) -> [&DeviceBuffer; 14] {
        [
            &self.chromosome_id,
            &self.position,
            &self.reference_span,
            &self.quality,
            &self.info_count,
            &self.allele_count,
            &self.sample_count,
            &self.format_count,
            &self.record_offsets,
            &self.alleles_start,
            &self.filters_start,
            &self.info_start,
            &self.genotypes_start,
            &self.record_end,
        ]
    }
}

/// BCF records decoded into columns that never left the device.
#[derive(Debug)]
pub struct DeviceRecordBatch {
    n: usize,
    tail: usize,
    columns: DeviceColumns,
}

macro_rules! column {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[must_use]
        pub fn $name(&self) -> &DeviceBuffer {
            &self.columns.$name
        }
    };
}

impl DeviceRecordBatch {
    /// Takes ownership of a backend's columns.
    ///
    /// `tail` is the offset of the first incompletely-buffered record, which
    /// the caller carries into the next batch — the same contract as
    /// [`scan_records`](crate::columnar::scan_records). For BCF that is the
    /// common case rather than the edge case: bcftools packs BGZF blocks full,
    /// so a batch almost always ends mid-record.
    ///
    /// Every column is checked against `n` and its element width. A backend
    /// that miscounts would otherwise hand a consumer a column its kernel reads
    /// past the end of, which is silent corruption rather than an error, so it
    /// is validated here rather than trusted.
    pub fn new(n: usize, tail: usize, columns: DeviceColumns) -> Result<Self> {
        let ordinal = columns.chromosome_id.device_ordinal();

        for (buffer, (name, width)) in columns.buffers().into_iter().zip(WIDTHS) {
            let expected = n * width;
            if buffer.byte_len() != expected {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is {} bytes, expected {expected} \
                         ({n} records x {width})",
                        buffer.byte_len()
                    ),
                });
            }
            // Columns split across devices cannot be read by one kernel, and
            // finding that out inside the kernel is not a diagnosis.
            if buffer.device_ordinal() != ordinal {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is on device {} but the batch is on {ordinal}",
                        buffer.device_ordinal()
                    ),
                });
            }
        }

        Ok(Self { n, tail, columns })
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the batch holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Offset of the first incompletely-buffered record in the source batch.
    #[must_use]
    pub fn tail(&self) -> usize {
        self.tail
    }

    /// Which device these columns live on.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.columns.chromosome_id.device_ordinal()
    }

    column!(
        chromosome_id,
        "`i32` — contig indices into the header dictionary."
    );
    column!(position, "`i32` — **0-based** positions.");
    column!(reference_span, "`i32` — spans on the reference (`rlen`).");
    column!(
        quality,
        "`u32` — QUAL as raw bits; `0x7F80_0001` is missing."
    );
    column!(info_count, "`u16` — number of INFO fields.");
    column!(allele_count, "`u16` — number of alleles, REF included.");
    column!(sample_count, "`u32` — number of samples.");
    column!(format_count, "`u8` — number of FORMAT keys.");
    column!(
        record_offsets,
        "`u64` — where each record starts in the inflated buffer."
    );
    column!(
        alleles_start,
        "`u32` — first allele, relative to the record."
    );
    column!(filters_start, "`u32` — FILTER, relative to the record.");
    column!(
        info_start,
        "`u32` — first INFO key, relative to the record."
    );
    column!(
        genotypes_start,
        "`u32` — genotype block, relative to the record."
    );
    column!(record_end, "`u32` — record length, i.e. one past its end.");

    /// Copies the columns back to the host.
    ///
    /// The escape hatch and the differential-testing hook: this must equal what
    /// [`RecordBatch::decode`] produces for the same input. Expensive by
    /// construction — it is the transfer this type exists to avoid.
    pub fn to_host(&self) -> Result<RecordBatch> {
        let alleles_start = read_u32(&self.columns.alleles_start)?;
        let filters_start = read_u32(&self.columns.filters_start)?;
        let info_start = read_u32(&self.columns.info_start)?;
        let genotypes_start = read_u32(&self.columns.genotypes_start)?;
        let record_end = read_u32(&self.columns.record_end)?;

        let bounds = (0..self.n)
            .map(|i| RecordBounds {
                alleles_start: alleles_start[i],
                filters_start: filters_start[i],
                info_start: info_start[i],
                genotypes_start: genotypes_start[i],
                record_end: record_end[i],
            })
            .collect();

        Ok(RecordBatch::from_columns(
            read_i32(&self.columns.chromosome_id)?,
            read_i32(&self.columns.position)?,
            read_i32(&self.columns.reference_span)?,
            read_u32(&self.columns.quality)?,
            read_u16(&self.columns.info_count)?,
            read_u16(&self.columns.allele_count)?,
            read_u32(&self.columns.sample_count)?,
            read_u8(&self.columns.format_count)?,
            read_u64(&self.columns.record_offsets)?
                .into_iter()
                .map(|o| o as usize)
                .collect(),
            bounds,
        ))
    }
}

macro_rules! reader {
    ($name:ident, $ty:ty) => {
        fn $name(buffer: &DeviceBuffer) -> Result<Vec<$ty>> {
            const WIDTH: usize = std::mem::size_of::<$ty>();
            let bytes = buffer.to_vec()?;
            Ok(bytes
                .chunks_exact(WIDTH)
                .map(|c| <$ty>::from_le_bytes(c.try_into().expect("chunk of WIDTH bytes")))
                .collect())
        }
    };
}

reader!(read_i32, i32);
reader!(read_u16, u16);
reader!(read_u32, u32);
reader!(read_u64, u64);

fn read_u8(buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    buffer.to_vec()
}

#[cfg(test)]
mod tests {
    use fritillaria_core::device::testing::HostAlloc;

    use super::*;

    /// Builds columns for `n` records, all zeroed, using host stand-ins.
    fn columns(n: usize) -> DeviceColumns {
        let buf = |width: usize| HostAlloc::buffer(vec![0u8; n * width]);
        DeviceColumns {
            chromosome_id: buf(4),
            position: buf(4),
            reference_span: buf(4),
            quality: buf(4),
            info_count: buf(2),
            allele_count: buf(2),
            sample_count: buf(4),
            format_count: buf(1),
            record_offsets: buf(8),
            alleles_start: buf(4),
            filters_start: buf(4),
            info_start: buf(4),
            genotypes_start: buf(4),
            record_end: buf(4),
        }
    }

    #[test]
    fn accepts_columns_that_match_the_record_count() {
        let batch = DeviceRecordBatch::new(7, 123, columns(7)).unwrap();
        assert_eq!(batch.len(), 7);
        assert_eq!(batch.tail(), 123);
        assert!(!batch.is_empty());
        assert_eq!(batch.position().byte_len(), 28);
    }

    #[test]
    fn an_empty_batch_is_legal() {
        // The common case at a seam, not an error: BCF batches routinely end
        // mid-record, and a batch that contained only a partial one has no
        // records to decode.
        let batch = DeviceRecordBatch::new(0, 0, columns(0)).unwrap();
        assert!(batch.is_empty());
        assert!(batch.to_host().unwrap().is_empty());
    }

    #[test]
    fn rejects_a_column_of_the_wrong_length() {
        // The failure this prevents is a kernel indexing past a column's
        // allocation, which corrupts silently rather than erroring.
        let mut cols = columns(4);
        cols.info_count = HostAlloc::buffer(vec![0u8; 6]); // 3 records, not 4
        let err = DeviceRecordBatch::new(4, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("info_count")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_a_column_sized_with_the_wrong_element_width() {
        // `info_count` is u16; sizing it as if it were u32 is the exact mistake
        // a backend makes when a column's type changes.
        let mut cols = columns(4);
        cols.info_count = HostAlloc::buffer(vec![0u8; 4 * 4]);
        assert!(DeviceRecordBatch::new(4, 0, cols).is_err());
    }

    #[test]
    fn rejects_columns_split_across_devices() {
        let mut cols = columns(2);
        cols.position = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![0u8; 8], 1)));
        let err = DeviceRecordBatch::new(2, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("device")),
            "got {err}"
        );
    }

    #[test]
    fn to_host_reads_columns_back_in_little_endian() {
        let mut cols = columns(2);
        cols.position = HostAlloc::buffer([100i32.to_le_bytes(), (-1i32).to_le_bytes()].concat());
        cols.sample_count = HostAlloc::buffer([2504u32.to_le_bytes(), 1u32.to_le_bytes()].concat());
        cols.info_count = HostAlloc::buffer([7u16.to_le_bytes(), 0u16.to_le_bytes()].concat());

        let host = DeviceRecordBatch::new(2, 0, cols)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(host.position(), &[100, -1]);
        assert_eq!(host.sample_count(), &[2504, 1]);
        assert_eq!(host.info_count(), &[7, 0]);
    }

    #[test]
    fn to_host_reassembles_the_bounds_per_record() {
        // The bounds are five separate device columns and one host struct per
        // record, so the transpose is a place a row can silently take another
        // row's values.
        let mut cols = columns(2);
        cols.alleles_start = HostAlloc::buffer([40u32.to_le_bytes(), 44u32.to_le_bytes()].concat());
        cols.filters_start = HostAlloc::buffer([48u32.to_le_bytes(), 52u32.to_le_bytes()].concat());
        cols.info_start = HostAlloc::buffer([50u32.to_le_bytes(), 54u32.to_le_bytes()].concat());
        cols.genotypes_start =
            HostAlloc::buffer([60u32.to_le_bytes(), 70u32.to_le_bytes()].concat());
        cols.record_end = HostAlloc::buffer([90u32.to_le_bytes(), 99u32.to_le_bytes()].concat());

        let host = DeviceRecordBatch::new(2, 0, cols)
            .unwrap()
            .to_host()
            .unwrap();
        let bounds = host.bounds();

        assert_eq!(bounds[0].alleles_start, 40);
        assert_eq!(bounds[0].filters_start, 48);
        assert_eq!(bounds[0].info_start, 50);
        assert_eq!(bounds[0].genotypes_start, 60);
        assert_eq!(bounds[0].record_end, 90);

        assert_eq!(bounds[1].alleles_start, 44);
        assert_eq!(bounds[1].record_end, 99);
    }

    #[test]
    fn quality_stays_raw_bits_through_the_round_trip() {
        // Missing QUAL is a signalling NaN. If this column were f32 anywhere in
        // the path, the bit pattern would not be guaranteed to survive.
        let mut cols = columns(2);
        cols.quality = HostAlloc::buffer(
            [
                0x7F80_0001u32.to_le_bytes(),
                60.0f32.to_bits().to_le_bytes(),
            ]
            .concat(),
        );

        let host = DeviceRecordBatch::new(2, 0, cols)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(host.quality_bits()[0], 0x7F80_0001);
        assert_eq!(host.quality_at(0), None, "missing must decode to None");
        assert_eq!(host.quality_at(1), Some(60.0));
    }
}
