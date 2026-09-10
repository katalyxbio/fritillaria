//! Records in device memory: the columnar batch a GPU consumer receives.
//!
//! Bytes in VRAM are not the deliverable. nvCOMP will hand anyone bytes; what
//! it cannot do is tell you where record 40,000 starts or what its position is.
//! This is that — parsed, columnar, and still on the device.
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
//! Fixed-width fields become dense arrays, one element per record. The
//! variable-length fields do **not**: they stay where they already are, in the
//! inflated buffer, and are described by offsets into it.
//!
//! ```text
//! record_offsets[i]                      where record i starts
//!   + NAME_START (36)                    read name
//!   + cigar_start[i]                     CIGAR, 4 bytes per op
//!   + sequence_start[i]                  sequence, 4-bit packed
//!   + quality_start[i]                   qualities
//!   + aux_start[i] .. record_end[i]      aux tag block
//! ```
//!
//! Boundaries are relative to the record start and `u32`, because a record is
//! bounded by its own `block_size`; only `record_offsets` needs the full 64-bit
//! range of the batch.
//!
//! **Copying the variable-length fields into packed payload columns was the
//! obvious design and is the wrong one here.** Sequence and qualities are the
//! bulk of a BAM, and long-read qualities are one byte per base; duplicating
//! them would roughly double VRAM for the batch and add a pass over the largest
//! thing in the pipeline. Pointing into bytes that are already resident costs
//! nothing. The consequence, which callers must know: **the
//! [`DeviceInflateBatch`](fritillaria_core::DeviceInflateBatch) these offsets
//! index has to outlive this batch.** A `compact()` that gathers dense payload
//! columns is the opt-in for consumers that genuinely want to drop the source.
//!
//! # Ownership
//!
//! Dropping frees every column. A consumer whose kernel is still reading them
//! must keep the batch alive — see [`fritillaria_core::device`].

use fritillaria_core::{DeviceBuffer, Error, Result};

use crate::columnar::batch::RecordBatch;

/// Bytes from a record's start to its read name: `block_size` plus the core.
pub const NAME_START: u32 = 36;

/// The columns of a [`DeviceRecordBatch`], as a backend hands them over.
///
/// A struct rather than fifteen positional arguments, so a backend cannot
/// silently transpose two columns of the same width — `position` and
/// `template_length` are both `i32` and swapping them would produce a batch
/// that validates perfectly and is wrong.
#[derive(Debug)]
pub struct DeviceColumns {
    /// `i32` — reference IDs; `-1` is unplaced.
    pub reference_sequence_id: DeviceBuffer,
    /// `i32` — **0-based** positions; `-1` is unplaced.
    pub position: DeviceBuffer,
    /// `u16` — bitwise FLAGs.
    pub flags: DeviceBuffer,
    /// `u8` — mapping qualities.
    pub mapping_quality: DeviceBuffer,
    /// `u32` — read lengths in bases.
    pub sequence_len: DeviceBuffer,
    /// `i32` — mate reference IDs.
    pub mate_reference_sequence_id: DeviceBuffer,
    /// `i32` — mate 0-based positions.
    pub mate_position: DeviceBuffer,
    /// `i32` — observed template lengths.
    pub template_length: DeviceBuffer,
    /// `u64` — where each record starts in the inflated buffer.
    pub record_offsets: DeviceBuffer,
    /// `u32` — CIGAR start, relative to the record start.
    pub cigar_start: DeviceBuffer,
    /// `u32` — packed sequence start, relative to the record start.
    pub sequence_start: DeviceBuffer,
    /// `u32` — qualities start, relative to the record start.
    pub quality_start: DeviceBuffer,
    /// `u32` — aux tag block start, relative to the record start.
    pub aux_start: DeviceBuffer,
    /// `u32` — one past the record's last byte, relative to its start.
    pub record_end: DeviceBuffer,
}

/// Element width of each column, in the order [`DeviceColumns`] declares them.
const WIDTHS: [(&str, usize); 14] = [
    ("reference_sequence_id", 4),
    ("position", 4),
    ("flags", 2),
    ("mapping_quality", 1),
    ("sequence_len", 4),
    ("mate_reference_sequence_id", 4),
    ("mate_position", 4),
    ("template_length", 4),
    ("record_offsets", 8),
    ("cigar_start", 4),
    ("sequence_start", 4),
    ("quality_start", 4),
    ("aux_start", 4),
    ("record_end", 4),
];

impl DeviceColumns {
    fn buffers(&self) -> [&DeviceBuffer; 14] {
        [
            &self.reference_sequence_id,
            &self.position,
            &self.flags,
            &self.mapping_quality,
            &self.sequence_len,
            &self.mate_reference_sequence_id,
            &self.mate_position,
            &self.template_length,
            &self.record_offsets,
            &self.cigar_start,
            &self.sequence_start,
            &self.quality_start,
            &self.aux_start,
            &self.record_end,
        ]
    }
}

/// BAM records decoded into columns that never left the device.
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
    /// [`scan_records`](crate::columnar::scan_records).
    ///
    /// Every column is checked against `n` and its element width. A backend
    /// that miscounts would otherwise hand a consumer a column its kernel reads
    /// past the end of, which is silent corruption rather than an error, so it
    /// is validated here rather than trusted.
    pub fn new(n: usize, tail: usize, columns: DeviceColumns) -> Result<Self> {
        let ordinal = columns.reference_sequence_id.device_ordinal();

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
        self.columns.reference_sequence_id.device_ordinal()
    }

    column!(
        reference_sequence_id,
        "`i32` — reference IDs; `-1` is unplaced."
    );
    column!(position, "`i32` — **0-based** positions; `-1` is unplaced.");
    column!(flags, "`u16` — bitwise FLAGs.");
    column!(mapping_quality, "`u8` — mapping qualities.");
    column!(sequence_len, "`u32` — read lengths in bases.");
    column!(mate_reference_sequence_id, "`i32` — mate reference IDs.");
    column!(mate_position, "`i32` — mate 0-based positions.");
    column!(template_length, "`i32` — observed template lengths.");
    column!(
        record_offsets,
        "`u64` — where each record starts in the inflated buffer."
    );
    column!(cigar_start, "`u32` — CIGAR start, relative to the record.");
    column!(
        sequence_start,
        "`u32` — packed sequence start, relative to the record."
    );
    column!(
        quality_start,
        "`u32` — qualities start, relative to the record."
    );
    column!(
        aux_start,
        "`u32` — aux block start, relative to the record."
    );
    column!(record_end, "`u32` — record length, i.e. one past its end.");

    /// Copies the fixed-width columns back to the host.
    ///
    /// The escape hatch and the differential-testing hook: this must equal what
    /// [`RecordBatch::decode`] produces for the same input. Expensive by
    /// construction — it is the transfer this type exists to avoid.
    pub fn to_host(&self) -> Result<RecordBatch> {
        Ok(RecordBatch::from_columns(
            read_i32(&self.columns.reference_sequence_id)?,
            read_i32(&self.columns.position)?,
            read_u8(&self.columns.mapping_quality)?,
            read_u16(&self.columns.flags)?,
            read_u32(&self.columns.sequence_len)?,
            read_i32(&self.columns.mate_reference_sequence_id)?,
            read_i32(&self.columns.mate_position)?,
            read_i32(&self.columns.template_length)?,
            read_u64(&self.columns.record_offsets)?
                .into_iter()
                .map(|o| o as usize)
                .collect(),
        ))
    }

    /// Copies the variable-length field boundaries back to the host.
    ///
    /// Separate from [`to_host`](Self::to_host) because these have no host
    /// counterpart: the host path reaches variable fields through
    /// [`Record`](crate::columnar::Record) views rather than materialising boundaries.
    /// Tests use this to check the device arithmetic against those views.
    pub fn field_bounds_to_host(&self) -> Result<FieldBounds> {
        Ok(FieldBounds {
            cigar_start: read_u32(&self.columns.cigar_start)?,
            sequence_start: read_u32(&self.columns.sequence_start)?,
            quality_start: read_u32(&self.columns.quality_start)?,
            aux_start: read_u32(&self.columns.aux_start)?,
            record_end: read_u32(&self.columns.record_end)?,
        })
    }
}

/// Variable-length field boundaries, relative to each record's start.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldBounds {
    /// CIGAR start. The read name occupies `NAME_START..cigar_start`.
    pub cigar_start: Vec<u32>,
    /// Packed sequence start.
    pub sequence_start: Vec<u32>,
    /// Qualities start.
    pub quality_start: Vec<u32>,
    /// Aux tag block start.
    pub aux_start: Vec<u32>,
    /// One past the record's last byte.
    pub record_end: Vec<u32>,
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
            reference_sequence_id: buf(4),
            position: buf(4),
            flags: buf(2),
            mapping_quality: buf(1),
            sequence_len: buf(4),
            mate_reference_sequence_id: buf(4),
            mate_position: buf(4),
            template_length: buf(4),
            record_offsets: buf(8),
            cigar_start: buf(4),
            sequence_start: buf(4),
            quality_start: buf(4),
            aux_start: buf(4),
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
        let batch = DeviceRecordBatch::new(0, 0, columns(0)).unwrap();
        assert!(batch.is_empty());
        assert!(batch.to_host().unwrap().is_empty());
    }

    #[test]
    fn rejects_a_column_of_the_wrong_length() {
        // The failure this prevents is a kernel indexing past a column's
        // allocation, which corrupts silently rather than erroring.
        let mut cols = columns(4);
        cols.flags = HostAlloc::buffer(vec![0u8; 6]); // 3 records, not 4
        let err = DeviceRecordBatch::new(4, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("flags")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_a_column_sized_with_the_wrong_element_width() {
        // `flags` is u16; sizing it as if it were i32 is the exact mistake a
        // backend makes when a column's type changes.
        let mut cols = columns(4);
        cols.flags = HostAlloc::buffer(vec![0u8; 4 * 4]);
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
        cols.flags = HostAlloc::buffer([99u16.to_le_bytes(), 4u16.to_le_bytes()].concat());
        cols.record_offsets =
            HostAlloc::buffer([7u64.to_le_bytes(), 4096u64.to_le_bytes()].concat());

        let batch = DeviceRecordBatch::new(2, 0, cols).unwrap();
        let host = batch.to_host().unwrap();

        assert_eq!(host.position(), &[100, -1]);
        assert_eq!(host.flags(), &[99, 4]);
        assert_eq!(host.record_offsets(), &[7, 4096]);
    }

    #[test]
    fn field_bounds_come_back_in_order() {
        let mut cols = columns(1);
        cols.cigar_start = HostAlloc::buffer(42u32.to_le_bytes().to_vec());
        cols.sequence_start = HostAlloc::buffer(46u32.to_le_bytes().to_vec());
        cols.quality_start = HostAlloc::buffer(48u32.to_le_bytes().to_vec());
        cols.aux_start = HostAlloc::buffer(52u32.to_le_bytes().to_vec());
        cols.record_end = HostAlloc::buffer(60u32.to_le_bytes().to_vec());

        let bounds = DeviceRecordBatch::new(1, 0, cols)
            .unwrap()
            .field_bounds_to_host()
            .unwrap();

        assert_eq!(bounds.cigar_start, [42]);
        assert_eq!(bounds.sequence_start, [46]);
        assert_eq!(bounds.quality_start, [48]);
        assert_eq!(bounds.aux_start, [52]);
        assert_eq!(bounds.record_end, [60]);
    }
}
