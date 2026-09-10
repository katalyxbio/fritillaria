//! FASTQ records in device memory: the columnar batch a GPU consumer receives.
//!
//! The narrowest of the three column sets in this workspace, and deliberately
//! so — a FASTQ record has no fixed-width fields to speak of. What it has is a
//! length and three spans, and that is what a GPU aligner reading raw reads
//! wants: lengths to size its work, offsets to find the bases.
//!
//! # Layout
//!
//! ```text
//! record_offsets[i]                      where record i starts
//!   + NAME_START (1)                     definition line, '@' excluded
//!   + sequence_start[i]                  the bases, one byte each
//!   + plus_start[i]                      the '+' separator line
//!   + quality_start[i] .. record_end[i]  Phred+33 scores, one byte per base
//! ```
//!
//! `sequence_len[i]` is the length of **both** the sequence and the quality
//! span — the validator requires them to agree, which is half of what makes a
//! record identifiable at all.
//!
//! # Why nothing is copied
//!
//! Sequence and quality *are* the file. Unlike BAM there is no 4-bit packing to
//! undo and no reason to rewrite them; unlike BCF there is no ragged structure
//! to flatten. Copying them into columns would double VRAM to produce bytes
//! identical to the ones already resident.
//!
//! The consequence, which callers must know: **the
//! [`DeviceInflateBatch`](fritillaria_core::DeviceInflateBatch) these offsets
//! index has to outlive this batch.**

use fritillaria_core::{DeviceBuffer, Error, Result};

use crate::columnar::batch::RecordBatch;
use crate::columnar::record::RecordBounds;

/// The columns of a [`DeviceRecordBatch`], as a backend hands them over.
///
/// A struct rather than five positional arguments: `sequence_start`,
/// `plus_start` and `quality_start` are all `u32` columns of the same length,
/// and transposing two would produce a batch that validates and is wrong.
#[derive(Debug)]
pub struct DeviceColumns {
    /// `u64` — where each record starts in the inflated buffer.
    pub record_offsets: DeviceBuffer,
    /// `u32` — sequence line start, relative to the record.
    pub sequence_start: DeviceBuffer,
    /// `u32` — `+` separator line start, relative to the record.
    pub plus_start: DeviceBuffer,
    /// `u32` — quality line start, relative to the record.
    pub quality_start: DeviceBuffer,
    /// `u32` — bases in the read; also the quality span's length.
    pub sequence_len: DeviceBuffer,
    /// `u32` — one past the record's last byte, relative to its start.
    pub record_end: DeviceBuffer,
}

/// Element width of each column, in the order [`DeviceColumns`] declares them.
const WIDTHS: [(&str, usize); 6] = [
    ("record_offsets", 8),
    ("sequence_start", 4),
    ("plus_start", 4),
    ("quality_start", 4),
    ("sequence_len", 4),
    ("record_end", 4),
];

impl DeviceColumns {
    fn buffers(&self) -> [&DeviceBuffer; 6] {
        [
            &self.record_offsets,
            &self.sequence_start,
            &self.plus_start,
            &self.quality_start,
            &self.sequence_len,
            &self.record_end,
        ]
    }
}

/// FASTQ records decoded into columns that never left the device.
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
    /// `tail` is the offset of the first incompletely-buffered record, which the
    /// caller carries into the next batch. For long reads that is routine: one
    /// ONT record can exceed several BGZF blocks.
    ///
    /// Every column is checked against `n` and its element width. A backend that
    /// miscounts would otherwise hand a consumer a column its kernel reads past
    /// the end of, which is silent corruption rather than an error.
    pub fn new(n: usize, tail: usize, columns: DeviceColumns) -> Result<Self> {
        let ordinal = columns.record_offsets.device_ordinal();

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
        self.columns.record_offsets.device_ordinal()
    }

    column!(
        record_offsets,
        "`u64` — where each record starts in the inflated buffer."
    );
    column!(
        sequence_start,
        "`u32` — sequence line start, relative to the record."
    );
    column!(
        plus_start,
        "`u32` — `+` line start, relative to the record."
    );
    column!(
        quality_start,
        "`u32` — quality line start, relative to the record."
    );
    column!(sequence_len, "`u32` — bases in the read.");
    column!(record_end, "`u32` — record length, i.e. one past its end.");

    /// Copies the columns back to the host.
    ///
    /// The escape hatch and the differential-testing hook: this must equal what
    /// [`RecordBatch::decode`] produces for the same input. Expensive by
    /// construction — it is the transfer this type exists to avoid.
    pub fn to_host(&self) -> Result<RecordBatch> {
        let sequence_start = read_u32(&self.columns.sequence_start)?;
        let plus_start = read_u32(&self.columns.plus_start)?;
        let quality_start = read_u32(&self.columns.quality_start)?;
        let sequence_len = read_u32(&self.columns.sequence_len)?;
        let record_end = read_u32(&self.columns.record_end)?;

        let bounds = (0..self.n)
            .map(|i| RecordBounds {
                sequence_start: sequence_start[i],
                plus_start: plus_start[i],
                quality_start: quality_start[i],
                sequence_len: sequence_len[i],
                record_end: record_end[i],
            })
            .collect();

        Ok(RecordBatch::from_columns(
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

reader!(read_u32, u32);
reader!(read_u64, u64);

#[cfg(test)]
mod tests {
    use fritillaria_core::device::testing::HostAlloc;

    use super::*;

    fn columns(n: usize) -> DeviceColumns {
        let buf = |width: usize| HostAlloc::buffer(vec![0u8; n * width]);
        DeviceColumns {
            record_offsets: buf(8),
            sequence_start: buf(4),
            plus_start: buf(4),
            quality_start: buf(4),
            sequence_len: buf(4),
            record_end: buf(4),
        }
    }

    #[test]
    fn accepts_columns_that_match_the_record_count() {
        let batch = DeviceRecordBatch::new(5, 99, columns(5)).unwrap();
        assert_eq!(batch.len(), 5);
        assert_eq!(batch.tail(), 99);
        assert_eq!(batch.sequence_len().byte_len(), 20);
    }

    #[test]
    fn an_empty_batch_is_legal() {
        // A batch holding only a partial record is routine for long reads.
        let batch = DeviceRecordBatch::new(0, 0, columns(0)).unwrap();
        assert!(batch.is_empty());
        assert!(batch.to_host().unwrap().is_empty());
    }

    #[test]
    fn rejects_a_column_of_the_wrong_length() {
        let mut cols = columns(4);
        cols.sequence_len = HostAlloc::buffer(vec![0u8; 12]); // 3 records, not 4
        let err = DeviceRecordBatch::new(4, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("sequence_len")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_offsets_sized_as_if_they_were_u32() {
        // record_offsets is the only 8-byte column, so sizing it like the
        // others is the natural mistake.
        let mut cols = columns(4);
        cols.record_offsets = HostAlloc::buffer(vec![0u8; 4 * 4]);
        assert!(DeviceRecordBatch::new(4, 0, cols).is_err());
    }

    #[test]
    fn rejects_columns_split_across_devices() {
        let mut cols = columns(2);
        cols.sequence_start = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![0u8; 8], 1)));
        let err = DeviceRecordBatch::new(2, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("device")),
            "got {err}"
        );
    }

    #[test]
    fn to_host_reassembles_the_bounds_per_record() {
        let mut cols = columns(2);
        cols.record_offsets = HostAlloc::buffer([0u64.to_le_bytes(), 40u64.to_le_bytes()].concat());
        cols.sequence_start = HostAlloc::buffer([5u32.to_le_bytes(), 7u32.to_le_bytes()].concat());
        cols.plus_start = HostAlloc::buffer([14u32.to_le_bytes(), 20u32.to_le_bytes()].concat());
        cols.quality_start = HostAlloc::buffer([16u32.to_le_bytes(), 22u32.to_le_bytes()].concat());
        cols.sequence_len = HostAlloc::buffer([8u32.to_le_bytes(), 12u32.to_le_bytes()].concat());
        cols.record_end = HostAlloc::buffer([25u32.to_le_bytes(), 35u32.to_le_bytes()].concat());

        let host = DeviceRecordBatch::new(2, 0, cols)
            .unwrap()
            .to_host()
            .unwrap();

        assert_eq!(host.record_offsets(), &[0, 40]);
        assert_eq!(host.sequence_lengths(), vec![8, 12]);
        assert_eq!(host.bounds()[0].sequence_start, 5);
        assert_eq!(host.bounds()[0].record_end, 25);
        assert_eq!(host.bounds()[1].quality_start, 22);
        assert_eq!(host.bounds()[1].plus_start, 20);
    }
}
