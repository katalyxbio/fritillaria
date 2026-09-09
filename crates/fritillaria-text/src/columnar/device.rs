//! Tab-delimited records in device memory.
//!
//! Offsets and nothing else. Unlike FASTA, which owns a compacted copy because
//! its payload is wrapped, text records are already contiguous — a record *is*
//! a line — so the columns point into the buffer and copy nothing.
//!
//! The consequence, which callers must know: **the buffer these offsets index
//! has to outlive this batch.**
//!
//! # Why the field table is flat
//!
//! Records have different field counts — 12 for this SAM, 10 for this VCF, and
//! genuinely ragged for BED. A `Vec<Vec<_>>` cannot cross to a device, and a
//! fixed stride would size every record like the widest. So the tabs live in
//! one flat array with a per-record boundary table: `starts[i]..starts[i + 1]`
//! is record `i`'s span, and `starts` has one more entry than there are
//! records.

use fritillaria_core::{DeviceBuffer, Error, Result};

use crate::columnar::batch::RecordBatch;

/// The columns of a [`DeviceRecordBatch`], as a backend hands them over.
#[derive(Debug)]
pub struct DeviceColumns {
    /// `u64` — where each record starts in the source buffer.
    pub record_offsets: DeviceBuffer,
    /// `u64` — one past each record's last byte, its newline excluded.
    pub record_ends: DeviceBuffer,
    /// `u64` — where each header line starts.
    pub header_offsets: DeviceBuffer,
    /// `u64` — every tab offset, grouped by record.
    pub tabs: DeviceBuffer,
    /// `u64` — per-record boundaries into `tabs`; one more entry than records.
    pub field_starts: DeviceBuffer,
}

/// Tab-delimited records scanned into columns that never left the device.
#[derive(Debug)]
pub struct DeviceRecordBatch {
    records: usize,
    headers: usize,
    tail: usize,
    columns: DeviceColumns,
}

impl DeviceRecordBatch {
    /// Takes ownership of a backend's columns.
    ///
    /// Every column is checked against the counts it should match. The field
    /// table gets the strictest check: `field_starts` must have exactly
    /// `records + 1` entries, because that invariant is what makes
    /// `starts[i + 1]` safe to read for the last record.
    pub fn new(
        records: usize,
        headers: usize,
        tail: usize,
        columns: DeviceColumns,
    ) -> Result<Self> {
        let ordinal = columns.record_offsets.device_ordinal();

        let expected: [(&str, &DeviceBuffer, usize); 4] = [
            ("record_offsets", &columns.record_offsets, records * 8),
            ("record_ends", &columns.record_ends, records * 8),
            ("header_offsets", &columns.header_offsets, headers * 8),
            (
                "field_starts",
                &columns.field_starts,
                if records == 0 { 0 } else { (records + 1) * 8 },
            ),
        ];

        for (name, buffer, want) in expected {
            if buffer.byte_len() != want {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is {} bytes, expected {want}",
                        buffer.byte_len()
                    ),
                });
            }
        }

        for (name, buffer) in [
            ("record_offsets", &columns.record_offsets),
            ("record_ends", &columns.record_ends),
            ("header_offsets", &columns.header_offsets),
            ("tabs", &columns.tabs),
            ("field_starts", &columns.field_starts),
        ] {
            if buffer.device_ordinal() != ordinal {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is on device {} but the batch is on {ordinal}",
                        buffer.device_ordinal()
                    ),
                });
            }
        }

        Ok(Self {
            records,
            headers,
            tail,
            columns,
        })
    }

    /// Number of records, header lines excluded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records
    }

    /// Whether the batch holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records == 0
    }

    /// Number of header lines.
    #[must_use]
    pub fn header_count(&self) -> usize {
        self.headers
    }

    /// Offset of the first line without a terminator.
    #[must_use]
    pub fn tail(&self) -> usize {
        self.tail
    }

    /// Which device the columns live on.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.columns.record_offsets.device_ordinal()
    }

    /// `u64` — where each record starts.
    #[must_use]
    pub fn record_offsets(&self) -> &DeviceBuffer {
        &self.columns.record_offsets
    }

    /// `u64` — one past each record's last byte.
    #[must_use]
    pub fn record_ends(&self) -> &DeviceBuffer {
        &self.columns.record_ends
    }

    /// `u64` — where each header line starts.
    #[must_use]
    pub fn header_offsets(&self) -> &DeviceBuffer {
        &self.columns.header_offsets
    }

    /// `u64` — every tab offset, grouped by record.
    #[must_use]
    pub fn tabs(&self) -> &DeviceBuffer {
        &self.columns.tabs
    }

    /// `u64` — per-record boundaries into [`tabs`](Self::tabs).
    #[must_use]
    pub fn field_starts(&self) -> &DeviceBuffer {
        &self.columns.field_starts
    }

    /// Copies every column back to the host.
    ///
    /// The differential-testing hook: this must equal what
    /// [`RecordBatch::decode`] produces for the same input.
    pub fn to_host(&self) -> Result<RecordBatch> {
        let starts = if self.records == 0 {
            vec![0]
        } else {
            read_u64(&self.columns.field_starts)?
        };
        Ok(RecordBatch::from_columns(
            read_usize(&self.columns.record_offsets)?,
            read_usize(&self.columns.record_ends)?,
            read_usize(&self.columns.header_offsets)?,
            read_u64(&self.columns.tabs)?
                .into_iter()
                .map(|v| v as usize)
                .collect(),
            starts.into_iter().map(|v| v as usize).collect(),
        ))
    }
}

fn read_u64(buffer: &DeviceBuffer) -> Result<Vec<u64>> {
    let bytes = buffer.to_vec()?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

fn read_usize(buffer: &DeviceBuffer) -> Result<Vec<usize>> {
    Ok(read_u64(buffer)?.into_iter().map(|v| v as usize).collect())
}

#[cfg(test)]
mod tests {
    use fritillaria_core::device::testing::HostAlloc;

    use super::*;

    fn columns(records: usize, headers: usize, tabs: usize) -> DeviceColumns {
        DeviceColumns {
            record_offsets: HostAlloc::buffer(vec![0u8; records * 8]),
            record_ends: HostAlloc::buffer(vec![0u8; records * 8]),
            header_offsets: HostAlloc::buffer(vec![0u8; headers * 8]),
            tabs: HostAlloc::buffer(vec![0u8; tabs * 8]),
            field_starts: HostAlloc::buffer(vec![
                0u8;
                if records == 0 { 0 } else { (records + 1) * 8 }
            ]),
        }
    }

    #[test]
    fn accepts_columns_that_match_the_counts() {
        let batch = DeviceRecordBatch::new(3, 2, 99, columns(3, 2, 7)).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.header_count(), 2);
        assert_eq!(batch.tail(), 99);
    }

    #[test]
    fn an_empty_batch_is_legal() {
        let batch = DeviceRecordBatch::new(0, 0, 0, columns(0, 0, 0)).unwrap();
        assert!(batch.is_empty());
        assert!(batch.to_host().unwrap().is_empty());
    }

    #[test]
    fn field_starts_must_have_one_more_entry_than_records() {
        // The invariant that makes `starts[i + 1]` safe for the last record. A
        // backend that sizes it like the other columns would walk off the end
        // of the table on every batch's final record.
        let mut cols = columns(4, 0, 9);
        cols.field_starts = HostAlloc::buffer(vec![0u8; 4 * 8]);
        let err = DeviceRecordBatch::new(4, 0, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("field_starts")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_a_column_of_the_wrong_length() {
        let mut cols = columns(4, 1, 9);
        cols.record_ends = HostAlloc::buffer(vec![0u8; 3 * 8]);
        assert!(DeviceRecordBatch::new(4, 1, 0, cols).is_err());
    }

    #[test]
    fn rejects_columns_split_across_devices() {
        let mut cols = columns(2, 0, 3);
        cols.tabs = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![0u8; 24], 1)));
        let err = DeviceRecordBatch::new(2, 0, 0, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("device")),
            "got {err}"
        );
    }

    #[test]
    fn to_host_rebuilds_the_ragged_field_table() {
        // Two records with different field counts, which is the case a flat
        // table plus boundaries exists for.
        let mut cols = columns(2, 0, 3);
        cols.record_offsets = HostAlloc::buffer([0u64.to_le_bytes(), 10u64.to_le_bytes()].concat());
        cols.record_ends = HostAlloc::buffer([9u64.to_le_bytes(), 20u64.to_le_bytes()].concat());
        cols.tabs = HostAlloc::buffer(
            [2u64.to_le_bytes(), 5u64.to_le_bytes(), 14u64.to_le_bytes()].concat(),
        );
        cols.field_starts = HostAlloc::buffer(
            [0u64.to_le_bytes(), 2u64.to_le_bytes(), 3u64.to_le_bytes()].concat(),
        );

        let host = DeviceRecordBatch::new(2, 0, 20, cols)
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(host.len(), 2);
        assert_eq!(host.fields().tabs(0), &[2, 5]);
        assert_eq!(host.fields().tabs(1), &[14]);
        assert_eq!(host.field_counts(), vec![3, 2]);
    }
}
