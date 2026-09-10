//! A reference genome resident in device memory.
//!
//! The odd one out among this workspace's device types, and deliberately so.
//! [`fritillaria_bam`], `-bcf` and `-fastq` all hand back columns that *point
//! into* the inflated buffer, because their payloads are the bulk of the file
//! and copying them would double VRAM for nothing.
//!
//! FASTA inverts that. Its payload is wrapped across lines, so pointing at it
//! is not enough — a consumer would have to skip a newline every 70 bases on
//! every read. So this type owns a **compacted** copy: the whole reference,
//! newline-free, contiguous, with an offset per contig.
//!
//! The copy costs 98.6% of the source, and that is the point rather than a
//! regret: it buys coalesced reads, and it is the precondition for 2-bit
//! packing, which is what makes a 3.1 Gbp genome sit in 775 MB instead of
//! 3.1 GB. The inflate batch can be dropped afterwards, which is *not* true of
//! the other three formats.
//!
//! # Not done: 2-bit packing
//!
//! Deliberately absent rather than forgotten. `ACGT` fits in two bits, but a
//! real reference is not only `ACGT` — `N` runs mark centromeres, telomeres and
//! assembly gaps, and IUPAC ambiguity codes appear in some assemblies. Packing
//! therefore needs a companion N-mask, and the committed fixture
//! (`testdata/controls.fa`, two phage genomes) is **100% ACGT**, so it cannot
//! test the case that makes packing hard. Doing it against a fixture that
//! cannot fail it would be worse than not doing it.

use fritillaria_core::{DeviceBuffer, Error, Result};

use crate::columnar::batch::RecordBatch;
use crate::columnar::record::RecordBounds;

/// The columns of a [`DeviceReference`], as a backend hands them over.
#[derive(Debug)]
pub struct DeviceColumns {
    /// `u8` — every contig's bases, concatenated, no newlines.
    pub sequence: DeviceBuffer,
    /// `u64` — where each contig begins in `sequence`.
    pub sequence_offsets: DeviceBuffer,
    /// `u64` — bases per contig.
    pub sequence_lengths: DeviceBuffer,
    /// `u64` — where each contig's record begins in the source buffer.
    pub record_offsets: DeviceBuffer,
}

/// A reference genome, compacted and resident on the device.
#[derive(Debug)]
pub struct DeviceReference {
    contigs: usize,
    total_bases: u64,
    columns: DeviceColumns,
}

impl DeviceReference {
    /// Takes ownership of a backend's columns.
    ///
    /// Checks the three per-contig columns against `contigs` and the sequence
    /// buffer against `total_bases`. A backend that miscounts would otherwise
    /// hand a consumer offsets that index past the reference, which corrupts
    /// silently rather than erroring.
    pub fn new(contigs: usize, total_bases: u64, columns: DeviceColumns) -> Result<Self> {
        let ordinal = columns.sequence.device_ordinal();

        for (buffer, name) in [
            (&columns.sequence_offsets, "sequence_offsets"),
            (&columns.sequence_lengths, "sequence_lengths"),
            (&columns.record_offsets, "record_offsets"),
        ] {
            let expected = contigs * 8;
            if buffer.byte_len() != expected {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is {} bytes, expected {expected} \
                         ({contigs} contigs x 8)",
                        buffer.byte_len()
                    ),
                });
            }
            if buffer.device_ordinal() != ordinal {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "column `{name}` is on device {} but the reference is on {ordinal}",
                        buffer.device_ordinal()
                    ),
                });
            }
        }

        if columns.sequence.byte_len() as u64 != total_bases {
            return Err(Error::InvalidDeviceBatch {
                reason: format!(
                    "sequence buffer is {} bytes but the contigs total {total_bases} bases",
                    columns.sequence.byte_len()
                ),
            });
        }

        Ok(Self {
            contigs,
            total_bases,
            columns,
        })
    }

    /// Number of contigs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.contigs
    }

    /// Whether the reference holds no contigs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contigs == 0
    }

    /// Total bases across every contig.
    #[must_use]
    pub fn total_bases(&self) -> u64 {
        self.total_bases
    }

    /// Which device the reference lives on.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.columns.sequence.device_ordinal()
    }

    /// The compacted bases: `u8`, contiguous, newline-free.
    #[must_use]
    pub fn sequence(&self) -> &DeviceBuffer {
        &self.columns.sequence
    }

    /// `u64` — where each contig begins in [`sequence`](Self::sequence).
    #[must_use]
    pub fn sequence_offsets(&self) -> &DeviceBuffer {
        &self.columns.sequence_offsets
    }

    /// `u64` — bases per contig.
    #[must_use]
    pub fn sequence_lengths(&self) -> &DeviceBuffer {
        &self.columns.sequence_lengths
    }

    /// `u64` — where each contig's record begins in the source buffer.
    #[must_use]
    pub fn record_offsets(&self) -> &DeviceBuffer {
        &self.columns.record_offsets
    }

    /// Copies the compacted reference back to the host.
    ///
    /// The differential-testing hook: this must equal what
    /// [`RecordBatch::compact`] produces for the same input. Expensive by
    /// construction — it is the transfer this type exists to avoid.
    pub fn sequence_to_host(&self) -> Result<Vec<u8>> {
        self.columns.sequence.to_vec()
    }

    /// Copies the contig index back to the host, without the sequence.
    ///
    /// Cheap: a few numbers per contig rather than the genome. `bounds` cannot
    /// be reconstructed from the device columns — the line widths are a
    /// property of the *source* layout, which compaction discards — so this
    /// returns the three columns rather than a [`RecordBatch`].
    pub fn index_to_host(&self) -> Result<ContigIndex> {
        Ok(ContigIndex {
            record_offsets: read_u64(&self.columns.record_offsets)?,
            sequence_offsets: read_u64(&self.columns.sequence_offsets)?,
            sequence_lengths: read_u64(&self.columns.sequence_lengths)?,
        })
    }
}

/// Where each contig sits, in the source and in the compacted reference.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContigIndex {
    /// Where each contig's record begins in the source buffer.
    pub record_offsets: Vec<u64>,
    /// Where each contig's bases begin in the compacted reference.
    pub sequence_offsets: Vec<u64>,
    /// Bases per contig.
    pub sequence_lengths: Vec<u64>,
}

impl ContigIndex {
    /// Builds the same index from a host batch, for comparison.
    #[must_use]
    pub fn from_batch(batch: &RecordBatch) -> Self {
        Self {
            record_offsets: batch.record_offsets().iter().map(|&o| o as u64).collect(),
            sequence_offsets: batch.sequence_offsets().to_vec(),
            sequence_lengths: batch.sequence_lengths(),
        }
    }

    /// Number of contigs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sequence_lengths.len()
    }

    /// Whether the index holds no contigs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequence_lengths.is_empty()
    }
}

/// The bounds a host batch holds that the device does not.
///
/// Exists so a caller can tell what compaction threw away: `line_bases` and
/// `line_width` describe the *source* wrapping, which no longer exists once the
/// newlines are gone.
#[must_use]
pub fn source_layout(batch: &RecordBatch) -> Vec<RecordBounds> {
    batch.bounds().to_vec()
}

fn read_u64(buffer: &DeviceBuffer) -> Result<Vec<u64>> {
    let bytes = buffer.to_vec()?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

#[cfg(test)]
mod tests {
    use fritillaria_core::device::testing::HostAlloc;

    use super::*;

    fn columns(contigs: usize, bases: u64) -> DeviceColumns {
        DeviceColumns {
            sequence: HostAlloc::buffer(vec![b'A'; bases as usize]),
            sequence_offsets: HostAlloc::buffer(vec![0u8; contigs * 8]),
            sequence_lengths: HostAlloc::buffer(vec![0u8; contigs * 8]),
            record_offsets: HostAlloc::buffer(vec![0u8; contigs * 8]),
        }
    }

    #[test]
    fn accepts_columns_that_match_the_contig_count_and_total() {
        let reference = DeviceReference::new(3, 30, columns(3, 30)).unwrap();
        assert_eq!(reference.len(), 3);
        assert_eq!(reference.total_bases(), 30);
        assert_eq!(reference.sequence().byte_len(), 30);
    }

    #[test]
    fn an_empty_reference_is_legal() {
        let reference = DeviceReference::new(0, 0, columns(0, 0)).unwrap();
        assert!(reference.is_empty());
        assert!(reference.index_to_host().unwrap().is_empty());
        assert!(reference.sequence_to_host().unwrap().is_empty());
    }

    #[test]
    fn rejects_a_sequence_buffer_that_disagrees_with_the_totals() {
        // The failure this prevents is the worst one available here: offsets
        // that index past the end of the reference, read by someone else's
        // kernel.
        let cols = columns(2, 20);
        let err = DeviceReference::new(2, 25, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("total")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_an_index_column_of_the_wrong_length() {
        let mut cols = columns(4, 40);
        cols.sequence_lengths = HostAlloc::buffer(vec![0u8; 3 * 8]);
        let err = DeviceReference::new(4, 40, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason }
                if reason.contains("sequence_lengths")),
            "got {err}"
        );
    }

    #[test]
    fn rejects_columns_split_across_devices() {
        let mut cols = columns(2, 20);
        cols.sequence_offsets = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![0u8; 16], 1)));
        let err = DeviceReference::new(2, 20, cols).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidDeviceBatch { reason } if reason.contains("device")),
            "got {err}"
        );
    }

    #[test]
    fn index_to_host_reads_columns_back_in_little_endian() {
        let mut cols = columns(2, 14);
        cols.sequence_offsets =
            HostAlloc::buffer([0u64.to_le_bytes(), 8u64.to_le_bytes()].concat());
        cols.sequence_lengths =
            HostAlloc::buffer([8u64.to_le_bytes(), 6u64.to_le_bytes()].concat());
        cols.record_offsets = HostAlloc::buffer([0u64.to_le_bytes(), 20u64.to_le_bytes()].concat());

        let index = DeviceReference::new(2, 14, cols)
            .unwrap()
            .index_to_host()
            .unwrap();

        assert_eq!(index.sequence_offsets, vec![0, 8]);
        assert_eq!(index.sequence_lengths, vec![8, 6]);
        assert_eq!(index.record_offsets, vec![0, 20]);
        assert_eq!(index.len(), 2);
    }
}
