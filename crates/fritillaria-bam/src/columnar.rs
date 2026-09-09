//! The columnar path: boundary discovery and device-resident columns.
//!
//! This is the half of the crate that is not vendored, and the reason the
//! workspace exists. [`io::Reader`](crate::io::Reader) hands back one
//! [`Record`](crate::Record) at a time, which is the right shape for a CPU
//! consumer and the wrong one for a GPU kernel; this module produces columns
//! instead, and [`DeviceRecordBatch`] keeps them in device memory.
//!
//! # Why this is split into two stages
//!
//! BAM records are variable-length and length-prefixed, so finding record *n+1*
//! requires having read record *n*. That makes boundary discovery inherently
//! sequential ([`record::scan_records`]) — but it is cheap, because it touches
//! only the 4-byte `block_size` prefix of each record and skips the body.
//!
//! Once boundaries are known, records become embarrassingly parallel: that is
//! [`batch`], which decodes fixed-width fields into columns.
//!
//! Records span BGZF block boundaries, so both stages run over the
//! *concatenated* decompressed buffer, never per block.
//!
//! # Naming
//!
//! [`Record`] here is a zero-copy view over the inflated buffer, and is a
//! different type from [`crate::Record`], the vendored owning record. Both are
//! deliberate: the vendored one is the drop-in API, this one is what the
//! columnar decode is built on.

pub mod aux;
pub mod batch;
pub mod blocked;
pub mod device;
pub mod header;
pub mod record;
pub mod seq;

pub use aux::{Array, Fields, Tag, Value, Values};
pub use batch::RecordBatch;
pub use blocked::{BlockedScan, Segment, scan_records_blocked};
pub use device::{DeviceColumns, DeviceRecordBatch, FieldBounds};
pub use header::{Header, ReferenceSequence};
pub use record::{RECORD_CORE_SIZE, Record, scan_records};
pub use seq::{CigarOp, cigar_op_kind, cigar_op_len, decode_base};

/// BAM magic: `BAM\1`.
pub const MAGIC: [u8; 4] = *b"BAM\x01";
