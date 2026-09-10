//! The columnar path: boundary discovery and device-resident columns.
//!
//! This is the half of the crate that is not vendored. [`io::Reader`](crate::io::Reader)
//! hands back one owning [`Record`](crate::Record) at a time, which is the right
//! shape for a CPU consumer and the wrong one for a GPU kernel; this module
//! produces offsets and lengths instead, leaving the bases where they already
//! are.
//!
//! # Why FASTQ is the easy one, and the hard one
//!
//! **Easy:** records are delimited, not length-prefixed, so there is no serial
//! chain to break. BAM and BCF each needed a trick to find record *n+1* without
//! first reading record *n*; FASTQ needs none, because newlines are findable in
//! parallel.
//!
//! **Hard:** `@` does not mark a record. It is Phred+33 Q31, an ordinary quality
//! score, so the marker is ambiguous in a way no binary format's is. The
//! resolution is a validator whose real work is done by the `+` on line 3 — see
//! [`record::looks_like_a_record`] and `docs/fastq-boundaries.md`.
//!
//! # Naming
//!
//! [`Record`] here is a zero-copy view over the inflated buffer, and is a
//! different type from [`crate::Record`], the vendored owning record. Both are
//! deliberate: the vendored one is the drop-in API, this one is what the
//! columnar path is built on.

pub mod batch;
pub mod device;
pub mod record;
pub mod speculative;

pub use batch::RecordBatch;
pub use device::{DeviceColumns, DeviceRecordBatch};
pub use record::{NAME_START, Record, RecordBounds, looks_like_a_record, scan_records};
pub use speculative::{Proof, SpeculativeScan, scan_records_speculative};
