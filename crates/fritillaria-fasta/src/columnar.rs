//! The columnar path: a contig index and newline-free sequence.
//!
//! This is the half of the crate that is not vendored, and its purpose is
//! narrower than the other formats'. A GPU aligner does not want FASTA
//! *records*; it wants the reference **resident and contiguous** in device
//! memory, with an index saying where each contig begins.
//!
//! # Why compaction is the deliverable
//!
//! FASTA wraps its sequence, conventionally at 60–80 columns, so the bases are
//! interrupted by a newline every 70 bytes. That is not a memory problem — the
//! newlines are 1.4% of a real reference — it is an *access* problem:
//!
//! - A 150-base lookup spans two or three newlines, so it cannot be a coalesced
//!   read.
//! - 2-bit packing, which is what makes a 3.1 Gbp genome fit comfortably in
//!   VRAM at 775 MB instead of 3.1 GB, requires contiguous bases first.
//!
//! Random access itself is *not* the motivation: with uniform wrapping the
//! coordinate conversion is arithmetic, which is exactly why `samtools faidx`
//! can index a FASTA with five numbers per contig. See
//! [`record::RecordBounds`], which mirrors those five.
//!
//! # And why this module is short
//!
//! Finding records is trivial. `>` at a line start is unambiguous, because a
//! sequence line never begins with one — so unlike FASTQ there is no decoy to
//! defend against, no speculative scan, and no tiling proof.

pub mod batch;
pub mod device;
pub mod record;

pub use batch::RecordBatch;
pub use device::{DeviceColumns, DeviceReference};
pub use record::{Record, RecordBounds, is_record_start, scan_records};
