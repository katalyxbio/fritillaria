//! The columnar path: boundary discovery over an inflated buffer.
//!
//! BCF is VCF's binary encoding, and it lives in a BGZF container — the same
//! container BAM uses. That is the whole reason this module is small: block
//! discovery, GPU-parallel inflate, CRC verification and virtual offsets are
//! container-level and already built, so BCF inherits them without a line of
//! new device code. The dividing line in this project is the container, not
//! the format.
//!
//! # Where BCF differs from BAM, and it matters more than expected
//!
//! Two differences, and the second one broke an assumption this project had
//! been building on:
//!
//! 1. **Records are interpretable only against the header.** Nearly every
//!    string in a record — the contig, every FILTER, INFO and FORMAT key — is
//!    an integer offset into a dictionary the header text defines. A BAM record
//!    stands alone apart from the reference *name*; a BCF record does not.
//!    See [`Dictionary`].
//!
//! 2. **Records cross BGZF block boundaries as a matter of course.** BAM's
//!    device-side boundary scan leans on htslib starting a new block rather
//!    than splitting an alignment, which makes a block start almost always a
//!    record start. `bcf_write` does no such thing. Measured on the committed
//!    fixtures, **0 of 56** interior block starts in `kg_phase3.bcf` are record
//!    starts, and 0 of 3 in `giab_hg002.bcf`. The BAM trick does not transfer.
//!    [`looks_like_a_record`] is the replacement's foundation; the reasoning is
//!    in `docs/bcf-boundaries.md`.
//!
//! # Layout
//!
//! [`header`] parses the magic, version and VCF text, and builds the two
//! dictionaries out of it. [`record`] walks record boundaries and gives a
//! zero-copy view over one record. [`typed`] decodes the BCF2 typed-value
//! encoding that every variable field uses, and is where the format's sharp
//! edges live.
//!
//! # Naming
//!
//! [`Record`] here is a zero-copy view over the inflated buffer, and is a
//! different type from [`crate::Record`], the vendored owning record.

pub mod batch;
pub mod header;
pub mod record;
pub mod speculative;
pub mod typed;

pub use batch::RecordBatch;
pub use header::{Dictionary, Header, parse_header};
pub use record::{
    Allele, FormatField, FormatFields, InfoFields, MIN_RECORD_SIZE, Record, RecordBounds,
    SITE_CORE_SIZE, decode_genotype, looks_like_a_record, scan_records,
};
pub use speculative::{Proof, SpeculativeScan, scan_records_speculative};
pub use typed::{Float, Floats, Int, Ints, Kind, Typed};

/// BCF magic: the three characters `BCF`, with the version in the next two
/// bytes rather than in the magic itself.
pub const MAGIC: [u8; 3] = *b"BCF";

/// The only major version this crate reads.
pub const MAJOR_VERSION: u8 = 2;
