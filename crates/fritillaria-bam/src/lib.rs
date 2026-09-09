//! BAM parsing.
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

pub mod batch;
pub mod header;
pub mod record;
pub mod seq;

pub use batch::RecordBatch;
pub use header::{Header, ReferenceSequence};
pub use record::{RECORD_CORE_SIZE, Record, scan_records};
pub use seq::{CigarOp, cigar_op_kind, cigar_op_len, decode_base};

/// BAM magic: `BAM\1`.
pub const MAGIC: [u8; 4] = *b"BAM\x01";
