//! Error types.

use std::io;

/// Convenience alias used across the workspace.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors produced anywhere in the pipeline.
///
/// Corruption variants carry enough context to identify *which* block failed;
/// a bare "invalid data" in a 500 GiB BAM is not actionable.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("i/o error")]
    Io(#[from] io::Error),

    /// The gzip/BGZF framing around a block is not valid.
    #[error("invalid BGZF block at offset {offset}: {reason}")]
    InvalidBlock { offset: u64, reason: &'static str },

    /// Framing parsed, but the payload did not survive verification.
    ///
    /// This is the variant that must never be silently swallowed: it means the
    /// bytes decompressed to something other than what was written.
    #[error(
        "checksum mismatch in block at offset {offset}: expected crc32 {expected:#010x}, got {actual:#010x}"
    )]
    ChecksumMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },

    #[error(
        "size mismatch in block at offset {offset}: ISIZE says {expected} bytes, inflated {actual}"
    )]
    SizeMismatch {
        offset: u64,
        expected: u32,
        actual: usize,
    },

    /// Decompression itself failed.
    #[error("inflate failed for block at offset {offset}: {reason}")]
    Inflate { offset: u64, reason: String },

    /// Malformed record or header content inside an otherwise valid block.
    #[error("malformed {format} at position {position}: {reason}")]
    Malformed {
        format: &'static str,
        position: u64,
        reason: String,
    },

    /// A device-resident batch violated its layout contract.
    ///
    /// Always a bug in a backend rather than bad input: it means the offsets a
    /// codec produced do not describe the buffer it produced. Caught eagerly
    /// because the alternative is a kernel reading past the end of a column.
    #[error("invalid device batch: {reason}")]
    InvalidDeviceBatch { reason: String },

    /// A GPU backend was requested but is unavailable or failed.
    #[error("cuda backend unavailable: {0}")]
    CudaUnavailable(String),

    #[error("cuda error: {0}")]
    Cuda(String),
}
