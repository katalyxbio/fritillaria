//! Shared types and the backend seam.
//!
//! This crate deliberately has no GPU dependency. It defines the vocabulary
//! (blocks, virtual offsets, errors) and the [`BlockCodec`] trait that every
//! backend — CPU reference, CUDA, optionally nvCOMP — implements. Format crates
//! depend on this and on the trait, never on a specific backend.

pub mod codec;
pub mod device;
pub mod error;
pub mod virtual_offset;

pub use codec::{BlockCodec, BlockSpan, InflateBatch};
pub use device::{DeviceAlloc, DeviceBlockCodec, DeviceBuffer, DeviceInflateBatch};
pub use error::{Error, Result};
pub use virtual_offset::VirtualOffset;

/// Maximum size of a BGZF block, compressed or uncompressed, in bytes.
///
/// The format caps both at 64 KiB. The `BC` subfield stores `size - 1` in a
/// `u16` precisely so that a full 65536-byte block is representable.
pub const MAX_BLOCK_SIZE: usize = 65536;
