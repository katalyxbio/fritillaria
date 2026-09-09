//! Shared types and the backend seam.
//!
//! This crate deliberately has no GPU dependency. It defines the vocabulary
//! (blocks, virtual offsets, errors) and the [`BlockCodec`] trait that every
//! backend — CPU reference, CUDA, optionally nvCOMP — implements. Format crates
//! depend on this and on the trait, never on a specific backend.
//!
//! # Two halves
//!
//! [`Position`] and [`Region`] are vendored from `noodles-core` (MIT,
//! © 2018 Michael Macias) and are the vocabulary every format crate here shares.
//! Everything else is the backend seam, which is ours. See `VENDORED.md`.

pub mod codec;
pub mod compress;
pub mod device;
pub mod error;
pub mod virtual_offset;

// Vendored, and not reformatted to this workspace's lint set — keeping it
// diffable against upstream is worth more than uniform style.
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod position;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod region;

pub use self::{position::Position, region::Region};

pub use codec::{BlockCodec, BlockSpan, InflateBatch};
pub use compress::{BlockCompressor, CompressedBatch, MAX_COMPRESSIBLE_PAYLOAD};
pub use device::{DeviceAlloc, DeviceBlockCodec, DeviceBuffer, DeviceInflateBatch};
pub use error::{Error, Result};
pub use virtual_offset::VirtualOffset;

/// Maximum size of a BGZF block, compressed or uncompressed, in bytes.
///
/// The format caps both at 64 KiB. The `BC` subfield stores `size - 1` in a
/// `u16` precisely so that a full 65536-byte block is representable.
pub const MAX_BLOCK_SIZE: usize = 65536;
