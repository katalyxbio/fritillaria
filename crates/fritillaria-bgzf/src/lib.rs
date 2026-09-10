//! BGZF: the blocked-gzip container under BAM, BCF, and `bgzip`ped text.
//!
//! A BGZF file is a sequence of ordinary gzip members, each carrying an extra
//! `BC` subfield giving its total size and each holding at most 64 KiB of
//! payload. Because every member is self-contained, block *i* can be inflated
//! without reference to block *i-1* — which is the entire reason this library
//! can use a GPU.
//!
//! # Two readers, and the difference matters
//!
//! - [`io::Reader`] is the **CPU** reader, vendored from `noodles-bgzf`. It
//!   implements [`std::io::Read`], so every format crate in this workspace is
//!   generic over it. Use it when you want bytes in host memory.
//! - [`BgzfReader`] and [`DeviceBgzfReader`] are the **codec-driven** readers.
//!   They batch blocks through a [`BlockCodec`](fritillaria_core::BlockCodec),
//!   which is where a GPU enters the picture, and `DeviceBgzfReader` keeps its
//!   output in device memory rather than copying it back.
//!
//! `BgzfReader` also implements [`io::Read`] and [`io::BufRead`], so it drops
//! straight into any reader written against the vendored CPU path — that is the
//! one-line migration this workspace exists to offer.
//!
//! # Provenance
//!
//! [`io`], [`gzi`], [`r#async`] and [`virtual_position`] are vendored from
//! `noodles-bgzf` (MIT, © 2018 Michael Macias). [`block`], [`discover`],
//! [`cpu`], [`read`], [`device_read`], [`device_write`] and [`write`] are
//! ours. See `VENDORED.md`.

// --- ours -------------------------------------------------------------------

pub mod block;
pub mod cpu;
pub mod device_read;
pub mod device_write;
pub mod discover;
#[cfg(any(test, feature = "testing"))]
pub mod host_device;
pub mod read;
pub mod write;

pub use block::{BlockHeader, EOF_BLOCK, is_eof_block};
pub use cpu::{CpuCodec, CpuCompressor};
pub use device_read::{DeviceBatch, DeviceBgzfReader};
pub use device_write::{DEFAULT_CHUNKS_PER_BATCH, DeviceBgzfWriter};
pub use discover::{BlockDiscovery, discover_blocks};
#[cfg(any(test, feature = "testing"))]
pub use host_device::{HostDeviceCodec, HostDeviceCompressor};
pub use read::{BgzfReader, DEFAULT_BLOCKS_PER_BATCH};
pub use write::{
    BgzfWriter, DEFAULT_PAYLOAD_SIZE, MAX_DEFLATE_STREAM, STORED_BLOCK_HEADER, frame_block,
    store_block,
};

// --- vendored ---------------------------------------------------------------
//
// Not reformatted to this workspace's lint set: keeping it diffable against
// upstream is worth more than uniform style.

#[cfg(feature = "async")]
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod r#async;

#[allow(clippy::pedantic, unreachable_pub)]
pub(crate) mod deflate;
#[allow(clippy::pedantic, unreachable_pub)]
mod gz;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod gzi;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod io;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod virtual_position;

pub use self::virtual_position::VirtualPosition;

// XLEN (2)
const GZIP_XLEN_SIZE: usize = 2;

// SI1 (1) + SI2 (1) + SLEN (2) + BSIZE (2)
const BGZF_XLEN: usize = 6;

// § 4.1 The BGZF compression format (2021-06-03): "Thus while `ISIZE` is stored as a `uint32_t` as
// per the gzip format, in BGZF it is limited to the range [0, 65536]."
const BGZF_MAX_ISIZE: usize = 1 << 16;

pub(crate) const BGZF_HEADER_SIZE: usize = gz::HEADER_SIZE + GZIP_XLEN_SIZE + BGZF_XLEN;
