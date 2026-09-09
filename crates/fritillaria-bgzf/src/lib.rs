//! BGZF: the blocked-gzip container under BAM and `bgzip`ped FASTQ.
//!
//! A BGZF file is a sequence of ordinary gzip members, each carrying an extra
//! `BC` subfield giving its total size and each holding at most 64 KiB of
//! payload. Because every member is self-contained, block *i* can be inflated
//! without reference to block *i-1* — which is the entire reason this library
//! can use a GPU.
//!
//! # Layout
//!
//! - [`block`] — header parsing and the format constants.
//! - [`discover`] — the sequential header walk that produces
//!   [`BlockSpan`](fritillaria_core::BlockSpan)s for a codec to consume.
//! - [`cpu`] — the CPU reference codec, correctness oracle for every backend.
//! - [`write`] — a CPU writer, used to build fixtures and as the compression baseline.

pub mod block;
pub mod cpu;
pub mod discover;
#[cfg(any(test, feature = "testing"))]
pub mod host_device;
pub mod read;
pub mod write;

pub use block::{BlockHeader, EOF_BLOCK, is_eof_block};
pub use cpu::CpuCodec;
pub use discover::{BlockDiscovery, discover_blocks};
#[cfg(any(test, feature = "testing"))]
pub use host_device::HostDeviceCodec;
pub use read::{BgzfReader, DEFAULT_BLOCKS_PER_BATCH};
pub use write::{BgzfWriter, DEFAULT_PAYLOAD_SIZE};
