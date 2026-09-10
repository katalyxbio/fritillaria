//! **fritillaria-bcf** handles the reading and writing of the BCF format.
//!
//! # Two APIs
//!
//! [`io`], [`fs`], [`record`] and [`r#async`] are vendored from `noodles-bcf`
//! (MIT, © 2018 Michael Macias) and are the drop-in CPU API. Put a
//! [`BgzfReader`](fritillaria_bgzf::BgzfReader) underneath [`io::Reader`] and
//! they decompress on the GPU without any other change.
//!
//! [`columnar`] is ours: zero-copy record views, the BCF2 typed-value decoder,
//! and the speculative boundary scan that the CUDA kernel is a translation of.
//!
//! See `VENDORED.md` for the provenance of every module.

// --- ours -------------------------------------------------------------------

pub mod columnar;

// --- vendored ---------------------------------------------------------------
//
// Not reformatted to this workspace's lint set: keeping it diffable against
// upstream is worth more than uniform style.

#[cfg(feature = "async")]
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod r#async;

#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod fs;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod io;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod record;

pub use self::record::Record;
