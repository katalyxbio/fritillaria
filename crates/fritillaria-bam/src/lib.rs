//! **fritillaria-bam** handles the reading and writing of the BAM (Binary Alignment/Map) file
//! format.
//!
//! The BAM format contains the same information as SAM (Sequence Alignment/Map), namely a SAM
//! header and a list of records.
//!
//! # Two APIs
//!
//! [`io`], [`bai`], [`fs`], [`record`] and [`r#async`] are vendored from
//! `noodles-bam` (MIT, © 2018 Michael Macias) and are the drop-in CPU API. Put
//! a [`BgzfReader`](fritillaria_bgzf::BgzfReader) underneath [`io::Reader`] and
//! they decompress on the GPU without any other change.
//!
//! [`columnar`] is ours, and is the part that has no equivalent upstream: it
//! turns an inflated buffer into columns, and can keep them in device memory.
//!
//! See `VENDORED.md` for the provenance of every module.
//!
//! # Examples
//!
//! ## Read all records
//!
//! ```no_run
//! # use std::{fs::File, io};
//! use fritillaria_bam as bam;
//!
//! let mut reader = File::open("sample.bam").map(bam::io::Reader::new)?;
//! let header = reader.read_header()?;
//!
//! for result in reader.records() {
//!     let record = result?;
//!     // ...
//! }
//! # Ok::<_, io::Error>(())
//! ```
//!
//! ## Query records
//!
//! Querying allows filtering records by region. It requires an associated BAM index (BAI).
//!
//! ```no_run
//! # use std::fs::File;
//! use fritillaria_bam as bam;
//!
//! let mut reader = bam::io::indexed_reader::Builder::default().build_from_path("sample.bam")?;
//! let header = reader.read_header()?;
//!
//! let region = "sq0:5-8".parse()?;
//! let query = reader.query(&header, &region)?;
//!
//! for result in query.records() {
//!     let record = result?;
//!     // ...
//! }
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

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
pub mod bai;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod fs;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod io;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
pub mod record;
#[allow(clippy::pedantic, missing_debug_implementations, unreachable_pub)]
mod record_ref;

pub use self::{record::Record, record_ref::RecordRef};
