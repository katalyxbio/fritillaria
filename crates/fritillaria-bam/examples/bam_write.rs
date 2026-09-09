//! Vendored from noodles; see VENDORED.md.
//!
//! The four merged crates apply this workspace's pedantic lint set, which
//! upstream does not. Relaxed here rather than rewritten, so the file stays
//! diffable against upstream.
#![allow(clippy::pedantic)]

//! Creates a new BAM file.
//!
//! This writes a SAM header, reference sequences, and one unmapped record to stdout.
//!
//! Verify the output by piping to `samtools view --no-PG --with-header`.

use std::io;

use fritillaria_bam as bam;
use fritillaria_sam::{
    self as sam,
    header::record::value::{Map, map::Program},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdout = io::stdout().lock();
    let mut writer = bam::io::Writer::new(stdout);

    let header = sam::Header::builder()
        .set_header(Default::default())
        .add_program("fritillaria-bam", Map::<Program>::default())
        .add_comment("an example BAM written by fritillaria-bam")
        .build();

    writer.write_header(&header)?;

    let record = bam::Record::default();
    writer.write_record(&header, &record)?;

    Ok(())
}
