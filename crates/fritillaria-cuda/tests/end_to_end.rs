//! The whole stack: a real htslib BAM, decompressed on the GPU, parsed on the CPU.
//!
//! This is the thesis of the project reduced to one test. The vendored reader does
//! the record parsing, the GPU does the decompression, and neither was modified
//! to suit the other. If this passes, the same substitution accelerates BCF,
//! `bgzip`ped VCF, and every tabix-indexed format, because they all read through
//! the same BGZF trait.
//!
//! Skips without a device — but `scripts/colab_job.py` fails the run if a
//! device test skips on a GPU VM, so this cannot quietly stop being checked.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bam as bam;
use fritillaria_bgzf::BgzfReader;
use fritillaria_cuda::CudaCodec;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// A GPU codec, or `None` with an explanation.
fn codec() -> Option<CudaCodec> {
    match CudaCodec::new() {
        Ok(codec) => Some(codec),
        Err(err) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

#[test]
fn the_vendored_reader_reads_an_htslib_bam_decompressed_on_the_gpu() {
    let Some(codec) = codec() else { return };

    let file = std::fs::File::open(fixture("htslib.bam")).unwrap();
    let mut reader = bam::io::Reader::from(BgzfReader::with_codec(file, codec));

    let header = reader.read_header().expect("header must parse");
    assert_eq!(header.reference_sequences().len(), 2);

    let records: Vec<_> = reader
        .records()
        .collect::<std::io::Result<Vec<_>>>()
        .expect("every record must decode");

    assert_eq!(records.len(), 8);
    assert_eq!(records[0].alignment_start().unwrap().unwrap().get(), 7);
    assert_eq!(records[0].flags().bits(), 99);
    assert_eq!(records[0].sequence().len(), 17);
    assert!(records[6].flags().is_unmapped());
}

#[test]
fn gpu_and_cpu_agree_on_a_real_multiblock_bam() {
    // The differential test that matters most: identical parsed output from a
    // real file, whichever backend decompressed it.
    let Some(codec) = codec() else { return };

    let gpu_positions: Vec<usize> = {
        let file = std::fs::File::open(fixture("htslib_multiblock.bam")).unwrap();
        let mut reader = bam::io::Reader::from(BgzfReader::with_codec(file, codec));
        reader.read_header().unwrap();
        reader
            .records()
            .map(|r| r.unwrap().alignment_start().unwrap().unwrap().get())
            .collect()
    };

    let cpu_positions: Vec<usize> = {
        let file = std::fs::File::open(fixture("htslib_multiblock.bam")).unwrap();
        let mut reader = bam::io::Reader::from(BgzfReader::new(file));
        reader.read_header().unwrap();
        reader
            .records()
            .map(|r| r.unwrap().alignment_start().unwrap().unwrap().get())
            .collect()
    };

    assert_eq!(gpu_positions.len(), 4000);
    assert_eq!(
        gpu_positions, cpu_positions,
        "GPU and CPU backends disagree on a real htslib file"
    );
}

#[test]
fn gpu_backed_reader_handles_small_batches() {
    // One block per batch forces the reader to stitch batches together while
    // the GPU path is active — block seams and batch seams at once.
    let Some(codec) = codec() else { return };

    let file = std::fs::File::open(fixture("htslib_multiblock.bam")).unwrap();
    let inner = BgzfReader::with_codec(file, codec).with_blocks_per_batch(1);
    let mut reader = bam::io::Reader::from(inner);

    reader.read_header().unwrap();
    assert_eq!(reader.records().count(), 4000);
}
