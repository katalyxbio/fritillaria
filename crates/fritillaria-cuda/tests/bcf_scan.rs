//! BCF boundary discovery on device, diffed against the CPU reference.
//!
//! The oracle is `fritillaria_bcf::columnar::speculative`, which is itself checked
//! against `bcftools` in `fritillaria-bcf/tests/bcftools.rs`. So a pass here
//! chains back to htslib rather than to our own opinion of the format.
//!
//! Two things are asserted separately, and the second matters as much:
//!
//! 1. **The boundaries are identical** to the serial scan's.
//! 2. **The tiling proof succeeded.** A silent fallback to the single-thread
//!    walk is correct and performs exactly like the serial design this
//!    replaces, so "it worked" is not the same as "it worked the fast way".
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use fritillaria_bcf::columnar::{
    Proof, header::parse_header, scan_records, scan_records_speculative,
};
use fritillaria_bgzf::{CpuCodec, DeviceBgzfReader, discover_blocks};
use fritillaria_core::{BlockCodec, DeviceBlockCodec, InflateBatch};
use fritillaria_cuda::{BcfScanner, CudaCodec};

const FIXTURES: &[&str] = &["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"];

fn parts() -> Option<(CudaCodec, BcfScanner)> {
    match (CudaCodec::new(), BcfScanner::new()) {
        (Ok(codec), Ok(scanner)) => Some((codec, scanner)),
        (Err(err), _) | (_, Err(err)) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(path(name)).expect("fixture missing; see testdata/README.md")
}

/// Inflates a whole file on the host, for the header and the oracle.
fn inflate_host(raw: &[u8]) -> InflateBatch {
    let spans = discover_blocks(raw, 0).expect("bcftools BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(raw, &spans, &mut out)
        .expect("bcftools BGZF must inflate");
    out
}

/// The header's sample and contig counts, which are what the sieve keys on.
fn header_of(raw: &[u8]) -> (usize, u32, u32) {
    let host = inflate_host(raw);
    let header = parse_header(host.data()).expect("bad BCF header");
    (
        header.records_start,
        u32::try_from(header.sample_count()).unwrap(),
        u32::try_from(header.dictionary.contigs.len()).unwrap(),
    )
}

#[test]
fn device_boundaries_match_the_host_reference() {
    let Some((codec, scanner)) = parts() else {
        return;
    };

    for name in FIXTURES {
        let raw = fixture(name);
        let (records_start, samples, contigs) = header_of(&raw);

        // Whole file as one batch, so the answer is directly comparable to the
        // host oracle over the same bytes.
        let spans = discover_blocks(&raw, 0).unwrap();
        let mut batch = fritillaria_core::DeviceInflateBatch::new();
        codec
            .inflate_batch_device(&raw, &spans, &mut batch)
            .expect("device inflate");

        let host = inflate_host(&raw);
        let expected =
            scan_records_speculative(host.data(), records_start, samples, contigs).unwrap();
        assert_eq!(expected.proof, Proof::Tiled, "{name}: oracle sanity");

        let got = scanner
            .scan(&batch, records_start, samples, contigs)
            .expect("device scan");

        assert_eq!(got.offsets, expected.offsets, "{name}: boundaries");
        assert_eq!(got.tail, expected.tail, "{name}: tail");
        assert_eq!(
            got.sieved, expected.sieved,
            "{name}: the kernel must sieve exactly what the reference does"
        );
        assert_eq!(got.validated, expected.validated, "{name}: validated");
        assert_eq!(
            got.proof,
            Proof::Tiled,
            "{name}: the tiling must hold on device, or the kernel bought nothing"
        );
    }
}

#[test]
fn device_boundaries_match_across_batch_seams() {
    // A BCF batch almost always ends mid-record, because bcftools packs BGZF
    // blocks full. Driving one block per batch puts a seam in the hardest place
    // available.
    //
    // Batch-local offsets are not comparable to whole-file ones — the reader
    // hands over a window and re-inflates carried blocks at its front — so the
    // comparison is on the record *count* and on the record *lengths*, which
    // are position-independent and would still catch a boundary landing one
    // byte off.
    let Some((codec, scanner)) = parts() else {
        return;
    };

    let mut seams = 0usize;
    for name in FIXTURES {
        let raw = fixture(name);
        let (records_start, samples, contigs) = header_of(&raw);
        let host = inflate_host(&raw);
        let buf = host.data();
        let (whole, _) = scan_records(buf, records_start).unwrap();

        // Every record's length, in file order: the position-independent
        // fingerprint of a correct scan.
        let expected: Vec<usize> = whole
            .iter()
            .enumerate()
            .map(|(i, &start)| whole.get(i + 1).copied().unwrap_or(buf.len()) - start)
            .collect();

        for blocks_per_batch in [1usize, 2] {
            let mut reader =
                DeviceBgzfReader::new(&raw[..], &codec).with_blocks_per_batch(blocks_per_batch);
            let mut lengths: Vec<usize> = Vec::new();
            let mut fallbacks = 0usize;
            let mut batches = 0usize;
            let mut header_parsed = false;

            while let Some(batch) = reader.next_batch().expect("batch") {
                let begin = if header_parsed {
                    batch.start
                } else {
                    let staged = batch.data.to_host().expect("header download");
                    let Ok(header) = parse_header(staged.data()) else {
                        reader.carry_from(0).expect("carry");
                        continue;
                    };
                    header_parsed = true;
                    header.records_start
                };

                let scan = scanner
                    .scan(&batch.data, begin, samples, contigs)
                    .expect("device scan");
                batches += 1;
                if matches!(scan.proof, Proof::Walked { .. }) {
                    fallbacks += 1;
                }

                for (i, &start) in scan.offsets.iter().enumerate() {
                    let end = scan.offsets.get(i + 1).copied().unwrap_or(scan.tail);
                    lengths.push(end - start);
                }

                let tail = scan.tail;
                drop(batch);
                reader.carry_from(tail).expect("carry");
            }

            let context = format!("{name} at {blocks_per_batch} blocks/batch");
            assert_eq!(
                lengths.len(),
                expected.len(),
                "{context}: record count across seams"
            );
            assert_eq!(lengths, expected, "{context}: record lengths across seams");
            assert_eq!(
                fallbacks, 0,
                "{context}: the tiling must hold at a seam too, or the design \
                 degrades to the serial walk once per batch"
            );

            // Whether a seam was actually crossed is a property of the reader,
            // not of this test, and `blocks_per_batch` budgets *compressed*
            // bytes — so on 20x-compressible genotype data a small number still
            // swallows a lot of file. Record it rather than assume it: the
            // assertion below is what makes "0 fallbacks" mean something.
            seams += batches.saturating_sub(1);
        }
    }

    assert!(
        seams >= 3,
        "no batch seam was crossed in any configuration, so nothing above \
         tested the seam behaviour it claims to ({seams} seams)"
    );
}
