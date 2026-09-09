//! The two numbers `docs/bcf-boundaries.md` says to measure before writing a
//! device-side BCF boundary scan.
//!
//! The design there is: run [`looks_like_a_record`] at every byte offset of a
//! block in parallel, prune to a handful of survivors, then confirm. Whether
//! that is worth building depends entirely on two figures that were reasoned
//! about rather than measured, and this project has a record of measurements
//! overturning reasoning.
//!
//! 1. **Survivors per block.** If a 64 KiB block yields single digits the
//!    scheme works. If it yields hundreds, the expensive confirmation step
//!    dominates and the parallelism has been spent on validation.
//! 2. **The serial cost it is being compared against.** `fritillaria-bam`
//!    measured its serial reconcile at 222 ns per block and the worry about it
//!    turned out to be unfounded. Records per batch is a much larger number
//!    than blocks per batch, so the same may not hold — but if the serial walk
//!    is fast enough there is nothing here to solve.
//!
//! Host-only and needs no GPU: the validator is the same code the kernel would
//! run, so its selectivity can be characterised locally.
//!
//! ```text
//! cargo run --release -p fritillaria-bcf --example scan_survivors -- <file.bcf>
//! ```

// Rates and per-block averages are printed to a few significant figures.
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use fritillaria_bcf::columnar::{
    header::parse_header, looks_like_a_record, record::Record, scan_records,
};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

/// Inflates a whole BCF and reports where each block's payload begins.
fn inflate(path: &str) -> (InflateBatch, Vec<usize>) {
    let raw = std::fs::read(path).expect("cannot read input");
    let spans = discover_blocks(&raw, 0).expect("not a BGZF file");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("inflate failed");

    let mut starts = Vec::with_capacity(spans.len());
    let mut at = 0usize;
    for span in &spans {
        starts.push(at);
        at += span.isize as usize;
    }
    (out, starts)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "testdata/kg_phase3.bcf".to_string());

    let (batch, block_starts) = inflate(&path);
    let buf = batch.data();
    let header = parse_header(buf).expect("bad BCF header");

    let samples = u32::try_from(header.sample_count()).expect("sample count fits");
    let contigs = u32::try_from(header.dictionary.contigs.len()).expect("contig count fits");

    println!("file:     {path}");
    println!(
        "inflated: {} bytes in {} blocks",
        buf.len(),
        block_starts.len()
    );
    println!("samples:  {samples}    contigs: {contigs}\n");

    // ---- the serial walk, which is the thing to beat ----------------------
    let started = Instant::now();
    let (offsets, tail) = scan_records(buf, header.records_start).expect("scan failed");
    let serial = started.elapsed();
    assert_eq!(tail, buf.len(), "input must end on a record boundary");

    let scanned = buf.len() - header.records_start;
    println!("-- serial scan (scan_records) --");
    println!("  records:      {}", offsets.len());
    println!("  wall:         {serial:?}");
    println!(
        "  per record:   {:.1} ns",
        serial.as_secs_f64() * 1e9 / offsets.len() as f64
    );
    println!(
        "  throughput:   {:.2} GiB/s of inflated bytes\n",
        scanned as f64 / serial.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
    );

    sweep(
        buf,
        &block_starts,
        &header,
        samples,
        contigs,
        &offsets,
        serial,
    );
}

/// Runs the validator at every byte offset of every block and reports what
/// survived. This is the measurement; everything above it is setup.
#[allow(clippy::too_many_arguments)]
fn sweep(
    buf: &[u8],
    block_starts: &[usize],
    header: &fritillaria_bcf::columnar::Header,
    samples: u32,
    contigs: u32,
    offsets: &[usize],
    serial: std::time::Duration,
) {
    // Every byte offset in every block is a candidate, which is exactly what a
    // thread-per-offset kernel would test.
    let real: std::collections::HashSet<usize> = offsets.iter().copied().collect();

    let mut candidates = 0u64;
    let mut survivors = 0u64;
    let mut confirmed = 0u64;
    let mut false_positives = 0u64;
    let mut worst_block = (0usize, 0u64);
    let mut blocks_measured = 0u64;

    let started = Instant::now();
    for (index, &start) in block_starts.iter().enumerate() {
        let end = block_starts.get(index + 1).copied().unwrap_or(buf.len());
        let from = start.max(header.records_start);
        if from >= end {
            continue; // header-only or the trailing empty block
        }
        blocks_measured += 1;

        let mut here = 0u64;
        for pos in from..end {
            candidates += 1;
            if !looks_like_a_record(buf, pos, samples, contigs) {
                continue;
            }
            survivors += 1;
            here += 1;

            // Stage 2: the full typed-value walk. A survivor that also passes
            // this is indistinguishable from a real record start by inspection
            // alone — only the chain can rule it out.
            let l_shared =
                u32::from_le_bytes(buf[pos..pos + 4].try_into().expect("in range")) as usize;
            let l_indiv =
                u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().expect("in range")) as usize;
            let record_end = pos + 8 + l_shared + l_indiv;
            let validated = Record::new(&buf[pos..record_end])
                .and_then(|r| r.validate())
                .is_ok();
            if validated {
                confirmed += 1;
                if !real.contains(&pos) {
                    false_positives += 1;
                }
            }
        }
        if here > worst_block.1 {
            worst_block = (index, here);
        }
    }
    let speculative = started.elapsed();

    println!("-- speculative validator, every byte offset of every block --");
    println!("  blocks:              {blocks_measured}");
    println!("  candidate offsets:   {candidates}");
    println!(
        "  survive stage 1:     {survivors}  ({:.6}% of candidates)",
        100.0 * survivors as f64 / candidates as f64
    );
    println!(
        "  survive stage 2:     {confirmed}  (of {} real record starts)",
        offsets.len()
    );
    println!("  FALSE POSITIVES:     {false_positives}");
    println!(
        "  survivors per block: {:.1} mean, {} worst (block {})",
        survivors as f64 / blocks_measured as f64,
        worst_block.1,
        worst_block.0
    );
    println!(
        "\n  (host wall for the whole sweep: {speculative:?} — a serial stand-in for what a\n   \
         kernel does in parallel, not a projection of kernel time)"
    );

    // ---- the comparison that decides it -----------------------------------
    let per_block_serial = serial.as_secs_f64() / blocks_measured as f64;
    println!("\n-- what this means --");
    println!(
        "  serial scan costs {:.0} ns per block of input.",
        per_block_serial * 1e9
    );
    if false_positives == 0 {
        println!("  Stage 1 + stage 2 identified record starts exactly, with no false positives.");
    } else {
        println!(
            "  {false_positives} offsets passed both stages without being record starts, so \
             chain reconciliation is load-bearing, not a formality."
        );
    }
}
