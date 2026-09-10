//! Time to **records** in device memory.
//!
//! `bench_inflate` measures decompression and stops there, which has been the
//! honest limit of every number in this project: it says how fast bytes land in
//! VRAM, not how fast *records* do. That gap is the thing the design is argued
//! on, so this closes it.
//!
//! What is measured is the shipping path, not a rig: `DeviceBgzfReader` feeding
//! `BamDecoder`, exactly as `tests/bam_reader.rs` drives it. Nothing comes back
//! to the host except the BAM header — which cannot be avoided, since
//! `records_start` is only knowable by reading it — and the three totals per
//! batch that decide how much to allocate.
//!
//! Two things it reports that a single number would hide:
//!
//! - **A sweep over batch size.** The reconcile phase is a single GPU thread
//!   walking the record chain over *blocks*, so its cost scales with the block
//!   count per batch. At 256 blocks that is irrelevant; the sweep is there to
//!   find where it stops being irrelevant, rather than assuming.
//! - **Untimed wall clock separately from the phase breakdown.** Attributing
//!   phases means synchronising between them, which removes overlap the real
//!   path gets. The breakdown is a map of where time goes; the wall clock from
//!   the untimed pass is the number to quote.
//!
//! ```text
//! cargo run --release --features cuda -p fritillaria-cuda \
//!     --example bench_decode -- <file.bam> [blocks_per_batch ...]
//! ```

// Byte counts and nanosecond figures are printed to a few significant digits;
// f64's mantissa is far more than a benchmark report needs.
#![allow(clippy::cast_precision_loss)]

#[cfg(feature = "cuda")]
use std::time::Duration;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// Only the `cuda` build reports throughput; without it this example is a stub
// that prints why it did nothing.
#[cfg(feature = "cuda")]
fn rate(bytes: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return f64::NAN;
    }
    mib(bytes) / elapsed.as_secs_f64()
}

#[cfg(feature = "cuda")]
mod bench {
    use super::{mib, rate};
    use std::time::{Duration, Instant};

    use fritillaria_bam::columnar::header::parse_header;
    use fritillaria_bgzf::DeviceBgzfReader;
    use fritillaria_core::DeviceBlockCodec;
    use fritillaria_cuda::{BamDecoder, DecodeTimings};

    /// One pass over the file. Returns wall clock split into the two stages a
    /// caller actually waits on, plus what came out.
    pub(crate) struct Pass {
        pub(crate) read_inflate: Duration,
        pub(crate) decode: Duration,
        pub(crate) records: u64,
        pub(crate) inflated: u64,
        pub(crate) batches: u64,
    }

    /// Reads the whole file into device columns, timing the two stages.
    ///
    /// `timings`, when given, turns on per-phase attribution inside the decode.
    /// That synchronises between phases, so a pass with it on is slower and its
    /// `decode` total is an upper bound — never quote the two together.
    pub(crate) fn run<C: DeviceBlockCodec>(
        path: &str,
        codec: &C,
        decoder: &BamDecoder,
        blocks_per_batch: usize,
        mut timings: Option<&mut DecodeTimings>,
    ) -> Result<Pass, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let mut reader = DeviceBgzfReader::new(file, codec).with_blocks_per_batch(blocks_per_batch);

        let mut pass = Pass {
            read_inflate: Duration::ZERO,
            decode: Duration::ZERO,
            records: 0,
            inflated: 0,
            batches: 0,
        };
        let mut header_parsed = false;

        loop {
            let started = Instant::now();
            let Some(batch) = reader.next_batch()? else {
                pass.read_inflate += started.elapsed();
                break;
            };
            pass.read_inflate += started.elapsed();
            pass.batches += 1;
            pass.inflated += batch.data.byte_len() as u64;

            let begin = if header_parsed {
                batch.start
            } else {
                // The one unavoidable host read, once per file.
                let host = batch.data.to_host()?;
                let header = parse_header(host.data())?;
                header_parsed = true;
                header.records_start
            };

            let started = Instant::now();
            let records = match timings.as_deref_mut() {
                Some(t) => decoder.decode_timed(&batch.data, begin, t)?,
                None => decoder.decode(&batch.data, begin)?,
            };
            pass.decode += started.elapsed();

            pass.records += records.len() as u64;
            let tail = records.tail();
            drop(records);
            reader.carry_from(tail)?;
        }

        Ok(pass)
    }

    pub(crate) fn report_sweep(size: u64, passes: &[(usize, Pass)]) {
        println!("-- time to records in device memory --\n");
        println!(
            "  {:>7}  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}",
            "blocks", "batches", "read+infl", "decode", "wall", "records/s"
        );
        for (blocks, p) in passes {
            let wall = p.read_inflate + p.decode;
            println!(
                "  {:>7}  {:>8}  {:>9.3}s  {:>9.3}s  {:>9.3}s  {:>12.0}",
                blocks,
                p.batches,
                p.read_inflate.as_secs_f64(),
                p.decode.as_secs_f64(),
                wall.as_secs_f64(),
                p.records as f64 / wall.as_secs_f64().max(f64::MIN_POSITIVE),
            );
        }

        if let Some((_, best)) = passes.iter().min_by_key(|(_, p)| p.read_inflate + p.decode) {
            let wall = best.read_inflate + best.decode;
            println!();
            println!("  records:    {}", best.records);
            println!("  inflated:   {:.2} MiB", mib(best.inflated));
            println!(
                "  best wall:  {:.3}s  ({:.0} MiB/s of compressed input, \
                 {:.0} MiB/s of records)",
                wall.as_secs_f64(),
                rate(size, wall),
                rate(best.inflated, wall),
            );
            println!();
        }
    }

    pub(crate) fn report_phases(t: &DecodeTimings) {
        let total = t.total();
        println!("-- decode phases (synchronised; an upper bound, not wall clock) --\n");
        println!("  {:>12}  {:>10}  {:>7}", "phase", "time", "share");
        for (name, d) in [
            ("setup", t.setup),
            ("scan", t.scan),
            ("reconcile", t.reconcile),
            ("totals", t.totals),
            ("emit", t.emit),
            ("decode", t.decode),
        ] {
            let share = if total.is_zero() {
                0.0
            } else {
                100.0 * d.as_secs_f64() / total.as_secs_f64()
            };
            println!("  {name:>12}  {:>9.3}s  {share:>6.1}%", d.as_secs_f64());
        }
        println!("  {:>12}  {:>9.3}s", "sum", total.as_secs_f64());
        println!();
        println!("  batches:  {}", t.batches);
        println!("  blocks:   {}", t.blocks);
        println!("  records:  {}", t.records);
        println!(
            "  reconcile per block: {:.0} ns",
            if t.blocks == 0 {
                0.0
            } else {
                t.reconcile.as_secs_f64() * 1e9 / t.blocks as f64
            }
        );
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: bench_decode <file.bam> [blocks_per_batch ...]")?;
    let sweep: Vec<usize> = {
        let rest: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if rest.is_empty() {
            vec![64, 256, 1024, 4096]
        } else {
            rest
        }
    };

    let size = std::fs::metadata(&path)?.len();
    println!("file:   {path}");
    println!("size:   {:.2} MiB\n", mib(size));

    #[cfg(feature = "cuda")]
    {
        let decoder = match BamDecoder::new() {
            Ok(decoder) => decoder,
            Err(err) => {
                println!("-- skipped: no usable CUDA device ({err}) --");
                return Ok(());
            }
        };

        // Use the codec that actually ships. nvCOMP is 5.5x our kernel on the
        // inflate phase, so measuring the fallback and calling the result
        // "time to records" would understate it by most of that gap.
        #[cfg(feature = "nvcomp")]
        let fast = fritillaria_cuda::NvcompCodec::new().ok();

        #[cfg(feature = "nvcomp")]
        if let Some(codec) = fast {
            println!("codec:  nvcomp (the shipping fast path)\n");
            measure(&path, &codec, &decoder, size, &sweep)?;
            return Ok(());
        }

        println!("codec:  cuda (our kernel; nvCOMP unavailable)\n");
        let codec = CudaCodec::new()?;
        measure(&path, &codec, &decoder, size, &sweep)?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = sweep;
        println!("built without the `cuda` feature; nothing to measure");
    }

    Ok(())
}

/// Runs the sweep and the phase breakdown for one codec.
#[cfg(feature = "cuda")]
fn measure<C: fritillaria_core::DeviceBlockCodec>(
    path: &str,
    codec: &C,
    decoder: &BamDecoder,
    size: u64,
    sweep: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    // Untimed: the number to quote.
    let mut passes = Vec::new();
    for blocks in sweep {
        passes.push((*blocks, bench::run(path, codec, decoder, *blocks, None)?));
    }
    bench::report_sweep(size, &passes);

    // Timed: the map of where the decode spends itself, at the default.
    let mut timings = DecodeTimings::default();
    let default_blocks = sweep
        .iter()
        .copied()
        .find(|&b| b == 256)
        .unwrap_or(sweep[0]);
    bench::run(path, codec, decoder, default_blocks, Some(&mut timings))?;
    println!("(phase breakdown at {default_blocks} blocks per batch)");
    bench::report_phases(&timings);
    Ok(())
}

#[cfg(feature = "cuda")]
use fritillaria_cuda::{BamDecoder, CudaCodec, DecodeTimings};
