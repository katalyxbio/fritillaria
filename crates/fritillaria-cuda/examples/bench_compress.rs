//! Time to a **written BGZF file** from records already in device memory.
//!
//! The write-side counterpart of `bench_decode`, and the one number the
//! compression work has been unable to quote. `docs/compression.md` measures the
//! ratio and says plainly that throughput is unmeasured; this is what closes
//! that.
//!
//! # What is timed, and what is deliberately not
//!
//! The workload is: payloads that are **already on the device** go in, a
//! spec-valid BGZF stream comes out on the host. That is the position a GPU tool
//! is in when it has produced records and has to write them, and it is the only
//! comparison that means anything — a benchmark that started from host bytes
//! would be measuring an upload this design exists to avoid.
//!
//! So the input is staged first, outside the clock: the file is read and
//! inflated onto the device, and only then does timing start. The inflate is
//! `bench_inflate`'s subject, not this one's.
//!
//! # The baseline is run separately, and is not a like-for-like race
//!
//! `scripts/colab_bench_job.py` times `bgzip -c` on the same file and machine;
//! this example does not, because the two are not the same workload and printing
//! them adjacent invites a ratio nobody should quote. `bgzip` compresses **host**
//! bytes with every core; this compresses **device** bytes with one GPU, and a
//! GPU producer choosing `bgzip` would first have to move the *uncompressed*
//! records to the host — the transfer the compression ratio makes expensive.
//!
//! So the uncompressed input size is reported here, and the D2H it implies is
//! left to be computed against a link rate `bench_inflate` measures on the same
//! machine. Which number matters depends entirely on where the records already
//! are, and this example refuses to decide that for the reader.
//!
//! ```text
//! cargo run --release --features nvcomp -p fritillaria-cuda \
//!     --example bench_compress -- <file.bam> [chunks_per_batch ...]
//! ```

// Byte counts and nanosecond figures print to a few significant digits; f64's
// mantissa is far more than a benchmark report needs.
#![allow(clippy::cast_precision_loss)]

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(feature = "nvcomp")]
mod bench {
    use super::mib;
    use std::time::{Duration, Instant};

    use fritillaria_bgzf::{DeviceBgzfWriter, discover_blocks};
    use fritillaria_core::{
        DeviceBlockCodec, DeviceBlockCompressor, DeviceBuffer, DeviceInflateBatch,
        MAX_COMPRESSIBLE_PAYLOAD,
    };
    use fritillaria_cuda::nvcomp::ffi::DeflateAlgorithm;
    use fritillaria_cuda::{CompressTimings, NvcompCodec, NvcompCompressor};

    /// What one pass produced and how long it took.
    pub(crate) struct Pass {
        pub(crate) wall: Duration,
        pub(crate) compressed: u64,
        pub(crate) blocks: u64,
        pub(crate) batches: u64,
    }

    /// Reads and inflates the file onto the device. **Not timed** — this is the
    /// read path's job, and including it would measure `bench_inflate` again.
    ///
    /// Device-resident only: a host copy would cost ~10 GiB of RAM on the real
    /// fixture and is not needed, since the writer is driven from block
    /// boundaries computed over the device buffer's length.
    pub(crate) fn stage(path: &str) -> Result<DeviceInflateBatch, Box<dyn std::error::Error>> {
        let raw = std::fs::read(path)?;
        let spans = discover_blocks(&raw, 0)?;

        let codec = NvcompCodec::new()?;
        let mut device = DeviceInflateBatch::new();
        codec.inflate_batch_device(&raw, &spans, &mut device)?;

        Ok(device)
    }

    /// One compression pass over an already-staged batch.
    pub(crate) fn run<C: DeviceBlockCompressor>(
        data: &DeviceBuffer,
        bounds: &[usize],
        compressor: C,
        chunks_per_batch: usize,
    ) -> Result<Pass, Box<dyn std::error::Error>> {
        // A sink that discards, so disk write speed does not enter the
        // measurement — the same discipline the htslib baseline uses.
        let mut writer = DeviceBgzfWriter::new(std::io::sink(), compressor)
            .with_chunks_per_batch(chunks_per_batch);

        let started = Instant::now();
        writer.write_batch(data, bounds)?;
        let blocks = writer.blocks_written();
        let batches = writer.batches_run();
        let compressed = writer.compressed_bytes();
        writer.finish()?;
        let wall = started.elapsed();

        Ok(Pass {
            wall,
            compressed,
            blocks,
            batches,
        })
    }

    /// Chunks the staged payload at a realistic block size.
    ///
    /// A fixture's own boundaries are whatever its writer chose; for a
    /// throughput number the input should look like what a producer emits, which
    /// is full blocks.
    pub(crate) fn block_bounds(total: usize) -> Vec<usize> {
        let mut bounds: Vec<usize> = (0..total).step_by(0xff00).collect();
        bounds.push(total);
        bounds
    }

    pub(crate) fn report(uncompressed: u64, passes: &[(usize, Pass)]) {
        println!("-- compression: device payloads to a BGZF stream --\n");
        println!(
            "  {:>7}  {:>8}  {:>8}  {:>10}  {:>12}  {:>8}",
            "chunks", "batches", "blocks", "wall", "MiB/s in", "ratio"
        );
        for (chunks, p) in passes {
            let secs = p.wall.as_secs_f64().max(f64::MIN_POSITIVE);
            println!(
                "  {:>7}  {:>8}  {:>8}  {:>9.3}s  {:>12.0}  {:>7.2}x",
                chunks,
                p.batches,
                p.blocks,
                p.wall.as_secs_f64(),
                mib(uncompressed) / secs,
                uncompressed as f64 / p.compressed as f64,
            );
        }
    }

    /// Payload the ladder sweeps, in bytes.
    ///
    /// **Bounded on purpose, and the first run of this example is why.** The
    /// ladder originally swept the whole 10 GiB at every rung; `HighRatio` alone
    /// took 95s and `MaxRatio` is slower still, so the run hit `colab exec`'s
    /// inactivity timeout partway through and took the `bgzip` baseline down
    /// with it — an hour of billable VM for a table missing its last row.
    ///
    /// Comparing rungs never needed the whole file: they are being measured
    /// against each other, and a gigabyte of real WGS payload is ample for that.
    /// The full-file number comes from the sweep above, at the shipping level.
    const LADDER_BYTES: usize = 1 << 30;

    /// The ladder, timed rather than only sized.
    ///
    /// `docs/compression.md` argued the default from ratio and VRAM alone. This
    /// is the third axis, and measuring it moved the default from `4` to `2`.
    pub(crate) fn report_ladder(
        data: &DeviceBuffer,
        bounds: &[usize],
        chunks_per_batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // A prefix of the chunk list, so every rung sees the same data and the
        // slow ones stay affordable.
        let cut = bounds
            .iter()
            .position(|&b| b - bounds[0] >= LADDER_BYTES)
            .map_or(bounds.len(), |i| i + 1);
        let bounds = &bounds[..cut];
        let uncompressed = (bounds[bounds.len() - 1] - bounds[0]) as u64;

        println!(
            "\n-- the ratio/throughput ladder ({:.0} MiB, a prefix) --\n",
            mib(uncompressed)
        );
        println!(
            "  {:>14}  {:>10}  {:>12}  {:>8}  {:>14}",
            "algorithm", "wall", "MiB/s in", "ratio", "scratch/chunk"
        );
        for algorithm in [
            DeflateAlgorithm::EntropyOnly,
            DeflateAlgorithm::LowRatio,
            DeflateAlgorithm::MediumRatio,
            DeflateAlgorithm::HighRatio,
            DeflateAlgorithm::MaxRatio,
        ] {
            let compressor = NvcompCompressor::with_algorithm(0, algorithm)?;
            let scratch = compressor.budget().scratch_per_chunk;
            let pass = run(data, bounds, &compressor, chunks_per_batch)?;
            let secs = pass.wall.as_secs_f64().max(f64::MIN_POSITIVE);
            println!(
                "  {:>14}  {:>9.3}s  {:>12.0}  {:>7.2}x  {:>14}",
                format!("{algorithm:?}"),
                pass.wall.as_secs_f64(),
                mib(uncompressed) / secs,
                uncompressed as f64 / pass.compressed as f64,
                scratch,
            );
        }
        Ok(())
    }

    pub(crate) fn max_chunk() -> usize {
        MAX_COMPRESSIBLE_PAYLOAD
    }

    /// Where the time goes, over the same prefix the ladder uses.
    ///
    /// The question this answers: `bgzip -c -@11` does 355 MiB/s and this path
    /// does 108 at the shipping rung. Until something attributed that, "the GPU
    /// is slower" was an observation and not a diagnosis.
    ///
    /// Synchronised between phases, so the sum exceeds the untimed wall clock
    /// above. Read it for *shares*, not for throughput.
    pub(crate) fn report_phases(
        data: &DeviceBuffer,
        bounds: &[usize],
        chunks_per_batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cut = bounds
            .iter()
            .position(|&b| b - bounds[0] >= LADDER_BYTES)
            .map_or(bounds.len(), |i| i + 1);
        let bounds = &bounds[..cut];

        let compressor = NvcompCompressor::new(0)?;
        let mut timings = CompressTimings::default();
        let mut out = fritillaria_core::CompressedBatch::new();

        let mut start = 0;
        while start + 1 < bounds.len() {
            let end = (start + chunks_per_batch + 1).min(bounds.len());
            compressor.compress_batch_device_timed(
                data,
                &bounds[start..end],
                &mut out,
                &mut timings,
            )?;
            start = end - 1;
        }

        let total = timings.total().as_secs_f64().max(f64::MIN_POSITIVE);
        println!(
            "\n-- where the time goes ({} batches, {} blocks; synchronised, so \
             the sum exceeds wall clock) --\n",
            timings.batches, timings.blocks
        );
        println!("  {:<22}  {:>10}  {:>8}", "phase", "time", "share");
        for (name, d) in timings.phases() {
            println!(
                "  {name:<22}  {:>9.3}s  {:>7.1}%",
                d.as_secs_f64(),
                d.as_secs_f64() / total * 100.0
            );
        }
        println!("  {:<22}  {:>9.3}s", "sum", total);
        Ok(())
    }
}

#[cfg(feature = "nvcomp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fritillaria_cuda::NvcompCompressor;

    /// Chunks per batch for the two bounded stages.
    ///
    /// Fixed rather than taken from the sweep's winner, because the sweep now
    /// runs *after* them. 4096 won it on both previous runs and the difference
    /// from 1024 was under 1%.
    const DEFAULT_CHUNKS: usize = 4096;

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bench_compress <file.bam> [chunks_per_batch ...]");
        std::process::exit(2);
    };
    // 256 is dropped from the default sweep: it was the slowest row and told us
    // nothing 1024 did not, and the full-file sweep is the one stage here with
    // no bound on it.
    let sweep: Vec<usize> = {
        let given: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
        if given.is_empty() {
            vec![1024, 4096]
        } else {
            given
        }
    };

    println!("staging {path} onto the device (not timed — see the module docs)");
    let device = bench::stage(&path)?;
    let data = device.data().ok_or("the file inflated to nothing")?;
    let uncompressed = device.byte_len() as u64;
    let bounds = bench::block_bounds(device.byte_len());

    println!(
        "  {:.1} MiB of payload in {} chunks (max chunk {})\n",
        uncompressed as f64 / (1024.0 * 1024.0),
        bounds.len() - 1,
        bench::max_chunk(),
    );

    // --- ordering, and it is deliberate ------------------------------------
    //
    // Bounded stages first, the unbounded one last. The full-file sweep is the
    // only stage here whose cost is not capped, and a previous run stalled in it
    // for 34 minutes and was killed — taking the phase breakdown, which ran
    // afterwards, with it. The diagnostic never fired at the thing that was
    // broken.
    //
    // That is the third time on this benchmark that an expensive stage has eaten
    // the stages behind it. Put the cheap, load-bearing ones in front.
    let compressor = NvcompCompressor::new(0)?;

    bench::report_phases(data, &bounds, DEFAULT_CHUNKS)?;
    bench::report_ladder(data, &bounds, DEFAULT_CHUNKS)?;

    let mut passes = Vec::new();
    for chunks in &sweep {
        passes.push((*chunks, bench::run(data, &bounds, &compressor, *chunks)?));
    }
    bench::report(uncompressed, &passes);

    Ok(())
}

#[cfg(not(feature = "nvcomp"))]
fn main() {
    let _ = mib(0);
    eprintln!(
        "bench_compress needs the `nvcomp` feature: device compression is \
         nvCOMP's, and our own kernel does not compress."
    );
    std::process::exit(2);
}
