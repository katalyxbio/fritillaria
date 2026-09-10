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
    use fritillaria_cuda::{NvcompCodec, NvcompCompressor};

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

    /// The ladder, timed rather than only sized.
    ///
    /// `docs/compression.md` argues the default from ratio and VRAM. This is the
    /// third axis, and the one nothing has measured: what level 4 costs in time
    /// against the entropy-only setting Parabricks ships.
    pub(crate) fn report_ladder(
        data: &DeviceBuffer,
        bounds: &[usize],
        uncompressed: u64,
        chunks_per_batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n-- the ratio/throughput ladder --\n");
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
}

#[cfg(feature = "nvcomp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fritillaria_cuda::NvcompCompressor;

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bench_compress <file.bam> [chunks_per_batch ...]");
        std::process::exit(2);
    };
    let sweep: Vec<usize> = {
        let given: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
        if given.is_empty() {
            vec![256, 1024, 4096]
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

    let compressor = NvcompCompressor::new(0)?;
    let mut passes = Vec::new();
    for chunks in &sweep {
        passes.push((*chunks, bench::run(data, &bounds, &compressor, *chunks)?));
    }
    bench::report(uncompressed, &passes);

    let best = sweep
        .iter()
        .zip(&passes)
        .min_by_key(|(_, (_, p))| p.wall)
        .map_or(1024, |(c, _)| *c);
    bench::report_ladder(data, &bounds, uncompressed, best)?;

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
