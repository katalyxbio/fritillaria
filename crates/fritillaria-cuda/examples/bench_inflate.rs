//! Where does BGZF decompression time actually go?
//!
//! ```text
//! cargo run --release --features cuda --example bench_inflate -- <file.bam> [chunk_mib]
//! ```
//!
//! Streams the file in bounded chunks, so a 3 GiB BAM does not need 9 GiB of
//! RAM to inflate. Reports a per-phase breakdown — host planning, H2D, kernel,
//! D2H, verification — plus CPU-reference throughput on the same machine.
//!
//! With `--features nvcomp` it runs **both** GPU codecs over the same file in
//! one process, which is the only fair way to compare them: same data, same
//! machine, same chunking, back to back.
//!
//! # Reading the output honestly
//!
//! - **The phase breakdown forces serialisation.** Phases are separated by
//!   stream synchronisation so each can be attributed, which removes the
//!   overlap a pipelined implementation would get. The sum is an upper bound on
//!   achievable wall-clock, not a measurement of it.
//! - **The CPU number is not a baseline unless this machine has real cores.**
//!   On a 2-vCPU Colab VM it is meaningless and the report says so. A CPU
//!   baseline worth quoting is htslib/libdeflate on real hardware; our
//!   `miniz_oxide` reference is a correctness oracle, not a performance target.
//! - Never compare a GPU number from one machine with a CPU number from
//!   another without labelling both.

// Reporting-only arithmetic: f64 precision is irrelevant for MiB/s figures,
// and splitting the report into more functions would obscure it.
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use std::io::Read;
use std::time::{Duration, Instant};

use fritillaria_bgzf::CpuCodec;
use fritillaria_core::{BlockCodec, BlockSpan, InflateBatch};

/// Compressed bytes read per chunk. Bounds memory; each chunk is one or more
/// kernel launches worth of blocks.
const DEFAULT_CHUNK_MIB: usize = 64;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Throughput in MiB/s, guarding against a zero-length measurement.
fn rate(bytes: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return f64::NAN;
    }
    mib(bytes) / elapsed.as_secs_f64()
}

/// Reads the next chunk, keeping any trailing partial block for the next round.
///
/// Returns the spans found and the bytes they index into.
fn next_chunk<R: Read>(
    reader: &mut R,
    carry: &mut Vec<u8>,
    chunk_bytes: usize,
    file_offset: &mut u64,
) -> std::io::Result<Option<(Vec<u8>, Vec<BlockSpan>)>> {
    let mut buf = std::mem::take(carry);
    let base = *file_offset - buf.len() as u64;
    let start = buf.len();
    buf.resize(start + chunk_bytes, 0);

    let mut filled = start;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    buf.truncate(filled);
    *file_offset = base + filled as u64;

    if filled == start {
        return Ok(None); // nothing new; stream is done
    }

    // A truncated prefix of a real BAM is expected here: block discovery stops
    // cleanly at the first incomplete block rather than erroring.
    let mut discovery = fritillaria_bgzf::BlockDiscovery::new(&buf, base);
    let mut spans = Vec::new();
    for span in discovery.by_ref() {
        match span {
            Ok(span) => spans.push(span),
            Err(_) => break,
        }
    }
    let consumed = discovery.position();
    carry.clear();
    carry.extend_from_slice(&buf[consumed..]);

    if spans.is_empty() {
        return Ok(None);
    }
    Ok(Some((buf, spans)))
}

/// Runs one codec over the whole file on the **device-resident** path and
/// reports where the time went.
///
/// Device-resident on purpose. The host path's D2H copy was 47% of runtime and
/// is pure waste for the intended consumer, so measuring it would bury the
/// thing being compared — the inflate itself — under a transfer neither codec
/// influences. The `kernel` row is the number that decides which codec ships.
#[cfg(feature = "cuda")]
fn bench_gpu<F>(
    label: &str,
    path: &str,
    chunk_bytes: usize,
    size: u64,
    mut inflate: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(
        &[u8],
        &[BlockSpan],
        &mut fritillaria_core::DeviceInflateBatch,
        &mut fritillaria_cuda::InflateTimings,
    ) -> fritillaria_core::Result<()>,
{
    use fritillaria_core::DeviceInflateBatch;
    use fritillaria_cuda::InflateTimings;

    let mut file = std::fs::File::open(path)?;
    let mut carry = Vec::new();
    let mut offset = 0u64;
    let mut out = DeviceInflateBatch::new();
    let mut totals = InflateTimings::default();
    let wall = Instant::now();

    while let Some((buf, spans)) = next_chunk(&mut file, &mut carry, chunk_bytes, &mut offset)? {
        let mut batch = InflateTimings::default();
        inflate(&buf, &spans, &mut out, &mut batch)?;
        totals.accumulate(&batch);
    }
    let wall = wall.elapsed();

    let inflated = totals.inflated_bytes;
    println!("-- stage 2+3: device-resident inflate + CRC32 -- {label}");
    println!("  blocks:      {}", totals.blocks);
    println!("  compressed:  {:.2} MiB", mib(totals.compressed_bytes));
    println!("  inflated:    {:.2} MiB", mib(inflated));
    println!(
        "  ratio:       {:.2}x",
        inflated as f64 / totals.compressed_bytes as f64
    );
    println!();
    println!("  phase          time          share   throughput (inflated)");
    for (name, d) in [
        ("plan (host)", totals.plan),
        ("upload H2D", totals.upload),
        ("kernel", totals.kernel),
        ("download D2H", totals.download),
        ("verify (host)", totals.verify),
    ] {
        let share = 100.0 * d.as_secs_f64() / totals.total().as_secs_f64();
        println!(
            "  {name:<14} {:>10.3}s   {share:>5.1}%   {:>8.0} MiB/s",
            d.as_secs_f64(),
            rate(inflated, d)
        );
    }
    println!(
        "  {:<14} {:>10.3}s",
        "sum of phases",
        totals.total().as_secs_f64()
    );
    println!(
        "  {:<14} {:>10.3}s   <- includes file read",
        "wall clock",
        wall.as_secs_f64()
    );
    println!(
        "\n  end-to-end: {:.0} MiB/s inflated, {:.0} MiB/s of compressed input\n",
        rate(inflated, wall),
        rate(size, wall)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: bench_inflate <file.bam> [chunk_mib]")?;
    let chunk_bytes = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CHUNK_MIB)
        * 1024
        * 1024;

    let size = std::fs::metadata(&path)?.len();
    println!("file:   {path}");
    println!("size:   {:.2} MiB", mib(size));
    println!("chunk:  {} MiB\n", chunk_bytes / 1024 / 1024);

    // ---- host-side block discovery, measured on its own ----
    // Stage 1 of the pipeline. If this dominates, no kernel work matters.
    let mut file = std::fs::File::open(&path)?;
    let mut discovery_time = Duration::ZERO;
    let mut total_blocks = 0u64;
    let mut compressed_seen = 0u64;
    {
        let mut carry = Vec::new();
        let mut offset = 0u64;
        loop {
            let started = Instant::now();
            let chunk = next_chunk(&mut file, &mut carry, chunk_bytes, &mut offset)?;
            let elapsed = started.elapsed();
            match chunk {
                Some((_, spans)) => {
                    discovery_time += elapsed;
                    total_blocks += spans.len() as u64;
                    compressed_seen += spans.iter().map(|s| s.payload_len as u64).sum::<u64>();
                }
                None => break,
            }
        }
    }
    println!("-- stage 1: block discovery (host, includes file read) --");
    println!("  blocks:     {total_blocks}");
    println!("  elapsed:    {discovery_time:?}");
    println!(
        "  throughput: {:.0} MiB/s of compressed input\n",
        rate(size, discovery_time)
    );

    #[cfg(feature = "cuda")]
    {
        use fritillaria_cuda::CudaContext;

        match CudaContext::new(0) {
            Err(err) => println!("-- GPU: skipped ({err}) --\n"),
            Ok(ctx) => bench_gpu(
                "our kernel (kernels/inflate.cu)",
                &path,
                chunk_bytes,
                size,
                |buf, spans, out, timings| ctx.inflate_batch_device_timed(buf, spans, out, timings),
            )?,
        }
    }

    #[cfg(feature = "nvcomp")]
    {
        use fritillaria_cuda::NvcompContext;

        match NvcompContext::new(0) {
            // Absent nvCOMP is normal; say so rather than reporting nothing.
            Err(err) => println!("-- nvCOMP: skipped ({err}) --\n"),
            Ok(ctx) => {
                println!(
                    "  (nvcomp {}, input alignment {})",
                    ctx.nvcomp_version(),
                    ctx.alignments().input
                );
                bench_gpu(
                    "nvCOMP batched DEFLATE",
                    &path,
                    chunk_bytes,
                    size,
                    |buf, spans, out, timings| {
                        ctx.inflate_batch_device_timed(buf, spans, out, timings)
                    },
                )?;
            }
        }
    }

    // ---- CPU reference on the same machine ----
    {
        let mut file = std::fs::File::open(&path)?;
        let mut carry = Vec::new();
        let mut offset = 0u64;
        let mut out = InflateBatch::new();
        let codec = CpuCodec::new();
        let mut inflated = 0u64;
        let wall = Instant::now();

        while let Some((buf, spans)) = next_chunk(&mut file, &mut carry, chunk_bytes, &mut offset)?
        {
            codec.inflate_batch(&buf, &spans, &mut out)?;
            inflated += out.data().len() as u64;
        }
        let wall = wall.elapsed();

        println!("-- CPU reference (miniz_oxide, single-threaded) --");
        println!("  inflated:   {:.2} MiB", mib(inflated));
        println!("  wall clock: {:.3}s", wall.as_secs_f64());
        println!("  throughput: {:.0} MiB/s inflated", rate(inflated, wall));
        println!(
            "\n  NOTE: only a baseline if this machine has real cores. On a 2-vCPU\n\
             \x20       cloud VM it is meaningless and must not be quoted as a speedup.\n\
             \x20       A baseline worth publishing is htslib/libdeflate on real hardware."
        );
    }

    let _ = compressed_seen;
    Ok(())
}
