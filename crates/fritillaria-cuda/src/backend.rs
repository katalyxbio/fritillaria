//! Device context and kernel launches.
//!
//! Compiled only with the `cuda` feature. Device memory and stream management
//! live here and nowhere else — no raw pointer or device-tied lifetime is
//! allowed to escape into the format crates.

use std::any::Any;
use std::fmt;
use std::panic;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::{
    CudaContext as RawContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use fritillaria_core::{
    BlockSpan, DeviceAlloc, DeviceBuffer, DeviceInflateBatch, Error, InflateBatch, MAX_BLOCK_SIZE,
    Result,
};

use crate::{CRC32_KERNEL_SRC, INFLATE_KERNEL_SRC};

/// Threads per block. 256 is a reasonable default for a memory-bound kernel
/// with one thread per BGZF block; tune against a real device before changing.
const BLOCK_DIM: u32 = 256;

/// An initialised CUDA device with this crate's kernels loaded.
///
/// Construction compiles kernels with NVRTC, so it is comparatively expensive —
/// create one and reuse it, do not build one per batch.
#[derive(Debug)]
pub struct CudaContext {
    #[allow(dead_code)]
    ctx: Arc<RawContext>,
    stream: Arc<CudaStream>,
    crc32: CudaFunction,
    inflate: CudaFunction,
    ordinal: i32,
}

pub(crate) fn driver_err(context: &str) -> impl FnOnce(cudarc::driver::DriverError) -> Error + '_ {
    move |err| Error::Cuda(format!("{context}: {err}"))
}

/// NVRTC-compiles one kernel source and loads a function out of it.
pub(crate) fn load_kernel(
    ctx: &Arc<RawContext>,
    src: &str,
    file: &str,
    function: &str,
) -> Result<CudaFunction> {
    let ptx = compile_ptx(src)
        .map_err(|err| Error::Cuda(format!("NVRTC failed to compile {file}: {err}")))?;
    let module = ctx
        .load_module(ptx)
        .map_err(driver_err(&format!("loading module for {file}")))?;
    module
        .load_function(function)
        .map_err(driver_err(&format!("loading {function}")))
}

/// Maps a device status code to a message. Mirrors the `FR_ERR_*` constants in
/// `kernels/inflate.cu`; keep the two in sync.
fn inflate_status_message(status: u32) -> &'static str {
    match status {
        1 => "truncated deflate stream",
        2 => "reserved deflate block type",
        3 => "stored block LEN/NLEN mismatch",
        4 => "output overran the size declared by ISIZE",
        5 => "invalid Huffman code",
        6 => "invalid or out-of-range back-reference distance",
        7 => "malformed Huffman table",
        8 => "invalid code-length repeat",
        9 => "code counts exceed the DEFLATE maximum",
        _ => "unknown device error",
    }
}

/// Where every block reads from and writes to, computed before any device work.
///
/// The output offsets are a prefix sum over `ISIZE`. Knowing each block's
/// output position up front, without decompressing anything, is precisely what
/// lets all the blocks be inflated at once.
pub(crate) struct BatchPlan {
    pub(crate) in_offsets: Vec<u64>,
    pub(crate) in_lengths: Vec<u32>,
    pub(crate) out_offsets: Vec<u64>,
    pub(crate) out_caps: Vec<u32>,
    pub(crate) total: usize,
}

impl BatchPlan {
    pub(crate) fn new(batch: &[u8], spans: &[BlockSpan]) -> Result<Self> {
        let mut plan = Self {
            in_offsets: Vec::with_capacity(spans.len()),
            in_lengths: Vec::with_capacity(spans.len()),
            out_offsets: Vec::with_capacity(spans.len()),
            out_caps: Vec::with_capacity(spans.len()),
            total: 0,
        };
        let mut total: u64 = 0;

        for span in spans {
            if span.isize as usize > MAX_BLOCK_SIZE {
                return Err(Error::InvalidBlock {
                    offset: span.compressed_offset,
                    reason: "ISIZE exceeds 64 KiB",
                });
            }
            // Bounds-checked on the host: out of range here would be an
            // out-of-bounds read on the device, which is far harder to debug.
            span.payload_start
                .checked_add(span.payload_len)
                .filter(|end| *end <= batch.len())
                .ok_or(Error::InvalidBlock {
                    offset: span.compressed_offset,
                    reason: "payload extends past the end of the batch",
                })?;

            plan.in_offsets.push(span.payload_start as u64);
            plan.in_lengths.push(span.payload_len as u32);
            plan.out_offsets.push(total);
            plan.out_caps.push(span.isize);
            total += u64::from(span.isize);
        }

        plan.total = usize::try_from(total)
            .map_err(|_| Error::Cuda("batch output size exceeds usize".to_string()))?;
        Ok(plan)
    }
}

/// Per-phase wall-clock for one `inflate_batch`, for finding the bottleneck.
///
/// The phases are separated by stream synchronisation so each can be attributed
/// individually. That **prevents the overlap a pipelined implementation would
/// get**, so the sum is an upper bound on real wall-clock, not a measurement of
/// it. Use it to see where time goes, not to quote total throughput.
#[derive(Clone, Copy, Debug, Default)]
pub struct InflateTimings {
    /// Host-side planning: bounds checks and the ISIZE prefix sum.
    pub plan: Duration,
    /// Host-to-device copy of the compressed batch and its span arrays.
    pub upload: Duration,
    /// Kernel execution, including the folded-in CRC32.
    pub kernel: Duration,
    /// Device-to-host copy of the decompressed output.
    pub download: Duration,
    /// Host-side CRC/ISIZE checking and batch assembly.
    pub verify: Duration,

    pub blocks: u64,
    pub compressed_bytes: u64,
    pub inflated_bytes: u64,
}

impl InflateTimings {
    /// Folds another batch's timings into this one.
    pub fn accumulate(&mut self, other: &Self) {
        self.plan += other.plan;
        self.upload += other.upload;
        self.kernel += other.kernel;
        self.download += other.download;
        self.verify += other.verify;
        self.blocks += other.blocks;
        self.compressed_bytes += other.compressed_bytes;
        self.inflated_bytes += other.inflated_bytes;
    }

    /// Sum of the measured phases.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.plan + self.upload + self.kernel + self.download + self.verify
    }
}

/// Device-side inputs and output buffers for one launch.
struct Uploaded {
    input: CudaSlice<u8>,
    in_offsets: CudaSlice<u64>,
    in_lengths: CudaSlice<u32>,
    out_offsets: CudaSlice<u64>,
    out_caps: CudaSlice<u32>,
    output: CudaSlice<u8>,
    status_buf: CudaSlice<u32>,
    produced_buf: CudaSlice<u32>,
    crc_buf: CudaSlice<u32>,
}

/// What the kernel produced. The payload stays on the device; only the small
/// per-block verification arrays come back.
///
/// For a 3 GiB BAM those arrays are 3 x 664 KB against ~10 GiB of payload, so
/// the host path pays one extra download and the device path pays none.
pub(crate) struct DeviceOutput {
    pub(crate) data: CudaSlice<u8>,
    /// Recorded on the stream immediately after the inflate launch, so a
    /// consumer can order against completion without a host synchronise.
    pub(crate) ready: CudaEvent,
    /// Bytes actually written across all blocks. The allocation may be one byte
    /// larger — a batch of nothing but EOF blocks inflates to zero bytes, and a
    /// zero-sized device allocation is not valid.
    pub(crate) total: usize,
    pub(crate) status: Vec<u32>,
    pub(crate) produced: Vec<u32>,
    pub(crate) crcs: Vec<u32>,
}

/// Checks every block against its trailer and returns the block boundaries.
///
/// Verification is mandatory, not a policy knob — see the [`BlockCodec`]
/// contract. A silently corrupt read in a genomics pipeline is worse than a
/// slow one, so any disagreement fails the whole batch.
///
/// **Every path goes through this function** — host, device, our own kernel and
/// nvCOMP alike. Engines differ only in how they spell a failure code, which is
/// what `describe` is for. A second copy of this logic is exactly how one of
/// those paths would drift into skipping a check the others still make.
pub(crate) fn verify_and_offsets(
    spans: &[BlockSpan],
    out: &DeviceOutput,
    describe: &dyn Fn(u32) -> String,
) -> Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(spans.len() + 1);
    offsets.push(0usize);
    let mut cursor = 0usize;

    for (i, span) in spans.iter().enumerate() {
        if out.status[i] != 0 {
            return Err(Error::Inflate {
                offset: span.compressed_offset,
                reason: describe(out.status[i]),
            });
        }
        if out.produced[i] != span.isize {
            return Err(Error::SizeMismatch {
                offset: span.compressed_offset,
                expected: span.isize,
                actual: out.produced[i] as usize,
            });
        }
        if out.crcs[i] != span.crc32 {
            return Err(Error::ChecksumMismatch {
                offset: span.compressed_offset,
                expected: span.crc32,
                actual: out.crcs[i],
            });
        }
        cursor += out.produced[i] as usize;
        offsets.push(cursor);
    }

    Ok(offsets)
}

/// A device allocation owned by this backend.
///
/// The concrete type behind a [`DeviceBuffer`](fritillaria_core::DeviceBuffer).
/// Format crates hold it as an opaque handle; only this crate downcasts it back
/// to recover the underlying slice.
///
/// Dropping this frees the device memory. A consumer whose kernel is still
/// reading it must keep the owning batch alive.
pub struct CudaAlloc {
    pub(crate) slice: CudaSlice<u8>,
    /// Logical length. May be less than the allocation: see [`DeviceOutput`].
    pub(crate) len: usize,
    pub(crate) stream: Arc<CudaStream>,
    /// Recorded after the inflate kernel; see [`CudaAlloc::ready_event`].
    pub(crate) ready: CudaEvent,
    pub(crate) ordinal: i32,
}

impl CudaAlloc {
    /// The device slice, for launching a kernel over it.
    ///
    /// Only the first [`byte_len`](DeviceAlloc::byte_len) bytes are meaningful.
    #[must_use]
    pub fn slice(&self) -> &CudaSlice<u8> {
        &self.slice
    }

    /// The stream this allocation's work was queued on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// An event recorded once the inflate kernel has produced these bytes.
    ///
    /// A consumer on its **own** stream orders against this without stalling
    /// the host:
    ///
    /// ```ignore
    /// my_stream.wait(alloc.ready_event())?;
    /// launch_my_kernel(alloc.slice(), &my_stream);
    /// ```
    ///
    /// Without it, a consumer on a different stream would have to synchronise
    /// the host — trading the PCIe stall this design removes for a latency one.
    #[must_use]
    pub fn ready_event(&self) -> &CudaEvent {
        &self.ready
    }
}

// The slice is the only large field and prints as a pointer and length; the
// stream would add noise without adding information.
impl fmt::Debug for CudaAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaAlloc")
            .field("len", &self.len)
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

impl DeviceAlloc for CudaAlloc {
    fn byte_len(&self) -> usize {
        self.len
    }

    fn device_ordinal(&self) -> i32 {
        self.ordinal
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::InvalidDeviceBatch {
                reason: format!(
                    "copy_to_host destination is {} bytes, allocation is {}",
                    dst.len(),
                    self.len
                ),
            });
        }
        if self.len == 0 {
            return Ok(());
        }

        // Slice to the logical length: the allocation may carry a pad byte.
        let view = self.slice.slice(0..self.len);
        self.stream
            .memcpy_dtoh(&view, dst)
            .map_err(driver_err("downloading device buffer"))?;
        // The copy is queued on the stream; the caller is holding a host buffer
        // and expects it filled on return.
        self.stream
            .synchronize()
            .map_err(driver_err("synchronising after download"))?;
        Ok(())
    }
}

/// Whether the CUDA driver library could be loaded at all.
///
/// cudarc **panics** rather than returning an error when `libcuda` cannot be
/// dlopened, which would turn "no GPU on this machine" into a test failure
/// instead of a skip. So the first probe runs behind `catch_unwind` with the
/// panic hook silenced, and the answer is cached: the hook swap is
/// process-global, and doing it once keeps concurrent tests from racing on it.
#[must_use]
pub(crate) fn driver_is_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let loaded = panic::catch_unwind(|| RawContext::new(0).is_ok()).unwrap_or(false);
        panic::set_hook(previous);
        loaded
    })
}

impl CudaContext {
    /// Opens device `ordinal` on a context of our own and compiles the kernels.
    ///
    /// Convenience for standalone use, benchmarks and tests. **A caller that
    /// already has a CUDA context should use
    /// [`with_context`](CudaContext::with_context) instead** — see there for
    /// why it matters.
    ///
    /// Returns [`Error::CudaUnavailable`] when there is no driver or no device,
    /// which callers should treat as "skip", not "fail".
    pub fn new(ordinal: usize) -> Result<Self> {
        if !driver_is_available() {
            return Err(Error::CudaUnavailable(
                "CUDA driver library not found (no GPU on this machine?)".to_string(),
            ));
        }

        let ctx = RawContext::new(ordinal).map_err(driver_err("opening device"))?;
        let stream = ctx.default_stream();
        Self::with_context(ctx, stream)
    }

    /// Builds a context around a CUDA context and stream the **caller** owns.
    ///
    /// This is the entry point that makes the library embeddable. A GPU tool
    /// already has a context and streams; if we allocated in a context of our
    /// own, every device buffer we produced would be unusable to it without a
    /// peer copy — which is exactly the transfer the device path exists to
    /// avoid. Allocating in the caller's context is what makes the handoff
    /// free.
    ///
    /// Work is queued on `stream`, and output buffers carry it (see
    /// [`CudaAlloc::stream`]) so a consumer can order against it.
    ///
    /// `stream` must belong to `ctx`; a mismatch is rejected rather than
    /// producing memory the caller cannot address.
    pub fn with_context(ctx: Arc<RawContext>, stream: Arc<CudaStream>) -> Result<Self> {
        if stream.context() != &ctx {
            return Err(Error::Cuda(
                "stream does not belong to the given context".to_string(),
            ));
        }

        let crc32 = load_kernel(&ctx, CRC32_KERNEL_SRC, "crc32.cu", "crc32_blocks")?;
        // A separate module: NVRTC compiles each source independently.
        let inflate = load_kernel(&ctx, INFLATE_KERNEL_SRC, "inflate.cu", "inflate_blocks")?;

        let ordinal = ctx.ordinal();
        Ok(Self {
            ctx,
            stream,
            crc32,
            inflate,
            ordinal: i32::try_from(ordinal)
                .map_err(|_| Error::Cuda(format!("device ordinal {ordinal} exceeds i32")))?,
        })
    }

    /// The CUDA context this was built on, for a caller that needs to allocate
    /// alongside our output.
    #[must_use]
    pub fn raw_context(&self) -> &Arc<RawContext> {
        &self.ctx
    }

    /// The stream work is queued on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Which device this context is bound to.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Inflates a batch of BGZF blocks on the device.
    ///
    /// Implements the [`BlockCodec`](fritillaria_core::BlockCodec) contract:
    /// every block's CRC32 and `ISIZE` are verified, and a mismatch fails the
    /// whole batch. The CRC is computed on-device inside the same kernel, so
    /// verification costs no extra pass over the data.
    pub fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()> {
        out.clear();
        if spans.is_empty() {
            return Ok(());
        }

        let mut timings = InflateTimings::default();
        self.inflate_batch_timed(batch, spans, out, &mut timings)
    }

    /// [`CudaContext::inflate_batch`], recording where the time went.
    ///
    /// Identical behaviour, plus a per-phase breakdown. See [`InflateTimings`]
    /// for why the phases do not sum to pipelined wall-clock.
    pub fn inflate_batch_timed(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
        timings: &mut InflateTimings,
    ) -> Result<()> {
        out.clear();
        if spans.is_empty() {
            return Ok(());
        }

        let started = Instant::now();
        let plan = BatchPlan::new(batch, spans)?;
        timings.plan += started.elapsed();
        timings.blocks += spans.len() as u64;
        timings.compressed_bytes += spans.iter().map(|s| s.payload_len as u64).sum::<u64>();
        timings.inflated_bytes += plan.total as u64;

        let device = self.launch_inflate(batch, &plan, timings)?;

        // The phase this whole design exists to remove: the entire inflated
        // payload crosses PCIe back to the host. `inflate_batch_device` skips
        // it; everything else is shared.
        let started = Instant::now();
        let mut data = self
            .stream
            .clone_dtoh(&device.data)
            .map_err(driver_err("downloading output"))?;
        data.truncate(device.total);
        timings.download += started.elapsed();

        let started = Instant::now();
        let offsets = verify_and_offsets(spans, &device, &|code| {
            inflate_status_message(code).to_string()
        })?;
        // Takes ownership so the decompressed batch is moved, not copied.
        *out = InflateBatch::from_parts(data, offsets)
            .ok_or_else(|| Error::Cuda("inconsistent output layout".to_string()))?;
        timings.verify += started.elapsed();
        Ok(())
    }

    /// Inflates a batch, leaving the output in device memory.
    ///
    /// The device-resident counterpart of
    /// [`inflate_batch`](CudaContext::inflate_batch), and the reason this
    /// backend exists: for a consumer that is another GPU kernel, the inflated
    /// payload never crosses PCIe at all.
    ///
    /// Verification is unchanged and still mandatory — it goes through the same
    /// [`verify_and_offsets`] the host path uses. What that costs here is a
    /// download of `3 * 4 * n_blocks` bytes of per-block status, which for a
    /// 3 GiB BAM is ~2 MB against ~10 GiB of payload left on the device.
    pub fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        let mut timings = InflateTimings::default();
        self.inflate_batch_device_timed(batch, spans, out, &mut timings)
    }

    /// [`CudaContext::inflate_batch_device`], recording where the time went.
    pub fn inflate_batch_device_timed(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
        timings: &mut InflateTimings,
    ) -> Result<()> {
        out.clear();
        if spans.is_empty() {
            return Ok(());
        }

        let started = Instant::now();
        let plan = BatchPlan::new(batch, spans)?;
        timings.plan += started.elapsed();
        timings.blocks += spans.len() as u64;
        timings.compressed_bytes += spans.iter().map(|s| s.payload_len as u64).sum::<u64>();
        timings.inflated_bytes += plan.total as u64;

        let device = self.launch_inflate(batch, &plan, timings)?;

        let started = Instant::now();
        let offsets = verify_and_offsets(spans, &device, &|code| {
            inflate_status_message(code).to_string()
        })?;
        let buffer = DeviceBuffer::new(Box::new(CudaAlloc {
            slice: device.data,
            len: device.total,
            stream: self.stream.clone(),
            ready: device.ready,
            ordinal: self.ordinal,
        }));
        out.adopt(buffer, offsets)?;
        timings.verify += started.elapsed();
        Ok(())
    }

    /// Copies a planned batch to the device and allocates its outputs.
    fn upload(&self, batch: &[u8], plan: &BatchPlan) -> Result<Uploaded> {
        let count = plan.out_caps.len();
        Ok(Uploaded {
            input: self
                .stream
                .clone_htod(batch)
                .map_err(driver_err("uploading compressed batch"))?,
            in_offsets: self
                .stream
                .clone_htod(&plan.in_offsets)
                .map_err(driver_err("uploading input offsets"))?,
            in_lengths: self
                .stream
                .clone_htod(&plan.in_lengths)
                .map_err(driver_err("uploading input lengths"))?,
            out_offsets: self
                .stream
                .clone_htod(&plan.out_offsets)
                .map_err(driver_err("uploading output offsets"))?,
            out_caps: self
                .stream
                .clone_htod(&plan.out_caps)
                .map_err(driver_err("uploading output capacities"))?,
            // `total` can legitimately be zero (a batch of only EOF blocks),
            // and a zero-sized allocation is not valid.
            output: self
                .stream
                .alloc_zeros::<u8>(plan.total.max(1))
                .map_err(driver_err("allocating output"))?,
            status_buf: self
                .stream
                .alloc_zeros::<u32>(count)
                .map_err(driver_err("allocating status"))?,
            produced_buf: self
                .stream
                .alloc_zeros::<u32>(count)
                .map_err(driver_err("allocating produced sizes"))?,
            crc_buf: self
                .stream
                .alloc_zeros::<u32>(count)
                .map_err(driver_err("allocating checksums"))?,
        })
    }

    /// Uploads a planned batch and runs the kernel.
    ///
    /// The inflated payload is **left on the device**; only the per-block
    /// status arrays are downloaded, because verification needs them and they
    /// are small. Whether the payload follows is the caller's choice, and it is
    /// the only difference between the host and device paths.
    fn launch_inflate(
        &self,
        batch: &[u8],
        plan: &BatchPlan,
        timings: &mut InflateTimings,
    ) -> Result<DeviceOutput> {
        let started = Instant::now();
        let total = plan.total;
        let count = plan.out_caps.len();
        let count_arg = i32::try_from(count)
            .map_err(|_| Error::Cuda(format!("batch of {count} blocks exceeds i32")))?;

        let Uploaded {
            input,
            in_offsets,
            in_lengths,
            out_offsets,
            out_caps,
            mut output,
            mut status_buf,
            mut produced_buf,
            mut crc_buf,
        } = self.upload(batch, plan)?;

        // Close out the upload phase before timing the kernel. The copies are
        // queued on the stream, so without this synchronise their cost would be
        // silently charged to whichever phase synchronises next.
        self.stream
            .synchronize()
            .map_err(driver_err("synchronising after upload"))?;
        timings.upload += started.elapsed();
        let started = Instant::now();

        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK_DIM as usize) as u32, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.inflate);
        builder
            .arg(&input)
            .arg(&in_offsets)
            .arg(&in_lengths)
            .arg(&mut output)
            .arg(&out_offsets)
            .arg(&out_caps)
            .arg(&mut status_buf)
            .arg(&mut produced_buf)
            .arg(&mut crc_buf)
            .arg(&count_arg);

        // SAFETY: the kernel signature matches the arguments pushed above; all
        // input spans were bounds-checked against `batch`, and the output
        // buffer is the prefix sum of the per-block capacities the kernel is
        // given, so no thread can write outside its own range.
        unsafe { builder.launch(cfg) }.map_err(driver_err("launching inflate_blocks"))?;

        // Recorded before any synchronise, so it marks kernel completion rather
        // than "the host got around to noticing". This is what a consumer on
        // another stream waits on.
        let ready = self
            .stream
            .record_event(None)
            .map_err(driver_err("recording completion event"))?;

        self.stream
            .synchronize()
            .map_err(driver_err("synchronising"))?;
        timings.kernel += started.elapsed();
        let started = Instant::now();

        // Verification data only: 12 bytes per block. The payload stays put.
        let status = self
            .stream
            .clone_dtoh(&status_buf)
            .map_err(driver_err("downloading status"))?;
        let produced = self
            .stream
            .clone_dtoh(&produced_buf)
            .map_err(driver_err("downloading produced sizes"))?;
        let crcs = self
            .stream
            .clone_dtoh(&crc_buf)
            .map_err(driver_err("downloading checksums"))?;
        timings.download += started.elapsed();

        Ok(DeviceOutput {
            data: output,
            ready,
            total,
            status,
            produced,
            crcs,
        })
    }

    /// Computes one CRC32 per block, in parallel across blocks.
    ///
    /// `offsets` and `lengths` describe where each block's payload sits within
    /// `data`; both must have the same length, and every
    /// `offset + length` must lie within `data`.
    pub fn crc32_blocks(&self, data: &[u8], offsets: &[u64], lengths: &[u32]) -> Result<Vec<u32>> {
        if offsets.len() != lengths.len() {
            return Err(Error::Cuda(format!(
                "offsets ({}) and lengths ({}) must have equal length",
                offsets.len(),
                lengths.len()
            )));
        }

        let count = offsets.len();
        if count == 0 {
            return Ok(Vec::new());
        }

        // Validate on the host: an out-of-range span would be an out-of-bounds
        // read on the device, which is far harder to diagnose.
        for (i, (&offset, &len)) in offsets.iter().zip(lengths).enumerate() {
            let end = offset
                .checked_add(u64::from(len))
                .ok_or_else(|| Error::Cuda(format!("block {i} span overflows")))?;
            if end > data.len() as u64 {
                return Err(Error::Cuda(format!(
                    "block {i} spans {offset}..{end}, past the {} byte buffer",
                    data.len()
                )));
            }
        }

        let d_data = self
            .stream
            .clone_htod(data)
            .map_err(driver_err("uploading payloads"))?;
        let d_offsets = self
            .stream
            .clone_htod(offsets)
            .map_err(driver_err("uploading offsets"))?;
        let d_lengths = self
            .stream
            .clone_htod(lengths)
            .map_err(driver_err("uploading lengths"))?;
        let mut output = self
            .stream
            .alloc_zeros::<u32>(count)
            .map_err(driver_err("allocating output"))?;

        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK_DIM as usize) as u32, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        // The kernel takes an `int`; refuse rather than wrap.
        let count_arg = i32::try_from(count)
            .map_err(|_| Error::Cuda(format!("batch of {count} blocks exceeds i32")))?;
        let mut builder = self.stream.launch_builder(&self.crc32);
        builder
            .arg(&d_data)
            .arg(&d_offsets)
            .arg(&d_lengths)
            .arg(&mut output)
            .arg(&count_arg);

        // SAFETY: the kernel signature matches the arguments pushed above, and
        // every span was bounds-checked against `data` before upload.
        unsafe { builder.launch(cfg) }.map_err(driver_err("launching crc32_blocks"))?;

        self.stream
            .synchronize()
            .map_err(driver_err("synchronising"))?;

        self.stream
            .clone_dtoh(&output)
            .map_err(driver_err("downloading results"))
    }
}
