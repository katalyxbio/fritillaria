//! NVIDIA nvCOMP as the GPU codec.
//!
//! # Why this exists
//!
//! Our own inflate kernel does ~2.2 GB/s. nvCOMP's batched DEFLATE is an order
//! of magnitude faster, and on Blackwell it dispatches to an on-die
//! Decompression Engine. Closing that gap ourselves is not a good use of
//! effort, and NVIDIA's own library is what an adopter expects to see. So
//! nvCOMP is the intended fast path and `kernels/inflate.cu` is the portable
//! fallback and a second differential-test oracle. See CLAUDE.md,
//! *Decisions made*.
//!
//! # How it maps onto BGZF
//!
//! nvCOMP's batched API takes an array of independent compressed buffers, which
//! is precisely the BGZF block model — that is why it drops in behind
//! [`BlockCodec`] with no change to the container layer.
//!
//! Three facts about the API shape the implementation. All three were measured
//! against nvCOMP 5.3.0.16 rather than assumed:
//!
//! | | value | consequence |
//! |---|---|---|
//! | output alignment | **1** | decompresses **straight into our dense buffer** |
//! | temp workspace | **0 bytes** | no scratch allocation |
//! | input alignment | 4 | BGZF payloads need restaging — see [`gather`] |
//!
//! The output alignment is the load-bearing one. `DeviceInflateBatch` requires
//! blocks to be densely packed, because a BAM record can span a block boundary
//! and a consumer must be able to read across it. Had nvCOMP demanded padded
//! output we would have needed a whole extra compaction pass over the inflated
//! data — the single most expensive thing in the pipeline. It does not.
//!
//! # Verification
//!
//! nvCOMP's deflate entry point never sees the gzip trailer, and its own
//! documentation says only limited validation is performed. So verification
//! stays ours: [`nvcompBatchedCRC32Async`](ffi::Nvcomp::crc32) computes a
//! checksum per block and it is compared against the CRC32 read out of each
//! block's trailer during host-side discovery. That is *stronger* than trusting
//! a self-check, because the comparison is against the value the file itself
//! records. Both engines then go through the same
//! [`verify_and_offsets`](crate::backend::verify_and_offsets).
//!
//! [`gather`]: https://docs.rs/fritillaria-cuda

pub mod ffi;

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    CudaContext as RawContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use fritillaria_core::{
    BlockCodec, BlockSpan, DeviceBlockCodec, DeviceBuffer, DeviceInflateBatch, Error, InflateBatch,
    MAX_BLOCK_SIZE, Result,
};

use crate::GATHER_KERNEL_SRC;
use crate::backend::{
    BatchPlan, CudaAlloc, DeviceOutput, InflateTimings, driver_err, load_kernel, verify_and_offsets,
};
use ffi::{AlignmentRequirements, Backend, DeflateDecompressOpts, Nvcomp, Status};

/// Threads per block for the restaging copy. One thread block handles one BGZF
/// block, so this is the width of the coalesced copy, not a work partition.
const GATHER_BLOCK_DIM: u32 = 256;

/// nvCOMP spells sizes `size_t`, cudarc wants a `DeviceRepr` type, and `u64` is
/// the one that is both. They are the same on every platform this builds for,
/// but assert it rather than assume — a mismatch would silently misread every
/// length in the batch.
const _: () = assert!(size_of::<usize>() == size_of::<u64>());

/// An nvCOMP-backed device context.
///
/// Construction loads the library and compiles the restaging kernel, so build
/// one and reuse it rather than one per batch.
#[derive(Debug)]
pub struct NvcompContext {
    ctx: Arc<RawContext>,
    stream: Arc<CudaStream>,
    lib: Nvcomp,
    gather: CudaFunction,
    opts: DeflateDecompressOpts,
    alignments: AlignmentRequirements,
    ordinal: i32,
}

/// The buffer nvCOMP will read compressed chunks out of.
struct Inputs {
    buffer: CudaSlice<u8>,
    /// Chunk offsets within `buffer`, each meeting nvCOMP's input alignment.
    offsets: Vec<u64>,
    /// The raw uploaded batch, when `buffer` is a restaged copy of it.
    ///
    /// Nothing reads it after the gather, but it is held until the whole launch
    /// is done: the gather is queued on the stream, so freeing it at the end of
    /// the staging call would be a use-after-free if the copy has not run yet.
    _raw: Option<CudaSlice<u8>>,
}

/// Everything one nvCOMP launch reads, writes, or reports through.
struct Uploaded {
    output: CudaSlice<u8>,
    in_ptrs: CudaSlice<u64>,
    out_ptrs: CudaSlice<u64>,
    chunk_bytes: CudaSlice<u64>,
    buffer_bytes: CudaSlice<u64>,
    produced: CudaSlice<u64>,
    status: CudaSlice<i32>,
    crcs: CudaSlice<u32>,
    crc_status: CudaSlice<i32>,
    temp: CudaSlice<u8>,
    temp_bytes: usize,
    /// The compressed input, held for the life of the launch: `in_ptrs` point
    /// into it, and nvCOMP reads it after this function has returned.
    _inputs: Inputs,
}

impl NvcompContext {
    /// Opens device `ordinal` on a context of our own and loads nvCOMP.
    ///
    /// Convenience for tests and benchmarks; an embedding caller should use
    /// [`with_context`](NvcompContext::with_context) so output lands in a
    /// context their kernels can address.
    pub fn new(ordinal: usize) -> Result<Self> {
        if !crate::backend::driver_is_available() {
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
    /// Same contract as [`CudaContext::with_context`](crate::CudaContext::with_context):
    /// output is allocated in `ctx` so a consumer's kernels can read it without
    /// a peer copy, and `stream` must belong to `ctx`.
    ///
    /// Returns [`Error::CudaUnavailable`] when nvCOMP is not installed. That is
    /// an ordinary outcome, not a failure — callers should fall back to
    /// [`CudaCodec`](crate::CudaCodec).
    pub fn with_context(ctx: Arc<RawContext>, stream: Arc<CudaStream>) -> Result<Self> {
        if stream.context() != &ctx {
            return Err(Error::Cuda(
                "stream does not belong to the given context".to_string(),
            ));
        }

        let lib = Nvcomp::load()?;
        let opts = DeflateDecompressOpts::new(Backend::Default);
        let alignments = lib.deflate_alignments(opts)?;

        if alignments.output > 1 {
            // Unreachable with 5.3, which reports 1. Refused rather than
            // worked around because the workaround is a full compaction pass
            // over the inflated data, and shipping that silently would make
            // this codec quietly slower than the one it replaces. See the
            // module docs.
            return Err(Error::Cuda(format!(
                "nvCOMP requires {}-byte output alignment; \
                 dense block packing needs 1 (see fritillaria_cuda::nvcomp docs)",
                alignments.output
            )));
        }

        let gather = load_kernel(&ctx, GATHER_KERNEL_SRC, "gather.cu", "gather_payloads")?;
        let ordinal = ctx.ordinal();
        let ordinal = i32::try_from(ordinal)
            .map_err(|_| Error::Cuda(format!("device ordinal {ordinal} exceeds i32")))?;

        Ok(Self {
            ctx,
            stream,
            lib,
            gather,
            opts,
            alignments,
            ordinal,
        })
    }

    /// The nvCOMP version that was loaded, as `major * 1000 + minor * 100`.
    #[must_use]
    pub fn nvcomp_version(&self) -> u32 {
        self.lib.version
    }

    /// Buffer alignments nvCOMP reported for deflate decompression.
    #[must_use]
    pub fn alignments(&self) -> AlignmentRequirements {
        self.alignments
    }

    /// The CUDA context this was built on.
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

    /// Uploads the batch, restaging payloads if nvCOMP's alignment needs it.
    ///
    /// The common case for a real BAM is that restaging *is* needed: a BGZF
    /// payload begins 18 bytes into its member and members sit at arbitrary
    /// offsets. The check is still worth making — it is a few thousand modulo
    /// operations against a copy of the whole compressed batch.
    fn stage_inputs(&self, batch: &[u8], plan: &BatchPlan) -> Result<Inputs> {
        let raw = self
            .stream
            .clone_htod(batch)
            .map_err(driver_err("uploading compressed batch"))?;

        let align = self.alignments.input.max(1) as u64;
        if plan.in_offsets.iter().all(|off| off % align == 0) {
            return Ok(Inputs {
                buffer: raw,
                offsets: plan.in_offsets.clone(),
                _raw: None,
            });
        }

        // Aligned prefix sum: every slot starts on a multiple of `align`.
        let mut staged = Vec::with_capacity(plan.in_lengths.len());
        let mut cursor = 0u64;
        for &len in &plan.in_lengths {
            staged.push(cursor);
            cursor += u64::from(len).next_multiple_of(align);
        }
        let staged_total = usize::try_from(cursor)
            .map_err(|_| Error::Cuda("staged batch size exceeds usize".to_string()))?;

        let mut staging = self
            .stream
            // `max(1)`: a batch of nothing but EOF blocks still has payloads,
            // but an all-empty one would ask for a zero-sized allocation.
            .alloc_zeros::<u8>(staged_total.max(1))
            .map_err(driver_err("allocating aligned staging buffer"))?;

        let src_offsets = self
            .stream
            .clone_htod(&plan.in_offsets)
            .map_err(driver_err("uploading payload offsets"))?;
        let lengths = self
            .stream
            .clone_htod(&plan.in_lengths)
            .map_err(driver_err("uploading payload lengths"))?;
        let dst_offsets = self
            .stream
            .clone_htod(&staged)
            .map_err(driver_err("uploading staged offsets"))?;

        let count = plan.in_lengths.len();
        let count_arg = i32::try_from(count)
            .map_err(|_| Error::Cuda(format!("batch of {count} blocks exceeds i32")))?;
        let cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (GATHER_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.gather);
        builder
            .arg(&raw)
            .arg(&src_offsets)
            .arg(&lengths)
            .arg(&mut staging)
            .arg(&dst_offsets)
            .arg(&count_arg);

        // SAFETY: the kernel signature matches the arguments pushed above.
        // `BatchPlan::new` bounds-checked every payload against `batch`, and
        // `staged` is a prefix sum of those same lengths rounded up, so both
        // the reads and the writes stay inside their buffers.
        unsafe { builder.launch(cfg) }.map_err(driver_err("launching gather_payloads"))?;

        Ok(Inputs {
            buffer: staging,
            offsets: staged,
            _raw: Some(raw),
        })
    }

    /// Copies a planned batch to the device and allocates everything nvCOMP
    /// needs to describe it.
    ///
    /// nvCOMP takes arrays of *pointers* rather than a base plus offsets, so
    /// this is where the plan's offsets are resolved against real device
    /// addresses.
    fn upload(&self, batch: &[u8], plan: &BatchPlan) -> Result<Uploaded> {
        let count = plan.out_caps.len();
        let inputs = self.stage_inputs(batch, plan)?;

        let in_ptrs: Vec<u64> = {
            let (base, _guard) = inputs.buffer.device_ptr(&self.stream);
            inputs.offsets.iter().map(|off| base + off).collect()
        };
        let chunk_bytes: Vec<u64> = plan.in_lengths.iter().map(|&n| u64::from(n)).collect();
        let buffer_bytes: Vec<u64> = plan.out_caps.iter().map(|&n| u64::from(n)).collect();

        let mut output = self
            .stream
            // A batch of only EOF blocks inflates to nothing, and a zero-sized
            // device allocation is not valid.
            .alloc_zeros::<u8>(plan.total.max(1))
            .map_err(driver_err("allocating output"))?;
        // Scoped so the guard is released before `output` is moved out. All of
        // this runs on one stream, so the ordering the guard enforces holds
        // anyway.
        let out_ptrs: Vec<u64> = {
            let (base, _guard) = output.device_ptr_mut(&self.stream);
            plan.out_offsets.iter().map(|off| base + off).collect()
        };

        // Measured at zero for deflate in 5.3, but asking costs nothing and
        // assuming would be an out-of-bounds write if that ever changes.
        let temp_bytes =
            self.lib
                .deflate_temp_size(count, MAX_BLOCK_SIZE, plan.total, self.opts)?;

        Ok(Uploaded {
            in_ptrs: self
                .stream
                .clone_htod(&in_ptrs)
                .map_err(driver_err("uploading input chunk pointers"))?,
            out_ptrs: self
                .stream
                .clone_htod(&out_ptrs)
                .map_err(driver_err("uploading output chunk pointers"))?,
            chunk_bytes: self
                .stream
                .clone_htod(&chunk_bytes)
                .map_err(driver_err("uploading compressed chunk sizes"))?,
            buffer_bytes: self
                .stream
                .clone_htod(&buffer_bytes)
                .map_err(driver_err("uploading output buffer sizes"))?,
            produced: self
                .stream
                .alloc_zeros::<u64>(count)
                .map_err(driver_err("allocating produced sizes"))?,
            status: self
                .stream
                .alloc_zeros::<i32>(count)
                .map_err(driver_err("allocating decompression statuses"))?,
            crcs: self
                .stream
                .alloc_zeros::<u32>(count)
                .map_err(driver_err("allocating checksums"))?,
            crc_status: self
                .stream
                .alloc_zeros::<i32>(count)
                .map_err(driver_err("allocating checksum statuses"))?,
            temp: self
                .stream
                .alloc_zeros::<u8>(temp_bytes.max(1))
                .map_err(driver_err("allocating nvcomp workspace"))?,
            temp_bytes,
            output,
            _inputs: inputs,
        })
    }

    /// Runs one batch through nvCOMP, leaving the payload on the device.
    ///
    /// Mirrors `CudaContext::launch_inflate`: the inflated bytes stay in device
    /// memory and only the per-block verification arrays come back.
    fn launch(
        &self,
        batch: &[u8],
        plan: &BatchPlan,
        timings: &mut InflateTimings,
    ) -> Result<DeviceOutput> {
        let started = Instant::now();
        let count = plan.out_caps.len();
        let mut up = self.upload(batch, plan)?;

        self.stream
            .synchronize()
            .map_err(driver_err("synchronising after upload"))?;
        timings.upload += started.elapsed();
        let started = Instant::now();

        self.decompress(count, &mut up)?;
        self.checksum(count, &mut up)?;

        let ready = self
            .stream
            .record_event(None)
            .map_err(driver_err("recording completion event"))?;
        self.stream
            .synchronize()
            .map_err(driver_err("synchronising after decompression"))?;
        timings.kernel += started.elapsed();
        let started = Instant::now();

        let produced = self
            .stream
            .clone_dtoh(&up.produced)
            .map_err(driver_err("downloading produced sizes"))?;
        let status = self
            .stream
            .clone_dtoh(&up.status)
            .map_err(driver_err("downloading decompression statuses"))?;
        let crcs = self
            .stream
            .clone_dtoh(&up.crcs)
            .map_err(driver_err("downloading checksums"))?;
        let crc_status = self
            .stream
            .clone_dtoh(&up.crc_status)
            .map_err(driver_err("downloading checksum statuses"))?;
        timings.download += started.elapsed();

        // A failed checksum *computation* is not a corrupt block; it means one
        // went unverified, which under this crate's contract is just as
        // unacceptable. Fail rather than accept an unchecked block.
        if let Some((i, &code)) = crc_status.iter().enumerate().find(|&(_, &c)| c != 0) {
            return Err(Error::Cuda(format!(
                "nvcomp could not checksum block {i}: {}",
                self.lib.describe(code)
            )));
        }

        Ok(DeviceOutput {
            data: up.output,
            ready,
            total: plan.total,
            status: status.into_iter().map(i32::cast_unsigned).collect(),
            // Saturating: a nonsense length from a failed chunk should surface
            // as a size mismatch naming the block, not a panic.
            produced: produced
                .into_iter()
                .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
                .collect(),
            crcs,
        })
    }

    /// The nvCOMP decompression call, with the pointer plumbing isolated.
    fn decompress(&self, count: usize, up: &mut Uploaded) -> Result<()> {
        let temp_bytes = up.temp_bytes;
        let (in_ptrs, _g0) = up.in_ptrs.device_ptr(&self.stream);
        let (chunk_bytes, _g1) = up.chunk_bytes.device_ptr(&self.stream);
        let (buffer_bytes, _g2) = up.buffer_bytes.device_ptr(&self.stream);
        let (out_ptrs, _g3) = up.out_ptrs.device_ptr(&self.stream);
        let (produced, _g4) = up.produced.device_ptr_mut(&self.stream);
        let (temp, _g5) = up.temp.device_ptr_mut(&self.stream);
        let (status, _g6) = up.status.device_ptr_mut(&self.stream);

        // SAFETY: every array was allocated with `count` elements just above.
        // Input chunks meet nvCOMP's alignment (`stage_inputs`), each output
        // chunk has exactly the `buffer_bytes` room reserved for it by the
        // ISIZE prefix sum, and the buffers all outlive the stream work because
        // the `_g*` guards keep them alive until this call returns.
        unsafe {
            self.lib.deflate_decompress(
                in_ptrs as *const *const c_void,
                chunk_bytes as *const usize,
                buffer_bytes as *const usize,
                produced as *mut usize,
                count,
                temp as *mut c_void,
                temp_bytes,
                out_ptrs as *const *mut c_void,
                self.opts,
                status as *mut Status,
                self.stream.cu_stream().cast(),
            )
        }
    }

    /// One CRC32 per block over the inflated output.
    ///
    /// Checksums the *declared* `ISIZE` extent rather than what nvCOMP reports
    /// it produced. Two reasons, and the second is the important one: the
    /// declared extent is exactly the space allocated, so the read cannot run
    /// past the buffer even if a chunk failed; and a block that produced the
    /// wrong length is caught by the size check in `verify_and_offsets` before
    /// its checksum is ever consulted.
    fn checksum(&self, count: usize, up: &mut Uploaded) -> Result<()> {
        // SAFETY: the stream belongs to this context and outlives the call.
        let conf = unsafe {
            self.lib
                .crc32_heuristic_conf(count, MAX_BLOCK_SIZE, self.stream.cu_stream().cast())
        }?;
        let opts = ffi::Crc32Opts::gzip(conf);

        let (out_ptrs, _g0) = up.out_ptrs.device_ptr(&self.stream);
        let (buffer_bytes, _g1) = up.buffer_bytes.device_ptr(&self.stream);
        let (crcs, _g2) = up.crcs.device_ptr_mut(&self.stream);
        let (status, _g3) = up.crc_status.device_ptr_mut(&self.stream);

        // SAFETY: all four arrays hold `count` elements, and the chunk pointers
        // and sizes are the same pair just handed to decompression, so they
        // describe memory we own and have already sized.
        unsafe {
            self.lib.crc32(
                out_ptrs as *const *const c_void,
                buffer_bytes as *const usize,
                count,
                crcs as *mut u32,
                opts,
                status as *mut Status,
                self.stream.cu_stream().cast(),
            )
        }
    }

    /// Inflates a batch, copying the result back to the host.
    pub fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()> {
        let mut timings = InflateTimings::default();
        self.inflate_batch_timed(batch, spans, out, &mut timings)
    }

    /// [`NvcompContext::inflate_batch`], recording where the time went.
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

        let plan = Self::plan(batch, spans, timings)?;
        let device = self.launch(batch, &plan, timings)?;

        let started = Instant::now();
        let mut data = self
            .stream
            .clone_dtoh(&device.data)
            .map_err(driver_err("downloading output"))?;
        data.truncate(device.total);
        timings.download += started.elapsed();

        let started = Instant::now();
        let offsets = verify_and_offsets(spans, &device, &|code| {
            self.lib.describe(code.cast_signed())
        })?;
        *out = InflateBatch::from_parts(data, offsets)
            .ok_or_else(|| Error::Cuda("inconsistent output layout".to_string()))?;
        timings.verify += started.elapsed();
        Ok(())
    }

    /// Inflates a batch, leaving the output in device memory.
    pub fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        let mut timings = InflateTimings::default();
        self.inflate_batch_device_timed(batch, spans, out, &mut timings)
    }

    /// [`NvcompContext::inflate_batch_device`], recording where the time went.
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

        let plan = Self::plan(batch, spans, timings)?;
        let device = self.launch(batch, &plan, timings)?;

        let started = Instant::now();
        let offsets = verify_and_offsets(spans, &device, &|code| {
            self.lib.describe(code.cast_signed())
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

    fn plan(batch: &[u8], spans: &[BlockSpan], timings: &mut InflateTimings) -> Result<BatchPlan> {
        let started = Instant::now();
        let plan = BatchPlan::new(batch, spans)?;
        timings.plan += started.elapsed();
        timings.blocks += spans.len() as u64;
        timings.compressed_bytes += spans.iter().map(|s| s.payload_len as u64).sum::<u64>();
        timings.inflated_bytes += plan.total as u64;
        Ok(plan)
    }
}

/// The nvCOMP implementation of [`BlockCodec`] and [`DeviceBlockCodec`].
///
/// Interchangeable with [`CudaCodec`](crate::CudaCodec) — same traits, same
/// mandatory verification, same device-resident output. Which one a caller gets
/// is a performance decision, not a semantic one.
#[derive(Debug)]
pub struct NvcompCodec {
    ctx: NvcompContext,
}

impl NvcompCodec {
    /// Opens the default device and loads nvCOMP.
    ///
    /// Returns [`Error::CudaUnavailable`] when there is no device or nvCOMP is
    /// not installed. Both are ordinary outcomes; fall back to
    /// [`CudaCodec`](crate::CudaCodec).
    pub fn new() -> Result<Self> {
        Ok(Self {
            ctx: NvcompContext::new(0)?,
        })
    }

    /// Builds a codec on a CUDA context and stream the **caller** owns.
    pub fn with_context(ctx: Arc<RawContext>, stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            ctx: NvcompContext::with_context(ctx, stream)?,
        })
    }

    /// The underlying context, for version and alignment reporting.
    #[must_use]
    pub fn context(&self) -> &NvcompContext {
        &self.ctx
    }

    /// The event marking completion of the inflate that produced `batch`.
    ///
    /// Same contract as [`CudaCodec::ready_event`](crate::CudaCodec::ready_event):
    /// a consumer on its own stream orders against this rather than
    /// synchronising the host.
    #[must_use]
    pub fn ready_event(batch: &DeviceInflateBatch) -> Option<&cudarc::driver::CudaEvent> {
        Some(
            batch
                .data()?
                .alloc()
                .as_any()
                .downcast_ref::<CudaAlloc>()?
                .ready_event(),
        )
    }

    /// [`inflate_batch`](BlockCodec::inflate_batch), recording where time went.
    pub fn inflate_batch_timed(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
        timings: &mut InflateTimings,
    ) -> Result<()> {
        self.ctx.inflate_batch_timed(batch, spans, out, timings)
    }

    /// [`inflate_batch_device`](DeviceBlockCodec::inflate_batch_device),
    /// recording where time went.
    pub fn inflate_batch_device_timed(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
        timings: &mut InflateTimings,
    ) -> Result<()> {
        self.ctx
            .inflate_batch_device_timed(batch, spans, out, timings)
    }
}

impl BlockCodec for NvcompCodec {
    fn name(&self) -> &'static str {
        "nvcomp"
    }

    fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()> {
        self.ctx.inflate_batch(batch, spans, out)
    }
}

impl DeviceBlockCodec for NvcompCodec {
    fn name(&self) -> &'static str {
        "nvcomp-device"
    }

    fn device_ordinal(&self) -> i32 {
        self.ctx.device_ordinal()
    }

    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        self.ctx.inflate_batch_device(batch, spans, out)
    }
}
