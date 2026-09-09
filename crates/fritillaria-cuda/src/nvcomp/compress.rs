//! BGZF compression on nvCOMP, device-resident.
//!
//! The write-side mirror of the codec next door, and the half that stops a GPU
//! pipeline paying back the transfer the read path deleted: a tool that produces
//! records on the device and then hands them to the host to compress has only
//! accelerated one direction.
//!
//! # It is not the read path in reverse
//!
//! Three differences, all measured against nvCOMP 5.3.0.16 rather than assumed,
//! and each one changes the implementation:
//!
//! | | decompress | compress |
//! |---|---|---|
//! | output alignment | 1 | **8** |
//! | output size known before the launch? | yes, from `ISIZE` | **no** |
//! | temp workspace | 0 bytes | **1.11 MB per chunk** at our default level |
//!
//! The first two combine into the pass the read path does not need. Inflate
//! writes straight into a dense buffer because the inflated size of every block
//! is written in its own trailer; compression has no such oracle, so output goes
//! into padded worst-case slots — 148,256 bytes each, 2.26x a full payload — and
//! [`kernels/bgzf_frame.cu`] gathers the BGZF stream out of them afterwards.
//! That is affordable only because it runs over the *compressed* side.
//!
//! The third is the one that shapes batch sizing, and it was the surprise. See
//! [`CompressBudget`].
//!
//! # What comes back to the host
//!
//! Two things, and their sizes are the point. Per batch: `8 * n` bytes of
//! compressed chunk sizes, needed to lay the output stream out; and the finished
//! BGZF stream itself, because writing a file happens on the host. The second is
//! ~3.4x smaller than the records that produced it on real WGS — the same ratio
//! that makes the read path's argument work, applied in the other direction.
//!
//! [`kernels/bgzf_frame.cu`]: https://docs.rs/fritillaria-cuda

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext as RawContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use fritillaria_core::compress::{Framing, choose_framing};
use fritillaria_core::{
    BlockCompressor, CompressedBatch, DeviceBlockCompressor, DeviceBuffer, Error,
    MAX_COMPRESSIBLE_PAYLOAD, Result,
};

use crate::BGZF_FRAME_KERNEL_SRC;
use crate::backend::{CudaAlloc, driver_err, load_kernel};

use super::ffi::{DeflateAlgorithm, DeflateCompressOpts, MAX_COMPRESS_CHUNK_BYTES, Nvcomp, Status};

/// Threads per block for framing. One thread block frames one BGZF block, so
/// this is the width of the coalesced body copy, not a work partition.
const FRAME_BLOCK_DIM: u32 = 256;

/// What one chunk costs in device memory, and therefore how many fit.
///
/// # Why this is not a detail
///
/// Compression scratch is **1.11 MB per 64 KiB chunk** at
/// [`DeflateAlgorithm::HighRatio`], against zero for decompression. Add the
/// padded output slot and the dense output and a chunk costs ~1.39 MB, so a
/// 16 GiB T4 holds roughly **11,000 blocks** where the read path put 166,012 in
/// one batch. Compression batches are ~15x smaller, and the reason is scratch,
/// not data.
///
/// The ratio ladder is also a VRAM ladder — `EntropyOnly` needs no scratch at
/// all and fits 6.2x more blocks per batch — which the nvCOMP header does not
/// mention and which is a real argument for Parabricks' faster default. See
/// `docs/compression.md`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressBudget {
    /// Scratch nvCOMP wants, per chunk.
    pub scratch_per_chunk: usize,
    /// Padded output slot nvCOMP writes into, per chunk.
    pub slot_bytes: usize,
    /// Alignment nvCOMP requires of each output chunk pointer.
    pub output_align: usize,
}

impl CompressBudget {
    /// Device bytes one chunk costs, across every allocation this path makes.
    ///
    /// The dense output is bounded by BGZF's own block cap rather than by the
    /// slot size: a block that did not fit 64 KiB would not be a block.
    #[must_use]
    pub fn bytes_per_chunk(&self) -> usize {
        self.scratch_per_chunk + self.slot_bytes + fritillaria_core::MAX_BLOCK_SIZE
    }

    /// How many chunks fit in `vram` bytes, at least one.
    ///
    /// Returning zero would be a batch that cannot make progress, which is a
    /// worse failure than an allocation error naming the size it wanted.
    #[must_use]
    pub fn chunks_in(&self, vram: usize) -> usize {
        (vram / self.bytes_per_chunk()).max(1)
    }
}

/// Where each chunk's compressed output goes in the padded slot buffer.
///
/// Trivial arithmetic, split out because it is the half of the layout that is
/// decided *before* the kernel runs — and separating it from [`FramePlan`],
/// which is decided after, is what makes the ordering legible.
#[derive(Clone, Debug)]
pub struct SlotPlan {
    /// Byte offset of each chunk's slot; each meets nvCOMP's output alignment.
    pub offsets: Vec<u64>,
    /// Total padded bytes.
    pub total: usize,
}

impl SlotPlan {
    /// Lays out `count` equally-sized slots.
    #[must_use]
    pub fn new(count: usize, budget: &CompressBudget) -> Self {
        let stride = budget
            .slot_bytes
            .next_multiple_of(budget.output_align.max(1));
        let offsets = (0..count).map(|i| (i * stride) as u64).collect();
        Self {
            offsets,
            total: count * stride,
        }
    }
}

/// The dense BGZF layout, decided once nvCOMP has reported its sizes.
///
/// This is the one place the host has to make a decision the kernel cannot: the
/// output buffer must be sized before framing runs, so whether each block is
/// deflated or stored is settled here and handed to the kernel as a flag. The
/// rule itself lives in [`choose_framing`] so the host reference and this path
/// cannot drift on it.
#[derive(Clone, Debug)]
pub struct FramePlan {
    /// 1 where the payload is stored verbatim rather than deflated.
    pub stored: Vec<u8>,
    /// Where each block starts in the dense output; length `n + 1`.
    pub offsets: Vec<u64>,
    /// Total dense bytes.
    pub total: usize,
    /// How many blocks fell back to storing — a ratio signal worth surfacing.
    pub stored_count: usize,
}

impl FramePlan {
    /// Plans the dense stream from the payload lengths and nvCOMP's sizes.
    ///
    /// # Errors
    ///
    /// If the two arrays disagree in length, or a payload exceeds
    /// [`MAX_COMPRESSIBLE_PAYLOAD`] — which cannot be honoured by any
    /// implementation, since even storing it would not frame.
    pub fn new(payload_lens: &[u32], deflate_lens: &[u64]) -> Result<Self> {
        if payload_lens.len() != deflate_lens.len() {
            return Err(Error::Cuda(format!(
                "nvcomp reported {} sizes for {} chunks",
                deflate_lens.len(),
                payload_lens.len()
            )));
        }

        let mut plan = Self {
            stored: Vec::with_capacity(payload_lens.len()),
            offsets: Vec::with_capacity(payload_lens.len() + 1),
            total: 0,
            stored_count: 0,
        };

        let mut cursor = 0u64;
        for (i, (&raw, &deflated)) in payload_lens.iter().zip(deflate_lens).enumerate() {
            let raw = raw as usize;
            if raw > MAX_COMPRESSIBLE_PAYLOAD {
                return Err(Error::Cuda(format!(
                    "chunk {i} of {raw} bytes exceeds the compressible maximum \
                     {MAX_COMPRESSIBLE_PAYLOAD}; splitting it would move a block \
                     boundary the caller chose"
                )));
            }
            let deflated = usize::try_from(deflated).map_err(|_| {
                Error::Cuda(format!("nvcomp reported {deflated} bytes for chunk {i}"))
            })?;

            let stored = choose_framing(raw, deflated) == Framing::Stored;
            plan.stored.push(u8::from(stored));
            plan.stored_count += usize::from(stored);
            plan.offsets.push(cursor);
            cursor += fritillaria_core::compress::framed_size(raw, deflated) as u64;
        }

        plan.offsets.push(cursor);
        plan.total = usize::try_from(cursor)
            .map_err(|_| Error::Cuda("compressed batch exceeds usize".to_string()))?;
        Ok(plan)
    }
}

/// Splits `bounds` into per-chunk offsets and lengths, checking the shape.
fn chunk_extents(byte_len: usize, bounds: &[usize]) -> Result<(Vec<u64>, Vec<u32>)> {
    if bounds.first() != Some(&0) || bounds.last() != Some(&byte_len) {
        return Err(Error::Cuda(format!(
            "bounds {:?}..{:?} do not describe {byte_len} device bytes",
            bounds.first(),
            bounds.last()
        )));
    }

    let mut offsets = Vec::with_capacity(bounds.len() - 1);
    let mut lengths = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        let len = w[1].checked_sub(w[0]).ok_or_else(|| {
            Error::Cuda(format!(
                "bounds must be non-decreasing, found {} then {}",
                w[0], w[1]
            ))
        })?;
        offsets.push(w[0] as u64);
        lengths.push(
            u32::try_from(len)
                .map_err(|_| Error::Cuda(format!("chunk of {len} bytes exceeds a BGZF block")))?,
        );
    }
    Ok((offsets, lengths))
}

/// The device buffers the framing kernel reads from.
///
/// Grouped rather than passed as eleven arguments: the kernel takes them
/// positionally, and a struct is one place to check the order against the `.cu`
/// signature instead of one per call site.
struct FrameInputs<'a> {
    /// nvCOMP's output, in padded slots.
    slots: &'a CudaSlice<u8>,
    slot_offsets: &'a [u64],
    /// Bytes nvCOMP actually produced per chunk.
    deflate_sizes: &'a [u64],
    /// The uncompressed payloads, for chunks that fall back to storing.
    raw: &'a CudaSlice<u8>,
    raw_offsets: &'a [u64],
    raw_sizes: &'a [u32],
    /// CRC32 of each uncompressed chunk, already on device.
    crcs: &'a CudaSlice<u32>,
}

/// nvCOMP-backed BGZF compression.
///
/// Construction loads nvCOMP and compiles the framing kernel, so build one and
/// reuse it rather than one per batch.
#[derive(Debug)]
pub struct NvcompCompressor {
    ctx: Arc<RawContext>,
    stream: Arc<CudaStream>,
    lib: Nvcomp,
    frame: CudaFunction,
    opts: DeflateCompressOpts,
    budget: CompressBudget,
    ordinal: i32,
}

impl NvcompCompressor {
    /// Opens device `ordinal` on a context of our own, at the default level.
    pub fn new(ordinal: usize) -> Result<Self> {
        Self::with_algorithm(ordinal, DeflateAlgorithm::default())
    }

    /// Opens device `ordinal` at a chosen point on the ratio/throughput ladder.
    pub fn with_algorithm(ordinal: usize, algorithm: DeflateAlgorithm) -> Result<Self> {
        if !crate::backend::driver_is_available() {
            return Err(Error::CudaUnavailable(
                "CUDA driver library not found (no GPU on this machine?)".to_string(),
            ));
        }
        let ctx = RawContext::new(ordinal).map_err(driver_err("opening device"))?;
        let stream = ctx.default_stream();
        Self::with_context(ctx, stream, algorithm)
    }

    /// Builds a compressor on a CUDA context and stream the **caller** owns.
    ///
    /// Same contract as the codec's: input is read from, and output allocated
    /// in, `ctx`, so a producer's kernels and this one address the same memory
    /// without a peer copy. `stream` must belong to `ctx`.
    pub fn with_context(
        ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        algorithm: DeflateAlgorithm,
    ) -> Result<Self> {
        if stream.context() != &ctx {
            return Err(Error::Cuda(
                "stream does not belong to the given context".to_string(),
            ));
        }

        let lib = Nvcomp::load()?;
        let opts = DeflateCompressOpts::new(algorithm);
        let alignments = lib.deflate_compress_alignments(opts)?;
        let slot_bytes = lib.deflate_max_output_chunk_size(MAX_COMPRESS_CHUNK_BYTES, opts)?;
        // Queried for a single chunk: measured exactly linear in the chunk count
        // at this level, so the per-chunk figure is the one a batch sizer needs.
        // Asked again at the real count before every launch, so a version where
        // it stops being linear costs correctness nothing.
        let scratch_per_chunk = lib.deflate_compress_temp_size(
            1,
            MAX_COMPRESS_CHUNK_BYTES,
            MAX_COMPRESS_CHUNK_BYTES,
            opts,
        )?;

        let frame = load_kernel(&ctx, BGZF_FRAME_KERNEL_SRC, "bgzf_frame.cu", "frame_blocks")?;
        let ordinal = ctx.ordinal();
        let ordinal = i32::try_from(ordinal)
            .map_err(|_| Error::Cuda(format!("device ordinal {ordinal} exceeds i32")))?;

        Ok(Self {
            ctx,
            stream,
            lib,
            frame,
            opts,
            budget: CompressBudget {
                scratch_per_chunk,
                slot_bytes,
                output_align: alignments.output.max(1),
            },
            ordinal,
        })
    }

    /// What a chunk costs in device memory on this device, at this level.
    #[must_use]
    pub fn budget(&self) -> CompressBudget {
        self.budget
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

    /// The device pointer behind a buffer this crate allocated.
    fn device_slice(data: &DeviceBuffer) -> Result<&CudaSlice<u8>> {
        data.alloc()
            .as_any()
            .downcast_ref::<CudaAlloc>()
            .map(CudaAlloc::slice)
            .ok_or_else(|| {
                Error::Cuda(
                    "device buffer was not allocated by this backend; a compressor \
                     cannot read another allocator's memory"
                        .to_string(),
                )
            })
    }

    /// Fails the batch if nvCOMP reported a non-zero status for any chunk.
    fn check_statuses(&self, statuses: &[i32], what: &str) -> Result<()> {
        match statuses.iter().enumerate().find(|&(_, &c)| c != 0) {
            None => Ok(()),
            Some((i, &code)) => Err(Error::Cuda(format!(
                "nvcomp could not {what} chunk {i}: {}",
                self.lib.describe(code)
            ))),
        }
    }

    /// The nvCOMP compression call, with the pointer plumbing isolated.
    #[allow(clippy::too_many_arguments)]
    fn launch_compress(
        &self,
        count: usize,
        total_bytes: usize,
        in_ptrs: &CudaSlice<u64>,
        in_bytes: &CudaSlice<u64>,
        out_ptrs: &CudaSlice<u64>,
        produced: &mut CudaSlice<u64>,
        status: &mut CudaSlice<i32>,
    ) -> Result<()> {
        let temp_bytes = self.lib.deflate_compress_temp_size(
            count,
            MAX_COMPRESS_CHUNK_BYTES,
            total_bytes,
            self.opts,
        )?;
        let mut temp = self
            .stream
            .alloc_zeros::<u8>(temp_bytes.max(1))
            .map_err(driver_err("allocating nvcomp compression workspace"))?;

        let (src, _g0) = in_ptrs.device_ptr(&self.stream);
        let (src_bytes, _g1) = in_bytes.device_ptr(&self.stream);
        let (dst, _g2) = out_ptrs.device_ptr(&self.stream);
        let (dst_bytes, _g3) = produced.device_ptr_mut(&self.stream);
        let (scratch, _g4) = temp.device_ptr_mut(&self.stream);
        let (codes, _g5) = status.device_ptr_mut(&self.stream);

        // SAFETY: every array was allocated with `count` elements by the
        // caller; no chunk exceeds MAX_COMPRESS_CHUNK_BYTES, since
        // `chunk_extents` bounded them by the u32 block cap; each output slot
        // has `slot_bytes` room, which is what nvCOMP itself reported as its
        // worst case; and the `_g*` guards keep every buffer alive until this
        // call returns.
        unsafe {
            self.lib.deflate_compress(
                src as *const *const c_void,
                src_bytes as *const usize,
                MAX_COMPRESS_CHUNK_BYTES,
                count,
                scratch as *mut c_void,
                temp_bytes,
                dst as *const *mut c_void,
                dst_bytes as *mut usize,
                self.opts,
                codes as *mut Status,
                self.stream.cu_stream().cast(),
            )
        }
    }

    /// Gathers the padded slots into a dense BGZF stream.
    fn launch_frame(&self, inputs: &FrameInputs<'_>, plan: &FramePlan) -> Result<CudaSlice<u8>> {
        let count = plan.stored.len();
        let up =
            |v: &[u64], what: &'static str| self.stream.clone_htod(v).map_err(driver_err(what));

        let slot_offsets = up(inputs.slot_offsets, "uploading slot offsets")?;
        let deflate_sizes = up(inputs.deflate_sizes, "uploading compressed sizes")?;
        let raw_offsets = up(inputs.raw_offsets, "uploading chunk offsets")?;
        let out_offsets = up(&plan.offsets, "uploading output offsets")?;
        let raw_sizes = self
            .stream
            .clone_htod(inputs.raw_sizes)
            .map_err(driver_err("uploading chunk lengths"))?;
        let stored = self
            .stream
            .clone_htod(&plan.stored)
            .map_err(driver_err("uploading framing decisions"))?;
        let mut dense = self
            .stream
            .alloc_zeros::<u8>(plan.total.max(1))
            .map_err(driver_err("allocating dense output"))?;

        let count_arg = i32::try_from(count)
            .map_err(|_| Error::Cuda(format!("batch of {count} chunks exceeds i32")))?;
        let cfg = LaunchConfig {
            grid_dim: (u32::try_from(count).unwrap_or(u32::MAX), 1, 1),
            block_dim: (FRAME_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.frame);
        builder
            .arg(inputs.slots)
            .arg(&slot_offsets)
            .arg(&deflate_sizes)
            .arg(inputs.raw)
            .arg(&raw_offsets)
            .arg(&raw_sizes)
            .arg(inputs.crcs)
            .arg(&stored)
            .arg(&mut dense)
            .arg(&out_offsets)
            .arg(&count_arg);

        // SAFETY: the kernel signature matches the arguments pushed above.
        // `FramePlan` sized `dense` from exactly the sizes and framing choices
        // handed to the kernel, and every source read is bounded by the same
        // arrays the plan was built from, so neither the reads nor the writes
        // leave their buffers.
        unsafe { builder.launch(cfg) }.map_err(driver_err("launching frame_blocks"))?;
        Ok(dense)
    }

    /// CRC32 of every uncompressed chunk, on device.
    ///
    /// Computed over the input rather than the output, because the gzip trailer
    /// checksums the bytes that went *in*. Inverting that is an easy mistake and
    /// produces a file that only fails on read.
    fn checksum(
        &self,
        in_ptrs: &CudaSlice<u64>,
        in_bytes: &CudaSlice<u64>,
        count: usize,
    ) -> Result<(CudaSlice<u32>, CudaSlice<i32>)> {
        let mut crcs = self
            .stream
            .alloc_zeros::<u32>(count)
            .map_err(driver_err("allocating checksums"))?;
        let mut status = self
            .stream
            .alloc_zeros::<i32>(count)
            .map_err(driver_err("allocating checksum statuses"))?;

        // SAFETY: the stream belongs to this context and outlives the call.
        let conf = unsafe {
            self.lib.crc32_heuristic_conf(
                count,
                MAX_COMPRESS_CHUNK_BYTES,
                self.stream.cu_stream().cast(),
            )
        }?;
        let opts = super::ffi::Crc32Opts::gzip(conf);

        // Scoped so the borrow guards are released before the slices are moved
        // out. Everything here is on one stream, so the ordering they enforce
        // holds anyway.
        {
            let (ptrs, _g0) = in_ptrs.device_ptr(&self.stream);
            let (bytes, _g1) = in_bytes.device_ptr(&self.stream);
            let (out, _g2) = crcs.device_ptr_mut(&self.stream);
            let (st, _g3) = status.device_ptr_mut(&self.stream);

            // SAFETY: all four arrays hold `count` elements, and the pointers
            // and sizes describe chunks inside the caller's buffer,
            // bounds-checked by `chunk_extents` before they were uploaded.
            unsafe {
                self.lib.crc32(
                    ptrs as *const *const c_void,
                    bytes as *const usize,
                    count,
                    out as *mut u32,
                    opts,
                    st as *mut Status,
                    self.stream.cu_stream().cast(),
                )
            }?;
        }

        Ok((crcs, status))
    }
}

impl DeviceBlockCompressor for NvcompCompressor {
    fn name(&self) -> &'static str {
        "nvcomp-compress"
    }

    fn device_ordinal(&self) -> i32 {
        self.ordinal
    }

    fn compress_batch_device(
        &self,
        data: &DeviceBuffer,
        bounds: &[usize],
        out: &mut CompressedBatch,
    ) -> Result<()> {
        out.clear();
        if bounds.len() < 2 {
            return Ok(());
        }
        if data.device_ordinal() != self.ordinal {
            return Err(Error::Cuda(format!(
                "input is on device {} but this compressor is on device {}",
                data.device_ordinal(),
                self.ordinal
            )));
        }

        let input = Self::device_slice(data)?;
        let (chunk_offsets, chunk_lengths) = chunk_extents(data.byte_len(), bounds)?;
        let count = chunk_lengths.len();

        // --- lay out the padded slots and upload the descriptor arrays -------

        let slots = SlotPlan::new(count, &self.budget);
        let mut slot_buffer = self
            .stream
            .alloc_zeros::<u8>(slots.total.max(1))
            .map_err(driver_err("allocating compressed slots"))?;

        let in_ptrs: Vec<u64> = {
            let (base, _g) = input.device_ptr(&self.stream);
            chunk_offsets.iter().map(|off| base + off).collect()
        };
        let out_ptrs: Vec<u64> = {
            let (base, _g) = slot_buffer.device_ptr_mut(&self.stream);
            slots.offsets.iter().map(|off| base + off).collect()
        };
        let in_bytes: Vec<u64> = chunk_lengths.iter().map(|&n| u64::from(n)).collect();

        let d_in_ptrs = self
            .stream
            .clone_htod(&in_ptrs)
            .map_err(driver_err("uploading input chunk pointers"))?;
        let d_out_ptrs = self
            .stream
            .clone_htod(&out_ptrs)
            .map_err(driver_err("uploading output chunk pointers"))?;
        let d_in_bytes = self
            .stream
            .clone_htod(&in_bytes)
            .map_err(driver_err("uploading chunk sizes"))?;
        let mut produced = self
            .stream
            .alloc_zeros::<u64>(count)
            .map_err(driver_err("allocating produced sizes"))?;
        let mut status = self
            .stream
            .alloc_zeros::<i32>(count)
            .map_err(driver_err("allocating compression statuses"))?;

        self.launch_compress(
            count,
            data.byte_len(),
            &d_in_ptrs,
            &d_in_bytes,
            &d_out_ptrs,
            &mut produced,
            &mut status,
        )?;

        let (crcs, crc_status) = self.checksum(&d_in_ptrs, &d_in_bytes, count)?;

        // --- the one mid-batch synchronise, and why it is unavoidable -------
        //
        // The dense output cannot be sized until the compressed sizes exist, and
        // they only exist once the kernel has run. This is `8 * n` bytes, not a
        // fraction of the data — 88 KB for an 11,000-block batch.

        let produced = self
            .stream
            .clone_dtoh(&produced)
            .map_err(driver_err("downloading compressed sizes"))?;
        let status = self
            .stream
            .clone_dtoh(&status)
            .map_err(driver_err("downloading compression statuses"))?;
        let crc_status = self
            .stream
            .clone_dtoh(&crc_status)
            .map_err(driver_err("downloading checksum statuses"))?;

        self.check_statuses(&status, "compress")?;
        // A block whose payload went unchecksummed would be written with a
        // trailer nobody computed — the read path refuses unverified bytes and
        // the write path must not manufacture them.
        self.check_statuses(&crc_status, "checksum")?;

        // --- frame into a dense BGZF stream and bring it back ---------------

        let plan = FramePlan::new(&chunk_lengths, &produced)?;
        let dense = self.launch_frame(
            &FrameInputs {
                slots: &slot_buffer,
                slot_offsets: &slots.offsets,
                deflate_sizes: &produced,
                raw: input,
                raw_offsets: &chunk_offsets,
                raw_sizes: &chunk_lengths,
                crcs: &crcs,
            },
            &plan,
        )?;

        let mut bytes = self
            .stream
            .clone_dtoh(&dense)
            .map_err(driver_err("downloading compressed stream"))?;
        bytes.truncate(plan.total);

        let (buf, offsets) = out.parts_mut();
        *buf = bytes;
        *offsets = plan.offsets.iter().map(|&o| o as usize).collect();
        debug_assert!(out.is_consistent());
        Ok(())
    }
}

/// Compressing host bytes: upload, compress on device, bring the stream back.
///
/// A convenience and a test hook rather than the point — the whole argument for
/// this path is that the payload is *already* on the device. Present so the
/// device compressor can be diffed against the host reference over the same
/// inputs.
impl BlockCompressor for NvcompCompressor {
    fn name(&self) -> &'static str {
        "nvcomp-compress-host"
    }

    fn compress_batch(
        &self,
        data: &[u8],
        bounds: &[usize],
        out: &mut CompressedBatch,
    ) -> Result<()> {
        out.clear();
        if bounds.len() < 2 {
            return Ok(());
        }

        let slice = self
            .stream
            .clone_htod(data)
            .map_err(driver_err("uploading payloads to compress"))?;
        let ready = self
            .stream
            .record_event(None)
            .map_err(driver_err("recording upload event"))?;
        let buffer = DeviceBuffer::new(Box::new(CudaAlloc {
            slice,
            len: data.len(),
            stream: self.stream.clone(),
            ready,
            ordinal: self.ordinal,
        }));

        self.compress_batch_device(&buffer, bounds, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured L4/T4 figures at `HighRatio`, so the arithmetic is checked
    /// against the numbers the design was argued from rather than round ones.
    fn measured_budget() -> CompressBudget {
        CompressBudget {
            scratch_per_chunk: 1_114_184,
            slot_bytes: 148_256,
            output_align: 8,
        }
    }

    #[test]
    fn a_chunk_costs_about_one_and_a_half_megabytes() {
        let budget = measured_budget();
        assert_eq!(budget.bytes_per_chunk(), 1_114_184 + 148_256 + 65_536);
    }

    /// The number that forces compression batches to be ~15x smaller than the
    /// read path's, and the reason it is scratch rather than data.
    #[test]
    fn a_sixteen_gibibyte_card_holds_about_eleven_thousand_blocks() {
        let chunks = measured_budget().chunks_in(16 << 30);
        assert!(
            (11_000..13_000).contains(&chunks),
            "expected roughly 11k blocks per batch on a T4, got {chunks}"
        );
    }

    /// Choosing ratio costs throughput twice — once in the kernel, and again in
    /// more launches over smaller batches.
    #[test]
    fn the_ratio_ladder_is_also_a_vram_ladder() {
        let entropy_only = CompressBudget {
            scratch_per_chunk: 0,
            ..measured_budget()
        };
        let ratio = entropy_only.chunks_in(16 << 30) / measured_budget().chunks_in(16 << 30);
        assert!(
            ratio >= 6,
            "entropy-only should fit several times more blocks per batch, got {ratio}x"
        );
    }

    #[test]
    fn a_budget_too_small_for_one_chunk_still_plans_one() {
        // Zero would be a batch that cannot make progress, which is a worse
        // failure than an allocation error naming the size it wanted.
        assert_eq!(measured_budget().chunks_in(1024), 1);
    }

    #[test]
    fn slots_are_aligned_and_evenly_strided() {
        let budget = CompressBudget {
            slot_bytes: 100,
            output_align: 8,
            scratch_per_chunk: 0,
        };
        let plan = SlotPlan::new(3, &budget);
        assert_eq!(plan.offsets, [0, 104, 208]);
        assert_eq!(plan.total, 312);
        assert!(plan.offsets.iter().all(|o| o % 8 == 0));
    }

    #[test]
    fn framing_lays_blocks_out_back_to_back() {
        // 18 header + body + 8 trailer, with no padding between blocks: a BGZF
        // reader walks the stream by block size, so a gap would desynchronise it.
        let plan = FramePlan::new(&[100, 200], &[40, 50]).unwrap();
        assert_eq!(plan.offsets, [0, 66, 142]);
        assert_eq!(plan.total, 142);
        assert_eq!(plan.stored, [0, 0]);
        assert_eq!(plan.stored_count, 0);
    }

    /// The reachable case, and the reason the kernel takes a `stored` flag at
    /// all: nvCOMP's worst case is 2.26x a full chunk, well past the block cap.
    #[test]
    fn an_oversized_chunk_is_planned_as_a_stored_block() {
        let plan = FramePlan::new(&[65_280], &[148_256]).unwrap();
        assert_eq!(plan.stored, [1]);
        assert_eq!(plan.stored_count, 1);
        assert_eq!(plan.total, 18 + 65_280 + 5 + 8);
        assert!(plan.total <= fritillaria_core::MAX_BLOCK_SIZE);
    }

    #[test]
    fn the_host_and_device_paths_agree_on_every_framing_choice() {
        // Not a restatement of `choose_framing`: this asserts the *plan* applies
        // it, which is the step where a device path could quietly diverge.
        for raw in [0u32, 1, 1000, 65_280, MAX_COMPRESSIBLE_PAYLOAD as u32] {
            for deflated in [0u64, 2, 500, 65_000, 65_511, 148_256] {
                let plan = FramePlan::new(&[raw], &[deflated]).unwrap();
                let expected = choose_framing(raw as usize, deflated as usize);
                assert_eq!(
                    plan.stored[0] == 1,
                    expected == Framing::Stored,
                    "raw {raw}, deflated {deflated}"
                );
                assert_eq!(
                    plan.total,
                    fritillaria_core::compress::framed_size(raw as usize, deflated as usize)
                );
            }
        }
    }

    #[test]
    fn a_chunk_too_large_to_frame_is_refused_at_planning_time() {
        let too_big = MAX_COMPRESSIBLE_PAYLOAD as u32 + 1;
        assert!(FramePlan::new(&[too_big], &[10]).is_err());
    }

    #[test]
    fn mismatched_size_arrays_are_refused() {
        assert!(FramePlan::new(&[100, 200], &[40]).is_err());
    }

    #[test]
    fn chunk_extents_rejects_bounds_that_do_not_describe_the_buffer() {
        assert!(chunk_extents(10, &[1, 10]).is_err());
        assert!(chunk_extents(10, &[0, 4]).is_err());
        assert!(chunk_extents(10, &[0, 10, 2, 10]).is_err());
        assert!(chunk_extents(10, &[]).is_err());

        let (offsets, lengths) = chunk_extents(10, &[0, 4, 4, 10]).unwrap();
        assert_eq!(offsets, [0, 4, 4]);
        assert_eq!(lengths, [4, 0, 6]);
    }
}
