# Device-resident API design

**Status:** written 2026-09-08. **All six steps are done.** The core vocabulary (`DeviceAlloc`,
`DeviceBuffer`, `DeviceInflateBatch`, `DeviceBlockCodec`) is in `fritillaria-core::device`, with
host-memory stand-ins behind a `testing` feature and a `HostDeviceCodec` in `fritillaria-bgzf`;
`CudaCodec` implements `DeviceBlockCodec` over the existing kernel and takes a caller-supplied
context and stream, exposing a completion event for cross-stream ordering; `NvcompCodec`
implements the same two traits over NVIDIA's library, dlopened at runtime. **Verified on a Tesla
T4**: 43 device tests, asserting byte-identity against the CPU reference, that output lands in
the caller's own context, and that our kernel and nvCOMP agree with each other on a real htslib
BAM. `DeviceBgzfReader` (step 5) and the columnar device decode (step 6) followed, each verified
on a T4 in turn; the per-step notes below record what each one found.

Two things changed while implementing, both recorded below: `DeviceInflateBatch::data()` returns
`Option<&DeviceBuffer>` because an empty batch genuinely owns no allocation, and a backend fills
a batch via `adopt()` rather than by construction, so device allocations can be reused across
batches.

## The problem

`InflateBatch` is host-resident: a `Vec<u8>` plus block offsets. The CUDA backend inflates on
the device and then copies everything back, which costs 54% of runtime and is pure waste when
the next consumer is another GPU kernel. This design removes that copy and defines what a GPU
tool actually receives.

Four constraints shape every decision below. They are not negotiable and they conflict, which
is why this needs designing rather than just writing:

1. **`fritillaria-core` must not depend on CUDA.** It holds the vocabulary and the trait seam;
   the whole workspace builds and tests on a machine with no toolkit.
2. **Device memory and streams live in `fritillaria-cuda::backend` and nowhere else.** No raw
   pointer or device-tied lifetime may escape into `fritillaria-bam`, `-bcf`, `-bgzf`.
3. **The drop-in must keep working, unchanged.** It is the migration path. Existing code doing
   `bam::io::Reader::from(BgzfReader::with_codec(f, codec))` must not break, and must not be
   made worse to accommodate the columnar path. Since the vendoring this is stronger, not
   weaker: `bam::io::Reader` *is* noodles' reader, byte-for-byte, so breaking it would mean
   editing vendored code and forfeiting the rebase path in `VENDORED.md`.
4. **Verification is mandatory.** CRC32 and `ISIZE` are checked in device mode too. Going
   device-resident must not become a way to silently skip verification.

## Shape of the answer

**Two traits and two batch types, not one of each with a residency flag.**

A runtime `Residency` enum was the obvious first design and it is wrong: it turns "I asked a CPU
codec for device output" into a runtime error that fires deep inside a read, and it forces every
consumer to handle a case that cannot happen for them. Splitting the trait makes the same
mistake a **compile error** and costs nothing, because no caller genuinely wants to be generic
over residency — a `BufRead` implementation fundamentally needs host bytes, and an aligner
kernel fundamentally needs device bytes.

```text
BlockCodec         -> InflateBatch         host bytes    CpuCodec, CudaCodec, NvcompCodec
DeviceBlockCodec   -> DeviceInflateBatch   device bytes            CudaCodec, NvcompCodec
```

`CpuCodec` implements only the first, so `DeviceBgzfReader::new(CpuCodec::new())` does not
compile. `BlockCodec` stays object-safe, so `select_codec() -> Box<dyn BlockCodec>` in the
facade is untouched.

## `fritillaria-core`: naming device memory without depending on CUDA

Core defines the *trait*; `fritillaria-cuda` provides the implementation. Core never names a
CUDA type, and format crates hold `DeviceBuffer` values without knowing what backs them.

```rust
/// An opaque, owning handle to a device allocation.
///
/// Format crates hold these and pass them around. They cannot dereference one:
/// getting at the pointer requires `fritillaria-cuda`, which is the only crate
/// permitted to know it is a `CUdeviceptr`.
pub trait DeviceAlloc: Send + Sync + std::fmt::Debug {
    fn byte_len(&self) -> usize;

    /// Which device this lives on. A consumer on a different device must copy,
    /// and multi-GPU callers need to be able to check rather than crash.
    fn device_ordinal(&self) -> i32;

    /// Downcast point. `fritillaria-cuda` uses this to recover the concrete
    /// allocation; nobody else has a reason to call it.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Explicit, always available, deliberately not free. Copying back to the
    /// host is exactly what this design exists to avoid, so it is a named
    /// method rather than a `Deref` that happens to be expensive.
    fn copy_to_host(&self, dst: &mut [u8]) -> Result<()>;
}

pub struct DeviceBuffer(Box<dyn DeviceAlloc>);
```

`DeviceBuffer` is `Send + Sync` and frees on drop. **Ownership rule, which must be documented on
every type that holds one:** dropping frees the allocation. A consumer whose kernel is still
reading it must keep the batch alive, or take ownership with `into_raw()`.

```rust
/// Inflated output that never left the device.
///
/// Offsets stay on the host: they are ~8 bytes per block (1.3 MB for a 3 GiB
/// BAM), the host needs them for virtual offsets and seeking, and uploading a
/// copy on demand is cheaper than keeping the host blind.
pub struct DeviceInflateBatch {
    /// `None` only when empty: an empty batch owns no allocation.
    data: Option<DeviceBuffer>,
    offsets: Vec<usize>,
    /// Device-side copy of `offsets`, uploaded lazily — only kernels that index
    /// blocks directly need it.
    device_offsets: Option<DeviceBuffer>,
}

impl DeviceInflateBatch {
    pub fn len(&self) -> usize;                 // blocks
    pub fn byte_len(&self) -> usize;            // inflated bytes
    pub fn offsets(&self) -> &[usize];
    pub fn block_range(&self, i: usize) -> Option<Range<usize>>;
    pub fn device_ordinal(&self) -> Option<i32>;
    pub fn data(&self) -> Option<&DeviceBuffer>;

    /// Backends fill a batch through this rather than constructing one, so a
    /// device allocation can be reused across batches — `cudaMalloc` is
    /// expensive enough that designing the reuse seam out would be a mistake.
    /// Validates the offsets against the buffer; a failed call leaves the
    /// previous contents intact.
    pub fn adopt(&mut self, data: DeviceBuffer, offsets: Vec<usize>) -> Result<()>;

    /// The escape hatch back to the host path. Named, explicit, expensive.
    pub fn to_host(&self) -> Result<InflateBatch>;
}
```

The `Option` on `data()` is not noise consumers live with: a reader yields only non-empty
batches, so it is resolved at that boundary.

Layout violations are their own error variant, because they are always a backend bug rather than
bad input, and the failure they prevent is a kernel reading past the end of a column:

```rust
Error::InvalidDeviceBatch { reason: String }
```

```rust
/// Contract identical to `BlockCodec`, including mandatory CRC32/ISIZE
/// verification — see below for how that works without a full round trip.
pub trait DeviceBlockCodec {
    fn name(&self) -> &'static str;
    fn device_ordinal(&self) -> i32;

    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()>;
}
```

### How verification survives

The inflate kernel already folds CRC32 in and writes a per-block status word. Verification
therefore costs **one D2H copy of `4 * n_blocks` bytes** — 664 KB for a 3 GiB BAM, versus the
10.11 GiB we are eliminating. That is a rounding error, and it keeps constraint 4 intact.

It does, however, force a synchronisation point per batch. If that shows up in a profile, the
fix is to check statuses on-device and set a single error flag, downloading one word instead of
`n`. Do not do this speculatively.

## `fritillaria-cuda`: the parts that must name a pointer

Everything device-specific is confined here, satisfying constraint 2.

```rust
/// A borrowed view of device memory: pointer, length, device.
pub struct DeviceSlice { ptr: CUdeviceptr, len: usize, ordinal: i32 }

impl DeviceSlice {
    /// # Safety
    /// Valid only while the owning batch is alive and only on `ordinal`.
    pub unsafe fn as_ptr(&self) -> CUdeviceptr;
}

/// Recover the pointer from an opaque buffer. Returns `None` if the buffer is
/// not CUDA-backed.
pub fn device_slice(buf: &DeviceBuffer) -> Option<DeviceSlice>;
```

### Context and stream: the caller owns both

**A library that insists on creating its own CUDA context cannot be embedded in someone else's
pipeline.** This is the single most important decision here. An aligner already has a context
and streams; allocating in a second context makes every buffer we produce unusable to it without
a peer copy.

```rust
impl CudaCodec {
    /// The normal path when embedding: adopt the caller's context and stream.
    /// Errors if `stream` does not belong to `ctx` — silently accepting that
    /// would hand back memory the caller cannot address from their stream.
    pub fn with_context(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self>;

    /// Convenience for standalone use and tests — creates a context on device 0.
    pub fn new() -> Result<Self>;
}
```

### Asynchronous handoff

A consumer must be able to enqueue its kernel without a host-side synchronise, or we have
replaced a PCIe stall with a latency stall.

```rust
impl CudaCodec {
    /// Event recorded when this batch's inflate completes. The consumer waits
    /// on it with `cudaStreamWaitEvent` and enqueues immediately — no host sync.
    pub fn ready_event(batch: &DeviceInflateBatch) -> Option<&CudaEvent>;
}
```

Verification's status download is the exception: it is a genuine host sync per batch. Callers
who want full overlap should pipeline batch *n+1*'s inflate behind batch *n*'s verification.

Implemented as step 3. The event is recorded **before** any synchronise, so it marks kernel
completion rather than "the host got around to noticing":

```rust
their_stream.wait(CudaCodec::ready_event(&batch).unwrap())?;
launch_their_kernel(&batch, &their_stream);
```

## `fritillaria-bam`: what a GPU tool actually consumes

Bytes in VRAM are not the deliverable — **records** in VRAM are. This is the part nvCOMP cannot
give anyone, and therefore the part that justifies the project.

```rust
pub struct DeviceRecordBatch {
    n: usize,

    // Fixed-width columns, one element per record.
    reference_sequence_id: DeviceBuffer,   // i32
    position: DeviceBuffer,                // i32
    flags: DeviceBuffer,                   // u16
    mapping_quality: DeviceBuffer,         // u8
    sequence_len: DeviceBuffer,            // u32
    mate_reference_sequence_id: DeviceBuffer,
    mate_position: DeviceBuffer,
    template_length: DeviceBuffer,

    // Variable-width fields: an offsets column plus a packed payload, the
    // standard Arrow-style layout. A GPU kernel indexes record i as
    // payload[offsets[i]..offsets[i + 1]].
    name_offsets: DeviceBuffer,    name_data: DeviceBuffer,
    cigar_offsets: DeviceBuffer,   cigar_data: DeviceBuffer,   // u32 ops
    sequence_offsets: DeviceBuffer, sequence_data: DeviceBuffer, // 4-bit packed
    quality_offsets: DeviceBuffer, quality_data: DeviceBuffer,
    aux_offsets: DeviceBuffer,     aux_data: DeviceBuffer,     // raw tag block
}
```

Three decisions worth stating, because each is a place a naive implementation loses the win:

- **Sequence stays 4-bit packed.** Unpacking doubles the memory and every consumer that wants
  2-bit or one-hot has to re-encode anyway. Offer `unpack_sequences()` as an opt-in kernel;
  do not do it by default.
- **Aux tags stay as a raw payload column plus offsets**, with tag *lookup* as a separate
  on-device index built only when asked. Long-read uBAM carries large `MM`/`ML`/`mv` tags, so
  eagerly parsing every tag would cost more than most consumers want to pay. But see the
  roadmap: decoding them at all is currently missing and is blocking for ONT/PacBio input.
- **Columns are separate allocations, not a struct-of-arrays in one block.** A consumer that
  only wants `position` and `flags` should not have to keep the sequence payload resident.

## What this looks like to a caller

Host path, unchanged — this is constraint 3 and it must keep compiling verbatim:

```rust
let mut reader = bam::io::Reader::from(BgzfReader::with_codec(file, CudaCodec::new()?));
```

Device path, new:

```rust
// The caller's context and stream, not ours.
let codec = CudaCodec::with_context(my_ctx.clone(), my_stream.clone());
let mut reader = DeviceBgzfReader::new(file, codec);

while let Some(batch) = reader.next_batch()? {          // DeviceInflateBatch
    let records = DeviceRecordBatch::decode(&batch)?;   // still on device

    my_stream.wait_event(CudaCodec::ready_event(&batch).unwrap());
    launch_my_aligner(&records, &my_stream);            // no host round trip

    // `records` and `batch` must outlive the kernel — see the ownership rule.
}
```

## Open questions

- ~~**Record boundary scan on device.**~~ **Answered, and none of the three options listed
  here was the answer.** The host version was going to need the bytes on the host, which is the
  transfer this design exists to delete; a device-side serial scan is 55,000 dependent loads;
  and "speculative parallel scan with validation" was right in spirit but vague about what makes
  it *correct* rather than usually-right.

  What resolved it was a property of htslib discovered while adding the ONT fixture: it calls
  `bgzf_flush_try` before each record and starts a new block rather than splitting one, so a
  block start is almost always a record start. Block starts are therefore *candidate*
  boundaries, already known host-side from the ISIZE prefix sum, giving one independent chain
  per block.

  Correctness does not rest on the guess. A block's speculative walk is adopted only once the
  true chain is proven to arrive at that block's start, and a walk from a true boundary is the
  true walk; everything else falls back to walking. So a wrong guess costs work and never
  accuracy — which is what makes ultra-long ONT reads, where a record covers whole blocks and
  most guesses are useless, correct rather than merely slow. `fritillaria-bam/src/blocked.rs` is
  the CPU reference and `kernels/bam_decode.cu` the translation.
- **Memory pressure.** Whole-batch device residency has a VRAM ceiling that host-resident output
  does not. 256 blocks is ~16 MiB, fine — but a caller holding many batches for a windowed
  algorithm is not. Needs either a pool or explicit backpressure.
- **Does `DeviceRecordBatch` belong in `fritillaria-bam` or a `fritillaria-bam-gpu`?** It names
  no CUDA type, so `-bam` is legal under constraint 2. Whether it is *tasteful* is a separate
  question; revisit if `-bam` accumulates device-shaped API.
- ~~**nvCOMP's output buffer.**~~ **Answered, and the answer is the good one.** Measured against
  nvCOMP 5.3.0.16's own headers and confirmed by calling the library:
  `nvcompBatchedDeflateDecompressAsync` takes an **array of output pointers**, and reports an
  **output alignment of 1**. So it writes straight into our densely packed buffer at the ISIZE
  prefix-sum offsets — no padding, no device-to-device compaction pass. `DeviceInflateBatch`'s
  shape is safe as designed.

  This was the right question to have asked. Dense packing is not a preference: a BAM record can
  span a block boundary, so a consumer must be able to read across one. Had nvCOMP wanted padded
  output, the fix would have been a full extra pass over the *inflated* data — the largest thing
  in the pipeline — which would have eaten most of what nvCOMP was adopted for.

  Two more numbers from the same probe, both favourable: **temp workspace is 0 bytes** for
  deflate decompression, and **input alignment is 4**. Only the last one costs anything, and it
  is dealt with in `kernels/gather.cu`.

## Implementation order

Each step is independently testable.

> **This list was wrong once, and the failure mode is worth remembering.** The original version
> had five steps covering only the *data path* — types, codec, nvCOMP, reader, columns — and
> silently omitted caller-owned context and the ready event, despite this document calling the
> former "the single most important decision here". A decomposition that enumerates the happy
> path of data flow will quietly drop interop concerns, because they are not steps *along* that
> path. **When splitting a design into steps, check the step list against the design's own
> stated decisions, not just against its data structures.**

1. ~~`DeviceAlloc` / `DeviceBuffer` / `DeviceInflateBatch` in core, with a **host-backed test
   implementation** so the whole design is testable with no GPU.~~ **Done.**
   `fritillaria_core::device`, plus `device::testing::HostAlloc` behind core's `testing` feature
   and `fritillaria_bgzf::HostDeviceCodec` behind bgzf's. The stand-in codec runs `CpuCodec` and
   wraps the result in host memory pretending to be device memory: every part of the seam is
   real except the transfer. Verified that `takes_device(CpuCodec::new())` fails to compile with
   `the trait DeviceBlockCodec is not implemented for CpuCodec`, which is the whole point of the
   two-trait split.
2. ~~`DeviceBlockCodec` for `CudaCodec`, reusing the existing kernel.~~ **Done, verified on a
   T4.** `launch_inflate` now leaves the payload on the device and returns only the per-block
   verification arrays; the host path downloads the payload as an explicit extra step and the
   device path skips it. That is the *only* difference between them — both call the same
   `verify_and_offsets`, because two copies of that logic is how the device path would drift
   into skipping a check the host path still makes.

   `CudaAlloc` carries a **logical** length that may be smaller than the allocation: a batch of
   only EOF blocks inflates to zero bytes and a zero-sized CUDA allocation is invalid, so the
   buffer is padded to one byte while reporting length 0. `tests/device_resident.rs` pins that
   the pad never leaks, alongside 10 more differential tests.
3. ~~**Interop: caller-owned context/stream, and the ready event.**~~ **Done, verified on a T4.**
   *Missing from the original list; see the note above.* `CudaContext::with_context(ctx, stream)`
   and `CudaCodec::with_context`, validating that the stream belongs to the context; an event
   recorded on the stream right after the inflate launch, reachable as
   `CudaAlloc::ready_event()` or `CudaCodec::ready_event(&batch)`. cudarc is re-exported as
   `fritillaria_cuda::cudarc` so an embedding caller builds those types against the version this
   crate links.

   Without this the device path is *correct but unusable*: output allocated in a context of our
   own needs a peer copy to reach the caller's kernels, which is the transfer the whole design
   removes.

   **`CudaContext::new(ordinal)` retains the device's primary context**, so two calls for the
   same device return the *same* context and a stream from either works with the other. The
   context check therefore only fires across devices, and its test is gated on
   `device_count() >= 2` — unreachable on a single-GPU VM.
4. ~~`NvcompCodec` implementing both traits, behind an `nvcomp` feature.~~ **Done, verified on a
   T4.** 12 differential tests in `tests/nvcomp.rs`, with the CPU reference as the oracle and
   our own kernel as a second one — three independent implementations agreeing byte-for-byte on
   a real htslib BAM.

   **The library is dlopened, not linked.** nvCOMP is proprietary, absent from the Colab image,
   and absent from this project's development machine. Linking it would make
   `cargo build --features nvcomp` impossible to run where the feature is actually developed,
   and vendoring it is not permitted. dlopen is also what cudarc already does for `libcuda`, so
   this adds no new kind of dependency. The cost is hand-maintained FFI declarations; they are
   pinned by a runtime version check and by layout assertions on every by-value struct, and
   `ffi::tests::the_real_library_agrees_with_these_declarations` calls the real library **with
   no GPU** — nvCOMP links CUDA statically and the alignment/temp entry points never touch the
   device, so the FFI surface is testable locally instead of costing a rented VM.

   **Verification stays ours.** nvCOMP's deflate entry point never sees the gzip trailer and its
   documentation says only limited validation is performed, so `nvcompBatchedCRC32Async` (whose
   `nvcompCRC32` preset is exactly gzip's CRC-32/ISO-HDLC) computes a checksum per block and it
   is compared against the CRC we parsed out of the trailer during host-side discovery. That is
   *stronger* than a self-check, because the comparison is against what the file itself records.
   Both engines then go through the same `verify_and_offsets`.

   **The one cost: input alignment.** nvCOMP wants 4-byte-aligned input chunks; a BGZF payload
   starts 18 bytes into its gzip member, at an arbitrary file offset. `kernels/gather.cu`
   restages payloads into aligned slots — one thread block per BGZF block, so the copy is
   coalesced — and is skipped entirely when a batch already satisfies the requirement. Doing
   this on the host instead would mean a memcpy of the whole compressed batch through host
   memory; on the device it runs at VRAM bandwidth over data already there.
5. ~~`DeviceBgzfReader`, driving successive batches and carrying the partial record forward.~~
   **Done, verified on a T4.** Landed after step 6, because the boundary scan was the open
   question and this is plumbing on top of the answer.

   The design note worth keeping: **the carry moves compressed blocks, not decompressed bytes.**
   Given the caller's tail offset the reader re-inflates the block containing it and everything
   after, at the front of the next batch. No device-to-device copy, no second buffer; the
   overlap is bounded by one record. The same mechanism makes a record larger than a batch
   resolve itself, since each round prepends the carry and appends a fresh read until the window
   is big enough.

   `next_batch` yields an **owned** `DeviceInflateBatch`. Lending one and reusing it would free
   the bytes columns point into while a consumer was still reading them; handing it over makes
   that lifetime the borrow checker's business. The cost is an allocation per batch, which is
   the right trade against a silent-corruption failure mode.
6. ~~`DeviceRecordBatch::decode` as a kernel, differential-tested against `RecordBatch::decode`.~~
   **Done, verified on a T4.** Four kernels — speculate per block, reconcile the true chain,
   emit offsets, decode fields — in `kernels/bam_decode.cu`, launched by
   `fritillaria_cuda::bam::BamDecoder`. 6 differential tests over all four fixtures.

   **The columns live in `fritillaria-bam` and the launcher in `fritillaria-cuda`, so the
   dependency points `-cuda -> -bam`.** Launching a kernel means naming a context, a stream and
   a pointer, and those are confined to `-cuda` by constraint 2; pointing the dependency the
   other way would have put them in a format crate. `DeviceRecordBatch` names no CUDA type.

   Variable-length fields are **not** copied into packed payload columns as sketched above.
   Sequence and qualities are the bulk of a BAM and long-read qualities are a byte per base, so
   duplicating them would roughly double VRAM for the batch and add a pass over the largest
   thing in the pipeline. They stay in the inflated buffer, reached through offset columns —
   which makes the ownership rule sharper: **the `DeviceInflateBatch` must outlive the
   `DeviceRecordBatch`.** A `compact()` remains the opt-in for consumers that want to drop the
   source.
