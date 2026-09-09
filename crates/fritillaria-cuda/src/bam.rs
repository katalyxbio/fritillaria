//! Columnar BAM decode on the device.
//!
//! Turns a [`DeviceInflateBatch`] of bytes into a [`DeviceRecordBatch`] of
//! records — without either ever leaving the device. This is the part nvCOMP
//! cannot do: it will hand anyone bytes, but it does not know where record
//! 40,000 starts or what its position is.
//!
//! The algorithm and the reason it is sound are in
//! [`fritillaria_bam::blocked`], which is the CPU reference these kernels are a
//! translation of and the oracle they are diffed against.
//! `kernels/bam_decode.cu` carries the same explanation next to the code.
//!
//! # Why this lives here rather than in `fritillaria-bam`
//!
//! Launching a kernel means naming a context, a stream and a pointer, and
//! CLAUDE.md's rule is that those live in this crate and nowhere else. So the
//! dependency points this way: `fritillaria-cuda` knows about BAM, and
//! `fritillaria-bam` stays free of any GPU crate, holding its columns as
//! opaque [`DeviceBuffer`](fritillaria_core::DeviceBuffer)s.
//!
//! # Ownership, and the one way to get this wrong
//!
//! The variable-length fields are **not copied**. `record_offsets` and the
//! field boundaries index into the inflate batch, which must therefore outlive
//! the record batch. Dropping the [`DeviceInflateBatch`] while a kernel is
//! still reading columns off it frees the bytes underneath — silent corruption,
//! not an error. See [`fritillaria_bam::device`] for why pointing at resident
//! bytes beats copying them.

use std::time::Duration;

use fritillaria_bam::device::DeviceRecordBatch;
use fritillaria_core::{DeviceInflateBatch, Result};
// Only the stub path constructs an error here; with `cuda` on, every error
// comes from `cuda_impl`.
#[cfg(not(feature = "cuda"))]
use fritillaria_core::Error;

/// Maps a status code from `bam_reconcile` to a message; mirrors the `FR_BAM_*`
/// constants in `kernels/bam_decode.cu`. Keep the two in sync.
#[cfg(feature = "cuda")]
fn status_message(status: u64) -> &'static str {
    match status {
        1 => "malformed record reached from a confirmed boundary",
        _ => "unknown device error",
    }
}

/// Where time goes inside a decode, phase by phase.
///
/// Same caveat as [`InflateTimings`](crate::InflateTimings): attributing phases
/// means synchronising between them, which removes overlap a pipelined
/// implementation would get. The sum is an upper bound on wall clock, not a
/// measurement of it.
///
/// `reconcile` is the one to watch. It is a single GPU thread walking the true
/// record chain over *blocks*, so it scales with the block count rather than
/// the record count — irrelevant at a few hundred blocks per batch, and the
/// first thing to check if whole-file batches ever get used.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeTimings {
    /// Uploading block starts and allocating the per-block scratch arrays.
    pub setup: Duration,
    /// Phase 1: speculative walks, one thread per block.
    pub scan: Duration,
    /// Phase 2: the serial reconcile, one thread over blocks.
    pub reconcile: Duration,
    /// Downloading the three totals — the one host synchronisation.
    pub totals: Duration,
    /// Phase 3: emitting record offsets, one thread per block.
    pub emit: Duration,
    /// Phase 4: field decode, one thread per record.
    pub decode: Duration,

    pub batches: u64,
    pub blocks: u64,
    pub records: u64,
    pub bytes: u64,
}

impl DecodeTimings {
    /// Folds another batch's timings into this one.
    pub fn accumulate(&mut self, other: &Self) {
        self.setup += other.setup;
        self.scan += other.scan;
        self.reconcile += other.reconcile;
        self.totals += other.totals;
        self.emit += other.emit;
        self.decode += other.decode;
        self.batches += other.batches;
        self.blocks += other.blocks;
        self.records += other.records;
        self.bytes += other.bytes;
    }

    /// Sum of the measured phases.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.setup + self.scan + self.reconcile + self.totals + self.emit + self.decode
    }
}

/// Decodes BAM records into device-resident columns.
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it rather than per batch.
#[derive(Debug)]
pub struct BamDecoder {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Inner,
}

impl BamDecoder {
    /// Opens the default device on a context of our own and compiles the
    /// kernels.
    ///
    /// Convenience for tests and standalone use. **A caller embedding this in
    /// their own pipeline should use [`with_context`](Self::with_context)**:
    /// columns allocated in a context of our own are unusable to their kernels
    /// without a peer copy, which is the transfer this whole design removes.
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: cuda_impl::Inner::new(0)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }

    /// Builds a decoder on a CUDA context and stream the **caller** owns.
    ///
    /// `stream` must belong to `ctx`. Use the same context the codec that
    /// produced the batch was built with, or the columns will point at memory
    /// this context cannot address.
    #[cfg(feature = "cuda")]
    pub fn with_context(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            inner: cuda_impl::Inner::with_context(ctx, stream)?,
        })
    }

    /// Discovers record boundaries and decodes fields into columns.
    ///
    /// `start` is where the first record begins — after the header for the
    /// first batch of a file, and 0 for a continuation. The returned batch's
    /// [`tail`](DeviceRecordBatch::tail) is the offset of the first
    /// incompletely-buffered record, which the caller carries forward exactly
    /// as with [`scan_records`](fritillaria_bam::scan_records).
    ///
    /// The columns borrow `batch` in every sense but the type system's: it must
    /// outlive the result. See the module docs.
    pub fn decode(&self, batch: &DeviceInflateBatch, start: usize) -> Result<DeviceRecordBatch> {
        self.decode_timed(batch, start, &mut DecodeTimings::default())
    }

    /// [`decode`](Self::decode), recording where the time went.
    ///
    /// Synchronises between phases to attribute them, so this is slower than
    /// [`decode`](Self::decode) and its total is an upper bound on wall clock.
    pub fn decode_timed(
        &self,
        batch: &DeviceInflateBatch,
        start: usize,
        timings: &mut DecodeTimings,
    ) -> Result<DeviceRecordBatch> {
        #[cfg(feature = "cuda")]
        {
            self.inner.decode(batch, start, timings)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, start, timings);
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;
    use std::time::Instant;

    use cudarc::driver::{
        CudaContext as RawContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use fritillaria_bam::device::{DeviceColumns, DeviceRecordBatch};
    use fritillaria_core::{DeviceAlloc, DeviceBuffer, DeviceInflateBatch, Error, Result};

    use super::{DecodeTimings, status_message};
    use crate::backend::{CudaAlloc, driver_err, load_kernel};

    const BLOCK_DIM: u32 = 256;

    /// What the boundary phases produced: the record offset column, and the
    /// two numbers the host had to come back for.
    struct Boundaries {
        record_offsets: CudaSlice<u8>,
        n_records: usize,
        tail: usize,
    }

    /// Element width of every column, in the order the kernel writes them.
    const FIXED_WIDTHS: [usize; 8] = [4, 4, 2, 1, 4, 4, 4, 4];
    const BOUND_WIDTHS: [usize; 5] = [4, 4, 4, 4, 4];

    pub(super) struct Inner {
        /// Held for its lifetime, not used directly: the loaded kernels belong
        /// to modules in this context, so it must outlive them.
        _ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        ordinal: i32,
        scan: CudaFunction,
        reconcile: CudaFunction,
        emit: CudaFunction,
        decode: CudaFunction,
    }

    impl std::fmt::Debug for Inner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("BamDecoder")
                .field("ordinal", &self.ordinal)
                .finish_non_exhaustive()
        }
    }

    fn grid(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK_DIM as usize).max(1) as u32, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// One thread. The reconcile phase is a dependent chain by nature.
    const SINGLE: LaunchConfig = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    impl Inner {
        fn from_parts(ctx: Arc<RawContext>, stream: Arc<CudaStream>, ordinal: i32) -> Result<Self> {
            let src = crate::BAM_DECODE_KERNEL_SRC;
            Ok(Self {
                scan: load_kernel(&ctx, src, "bam_decode.cu", "bam_scan_blocks")?,
                reconcile: load_kernel(&ctx, src, "bam_decode.cu", "bam_reconcile")?,
                emit: load_kernel(&ctx, src, "bam_decode.cu", "bam_emit_offsets")?,
                decode: load_kernel(&ctx, src, "bam_decode.cu", "bam_decode_fields")?,
                _ctx: ctx,
                stream,
                ordinal,
            })
        }

        pub(super) fn new(ordinal: usize) -> Result<Self> {
            if !crate::backend::driver_is_available() {
                return Err(Error::CudaUnavailable("no CUDA driver".to_string()));
            }
            let ctx = RawContext::new(ordinal).map_err(driver_err("opening device"))?;
            let stream = ctx.default_stream();
            let n = i32::try_from(ordinal)
                .map_err(|_| Error::Cuda(format!("device ordinal {ordinal} exceeds i32")))?;
            Self::from_parts(ctx, stream, n)
        }

        pub(super) fn with_context(ctx: Arc<RawContext>, stream: Arc<CudaStream>) -> Result<Self> {
            let ordinal = i32::try_from(ctx.ordinal())
                .map_err(|_| Error::Cuda("device ordinal exceeds i32".to_string()))?;
            Self::from_parts(ctx, stream, ordinal)
        }

        /// Allocates `n * width` bytes, padded to one when empty.
        ///
        /// A zero-sized CUDA allocation is invalid and an empty batch is legal,
        /// so the pad exists; the logical length stays 0 so it never leaks into
        /// `byte_len` or a download.
        fn alloc(&self, n: usize, width: usize) -> Result<CudaSlice<u8>> {
            self.stream
                .alloc_zeros::<u8>((n * width).max(1))
                .map_err(driver_err("allocating column"))
        }

        /// Blocks until queued work finishes.
        ///
        /// Only for phase attribution: launches are asynchronous, so without
        /// this every phase would be charged to whichever one synchronises
        /// next. It is why `decode_timed` is slower than `decode`.
        fn sync(&self) -> Result<()> {
            self.stream
                .synchronize()
                .map_err(driver_err("synchronising between decode phases"))
        }

        /// Wraps a device slice as an owning column of `bytes` logical bytes.
        ///
        /// Each column carries its own completion event, all recorded after the
        /// decode launch, so a consumer can wait on whichever column it is
        /// about to read without synchronising the host.
        fn column(&self, slice: CudaSlice<u8>, bytes: usize) -> Result<DeviceBuffer> {
            let ready = self
                .stream
                .record_event(None)
                .map_err(driver_err("recording completion event"))?;
            Ok(DeviceBuffer::new(Box::new(CudaAlloc {
                slice,
                len: bytes,
                stream: self.stream.clone(),
                ready,
                ordinal: self.ordinal,
            })))
        }

        pub(super) fn decode(
            &self,
            batch: &DeviceInflateBatch,
            start: usize,
            timings: &mut DecodeTimings,
        ) -> Result<DeviceRecordBatch> {
            let Some(data) = batch.data() else {
                return self.empty(start);
            };
            let alloc = data
                .alloc()
                .as_any()
                .downcast_ref::<CudaAlloc>()
                .ok_or_else(|| Error::InvalidDeviceBatch {
                    reason: "batch was not produced by the CUDA backend".to_string(),
                })?;
            if alloc.device_ordinal() != self.ordinal {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "batch is on device {} but this decoder is on {}",
                        alloc.device_ordinal(),
                        self.ordinal
                    ),
                });
            }

            let len = data.byte_len();
            if len == 0 {
                return self.empty(start);
            }

            timings.batches += 1;
            timings.blocks += batch.len() as u64;
            timings.bytes += len as u64;

            let bounds = self.boundaries(alloc, len, batch, start, timings)?;
            timings.records += bounds.n_records as u64;
            let columns = self.decode_fields(alloc, bounds, timings)?;
            DeviceRecordBatch::new(columns.0, columns.1, columns.2)
        }

        /// Phases 1-3: speculate per block, reconcile the true chain, emit the
        /// record offsets. Returns the offsets column and what the reconcile
        /// found.
        fn boundaries(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            batch: &DeviceInflateBatch,
            start: usize,
            timings: &mut DecodeTimings,
        ) -> Result<Boundaries> {
            let phase = Instant::now();
            let n_blocks = batch.len();
            let n_blocks_arg = i32::try_from(n_blocks)
                .map_err(|_| Error::Cuda(format!("{n_blocks} blocks exceeds i32")))?;
            let len_arg = len as u64;
            let start_arg = start as u64;

            let starts: Vec<u64> = batch.offsets().iter().map(|&o| o as u64).collect();
            let block_starts = self
                .stream
                .clone_htod(&starts)
                .map_err(driver_err("uploading block starts"))?;

            // Phase 1: speculate, one thread per block.
            let mut spec_counts = self.alloc(n_blocks, 4)?;
            let mut spec_lands = self.alloc(n_blocks, 8)?;
            let mut spec_complete = self.alloc(n_blocks, 1)?;
            let mut spec_valid = self.alloc(n_blocks, 1)?;
            self.sync()?;
            timings.setup += phase.elapsed();
            let phase = Instant::now();

            let mut builder = self.stream.launch_builder(&self.scan);
            builder
                .arg(alloc.slice())
                .arg(&len_arg)
                .arg(&block_starts)
                .arg(&n_blocks_arg)
                .arg(&mut spec_counts)
                .arg(&mut spec_lands)
                .arg(&mut spec_complete)
                .arg(&mut spec_valid);
            // SAFETY: the kernel signature matches the arguments above. Every
            // read is bounds-checked against `len` inside the kernel, which is
            // the batch's own logical length, and each thread writes only its
            // own element of the four output arrays.
            unsafe { builder.launch(grid(n_blocks)) }
                .map_err(driver_err("launching bam_scan_blocks"))?;
            self.sync()?;
            timings.scan += phase.elapsed();
            let phase = Instant::now();

            // Phase 2: reconcile, one thread walking the true chain over blocks.
            let mut entries = self.alloc(n_blocks, 8)?;
            let mut counts = self.alloc(n_blocks, 4)?;
            let mut first_index = self.alloc(n_blocks, 4)?;
            let mut totals = self.alloc(3, 8)?;

            let mut builder = self.stream.launch_builder(&self.reconcile);
            builder
                .arg(alloc.slice())
                .arg(&len_arg)
                .arg(&block_starts)
                .arg(&n_blocks_arg)
                .arg(&start_arg)
                .arg(&spec_counts)
                .arg(&spec_lands)
                .arg(&spec_complete)
                .arg(&spec_valid)
                .arg(&mut entries)
                .arg(&mut counts)
                .arg(&mut first_index)
                .arg(&mut totals);
            // SAFETY: as above. The single thread writes one element per block
            // plus three totals, and its walk is bounds-checked against `len`
            // at every step.
            unsafe { builder.launch(SINGLE) }.map_err(driver_err("launching bam_reconcile"))?;
            self.sync()?;
            timings.reconcile += phase.elapsed();
            let phase = Instant::now();

            // The one genuine host synchronisation: the record count decides
            // how much to allocate next, so it cannot be deferred. Three words.
            let totals: Vec<u64> = self
                .stream
                .clone_dtoh(&totals)
                .map_err(driver_err("downloading record totals"))?
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().expect("chunk of 8")))
                .collect();
            let (n_records, tail, status) = (totals[0], totals[1], totals[2]);
            if status != 0 {
                return Err(Error::Malformed {
                    format: "bam",
                    position: tail,
                    reason: status_message(status).to_string(),
                });
            }
            let n_records = usize::try_from(n_records)
                .map_err(|_| Error::Cuda("record count exceeds usize".to_string()))?;
            let tail = usize::try_from(tail)
                .map_err(|_| Error::Cuda("tail offset exceeds usize".to_string()))?;
            timings.totals += phase.elapsed();
            let phase = Instant::now();

            // Phase 3: emit offsets, one thread per block.
            let mut record_offsets = self.alloc(n_records, 8)?;
            let mut builder = self.stream.launch_builder(&self.emit);
            builder
                .arg(alloc.slice())
                .arg(&entries)
                .arg(&counts)
                .arg(&first_index)
                .arg(&n_blocks_arg)
                .arg(&mut record_offsets);
            // SAFETY: `first_index` is the exclusive prefix sum of `counts`, so
            // block b writes exactly `counts[b]` elements starting at
            // `first_index[b]`; the ranges tile `[0, n_records)` without
            // overlap. Both were produced by the reconcile kernel above.
            unsafe { builder.launch(grid(n_blocks)) }
                .map_err(driver_err("launching bam_emit_offsets"))?;
            self.sync()?;
            timings.emit += phase.elapsed();

            Ok(Boundaries {
                record_offsets,
                n_records,
                tail,
            })
        }

        /// Phase 4: one thread per record, writing every column.
        ///
        /// Returns the record count and tail alongside the columns so the
        /// caller assembles the batch in one place.
        fn decode_fields(
            &self,
            alloc: &CudaAlloc,
            bounds: Boundaries,
            timings: &mut DecodeTimings,
        ) -> Result<(usize, usize, DeviceColumns)> {
            let phase = Instant::now();
            let Boundaries {
                record_offsets,
                n_records: n,
                tail,
            } = bounds;
            let n_arg =
                u32::try_from(n).map_err(|_| Error::Cuda(format!("{n} records exceeds u32")))?;

            let mut reference_sequence_id = self.alloc(n, FIXED_WIDTHS[0])?;
            let mut position = self.alloc(n, FIXED_WIDTHS[1])?;
            let mut flags = self.alloc(n, FIXED_WIDTHS[2])?;
            let mut mapping_quality = self.alloc(n, FIXED_WIDTHS[3])?;
            let mut sequence_len = self.alloc(n, FIXED_WIDTHS[4])?;
            let mut mate_reference_sequence_id = self.alloc(n, FIXED_WIDTHS[5])?;
            let mut mate_position = self.alloc(n, FIXED_WIDTHS[6])?;
            let mut template_length = self.alloc(n, FIXED_WIDTHS[7])?;
            let mut cigar_start = self.alloc(n, BOUND_WIDTHS[0])?;
            let mut sequence_start = self.alloc(n, BOUND_WIDTHS[1])?;
            let mut quality_start = self.alloc(n, BOUND_WIDTHS[2])?;
            let mut aux_start = self.alloc(n, BOUND_WIDTHS[3])?;
            let mut record_end = self.alloc(n, BOUND_WIDTHS[4])?;

            let mut builder = self.stream.launch_builder(&self.decode);
            builder
                .arg(alloc.slice())
                .arg(&record_offsets)
                .arg(&n_arg)
                .arg(&mut reference_sequence_id)
                .arg(&mut position)
                .arg(&mut flags)
                .arg(&mut mapping_quality)
                .arg(&mut sequence_len)
                .arg(&mut mate_reference_sequence_id)
                .arg(&mut mate_position)
                .arg(&mut template_length)
                .arg(&mut cigar_start)
                .arg(&mut sequence_start)
                .arg(&mut quality_start)
                .arg(&mut aux_start)
                .arg(&mut record_end);
            // SAFETY: the kernel signature matches. Every column was allocated
            // with `n` elements of the width the kernel writes, and thread i
            // writes only element i of each. `record_offsets` was filled by the
            // emit kernel on this same stream.
            unsafe { builder.launch(grid(n)) }
                .map_err(driver_err("launching bam_decode_fields"))?;
            self.sync()?;
            timings.decode += phase.elapsed();

            let columns = DeviceColumns {
                reference_sequence_id: self.column(reference_sequence_id, n * 4)?,
                position: self.column(position, n * 4)?,
                flags: self.column(flags, n * 2)?,
                mapping_quality: self.column(mapping_quality, n)?,
                sequence_len: self.column(sequence_len, n * 4)?,
                mate_reference_sequence_id: self.column(mate_reference_sequence_id, n * 4)?,
                mate_position: self.column(mate_position, n * 4)?,
                template_length: self.column(template_length, n * 4)?,
                record_offsets: self.column(record_offsets, n * 8)?,
                cigar_start: self.column(cigar_start, n * 4)?,
                sequence_start: self.column(sequence_start, n * 4)?,
                quality_start: self.column(quality_start, n * 4)?,
                aux_start: self.column(aux_start, n * 4)?,
                record_end: self.column(record_end, n * 4)?,
            };
            Ok((n, tail, columns))
        }

        /// An empty batch: no records, and the tail is where we started.
        fn empty(&self, start: usize) -> Result<DeviceRecordBatch> {
            let column = |width: usize| -> Result<DeviceBuffer> {
                let slice = self.alloc(0, width)?;
                self.column(slice, 0)
            };
            let columns = DeviceColumns {
                reference_sequence_id: column(4)?,
                position: column(4)?,
                flags: column(2)?,
                mapping_quality: column(1)?,
                sequence_len: column(4)?,
                mate_reference_sequence_id: column(4)?,
                mate_position: column(4)?,
                template_length: column(4)?,
                record_offsets: column(8)?,
                cigar_start: column(4)?,
                sequence_start: column(4)?,
                quality_start: column(4)?,
                aux_start: column(4)?,
                record_end: column(4)?,
            };
            DeviceRecordBatch::new(0, start, columns)
        }
    }
}
