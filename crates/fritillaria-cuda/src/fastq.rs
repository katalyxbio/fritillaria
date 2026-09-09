//! FASTQ record boundary discovery and columnar decode on device.
//!
//! The CPU reference is `fritillaria_fastq::columnar::speculative` and the
//! kernels are `kernels/fastq_scan.cu`; read the former for the correctness
//! argument. This module is the launcher between them.
//!
//! # What runs where
//!
//! Same division as [`bcf`](crate::bcf), for the same reason. The sieve touches
//! every byte of the batch, so it is on device. The sort and the tiling proof
//! operate on a few thousand offsets and cost microseconds anywhere, so they are
//! on the host — a transfer that scales with the *record count* rather than with
//! the data, and so three orders of magnitude smaller than the one this library
//! exists to delete.
//!
//! # Where FASTQ is cheaper than BCF
//!
//! BCF sieves at every byte offset because a record can begin anywhere, then
//! runs a second validation stage over the survivors. FASTQ records begin only
//! after a newline, so the candidate set is the line starts, and the validator
//! is already the full check — there is no second stage because there is nothing
//! for one to prune. See `docs/fastq-boundaries.md`.
//!
//! # Where it is more awkward
//!
//! BCF's records carry their own length, so the host can prove a tiling from the
//! offsets alone. A FASTQ record's length is only known by walking its four
//! lines, which is the device's job — so the proof needs the decode kernel's
//! output, and decode therefore runs *before* the tiling is proved rather than
//! after. A failure discards those columns and falls back to the on-device walk.
//!
//! # The fallback stays on device
//!
//! When the tiling fails the boundaries have to come from somewhere. Copying the
//! batch back to walk it on the host would pay exactly the transfer this library
//! deletes, so a single-thread kernel walks the chain in place: slow, rare, and
//! residency-preserving.

#[cfg(not(feature = "cuda"))]
use fritillaria_core::Error;
use fritillaria_core::{DeviceInflateBatch, Result};

/// Boundary discovery and decode for FASTQ batches already on the device.
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it rather than per batch.
#[derive(Debug)]
pub struct FastqScanner {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Inner,
}

impl FastqScanner {
    /// Opens the default device on a context of our own and compiles the
    /// kernels.
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

    /// Builds a scanner on a CUDA context and stream the **caller** owns.
    ///
    /// `stream` must belong to `ctx`, and it should be the same context the
    /// codec that produced the batch allocated in — otherwise the kernels cannot
    /// read it.
    #[cfg(feature = "cuda")]
    pub fn with_context(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            inner: cuda_impl::Inner::with_context(ctx, stream)?,
        })
    }

    /// Finds every record boundary in a device-resident batch.
    ///
    /// `start` is where records begin: 0 for the first batch of a file, and the
    /// previous batch's tail after that.
    ///
    /// Does the same work as [`decode`](Self::decode) and throws the columns
    /// away, because a FASTQ record's length is only knowable by decoding it.
    /// Prefer `decode` unless the boundaries really are all you want.
    pub fn scan(
        &self,
        batch: &DeviceInflateBatch,
        start: usize,
    ) -> Result<fritillaria_fastq::columnar::SpeculativeScan> {
        #[cfg(feature = "cuda")]
        {
            self.inner.run(batch, start).map(|(scan, _)| scan)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, start);
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }

    /// Scans a batch and decodes it into device-resident columns.
    ///
    /// The returned batch **borrows** `batch` in every practical sense: its
    /// bounds are offsets into that buffer, so the inflate batch must outlive
    /// it.
    pub fn decode(
        &self,
        batch: &DeviceInflateBatch,
        start: usize,
    ) -> Result<fritillaria_fastq::columnar::DeviceRecordBatch> {
        #[cfg(feature = "cuda")]
        {
            self.inner.decode(batch, start)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, start);
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext as RawContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use fritillaria_core::{DeviceBuffer, DeviceInflateBatch, Error, Result};
    use fritillaria_fastq::columnar::{DeviceColumns, DeviceRecordBatch, Proof, SpeculativeScan};

    use crate::backend::{CudaAlloc, driver_err, load_kernel};

    const BLOCK_DIM: u32 = 256;

    /// Slots reserved for survivors, as a fraction of the batch.
    ///
    /// Illumina is the dense case at roughly one record per 216 bytes of
    /// inflated FASTQ; one slot per 64 bytes is over 3x that. Overflow is
    /// detected and reported rather than clamped — a truncated survivor list
    /// would fail the tiling and fall back silently, which performs exactly like
    /// working.
    const BYTES_PER_SLOT: usize = 64;
    const MIN_SLOTS: usize = 1024;

    pub(super) struct Inner {
        /// Held for its lifetime: the loaded kernels belong to modules in this
        /// context, so it must outlive them.
        _ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        sieve: CudaFunction,
        decode_fn: CudaFunction,
        walk: CudaFunction,
        ordinal: i32,
    }

    impl std::fmt::Debug for Inner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("FastqScanner")
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

    /// One thread. The fallback walk is a dependent chain by nature.
    const SINGLE: LaunchConfig = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    /// A scan plus the columns that produced it, if there were any records.
    ///
    /// The offsets buffer travels with the columns because it *becomes* the
    /// `record_offsets` column — it is uploaded once and handed over, not copied.
    type Run = (SpeculativeScan, Option<(CudaSlice<u8>, Decoded)>);

    /// The decode kernel's five output columns, plus the ends the proof needs.
    pub(super) struct Decoded {
        sequence_start: CudaSlice<u8>,
        plus_start: CudaSlice<u8>,
        quality_start: CudaSlice<u8>,
        sequence_len: CudaSlice<u8>,
        record_end: CudaSlice<u8>,
        /// `record_end` read back, so the host can prove the tiling.
        ends: Vec<u32>,
    }

    impl Inner {
        fn from_parts(ctx: Arc<RawContext>, stream: Arc<CudaStream>, ordinal: i32) -> Result<Self> {
            let src = crate::FASTQ_SCAN_KERNEL_SRC;
            Ok(Self {
                sieve: load_kernel(&ctx, src, "fastq_scan.cu", "fastq_sieve")?,
                decode_fn: load_kernel(&ctx, src, "fastq_scan.cu", "fastq_decode")?,
                walk: load_kernel(&ctx, src, "fastq_scan.cu", "fastq_walk")?,
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

        fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
            &self,
            n: usize,
        ) -> Result<CudaSlice<T>> {
            self.stream
                .alloc_zeros::<T>(n.max(1))
                .map_err(driver_err("allocating scan scratch"))
        }

        /// A zero-sized CUDA allocation is invalid, so every column is at least
        /// one byte; `DeviceBuffer` records the logical length separately.
        fn bytes(&self, n: usize) -> Result<CudaSlice<u8>> {
            self.stream
                .alloc_zeros::<u8>(n.max(1))
                .map_err(driver_err("allocating column"))
        }

        fn read<T: cudarc::driver::DeviceRepr + Default + Clone>(
            &self,
            slice: &CudaSlice<T>,
        ) -> Result<Vec<T>> {
            self.stream
                .clone_dtoh(slice)
                .map_err(driver_err("reading scan results"))
        }

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

        /// Resolves the CUDA allocation behind a batch, checking it is ours.
        fn alloc_of<'a>(&self, data: &'a DeviceBuffer) -> Result<&'a CudaAlloc> {
            let alloc = data
                .alloc()
                .as_any()
                .downcast_ref::<CudaAlloc>()
                .ok_or_else(|| {
                    Error::Cuda("batch was not produced by the CUDA backend".to_string())
                })?;
            if alloc.ordinal != self.ordinal {
                return Err(Error::Cuda(format!(
                    "batch is on device {} but this scanner is on {}",
                    alloc.ordinal, self.ordinal
                )));
            }
            Ok(alloc)
        }

        fn upload_offsets(&self, offsets: &[usize]) -> Result<CudaSlice<u8>> {
            // As bytes rather than u64 so the same allocation becomes the
            // `record_offsets` column; the kernel reads it as `const u64*`,
            // which is what it is.
            let mut raw = Vec::with_capacity(offsets.len() * 8);
            for &o in offsets {
                raw.extend_from_slice(&(o as u64).to_le_bytes());
            }
            self.stream
                .clone_htod(&raw)
                .map_err(driver_err("uploading record offsets"))
        }

        /// Sieve, then sort. Returns unproven candidate offsets.
        ///
        /// `None` means the survivor array overflowed and the caller must walk.
        fn sieve_sorted(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            start: usize,
            capacity: usize,
        ) -> Result<Option<(Vec<usize>, usize)>> {
            let capacity_u32 = u32::try_from(capacity)
                .map_err(|_| Error::Cuda("survivor capacity exceeds u32".to_string()))?;

            let sieved = self.zeros::<u64>(capacity)?;
            let count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;
            let anchor = self.zeros::<u64>(1)?;

            let (len_u64, start_u64) = (len as u64, start as u64);
            let mut builder = self.stream.launch_builder(&self.sieve);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(&start_u64)
                .arg(&sieved)
                .arg(&count)
                .arg(&capacity_u32)
                .arg(&overflow)
                .arg(&anchor);
            // SAFETY: the kernel signature matches. `sieved` holds `capacity`
            // u64 slots and writes past it are counted into `overflow` rather
            // than performed.
            unsafe { builder.launch(grid(len - start)) }
                .map_err(driver_err("launching fastq_sieve"))?;

            if self.read(&overflow)?[0] != 0 {
                return Ok(None);
            }

            let found = self.read(&count)?[0] as usize;
            let mut offsets: Vec<usize> = self.read(&sieved)?[..found]
                .iter()
                .map(|&o| o as usize)
                .collect();
            // The atomic append gives no ordering.
            offsets.sort_unstable();
            let anchor = self.read(&anchor)?[0] as usize;
            Ok(Some((offsets, anchor)))
        }

        /// Runs the decode kernel over `offsets`, returning the columns.
        fn decode_offsets(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            offsets: &CudaSlice<u8>,
            n: usize,
        ) -> Result<Decoded> {
            let mut sequence_start = self.bytes(n * 4)?;
            let mut plus_start = self.bytes(n * 4)?;
            let mut quality_start = self.bytes(n * 4)?;
            let mut sequence_len = self.bytes(n * 4)?;
            let mut record_end = self.bytes(n * 4)?;
            let mut errors = self.zeros::<u32>(1)?;

            let len_u64 = len as u64;
            let n_u32 = u32::try_from(n)
                .map_err(|_| Error::Cuda("record count exceeds u32".to_string()))?;

            let mut builder = self.stream.launch_builder(&self.decode_fn);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(offsets)
                .arg(&n_u32)
                .arg(&mut sequence_start)
                .arg(&mut plus_start)
                .arg(&mut quality_start)
                .arg(&mut sequence_len)
                .arg(&mut record_end)
                .arg(&mut errors);
            // SAFETY: every column holds `n` elements of the width the kernel
            // writes, and thread i writes only element i of each.
            unsafe { builder.launch(grid(n)) }.map_err(driver_err("launching fastq_decode"))?;

            let failures = self.read(&errors)?[0];
            if failures != 0 {
                // Refusing the whole batch rather than shipping columns that are
                // right for most rows.
                return Err(Error::Malformed {
                    format: "fastq",
                    position: 0,
                    reason: format!("{failures} of {n} records failed to decode"),
                });
            }

            let ends = self
                .read(&record_end)?
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect();

            Ok(Decoded {
                sequence_start,
                plus_start,
                quality_start,
                sequence_len,
                record_end,
                ends,
            })
        }

        /// Single-thread walk on device, for when the tiling fails.
        fn walk_records(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            start: usize,
            capacity: usize,
        ) -> Result<(Vec<usize>, usize)> {
            let out = self.zeros::<u64>(capacity)?;
            let count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;
            let tail = self.zeros::<u64>(1)?;

            let capacity_u32 = u32::try_from(capacity)
                .map_err(|_| Error::Cuda("walk capacity exceeds u32".to_string()))?;
            let (len_u64, start_u64) = (len as u64, start as u64);

            let mut builder = self.stream.launch_builder(&self.walk);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(&start_u64)
                .arg(&out)
                .arg(&count)
                .arg(&capacity_u32)
                .arg(&overflow)
                .arg(&tail);
            // SAFETY: single thread, every write bounded by `capacity`.
            unsafe { builder.launch(SINGLE) }.map_err(driver_err("launching fastq_walk"))?;

            if self.read(&overflow)?[0] != 0 {
                return Err(Error::Cuda(format!(
                    "fastq_walk found more than {capacity} records in one batch"
                )));
            }
            let n = self.read(&count)?[0] as usize;
            let offsets = self.read(&out)?[..n].iter().map(|&o| o as usize).collect();
            let tail = self.read(&tail)?[0] as usize;
            Ok((offsets, tail))
        }

        /// The whole pipeline: sieve, decode, prove, fall back if needed.
        ///
        /// Returns the scan and the columns it produced, so `scan` and `decode`
        /// share one implementation and cannot drift apart.
        pub(super) fn run(&self, batch: &DeviceInflateBatch, start: usize) -> Result<Run> {
            let Some(data) = batch.data() else {
                return Ok((empty(start), None));
            };
            let len = data.byte_len();
            if start >= len {
                // Nothing can begin here. Clamped so a caller feeding the tail
                // straight back to `carry_from` cannot name an offset outside
                // the batch.
                return Ok((empty(start.min(len)), None));
            }

            let alloc = self.alloc_of(data)?;
            let capacity = (len / BYTES_PER_SLOT).max(MIN_SLOTS);

            let sieved = self.sieve_sorted(alloc, len, start, capacity)?;
            let survivors = sieved.as_ref().map_or(0, |(o, _)| o.len());

            // The speculative route: decode the candidates, then check their
            // ends tile. A FASTQ record's length is not in its bytes the way a
            // BAM or BCF record's is, so the decode has to happen first.
            // `anchor` is `start` advanced past a seam newline, computed on
            // device because the host does not have the bytes. Without it a
            // batch opening on one would fail the tiling and drop to the walk.
            if let Some((offsets, anchor)) = sieved
                && !offsets.is_empty()
                && offsets[0] == anchor
            {
                let device_offsets = self.upload_offsets(&offsets)?;
                let decoded = self.decode_offsets(alloc, len, &device_offsets, offsets.len())?;

                if let Some(tail) = tiling_ends_at(anchor, len, &offsets, &decoded.ends) {
                    return Ok((
                        SpeculativeScan {
                            offsets,
                            tail,
                            candidates: 0,
                            survivors,
                            proof: Proof::Tiled,
                        },
                        Some((device_offsets, decoded)),
                    ));
                }
            }

            // Fallback: walk on device, then decode what it found.
            let (offsets, tail) = self.walk_records(alloc, len, start, capacity)?;
            let records = offsets.len();
            let columns = if records == 0 {
                None
            } else {
                let device_offsets = self.upload_offsets(&offsets)?;
                let decoded = self.decode_offsets(alloc, len, &device_offsets, records)?;
                Some((device_offsets, decoded))
            };

            Ok((
                SpeculativeScan {
                    offsets,
                    tail,
                    candidates: 0,
                    survivors,
                    proof: Proof::Walked { survivors, records },
                },
                columns,
            ))
        }

        pub(super) fn decode(
            &self,
            batch: &DeviceInflateBatch,
            start: usize,
        ) -> Result<DeviceRecordBatch> {
            let (scan, columns) = self.run(batch, start)?;
            let Some((offsets, decoded)) = columns else {
                return self.empty_batch(scan.tail);
            };
            let n = scan.offsets.len();

            let columns = DeviceColumns {
                record_offsets: self.column(offsets, n * 8)?,
                sequence_start: self.column(decoded.sequence_start, n * 4)?,
                plus_start: self.column(decoded.plus_start, n * 4)?,
                quality_start: self.column(decoded.quality_start, n * 4)?,
                sequence_len: self.column(decoded.sequence_len, n * 4)?,
                record_end: self.column(decoded.record_end, n * 4)?,
            };
            DeviceRecordBatch::new(n, scan.tail, columns)
        }

        fn empty_batch(&self, tail: usize) -> Result<DeviceRecordBatch> {
            let column = || -> Result<DeviceBuffer> {
                let slice = self.bytes(0)?;
                self.column(slice, 0)
            };
            let columns = DeviceColumns {
                record_offsets: column()?,
                sequence_start: column()?,
                plus_start: column()?,
                quality_start: column()?,
                sequence_len: column()?,
                record_end: column()?,
            };
            DeviceRecordBatch::new(0, tail, columns)
        }
    }

    /// Where the candidates tile up to, or `None` if they do not tile.
    ///
    /// Mirrors `prove_tiling` in the reference, except that the record lengths
    /// come from the decode kernel rather than being recomputed — the host does
    /// not have the bytes.
    ///
    /// The reference also refuses a tiling that leaves a whole record unclaimed.
    /// That case cannot arise here: the sieve tests *every* line start, so a
    /// record it missed would have to have failed the same validator the walk
    /// would apply. A short tiling therefore means a partial trailing record,
    /// which is exactly what the tail reports.
    fn tiling_ends_at(start: usize, len: usize, offsets: &[usize], ends: &[u32]) -> Option<usize> {
        let mut at = start;
        for (&offset, &end) in offsets.iter().zip(ends) {
            if offset != at {
                return None;
            }
            let next = at.checked_add(end as usize)?;
            if next <= at || next > len {
                return None;
            }
            at = next;
        }
        Some(at)
    }

    fn empty(start: usize) -> SpeculativeScan {
        SpeculativeScan {
            offsets: Vec::new(),
            tail: start,
            candidates: 0,
            survivors: 0,
            proof: Proof::Tiled,
        }
    }
}
