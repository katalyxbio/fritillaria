//! BCF record boundary discovery on device.
//!
//! The CPU reference is `fritillaria_bcf::speculative` and the kernels are
//! `kernels/bcf_scan.cu`; read the former for the correctness argument. This
//! module is the launcher between them, and it is where the division of labour
//! is decided.
//!
//! # What runs where, and why the proof is on the host
//!
//! The sieve touches every byte of the batch, so it must be on device — that is
//! the whole reason this exists. Everything after it operates on a few thousand
//! offsets, and proving a tiling over those costs microseconds anywhere.
//!
//! So the split is: **sieve and validate on device, sort and prove on the
//! host.** The offsets come back — a few tens of kilobytes per batch against
//! the batch's tens of megabytes — which is a transfer this project would
//! normally refuse. It is worth naming why it is different: the D2H the design
//! exists to delete is the *inflated payload*, which scales with the data. This
//! one scales with the record count and is three orders of magnitude smaller.
//!
//! It stops being right the moment a device-side columnar BCF decode exists to
//! consume the offsets in VRAM. Until then, keeping them on device would be
//! plumbing for a consumer that does not exist.
//!
//! # The fallback stays on device
//!
//! When the tiling fails, the boundaries have to come from somewhere. Copying
//! the batch back to walk it on the host would pay exactly the transfer this
//! library deletes, so instead a single-thread kernel walks the chain in place:
//! slow, rare, and residency-preserving. Nothing in real data has triggered it,
//! so it is tested directly rather than waited for.

#[cfg(not(feature = "cuda"))]
use fritillaria_core::Error;
use fritillaria_core::{DeviceInflateBatch, Result};

/// Boundary discovery for BCF batches already resident on the device.
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it rather than per batch.
#[derive(Debug)]
pub struct BcfScanner {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Inner,
}

impl BcfScanner {
    /// Opens the default device on a context of our own and compiles the
    /// kernels.
    ///
    /// A caller that already has a CUDA context should use
    /// [`with_context`](BcfScanner::with_context) instead.
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
    /// codec that produced the batch allocated in — otherwise the kernels
    /// cannot read it.
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
    /// `start` is where records begin — from the BAM/BCF header for the first
    /// batch, and from the previous batch's tail after that. `samples` and
    /// `contigs` come from the header and are what give the sieve its
    /// selectivity.
    ///
    /// The result is always what [`fritillaria_bcf::scan_records`] would
    /// produce; [`SpeculativeScan::proof`](fritillaria_bcf::SpeculativeScan)
    /// says which route produced it, and a driver should count the fallbacks
    /// rather than assume there are none.
    pub fn scan(
        &self,
        batch: &DeviceInflateBatch,
        start: usize,
        samples: u32,
        contigs: u32,
    ) -> Result<fritillaria_bcf::SpeculativeScan> {
        #[cfg(feature = "cuda")]
        {
            self.inner.scan(batch, start, samples, contigs)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, start, samples, contigs);
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
    use fritillaria_bcf::{Proof, SpeculativeScan};
    use fritillaria_core::{DeviceInflateBatch, Error, Result};

    use crate::backend::{CudaAlloc, driver_err, load_kernel};

    const BLOCK_DIM: u32 = 256;

    /// Slots reserved for survivors, as a fraction of the batch.
    ///
    /// Real files yield about one survivor per 5 KB of panel data and one per
    /// 125 bytes of sites-only data, so one slot per 64 bytes is roughly 2x the
    /// worst case measured. Overflow is detected and reported rather than
    /// clamped — a truncated survivor list would fail the tiling and fall back
    /// silently, which performs exactly like working.
    const BYTES_PER_SLOT: usize = 64;
    const MIN_SLOTS: usize = 1024;

    pub(super) struct Inner {
        /// Held for its lifetime: the loaded kernels belong to modules in this
        /// context, so it must outlive them.
        _ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        sieve: CudaFunction,
        validate: CudaFunction,
        probe_tail: CudaFunction,
        walk: CudaFunction,
        ordinal: i32,
    }

    impl std::fmt::Debug for Inner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("BcfScanner")
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

    impl Inner {
        fn from_parts(ctx: Arc<RawContext>, stream: Arc<CudaStream>, ordinal: i32) -> Result<Self> {
            let src = crate::BCF_SCAN_KERNEL_SRC;
            Ok(Self {
                sieve: load_kernel(&ctx, src, "bcf_scan.cu", "bcf_sieve")?,
                validate: load_kernel(&ctx, src, "bcf_scan.cu", "bcf_validate")?,
                probe_tail: load_kernel(&ctx, src, "bcf_scan.cu", "bcf_probe_tail")?,
                walk: load_kernel(&ctx, src, "bcf_scan.cu", "bcf_walk")?,
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

        fn read<T: cudarc::driver::DeviceRepr + Default + Clone>(
            &self,
            slice: &CudaSlice<T>,
        ) -> Result<Vec<T>> {
            self.stream
                .clone_dtoh(slice)
                .map_err(driver_err("reading scan results"))
        }

        pub(super) fn scan(
            &self,
            batch: &DeviceInflateBatch,
            start: usize,
            samples: u32,
            contigs: u32,
        ) -> Result<SpeculativeScan> {
            let Some(data) = batch.data() else {
                return Ok(empty(start));
            };
            let len = data.byte_len();
            if start >= len {
                // No record can begin here. The tail is clamped so a caller
                // feeding it straight back to `carry_from` cannot name an
                // offset outside the batch — unreachable after a successful
                // header parse, since that needs the bytes to be present, but
                // cheaper to prevent than to diagnose.
                return Ok(empty(start.min(len)));
            }

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

            let capacity = (len / BYTES_PER_SLOT).max(MIN_SLOTS);
            let capacity_u32 = u32::try_from(capacity)
                .map_err(|_| Error::Cuda("survivor capacity exceeds u32".to_string()))?;

            let sieved = self.zeros::<u64>(capacity)?;
            let sieved_ends = self.zeros::<u64>(capacity)?;
            let sieved_count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;

            let (len_u64, start_u64) = (len as u64, start as u64);
            let mut builder = self.stream.launch_builder(&self.sieve);
            builder.arg(alloc.slice());
            builder.arg(&len_u64);
            builder.arg(&start_u64);
            builder.arg(&samples);
            builder.arg(&contigs);
            builder.arg(&sieved);
            builder.arg(&sieved_ends);
            builder.arg(&sieved_count);
            builder.arg(&capacity_u32);
            builder.arg(&overflow);
            unsafe { builder.launch(grid(len - start)) }
                .map_err(driver_err("launching bcf_sieve"))?;

            let n_sieved = self.read(&sieved_count)?[0] as usize;

            // Overflow means the survivor list is truncated, so the tiling
            // would fail for a reason that has nothing to do with the data.
            // Go straight to the walk rather than let a capacity bug present
            // as a false positive.
            if self.read(&overflow)?[0] != 0 || n_sieved > capacity {
                return self.walk(alloc.slice(), len, start, n_sieved, n_sieved);
            }

            let validated = self.zeros::<u64>(capacity)?;
            let validated_ends = self.zeros::<u64>(capacity)?;
            let validated_count = self.zeros::<u32>(1)?;
            let n_sieved_u32 = u32::try_from(n_sieved).expect("bounded by capacity");
            let mut builder = self.stream.launch_builder(&self.validate);
            builder.arg(alloc.slice());
            builder.arg(&len_u64);
            builder.arg(&sieved);
            builder.arg(&sieved_ends);
            builder.arg(&n_sieved_u32);
            builder.arg(&validated);
            builder.arg(&validated_ends);
            builder.arg(&validated_count);
            unsafe { builder.launch(grid(n_sieved)) }
                .map_err(driver_err("launching bcf_validate"))?;

            let n_validated = self.read(&validated_count)?[0] as usize;
            let mut pairs: Vec<(u64, u64)> = self
                .read(&validated)?
                .into_iter()
                .zip(self.read(&validated_ends)?)
                .take(n_validated)
                .collect();
            // Both kernels append with an atomic, so the order is arbitrary.
            // Sorting a few thousand pairs is what buys a one-atomic compaction
            // instead of a two-level prefix sum on device.
            pairs.sort_unstable();

            let offsets: Vec<usize> = pairs.iter().map(|&(pos, _)| pos as usize).collect();
            let ends: Vec<usize> = pairs.iter().map(|&(_, end)| end as usize).collect();

            if let Some(at) = tiling_ends_at(start, len, &offsets, &ends) {
                // The one thing no survivor carries: whether a *whole* record
                // begins where the tiling stops. If one does, the sieve missed
                // it and adopting this would drop a record.
                if !self.whole_record_at(alloc.slice(), len, at)? {
                    return Ok(SpeculativeScan {
                        offsets,
                        tail: at,
                        sieved: n_sieved,
                        validated: n_validated,
                        proof: Proof::Tiled,
                    });
                }
            }

            self.walk(alloc.slice(), len, start, n_sieved, n_validated)
        }

        /// Whether a complete record begins at `pos`, asked of the device.
        ///
        /// One 4-byte transfer per batch, against the batch's tens of
        /// megabytes.
        fn whole_record_at(&self, buf: &CudaSlice<u8>, len: usize, pos: usize) -> Result<bool> {
            let flag = self.zeros::<u32>(1)?;
            let (len_u64, pos_u64) = (len as u64, pos as u64);
            let mut builder = self.stream.launch_builder(&self.probe_tail);
            builder.arg(buf);
            builder.arg(&len_u64);
            builder.arg(&pos_u64);
            builder.arg(&flag);
            unsafe { builder.launch(SINGLE) }.map_err(driver_err("launching bcf_probe_tail"))?;
            Ok(self.read(&flag)?[0] != 0)
        }

        /// The fallback: a single device thread following the true chain.
        fn walk(
            &self,
            buf: &CudaSlice<u8>,
            len: usize,
            start: usize,
            sieved: usize,
            validated: usize,
        ) -> Result<SpeculativeScan> {
            // A record is at least 32 bytes, so this can never truncate.
            let capacity = ((len - start) / 32 + 1).max(MIN_SLOTS);
            let capacity_u32 = u32::try_from(capacity)
                .map_err(|_| Error::Cuda("walk capacity exceeds u32".to_string()))?;

            let out = self.zeros::<u64>(capacity)?;
            let count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;
            let tail = self.zeros::<u64>(1)?;

            let (len_u64, start_u64) = (len as u64, start as u64);
            let mut builder = self.stream.launch_builder(&self.walk);
            builder.arg(buf);
            builder.arg(&len_u64);
            builder.arg(&start_u64);
            builder.arg(&out);
            builder.arg(&count);
            builder.arg(&capacity_u32);
            builder.arg(&overflow);
            builder.arg(&tail);
            unsafe { builder.launch(SINGLE) }.map_err(driver_err("launching bcf_walk"))?;

            if self.read(&overflow)?[0] != 0 {
                return Err(Error::Cuda(
                    "the fallback walk overflowed, which a 32-byte minimum record makes impossible"
                        .to_string(),
                ));
            }
            let n = self.read(&count)?[0] as usize;
            let mut offsets = self.read(&out)?;
            offsets.truncate(n);
            let tail = self.read(&tail)?[0] as usize;

            Ok(SpeculativeScan {
                offsets: offsets.iter().map(|&o| o as usize).collect(),
                tail,
                sieved,
                validated,
                proof: Proof::Walked {
                    survivors: validated,
                    records: n,
                },
            })
        }
    }

    /// Where the candidates tile up to, or `None` if they do not tile.
    ///
    /// Mirrors [`prove_tiling`] except for the final question — whether what
    /// remains is a partial record or a whole one the sieve missed — which
    /// needs a byte the survivors do not carry and is asked of the device by
    /// `whole_record_at`.
    fn tiling_ends_at(
        start: usize,
        len: usize,
        offsets: &[usize],
        ends: &[usize],
    ) -> Option<usize> {
        let mut at = start;
        for (&candidate, &end) in offsets.iter().zip(ends) {
            if candidate != at || end > len || end <= candidate {
                return None;
            }
            at = end;
        }
        Some(at)
    }

    fn empty(start: usize) -> SpeculativeScan {
        SpeculativeScan {
            offsets: Vec::new(),
            tail: start,
            sieved: 0,
            validated: 0,
            proof: Proof::Tiled,
        }
    }
}
