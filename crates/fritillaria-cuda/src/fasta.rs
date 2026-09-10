//! A reference genome compacted into device memory.
//!
//! The CPU reference is `fritillaria_fasta::columnar` and the kernels are
//! `kernels/fasta_compact.cu`.
//!
//! # Different from the other three, on purpose
//!
//! [`BamDecoder`](crate::BamDecoder), [`BcfScanner`](crate::BcfScanner) and
//! [`FastqScanner`](crate::FastqScanner) all emit *offsets into* the inflated
//! buffer, because their payloads are the bulk of the file and duplicating them
//! would double VRAM for nothing.
//!
//! FASTA is the case where that does not work. Its sequence is wrapped across
//! lines, so an offset is not enough — a consumer reading 150 bases would hit a
//! newline every 70. So this produces an owned, compacted copy, and the source
//! buffer can be dropped afterwards, which is *not* true of the other three.
//!
//! # Contigs, not batches
//!
//! A reference is read once at startup and kept, so there is no batching here
//! and no carry. The unit is the whole file.

#[cfg(not(feature = "cuda"))]
use fritillaria_core::Error;
use fritillaria_core::{DeviceInflateBatch, Result};

/// Compacts a wrapped FASTA into a contiguous device-resident reference.
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it.
#[derive(Debug)]
pub struct FastaCompactor {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Inner,
}

impl FastaCompactor {
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

    /// Builds a compactor on a CUDA context and stream the **caller** owns.
    #[cfg(feature = "cuda")]
    pub fn with_context(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            inner: cuda_impl::Inner::with_context(ctx, stream)?,
        })
    }

    /// Compacts every contig in `batch` into one contiguous device buffer.
    ///
    /// `batch` must hold the **whole** FASTA: a reference is read once and
    /// kept, and a FASTA record does not announce its length, so a partial
    /// buffer would silently truncate its last contig. See
    /// `fritillaria_fasta::columnar::scan_records`.
    ///
    /// The returned reference **owns** its bases, so `batch` may be dropped.
    pub fn compact(
        &self,
        batch: &DeviceInflateBatch,
    ) -> Result<fritillaria_fasta::columnar::DeviceReference> {
        #[cfg(feature = "cuda")]
        {
            self.inner.compact(batch)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = batch;
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
    use fritillaria_fasta::columnar::{DeviceColumns, DeviceReference, RecordBatch};

    use crate::backend::{CudaAlloc, driver_err, load_kernel};

    const BLOCK_DIM: u32 = 256;

    /// A reference has thousands of contigs at most; this is far beyond any
    /// real assembly and overflow is reported rather than clamped.
    const MAX_CONTIGS: usize = 1 << 20;

    pub(super) struct Inner {
        _ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        find: CudaFunction,
        compact_uniform: CudaFunction,
        compact_scan: CudaFunction,
        ordinal: i32,
    }

    impl std::fmt::Debug for Inner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("FastaCompactor")
                .field("ordinal", &self.ordinal)
                .finish_non_exhaustive()
        }
    }

    fn grid(n: u64) -> LaunchConfig {
        let blocks = n.div_ceil(u64::from(BLOCK_DIM)).max(1);
        LaunchConfig {
            grid_dim: (u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    const SINGLE: LaunchConfig = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    impl Inner {
        fn from_parts(ctx: Arc<RawContext>, stream: Arc<CudaStream>, ordinal: i32) -> Result<Self> {
            let src = crate::FASTA_COMPACT_KERNEL_SRC;
            Ok(Self {
                find: load_kernel(&ctx, src, "fasta_compact.cu", "fasta_find_contigs")?,
                compact_uniform: load_kernel(
                    &ctx,
                    src,
                    "fasta_compact.cu",
                    "fasta_compact_uniform",
                )?,
                compact_scan: load_kernel(&ctx, src, "fasta_compact.cu", "fasta_compact_scan")?,
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
                .map_err(driver_err("allocating scratch"))
        }

        fn read<T: cudarc::driver::DeviceRepr + Default + Clone>(
            &self,
            slice: &CudaSlice<T>,
        ) -> Result<Vec<T>> {
            self.stream
                .clone_dtoh(slice)
                .map_err(driver_err("reading results"))
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

        fn upload_u64(&self, values: &[u64]) -> Result<CudaSlice<u8>> {
            let mut raw = Vec::with_capacity(values.len() * 8);
            for &v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            self.stream
                .clone_htod(&raw)
                .map_err(driver_err("uploading contig index"))
        }

        pub(super) fn compact(&self, batch: &DeviceInflateBatch) -> Result<DeviceReference> {
            let Some(data) = batch.data() else {
                return self.empty();
            };
            let len = data.byte_len();
            if len == 0 {
                return self.empty();
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
                    "batch is on device {} but this compactor is on {}",
                    alloc.ordinal, self.ordinal
                )));
            }

            // Contig starts, on device. Tiny result; the host sorts it.
            let starts = self.find_contigs(alloc, len)?;
            if starts.is_empty() {
                return self.empty();
            }

            // The per-contig layout comes from the host reference, over the
            // *definition lines only* — a few hundred bytes per contig, not the
            // genome. Measuring line widths on device would need a second
            // kernel to save a transfer that is already negligible.
            let host = self.layout(alloc, len, &starts)?;

            let total = host.total_bases();
            let mut sequence = self
                .stream
                .alloc_zeros::<u8>(total.max(1) as usize)
                .map_err(driver_err("allocating the compacted reference"))?;

            for (i, &record_start) in starts.iter().enumerate() {
                let bounds = host.bounds()[i];
                let span = bounds.sequence_span();
                if span == 0 {
                    continue;
                }
                let src_at = record_start + u64::from(bounds.sequence_start);
                let dst_at = host.sequence_offsets()[i];

                let src = alloc
                    .slice()
                    .slice(src_at as usize..(src_at + span) as usize);
                let mut dst =
                    sequence.slice_mut(dst_at as usize..(dst_at + bounds.sequence_len) as usize);

                if bounds.uniform {
                    let mut builder = self.stream.launch_builder(&self.compact_uniform);
                    let width = bounds.line_width;
                    builder.arg(&src).arg(&span).arg(&mut dst).arg(&width);
                    // SAFETY: `dst` is exactly `sequence_len` bytes and every
                    // destination is `i - i/line_width < sequence_len` for a
                    // non-newline byte of a uniformly wrapped contig.
                    unsafe { builder.launch(grid(span)) }
                        .map_err(driver_err("launching fasta_compact_uniform"))?;
                } else {
                    let mut builder = self.stream.launch_builder(&self.compact_scan);
                    builder.arg(&src).arg(&span).arg(&mut dst);
                    // SAFETY: single thread, and it writes exactly one byte per
                    // non-newline source byte, which is `sequence_len` of them.
                    unsafe { builder.launch(SINGLE) }
                        .map_err(driver_err("launching fasta_compact_scan"))?;
                }
            }

            let lengths: Vec<u64> = host.sequence_lengths();
            let columns = DeviceColumns {
                sequence: self.column(sequence, total as usize)?,
                sequence_offsets: {
                    let slice = self.upload_u64(host.sequence_offsets())?;
                    self.column(slice, host.len() * 8)?
                },
                sequence_lengths: {
                    let slice = self.upload_u64(&lengths)?;
                    self.column(slice, host.len() * 8)?
                },
                record_offsets: {
                    let slice = self.upload_u64(&starts)?;
                    self.column(slice, host.len() * 8)?
                },
            };
            DeviceReference::new(host.len(), total, columns)
        }

        /// Finds every `>` at a line start, sorted.
        fn find_contigs(&self, alloc: &CudaAlloc, len: usize) -> Result<Vec<u64>> {
            let out = self.zeros::<u64>(MAX_CONTIGS)?;
            let count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;
            let capacity = u32::try_from(MAX_CONTIGS).expect("MAX_CONTIGS fits u32");
            let (len_u64, start) = (len as u64, 0u64);

            let mut builder = self.stream.launch_builder(&self.find);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(&start)
                .arg(&out)
                .arg(&count)
                .arg(&capacity)
                .arg(&overflow);
            // SAFETY: `out` holds `capacity` slots and writes past it are
            // counted into `overflow` rather than performed.
            unsafe { builder.launch(grid(len as u64)) }
                .map_err(driver_err("launching fasta_find_contigs"))?;

            if self.read(&overflow)?[0] != 0 {
                return Err(Error::Cuda(format!(
                    "FASTA has more than {MAX_CONTIGS} contigs"
                )));
            }
            let n = self.read(&count)?[0] as usize;
            let mut starts = self.read(&out)?[..n].to_vec();
            starts.sort_unstable(); // the atomic append gives no ordering
            Ok(starts)
        }

        /// Measures each contig's layout on the host.
        ///
        /// Downloads the whole buffer, which for a reference genome is the one
        /// transfer this type cannot avoid — the line widths are a property of
        /// the source text. It is a *read* of data already being read, once, at
        /// startup, and the result is the compacted reference staying resident.
        fn layout(&self, alloc: &CudaAlloc, len: usize, starts: &[u64]) -> Result<RecordBatch> {
            let bytes = self
                .stream
                .clone_dtoh(alloc.slice())
                .map_err(driver_err("downloading the FASTA for layout"))?;
            let mut batch = RecordBatch::new();
            batch.decode(&bytes[..len], starts[0] as usize, true)?;
            Ok(batch)
        }

        fn empty(&self) -> Result<DeviceReference> {
            let column = || -> Result<DeviceBuffer> {
                let slice = self.zeros::<u8>(0)?;
                self.column(slice, 0)
            };
            let columns = DeviceColumns {
                sequence: column()?,
                sequence_offsets: column()?,
                sequence_lengths: column()?,
                record_offsets: column()?,
            };
            DeviceReference::new(0, 0, columns)
        }
    }
}
