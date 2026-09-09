//! Tab-delimited genomic text scanned into device columns.
//!
//! The CPU reference is `fritillaria_text::columnar` and the kernels are
//! `kernels/text_scan.cu`.
//!
//! # One scanner, five formats
//!
//! SAM, VCF, BED, GFF and GTF share their record framing exactly, so this is
//! the only launcher in the workspace that is not per-format. The `Dialect`
//! decides which leading byte marks a header and nothing else.
//!
//! # The host does two things, both at record scale
//!
//! The atomic append gives no ordering, so the host sorts the line offsets —
//! the same transfer every other format here already pays, scaling with
//! *records* rather than with data.
//!
//! It also runs the prefix sum over per-record field counts. That could be a
//! device scan, and on a very wide VCF it eventually should be; at record scale
//! it is microseconds and one fewer kernel. Unmeasured, like everything else in
//! this crate's timing — see `docs/` for what has actually been benchmarked
//! (decompression) and what has not (everything after it).

#[cfg(not(feature = "cuda"))]
use fritillaria_core::Error;
use fritillaria_core::Result;
use fritillaria_text::columnar::Dialect;

/// Scans tab-delimited text already resident on the device.
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it.
#[derive(Debug)]
pub struct TextScanner {
    #[cfg(feature = "cuda")]
    inner: cuda_impl::Inner,
}

impl TextScanner {
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
    #[cfg(feature = "cuda")]
    pub fn with_context(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            inner: cuda_impl::Inner::with_context(ctx, stream)?,
        })
    }

    /// Scans a device-resident buffer into columns.
    ///
    /// The returned batch holds **offsets into `batch`**, not copies, so the
    /// inflate batch must outlive it.
    pub fn scan(
        &self,
        batch: &fritillaria_core::DeviceInflateBatch,
        start: usize,
        dialect: Dialect,
    ) -> Result<fritillaria_text::columnar::DeviceRecordBatch> {
        #[cfg(feature = "cuda")]
        {
            self.inner.scan(batch, start, dialect)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, start, dialect);
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
    use fritillaria_text::columnar::{DeviceColumns, DeviceRecordBatch, Dialect};

    use crate::backend::{CudaAlloc, driver_err, load_kernel};

    const BLOCK_DIM: u32 = 256;

    /// Slots reserved for lines, as a fraction of the buffer.
    ///
    /// A SAM record is ~250 bytes, a VCF record with many samples far more, and
    /// the shortest realistic line here is a BED3 row at ~20. One slot per 8
    /// bytes is comfortably beyond any of them. Overflow is reported rather
    /// than clamped: a truncated line list would silently drop records.
    const BYTES_PER_LINE_SLOT: usize = 8;
    const MIN_SLOTS: usize = 1024;

    pub(super) struct Inner {
        _ctx: Arc<RawContext>,
        stream: Arc<CudaStream>,
        find_lines: CudaFunction,
        count_fields: CudaFunction,
        write_fields: CudaFunction,
        ordinal: i32,
    }

    impl std::fmt::Debug for Inner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TextScanner")
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

    impl Inner {
        fn from_parts(ctx: Arc<RawContext>, stream: Arc<CudaStream>, ordinal: i32) -> Result<Self> {
            let src = crate::TEXT_SCAN_KERNEL_SRC;
            Ok(Self {
                find_lines: load_kernel(&ctx, src, "text_scan.cu", "text_find_lines")?,
                count_fields: load_kernel(&ctx, src, "text_scan.cu", "text_count_fields")?,
                write_fields: load_kernel(&ctx, src, "text_scan.cu", "text_write_fields")?,
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

        fn upload(&self, values: &[u64]) -> Result<CudaSlice<u8>> {
            let mut raw = Vec::with_capacity(values.len() * 8);
            for &v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            self.stream
                .clone_htod(&raw)
                .map_err(driver_err("uploading offsets"))
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

        pub(super) fn scan(
            &self,
            batch: &DeviceInflateBatch,
            start: usize,
            dialect: Dialect,
        ) -> Result<DeviceRecordBatch> {
            let Some(data) = batch.data() else {
                return self.empty(start);
            };
            let len = data.byte_len();
            if start >= len {
                return self.empty(start.min(len));
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

            let (record_offsets, header_offsets) = self.find_lines(alloc, len, start, dialect)?;
            if record_offsets.is_empty() {
                return self.headers_only(&header_offsets, len);
            }
            self.fields(alloc, len, &record_offsets, &header_offsets)
        }

        /// Phase 1: line starts, sorted and split by kind.
        fn find_lines(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            start: usize,
            dialect: Dialect,
        ) -> Result<(Vec<u64>, Vec<u64>)> {
            let capacity = (len / BYTES_PER_LINE_SLOT).max(MIN_SLOTS);
            let capacity_u32 = u32::try_from(capacity)
                .map_err(|_| Error::Cuda("line capacity exceeds u32".to_string()))?;
            let lines = self.zeros::<u64>(capacity)?;
            let kinds = self.zeros::<u8>(capacity)?;
            let count = self.zeros::<u32>(1)?;
            let overflow = self.zeros::<u32>(1)?;

            let (len_u64, start_u64) = (len as u64, start as u64);
            let comment = u32::from(dialect.comment);
            let secondary = dialect.secondary.map_or(u32::MAX, u32::from);

            let mut builder = self.stream.launch_builder(&self.find_lines);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(&start_u64)
                .arg(&comment)
                .arg(&secondary)
                .arg(&lines)
                .arg(&kinds)
                .arg(&count)
                .arg(&capacity_u32)
                .arg(&overflow);
            // SAFETY: `lines` and `kinds` each hold `capacity` slots and writes
            // past them are counted into `overflow` rather than performed.
            unsafe { builder.launch(grid((len - start) as u64)) }
                .map_err(driver_err("launching text_find_lines"))?;

            if self.read(&overflow)?[0] != 0 {
                return Err(Error::Cuda(format!(
                    "more than {capacity} lines in one batch"
                )));
            }
            let found = self.read(&count)?[0] as usize;

            // The atomic append gives no ordering, so sort — carrying each
            // line's kind with it, since the kinds are keyed to the unsorted
            // slots.
            let raw_lines = self.read(&lines)?;
            let raw_kinds = self.read(&kinds)?;
            let mut pairs: Vec<(u64, u8)> =
                (0..found).map(|i| (raw_lines[i], raw_kinds[i])).collect();
            pairs.sort_unstable_by_key(|&(offset, _)| offset);

            let mut record_offsets = Vec::new();
            let mut header_offsets = Vec::new();
            for &(offset, kind) in &pairs {
                if kind == 1 {
                    header_offsets.push(offset);
                } else {
                    record_offsets.push(offset);
                }
            }
            Ok((record_offsets, header_offsets))
        }

        /// A file of nothing but header lines: legal, and the case that would
        /// index an empty field table if the record path ran anyway.
        fn headers_only(&self, header_offsets: &[u64], tail: usize) -> Result<DeviceRecordBatch> {
            if header_offsets.is_empty() {
                return self.empty(tail);
            }
            let n = header_offsets.len();
            let slice = self.upload(header_offsets)?;
            DeviceRecordBatch::new(
                0,
                n,
                tail,
                DeviceColumns {
                    record_offsets: self.column(self.zeros::<u8>(0)?, 0)?,
                    record_ends: self.column(self.zeros::<u8>(0)?, 0)?,
                    header_offsets: self.column(slice, n * 8)?,
                    tabs: self.column(self.zeros::<u8>(0)?, 0)?,
                    field_starts: self.column(self.zeros::<u8>(0)?, 0)?,
                },
            )
        }

        /// Phases 2 to 4: count the tabs, prefix-sum, write them.
        fn fields(
            &self,
            alloc: &CudaAlloc,
            len: usize,
            record_offsets: &[u64],
            header_offsets: &[u64],
        ) -> Result<DeviceRecordBatch> {
            let len_u64 = len as u64;
            let n = record_offsets.len();
            let n_u32 = u32::try_from(n)
                .map_err(|_| Error::Cuda("record count exceeds u32".to_string()))?;
            let offsets = self.upload(record_offsets)?;

            // Phase 2: per-record tab counts and ends.
            let counts = self.zeros::<u32>(n)?;
            let ends = self.zeros::<u64>(n)?;
            let mut builder = self.stream.launch_builder(&self.count_fields);
            builder
                .arg(alloc.slice())
                .arg(&len_u64)
                .arg(&offsets)
                .arg(&n_u32)
                .arg(&counts)
                .arg(&ends);
            // SAFETY: `counts` and `ends` each hold `n` elements and thread r
            // writes only element r.
            unsafe { builder.launch(grid(n as u64)) }
                .map_err(driver_err("launching text_count_fields"))?;

            // Phase 3: the prefix sum, on the host at record scale.
            let per_record = self.read(&counts)?;
            let mut starts = Vec::with_capacity(n + 1);
            let mut total = 0u64;
            starts.push(0u64);
            for &c in &per_record[..n] {
                total += u64::from(c);
                starts.push(total);
            }
            let starts_dev = self.upload(&starts)?;

            // Phase 4: write the tabs into their reserved spans.
            let tabs = self.zeros::<u64>(total.max(1) as usize)?;
            let mut builder = self.stream.launch_builder(&self.write_fields);
            builder
                .arg(alloc.slice())
                .arg(&offsets)
                .arg(&ends)
                .arg(&n_u32)
                .arg(&starts_dev)
                .arg(&tabs);
            // SAFETY: each thread writes only into `starts[r]..starts[r + 1]`,
            // which the prefix sum made disjoint, and the kernel stops at the
            // upper bound.
            unsafe { builder.launch(grid(n as u64)) }
                .map_err(driver_err("launching text_write_fields"))?;

            // The tail: a final line with no newline is either the last of the
            // file or one the batch cut in half, and the bytes do not say
            // which. Reported so the caller can carry it, exactly as FASTA does.
            let end_values = self.read(&ends)?;
            let tail = if end_values[n - 1] as usize >= len {
                record_offsets[n - 1] as usize
            } else {
                len
            };

            let header_len = header_offsets.len();
            let header_slice = self.upload(header_offsets)?;
            let ends_bytes = self.upload(&end_values[..n])?;
            let tabs_bytes = {
                let raw = self.read(&tabs)?;
                self.upload(&raw[..total as usize])?
            };

            let columns = DeviceColumns {
                record_offsets: self.column(offsets, n * 8)?,
                record_ends: self.column(ends_bytes, n * 8)?,
                header_offsets: self.column(header_slice, header_len * 8)?,
                tabs: self.column(tabs_bytes, total as usize * 8)?,
                field_starts: self.column(starts_dev, (n + 1) * 8)?,
            };
            DeviceRecordBatch::new(n, header_len, tail, columns)
        }

        fn empty(&self, tail: usize) -> Result<DeviceRecordBatch> {
            let column = || -> Result<DeviceBuffer> {
                let slice = self.zeros::<u8>(0)?;
                self.column(slice, 0)
            };
            DeviceRecordBatch::new(
                0,
                0,
                tail,
                DeviceColumns {
                    record_offsets: column()?,
                    record_ends: column()?,
                    header_offsets: column()?,
                    tabs: column()?,
                    field_starts: column()?,
                },
            )
        }
    }
}
