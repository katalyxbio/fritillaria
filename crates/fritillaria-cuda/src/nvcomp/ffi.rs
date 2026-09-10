//! Raw bindings to the parts of nvCOMP this crate uses.
//!
//! # Why dlopen instead of linking
//!
//! nvCOMP is a proprietary NVIDIA binary under an EULA, distributed separately
//! from the CUDA toolkit and absent from most images (including Colab's). Three
//! things follow, and together they settle the question:
//!
//! - **We must not redistribute it**, so it cannot be vendored.
//! - **Linking it would make the `nvcomp` feature unbuildable** on any machine
//!   without it — including the primary development machine for this project,
//!   which has no CUDA toolkit at all. `cargo build --features nvcomp` has to
//!   keep working there or the feature cannot be developed.
//! - It is exactly the discipline already applied twice: cudarc dlopens
//!   `libcuda`, and kernels are NVRTC-compiled at runtime. A build-time
//!   dependency here would be the only one in the workspace.
//!
//! The cost is that these declarations are hand-maintained against the C
//! headers and mismatches surface at runtime rather than compile time. They are
//! transcribed from **nvCOMP 5.3.0.16** (`include/nvcomp/deflate.h`,
//! `crc32.h`, `shared_types.h`); [`Nvcomp::load`] checks the runtime version and
//! refuses anything older than it was written against.
//!
//! Every struct here carries a `reserved` tail that NVIDIA requires be zeroed —
//! it is their forward-compatibility mechanism, so the constructors below zero
//! the whole struct rather than filling fields individually.

use std::ffi::{CStr, OsStr, c_char, c_int, c_void};
use std::fmt;

use fritillaria_core::{Error, Result};
use libloading::{Library, Symbol};

/// Opaque CUDA stream handle.
///
/// nvCOMP's headers spell this `cudaStream_t` (the runtime API type) while
/// cudarc hands us a `CUstream` (the driver API type). They are the same
/// pointer — the runtime's streams *are* driver streams — which is what makes
/// driver/runtime interop work at all.
pub type Stream = *mut c_void;

/// `nvcompStatus_t`. Zero is success; everything else is an error code.
pub type Status = c_int;

pub const NVCOMP_SUCCESS: Status = 0;

/// `nvcompDecompressBackend_t`.
///
/// `HARDWARE` selects Blackwell's on-die Decompression Engine. Exposed because
/// it costs nothing to pass through, but left at `DEFAULT` so nvCOMP picks —
/// there is no such hardware on any device this project has tested on.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum Backend {
    #[default]
    Default = 0,
    Hardware = 1,
    Cuda = 2,
}

/// `nvcompBatchedDeflateDecompressOpts_t` — 64 bytes, passed by value.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DeflateDecompressOpts {
    pub backend: c_int,
    pub sort_before_hw_decompress: c_int,
    pub reserved: [c_char; 56],
}

impl DeflateDecompressOpts {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend: backend as c_int,
            sort_before_hw_decompress: 0,
            reserved: [0; 56],
        }
    }
}

impl fmt::Debug for DeflateDecompressOpts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeflateDecompressOpts")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

/// `nvcompBatchedDeflateCompressOpts_t` — 64 bytes, passed by value.
///
/// The single field selects a point on nvCOMP's throughput/ratio ladder; see
/// [`DeflateAlgorithm`] for what each one means and why our default is not
/// nvCOMP's.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DeflateCompressOpts {
    pub algorithm: c_int,
    pub reserved: [c_char; 60],
}

impl DeflateCompressOpts {
    #[must_use]
    pub fn new(algorithm: DeflateAlgorithm) -> Self {
        Self {
            algorithm: algorithm as c_int,
            reserved: [0; 60],
        }
    }
}

impl fmt::Debug for DeflateCompressOpts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeflateCompressOpts")
            .field("algorithm", &self.algorithm)
            .finish_non_exhaustive()
    }
}

/// The `algorithm` field of [`DeflateCompressOpts`], with NVIDIA's own
/// descriptions from `deflate.h`.
///
/// # Why the default is [`DeflateAlgorithm::HighRatio`] and not nvCOMP's
///
/// nvCOMP defaults to `1`, and NVIDIA Parabricks' `--gpuwrite-deflate-algo`
/// defaults to `0` — entropy-only. Both optimise for throughput, which is the
/// right call for a batch aligner whose BAM is often an intermediate.
///
/// A library does not get to assume that: the file we write is somebody's
/// archive, and a measured 20–48% size penalty (see `docs/compression.md`) is
/// paid by every future reader of it. `4` is the first rung NVIDIA documents as
/// beating Zlib level 6, which is roughly where output size reaches parity with
/// the libdeflate-level-6 that htslib writes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum DeflateAlgorithm {
    /// Highest throughput, **entropy-only**. Parabricks' default, not ours.
    EntropyOnly = 0,
    /// High throughput, low ratio. nvCOMP's own default.
    LowRatio = 1,
    /// Medium; documented as beating Zlib level 1.
    MediumRatio = 2,
    /// Lower throughput, higher ratio; documented as beating Zlib level 6.
    /// Ours, and see the type's docs for why it is not nvCOMP's.
    #[default]
    HighRatio = 4,
    /// Lowest throughput, highest ratio.
    MaxRatio = 5,
}

/// nvCOMP refuses a chunk larger than this: "Chunk sizes must not exceed 65536
/// bytes. For best performance, a chunk size of 65536 bytes is recommended."
///
/// BGZF caps an uncompressed block at 64 KiB, so the format's limit and the
/// codec's are the same number and the recommended size is the one we already
/// want to write. Nothing to reconcile — but it is asserted rather than assumed,
/// because a BGZF block *may* legally carry 65536 bytes and that is exactly the
/// boundary.
pub const MAX_COMPRESS_CHUNK_BYTES: usize = 65536;

/// `nvcompAlignmentRequirements_t`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct AlignmentRequirements {
    pub input: usize,
    pub output: usize,
    pub temp: usize,
}

/// `nvcompCRC32Spec_t` — 32 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Crc32Spec {
    pub poly: u32,
    pub init: u32,
    pub ref_in: bool,
    pub ref_out: bool,
    pub xorout: u32,
    pub reserved: [c_char; 16],
}

impl Crc32Spec {
    /// nvCOMP's `nvcompCRC32` preset: CRC-32/ISO-HDLC.
    ///
    /// This is the variant gzip uses, so it is the one BGZF block trailers are
    /// written with. `0x04C11DB7` is the non-reflected form of the `0xEDB88320`
    /// in `kernels/crc32.cu`; `ref_in`/`ref_out` are what make the two agree.
    /// A test pins this against `crc32fast` rather than trusting the reading.
    #[must_use]
    pub const fn gzip() -> Self {
        Self {
            poly: 0x04C1_1DB7,
            init: 0xFFFF_FFFF,
            ref_in: true,
            ref_out: true,
            xorout: 0xFFFF_FFFF,
            reserved: [0; 16],
        }
    }
}

/// `nvcompCRC32KernelConf_t` — 32 bytes.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Crc32KernelConf {
    pub kernel_kind: c_int,
    pub bytes_per_read: i32,
    pub blocks_per_msg: i32,
    pub reserved: [c_char; 20],
}

/// `nvcompBatchedCRC32Opts_t` — 128 bytes, passed by value.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Crc32Opts {
    pub spec: Crc32Spec,
    pub kernel_conf: Crc32KernelConf,
    pub reserved: [c_char; 64],
}

impl Crc32Opts {
    #[must_use]
    pub fn gzip(kernel_conf: Crc32KernelConf) -> Self {
        Self {
            spec: Crc32Spec::gzip(),
            kernel_conf,
            reserved: [0; 64],
        }
    }
}

/// `nvcompCRC32OnlySegment` — each chunk is a complete message.
pub const CRC32_ONLY_SEGMENT: c_int = 0;

/// `nvcompProperties_t`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Properties {
    pub version: u32,
    pub cudart_version: u32,
}

/// The version these declarations were transcribed from, as nvCOMP encodes it
/// (major * 1000 + minor * 100 + patch). Older libraries are refused rather
/// than called with a struct layout they may not share.
const MIN_VERSION: u32 = 5300;

type FnGetProperties = unsafe extern "C" fn(*mut Properties) -> Status;
type FnStatusString = unsafe extern "C" fn(Status) -> *const c_char;
type FnDeflateAlignments =
    unsafe extern "C" fn(DeflateDecompressOpts, *mut AlignmentRequirements) -> Status;
type FnDeflateTempSize =
    unsafe extern "C" fn(usize, usize, DeflateDecompressOpts, *mut usize, usize) -> Status;
type FnDeflateDecompress = unsafe extern "C" fn(
    *const *const c_void, // device_compressed_chunk_ptrs
    *const usize,         // device_compressed_chunk_bytes
    *const usize,         // device_uncompressed_buffer_bytes
    *mut usize,           // device_uncompressed_chunk_bytes (out)
    usize,                // num_chunks
    *mut c_void,          // device_temp_ptr
    usize,                // temp_bytes
    *const *mut c_void,   // device_uncompressed_chunk_ptrs
    DeflateDecompressOpts,
    *mut Status, // device_statuses
    Stream,
) -> Status;
type FnDeflateCompressAlignments =
    unsafe extern "C" fn(DeflateCompressOpts, *mut AlignmentRequirements) -> Status;
type FnDeflateCompressTempSize =
    unsafe extern "C" fn(usize, usize, DeflateCompressOpts, *mut usize, usize) -> Status;
type FnDeflateMaxOutputChunkSize =
    unsafe extern "C" fn(usize, DeflateCompressOpts, *mut usize) -> Status;
type FnDeflateCompress = unsafe extern "C" fn(
    *const *const c_void, // device_uncompressed_chunk_ptrs
    *const usize,         // device_uncompressed_chunk_bytes
    usize,                // max_uncompressed_chunk_bytes
    usize,                // num_chunks
    *mut c_void,          // device_temp_ptr
    usize,                // temp_bytes
    *const *mut c_void,   // device_compressed_chunk_ptrs
    *mut usize,           // device_compressed_chunk_bytes (out)
    DeflateCompressOpts,
    *mut Status, // device_statuses
    Stream,
) -> Status;
type FnCrc32HeuristicConf =
    unsafe extern "C" fn(*const usize, usize, *mut Crc32KernelConf, usize, Stream) -> Status;
type FnCrc32 = unsafe extern "C" fn(
    *const *const c_void, // device_input_chunk_ptrs
    *const usize,         // device_input_chunk_bytes
    usize,                // num_chunks
    *mut u32,             // device_crc32_ptr
    Crc32Opts,
    c_int,       // segment_kind
    *mut Status, // device_statuses
    Stream,
) -> Status;

/// The nvCOMP entry points this crate calls, resolved at load time.
///
/// `_lib` is kept alive because every function pointer above borrows from it;
/// dropping the `Library` would unmap the code they point into.
pub struct Nvcomp {
    _lib: Library,
    pub version: u32,
    status_string: FnStatusString,
    deflate_decompress_alignments: FnDeflateAlignments,
    deflate_decompress_temp_size: FnDeflateTempSize,
    deflate_decompress: FnDeflateDecompress,
    deflate_compress_alignments: FnDeflateCompressAlignments,
    deflate_compress_temp_size: FnDeflateCompressTempSize,
    deflate_max_output_chunk_size: FnDeflateMaxOutputChunkSize,
    deflate_compress: FnDeflateCompress,
    crc32_heuristic_conf: FnCrc32HeuristicConf,
    crc32: FnCrc32,
}

impl fmt::Debug for Nvcomp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nvcomp")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

/// Candidate library names, in the order they are tried.
///
/// The versioned soname comes first: it is what a correctly installed
/// redistributable provides, and the bare `.so` is usually a developer symlink.
const CANDIDATES: &[&str] = &["libnvcomp.so.5", "libnvcomp.so"];

/// Environment variable naming an explicit library path.
///
/// Needed because nvCOMP is normally *not* on the loader path: the Colab image
/// has no nvCOMP at all, so remote jobs unpack the redistributable to a scratch
/// directory and point this at it.
pub const LIB_PATH_ENV: &str = "FRITILLARIA_NVCOMP_LIB";

/// Resolves a symbol, turning a missing one into a useful error rather than a
/// panic. A missing symbol means a version older than we support far more often
/// than it means a corrupt library, so the message says so.
fn symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: the caller names a symbol whose Rust type matches the C
    // declaration transcribed above; the returned pointer is immediately copied
    // out and the `Library` is kept alive for as long as it is used.
    let sym: Symbol<'_, T> = unsafe { lib.get(name) }.map_err(|err| {
        Error::CudaUnavailable(format!(
            "nvCOMP is missing `{}` — is it older than 5.3? ({err})",
            String::from_utf8_lossy(name).trim_end_matches('\0'),
        ))
    })?;
    Ok(*sym)
}

impl Nvcomp {
    /// Loads nvCOMP from `FRITILLARIA_NVCOMP_LIB`, or from the loader path.
    ///
    /// Returns [`Error::CudaUnavailable`] when the library is not installed,
    /// which callers should treat as "use another codec", never as a hard
    /// failure — nvCOMP being absent is the normal case, not an error.
    pub fn load() -> Result<Self> {
        let explicit = std::env::var_os(LIB_PATH_ENV);
        let mut attempts = Vec::new();

        // An explicit path is a statement of intent: if it does not load, say
        // so rather than silently falling back to some other copy.
        let lib = if let Some(path) = &explicit {
            Self::open(OsStr::new(path), &mut attempts)?
        } else {
            let mut found = None;
            for name in CANDIDATES {
                if let Ok(lib) = Self::open(OsStr::new(name), &mut attempts) {
                    found = Some(lib);
                    break;
                }
            }
            found.ok_or_else(|| {
                Error::CudaUnavailable(format!(
                    "nvCOMP not found (set {LIB_PATH_ENV} to its path): {}",
                    attempts.join("; ")
                ))
            })?
        };

        let get_properties: FnGetProperties = symbol(&lib, b"nvcompGetProperties\0")?;
        let mut properties = Properties::default();
        // SAFETY: `get_properties` came from this library under the name whose
        // C signature the type above transcribes, and the out-pointer is valid.
        let status = unsafe { get_properties(&raw mut properties) };
        if status != NVCOMP_SUCCESS {
            return Err(Error::CudaUnavailable(format!(
                "nvcompGetProperties failed with status {status}"
            )));
        }
        if properties.version < MIN_VERSION {
            return Err(Error::CudaUnavailable(format!(
                "nvCOMP {} is older than the {MIN_VERSION} these bindings target",
                properties.version
            )));
        }

        Ok(Self {
            version: properties.version,
            status_string: symbol(&lib, b"nvcompGetStatusString\0")?,
            deflate_decompress_alignments: symbol(
                &lib,
                b"nvcompBatchedDeflateDecompressGetRequiredAlignments\0",
            )?,
            deflate_decompress_temp_size: symbol(
                &lib,
                b"nvcompBatchedDeflateDecompressGetTempSizeAsync\0",
            )?,
            deflate_decompress: symbol(&lib, b"nvcompBatchedDeflateDecompressAsync\0")?,
            deflate_compress_alignments: symbol(
                &lib,
                b"nvcompBatchedDeflateCompressGetRequiredAlignments\0",
            )?,
            deflate_compress_temp_size: symbol(
                &lib,
                b"nvcompBatchedDeflateCompressGetTempSizeAsync\0",
            )?,
            deflate_max_output_chunk_size: symbol(
                &lib,
                b"nvcompBatchedDeflateCompressGetMaxOutputChunkSize\0",
            )?,
            deflate_compress: symbol(&lib, b"nvcompBatchedDeflateCompressAsync\0")?,
            crc32_heuristic_conf: symbol(&lib, b"nvcompBatchedCRC32GetHeuristicConf\0")?,
            crc32: symbol(&lib, b"nvcompBatchedCRC32Async\0")?,
            _lib: lib,
        })
    }

    fn open(name: &OsStr, attempts: &mut Vec<String>) -> Result<Library> {
        // SAFETY: dlopen runs the library's initialisers, which is inherently
        // outside Rust's model. nvCOMP is a well-behaved shared library; the
        // real precondition is that `name` names nvCOMP and not something else,
        // which is the caller's (or the operator's) responsibility.
        match unsafe { Library::new(name) } {
            Ok(lib) => Ok(lib),
            Err(err) => {
                let message = format!("{}: {err}", name.to_string_lossy());
                attempts.push(message.clone());
                Err(Error::CudaUnavailable(message))
            }
        }
    }

    /// nvCOMP's own description of a status code.
    #[must_use]
    pub fn describe(&self, status: Status) -> String {
        // SAFETY: `status_string` is nvCOMP's own; it returns a pointer to a
        // static NUL-terminated string for any input, including unknown codes.
        let ptr = unsafe { (self.status_string)(status) };
        if ptr.is_null() {
            return format!("nvcomp status {status}");
        }
        // SAFETY: non-null and NUL-terminated per the header's contract; the
        // string is static, so the borrow cannot dangle before it is copied.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// Turns a host-side status into an error, naming the call that produced it.
    pub fn check(&self, status: Status, context: &str) -> Result<()> {
        if status == NVCOMP_SUCCESS {
            return Ok(());
        }
        Err(Error::Cuda(format!(
            "{context}: {} (nvcomp status {status})",
            self.describe(status)
        )))
    }

    /// Buffer alignment nvCOMP requires for deflate decompression.
    ///
    /// Queried rather than hardcoded: the header's
    /// `nvcompDeflateRequiredDecompressionAlignment` is the *most restrictive*
    /// across input, output and temp, and the per-buffer values matter a great
    /// deal here — see [`super::NvcompContext`] on why an output alignment of 1
    /// is what makes the dense layout possible.
    pub fn deflate_decompress_alignments(
        &self,
        opts: DeflateDecompressOpts,
    ) -> Result<AlignmentRequirements> {
        let mut out = AlignmentRequirements::default();
        // SAFETY: transcribed signature; `out` is a valid, writable local.
        let status = unsafe { (self.deflate_decompress_alignments)(opts, &raw mut out) };
        self.check(status, "querying deflate alignment requirements")?;
        Ok(out)
    }

    /// Scratch bytes nvCOMP needs for a batch of this shape.
    ///
    /// Measured at zero for deflate in 5.3, but queried anyway: it is cheap,
    /// it does not touch the device, and assuming zero would be a silent
    /// out-of-bounds if a future version needed scratch.
    pub fn deflate_decompress_temp_size(
        &self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        opts: DeflateDecompressOpts,
    ) -> Result<usize> {
        let mut temp_bytes = 0usize;
        // SAFETY: transcribed signature; the out-pointer is a valid local and
        // this entry point does not touch the device.
        let status = unsafe {
            (self.deflate_decompress_temp_size)(
                num_chunks,
                max_uncompressed_chunk_bytes,
                opts,
                &raw mut temp_bytes,
                max_total_uncompressed_bytes,
            )
        };
        self.check(status, "querying deflate temp size")?;
        Ok(temp_bytes)
    }

    /// Queues batched deflate decompression on `stream`.
    ///
    /// # Safety
    ///
    /// Every pointer must be device-accessible and stay alive until the stream
    /// has caught up. `chunk_ptrs`, `chunk_bytes`, `buffer_bytes`,
    /// `out_bytes`, `out_ptrs` and `statuses` must each have `num_chunks`
    /// elements, each input chunk must meet the alignment reported by
    /// [`Nvcomp::deflate_alignments`], and each output buffer must have room
    /// for the corresponding `buffer_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn deflate_decompress(
        &self,
        chunk_ptrs: *const *const c_void,
        chunk_bytes: *const usize,
        buffer_bytes: *const usize,
        out_bytes: *mut usize,
        num_chunks: usize,
        temp: *mut c_void,
        temp_bytes: usize,
        out_ptrs: *const *mut c_void,
        opts: DeflateDecompressOpts,
        statuses: *mut Status,
        stream: Stream,
    ) -> Result<()> {
        // SAFETY: the caller guarantees the pointer/length contract above.
        let status = unsafe {
            (self.deflate_decompress)(
                chunk_ptrs,
                chunk_bytes,
                buffer_bytes,
                out_bytes,
                num_chunks,
                temp,
                temp_bytes,
                out_ptrs,
                opts,
                statuses,
                stream,
            )
        };
        self.check(status, "launching deflate decompression")
    }

    /// Buffer alignment nvCOMP requires for deflate **compression**.
    ///
    /// Queried per-options rather than shared with the decompress side: the
    /// header ties the requirement to the `compress_opts` passed, so nothing
    /// guarantees the two directions agree, and the output alignment decides
    /// whether compressed blocks can be written straight into a dense BGZF
    /// stream or have to be compacted afterwards.
    pub fn deflate_compress_alignments(
        &self,
        opts: DeflateCompressOpts,
    ) -> Result<AlignmentRequirements> {
        let mut out = AlignmentRequirements::default();
        // SAFETY: transcribed signature; `out` is a valid, writable local.
        let status = unsafe { (self.deflate_compress_alignments)(opts, &raw mut out) };
        self.check(status, "querying deflate compression alignments")?;
        Ok(out)
    }

    /// Scratch bytes nvCOMP needs to compress a batch of this shape.
    ///
    /// Unlike the decompress side — which needs none in 5.3 — compression is
    /// expected to want real scratch, so this value is load-bearing rather than
    /// a formality. Despite the `Async` in nvCOMP's name for this entry point it
    /// takes no stream and does not touch the device.
    pub fn deflate_compress_temp_size(
        &self,
        num_chunks: usize,
        max_uncompressed_chunk_bytes: usize,
        max_total_uncompressed_bytes: usize,
        opts: DeflateCompressOpts,
    ) -> Result<usize> {
        let mut temp_bytes = 0usize;
        // SAFETY: transcribed signature; the out-pointer is a valid local.
        let status = unsafe {
            (self.deflate_compress_temp_size)(
                num_chunks,
                max_uncompressed_chunk_bytes,
                opts,
                &raw mut temp_bytes,
                max_total_uncompressed_bytes,
            )
        };
        self.check(status, "querying deflate compression temp size")?;
        Ok(temp_bytes)
    }

    /// Worst-case compressed size for a chunk of `max_uncompressed_chunk_bytes`.
    ///
    /// The output side has to be preallocated at this size per chunk, because
    /// the real sizes are only known once the kernel has run and reading them
    /// back to size the allocation would mean a host synchronise mid-batch. So
    /// BGZF output is written into worst-case slots and compacted, which is the
    /// inverse of the read path — there, an output alignment of 1 let nvCOMP
    /// write straight into a dense buffer.
    ///
    /// Note this can exceed the input size: DEFLATE on incompressible data
    /// stores it verbatim plus framing, and BGZF has to cope with that anyway.
    pub fn deflate_max_output_chunk_size(
        &self,
        max_uncompressed_chunk_bytes: usize,
        opts: DeflateCompressOpts,
    ) -> Result<usize> {
        let mut max_compressed = 0usize;
        // SAFETY: transcribed signature; the out-pointer is a valid local.
        let status = unsafe {
            (self.deflate_max_output_chunk_size)(
                max_uncompressed_chunk_bytes,
                opts,
                &raw mut max_compressed,
            )
        };
        self.check(status, "querying maximum compressed chunk size")?;
        Ok(max_compressed)
    }

    /// Queues batched deflate compression on `stream`.
    ///
    /// # Safety
    ///
    /// Every pointer must be device-accessible and stay alive until the stream
    /// has caught up. `in_ptrs`, `in_bytes`, `out_ptrs`, `out_bytes` and
    /// `statuses` must each have `num_chunks` elements; no input chunk may
    /// exceed [`MAX_COMPRESS_CHUNK_BYTES`]; each input and output buffer must
    /// meet the alignment reported by [`Nvcomp::deflate_compress_alignments`];
    /// and each output buffer must have room for
    /// [`Nvcomp::deflate_max_output_chunk_size`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn deflate_compress(
        &self,
        in_ptrs: *const *const c_void,
        in_bytes: *const usize,
        max_uncompressed_chunk_bytes: usize,
        num_chunks: usize,
        temp: *mut c_void,
        temp_bytes: usize,
        out_ptrs: *const *mut c_void,
        out_bytes: *mut usize,
        opts: DeflateCompressOpts,
        statuses: *mut Status,
        stream: Stream,
    ) -> Result<()> {
        // SAFETY: the caller guarantees the pointer/length contract above.
        let status = unsafe {
            (self.deflate_compress)(
                in_ptrs,
                in_bytes,
                max_uncompressed_chunk_bytes,
                num_chunks,
                temp,
                temp_bytes,
                out_ptrs,
                out_bytes,
                opts,
                statuses,
                stream,
            )
        };
        self.check(status, "launching deflate compression")
    }

    /// Picks a CRC32 kernel configuration for a batch of this shape.
    ///
    /// Passing a null sizes array with an explicit `max_input_chunk_bytes` lets
    /// nvCOMP choose without reading the per-chunk sizes off the device, which
    /// it would otherwise have to synchronise for. BGZF caps a block at 64 KiB,
    /// so the bound is known without looking.
    ///
    /// # Safety
    ///
    /// `stream` must be a live CUDA stream. nvCOMP may queue work on it.
    pub unsafe fn crc32_heuristic_conf(
        &self,
        num_chunks: usize,
        max_input_chunk_bytes: usize,
        stream: Stream,
    ) -> Result<Crc32KernelConf> {
        let mut conf = Crc32KernelConf::default();
        // SAFETY: transcribed signature. A null sizes array is explicitly
        // permitted (`nvcompCRC32IgnoredInputChunkBytes`) when a non-zero
        // maximum chunk size is supplied, which it is.
        let status = unsafe {
            (self.crc32_heuristic_conf)(
                std::ptr::null(),
                num_chunks,
                &raw mut conf,
                max_input_chunk_bytes,
                stream,
            )
        };
        self.check(status, "choosing a CRC32 kernel configuration")?;
        Ok(conf)
    }

    /// Queues batched CRC32 on `stream`, one checksum per chunk.
    ///
    /// # Safety
    ///
    /// Every pointer must be device-accessible and stay alive until the stream
    /// has caught up. `chunk_ptrs`, `chunk_bytes`, `crcs` and `statuses` must
    /// each have `num_chunks` elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn crc32(
        &self,
        chunk_ptrs: *const *const c_void,
        chunk_bytes: *const usize,
        num_chunks: usize,
        crcs: *mut u32,
        opts: Crc32Opts,
        statuses: *mut Status,
        stream: Stream,
    ) -> Result<()> {
        // SAFETY: the caller guarantees the pointer/length contract above.
        let status = unsafe {
            (self.crc32)(
                chunk_ptrs,
                chunk_bytes,
                num_chunks,
                crcs,
                opts,
                CRC32_ONLY_SEGMENT,
                statuses,
                stream,
            )
        };
        self.check(status, "launching CRC32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layouts nvCOMP passes **by value**. A mismatch here would corrupt
    /// arguments silently rather than failing to link, so pin the sizes the
    /// headers imply.
    #[test]
    fn by_value_structs_match_the_c_layout() {
        assert_eq!(size_of::<DeflateDecompressOpts>(), 64);
        assert_eq!(size_of::<DeflateCompressOpts>(), 64);
        assert_eq!(size_of::<Crc32Spec>(), 32);
        assert_eq!(size_of::<Crc32KernelConf>(), 32);
        assert_eq!(size_of::<Crc32Opts>(), 128);
        assert_eq!(size_of::<AlignmentRequirements>(), 24);
    }

    #[test]
    fn the_crc_preset_is_the_gzip_variant() {
        let spec = Crc32Spec::gzip();
        // 0x04C11DB7 reflected is 0xEDB88320, the constant in kernels/crc32.cu.
        assert_eq!(spec.poly.reverse_bits(), 0xEDB8_8320);
        assert!(spec.ref_in && spec.ref_out);
        assert_eq!(spec.init, u32::MAX);
        assert_eq!(spec.xorout, u32::MAX);
    }

    /// Loads the real library and calls into it, **without needing a GPU**.
    ///
    /// nvCOMP links CUDA statically and these two entry points do not touch the
    /// device, so this runs on a machine with no driver at all. That makes it
    /// the only check of the hand-written struct layouts and symbol names that
    /// does not cost a rented VM — and a layout mistake here would corrupt
    /// arguments silently rather than failing to link.
    ///
    /// Skipped unless `FRITILLARIA_NVCOMP_LIB` points at a library, since
    /// nvCOMP is normally absent.
    #[test]
    fn the_real_library_agrees_with_these_declarations() {
        if std::env::var_os(LIB_PATH_ENV).is_none() {
            eprintln!("NOTE: {LIB_PATH_ENV} unset, not exercising the real nvCOMP");
            return;
        }

        let lib = Nvcomp::load().expect("loading the library named by the environment");
        assert!(lib.version >= MIN_VERSION);

        let alignments = lib
            .deflate_decompress_alignments(DeflateDecompressOpts::new(Backend::Default))
            .expect("querying alignments");
        // Sanity, not a spec: nonsense here means the struct came back wrong.
        for value in [alignments.input, alignments.output, alignments.temp] {
            assert!(
                value.is_power_of_two() && value <= 4096,
                "implausible alignment {value} — check the struct layout"
            );
        }

        let temp = lib
            .deflate_decompress_temp_size(
                1024,
                65536,
                1024 * 65536,
                DeflateDecompressOpts::new(Backend::Default),
            )
            .expect("querying temp size");
        assert!(temp < 1 << 40, "implausible temp size {temp}");

        assert!(!lib.describe(NVCOMP_SUCCESS).is_empty());
    }

    /// The same trick, applied to the compression half.
    ///
    /// All three query entry points are host-side — they only reason about
    /// sizes — so the compress FFI is checkable on a machine with no GPU, just
    /// as the decompress one is. That matters more here than there: the read
    /// path has a byte-identity oracle to catch a bad transcription downstream,
    /// and the write path has none, because two valid DEFLATE streams of the
    /// same input legitimately differ.
    ///
    /// Every algorithm is queried rather than only our default, since the
    /// header ties both the alignment and the scratch size to `compress_opts`
    /// and nothing says they are constant across the ladder.
    #[test]
    fn the_real_library_agrees_about_compression_too() {
        use DeflateAlgorithm::{EntropyOnly, HighRatio, LowRatio, MaxRatio, MediumRatio};

        const CHUNKS: usize = 1024;
        const PAYLOAD: usize = CHUNKS * MAX_COMPRESS_CHUNK_BYTES;

        if std::env::var_os(LIB_PATH_ENV).is_none() {
            eprintln!("NOTE: {LIB_PATH_ENV} unset, not exercising the real nvCOMP");
            return;
        }

        let lib = Nvcomp::load().expect("loading the library named by the environment");
        let mut previous_temp = 0usize;
        for algorithm in [EntropyOnly, LowRatio, MediumRatio, HighRatio, MaxRatio] {
            let opts = DeflateCompressOpts::new(algorithm);

            let alignments = lib
                .deflate_compress_alignments(opts)
                .unwrap_or_else(|err| panic!("alignments for {algorithm:?}: {err}"));
            for value in [alignments.input, alignments.output, alignments.temp] {
                assert!(
                    value.is_power_of_two() && value <= 4096,
                    "implausible alignment {value} for {algorithm:?} — check the struct layout"
                );
            }

            // The read path gets output alignment 1, which is what lets nvCOMP
            // inflate straight into a dense buffer. Compression does not, so
            // compressed blocks land in padded slots and the BGZF stream has to
            // be compacted out of them. Pinned because that compaction pass is
            // a design consequence, and if a future version relaxed this to 1
            // the pass could be deleted rather than silently kept.
            assert_eq!(
                alignments.output, 8,
                "compression output alignment changed for {algorithm:?}"
            );

            let temp = lib
                .deflate_compress_temp_size(CHUNKS, MAX_COMPRESS_CHUNK_BYTES, PAYLOAD, opts)
                .unwrap_or_else(|err| panic!("temp size for {algorithm:?}: {err}"));

            // Scratch is what caps a compression batch, so the shape of this
            // number is load-bearing rather than incidental. Measured 5.3.0.16:
            // exactly 0x, 5.5x, 10x, 17x and 18x the batch payload. Asserting
            // the *ordering* rather than the constants catches a scrambled
            // `algorithm` field — the one way a bad struct layout could show up
            // as plausible answers — without breaking on a legitimate retune.
            assert!(
                temp >= previous_temp,
                "scratch should not shrink as ratio rises: {algorithm:?} wants {temp}, \
                 the rung below wanted {previous_temp}"
            );
            assert!(
                temp <= 32 * PAYLOAD,
                "implausible scratch {temp} for a {PAYLOAD}-byte payload at {algorithm:?}"
            );
            previous_temp = temp;

            // The number every output allocation is sized from, so a wrong one
            // is an overflow rather than a wrong answer. It legitimately exceeds
            // the input — incompressible data is stored verbatim plus framing —
            // and measured 2.26x, which is far more than the 1.0006x DEFLATE's
            // own worst case implies. Bound loosely; the exact value is
            // recorded in docs/compression.md because it sizes every batch.
            let max_out = lib
                .deflate_max_output_chunk_size(MAX_COMPRESS_CHUNK_BYTES, opts)
                .unwrap_or_else(|err| panic!("max output size for {algorithm:?}: {err}"));
            assert!(
                (MAX_COMPRESS_CHUNK_BYTES..=4 * MAX_COMPRESS_CHUNK_BYTES).contains(&max_out),
                "implausible max compressed size {max_out} for a {MAX_COMPRESS_CHUNK_BYTES}-byte \
                 chunk at {algorithm:?}"
            );
        }

        // Entropy-only is the one rung that needs no scratch at all, which is
        // most of why it is the throughput default everywhere else.
        let none = lib
            .deflate_compress_temp_size(
                CHUNKS,
                MAX_COMPRESS_CHUNK_BYTES,
                PAYLOAD,
                DeflateCompressOpts::new(EntropyOnly),
            )
            .expect("temp size for EntropyOnly");
        assert_eq!(none, 0);
    }

    /// Scratch grows no faster than linearly in the chunk count.
    ///
    /// This is what makes a compression batch sizeable at all, and it is the
    /// property a batch sizer divides by: multiply the one-chunk cost and you
    /// get an upper bound on what `n` chunks need, so "how many blocks fit in
    /// the VRAM budget" has a safe answer. Superlinear growth would leave no
    /// safe batch size at all.
    ///
    /// About **1.11 MB per 64 KiB chunk** at our default — 17x the chunk itself,
    /// and the dominant term in the memory budget.
    ///
    /// # The exact value is device-dependent, and this test learned that the
    /// # expensive way
    ///
    /// An earlier version asserted scratch was *exactly* `n * per_chunk` at the
    /// upper rungs, because that is what nvCOMP reports on a machine with no
    /// CUDA driver — where all of this was developed. On a real L4 it reports
    /// **1,114,440** per chunk against this machine's **1,114,184**, and 7 chunks
    /// want 1,536 bytes *less* than 7x that. So the entry point does consult the
    /// device when there is one, and an equality here was pinning an
    /// environment rather than a contract.
    ///
    /// The consequence beyond this test: any per-chunk figure written down in
    /// `docs/compression.md` is *this machine's*, and a batch sizer must query
    /// the target device rather than use it. [`super::CompressBudget`] does.
    #[test]
    fn compression_scratch_is_linear_in_the_chunk_count() {
        if std::env::var_os(LIB_PATH_ENV).is_none() {
            eprintln!("NOTE: {LIB_PATH_ENV} unset, not exercising the real nvCOMP");
            return;
        }

        let lib = Nvcomp::load().expect("loading the library named by the environment");

        for algorithm in [
            DeflateAlgorithm::LowRatio,
            DeflateAlgorithm::MediumRatio,
            DeflateAlgorithm::HighRatio,
            DeflateAlgorithm::MaxRatio,
        ] {
            let opts = DeflateCompressOpts::new(algorithm);
            let temp = |chunks: usize, chunk_bytes: usize| {
                lib.deflate_compress_temp_size(chunks, chunk_bytes, chunks * chunk_bytes, opts)
                    .expect("querying compression temp size")
            };

            let per_chunk = temp(1, MAX_COMPRESS_CHUNK_BYTES);
            for chunks in [1usize, 7, 64, 1024, 16384] {
                let measured = temp(chunks, MAX_COMPRESS_CHUNK_BYTES);
                let modelled = per_chunk * chunks;

                // The property a batch sizer actually needs: multiplying the
                // one-chunk cost must never *under*-estimate, or a batch sized
                // from it would not fit. Asserted as a bound rather than an
                // equality on purpose — see the note above about the same test
                // failing on a real device after passing without one.
                assert!(
                    measured <= modelled,
                    "scratch grows faster than per-chunk at {algorithm:?}: \
                     {chunks} chunks measured {measured}, modelled {modelled}"
                );
                // And it must not over-estimate wildly either, or the sizer
                // leaves most of the card idle. Measured deviations: 8 KiB in
                // 369 MB at `LowRatio` with no device, 1536 bytes in 7.8 MB at
                // `HighRatio` on an L4 — both far inside this.
                let slack = modelled - measured;
                assert!(
                    slack * 1000 <= modelled,
                    "scratch is not near-linear in the chunk count at \
                     {algorithm:?}: {chunks} chunks measured {measured}, \
                     modelled {modelled}"
                );
            }

            // Smaller chunks must want less scratch, or sizing a batch by the
            // 64 KiB worst case would not be conservative.
            assert!(
                temp(64, MAX_COMPRESS_CHUNK_BYTES / 2) < temp(64, MAX_COMPRESS_CHUNK_BYTES),
                "scratch should fall with chunk size at {algorithm:?}"
            );
        }
    }

    /// nvCOMP's chunk ceiling and BGZF's block ceiling are the same number.
    ///
    /// Pinned because the whole "the batched API *is* the BGZF block model"
    /// argument rests on it, and a BGZF block may legally carry exactly 65536
    /// uncompressed bytes — so this is the boundary, not a comfortable margin.
    #[test]
    fn the_chunk_ceiling_matches_a_full_bgzf_block() {
        assert_eq!(MAX_COMPRESS_CHUNK_BYTES, fritillaria_core::MAX_BLOCK_SIZE);
    }

    /// Our default is deliberately not nvCOMP's, and not Parabricks'.
    #[test]
    fn the_default_algorithm_favours_ratio_over_throughput() {
        assert_eq!(DeflateAlgorithm::default(), DeflateAlgorithm::HighRatio);
        // nvCOMP's own default is 1 and Parabricks ships 0; both are below us.
        assert!(DeflateAlgorithm::default() as i32 > DeflateAlgorithm::LowRatio as i32);
        assert_eq!(
            DeflateCompressOpts::new(DeflateAlgorithm::default()).algorithm,
            4
        );
    }

    #[test]
    fn a_missing_library_is_unavailable_not_a_panic() {
        // Goes through `open` rather than setting `FRITILLARIA_NVCOMP_LIB` and
        // calling `load`. The environment is process-global while cargo runs
        // tests as threads in one process, so mutating it here raced with
        // `the_real_library_agrees_with_these_declarations` reading it — that
        // test would intermittently try to load `/nonexistent` and fail. An
        // observed flake, not a hypothetical one.
        let mut attempts = Vec::new();
        let err = Nvcomp::open(OsStr::new("/nonexistent/libnvcomp.so.5"), &mut attempts)
            .expect_err("a path that does not exist must not load");
        assert!(
            matches!(err, Error::CudaUnavailable(_)),
            "absent nvCOMP must be reported as unavailable, got {err:?}"
        );
        assert_eq!(attempts.len(), 1, "the failed path should be recorded");
    }
}
