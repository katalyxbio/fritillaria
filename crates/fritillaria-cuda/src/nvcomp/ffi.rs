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
    deflate_alignments: FnDeflateAlignments,
    deflate_temp_size: FnDeflateTempSize,
    deflate_decompress: FnDeflateDecompress,
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
            deflate_alignments: symbol(
                &lib,
                b"nvcompBatchedDeflateDecompressGetRequiredAlignments\0",
            )?,
            deflate_temp_size: symbol(&lib, b"nvcompBatchedDeflateDecompressGetTempSizeAsync\0")?,
            deflate_decompress: symbol(&lib, b"nvcompBatchedDeflateDecompressAsync\0")?,
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
    pub fn deflate_alignments(&self, opts: DeflateDecompressOpts) -> Result<AlignmentRequirements> {
        let mut out = AlignmentRequirements::default();
        // SAFETY: transcribed signature; `out` is a valid, writable local.
        let status = unsafe { (self.deflate_alignments)(opts, &raw mut out) };
        self.check(status, "querying deflate alignment requirements")?;
        Ok(out)
    }

    /// Scratch bytes nvCOMP needs for a batch of this shape.
    ///
    /// Measured at zero for deflate in 5.3, but queried anyway: it is cheap,
    /// it does not touch the device, and assuming zero would be a silent
    /// out-of-bounds if a future version needed scratch.
    pub fn deflate_temp_size(
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
            (self.deflate_temp_size)(
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
            .deflate_alignments(DeflateDecompressOpts::new(Backend::Default))
            .expect("querying alignments");
        // Sanity, not a spec: nonsense here means the struct came back wrong.
        for value in [alignments.input, alignments.output, alignments.temp] {
            assert!(
                value.is_power_of_two() && value <= 4096,
                "implausible alignment {value} — check the struct layout"
            );
        }

        let temp = lib
            .deflate_temp_size(
                1024,
                65536,
                1024 * 65536,
                DeflateDecompressOpts::new(Backend::Default),
            )
            .expect("querying temp size");
        assert!(temp < 1 << 40, "implausible temp size {temp}");

        assert!(!lib.describe(NVCOMP_SUCCESS).is_empty());
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
