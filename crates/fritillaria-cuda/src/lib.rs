//! CUDA backend.
//!
//! Kernels ship as `.cu` **source** and are compiled by NVRTC on whichever
//! machine has the GPU. Nothing here requires `nvcc` at build time, so the
//! whole workspace builds on a machine with no CUDA toolkit — which is the
//! normal development case for this project (see CLAUDE.md).
//!
//! # Feature gate
//!
//! Without the `cuda` feature this crate still compiles; every entry point
//! reports the backend as unavailable. That keeps `cargo build` and
//! `cargo test` working locally while leaving the API shape identical.
//!
//! # What runs on the device
//!
//! - `kernels/inflate.cu` — DEFLATE inflate, one thread per BGZF block,
//!   with CRC32 folded in so verification costs no extra pass.
//! - `kernels/crc32.cu` — standalone per-block CRC32, kept for verifying
//!   already-decompressed data and as the simplest end-to-end NVRTC check.
//! - `kernels/gather.cu` — restages BGZF payloads at aligned offsets for
//!   nvCOMP. Only used by the `nvcomp` feature.
//!
//! # Two GPU codecs
//!
//! [`CudaCodec`] runs our own inflate kernel; [`NvcompCodec`] runs NVIDIA's.
//! They implement the same two traits with the same mandatory verification, so
//! choosing between them is a performance decision. nvCOMP is the intended fast
//! path and ours is the portable fallback — see [`nvcomp`] and CLAUDE.md.

#[cfg(feature = "cuda")]
mod backend;

pub mod bam;

#[cfg(feature = "nvcomp")]
pub mod nvcomp;

#[cfg(feature = "cuda")]
pub use backend::{CudaAlloc, CudaContext, InflateTimings};
pub use bam::{BamDecoder, DecodeTimings};

#[cfg(feature = "nvcomp")]
pub use nvcomp::{NvcompCodec, NvcompContext};

/// The cudarc this crate links, re-exported.
///
/// [`CudaCodec::with_context`] takes cudarc types, so a caller embedding this
/// library must build them against the same version. Going through this
/// re-export makes that automatic instead of a version-mismatch error that
/// reads as an unrelated type error.
#[cfg(feature = "cuda")]
pub use cudarc;

#[cfg_attr(feature = "cuda", allow(unused_imports))]
use fritillaria_core::{
    BlockCodec, BlockSpan, DeviceBlockCodec, DeviceInflateBatch, Error, InflateBatch, Result,
};

/// Source of the CRC32 kernel, compiled at runtime by NVRTC.
pub const CRC32_KERNEL_SRC: &str = include_str!("../kernels/crc32.cu");

/// Source of the inflate kernel, compiled at runtime by NVRTC.
pub const INFLATE_KERNEL_SRC: &str = include_str!("../kernels/inflate.cu");

/// Source of the BAM boundary-scan and columnar decode kernels, compiled at
/// runtime by NVRTC.
///
/// The CPU reference for these is `fritillaria_bam::blocked`; see [`bam`].
pub const BAM_DECODE_KERNEL_SRC: &str = include_str!("../kernels/bam_decode.cu");

/// Source of the payload-restaging kernel, compiled at runtime by NVRTC.
///
/// Only the nvCOMP path needs this; see [`nvcomp`] for why BGZF payloads have
/// to be moved before nvCOMP will read them.
pub const GATHER_KERNEL_SRC: &str = include_str!("../kernels/gather.cu");

/// Whether this build can use a GPU at all.
///
/// `false` when the `cuda` feature is off. A `true` here does not promise a
/// device is present — see [`device_is_available`].
#[must_use]
pub const fn is_compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

/// Whether a usable CUDA device is actually present.
///
/// Always `false` without the `cuda` feature. With it, probes the driver once
/// and caches the answer.
#[must_use]
pub fn device_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        backend::driver_is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// The CUDA implementation of [`BlockCodec`].
///
/// Owns a device context, so constructing one compiles the kernels — build it
/// once and reuse it rather than per batch.
#[derive(Debug)]
pub struct CudaCodec {
    #[cfg(feature = "cuda")]
    ctx: CudaContext,
}

impl CudaCodec {
    /// Opens the default device on a context of our own and compiles the
    /// kernels.
    ///
    /// Convenience for standalone use, benchmarks and tests. **A caller that
    /// already has a CUDA context should use
    /// [`with_context`](CudaCodec::with_context)** — allocating in a context of
    /// our own would make every buffer we produce unusable to their kernels
    /// without a peer copy.
    ///
    /// Returns [`Error::CudaUnavailable`] when this build has no CUDA support
    /// or no device is present. Callers should treat that as "use the CPU
    /// codec", never as "silently produce nothing".
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                ctx: CudaContext::new(0)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }

    /// Builds a codec on a CUDA context and stream the **caller** owns.
    ///
    /// This is the entry point that makes the library embeddable in another
    /// GPU tool: output is allocated in the caller's context, so their kernels
    /// can read it directly. `stream` must belong to `ctx`.
    ///
    /// The cudarc types here are re-exported as [`crate::cudarc`], so a caller
    /// can construct them against exactly the version this crate links.
    #[cfg(feature = "cuda")]
    pub fn with_context(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            ctx: CudaContext::with_context(ctx, stream)?,
        })
    }

    /// The event marking completion of the inflate that produced `batch`.
    ///
    /// A consumer on its own stream orders against this instead of
    /// synchronising the host. Returns `None` for an empty batch, or one not
    /// produced by this backend.
    ///
    /// ```ignore
    /// my_stream.wait(CudaCodec::ready_event(&batch).unwrap())?;
    /// launch_my_kernel(&batch, &my_stream);
    /// ```
    #[cfg(feature = "cuda")]
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
}

impl BlockCodec for CudaCodec {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.ctx.inflate_batch(batch, spans, out)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, spans, out);
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }
}

/// The CUDA implementation of [`DeviceBlockCodec`].
///
/// Same kernel and the same mandatory verification as the [`BlockCodec`] impl
/// above — the two differ only in whether the inflated payload is copied back
/// to the host. For a consumer that is another GPU kernel it should not be, and
/// that copy is 54% of measured pipeline runtime.
impl DeviceBlockCodec for CudaCodec {
    fn name(&self) -> &'static str {
        "cuda-device"
    }

    fn device_ordinal(&self) -> i32 {
        #[cfg(feature = "cuda")]
        {
            self.ctx.device_ordinal()
        }
        // Nothing was allocated, so no ordinal is meaningful; every call that
        // could produce a buffer fails anyway.
        #[cfg(not(feature = "cuda"))]
        {
            -1
        }
    }

    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.ctx.inflate_batch_device(batch, spans, out)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, spans, out);
            Err(Error::CudaUnavailable(
                "built without the `cuda` feature".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_embedded() {
        assert!(CRC32_KERNEL_SRC.contains("crc32_blocks"));
        assert!(INFLATE_KERNEL_SRC.contains("inflate_blocks"));
        assert!(GATHER_KERNEL_SRC.contains("gather_payloads"));
        for name in [
            "bam_scan_blocks",
            "bam_reconcile",
            "bam_emit_offsets",
            "bam_decode_fields",
        ] {
            assert!(
                BAM_DECODE_KERNEL_SRC.contains(name),
                "{name} must be present for the launcher to find it"
            );
        }
        for src in [
            CRC32_KERNEL_SRC,
            INFLATE_KERNEL_SRC,
            GATHER_KERNEL_SRC,
            BAM_DECODE_KERNEL_SRC,
        ] {
            assert!(
                src.contains("extern \"C\""),
                "NVRTC needs C linkage to look the kernel up by name"
            );
        }
    }

    #[test]
    fn availability_matches_the_feature() {
        assert_eq!(is_compiled_with_cuda(), cfg!(feature = "cuda"));
        if !is_compiled_with_cuda() {
            assert!(!device_is_available());
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn construction_fails_without_the_feature() {
        assert!(matches!(
            CudaCodec::new().unwrap_err(),
            Error::CudaUnavailable(_)
        ));
    }
}
