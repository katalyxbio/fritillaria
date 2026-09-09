//! GPU-accelerated compression and parsing of gzipped genomic files.
//!
//! The facade over the workspace. See CLAUDE.md for the architecture; the short
//! version is that BGZF is a sequence of *independent* gzip members, which is
//! what makes block-parallel decompression possible.
//!
//! # Which inputs get the fast path
//!
//! This matters more than any benchmark number:
//!
//! | Input | Container | Path |
//! |---|---|---|
//! | BAM, BCF, `bgzip`ped VCF/FASTQ/text | BGZF | block-parallel (the point of this library) |
//! | plain `.fastq.gz` | one DEFLATE stream | **sequential** — cannot be split |
//! | SAM, FASTA, BED, GFF, GTF | text | CPU |
//! | CRAM | its own | CPU |
//!
//! An ordinary `gzip` file is one DEFLATE stream with a 32 KiB sliding window,
//! so back-references make it inherently serial. [`select_codec`] will not
//! pretend otherwise, and the API surfaces the distinction rather than quietly
//! degrading to CPU speed.
//!
//! The dividing line is the **container, not the format**: anything inside BGZF
//! gets the GPU path, because block discovery, parallel inflate, CRC
//! verification and virtual offsets are all container-level.
//!
//! # The format crates
//!
//! Every format noodles covers is here, because the CPU implementations *are*
//! noodles, vendored and renamed (MIT, © 2018 Michael Macias — see
//! `VENDORED.md`). Migrating is a rename:
//!
//! ```ignore
//! use noodles_bam as bam;        // before
//! use fritillaria::bam;          // after
//! ```
//!
//! The GPU work is additive on top of that API, and lives in the `columnar`
//! module of the crates that have one — [`bam::columnar`], [`bcf::columnar`] —
//! plus the codec-driven readers in [`bgzf`]. Nothing in the vendored API
//! changed, which is what makes the migration a rename rather than a rewrite.
//!
//! # Backend selection
//!
//! ```
//! use fritillaria::{Backend, select_codec};
//!
//! // Falls back to the CPU reference when no GPU is available, and says so.
//! let (codec, backend) = select_codec(Backend::Auto)?;
//! assert_eq!(codec.name(), backend.name());
//! # Ok::<(), fritillaria::Error>(())
//! ```
//!
//! Asking for [`Backend::Cuda`] or [`Backend::Nvcomp`] explicitly is an
//! assertion, not a preference: it errors when that backend is unusable rather
//! than quietly returning a CPU codec.
//!
//! `Auto` prefers nvCOMP, then our own CUDA kernel, then the CPU reference.

// Always present: the container, the codec seam, and the GPU backend are what
// this crate is, not formats it re-exports.
pub use fritillaria_bgzf as bgzf;
pub use fritillaria_core as core;
pub use fritillaria_cuda as cuda;
/// The shared line and field scanner behind SAM, VCF, BED, GFF and GTF.
pub use fritillaria_text as text;

// BGZF-contained: these get the GPU path.
#[cfg(feature = "bam")]
#[doc(inline)]
pub use fritillaria_bam as bam;
#[cfg(feature = "bcf")]
#[doc(inline)]
pub use fritillaria_bcf as bcf;
#[cfg(feature = "csi")]
#[doc(inline)]
pub use fritillaria_csi as csi;
#[cfg(feature = "tabix")]
#[doc(inline)]
pub use fritillaria_tabix as tabix;

// Data models and text formats: CPU, and labelled as such.
#[cfg(feature = "bed")]
#[doc(inline)]
pub use fritillaria_bed as bed;
#[cfg(feature = "cram")]
#[doc(inline)]
pub use fritillaria_cram as cram;
#[cfg(feature = "fasta")]
#[doc(inline)]
pub use fritillaria_fasta as fasta;
#[cfg(feature = "fastq")]
#[doc(inline)]
pub use fritillaria_fastq as fastq;
#[cfg(feature = "gff")]
#[doc(inline)]
pub use fritillaria_gff as gff;
#[cfg(feature = "gtf")]
#[doc(inline)]
pub use fritillaria_gtf as gtf;
#[cfg(feature = "sam")]
#[doc(inline)]
pub use fritillaria_sam as sam;
#[cfg(feature = "util")]
#[doc(inline)]
pub use fritillaria_util as util;
#[cfg(feature = "vcf")]
#[doc(inline)]
pub use fritillaria_vcf as vcf;

// Remote access. Off by default: these pull in a TLS stack.
#[cfg(feature = "htsget")]
#[doc(inline)]
pub use fritillaria_htsget as htsget;
#[cfg(feature = "refget")]
#[doc(inline)]
pub use fritillaria_refget as refget;

pub use fritillaria_bgzf::{CpuCodec, CpuCompressor};
pub use fritillaria_core::{
    BlockCodec, BlockCompressor, BlockSpan, CompressedBatch, Error, InflateBatch,
    MAX_COMPRESSIBLE_PAYLOAD, Result, VirtualOffset,
};

/// Which decompression backend to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Use the GPU when one is usable, otherwise the CPU reference.
    #[default]
    Auto,
    /// Force the CPU reference. Always available; the correctness oracle.
    Cpu,
    /// Force CUDA with our own inflate kernel. Errors rather than falling back.
    Cuda,
    /// Force NVIDIA's nvCOMP. Errors rather than falling back.
    ///
    /// Needs the `nvcomp` feature *and* the library installed at runtime; it is
    /// dlopened, not linked, so a build with the feature on still runs where
    /// nvCOMP is absent.
    Nvcomp,
}

impl Backend {
    /// The backend name as a codec reports it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Cuda => "cuda",
            Self::Nvcomp => "nvcomp",
            Self::Auto => {
                if nvcomp_is_usable() {
                    "nvcomp"
                } else if cuda_is_usable() {
                    "cuda"
                } else {
                    "cpu-reference"
                }
            }
        }
    }
}

/// Whether a CUDA codec could actually run here.
///
/// False without the `cuda` feature, and false when the feature is on but no
/// usable device is present.
#[must_use]
pub fn cuda_is_usable() -> bool {
    fritillaria_cuda::device_is_available()
}

/// Whether nvCOMP could actually run here.
///
/// False without the `nvcomp` feature, without a device, or when the library is
/// not installed — it is dlopened at runtime, so the last case is a normal
/// outcome rather than a build error.
///
/// This probe **constructs a codec**, which loads the library and compiles a
/// kernel. Call it once and keep the answer; `select_codec` already does.
#[must_use]
pub fn nvcomp_is_usable() -> bool {
    #[cfg(feature = "nvcomp")]
    {
        fritillaria_cuda::NvcompCodec::new().is_ok()
    }
    #[cfg(not(feature = "nvcomp"))]
    {
        false
    }
}

/// Builds an nvCOMP codec, or explains why it is unavailable.
fn nvcomp_codec() -> Result<Box<dyn BlockCodec>> {
    #[cfg(feature = "nvcomp")]
    {
        Ok(Box::new(fritillaria_cuda::NvcompCodec::new()?))
    }
    #[cfg(not(feature = "nvcomp"))]
    {
        Err(Error::CudaUnavailable(
            "built without the `nvcomp` feature".to_string(),
        ))
    }
}

/// Picks a codec, returning it alongside the backend actually chosen.
///
/// Returning the resolved backend is the point: a caller that asked for `Auto`
/// must be able to find out whether it got a GPU, rather than discovering the
/// answer from a disappointing benchmark.
///
/// `Auto` falls back to the CPU reference and says so. `Cuda` is an assertion,
/// not a preference: it errors rather than falling back, because a silent
/// downgrade to CPU speed is the most misleading thing this library could do.
pub fn select_codec(backend: Backend) -> Result<(Box<dyn BlockCodec>, Backend)> {
    match backend {
        Backend::Cpu => Ok((Box::new(CpuCodec::new()), Backend::Cpu)),
        Backend::Cuda => Ok((Box::new(fritillaria_cuda::CudaCodec::new()?), Backend::Cuda)),
        Backend::Nvcomp => Ok((nvcomp_codec()?, Backend::Nvcomp)),
        // nvCOMP first: it is the faster of the two GPU codecs and the one an
        // adopter expects to see. Ours is the portable fallback for machines
        // that do not have it. See CLAUDE.md, *Decisions made*.
        Backend::Auto => {
            if let Ok(codec) = nvcomp_codec() {
                return Ok((codec, Backend::Nvcomp));
            }
            match fritillaria_cuda::CudaCodec::new() {
                Ok(codec) => Ok((Box::new(codec), Backend::Cuda)),
                Err(_) => Ok((Box::new(CpuCodec::new()), Backend::Cpu)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_to_a_working_backend() {
        let (codec, backend) = select_codec(Backend::Auto).unwrap();
        assert_eq!(codec.name(), backend.name());
        assert_ne!(
            backend,
            Backend::Auto,
            "Auto must resolve to a concrete backend"
        );
    }

    #[test]
    fn cpu_backend_is_always_available() {
        let (codec, backend) = select_codec(Backend::Cpu).unwrap();
        assert_eq!(backend, Backend::Cpu);
        assert_eq!(codec.name(), "cpu-reference");
    }

    #[test]
    fn explicit_nvcomp_never_silently_falls_back() {
        // Same guarantee as CUDA below, and it matters more here: nvCOMP can be
        // missing for a reason that has nothing to do with the GPU (the library
        // simply is not installed), so the tempting failure mode is to shrug
        // and hand back something slower under the same name.
        match select_codec(Backend::Nvcomp) {
            Ok((codec, backend)) => {
                assert_eq!(backend, Backend::Nvcomp);
                assert_eq!(codec.name(), "nvcomp");
            }
            Err(err) => assert!(
                matches!(err, Error::CudaUnavailable(_)),
                "expected an availability error, got {err:?}"
            ),
        }
    }

    #[test]
    fn explicit_cuda_never_silently_falls_back() {
        // Asking for CUDA and quietly getting CPU is the failure mode this
        // guards against: either a real CUDA codec, or an error. Never a
        // CPU codec wearing the wrong label.
        match select_codec(Backend::Cuda) {
            Ok((codec, backend)) => {
                assert_eq!(backend, Backend::Cuda);
                assert_eq!(codec.name(), "cuda");
            }
            Err(err) => assert!(
                matches!(err, Error::CudaUnavailable(_)),
                "expected an availability error, got {err:?}"
            ),
        }
    }

    #[test]
    fn auto_matches_the_probe() {
        let (_, backend) = select_codec(Backend::Auto).unwrap();
        let expected = if nvcomp_is_usable() {
            Backend::Nvcomp
        } else if cuda_is_usable() {
            Backend::Cuda
        } else {
            Backend::Cpu
        };
        assert_eq!(backend, expected);
    }
}
