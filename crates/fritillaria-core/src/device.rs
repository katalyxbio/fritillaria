//! Device-resident output: the vocabulary, with no GPU dependency.
//!
//! This module exists so that inflated bytes never have to come back to the
//! host. The measured cost of returning them is 54% of pipeline runtime, and
//! for the use case this library targets — feeding another GPU kernel — every
//! bit of it is waste.
//!
//! # How this stays CUDA-free
//!
//! Core names a *trait*, [`DeviceAlloc`]; `fritillaria-cuda` implements it. A
//! [`DeviceBuffer`] is an opaque owning handle that format crates can hold,
//! move and drop without ever learning it wraps a `CUdeviceptr`. Recovering the
//! pointer requires [`DeviceAlloc::as_any`] and a downcast, which only
//! `fritillaria-cuda` has any reason to do.
//!
//! That is what keeps the rule in CLAUDE.md true — device memory and stream
//! management live in one crate — while still letting `fritillaria-bam` hand a
//! caller device-resident columns.
//!
//! # Why a separate trait from `BlockCodec`
//!
//! [`DeviceBlockCodec`] is deliberately *not* a mode or a flag on
//! [`BlockCodec`](crate::BlockCodec). A residency enum would turn "I asked a CPU
//! codec for device output" into a runtime error firing deep inside a read, and
//! would force every consumer to handle a case that cannot arise for it. Two
//! traits make that a compile error instead, and cost nothing: no caller is
//! genuinely generic over residency. A `BufRead` implementation fundamentally
//! needs host bytes; an aligner kernel fundamentally needs device bytes.
//!
//! `CpuCodec` implements only [`BlockCodec`](crate::BlockCodec), so it cannot be
//! passed where device output is required. [`BlockCodec`](crate::BlockCodec)
//! also stays object-safe, so `Box<dyn BlockCodec>` in the facade is untouched.
//!
//! # Ownership
//!
//! **Dropping a [`DeviceBuffer`] frees the allocation.** A consumer whose kernel
//! is still reading it must keep the owning batch alive until that kernel
//! completes. This is the one rule a caller can get wrong in a way that
//! produces silent corruption rather than an error.

use std::any::Any;
use std::fmt;

use crate::codec::{BlockSpan, InflateBatch};
use crate::error::{Error, Result};

/// An owning handle to memory on a device.
///
/// Implemented by backends. `fritillaria-core` never constructs one; it only
/// defines what every backend's allocation must be able to do.
pub trait DeviceAlloc: Send + Sync + fmt::Debug {
    /// Size of the allocation in bytes.
    fn byte_len(&self) -> usize;

    /// Which device this memory lives on.
    ///
    /// A consumer holding its own context must check this before launching:
    /// memory from another device is not addressable without a peer copy, and
    /// finding out inside a kernel is not a diagnosis anyone can act on.
    fn device_ordinal(&self) -> i32;

    /// Downcast point for the backend that created this allocation.
    ///
    /// `fritillaria-cuda` uses this to recover the concrete allocation and its
    /// pointer. Nothing else has a reason to call it.
    fn as_any(&self) -> &dyn Any;

    /// Copies the whole allocation into `dst`, which must be exactly
    /// [`byte_len`](DeviceAlloc::byte_len) bytes.
    ///
    /// Named rather than implicit, and never a `Deref`, because this is the
    /// exact transfer the module exists to avoid. Callers should have to mean
    /// it.
    fn copy_to_host(&self, dst: &mut [u8]) -> Result<()>;
}

/// Opaque, owning handle to a device allocation.
///
/// Format crates hold these without depending on any GPU crate. Dropping frees
/// the underlying memory — see the module-level ownership note.
#[derive(Debug)]
pub struct DeviceBuffer(Box<dyn DeviceAlloc>);

impl DeviceBuffer {
    /// Wraps a backend allocation.
    #[must_use]
    pub fn new(alloc: Box<dyn DeviceAlloc>) -> Self {
        Self(alloc)
    }

    /// Size of the allocation in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.0.byte_len()
    }

    /// Whether the allocation holds no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }

    /// Which device this memory lives on.
    #[must_use]
    pub fn device_ordinal(&self) -> i32 {
        self.0.device_ordinal()
    }

    /// The backend allocation, for a backend that needs to downcast it.
    #[must_use]
    pub fn alloc(&self) -> &dyn DeviceAlloc {
        &*self.0
    }

    /// Copies the whole allocation into `dst` (exactly `byte_len()` bytes).
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        self.0.copy_to_host(dst)
    }

    /// Copies the whole allocation into a fresh `Vec`.
    ///
    /// Convenience over [`copy_to_host`](DeviceBuffer::copy_to_host); just as
    /// expensive, and just as deliberate.
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.byte_len()];
        self.copy_to_host(&mut out)?;
        Ok(out)
    }
}

/// Inflated output that never left the device.
///
/// The device-resident counterpart of [`InflateBatch`]. Same concatenated
/// layout — blocks back to back, because BAM records span block boundaries —
/// but the bytes are in device memory.
///
/// # Why offsets stay on the host
///
/// They are 8 bytes per block: ~1.3 MB for a 3 GiB BAM against the 10 GiB of
/// payload. The host needs them for virtual offsets and seeking, so keeping
/// them here costs almost nothing and keeps the host able to reason about the
/// batch. Kernels that index blocks directly can have a device-side copy
/// uploaded on demand — see [`set_device_offsets`](Self::set_device_offsets).
#[derive(Debug, Default)]
pub struct DeviceInflateBatch {
    /// `None` only when the batch is empty; an empty batch owns no allocation.
    data: Option<DeviceBuffer>,
    /// Block *i* occupies `data[offsets[i]..offsets[i + 1]]`; length `n + 1`.
    offsets: Vec<usize>,
    device_offsets: Option<DeviceBuffer>,
}

impl DeviceInflateBatch {
    /// Creates an empty batch, owning no device memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            offsets: vec![0],
            device_offsets: None,
        }
    }

    /// Takes ownership of a backend's output buffer and its block boundaries.
    ///
    /// `offsets` must have length `n + 1`, start at 0, be non-decreasing, and
    /// end at `data.byte_len()`. A backend that gets this wrong would otherwise
    /// hand a consumer a column that reads past its allocation, so it is
    /// checked here rather than trusted.
    ///
    /// Any previously held buffer is dropped, which frees it.
    pub fn adopt(&mut self, data: DeviceBuffer, offsets: Vec<usize>) -> Result<()> {
        let invalid = |reason: String| Error::InvalidDeviceBatch { reason };

        if offsets.first() != Some(&0) {
            return Err(invalid(format!(
                "offsets must start at 0, got {:?}",
                offsets.first()
            )));
        }
        if offsets.last() != Some(&data.byte_len()) {
            return Err(invalid(format!(
                "offsets must end at the buffer length {}, got {:?}",
                data.byte_len(),
                offsets.last()
            )));
        }
        if let Some([prev, next, ..]) = offsets.windows(2).find(|w| w[0] > w[1]) {
            return Err(invalid(format!(
                "offsets must be non-decreasing, found {prev} then {next}"
            )));
        }

        self.data = Some(data);
        self.offsets = offsets;
        // Stale: it described the buffer we just replaced.
        self.device_offsets = None;
        Ok(())
    }

    /// Releases the device memory and resets to the empty state.
    pub fn clear(&mut self) {
        self.data = None;
        self.offsets.clear();
        self.offsets.push(0);
        self.device_offsets = None;
    }

    /// The inflated bytes, or `None` if the batch is empty.
    ///
    /// The `Option` is not noise a consumer has to live with: a reader yields
    /// only non-empty batches, so it is resolved at that boundary.
    #[must_use]
    pub fn data(&self) -> Option<&DeviceBuffer> {
        self.data.as_ref()
    }

    /// Number of blocks in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the batch holds no blocks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total inflated bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.as_ref().map_or(0, DeviceBuffer::byte_len)
    }

    /// Block boundaries as offsets into the device buffer; length `n + 1`.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Byte range of block `index` within the device buffer.
    #[must_use]
    pub fn block_range(&self, index: usize) -> Option<std::ops::Range<usize>> {
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        Some(start..end)
    }

    /// Which device this batch lives on, or `None` if it is empty.
    #[must_use]
    pub fn device_ordinal(&self) -> Option<i32> {
        self.data.as_ref().map(DeviceBuffer::device_ordinal)
    }

    /// Attaches a device-side copy of the block offsets.
    ///
    /// Uploaded by the backend on demand: only kernels that index blocks
    /// directly need it, and most consumers work over the concatenated buffer.
    pub fn set_device_offsets(&mut self, offsets: DeviceBuffer) {
        self.device_offsets = Some(offsets);
    }

    /// The device-side copy of the block offsets, if one has been uploaded.
    #[must_use]
    pub fn device_offsets(&self) -> Option<&DeviceBuffer> {
        self.device_offsets.as_ref()
    }

    /// Copies everything back to the host.
    ///
    /// The escape hatch to the host path, and the differential-testing hook:
    /// this must be byte-identical to what the corresponding
    /// [`BlockCodec`](crate::BlockCodec) produces for the same input. Expensive
    /// by construction — it is precisely the transfer this type avoids.
    pub fn to_host(&self) -> Result<InflateBatch> {
        let Some(data) = self.data.as_ref() else {
            return Ok(InflateBatch::new());
        };
        let bytes = data.to_vec()?;
        InflateBatch::from_parts(bytes, self.offsets.clone()).ok_or_else(|| {
            Error::InvalidDeviceBatch {
                reason: "offsets did not describe the buffer".to_string(),
            }
        })
    }
}

/// A backend that inflates BGZF blocks and leaves the output on the device.
///
/// # Contract
///
/// Identical to [`BlockCodec`](crate::BlockCodec) in every respect that
/// matters, and in particular: implementations **must** verify each block's
/// CRC32 and `ISIZE` and fail the batch on mismatch. Staying on the device is
/// not a licence to skip verification.
///
/// That verification is cheap here rather than free: the inflate kernel already
/// folds CRC32 in and writes a per-block status word, so checking costs a
/// download of `4 * n_blocks` bytes — 664 KB for a 3 GiB BAM, against the
/// ~10 GiB of payload that stays put.
///
/// [`DeviceInflateBatch::to_host`] must be byte-identical to the
/// [`BlockCodec`](crate::BlockCodec) path for the same input. That equivalence
/// is what the differential tests assert.
pub trait DeviceBlockCodec {
    /// Human-readable backend name, for diagnostics and benchmark labels.
    fn name(&self) -> &'static str;

    /// Which device this codec allocates on.
    fn device_ordinal(&self) -> i32;

    /// Inflates every span in `spans` into `out`, which is replaced wholesale.
    ///
    /// `batch` is the contiguous compressed bytes the spans index into. Reuse
    /// `out` across calls: a backend may keep its allocation when the next
    /// batch fits, and device allocation is expensive enough to be worth it.
    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()>;
}

/// A shared reference to a codec is itself a codec.
///
/// Building a codec compiles kernels and pins a device context, so it is
/// expensive and deliberately not `Clone`. Without this, a caller driving
/// several readers — different files, or the same file at different batch
/// sizes — would have to build one codec each. The alternative was making every
/// reader generic over ownership, which pushes the same problem onto every
/// caller instead of solving it once.
impl<C: DeviceBlockCodec + ?Sized> DeviceBlockCodec for &C {
    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn device_ordinal(&self) -> i32 {
        (**self).device_ordinal()
    }

    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        (**self).inflate_batch_device(batch, spans, out)
    }
}

/// A host-memory stand-in for device memory.
///
/// This is what makes the device design testable on a machine with no GPU —
/// which is where all host-side development happens, so it is load-bearing
/// rather than a convenience. It implements the same traits with `Vec<u8>`
/// behind them, so layout invariants, ownership and the codec seam can all be
/// exercised locally; only the transfer is fictional.
///
/// Behind the `testing` feature: it is scaffolding, not something to ship in a
/// caller's dependency graph by default.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::{Any, DeviceAlloc, DeviceBuffer, Error, Result};

    /// A [`DeviceAlloc`] backed by ordinary host memory.
    #[derive(Debug)]
    pub struct HostAlloc {
        bytes: Vec<u8>,
        ordinal: i32,
    }

    impl HostAlloc {
        /// Wraps bytes as a pretend device allocation on device 0.
        #[must_use]
        pub fn new(bytes: Vec<u8>) -> Self {
            Self { bytes, ordinal: 0 }
        }

        /// Wraps bytes as a pretend allocation on a specific device.
        ///
        /// Lets tests cover the cross-device case, which is otherwise
        /// unreachable without two GPUs.
        #[must_use]
        pub fn on_device(bytes: Vec<u8>, ordinal: i32) -> Self {
            Self { bytes, ordinal }
        }

        /// Wraps bytes directly as a [`DeviceBuffer`].
        #[must_use]
        pub fn buffer(bytes: Vec<u8>) -> DeviceBuffer {
            DeviceBuffer::new(Box::new(Self::new(bytes)))
        }

        /// The bytes, without pretending a transfer happened.
        #[must_use]
        pub fn bytes(&self) -> &[u8] {
            &self.bytes
        }
    }

    impl DeviceAlloc for HostAlloc {
        fn byte_len(&self) -> usize {
            self.bytes.len()
        }

        fn device_ordinal(&self) -> i32 {
            self.ordinal
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
            if dst.len() != self.bytes.len() {
                return Err(Error::InvalidDeviceBatch {
                    reason: format!(
                        "copy_to_host destination is {} bytes, allocation is {}",
                        dst.len(),
                        self.bytes.len()
                    ),
                });
            }
            dst.copy_from_slice(&self.bytes);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::HostAlloc;
    use super::*;

    #[test]
    fn an_empty_batch_owns_nothing() {
        let batch = DeviceInflateBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.byte_len(), 0);
        assert_eq!(batch.offsets(), &[0]);
        assert!(batch.data().is_none());
        assert!(batch.device_ordinal().is_none());
    }

    #[test]
    fn adopts_a_valid_layout() {
        let mut batch = DeviceInflateBatch::new();
        batch
            .adopt(HostAlloc::buffer(b"helloworld".to_vec()), vec![0, 5, 5, 10])
            .unwrap();

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.byte_len(), 10);
        assert_eq!(batch.block_range(0), Some(0..5));
        // An empty block is legal — the BGZF EOF marker is exactly this — and
        // must not read as an absent one.
        assert_eq!(batch.block_range(1), Some(5..5));
        assert_eq!(batch.block_range(2), Some(5..10));
        assert_eq!(batch.block_range(3), None);
    }

    #[test]
    fn rejects_offsets_that_do_not_describe_the_buffer() {
        let cases: &[(&str, Vec<usize>)] = &[
            ("missing leading zero", vec![1, 3]),
            ("does not end at byte_len", vec![0, 2]),
            ("not monotonic", vec![0, 3, 1, 3]),
            ("no sentinel", vec![]),
        ];

        for (why, offsets) in cases {
            let mut batch = DeviceInflateBatch::new();
            let err = batch.adopt(HostAlloc::buffer(b"abc".to_vec()), offsets.clone());
            assert!(
                matches!(err, Err(Error::InvalidDeviceBatch { .. })),
                "{why}: a backend must not be able to hand over a buffer its \
                 offsets read past"
            );
        }
    }

    #[test]
    fn to_host_matches_the_host_batch_layout() {
        // The differential-testing hook: device output copied back must be
        // indistinguishable from what the host path builds.
        let mut device = DeviceInflateBatch::new();
        device
            .adopt(HostAlloc::buffer(b"alphabeta".to_vec()), vec![0, 5, 9])
            .unwrap();

        let mut host = InflateBatch::new();
        host.push_block(b"alpha");
        host.push_block(b"beta");

        let copied = device.to_host().unwrap();
        assert_eq!(copied.data(), host.data());
        assert_eq!(copied.offsets(), host.offsets());
    }

    #[test]
    fn to_host_on_an_empty_batch_is_an_empty_batch() {
        let copied = DeviceInflateBatch::new().to_host().unwrap();
        assert!(copied.is_empty());
        assert_eq!(copied.offsets(), &[0]);
    }

    #[test]
    fn clear_releases_the_buffer_and_keeps_the_sentinel() {
        let mut batch = DeviceInflateBatch::new();
        batch
            .adopt(HostAlloc::buffer(b"data".to_vec()), vec![0, 4])
            .unwrap();
        batch.clear();

        assert!(batch.is_empty());
        assert!(batch.data().is_none());
        assert_eq!(batch.offsets(), &[0]);
    }

    #[test]
    fn adopting_again_replaces_the_previous_buffer() {
        let mut batch = DeviceInflateBatch::new();
        batch
            .adopt(HostAlloc::buffer(b"first".to_vec()), vec![0, 5])
            .unwrap();
        batch.set_device_offsets(HostAlloc::buffer(vec![0, 5]));
        assert!(batch.device_offsets().is_some());

        batch
            .adopt(HostAlloc::buffer(b"second!".to_vec()), vec![0, 7])
            .unwrap();

        assert_eq!(batch.to_host().unwrap().data(), b"second!");
        assert!(
            batch.device_offsets().is_none(),
            "uploaded offsets described the replaced buffer and must not \
             survive it"
        );
    }

    #[test]
    fn a_rejected_adopt_leaves_the_batch_untouched() {
        let mut batch = DeviceInflateBatch::new();
        batch
            .adopt(HostAlloc::buffer(b"good".to_vec()), vec![0, 4])
            .unwrap();

        let _ = batch.adopt(HostAlloc::buffer(b"bad".to_vec()), vec![0, 99]);

        assert_eq!(
            batch.to_host().unwrap().data(),
            b"good",
            "a failed adopt must not corrupt the batch it was replacing"
        );
    }

    #[test]
    fn buffers_report_their_device() {
        let buf = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![1, 2, 3], 2)));
        assert_eq!(buf.device_ordinal(), 2);

        let mut batch = DeviceInflateBatch::new();
        batch.adopt(buf, vec![0, 3]).unwrap();
        assert_eq!(
            batch.device_ordinal(),
            Some(2),
            "a consumer must be able to check this before launching a kernel"
        );
    }

    #[test]
    fn copy_to_host_rejects_a_mismatched_destination() {
        let buf = HostAlloc::buffer(b"abcd".to_vec());
        let mut too_small = [0u8; 2];
        assert!(buf.copy_to_host(&mut too_small).is_err());
        let mut exact = [0u8; 4];
        buf.copy_to_host(&mut exact).unwrap();
        assert_eq!(&exact, b"abcd");
    }

    #[test]
    fn the_backend_can_downcast_its_own_allocation() {
        // The seam that lets fritillaria-cuda recover a CUdeviceptr while core
        // stays free of any CUDA type.
        let buf = HostAlloc::buffer(b"payload".to_vec());
        let recovered = buf
            .alloc()
            .as_any()
            .downcast_ref::<HostAlloc>()
            .expect("a backend must be able to recover its own allocation");
        assert_eq!(recovered.bytes(), b"payload");
    }
}
