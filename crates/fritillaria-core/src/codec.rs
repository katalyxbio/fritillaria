//! The backend seam.
//!
//! Every decompression backend implements [`BlockCodec`]. The CPU reference is
//! the correctness oracle; CUDA and (optionally) nvCOMP must produce
//! byte-identical output for the same input. Format crates are generic over
//! this trait and never name a concrete backend.

use crate::error::Result;

/// Where one BGZF block lives and what it should inflate to.
///
/// Produced by block discovery (a cheap sequential header walk) and consumed by
/// a codec. Carrying `isize` and `crc32` here is what lets a backend compute
/// output offsets by prefix sum *before* decompressing anything — which is the
/// reason blocks can be inflated in parallel at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSpan {
    /// Offset of the block's first byte within the compressed stream.
    ///
    /// Absolute file offset, not an index into whatever slice is being passed
    /// to the codec — used for error reporting and virtual offsets.
    pub compressed_offset: u64,
    /// Offset of the deflate payload relative to the start of the batch slice.
    pub payload_start: usize,
    /// Length of the deflate payload in bytes.
    pub payload_len: usize,
    /// Expected inflated size, from the block's `ISIZE` trailer.
    pub isize: u32,
    /// Expected CRC32 of the inflated bytes, from the block's trailer.
    pub crc32: u32,
}

impl BlockSpan {
    /// The deflate payload for this block within `batch`.
    ///
    /// Returns `None` if the span does not lie inside the slice.
    #[must_use]
    pub fn payload<'a>(&self, batch: &'a [u8]) -> Option<&'a [u8]> {
        batch.get(self.payload_start..self.payload_start.checked_add(self.payload_len)?)
    }
}

/// Inflated output for a batch of blocks.
///
/// Blocks are concatenated into one buffer because BAM records span block
/// boundaries — a per-block `Vec<Vec<u8>>` would force a copy to stitch them
/// back together before records could be parsed.
///
/// Currently host-resident. The CUDA backend will grow a device-resident
/// counterpart so decoded fields never round-trip through host memory; the
/// trait is shaped to allow that without changing callers.
#[derive(Clone, Debug, Default)]
pub struct InflateBatch {
    data: Vec<u8>,
    /// Block *i* occupies `data[offsets[i]..offsets[i + 1]]`; length `n + 1`.
    offsets: Vec<usize>,
}

impl InflateBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Adopts an already-concatenated buffer and its block boundaries.
    ///
    /// The GPU path already produces exactly this layout, so re-appending block
    /// by block would copy the whole batch for nothing.
    ///
    /// `offsets` must have length `n + 1`, start at 0, be non-decreasing, and
    /// end at `data.len()`; violating that is a programming error, so this
    /// returns `None` rather than panicking deep inside a later read.
    #[must_use]
    pub fn from_parts(data: Vec<u8>, offsets: Vec<usize>) -> Option<Self> {
        if offsets.first() != Some(&0) || offsets.last() != Some(&data.len()) {
            return None;
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return None;
        }
        Some(Self { data, offsets })
    }

    /// Clears the batch, keeping allocated capacity for reuse across batches.
    pub fn clear(&mut self) {
        self.data.clear();
        self.offsets.clear();
        self.offsets.push(0);
    }

    /// Reserves room for `total` inflated bytes.
    pub fn reserve(&mut self, total: usize) {
        self.data.reserve(total);
    }

    /// Appends one block's inflated bytes.
    pub fn push_block(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
        self.offsets.push(self.data.len());
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

    /// The concatenated payload of every block, in file order.
    ///
    /// This is the buffer record-boundary discovery runs over.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable access to the concatenated payload.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }

    /// Block boundaries as offsets into [`InflateBatch::data`]; length `n + 1`.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The bytes of block `index`.
    #[must_use]
    pub fn block(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        self.data.get(start..end)
    }
}

/// A backend that inflates batches of independent BGZF blocks.
///
/// # Contract
///
/// Implementations **must** verify each block's CRC32 and `ISIZE` and fail the
/// batch on mismatch. Skipping verification is not an implementation choice: a
/// silently corrupt read in a genomics pipeline is worse than a slow one. A
/// backend that cannot verify should return an error rather than unverified
/// bytes.
///
/// Implementations must be byte-identical to the CPU reference for all inputs.
/// That equivalence is what the differential tests assert.
pub trait BlockCodec {
    /// Human-readable backend name, for diagnostics and benchmark labels.
    fn name(&self) -> &'static str;

    /// Inflates every span in `spans`, appending to `out` in order.
    ///
    /// `batch` is the contiguous compressed bytes the spans index into.
    /// `out` is cleared first; reuse it across calls to avoid reallocating.
    fn inflate_batch(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut InflateBatch,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_batch_has_one_offset() {
        let batch = InflateBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.offsets(), &[0]);
    }

    #[test]
    fn tracks_block_boundaries() {
        let mut batch = InflateBatch::new();
        batch.push_block(b"hello");
        batch.push_block(b"");
        batch.push_block(b"world");

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.data(), b"helloworld");
        assert_eq!(batch.offsets(), &[0, 5, 5, 10]);
        assert_eq!(batch.block(0), Some(&b"hello"[..]));
        // An empty block is legal (the EOF marker is exactly this) and must not
        // be confused with an absent one.
        assert_eq!(batch.block(1), Some(&b""[..]));
        assert_eq!(batch.block(2), Some(&b"world"[..]));
        assert_eq!(batch.block(3), None);
    }

    #[test]
    fn clear_preserves_the_sentinel_offset() {
        let mut batch = InflateBatch::new();
        batch.push_block(b"data");
        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.offsets(), &[0]);
    }

    #[test]
    fn from_parts_adopts_a_valid_layout() {
        let batch = InflateBatch::from_parts(b"helloworld".to_vec(), vec![0, 5, 5, 10]).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.block(0), Some(&b"hello"[..]));
        assert_eq!(batch.block(1), Some(&b""[..]));
        assert_eq!(batch.block(2), Some(&b"world"[..]));
    }

    #[test]
    fn from_parts_rejects_inconsistent_offsets() {
        // Missing leading zero.
        assert!(InflateBatch::from_parts(b"abc".to_vec(), vec![1, 3]).is_none());
        // Does not end at data.len().
        assert!(InflateBatch::from_parts(b"abc".to_vec(), vec![0, 2]).is_none());
        // Not monotonic.
        assert!(InflateBatch::from_parts(b"abc".to_vec(), vec![0, 3, 1, 3]).is_none());
        // Empty offsets have no sentinel.
        assert!(InflateBatch::from_parts(Vec::new(), Vec::new()).is_none());
    }

    #[test]
    fn from_parts_round_trips_with_push_block() {
        // The two construction paths must produce identical batches, or the
        // CPU and GPU codecs would disagree on layout alone.
        let mut pushed = InflateBatch::new();
        pushed.push_block(b"alpha");
        pushed.push_block(b"beta");

        let adopted = InflateBatch::from_parts(b"alphabeta".to_vec(), vec![0, 5, 9]).unwrap();

        assert_eq!(pushed.data(), adopted.data());
        assert_eq!(pushed.offsets(), adopted.offsets());
    }

    #[test]
    fn payload_slicing_is_bounds_checked() {
        let span = BlockSpan {
            compressed_offset: 0,
            payload_start: 2,
            payload_len: 3,
            isize: 0,
            crc32: 0,
        };
        assert_eq!(span.payload(b"..abc.."), Some(&b"abc"[..]));
        assert_eq!(span.payload(b"..ab"), None);
    }
}
