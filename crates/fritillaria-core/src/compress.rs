//! The compression seam.
//!
//! The mirror image of [`BlockCodec`](crate::BlockCodec), and deliberately shaped
//! so that the two compose: a compressor consumes exactly what a codec produces
//! — a concatenated buffer plus block boundaries — and produces a byte stream a
//! codec can read back. That makes the round trip the primary test, which
//! matters more here than on the read path because **compression has no
//! byte-identity oracle**. Two valid DEFLATE streams of the same input
//! legitimately differ, so a backend cannot be diffed against the CPU reference
//! the way an inflater can. What is left is: does it round-trip, does `samtools`
//! accept it, and is the ratio at least as good as a stated floor.
//!
//! # One block out per block in
//!
//! [`BlockCompressor::compress_batch`] emits exactly one BGZF block per input
//! chunk. That is a contract, not an implementation detail, and it is worth more
//! than it looks:
//!
//! **Block boundaries carry information.** htslib starts a new BGZF block rather
//! than splitting a BAM record, which is why a block start is almost always a
//! record start — and that is the entire reason this library's record scan
//! parallelises instead of walking one serial chain per batch. A compressor that
//! silently split an over-large chunk to make it fit would produce a file that
//! is valid, reads correctly, and is *slower for us to read*, with nothing to
//! indicate why. Refusing is the only honest option.
//!
//! Keeping the guarantee unconditional needs one restriction, which is where the
//! [`MAX_COMPRESSIBLE_PAYLOAD`] limit comes from: a chunk is capped slightly
//! below the format's own payload maximum so that the worst case — storing the
//! bytes uncompressed, which costs 5 bytes of DEFLATE framing — still fits
//! BGZF's 64 KiB whole-block cap. Within that limit no input can defeat the
//! one-to-one mapping.

use crate::MAX_BLOCK_SIZE;
use crate::error::Result;

/// Bytes of gzip/BGZF header before a block's deflate stream.
///
/// 10 fixed gzip bytes, 2 of `XLEN`, and the 6-byte `BC` subfield.
pub const BGZF_HEADER_SIZE: usize = 18;

/// Bytes of gzip trailer after it: CRC32 then `ISIZE`.
pub const BGZF_TRAILER_SIZE: usize = 8;

/// Bytes a DEFLATE **stored** block adds to its payload.
///
/// One byte of `BFINAL`/`BTYPE`, then `LEN` and `NLEN`. The number matters
/// because it has no data-dependent term — see [`MAX_COMPRESSIBLE_PAYLOAD`].
pub const STORED_BLOCK_HEADER: usize = 5;

/// Largest chunk a [`BlockCompressor`] accepts, in uncompressed bytes.
///
/// 65505 = 65536 (the block cap) − 18 (BGZF header) − 8 (gzip trailer) − 5 (a
/// DEFLATE stored block's own header). The last term is what makes the
/// [one-to-one guarantee](self#one-block-out-per-block-in) unconditional:
/// incompressible input falls back to a stored block, which is exactly `len + 5`
/// bytes, so a chunk at this limit frames to precisely 65536 bytes and anything
/// smaller has room to spare.
///
/// It is 5 bytes under what the format would allow a *payload* to be. Writers
/// use 65280 anyway — htslib's choice and our [default] — so the difference is
/// unreachable in practice and buys a total contract.
///
/// [default]: ../fritillaria_bgzf/write/constant.DEFAULT_PAYLOAD_SIZE.html
pub const MAX_COMPRESSIBLE_PAYLOAD: usize = MAX_DEFLATE_STREAM - STORED_BLOCK_HEADER;

/// Largest deflate stream that still fits a block once framed.
///
/// The cap BGZF imposes is on the whole block, so this — not
/// [`MAX_BLOCK_SIZE`] — is what a compressor's output has to come in under.
pub const MAX_DEFLATE_STREAM: usize = MAX_BLOCK_SIZE - BGZF_HEADER_SIZE - BGZF_TRAILER_SIZE;

/// How one block's payload ends up encoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Framing {
    /// The compressor's deflate stream is used as-is.
    Deflated,
    /// The payload is stored verbatim in a DEFLATE stored block.
    Stored,
}

/// Chooses between a compressor's output and storing the payload verbatim.
///
/// **One rule, two implementations downstream**, which is why it lives here
/// rather than in either of them: the host reference calls this directly, and
/// the nvCOMP path calls it while planning where each block lands in the output
/// stream — a kernel cannot make the choice itself, because the host has to size
/// the output buffer before the kernel runs. Two copies of the rule would be two
/// chances to disagree about where a block starts, which is the class of bug
/// that produces a plausible-looking file rather than an error.
///
/// Storing wins in two cases and only the first is about correctness:
///
/// - the deflate stream is too large to frame within BGZF's 64 KiB block cap,
///   which is reachable — nvCOMP's worst case is 2.26x the chunk;
/// - it is merely *larger* than the payload plus a stored block's 5-byte header,
///   which is a ratio loss for nothing.
#[must_use]
pub fn choose_framing(payload_len: usize, deflate_len: usize) -> Framing {
    // Compared against the stream limit rather than summing up to the block cap
    // on purpose: nvCOMP reports sizes this code does not choose, and a sum
    // would be an overflow rather than a rejection for an absurd one.
    if deflate_len > MAX_DEFLATE_STREAM || deflate_len > payload_len + STORED_BLOCK_HEADER {
        Framing::Stored
    } else {
        Framing::Deflated
    }
}

/// Total framed size of one block, given the choice [`choose_framing`] makes.
///
/// The number a host has to know *before* the bytes exist, to lay out a dense
/// output stream.
#[must_use]
pub fn framed_size(payload_len: usize, deflate_len: usize) -> usize {
    let body = match choose_framing(payload_len, deflate_len) {
        Framing::Deflated => deflate_len,
        Framing::Stored => payload_len + STORED_BLOCK_HEADER,
    };
    BGZF_HEADER_SIZE + body + BGZF_TRAILER_SIZE
}

/// Compressed output for a batch of blocks: a ready-to-write BGZF byte stream.
///
/// Holds **framed** blocks — gzip header, deflate stream, CRC32 and `ISIZE` —
/// concatenated in order, so `data()` can go straight to a file. It is not the
/// compressed counterpart of [`InflateBatch`](crate::InflateBatch) with the
/// bytes swapped: that one holds naked payloads, this one holds whole blocks.
/// Confusing the two produces a file that looks like data, so they are separate
/// types rather than one type with a flag.
///
/// What is *not* here is the 28-byte EOF block. A batch is a batch; termination
/// is a property of a file, and a writer appends it once at the end. Omitting it
/// is how a file comes to be reported as truncated by every downstream tool, so
/// it is the writer's job and stated in both places.
#[derive(Clone, Debug, Default)]
pub struct CompressedBatch {
    data: Vec<u8>,
    /// Block *i* occupies `data[offsets[i]..offsets[i + 1]]`; length `n + 1`.
    offsets: Vec<usize>,
}

impl CompressedBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Clears the batch, keeping allocated capacity for reuse across batches.
    pub fn clear(&mut self) {
        self.data.clear();
        self.offsets.clear();
        self.offsets.push(0);
    }

    /// Reserves room for `total` compressed bytes.
    pub fn reserve(&mut self, total: usize) {
        self.data.reserve(total);
    }

    /// The buffer blocks are appended to, and the boundary list to update.
    ///
    /// Backends frame directly into this rather than into a scratch buffer that
    /// is then copied. The caller must push one offset per block appended, or
    /// the batch stops describing itself — which
    /// [`is_consistent`](Self::is_consistent) checks.
    pub fn parts_mut(&mut self) -> (&mut Vec<u8>, &mut Vec<usize>) {
        (&mut self.data, &mut self.offsets)
    }

    /// Records that a block ends at the current end of the buffer.
    pub fn end_block(&mut self) {
        self.offsets.push(self.data.len());
    }

    /// Whether the boundaries describe the buffer.
    ///
    /// Cheap enough to assert after a backend has filled a batch.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.offsets.first() == Some(&0)
            && self.offsets.last() == Some(&self.data.len())
            && self.offsets.windows(2).all(|w| w[0] <= w[1])
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

    /// The framed blocks, back to back — the bytes to write to a file.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Block boundaries as offsets into [`data`](Self::data); length `n + 1`.
    ///
    /// These are the compressed offsets a `gzi` index or a
    /// [`VirtualOffset`](crate::VirtualOffset) is built from, which is why they
    /// are kept rather than recomputed by re-walking the stream later.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The framed bytes of block `index`.
    #[must_use]
    pub fn block(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        self.data.get(start..end)
    }

    /// Total compressed bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

/// A backend that compresses batches of payloads into BGZF blocks.
///
/// # Contract
///
/// - **Exactly one output block per input chunk**, in order. See the
///   [module note](self#one-block-out-per-block-in) for why this is load-bearing
///   rather than tidy.
/// - Every chunk must be at most [`MAX_COMPRESSIBLE_PAYLOAD`] bytes; a longer
///   one is an error, because no implementation could honour the guarantee for
///   it.
/// - Output must be **spec-valid BGZF** that `samtools` reads: correct `BC`
///   subfield (total block size *minus one*), CRC32 and `ISIZE` of the
///   *uncompressed* payload. That last point is the one a device path is most
///   likely to get wrong — the checksum is of the bytes going in, so it must be
///   computed before compressing, not after.
/// - The EOF block is **not** emitted; see [`CompressedBatch`].
///
/// Unlike [`BlockCodec`](crate::BlockCodec), implementations are *not* required
/// to be byte-identical to each other. They cannot be: compression level and
/// algorithm legitimately change the output. The equivalence that is required is
/// weaker and stated as a round trip — inflating a backend's output must
/// reproduce the input exactly, block for block.
pub trait BlockCompressor {
    /// Human-readable backend name, for diagnostics and benchmark labels.
    fn name(&self) -> &'static str;

    /// Compresses each chunk of `data` delimited by `bounds` into one block.
    ///
    /// `bounds` has length `n + 1`, starts at 0 and ends at `data.len()` — the
    /// same shape [`InflateBatch::offsets`](crate::InflateBatch::offsets)
    /// produces, so a batch that was just inflated can be handed straight back.
    ///
    /// `out` is cleared first; reuse it across calls to avoid reallocating.
    ///
    /// # Errors
    ///
    /// If `bounds` does not describe `data`, or any chunk exceeds
    /// [`MAX_COMPRESSIBLE_PAYLOAD`].
    fn compress_batch(
        &self,
        data: &[u8],
        bounds: &[usize],
        out: &mut CompressedBatch,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_batch_has_one_offset() {
        let batch = CompressedBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.offsets(), &[0]);
        assert!(batch.is_consistent());
    }

    #[test]
    fn tracks_block_boundaries() {
        let mut batch = CompressedBatch::new();
        for block in [&b"first"[..], b"second"] {
            batch.parts_mut().0.extend_from_slice(block);
            batch.end_block();
        }

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.byte_len(), 11);
        assert_eq!(batch.block(0), Some(&b"first"[..]));
        assert_eq!(batch.block(1), Some(&b"second"[..]));
        assert_eq!(batch.block(2), None);
        assert!(batch.is_consistent());
    }

    #[test]
    fn a_backend_that_forgets_a_boundary_is_detectable() {
        // The one way filling a batch in place can go wrong, and it would
        // otherwise surface as a truncated final block much later.
        let mut batch = CompressedBatch::new();
        batch.parts_mut().0.extend_from_slice(b"unterminated");
        assert!(!batch.is_consistent());
    }

    #[test]
    fn clear_preserves_the_sentinel_offset() {
        let mut batch = CompressedBatch::new();
        batch.parts_mut().0.extend_from_slice(b"data");
        batch.end_block();
        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.offsets(), &[0]);
        assert!(batch.is_consistent());
    }

    /// The limit is derived, not chosen, so the derivation is what is checked.
    #[test]
    fn a_chunk_at_the_limit_frames_to_exactly_one_maximal_block() {
        assert_eq!(MAX_COMPRESSIBLE_PAYLOAD, 65505);
        assert_eq!(
            framed_size(MAX_COMPRESSIBLE_PAYLOAD, usize::MAX),
            MAX_BLOCK_SIZE,
            "the worst case for the largest legal chunk must be exactly the cap"
        );
    }

    #[test]
    fn a_deflate_stream_too_large_to_frame_falls_back_to_storing() {
        // The reachable case: nvCOMP's worst case is 2.26x the chunk.
        let payload = 65_280;
        assert_eq!(choose_framing(payload, 148_256), Framing::Stored);
        assert_eq!(
            framed_size(payload, 148_256),
            BGZF_HEADER_SIZE + payload + STORED_BLOCK_HEADER + BGZF_TRAILER_SIZE
        );
    }

    #[test]
    fn a_stream_that_would_grow_the_payload_falls_back_to_storing() {
        // Not about the cap — this one would frame fine, it is just a waste.
        assert_eq!(choose_framing(1000, 1006), Framing::Stored);
        assert_eq!(
            framed_size(1000, 1006),
            BGZF_HEADER_SIZE + 1005 + BGZF_TRAILER_SIZE
        );
    }

    #[test]
    fn a_stream_no_larger_than_a_stored_block_is_used_as_it_is() {
        // Exactly equal must not flip: switching would change the bytes for no
        // gain, and the host and device paths would then disagree on a tie.
        assert_eq!(choose_framing(1000, 1005), Framing::Deflated);
        assert_eq!(choose_framing(1000, 400), Framing::Deflated);
        assert_eq!(
            framed_size(1000, 400),
            BGZF_HEADER_SIZE + 400 + BGZF_TRAILER_SIZE
        );
    }

    #[test]
    fn an_empty_payload_still_deflates() {
        // The EOF marker is this case, and it must stay 28 bytes.
        assert_eq!(choose_framing(0, 2), Framing::Deflated);
        assert_eq!(framed_size(0, 2), 28);
    }
}
