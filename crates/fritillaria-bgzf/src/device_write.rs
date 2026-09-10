//! A streaming BGZF writer whose input stays on the device.
//!
//! The mirror of [`DeviceBgzfReader`](crate::DeviceBgzfReader), and the piece
//! that closes the loop: until this existed, a GPU tool could read a BAM without
//! its records ever touching host memory and then had to bring all of them back
//! to write one. Half an accelerated pipeline.
//!
//! # What it adds over the compressor
//!
//! [`DeviceBlockCompressor`] compresses *one batch*. This drives it over a whole
//! file, and the three things it adds are the three a caller would otherwise get
//! wrong:
//!
//! - **Batching against a VRAM budget.** Compression scratch is ~1.11 MB per
//!   64 KiB chunk, so a batch that fits the read path's 166,012 blocks would
//!   need 185 GB. Batches here are ~15x smaller and the writer, not the caller,
//!   decides where they split.
//! - **The EOF block.** [`finish`](DeviceBgzfWriter::finish) writes it. Without
//!   it every tool reports the file as truncated, and it is the single easiest
//!   thing to leave out — which is why it is a method that must be called rather
//!   than something `Drop` might miss.
//! - **The compressed offset.** [`virtual_position`](DeviceBgzfWriter::virtual_position)
//!   tracks where the next block will land, which is what a BAM writer needs to
//!   build a BAI as it goes.
//!
//! # Block boundaries are the caller's, and that is the whole point
//!
//! [`write_batch`](DeviceBgzfWriter::write_batch) takes the boundaries the
//! caller chose and never moves them. For a BAM that matters: htslib ends a
//! block rather than splitting a record, which is what makes a block start
//! almost always a record start and gives this library's GPU scan one chain per
//! block instead of one serial chain per batch. A writer that re-chunked would
//! emit a file that is valid, reads correctly, and is *slower for us to read*.
//!
//! Batching does not move them either. A batch boundary always falls *on* a
//! chunk boundary, so splitting a chunk list into batches is invisible in the
//! output — the same bytes come out whatever the batch size, and
//! `tests/device_write.rs` asserts exactly that down to one chunk per batch.
//!
//! [`write_all_device`](DeviceBgzfWriter::write_all_device) is the convenience
//! for formats where boundaries carry nothing, and it says so.

use std::io::Write;

use fritillaria_core::{
    CompressedBatch, DeviceBlockCompressor, DeviceBuffer, Error, MAX_COMPRESSIBLE_PAYLOAD, Result,
};

use crate::block::EOF_BLOCK;
use crate::virtual_position::VirtualPosition;
use crate::write::DEFAULT_PAYLOAD_SIZE;

/// Chunks per compression batch when the caller does not choose.
///
/// Deliberately far below the read path's 256-blocks-per-batch *in bytes*: at
/// ~1.39 MB of device memory per chunk this is roughly 1.4 GB, which fits every
/// card this project has run on with room for the caller's own data. A caller
/// who knows their VRAM should use [`CompressBudget::chunks_in`] and
/// [`with_chunks_per_batch`](DeviceBgzfWriter::with_chunks_per_batch).
///
/// [`CompressBudget::chunks_in`]: https://docs.rs/fritillaria-cuda
pub const DEFAULT_CHUNKS_PER_BATCH: usize = 1024;

/// Writes BGZF to a host sink from payloads that live on the device.
///
/// The EOF block is written by [`finish`](DeviceBgzfWriter::finish). Dropping
/// the writer without calling it produces a file every downstream tool reports
/// as truncated, so `finish` is not optional.
#[derive(Debug)]
pub struct DeviceBgzfWriter<W, C> {
    inner: W,
    compressor: C,
    chunks_per_batch: usize,
    payload_size: usize,

    /// Reused across batches; compression allocates enough already.
    batch: CompressedBatch,

    blocks_written: u64,
    batches_run: u64,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
}

impl<W: Write, C: DeviceBlockCompressor> DeviceBgzfWriter<W, C> {
    /// Creates a writer over `inner`, compressing with `compressor`.
    pub fn new(inner: W, compressor: C) -> Self {
        Self {
            inner,
            compressor,
            chunks_per_batch: DEFAULT_CHUNKS_PER_BATCH,
            payload_size: DEFAULT_PAYLOAD_SIZE,
            batch: CompressedBatch::new(),
            blocks_written: 0,
            batches_run: 0,
            compressed_bytes: 0,
            uncompressed_bytes: 0,
        }
    }

    /// Sets how many chunks go to the device in one compression batch.
    ///
    /// This is a **memory** knob, not a throughput one: scratch dominates, so
    /// the ceiling is whatever the card has. Clamped to at least one, because a
    /// batch of nothing cannot make progress.
    #[must_use]
    pub fn with_chunks_per_batch(mut self, chunks: usize) -> Self {
        self.chunks_per_batch = chunks.max(1);
        self
    }

    /// Sets the chunk size [`write_all_device`](Self::write_all_device) uses.
    ///
    /// Has no effect on [`write_batch`](Self::write_batch), which uses the
    /// caller's boundaries.
    #[must_use]
    pub fn with_payload_size(mut self, size: usize) -> Self {
        self.payload_size = size.clamp(1, MAX_COMPRESSIBLE_PAYLOAD);
        self
    }

    /// Compresses `data` at the caller's `bounds` and writes the blocks.
    ///
    /// `bounds` has length `n + 1` and is non-decreasing. Passing what a
    /// [`DeviceInflateBatch`](fritillaria_core::DeviceInflateBatch) hands out
    /// writes a just-read batch straight back.
    ///
    /// Every boundary survives into the output; see the module docs for why that
    /// is a contract rather than a convenience. Chunks are split into batches
    /// internally, and a batch boundary always falls on a chunk boundary, so the
    /// output does not depend on
    /// [`with_chunks_per_batch`](Self::with_chunks_per_batch) —
    /// [`batches_run`](Self::batches_run) is what makes that testable rather
    /// than merely claimed.
    ///
    /// # Errors
    ///
    /// If `bounds` is empty, decreasing, or reaches past `data`, if any chunk
    /// exceeds [`MAX_COMPRESSIBLE_PAYLOAD`], or if the sink or the device fails.
    pub fn write_batch(&mut self, data: &DeviceBuffer, bounds: &[usize]) -> Result<()> {
        let Some(&last) = bounds.last() else {
            return Err(self.malformed("bounds must have at least one entry".to_string()));
        };
        if last > data.byte_len() {
            return Err(self.malformed(format!(
                "bounds reach byte {last} of {} device bytes",
                data.byte_len()
            )));
        }
        if bounds.len() < 2 {
            return Ok(());
        }

        // Batches are windows of the boundary list, overlapping by one so each
        // starts where the last ended. `bounds` names ranges rather than
        // covering the buffer, so a window can be handed to the compressor
        // as-is: no rebasing, no sub-slicing a `DeviceBuffer` (which cannot be
        // done without naming a backend), and no copy.
        //
        // The overlap is why a batch boundary always lands on a chunk boundary,
        // which is what makes the output independent of `chunks_per_batch`.
        let mut start = 0;
        while start + 1 < bounds.len() {
            let end = (start + self.chunks_per_batch + 1).min(bounds.len());
            let window = &bounds[start..end];

            self.compressor
                .compress_batch_device(data, window, &mut self.batch)?;
            self.uncompressed_bytes += (window[window.len() - 1] - window[0]) as u64;
            self.flush_batch()?;

            start = end - 1;
        }

        Ok(())
    }

    fn malformed(&self, reason: String) -> Error {
        Error::Malformed {
            format: "bgzf",
            position: self.compressed_bytes,
            reason,
        }
    }

    /// Writes the compressed batch to the sink and updates the counters.
    fn flush_batch(&mut self) -> Result<()> {
        self.inner.write_all(self.batch.data())?;
        self.batches_run += 1;
        self.blocks_written += self.batch.len() as u64;
        self.compressed_bytes += self.batch.byte_len() as u64;
        Ok(())
    }

    /// Compresses everything in `data`, chunked at the configured payload size.
    ///
    /// **The convenience, and it is only correct when block boundaries carry no
    /// meaning.** For a BAM or BCF they do — see the module docs — so use
    /// [`write_batch`](Self::write_batch) and pass record-aligned bounds. This
    /// is for `bgzip`ping text, where any split is as good as any other.
    pub fn write_all_device(&mut self, data: &DeviceBuffer) -> Result<()> {
        let len = data.byte_len();
        let mut bounds: Vec<usize> = (0..len).step_by(self.payload_size).collect();
        bounds.push(len);
        self.write_batch(data, &bounds)
    }

    /// Blocks written so far, excluding the EOF block.
    #[must_use]
    pub fn blocks_written(&self) -> u64 {
        self.blocks_written
    }

    /// Compression batches run so far.
    ///
    /// Exposed for diagnostics, and because without it the claim that batching
    /// does not change the output is untestable: a writer that quietly ignored
    /// [`with_chunks_per_batch`](Self::with_chunks_per_batch) would satisfy that
    /// claim vacuously. CLAUDE.md records a rented VM spent on the same shape of
    /// mistake — a test asserting several batches over a fixture that arrived in
    /// one.
    #[must_use]
    pub fn batches_run(&self) -> u64 {
        self.batches_run
    }

    /// Compressed bytes written so far, excluding the EOF block.
    #[must_use]
    pub fn compressed_bytes(&self) -> u64 {
        self.compressed_bytes
    }

    /// Uncompressed bytes consumed so far.
    #[must_use]
    pub fn uncompressed_bytes(&self) -> u64 {
        self.uncompressed_bytes
    }

    /// Where the **next** block will start.
    ///
    /// Always block-aligned — the uncompressed half is zero — because this
    /// writer only ever emits whole blocks. That is what a BAM writer records
    /// per reference to build a BAI as it goes, and it is why the counters above
    /// are tracked rather than left to the caller to recompute by re-walking the
    /// output.
    #[must_use]
    pub fn virtual_position(&self) -> VirtualPosition {
        VirtualPosition::try_from((self.compressed_bytes, 0)).unwrap_or_default()
    }

    /// Writes the EOF block and returns the underlying sink.
    ///
    /// Not optional: its absence is how `samtools` decides a file is truncated.
    pub fn finish(mut self) -> Result<W> {
        self.inner.write_all(&EOF_BLOCK)?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}
