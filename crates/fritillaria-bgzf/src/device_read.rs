//! A streaming BGZF reader whose output stays on the device.
//!
//! [`BgzfReader`](crate::BgzfReader) serves *bytes* and therefore has to bring
//! them back to the host. This one serves *batches*, which never leave the
//! device — it is what turns `BamDecoder` from something you hand a batch into
//! something you can point at a file.
//!
//! # The seam this exists for
//!
//! A BAM record can span BGZF blocks, and an ultra-long ONT read can be larger
//! than a whole batch of them. So a batch usually ends part-way through a
//! record, and the leftover has to reach the next batch contiguously.
//!
//! The trick is to carry **compressed blocks**, not decompressed bytes. When the
//! caller reports where it stopped, the reader finds the block containing that
//! offset and re-includes that block and everything after it in the next batch.
//! No device-to-device copy, no second buffer: the overlap is re-inflated, and
//! it is at most one record's worth of blocks — typically one block out of 256.
//!
//! It also means a record larger than a batch resolves itself. Each round
//! prepends the carried blocks and appends a fresh read, so the window grows
//! until the record fits. That is the "must grow its buffer to fit a whole
//! record" rule from CLAUDE.md, and it is handled here rather than left to
//! every caller.
//!
//! # Ownership: why batches are handed over rather than lent
//!
//! [`next_batch`](DeviceBgzfReader::next_batch) yields an owned
//! [`DeviceInflateBatch`]. That is deliberate and costs an allocation per batch.
//! Columns decoded from a batch point *into* its buffer, so a reader that kept
//! the batch and reused it would free those bytes underneath a consumer still
//! reading them — silent corruption, the one failure mode this design can
//! produce. Handing the batch over makes the lifetime the caller's problem in a
//! way the borrow checker can see.

use std::io::{self, Read};

use fritillaria_core::{
    BlockSpan, DeviceBlockCodec, DeviceInflateBatch, Error, MAX_BLOCK_SIZE, Result,
};

use crate::discover::BlockDiscovery;
use crate::read::DEFAULT_BLOCKS_PER_BATCH;

/// One batch of inflated blocks, still on the device.
#[derive(Debug)]
pub struct DeviceBatch {
    /// The inflated bytes. Owned by the caller — see the module docs.
    pub data: DeviceInflateBatch,
    /// Where the caller should begin reading.
    ///
    /// Non-zero when a record was carried from the previous batch: the carried
    /// blocks are re-inflated at the front, and this is where the unconsumed
    /// record begins within them.
    pub start: usize,
}

/// Reads a BGZF stream into device memory, a batch of blocks at a time.
#[derive(Debug)]
pub struct DeviceBgzfReader<R, C> {
    inner: R,
    codec: C,
    blocks_per_batch: usize,

    /// Compressed bytes of the batch most recently handed out.
    compressed: Vec<u8>,
    /// Absolute file offset of `compressed[0]`.
    batch_offset: u64,
    /// Compressed bytes to lead the next batch: a partial trailing block, plus
    /// any whole blocks carried back by [`DeviceBgzfReader::carry_from`].
    carry: Vec<u8>,
    /// Uncompressed offset within the carried blocks where reading resumes.
    carry_start: usize,

    spans: Vec<BlockSpan>,
    inner_exhausted: bool,
}

impl<R: Read, C: DeviceBlockCodec> DeviceBgzfReader<R, C> {
    /// Creates a reader over `inner`, inflating with `codec`.
    pub fn new(inner: R, codec: C) -> Self {
        Self {
            inner,
            codec,
            blocks_per_batch: DEFAULT_BLOCKS_PER_BATCH,
            compressed: Vec::new(),
            batch_offset: 0,
            carry: Vec::new(),
            carry_start: 0,
            spans: Vec::new(),
            inner_exhausted: false,
        }
    }

    /// Sets how many blocks are inflated per batch.
    ///
    /// This is a floor, not a cap: a record larger than the batch grows the
    /// window until it fits.
    ///
    /// **It budgets *compressed* bytes**, at the 64 KiB a block occupies in the
    /// worst case, so the blocks actually read scale with how well the data
    /// compresses. On a BAM at the 3.37x measured on real WGS that is roughly
    /// three blocks per unit; on a BCF of genotypes, which compresses **20x**,
    /// it is closer to twenty. Asking for 8 on a 182 KB BCF reads the whole
    /// file in one batch.
    ///
    /// That surprised a test rather than a user, and it is not a bug — a fixed
    /// compressed budget is what bounds memory, which is the thing worth
    /// bounding. But it does mean the name is a request, not a count, and that
    /// a caller sizing batches for a target record count cannot get there from
    /// here without knowing the ratio.
    #[must_use]
    pub fn with_blocks_per_batch(mut self, blocks: usize) -> Self {
        self.blocks_per_batch = blocks.max(1);
        self
    }

    /// The backend actually in use, for diagnostics.
    pub fn codec_name(&self) -> &'static str {
        self.codec.name()
    }

    /// Which device batches are allocated on.
    pub fn device_ordinal(&self) -> i32 {
        self.codec.device_ordinal()
    }

    /// Consumes the reader, returning the underlying byte source.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Uncompressed offset of each block in the batch just handed out;
    /// length `n + 1`.
    fn block_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.spans.len() + 1);
        let mut acc = 0usize;
        offsets.push(0);
        for span in &self.spans {
            acc += span.isize as usize;
            offsets.push(acc);
        }
        offsets
    }

    /// Reports that everything before `tail` was consumed.
    ///
    /// `tail` is an offset into the batch most recently returned — for BAM,
    /// exactly what [`scan_records`](fritillaria_bam::scan_records) and
    /// `DeviceRecordBatch::tail` report. The block containing it, and every
    /// block after it, is re-inflated at the front of the next batch.
    ///
    /// Not calling this means the whole batch was consumed. Calling it with the
    /// batch's full length says the same thing.
    pub fn carry_from(&mut self, tail: usize) -> Result<()> {
        let offsets = self.block_offsets();
        let total = offsets.last().copied().unwrap_or(0);

        if tail > total {
            return Err(Error::InvalidDeviceBatch {
                reason: format!("carry offset {tail} is past the batch's {total} bytes"),
            });
        }
        if tail == total {
            // Nothing unconsumed; the partial-block carry from the last read
            // already leads the next batch.
            self.carry_start = 0;
            return Ok(());
        }

        // The block `tail` falls in. `partition_point` gives the first block
        // starting after `tail`, so the one before it is the container.
        let block = offsets.partition_point(|&o| o <= tail) - 1;
        let span = &self.spans[block];
        let block_start = (span.compressed_offset - self.batch_offset) as usize;

        // Everything from that block on, including the partial trailing block
        // that already sits at the end of `compressed`.
        self.carry = self.compressed[block_start..].to_vec();
        self.carry_start = tail - offsets[block];
        Ok(())
    }

    /// Inflates the next run of blocks.
    ///
    /// Returns `None` at end of stream. The returned batch is owned by the
    /// caller and must outlive anything decoded from it.
    pub fn next_batch(&mut self) -> Result<Option<DeviceBatch>> {
        loop {
            if self.inner_exhausted && self.carry.is_empty() {
                return Ok(None);
            }
            // Nothing new can be read and the window would be rebuilt
            // identically: the stream ends inside a record.
            if self.inner_exhausted
                && !self.compressed.is_empty()
                && self.carry.len() == self.compressed.len()
            {
                return Err(Error::Malformed {
                    format: "bgzf",
                    position: self.batch_offset,
                    reason: "stream ends inside a record: \
                             the trailing bytes are not a complete record"
                        .to_string(),
                });
            }

            // Carried blocks lead the new window.
            self.batch_offset += self.compressed.len() as u64;
            self.batch_offset -= self.carry.len() as u64;
            self.compressed.clear();
            self.compressed.append(&mut self.carry);

            let want = self.blocks_per_batch * MAX_BLOCK_SIZE;
            let start = self.compressed.len();
            self.compressed.resize(start + want, 0);

            let mut filled = start;
            while filled < self.compressed.len() {
                match self.inner.read(&mut self.compressed[filled..]) {
                    Ok(0) => {
                        self.inner_exhausted = true;
                        break;
                    }
                    Ok(n) => filled += n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(Error::Io(e)),
                }
            }
            self.compressed.truncate(filled);

            let mut discovery = BlockDiscovery::new(&self.compressed, self.batch_offset);
            self.spans.clear();
            for span in discovery.by_ref() {
                self.spans.push(span?);
            }
            let consumed = discovery.position();

            // Whatever follows the last complete block leads the next window.
            // At true EOF it means the file is truncated.
            self.carry.clear();
            self.carry.extend_from_slice(&self.compressed[consumed..]);
            if self.inner_exhausted && !self.carry.is_empty() {
                return Err(Error::Malformed {
                    format: "bgzf",
                    position: self.batch_offset + consumed as u64,
                    reason: format!(
                        "truncated BGZF: {} trailing bytes are not a complete block",
                        self.carry.len()
                    ),
                });
            }

            if self.spans.is_empty() {
                if self.inner_exhausted {
                    return Ok(None);
                }
                continue;
            }

            let mut data = DeviceInflateBatch::new();
            self.codec
                .inflate_batch_device(&self.compressed, &self.spans, &mut data)?;

            // A batch of only empty blocks — a BGZF EOF marker on its own.
            // Mid-stream that just means read more; at the end it means there
            // is nothing left to decode, so say so rather than handing back an
            // empty batch the caller has to special-case. A carried record
            // cannot be hiding here: its blocks would have inflated to bytes.
            if data.byte_len() == 0 {
                if self.inner_exhausted {
                    return Ok(None);
                }
                continue;
            }

            let start = std::mem::take(&mut self.carry_start);
            return Ok(Some(DeviceBatch { data, start }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_device::HostDeviceCodec;
    use crate::write::BgzfWriter;

    // A stand-in for BAM's record framing: a 4-byte little-endian length
    // prefix followed by that many payload bytes. Same shape as the thing the
    // carry logic exists for, without dragging a format crate in — this reader
    // knows nothing about BAM and neither should its tests.

    fn record(tag: u8, payload_len: usize) -> Vec<u8> {
        let mut r = (payload_len as u32).to_le_bytes().to_vec();
        r.extend(std::iter::repeat_n(tag, payload_len));
        r
    }

    /// Walks records from `start`, returning their tags and the offset of the
    /// first record the buffer cuts short.
    fn walk(buf: &[u8], start: usize) -> (Vec<u8>, usize) {
        let mut tags = Vec::new();
        let mut pos = start;
        loop {
            let Some(head) = buf.get(pos..pos + 4) else {
                return (tags, pos);
            };
            let len = u32::from_le_bytes(head.try_into().unwrap()) as usize;
            let Some(payload) = buf.get(pos + 4..pos + 4 + len) else {
                return (tags, pos);
            };
            tags.push(payload[0]);
            pos += 4 + len;
        }
    }

    fn bgzf(data: &[u8], payload: usize) -> Vec<u8> {
        let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(payload);
        writer.write_data(data).unwrap();
        writer.finish().unwrap()
    }

    /// Drives the reader the way a record consumer must: decode what is
    /// buffered, report where it stopped, repeat. This is the loop
    /// `DeviceBamReader` will encapsulate.
    fn drive(encoded: &[u8], blocks_per_batch: usize) -> Vec<u8> {
        let mut reader = DeviceBgzfReader::new(encoded, HostDeviceCodec::new())
            .with_blocks_per_batch(blocks_per_batch);
        let mut tags = Vec::new();

        while let Some(batch) = reader.next_batch().unwrap() {
            let host = batch.data.to_host().unwrap();
            let (found, tail) = walk(host.data(), batch.start);
            tags.extend(found);
            reader.carry_from(tail).unwrap();
        }
        tags
    }

    fn stream(tags: &[u8], payload_len: usize) -> (Vec<u8>, Vec<u8>) {
        let mut buf = Vec::new();
        for &t in tags {
            buf.extend(record(t, payload_len));
        }
        (buf, tags.to_vec())
    }

    #[test]
    fn recovers_every_record_at_every_batch_size() {
        let (data, expected) = stream(&[1, 2, 3, 4, 5], 40);
        // A payload size chosen so records straddle block boundaries.
        let encoded = bgzf(&data, 37);

        for blocks in [1usize, 2, 3, 8, 64] {
            assert_eq!(
                drive(&encoded, blocks),
                expected,
                "with {blocks} block(s) per batch"
            );
        }
    }

    #[test]
    fn a_record_larger_than_a_batch_grows_the_window() {
        // The ONT case: one record cannot fit in a batch, so the reader has to
        // keep carrying and appending until it does. A driver that assumed a
        // record fits in one batch would spin or lose it.
        let (data, expected) = stream(&[7, 8], 20_000);
        let encoded = bgzf(&data, 512);

        // 512-byte blocks, one per batch: each record spans ~40 of them.
        assert_eq!(drive(&encoded, 1), expected);
    }

    #[test]
    fn the_carry_re_inflates_only_the_blocks_it_needs() {
        // The carry is whole compressed blocks, so the overlap should be the
        // block holding the partial record and no more. Measured as bytes
        // re-inflated across the run.
        let (data, _) = stream(&[1, 2, 3, 4, 5, 6, 7, 8], 300);
        let encoded = bgzf(&data, 256);

        let mut reader =
            DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::new()).with_blocks_per_batch(2);
        let mut inflated_total = 0usize;
        while let Some(batch) = reader.next_batch().unwrap() {
            inflated_total += batch.data.byte_len();
            let host = batch.data.to_host().unwrap();
            let (_, tail) = walk(host.data(), batch.start);
            reader.carry_from(tail).unwrap();
        }

        // Some re-inflation is inherent; an unbounded amount would mean the
        // carry is not advancing.
        assert!(
            inflated_total < data.len() * 3,
            "re-inflated {inflated_total} bytes for a {} byte stream",
            data.len()
        );
    }

    #[test]
    fn an_empty_stream_yields_no_batches() {
        let encoded = bgzf(&[], 100);
        let mut reader = DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::new());
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn carrying_the_whole_batch_at_eof_is_an_error_not_a_loop() {
        // A tail of 0 means nothing was consumed. With more input that is legal
        // and the window grows; at EOF it means the stream ends inside a
        // record, which must be reported rather than spun on forever.
        let (data, _) = stream(&[1], 40);
        let encoded = bgzf(&data, 4096);

        let mut reader = DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::new());
        let batch = reader.next_batch().unwrap().expect("one batch");
        assert!(batch.data.byte_len() > 0);
        reader.carry_from(0).unwrap();

        let err = reader.next_batch().unwrap_err();
        assert!(
            matches!(&err, Error::Malformed { format: "bgzf", .. }),
            "got {err}"
        );
    }

    #[test]
    fn carry_past_the_end_is_rejected() {
        let (data, _) = stream(&[1], 40);
        let encoded = bgzf(&data, 4096);

        let mut reader = DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::new());
        let batch = reader.next_batch().unwrap().unwrap();
        let len = batch.data.byte_len();
        assert!(reader.carry_from(len).is_ok(), "the exact end is legal");
        assert!(reader.carry_from(len + 1).is_err());
    }

    #[test]
    fn a_truncated_block_is_an_error() {
        let (data, _) = stream(&[1, 2], 40);
        let mut encoded = bgzf(&data, 64);
        encoded.truncate(encoded.len() - 10);

        let mut reader = DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::new());
        let mut saw_error = false;
        loop {
            match reader.next_batch() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(saw_error, "truncation must not be silently accepted");
    }

    #[test]
    fn batches_report_the_codec_device() {
        let encoded = bgzf(&stream(&[1], 40).0, 4096);
        let reader = DeviceBgzfReader::new(&encoded[..], HostDeviceCodec::on_device(3));
        assert_eq!(reader.device_ordinal(), 3);
        assert_eq!(reader.codec_name(), "host-device-stub");
    }
}
