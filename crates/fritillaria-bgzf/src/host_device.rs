//! A [`DeviceBlockCodec`] that keeps everything in host memory.
//!
//! # Why this exists
//!
//! All host-side development happens on a machine with no GPU, and renting one
//! costs money and minutes per iteration. Without a stand-in, the entire
//! device-resident half of the design — the codec seam, batch layout
//! invariants, buffer ownership, and every format crate that will consume
//! device columns — would be untestable locally and only exercised on a Colab
//! VM.
//!
//! So this implements the device trait with [`CpuCodec`] behind it and host
//! memory pretending to be device memory. Everything about the seam is real
//! except the transfer.
//!
//! # What it does and does not prove
//!
//! It proves the *shape* is workable: that a device codec can be written
//! against these traits, that the layout invariants hold, and that
//! [`DeviceInflateBatch::to_host`] round-trips. It proves nothing about a
//! kernel, a stream, or a pointer, and it must never be mistaken for a
//! performance path — it is strictly slower than [`CpuCodec`], since it does the
//! same work and then wraps the result.

use fritillaria_core::device::testing::HostAlloc;
use fritillaria_core::{
    BlockCodec, BlockSpan, DeviceBlockCodec, DeviceBuffer, DeviceInflateBatch, InflateBatch, Result,
};

use crate::cpu::CpuCodec;

/// A device codec backed by host memory, for testing the device seam.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDeviceCodec {
    ordinal: i32,
}

impl HostDeviceCodec {
    /// Creates the codec, reporting device 0.
    #[must_use]
    pub const fn new() -> Self {
        Self { ordinal: 0 }
    }

    /// Creates the codec reporting a specific device ordinal.
    ///
    /// Lets tests cover consumer-side device checks that would otherwise need
    /// two GPUs to reach.
    #[must_use]
    pub const fn on_device(ordinal: i32) -> Self {
        Self { ordinal }
    }
}

impl DeviceBlockCodec for HostDeviceCodec {
    fn name(&self) -> &'static str {
        "host-device-stub"
    }

    fn device_ordinal(&self) -> i32 {
        self.ordinal
    }

    fn inflate_batch_device(
        &self,
        batch: &[u8],
        spans: &[BlockSpan],
        out: &mut DeviceInflateBatch,
    ) -> Result<()> {
        // Verification lives in CpuCodec and is not bypassed here: the device
        // contract requires CRC32 and ISIZE checks just as the host one does.
        let mut host = InflateBatch::new();
        CpuCodec::new().inflate_batch(batch, spans, &mut host)?;

        let offsets = host.offsets().to_vec();
        let data = DeviceBuffer::new(Box::new(HostAlloc::on_device(
            host.data().to_vec(),
            self.ordinal,
        )));
        out.adopt(data, offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::discover_blocks;
    use crate::write::BgzfWriter;
    use fritillaria_core::Error;

    fn encode(payloads: &[&[u8]]) -> Vec<u8> {
        let mut writer = BgzfWriter::new(Vec::new());
        for payload in payloads {
            writer.write_block(payload).unwrap();
        }
        writer.finish().unwrap()
    }

    /// The differential assertion the whole design turns on: device output
    /// copied back must be byte-identical to the host path, layout included.
    #[test]
    fn device_output_matches_the_host_codec_exactly() {
        let data = encode(&[b"alpha", b"beta", b"gamma"]);
        let spans = discover_blocks(&data, 0).unwrap();

        let mut host = InflateBatch::new();
        CpuCodec::new()
            .inflate_batch(&data, &spans, &mut host)
            .unwrap();

        let mut device = DeviceInflateBatch::new();
        HostDeviceCodec::new()
            .inflate_batch_device(&data, &spans, &mut device)
            .unwrap();
        let copied = device.to_host().unwrap();

        assert_eq!(copied.data(), host.data());
        assert_eq!(copied.offsets(), host.offsets());
        assert_eq!(device.len(), host.len());
    }

    #[test]
    fn survives_records_spanning_many_blocks() {
        // The seam this class of library actually breaks at. Small payloads
        // force many blocks, so the concatenated device buffer must be
        // contiguous across every boundary.
        let payload: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(997);
        writer.write_data(&payload).unwrap();
        let data = writer.finish().unwrap();

        let spans = discover_blocks(&data, 0).unwrap();
        assert!(spans.len() > 10, "test needs many blocks to be meaningful");

        let mut device = DeviceInflateBatch::new();
        HostDeviceCodec::new()
            .inflate_batch_device(&data, &spans, &mut device)
            .unwrap();

        assert_eq!(device.to_host().unwrap().data(), payload.as_slice());
    }

    #[test]
    fn block_ranges_index_the_device_buffer() {
        // What a kernel indexing blocks directly would rely on.
        let data = encode(&[b"alpha", b"beta"]);
        let spans = discover_blocks(&data, 0).unwrap();

        let mut device = DeviceInflateBatch::new();
        HostDeviceCodec::new()
            .inflate_batch_device(&data, &spans, &mut device)
            .unwrap();

        let bytes = device.data().unwrap().to_vec().unwrap();
        assert_eq!(&bytes[device.block_range(0).unwrap()], b"alpha");
        assert_eq!(&bytes[device.block_range(1).unwrap()], b"beta");
        // The EOF block: empty, present, and not confusable with absent.
        assert_eq!(device.block_range(2), Some(9..9));
    }

    #[test]
    fn verification_is_not_skipped_on_the_device_path() {
        // The device contract requires CRC32/ISIZE checking. Staying on the
        // device must never become a way to get unverified bytes.
        let data = encode(&[b"payload"]);
        let mut spans = discover_blocks(&data, 0).unwrap();
        spans[0].crc32 ^= 0xffff_ffff;

        let mut device = DeviceInflateBatch::new();
        assert!(matches!(
            HostDeviceCodec::new()
                .inflate_batch_device(&data, &spans, &mut device)
                .unwrap_err(),
            Error::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn reuses_the_output_batch_across_calls() {
        let codec = HostDeviceCodec::new();
        let mut out = DeviceInflateBatch::new();

        for payload in [&b"first"[..], &b"second"[..]] {
            let data = encode(&[payload]);
            let spans = discover_blocks(&data, 0).unwrap();
            codec.inflate_batch_device(&data, &spans, &mut out).unwrap();

            let bytes = out.to_host().unwrap();
            assert_eq!(
                bytes.block(0),
                Some(payload),
                "stale device buffer from the prior batch"
            );
        }
    }

    #[test]
    fn reports_the_device_it_allocated_on() {
        let data = encode(&[b"x"]);
        let spans = discover_blocks(&data, 0).unwrap();

        let mut out = DeviceInflateBatch::new();
        HostDeviceCodec::on_device(3)
            .inflate_batch_device(&data, &spans, &mut out)
            .unwrap();

        assert_eq!(out.device_ordinal(), Some(3));
    }

    #[test]
    fn an_empty_input_yields_an_empty_batch() {
        let mut out = DeviceInflateBatch::new();
        HostDeviceCodec::new()
            .inflate_batch_device(&[], &[], &mut out)
            .unwrap();

        assert!(out.is_empty());
        assert_eq!(out.byte_len(), 0);
        assert!(out.to_host().unwrap().is_empty());
    }
}
