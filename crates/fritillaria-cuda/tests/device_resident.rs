//! Differential tests for the device-resident inflate path.
//!
//! The claim under test is narrow and total: **inflating to device memory must
//! produce exactly what inflating to host memory produces.** Staying on the
//! device is a transport decision, not a semantic one — same kernel, same
//! verification, same bytes, same block layout.
//!
//! So every assertion here copies the device batch back with `to_host()` and
//! compares it against the CPU reference. That round trip is precisely what the
//! device path exists to avoid in production; here it is the measuring
//! instrument.
//!
//! Needs a real device, so these **skip** rather than fail when none is
//! present. `scripts/colab_job.py` treats a skip on a GPU VM as a failure, so
//! skipping cannot quietly hide a broken path.

#![cfg(feature = "cuda")]

use fritillaria_bgzf::{BgzfWriter, CpuCodec, discover_blocks};
use fritillaria_core::{
    BlockCodec, DeviceAlloc, DeviceBlockCodec, DeviceInflateBatch, Error, InflateBatch,
};
use fritillaria_cuda::cudarc::driver::CudaContext as DriverContext;
use fritillaria_cuda::{CudaAlloc, CudaCodec};

fn codec() -> Option<CudaCodec> {
    match CudaCodec::new() {
        Ok(codec) => Some(codec),
        Err(err) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

/// Encodes payloads as one BGZF file, one block each.
fn bgzf(payloads: &[&[u8]]) -> Vec<u8> {
    let mut writer = BgzfWriter::new(Vec::new());
    for payload in payloads {
        writer.write_block(payload).unwrap();
    }
    writer.finish().unwrap()
}

/// The core assertion: device output, copied back, equals the CPU reference.
fn assert_device_matches_cpu(codec: &CudaCodec, data: &[u8], what: &str) {
    let spans = discover_blocks(data, 0).unwrap();

    let mut expected = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(data, &spans, &mut expected)
        .unwrap_or_else(|e| panic!("CPU reference failed on {what}: {e:?}"));

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(data, &spans, &mut device)
        .unwrap_or_else(|e| panic!("device inflate failed on {what}: {e:?}"));

    let actual = device
        .to_host()
        .unwrap_or_else(|e| panic!("copying back failed on {what}: {e:?}"));

    assert_eq!(actual.data(), expected.data(), "payload differs on {what}");
    assert_eq!(
        actual.offsets(),
        expected.offsets(),
        "block layout differs on {what} — records spanning a boundary would \
         land in the wrong place"
    );
    assert_eq!(
        device.len(),
        expected.len(),
        "block count differs on {what}"
    );
    assert_eq!(
        device.byte_len(),
        expected.data().len(),
        "reported byte length differs on {what}"
    );
}

#[test]
fn matches_the_cpu_reference() {
    let Some(codec) = codec() else { return };

    assert_device_matches_cpu(&codec, &bgzf(&[b"hello world"]), "one small block");
    assert_device_matches_cpu(
        &codec,
        &bgzf(&[b"alpha", b"beta", b"gamma"]),
        "three blocks",
    );
    assert_device_matches_cpu(&codec, &bgzf(&[b""]), "an empty block");

    // Highly compressible: dynamic Huffman with long back-references.
    let repetitive = vec![b'A'; 60_000];
    assert_device_matches_cpu(&codec, &bgzf(&[&repetitive]), "a repetitive block");

    // Incompressible: forces stored blocks.
    let random: Vec<u8> = (0..50_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    assert_device_matches_cpu(&codec, &bgzf(&[&random]), "an incompressible block");

    // Mixed block types in a single launch — different threads take different
    // paths through the kernel.
    assert_device_matches_cpu(
        &codec,
        &bgzf(&[&repetitive, &random, b"", b"tiny"]),
        "mixed block types",
    );
}

#[test]
fn matches_the_cpu_reference_across_many_blocks() {
    // Where this class of library actually breaks: the concatenated buffer must
    // be seamless across every block boundary, because BAM records span them.
    let Some(codec) = codec() else { return };

    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(997);
    writer.write_data(&payload).unwrap();
    let data = writer.finish().unwrap();

    let spans = discover_blocks(&data, 0).unwrap();
    assert!(spans.len() > 100, "test needs many blocks to be meaningful");

    assert_device_matches_cpu(&codec, &data, "200 KiB across ~200 blocks");
}

#[test]
fn handles_a_batch_that_inflates_to_nothing() {
    // A file of only EOF blocks inflates to zero bytes. The device allocation
    // is padded to one byte because a zero-sized allocation is invalid, so this
    // pins that the *logical* length stays zero and does not leak the pad.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[]); // just the EOF block
    let spans = discover_blocks(&data, 0).unwrap();
    assert_eq!(spans.len(), 1);

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    assert_eq!(device.len(), 1, "the EOF block is present");
    assert_eq!(device.byte_len(), 0, "and inflates to nothing");
    assert_eq!(device.offsets(), &[0, 0]);
    assert!(device.to_host().unwrap().data().is_empty());
}

#[test]
fn an_empty_span_list_yields_an_empty_batch() {
    let Some(codec) = codec() else { return };

    let mut device = DeviceInflateBatch::new();
    codec.inflate_batch_device(&[], &[], &mut device).unwrap();

    assert!(device.is_empty());
    assert!(device.data().is_none());
}

#[test]
fn verification_is_not_skipped_on_the_device_path() {
    // The contract says staying on the device is never a licence to hand back
    // unverified bytes. Each of these corrupts one input to the check.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[b"the quick brown fox jumps over the lazy dog"]);

    let mut spans = discover_blocks(&data, 0).unwrap();
    spans[0].crc32 ^= 0xffff_ffff;
    let mut device = DeviceInflateBatch::new();
    assert!(
        matches!(
            codec.inflate_batch_device(&data, &spans, &mut device),
            Err(Error::ChecksumMismatch { .. })
        ),
        "a bad CRC must fail the batch"
    );

    let mut spans = discover_blocks(&data, 0).unwrap();
    spans[0].isize += 1;
    let mut device = DeviceInflateBatch::new();
    assert!(
        matches!(
            codec.inflate_batch_device(&data, &spans, &mut device),
            Err(Error::SizeMismatch { .. })
        ),
        "a bad ISIZE must fail the batch"
    );

    let mut corrupt = data.clone();
    let spans = discover_blocks(&data, 0).unwrap();
    corrupt[spans[0].payload_start + 2] ^= 0x01;
    let mut device = DeviceInflateBatch::new();
    assert!(
        matches!(
            codec.inflate_batch_device(&corrupt, &spans, &mut device),
            Err(Error::ChecksumMismatch { .. }
                | Error::SizeMismatch { .. }
                | Error::Inflate { .. })
        ),
        "a corrupted payload must fail the batch"
    );
}

#[test]
fn a_failed_batch_does_not_leave_stale_output() {
    // A caller reusing a batch across calls must never read the previous
    // batch's bytes after a failure.
    let Some(codec) = codec() else { return };

    let good = bgzf(&[b"first payload"]);
    let spans = discover_blocks(&good, 0).unwrap();
    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&good, &spans, &mut device)
        .unwrap();
    assert_eq!(device.to_host().unwrap().data(), b"first payload");

    let bad = bgzf(&[b"second payload"]);
    let mut bad_spans = discover_blocks(&bad, 0).unwrap();
    bad_spans[0].crc32 ^= 0xffff_ffff;
    assert!(
        codec
            .inflate_batch_device(&bad, &bad_spans, &mut device)
            .is_err()
    );

    assert!(
        device.is_empty(),
        "a failed inflate must leave the batch empty, not holding the \
         previous call's payload"
    );
}

#[test]
fn reuses_the_output_batch_across_calls() {
    let Some(codec) = codec() else { return };
    let mut device = DeviceInflateBatch::new();

    for payload in [&b"first"[..], &b"second"[..], &b"third"[..]] {
        let data = bgzf(&[payload]);
        let spans = discover_blocks(&data, 0).unwrap();
        codec
            .inflate_batch_device(&data, &spans, &mut device)
            .unwrap();

        assert_eq!(
            device.to_host().unwrap().block(0),
            Some(payload),
            "stale device buffer from the prior batch"
        );
    }
}

#[test]
fn block_ranges_index_the_device_buffer() {
    // What a kernel indexing individual blocks relies on.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[b"alpha", b"", b"beta"]);
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    let bytes = device.data().unwrap().to_vec().unwrap();
    assert_eq!(&bytes[device.block_range(0).unwrap()], b"alpha");
    assert_eq!(&bytes[device.block_range(1).unwrap()], b"");
    assert_eq!(&bytes[device.block_range(2).unwrap()], b"beta");
}

#[test]
fn output_reports_the_device_it_lives_on() {
    // A consumer holding its own context must be able to check this before
    // launching: memory from another device is not addressable.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[b"payload"]);
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    assert_eq!(device.device_ordinal(), Some(codec.device_ordinal()));
    assert_eq!(codec.device_ordinal(), 0, "tests open device 0");
}

#[test]
fn the_backend_can_recover_its_own_allocation() {
    // The seam that keeps CUDA out of the format crates: they hold an opaque
    // DeviceBuffer, and only this crate downcasts back to a real device slice.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[b"payload"]);
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    let alloc = device
        .data()
        .unwrap()
        .alloc()
        .as_any()
        .downcast_ref::<CudaAlloc>()
        .expect("the CUDA backend must be able to recover its own allocation");

    assert_eq!(alloc.byte_len(), b"payload".len());
    assert_eq!(
        alloc.slice().len(),
        b"payload".len(),
        "a kernel needs the real slice, not a copy"
    );
}

#[test]
fn allocates_in_a_caller_supplied_context() {
    // The property that makes this embeddable: a GPU tool already owns a
    // context, and output allocated in a *different* one would need a peer copy
    // to be readable by its kernels — the exact transfer this path removes.
    if !fritillaria_cuda::device_is_available() {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }
    let their_ctx = DriverContext::new(0).unwrap();
    let their_stream = their_ctx.new_stream().unwrap();
    let codec = CudaCodec::with_context(their_ctx.clone(), their_stream.clone()).unwrap();

    let data = bgzf(&[b"allocated in the caller's context"]);
    let spans = discover_blocks(&data, 0).unwrap();
    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    let alloc = device
        .data()
        .unwrap()
        .alloc()
        .as_any()
        .downcast_ref::<CudaAlloc>()
        .unwrap();

    assert_eq!(
        alloc.stream().context(),
        &their_ctx,
        "output must live in the caller's context, not one of ours"
    );
    assert_eq!(
        device.to_host().unwrap().data(),
        b"allocated in the caller's context"
    );
}

#[test]
fn accepts_any_stream_from_the_same_context() {
    // `DriverContext::new(ordinal)` retains the device's *primary* context, so
    // two calls for the same device return the same context — this is not two
    // contexts, and a stream from either is legitimately usable with the other.
    // An earlier version of this test assumed otherwise and failed on a T4.
    if !fritillaria_cuda::device_is_available() {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }
    let first = DriverContext::new(0).unwrap();
    let second = DriverContext::new(0).unwrap();
    assert_eq!(
        first, second,
        "new(0) twice must retain the same primary context"
    );

    let stream = second.new_stream().unwrap();
    assert!(
        CudaCodec::with_context(first, stream).is_ok(),
        "a stream from the same primary context must be accepted"
    );
}

#[test]
fn rejects_a_stream_from_another_device() {
    // Silently accepting this would produce memory the caller cannot address
    // from the stream they handed us.
    //
    // Only reachable with two devices: on one GPU every context is the same
    // primary context (see above), so the mismatch cannot be constructed. Not
    // a "SKIP:" — that marker means the driver was unusable and fails the
    // remote job. This is a capability the VM legitimately lacks.
    if !fritillaria_cuda::device_is_available() {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }
    let devices = DriverContext::device_count().unwrap_or(0);
    if devices < 2 {
        eprintln!("NOTE: cross-device rejection needs 2 GPUs, found {devices}");
        return;
    }

    let ctx_a = DriverContext::new(0).unwrap();
    let ctx_b = DriverContext::new(1).unwrap();
    let stream_b = ctx_b.new_stream().unwrap();

    assert!(
        CudaCodec::with_context(ctx_a, stream_b).is_err(),
        "a stream from another device's context must be rejected, not used"
    );
}

#[test]
fn a_consumer_can_order_on_the_ready_event() {
    // The zero-stall handoff: a consumer on its own stream waits on our event
    // rather than synchronising the host.
    if !fritillaria_cuda::device_is_available() {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }
    let ctx = DriverContext::new(0).unwrap();
    let our_stream = ctx.new_stream().unwrap();
    let codec = CudaCodec::with_context(ctx.clone(), our_stream).unwrap();

    let payload: Vec<u8> = (0..80_000u32).map(|i| (i % 251) as u8).collect();
    let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(4096);
    writer.write_data(&payload).unwrap();
    let data = writer.finish().unwrap();
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    // A consumer's own stream, distinct from the one the inflate ran on.
    let their_stream = ctx.new_stream().unwrap();
    let event = CudaCodec::ready_event(&device).expect("a non-empty batch must expose its event");
    their_stream
        .wait(event)
        .expect("a consumer must be able to order against inflate completion");

    // Reading through the consumer's stream after the wait must see complete,
    // correct data — that is what the ordering guarantee means.
    let alloc = device
        .data()
        .unwrap()
        .alloc()
        .as_any()
        .downcast_ref::<CudaAlloc>()
        .unwrap();
    let seen = their_stream.clone_dtoh(alloc.slice()).unwrap();
    assert_eq!(&seen[..payload.len()], payload.as_slice());
}

#[test]
fn an_empty_batch_has_no_ready_event() {
    if !fritillaria_cuda::device_is_available() {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }
    let device = DeviceInflateBatch::new();
    assert!(CudaCodec::ready_event(&device).is_none());
}

#[test]
fn device_and_host_paths_agree_on_a_real_htslib_bam() {
    // Synthetic fixtures come from our own writer and can only prove
    // self-consistency. This one was written by htslib.
    let Some(codec) = codec() else { return };

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/htslib_multiblock.bam"
    );
    let Ok(data) = std::fs::read(path) else {
        eprintln!("SKIP: {path} not found");
        return;
    };

    assert_device_matches_cpu(&codec, &data, "htslib_multiblock.bam");
}
