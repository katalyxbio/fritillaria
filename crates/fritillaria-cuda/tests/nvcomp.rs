//! Differential tests for the nvCOMP codec.
//!
//! nvCOMP is a black box we do not control, which changes what these tests are
//! for. The kernel tests ask "is our DEFLATE implementation correct?"; these ask
//! **"does our plumbing hand nvCOMP the right pointers, and does the result
//! still satisfy the `BlockCodec` contract?"** Almost every way this can break
//! is a layout mistake on our side — a chunk pointer off by the gzip header, an
//! unaligned input, a checksum computed over the wrong extent — and all of them
//! produce plausible-looking wrong bytes rather than an error.
//!
//! So the CPU reference is the oracle throughout, and where a device is present
//! our own kernel is a second one. Three independent implementations agreeing
//! byte-for-byte is a much stronger statement than either pair.
//!
//! # Skipping
//!
//! Absent nvCOMP is normal on a development machine and reported with `NOTE:`,
//! which `scripts/colab_job.py` treats as a legitimately absent capability.
//! But if `FRITILLARIA_NVCOMP_LIB` is **set**, someone deliberately pointed us
//! at a library, so failing to load it is a hard error rather than a skip —
//! otherwise a broken remote bootstrap would report as a clean pass, which is
//! the exact failure this project has already hit three times (see CLAUDE.md).

#![cfg(feature = "nvcomp")]

use std::path::PathBuf;

use fritillaria_bgzf::{BgzfWriter, CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, DeviceBlockCodec, DeviceInflateBatch, Error, InflateBatch};
use fritillaria_cuda::nvcomp::ffi::LIB_PATH_ENV;
use fritillaria_cuda::{CudaCodec, NvcompCodec};

fn codec() -> Option<NvcompCodec> {
    match NvcompCodec::new() {
        Ok(codec) => Some(codec),
        Err(err) => {
            // An explicit path that does not load is a broken install, not an
            // absent capability. Fail loudly rather than let a green run mean
            // "nvCOMP was never exercised".
            //
            // Gated on a device actually being present: the same env var is set
            // on this project's GPU-less development machine, where failing
            // here would only be reporting "no GPU" twice.
            assert!(
                std::env::var_os(LIB_PATH_ENV).is_none()
                    || !fritillaria_cuda::device_is_available(),
                "{LIB_PATH_ENV} is set and a device is present, but nvCOMP \
                 would not load: {err}"
            );
            eprintln!("NOTE: nvCOMP or CUDA unavailable on this machine ({err})");
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

fn multiblock_bam() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/htslib_multiblock.bam");
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The core assertion: nvCOMP's output equals the CPU reference, on both the
/// host and the device path.
fn assert_matches_cpu(codec: &NvcompCodec, data: &[u8], what: &str) {
    let spans = discover_blocks(data, 0).unwrap();

    let mut expected = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(data, &spans, &mut expected)
        .unwrap_or_else(|e| panic!("CPU reference failed on {what}: {e:?}"));

    let mut host = InflateBatch::new();
    BlockCodec::inflate_batch(codec, data, &spans, &mut host)
        .unwrap_or_else(|e| panic!("nvcomp host inflate failed on {what}: {e:?}"));
    assert_eq!(
        host.data(),
        expected.data(),
        "host payload differs on {what}"
    );
    assert_eq!(
        host.offsets(),
        expected.offsets(),
        "host block layout differs on {what}"
    );

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(data, &spans, &mut device)
        .unwrap_or_else(|e| panic!("nvcomp device inflate failed on {what}: {e:?}"));
    let actual = device
        .to_host()
        .unwrap_or_else(|e| panic!("copying back failed on {what}: {e:?}"));

    assert_eq!(
        actual.data(),
        expected.data(),
        "device payload differs on {what}"
    );
    assert_eq!(
        actual.offsets(),
        expected.offsets(),
        "device block layout differs on {what} — records spanning a boundary \
         would land in the wrong place"
    );
    assert_eq!(device.byte_len(), expected.data().len());
}

#[test]
fn matches_the_cpu_reference() {
    let Some(codec) = codec() else { return };

    assert_matches_cpu(&codec, &bgzf(&[b"hello world"]), "one small block");
    assert_matches_cpu(
        &codec,
        &bgzf(&[b"alpha", b"beta", b"gamma"]),
        "three blocks",
    );
    assert_matches_cpu(&codec, &bgzf(&[b""]), "an empty block");

    // Highly compressible: dynamic Huffman with long back-references.
    let repetitive = vec![b'A'; 60_000];
    assert_matches_cpu(&codec, &bgzf(&[&repetitive]), "a repetitive block");

    // Incompressible: forces stored blocks.
    let random: Vec<u8> = (0..50_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    assert_matches_cpu(&codec, &bgzf(&[&random]), "an incompressible block");

    // Mixed types in one launch: nvCOMP dispatches per chunk, so a batch that
    // is uniform proves much less than one that is not.
    assert_matches_cpu(
        &codec,
        &bgzf(&[&repetitive, &random, b"", b"tiny"]),
        "mixed block types",
    );
}

#[test]
fn matches_the_cpu_reference_across_many_blocks() {
    // Where this class of library actually breaks: the concatenated buffer must
    // be seamless across every boundary, because BAM records span them.
    let Some(codec) = codec() else { return };

    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(997);
    writer.write_data(&payload).unwrap();
    let data = writer.finish().unwrap();

    let spans = discover_blocks(&data, 0).unwrap();
    assert!(spans.len() > 100, "test needs many blocks to be meaningful");

    assert_matches_cpu(&codec, &data, "200 KiB across ~200 blocks");
}

#[test]
fn matches_the_cpu_reference_on_a_real_htslib_bam() {
    // Self-generated fixtures share our assumptions; this one does not. It is
    // also the only case here with realistic block sizes and compression.
    let Some(codec) = codec() else { return };

    let data = multiblock_bam();
    assert_matches_cpu(&codec, &data, "htslib_multiblock.bam");
}

#[test]
fn agrees_with_our_own_kernel() {
    // A third independent implementation. If nvCOMP and our kernel agree with
    // each other *and* with miniz_oxide, a shared misunderstanding is about as
    // close to ruled out as this can get.
    let Some(nvcomp) = codec() else { return };
    let Ok(cuda) = CudaCodec::new() else {
        eprintln!("NOTE: no CUDA device for the cross-codec comparison");
        return;
    };

    let data = multiblock_bam();
    let spans = discover_blocks(&data, 0).unwrap();

    let mut theirs = DeviceInflateBatch::new();
    nvcomp
        .inflate_batch_device(&data, &spans, &mut theirs)
        .unwrap();

    let mut ours = DeviceInflateBatch::new();
    cuda.inflate_batch_device(&data, &spans, &mut ours).unwrap();

    assert_eq!(
        theirs.to_host().unwrap().data(),
        ours.to_host().unwrap().data(),
        "nvcomp and our inflate kernel disagree"
    );
    assert_eq!(theirs.offsets(), ours.offsets());
}

#[test]
fn verification_is_not_skipped() {
    // nvCOMP's deflate path never sees the gzip trailer and its own docs say it
    // validates only lightly. Verification is therefore entirely ours, and this
    // pins that it actually happens.
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
fn handles_a_batch_that_inflates_to_nothing() {
    // An EOF-only file: every chunk declares a zero-byte output buffer, which
    // is the sort of degenerate input a batched C API is most likely to reject.
    let Some(codec) = codec() else { return };

    let data = bgzf(&[]);
    let spans = discover_blocks(&data, 0).unwrap();
    assert_eq!(spans.len(), 1);

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    assert_eq!(device.len(), 1, "the EOF block is present");
    assert_eq!(device.byte_len(), 0, "and inflates to nothing");
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
fn a_failed_batch_does_not_leave_stale_output() {
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
        "a failed inflate must leave the batch empty"
    );
}

#[test]
fn the_alignment_assumptions_still_hold() {
    // These three values decide the shape of the implementation, so they are
    // asserted rather than assumed. An output alignment above 1 would mean
    // nvCOMP cannot write dense blocks, and dense is not negotiable — a BAM
    // record spanning a block boundary has to be contiguous.
    let Some(codec) = codec() else { return };

    let alignments = codec.context().alignments();
    assert_eq!(
        alignments.output, 1,
        "nvCOMP now needs padded output; dense block packing would require a \
         compaction pass (see the fritillaria_cuda::nvcomp docs)"
    );
    assert!(
        alignments.input.is_power_of_two(),
        "input alignment {} is not a power of two, so next_multiple_of is the \
         wrong staging rule",
        alignments.input
    );
    assert!(codec.context().nvcomp_version() >= 5300);
}

#[test]
fn restages_payloads_that_are_not_aligned() {
    // The gather path, exercised deliberately. A BGZF payload begins 18 bytes
    // into its member, so a file whose blocks land on odd offsets produces
    // chunk pointers nvCOMP will not accept unmodified. This asserts the
    // restaging is both triggered and correct.
    let Some(codec) = codec() else { return };

    let payload: Vec<u8> = (0..40_000u32).map(|i| (i % 253) as u8).collect();
    let mut writer = BgzfWriter::new(Vec::new()).with_payload_size(101);
    writer.write_data(&payload).unwrap();
    let data = writer.finish().unwrap();

    let spans = discover_blocks(&data, 0).unwrap();
    let align = codec.context().alignments().input.max(1);
    let misaligned = spans
        .iter()
        .filter(|s| s.payload_start % align != 0)
        .count();
    assert!(
        align == 1 || misaligned > 0,
        "fixture was meant to produce unaligned payloads but all {} are aligned",
        spans.len()
    );

    assert_matches_cpu(&codec, &data, "many small, unaligned blocks");
}

#[test]
fn output_reports_the_device_it_lives_on() {
    let Some(codec) = codec() else { return };

    let data = bgzf(&[b"payload"]);
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    assert_eq!(device.device_ordinal(), Some(codec.device_ordinal()));
}

#[test]
fn a_consumer_can_order_on_the_ready_event() {
    // The handoff a GPU consumer actually uses: wait on the event from its own
    // stream instead of synchronising the host.
    let Some(codec) = codec() else { return };

    // Big enough that the launch is not instantaneous, but within what
    // `BgzfWriter` accepts: a block's *compressed* form has to fit in 64 KiB
    // alongside its header and trailer, so the payload cap is below
    // `MAX_BLOCK_SIZE`.
    let payload = vec![b'Z'; 60_000];
    let data = bgzf(&[&payload]);
    let spans = discover_blocks(&data, 0).unwrap();

    let mut device = DeviceInflateBatch::new();
    codec
        .inflate_batch_device(&data, &spans, &mut device)
        .unwrap();

    let event = NvcompCodec::ready_event(&device).expect("a non-empty batch has a ready event");
    let consumer = codec
        .context()
        .raw_context()
        .new_stream()
        .expect("creating a consumer stream");
    consumer.wait(event).expect("ordering against inflate");
    consumer.synchronize().unwrap();

    assert_eq!(device.to_host().unwrap().data().len(), payload.len());
}
