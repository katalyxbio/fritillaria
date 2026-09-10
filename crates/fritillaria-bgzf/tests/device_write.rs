//! The device BGZF writer, driven with no GPU.
//!
//! # The seam this exists for
//!
//! Everything the writer adds over the compressor is host logic: where a batch
//! splits, that a split does not move a block boundary, that the counters track,
//! that the EOF block lands. None of it needs a kernel — and this project has burned a
//! rented VM on exactly this class of mistake, when a test asserted
//! several batches and the fixture arrived in one. So `HostDeviceCompressor`
//! stands in and the whole loop is exercised locally.
//!
//! The claim that matters most here is negative: **the output must not depend on
//! the batch size.** A batch boundary always falls on a chunk boundary, so
//! compressing 40 chunks in one batch and in forty must produce identical bytes.
//! If that ever stops holding, a caller's block boundaries — which in a BAM are
//! record boundaries the GPU scan depends on — are being moved by a knob that is
//! supposed to be about memory.

use fritillaria_bgzf::{
    CpuCodec, DeviceBgzfWriter, EOF_BLOCK, HostDeviceCompressor, discover_blocks, is_eof_block,
};
use fritillaria_core::device::testing::HostAlloc;
use fritillaria_core::{BlockCodec, DeviceBuffer, InflateBatch, MAX_COMPRESSIBLE_PAYLOAD};

fn device(bytes: Vec<u8>) -> DeviceBuffer {
    HostAlloc::buffer(bytes)
}

/// Inflates a written file back to payloads and their boundaries.
fn read_back(stream: &[u8]) -> InflateBatch {
    let spans = discover_blocks(stream, 0).expect("output is not valid BGZF");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(stream, &spans, &mut out)
        .expect("output failed to inflate");
    out
}

fn writer(chunks_per_batch: usize) -> DeviceBgzfWriter<Vec<u8>, HostDeviceCompressor> {
    DeviceBgzfWriter::new(Vec::new(), HostDeviceCompressor::new())
        .with_chunks_per_batch(chunks_per_batch)
}

fn payload(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn a_written_file_reads_back_with_its_boundaries_intact() {
    let data = payload(30_000, 7);
    let bounds = vec![0, 1000, 1000, 12_345, 30_000];

    let mut w = writer(2);
    w.write_batch(&device(data.clone()), &bounds).unwrap();
    let stream = w.finish().unwrap();

    let back = read_back(&stream);
    assert_eq!(back.data(), data.as_slice());
    assert_eq!(
        back.offsets(),
        [&bounds[..], &[30_000]].concat(),
        "the caller's block boundaries must survive, plus the EOF block"
    );
}

/// The negative claim: batching is a memory knob and must not touch the output.
#[test]
fn the_output_does_not_depend_on_the_batch_size() {
    let data = payload(50_000, 3);
    let bounds: Vec<usize> = (0..=40).map(|i| i * 1250).collect();
    assert_eq!(*bounds.last().unwrap(), data.len());

    let mut reference = None;
    let mut batch_counts = Vec::new();
    for chunks_per_batch in [1usize, 2, 3, 7, 39, 40, 41, 1024] {
        let mut w = writer(chunks_per_batch);
        w.write_batch(&device(data.clone()), &bounds).unwrap();
        batch_counts.push((chunks_per_batch, w.batches_run()));
        let stream = w.finish().unwrap();

        match &reference {
            None => reference = Some(stream),
            Some(expected) => assert_eq!(
                &stream, expected,
                "output changed at {chunks_per_batch} chunks per batch; a memory \
                 knob moved a block boundary"
            ),
        }
    }

    // Without this the claim above is vacuous: a writer that ignored the knob
    // and always used one batch would satisfy it trivially.
    assert_eq!(
        batch_counts,
        vec![
            (1, 40),
            (2, 20),
            (3, 14),
            (7, 6),
            (39, 2),
            (40, 1),
            (41, 1),
            (1024, 1)
        ],
        "the batch size knob did not actually change how the work was split"
    );

    // And it is 40 blocks plus EOF however it was batched.
    assert_eq!(read_back(reference.as_ref().unwrap()).len(), 41);
}

#[test]
fn every_file_ends_with_the_eof_block() {
    let mut w = writer(4);
    w.write_batch(&device(b"data".to_vec()), &[0, 4]).unwrap();
    let stream = w.finish().unwrap();

    assert!(
        is_eof_block(&stream[stream.len() - EOF_BLOCK.len()..]),
        "without the EOF block every tool reports truncation"
    );
}

#[test]
fn an_empty_file_is_just_the_eof_block() {
    let stream = writer(4).finish().unwrap();
    assert_eq!(stream, EOF_BLOCK);
}

/// Writing nothing must not emit a block, and must not upset the counters.
#[test]
fn writing_no_chunks_is_a_no_op() {
    let mut w = writer(4);
    w.write_batch(&device(Vec::new()), &[0]).unwrap();
    assert_eq!(w.blocks_written(), 0);
    assert_eq!(w.compressed_bytes(), 0);
    assert_eq!(w.uncompressed_bytes(), 0);
    assert_eq!(w.finish().unwrap(), EOF_BLOCK);
}

/// The counters are what a BAM writer builds an index from, so they have to
/// mean what they say across a batch split rather than only at the end.
#[test]
fn the_counters_track_across_batches() {
    let data = payload(9000, 11);
    let bounds: Vec<usize> = (0..=9).map(|i| i * 1000).collect();

    let mut w = writer(2);
    w.write_batch(&device(data.clone()), &bounds).unwrap();

    assert_eq!(w.blocks_written(), 9);
    assert_eq!(w.uncompressed_bytes(), 9000);

    let compressed = w.compressed_bytes();
    let stream = w.finish().unwrap();
    assert_eq!(
        compressed as usize,
        stream.len() - EOF_BLOCK.len(),
        "compressed_bytes must exclude the EOF block and match what was written"
    );
}

/// A virtual position is what a BAI records, and it must name a real block.
#[test]
fn the_virtual_position_names_the_next_block() {
    let data = payload(4000, 5);
    let bounds: Vec<usize> = (0..=4).map(|i| i * 1000).collect();

    let mut w = writer(1);
    let mut starts = Vec::new();
    for i in 0..4 {
        starts.push(w.virtual_position().compressed());
        w.write_batch(&device(data[..].to_vec()), &bounds[i..=i + 1])
            .unwrap();
    }
    let stream = w.finish().unwrap();

    // Every recorded position must be the start of an actual block.
    let spans = discover_blocks(&stream, 0).unwrap();
    let real: Vec<u64> = spans.iter().map(|s| s.compressed_offset).collect();
    for start in &starts {
        assert!(
            real.contains(start),
            "virtual position {start} is not a block start; a BAI built from it \
             would seek into the middle of a block"
        );
    }
    assert_eq!(starts[0], 0);
}

#[test]
fn write_all_device_chunks_at_the_payload_size() {
    let data = payload(2500, 2);

    let mut w =
        DeviceBgzfWriter::new(Vec::new(), HostDeviceCompressor::new()).with_payload_size(1000);
    w.write_all_device(&device(data.clone())).unwrap();
    assert_eq!(w.blocks_written(), 3, "1000 + 1000 + 500");
    let stream = w.finish().unwrap();

    let back = read_back(&stream);
    assert_eq!(back.data(), data.as_slice());
    assert_eq!(back.offsets(), [0, 1000, 2000, 2500, 2500]);
}

#[test]
fn a_chunk_too_large_to_frame_is_refused() {
    let data = payload(MAX_COMPRESSIBLE_PAYLOAD + 1, 1);
    let mut w = writer(4);
    assert!(
        w.write_batch(&device(data.clone()), &[0, data.len()])
            .is_err(),
        "splitting it would move a block boundary the caller chose"
    );
}

#[test]
fn bounds_reaching_past_the_buffer_are_refused() {
    let mut w = writer(4);
    assert!(w.write_batch(&device(vec![0u8; 10]), &[0, 11]).is_err());
    assert!(
        w.write_batch(&device(vec![0u8; 10]), &[0, 10, 2, 10])
            .is_err()
    );
    assert!(w.write_batch(&device(vec![0u8; 10]), &[]).is_err());
}

/// A compressor on another device must refuse rather than read memory it cannot
/// address — the failure that on real hardware is a fault inside a kernel.
#[test]
fn a_buffer_from_another_device_is_refused() {
    let mut w = DeviceBgzfWriter::new(Vec::new(), HostDeviceCompressor::on_device(0));
    let elsewhere = DeviceBuffer::new(Box::new(HostAlloc::on_device(vec![1, 2, 3], 1)));
    assert!(w.write_batch(&elsewhere, &[0, 3]).is_err());
}

/// Several `write_batch` calls must concatenate, so a caller can stream a file
/// through in whatever pieces it produces them.
#[test]
fn successive_writes_concatenate() {
    let mut w = writer(2);
    for part in [&b"alpha"[..], b"beta", b"gamma"] {
        w.write_batch(&device(part.to_vec()), &[0, part.len()])
            .unwrap();
    }
    assert_eq!(w.blocks_written(), 3);
    let stream = w.finish().unwrap();

    let back = read_back(&stream);
    assert_eq!(back.data(), b"alphabetagamma");
    assert_eq!(back.offsets(), [0, 5, 9, 14, 14]);
}

/// The round trip that ties the two halves together: read a real file into
/// "device" memory, write it straight back out, and get the same records.
#[test]
fn a_real_fixture_survives_a_read_write_round_trip() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/pacbio_hifi.bam");
    let raw = std::fs::read(path).expect("fixture missing; see testdata/README.md");

    let inflated = read_back(&raw);

    // One chunk per batch, so the fixture's 13 blocks span 13 batches — the
    // hardest split available and the one a larger default would never reach.
    let mut w = writer(1);
    w.write_batch(&device(inflated.data().to_vec()), inflated.offsets())
        .unwrap();
    let stream = w.finish().unwrap();

    let back = read_back(&stream);
    assert_eq!(back.data(), inflated.data());
    assert_eq!(
        &back.offsets()[..inflated.offsets().len()],
        inflated.offsets(),
        "block boundaries moved, which costs the GPU record scan its \
         per-block parallelism"
    );

    // And htslib agrees, which is the acceptance bar.
    if let Ok(out) = std::process::Command::new("samtools")
        .args(["view", "-c"])
        .arg({
            let dir = std::env::temp_dir().join("fritillaria-device-write");
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("out.bam");
            std::fs::write(&p, &stream).unwrap();
            p
        })
        .output()
    {
        assert!(out.status.success(), "samtools rejected our output");
        let count: usize = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();
        assert_eq!(count, 20, "the fixture holds 20 HiFi reads");
    }
}
