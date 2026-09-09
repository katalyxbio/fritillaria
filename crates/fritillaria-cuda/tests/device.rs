//! Device tests.
//!
//! These need a real GPU, so they **skip** rather than fail when none is
//! present — that is what lets `cargo test --features cuda` be run anywhere.
//! A skipped test prints why; silence would be indistinguishable from passing.
//!
//! The assertions are differential: GPU output is compared against the CPU
//! implementation, which is the correctness oracle for every backend.

#![cfg(feature = "cuda")]

use fritillaria_cuda::CudaContext;

/// Opens a device, or returns `None` with an explanation.
fn device() -> Option<CudaContext> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            eprintln!("SKIP: no usable CUDA device ({err})");
            None
        }
    }
}

/// Splits `data` into `count` roughly equal spans.
fn spans(len: usize, count: usize) -> (Vec<u64>, Vec<u32>) {
    let chunk = len / count;
    let mut offsets = Vec::with_capacity(count);
    let mut lengths = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * chunk;
        let end = if i == count - 1 { len } else { start + chunk };
        offsets.push(start as u64);
        lengths.push((end - start) as u32);
    }
    (offsets, lengths)
}

#[test]
fn crc32_matches_the_cpu_reference() {
    let Some(ctx) = device() else { return };

    let data: Vec<u8> = (0..100_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let (offsets, lengths) = spans(data.len(), 64);

    let gpu = ctx.crc32_blocks(&data, &offsets, &lengths).unwrap();

    for (i, (&offset, &len)) in offsets.iter().zip(&lengths).enumerate() {
        let slice = &data[offset as usize..offset as usize + len as usize];
        let expected = crc32fast::hash(slice);
        assert_eq!(
            gpu[i], expected,
            "block {i} disagrees with the CPU reference"
        );
    }
}

#[test]
fn crc32_handles_empty_and_single_byte_blocks() {
    let Some(ctx) = device() else { return };

    // Empty blocks are legal — the BGZF EOF marker is exactly one.
    let data = b"x".to_vec();
    let offsets = vec![0u64, 0, 1];
    let lengths = vec![0u32, 1, 0];

    let gpu = ctx.crc32_blocks(&data, &offsets, &lengths).unwrap();

    assert_eq!(gpu[0], crc32fast::hash(b""));
    assert_eq!(gpu[1], crc32fast::hash(b"x"));
    assert_eq!(gpu[2], crc32fast::hash(b""));
}

#[test]
fn crc32_of_no_blocks_is_empty() {
    let Some(ctx) = device() else { return };
    assert!(ctx.crc32_blocks(&[], &[], &[]).unwrap().is_empty());
}

#[test]
fn crc32_rejects_out_of_range_spans() {
    let Some(ctx) = device() else { return };
    // Bounds are checked on the host: on the device this would be an
    // out-of-bounds read, which is far harder to diagnose.
    assert!(ctx.crc32_blocks(b"short", &[0], &[999]).is_err());
    assert!(ctx.crc32_blocks(b"short", &[0, 1], &[1]).is_err());
}
