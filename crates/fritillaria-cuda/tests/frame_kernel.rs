//! The BGZF framing kernel, run on the CPU and diffed against `frame_block`.
//!
//! # Why this is not a device test
//!
//! `kernels/bgzf_frame.cu` is straight-line byte writes — no shared memory, no
//! warp cooperation, no atomics — so a shim that defines `__global__` away and
//! loops over the thread indices executes the real source faithfully. That makes
//! the *bytes* checkable on a machine with no GPU, which is where all host-side
//! development on this project happens.
//!
//! The rule this serves is one CLAUDE.md records having learned the hard way:
//! **before renting a VM, run every part of the test that does not need the
//! device.** A remote run should be finding out whether the GPU agrees, not
//! whether the header layout is right.
//!
//! What this deliberately does not cover: NVRTC accepting the source, the launch
//! configuration, and anything at all about nvCOMP. Those need hardware, and
//! `tests/nvcomp_compress.rs` is where they are checked.
//!
//! The oracle is `fritillaria_bgzf::frame_block` — the same function the CPU
//! compressor uses — so agreement here means the two paths cannot drift on the
//! `BC` off-by-one, which is the classic BGZF bug and the one most likely to
//! produce a file that looks fine until something else reads it.

use std::path::PathBuf;
use std::process::Command;

use fritillaria_bgzf::{STORED_BLOCK_HEADER, frame_block, store_block};
use fritillaria_core::compress::{BGZF_HEADER_SIZE, BGZF_TRAILER_SIZE, Framing, choose_framing};

/// One chunk's worth of what the kernel is handed.
struct Chunk {
    payload: Vec<u8>,
    /// What a compressor claims it produced. Not required to be real deflate —
    /// the kernel copies it verbatim and never inspects it, which is what lets
    /// the over-cap case be constructed without a 148 KB compressor run.
    deflate: Vec<u8>,
}

/// A compiled shim with a scratch directory of its own.
///
/// Per-test rather than shared: cargo runs tests concurrently, and a shared
/// executable is one `Text file busy` away from a failure that looks like a
/// kernel bug and is not.
struct Harness {
    exe: PathBuf,
    dir: PathBuf,
}

/// Compiles the shim, or `None` if there is no C++ compiler here.
///
/// A missing compiler is reported loudly by every caller, not swallowed: a quiet
/// skip is how this project has twice shipped a green run over work that never
/// happened.
fn build_harness(name: &str) -> Option<Harness> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("frame-kernel")
        .join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("frame_host");

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new("c++")
        .arg("-O1")
        .arg("-std=c++17")
        .arg("-I")
        .arg(manifest.join("kernels"))
        .arg(manifest.join("tests/harness/frame_host.cpp"))
        .arg("-o")
        .arg(&exe)
        .status()
        .ok()?;

    assert!(
        status.success(),
        "the framing kernel did not compile as C++; that is a real defect in \
         kernels/bgzf_frame.cu, not a harness problem"
    );
    Some(Harness { exe, dir })
}

fn push_u64s(buf: &mut Vec<u8>, values: impl IntoIterator<Item = u64>) {
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Runs the kernel over `chunks` at `block_dim` threads and returns the stream.
fn run_kernel(h: &Harness, chunks: &[Chunk], block_dim: u64) -> Vec<u8> {
    let count = chunks.len();

    // The same layout `FramePlan` computes, recomputed here from the oracle's
    // own rule rather than by calling the planner: a test that fed the planner's
    // output back to a kernel the planner also sized would be checking that one
    // module agrees with itself.
    let mut slot_offsets = Vec::new();
    let mut deflate_blob = Vec::new();
    for chunk in chunks {
        slot_offsets.push(deflate_blob.len() as u64);
        deflate_blob.extend_from_slice(&chunk.deflate);
    }

    let mut raw_offsets = Vec::new();
    let mut raw_blob = Vec::new();
    for chunk in chunks {
        raw_offsets.push(raw_blob.len() as u64);
        raw_blob.extend_from_slice(&chunk.payload);
    }

    let mut out_offsets = Vec::new();
    let mut stored = Vec::new();
    let mut cursor = 0u64;
    for chunk in chunks {
        out_offsets.push(cursor);
        let framing = choose_framing(chunk.payload.len(), chunk.deflate.len());
        stored.push(u64::from(framing == Framing::Stored));
        let body = match framing {
            Framing::Deflated => chunk.deflate.len(),
            Framing::Stored => chunk.payload.len() + STORED_BLOCK_HEADER,
        };
        cursor += (BGZF_HEADER_SIZE + body + BGZF_TRAILER_SIZE) as u64;
    }
    let total = cursor;

    let mut input = Vec::new();
    push_u64s(&mut input, [count as u64, total, block_dim]);
    push_u64s(&mut input, slot_offsets);
    push_u64s(&mut input, chunks.iter().map(|c| c.deflate.len() as u64));
    push_u64s(&mut input, raw_offsets);
    push_u64s(&mut input, out_offsets);
    push_u64s(&mut input, chunks.iter().map(|c| c.payload.len() as u64));
    push_u64s(
        &mut input,
        chunks
            .iter()
            .map(|c| u64::from(crc32fast::hash(&c.payload))),
    );
    push_u64s(&mut input, stored);
    push_u64s(&mut input, [deflate_blob.len() as u64]);
    input.extend_from_slice(&deflate_blob);
    push_u64s(&mut input, [raw_blob.len() as u64]);
    input.extend_from_slice(&raw_blob);

    let in_path = h.dir.join("input.bin");
    let out_path = h.dir.join("output.bin");
    std::fs::write(&in_path, &input).unwrap();

    let status = Command::new(&h.exe)
        .arg(&in_path)
        .arg(&out_path)
        .status()
        .unwrap();
    assert!(status.success(), "the framing shim exited {status}");

    std::fs::read(&out_path).unwrap()
}

/// What `frame_block` produces for the same chunks — the oracle.
fn reference(chunks: &[Chunk]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in chunks {
        let crc = crc32fast::hash(&chunk.payload);
        let isize = chunk.payload.len() as u32;
        match choose_framing(chunk.payload.len(), chunk.deflate.len()) {
            Framing::Deflated => frame_block(&mut out, &chunk.deflate, crc, isize).unwrap(),
            Framing::Stored => {
                let mut stored = Vec::new();
                store_block(&chunk.payload, &mut stored);
                frame_block(&mut out, &stored, crc, isize).unwrap();
            }
        }
    }
    out
}

fn chunk(payload: &[u8], level: u8) -> Chunk {
    Chunk {
        payload: payload.to_vec(),
        deflate: miniz_oxide::deflate::compress_to_vec(payload, level),
    }
}

fn incompressible(len: usize) -> Vec<u8> {
    let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
    (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u8
        })
        .collect()
}

#[test]
fn the_kernel_frames_exactly_what_frame_block_frames() {
    let Some(h) = build_harness("oracle") else {
        panic!(
            "no C++ compiler found, so the framing kernel went unchecked. This \
             must fail rather than skip: the whole point is to catch here what \
             would otherwise be found on a rented VM."
        );
    };

    let chunks = vec![
        // An empty block: the BGZF EOF marker is exactly this shape.
        chunk(b"", 6),
        chunk(b"the quick brown fox jumps over the lazy dog", 6),
        // Highly compressible, so the deflate body is much shorter than the
        // payload and the trailer lands somewhere different from the input.
        chunk(&vec![b'A'; 40_000], 6),
        // Incompressible at a full default payload, which is where the stored
        // fallback and the BC field are both closest to their limits.
        chunk(&incompressible(65_280), 6),
        chunk(&incompressible(1024), 6),
    ];

    let ours = run_kernel(&h, &chunks, 256);
    assert_eq!(
        ours,
        reference(&chunks),
        "the kernel and frame_block disagree on the framed bytes"
    );
}

/// The case the `stored` flag exists for, and the one no real compressor on this
/// machine will produce: nvCOMP's worst case is 2.26x a full chunk.
#[test]
fn an_over_cap_deflate_stream_is_stored_instead() {
    let h = build_harness("over-cap").expect("no C++ compiler");

    let payload = incompressible(65_280);
    let chunks = vec![Chunk {
        payload: payload.clone(),
        // 148,256 is nvCOMP's reported worst case for a 64 KiB chunk. The bytes
        // are never read — the kernel stores the payload instead — so what
        // matters is only that the *length* triggers the fallback.
        deflate: vec![0u8; 148_256],
    }];

    let ours = run_kernel(&h, &chunks, 256);

    assert_eq!(ours, reference(&chunks));
    assert_eq!(
        ours.len(),
        BGZF_HEADER_SIZE + payload.len() + STORED_BLOCK_HEADER + BGZF_TRAILER_SIZE
    );
    assert!(
        ours.len() <= fritillaria_core::MAX_BLOCK_SIZE,
        "a stored fallback must still fit the block cap"
    );

    // And it is readable, which is the claim that actually matters.
    let mut stream = ours.clone();
    stream.extend_from_slice(&fritillaria_bgzf::EOF_BLOCK);
    let mut reader = fritillaria_bgzf::BgzfReader::new(std::io::Cursor::new(stream));
    let mut got = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
    assert_eq!(got, payload);
}

/// The kernel divides the body copy across `blockDim.x` threads, so a stride bug
/// would leave holes that a single width could not expose.
#[test]
fn the_output_does_not_depend_on_the_thread_count() {
    let h = build_harness("thread-count").expect("no C++ compiler");

    let chunks = vec![
        chunk(&incompressible(4096), 6),
        chunk(&vec![b'G'; 9000], 6),
        chunk(b"", 6),
    ];
    let expected = reference(&chunks);

    for block_dim in [1, 2, 31, 32, 33, 256, 1024] {
        assert_eq!(
            run_kernel(&h, &chunks, block_dim),
            expected,
            "framing changed at {block_dim} threads per block"
        );
    }
}

/// Reading a whole real file back out is the end-to-end shape of what the device
/// path does, minus nvCOMP.
#[test]
fn a_real_fixture_reframed_by_the_kernel_still_reads() {
    let h = build_harness("fixture").expect("no C++ compiler");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/pacbio_hifi.bam");
    let raw = std::fs::read(path).expect("fixture missing; see testdata/README.md");

    let spans = fritillaria_bgzf::discover_blocks(&raw, 0).unwrap();
    let mut inflated = fritillaria_core::InflateBatch::new();
    fritillaria_core::BlockCodec::inflate_batch(
        &fritillaria_bgzf::CpuCodec::new(),
        &raw,
        &spans,
        &mut inflated,
    )
    .unwrap();

    let chunks: Vec<Chunk> = (0..inflated.len())
        .map(|i| chunk(inflated.block(i).unwrap(), 6))
        .collect();

    let framed = run_kernel(&h, &chunks, 256);
    assert_eq!(framed, reference(&chunks));

    let mut reader = fritillaria_bgzf::BgzfReader::new(std::io::Cursor::new(framed));
    let mut got = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
    assert_eq!(got, inflated.data(), "the reframed file lost its payload");
}
