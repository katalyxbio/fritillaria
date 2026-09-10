//! Indexed access on the codec-driven reader.
//!
//! Until this existed, `BgzfReader` implemented `io::Read` and `io::BufRead`
//! but not `io::Seek`, so a region query could not use it at all and fell back
//! to the vendored CPU reader. That is the wrong way round: a query reads a
//! *small part* of a large file, which is exactly where paying for a GPU is
//! hardest to justify — but it is also where the fallback was silent.
//!
//! The oracle throughout is the vendored `io::Reader`, which is noodles' code
//! we neither wrote nor modified. Both readers seek to the same virtual
//! positions and must return the same bytes.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

use fritillaria_bgzf::io::Seek as BgzfSeek;
use fritillaria_bgzf::{BgzfReader, VirtualPosition, discover_blocks, io as bgzf_io};
use fritillaria_core::BlockCodec;

const FIXTURE: &str = "htslib_multiblock.bam";

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn raw() -> Vec<u8> {
    std::fs::read(path(FIXTURE)).expect("fixture missing; see testdata/README.md")
}

fn ours() -> BgzfReader<std::io::Cursor<Vec<u8>>> {
    BgzfReader::new(std::io::Cursor::new(raw()))
}

fn vendored() -> bgzf_io::Reader<std::io::Cursor<Vec<u8>>> {
    bgzf_io::Reader::new(std::io::Cursor::new(raw()))
}

/// Every block's start, paired with how many bytes it inflates to.
///
/// The length matters: an uncompressed offset is only valid *within its own
/// block*, and blocks differ. The header block of this fixture inflates to 223
/// bytes while the rest hold tens of thousands, so a test that reused one
/// block's offsets on another would be generating invalid positions and
/// comparing two readers' undefined behaviour.
fn blocks() -> Vec<(VirtualPosition, usize)> {
    let raw = raw();
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let mut inflated = fritillaria_core::InflateBatch::new();
    fritillaria_bgzf::CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut inflated)
        .expect("must inflate");
    let offsets = inflated.offsets();

    spans
        .iter()
        .enumerate()
        .map(|(i, span)| {
            let pos = VirtualPosition::try_from((span.compressed_offset, 0))
                .expect("valid virtual position");
            (pos, offsets[i + 1] - offsets[i])
        })
        .collect()
}

/// Just the starts, for tests that do not care about lengths.
fn block_starts() -> Vec<VirtualPosition> {
    blocks().into_iter().map(|(pos, _)| pos).collect()
}

#[test]
fn seeking_to_a_block_start_matches_the_vendored_reader() {
    let starts = block_starts();
    assert!(
        starts.len() > 3,
        "need several blocks; got {}",
        starts.len()
    );

    let (mut ours, mut theirs) = (ours(), vendored());
    for &pos in &starts {
        ours.seek_to_virtual_position(pos).expect("our seek");
        theirs.seek_to_virtual_position(pos).expect("vendored seek");

        let mut mine = vec![0u8; 4096];
        let mut theirs_buf = vec![0u8; 4096];
        let n = ours.read(&mut mine).expect("our read");
        let m = theirs.read(&mut theirs_buf).expect("vendored read");

        // The two readers batch differently, so a single `read` may return
        // different amounts. Compare the overlap, which is what a caller sees.
        // The EOF marker inflates to nothing, so a zero-length read there is
        // correct rather than a failure.
        let k = n.min(m);
        assert_eq!(
            &mine[..k],
            &theirs_buf[..k],
            "bytes after seeking to {pos:?}"
        );
    }
}

#[test]
fn seeking_inside_a_block_matches_the_vendored_reader() {
    // The half that a plain file seek cannot do: the lower 16 bits name a byte
    // *inside* a block, which is only meaningful once the block is inflated.
    let (mut ours, mut theirs) = (ours(), vendored());

    let mut checked = 0usize;
    for (start, len) in blocks() {
        // Only offsets that are genuinely inside *this* block.
        for offset in [1u16, 17, 250, 1000] {
            if usize::from(offset) >= len {
                continue;
            }
            let pos = VirtualPosition::try_from((start.compressed(), offset))
                .expect("valid virtual position");
            ours.seek_to_virtual_position(pos).expect("our seek");
            theirs.seek_to_virtual_position(pos).expect("vendored seek");
            checked += 1;

            let mut mine = vec![0u8; 512];
            let mut theirs_buf = vec![0u8; 512];
            let n = ours.read(&mut mine).expect("our read");
            let m = theirs.read(&mut theirs_buf).expect("vendored read");
            let k = n.min(m);
            assert_eq!(
                &mine[..k],
                &theirs_buf[..k],
                "bytes after seeking to {pos:?}"
            );
        }
    }
    assert!(
        checked > 8,
        "only {checked} in-block offsets were exercised"
    );
}

#[test]
fn seeking_backwards_rereads_the_same_bytes() {
    // The failure this guards is stale batching state: a carry or a cursor left
    // over from before the seek would prepend bytes from the old position,
    // which produces plausible data from the wrong part of the file.
    let starts = block_starts();
    let mut reader = ours();

    let first = starts[0];
    reader.seek_to_virtual_position(first).unwrap();
    let mut before = vec![0u8; 2048];
    let n = reader.read(&mut before).unwrap();

    // Wander far away, then come back.
    reader
        .seek_to_virtual_position(*starts.last().unwrap())
        .unwrap();
    let mut scratch = vec![0u8; 2048];
    let _ = reader.read(&mut scratch).unwrap();

    reader.seek_to_virtual_position(first).unwrap();
    let mut after = vec![0u8; 2048];
    let m = reader.read(&mut after).unwrap();

    assert_eq!(n, m, "same position must yield the same read length");
    assert_eq!(before[..n], after[..m], "seeking back must reread exactly");
}

#[test]
fn seeking_with_a_partial_block_buffered_discards_it() {
    // `carry` holds the bytes of a block cut short by the end of a read. They
    // describe the *old* position, so a seek must drop them -- keeping them
    // prepends bytes from elsewhere in the file to the new batch.
    //
    // Constructed rather than waited for: at the default 256 blocks per batch
    // this 15-block fixture loads in one go and never has a carry at all, so
    // every other test in this file would pass with the reset removed. Deleting
    // `self.carry.clear()` from `seek_to` turns this red and nothing else.
    let starts = block_starts();
    let mut reader = BgzfReader::new(std::io::Cursor::new(raw())).with_blocks_per_batch(1);

    // Read enough to leave a partial block buffered.
    let mut scratch = vec![0u8; 4096];
    let _primed = reader.read(&mut scratch).expect("prime the reader");

    let target = starts[1];
    reader.seek_to_virtual_position(target).expect("seek");
    let mut got = vec![0u8; 2048];
    let n = reader.read(&mut got).expect("read after seek");

    let mut theirs = vendored();
    theirs
        .seek_to_virtual_position(target)
        .expect("vendored seek");
    let mut want = vec![0u8; 2048];
    let m = theirs.read(&mut want).expect("vendored read");

    let k = n.min(m);
    assert!(k > 0, "expected bytes at {target:?}");
    assert_eq!(
        &got[..k],
        &want[..k],
        "stale carry leaked into the new batch"
    );
}

#[test]
fn the_reported_position_matches_the_one_sought() {
    let starts = block_starts();
    let mut reader = ours();

    for &pos in starts.iter().take(8) {
        reader.seek_to_virtual_position(pos).unwrap();
        assert_eq!(
            bgzf_io::Read::virtual_position(&reader),
            pos,
            "virtual_position must agree with the seek that produced it"
        );
    }
}

#[test]
fn a_compressed_offset_that_is_not_a_block_boundary_is_rejected() {
    // Silently starting mid-block would inflate garbage. Better to refuse.
    let mut reader = ours();
    let pos = VirtualPosition::try_from((3u64, 0)).expect("valid virtual position");
    let err = reader.seek_to_virtual_position(pos).unwrap_err();
    assert!(
        matches!(
            err.kind(),
            std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData
        ),
        "got {err:?}"
    );
}

#[test]
fn an_uncompressed_offset_past_the_block_is_rejected() {
    // The bug this pins: the bound must be the *block*, not the batch. A batch
    // holds up to 256 blocks, so checking against it lets an out-of-range
    // offset resolve into a later block and return plausible bytes from the
    // wrong record.
    let (first, len) = blocks()[0];
    assert_eq!(len, 223, "the header block of this fixture is 223 bytes");

    let mut reader = ours();
    // One past the block, and far short of the batch, which is the case a
    // batch-wide check would wave through.
    let pos = VirtualPosition::try_from((first.compressed(), len as u16 + 1))
        .expect("valid virtual position");
    let err = reader.seek_to_virtual_position(pos).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "got {err:?}");

    // The last byte of the block is still reachable.
    let ok = VirtualPosition::try_from((first.compressed(), len as u16 - 1)).unwrap();
    reader.seek_to_virtual_position(ok).expect("in-block seek");
}

#[test]
fn seek_with_index_only_accepts_seek_from_start() {
    // Matching the vendored reader. End and Current are not expressible from a
    // gzi index, which carries no uncompressed length.
    let mut reader = ours();
    let index = fritillaria_bgzf::gzi::Index::default();
    let err = reader
        .seek_with_index(&index, SeekFrom::End(0))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "got {err:?}");
}

#[test]
fn a_region_query_runs_on_our_reader() {
    // The whole point, end to end: `bam::io::Reader::query` is generic over
    // `bgzf::io::BufRead + bgzf::io::Seek`, so implementing Seek is what lets a
    // real indexed read use the codec-driven path. Compared against the same
    // query through the vendored reader.
    use fritillaria_bam as bam;

    let index = bam::bai::fs::read(path("htslib_multiblock.bam.bai")).expect("read the .bai");

    let file = File::open(path(FIXTURE)).unwrap();
    let mut mine = bam::io::Reader::from(BgzfReader::new(file));
    let header = mine.read_header().expect("our header");

    let file = File::open(path(FIXTURE)).unwrap();
    let mut theirs = bam::io::Reader::new(BufReader::new(file));
    let their_header = theirs.read_header().expect("vendored header");

    // Query the first reference sequence in full.
    let (name, reference) = header
        .reference_sequences()
        .first()
        .expect("at least one reference sequence");
    let region = format!("{}:1-{}", name, usize::from(reference.length()))
        .parse()
        .expect("valid region");

    let ours: Vec<_> = mine
        .query(&header, &index, &region)
        .expect("our query")
        .records()
        .map(|r| r.expect("our record"))
        .collect();
    let expected: Vec<_> = theirs
        .query(&their_header, &index, &region)
        .expect("vendored query")
        .records()
        .map(|r| r.expect("vendored record"))
        .collect();

    assert!(!expected.is_empty(), "the query must match some records");
    assert_eq!(ours.len(), expected.len(), "record count");
    for (i, (mine, want)) in ours.iter().zip(&expected).enumerate() {
        // Compare the decoded fields rather than the raw buffer: `Record`
        // does not expose its bytes, and these are what a caller sees.
        assert_eq!(mine.name(), want.name(), "record {i}: name");
        assert_eq!(mine.flags(), want.flags(), "record {i}: flags");
        assert_eq!(
            mine.alignment_start().map(Result::ok),
            want.alignment_start().map(Result::ok),
            "record {i}: position"
        );
    }
}

#[test]
fn reading_the_whole_file_still_works_after_the_seek_impl() {
    // Regression guard: `seek_to` resets every batching field, and getting that
    // wrong is invisible until a sequential read is also exercised.
    let mut ours = ours();
    let mut theirs = vendored();
    let (mut mine, mut want) = (Vec::new(), Vec::new());
    ours.read_to_end(&mut mine).unwrap();
    theirs.read_to_end(&mut want).unwrap();
    assert_eq!(mine, want, "sequential read must be unaffected");
}

/// Silences an unused-import warning on `Seek` for the inner cursor.
fn _assert_inner_is_seekable<T: Seek>(_: &T) {}
