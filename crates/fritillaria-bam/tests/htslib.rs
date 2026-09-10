//! Validation against files written by htslib.
//!
//! # Why this exists
//!
//! Every other test in this workspace builds its fixtures with this crate's own
//! [`BgzfWriter`], which makes them circular: a misunderstanding shared by the
//! writer and the reader passes cleanly and fails on the first real file.
//!
//! These tests read BAMs produced by `samtools` (htslib 1.16) and check the
//! results against what `samtools view` reports. That is the only thing here
//! that can catch a systematically wrong reading of the spec.
//!
//! Fixtures live in `testdata/` and are committed deliberately; regenerate them
//! with the commands recorded in `testdata/README.md`.

use std::path::PathBuf;

use fritillaria_bam::columnar::{RecordBatch, header::parse_header, seq};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// Decompresses a whole BAM through the CPU codec.
fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("htslib BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("htslib BGZF must inflate and verify");
    out
}

#[test]
fn reads_an_htslib_written_bgzf_container() {
    let raw = std::fs::read(fixture("htslib.bam")).unwrap();
    let spans = discover_blocks(&raw, 0).unwrap();

    // htslib terminates every file with the empty EOF block.
    assert!(spans.len() >= 2, "expected at least a data block and EOF");
    assert_eq!(
        spans.last().unwrap().isize,
        0,
        "last block must be the empty EOF marker"
    );

    // Spans must tile the file exactly; a gap means a misread block size.
    let total: u64 = raw.len() as u64;
    let last = spans.last().unwrap();
    let end = last.compressed_offset
        + (last.payload_start as u64 - last.compressed_offset)
        + last.payload_len as u64
        + 8;
    assert_eq!(end, total, "spans must cover the whole file");
}

#[test]
fn parses_an_htslib_written_header() {
    let batch = inflate("htslib.bam");
    let header = parse_header(batch.data()).expect("htslib header must parse");

    // Ground truth from `samtools view -H`.
    assert_eq!(header.references.len(), 2);
    assert_eq!(header.references[0].name, b"chr1");
    assert_eq!(header.references[0].length, 1000);
    assert_eq!(header.references[1].name, b"chr2");
    assert_eq!(header.references[1].length, 2000);

    let text = String::from_utf8_lossy(&header.text);
    assert!(
        text.starts_with("@HD\tVN:1.6\tSO:coordinate"),
        "got {text:?}"
    );
    assert!(text.contains("@RG\tID:rg1\tSM:sample1"));
}

#[test]
fn decodes_htslib_records_matching_samtools() {
    let inflated = inflate("htslib.bam");
    let buf = inflated.data();
    let header = parse_header(buf).unwrap();

    let mut batch = RecordBatch::new();
    let tail = batch.decode(buf, header.records_start).unwrap();

    assert_eq!(batch.len(), 8, "samtools view -c reports 8 records");
    assert_eq!(tail, buf.len(), "the last record must end exactly at EOF");

    // Ground truth from `samtools view`. BAM positions are 0-based, SAM's are
    // 1-based, so every expected position here is the SAM value minus one.
    let expected_pos = [6, 8, 8, 15, 28, 36, -1, 0];
    let expected_flags = [99u16, 0, 0, 0, 0, 83, 4, 0];
    let expected_mapq = [60u8, 30, 30, 30, 30, 60, 0, 40];
    let expected_seq_len = [17u32, 14, 6, 11, 6, 9, 9, 5];
    let expected_ref_id = [0i32, 0, 0, 0, 0, 0, -1, 1];

    assert_eq!(batch.position(), &expected_pos);
    assert_eq!(batch.flags(), &expected_flags);
    assert_eq!(batch.mapping_quality(), &expected_mapq);
    assert_eq!(batch.sequence_len(), &expected_seq_len);
    assert_eq!(batch.reference_sequence_id(), &expected_ref_id);
}

#[test]
fn decodes_htslib_variable_length_fields() {
    let inflated = inflate("htslib.bam");
    let buf = inflated.data();
    let header = parse_header(buf).unwrap();

    let mut batch = RecordBatch::new();
    batch.decode(buf, header.records_start).unwrap();
    let records: Vec<_> = batch.records(buf).collect();

    let names: Vec<_> = records.iter().map(|r| r.name().to_vec()).collect();
    assert_eq!(names[0], b"r001");
    assert_eq!(names[7], b"r008");

    assert_eq!(records[0].sequence(), b"TTAGATAAAGGATACTG");
    assert_eq!(
        records[7].sequence(),
        b"ACGTN",
        "N must decode from nibble 15"
    );

    // r003 is 6 bases — odd-length packing is r004 (11 bases).
    assert_eq!(records[3].sequence(), b"ATAGCTTCAGC");
    assert_eq!(
        records[3].sequence_packed().len(),
        6,
        "11 bases pack into 6 bytes with a padding nibble"
    );

    // r005 has `*` qualities in SAM: 0xFF repeated, not an empty field.
    assert!(
        seq::qualities_are_absent(records[4].qualities()),
        "absent qualities must be 0xFF * l_seq"
    );
    assert_eq!(records[4].qualities().len(), 6);

    // r007 is unmapped with `*` CIGAR.
    assert_eq!(
        records[6].cigar_op_count(),
        0,
        "unmapped record has no CIGAR"
    );
    assert_eq!(records[6].position(), -1);
}

#[test]
fn decodes_htslib_cigar_operations() {
    let inflated = inflate("htslib.bam");
    let buf = inflated.data();
    let header = parse_header(buf).unwrap();

    let mut batch = RecordBatch::new();
    batch.decode(buf, header.records_start).unwrap();
    let records: Vec<_> = batch.records(buf).collect();

    // r001 = 8M2I4M1D3M per `samtools view`.
    let raw = records[0].cigar_raw();
    let ops: Vec<_> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .map(|op| {
            (
                seq::cigar_op_kind(op).expect("valid op"),
                seq::cigar_op_len(op),
            )
        })
        .collect();

    assert_eq!(
        ops,
        vec![(b'M', 8), (b'I', 2), (b'M', 4), (b'D', 1), (b'M', 3)]
    );

    // r003 = 5H6M: hard clips are real ops and must not be skipped.
    let raw = records[2].cigar_raw();
    let first = u32::from_le_bytes(raw[..4].try_into().unwrap());
    assert_eq!(seq::cigar_op_kind(first), Some(b'H'));
    assert_eq!(seq::cigar_op_len(first), 5);
}

/// Runs a samtools subcommand, or `None` if samtools is unavailable.
///
/// Absent on the Colab image, so remote runs skip rather than fail — but the
/// skip is loud, because a silently skipped validation is worthless.
fn samtools(args: &[&str]) -> Option<std::process::Output> {
    match std::process::Command::new("samtools").args(args).output() {
        Ok(output) => Some(output),
        Err(err) => {
            eprintln!("SKIP: samtools unavailable ({err})");
            None
        }
    }
}

#[test]
fn samtools_accepts_what_our_writer_produces() {
    // The other direction of the round trip: decompress an htslib BAM,
    // re-compress it with our own BGZF writer, and hand it back to htslib.
    // Being able to read htslib's output proves nothing about writing.
    let inflated = inflate("htslib.bam");

    let mut writer = fritillaria_bgzf::BgzfWriter::new(Vec::new()).with_payload_size(4096);
    writer.write_data(inflated.data()).unwrap();
    let ours = writer.finish().unwrap();

    let dir = std::env::temp_dir().join(format!("fritillaria-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ours.bam");
    std::fs::write(&path, &ours).unwrap();

    let Some(check) = samtools(&["quickcheck", "-v", path.to_str().unwrap()]) else {
        return;
    };
    assert!(
        check.status.success(),
        "samtools quickcheck rejected our BGZF: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let count = samtools(&["view", "-c", path.to_str().unwrap()]).unwrap();
    assert!(
        count.status.success(),
        "samtools view failed on our BGZF: {}",
        String::from_utf8_lossy(&count.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&count.stdout).trim(),
        "8",
        "samtools must read back every record we wrote"
    );

    // And the decoded contents must survive, not merely the framing.
    let view = samtools(&["view", path.to_str().unwrap()]).unwrap();
    let text = String::from_utf8_lossy(&view.stdout);
    assert!(text.contains("TTAGATAAAGGATACTG"), "record payload altered");
    assert!(text.lines().next().unwrap().starts_with("r001"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn our_block_sizing_is_independent_of_htslibs() {
    // We deliberately re-block at a different payload size than htslib used.
    // If samtools still reads it, our BC/ISIZE/CRC framing is right rather
    // than accidentally matching htslib's layout.
    let inflated = inflate("htslib.bam");

    let mut writer = fritillaria_bgzf::BgzfWriter::new(Vec::new()).with_payload_size(200);
    writer.write_data(inflated.data()).unwrap();
    let ours = writer.finish().unwrap();

    let spans = discover_blocks(&ours, 0).unwrap();
    assert!(
        spans.len() > 3,
        "expected many small blocks, got {}",
        spans.len()
    );

    let dir = std::env::temp_dir().join(format!("fritillaria-rb-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("reblocked.bam");
    std::fs::write(&path, &ours).unwrap();

    if let Some(count) = samtools(&["view", "-c", path.to_str().unwrap()]) {
        assert!(
            count.status.success(),
            "samtools rejected our re-blocked file: {}",
            String::from_utf8_lossy(&count.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&count.stdout).trim(), "8");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reads_records_spanning_bgzf_block_boundaries() {
    // 4000 records over 15 BGZF blocks, so records certainly straddle block
    // boundaries. This is the seam where this class of library actually breaks.
    let inflated = inflate("htslib_multiblock.bam");
    let buf = inflated.data();
    assert!(inflated.len() > 10, "fixture must span many blocks");

    let header = parse_header(buf).unwrap();
    assert_eq!(header.references[0].name, b"chr1");

    let mut batch = RecordBatch::new();
    let tail = batch.decode(buf, header.records_start).unwrap();

    assert_eq!(batch.len(), 4000, "samtools view -c reports 4000 records");
    assert_eq!(tail, buf.len(), "no partial record at the end of the file");

    // Generated at 1, 101, 201, ... in SAM, so 0, 100, 200, ... in BAM.
    let positions = batch.position();
    assert_eq!(positions[0], 0);
    assert_eq!(positions[1], 100);
    assert_eq!(positions[3999], 399_900);

    // Every record is 100 bases with a 100M CIGAR.
    assert!(batch.sequence_len().iter().all(|&l| l == 100));
    assert!(batch.mapping_quality().iter().all(|&q| q == 60));

    // Spot-check a record deep in the file, where an offset drift would show.
    let last = batch.record(buf, 3999).unwrap();
    assert_eq!(last.name(), b"read003999");
    assert_eq!(last.sequence().len(), 100);
    assert_eq!(last.qualities().len(), 100);
}
