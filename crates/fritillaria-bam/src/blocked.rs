//! Record boundary discovery, decomposed so it can run in parallel.
//!
//! # The problem
//!
//! [`scan_records`](crate::scan_records) is inherently sequential: record *n+1*
//! starts where record *n* ends, and the only way to learn that is to read
//! record *n*'s length prefix. On the device that is fatal — a serial walk of
//! dependent loads over 55,000 records costs more than inflating the batch did.
//!
//! # The way out, and why it is sound
//!
//! htslib calls `bgzf_flush_try` before writing each record: rather than split a
//! record across two BGZF blocks, it ends the block early and starts the next
//! one on a record boundary. So **in an htslib-written BAM, the start of every
//! block is also the start of a record** — unless a record is larger than a
//! block (65280 bytes), which only ultra-long reads reach.
//!
//! That makes each block start a *candidate* boundary, giving one independent
//! chain per block instead of one chain for the whole batch. The property is a
//! habit of one writer, not a rule of the format, so it is used as a hint and
//! never trusted:
//!
//! 1. **Speculate** (parallel): walk from every block start, bounded by the next
//!    one. Cheap, and wrong for any block a long record runs through.
//! 2. **Reconcile** (serial, but over *blocks* — hundreds, not millions):
//!    follow the true chain from the known first record. At each block, if the
//!    true chain arrives exactly at the block start, adopt that block's
//!    speculative result wholesale; otherwise walk the block directly.
//! 3. **Emit** (parallel): with each block's entry point and record count known,
//!    write out the offsets.
//!
//! **Correctness does not depend on the speculation being right.** A block's
//! precomputed walk is used only when the true chain is proven to arrive at that
//! block's start, and a walk from a true boundary *is* the true walk. A wrong
//! guess costs work, never accuracy. Step 2 falls back to walking, which is what
//! makes ultra-long ONT reads — where a record covers whole blocks and most
//! guesses are wrong — correct rather than merely slow.
//!
//! This module is the CPU reference for `kernels/bam_decode.cu`. The kernel is a
//! direct translation of the three phases below, so this is the oracle the
//! device output is diffed against, and the only version testable without a GPU.

use fritillaria_core::{Error, Result};

use crate::record::RECORD_CORE_SIZE;

/// Bytes before a record's read name: the `block_size` prefix plus the core.
const NAME_START: usize = 4 + RECORD_CORE_SIZE;

/// What a walk over one segment found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// Records that start within the segment.
    pub count: usize,
    /// First offset at or past the segment end, i.e. where the next segment's
    /// first record begins.
    pub land: usize,
    /// Whether the walk ended on a record boundary rather than part-way through
    /// a record that the buffer cuts short.
    pub complete: bool,
}

/// A blocked scan: the record offsets, plus the per-block decomposition the
/// device kernels need to rebuild them in parallel.
#[derive(Clone, Debug, Default)]
pub struct BlockedScan {
    /// Start offset of every complete record, in file order.
    pub offsets: Vec<usize>,
    /// Offset of the first incompletely-buffered record, or `buf.len()`.
    pub tail: usize,
    /// Per block: where the first record starting in that block begins, or
    /// `None` when a record from an earlier block covers all of it.
    pub entries: Vec<Option<usize>>,
    /// Per block: how many records start in it.
    pub counts: Vec<usize>,
    /// Per block: index into [`offsets`](BlockedScan::offsets) of its first
    /// record — the exclusive prefix sum of `counts`, which is what lets the
    /// emit phase write every block's offsets concurrently.
    pub first_index: Vec<usize>,
    /// Blocks whose speculative walk could not be used, so the reconcile phase
    /// walked them itself.
    ///
    /// Exposed because it is the only direct measure of how well the
    /// speculation is working, and because a test that means to exercise the
    /// fallback needs to be able to prove it did. Expect 1 for an ordinary
    /// htslib file — the block holding the header, where the first record
    /// starts part-way in — and more only where records span blocks.
    pub fallbacks: usize,
}

fn read_u32(buf: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?))
}

fn read_u16(buf: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?))
}

/// Whether the variable-length fields declared by a record fit inside it.
///
/// This is the check that makes a speculative walk safe. Starting mid-record
/// yields an arbitrary `block_size`, and without this the walk would stride off
/// into nonsense; with it, a bad guess is caught within a record or two because
/// `l_read_name`, `n_cigar_op` and `l_seq` have to add up to exactly what the
/// length prefix claims.
fn fields_fit(buf: &[u8], start: usize, end: usize) -> bool {
    let Some(name_len) = buf.get(start + 12).copied() else {
        return false;
    };
    if name_len == 0 {
        return false; // the NUL terminator is counted, so this is never 0
    }
    let (Some(n_cigar), Some(l_seq)) = (read_u16(buf, start + 16), read_u32(buf, start + 20))
    else {
        return false;
    };

    let mut need = NAME_START;
    for part in [
        usize::from(name_len),
        4 * usize::from(n_cigar),
        (l_seq as usize).div_ceil(2),
        l_seq as usize,
    ] {
        let Some(next) = need.checked_add(part) else {
            return false;
        };
        need = next;
    }
    start
        .checked_add(need)
        .is_some_and(|fields_end| fields_end <= end)
}

/// Walks records from `from` until reaching `until` or running out of buffer.
///
/// Returns `None` if a record is malformed — which, for a speculative walk from
/// a block start that is *not* a record boundary, is the expected outcome and
/// not an error. The caller decides which it is.
#[must_use]
pub fn walk_segment(buf: &[u8], from: usize, until: usize) -> Option<Segment> {
    let mut cur = from;
    let mut count = 0usize;

    loop {
        if cur >= until {
            return Some(Segment {
                count,
                land: cur,
                complete: true,
            });
        }
        let Some(block_size) = read_u32(buf, cur) else {
            // Fewer than four bytes left: a partial length prefix, not an error.
            return Some(Segment {
                count,
                land: cur,
                complete: false,
            });
        };
        let block_size = block_size as usize;
        if block_size < RECORD_CORE_SIZE {
            return None;
        }
        let end = cur.checked_add(4 + block_size)?;
        if end > buf.len() {
            // The record is real but the buffer stops inside it: this is the
            // tail the caller carries forward, not corruption.
            return Some(Segment {
                count,
                land: cur,
                complete: false,
            });
        }
        if !fields_fit(buf, cur, end) {
            return None;
        }
        count += 1;
        cur = end;
    }
}

/// Discovers record boundaries using block starts as candidate boundaries.
///
/// `block_starts` is the block layout of `buf`, length `n + 1`, starting at 0
/// and ending at `buf.len()` — exactly
/// [`DeviceInflateBatch::offsets`](fritillaria_core::DeviceInflateBatch::offsets).
/// `start` is where the first record begins, after the header.
///
/// Equivalent to [`scan_records`](crate::scan_records) in output; see the module
/// docs for why it is decomposed this way.
pub fn scan_records_blocked(
    buf: &[u8],
    block_starts: &[usize],
    start: usize,
) -> Result<BlockedScan> {
    let malformed = |position: usize, reason: &str| Error::Malformed {
        format: "bam",
        position: position as u64,
        reason: reason.to_string(),
    };

    if block_starts.first() != Some(&0) || block_starts.last() != Some(&buf.len()) {
        return Err(malformed(
            0,
            "block starts must span the buffer from 0 to its length",
        ));
    }
    let n = block_starts.len() - 1;

    // Phase 1: speculate. Independent per block, so this is the parallel one.
    let speculative: Vec<Option<Segment>> = (0..n)
        .map(|b| walk_segment(buf, block_starts[b], block_starts[b + 1]))
        .collect();

    // Phase 2: reconcile. Serial, but over blocks rather than records.
    let mut entries = vec![None; n];
    let mut counts = vec![0usize; n];
    let mut first_index = vec![0usize; n];
    let mut cur = start;
    let mut emitted = 0usize;
    let mut tail = buf.len();
    let mut truncated = false;
    let mut fallbacks = 0usize;

    for b in 0..n {
        first_index[b] = emitted;
        if truncated || cur >= block_starts[b + 1] {
            // Either the buffer already ran out, or a record that started
            // earlier covers this whole block, so nothing starts in it.
            continue;
        }

        // The fast path is only taken when the true chain lands exactly on the
        // block start, which is the proof that the speculative walk from there
        // was the true walk.
        let segment = if cur == block_starts[b] {
            speculative[b]
        } else {
            fallbacks += 1;
            walk_segment(buf, cur, block_starts[b + 1])
        };
        let Some(segment) = segment else {
            return Err(malformed(cur, "malformed record during boundary scan"));
        };

        entries[b] = Some(cur);
        counts[b] = segment.count;
        emitted += segment.count;
        cur = segment.land;
        if !segment.complete {
            tail = segment.land;
            truncated = true;
        }
    }

    if !truncated {
        tail = cur;
    }

    // Phase 3: emit. Each block writes into its own slice of `offsets`, so this
    // is the second parallel phase.
    let mut offsets = vec![0usize; emitted];
    for b in 0..n {
        let Some(entry) = entries[b] else { continue };
        let mut cur = entry;
        for i in 0..counts[b] {
            offsets[first_index[b] + i] = cur;
            // Every offset here was validated in phase 1 or 2.
            cur += 4 + read_u32(buf, cur).unwrap_or(0) as usize;
        }
    }

    Ok(BlockedScan {
        offsets,
        tail,
        entries,
        counts,
        first_index,
        fallbacks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::scan_records;
    use crate::record::tests::build_record;

    /// Splits a buffer into `n` equal blocks, the way BGZF would.
    fn even_blocks(len: usize, block: usize) -> Vec<usize> {
        let mut starts: Vec<usize> = (0..len).step_by(block.max(1)).collect();
        starts.push(len);
        starts.dedup();
        starts
    }

    fn records(names: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        for name in names {
            buf.extend_from_slice(&build_record(name, b"ACGT", None));
        }
        buf
    }

    #[test]
    fn agrees_with_the_sequential_scan() {
        let buf = records(&[b"a", b"bb", b"ccc", b"dddd", b"e"]);
        let (expected, expected_tail) = scan_records(&buf, 0).unwrap();

        // Every blocking, including ones that cut records in half — which is
        // what makes the speculative walks wrong and the fallback necessary.
        for block in 1..=buf.len() {
            let starts = even_blocks(buf.len(), block);
            let scan = scan_records_blocked(&buf, &starts, 0).unwrap();
            assert_eq!(scan.offsets, expected, "block size {block}");
            assert_eq!(scan.tail, expected_tail, "block size {block}");
        }
    }

    #[test]
    fn block_starts_that_are_record_starts_take_the_fast_path() {
        // The htslib layout: each block ends exactly on a record boundary.
        let one = build_record(b"r", b"ACGT", None);
        let buf = records(&[b"r", b"r", b"r", b"r"]);
        let starts = vec![0, one.len(), 2 * one.len(), 3 * one.len(), buf.len()];

        let scan = scan_records_blocked(&buf, &starts, 0).unwrap();
        assert_eq!(scan.counts, vec![1, 1, 1, 1]);
        assert_eq!(scan.first_index, vec![0, 1, 2, 3]);
        assert_eq!(
            scan.entries,
            vec![
                Some(0),
                Some(one.len()),
                Some(2 * one.len()),
                Some(3 * one.len())
            ]
        );
    }

    #[test]
    fn a_record_covering_whole_blocks_leaves_them_empty() {
        // The ONT case: one record spans several blocks, so those blocks have no
        // record starting in them at all and their speculative walks are junk.
        let big = build_record(b"long", &vec![b'A'; 4000], None);
        let mut buf = big.clone();
        buf.extend_from_slice(&build_record(b"after", b"ACGT", None));

        let starts = even_blocks(buf.len(), 512);
        let scan = scan_records_blocked(&buf, &starts, 0).unwrap();

        let (expected, _) = scan_records(&buf, 0).unwrap();
        assert_eq!(scan.offsets, expected);
        assert_eq!(scan.counts[0], 1, "the long record starts in block 0");
        assert!(
            scan.counts[1..scan.counts.len() - 1]
                .iter()
                .all(|&c| c == 0),
            "blocks the long record runs through must hold no record starts: {:?}",
            scan.counts
        );
        assert!(
            scan.entries[1].is_none(),
            "a block fully covered by an earlier record has no entry point"
        );
    }

    #[test]
    fn a_partial_trailing_record_reports_the_tail() {
        let mut buf = records(&[b"first", b"second"]);
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"third", b"ACGT", None));
        buf.truncate(buf.len() - 5);

        for block in [1usize, 7, 32, 100] {
            let starts = even_blocks(buf.len(), block);
            let scan = scan_records_blocked(&buf, &starts, 0).unwrap();
            assert_eq!(scan.offsets.len(), 2, "block size {block}");
            assert_eq!(scan.tail, boundary, "block size {block}");
        }
    }

    #[test]
    fn honours_a_start_offset_past_a_header() {
        let header = vec![0xAAu8; 37];
        let mut buf = header.clone();
        buf.extend_from_slice(&records(&[b"a", b"b"]));

        let starts = even_blocks(buf.len(), 16);
        let scan = scan_records_blocked(&buf, &starts, header.len()).unwrap();

        let (expected, _) = scan_records(&buf, header.len()).unwrap();
        assert_eq!(scan.offsets, expected);
        assert_eq!(
            scan.counts[0], 0,
            "block 0 is header bytes; no record starts in it"
        );
    }

    #[test]
    fn first_index_is_the_prefix_sum_of_counts() {
        let buf = records(&[b"a", b"b", b"c", b"d", b"e", b"f"]);
        let starts = even_blocks(buf.len(), 40);
        let scan = scan_records_blocked(&buf, &starts, 0).unwrap();

        let mut acc = 0;
        for (b, &count) in scan.counts.iter().enumerate() {
            assert_eq!(scan.first_index[b], acc, "block {b}");
            acc += count;
        }
        assert_eq!(acc, scan.offsets.len());
    }

    #[test]
    fn a_malformed_record_on_the_true_chain_is_an_error() {
        // Distinct from a speculative walk failing, which is routine: this
        // record is reached from a proven boundary, so it is real corruption.
        let mut buf = records(&[b"a", b"b"]);
        buf[0] = 0; // block_size below the 32-byte core
        let starts = even_blocks(buf.len(), 16);
        assert!(matches!(
            scan_records_blocked(&buf, &starts, 0),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn rejects_block_starts_that_do_not_span_the_buffer() {
        let buf = records(&[b"a"]);
        for starts in [vec![0, buf.len() - 1], vec![1, buf.len()], vec![]] {
            assert!(
                scan_records_blocked(&buf, &starts, 0).is_err(),
                "block starts {starts:?} do not describe the buffer"
            );
        }
    }

    #[test]
    fn an_empty_buffer_scans_to_nothing() {
        let scan = scan_records_blocked(&[], &[0], 0).unwrap();
        assert!(scan.offsets.is_empty());
        assert_eq!(scan.tail, 0);
    }

    #[test]
    fn walk_rejects_a_record_whose_fields_overflow_its_length() {
        // The validation that keeps a speculative walk from striding into
        // nonsense: block_size says one thing, the field lengths another.
        let mut buf = build_record(b"r", b"ACGT", None);
        // Claim a 400-base sequence in a record sized for four. l_seq is at
        // offset 20; offset 24 is next_refID and patching it proves nothing.
        buf[20..24].copy_from_slice(&400u32.to_le_bytes());
        assert!(walk_segment(&buf, 0, buf.len()).is_none());
    }
}
