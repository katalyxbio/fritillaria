//! Record boundary discovery, decomposed so it can run in parallel.
//!
//! # The problem, and why BAM's answer does not port
//!
//! [`scan_records`](crate::columnar::scan_records) is inherently sequential: record *n+1*
//! begins where record *n* ends, and the only way to learn that is to read
//! record *n*'s length prefixes. On a device that is fatal — a serial walk of
//! dependent loads costs more than inflating the batch did.
//!
//! `fritillaria_bam::columnar::blocked` escapes it by leaning on htslib: `bgzf_flush_try`
//! ends a BGZF block early rather than splitting an alignment, so a block start
//! is *almost always* a record start, giving one independent chain per block.
//!
//! **`bcf_write` does not do that.** It calls `bgzf_write` directly, which packs
//! blocks to the 65280-byte cap, so a BCF record straddles essentially every
//! interior block boundary. Measured on the committed fixtures: **0 of 56**
//! interior block starts in `kg_phase3.bcf` are record starts, and 0 of 3 in
//! `giab_hg002.bcf`. Every speculation would be wrong and every block would
//! fall back to walking, collapsing to the one serial chain the design exists
//! to avoid.
//!
//! # What replaces it
//!
//! Speculate at **every byte offset**, not at block starts, and prune with a
//! validator. Four phases, three of them parallel:
//!
//! 1. **Sieve** (parallel, one thread per byte): [`looks_like_a_record`] at
//!    every offset. Fixed cost, no data-dependent loop.
//! 2. **Validate** (parallel, one thread per survivor): walk the survivor's
//!    typed-value chain and require its fields to end exactly where `l_shared`
//!    and `l_indiv` say. Cost is proportional to the record, so it runs only on
//!    what stage 1 let through.
//! 3. **Prove** (parallel, one thread per survivor): see below.
//! 4. **Fall back** (serial): if the proof fails, walk the chain.
//!
//! ## The proof, which is what makes this better than BAM's version
//!
//! BAM reconciles by following the true chain over *blocks* — serial, though
//! over hundreds rather than millions. BCF does not need to:
//!
//! > If the surviving offsets, in order, **tile** the buffer — the first is the
//! > known first record, each one's computed end is exactly the next, and the
//! > last ends exactly at the end — then they *are* the true record chain.
//!
//! By induction: the true chain starts at the same known offset, and each
//! record's length is determined by its own bytes, so the true successor of
//! `offsets[i]` is `offsets[i] + 8 + l_shared + l_indiv`, which the tiling
//! asserts equals `offsets[i+1]`. There is no room for a different answer.
//!
//! That check is O(1) per record and embarrassingly parallel, so **the serial
//! phase disappears entirely** when it succeeds. Correctness does not depend on
//! it succeeding: a failure means an extra or missing survivor, and the walk
//! below is taken instead. A wrong guess costs work, never accuracy.
//!
//! ## Does it succeed on real files?
//!
//! Measured with `examples/scan_survivors.rs`, sweeping every byte offset:
//!
//! | file | records | candidate offsets | false positives |
//! |---|---|---|---|
//! | `kg_phase3.bcf` | 715 | 3,670,536 | **0** |
//! | `giab_hg002.bcf` | 275 | 172,632 | **0** |
//! | sites-only chr22, 46,576 records | 46,576 | 5,808,422 | **0** |
//!
//! Stage 1 alone was exact on all three: the survivor set equalled the record
//! set before stage 2 ran. The sites-only file is the adversarial case, since
//! `n_sample == 0` makes the strongest check vacuous, and it was still exact.
//!
//! **Do not read that as a guarantee.** Nothing in the format forbids a false
//! positive; a record's payload may contain bytes that satisfy the validator.
//! The design is built so that costs work rather than accuracy when it happens,
//! and `a_false_positive_forces_the_fallback_and_still_scans_correctly` below
//! constructs one deliberately.
//!
//! This module is the CPU reference for a kernel that does not exist yet. It is
//! written to be a direct translation — each phase maps to one launch — so that
//! when the kernel is written, this is the oracle its output is diffed against.

use fritillaria_core::Result;

use crate::columnar::record::{Record, looks_like_a_record, scan_records};

/// How a scan arrived at its answer.
///
/// Public because it is the thing worth asserting in a test and worth logging
/// in a driver: a run that silently fell back to the serial walk on every batch
/// is working correctly and performing like the design it replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proof {
    /// The survivors tiled the buffer, so no chain walk was needed.
    Tiled,
    /// They did not, so the boundaries came from the serial walk.
    ///
    /// Carries how many survivors there were against how many records exist —
    /// a large excess means the validator is admitting false positives on this
    /// data, which is the number to look at before optimising anything else.
    Walked { survivors: usize, records: usize },
}

/// A speculative scan: the boundaries, plus how they were established.
#[derive(Clone, Debug)]
pub struct SpeculativeScan {
    /// Start offset of every complete record, in file order. Identical to what
    /// [`scan_records`] returns, whichever path produced it.
    pub offsets: Vec<usize>,
    /// Offset of the first incompletely-buffered record, or `buf.len()`.
    pub tail: usize,
    /// Offsets that survived the fixed-cost sieve.
    pub sieved: usize,
    /// Offsets that also survived full validation.
    pub validated: usize,
    /// Which path produced [`offsets`](SpeculativeScan::offsets).
    pub proof: Proof,
}

/// Where the record beginning at `pos` ends, if its lengths are readable.
fn record_end(buf: &[u8], pos: usize) -> Option<usize> {
    let l_shared = u32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?) as usize;
    let l_indiv = u32::from_le_bytes(buf.get(pos + 4..pos + 8)?.try_into().ok()?) as usize;
    pos.checked_add(8)?
        .checked_add(l_shared)?
        .checked_add(l_indiv)
}

/// Phase 1: every byte offset in `[start, buf.len())` that passes the sieve.
///
/// One thread per offset on a device. Kept separate so its selectivity can be
/// measured on its own — see `examples/scan_survivors.rs`.
#[must_use]
pub fn sieve(buf: &[u8], start: usize, samples: u32, contigs: u32) -> Vec<usize> {
    (start..buf.len())
        .filter(|&pos| looks_like_a_record(buf, pos, samples, contigs))
        .collect()
}

/// Phase 2: survivors whose typed-value chain agrees with their declared lengths.
///
/// One thread per survivor on a device.
#[must_use]
pub fn validate(buf: &[u8], candidates: &[usize]) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&pos| {
            record_end(buf, pos).is_some_and(|end| {
                end <= buf.len()
                    && Record::new(&buf[pos..end]).is_ok_and(|record| record.validate().is_ok())
            })
        })
        .collect()
}

/// Phase 3: whether `candidates` tile `[start, tail)` exactly.
///
/// One thread per survivor on a device, plus a reduction. Returns the tail the
/// tiling ends at, or `None` if it does not tile.
///
/// A partial trailing record is expected rather than exceptional — for BCF it
/// is the common case at a batch edge — so the tiling is allowed to stop short
/// of `buf.len()` provided the remainder is a record that runs past the end.
#[must_use]
pub fn prove_tiling(buf: &[u8], start: usize, candidates: &[usize]) -> Option<usize> {
    let mut at = start;
    for (index, &candidate) in candidates.iter().enumerate() {
        if candidate != at {
            return None;
        }
        let end = record_end(buf, candidate)?;
        if end > buf.len() {
            // A validated candidate cannot extend past the buffer, so this is
            // unreachable in practice; refuse rather than trust that.
            return None;
        }
        at = end;
        debug_assert!(index < candidates.len());
    }
    // Everything left over must be a genuine partial record, not a gap the
    // sieve missed: too short to hold a prefix, or a record that runs past the
    // end of the buffer.
    match record_end(buf, at) {
        None => Some(at),                         // fewer than 8 bytes left
        Some(end) if end > buf.len() => Some(at), // record continues past the batch
        Some(_) => None,                          // a whole record no candidate claimed
    }
}

/// Scans record boundaries speculatively, falling back to the serial walk.
///
/// `samples` and `contigs` come from the header and are what give the sieve its
/// selectivity. The result is always identical to [`scan_records`]; only the
/// route differs, and [`SpeculativeScan::proof`] says which was taken.
pub fn scan_records_speculative(
    buf: &[u8],
    start: usize,
    samples: u32,
    contigs: u32,
) -> Result<SpeculativeScan> {
    let sieved = sieve(buf, start, samples, contigs);
    let validated = validate(buf, &sieved);

    if let Some(tail) = prove_tiling(buf, start, &validated) {
        return Ok(SpeculativeScan {
            offsets: validated.clone(),
            tail,
            sieved: sieved.len(),
            validated: validated.len(),
            proof: Proof::Tiled,
        });
    }

    let (offsets, tail) = scan_records(buf, start)?;
    Ok(SpeculativeScan {
        proof: Proof::Walked {
            survivors: validated.len(),
            records: offsets.len(),
        },
        offsets,
        tail,
        sieved: sieved.len(),
        validated: validated.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columnar::record::SITE_CORE_SIZE;

    /// A minimal record: `n_allele` alleles, no INFO, no FORMAT.
    fn record(chrom: i32, pos: i32, alleles: &[&[u8]], samples: u32) -> Vec<u8> {
        let mut shared = Vec::new();
        shared.extend_from_slice(&chrom.to_le_bytes());
        shared.extend_from_slice(&pos.to_le_bytes());
        shared.extend_from_slice(&1i32.to_le_bytes());
        shared.extend_from_slice(&0x7F80_0001u32.to_le_bytes());
        shared.extend_from_slice(&0u16.to_le_bytes()); // n_info
        shared.extend_from_slice(&u16::try_from(alleles.len()).unwrap().to_le_bytes());
        shared.extend_from_slice(&samples.to_le_bytes()[..3]);
        shared.push(0); // n_fmt
        shared.push(0x07); // ID missing
        for allele in alleles {
            shared.push(u8::try_from(allele.len()).unwrap() << 4 | 0x07);
            shared.extend_from_slice(allele);
        }
        shared.push(0x00); // FILTER missing

        let mut out = Vec::new();
        out.extend_from_slice(&u32::try_from(shared.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // l_indiv
        out.extend_from_slice(&shared);
        out
    }

    /// A buffer of `n` records preceded by a stand-in header.
    fn buffer(n: usize, samples: u32) -> (Vec<u8>, usize, Vec<usize>) {
        let start = 9;
        let mut buf = vec![0xAA; start];
        let mut starts = Vec::new();
        for i in 0..n {
            starts.push(buf.len());
            buf.extend(record(
                2,
                1000 + i32::try_from(i).unwrap(),
                &[b"ACGT", b"A"],
                samples,
            ));
        }
        (buf, start, starts)
    }

    #[test]
    fn the_tiling_proof_replaces_the_walk_on_clean_input() {
        let (buf, start, expected) = buffer(16, 1);
        let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
        assert_eq!(scan.proof, Proof::Tiled, "no serial walk should be needed");
        assert_eq!(scan.offsets, expected);
        assert_eq!(scan.tail, buf.len());
        assert_eq!(scan.sieved, 16);
        assert_eq!(scan.validated, 16);
    }

    #[test]
    fn it_agrees_with_the_serial_scan() {
        for n in [0, 1, 2, 17] {
            let (buf, start, _) = buffer(n, 1);
            let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
            let (offsets, tail) = scan_records(&buf, start).unwrap();
            assert_eq!(scan.offsets, offsets, "{n} records");
            assert_eq!(scan.tail, tail, "{n} records");
        }
    }

    #[test]
    fn a_partial_trailing_record_still_tiles() {
        // The common case at a batch edge for BCF, so it must not be treated as
        // a proof failure — falling back on every batch would defeat the point.
        let (mut buf, start, expected) = buffer(4, 1);
        let truncated = buf.len() - 5;
        buf.truncate(truncated);

        let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
        assert_eq!(scan.proof, Proof::Tiled);
        assert_eq!(scan.offsets, expected[..3]);
        assert_eq!(
            scan.tail, expected[3],
            "the tail points at the partial record"
        );
    }

    #[test]
    fn a_prefix_too_short_for_the_length_words_still_tiles() {
        let (mut buf, start, expected) = buffer(3, 1);
        buf.truncate(expected[2] + 3);
        let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
        assert_eq!(scan.proof, Proof::Tiled);
        assert_eq!(scan.offsets, expected[..2]);
        assert_eq!(scan.tail, expected[2]);
    }

    #[test]
    fn a_false_positive_forces_the_fallback_and_still_scans_correctly() {
        // Measured false positives on real files: zero, across 9.6M candidate
        // offsets. Nothing in the format forbids one, so the property that
        // matters is not that they never happen but that one costs work rather
        // than accuracy. This constructs the case the measurement never found.
        //
        // The decoy is a whole valid record buried inside another record's
        // allele payload, so it passes the sieve *and* full validation and is
        // rejected only by the tiling.
        let decoy = record(1, 7, &[b"A", b"C"], 1);
        let mut allele = decoy.clone();
        allele.resize(decoy.len(), 0);

        let mut shared = Vec::new();
        shared.extend_from_slice(&2i32.to_le_bytes());
        shared.extend_from_slice(&500i32.to_le_bytes());
        shared.extend_from_slice(&1i32.to_le_bytes());
        shared.extend_from_slice(&0x7F80_0001u32.to_le_bytes());
        shared.extend_from_slice(&0u16.to_le_bytes());
        shared.extend_from_slice(&1u16.to_le_bytes()); // one allele
        shared.extend_from_slice(&1u32.to_le_bytes()[..3]);
        shared.push(0);
        shared.push(0x07);
        // One long allele whose bytes happen to be a complete record.
        shared.push(0xF7);
        shared.push(0x13);
        shared.extend_from_slice(&u32::try_from(allele.len()).unwrap().to_le_bytes());
        shared.extend_from_slice(&allele);
        shared.push(0x00);

        let mut host = Vec::new();
        host.extend_from_slice(&u32::try_from(shared.len()).unwrap().to_le_bytes());
        host.extend_from_slice(&0u32.to_le_bytes());
        host.extend_from_slice(&shared);

        let start = 9;
        let mut buf = vec![0xAA; start];
        let expected = vec![buf.len()];
        buf.extend(&host);

        let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
        assert!(
            scan.validated > 1,
            "the decoy must actually survive both stages, or this tests nothing \
             (sieved {}, validated {})",
            scan.sieved,
            scan.validated
        );
        assert!(
            matches!(scan.proof, Proof::Walked { .. }),
            "an extra survivor must break the tiling"
        );
        assert_eq!(
            scan.offsets, expected,
            "the fallback must still produce the true boundaries"
        );
        assert_eq!(scan.tail, buf.len());
    }

    /// A buffer whose records use the given contig indices.
    ///
    /// Passing a `contigs` bound that excludes some of them is how these tests
    /// manufacture a **sieve false negative** — a real record the sieve
    /// rejects. Real data should never produce one, which is exactly why the
    /// invariants that guard against it need building deliberately: the first
    /// mutation run showed all three were untested.
    fn mixed(chroms: &[i32]) -> (Vec<u8>, usize, Vec<usize>) {
        let start = 9;
        let mut buf = vec![0xAA; start];
        let mut starts = Vec::new();
        for (i, &chrom) in chroms.iter().enumerate() {
            starts.push(buf.len());
            buf.extend(record(
                chrom,
                1000 + i32::try_from(i).unwrap(),
                &[b"ACGT", b"A"],
                1,
            ));
        }
        (buf, start, starts)
    }

    #[test]
    fn the_tiling_must_anchor_at_the_known_first_record() {
        // If the sieve misses the *first* record, the survivors still tile each
        // other perfectly — just starting one record late. Anchoring at
        // `candidates[0]` instead of `start` would accept that and silently
        // drop a record, which is the one way this design can lose data rather
        // than merely lose speed.
        let (buf, start, expected) = mixed(&[3, 0, 1]);
        let scan = scan_records_speculative(&buf, start, 1, 2).unwrap();

        assert_eq!(
            scan.validated, 2,
            "contig 3 is out of range, so the first record must be missed"
        );
        assert!(
            matches!(scan.proof, Proof::Walked { .. }),
            "an unclaimed leading record must break the tiling"
        );
        assert_eq!(scan.offsets, expected, "the fallback must find all three");
    }

    #[test]
    fn the_tiling_must_account_for_the_bytes_after_the_last_survivor() {
        // The mirror image: the sieve misses the *last* record. What remains is
        // a whole record, not a partial one, so accepting it as a clean tail
        // would drop it.
        let (buf, start, expected) = mixed(&[0, 1, 3]);
        let scan = scan_records_speculative(&buf, start, 1, 2).unwrap();

        assert_eq!(scan.validated, 2, "contig 3 is out of range");
        assert!(
            matches!(scan.proof, Proof::Walked { .. }),
            "a whole unclaimed trailing record is not a partial record"
        );
        assert_eq!(scan.offsets, expected);
        assert_eq!(scan.tail, buf.len());
    }

    #[test]
    fn validation_rejects_what_the_cheap_sieve_admits() {
        // The sieve checks framing only — lengths, n_sample, CHROM — so a
        // record whose framing is intact but whose typed-value chain is corrupt
        // sails through it. Stage 2 is what catches that, and on real files the
        // two stages agree exactly, so nothing else here proves stage 2 does
        // any work at all.
        let (mut buf, start, _) = buffer(3, 1);
        let (offsets, _) = scan_records(&buf, start).unwrap();

        // Break the second record's first allele descriptor: claim 14 bytes of
        // string where 4 were written. l_shared and l_indiv are untouched, so
        // the framing still adds up and only the chain walk disagrees.
        let id_at = offsets[1] + 8 + SITE_CORE_SIZE + 1;
        assert_eq!(buf[id_at], 0x47, "expected the 4-character REF descriptor");
        buf[id_at] = 0xE7;

        let sieved = sieve(&buf, start, 1, 4);
        let validated = validate(&buf, &sieved);
        assert_eq!(sieved.len(), 3, "the sieve sees framing only");
        assert_eq!(
            validated.len(),
            2,
            "validation must reject the record whose fields disagree with l_shared"
        );
    }

    /// A 32-byte prefix that satisfies the sieve but decodes into nothing.
    ///
    /// `l_shared` is the bare 24-byte minimum, so the shared block ends before
    /// the ID that `n_allele = 1` promises. The sieve reads only these 32 bytes
    /// and cannot tell; the chain walk cannot miss it.
    fn decoy_prefix() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&24u32.to_le_bytes()); // l_shared: the minimum
        out.extend_from_slice(&0u32.to_le_bytes()); // l_indiv
        out.extend_from_slice(&0i32.to_le_bytes()); // CHROM, in range
        out.extend_from_slice(&100i32.to_le_bytes()); // POS
        out.extend_from_slice(&1i32.to_le_bytes()); // rlen
        out.extend_from_slice(&0x7F80_0001u32.to_le_bytes()); // QUAL
        out.extend_from_slice(&0u16.to_le_bytes()); // n_info
        out.extend_from_slice(&1u16.to_le_bytes()); // n_allele: promises a REF
        out.extend_from_slice(&1u32.to_le_bytes()[..3]); // n_sample matches
        out.push(0); // n_fmt, consistent with l_indiv == 0
        assert_eq!(out.len(), 32);
        out
    }

    #[test]
    fn validation_is_what_lets_the_tiling_survive_a_sieve_false_positive() {
        // Stage 2 is the one phase with no evidence from real data behind it:
        // across 399M candidate offsets on four files the sieve was already
        // exact, so validation pruned nothing and no measurement can justify
        // its cost. This is the case it exists for, built by hand.
        //
        // Without it the decoy survives, breaks the tiling, and the whole batch
        // falls back to the serial walk — correct, but at the cost the design
        // exists to avoid.
        let mut payload = decoy_prefix();
        payload.resize(64, 0x00);

        let mut shared = Vec::new();
        shared.extend_from_slice(&2i32.to_le_bytes());
        shared.extend_from_slice(&500i32.to_le_bytes());
        shared.extend_from_slice(&1i32.to_le_bytes());
        shared.extend_from_slice(&0x7F80_0001u32.to_le_bytes());
        shared.extend_from_slice(&0u16.to_le_bytes());
        shared.extend_from_slice(&1u16.to_le_bytes());
        shared.extend_from_slice(&1u32.to_le_bytes()[..3]);
        shared.push(0);
        shared.push(0x07); // ID missing
        shared.push(0xF7); // one long allele: the decoy's hiding place
        shared.push(0x13);
        shared.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        shared.extend_from_slice(&payload);
        shared.push(0x00); // FILTER missing

        let start = 9;
        let mut buf = vec![0xAA; start];
        let expected = vec![buf.len()];
        buf.extend_from_slice(&u32::try_from(shared.len()).unwrap().to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&shared);

        let sieved = sieve(&buf, start, 1, 4);
        assert_eq!(
            sieved.len(),
            2,
            "the decoy must pass the sieve, or this tests nothing"
        );
        let validated = validate(&buf, &sieved);
        assert_eq!(validated.len(), 1, "validation must reject the decoy");

        let scan = scan_records_speculative(&buf, start, 1, 4).unwrap();
        assert_eq!(
            scan.proof,
            Proof::Tiled,
            "with the decoy pruned the survivors tile and no walk is needed"
        );
        assert_eq!(scan.offsets, expected);

        // And the fallback would still be correct without it, which is why
        // dropping stage 2 would cost speed rather than accuracy.
        assert_eq!(prove_tiling(&buf, start, &sieved), None);
    }

    #[test]
    fn an_empty_range_tiles_trivially() {
        let buf = vec![0xAA; 9];
        let scan = scan_records_speculative(&buf, 9, 1, 4).unwrap();
        assert_eq!(scan.proof, Proof::Tiled);
        assert!(scan.offsets.is_empty());
        assert_eq!(scan.tail, 9);
    }
}
