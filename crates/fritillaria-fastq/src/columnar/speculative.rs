//! The parallelisable scan: sieve candidate line starts, then prove they tile.
//!
//! The CPU reference a device kernel is diffed against, and the same design as
//! `fritillaria_bcf::columnar::speculative` — with two differences, both of which
//! make FASTQ the simpler case.
//!
//! # One stage, not two
//!
//! BCF sieves cheaply and then runs a full typed-value walk over the survivors,
//! because its sieve lets through about 12.7 candidates per 64 KiB block. FASTQ's
//! validator *is* the full check — it reads the whole record's framing — and
//! measured over 1,626,353 byte offsets of real ONT, HiFi and Illumina data it
//! admitted **nothing but true record starts**. There is no second stage because
//! there is nothing for one to prune.
//!
//! That is a statement about the data measured, not a theorem. The guarantee is
//! the tiling proof below. See `docs/fastq-boundaries.md`.
//!
//! # Candidates are line starts, not every byte
//!
//! A record begins just after a newline (or at `start`). Restricting candidates
//! to those offsets cuts the set by roughly the average line length before any
//! validation runs, and costs one trivially parallel pass to find them.
//!
//! # The proof is what makes speculation safe
//!
//! Survivors being exact on the data measured is evidence, not a guarantee. The
//! guarantee is [`prove_tiling`]: if the survivors, in order, tile the buffer —
//! the first is the known first record, each one's end is exactly the next, and
//! the remainder runs past the end — then they **are** the true chain, by
//! induction on lengths each record determines from its own bytes. A failed
//! tiling falls back to the serial walk, so a wrong guess costs work and never
//! accuracy.

use fritillaria_core::Result;

use crate::columnar::record::{bounds_at, scan_records};

/// How a scan's boundaries were established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proof {
    /// The survivors tiled the buffer, so they are the chain. Fully parallel.
    Tiled,
    /// The tiling failed and the boundaries came from the serial walk.
    ///
    /// Correct, and exactly as fast as the design this replaces. A driver
    /// should count these rather than assume they are zero.
    Walked {
        /// Survivors the sieve produced, for diagnosing why it failed.
        survivors: usize,
        /// Records the walk actually found.
        records: usize,
    },
}

/// The result of a speculative scan.
#[derive(Clone, Debug)]
pub struct SpeculativeScan {
    /// Record start offsets, in file order.
    pub offsets: Vec<usize>,
    /// Offset of the first incompletely-buffered record.
    pub tail: usize,
    /// Candidate line starts the sieve considered.
    pub candidates: usize,
    /// Candidates that survived the validator.
    pub survivors: usize,
    /// Which route produced the offsets.
    pub proof: Proof,
}

/// Every offset that could begin a record: `start`, and each byte after a
/// newline.
///
/// Trivially parallel — one thread per byte, testing `buf[i - 1] == b'\n'`.
#[must_use]
pub fn candidate_starts(buf: &[u8], start: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if start < buf.len() {
        out.push(start);
    }
    let mut at = start;
    while let Some(i) = memchr::memchr(b'\n', &buf[at..]) {
        let next = at + i + 1;
        if next >= buf.len() {
            break;
        }
        out.push(next);
        at = next;
    }
    out
}

/// Keeps the candidates that look like records.
///
/// One thread per candidate on a device; nothing here depends on any other
/// candidate's result, which is the property the whole design rests on.
#[must_use]
pub fn sieve(buf: &[u8], candidates: &[usize]) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&pos| bounds_at(buf, pos).is_some())
        .collect()
}

/// Where the survivors tile up to, or `None` if they do not tile.
///
/// The survivors must start at `start`, each end exactly where the next begins,
/// and the last must either end the buffer or leave a remainder too short to be
/// a record. Anything else means the sieve missed one or admitted one, and the
/// caller must fall back.
#[must_use]
pub fn prove_tiling(buf: &[u8], start: usize, survivors: &[usize]) -> Option<usize> {
    let mut at = start;
    for &candidate in survivors {
        if candidate != at {
            return None;
        }
        let bounds = bounds_at(buf, candidate)?;
        let end = at + bounds.record_end as usize;
        if end <= at || end > buf.len() {
            return None;
        }
        at = end;
    }

    // Whatever is left must not be a whole record the sieve missed. If it is,
    // the tiling is incomplete and claiming otherwise would silently drop it.
    if at < buf.len() && bounds_at(buf, at).is_some() {
        return None;
    }
    Some(at)
}

/// Finds record boundaries, speculatively where possible.
///
/// Always returns what [`scan_records`] would; [`SpeculativeScan::proof`] says
/// which route got there.
pub fn scan_records_speculative(buf: &[u8], start: usize) -> Result<SpeculativeScan> {
    let candidates = candidate_starts(buf, start);
    let survivors = sieve(buf, &candidates);

    if let Some(tail) = prove_tiling(buf, start, &survivors) {
        return Ok(SpeculativeScan {
            offsets: survivors.clone(),
            tail,
            candidates: candidates.len(),
            survivors: survivors.len(),
            proof: Proof::Tiled,
        });
    }

    let (offsets, tail) = scan_records(buf, start)?;
    let records = offsets.len();
    Ok(SpeculativeScan {
        offsets,
        tail,
        candidates: candidates.len(),
        survivors: survivors.len(),
        proof: Proof::Walked {
            survivors: survivors.len(),
            records,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columnar::record::tests::build_record;

    fn buffer(specs: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (name, sequence) in specs {
            buf.extend_from_slice(&build_record(name, sequence, b""));
        }
        buf
    }

    #[test]
    fn the_speculative_answer_matches_the_serial_one() {
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGTAC"), (b"c", b"AC")]);
        let scan = scan_records_speculative(&buf, 0).unwrap();
        let (expected, tail) = scan_records(&buf, 0).unwrap();

        assert_eq!(scan.offsets, expected);
        assert_eq!(scan.tail, tail);
        assert_eq!(
            scan.proof,
            Proof::Tiled,
            "real data must take the fast path"
        );
    }

    #[test]
    fn candidates_are_line_starts_and_outnumber_records() {
        // The sieve's input is four times the record count, which is the cost
        // being traded for parallelism.
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT")]);
        let candidates = candidate_starts(&buf, 0);
        assert_eq!(candidates.len(), 8, "four lines per record");
        assert_eq!(sieve(&buf, &candidates).len(), 2);
    }

    #[test]
    fn a_decoy_at_a_quality_line_start_is_sieved_out() {
        // The case the validator exists for, and the one real data supplies 83
        // instances of in testdata/illumina.fastq.gz.
        let mut buf = build_record(b"r0", b"ACGT", b"");
        let quality_start = buf.len() - 5;
        buf[quality_start] = b'@';
        buf.extend_from_slice(&build_record(b"r1", b"TTTT", b""));

        let candidates = candidate_starts(&buf, 0);
        assert!(
            candidates.contains(&quality_start),
            "the decoy must reach the sieve, or this tests nothing"
        );

        let scan = scan_records_speculative(&buf, 0).unwrap();
        assert_eq!(scan.offsets.len(), 2);
        assert_eq!(scan.proof, Proof::Tiled);
    }

    #[test]
    fn a_partial_trailing_record_is_left_in_the_tail() {
        let mut buf = buffer(&[(b"whole", b"ACGT")]);
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"partial", b"ACGTACGT", b""));
        buf.truncate(buf.len() - 4);

        let scan = scan_records_speculative(&buf, 0).unwrap();
        assert_eq!(scan.offsets.len(), 1);
        assert_eq!(scan.tail, boundary);
        assert_eq!(scan.proof, Proof::Tiled);
    }

    #[test]
    fn a_missed_record_fails_the_tiling_rather_than_being_dropped() {
        // Hand-built: prove_tiling is handed a survivor list with a record
        // removed, exactly what a sieve false negative would produce. It must
        // refuse rather than report a shorter tiling that happens to be
        // self-consistent up to that point.
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT"), (b"c", b"ACGT")]);
        let (all, _) = scan_records(&buf, 0).unwrap();
        assert_eq!(all.len(), 3);

        let missing_last = &all[..2];
        assert!(
            prove_tiling(&buf, 0, missing_last).is_none(),
            "a whole record left unclaimed must fail the proof"
        );

        let missing_middle = [all[0], all[2]];
        assert!(
            prove_tiling(&buf, 0, &missing_middle).is_none(),
            "a gap in the middle must fail the proof"
        );
    }

    #[test]
    fn a_spurious_survivor_fails_the_tiling() {
        // The other direction: a false positive between two real records does
        // not line up with the previous record's end.
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT")]);
        let (all, _) = scan_records(&buf, 0).unwrap();
        let spurious = [all[0], all[0] + 1, all[1]];
        assert!(prove_tiling(&buf, 0, &spurious).is_none());
    }

    #[test]
    fn falling_back_still_gives_the_right_answer() {
        // Drive the fallback directly: scan_records_speculative only reaches it
        // when the proof fails, which real data does not do, so the path is
        // exercised rather than waited for.
        let buf = buffer(&[(b"a", b"ACGT"), (b"b", b"ACGT")]);
        let (expected, tail) = scan_records(&buf, 0).unwrap();

        let survivors = vec![expected[0]]; // pretend the sieve missed one
        assert!(prove_tiling(&buf, 0, &survivors).is_none());

        let scan = scan_records_speculative(&buf, 0).unwrap();
        assert_eq!(scan.offsets, expected);
        assert_eq!(scan.tail, tail);
    }

    #[test]
    fn an_empty_buffer_tiles_trivially() {
        let scan = scan_records_speculative(&[], 0).unwrap();
        assert!(scan.offsets.is_empty());
        assert_eq!(scan.tail, 0);
        assert_eq!(scan.proof, Proof::Tiled);
    }
}
