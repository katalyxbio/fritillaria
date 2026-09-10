//! The two packed encodings in a BAM record.
//!
//! Both are small, both are easy to get subtly wrong, and both produce
//! plausible-looking garbage when wrong rather than an error — so they are
//! tested exhaustively.

/// Nibble-to-base table. The nibble *is* the index.
///
/// Note `=` at index 0 (meaning "same as reference") and `N` at 15.
pub const BASES: [u8; 16] = *b"=ACMGRSVTWYHKDBN";

/// CIGAR operation letters. The low 4 bits of a CIGAR `u32` index this.
pub const CIGAR_OPS: [u8; 9] = *b"MIDNSHP=X";

/// Largest length a single CIGAR op can encode.
///
/// The length occupies the upper 28 bits, so it saturates at 2^28 - 1.
pub const MAX_CIGAR_OP_LEN: u32 = (1 << 28) - 1;

/// A decoded CIGAR operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CigarOp {
    /// Operation letter, one of `MIDNSHP=X`.
    pub kind: u8,
    /// Operation length.
    pub len: u32,
}

/// Extracts the length from a raw CIGAR `u32` (the upper 28 bits).
#[must_use]
pub const fn cigar_op_len(raw: u32) -> u32 {
    raw >> 4
}

/// Extracts the operation letter from a raw CIGAR `u32` (the low 4 bits).
///
/// Returns `None` for the reserved values 9-15.
#[must_use]
pub const fn cigar_op_kind(raw: u32) -> Option<u8> {
    let index = (raw & 0xf) as usize;
    if index < CIGAR_OPS.len() {
        Some(CIGAR_OPS[index])
    } else {
        None
    }
}

/// Decodes a raw CIGAR `u32`.
#[must_use]
pub fn decode_cigar_op(raw: u32) -> Option<CigarOp> {
    Some(CigarOp {
        kind: cigar_op_kind(raw)?,
        len: cigar_op_len(raw),
    })
}

/// Decodes base `index` from a 4-bit packed sequence.
///
/// Two bases per byte, **high nibble first**. Returns `None` past the end.
#[must_use]
pub fn decode_base(packed: &[u8], index: usize) -> Option<u8> {
    let byte = *packed.get(index / 2)?;
    let nibble = if index.is_multiple_of(2) {
        byte >> 4
    } else {
        byte & 0x0f
    };
    Some(BASES[nibble as usize])
}

/// Number of bytes a 4-bit packed sequence of `l_seq` bases occupies.
///
/// Odd lengths round up; the final low nibble is padding.
#[must_use]
pub const fn packed_len(l_seq: usize) -> usize {
    l_seq.div_ceil(2)
}

/// Decodes `l_seq` bases into an ASCII sequence.
#[must_use]
pub fn decode_sequence(packed: &[u8], l_seq: usize) -> Vec<u8> {
    (0..l_seq).filter_map(|i| decode_base(packed, i)).collect()
}

/// Whether a quality string means "qualities absent".
///
/// Missing qualities are `0xFF` repeated `l_seq` times, **not** an empty field.
/// An empty slice for a non-empty read is malformed, not "missing".
#[must_use]
pub fn qualities_are_absent(qual: &[u8]) -> bool {
    !qual.is_empty() && qual.iter().all(|&q| q == 0xff)
}

#[cfg(test)]
mod tests {
    // Fixture builders use literal sizes and spell out CIGAR op codes
    // explicitly (`| 0` is op M); both read better than the lint-clean form.
    #![allow(clippy::cast_possible_wrap, clippy::identity_op)]

    use super::*;

    #[test]
    fn base_table_matches_the_spec_string() {
        assert_eq!(&BASES, b"=ACMGRSVTWYHKDBN");
        assert_eq!(BASES[0], b'=');
        assert_eq!(BASES[1], b'A');
        assert_eq!(BASES[2], b'C');
        assert_eq!(BASES[4], b'G');
        assert_eq!(BASES[8], b'T');
        assert_eq!(BASES[15], b'N');
    }

    #[test]
    fn decodes_high_nibble_first() {
        // 0x12 is A then C, not C then A. Getting this backwards reverse-
        // complements nothing and corrupts everything.
        let packed = [0x12];
        assert_eq!(decode_base(&packed, 0), Some(b'A'));
        assert_eq!(decode_base(&packed, 1), Some(b'C'));
    }

    #[test]
    fn decodes_a_full_sequence() {
        // ACGT = nibbles 1,2,4,8 = bytes 0x12 0x48
        assert_eq!(decode_sequence(&[0x12, 0x48], 4), b"ACGT");
    }

    #[test]
    fn odd_length_ignores_the_padding_nibble() {
        // 3 bases in 2 bytes; the final low nibble is padding and must not
        // appear as a fourth base.
        let packed = [0x12, 0x40];
        assert_eq!(decode_sequence(&packed, 3), b"ACG");
        assert_eq!(packed_len(3), 2);
    }

    #[test]
    fn packed_len_rounds_up() {
        assert_eq!(packed_len(0), 0);
        assert_eq!(packed_len(1), 1);
        assert_eq!(packed_len(2), 1);
        assert_eq!(packed_len(3), 2);
        assert_eq!(packed_len(4), 2);
    }

    #[test]
    fn every_nibble_decodes() {
        for nibble in 0u8..16 {
            let packed = [nibble << 4];
            assert_eq!(decode_base(&packed, 0), Some(BASES[nibble as usize]));
        }
    }

    #[test]
    fn cigar_splits_into_length_and_op() {
        // 10M => len 10, op 0 ('M')
        let raw = (10 << 4) | 0;
        assert_eq!(cigar_op_len(raw), 10);
        assert_eq!(cigar_op_kind(raw), Some(b'M'));
        assert_eq!(
            decode_cigar_op(raw),
            Some(CigarOp {
                kind: b'M',
                len: 10
            })
        );
    }

    #[test]
    fn cigar_op_table_matches_the_spec_string() {
        for (index, &letter) in CIGAR_OPS.iter().enumerate() {
            assert_eq!(cigar_op_kind(index as u32), Some(letter));
        }
        assert_eq!(cigar_op_kind(0), Some(b'M'));
        assert_eq!(cigar_op_kind(4), Some(b'S'));
        assert_eq!(cigar_op_kind(8), Some(b'X'));
    }

    #[test]
    fn reserved_cigar_ops_are_rejected() {
        // 9-15 are undefined; silently mapping them to a real op would hide
        // corruption.
        for reserved in 9u32..16 {
            assert_eq!(cigar_op_kind(reserved), None);
        }
    }

    #[test]
    fn cigar_length_saturates_at_28_bits() {
        let raw = u32::MAX;
        assert_eq!(cigar_op_len(raw), MAX_CIGAR_OP_LEN);
        assert_eq!(cigar_op_len(raw), 268_435_455);
    }

    #[test]
    fn absent_qualities_are_all_0xff_not_empty() {
        assert!(qualities_are_absent(&[0xff, 0xff, 0xff]));
        assert!(!qualities_are_absent(&[0xff, 0x20, 0xff]));
        assert!(
            !qualities_are_absent(&[]),
            "an empty field is malformed, not 'absent'"
        );
    }
}
