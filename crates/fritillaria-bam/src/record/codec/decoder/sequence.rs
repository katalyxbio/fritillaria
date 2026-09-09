use std::{error, fmt, num};

use fritillaria_sam::alignment::record_buf::Sequence;

/// An error when a raw BAM record sequence fails to parse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Unexpected EOF.
    UnexpectedEof,
    /// The length is invalid.
    InvalidLength(num::TryFromIntError),
}

impl error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::UnexpectedEof => None,
            Self::InvalidLength(e) => Some(e),
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected EOF"),
            Self::InvalidLength(_) => write!(f, "invalid length"),
        }
    }
}

pub(super) fn read_length(src: &mut &[u8]) -> Result<usize, DecodeError> {
    read_u32_le(src).and_then(|n| usize::try_from(n).map_err(DecodeError::InvalidLength))
}

const NIBBLE_PAIRS: [[u8; 2]; 256] = {
    const BASES: [u8; 16] = *b"=ACMGRSVTWYHKDBN";

    let mut table = [[0u8; 2]; 256];
    let mut i = 0;

    while i < 256 {
        table[i] = [BASES[i >> 4], BASES[i & 0xf]];
        i += 1;
    }

    table
};

pub(super) fn read_sequence(
    src: &mut &[u8],
    sequence: &mut Sequence,
    base_count: usize,
) -> Result<(), DecodeError> {
    let len = base_count.div_ceil(2);

    let (buf, rest) = src
        .split_at_checked(len)
        .ok_or(DecodeError::UnexpectedEof)?;

    *src = rest;

    let bases = buf.iter().flat_map(|&b| NIBBLE_PAIRS[b as usize]);

    let dst = sequence.as_mut();
    dst.clear();
    dst.extend(bases);
    dst.truncate(base_count);

    Ok(())
}

fn read_u32_le(src: &mut &[u8]) -> Result<u32, DecodeError> {
    let (buf, rest) = src.split_first_chunk().ok_or(DecodeError::UnexpectedEof)?;
    *src = rest;
    Ok(u32::from_le_bytes(*buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_length() {
        let mut src = &8u32.to_le_bytes()[..];
        assert_eq!(read_length(&mut src), Ok(8));

        let mut src = &[][..];
        assert_eq!(read_length(&mut src), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn test_read_sequence() -> Result<(), Box<dyn std::error::Error>> {
        fn t(mut src: &[u8], buf: &mut Sequence, expected: &Sequence) -> Result<(), DecodeError> {
            buf.as_mut().clear();
            read_sequence(&mut src, buf, expected.len())?;
            assert_eq!(buf, expected);
            Ok(())
        }

        let mut sequence = Sequence::default();

        t(&[], &mut sequence, &Sequence::default())?;
        t(&[0x12, 0x40], &mut sequence, &Sequence::from(b"ACG"))?;
        t(&[0x12, 0x48], &mut sequence, &Sequence::from(b"ACGT"))?;

        sequence.as_mut().clear();
        let mut src = &b""[..];
        assert_eq!(
            read_sequence(&mut src, &mut sequence, 4),
            Err(DecodeError::UnexpectedEof)
        );

        Ok(())
    }

    #[test]
    fn test_nibble_pairs() {
        let expected = *b"=ACMGRSVTWYHKDBN";

        for (code, base) in expected.iter().enumerate() {
            assert_eq!(NIBBLE_PAIRS[code << 4][0], *base, "high nibble {code}");
            assert_eq!(NIBBLE_PAIRS[code][1], *base, "low nibble {code}");
        }
    }
}
