use std::io;

use fritillaria_sam::alignment::record::{Sequence, SequenceRef};

use super::num::{write_u8, write_u32_le};

const EQ: u8 = b'=';

pub(super) fn write_length(dst: &mut Vec<u8>, base_count: usize) -> io::Result<()> {
    let n =
        u32::try_from(base_count).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    write_u32_le(dst, n);

    Ok(())
}

pub(super) fn write_sequence(
    dst: &mut Vec<u8>,
    read_length: usize,
    sequence: SequenceRef<'_>,
) -> io::Result<()> {
    if sequence.is_empty() {
        return Ok(());
    }

    // § 1.4.10 "`SEQ`" (2022-08-22): "If not a '*', the length of the sequence must equal the sum
    // of lengths of `M`/`I`/`S`/`=`/`X` operations in `CIGAR`."
    if read_length > 0 && sequence.len() != read_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "read length-sequence length mismatch",
        ));
    }

    match sequence {
        SequenceRef::FourBitPacked(s) => write_four_bit_packed_sequence(dst, s.as_ref()),
        SequenceRef::Raw(s) => write_raw_sequence(dst, s),
        SequenceRef::Sequence(s) => write_generic_sequence(dst, s)?,
    }

    Ok(())
}

fn write_four_bit_packed_sequence(dst: &mut Vec<u8>, src: &[u8]) {
    dst.extend(src);
}

fn write_raw_sequence(dst: &mut Vec<u8>, src: &[u8]) {
    let (chunks, remainder) = src.as_chunks::<2>();

    dst.extend(chunks.iter().map(|&[l, r]| pack_bases(l, r)));

    if let &[l] = remainder {
        // § 4.2.3 "SEQ and QUAL encoding" (2021-06-03): "When `l_seq` is odd the bottom 4 bits of
        // the last byte are undefined, but we recommend writing these as zero."
        let b = pack_bases(l, EQ);
        dst.push(b);
    }
}

fn write_generic_sequence<S>(dst: &mut Vec<u8>, sequence: S) -> io::Result<()>
where
    S: Sequence,
{
    let mut bases = sequence.iter();

    while let Some(l) = bases.next() {
        // § 4.2.3 "SEQ and QUAL encoding" (2021-06-03): "When `l_seq` is odd the bottom 4 bits of
        // the last byte are undefined, but we recommend writing these as zero."
        let r = bases.next().unwrap_or(EQ);
        let n = pack_bases(l, r);
        write_u8(dst, n);
    }

    Ok(())
}

fn pack_bases(l: u8, r: u8) -> u8 {
    (encode_base(l) << 4) | encode_base(r)
}

// § 4.2.3 "SEQ and QUAL encoding" (2023-11-16): "The case-insensitive base codes [...] are mapped
// to [0, 15] respectively with all other characters mapping to 'N' (value 15)".
fn encode_base(n: u8) -> u8 {
    const CODES: [u8; 256] = build_codes();
    CODES[usize::from(n)]
}

const fn build_codes() -> [u8; 256] {
    // § 4.2.3 "SEQ and QUAL encoding" (2024-11-06)
    const BASES: [u8; 16] = *b"=ACMGRSVTWYHKDBN";
    const N: u8 = 0x0f;

    let mut i = 0;
    let mut codes = [N; 256];

    while i < BASES.len() {
        let base = BASES[i];

        // SAFETY: `i < BASES.len() <= u8::MAX`.
        let code = i as u8;

        // SAFETY: `base <= b'Y' < b'y' < codes.len() == 256`.
        codes[base as usize] = code;
        codes[base.to_ascii_lowercase() as usize] = code;

        i += 1;
    }

    codes
}

#[cfg(test)]
mod tests {
    use fritillaria_sam::alignment::record_buf::Sequence as SequenceBuf;

    use super::*;

    #[test]
    fn test_write_length() -> io::Result<()> {
        let mut buf = Vec::new();
        write_length(&mut buf, 8)?;
        assert_eq!(buf, [0x08, 0x00, 0x00, 0x00]);
        Ok(())
    }

    #[cfg(not(target_pointer_width = "16"))]
    #[test]
    fn test_write_length_with_out_of_range_base_count() {
        let mut buf = Vec::new();

        assert!(matches!(
            write_length(&mut buf, usize::MAX),
            Err(e) if e.kind() == io::ErrorKind::InvalidInput
        ));
    }

    #[test]
    fn test_write_sequence() -> Result<(), Box<dyn std::error::Error>> {
        fn t(buf: &mut Vec<u8>, sequence: &SequenceBuf, expected: &[u8]) -> io::Result<()> {
            buf.clear();
            let s = SequenceRef::Sequence(Box::new(sequence));
            write_sequence(buf, s.len(), s)?;
            assert_eq!(buf, expected);
            Ok(())
        }

        let mut buf = Vec::new();

        t(&mut buf, &SequenceBuf::default(), &[])?;
        t(&mut buf, &SequenceBuf::from(b"ACG"), &[0x12, 0x40])?;
        t(&mut buf, &SequenceBuf::from(b"ACGT"), &[0x12, 0x48])?;

        buf.clear();
        let sequence = SequenceBuf::default();
        let s = SequenceRef::Sequence(Box::new(&sequence));
        write_sequence(&mut buf, 2, s)?;
        assert!(buf.is_empty());

        buf.clear();
        let sequence = SequenceBuf::from(b"A");
        let s = SequenceRef::Sequence(Box::new(&sequence));
        assert!(matches!(
            write_sequence(&mut buf, 2, s),
            Err(e) if e.kind() == io::ErrorKind::InvalidInput,
        ));

        Ok(())
    }

    #[test]
    fn test_write_four_bit_packed_sequence() {
        let mut dst = Vec::new();
        let sequence = &[0x12, 0x48]; // ACGT
        write_four_bit_packed_sequence(&mut dst, sequence);
        assert_eq!(dst, sequence);
    }

    #[test]
    fn test_write_raw_sequence() {
        let mut dst = Vec::new();

        dst.clear();
        let sequence = b"ACG";
        write_raw_sequence(&mut dst, sequence);
        assert_eq!(dst, [0x12, 0x40]);

        dst.clear();
        let sequence = b"ACGT";
        write_raw_sequence(&mut dst, sequence);
        assert_eq!(dst, [0x12, 0x48]);
    }

    #[test]
    fn test_pack_bases() {
        assert_eq!(pack_bases(b'A', b'C'), 0x12);
    }

    #[test]
    fn test_encode_base() {
        const BASES: [u8; 16] = *b"=ACMGRSVTWYHKDBN";

        for (i, b) in (0..).zip(BASES) {
            assert_eq!(encode_base(b), i);
            assert_eq!(encode_base(b.to_ascii_lowercase()), i);
        }

        assert_eq!(encode_base(b'X'), 15);
        assert_eq!(encode_base(b'x'), 15);
    }
}
