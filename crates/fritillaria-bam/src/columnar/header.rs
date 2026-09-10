//! The BAM header.
//!
//! ```text
//! magic   4                 "BAM\1"
//! l_text  4                 length of the SAM text header
//! text    l_text            SAM text header (not NUL-terminated in general)
//! n_ref   4                 number of reference sequences
//!   per reference:
//!     l_name  4             length of name, including trailing NUL
//!     name    l_name        reference name
//!     l_ref   4             reference length
//! ```

use fritillaria_core::{Error, Result};

use crate::columnar::MAGIC;

/// One reference sequence entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceSequence {
    /// Name, with the trailing NUL stripped.
    pub name: Vec<u8>,
    /// Length in bases.
    pub length: u32,
}

/// A parsed BAM header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    /// The SAM text header, verbatim.
    pub text: Vec<u8>,
    /// Reference sequences, in the order that `refID` indexes.
    pub references: Vec<ReferenceSequence>,
    /// Byte offset at which the first alignment record begins.
    pub records_start: usize,
}

fn read_i32(buf: &[u8], pos: usize) -> Option<i32> {
    Some(i32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?))
}

/// Parses the header at the start of a decompressed BAM stream.
pub fn parse_header(buf: &[u8]) -> Result<Header> {
    let malformed = |position: usize, reason: &str| Error::Malformed {
        format: "bam header",
        position: position as u64,
        reason: reason.to_string(),
    };

    if buf.len() < 12 {
        return Err(malformed(0, "truncated header"));
    }
    if buf[..4] != MAGIC {
        return Err(malformed(0, "bad magic; expected BAM\\1"));
    }

    let l_text = read_i32(buf, 4).ok_or_else(|| malformed(4, "truncated l_text"))?;
    let l_text = usize::try_from(l_text).map_err(|_| malformed(4, "negative l_text"))?;

    let text_start: usize = 8;
    let text_end = text_start
        .checked_add(l_text)
        .ok_or_else(|| malformed(4, "l_text overflows"))?;
    let text = buf
        .get(text_start..text_end)
        .ok_or_else(|| malformed(4, "l_text runs past the end of the buffer"))?
        .to_vec();

    let n_ref = read_i32(buf, text_end).ok_or_else(|| malformed(text_end, "truncated n_ref"))?;
    let n_ref = usize::try_from(n_ref).map_err(|_| malformed(text_end, "negative n_ref"))?;

    let mut pos = text_end + 4;
    let mut references = Vec::with_capacity(n_ref.min(1024));

    for index in 0..n_ref {
        let l_name = read_i32(buf, pos)
            .ok_or_else(|| malformed(pos, &format!("truncated l_name for reference {index}")))?;
        let l_name = usize::try_from(l_name).map_err(|_| malformed(pos, "negative l_name"))?;
        pos += 4;

        let raw = buf
            .get(pos..pos + l_name)
            .ok_or_else(|| malformed(pos, &format!("truncated name for reference {index}")))?;
        // l_name counts the NUL terminator.
        let name = raw.strip_suffix(b"\0").unwrap_or(raw).to_vec();
        pos += l_name;

        let length = read_i32(buf, pos)
            .ok_or_else(|| malformed(pos, &format!("truncated l_ref for reference {index}")))?;
        pos += 4;

        references.push(ReferenceSequence {
            name,
            length: length.max(0).cast_unsigned(),
        });
    }

    Ok(Header {
        text,
        references,
        records_start: pos,
    })
}

#[cfg(test)]
mod tests {
    // Fixture builders use literal sizes and spell out CIGAR op codes
    // explicitly (`| 0` is op M); both read better than the lint-clean form.
    #![allow(clippy::cast_possible_wrap, clippy::identity_op)]

    use super::*;

    fn build_header(text: &[u8], refs: &[(&[u8], u32)]) -> Vec<u8> {
        let mut buf = MAGIC.to_vec();
        buf.extend_from_slice(&(text.len() as i32).to_le_bytes());
        buf.extend_from_slice(text);
        buf.extend_from_slice(&(refs.len() as i32).to_le_bytes());
        for (name, length) in refs {
            buf.extend_from_slice(&((name.len() + 1) as i32).to_le_bytes());
            buf.extend_from_slice(name);
            buf.push(0);
            buf.extend_from_slice(&length.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parses_text_and_references() {
        let raw = build_header(
            b"@HD\tVN:1.6\n",
            &[(b"chr1", 248_956_422), (b"chr2", 242_193_529)],
        );
        let header = parse_header(&raw).unwrap();

        assert_eq!(header.text, b"@HD\tVN:1.6\n");
        assert_eq!(header.references.len(), 2);
        assert_eq!(header.references[0].name, b"chr1");
        assert_eq!(header.references[0].length, 248_956_422);
        assert_eq!(header.references[1].name, b"chr2");
        assert_eq!(
            header.records_start,
            raw.len(),
            "records begin immediately after the reference list"
        );
    }

    #[test]
    fn parses_header_with_no_references() {
        let raw = build_header(b"@HD\tVN:1.6\n", &[]);
        let header = parse_header(&raw).unwrap();
        assert!(header.references.is_empty());
        assert_eq!(header.records_start, raw.len());
    }

    #[test]
    fn parses_header_with_empty_text() {
        let raw = build_header(b"", &[(b"chr1", 100)]);
        let header = parse_header(&raw).unwrap();
        assert!(header.text.is_empty());
        assert_eq!(header.references.len(), 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = build_header(b"", &[]);
        raw[3] = 0x02;
        assert!(parse_header(&raw).is_err());
    }

    #[test]
    fn rejects_truncation_at_each_stage() {
        let raw = build_header(b"@HD\n", &[(b"chr1", 100)]);
        for len in 0..raw.len() {
            assert!(
                parse_header(&raw[..len]).is_err(),
                "a header truncated to {len} bytes must not parse"
            );
        }
        assert!(parse_header(&raw).is_ok());
    }

    #[test]
    fn rejects_l_text_running_past_the_buffer() {
        let mut raw = build_header(b"@HD\n", &[]);
        raw[4..8].copy_from_slice(&1_000_000i32.to_le_bytes());
        assert!(parse_header(&raw).is_err());
    }
}
