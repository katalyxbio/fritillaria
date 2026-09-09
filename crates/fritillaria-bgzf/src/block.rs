//! BGZF block framing.
//!
//! ```text
//! offset  size    field
//! 0       1       ID1   = 0x1f
//! 1       1       ID2   = 0x8b
//! 2       1       CM    = 8 (deflate)
//! 3       1       FLG   = FEXTRA (0x04) set
//! 4       4       MTIME
//! 8       1       XFL
//! 9       1       OS
//! 10      2       XLEN
//! 12      XLEN    extra subfields, one of which must be `BC`
//! ..      ..      CDATA (raw deflate)
//! bsize-8 4       CRC32 of the uncompressed payload
//! bsize-4 4       ISIZE, uncompressed payload length
//! ```
//!
//! The `BC` subfield is `SI1='B'`, `SI2='C'`, `SLEN=2`, then a `u16` holding
//! **total block size minus one**. That minus-one is the classic BGZF bug:
//! it exists so a full 65536-byte block fits in a `u16`.

use fritillaria_core::{Error, MAX_BLOCK_SIZE, Result};

/// Fixed part of the gzip header preceding the extra subfields.
pub const FIXED_HEADER_SIZE: usize = 12;
/// Size of the CRC32 + ISIZE trailer.
pub const TRAILER_SIZE: usize = 8;
/// gzip `FEXTRA` flag; BGZF requires it.
const FLG_FEXTRA: u8 = 0x04;
/// First byte of the BGZF extra subfield identifier.
const SI1: u8 = b'B';
/// Second byte of the BGZF extra subfield identifier.
const SI2: u8 = b'C';

/// The 28-byte empty block that must terminate every BGZF file.
///
/// Its absence is how `samtools` decides a file is truncated, so a writer that
/// omits it produces output that looks corrupt to every downstream tool.
pub const EOF_BLOCK: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Whether `buf` begins with the standard EOF marker.
#[must_use]
pub fn is_eof_block(buf: &[u8]) -> bool {
    buf.len() >= EOF_BLOCK.len() && buf[..EOF_BLOCK.len()] == EOF_BLOCK
}

/// A parsed BGZF block header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    /// Total block size in bytes: header + payload + trailer.
    ///
    /// This is the `BC` value *plus one*.
    pub block_size: usize,
    /// Length of the header, i.e. offset of the deflate payload.
    pub header_size: usize,
}

impl BlockHeader {
    /// Length of the deflate payload.
    #[must_use]
    pub const fn payload_size(&self) -> usize {
        self.block_size - self.header_size - TRAILER_SIZE
    }
}

/// Reads a `u16` little-endian at `pos`, or `None` if out of bounds.
fn read_u16(buf: &[u8], pos: usize) -> Option<u16> {
    let bytes = buf.get(pos..pos + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Reads a `u32` little-endian at `pos`, or `None` if out of bounds.
pub(crate) fn read_u32(buf: &[u8], pos: usize) -> Option<u32> {
    let bytes = buf.get(pos..pos + 4)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Total header length (fixed part plus extra field) for the block at `buf[0]`.
///
/// Returns `None` when `buf` is too short to determine it. Callers walking a
/// partially-buffered stream use this to tell "need more bytes" apart from
/// "corrupt", which [`parse_header`] cannot distinguish on its own.
#[must_use]
pub fn header_len(buf: &[u8]) -> Option<usize> {
    if buf.len() < FIXED_HEADER_SIZE {
        return None;
    }
    Some(FIXED_HEADER_SIZE + read_u16(buf, 10)? as usize)
}

/// Parses the header of the block starting at `buf[0]`.
///
/// `offset` is the block's absolute position in the file and is used only for
/// error reporting.
pub fn parse_header(buf: &[u8], offset: u64) -> Result<BlockHeader> {
    let invalid = |reason| Error::InvalidBlock { offset, reason };

    if buf.len() < FIXED_HEADER_SIZE {
        return Err(invalid("truncated header"));
    }
    if buf[0] != 0x1f || buf[1] != 0x8b {
        return Err(invalid("not a gzip member (bad magic)"));
    }
    if buf[2] != 8 {
        return Err(invalid("unsupported compression method"));
    }
    if buf[3] & FLG_FEXTRA == 0 {
        // A plain gzip member. Reaching here usually means the file is an
        // ordinary .gz rather than a bgzipped one — which cannot be
        // block-parallelised at all, so the distinction matters to the caller.
        return Err(invalid("missing FEXTRA; not BGZF (plain gzip?)"));
    }

    let xlen = read_u16(buf, 10).ok_or_else(|| invalid("truncated XLEN"))? as usize;
    let header_size = FIXED_HEADER_SIZE + xlen;
    if buf.len() < header_size {
        return Err(invalid("truncated extra field"));
    }

    let bsize = find_bc_subfield(&buf[FIXED_HEADER_SIZE..header_size])
        .ok_or_else(|| invalid("missing BC subfield; not BGZF"))?;

    // The +1 is the whole point of the BC encoding.
    let block_size = bsize as usize + 1;

    if block_size > MAX_BLOCK_SIZE {
        return Err(invalid("block size exceeds 64 KiB"));
    }
    if block_size < header_size + TRAILER_SIZE {
        return Err(invalid("block size smaller than its own framing"));
    }

    Ok(BlockHeader {
        block_size,
        header_size,
    })
}

/// Scans gzip extra subfields for `BC`, returning its raw `u16` value.
///
/// The subfield is not required to be first, and unknown subfields must be
/// skipped rather than rejected.
fn find_bc_subfield(mut extra: &[u8]) -> Option<u16> {
    while extra.len() >= 4 {
        let si1 = extra[0];
        let si2 = extra[1];
        let slen = u16::from_le_bytes([extra[2], extra[3]]) as usize;
        let payload = extra.get(4..4 + slen)?;

        if si1 == SI1 && si2 == SI2 {
            if slen != 2 {
                return None;
            }
            return Some(u16::from_le_bytes([payload[0], payload[1]]));
        }

        extra = &extra[4 + slen..];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a header with the given BC value and optional extra subfields.
    fn header_with(bsize_minus_one: u16, leading_subfields: &[u8]) -> Vec<u8> {
        let xlen = leading_subfields.len() + 6;
        let mut buf = vec![0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff];
        buf.extend_from_slice(&(xlen as u16).to_le_bytes());
        buf.extend_from_slice(leading_subfields);
        buf.extend_from_slice(&[SI1, SI2, 0x02, 0x00]);
        buf.extend_from_slice(&bsize_minus_one.to_le_bytes());
        buf
    }

    #[test]
    fn eof_block_is_self_consistent() {
        let header = parse_header(&EOF_BLOCK, 0).unwrap();
        assert_eq!(header.block_size, EOF_BLOCK.len());
        assert_eq!(header.header_size, 18);
        assert_eq!(header.payload_size(), 2);
        assert!(is_eof_block(&EOF_BLOCK));
    }

    #[test]
    fn eof_block_trailer_is_empty() {
        // ISIZE and CRC32 of an empty payload are both zero.
        assert_eq!(read_u32(&EOF_BLOCK, EOF_BLOCK.len() - 8), Some(0));
        assert_eq!(read_u32(&EOF_BLOCK, EOF_BLOCK.len() - 4), Some(0));
    }

    #[test]
    fn bc_value_is_size_minus_one() {
        // The regression guard for the classic off-by-one: BC=27 means 28 bytes.
        let mut buf = header_with(27, &[]);
        buf.resize(28, 0);
        assert_eq!(parse_header(&buf, 0).unwrap().block_size, 28);
    }

    #[test]
    fn accepts_full_64kib_block() {
        // BC is u16 and holds size-1 exactly so that 65536 is representable.
        let mut buf = header_with(u16::MAX, &[]);
        buf.resize(MAX_BLOCK_SIZE, 0);
        assert_eq!(
            parse_header(&buf, 0).unwrap().block_size,
            MAX_BLOCK_SIZE,
            "a maximal block must parse, not overflow"
        );
    }

    #[test]
    fn skips_unknown_subfields_before_bc() {
        // Subfield ordering is not fixed by the spec; a parser that assumes BC
        // comes first works on htslib output and fails on other writers.
        let foreign = [b'Z', b'Z', 0x03, 0x00, 0xaa, 0xbb, 0xcc];
        let mut buf = header_with(99, &foreign);
        buf.resize(100, 0);

        let header = parse_header(&buf, 0).unwrap();
        assert_eq!(header.block_size, 100);
        assert_eq!(header.header_size, FIXED_HEADER_SIZE + foreign.len() + 6);
    }

    #[test]
    fn rejects_plain_gzip() {
        // FEXTRA clear: an ordinary .gz, which cannot be block-parallelised.
        let mut buf = header_with(27, &[]);
        buf[3] = 0x00;
        buf.resize(28, 0);

        let err = parse_header(&buf, 0).unwrap_err();
        assert!(
            matches!(err, Error::InvalidBlock { reason, .. } if reason.contains("FEXTRA")),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_missing_bc_subfield() {
        let mut buf = vec![0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff];
        buf.extend_from_slice(&7u16.to_le_bytes());
        buf.extend_from_slice(&[b'Z', b'Z', 0x03, 0x00, 0xaa, 0xbb, 0xcc]);
        buf.resize(64, 0);

        assert!(matches!(
            parse_header(&buf, 0).unwrap_err(),
            Error::InvalidBlock { reason, .. } if reason.contains("BC")
        ));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(parse_header(&[0x1f], 0).is_err());
        let mut buf = header_with(27, &[]);
        buf[0] = 0x00;
        assert!(parse_header(&buf, 0).is_err());
    }

    #[test]
    fn rejects_block_smaller_than_its_framing() {
        // BC=0 => block_size 1, which cannot hold an 18-byte header.
        let buf = header_with(0, &[]);
        assert!(matches!(
            parse_header(&buf, 0).unwrap_err(),
            Error::InvalidBlock { reason, .. } if reason.contains("framing")
        ));
    }

    #[test]
    fn error_carries_the_offset() {
        let err = parse_header(&[0x00, 0x00], 4096).unwrap_err();
        assert!(matches!(err, Error::InvalidBlock { offset: 4096, .. }));
    }
}
