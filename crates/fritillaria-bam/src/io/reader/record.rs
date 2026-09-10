use std::{
    io::{self, Read},
    mem,
    ops::Range,
};

pub(super) fn read_record<R>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<usize>
where
    R: Read,
{
    let block_size = match read_block_size(reader)? {
        0 => return Ok(0),
        n => n,
    };

    buf.resize(block_size, 0);
    reader.read_exact(buf)?;

    validate(buf)?;

    Ok(block_size)
}

fn read_block_size<R>(reader: &mut R) -> io::Result<usize>
where
    R: Read,
{
    let mut buf = [0; mem::size_of::<u32>()];
    read_exact_or_eof(reader, &mut buf)?;
    let n = u32::from_le_bytes(buf);
    usize::try_from(n).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_exact_or_eof<R>(reader: &mut R, mut buf: &mut [u8]) -> io::Result<()>
where
    R: Read,
{
    let mut bytes_read = 0;

    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                buf = &mut buf[n..];
                bytes_read += n;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }

    if bytes_read > 0 && !buf.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn validate(src: &[u8]) -> io::Result<()> {
    const MIN_BUF_LENGTH: usize = 32;
    const NAME_LENGTH_INDEX: usize = 8;
    const CIGAR_OP_COUNT_RANGE: Range<usize> = 12..14;
    const READ_LENGTH_RANGE: Range<usize> = 16..20;

    if src.len() < MIN_BUF_LENGTH {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }

    let name_len = usize::from(src[NAME_LENGTH_INDEX]);

    let buf = &src[CIGAR_OP_COUNT_RANGE];
    // SAFETY: `buf.len() == mem::size_of::<u16>()`.
    let cigar_op_count = usize::from(u16::from_le_bytes(buf.try_into().unwrap()));

    let buf = &src[READ_LENGTH_RANGE];
    // SAFETY: `buf.len() == mem::size_of::<u32>()`.
    let base_count = usize::try_from(u32::from_le_bytes(buf.try_into().unwrap()))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let quality_scores_end = MIN_BUF_LENGTH
        + name_len
        + (cigar_op_count * mem::size_of::<u32>())
        + base_count.div_ceil(2)
        + base_count;

    if src.len() < quality_scores_end {
        Err(io::Error::from(io::ErrorKind::UnexpectedEof))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_block_size() -> io::Result<()> {
        let data = [0x08, 0x00, 0x00, 0x00];
        let mut reader = &data[..];
        assert_eq!(read_block_size(&mut reader)?, 8);

        let data = [];
        let mut reader = &data[..];
        assert_eq!(read_block_size(&mut reader)?, 0);

        let data = [0x08];
        let mut reader = &data[..];
        assert!(matches!(
            read_block_size(&mut reader),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof
        ));

        Ok(())
    }
}
