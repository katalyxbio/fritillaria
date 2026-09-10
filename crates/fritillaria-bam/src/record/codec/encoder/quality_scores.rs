use std::{io, iter};

use fritillaria_sam::alignment::record::{QualityScores, QualityScoresRef};

use super::num::write_u8;

pub(super) fn write_quality_scores(
    dst: &mut Vec<u8>,
    base_count: usize,
    quality_scores: QualityScoresRef<'_>,
) -> io::Result<()> {
    // § 4.2.3 SEQ and QUAL encoding (2022-08-22)
    const MISSING: u8 = 255;

    if quality_scores.len() == base_count {
        match quality_scores {
            QualityScoresRef::Raw(s) => write_raw_quality_scores(dst, s)?,
            QualityScoresRef::Offset(s, offset) => write_offset_quality_scores(dst, s, offset)?,
            QualityScoresRef::QualityScores(s) => write_generic_quality_scores(dst, s)?,
        }
    } else if quality_scores.is_empty() {
        dst.extend(iter::repeat_n(MISSING, base_count));
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "sequence-quality scores length mismatch: expected {}, got {}",
                base_count,
                quality_scores.len()
            ),
        ));
    }

    Ok(())
}

fn write_raw_quality_scores(dst: &mut Vec<u8>, src: &[u8]) -> io::Result<()> {
    if src.iter().all(|&n| is_valid_score(n)) {
        dst.extend(src);
        Ok(())
    } else {
        Err(io::Error::from(io::ErrorKind::InvalidInput))
    }
}

fn write_offset_quality_scores(dst: &mut Vec<u8>, src: &[u8], offset: u8) -> io::Result<()> {
    for n in src {
        let m = n
            .checked_sub(offset)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;

        if is_valid_score(m) {
            dst.push(m);
        } else {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
    }

    Ok(())
}

fn write_generic_quality_scores<S>(dst: &mut Vec<u8>, quality_scores: S) -> io::Result<()>
where
    S: QualityScores,
{
    for result in quality_scores.iter() {
        let n = result?;

        if is_valid_score(n) {
            write_u8(dst, n);
        } else {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
    }

    Ok(())
}

fn is_valid_score(score: u8) -> bool {
    // § 4.2.3 "SEQ and QUAL encoding" (2023-05-24): "Base qualities are stored as bytes in the
    // range [0, 93]..."
    const MAX_SCORE: u8 = 93;
    score <= MAX_SCORE
}

#[cfg(test)]
mod tests {
    use fritillaria_sam::alignment::record_buf::QualityScores as QualityScoresBuf;

    use super::*;

    #[test]
    fn test_write_quality_scores() -> Result<(), Box<dyn std::error::Error>> {
        fn t(
            buf: &mut Vec<u8>,
            base_count: usize,
            quality_scores: &QualityScoresBuf,
            expected: &[u8],
        ) -> io::Result<()> {
            buf.clear();
            let s = QualityScoresRef::QualityScores(Box::new(quality_scores));
            write_quality_scores(buf, base_count, s)?;
            assert_eq!(buf, expected);
            Ok(())
        }

        let mut buf = Vec::new();

        t(&mut buf, 0, &QualityScoresBuf::default(), &[])?;
        t(
            &mut buf,
            4,
            &QualityScoresBuf::default(),
            &[0xff, 0xff, 0xff, 0xff],
        )?;

        let quality_scores = [45, 35, 43, 50].into_iter().collect();
        t(&mut buf, 4, &quality_scores, &[0x2d, 0x23, 0x2b, 0x32])?;

        buf.clear();
        let quality_scores = [45, 35, 43, 50].into_iter().collect();
        let s = QualityScoresRef::QualityScores(Box::new(&quality_scores));
        assert!(matches!(
            write_quality_scores(&mut buf, 3, s),
            Err(e) if e.kind() == io::ErrorKind::InvalidInput
        ));

        Ok(())
    }

    #[test]
    fn test_write_raw_quality_scores() -> io::Result<()> {
        let mut dst = Vec::new();

        dst.clear();
        let quality_scores = [45, 35, 43, 50];
        write_raw_quality_scores(&mut dst, &quality_scores)?;
        assert_eq!(dst, [45, 35, 43, 50]);

        dst.clear();
        let quality_scores = [255];
        assert!(matches!(
            write_raw_quality_scores(&mut dst, &quality_scores),
            Err(e) if e.kind() == io::ErrorKind::InvalidInput
        ));

        Ok(())
    }

    #[test]
    fn test_write_offset_quality_scores() -> io::Result<()> {
        let mut dst = Vec::new();

        dst.clear();
        let (quality_scores, offset) = (b"NDLS", b'!');
        write_offset_quality_scores(&mut dst, quality_scores, offset)?;
        assert_eq!(dst, [45, 35, 43, 50]);

        dst.clear();
        let quality_scores = [255];
        assert!(matches!(
            write_offset_quality_scores(&mut dst, &quality_scores, offset),
            Err(e) if e.kind() == io::ErrorKind::InvalidInput
        ));

        Ok(())
    }
}
