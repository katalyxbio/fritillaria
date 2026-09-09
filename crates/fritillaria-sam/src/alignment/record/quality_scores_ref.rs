use super::QualityScores;

#[doc(hidden)]
pub enum QualityScoresRef<'a> {
    Raw(&'a [u8]),
    Offset(&'a [u8], u8),
    QualityScores(Box<dyn QualityScores + 'a>),
}

impl QualityScoresRef<'_> {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Raw(src) => src.is_empty(),
            Self::Offset(src, _) => src.is_empty(),
            Self::QualityScores(quality_scores) => quality_scores.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Raw(src) => src.len(),
            Self::Offset(src, _) => src.len(),
            Self::QualityScores(quality_scores) => quality_scores.len(),
        }
    }
}
