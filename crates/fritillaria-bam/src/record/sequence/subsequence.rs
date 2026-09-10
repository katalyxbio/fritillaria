use fritillaria_sam as sam;

use super::{Iter, decode_bases};

/// A BAM record subsequence.
#[derive(Debug, Eq, PartialEq)]
pub struct Subsequence<'a> {
    src: &'a [u8],
    // [start, end)
    start: usize,
    end: usize,
}

impl<'a> Subsequence<'a> {
    pub(super) fn new(src: &'a [u8], start: usize, end: usize) -> Self {
        Self { src, start, end }
    }

    /// Returns whether there are any bases.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bases.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns the base at the given index.
    pub fn get(&self, i: usize) -> Option<u8> {
        let j = self.start + i;

        if j < self.end {
            let k = j / 2;
            let n = self.src[k];
            let [l, r] = decode_bases(n);
            let b = if j.is_multiple_of(2) { l } else { r };
            Some(b)
        } else {
            None
        }
    }

    /// Returns an iterator over the bases in the subsequence.
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        Iter::new(self.src, self.start, self.end)
    }
}

impl sam::alignment::record::Sequence for Subsequence<'_> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, i: usize) -> Option<u8> {
        self.get(i)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = u8> + '_> {
        Box::new(self.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &[u8] = &[0x12];

    #[test]
    fn test_is_empty() {
        let subsequence = Subsequence::new(SRC, 0, 0);
        assert!(subsequence.is_empty());

        let subsequence = Subsequence::new(SRC, 0, 2);
        assert!(!subsequence.is_empty());
    }

    #[test]
    fn test_len() {
        let subsequence = Subsequence::new(SRC, 0, 0);
        assert_eq!(subsequence.len(), 0);

        let subsequence = Subsequence::new(SRC, 0, 2);
        assert_eq!(subsequence.len(), 2);
    }

    #[test]
    fn test_get() {
        let subsequence = Subsequence::new(SRC, 0, 0);
        assert!(subsequence.get(0).is_none());

        let subsequence = Subsequence::new(SRC, 1, 2);
        assert_eq!(subsequence.get(0), Some(b'C'));
    }

    #[test]
    fn test_iter() {
        let subsequence = Subsequence::new(SRC, 0, 0);
        assert!(subsequence.iter().next().is_none());

        let subsequence = Subsequence::new(SRC, 0, 2);
        assert_eq!(subsequence.iter().collect::<Vec<_>>(), [b'A', b'C']);
    }
}
