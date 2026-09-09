use std::{iter::FusedIterator, slice};

use super::decode_base;

pub struct Iter<'a> {
    iter: slice::Iter<'a, u8>,
    i: usize,
    base_count: usize,
    low_base: Option<u8>,
}

impl<'a> Iter<'a> {
    pub fn new(src: &'a [u8], base_count: usize) -> Self {
        Self {
            iter: src.iter(),
            i: 0,
            base_count,
            low_base: None,
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.base_count {
            return None;
        }

        let base = if let Some(n) = self.low_base.take() {
            Some(n)
        } else if let Some(&n) = self.iter.next() {
            let (h, l) = (decode_base(n >> 4), decode_base(n));
            self.low_base = Some(l);
            Some(h)
        } else {
            None
        };

        self.i += 1;

        base
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.base_count - self.i;
        (len, Some(len))
    }
}

impl ExactSizeIterator for Iter<'_> {}

impl FusedIterator for Iter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next() {
        let mut iter = Iter::new(&[], 0);
        assert!(iter.next().is_none());

        let iter = Iter::new(&[0x12, 0x40], 3);
        assert_eq!(iter.collect::<Vec<_>>(), b"ACG");

        let iter = Iter::new(&[0x12, 0x48], 4);
        assert_eq!(iter.collect::<Vec<_>>(), b"ACGT");
    }
}
