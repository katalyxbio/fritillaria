mod iter;

use self::iter::Iter;

pub struct FourBitPacked<'a> {
    src: &'a [u8],
    base_count: usize,
}

impl<'a> FourBitPacked<'a> {
    pub fn new(src: &'a [u8], base_count: usize) -> Self {
        Self { src, base_count }
    }

    pub fn is_empty(&self) -> bool {
        self.base_count == 0
    }

    pub fn len(&self) -> usize {
        self.base_count
    }

    pub fn get(&self, i: usize) -> Option<u8> {
        if i < self.base_count
            && let Some(&n) = self.src.get(i / 2)
        {
            if i.is_multiple_of(2) {
                Some(decode_base(n >> 4))
            } else {
                Some(decode_base(n))
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> Iter<'a> {
        Iter::new(self.src, self.base_count)
    }
}

impl<'a> AsRef<[u8]> for FourBitPacked<'a> {
    fn as_ref(&self) -> &'a [u8] {
        self.src
    }
}

fn decode_base(n: u8) -> u8 {
    const BASE_LUT: &[u8; 16] = b"=ACMGRSVTWYHKDBN";
    BASE_LUT[usize::from(n & 0x0f)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_empty() {
        let sequence = FourBitPacked {
            src: &[],
            base_count: 0,
        };
        assert!(sequence.is_empty());

        let sequence = FourBitPacked {
            src: &[0x12, 0x40],
            base_count: 3,
        };
        assert!(!sequence.is_empty());
    }

    #[test]
    fn test_len() {
        let sequence = FourBitPacked {
            src: &[0x12, 0x40],
            base_count: 3,
        };

        assert_eq!(sequence.len(), 3);
    }

    #[test]
    fn test_get() {
        let sequence = FourBitPacked {
            src: &[0x12, 0x40],
            base_count: 3,
        };

        assert_eq!(sequence.get(0), Some(b'A'));
        assert_eq!(sequence.get(1), Some(b'C'));
        assert_eq!(sequence.get(2), Some(b'G'));
        assert!(sequence.get(3).is_none());
    }

    #[test]
    fn test_decode_base() {
        const EXPECTED_BASES: &[u8; 16] = b"=ACMGRSVTWYHKDBN";

        for (n, &expected) in (0..15).zip(EXPECTED_BASES) {
            assert_eq!(decode_base(n), expected);
        }

        assert_eq!(decode_base(16), b'=');
        assert_eq!(decode_base(255), b'N');
    }
}
