#![doc(hidden)]

mod four_bit_packed;

pub use self::four_bit_packed::FourBitPacked;
use super::Sequence;

pub enum SequenceRef<'a> {
    FourBitPacked(FourBitPacked<'a>),
    Raw(&'a [u8]),
    Sequence(Box<dyn Sequence + 'a>),
}

impl SequenceRef<'_> {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::FourBitPacked(inner) => inner.is_empty(),
            Self::Raw(src) => src.is_empty(),
            Self::Sequence(sequence) => sequence.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::FourBitPacked(inner) => inner.len(),
            Self::Raw(src) => src.len(),
            Self::Sequence(sequence) => sequence.len(),
        }
    }

    pub fn get(&self, i: usize) -> Option<u8> {
        match self {
            Self::FourBitPacked(inner) => inner.get(i),
            Self::Raw(src) => src.get(i).copied(),
            Self::Sequence(sequence) => sequence.get(i),
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = u8> + '_> {
        match self {
            Self::FourBitPacked(inner) => Box::new(inner.iter()),
            Self::Raw(src) => Box::new(src.iter().copied()),
            Self::Sequence(sequence) => sequence.iter(),
        }
    }
}
