//! BGZF virtual offsets.
//!
//! A virtual offset packs "which block" and "where inside it" into one `u64`:
//! the upper 48 bits are the byte offset of the block's *first* byte in the
//! compressed stream, the lower 16 bits the offset into that block's
//! decompressed payload. This is the addressing used by BAI/CSI indexes.

use std::fmt;

/// Number of low bits reserved for the within-block offset.
const SHIFT: u32 = 16;
/// Mask selecting the within-block offset.
const MASK: u64 = (1 << SHIFT) - 1;
/// Largest representable compressed offset (48 bits).
pub const MAX_COMPRESSED: u64 = (1 << 48) - 1;
/// Largest representable uncompressed offset (16 bits).
pub const MAX_UNCOMPRESSED: u16 = u16::MAX;

/// A packed `(compressed_offset, uncompressed_offset)` pair.
///
/// Ordering is the natural `u64` ordering, which is also position order in the
/// file — that is what makes these usable as index keys.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtualOffset(u64);

impl VirtualOffset {
    /// Packs a compressed/uncompressed offset pair.
    ///
    /// Returns `None` if `compressed` does not fit in 48 bits. Note the
    /// uncompressed offset cannot overflow: a block holds at most 65536 bytes
    /// and offset 65536 is by definition the start of the *next* block.
    #[must_use]
    pub const fn new(compressed: u64, uncompressed: u16) -> Option<Self> {
        if compressed > MAX_COMPRESSED {
            return None;
        }
        Some(Self((compressed << SHIFT) | (uncompressed as u64)))
    }

    /// Packs without checking that `compressed` fits in 48 bits.
    ///
    /// Wraps silently on overflow; prefer [`VirtualOffset::new`].
    #[must_use]
    pub const fn new_unchecked(compressed: u64, uncompressed: u16) -> Self {
        Self((compressed << SHIFT) | (uncompressed as u64))
    }

    /// Byte offset of the containing block in the compressed stream.
    #[must_use]
    pub const fn compressed(self) -> u64 {
        self.0 >> SHIFT
    }

    /// Byte offset within the block's decompressed payload.
    #[must_use]
    pub const fn uncompressed(self) -> u16 {
        (self.0 & MASK) as u16
    }

    /// The raw packed representation, as stored in BAI/CSI.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for VirtualOffset {
    fn from(raw: u64) -> Self {
        Self(raw)
    }
}

impl From<VirtualOffset> for u64 {
    fn from(offset: VirtualOffset) -> Self {
        offset.0
    }
}

impl fmt::Debug for VirtualOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtualOffset")
            .field("compressed", &self.compressed())
            .field("uncompressed", &self.uncompressed())
            .finish()
    }
}

impl fmt::Display for VirtualOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.compressed(), self.uncompressed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_components() {
        let offset = VirtualOffset::new(0x0001_0203_0405, 0x0607).unwrap();
        assert_eq!(offset.compressed(), 0x0001_0203_0405);
        assert_eq!(offset.uncompressed(), 0x0607);
    }

    #[test]
    fn packs_into_documented_bit_layout() {
        // Upper 48 bits compressed, lower 16 uncompressed — not the reverse.
        let offset = VirtualOffset::new(1, 0).unwrap();
        assert_eq!(offset.as_u64(), 1 << 16);

        let offset = VirtualOffset::new(0, 1).unwrap();
        assert_eq!(offset.as_u64(), 1);
    }

    #[test]
    fn zero_is_start_of_file() {
        let offset = VirtualOffset::default();
        assert_eq!(offset.compressed(), 0);
        assert_eq!(offset.uncompressed(), 0);
    }

    #[test]
    fn rejects_compressed_offset_wider_than_48_bits() {
        assert!(VirtualOffset::new(MAX_COMPRESSED, 0).is_some());
        assert!(VirtualOffset::new(MAX_COMPRESSED + 1, 0).is_none());
    }

    #[test]
    fn accepts_maximum_within_block_offset() {
        let offset = VirtualOffset::new(0, MAX_UNCOMPRESSED).unwrap();
        assert_eq!(offset.uncompressed(), MAX_UNCOMPRESSED);
    }

    #[test]
    fn orders_by_file_position() {
        // Ordering must be block-major: a later block always sorts after an
        // earlier one regardless of within-block offset.
        let first = VirtualOffset::new(10, 65_535).unwrap();
        let second = VirtualOffset::new(11, 0).unwrap();
        assert!(first < second);
    }

    #[test]
    fn survives_raw_u64_round_trip() {
        let offset = VirtualOffset::new(123_456, 789).unwrap();
        assert_eq!(VirtualOffset::from(offset.as_u64()), offset);
    }
}
