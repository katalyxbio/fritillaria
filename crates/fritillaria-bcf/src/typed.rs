//! BCF2 typed values.
//!
//! Everything variable in a BCF record — ID, alleles, FILTER, every INFO value,
//! every FORMAT field — is a *typed value*: a descriptor byte followed by that
//! many elements of that type. There is no way to skip a field without decoding
//! its descriptor, so this module is on the path of every record walk, not just
//! of callers who want values.
//!
//! ```text
//! descriptor  1        (count << 4) | type
//! [length]    typed    present only when count == 15: the true count, itself
//!                      a typed atomic integer
//! elements    n*size
//! ```
//!
//! # The three things that bite
//!
//! **The count escape.** A count of 15 does not mean fifteen; it means "15 or
//! more, and the real count is the next typed integer in the stream". A reader
//! that takes the nibble at face value reads 15 elements of a 200-element field
//! and then resynchronises onto garbage — plausibly, and without erroring.
//! Both fixtures hit this: `giab_hg002.bcf` escapes 1407 times.
//!
//! **MISSING and END_OF_VECTOR are in-band values, per width.** `0x80` is a
//! missing `int8`, `0x81` is end-of-vector, and the equivalents at 16 and 32
//! bits are *not* sign-extensions of those — `0x8000` is missing at 16 bits and
//! is a perfectly ordinary `-32768` if you widened an `int8` first. So values
//! must be classified at their own width, before any widening. [`Int`] exists
//! to make that impossible to get wrong.
//!
//! **END_OF_VECTOR is padding, not data.** A FORMAT field stores one
//! fixed-length slot per sample, so a sample with fewer values than the widest
//! one is padded — a haploid genotype in a diploid file, which is every male
//! chrX call. Counting those pad values as alleles turns a haploid call into a
//! diploid one with a bogus second allele. `testdata/kg_phase3.bcf` has 214
//! records that do this.
//!
//! # Missing floats are signalling NaNs
//!
//! Float MISSING is `0x7F80_0001` and END_OF_VECTOR is `0x7F80_0002`. Both are
//! NaN, so `==` cannot distinguish them from a real NaN payload and neither can
//! `is_nan()`. They must be compared as **bit patterns**, which is why
//! [`Float::classify`] goes through `to_bits`.

use fritillaria_core::{Error, Result};

/// The low nibble of a descriptor: which atomic type the elements are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Type 0: a MISSING value carrying no explicit type.
    Missing,
    /// Type 1: 8-bit integer.
    Int8,
    /// Type 2: 16-bit integer.
    Int16,
    /// Type 3: 32-bit integer.
    Int32,
    /// Type 5: 32-bit IEEE-754 float.
    Float,
    /// Type 7: ASCII character. A string is a vector of these.
    Character,
}

impl Kind {
    /// Decodes the low nibble of a descriptor byte.
    ///
    /// Types 4, 6 and 8-15 are reserved; rejecting them is what keeps a
    /// speculative walk from wandering through arbitrary bytes.
    pub fn from_code(code: u8) -> Result<Self> {
        Ok(match code & 0x0F {
            0 => Self::Missing,
            1 => Self::Int8,
            2 => Self::Int16,
            3 => Self::Int32,
            5 => Self::Float,
            7 => Self::Character,
            other => {
                return Err(Error::Malformed {
                    format: "bcf",
                    position: 0,
                    reason: format!("reserved BCF type code {other}"),
                });
            }
        })
    }

    /// Bytes per element.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Missing => 0,
            Self::Int8 | Self::Character => 1,
            Self::Int16 => 2,
            Self::Int32 | Self::Float => 4,
        }
    }

    /// Whether elements are integers, of any width.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32)
    }
}

/// One integer element, classified at its own width.
///
/// The reason this is not just `i32`: the missing and end-of-vector sentinels
/// are width-specific bit patterns, so widening first destroys the information
/// needed to recognise them. Returning a classified value makes the mistake
/// unrepresentable rather than merely documented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int {
    /// A real value, widened to `i32`.
    Value(i32),
    /// The MISSING sentinel for this width.
    Missing,
    /// The END_OF_VECTOR sentinel: padding, and everything after it in this
    /// element's vector is padding too.
    EndOfVector,
    /// One of the six other reserved values for this width.
    Reserved(i32),
}

impl Int {
    /// The value, or `None` for missing, padding, or reserved.
    #[must_use]
    pub const fn value(self) -> Option<i32> {
        match self {
            Self::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Classifies a raw element of the given integer width.
    ///
    /// The reserved range is the low eight values of the sentinel space —
    /// `0x80..=0x87` at 8 bits, and the equivalents at 16 and 32.
    fn classify(raw: i32, kind: Kind) -> Self {
        let base = match kind {
            Kind::Int8 => i32::from(i8::MIN),
            Kind::Int16 => i32::from(i16::MIN),
            Kind::Int32 => i32::MIN,
            _ => return Self::Value(raw),
        };
        match raw.wrapping_sub(base) {
            0 => Self::Missing,
            1 => Self::EndOfVector,
            2..=7 => Self::Reserved(raw),
            _ => Self::Value(raw),
        }
    }
}

/// One float element, classified by bit pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Float {
    /// A real value — possibly a genuine NaN or infinity, which BCF permits.
    Value(f32),
    /// The MISSING sentinel, `0x7F80_0001`.
    Missing,
    /// The END_OF_VECTOR sentinel, `0x7F80_0002`.
    EndOfVector,
    /// One of the five other reserved patterns, `0x7F80_0003..=0x7F80_0007`.
    Reserved(u32),
}

impl Float {
    /// The value, or `None` for missing, padding, or reserved.
    #[must_use]
    pub const fn value(self) -> Option<f32> {
        match self {
            Self::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Classifies a raw 32-bit pattern.
    ///
    /// By bits, never by `==` or `is_nan()`: the sentinels are NaNs, so float
    /// comparison cannot tell them from a real NaN and reports every NaN equal
    /// to none of them.
    #[must_use]
    pub const fn classify(bits: u32) -> Self {
        match bits {
            0x7F80_0001 => Self::Missing,
            0x7F80_0002 => Self::EndOfVector,
            0x7F80_0003..=0x7F80_0007 => Self::Reserved(bits),
            _ => Self::Value(f32::from_bits(bits)),
        }
    }
}

/// A decoded typed value: what it is, how many, and where its elements are.
///
/// Elements are left in the buffer rather than copied out. That is the same
/// choice `fritillaria-bam::aux` makes and for the same reason — a BCF FORMAT
/// field over 2504 samples is the largest thing in the record, and materialising
/// it to look at one sample would cost more than the whole walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Typed<'a> {
    kind: Kind,
    count: usize,
    data: &'a [u8],
    /// Total encoded size, descriptor and length escape included, so a caller
    /// walking a sequence of values knows where the next one starts.
    encoded_len: usize,
}

impl<'a> Typed<'a> {
    /// Builds a value from an already-decoded descriptor and its elements.
    ///
    /// For FORMAT fields, where one descriptor covers every sample and each
    /// sample's slot is sliced out of the shared payload. `encoded_len` is the
    /// slot's length, since a slot carries no descriptor of its own.
    pub(crate) const fn from_parts(kind: Kind, count: usize, data: &'a [u8]) -> Self {
        Self {
            kind,
            count,
            data,
            encoded_len: data.len(),
        }
    }

    /// Element type.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// Number of elements. Zero means the whole value is missing.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Whether the value is absent altogether — a count of zero, however typed.
    ///
    /// Distinct from a vector *containing* missing elements, which has a
    /// non-zero count; the spec keeps the two apart and so does this.
    #[must_use]
    pub const fn is_missing(&self) -> bool {
        self.count == 0 || matches!(self.kind, Kind::Missing)
    }

    /// The raw element bytes, descriptor excluded.
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Bytes this value occupies in the stream, descriptor included.
    #[must_use]
    pub const fn encoded_len(&self) -> usize {
        self.encoded_len
    }

    /// The value as a string, for `Character` values.
    ///
    /// Returns `None` for any other type. Not NUL-terminated and not validated
    /// as UTF-8 — VCF is ASCII, and callers that need `str` should say so.
    #[must_use]
    pub const fn as_str(&self) -> Option<&'a [u8]> {
        match self.kind {
            Kind::Character => Some(self.data),
            _ => None,
        }
    }

    /// Integer elements, classified at their own width.
    ///
    /// Returns `None` for non-integer values. Empty for a missing value.
    #[must_use]
    pub const fn ints(&self) -> Option<Ints<'a>> {
        if self.kind.is_integer() {
            Some(Ints {
                kind: self.kind,
                data: self.data,
                pos: 0,
            })
        } else {
            None
        }
    }

    /// Float elements, classified by bit pattern.
    #[must_use]
    pub const fn floats(&self) -> Option<Floats<'a>> {
        match self.kind {
            Kind::Float => Some(Floats {
                data: self.data,
                pos: 0,
            }),
            _ => None,
        }
    }

    /// The single integer value of an atomic integer, or `None`.
    ///
    /// This is the shape dictionary keys take — INFO keys, FORMAT keys, FILTER
    /// entries — so it is the most-called accessor in a record walk.
    #[must_use]
    pub fn as_int(&self) -> Option<i32> {
        if self.count != 1 {
            return None;
        }
        self.ints()?.next()?.value()
    }
}

/// Iterator over the integer elements of a [`Typed`].
///
/// Deliberately not `Copy`: an iterator that copies on use advances the copy
/// and leaves the original where it was, which reads as an infinite loop.
#[derive(Clone, Debug)]
pub struct Ints<'a> {
    kind: Kind,
    data: &'a [u8],
    pos: usize,
}

impl Iterator for Ints<'_> {
    type Item = Int;

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.kind.size();
        let raw = self.data.get(self.pos..self.pos + size)?;
        self.pos += size;
        let widened = match self.kind {
            Kind::Int8 => i32::from(raw[0].cast_signed()),
            Kind::Int16 => i32::from(i16::from_le_bytes(raw.try_into().ok()?)),
            _ => i32::from_le_bytes(raw.try_into().ok()?),
        };
        Some(Int::classify(widened, self.kind))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.data.len() - self.pos) / self.kind.size();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Ints<'_> {}

/// Iterator over the float elements of a [`Typed`].
///
/// Not `Copy`, for the reason given on [`Ints`].
#[derive(Clone, Debug)]
pub struct Floats<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for Floats<'_> {
    type Item = Float;

    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.data.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(Float::classify(u32::from_le_bytes(raw.try_into().ok()?)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.data.len() - self.pos) / 4;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Floats<'_> {}

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "bcf",
        position: position as u64,
        reason: reason.into(),
    }
}

/// Reads the descriptor at `pos` and returns `(count, kind, header_len)`.
///
/// `header_len` covers the descriptor byte plus the length escape when present.
pub fn read_descriptor(buf: &[u8], pos: usize) -> Result<(usize, Kind, usize)> {
    let descriptor = *buf
        .get(pos)
        .ok_or_else(|| malformed(pos, "truncated type descriptor"))?;
    let kind = Kind::from_code(descriptor)?;
    let nibble = usize::from(descriptor >> 4);

    if nibble < 15 {
        return Ok((nibble, kind, 1));
    }

    // The escape: the true count follows as a typed atomic integer. It must
    // itself be atomic and integral, so a nested escape here is malformed
    // rather than a recursion to follow.
    let inner = read(buf, pos + 1)?;
    if inner.count() != 1 || !inner.kind().is_integer() {
        return Err(malformed(
            pos + 1,
            "escaped vector length is not an atomic integer",
        ));
    }
    let count = inner
        .as_int()
        .ok_or_else(|| malformed(pos + 1, "escaped vector length is a sentinel value"))?;
    let count = usize::try_from(count)
        .map_err(|_| malformed(pos + 1, format!("negative escaped vector length {count}")))?;
    // 15 is the smallest count the escape is allowed to encode. A writer
    // emitting a shorter one would still be readable, but it signals a
    // misparse far more often than a quirky writer, and on a speculative walk
    // that distinction is the whole point.
    if count < 15 {
        return Err(malformed(
            pos + 1,
            format!("escaped vector length {count} is below the 15-element threshold"),
        ));
    }
    Ok((count, kind, 1 + inner.encoded_len()))
}

/// Reads one typed value at `pos`.
pub fn read(buf: &[u8], pos: usize) -> Result<Typed<'_>> {
    let (count, kind, header_len) = read_descriptor(buf, pos)?;
    let bytes = count
        .checked_mul(kind.size())
        .ok_or_else(|| malformed(pos, "value size overflows"))?;
    let start = pos + header_len;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| malformed(pos, "value extends past usize"))?;
    let data = buf
        .get(start..end)
        .ok_or_else(|| malformed(pos, "value runs past the end of the buffer"))?;

    Ok(Typed {
        kind,
        count,
        data,
        encoded_len: header_len + bytes,
    })
}

/// Skips one typed value, returning the offset just past it.
///
/// The record walk uses this far more than it uses values: FORMAT fields over
/// thousands of samples are stepped over wholesale when the caller wants a
/// different field.
pub fn skip(buf: &[u8], pos: usize) -> Result<usize> {
    Ok(pos + read(buf, pos)?.encoded_len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_specs_acac_example() {
        // 0x47 = 4 characters, then ACAC.
        let buf = [0x47, b'A', b'C', b'A', b'C'];
        let v = read(&buf, 0).unwrap();
        assert_eq!(v.kind(), Kind::Character);
        assert_eq!(v.count(), 4);
        assert_eq!(v.as_str(), Some(&b"ACAC"[..]));
        assert_eq!(v.encoded_len(), 5);
    }

    #[test]
    fn reads_the_specs_escaped_string_example() {
        // 0xF7 = characters with escaped length, 0x11 0x1B = int8 27.
        let mut buf = vec![0xF7, 0x11, 0x1B];
        buf.extend_from_slice(b"VariantCallFormatSampleText");
        let v = read(&buf, 0).unwrap();
        assert_eq!(v.count(), 27);
        assert_eq!(v.as_str(), Some(&b"VariantCallFormatSampleText"[..]));
        assert_eq!(v.encoded_len(), 3 + 27);
        assert_eq!(skip(&buf, 0).unwrap(), buf.len());
    }

    #[test]
    fn reads_the_specs_escaped_vector_example() {
        // 0xF1 0x11 0x10 then 1..=16 as int8.
        let mut buf = vec![0xF1, 0x11, 0x10];
        buf.extend(1u8..=16);
        let v = read(&buf, 0).unwrap();
        assert_eq!(v.count(), 16);
        let values: Vec<_> = v.ints().unwrap().map(|i| i.value().unwrap()).collect();
        assert_eq!(values, (1..=16).collect::<Vec<i32>>());
    }

    #[test]
    fn the_count_nibble_of_fifteen_never_means_fifteen() {
        // The bug this guards: taking the nibble at face value reads 15
        // elements of a 16-element field and resynchronises onto garbage.
        let mut buf = vec![0xF1, 0x11, 0x10];
        buf.extend(1u8..=16);
        assert_eq!(read(&buf, 0).unwrap().count(), 16);
        // ...and a genuine 14 is inline, with no escape byte.
        let mut inline = vec![0xE1];
        inline.extend(1u8..=14);
        let v = read(&inline, 0).unwrap();
        assert_eq!(v.count(), 14);
        assert_eq!(v.encoded_len(), 15);
    }

    #[test]
    fn a_zero_count_is_a_missing_value() {
        for descriptor in [0x00u8, 0x01, 0x05, 0x07] {
            let buf = [descriptor];
            let v = read(&buf, 0).unwrap();
            assert!(v.is_missing());
            assert_eq!(v.count(), 0);
            assert_eq!(v.encoded_len(), 1);
        }
    }

    #[test]
    fn integer_sentinels_are_recognised_at_their_own_width() {
        // 0x80 as int8 is MISSING...
        let v = read(&[0x11, 0x80], 0).unwrap();
        assert_eq!(v.ints().unwrap().next(), Some(Int::Missing));
        assert_eq!(v.as_int(), None);

        // ...but -128 widened is an ordinary int16 value, not a sentinel. This
        // is the mistake that classifying after widening would make.
        let v = read(&[0x12, 0x80, 0xFF], 0).unwrap();
        assert_eq!(v.ints().unwrap().next(), Some(Int::Value(-128)));

        // The int16 sentinel is 0x8000.
        let v = read(&[0x12, 0x00, 0x80], 0).unwrap();
        assert_eq!(v.ints().unwrap().next(), Some(Int::Missing));

        // And the int32 one is 0x80000000.
        let v = read(&[0x13, 0x00, 0x00, 0x00, 0x80], 0).unwrap();
        assert_eq!(v.ints().unwrap().next(), Some(Int::Missing));
    }

    #[test]
    fn end_of_vector_is_distinct_from_missing_at_every_width() {
        for (descriptor, bytes) in [
            (0x11u8, vec![0x81u8]),
            (0x12, vec![0x01, 0x80]),
            (0x13, vec![0x01, 0x00, 0x00, 0x80]),
        ] {
            let mut buf = vec![descriptor];
            buf.extend(bytes);
            let v = read(&buf, 0).unwrap();
            assert_eq!(v.ints().unwrap().next(), Some(Int::EndOfVector));
        }
    }

    #[test]
    fn the_six_reserved_integer_values_are_neither() {
        for raw in 0x82u8..=0x87 {
            let buf = [0x11, raw];
            let v = read(&buf, 0).unwrap();
            assert!(matches!(v.ints().unwrap().next(), Some(Int::Reserved(_))));
        }
        // 0x88 is past the reserved range and is an ordinary -120.
        let v = read(&[0x11, 0x88], 0).unwrap();
        assert_eq!(v.ints().unwrap().next(), Some(Int::Value(-120)));
    }

    #[test]
    fn float_sentinels_are_compared_by_bits_not_by_value() {
        let missing = Float::classify(0x7F80_0001);
        let eov = Float::classify(0x7F80_0002);
        assert_eq!(missing, Float::Missing);
        assert_eq!(eov, Float::EndOfVector);

        // A real quiet NaN is a value. `is_nan()` cannot tell these three
        // apart, which is the whole reason classification is by bits.
        let nan = Float::classify(0x7FC0_0000);
        assert!(matches!(nan, Float::Value(v) if v.is_nan()));

        for bits in 0x7F80_0003u32..=0x7F80_0007 {
            assert!(matches!(Float::classify(bits), Float::Reserved(_)));
        }
        // Infinity, 0x7F800000, is a legal BCF value and sits just below the
        // sentinel range.
        assert_eq!(
            Float::classify(0x7F80_0000),
            Float::Value(f32::INFINITY),
            "+Inf is a value, not a sentinel"
        );
    }

    #[test]
    fn a_format_descriptor_covers_every_samples_slot() {
        // The spec's mixed-length example: sample A = [1, END_OF_VECTOR],
        // sample B = [2, 3]. One descriptor, then two slots of two elements.
        //
        // This is where FORMAT departs from every other typed value, and the
        // asymmetry is easy to get wrong in the safe-looking direction: `read`
        // yields exactly one slot, because a lone typed value *is* one. Scaling
        // by the sample count is the record walk's job — see
        // `record::FormatFields`.
        let buf = [0x21, 0x01, 0x81, 0x02, 0x03];

        let (count, kind, header_len) = read_descriptor(&buf, 0).unwrap();
        assert_eq!((count, kind, header_len), (2, Kind::Int8, 1));

        let stride = count * kind.size();
        let a = Typed::from_parts(kind, count, &buf[header_len..header_len + stride]);
        let b = Typed::from_parts(kind, count, &buf[header_len + stride..]);
        assert_eq!(
            a.ints().unwrap().collect::<Vec<_>>(),
            vec![Int::Value(1), Int::EndOfVector]
        );
        assert_eq!(
            b.ints().unwrap().collect::<Vec<_>>(),
            vec![Int::Value(2), Int::Value(3)]
        );

        // And read() stops after the first slot, which is the trap.
        let one = read(&buf, 0).unwrap();
        assert_eq!(one.encoded_len(), 3);
        assert_eq!(
            one.ints().unwrap().collect::<Vec<_>>(),
            vec![Int::Value(1), Int::EndOfVector]
        );
    }

    #[test]
    fn rejects_reserved_type_codes() {
        for code in [4u8, 6, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert!(
                read(&[0x10 | code, 0, 0, 0, 0], 0).is_err(),
                "type code {code} is reserved and must not decode"
            );
        }
    }

    #[test]
    fn rejects_a_value_running_past_the_buffer() {
        assert!(read(&[0x43, 0, 0, 0], 0).is_err(), "4 int32s need 16 bytes");
        assert!(read(&[0xF1], 0).is_err(), "escape with no length");
        assert!(read(&[0xF1, 0x11], 0).is_err(), "length with no value byte");
    }

    #[test]
    fn rejects_an_escape_that_encodes_a_short_count() {
        // 0xF7 with a length of 3 — readable, but far likelier a misparse than
        // a quirky writer, and on a speculative walk that is the difference.
        assert!(read(&[0xF7, 0x11, 0x03, b'a', b'b', b'c'], 0).is_err());
    }

    #[test]
    fn rejects_a_nested_escape_as_the_length() {
        assert!(read(&[0xF1, 0xF1, 0x11, 0x10], 0).is_err());
    }

    #[test]
    fn as_int_refuses_vectors_and_sentinels() {
        let v = read(&[0x21, 0x01, 0x02], 0).unwrap();
        assert_eq!(v.as_int(), None, "a 2-element vector is not atomic");
        let v = read(&[0x11, 0x80], 0).unwrap();
        assert_eq!(v.as_int(), None, "MISSING is not a value");
        let v = read(&[0x11, 0x07], 0).unwrap();
        assert_eq!(v.as_int(), Some(7));
    }

    #[test]
    fn iterators_report_an_exact_length() {
        let mut buf = vec![0x31];
        buf.extend([1u8, 2, 3]);
        let v = read(&buf, 0).unwrap();
        assert_eq!(v.ints().unwrap().len(), 3);

        let mut buf = vec![0x25];
        buf.extend(1.0f32.to_le_bytes());
        buf.extend(2.0f32.to_le_bytes());
        let v = read(&buf, 0).unwrap();
        assert_eq!(v.floats().unwrap().len(), 2);
        assert_eq!(
            v.floats()
                .unwrap()
                .map(|f| f.value().unwrap())
                .collect::<Vec<_>>(),
            vec![1.0, 2.0]
        );
    }
}
