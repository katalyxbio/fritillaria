//! Aux tag decoding.
//!
//! ```text
//! tag    2 bytes   two printable ASCII characters
//! type   1 byte    one of AcCsSiIfZHB
//! value  varies    see below
//! ```
//!
//! | type | value |
//! |---|---|
//! | `A` | one printable character |
//! | `c` `C` `s` `S` `i` `I` | little-endian integer, 1/1/2/2/4/4 bytes |
//! | `f` | little-endian IEEE-754 single |
//! | `Z` | NUL-terminated string |
//! | `H` | NUL-terminated string of hex digit *pairs* |
//! | `B` | subtype byte, `uint32` count, then that many elements |
//!
//! Two things here are the usual source of silent corruption, and both are why
//! this is a checked iterator rather than a set of offset arithmetic helpers:
//!
//! - **`B` cannot be skipped without being partly parsed.** Its payload length
//!   depends on a subtype and a count that are stored *inside* the value, so a
//!   parser that tries to stride over aux fields by type alone desynchronises
//!   the moment it meets one — and then reports later fields that do not exist.
//! - **Integer width is not recoverable from SAM.** htslib writes the narrowest
//!   type that fits, so the same logical tag is `C` in one file and `I` in
//!   another, while `samtools view` prints both as `i`. The width is preserved
//!   here ([`Value::Int8`] … [`Value::UInt32`]) so a value can be written back
//!   unchanged; use [`Value::as_int`] when only the number matters.
//!
//! Decoding is zero-copy and lazy. `Z`/`H` borrow their bytes, and a `B` array
//! is a view over the raw little-endian payload that decodes elements on
//! iteration — BAM does not align aux values, so the bytes cannot simply be
//! reinterpreted as `&[i32]` and are read with `from_le_bytes` instead.

use std::marker::PhantomData;

use fritillaria_core::{Error, Result};

/// A two-character aux tag, e.g. `NM`, `MM`, `RG`.
pub type Tag = [u8; 2];

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "bam",
        position: position as u64,
        reason: reason.into(),
    }
}

/// One element of a `B` array.
///
/// Sealed by [`Element::SIZE`] and [`Element::from_le_bytes`]; implemented for
/// exactly the seven types the spec allows as `B` subtypes.
pub trait Element: Copy + Sized {
    /// Encoded width in bytes.
    const SIZE: usize;
    /// The BAM subtype character.
    const SUBTYPE: u8;
    /// Decodes one element from exactly `SIZE` little-endian bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

macro_rules! impl_element {
    ($($ty:ty => $subtype:literal),* $(,)?) => {
        $(impl Element for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            const SUBTYPE: u8 = $subtype;
            fn from_le_bytes(bytes: &[u8]) -> Self {
                // The caller slices exactly SIZE bytes, so this cannot fail.
                <$ty>::from_le_bytes(bytes.try_into().expect("slice of SIZE bytes"))
            }
        })*
    };
}

impl_element! {
    i8 => b'c', u8 => b'C',
    i16 => b's', u16 => b'S',
    i32 => b'i', u32 => b'I',
    f32 => b'f',
}

/// A typed, zero-copy view over the elements of a `B` array.
///
/// Holds the raw little-endian payload; elements are decoded on access.
#[derive(Clone, Copy)]
pub struct Values<'a, T> {
    raw: &'a [u8],
    _marker: PhantomData<T>,
}

impl<'a, T: Element + 'a> Values<'a, T> {
    pub(crate) const fn new(raw: &'a [u8]) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw.len() / T::SIZE
    }

    /// Whether the array is empty. A zero-length `B` array is legal.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// The element at `index`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T> {
        let start = index.checked_mul(T::SIZE)?;
        self.raw
            .get(start..start.checked_add(T::SIZE)?)
            .map(T::from_le_bytes)
    }

    /// Iterates the decoded elements.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + 'a {
        self.raw.chunks_exact(T::SIZE).map(T::from_le_bytes)
    }

    /// The undecoded little-endian payload.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.raw
    }
}

impl<'a, T: Element> IntoIterator for Values<'a, T> {
    type Item = T;
    type IntoIter = std::iter::Map<std::slice::ChunksExact<'a, u8>, fn(&'a [u8]) -> T>;

    fn into_iter(self) -> Self::IntoIter {
        self.raw.chunks_exact(T::SIZE).map(T::from_le_bytes)
    }
}

impl<T: Element + std::fmt::Debug> std::fmt::Debug for Values<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A `B` array value, one variant per subtype.
#[derive(Clone, Copy, Debug)]
pub enum Array<'a> {
    /// `B:c`
    Int8(Values<'a, i8>),
    /// `B:C`
    UInt8(Values<'a, u8>),
    /// `B:s`
    Int16(Values<'a, i16>),
    /// `B:S`
    UInt16(Values<'a, u16>),
    /// `B:i`
    Int32(Values<'a, i32>),
    /// `B:I`
    UInt32(Values<'a, u32>),
    /// `B:f`
    Float(Values<'a, f32>),
}

impl Array<'_> {
    /// The BAM subtype character, one of `cCsSiIf`.
    #[must_use]
    pub const fn subtype(&self) -> u8 {
        match self {
            Self::Int8(_) => b'c',
            Self::UInt8(_) => b'C',
            Self::Int16(_) => b's',
            Self::UInt16(_) => b'S',
            Self::Int32(_) => b'i',
            Self::UInt32(_) => b'I',
            Self::Float(_) => b'f',
        }
    }

    /// Number of elements, whatever the subtype.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Int8(v) => v.len(),
            Self::UInt8(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::UInt16(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::Float(v) => v.len(),
        }
    }

    /// Whether the array is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A decoded aux value.
///
/// Integer width is preserved rather than widened, because it is part of the
/// encoding and cannot be recovered from SAM text. See the module docs.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    /// `A` — a single printable character.
    Character(u8),
    /// `c`
    Int8(i8),
    /// `C`
    UInt8(u8),
    /// `s`
    Int16(i16),
    /// `S`
    UInt16(u16),
    /// `i`
    Int32(i32),
    /// `I`
    UInt32(u32),
    /// `f`
    Float(f32),
    /// `Z` — a NUL-terminated string, NUL excluded.
    String(&'a [u8]),
    /// `H` — hex digit pairs, NUL excluded and *not* decoded to bytes.
    Hex(&'a [u8]),
    /// `B`
    Array(Array<'a>),
}

impl Value<'_> {
    /// The BAM type character.
    #[must_use]
    pub const fn ty(&self) -> u8 {
        match self {
            Self::Character(_) => b'A',
            Self::Int8(_) => b'c',
            Self::UInt8(_) => b'C',
            Self::Int16(_) => b's',
            Self::UInt16(_) => b'S',
            Self::Int32(_) => b'i',
            Self::UInt32(_) => b'I',
            Self::Float(_) => b'f',
            Self::String(_) => b'Z',
            Self::Hex(_) => b'H',
            Self::Array(_) => b'B',
        }
    }

    /// Any integer variant widened to `i64`; `None` for everything else.
    ///
    /// `i64` is used because it is the only type covering all of `c`…`I`
    /// without loss — `u32` values above `i32::MAX` are real and occur in this
    /// project's own PacBio fixture.
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int8(n) => Some(*n as i64),
            Self::UInt8(n) => Some(*n as i64),
            Self::Int16(n) => Some(*n as i64),
            Self::UInt16(n) => Some(*n as i64),
            Self::Int32(n) => Some(*n as i64),
            Self::UInt32(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// The value as a float; `None` unless it is an `f`.
    #[must_use]
    pub const fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(n) => Some(*n),
            _ => None,
        }
    }

    /// The bytes of a `Z` or `H` value.
    #[must_use]
    pub const fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::String(s) | Self::Hex(s) => Some(s),
            _ => None,
        }
    }

    /// The array, if this is a `B`.
    #[must_use]
    pub const fn as_array(&self) -> Option<Array<'_>> {
        match self {
            Self::Array(a) => Some(*a),
            _ => None,
        }
    }
}

/// Reads one value of type `ty` starting at `pos`, returning it and the offset
/// just past it.
fn read_value(buf: &[u8], pos: usize, ty: u8) -> Result<(Value<'_>, usize)> {
    /// Fixed-width scalars: slice `n` bytes or fail with a consistent message.
    fn fixed(buf: &[u8], pos: usize, n: usize, ty: u8) -> Result<&[u8]> {
        buf.get(pos..pos + n).ok_or_else(|| {
            malformed(
                pos,
                format!("type '{}' needs {n} bytes, {} remain", ty as char, {
                    buf.len().saturating_sub(pos)
                }),
            )
        })
    }

    /// `Z` and `H` are NUL-terminated; an unterminated one is malformed rather
    /// than "runs to the end of the record", which would silently absorb every
    /// following tag.
    fn nul_terminated(buf: &[u8], pos: usize, ty: u8) -> Result<(&[u8], usize)> {
        let rest = buf.get(pos..).unwrap_or_default();
        let end = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| malformed(pos, format!("unterminated '{}' string", ty as char)))?;
        Ok((&rest[..end], pos + end + 1))
    }

    match ty {
        b'A' => {
            let b = fixed(buf, pos, 1, ty)?[0];
            Ok((Value::Character(b), pos + 1))
        }
        b'c' => Ok((
            Value::Int8(fixed(buf, pos, 1, ty)?[0].cast_signed()),
            pos + 1,
        )),
        b'C' => Ok((Value::UInt8(fixed(buf, pos, 1, ty)?[0]), pos + 1)),
        b's' => Ok((
            Value::Int16(i16::from_le_bytes(
                fixed(buf, pos, 2, ty)?.try_into().expect("2 bytes"),
            )),
            pos + 2,
        )),
        b'S' => Ok((
            Value::UInt16(u16::from_le_bytes(
                fixed(buf, pos, 2, ty)?.try_into().expect("2 bytes"),
            )),
            pos + 2,
        )),
        b'i' => Ok((
            Value::Int32(i32::from_le_bytes(
                fixed(buf, pos, 4, ty)?.try_into().expect("4 bytes"),
            )),
            pos + 4,
        )),
        b'I' => Ok((
            Value::UInt32(u32::from_le_bytes(
                fixed(buf, pos, 4, ty)?.try_into().expect("4 bytes"),
            )),
            pos + 4,
        )),
        b'f' => Ok((
            Value::Float(f32::from_le_bytes(
                fixed(buf, pos, 4, ty)?.try_into().expect("4 bytes"),
            )),
            pos + 4,
        )),
        b'Z' => {
            let (s, next) = nul_terminated(buf, pos, ty)?;
            Ok((Value::String(s), next))
        }
        b'H' => {
            let (s, next) = nul_terminated(buf, pos, ty)?;
            if s.len() % 2 != 0 || !s.iter().all(u8::is_ascii_hexdigit) {
                return Err(malformed(
                    pos,
                    "'H' value is not an even-length run of hex digits",
                ));
            }
            Ok((Value::Hex(s), next))
        }
        b'B' => read_array(buf, pos),
        other => Err(malformed(
            pos,
            format!("unknown aux type byte {other:#04x} ('{}')", other as char),
        )),
    }
}

/// Reads a `B` array: subtype byte, `uint32` count, then `count` elements.
fn read_array(buf: &[u8], pos: usize) -> Result<(Value<'_>, usize)> {
    let subtype = *buf
        .get(pos)
        .ok_or_else(|| malformed(pos, "'B' value truncated before its subtype"))?;
    let count = buf
        .get(pos + 1..pos + 5)
        .ok_or_else(|| malformed(pos, "'B' value truncated before its count"))?;
    let count = u32::from_le_bytes(count.try_into().expect("4 bytes")) as usize;

    let width = match subtype {
        b'c' | b'C' => 1,
        b's' | b'S' => 2,
        b'i' | b'I' | b'f' => 4,
        other => {
            return Err(malformed(
                pos,
                format!("unknown 'B' subtype {other:#04x} ('{}')", other as char),
            ));
        }
    };

    let start = pos + 5;
    // Computed in usize with a checked multiply: `count` is attacker-controlled
    // and a wrapping product would slice a short payload and report it as full.
    let len = count
        .checked_mul(width)
        .ok_or_else(|| malformed(pos, format!("'B' array of {count} elements overflows")))?;
    let raw = buf.get(start..start + len).ok_or_else(|| {
        malformed(
            pos,
            format!(
                "'B:{}' array declares {count} elements ({len} bytes) but only {} remain",
                subtype as char,
                buf.len().saturating_sub(start)
            ),
        )
    })?;

    let array = match subtype {
        b'c' => Array::Int8(Values::new(raw)),
        b'C' => Array::UInt8(Values::new(raw)),
        b's' => Array::Int16(Values::new(raw)),
        b'S' => Array::UInt16(Values::new(raw)),
        b'i' => Array::Int32(Values::new(raw)),
        b'I' => Array::UInt32(Values::new(raw)),
        b'f' => Array::Float(Values::new(raw)),
        _ => unreachable!("subtype validated above"),
    };
    Ok((Value::Array(array), start + len))
}

/// Iterator over the aux fields of one record.
///
/// Yields `Result` because a malformed field desynchronises everything after
/// it: once a length is wrong, the next "tag" is really payload bytes. The
/// iterator fuses after an error rather than resynchronising on garbage.
#[derive(Clone, Debug)]
pub struct Fields<'a> {
    buf: &'a [u8],
    pos: usize,
    done: bool,
}

impl<'a> Fields<'a> {
    /// Iterates the fields of a raw aux block, as returned by
    /// [`Record::aux_raw`](crate::columnar::Record::aux_raw).
    #[must_use]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            done: false,
        }
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(Tag, Value<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.buf.len() {
            return None;
        }

        let Some(head) = self.buf.get(self.pos..self.pos + 3) else {
            self.done = true;
            return Some(Err(malformed(
                self.pos,
                format!(
                    "aux field needs a 2-byte tag and a type byte, {} remain",
                    self.buf.len() - self.pos
                ),
            )));
        };

        let tag: Tag = [head[0], head[1]];
        let ty = head[2];
        match read_value(self.buf, self.pos + 3, ty) {
            Ok((value, next)) => {
                self.pos = next;
                Some(Ok((tag, value)))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one aux field.
    fn field(tag: Tag, ty: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = tag.to_vec();
        v.push(ty);
        v.extend_from_slice(payload);
        v
    }

    fn array_field(tag: Tag, subtype: u8, count: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = tag.to_vec();
        v.push(b'B');
        v.push(subtype);
        v.extend_from_slice(&count.to_le_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn decode(buf: &[u8]) -> Result<Vec<(Tag, Value<'_>)>> {
        Fields::new(buf).collect()
    }

    #[test]
    fn decodes_every_scalar_type() {
        // Deliberately covers the widths the PacBio fixture does not contain:
        // htslib writes the narrowest type that fits, so 'c', 's' and 'i' never
        // appear there, and 'A'/'H' appear in no real fixture at all.
        let mut buf = Vec::new();
        buf.extend(field(*b"XA", b'A', b"c"));
        buf.extend(field(*b"Xc", b'c', &(-5i8).to_le_bytes()));
        buf.extend(field(*b"XC", b'C', &250u8.to_le_bytes()));
        buf.extend(field(*b"Xs", b's', &(-300i16).to_le_bytes()));
        buf.extend(field(*b"XS", b'S', &60000u16.to_le_bytes()));
        buf.extend(field(*b"Xi", b'i', &(-70000i32).to_le_bytes()));
        buf.extend(field(*b"XI", b'I', &4_000_000_000u32.to_le_bytes()));
        buf.extend(field(*b"Xf", b'f', &1.5f32.to_le_bytes()));
        buf.extend(field(*b"XZ", b'Z', b"hello\0"));
        buf.extend(field(*b"XH", b'H', b"1AE3\0"));

        let fields = decode(&buf).unwrap();
        let types: Vec<u8> = fields.iter().map(|(_, v)| v.ty()).collect();
        assert_eq!(types, b"AcCsSiIfZH");

        assert!(matches!(fields[0].1, Value::Character(b'c')));
        assert_eq!(fields[1].1.as_int(), Some(-5));
        assert_eq!(fields[2].1.as_int(), Some(250));
        assert_eq!(fields[3].1.as_int(), Some(-300));
        assert_eq!(fields[4].1.as_int(), Some(60000));
        assert_eq!(fields[5].1.as_int(), Some(-70000));
        assert_eq!(fields[6].1.as_int(), Some(4_000_000_000));
        assert_eq!(fields[7].1.as_float(), Some(1.5));
        assert_eq!(fields[8].1.as_bytes(), Some(&b"hello"[..]));
        assert_eq!(fields[9].1.as_bytes(), Some(&b"1AE3"[..]));
    }

    #[test]
    fn a_u32_above_i32_max_survives_as_int() {
        // The reason as_int returns i64: widening through i32 would wrap this
        // to a negative number, and 'I' values this large are in real data.
        let buf = field(*b"XI", b'I', &4_000_000_000u32.to_le_bytes());
        let fields = decode(&buf).unwrap();
        assert_eq!(fields[0].1.as_int(), Some(4_000_000_000));
    }

    #[test]
    fn decodes_every_array_subtype() {
        let cases: Vec<(u8, u32, Vec<u8>, usize)> = vec![
            (b'c', 3, vec![0xfb, 0x01, 0x7f], 3),
            (b'C', 3, vec![0xfb, 0x01, 0x7f], 3),
            (b's', 2, (-300i16).to_le_bytes().repeat(2), 2),
            (b'S', 2, 60000u16.to_le_bytes().repeat(2), 2),
            (b'i', 2, (-70000i32).to_le_bytes().repeat(2), 2),
            (b'I', 2, 4_000_000_000u32.to_le_bytes().repeat(2), 2),
            (b'f', 2, 1.5f32.to_le_bytes().repeat(2), 2),
        ];

        for (subtype, count, payload, expected_len) in cases {
            let buf = array_field(*b"XB", subtype, count, &payload);
            let fields = decode(&buf).unwrap();
            let array = fields[0].1.as_array().expect("a B value");
            assert_eq!(array.subtype(), subtype, "subtype {}", subtype as char);
            assert_eq!(array.len(), expected_len, "len for {}", subtype as char);
        }
    }

    #[test]
    fn array_elements_decode_with_the_right_sign_and_width() {
        let buf = array_field(*b"Xc", b'c', 3, &[0xfb, 0x01, 0x7f]);
        let fields = decode(&buf).unwrap();
        let Some(Array::Int8(v)) = fields[0].1.as_array() else {
            panic!("expected B:c");
        };
        assert_eq!(v.iter().collect::<Vec<_>>(), vec![-5, 1, 127]);

        // Same bytes, unsigned subtype: 0xfb must read as 251, not -5.
        let buf = array_field(*b"XC", b'C', 3, &[0xfb, 0x01, 0x7f]);
        let fields = decode(&buf).unwrap();
        let Some(Array::UInt8(v)) = fields[0].1.as_array() else {
            panic!("expected B:C");
        };
        assert_eq!(v.iter().collect::<Vec<_>>(), vec![251, 1, 127]);
    }

    #[test]
    fn an_empty_array_is_legal() {
        let buf = array_field(*b"XB", b'i', 0, &[]);
        let fields = decode(&buf).unwrap();
        let array = fields[0].1.as_array().unwrap();
        assert!(array.is_empty());
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn an_empty_string_is_legal() {
        let buf = field(*b"XZ", b'Z', b"\0");
        let fields = decode(&buf).unwrap();
        assert_eq!(fields[0].1.as_bytes(), Some(&b""[..]));
    }

    #[test]
    fn an_empty_aux_block_yields_nothing() {
        assert!(decode(&[]).unwrap().is_empty());
    }

    #[test]
    fn a_field_after_an_array_is_found() {
        // The desynchronisation case: a parser that strides over 'B' without
        // reading its subtype and count lands mid-payload and invents a tag.
        let mut buf = array_field(*b"XB", b'i', 2, &[1, 0, 0, 0, 2, 0, 0, 0]);
        buf.extend(field(*b"NM", b'C', &[7]));

        let fields = decode(&buf).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[1].0, *b"NM");
        assert_eq!(fields[1].1.as_int(), Some(7));
    }

    #[test]
    fn a_truncated_scalar_is_an_error() {
        let buf = field(*b"XI", b'I', &[1, 2]); // 'I' needs 4 bytes
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_unterminated_string_is_an_error() {
        // Must not silently run to the end of the block and swallow later tags.
        let buf = field(*b"XZ", b'Z', b"no terminator");
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_array_longer_than_its_payload_is_an_error() {
        let buf = array_field(*b"XB", b'i', 1000, &[1, 0, 0, 0]);
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_array_count_that_overflows_is_an_error() {
        // count * width must not wrap into a small, satisfiable length.
        let buf = array_field(*b"XB", b'i', u32::MAX, &[1, 0, 0, 0]);
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_unknown_type_byte_is_an_error() {
        let buf = field(*b"XX", b'Q', &[0]);
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_unknown_array_subtype_is_an_error() {
        let buf = array_field(*b"XB", b'q', 1, &[0]);
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn an_odd_length_hex_value_is_an_error() {
        let buf = field(*b"XH", b'H', b"1AE\0");
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn a_trailing_partial_tag_is_an_error() {
        let mut buf = field(*b"NM", b'C', &[7]);
        buf.extend_from_slice(b"XZ"); // tag with no type byte
        assert!(matches!(
            decode(&buf),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn the_iterator_stops_after_an_error() {
        // Otherwise a caller using `filter_map(Result::ok)` would resynchronise
        // on payload bytes and report tags that were never written.
        let mut buf = field(*b"XZ", b'Z', b"unterminated");
        buf.extend(field(*b"NM", b'C', &[7]));
        let mut fields = Fields::new(&buf);
        assert!(fields.next().unwrap().is_err());
        assert!(fields.next().is_none(), "iterator must fuse after an error");
    }
}
