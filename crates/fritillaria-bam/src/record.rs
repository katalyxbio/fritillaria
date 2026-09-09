//! Stage 4: record boundary discovery, and a zero-copy record view.
//!
//! ```text
//! offset  size          field
//! 0       4             block_size   (length of everything after this field)
//! 4       4             refID
//! 8       4             pos          (0-based; -1 = unplaced)
//! 12      1             l_read_name  (includes the trailing NUL)
//! 13      1             mapq
//! 14      2             bin
//! 16      2             n_cigar_op
//! 18      2             flag
//! 20      4             l_seq
//! 24      4             next_refID
//! 28      4             next_pos
//! 32      4             tlen
//! 36      l_read_name   read name, NUL-terminated
//! ..      4*n_cigar_op  CIGAR
//! ..      (l_seq+1)/2   sequence, 4-bit packed
//! ..      l_seq         qualities (0xFF * l_seq if absent)
//! ..      remainder     aux tags
//! ```

use fritillaria_core::{Error, Result};

use crate::aux::{Array, Fields, Tag, Value, Values};
use crate::seq::{cigar_op_kind, cigar_op_len, decode_sequence, packed_len};

/// Size of the fixed core, excluding the `block_size` prefix.
pub const RECORD_CORE_SIZE: usize = 32;
/// Minimum bytes a record occupies, including its `block_size` prefix.
pub const MIN_RECORD_SIZE: usize = 4 + RECORD_CORE_SIZE;

fn read_i32(buf: &[u8], pos: usize) -> Option<i32> {
    Some(i32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?))
}

fn read_u32(buf: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?))
}

fn read_u16(buf: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?))
}

/// Walks record boundaries, returning the start offset of each record.
///
/// Sequential by necessity — each record's length is only known after reading
/// its prefix — but cheap, since the body is skipped rather than parsed.
///
/// A trailing partial record is **not** an error: it means the batch ended
/// mid-record. Those bytes are reported via the returned tail offset so the
/// caller can carry them into the next batch. This is the seam where this
/// class of library actually breaks.
///
/// Returns `(offsets, tail)` where `tail` is the offset of the first
/// incompletely-buffered record, or `buf.len()` if the buffer ends cleanly.
pub fn scan_records(buf: &[u8], start: usize) -> Result<(Vec<usize>, usize)> {
    let mut offsets = Vec::new();
    let mut pos = start;

    loop {
        let Some(block_size) = read_u32(buf, pos) else {
            break; // fewer than 4 bytes left: partial prefix
        };

        let block_size = block_size as usize;
        if block_size < RECORD_CORE_SIZE {
            return Err(Error::Malformed {
                format: "bam",
                position: pos as u64,
                reason: format!(
                    "block_size {block_size} smaller than the {RECORD_CORE_SIZE}-byte core"
                ),
            });
        }

        let Some(end) = pos.checked_add(4 + block_size) else {
            return Err(Error::Malformed {
                format: "bam",
                position: pos as u64,
                reason: "record length overflows".to_string(),
            });
        };

        if end > buf.len() {
            break; // partial record: stop, report the tail
        }

        offsets.push(pos);
        pos = end;
    }

    Ok((offsets, pos))
}

/// A borrowed view over one BAM record.
///
/// Holds no owned data; variable-length fields are sliced from the underlying
/// decompressed buffer on demand. This is the record-at-a-time adapter over the
/// columnar core — convenient, but materialising one per record is exactly what
/// the columnar path exists to avoid.
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    buf: &'a [u8],
}

impl<'a> Record<'a> {
    /// Wraps the bytes of a single record, `block_size` prefix included.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < MIN_RECORD_SIZE {
            return Err(Error::Malformed {
                format: "bam",
                position: 0,
                reason: format!("record of {} bytes is too short", buf.len()),
            });
        }
        Ok(Self { buf })
    }

    /// Reference sequence ID; `-1` for unmapped/unplaced.
    #[must_use]
    pub fn reference_sequence_id(&self) -> i32 {
        read_i32(self.buf, 4).unwrap_or(-1)
    }

    /// **0-based** leftmost position. `-1` means unplaced.
    ///
    /// SAM shows this 1-based; converting is the caller's job and forgetting to
    /// is an off-by-one that silently shifts every coordinate.
    #[must_use]
    pub fn position(&self) -> i32 {
        read_i32(self.buf, 8).unwrap_or(-1)
    }

    /// Length of the read name, including its trailing NUL.
    #[must_use]
    pub fn name_len(&self) -> usize {
        self.buf.get(12).copied().unwrap_or(0) as usize
    }

    /// Mapping quality.
    #[must_use]
    pub fn mapping_quality(&self) -> u8 {
        self.buf.get(13).copied().unwrap_or(0)
    }

    /// Precomputed index bin.
    #[must_use]
    pub fn bin(&self) -> u16 {
        read_u16(self.buf, 14).unwrap_or(0)
    }

    /// Number of CIGAR operations.
    ///
    /// A value of 2 combined with a `CG` aux tag may indicate a long CIGAR that
    /// overflowed this 16-bit field; see the module docs of [`crate`].
    #[must_use]
    pub fn cigar_op_count(&self) -> usize {
        read_u16(self.buf, 16).unwrap_or(0) as usize
    }

    /// Bitwise FLAG.
    #[must_use]
    pub fn flags(&self) -> u16 {
        read_u16(self.buf, 18).unwrap_or(0)
    }

    /// Number of bases in the read.
    #[must_use]
    pub fn sequence_len(&self) -> usize {
        read_u32(self.buf, 20).unwrap_or(0) as usize
    }

    /// Mate reference sequence ID.
    #[must_use]
    pub fn mate_reference_sequence_id(&self) -> i32 {
        read_i32(self.buf, 24).unwrap_or(-1)
    }

    /// Mate 0-based position.
    #[must_use]
    pub fn mate_position(&self) -> i32 {
        read_i32(self.buf, 28).unwrap_or(-1)
    }

    /// Observed template length.
    #[must_use]
    pub fn template_length(&self) -> i32 {
        read_i32(self.buf, 32).unwrap_or(0)
    }

    // Variable-length fields are laid out back to back after the core, each
    // starting where the previous one ends. These helpers exist so that offset
    // is derived once rather than re-summed at every accessor — recomputing it
    // inline is how fields end up silently shifted by a few bytes.

    /// Offset of the read name. `MIN_RECORD_SIZE` already counts the
    /// `block_size` prefix, so there is no further `+ 4` here.
    const fn name_start() -> usize {
        MIN_RECORD_SIZE
    }

    fn cigar_start(&self) -> usize {
        Self::name_start() + self.name_len()
    }

    fn sequence_start(&self) -> usize {
        self.cigar_start() + self.cigar_op_count() * 4
    }

    fn qualities_start(&self) -> usize {
        self.sequence_start() + packed_len(self.sequence_len())
    }

    fn aux_start(&self) -> usize {
        self.qualities_start() + self.sequence_len()
    }

    /// Read name without its trailing NUL.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        let raw = self
            .buf
            .get(Self::name_start()..self.cigar_start())
            .unwrap_or_default();
        // l_read_name counts the NUL; callers want the name itself.
        raw.strip_suffix(b"\0").unwrap_or(raw)
    }

    /// Raw CIGAR operations, one `u32` each.
    #[must_use]
    pub fn cigar_raw(&self) -> &'a [u8] {
        self.buf
            .get(self.cigar_start()..self.sequence_start())
            .unwrap_or_default()
    }

    /// 4-bit packed sequence.
    #[must_use]
    pub fn sequence_packed(&self) -> &'a [u8] {
        self.buf
            .get(self.sequence_start()..self.qualities_start())
            .unwrap_or_default()
    }

    /// Sequence decoded to ASCII bases.
    #[must_use]
    pub fn sequence(&self) -> Vec<u8> {
        decode_sequence(self.sequence_packed(), self.sequence_len())
    }

    /// Phred qualities, unshifted. All `0xFF` means absent.
    #[must_use]
    pub fn qualities(&self) -> &'a [u8] {
        self.buf
            .get(self.qualities_start()..self.aux_start())
            .unwrap_or_default()
    }

    /// Raw aux tag block; undecoded.
    ///
    /// Bounded by this record's own length, not the end of the buffer — the
    /// buffer usually continues into the next record.
    #[must_use]
    pub fn aux_raw(&self) -> &'a [u8] {
        let end = 4 + read_u32(self.buf, 0).unwrap_or(0) as usize;
        self.buf.get(self.aux_start()..end).unwrap_or_default()
    }

    /// Decoded aux fields.
    ///
    /// Lazy and zero-copy; see [`crate::aux`] for what the values borrow. Each
    /// item is a `Result` because one malformed field invalidates every field
    /// after it.
    #[must_use]
    pub fn aux(&self) -> Fields<'a> {
        Fields::new(self.aux_raw())
    }

    /// Looks up a single aux tag.
    ///
    /// `Ok(None)` means the tag is absent; `Err` means the aux block is
    /// malformed at or before where the tag would have been. Scans linearly,
    /// which is what the format allows — there is no index.
    pub fn aux_get(&self, tag: Tag) -> Result<Option<Value<'a>>> {
        for field in self.aux() {
            let (t, value) = field?;
            if t == tag {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// CIGAR operations as raw `u32`s, exactly as stored in this record.
    ///
    /// This is the *stored* CIGAR: if it is the long-CIGAR placeholder, this
    /// returns the two placeholder ops, not the real ones. Use
    /// [`Record::cigar_resolved`] to follow `CG`.
    #[must_use]
    pub fn cigar(&self) -> Values<'a, u32> {
        Values::new(self.cigar_raw())
    }

    /// Whether the stored CIGAR is the placeholder for a long CIGAR.
    ///
    /// `n_cigar_op` is a 16-bit field, so a CIGAR of more than 65535 operations
    /// cannot be stored inline. The spec's workaround puts the real CIGAR in a
    /// `CG` tag of type `B:I` and leaves `<l_seq>S<ref_len>N` in its place.
    ///
    /// The shape is checked rather than merely counting ops, because a genuine
    /// two-op `S`/`N` record is legal and must not be mistaken for a
    /// placeholder: the soft clip has to cover the *whole* read.
    #[must_use]
    pub fn has_long_cigar_placeholder(&self) -> bool {
        if self.cigar_op_count() != 2 {
            return false;
        }
        let ops = self.cigar();
        let (Some(first), Some(second)) = (ops.get(0), ops.get(1)) else {
            return false;
        };
        cigar_op_kind(first) == Some(b'S')
            && cigar_op_len(first) as usize == self.sequence_len()
            && cigar_op_kind(second) == Some(b'N')
    }

    /// CIGAR operations, following the `CG` tag when the stored CIGAR is the
    /// long-CIGAR placeholder.
    ///
    /// Returns the same `u32` encoding either way, so callers need no special
    /// case. `CG` is a `B:I` array of exactly the ops that would have been
    /// inline, so this is a change of location, not of encoding.
    ///
    /// Errors if the aux block is malformed, or if the placeholder is present
    /// but `CG` is missing or is not a `B:I` array — a record claiming a long
    /// CIGAR with nowhere to read it from is corrupt, and silently returning
    /// the placeholder would hand back a CIGAR that does not describe the read.
    pub fn cigar_resolved(&self) -> Result<Values<'a, u32>> {
        if !self.has_long_cigar_placeholder() {
            return Ok(self.cigar());
        }

        match self.aux_get(*b"CG")? {
            Some(Value::Array(Array::UInt32(values))) => Ok(values),
            Some(other) => Err(Error::Malformed {
                format: "bam",
                position: 0,
                reason: format!(
                    "long-CIGAR placeholder present but CG is type '{}', not B:I",
                    other.ty() as char
                ),
            }),
            None => Err(Error::Malformed {
                format: "bam",
                position: 0,
                reason: "long-CIGAR placeholder present but no CG tag".to_string(),
            }),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    // Fixture builders use literal sizes and spell out CIGAR op codes
    // explicitly (`| 0` is op M); both read better than the lint-clean form.
    #![allow(clippy::cast_possible_wrap, clippy::identity_op)]

    use super::*;

    /// Builds a minimal but spec-shaped record.
    pub(crate) fn build_record(name: &[u8], seq: &[u8], quals: Option<&[u8]>) -> Vec<u8> {
        // Encode bases to nibbles, high nibble first.
        let mut packed = vec![0u8; seq.len().div_ceil(2)];
        for (i, base) in seq.iter().enumerate() {
            let nibble = crate::seq::BASES.iter().position(|b| b == base).unwrap() as u8;
            if i % 2 == 0 {
                packed[i / 2] |= nibble << 4;
            } else {
                packed[i / 2] |= nibble;
            }
        }

        let mut name_field = name.to_vec();
        name_field.push(0); // NUL terminator, counted by l_read_name

        let cigar: [u32; 1] = [((seq.len() as u32) << 4) | 0]; // <l_seq>M
        let quals = quals.map_or_else(|| vec![0xff; seq.len()], <[u8]>::to_vec);

        let mut body = Vec::new();
        body.extend_from_slice(&0i32.to_le_bytes()); // refID
        body.extend_from_slice(&100i32.to_le_bytes()); // pos
        body.push(name_field.len() as u8); // l_read_name
        body.push(60); // mapq
        body.extend_from_slice(&4680u16.to_le_bytes()); // bin
        body.extend_from_slice(&1u16.to_le_bytes()); // n_cigar_op
        body.extend_from_slice(&99u16.to_le_bytes()); // flag
        body.extend_from_slice(&(seq.len() as u32).to_le_bytes()); // l_seq
        body.extend_from_slice(&0i32.to_le_bytes()); // next_refID
        body.extend_from_slice(&200i32.to_le_bytes()); // next_pos
        body.extend_from_slice(&150i32.to_le_bytes()); // tlen
        assert_eq!(body.len(), RECORD_CORE_SIZE);

        body.extend_from_slice(&name_field);
        for op in cigar {
            body.extend_from_slice(&op.to_le_bytes());
        }
        body.extend_from_slice(&packed);
        body.extend_from_slice(&quals);

        let mut record = (body.len() as u32).to_le_bytes().to_vec();
        record.extend_from_slice(&body);
        record
    }

    #[test]
    fn reads_every_core_field() {
        let raw = build_record(b"read1", b"ACGT", None);
        let record = Record::new(&raw).unwrap();

        assert_eq!(record.reference_sequence_id(), 0);
        assert_eq!(record.position(), 100);
        assert_eq!(record.mapping_quality(), 60);
        assert_eq!(record.bin(), 4680);
        assert_eq!(record.cigar_op_count(), 1);
        assert_eq!(record.flags(), 99);
        assert_eq!(record.sequence_len(), 4);
        assert_eq!(record.mate_reference_sequence_id(), 0);
        assert_eq!(record.mate_position(), 200);
        assert_eq!(record.template_length(), 150);
    }

    #[test]
    fn reads_variable_length_fields() {
        let raw = build_record(b"read1", b"ACGT", Some(&[30, 31, 32, 33]));
        let record = Record::new(&raw).unwrap();

        assert_eq!(record.name(), b"read1", "trailing NUL must be stripped");
        assert_eq!(record.sequence(), b"ACGT");
        assert_eq!(record.qualities(), &[30, 31, 32, 33]);
        assert_eq!(record.cigar_raw().len(), 4);
        assert!(record.aux_raw().is_empty());
    }

    #[test]
    fn handles_odd_length_sequence() {
        let raw = build_record(b"r", b"ACG", None);
        let record = Record::new(&raw).unwrap();
        assert_eq!(record.sequence_len(), 3);
        assert_eq!(record.sequence(), b"ACG");
        assert_eq!(record.sequence_packed().len(), 2);
        assert_eq!(record.qualities().len(), 3);
    }

    #[test]
    fn handles_absent_qualities() {
        let raw = build_record(b"r", b"ACGT", None);
        let record = Record::new(&raw).unwrap();
        assert_eq!(record.qualities(), &[0xff; 4]);
        assert!(crate::seq::qualities_are_absent(record.qualities()));
    }

    #[test]
    fn handles_empty_sequence() {
        let raw = build_record(b"r", b"", None);
        let record = Record::new(&raw).unwrap();
        assert_eq!(record.sequence_len(), 0);
        assert!(record.sequence().is_empty());
        assert!(record.qualities().is_empty());
    }

    #[test]
    fn scans_consecutive_record_boundaries() {
        let mut buf = Vec::new();
        let mut expected = Vec::new();
        for name in [&b"a"[..], &b"bb"[..], &b"ccc"[..]] {
            expected.push(buf.len());
            buf.extend_from_slice(&build_record(name, b"ACGT", None));
        }

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets, expected);
        assert_eq!(tail, buf.len(), "buffer ends exactly on a record boundary");
    }

    #[test]
    fn scan_stops_at_a_record_split_across_the_batch_edge() {
        // The seam that breaks this class of library: a batch ending mid-record.
        let mut buf = build_record(b"first", b"ACGT", None);
        let boundary = buf.len();
        buf.extend_from_slice(&build_record(b"second", b"ACGT", None));
        buf.truncate(buf.len() - 5); // clip the last record

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets, vec![0], "partial record must not be yielded");
        assert_eq!(tail, boundary, "tail must point at the partial record");
    }

    #[test]
    fn scan_stops_on_a_partial_length_prefix() {
        let mut buf = build_record(b"r", b"ACGT", None);
        let boundary = buf.len();
        buf.extend_from_slice(&[0x00, 0x01]); // 2 bytes of the next prefix

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets.len(), 1);
        assert_eq!(tail, boundary);
    }

    #[test]
    fn scan_rejects_an_impossibly_small_record() {
        let buf = 4u32.to_le_bytes().to_vec();
        assert!(matches!(
            scan_records(&buf, 0),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn scan_of_an_empty_buffer_is_empty_not_an_error() {
        let (offsets, tail) = scan_records(&[], 0).unwrap();
        assert!(offsets.is_empty());
        assert_eq!(tail, 0);
    }

    #[test]
    fn rejects_a_truncated_record() {
        assert!(Record::new(&[0u8; 8]).is_err());
    }

    /// Builds a record with an explicit CIGAR and aux block, for the cases
    /// `build_record` cannot express.
    fn build_record_with(seq_len: usize, cigar: &[u32], aux: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&0i32.to_le_bytes()); // refID
        body.extend_from_slice(&0i32.to_le_bytes()); // pos
        body.push(2); // l_read_name, "r\0"
        body.push(60);
        body.extend_from_slice(&0u16.to_le_bytes()); // bin
        body.extend_from_slice(&(cigar.len() as u16).to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // flag
        body.extend_from_slice(&(seq_len as u32).to_le_bytes());
        body.extend_from_slice(&(-1i32).to_le_bytes()); // next_refID
        body.extend_from_slice(&(-1i32).to_le_bytes()); // next_pos
        body.extend_from_slice(&0i32.to_le_bytes()); // tlen
        assert_eq!(body.len(), RECORD_CORE_SIZE);

        body.extend_from_slice(b"r\0");
        for op in cigar {
            body.extend_from_slice(&op.to_le_bytes());
        }
        body.extend_from_slice(&vec![0x11; seq_len.div_ceil(2)]); // packed seq
        body.extend_from_slice(&vec![0xff; seq_len]); // absent quals
        body.extend_from_slice(aux);

        let mut record = (body.len() as u32).to_le_bytes().to_vec();
        record.extend_from_slice(&body);
        record
    }

    /// `<len><op>` packed the way BAM stores a CIGAR operation.
    fn op(len: u32, kind: u8) -> u32 {
        let code = crate::seq::CIGAR_OPS
            .iter()
            .position(|&k| k == kind)
            .unwrap();
        (len << 4) | code as u32
    }

    /// A `CG:B:I` aux field holding `ops`.
    fn cg_tag(ops: &[u32]) -> Vec<u8> {
        let mut v = b"CGBI".to_vec();
        v.extend_from_slice(&(ops.len() as u32).to_le_bytes());
        for o in ops {
            v.extend_from_slice(&o.to_le_bytes());
        }
        v
    }

    #[test]
    fn resolves_a_long_cigar_from_the_cg_tag() {
        // The >65535-op overflow. No fixture in this repo exercises it: HiFi
        // reads are far too accurate to produce that many ops, so this is
        // built to spec by hand.
        let seq_len = 1000;
        let real: Vec<u32> = (0..70_000).map(|_| op(1, b'M')).collect();
        let placeholder = [op(seq_len as u32, b'S'), op(500, b'N')];
        let raw = build_record_with(seq_len, &placeholder, &cg_tag(&real));
        let record = Record::new(&raw).unwrap();

        assert!(record.has_long_cigar_placeholder());
        assert_eq!(
            record.cigar().len(),
            2,
            "the stored CIGAR is the placeholder"
        );

        let resolved = record.cigar_resolved().unwrap();
        assert_eq!(resolved.len(), 70_000);
        assert_eq!(resolved.get(0), Some(op(1, b'M')));
        assert_eq!(resolved.get(69_999), Some(op(1, b'M')));
    }

    #[test]
    fn a_genuine_two_op_record_is_not_a_placeholder() {
        // The false positive that matters: `S` then `N` is a legal CIGAR. It is
        // only a placeholder when the soft clip covers the entire read.
        let raw = build_record_with(1000, &[op(10, b'S'), op(500, b'N')], &[]);
        let record = Record::new(&raw).unwrap();

        assert!(!record.has_long_cigar_placeholder());
        assert_eq!(record.cigar_resolved().unwrap().len(), 2);
    }

    #[test]
    fn a_placeholder_without_a_cg_tag_is_an_error() {
        // Returning the placeholder here would hand back a CIGAR that does not
        // describe the read, which is worse than failing.
        let raw = build_record_with(1000, &[op(1000, b'S'), op(500, b'N')], &[]);
        let record = Record::new(&raw).unwrap();

        assert!(record.has_long_cigar_placeholder());
        assert!(matches!(
            record.cigar_resolved(),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn a_placeholder_with_a_wrongly_typed_cg_is_an_error() {
        let mut aux = b"CGZ".to_vec();
        aux.extend_from_slice(b"not an array\0");
        let raw = build_record_with(1000, &[op(1000, b'S'), op(500, b'N')], &aux);
        let record = Record::new(&raw).unwrap();

        assert!(matches!(
            record.cigar_resolved(),
            Err(Error::Malformed { format: "bam", .. })
        ));
    }

    #[test]
    fn cigar_resolved_passes_through_an_ordinary_cigar() {
        let raw = build_record(b"read1", b"ACGT", None);
        let record = Record::new(&raw).unwrap();
        let ops = record.cigar_resolved().unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops.get(0), Some(op(4, b'M')));
    }

    #[test]
    fn aux_lookup_finds_a_tag_and_reports_absence() {
        let mut aux = b"NMC".to_vec();
        aux.push(7);
        aux.extend_from_slice(b"XZZhello\0");
        let raw = build_record_with(4, &[op(4, b'M')], &aux);
        let record = Record::new(&raw).unwrap();

        assert_eq!(record.aux_get(*b"NM").unwrap().unwrap().as_int(), Some(7));
        assert_eq!(
            record.aux_get(*b"XZ").unwrap().unwrap().as_bytes(),
            Some(&b"hello"[..])
        );
        assert!(record.aux_get(*b"ZZ").unwrap().is_none());
        assert_eq!(record.aux().count(), 2);
    }
}
