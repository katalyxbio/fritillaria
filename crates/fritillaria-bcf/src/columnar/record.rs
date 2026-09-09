//! Record boundary discovery, and a zero-copy record view.
//!
//! ```text
//! offset  size   field
//! 0       4      l_shared    bytes from CHROM to the end of INFO
//! 4       4      l_indiv     bytes of FORMAT and the genotype columns
//! 8       4      CHROM       int32, an offset into the contig dictionary
//! 12      4      POS         int32, 0-based
//! 16      4      rlen        int32, span on the reference
//! 20      4      QUAL        float, 0x7F800001 when missing
//! 24      2      n_info      uint16
//! 26      2      n_allele    uint16
//! 28      3      n_sample    uint24, little-endian
//! 31      1      n_fmt       uint8
//! 32      ..     ID          typed string
//! ..      ..     REF+ALT     n_allele typed strings
//! ..      ..     FILTER      typed vector of dictionary offsets
//! ..      ..     INFO        n_info pairs of (typed key, typed value)
//! ..      ..     FORMAT      n_fmt of (typed key, one descriptor, n_sample slots)
//! ```
//!
//! # Two length prefixes, not one
//!
//! Boundary discovery is the same shape as BAM's — read a prefix, skip the
//! body — but the length is split in two, and the split is the point: a caller
//! that only wants sites can skip `l_indiv` entirely, which on
//! `testdata/kg_phase3.bcf` is over 99% of the file. The scan below needs both,
//! since the record ends after them.
//!
//! # BCF records cross BGZF blocks, essentially always
//!
//! `fritillaria-bam` leans on a habit of htslib's: it calls `bgzf_flush_try`
//! before writing each alignment, so a BAM block ends early rather than
//! splitting a record, and a block start is *almost always* a record start.
//! That is what makes BAM's per-block speculative scan work.
//!
//! **It is false for BCF.** `bcf_write` goes straight to `bgzf_write`, which
//! packs blocks full, so a record straddles nearly every interior boundary.
//! Measured on the fixtures: **0 of 56** interior block starts in
//! `kg_phase3.bcf` are record starts, and 0 of 3 in `giab_hg002.bcf`. The BAM
//! trick does not transfer, and assuming it does would not fail loudly — it
//! would resynchronise onto a plausible wrong offset.
//!
//! Hence [`looks_like_a_record`], which is the seed of the replacement: a
//! fixed-cost validator strong enough to test a *guessed* offset, so a device
//! scan can try many candidate offsets per block in parallel rather than
//! following one serial chain. See `docs/bcf-boundaries.md`.

use fritillaria_core::{Error, Result};

use crate::columnar::typed::{self, Float, Int, Kind, Typed};

/// Bytes of fixed core after the two length prefixes: CHROM through `n_fmt`.
pub const SITE_CORE_SIZE: usize = 24;
/// Minimum bytes a record occupies, both length prefixes included.
pub const MIN_RECORD_SIZE: usize = 8 + SITE_CORE_SIZE;

fn malformed(position: usize, reason: impl Into<String>) -> Error {
    Error::Malformed {
        format: "bcf",
        position: position as u64,
        reason: reason.into(),
    }
}

fn read_u32(buf: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(pos..pos + 4)?.try_into().ok()?))
}

fn read_i32(buf: &[u8], pos: usize) -> Option<i32> {
    read_u32(buf, pos).map(u32::cast_signed)
}

fn read_u16(buf: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?))
}

/// `n_sample`, stored as three little-endian bytes.
fn read_u24(buf: &[u8], pos: usize) -> Option<u32> {
    let raw = buf.get(pos..pos + 3)?;
    Some(u32::from(raw[0]) | u32::from(raw[1]) << 8 | u32::from(raw[2]) << 16)
}

/// Walks record boundaries, returning the start offset of each record.
///
/// Sequential by necessity, and cheap for the same reason BAM's is: only the
/// two length prefixes are read, and the body is stepped over.
///
/// A trailing partial record is **not** an error — the batch ended mid-record.
/// Those bytes are reported through the returned tail so the caller can carry
/// them forward, which for BCF is the common case rather than the edge case
/// (see the module docs).
///
/// Returns `(offsets, tail)` where `tail` is the offset of the first
/// incompletely-buffered record, or `buf.len()` if the buffer ends cleanly.
pub fn scan_records(buf: &[u8], start: usize) -> Result<(Vec<usize>, usize)> {
    let mut offsets = Vec::new();
    let mut pos = start;

    loop {
        let (Some(l_shared), Some(l_indiv)) = (read_u32(buf, pos), read_u32(buf, pos + 4)) else {
            break; // fewer than 8 bytes left: partial prefix
        };

        if (l_shared as usize) < SITE_CORE_SIZE {
            return Err(malformed(
                pos,
                format!("l_shared {l_shared} smaller than the {SITE_CORE_SIZE}-byte site core"),
            ));
        }

        let Some(end) = pos
            .checked_add(8)
            .and_then(|p| p.checked_add(l_shared as usize))
            .and_then(|p| p.checked_add(l_indiv as usize))
        else {
            return Err(malformed(pos, "record length overflows"));
        };

        if end > buf.len() {
            break; // partial record: stop, report the tail
        }

        offsets.push(pos);
        pos = end;
    }

    Ok((offsets, pos))
}

/// Whether the bytes at `pos` plausibly begin a record.
///
/// A **fixed-cost** test — it reads the 32-byte prefix and nothing else, with
/// no data-dependent loop — because its reason for existing is to be run at
/// many candidate offsets in parallel on a device, where following a chain is
/// the thing being avoided.
///
/// `expected_samples` and `contig_count` come from the header. They are what
/// makes the test strong: the spec requires every record's `n_sample` to equal
/// the header's sample count, and `CHROM` to name a declared contig, so a
/// random 32 bytes must match a specific 3-byte value and land in a small range
/// to survive.
///
/// **A `true` is a hypothesis, not a verdict.** Confirming it means walking the
/// typed-value chain ([`Record::validate`]) and, past that, showing the record
/// chain actually arrives at this offset — the same discipline
/// `fritillaria_bam::columnar::blocked` applies to its per-block guesses.
#[must_use]
pub fn looks_like_a_record(
    buf: &[u8],
    pos: usize,
    expected_samples: u32,
    contig_count: u32,
) -> bool {
    let (Some(l_shared), Some(l_indiv)) = (read_u32(buf, pos), read_u32(buf, pos + 4)) else {
        return false;
    };
    if (l_shared as usize) < SITE_CORE_SIZE {
        return false;
    }
    let Some(end) = pos
        .checked_add(8)
        .and_then(|p| p.checked_add(l_shared as usize))
        .and_then(|p| p.checked_add(l_indiv as usize))
    else {
        return false;
    };
    if end > buf.len() {
        return false;
    }

    let (Some(chrom), Some(position), Some(rlen)) = (
        read_i32(buf, pos + 8),
        read_i32(buf, pos + 12),
        read_i32(buf, pos + 16),
    ) else {
        return false;
    };
    let (Some(n_allele), Some(n_sample), Some(n_fmt)) = (
        read_u16(buf, pos + 26),
        read_u24(buf, pos + 28),
        buf.get(pos + 31).copied(),
    ) else {
        return false;
    };

    if chrom < 0 || chrom.cast_unsigned() >= contig_count {
        return false;
    }
    // POS is 0-based; -1 marks a telomere record, and nothing below that is
    // representable.
    if position < -1 || rlen < 0 {
        return false;
    }
    if n_sample != expected_samples {
        return false;
    }
    // Every record has a REF, so n_allele is at least 1.
    if n_allele == 0 {
        return false;
    }
    // With no FORMAT fields there is no genotype block at all, and with some
    // there must be enough room for a key and a descriptor apiece.
    if n_fmt == 0 {
        return l_indiv == 0;
    }
    l_indiv as usize >= usize::from(n_fmt) * 2
}

/// A borrowed view over one BCF record.
///
/// Holds no owned data; every variable-length field is sliced from the
/// underlying buffer on demand.
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    buf: &'a [u8],
    /// Offset of the first byte after the shared block.
    indiv_start: usize,
}

impl<'a> Record<'a> {
    /// Wraps the bytes of a single record, both length prefixes included.
    ///
    /// `buf` must be exactly the record; a longer slice is an error rather than
    /// a tolerated overshoot, because silently accepting one would let a
    /// mis-scanned boundary through as a valid record.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < MIN_RECORD_SIZE {
            return Err(malformed(
                0,
                format!(
                    "record of {} bytes is shorter than {MIN_RECORD_SIZE}",
                    buf.len()
                ),
            ));
        }
        let (Some(l_shared), Some(l_indiv)) = (read_u32(buf, 0), read_u32(buf, 4)) else {
            return Err(malformed(0, "truncated length prefixes"));
        };
        let (l_shared, l_indiv) = (l_shared as usize, l_indiv as usize);
        if l_shared < SITE_CORE_SIZE {
            return Err(malformed(
                0,
                format!("l_shared {l_shared} smaller than the {SITE_CORE_SIZE}-byte site core"),
            ));
        }
        if 8 + l_shared + l_indiv != buf.len() {
            return Err(malformed(
                0,
                format!(
                    "record claims {} bytes but was given {}",
                    8 + l_shared + l_indiv,
                    buf.len()
                ),
            ));
        }
        Ok(Self {
            buf,
            indiv_start: 8 + l_shared,
        })
    }

    /// Contig index — an offset into [`crate::columnar::Dictionary::contigs`], not a name.
    #[must_use]
    pub fn chromosome_id(&self) -> i32 {
        read_i32(self.buf, 8).unwrap_or(-1)
    }

    /// **0-based** leftmost position.
    ///
    /// VCF text shows this 1-based. Converting is the caller's job, and
    /// forgetting to is an off-by-one that shifts every coordinate silently.
    #[must_use]
    pub fn position(&self) -> i32 {
        read_i32(self.buf, 12).unwrap_or(-1)
    }

    /// Span on the reference — the REF length, or the END-implied length for a
    /// symbolic allele.
    #[must_use]
    pub fn reference_span(&self) -> i32 {
        read_i32(self.buf, 16).unwrap_or(0)
    }

    /// QUAL, or `None` when missing.
    ///
    /// Missing is the signalling NaN `0x7F800001`, so this cannot be a plain
    /// `f32` with NaN meaning absent — BCF permits a genuine NaN QUAL and the
    /// two must stay distinguishable.
    #[must_use]
    pub fn quality(&self) -> Option<f32> {
        read_u32(self.buf, 20).and_then(|bits| Float::classify(bits).value())
    }

    /// Number of INFO fields.
    #[must_use]
    pub fn info_count(&self) -> usize {
        read_u16(self.buf, 24).unwrap_or(0) as usize
    }

    /// Number of alleles, REF included, so always at least 1.
    #[must_use]
    pub fn allele_count(&self) -> usize {
        read_u16(self.buf, 26).unwrap_or(0) as usize
    }

    /// Number of samples, which must equal the header's.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        read_u24(self.buf, 28).unwrap_or(0) as usize
    }

    /// Number of FORMAT keys.
    #[must_use]
    pub fn format_count(&self) -> usize {
        usize::from(self.buf.get(31).copied().unwrap_or(0))
    }

    /// The variant ID, or `None` when it is `.`.
    pub fn id(&self) -> Result<Option<&'a [u8]>> {
        let value = typed::read(self.buf, SITE_CORE_SIZE + 8)?;
        if value.is_missing() {
            return Ok(None);
        }
        value
            .as_str()
            .map(Some)
            .ok_or_else(|| malformed(SITE_CORE_SIZE + 8, "ID is not a string"))
    }

    /// REF and the ALT alleles, in order; the first is REF.
    pub fn alleles(&self) -> Result<Vec<&'a [u8]>> {
        let mut pos = typed::skip(self.buf, SITE_CORE_SIZE + 8)?;
        let mut out = Vec::with_capacity(self.allele_count());
        for index in 0..self.allele_count() {
            let value = typed::read(self.buf, pos)?;
            let text = value
                .as_str()
                .ok_or_else(|| malformed(pos, format!("allele {index} is not a string")))?;
            out.push(text);
            pos += value.encoded_len();
        }
        Ok(out)
    }

    /// FILTER entries as dictionary offsets; empty when the field is `.`.
    ///
    /// PASS is offset 0, always — a record may say PASS with no `##FILTER`
    /// line declaring it, so the dictionary reserves the slot regardless.
    pub fn filters(&self) -> Result<Vec<i32>> {
        let pos = self.filter_offset()?;
        let value = typed::read(self.buf, pos)?;
        if value.is_missing() {
            return Ok(Vec::new());
        }
        let ints = value
            .ints()
            .ok_or_else(|| malformed(pos, "FILTER is not an integer vector"))?;
        Ok(ints.filter_map(Int::value).collect())
    }

    /// The INFO fields, as `(key, value)` where the key is a dictionary offset.
    pub fn info(&self) -> Result<InfoFields<'a>> {
        let pos = typed::skip(self.buf, self.filter_offset()?)?;
        Ok(InfoFields {
            buf: self.buf,
            pos,
            remaining: self.info_count(),
            end: self.indiv_start,
        })
    }

    /// Looks up one INFO field by dictionary offset.
    ///
    /// Linear, which is right: `n_info` is at most a few dozen and building an
    /// index would cost more than the scan.
    pub fn info_get(&self, key: i32) -> Result<Option<Typed<'a>>> {
        for field in self.info()? {
            let (found, value) = field?;
            if found == key {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// The FORMAT fields — the genotype block.
    #[must_use]
    pub fn formats(&self) -> FormatFields<'a> {
        FormatFields {
            buf: self.buf,
            pos: self.indiv_start,
            remaining: self.format_count(),
            samples: self.sample_count(),
            end: self.buf.len(),
        }
    }

    /// Looks up one FORMAT field by dictionary offset.
    pub fn format_get(&self, key: i32) -> Result<Option<FormatField<'a>>> {
        for field in self.formats() {
            let field = field?;
            if field.key == key {
                return Ok(Some(field));
            }
        }
        Ok(None)
    }

    /// Walks every typed value and checks that both blocks are consumed exactly.
    ///
    /// This is the strong confirmation behind [`looks_like_a_record`]'s cheap
    /// hypothesis. `l_shared` and `l_indiv` are declared lengths, so a record
    /// whose fields end short of or past them is one whose framing disagrees
    /// with its contents — which is exactly what a misparse looks like, and
    /// exactly what a whole-file read would otherwise sail through.
    pub fn validate(&self) -> Result<()> {
        let mut pos = typed::skip(self.buf, SITE_CORE_SIZE + 8)?; // ID
        for _ in 0..self.allele_count() {
            pos = typed::skip(self.buf, pos)?;
        }
        pos = typed::skip(self.buf, pos)?; // FILTER
        for _ in 0..self.info_count() {
            pos = typed::skip(self.buf, pos)?; // key
            pos = typed::skip(self.buf, pos)?; // value
        }
        if pos != self.indiv_start {
            return Err(malformed(
                pos,
                format!(
                    "site fields end at {pos} but l_shared says {}",
                    self.indiv_start
                ),
            ));
        }
        // The genotype block must be consumed exactly too. Iterating the fields
        // only proves none of them runs *past* l_indiv; fields that stop short
        // leave unclaimed bytes, which is the same disagreement between framing
        // and contents that the check above catches on the shared block.
        //
        // Found by writing the kernel: `kernels/bcf_scan.cu` asserted this and
        // the reference did not, so they would have disagreed on a record no
        // real file contains. Being a differential oracle means the strictness
        // has to match in both directions.
        let mut fields = self.formats();
        for field in &mut fields {
            field?;
        }
        let at = fields.position();
        if at != self.buf.len() {
            return Err(malformed(
                at,
                format!(
                    "genotype fields end at {at} but the record ends at {}",
                    self.buf.len()
                ),
            ));
        }
        Ok(())
    }

    /// Offset of the FILTER value: past ID and every allele.
    fn filter_offset(&self) -> Result<usize> {
        let mut pos = typed::skip(self.buf, SITE_CORE_SIZE + 8)?;
        for _ in 0..self.allele_count() {
            pos = typed::skip(self.buf, pos)?;
        }
        Ok(pos)
    }
}

/// Iterator over a record's INFO fields.
#[derive(Clone, Debug)]
pub struct InfoFields<'a> {
    buf: &'a [u8],
    pos: usize,
    remaining: usize,
    end: usize,
}

impl<'a> Iterator for InfoFields<'a> {
    type Item = Result<(i32, Typed<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        let result = (|| {
            let key = typed::read(self.buf, self.pos)?;
            let key_offset = key
                .as_int()
                .ok_or_else(|| malformed(self.pos, "INFO key is not an atomic integer"))?;
            let value_at = self.pos + key.encoded_len();
            let value = typed::read(self.buf, value_at)?;
            let next = value_at + value.encoded_len();
            if next > self.end {
                return Err(malformed(value_at, "INFO field runs past the shared block"));
            }
            self.pos = next;
            Ok((key_offset, value))
        })();

        if result.is_err() {
            self.remaining = 0; // fuse: one error, not a cascade of them
        }
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

/// One FORMAT field: a key, one descriptor, and one fixed-width slot per sample.
///
/// The layout is by field then by sample, which is the transpose of VCF text —
/// and, not coincidentally, already columnar. A GPU consumer wanting one field
/// across all samples gets a contiguous run; wanting all fields of one sample
/// gets a stride.
#[derive(Clone, Copy, Debug)]
pub struct FormatField<'a> {
    /// Dictionary offset of the field key.
    pub key: i32,
    /// Element type of every sample's slot.
    pub kind: Kind,
    /// Elements per sample — the widest sample's count, with shorter samples
    /// padded by END_OF_VECTOR.
    pub count: usize,
    /// Number of samples.
    pub samples: usize,
    /// All samples' slots, back to back.
    pub data: &'a [u8],
}

impl<'a> FormatField<'a> {
    /// One sample's slot.
    ///
    /// Trailing END_OF_VECTOR elements are *kept*, not trimmed: whether they
    /// mean "haploid" or "this sample has fewer values" is the caller's
    /// interpretation, and dropping them here would erase the distinction.
    #[must_use]
    pub fn sample(&self, index: usize) -> Option<Typed<'a>> {
        let width = self.count * self.kind.size();
        let start = index.checked_mul(width)?;
        let slot = self.data.get(start..start.checked_add(width)?)?;
        Some(Typed::from_parts(self.kind, self.count, slot))
    }

    /// Bytes one sample's slot occupies.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.count * self.kind.size()
    }
}

/// Iterator over a record's FORMAT fields.
#[derive(Clone, Debug)]
pub struct FormatFields<'a> {
    buf: &'a [u8],
    pos: usize,
    remaining: usize,
    samples: usize,
    end: usize,
}

impl FormatFields<'_> {
    /// Where the walk has reached — the end of the last field yielded, or the
    /// start of the genotype block before any.
    ///
    /// [`Record::validate`] needs this: iterating the fields proves none runs
    /// past `l_indiv`, but not that together they fill it.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FormatFields<'a> {
    type Item = Result<FormatField<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        let result = (|| {
            let key = typed::read(self.buf, self.pos)?;
            let key_offset = key
                .as_int()
                .ok_or_else(|| malformed(self.pos, "FORMAT key is not an atomic integer"))?;

            // One descriptor for the whole field, then n_sample slots of it —
            // so unlike INFO, the payload length is n_sample times what the
            // descriptor alone implies.
            let at = self.pos + key.encoded_len();
            let (count, kind, header_len) = typed::read_descriptor(self.buf, at)?;
            let stride = count
                .checked_mul(kind.size())
                .ok_or_else(|| malformed(at, "FORMAT slot width overflows"))?;
            let total = stride
                .checked_mul(self.samples)
                .ok_or_else(|| malformed(at, "FORMAT field size overflows"))?;

            let start = at + header_len;
            let next = start
                .checked_add(total)
                .ok_or_else(|| malformed(at, "FORMAT field extends past usize"))?;
            if next > self.end {
                return Err(malformed(at, "FORMAT field runs past the genotype block"));
            }
            self.pos = next;

            Ok(FormatField {
                key: key_offset,
                kind,
                count,
                samples: self.samples,
                data: &self.buf[start..next],
            })
        })();

        if result.is_err() {
            self.remaining = 0;
        }
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

/// One sample's genotype, decoded from a `GT` FORMAT slot.
///
/// `(allele + 1) << 1 | phased`, so 0 is a missing allele and 1 is unused.
/// END_OF_VECTOR pads a sample with lower ploidy than the widest at the site —
/// every haploid male call on chrX, in a file that also holds diploid females.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allele {
    /// Allele index into REF+ALT, or `None` for `.`.
    pub index: Option<u32>,
    /// Whether this allele is phased with the one before it.
    pub phased: bool,
}

/// Decodes a `GT` slot into alleles, stopping at the END_OF_VECTOR padding.
///
/// The stop is the point. A haploid call in a diploid file is stored as
/// `[allele, END_OF_VECTOR]`, and reading the pad as data turns it into a
/// diploid call with a bogus second allele — plausible, wrong, and silent.
pub fn decode_genotype(slot: &Typed<'_>) -> Result<Vec<Allele>> {
    let ints = slot
        .ints()
        .ok_or_else(|| malformed(0, "GT is not an integer vector"))?;
    let mut out = Vec::with_capacity(slot.count());
    for element in ints {
        match element {
            Int::EndOfVector => break,
            Int::Value(raw) => {
                let encoded = raw.cast_unsigned();
                out.push(Allele {
                    index: (encoded >> 1).checked_sub(1),
                    phased: encoded & 1 == 1,
                });
            }
            // A 0 byte is `.` and classifies as an ordinary value, so Missing
            // here means the sentinel proper: no allele, no phase.
            Int::Missing | Int::Reserved(_) => out.push(Allele {
                index: None,
                phased: false,
            }),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use super::*;

    /// Builds one record. `alleles` are REF then ALTs.
    fn build(
        chrom: i32,
        pos: i32,
        alleles: &[&[u8]],
        info: &[(i32, Vec<u8>)],
        formats: &[(i32, Vec<u8>)],
        n_sample: u32,
    ) -> Vec<u8> {
        let mut shared = Vec::new();
        shared.extend_from_slice(&chrom.to_le_bytes());
        shared.extend_from_slice(&pos.to_le_bytes());
        shared.extend_from_slice(&1i32.to_le_bytes()); // rlen
        shared.extend_from_slice(&0x7F80_0001u32.to_le_bytes()); // QUAL missing
        shared.extend_from_slice(&(info.len() as u16).to_le_bytes());
        shared.extend_from_slice(&(alleles.len() as u16).to_le_bytes());
        shared.extend_from_slice(&n_sample.to_le_bytes()[..3]);
        shared.push(u8::try_from(formats.len()).unwrap());
        shared.push(0x07); // ID: missing string
        for allele in alleles {
            shared.push(u8::try_from(allele.len()).unwrap() << 4 | 0x07);
            shared.extend_from_slice(allele);
        }
        shared.push(0x00); // FILTER: missing
        for (key, value) in info {
            shared.push(0x11);
            shared.push(u8::try_from(*key).unwrap());
            shared.extend_from_slice(value);
        }

        let mut indiv = Vec::new();
        for (key, payload) in formats {
            indiv.push(0x11);
            indiv.push(u8::try_from(*key).unwrap());
            indiv.extend_from_slice(payload);
        }

        let mut out = Vec::new();
        out.extend_from_slice(&(shared.len() as u32).to_le_bytes());
        out.extend_from_slice(&(indiv.len() as u32).to_le_bytes());
        out.extend_from_slice(&shared);
        out.extend_from_slice(&indiv);
        out
    }

    fn simple() -> Vec<u8> {
        build(
            1,
            99,
            &[b"A", b"C"],
            &[(3, vec![0x11, 0x2A])],
            &[(4, vec![0x21, 0x02, 0x04])],
            1,
        )
    }

    #[test]
    fn reads_the_site_core() {
        let raw = simple();
        let record = Record::new(&raw).unwrap();
        assert_eq!(record.chromosome_id(), 1);
        assert_eq!(record.position(), 99);
        assert_eq!(record.reference_span(), 1);
        assert_eq!(record.quality(), None, "0x7F800001 is missing, not a NaN");
        assert_eq!(record.info_count(), 1);
        assert_eq!(record.allele_count(), 2);
        assert_eq!(record.sample_count(), 1);
        assert_eq!(record.format_count(), 1);
        record.validate().unwrap();
    }

    #[test]
    fn reads_id_alleles_filters_and_info() {
        let raw = simple();
        let record = Record::new(&raw).unwrap();
        assert_eq!(record.id().unwrap(), None);
        assert_eq!(record.alleles().unwrap(), vec![&b"A"[..], &b"C"[..]]);
        assert!(record.filters().unwrap().is_empty());
        assert_eq!(record.info_get(3).unwrap().unwrap().as_int(), Some(42));
        assert!(record.info_get(9).unwrap().is_none());
    }

    #[test]
    fn a_format_field_slices_one_slot_per_sample() {
        // Two samples, GT diploid: 0/1 and 1|1.
        let raw = build(
            0,
            0,
            &[b"A", b"C"],
            &[],
            &[(4, vec![0x21, 0x02, 0x04, 0x04, 0x05])],
            2,
        );
        let record = Record::new(&raw).unwrap();
        let gt = record.format_get(4).unwrap().unwrap();
        assert_eq!(gt.kind, Kind::Int8);
        assert_eq!(gt.count, 2);
        assert_eq!(gt.stride(), 2);
        assert_eq!(gt.samples, 2);

        let first = decode_genotype(&gt.sample(0).unwrap()).unwrap();
        assert_eq!(
            first,
            vec![
                Allele {
                    index: Some(0),
                    phased: false
                },
                Allele {
                    index: Some(1),
                    phased: false
                },
            ]
        );
        let second = decode_genotype(&gt.sample(1).unwrap()).unwrap();
        assert_eq!(
            second,
            vec![
                Allele {
                    index: Some(1),
                    phased: false
                },
                Allele {
                    index: Some(1),
                    phased: true
                },
            ]
        );
        assert!(gt.sample(2).is_none());
    }

    #[test]
    fn a_haploid_call_stops_at_the_end_of_vector_pad() {
        // The spec's own example: a haploid male and a diploid female at the
        // same chrX site. Reading the pad as data would make the male diploid.
        let raw = build(
            0,
            0,
            &[b"A", b"C"],
            &[],
            &[(4, vec![0x21, 0x02, 0x81, 0x02, 0x04])],
            2,
        );
        let record = Record::new(&raw).unwrap();
        let gt = record.format_get(4).unwrap().unwrap();
        assert_eq!(
            decode_genotype(&gt.sample(0).unwrap()).unwrap(),
            vec![Allele {
                index: Some(0),
                phased: false
            }],
            "the padded sample is haploid"
        );
        assert_eq!(decode_genotype(&gt.sample(1).unwrap()).unwrap().len(), 2);
    }

    #[test]
    fn a_dot_allele_decodes_as_no_index() {
        let raw = build(0, 0, &[b"A", b"C"], &[], &[(4, vec![0x21, 0x00, 0x00])], 1);
        let record = Record::new(&raw).unwrap();
        let gt = record.format_get(4).unwrap().unwrap();
        assert_eq!(
            decode_genotype(&gt.sample(0).unwrap()).unwrap(),
            vec![
                Allele {
                    index: None,
                    phased: false
                },
                Allele {
                    index: None,
                    phased: false
                },
            ]
        );
    }

    #[test]
    fn scan_finds_every_boundary_and_a_clean_tail() {
        let mut buf = vec![0xAAu8; 5]; // a stand-in header
        let mut expected = Vec::new();
        for i in 0..4 {
            expected.push(buf.len());
            buf.extend(build(0, i, &[b"A", b"C"], &[], &[], 0));
        }
        let (offsets, tail) = scan_records(&buf, 5).unwrap();
        assert_eq!(offsets, expected);
        assert_eq!(tail, buf.len());
    }

    #[test]
    fn scan_reports_a_partial_trailing_record() {
        let mut buf = build(0, 1, &[b"A", b"C"], &[], &[], 0);
        let first_len = buf.len();
        let second = build(0, 2, &[b"A", b"G"], &[], &[], 0);
        buf.extend_from_slice(&second[..second.len() - 3]);

        let (offsets, tail) = scan_records(&buf, 0).unwrap();
        assert_eq!(offsets, vec![0]);
        assert_eq!(tail, first_len, "the tail points at the partial record");

        // And a prefix too short even for the two length words.
        let (offsets, tail) = scan_records(&buf[..first_len + 3], 0).unwrap();
        assert_eq!(offsets, vec![0]);
        assert_eq!(tail, first_len);
    }

    #[test]
    fn scan_rejects_an_impossible_l_shared() {
        let mut buf = build(0, 1, &[b"A"], &[], &[], 0);
        buf[0..4].copy_from_slice(&4u32.to_le_bytes());
        assert!(scan_records(&buf, 0).is_err());
    }

    #[test]
    fn record_new_rejects_a_length_disagreement() {
        let raw = simple();
        assert!(
            Record::new(&raw[..raw.len() - 1]).is_err(),
            "one byte short must not parse"
        );
        let mut longer = raw.clone();
        longer.push(0);
        assert!(
            Record::new(&longer).is_err(),
            "trailing slack means the boundary was wrong"
        );
    }

    #[test]
    fn validate_catches_fields_disagreeing_with_l_shared() {
        let mut raw = simple();
        // Claim one more allele than the record encodes. The framing still
        // adds up, so only the chain walk notices.
        raw[8 + 18..8 + 20].copy_from_slice(&3u16.to_le_bytes());
        let record = Record::new(&raw).unwrap();
        assert!(record.validate().is_err());
    }

    #[test]
    fn validate_catches_a_genotype_block_the_fields_do_not_fill() {
        // The mirror of the l_shared check, and it was missing until the CUDA
        // translation asserted it and the reference did not. Iterating the
        // FORMAT fields proves none runs *past* l_indiv; it says nothing about
        // them stopping short and leaving bytes no field claims.
        let mut raw = simple();
        let l_indiv = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        raw[4..8].copy_from_slice(&(l_indiv + 1).to_le_bytes());
        raw.push(0);

        let record = Record::new(&raw).unwrap();
        assert_eq!(
            record.format_count(),
            1,
            "the fields themselves are still well-formed"
        );
        for field in record.formats() {
            field.expect("no field runs past the block");
        }
        assert!(
            record.validate().is_err(),
            "a byte no FORMAT field claims must be caught"
        );
    }

    #[test]
    fn looks_like_a_record_accepts_the_real_start() {
        let raw = simple();
        assert!(looks_like_a_record(&raw, 0, 1, 4));
    }

    #[test]
    fn looks_like_a_record_rejects_wrong_sample_counts_and_contigs() {
        let raw = simple();
        assert!(
            !looks_like_a_record(&raw, 0, 2, 4),
            "n_sample must equal the header's"
        );
        assert!(
            !looks_like_a_record(&raw, 0, 1, 1),
            "CHROM 1 is out of range for a 1-contig header"
        );
    }

    #[test]
    fn looks_like_a_record_rejects_almost_every_wrong_offset() {
        // The property the device scan will depend on: a validator run at every
        // byte offset of a record must find the one real start and few others.
        // It need not be perfect — a survivor is only a hypothesis — but it has
        // to prune hard enough to be worth running.
        let mut buf = Vec::new();
        let mut starts = Vec::new();
        for i in 0..8 {
            starts.push(buf.len());
            buf.extend(build(
                2,
                1000 + i,
                &[b"ACGT", b"A"],
                &[(3, vec![0x11, 0x2A]), (5, vec![0x25, 0, 0, 0x80, 0x3F])],
                &[(4, vec![0x21, 0x02, 0x04])],
                1,
            ));
        }
        let survivors: Vec<usize> = (0..buf.len())
            .filter(|&p| looks_like_a_record(&buf, p, 1, 4))
            .collect();
        for start in &starts {
            assert!(
                survivors.contains(start),
                "the validator must never reject a real record start"
            );
        }
        assert_eq!(
            survivors, starts,
            "no false positive survived on this fixture"
        );
    }
}
