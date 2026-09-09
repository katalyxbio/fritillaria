//! The BCF header.
//!
//! ```text
//! magic    3        "BCF"
//! major    1        2
//! minor    1        2
//! l_text   4        length of `text`, INCLUDING its terminating NUL
//! text     l_text   the VCF header, NUL-terminated
//! ```
//!
//! Simpler than BAM's, and deliberately so: BCF keeps the whole header as VCF
//! text rather than in a binary structure. That is not a convenience — the text
//! is *load-bearing*, because records encode almost every string as an integer
//! offset into dictionaries that only the text defines:
//!
//! - `CHROM` is an offset into the ordered list of `##contig` lines.
//! - `FILTER`, `INFO` and `FORMAT` keys are offsets into a single shared
//!   dictionary built from the `##FILTER`/`##INFO`/`##FORMAT` lines.
//!
//! So a record is not interpretable without the header, where a BAM record is
//! interpretable without one except for the reference *name*. See
//! [`Dictionary`].
//!
//! # `IDX`, and why the implicit numbering cannot be trusted
//!
//! The dictionary offset of an entry is its position among the header lines
//! that define dictionary entries — implicitly. But removing one tag would
//! then renumber every later tag and force a rewrite of every record, so the
//! spec allows an explicit `IDX=` attribute that pins the number. When `IDX` is
//! present on one entry the spec requires it on all of them.
//!
//! bcftools writes `IDX` routinely, so treating the numbering as implicit is
//! not a rare-input bug, it is a wrong-on-most-real-files bug. Both fixtures in
//! `testdata/` carry `IDX`.

use std::collections::HashMap;

use fritillaria_core::{Error, Result};

use crate::{MAGIC, MAJOR_VERSION};

/// The dictionaries a record's integer offsets index into.
///
/// Two of them, and they are separate namespaces: `CHROM` indexes the contig
/// dictionary, while `FILTER`, `INFO` and `FORMAT` keys all index one shared
/// string dictionary. An `INFO` key and a `FILTER` key may therefore collide on
/// a number and mean different things — resolving one against the wrong
/// dictionary yields a plausible wrong answer rather than an error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dictionary {
    /// Contig names, indexed by the record's `CHROM` field.
    pub contigs: Vec<Vec<u8>>,
    /// FILTER/INFO/FORMAT keys, indexed by the record's key offsets.
    ///
    /// Sparse in principle: `IDX` may leave gaps, so entries that no header
    /// line claims are empty.
    pub strings: Vec<Vec<u8>>,
}

impl Dictionary {
    /// The contig name for a record's `CHROM`, or `None` if out of range.
    #[must_use]
    pub fn contig(&self, index: i32) -> Option<&[u8]> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.contigs.get(i))
            .map(Vec::as_slice)
    }

    /// The FILTER/INFO/FORMAT key for a dictionary offset.
    ///
    /// `None` for an offset out of range *or* for a gap left by `IDX`; both
    /// mean "this file does not define that key", which is the answer a caller
    /// wants either way.
    #[must_use]
    pub fn string(&self, index: i32) -> Option<&[u8]> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.strings.get(i))
            .filter(|entry| !entry.is_empty())
            .map(Vec::as_slice)
    }
}

/// A parsed BCF header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    /// The VCF header text, with the terminating NUL stripped.
    pub text: Vec<u8>,
    /// Sample names, in the column order the record genotype block uses.
    pub samples: Vec<Vec<u8>>,
    /// The contig and string dictionaries records index into.
    pub dictionary: Dictionary,
    /// Byte offset at which the first record begins.
    pub records_start: usize,
}

impl Header {
    /// Number of samples, which every record's `n_sample` must equal.
    ///
    /// The spec requires the equality, which makes this the single strongest
    /// check available for validating a *speculative* record start — see
    /// [`crate::record::looks_like_a_record`].
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Parses the header at the start of a decompressed BCF stream.
pub fn parse_header(buf: &[u8]) -> Result<Header> {
    let malformed = |position: usize, reason: &str| Error::Malformed {
        format: "bcf header",
        position: position as u64,
        reason: reason.to_string(),
    };

    if buf.len() < 9 {
        return Err(malformed(0, "truncated header"));
    }
    if buf[..3] != MAGIC {
        return Err(malformed(0, "bad magic; expected BCF"));
    }
    // The minor version is deliberately not checked. 2.1 and 2.2 differ in the
    // header text they permit, not in the binary framing this crate walks, so
    // refusing 2.1 would reject files we can in fact read.
    if buf[3] != MAJOR_VERSION {
        return Err(malformed(
            3,
            &format!("unsupported major version {}; expected 2", buf[3]),
        ));
    }

    let l_text = buf
        .get(5..9)
        .and_then(|raw| raw.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| malformed(5, "truncated l_text"))? as usize;
    let text_end = 9usize
        .checked_add(l_text)
        .ok_or_else(|| malformed(5, "l_text overflows"))?;
    let raw = buf
        .get(9..text_end)
        .ok_or_else(|| malformed(5, "l_text runs past the end of the buffer"))?;

    // l_text counts the NUL. Some writers pad with more than one, so strip all
    // of them rather than exactly one.
    let text = raw
        .iter()
        .rposition(|&b| b != 0)
        .map_or_else(Vec::new, |end| raw[..=end].to_vec());

    let (samples, dictionary) = parse_text(&text)?;

    Ok(Header {
        text,
        samples,
        dictionary,
        records_start: text_end,
    })
}

/// Splits the VCF header text into sample names and the two dictionaries.
fn parse_text(text: &[u8]) -> Result<(Vec<Vec<u8>>, Dictionary)> {
    let mut contigs = Vec::new();
    // Collected as (index, name) and only laid out once every line is seen,
    // because IDX may assign numbers in any order and leave gaps.
    let mut strings: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut samples = Vec::new();

    // PASS occupies slot 0 whether or not a ##FILTER line declares it: a
    // record may say PASS with nothing in the header listing it, so htslib
    // registers PASS before parsing anything else and the implicit numbering
    // of real entries starts at 1. Seeding it here rather than patching slot 0
    // afterwards is what makes an undeclared PASS not collide with the first
    // declared key.
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::from([(b"PASS".to_vec(), 0)]);
    strings.push((0, b"PASS".to_vec()));
    let mut next_implicit = 1usize;

    for line in text.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix(b"#CHROM") {
            samples = parse_sample_columns(rest);
            continue;
        }

        let Some(body) = structured_body(line) else {
            continue;
        };
        let Some(id) = attribute(body, b"ID") else {
            continue;
        };

        if line.starts_with(b"##contig=") {
            contigs.push(id);
            continue;
        }
        if !(line.starts_with(b"##FILTER=")
            || line.starts_with(b"##INFO=")
            || line.starts_with(b"##FORMAT="))
        {
            continue;
        }

        // One ID declared as both INFO and FORMAT — DP and GT routinely are —
        // is one dictionary entry, not two. Skipping the repeat is not just
        // deduplication: were it to consume a number, every later key would
        // shift by one and resolve to its neighbour.
        if seen.contains_key(&id) {
            continue;
        }

        let index = match attribute(body, b"IDX") {
            Some(raw) => std::str::from_utf8(&raw)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| Error::Malformed {
                    format: "bcf header",
                    position: 0,
                    reason: format!(
                        "unparseable IDX on header line for {}",
                        String::from_utf8_lossy(&id)
                    ),
                })?,
            None => next_implicit,
        };
        next_implicit = next_implicit.max(index + 1);
        seen.insert(id.clone(), index);
        strings.push((index, id));
    }

    let mut table: Vec<Vec<u8>> = Vec::new();
    for (index, id) in strings {
        if index >= table.len() {
            table.resize(index + 1, Vec::new());
        }
        table[index] = id;
    }

    Ok((
        samples,
        Dictionary {
            contigs,
            strings: table,
        },
    ))
}

/// The sample names in a `#CHROM` line, given everything after `#CHROM`.
///
/// The eight fixed columns after CHROM are POS ID REF ALT QUAL FILTER INFO, and
/// FORMAT appears only when there are samples — so a sites-only BCF has seven
/// and no samples, which is why this counts from the end of the fixed set
/// rather than assuming FORMAT is present.
fn parse_sample_columns(rest: &[u8]) -> Vec<Vec<u8>> {
    const FIXED_AFTER_CHROM: usize = 8; // POS ID REF ALT QUAL FILTER INFO FORMAT
    rest.split(|&b| b == b'\t')
        .skip(1) // the empty piece before the first tab
        .skip(FIXED_AFTER_CHROM)
        .map(<[u8]>::to_vec)
        .collect()
}

/// The `<...>` body of a structured header line.
fn structured_body(line: &[u8]) -> Option<&[u8]> {
    let open = line.iter().position(|&b| b == b'<')?;
    let close = line.iter().rposition(|&b| b == b'>')?;
    (close > open).then(|| &line[open + 1..close])
}

/// The value of `key=` in a structured header line body.
///
/// Quote-aware, because `Description="a,b=c"` is normal and splitting on commas
/// without it finds keys inside prose. Values are returned unquoted.
fn attribute(body: &[u8], key: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0;
    while pos < body.len() {
        let start = pos;
        let mut eq = None;
        let mut quoted = false;
        while pos < body.len() {
            match body[pos] {
                b'"' => quoted = !quoted,
                b'=' if !quoted && eq.is_none() => eq = Some(pos),
                b',' if !quoted => break,
                _ => {}
            }
            pos += 1;
        }
        if let Some(eq) = eq {
            let name = trim(&body[start..eq]);
            if name == key {
                let mut value = trim(&body[eq + 1..pos]);
                if value.len() >= 2 && value.first() == Some(&b'"') && value.last() == Some(&b'"') {
                    value = &value[1..value.len() - 1];
                }
                return Some(value.to_vec());
            }
        }
        pos += 1; // step over the comma
    }
    None
}

fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|b| !b.is_ascii_whitespace());
    let end = s.iter().rposition(|b| !b.is_ascii_whitespace());
    match (start, end) {
        (Some(a), Some(b)) => &s[a..=b],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(text: &str) -> Vec<u8> {
        let mut buf = MAGIC.to_vec();
        buf.push(MAJOR_VERSION);
        buf.push(2);
        let mut payload = text.as_bytes().to_vec();
        payload.push(0);
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    const MINIMAL: &str = concat!(
        "##fileformat=VCFv4.3\n",
        "##FILTER=<ID=PASS,Description=\"All filters passed\">\n",
        "##contig=<ID=chr1,length=100>\n",
        "##contig=<ID=chr2,length=200>\n",
        "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1\tNA2",
    );

    #[test]
    fn parses_text_samples_and_dictionaries() {
        let raw = build(MINIMAL);
        let header = parse_header(&raw).unwrap();

        assert!(header.text.starts_with(b"##fileformat=VCFv4.3"));
        assert!(!header.text.ends_with(b"\0"), "the NUL must be stripped");
        assert_eq!(header.samples, vec![b"NA1".to_vec(), b"NA2".to_vec()]);
        assert_eq!(header.sample_count(), 2);
        assert_eq!(header.dictionary.contigs.len(), 2);
        assert_eq!(header.dictionary.contig(0), Some(&b"chr1"[..]));
        assert_eq!(header.dictionary.contig(1), Some(&b"chr2"[..]));
        assert_eq!(header.dictionary.contig(2), None);
        assert_eq!(header.dictionary.contig(-1), None);
        assert_eq!(header.records_start, raw.len());
    }

    #[test]
    fn the_implicit_dictionary_numbers_in_header_order() {
        let header = parse_header(&build(MINIMAL)).unwrap();
        // contig lines take no string-dictionary slot; only FILTER/INFO/FORMAT.
        assert_eq!(header.dictionary.string(0), Some(&b"PASS"[..]));
        assert_eq!(header.dictionary.string(1), Some(&b"DP"[..]));
        assert_eq!(header.dictionary.string(2), Some(&b"GT"[..]));
    }

    #[test]
    fn an_explicit_idx_overrides_the_implicit_numbering() {
        // The case the implicit scheme gets wrong: a tag was deleted, so the
        // numbers have a gap and no longer match header order.
        let text = concat!(
            "##fileformat=VCFv4.3\n",
            "##FILTER=<ID=PASS,Description=\"x\",IDX=0>\n",
            "##contig=<ID=chr1,length=100,IDX=0>\n",
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"y\",IDX=7>\n",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"z\",IDX=3>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1",
        );
        let header = parse_header(&build(text)).unwrap();
        assert_eq!(header.dictionary.string(7), Some(&b"DP"[..]));
        assert_eq!(header.dictionary.string(3), Some(&b"GT"[..]));
        assert_eq!(header.dictionary.string(0), Some(&b"PASS"[..]));
        // The gaps IDX leaves are not entries, and must not resolve to a
        // neighbouring key just because the slot exists.
        assert_eq!(header.dictionary.string(1), None);
        assert_eq!(header.dictionary.string(2), None);
    }

    #[test]
    fn pass_occupies_slot_zero_even_when_undeclared() {
        // Legal: a record may say PASS without any ##FILTER line declaring it.
        let text = concat!(
            "##fileformat=VCFv4.3\n",
            "##contig=<ID=chr1,length=100>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        );
        let header = parse_header(&build(text)).unwrap();
        assert_eq!(header.dictionary.string(0), Some(&b"PASS"[..]));
    }

    #[test]
    fn a_key_declared_as_both_info_and_format_shares_one_slot() {
        let text = concat!(
            "##fileformat=VCFv4.3\n",
            "##FILTER=<ID=PASS,Description=\"x\">\n",
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"y\">\n",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"y\">\n",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"z\">\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1",
        );
        let header = parse_header(&build(text)).unwrap();
        assert_eq!(header.dictionary.string(1), Some(&b"DP"[..]));
        assert_eq!(
            header.dictionary.string(2),
            Some(&b"GT"[..]),
            "the second DP declaration must not consume a slot"
        );
    }

    #[test]
    fn a_sites_only_header_has_no_samples() {
        let text = concat!(
            "##fileformat=VCFv4.3\n",
            "##contig=<ID=chr1,length=100>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        );
        let header = parse_header(&build(text)).unwrap();
        assert!(header.samples.is_empty());
        assert_eq!(header.sample_count(), 0);
    }

    #[test]
    fn commas_inside_a_quoted_description_do_not_split_attributes() {
        let text = concat!(
            "##fileformat=VCFv4.3\n",
            "##FILTER=<ID=PASS,Description=\"a, b, ID=zzz\">\n",
            "##contig=<ID=chr1,length=100>\n",
            "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"x,y\">\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        );
        let header = parse_header(&build(text)).unwrap();
        assert_eq!(header.dictionary.string(0), Some(&b"PASS"[..]));
        assert_eq!(header.dictionary.string(1), Some(&b"AC"[..]));
        assert_eq!(header.dictionary.contig(0), Some(&b"chr1"[..]));
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut raw = build(MINIMAL);
        raw[2] = b'X';
        assert!(parse_header(&raw).is_err());

        let mut raw = build(MINIMAL);
        raw[3] = 3;
        assert!(parse_header(&raw).is_err());
    }

    #[test]
    fn accepts_any_minor_version() {
        // 2.1 differs from 2.2 in permitted header text, not in framing.
        let mut raw = build(MINIMAL);
        raw[4] = 1;
        assert!(parse_header(&raw).is_ok());
    }

    #[test]
    fn rejects_truncation_at_each_stage() {
        let raw = build(MINIMAL);
        for len in 0..raw.len() {
            assert!(
                parse_header(&raw[..len]).is_err(),
                "a header truncated to {len} bytes must not parse"
            );
        }
        assert!(parse_header(&raw).is_ok());
    }

    #[test]
    fn rejects_l_text_running_past_the_buffer() {
        let mut raw = build(MINIMAL);
        raw[5..9].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert!(parse_header(&raw).is_err());
    }
}
