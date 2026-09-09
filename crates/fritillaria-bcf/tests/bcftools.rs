//! Validation against files written by bcftools.
//!
//! # Why this exists
//!
//! A parser tested only on fixtures it can also write proves self-consistency
//! and nothing else. `testdata/kg_phase3.bcf` and `testdata/giab_hg002.bcf`
//! were written by bcftools 1.16 from real public callsets, and the expected
//! values here come from `bcftools query`, not from this crate. That is the
//! only thing in the suite that can catch a systematically wrong reading of the
//! BCF2 spec.
//!
//! # What the fixtures are for
//!
//! Between them they cover the encoding, and each test that depends on a
//! property asserts that property of the fixture first — a filename is not
//! evidence, and a fixture regenerated from different data must fail loudly
//! rather than quietly stop testing anything.
//!
//! - `kg_phase3.bcf` — 1000 Genomes phase 3, 2504 samples. Wide genotype
//!   blocks, phased GT, and chrX records where haploid males force
//!   END_OF_VECTOR padding next to diploid females. INFO carries int8, int16,
//!   int32, float, string and Flag values.
//! - `giab_hg002.bcf` — GIAB HG002 v4.2.1, one sample. Long INFO strings that
//!   force the vector-length escape 1407 times, `Number=R` FORMAT vectors, a
//!   large FILTER dictionary, and missing FORMAT values.
//!
//! Regenerate with the commands in `testdata/README.md`.

// Expected-vs-actual pairs are named after the VCF fields they hold, so
// `want_ac`/`want_an`/`want_af` sit next to each other by design. Renaming them
// to satisfy `similar_names` would lose the correspondence to AC/AN/AF, which
// is the thing making these assertions readable.
#![allow(clippy::similar_names)]

use std::path::PathBuf;
use std::process::Command;

use fritillaria_bcf::{
    Int, Kind, Proof, decode_genotype, header::parse_header, record::Record, scan_records,
    scan_records_speculative,
};
use fritillaria_bgzf::{CpuCodec, discover_blocks};
use fritillaria_core::{BlockCodec, InflateBatch};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

/// Decompresses a whole BCF through the CPU codec.
fn inflate(name: &str) -> InflateBatch {
    let raw = std::fs::read(fixture(name)).expect("fixture missing; see testdata/README.md");
    let spans = discover_blocks(&raw, 0).expect("bcftools BGZF must parse");
    let mut out = InflateBatch::new();
    CpuCodec::new()
        .inflate_batch(&raw, &spans, &mut out)
        .expect("bcftools BGZF must inflate and verify");
    out
}

/// Runs a bcftools subcommand, or `None` if bcftools is unavailable.
fn bcftools(args: &[&str]) -> Option<std::process::Output> {
    match Command::new("bcftools").args(args).output() {
        Ok(output) => Some(output),
        Err(err) => {
            eprintln!("SKIP: bcftools unavailable ({err})");
            None
        }
    }
}

/// `bcftools query -f <format>` over a fixture, as lines.
fn query(name: &str, format: &str) -> Option<Vec<String>> {
    let path = fixture(name);
    let out = bcftools(&["query", "-f", format, path.to_str().unwrap()])?;
    assert!(
        out.status.success(),
        "bcftools query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_owned)
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// The container
// ---------------------------------------------------------------------------

#[test]
fn a_bcf_is_bgzf_and_needs_no_new_container_code() {
    // The claim the whole "container, not format" design rests on: BCF goes
    // through exactly the BGZF path BAM does. If this ever needs a special
    // case, the claim is wrong.
    for name in ["kg_phase3.bcf", "giab_hg002.bcf"] {
        let raw = std::fs::read(fixture(name)).unwrap();
        let spans = discover_blocks(&raw, 0).unwrap();
        assert!(spans.len() >= 2, "{name}: expected data blocks and an EOF");
        assert_eq!(
            spans.last().unwrap().isize,
            0,
            "{name}: last block must be the empty EOF marker"
        );
    }
}

#[test]
fn bcf_records_cross_block_boundaries_where_bam_records_do_not() {
    // The finding that shaped `record::looks_like_a_record`. BAM leans on
    // htslib flushing before each alignment, so a block start is almost always
    // a record start. `bcf_write` packs blocks full instead, so the opposite
    // holds — and a scan that assumed otherwise would resynchronise onto a
    // plausible wrong offset rather than fail.
    for (name, expected_interior) in [("kg_phase3.bcf", 56), ("giab_hg002.bcf", 3)] {
        let raw = std::fs::read(fixture(name)).unwrap();
        let spans = discover_blocks(&raw, 0).unwrap();

        // Uncompressed offset at which each block begins.
        let mut block_starts = Vec::new();
        let mut at = 0usize;
        for span in &spans {
            block_starts.push(at);
            at += span.isize as usize;
        }

        let batch = inflate(name);
        let buf = batch.data();
        let header = parse_header(buf).unwrap();
        let (offsets, tail) = scan_records(buf, header.records_start).unwrap();
        assert_eq!(tail, buf.len(), "{name}: must end on a record boundary");

        // Interior block starts: past the header, and not the trailing empty
        // EOF block (which starts exactly at the end).
        let interior: Vec<usize> = block_starts
            .iter()
            .copied()
            .filter(|&s| s > header.records_start && s < buf.len())
            .collect();
        assert_eq!(
            interior.len(),
            expected_interior,
            "{name}: fixture no longer has the block structure this tests"
        );

        let coincide = interior
            .iter()
            .filter(|s| offsets.binary_search(s).is_ok())
            .count();
        assert_eq!(
            coincide, 0,
            "{name}: {coincide} interior block starts coincide with a record start; \
             if this ever stops being 0 the per-block speculation BAM uses may apply"
        );
    }
}

// ---------------------------------------------------------------------------
// The header
// ---------------------------------------------------------------------------

#[test]
fn parses_a_bcftools_written_header() {
    let batch = inflate("kg_phase3.bcf");
    let header = parse_header(batch.data()).unwrap();

    assert!(header.text.starts_with(b"##fileformat=VCF"));
    assert!(!header.text.ends_with(b"\0"), "the NUL must be stripped");
    assert_eq!(header.sample_count(), 2504, "1000 Genomes phase 3 panel");
    assert_eq!(header.samples[0], b"HG00096");
    assert_eq!(header.dictionary.contigs.len(), 86);

    let batch = inflate("giab_hg002.bcf");
    let header = parse_header(batch.data()).unwrap();
    assert_eq!(header.sample_count(), 1);
    assert_eq!(header.samples[0], b"HG002");
    assert_eq!(header.dictionary.contigs.len(), 195);
}

#[test]
fn sample_names_match_bcftools_query_list_samples() {
    for name in ["kg_phase3.bcf", "giab_hg002.bcf"] {
        let path = fixture(name);
        let Some(out) = bcftools(&["query", "-l", path.to_str().unwrap()]) else {
            return;
        };
        let expected: Vec<&str> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .collect::<Vec<_>>()
            .iter()
            .map(|s| Box::leak(s.to_string().into_boxed_str()) as &str)
            .collect();

        let batch = inflate(name);
        let header = parse_header(batch.data()).unwrap();
        let ours: Vec<String> = header
            .samples
            .iter()
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        assert_eq!(ours.len(), expected.len(), "{name}: sample count");
        assert_eq!(ours, expected, "{name}: sample names and order");
    }
}

#[test]
fn the_dictionary_resolves_keys_bcftools_reports_by_name() {
    // The dictionary is what makes a record interpretable at all: get the
    // numbering wrong and every INFO value attaches to the wrong key, with no
    // error anywhere. Checking it against names bcftools prints is the only
    // way to know the numbering is right rather than merely self-consistent.
    let batch = inflate("kg_phase3.bcf");
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();

    let Some(expected) = query("kg_phase3.bcf", "%POS\t%AC\t%AN\t%AF\n") else {
        return;
    };
    assert_eq!(expected.len(), offsets.len());

    // Resolve the three keys by name, through the dictionary, exactly as a
    // caller would.
    let (ac, an, af) = (
        key(&header, b"AC"),
        key(&header, b"AN"),
        key(&header, b"AF"),
    );

    for (line, &start) in expected.iter().zip(&offsets) {
        let mut fields = line.split('\t');
        let pos: i32 = fields.next().unwrap().parse().unwrap();
        let want_ac = fields.next().unwrap();
        let want_an = fields.next().unwrap();
        let want_af = fields.next().unwrap();

        let end = next_start(buf, start);
        let record = Record::new(&buf[start..end]).unwrap();
        assert_eq!(record.position() + 1, pos, "VCF POS is 1-based");

        // AC is Number=A, so one value per ALT allele.
        let got_ac = record
            .info_get(ac)
            .unwrap()
            .map(|v| {
                v.ints()
                    .unwrap()
                    .filter_map(Int::value)
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        assert_eq!(got_ac, want_ac, "AC at POS {pos}");

        let got_an = record
            .info_get(an)
            .unwrap()
            .and_then(|v| v.as_int())
            .map(|i| i.to_string())
            .unwrap_or_default();
        assert_eq!(got_an, want_an, "AN at POS {pos}");

        // AF is a float; compare numerically, since bcftools formats it.
        let got_af: Vec<f32> = record
            .info_get(af)
            .unwrap()
            .map(|v| {
                v.floats()
                    .unwrap()
                    .filter_map(fritillaria_bcf::Float::value)
                    .collect()
            })
            .unwrap_or_default();
        let want: Vec<f32> = want_af.split(',').filter_map(|s| s.parse().ok()).collect();
        assert_eq!(got_af.len(), want.len(), "AF arity at POS {pos}");
        for (g, w) in got_af.iter().zip(&want) {
            assert!(
                (g - w).abs() <= w.abs() * 1e-5 + 1e-7,
                "AF at POS {pos}: got {g}, bcftools says {w}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// The dictionary offset of a FILTER/INFO/FORMAT key, by name.
///
/// Resolving by name is the point: a test that hardcoded the number would pass
/// against a dictionary numbered wrongly, which is exactly the failure the
/// fixtures exist to catch.
fn key(header: &fritillaria_bcf::Header, name: &[u8]) -> i32 {
    let index = header
        .dictionary
        .strings
        .iter()
        .position(|entry| entry == name)
        .unwrap_or_else(|| panic!("{} not in the dictionary", String::from_utf8_lossy(name)));
    i32::try_from(index).expect("dictionary offsets are small")
}

/// Where the record beginning at `start` ends.
fn next_start(buf: &[u8], start: usize) -> usize {
    let l_shared = u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()) as usize;
    let l_indiv = u32::from_le_bytes(buf[start + 4..start + 8].try_into().unwrap()) as usize;
    start + 8 + l_shared + l_indiv
}

#[test]
fn every_record_validates_against_its_own_declared_lengths() {
    // The strongest self-check available without bcftools: walk every typed
    // value of every record and require the fields to end exactly where
    // l_shared and l_indiv say they do. A misread descriptor cannot survive
    // this, because it desynchronises the walk.
    for (name, expected) in [("kg_phase3.bcf", 715), ("giab_hg002.bcf", 275)] {
        let batch = inflate(name);
        let buf = batch.data();
        let header = parse_header(buf).unwrap();
        let (offsets, tail) = scan_records(buf, header.records_start).unwrap();
        assert_eq!(offsets.len(), expected, "{name}: record count");
        assert_eq!(tail, buf.len(), "{name}: no partial trailing record");

        for &start in &offsets {
            let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
            assert_eq!(
                record.sample_count(),
                header.sample_count(),
                "{name}: n_sample must equal the header's"
            );
            record
                .validate()
                .unwrap_or_else(|e| panic!("{name}: record at {start} failed to validate: {e}"));
        }
    }
}

#[test]
fn sites_match_bcftools_query() {
    for name in ["kg_phase3.bcf", "giab_hg002.bcf"] {
        let Some(expected) = query(name, "%CHROM\t%POS\t%ID\t%REF\t%ALT\t%QUAL\t%FILTER\n") else {
            return;
        };

        let batch = inflate(name);
        let buf = batch.data();
        let header = parse_header(buf).unwrap();
        let (offsets, _) = scan_records(buf, header.records_start).unwrap();
        assert_eq!(expected.len(), offsets.len(), "{name}: record count");

        for (line, &start) in expected.iter().zip(&offsets) {
            let mut want = line.split('\t');
            let (chrom, pos, id, reference, alt, qual, filter) = (
                want.next().unwrap(),
                want.next().unwrap(),
                want.next().unwrap(),
                want.next().unwrap(),
                want.next().unwrap(),
                want.next().unwrap(),
                want.next().unwrap(),
            );

            let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
            let context = format!("{name} at {chrom}:{pos}");

            let got_chrom = header
                .dictionary
                .contig(record.chromosome_id())
                .expect("CHROM must resolve");
            assert_eq!(
                String::from_utf8_lossy(got_chrom),
                chrom,
                "{context}: CHROM"
            );
            assert_eq!(
                (record.position() + 1).to_string(),
                pos,
                "{context}: POS is 0-based in BCF and 1-based in VCF"
            );

            let got_id = record.id().unwrap().map_or_else(
                || ".".to_string(),
                |v| String::from_utf8_lossy(v).into_owned(),
            );
            assert_eq!(got_id, id, "{context}: ID");

            let alleles = record.alleles().unwrap();
            assert_eq!(
                String::from_utf8_lossy(alleles[0]),
                reference,
                "{context}: REF"
            );
            let got_alt = alleles[1..]
                .iter()
                .map(|a| String::from_utf8_lossy(a).into_owned())
                .collect::<Vec<_>>()
                .join(",");
            let got_alt = if got_alt.is_empty() {
                ".".to_string()
            } else {
                got_alt
            };
            assert_eq!(got_alt, alt, "{context}: ALT");

            let got_qual = record
                .quality()
                .map_or_else(|| ".".to_string(), |q| format!("{q}"));
            if qual == "." {
                assert_eq!(got_qual, ".", "{context}: QUAL missing");
            } else {
                let want: f32 = qual.parse().unwrap();
                let got = record
                    .quality()
                    .unwrap_or_else(|| panic!("{context}: QUAL"));
                assert!(
                    (got - want).abs() <= 1e-3,
                    "{context}: QUAL {got} vs {want}"
                );
            }

            let got_filter = record
                .filters()
                .unwrap()
                .iter()
                .map(|&f| {
                    String::from_utf8_lossy(
                        header.dictionary.string(f).expect("FILTER must resolve"),
                    )
                    .into_owned()
                })
                .collect::<Vec<_>>()
                .join(";");
            let got_filter = if got_filter.is_empty() {
                ".".to_string()
            } else {
                got_filter
            };
            assert_eq!(got_filter, filter, "{context}: FILTER");
        }
    }
}

/// Renders one sample's genotype the way `bcftools query %GT` does.
fn render_genotype(alleles: &[fritillaria_bcf::Allele]) -> String {
    let mut out = String::new();
    for (i, allele) in alleles.iter().enumerate() {
        if i > 0 {
            out.push(if allele.phased { '|' } else { '/' });
        }
        match allele.index {
            Some(index) => out.push_str(&index.to_string()),
            None => out.push('.'),
        }
    }
    out
}

#[test]
fn genotypes_match_bcftools_including_phase() {
    // GT is where a wrong reading is most likely to look right: the encoding
    // is (allele + 1) << 1 | phased, so an off-by-one in either direction
    // still produces valid-looking genotypes.
    let name = "giab_hg002.bcf";
    let Some(expected) = query(name, "[%GT]\n") else {
        return;
    };

    let batch = inflate(name);
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let gt = key(&header, b"GT");
    assert_eq!(expected.len(), offsets.len());

    for (want, &start) in expected.iter().zip(&offsets) {
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
        let field = record.format_get(gt).unwrap().expect("GT must be present");
        let alleles = decode_genotype(&field.sample(0).unwrap()).unwrap();
        assert_eq!(
            &render_genotype(&alleles),
            want,
            "GT at record starting {start}"
        );
    }
}

#[test]
fn every_genotype_of_every_sample_matches_bcftools() {
    // 1.8M genotypes across 2504 samples, which is the only way to test the
    // parts a single-sample fixture cannot reach: the per-sample stride, and
    // END_OF_VECTOR padding on real haploid chrX calls. Treating a pad as data
    // there yields a second allele bcftools does not report, and treating the
    // stride as anything but count*size shifts every sample by a constant.
    let name = "kg_phase3.bcf";
    let Some(expected) = query(name, "[%GT\t]\n") else {
        return;
    };

    let batch = inflate(name);
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let gt = key(&header, b"GT");
    assert_eq!(expected.len(), offsets.len());

    let mut compared = 0u64;
    let mut haploid = 0u64;
    for (line, &start) in expected.iter().zip(&offsets) {
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
        let field = record.format_get(gt).unwrap().expect("GT must be present");
        assert_eq!(field.samples, header.sample_count());

        let mut want = line.split('\t');
        for sample in 0..field.samples {
            let want = want.next().expect("bcftools must print every sample");
            let alleles = decode_genotype(&field.sample(sample).unwrap()).unwrap();
            if alleles.len() == 1 {
                haploid += 1;
            }
            assert_eq!(
                &render_genotype(&alleles),
                want,
                "record at {start}, sample {sample}"
            );
            compared += 1;
        }
    }

    assert_eq!(compared, 715 * 2504, "every genotype must be compared");
    assert!(
        haploid > 1000,
        "the fixture must still hold haploid chrX calls; saw {haploid}"
    );
}

#[test]
fn haploid_calls_are_padded_with_end_of_vector_and_decode_as_haploid() {
    // The fixture's chrX half exists for this: haploid males stored alongside
    // diploid females, so the shorter genotype is padded. Reading the padding
    // as data turns a haploid call into a diploid one with a bogus allele —
    // wrong, plausible, and silent.
    let name = "kg_phase3.bcf";
    let batch = inflate(name);
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let gt = key(&header, b"GT");

    let mut padded_records = 0;
    let mut haploid_samples = 0;
    for &start in &offsets {
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
        let field = record.format_get(gt).unwrap().expect("GT must be present");
        if field.count < 2 {
            continue; // no room for padding at this site
        }
        let mut padded_here = false;
        for sample in 0..field.samples {
            let slot = field.sample(sample).unwrap();
            let raw: Vec<Int> = slot.ints().unwrap().collect();
            if raw.contains(&Int::EndOfVector) {
                padded_here = true;
                haploid_samples += 1;
                let decoded = decode_genotype(&slot).unwrap();
                assert!(
                    decoded.len() < raw.len(),
                    "decoding must stop at the pad, not count it as an allele"
                );
            }
        }
        if padded_here {
            padded_records += 1;
        }
    }

    assert_eq!(
        padded_records, 214,
        "the fixture must still contain the haploid chrX records this tests"
    );
    assert!(
        haploid_samples > 1000,
        "expected many haploid samples, saw {haploid_samples}"
    );
}

#[test]
fn format_vectors_match_bcftools() {
    // AD is Number=R, so one value per allele, and bcftools prints it
    // comma-joined. A wrong stride would shift every sample's values by a
    // constant and still produce plausible depths.
    //
    // PS is in here for a second reason: it is missing on every record, and
    // bcftools writes that as an int8 `0x80`. That is the MISSING sentinel *at
    // 8 bits* — classify it after widening to i32 and it becomes an ordinary
    // -128, so the field reads as a real phase set. Nothing else in either
    // fixture catches that.
    let name = "giab_hg002.bcf";
    let Some(expected) = query(name, "[%AD\t%DP\t%GQ\t%PS]\n") else {
        return;
    };

    let batch = inflate(name);
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let (ad, dp, gq, ps) = (
        key(&header, b"AD"),
        key(&header, b"DP"),
        key(&header, b"GQ"),
        key(&header, b"PS"),
    );

    let mut sentinels = 0;
    for (line, &start) in expected.iter().zip(&offsets) {
        let mut want = line.split('\t');
        let (want_ad, want_dp, want_gq, want_ps) = (
            want.next().unwrap(),
            want.next().unwrap(),
            want.next().unwrap(),
            want.next().unwrap(),
        );
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();

        let joined = |key: i32| -> String {
            let Some(field) = record.format_get(key).unwrap() else {
                return ".".to_string();
            };
            let slot = field.sample(0).unwrap();
            let values: Vec<String> = slot
                .ints()
                .unwrap()
                .take_while(|v| *v != Int::EndOfVector)
                .map(|v| match v {
                    Int::Value(n) => n.to_string(),
                    _ => ".".to_string(),
                })
                .collect();
            if values.is_empty() {
                ".".to_string()
            } else {
                values.join(",")
            }
        };

        assert_eq!(joined(ad), want_ad, "AD at record {start}");
        assert_eq!(joined(dp), want_dp, "DP at record {start}");
        assert_eq!(joined(gq), want_gq, "GQ at record {start}");
        assert_eq!(joined(ps), want_ps, "PS at record {start}");

        // Confirm the fixture really is exercising the 8-bit sentinel rather
        // than encoding PS absent some other way.
        let field = record.format_get(ps).unwrap().expect("PS must be present");
        if field.kind == Kind::Int8
            && field
                .sample(0)
                .unwrap()
                .ints()
                .unwrap()
                .any(|v| v == Int::Missing)
        {
            sentinels += 1;
        }
    }
    assert_eq!(
        sentinels,
        offsets.len(),
        "every record's PS should be an int8 MISSING sentinel; the fixture changed"
    );
}

#[test]
fn info_strings_match_bcftools_across_the_length_escape() {
    // giab's callsetnames run to 266 characters, so nearly every record
    // crosses the 15-element escape. Taking the count nibble at face value
    // would read 15 bytes and resynchronise onto the middle of a string.
    let name = "giab_hg002.bcf";
    let Some(expected) = query(name, "%INFO/callsetnames\n") else {
        return;
    };

    let batch = inflate(name);
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let callsetnames = key(&header, b"callsetnames");

    let mut escaped = 0;
    let mut longest = 0;
    for (want, &start) in expected.iter().zip(&offsets) {
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
        let got = record.info_get(callsetnames).unwrap().map_or_else(
            || ".".to_string(),
            |v| String::from_utf8_lossy(v.as_str().unwrap()).into_owned(),
        );
        assert_eq!(&got, want, "callsetnames at record {start}");
        if got.len() >= 15 {
            escaped += 1;
        }
        longest = longest.max(got.len());
    }
    assert!(
        escaped > 200,
        "the fixture must still exercise the length escape; only {escaped} records did"
    );
    assert_eq!(
        longest, 196,
        "the fixture's longest callsetnames value changed; the escape is only \
         meaningfully exercised while these run to hundreds of bytes"
    );
}

#[test]
fn info_values_use_every_integer_width_and_floats() {
    // htslib writes the narrowest type that fits, so the width distribution is
    // a property of the data, not a choice. A reader that handled only int8
    // and int32 would pass every test built on one fixture.
    let batch = inflate("kg_phase3.bcf");
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();

    let mut widths = std::collections::HashSet::new();
    for &start in &offsets {
        let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
        for field in record.info().unwrap() {
            let (_, value) = field.unwrap();
            widths.insert(value.kind());
        }
    }
    for expected in [
        Kind::Int8,
        Kind::Int16,
        Kind::Int32,
        Kind::Float,
        Kind::Character,
    ] {
        assert!(
            widths.contains(&expected),
            "the fixture no longer exercises {expected:?}; it covers {widths:?}"
        );
    }
}

#[test]
fn a_flag_info_field_is_present_with_no_value() {
    // Flags are the odd one out: present means an int8 1, absent means the key
    // does not appear at all. There is no false.
    let batch = inflate("kg_phase3.bcf");
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let (offsets, _) = scan_records(buf, header.records_start).unwrap();
    let flag = key(&header, b"MULTI_ALLELIC");

    let present = offsets
        .iter()
        .filter(|&&start| {
            let record = Record::new(&buf[start..next_start(buf, start)]).unwrap();
            record.info_get(flag).unwrap().is_some()
        })
        .count();
    assert!(present > 0, "no record carries the flag");
    assert!(present < offsets.len(), "every record carries the flag");
}

// ---------------------------------------------------------------------------
// The speculative scan
// ---------------------------------------------------------------------------

#[test]
fn the_speculative_scan_agrees_with_the_serial_one_on_real_files() {
    // The device-side design, exercised on the host against real bcftools
    // output. Two separate claims, and the second is the interesting one:
    // the boundaries must be identical, and they must have been established
    // by the parallel tiling proof rather than by falling back to the walk.
    // A silent fallback would be correct and would perform exactly like the
    // serial design it replaces, so it has to be asserted, not assumed.
    for name in ["kg_phase3.bcf", "giab_hg002.bcf", "giab_hg002_idx_gap.bcf"] {
        let batch = inflate(name);
        let buf = batch.data();
        let header = parse_header(buf).unwrap();
        let samples = u32::try_from(header.sample_count()).unwrap();
        let contigs = u32::try_from(header.dictionary.contigs.len()).unwrap();

        let (expected, expected_tail) = scan_records(buf, header.records_start).unwrap();
        let scan = scan_records_speculative(buf, header.records_start, samples, contigs).unwrap();

        assert_eq!(scan.offsets, expected, "{name}: boundaries");
        assert_eq!(scan.tail, expected_tail, "{name}: tail");
        assert_eq!(
            scan.proof,
            Proof::Tiled,
            "{name}: the tiling proof must succeed, or the parallel design buys nothing"
        );
        assert_eq!(
            scan.validated,
            expected.len(),
            "{name}: {} offsets survived validation against {} real records — \
             any excess is a false positive and the number to watch",
            scan.validated,
            expected.len()
        );
    }
}

#[test]
fn the_speculative_scan_handles_a_partial_record_at_every_truncation() {
    // A BCF batch almost always ends mid-record, because bcftools packs BGZF
    // blocks full. So the partial-tail path is the common case, not an edge
    // case, and it has to keep tiling rather than fall back once per batch.
    let batch = inflate("giab_hg002.bcf");
    let buf = batch.data();
    let header = parse_header(buf).unwrap();
    let samples = u32::try_from(header.sample_count()).unwrap();
    let contigs = u32::try_from(header.dictionary.contigs.len()).unwrap();

    let (all, _) = scan_records(buf, header.records_start).unwrap();

    // Cut inside each of the first 40 records, at three points apiece.
    let mut cuts = 0;
    for window in all.windows(2).take(40) {
        let (start, end) = (window[0], window[1]);
        for cut in [start + 1, start + (end - start) / 2, end - 1] {
            let truncated = &buf[..cut];
            let scan = scan_records_speculative(truncated, header.records_start, samples, contigs)
                .unwrap();
            let (expected, expected_tail) = scan_records(truncated, header.records_start).unwrap();

            assert_eq!(scan.offsets, expected, "cut at {cut}");
            assert_eq!(scan.tail, expected_tail, "cut at {cut}");
            assert_eq!(
                scan.tail, start,
                "cut at {cut}: the tail must point at the record the cut lands in"
            );
            assert_eq!(scan.proof, Proof::Tiled, "cut at {cut}");
            cuts += 1;
        }
    }
    assert_eq!(cuts, 120, "every truncation must have been exercised");
}
