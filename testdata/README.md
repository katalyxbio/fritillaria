# Test fixtures

Written by **htslib** (`samtools 1.16.1`, `bcftools 1.16`), not by this crate.
That is the entire point: every other fixture in the workspace comes from our
own `BgzfWriter`, so a misunderstanding shared by our writer and reader would
pass silently. These are the only files that can catch a systematically wrong
reading of the spec.

Committed deliberately — they are small, and regenerating them requires an
htslib install that not every machine has (the Colab image has none).

| File | Contents |
|---|---|
| `htslib.bam` | 8 records, 2 references, one BGZF data block |
| `htslib_multiblock.bam` | 4000 records over 15 BGZF blocks |
| `pacbio_hifi.bam` | 20 real PacBio HiFi reads, 14 BGZF blocks, tag-heavy |
| `ont_ultralong.bam` | 4 real ONT ultra-long reads; a 254 KB record spanning 4 blocks |
| `kg_phase3.bcf` | 715 real 1000 Genomes records, 2504 samples, 58 blocks |
| `giab_hg002.bcf` | 275 real GIAB HG002 records, 1 sample, rich INFO and FORMAT |
| `giab_hg002_idx_gap.bcf` | the same, with one INFO tag removed so `IDX` has a gap |

## What no fixture here covers

**The `CG` long-CIGAR overflow**, and only that one. It needs more than 65535
CIGAR operations in a single record, so the real CIGAR moves into a `CG:B:I`
tag. `record.rs` covers it with hand-built records; nothing here covers it with
real data.

Not for want of looking. `ont_ultralong.bam`'s source was aligned by
`minimap2 -L`, which is exactly the flag that emits `CG`, so such records should
exist somewhere in that 187 GB file. A targeted search did not find one:

| searched | records | max `n_cigar_op` |
|---|---|---|
| nine region queries across seven chromosomes | ~12,200 | 39,113 |
| BAI bins at 2^20 / 2^26 / 2^23 / 2^29 | ~15,900 | 46,943 |

~28,000 records and ~370 MB, and the closest was **46,943 ops — 72% of the
threshold**. The search used the index rather than sampling blindly: BAI puts a
read in the smallest bin containing its reference span, so the high bins are
enriched for long reads. Note `samtools view` reconstructs the CIGAR from `CG`
and *drops the tag*, so the encoding is invisible from SAM text; detecting it
means reading the raw BAM for `n_cigar_op == 2` plus a `CG` tag.

The longest read seen anywhere was 444,940 bp. Ops per base varies from 0.08 to
0.21, so overflow needs roughly a 300–800 kb read — rare enough that a bounded
search missed it, not rare enough to call absent.

### Previously listed here and now fixed

**A record spanning a BGZF block boundary.** This was listed as covered long
before it was. htslib calls `bgzf_flush_try` before each record and starts a new
block rather than splitting a record that would not fit, so an htslib-written
BAM only splits records *larger than 65280 bytes* — and Illumina records are
~300 bytes while HiFi tops out at 56 KB. That is why `htslib_multiblock.bam`'s
blocks hold 65190 bytes rather than a full 65280, and why, despite its name, no
record in it spans anything. `ont_ultralong.bam` fixes this for real.

## `pacbio_hifi.bam`

The long-read fixture. Aux tags are a trailing detail in aligned Illumina data
and the *payload* in long-read data, so this is the only file here that tests
aux decoding against anything realistic.

Contents: 20 reads of 11.0–31.0 kbp, 86 references (GRCh37), 0.60 MB
uncompressed over 14 BGZF blocks, and 25–29 aux tags per record. Between them
they exercise the
binary scalar types `C`, `I`, `S`, `f`, `Z` and the `B` subtypes `C`, `S`, `i`,
`f` — including `MM`/`ML` base modifications and per-base kinetics arrays.

The types this file does *not* contain (`A`, `H`, `c`, `s`, `i` scalars and
`B:c`, `B:s`, `B:I`) are covered synthetically in `aux.rs`. Note this is not a
gap in the fixture so much as a property of htslib: it writes the **narrowest
integer type that fits**, so `c`/`s`/`i` appear only when a value is negative or
large, and `samtools view` prints every integer width as `i` regardless.

### Regenerating

Source: [GIAB](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_HiFi-Revio_20231031/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh37.bam)
— HG002/NA24385, PacBio HiFi Revio, 2023-10-31, 48x, aligned to GRCh37. Public
NIST reference data. The source is 77 GB, so it is streamed rather than
downloaded: `samtools` reads it over HTTPS and only the leading blocks are
fetched.

```bash
URL=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/\
HG002_NA24385_son/PacBio_HiFi-Revio_20231031/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh37.bam
# 101 header lines + the first 20 records.
samtools view -h "$URL" | head -n 121 > /tmp/pacbio.sam
samtools view -b /tmp/pacbio.sam -o testdata/pacbio_hifi.bam
```

`md5sum` of the committed file is `aee35dc0f57348e5b977caeb06ba566d`, but see
the note below: `samtools view -b` stamps its own `@PG` line, so the bytes are
not reproducible across samtools versions and the tests assert on content.

## `htslib.bam`

Chosen to exercise the cases that break naive parsers, all in one file:

- `r001` — insertion and deletion in the CIGAR, mate fields set
- `r002` — soft clip and pad ops
- `r003` — **hard clip** (`5H6M`), an op consuming no sequence
- `r004` — skip op (`N`), and an **odd-length sequence** (11 bases → 6 bytes with
  a padding nibble)
- `r005` — **absent qualities** (`*` in SAM, stored as `0xFF` repeated)
- `r006` — reverse strand, negative template length
- `r007` — **unmapped**: no reference, position `-1`, empty CIGAR
- `r008` — `N` base (nibble 15) and a **`B` array aux tag**

## Regenerating

```bash
samtools view -b -o testdata/htslib.bam <source.sam>
```

The source SAM for `htslib.bam` is reproduced below. `htslib_multiblock.bam` is
4000 records of 100 bases at positions 1, 101, 201, … on a single `chr1` of
length 5000000, generated with a seeded RNG so it is reproducible.

Note `samtools view -b` records its own `@PG` line in the header, so the output
is not byte-reproducible across samtools versions. Tests assert on parsed
content, never on file bytes.

```
@HD	VN:1.6	SO:coordinate
@SQ	SN:chr1	LN:1000
@SQ	SN:chr2	LN:2000
@RG	ID:rg1	SM:sample1
@PG	ID:test	PN:test	VN:1.0
r001	99	chr1	7	60	8M2I4M1D3M	=	37	39	TTAGATAAAGGATACTG	IIIIIIIIIIIIIIIII	RG:Z:rg1	NM:i:1
r002	0	chr1	9	30	3S6M1P1I4M	*	0	0	AAAAGATAAGGATA	IIIIIIIIIIIIII	NM:i:0
r003	0	chr1	9	30	5H6M	*	0	0	AGCTAA	IIIIII	XA:A:c	XI:i:-5
r004	0	chr1	16	30	6M14N5M	*	0	0	ATAGCTTCAGC	IIIIIIIIIII	XF:f:1.5
r005	0	chr1	29	30	6M	*	0	0	TAGGCT	*
r006	83	chr1	37	60	9M	=	7	-39	CAGCGGCAT	IIIIIIIII	NM:i:1
r007	4	*	0	0	*	*	0	0	AACCGGTTA	IIIIIIIII
r008	0	chr2	1	40	5M	*	0	0	ACGTN	IIIII	XB:B:i,1,2,3
```

## `ont_ultralong.bam`

The cross-block fixture, and the only file here whose records are larger than a
BGZF block. Four reads from GIAB HG002 ONT ultra-long, 195 references (GRCh38),
0.40 MB uncompressed over 8 BGZF blocks:

| record | bytes | `l_seq` | CIGAR ops | block boundaries crossed |
|---|---|---|---|---|
| 0 | 254,293 | 162,932 | 2,251 | **3** |
| 1 | 113,876 | 73,028 | 967 | 1 |
| 2 | 4,277 | **0** | 1,037 | 0 |
| 3 | 9,535 | 5,556 | 267 | 0 |

Record 2 is a secondary alignment with no stored sequence and no qualities but a
real 1037-op CIGAR — every variable-length offset in a record is derived from
`l_seq`, so a zero there is the case most likely to expose offset arithmetic
that is subtly wrong.

Aux tags are sparser than PacBio's but add one type real HiFi data does not
contain: `tp:A`, so `A` is now covered by a real file rather than only by
`htslib.bam`'s hand-written `XA:A:c`. Binary types present: `A`, `C`, `S`, `f`,
`Z`. No `B` arrays — `pacbio_hifi.bam` covers those.

### Regenerating

Source: [GIAB](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh38_ONT-UL_UCSC_20200508.phased.bam)
— HG002/NA24385, UCSC ultra-long PromethION, 2020-05-08, phased, aligned to
GRCh38 with `minimap2 -L`. Public NIST reference data. The source is 187 GB and
its `.bai` is 54 MB; both are streamed, and the region below was chosen because
it contains the 162,932 bp read.

```bash
URL=.../HG002_GRCh38_ONT-UL_UCSC_20200508.phased.bam
curl -s -o ont.bai "$URL.bai"
samtools view -H "$URL" > hdr.sam
samtools view -X "$URL" ont.bai chr18:16636733-16636734 > region.sam
# Longest read, one mid-size, the sequence-less secondary, one ordinary read.
# See testdata/README.md history for the exact selection.
cat hdr.sam picks.sam > fixture.sam
samtools view -b fixture.sam -o testdata/ont_ultralong.bam
```

`md5sum` is `82c2240869cff15b09caab4e8b71a248`; as with the others, `samtools
view -b` stamps its own `@PG` so the bytes are not reproducible across samtools
versions and the tests assert on content.

## The BCF fixtures

Written by `bcftools 1.16` from real public callsets. Between them they cover
the BCF2 encoding; `crates/fritillaria-bcf/tests/bcftools.rs` asserts the
relevant property of the fixture *before* relying on it, so a file regenerated
from different data fails loudly instead of quietly testing nothing.

### The block structure is the interesting part

BCF blocks are packed **full** — 65280-byte payloads, right up to the cap —
because `bcf_write` goes straight to `bgzf_write` with none of the
`bgzf_flush_try` that makes htslib start a fresh BGZF block per BAM alignment.
Measured on these files:

| File | interior block starts | that are also record starts |
|---|---|---|
| `kg_phase3.bcf` | 56 | **0** |
| `giab_hg002.bcf` | 3 | **0** |

So a BCF record straddles essentially every interior block boundary, where a
BAM record straddles almost none. That inverts the assumption
`fritillaria-bam`'s device scan is built on, and it is why
`fritillaria-bcf::columnar::looks_like_a_record` exists. See `docs/bcf-boundaries.md`.

### `kg_phase3.bcf`

715 records, 2504 samples, 58 BGZF blocks, 3.7 MB inflated from 182 KB — real
WGS genotypes compress **20x**, so this is also the first fixture whose ratio is
in the range real data actually has.

Two regions concatenated, because neither covers the encoding alone:

- **chr22:16,050,000-16,065,000** — a CNV record with four symbolic alleles and
  an `END` past 16 million, which is the only `int32` INFO value in either
  fixture; a 17-base REF and a 16-base ALT, both past the 15-element escape.
- **X:20,000,000-20,010,000** — non-PAR, so haploid males sit beside diploid
  females and **214 records** carry `END_OF_VECTOR` padding in `GT`. Nothing
  else here produces that, and reading the pad as data silently turns a haploid
  call diploid.

INFO covers `int8`, `int16`, `int32`, `float`, `char` and Flag, and vectors with
`Number=A`, `Number=2` and `Number=.`.

```bash
K=https://1000genomes.s3.amazonaws.com/release/20130502
bcftools view -r 22:16050000-16065000 \
  "$K/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" \
  -Ob -o p22.bcf && bcftools index p22.bcf
bcftools view -r X:20000000-20010000 \
  "$K/ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz" \
  -Ob -o pX.bcf && bcftools index pX.bcf
bcftools concat -a p22.bcf pX.bcf -Ob -o testdata/kg_phase3.bcf
```

The AWS mirror, not EBI's FTP: identical bytes and the same public dataset, but
EBI served this at 0.2 MiB/s on 2026-09-09. `md5sum` is
`1c24f38edb1e3364f65eb9ded5301816`.

### `giab_hg002.bcf`

275 records, one sample, 5 BGZF blocks. Complements the panel fixture at the
other extreme — narrow genotypes, very rich sites:

- INFO strings running to **196 characters**, so the 15-element vector-length
  escape fires **1407 times**. Taking the count nibble at face value reads 15
  bytes and resynchronises onto the middle of a string.
- `FORMAT` with six keys including `Number=R` vectors (`AD`, `ADALL`), so a
  wrong per-sample stride shifts every value by a constant and still looks like
  plausible depths.
- `PS` missing on every record, written as an `int8` `0x80`. That is the MISSING
  sentinel *at 8 bits*; classify it after widening to `i32` and it reads as an
  ordinary -128. Nothing else in either fixture catches that.
- A ten-entry `FILTER` dictionary.

```bash
G=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio
bcftools view -r chr20:1000000-1200000 \
  "$G/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
  -Ob -o testdata/giab_hg002.bcf
```

`md5sum` is `93e114d6ca6688ad01906a139315cd6e`.

### `giab_hg002_idx_gap.bcf`

The same file with one INFO tag removed. It exists because **without it the
`IDX` handling is untested**, and that is not obvious: bcftools writes `IDX=` on
every dictionary line, but on a freshly converted file those numbers equal the
declaration order — so a reader that ignores `IDX` entirely passes every test
built on the other two fixtures. Verified: that exact mutation was green until
this file existed.

Removing a tag is the situation `IDX` was added to the spec for. The numbers
survive the deletion, so `platforms` (11) disappears and `platformnames` keeps
12, leaving a gap and an order that no longer matches position.

```bash
bcftools annotate -x INFO/platforms testdata/giab_hg002.bcf \
  -Ob -o testdata/giab_hg002_idx_gap.bcf
```

`md5sum` is `ede01a789589b91a317d153ab2ab0dd8`.

### What no BCF fixture here covers

- **Float `END_OF_VECTOR` and float MISSING in a FORMAT field.** Both are
  covered synthetically in `typed.rs`; no real file here has a float FORMAT
  vector of varying length.
- **A sites-only BCF** (no genotype block at all, `l_indiv == 0`). Covered by
  unit tests only.
- **`IDX` in a non-ascending order.** The gap fixture has ascending numbers with
  a hole; nothing produces genuinely permuted ones.

## FASTQ fixtures

Generated from the BAM fixtures already here, so the reads are real and the
container is htslib's rather than our own writer's — the circularity rule in
CLAUDE.md applies to FASTQ too.

```bash
samtools fastq -0 testdata/ont_ultralong.fastq.gz  testdata/ont_ultralong.bam
samtools fastq -0 testdata/pacbio_hifi.fastq.gz    testdata/pacbio_hifi.bam
samtools fastq -0 testdata/illumina.fastq.gz       testdata/htslib_multiblock.bam
```

`-0` and not `-o`: `-o` catches reads flagged READ1/READ2, and these are
unpaired, so with `-o` they go to stdout and the file gets an empty BGZF stream.
That looks like success — a valid 28-byte file — which is exactly the sort of
quiet failure this directory's README exists to prevent.

| File | Reads | Records | Sequence length | BGZF blocks |
|---|---|---|---|---|
| `ont_ultralong.fastq.gz` | ONT ultra-long | 3 | 5,556–162,932 | 9 |
| `pacbio_hifi.fastq.gz` | PacBio HiFi | 8 | 13,661–20,499 | — |
| `illumina.fastq.gz` | Illumina | 4,000 | 100 | — |
| `pacbio_hifi.plain.fastq.gz` | PacBio HiFi | 8 | 13,661–20,499 | **none — plain gzip** |

**What each one is for.**

- **ONT** is the seam fixture: a single record of 162,932 bases spans many BGZF
  blocks, so the carry-the-partial-record loop is exercised for real rather than
  assumed. Same role `ont_ultralong.bam` plays for BAM.
- **Illumina** is the *adversarial* one, and the reason it is here rather than
  being redundant with the others. **83 of its quality lines begin with `@`**,
  and 13,159 `@` bytes appear inside quality lines — `@` is Phred+33 Q31, an
  ordinary score. Those are precisely the decoys a record-start scan must
  reject. Without this fixture a sweep reporting zero false positives would be
  reporting that the hard case never came up. See `docs/fastq-boundaries.md`.
- **`pacbio_hifi.plain.fastq.gz`** is a single DEFLATE stream, deliberately not
  bgzipped: `1f 8b 08 00` versus BGZF's `1f 8b 08 04`. It is the fixture for the
  path that **cannot** be block-parallel, so the API's refusal to pretend
  otherwise is testable rather than merely documented.

If any of these is regenerated from different reads, check the decoy counts
still hold — a fixture that is supposed to exercise a case should assert that it
does, which is the lesson `htslib_multiblock.bam` taught by not doing it.
