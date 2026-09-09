# Test fixtures

Written by **htslib** (`samtools 1.16.1`), not by this crate. That is the entire
point: every other fixture in the workspace comes from our own `BgzfWriter`, so
a misunderstanding shared by our writer and reader would pass silently. These
are the only files that can catch a systematically wrong reading of the spec.

Committed deliberately — they are small, and regenerating them requires a
samtools install that not every machine has (the Colab image has none).

| File | Contents |
|---|---|
| `htslib.bam` | 8 records, 2 references, one BGZF data block |
| `htslib_multiblock.bam` | 4000 records over 15 BGZF blocks |
| `pacbio_hifi.bam` | 20 real PacBio HiFi reads, 25 BGZF blocks, tag-heavy |

## What no fixture here covers

Recorded because both were assumed to be covered and are not:

- **A record spanning a BGZF block boundary.** htslib calls `bgzf_flush_try`
  before each record, which starts a new block rather than splitting a record
  that would not fit. So an htslib-written BAM only splits records *larger than
  65280 bytes*, and the largest record in any fixture here is 56 KB. That is why
  `htslib_multiblock.bam`'s blocks hold 65190 bytes rather than a full 65280.
  Cross-block reads are covered by `cpu.rs`'s
  `concatenation_is_seamless_across_blocks`, which uses our own writer with a
  small payload size — necessarily, since htslib will not produce the case.
- **The `CG` long-CIGAR overflow.** It needs more than 65535 CIGAR operations in
  one record; HiFi's accuracy caps `pacbio_hifi.bam` at 790. Covered by
  hand-built records in `record.rs`. **Ultra-long ONT data would exercise both
  of these**, and is the fixture still worth adding.

## `pacbio_hifi.bam`

The long-read fixture. Aux tags are a trailing detail in aligned Illumina data
and the *payload* in long-read data, so this is the only file here that tests
aux decoding against anything realistic.

Contents: 20 reads of 11.0–31.0 kbp, 86 references (GRCh37), 1.14 MB
uncompressed over 25 BGZF blocks, and 25–29 aux tags per record. Between them
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
