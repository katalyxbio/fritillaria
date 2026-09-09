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
