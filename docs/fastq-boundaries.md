# FASTQ record boundaries on device

Measured 2026-09-09, before writing any of it — the same order that changed the
BCF design twice and killed one of its stages.

## The problem, and why it is not BAM's or BCF's

BAM and BCF records are **length-prefixed**: finding record *n+1* means reading
record *n*'s length. That is a serial chain, and both formats needed a trick to
break it — BAM speculates per BGZF block because htslib starts a fresh block per
alignment, BCF speculates per candidate offset because bcftools does not.

FASTQ is **delimited**, not prefixed. A record is four lines:

```text
@name [description]
SEQUENCE
+[name]
QUALITY
```

So there is no chain to break. Newlines are findable in parallel with no
dependency at all. The difficulty moves somewhere else: **deciding which line
starts a record.** `@` is a legal quality character — Phred+33 `@` is Q31, a
thoroughly ordinary score — so "line begins with `@`" does not identify a record.

## The validator

At a candidate offset, with no data-dependent loop beyond finding three
newlines:

1. `buf[p] == '@'`
2. line 3 begins with `'+'`
3. `len(line 2) == len(line 4)` — sequence and quality must agree

## What it measures at

Swept at **every byte offset** of three fixtures, against ground truth:

| Fixture | Reads | Offsets swept | Records | Survivors | False positives |
|---|---|---|---|---|---|
| `ont_ultralong.fastq.gz` | ONT, 5.5–163 kb | 483,158 | 3 | 3 | **0** |
| `pacbio_hifi.fastq.gz` | HiFi, 13–20 kb | 279,195 | 8 | 8 | **0** |
| `illumina.fastq.gz` | 4,000 × 100 bp | 864,000 | 4,000 | 4,000 | **0** |
| **total** | | **1,626,353** | 4,011 | 4,011 | **0** |

**Survivors equal records exactly.** Unlike BCF — where the sieve let through
12.7 candidates per 64 KiB block and a second stage had to reject them — the
FASTQ validator admits nothing but true record starts on real data.

## The decoys are real, which is what makes the zero mean something

A sweep finding no false positives proves nothing if the hard case never
appears. It does appear:

| Fixture | `@` inside quality lines | Quality lines **starting** with `@` |
|---|---|---|
| ONT | 3,499 | 0 |
| HiFi | 0 | 0 |
| Illumina | 9,660 | **83** |

**83 quality lines begin with `@`** — the exact decoy the validator exists to
reject — and 13,159 `@` bytes sit inside quality lines. All rejected.

**Why they are rejected.** Take a decoy `@` inside a quality line. Line 1 is the
rest of that quality line, line 2 is the next record's `@name` line, and line 3
is the next record's *sequence*. The check requires line 3 to start with `+`, and
a sequence line never does — across all three fixtures, **0 sequence lines start
with `+`**.

### Which check is load-bearing — corrected 2026-09-09

**This section previously said the `+` check "does essentially all of the work".
That was an inference, and mutation testing showed it is wrong on real data.**

Deleting the `+` check entirely passed every test in the crate, including the
differential run against noodles' own parser over all three fixtures. The reason
is that the **length** check catches the same decoys: for a decoy at a quality
line, line 2 is the next record's *name* line and line 4 is its `+` line, and
those two rarely have the same length. The `+` check never got to prove itself.

Both checks are needed, and they cover different cases:

| Check | Rejects |
|---|---|
| `len(line 2) == len(line 4)` | every decoy in the committed fixtures |
| line 3 starts with `+` | decoys where those lengths happen to coincide |

`only_the_plus_check_rejects_a_length_matched_decoy` constructs the second case
by hand — a record whose name line and `+` line are both 3 bytes — because real
data does not supply one and a check nothing exercises is a check nothing
protects.

The general lesson, and it is the same one the BCF work recorded: **a validator
with several conditions needs a case per condition.** Measuring that the whole
thing admits nothing does not tell you which part is doing the admitting.

## Status

**Implemented, 2026-09-09.** Host reference in `fritillaria_fastq::columnar`,
kernels in `kernels/fastq_scan.cu`, launcher `fritillaria_cuda::FastqScanner`.
The GPU-free driver loop is `fritillaria-fastq/tests/device_reader.rs`; the
differential test against the CPU reference is
`fritillaria-cuda/tests/fastq_decode.rs`.

One thing the implementation found that the measurement had not:

**The decode must run *before* the tiling is proved, which is the opposite of
BCF.** A BCF record carries `l_shared`/`l_indiv`, so the host can compute every
record's end from the offsets alone and prove a tiling with no further device
work. A FASTQ record's length is not in its bytes — it is the sum of four line
lengths, which only a walk discovers. So the proof consumes the decode kernel's
`record_end` output, and a failed proof discards those columns and falls back to
the on-device walk. Slightly more work in the failure case, and none in the
success case, which is the one that happens.

## What follows for the kernel

- **One thread per candidate line start**, exactly as BCF's sieve does, then the
  same tiling proof: if the survivors tile the buffer, they *are* the true
  records. A failed tiling falls back to a serial walk, so a wrong guess costs
  work and never accuracy.
- **No second validation stage.** BCF keeps one for asymmetry despite it pruning
  nothing; here there is not even an argument for it, because the survivors are
  already exact and the sieve is already the full validator.
- **Candidates are line starts, not every byte.** A newline scan is a trivially
  parallel pass, and it cuts the candidate set by roughly the average line
  length before any validation runs.

## What this does not cover

- **Wrapped FASTQ**, where a sequence is split across several lines. Legal in
  older tools, never written by htslib, and it breaks the length check — so it
  would fail the tiling and fall back to the serial walk rather than produce
  wrong boundaries. Detected, not supported.
- **Plain `.fastq.gz`** — a single DEFLATE stream, which cannot be
  block-parallel at all. This document is about what happens *after*
  decompression; getting there is the container's problem, and for plain gzip
  the container does not cooperate.
- **Nothing has been timed.** Correctness first, and the BAM reconcile worry
  turned out to be unfounded — do not assume this one is real either.
