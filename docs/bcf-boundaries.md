# BCF record boundaries on a device

## The finding

`fritillaria-bam`'s device-side boundary scan rests on a habit of htslib's:
`bcf_write`'s BAM equivalent calls `bgzf_flush_try` before each alignment and
starts a fresh BGZF block rather than splitting a record that would not fit. So
a BAM block start is *almost always* a record start, which turns one serial
chain per batch into one independent chain per block — the thing that made the
scan parallel at all.

**BCF does not do this.** `bcf_write` calls `bgzf_write` directly, which packs
blocks to the 65280-byte payload cap. Measured on the committed fixtures:

| File | interior block starts | that are also record starts |
|---|---|---|
| `kg_phase3.bcf` | 56 | **0** |
| `giab_hg002.bcf` | 3 | **0** |

Zero, not "few". A BCF record straddles essentially every interior block
boundary. `bcf_records_cross_block_boundaries_where_bam_records_do_not` in
`crates/fritillaria-bcf/tests/bcftools.rs` asserts this, so if a future htslib
changes the behaviour the test says so rather than the assumption rotting
quietly.

This was found by measuring the first fixture, not by reasoning about htslib —
which is the argument for building the fixture before the parser rather than
after.

## Why it matters

The BAM approach cannot be ported. Its correctness argument is "speculate at
each block start, then adopt a block's precomputed walk only once the true chain
is proven to arrive there" — a wrong guess costs work, never accuracy. That
still holds for BCF, but the guess is now *always* wrong, so every block falls
back to walking and the scan degenerates to one serial chain over the batch.

At ~5 KB per record on `kg_phase3.bcf`, a 256-block batch holds roughly 3,300
records. A single GPU thread walking 3,300 dependent loads is exactly the
pattern the BAM work existed to avoid, and `fritillaria-bam`'s measurement puts
the cost of dependent chasing well above the inflate it would be hiding behind.

## The replacement, and why BCF makes it easier than BAM would

Speculate at **many candidate offsets per block** rather than at the block start
alone, and prune with a validator. This is only viable if the validator is cheap
and rejects hard. BCF's is, and BAM's would not have been:

| check | why a random offset fails it |
|---|---|
| `n_sample == header.sample_count()` | a specific 3-byte value at a fixed offset |
| `0 <= CHROM < n_contig` | 4 bytes constrained to a small range |
| `l_shared >= 24` and `pos >= -1` and `rlen >= 0` | sign and magnitude constraints |
| `8 + l_shared + l_indiv` lands in the buffer | two 4-byte lengths must agree with the batch |
| `n_allele >= 1`, `l_indiv` consistent with `n_fmt` | cross-field agreement |

`n_sample` is the strong one and it has no BAM analogue: every record in a file
carries the *same* value, known from the header before the scan starts. A BAM
record's fields are all record-specific, so an equivalent validator would have
only range checks to work with.

`looks_like_a_record` in `crates/fritillaria-bcf/src/record.rs` implements this.
It is deliberately **fixed-cost** — it reads the 32-byte prefix and nothing
else, with no data-dependent loop — because the point is to run it at thousands
of offsets in parallel. On a synthetic buffer of eight records it accepts every
real start and nothing else; that is a unit test, not a guarantee, and the
design does not depend on it being exact.

### The staging that keeps it correct

A `true` is a hypothesis. Three levels, in increasing cost, and none of them can
be skipped:

1. **`looks_like_a_record`** — fixed cost, run at every candidate offset,
   massively parallel. Prunes to a handful of survivors per block.
2. **`Record::validate`** — walks the typed-value chain and requires the fields
   to end *exactly* where `l_shared` and `l_indiv` say. A misread descriptor
   desynchronises the walk and cannot survive this. Cost is proportional to the
   record, so it runs only on survivors.
3. **Chain reconciliation** — the same rule `fritillaria-bam::blocked` uses:
   adopt a block's speculative walk only once the true record chain is proven to
   arrive at its start. A wrong guess costs work, never accuracy.

Only step 3 makes the result correct. Steps 1 and 2 make step 3 cheap by
shrinking how much chain has to be walked before a block's precomputed answer
can be adopted.

## Status

Not implemented. `looks_like_a_record` and `Record::validate` exist and are
tested; the device kernel and the reconcile pass are not written. The host path
uses `scan_records`, which is an ordinary serial walk and entirely adequate
there — BAM's is too.

## What to measure before building it

The design above is reasoning, and this project has a record of measurements
overturning reasoning. Two numbers decide whether it is worth writing:

- **The false-positive rate of `looks_like_a_record` on real inflated BCF**, per
  block, not on a synthetic buffer. If survivors per 64 KB block are in the
  single digits the scheme works; if they are in the hundreds, step 2 dominates
  and the parallelism is spent on validation.
- **The serial-chain cost it is being compared against.** `fritillaria-bam`
  measured reconcile at 222 ns per block and the worry about it turned out to be
  unfounded. Records per batch is a much larger number than blocks per batch, so
  the same may not hold — but it should be measured on a BCF batch before any
  kernel is written, because if the serial walk is fast enough there is nothing
  here to solve.

Note also that a BCF batch is unusually cheap to *inflate* relative to its
record count: real WGS genotypes compress 20x, against 3.37x for the BAM
measured in CLAUDE.md. That shifts the balance toward decode mattering more, not
less.
