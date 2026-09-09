# BCF record boundaries on a device

## The finding

`fritillaria-bam`'s device-side boundary scan rests on a habit of htslib's: its
BAM writer calls `bgzf_flush_try` before each alignment and starts a fresh BGZF
block rather than splitting a record that would not fit. So
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
of offsets in parallel.

### The staging

A `true` is a hypothesis, so it is confirmed in increasing order of cost:

1. **`looks_like_a_record`** — fixed cost, run at every candidate offset,
   massively parallel. Prunes to a handful of survivors per block.
2. **`Record::validate`** — walks the typed-value chain and requires the fields
   to end *exactly* where `l_shared` and `l_indiv` say. A misread descriptor
   desynchronises the walk and cannot survive this. Cost is proportional to the
   record, so it runs only on survivors.
3. **The proof** — originally BAM's serial chain reconciliation. Measurement
   replaced it with something better; see the next section.

## The proof, revised by measurement

The design above originally ended with **chain reconciliation** — BAM's serial walk over blocks,
carried across unchanged. Measuring the validator showed that is not needed, and the replacement
is strictly better:

> If the surviving offsets, in order, **tile** the buffer — the first is the known first record,
> each one's computed end is exactly the next, and what remains is a record running past the end
> — then they **are** the true record chain.

By induction: the true chain starts at the same known offset, and each record's length is
determined by its own bytes, so the true successor of `offsets[i]` is
`offsets[i] + 8 + l_shared + l_indiv`, which the tiling asserts equals `offsets[i+1]`. There is
no room for a different answer.

That check is O(1) per record and embarrassingly parallel, so **the serial phase disappears
entirely** rather than merely shrinking. Correctness does not depend on it succeeding: a failure
means an extra or missing survivor, and the serial walk is taken instead. A wrong guess costs
work, never accuracy — the same discipline as BAM, with a cheaper proof.

## Measured, 2026-09-09

`cargo run --release -p fritillaria-bcf --example scan_survivors -- <file>`, sweeping
[`looks_like_a_record`] at **every byte offset of every block** — exactly what a thread-per-offset
kernel would test.

| file | records | candidate offsets | survive sieve | survive validation | false positives |
|---|---|---|---|---|---|
| `giab_hg002.bcf` | 275 | 172,632 | 275 | 275 | **0** |
| `kg_phase3.bcf` | 715 | 3,670,536 | 715 | 715 | **0** |
| sites-only chr22 | 46,576 | 5,808,422 | 46,576 | 46,576 | **0** |
| chr22:16.05–19 Mb, 2504 samples | 75,783 | **389,198,715** | 75,783 | 75,783 | **0** |

**399 million candidate offsets, zero false positives.** Survivors per 64 KiB block: 12.7 mean,
13 worst on the panel files. The two questions this document said to answer first are answered:

- **Survivors per block are in single digits**, not hundreds. The scheme works.
- **The serial walk is not the cheap option it was on BAM.** It costs 432 ns *per record* on the
  host, and a batch holds thousands of records where a BAM batch holds hundreds of blocks. This
  is the case BAM's reconcile measurement did *not* generalise to, which is why it needed
  measuring rather than assuming.

The third row is the adversarial case and was included deliberately: a sites-only file has
`n_sample == 0`, which makes the strongest check vacuous. It was still exact.

**None of that is a guarantee.** Nothing in the format forbids a record's payload from containing
bytes that satisfy the validator, and the implementation is built so that when it happens it
costs a fallback rather than a wrong answer.

### What the measurement did *not* justify

**Stage 2 pruned nothing.** On all four files the sieve alone was already exact, so full
validation is currently pure insurance with a real cost — it walks each survivor's typed chain,
which across all survivors is roughly a pass over the batch.

It is kept, and the honest reason is asymmetry rather than evidence: without it, a single sieve
false positive breaks the tiling and sends a whole batch to the serial walk, and
`validation_is_what_lets_the_tiling_survive_a_sieve_false_positive` builds exactly that case by
hand because no real file produced one. **A device implementation may reasonably skip it** and
accept more fallbacks; correctness does not depend on it, since a tiling that holds is the true
chain whether or not validation ran.

## Status

**Host reference and kernel both done, verified on a T4.**

`crates/fritillaria-bcf/src/speculative.rs` implements all four phases on the host —
`sieve`, `validate`, `prove_tiling`, and the serial fallback — as the CPU reference the kernels
are diffed against. Each phase is a separate public function so it maps to one launch.

`crates/fritillaria-cuda/kernels/bcf_scan.cu` is the device translation, launched by
`fritillaria_cuda::BcfScanner`. On a T4 it reproduces the host reference exactly on all three
fixtures — same boundaries, same tail, same survivor counts — and the tiling proof held on every
batch, including at seams driven one block at a time.

### The division of labour, and the transfer it accepts

| phase | where | why |
|---|---|---|
| sieve | device, one thread per byte | touches every byte of the batch |
| validate | device, one thread per survivor | proportional to the record, runs only on survivors |
| sort + prove | **host**, over a few thousand offsets | microseconds anywhere |
| fallback walk | device, one thread | so a failure costs a slow pass, not a D2H of the batch |

The offsets come back to the host — tens of kilobytes against the batch's tens of megabytes.
That is a transfer this project normally refuses, and the distinction is worth stating: the D2H
the design exists to delete is the *inflated payload*, which scales with the data. This one
scales with the record count. It stops being the right call the moment a device-side columnar
BCF decode exists to consume the offsets in VRAM, and not before — building that plumbing for a
consumer that does not exist is how untested code ships.

The fallback walk is the exception and stays on device deliberately: copying the batch back to
walk it on the host would pay exactly the transfer this library removes.

### What the kernel work found

**`Record::validate` was too lax, and the kernel caught it.** The reference iterated the FORMAT
fields — proving none runs *past* `l_indiv` — but never checked that together they fill it. The
CUDA translation asserted the stricter thing, so the two would have disagreed on a record no real
file contains. Being a differential oracle means the strictness has to match in both directions;
the reference now checks it and a mutation confirms the check bites.

**`with_blocks_per_batch` budgets *compressed* bytes.** At 64 KiB per block that is about three
real blocks on a BAM at 3.37x, and about twenty on genotype BCF at 20x. Asking for 8 on the
182 KB panel fixture reads the whole file in one batch — which is how a GPU test asserting
"several batches" failed on a rented VM for a reason that had nothing to do with the GPU.
`fritillaria-bcf/tests/device_reader.rs` now drives the same loop through `HostDeviceCodec`, so
that class of mistake is caught in a second on a machine with no CUDA.

## What to measure next

Correctness is established; performance is not. Nothing here has been timed:

- **Whether the sieve is bandwidth-bound.** One thread per byte, each reading 32, is a heavily
  overlapping pattern that should sit in cache — but the useful comparison is against the 0.86s
  nvCOMP spends inflating the same data on an L4, and that number does not exist yet.
- **What the fallback actually costs**, so the cost of a false positive is known rather than
  assumed rare-and-therefore-fine.
- **Whether the offsets round trip matters.** It is small, but it is a host synchronisation per
  batch, and BAM's decode showed per-batch overhead dominating the phase it lives in.
