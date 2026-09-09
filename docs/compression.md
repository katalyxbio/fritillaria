# GPU BGZF compression: the measurement before the design

Measured 2026-09-09, before writing any of it — the same order that killed one
BCF stage and reshaped the FASTQ scan.

## Why this is not optional

Every other piece of this library removes a transfer. A GPU tool that produces
records on-device and then writes a BAM has to get them back to the host to
compress them, which pays back exactly the device-to-host copy the read path
deletes. Without device-side compression only half the pipeline is accelerated.

## The prior art says yes, and says something more useful

**NVIDIA Parabricks `fq2bam` already does this.** It compresses the final BAM
on-device with nvCOMP, behind a `--gpuwrite` flag, with a `--gpuwrite-deflate-algo`
knob. So GPU BGZF compression is not speculative — it ships in a product, and the
output is read by ordinary samtools.

The interesting part is what they chose. Parabricks defaults to
`--gpuwrite-deflate-algo 0`, documented as the *fastest*, and only later added
`3` for "more compression at slightly slower speed". Level 0 in nvCOMP's own
header is **entropy-only compression**. So the shipping default trades file size
for throughput, and the ratio question is live enough that users asked for a
knob.

**That is the question this document answers, and it is not "is it fast".**
Parabricks settles that. It is: *what does the fast setting cost in file size,
and can we afford it?*

## The bar: what htslib actually achieves

htslib writes BGZF with libdeflate at level 6. Measured on the committed
fixtures, per-block, payload against inflated size:

| Fixture | Data | Blocks | Compressed | Inflated | Ratio |
|---|---|---|---|---|---|
| `pacbio_hifi.bam` | real HiFi reads | 13 | 150,879 | 602,600 | **3.99x** |
| `kg_phase3.bcf` | 2504-sample genotypes | 57 | 181,153 | 3,707,763 | **20.47x** |
| `ont_ultralong.bam` | real ONT reads | 7 | 231,051 | 396,164 | 1.71x |
| `htslib_multiblock.bam` | **synthetic** | 14 | 465,847 | 820,223 | 1.76x |

**Do not benchmark compression on `htslib_multiblock.bam`.** Its records are
generated from near-random bytes, so it barely compresses and would flatter any
compressor by making the ratio spread look small. The real files are the HiFi
BAM and the two BCFs, and they differ from each other by 5x.

## What a fast setting costs

Same payloads, recompressed with zlib at three levels. Sizes are exact; the
throughput column is Python-mediated and only good for the shape of the curve.

**`pacbio_hifi.bam` — 602,600 bytes uncompressed**

| Compressor | Bytes | Ratio | vs htslib |
|---|---|---|---|
| htslib (libdeflate 6) | 150,879 | 3.99x | — |
| zlib level 1 | 181,503 | 3.32x | **+20.3%** |
| zlib level 6 | 159,979 | 3.77x | +6.0% |
| zlib level 9 | 156,463 | 3.85x | +3.7% |

**`kg_phase3.bcf` — 3,707,763 bytes uncompressed**

| Compressor | Bytes | Ratio | vs htslib |
|---|---|---|---|
| htslib (as written by bcftools) | 181,153 | 20.47x | — |
| zlib level 1 | 267,693 | 13.85x | **+47.8%** |
| zlib level 6 | 182,812 | 20.28x | +0.9% |
| zlib level 9 | 107,039 | 34.64x | −40.9% |

Two things fall out:

- **A "fast" setting costs 20–48% in file size.** On a 100 GB BAM that is 20 GB,
  stored for years. This is the number that decides the design, and it is much
  larger than the wall-clock it buys.
- **libdeflate genuinely beats zlib** at nominally equal effort — 150,879 against
  zlib-9's 156,463 on HiFi, while being far faster. htslib is a real bar, not a
  formality.
- The BCF row where zlib-9 wins by 41% is not us beating htslib: it shows
  bcftools wrote that file at level 6, and genotype data keeps compressing well
  past it. The bar is *libdeflate at the level the writer used*, not libdeflate
  at its best.

## What nvCOMP offers

From `nvcomp/deflate.h` 5.3.0.16, NVIDIA's own words:

| `algorithm` | Documented as |
|---|---|
| 0 | highest-throughput, **entropy-only** compression |
| 1 | high-throughput, low compression ratio — **nvCOMP's default** |
| 2 | medium-throughput, medium ratio, **beats Zlib level 1** |
| 3 | placeholder; currently falls through to the same as 2 |
| 4 | lower-throughput, higher ratio, **beats Zlib level 6** |
| 5 | lowest-throughput, highest ratio |

Reading that against the table above is the whole decision:

- **Level 2 "beats Zlib level 1"** — and Zlib level 1 is +20% to +48% against
  htslib. Beating it is a low bar and still well short.
- **Level 4 "beats Zlib level 6"** — and Zlib level 6 is +6.0% / +0.9% against
  htslib. That is the first rung where output size is roughly at parity.
- **Parabricks defaults to 0**, which is below all of these.

## Decision: default to ratio parity, not to speed

**Our default should be `algorithm = 4`, and the level must be caller-visible.**

This is deliberately the opposite of Parabricks' default, and the reason is the
project's own thesis. The argument for this architecture is that *PCIe carries
compressed bytes instead of decompressed ones, so the compression ratio is an
effective bandwidth multiplier on the link* — 3.37x on real WGS. A writer that
defaults to entropy-only compression would hand back files that are 20–48%
larger, degrading the multiplier for every subsequent reader of that file, to
save wall-clock on the one write. That trades a permanent cost for a transient
one.

Parabricks' choice is defensible for their workload: `fq2bam` is a batch
alignment step where end-to-end runtime is the headline number, and the BAM is
often an intermediate. A library does not get to assume that. The file we write
is somebody's archive.

So: level 4 by default, `0`–`5` exposed, and the chosen level recorded in
whatever we print, so a user who picks speed knows they picked it.

## The memory cost, measured — and it is the real constraint

Queried from the real library on **a machine with no GPU**, because all three
sizing entry points are host-side. Per 64 KiB chunk:

| `algorithm` | scratch / chunk | ×chunk | output slot | ×chunk | **total VRAM / chunk** | ×chunk |
|---|---|---|---|---|---|---|
| 0 entropy-only | 0 | 0x | 148,256 | 2.26x | 213,792 | **3.3x** |
| 1 low ratio | 360,480 | 5.5x | 148,256 | 2.26x | 574,272 | **8.8x** |
| 2 medium | 655,392 | 10x | 148,256 | 2.26x | 869,184 | **13.3x** |
| **4 high ratio** | **1,114,184** | **17x** | 148,256 | 2.26x | **1,327,976** | **20.3x** |
| 5 max ratio | 1,179,720 | 18x | 148,256 | 2.26x | 1,393,512 | **21.3x** |

Three things here were not guessable, and all three shape the implementation.

**Scratch is 17x the payload at our chosen default.** It is exactly linear in the
chunk count (verified to 16,384 chunks; the lower rungs deviate by a few parts
per million, which is bounded rather than modelled), so a batch sizer can divide
a VRAM budget by a constant. But the constant is large: compressing 64 KiB costs
over a megabyte of scratch.

**So compression batches must be far smaller than decompression batches.** The
read path put 166,012 blocks in one batch. At 1.33 MB per block, a 16 GiB T4
holds roughly **11,000 blocks — about 708 MiB of payload — and an L4 about
17,800**. That is a 15x reduction in batch size against the read path, and it
comes entirely from scratch, not from the data.

**The ratio ladder is also a VRAM ladder**, which the header does not mention.
Level 0 fits 6.2x more blocks per batch than level 4 — so choosing ratio costs
throughput twice: once in the kernel, and again in more launches over smaller
batches. This is a genuine argument for Parabricks' default that the ratio
tables alone do not show, and it is why the level has to stay caller-visible.

It does not overturn the choice. A 20–48% size penalty is permanent and paid by
every future reader; a smaller batch is a scheduling cost paid once, and the
measured per-batch overhead on the read path was small. But the decision is now
made against the real tradeoff rather than half of it.

**The output slot is 2.26x the chunk**, not the ~1.0006x DEFLATE's own
worst-case expansion implies. nvCOMP wants generous room, and since real sizes
are only known after the kernel runs, output has to be preallocated at that
worst case per chunk.

**Output alignment is 8, where decompression's is 1.** That difference matters:
an output alignment of 1 is what lets nvCOMP inflate straight into a dense
buffer, and it is why the read path needs no compaction pass. Compression gets
no such gift, so compressed blocks land in padded worst-case slots and the BGZF
stream must be gathered out of them. That pass is over the *compressed* data —
the small side — so it is far cheaper than the equivalent would have been on the
read path, where it would have touched the largest buffer in the pipeline.

Both facts are now pinned by tests rather than remembered: `ffi.rs` asserts the
output alignment is 8 and that scratch is linear, so a future nvCOMP that
relaxed either would fail loudly and the compaction pass could be deleted on
evidence.

## The CPU reference, built and measured 2026-09-10

`fritillaria_bgzf::CpuCompressor` implements `BlockCompressor`: it takes a
concatenated buffer plus block boundaries — exactly what a `BlockCodec` hands
back — and produces a ready-to-write BGZF byte stream. It is the oracle the
device path is diffed against, and the local baseline for the ratio.

Every committed fixture, inflated and recompressed at miniz level 6, against the
htslib bytes that came in:

| Fixture | htslib | ours | vs htslib | our ratio |
|---|---|---|---|---|
| `pacbio_hifi.bam` | 151,245 | 160,291 | **+6.0%** | 3.76x |
| `ont_ultralong.bam` | 231,261 | 237,464 | +2.7% | 1.67x |
| `htslib_multiblock.bam` | 466,239 | 471,795 | +1.2% | 1.74x |
| `kg_phase3.bcf` | 182,663 | 187,672 | +2.7% | 19.76x |
| `giab_hg002.bcf` | 15,492 | 15,649 | +1.0% | 12.61x |

**+1.0% to +6.0% behind libdeflate at the same nominal level**, which is the
expected shape — libdeflate is genuinely better than zlib — and the number nvCOMP
has to be read against. It is *asserted*, not printed: the test fails past +10%,
because a compressor that quietly stopped compressing would otherwise pass every
other check.

### One block out per block in — and why it is a contract

The compressor emits exactly one BGZF block per input chunk, and errors rather
than splitting one that will not fit.

**This protects the reader, not the writer.** htslib starts a new block rather
than splitting a BAM record, which is what makes a block start almost always a
record start — the property the GPU record scan leans on to get one independent
chain per block instead of one serial chain per batch. A compressor that split
an over-large chunk to make it fit would emit a file that is valid, reads
correctly, and is *slower for us to read*, with nothing to indicate why. That is
the worst kind of bug this project can ship, so the seam refuses instead.

`BgzfWriter::write_block` used to do exactly the wrong thing here — split the
payload and retry — and now delegates to the compressor.

**Making the guarantee unconditional needed one measurement.** A general-purpose
deflate has no bound on its output: miniz at level 6 expands random input by 15
bytes, because it emits three stored blocks rather than one, and that is enough
to push a maximal chunk past BGZF's 64 KiB whole-block cap. A hand-built DEFLATE
stored block is exactly `len + 5` with no data-dependent term. So the fallback is
to store, and the chunk limit is `65536 − 18 − 8 − 5 = 65505`
(`MAX_COMPRESSIBLE_PAYLOAD`) — 5 bytes under what the format would allow a
payload to be, and unreachable in practice since writers use 65280 anyway.

Taking the smaller of deflate and stored is also free ratio on incompressible
input, which is the case that would otherwise *cost* bytes rather than merely
fail to save them.

**The device path inherits all of this.** nvCOMP's worst case is 2.26x the chunk,
far past the cap, and a kernel cannot re-run one chunk mid-launch — so it must
detect over-cap blocks after the launch and re-emit them as stored blocks, which
is precisely what `store_block` does. Same code, host side.

### What the tests can and cannot do

There is no byte oracle, so the net is three checks, and each one catches
something the others do not. Verified by mutation:

| Mutation | Caught by |
|---|---|
| never fall back to storing | one-block-per-chunk, at the maximal chunk |
| always store (stopped compressing) | the ratio floor, and the EOF marker |
| CRC over the compressed stream | round trip, and `samtools` |

The third is the one worth naming: the gzip trailer checksums the bytes *going
in*, so a device path must CRC before compressing, not after. It is an easy
inversion to make and produces a file that only fails on read.

## What is still unmeasured

Everything above about nvCOMP is **NVIDIA's claim, not our measurement.** The
zlib and htslib numbers are ours, on our fixtures; the nvCOMP ladder is a header
comment. Before any of this is believed:

1. **nvCOMP's actual ratio at each level on real BAM and BCF payloads**, against
   the htslib bar in the table above. "Beats Zlib level 6" is a claim about
   zlib, and libdeflate is better than zlib.
2. **Throughput at level 4**, which is the level we would actually ship. The
   9.39 GB/s figure quoted for an H100 is presumably level 0 or 1, and does not
   transfer.
3. ~~**Whether the output is spec-valid BGZF that samtools accepts**~~ — settled
   for the *host* path on 2026-09-10: `samtools`/`bcftools` read every
   recompressed fixture and report the same record counts, and re-framing a
   fixture's trailing empty block reproduces the 28-byte EOF marker byte for
   byte. The device path still has to clear the same bar, but the harness that
   checks it now exists and runs locally.

The memory table above *is* ours and needs no device — nvCOMP answers those
queries on a machine with no driver, which is why they were measured before
anything was built rather than discovered on a rented VM. Only the ratio and
throughput columns need hardware.

## The API, transcribed

`nvcompBatchedDeflateCompressAsync` mirrors the decompress entry point we
already dlopen:

```c
nvcompStatus_t nvcompBatchedDeflateCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t     *device_uncompressed_chunk_bytes,
  size_t            max_uncompressed_chunk_bytes,
  size_t            num_chunks,
  void             *device_temp_ptr,
  size_t            temp_bytes,
  void *const      *device_compressed_chunk_ptrs,
  size_t           *device_compressed_chunk_bytes,
  nvcompBatchedDeflateCompressOpts_t compress_opts,
  nvcompStatus_t   *device_statuses,
  cudaStream_t      stream);
```

`nvcompBatchedDeflateCompressOpts_t` is **64 bytes** (`int algorithm` plus 60
reserved bytes that must be zeroed), and nvCOMP's own default is `algorithm = 1`.
It is passed **by value**, so the `size_of` assertion discipline already applied
to the decompress options applies here too: a layout mismatch corrupts arguments
silently rather than failing to link.

The batched model is the same one that made decompression drop in — an array of
independent chunks is exactly the BGZF block model. The output side is new work:
BGZF needs each compressed block wrapped in a gzip member with the `BC` subfield
carrying *total block size minus 1*, a CRC32 of the **uncompressed** payload, and
`ISIZE`. The CRC is already computed on-device by
`nvcompBatchedCRC32Async`, which we call today for verification.
