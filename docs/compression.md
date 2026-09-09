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
3. **Whether the output is spec-valid BGZF that samtools accepts** — including
   the 28-byte EOF block, or tools report truncation. This is the acceptance
   bar and it is binary.

Note that unlike decompression, there is no byte-identity oracle available: two
valid DEFLATE streams of the same input legitimately differ. The test has to be
round-trip plus htslib acceptance plus a ratio floor, which is a weaker net than
the read path enjoys — so the ratio floor needs to be an actual assertion, not a
number in a report.

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
