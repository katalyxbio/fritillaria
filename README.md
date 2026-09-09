# fritillaria

**[noodles](https://github.com/zaeleus/noodles) for GPU.** A Rust-native library for genomic
file formats that puts records **in GPU memory**, so GPU-accelerated tools can be built on top
of it instead of rewriting BGZF, BAM and BCF parsing from scratch.

The motivating case: decompress a BAM directly into device memory to feed a GPU-accelerated
tool, so the data never crosses PCIe in its decompressed form. Not just alignment — variant
calling, QC and general pipeline I/O are the same problem, which is why format coverage matters
as much as speed. A GPU tool that hits an unsupported format has to fall back to the CPU and
round-trip its data, losing the entire benefit.

**Already using noodles? Adoption is a rename.** The CPU half of this library *is* noodles —
all 18 crates, vendored and renamed, MIT © 2018 Michael Macias (see [VENDORED.md](VENDORED.md)).
Every format it reads, this reads, with the same API:

```rust
use noodles_bam as bam;   // before
use fritillaria::bam;     // after
```

Putting a GPU underneath is one more line, because `BgzfReader` implements the BGZF traits every
format reader is generic over. Device-resident and columnar access are additive on top of that,
never a replacement for it.

**This is not trying to be a faster `samtools`.** Against `bgzip` as a standalone decompressor
we lose, and that is the wrong comparison: htslib produces bytes in host RAM, which is not where
a GPU consumer needs them. The comparison that matters is *time to records in device memory*,
where PCIe carries compressed bytes instead of decompressed ones — a 3.37x reduction in link
traffic on real WGS data. See *Performance*.

> **Early, and honest about it.** Read and write work for every format, because that half is
> noodles. The GPU half is narrower than the format list suggests: **BAM**, **BCF** and
> **bgzipped FASTQ** go in and device-resident columns come out, verified against
> htslib-written files on a real GPU. Everything else in a BGZF container gets GPU
> *decompression* and no more, and the text formats get neither. The table below keeps those
> columns apart on purpose.

## Which inputs get the GPU

The dividing line is the **container**, not the format.

BGZF — the container under BAM, BCF, and anything `bgzip`ped — is a sequence of *independent*
gzip members, each holding at most 64 KiB. Block *i* needs nothing from block *i-1*, so blocks
map cleanly onto parallel hardware. Everything in a BGZF container therefore shares one
accelerated path: block discovery, parallel inflate, CRC verification, and virtual offsets are
all container-level.

An ordinary `.gz` is **not** that: it is a single DEFLATE stream with a 32 KiB sliding window,
and back-references make it inherently serial. It cannot be split without a pre-built index.

This library will not pretend otherwise. Which path an input takes is public API, because
silently delivering CPU speed to someone who came for a GPU is the worst thing it could do.

Every format below **reads and writes on the CPU today**, because that half is noodles. What
varies is the GPU column, and it is the only one worth tracking.

| Format | Container | GPU path |
|---|---|---|
| BAM (aligned and unaligned) | BGZF | **full** — inflate, record scan, columnar decode, device-resident |
| BCF | BGZF | **full** — boundary scan and columnar site-core decode, device-resident |
| FASTQ (`bgzip`ped) | BGZF | **full** — boundary scan and columnar decode, device-resident |
| `bgzip`ped VCF, tabix-indexed files | BGZF | decompression only |
| SAM, BED, GFF/GTF, FASTA, CRAM | text / own | none, and none claimed |
| plain `.gz` (any format) | one DEFLATE stream | cannot be block-parallel |

**Breadth is done; depth is partial.** Three formats have the device-resident columnar path this
project exists for — BAM, BCF and bgzipped FASTQ, which between them cover raw-read input,
alignment and variant calling. The rest is the roadmap, not a promise.

Note that *decompression* and *parsing* are separate claims. Everything in a BGZF container gets
GPU decompression from one container-level kernel; parsing has no shared layer, so each format
needs its own and the three that exist did not generalise to one another.

Note that BAM is an **input** format as well as an output one — Nanopore and PacBio deliver raw
reads as unaligned BAM, where the basecaller's output lives in aux tags (`MM`/`ML` base
modifications, per-base kinetics) rather than being trailing detail. Those are decoded: every
scalar type and every `B` array subtype, zero-copy, validated tag-by-tag against `samtools view`
on real PacBio HiFi reads. The `CG` long-CIGAR workaround is implemented too, though only tested
against hand-built records — even ultra-long ONT reads in a 187 GB GIAB file topped out at
46,943 of the 65,535 operations needed to trigger it.

Ultra-long reads matter for a second reason: an ONT record can be far larger than a 64 KiB BGZF
block, so records genuinely span blocks and can exceed a whole batch. `testdata/ont_ultralong.bam`
carries a 254 KB record crossing three block boundaries, and the tests drive the
carry-the-partial-record loop a real consumer has to write.

## Status

19 crates. Twelve are vendored noodles, unchanged. Five hold **both** halves — the vendored
CPU API at the crate root, ours alongside it — and two are entirely ours.

| Crate | What works |
|---|---|
| `fritillaria-core` | *Ours:* errors, `VirtualOffset`, the `BlockCodec`/`DeviceBlockCodec` seams. *Vendored:* `Position`, `Region` |
| `fritillaria-bgzf` | *Ours:* block discovery, CPU codec, writer, batched `BgzfReader`, device-resident `DeviceBgzfReader`. *Vendored:* `io::{Reader, Writer}`, `gzi` |
| `fritillaria-bam` | *Ours* (`columnar`): record boundary scan, `RecordBatch`, zero-copy `Record`, aux tags, device columns. *Vendored:* `io`, `bai`, `fs`, `record` |
| `fritillaria-bcf` | *Ours* (`columnar`): header + dictionaries, boundary scan (host **and** device), BCF2 typed values, device columns. *Vendored:* `io`, `fs`, `record` |
| `fritillaria-fastq` | *Ours* (`columnar`): validator, boundary scan, `RecordBatch`, device columns. *Vendored:* `io`, `fai`, `fs`, `record` |
| `fritillaria-cuda` | All ours. DEFLATE inflate + CRC32 kernels, device-resident output, nvCOMP codec, columnar decode for BAM, BCF and FASTQ — verified on a Tesla T4 |
| `fritillaria` | Facade: re-exports every format, plus backend selection (nvCOMP → our kernel → CPU) |
| `-sam -vcf -csi -tabix -cram -bed -fasta -gff -gtf -util -htsget -refget` | Vendored unchanged |

Our own code is about 19k lines; the vendored half is 137k. The GPU work is the smaller number
and the reason the project exists — see [VENDORED.md](VENDORED.md) for the exact split.

Two GPU codecs sit behind the same traits with the same mandatory verification, so choosing
between them is a performance decision rather than a semantic one. NVIDIA's **nvCOMP** is the
fast path; our own inflate kernel is the portable fallback and a second test oracle. It covers
all three DEFLATE block types (stored, fixed and dynamic Huffman), one thread per BGZF block,
with CRC32 folded in so verification costs no extra pass.

Correctness is checked by differential testing against `miniz_oxide` on real hardware — and for
nvCOMP against our kernel as well, so three independent implementations have to agree
byte-for-byte on an htslib-written BAM.

**The vendored readers sit on the GPU path with no changes to them.** `BgzfReader` implements
the BGZF reader traits that every format crate is generic over, so `bam::io::Reader` reads a
real htslib BAM whose blocks were inflated on the GPU — verified on a T4, with GPU and CPU
backends producing identical records. The same substitution accelerates every other BGZF format
— verified for BCF, where `bcf::io::Reader` reads a bcftools-written file through this reader
with no new code on our side.

That property is not just convenience: it is what keeps the vendored modules byte-for-byte
upstream, and so keeps a rebase onto a newer noodles mechanical. See [VENDORED.md](VENDORED.md).

```rust
use fritillaria::{bam, bgzf::BgzfReader, cuda::CudaCodec};

// Same API as noodles, GPU decompression underneath.
let inner = BgzfReader::with_codec(file, CudaCodec::new()?);
let mut reader = bam::io::Reader::from(inner);
let header = reader.read_header()?;
for record in reader.records() { /* ... */ }
```

Correctness is checked against htslib, not just against ourselves: `testdata/` holds
`samtools`-written BAMs, and the tests both read those and hand our writer's output back to
`samtools`.

### Performance

Measured over a 3 GiB prefix of a real 1000 Genomes WGS BAM (166,012 blocks, 10.11 GiB
inflated, 3.37x ratio) on an L4. Both GPU codecs in the same run, back to back on the same file
— the only fair way to compare them. Device-resident, so `download` is just the per-block
verification arrays rather than the payload.

| Phase | our kernel | nvCOMP |
|---|---|---|
| upload H2D | 0.86s | 0.89s |
| **kernel (inflate + CRC32)** | 4.68s | **0.86s** |
| download D2H | 0.003s | 0.003s |
| **sum of phases** | 5.55s | **1.75s** |

**nvCOMP is 5.5x faster on the kernel phase**, at 11.8 GiB/s of output — and that is *including*
the per-block CRC32, which our kernel folds into inflate and nvCOMP needs a second pass for.
Verification is never skipped on either path.

Two costs worth naming because they turned out to be small: nvCOMP requires 4-byte-aligned input
and a BGZF payload starts 18 bytes into its gzip member, so payloads are restaged on-device
first — that is the entire 31 ms difference in the upload row, 0.6% of the phase. And the earlier
D2H figure of 7.2s, once 54% of runtime, is gone: keeping output on the device deleted it rather
than optimising it.

### Time to records in device memory — the comparison that matters

Measured end to end on the same L4 and the same file: a BAM on disk in, **29,887,809 parsed
records in device columns** out. This is the shipping path — `DeviceBgzfReader` feeding
`BamDecoder` — not a rig, and nothing returns to the host but the BAM header.

| Path | Produces | Wall |
|---|---|---|
| **fritillaria + nvCOMP** | 29.9M **parsed records**, columnar, in VRAM | **2.812s** |
| `bgzip -d -@11` | unparsed **bytes** in host RAM | 4.00s |
| `bgzip -@11` then upload | unparsed bytes in VRAM | 4.00s pipelined / 6.97s sequential |

**About 1.4x on wall clock against a 12-core htslib — while delivering something it has not
produced at all.** The CPU path ends with raw bytes and still has every record to parse.

Two things the breakdown shows, both of which the design was betting on without evidence until
now:

- **Turning bytes into records costs ~7% of the run** — 0.188s against 2.625s of getting the
  bytes there. Decoding on the device is close to free next to moving the data.
- **Batch size barely matters** between 256 and 4096 blocks. The serial reconcile step in the
  boundary scan measures 222 ns per block, or 37 ms across the whole file, so the phase that
  looked most likely to disappoint does not.

The structural point is the durable one: **PCIe carries compressed bytes instead of decompressed
ones, so the compression ratio becomes an effective bandwidth multiplier on the link** — 3.37x
fewer bytes on real WGS. That does not depend on any codec being good and does not go away on
better hardware. The host also never has to hold the 10.11 GiB at all.

> An earlier version of this table read **3.9x**. That number was inflate-only: it decoded no
> records and excluded host-side block discovery, so it was not the metric it was labelled with.
> 1.4x is the measured figure and it supersedes it.

### As a standalone decompressor, htslib still wins

Same machine, same file. `bgzip -d` on 11 threads does the job in 3.5-4.0s; getting our output
back into host RAM means paying the device-to-host copy this whole design exists to delete,
which puts the GPU path behind it.

| | wall | MiB/s of compressed input | htslib |
|---|---|---|---|
| `bgzip -d -@ 1` | 24.24s | 127 | 1.16 |
| **`bgzip -d -@ 11`** | **3.48s** | **882** | 1.16 |
| `bgzip -d -@ 11` | 4.00s | 768 | 1.13 |

The two `-@ 11` rows are the same command on different VM images, and the spread is why only
same-run comparisons are used above. nvCOMP's inflate reproduced exactly across those runs
(1.754s both times), so the variance is in the baseline, not in us.

So there are two different numbers here and they answer different questions. If you want bytes
in host memory, use htslib. If you want records in device memory, that is what this is for.
**No standalone decompression speedup is claimed anywhere in this project.**

Caveat on all of the above: Colab reports PCIe **gen 1 x16** (~4 GB/s), well below a real gen 4
link. Upload dominates the nvCOMP path here, so on faster hardware the balance shifts back
toward the codec. The 3.37x ratio advantage is invariant.

### What's next

Decompression is the on-ramp, not the product — and with inflate now balanced against the
upload, further codec work buys little. The effort belongs downstream of it:

1. **Host-side block discovery**, which is now the likely bottleneck: a sequential walk of BGZF
   headers that measures 2.36s standalone against 0.86s of GPU inflate.
2. **Columnar BCF decode on device.** Parsing, and now boundary discovery, work end to end;
   turning the boundaries into device-resident columns is what BAM has and BCF does not. Then
   `bgzip`ped VCF and FASTQ.
3. **GPU-side BGZF compression**, so a tool that produces records on-device can write them back
   without paying the transfer the read path just removed.

Then indexes and region queries, and CPU text formats for breadth. CRAM is not currently planned.

## Building

```bash
cargo build          # no CUDA toolkit needed
cargo test           # CPU paths
```

`cargo build --features cuda` also works without a GPU or `nvcc`: cudarc dlopens `libcuda` at
runtime and kernels are compiled by NVRTC on whichever machine has the device. Tests that need a
GPU skip rather than fail.

`cargo build --features nvcomp` likewise works with nvCOMP absent. It is dlopened at runtime,
never linked — NVIDIA's library is proprietary and cannot be redistributed, and requiring it at
build time would make the feature undevelopable on a machine without it. Point
`FRITILLARIA_NVCOMP_LIB` at the library to use it, or install it where the loader will find it;
`select_codec` falls back to our kernel when it is missing rather than failing.

To run them for real on a rented Colab T4 (the VM is released automatically, including on
failure):

```bash
./scripts/colab_test.sh
```

## Design

Three things shape everything else:

1. **A noodles-shaped API, with columnar access added rather than substituted.** The familiar
   record-at-a-time surface is what makes migration possible; dense columns are what make the
   GPU worth having. Fixed-width fields land in `&[i32]`-style columns instead of per-record
   structs, because that is what a GPU consumer wants next and re-scattering them into
   `Vec<Record>` would throw the win away.
2. **The container is the unit of acceleration, not the format.** Everything BGZF-contained
   shares one GPU path, so each new BGZF format costs only its record parsing. This is why BCF
   and `bgzip`ped VCF come before text formats.

   BCF is the first test of that, and it came back split. The container half held exactly as
   claimed: `noodles_bcf` reads a bcftools-written BCF through this reader with **zero new code**
   in the BGZF crate. The record half did not. BAM's device-side boundary scan relies on htslib
   starting a fresh block rather than splitting an alignment, so a block start is almost always a
   record start — but `bcf_write` packs blocks full, and **0 of 56** interior block boundaries in
   the BCF fixture fall on a record start. Same container, inverted assumption. The replacement
   now runs on device and is verified on a T4. The result worth stating: speculate at every byte
   offset, and if the surviving offsets *tile* the buffer they **are** the record chain, by
   induction — which is O(1) per record and parallel, so BAM's serial reconcile phase disappears
   rather than shrinking. Swept over 399 million candidate offsets on four real files, the
   validator produced **zero false positives**.
   [`docs/bcf-boundaries.md`](docs/bcf-boundaries.md). What is left for BCF is turning those
   boundaries into device-resident columns, as BAM already does.
3. **A CPU reference for every kernel.** It is the correctness oracle: GPU output is diffed
   against it, and it is the only path testable without renting a VM.

Verification is not optional. Every codec must check each block's CRC32 and `ISIZE` and fail on
mismatch — a silently corrupt read in a genomics pipeline is worse than a slow one.

See [CLAUDE.md](CLAUDE.md) for architecture, format invariants, and the remote-GPU workflow.

## License

Our own code is under the [Apache License 2.0](LICENSE).

**Most of this repository is derived from [noodles](https://github.com/zaeleus/noodles) and is
MIT-licensed, © 2018 Michael Macias.** The notice is in
[`LICENSE-MIT-noodles.txt`](LICENSE-MIT-noodles.txt), at the root and again in each crate that
carries vendored code. [VENDORED.md](VENDORED.md) records the upstream commit and the
vendored/ours split per crate.

| | Licence |
|---|---|
| Vendored crates, unmodified | MIT |
| `fritillaria`, `fritillaria-cuda` | Apache-2.0 |
| `fritillaria-{core,bgzf,bam,bcf}` | `Apache-2.0 AND MIT` — they contain both |

MIT code may be redistributed inside an Apache-2.0 work; the reverse is not, so nothing here can
be contributed back upstream to noodles without relicensing it.

Note also that the `nvcomp` feature dlopens NVIDIA's nvCOMP at runtime; it is proprietary,
licensed separately under NVIDIA's own terms, and is neither vendored nor redistributed here.
Building or using that feature means obtaining nvCOMP yourself. Every other path — including the
CUDA fallback codec — is Apache-2.0 and MIT all the way down.
