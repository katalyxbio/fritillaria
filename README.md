# fritillaria

**[noodles](https://github.com/zaeleus/noodles) for GPU.** A Rust library for genomic file
formats that delivers parsed records **into GPU memory**, so GPU-accelerated tools can be built
on top of it instead of rewriting BGZF, BAM and BCF parsing from scratch.

The motivating case: read a BAM straight into device memory to feed a GPU-accelerated tool, so
the data never crosses PCIe in its decompressed form. Not only alignment — variant calling, QC
and general pipeline I/O are the same problem, which is why format coverage matters as much as
speed. A GPU tool that meets an unsupported format has to fall back to the CPU and round-trip
its data, losing the benefit entirely.

```rust
use fritillaria::{bgzf::DeviceBgzfReader, cuda::BamDecoder, select_codec, Backend};

let (codec, backend) = select_codec(Backend::Auto)?;   // nvCOMP → our kernel → CPU
let mut reader = DeviceBgzfReader::new(file, &codec);
let decoder = BamDecoder::new()?;

while let Some(batch) = reader.next_batch()? {
    let records = decoder.decode(&batch.data, records_start)?;
    // records.position(), .flags(), .sequence_len() … all still in VRAM
    let tail = records.tail();
    drop(batch);
    reader.carry_from(tail)?;
}
```

## Already using noodles? Adoption is a rename

The CPU half of this library *is* noodles — all 18 crates, vendored and renamed, MIT © 2018
Michael Macias (see [VENDORED.md](VENDORED.md)). Every format it reads, this reads, with the
same API:

```rust
use noodles_bam as bam;   // before
use fritillaria::bam;     // after
```

Putting a GPU underneath is one more line, because `BgzfReader` implements the BGZF traits every
format reader is generic over:

```rust
let mut reader = bam::io::Reader::from(BgzfReader::with_codec(file, CudaCodec::new()?));
```

Device-resident and columnar access are **additive** on top of that, never a replacement. The
vendored readers work unchanged, which is also what keeps a future rebase onto upstream noodles
mechanical.

## Which inputs get the GPU

The dividing line is the **container**, not the format.

BGZF — under BAM, BCF, and anything `bgzip`ped — is a sequence of *independent* gzip members,
each holding at most 64 KiB. Block *i* needs nothing from block *i-1*, so blocks map cleanly
onto parallel hardware. Everything in a BGZF container therefore shares one accelerated path:
block discovery, parallel inflate, CRC verification and virtual offsets are all container-level.

An ordinary `.gz` is not that: one DEFLATE stream with a 32 KiB sliding window, where
back-references make it inherently serial. It cannot be split without a pre-built index.

Every format below **reads and writes on the CPU today**, because that half is noodles.
*Decompression* and *parsing* are separate columns on purpose — one container-level kernel
serves every BGZF format, while parsing has no shared layer and each format needs its own.

| Format | Container | GPU decompression | GPU parsing |
|---|---|---|---|
| BAM (aligned and unaligned) | BGZF | yes | **full** — record scan and columnar decode |
| BCF | BGZF | yes | **full** — boundary scan and site-core columns |
| FASTQ (`bgzip`ped) | BGZF | yes | **full** — boundary scan and columns |
| FASTA | text | — | **compaction** — reference contiguous in VRAM |
| SAM, VCF, BED, GFF, GTF | text | if `bgzip`ped | **framing** — shared line and field scan |
| CRAM | its own | — | — |
| plain `.gz` (any format) | one DEFLATE stream | cannot be block-parallel | — |

Which path an input takes is public API rather than an implementation detail, because silently
delivering CPU speed to someone who came for a GPU is the worst thing this library could do.

BAM is an **input** format as well as an output one: Nanopore and PacBio deliver raw reads as
unaligned BAM, where the basecaller's output lives in aux tags (`MM`/`ML` base modifications,
per-base kinetics) rather than being trailing detail. Those are decoded — every scalar type and
every `B` array subtype, zero-copy, validated tag by tag against `samtools view` on real PacBio
HiFi reads.

Ultra-long reads matter for a second reason: an ONT record can be far larger than a 64 KiB BGZF
block, so records genuinely span blocks and can exceed a whole batch.
`testdata/ont_ultralong.bam` carries a 254 KB record crossing three block boundaries, and the
tests drive the carry-the-partial-record loop a real consumer has to write.

## Performance

Measured over a 3 GiB prefix of a real 1000 Genomes WGS BAM (166,012 blocks, 10.11 GiB inflated,
3.37x ratio) on an NVIDIA L4. Both GPU codecs in the same run, back to back on the same file.
Device-resident, so `download` is just the per-block verification arrays rather than the payload.

| Phase | our kernel | nvCOMP |
|---|---|---|
| upload H2D | 0.86s | 0.89s |
| **inflate + CRC32** | 4.68s | **0.86s** |
| download D2H | 0.003s | 0.003s |
| **sum of phases** | 5.55s | **1.75s** |

**nvCOMP inflates at 11.8 GiB/s of output on an L4** — 5.5x our own kernel, and that includes
the per-block CRC32. Verification is never skipped on either path: the checksum is compared
against what the file itself records, which is stronger than trusting a library self-check.

NVIDIA's nvCOMP is the default fast path; our own inflate kernel is the portable fallback and a
second test oracle. Keeping output on the device deleted a 7.2s device-to-host copy that had
been 54% of runtime — removed rather than optimised.

### Time to records in device memory

The metric the design is actually argued on. Measured end to end on the same L4 and file: a BAM
on disk in, **29,887,809 parsed records in device columns** out, through the shipping path.
Nothing returns to the host but the BAM header.

| Path | Produces | Wall |
|---|---|---|
| **fritillaria + nvCOMP** | 29.9M **parsed records**, columnar, in VRAM | **2.81s** |
| `bgzip -d -@11` | unparsed **bytes** in host RAM | 4.00s |
| `bgzip -@11` then upload | unparsed bytes in VRAM | 4.00s pipelined / 6.97s sequential |

**About 1.4x on wall clock against a 12-core htslib — while delivering something it has not
produced at all.** The CPU path ends with raw bytes and still has every record to parse.

Two things the breakdown shows:

- **Turning bytes into records costs ~7% of the run** — 0.19s against 2.63s of getting the bytes
  there. Decoding on the device is close to free next to moving the data.
- **Batch size barely matters** between 256 and 4096 blocks per batch.

The structural point is the durable one: **PCIe carries compressed bytes instead of decompressed
ones, so the compression ratio becomes an effective bandwidth multiplier on the link** — 3.37x
fewer bytes on real WGS. That does not depend on any codec being good, and does not go away on
better hardware. The host never has to hold the 10.11 GiB at all.

### Two different questions

If you want **bytes in host memory**, use htslib: `bgzip -d -@11` does that in ~3.5s, and
getting our output back to the host means paying the device-to-host copy this design exists to
delete. No standalone decompression speedup is claimed anywhere in this project.

If you want **records in device memory**, that is what this is for, and it is the case htslib
does not address at all.

One caveat on the numbers above: the benchmark VM reports PCIe **gen 1 x16** (~4 GB/s), well
below a real gen 4 link. Upload dominates the nvCOMP path there, so on faster hardware the
balance shifts back toward the codec. The 3.37x ratio advantage is invariant.

## Status

20 crates. Eleven are vendored noodles, unchanged. Six hold **both** halves — the vendored CPU
API at the crate root, ours alongside it — and three are entirely ours.

| Crate | What works |
|---|---|
| `fritillaria-core` | *Ours:* errors, `VirtualOffset`, the `BlockCodec`/`DeviceBlockCodec` seams. *Vendored:* `Position`, `Region` |
| `fritillaria-bgzf` | *Ours:* block discovery, CPU codec, writer, batched and device-resident readers, indexed seek. *Vendored:* `io`, `gzi` |
| `fritillaria-bam` | *Ours* (`columnar`): boundary scan, `RecordBatch`, zero-copy `Record`, aux tags, device columns. *Vendored:* `io`, `bai`, `fs`, `record` |
| `fritillaria-bcf` | *Ours* (`columnar`): header + dictionaries, boundary scan, BCF2 typed values, device columns. *Vendored:* `io`, `fs`, `record` |
| `fritillaria-fastq` | *Ours* (`columnar`): validator, boundary scan, device columns. *Vendored:* `io`, `fai`, `fs`, `record` |
| `fritillaria-fasta` | *Ours* (`columnar`): contig index, newline compaction, `DeviceReference`. *Vendored:* `io`, `fai`, `fs`, `record` |
| `fritillaria-text` | All ours. One line and field scanner for SAM, VCF, BED, GFF and GTF |
| `fritillaria-cuda` | All ours. DEFLATE inflate + CRC32 kernels, device-resident output, nvCOMP codec, columnar decode for BAM, BCF and FASTQ, FASTA compaction, text scan |
| `fritillaria` | Facade: re-exports every format, plus backend selection |
| `-sam -vcf -csi -tabix -cram -bed -gff -gtf -util -htsget -refget` | Vendored unchanged |

**Verified on real hardware.** 80 device tests pass on an NVIDIA T4 and 2,496 on the host.
Correctness is established by differential testing throughout: GPU output against a CPU
reference, our parsers against `samtools`, `bcftools` and the vendored noodles readers, and —
for nvCOMP — against our own kernel as well, so three independent implementations must agree
byte for byte on an htslib-written BAM.

Fixtures are real files from real writers: htslib, bcftools, NCBI, UCSC, GIAB. Self-generated
files prove only self-consistency, so `testdata/` avoids them.

Indexed access works on the GPU path too: `BgzfReader` implements `bgzf::io::Seek`, so a BAI- or
tabix-driven region query runs on the codec-driven reader rather than falling back to the CPU.

## What's next

1. **GPU-side BGZF compression**, so a tool that produces records on-device can write them back
   without paying the transfer the read path removes. The first step is a measurement rather
   than a kernel: nvCOMP's Deflate compress ratio against `libdeflate`, since a fast compressor
   that produced bigger files would undercut the link advantage the design rests on.
2. **Columnar decode for VCF.gz**, on top of the text scan that already finds its fields.
3. **2-bit packing for FASTA**, which is what makes a 3.1 Gbp reference sit in 775 MB rather
   than 3.1 GB. It needs a companion mask for `N` runs and IUPAC codes.

CRAM is not currently planned: it is its own container with its own codecs, so little of this
stack applies to it.

## Building

```bash
cargo build                       # host only; no CUDA toolkit needed
cargo build --features cuda       # compiles without nvcc — kernels are NVRTC-compiled at runtime
cargo build --features nvcomp     # compiles without nvCOMP installed — it is dlopened, never linked
cargo test                        # all CPU paths, including the vendored suites
cargo test -p fritillaria-cuda    # our code only, the fast inner loop
```

Both GPU features build on a machine with no GPU and no toolkit: cudarc dlopens `libcuda`,
kernels are compiled at runtime by NVRTC, and nvCOMP is dlopened rather than linked. Only
*running* device code needs hardware, and device tests skip rather than fail without it.

`sm_75` (Turing) is the verified floor.

## Design

The pipeline decomposes into independently testable stages:

1. **Block discovery** — walk BGZF headers to collect `(offset, size)` pairs. Sequential and
   I/O-bound; batched so the GPU gets enough work per launch.
2. **Decompression** — one thread block per BGZF block, into a device buffer, with per-block
   output offsets from a prefix sum over `ISIZE`.
3. **Verification** — CRC32 and `ISIZE` per block, on device, never skipped by default. A silent
   corrupt read in a genomics pipeline is worse than being slower.
4. **Boundary discovery** — records are variable-length and span blocks, so this runs over the
   concatenated buffer. Each format needs its own approach, and they do not generalise to one
   another.
5. **Field decode** — with boundaries known, records are embarrassingly parallel. Output is
   **columnar**, which is what a GPU consumer wants next anyway.

The load-bearing rule: **device memory and streams live in `fritillaria-cuda` and nowhere else.**
No raw pointer or device-tied lifetime escapes into the format crates — they depend on a trait
in `fritillaria-core` and never name a concrete backend. That is what lets a format crate hand
back device-resident columns while remaining compilable with no GPU in sight.

`CudaCodec::with_context` allocates in a **caller-supplied** context, and `ready_event` lets a
consumer on its own stream order against inflate completion without a host synchronise. A
library that insisted on owning the context could not be embedded in someone else's pipeline.

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
| `fritillaria`, `fritillaria-cuda`, `fritillaria-text` | Apache-2.0 |
| `fritillaria-{core,bgzf,bam,bcf,fastq,fasta}` | `Apache-2.0 AND MIT` — they contain both |

MIT code may be redistributed inside an Apache-2.0 work; the reverse is not, so nothing here can
be contributed back upstream to noodles without relicensing it.

The `nvcomp` feature dlopens NVIDIA's nvCOMP at runtime. It is proprietary, licensed separately
under NVIDIA's own terms, and is neither vendored nor redistributed here — using that feature
means obtaining nvCOMP yourself. Every other path, including the CUDA fallback codec, is
Apache-2.0 and MIT all the way down.
