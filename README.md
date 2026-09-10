# fritillaria

A Rust library for genomic file formats that delivers parsed records **into GPU memory**, so
GPU-accelerated tools can be built on top of it instead of rewriting BGZF, BAM and BCF parsing
from scratch. Its CPU half is [noodles](https://github.com/zaeleus/noodles), vendored whole, with
a device-resident path added underneath.

The motivating case: read a BAM straight into device memory to feed a GPU-accelerated tool, so
the data never crosses PCIe in its decompressed form and the host never has to hold it. Not only
alignment — variant calling, QC and general pipeline I/O are the same problem, which is why
format coverage matters as much as anything else. A GPU tool that meets an unsupported format
has to fall back to the CPU and round-trip its data, losing the benefit entirely.

**This is about where the data lives, not about a faster decompressor.** If what you want is
bytes in host memory, htslib already does that well and this project does not claim to beat it;
see [Two different questions](#two-different-questions).

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

### Writing

Both write paths ship, and `Auto` resolves them **differently from the read path** on purpose:

| | `Auto` picks | why |
|---|---|---|
| `select_codec` (read) | **GPU** | keeps the output device-resident, which is the point |
| `select_compressor` (write) | **CPU** | GPU compression is slower than multithreaded `bgzip` at comparable output |

The failure mode this library is designed against — reaching for it expecting acceleration and
silently getting CPU speed — inverts on the write side into silently getting something *slower*
than the CPU. So `Auto` will not choose the GPU for you there. `Backend::Nvcomp` is an explicit
act, and `Backend::Cuda` is an error rather than a quiet substitution, because our own kernel
inflates but does not compress.

GPU compression is still the right choice when records are **already in VRAM**, where the CPU
alternative owes a device-to-host copy of the *uncompressed* data first. That case uses
`DeviceBgzfWriter` directly. Output is spec-valid BGZF that `samtools` reads; see
[`docs/compression.md`](docs/compression.md).

## What device residency buys

A conventional pipeline feeding a GPU does four things: read compressed bytes, decompress them on
the CPU, upload the result, then parse it. This does the last three on the device, and the
argument for that is structural rather than a benchmark result.

**PCIe carries compressed bytes instead of decompressed ones.** The compression ratio becomes an
effective bandwidth multiplier on the link — real 1000 Genomes WGS compresses **3.37x**, so the
same link delivers 3.37x the data. This is a property of the file, not of any codec being good.
It does not go away on better hardware, and it is why the architecture is right even where the
kernels are not the fastest part.

**The host never has to hold the decompressed data.** A 3 GiB BAM prefix inflates to 10.11 GiB;
on the device-resident path none of that is allocated host-side or copied back. An early version
did copy it back, and that transfer alone was 54% of runtime — deleted rather than optimised.

**Records arrive parsed and columnar, which is what the next kernel wants.** The comparison that
matters is not against a decompressor, because a decompressor hands back bytes and knows nothing
about a BAM record. nvCOMP does not parse; neither does htslib. The output here is device-side
columns — positions, flags, MAPQ, sequence and quality offsets — indexable straight from a
consumer's kernel, in the caller's own CUDA context, so they are usable in someone else's
pipeline rather than only in ours.

**Verification is never skipped.** Per-block CRC32 and `ISIZE` are checked on device against what
the file itself records, which is stronger than trusting a library's self-check, and costs
nothing extra because it rides along with decompression.

### Two different questions

If you want **bytes in host memory**, use htslib. Multithreaded `bgzip -d` is very good at it, and
getting our output back to the host would mean paying the device-to-host copy this design exists
to delete. **No standalone decompression speedup is claimed anywhere in this project**, and the
same holds on the write path, where multithreaded `bgzip` is faster than the GPU at comparable
output.

If you want **records in device memory**, that is what this is for, and it is a case htslib does
not address at all.

Measurements — phase breakdowns, ratio ladders and the htslib baselines behind those statements —
are in [`docs/`](docs/) rather than here, since they are specific to the hardware they were taken
on.

## Status

20 crates. Eleven are vendored noodles, unchanged. Six hold **both** halves — the vendored CPU
API at the crate root, ours alongside it — and three are entirely ours.

| Crate | What works |
|---|---|
| `fritillaria-core` | *Ours:* errors, `VirtualOffset`, the `BlockCodec`/`DeviceBlockCodec` seams. *Vendored:* `Position`, `Region` |
| `fritillaria-bgzf` | *Ours:* block discovery, CPU codec and compressor, batched and device-resident readers and writers, indexed seek. *Vendored:* `io`, `gzi` |
| `fritillaria-bam` | *Ours* (`columnar`): boundary scan, `RecordBatch`, zero-copy `Record`, aux tags, device columns. *Vendored:* `io`, `bai`, `fs`, `record` |
| `fritillaria-bcf` | *Ours* (`columnar`): header + dictionaries, boundary scan, BCF2 typed values, device columns. *Vendored:* `io`, `fs`, `record` |
| `fritillaria-fastq` | *Ours* (`columnar`): validator, boundary scan, device columns. *Vendored:* `io`, `fai`, `fs`, `record` |
| `fritillaria-fasta` | *Ours* (`columnar`): contig index, newline compaction, `DeviceReference`. *Vendored:* `io`, `fai`, `fs`, `record` |
| `fritillaria-text` | All ours. One line and field scanner for SAM, VCF, BED, GFF and GTF |
| `fritillaria-cuda` | All ours. DEFLATE inflate + CRC32 kernels, device-resident output, nvCOMP codec and compressor, BGZF framing kernel, columnar decode for BAM, BCF and FASTQ, FASTA compaction, text scan |
| `fritillaria` | Facade: re-exports every format, plus backend selection |
| `-sam -vcf -csi -tabix -cram -bed -gff -gtf -util -htsget -refget` | Vendored unchanged |

**Verified on real hardware** — the read and decode paths on an NVIDIA T4, the compression path
on an L4, with the rest of the suite on the host. Correctness rests on differential testing
throughout: GPU output against a CPU reference, our parsers against `samtools`, `bcftools` and
the vendored readers, and — for nvCOMP — against our own kernel as well, so three independent
implementations must agree byte for byte on an htslib-written BAM.

Device tests **skip** rather than fail when no GPU is present, so the remote job treats a skip on
a GPU machine as a failure. Otherwise "nothing ran" would report as success, which it did once.

Fixtures are real files from real writers: htslib, bcftools, NCBI, UCSC, GIAB. Self-generated
files prove only self-consistency, so `testdata/` avoids them.

Indexed access works on the GPU path too: `BgzfReader` implements `bgzf::io::Seek`, so a BAI- or
tabix-driven region query runs on the codec-driven reader rather than falling back to the CPU.

## What's next

1. **Columnar decode for VCF.gz**, on top of the text scan that already finds its fields.
2. **2-bit packing for FASTA**, which is what makes a 3.1 Gbp reference sit in 775 MB rather
   than 3.1 GB. It needs a companion mask for `N` runs and IUPAC codes.
3. **A rebase onto a newer noodles.** Vendoring froze the CPU half at one upstream commit, so
   fixes there do not arrive on their own. The vendored modules are kept byte-for-byte upstream
   apart from the rename precisely to keep that mechanical; the procedure is in
   [VENDORED.md](VENDORED.md) and has not been exercised yet.

Known gaps, stated rather than left to be discovered: a plain `.gz` cannot be block-parallel and
gets a CPU path with no speedup, which the API says explicitly; the text formats get line and
field framing but no field interpretation on device; and the `CG` long-CIGAR encoding is
implemented but tested only against hand-built records, because no real file containing one has
been found yet.

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

### Two GPU codecs, and why both

NVIDIA's **nvCOMP** is the default device path. Decompression is a component to buy rather than a
race to win: it is NVIDIA's own library and the reference answer for GPU DEFLATE, and a genomics
tool that hand-rolled a kernel instead would owe every prospective adopter a justification.

Our **own inflate kernel** ships alongside it, and stays for two reasons that are not
performance. It is the fallback on a machine without nvCOMP, which is proprietary and dlopened
rather than redistributed. And it is a third differential-test oracle: the CPU reference, our
kernel and nvCOMP must all agree byte for byte on an htslib-written BAM, which is a stronger
check than either GPU path alone could give.

Both sit behind the same `BlockCodec`/`DeviceBlockCodec` traits and are subject to the same
mandatory CRC verification, so neither can drift into skipping a check the other makes.

## License

Our own code is under the [Apache License 2.0](LICENSE).

**Most of this repository is derived from [noodles](https://github.com/zaeleus/noodles) and is
MIT-licensed, © 2018 Michael Macias.** The notice is in
[`third-party-licenses/noodles-MIT.txt`](third-party-licenses/noodles-MIT.txt), and again in each
crate that carries vendored code, so it ships with every published crate.
[VENDORED.md](VENDORED.md) records the upstream commit and the vendored/ours split per crate.

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
