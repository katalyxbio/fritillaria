# fritillaria

**[noodles](https://github.com/zaeleus/noodles) for GPU.** A Rust-native library for genomic
file formats that puts records **in GPU memory**, so GPU-accelerated tools can be built on top
of it instead of rewriting BGZF, BAM and BCF parsing from scratch.

The motivating case: decompress a BAM directly into device memory to feed a GPU-accelerated
tool, so the data never crosses PCIe in its decompressed form. Not just alignment — variant
calling, QC and general pipeline I/O are the same problem, which is why format coverage matters
as much as speed. A GPU tool that hits an unsupported format has to fall back to the CPU and
round-trip its data, losing the entire benefit.

**Already using noodles? Adoption is one line.** `BgzfReader` implements the BGZF traits every
noodles format crate is generic over, so GPU decompression slots underneath your existing code
with its record parsing untouched. Device-resident and columnar access are additive on top.

**This is not trying to be a faster `samtools`.** Against `bgzip` as a standalone decompressor
we lose, and that is the wrong comparison: htslib produces bytes in host RAM, which is not where
a GPU consumer needs them. The comparison that matters is *time to records in device memory*,
where PCIe carries compressed bytes instead of decompressed ones — a 3.37x reduction in link
traffic on real WGS data. See *Performance*.

> **Early, and honest about it.** GPU decompression runs end to end and is verified against
> htslib-written files. Output is still copied back to the host, which is the single biggest
> thing left to fix.

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

| Format | Container | Path |
|---|---|---|
| BAM (aligned and unaligned), BCF, `bgzip`ped VCF/FASTQ, tabix-indexed files | BGZF | GPU block-parallel |
| SAM, BED, GFF/GTF, FASTA | text | CPU |
| plain `.gz` (any format) | one DEFLATE stream | CPU — cannot be block-parallel |

Only BAM is implemented so far; the rest is the roadmap, not a promise.

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

| Crate | What works |
|---|---|
| `fritillaria-core` | Types, errors, `VirtualOffset`, the `BlockCodec` and `DeviceBlockCodec` seams |
| `fritillaria-bgzf` | Block discovery, CPU codec, writer, batched `BgzfReader` |
| `fritillaria-bam` | Header, record boundary scan, columnar `RecordBatch`, zero-copy `Record`, aux tags |
| `fritillaria-cuda` | DEFLATE inflate + CRC32 kernels, device-resident output, nvCOMP codec — verified on a Tesla T4 |
| `fritillaria` | Facade and backend selection (nvCOMP → our kernel → CPU) |

Two GPU codecs sit behind the same traits with the same mandatory verification, so choosing
between them is a performance decision rather than a semantic one. NVIDIA's **nvCOMP** is the
fast path; our own inflate kernel is the portable fallback and a second test oracle. It covers
all three DEFLATE block types (stored, fixed and dynamic Huffman), one thread per BGZF block,
with CRC32 folded in so verification costs no extra pass.

Correctness is checked by differential testing against `miniz_oxide` on real hardware — and for
nvCOMP against our kernel as well, so three independent implementations have to agree
byte-for-byte on an htslib-written BAM.

**noodles works on top of this, unforked.** `BgzfReader` implements the BGZF reader traits that
every noodles format crate is generic over, so `noodles_bam::io::Reader` reads a real htslib BAM
whose blocks were inflated on the GPU — verified on a T4, with GPU and CPU backends producing
identical records. The same substitution should accelerate BCF, `bgzip`ped VCF, and tabix-indexed
formats without forking a line of noodles.

```rust
use fritillaria_bgzf::BgzfReader;
use noodles_bam as bam;

// Same noodles API, GPU decompression underneath.
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

If the consumer is a GPU kernel, the CPU path is not finished when it finishes decompressing:
it still has to upload 10.11 GiB. Upload figures use our own measured H2D rate of 3.48 GiB/s;
"pipelined" is the max of the two phases, which is the fair reading for both sides.

| Path | Work | Pipelined |
|---|---|---|
| `bgzip -@11` → H2D | decompress 3.48s, upload 10.11 GiB (2.90s) | **3.48s** |
| fritillaria, our kernel | upload 0.86s, inflate 4.68s | 4.68s |
| **fritillaria + nvCOMP** | upload 0.89s, inflate 0.86s | **0.89s** |

**3.9x ahead of a 12-core htslib on the metric that matters for a GPU consumer.**

The structural point is the durable one: **PCIe carries compressed bytes instead of decompressed
ones, so the compression ratio becomes an effective bandwidth multiplier on the link** — 3.37x
fewer bytes on real WGS. That does not depend on any codec being good and does not go away on
better hardware. The host also never has to hold the 10.11 GiB at all.

### As a standalone decompressor, htslib still wins

Same machine, same run, same file. `bgzip -d` on 11 threads does the job in **3.48s**; getting
our output back into host RAM means paying the device-to-host copy this whole design exists to
delete, which puts the GPU path behind it.

| | wall | MiB/s of compressed input |
|---|---|---|
| `bgzip -d -@ 1` | 24.24s | 127 |
| **`bgzip -d -@ 11`** | **3.48s** | **882** |

So there are two different numbers here and they answer different questions. If you want bytes
in host memory, use htslib. If you want records in device memory, that is what this is for.
**No standalone decompression speedup is claimed anywhere in this project.**

Caveat on all of the above: Colab reports PCIe **gen 1 x16** (~4 GB/s), well below a real gen 4
link. Upload dominates the nvCOMP path here, so on faster hardware the balance shifts back
toward the codec. The 3.37x ratio advantage is invariant.

### What's next

Decompression is the on-ramp, not the product — and with inflate now balanced against the
upload, further codec work buys little. The effort belongs downstream of it:

1. **Columnar record decode on-device.** Turning bytes in VRAM into *records* in VRAM is the
   actual deliverable, and the part neither nvCOMP nor any raw codec addresses.
2. **BCF and `bgzip`ped VCF**, which should be close to free through the noodles seam, then
   `bgzip`ped FASTQ.
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
   and `bgzip`ped VCF come before text formats: they are nearly free.
3. **A CPU reference for every kernel.** It is the correctness oracle: GPU output is diffed
   against it, and it is the only path testable without renting a VM.

Verification is not optional. Every codec must check each block's CRC32 and `ISIZE` and fail on
mismatch — a silently corrupt read in a genomics pipeline is worse than a slow one.

See [CLAUDE.md](CLAUDE.md) for architecture, format invariants, and the remote-GPU workflow.

## License

MIT OR Apache-2.0
