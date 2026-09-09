#!/usr/bin/env python3
"""Benchmark BGZF decompression on real WGS data. Runs on the Colab VM.

Invoked via `JOB=scripts/colab_bench_job.py ./scripts/colab_test.sh [gpu]`.

Downloads a prefix of a public 1000 Genomes BAM rather than a whole one: the
file supports HTTP range requests, throughput is measured in GiB/s so a few GiB
is statistically identical to a hundred, and download time is billable VM time.
A prefix is a valid BAM header followed by real records, and block discovery
stops cleanly at the truncated final block.

Public reference data on purpose — no consent or controlled-access question,
and anyone can reproduce the number.

Two things this measures that a single machine otherwise cannot:

1. **The htslib baseline.** `bgzip -d` is exactly our workload (BGZF decompress,
   no record parsing), from the reference implementation, on the same hardware
   and the same file. Single- and multi-threaded. This is the only honest
   comparison; our `miniz_oxide` reference is a correctness oracle, not a
   performance target.
2. **PCIe link details.** D2H measured only 1.45 GB/s on a T4, well under what
   pageable PCIe 3.0 x16 should manage. Running on a second GPU with a
   different link tells us whether that is a bandwidth limit (fix: keep output
   device-resident) or a software one (fix: pinned/reused host buffers).

Override with FRITILLARIA_BAM_URL / FRITILLARIA_BAM_BYTES.
"""

import os
import subprocess
import sys
import urllib.request

TOOLCHAIN = "1.94"
ROOT = "/content/fritillaria"

# The nvCOMP bootstrap is shared with the test job rather than duplicated — one
# pinned version and one digest, checked in one place.
#
# Imported by path, not relative to `__file__`: `colab exec -f` sends this
# file's *contents* to a Jupyter kernel, where `__file__` is undefined. The
# uploaded source tree is the reliable anchor.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from colab_job import ensure_nvcomp  # noqa: E402  (path set up just above)

# The AWS mirror of the same 1000 Genomes file, not EBI's FTP. Identical bytes
# and the same public dataset, but EBI served this at **0.2 MiB/s** on
# 2026-09-09 — 85 minutes for a 1 GiB prefix, all of it billable VM time — while
# S3 sustained orders of magnitude more. The mirror is a throughput decision
# only; anyone can still reproduce the number from either.
BAM_URL = os.environ.get(
    "FRITILLARIA_BAM_URL",
    "https://1000genomes.s3.amazonaws.com/phase3/data/HG00096/"
    "alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam",
)
BAM_BYTES = int(os.environ.get("FRITILLARIA_BAM_BYTES", 3 * 1024**3))
BAM_PATH = "/content/wgs_prefix.bam"



# The 28-byte empty block every BGZF file must end with.
EOF_BLOCK = bytes([
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00,
    0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
])


def fetch_prefix(url, path, nbytes):
    """Downloads a byte-range prefix, printing progress as it goes.

    Progress is not a nicety here. `colab exec --timeout` is an **inactivity**
    timeout on kernel output, so a silent multi-minute download is
    indistinguishable from a hung job and gets killed once the mirror is slow
    enough — which is exactly what happened on 2026-09-09, when a working
    download from EBI was about to be cut off at the 50-minute mark having
    printed nothing since it started. Emitting a line per chunk keeps the timer
    fed and makes a slow mirror visible instead of fatal.

    Streamed in chunks rather than shelling out to curl so the reporting is ours
    and a partial file is never left behind to be mistaken for a whole one.
    """
    import time as _time

    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    started = _time.monotonic()
    written = 0
    step = 256 * 1024**2
    next_report = step
    tmp = path + ".partial"

    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        while written < nbytes:
            chunk = response.read(min(8 * 1024**2, nbytes - written))
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if written >= next_report:
                elapsed = _time.monotonic() - started
                print(f"[bench]   {written / 1024**3:.2f} GiB "
                      f"({written / elapsed / 1024**2:.1f} MiB/s)", flush=True)
                next_report += step

    if written < nbytes:
        os.remove(tmp)
        raise RuntimeError(
            f"short download: got {written} of {nbytes} bytes. A truncated "
            f"prefix would still parse, so this must fail loudly rather than "
            f"silently benchmark a smaller file than reported."
        )
    os.replace(tmp, path)
    elapsed = _time.monotonic() - started
    print(f"[bench] fetched {written / 1024**3:.2f} GiB in {elapsed:.0f}s", flush=True)


def repair_bgzf(path):
    """Turns a byte-range prefix into a valid BGZF stream.

    A raw prefix ends mid-block and has no EOF marker. Our reader tolerates
    that by design, but `bgzip` rightly refuses it — so without this the two
    sides of the comparison are not given the same input, and the baseline
    silently fails. Trim to the last complete block and append the EOF block.
    """
    size = os.path.getsize(path)
    pos = 0
    with open(path, "rb") as f:
        while pos + 18 <= size:
            f.seek(pos)
            head = f.read(12)
            if len(head) < 12 or head[0] != 0x1F or head[1] != 0x8B:
                break
            xlen = int.from_bytes(head[10:12], "little")
            extra = f.read(xlen)
            if len(extra) < xlen:
                break
            bsize = None
            i = 0
            while i + 4 <= xlen:
                slen = int.from_bytes(extra[i + 2:i + 4], "little")
                if extra[i] == 0x42 and extra[i + 1] == 0x43 and slen == 2:
                    bsize = int.from_bytes(extra[i + 4:i + 6], "little") + 1
                    break
                i += 4 + slen
            if bsize is None or pos + bsize > size:
                break
            pos += bsize

    with open(path, "r+b") as f:
        f.truncate(pos)
        f.seek(pos)
        f.write(EOF_BLOCK)
    print(f"[bench] repaired: {size} -> {pos + len(EOF_BLOCK)} bytes "
          f"(trimmed {size - pos} partial, added EOF block)", flush=True)


def run(cmd, check=True):
    """Runs a command, streaming output so `colab exec` does not time out."""
    print(f"[bench] $ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return "".join(lines)


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"[bench] {ROOT} not found")
    os.chdir(ROOT)

    print("=" * 68, flush=True)
    print("HARDWARE (a throughput number without this is meaningless)", flush=True)
    print("=" * 68, flush=True)
    run("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,"
        "pcie.link.gen.current,pcie.link.width.current --format=csv", check=False)
    run("nproc && free -g | awk '/Mem:/{print $2\" GiB RAM\"}' "
        "&& lscpu | grep -E '^Model name' | head -1", check=False)

    if subprocess.run("command -v cargo", shell=True).returncode != 0:
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            f"| sh -s -- -y --profile minimal --default-toolchain {TOOLCHAIN}")
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]

    # Build while the download runs would be nicer, but keep it simple and
    # sequential — the whole job is a few minutes.
    if not os.path.exists(BAM_PATH):
        print(f"\n[bench] fetching {BAM_BYTES / 1024**3:.1f} GiB prefix", flush=True)
        fetch_prefix(BAM_URL, BAM_PATH, BAM_BYTES)
        repair_bgzf(BAM_PATH)
    run(f"ls -la {BAM_PATH}")

    if os.environ.get("FRITILLARIA_SKIP_GPU"):
        print("\n[bench] FRITILLARIA_SKIP_GPU set — baseline only", flush=True)
    else:
        # nvCOMP is the whole point of this run: the example benchmarks both
        # GPU codecs back to back over the same file, which is the only fair
        # way to compare them. Absent nvCOMP the example says so and reports
        # our kernel alone, so a failed fetch cannot pass as a comparison.
        features = "cuda"
        if os.environ.get("NVCOMP", "1") != "0":
            os.environ["FRITILLARIA_NVCOMP_LIB"] = ensure_nvcomp()
            features = "nvcomp"

        # Release profile: a debug-build inflate benchmark measures nothing.
        run(f"cargo build --release --features {features} -p fritillaria-cuda "
            "--example bench_inflate --example bench_decode")

        print("\n" + "=" * 68, flush=True)
        print("FRITILLARIA (GPU) — decompression only", flush=True)
        print("=" * 68, flush=True)
        run(f"cargo run --release --quiet --features {features} -p fritillaria-cuda "
            f"--example bench_inflate -- {BAM_PATH} 64")

        # The metric the design is actually argued on. Kept separate from the
        # inflate benchmark rather than folded into it, because the two answer
        # different questions and quoting one as the other is the mistake this
        # project keeps warning itself about.
        print("\n" + "=" * 68, flush=True)
        print("FRITILLARIA (GPU) — time to RECORDS in device memory", flush=True)
        print("=" * 68, flush=True)
        run(f"cargo run --release --quiet --features {features} -p fritillaria-cuda "
            f"--example bench_decode -- {BAM_PATH}")

    print("\n" + "=" * 68, flush=True)
    print("HTSLIB BASELINE (bgzip -d: same workload, same file, same machine)", flush=True)
    print("=" * 68, flush=True)
    if subprocess.run("command -v bgzip", shell=True).returncode != 0:
        run("apt-get install -y -qq tabix >/dev/null 2>&1 || "
            "apt-get install -y -qq samtools >/dev/null 2>&1", check=False)

    if subprocess.run("command -v bgzip", shell=True).returncode != 0:
        print("[bench] bgzip unavailable — no htslib baseline this run", flush=True)
    else:
        run("bgzip --version | head -1", check=False)
        threads = max(1, os.cpu_count() or 1)
        # `bgzip -d` decompresses BGZF and nothing else, which is exactly the
        # workload bench_inflate measures. Discard output so disk write speed
        # does not enter the measurement.
        # Timed in Python, not with /usr/bin/time: the Colab image does not
        # ship it, and `time` as a bash builtin writes to stderr in a way that
        # is easy to lose. Silent failure here cost a whole run once.
        import time as _time

        size = os.path.getsize(BAM_PATH)
        for n in dict.fromkeys([1, max(1, threads - 1)]):
            print(f"\n-- bgzip -d -@ {n} --", flush=True)
            started = _time.monotonic()
            code = subprocess.run(
                f"bgzip -d -@ {n} -c {BAM_PATH} > /dev/null", shell=True
            ).returncode
            elapsed = _time.monotonic() - started
            if code != 0:
                print(f"  FAILED (exit {code})", flush=True)
                continue
            print(f"  wall {elapsed:.2f}s   "
                  f"{size / elapsed / 1024**2:.0f} MiB/s of compressed input",
                  flush=True)

    print("\n[bench] OK", flush=True)


if __name__ == "__main__":
    main()
