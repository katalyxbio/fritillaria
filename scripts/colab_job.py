#!/usr/bin/env python3
"""Remote side of the GPU test run. Executed on the Colab VM.

Uploaded and invoked by `scripts/colab_test.sh` — not run directly.

The VM has no Rust, so this installs a pinned toolchain first. Because every
job starts from a bare VM there is no cargo registry cache and no `target/`,
so this builds cold every time. Keep the device crate's dependency tree small:
anything heavy lands here on every single run.
"""

import hashlib
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

TOOLCHAIN = "1.94"
ROOT = "/content/fritillaria"

# nvCOMP is not on the Colab image and is not redistributable, so every run
# fetches NVIDIA's own redistributable. Pinned by version *and* digest: this
# is a 44 MB proprietary binary we dlopen, and silently running a different
# one than the bindings were written against is exactly the kind of failure
# that shows up as wrong bytes rather than an error.
NVCOMP_VERSION = "5.3.0.16"
NVCOMP_URL = (
    "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/"
    f"linux-x86_64/nvcomp-linux-x86_64-{NVCOMP_VERSION}_cuda12-archive.tar.xz"
)
NVCOMP_SHA256 = "1def6bb0fa51d8ea3fe0c43ae1c58df2f63808ced7444267e14245583ec23f6f"
NVCOMP_DIR = "/content/nvcomp"


def run(cmd, check=True):
    """Runs a shell command and echoes its output.

    Output is captured and re-printed rather than inherited: this runs inside a
    Jupyter kernel, whose stdout capture does not pick up a child process's
    inherited file descriptors. Without this the log shows the commands but
    none of their results.
    """
    print(f"[job] $ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Streamed line by line, not captured and printed at the end: a cold
    # `cargo build` runs for minutes, and `colab exec` times out waiting for
    # output if the kernel stays silent that long.
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()

    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return "".join(lines)


def report_device():
    """Records the hardware. Benchmark numbers are meaningless without it,
    and the VM is gone by the time anyone reads the log."""
    run("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap "
        "--format=csv,noheader", check=False)
    run("nvcc --version | tail -2", check=False)
    # tests/frame_kernel.rs compiles bgzf_frame.cu as ordinary C++ to check its
    # bytes without a GPU. It runs here too, and fails loudly with no compiler —
    # so report one up front rather than have that look like a kernel bug.
    run("c++ --version | head -1", check=False)


def ensure_rust():
    """Installs a pinned Rust toolchain if the VM has none."""
    if subprocess.run("command -v cargo", shell=True).returncode == 0:
        print("[job] cargo already present", flush=True)
    else:
        print("[job] installing Rust (a fresh VM has none)", flush=True)
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            f"| sh -s -- -y --profile minimal --default-toolchain {TOOLCHAIN}")
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]


def ensure_samtools():
    """Installs samtools, so the write path's acceptance bar is actually checked.

    Not on the Colab image. The compression tests treat it as a legitimately
    absent capability and print `NOTE:` rather than failing, which is right —
    but on a GPU VM that would mean the one binary question about our output
    ("does htslib read it?") goes unanswered on the only run that produces
    GPU-compressed bytes. So install it, and say plainly in the log whether it
    is there: a `NOTE:` nobody notices is how a gap becomes permanent.
    """
    if subprocess.run("command -v samtools", shell=True).returncode == 0:
        print("[job] samtools already present", flush=True)
        return True

    print("[job] installing samtools (absent from the Colab image)", flush=True)
    run("apt-get install -y -qq samtools", check=False)

    ok = subprocess.run("command -v samtools", shell=True).returncode == 0
    if ok:
        run("samtools --version | head -2", check=False)
    else:
        # Deliberately not fatal: the GPU work is what the VM is for, and the
        # same bar is cleared locally for the host compressor. But it must be
        # loud, because the tests themselves will only whisper.
        print("[job] WARNING: samtools unavailable — htslib acceptance of the "
              "GPU-compressed BAM went unchecked on this run", flush=True)
    return ok


def ensure_nvcomp():
    """Fetches NVIDIA's nvCOMP redistributable and returns its library path.

    Raises rather than returning None on any failure. The tests treat absent
    nvCOMP as a legitimately missing capability and skip, so a download that
    failed quietly would turn into a green run that never exercised nvCOMP at
    all — the failure mode this project has already hit three times.
    """
    lib = os.path.join(NVCOMP_DIR, "lib", "libnvcomp.so.5")
    if os.path.exists(lib):
        print(f"[job] nvCOMP already present at {lib}", flush=True)
        return lib

    archive = f"/content/nvcomp-{NVCOMP_VERSION}.tar.xz"
    if not os.path.exists(archive):
        print(f"[job] downloading nvCOMP {NVCOMP_VERSION}", flush=True)
        urllib.request.urlretrieve(NVCOMP_URL, archive)

    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != NVCOMP_SHA256:
        raise RuntimeError(
            f"nvCOMP digest mismatch: expected {NVCOMP_SHA256}, "
            f"got {digest.hexdigest()}"
        )

    print("[job] extracting nvCOMP", flush=True)
    os.makedirs(NVCOMP_DIR, exist_ok=True)
    with tarfile.open(archive) as tar:
        # The archive has a single versioned top-level directory; strip it so
        # the library lands at a stable path.
        prefix = f"nvcomp-linux-x86_64-{NVCOMP_VERSION}_cuda12-archive/"
        members = []
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            if member.name:
                members.append(member)
        try:
            tar.extractall(NVCOMP_DIR, members=members, filter="tar")
        except TypeError:
            # `filter` predates neither 3.11.4 nor the Colab image reliably.
            tar.extractall(NVCOMP_DIR, members=members)

    if not os.path.exists(lib):
        raise RuntimeError(f"nvCOMP extracted but {lib} is missing")
    print(f"[job] nvCOMP ready at {lib}", flush=True)
    return lib


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"[job] {ROOT} not found — upload the source first")

    os.chdir(ROOT)
    report_device()
    ensure_rust()
    ensure_samtools()

    # nvCOMP is the intended fast path, so the default run exercises it. Set
    # NVCOMP=0 when iterating on something else and the ~55 MB fetch is pure
    # overhead; the nvcomp tests then skip and the rest still run.
    features = "cuda"
    if os.environ.get("NVCOMP", "1") != "0":
        os.environ["FRITILLARIA_NVCOMP_LIB"] = ensure_nvcomp()
        features = "nvcomp"
    else:
        print("[job] NVCOMP=0, skipping nvCOMP (its tests will report NOTE:)", flush=True)

    # Only the CUDA-gated crate. The CPU paths are already covered locally and
    # rebuilding the whole workspace here costs money for no extra signal.
    #
    # The build is timed and reported separately because a fresh VM has no cargo
    # registry cache and no target/, so this is a genuine cold build and the
    # only place its cost is observable. Vendoring noodles put sam/vcf/csi into
    # this crate's transitive tree, against a standing "keep the device crate
    # thin" constraint; the number below is what decides whether that needs
    # fixing. Timed in Python rather than with /usr/bin/time, which is absent
    # from the Colab image and failed silently once already.
    started = time.monotonic()
    run(f"cargo build -p fritillaria-cuda --features {features} --tests")
    build_seconds = time.monotonic() - started
    print(f"[job] COLD_BUILD_SECONDS {build_seconds:.1f}", flush=True)

    # `--no-fail-fast` because the VM is billable and one rental should yield
    # every result it can. Without it cargo stops at the first failing test
    # *binary*, so a broken unit test in the lib target means the device tests —
    # the entire reason for renting a GPU — never run at all. Observed
    # 2026-09-10: an over-strict FFI assertion failed and the whole compression
    # measurement was lost, at the cost of a full L4 bootstrap.
    #
    # It does not weaken the check: cargo still exits non-zero if anything
    # failed, `run` still aborts on that, and the `[job] OK` sentinel is still
    # only reachable when every test passed.
    output = run(
        f"cargo test -p fritillaria-cuda --features {features} "
        "--no-fail-fast -- --nocapture"
    )

    # Device tests skip when no GPU is present, which is correct locally but a
    # silent failure here — we paid for a GPU VM precisely to exercise them.
    # Without this guard, "0 tests actually ran" reports as success.
    #
    # Matched on the "SKIP:" prefix specifically, not bare "SKIP". A test that
    # cannot run because the *hardware lacks a capability* (two GPUs, say) is a
    # legitimate skip on a correct VM, not a broken driver; those print "NOTE:"
    # and must not fail the run. Keep the two markers distinct.
    if "SKIP:" in output:
        sys.exit("[job] FAIL: device tests skipped on a GPU VM — the driver was not usable")

    print("[job] OK", flush=True)


if __name__ == "__main__":
    main()
