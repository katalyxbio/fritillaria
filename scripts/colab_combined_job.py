#!/usr/bin/env python3
"""Runs the device tests and then the benchmarks, on one rented VM.

Uploaded and invoked by `scripts/colab_test.sh` with `JOB` and `SENTINEL` set:

    JOB=scripts/colab_combined_job.py SENTINEL='^\\[combined\\] OK' \\
        ./scripts/colab_test.sh L4

A VM bills for wall-clock whether or not it is working, and both jobs pay the
same bootstrap — Rust, nvCOMP, samtools, a cold build. Doing them separately
pays it twice.

# The sentinel, and why this file needs its own

`colab_test.sh` decides a run succeeded by finding a positive marker in the
output, because `colab exec` exits 0 even when the remote code fails. The two
jobs print `[job] OK` and `[bench] OK`, and the default pattern matches either.

**That is exactly the failure this project keeps re-learning.** Chained, a
passing test run would put `[job] OK` in the log before the benchmark ran at
all, so a benchmark that then failed would still report as a pass — a sentinel
the failing path can produce. Hence `[combined] OK`, printed only after both
have returned, and `colab_test.sh` must be told to require it.

Ordering is deliberate: tests first, because they are fast and the benchmark
downloads several gigabytes. A broken build should not cost that download.
"""

import runpy
import sys

ROOT = "/content/fritillaria"
JOBS = ("scripts/colab_job.py", "scripts/colab_bench_job.py")


def main():
    for job in JOBS:
        print(f"\n[combined] ===== {job} =====", flush=True)
        # `run_name="__main__"` so each job's own entry point fires. A job that
        # fails calls sys.exit, which raises SystemExit through this loop and
        # stops the next one from running — which is what should happen, and is
        # also what keeps the sentinel below unreachable.
        runpy.run_path(f"{ROOT}/{job}", run_name="__main__")

    print("\n[combined] OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
