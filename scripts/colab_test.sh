#!/usr/bin/env bash
# Run the CUDA-gated tests on a fresh Colab GPU VM, then release it.
#
#   ./scripts/colab_test.sh [gpu]        # default: T4 (the verified one)
#
# Provisions a VM, uploads the source, runs the device tests, and stops the VM
# on every exit path including failure and Ctrl-C. A held session bills for
# idle wall-clock time, so nothing here is left running.
#
# `colab run` cannot be used directly: it provisions a bare VM with no way to
# get local files onto it first. Hence new + upload + exec + stop, with the
# session alive only for the duration of the job.

set -euo pipefail

GPU="${1:-T4}"
# Inactivity timeout for remote execution, in seconds (CLI default is 30).
EXEC_TIMEOUT="${EXEC_TIMEOUT:-1800}"
# Which remote job to run. Override to benchmark instead of test.
JOB="${JOB:-scripts/colab_job.py}"
# The remote job's own success marker. Every job must print one, and only its
# successful path may reach it — `colab exec`'s exit code says the kernel ran
# the code, not that the code worked. `colab_job.py` prints `[job] OK`,
# `colab_bench_job.py` prints `[bench] OK`; matching only the former meant a
# *successful* benchmark reported as a failure.
SENTINEL="${SENTINEL:-^\[(job|bench)\] OK}"
# Environment variables forwarded into the remote job. They cannot simply be
# exported: the job runs inside a Jupyter kernel on the VM, which inherits
# nothing from this shell, so setting one here without forwarding it looks like
# it works and silently does nothing.
FORWARD_ENV=(NVCOMP FRITILLARIA_BAM_BYTES FRITILLARIA_BAM_URL FRITILLARIA_SKIP_GPU)
SESSION="fritillaria-test-$$"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL="$(mktemp -t fritillaria-src-XXXXXX.tgz)"
LOG="$(mktemp -t fritillaria-run-XXXXXX.log)"

cleanup() {
    local status=$?
    rm -f "$TARBALL" "$LOG"
    # Best-effort: if allocation never happened, stop is a harmless no-op.
    echo "[colab-test] releasing $SESSION"
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "[colab-test] packaging source"
# testdata/ carries the htslib fixtures the end-to-end tests read.
tar czf "$TARBALL" -C "$ROOT" crates Cargo.toml rust-toolchain.toml scripts testdata

echo "[colab-test] allocating $GPU session $SESSION"
# A 400 here means no quota/entitlement for this accelerator on this account.
colab new -s "$SESSION" --gpu "$GPU"

echo "[colab-test] uploading source"
colab upload -s "$SESSION" "$TARBALL" /content/src.tgz

echo "[colab-test] extracting"
# Via `exec` rather than `console`: console wraps bash in tmux and injects
# terminal-control bytes into piped output, which makes it unparseable.
#
# Sentinel-checked for the same reason the job step below is: `colab exec`
# reports whether the kernel ran the code, not whether the code worked. An
# extraction that failed here would surface much later as a confusing build
# error about missing sources.
# The heredoc binds to `colab exec`, not to `tee`. Written the other way round
# it silently defeats its own check: the script text would go to `tee` instead
# of the VM, and the log would still contain the string `EXTRACT_OK` because
# the source line printing it was echoed. Hence the anchored grep too.
colab exec -s "$SESSION" --timeout "$EXEC_TIMEOUT" <<'PY' 2>&1 | tee "$LOG"
import tarfile, os
os.makedirs('/content/fritillaria', exist_ok=True)
with tarfile.open('/content/src.tgz') as t:
    t.extractall('/content/fritillaria')
print('EXTRACT_OK')
PY

if ! grep -q '^EXTRACT_OK' "$LOG"; then
    echo "[colab-test] FAILED: the source did not extract on the VM" >&2
    exit 1
fi

# Serialised here rather than on the VM so the quoting is Python's problem
# rather than bash's. FRITILLARIA_NVCOMP_LIB is deliberately not forwardable: a
# path valid locally would not exist on the VM, and the job derives its own.
REMOTE_ENV="$(FORWARD="${FORWARD_ENV[*]}" python3 -c '
import json, os
print(json.dumps({k: os.environ[k] for k in os.environ["FORWARD"].split() if k in os.environ}))
')"

echo "[colab-test] running $JOB (env: $REMOTE_ENV)"
# --timeout defaults to 30 SECONDS and is an inactivity timeout on kernel
# output. A cold `cargo build` goes quiet for longer than that while compiling
# a single large crate, so it must be raised or the run dies mid-build.
#
# `colab exec` exits 0 even when the remote script exits non-zero — the status
# it reports is "the kernel ran your code", not "your code succeeded". So its
# exit code is worthless here and `set -e` will not catch a failed test run.
# Observed 2026-09-08: a device test failed on the VM and this script printed
# "passed". The remote job's own success sentinel is the only trustworthy
# signal, so require it explicitly.
#
# `tee` keeps the output streaming (needed for both the inactivity timeout and
# visibility) while still leaving something to inspect afterwards.
# Runs the *uploaded* copy through runpy rather than `-f` on the local file, so
# the environment can be injected first. Same bytes either way — the tarball was
# built from this tree moments ago — and it keeps one source of truth on the VM.
colab exec -s "$SESSION" --timeout "$EXEC_TIMEOUT" <<REMOTE 2>&1 | tee "$LOG"
import os, runpy
os.environ.update($REMOTE_ENV)
runpy.run_path("/content/fritillaria/$JOB", run_name="__main__")
REMOTE

if ! grep -qE "$SENTINEL" "$LOG"; then
    echo "[colab-test] FAILED: the remote job did not report success" >&2
    echo "[colab-test] (no sentinel matching /$SENTINEL/ in its output)" >&2
    exit 1
fi

echo "[colab-test] passed"
