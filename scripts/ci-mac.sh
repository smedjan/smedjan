#!/usr/bin/env bash
# Mac GPU CI gate for smedjan (Metal). Runs the §5 verification protocol, formatting
# gates, CUDA compile parity, train smokes, and the buffer-pool sanitizer suite — the
# GPU-correctness gate that the host-only workflow previously lacked. Intended to run
# on an always-on Apple-Silicon Mac (mini) via launchd/cron, or by hand on air. Exits
# non-zero with a one-line summary on any failure.
#
#   SMEDJAN_REPO      repo path        (default: $HOME/projects/smedjan)
#   SMEDJAN_CI_PULL   1 = fetch+reset to origin/main before testing (default: 0 = test tree)
#   SMEDJAN_CI_LOG    per-gate log dir (default: /tmp/smedjan-ci)
#   SMEDJAN_CI_LOCK   host-wide GPU lock dir (default: /tmp/smedjan-mac-gpu-ci.lock)
set -uo pipefail
export CARGO_INCREMENTAL=0

REPO="${SMEDJAN_REPO:-$HOME/projects/smedjan}"
PULL="${SMEDJAN_CI_PULL:-0}"
FEAT=(--no-default-features --features metal)
LOG_DIR="${SMEDJAN_CI_LOG:-/tmp/smedjan-ci}"
LOCK_DIR="${SMEDJAN_CI_LOCK:-/tmp/smedjan-mac-gpu-ci.lock}"
LOCK_WAIT_SECS="${SMEDJAN_CI_LOCK_WAIT_SECS:-14400}"
mkdir -p "$LOG_DIR"
ts() { date +%Y-%m-%dT%H:%M:%S; }
cd "$REPO" || { echo "FAIL: repo $REPO not found"; exit 2; }

acquire_lock() {
  local start now elapsed owner
  start=$(date +%s)
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [[ -z "$owner" ]]; then
      sleep 1
      owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    fi
    if [[ -z "$owner" ]] || ! kill -0 "$owner" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      continue
    fi
    now=$(date +%s)
    elapsed=$((now - start))
    if (( elapsed >= LOCK_WAIT_SECS )); then
      echo "FAIL: timed out waiting for Mac GPU CI lock $LOCK_DIR"
      exit 2
    fi
    echo "WAIT: another Mac GPU CI run owns $LOCK_DIR; retrying in 10s"
    sleep 10
  done
  echo "$$" > "$LOCK_DIR/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
}

acquire_lock

if [[ "$PULL" == "1" ]]; then
  git fetch --quiet origin && git reset --hard --quiet origin/main \
    || { echo "FAIL: git sync to origin/main"; exit 2; }
fi
HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "== smedjan Mac CI @ $(ts) — $HEAD =="

NAMES=(); RESULTS=()
gate() { # name  command...
  local name=$1; shift
  echo "---- $name ----"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "PASS: $name"; NAMES+=("$name"); RESULTS+=(PASS)
  else
    echo "FAIL: $name (see $LOG_DIR/$name.log)"; tail -30 "$LOG_DIR/$name.log"
    NAMES+=("$name"); RESULTS+=(FAIL)
  fi
}

# §5 protocol
# Metal tests share a real GPU and thread-local backend flags/caches. Run GPU gates serially;
# default rust-test parallelism has produced false failures in exact numeric comparisons.
gate fmt        cargo fmt --check
gate diffcheck  git diff --check
gate unit       cargo test   "${FEAT[@]}" -- --test-threads=1
gate clippy     cargo clippy --no-default-features --features metal,bufsan --all-targets -- -D warnings
gate cuda_check cargo check  --no-default-features --features cuda
gate train_smoke ./scripts/train-smoke.sh
# Phase B sanitizer: whole suite under NaN-poison + a quarantine differential
gate bufsan     cargo test   --no-default-features --features metal,bufsan -- --test-threads=1

echo "== summary @ $(ts) — $HEAD =="
fail=0
for i in "${!NAMES[@]}"; do
  printf "  %-12s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"
  [[ "${RESULTS[$i]}" == FAIL ]] && fail=1
done
if [[ $fail == 0 ]]; then echo "ALL GATES PASS ($HEAD)"; else echo "CI FAILED ($HEAD)"; fi
exit $fail
