#!/usr/bin/env bash
# Mac GPU CI gate for andreai (Metal). Runs the §5 verification protocol plus the
# buffer-pool sanitizer suite — the GPU-correctness gate that the host-only workflow
# previously lacked. Intended to run on an always-on Apple-Silicon Mac (mini) via
# launchd/cron, or by hand on air. Exits non-zero with a one-line summary on any failure.
#
#   ANDREAI_REPO      repo path        (default: $HOME/projects/andreai)
#   ANDREAI_CI_PULL   1 = fetch+reset to origin/main before testing (default: 0 = test tree)
#   ANDREAI_CI_LOG    per-gate log dir (default: /tmp/andreai-ci)
set -uo pipefail

REPO="${ANDREAI_REPO:-$HOME/projects/andreai}"
PULL="${ANDREAI_CI_PULL:-0}"
FEAT=(--no-default-features --features metal)
LOG_DIR="${ANDREAI_CI_LOG:-/tmp/andreai-ci}"
mkdir -p "$LOG_DIR"
ts() { date +%Y-%m-%dT%H:%M:%S; }
cd "$REPO" || { echo "FAIL: repo $REPO not found"; exit 2; }

if [[ "$PULL" == "1" ]]; then
  git fetch --quiet origin && git reset --hard --quiet origin/main \
    || { echo "FAIL: git sync to origin/main"; exit 2; }
fi
HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "== andreai Mac CI @ $(ts) — $HEAD =="

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
gate unit       cargo test   "${FEAT[@]}" -- --test-threads=1
gate serial_gpu cargo test   "${FEAT[@]}" -- --include-ignored --test-threads=1
gate clippy     cargo clippy "${FEAT[@]}" --all-targets -- -D warnings
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
