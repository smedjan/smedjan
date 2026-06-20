#!/usr/bin/env bash
# Poll origin/main; when it advances, run the Mac CI gate (ci-mac.sh) and record the verdict.
# Invoked on a schedule by launchd on an always-on Apple-Silicon Mac (mini). Runs against a
# DEDICATED checkout (ANDREAI_CI_REPO), never a human's working tree, so the gate's hard-reset
# to origin/main can't clobber local dev work. Marks each tested commit (pass OR fail) so it
# doesn't re-run the same commit every interval.
#
#   ANDREAI_CI_REPO    dedicated CI clone (default: $HOME/ci/andreai)
#   ANDREAI_CI_STATE   last-tested-commit marker file (default: $HOME/.andreai-ci-last)
#   ANDREAI_CI_REPORT  latest run report (default: $HOME/.andreai-ci-report.txt)
#   ANDREAI_CI_POLL_LOCK  host-wide poller lock dir (default: /tmp/andreai-mac-gpu-ci-poll.lock)
set -uo pipefail

CI_REPO="${ANDREAI_CI_REPO:-$HOME/ci/andreai}"
STATE="${ANDREAI_CI_STATE:-$HOME/.andreai-ci-last}"
REPORT="${ANDREAI_CI_REPORT:-$HOME/.andreai-ci-report.txt}"
POLL_LOCK="${ANDREAI_CI_POLL_LOCK:-/tmp/andreai-mac-gpu-ci-poll.lock}"

mkdir -p "$(dirname "$POLL_LOCK")" || { echo "FAIL: cannot create poll lock parent"; exit 2; }
while ! mkdir "$POLL_LOCK" 2>/dev/null; do
  owner=$(cat "$POLL_LOCK/pid" 2>/dev/null || true)
  if [[ -z "$owner" ]] || ! kill -0 "$owner" 2>/dev/null; then
    rm -rf "$POLL_LOCK"
    continue
  fi
  echo "SKIP: another andreai CI poller owns $POLL_LOCK"
  exit 0
done
echo "$$" > "$POLL_LOCK/pid"
trap 'rm -rf "$POLL_LOCK"' EXIT INT TERM

cd "$CI_REPO" 2>/dev/null || { echo "FAIL: CI checkout $CI_REPO missing (clone origin there first)"; exit 2; }
git fetch --quiet origin || { echo "FAIL: git fetch"; exit 2; }

REMOTE=$(git rev-parse origin/main)
LAST=$(cat "$STATE" 2>/dev/null || echo none)
if [[ "$REMOTE" == "$LAST" ]]; then
  exit 0   # nothing new since the last tested commit
fi

ts=$(date +%Y-%m-%dT%H:%M:%S)
echo "[$ts] origin/main advanced to ${REMOTE:0:9} (was ${LAST:0:9}) — running Mac CI gate"
ANDREAI_REPO="$CI_REPO" ANDREAI_CI_PULL=1 "$CI_REPO/scripts/ci-mac.sh" > "$REPORT" 2>&1
rc=$?
tail -10 "$REPORT"
echo "$REMOTE" > "$STATE"   # record tested (pass or fail) so we don't loop on the same commit
if [[ $rc == 0 ]]; then echo "CI PASS ${REMOTE:0:9}"; else echo "CI FAIL ${REMOTE:0:9} (see $REPORT)"; fi
exit $rc
