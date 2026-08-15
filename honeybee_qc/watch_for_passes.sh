#!/usr/bin/env bash
# Poll a chunked run's reports as they land and flag any task that passes.
#
# Two things count as a pass, and they are labelled differently because they
# mean different things:
#
#   genuine pass   the task's own verdict is clean or pass_with_issues.
#   excluded pass  the task failed, but nothing it failed is in the check set
#                  being watched -- so under that restricted scope it passes.
#                  Defaults to the CLI's RESTRICTED_CHECKS (imported, not
#                  copied, so the two cannot drift).
#
# Unauditable tasks are never a pass. Each hit prints one PASS_FOUND line so a
# caller can notify on that pattern, and each task is reported once: the ledger
# in --seen-file keeps a restart from re-announcing everything.
#
# Reports are opened read-only and nothing is written inside the run dir, so
# this is safe to point at a run that is still going.
#
# Usage: honeybee_qc/watch_for_passes.sh RUN_DIR [options]
#
#   --checks 270,280,...   checks that count as a fail (default: RESTRICTED_CHECKS)
#   --done-log PATH        log to watch for the run's completion line; relative
#                          paths resolve inside RUN_DIR. Without it, polls until
#                          interrupted.
#   --done-pattern TEXT    the completion line to grep for (required with --done-log)
#   --interval SECONDS     poll interval (default: 60)
#   --seen-file PATH       the reported-tasks ledger. Defaults to a per-run file
#                          under /tmp that is cleared on startup; naming one
#                          instead keeps the ledger across restarts.
#   --once                 make a single pass and exit, without polling
#
# Example:
#   honeybee_qc/watch_for_passes.sh honeybee_qc/audit_runs/honeybee_l1_full_20260814_batch6 \
#     --done-log run_batch6_parallel.log --done-pattern "BATCH6 RUN COMPLETE"
set -uo pipefail

usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; }

RUN_DIR=""
CHECKS=""
DONE_LOG=""
DONE_PATTERN=""
INTERVAL=60
SEEN_FILE=""
ONCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --checks) CHECKS="${2:-}"; shift 2 ;;
    --done-log) DONE_LOG="${2:-}"; shift 2 ;;
    --done-pattern) DONE_PATTERN="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --seen-file) SEEN_FILE="${2:-}"; shift 2 ;;
    --once) ONCE=1; shift ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$RUN_DIR" ]; then
        echo "unexpected argument: $1 (only one run dir)" >&2
        exit 2
      fi
      RUN_DIR="$1"; shift ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  echo "a run directory is required" >&2
  usage >&2
  exit 2
fi

# Absolute before the cd below, so a run dir relative to the caller's cwd works.
if ! RESOLVED="$(cd "$RUN_DIR" 2>/dev/null && pwd)"; then
  echo "run dir not found: $RUN_DIR" >&2
  exit 1
fi
RUN_DIR="$RESOLVED"

if [ -n "$DONE_LOG" ] && [ -z "$DONE_PATTERN" ]; then
  echo "--done-log needs --done-pattern: what line means finished?" >&2
  exit 2
fi
case "$DONE_LOG" in
  ""|/*) ;;
  *) DONE_LOG="$RUN_DIR/$DONE_LOG" ;;
esac

cd "$(dirname "$0")/.." || exit 1  # repo root, so `import honeybee_qc` resolves

if [ -z "$CHECKS" ]; then
  CHECKS="$(python3 -c 'from honeybee_qc.cli import RESTRICTED_CHECKS
print(",".join(str(c) for c in RESTRICTED_CHECKS))')" || {
    echo "could not read RESTRICTED_CHECKS; pass --checks explicitly" >&2
    exit 1
  }
fi

if [ -z "$SEEN_FILE" ]; then
  SEEN_FILE="/tmp/watch_for_passes_$(basename "$RUN_DIR").txt"
  : > "$SEEN_FILE"
else
  touch "$SEEN_FILE"
fi

echo "watching $RUN_DIR (checks: $CHECKS, ledger: $SEEN_FILE)"

check_reports() {
  python3 - "$SEEN_FILE" "$RUN_DIR" "$CHECKS" <<'PYEOF'
import glob, json, os, sys

seen_path, run_dir, checks = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(seen_path) as handle:
        seen = set(handle.read().split())
except FileNotFoundError:
    seen = set()

CHECKED = {int(c) for c in checks.split(",") if c.strip()}

paths = sorted(
    set(glob.glob(os.path.join(run_dir, "**", "chunk_*_report.json"), recursive=True))
    | set(glob.glob(os.path.join(run_dir, "**", "report.json"), recursive=True))
)

new_seen = set(seen)
for p in paths:
    try:
        with open(p) as handle:
            r = json.load(handle)
    except (OSError, json.JSONDecodeError):
        # A report still being written is normal mid-run, not an error.
        continue
    for t in r.get('tasks', []):
        tid = t['task_id']
        if tid in seen:
            continue
        v = t.get('verdict')
        fails = set(t.get('fail_checks') or [])
        rel = os.path.relpath(p, run_dir)
        if v in ('clean', 'pass_with_issues'):
            print(f"PASS_FOUND: {tid} verdict={v} (genuine pass) src={rel}")
        elif v != 'unauditable' and not (fails & CHECKED):
            print(f"PASS_FOUND: {tid} verdict={v} (excluded pass, fails={sorted(fails)}) src={rel}")
        new_seen.add(tid)

with open(seen_path, 'w') as f:
    f.write('\n'.join(sorted(new_seen)))
PYEOF
}

while true; do
  check_reports
  if [ "$ONCE" = 1 ]; then
    break
  fi
  if [ -n "$DONE_LOG" ] && grep -q "$DONE_PATTERN" "$DONE_LOG" 2>/dev/null; then
    check_reports  # one final pass in case the last chunk just landed
    echo "WATCH_DONE: $(basename "$RUN_DIR") finished, no more chunks to check"
    break
  fi
  sleep "$INTERVAL"
done
