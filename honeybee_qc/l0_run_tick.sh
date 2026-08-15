#!/usr/bin/env bash
# Run one tick of the L0 overnight restricted-scope audit pipeline by hand.
# Safe to re-run: already-processed tasks are skipped, cached model calls are free.
# Never exits the *shell* on a bad tick -- logs the failure and returns its
# exit code so a caller (loop, cron, coworker) can just try again next time.
#
# Usage:
#   honeybee_qc/l0_run_tick.sh [run-dir] [workers]              # discover + audit + append
#   honeybee_qc/l0_run_tick.sh [run-dir] [workers] --discover-only   # just print eligible task IDs (JSON), no audit
set -uo pipefail

RUN_DIR="${1:-honeybee_qc/audit_runs/l0_overnight_20260813}"
WORKERS="${2:-30}"
EXTRA_FLAG="${3:-}"
cd "$(dirname "$0")/.."

if [ "$EXTRA_FLAG" != "--discover-only" ]; then
  echo "Checking prerequisites..."
  command -v claude >/dev/null 2>&1 || { echo "ERROR: 'claude' CLI not found on PATH. This pipeline shells out to it for every model call."; exit 1; }
  python3 -c "from honeybee_qc.sources import find_chrome; import sys; sys.exit(0 if find_chrome() else 1)" \
    || echo "WARNING: couldn't find a local Chrome install — conversation scraping (--fetch-conversations) may fail."
  [ -f ~/.cursor/redash.env ] || echo "WARNING: ~/.cursor/redash.env not found — the Snowflake pull step may fail."
fi

mkdir -p "$RUN_DIR"

python3 -m honeybee_qc.l0_overnight_tick \
  --run-dir "$RUN_DIR" \
  --workers "$WORKERS" \
  $EXTRA_FLAG
code=$?

if [ "$EXTRA_FLAG" != "--discover-only" ]; then
  if [ $code -ne 0 ]; then
    echo "Tick exited $code — see $RUN_DIR/tick_errors.log for detail. Not fatal: safe to just try again next tick."
  else
    echo "Done. Run honeybee_qc/l0_status.sh $RUN_DIR to see the results."
  fi
fi
exit $code
