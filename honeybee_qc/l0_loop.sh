#!/usr/bin/env bash
# Overnight heartbeat loop for the L0 restricted-scope QC pipeline.
#
# Runs INDEFINITELY, waking every 2 hours (default) to print a sentinel line
# that an agent turn reacts to (see L0_OVERNIGHT_GUIDE.md). This script does
# NOT run the audit itself -- it only signals "time to tick". The actual
# pull/discover/WIP-row/audit/Sheet-sync sequence happens in the agent turn
# that reacts to the sentinel, because Sheet writes need the Google Sheets
# MCP connection, which only exists inside a Cursor chat.
#
# Stops ONLY when a stop file appears at $RUN_DIR/STOP_LOOP -- never on its
# own, never after some fixed number of hours, never because of a transient
# error (there's nothing in this script that can throw one). To stop it:
#   honeybee_qc/l0_stop_loop.sh [run-dir]
#
# Usage: honeybee_qc/l0_loop.sh [run-dir] [interval-seconds]
set -uo pipefail

RUN_DIR="${1:-honeybee_qc/audit_runs/l0_overnight_20260813}"
INTERVAL="${2:-7200}"
cd "$(dirname "$0")/.."
mkdir -p "$RUN_DIR"

STOP_FILE="$RUN_DIR/STOP_LOOP"
PID_FILE="$RUN_DIR/loop.pid"
SPREADSHEET_ID="1HrzymwVhHnfalZnINFdRYJtCmYjkF8mtHiN80fqv30w"

rm -f "$STOP_FILE"
echo $$ > "$PID_FILE"

echo "AGENT_LOOP_STARTED_l0_overnight pid=$$ run_dir=$RUN_DIR interval=${INTERVAL}s stop_file=$STOP_FILE"

CHUNK=60
if [ "$INTERVAL" -lt "$CHUNK" ]; then
  CHUNK="$INTERVAL"
fi
while true; do
  elapsed=0
  while [ "$elapsed" -lt "$INTERVAL" ]; do
    if [ -f "$STOP_FILE" ]; then
      echo "AGENT_LOOP_STOPPED_l0_overnight stop file detected at $STOP_FILE; exiting cleanly"
      rm -f "$PID_FILE"
      exit 0
    fi
    sleep "$CHUNK"
    elapsed=$((elapsed + CHUNK))
  done
  PROMPT="Restricted-scope L0 overnight tick. 1) Run: python3 -m honeybee_qc.l0_overnight_tick --run-dir $RUN_DIR --discover-only ; read its JSON eligible_task_ids. 2) For each eligible task id that is not already a row in the Tasks tab of spreadsheet $SPREADSHEET_ID, append a WIP row (Status=WIP, other columns blank) BEFORE auditing -- this must happen before step 3. 3) Then run: honeybee_qc/l0_run_tick.sh $RUN_DIR 30 to actually audit and append to results.csv. 4) Read any results.csv rows added by step 3 and update those same Sheet rows (by task id) with the final decision/reasoning/issues/status, replacing the WIP placeholder rather than duplicating the row. 5) If step 3 exited nonzero, leave the affected rows marked WIP/error rather than guessing a result, check $RUN_DIR/tick_errors.log, and flag the failure clearly in chat instead of silently retrying forever."
  echo "AGENT_LOOP_TICK_l0_overnight $(python3 -c 'import json,sys; print(json.dumps({"run_dir": sys.argv[1], "prompt": sys.argv[2]}))' "$RUN_DIR" "$PROMPT")"
done
