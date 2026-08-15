#!/usr/bin/env bash
# Quick, read-only status check for the L0 overnight restricted-scope run.
# Usage: honeybee_qc/l0_status.sh [run-dir]
set -euo pipefail

RUN_DIR="${1:-honeybee_qc/audit_runs/l0_overnight_20260813}"
cd "$(dirname "$0")/.."

if [ ! -d "$RUN_DIR" ]; then
  echo "Run dir not found: $RUN_DIR"
  exit 1
fi

echo "== Run dir: $RUN_DIR =="
echo

if [ -f "$RUN_DIR/loop.pid" ]; then
  pid=$(cat "$RUN_DIR/loop.pid")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "Heartbeat loop: RUNNING (pid $pid). Stop with: honeybee_qc/l0_stop_loop.sh $RUN_DIR"
  else
    echo "Heartbeat loop: pid file present ($pid) but not running -- it likely died with its chat session. Restart with: honeybee_qc/l0_loop.sh $RUN_DIR 7200"
  fi
else
  echo "Heartbeat loop: no pid file found -- not running (or never started). Start with: honeybee_qc/l0_loop.sh $RUN_DIR 7200"
fi
if [ -f "$RUN_DIR/STOP_LOOP" ]; then
  echo "NOTE: a STOP_LOOP file exists in this run dir -- a fresh loop will delete it on startup, but if a loop is currently running, it will exit shortly."
fi

if [ -f "$RUN_DIR/tick_errors.log" ]; then
  n=$(wc -l < "$RUN_DIR/tick_errors.log" | tr -d ' ')
  echo "tick_errors.log: $n line(s) -- last 5:"
  tail -n 5 "$RUN_DIR/tick_errors.log"
fi
echo

if [ -f "$RUN_DIR/processed_ids.txt" ]; then
  n=$(wc -l < "$RUN_DIR/processed_ids.txt" | tr -d ' ')
  echo "Tasks processed so far: $n"
else
  echo "Tasks processed so far: 0 (no processed_ids.txt yet)"
fi

if [ -f "$RUN_DIR/results.csv" ]; then
  rows=$(($(wc -l < "$RUN_DIR/results.csv" | tr -d ' ') - 1))
  echo "Rows in results.csv: $rows"
  echo
  echo "-- Last 5 results --"
  python3 -c "
import csv
with open('$RUN_DIR/results.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows[-5:]:
    print(f\"{r.get('task_id','?'):>26}  {r.get('status','?'):>10}  {r.get('task_verdict','?'):>16}  fail_checks={r.get('fail_checks','')}\")
"
else
  echo "No results.csv yet (no tasks audited in any tick)."
fi

echo
if [ -f "$RUN_DIR/cache.db" ]; then
  python3 -c "
import sqlite3
conn = sqlite3.connect('file:$RUN_DIR/cache.db?mode=ro', uri=True, timeout=5)
print('Cached model calls:', conn.execute('select count(*) from response_cache').fetchone()[0])
"
fi

