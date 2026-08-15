#!/usr/bin/env bash
# Stop the L0 overnight heartbeat loop started by honeybee_qc/l0_loop.sh.
# This is the ONLY thing that stops the loop -- it does not stop on its own.
# Usage: honeybee_qc/l0_stop_loop.sh [run-dir]
set -uo pipefail

RUN_DIR="${1:-honeybee_qc/audit_runs/l0_overnight_20260813}"
cd "$(dirname "$0")/.."

STOP_FILE="$RUN_DIR/STOP_LOOP"
PID_FILE="$RUN_DIR/loop.pid"

touch "$STOP_FILE"
echo "Wrote stop file: $STOP_FILE"
echo "The loop checks for this every ~60s, so it should exit within a minute."

if [ -f "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE")
  echo "Waiting up to 75s for pid $pid to exit on its own..."
  for i in $(seq 1 15); do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "Loop (pid $pid) has exited."
      exit 0
    fi
    sleep 5
  done
  echo "Loop (pid $pid) did not exit in time -- killing it directly as a fallback."
  kill "$pid" 2>/dev/null || true
else
  echo "No pid file found at $PID_FILE -- if you know the loop's shell/pid from the terminal it was started in, kill it directly."
fi
