#!/usr/bin/env bash
# Local Honeybee QC: eval → fixer → invariants gate → paste sheet.
# Do not kill Chrome. Do not raise hydrate-workers above 8.
#
#   ./honeybee_qc/team_run.sh all    --csv pulled.csv --project l1_sxs --slots A,B --out honeybee_qc/runs/my_batch
#   ./honeybee_qc/team_run.sh eval   --csv pulled.csv --project l1_sxs --slots A,B --out honeybee_qc/runs/my_batch
#   ./honeybee_qc/team_run.sh fixer  --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
#   ./honeybee_qc/team_run.sh gate   --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
#   ./honeybee_qc/team_run.sh sheet  --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
#   ./honeybee_qc/team_run.sh pull-sql --project l1_sxs [--ids-file ids.txt]
set -euo pipefail

ONLY_CHECKS="90,91,95,100,110,270,300,310,400,450,470"
HYDRATE_WORKERS=8
WORKERS=70
TASK_WORKERS=20
MODEL="claude-haiku-4-5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of honeybee_qc/ — required for `python3 -m honeybee_qc.*`
if [[ "$(basename "$SCRIPT_DIR")" == "honeybee_qc" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  ROOT="$SCRIPT_DIR"
fi

if [[ -x /opt/homebrew/bin/python3 ]]; then
  PY=/opt/homebrew/bin/python3
else
  PY="$(command -v python3)"
fi

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
  echo "need Homebrew Python 3.13 (got $($PY --version 2>&1)). brew install python@3.13" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

usage() {
  cat <<'EOF'
Local Honeybee QC: eval → fixer → invariants gate → paste sheet.
Do not kill Chrome. Do not raise hydrate-workers above 8.

  ./honeybee_qc/team_run.sh all    --csv pulled.csv --project l1_sxs --slots A,B --out honeybee_qc/runs/my_batch
  ./honeybee_qc/team_run.sh eval   --csv pulled.csv --project l1_sxs --slots A,B --out honeybee_qc/runs/my_batch
  ./honeybee_qc/team_run.sh fixer  --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
  ./honeybee_qc/team_run.sh gate   --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
  ./honeybee_qc/team_run.sh sheet  --csv pulled.csv --project l1_sxs --out honeybee_qc/runs/my_batch
  ./honeybee_qc/team_run.sh pull-sql --project l1_sxs [--ids-file ids.txt]

Flags: --csv PATH  --project KEY  --slots A,B|A,C  --out DIR
       --ids-file PATH   (pull-sql only)
       --hydrate-workers N  (capped at 16; default 8)
       --workers N          (LLM cap; default 70)
       --task-workers N     (tasks in flight; default 20)
EOF
}

CMD="${1:-}"
shift || true

CSV=""
PROJECT=""
SLOTS=""
OUT=""
IDS_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV="${2:-}"; shift 2 ;;
    --project) PROJECT="${2:-}"; shift 2 ;;
    --slots) SLOTS="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --ids-file) IDS_FILE="${2:-}"; shift 2 ;;
    --hydrate-workers)
      HYDRATE_WORKERS="${2:-8}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:-70}"
      shift 2
      ;;
    --task-workers)
      TASK_WORKERS="${2:-20}"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown flag: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$HYDRATE_WORKERS" -gt 16 ]]; then
  echo "refusing --hydrate-workers $HYDRATE_WORKERS (cap is 16; each worker is a Chrome)" >&2
  exit 2
fi

need() {
  local name="$1" value="$2"
  if [[ -z "$value" ]]; then
    echo "missing $name" >&2
    usage >&2
    exit 2
  fi
}

run_py() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -ims "$PY" -u "$@"
  else
    "$PY" -u "$@"
  fi
}

cmd_pull_sql() {
  need "--project" "$PROJECT"
  "$PY" - "$PROJECT" "${IDS_FILE:-}" <<'PY'
import sys
from pathlib import Path

from honeybee_qc.ingest import latest_attempt_sql
from honeybee_qc.projects import profile_by_key

key = sys.argv[1]
ids_file = sys.argv[2]
profile = profile_by_key(key)
if profile is None:
    raise SystemExit(f"unknown --project {key!r}")
sql = latest_attempt_sql(profile)
if ids_file:
    ids = []
    for line in Path(ids_file).read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip().strip("'\"")
        if token:
            ids.append(token)
    if not ids:
        raise SystemExit(f"no task IDs in {ids_file}")
    short = [i for i in ids if len(i) != 24]
    if short:
        raise SystemExit("IDs must be full 24-hex: " + ", ".join(short[:5]))
    listed = ",".join(f"'{i}'" for i in ids)
    needle = f"where project = '{profile.project_id}'"
    sql = sql.replace(needle, needle + f"\n    and task in ({listed})", 1)
print(sql.strip())
PY
}

cmd_eval() {
  need "--csv" "$CSV"
  need "--project" "$PROJECT"
  need "--slots" "$SLOTS"
  need "--out" "$OUT"
  mkdir -p "$OUT/snapshots" "$OUT/fixer"
  if [[ -f "$OUT/report.json" ]]; then
    echo "report.json already present at $OUT/report.json — skip eval" >&2
    return 0
  fi
  echo "eval  project=$PROJECT slots=$SLOTS hydrate=$HYDRATE_WORKERS workers=$WORKERS task-workers=$TASK_WORKERS out=$OUT" >&2
  run_py -m honeybee_qc.cli \
    "$CSV" --from-snowflake \
    --project "$PROJECT" \
    --rating-stage --informed-stage \
    --only-checks "$ONLY_CHECKS" \
    --audit-slots "$SLOTS" \
    --fetch-conversations --snapshot-dir "$OUT/snapshots" \
    --hydrate-workers "$HYDRATE_WORKERS" \
    --workers "$WORKERS" --task-workers "$TASK_WORKERS" \
    --model "$MODEL" \
    --cache-db "$OUT/cache.db" \
    --out "$OUT/report.json"
}

cmd_fixer() {
  need "--csv" "$CSV"
  need "--project" "$PROJECT"
  need "--out" "$OUT"
  mkdir -p "$OUT/fixer" "$OUT/snapshots"
  if [[ ! -f "$OUT/report.json" ]]; then
    echo "missing $OUT/report.json — run eval first" >&2
    exit 1
  fi
  if [[ -f "$OUT/fixer/fixer_results.json" ]]; then
    echo "fixer_results.json present — skip fixer" >&2
    return 0
  fi
  echo "fixer  project=$PROJECT hydrate=$HYDRATE_WORKERS out=$OUT/fixer" >&2
  run_py -m honeybee_qc.fixer \
    --report "$OUT/report.json" \
    --tasks-csv "$CSV" --from-snowflake \
    --project "$PROJECT" \
    --fetch-conversations --snapshot-dir "$OUT/snapshots" \
    --hydrate-workers "$HYDRATE_WORKERS" \
    --workers "$WORKERS" --task-workers "$TASK_WORKERS" \
    --model "$MODEL" \
    --cache-db "$OUT/cache.db" \
    --out-dir "$OUT/fixer"
}

cmd_gate() {
  need "--csv" "$CSV"
  need "--project" "$PROJECT"
  need "--out" "$OUT"
  if [[ ! -d "$OUT/fixer" ]]; then
    echo "missing $OUT/fixer — run fixer first" >&2
    exit 1
  fi
  echo "gate  project=$PROJECT fixer=$OUT/fixer" >&2
  run_py -m honeybee_qc.fixer_invariants \
    --tasks-csv "$CSV" --from-snowflake \
    --project "$PROJECT" \
    --fixer-dir "$OUT/fixer" \
    --snapshot-dir "$OUT/snapshots" \
    --hydrate-workers "$HYDRATE_WORKERS" \
    --out-dir "$OUT/fixer"
}

cmd_sheet() {
  need "--csv" "$CSV"
  need "--project" "$PROJECT"
  need "--out" "$OUT"
  if [[ ! -f "$OUT/fixer/fixer_results.json" ]]; then
    echo "missing $OUT/fixer/fixer_results.json — run fixer first" >&2
    exit 1
  fi
  echo "sheet  project=$PROJECT → $OUT/fixer/backfill_steps.csv" >&2
  run_py -m honeybee_qc.backfill_steps \
    --tasks-csv "$CSV" \
    --fixer-dir "$OUT/fixer" \
    --project "$PROJECT" \
    --out "$OUT/fixer/backfill_steps.csv"
}

case "$CMD" in
  pull-sql) cmd_pull_sql ;;
  eval) cmd_eval ;;
  fixer) cmd_fixer ;;
  gate) cmd_gate ;;
  sheet) cmd_sheet ;;
  all)
    cmd_eval
    cmd_fixer
    set +e
    cmd_gate
    gate_rc=$?
    set -e
    cmd_sheet
    exit "$gate_rc"
    ;;
  ""|-h|--help) usage; exit 0 ;;
  *)
    echo "unknown command: $CMD" >&2
    usage >&2
    exit 2
    ;;
esac
