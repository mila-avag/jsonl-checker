"""One tick of the L0 overnight restricted-scope QC loop.

Pulls `taskattempts` rows on the L0 project **attempted from yesterday
onward** (not the full history -- this loop tracks fresh activity, not a
backlog), keeps only the ones that have actually reached the rating or
likert stage (a populated `criterion_ratings` score or a recorded
`sxs.likert` -- not just the presence of the step, which is there empty from
the moment the rubric is authored), skips anything already in this run's
processed-ID ledger, audits the rest with the restricted-scope CLI mode
(`--restricted-checks`: 270 split per model, 280/310, 400/460, 450/470, no
rubric stage, no check 300), and appends one row per task to a running CSV.

Two phases, so a caller (e.g. the overnight loop) can mark tasks "WIP" in a
tracker *before* spending the several minutes it takes to audit them:

    --discover-only   pull + filter + skip-processed, print the eligible
                       task IDs as JSON, touch nothing else (no CSV, no
                       processed-ID ledger, no audit).
    (default)          do the discover step again (cheap, idempotent) and
                       then actually run the audit + append to results.csv +
                       mark processed. This is also the single command a
                       human runs by hand -- see l0_run_tick.sh.

Usage:
    python3 -m honeybee_qc.l0_overnight_tick --run-dir honeybee_qc/audit_runs/l0_overnight_20260812
    python3 -m honeybee_qc.l0_overnight_tick --run-dir ... --discover-only

Never raises out of main(): every failure mode is caught, logged to
<run-dir>/tick_errors.log, and reported as a nonzero exit code plus a clear
one-line message on stdout, so a loop calling this over and over never dies
on a single bad tick.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Task `response` blobs (rendered turn manifests, share links, etc.) routinely
# exceed Python's default 128 KiB per-field CSV cap on larger conversations.
# sys.maxsize can overflow the underlying C long on some platforms; back off
# until it's accepted rather than letting that take out every future tick.
_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_field_limit)
        break
    except OverflowError:
        _field_limit //= 2

sys.path.insert(0, os.path.expanduser("~/.cursor/skills/redash/scripts"))
import redash_client as rc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from honeybee_qc.ingest import ingest_attempt  # noqa: E402

L0_PROJECT_ID = "6a7a2d34ac683589810bfa03"
DEFAULT_WORKERS = 30

# `attempted_at >= <yesterday, midnight, in the warehouse's own local time>`.
# Recomputed fresh on every call (no hardcoded date) so the window keeps
# sliding across a multi-day overnight run.
PULL_SQL = f"""
with recent as (
  select *
  from public.taskattempts
  where project = '{L0_PROJECT_ID}'
    and attempted_at >= dateadd(day, -1, date_trunc('day', current_timestamp()))
),
latest as (
  select *,
         row_number() over (partition by task
                            order by attempt_version desc, attempted_at desc) as rn
  from recent
)
select task, attempted_by, review_status, response
from latest
where rn = 1
"""

RESULTS_COLUMNS = [
    "tick_at",
    "task_id",
    "task_verdict",
    "fail_checks",
    "non_fail_checks",
    "not_evaluated_checks",
    "check_270_a",
    "check_270_b",
    "reasoning",
    "issues",
    "status",
]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log_error(run_dir: Path, msg: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "tick_errors.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{now_iso()}] {msg}\n")


def reached_rating_stage(task) -> bool:
    if task.sxs.likert is not None:
        return True
    return any(cr.score != 0 for cr in task.criterion_ratings)


def load_processed(run_dir: Path) -> set[str]:
    path = run_dir / "processed_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_processed(run_dir: Path, task_ids: list[str]) -> None:
    with open(run_dir / "processed_ids.txt", "a", encoding="utf-8") as fh:
        for tid in task_ids:
            fh.write(tid + "\n")


def pull_with_retries(retries: int = 3, backoff_s: float = 5.0):
    """Snowflake/Redash can transiently hiccup overnight; don't let one flaky
    call take down the whole tick (or the loop calling it)."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _, _, csv_text, _ = rc.run_sql(PULL_SQL, cache_key="l0_overnight::pull")
            return csv_text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"Snowflake pull failed after {retries} attempts: {last_exc}") from last_exc


def discover(run_dir: Path) -> dict:
    """Pull + filter + skip-processed. Read-only: touches no ledger, no CSV."""
    csv_text = pull_with_retries()
    # NOT csv_text.splitlines() -- a `response` field routinely contains
    # embedded newlines (it's a JSON blob), and naive line-splitting before
    # csv.DictReader sees the text breaks quoted fields across "rows",
    # misaligning columns for every row after the first big one.
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    processed = load_processed(run_dir)
    eligible_rows = []
    eligible_task_ids = []
    total_reached_stage = 0
    for row in rows:
        # Redash/Snowflake column casing on this row is not guaranteed (it's
        # upper-cased through the CSV export, lower-cased through most
        # drivers -- ingest_attempt already normalizes this internally, so
        # get the task id from its result rather than re-deriving it here
        # with a case-sensitive lookup on the raw row).
        try:
            result = ingest_attempt(row)
        except Exception as exc:  # noqa: BLE001
            hint = {str(k).lower(): v for k, v in row.items()}.get("task")
            log_error(run_dir, f"ingest failed for row task={hint}: {exc}")
            continue
        if not result.ok or result.task is None:
            continue
        if not reached_rating_stage(result.task):
            continue
        total_reached_stage += 1
        if result.task.task_id in processed:
            continue
        eligible_rows.append(row)
        eligible_task_ids.append(result.task.task_id)

    return {
        "pulled": len(rows),
        "total_reached_stage": total_reached_stage,
        "already_processed": len(processed),
        "eligible_rows": eligible_rows,
        "eligible_task_ids": eligible_task_ids,
        "fieldnames": reader.fieldnames,
    }


def issues_and_reasoning(task_report: dict) -> tuple[str, str]:
    fails = [c for c in task_report["checks"] if c["band"] in ("fail", "non_fail")]
    if not fails:
        return "", "No fails or issues found among the audited checks."
    issues = "; ".join(f"check {c['check_id']}: {c['band']}" for c in fails)
    reasoning = " | ".join(
        f"check {c['check_id']}: {(c['measurement'].get('notes') or '')[:200]}"
        for c in fails
    )
    return issues, reasoning


def run_audit(run_dir: Path, disco: dict, workers: int, tick_at: str) -> int:
    eligible_rows = disco["eligible_rows"]
    if not eligible_rows:
        return 0

    tick_input_csv = run_dir / "tick_input.csv"
    with open(tick_input_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=disco["fieldnames"])
        writer.writeheader()
        writer.writerows(eligible_rows)

    tick_report_path = run_dir / "tick_report.json"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "honeybee_qc.cli",
        str(tick_input_csv),
        "--from-snowflake",
        # 270/400/450 need the model's actual response text to judge against;
        # without this every one of them abstains (`not_evaluated`), which is
        # honest but useless for a task whose share links are still live.
        "--fetch-conversations",
        "--snapshot-dir",
        str(snapshot_dir),
        "--rating-stage",
        "--informed-stage",
        "--restricted-checks",
        "--cache-db",
        str(run_dir / "cache.db"),
        "--out",
        str(tick_report_path),
        "--workers",
        str(workers),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        log_error(run_dir, f"cli timed out after 3600s for {len(eligible_rows)} task(s)")
        print("cli timed out; skipping CSV append this tick")
        return 1
    except Exception as exc:  # noqa: BLE001
        log_error(run_dir, f"cli subprocess failed to launch: {exc}\n{traceback.format_exc()}")
        print(f"cli subprocess failed to launch: {exc}")
        return 1

    print(proc.stderr.strip())
    if proc.returncode != 0:
        log_error(run_dir, f"cli exited {proc.returncode}. stderr tail:\n{proc.stderr[-2000:]}")
        print(f"cli exited {proc.returncode}; skipping CSV append this tick")
        return proc.returncode

    try:
        report = json.loads(tick_report_path.read_text())
    except Exception as exc:  # noqa: BLE001
        log_error(run_dir, f"could not parse tick_report.json: {exc}")
        print(f"could not parse tick_report.json: {exc}")
        return 1

    by_model = report.get("check_270_by_model", {})

    results_csv = run_dir / "results.csv"
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        for task_report in report.get("tasks", []):
            task_id = task_report["task_id"]
            split = by_model.get(task_id, {})
            issues, reasoning = issues_and_reasoning(task_report)
            writer.writerow(
                {
                    "tick_at": tick_at,
                    "task_id": task_id,
                    "task_verdict": task_report["verdict"],
                    "fail_checks": ",".join(str(c) for c in task_report["fail_checks"]),
                    "non_fail_checks": ",".join(
                        str(c) for c in task_report["non_fail_checks"]
                    ),
                    "not_evaluated_checks": ",".join(
                        str(c) for c in task_report["not_evaluated_checks"]
                    ),
                    "check_270_a": split.get("A", {}).get("band", ""),
                    "check_270_b": split.get("B", {}).get("band", ""),
                    "reasoning": reasoning,
                    "issues": issues,
                    "status": "audited",
                }
            )

    append_processed(run_dir, [t["task_id"] for t in report.get("tasks", [])])
    print(f"audited {len(report.get('tasks', []))} tasks this tick; appended to {results_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="print eligible task IDs as JSON and exit; touches no ledger/CSV/audit",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tick_at = now_iso()

    try:
        disco = discover(run_dir)
    except Exception as exc:  # noqa: BLE001
        log_error(run_dir, f"discover phase failed: {exc}\n{traceback.format_exc()}")
        print(f"tick {tick_at}: discover phase failed ({exc}); nothing audited this tick")
        return 1

    status_line = (
        f"tick {tick_at}: pulled {disco['pulled']} L0 taskattempts (attempted-at >= yesterday), "
        f"{disco['total_reached_stage']} at rating/likert stage, "
        f"{len(disco['eligible_rows'])} new, {disco['already_processed']} already processed"
    )

    if args.discover_only:
        print(json.dumps({
            "tick_at": tick_at,
            "eligible_task_ids": disco["eligible_task_ids"],
            "total_reached_stage": disco["total_reached_stage"],
            "already_processed": disco["already_processed"],
            "pulled": disco["pulled"],
        }))
        return 0

    print(status_line)
    if not disco["eligible_rows"]:
        print("no new eligible tasks; nothing audited this tick")
        return 0

    try:
        return run_audit(run_dir, disco, args.workers, tick_at)
    except Exception as exc:  # noqa: BLE001
        log_error(run_dir, f"audit phase failed: {exc}\n{traceback.format_exc()}")
        print(f"audit phase failed ({exc}); nothing appended this tick")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
