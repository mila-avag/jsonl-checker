"""Split a task-id list into fixed-size chunk CSVs for a sequential run.

Chunking is what makes a long run answerable while it is still going. The audit
pipeline writes its report once, at the end, so a single invocation over 300 tasks
is a ten-hour job with no verdict available until the tenth hour -- and if it dies
at hour nine (which has happened on this machine: see batch9's status log, where a
run vanished under memory pressure with no exit code) everything not already in the
response cache is wall-clock lost. Twelve chunks of twenty-five each cost the same
model calls and give up a complete, readable report roughly every fifty minutes.

The chunk size is a recoverability decision, not a performance one. Throughput is
set by `--workers`/`--task-workers` inside a chunk and is unaffected by where the
boundaries fall; what the size buys is the granularity of the answer and the size
of the blast radius when something dies.

Usage:
    python3 -m honeybee_qc.chunk_tasks \\
        --task-ids honeybee_qc/audit_runs/rerun_20260815/task_ids.txt \\
        --source honeybee_qc/audit_runs/.../pulled_300.csv \\
        --out-dir honeybee_qc/audit_runs/rerun_20260815/chunks \\
        --size 25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def load_ids(path: Path) -> list[str]:
    """Task ids, one per line or comma-separated, order preserved and deduped."""
    raw = path.read_text(encoding="utf-8")
    out: list[str] = []
    seen: set[str] = set()
    for piece in raw.replace(",", "\n").split("\n"):
        task_id = piece.strip()
        if task_id and task_id not in seen:
            seen.add(task_id)
            out.append(task_id)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-ids", required=True, help="file of task ids to include")
    ap.add_argument("--source", required=True, help="taskattempts CSV to slice")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=25, help="tasks per chunk")
    args = ap.parse_args(argv)

    wanted = load_ids(Path(args.task_ids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read once, index by task id, then emit in the order the list asked for, so a
    # chunk's contents are predictable from the input rather than from however the
    # export happened to be sorted.
    with open(args.source, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: dict[str, list[str]] = {}
        for row in reader:
            if row and row[0] in set(wanted):
                rows.setdefault(row[0], row)

    missing = [t for t in wanted if t not in rows]
    present = [t for t in wanted if t in rows]

    print(f"requested {len(wanted)} tasks; {len(present)} found in {args.source}")
    if missing:
        print(f"MISSING {len(missing)} -- not in the source export:", file=sys.stderr)
        for task_id in missing:
            print(f"  {task_id}", file=sys.stderr)
        print(
            "These need a Snowflake pull before they can be audited; a chunk is "
            "not written for them, rather than written short and silently under-"
            "counting the batch.",
            file=sys.stderr,
        )

    written = 0
    for index in range(0, len(present), args.size):
        batch = present[index : index + args.size]
        path = out_dir / f"chunk_{written:02d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for task_id in batch:
                writer.writerow(rows[task_id])
        print(f"  wrote {path.name}: {len(batch)} tasks")
        written += 1

    print(f"{written} chunks of up to {args.size} tasks in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
