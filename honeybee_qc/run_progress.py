"""Summarise a chunked run that is still going.

The pipeline's own report is written per invocation, so during a twelve-chunk run
the answer to "what is failing so far" is spread across however many
`chunk_NN_report.json` files happen to exist, plus one chunk in flight with no file
yet. Reading those by hand is how a long run ends up unobserved until it finishes.

This reads every chunk report present, prints the batch verdict split and per-check
fail rates over the tasks completed so far, and -- because the point of the current
build is that a fail must be attributable -- separates what each check counted from
what its confirming pass declined to count. A fail rate that dropped because the
checks got quieter is a different fact from one that dropped because the work got
better, and only the suppression tallies distinguish them.

Nothing here waits on the run. Call it whenever, as often as you like; it opens the
reports read-only and never touches the cache the live process is writing to.

Usage:
    python3 -m honeybee_qc.run_progress honeybee_qc/audit_runs/rerun_20260815
    python3 -m honeybee_qc.run_progress <dir> --by-task     # per-task verdicts too
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .registry import REGISTRY

# Checks whose fail no longer follows from the raw measurement alone: each runs a
# second, informed call and reports what it declined to count. Listed here so the
# summary can show the two numbers side by side.
CONFIRMED_CHECKS = (300, 400, 95)


def chunk_reports(run_dir: Path) -> list[tuple[str, dict]]:
    """Every chunk report on disk, in chunk order."""
    out: list[tuple[str, dict]] = []
    for path in sorted(run_dir.glob("**/chunk_*_report.json")):
        try:
            out.append((path.stem.replace("_report", ""), json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            # A chunk still being written is normal mid-run, not an error worth
            # stopping for; say so and move on.
            print(f"  ({path.name}: not readable yet -- {exc})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--by-task", action="store_true", help="list every task's verdict")
    ap.add_argument(
        "--expected-chunks", type=int, default=0, help="to report progress as a fraction"
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    reports = chunk_reports(run_dir)
    if not reports:
        print(f"no chunk reports in {run_dir} yet")
        return 0

    verdicts: Counter[str] = Counter()
    spend = 0.0
    tasks_done = 0
    per_check: dict[int, Counter[str]] = defaultdict(Counter)
    suppressed: Counter[str] = Counter()
    blind_only: dict[int, list[str]] = defaultdict(list)
    task_rows: list[tuple[str, str, list[int]]] = []

    for name, report in reports:
        counts = report.get("batch", {}).get("task_counts", {})
        for key in ("fail", "pass", "pass_with_issues", "unauditable"):
            verdicts[key] += counts.get(key, 0)
        spend += float(report.get("cost_usd") or 0.0)
        tasks_done += counts.get("total", 0)

        for task in report.get("tasks", []):
            task_rows.append(
                (task["task_id"], task.get("verdict", "?"), task.get("fail_checks", []))
            )
            for check in task.get("checks", []):
                per_check[check["check_id"]][check["band"]] += 1

        for key, value in (report.get("batch", {}).get("suppressed_findings") or {}).items():
            suppressed[key] += value
        for check_id, ids in (report.get("batch", {}).get("blind_only_tasks") or {}).items():
            blind_only[int(check_id)].extend(ids)

    header = f"{len(reports)} chunk(s) complete"
    if args.expected_chunks:
        header += f" of {args.expected_chunks}"
    print(f"\n{header} -- {tasks_done} tasks audited, ${spend:.2f} spent\n")

    total = sum(verdicts.values()) or 1
    print("task verdicts so far")
    for key in ("fail", "pass_with_issues", "pass", "unauditable"):
        if verdicts[key]:
            print(f"  {key:18} {verdicts[key]:4}  ({verdicts[key] / total:5.1%})")

    print("\nper-check results so far (fails / judged)")
    print(f"  {'check':>5}  {'fail':>5} {'judged':>6} {'rate':>6}  name")
    for check_id in sorted(per_check):
        bands = per_check[check_id]
        judged = sum(v for k, v in bands.items() if k != "not_evaluated")
        fails = bands.get("fail", 0)
        rate = f"{fails / judged:.0%}" if judged else "-"
        spec = REGISTRY.get(check_id)
        name = spec.sub_dimension if spec else ""
        mark = " *" if check_id in CONFIRMED_CHECKS else "  "
        print(f"  {check_id:>5}{mark} {fails:>5} {judged:>6} {rate:>6}  {name}")

    if suppressed:
        print(
            "\nfindings the confirming passes declined to count (* checks above)\n"
            "  these were found and are NOT in the fail counts, so a lower fail rate\n"
            "  here is not by itself evidence the work improved:"
        )
        for key in sorted(suppressed):
            print(f"    {key:42} {suppressed[key]:4}")
        for check_id in sorted(blind_only):
            ids = sorted(set(blind_only[check_id]))
            print(
                f"    check {check_id}: {len(ids)} task(s) where the blind pass "
                "disagreed and nothing informed confirmed it"
            )

    if args.by_task:
        print("\nper-task verdicts")
        for task_id, verdict, fail_checks in sorted(task_rows, key=lambda r: r[1]):
            failed = ",".join(str(c) for c in fail_checks) or "-"
            print(f"  {task_id}  {verdict:18} fails: {failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
