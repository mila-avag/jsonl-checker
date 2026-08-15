"""Score a re-run against a manual audit's per-(task, check) ground truth.

When a batch's fails are read by hand, each one gets classified as a real defect,
an artifact of how the check worked, or genuinely ambiguous. That hand-read set is
the only thing available to check a subsequent fix against: a fail rate that simply
went down proves nothing, because the cheapest way to lower a fail rate is to break
the check.

So both directions are scored, and the second matters more:

  - a FALSE_FAIL that is no longer failing is a fix that worked;
  - a TRUE_FAIL that is no longer failing is a regression, and the whole point of
    listing them is that this script cannot report success without them.

BORDERLINE cases are reported and deliberately not scored. The audit judged them
defensible either way, so counting them in either column would be reading a
preference into evidence that did not support one.

Two ways a listed (task, check) can go unscored are counted and named rather than
folded into a column, because both would otherwise read as good news: one that the
re-run did not cover (a narrowed or chunked run carries only some of the audited
tasks), and one that was not actually failing in the before report (a mismatched
before file, or ground truth written against a different run). A TRUE_FAIL in
either state is not a fail this build still catches, so the closing line does not
claim a clean regression guard while any remain.

The ground-truth JSON is `{check_id: {FALSE_FAIL|TRUE_FAIL|BORDERLINE: [task_id]}}`.
Any key that is not a check id (a `_comment`, say) is ignored.

Usage:
    python3 -m honeybee_qc.score_against_manual_audit \\
        --gt honeybee_qc/audit_runs/revalidate_20260815/gt.json \\
        --before honeybee_qc/audit_runs/honeybee_l1_full_20260814_batch9/combined_report.json \\
        --after honeybee_qc/audit_runs/revalidate_20260815/chunk_00_report.json

With no `--after` the newest `*report*.json` in `--run-dir` (default: the directory
holding `--gt`) is scored, so a chunked run can be compared as each chunk lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

VERDICTS = ("FALSE_FAIL", "TRUE_FAIL", "BORDERLINE")

MARKS = {
    "fixed": "FIXED  ",
    "still_failing": "STILL FAILING",
    "regressed": "REGRESSED",
    "held": "held   ",
    "unscored": "(unscored)",
    "not_failing_before": "NOT FAILING BEFORE",
    "missing": "outside this run",
}


@dataclass
class Row:
    """One (task, check) the audit listed, and how the re-run treated it."""

    check_id: str
    task_id: str
    verdict: str
    old_band: str
    new_band: str
    outcome: str
    why: str = ""


def index_bands(report: dict) -> dict[tuple[str, int], dict]:
    """(task_id, check_id) -> the verdict's band and its measurement."""
    out: dict[tuple[str, int], dict] = {}
    for task in report.get("tasks") or []:
        for check in task.get("checks") or []:
            out[(task["task_id"], check["check_id"])] = check
    return out


def load_bands(report_path: Path) -> dict[tuple[str, int], dict]:
    return index_bands(json.loads(Path(report_path).read_text(encoding="utf-8")))


def scored_checks(gt: dict) -> list[str]:
    """The check ids in a ground-truth file, numerically ordered."""
    return sorted((k for k in gt if k.strip().isdigit()), key=int)


def classify(verdict: str, old_band: str | None, new_band: str | None) -> str:
    """This (task, check)'s outcome, from the audit's verdict and the two bands."""
    if new_band is None:
        return "missing"
    if verdict == "BORDERLINE":
        return "unscored"
    if old_band != "fail":
        # Nothing to have fixed or regressed: whatever this is, it is not the
        # fail the audit read, so scoring it either way would be inventing a
        # result out of a file mismatch.
        return "not_failing_before"
    still_failing = new_band == "fail"
    if verdict == "FALSE_FAIL":
        return "still_failing" if still_failing else "fixed"
    return "held" if still_failing else "regressed"


def compare(
    gt: dict,
    before: dict[tuple[str, int], dict],
    after: dict[tuple[str, int], dict],
    checks: list[str] | None = None,
) -> list[Row]:
    """One row per (task, check) the ground truth lists, in reporting order."""
    rows: list[Row] = []
    for check_id in checks if checks is not None else scored_checks(gt):
        groups = gt.get(check_id)
        if not groups:
            continue
        for verdict in VERDICTS:
            for task_id in groups.get(verdict, []):
                key = (task_id, int(check_id))
                old, new = before.get(key), after.get(key)
                old_band = old["band"] if old else "?"
                new_band = new["band"] if new else "-"
                rows.append(
                    Row(
                        check_id=check_id,
                        task_id=task_id,
                        verdict=verdict,
                        old_band=old_band,
                        new_band=new_band,
                        outcome=classify(verdict, old_band, new["band"] if new else None),
                        why=_why(new) if new else "",
                    )
                )
    return rows


def unguarded_true_fails(rows: list[Row]) -> list[Row]:
    """TRUE_FAILs this run could not score, so the regression guard did not cover."""
    return [
        r
        for r in rows
        if r.verdict == "TRUE_FAIL" and r.outcome in ("missing", "not_failing_before")
    ]


def _why(check: dict) -> str:
    """The reason the new build gives, short enough to read in a table."""
    counts = (check.get("measurement") or {}).get("counts") or {}
    bits: list[str] = []
    if counts.get("blind_only_not_counted"):
        bits.append(f"blind_only={counts['blind_only_not_counted']}")
    if counts.get("truncated_evidence_not_counted"):
        bits.append(f"truncated={counts['truncated_evidence_not_counted']}")
    if counts.get("unadjudicated_not_counted"):
        bits.append(f"unadjudicated={counts['unadjudicated_not_counted']}")
    if counts.get("counted_disagreements") is not None:
        bits.append(f"counted={counts['counted_disagreements']}")
    if counts.get("not_counted_because"):
        bits.append(str(counts["not_counted_because"]))
    if counts.get("cleared_by_confirmation"):
        bits.append(f"cleared={len(counts['cleared_by_confirmation'])}")
    return ("  " + ", ".join(bits)) if bits else ""


def after_path(explicit: str | None, run_dir: Path) -> Path | None:
    """The re-run report to score: the argument, else the newest one present."""
    if explicit:
        return Path(explicit)
    candidates = sorted(
        run_dir.glob("*report*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="the manual audit's ground-truth JSON")
    ap.add_argument("--before", required=True, help="the report the audit was read against")
    ap.add_argument("--after", help="the re-run report to score; default: newest in --run-dir")
    ap.add_argument("--run-dir", help="where to find the newest report; default: --gt's directory")
    ap.add_argument("--checks", help="comma-separated check ids; default: every check in --gt")
    args = ap.parse_args(argv)

    gt_path = Path(args.gt)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir) if args.run_dir else gt_path.resolve().parent

    target = after_path(args.after, run_dir)
    if target is None:
        print(
            f"no *report*.json in {run_dir} yet -- is the run still going?", file=sys.stderr
        )
        return 1

    checks = (
        [c.strip() for c in args.checks.split(",") if c.strip()] if args.checks else None
    )
    before, after = load_bands(Path(args.before)), load_bands(target)
    rows = compare(gt, before, after, checks)
    totals = Counter(r.outcome for r in rows)

    print("=" * 78)
    print("RE-VALIDATION AGAINST THE MANUAL AUDIT'S GROUND TRUTH")
    print(f"scoring: {target.name}  ({len({k[0] for k in after})} tasks in the re-run)")
    print("=" * 78)

    shown: set[str] = set()
    for row in rows:
        if row.outcome == "missing":
            continue
        if row.check_id not in shown:
            print(f"\n### check {row.check_id}")
            shown.add(row.check_id)
        print(
            f"  {row.task_id[-6:]}  {row.verdict:10}  {row.old_band:>4} -> "
            f"{row.new_band:<12} {MARKS[row.outcome]:18}{row.why}"
        )

    print("\n" + "=" * 78)
    print(
        f"false fails fixed:      {totals['fixed']}\n"
        f"false fails remaining:  {totals['still_failing']}\n"
        f"true fails held:        {totals['held']}\n"
        f"true fails REGRESSED:   {totals['regressed']}\n"
        f"outside this run:       {totals['missing']}"
    )
    if totals["not_failing_before"]:
        print(
            f"not failing in --before: {totals['not_failing_before']}  <- scored in no "
            "column; check that --before is the run the ground truth was written against"
        )

    regressions = [r for r in rows if r.outcome == "regressed"]
    if regressions:
        print("\nREGRESSIONS -- a real defect this build stopped catching:")
        for row in regressions:
            print(f"  check {row.check_id} / {row.task_id}")
        return 0

    unguarded = unguarded_true_fails(rows)
    if not any(r.verdict == "TRUE_FAIL" for r in rows):
        print(
            "\nNo regression guard: the ground truth lists no true fails for the "
            "checks scored, so this can only confirm fixes."
        )
    elif unguarded:
        print(
            f"\nNo regressions among the {totals['held']} true fail(s) this run scored -- "
            f"but {len(unguarded)} more went unscored, so the regression guard is "
            "incomplete until they are re-run:"
        )
        for row in unguarded:
            print(f"  check {row.check_id} / {row.task_id}  ({MARKS[row.outcome].strip()})")
    else:
        print("\nNo regressions: every fail the audit called real is still failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
