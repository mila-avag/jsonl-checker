"""Score an audit run against human QC verdicts from the audit sheet.

Reads the QC export, extracts every bracketed error code an auditor cited,
and compares that to what the pipeline reported for the same task.

Two views are printed because they answer different questions:

  raw       every QC claim is treated as correct, which is the pessimistic
            reading and the one to quote externally.
  adjusted  drops tasks with no stated reason and individual claims that were
            checked by hand and did not reproduce, which is the fairer read of
            how the pipeline performs when the ground truth is itself sound.

Both are reported per check and at the task-verdict level, and both are also
restricted to the checks this build actually implements, since a check with no
implementation can only ever lose recall and says nothing about the rest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ERROR_CODES

HONEYBEE_PROJECT_ID = "6a567b80aaa4d140131eae6c"

DETERMINISTIC = (90, 96, 100, 460)
RUBRIC_STAGE = (200, 210, 230, 240, 250, 260)
RATING_STAGE = (270, 300, 400)
INFORMED_STAGE = (70, 75, 80, 85, 95, 110, 220, 280, 310, 450, 470)

_LABEL = re.compile(r"\[(?:Non-)?Fail[^\]]*\]")


def label_index() -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    for check_id, bands in ERROR_CODES.items():
        for band, label in bands.items():
            index[label.strip().lower()] = (check_id, band)
    return index


@dataclass
class GroundTruth:
    task_id: str
    verdict: str = ""
    score: str = ""
    fail_checks: set[int] = field(default_factory=set)
    non_fail_checks: set[int] = field(default_factory=set)
    rows: int = 0

    @property
    def cited(self) -> set[int]:
        return self.fail_checks | self.non_fail_checks


def read_ground_truth(path: Path) -> dict[str, GroundTruth]:
    """Read a QC validations export.

    Two export shapes have been seen from this project, and this reads either:

      * The older sheet: one row per task-audit, columns `Project ID` / `Task`
        / `Pass/Fail` / `Score` / `FEEDBACK`, bracketed error codes embedded in
        the free-text feedback column.
      * The current `project-<id>_qc_validations.csv` export: one row per
        *dimension* an auditor rated (a task can have more than one row),
        columns `Task ID` / `QC Score` / `Selected error categories`. There is
        no `Pass/Fail` column here; QC's own scale never emits `1` (this
        build's own no-2/no-4 convention is not theirs), so `2` is their fail
        score. The bracketed codes live pre-extracted in `Selected error
        categories` rather than folded into a feedback paragraph, but the same
        regex finds them either way.

    Detected by column presence rather than a flag, so one function keeps
    working on whichever export the caller happens to have.
    """
    index = label_index()
    truth: dict[str, GroundTruth] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        current_shape = "Task ID" in fields and "QC Score" in fields
        for row in reader:
            if not current_shape and row.get("Project ID") != HONEYBEE_PROJECT_ID:
                continue
            task_id = row["Task ID"] if current_shape else row["Task"]
            gt = truth.setdefault(task_id, GroundTruth(task_id=task_id))
            gt.rows += 1

            if current_shape:
                score = (row.get("QC Score") or "").strip()
                row_verdict = "Fail" if score in ("1", "2") else "Pass" if score else ""
                label_source = row.get("Selected error categories") or ""
            else:
                score = row.get("Score") or ""
                row_verdict = row.get("Pass/Fail") or ""
                label_source = row.get("FEEDBACK") or ""

            # A task audited twice keeps the harsher verdict; either auditor
            # finding a failure means the task had one. The numeric score
            # keeps the lower (harsher) reading for the same reason.
            if row_verdict == "Fail" or not gt.verdict:
                gt.verdict = row_verdict or gt.verdict
            if score and (not gt.score or (score.isdigit() and (not gt.score.isdigit() or int(score) < int(gt.score)))):
                gt.score = score

            for match in set(_LABEL.findall(label_source)):
                hit = index.get(match.strip().lower())
                if not hit:
                    continue
                check_id, band = hit
                if band == "fail":
                    gt.fail_checks.add(check_id)
                else:
                    gt.non_fail_checks.add(check_id)
    return truth


@dataclass
class Predicted:
    task_id: str
    verdict: str
    fail_checks: set[int]
    non_fail_checks: set[int]
    not_evaluated: set[int]

    @property
    def flagged(self) -> set[int]:
        return self.fail_checks | self.non_fail_checks


def read_predictions(path: Path) -> dict[str, Predicted]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, Predicted] = {}
    for task in payload["tasks"]:
        out[task["task_id"]] = Predicted(
            task_id=task["task_id"],
            verdict=task["verdict"],
            fail_checks=set(task.get("fail_checks") or []),
            non_fail_checks=set(task.get("non_fail_checks") or []),
            not_evaluated=set(task.get("not_evaluated_checks") or []),
        )
    return out


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.1%}"


def _line(name: str, counts: Counts) -> str:
    return (
        f"  {name:34} tp={counts.tp:3} fp={counts.fp:3} fn={counts.fn:3}  "
        f"precision={_pct(counts.precision)}  recall={_pct(counts.recall)}  "
        f"f1={_pct(counts.f1)}"
    )


def score(
    truth: dict[str, GroundTruth],
    predicted: dict[str, Predicted],
    task_ids: list[str],
    restrict: set[int] | None,
    drop_claims: dict[str, set[int]],
) -> tuple[Counts, Counts, dict[int, Counts]]:
    """Return (check-level, task-verdict-level, per-check) counts."""
    check_level = Counts()
    task_level = Counts()
    per_check: dict[int, Counts] = defaultdict(Counts)

    for task_id in task_ids:
        gt = truth[task_id]
        pred = predicted[task_id]

        actual = set(gt.cited) - drop_claims.get(task_id, set())
        flagged = set(pred.flagged)
        if restrict is not None:
            actual &= restrict
            flagged &= restrict

        for check_id in actual | flagged:
            counts = per_check[check_id]
            if check_id in actual and check_id in flagged:
                counts.tp += 1
                check_level.tp += 1
            elif check_id in flagged:
                counts.fp += 1
                check_level.fp += 1
            else:
                counts.fn += 1
                check_level.fn += 1

        # A task fails QC if any cited issue sits in the fail band; the audit
        # fails it if any implemented check landed there.
        gt_fail = bool((gt.fail_checks - drop_claims.get(task_id, set())))
        pred_fail = bool(pred.fail_checks & restrict) if restrict else bool(pred.fail_checks)
        if gt_fail and pred_fail:
            task_level.tp += 1
        elif pred_fail:
            task_level.fp += 1
        elif gt_fail:
            task_level.fn += 1

    return check_level, task_level, per_check


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", required=True, help="the QC audit sheet CSV")
    ap.add_argument("--report", required=True, help="the audit JSON produced by the CLI")
    ap.add_argument(
        "--drop-claim",
        action="append",
        default=[],
        metavar="TASK:CHECK",
        help="a QC claim verified by hand as not reproducing; excluded from the adjusted view",
    )
    args = ap.parse_args(argv)

    truth = read_ground_truth(Path(args.sheet))
    predicted = read_predictions(Path(args.report))

    drop_claims: dict[str, set[int]] = defaultdict(set)
    for spec in args.drop_claim:
        task_id, _, check = spec.partition(":")
        drop_claims[task_id].add(int(check))

    shared = [t for t in truth if t in predicted]
    graded = [t for t in shared if truth[t].verdict in ("Pass", "Fail")]
    unreasoned = [
        t for t in graded if truth[t].verdict == "Fail" and not truth[t].fail_checks
    ]
    adjusted_ids = [t for t in graded if t not in unreasoned]

    implemented = (
        set(DETERMINISTIC) | set(RUBRIC_STAGE) | set(RATING_STAGE) | set(INFORMED_STAGE)
    )

    print(f"tasks in sheet: {len(truth)}   in report: {len(predicted)}   scored: {len(graded)}")
    print(f"  QC fail: {sum(1 for t in graded if truth[t].verdict == 'Fail')}   "
          f"QC pass: {sum(1 for t in graded if truth[t].verdict == 'Pass')}")
    if unreasoned:
        print(f"  failing with no stated reason (dropped when adjusted): {', '.join(unreasoned)}")
    for task_id, checks in drop_claims.items():
        print(f"  claim dropped when adjusted: {task_id} check {sorted(checks)}")
    print()

    for title, ids, drops in (
        ("RAW (every QC claim taken as correct)", graded, {}),
        ("ADJUSTED (unreasoned tasks and non-reproducing claims removed)", adjusted_ids, drop_claims),
    ):
        print(title)
        for scope_name, restrict in (
            ("all checks", None),
            ("implemented checks only", implemented),
        ):
            check_level, task_level, _ = score(truth, predicted, ids, restrict, drops)
            print(f" {scope_name}:")
            print(_line("issue detection (per check)", check_level))
            print(_line("task verdict (fail vs pass)", task_level))
        print()

    print("PER-CHECK BREAKDOWN (adjusted, implemented checks only)")
    _, _, per_check = score(truth, predicted, adjusted_ids, implemented, drop_claims)
    stage_of = {
        **{c: "deterministic" for c in DETERMINISTIC},
        **{c: "rubric" for c in RUBRIC_STAGE},
        **{c: "rating" for c in RATING_STAGE},
        **{c: "informed" for c in INFORMED_STAGE},
    }
    for check_id in sorted(per_check):
        counts = per_check[check_id]
        label = ERROR_CODES.get(check_id, {}).get("fail") or ERROR_CODES.get(check_id, {}).get(
            "non_fail", ""
        )
        print(
            f"  {check_id:4} {stage_of.get(check_id,'-'):14} tp={counts.tp:3} fp={counts.fp:3} "
            f"fn={counts.fn:3}  {label}"
        )

    print()
    print("UNIMPLEMENTED CHECKS QC CITED (pure recall loss, excluded above)")
    missing: dict[int, int] = defaultdict(int)
    for task_id in adjusted_ids:
        for check_id in truth[task_id].cited - implemented:
            missing[check_id] += 1
    for check_id, hits in sorted(missing.items(), key=lambda kv: -kv[1]):
        label = ERROR_CODES.get(check_id, {}).get("fail") or ""
        print(f"  {check_id:4} cited on {hits} task(s)  {label}")

    print()
    print("PER-TASK DETAIL (adjusted)")
    for task_id in sorted(adjusted_ids, key=lambda t: truth[t].verdict):
        gt, pred = truth[task_id], predicted[task_id]
        actual = (gt.cited - drop_claims.get(task_id, set())) & implemented
        flagged = pred.flagged & implemented
        print(
            f"  {task_id}  QC={gt.verdict:4} score={gt.score or '-':2}  "
            f"audit={pred.verdict:16} "
            f"hit={sorted(actual & flagged)} missed={sorted(actual - flagged)} "
            f"extra={sorted(flagged - actual)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
