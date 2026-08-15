"""Task roll-up and batch metrics.

Two habits from the previous project prevented most credibility disputes and are
enforced here: keep the unauditable bucket separate from quality rates
everywhere, and state denominators on the face of every table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .config import DEFAULT_POLICY, Policy
from .errors import is_known_code
from .registry import REGISTRY
from .scoring import CheckVerdict

TaskVerdict = Literal["unauditable", "fail", "clean", "pass_with_issues"]


@dataclass
class TaskRollup:
    task_id: str
    verdict: TaskVerdict
    fail_checks: list[int] = field(default_factory=list)
    non_fail_checks: list[int] = field(default_factory=list)
    not_evaluated_checks: list[int] = field(default_factory=list)
    verdicts: list[CheckVerdict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "verdict": self.verdict,
            "fail_checks": list(self.fail_checks),
            "non_fail_checks": list(self.non_fail_checks),
            "not_evaluated_checks": list(self.not_evaluated_checks),
        }


def validate_verdicts(verdicts: list[CheckVerdict]) -> None:
    """Reject structurally invalid output before it can reach a report."""
    for v in verdicts:
        if v.check_id not in REGISTRY:
            raise ValueError(f"unknown check_id {v.check_id}")
        if v.score == 2:
            raise ValueError(f"check {v.check_id} emitted score 2, which is unused")
        if v.band == "clean" and v.error_code is not None:
            raise ValueError(f"check {v.check_id} is clean but carries an error code")
        if v.band in ("fail", "non_fail"):
            if not v.error_code:
                raise ValueError(f"check {v.check_id} band {v.band} has no error code")
            if not is_known_code(v.error_code):
                raise ValueError(
                    f"check {v.check_id} emitted unknown error code {v.error_code!r}"
                )


def roll_up_task(
    task_id: str, verdicts: list[CheckVerdict], policy: Policy = DEFAULT_POLICY
) -> TaskRollup:
    validate_verdicts(verdicts)

    fails = [v.check_id for v in verdicts if v.band == "fail"]
    non_fails = [v.check_id for v in verdicts if v.band == "non_fail"]
    not_evaluated = [v.check_id for v in verdicts if v.band == "not_evaluated"]

    # Unauditable short-circuits: it is reported as its own bucket and never mixed
    # into quality fail rates.
    if 1000 in fails:
        verdict: TaskVerdict = "unauditable"
    elif fails:
        verdict = "fail"
    elif non_fails:
        verdict = "pass_with_issues"
    elif not_evaluated and policy.not_evaluated_blocks_clean:
        verdict = "pass_with_issues"
    else:
        verdict = "clean"

    return TaskRollup(
        task_id=task_id,
        verdict=verdict,
        fail_checks=sorted(fails),
        non_fail_checks=sorted(non_fails),
        not_evaluated_checks=sorted(not_evaluated),
        verdicts=list(verdicts),
    )


@dataclass
class CheckRate:
    check_id: int
    dimension: str
    sub_dimension: str
    denominator: int
    fails: int
    non_fails: int
    clean: int
    not_evaluated: int

    @property
    def fail_rate(self) -> float:
        return (self.fails / self.denominator) if self.denominator else 0.0

    @property
    def non_fail_rate(self) -> float:
        return (self.non_fails / self.denominator) if self.denominator else 0.0

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "dimension": self.dimension,
            "sub_dimension": self.sub_dimension,
            "denominator": self.denominator,
            "fails": self.fails,
            "non_fails": self.non_fails,
            "clean": self.clean,
            "not_evaluated": self.not_evaluated,
            "fail_rate": round(self.fail_rate, 4),
            "non_fail_rate": round(self.non_fail_rate, 4),
        }


@dataclass
class BatchReport:
    task_counts: dict[str, int] = field(default_factory=dict)
    check_rates: list[CheckRate] = field(default_factory=list)
    error_code_distribution: dict[str, int] = field(default_factory=dict)
    gate_rate_distribution: dict[int, list[float]] = field(default_factory=dict)
    # The distribution above is anonymous, so the batch can report that some task
    # ran at 40% without ever saying which, and a reader chasing the outlier picks
    # one from the fail list. Keyed by task, the same rates can be checked.
    gate_rate_by_task: dict[int, dict[str, float]] = field(default_factory=dict)
    # What the checks found and declined to count, and why, across the batch.
    #
    # Without this the change that produced it is invisible at the level anyone
    # reads: checks 300, 400 and 95 now each require a second, informed call to
    # confirm a finding before it is published, and a reader comparing this run's
    # fail rate against an earlier one has no way to tell a genuine improvement in
    # the work from findings this build stopped counting. Each key names the reason
    # a finding was suppressed, so the two questions -- "how many defects" and "how
    # many artifacts of our own measurement" -- have separate answers.
    suppressed_findings: dict[str, int] = field(default_factory=dict)
    # Tasks whose check-300 or check-400 verdict rests on nothing a confirming
    # reader supported. The plainest available answer to "did this fail purely
    # because of the blind rater".
    blind_only_tasks: dict[int, list[str]] = field(default_factory=dict)
    policy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_counts": dict(self.task_counts),
            "check_rates": [r.to_dict() for r in self.check_rates],
            "suppressed_findings": dict(sorted(self.suppressed_findings.items())),
            "blind_only_tasks": {
                k: sorted(v) for k, v in sorted(self.blind_only_tasks.items())
            },
            "error_code_distribution": dict(self.error_code_distribution),
            "gate_rate_distribution": {
                k: [round(v, 4) for v in vals]
                for k, vals in self.gate_rate_distribution.items()
            },
            "gate_rate_by_task": {
                k: {task_id: round(v, 4) for task_id, v in sorted(rates.items())}
                for k, rates in self.gate_rate_by_task.items()
            },
            "policy": dict(self.policy),
        }


def _tally_suppressed_findings(report: BatchReport, rollups: list[TaskRollup]) -> None:
    """Count what the confirming passes took out of the fail rate, by reason.

    Read off the verdicts' own measurement counts rather than recomputed, so this
    can never disagree with the per-task numbers it summarises. A check that
    publishes none of these keys contributes nothing here, which is what keeps this
    from needing to know which checks confirm anything.
    """

    def bump(key: str, n: int) -> None:
        if n:
            report.suppressed_findings[key] = report.suppressed_findings.get(key, 0) + n

    for r in rollups:
        for v in r.verdicts:
            counts = v.measurement.counts
            if v.check_id == 300:
                bump("300_blind_only_rating_gaps", counts.get("blind_only_not_counted", 0))
                bump(
                    "300_truncated_evidence_rating_gaps",
                    counts.get("truncated_evidence_not_counted", 0),
                )
                bump("300_unadjudicated_rating_gaps", counts.get("unadjudicated_not_counted", 0))
                if counts.get("blind_disagreement_only"):
                    report.blind_only_tasks.setdefault(300, []).append(v.task_id)
            elif v.check_id == 400:
                reason = counts.get("not_counted_because") or ""
                if reason:
                    bump(f"400_ranking_gap_{reason}", 1)
                if counts.get("blind_disagreement_only"):
                    report.blind_only_tasks.setdefault(400, []).append(v.task_id)
            elif v.check_id == 95:
                bump("95_cleared_by_confirmation", len(counts.get("cleared_by_confirmation") or []))


def build_batch_report(
    rollups: list[TaskRollup], policy: Policy = DEFAULT_POLICY
) -> BatchReport:
    report = BatchReport(policy=policy.provenance())

    for r in rollups:
        report.task_counts[r.verdict] = report.task_counts.get(r.verdict, 0) + 1
    report.task_counts["total"] = len(rollups)

    # "Unauditable" flags a task-level integrity problem -- almost always an
    # empty rubric -- not proof that every check on the task is fiction. A
    # check that genuinely never ran (because it reads the rubric, or
    # because the task failed a more fundamental integrity check like a
    # provenance mismatch) reports `not_evaluated` and is excluded from its
    # own denominator below via the same band filter every other task gets.
    # A check that did run despite the flag -- the blind SxS pick, a
    # contributor's justification, neither of which reads the rubric --
    # still counts. Only check 1000 itself, and this task-level bucket, stay
    # carved out of every quality rate.
    auditable = [r for r in rollups if r.verdict != "unauditable"]
    report.task_counts["auditable"] = len(auditable)

    _tally_suppressed_findings(report, rollups)

    per_check: dict[int, list[CheckVerdict]] = {}
    for r in rollups:
        for v in r.verdicts:
            if v.check_id == 1000:
                continue
            per_check.setdefault(v.check_id, []).append(v)

    for check_id in sorted(per_check):
        vs = per_check[check_id]
        spec = REGISTRY[check_id]
        report.check_rates.append(
            CheckRate(
                check_id=check_id,
                dimension=spec.dimension,
                sub_dimension=spec.sub_dimension,
                denominator=sum(1 for v in vs if v.band != "not_evaluated"),
                fails=sum(1 for v in vs if v.band == "fail"),
                non_fails=sum(1 for v in vs if v.band == "non_fail"),
                clean=sum(1 for v in vs if v.band == "clean"),
                not_evaluated=sum(1 for v in vs if v.band == "not_evaluated"),
            )
        )

    for r in rollups:
        for v in r.verdicts:
            if v.check_id == 1000:
                continue
            if v.error_code:
                report.error_code_distribution[v.error_code] = (
                    report.error_code_distribution.get(v.error_code, 0) + 1
                )
            # A batch failing the 10% gate at 11% is a very different conversation
            # from one failing at 40%, so keep the underlying rates, not just the split.
            if v.check_id in (230, 240, 250, 260, 270) and v.measurement.rate is not None:
                report.gate_rate_distribution.setdefault(v.check_id, []).append(
                    v.measurement.rate
                )
                report.gate_rate_by_task.setdefault(v.check_id, {})[r.task_id] = (
                    v.measurement.rate
                )

    return report
