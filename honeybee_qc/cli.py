"""The audit CLI.

    python -m honeybee_qc.cli tasks.jsonl --out report.json
    python -m honeybee_qc.cli tasks.jsonl --verify-provenance --snapshot-dir snaps
    python -m honeybee_qc.cli tasks.jsonl --rubric-stage --rating-stage --cache-db c.db

Stages run cheapest first, and each one can retire a task before a more
expensive stage pays for it:

  1. Deterministic: preflight, check 90 (links), check 96 (minimum turns), check
     100's turn-1 fail, check 460's direction contradiction. No model calls.
  2. Provenance (--verify-provenance): renders each share link and compares it to
     the uploaded PDF. Browser work, no model calls.
  3. Rubric stage (--rubric-stage): one call per criterion, roughly 21 per task.
  4. Rating stage (--rating-stage): the blind pass over checks 270, 300, and 400,
     roughly 53 calls per task and by far the most expensive thing here.
  5. Informed stage (--informed-stage): checks 70, 75, 80, 85, 95, 110, 220, 280,
     310, 450, and 470, roughly 14 calls per task. Mostly audits the
     contributor's stated reasoning, so it sees their work and must run after
     the blind pass.

A task found unauditable for a structural reason -- a provenance mismatch, a
cross-task duplicate, a malformed field -- never reaches stages 3, 4, or 5:
none of those checks can be trusted once the submission itself is suspect.
An empty rubric is narrower. It zeroes stage 3 outright (nothing to build a
criterion audit from) and the two rating-stage/informed-stage checks that
grade against specific criteria (270, 280), but the blind SxS pick (400),
the 8-dimension ratings (300), the contributor's justifications (450), and
the rest of the informed stage still read the transcript, not the rubric,
and still run. Checks with no stage yet report not_evaluated, so the report
shape does not change as phases land.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import DEFAULT_POLICY, Policy
from .gates import (
    build_inversion_findings,
    evaluate_check_91,
    evaluate_check_96,
    evaluate_check_97,
    evaluate_check_100,
    evaluate_check_460,
)
from .hydrate import hydrate_conversations
from .informed_stages import (
    CALL_NAME_TO_CHECK,
    INFORMED_CHECKS,
    InformedStageResult,
    estimate_informed_calls,
    informed_calls_by_check,
    not_evaluated_informed_verdicts,
    run_informed_stage,
)
from .links import evaluate_check_90
from .llm import DEFAULT_EFFORT, DEFAULT_MODEL, ModelClient, build_client
from .models import Task
from .ingest import load_taskattempts_csv
from .projects import (
    PROFILES,
    disabled_checks_for,
    drop_disabled,
    profile_by_key,
    profile_for_project,
)
from .preflight import RUBRIC_EMPTY_REASON, PreflightResult, load_tasks, validate_batch
from .provenance import DuplicateFinding
from .rating_stages import (
    RatingStageResult,
    estimate_adjudication_calls,
    estimate_rating_calls,
    not_evaluated_rating_verdicts,
    run_rating_stage,
    split_check_270_by_model,
)
from .registry import ORDER, REGISTRY
from .rollup import build_batch_report, roll_up_task
from .sampling import project_cost
from .scoring import CheckVerdict, Measurement, build_verdict
from .stages import (
    RubricStageResult,
    not_evaluated_rubric_verdicts,
    run_rubric_stage,
    stage_policy_version,
)
from .verify import (
    TaskProvenance,
    collect_duplicates,
    dead_link_reasons,
    unauditable_reasons,
    verify_task,
)

DETERMINISTIC_NOW = (90, 91, 96, 97, 100, 460)
RUBRIC_STAGE_CHECKS = (200, 210, 230, 240, 250, 260)
RATING_STAGE_CHECKS = (270, 300, 400)
INFORMED_STAGE_CHECKS = INFORMED_CHECKS

# The restricted-scope run: rubric-rating correctness (270, reported per model
# via `split_check_270_by_model` as well as pooled), SxS likert/ranking (400,
# 460), turn-citation correctness (280, 310), and justification/verdict quality
# (450, 470) -- and nothing from the rubric-authoring stage or check 300 (the
# 8-dimension ratings), which this run deliberately does not grade.
RESTRICTED_CHECKS = (270, 280, 310, 400, 450, 460, 470)


def run_deterministic(
    task: Task,
    policy: Policy = DEFAULT_POLICY,
    provenance: TaskProvenance | None = None,
    extra: list[CheckVerdict] | None = None,
) -> list[CheckVerdict]:
    # Provenance is the verifier's channel and is off on most runs; hydration
    # sees the same dead page on every run that fetches. Both render the reason
    # identically so a link found dead by each is counted once.
    dead_links = list(
        dict.fromkeys(
            dead_link_reasons(provenance)
            + [
                f"final_link_{sub.model}:share page unavailable"
                for sub in task.submissions()
                if sub.share_page_dead
            ]
        )
    )
    disabled = disabled_checks_for(task.project_id)
    verdicts = []
    if 90 not in disabled:
        verdicts.append(evaluate_check_90(task, policy, dead_links=dead_links))
    if 91 not in disabled:
        verdicts.append(evaluate_check_91(task, policy))
    if 96 not in disabled:
        verdicts.append(evaluate_check_96(task, policy))
    if 97 not in disabled:
        verdicts.append(evaluate_check_97(task, policy))
    if 100 not in disabled:
        verdicts.append(evaluate_check_100(task, auditor_key_turn=None, policy=policy))
    if 460 not in disabled:
        verdicts.append(
            evaluate_check_460(task, build_inversion_findings(task, policy=policy), policy)
        )
    verdicts += list(extra or [])
    covered = {v.check_id for v in verdicts}
    for check_id in ORDER:
        if check_id in covered or check_id in disabled:
            continue
        verdicts.append(
            build_verdict(
                task_id=task.task_id,
                check_id=check_id,
                band="not_evaluated",
                measurement=Measurement(
                    notes="Requires a model stage; not implemented in phase 1."
                ),
                policy=policy,
            )
        )
    return sorted(verdicts, key=lambda v: v.check_id)


def unauditable_verdict(task_id: str, reasons: list[str], policy: Policy) -> CheckVerdict:
    return build_verdict(
        task_id=task_id,
        check_id=1000,
        band="fail",
        measurement=Measurement(
            counts={"reasons": len(reasons)}, notes="; ".join(reasons)
        ),
        contributing_items=reasons,
        policy=policy,
    )


def rubric_empty_check_1000_verdict(task_id: str, policy: Policy) -> CheckVerdict:
    """Check 1000 for the `rubric_empty_only` case: never a fail.

    `RUBRIC_EMPTY_REASON` zeroes the denominator for the rubric-authoring
    checks and 270/280, exactly like 220/280 report `not_evaluated` when they
    have no rubric to read (see `informed_stages.py`'s `_not_evaluated` call
    sites). It says nothing about whether the task itself is auditable, so
    1000 must land in the same band those checks do rather than fail --
    otherwise every task in a batch pulled specifically for empty rubrics
    would roll up "unauditable" over an expected, benign condition.
    """
    return build_verdict(
        task_id=task_id,
        check_id=1000,
        band="not_evaluated",
        measurement=Measurement(
            notes="rubric is empty by design; task is otherwise auditable"
        ),
        policy=policy,
    )


@dataclass
class _TaskOutcome:
    """One task's results, kept together so a parallel run can be reordered."""

    task_id: str = ""
    rollup: object | None = None
    rubric: RubricStageResult | None = None
    rating: RatingStageResult | None = None
    informed: InformedStageResult | None = None
    errors: list[StageError] = field(default_factory=list)
    # Populated only in restricted-scope runs that keep check 270: the pooled
    # verdict in `rollup.verdicts` stays the source of truth, this is the same
    # numbers split out by model slot.
    check_270_by_model: dict[str, CheckVerdict] = field(default_factory=dict)


@dataclass
class StageError:
    """A stage that raised, recorded instead of ending the batch.

    A run costs hours and money, so one task's bug must not discard the other
    sixteen. The failure is reported as loudly as a crash would have been, but
    the checks it fed become `not_evaluated` rather than absent.
    """

    task_id: str
    stage: str
    error: str

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "stage": self.stage, "error": self.error}


@dataclass
class TaskProgress:
    """One task's result, at the moment it finished rather than at the end.

    A full batch runs for hours and the report is written only once, at the
    close, so without this the operator has no way to tell a slow run from a
    wedged one, and no pass/fail signal until every task is done.
    """

    done: int
    total: int
    task_id: str
    verdict: str
    fail_checks: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    errored: bool = False

    def line(
        self, spent_usd: float, failed_so_far: int, errored_so_far: int = 0
    ) -> str:
        """A single line, self-contained enough to read in a tailed log.

        Errored tasks are counted apart from failing ones: a task that crashed
        has no verdict, and folding it into the fail count would misreport a bug
        in the pipeline as a defect in the work being audited.
        """
        checks = (
            " fails=" + ",".join(str(c) for c in self.fail_checks)
            if self.fail_checks
            else ""
        )
        errors = f", {errored_so_far} errored" if errored_so_far else ""
        return (
            f"[{self.done}/{self.total}] {self.task_id} {self.verdict}{checks} "
            f"{self.elapsed_s:.0f}s ${self.cost_usd:.2f} "
            f"| {failed_so_far} failing{errors}, ${spent_usd:.2f} spent"
        )


def _progress_for(
    outcome: _TaskOutcome, done: int, total: int, elapsed_s: float
) -> TaskProgress:
    """Read a finished task's headline numbers off its outcome.

    A task that died before rollup has no verdict of its own; it is reported as
    `error` rather than silently as a pass, since a crashed task is exactly the
    kind of thing an operator watching a long run needs to see immediately.
    """
    stages = (outcome.rubric, outcome.rating, outcome.informed)
    return TaskProgress(
        done=done,
        total=total,
        task_id=outcome.task_id,
        verdict=outcome.rollup.verdict if outcome.rollup is not None else "error",
        fail_checks=list(outcome.rollup.fail_checks) if outcome.rollup else [],
        cost_usd=sum(s.cost_usd for s in stages if s is not None),
        elapsed_s=elapsed_s,
        errored=bool(outcome.errors),
    )


class ProgressPrinter:
    """Prints one line per finished task, with running totals.

    Writes to stderr because stdout may carry the report itself. Counting and
    printing hold a lock: tasks finish on several threads at once, and a torn
    line in the log is worse than no line at all.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.spent_usd = 0.0
        self.failed = 0
        self.errored = 0
        self._lock = threading.Lock()

    def __call__(self, progress: TaskProgress) -> None:
        with self._lock:
            self.spent_usd += progress.cost_usd
            if progress.verdict in ("fail", "unauditable"):
                self.failed += 1
            if progress.errored:
                self.errored += 1
            print(
                progress.line(self.spent_usd, self.failed, self.errored),
                file=self.stream,
                flush=True,
            )


class ProgressJsonl:
    """Appends one finished task per line, so a stopped run keeps its verdicts.

    The report is written once, at the close of the batch, which means a run
    killed partway through loses every verdict it had already computed. The
    model calls themselves survive in the cache, but the verdicts do not, and
    recovering them costs a second pass over the whole batch. Appending and
    flushing per task turns that into a file the next run can read.

    Opened in append mode on purpose: a restart adds to the record rather than
    truncating what the previous attempt managed to finish.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, progress: TaskProgress) -> None:
        record = {
            "task_id": progress.task_id,
            "verdict": progress.verdict,
            "fail_checks": list(progress.fail_checks),
            "cost_usd": progress.cost_usd,
            "elapsed_s": progress.elapsed_s,
            "errored": progress.errored,
            "done": progress.done,
            "total": progress.total,
            "finished_at": time.time(),
        }
        with self._lock:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class ProgressFanout:
    """Feeds each finished task to several sinks.

    One sink raising must not rob the others of the record, so each is called
    in isolation; losing the durable line because a print failed would defeat
    the point of writing it.
    """

    def __init__(self, *sinks) -> None:
        self.sinks = [s for s in sinks if s is not None]

    def __call__(self, progress: TaskProgress) -> None:
        for sink in self.sinks:
            try:
                sink(progress)
            except Exception:
                continue


def audit_batch(
    tasks: list[Task],
    policy: Policy = DEFAULT_POLICY,
    fetch=None,
    read_pdf=None,
    client: ModelClient | None = None,
    workers: int = 4,
    rubric_stage: bool = True,
    rating_stage: bool = False,
    informed_stage: bool = False,
    task_workers: int = 1,
    restricted_checks: tuple[int, ...] | None = None,
    skip_rating_checks: tuple[int, ...] | None = None,
    on_task_done: Callable[[TaskProgress], None] | None = None,
) -> dict:
    preflight: list[PreflightResult] = validate_batch(
        tasks, policy, required_checks=restricted_checks
    )

    provenance: list[TaskProvenance] = []
    duplicates: list[DuplicateFinding] = []
    if policy.verify_provenance:
        provenance = [verify_task(t, policy, fetch, read_pdf) for t in tasks]
        duplicates = collect_duplicates(tasks, provenance)
    by_task = {p.task_id: p for p in provenance}

    def audit_one(task: Task, pre: PreflightResult) -> _TaskOutcome:
        tp = by_task.get(task.task_id)
        reasons = list(pre.hard_failures)
        reasons += unauditable_reasons(task.task_id, tp, duplicates, policy)
        unauditable = bool(reasons)
        # An empty rubric, and nothing else wrong, blocks only the checks
        # that read the rubric (stage 3 in full, plus 270/280 downstream).
        # Any other reason -- a provenance mismatch, a duplicate, a
        # malformed field -- means the submission itself cannot be trusted,
        # so it still blocks every model stage.
        rubric_empty_only = reasons == [RUBRIC_EMPTY_REASON]
        fully_unauditable = unauditable and not rubric_empty_only
        out = _TaskOutcome(task_id=task.task_id)

        def guarded(stage: str, run, on_failure) -> list[CheckVerdict]:
            """Run one stage; on failure record it and fall back to not_evaluated.

            KeyboardInterrupt and SystemExit are deliberately not caught, so an
            operator can still stop the run.
            """
            try:
                return run()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                detail = f"{type(exc).__name__}: {exc}"
                out.errors.append(StageError(task.task_id, stage, detail))
                print(
                    f"  error: {task.task_id} {stage} stage failed, continuing "
                    f"({detail})",
                    file=sys.stderr,
                    flush=True,
                )
                return on_failure(f"{stage} stage failed: {detail}")

        # Every model stage costs real money, so none runs for a task where it
        # cannot produce anything real. Stage 3 is skipped whenever the task
        # is unauditable for any reason -- an empty rubric leaves it nothing
        # to build a criterion audit from either way.
        extra: list[CheckVerdict] = []
        if client is not None and rubric_stage:
            if unauditable:
                extra += not_evaluated_rubric_verdicts(
                    task, "task is unauditable; rubric stage skipped", policy
                )
            else:

                def run_rubric() -> list[CheckVerdict]:
                    out.rubric = run_rubric_stage(task, client, policy, workers=workers)
                    return list(out.rubric.verdicts)

                extra += guarded(
                    "rubric",
                    run_rubric,
                    lambda why: not_evaluated_rubric_verdicts(task, why, policy),
                )

        if client is not None and rating_stage:
            if fully_unauditable:
                extra += not_evaluated_rating_verdicts(
                    task, "task is unauditable; rating stage skipped", policy
                )
            else:
                # An empty rubric still reaches here: `run_rating_stage`
                # already builds zero 270 requests when `task.rubric` is
                # empty and reports that check not_evaluated on its own, but
                # 300 (the 8-dimension ratings) and 400 (the blind SxS pick)
                # never read the rubric at all and still get judged.
                rating_checks = (
                    set(restricted_checks) if restricted_checks is not None else None
                )
                if skip_rating_checks:
                    base = (
                        rating_checks
                        if rating_checks is not None
                        else set(RATING_STAGE_CHECKS)
                    )
                    rating_checks = base - set(skip_rating_checks)

                def run_rating() -> list[CheckVerdict]:
                    out.rating = run_rating_stage(
                        task, client, policy, workers=workers, checks=rating_checks
                    )
                    if restricted_checks and 270 in restricted_checks:
                        out.check_270_by_model = split_check_270_by_model(
                            task, out.rating, policy
                        )
                    return list(out.rating.verdicts)

                extra += guarded(
                    "rating",
                    run_rating,
                    lambda why: not_evaluated_rating_verdicts(task, why, policy),
                )

        # Runs after the blind pass and never feeds it: these prompts carry the
        # contributor's own ratings, which would anchor 270, 300, and 400.
        if client is not None and informed_stage:
            if fully_unauditable:
                extra += not_evaluated_informed_verdicts(
                    task, "task is unauditable; informed stage skipped", policy
                )
            else:
                # An empty rubric still runs this stage: 220 (rubric autofail)
                # and 280 (turn citations for criterion ratings) decide their
                # own not_evaluated internally when there is no rubric to
                # read, but 70/75/80/85/95/110 (prompt and target-outcome
                # quality), 310 (turn citations for the dimension ratings),
                # 450 (justifications), and 470 (SxS verdict) don't depend on
                # the rubric and still get judged.

                def run_informed() -> list[CheckVerdict]:
                    informed_checks = (
                        set(restricted_checks) if restricted_checks is not None else None
                    )
                    out.informed = run_informed_stage(
                        task,
                        client,
                        policy,
                        workers=workers,
                        checks=informed_checks,
                    )
                    return list(out.informed.verdicts)

                extra += guarded(
                    "informed",
                    run_informed,
                    lambda why: not_evaluated_informed_verdicts(task, why, policy),
                )

        verdicts = run_deterministic(task, policy, tp, extra=extra)
        if reasons:
            verdicts = [v for v in verdicts if v.check_id != 1000]
            # Only a genuine integrity problem fails 1000 and rolls the task
            # up as "unauditable". `rubric_empty_only` reaches this branch
            # too (reasons is non-empty), but an empty rubric alone must not
            # -- see `rubric_empty_check_1000_verdict`.
            if fully_unauditable:
                verdicts.append(unauditable_verdict(task.task_id, reasons, policy))
            else:
                verdicts.append(rubric_empty_check_1000_verdict(task.task_id, policy))
            verdicts.sort(key=lambda v: v.check_id)
        if restricted_checks is not None:
            # Every other check still ran deterministically (or, for the
            # informed stage, cannot be withheld without risking a gate that
            # expects real data -- see `run_informed_stage`'s docstring) and
            # would otherwise report a `clean`/`not_evaluated` verdict for a
            # dimension this run was never asked to grade. Dropping them here,
            # after rollup would have seen them, keeps the report to exactly
            # the checks in scope.
            verdicts = [
                v for v in verdicts if v.check_id in restricted_checks or v.check_id == 1000
            ]
        out.rollup = roll_up_task(task.task_id, verdicts, policy)
        return out

    done = itertools.count(1)

    def audit_one_isolated(task: Task, pre: PreflightResult) -> _TaskOutcome:
        """The last line of defence: no single task can end the batch."""
        started = time.monotonic()
        try:
            outcome = audit_one(task, pre)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            detail = f"{type(exc).__name__}: {exc}"
            print(
                f"  error: {task.task_id} could not be audited at all, continuing "
                f"({detail})\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            outcome = _TaskOutcome(
                task_id=task.task_id,
                errors=[StageError(task.task_id, "task", detail)],
            )
        if on_task_done is not None:
            # Reporting progress must never be the thing that kills a task that
            # otherwise succeeded, so this mirrors the isolation above.
            try:
                on_task_done(
                    _progress_for(
                        outcome,
                        done=next(done),
                        total=len(tasks),
                        elapsed_s=time.monotonic() - started,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - progress is not worth a task
                print(
                    f"  warning: progress reporting failed for {task.task_id} "
                    f"({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                    flush=True,
                )
        return outcome

    # Auditing tasks concurrently is what keeps the worker pool full: stages that
    # batch per model build only a handful of requests per task, far too few to
    # saturate it alone. Total in-flight calls stay bounded because the client is
    # throttled globally, not per pool.
    if task_workers > 1 and client is not None and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=task_workers) as pool:
            outcomes = list(pool.map(audit_one_isolated, tasks, preflight))
    else:
        outcomes = [audit_one_isolated(t, p) for t, p in zip(tasks, preflight)]

    # Reassembled in task order, so a report never depends on which task finished
    # first.
    rollups = [o.rollup for o in outcomes if o.rollup is not None]
    stage_results: list[RubricStageResult] = [
        o.rubric for o in outcomes if o.rubric is not None
    ]
    rating_results: list[RatingStageResult] = [
        o.rating for o in outcomes if o.rating is not None
    ]
    informed_results: list[InformedStageResult] = [
        o.informed for o in outcomes if o.informed is not None
    ]

    stage_errors = [e for o in outcomes for e in o.errors]

    report = build_batch_report(rollups, policy)
    check_270_by_model = {
        o.task_id: {model: v.to_dict() for model, v in o.check_270_by_model.items()}
        for o in outcomes
        if o.check_270_by_model
    }
    return {
        "batch": report.to_dict(),
        **({"check_270_by_model": check_270_by_model} if check_270_by_model else {}),
        "stage_errors": [e.to_dict() for e in stage_errors],
        "preflight": [p.to_dict() for p in preflight],
        "provenance": [p.to_dict() for p in provenance],
        "duplicates": [d.to_dict() for d in duplicates],
        "rubric_stage": [s.to_dict() for s in stage_results],
        "rating_stage": [r.to_dict() for r in rating_results],
        "informed_stage": [i.to_dict() for i in informed_results],
        "cost_usd": round(
            sum(s.cost_usd for s in stage_results)
            + sum(r.cost_usd for r in rating_results)
            + sum(i.cost_usd for i in informed_results),
            4,
        ),
        "review_queue": [
            {"task_id": p.task_id, "model": r.model, "verdict": r.verdict,
             "reasons": r.reasons}
            for p in provenance
            for r in p.needs_review
        ],
        "tasks": [
            {**r.to_dict(), "checks": [v.to_dict() for v in r.verdicts]}
            for r in rollups
        ],
    }


def measured_unit_costs(cache_path: str) -> dict[int, float]:
    """Mean dollars per call, per check, from a completed run's cache.

    A projection built on this is a restatement of a bill the user has already
    paid rather than an estimate, which matters because the sampled checks are the
    expensive ones: a 450 call costs an order of magnitude more than a 470 call, so
    a flat per-call average would misprice every configuration.
    """
    totals: dict[int, list[float]] = {}
    try:
        conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True, timeout=10)
        rows = conn.execute("SELECT request_key, cost_usd FROM response_cache").fetchall()
        conn.close()
    except Exception as exc:  # noqa: BLE001 - a missing cache must not end a dry run
        print(f"could not price from {cache_path}: {exc}", file=sys.stderr)
        return {}

    for request_key, cost in rows:
        parts = str(request_key).split("::")
        name = parts[1] if len(parts) > 1 else parts[0]
        check_id = CALL_NAME_TO_CHECK.get(name)
        if check_id is not None and cost:
            totals.setdefault(check_id, []).append(float(cost))
    return {cid: sum(v) / len(v) for cid, v in totals.items() if v}


def report_sampling_cost(
    tasks: list[Task], policy: Policy, cache_path: str | None
) -> None:
    """What the requested sample configuration costs against a single pass.

    Printed on every informed-stage dry run because this is the user's money and
    the sample count is the one knob that multiplies it.
    """
    calls_by_check: dict[int, int] = {}
    for task in tasks:
        for check_id, n in informed_calls_by_check(task).items():
            calls_by_check[check_id] = calls_by_check.get(check_id, 0) + n

    unit = measured_unit_costs(cache_path) if cache_path else {}
    cost = project_cost(calls_by_check, unit, policy)
    payload = cost.to_dict()

    print(
        f"sampling: {payload['single_sample_calls']} judgments -> "
        f"{payload['sampled_calls']} calls "
        f"(+{payload['extra_calls']} for repeats)",
        file=sys.stderr,
    )
    for check_id in sorted(calls_by_check):
        samples = cost.samples_by_check.get(check_id, 1)
        if samples > 1:
            print(
                f"  check {check_id}: {calls_by_check[check_id]} judgments x "
                f"{samples} draws",
                file=sys.stderr,
            )
    if unit:
        print(
            f"  projected spend ${payload['sampled_usd']:.2f} against "
            f"${payload['single_sample_usd']:.2f} at one draw "
            f"(+${payload['extra_usd']:.2f}), priced from {cache_path}",
            file=sys.stderr,
        )
    else:
        print(
            "  no per-call prices available; pass --price-from-cache to value this "
            "configuration in dollars",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HoneyBee_v2 QC deterministic audit")
    ap.add_argument("input", help="JSONL of tasks, or a taskattempts CSV with --from-snowflake")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument(
        "--from-snowflake",
        action="store_true",
        help="read a taskattempts export (task, response) instead of the JSONL contract",
    )
    ap.add_argument(
        "--project",
        default="",
        metavar="KEY",
        help="force a project profile (l1, l1_sxs, aspirational, …) instead of "
             "resolving from each row's project id. Needed when one project id "
             "has more than one taxonomy revision",
    )
    ap.add_argument(
        "--verify-provenance",
        action="store_true",
        help="render each share link and check it matches the uploaded PDF",
    )
    ap.add_argument("--snapshot-dir", default=None, help="where to archive rendered DOMs")
    ap.add_argument("--fetch-timeout", type=int, default=None, help="per-link seconds")
    ap.add_argument(
        "--hydrate-workers",
        type=int,
        default=None,
        help="browser renders to run at once, defaulting to --workers. Hydration is "
        "the one CPU-bound stage -- every worker is a headless Chrome -- so the "
        "count that keeps the judging stages busy will thrash the machine here and "
        "return empty DOMs. Set this lower than --workers to render gently and "
        "still judge at full width.",
    )
    ap.add_argument(
        "--fetch-conversations",
        action="store_true",
        help="render each final share link so the rating stage has turns to judge; "
             "without it that stage abstains on every task ingested from Snowflake",
    )
    ap.add_argument(
        "--rubric-stage",
        action="store_true",
        help="run checks 200/230/240/250/260 (one model call per rubric criterion)",
    )
    ap.add_argument(
        "--rating-stage",
        action="store_true",
        help="run the blind pass over checks 270/300/400 (~53 model calls per task)",
    )
    ap.add_argument(
        "--informed-stage",
        action="store_true",
        help="run the informed pass over checks 70/75/80/85/95/110/220/280/310/450/470 "
             "(~14 model calls per task; shows the contributor's own work, so it "
             "runs after the blind pass and never feeds it)",
    )
    ap.add_argument(
        "--restricted-checks",
        action="store_true",
        help=f"report only checks {list(RESTRICTED_CHECKS)} -- rubric-rating "
             "correctness (270, also split per model), SxS likert/ranking "
             "(400/460), turn-citation correctness (280/310), and "
             "justification/verdict quality (450/470). Skips check 300's model "
             "calls entirely; pass no --rubric-stage to also skip the "
             "rubric-authoring checks, which this does not touch",
    )
    ap.add_argument(
        "--audit-slots",
        default="",
        metavar="SLOT,SLOT",
        help="only grade per-model checks (270/280/300/310/90/95/450-dims) for "
             "these slots (e.g. D). SxS comparisons still run in full because "
             "both sides of the pair have to be in front of the judge",
    )
    ap.add_argument(
        "--relabel-c-to-b",
        default="",
        metavar="FILE",
        help="task IDs (one per line) whose C trajectory is GPT: promote C to B, "
             "keep the AC ranking pair rewritten as A-vs-B, and drop Claude-as-B "
             "plus the simplified trajectory so they are not hydrated",
    )
    ap.add_argument(
        "--only-checks",
        default="",
        metavar="CHECK,CHECK",
        help="report and spend model calls only on these check IDs. Overrides "
             "--restricted-checks when both are passed",
    )
    ap.add_argument(
        "--fill-missing-dims",
        default="",
        metavar="SLOT",
        help="for tasks missing 1-5 dimension ratings on this slot, generate "
             "scores and contributor-voice justifications to a CSV (not fed "
             "into check 300)",
    )
    ap.add_argument(
        "--fill-out",
        default=None,
        help="where to write --fill-missing-dims CSV; defaults next to --out",
    )
    ap.add_argument(
        "--skip-rating-checks",
        default="",
        metavar="CHECK,CHECK",
        help="comma-separated check IDs to drop from the rating stage's model "
             f"calls entirely, out of {list(RATING_STAGE_CHECKS)} (e.g. '270' to "
             "skip rubric-rating correctness while still spending calls on "
             "300/400). A dropped check reports not_evaluated for free, the "
             "same as thin evidence -- no separate skip case was needed. "
             "Ignored where --restricted-checks already fixes which of "
             "270/300/400 run",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many model calls the requested stages would make, then exit",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument(
        "--workers",
        type=int,
        default=30,
        help="cap on model calls in flight at once, across every task and stage",
    )
    ap.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="how many tasks to audit at once. Raise this when a stage builds too "
             "few calls per task to keep --workers busy: the informed stage builds "
             "13, so it idles most of a large pool on its own",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress the per-task progress line. Progress goes to stderr as each "
             "task finishes, so a run of any size reports pass/fail and spend "
             "while it works instead of only in the final report",
    )
    ap.add_argument("--cache-db", default=None, help="SQLite path for response caching")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--model-timeout", type=int, default=300)
    ap.add_argument(
        "--samples",
        action="append",
        default=[],
        metavar="CHECK=N",
        help="draw a check's judgments N times and keep only what a majority of "
             "draws agree on. Repeatable. Overrides the policy default per check, "
             "because sampling every check costs several times a single pass",
    )
    ap.add_argument(
        "--model-for",
        action="append",
        default=[],
        metavar="CHECK=MODEL",
        dest="model_for",
        help="send one check's calls to a different model than --model. "
             "Repeatable. Neither cost nor accuracy is spread evenly over the "
             "checks, so a cheap judge that is good enough for most of them can "
             "be bought back to the strong one where it is not",
    )
    ap.add_argument(
        "--no-sampling",
        action="store_true",
        help="one draw per judgment for every check, which is the cheapest "
             "configuration and the one whose findings a rerun disagrees with",
    )
    ap.add_argument(
        "--price-from-cache",
        default=None,
        metavar="PATH",
        help="with --dry-run, price the requested sample configuration off the "
             "measured per-call spend in an existing cache rather than a guess",
    )
    args = ap.parse_args(argv)

    policy = DEFAULT_POLICY
    overrides = {}
    if args.no_sampling:
        overrides["samples_by_check"] = {}
    elif args.samples:
        samples = dict(DEFAULT_POLICY.samples_by_check)
        for spec in args.samples:
            check, _, count = spec.partition("=")
            if not count.isdigit():
                ap.error(f"--samples expects CHECK=N, got {spec!r}")
            samples[int(check)] = int(count)
        overrides["samples_by_check"] = samples

    # Checks are named by id on the command line and by request-kind inside the
    # client, and one check can own several kinds (450 is asked twice, once per
    # justification target). Translating here keeps `llm` free of the stage
    # vocabulary.
    model_by_kind: dict[str, str] = {}
    for spec in args.model_for:
        check, _, target = spec.partition("=")
        if not check.strip().isdigit() or not target.strip():
            ap.error(f"--model-for expects CHECK=MODEL, got {spec!r}")
        check_id = int(check)
        kinds = [k for k, c in CALL_NAME_TO_CHECK.items() if c == check_id]
        if not kinds:
            ap.error(
                f"--model-for: check {check_id} makes no informed-stage calls of "
                "its own, so there is nothing to route"
            )
        for kind in kinds:
            model_by_kind[kind] = target.strip()

    if args.verify_provenance:
        overrides["verify_provenance"] = True
    if args.snapshot_dir:
        overrides["snapshot_dir"] = args.snapshot_dir
    if args.fetch_timeout:
        overrides["fetch_timeout_s"] = args.fetch_timeout
    if args.fetch_conversations:
        overrides["fetch_attachment_text"] = True

    forced_profile = None
    if args.project:
        forced_profile = profile_by_key(args.project)
        if forced_profile is None:
            ap.error(
                f"unknown --project {args.project!r}; "
                f"known: {', '.join(p.key for p in PROFILES)}"
            )
        # This run's disabled set belongs to the forced profile, not to
        # whatever `profile_for_project(task.project_id)` would pick -- the
        # SxS-queue revision shares L1's project id.
        from honeybee_qc import projects as projects_mod

        def _disabled(_pid: str, _profile=forced_profile):
            return _profile.disabled_checks

        projects_mod.disabled_checks_for = _disabled
        global disabled_checks_for
        disabled_checks_for = _disabled
        print(
            f"project: forcing profile {forced_profile.key} "
            f"({forced_profile.name})",
            file=sys.stderr,
        )

    if args.from_snowflake:
        tasks, ingest_results = load_taskattempts_csv(
            Path(args.input), profile=forced_profile
        )
        for result in ingest_results:
            if result.error:
                print(f"ingest: skipped {result.task_id}: {result.error}", file=sys.stderr)
        uncategorised = sum(r.uncategorised for r in ingest_results)
        unplaceable = sum(r.unplaceable for r in ingest_results)
        print(
            f"ingest: {len(tasks)} tasks, {uncategorised} criteria with no category "
            f"recorded, {unplaceable} carrying a label this build cannot place",
            file=sys.stderr,
        )
    else:
        tasks = load_tasks(Path(args.input))

    if args.relabel_c_to_b:
        from .remap_gpt_as_b import (
            AC_SELECTOR_STEP,
            apply_relabel_c_to_b,
            load_task_ids,
            narrow_to_ab_only,
        )

        relabel_ids = load_task_ids(args.relabel_c_to_b)
        hit = apply_relabel_c_to_b(tasks, relabel_ids)
        missing = sorted(relabel_ids - set(hit))
        print(
            f"relabel-c-to-b: promoted GPT-on-C to B on {len(hit)} task(s); "
            f"AC ranking rewritten as A-vs-B (selector {AC_SELECTOR_STEP}); "
            "Claude-as-B and simplified dropped from hydration",
            file=sys.stderr,
        )
        if missing:
            print(
                f"relabel-c-to-b: {len(missing)} id(s) not in this batch: "
                + ", ".join(missing),
                file=sys.stderr,
            )
        # The 26 that already had GPT on B still ingested C (Claude) and the
        # AC pair. This run is A vs GPT only -- drop those so hydrate skips
        # Claude and 400/450/470 do not also judge A vs Claude. Gated on this
        # flag so `--audit-slots D` still sees DA/DB/DC.
        for task in tasks:
            narrow_to_ab_only(task)
        print(
            "relabel-c-to-b: every task narrowed to A-vs-B; C/S not hydrated",
            file=sys.stderr,
        )

    if args.audit_slots:
        wanted = tuple(
            s.strip().upper() for s in args.audit_slots.split(",") if s.strip()
        )
        for task in tasks:
            task.audit_slots = wanted
        print(
            f"audit-slots: per-model checks limited to {','.join(wanted)}; "
            "SxS comparisons still use every pair",
            file=sys.stderr,
        )

    # A project can state its own values for things the base policy fixes -- the
    # weight scale being the one that matters, since Honeybee Aspirational's is
    # 1-3 where the base is 1-5. Applied under the command line, so an explicit
    # flag still wins. A mixed batch is fine when every project in it states the
    # same overrides (the Aspirational clone shares V5's); quietly mixing two
    # different scales is the failure this is here to prevent.
    project_ids = {t.project_id for t in tasks}
    profiles = [profile_for_project(pid) for pid in sorted(project_ids)]
    known = [p for p in profiles if p is not None]
    # Apply only when every project in the batch is known and they all state
    # the same non-empty overrides. The Aspirational clone is that case; an
    # L1 row mixed in is not.
    same_overrides = (
        len(known) == len(project_ids)
        and bool(known)
        and bool(known[0].policy_overrides)
        and all(p.policy_overrides == known[0].policy_overrides for p in known)
    )
    if same_overrides:
        policy = known[0].policy(policy)
        label = known[0].key if len(known) == 1 else "shared"
        print(
            f"policy: applied {label} overrides "
            f"({', '.join(sorted(known[0].policy_overrides))})",
            file=sys.stderr,
        )
    elif len(project_ids) > 1 and any(p.policy_overrides for p in known):
        print(
            f"policy: batch spans {len(project_ids)} projects and "
            f"{', '.join(p.key for p in known if p.policy_overrides)} state "
            "their own policy values; none applied. Run one project at a time.",
            file=sys.stderr,
        )

    if overrides:
        policy = replace(policy, **overrides)

    hydration = None
    if args.fetch_conversations and not args.dry_run:
        from .sources import (
            MIN_FREE_DISK_BYTES,
            free_disk_bytes,
            sweep_stale_browser_profiles,
        )

        swept = sweep_stale_browser_profiles()
        if swept:
            print(
                f"cleaned {swept} browser profile(s) left by an interrupted run",
                file=sys.stderr,
            )
        free = free_disk_bytes()
        if 0 <= free < MIN_FREE_DISK_BYTES:
            # Every render would fail, and each one would look like the host
            # timing out rather than a full disk, so say it once and stop.
            print(
                f"refusing to start: {free / 1024**3:.1f} GiB free, below the "
                f"{MIN_FREE_DISK_BYTES / 1024**3:.0f} GiB headless Chrome needs to "
                f"render a page. Every fetch would return an empty DOM.",
                file=sys.stderr,
            )
            return 2

        n_hydrate = args.hydrate_workers or args.workers
        print(
            f"hydrating conversations ({len(tasks)} tasks, {n_hydrate} browser workers)...",
            file=sys.stderr,
            flush=True,
        )
        hydration = hydrate_conversations(
            tasks, policy, workers=n_hydrate
        )
        print(
            f"conversations: hydrated {hydration.hydrated}/{hydration.submissions} "
            f"submissions, {hydration.turns} turns",
            file=sys.stderr,
        )
        for line in hydration.dead_pages:
            print(f"  dead share page: {line}", file=sys.stderr)
        for line in hydration.failures:
            print(f"  not fetched: {line}", file=sys.stderr)

    if args.dry_run:
        rubric_calls = sum(len(t.rubric) + 1 for t in tasks) if args.rubric_stage else 0
        rating_calls = (
            sum(
                estimate_rating_calls(t) + estimate_adjudication_calls(t, policy)
                for t in tasks
            )
            if args.rating_stage
            else 0
        )
        informed_calls = (
            sum(estimate_informed_calls(t, policy) for t in tasks)
            if args.informed_stage
            else 0
        )
        print(
            f"tasks={len(tasks)} rubric_calls={rubric_calls} "
            f"rating_calls={rating_calls} informed_calls={informed_calls} "
            f"total={rubric_calls + rating_calls + informed_calls}",
            file=sys.stderr,
        )
        if args.informed_stage:
            report_sampling_cost(tasks, policy, args.price_from_cache)
        return 0

    client: ModelClient | None = None
    cache = None
    need_client = (
        args.rubric_stage
        or args.rating_stage
        or args.informed_stage
        or bool(args.fill_missing_dims)
    )
    if need_client:
        client, cache = build_client(
            model=args.model,
            effort=args.effort,
            timeout_s=args.model_timeout,
            cache_path=args.cache_db,
            policy_version=stage_policy_version(policy),
            refresh=args.refresh_cache,
            max_in_flight=args.workers,
            model_by_kind=model_by_kind,
        )
        for check_id in sorted(
            {CALL_NAME_TO_CHECK[k] for k in model_by_kind if k in CALL_NAME_TO_CHECK}
        ):
            target = model_by_kind[
                next(k for k, c in CALL_NAME_TO_CHECK.items() if c == check_id)
            ]
            print(f"routing check {check_id} to {target}", file=sys.stderr)

    if args.fill_missing_dims and client is not None and not args.dry_run:
        from .fill_dims import fill_missing_dims, write_fill_csv

        fill_slot = args.fill_missing_dims.strip().upper()
        fill_rows = fill_missing_dims(
            tasks,
            client,
            slot=fill_slot,  # type: ignore[arg-type]
            policy=policy,
            workers=args.workers,
        )
        fill_path = Path(
            args.fill_out
            or (
                str(Path(args.out).with_name(f"{fill_slot.lower()}_dims_fill.csv"))
                if args.out
                else f"{fill_slot.lower()}_dims_fill.csv"
            )
        )
        write_fill_csv(fill_path, fill_rows, slot=fill_slot)  # type: ignore[arg-type]
        filled = sum(
            1
            for task in tasks
            if any(d.model == fill_slot for d in task.dimension_ratings)
        )
        print(
            f"fill-dims: wrote {len(fill_rows)} task(s) missing {fill_slot} "
            f"1-5s to {fill_path}; attached as contributor ratings on {filled} tasks",
            file=sys.stderr,
        )

    only_checks = tuple(
        int(c) for c in args.only_checks.split(",") if c.strip()
    )
    restricted_checks = (
        only_checks
        if only_checks
        else (RESTRICTED_CHECKS if args.restricted_checks else None)
    )

    payload = audit_batch(
        tasks,
        policy,
        client=client,
        workers=args.workers,
        rubric_stage=args.rubric_stage,
        rating_stage=args.rating_stage,
        informed_stage=args.informed_stage,
        task_workers=args.task_workers,
        restricted_checks=restricted_checks,
        skip_rating_checks=tuple(
            int(c) for c in args.skip_rating_checks.split(",") if c.strip()
        )
        or None,
        on_task_done=None if args.no_progress else ProgressPrinter(),
    )
    if hydration is not None:
        payload["hydration"] = hydration.to_dict()
    if cache is not None:
        payload["cache"] = cache.stats()
        cache.close()

    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)

    counts = payload["batch"]["task_counts"]
    print(
        f"tasks={counts.get('total', 0)} auditable={counts.get('auditable', 0)} "
        f"unauditable={counts.get('unauditable', 0)} fail={counts.get('fail', 0)} "
        f"pass_with_issues={counts.get('pass_with_issues', 0)} clean={counts.get('clean', 0)}",
        file=sys.stderr,
    )
    if policy.verify_provenance:
        print(
            f"provenance: {len(payload['review_queue'])} submissions need human review, "
            f"{len(payload['duplicates'])} duplicate findings",
            file=sys.stderr,
        )
    implemented = list(DETERMINISTIC_NOW)
    if args.restricted_checks:
        implemented = [c for c in implemented if c in RESTRICTED_CHECKS]
        print(
            f"restricted-checks: reporting only {list(RESTRICTED_CHECKS)}",
            file=sys.stderr,
        )
        by_model = payload.get("check_270_by_model") or {}
        for task_id, split in by_model.items():
            for model, v in split.items():
                print(
                    f"  270 [{task_id}::{model}]: {v['band']} "
                    f"({v['measurement']['numerator']}/{v['measurement']['denominator']})",
                    file=sys.stderr,
                )
    if args.rubric_stage:
        implemented += list(RUBRIC_STAGE_CHECKS)
        calls = sum(s["calls"] for s in payload["rubric_stage"])
        cached = sum(s["cached_calls"] for s in payload["rubric_stage"])
        print(f"rubric stage: {calls} calls ({cached} cached)", file=sys.stderr)
    if args.rating_stage:
        implemented += (
            [c for c in RATING_STAGE_CHECKS if c in RESTRICTED_CHECKS]
            if args.restricted_checks
            else list(RATING_STAGE_CHECKS)
        )
        stages = payload["rating_stage"]
        calls = sum(s["calls"] for s in stages)
        cached = sum(s["cached_calls"] for s in stages)
        abstained = sum(sum(s["abstentions"].values()) for s in stages)
        judged = sum(
            s["criterion_judgments"] + s["dimension_judgments"] for s in stages
        )
        print(f"rating stage: {calls} calls ({cached} cached)", file=sys.stderr)
        if judged + abstained:
            print(
                f"  {abstained} of {judged + abstained} judgments abstained for lack of "
                f"auditable evidence ({abstained / (judged + abstained):.0%})",
                file=sys.stderr,
            )
    if args.informed_stage:
        implemented += (
            [c for c in INFORMED_STAGE_CHECKS if c in RESTRICTED_CHECKS]
            if args.restricted_checks
            else list(INFORMED_STAGE_CHECKS)
        )
        stages = payload["informed_stage"]
        calls = sum(s["calls"] for s in stages)
        cached = sum(s["cached_calls"] for s in stages)
        print(f"informed stage: {calls} calls ({cached} cached)", file=sys.stderr)
        alleged = sum(len(s["autofails"]) for s in stages)
        confirmed = sum(
            1 for s in stages for a in s["autofails"] if a["confirmed"]
        )
        if alleged:
            print(
                f"  {confirmed} of {alleged} alleged rubric autofails survived the "
                f"second opinion",
                file=sys.stderr,
            )
        sampled = sum(s["agreement"]["sampled_claims"] for s in stages)
        if sampled:
            split = sum(s["agreement"]["split_claims"] for s in stages)
            dropped = sum(s["agreement"]["dropped_claims"] for s in stages)
            print(
                f"  {split} of {sampled} sampled claims split across draws "
                f"({split / sampled:.0%}); {dropped} were made by a minority and "
                f"discarded",
                file=sys.stderr,
            )
    if args.rubric_stage or args.rating_stage or args.informed_stage:
        print(f"model spend: ${payload['cost_usd']:.4f}", file=sys.stderr)
    if payload.get("stage_errors"):
        print(
            f"stage errors: {len(payload['stage_errors'])} (results kept for every "
            f"other task; see stage_errors in the output)",
            file=sys.stderr,
        )
        for err in payload["stage_errors"]:
            print(
                f"  {err['task_id']} {err['stage']}: {err['error']}", file=sys.stderr
            )
    print(
        f"checks implemented: {sorted(implemented)} of {len(REGISTRY)} dimensions",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
