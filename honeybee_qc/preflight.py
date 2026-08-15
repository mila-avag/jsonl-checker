"""Input adapter and deterministic preflight.

TARA's most persistent source of wasted runs was accepting a delivery record
missing fields the prompts silently depended on, producing confident nonsense.
Hard-fail loudly instead, before spending model calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_POLICY, Policy
from .models import (
    CriterionRating,
    DimensionRating,
    KeyTurn,
    ModelSubmission,
    RubricCriterion,
    Sxs,
    Task,
    Turn,
    TurnLink,
)
from .taxonomies import CONDITIONAL_DIMENSIONS, RATING_DIMENSIONS, UNAUDITED_DIMENSIONS

# Audit workflow step 3: "Both the models require at least 7 turns." Check 96
# scores this; the warning stays because it fires on the raw hydrated count and
# so reaches the operator before 96 has decided whether the shortfall is the
# contributor's or a fetch that stopped early.
MIN_EXPECTED_EXCHANGES = DEFAULT_POLICY.min_turns_per_model

# Named so `cli.py` can single this reason out: an empty rubric zeroes the
# denominator for the rubric-authoring checks (200/210/220/230/240/250/260)
# and the rating stage's per-criterion check (270/280), but it says nothing
# about whether the conversation happened, which model the contributor
# preferred, or whether their justification holds up -- those checks read
# the transcript and the contributor's own SxS pick, never the rubric, and
# stay eligible to run.
RUBRIC_EMPTY_REASON = (
    "rubric is empty; the denominator for checks 230/240/250 would be zero"
)


@dataclass
class PreflightResult:
    task_id: str
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _turns(raw: Iterable[dict[str, Any]] | None) -> list[Turn]:
    out: list[Turn] = []
    for t in raw or []:
        out.append(
            Turn(
                index=int(t.get("turn", t.get("index", 0)) or 0),
                role=t.get("role", "user"),
                text=t.get("text", "") or "",
                artifacts_mentioned=list(t.get("artifacts_mentioned", []) or []),
                tool_calls=list(t.get("tool_calls", []) or []),
            )
        )
    return out


def _submission(raw: dict[str, Any] | None, slot: str) -> ModelSubmission | None:
    if raw is None:
        return None
    return ModelSubmission(
        model=slot,  # type: ignore[arg-type]
        final_link=raw.get("final_link", "") or "",
        turn_links=[
            TurnLink(turn=int(l.get("turn", 0) or 0), url=l.get("url", "") or "")
            for l in raw.get("turn_links", []) or []
        ],
        transcript_pdf=raw.get("transcript_pdf", "") or "",
        attachments=list(raw.get("attachments", []) or []),
        conversation=_turns(raw.get("conversation")),
        declared_provider=raw.get("provider", "") or "",
    )


def parse_task(raw: dict[str, Any]) -> Task:
    return Task(
        task_id=str(raw.get("task_id", "") or ""),
        project_id=raw.get("project_id", "6a70ffe56999de9083413f7d"),
        seeded_prompt=raw.get("seeded_prompt", "") or "",
        user_goal=raw.get("user_goal", "") or "",
        target_deliverables=list(raw.get("target_deliverables", []) or []),
        prompts=_turns(raw.get("prompts")),
        target_outcome=list(raw.get("target_outcome", []) or []),
        model_a=_submission(raw.get("model_a"), "A"),
        model_b=_submission(raw.get("model_b"), "B"),
        key_turn=KeyTurn(
            turn_index=(raw.get("key_turn") or {}).get("turn_index"),
            justification=(raw.get("key_turn") or {}).get("justification", "") or "",
        ),
        rubric=[
            RubricCriterion(
                criterion_id=str(c.get("criterion_id", "") or ""),
                text=c.get("text", "") or "",
                l1_label=c.get("l1_label"),
                l2_label=c.get("l2_label"),
                weight=c.get("weight"),
                is_process_criterion=bool(c.get("is_process_criterion", False)),
            )
            for c in raw.get("rubric", []) or []
        ],
        criterion_ratings=[
            CriterionRating(
                criterion_id=str(r.get("criterion_id", "") or ""),
                model=r.get("model", "A"),
                score=int(r.get("score", 0) or 0),
                relevant_turns=list(r.get("relevant_turns", []) or []),
            )
            for r in raw.get("criterion_ratings", []) or []
        ],
        dimension_ratings=[
            DimensionRating(
                model=d.get("model", "A"),
                dimension=d.get("dimension", "") or "",
                rating=d.get("rating"),
                not_applicable=bool(d.get("not_applicable", False)),
                justification=d.get("justification", "") or "",
                relevant_turns=list(d.get("relevant_turns", []) or []),
            )
            for d in raw.get("dimension_ratings", []) or []
        ],
        sxs=Sxs(
            likert=(raw.get("sxs") or {}).get("likert"),
            justification=(raw.get("sxs") or {}).get("justification", "") or "",
        ),
    )


def load_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(parse_task(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
    return tasks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_turn_sequence(turns: list[Turn], label: str, res: PreflightResult) -> None:
    if not turns:
        return
    indices = [t.index for t in turns]
    if any(i < 1 for i in indices):
        res.hard_failures.append(f"{label}: turn indices must be 1-based")
    if indices != sorted(indices):
        res.hard_failures.append(f"{label}: turn indices are non-monotonic")
    if turns[0].role != "user":
        res.hard_failures.append(f"{label}: first turn must be a user turn")


def validate_task(task: Task, policy: Policy = DEFAULT_POLICY) -> PreflightResult:
    res = PreflightResult(task_id=task.task_id)

    if not task.task_id:
        res.hard_failures.append("task_id is missing")

    if not task.prompts:
        # Turn text lives on the share pages and is fetched separately, so a task
        # straight out of Snowflake legitimately has none yet. The rubric and
        # informed prompt-builders fall back to the hydrated conversation's user
        # turns when this is empty, so a task with a conversation still gets
        # judged against its real prompts; only a task with neither loses that
        # context. Rejecting the task here would void the other twenty checks.
        res.warnings.append(
            "prompts is empty; the conversation has not been fetched, so the "
            "rating stage will report not_evaluated"
        )
    _validate_turn_sequence(task.prompts, "prompts", res)

    if not task.rubric:
        res.hard_failures.append(RUBRIC_EMPTY_REASON)

    seen_ids: set[str] = set()
    for c in task.rubric:
        if not c.criterion_id:
            res.hard_failures.append("a rubric criterion has no criterion_id")
            continue
        if c.criterion_id in seen_ids:
            res.hard_failures.append(f"duplicate criterion_id {c.criterion_id!r}")
        seen_ids.add(c.criterion_id)
        if not c.text.strip():
            res.hard_failures.append(f"criterion {c.criterion_id} has empty text")

        lo, hi = policy.weight_scale
        if c.weight is not None and not isinstance(c.weight, int):
            res.hard_failures.append(
                f"criterion {c.criterion_id} weight {c.weight!r} is not an integer"
            )
        elif c.weight is not None and (c.weight < lo or c.weight > hi):
            # A weight outside the scale is a defect in the work, not malformed
            # input: live rubrics carry weights from -3 to -5 against a spec that
            # says 1-5 with no negatives. Hard-failing here would mark the task
            # unauditable and throw away the twenty other checks over one weight,
            # so it is a warning and check 260 counts it as maximally wrong.
            res.warnings.append(
                f"criterion {c.criterion_id} weight {c.weight!r} is outside {lo}-{hi}; "
                "check 260 will count it as a two-level disagreement"
            )
        if c.l1_label and c.l1_label not in _l1_set():
            res.hard_failures.append(
                f"criterion {c.criterion_id} has unknown L1 label {c.l1_label!r}"
            )
        if c.l2_label and not policy.l2_labels_configured:
            res.warnings.append(
                f"criterion {c.criterion_id} carries an L2 label but the L2 taxonomy is "
                "unconfigured; check 210 will report not_evaluated"
            )
        elif c.l2_label:
            parent = _l1_of_l2().get(c.l2_label)
            if parent is None:
                # The form's taxonomy has been revised at least once: 32 live
                # criteria carry leaves ("clarification timing", "tool action
                # execution") that no revision we hold defines. That is a leaf we
                # cannot place, not a task we cannot audit, so 210 skips the
                # criterion rather than scoring a disagreement against a list the
                # contributor was never offered.
                res.warnings.append(
                    f"criterion {c.criterion_id} has unknown L2 label {c.l2_label!r}; "
                    "check 210 will skip it"
                )
            elif c.l1_label and parent != c.l1_label:
                # A real contributor choice, not a typo: the form stores L1 and L2
                # as one string, so a mismatch means the pair was rewritten by hand
                # somewhere upstream. Check 210 cannot judge a leaf it cannot place.
                res.hard_failures.append(
                    f"criterion {c.criterion_id} pairs L2 {c.l2_label!r} with L1 "
                    f"{c.l1_label!r}, but that leaf belongs under {parent!r}"
                )

    known = task.criterion_ids()
    for r in task.criterion_ratings:
        if r.criterion_id not in known:
            res.hard_failures.append(
                f"criterion_ratings references unknown criterion_id {r.criterion_id!r}"
            )
        if r.score not in policy.criterion_rating_values:
            res.hard_failures.append(
                f"criterion rating for {r.criterion_id} model {r.model} has score "
                f"{r.score!r}, expected one of {list(policy.criterion_rating_values)}"
            )
        if r.score == 0 and not r.relevant_turns:
            res.warnings.append(
                f"criterion {r.criterion_id} model {r.model} scored 0 with no relevant "
                "turns cited; counted under check 280's missing_turn category"
            )

    for d in task.dimension_ratings:
        if d.dimension in UNAUDITED_DIMENSIONS:
            # Collected by the form, absent from check 300's list. Carried, not
            # audited, and not a reason to reject the task.
            continue
        if d.dimension not in RATING_DIMENSIONS:
            res.hard_failures.append(
                f"dimension_ratings references unknown dimension {d.dimension!r}"
            )
        if d.not_applicable and d.dimension not in CONDITIONAL_DIMENSIONS:
            res.warnings.append(
                f"dimension {d.dimension!r} marked not_applicable, but only "
                f"{sorted(CONDITIONAL_DIMENSIONS)} is conditional"
            )
        if not d.not_applicable:
            lo, hi = policy.dimension_rating_scale
            if d.rating is None:
                res.hard_failures.append(
                    f"dimension {d.dimension!r} model {d.model} has no rating"
                )
            elif d.rating < lo or d.rating > hi:
                res.hard_failures.append(
                    f"dimension {d.dimension!r} model {d.model} rating {d.rating} is "
                    f"outside the {lo}-{hi} scale"
                )

    lo, hi = policy.likert_scale
    if task.sxs.likert is None:
        res.hard_failures.append("sxs.likert is missing")
    elif task.sxs.likert < lo or task.sxs.likert > hi:
        res.hard_failures.append(
            f"sxs.likert {task.sxs.likert} is outside the {lo}-{hi} scale"
        )

    for sub in task.submissions():
        label = f"model {sub.model} conversation"
        _validate_turn_sequence(sub.conversation, label, res)
        minimum = policy.min_turns_per_model
        if sub.conversation and sub.exchange_count() < minimum:
            res.warnings.append(
                f"model {sub.model}: {sub.exchange_count()} exchanges, fewer than the "
                f"{minimum} turns both models are required to run; check 96 decides "
                "whether that is the contributor's shortfall or an incomplete fetch"
            )
        if not sub.transcript_pdf:
            res.warnings.append(f"model {sub.model}: no transcript PDF supplied")

    if task.key_turn.turn_index is not None:
        for sub in task.submissions():
            if not sub.conversation:
                continue
            indices = sub.turn_indices()
            # A key turn past the last turn we retrieved says nothing about the
            # contributor: share pages routinely render only part of a long
            # conversation, and the fetch stops there. Failing the task hard on
            # that discards good work over our own shortfall, and the spec
            # reserves unauditable for genuinely unusable submissions. A gap in
            # the middle is different -- nothing legitimate produces one -- so
            # that stays a hard failure.
            if task.key_turn.turn_index > max(indices, default=0):
                res.warnings.append(
                    f"key_turn {task.key_turn.turn_index} is beyond the "
                    f"{max(indices, default=0)} turns fetched for model "
                    f"{sub.model}; the conversation is incomplete, so checks "
                    f"reading that turn cannot be evaluated"
                )
            elif task.key_turn.turn_index not in indices:
                res.hard_failures.append(
                    f"key_turn {task.key_turn.turn_index} does not exist in model "
                    f"{sub.model}'s conversation"
                )
            elif not sub.has_assistant_at(task.key_turn.turn_index):
                # Same reasoning one turn in: a fetch that stopped after the user
                # message leaves the final exchange with no reply, which looks
                # identical to selecting a user turn. Only the second is the
                # contributor's mistake.
                if task.key_turn.turn_index == max(indices, default=0):
                    res.warnings.append(
                        f"key_turn {task.key_turn.turn_index} is the last turn "
                        f"fetched for model {sub.model} and has no reply; the "
                        f"conversation is cut off mid-exchange"
                    )
                else:
                    res.hard_failures.append(
                        f"key_turn {task.key_turn.turn_index} does not resolve to an "
                        f"assistant turn in model {sub.model}'s conversation"
                    )

    if not task.target_deliverables:
        res.warnings.append(
            "no target_deliverables supplied; the coverage pass has no customer "
            "checklist and must infer requirements from the prompts"
        )

    return res


def _l1_set() -> frozenset[str]:
    from .taxonomies import L1_LABELS

    return frozenset(L1_LABELS)


def _l1_of_l2() -> dict[str, str]:
    from .taxonomies import L1_OF_L2

    return L1_OF_L2


def validate_batch(tasks: list[Task], policy: Policy = DEFAULT_POLICY) -> list[PreflightResult]:
    results = [validate_task(t, policy) for t in tasks]
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.task_id] = counts.get(t.task_id, 0) + 1
    for res in results:
        if counts.get(res.task_id, 0) > 1:
            res.hard_failures.append(f"task_id {res.task_id!r} is duplicated in the batch")
    return results
