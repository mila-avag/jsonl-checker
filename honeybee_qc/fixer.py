"""Rewrite the contributor fields a QC fail actually named, one call per field.

A Honeybee task that fails the audit does not have to be scrapped. Most fails
live in a score, a justification, a weight, a key-turn pick, or the ranking --
fields a second pass can edit. Trajectories cannot: a dead share link, a
duplicate Gemini URL, a missing artifact, a simplified run that is the wrong
length, or a prompt defect (clarity, constraints, request, environment,
reasoning requirement) is the conversation itself. Editing the prompt would
force every model to be re-run, so those tasks are unfixable.

Unfixable, by the contract this file exists to honour:

  * Unauditable -- the submission itself cannot be trusted, so nothing is edited
  * a fail that has no editable field (prompt-only, or a trajectory check with
    no other fail): Valid Links, Unique Links, Output Artifacts Preserved,
    Minimum Turns, or a simplified run whose turn count is outside bounds
  * after the proposed score/weight edits, a *97-only* task whose simplified
    weighted pass rate is still above the project's ceiling

Clarity and Specificity / Unique Ground Truth (60) and Constraints (72) are
not in that list. They are not rewritten (the opening prompt still would
force a re-run), and they are not a reason to call the task unfixable -- a
task that fails only those is skipped.

A trajectory or prompt fail does **not** block the rest of the task. Those
checks are skipped as fields (you cannot rewrite a dead share link or the
opening prompt), and every other fail -- scores, turns, weights, ranking,
rubric text -- still gets a rewrite. The leftover issue is recorded on the
result so the paste sheet can name it.

Usage::

    python3 -m honeybee_qc.fixer \\
        --report honeybee_qc/audit_runs/.../report.json \\
        --tasks-csv /tmp/aspirational_35.csv --from-snowflake \\
        --cache-db honeybee_qc/audit_runs/.../cache.db \\
        --out-dir honeybee_qc/audit_runs/.../fixer \\
        --workers 60 --task-workers 15 --hydrate-workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .config import DEFAULT_POLICY, Policy
from .context import (
    golden_answer_block,
    numbered_prompts,
    render_comparison,
    render_conversation,
)
from .hydrate import hydrate_conversations
from .ingest import DIMENSION_FIELDS, load_taskattempts_csv
from .llm import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    ModelClient,
    ModelRequest,
    build_client,
    run_requests,
)
from .models import COMPARISON_SLOTS, Task
from .preflight import load_tasks
from .projects import (
    PROFILES,
    ProjectProfile,
    profile_by_key,
    profile_for_project,
)
from .registry import check_name
from .taxonomies import L1_LABELS

# Trajectory defects cannot be rewritten. 96 is disabled on Aspirational but is
# still a conversation-length problem wherever it fires. 97's turn-count leg is
# handled separately so the score-ceiling leg can still be a field edit.
TRAJECTORY_CHECKS = frozenset({90, 91, 95, 96})
# Changing the opening prompt would invalidate every share link already filed.
# 60 (clarity / unique ground truth) and 72 (constraints) are still not
# rewritten, but they are ignored for the unfixable decision -- a leftover
# on those two does not park the task.
PROMPT_CHECKS = frozenset({60, 71, 72, 76, 85})
FIXER_IGNORED_CHECKS = frozenset({60, 72})
UNAUDITABLE_CHECK = 1000
UNFIXABLE_CHECKS = TRAJECTORY_CHECKS | PROMPT_CHECKS | {UNAUDITABLE_CHECK}
RUBRIC_QUALITY_CHECKS = frozenset({230, 240, 250})

Kind = Literal[
    "prompt_text",
    "target_outcome",
    "golden_answer",
    "known_information",
    "not_known_information",
    "criterion_text",
    "criterion_l1",
    "criterion_weight",
    "criterion_score",
    "criterion_turns",
    "dimension_rating",
    "dimension_justification",
    "dimension_turns",
    "key_turn_index",
    "key_turn_justification",
    "preference_likert",
    "preference_justification",
    "produced_final_outcome",
]

SYSTEM_PROMPT = (
    "You are correcting one field on a Honeybee model-comparison task. An "
    "independent QC audit already failed this task and named the defect. You "
    "do not re-litigate whether the audit is right, except for turn citations: "
    "keep any current citation that supports the claim, even if the audit "
    "flagged it, and never invent turn numbers to fill an empty list. Change "
    "only the field you were given. Do not invent share links, uploaded files, "
    "or conversation turns. Write the value a reviewer should paste back into "
    "the form. A score and its justification (or a Likert and its writeup) "
    "must agree: never leave a high writeup next to a low score, or a "
    "preference paragraph that names the opposite winner from the Likert. "
    "Criterion scores are binary: 0 = not met, 1 = met. Never write 0 to mean "
    "pass or 1 to mean fail. Dimension ratings are 1-5 (1 worst, 5 best)."
)

def ranking_scale_instruction(comparison: str | None) -> str:
    """Spell 1 as the left slot. Hardcoding A is how D-better writeups shipped as 7.

    Honeybee's paste form is not the auditor's classic bipolar 7. 3-5 are ties.
    Teaching 3 = 'slightly better left' is how we shipped Likert 3 next to an
    A-win writeup on a form that means tie.
    """
    left, right = "left", "right"
    if comparison and len(comparison) >= 2:
        left, right = comparison[0], comparison[1]
    return f"""
Likert scale for this comparison (do not invert it):
  1 = Model {left} is much better
  2 = Model {left} is slightly better
  3 = tie
  4 = tie
  5 = tie
  6 = Model {right} is slightly better
  7 = Model {right} is much better
1-2 prefer Model {left}. 3-5 are ties (3 is not a slight {left} win). 6-7 prefer
Model {right}. A paragraph that prefers Model {left} must be 1-2. A paragraph
that prefers Model {right} must be 6-7. A tie / mixed / neither writeup must be 3-5.
"""

SCORE_POLARITY_INSTRUCTION = """
Criterion score polarity (do not invert it):
  0 = the criterion is NOT met
  1 = the criterion IS met
Never write 0 because the model did the thing, and never write 1 because it failed.
Notes must use those words: "met" goes with 1, "not met" goes with 0.
"""

CRITERION_TEXT_INSTRUCTION = """
This field is rubric criterion text, not a rating. Rewrite it as a testable
requirement a rater could score 0 or 1. Do not write Yes, No, Not met, Satisfied,
or a verdict about any model into the criterion. Do not invent a new criterion
id. If the current text is already a testable requirement, return it unchanged.
"""

TURN_KEEP_INSTRUCTION = """
Turn-citation rules for this field:
- The audit named specific turn numbers as incorrect. Those are listed in the
  feedback. Keep every other cited turn.
- Drop or replace a flagged turn only when it clearly does not support the
  claim: wrong conversation, unrelated to the criterion or justification, or
  the model text at that exchange does not show what was claimed. A defensible
  citation — including off-by-one of a claim the model text bears out — must
  be kept.
- If the current list is already defensible, return it unchanged.
- Do not invent citations. Do not swap in a "better" turn when the original
  supports the claim. Do not replace a turn you cannot open.
"""

# Score/writeup pairs that must be edited together. Changing one and leaving
# the other is how we shipped 2s next to "perfect" writeups and Likert 6
# next to "Gemini was slightly preferred".
_SCORE_KINDS = frozenset({"dimension_rating", "preference_likert"})
_JUSTIF_KINDS = frozenset({"dimension_justification", "preference_justification"})
_PAIR_NOTE = (
    "This field is paired with a score or Likert that is also being "
    "corrected. The two must agree. Do not defend the current value if the "
    "audit says it is wrong. A 2-score with a 5-writeup (or a Likert that "
    "picks B while the paragraph prefers A) is itself a fail."
)

TEXT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

INT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

SCORE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "integer", "enum": [0, 1]},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

TURNS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

L1_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "enum": list(L1_LABELS)},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

BOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

LIST_TEXT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "value": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "notes": {"type": "string"},
    },
    "required": ["value", "notes"],
    "additionalProperties": False,
}

SCHEMAS: dict[str, dict] = {
    "prompt_text": TEXT_SCHEMA,
    "target_outcome": LIST_TEXT_SCHEMA,
    "golden_answer": TEXT_SCHEMA,
    "known_information": TEXT_SCHEMA,
    "not_known_information": TEXT_SCHEMA,
    "criterion_text": TEXT_SCHEMA,
    "criterion_l1": L1_SCHEMA,
    "criterion_weight": INT_SCHEMA,
    "criterion_score": SCORE_SCHEMA,
    "criterion_turns": TURNS_SCHEMA,
    "dimension_rating": INT_SCHEMA,
    "dimension_justification": TEXT_SCHEMA,
    "dimension_turns": TURNS_SCHEMA,
    "key_turn_index": INT_SCHEMA,
    "key_turn_justification": TEXT_SCHEMA,
    "preference_likert": INT_SCHEMA,
    "preference_justification": TEXT_SCHEMA,
    "produced_final_outcome": BOOL_SCHEMA,
}

DIMENSION_TO_PREFIX = {name: prefix for prefix, name in DIMENSION_FIELDS.items()}

# One CSV per backfill card. A task can appear in several files when it failed
# across rubric text, scores, and dimension ratings at once.
CSV_GROUPS: dict[str, frozenset[str]] = {
    "rubrics": frozenset({"criterion_text", "criterion_l1", "criterion_weight"}),
    "scoring": frozenset({"criterion_score", "criterion_turns"}),
    "grading": frozenset(
        {
            "dimension_rating",
            "dimension_justification",
            "dimension_turns",
        }
    ),
    "ranking": frozenset({"preference_likert", "preference_justification"}),
    "key_turns": frozenset({"key_turn_index", "key_turn_justification"}),
}

PATCH_COLUMNS = [
    "task_id",
    "status",
    "unfixable_reason",
    "fail_checks",
    "group",
    "kind",
    "checks",
    "slot",
    "criterion_id",
    "dimension",
    "comparison",
    "old_value",
    "new_value",
    "step_id",
    "export_field",
    "notes",
    "error",
    "simplified_rate_before",
    "simplified_rate_after",
]


@dataclass
class FieldJob:
    """One editable form field, one model call."""

    kind: Kind
    check_ids: tuple[int, ...]
    current_value: Any
    feedback: str
    slot: str | None = None
    criterion_id: str | None = None
    dimension: str | None = None
    comparison: str | None = None  # e.g. "AB"
    flagged_turns: tuple[int, ...] = ()

    @property
    def field_id(self) -> str:
        parts = [self.kind]
        for piece in (self.slot, self.criterion_id, self.dimension, self.comparison):
            if piece:
                parts.append(str(piece))
        return "::".join(parts)

    def merge(self, other: FieldJob) -> FieldJob:
        checks = tuple(sorted(set(self.check_ids + other.check_ids)))
        feedback = self.feedback
        if other.feedback and other.feedback not in feedback:
            feedback = f"{feedback}\n\n---\n\n{other.feedback}"
        return replace(
            self,
            check_ids=checks,
            feedback=feedback,
            flagged_turns=tuple(sorted(set(self.flagged_turns + other.flagged_turns))),
        )


@dataclass
class FixPlan:
    task_id: str
    status: Literal["fixable", "unfixable", "skipped"]
    unfixable_reason: str = ""
    fail_checks: list[int] = field(default_factory=list)
    jobs: list[FieldJob] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "unfixable_reason": self.unfixable_reason,
            "unfixable_issues": unfixable_issue_names(
                self.fail_checks, self.unfixable_reason
            ),
            "fail_checks": unique_check_names(self.fail_checks),
            "jobs": [job_to_dict(j) for j in self.jobs],
        }


@dataclass
class FieldPatch:
    field_id: str
    kind: Kind
    check_ids: list[int]
    old_value: Any
    new_value: Any
    notes: str = ""
    error: str = ""
    slot: str | None = None
    criterion_id: str | None = None
    dimension: str | None = None
    comparison: str | None = None
    step_id: str = ""
    export_field: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checks"] = [check_name(c) for c in self.check_ids]
        del payload["check_ids"]
        return payload


@dataclass
class TaskFixResult:
    task_id: str
    status: Literal["fixable", "unfixable", "skipped", "error"]
    unfixable_reason: str = ""
    fail_checks: list[int] = field(default_factory=list)
    patches: list[FieldPatch] = field(default_factory=list)
    simplified_rate_before: float | None = None
    simplified_rate_after: float | None = None
    calls: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "unfixable_reason": self.unfixable_reason,
            "unfixable_issues": unfixable_issue_names(
                self.fail_checks, self.unfixable_reason
            ),
            "fail_checks": unique_check_names(self.fail_checks),
            "simplified_rate_before": self.simplified_rate_before,
            "simplified_rate_after": self.simplified_rate_after,
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 4),
            "patches": [p.to_dict() for p in self.patches],
        }


def job_to_dict(job: FieldJob) -> dict:
    return {
        "field_id": job.field_id,
        "kind": job.kind,
        "checks": [check_name(c) for c in job.check_ids],
        "slot": job.slot,
        "criterion_id": job.criterion_id,
        "dimension": job.dimension,
        "comparison": job.comparison,
        "current_value": job.current_value,
        "feedback": job.feedback,
    }


def _checks_by_id(report_task: dict) -> dict[int, dict]:
    return {c["check_id"]: c for c in report_task.get("checks") or [] if "check_id" in c}


def _fail_ids(report_task: dict) -> list[int]:
    return [int(c) for c in (report_task.get("fail_checks") or [])]


def unique_check_names(check_ids: list[int]) -> list[str]:
    """Check names in first-seen order, no duplicate labels.

    Rubric Ratings / Relevant Turns and Dimension Ratings / Relevant Turns
    share the display name `Relevant Turns`; a paste sheet should say it once.
    """
    names: list[str] = []
    for check_id in check_ids:
        name = check_name(check_id)
        if name not in names:
            names.append(name)
    return names


def unfixable_issue_names(fail_checks: list[int], reason: str) -> list[str]:
    """The unique checks that actually block a field rewrite."""
    if not reason:
        return []
    blocking: list[str] = []
    for check_id in fail_checks:
        if check_id in TRAJECTORY_CHECKS or check_id == UNAUDITABLE_CHECK:
            name = check_name(check_id)
            if name not in blocking:
                blocking.append(name)
    lowered = reason.lower()
    if 97 in fail_checks and (
        "turn" in lowered or "outside" in lowered or "pass rate" in lowered
    ):
        name = check_name(97)
        if name not in blocking:
            blocking.append(name)
    if "no editable field" in lowered:
        for name in unique_check_names(fail_checks):
            if name not in blocking:
                blocking.append(name)
    return blocking


def _feedback(check: dict) -> str:
    mid = check.get("measurement") or {}
    parts = [
        check_name(int(check["check_id"]))
        if check.get("check_id") is not None
        else "unknown check",
        f"band={check.get('band')} error={check.get('error_code') or ''}",
    ]
    if mid.get("notes"):
        parts.append(str(mid["notes"]))
    items = check.get("contributing_items") or []
    if items:
        parts.append("contributing items: " + ", ".join(str(i) for i in items[:40]))
    return "\n".join(parts)


def _criterion(task: Task, criterion_id: str):
    return next((c for c in task.rubric if c.criterion_id == criterion_id), None)


def _rating(task: Task, criterion_id: str, slot: str):
    return next(
        (
            r
            for r in task.criterion_ratings
            if r.criterion_id == criterion_id and r.model == slot
        ),
        None,
    )


def _dim(task: Task, slot: str, dimension: str):
    return next(
        (
            d
            for d in task.dimension_ratings
            if d.model == slot and d.dimension == dimension
        ),
        None,
    )


def _comparison(task: Task, label: str | None):
    pairs = task.comparison_list()
    if not label:
        return pairs[0] if pairs else task.sxs
    for sxs in pairs:
        if sxs.pair_label == label or f"{sxs.left}-{sxs.right}" == label:
            return sxs
    return pairs[0] if pairs else task.sxs


def simplified_weighted_rate(
    task: Task,
    scores: dict[str, int] | None = None,
    weights: dict[str, int] | None = None,
) -> float | None:
    """Weighted S pass rate after optional score/weight overrides. None if no S."""
    if task.simplified is None:
        return None
    s_ratings = [r for r in task.criterion_ratings if r.model == "S"]
    if not s_ratings:
        return None
    weight_map = {c.criterion_id: (c.weight or 0) for c in task.rubric}
    if weights:
        weight_map.update(weights)
    total = 0
    passed = 0
    for rating in s_ratings:
        weight = weight_map.get(rating.criterion_id, 0)
        total += weight
        score = scores[rating.criterion_id] if scores and rating.criterion_id in scores else rating.score
        if score == 1:
            passed += weight
    return (passed / total) if total else 0.0


def simplified_turn_count(task: Task) -> int:
    if task.simplified is None:
        return 0
    return task.simplified.exchange_count() or len(task.simplified.turn_manifest)


def unfixable_reason(task: Task, report_task: dict, profile: ProjectProfile | None) -> str:
    """Why this task cannot be repaired by editing fields. Empty if it can."""
    fails = set(_fail_ids(report_task))
    if report_task.get("verdict") == "unauditable" or UNAUDITABLE_CHECK in fails:
        return "task is unauditable; the submission itself cannot be trusted"
    hit = sorted(fails & TRAJECTORY_CHECKS)
    if hit:
        labelled = ", ".join(check_name(n) for n in hit)
        return f"trajectory issue: {labelled}"

    check_97 = _checks_by_id(report_task).get(97)
    if check_97 and check_97.get("band") == "fail":
        bounds = (
            profile.simplified_turn_bounds
            if profile and profile.simplified_turn_bounds is not None
            else (1, 3)
        )
        turns = simplified_turn_count(task)
        lo, hi = bounds
        if turns and (turns < lo or turns > hi):
            return (
                f"simplified trajectory has {turns} turns, outside the {lo}-{hi} bound"
            )
        notes = ((check_97.get("measurement") or {}).get("notes") or "")
        counts = (check_97.get("measurement") or {}).get("counts") or {}
        if "outside" in notes or counts.get("out_of_bounds") or counts.get("turns_out_of_bounds"):
            return f"simplified trajectory turn count is outside bounds ({notes or counts})"
    return ""


def _merge_jobs(jobs: list[FieldJob]) -> list[FieldJob]:
    """One call per field: two checks naming the same field share a prompt."""
    by_id: dict[str, FieldJob] = {}
    order: list[str] = []
    for job in jobs:
        key = job.field_id
        if key in by_id:
            by_id[key] = by_id[key].merge(job)
        else:
            by_id[key] = job
            order.append(key)
    return [by_id[k] for k in order]


def _pair_id(job: FieldJob) -> tuple | None:
    """Identity shared by a score and its writeup. None if the field is unpaired."""
    if job.kind in {"dimension_rating", "dimension_justification"}:
        if job.slot and job.dimension:
            return ("dim", job.slot, job.dimension)
        return None
    if job.kind in {"preference_likert", "preference_justification"}:
        if job.comparison:
            return ("rank", job.comparison)
        return None
    return None


def _companion_job(task: Task, job: FieldJob) -> FieldJob | None:
    """The other half of a score/writeup pair, or None if that half does not exist."""
    feedback = job.feedback + "\n\n" + _PAIR_NOTE
    if job.kind == "dimension_rating":
        if not job.slot or not job.dimension:
            return None
        dim = _dim(task, job.slot, job.dimension)
        return FieldJob(
            kind="dimension_justification",
            check_ids=job.check_ids,
            current_value=dim.justification if dim else "",
            feedback=feedback,
            slot=job.slot,
            dimension=job.dimension,
        )
    if job.kind == "dimension_justification":
        if not job.slot or not job.dimension:
            return None
        dim = _dim(task, job.slot, job.dimension)
        return FieldJob(
            kind="dimension_rating",
            check_ids=job.check_ids,
            current_value=dim.rating if dim else None,
            feedback=feedback,
            slot=job.slot,
            dimension=job.dimension,
        )
    if job.kind == "preference_likert":
        sxs = _comparison(task, job.comparison)
        return FieldJob(
            kind="preference_justification",
            check_ids=job.check_ids,
            current_value=sxs.justification if sxs else "",
            feedback=feedback,
            comparison=job.comparison or (sxs.pair_label if sxs else None),
        )
    if job.kind == "preference_justification":
        sxs = _comparison(task, job.comparison)
        return FieldJob(
            kind="preference_likert",
            check_ids=job.check_ids,
            current_value=sxs.likert if sxs else None,
            feedback=feedback,
            comparison=job.comparison or (sxs.pair_label if sxs else None),
        )
    return None


def _ensure_paired_jobs(task: Task, jobs: list[FieldJob]) -> list[FieldJob]:
    """If we rewrite a score, rewrite its writeup, and the other way around."""
    kinds_by_pair: dict[tuple, set[str]] = {}
    for job in jobs:
        pid = _pair_id(job)
        if pid is None:
            continue
        kinds_by_pair.setdefault(pid, set()).add(job.kind)
    extra: list[FieldJob] = []
    for job in jobs:
        pid = _pair_id(job)
        if pid is None:
            continue
        present = kinds_by_pair.get(pid) or set()
        needs_justif = job.kind in _SCORE_KINDS and not (present & _JUSTIF_KINDS)
        needs_score = job.kind in _JUSTIF_KINDS and not (present & _SCORE_KINDS)
        if not (needs_justif or needs_score):
            continue
        companion = _companion_job(task, job)
        if companion is None:
            continue
        cid = _pair_id(companion)
        if cid is not None:
            kinds_by_pair.setdefault(cid, set()).add(companion.kind)
        extra.append(companion)
    if not extra:
        return jobs
    return _merge_jobs([*jobs, *extra])


def _wave_jobs(jobs: list[FieldJob]) -> tuple[list[FieldJob], list[FieldJob]]:
    """Scores first, writeups second, so the writeup prompt can see the new score."""
    pair_kinds: dict[tuple, set[str]] = {}
    for job in jobs:
        pid = _pair_id(job)
        if pid is not None:
            pair_kinds.setdefault(pid, set()).add(job.kind)
    wave1: list[FieldJob] = []
    wave2: list[FieldJob] = []
    for job in jobs:
        pid = _pair_id(job)
        if (
            pid is not None
            and job.kind in _JUSTIF_KINDS
            and (pair_kinds.get(pid) or set()) & _SCORE_KINDS
        ):
            wave2.append(job)
        else:
            wave1.append(job)
    return wave1, wave2


def _flagged_incorrect_citations(
    counts: dict[str, Any],
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Items the auditor named as wrong citations, with those turn numbers.

    Missing citations and unverifiable fetches are not incorrect selections.
    Rewriting those invents turns the contributor never picked.
    """
    reasons = counts.get("reasons_by_item") or {}
    flagged = counts.get("turns_by_item") or {}
    out: list[tuple[str, str, tuple[int, ...]]] = []
    for key, why in reasons.items():
        if not isinstance(why, list) or "incorrect_turns" not in why:
            continue
        raw = flagged.get(key) or []
        try:
            turns = tuple(int(t) for t in raw)
        except (TypeError, ValueError):
            continue
        if not turns:
            continue
        slot, rest = _parse_model_item(str(key))
        if slot is None or not rest:
            continue
        out.append((slot, rest, turns))
    return out


def _same_turns(old: Any, new: Any) -> bool:
    def _norm(value: Any) -> list[int] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, list):
            return None
        try:
            return sorted(int(x) for x in value)
        except (TypeError, ValueError):
            return None

    a, b = _norm(old), _norm(new)
    return a is not None and a == b


_PRODUCT_SLOTS = set(COMPARISON_SLOTS)
_RATING_SLOTS = _PRODUCT_SLOTS | {"S"}


def _normalize_dimension(raw: str) -> str:
    text = str(raw or "").strip()
    if text.endswith("::missing_turn"):
        text = text[: -len("::missing_turn")]
    if text in DIMENSION_TO_PREFIX:
        return text
    if text in DIMENSION_FIELDS:
        return DIMENSION_FIELDS[text]
    snake = text.replace("&", "and").replace(" ", "_").replace("-", "_").lower()
    while "__" in snake:
        snake = snake.replace("__", "_")
    if snake in DIMENSION_FIELDS:
        return DIMENSION_FIELDS[snake]
    collapsed = snake.replace("_and_", "_")
    if collapsed in DIMENSION_FIELDS:
        return DIMENSION_FIELDS[collapsed]
    return text


def _parse_model_item(item: str) -> tuple[str | None, str]:
    """`A::C1` or `D::Outcome quality` -> (slot, rest). Bare ids stay (None, item)."""
    text = str(item)
    for prefix in ("280::", "281::", "310::"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            if text.endswith("::missing_turn"):
                text = text[: -len("::missing_turn")]
            break
    if "::" in text:
        left, right = text.split("::", 1)
        if left in _RATING_SLOTS:
            return left, right
    return None, text


def _jobs_from_check(
    task: Task, check: dict, extra: dict[str, Any], profile=None
) -> list[FieldJob]:
    check_id = int(check["check_id"])
    if check.get("band") != "fail":
        return []
    feedback = _feedback(check)
    items = [str(i) for i in (check.get("contributing_items") or [])]
    counts = (check.get("measurement") or {}).get("counts") or {}
    jobs: list[FieldJob] = []

    if check_id in PROMPT_CHECKS:
        return []

    if check_id == 80:
        jobs.append(
            FieldJob(
                kind="target_outcome",
                check_ids=(80,),
                current_value=list(task.target_deliverables or task.target_outcome),
                feedback=feedback,
            )
        )
        return jobs

    if check_id == 86:
        jobs.append(
            FieldJob(
                kind="known_information",
                check_ids=(86,),
                current_value=task.known_information,
                feedback=feedback,
            )
        )
        jobs.append(
            FieldJob(
                kind="not_known_information",
                check_ids=(86,),
                current_value=task.not_known_information,
                feedback=feedback,
            )
        )
        return jobs

    if check_id == 82:
        for sub in task.trajectories():
            if sub.produced_final_outcome is None:
                continue
            jobs.append(
                FieldJob(
                    kind="produced_final_outcome",
                    check_ids=(82,),
                    slot=sub.model,
                    current_value=sub.produced_final_outcome,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id == 97:
        for rating in task.criterion_ratings:
            if rating.model != "S" or rating.score != 1:
                continue
            crit = _criterion(task, rating.criterion_id)
            jobs.append(
                FieldJob(
                    kind="criterion_score",
                    check_ids=(97,),
                    slot="S",
                    criterion_id=rating.criterion_id,
                    current_value=rating.score,
                    feedback=(
                        feedback
                        + f"\n\nCriterion {rating.criterion_id}"
                        + (f": {crit.text}" if crit else "")
                        + " currently PASSES on the simplified trajectory. "
                        "Return 0 if this conversation does not honestly meet it."
                    ),
                )
            )
        return jobs

    if check_id == 100:
        slots: list[str] = []
        for item in items:
            if item.startswith("key_turn_"):
                slots.append(item.split("key_turn_", 1)[1])
            elif item in _PRODUCT_SLOTS:
                slots.append(item)
        if not slots:
            for slot, kt in task.key_turns().items():
                if slot == "S":
                    continue
                if kt.turn_index == 1 or kt.over_flagged:
                    slots.append(slot)
        for slot in dict.fromkeys(slots):
            sub = task.submission(slot)  # type: ignore[arg-type]
            current = sub.key_turn.turn_index if sub else task.key_turn.turn_index
            jobs.append(
                FieldJob(
                    kind="key_turn_index",
                    check_ids=(100,),
                    slot=slot,
                    current_value=current,
                    feedback=feedback + "\nThe key turn must not be turn 1.",
                )
            )
        return jobs

    if check_id == 110:
        targets = items or [s.model for s in task.submissions()]
        seen: set[str] = set()
        for item in targets:
            slot, _ = _parse_model_item(item)
            slot = slot or (item if item in _PRODUCT_SLOTS else None)
            if not slot or slot in seen:
                continue
            seen.add(slot)
            sub = task.submission(slot)  # type: ignore[arg-type]
            just = (sub.key_turn.justification if sub else "") or task.key_turn.justification
            jobs.append(
                FieldJob(
                    kind="key_turn_justification",
                    check_ids=(110,),
                    slot=slot,
                    current_value=just,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id == 200:
        findings = extra.get("rubric_findings") or []
        for item in items:
            crit = _criterion(task, item)
            auditor = ""
            for finding in findings:
                if finding.get("criterion_id") == item and finding.get("l1"):
                    auditor = finding["l1"].get("auditor") or ""
            jobs.append(
                FieldJob(
                    kind="criterion_l1",
                    check_ids=(200,),
                    criterion_id=item,
                    current_value=crit.l1_label if crit else "",
                    feedback=feedback
                    + (f"\nAuditor L1: {auditor}" if auditor else ""),
                )
            )
        return jobs

    if check_id in RUBRIC_QUALITY_CHECKS:
        findings = extra.get("rubric_findings") or []
        by_id = {f.get("criterion_id"): f for f in findings}
        for item in items:
            # Auditor-invented ids (missing::<something>) are not pasteable
            # rubric rows. Writing "Not met" into a fake key is how we shipped
            # verdicts as criterion text.
            if str(item).startswith("missing::"):
                continue
            crit = _criterion(task, item)
            issue_lines = []
            for issue in (by_id.get(item) or {}).get("issues") or []:
                issue_lines.append(
                    f"[{issue.get('category')}] {issue.get('description')} "
                    f"(evidence: {issue.get('evidence')})"
                )
            jobs.append(
                FieldJob(
                    kind="criterion_text",
                    check_ids=(check_id,),
                    criterion_id=item,
                    current_value=crit.text if crit else "",
                    feedback=feedback
                    + (("\n" + "\n".join(issue_lines)) if issue_lines else ""),
                )
            )
        return jobs

    if check_id == 260:
        findings = extra.get("rubric_findings") or []
        by_id = {f.get("criterion_id"): f for f in findings}
        for item in items:
            if str(item).startswith("missing::"):
                continue
            crit = _criterion(task, item)
            weight_info = (by_id.get(item) or {}).get("weight") or {}
            jobs.append(
                FieldJob(
                    kind="criterion_weight",
                    check_ids=(260,),
                    criterion_id=item,
                    current_value=crit.weight if crit else None,
                    feedback=feedback
                    + (
                        f"\nDefensible range: {weight_info.get('defensible_low')}"
                        f"-{weight_info.get('defensible_high')}; auditor pick "
                        f"{weight_info.get('auditor')}"
                        if weight_info
                        else ""
                    ),
                )
            )
        return jobs

    if check_id in (270, 271):
        disagreeing = counts.get("disagreeing_items") or [
            f"{slot}::{cid}"
            for cid in items
            for slot in _RATING_SLOTS
            if _rating(task, cid, slot)
        ]
        allowed = {"S"} if check_id == 271 else _PRODUCT_SLOTS
        for item in disagreeing:
            slot, criterion_id = _parse_model_item(str(item))
            if not slot or not criterion_id or slot not in allowed:
                continue
            rating = _rating(task, criterion_id, slot)
            jobs.append(
                FieldJob(
                    kind="criterion_score",
                    check_ids=(check_id,),
                    slot=slot,
                    criterion_id=criterion_id,
                    current_value=rating.score if rating else None,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id in (280, 281):
        for slot, criterion_id, flagged in _flagged_incorrect_citations(counts):
            spec = profile.trajectory(slot) if profile is not None else None
            if spec is not None and not spec.turns_step:
                continue
            if spec is None and slot not in {"A", "B", "C"}:
                continue
            rating = _rating(task, criterion_id, slot)
            jobs.append(
                FieldJob(
                    kind="criterion_turns",
                    check_ids=(check_id,),
                    slot=slot,
                    criterion_id=criterion_id,
                    current_value=list(rating.relevant_turns) if rating else [],
                    flagged_turns=flagged,
                    feedback=feedback
                    + f"\nFlagged citations (replace only if truly unsupported): {list(flagged)}",
                )
            )
        return jobs

    if check_id in (300, 305):
        keys = [
            k
            for k, reasons in (counts.get("reasons_by_item") or {}).items()
            if "na_mismatch" not in reasons
        ] or items
        for item in keys:
            slot, dimension = _parse_model_item(str(item))
            if not slot or not dimension:
                continue
            dim = _dim(task, slot, _normalize_dimension(dimension))
            jobs.append(
                FieldJob(
                    kind="dimension_rating",
                    check_ids=(check_id,),
                    slot=slot,
                    dimension=_normalize_dimension(dimension),
                    current_value=dim.rating if dim else None,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id == 310:
        for slot, dimension, flagged in _flagged_incorrect_citations(counts):
            canonical = _normalize_dimension(dimension)
            dim = _dim(task, slot, canonical)
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=slot,
                    dimension=canonical,
                    current_value=list(dim.relevant_turns) if dim else [],
                    flagged_turns=flagged,
                    feedback=feedback
                    + f"\nFlagged citations (replace only if truly unsupported): {list(flagged)}",
                )
            )
        return jobs

    if check_id == 400:
        for sxs in task.comparison_list():
            jobs.append(
                FieldJob(
                    kind="preference_likert",
                    check_ids=(400,),
                    comparison=sxs.pair_label,
                    current_value=sxs.likert,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id == 450:
        triggered = counts.get("conditions_by_item") or {i: [] for i in items}
        for item, conditions in triggered.items():
            extra_fb = feedback + (
                f"\nConditions on {item}: {', '.join(map(str, conditions))}"
                if conditions
                else ""
            )
            if str(item).startswith("ranking"):
                label = None
                if "::" in str(item):
                    label = str(item).split("::", 1)[1]
                sxs = _comparison(task, label)
                jobs.append(
                    FieldJob(
                        kind="preference_justification",
                        check_ids=(450,),
                        comparison=sxs.pair_label,
                        current_value=sxs.justification,
                        feedback=extra_fb,
                    )
                )
                continue
            slot, dimension = _parse_model_item(str(item))
            if not slot or not dimension:
                continue
            dim = _dim(task, slot, dimension)
            jobs.append(
                FieldJob(
                    kind="dimension_justification",
                    check_ids=(450,),
                    slot=slot,
                    dimension=dimension,
                    current_value=dim.justification if dim else "",
                    feedback=extra_fb,
                )
            )
        return jobs

    if check_id == 470:
        for sxs in task.comparison_list():
            jobs.append(
                FieldJob(
                    kind="preference_justification",
                    check_ids=(470,),
                    comparison=sxs.pair_label,
                    current_value=sxs.justification,
                    feedback=feedback + "\nThe ranking writeup must state a preference.",
                )
            )
        return jobs

    return jobs


def plan_task(
    task: Task,
    report_task: dict,
    profile: ProjectProfile | None = None,
    extra: dict[str, Any] | None = None,
) -> FixPlan:
    fails = _fail_ids(report_task)
    verdict = report_task.get("verdict") or ""
    if verdict in {"clean", "pass_with_issues"} and not fails:
        return FixPlan(task_id=task.task_id, status="skipped", fail_checks=fails)

    leftover = unfixable_reason(task, report_task, profile)
    if leftover.startswith("task is unauditable"):
        return FixPlan(
            task_id=task.task_id,
            status="unfixable",
            unfixable_reason=leftover,
            fail_checks=fails,
        )

    extra = extra or {}
    skip_checks = set(UNFIXABLE_CHECKS)
    # Turn count is a property of the conversation. Score flips cannot shorten S.
    if leftover and (
        "turn" in leftover.lower() or "outside" in leftover.lower()
    ):
        skip_checks.add(97)
    jobs: list[FieldJob] = []
    for check in report_task.get("checks") or []:
        if check.get("check_id") in skip_checks:
            continue
        jobs.extend(_jobs_from_check(task, check, extra, profile))
    jobs = _ensure_paired_jobs(task, _merge_jobs(jobs))
    if not jobs:
        actionable = [c for c in fails if c not in FIXER_IGNORED_CHECKS]
        if not actionable and not leftover:
            return FixPlan(
                task_id=task.task_id,
                status="skipped",
                fail_checks=fails,
            )
        return FixPlan(
            task_id=task.task_id,
            status="unfixable",
            unfixable_reason=leftover
            or "failed, but no editable field mapped from the fail checks",
            fail_checks=fails,
        )
    return FixPlan(
        task_id=task.task_id,
        status="fixable",
        unfixable_reason=leftover,
        fail_checks=fails,
        jobs=jobs,
    )


def export_locator(job: FieldJob, profile: ProjectProfile | None) -> tuple[str, str]:
    """`(step_id, export_field)` a backfill card can write. Empty if unknown."""
    if profile is None:
        return "", ""
    spec = profile.trajectory(job.slot) if job.slot else None
    suffix = spec.suffix if spec else ""
    prefix = DIMENSION_TO_PREFIX.get(job.dimension or "", "")

    if job.kind == "prompt_text":
        return profile.prompt_step, "final_hardened_prompt"
    if job.kind == "target_outcome":
        return profile.target_outcome_step, "criteria"
    if job.kind == "golden_answer":
        return profile.golden_answer_step or "", "golden_ground_truth_answer"
    if job.kind == "known_information":
        return profile.known_information_step or "", "known_information"
    if job.kind == "not_known_information":
        return profile.known_information_step or "", "not_known_information"
    if job.kind in {"criterion_text", "criterion_l1", "criterion_weight"}:
        return profile.rubric_step, f"criteria[{job.criterion_id}].{job.kind.split('_')[-1]}"
    if job.kind == "criterion_score" and spec:
        return spec.scores_step, f"responseRatings[{spec.rating_key}].{job.criterion_id}.score"
    if job.kind == "criterion_turns" and spec and spec.turns_step:
        return spec.turns_step, f"responseRatings[{spec.rating_key}].{job.criterion_id}"
    if job.kind == "dimension_rating" and spec and spec.dims_step and prefix:
        return spec.dims_step, f"{prefix}_{suffix}"
    if job.kind == "dimension_justification" and spec and spec.dims_step and prefix:
        return spec.dims_step, f"{prefix}_justif_{suffix}"
    if job.kind == "dimension_turns" and spec and spec.dims_step and prefix:
        return spec.dims_step, f"{prefix}_turns_{suffix}"
    if job.kind == "key_turn_index" and spec:
        return spec.run_step, spec.field_name("key_turn")
    if job.kind == "key_turn_justification" and spec:
        return spec.run_step, spec.field_name("key_turn_justification")
    if job.kind in {"preference_likert", "preference_justification"}:
        label = job.comparison or ""
        for comparison in profile.comparisons:
            if f"{comparison.left}{comparison.right}" == label:
                field = (
                    "preferenceLikert"
                    if job.kind == "preference_likert"
                    else "fieldResponses.preference_justification"
                )
                return comparison.selector_step, field
        if profile.comparisons:
            field = (
                "preferenceLikert"
                if job.kind == "preference_likert"
                else "fieldResponses.preference_justification"
            )
            return profile.comparisons[0].selector_step, field
    if job.kind == "produced_final_outcome" and spec:
        return spec.run_step, spec.produced_outcome_field
    return "", ""


def _conversation_block(task: Task, slot: str | None, policy: Policy) -> str:
    subs = []
    if slot:
        sub = task.submission(slot)  # type: ignore[arg-type]
        if sub:
            subs = [sub]
    else:
        subs = list(task.trajectories())
    parts = []
    for sub in subs:
        # 8000 chars made later turns look missing, then the model invented
        # cites and flipped scores. This pass is one field, so it can read
        # the same budget as informed adjudication.
        text, _ = render_conversation(
            sub, policy, max_chars=policy.adjudication_max_conversation_chars
        )
        parts.append(f"## Model {sub.model} conversation\n{text}")
    return "\n\n".join(parts)


def _ranking_conversation_block(
    task: Task, comparison: str | None, policy: Policy
) -> str:
    """Only the two models in this SxS pair. Rendering A/B/C/D together is how
    DB writeups became A-vs-B comparisons that never named D."""
    if comparison and len(comparison) >= 2:
        left, right = comparison[0], comparison[1]
        rendered, _ = render_comparison(
            task.submission(left),  # type: ignore[arg-type]
            task.submission(right),  # type: ignore[arg-type]
            policy,
        )
        return rendered
    return _conversation_block(task, None, policy)


def build_fix_request(
    task: Task,
    job: FieldJob,
    policy: Policy = DEFAULT_POLICY,
    paired_corrected_value: Any = None,
) -> ModelRequest:
    where = []
    if job.slot:
        where.append(f"model slot {job.slot}")
    if job.criterion_id:
        crit = _criterion(task, job.criterion_id)
        where.append(f"criterion {job.criterion_id}" + (f": {crit.text}" if crit else ""))
    if job.dimension:
        where.append(f"dimension {job.dimension}")
    if job.comparison:
        where.append(f"comparison {job.comparison}")

    paired_block = ""
    if paired_corrected_value is not None:
        paired_block = (
            "\n## Paired field (already corrected)\n\n"
            "The score or Likert that must agree with this field is being set "
            f"to:\n\n{json.dumps(paired_corrected_value, ensure_ascii=False)}\n\n"
            "Write this field so it defends that value. Do not contradict it.\n"
        )

    turn_block = ""
    if job.kind in {"criterion_turns", "dimension_turns"}:
        turn_block = "\n" + TURN_KEEP_INSTRUCTION
        if job.flagged_turns:
            turn_block += (
                f"\nFlagged as incorrect: {list(job.flagged_turns)}\n"
            )
    if job.kind in {"preference_likert", "preference_justification"}:
        turn_block += "\n" + ranking_scale_instruction(job.comparison)
        if job.comparison:
            turn_block += f"This field is the {job.comparison} pair.\n"
    if job.kind == "criterion_score":
        turn_block += "\n" + SCORE_POLARITY_INSTRUCTION
    if job.kind == "criterion_text":
        turn_block += "\n" + CRITERION_TEXT_INSTRUCTION

    prompt = f"""# Field to correct

kind: {job.kind}
{' | '.join(where)}

## Current value

{json.dumps(job.current_value, ensure_ascii=False, indent=2) if not isinstance(job.current_value, str) else job.current_value}

## QC audit feedback for this field

{job.feedback}
{paired_block}
{golden_answer_block(task)}
## Task prompt

{task.seeded_prompt or numbered_prompts(task)}

{
        _ranking_conversation_block(task, job.comparison, policy)
        if job.kind in {"preference_likert", "preference_justification"}
        else _conversation_block(task, job.slot, policy)
    }

## Your task

Return a corrected `value` for this one field. Keep the same type. If the current
value is already defensible against the conversation, return it unchanged and say
so in `notes`. Do not edit any other field. If a paired score/Likert is shown
above, this value must agree with it.
{turn_block}
"""
    return ModelRequest(
        key=f"fix::{task.task_id}::{job.field_id}",
        prompt=prompt,
        schema=SCHEMAS[job.kind],
        system=SYSTEM_PROMPT,
        metadata={
            "task_id": task.task_id,
            "kind": job.kind,
            "field_id": job.field_id,
            "check_ids": list(job.check_ids),
        },
    )


def _apply_ceiling(
    task: Task,
    plan: FixPlan,
    patches: list[FieldPatch],
    profile: ProjectProfile | None,
) -> str:
    """Empty if the proposed edits keep S under the ceiling (or there is no S)."""
    if task.simplified is None:
        return ""
    ceiling = (
        profile.simplified_max_pass_rate
        if profile and profile.simplified_max_pass_rate is not None
        else 0.50
    )
    scores: dict[str, int] = {}
    weights: dict[str, int] = {}
    for patch in patches:
        if patch.error or patch.new_value is None:
            continue
        if patch.kind == "criterion_score" and patch.slot == "S" and patch.criterion_id:
            scores[patch.criterion_id] = int(patch.new_value)
        if patch.kind == "criterion_weight" and patch.criterion_id:
            weights[patch.criterion_id] = int(patch.new_value)
    rate = simplified_weighted_rate(task, scores=scores or None, weights=weights or None)
    if rate is None:
        return ""
    if rate > ceiling:
        return (
            f"simplified weighted pass rate would be {rate:.1%} after fixes, "
            f"above the {ceiling:.0%} ceiling"
        )
    return ""


def apply_plan(
    task: Task,
    plan: FixPlan,
    responses: dict[str, Any],
    profile: ProjectProfile | None = None,
    cost_usd: float = 0.0,
) -> TaskFixResult:
    if plan.status != "fixable":
        return TaskFixResult(
            task_id=plan.task_id,
            status=plan.status,
            unfixable_reason=plan.unfixable_reason,
            fail_checks=plan.fail_checks,
            simplified_rate_before=simplified_weighted_rate(task),
        )

    patches: list[FieldPatch] = []
    for job in plan.jobs:
        step_id, export_field = export_locator(job, profile)
        response = responses.get(f"fix::{task.task_id}::{job.field_id}")
        if not isinstance(response, dict) or "value" not in response:
            patches.append(
                FieldPatch(
                    field_id=job.field_id,
                    kind=job.kind,
                    check_ids=list(job.check_ids),
                    old_value=job.current_value,
                    new_value=None,
                    error="no usable model response",
                    slot=job.slot,
                    criterion_id=job.criterion_id,
                    dimension=job.dimension,
                    comparison=job.comparison,
                    step_id=step_id,
                    export_field=export_field,
                )
            )
            continue
        new_value = response.get("value")
        if job.kind in {"criterion_turns", "dimension_turns"} and _same_turns(
            job.current_value, new_value
        ):
            continue
        patches.append(
            FieldPatch(
                field_id=job.field_id,
                kind=job.kind,
                check_ids=list(job.check_ids),
                old_value=job.current_value,
                new_value=new_value,
                notes=str(response.get("notes") or ""),
                slot=job.slot,
                criterion_id=job.criterion_id,
                dimension=job.dimension,
                comparison=job.comparison,
                step_id=step_id,
                export_field=export_field,
            )
        )

    before = simplified_weighted_rate(task)
    ceiling = _apply_ceiling(task, plan, patches, profile)
    after_scores = {
        p.criterion_id: int(p.new_value)
        for p in patches
        if p.kind == "criterion_score" and p.slot == "S" and p.criterion_id and p.new_value is not None and not p.error
    }
    after_weights = {
        p.criterion_id: int(p.new_value)
        for p in patches
        if p.kind == "criterion_weight" and p.criterion_id and p.new_value is not None and not p.error
    }
    after = simplified_weighted_rate(
        task, scores=after_scores or None, weights=after_weights or None
    )
    leftover = plan.unfixable_reason
    if ceiling:
        leftover = f"{leftover}; {ceiling}" if leftover else ceiling
        only_s = all(97 in p.check_ids for p in patches) if patches else True
        if only_s:
            return TaskFixResult(
                task_id=plan.task_id,
                status="unfixable",
                unfixable_reason=leftover,
                fail_checks=plan.fail_checks,
                simplified_rate_before=before,
                simplified_rate_after=after,
                calls=len(plan.jobs),
                cost_usd=cost_usd,
            )
    return TaskFixResult(
        task_id=plan.task_id,
        status="fixable",
        unfixable_reason=leftover,
        fail_checks=plan.fail_checks,
        patches=patches,
        simplified_rate_before=before,
        simplified_rate_after=after,
        calls=len(plan.jobs),
        cost_usd=cost_usd,
    )


def _rubric_findings_index(report: dict) -> dict[str, list]:
    return {
        block["task_id"]: block.get("findings") or []
        for block in report.get("rubric_stage") or []
        if block.get("task_id")
    }


def plan_report(
    tasks: list[Task],
    report: dict,
    profile_by_task: dict[str, ProjectProfile | None] | None = None,
) -> list[FixPlan]:
    by_id = {t.task_id: t for t in tasks}
    findings = _rubric_findings_index(report)
    plans = []
    for report_task in report.get("tasks") or []:
        task = by_id.get(report_task["task_id"])
        if task is None:
            plans.append(
                FixPlan(
                    task_id=report_task["task_id"],
                    status="unfixable",
                    unfixable_reason="task missing from the CSV",
                    fail_checks=_fail_ids(report_task),
                )
            )
            continue
        profile = (profile_by_task or {}).get(task.task_id)
        if profile is None:
            profile = profile_for_project(task.project_id)
        plans.append(
            plan_task(
                task,
                report_task,
                profile,
                extra={"rubric_findings": findings.get(task.task_id) or []},
            )
        )
    return plans


def run_fixer(
    tasks: list[Task],
    report: dict,
    client: ModelClient,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 8,
    task_workers: int = 1,
    dry_run: bool = False,
    profile_by_task: dict[str, ProjectProfile | None] | None = None,
) -> list[TaskFixResult]:
    if profile_by_task is None:
        profile_by_task = {t.task_id: profile_for_project(t.project_id) for t in tasks}
    plans = plan_report(tasks, report, profile_by_task)
    by_id = {t.task_id: t for t in tasks}

    # One batch per wave of `task_workers` fixable tasks, matching the audit's
    # --task-workers: that many tasks in flight, sharing a global --workers cap.
    request_batches: list[list[ModelRequest]] = []
    pending_wave2: list[tuple[Task, FieldJob, FixPlan]] = []
    if not dry_run:
        current: list[ModelRequest] = []
        n_in_wave = 0
        wave = max(1, task_workers)
        for plan in plans:
            task = by_id.get(plan.task_id)
            if task is None or plan.status != "fixable":
                continue
            w1, w2 = _wave_jobs(plan.jobs)
            current.extend(build_fix_request(task, job, policy) for job in w1)
            pending_wave2.extend((task, job, plan) for job in w2)
            n_in_wave += 1
            if n_in_wave >= wave:
                request_batches.append(current)
                current = []
                n_in_wave = 0
        if current:
            request_batches.append(current)

    responses_by_key: dict[str, Any] = {}
    cost = 0.0
    for batch in request_batches:
        for response in run_requests(client, batch, workers=workers):
            cost += response.cost_usd
            if response.ok:
                responses_by_key[response.key] = response.data

    # Writeups run after scores so the prompt can name the corrected value.
    if not dry_run and pending_wave2:
        wave2_reqs: list[ModelRequest] = []
        for task, job, plan in pending_wave2:
            paired_value = None
            pid = _pair_id(job)
            for other in plan.jobs:
                if _pair_id(other) == pid and other.kind in _SCORE_KINDS:
                    data = responses_by_key.get(f"fix::{task.task_id}::{other.field_id}")
                    if isinstance(data, dict) and "value" in data:
                        paired_value = data["value"]
                    break
            wave2_reqs.append(
                build_fix_request(
                    task, job, policy, paired_corrected_value=paired_value
                )
            )
        for response in run_requests(client, wave2_reqs, workers=workers):
            cost += response.cost_usd
            if response.ok:
                responses_by_key[response.key] = response.data

    results = []
    for plan in plans:
        task = by_id.get(plan.task_id)
        if task is None:
            results.append(
                TaskFixResult(
                    task_id=plan.task_id,
                    status="unfixable",
                    unfixable_reason=plan.unfixable_reason,
                    fail_checks=plan.fail_checks,
                )
            )
            continue
        if dry_run and plan.status == "fixable":
            dry_patches = []
            for job in plan.jobs:
                step_id, export_field = export_locator(
                    job, profile_by_task.get(task.task_id)
                )
                dry_patches.append(
                    FieldPatch(
                        field_id=job.field_id,
                        kind=job.kind,
                        check_ids=list(job.check_ids),
                        old_value=job.current_value,
                        new_value=None,
                        notes="dry-run; no model call",
                        slot=job.slot,
                        criterion_id=job.criterion_id,
                        dimension=job.dimension,
                        comparison=job.comparison,
                        step_id=step_id,
                        export_field=export_field,
                    )
                )
            results.append(
                TaskFixResult(
                    task_id=plan.task_id,
                    status="fixable",
                    fail_checks=plan.fail_checks,
                    calls=len(plan.jobs),
                    simplified_rate_before=simplified_weighted_rate(task),
                    patches=dry_patches,
                )
            )
            continue
        results.append(
            apply_plan(
                task,
                plan,
                responses_by_key,
                profile_by_task.get(task.task_id),
                cost_usd=0.0,
            )
        )
    if request_batches:
        # Cost is batch-wide; attach it to the first fixable result so the
        # summary can still report a total without attributing cents per field.
        for result in results:
            if result.status == "fixable":
                result.cost_usd = cost
                break
    return results


def _group_for_kind(kind: str) -> str:
    for group, kinds in CSV_GROUPS.items():
        if kind in kinds:
            return group
    return "other"


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_outputs(out_dir: Path, plans: list[FixPlan], results: list[TaskFixResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fixer_plan.json").write_text(
        json.dumps([p.to_dict() for p in plans], indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "fixer_results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    patches: list[dict] = []
    unfixable: list[dict] = []
    tasks_rows: list[dict] = []
    for result in results:
        fail_names = ", ".join(unique_check_names(result.fail_checks))
        issue_names = ", ".join(
            unfixable_issue_names(result.fail_checks, result.unfixable_reason)
        )
        tasks_rows.append(
            {
                "task_id": result.task_id,
                "status": result.status,
                "unfixable_reason": result.unfixable_reason,
                "unfixable_issues": issue_names,
                "fail_checks": fail_names,
                "n_patches": len(result.patches) if result.status == "fixable" else 0,
                "simplified_rate_before": (
                    ""
                    if result.simplified_rate_before is None
                    else f"{result.simplified_rate_before:.4f}"
                ),
                "simplified_rate_after": (
                    ""
                    if result.simplified_rate_after is None
                    else f"{result.simplified_rate_after:.4f}"
                ),
                "calls": result.calls,
            }
        )
        if result.unfixable_reason:
            unfixable.append(
                {
                    "task_id": result.task_id,
                    "status": result.status,
                    "unfixable_reason": result.unfixable_reason,
                    "unfixable_issues": issue_names,
                    "fail_checks": fail_names,
                    "simplified_rate_before": (
                        ""
                        if result.simplified_rate_before is None
                        else f"{result.simplified_rate_before:.4f}"
                    ),
                    "simplified_rate_after": (
                        ""
                        if result.simplified_rate_after is None
                        else f"{result.simplified_rate_after:.4f}"
                    ),
                }
            )
        if result.status != "fixable":
            continue
        for patch in result.patches:
            patches.append(
                {
                    "task_id": result.task_id,
                    "status": result.status,
                    "unfixable_reason": "",
                    "fail_checks": fail_names,
                    "group": _group_for_kind(patch.kind),
                    "kind": patch.kind,
                    "checks": ", ".join(check_name(c) for c in patch.check_ids),
                    "slot": patch.slot or "",
                    "criterion_id": patch.criterion_id or "",
                    "dimension": patch.dimension or "",
                    "comparison": patch.comparison or "",
                    "old_value": _csv_cell(patch.old_value),
                    "new_value": _csv_cell(patch.new_value),
                    "step_id": patch.step_id,
                    "export_field": patch.export_field,
                    "notes": patch.notes,
                    "error": patch.error,
                    "simplified_rate_before": (
                        ""
                        if result.simplified_rate_before is None
                        else f"{result.simplified_rate_before:.4f}"
                    ),
                    "simplified_rate_after": (
                        ""
                        if result.simplified_rate_after is None
                        else f"{result.simplified_rate_after:.4f}"
                    ),
                }
            )

    (out_dir / "fixer_backfill.json").write_text(
        json.dumps(patches, indent=2, default=str), encoding="utf-8"
    )
    _write_csv(out_dir / "all_patches.csv", patches, PATCH_COLUMNS)
    _write_csv(
        out_dir / "tasks.csv",
        tasks_rows,
        [
            "task_id",
            "status",
            "unfixable_reason",
            "unfixable_issues",
            "fail_checks",
            "n_patches",
            "simplified_rate_before",
            "simplified_rate_after",
            "calls",
        ],
    )
    _write_csv(
        out_dir / "unfixable.csv",
        unfixable,
        [
            "task_id",
            "status",
            "unfixable_reason",
            "unfixable_issues",
            "fail_checks",
            "simplified_rate_before",
            "simplified_rate_after",
        ],
    )
    grouped: dict[str, list[dict]] = {name: [] for name in CSV_GROUPS}
    grouped["other"] = []
    for row in patches:
        grouped.setdefault(row["group"], []).append(row)
    for group, rows in grouped.items():
        _write_csv(out_dir / f"{group}.csv", rows, PATCH_COLUMNS)


def _summarize(results: list[TaskFixResult]) -> str:
    counts: dict[str, int] = {}
    calls = 0
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        calls += result.calls
    bits = [f"{k}={v}" for k, v in sorted(counts.items())]
    return f"tasks={len(results)} {' '.join(bits)} field_calls={calls}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--from-snowflake", action="store_true")
    ap.add_argument(
        "--project",
        default="",
        metavar="KEY",
        help="force a project profile (l1, l1_sxs, aspirational, …) instead of "
             "resolving from each row's project id",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--fetch-conversations", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="fixable tasks in flight at once; field calls still share --workers",
    )
    ap.add_argument("--hydrate-workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task-ids", default=None, help="comma-separated subset")
    ap.add_argument(
        "--fill-dims",
        default=None,
        help="generated 1-5s CSV to attach as contributor ratings before planning",
    )
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    forced_profile = None
    if args.project:
        forced_profile = profile_by_key(args.project)
        if forced_profile is None:
            ap.error(
                f"unknown --project {args.project!r}; "
                f"known: {', '.join(p.key for p in PROFILES)}"
            )
        print(
            f"project: forcing profile {forced_profile.key} ({forced_profile.name})",
            file=sys.stderr,
        )
    if args.from_snowflake:
        tasks, ingest_results = load_taskattempts_csv(
            args.tasks_csv, profile=forced_profile
        )
        for row in ingest_results:
            if row.error:
                print(f"ingest: skipped {row.task_id}: {row.error}", file=sys.stderr)
    else:
        tasks = load_tasks(args.tasks_csv)

    wanted = (
        {t.strip() for t in args.task_ids.split(",") if t.strip()}
        if args.task_ids
        else {t["task_id"] for t in report.get("tasks") or []}
    )
    tasks = [t for t in tasks if t.task_id in wanted]

    if args.fill_dims:
        from .fill_dims import apply_fill_to_tasks, load_fill_csv

        fill_rows = load_fill_csv(Path(args.fill_dims))
        slot = str((fill_rows[0].get("slot") if fill_rows else "") or "D").upper()
        attached = apply_fill_to_tasks(tasks, fill_rows, slot)  # type: ignore[arg-type]
        print(
            f"fill-dims: attached generated ratings on {attached} tasks",
            file=sys.stderr,
        )

    policy = DEFAULT_POLICY
    if args.snapshot_dir:
        policy = replace(policy, snapshot_dir=args.snapshot_dir)
    if forced_profile is not None:
        policy = forced_profile.policy(policy)
    else:
        # Aspirational policy overrides (weight scale, justification rule, etc.)
        # when every remaining task is on that project.
        project_ids = {t.project_id for t in tasks}
        if len(project_ids) == 1:
            chosen = profile_for_project(next(iter(project_ids)))
            if chosen is not None:
                policy = chosen.policy(policy)

    if args.fetch_conversations and not args.dry_run:
        hydration = hydrate_conversations(
            tasks, policy, workers=args.hydrate_workers
        )
        print(
            f"conversations: hydrated {hydration.hydrated}/{hydration.submissions} "
            f"submissions, {hydration.turns} turns",
            file=sys.stderr,
        )

    if args.dry_run:
        from .llm import FakeModelClient

        client: ModelClient = FakeModelClient(lambda req: None)
        cache = None
    else:
        client, cache = build_client(
            model=args.model,
            effort=args.effort,
            cache_path=args.cache_db,
            policy_version="fixer-v1",
            max_in_flight=args.workers,
        )
    profile_by_task = {
        t.task_id: forced_profile or profile_for_project(t.project_id) for t in tasks
    }
    plans = plan_report(tasks, report, profile_by_task)
    try:
        results = run_fixer(
            tasks,
            report,
            client,
            policy,
            workers=args.workers,
            task_workers=args.task_workers,
            dry_run=args.dry_run,
            profile_by_task=profile_by_task,
        )
    finally:
        if cache is not None:
            cache.close()
    write_outputs(Path(args.out_dir), plans, results)
    print(_summarize(results), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
