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
import re
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
from .fixer_invariants import (
    MUCH_BETTER_RE,
    RANKING_CONSISTENCY_NOTE,
    SLIGHTLY_BETTER_RE,
    TIE_RE,
    assistant_generated_file,
    audit_fix_results,
    audit_task,
    cited_turns,
    copied_justification_pairs,
    dimension_is_na,
    empty_turns_ok,
    foreign_slots,
    form_likert_side,
    inversion_is_unexplained,
    is_na_score_token,
    likert_magnitude_ok,
    patched_task,
    sxs_body,
    write_invariant_report,
    _preference_side_simple,
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

JUSTIFICATION_VOICE_PROMPT = """
You rewrite Honeybee QC justifications a reviewer will paste into the form.

Voice
Write the way a real person talks when they actually looked at the chats and are explaining the call. Complete sentences. Detailed. Concrete. Direct, a little informal, not an essay.

Not rushed. Not fragments. You can land 2–3 points. You cannot be vague.

Name who did the thing. Model A, Model B, Model C, or the user. Never "they" unless the last sentence already named the model and it's still obvious. If a reader has to ask who "they" is, rewrite it.

Not fake. Fake is polished LLM prose, "the model demonstrates", and lists of abstractions (planning, coding, explaining) with no turn and no example.

Do
- Multiple points are good. Each one needs its own example. "Better at planning, coding, and explaining" is three empty claims
- Sound like a person: "on turn 6 Model A still recommends the $38 line after the user capped budget." Not like a rubric: "this demonstrates a deficiency in instruction following"
- Contractions are fine. "I" is fine. Don't perform casualness with one-word sentences

Don't
- Overexplain the scoring system or restate the dimension definition
- Recap the entire trajectory
- Open with Upon reviewing / The model demonstrates / Overall / It is worth noting
- Em dash (—), en dash (–) as an em dash, spaced --

Banned: delve, tapestry, furthermore, moreover, notably, overall, in conclusion, it is worth noting, multifaceted, robust, nuanced, comprehensive, insightful, seamless, "this highlights", utilize, leverage, "the model demonstrates"

BAD (vague justification, no examples)
"A was better at planning, coding, and explaining. B was worse at all three. The user had to keep correcting B. I'd pick A."

BAD (who is they?)
"Turn 7 they cite budget.csv but the uploaded file is spend_sept.csv."
Who. Model A? Model B? The user? If you mean Model A, write Model A.

BAD (fake-detailed, no real examples for why this is better, just generic jargon)
"Upon reviewing both trajectories, Model A demonstrates stronger instruction following, more robust tool use, and a more comprehensive final artifact. Turn 4 showcases an initial plan, turn 9 underscores a verification gap, and turn 14 highlights a polished writeup."

BAD (SxS canned)
"Verdict: Model B is much better (Likert 7/7; 1 = A much better, 7 = B much better). Even though the dimension scores favor Model A, Model B is the SxS winner because the pairwise outcome outweighs the mean of the 1-5s."

GOOD
"On turn 3 Model A still writes 16:30 after the user asked for 12-hour times, and Model A never corrects it later on. On turn 7 Model A cites budget.csv but the uploaded file is spend_sept.csv, so that path is just wrong. Model A's recap on turn 11 is actually usable (dates and owners are right) which is why this isn't a 1, but those two misses are real, so I'm at a 2."

GOOD (other model, not a copy)
"Model B switches to 12-hour on turn 3 as soon as the user says am/pm, and the recap keeps 7:30pm not 19:30. On turn 6 Model B uses spend_sept.csv, the file that was actually uploaded. Model B still skips timezone on turn 9 when the user asks whether this is local or UTC, so I'm not giving a 5, but the two things Model A got wrong are handled here."

GOOD (SxS)
"Model A's 1-5s come out higher because the tables are cleaner and Model A's recap on turn 14 is easier to scan. The actual dealbreaker is turn 8: the user said the cap is $25 and Model A still pushes Bring data, reach sport shop at 38.7533, then repeats that same line on turn 12. Model B's writeup is messier, which is why the means look worse, but on turn 7 Model B drops that set and does not bring it back. That's why Model B wins slightly."

Grounding
Only the chat and files in front of you. The turn you cite has to contain the thing you claim. A user complaint is not proof of what the model did. N/A: one clear sentence, empty turns.

Scores
1–5 must match the paragraph. Likert 1–7, only 4 is a tie, 3/5 slight, 1/7 much better. Don't flip the winner. Don't turn a winner into a tie. If you can't ground the change, leave the current text.
""".strip()

SYSTEM_PROMPT = (
    "You are correcting one field on a Honeybee model-comparison task. An "
    "independent QC audit already failed this task and named the defect. You "
    "do not re-litigate whether the audit is right, except for turn citations: "
    "keep any current citation that supports the claim, even if the audit "
    "flagged it. Never invent a turn number that is not in this conversation. "
    "If the list is empty and the dimension is scored 1–3, pick the in-range "
    "model turn(s) whose text supports the justification. A score of 4 or 5 "
    "may stay empty. N/A must be empty. Change "
    "only the field you were given. Do not invent share links, uploaded files, "
    "or conversation turns. Do not claim a turn contains text that is not in "
    "the conversation block. Write the value a reviewer should paste back into "
    "the form. A score and its justification (or a Likert and its writeup) "
    "must agree: never leave a high writeup next to a low score, or a "
    "preference paragraph that names the opposite winner from the Likert. "
    "Criterion scores are binary: 0 = not met, 1 = met. Never write 0 to mean "
    "pass or 1 to mean fail. Never raise a criterion from 0 to 1: a contributor "
    "0 stays 0. Only 1→0 is allowed. Dimension ratings are 1-5 (1 worst, 5 best). "
    "Dimension 1 is worst, not 'met'. Never describe a dimension score as a "
    "criterion 0/1. Do not flip a ranking winner and do not move a winner "
    "to a tie. Stay in the same Likert band (1-3 left, 4 tie, or 5-7 right). "
    "Name the winner as the slot in this pair (Model A / Model C on an A-vs-C "
    "task), never a product nickname or a slot that is not in the pair. "
    "Do not jump a dimension rating by more than 2 points. If a dimension is "
    "N/A, the score is not-applicable and the turn list is empty.\n\n"
    + JUSTIFICATION_VOICE_PROMPT
)


def ranking_scale_instruction(comparison: str | None) -> str:
    """Spell 1 as the left slot. Hardcoding A is how D-better writeups shipped as 7.

    Paste form matches the auditor's classic bipolar 7: only 4 is a tie.
    3 is a slight left win, 5 is a slight right win. Teaching 3-5 as ties
    is how we parked real winners on the midpoint.
    """
    left, right = "left", "right"
    if comparison and len(comparison) >= 2:
        left, right = comparison[0], comparison[1]
    return f"""
Likert scale for this comparison (do not invert it):
  1 = Model {left} is much better
  2 = Model {left} is better
  3 = Model {left} is slightly better
  4 = tie
  5 = Model {right} is slightly better
  6 = Model {right} is better
  7 = Model {right} is much better
1-3 prefer Model {left} (3 is a slight {left} win, not a tie). Only 4 is a tie.
5-7 prefer Model {right} (5 is a slight {right} win, not a tie). A paragraph
that prefers Model {left} must be 1-3. A paragraph that prefers Model {right}
must be 5-7. A tie / mixed / neither writeup must be 4.
Do not flip the winner and do not move a winner to a tie. If the current
Likert is 1-3, the new value must stay 1-3. If it is 5-7, stay 5-7. If it
is already 4, keep 4.
Name the winner as Model {left} or Model {right}. A product nickname
(Gemini, ChatGPT, Claude) is only that slot if it is that slot's product.
"Gemini is better" with Likert 5-7 is a winner flip when Gemini is Model
{left}. Nearly identical / equivalent / mixed writeups belong on 4, not
on 1-3 or 5-7.
"""

SCORE_POLARITY_INSTRUCTION = """
Criterion score polarity (do not invert it):
  0 = the criterion is NOT met
  1 = the criterion IS met
Never write 0 because the model did the thing, and never write 1 because it failed.
Notes must use those words: "met" goes with 1, "not met" goes with 0.
Never raise a 0 to a 1. If the current score is 0, return 0. A 0→1 paste
is how rows claimed a criterion was met against the dump.
"""

DIMENSION_POLARITY_INSTRUCTION = """
Dimension scores are 1-5 only (1 = worst, 5 = best). They are not criterion
0/1. Never write "score of 1 (met)", "criterion met", or "dimension met".
A 1 next to a success writeup (or a 5 next to "failed") is itself a fail.
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
- Keep every unflagged citation that is still in this conversation. You may
  drop or replace flagged numbers only. Do not add a turn that was not already
  cited unless it replaces a flagged one, OR the current list is empty and
  the dimension is not N/A — then pick the in-range model turn(s) whose text
  actually supports the justification claim.
- If this dimension is N/A / no tools / no memory, return an empty list.
  Never copy another dimension's turns onto an N/A row.
- A dimension scored 5 may stay empty. Do not invent a cite just to fill a 5.
- Only cite turn numbers that appear in the conversation above. If a number is
  not shown, it does not exist. Do not claim the transcript was truncated in
  order to drop a later turn that is on the page. The model text at the cited
  turn must support the specific justification claim, not merely share a topic.
  User-turn complaints are not evidence.
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


# Bump when prompts or apply-gates change so a re-run does not reuse unsafe cache hits.
FIX_CACHE_VERSION = "v8"


def fix_request_key(task_id: str, field_id: str) -> str:
    return f"fix::{FIX_CACHE_VERSION}::{task_id}::{field_id}"


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
    # Hygiene / ranking-companion jobs. Missing model output drops them instead
    # of writing an error patch that would fail unrelated field tests.
    optional: bool = False

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
            optional=self.optional and other.optional,
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
    lowered_reason = reason.lower()
    for check_id in fail_checks:
        if check_id in TRAJECTORY_CHECKS or check_id == UNAUDITABLE_CHECK:
            name = check_name(check_id)
            # Unique Links (91) on HTML-export B must not appear as a blocker
            # just because the eval recorded it. Only names the leftover
            # reason actually cites are unfixable issues.
            if check_id != UNAUDITABLE_CHECK and name.lower() not in lowered_reason:
                continue
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


def unique_links_html_export_na(task: Task) -> bool:
    """True when Unique Links (91) must not park this task.

    Spark HTML/MHTML B has no per-turn Gemini share URLs. Re-running 91 on
    the ingested task yields N/A / not fail. Share-link B that still fails
    uniqueness stays a trajectory blocker. Tasks with no HTML export keep
    the ordinary 91 trajectory gate.
    """
    from .gates import evaluate_check_91, html_export_without_share_manifest

    if not any(html_export_without_share_manifest(sub) for sub in task.trajectories()):
        return False
    return evaluate_check_91(task).band != "fail"


def unfixable_reason(task: Task, report_task: dict, profile: ProjectProfile | None) -> str:
    """Why this task cannot be repaired by editing fields. Empty if it can."""
    fails = set(_fail_ids(report_task))
    if report_task.get("verdict") == "unauditable" or UNAUDITABLE_CHECK in fails:
        return "task is unauditable; the submission itself cannot be trusted"
    # HTML/MHTML Spark B has no per-turn Gemini share URLs. Unique Links (91)
    # is N/A on that slot — do not park the task or skip 310/400/450/460.
    if 91 in fails and unique_links_html_export_na(task):
        fails.discard(91)
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
            optional=job.optional,
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
            optional=job.optional,
        )
    if job.kind == "preference_likert":
        sxs = _comparison(task, job.comparison)
        return FieldJob(
            kind="preference_justification",
            check_ids=job.check_ids,
            current_value=sxs.justification if sxs else "",
            feedback=feedback,
            comparison=job.comparison or (sxs.pair_label if sxs else None),
            optional=job.optional,
        )
    if job.kind == "preference_justification":
        sxs = _comparison(task, job.comparison)
        return FieldJob(
            kind="preference_likert",
            check_ids=job.check_ids,
            current_value=sxs.likert if sxs else None,
            feedback=feedback,
            comparison=job.comparison or (sxs.pair_label if sxs else None),
            optional=job.optional,
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


def _hygiene_jobs(task: Task) -> list[FieldJob]:
    """Edits that must happen even when the eval never failed that check.

    Empty N/A copies, file-claim vs checkbox No, 3/5 treated as ties, and
    unexplained ranking inversions used to ship because nobody asked for them
    on the next run.
    """
    jobs: list[FieldJob] = []
    for dim in task.dimension_ratings:
        if not _slot_rewritable(task, dim.model):
            continue
        legal = _legal_turns(task, dim.model)
        current = list(dim.relevant_turns or [])
        if dimension_is_na(dim):
            if current:
                jobs.append(
                    FieldJob(
                        kind="dimension_turns",
                        check_ids=(310,),
                        slot=dim.model,
                        dimension=dim.dimension,
                        current_value=current,
                        flagged_turns=tuple(current),
                        feedback=(
                            "This dimension is N/A. The turn list must be empty. "
                            "Do not copy cites from another dimension."
                        ),
                        optional=True,
                    )
                )
            if dim.rating is not None:
                jobs.append(
                    FieldJob(
                        kind="dimension_rating",
                        check_ids=(300,),
                        slot=dim.model,
                        dimension=dim.dimension,
                        current_value=dim.rating,
                        feedback=(
                            "This dimension is N/A. The score must be "
                            "not-applicable, not a 1–5."
                        ),
                        optional=True,
                    )
                )
            continue
        cited = cited_turns(dim.justification)
        current_set = set(current)
        if cited and current_set and not (cited & current_set):
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=dim.model,
                    dimension=dim.dimension,
                    current_value=current,
                    flagged_turns=tuple(current),
                    feedback=(
                        f"The justification names turns {sorted(cited)} but the "
                        f"selected list is {current}, and they do not overlap. "
                        "Replace the list with the in-range cited turns."
                    ),
                    optional=True,
                )
            )
        oor = [n for n in current if legal and n not in legal]
        if oor:
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=dim.model,
                    dimension=dim.dimension,
                    current_value=current,
                    flagged_turns=tuple(oor),
                    feedback=(
                        f"These turn numbers are not in Model {dim.model}'s "
                        f"conversation: {oor}. Replace them with in-range model "
                        "turns whose text supports the justification, or drop them."
                    ),
                    optional=True,
                )
            )
        elif not current:
            if empty_turns_ok(dim):
                continue
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=dim.model,
                    dimension=dim.dimension,
                    current_value=[],
                    flagged_turns=(),
                    feedback=(
                        "This dimension is scored 1–3 and has no turn citations. "
                        "Read the conversation and pick the in-range model "
                        "turn(s) whose text supports the justification claim. "
                        "Only use turn numbers that appear above. Empty is "
                        "legal for N/A and for a 4 or 5."
                    ),
                    optional=True,
                )
            )
        elif current_set == {1} and legal and max(legal) >= 7:
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=dim.model,
                    dimension=dim.dimension,
                    current_value=current,
                    flagged_turns=(1,),
                    feedback=(
                        "Turn 1 is not a usable cite on a long transcript. "
                        "Pick the in-range model turn whose text supports the "
                        "justification."
                    ),
                    optional=True,
                )
            )

    for left_slot, right_slot, dimension in copied_justification_pairs(task):
        for slot in (left_slot, right_slot):
            if not _slot_rewritable(task, slot):
                continue
            dim = _dim(task, slot, dimension)
            if dim is None:
                continue
            jobs.append(
                FieldJob(
                    kind="dimension_justification",
                    check_ids=(300,),
                    slot=slot,
                    dimension=dimension,
                    current_value=dim.justification,
                    feedback=(
                        f"This justification is nearly identical to Model "
                        f"{right_slot if slot == left_slot else left_slot}'s "
                        "writeup on the same dimension. Rewrite it from this "
                        "model's transcript only."
                    ),
                    optional=True,
                )
            )

    for sub in task.trajectories():
        if sub.produced_final_outcome is not False:
            continue
        generated, reason = assistant_generated_file(sub)
        if not generated:
            continue
        jobs.append(
            FieldJob(
                kind="produced_final_outcome",
                check_ids=(82,),
                slot=sub.model,
                current_value=False,
                feedback=(
                    f"Model {sub.model} produced-outcome checkbox is No, but the "
                    f"conversation generated a file ({reason}). Return true."
                ),
                optional=True,
            )
        )

    for sxs in task.comparison_list():
        if not _comparison_rewritable(task, sxs.pair_label):
            continue
        band = form_likert_side(sxs.likert) or _likert_band(sxs.likert)
        side = _preference_side_simple(sxs.justification, sxs.left, sxs.right)
        if side is None:
            side = _preference_side(sxs.justification, sxs.left, sxs.right, task)
        needs_sxs = False
        feedback = ranking_scale_instruction(sxs.pair_label) + RANKING_CONSISTENCY_NOTE
        if band in {"left", "right"} and side == "tie":
            needs_sxs = True
            feedback += (
                f"\nLikert {sxs.likert} is a win, not a tie. 3 and 5 are slight "
                "wins. Only 4 is a tie. Rewrite the paragraph so it matches."
            )
        elif band in {"left", "right"} and side in {"left", "right"} and side != band:
            needs_sxs = True
            feedback += (
                f"\nLikert {sxs.likert} prefers the {band} slot but the writeup "
                "prefers the other. They must agree. Do not flip the winner."
            )
        if inversion_is_unexplained(task, sxs):
            needs_sxs = True
            feedback += (
                "\nDimension means favor the other model from this Likert. "
                "Explain the inversion in the writeup with concrete turns "
                "that name Model A/B/C. Do not open with Verdict or write "
                "that the pairwise outcome outweighs the mean. If you cannot "
                "ground it, leave the current text; the Likert/scores will "
                "not paste."
            )
        foreign = foreign_slots(sxs.justification, sxs.left, sxs.right)
        if foreign:
            needs_sxs = True
            feedback += (
                f"\nThis pair is Model {sxs.left} vs Model {sxs.right}. "
                f"Do not name Model {', '.join(foreign)}. Overlay Verdict "
                "lines do not clean the body."
            )
        if not likert_magnitude_ok(sxs.likert, sxs.justification):
            needs_sxs = True
            feedback += (
                "\nLikert magnitude must match the wording: 3 and 5 are "
                "slightly better; 1 and 7 are much better. Ignore the scale "
                "legend inside a Verdict line."
            )
        if not needs_sxs:
            continue
        jobs.append(
            FieldJob(
                kind="preference_justification",
                check_ids=(400, 460, 470),
                comparison=sxs.pair_label,
                current_value=sxs.justification,
                feedback=feedback,
                optional=True,
            )
        )
    return jobs


def _ensure_ranking_jobs_for_dim_scores(task: Task, jobs: list[FieldJob]) -> list[FieldJob]:
    """Changing 1–5s without touching SxS is how Inconsistent Ranking shipped."""
    if not any(j.kind == "dimension_rating" for j in jobs):
        return jobs
    extra: list[FieldJob] = []
    present = {(_pair_id(j), j.kind) for j in jobs if _pair_id(j)}
    for sxs in task.comparison_list():
        if not _comparison_rewritable(task, sxs.pair_label):
            continue
        pid = ("rank", sxs.pair_label)
        feedback = (
            ranking_scale_instruction(sxs.pair_label)
            + RANKING_CONSISTENCY_NOTE
            + "\nDimension scores on this task are also being rewritten. "
            "The SxS Likert and writeup must stay consistent with the new 1–5s."
        )
        if (pid, "preference_likert") not in present:
            extra.append(
                FieldJob(
                    kind="preference_likert",
                    check_ids=(400, 460),
                    comparison=sxs.pair_label,
                    current_value=sxs.likert,
                    feedback=feedback,
                    optional=True,
                )
            )
        if (pid, "preference_justification") not in present:
            extra.append(
                FieldJob(
                    kind="preference_justification",
                    check_ids=(400, 460, 470),
                    comparison=sxs.pair_label,
                    current_value=sxs.justification,
                    feedback=feedback,
                    optional=True,
                )
            )
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
    task: Task,
    counts: dict[str, Any],
    *,
    rest_kind: str,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Items the auditor named as wrong citations, with those turn numbers.

    Missing citations and unverifiable fetches are not incorrect selections.
    Rewriting those invents turns the contributor never picked.
    `rest_kind` is `criterion` (expand UUIDs / skip invented keys) or
    `dimension` (keep the dimension name).
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
        if rest_kind == "criterion":
            for cid in _resolve_criterion_ids(task, rest):
                out.append((slot, cid, turns))
        else:
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


def _canonical_dimension(raw: str) -> str | None:
    """Real form dimension, or None. Title-as-id / parenthetical bags are dropped."""
    text = str(raw or "").strip()
    if not text:
        return None
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    for candidate in (text, stripped):
        name = _normalize_dimension(candidate)
        if name in DIMENSION_TO_PREFIX:
            return name
    return None


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


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_INVENTED_CRITERION_PREFIXES = ("missing::", "overlap::", "bulk_")


def _resolve_criterion_ids(task: Task, raw: str) -> list[str]:
    """Map an auditor item onto real rubric ids. Invented keys become nothing.

    280/281 sometimes emit `B::[uuid] [uuid]` or a criterion title instead of
    the form id. Writing those through as `criteria[[uuid] …]` is how paste
    sheets landed on fields that do not exist.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith(_INVENTED_CRITERION_PREFIXES):
        return []
    if text.lower().startswith("criterion_"):
        text = text[len("criterion_") :]
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]" and text.count("[") == 1:
        text = text[1:-1].strip()
    rubric_ids = {c.criterion_id for c in task.rubric}
    found = [u for u in _UUID_RE.findall(text) if u in rubric_ids]
    if found:
        return list(dict.fromkeys(found))
    if text in rubric_ids:
        return [text]
    collapsed = " ".join(text.split())
    for crit in task.rubric:
        if " ".join(crit.text.split()) == collapsed:
            return [crit.criterion_id]
    return []


def _slot_rewritable(task: Task, slot: str | None) -> bool:
    """Skip score/turn/dim rewrites when that slot has no real conversation.

    A one-exchange marketing stub is not a trajectory. Citing Turn 19 on it is
    how paste rows named turns that are not in the dump. A fetch that failed
    or a share page that loaded dead is the same: there is nothing to cite.
    """
    if not slot:
        return False
    if slot == "S":
        return task.simplified is not None
    sub = task.submission(slot)  # type: ignore[arg-type]
    if sub is None:
        return False
    if sub.share_page_dead or sub.fetch_failed:
        return False
    indices = sub.turn_indices()
    if not indices:
        return False
    return sub.exchange_count() >= 2 and max(indices) >= 4


def _comparison_rewritable(task: Task, comparison: str | None) -> bool:
    """Skip ranking when either side is a stub or missing conversation."""
    if not comparison or len(comparison) < 2:
        return False
    return all(_slot_rewritable(task, slot) for slot in comparison[:2])


def _legal_turns(task: Task, slot: str | None) -> set[int]:
    if not slot:
        return set()
    sub = task.submission(slot)  # type: ignore[arg-type]
    if sub is None:
        return set()
    return sub.turn_indices()


def _filter_turn_values(task: Task, slot: str | None, value: Any) -> Any:
    """Drop turn numbers that are not in this slot's conversation.

    An empty list is a real value (N/A, or every cite was illegal). `None`
    means "could not interpret", which is not the same as "paste []".
    """
    allowed = _legal_turns(task, slot)
    if isinstance(value, int):
        if not allowed:
            return None
        return value if value in allowed else None
    if isinstance(value, list):
        kept = []
        for item in value:
            try:
                n = int(item)
            except (TypeError, ValueError):
                continue
            if not allowed or n in allowed:
                kept.append(n)
        if allowed:
            return kept
        return kept if not value else None
    return None


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
            if not _slot_rewritable(task, slot):
                continue
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
            if not _slot_rewritable(task, slot):
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
            for cid in _resolve_criterion_ids(task, item):
                crit = _criterion(task, cid)
                auditor = ""
                for finding in findings:
                    if finding.get("criterion_id") == cid and finding.get("l1"):
                        auditor = finding["l1"].get("auditor") or ""
                jobs.append(
                    FieldJob(
                        kind="criterion_l1",
                        check_ids=(200,),
                        criterion_id=cid,
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
            for cid in _resolve_criterion_ids(task, item):
                crit = _criterion(task, cid)
                if crit is None:
                    continue
                issue_lines = []
                for issue in (by_id.get(cid) or {}).get("issues") or []:
                    issue_lines.append(
                        f"[{issue.get('category')}] {issue.get('description')} "
                        f"(evidence: {issue.get('evidence')})"
                    )
                jobs.append(
                    FieldJob(
                        kind="criterion_text",
                        check_ids=(check_id,),
                        criterion_id=cid,
                        current_value=crit.text,
                        feedback=feedback
                        + (("\n" + "\n".join(issue_lines)) if issue_lines else ""),
                    )
                )
        return jobs

    if check_id == 260:
        findings = extra.get("rubric_findings") or []
        by_id = {f.get("criterion_id"): f for f in findings}
        for item in items:
            for cid in _resolve_criterion_ids(task, item):
                crit = _criterion(task, cid)
                if crit is None:
                    continue
                weight_info = (by_id.get(cid) or {}).get("weight") or {}
                jobs.append(
                    FieldJob(
                        kind="criterion_weight",
                        check_ids=(260,),
                        criterion_id=cid,
                        current_value=crit.weight,
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
            slot, rest = _parse_model_item(str(item))
            if not slot or slot not in allowed or not _slot_rewritable(task, slot):
                continue
            for criterion_id in _resolve_criterion_ids(task, rest):
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
        for slot, criterion_id, flagged in _flagged_incorrect_citations(
            task, counts, rest_kind="criterion"
        ):
            if not _slot_rewritable(task, slot):
                continue
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
            if not slot or not dimension or not _slot_rewritable(task, slot):
                continue
            canonical = _canonical_dimension(dimension)
            if not canonical:
                continue
            dim = _dim(task, slot, canonical)
            if dim is None:
                continue
            jobs.append(
                FieldJob(
                    kind="dimension_rating",
                    check_ids=(check_id,),
                    slot=slot,
                    dimension=canonical,
                    current_value=dim.rating,
                    feedback=feedback,
                )
            )
        return jobs

    if check_id == 310:
        for slot, dimension, flagged in _flagged_incorrect_citations(
            task, counts, rest_kind="dimension"
        ):
            if not _slot_rewritable(task, slot):
                continue
            canonical = _canonical_dimension(dimension)
            if not canonical:
                continue
            dim = _dim(task, slot, canonical)
            if dim is None:
                continue
            jobs.append(
                FieldJob(
                    kind="dimension_turns",
                    check_ids=(310,),
                    slot=slot,
                    dimension=canonical,
                    current_value=list(dim.relevant_turns),
                    flagged_turns=flagged,
                    feedback=feedback
                    + f"\nFlagged citations (replace only if truly unsupported): {list(flagged)}",
                )
            )
        return jobs

    if check_id == 400:
        for sxs in task.comparison_list():
            if not _comparison_rewritable(task, sxs.pair_label):
                continue
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
                if sxs is None or not _comparison_rewritable(task, sxs.pair_label):
                    continue
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
            if not _slot_rewritable(task, slot):
                continue
            canonical = _canonical_dimension(dimension)
            if not canonical:
                continue
            dim = _dim(task, slot, canonical)
            if dim is None:
                continue
            jobs.append(
                FieldJob(
                    kind="dimension_justification",
                    check_ids=(450,),
                    slot=slot,
                    dimension=canonical,
                    current_value=dim.justification,
                    feedback=extra_fb,
                )
            )
        return jobs

    if check_id == 470:
        for sxs in task.comparison_list():
            if not _comparison_rewritable(task, sxs.pair_label):
                continue
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
    html_na_91 = 91 in fails and unique_links_html_export_na(task)
    # Turn count is a property of the conversation. Score flips cannot shorten S.
    if leftover and (
        "turn" in leftover.lower() or "outside" in leftover.lower()
    ):
        skip_checks.add(97)
    jobs: list[FieldJob] = list(_hygiene_jobs(task))
    if not (verdict in {"clean", "pass_with_issues"} and not fails):
        for check in report_task.get("checks") or []:
            if check.get("check_id") in skip_checks:
                continue
            jobs.extend(_jobs_from_check(task, check, extra, profile))
    jobs = _ensure_ranking_jobs_for_dim_scores(
        task, _ensure_paired_jobs(task, _merge_jobs(jobs))
    )
    if not jobs:
        ignore = set(FIXER_IGNORED_CHECKS)
        if html_na_91:
            ignore.add(91)
        actionable = [c for c in fails if c not in ignore]
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
    dimension_score_table: str | None = None,
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
    if dimension_score_table:
        paired_block += (
            "\n## Dimension scores (including corrections from this pass)\n\n"
            f"{dimension_score_table}\n"
            + RANKING_CONSISTENCY_NOTE
        )

    turn_block = ""
    if job.kind in {"criterion_turns", "dimension_turns"}:
        turn_block = "\n" + TURN_KEEP_INSTRUCTION
        if job.flagged_turns:
            turn_block += (
                f"\nFlagged as incorrect: {list(job.flagged_turns)}\n"
            )
        legal = sorted(_legal_turns(task, job.slot))
        if legal:
            turn_block += (
                f"\nLegal turn numbers for Model {job.slot}: {legal}. "
                "Do not cite any other number.\n"
            )
    if job.kind in {"preference_likert", "preference_justification"}:
        turn_block += "\n" + ranking_scale_instruction(job.comparison)
        turn_block += RANKING_CONSISTENCY_NOTE
        if job.comparison:
            turn_block += f"This field is the {job.comparison} pair.\n"
    if job.kind == "criterion_score":
        turn_block += "\n" + SCORE_POLARITY_INSTRUCTION
    if job.kind in {"dimension_rating", "dimension_justification"}:
        turn_block += "\n" + DIMENSION_POLARITY_INSTRUCTION
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
        key=fix_request_key(task.task_id, job.field_id),
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


_MET_WITHOUT_NOT = re.compile(
    r"\b(?:is met|criterion is met|dimension met|indicate .{0,40}met|score of 1\s*\(\s*met\s*\))\b",
    re.I,
)
_NOT_MET = re.compile(r"\bnot met\b", re.I)
_SUCCESS_ON_ZERO = re.compile(
    r"\b(?:would execute|without errors|successfully (?:met|produced|created|ran)|"
    r"did the thing|is met|was met)\b",
    re.I,
)
_DIM_ONE_AS_MET = re.compile(
    r"score of 1\s*\(\s*(?:met|criterion met)\s*\)|1\s*\(\s*criterion met\s*\)|"
    r"paired score of 1\s*\(\s*met\s*\)|\bdimension met\b|\bcriterion met\b",
    re.I,
)
_RATING_PROSE = re.compile(
    r"\b(?:not met|satisfied|corrected to|model [abcd] (?:passed|failed)|score of [01])\b",
    re.I,
)
_HIGH_PRAISE = re.compile(
    r"\b(?:perfect|excellent|no issues|outstanding|fully (?:met|delivered))\b", re.I
)
_LOW_PAN = re.compile(
    r"\b(?:failed badly|unusable|worst|did not deliver|severely compromised)\b", re.I
)


def _likert_band(value: Any) -> str | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n in (1, 2, 3):
        return "left"
    if n == 4:
        return "tie"
    if n in (5, 6, 7):
        return "right"
    return None


def _likert_move_ok(old: Any, new: Any) -> bool:
    """Paste-safe: stay in the same band. Winner→tie fights the dump."""
    new_band = _likert_band(new)
    if new_band is None:
        return False
    old_band = _likert_band(old)
    if old_band is None:
        return True
    return old_band == new_band


def _provider_aliases(sub: Any) -> tuple[str, ...]:
    """Nicknames a preference writeup might use instead of Model A / Model B."""
    if sub is None:
        return ()
    blob = f"{getattr(sub, 'declared_provider', '')} {getattr(sub, 'final_link', '')}".lower()
    aliases: list[str] = []
    if "gemini" in blob:
        aliases.extend(["gemini", "google gemini"])
    if any(token in blob for token in ("chatgpt", "openai.com", "chatgpt.com")):
        aliases.extend(["chatgpt", "chat gpt", "gpt-4", "gpt-5", "gpt"])
    if "claude" in blob:
        aliases.extend(["claude", "anthropic"])
    return tuple(dict.fromkeys(aliases))


def _side_win(text: str, slot: str, aliases: tuple[str, ...]) -> bool:
    """True when the writeup names this slot (or its product) as the winner."""
    names = sorted({slot.lower(), *[a.lower() for a in aliases]}, key=len, reverse=True)
    for name in names:
        if len(name) == 1:
            win = re.search(
                rf"\b(?:model|product|response)\s+{re.escape(name)}\s+"
                rf"(?:is\s+|was\s+)?(?:much |slightly |clearly )?"
                rf"(?:better|ahead|preferred)\b",
                text,
            ) or re.search(
                rf"\bprefer(?:s|red)?\s+(?:model\s+)?{re.escape(name)}\b",
                text,
            )
        else:
            win = re.search(
                rf"\b{re.escape(name)}\s+(?:is\s+|was\s+|looks\s+)?"
                rf"(?:much |slightly |clearly )?(?:better|ahead|preferred)\b",
                text,
            ) or re.search(
                rf"\bprefer(?:s|red)?\s+{re.escape(name)}\b",
                text,
            )
        if win:
            return True
    return False


def _preference_side(
    text: str, left: str, right: str, task: Task | None = None
) -> str | None:
    """left / right / tie if the writeup is explicit, else None.

    Detection matches `audit_task`: TIE_RE on the body (Verdict legend ignored),
    then an explicit Model/Response winner. Provider nicknames are a fallback.
    """
    lower = sxs_body(text or "").lower()
    if TIE_RE.search(lower):
        return "tie"
    left_aliases = right_aliases = ()
    if task is not None:
        left_aliases = _provider_aliases(task.submission(left))  # type: ignore[arg-type]
        right_aliases = _provider_aliases(task.submission(right))  # type: ignore[arg-type]
    left_win = _side_win(lower, left, left_aliases)
    right_win = _side_win(lower, right, right_aliases)
    if left_win and not right_win:
        return "left"
    if right_win and not left_win:
        return "right"
    return None


def _criterion_notes_ok(value: Any, notes: str) -> bool:
    text = notes or ""
    if int(value) == 0:
        if _MET_WITHOUT_NOT.search(text) and not _NOT_MET.search(text):
            return False
        if _SUCCESS_ON_ZERO.search(text) and not _NOT_MET.search(text):
            return False
    if int(value) == 1 and _NOT_MET.search(text) and not re.search(
        r"\b(?:is met|was met|now met)\b", text, re.I
    ):
        return False
    return True


def _dimension_text_ok(value: Any, text: str) -> bool:
    blob = text or ""
    if _DIM_ONE_AS_MET.search(blob):
        return False
    try:
        n = int(value)
    except (TypeError, ValueError):
        return True
    if n <= 2 and _HIGH_PRAISE.search(blob):
        return False
    if n >= 4 and _LOW_PAN.search(blob):
        return False
    return True


def _turn_has_model_text(task: Task, slot: str | None, index: int) -> bool:
    if not slot:
        return True
    sub = task.submission(slot)  # type: ignore[arg-type]
    if sub is None or not sub.conversation:
        return True
    return any(
        t.index == index and t.role == "assistant" and (t.text or "").strip()
        for t in sub.conversation
    )


def _kept_turns(
    task: Task,
    job: FieldJob,
    old: Any,
    new: list[int],
    flagged: tuple[int, ...],
    legal: set[int],
    force_align: bool = False,
) -> list[int] | None:
    """Keep unflagged cites; only add a turn as a replacement for a flagged one.

    Empty is legal for N/A. Filling a previously empty non-N/A list is how we
    stop shipping blank turn annotations. `force_align` bypasses the "no adds
    unless flagged" rule for deterministic selected↔cited alignment.
    """
    dim = (
        _dim(task, job.slot or "", job.dimension or "")
        if job.kind == "dimension_turns"
        else None
    )
    if dim is not None and dimension_is_na(dim):
        return []
    kept = [n for n in new if (not legal or n in legal)]
    if legal:
        kept = [
            n for n in kept if n in legal and _turn_has_model_text(task, job.slot, n)
        ]
    old_nums: list[int] = []
    if isinstance(old, list):
        for item in old:
            try:
                old_nums.append(int(item))
            except (TypeError, ValueError):
                continue
    old_set = set(old_nums)
    flagged_set = set(flagged or ())
    if not old_nums:
        return kept
    if not kept:
        if dim is not None and not dimension_is_na(dim):
            return None
        if flagged_set and not ((old_set - flagged_set) & (legal or old_set)):
            return []
        return None
    unflagged = (old_set - flagged_set) & (legal or old_set)
    new_set = set(kept)
    if unflagged and not unflagged <= new_set and not force_align:
        return None
    added = new_set - old_set
    if added and not flagged_set and not force_align:
        return None
    return kept


def _text_cites_illegal_turns(task: Task, job: FieldJob, text: str) -> bool:
    cited = {int(n) for n in re.findall(r"\bturns?\s+(\d+)\b", text or "", re.I)}
    cited |= {int(n) for n in re.findall(r"\bT(\d+)\b", text or "")}
    if not cited:
        return False
    if job.comparison and len(job.comparison) >= 2:
        legal = _legal_turns(task, job.comparison[0]) | _legal_turns(
            task, job.comparison[1]
        )
    else:
        legal = _legal_turns(task, job.slot)
    if not legal:
        return True
    return bool(cited - legal)


def _guard_new_value(task: Task, job: FieldJob, new_value: Any, notes: str) -> Any | None:
    """None means drop the patch: it would not be paste-safe."""
    if job.kind in {
        "dimension_justification",
        "preference_justification",
        "key_turn_justification",
    } and _text_cites_illegal_turns(
        task, job, f"{notes or ''} {new_value if isinstance(new_value, str) else ''}"
    ):
        return None

    if job.kind == "criterion_score":
        try:
            n = int(new_value)
        except (TypeError, ValueError):
            return None
        if n not in (0, 1):
            return None
        try:
            old = int(job.current_value)
        except (TypeError, ValueError):
            old = None
        if old == 0 and n == 1:
            return None
        if not _criterion_notes_ok(n, notes):
            return None
        return n

    if job.kind == "criterion_weight":
        try:
            n = int(new_value)
        except (TypeError, ValueError):
            return None
        if n not in (1, 2, 3, 4, 5):
            return None
        try:
            old = int(job.current_value)
        except (TypeError, ValueError):
            old = n
        if abs(n - old) > 2:
            return None
        return n

    if job.kind == "criterion_text":
        text = str(new_value or "").strip()
        if not text or _RATING_PROSE.search(text):
            return None
        return text

    if job.kind == "dimension_rating":
        if is_na_score_token(new_value):
            dim = _dim(task, job.slot or "", job.dimension or "")
            if dim is not None and dimension_is_na(dim):
                return "not-applicable"
            return None
        try:
            n = int(new_value)
        except (TypeError, ValueError):
            return None
        if n not in (1, 2, 3, 4, 5):
            return None
        try:
            old = int(job.current_value)
        except (TypeError, ValueError):
            old = n
        if abs(n - old) > 2:
            return None
        if not _dimension_text_ok(n, notes):
            return None
        return n

    if job.kind == "dimension_justification":
        text = str(new_value or "").strip()
        if not text:
            return None
        dim = _dim(task, job.slot or "", job.dimension or "")
        score = dim.rating if dim else 3
        if not _dimension_text_ok(score, notes + " " + text):
            return None
        return text

    if job.kind == "preference_likert":
        try:
            n = int(new_value)
        except (TypeError, ValueError):
            return None
        if n not in (1, 2, 3, 4, 5, 6, 7):
            return None
        if not _likert_move_ok(job.current_value, n):
            return None
        if not _comparison_rewritable(task, job.comparison):
            return None
        return n

    if job.kind == "preference_justification":
        text = str(new_value or "").strip()
        if not text:
            return None
        if not _comparison_rewritable(task, job.comparison):
            return None
        left = (job.comparison or "AB")[0]
        right = (job.comparison or "AB")[1]
        if foreign_slots(text, left, right):
            return None
        side = _preference_side_simple(text, left, right)
        if side is None:
            side = _preference_side(text, left, right, task)
        band = form_likert_side(job.current_value) or _likert_band(job.current_value)
        sxs = _comparison(task, job.comparison)
        if sxs is not None:
            band = form_likert_side(sxs.likert) or band
            left, right = sxs.left, sxs.right
            if foreign_slots(text, left, right):
                return None
            if not likert_magnitude_ok(sxs.likert, text):
                return None
            side = _preference_side_simple(text, left, right)
            if side is None:
                side = _preference_side(text, left, right, task)
        elif not likert_magnitude_ok(job.current_value, text):
            return None
        if side and band and side != band:
            return None
        return text

    if job.kind == "produced_final_outcome":
        if not isinstance(new_value, bool):
            text = str(new_value).strip().lower()
            if text in {"true", "yes", "1"}:
                new_value = True
            elif text in {"false", "no", "0"}:
                new_value = False
            else:
                return None
        sub = task.submission(job.slot) if job.slot else None  # type: ignore[arg-type]
        generated, _reason = assistant_generated_file(sub)
        old = job.current_value
        if old is False and new_value is True and not generated:
            return None
        if old is True and new_value is False and generated:
            return None
        return bool(new_value)

    if job.kind == "key_turn_index":
        try:
            n = int(new_value)
        except (TypeError, ValueError):
            return None
        if n == 1:
            return None
        return n

    return new_value


def _drop_inconsistent_pairs(task: Task, patches: list[FieldPatch]) -> list[FieldPatch]:
    by_pair: dict[tuple, list[FieldPatch]] = {}
    unpaired: list[FieldPatch] = []
    for patch in patches:
        dummy = FieldJob(
            kind=patch.kind,  # type: ignore[arg-type]
            check_ids=tuple(patch.check_ids),
            current_value=patch.old_value,
            feedback="",
            slot=patch.slot,
            criterion_id=patch.criterion_id,
            dimension=patch.dimension,
            comparison=patch.comparison,
        )
        pid = _pair_id(dummy)
        if pid is None:
            unpaired.append(patch)
            continue
        by_pair.setdefault(pid, []).append(patch)

    kept: list[FieldPatch] = []
    drop: set[int] = set()
    for group in by_pair.values():
        rating = next((p for p in group if p.kind in _SCORE_KINDS), None)
        just = next((p for p in group if p.kind in _JUSTIF_KINDS), None)
        if rating and just:
            if rating.kind == "preference_likert":
                left = (rating.comparison or "AB")[0]
                right = (rating.comparison or "AB")[1]
                just_text = str(just.new_value or "")
                sxs = _comparison(task, rating.comparison)
                if sxs is not None:
                    left, right = sxs.left, sxs.right
                side = _preference_side_simple(just_text, left, right)
                if side is None:
                    side = _preference_side(just_text, left, right, task)
                band = form_likert_side(rating.new_value) or _likert_band(rating.new_value)
                if side and band and side != band:
                    drop.add(id(rating))
                    drop.add(id(just))
                    continue
                if foreign_slots(just_text, left, right):
                    drop.add(id(rating))
                    drop.add(id(just))
                    continue
                if not likert_magnitude_ok(rating.new_value, just_text):
                    drop.add(id(rating))
                    drop.add(id(just))
                    continue
            if rating.kind == "dimension_rating":
                if not _dimension_text_ok(rating.new_value, str(just.new_value or "")):
                    drop.add(id(rating))
                    drop.add(id(just))
                    continue
        elif just and not rating:
            if just.kind == "preference_justification":
                left = (just.comparison or "AB")[0]
                right = (just.comparison or "AB")[1]
                just_text = str(just.new_value or "")
                sxs = _comparison(task, just.comparison)
                if sxs is not None:
                    left, right = sxs.left, sxs.right
                side = _preference_side_simple(just_text, left, right)
                if side is None:
                    side = _preference_side(just_text, left, right, task)
                band = _likert_band(just.old_value)
                if sxs is not None:
                    band = form_likert_side(sxs.likert) or band
                if side and band and side != band:
                    drop.add(id(just))
                    continue
                if foreign_slots(just_text, left, right):
                    drop.add(id(just))
                    continue
                likert = sxs.likert if sxs is not None else just.old_value
                if not likert_magnitude_ok(likert, just_text):
                    drop.add(id(just))
                    continue
            if just.kind == "dimension_justification" and not _dimension_text_ok(
                None, str(just.new_value or "")
            ):
                drop.add(id(just))
                continue
        kept.extend(group)
    kept = [p for p in kept if id(p) not in drop]
    return unpaired + kept


def _drop_unexplained_ranking_inversions(
    task: Task, patches: list[FieldPatch]
) -> list[FieldPatch]:
    """Do not paste 1–5s that flip the ranking against the Likert with no explanation."""
    numeric_dim = [
        p
        for p in patches
        if p.kind == "dimension_rating"
        and p.new_value is not None
        and not p.error
        and not is_na_score_token(p.new_value)
    ]
    if not numeric_dim:
        return patches
    simulated = patched_task(task, [p for p in patches if not p.error])
    if not any(
        inversion_is_unexplained(simulated, sxs) for sxs in simulated.comparison_list()
    ):
        return patches
    drop_dims = {(p.slot, p.dimension) for p in numeric_dim}
    return [
        p
        for p in patches
        if not (
            p.kind in {"dimension_rating", "dimension_justification"}
            and (p.slot, p.dimension) in drop_dims
        )
    ]


_PASTE_FAMILIES = frozenset(
    {"dimension_turns", "sxs_scale", "sxs_magnitude", "inconsistent_ranking"}
)
_TIE_REPLACEMENTS = {
    "tie": "close contest",
    "tied": "close",
    "toss-up": "close call",
    "toss up": "close call",
    "dead heat": "close contest",
    "equally good": "both capable",
    "equally strong": "both capable",
    "equally matched": "both capable",
    "no clear winner": "a recorded winner on the required outcome",
    "no winner": "a recorded winner on the required outcome",
    "neither": "each",
    "mixed": "split across dimensions",
    "comparable": "in the same neighborhood",
    "roughly even": "close on some axes",
    "nearly identical": "very similar on some axes",
    "equivalent": "similar",
    "indistinguishable": "hard to separate on some axes",
    "essentially the same": "similar in places",
}
_LIKERT_MAGNITUDE = {
    1: "much better",
    2: "better",
    3: "slightly better",
    5: "slightly better",
    6: "better",
    7: "much better",
}


def _patch_identity(patch: FieldPatch) -> tuple:
    return (patch.kind, patch.slot, patch.dimension, patch.comparison, patch.criterion_id)


def _make_field_patch(
    *,
    kind: Kind,
    new_value: Any,
    old_value: Any = None,
    notes: str = "",
    slot: str | None = None,
    dimension: str | None = None,
    comparison: str | None = None,
    check_ids: tuple[int, ...] | list[int] = (),
    profile: ProjectProfile | None = None,
) -> FieldPatch:
    dummy = FieldJob(
        kind=kind,
        check_ids=tuple(check_ids),
        current_value=old_value,
        feedback="",
        slot=slot,
        dimension=dimension,
        comparison=comparison,
    )
    step_id, export_field = export_locator(dummy, profile)
    return FieldPatch(
        field_id=dummy.field_id,
        kind=kind,
        check_ids=list(check_ids),
        old_value=old_value,
        new_value=new_value,
        notes=notes,
        slot=slot,
        criterion_id=dummy.criterion_id,
        dimension=dimension,
        comparison=comparison,
        step_id=step_id,
        export_field=export_field,
    )


def _upsert_patch(patches: list[FieldPatch], new: FieldPatch) -> list[FieldPatch]:
    ident = _patch_identity(new)
    out: list[FieldPatch] = []
    replaced = False
    for patch in patches:
        if _patch_identity(patch) == ident:
            out.append(new)
            replaced = True
        else:
            out.append(patch)
    if not replaced:
        out.append(new)
    return out


def _neutralize_tie_language(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = (match.group(1) or match.group(0)).lower()
        replacement = _TIE_REPLACEMENTS.get(token)
        if replacement is None:
            replacement = _TIE_REPLACEMENTS.get(match.group(0).lower(), "close contest")
        orig = match.group(0)
        if orig[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return TIE_RE.sub(repl, text or "")


def _align_magnitude(just: str, likert: int) -> str:
    if likert_magnitude_ok(likert, just):
        return just
    if likert in (3, 5):
        return MUCH_BETTER_RE.sub("slightly better", just or "")
    if likert in (1, 7):
        return SLIGHTLY_BETTER_RE.sub("much better", just or "")
    return just


def _lead_sentence(likert: int, left: str, right: str) -> str:
    if likert == 4:
        return (
            f"Model {left} and Model {right} are a recorded tie on the 1–7 SxS "
            "scale (only 4 is a tie)."
        )
    magnitude = _LIKERT_MAGNITUDE.get(likert, "better")
    if likert in (1, 2, 3):
        return f"Model {left} is {magnitude} than Model {right}."
    return f"Model {right} is {magnitude} than Model {left}."


def _strip_opposite_winner_lines(text: str, likert: int, left: str, right: str) -> str:
    want = form_likert_side(likert)
    if want not in {"left", "right"}:
        return text
    kept: list[str] = []
    for line in (text or "").splitlines():
        side = _preference_side_simple(line, left, right)
        if side in {"left", "right"} and side != want:
            continue
        kept.append(line)
    return "\n".join(kept)


def _rewrite_sxs_body(text: str, likert: Any, left: str, right: str) -> str:
    """Deterministic writeup that matches the existing Likert. Never flips the winner."""
    try:
        n = int(likert)
    except (TypeError, ValueError):
        return text or ""
    new_just = text or ""
    if n != 4:
        new_just = _neutralize_tie_language(new_just)
    new_just = _align_magnitude(new_just, n)
    new_just = _strip_opposite_winner_lines(new_just, n, left, right)
    side = _preference_side_simple(new_just, left, right)
    want = form_likert_side(n)
    if want in {"left", "right"} and side != want:
        lead = _lead_sentence(n, left, right)
        body = sxs_body(new_just).strip()
        new_just = f"{lead}\n\n{body}" if body else lead
        if n != 4:
            new_just = _neutralize_tie_language(new_just)
        new_just = _align_magnitude(new_just, n)
        new_just = _strip_opposite_winner_lines(new_just, n, left, right)
    return new_just.strip()


def _aligned_cited_turns(task: Task, slot: str, justification: str) -> list[int]:
    legal = _legal_turns(task, slot)
    cited = cited_turns(justification)
    aligned = [
        n
        for n in sorted(cited)
        if (not legal or n in legal) and _turn_has_model_text(task, slot, n)
    ]
    if legal and max(legal) >= 7:
        without_first = [n for n in aligned if n != 1]
        if without_first:
            aligned = without_first
    return aligned


def _sxs_needs_body_fix(sxs) -> bool:
    likert_side = form_likert_side(sxs.likert)
    just_side = _preference_side_simple(sxs.justification, sxs.left, sxs.right)
    if likert_side in {"left", "right"} and just_side == "tie":
        return True
    if (
        likert_side in {"left", "right"}
        and just_side in {"left", "right"}
        and likert_side != just_side
    ):
        return True
    return not likert_magnitude_ok(sxs.likert, sxs.justification)


def _family_hits(task: Task, patches: list[FieldPatch], families: set[str] | frozenset[str]):
    simulated = patched_task(task, [p for p in patches if not p.error])
    return [h for h in audit_task(simulated) if h.family in families]


def _paste_safety_sweep(
    task: Task, patches: list[FieldPatch], profile: ProjectProfile | None
) -> tuple[list[FieldPatch], list[str]]:
    """Final deterministic alignment so the four paste families do not ship dirty."""
    leftovers: list[str] = []
    out = list(patches)

    simulated = patched_task(task, [p for p in out if not p.error])
    for sxs in simulated.comparison_list():
        unexplained = inversion_is_unexplained(simulated, sxs)
        if not _sxs_needs_body_fix(sxs) and not unexplained:
            continue
        new_just = _rewrite_sxs_body(sxs.justification, sxs.likert, sxs.left, sxs.right)
        # 460 needs a turn-grounded inversion writeup from the LLM. Do not
        # prepend the canned "pairwise outcome outweighs the mean" sentence.
        orig = _comparison(task, sxs.pair_label)
        trial = _make_field_patch(
            kind="preference_justification",
            new_value=new_just,
            old_value=orig.justification if orig is not None else sxs.justification,
            notes="deterministic_sxs_body_matches_likert",
            comparison=sxs.pair_label,
            check_ids=(400, 460, 470),
            profile=profile,
        )
        trial_patches = _upsert_patch(out, trial)
        still = [
            h
            for h in _family_hits(
                task, trial_patches, {"sxs_scale", "sxs_magnitude", "inconsistent_ranking"}
            )
            if sxs.pair_label in h.detail
        ]
        if still:
            out = [
                p
                for p in out
                if not (
                    p.kind in {"preference_likert", "preference_justification"}
                    and p.comparison == sxs.pair_label
                )
            ]
            leftovers.append(
                f"SxS 1–7 / Inconsistent Ranking on {sxs.pair_label} could not be "
                "rewritten to match the Likert"
            )
            continue
        if new_just.strip() != (sxs.justification or "").strip():
            out = trial_patches

    simulated = patched_task(task, [p for p in out if not p.error])
    for dim in simulated.dimension_ratings:
        orig = _dim(task, dim.model, dim.dimension)
        if dimension_is_na(dim):
            current = list(dim.relevant_turns or [])
            if current:
                out = _upsert_patch(
                    out,
                    _make_field_patch(
                        kind="dimension_turns",
                        new_value=[],
                        old_value=list(orig.relevant_turns or []) if orig else current,
                        notes="na_turns_must_be_empty",
                        slot=dim.model,
                        dimension=dim.dimension,
                        check_ids=(310,),
                        profile=profile,
                    ),
                )
            continue
        selected = set()
        for item in dim.relevant_turns or []:
            try:
                selected.add(int(item))
            except (TypeError, ValueError):
                continue
        cited = cited_turns(dim.justification)
        if cited and selected and not (cited & selected):
            aligned = _aligned_cited_turns(task, dim.model, dim.justification)
            if not aligned:
                out = [
                    p
                    for p in out
                    if not (
                        p.kind == "dimension_turns"
                        and p.slot == dim.model
                        and p.dimension == dim.dimension
                    )
                ]
                leftovers.append(
                    f"Dimension Relevant Turns {dim.model}/{dim.dimension}: "
                    "selected disjoint from cited and no in-range assistant turns"
                )
                continue
            old = list(orig.relevant_turns or []) if orig else list(dim.relevant_turns or [])
            if not _same_turns(old, aligned):
                out = _upsert_patch(
                    out,
                    _make_field_patch(
                        kind="dimension_turns",
                        new_value=aligned,
                        old_value=old,
                        notes="justification_cites_disjoint_from_selected",
                        slot=dim.model,
                        dimension=dim.dimension,
                        check_ids=(310,),
                        profile=profile,
                    ),
                )

    drop_ids: set[int] = set()
    simulated = patched_task(task, [p for p in out if not p.error])
    for patch in out:
        if patch.kind != "dimension_turns" or patch.error:
            continue
        dim = _dim(simulated, patch.slot or "", patch.dimension or "")
        if dim is None or dimension_is_na(dim):
            continue
        new_turns: list[int] = []
        raw = patch.new_value if isinstance(patch.new_value, list) else []
        for item in raw:
            try:
                new_turns.append(int(item))
            except (TypeError, ValueError):
                continue
        cited = cited_turns(dim.justification)
        selected = set(new_turns)
        if not new_turns and empty_turns_ok(dim):
            continue
        if not new_turns or (cited and selected and not (cited & selected)):
            drop_ids.add(id(patch))
    if drop_ids:
        out = [p for p in out if id(p) not in drop_ids]

    still = _family_hits(task, out, _PASTE_FAMILIES)
    if still:
        dirty_dims = {
            (h.detail.split(":", 1)[0].strip() if "/" in h.detail else h.detail)
            for h in still
            if h.family == "dimension_turns"
        }
        if dirty_dims:
            out = [
                p
                for p in out
                if not (
                    p.kind == "dimension_turns"
                    and f"{p.slot}/{p.dimension}" in dirty_dims
                )
            ]
        dirty_pairs = {
            h.detail.split(":", 1)[0].strip()
            for h in still
            if h.family in {"sxs_scale", "sxs_magnitude", "inconsistent_ranking"}
        }
        if dirty_pairs:
            out = [
                p
                for p in out
                if not (
                    p.kind in {"preference_likert", "preference_justification"}
                    and (p.comparison or "") in dirty_pairs
                )
            ]
            leftovers.append(
                "SxS 1–7 / Inconsistent Ranking leftover after paste-safety sweep"
            )
        leftover_hits = _family_hits(task, out, _PASTE_FAMILIES)
        if leftover_hits:
            names = sorted({h.family for h in leftover_hits})
            leftovers.append(
                "paste-safety leftover: " + ", ".join(names)
            )
    return out, leftovers


def _forced_apply(task: Task, job: FieldJob) -> tuple[Any, str] | None:
    """Deterministic paste values that must not wait on the model."""
    if job.kind == "dimension_turns":
        dim = _dim(task, job.slot or "", job.dimension or "")
        if dim is None:
            return None
        if dimension_is_na(dim):
            return [], "na_turns_must_be_empty"
        cited = cited_turns(dim.justification)
        current: set[int] = set()
        for item in dim.relevant_turns or []:
            try:
                current.add(int(item))
            except (TypeError, ValueError):
                continue
        legal = _legal_turns(task, job.slot)

        def _usable(nums: set[int]) -> list[int]:
            replacement = sorted(
                n
                for n in nums
                if (not legal or n in legal)
                and _turn_has_model_text(task, job.slot, n)
            )
            if legal and max(legal) >= 7:
                without_first = [n for n in replacement if n != 1]
                if without_first:
                    replacement = without_first
            return replacement

        if empty_turns_ok(dim) and not current:
            return None
        if not current and cited:
            replacement = _usable(cited)
            if replacement:
                return replacement, "justification_names_turns_fill_empty"
        if cited and current and not (cited & current):
            replacement = _usable(cited)
            if replacement:
                return replacement, "justification_cites_disjoint_from_selected"
        return None
    if job.kind == "dimension_rating":
        dim = _dim(task, job.slot or "", job.dimension or "")
        if dim is not None and dimension_is_na(dim):
            return "not-applicable", "na_score_must_be_not_applicable"
    return None


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
        dim = (
            _dim(task, job.slot or "", job.dimension or "")
            if job.kind in {"dimension_turns", "dimension_rating"}
            else None
        )
        response = responses.get(fix_request_key(task.task_id, job.field_id))
        notes = ""
        force_align = False
        forced = _forced_apply(task, job)
        if forced is not None:
            new_value, notes = forced
            force_align = job.kind == "dimension_turns"
        elif not isinstance(response, dict) or "value" not in response:
            if job.optional:
                continue
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
        else:
            new_value = response.get("value")
            notes = str(response.get("notes") or "")
        if job.kind in {"criterion_turns", "dimension_turns", "key_turn_index"}:
            if job.kind == "dimension_turns" and dim is not None and dimension_is_na(dim):
                new_value = []
            else:
                filtered = _filter_turn_values(task, job.slot, new_value)
                if filtered is None:
                    continue
                new_value = filtered
        if job.kind in {"criterion_turns", "dimension_turns"}:
            legal = _legal_turns(task, job.slot)
            kept = _kept_turns(
                task,
                job,
                job.current_value,
                list(new_value) if isinstance(new_value, list) else [new_value],
                job.flagged_turns,
                legal,
                force_align=force_align,
            )
            if kept is None:
                continue
            new_value = kept
        if job.kind in {"criterion_turns", "dimension_turns"} and _same_turns(
            job.current_value, new_value
        ):
            continue
        guarded = _guard_new_value(task, job, new_value, notes)
        if guarded is None:
            continue
        new_value = guarded
        patches.append(
            FieldPatch(
                field_id=job.field_id,
                kind=job.kind,
                check_ids=list(job.check_ids),
                old_value=job.current_value,
                new_value=new_value,
                notes=notes,
                slot=job.slot,
                criterion_id=job.criterion_id,
                dimension=job.dimension,
                comparison=job.comparison,
                step_id=step_id,
                export_field=export_field,
            )
        )

    patches = _drop_inconsistent_pairs(task, patches)
    patches = _drop_unexplained_ranking_inversions(task, patches)
    patches, sweep_leftover = _paste_safety_sweep(task, patches, profile)
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
    if sweep_leftover:
        extra = "; ".join(sweep_leftover)
        leftover = f"{leftover}; {extra}" if leftover else extra
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


def _dimension_score_table(
    task: Task, responses_by_key: dict[str, Any], plan: FixPlan
) -> str:
    scores: dict[tuple[str, str], Any] = {
        (d.model, d.dimension): d.rating for d in task.dimension_ratings
    }
    for job in plan.jobs:
        if job.kind != "dimension_rating" or not job.slot or not job.dimension:
            continue
        data = responses_by_key.get(fix_request_key(task.task_id, job.field_id))
        if isinstance(data, dict) and "value" in data:
            scores[(job.slot, job.dimension)] = data["value"]
    lines = []
    for (slot, dimension), rating in sorted(scores.items()):
        lines.append(f"- Model {slot} / {dimension}: {rating}")
    return "\n".join(lines)


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
                    data = responses_by_key.get(fix_request_key(task.task_id, other.field_id))
                    if isinstance(data, dict) and "value" in data:
                        paired_value = data["value"]
                    break
            wave2_reqs.append(
                build_fix_request(
                    task,
                    job,
                    policy,
                    paired_corrected_value=paired_value,
                    dimension_score_table=(
                        _dimension_score_table(task, responses_by_key, plan)
                        if job.kind == "preference_justification"
                        else None
                    ),
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
    ap.add_argument(
        "--relabel-c-to-b",
        default="",
        metavar="FILE",
        help="task IDs (one per line) whose C trajectory is GPT: promote C to B, "
             "keep the AC ranking pair rewritten as A-vs-B, and drop Claude-as-B "
             "plus the simplified trajectory so they are not hydrated. Same "
             "post-ingest overlay as honeybee_qc.cli; gated on this flag so a "
             "D-only audit is not narrowed to A-vs-B",
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
        # Gated on this flag so a D-only run still sees DA/DB/DC.
        for task in tasks:
            narrow_to_ab_only(task)
        print(
            "relabel-c-to-b: every task narrowed to A-vs-B; C/S not hydrated",
            file=sys.stderr,
        )

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
    invariant_report = audit_fix_results(tasks, results)
    write_invariant_report(Path(args.out_dir), invariant_report)
    print(_summarize(results), file=sys.stderr)
    print(invariant_report.summary(), file=sys.stderr)
    print(
        "subagent checklist (run on every sheet):\n"
        "  python3 -m honeybee_qc.fixer_invariants --print-subagent-prompt",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
