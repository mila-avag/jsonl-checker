"""Fill missing 1-5 dimension ratings for one slot.

Used when a taxonomy revision added a product (Model D) and the dimension-
rating step was not backfilled on every task. Scores and justifications are
written for a reviewer to paste; they are not fed back into check 300, which
would be grading our own numbers.

Justifications are asked for in a contributor voice: short, specific, no
em dashes, no auditor-speak. After fill, the ratings are attached to the
task as if the contributor had filed them, so Dimension Correctness /
Relevant Turns / Justifications can run against them. Check 300 still
re-rates blind under a different cache key, so it is not scoring the fill
call against itself.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from .config import DEFAULT_POLICY, Policy
from .context import render_conversation
from .ingest import DIMENSION_FIELDS
from .llm import ModelClient, ModelRequest, run_requests
from .models import DimensionRating, ModelSlot, Task
from .rating_prompts import DIMENSION_GUIDANCE
from .taxonomies import COLLECTED_DIMENSIONS, RATING_DIMENSIONS

FILL_SYSTEM_PROMPT = (
    "You are a Scale annotator filling Honeybee dimension ratings for one model. "
    "Write the way a tired-but-careful person writes in a form: contractions are "
    "fine, sentences can start with And or But, and you can say I. Name the "
    "model (Model A, Model B, Model C) or the user. Never they unless the last "
    "sentence already named the model. "
    "Do not sound like a rubric or an audit report. Do not use em dashes or "
    "en dashes. Do not write moreover, furthermore, overall, notably, or "
    "it is important to note. Two or three sentences. Name the turns you "
    "actually used, like turn 4 or around turn 12. If a dimension does not "
    "apply, say so plainly and set not_applicable."
)

FILL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rating": {"type": ["integer", "null"]},
        "not_applicable": {"type": "boolean"},
        "justification": {"type": "string"},
        "relevant_turns": {"type": "array", "items": {"type": "integer"}},
        "cannot_determine": {"type": "boolean"},
    },
    "required": [
        "rating",
        "not_applicable",
        "justification",
        "relevant_turns",
        "cannot_determine",
    ],
    "additionalProperties": False,
}

_DASH_RE = re.compile(r"[\u2012\u2013\u2014\u2015]+")

DIMENSION_TO_PREFIX = {dimension: prefix for prefix, dimension in DIMENSION_FIELDS.items()}

FILL_DIMENSIONS: tuple[str, ...] = tuple(
    d for d in (*RATING_DIMENSIONS, "Collaboration quality") if d in COLLECTED_DIMENSIONS
)


def _humanize(text: str) -> str:
    cleaned = _DASH_RE.sub(",", (text or "").strip())
    cleaned = cleaned.replace(" -- ", ", ").replace(" - ", ", ")
    return cleaned


def _turns_payload(turns: list[int]) -> list[dict]:
    out = []
    for turn in turns:
        if not isinstance(turn, int) or turn <= 0:
            continue
        out.append(
            {
                "fullText": f"Turn {turn}",
                "parts": [{"id": "turn_number", "value": str(turn)}],
            }
        )
    return out


def missing_dim_slots(task: Task, slot: ModelSlot) -> bool:
    return not any(d.model == slot for d in task.dimension_ratings)


def build_fill_prompt(task: Task, dimension: str, slot: ModelSlot, policy: Policy) -> str:
    sub = task.submission(slot)
    if sub is None:
        raise ValueError(f"task {task.task_id} has no submission for slot {slot}")
    conversation, _profile = render_conversation(sub, policy)
    lo, hi = policy.dimension_rating_scale
    return f"""## The user's goal

{task.user_goal or task.seeded_prompt or '(not stated)'}

## Conversation with Model {slot}

{conversation}

## Dimension

{dimension}: {DIMENSION_GUIDANCE.get(dimension, '')}

## What to fill in

Score Model {slot} on this dimension, {lo} (worst) to {hi} (best).

Write `justification` as 2-3 sentences a person would type into this form.
No em dashes. No auditor voice. Point at real turns.

If the dimension does not apply (memory when nothing carries across turns,
connectors when none were used), set not_applicable true, rating null, and
say that in the justification. If the transcript cannot support a score,
set cannot_determine true.

Return JSON matching the schema.
"""


def fill_missing_dims(
    tasks: list[Task],
    client: ModelClient,
    slot: ModelSlot = "D",
    policy: Policy = DEFAULT_POLICY,
    workers: int = 32,
) -> list[dict]:
    """Generate scores + justifications for tasks missing this slot's 1-5s."""
    requests: list[ModelRequest] = []
    wanted: list[Task] = []
    for task in tasks:
        if not missing_dim_slots(task, slot):
            continue
        if task.submission(slot) is None:
            continue
        wanted.append(task)
        for dimension in FILL_DIMENSIONS:
            requests.append(
                ModelRequest(
                    key=f"{task.task_id}::fill_dim::{slot}::{dimension}",
                    prompt=build_fill_prompt(task, dimension, slot, policy),
                    schema=FILL_SCHEMA,
                    system=FILL_SYSTEM_PROMPT,
                    metadata={
                        "task_id": task.task_id,
                        "slot": slot,
                        "dimension": dimension,
                    },
                )
            )
    if not requests:
        return []
    responses = run_requests(client, requests, workers=workers)
    by_key = {r.key: r for r in responses}
    rows: list[dict] = []
    suffix = slot.lower()
    for task in wanted:
        row: dict = {"task_id": task.task_id, "slot": slot, "generated": "yes"}
        notes: list[str] = []
        for dimension in FILL_DIMENSIONS:
            prefix = DIMENSION_TO_PREFIX.get(dimension)
            if not prefix:
                continue
            key = f"{task.task_id}::fill_dim::{slot}::{dimension}"
            response = by_key.get(key)
            data = response.data if response and response.ok else None
            if not isinstance(data, dict) or data.get("cannot_determine"):
                notes.append(dimension)
                row[f"{prefix}_{suffix}"] = ""
                row[f"{prefix}_justif_{suffix}"] = ""
                row[f"{prefix}_turns_{suffix}"] = ""
                continue
            na = bool(data.get("not_applicable"))
            rating = data.get("rating")
            justif = _humanize(str(data.get("justification") or ""))
            turns = [
                int(t)
                for t in (data.get("relevant_turns") or [])
                if isinstance(t, int) or str(t).isdigit()
            ]
            row[f"{prefix}_{suffix}"] = "" if na else ("" if rating is None else str(int(rating)))
            row[f"{prefix}_justif_{suffix}"] = justif
            row[f"{prefix}_turns_{suffix}"] = json.dumps(
                _turns_payload(turns), ensure_ascii=False
            )
        row["unfilled_dimensions"] = "; ".join(notes)
        rows.append(row)
    apply_fill_to_tasks(tasks, rows, slot)
    return rows


def _turns_from_cell(raw: str) -> list[int]:
    if not (raw or "").strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    turns: list[int] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if isinstance(part, dict) and str(part.get("value") or "").isdigit():
                turns.append(int(part["value"]))
    return turns


def ratings_from_fill_row(row: dict, slot: ModelSlot) -> list[DimensionRating]:
    suffix = slot.lower()
    ratings: list[DimensionRating] = []
    for prefix, dimension in DIMENSION_FIELDS.items():
        score_raw = str(row.get(f"{prefix}_{suffix}") or "").strip()
        justif = str(row.get(f"{prefix}_justif_{suffix}") or "").strip()
        turns = _turns_from_cell(str(row.get(f"{prefix}_turns_{suffix}") or ""))
        if not score_raw and not justif:
            continue
        na = score_raw.lower().replace("_", "-") in {"not-applicable", "n/a", "na"}
        rating = None
        if score_raw.lstrip("-").isdigit():
            rating = int(score_raw)
        ratings.append(
            DimensionRating(
                model=slot,
                dimension=dimension,
                rating=None if na else rating,
                not_applicable=na,
                justification=justif,
                relevant_turns=turns,
            )
        )
    return ratings


def apply_fill_to_tasks(
    tasks: list[Task], rows: list[dict], slot: ModelSlot = "D"
) -> int:
    """Treat generated 1-5s as the contributor's filed ratings for this slot."""
    by_id = {str(row.get("task_id") or ""): row for row in rows}
    attached = 0
    for task in tasks:
        row = by_id.get(task.task_id)
        if not row:
            continue
        existing = {
            d.dimension for d in task.dimension_ratings if d.model == slot
        }
        added = 0
        for rating in ratings_from_fill_row(row, slot):
            if rating.dimension in existing:
                continue
            task.dimension_ratings.append(rating)
            existing.add(rating.dimension)
            added += 1
        if added:
            attached += 1
    return attached


def write_fill_csv(path: Path, rows: list[dict], slot: ModelSlot = "D") -> None:
    suffix = slot.lower()
    columns = ["task_id", "slot", "generated", "unfilled_dimensions"]
    for prefix in DIMENSION_FIELDS:
        columns.extend(
            (
                f"{prefix}_{suffix}",
                f"{prefix}_justif_{suffix}",
                f"{prefix}_turns_{suffix}",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_fill_csv(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
