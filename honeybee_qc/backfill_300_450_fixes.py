"""Backfill script for checks 300 (Dimension Rating Correctness) and 450
(Dimension Rating Justifications), for the subset of each that lives in the
per-model `step-ResponseTextCollection-*` dimension-rating steps.

Why 300 and 450 share one CSV: both read and write the exact same annotation
step per model (`<dim>_<a|b>` for 300's score, `<dim>_justif_<a|b>` for 450's
justification text) -- `step-ResponseTextCollection-cb06a191fca3` for model A,
`step-ResponseTextCollection-2a199aa081bb` for model B, confirmed against the
raw `taskattempts` rows for this project rather than assumed from the field
prefixes alone. A backfill card only ever touches one step at a time, so
whenever two checks land on the same step their fixes belong in the same CSV;
see `backfill_400_ranking_fixes.py` for the separate `ranking` justification
and Likert step, which is a genuinely different step
(`step-QuantitativeModelResponseSelector-cd151dbf1b55`) even though check 450
also produces a finding for it (`item_id == "ranking"`).

What each check's fix actually is:

- **300** is a pure lookup, not a judgment call. `evaluate_check_300` already
  disagrees with the contributor's rating using an independently blind-rated
  `auditor_rating` for the exact same (model, dimension) -- that number is
  sitting in the cached rating-stage response already on disk. The fix is
  just "write the auditor's number", nothing is inferred or guessed.
  EXCEPT: `na_mismatch` items (auditor says a dimension does not apply, or
  vice versa) are not a value substitution. Most of these are on
  "Tool & connector reliability", which -- unlike "Memory & personalization"
  -- has no `_applicable_` field in the annotation schema at all, so there is
  no field to write "N/A" into even if we wanted to. Those items (and,
  playing it safe, the whole task's row in this CSV) are left unfixable
  rather than half-fixing a task and leaving its other, real disagreement
  silently unaddressed.

- **450** requires actually rewriting prose: the cached informed-stage
  response names the specific defect(s) in the contributor's justification
  (wrong turns cited, unsupported/inaccurate claims, generic text, etc.) but
  the fix -- a corrected justification that keeps the same rating but no
  longer trips those specific conditions -- has to be generated, grounded in
  the same conversation the auditor read.

Contract (per the user): every fix this script writes is either a real,
non-blank value, or the task is not written at all for this CSV --
"unfixable_reason" is filled and every fix column for that task's row stays
blank. There is no low-confidence middle ground here (contrast with the 310
backfill's fallback path): a wrong rating and a wrong justification are much
higher-stakes than a turn citation, so an item this script cannot resolve
with real, cached evidence marks the whole task unfixable rather than
guessing.

Usage:
    python3 -m honeybee_qc.backfill_300_450_fixes \
        --report honeybee_qc/audit_runs/.../chunks/chunk_1000fix_report.json \
        --tasks-csv honeybee_qc/audit_runs/.../pulled_30.csv \
        --cache-db honeybee_qc/audit_runs/.../cache.db \
        --snapshot-dir honeybee_qc/audit_runs/.../snapshots \
        --out honeybee_qc/audit_runs/.../300_450_fixes_backfill.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_POLICY, POLICY_VERSION, Policy
from .context import render_conversation
from .hydrate import hydrate_conversations
from .informed_stages import build_informed_requests
from .ingest import DIMENSION_FIELDS, load_taskattempts_csv
from .llm import ModelRequest, ModelResponse, build_client, run_requests
from .models import ModelSubmission, Task
from .rating_stages import build_rating_requests

DIMENSION_TO_PREFIX = {dimension: prefix for prefix, dimension in DIMENSION_FIELDS.items()}

JUSTIF_FIX_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "corrected_justification": {"type": "string", "minLength": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["corrected_justification", "reasoning"],
    "additionalProperties": False,
}

EMPTY_RETRY_SUFFIX = """

## One more thing

A prior attempt at this came back with an empty `corrected_justification`. That
is not usable: there is always something true and specific to say about how the
model actually performed on this dimension, even if the honest answer is a
shorter, more modest claim than the original. Write a real, non-empty
justification, or if you truly believe none of the flagged defects are real,
say so in `reasoning` and still return the *original* text unchanged (never an
empty string).
"""


def _cache_lookup(conn: sqlite3.Connection | None, request_key: str) -> dict | None:
    """Latest cached response payload for a request key, or None."""
    if conn is None:
        return None
    row = conn.execute(
        "SELECT payload FROM response_cache WHERE request_key = ? ORDER BY created_at DESC LIMIT 1",
        (request_key,),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def extract_300_items(report_task: dict) -> tuple[dict[tuple[str, str], list[str]], list[str]]:
    """`{(model, dimension): reasons}` for fixable items, and a list of
    `(model, dimension)` labels that carry `na_mismatch` and cannot be fixed
    by substitution -- see module docstring."""
    fixable: dict[tuple[str, str], list[str]] = {}
    blocking: list[str] = []
    for check in report_task.get("checks", []):
        if check.get("check_id") != 300:
            continue
        reasons_by_item = ((check.get("measurement") or {}).get("counts") or {}).get(
            "reasons_by_item"
        ) or {}
        for item_key, reasons in reasons_by_item.items():
            # item_key is "<model>::<dimension>" for check 300 (no check-id prefix,
            # unlike 310/280's namespaced keys -- confirmed against the report).
            if "na_mismatch" in reasons:
                blocking.append(item_key)
                continue
            model, dimension = item_key.split("::", 1)
            fixable[(model, dimension)] = reasons
    return fixable, blocking


def extract_450_items(report_task: dict) -> dict[str, list[str]]:
    """`{item_id: conditions}` for every check-450 item flagged, item_id being
    e.g. "B::Collaboration quality" or "ranking"."""
    out: dict[str, list[str]] = {}
    for check in report_task.get("checks", []):
        if check.get("check_id") != 450:
            continue
        conditions_by_item = ((check.get("measurement") or {}).get("counts") or {}).get(
            "conditions_by_item"
        ) or {}
        for item_id, conditions in conditions_by_item.items():
            out[item_id] = conditions
    return out


def _defect_summary(judgment: dict) -> str:
    lines = []
    if judgment.get("is_generic"):
        lines.append("- Generic: reads as boilerplate that could apply to almost any response.")
    if judgment.get("is_skewed"):
        lines.append("- Skewed: overstates one side, ignoring a real weakness or strength.")
    for key, label in (
        ("contradicts_verdict_claims", "claim(s) that contradict the rating/verdict they defend"),
        ("inaccurate_primary_claims", "primary claim(s) that are factually inaccurate"),
        ("inaccurate_secondary_claims", "secondary claim(s) that are factually inaccurate"),
        ("unsupported_claims", "claim(s) made with no support in the conversation"),
        ("inaccurate_evidence", "cited piece(s) of evidence that are themselves wrong"),
        ("misconstrued_evidence", "piece(s) of evidence that are real but misread"),
    ):
        n = int(judgment.get(key) or 0)
        if n:
            lines.append(f"- {n} {label}.")
    quotes = judgment.get("contradicting_quotes") or []
    if quotes:
        lines.append("Quoted spans the audit flagged as contradicting or unsupported:")
        lines.extend(f'  "{q}"' for q in quotes)
    return "\n".join(lines) or "(the audit flagged this justification but recorded no specific counts)"


def build_dimension_justif_fix_prompt(
    sub: ModelSubmission, dimension: str, rating: int | None, current: str, judgment: dict, policy: Policy
) -> str:
    conversation, _ = render_conversation(sub, policy)
    return f"""## Conversation with Model {sub.model}

{conversation}

## The contributor's justification for this one dimension

Dimension: {dimension}
Rating: {rating if rating is not None else "N/A"}
Current justification: {" ".join((current or "(empty)").split())}

## What an independent audit found wrong with it

{_defect_summary(judgment)}

## Your task

Rewrite this justification so it no longer has the defect(s) above, while
defending the *same* rating ({rating if rating is not None else "N/A"}) using
only what the conversation above actually shows. Keep it specific and grounded
-- cite concrete behavior, not generic praise or criticism. Return the full
replacement text in `corrected_justification`; it must never be empty. Explain
what you changed and why in `reasoning`.
"""


def build_ranking_justif_fix_prompt(task: Task, current: str, judgment: dict, policy: Policy) -> str:
    from .context import render_comparison

    comparison = render_comparison(task.model_a, task.model_b, policy)[0]
    return f"""## The two conversations

{comparison}

## The contributor's comparison

Likert rating: {task.sxs.likert if task.sxs.likert is not None else "(none)"}
Preferred model index: {task.sxs.winner_index if task.sxs.winner_index is not None else "(none)"}
Current justification: {" ".join((current or "(empty)").split())}

## What an independent audit found wrong with it

{_defect_summary(judgment)}

## Your task

Rewrite this ranking justification so it no longer has the defect(s) above,
while still defending the *same* preference (same Likert direction and winner)
using only what the two conversations above actually show. Return the full
replacement text in `corrected_justification`; it must never be empty. Explain
what you changed and why in `reasoning`.
"""


def resolve_empty_justif_responses(
    client, keys: list[str], prompts: list[str], responses: list[ModelResponse], workers: int
) -> list[ModelResponse]:
    retry_indices = [
        i
        for i, r in enumerate(responses)
        if r.ok and not (r.data or {}).get("corrected_justification", "").strip()
    ]
    if not retry_indices:
        return responses
    retry_requests = [
        ModelRequest(
            key=f"{keys[i]}::retry",
            prompt=prompts[i] + EMPTY_RETRY_SUFFIX,
            schema=JUSTIF_FIX_SCHEMA,
        )
        for i in retry_indices
    ]
    retry_responses = run_requests(client, retry_requests, workers=workers)
    out = list(responses)
    for i, resp in zip(retry_indices, retry_responses):
        out[i] = resp
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--task-ids",
        default=None,
        help="comma-separated task ids; default is every task whose fail_checks "
        "includes 300, plus any task with a non-ranking item in check 450",
    )
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hydrate-workers", type=int, default=8)
    args = ap.parse_args(argv)

    policy = DEFAULT_POLICY
    if args.snapshot_dir:
        policy = replace(policy, snapshot_dir=args.snapshot_dir)

    report = json.loads(Path(args.report).read_text())
    report_by_task = {t["task_id"]: t for t in report["tasks"]}

    if args.task_ids:
        target_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    else:
        target_ids = []
        for t in report["tasks"]:
            fails300 = 300 in (t.get("fail_checks") or [])
            has_dim_450 = any(
                item_id != "ranking"
                for c in t.get("checks", [])
                if c.get("check_id") == 450
                for item_id in (
                    ((c.get("measurement") or {}).get("counts") or {}).get("conditions_by_item") or {}
                )
            )
            if fails300 or has_dim_450:
                target_ids.append(t["task_id"])

    print(f"backfilling checks 300+450(dimension) for {len(target_ids)} task(s)", file=sys.stderr)

    ingested, _ingest_errors = load_taskattempts_csv(args.tasks_csv)
    hydration = hydrate_conversations(ingested, policy, workers=args.hydrate_workers)
    print(
        f"hydrated {hydration.hydrated}/{hydration.submissions} submissions for fix generation",
        file=sys.stderr,
    )
    task_by_id: dict[str, Task] = {t.task_id: t for t in ingested}

    conn = sqlite3.connect(args.cache_db) if args.cache_db and Path(args.cache_db).exists() else None
    client, _cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version=f"{POLICY_VERSION}+300-450-backfill-v1",
    )

    columns = (
        ["task_id", "unfixable_reason"]
        + [f"{prefix}_{s}" for prefix in DIMENSION_FIELDS for s in ("a", "b")]
        + [f"{prefix}_justif_{s}" for prefix in DIMENSION_FIELDS for s in ("a", "b")]
    )
    rows: dict[str, dict[str, str]] = {}
    unfixable: dict[str, str] = {}

    # 450 rewrite requests are batched across every task/item up front so the
    # single LLM round-trip covers the whole run, not one call per task.
    pending_450: list[tuple[str, str, str]] = []  # (task_id, item_id, column)
    keys_450: list[str] = []
    prompts_450: list[str] = []

    for task_id in target_ids:
        report_task = report_by_task.get(task_id)
        task = task_by_id.get(task_id)
        if report_task is None or task is None:
            unfixable[task_id] = "task not found in report/tasks-csv"
            continue
        row: dict[str, str] = {}
        reason_parts: list[str] = []

        # --- Check 300: deterministic lookup of the auditor's own rating ---
        fixable_300, blocking_300 = extract_300_items(report_task)
        if blocking_300:
            reason_parts.append(
                "check 300 na_mismatch on "
                + ", ".join(sorted(blocking_300))
                + " -- the auditor found this dimension inapplicable (or vice "
                "versa), which is not a value substitution; most of these "
                "dimensions (e.g. Tool & connector reliability) have no "
                "`_applicable_` field in the schema to record N/A in at all"
            )
        rating_requests = build_rating_requests(task, policy, guard=False)
        rating_by_key = {r.key: r for r in rating_requests if r.metadata.get("check_id") == 300}
        for (model, dimension), _reasons in fixable_300.items():
            match = next(
                (
                    r
                    for r in rating_by_key.values()
                    if r.metadata.get("model") == model and r.metadata.get("dimension") == dimension
                ),
                None,
            )
            label = f"300::{model}::{dimension}"
            if match is None:
                reason_parts.append(f"{label}: could not rebuild the original rating request")
                continue
            cached = _cache_lookup(conn, match.key)
            if cached is None:
                reason_parts.append(f"{label}: no cached blind-rating response on file")
                continue
            if cached.get("cannot_determine"):
                reason_parts.append(f"{label}: auditor abstained on this dimension (cannot_determine)")
                continue
            if cached.get("not_applicable"):
                reason_parts.append(f"{label}: auditor rating is itself N/A, contradicts a rating_delta finding")
                continue
            rating = cached.get("rating")
            lo, hi = policy.dimension_rating_scale
            if not isinstance(rating, int) or not (lo <= rating <= hi):
                reason_parts.append(f"{label}: cached rating {rating!r} is not a valid {lo}-{hi} score")
                continue
            prefix = DIMENSION_TO_PREFIX[dimension]
            row[f"{prefix}_{model.lower()}"] = str(rating)

        # --- Check 450 (dimension-level only; "ranking" belongs to the other CSV) ---
        items_450 = extract_450_items(report_task)
        informed_requests = build_informed_requests(task, policy)
        justif_by_model = {
            r.metadata.get("model"): r
            for r in informed_requests
            if r.metadata.get("check_id") == 450 and r.metadata.get("model") is not None
        }
        for item_id, _conditions in items_450.items():
            if item_id == "ranking":
                continue
            model, dimension = item_id.split("::", 1)
            req = justif_by_model.get(model)
            label = f"450::{item_id}"
            if req is None:
                reason_parts.append(f"{label}: could not rebuild the original justification request")
                continue
            cached = _cache_lookup(conn, req.key)
            if cached is None:
                reason_parts.append(f"{label}: no cached informed-stage response on file")
                continue
            judgment = next(
                (j for j in cached.get("justifications", []) if j.get("item_id") == item_id), None
            )
            if judgment is None:
                reason_parts.append(f"{label}: item not found in the cached justifications list")
                continue
            if judgment.get("unverifiable"):
                reason_parts.append(f"{label}: audit itself could not verify this justification -- nothing to rewrite against")
                continue
            sub = task.model_a if model == "A" else task.model_b
            drating = next(
                (d for d in task.dimension_ratings if d.model == model and d.dimension == dimension), None
            )
            if sub is None or drating is None or not sub.assistant_turns():
                reason_parts.append(f"{label}: submission not hydrated (model replies unavailable) -- cannot ground a rewrite")
                continue
            prompt = build_dimension_justif_fix_prompt(
                sub, dimension, drating.rating, drating.justification, judgment, policy
            )
            prefix = DIMENSION_TO_PREFIX[dimension]
            column = f"{prefix}_justif_{model.lower()}"
            key = f"{task_id}::450fix::{item_id}"
            keys_450.append(key)
            prompts_450.append(prompt)
            pending_450.append((task_id, item_id, column))

        rows[task_id] = row
        if reason_parts:
            unfixable[task_id] = "; ".join(reason_parts)

    print(f"sending {len(pending_450)} justification-rewrite request(s)", file=sys.stderr)
    requests_450 = [ModelRequest(key=k, prompt=p, schema=JUSTIF_FIX_SCHEMA) for k, p in zip(keys_450, prompts_450)]
    responses_450 = run_requests(client, requests_450, workers=args.workers)
    responses_450 = resolve_empty_justif_responses(client, keys_450, prompts_450, responses_450, args.workers)

    for (task_id, item_id, column), response in zip(pending_450, responses_450):
        label = f"450::{item_id}"
        if task_id not in rows:
            continue
        if not response.ok:
            unfixable.setdefault(task_id, "")
            unfixable[task_id] = (unfixable[task_id] + "; " if unfixable[task_id] else "") + f"{label}: {response.error}"
            continue
        text = str(response.data.get("corrected_justification") or "").strip()
        if not text:
            unfixable.setdefault(task_id, "")
            unfixable[task_id] = (
                unfixable[task_id] + "; " if unfixable[task_id] else ""
            ) + f"{label}: rewrite came back empty even after retry"
            continue
        rows[task_id][column] = text

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for task_id in target_ids:
            if task_id in unfixable:
                writer.writerow({"task_id": task_id, "unfixable_reason": unfixable[task_id]})
                continue
            out_row = {"task_id": task_id, "unfixable_reason": ""}
            out_row.update(rows.get(task_id, {}))
            writer.writerow(out_row)

    fixable_count = sum(1 for t in target_ids if t not in unfixable)
    print(
        f"wrote {out_path}: {fixable_count} fixable, {len(unfixable)} unfixable of {len(target_ids)}",
        file=sys.stderr,
    )
    for tid, reason in unfixable.items():
        print(f"  UNFIXABLE {tid}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
