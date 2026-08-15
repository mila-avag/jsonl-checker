"""Backfill script for check 310 (Dimension Ratings / Relevant Turns) fails.

For every (model, dimension) item an audit report flagged as `incorrect_turns`
(or `missing_turn`) under check 310, this re-reads the hydrated conversation and
the contributor's own justification, and asks the model which turn number(s)
actually show that specific claim. It emits a CSV shaped for a backfill card:
one row per task, one column per `<dimension>_turns_<a|b>` annotation key,
containing the corrected turns array in the exact JSON shape the annotation
field uses -- blank wherever that item needed no fix.

Contract: every corrected turns array this script writes is non-empty. This is
deliberate, not a style choice -- `missing_turn_findings` in `informed_stages.py`
scores a citation-less item as its own defect, on the same footing as (and by
default counted the same as) an incorrectly-cited one: the spec treats an absent
citation as at least as bad as a wrong one, never better. A backfill whose fix
resolves to an empty array would replace one instance of that exact defect with
another, so:

  - The primary path (conversation, including the model's replies, was
    hydrated) asks the model for a non-empty `corrected_turns` array and
    retries once with a stronger instruction if it ever comes back empty --
    every dimension rating is *about* something the model did, so there is
    always at least one turn to point at.
  - The fallback path (the model's replies were never fetched for this
    submission, so there is nothing to verify a claim against) does not sit
    out and leave the cell blank. It re-asks using only the user's side of the
    conversation -- which is a genuinely different, weaker question ("where did
    the user raise the point this justification is about", not "where did the
    model do the thing") -- and marks the result as low-confidence directly in
    the cell's `fullText` (e.g. "Turn 5 (low confidence -- based on user turns
    only, model replies unavailable)") so a reviewer applying the backfill can
    tell at a glance which citations were verified against the model's actual
    behavior and which are a best-effort stand-in. Before taking this path the
    script does not just accept "not hydrated" at face value -- see
    `_best_effort_conversation`, which tries the submission's own
    `transcript_pdf` export (with OCR enabled) as a second source before
    falling back to user-only turns, and discards it if the extracted text
    doesn't plausibly belong to this conversation.

Reusable: pass `--task-ids` to target any task set; the default is every task
in `--report` whose `fail_checks` includes 310.

Usage:
    python3 -m honeybee_qc.backfill_310_fixes \
        --report honeybee_qc/audit_runs/.../chunks/chunk_fuzzy95_report.json \
        --tasks-csv honeybee_qc/audit_runs/.../pulled_30.csv \
        --cache-db honeybee_qc/audit_runs/.../cache.db \
        --snapshot-dir honeybee_qc/audit_runs/.../snapshots \
        --out honeybee_qc/audit_runs/.../310_fixes_backfill.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_POLICY, POLICY_VERSION, Policy
from .context import render_conversation
from .hydrate import hydrate_conversations
from .ingest import DIMENSION_FIELDS, load_taskattempts_csv
from .llm import ModelRequest, ModelResponse, build_client, run_requests
from .models import DimensionRating, ModelSubmission, Task
from .sources import read_pdf_transcript

DIMENSION_TO_PREFIX = {dimension: prefix for prefix, dimension in DIMENSION_FIELDS.items()}

LOW_CONFIDENCE_NOTE = "low confidence -- based on user turns only, model replies unavailable"

# `minItems` is a belt on top of the retry below: some backends validate the
# schema before ever showing the response to us, which turns "empty" into a
# request-level failure (caught and retried the same as any other) instead of
# a schema-valid answer we would otherwise have to inspect and reject by hand.
FIX_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "corrected_turns": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["corrected_turns", "reasoning"],
    "additionalProperties": False,
}

EMPTY_RETRY_SUFFIX = """

## One more thing

A prior attempt at this exact question came back with an empty `corrected_turns`
list. That is not an acceptable answer here: every dimension rating is about
something specific the model (or, if noted above, the user) did, so there is
always at least one turn in the conversation that comes closest to showing it,
even if imperfectly. Leaving the citation empty is its own separate defect --
worse than an imperfect-but-present one, since it gives a reviewer nothing to
check at all. Pick the single turn that comes closest to the claim and explain
the imperfection in `reasoning` rather than returning no turns.
"""


def _turns_column(dimension: str, model: str) -> str:
    prefix = DIMENSION_TO_PREFIX[dimension]
    return f"{prefix}_turns_{model.lower()}"


def annotation_turns_json(turn_numbers: list[int], note: str | None = None) -> str:
    """The exact `*_turns_*` annotation shape: a list of turn-number parts.

    `note`, when given, is appended to `fullText` only -- `parts`/`value` stay a
    clean turn number for anything downstream that parses the annotation
    programmatically (`ingest.py`'s `_turn_numbers` only ever reads `parts`).
    """
    return json.dumps(
        [
            {
                "parts": [{"id": "turn_number", "value": str(n)}],
                "fullText": f"Turn {n}" + (f" ({note})" if note else ""),
            }
            for n in turn_numbers
        ]
    )


def build_310_fix_prompt(
    sub: ModelSubmission,
    drating: DimensionRating,
    wrong_turns: list[int],
    good_turns: list[int],
    policy: Policy,
) -> str:
    conversation, _ = render_conversation(sub, policy)
    cited = sorted(set(drating.relevant_turns))
    wrong = sorted(set(wrong_turns))
    good = sorted(set(good_turns))
    return f"""## Conversation with Model {sub.model}

{conversation}

## The contributor's rating for this one dimension

Dimension: {drating.dimension}
Rating: {"N/A" if drating.not_applicable else drating.rating}
Justification: {" ".join((drating.justification or "(empty)").split())}

## What an independent audit found

The contributor cited turn(s) {cited or "(none)"} as evidence for this justification.
The audit determined turn(s) {wrong or "(none)"} do NOT actually show the specific
claim the justification makes -- they may be topically related without showing the
exact behavior described, or belong to the other model's conversation. Turn(s)
{good or "(none)"} were not flagged and should be treated as already correct.

## Your task

Read the justification's specific claim and the conversation above, then find which
turn number(s) actually show that specific claim -- not merely the right topic, the
exact behavior described. Return the complete corrected list of turn numbers to cite
for this dimension in `corrected_turns`: keep whichever already-confirmed turns still
apply, drop ones that do not, and add whichever turn(s) you find that actually support
the claim. If no single turn fully supports the claim, cite whichever comes closest and
say so in `reasoning`. Never invent a turn number that is not present in the
conversation above. `corrected_turns` must never be empty -- if you are torn between
several imperfect options, pick the closest one and explain the gap in `reasoning`.
"""


def build_user_only_fallback_prompt(sub: ModelSubmission, drating: DimensionRating, policy: Policy) -> str:
    """Weaker fallback question for a submission whose model replies never fetched.

    Cannot ask "where did the model do X" with no visibility into what the model
    did. Asks the honestly answerable question instead: which user turn raised
    the point the justification is about.
    """
    conversation, _ = render_conversation(sub, policy)
    return f"""## Conversation with Model {sub.model} (model's own replies were never fetched -- user turns only)

{conversation}

## The contributor's rating for this one dimension

Dimension: {drating.dimension}
Rating: {"N/A" if drating.not_applicable else drating.rating}
Justification: {" ".join((drating.justification or "(empty)").split())}

## Why you are being asked a different question here

Normally this check verifies a citation against what the *model* did. That is not
possible for this submission -- the model's replies were never fetched, so only the
user's side of the conversation is shown above. You cannot confirm or deny what the
model actually did.

## Your task

Identify which user turn(s) raise, request, or provide the context for the specific
point this justification is about -- i.e. where the user set up the situation the
justification is describing the model's handling of. This is a best-effort stand-in
for a real citation, not a verification of the model's behavior. Return the turn
number(s) in `corrected_turns`; it must never be empty -- if no user turn addresses
this precisely, pick the closest and most relevant one (turn 1 if truly nothing else
fits) and say so in `reasoning`. Never invent a turn number not present above.
"""


def resolve_empty_responses(
    client,
    keys: list[str],
    prompts: list[str],
    responses: list[ModelResponse],
    workers: int,
    schema: dict = FIX_SCHEMA,
    retry_suffix: str = EMPTY_RETRY_SUFFIX,
) -> list[ModelResponse]:
    """Batched retry for every response that came back schema-valid but empty.

    A schema-invalid/empty-array response from a strict backend already surfaces
    as `resp.ok is False`, which `RetryingClient` retries on its own -- this only
    covers the rarer case of a *valid* empty list, which is not on its own an
    error the lower layers see any reason to redo. Batched (not one-by-one) so a
    handful of empty answers in a large run cost one extra round of concurrency,
    not a serial tail.
    """
    retry_indices = [
        i for i, r in enumerate(responses) if r.ok and not (r.data or {}).get("corrected_turns")
    ]
    if not retry_indices:
        return responses
    retry_requests = [
        ModelRequest(key=f"{keys[i]}::retry", prompt=prompts[i] + retry_suffix, schema=schema)
        for i in retry_indices
    ]
    retry_responses = run_requests(client, retry_requests, workers=workers)
    out = list(responses)
    for i, resp in zip(retry_indices, retry_responses):
        out[i] = resp
    return out


def _best_effort_conversation(task: Task, sub: ModelSubmission, policy: Policy) -> bool:
    """Tries the submission's `transcript_pdf` export as a second hydration source.

    `hydrate_conversations` only ever tries `final_link`; a contributor's PDF
    transcript export is a real, independent source it never attempts. Enabled
    here with OCR on, since the export is sometimes a vector-only print with no
    text layer. Returns True and mutates `sub.conversation` only if the result
    plausibly belongs to this task -- checked by looking for a handful of the
    task's own prompt words in the extracted text, because this source has been
    observed to resolve to a byte-identical file shared with an unrelated task
    (a data pipeline artifact, not a transcript of this conversation at all),
    and feeding that in would be strictly worse than admitting the model's
    replies are unavailable.
    """
    if sub.assistant_turns() or not sub.transcript_pdf:
        return False
    ocr_policy = replace(policy, pdf_ocr_fallback=True)
    transcript = read_pdf_transcript(sub.transcript_pdf, ocr_policy)
    if transcript.error or not transcript.full_text.strip():
        return False
    prompt_words = {w.lower() for w in (task.seeded_prompt or task.user_goal or "").split() if len(w) > 4}
    text_lower = transcript.full_text.lower()
    overlap = sum(1 for w in prompt_words if w in text_lower)
    if prompt_words and overlap / len(prompt_words) < 0.05:
        return False
    from .context import conversation_from_transcript

    sub.conversation = conversation_from_transcript(transcript)
    return bool(sub.conversation)


def extract_310_items(report_task: dict) -> dict[tuple[str, str], list[int]]:
    """`{(model, dimension): [wrong turn numbers]}` for every item check 310 flagged."""
    out: dict[tuple[str, str], list[int]] = {}
    for check in report_task.get("checks", []):
        if check.get("check_id") != 310:
            continue
        counts = (check.get("measurement") or {}).get("counts") or {}
        turns_by_item = counts.get("turns_by_item") or {}
        reasons_by_item = counts.get("reasons_by_item") or {}
        for item_key, reasons in reasons_by_item.items():
            if "incorrect_turns" not in reasons and "missing_turn" not in reasons:
                continue
            # item_key is "310::<model>::<dimension>"
            _, model, dimension = item_key.split("::", 2)
            out[(model, dimension)] = list(turns_by_item.get(item_key, []))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="a report json with a top-level 'tasks' list")
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--task-ids",
        default=None,
        help="comma-separated task ids to restrict to; default is every task whose "
        "fail_checks includes 310",
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
        target_ids = [t["task_id"] for t in report["tasks"] if 310 in (t.get("fail_checks") or [])]

    print(f"backfilling check 310 for {len(target_ids)} task(s)", file=sys.stderr)

    ingested, _ingest_errors = load_taskattempts_csv(args.tasks_csv)
    hydration = hydrate_conversations(ingested, policy, workers=args.hydrate_workers)
    print(
        f"hydrated {hydration.hydrated}/{hydration.submissions} submissions for fix generation",
        file=sys.stderr,
    )
    for line in hydration.failures:
        print(f"  not fetched: {line}", file=sys.stderr)
    task_by_id: dict[str, Task] = {t.task_id: t for t in ingested}

    client, _cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version=f"{POLICY_VERSION}+310-backfill-v2",
    )

    # (task_id, model, dimension, is_fallback) -> built alongside the prompt so
    # the same index lines up across keys/prompts/responses.
    pending: list[tuple[str, str, str, bool]] = []
    keys: list[str] = []
    prompts: list[str] = []
    rescued_via_pdf: list[str] = []

    for task_id in target_ids:
        report_task = report_by_task.get(task_id)
        task = task_by_id.get(task_id)
        if report_task is None or task is None:
            print(f"  skip {task_id}: not found in report/tasks-csv", file=sys.stderr)
            continue
        items = extract_310_items(report_task)
        if not items:
            print(f"  skip {task_id}: no flagged 310 items in this report", file=sys.stderr)
            continue
        for (model, dimension), wrong_turns in items.items():
            sub = task.model_a if model == "A" else task.model_b if model == "B" else None
            drating = next(
                (d for d in task.dimension_ratings if d.model == model and d.dimension == dimension),
                None,
            )
            if sub is None or drating is None:
                print(
                    f"  skip {task_id}::{model}::{dimension}: submission or rating not found",
                    file=sys.stderr,
                )
                continue

            if not sub.assistant_turns():
                if _best_effort_conversation(task, sub, policy):
                    rescued_via_pdf.append(f"{task_id}::{model}::{dimension}")

            key = f"{task_id}::310fix::{model}::{dimension}"
            if sub.assistant_turns():
                good_turns = [t for t in drating.relevant_turns if t not in wrong_turns]
                prompt = build_310_fix_prompt(sub, drating, wrong_turns, good_turns, policy)
                is_fallback = False
            else:
                # Genuinely exhausted: final_link failed to parse, transcript_pdf
                # is absent/unusable/doesn't match this task, no other source
                # exists. Ask the honestly answerable question instead of
                # leaving the cell blank.
                prompt = build_user_only_fallback_prompt(sub, drating, policy)
                is_fallback = True

            keys.append(key)
            prompts.append(prompt)
            pending.append((task_id, model, dimension, is_fallback))

    print(f"sending {len(pending)} fix request(s)", file=sys.stderr)
    requests = [ModelRequest(key=k, prompt=p, schema=FIX_SCHEMA) for k, p in zip(keys, prompts)]
    responses = run_requests(client, requests, workers=args.workers)
    responses = resolve_empty_responses(client, keys, prompts, responses, workers=args.workers)

    rows: dict[str, dict[str, str]] = {tid: {} for tid in target_ids}
    still_empty: list[str] = []
    for (task_id, model, dimension, is_fallback), response in zip(pending, responses):
        column = _turns_column(dimension, model)
        label = f"{task_id}::{model}::{dimension}"
        if not response.ok:
            print(f"  ERROR {label}: {response.error}", file=sys.stderr)
            still_empty.append(label)
            continue
        corrected = sorted({int(n) for n in (response.data.get("corrected_turns") or [])})
        if not corrected:
            # Both the original attempt and the stronger-instruction retry came
            # back empty. This should not happen given the schema + retry above;
            # if it does, it is surfaced loudly rather than silently written as
            # a blank cell.
            still_empty.append(label)
            print(f"  STILL EMPTY after retry {label} -- left blank, needs human review", file=sys.stderr)
            continue
        note = LOW_CONFIDENCE_NOTE if is_fallback else None
        rows[task_id][column] = annotation_turns_json(corrected, note=note)
        reasoning = str(response.data.get("reasoning") or "")[:100]
        tag = " [LOW CONFIDENCE]" if is_fallback else ""
        print(f"  {label}: corrected_turns={corrected}{tag}  {reasoning}", file=sys.stderr)

    columns = ["task_id"] + [
        f"{prefix}_turns_{suffix}" for prefix in DIMENSION_FIELDS for suffix in ("a", "b")
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for task_id in target_ids:
            row = {"task_id": task_id}
            row.update(rows.get(task_id, {}))
            writer.writerow(row)

    total_cost = sum(r.cost_usd for r in responses)
    print(f"wrote {out_path} (${total_cost:.4f} spend)", file=sys.stderr)
    if rescued_via_pdf:
        print(f"\n{len(rescued_via_pdf)} item(s) recovered a real conversation via transcript_pdf:", file=sys.stderr)
        for line in rescued_via_pdf:
            print(f"  {line}", file=sys.stderr)
    if still_empty:
        print(
            f"\n{len(still_empty)} item(s) could not get even a fallback answer -- "
            "left blank, needs human review:",
            file=sys.stderr,
        )
        for line in still_empty:
            print(f"  {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
