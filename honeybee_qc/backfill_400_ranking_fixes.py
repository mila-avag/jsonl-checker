"""Backfill script for check 400 (SxS Ranking Disagreement) and the `ranking`
item of check 450 (Justifications), which live in the *other* annotation
step from `backfill_300_450_fixes.py`.

Why these two share a CSV, separate from 300/450's dimension-level fixes:
`preferenceLikert`, `responseIdx`, and `fieldResponses.preference_justification`
are all read out of the single `step-QuantitativeModelResponseSelector-*` step
(confirmed against the raw `taskattempts` rows), not the per-model
`ResponseTextCollection` steps 300/450 use for every other item. A check-450
finding whose `item_id == "ranking"` is judging that exact justification field,
so its fix belongs here, not in the dimension-justifications CSV, even though
both come from check 450.

What each fix is:

- **400 (preferenceLikert)** is a pure lookup, like 300: `evaluate_check_400`
  already computed an independent `auditor_likert` for this exact task, sitting
  in the report's own `measurement.counts` -- no cache lookup or judgment call
  needed, just write it.
- **responseIdx**: this project's `preference_encoding` policy is `bipolar_7`
  (a single 1-7 scale where the number itself carries direction: below the
  scale's midpoint favors model A, above favors B), confirmed in
  `honeybee_qc/config.py`. `responseIdx` is a separate field recorded
  alongside the Likert and needs to keep agreeing with it, so this script also
  fixes `responseIdx` whenever the auditor's Likert crosses the midpoint
  relative to the contributor's -- otherwise the backfilled Likert would
  contradict the untouched `responseIdx`, leaving the annotation internally
  inconsistent in a new way. If the auditor's value doesn't cross the
  midpoint, this column is left blank (no change needed).
- **ranking justification**: same LLM-rewrite approach as the dimension-level
  450 fixes -- grounded in both conversations and the audit's own recorded
  defect(s) for the "ranking" item.

Contract (per the user): a task this script cannot resolve with real, cached
or reported evidence is written with `unfixable_reason` filled and every fix
column left blank -- never a guessed or low-confidence value.

Usage:
    python3 -m honeybee_qc.backfill_400_ranking_fixes \
        --report honeybee_qc/audit_runs/.../chunks/chunk_1000fix_report.json \
        --tasks-csv honeybee_qc/audit_runs/.../pulled_30.csv \
        --cache-db honeybee_qc/audit_runs/.../cache.db \
        --snapshot-dir honeybee_qc/audit_runs/.../snapshots \
        --out honeybee_qc/audit_runs/.../400_ranking_fixes_backfill.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

from .backfill_300_450_fixes import (
    JUSTIF_FIX_SCHEMA,
    _cache_lookup,
    build_ranking_justif_fix_prompt,
    resolve_empty_justif_responses,
)
from .config import DEFAULT_POLICY, POLICY_VERSION
from .hydrate import hydrate_conversations
from .informed_stages import build_informed_requests
from .ingest import load_taskattempts_csv
from .llm import ModelRequest, build_client, run_requests
from .models import Task

COLUMNS = ["task_id", "unfixable_reason", "preferenceLikert", "responseIdx", "preference_justification"]


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
        "includes 400, plus any task with a 'ranking' item in check 450",
    )
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hydrate-workers", type=int, default=8)
    args = ap.parse_args(argv)

    policy = DEFAULT_POLICY
    if args.snapshot_dir:
        policy = replace(policy, snapshot_dir=args.snapshot_dir)
    if policy.preference_encoding != "bipolar_7":
        print(
            f"WARNING: policy.preference_encoding is {policy.preference_encoding!r}, "
            "not bipolar_7 -- the responseIdx-flip logic below assumes bipolar_7 and "
            "may be wrong for this policy. Review before trusting responseIdx fixes.",
            file=sys.stderr,
        )

    report = json.loads(Path(args.report).read_text())
    report_by_task = {t["task_id"]: t for t in report["tasks"]}

    if args.task_ids:
        target_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    else:
        target_ids = []
        for t in report["tasks"]:
            fails400 = 400 in (t.get("fail_checks") or [])
            has_ranking_450 = any(
                "ranking"
                in (
                    ((c.get("measurement") or {}).get("counts") or {}).get("conditions_by_item") or {}
                )
                for c in t.get("checks", [])
                if c.get("check_id") == 450
            )
            if fails400 or has_ranking_450:
                target_ids.append(t["task_id"])

    print(f"backfilling check 400 + 450(ranking) for {len(target_ids)} task(s)", file=sys.stderr)

    ingested, _ingest_errors = load_taskattempts_csv(args.tasks_csv)
    hydration = hydrate_conversations(ingested, policy, workers=args.hydrate_workers)
    print(f"hydrated {hydration.hydrated}/{hydration.submissions} submissions", file=sys.stderr)
    task_by_id: dict[str, Task] = {t.task_id: t for t in ingested}

    conn = sqlite3.connect(args.cache_db) if args.cache_db and Path(args.cache_db).exists() else None
    client, _cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version=f"{POLICY_VERSION}+400-ranking-backfill-v1",
    )

    rows: dict[str, dict[str, str]] = {}
    unfixable: dict[str, str] = {}
    pending_ranking: list[tuple[str, str]] = []  # (task_id, key)
    keys: list[str] = []
    prompts: list[str] = []

    lo, hi = policy.likert_scale
    midpoint = (lo + hi) / 2

    for task_id in target_ids:
        report_task = report_by_task.get(task_id)
        task = task_by_id.get(task_id)
        if report_task is None or task is None:
            unfixable[task_id] = "task not found in report/tasks-csv"
            continue
        row: dict[str, str] = {}
        reason_parts: list[str] = []

        check_400 = next((c for c in report_task.get("checks", []) if c.get("check_id") == 400), None)
        if check_400 is not None:
            counts = (check_400.get("measurement") or {}).get("counts") or {}
            auditor_likert = counts.get("auditor_likert")
            contributor_likert = counts.get("contributor_likert")
            if not isinstance(auditor_likert, int):
                reason_parts.append(
                    f"400: measurement.counts.auditor_likert missing or not an int ({auditor_likert!r})"
                )
            else:
                row["preferenceLikert"] = str(auditor_likert)
                if isinstance(contributor_likert, int):
                    old_side = "A" if contributor_likert < midpoint else "B" if contributor_likert > midpoint else "tie"
                    new_side = "A" if auditor_likert < midpoint else "B" if auditor_likert > midpoint else "tie"
                    if old_side != new_side and new_side != "tie":
                        row["responseIdx"] = "0" if new_side == "A" else "1"
                    # old_side == new_side, or new_side == "tie": responseIdx
                    # already agrees (or there is no clear winner to encode),
                    # so it is left blank -- no change needed.

        items_450 = {}
        for c in report_task.get("checks", []):
            if c.get("check_id") == 450:
                items_450.update(
                    ((c.get("measurement") or {}).get("counts") or {}).get("conditions_by_item") or {}
                )
        if "ranking" in items_450:
            informed_requests = build_informed_requests(task, policy)
            req = next(
                (
                    r
                    for r in informed_requests
                    if r.metadata.get("check_id") == 450 and r.metadata.get("model") is None
                ),
                None,
            )
            if req is None:
                reason_parts.append("450::ranking: could not rebuild the original ranking-justification request")
            else:
                cached = _cache_lookup(conn, req.key)
                if cached is None:
                    reason_parts.append("450::ranking: no cached informed-stage response on file")
                else:
                    judgment = next(
                        (j for j in cached.get("justifications", []) if j.get("item_id") == "ranking"), None
                    )
                    if judgment is None:
                        reason_parts.append("450::ranking: item not found in the cached justifications list")
                    elif judgment.get("unverifiable"):
                        reason_parts.append("450::ranking: audit itself could not verify this justification")
                    elif not (task.model_a and task.model_a.assistant_turns() and task.model_b and task.model_b.assistant_turns()):
                        reason_parts.append("450::ranking: one or both submissions not hydrated -- cannot ground a rewrite")
                    else:
                        prompt = build_ranking_justif_fix_prompt(
                            task, task.sxs.justification, judgment, policy
                        )
                        key = f"{task_id}::450fix::ranking"
                        keys.append(key)
                        prompts.append(prompt)
                        pending_ranking.append((task_id, "preference_justification"))

        rows[task_id] = row
        if reason_parts:
            unfixable[task_id] = "; ".join(reason_parts)

    print(f"sending {len(pending_ranking)} ranking-justification rewrite request(s)", file=sys.stderr)
    requests = [ModelRequest(key=k, prompt=p, schema=JUSTIF_FIX_SCHEMA) for k, p in zip(keys, prompts)]
    responses = run_requests(client, requests, workers=args.workers)
    responses = resolve_empty_justif_responses(client, keys, prompts, responses, args.workers)

    for (task_id, column), response in zip(pending_ranking, responses):
        if task_id not in rows:
            continue
        if not response.ok:
            unfixable[task_id] = (unfixable.get(task_id, "") + "; " if unfixable.get(task_id) else "") + f"450::ranking: {response.error}"
            continue
        text = str(response.data.get("corrected_justification") or "").strip()
        if not text:
            unfixable[task_id] = (
                unfixable.get(task_id, "") + "; " if unfixable.get(task_id) else ""
            ) + "450::ranking: rewrite came back empty even after retry"
            continue
        rows[task_id][column] = text

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
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
