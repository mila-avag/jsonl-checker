"""The keep / edit / delete fixer for rubric criteria the audit flagged.

Mirrors the tara pipeline's two-stage separation (see
`apps/tara_eval_open_claw_rl_main_2/run_threshold_eval.py` and
`scripts/rewrites/run_rewrite_agents.py`): a decision call never also produces
the fix. The decision agent has only the criterion and the issues raised
against it, and writes instructions a second, independent agent acts on
without re-deriving the reasoning:

  1. Resolution stage -- one call per flagged criterion. Reads the criterion,
     the task context, and every issue the rubric audit raised against it, and
     decides `keep | edit | delete`, writing a concrete `resolution_comment`
     for whichever of those the fixer stage will need it for.
  2. Fixer stage -- one call per criterion resolved `edit`. Given the original
     text and the resolution comment (not the raw issues -- the comment has
     already distilled them into an instruction), writes replacement text.
     `delete` needs no call: the resolution stage's own comment is the
     deletion reason. `keep` needs no call either.

Input is the rubric-stage half of an existing audit report (already has
`issues` per criterion) plus the raw tasks CSV (has the rubric's own text and
the prompt/deliverables context). Output is one CSV row per flagged
criterion, the same worklist-as-contract shape as tara's
`resolution_results.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .context import numbered_prompts
from .ingest import load_taskattempts_csv
from .llm import DEFAULT_EFFORT, DEFAULT_MODEL, ModelClient, ModelRequest, build_client, run_requests
from .models import RubricCriterion, Task

RESOLUTION_SYSTEM_PROMPT = (
    "You are deciding how to remediate a single rubric criterion that an "
    "independent audit already flagged. You do not re-litigate whether the "
    "issues are valid -- assume they are -- you decide whether the criterion "
    "is worth saving. Write for a second agent who will act on your comment "
    "alone, without seeing the original issues."
)

RESOLUTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "criterion_id": {"type": "string"},
        "resolution_action": {"type": "string", "enum": ["keep", "edit", "delete"]},
        "resolution_comment": {"type": "string"},
    },
    "required": ["criterion_id", "resolution_action", "resolution_comment"],
    "additionalProperties": False,
}

REWRITE_SYSTEM_PROMPT = (
    "You are fixing a single rubric criterion used to grade an AI agent's "
    "output on a task. You were told what is wrong with it and what to do "
    "about that; you were not told the original issue report, so act on the "
    "instruction as given rather than re-deriving your own diagnosis."
)

REWRITE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "new_rubric": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["new_rubric", "notes"],
    "additionalProperties": False,
}


@dataclass
class FlaggedCriterion:
    task_id: str
    criterion: RubricCriterion
    issues: list[dict]


@dataclass
class FixResult:
    task_id: str
    criterion_id: str
    old_rubric: str
    issue_categories: str
    issue_summary: str
    resolution_action: str = ""
    resolution_comment: str = ""
    new_rubric: str = ""
    notes: str = ""
    error: str = ""


def _issue_summary(issues: list[dict]) -> str:
    parts = []
    for issue in issues:
        category = issue.get("category", "")
        description = issue.get("description", "")
        parts.append(f"[{category}] {description}")
    return " | ".join(parts)


def _context_block(task: Task) -> str:
    lines = ["## Task context", ""]
    if task.user_goal:
        lines.append(f"User's goal: {task.user_goal}")
    if task.target_deliverables:
        lines.append("")
        lines.append("The contributor's own target outcome list:")
        lines += [f"  - {d}" for d in task.target_deliverables]
    lines.append("")
    lines.append("User prompts, in order:")
    lines.append(numbered_prompts(task))
    return "\n".join(lines)


def _resolution_prompt(flagged: FlaggedCriterion, task: Task) -> str:
    criterion = flagged.criterion
    issue_lines = []
    for issue in flagged.issues:
        issue_lines.append(
            f"  - category: {issue.get('category', '')}\n"
            f"    description: {issue.get('description', '')}\n"
            f"    evidence (verbatim quote the audit flagged): "
            f"{issue.get('evidence', '')}"
        )
    return f"""{_context_block(task)}

## The rubric criterion under review

  [{criterion.criterion_id}] {' '.join(criterion.text.split())}

## Issues an independent audit raised against it

{chr(10).join(issue_lines)}

## Your task

Decide `resolution_action`:
  - "keep": the issues are minor enough, or defensible enough, that the
    criterion should stand as written. Leave `resolution_comment` empty.
  - "edit": the criterion tests something real and worth keeping, but its
    wording needs to change to fix the issue(s) above -- e.g. tighten a vague
    term, restate a value the criterion needs so it no longer depends on
    something absent, split an overlapping criterion apart, or narrow an
    overbroad one. In `resolution_comment`, write precise, concrete edit
    instructions a second agent can follow without seeing the issues above:
    name exactly what to change and, where useful, what the new wording
    should say.
  - "delete": the criterion cannot be salvaged by editing -- it tests
    something the task never asked for, is fully redundant with a sibling,
    or asserts something no reading of the prompt supports. In
    `resolution_comment`, write one or two sentences a reviewer could use as
    the deletion's justification.

Set `criterion_id` to "{criterion.criterion_id}".
Return JSON matching the schema."""


def build_rewrite_prompt(task: Task, criterion: RubricCriterion, resolution_comment: str) -> str:
    return f"""{_context_block(task)}

## The rubric criterion to fix (verbatim)

  {criterion.text.strip()}

## What to change and why (from the resolution review)

  {resolution_comment}

## Your task

Rewrite the criterion so it addresses the instruction above while still
testing the same underlying requirement. Keep it a single, self-contained,
positively phrased sentence a grader could apply without reading anything
else. Do not introduce a new requirement the task never asked for.

Return JSON matching the schema: `new_rubric` (the corrected criterion text)
and `notes` (1-2 sentences on what you changed and why)."""


def load_flagged_criteria(
    report_path: Path, tasks: list[Task], task_ids: set[str] | None = None
) -> list[FlaggedCriterion]:
    report = json.loads(report_path.read_text())
    rubric_stage = report.get("rubric_stage") or []
    by_task = {t.task_id: t for t in tasks}

    flagged: list[FlaggedCriterion] = []
    for entry in rubric_stage:
        task_id = entry.get("task_id")
        if task_ids is not None and task_id not in task_ids:
            continue
        task = by_task.get(task_id)
        if task is None:
            print(f"fixer: {task_id} has findings but no matching task in --tasks-csv, skipping", file=sys.stderr)
            continue
        criteria_by_id = {c.criterion_id: c for c in task.rubric}
        for finding in entry.get("findings") or []:
            issues = finding.get("issues") or []
            if not issues:
                continue
            criterion = criteria_by_id.get(finding.get("criterion_id"))
            if criterion is None:
                continue
            flagged.append(FlaggedCriterion(task_id=task_id, criterion=criterion, issues=issues))
    return flagged


def run_fixer(
    flagged: list[FlaggedCriterion],
    tasks_by_id: dict[str, Task],
    client: ModelClient,
    workers: int = 4,
) -> list[FixResult]:
    if not flagged:
        return []

    resolution_requests = []
    for f in flagged:
        task = tasks_by_id[f.task_id]
        resolution_requests.append(
            ModelRequest(
                key=f"resolution:{f.task_id}:{f.criterion.criterion_id}",
                prompt=_resolution_prompt(f, task),
                schema=RESOLUTION_SCHEMA,
                system=RESOLUTION_SYSTEM_PROMPT,
                metadata={"task_id": f.task_id, "criterion_id": f.criterion.criterion_id},
            )
        )

    print(f"fixer: resolution stage, {len(resolution_requests)} criteria", file=sys.stderr)
    resolution_responses = run_requests(client, resolution_requests, workers=workers)

    results: list[FixResult] = []
    to_rewrite: list[tuple[FlaggedCriterion, str]] = []
    for f, resp in zip(flagged, resolution_responses):
        result = FixResult(
            task_id=f.task_id,
            criterion_id=f.criterion.criterion_id,
            old_rubric=f.criterion.text,
            issue_categories=", ".join(sorted({i.get("category", "") for i in f.issues})),
            issue_summary=_issue_summary(f.issues),
        )
        if not resp.ok:
            result.error = resp.error or "no response"
            results.append(result)
            continue
        data = resp.data or {}
        action = data.get("resolution_action", "")
        result.resolution_action = action
        result.resolution_comment = data.get("resolution_comment", "")
        results.append(result)
        if action == "edit":
            to_rewrite.append((f, result))

    if to_rewrite:
        rewrite_requests = [
            ModelRequest(
                key=f"rewrite:{f.task_id}:{f.criterion.criterion_id}",
                prompt=build_rewrite_prompt(tasks_by_id[f.task_id], f.criterion, result.resolution_comment),
                schema=REWRITE_SCHEMA,
                system=REWRITE_SYSTEM_PROMPT,
                metadata={"task_id": f.task_id, "criterion_id": f.criterion.criterion_id},
            )
            for f, result in to_rewrite
        ]
        print(f"fixer: rewrite stage, {len(rewrite_requests)} criteria marked edit", file=sys.stderr)
        rewrite_responses = run_requests(client, rewrite_requests, workers=workers)
        for (f, result), resp in zip(to_rewrite, rewrite_responses):
            if not resp.ok:
                result.error = resp.error or "no response"
                continue
            data = resp.data or {}
            result.new_rubric = data.get("new_rubric", "")
            result.notes = data.get("notes", "")

    return results


def write_csv(results: list[FixResult], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task_id",
                "criterion_id",
                "old_rubric",
                "issue_categories",
                "issue_summary",
                "resolution_action",
                "resolution_comment",
                "new_rubric",
                "notes",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.task_id,
                    r.criterion_id,
                    r.old_rubric,
                    r.issue_categories,
                    r.issue_summary,
                    r.resolution_action,
                    r.resolution_comment,
                    r.new_rubric,
                    r.notes,
                    r.error,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="audit JSON with a rubric_stage section")
    ap.add_argument("--tasks-csv", required=True, help="the Snowflake taskattempts export the report was built from")
    ap.add_argument(
        "--task-ids",
        default=None,
        help="comma-separated task ids to fix; omit to fix every flagged criterion in the report",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    task_ids = set(args.task_ids.split(",")) if args.task_ids else None

    tasks, ingest_results = load_taskattempts_csv(Path(args.tasks_csv))
    for r in ingest_results:
        if r.error:
            print(f"fixer: ingest skipped {r.task_id}: {r.error}", file=sys.stderr)
    tasks_by_id = {t.task_id: t for t in tasks}

    flagged = load_flagged_criteria(Path(args.report), tasks, task_ids)
    print(f"fixer: {len(flagged)} flagged criteria in scope", file=sys.stderr)
    if not flagged:
        write_csv([], Path(args.out))
        return 0

    client, _cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version="rubric-fixer-v1",
    )

    results = run_fixer(flagged, tasks_by_id, client, workers=args.workers)
    write_csv(results, Path(args.out))

    counts: dict[str, int] = {}
    for r in results:
        counts[r.resolution_action or "error"] = counts.get(r.resolution_action or "error", 0) + 1
    print(f"fixer: wrote {len(results)} rows to {args.out} -- {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
