"""Export everything an auditor needs to check a task by hand.

The audit report says what the pipeline concluded; it does not carry the material
the conclusion was drawn from. This joins the contributor's submission -- target
outcome, the rubric verbatim with its labels, weights, per-model ratings and cited
turns, the dimension ratings and justifications, the key turn, the side-by-side
verdict -- to the hydrated conversations and to the audit's own per-criterion
findings, so one screen answers "is this call right".

Conversation text is capped rather than dropped: a full export runs to about 2.9M
characters, which no viewer should hold. Turns a rating cites keep a larger
budget, since those are the ones an auditor reads closely, and every truncation is
marked so nothing looks complete when it is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_POLICY
from .hydrate import hydrate_conversations
from .ingest import load_taskattempts_csv
from .models import Task

# Character budgets per turn. User turns are never truncated: they are the request
# the whole audit is judged against, and they are small.
CITED_TURN_CHARS = 1100
PLAIN_TURN_CHARS = 400


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _clip(text: str, limit: int) -> tuple[str, bool]:
    text = _clean(text)
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def _cited_turns(task: Task, model: str) -> set[int]:
    cited: set[int] = set()
    for rating in task.criterion_ratings:
        if rating.model == model:
            cited.update(rating.relevant_turns or [])
    for rating in task.dimension_ratings:
        if rating.model == model:
            cited.update(rating.relevant_turns or [])
    if task.key_turn.turn_index is not None:
        cited.add(task.key_turn.turn_index)
    return cited


def _conversation(task: Task, model: str, turns) -> list[dict]:
    cited = _cited_turns(task, model)
    out: list[dict] = []
    for turn in turns:
        is_cited = turn.index in cited
        limit = (
            10**9
            if turn.role == "user"
            else (CITED_TURN_CHARS if is_cited else PLAIN_TURN_CHARS)
        )
        text, truncated = _clip(turn.text, limit)
        out.append(
            {
                "n": turn.index,
                "r": turn.role,
                "t": text,
                **({"cut": 1} if truncated else {}),
                **({"cited": 1} if is_cited else {}),
                **({"art": turn.artifacts_mentioned[:6]} if turn.artifacts_mentioned else {}),
            }
        )
    return out


def _justification_conditions(payload: dict) -> dict[str, dict[str, list[str]]]:
    """Which justification tripped which of check 450's conditions, per task.

    The gate also reports a flat union of every condition tripped anywhere on the
    task. Hanging that off a justification would have each of the fifteen answer for
    the others' faults, so only the per-item map is read. A report written before the
    gate carried that map falls back to the informed stage's own per-item record, and
    a run scoped to pooled justifications yields keys no dimension matches, which is
    the honest result: pooling destroys the attribution.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for task in payload.get("tasks") or []:
        for check in task.get("checks") or []:
            if check.get("check_id") != 450:
                continue
            mapping = ((check.get("measurement") or {}).get("counts") or {}).get(
                "conditions_by_item"
            )
            if isinstance(mapping, dict):
                out[task.get("task_id", "")] = {
                    str(item): list(conditions) for item, conditions in mapping.items()
                }

    for stage in payload.get("informed_stage") or []:
        task_id = stage.get("task_id", "")
        if task_id in out:
            continue
        entries = {
            str(entry.get("item")): list(entry.get("conditions") or [])
            for entry in stage.get("justifications") or []
            if entry.get("conditions")
        }
        if entries:
            out[task_id] = entries
    return out


def build(tasks_csv: Path, report_path: Path | None) -> dict:
    tasks, _ = load_taskattempts_csv(tasks_csv)
    report = hydrate_conversations(tasks, DEFAULT_POLICY, workers=8)

    findings_by_task: dict[str, dict[str, dict]] = {}
    conditions_by_task: dict[str, dict[str, list[str]]] = {}
    if report_path and report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for stage in payload.get("rubric_stage") or []:
            findings_by_task[stage["task_id"]] = {
                f["criterion_id"]: f for f in stage.get("findings") or []
            }
        conditions_by_task = _justification_conditions(payload)

    out_tasks = []
    for task in tasks:
        findings = findings_by_task.get(task.task_id, {})
        conditions = conditions_by_task.get(task.task_id, {})

        scores: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
        for rating in task.criterion_ratings:
            scores.setdefault(rating.model, {})[rating.criterion_id] = {
                "s": rating.score,
                "turns": rating.relevant_turns or [],
            }

        criteria = []
        for position, criterion in enumerate(task.rubric, start=1):
            finding = findings.get(criterion.criterion_id) or {}
            audit: dict = {}
            issues = [
                {
                    "cat": issue.get("category", ""),
                    "sev": issue.get("severity", ""),
                    "why": _clip(issue.get("description", ""), 320)[0],
                    "quote": _clip(issue.get("evidence", ""), 180)[0],
                }
                for issue in finding.get("issues") or []
            ]
            if issues:
                audit["issues"] = issues
            for key in ("l1", "l2"):
                value = finding.get(key)
                if value and value.get("incorrect"):
                    audit[key] = value.get("auditor", "")
            weight = finding.get("weight")
            if weight and weight.get("delta"):
                audit["w"] = weight.get("auditor")

            criteria.append(
                {
                    "n": f"C{position}",
                    "id": criterion.criterion_id,
                    "t": _clean(criterion.text),
                    "l1": criterion.l1_label or "",
                    "l2": criterion.l2_label or "",
                    "w": criterion.weight,
                    "proc": 1 if criterion.is_process_criterion else 0,
                    "a": scores.get("A", {}).get(criterion.criterion_id),
                    "b": scores.get("B", {}).get(criterion.criterion_id),
                    **({"audit": audit} if audit else {}),
                }
            )

        dimensions = []
        for rating in task.dimension_ratings:
            tripped = conditions.get(f"{rating.model}::{rating.dimension}") or []
            dimensions.append(
                {
                    "m": rating.model,
                    "d": rating.dimension,
                    "r": None if rating.not_applicable else rating.rating,
                    "na": 1 if rating.not_applicable else 0,
                    "j": _clip(rating.justification, 1200)[0],
                    "turns": rating.relevant_turns or [],
                    **({"cond": tripped} if tripped else {}),
                }
            )

        submissions = []
        for sub in task.submissions():
            submissions.append(
                {
                    "m": sub.model,
                    "link": sub.final_link,
                    "provider": sub.declared_provider,
                    "attachments": [
                        (sub.attachment_names.get(a) or a) for a in sub.attachments[:12]
                    ],
                    "turns": _conversation(task, sub.model, sub.conversation),
                }
            )

        out_tasks.append(
            {
                "id": task.task_id,
                "deliverables": [_clean(d) for d in task.target_deliverables],
                "outcome": [_clean(o) for o in task.target_outcome],
                "prompts": [
                    {"n": p.index, "r": p.role, "t": _clean(p.text)} for p in task.prompts
                ],
                "keyTurn": task.key_turn.turn_index,
                "keyTurnWhy": _clip(task.key_turn.justification, 1500)[0],
                "likert": task.sxs.likert,
                "sxsWhy": _clip(task.sxs.justification, 1800)[0],
                **(
                    {"sxsCond": conditions["ranking"]}
                    if conditions.get("ranking")
                    else {}
                ),
                "criteria": criteria,
                "dimensions": dimensions,
                "subs": submissions,
            }
        )

    return {
        "tasks": out_tasks,
        "hydration": {
            "submissions": report.submissions,
            "hydrated": report.hydrated,
            "turns": report.turns,
            "failures": report.failures,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--report", default=None, help="audit JSON, for per-criterion findings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    data = build(Path(args.tasks_csv), Path(args.report) if args.report else None)
    text = json.dumps(data, separators=(",", ":"))
    Path(args.out).write_text(text, encoding="utf-8")
    criteria = sum(len(t["criteria"]) for t in data["tasks"])
    turns = sum(len(s["turns"]) for t in data["tasks"] for s in t["subs"])
    print(
        f"wrote {args.out}: {len(data['tasks'])} tasks, {criteria} criteria, "
        f"{turns} turns, {len(text) / 1024:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
