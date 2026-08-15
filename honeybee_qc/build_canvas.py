"""Render an audit run and the human QC verdicts into a per-task review canvas.

The audit's own JSON is complete but not readable: a gate reports a rate and a
threshold, and the reason behind it lives in a separate per-criterion record.
This joins the two so each task states its verdict, the exact spec error code
behind it, the evidence, and what the human auditor said about the same task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .errors import ERROR_CODES
from .registry import REGISTRY
from .score_against_qc import (
    DETERMINISTIC,
    INFORMED_STAGE,
    RATING_STAGE,
    RUBRIC_STAGE,
    read_ground_truth,
    read_predictions,
)

STAGE_OF = {
    **{c: "deterministic" for c in DETERMINISTIC},
    **{c: "rubric" for c in RUBRIC_STAGE},
    **{c: "rating" for c in RATING_STAGE},
    **{c: "informed" for c in INFORMED_STAGE},
}


def _spec_label(check_id: int, band: str) -> str:
    bands = ERROR_CODES.get(check_id, {})
    return bands.get(band) or bands.get("fail") or bands.get("non_fail") or ""


def _check_name(check_id: int) -> str:
    spec = REGISTRY.get(check_id)
    if not spec:
        return f"check {check_id}"
    sub = getattr(spec, "sub_dimension", "")
    return f"{spec.dimension}{' / ' + sub if sub else ''}"


def _justification(check: dict, findings: dict[str, dict], rubric: dict | None) -> str:
    """Plain-language reason a check landed where it did."""
    measurement = check.get("measurement") or {}
    parts: list[str] = []

    rate, threshold = measurement.get("rate"), measurement.get("threshold")
    num, den = measurement.get("numerator"), measurement.get("denominator")
    if rate is not None and den:
        line = f"{num} of {den} criteria ({rate:.0%})"
        if threshold is not None:
            line += f", against a {threshold:.0%} threshold"
        parts.append(line)

    counts = measurement.get("counts") or {}
    if check["check_id"] == 460 and counts:
        parts.append(
            f"dimension ratings favour {counts.get('dimension_direction')}, "
            f"Likert selects {counts.get('likert_direction')}"
        )
    elif check["check_id"] == 90 and counts:
        parts.append(
            f"{counts.get('final_links_valid', 0)} of "
            f"{counts.get('final_links_checked', 0)} final links valid, "
            f"{counts.get('dead_links', 0)} dead"
        )

    notes = (measurement.get("notes") or "").strip()
    if notes:
        parts.append(notes)
    return " · ".join(parts)


def _evidence(check: dict, findings: dict[str, dict], label_of: dict[str, str]) -> list[dict]:
    """The individual criteria a census gate counted, with the model's reasoning."""
    if check["check_id"] not in (230, 240, 250, 200, 210, 260):
        return []

    wanted = {"230": {"major"}, "240": {"major", "moderate"}, "250": {"major", "moderate", "minor"}}
    out: list[dict] = []
    # Where the per-criterion record is absent the gate's own per-item map is the
    # only attributed reason left; the task-level tallies beside it belong to the
    # whole rubric and would be a guess against any one criterion.
    attributed = _attribution_details(check)

    for criterion_id in check.get("contributing_items") or []:
        finding = findings.get(criterion_id)
        if not finding:
            out.append(
                {
                    "criterion": _item_label(
                        criterion_id, check["check_id"], label_of
                    ),
                    "detail": "; ".join(attributed.get(criterion_id, [])),
                }
            )
            continue

        if check["check_id"] in (230, 240, 250):
            severities = wanted[str(check["check_id"])]
            for issue in finding.get("issues") or []:
                if issue.get("severity") in severities:
                    out.append(
                        {
                            "criterion": label_of.get(criterion_id, criterion_id),
                            "detail": f"{issue.get('category')}: {issue.get('description')}",
                            "quote": issue.get("evidence") or "",
                        }
                    )
        elif check["check_id"] == 200 and finding.get("l1"):
            l1 = finding["l1"]
            out.append(
                {
                    "criterion": label_of.get(criterion_id, criterion_id),
                    "detail": f"labelled {l1['contributor']}, should be {l1['auditor']}",
                }
            )
        elif check["check_id"] == 210 and finding.get("l2"):
            l2 = finding["l2"]
            out.append(
                {
                    "criterion": label_of.get(criterion_id, criterion_id),
                    "detail": f"labelled {l2['contributor']}, should be {l2['auditor']}",
                }
            )
        elif check["check_id"] == 260 and finding.get("weight"):
            weight = finding["weight"]
            low = weight.get("defensible_low")
            high = weight.get("defensible_high")
            if low is None or high is None:
                detail = f"weighted {weight['contributor']}, should be {weight['auditor']}"
            else:
                span = f"{low}" if low == high else f"{low}-{high}"
                detail = (
                    f"weighted {weight['contributor']}, outside the defensible {span}"
                )
            out.append(
                {
                    "criterion": label_of.get(criterion_id, criterion_id),
                    "detail": detail,
                }
            )
    return out[:12]


# Counts that mean "this much of the population could not be judged". A check can
# report a band and still have set evidence aside, and a clean band earned by
# discarding half the citations is not the same claim as a clean band earned on
# all of them.
_SET_ASIDE_KEYS = {
    "unverifiable_turns": "turn citation(s) outside the fetched conversation",
    "unverifiable_claims": "claim(s) whose evidence could not be checked",
    "references_unverifiable": "reference(s) that could not be resolved",
    "unevidenced_discarded": "finding(s) discarded for quoting no evidence",
    "abstained": "judgment(s) abstained",
}

# A stage the operator did not run, as opposed to a judgement the pipeline
# declined to make. Both must be visible, but they are different statements.
_STAGE_NOT_RUN = "Requires a model stage; not implemented in phase 1."


def _abstention(check: dict) -> dict | None:
    """Why a check produced no usable judgement, or None if it judged normally.

    Silence is the failure mode this guards against: a dimension the pipeline
    never evaluated renders identically to one it evaluated and found clean, so
    the reader credits the contributor for work nobody checked.
    """
    if check["band"] != "not_evaluated" or check["check_id"] == 1000:
        return None
    notes = ((check.get("measurement") or {}).get("notes") or "").strip()
    return {
        "id": check["check_id"],
        "name": _check_name(check["check_id"]),
        "stage": STAGE_OF.get(check["check_id"], "not implemented"),
        "kind": "stage not run" if notes == _STAGE_NOT_RUN else "abstained",
        "why": notes,
    }


def _set_aside(check: dict) -> list[str]:
    """Evidence this check saw but could not judge, however it banded."""
    counts = (check.get("measurement") or {}).get("counts") or {}
    out: list[str] = []
    for key, label in _SET_ASIDE_KEYS.items():
        value = counts.get(key)
        if isinstance(value, int) and value > 0:
            out.append(f"{value} {label}")
    # 85's file half reports its own blocker rather than a count, because a scan
    # that never ran has nothing to count.
    if counts.get("ran") is False and counts.get("blocked_reason"):
        out.append(str(counts["blocked_reason"]))
    return out


# Measurement keys carrying per-item attribution, most specific first. Everything
# beside them in `counts` is a task-wide union: `conditions_triggered` names every
# condition any one justification tripped, so rendering it against the item list
# accuses fifteen justifications of one justification's fault.
_ATTRIBUTION_KEYS = (
    "conditions_by_item",
    "reasons_by_item",
    "severities_by_item",
    "categories_by_item",
    "classifications_by_item",
    "status_by_item",
    "turns_by_item",
    "unverifiable_turns_by_item",
    "basis_by_model",
)

_ATTRIBUTION_PREFIX = {
    "turns_by_item": "cites turn(s) ",
    "unverifiable_turns_by_item": "turn(s) outside the fetched conversation: ",
    "basis_by_model": "matched by ",
}


def _item_label(item: str, check_id: int, label_of: dict[str, str]) -> str:
    """Name an item the way the audit sheet does, without its bookkeeping prefix."""
    name = item
    if name.startswith(f"{check_id}::"):
        name = name[len(f"{check_id}::") :]
    if name.startswith("missing::"):
        return f"missing criterion {name[len('missing::') :]}"
    return label_of.get(name, name)


def _attribution_details(check: dict) -> dict[str, list[str]]:
    """Per-item reasons a gate recorded, keyed by the item that tripped them.

    Reports written before the gates carried these maps have none of these keys and
    get an empty result, so an old report falls through to the stage records rather
    than losing its evidence.
    """
    counts = (check.get("measurement") or {}).get("counts") or {}
    details: dict[str, list[str]] = {}

    for key in _ATTRIBUTION_KEYS:
        mapping = counts.get(key)
        if not isinstance(mapping, dict):
            continue
        prefix = _ATTRIBUTION_PREFIX.get(key, "")
        for item, value in mapping.items():
            if not value:
                continue
            named = (
                ", ".join(str(v) for v in value)
                if isinstance(value, (list, tuple))
                else str(value)
            )
            details.setdefault(str(item), []).append(f"{prefix}{named}")
    return details


def _attributed_evidence(check: dict, label_of: dict[str, str]) -> list[dict]:
    """One row per item, naming only what that item was actually faulted for."""
    details = _attribution_details(check)
    return [
        {
            "criterion": _item_label(item, check["check_id"], label_of),
            "detail": "; ".join(details[item]),
        }
        for item in sorted(details)
    ][:12]


def _informed_evidence(
    check: dict, informed: dict | None, label_of: dict[str, str]
) -> list[dict]:
    """What the informed pass actually objected to, per check.

    The gate reports only a count, so without this the eight informed checks read
    as a verdict with no reason attached.
    """
    if not informed:
        return []
    check_id = check["check_id"]
    out: list[dict] = []

    if check_id == 75:
        # Neither 70 nor 85 has a case here -- their evidence lives only in
        # `contributing_items` -- but 75's two quotes are worth naming
        # separately, the same way 470's stated-preference quote is below.
        pc = informed.get("prompt_consistency")
        if pc and pc.get("assessment") != "same":
            out.append(
                {
                    "criterion": "pre-seeded vs. submitted prompt",
                    "detail": f"assessed as {pc.get('assessment')}",
                    "quote": pc.get("submitted_quote") or "",
                }
            )
    elif check_id == 80:
        for requirement in check.get("contributing_items") or []:
            out.append({"criterion": "target outcome", "detail": requirement})
    elif check_id == 95:
        for entry in informed.get("artifacts") or []:
            if entry.get("shortfall") or entry.get("missing"):
                named = ", ".join(entry.get("missing") or []) or (
                    f"{entry.get('shortfall')} file(s) short"
                )
                out.append(
                    {
                        "criterion": f"model {entry.get('model')}",
                        "detail": f"{named} (matched by {entry.get('basis')})",
                    }
                )
    elif check_id == 110 and informed.get("key_turn_justification_issue"):
        out.append(
            {
                "criterion": "key turn justification",
                "detail": "does not justify the turn the contributor chose",
            }
        )
    elif check_id == 220:
        for autofail in informed.get("autofails") or []:
            criterion_id = autofail.get("criterion_id", "")
            out.append(
                {
                    "criterion": label_of.get(criterion_id, criterion_id),
                    "detail": (
                        "second opinion confirmed the autofail"
                        if autofail.get("confirmed")
                        else "alleged autofail not confirmed on review"
                    ),
                }
            )
    elif check_id in (280, 310):
        for entry in informed.get("relevant_turns") or []:
            if entry.get("check_id") != check_id:
                continue
            item = entry.get("item", "")
            details = []
            if entry.get("incorrect"):
                details.append(
                    "cites turn(s) "
                    + ", ".join(str(t) for t in entry["incorrect"])
                    + " that do not support it"
                )
            if entry.get("missing_turn"):
                details.append("cites no turn at all")
            # A citation past the end of a partly fetched conversation is reported
            # separately: the audit cannot see the turn, so it cannot fault it.
            if entry.get("unverifiable"):
                details.append(
                    "turn(s) "
                    + ", ".join(str(t) for t in entry["unverifiable"])
                    + " were outside the fetched conversation"
                )
            if details:
                out.append(
                    {
                        "criterion": f"{entry.get('model')} {label_of.get(item, item)}",
                        "detail": "; ".join(details),
                    }
                )
    elif check_id == 450:
        for entry in informed.get("justifications") or []:
            if entry.get("conditions"):
                out.append(
                    {
                        "criterion": entry.get("item", ""),
                        "detail": ", ".join(entry["conditions"]),
                    }
                )
    elif check_id == 470 and not informed.get("states_preference"):
        out.append(
            {
                "criterion": "side-by-side justification",
                "detail": "never states which response is preferred",
            }
        )
    return out[:12]


def build(report_path: Path, sheet_path: Path, tasks_csv: Path | None) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    truth = read_ground_truth(sheet_path)
    predicted = read_predictions(report_path)

    rubric_by_task = {r["task_id"]: r for r in payload.get("rubric_stage") or []}
    rating_by_task = {r["task_id"]: r for r in payload.get("rating_stage") or []}
    informed_by_task = {r["task_id"]: r for r in payload.get("informed_stage") or []}

    # Criterion ids are opaque; the audit sheet talks in C-numbers, so map back.
    label_of: dict[str, str] = {}
    if tasks_csv and tasks_csv.exists():
        from .ingest import load_taskattempts_csv

        ingested, _ = load_taskattempts_csv(tasks_csv)
        for task in ingested:
            for position, criterion in enumerate(task.rubric, start=1):
                label_of[criterion.criterion_id] = f"C{position}"

    tasks_out = []
    for task in payload["tasks"]:
        task_id = task["task_id"]
        gt = truth.get(task_id)
        rubric = rubric_by_task.get(task_id)
        informed = informed_by_task.get(task_id)
        findings = {
            f["criterion_id"]: f for f in (rubric or {}).get("findings", [])
        }

        checks_out = []
        abstentions = []
        for check in task["checks"]:
            abstained = _abstention(check)
            if abstained:
                abstentions.append(abstained)
                continue
            if check["band"] == "clean":
                # A clean band that had to discard evidence is still reported, so
                # the reader can tell a verified clean from an unverified one.
                held_back = _set_aside(check)
                if held_back:
                    abstentions.append(
                        {
                            "id": check["check_id"],
                            "name": _check_name(check["check_id"]),
                            "stage": STAGE_OF.get(check["check_id"], "not implemented"),
                            "kind": "clean, but evidence set aside",
                            "why": "; ".join(held_back),
                        }
                    )
                continue
            checks_out.append(
                {
                    "id": check["check_id"],
                    "name": _check_name(check["check_id"]),
                    "band": check["band"],
                    "stage": STAGE_OF.get(check["check_id"], "not implemented"),
                    "code": check.get("error_code") or _spec_label(check["check_id"], check["band"]),
                    "why": _justification(check, findings, rubric),
                    "evidence": _evidence(check, findings, label_of)
                    or _attributed_evidence(check, label_of)
                    or _informed_evidence(check, informed, label_of),
                    "set_aside": _set_aside(check),
                }
            )
        checks_out.sort(key=lambda c: (c["band"] != "fail", c["id"]))
        abstentions.sort(key=lambda c: c["id"])

        qc_checks = []
        if gt:
            for check_id in sorted(gt.fail_checks):
                qc_checks.append({"id": check_id, "band": "fail", "code": _spec_label(check_id, "fail")})
            for check_id in sorted(gt.non_fail_checks - gt.fail_checks):
                qc_checks.append(
                    {"id": check_id, "band": "non_fail", "code": _spec_label(check_id, "non_fail")}
                )

        implemented = (
            set(DETERMINISTIC)
            | set(RUBRIC_STAGE)
            | set(RATING_STAGE)
            | set(INFORMED_STAGE)
        )
        audit_ids = {c["id"] for c in checks_out}
        qc_ids = {c["id"] for c in qc_checks}

        tasks_out.append(
            {
                "task_id": task_id,
                "audit_verdict": task["verdict"],
                "qc_verdict": (gt.verdict if gt else ""),
                "qc_score": (gt.score if gt else ""),
                "checks": checks_out,
                "abstentions": abstentions,
                "qc_checks": qc_checks,
                "agreed": sorted((audit_ids & qc_ids)),
                "missed": sorted((qc_ids & implemented) - audit_ids),
                "unreachable": sorted(qc_ids - implemented),
                "extra": sorted(audit_ids - qc_ids),
                "criteria": (rubric or {}).get("criteria_audited", 0),
                "severity": (rubric or {}).get("census", {}),
                "rating_abstentions": sum(
                    (rating_by_task.get(task_id, {}).get("abstentions") or {}).values()
                ),
            }
        )

    order = {"Fail": 0, "": 1, "Pass": 2}
    tasks_out.sort(key=lambda t: (order.get(t["qc_verdict"], 1), t["task_id"]))

    return {
        "tasks": tasks_out,
        "cost_usd": payload.get("cost_usd", 0),
        "hydration": payload.get("hydration") or {},
        "stages_run": {
            "rubric": bool(payload.get("rubric_stage")),
            "rating": bool(payload.get("rating_stage")),
            "informed": bool(payload.get("informed_stage")),
        },
        "stage_errors": payload.get("stage_errors") or [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--tasks-csv", default=None)
    ap.add_argument("--out", required=True, help="JSON consumed by the canvas")
    args = ap.parse_args(argv)

    data = build(
        Path(args.report),
        Path(args.sheet),
        Path(args.tasks_csv) if args.tasks_csv else None,
    )
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(data['tasks'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
