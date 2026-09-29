"""One backfill CSV: task_id plus one column per step.

Takes the original task JSON (`response.before`) and the fixer's patches, and
writes a sheet a reviewer can paste back into the form. Each non-empty cell is
the existing object for that step with the patches merged in -- unfixed keys
stay. A cell is left blank when that step had nothing to update on that task.

Rating-step cells are the full `responseRatings` wrapper
(`{"response-a": {<id>: {...}, ...}}`), ready to paste over `output.responseRatings`.
Prompt steps are omitted: those fails are unfixable (new trajectories required).

Usage::

    python3 -m honeybee_qc.backfill_steps \\
      --tasks-csv /tmp/aspirational_35.csv \\
      --fixer-dir honeybee_qc/audit_runs/aspirational_v5_35/fixer \\
      --out honeybee_qc/audit_runs/aspirational_v5_35/fixer/backfill_steps.csv

Writes two paste sheets next to `--out`: `backfill_steps.csv` (one column per
form step) and `backfill_dims.csv` (one column per dimension annotation).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fixer_invariants import is_na_score_token
from .fill_dims import load_fill_csv
from .fixer import FieldJob, export_locator
from .ingest import DIMENSION_FIELDS, _loads, _output
from .projects import ProjectProfile, TrajectorySpec, resolve_profile
from .taxonomies import parse_criterion_category

SCORE_OPTIONS = {
    0: {"id": "0", "title": "Fail", "color": "red"},
    1: {"id": "1", "title": "Pass", "color": "green"},
}
DIMENSION_TO_PREFIX = {dimension: prefix for prefix, dimension in DIMENSION_FIELDS.items()}


@dataclass(frozen=True)
class StepColumn:
    name: str
    step_id: str
    shape: str
    rating_key: str = ""
    slot: str | None = None

    def paste_path(self) -> str:
        if self.shape == "ratings_inner":
            return "output.responseRatings"
        if self.shape == "annotations":
            return "output.annotations"
        if self.shape == "criteria":
            return "output.criteria"
        if self.shape == "selector":
            return "output.preferenceLikert / output.fieldResponses.preference_justification"
        if self.shape == "run_fields":
            return "output.{key_turn_*, produced_outcome}"
        return "output"


def columns_for_profile(profile: ProjectProfile) -> list[StepColumn]:
    cols: list[StepColumn] = []
    for spec in profile.trajectories:
        letter = spec.slot.lower()
        cols.append(
            StepColumn(
                f"{letter}_scores",
                spec.scores_step,
                "ratings_inner",
                spec.rating_key,
                spec.slot,
            )
        )
        if spec.turns_step:
            cols.append(
                StepColumn(
                    f"{letter}_turns",
                    spec.turns_step,
                    "ratings_inner",
                    spec.rating_key,
                    spec.slot,
                )
            )
        if spec.dims_step:
            cols.append(
                StepColumn(
                    f"{letter}_dims", spec.dims_step, "annotations", slot=spec.slot
                )
            )
        run_name = f"{letter}_key_turn" if spec.slot != "S" else "s_run"
        cols.append(StepColumn(run_name, spec.run_step, "run_fields", slot=spec.slot))
    cols.append(StepColumn("rubric", profile.rubric_step, "criteria"))
    for comparison in profile.comparisons:
        cols.append(
            StepColumn(
                f"rank_{comparison.left.lower()}{comparison.right.lower()}",
                comparison.selector_step,
                "selector",
            )
        )
    if profile.target_outcome_step:
        cols.append(
            StepColumn("target_outcome", profile.target_outcome_step, "criteria")
        )
    if profile.golden_answer_step:
        cols.append(
            StepColumn("golden_answer", profile.golden_answer_step, "text_fields")
        )
    if profile.known_information_step:
        cols.append(
            StepColumn(
                "known_information", profile.known_information_step, "text_fields"
            )
        )
    return cols


def load_befores(path: Path) -> tuple[list[str], dict[str, dict]]:
    """CSV row order, then `before` keyed by task id."""
    csv.field_size_limit(sys.maxsize)
    order: list[str] = []
    befores: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lowered = {str(k).lower(): v for k, v in row.items()}
            task_id = str(lowered.get("task") or lowered.get("task_id") or "").strip()
            if not task_id:
                continue
            response = _loads(lowered.get("response"))
            before = response.get("before") if isinstance(response, dict) else None
            if task_id not in befores:
                order.append(task_id)
            if isinstance(before, dict):
                befores[task_id] = before
    return order, befores


def _issue_names(result: dict) -> list[str]:
    raw = result.get("unfixable_issues")
    if isinstance(raw, list):
        names = [str(x) for x in raw if x]
    elif raw:
        names = [str(raw)]
    else:
        reason = result.get("unfixable_reason")
        names = [str(reason)] if reason else []
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _unfixable_issues_cell(result: dict) -> str:
    return ", ".join(_issue_names(result))


def _merge_fixer_group(task_id: str, group: list[dict]) -> dict:
    """Collapse one task's fixer results without dropping leftover issues.

    `main` used to keep only status and patches, which blanked the
    `unfixable_issues` column on every paste row.
    """
    patches: list[dict] = []
    status = "unfixable"
    issues: list[str] = []
    seen: set[str] = set()
    reason = ""
    for result in group:
        patches.extend(result.get("patches") or [])
        if result.get("status") == "fixable":
            status = "fixable"
        for name in _issue_names(result):
            if name not in seen:
                seen.add(name)
                issues.append(name)
        if not reason:
            reason = str(result.get("unfixable_reason") or "")
    return {
        "task_id": task_id,
        "status": status,
        "patches": patches,
        "unfixable_issues": issues,
        "unfixable_reason": reason,
    }


def _issues_by_task(results: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for result in results:
        task_id = str(result.get("task_id") or "")
        if not task_id:
            continue
        cell = _unfixable_issues_cell(result)
        if cell:
            out[task_id] = cell
    return out


def load_fixer_results(fixer_dir: Path) -> list[dict]:
    path = fixer_dir / "fixer_results.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list of task results")
    return data


def _csv_fields(columns: list[StepColumn]) -> list[str]:
    """Header names. SxS ranking splits into likert + justification, no JSON."""
    names: list[str] = []
    for col in columns:
        if col.shape == "selector":
            names.append(f"{col.name}_likert")
            names.append(f"{col.name}_justification")
        else:
            names.append(col.name)
    return names


def _selector_plain(cell: dict) -> tuple[str, str]:
    likert = cell.get("preferenceLikert")
    likert_s = "" if likert is None else str(likert)
    responses = cell.get("fieldResponses")
    just = ""
    if isinstance(responses, dict):
        raw = responses.get("preference_justification")
        if raw is None:
            just = ""
        elif isinstance(raw, str):
            just = raw
        else:
            just = str(raw)
    return likert_s, just


def _cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [int(text)] if text.lstrip("-").isdigit() else []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(int(item))
            elif str(item).lstrip("-").isdigit():
                out.append(int(item))
        return out
    return []


def _turn_title(nums: list[int]) -> str:
    return ", ".join(f"Turn {n}" for n in nums) if nums else ""


def _turns_annotation(nums: list[int]) -> list[dict]:
    return [
        {
            "fullText": f"Turn {n}",
            "parts": [{"id": "turn_number", "value": str(n)}],
        }
        for n in nums
    ]


def _replace_l1(raw: str, new_l1: str) -> str:
    form_l1 = str(new_l1 or "").strip().lower()
    parsed = parse_criterion_category(raw)
    if parsed.l2_label:
        desc = raw.split(":", 1)[1].strip() if raw and ":" in raw else ""
        leaf = parsed.l2_label.lower()
        if desc:
            return f"{form_l1}-{leaf}: {desc}"
        return f"{form_l1}-{leaf}"
    return form_l1


def _inner_ratings(before: dict, step_id: str, rating_key: str) -> dict:
    ratings = _output(before, step_id).get("responseRatings")
    if not isinstance(ratings, dict):
        return {}
    inner = ratings.get(rating_key)
    return deepcopy(inner) if isinstance(inner, dict) else {}


def _spec(profile: ProjectProfile, slot: str | None) -> TrajectorySpec | None:
    if not slot:
        return None
    return profile.trajectory(slot)  # type: ignore[arg-type]


def _step_id_for(patch: dict, profile: ProjectProfile) -> str:
    if patch.get("step_id"):
        return str(patch["step_id"])
    job = FieldJob(
        kind=patch["kind"],  # type: ignore[arg-type]
        check_ids=(),
        current_value=None,
        feedback="",
        slot=patch.get("slot") or None,
        criterion_id=patch.get("criterion_id") or None,
        dimension=patch.get("dimension") or None,
        comparison=patch.get("comparison") or None,
    )
    step_id, _ = export_locator(job, profile)
    return step_id


def _successful_patches(results: list[dict], profile: ProjectProfile) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = {}
    for result in results:
        task_id = str(result.get("task_id") or "")
        if not task_id:
            continue
        kept: list[dict] = []
        for patch in result.get("patches") or []:
            if patch.get("error") or patch.get("new_value") is None:
                continue
            if patch.get("kind") == "prompt_text":
                continue
            patch = dict(patch)
            patch["step_id"] = _step_id_for(patch, profile)
            if patch["step_id"]:
                kept.append(patch)
        by_task.setdefault(task_id, []).extend(kept)
    return by_task


def _sync_score_entry(entry: dict, score: int) -> None:
    """`id` is the rating option (0=Fail/red, 1=Pass/green), not the criterion UUID."""
    option = SCORE_OPTIONS.get(int(score)) or SCORE_OPTIONS[1 if int(score) else 0]
    entry["score"] = int(score)
    entry["id"] = option["id"]
    entry["title"] = option["title"]
    entry["color"] = option["color"]


def _sync_turn_entry(entry: dict, nums: list[int]) -> None:
    """`id` is the selected turn option, matching `title` (`Turn N`)."""
    entry["title"] = _turn_title(nums)
    entry["id"] = str(nums[0]) if len(nums) == 1 else ",".join(str(n) for n in nums)
    entry.setdefault("color", "none")
    entry.setdefault("justification", "")


def _apply_ratings_inner(
    before: dict, column: StepColumn, patches: list[dict]
) -> dict:
    inner = _inner_ratings(before, column.step_id, column.rating_key)
    for patch in patches:
        cid = str(patch.get("criterion_id") or "")
        if not cid:
            continue
        entry = inner.get(cid)
        if not isinstance(entry, dict):
            entry = {"justification": ""}
            inner[cid] = entry
        if patch["kind"] == "criterion_score":
            _sync_score_entry(entry, int(patch["new_value"]))
        elif patch["kind"] == "criterion_turns":
            nums = _as_int_list(patch["new_value"])
            if nums:
                _sync_turn_entry(entry, nums)
    if column.name.endswith("_scores"):
        for entry in inner.values():
            if isinstance(entry, dict) and entry.get("score") is not None:
                try:
                    _sync_score_entry(entry, int(entry["score"]))
                except (TypeError, ValueError):
                    continue
    return {column.rating_key: inner}


def _apply_annotations(
    before: dict, column: StepColumn, patches: list[dict], profile: ProjectProfile
) -> dict:
    annotations = deepcopy(_output(before, column.step_id).get("annotations") or {})
    if not isinstance(annotations, dict):
        annotations = {}
    spec = _spec(profile, column.slot)
    suffix = spec.suffix if spec else (column.slot or "").lower()
    for patch in patches:
        prefix = DIMENSION_TO_PREFIX.get(str(patch.get("dimension") or ""), "")
        if not prefix:
            continue
        if patch["kind"] == "dimension_rating":
            annotations[f"{prefix}_{suffix}"] = patch["new_value"]
        elif patch["kind"] == "dimension_justification":
            annotations[f"{prefix}_justif_{suffix}"] = patch["new_value"]
        elif patch["kind"] == "dimension_turns":
            nums = _as_int_list(patch["new_value"])
            # Empty is a real edit: N/A / no-tools dimensions must not keep a
            # copied turn list. Blanking the cell would leave the old annotation.
            annotations[f"{prefix}_turns_{suffix}"] = _turns_annotation(nums)
    return annotations


def _apply_criteria(before: dict, column: StepColumn, patches: list[dict]) -> list:
    criteria = deepcopy(_output(before, column.step_id).get("criteria") or [])
    if not isinstance(criteria, list):
        criteria = []
    by_id = {
        str(item.get("id")): item
        for item in criteria
        if isinstance(item, dict) and item.get("id") is not None
    }
    for patch in patches:
        if patch["kind"] == "target_outcome":
            # Whole checklist replacement when the fixer rewrites the list.
            value = patch["new_value"]
            if isinstance(value, list):
                return deepcopy(value)
            continue
        cid = str(patch.get("criterion_id") or "")
        item = by_id.get(cid)
        if item is None:
            item = {"id": cid, "title": "", "weight": 0, "annotations": {}}
            criteria.append(item)
            by_id[cid] = item
        if patch["kind"] == "criterion_text":
            item["title"] = patch["new_value"]
        elif patch["kind"] == "criterion_weight":
            item["weight"] = int(patch["new_value"])
        elif patch["kind"] == "criterion_l1":
            annotations = item.setdefault("annotations", {})
            if not isinstance(annotations, dict):
                annotations = {}
                item["annotations"] = annotations
            raw = str(annotations.get("criterion_category") or "")
            annotations["criterion_category"] = _replace_l1(raw, str(patch["new_value"]))
    return criteria


def _apply_selector(before: dict, column: StepColumn, patches: list[dict]) -> dict:
    output = _output(before, column.step_id)
    field_responses = deepcopy(output.get("fieldResponses") or {})
    if not isinstance(field_responses, dict):
        field_responses = {}
    cell = {
        "preferenceLikert": deepcopy(output.get("preferenceLikert")),
        "responseIdx": deepcopy(output.get("responseIdx")),
        "fieldResponses": field_responses,
    }
    for patch in patches:
        if patch["kind"] == "preference_likert":
            cell["preferenceLikert"] = patch["new_value"]
            try:
                likert = int(patch["new_value"])
            except (TypeError, ValueError):
                likert = None
            if likert in (1, 2, 3):
                cell["responseIdx"] = 0
            elif likert in (5, 6, 7):
                cell["responseIdx"] = 1
            elif likert == 4:
                # 1-7 bipolar: only 4 is a tie. 3 = A slightly better, 5 = opponent
                # slightly better; those must not land in the tie bucket.
                cell["responseIdx"] = -1
        elif patch["kind"] == "preference_justification":
            field_responses["preference_justification"] = patch["new_value"]
    return cell


def _apply_run_fields(
    before: dict, column: StepColumn, patches: list[dict], profile: ProjectProfile
) -> dict:
    output = _output(before, column.step_id)
    spec = _spec(profile, column.slot)
    keys: list[str] = []
    if spec:
        for logical in ("key_turn", "key_turn_justification"):
            name = spec.field_name(logical)
            if name:
                keys.append(name)
        if spec.produced_outcome_field:
            keys.append(spec.produced_outcome_field)
    cell = {key: deepcopy(output[key]) for key in keys if key in output}
    for patch in patches:
        if spec is None:
            continue
        if patch["kind"] == "key_turn_index":
            name = spec.field_name("key_turn")
            if name:
                cell[name] = patch["new_value"]
        elif patch["kind"] == "key_turn_justification":
            name = spec.field_name("key_turn_justification")
            if name:
                cell[name] = patch["new_value"]
        elif patch["kind"] == "produced_final_outcome" and spec.produced_outcome_field:
            cell[spec.produced_outcome_field] = patch["new_value"]
    return cell


def _apply_text_fields(before: dict, column: StepColumn, patches: list[dict]) -> dict:
    output = _output(before, column.step_id)
    seed_keys = (
        "golden_ground_truth_answer",
        "golden_answer",
        "known_information",
        "not_known_information",
    )
    cell = {key: deepcopy(output[key]) for key in seed_keys if key in output}
    for patch in patches:
        if patch["kind"] == "golden_answer":
            key = (
                "golden_ground_truth_answer"
                if "golden_ground_truth_answer" in output or "golden_ground_truth_answer" in cell
                else "golden_answer"
                if "golden_answer" in output or "golden_answer" in cell
                else "golden_ground_truth_answer"
            )
            cell[key] = patch["new_value"]
        elif patch["kind"] in {"known_information", "not_known_information"}:
            cell[patch["kind"]] = patch["new_value"]
    return cell


def apply_step(
    before: dict,
    column: StepColumn,
    patches: list[dict],
    profile: ProjectProfile,
) -> Any:
    if column.shape == "ratings_inner":
        return _apply_ratings_inner(before, column, patches)
    if column.shape == "annotations":
        return _apply_annotations(before, column, patches, profile)
    if column.shape == "criteria":
        return _apply_criteria(before, column, patches)
    if column.shape == "selector":
        return _apply_selector(before, column, patches)
    if column.shape == "run_fields":
        return _apply_run_fields(before, column, patches, profile)
    if column.shape == "text_fields":
        return _apply_text_fields(before, column, patches)
    raise ValueError(f"unknown column shape {column.shape}")


def build_backfill_rows(
    befores: dict[str, dict],
    task_ids: list[str],
    results: list[dict],
    profile: ProjectProfile,
    always_emit_slots: set[str] | None = None,
) -> tuple[list[StepColumn], list[dict[str, str]]]:
    columns = columns_for_profile(profile)
    by_step = {col.step_id: col.name for col in columns}
    patches_by_task = _successful_patches(results, profile)
    issues = _issues_by_task(results)
    fields = _csv_fields(columns)
    force = always_emit_slots or set()
    rows: list[dict[str, str]] = []
    for task_id in task_ids:
        row = {name: "" for name in fields}
        row["task_id"] = task_id
        row["unfixable_issues"] = issues.get(task_id, "")
        before = befores.get(task_id)
        grouped: dict[str, list[dict]] = {col.name: [] for col in columns}
        for patch in patches_by_task.get(task_id) or []:
            name = by_step.get(str(patch.get("step_id") or ""))
            if name:
                grouped[name].append(patch)
        if before is not None:
            for col in columns:
                patches = grouped[col.name]
                force_dims = (
                    col.shape == "annotations"
                    and col.slot in force
                    and not patches
                )
                if not patches and not force_dims:
                    continue
                applied = apply_step(before, col, patches, profile)
                if not applied:
                    continue
                if col.shape == "selector" and isinstance(applied, dict):
                    likert, just = _selector_plain(applied)
                    row[f"{col.name}_likert"] = likert
                    row[f"{col.name}_justification"] = just
                else:
                    row[col.name] = _cell(applied)
        rows.append(row)
    return columns, rows


def write_backfill(
    out_path: Path, columns: list[StepColumn], rows: list[dict[str, str]]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "unfixable_issues"] + _csv_fields(columns)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    sidecar: dict[str, dict] = {}
    for col in columns:
        meta = {
            "step_id": col.step_id,
            "path": col.paste_path(),
            "shape": col.shape,
            "rating_key": col.rating_key,
        }
        if col.shape == "selector":
            sidecar[f"{col.name}_likert"] = {
                **meta,
                "path": "output.preferenceLikert",
            }
            sidecar[f"{col.name}_justification"] = {
                **meta,
                "path": "output.fieldResponses.preference_justification",
            }
        else:
            sidecar[col.name] = meta
    columns_path = out_path.with_suffix(".columns.json")
    columns_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def dimension_field_columns(suffix: str) -> list[str]:
    """Annotation keys for one model, in form order: score, justification, turns."""
    cols: list[str] = []
    for prefix in DIMENSION_FIELDS:
        cols.append(f"{prefix}_{suffix}")
        cols.append(f"{prefix}_justif_{suffix}")
        cols.append(f"{prefix}_turns_{suffix}")
    return cols


def _dimension_cell(kind: str, new_value: Any) -> str:
    if kind == "dimension_rating":
        if is_na_score_token(new_value):
            return "not-applicable"
        return str(int(new_value))
    if kind == "dimension_justification":
        return str(new_value)
    if kind == "dimension_turns":
        nums = _as_int_list(new_value)
        return _cell(_turns_annotation(nums))
    return ""


def _flat_dim_cells(before: dict | None, spec: TrajectorySpec) -> dict[str, str]:
    columns = dimension_field_columns(spec.suffix)
    out = {col: "" for col in columns}
    if not before or not spec.dims_step:
        return out
    annotations = _output(before, spec.dims_step).get("annotations") or {}
    if not isinstance(annotations, dict):
        return out
    suffix = spec.suffix
    for key in columns:
        value = annotations.get(key)
        if value in (None, ""):
            continue
        if key.endswith(f"_turns_{suffix}"):
            out[key] = value if isinstance(value, str) else _cell(value)
        elif key.endswith(f"_justif_{suffix}"):
            out[key] = str(value)
        else:
            try:
                out[key] = str(int(value))
            except (TypeError, ValueError):
                out[key] = str(value)
    return out


def _fill_cells(row: dict, suffix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in dimension_field_columns(suffix):
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        out[key] = raw if isinstance(raw, str) else str(raw)
    return out


def seed_fill_into_befores(
    befores: dict[str, dict],
    fill_rows: list[dict],
    profile: ProjectProfile,
) -> set[str]:
    """Write generated 1-5s into empty dim-step annotations. Returns seeded slots."""
    seeded: set[str] = set()
    for row in fill_rows:
        task_id = str(row.get("task_id") or "").strip()
        slot = str(row.get("slot") or "D").strip().upper() or "D"
        spec = profile.trajectory(slot)  # type: ignore[arg-type]
        if not task_id or spec is None or not spec.dims_step:
            continue
        before = befores.get(task_id)
        if before is None:
            continue
        step = before.get(spec.dims_step)
        if not isinstance(step, dict):
            step = {}
            before[spec.dims_step] = step
        output = step.get("output")
        if not isinstance(output, dict):
            output = {}
            step["output"] = output
        annotations = output.get("annotations")
        if not isinstance(annotations, dict):
            annotations = {}
            output["annotations"] = annotations
        suffix = spec.suffix
        wrote = False
        for key, raw in _fill_cells(row, suffix).items():
            if annotations.get(key) not in (None, ""):
                continue
            if key.endswith(f"_turns_{suffix}"):
                try:
                    annotations[key] = json.loads(raw)
                except json.JSONDecodeError:
                    annotations[key] = raw
            elif key.endswith(f"_justif_{suffix}"):
                annotations[key] = raw
            else:
                try:
                    annotations[key] = int(raw)
                except (TypeError, ValueError):
                    annotations[key] = raw
            wrote = True
        if wrote or annotations:
            seeded.add(slot)
    return seeded


def build_dimension_sheets(
    results: list[dict],
    profile: ProjectProfile,
    befores: dict[str, dict] | None = None,
    fill_rows: list[dict] | None = None,
) -> dict[str, tuple[list[str], list[dict[str, str]]]]:
    """One sheet per model. Existing / generated values first; fixer patches win."""
    specs = [s for s in profile.trajectories if s.dims_step]
    patches_by_task = _successful_patches(results, profile)
    issues = _issues_by_task(results)
    fill_by_task = {
        str(row.get("task_id") or ""): row for row in (fill_rows or []) if row.get("task_id")
    }
    fill_slots = {
        str(row.get("slot") or "D").upper()
        for row in (fill_rows or [])
        if row.get("task_id")
    }
    task_ids: list[str] = []
    seen: set[str] = set()
    for result in results:
        task_id = str(result.get("task_id") or "")
        if task_id and task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)
    sheets: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for spec in specs:
        suffix = spec.suffix
        columns = dimension_field_columns(suffix)
        rows: list[dict[str, str]] = []
        for task_id in task_ids:
            row = {col: "" for col in columns}
            row["task_id"] = task_id
            row["unfixable_issues"] = issues.get(task_id, "")
            fill_slot = str((fill_by_task.get(task_id) or {}).get("slot") or "").upper()
            if spec.slot in fill_slots or fill_slot == spec.slot:
                row.update(_flat_dim_cells((befores or {}).get(task_id), spec))
            fill = fill_by_task.get(task_id)
            if fill and str(fill.get("slot") or spec.slot).upper() == spec.slot:
                for key, value in _fill_cells(fill, suffix).items():
                    if not row.get(key):
                        row[key] = value
            for patch in patches_by_task.get(task_id) or []:
                if patch.get("slot") != spec.slot:
                    continue
                prefix = DIMENSION_TO_PREFIX.get(str(patch.get("dimension") or ""), "")
                if not prefix:
                    continue
                if patch["kind"] == "dimension_rating":
                    key = f"{prefix}_{suffix}"
                elif patch["kind"] == "dimension_justification":
                    key = f"{prefix}_justif_{suffix}"
                elif patch["kind"] == "dimension_turns":
                    key = f"{prefix}_turns_{suffix}"
                else:
                    continue
                row[key] = _dimension_cell(patch["kind"], patch["new_value"])
            rows.append(row)
        sheets[spec.slot.lower()] = (columns, rows)
    return sheets


def write_dimension_sheets(out_dir: Path, sheets: dict[str, tuple[list[str], list[dict[str, str]]]]) -> list[Path]:
    """One dimension sheet: one row per task, every model's annotation keys."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sheets:
        return []
    task_ids: list[str] = []
    seen: set[str] = set()
    issues: dict[str, str] = {}
    by_slot: dict[str, dict[str, dict[str, str]]] = {}
    kept: list[tuple[str, list[str]]] = []
    for slot, (columns, rows) in sheets.items():
        by_task = {str(row.get("task_id") or ""): row for row in rows}
        by_slot[slot] = by_task
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id and task_id not in seen:
                seen.add(task_id)
                task_ids.append(task_id)
            if task_id and row.get("unfixable_issues"):
                issues.setdefault(task_id, str(row.get("unfixable_issues") or ""))
        if any(row.get(col) for row in rows for col in columns):
            kept.append((slot, columns))
    if not kept:
        kept = [(slot, columns) for slot, (columns, _rows) in sheets.items()]
    fieldnames = ["task_id", "unfixable_issues"]
    for _slot, columns in kept:
        fieldnames.extend(columns)
    combined: list[dict[str, str]] = []
    for task_id in task_ids:
        row = {name: "" for name in fieldnames}
        row["task_id"] = task_id
        row["unfixable_issues"] = issues.get(task_id, "")
        for slot, columns in kept:
            src = by_slot.get(slot, {}).get(task_id) or {}
            if not row["unfixable_issues"]:
                row["unfixable_issues"] = str(src.get("unfixable_issues") or "")
            for col in columns:
                row[col] = src.get(col, "")
        combined.append(row)
    path = out_dir / "backfill_dims.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in combined:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    filled = sum(
        1
        for row in combined
        if any(row.get(name) for name in fieldnames if name not in {"task_id", "unfixable_issues"})
    )
    print(
        f"wrote {path} ({len(combined)} rows, {filled} with at least one field)",
        file=sys.stderr,
    )
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", required=True, type=Path)
    parser.add_argument("--fixer-dir", required=True, type=Path, action="append")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project", default="aspirational")
    parser.add_argument(
        "--fill-dims",
        type=Path,
        default=None,
        help="generated 1-5s CSV (fill_dims) merged into dim steps before patches",
    )
    args = parser.parse_args(argv)

    order, befores = load_befores(args.tasks_csv)
    if not order:
        raise SystemExit(f"no tasks in {args.tasks_csv}")
    results: list[dict] = []
    for fixer_dir in args.fixer_dir:
        results.extend(load_fixer_results(fixer_dir))
    by_id: dict[str, list[dict]] = {}
    for result in results:
        by_id.setdefault(str(result.get("task_id") or ""), []).append(result)
    merged: list[dict] = []
    used: set[str] = set()
    for task_id in order:
        merged.append(_merge_fixer_group(task_id, by_id.get(task_id, [])))
        used.add(task_id)
    for task_id, group in by_id.items():
        if task_id in used:
            continue
        merged.append(_merge_fixer_group(task_id, group))
    profile = resolve_profile(key=args.project)
    fill_rows: list[dict] = []
    always_emit: set[str] = set()
    if args.fill_dims:
        fill_rows = load_fill_csv(args.fill_dims)
        always_emit = seed_fill_into_befores(befores, fill_rows, profile)
    columns, rows = build_backfill_rows(
        befores, order, merged, profile, always_emit_slots=always_emit
    )
    write_backfill(args.out, columns, rows)
    filled = sum(1 for row in rows if any(row.get(col.name) for col in columns))
    print(
        f"wrote {args.out} ({len(rows)} rows, {filled} with at least one step, "
        f"{len(columns)} step columns)",
        file=sys.stderr,
    )
    print(f"column map: {args.out.with_suffix('.columns.json')}", file=sys.stderr)
    write_dimension_sheets(
        args.out.parent,
        build_dimension_sheets(merged, profile, befores=befores, fill_rows=fill_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
