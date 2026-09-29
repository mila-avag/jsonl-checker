"""Per-step backfill CSV: merged objects, blank when a step has nothing to update."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from honeybee_qc.backfill_steps import (
    apply_step,
    build_backfill_rows,
    columns_for_profile,
    main,
)
from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL


ASP = HONEYBEE_ASPIRATIONAL


def _patch(**kwargs) -> dict:
    payload = {
        "kind": "criterion_score",
        "new_value": 1,
        "slot": "A",
        "criterion_id": "1",
        "dimension": "",
        "comparison": "",
        "error": "",
        "step_id": "",
    }
    payload.update(kwargs)
    return payload


def _result(task_id: str, patches: list[dict], status: str = "fixable", **extra) -> dict:
    payload = {"task_id": task_id, "status": status, "patches": patches}
    payload.update(extra)
    return payload


def _before() -> dict:
    return {
        "step-RubricCriteriaRating-5665ecba649a": {
            "output": {
                "responseRatings": {
                    "response-a": {
                        "1": {"id": "1", "score": 0, "keep": "yes"},
                        "2": {"id": "2", "score": 1},
                    }
                }
            }
        },
        "step-RubricCriteriaRating-e2c61a59aef7": {
            "output": {
                "responseRatings": {
                    "response-a": {"1": {"id": "1", "title": "Turn 11"}}
                }
            }
        },
        "step-ResponseTextCollection-cb06a191fca3": {
            "output": {
                "annotations": {
                    "outcome_quality_a": 4,
                    "outcome_quality_justif_a": "solid",
                    "outcome_quality_turns_a": [
                        {"parts": [{"id": "turn_number", "value": "3"}], "fullText": "Turn 3"}
                    ],
                    "communication_quality_a": 5,
                }
            }
        },
        "step-RubricCriteriaBuilder-rubric01": {
            "output": {
                "criteria": [
                    {
                        "id": "1",
                        "title": "old text",
                        "weight": 2,
                        "annotations": {
                            "criterion_category": "outcome quality-artifact delivery: a usable artifact"
                        },
                    },
                    {"id": "2", "title": "keep me", "weight": 1, "annotations": {}},
                ]
            }
        },
        "step-QuantitativeModelResponseSelector-cd151dbf1b55": {
            "output": {
                "preferenceLikert": 2,
                "responseIdx": 0,
                "fieldResponses": {
                    "preference_justification": "old",
                    "other": "keep",
                },
            }
        },
        "step-TextCollection-29cf069962f8": {
            "output": {
                "key_turn_index_a": 1,
                "key_turn_justification_a": "because",
                "gemini_custom_final_trajectory_a": "https://example.test/a",
            }
        },
    }


def _write_csv(path: Path, tasks: list[tuple[str, dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["TASK", "RESPONSE"])
        writer.writeheader()
        for task_id, before in tasks:
            writer.writerow(
                {
                    "TASK": task_id,
                    "RESPONSE": json.dumps({"before": before}),
                }
            )


def test_score_cell_is_wrapped_under_the_response_key():
    before = _before()
    columns = {c.name: c for c in columns_for_profile(ASP)}
    cell = apply_step(
        before,
        columns["a_scores"],
        [_patch(kind="criterion_score", criterion_id="1", new_value=1, slot="A")],
        ASP,
    )
    inner = cell["response-a"]
    assert inner["1"]["score"] == 1
    assert inner["1"]["id"] == "1"
    assert inner["1"]["title"] == "Pass"
    assert inner["1"]["color"] == "green"
    assert inner["1"]["keep"] == "yes"
    assert inner["2"]["score"] == 1
    assert inner["2"]["title"] == "Pass"
    assert inner["2"]["color"] == "green"


def test_score_patch_sets_fail_pass_title_and_option_id():
    before = _before()
    before["step-RubricCriteriaRating-5665ecba649a"]["output"]["responseRatings"][
        "response-a"
    ] = {
        "0b53c3f7-17a8-4fe6-aac8-efdb3442bce0": {
            "color": "none",
            "id": "0",
            "justification": "",
            "score": 0,
            "title": "",
        }
    }
    columns = {c.name: c for c in columns_for_profile(ASP)}
    cell = apply_step(
        before,
        columns["a_scores"],
        [
            _patch(
                kind="criterion_score",
                criterion_id="0b53c3f7-17a8-4fe6-aac8-efdb3442bce0",
                new_value=1,
                slot="A",
            )
        ],
        ASP,
    )
    entry = cell["response-a"]["0b53c3f7-17a8-4fe6-aac8-efdb3442bce0"]
    assert entry["score"] == 1
    assert entry["id"] == "1"
    assert entry["title"] == "Pass"
    assert entry["color"] == "green"


def test_turn_patch_sets_option_id_to_the_turn_number():
    before = _before()
    columns = {c.name: c for c in columns_for_profile(ASP)}
    cell = apply_step(
        before,
        columns["a_turns"],
        [_patch(kind="criterion_turns", criterion_id="1", new_value=[7], slot="A")],
        ASP,
    )
    entry = cell["response-a"]["1"]
    assert entry["id"] == "7"
    assert entry["title"] == "Turn 7"
    assert entry["color"] == "none"



def test_empty_column_when_that_step_has_no_patches():
    _, rows = build_backfill_rows(
        {"t1": _before()},
        ["t1"],
        [_result("t1", [_patch(kind="criterion_score", slot="A", criterion_id="1", new_value=1)])],
        ASP,
    )
    row = rows[0]
    assert json.loads(row["a_scores"])["response-a"]["1"]["score"] == 1
    assert row["b_scores"] == ""
    assert row["a_turns"] == ""
    assert row["rubric"] == ""


def test_unfixable_and_untouched_tasks_still_get_a_blank_row():
    before = _before()
    _, rows = build_backfill_rows(
        {"fixable": before, "dead": before},
        ["fixable", "dead"],
        [
            _result(
                "fixable",
                [_patch(kind="criterion_score", slot="A", new_value=1)],
                unfixable_issues=["Unique Links"],
            ),
            _result("dead", [], status="unfixable", unfixable_issues=["Valid Links"]),
        ],
        ASP,
    )
    assert rows[0]["task_id"] == "fixable"
    assert rows[0]["a_scores"]
    assert rows[0]["unfixable_issues"] == "Unique Links"
    assert rows[1]["task_id"] == "dead"
    assert rows[1]["a_scores"] == ""
    assert rows[1]["a_dims"] == ""
    assert rows[1]["unfixable_issues"] == "Valid Links"


def test_turns_and_dims_and_rubric_and_ranking_and_key_turn(tmp_path):
    before = _before()
    patches = [
        _patch(kind="criterion_turns", slot="A", criterion_id="1", new_value=[7]),
        _patch(
            kind="dimension_turns",
            slot="A",
            dimension="Outcome quality",
            criterion_id="",
            new_value=[4, 8],
        ),
        _patch(kind="criterion_text", slot="", criterion_id="1", new_value="rewritten"),
        _patch(kind="criterion_weight", slot="", criterion_id="1", new_value=3),
        _patch(kind="criterion_l1", slot="", criterion_id="1", new_value="Communication Quality"),
        _patch(kind="preference_likert", slot="", comparison="AB", new_value=7, criterion_id=""),
        _patch(
            kind="preference_justification",
            slot="",
            comparison="AB",
            new_value="B is clearly better",
            criterion_id="",
        ),
        _patch(kind="key_turn_index", slot="A", new_value=9, criterion_id=""),
    ]
    csv_path = tmp_path / "tasks.csv"
    fixer_dir = tmp_path / "fixer"
    fixer_dir.mkdir()
    _write_csv(csv_path, [("t1", before), ("t2", before)])
    (fixer_dir / "fixer_results.json").write_text(
        json.dumps(
            [
                _result("t1", patches, unfixable_issues=["Unique Links"]),
                _result("t2", [], status="unfixable", unfixable_issues=["Valid Links"]),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "backfill_steps.csv"
    assert main(["--tasks-csv", str(csv_path), "--fixer-dir", str(fixer_dir), "--out", str(out)]) == 0

    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["task_id"] for r in rows] == ["t1", "t2"]
    assert rows[0]["unfixable_issues"] == "Unique Links"
    assert rows[1]["unfixable_issues"] == "Valid Links"
    with (tmp_path / "backfill_dims.csv").open(encoding="utf-8") as handle:
        dim_rows = list(csv.DictReader(handle))
    assert dim_rows[0]["unfixable_issues"] == "Unique Links"
    assert dim_rows[1]["unfixable_issues"] == "Valid Links"
    assert rows[1]["a_turns"] == ""
    assert rows[1]["rank_ab_likert"] == ""
    assert rows[1]["rank_ab_justification"] == ""
    assert "rank_ab" not in rows[0]

    turns = json.loads(rows[0]["a_turns"])
    assert turns["response-a"]["1"]["title"] == "Turn 7"
    assert turns["response-a"]["1"]["id"] == "7"
    dims = json.loads(rows[0]["a_dims"])
    assert dims["communication_quality_a"] == 5
    assert dims["outcome_quality_turns_a"] == [
        {"fullText": "Turn 4", "parts": [{"id": "turn_number", "value": "4"}]},
        {"fullText": "Turn 8", "parts": [{"id": "turn_number", "value": "8"}]},
    ]
    rubric = json.loads(rows[0]["rubric"])
    by_id = {c["id"]: c for c in rubric}
    assert by_id["1"]["title"] == "rewritten"
    assert by_id["1"]["weight"] == 3
    assert by_id["1"]["annotations"]["criterion_category"].startswith(
        "communication quality-artifact delivery:"
    )
    assert by_id["2"]["title"] == "keep me"
    assert rows[0]["rank_ab_likert"] == "7"
    assert rows[0]["rank_ab_justification"] == "B is clearly better"
    assert not str(rows[0]["rank_ab_justification"]).startswith("{")
    key = json.loads(rows[0]["a_key_turn"])
    assert key["key_turn_index_a"] == 9
    assert key["key_turn_justification_a"] == "because"
    assert "gemini_custom_final_trajectory_a" not in key
    assert rows[0]["a_scores"] == ""
    mapping = json.loads(out.with_suffix(".columns.json").read_text(encoding="utf-8"))
    assert mapping["a_scores"]["step_id"] == "step-RubricCriteriaRating-5665ecba649a"
    assert mapping["a_scores"]["path"] == "output.responseRatings"


def test_failed_patches_do_not_fill_a_cell():
    _, rows = build_backfill_rows(
        {"t1": _before()},
        ["t1"],
        [
            _result(
                "t1",
                [
                    _patch(kind="criterion_score", error="model failed", new_value=None),
                    _patch(kind="criterion_score", new_value=None),
                ],
            )
        ],
        ASP,
    )
    assert rows[0]["a_scores"] == ""


def test_dimension_sheets_are_one_field_per_column_per_model(tmp_path):
    from honeybee_qc.backfill_steps import build_dimension_sheets

    results = [
        _result(
            "t1",
            [
                _patch(
                    kind="dimension_rating",
                    slot="A",
                    dimension="Outcome quality",
                    criterion_id="",
                    new_value=2,
                ),
                _patch(
                    kind="dimension_turns",
                    slot="A",
                    dimension="Outcome quality",
                    criterion_id="",
                    new_value=[4, 8],
                ),
                _patch(
                    kind="dimension_turns",
                    slot="B",
                    dimension="Memory & personalization",
                    criterion_id="",
                    new_value=[3],
                ),
            ],
        ),
        _result("t2", [], status="unfixable", unfixable_issues=["Unique Links"]),
    ]
    sheets = build_dimension_sheets(results, ASP)
    a_cols, a_rows = sheets["a"]
    b_cols, b_rows = sheets["b"]
    assert a_rows[0]["unfixable_issues"] == ""
    assert a_rows[1]["unfixable_issues"] == "Unique Links"
    assert "outcome_quality_a" in a_cols
    assert "outcome_quality_justif_a" in a_cols
    assert "outcome_quality_turns_a" in a_cols
    assert "outcome_quality_b" not in a_cols
    assert a_rows[0]["outcome_quality_a"] == "2"
    assert a_rows[0]["outcome_quality_justif_a"] == ""
    assert json.loads(a_rows[0]["outcome_quality_turns_a"]) == [
        {"fullText": "Turn 4", "parts": [{"id": "turn_number", "value": "4"}]},
        {"fullText": "Turn 8", "parts": [{"id": "turn_number", "value": "8"}]},
    ]
    assert a_rows[1]["outcome_quality_a"] == ""
    assert b_rows[0]["memory_personalization_turns_b"]
    assert b_rows[0]["outcome_quality_b"] == ""

    from honeybee_qc.backfill_steps import write_dimension_sheets

    written = write_dimension_sheets(tmp_path, sheets)
    assert written == [tmp_path / "backfill_dims.csv"]
    with written[0].open(encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert "outcome_quality_a" in combined[0]
    assert "memory_personalization_turns_b" in combined[0]
    assert combined[0]["outcome_quality_a"] == "2"
    assert combined[0]["memory_personalization_turns_b"]
    assert not (tmp_path / "backfill_dims_a.csv").exists()


def test_merge_keeps_leftover_issues_on_a_partial_fix():
    from honeybee_qc.backfill_steps import _merge_fixer_group

    merged = _merge_fixer_group(
        "t1",
        [
            _result(
                "t1",
                [_patch(kind="criterion_score", new_value=1)],
                unfixable_issues=["Unique Links", "Valid Links"],
                unfixable_reason="trajectory issue: Unique Links, Valid Links",
            )
        ],
    )
    assert merged["status"] == "fixable"
    assert merged["unfixable_issues"] == ["Unique Links", "Valid Links"]
    assert merged["patches"]


def test_model_d_columns_include_d_and_three_sxs_pairs():
    from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL_D

    names = [c.name for c in columns_for_profile(HONEYBEE_ASPIRATIONAL_D)]
    assert "d_scores" in names
    assert "d_dims" in names
    assert "d_turns" not in names
    assert names.count("rank_da") == 1
    assert names.count("rank_db") == 1
    assert names.count("rank_dc") == 1


def test_fill_dims_csv_lands_in_d_sheet_and_d_dims_cell(tmp_path):
    from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL_D

    d_step = HONEYBEE_ASPIRATIONAL_D.trajectory("D").dims_step
    before = {
        d_step: {"output": {"annotations": {"outcome_quality_d": 4}}},
    }
    csv_path = tmp_path / "tasks.csv"
    fixer_dir = tmp_path / "fixer"
    fixer_dir.mkdir()
    _write_csv(csv_path, [("orig", before), ("gen", {})])
    (fixer_dir / "fixer_results.json").write_text(
        json.dumps(
            [
                _result("orig", [], status="fixable"),
                _result("gen", [], status="unfixable", unfixable_issues=["Valid Links"]),
            ]
        ),
        encoding="utf-8",
    )
    fill_path = tmp_path / "d_dims_fill.csv"
    with fill_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "slot",
                "outcome_quality_d",
                "outcome_quality_justif_d",
                "outcome_quality_turns_d",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "gen",
                "slot": "D",
                "outcome_quality_d": "2",
                "outcome_quality_justif_d": "Turn 3 dropped it.",
                "outcome_quality_turns_d": json.dumps(
                    [{"parts": [{"id": "turn_number", "value": "3"}], "fullText": "Turn 3"}]
                ),
            }
        )
    out = tmp_path / "backfill_steps.csv"
    assert (
        main(
            [
                "--tasks-csv",
                str(csv_path),
                "--fixer-dir",
                str(fixer_dir),
                "--out",
                str(out),
                "--project",
                "aspirational_d",
                "--fill-dims",
                str(fill_path),
            ]
        )
        == 0
    )
    with out.open(encoding="utf-8") as handle:
        rows = {r["task_id"]: r for r in csv.DictReader(handle)}
    orig_dims = json.loads(rows["orig"]["d_dims"])
    assert orig_dims["outcome_quality_d"] == 4
    gen_dims = json.loads(rows["gen"]["d_dims"])
    assert gen_dims["outcome_quality_d"] == 2
    assert gen_dims["outcome_quality_justif_d"] == "Turn 3 dropped it."
    with (tmp_path / "backfill_dims.csv").open(encoding="utf-8") as handle:
        dim_rows = {r["task_id"]: r for r in csv.DictReader(handle)}
    assert dim_rows["orig"]["outcome_quality_d"] == "4"
    assert dim_rows["gen"]["outcome_quality_d"] == "2"
    assert dim_rows["gen"]["outcome_quality_justif_d"] == "Turn 3 dropped it."
    assert dim_rows["gen"]["unfixable_issues"] == "Valid Links"


def test_empty_dimension_turns_clears_copied_annotation():
    col = next(c for c in columns_for_profile(ASP) if c.name == "a_dims")
    applied = apply_step(
        _before(),
        col,
        [
            _patch(
                kind="dimension_turns",
                slot="A",
                dimension="Outcome quality",
                criterion_id="",
                new_value=[],
            )
        ],
        ASP,
    )
    assert applied["outcome_quality_turns_a"] == []


def test_likert_4_is_tie_not_left_or_right():
    col = next(c for c in columns_for_profile(ASP) if c.shape == "selector")
    applied = apply_step(
        _before(),
        col,
        [_patch(kind="preference_likert", slot="", comparison="AB", new_value=4, criterion_id="")],
        ASP,
    )
    assert applied["preferenceLikert"] == 4
    assert applied["responseIdx"] == -1


def test_l1_sxs_c_maps_model_a_produced_outcome_field():
    from honeybee_qc.projects import profile_by_key

    profile = profile_by_key("l1_sxs_c")
    assert profile is not None
    a = profile.trajectory("A")
    c = profile.trajectory("C")
    assert a is not None and c is not None
    assert a.produced_outcome_field == "model_produced_final_outcome_a"
    assert c.produced_outcome_field == "model_produced_final_outcome_c"

