"""Model D ingest: fourth product, three D-anchored SxS pairs, audit-slots."""

from __future__ import annotations

import json

from honeybee_qc.ingest import ingest_attempt
from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL_D, profile_for_project
from honeybee_qc.rating_stages import build_rating_requests, estimate_rating_calls


def _step(output: dict) -> dict:
    return {"output": output}


def _row(before: dict, task: str = "t-d") -> dict:
    return {
        "TASK": task,
        "PROJECT": HONEYBEE_ASPIRATIONAL_D.project_id,
        "RESPONSE": json.dumps(
            {
                "before": before,
                "turns": [],
                "after": {},
                "metrics": {},
                "dataSourceResults": {},
            }
        ),
    }


def _selector(left_url: str, right_url: str, likert: int, idx: int) -> dict:
    return _step(
        {
            "preferenceLikert": likert,
            "responseIdx": idx,
            "fieldResponses": {"preference_justification": "D vs the other one"},
            "candidates": [
                {"id": "response-d", "index": 0, "message": {"content": left_url}},
                {"id": "response-x", "index": 1, "message": {"content": right_url}},
            ],
        }
    )


def test_profile_resolves_the_model_d_project():
    profile = profile_for_project("6a95bca6e0021078893ae170")
    assert profile is HONEYBEE_ASPIRATIONAL_D
    assert [t.slot for t in profile.trajectories] == ["A", "B", "C", "D"]
    assert [(c.left, c.right) for c in profile.comparisons] == [
        ("D", "A"),
        ("D", "B"),
        ("D", "C"),
    ]
    assert profile.has_simplified is False
    assert 97 in profile.disabled_checks


def test_ingest_reads_model_d_and_three_sxs_pairs():
    d = HONEYBEE_ASPIRATIONAL_D.trajectory("D")
    a = HONEYBEE_ASPIRATIONAL_D.trajectory("A")
    b = HONEYBEE_ASPIRATIONAL_D.trajectory("B")
    c = HONEYBEE_ASPIRATIONAL_D.trajectory("C")
    assert d and a and b and c
    before = {
        HONEYBEE_ASPIRATIONAL_D.rubric_step: _step(
            {"criteria": [{"id": "1", "title": "does the thing", "weight": 2}]}
        ),
        a.run_step: _step(
            {"gemini_custom_final_trajectory_a": "https://share.gemini.google/aaaa"}
        ),
        b.run_step: _step({"gpt_trajectory_b": "https://chatgpt.com/share/bbbb"}),
        c.run_step: _step({"claude_trajectory_c": "https://claude.ai/share/cccc"}),
        d.run_step: _step(
            {
                "model_d_trajectory_link": "https://share.gemini.google/dddd",
                "model_d_produced_output": True,
                "model_d_chat_pdf": [
                    {"s3Url": "https://s3.test/d.pdf", "name": "d.pdf"}
                ],
                "model_d_final_outputs": [
                    {"s3Url": "https://s3.test/out.md", "name": "out.md"}
                ],
            }
        ),
        d.scores_step: _step(
            {"responseRatings": {"response-d": {"1": {"score": 0, "title": "Fail"}}}}
        ),
        a.scores_step: _step(
            {"responseRatings": {"response-a": {"1": {"score": 1, "title": "Pass"}}}}
        ),
        HONEYBEE_ASPIRATIONAL_D.comparisons[0].selector_step: _selector(
            "https://share.gemini.google/dddd",
            "https://share.gemini.google/aaaa",
            6,
            1,
        ),
        HONEYBEE_ASPIRATIONAL_D.comparisons[1].selector_step: _selector(
            "https://share.gemini.google/dddd",
            "https://chatgpt.com/share/bbbb",
            7,
            1,
        ),
        HONEYBEE_ASPIRATIONAL_D.comparisons[2].selector_step: _selector(
            "https://share.gemini.google/dddd",
            "https://claude.ai/share/cccc",
            4,
            -1,
        ),
        d.manifest_step: _step(
            {
                "criteria": [
                    {
                        "title": "open the doc",
                        "annotations": {
                            "simp_turn_number": 1,
                            "simp_turn_link": "https://share.gemini.google/dddd1",
                        },
                    },
                    {
                        "title": "write the recap",
                        "annotations": {
                            "simp_turn_number": 2,
                            "simp_turn_link": "https://share.gemini.google/dddd2",
                        },
                    },
                ]
            }
        ),
    }
    result = ingest_attempt(_row(before))
    assert result.ok, result.error
    task = result.task
    assert task is not None
    assert task.model_d is not None
    assert task.model_d.final_link.endswith("/dddd")
    assert task.model_d.produced_final_outcome is True
    assert [e.turn_number for e in task.model_d.turn_manifest] == [1, 2]
    assert {r.model for r in task.criterion_ratings} >= {"A", "D"}
    assert [p.pair_label for p in task.comparison_list()] == ["DA", "DB", "DC"]
    assert task.comparison_list()[0].winner_index == 1
    assert task.comparison_list()[2].winner_index is None
    assert task.expected_slots == ("A", "B", "C", "D")


def test_audit_slots_d_only_spends_270_on_d_and_still_asks_three_sxs():
    result = ingest_attempt(
        _row(
            {
                HONEYBEE_ASPIRATIONAL_D.rubric_step: _step(
                    {"criteria": [{"id": "1", "title": "x", "weight": 1}]}
                ),
                HONEYBEE_ASPIRATIONAL_D.trajectory("A").run_step: _step(
                    {
                        "gemini_custom_final_trajectory_a": "https://share.gemini.google/a"
                    }
                ),
                HONEYBEE_ASPIRATIONAL_D.trajectory("B").run_step: _step(
                    {"gpt_trajectory_b": "https://chatgpt.com/share/b"}
                ),
                HONEYBEE_ASPIRATIONAL_D.trajectory("C").run_step: _step(
                    {"claude_trajectory_c": "https://claude.ai/share/c"}
                ),
                HONEYBEE_ASPIRATIONAL_D.trajectory("D").run_step: _step(
                    {"model_d_trajectory_link": "https://share.gemini.google/d"}
                ),
                HONEYBEE_ASPIRATIONAL_D.trajectory("D").scores_step: _step(
                    {
                        "responseRatings": {
                            "response-d": {"1": {"score": 0, "title": "Fail"}}
                        }
                    }
                ),
                **{
                    pair.selector_step: _selector("u", "v", 5, 1)
                    for pair in HONEYBEE_ASPIRATIONAL_D.comparisons
                },
            }
        )
    )
    task = result.task
    assert task is not None
    task.audit_slots = ("D",)
    requests = build_rating_requests(task)
    kinds = [r.metadata.get("check_id") for r in requests]
    models_270 = [
        r.metadata.get("model")
        for r in requests
        if r.metadata.get("check_id") == 270
    ]
    pairs = [r.metadata.get("pair") for r in requests if r.metadata.get("check_id") == 400]
    assert models_270 == ["D"]
    assert 300 not in kinds
    assert pairs == ["DA", "DB", "DC"]
    assert estimate_rating_calls(task) == 1 + 3


def test_generated_d_dims_are_treated_as_contributor_ratings():
    from honeybee_qc.fill_dims import apply_fill_to_tasks
    from honeybee_qc.models import Task

    task = Task(task_id="t-fill")
    rows = [
        {
            "task_id": "t-fill",
            "outcome_quality_d": "2",
            "outcome_quality_justif_d": "Turn 3 dropped the file.",
            "outcome_quality_turns_d": json.dumps(
                [
                    {
                        "fullText": "Turn 3",
                        "parts": [{"id": "turn_number", "value": "3"}],
                    }
                ]
            ),
        }
    ]
    assert apply_fill_to_tasks([task], rows, "D") == 1
    assert len(task.dimension_ratings) == 1
    dim = task.dimension_ratings[0]
    assert dim.model == "D"
    assert dim.dimension == "Outcome quality"
    assert dim.rating == 2
    assert dim.relevant_turns == [3]
    task.audit_slots = ("D",)
    # Once attached, 300 is in scope.
    from honeybee_qc.models import ModelSubmission

    task.model_d = ModelSubmission(model="D", final_link="https://share.gemini.google/d")
    requests = build_rating_requests(task, guard=False)
    assert any(r.metadata.get("check_id") == 300 for r in requests)


def test_fixer_plans_model_d_scores_and_dimension_turns():
    from honeybee_qc.fixer import plan_task
    from honeybee_qc.models import CriterionRating, DimensionRating
    from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL_D
    from honeybee_qc.tests.fixtures import GEMINI, make_submission, make_task

    task = make_task("t-d-fix", n_criteria=1)
    task.project_id = HONEYBEE_ASPIRATIONAL_D.project_id
    task.expected_slots = ("A", "B", "C", "D")
    task.model_d = make_submission("D", GEMINI, 4, per_turn_links=False)
    task.criterion_ratings.append(CriterionRating(criterion_id="C1", model="D", score=1))
    task.dimension_ratings.append(
        DimensionRating(
            model="D",
            dimension="Trust & grounding",
            rating=3,
            relevant_turns=[2],
            justification="Turn 2 overclaimed.",
        )
    )
    report = {
        "task_id": "t-d-fix",
        "verdict": "fail",
        "fail_checks": [270, 280, 310],
        "checks": [
            {
                "check_id": 270,
                "dimension": "x",
                "sub_dimension": "",
                "band": "fail",
                "error_code": "",
                "contributing_items": ["C1"],
                "measurement": {
                    "notes": "",
                    "counts": {"disagreeing_items": ["D::C1"]},
                },
            },
            {
                "check_id": 280,
                "dimension": "x",
                "sub_dimension": "",
                "band": "fail",
                "error_code": "",
                "contributing_items": ["280::C1::missing_turn"],
                "measurement": {
                    "notes": "",
                    "counts": {
                        "reasons_by_item": {
                            "280::D::C1": ["missing_turn"],
                        },
                        "turns_by_item": {},
                    },
                },
            },
            {
                "check_id": 310,
                "dimension": "x",
                "sub_dimension": "",
                "band": "fail",
                "error_code": "",
                "contributing_items": ["310::trust_grounding"],
                "measurement": {
                    "notes": "",
                    "counts": {
                        "reasons_by_item": {
                            "310::D::trust_grounding": ["incorrect_turns"],
                        },
                        "turns_by_item": {"310::D::trust_grounding": [2]},
                    },
                },
            },
        ],
    }
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL_D)
    assert plan.status == "fixable"
    assert ("criterion_score", "D") in {(j.kind, j.slot) for j in plan.jobs}
    assert ("dimension_turns", "D") in {(j.kind, j.slot) for j in plan.jobs}
    assert not any(j.kind == "criterion_turns" for j in plan.jobs)
    dim_job = next(j for j in plan.jobs if j.kind == "dimension_turns")
    assert dim_job.dimension == "Trust & grounding"

