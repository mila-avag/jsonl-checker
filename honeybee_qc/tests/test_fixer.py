"""The keep-the-task fixer: trajectory/ceiling gates and one call per field."""

from __future__ import annotations

import json

from honeybee_qc.fixer import (
    FieldJob,
    apply_plan,
    build_fix_request,
    export_locator,
    main,
    plan_task,
    run_fixer,
    simplified_weighted_rate,
    unfixable_reason,
)
from honeybee_qc.tests.test_cli import write_batch
from honeybee_qc.llm import FakeModelClient
from honeybee_qc.models import CriterionRating, KeyTurn
from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL
from honeybee_qc.tests.fixtures import CLAUDE, GEMINI, GPT, make_submission, make_task


ASP = HONEYBEE_ASPIRATIONAL.project_id


def _aspirational(task, s_exchanges=2, s_score=1):
    task.project_id = ASP
    task.expected_slots = ("A", "B", "C")
    task.simplified = make_submission("S", GEMINI, s_exchanges, per_turn_links=False)
    task.criterion_ratings = [
        *task.criterion_ratings,
        *[
            CriterionRating(criterion_id=c.criterion_id, model="S", score=s_score)
            for c in task.rubric
        ],
    ]
    return task


def _report(task_id, verdict, fail_checks, checks):
    return {
        "task_id": task_id,
        "verdict": verdict,
        "fail_checks": fail_checks,
        "checks": checks,
    }


def _check(check_id, band, items=None, counts=None, notes="", error=""):
    return {
        "check_id": check_id,
        "dimension": "test",
        "sub_dimension": "",
        "band": band,
        "error_code": error,
        "contributing_items": list(items or []),
        "measurement": {"notes": notes, "counts": dict(counts or {})},
    }


def test_dead_links_do_not_block_other_field_edits():
    """Valid Links cannot be rewritten; Rubric Evaluation still can."""
    task = _aspirational(make_task("t-90"))
    report = _report(
        "t-90",
        "fail",
        [90, 270],
        [
            _check(90, "fail", items=["B: missing"], error="[Fail - Missing/Invalid Links]"),
            _check(270, "fail", items=["C1"], counts={"disagreeing_items": ["A::C1"]}),
        ],
    )
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    assert plan.status == "fixable"
    assert "trajectory" in plan.unfixable_reason
    assert {j.kind for j in plan.jobs} == {"criterion_score"}
    assert all(90 not in j.check_ids for j in plan.jobs)


def test_unique_links_and_artifacts_are_trajectory_issues():
    task = _aspirational(make_task("t-91"))
    for check_id in (91, 95):
        report = _report(task.task_id, "fail", [check_id], [_check(check_id, "fail")])
        assert unfixable_reason(task, report, HONEYBEE_ASPIRATIONAL).startswith(
            "trajectory"
        )
        plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
        assert plan.status == "unfixable"
        assert plan.jobs == []


def test_unauditable_is_unfixable():
    task = make_task("t-1000")
    report = _report("t-1000", "unauditable", [91, 1000], [_check(1000, "fail")])
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    assert plan.status == "unfixable"
    assert "unauditable" in plan.unfixable_reason


def test_simplified_turn_count_outside_bounds_is_unfixable():
    task = _aspirational(make_task("t-97-turns", n_criteria=4), s_exchanges=8)
    report = _report(
        "t-97-turns",
        "fail",
        [97],
        [_check(97, "fail", notes="8 turns is outside the 1-3 bound")],
    )
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    assert plan.status == "unfixable"
    assert "turns" in plan.unfixable_reason


def test_prompt_fails_do_not_block_other_field_edits():
    """The prompt itself is not rewritten; scores and the rest still are."""
    task = make_task("t-prompt")
    report = _report(
        "t-prompt",
        "fail",
        [60, 72, 270],
        [
            _check(60, "fail", notes="prompt is ambiguous"),
            _check(72, "fail", notes="constraints contradict each other"),
            _check(270, "fail", counts={"disagreeing_items": ["A::C1"]}),
        ],
    )
    plan = plan_task(task, report)
    assert plan.status == "fixable"
    assert plan.unfixable_reason == ""
    assert all(j.kind != "prompt_text" for j in plan.jobs)
    assert {j.field_id for j in plan.jobs} == {"criterion_score::A::C1"}


def test_clarity_and_constraints_alone_are_skipped_not_unfixable():
    """60 and 72 are not rewritten, and they do not park the task."""
    task = make_task("t-ignored-prompt")
    for check_id in (60, 72):
        report = _report(task.task_id, "fail", [check_id], [_check(check_id, "fail")])
        plan = plan_task(task, report)
        assert plan.status == "skipped"
        assert plan.unfixable_reason == ""
        assert plan.jobs == []
    both = _report(
        task.task_id,
        "fail",
        [60, 72],
        [_check(60, "fail"), _check(72, "fail")],
    )
    plan = plan_task(task, both)
    assert plan.status == "skipped"
    assert plan.jobs == []


def test_a_prompt_only_fail_has_nothing_to_edit():
    task = make_task("t-prompt-each")
    for check_id in (71, 76, 85):
        report = _report(task.task_id, "fail", [check_id], [_check(check_id, "fail")])
        assert unfixable_reason(task, report, None) == ""
        plan = plan_task(task, report)
        assert plan.status == "unfixable"
        assert "no editable field" in plan.unfixable_reason
        assert plan.jobs == []


def test_unfixable_issue_names_are_unique_and_only_the_blockers():
    from honeybee_qc.fixer import unfixable_issue_names, unique_check_names

    names = unique_check_names([90, 280, 310, 90])
    assert names == ["Valid Links", "Relevant Turns"]
    blockers = unfixable_issue_names(
        [90, 91, 270, 310],
        "trajectory issue: Valid Links, Unique Links",
    )
    assert blockers == ["Valid Links", "Unique Links"]
    assert "Relevant Turns" not in blockers
    assert unfixable_issue_names([270], "") == []


def test_rating_disagreement_is_one_call_per_score_field():
    task = make_task("t-270", n_criteria=5)
    report = _report(
        "t-270",
        "fail",
        [270],
        [
            _check(
                270,
                "fail",
                items=["C1", "C2"],
                counts={"disagreeing_items": ["A::C1", "B::C1", "A::C2"]},
            )
        ],
    )
    plan = plan_task(task, report)
    assert plan.status == "fixable"
    assert {j.field_id for j in plan.jobs} == {
        "criterion_score::A::C1",
        "criterion_score::B::C1",
        "criterion_score::A::C2",
    }
    reqs = [build_fix_request(task, job) for job in plan.jobs]
    assert len(reqs) == 3
    assert len({r.key for r in reqs}) == 3


def test_ranking_justification_and_verdict_share_one_call():
    task = make_task("t-450")
    report = _report(
        "t-450",
        "fail",
        [450, 470],
        [
            _check(
                450,
                "fail",
                items=["ranking"],
                counts={"conditions_by_item": {"ranking": ["generic"]}},
            ),
            _check(470, "fail"),
        ],
    )
    plan = plan_task(task, report)
    justifs = [j for j in plan.jobs if j.kind == "preference_justification"]
    assert len(justifs) == 1
    assert set(justifs[0].check_ids) == {450, 470}
    likerts = [j for j in plan.jobs if j.kind == "preference_likert"]
    assert len(likerts) == 1
    assert likerts[0].comparison == justifs[0].comparison


    req = build_fix_request(task, likerts[0])
    pair = likerts[0].comparison or "AB"
    assert f"1 = Model {pair[0]} is much better" in req.prompt
    assert f"7 = Model {pair[1]} is much better" in req.prompt
    assert "3 = tie" in req.prompt
    assert "3-5 are ties" in req.prompt
    assert "must be 1-2" in req.prompt
    assert "must be 6-7" in req.prompt
    assert "This field is the" in req.prompt
    assert "0 = not met, 1 = met" in req.system


def test_ranking_fix_prompt_only_renders_the_compared_pair():
    task = make_task("t-pair")
    task.model_c = make_submission("C", CLAUDE, 4)
    task.model_d = make_submission("D", GEMINI, 4)
    task.model_b = make_submission("B", GPT, 4)
    job = FieldJob(
        kind="preference_justification",
        check_ids=(450,),
        current_value="Product B vs Product A, never names D.",
        feedback="Wrong pair.",
        comparison="DB",
    )
    req = build_fix_request(task, job, paired_corrected_value=6)
    assert "## Model D" in req.prompt
    assert "## Model B" in req.prompt
    assert "## Model C" not in req.prompt
    assert "## Model A" not in req.prompt
    assert "1 = Model D is much better" in req.prompt
    assert "7 = Model B is much better" in req.prompt
    assert "3 = tie" in req.prompt
    assert "must be 1-2" in req.prompt


def test_dimension_score_fix_also_rewrites_the_justification():
    task = make_task("t-300")
    report = _report(
        "t-300",
        "fail",
        [300],
        [
            _check(
                300,
                "fail",
                items=["A::Outcome quality"],
                counts={"reasons_by_item": {"A::Outcome quality": ["too_high"]}},
            )
        ],
    )
    plan = plan_task(task, report)
    assert plan.status == "fixable"
    pairs = {(j.kind, j.slot, j.dimension) for j in plan.jobs}
    assert ("dimension_rating", "A", "Outcome quality") in pairs
    assert ("dimension_justification", "A", "Outcome quality") in pairs


def test_likert_fix_also_rewrites_the_writeup():
    task = make_task("t-400")
    report = _report("t-400", "fail", [400], [_check(400, "fail")])
    plan = plan_task(task, report)
    assert plan.status == "fixable"
    kinds = {j.kind for j in plan.jobs}
    assert "preference_likert" in kinds
    assert "preference_justification" in kinds
    labels = {j.comparison for j in plan.jobs if j.kind == "preference_likert"}
    justif_labels = {
        j.comparison for j in plan.jobs if j.kind == "preference_justification"
    }
    assert labels == justif_labels


def test_justification_prompt_sees_the_corrected_score():
    task = make_task("t-pair-wave")
    report = {
        "tasks": [
            _report(
                "t-pair-wave",
                "fail",
                [300],
                [
                    _check(
                        300,
                        "fail",
                        items=["A::Outcome quality"],
                        counts={"reasons_by_item": {"A::Outcome quality": ["too_high"]}},
                    )
                ],
            )
        ]
    }
    seen = []

    def respond(req):
        seen.append(req)
        if "dimension_rating" in req.key:
            return {"value": 2, "notes": "too high"}
        return {"value": "The model did not deliver a usable outcome.", "notes": "matches 2"}

    results = run_fixer([task], report, FakeModelClient(respond), workers=1)
    assert results[0].status == "fixable"
    justif_reqs = [r for r in seen if "dimension_justification" in r.key]
    rating_reqs = [r for r in seen if "dimension_rating" in r.key]
    assert len(rating_reqs) == 1
    assert len(justif_reqs) == 1
    assert "Paired field" in justif_reqs[0].prompt
    assert "2" in justif_reqs[0].prompt


def test_key_turn_is_fixable_because_it_is_a_field_not_a_trajectory():
    task = make_task("t-100", key_turn=1)
    task.model_a.key_turn = KeyTurn(turn_index=1, justification="first turn")
    report = _report(
        "t-100",
        "fail",
        [100],
        [_check(100, "fail", items=["key_turn_A"], notes="key turn is turn 1 on A")],
    )
    plan = plan_task(task, report)
    assert plan.status == "fixable"
    assert plan.jobs[0].kind == "key_turn_index"
    assert plan.jobs[0].slot == "A"


def test_simplified_ceiling_after_honest_fixes_marks_unfixable():
    task = _aspirational(make_task("t-97", n_criteria=4), s_exchanges=2, s_score=1)
    # All four S criteria pass at weight 3 → 100% > 50%.
    report = _report(
        "t-97",
        "fail",
        [97],
        [_check(97, "fail", notes="weighted pass rate 100% exceeds the 50% ceiling")],
    )
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    assert plan.status == "fixable"
    assert {j.slot for j in plan.jobs} == {"S"}
    # The fixer agrees every S criterion still passes.
    responses = {
        f"fix::{task.task_id}::{job.field_id}": {"value": 1, "notes": "still a pass"}
        for job in plan.jobs
    }
    result = apply_plan(task, plan, responses, HONEYBEE_ASPIRATIONAL)
    assert result.status == "unfixable"
    assert "50%" in result.unfixable_reason
    assert result.patches == []
    assert result.simplified_rate_after == 1.0


def test_simplified_ceiling_is_fixable_when_enough_scores_flip():
    task = _aspirational(make_task("t-97b", n_criteria=4), s_exchanges=2, s_score=1)
    report = _report(
        "t-97b",
        "fail",
        [97],
        [_check(97, "fail", notes="weighted pass rate 100% exceeds the 50% ceiling")],
    )
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    # Flip C1,C2,C3 to fail; leave C4 passing → 25%.
    responses = {}
    for job in plan.jobs:
        value = 1 if job.criterion_id == "C4" else 0
        responses[f"fix::{task.task_id}::{job.field_id}"] = {
            "value": value,
            "notes": "ok",
        }
    result = apply_plan(task, plan, responses, HONEYBEE_ASPIRATIONAL)
    assert result.status == "fixable"
    assert result.simplified_rate_after == 0.25
    assert len(result.patches) == 4
    assert all(p.step_id for p in result.patches)


def test_clean_tasks_are_skipped():
    task = make_task("t-clean")
    report = _report("t-clean", "clean", [], [_check(270, "clean")])
    plan = plan_task(task, report)
    assert plan.status == "skipped"
    assert plan.jobs == []


def test_run_fixer_makes_one_call_per_field_and_none_for_unfixable():
    good = make_task("good", n_criteria=3)
    bad = _aspirational(make_task("bad", n_criteria=3))
    report = {
        "tasks": [
            _report(
                "good",
                "fail",
                [270],
                [
                    _check(
                        270,
                        "fail",
                        counts={"disagreeing_items": ["A::C1", "B::C2"]},
                    )
                ],
            ),
            _report("bad", "fail", [90], [_check(90, "fail")]),
        ]
    }
    seen = []

    def respond(req):
        seen.append(req.key)
        return {"value": 0, "notes": "flipped"}

    results = run_fixer(
        [good, bad],
        report,
        FakeModelClient(respond),
        workers=1,
    )
    by_id = {r.task_id: r for r in results}
    assert by_id["good"].status == "fixable"
    assert by_id["bad"].status == "unfixable"
    assert len(seen) == 2
    assert all(key.startswith("fix::good::") for key in seen)


def test_aspirational_score_locator_points_at_the_rating_step():
    task = _aspirational(make_task("loc", n_criteria=1))
    report = _report(
        "loc",
        "fail",
        [270],
        [_check(270, "fail", counts={"disagreeing_items": ["C::C1"]})],
    )
    # C is not on make_task; add a C rating so the job is emitted.
    task.model_c = make_submission("C", GEMINI, 4, per_turn_links=False)
    task.criterion_ratings.append(CriterionRating(criterion_id="C1", model="C", score=1))
    plan = plan_task(task, report, HONEYBEE_ASPIRATIONAL)
    assert len(plan.jobs) == 1
    step, field = export_locator(plan.jobs[0], HONEYBEE_ASPIRATIONAL)
    spec = HONEYBEE_ASPIRATIONAL.trajectory("C")
    assert step == spec.scores_step
    assert spec.rating_key in field
    assert "C1" in field
    assert "score" in field


def test_export_locator_knows_model_d_dims():
    from honeybee_qc.fixer import FieldJob
    from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL_D

    spec = HONEYBEE_ASPIRATIONAL_D.trajectory("D")
    job = FieldJob(
        kind="dimension_rating",
        check_ids=(300,),
        current_value=3,
        feedback="",
        slot="D",
        dimension="Outcome quality",
    )
    step, field = export_locator(job, HONEYBEE_ASPIRATIONAL_D)
    assert spec is not None
    assert step == spec.dims_step
    assert field == "outcome_quality_d"


def test_cli_dry_run_writes_a_plan_without_calling_a_model(tmp_path):
    task = make_task("cli-1", n_criteria=3)
    csv_path = write_batch(tmp_path, [task])
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "tasks": [
                    _report(
                        "cli-1",
                        "fail",
                        [270],
                        [
                            _check(
                                270,
                                "fail",
                                counts={"disagreeing_items": ["A::C1"]},
                            )
                        ],
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "fixer"
    assert (
        main(
            [
                "--report",
                str(report_path),
                "--tasks-csv",
                str(csv_path),
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads((out_dir / "fixer_plan.json").read_text())
    results = json.loads((out_dir / "fixer_results.json").read_text())
    assert plan[0]["status"] == "fixable"
    assert plan[0]["fail_checks"] == ["Rubric Evaluation"]
    assert plan[0]["jobs"][0]["kind"] == "criterion_score"
    assert plan[0]["jobs"][0]["checks"] == ["Rubric Evaluation"]
    assert results[0]["calls"] == 1
    assert results[0]["patches"][0]["notes"] == "dry-run; no model call"
    assert (out_dir / "all_patches.csv").is_file()
    assert (out_dir / "tasks.csv").is_file()
    assert (out_dir / "scoring.csv").is_file()
    scoring = (out_dir / "scoring.csv").read_text(encoding="utf-8")
    assert "old_value" in scoring
    assert "new_value" in scoring
    assert "cli-1" in scoring
    assert "criterion_score" in scoring


def test_relevant_turns_jobs_only_cover_flagged_incorrect_citations():
    task = make_task("t-310")
    report = _report(
        "t-310",
        "fail",
        [310],
        [
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {
                        "310::A::Outcome quality": ["incorrect_turns"],
                        "310::A::Memory & personalization": ["missing_turn"],
                        "310::B::Communication quality": ["unverifiable_turns"],
                    },
                    "turns_by_item": {"310::A::Outcome quality": [3]},
                },
            )
        ],
    )
    plan = plan_task(task, report)
    turns = [j for j in plan.jobs if j.kind == "dimension_turns"]
    assert len(turns) == 1
    assert turns[0].slot == "A"
    assert turns[0].dimension == "Outcome quality"
    assert turns[0].flagged_turns == (3,)
    req = build_fix_request(task, turns[0])
    assert "Flagged as incorrect" in req.prompt
    assert "Keep every other cited turn" in req.prompt
    assert "never invent turn numbers" in req.system


def test_unchanged_turn_lists_are_not_patched():
    task = make_task("t-keep")
    report = _report(
        "t-keep",
        "fail",
        [310],
        [
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {
                        "310::A::Outcome quality": ["incorrect_turns"]
                    },
                    "turns_by_item": {"310::A::Outcome quality": [3]},
                },
            )
        ],
    )
    plan = plan_task(task, report)
    job = next(j for j in plan.jobs if j.kind == "dimension_turns")
    result = apply_plan(
        task,
        plan,
        {
            f"fix::{task.task_id}::{job.field_id}": {
                "value": [3],
                "notes": "original is fine",
            }
        },
    )
    assert result.status == "fixable"
    assert result.patches == []


def test_criterion_score_prompt_states_zero_is_not_met():
    task = make_task("t-polarity")
    report = _report(
        "t-polarity",
        "fail",
        [270],
        [_check(270, "fail", counts={"disagreeing_items": ["A::C1"]})],
    )
    plan = plan_task(task, report)
    job = next(j for j in plan.jobs if j.kind == "criterion_score")
    req = build_fix_request(task, job)
    assert "0 = the criterion is NOT met" in req.prompt
    assert "1 = the criterion IS met" in req.prompt
    assert "0 = not met, 1 = met" in req.system


def test_rubric_quality_skips_invented_missing_ids():
    task = make_task("t-missing")
    report = _report(
        "t-missing",
        "fail",
        [230],
        [_check(230, "fail", items=["missing::ghost", "C1"])],
    )
    plan = plan_task(task, report)
    texts = [j for j in plan.jobs if j.kind == "criterion_text"]
    assert [j.criterion_id for j in texts] == ["C1"]
    req = build_fix_request(task, texts[0])
    assert "not a rating" in req.prompt
    assert "Do not write Yes" in req.prompt
