"""The keep-the-task fixer: trajectory/ceiling gates and one call per field."""

from __future__ import annotations

import json

from honeybee_qc.fixer import (
    JUSTIFICATION_VOICE_PROMPT,
    SYSTEM_PROMPT,
    FieldJob,
    FixPlan,
    apply_plan,
    build_fix_request,
    export_locator,
    fix_request_key,
    main,
    plan_task,
    run_fixer,
    simplified_weighted_rate,
    unfixable_reason,
)
from honeybee_qc.fixer_invariants import TIE_RE, audit_task, patched_task, sxs_body
from honeybee_qc.tests.test_cli import write_batch
from honeybee_qc.llm import FakeModelClient
from honeybee_qc.models import CriterionRating, KeyTurn, Sxs, Turn, TurnManifestEntry
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


def _html_export_b(task, suffix="mhtml"):
    task.model_b.final_link = ""
    task.model_b.declared_provider = "gemini"
    task.model_b.transcript_pdf = f"https://cdn.example.com/Gemini_Spark.{suffix}"
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link="")
        for i in range(1, 8)
    ]
    return task


def test_html_export_unique_links_does_not_park_other_fixes():
    """91 on Spark HTML-B is N/A; 310/400 still get jobs."""
    from honeybee_qc.fixer import unfixable_issue_names

    task = _html_export_b(make_task("t-html-91"))
    report = _report(
        task.task_id,
        "fail",
        [91, 310, 400],
        [
            _check(91, "fail", items=["B:turn 1:missing"]),
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {
                        "310::A::Outcome quality": ["incorrect_turns"]
                    },
                    "turns_by_item": {"310::A::Outcome quality": [3]},
                },
            ),
            _check(400, "fail"),
        ],
    )
    leftover = unfixable_reason(task, report, None)
    assert leftover == ""
    plan = plan_task(task, report, None)
    assert plan.status == "fixable"
    assert {j.kind for j in plan.jobs} >= {"dimension_turns", "preference_likert"}
    assert all(91 not in j.check_ids for j in plan.jobs)
    assert "Unique Links" not in unfixable_issue_names(plan.fail_checks, plan.unfixable_reason)


def test_html_export_unique_links_alone_is_skipped_not_unfixable():
    task = _html_export_b(make_task("t-html-91-only"), suffix="html")
    report = _report(task.task_id, "fail", [91], [_check(91, "fail")])
    assert unfixable_reason(task, report, None) == ""
    plan = plan_task(task, report, None)
    assert plan.status == "skipped"
    assert plan.unfixable_reason == ""


def test_share_link_b_reused_urls_still_unique_links_unfixable():
    """Duplicated Gemini share URLs on B remain a real Unique Links park."""
    task = make_task("t-91-share-b")
    url = "https://gemini.google.com/share/aaaaaaaa"
    task.model_b.declared_provider = "gemini"
    task.model_b.final_link = url
    task.model_b.transcript_pdf = "https://cdn.example.com/Gemini_Spark.mhtml"
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link=url)
        for i in range(1, 5)
    ]
    report = _report(task.task_id, "fail", [91], [_check(91, "fail")])
    assert unfixable_reason(task, report, None).startswith("trajectory")
    plan = plan_task(task, report, None)
    assert plan.status == "unfixable"
    assert "Unique Links" in plan.unfixable_reason


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
    assert "4 = tie" in req.prompt
    assert "Only 4 is a tie" in req.prompt
    assert "must be 1-3" in req.prompt
    assert "must be 5-7" in req.prompt
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
    assert "4 = tie" in req.prompt
    assert "must be 1-3" in req.prompt


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
    kinds = {j.kind for j in plan.jobs}
    assert "preference_likert" in kinds
    assert "preference_justification" in kinds


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
        fix_request_key(task.task_id, job.field_id): {"value": 1, "notes": "still a pass"}
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
        responses[fix_request_key(task.task_id, job.field_id)] = {
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
    assert all(key.startswith("fix::v8::good::") for key in seen)


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
    assert "Never invent a turn number that is not in this conversation" in req.system


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
            fix_request_key(task.task_id, job.field_id): {
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
    assert "Never raise a 0 to a 1" in req.prompt
    assert "Never raise a criterion from 0 to 1" in req.system


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


def test_rubric_quality_skips_overlap_and_unknown_ids():
    task = make_task("t-overlap")
    report = _report(
        "t-overlap",
        "fail",
        [230],
        [
            _check(
                230,
                "fail",
                items=[
                    "overlap::11a38ed6-c9d0-4f7f-9206-3443ab610de8+86a1bac4-9d12-4bdf-9123-c0c1d711319b",
                    "criterion_not_a_real_row",
                    "C1",
                ],
            )
        ],
    )
    plan = plan_task(task, report)
    assert [j.criterion_id for j in plan.jobs if j.kind == "criterion_text"] == ["C1"]


def test_relevant_turns_expands_bracketed_uuid_bags_onto_real_ids():
    task = make_task("t-brackets", n_criteria=2)
    uid = "03de09cd-2150-4e0b-a1ab-acbbee2e05fe"
    uid2 = "0b6a0523-99d0-476e-ba45-e6de61864e7a"
    task.rubric[0].criterion_id = uid
    task.rubric[1].criterion_id = uid2
    for rating in task.criterion_ratings:
        if rating.criterion_id == "C1":
            rating.criterion_id = uid
        elif rating.criterion_id == "C2":
            rating.criterion_id = uid2
    key = f"281::B::[{uid}] [{uid2}]"
    report = _report(
        "t-brackets",
        "fail",
        [280],
        [
            _check(
                280,
                "fail",
                counts={
                    "reasons_by_item": {key: ["incorrect_turns"]},
                    "turns_by_item": {key: [3]},
                },
            )
        ],
    )
    plan = plan_task(task, report)
    turn_jobs = [j for j in plan.jobs if j.kind == "criterion_turns"]
    assert {(j.slot, j.criterion_id) for j in turn_jobs} == {("B", uid), ("B", uid2)}
    assert all("[" not in (j.criterion_id or "") for j in turn_jobs)


def test_stub_conversation_is_not_rewritten():
    task = make_task("t-stub")
    task.model_b = make_submission("B", GPT, 1, per_turn_links=False)
    report = _report(
        "t-stub",
        "fail",
        [270, 280],
        [
            _check(270, "fail", counts={"disagreeing_items": ["A::C1", "B::C1"]}),
            _check(
                280,
                "fail",
                counts={
                    "reasons_by_item": {
                        "280::A::C1": ["incorrect_turns"],
                        "280::B::C1": ["incorrect_turns"],
                    },
                    "turns_by_item": {"280::A::C1": [3], "280::B::C1": [8]},
                },
            ),
        ],
    )
    plan = plan_task(task, report)
    assert {j.field_id for j in plan.jobs} == {
        "criterion_score::A::C1",
        "criterion_turns::A::C1",
    }


def test_apply_plan_drops_turns_that_are_not_in_the_conversation():
    task = make_task("t-clamp", exchanges=5)
    report = _report(
        "t-clamp",
        "fail",
        [310],
        [
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {"310::A::Outcome quality": ["incorrect_turns"]},
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
        {fix_request_key(task.task_id, job.field_id): {"value": [4, 19], "notes": "ok"}},
    )
    assert result.patches[0].new_value == [4]


def test_title_as_criterion_id_maps_onto_the_rubric_row():
    task = make_task("t-title", n_criteria=1)
    title = task.rubric[0].text
    report = _report(
        "t-title",
        "fail",
        [230],
        [_check(230, "fail", items=[title])],
    )
    plan = plan_task(task, report)
    assert [j.criterion_id for j in plan.jobs] == ["C1"]


def _apply(task, report, responses):
    plan = plan_task(task, report)
    return plan, apply_plan(task, plan, responses)


def test_unknown_dimension_title_is_not_a_job():
    task = make_task("t-title-dim")
    report = _report(
        "t-title-dim",
        "fail",
        [310, 300],
        [
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {
                        "310::B::Project context / Shopify architecture specifics": [
                            "incorrect_turns"
                        ],
                        "310::A::Communication quality (dimension 2)": ["incorrect_turns"],
                    },
                    "turns_by_item": {
                        "310::B::Project context / Shopify architecture specifics": [3],
                        "310::A::Communication quality (dimension 2)": [3],
                    },
                },
            ),
            _check(
                300,
                "fail",
                items=["B::not a real dimension"],
                counts={"reasons_by_item": {"B::not a real dimension": ["too_high"]}},
            ),
        ],
    )
    plan = plan_task(task, report)
    dims = {(j.kind, j.slot, j.dimension) for j in plan.jobs}
    assert ("dimension_turns", "A", "Communication quality") in dims
    assert all("Shopify" not in (j.dimension or "") for j in plan.jobs)
    assert all(j.dimension != "not a real dimension" for j in plan.jobs)


def test_ranking_skipped_when_either_slot_is_a_stub():
    task = make_task("t-rank-stub")
    task.model_b = make_submission("B", GPT, 1, per_turn_links=False)
    report = _report(
        "t-rank-stub",
        "fail",
        [400, 470],
        [_check(400, "fail"), _check(470, "fail")],
    )
    plan = plan_task(task, report)
    assert not any(
        j.kind in {"preference_likert", "preference_justification"} for j in plan.jobs
    )


def test_apply_drops_inverted_criterion_polarity_notes():
    task = make_task("t-zero-met", n_criteria=1)
    report = _report(
        "t-zero-met",
        "fail",
        [270],
        [_check(270, "fail", counts={"disagreeing_items": ["A::C1"]})],
    )
    plan, result = _apply(
        task,
        report,
        {
            fix_request_key("t-zero-met", "criterion_score::A::C1"): {
                "value": 0,
                "notes": "score corrected to 0 to indicate the criterion is met",
            }
        },
    )
    assert plan.status == "fixable"
    assert result.patches == []


def test_apply_drops_dimension_one_described_as_met():
    task = make_task("t-dim-met")
    report = _report(
        "t-dim-met",
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
    rating = next(j for j in plan.jobs if j.kind == "dimension_rating")
    just = next(j for j in plan.jobs if j.kind == "dimension_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, rating.field_id): {
                "value": 1,
                "notes": "supports the paired score of 1 (met)",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "Turns 2-8 supply the deliverable; score of 1 (met).",
                "notes": "ok",
            },
        },
    )
    assert result.patches == []


def test_apply_drops_likert_winner_flip():
    task = make_task("t-flip", likert=6)
    task.sxs.justification = "Model B is clearly better on outcome."
    report = _report("t-flip", "fail", [400], [_check(400, "fail")])
    plan = plan_task(task, report)
    likert = next(j for j in plan.jobs if j.kind == "preference_likert")
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, likert.field_id): {
                "value": 2,
                "notes": "A better",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "Model A is slightly better on grounding.",
                "notes": "flipped",
            },
        },
    )
    assert not any(p.kind == "preference_likert" for p in result.patches)


def test_apply_drops_winner_to_tie():
    task = make_task("t-to-tie", likert=2)
    task.sxs.justification = "Model A is slightly better."
    report = _report("t-to-tie", "fail", [400], [_check(400, "fail")])
    plan = plan_task(task, report)
    likert = next(j for j in plan.jobs if j.kind == "preference_likert")
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, likert.field_id): {
                "value": 4,
                "notes": "midpoint tie",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "Neither model is clearly ahead; treat as a tie.",
                "notes": "tie",
            },
        },
    )
    assert not any(p.kind == "preference_likert" for p in result.patches)


def test_apply_keeps_same_side_strength_change():
    """3 is a slight A win, not a tie. 2 → 3 stays on A."""
    task = make_task("t-slight", likert=2)
    task.sxs.justification = "Model A is better."
    report = _report("t-slight", "fail", [400], [_check(400, "fail")])
    plan = plan_task(task, report)
    likert = next(j for j in plan.jobs if j.kind == "preference_likert")
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, likert.field_id): {
                "value": 3,
                "notes": "slight A",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "Model A is slightly better.",
                "notes": "same side",
            },
        },
    )
    likerts = [p for p in result.patches if p.kind == "preference_likert"]
    assert likerts and likerts[0].new_value == 3


def test_apply_drops_unflagged_turn_removals():
    task = make_task("t-keep-unflagged")
    dim = next(
        d
        for d in task.dimension_ratings
        if d.model == "A" and d.dimension == "Outcome quality"
    )
    dim.relevant_turns = [3, 5]
    report = _report(
        "t-keep-unflagged",
        "fail",
        [310],
        [
            _check(
                310,
                "fail",
                counts={
                    "reasons_by_item": {"310::A::Outcome quality": ["incorrect_turns"]},
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
        {fix_request_key(task.task_id, job.field_id): {"value": [4], "notes": "swap all"}},
    )
    assert result.patches == []


def test_apply_drops_key_turn_one_and_out_of_range_weight():
    task = make_task("t-kt1", key_turn=1, n_criteria=1)
    task.model_a.key_turn = KeyTurn(turn_index=1, justification="first")
    report = _report(
        "t-kt1",
        "fail",
        [100, 260],
        [
            _check(100, "fail", items=["A"]),
            _check(260, "fail", items=["C1"]),
        ],
    )
    plan = plan_task(task, report)
    kt = next(j for j in plan.jobs if j.kind == "key_turn_index")
    wt = next(j for j in plan.jobs if j.kind == "criterion_weight")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, kt.field_id): {"value": 1, "notes": "still 1"},
            fix_request_key(task.task_id, wt.field_id): {"value": 7, "notes": "off scale"},
        },
    )
    assert result.patches == []


def test_apply_drops_rating_prose_in_criterion_text():
    task = make_task("t-prose", n_criteria=1)
    report = _report(
        "t-prose",
        "fail",
        [230],
        [_check(230, "fail", items=["C1"])],
    )
    plan, result = _apply(
        task,
        report,
        {
            fix_request_key("t-prose", "criterion_text::C1"): {
                "value": "Not met. Model A failed this.",
                "notes": "rewrite",
            }
        },
    )
    assert result.patches == []


def test_dimension_prompt_forbids_one_as_met():
    task = make_task("t-dim-prompt")
    report = _report(
        "t-dim-prompt",
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
    job = next(j for j in plan.jobs if j.kind == "dimension_rating")
    req = build_fix_request(task, job)
    assert "1 = worst" in req.prompt
    assert "score of 1 (met)" in req.prompt
    assert "Do not flip a ranking winner" in req.system


def test_apply_drops_justification_that_cites_missing_turns():
    task = make_task("t-ghost-turn", exchanges=5)
    report = _report(
        "t-ghost-turn",
        "fail",
        [450],
        [
            _check(
                450,
                "fail",
                items=["A::Outcome quality"],
                counts={"conditions_by_item": {"A::Outcome quality": ["generic"]}},
            )
        ],
    )
    plan = plan_task(task, report)
    just = next(j for j in plan.jobs if j.kind == "dimension_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, just.field_id): {
                "value": "Turn 19 is the Gmail draft that actually appeared.",
                "notes": "ok",
            }
        },
    )
    just_patches = [p for p in result.patches if p.kind == "dimension_justification"]
    assert just_patches == []


def test_apply_drops_criterion_zero_to_one():
    """Dump-review UNSAFE: 0→1 claimed met against a conversation that did not."""
    task = make_task("t-no-raise", n_criteria=1)
    for rating in task.criterion_ratings:
        if rating.model == "A":
            rating.score = 0
    report = _report(
        "t-no-raise",
        "fail",
        [270],
        [_check(270, "fail", counts={"disagreeing_items": ["A::C1"]})],
    )
    plan, result = _apply(
        task,
        report,
        {
            fix_request_key("t-no-raise", "criterion_score::A::C1"): {
                "value": 1,
                "notes": "the criterion is met on turn 4",
            }
        },
    )
    assert plan.status == "fixable"
    assert result.patches == []


def test_apply_drops_gemini_named_winner_on_b_likert():
    """Likert 6 is B, but the writeup names Gemini (A). That shipped as UNSAFE."""
    task = make_task("t-gemini-flip", likert=6)
    task.model_a.declared_provider = "gemini"
    task.model_b.declared_provider = "chatgpt"
    task.sxs.justification = "Model B is better on the plan."
    report = _report("t-gemini-flip", "fail", [400, 470], [_check(400, "fail"), _check(470, "fail")])
    plan = plan_task(task, report)
    likert = next(j for j in plan.jobs if j.kind == "preference_likert")
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, likert.field_id): {
                "value": 6,
                "notes": "stay B",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "Gemini is much better because its turn 21 begins with Phase 0.",
                "notes": "product nickname",
            },
        },
    )
    assert not any(p.kind == "preference_justification" for p in result.patches)


def test_apply_drops_nearly_identical_writeup_on_winner_likert():
    task = make_task("t-equiv", likert=3)
    task.sxs.justification = "Model A is slightly better."
    report = _report("t-equiv", "fail", [400, 470], [_check(400, "fail"), _check(470, "fail")])
    plan = plan_task(task, report)
    likert = next(j for j in plan.jobs if j.kind == "preference_likert")
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, likert.field_id): {
                "value": 3,
                "notes": "stay A",
            },
            fix_request_key(task.task_id, just.field_id): {
                "value": "The two responses are nearly identical and equivalent.",
                "notes": "tie language",
            },
        },
    )
    assert not any(p.kind == "preference_justification" for p in result.patches)


def test_fetch_failed_slot_is_not_rewritten():
    task = make_task("t-fetch-fail")
    task.model_b.fetch_failed = True
    report = _report(
        "t-fetch-fail",
        "fail",
        [270, 280],
        [
            _check(270, "fail", counts={"disagreeing_items": ["A::C1", "B::C1"]}),
            _check(
                280,
                "fail",
                counts={
                    "reasons_by_item": {
                        "280::A::C1": ["incorrect_turns"],
                        "280::B::C1": ["incorrect_turns"],
                    },
                    "turns_by_item": {"280::A::C1": [3], "280::B::C1": [12]},
                },
            ),
        ],
    )
    plan = plan_task(task, report)
    assert {j.field_id for j in plan.jobs} == {
        "criterion_score::A::C1",
        "criterion_turns::A::C1",
    }


def test_relabel_c_to_b_promotes_gpt_and_keeps_ac_ranking():
    from honeybee_qc.remap_gpt_as_b import apply_relabel_c_to_b, narrow_to_ab_only

    task = make_task("relabel-1", n_criteria=2)
    task.project_id = ASP
    task.model_b = make_submission("B", CLAUDE, 8)
    task.model_c = make_submission("C", GPT, 8)
    task.criterion_ratings = [
        r for r in task.criterion_ratings if r.model != "B"
    ] + [
        CriterionRating(criterion_id=c.criterion_id, model="C", score=1)
        for c in task.rubric
    ]
    task.comparisons = [
        Sxs(likert=6, justification="A vs Claude", left="A", right="B"),
        Sxs(likert=2, justification="A vs GPT", left="A", right="C"),
    ]
    task.sxs = task.comparisons[0]
    apply_relabel_c_to_b([task], {"relabel-1"})
    narrow_to_ab_only(task)
    assert task.model_b is not None
    assert task.model_b.final_link == GPT
    assert task.model_b.model == "B"
    assert task.model_c is None
    assert task.sxs.right == "B"
    assert task.sxs.justification == "A vs GPT"
    b = next(r for r in task.criterion_ratings if r.model == "B")
    assert b.score == 1


def test_fixer_cli_accepts_relabel_c_to_b(tmp_path):
    task = make_task("cli-relabel", n_criteria=2)
    csv_path = write_batch(tmp_path, [task])
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "tasks": [
                    _report(
                        "cli-relabel",
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
    ids = tmp_path / "gpt_as_b.txt"
    ids.write_text("ffffffffffffffffffffffff\n", encoding="utf-8")
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
                "--relabel-c-to-b",
                str(ids),
            ]
        )
        == 0
    )
    assert (out_dir / "fixer_plan.json").is_file()


def test_hygiene_clears_na_turns_without_a_model_call():
    task = make_task("t-na-turns")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Tool & connector reliability":
            dim.not_applicable = True
            dim.justification = "N/A — no tools were used."
            dim.relevant_turns = [3, 7]
    plan = plan_task(task, _report("t-na-turns", "clean", [], []))
    assert plan.status == "fixable"
    jobs = [
        j
        for j in plan.jobs
        if j.kind == "dimension_turns"
        and j.slot == "A"
        and j.dimension == "Tool & connector reliability"
    ]
    assert jobs
    result = apply_plan(task, plan, {})
    patches = [p for p in result.patches if p.kind == "dimension_turns"]
    assert patches
    assert patches[0].new_value == []


def test_hygiene_fills_empty_non_na_turns():
    task = make_task("t-empty-turns")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.relevant_turns = []
    plan = plan_task(task, _report("t-empty-turns", "clean", [], []))
    job = next(
        j
        for j in plan.jobs
        if j.kind == "dimension_turns" and j.dimension == "Outcome quality"
    )
    result = apply_plan(
        task,
        plan,
        {fix_request_key(task.task_id, job.field_id): {"value": [5], "notes": "claim is on turn 5"}},
    )
    patches = [p for p in result.patches if p.kind == "dimension_turns"]
    assert patches[0].new_value == [5]


def test_hygiene_skips_empty_turns_on_a_five():
    task = make_task("t-empty-five")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.rating = 5
            dim.relevant_turns = []
    plan = plan_task(task, _report("t-empty-five", "clean", [], []))
    jobs = [
        j
        for j in plan.jobs
        if j.kind == "dimension_turns"
        and j.dimension == "Outcome quality"
        and j.slot == "A"
    ]
    assert jobs == []


def test_hygiene_skips_empty_turns_on_a_four():
    task = make_task("t-empty-four")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.rating = 4
            dim.relevant_turns = []
    plan = plan_task(task, _report("t-empty-four", "clean", [], []))
    jobs = [
        j
        for j in plan.jobs
        if j.kind == "dimension_turns"
        and j.dimension == "Outcome quality"
        and j.slot == "A"
    ]
    assert jobs == []


def test_empty_list_fills_from_turns_named_in_the_justification():
    task = make_task("t-empty-named")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.justification = "Turn 8 is where the clip is exported."
            dim.relevant_turns = []
            dim.rating = 3
    plan = plan_task(task, _report("t-empty-named", "clean", [], []))
    job = next(
        j
        for j in plan.jobs
        if j.kind == "dimension_turns"
        and j.dimension == "Outcome quality"
        and j.slot == "A"
    )
    result = apply_plan(task, plan, {})
    patches = [
        p
        for p in result.patches
        if p.kind == "dimension_turns" and p.dimension == "Outcome quality" and p.slot == "A"
    ]
    assert patches
    assert patches[0].new_value == [8]


def test_hygiene_flips_pfo_no_when_files_exist():
    task = make_task("t-pfo")
    task.model_a.produced_final_outcome = False
    plan = plan_task(task, _report("t-pfo", "clean", [], []))
    job = next(j for j in plan.jobs if j.kind == "produced_final_outcome")
    result = apply_plan(
        task,
        plan,
        {fix_request_key(task.task_id, job.field_id): {"value": True, "notes": "file was generated"}},
    )
    patches = [p for p in result.patches if p.kind == "produced_final_outcome"]
    assert patches[0].new_value is True


def test_hygiene_rewrites_tie_writeup_on_a_slight_win():
    task = make_task("t-sxs-3", likert=3)
    task.sxs.justification = "This is a tie; neither model is ahead."
    plan = plan_task(task, _report("t-sxs-3", "clean", [], []))
    kinds = {j.kind for j in plan.jobs}
    assert "preference_justification" in kinds
    assert "preference_likert" in kinds


def test_dim_score_patches_that_invert_ranking_are_dropped():
    task = make_task(
        "t-invert",
        dimension_rating_a=4,
        dimension_rating_b=4,
        likert=7,
    )
    task.sxs.justification = "Model B is much better than Model A on every dimension."
    jobs = []
    responses = {}
    for dimension in (
        "Communication quality",
        "Trust & grounding",
        "Tool & connector reliability",
    ):
        job = FieldJob(
            kind="dimension_rating",
            check_ids=(300,),
            slot="B",
            dimension=dimension,
            current_value=4,
            feedback="too high vs the conversation",
        )
        jobs.append(job)
        responses[fix_request_key(task.task_id, job.field_id)] = {
            "value": 2,
            "notes": "failed badly on this dimension",
        }
    plan = FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300], jobs=jobs)
    result = apply_plan(task, plan, responses)
    assert not any(p.kind == "dimension_rating" and not p.error for p in result.patches)


def test_dim_score_inversion_kept_when_writeup_explains_it():
    task = make_task(
        "t-invert-ok",
        dimension_rating_a=4,
        dimension_rating_b=4,
        likert=7,
    )
    task.sxs.justification = "Model B is much better."
    jobs = []
    responses = {}
    for dimension in (
        "Communication quality",
        "Trust & grounding",
        "Tool & connector reliability",
    ):
        job = FieldJob(
            kind="dimension_rating",
            check_ids=(300,),
            slot="B",
            dimension=dimension,
            current_value=4,
            feedback="too high",
        )
        jobs.append(job)
        responses[fix_request_key(task.task_id, job.field_id)] = {
            "value": 2,
            "notes": "failed badly on this dimension",
        }
    just = FieldJob(
        kind="preference_justification",
        check_ids=(460,),
        comparison="AB",
        current_value=task.sxs.justification,
        feedback="explain inversion",
    )
    jobs.append(just)
    responses[fix_request_key(task.task_id, just.field_id)] = {
        "value": (
            "Despite Model A's higher dimension scores, Model B is much better "
            "because of a decisive hallucination on the only required deliverable."
        ),
        "notes": "explains inversion",
    }
    plan = FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300, 460], jobs=jobs)
    result = apply_plan(task, plan, responses)
    assert any(p.kind == "dimension_rating" and p.new_value == 2 for p in result.patches)
    assert any(p.kind == "preference_justification" for p in result.patches)


def test_hygiene_rewrites_model_b_on_an_ac_pair():
    task = make_task("t-ac-slot", likert=7)
    task.model_c = make_submission("C", CLAUDE, 20)
    task.sxs.left = "A"
    task.sxs.right = "C"
    task.sxs.justification = "The Response B is much better than Model A."
    plan = plan_task(task, _report("t-ac-slot", "clean", [], []))
    assert any(j.kind == "preference_justification" for j in plan.jobs)
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, just.field_id): {
                "value": "The Response B is much better than Model A.",
                "notes": "kept the original body",
            }
        },
    )
    assert not any(
        p.kind == "preference_justification" and not p.error for p in result.patches
    )


def test_hygiene_drops_much_better_on_a_slight_win():
    task = make_task("t-mag", likert=5)
    task.sxs.justification = "Model B is much better than Model A on the file."
    plan = plan_task(task, _report("t-mag", "clean", [], []))
    assert any(j.kind == "preference_justification" for j in plan.jobs)
    just = next(j for j in plan.jobs if j.kind == "preference_justification")
    result = apply_plan(
        task,
        plan,
        {
            fix_request_key(task.task_id, just.field_id): {
                "value": "Model B is much better than Model A on the file.",
                "notes": "still says much",
            }
        },
    )
    just_patches = [
        p
        for p in result.patches
        if p.kind == "preference_justification" and not p.error
    ]
    assert just_patches
    assert "slightly better" in just_patches[0].new_value
    patched = patched_task(task, result.patches)
    assert "sxs_magnitude" not in {h.family for h in audit_task(patched)}


def test_dimension_jump_over_two_is_dropped():
    task = make_task("t-jump", dimension_rating_a=5, likert=4)
    job = FieldJob(
        kind="dimension_rating",
        check_ids=(300,),
        slot="A",
        dimension="Outcome quality",
        current_value=5,
        feedback="too high",
    )
    result = apply_plan(
        task,
        FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300], jobs=[job]),
        {fix_request_key(task.task_id, job.field_id): {"value": 1, "notes": "failed badly"}},
    )
    assert not any(p.kind == "dimension_rating" and not p.error for p in result.patches)


def test_dimension_jump_of_two_is_kept():
    task = make_task("t-jump-ok", dimension_rating_a=5, likert=4)
    job = FieldJob(
        kind="dimension_rating",
        check_ids=(300,),
        slot="A",
        dimension="Outcome quality",
        current_value=5,
        feedback="a bit high",
    )
    result = apply_plan(
        task,
        FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300], jobs=[job]),
        {fix_request_key(task.task_id, job.field_id): {"value": 3, "notes": "weaker than claimed"}},
    )
    assert any(p.kind == "dimension_rating" and p.new_value == 3 for p in result.patches)


def test_na_dimension_pastes_not_applicable_score():
    task = make_task("t-na-score")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Tool & connector reliability":
            dim.justification = "N/A — no tools were used."
            dim.not_applicable = False
            dim.rating = 5
            dim.relevant_turns = []
    plan = plan_task(task, _report("t-na-score", "clean", [], []))
    assert any(
        j.kind == "dimension_rating"
        and j.dimension == "Tool & connector reliability"
        for j in plan.jobs
    )
    result = apply_plan(task, plan, {})
    patches = [
        p
        for p in result.patches
        if p.kind == "dimension_rating"
        and p.dimension == "Tool & connector reliability"
    ]
    assert patches
    assert patches[0].new_value == "not-applicable"


def test_disjoint_justification_turns_replace_the_selected_list():
    task = make_task("t-disjoint")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.justification = "Turn 7 is where the clip is exported."
            dim.relevant_turns = [3]
    plan = plan_task(task, _report("t-disjoint", "clean", [], []))
    assert any(
        j.kind == "dimension_turns" and j.dimension == "Outcome quality"
        for j in plan.jobs
    )
    result = apply_plan(task, plan, {})
    patches = [
        p
        for p in result.patches
        if p.kind == "dimension_turns" and p.dimension == "Outcome quality"
    ]
    assert patches
    assert patches[0].new_value == [7]


def _unique_index_conversation(n: int = 20, prefix: str = "") -> list[Turn]:
    """Live-shaped transcript: odd indices are user, even are assistant."""
    turns: list[Turn] = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        turns.append(Turn(index=i, role=role, text=f"{prefix}{role} message {i}"))
    return turns


def test_hydrated_disjoint_turns_still_align_to_cited_assistant_turns():
    """_kept_turns used to drop the forced replacement when flagged_turns was empty."""
    task = make_task("t-hydrated-disjoint", exchanges=20)
    task.model_a.conversation = _unique_index_conversation(20, "A: ")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Trust & grounding":
            dim.justification = (
                "Turn 12 is where the claim is grounded; turn 1 is only the ask."
            )
            dim.relevant_turns = [16]
    job = FieldJob(
        kind="dimension_turns",
        check_ids=(310,),
        slot="A",
        dimension="Trust & grounding",
        current_value=[16],
        flagged_turns=(),
        feedback="selected disjoint from cited",
        optional=True,
    )
    result = apply_plan(
        task,
        FixPlan(task_id=task.task_id, status="fixable", fail_checks=[310], jobs=[job]),
        {},
    )
    patches = [
        p
        for p in result.patches
        if p.kind == "dimension_turns"
        and p.slot == "A"
        and p.dimension == "Trust & grounding"
    ]
    assert patches
    selected = {int(n) for n in patches[0].new_value}
    assert selected & {1, 12}
    assert patches[0].new_value
    patched = patched_task(task, result.patches)
    hits = [
        h
        for h in audit_task(patched)
        if h.family == "dimension_turns" and h.detail.startswith("A/Trust")
    ]
    assert hits == []


def test_justification_patch_realigns_selected_turns_to_the_new_writeup():
    task = make_task("t-just-realign", exchanges=20)
    task.model_a.conversation = _unique_index_conversation(20, "A: ")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.justification = "Turn 4 is the export."
            dim.relevant_turns = [4]
    just = FieldJob(
        kind="dimension_justification",
        check_ids=(300,),
        slot="A",
        dimension="Outcome quality",
        current_value="Turn 4 is the export.",
        feedback="rewrite cites",
    )
    result = apply_plan(
        task,
        FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300], jobs=[just]),
        {
            fix_request_key(task.task_id, just.field_id): {
                "value": "Turn 8 and turn 12 are where the clip is exported.",
                "notes": "new cites",
            }
        },
    )
    turns = [
        p
        for p in result.patches
        if p.kind == "dimension_turns"
        and p.slot == "A"
        and p.dimension == "Outcome quality"
    ]
    assert turns
    assert set(turns[0].new_value) & {8, 12}
    patched = patched_task(task, result.patches)
    dim = next(
        d
        for d in patched.dimension_ratings
        if d.model == "A" and d.dimension == "Outcome quality"
    )
    cited = {8, 12}
    assert set(dim.relevant_turns) & cited
    hits = [
        h
        for h in audit_task(patched)
        if h.family == "dimension_turns" and h.detail.startswith("A/Outcome quality")
    ]
    assert hits == []


def test_dropped_llm_tie_writeup_does_not_leave_sxs_scale():
    for likert in (5, 6):
        task = make_task(f"t-sxs-tie-{likert}", likert=likert)
        task.sxs.justification = (
            "This is a tie; both are comparable and essentially the same."
        )
        just = FieldJob(
            kind="preference_justification",
            check_ids=(400, 470),
            comparison="AB",
            current_value=task.sxs.justification,
            feedback="rewrite",
            optional=True,
        )
        result = apply_plan(
            task,
            FixPlan(
                task_id=task.task_id,
                status="fixable",
                fail_checks=[400, 470],
                jobs=[just],
            ),
            {},
        )
        just_patches = [
            p
            for p in result.patches
            if p.kind == "preference_justification" and not p.error
        ]
        assert just_patches, f"Likert {likert} left the original tie writeup"
        assert TIE_RE.search(sxs_body(just_patches[0].new_value)) is None
        patched = patched_task(task, result.patches)
        families = {h.family for h in audit_task(patched)}
        assert "sxs_scale" not in families


def test_likert_5_and_3_much_are_rewritten_not_left_dirty():
    cases = (
        (5, "Model B is much better than Model A on the file."),
        (3, "Model A is much better than Model B on the file."),
    )
    for likert, old in cases:
        task = make_task(f"t-mag-body-{likert}", likert=likert)
        task.sxs.justification = old
        just = FieldJob(
            kind="preference_justification",
            check_ids=(400, 470),
            comparison="AB",
            current_value=old,
            feedback="magnitude",
            optional=True,
        )
        result = apply_plan(
            task,
            FixPlan(
                task_id=task.task_id,
                status="fixable",
                fail_checks=[400, 470],
                jobs=[just],
            ),
            {
                fix_request_key(task.task_id, just.field_id): {
                    "value": old,
                    "notes": "still says much",
                }
            },
        )
        patched = patched_task(task, result.patches)
        families = {h.family for h in audit_task(patched)}
        assert "sxs_magnitude" not in families, families
        just_patches = [
            p
            for p in result.patches
            if p.kind == "preference_justification" and not p.error
        ]
        assert just_patches
        assert "slightly better" in just_patches[0].new_value


def test_dimension_rating_inversion_without_explaining_body_does_not_ship():
    task = make_task(
        "t-invert-no-explain",
        dimension_rating_a=4,
        dimension_rating_b=4,
        likert=7,
    )
    task.sxs.justification = "Model B is much better than Model A on every dimension."
    jobs = []
    responses = {}
    for dimension in (
        "Communication quality",
        "Trust & grounding",
        "Tool & connector reliability",
    ):
        job = FieldJob(
            kind="dimension_rating",
            check_ids=(300,),
            slot="B",
            dimension=dimension,
            current_value=4,
            feedback="too high vs the conversation",
        )
        jobs.append(job)
        responses[fix_request_key(task.task_id, job.field_id)] = {
            "value": 2,
            "notes": "failed badly on this dimension",
        }
    plan = FixPlan(task_id=task.task_id, status="fixable", fail_checks=[300], jobs=jobs)
    result = apply_plan(task, plan, responses)
    assert not any(p.kind == "dimension_rating" and not p.error for p in result.patches)
    patched = patched_task(task, result.patches)
    families = {h.family for h in audit_task(patched)}
    assert "inconsistent_ranking" not in families


def test_fixer_system_prompt_keeps_justification_voice():
    assert JUSTIFICATION_VOICE_PROMPT in SYSTEM_PROMPT
    text = JUSTIFICATION_VOICE_PROMPT
    assert "em dash" in text.lower()
    assert "Model A" in text and "Model B" in text and "Model C" in text
    assert 'Never "they"' in text
    assert "BAD" in text and "GOOD" in text
    assert "on turn 6 Model A still recommends the $38 line" in text
    assert "they still recommend the $38 line" not in text
    assert "pairwise outcome outweighs the mean" in text
    assert "Upon reviewing both trajectories" in text


def test_leftover_unexplained_inversion_does_not_get_canned_writeup():
    task = make_task(
        "t-leftover-invert",
        dimension_rating_a=1,
        dimension_rating_b=5,
        likert=2,
    )
    task.sxs.justification = "Model A is better than Model B on the file."
    plan = plan_task(task, _report("t-leftover-invert", "clean", [], []))
    result = apply_plan(task, plan, {})
    just_patches = [
        p
        for p in result.patches
        if p.kind == "preference_justification" and not p.error
    ]
    for patch in just_patches:
        body = patch.new_value or ""
        assert "pairwise outcome outweighs the mean" not in body
        assert "Even though the dimension scores favor" not in body
    patched = patched_task(task, result.patches)
    families = {h.family for h in audit_task(patched)}
    assert "inconsistent_ranking" in families
    leftover = result.unfixable_reason or ""
    assert "Inconsistent Ranking" in leftover
