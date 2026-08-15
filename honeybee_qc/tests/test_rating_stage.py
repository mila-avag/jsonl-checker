"""Tests for the blind rating pass: checks 270, 300, 400.

Three properties matter more than the rest and are tested hardest:

  1. Abstentions leave the denominator, and a task with too few judgeable items
     returns not_evaluated rather than a rate over the survivors.
  2. No blind prompt exposes a contributor judgment.
  3. A failed or malformed call never becomes a disagreement.
"""

from __future__ import annotations

import dataclasses

import pytest

from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.context import (
    is_placeholder_turn,
    profile_evidence,
    render_comparison,
    render_conversation,
)
from honeybee_qc.llm import FakeModelClient, ModelRequest, ModelResponse
from honeybee_qc.models import Turn
from honeybee_qc.rating_prompts import (
    assert_blind,
    build_criterion_rating_prompt,
    build_dimension_adjudication_prompt,
    build_dimension_rating_prompt,
    build_likert_prompt,
)
from honeybee_qc.rating_stages import (
    build_rating_requests,
    estimate_rating_calls,
    run_rating_stage,
)
from honeybee_qc.taxonomies import RATING_DIMENSIONS
from honeybee_qc.tests.fixtures import make_task

# The exact text a real Gemini share page rendered for a turn that delivered a video.
REAL_PLACEHOLDER = "Your video is ready! 0:00 / 0:10"


# ---------------------------------------------------------------------------
# Artifact blindness
# ---------------------------------------------------------------------------


def test_real_delivering_turn_is_recognised_as_a_placeholder():
    turn = Turn(index=7, role="assistant", text=REAL_PLACEHOLDER)
    assert is_placeholder_turn(turn)


def test_a_long_substantive_turn_is_not_a_placeholder_even_if_it_mentions_creating():
    turn = Turn(
        index=3,
        role="assistant",
        text=(
            "I have created an outline for the animation. First the gyroscope is shown "
            "at rest, then a torque is applied to the spin axis, and the precession is "
            "traced with a dotted line so the direction of the resulting motion is "
            "readable at a glance. Each stage is colour coded consistently."
        ),
    )
    assert not is_placeholder_turn(turn)


def test_user_turns_are_never_placeholders():
    assert not is_placeholder_turn(Turn(index=1, role="user", text=REAL_PLACEHOLDER))


def test_placeholder_turns_are_labelled_and_counted():
    task = make_task(exchanges=4)
    sub = task.model_a
    sub.conversation[-1] = Turn(index=4, role="assistant", text=REAL_PLACEHOLDER)

    rendered, profile = render_conversation(sub)

    assert profile.placeholder_turns == 1
    assert profile.artifact_dependent
    assert "content is not in the transcript" in rendered
    assert "1 of 4 model turns" in rendered


def test_attachment_filenames_are_offered_without_implying_access_to_contents():
    task = make_task(exchanges=2)
    task.model_a.attachments = ["gyroscope_final.mp4"]
    rendered, _ = render_conversation(task.model_a)
    assert "gyroscope_final.mp4" in rendered
    assert "contents NOT available" in rendered


def test_turn_numbers_are_rendered_so_citations_can_be_compared():
    task = make_task(exchanges=3)
    rendered, _ = render_conversation(task.model_a)
    for i in (1, 2, 3):
        assert f"[turn {i}]" in rendered


def test_conversation_is_truncated_to_budget_and_says_so():
    task = make_task(exchanges=200)
    policy = dataclasses.replace(DEFAULT_POLICY, max_conversation_chars=2000)
    rendered, profile = render_conversation(task.model_a, policy)
    assert profile.truncated
    assert "truncated" in rendered
    assert len(rendered) < 4000


def test_one_enormous_turn_is_truncated_in_the_middle_keeping_both_ends():
    task = make_task(exchanges=1)
    task.model_a.conversation[-1] = Turn(
        index=1, role="assistant", text="START " + ("filler " * 5000) + "END"
    )
    policy = dataclasses.replace(DEFAULT_POLICY, max_turn_chars=500)
    rendered, _ = render_conversation(task.model_a, policy)
    assert "START" in rendered and "END" in rendered
    assert "characters omitted" in rendered


def test_comparison_splits_the_budget_so_a_long_model_a_cannot_crowd_out_model_b():
    task = make_task(exchanges=60)
    policy = dataclasses.replace(DEFAULT_POLICY, max_comparison_chars=4000)
    rendered, _ = render_comparison(task.model_a, task.model_b, policy)
    assert "## Model A" in rendered and "## Model B" in rendered
    a_block = rendered.split("## Model B")[0]
    b_block = rendered.split("## Model B")[1]
    # Neither side is starved: both got real content.
    assert len(a_block) > 500 and len(b_block) > 500


def test_evidence_profile_reports_ratio_of_judgeable_turns():
    task = make_task(exchanges=4)
    long_text = " ".join(["word"] * 60)
    task.model_a.conversation = [
        Turn(index=1, role="user", text="do it"),
        Turn(index=1, role="assistant", text=long_text),
        Turn(index=2, role="user", text="now the video"),
        Turn(index=2, role="assistant", text=REAL_PLACEHOLDER),
    ]
    profile = profile_evidence(task.model_a)
    assert profile.assistant_turns == 2
    assert profile.substantive_turns == 1
    assert profile.placeholder_turns == 1
    assert profile.substantive_ratio == 0.5


# ---------------------------------------------------------------------------
# Blindness of the prompts
# ---------------------------------------------------------------------------


def test_criterion_rating_prompt_hides_the_contributor_score():
    task = make_task(n_criteria=3, exchanges=3)
    prompt = build_criterion_rating_prompt(task, task.rubric[0], task.model_a)
    assert assert_blind(prompt, task).clean
    assert "contributor" not in prompt.lower()


def test_dimension_prompt_hides_the_contributor_rating_and_justification():
    task = make_task(n_criteria=3, exchanges=3)
    prompt = build_dimension_rating_prompt(task, RATING_DIMENSIONS[0], task.model_a)
    assert assert_blind(prompt, task).clean
    assert "because of X" not in prompt


def test_likert_prompt_hides_the_contributor_likert_and_its_justification():
    task = make_task(n_criteria=3, exchanges=3)
    prompt = build_likert_prompt(task)
    assert assert_blind(prompt, task).clean
    assert "I prefer Model A" not in prompt


def test_guard_catches_a_justification_leaking_through_any_field():
    task = make_task(n_criteria=2, exchanges=2)
    # A plausible regression: someone routes contributor context into user_goal.
    task.user_goal = f"Produce a teaching aid. {task.sxs.justification}"
    prompt = build_criterion_rating_prompt(task, task.rubric[0], task.model_a)
    report = assert_blind(prompt, task)
    assert not report.clean
    assert any("justification leaked" in m for m in report.leaked)


def test_guard_catches_contributor_framing_phrases():
    task = make_task(n_criteria=2, exchanges=2)
    report = assert_blind("The contributor rated this 7 out of 10.", task)
    assert not report.clean


def test_request_building_refuses_to_send_a_leaking_prompt():
    task = make_task(n_criteria=2, exchanges=2)
    task.user_goal = task.sxs.justification
    with pytest.raises(RuntimeError, match="exposes contributor judgments"):
        build_rating_requests(task)


# ---------------------------------------------------------------------------
# The abstention has to be reachable from the prompt, not just from the schema
# ---------------------------------------------------------------------------


def test_the_blind_prompts_name_the_markers_they_must_abstain_on():
    """`cannot_determine` existed before this and was still used on 12 of 305 calls
    while a quarter of them argued from a cut they could see. The gap was that the
    prompts never quoted the two strings the renderer writes into the transcript,
    so a judge had to infer that the marker was the abstention trigger."""
    task = make_task(n_criteria=3, exchanges=3)
    prompts = [
        build_criterion_rating_prompt(task, task.rubric[0], task.model_a),
        build_dimension_rating_prompt(task, RATING_DIMENSIONS[0], task.model_a),
        build_likert_prompt(task),
    ]
    for prompt in prompts:
        assert "[... conversation truncated here;" in prompt
        assert "[DELIVERABLE PRODUCED HERE" in prompt
        assert "why_unverifiable" in prompt
        assert assert_blind(prompt, task).clean


def test_naming_the_markers_did_not_cost_the_prompts_their_blindness():
    """The instruction talks about what the judge was and was not shown, which is
    exactly the register in which a contributor value could slip in. The guard is
    the reason this text is safe to add, so it is asserted on the real prompts with
    a task whose contributor fields are populated."""
    task = make_task(n_criteria=6, exchanges=4, likert=7)
    for prompt in (
        build_criterion_rating_prompt(task, task.rubric[2], task.model_b),
        build_dimension_rating_prompt(task, RATING_DIMENSIONS[1], task.model_b),
        build_likert_prompt(task),
    ):
        report = assert_blind(prompt, task)
        assert report.clean, report.leaked
        assert "contributor" not in prompt.lower()


def test_an_abstention_reports_what_was_missing_rather_than_its_reasoning():
    """The abstention is only actionable if it names the evidence, because raising
    the render budget for the conversations that need it is the response, and that
    decision needs to know which span was cut."""
    task = make_task(n_criteria=2, exchanges=3)

    def respond(req: ModelRequest):
        check_id = req.metadata.get("check_id")
        if check_id == 270:
            return {
                "criterion_id": req.metadata["criterion_id"],
                "verdict": "cannot_determine",
                "cannot_determine": True,
                "why_unverifiable": "the slide deck's content is not in the transcript",
                "reasoning": "long prose a reviewer would have to read",
                "confidence": "high",
            }
        if check_id == 300:
            return {
                "dimension": req.metadata["dimension"],
                "rating": None,
                "cannot_determine": True,
                "why_unverifiable": "exchanges 9-14 are past the truncation marker",
                "reasoning": "long prose a reviewer would have to read",
                "confidence": "low",
            }
        return {
            "likert": None,
            "cannot_determine": True,
            "why_unverifiable": "both deliverables are absent from the transcript",
            "reasoning": "long prose a reviewer would have to read",
            "confidence": "low",
        }

    result = run_rating_stage(task, FakeModelClient(respond))

    reasons = {a.check_id: a.reason for a in result.abstentions}
    assert reasons[270] == "the slide deck's content is not in the transcript"
    assert reasons[300] == "exchanges 9-14 are past the truncation marker"
    assert reasons[400] == "both deliverables are absent from the transcript"
    assert all(
        "long prose" not in a.reason for a in result.abstentions
    )
    # And the abstentions still cost the contributor nothing on any of the three.
    assert {_verdict(result, c).band for c in (270, 300, 400)} == {"not_evaluated"}
    assert _verdict(result, 270).measurement.counts["judged"] == 0


def test_an_abstention_without_a_named_gap_still_falls_back_to_the_reasoning():
    """Not every model will fill the field. An abstention with a vague reason is
    still an abstention -- what it must never do is become a disagreement."""
    task = make_task(n_criteria=4, exchanges=3)

    def respond(req: ModelRequest):
        if req.metadata.get("check_id") != 270:
            return {
                "dimension": req.metadata.get("dimension"),
                "rating": 4,
                "evidence_turns": [1],
                "confidence": "high",
            }
        cid = req.metadata["criterion_id"]
        if cid != "C1":
            return {
                "criterion_id": cid,
                "verdict": "pass",
                "evidence_turns": [1],
                "reasoning": "turn 1 satisfies it",
                "confidence": "high",
            }
        return {
            "criterion_id": cid,
            "verdict": "cannot_determine",
            "reasoning": "the video is not viewable here",
            "confidence": "low",
        }

    result = run_rating_stage(task, FakeModelClient(respond))
    verdict = _verdict(result, 270)
    assert [a.reason for a in result.abstentions if a.check_id == 270] == [
        "the video is not viewable here"
    ] * 2
    assert verdict.measurement.denominator == 3
    assert verdict.measurement.counts["disagreeing_criteria"] == 0
    assert verdict.band == "clean"


# ---------------------------------------------------------------------------
# Request volume
# ---------------------------------------------------------------------------


def test_one_call_per_judgment_and_nothing_is_batched():
    task = make_task(n_criteria=20, exchanges=5)
    requests = build_rating_requests(task)
    # 20 criteria x 2 models, 6 dimensions x 2 models, 1 comparison.
    assert len(requests) == 40 + 12 + 1 == estimate_rating_calls(task)
    assert len({r.key for r in requests}) == len(requests)


def test_a_single_model_task_produces_no_comparison_call():
    task = make_task(n_criteria=4, exchanges=3)
    task.model_b = None
    requests = build_rating_requests(task)
    assert len(requests) == 4 + 6
    assert not any(r.metadata.get("check_id") == 400 for r in requests)


# ---------------------------------------------------------------------------
# Stage behaviour
# ---------------------------------------------------------------------------


def _responder(
    *,
    criterion=lambda req: "pass",
    dimension=lambda req: 4,
    likert=4,
):
    """Fake client whose per-check answers are supplied by callables."""

    def respond(req: ModelRequest):
        check_id = req.metadata.get("check_id")
        if check_id == 270:
            verdict = criterion(req)
            return {
                "criterion_id": req.metadata["criterion_id"],
                "verdict": verdict,
                "evidence_turns": [1],
                "reasoning": "because of turn 1",
                "confidence": "high",
            }
        if check_id == 300:
            value = dimension(req)
            return {
                "dimension": req.metadata["dimension"],
                "rating": value if isinstance(value, int) else None,
                "not_applicable": value == "na",
                "cannot_determine": value == "cannot_determine",
                "evidence_turns": [2],
                "reasoning": "because of turn 2",
                "confidence": "medium",
            }
        return {
            "likert": likert if isinstance(likert, int) else None,
            "cannot_determine": likert == "cannot_determine",
            "reasoning": "the two are close",
            "confidence": "high",
        }

    return FakeModelClient(respond)


def _verdict(result, check_id):
    return next(v for v in result.verdicts if v.check_id == check_id)


def test_full_agreement_is_clean_on_all_three_checks():
    task = make_task(n_criteria=10, exchanges=4)
    client = _responder()
    result = run_rating_stage(task, client, workers=4)

    assert not result.errors
    assert _verdict(result, 270).band == "clean"
    assert _verdict(result, 300).band == "clean"
    assert _verdict(result, 400).band == "clean"
    assert result.calls == estimate_rating_calls(task)


def test_a_criterion_disagreed_on_for_both_models_counts_once():
    task = make_task(n_criteria=10, exchanges=4)
    # Auditor fails C1 for both models: one criterion of 10, two judgments of 20.
    client = _responder(
        criterion=lambda req: "fail" if req.metadata["criterion_id"] == "C1" else "pass"
    )
    result = run_rating_stage(task, client)

    verdict = _verdict(result, 270)
    assert verdict.measurement.counts["disagreeing_criteria"] == 1
    assert verdict.measurement.counts["disagreeing_judgments"] == 2
    assert verdict.measurement.denominator == 10
    assert verdict.band == "non_fail"
    assert set(verdict.contributing_items) == {"C1"}
    # The per-model detail the criterion count folds away is still reported.
    assert set(verdict.measurement.counts["disagreeing_items"]) == {"A::C1", "B::C1"}


def test_excluding_abstentions_makes_the_rate_threshold_more_sensitive():
    """A documented consequence of shrinking the denominator.

    One disagreeing criterion out of the 10 the contributor wrote is 10% and does
    not fail. The same criterion out of the 5 the auditor could judge is 20% and
    does. That is the intended reading -- a fifth of what could be checked was
    wrong -- and the judged-ratio floor is what stops the base from shrinking
    without limit.
    """
    task = make_task(n_criteria=10, exchanges=4)
    # Both models abstain on C6-C10, so those criteria leave the denominator.
    abstaining = {f"C{i}" for i in range(6, 11)}

    def criterion(req):
        cid = req.metadata["criterion_id"]
        if cid in abstaining:
            return "cannot_determine"
        return "fail" if cid == "C1" else "pass"

    verdict = _verdict(run_rating_stage(task, _responder(criterion=criterion)), 270)
    assert verdict.measurement.denominator == 5
    assert verdict.measurement.rate == pytest.approx(0.20)
    assert verdict.band == "fail"
    assert verdict.measurement.counts["judged_ratio"] == 0.5


def test_three_disagreeing_criteria_of_ten_passes_the_rate_threshold():
    """30% of the criteria, so the rate leg fails the task while the absolute leg
    of 5 criteria stays silent. The absolute leg is exercised at gate level, where
    a rubric large enough for it to bind first does not cost 50-odd model calls."""
    task = make_task(n_criteria=10, exchanges=4)
    failing = {"C1", "C2", "C3"}
    client = _responder(
        criterion=lambda req: "fail" if req.metadata["criterion_id"] in failing else "pass"
    )
    result = run_rating_stage(task, client)
    verdict = _verdict(result, 270)
    assert verdict.measurement.counts["disagreeing_criteria"] == 3
    assert verdict.measurement.counts["disagreeing_judgments"] == 6
    assert verdict.measurement.rate == pytest.approx(0.30)
    assert verdict.band == "fail"


def test_abstentions_leave_the_denominator_rather_than_counting_either_way():
    task = make_task(n_criteria=10, exchanges=4)
    # Half the criteria concern the video, which the transcript does not contain.
    abstaining = {f"C{i}" for i in range(1, 6)}
    client = _responder(
        criterion=lambda req: (
            "cannot_determine" if req.metadata["criterion_id"] in abstaining else "pass"
        )
    )
    result = run_rating_stage(task, client)

    verdict = _verdict(result, 270)
    assert result.abstained(270) == 10
    assert verdict.measurement.denominator == 5  # the 5 criteria that survived
    assert verdict.measurement.counts["abstained"] == 10
    assert verdict.measurement.counts["disagreeing_criteria"] == 0
    assert verdict.band == "clean"
    assert "every model's judgment on it abstained" in verdict.measurement.notes


def test_a_mostly_unauditable_task_is_not_evaluated_instead_of_scored():
    task = make_task(n_criteria=10, exchanges=4)
    abstaining = {f"C{i}" for i in range(1, 9)}  # 16 of 20 judgments abstain
    client = _responder(
        criterion=lambda req: (
            "cannot_determine" if req.metadata["criterion_id"] in abstaining else "pass"
        )
    )
    result = run_rating_stage(task, client)

    verdict = _verdict(result, 270)
    assert verdict.band == "not_evaluated"
    assert verdict.score is None
    assert verdict.confidence == "low"
    assert verdict.measurement.counts == {"judged": 4, "abstained": 16}
    assert "deliverables are not present" in verdict.measurement.notes


def test_abstaining_on_every_criterion_is_not_evaluated_not_clean():
    task = make_task(n_criteria=5, exchanges=3)
    client = _responder(criterion=lambda req: "cannot_determine")
    result = run_rating_stage(task, client)
    verdict = _verdict(result, 270)
    assert verdict.band == "not_evaluated"
    assert verdict.measurement.counts["judged"] == 0


def test_dimension_abstention_is_excluded_and_reported():
    task = make_task(n_criteria=3, exchanges=3)
    outcome = RATING_DIMENSIONS[0]
    client = _responder(
        dimension=lambda req: (
            "cannot_determine" if req.metadata["dimension"] == outcome else 4
        )
    )
    result = run_rating_stage(task, client)

    verdict = _verdict(result, 300)
    assert result.abstained(300) == 2
    assert verdict.measurement.counts["abstained"] == 2
    assert len(result.dimension_findings) == 10


def test_dimension_not_applicable_is_a_judgment_not_an_abstention():
    task = make_task(n_criteria=3, exchanges=3)
    memory = "Memory & personalization"
    client = _responder(
        dimension=lambda req: "na" if req.metadata["dimension"] == memory else 7
    )
    result = run_rating_stage(task, client)

    assert result.abstained(300) == 0
    na_findings = [f for f in result.dimension_findings if f.dimension == memory]
    assert len(na_findings) == 2
    assert all(f.auditor_na and f.auditor_rating is None for f in na_findings)
    # Contributor rated it, auditor said N/A: a mismatch, not silence.
    assert all(f.na_mismatch for f in na_findings)


def test_dimension_rating_outside_the_scale_is_an_error_not_a_disagreement():
    task = make_task(n_criteria=3, exchanges=3)
    client = _responder(dimension=lambda req: 99)
    result = run_rating_stage(task, client)

    assert len(result.dimension_findings) == 0
    assert len(result.errors) == 12
    assert all("outside the 1-5 scale" in e for e in result.errors)
    assert _verdict(result, 300).band == "not_evaluated"


def test_likert_delta_of_three_fails_check_400():
    task = make_task(n_criteria=3, exchanges=3, likert=4)
    client = _responder(likert=7)
    result = run_rating_stage(task, client)
    verdict = _verdict(result, 400)
    assert verdict.measurement.counts["delta"] == 3
    assert verdict.band == "fail"


def test_likert_abstention_yields_not_evaluated_with_the_reason():
    task = make_task(n_criteria=3, exchanges=3)
    client = _responder(likert="cannot_determine")
    result = run_rating_stage(task, client)
    verdict = _verdict(result, 400)
    assert verdict.band == "not_evaluated"
    assert "abstained" in verdict.measurement.notes


def test_a_failed_call_is_an_error_and_never_a_disagreement():
    task = make_task(n_criteria=5, exchanges=3)

    def respond(req: ModelRequest):
        if req.metadata.get("check_id") == 270:
            # No metadata echoed back, as a launch failure would do.
            return ModelResponse(key=req.key, error="timeout after 300s")
        return _responder().complete(req).data

    result = run_rating_stage(task, FakeModelClient(respond))

    assert len(result.criterion_findings) == 0
    assert len(result.errors) == 10
    assert all("timeout" in e for e in result.errors)
    assert _verdict(result, 270).band == "not_evaluated"
    # The other two checks still report.
    assert _verdict(result, 300).band == "clean"
    assert _verdict(result, 400).band == "clean"


def test_an_unknown_verdict_string_is_rejected():
    task = make_task(n_criteria=2, exchanges=3)
    client = _responder(criterion=lambda req: "probably fine")
    result = run_rating_stage(task, client)
    assert len(result.criterion_findings) == 0
    assert all("unknown verdict" in e for e in result.errors)


def test_a_criterion_the_contributor_never_rated_is_dropped_not_counted():
    task = make_task(n_criteria=5, exchanges=3)
    task.criterion_ratings = [r for r in task.criterion_ratings if r.criterion_id != "C1"]
    client = _responder()
    result = run_rating_stage(task, client)

    assert len(result.criterion_findings) == 8
    assert len(result.dropped) == 2
    assert not result.errors


def test_a_task_with_no_submissions_is_not_evaluated_without_spending_calls():
    task = make_task(n_criteria=5, exchanges=3)
    task.model_a = None
    task.model_b = None
    client = _responder()
    result = run_rating_stage(task, client)

    assert client.calls == []
    assert {v.check_id for v in result.verdicts} == {270, 300, 400}
    assert all(v.band == "not_evaluated" for v in result.verdicts)


def test_cost_and_cache_hits_are_accumulated():
    task = make_task(n_criteria=3, exchanges=3)

    def respond(req: ModelRequest):
        base = _responder().complete(req)
        return ModelResponse(
            key=req.key,
            data=base.data,
            cost_usd=0.01,
            cached=req.metadata.get("check_id") == 300,
            metadata=dict(req.metadata),
        )

    result = run_rating_stage(task, FakeModelClient(respond))
    assert result.calls == 6 + 12 + 1
    assert result.cost_usd == pytest.approx(0.19)
    assert result.cached_calls == 12


def test_summary_dict_reports_abstentions_and_evidence_per_model():
    task = make_task(n_criteria=3, exchanges=3)
    client = _responder(criterion=lambda req: "cannot_determine")
    summary = run_rating_stage(task, client).to_dict()

    assert summary["abstentions"]["270"] == 6
    assert len(summary["evidence"]) == 2
    assert {e["model"] for e in summary["evidence"]} == {"A", "B"}


# ---------------------------------------------------------------------------
# Informed adjudication, and the labels that say what a fail rests on
# ---------------------------------------------------------------------------
#
# A manual audit of twelve check-300 fails found seven were not defects: a blind
# judge's number three points from the contributor's, on a subjective five-point
# scale, sometimes read from a conversation the render budget had cut in half.
# These tests pin the two things that changed. A rating gap alone no longer fails
# a task, and whatever happens to a gap is named in the output.


def _adjudicating_responder(
    *,
    dimension=lambda req: 4,
    likert=4,
    adjudication="indefensible",
    adjudication_why="turn 3 shows the opposite",
):
    """A fake client that answers the blind pass and the confirming pass both."""

    def respond(req: ModelRequest):
        if "::adjudicate::" in req.key:
            verdict = (
                adjudication(req) if callable(adjudication) else adjudication
            )
            return {
                "verdict": verdict,
                "why": adjudication_why,
                "evidence_turns": [3],
                "why_unverifiable": (
                    "the deliverable is not in the transcript"
                    if verdict == "cannot_determine"
                    else ""
                ),
                "confidence": "high",
            }
        check_id = req.metadata.get("check_id")
        if check_id == 270:
            return {
                "criterion_id": req.metadata["criterion_id"],
                "verdict": "pass",
                "evidence_turns": [1],
                "why_unverifiable": "",
                "reasoning": "because of turn 1",
                "confidence": "high",
            }
        if check_id == 300:
            value = dimension(req)
            return {
                "dimension": req.metadata["dimension"],
                "rating": value if isinstance(value, int) else None,
                "not_applicable": value == "na",
                "cannot_determine": value == "cannot_determine",
                "why_unverifiable": "",
                "evidence_turns": [2],
                "reasoning": "because of turn 2",
                "confidence": "medium",
            }
        return {
            "likert": likert if isinstance(likert, int) else None,
            "cannot_determine": likert == "cannot_determine",
            "why_unverifiable": "",
            "reasoning": "the two are close",
            "confidence": "high",
        }

    return FakeModelClient(respond)


def test_a_confirmed_rating_gap_still_fails_check_300():
    """The fix must not make 300 unable to fire."""
    task = make_task(n_criteria=4, exchanges=4, dimension_rating_a=5, dimension_rating_b=5)
    # Blind rater says 1 where the contributor said 5: a 4-point gap on every
    # dimension, and an informed reader agrees the 5 is indefensible.
    client = _adjudicating_responder(dimension=lambda req: 1, adjudication="indefensible")

    result = run_rating_stage(task, client)
    verdict = _verdict(result, 300)

    assert verdict.band == "fail"
    counts = verdict.measurement.counts
    assert counts["counted_disagreements"] == counts["rating_gaps_found"]
    assert counts["blind_only_not_counted"] == 0
    assert counts["blind_disagreement_only"] is False
    assert "every one of them confirmed" in verdict.measurement.notes


def test_a_gap_the_informed_pass_calls_defensible_does_not_fail_check_300():
    task = make_task(n_criteria=4, exchanges=4, dimension_rating_a=5, dimension_rating_b=5)
    client = _adjudicating_responder(
        dimension=lambda req: 1,
        adjudication="defensible",
        adjudication_why="both readings are supportable on this evidence",
    )

    result = run_rating_stage(task, client)
    verdict = _verdict(result, 300)

    # Same blind deltas as the test above, opposite verdict.
    assert verdict.band == "clean"
    counts = verdict.measurement.counts
    assert counts["counted_disagreements"] == 0
    assert counts["blind_only_not_counted"] == counts["rating_gaps_found"] > 0
    assert counts["blind_disagreement_only"] is True
    assert "NO confirmed rating gaps" in verdict.measurement.notes
    # Every dropped gap is still named, so nothing is silently discarded.
    assert counts["blind_only_items"]
    for item in counts["blind_only_items"]:
        assert "adjudicated_defensible" in counts["reasons_by_item"][item]


def test_an_abstained_adjudication_leaves_the_gap_uncounted_and_labelled():
    task = make_task(n_criteria=4, exchanges=4, dimension_rating_a=5, dimension_rating_b=5)
    client = _adjudicating_responder(
        dimension=lambda req: 1, adjudication="cannot_determine"
    )

    result = run_rating_stage(task, client)
    counts = _verdict(result, 300).measurement.counts

    assert counts["counted_disagreements"] == 0
    assert counts["unadjudicated_not_counted"] > 0
    assert counts["blind_disagreement_only"] is True


def test_a_gap_read_from_a_truncated_conversation_is_not_counted():
    task = make_task(n_criteria=2, exchanges=6, dimension_rating_a=5, dimension_rating_b=5)
    # A budget this small cuts every conversation, which is the condition the
    # recorded batch-9 fails were produced under and never disclosed.
    policy = dataclasses.replace(DEFAULT_POLICY, max_conversation_chars=200)
    client = _adjudicating_responder(dimension=lambda req: 1, adjudication="indefensible")

    result = run_rating_stage(task, client, policy=policy)
    counts = _verdict(result, 300).measurement.counts

    # Even with the informed pass agreeing: both passes read the same cut
    # conversation, so the second does not independently check the first.
    assert counts["counted_disagreements"] == 0
    assert counts["truncated_evidence_not_counted"] > 0
    for item in counts["truncated_evidence_items"]:
        assert "truncated_evidence" in counts["reasons_by_item"][item]


def test_truncation_reaches_the_finding_at_all():
    """The plumbing this all rests on: the render's own state, on the judgment."""
    task = make_task(n_criteria=1, exchanges=6)
    policy = dataclasses.replace(DEFAULT_POLICY, max_conversation_chars=200)
    client = _adjudicating_responder()

    result = run_rating_stage(task, client, policy=policy)

    assert result.dimension_findings
    assert all(f.truncated for f in result.dimension_findings)
    assert all(f.truncated for f in result.criterion_findings)


def test_a_full_render_leaves_the_truncation_flag_off():
    task = make_task(n_criteria=1, exchanges=4)
    result = run_rating_stage(task, _adjudicating_responder())

    assert result.dimension_findings
    assert not any(f.truncated for f in result.dimension_findings)


def test_agreement_spends_nothing_on_adjudication():
    """The confirming pass is priced on what is actually in dispute."""
    task = make_task(n_criteria=4, exchanges=4)
    client = _adjudicating_responder()  # blind rating matches the contributor's 4

    result = run_rating_stage(task, client)

    assert result.calls == estimate_rating_calls(task)
    assert not any(f.adjudication for f in result.dimension_findings)


def test_adjudication_prompts_are_exempt_from_the_blindness_guard_but_the_blind_ones_are_not():
    """The two passes are different questions and must stay separable.

    The confirming prompt shows the contributor's rating on purpose -- that is the
    whole point of it -- so it cannot go through `assert_blind`. The blind prompts
    still must.
    """
    task = make_task(n_criteria=2, exchanges=4)
    for request in build_rating_requests(task):
        assert assert_blind(request.prompt, task).clean

    prompt, _ = build_dimension_adjudication_prompt(
        task,
        RATING_DIMENSIONS[0],
        task.model_a,
        contributor_rating=5,
        contributor_justification=task.dimension_ratings[0].justification,
        auditor_rating=2,
    )
    # Deliberately not blind: it carries the contributor's own justification.
    assert not assert_blind(prompt, task).clean
    assert "5" in prompt


def test_an_asymmetric_side_by_side_render_is_not_counted_as_a_ranking_defect():
    """One sampled fail ranked a one-exchange Model B against an eight-exchange A."""
    task = make_task(n_criteria=1, exchanges=6, likert=1)
    # Model B's turns are long enough that the shared budget cuts it far sooner.
    task.model_b.conversation = [
        Turn(index=i, role=("user" if i % 2 else "assistant"), text="word " * 4000)
        for i in range(1, 7)
    ]
    policy = dataclasses.replace(DEFAULT_POLICY, max_comparison_chars=12000)
    client = _adjudicating_responder(likert=7, adjudication="indefensible")

    result = run_rating_stage(task, client, policy=policy)
    verdict = _verdict(result, 400)
    counts = verdict.measurement.counts

    assert counts["delta"] >= policy.likert_fail_delta
    assert counts["render_asymmetric"] is True
    assert counts["not_counted_because"] == "asymmetric_render"
    assert counts["counted_delta"] == 0
    assert verdict.band != "fail"
    assert "comparable depth" in verdict.measurement.notes


def test_a_ranking_gap_the_informed_pass_calls_defensible_is_labelled_not_failed():
    task = make_task(n_criteria=1, exchanges=4, likert=1)
    client = _adjudicating_responder(likert=7, adjudication="defensible")

    verdict = _verdict(run_rating_stage(task, client), 400)
    counts = verdict.measurement.counts

    assert counts["delta"] == 6
    assert counts["counted_delta"] == 0
    assert counts["not_counted_because"] == "blind_only"
    assert counts["blind_disagreement_only"] is True
    assert verdict.band == "clean"
    assert "found the preference defensible" in verdict.measurement.notes


def test_a_confirmed_ranking_gap_still_fails_check_400():
    task = make_task(n_criteria=1, exchanges=4, likert=1)
    client = _adjudicating_responder(likert=7, adjudication="indefensible")

    verdict = _verdict(run_rating_stage(task, client), 400)

    assert verdict.band == "fail"
    assert verdict.measurement.counts["counted_delta"] == 6
    assert verdict.measurement.counts["not_counted_because"] == ""


def test_the_comparison_discloses_how_much_of_each_side_was_shown():
    task = make_task(n_criteria=1, exchanges=6)
    task.model_b.conversation = [
        Turn(index=i, role=("user" if i % 2 else "assistant"), text="word " * 4000)
        for i in range(1, 7)
    ]
    policy = dataclasses.replace(DEFAULT_POLICY, max_comparison_chars=12000)

    rendered, profiles = render_comparison(task.model_a, task.model_b, policy)

    assert "How much of each conversation you are being shown" in rendered
    assert "turns shown" in rendered
    assert "not comparable" in rendered
    assert profiles[0].turns_rendered != profiles[1].turns_rendered


def test_unspent_budget_on_one_side_is_handed_to_the_other():
    """A short Model A should stop costing Model B the depth it did not need."""
    task = make_task(n_criteria=1, exchanges=8)
    task.model_a.conversation = [Turn(index=1, role="user", text="short")]
    task.model_b.conversation = [
        Turn(index=i, role=("user" if i % 2 else "assistant"), text="word " * 300)
        for i in range(1, 9)
    ]
    policy = dataclasses.replace(DEFAULT_POLICY, max_comparison_chars=12000)

    _, profiles = render_comparison(task.model_a, task.model_b, policy)
    b = next(p for p in profiles if p.model == "B")

    _, half_budget_only = render_conversation(
        task.model_b, policy, max_chars=policy.max_comparison_chars // 2
    )

    # On its own half of the budget B loses most of the conversation; handed the
    # slack a one-turn Model A left behind, all of it survives.
    assert half_budget_only.truncated and half_budget_only.turns_rendered == 3
    assert not b.truncated
    assert b.turns_rendered == b.turns_available == 8


# ---------------------------------------------------------------------------
# The exported chat PDF, as the fallback when a share page yields no replies
# ---------------------------------------------------------------------------
#
# `ingest` has always recorded `model_{a|b}_chat_download` and nothing read it. A
# submission whose share page did not render therefore reached the blind judges as
# a user-turn-only skeleton, and every judgment against it was an abstention or a
# guess. In batch 9, three slot-B submissions carried no share link at all while
# carrying the PDF, and 293 of 296 submissions carry one as a safety net.


def _pdf_policy(**overrides):
    return dataclasses.replace(
        DEFAULT_POLICY, fetch_attachment_text=True, **overrides
    )


def test_the_chat_pdf_stands_in_when_the_share_page_yielded_no_model_replies(monkeypatch):
    import honeybee_qc.context as context

    monkeypatch.setattr(
        context,
        "fetch_attachment_text",
        lambda url, policy, mime_type="": (
            "User: make me a gyroscope animation. Assistant: here is the clip, "
            "rendered at 4K with consistent colour coding."
            if url.endswith(".pdf")
            else ""
        ),
    )
    task = make_task(exchanges=4)
    sub = dataclasses.replace(
        task.model_a,
        conversation=[Turn(index=1, role="user", text="make me a gyroscope animation")],
        transcript_pdf="https://example.test/chat.pdf",
        attachments=[],
    )

    rendered, profile = render_conversation(sub, _pdf_policy())

    assert profile.pdf_fallback_used
    assert "consistent colour coding" in rendered
    # The judge must not number turns off a print of the page.
    assert "NOT read turn numbers off it" in rendered
    # And the old "you are seeing only the user's side" note is no longer the
    # right thing to say, because it is no longer true.
    assert "the model's replies were not fetched" not in rendered


def test_the_pdf_fallback_stays_out_of_the_way_when_the_replies_are_present(monkeypatch):
    import honeybee_qc.context as context

    called: list[str] = []

    def fake(url, policy, mime_type=""):
        called.append(url)
        return "should not be read"

    monkeypatch.setattr(context, "fetch_attachment_text", fake)
    task = make_task(exchanges=4)
    sub = dataclasses.replace(
        task.model_a, transcript_pdf="https://example.test/chat.pdf", attachments=[]
    )

    _, profile = render_conversation(sub, _pdf_policy())

    assert not profile.pdf_fallback_used
    assert "https://example.test/chat.pdf" not in called


def test_the_pdf_fallback_is_off_without_the_fetch_flag_or_its_own_policy_flag(monkeypatch):
    """It reads from the network, so it obeys the flag every other fetch obeys."""
    import honeybee_qc.context as context

    monkeypatch.setattr(
        context, "fetch_attachment_text", lambda url, policy, mime_type="": "transcript text"
    )
    sub = dataclasses.replace(
        make_task(exchanges=4).model_a,
        conversation=[Turn(index=1, role="user", text="hello")],
        transcript_pdf="https://example.test/chat.pdf",
        attachments=[],
    )

    _, no_fetch = render_conversation(sub, DEFAULT_POLICY)
    _, opted_out = render_conversation(sub, _pdf_policy(transcript_pdf_fallback=False))

    assert not no_fetch.pdf_fallback_used
    assert not opted_out.pdf_fallback_used


def test_an_unreadable_pdf_falls_back_to_the_old_user_turns_only_warning(monkeypatch):
    import honeybee_qc.context as context

    monkeypatch.setattr(
        context, "fetch_attachment_text", lambda url, policy, mime_type="": "   "
    )
    sub = dataclasses.replace(
        make_task(exchanges=4).model_a,
        conversation=[Turn(index=1, role="user", text="hello")],
        transcript_pdf="https://example.test/chat.pdf",
        attachments=[],
    )

    rendered, profile = render_conversation(sub, _pdf_policy())

    assert not profile.pdf_fallback_used
    assert "the model's replies were not fetched" in rendered
