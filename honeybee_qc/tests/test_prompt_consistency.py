"""Tests for check 75, Prompt / Pre-seeded/Opening Prompt Consistency.

The check exists because the customer's QC cites it in the field under its own
bracketed label, and it is net-new the same way 70 is: tab 2's custom block
assigns it no rubric position. Two properties carry the risk, and both are
properties sibling checks got wrong:

  1. The bar is underlying intent, not wording. Paraphrasing, a reworded
     opening, and added context (a persona, a reason for asking) are all
     expected and must land in the non-fail band, not the fail band. Only a
     genuinely different subject or request may fail.
  2. Evidence is mandatory, the same discipline 70 and 450's `contradicts_verdict`
     apply. A mismatch (or a match) nobody can quote from both sides is not a
     finding, and must not score against the contributor.

The calibration cases below are real, dated QC citations (both from
`audit_runs/qc_validations_source_20260812.csv`), not synthetic stand-ins:
task `6a7190c542368ece68aa26f7` (fail) and task `6a7190c542368ece68aa265e`
(non-fail). Both prompt texts are pulled verbatim from the taskattempt export
via the `data-source-form-prefiller-41acb4777bed` and prompt-step fields this
check reads.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.errors import ERROR_CODES
from honeybee_qc.findings import PromptConsistencyFinding
from honeybee_qc.gates import evaluate_check_75
from honeybee_qc.informed_prompts import build_prompt_consistency_prompt
from honeybee_qc.informed_stages import (
    build_informed_requests,
    estimate_informed_calls,
    parse_prompt_consistency,
    run_informed_stage,
)
from honeybee_qc.llm import FakeModelClient, ModelResponse
from honeybee_qc.registry import ORDER, REGISTRY
from honeybee_qc.tests.fixtures import make_task

# ---------------------------------------------------------------------------
# Real calibration text, task 6a7190c542368ece68aa26f7 -- FAIL
# ---------------------------------------------------------------------------

FAIL_PRE_SEEDED = (
    '"2033 Leavenworth HOA - write a detailed timeline of events and also an '
    "analysis of the breaches of Davis Stirling Act in California, Corporate "
    "Law in California, and the CC&R's. Also, what about side garden "
    'maintenance and watering? Insider dealing/voting?"'
)
FAIL_SUBMITTED = (
    "I hate to be a pain but I think I may have found a serious problem in "
    "the Vantage S-1 matter, but I want to be careful before I call it one. "
    "VantageS1_AnalystSummaryMemo_v3.docx from June 19 says the commercial "
    "exhibits are fully reconciled and scrutiny-ready, and Vikram's reply "
    "that day says it is exactly what the GC needs. But one of the entries "
    'marked "Reconciled" is the Filament/Meridian MSA at a $2.8 million ACV, '
    "sourced to the April 29 executed MSA.\n\n"
    "That is bothering me because Meridian was matter 26-0074 for Filament, "
    "not Vantage, and I remember us being very deliberate about keeping the "
    "Filament work inside that matter team. The Vantage S-1 review was "
    "separately being treated as confidential to its own matter team.\n\n"
    "Can you actually go through the record carefully and help me work out "
    "what actually happened here, what the Vantage record can still support "
    "on its own, and what I should worried about before anyone relies on "
    "that S-1 work again?"
)

# ---------------------------------------------------------------------------
# Real calibration text, task 6a7190c542368ece68aa265e -- NON-FAIL
# ---------------------------------------------------------------------------

NON_FAIL_PRE_SEEDED = (
    "Educational physics animation on dark background. Step-by-step 2D "
    'diagram of a gyroscope: Step 1 (0\u20133s) \u2014 a simple labeled circle '
    'labeled "FLYWHEEL" with curved rotation arrows showing high-speed spin, '
    'RPM counter "9,750 RPM" appears beside it. Step 2 (3\u20137s) \u2014 a square '
    '"GIMBAL FRAME" appears around the flywheel; an external blue force '
    "arrow pushes down on the frame from the left. Step 3 (7\u201311s) \u2014 the "
    "gimbal tilts on the input axis, and a perpendicular bright amber arrow "
    'labeled "PRECESSION FORCE" shoots out 90 degrees to the input force. A '
    "short equation appears: F_input \u2192 F_precession (\u22a5). Clean flat 2D "
    "motion graphics, color-coded: blue = input force, amber = precession "
    "output. No photorealism \u2014 pure physics explainer diagram style. "
    "11-second clip, 4K."
)
NON_FAIL_SUBMITTED = (
    "I'm a physics teacher, and I want to show my students a short "
    "animation explaining how a gyroscope works, specifically why pushing "
    "on it makes it tilt sideways instead of the way you'd expect. Can you "
    "make me one? It should build up in three steps:\n\n"
    'First (roughly 0\u20133s), just the spinning part: a simple circle labeled '
    '"FLYWHEEL" with curved arrows around it showing it spinning fast, and a '
    'little counter next to it reading "9,750 RPM".\n\n'
    'Then (3\u20137s), a square labeled "GIMBAL FRAME" appears around the '
    "flywheel, and a blue arrow pushes down on the frame from the left \u2014 "
    "that's the external force.\n\n"
    'Finally (7\u201311s), the frame tilts, and a bright amber arrow labeled '
    '"PRECESSION FORCE" shoots out at 90 degrees to the blue one. I want the '
    "color coding to be consistent so students immediately see blue = what "
    "you push, amber = what actually happens.\n\n"
    "The whole thing should come out to an 11-second clip, and I'd like it "
    "in 4K so it holds up on the classroom projector."
)

# The exact, verbatim reasoning cited by human QC for each real calibration
# task, reused here as the expected reasoning per the calibration requirement.
NON_FAIL_QC_REASONING = (
    "The opening question of the conversations is not the same as the "
    "pre-seeded prompt but has the same core intent: The pre-seeded prompt "
    'begins "Educational physics animation on dark background..." while the '
    "user prompt begins by establishing a context \"I'm a physics teacher, "
    "and I want to show my students a short animation explaining how a "
    'gyroscope works..." but the rest of the prompt contains the same core '
    "information/intent from the pre-seeded prompt."
)
FAIL_QC_REASONING = "The preseeded prompt and input prompt are completely different."


def preseeded_task(pre_seeded: str | None = "a pre-seeded prompt"):
    return dataclasses.replace(make_task(), pre_seeded_prompt=pre_seeded)


def _real_calibration_task(pre_seeded: str, submitted: str):
    """A task whose *rendered* prompt (what the judge is shown) is the real
    calibration text on both sides, isolating the submission from the
    fixture's default synthetic conversation turns."""
    task = dataclasses.replace(
        make_task(), pre_seeded_prompt=pre_seeded, seeded_prompt=submitted, prompts=[]
    )
    for sub in (task.model_a, task.model_b):
        sub.conversation = []
    return task


def _response(data: dict) -> ModelResponse:
    return ModelResponse(key="k", data=data)


def finding(assessment: str, **kwargs) -> PromptConsistencyFinding:
    """An evidenced finding by default; pass empty quotes to strip the evidence."""
    return PromptConsistencyFinding(
        assessment=assessment,  # type: ignore[arg-type]
        pre_seeded_quote=kwargs.pop("pre_seeded_quote", "the pre-seed's core ask"),
        submitted_quote=kwargs.pop("submitted_quote", "the submission's core ask"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def test_a_verbatim_match_is_clean():
    verdict = evaluate_check_75(preseeded_task(), finding("same"))
    assert verdict.band == "clean"
    assert verdict.score == 5
    assert verdict.error_code is None


def test_only_the_different_assessment_can_ever_fail_a_task():
    bands = {
        assessment: evaluate_check_75(preseeded_task(), finding(assessment)).band
        for assessment in ("same", "reworded_same_intent", "different")
    }
    assert bands == {
        "same": "clean",
        "reworded_same_intent": "non_fail",
        "different": "fail",
    }
    assert [a for a, b in bands.items() if b == "fail"] == ["different"]


# ---------------------------------------------------------------------------
# The real calibration cases
# ---------------------------------------------------------------------------


def test_the_real_fail_citation_task_6a7190c542368ece68aa26f7():
    """Human QC: 'The preseeded prompt and input prompt are completely
    different.', cited as `[Fail - Pre-seeded/Opening Prompts Inconsistency]`
    (plural "Prompts" in the fail label -- reproduced verbatim). The two prompts
    here are on entirely unrelated subjects: an HOA/CC&R timeline versus a legal
    matter-conflict memo."""
    task = dataclasses.replace(
        preseeded_task(FAIL_PRE_SEEDED), seeded_prompt=FAIL_SUBMITTED
    )
    result = finding(
        "different",
        pre_seeded_quote="write a detailed timeline of events and also an "
        "analysis of the breaches of Davis Stirling Act",
        submitted_quote="I think I may have found a serious problem in the "
        "Vantage S-1 matter",
        reasoning=FAIL_QC_REASONING,
    )
    verdict = evaluate_check_75(task, result)
    assert verdict.band == "fail"
    assert verdict.score == 1
    assert verdict.error_code == "[Fail - Pre-seeded/Opening Prompts Inconsistency]"
    assert verdict.error_code == ERROR_CODES[75]["fail"]
    assert verdict.contributing_items == [result.pre_seeded_quote, result.submitted_quote]


def test_the_real_non_fail_citation_task_6a7190c542368ece68aa265e():
    """Human QC's own verbatim reasoning, cited as `[Non-Fail - Pre-seeded/
    Opening Prompt Inconsistency]` (singular "Prompt", unlike the fail label --
    the inconsistency is the human's own and is preserved on purpose). The
    submitted prompt opens by establishing a teacher persona before restating
    the pre-seed's gyroscope-animation ask in its own words."""
    task = dataclasses.replace(
        preseeded_task(NON_FAIL_PRE_SEEDED), seeded_prompt=NON_FAIL_SUBMITTED
    )
    result = finding(
        "reworded_same_intent",
        pre_seeded_quote="Educational physics animation on dark background",
        submitted_quote="I'm a physics teacher, and I want to show my "
        "students a short animation explaining how a gyroscope works",
        reasoning=NON_FAIL_QC_REASONING,
    )
    verdict = evaluate_check_75(task, result)
    assert verdict.band == "non_fail"
    assert verdict.score == 3
    assert verdict.error_code == "[Non-Fail - Pre-seeded/Opening Prompt Inconsistency]"
    assert verdict.error_code == ERROR_CODES[75]["non_fail"]
    assert "same core intent" in result.reasoning


def test_the_two_calibration_labels_are_inconsistently_pluralised_on_purpose():
    """The human's own labeling differs between bands ("Prompts" vs. "Prompt"),
    and normalising either would break the join against the customer's own
    QC export, the same reason 240's stray "+" is preserved verbatim."""
    assert ERROR_CODES[75]["fail"] == "[Fail - Pre-seeded/Opening Prompts Inconsistency]"
    assert ERROR_CODES[75]["non_fail"] == "[Non-Fail - Pre-seeded/Opening Prompt Inconsistency]"
    assert "Prompts" in ERROR_CODES[75]["fail"]
    assert "Prompts" not in ERROR_CODES[75]["non_fail"]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_an_allegation_quoting_neither_side_cannot_stand():
    verdict = evaluate_check_75(
        preseeded_task(), finding("different", pre_seeded_quote="", submitted_quote="")
    )
    assert verdict.band == "clean"
    assert verdict.measurement.counts["evidenced"] is False
    assert "Discarded here" in verdict.measurement.notes


def test_quoting_only_one_side_is_not_enough():
    pre_seed_only = finding("different", submitted_quote="")
    submitted_only = finding("different", pre_seeded_quote="")
    assert evaluate_check_75(preseeded_task(), pre_seed_only).band == "clean"
    assert evaluate_check_75(preseeded_task(), submitted_only).band == "clean"


def test_a_failing_verdict_carries_both_quotes_as_its_contributing_items():
    verdict = evaluate_check_75(preseeded_task(), finding("different"))
    assert verdict.contributing_items == [
        "the pre-seed's core ask",
        "the submission's core ask",
    ]


def test_missing_evidence_on_a_same_finding_changes_nothing():
    verdict = evaluate_check_75(
        preseeded_task(), finding("same", pre_seeded_quote="", submitted_quote="")
    )
    assert verdict.band == "clean"
    assert "Discarded here" not in verdict.measurement.notes


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


def test_no_pre_seeded_prompt_is_not_evaluated_rather_than_clean():
    verdict = evaluate_check_75(preseeded_task(pre_seeded=None), finding("same"))
    assert verdict.band == "not_evaluated"
    assert verdict.score is None
    assert "nothing to compare the submitted prompt against" in verdict.measurement.notes


def test_a_whitespace_only_pre_seed_counts_as_absent():
    assert evaluate_check_75(preseeded_task(pre_seeded="   "), None).band == "not_evaluated"


def test_an_unaudited_task_is_not_evaluated():
    verdict = evaluate_check_75(preseeded_task(), None)
    assert verdict.band == "not_evaluated"
    assert verdict.confidence == "low"


def test_no_call_is_built_when_there_is_no_pre_seed_to_compare_against():
    task = make_task()
    assert task.pre_seeded_prompt is None
    keys = {r.metadata["check_id"] for r in build_informed_requests(task)}
    assert 75 not in keys
    assert 75 in {r.metadata["check_id"] for r in build_informed_requests(preseeded_task())}


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------


def test_the_prompt_shows_both_texts_and_the_intent_bar():
    """Also confirms the calibration fixtures are real text pulled from the
    taskattempt export, not synthetic stand-ins: both prompts survive verbatim
    into the exact prompt the judge would be shown."""
    task = _real_calibration_task(NON_FAIL_PRE_SEEDED, NON_FAIL_SUBMITTED)
    text = build_prompt_consistency_prompt(task)
    assert "Educational physics animation on dark background" in text
    assert "9,750 RPM" in text
    assert "PRECESSION FORCE" in text
    assert "I'm a physics teacher" in text
    assert "underlying request" in text
    assert "not whether the wording matches" in text
    # The bar this check has to hold: added context is not a substitution.
    assert "reworded_same_intent" in text


def test_the_parser_carries_both_quotes():
    parsed = parse_prompt_consistency(
        _response(
            {
                "assessment": "reworded_same_intent",
                "pre_seeded_quote": "the pre-seed's ask",
                "submitted_quote": "the submission's ask",
                "reasoning": "same underlying request, reworded opening.",
                "confidence": "medium",
            }
        )
    )
    assert parsed.assessment == "reworded_same_intent"
    assert parsed.pre_seeded_quote == "the pre-seed's ask"
    assert parsed.submitted_quote == "the submission's ask"
    assert parsed.confidence == "medium"
    assert parsed.is_issue and not parsed.is_fail


def test_an_unrecognised_assessment_falls_back_to_same():
    """A malformed enum must not manufacture a fail on a check that publishes a
    fail label."""
    parsed = parse_prompt_consistency(
        _response(
            {
                "assessment": "totally different thing",
                "pre_seeded_quote": "x",
                "submitted_quote": "y",
            }
        )
    )
    assert parsed.assessment == "same"
    assert evaluate_check_75(preseeded_task(), parsed).band == "clean"


# ---------------------------------------------------------------------------
# Registration and accounting
# ---------------------------------------------------------------------------


def test_the_dimension_is_registered_so_it_appears_in_every_report():
    assert 75 in REGISTRY
    assert 75 in ORDER
    assert REGISTRY[75].dimension == "Prompt"
    assert REGISTRY[75].sub_dimension == "Pre-seeded/Opening Prompt Consistency"
    assert REGISTRY[75].shape == "C"
    assert "Net-new" in REGISTRY[75].notes


def test_the_call_estimator_counts_the_consistency_call_only_when_it_will_run():
    without = estimate_informed_calls(make_task())
    with_preseed = estimate_informed_calls(preseeded_task())
    assert with_preseed == without + 1


def test_the_stage_reports_a_verdict_for_75_on_a_task_with_a_pre_seed():
    client = FakeModelClient(
        lambda r: (
            {
                "assessment": "reworded_same_intent",
                "pre_seeded_quote": "Educational physics animation on dark background",
                "submitted_quote": "I'm a physics teacher",
                "reasoning": NON_FAIL_QC_REASONING,
                "confidence": "high",
            }
            if r.metadata.get("check_id") == 75
            else {"confidence": "high"}
        )
    )
    result = run_informed_stage(preseeded_task(NON_FAIL_PRE_SEEDED), client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 75)
    assert verdict.band == "non_fail"
    assert result.prompt_consistency is not None
    assert result.to_dict()["prompt_consistency"]["assessment"] == "reworded_same_intent"


def test_the_stage_reports_75_as_not_evaluated_when_the_call_fails():
    client = FakeModelClient(lambda r: ModelResponse(key=r.key, error="upstream timeout"))
    result = run_informed_stage(preseeded_task(), client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 75)
    assert verdict.band == "not_evaluated"
    assert result.prompt_consistency is None


def test_the_stage_reports_75_as_not_evaluated_on_a_task_with_no_pre_seed():
    result = run_informed_stage(make_task(), FakeModelClient(lambda r: {"confidence": "high"}), workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 75)
    assert verdict.band == "not_evaluated"
