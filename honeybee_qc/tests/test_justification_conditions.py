"""Check 450's `contradicts_verdict` condition, after the over-firing audit.

The condition fired on 16 justifications across 8 of 17 tasks and only one of
those readings survived scrutiny. Two failure modes accounted for most of the
rest and both are pinned here: the auditor counting a claim it could not
exhibit, and the auditor reading the spec's bipolar 1-7 Likert as a 1-5 scale
and so treating "a slight preference for Product B" as contradicting a 5.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.config import Policy
from honeybee_qc.findings import JustificationFinding
from honeybee_qc.informed_prompts import (
    JUSTIFICATION_SCHEMA,
    build_ranking_justification_prompt,
)
from honeybee_qc.informed_stages import parse_justifications
from honeybee_qc.llm import ModelResponse

from .fixtures import make_task


def _finding(item_id: str = "B::Communication quality", **kwargs) -> JustificationFinding:
    return JustificationFinding(item_id=item_id, **kwargs)


def test_unquoted_contradiction_does_not_trigger():
    """The whole of the "why not a 4 or 5?" family lands here.

    On task 6a7190c542368ece68aa2694 the auditor counted a contradiction on four
    justifications whose only fault was praising a model while rating it 3, and
    in every case it could point at no claim that argued the other way.
    """
    finding = _finding(rated_value=3, contradicts_verdict_claims=1)
    assert finding.contradicting_quotes == []
    assert finding.contradicts_verdict is False
    assert "contradicts_verdict" not in finding.triggered_conditions()
    assert finding.has_any_issue() is False


def test_quoted_contradiction_still_triggers():
    """The one reading that survived: a rating of 1 defended by "not applicable".

    Task 6a7190c542368ece68aa26a8, B::Tool & connector reliability -- the
    contributor gave the worst available score and the argument offered for it
    was that no connector was ever involved. That is a supporting claim pulling
    against its own verdict and it must keep failing.
    """
    finding = _finding(
        item_id="B::Tool & connector reliability",
        rated_value=1,
        contradicts_verdict_claims=1,
        contradicting_quotes=[
            "The model generated the output with fictional data, hence no "
            "connected service was used."
        ],
    )
    assert finding.contradicts_verdict is True
    assert "contradicts_verdict" in finding.triggered_conditions()
    assert finding.has_any_issue() is True


def test_blank_quotes_do_not_count_as_evidence():
    finding = _finding(contradicts_verdict_claims=2, contradicting_quotes=["", "   "])
    assert finding.contradicts_verdict is False


def test_parser_reads_contradicting_quotes():
    response = ModelResponse(
        key="t::ranking_justification",
        data={
            "justifications": [
                {
                    "item_id": "ranking",
                    "rated_value": 5,
                    "contradicts_verdict_claims": 1,
                    "is_generic": False,
                    "is_skewed": False,
                    "inaccurate_primary_claims": 0,
                    "inaccurate_secondary_claims": 0,
                    "unsupported_claims": 0,
                    "inaccurate_evidence": 0,
                    "misconstrued_evidence": 0,
                    "unverifiable_claims": 0,
                    "specifics_quoted": [],
                    "contradicting_quotes": ["  ", "it never used a connector"],
                    "reasoning": "",
                }
            ]
        },
    )
    (finding,) = parse_justifications(response)
    assert finding.contradicting_quotes == ["it never used a connector"]
    assert finding.contradicts_verdict is True


def test_schema_requires_the_quotes():
    props = JUSTIFICATION_SCHEMA["properties"]["justifications"]["items"]
    assert "contradicting_quotes" in props["properties"]
    assert "contradicting_quotes" in props["required"]


def test_ranking_prompt_states_the_likert_scale():
    """Two of the sixteen triggers were the auditor guessing the scale.

    It wrote "a Likert rating of 5 on a standard 1-5 scale represents the
    strongest possible preference" on two separate tasks, and failed both for
    calling their own 5 a slight lean toward Model B.
    """
    task = make_task()
    prompt = build_ranking_justification_prompt(task)
    assert "1-7" in prompt
    assert "strongest preference for Model A" in prompt
    assert "4 is a tie" in prompt


def test_d_vs_a_scale_note_puts_d_on_1():
    """A DA pair that still says 1 = Model A is the polarity bug."""
    from honeybee_qc.informed_prompts import _likert_scale_note

    note = _likert_scale_note(Policy(), "D", "A")
    assert "1 is the strongest preference for Model D" in note
    assert "the strongest for Model A" in note


def test_margin_encoding_describes_the_other_form():
    task = make_task()
    prompt = build_ranking_justification_prompt(
        task, Policy(preference_encoding="direction_magnitude")
    )
    assert "margin of 1-5" in prompt
    assert "strongest preference for Model A" not in prompt


def test_the_specs_per_condition_floors_are_unchanged_by_the_task_level_share():
    """The proportional task-level gate must not have leaked into the conditions.

    Every floor here is the spec's own except the subordinate-inaccuracy one, and
    each is a statement about one argument: one inaccurate load-bearing claim or
    one fabricated quote is enough, a misread of a real quote needs two. The
    share of justifications that must trip is a separate, and separately
    invented, question.

    The exception is deliberate and is pinned in its own test below, so a floor
    drifting from the spec by accident still fails here.
    """
    assert _finding(inaccurate_primary_claims=1).triggered_conditions() == ["is_inaccurate"]
    assert _finding(unsupported_claims=1).triggered_conditions() == []
    assert _finding(unsupported_claims=2).triggered_conditions() == ["lacks_evidence"]
    assert _finding(inaccurate_evidence=1).triggered_conditions() == [
        "cites_incorrect_evidence"
    ]
    assert _finding(misconstrued_evidence=1).triggered_conditions() == []
    assert _finding(misconstrued_evidence=2).triggered_conditions() == [
        "misconstrues_evidence"
    ]


def test_one_wrong_subordinate_claim_is_inaccurate_and_that_departs_from_the_spec():
    """The one floor in this check that is ours rather than the customer's.

    The auditor instructions say two subordinate errors, on the equivalence that
    two of them are worth one load-bearing error. We fail on one. The reason is
    in `findings.SECONDARY_INACCURACY_FLOOR`, and the consequence is that the
    severe escape in `justification_severe_conditions` is reachable from a single
    subordinate inaccuracy, so this is a wider door than the measurements quoted
    around that field were taken through.

    Pinned separately from the spec-conformance floors so that restoring the
    spec's reading is a one-line change with one failing test, rather than
    something that has to be rediscovered from the gate's behaviour.
    """
    from honeybee_qc.findings import SECONDARY_INACCURACY_FLOOR

    assert SECONDARY_INACCURACY_FLOOR == 1
    assert _finding(inaccurate_secondary_claims=1).triggered_conditions() == [
        "is_inaccurate"
    ]
    assert _finding(inaccurate_secondary_claims=2).triggered_conditions() == [
        "is_inaccurate"
    ]


def test_the_secondary_floor_is_not_advertised_to_the_judge():
    """The judge buckets by what the rating leans on, not by what it costs.

    `inaccurate_primary_claims` and `inaccurate_evidence` carry "One is enough to
    fail" in their own descriptions because the spec puts it there. The
    subordinate count must not acquire the same sentence just because its floor
    now matches: the prompt tells the judge outright that the weight of the
    consequence is the gate's business, and warns that a judge who knows the
    price of a bucket will stop writing findings down. That failure mode is the
    one no threshold downstream can recover from.
    """
    from honeybee_qc.informed_prompts import JUSTIFICATION_RUBRIC

    secondary = JUSTIFICATION_RUBRIC.split("**inaccurate_secondary_claims**", 1)[1]
    secondary = secondary.split("**unsupported_claims**", 1)[0]
    assert "needed to fail" not in secondary
    assert "One is enough" not in secondary

    # The two the spec does stamp keep their wording, so this is a statement
    # about the subordinate count and not a blanket ban on the phrase.
    assert "One is enough to fail" in JUSTIFICATION_RUBRIC


def test_the_task_level_share_travels_with_the_draw_agreement_bar():
    """`justification_fail_rate` has no counterpart in the spec, unlike every
    other threshold `test_spec_conformance` pins. If it ever acquires one, this
    test is the place that should stop being true.

    It went 0.30 -> 0.20 on its own evidence, and 0.20 -> 0.10 as one half of a
    pair. The other half is `justification_draw_agreement`: a defect now has to be
    reported by all five draws instead of three, which cuts the surviving population
    hard, and a share calibrated against a bare majority does not survive being
    applied to what is left of it. Neither number should be read, or moved, alone --
    swept jointly, the pair gives 38 task-level false fails against 42 published at
    unchanged recall, while the share alone at the old majority gives 43.
    """
    from honeybee_qc.config import DEFAULT_POLICY

    assert DEFAULT_POLICY.justification_fail_rate == 0.10
    assert DEFAULT_POLICY.justification_draw_agreement == 5
    assert DEFAULT_POLICY.samples_by_check[450] == 5, (
        "the bar is stated against the nominal draw count; if 450 stops drawing "
        "five samples, 5 becomes unanimity over however many it does draw"
    )
    assert DEFAULT_POLICY.justification_scope == "per_justification"


def test_the_severe_conditions_are_the_ones_the_spec_calls_one_is_enough():
    """The exemptions from that share, unlike the share itself, are the spec's.

    The auditor's instructions say "One is enough to fail" for exactly two
    counts -- the fabricated or misattributed citation and the false claim the
    rating rests on -- and nothing else in the condition list carries that
    wording, so nothing else escapes the share.

    `is_inaccurate` no longer names only those two, because its subordinate
    branch now has a floor of 1 as well. That widening is ours; what this test
    still holds is the membership of the tuple, which did not change with it.
    """
    from honeybee_qc.config import DEFAULT_POLICY

    assert DEFAULT_POLICY.justification_severe_conditions == (
        "is_inaccurate",
        "cites_incorrect_evidence",
    )
    assert _finding(inaccurate_primary_claims=1).triggered_conditions() == ["is_inaccurate"]
    assert _finding(inaccurate_evidence=1).triggered_conditions() == [
        "cites_incorrect_evidence"
    ]
    softer = (
        _finding(unsupported_claims=2),
        _finding(misconstrued_evidence=2),
        _finding(is_skewed=True),
        _finding(is_generic=True),
    )
    for finding in softer:
        assert not set(finding.triggered_conditions()) & set(
            DEFAULT_POLICY.justification_severe_conditions
        )


def test_cites_incorrect_evidence_is_reachable_and_must_not_be_treated_as_dead():
    """A correction to an earlier analysis, pinned so it cannot be re-derived wrong.

    That analysis observed severe=("cites_incorrect_evidence",) giving results
    byte-identical to severe=() on 151 tasks and concluded the condition never
    fires. It fires -- 7 items on 3 sheet-22 tasks and 12 items on 7 of 60
    `priority_batch` tasks. The identity proved only that it was never *decisive*,
    because every task carrying it also carried `is_inaccurate` or cleared the
    share honestly. Under `justification_draw_agreement = 5` one escalation of 60
    rests on it alone.

    So a single fabricated citation must reach the fail band with nothing beside
    it, at the share's own denominator, and with no help from `is_inaccurate`.
    """
    from honeybee_qc.config import DEFAULT_POLICY
    from honeybee_qc.gates import evaluate_check_450
    from honeybee_qc.tests.fixtures import make_task

    one_fabricated = _finding(inaccurate_evidence=1)
    assert one_fabricated.triggered_conditions() == ["cites_incorrect_evidence"]
    assert "is_inaccurate" not in one_fabricated.triggered_conditions()

    task = make_task()
    findings = [
        dataclasses.replace(one_fabricated, item_id="j0"),
        *(JustificationFinding(item_id=f"j{i}") for i in range(1, 15)),
    ]
    verdict = evaluate_check_450(task, findings)
    assert verdict.band == "fail", "1 of 15 is 6.7%, so only the severe route can hold"
    assert verdict.measurement.counts["severe_conditions_by_item"] == {
        "j0": ["cites_incorrect_evidence"]
    }
    # And it is the escape that carries it, not the share.
    assert verdict.measurement.rate is not None and verdict.measurement.rate < 0.10
    assert verdict.measurement.counts["failed_on_severity"] is True
