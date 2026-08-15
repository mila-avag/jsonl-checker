"""Check 450's `contradicts_verdict` condition, after the over-firing audit.

The condition fired on 16 justifications across 8 of 17 tasks and only one of
those readings survived scrutiny. Two failure modes accounted for most of the
rest and both are pinned here: the auditor counting a claim it could not
exhibit, and the auditor reading the spec's bipolar 1-7 Likert as a 1-5 scale
and so treating "a slight preference for Product B" as contradicting a 5.
"""

from __future__ import annotations

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


def test_margin_encoding_describes_the_other_form():
    task = make_task()
    prompt = build_ranking_justification_prompt(
        task, Policy(preference_encoding="direction_magnitude")
    )
    assert "margin of 1-5" in prompt
    assert "strongest preference for Model A" not in prompt


def test_the_specs_per_condition_floors_are_unchanged_by_the_task_level_share():
    """The proportional task-level gate must not have leaked into the conditions.

    Each floor here is the spec's own and is a statement about one argument: one
    inaccurate load-bearing claim or one fabricated quote is enough, a
    subordinate error or a misread of a real quote needs two. The share of
    justifications that must trip is a separate, and separately invented,
    question.
    """
    assert _finding(inaccurate_primary_claims=1).triggered_conditions() == ["is_inaccurate"]
    assert _finding(inaccurate_secondary_claims=1).triggered_conditions() == []
    assert _finding(inaccurate_secondary_claims=2).triggered_conditions() == ["is_inaccurate"]
    assert _finding(unsupported_claims=1).triggered_conditions() == []
    assert _finding(unsupported_claims=2).triggered_conditions() == ["lacks_evidence"]
    assert _finding(inaccurate_evidence=1).triggered_conditions() == [
        "cites_incorrect_evidence"
    ]
    assert _finding(misconstrued_evidence=1).triggered_conditions() == []
    assert _finding(misconstrued_evidence=2).triggered_conditions() == [
        "misconstrues_evidence"
    ]


def test_the_task_level_share_is_declared_as_ours():
    """`justification_fail_rate` has no counterpart in the spec, unlike every
    other threshold `test_spec_conformance` pins. If it ever acquires one, this
    test is the place that should stop being true."""
    from honeybee_qc.config import DEFAULT_POLICY

    assert DEFAULT_POLICY.justification_fail_rate == 0.30
    assert DEFAULT_POLICY.justification_scope == "per_justification"
