"""Tests for check 70, Prompt / Domain Relevance.

The check exists because the customer's QC cites `[Fail - Domain Relevance
Prompt]` in the field, and it is net-new: tab 1 names the dimension and states
the 5-of-5 bar, but tab 2 gives it no rubric position.

Two properties carry the risk, and both are properties sibling checks got wrong:

  1. The fail bar is "egregious". Real prompts sit at the edges of their domains
     as a matter of course, so a gate that fails on any imperfect overlap fails
     most of the corpus. Only `unrelated` may reach the fail band.
  2. Evidence is mandatory. A misalignment nobody can quote from both sides is
     not a finding, and must not score against the contributor.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.errors import ERROR_CODES
from honeybee_qc.findings import DomainRelevanceFinding
from honeybee_qc.gates import evaluate_check_70
from honeybee_qc.informed_prompts import build_domain_relevance_prompt
from honeybee_qc.informed_stages import (
    build_informed_requests,
    estimate_informed_calls,
    parse_domain_relevance,
    run_informed_stage,
)
from honeybee_qc.llm import FakeModelClient, ModelResponse
from honeybee_qc.registry import ORDER, REGISTRY
from honeybee_qc.tests.fixtures import make_task

# The persona and the prompt in the fixture are drawn from the real export: an
# educator commissioning a physics teaching aid.
DOMAIN = "Educator and instructional designer"


def assigned_task(domain: str = DOMAIN, category: str = "daily life specialist"):
    return dataclasses.replace(make_task(), assigned_domain=domain, prompt_category=category)


def _response(data: dict) -> ModelResponse:
    return ModelResponse(key="k", data=data)


def finding(assessment: str, **kwargs) -> DomainRelevanceFinding:
    """An evidenced finding by default; pass empty quotes to strip the evidence."""
    return DomainRelevanceFinding(
        assessment=assessment,  # type: ignore[arg-type]
        assigned_domain=DOMAIN,
        prompt_quote=kwargs.pop("prompt_quote", "explaining how a gyroscope works"),
        domain_basis=kwargs.pop("domain_basis", DOMAIN),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def test_a_prompt_matching_its_assigned_domain_is_clean():
    verdict = evaluate_check_70(assigned_task(), finding("aligned"))
    assert verdict.band == "clean"
    assert verdict.score == 5
    assert verdict.error_code is None


def test_a_genuinely_different_domain_fails_with_the_cited_spec_label():
    task = assigned_task(domain="Backend software engineer")
    verdict = evaluate_check_70(
        task,
        DomainRelevanceFinding(
            assessment="unrelated",
            assigned_domain="Backend software engineer",
            prompt_quote="diagnose this patient's retroperitoneal fibrosis",
            domain_basis="Backend software engineer",
            reasoning="The prompt is a clinical radiology question.",
        ),
    )
    assert verdict.band == "fail"
    assert verdict.score == 1
    assert verdict.error_code == "[Fail - Domain Relevance Prompt]"
    assert verdict.error_code == ERROR_CODES[70]["fail"]


def test_mild_topical_drift_is_a_non_fail_and_never_a_fail():
    """The spec word is "egregious". A prompt leaning toward a neighbouring
    speciality has not left its domain, so it cannot reach the fail band."""
    verdict = evaluate_check_70(assigned_task(), finding("partial"))
    assert verdict.band == "non_fail"
    assert verdict.score == 3
    assert verdict.error_code == "[Non-Fail - Domain Relevance]"


def test_only_the_unrelated_assessment_can_ever_fail_a_task():
    bands = {
        assessment: evaluate_check_70(assigned_task(), finding(assessment)).band
        for assessment in ("aligned", "partial", "unrelated")
    }
    assert bands == {"aligned": "clean", "partial": "non_fail", "unrelated": "fail"}
    assert [a for a, b in bands.items() if b == "fail"] == ["unrelated"]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_an_allegation_quoting_neither_side_cannot_stand():
    verdict = evaluate_check_70(
        assigned_task(), finding("unrelated", prompt_quote="", domain_basis="")
    )
    assert verdict.band == "clean"
    assert verdict.measurement.counts["evidenced"] is False
    assert "Discarded here" in verdict.measurement.notes


def test_quoting_only_one_side_is_not_enough():
    """Naming a mismatch requires the text on both sides that disagrees; one
    quote alone shows what the prompt says but not what it failed to match."""
    prompt_only = finding("unrelated", domain_basis="")
    domain_only = finding("unrelated", prompt_quote="")
    assert evaluate_check_70(assigned_task(), prompt_only).band == "clean"
    assert evaluate_check_70(assigned_task(), domain_only).band == "clean"


def test_a_failing_verdict_carries_both_quotes_as_its_contributing_items():
    verdict = evaluate_check_70(assigned_task(), finding("unrelated"))
    assert verdict.contributing_items == ["explaining how a gyroscope works", DOMAIN]


def test_missing_evidence_on_an_aligned_finding_changes_nothing():
    """The evidence bar guards against invented issues, so a clean answer that
    quotes nothing is still clean rather than not_evaluated."""
    verdict = evaluate_check_70(
        assigned_task(), finding("aligned", prompt_quote="", domain_basis="")
    )
    assert verdict.band == "clean"
    assert "Discarded here" not in verdict.measurement.notes


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


def test_no_assigned_domain_is_not_evaluated_rather_than_clean():
    verdict = evaluate_check_70(assigned_task(domain=""), finding("aligned"))
    assert verdict.band == "not_evaluated"
    assert verdict.score is None
    assert "nothing to measure the prompt against" in verdict.measurement.notes


def test_a_whitespace_only_domain_counts_as_absent():
    assert evaluate_check_70(assigned_task(domain="   "), None).band == "not_evaluated"


def test_an_unaudited_prompt_is_not_evaluated():
    verdict = evaluate_check_70(assigned_task(), None)
    assert verdict.band == "not_evaluated"
    assert verdict.confidence == "low"


def test_no_call_is_built_when_there_is_no_domain_to_compare_against():
    task = make_task()
    assert task.assigned_domain == ""
    keys = {r.metadata["check_id"] for r in build_informed_requests(task)}
    assert 70 not in keys
    assert 70 in {r.metadata["check_id"] for r in build_informed_requests(assigned_task())}


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------


def test_the_prompt_shows_the_assignment_and_holds_the_egregious_bar():
    text = build_domain_relevance_prompt(assigned_task())
    assert DOMAIN in text
    assert "daily life specialist" in text
    assert "egregious" in text
    # The bar the sibling checks got wrong: breadth is not misalignment, and a
    # plausible user of a subject is working within their domain.
    assert "adjacent" in text
    assert "plausible *user*" in text
    assert "clearly related to the assigned domain" in text


def test_the_prompt_says_no_category_when_none_was_assigned():
    text = build_domain_relevance_prompt(assigned_task(category=""))
    assert "(none assigned)" in text


# ---------------------------------------------------------------------------
# A pre-seeded scenario discarded wholesale is a domain violation too, not
# just a check-75 problem: the assigned domain was commissioned through that
# specific scenario, and swapping it for an unconnected one leaves the
# persona-level label satisfied by coincidence, not by anything the seed did.
# ---------------------------------------------------------------------------


def test_no_pre_seed_leaves_the_prompt_unchanged():
    task = assigned_task()
    task.pre_seeded_prompt = None
    text = build_domain_relevance_prompt(task)
    assert "pre-seeded" not in text.lower()


def test_a_pre_seed_is_shown_and_scenario_survival_is_the_bar():
    task = assigned_task()
    task.pre_seeded_prompt = "Explain orbital mechanics for a Mars transfer window."
    text = build_domain_relevance_prompt(task)
    assert "Explain orbital mechanics for a Mars transfer window." in text
    assert "abandons this scenario" in text
    assert "scenario survived" in text


def test_a_blank_pre_seed_is_treated_the_same_as_none():
    task = assigned_task()
    task.pre_seeded_prompt = "   "
    text = build_domain_relevance_prompt(task)
    assert "pre-seeded" not in text.lower()


def test_the_parser_carries_both_quotes_and_the_assignment_it_judged():
    parsed = parse_domain_relevance(
        _response(
            {
                "assessment": "partial",
                "prompt_quote": "configure stockpiles in Dwarf Fortress",
                "domain_basis": "Student",
                "reasoning": "Gaming logistics is a stretch for the persona.",
                "confidence": "medium",
            }
        ),
        assigned_task(domain="Student"),
    )
    assert parsed.assessment == "partial"
    assert parsed.assigned_domain == "Student"
    assert parsed.prompt_quote == "configure stockpiles in Dwarf Fortress"
    assert parsed.domain_basis == "Student"
    assert parsed.confidence == "medium"
    assert parsed.is_issue and not parsed.is_egregious


def test_an_unrecognised_assessment_falls_back_to_aligned():
    """A malformed enum must not manufacture a fail on a check that publishes a
    fail label."""
    parsed = parse_domain_relevance(
        _response({"assessment": "totally different", "prompt_quote": "x", "domain_basis": "y"}),
        assigned_task(),
    )
    assert parsed.assessment == "aligned"
    assert evaluate_check_70(assigned_task(), parsed).band == "clean"


# ---------------------------------------------------------------------------
# Registration and accounting
# ---------------------------------------------------------------------------


def test_the_dimension_is_registered_so_it_appears_in_every_report():
    assert 70 in REGISTRY
    assert 70 in ORDER
    assert REGISTRY[70].dimension == "Prompt"
    assert REGISTRY[70].sub_dimension == "Domain Relevance"


def test_the_call_estimator_counts_the_domain_call_only_when_it_will_run():
    without = estimate_informed_calls(make_task())
    with_domain = estimate_informed_calls(assigned_task())
    assert with_domain == without + 1


def test_the_stage_reports_a_verdict_for_70_on_a_task_with_a_domain():
    client = FakeModelClient(
        lambda r: (
            {
                "assessment": "aligned",
                "prompt_quote": "explaining how a gyroscope works",
                "domain_basis": DOMAIN,
                "reasoning": "A teaching aid is squarely an educator's work.",
                "confidence": "high",
            }
            if r.metadata.get("check_id") == 70
            else {"confidence": "high"}
        )
    )
    result = run_informed_stage(assigned_task(), client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 70)
    assert verdict.band == "clean"
    assert result.domain_relevance is not None
    assert result.to_dict()["domain_relevance"]["assessment"] == "aligned"


def test_the_stage_reports_70_as_not_evaluated_when_the_call_fails():
    client = FakeModelClient(lambda r: ModelResponse(key=r.key, error="upstream timeout"))
    result = run_informed_stage(assigned_task(), client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 70)
    assert verdict.band == "not_evaluated"
    assert result.domain_relevance is None
