"""The spec's General Grading Instructions, as behaviour rather than prose.

Source: "HoneyBee_v2 (Ultra Evals Experts) - ... - QC Spec Doc - Audit Workflow",
the General Grading Instructions block. Two of its numbered rules are load-bearing
here and were being applied wrongly or not at all:

  * Rule 4: "For SLA projects, if a field is not relevant to your claimed attempt
    because it is not shown in the UI or does not apply to the task type, choose
    the 'no issues' option." An inapplicable field is a recorded clean, which in
    our band vocabulary is `clean` and not `not_evaluated`. Tab 1 says the same
    thing from the other direction: the pass criterion for each rating dimension
    is "Your rating matches the contributor's OR the contributor marks 'not
    applicable' where the response cannot be assessed or vice versa".

  * Rule 1, fifth bullet: "If a contributor presents a reasonable argument, we
    should accept their interpretation." The instruction has to reach the judge
    that reads the contributor's argument, and it has to reach the definition it
    is meant to soften rather than sit eighty lines away from it.

The distinction these tests defend hardest is between a dimension that does not
apply -- clean, counted -- and a dimension we could not judge -- not_evaluated,
excluded. `build_batch_report` drops `not_evaluated` from the denominator, so
collapsing the first into the second shrinks published denominators on exactly
the tasks where the spec asked for a clean.
"""

from __future__ import annotations

from honeybee_qc.config import DEFAULT_POLICY, Policy
from honeybee_qc.findings import DimensionRatingFinding
from honeybee_qc.gates import evaluate_check_300
from honeybee_qc.informed_prompts import INFORMED_SYSTEM_PROMPT, JUSTIFICATION_RUBRIC
from honeybee_qc.llm import ModelResponse
from honeybee_qc.models import Task
from honeybee_qc.prompts import SYSTEM_PROMPT as RUBRIC_SYSTEM_PROMPT
from honeybee_qc.rating_prompts import CONTRIBUTOR_MARKERS, RATING_SYSTEM_PROMPT
from honeybee_qc.rating_stages import _thin_evidence_verdict, parse_dimension_rating
from honeybee_qc.rollup import build_batch_report, roll_up_task

TASK_ID = "t-general-grading"


def _na(dimension: str) -> DimensionRatingFinding:
    return DimensionRatingFinding(
        model="A", dimension=dimension, contributor_na=True, auditor_na=True
    )


def _published(verdict) -> tuple[int, int, int]:
    """(denominator, clean, not_evaluated) as the batch report would publish them."""
    report = build_batch_report([roll_up_task(TASK_ID, [verdict], DEFAULT_POLICY)])
    rate = next(r for r in report.check_rates if r.check_id == verdict.check_id)
    return rate.denominator, rate.clean, rate.not_evaluated


# ---------------------------------------------------------------------------
# Rule 4: an inapplicable field is "no issues"
# ---------------------------------------------------------------------------


def test_inapplicable_dimensions_land_in_the_denominator_as_clean():
    """Every dimension inapplicable is a clean task, not an unevaluated one."""
    findings = [_na("Tool & connector reliability"), _na("Memory & personalization")]
    verdict = evaluate_check_300(Task(task_id=TASK_ID), findings, DEFAULT_POLICY)

    assert verdict.band == "clean"
    # The rating comparison has nothing to compare, which is not the same as the
    # check having nothing to say.
    assert verdict.measurement.denominator == 0
    assert _published(verdict) == (1, 1, 0)


def test_one_inapplicable_dimension_does_not_shrink_the_task_count():
    findings = [
        DimensionRatingFinding(
            model="A", dimension="Outcome quality", contributor_rating=4, auditor_rating=4
        ),
        _na("Memory & personalization"),
    ]
    verdict = evaluate_check_300(Task(task_id=TASK_ID), findings, DEFAULT_POLICY)

    assert verdict.band == "clean"
    assert verdict.measurement.denominator == 1
    assert _published(verdict) == (1, 1, 0)


def test_a_dimension_we_could_not_judge_is_still_not_evaluated():
    """The other half of the distinction, and the reason it cannot be collapsed.

    Nothing was inapplicable here; nothing was judgeable either. That is our
    mechanical failure to report on, not a "no issues" the contributor earned, and
    it leaves the published denominator.
    """
    verdict = _thin_evidence_verdict(
        Task(task_id=TASK_ID), 300, judged=0, abstained=0, policy=DEFAULT_POLICY
    )
    assert verdict is not None and verdict.band == "not_evaluated"
    assert _published(verdict) == (0, 0, 1)


def test_a_missing_contributor_entry_is_not_a_clean():
    """A dimension absent from the contributor's form yields no judgment at all.

    Rule 4 is about a field that does not apply, not about a form entry we failed
    to read: recording the second as a clean would credit the contributor for a
    dimension nobody rated.
    """
    response = ModelResponse(
        key="k",
        data={"rating": 4, "not_applicable": False, "confidence": "high"},
    )
    finding, abstention, error = parse_dimension_rating(
        response,
        "Memory & personalization",
        "A",
        contributor_rating=None,
        contributor_na=False,
        has_contributor_entry=False,
    )
    assert (finding, abstention, error) == (None, None, "")

    # With the entry present and both sides calling it inapplicable, the same call
    # does produce a judgment -- the one rule 4 wants counted as clean.
    finding, _, _ = parse_dimension_rating(
        ModelResponse(key="k", data={"not_applicable": True, "confidence": "high"}),
        "Memory & personalization",
        "A",
        contributor_rating=None,
        contributor_na=True,
        has_contributor_entry=True,
    )
    assert finding is not None and finding.both_na


# ---------------------------------------------------------------------------
# Rule 4, continued: an applicability divergence is not a defect
# ---------------------------------------------------------------------------


def test_na_mismatch_is_reported_without_being_counted_by_default():
    findings = [
        DimensionRatingFinding(
            model="A",
            dimension="Memory & personalization",
            contributor_na=True,
            auditor_rating=4,
        )
    ]
    verdict = evaluate_check_300(Task(task_id=TASK_ID), findings, DEFAULT_POLICY)
    counts = verdict.measurement.counts

    assert DEFAULT_POLICY.na_mismatch_counts_as == "ignored"
    assert verdict.band == "clean", (
        "an applicability divergence with no numeric disagreement is inside the "
        "spec's 'no issues' cell and must not manufacture a defect"
    )
    assert counts["total_disagreements"] == 0
    # Still visible: the two sides read the task type differently, which is real
    # signal even when it is not a defect.
    assert counts["na_mismatches"] == 1
    assert "na_mismatch" in counts["reasons_by_item"]["A::Memory & personalization"]


def test_na_mismatch_can_still_be_made_a_defect_by_policy():
    """Only the default changed. If the customer says otherwise, the knob works."""
    findings = [
        DimensionRatingFinding(
            model="A",
            dimension="Memory & personalization",
            contributor_na=True,
            auditor_rating=4,
        )
    ]
    task = Task(task_id=TASK_ID)
    assert evaluate_check_300(task, findings, Policy(na_mismatch_counts_as="minor")).band == (
        "non_fail"
    )
    strict = evaluate_check_300(task, findings, Policy(na_mismatch_counts_as="major"))
    assert strict.measurement.counts["major_disagreements"] == 1


# ---------------------------------------------------------------------------
# Rule 1: a reasonable argument has its interpretation accepted
# ---------------------------------------------------------------------------


def test_informed_judges_are_told_to_accept_a_reasonable_argument():
    """The informed judge reads the argument, so charity must be about the argument.

    "A requirement admits more than one reasonable reading" tells a judge that has
    the contributor's reasoning in front of it nothing about what to do with it.
    """
    text = INFORMED_SYSTEM_PROMPT.lower()
    assert "reasonable argument" in text
    assert "accept their interpretation" in text
    assert "would have argued differently" in text


def test_misconstrued_evidence_carries_its_own_charity_carve_out():
    """The carve-out has to be inside the definition it softens.

    The general instruction lower down in the same prompt is too far from the
    count to reach a judge applying it; this is the count the disputed Athletes
    Untapped flag was filed under.
    """
    definition = JUSTIFICATION_RUBRIC.split("**misconstrued_evidence**")[1]
    # Up to the next bolded count, so the carve-out has to be inside this one's
    # definition rather than anywhere in the rubric. Wrapping is normalised: the
    # prompt is hard-wrapped and a phrase can straddle a line break.
    definition = " ".join(definition.split("**")[0].lower().split())

    assert "more than one reading" in definition
    assert "reasonable" in definition
    assert "no sound reading" in definition


def test_rubric_charity_is_not_scoped_to_the_contributors_wording():
    """A reading can be reasonable without the wording spelling it out."""
    text = RUBRIC_SYSTEM_PROMPT.lower()
    assert "the contributor's wording admits" not in text
    assert "the criterion admits a reasonable interpretation" in text


# ---------------------------------------------------------------------------
# Rule 3: "the prompt or task instructions", both sources
# ---------------------------------------------------------------------------


def test_every_judge_names_both_sources_of_an_explicit_instruction():
    """Rule 3 names two sources: "explicitly asked for in the prompt or task
    instructions". A judge told only about the prompt penalises work the task
    instructions required, and vice versa."""
    for name, prompt in (
        ("rubric", RUBRIC_SYSTEM_PROMPT),
        ("rating", RATING_SYSTEM_PROMPT),
        ("informed", INFORMED_SYSTEM_PROMPT),
    ):
        assert "the prompt or the task instructions" in prompt, name


def test_the_blind_system_prompt_still_trips_no_contributor_marker():
    """Charity wording must never be the thing that unblinds the rating pass.

    `assert_blind` only ever sees the user prompt, so a marker here would not be
    caught by it -- which is exactly why it is asserted here.
    """
    lowered = RATING_SYSTEM_PROMPT.lower()
    for marker in CONTRIBUTOR_MARKERS:
        assert marker not in lowered, marker
