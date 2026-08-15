"""Pins the implementation to the QC spec's Dimension Definition tab.

Every value here was verified against
"HoneyBee_v2 ... QC Spec Doc - 2. Dimension Definition.csv" by diffing the tab
against the code. These tests exist so a later edit cannot silently drift: the
error-code strings land in a customer's report column verbatim, and the
thresholds decide pass or fail.

Values are written out longhand rather than parsed from the CSV on purpose. A
test that reads the same source as the code would pass even if both were wrong,
and the CSV is not shipped with the package.
"""

from __future__ import annotations

import pytest

from honeybee_qc.config import DEFAULT_POLICY as P
from honeybee_qc.errors import ERROR_CODES
from honeybee_qc.registry import ORDER, REGISTRY
from honeybee_qc.taxonomies import (
    ISSUE_DEFINITIONS,
    MAJOR_CATEGORIES,
    MINOR_CATEGORIES,
    MODERATE_CATEGORIES,
    RATING_DIMENSIONS,
    WEIGHT_CONDITIONED_CATEGORIES,
    WEIGHT_DEFINITIONS,
    weight_confirms_double_negative,
)

# Order of Appearance in Rubric, column D of the tab.
SPEC_CHECK_IDS = (
    80, 90, 95, 100, 110, 200, 210, 220, 230, 240, 250, 260,
    270, 280, 300, 310, 400, 450, 460, 470, 1000,
)

# Dimensions the registry grades that tab 2 gives no rubric position. Their IDs
# are local, so they are excluded from the verbatim pins above and justified
# individually below. This tuple is the single declaration of which IDs may sit
# outside tab 2; the count pins in test_shapes_and_codes read it rather than
# repeating a total.
NET_NEW_CHECK_IDS = (70, 75, 85, 96)

# Derived from which of the "1-2" / "3-4" columns hold N/A.
SPEC_SHAPES = {
    80: "B", 90: "A", 95: "A", 100: "C", 110: "B", 200: "C", 210: "B",
    220: "A", 230: "C", 240: "C", 250: "C", 260: "C", 270: "C", 280: "C",
    300: "C", 310: "C", 400: "C", 450: "C", 460: "A", 470: "A", 1000: "A",
}

# Verbatim from the "1 (Custom)" and "3 - 4 (Custom)" columns.
SPEC_ERROR_CODES: dict[int, dict[str, str | None]] = {
    80:   {"fail": None, "non_fail": "[Non-Fail - Misaligned Target Outcome]"},
    90:   {"fail": "[Fail - Missing/Invalid Links]", "non_fail": None},
    95:   {"fail": "[Fail - Artifacts Not Preserved]", "non_fail": None},
    100:  {"fail": "[Fail - First Turn is Key Turn]",
           "non_fail": "[Non-Fail - Misidentified Key Turn]"},
    110:  {"fail": None,
           "non_fail": "[Non-Fail - Incorrect Key Turn Justification]"},
    200:  {"fail": "[Fail - L1 Labels/Annotations]",
           "non_fail": "[Non-Fail - L1 Labels/Annotations]"},
    210:  {"fail": None, "non_fail": "[Non-Fail - L2 Labels/Annotations]"},
    220:  {"fail": "[Fail - Rubric Autofail]", "non_fail": None},
    230:  {"fail": "[Fail - 10%+ Major Rubric Errors]",
           "non_fail": "[Non-Fail - < 10% Major Rubric Errors]"},
    # The spec's own non-fail string for 240 reads "< 15%+", which is a typo in
    # the source. It is reproduced exactly: this string is published.
    240:  {"fail": "[Fail - 15%+ Major/Moderate Rubric Errors]",
           "non_fail": "[Non-Fail - < 15%+ Major/Moderate Rubric Errors]"},
    250:  {"fail": "[Fail - 20%+ Major/Moderate/Minor Rubric Errors]",
           "non_fail": "[Non-Fail - < 20% Major/Moderate/Minor Rubric Errors]"},
    260:  {"fail": "[Fail - Criteria Weights]",
           "non_fail": "[Non-Fail - Criteria Weights]"},
    270:  {"fail": "[Fail - Egregious Rating Disagreement]",
           "non_fail": "[Non-Fail - Minor Rating Disagreement]"},
    280:  {"fail": "[Fail - Incorrect Relevant Turns]",
           "non_fail": "[Non-Fail - Incorrect Relevant Turns]"},
    300:  {"fail": "[Fail - Dimension Rating Disagreement]",
           "non_fail": "[Non-Fail - Dimension Rating Disagreement]"},
    310:  {"fail": "[Fail - Incorrect Relevant Turns]",
           "non_fail": "[Non-Fail - Incorrect Relevant Turns]"},
    400:  {"fail": "[Fail - Major Ranking Disagreement]",
           "non_fail": "[Non-Fail - Loose Ranking Disagreement]"},
    450:  {"fail": "[Fail - Bad Justifications]",
           "non_fail": "[Non-Fail - Bad Justifications]"},
    460:  {"fail": "[Fail - Inconsistent Ranking]", "non_fail": None},
    470:  {"fail": "[Fail - Lacks Verdict]", "non_fail": None},
    # The only check whose fail cell carries no bracketed code.
    1000: {"fail": "This task is unauditable", "non_fail": None},
}


def test_the_registry_holds_the_specs_twenty_one_dimensions():
    """Every tab-2 dimension is graded and none has been quietly dropped.

    Written as a subset rather than equality so a net-new dimension can be added
    without loosening the pin on the contracted twenty-one.
    """
    assert set(SPEC_CHECK_IDS) <= set(ORDER)
    assert len(SPEC_CHECK_IDS) == 21


def test_every_registry_id_is_either_from_tab_two_or_declared_net_new():
    assert set(ORDER) == set(SPEC_CHECK_IDS) | set(NET_NEW_CHECK_IDS)


@pytest.mark.parametrize("check_id", NET_NEW_CHECK_IDS)
def test_net_new_dimensions_say_so_in_their_notes(check_id):
    """A local ID is only defensible if the report says it is one. Without this a
    reader assumes 70 is a rubric position the customer assigned."""
    assert "Net-new" in REGISTRY[check_id].notes


@pytest.mark.parametrize("check_id", SPEC_CHECK_IDS)
def test_shape_matches_the_specs_na_pattern(check_id):
    assert REGISTRY[check_id].shape == SPEC_SHAPES[check_id]


@pytest.mark.parametrize("check_id", SPEC_CHECK_IDS)
def test_error_codes_are_verbatim(check_id):
    """A band the spec marks N/A must have no code, since its shape forbids it."""
    codes = ERROR_CODES.get(check_id, {})
    for band, expected in SPEC_ERROR_CODES[check_id].items():
        assert codes.get(band) == expected, f"check {check_id} band {band}"


def test_check_70_reproduces_the_labels_the_auditors_actually_cite():
    """Check 70's strings come from the QC export, not from a tab-2 cell, because
    tab 1 states the 5-of-5 bar and no band labels at all.

    The two are not symmetrical -- the fail label ends in "Prompt" and the
    non-fail label does not -- and both are reproduced as cited. Normalising
    either would break the join against the customer's own tooling, the same
    reason 240's "< 15%+" typo is preserved.
    """
    assert ERROR_CODES[70] == {
        "fail": "[Fail - Domain Relevance Prompt]",
        "non_fail": "[Non-Fail - Domain Relevance]",
    }


def test_check_70_grades_the_bar_tab_one_states():
    """Verbatim from tab 1, row 11: Prompt / Domain Relevance, whose "5 out of 5"
    definition reads "- The Prompt(s) are clearly related to the assigned
    domain." Shape C: the fail wording cited in the field is "egregious", which
    needs a middle band beneath it for a prompt that merely drifts."""
    spec = REGISTRY[70]
    assert (spec.dimension, spec.sub_dimension) == ("Prompt", "Domain Relevance")
    assert spec.shape == "C"
    assert spec.deterministic is False


def test_280_and_310_share_error_strings_which_is_why_findings_are_namespaced():
    assert ERROR_CODES[280] == ERROR_CODES[310]


def test_thresholds_match_the_spec():
    # "At least 30% (inclusive) of the labels are incorrect"
    assert P.l1_fail_rate == 0.30
    # "10% / 15% / 20% or more of the criteria contain ... issues"
    assert P.rubric_major_fail_rate == 0.10
    assert P.rubric_major_moderate_fail_rate == 0.15
    assert P.rubric_all_fail_rate == 0.20
    # "At least 5% ... off by two levels. At least 30% ... off by one level or two."
    assert P.weights_two_level_fail_rate == 0.05
    assert P.weights_any_level_fail_rate == 0.30
    # "5 or more criteria, or 20% or more of the criteria"
    assert P.rubric_eval_fail_count == 5
    assert P.rubric_eval_fail_rate == 0.20
    # "There are at least 3 incorrectly selected turns."
    assert P.relevant_turns_fail_count == 3
    # "differ by 3 or more points"; "2 or more major"; "5 or more major or minor"
    assert P.dimension_major_delta == 3
    assert P.dimension_fail_major_count == 2
    assert P.dimension_fail_total_count == 5
    # "likert ratings differ by 3 or more points"
    assert P.likert_fail_delta == 3


def test_weight_definitions_match_the_customers_weights_table():
    """The Weights/Definition table, longhand. The prompt shows these definitions to
    the auditor and the defensible range check 260 gates on is read off them, so a
    paraphrase that shifts a level shifts the measurement with it."""
    assert WEIGHT_DEFINITIONS == {
        1: (
            "Minor positive. A nice-to-have quality that provides a small "
            "improvement but is not essential."
        ),
        2: (
            "Moderately important. A valuable behavior or attribute that "
            "meaningfully improves the result."
        ),
        3: (
            "Important. A key quality that strongly contributes to task success "
            "and user satisfaction."
        ),
        4: (
            "Highly important. A critical success factor whose presence "
            "substantially elevates the quality of the outcome."
        ),
        5: (
            "Essential - a defining requirement. Doing this well is fundamental to "
            "task success and should heavily influence the final score."
        ),
    }


def test_weight_scale_is_one_to_five_with_no_negatives():
    """Check 260: "Criteria weights should be on a scale of 1 to 5 (no negative
    weights)". The audit workflow tab's 8-10 / 4-7 bands are shared boilerplate
    from another project and do not govern here.

    This fixes the scale the auditor assigns on, and makes a negative contributor
    weight a 260 error. It does not assert that no contributor wrote one: they did,
    which is why framing/double negative's negative-weight condition still bites.
    """
    assert P.weight_scale == (1, 5)
    buckets = P.weight_buckets
    assert buckets.bucket(1) == "low" and buckets.bucket(2) == "low"
    assert buckets.bucket(3) == "medium"
    assert buckets.bucket(4) == "high" and buckets.bucket(5) == "high"
    with pytest.raises(ValueError):
        buckets.bucket(-5)
    with pytest.raises(ValueError):
        buckets.bucket(8)


def test_fail_maps_to_one_because_the_custom_column_is_headed_1_not_1_to_2():
    """Standard dimensions use a "1- 2" column; the custom dimensions this project
    uses are headed "1 (Custom)", so a fail is exactly 1 and score 2 is unused."""
    assert P.fail_score == 1
    assert P.non_fail_score == 3
    assert P.clean_score == 5


def test_severity_census_matches_the_specs_three_tiers():
    assert MAJOR_CATEGORIES == frozenset(
        {"inaccurate", "counterproductive", "missing_critical"}
    )
    assert MODERATE_CATEGORIES == frozenset(
        {
            "not_self_contained",
            "non_atomic",
            "overlapping",
            "vague_subjective",
            "missing_non_critical",
            "framing_double_negative",
        }
    )
    assert MINOR_CATEGORIES == frozenset({"unnecessary"})


def test_framing_double_negative_requires_a_negative_weight():
    """The audit workflow tab defines it conditionally: "When a criterion with a
    negative weight is framed negatively (checks for the absence of something,
    instead of its presence)".

    Both halves are required. Check 260's "no negative weights" says what a
    contributor ought to write, not what the data contains -- live rubrics carry
    weights from -3 to -5 -- so the condition is satisfiable and dropping it turned
    a conditional moderate error into an unconditional one.
    """
    assert WEIGHT_CONDITIONED_CATEGORIES == frozenset({"framing_double_negative"})
    assert weight_confirms_double_negative(-5)
    assert weight_confirms_double_negative(-1)
    assert not weight_confirms_double_negative(1)
    assert not weight_confirms_double_negative(5)
    assert not weight_confirms_double_negative(0)
    assert not weight_confirms_double_negative(None)


def test_the_criterion_prompt_asks_only_for_the_phrasing_half():
    """The weight half cannot be delegated to the model: the criterion prompt
    withholds the contributor's weight so checks 200 and 260 measure independent
    judgments. The enum therefore has to say so, or the model infers a weight."""
    definition = ISSUE_DEFINITIONS["framing_double_negative"]
    assert "withheld" in definition
    assert "negative weight" in definition


def test_rating_dimensions_match_check_300_verbatim():
    assert RATING_DIMENSIONS == (
        "Outcome quality",
        "Communication quality",
        "Trust & grounding",
        "Interaction efficiency",
        "Tool & connector reliability",
        "Memory & personalization",
    )


def test_the_dimension_rating_scale_comes_from_the_authoring_form():
    """Check 300 says "differ by 3 or more points" but no tab states the scale.

    The authoring form does: every `<dim>_<a|b>` field is `number 1-5`
    (honeybee-l1-task-browsing reference.md, "Dimension-rating fields"). This was
    an assumed 1-10 until that landed, which understated every disagreement --
    a 3-point gap is most of a 1-5 range and less than a third of a 1-10 one.
    """
    assert P.dimension_rating_scale == (1, 5)


def test_the_preference_encoding_conflict_stays_visible():
    """The spec and the authoring form disagree about how preference is stored.

    Tab 1 (rows 135, 137) describes one 1-7 Likert bucketed (1,2)/(3,4,5)/(6,7),
    so the value carries direction. The form stores `responseIdx` (0=A, 1=B)
    beside a `preferenceLikert` of 1-5 that carries only the margin. The default
    follows the spec because that is what this audit is contracted against;
    reading tasks out of Snowflake requires flipping the encoding.
    """
    assert P.preference_encoding == "bipolar_7"
    assert P.likert_scale == (1, 7)
    assert P.preference_margin_scale == (1, 5)
