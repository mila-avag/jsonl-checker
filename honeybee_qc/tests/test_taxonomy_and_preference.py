"""The L1/L2 taxonomy, its string encoding, and preference normalisation.

These cover the three places where the QC spec and the authoring form disagree
about what the data looks like. The spec is what the audit is contracted
against; the form is what actually arrives. Where they diverge the divergence is
pinned here rather than smoothed over, so a future export that settles the
question fails loudly instead of silently changing published rates.
"""

from __future__ import annotations

import pytest

from honeybee_qc.config import Policy
from honeybee_qc.findings import LikertFinding
from honeybee_qc.gates import likert_direction, signed_preference
from honeybee_qc.taxonomies import (
    L1_LABELS,
    L1_OF_L2,
    L2_BY_L1,
    L2_DEFINITIONS,
    L2_LABELS,
    SPEC_L1_LABELS,
    parse_criterion_category,
)

DIRECTIONAL = Policy(preference_encoding="direction_magnitude")


# ---------------------------------------------------------------------------
# Taxonomy shape
# ---------------------------------------------------------------------------


def test_the_form_adds_one_l1_the_spec_never_lists():
    """Check 200's list has seven categories; the form offers eight.

    Auditing against the spec's seven would leave the auditor unable to express
    a label the contributor can select, turning every tool/connector criterion
    into an L1 disagreement nobody could have avoided.
    """
    assert len(SPEC_L1_LABELS) == 7
    assert set(SPEC_L1_LABELS) < set(L1_LABELS)
    assert set(L1_LABELS) - set(SPEC_L1_LABELS) == {"Tool and Connector Reliability"}


def test_every_l1_carries_at_least_one_l2_leaf():
    assert set(L2_BY_L1) == set(L1_LABELS)
    for l1, leaves in L2_BY_L1.items():
        assert leaves, f"{l1} has no L2 leaves"


def test_l2_leaves_are_unique_and_fully_parented():
    assert len(L2_LABELS) == len(set(L2_LABELS))
    assert set(L1_OF_L2) == set(L2_LABELS)
    assert all(L1_OF_L2[leaf] in L1_LABELS for leaf in L2_LABELS)
    assert all(L2_DEFINITIONS[leaf].strip() for leaf in L2_LABELS)


def test_the_leaf_count_records_the_sources_own_off_by_one():
    """reference.md says "25 L2 leaves" and then lists 26.

    The extra is Tool and Connector Reliability's leaf, which repeats its
    parent's name and was presumably not counted as distinct. It is kept: drop it
    and that L1 becomes unlabelable at L2.
    """
    assert len(L2_LABELS) == 26
    assert L2_BY_L1["Tool and Connector Reliability"][0][0] == "Tool and Connector Reliability"


# ---------------------------------------------------------------------------
# criterion_category parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, l1, l2",
    [
        (
            "outcome quality-artifact delivery: a usable artifact was actually produced.",
            "Outcome Quality",
            "Artifact Delivery",
        ),
        # The L2 carries its own hyphen, so splitting on the first one is wrong.
        (
            "trust and grounding-anti-hallucination: nothing invented.",
            "Trust and Grounding",
            "Anti-Hallucination",
        ),
        # The L1 name repeats as its only L2.
        (
            "tool and connector reliability-tool and connector reliability: calls.",
            "Tool and Connector Reliability",
            "Tool and Connector Reliability",
        ),
        # A slash inside the leaf name.
        (
            "outcome quality-artifact / task correctness: the result is right.",
            "Outcome Quality",
            "Artifact / Task Correctness",
        ),
    ],
)
def test_parses_the_forms_single_string_into_canonical_labels(raw, l1, l2):
    parsed = parse_criterion_category(raw)
    assert parsed.error is None
    assert (parsed.l1_label, parsed.l2_label) == (l1, l2)


def test_a_leaf_under_the_wrong_parent_is_reported_not_accepted():
    parsed = parse_criterion_category("safety-goal elicitation: mismatched parent")
    assert parsed.l1_label == "Safety"
    assert parsed.l2_label == "Goal Elicitation"
    assert "does not belong under" in (parsed.error or "")


@pytest.mark.parametrize("raw", ["", None, "   ", "nonsense-whatever: x"])
def test_unparseable_categories_return_an_error_rather_than_raising(raw):
    """A label this build has never seen is a finding for 200/210 to report, not
    a crash that drops the whole task."""
    parsed = parse_criterion_category(raw)
    assert parsed.error
    assert parsed.l1_label is None


def test_a_known_l1_with_an_unknown_leaf_keeps_the_l1():
    parsed = parse_criterion_category("safety-invented leaf: x")
    assert parsed.l1_label == "Safety"
    assert parsed.l2_label is None
    assert "unrecognised L2" in (parsed.error or "")


# ---------------------------------------------------------------------------
# Preference normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "likert, direction",
    [(1, "A"), (2, "A"), (3, "neutral"), (4, "neutral"), (5, "neutral"), (6, "B"), (7, "B")],
)
def test_bipolar_direction_follows_the_specs_buckets(likert, direction):
    assert likert_direction(likert) == direction


@pytest.mark.parametrize("winner, direction", [(0, "A"), (1, "B")])
def test_directional_encoding_reads_the_winner_field(winner, direction):
    assert likert_direction(3, DIRECTIONAL, winner_index=winner) == direction


def test_directional_encoding_has_no_neutral_without_a_winner():
    """responseIdx always names a side, so a recorded preference always has one.
    Absent the field there is nothing to read, which is silence, not a tie."""
    assert likert_direction(3, DIRECTIONAL, winner_index=None) == "neutral"


def test_the_bipolar_centre_cancels_so_raw_deltas_stay_correct():
    assert signed_preference(4) == 0
    assert signed_preference(1) == -3
    assert signed_preference(7) == 3
    assert LikertFinding(contributor_likert=2, auditor_likert=5).delta == 3


def test_a_flipped_winner_separates_the_two_sides_of_the_axis():
    """The failure the raw delta misses: both sides say "4", but one means A by 4
    and the other B by 4. Reading the margins alone scores that as agreement."""
    finding = LikertFinding(
        contributor_likert=4,
        auditor_likert=4,
        contributor_winner_index=0,
        auditor_winner_index=1,
    )
    assert finding.delta == 8


def test_agreeing_on_the_winner_leaves_only_the_margin_gap():
    finding = LikertFinding(
        contributor_likert=1,
        auditor_likert=4,
        contributor_winner_index=1,
        auditor_winner_index=1,
    )
    assert finding.delta == 3


def test_signed_preference_needs_a_winner_under_the_directional_encoding():
    assert signed_preference(4, winner_index=None, policy=DIRECTIONAL) is None
    assert signed_preference(4, winner_index=0, policy=DIRECTIONAL) == -4
    assert signed_preference(4, winner_index=1, policy=DIRECTIONAL) == 4
