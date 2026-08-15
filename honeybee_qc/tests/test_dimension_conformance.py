"""N/A is the absence of a rating, not a rating.

Two invariants the rest of the dimension machinery depends on: a dimension marked
N/A can never also carry a number, and an N/A dimension never appears in the
denominator of a rating comparison.
"""

from __future__ import annotations

from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.findings import DimensionRatingFinding
from honeybee_qc.gates import evaluate_check_300
from honeybee_qc.models import Task
from honeybee_qc.rating_prompts import DIMENSION_RATING_SCHEMA


def _task() -> Task:
    return Task(task_id="t-na")


def test_na_and_rating_cannot_coexist() -> None:
    """A finding built with both keeps only the N/A."""
    f = DimensionRatingFinding(
        model="A",
        dimension="Tool & connector reliability",
        contributor_rating=2,
        auditor_rating=4,
        contributor_na=True,
        auditor_na=True,
    )
    assert f.contributor_rating is None
    assert f.auditor_rating is None
    assert f.delta is None


def test_na_is_representable_as_absent_rating_in_the_schema() -> None:
    """The judge can decline a rating rather than inventing one."""
    props = DIMENSION_RATING_SCHEMA["properties"]
    assert "not_applicable" in props
    assert "null" in props["rating"]["type"], (
        "rating must be nullable, or a dimension the conversation never exercised "
        "still forces a fabricated number"
    )


def test_na_dimensions_leave_the_comparison_denominator() -> None:
    """Both-N/A dimensions are excluded; the denominator counts only comparables."""
    findings = [
        DimensionRatingFinding(
            model="A", dimension="Outcome quality",
            contributor_rating=4, auditor_rating=4,
        ),
        DimensionRatingFinding(
            model="A", dimension="Tool & connector reliability",
            contributor_na=True, auditor_na=True,
        ),
        DimensionRatingFinding(
            model="A", dimension="Memory & personalization",
            contributor_na=True, auditor_na=True,
        ),
    ]
    verdict = evaluate_check_300(_task(), findings, DEFAULT_POLICY)
    assert verdict.measurement.denominator == 1


def test_applicability_disagreement_is_not_a_rating_disagreement() -> None:
    """One-sided N/A stays visible as its own condition."""
    findings = [
        DimensionRatingFinding(
            model="A", dimension="Tool & connector reliability",
            contributor_rating=2, auditor_na=True,
        ),
    ]
    verdict = evaluate_check_300(_task(), findings, DEFAULT_POLICY)
    counts = verdict.measurement.counts
    assert counts["na_mismatches"] == 1
    reasons = counts["reasons_by_item"]["A::Tool & connector reliability"]
    assert "na_mismatch" in reasons
    assert "rating_delta" not in reasons
