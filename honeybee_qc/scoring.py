"""Bands, shapes, and the verdict envelope.

Shape is enforced here rather than only in prompts: a Shape B check cannot
express a fail, so it cannot hallucinate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import DEFAULT_POLICY, Policy
from .errors import error_code

Band = Literal["fail", "non_fail", "clean", "not_evaluated"]
Shape = Literal["A", "B", "C"]

# Shape A: binary, fail or clean; the 3-4 cell is N/A.
# Shape B: non-fail only; the 1 cell is N/A, so the check can never fail a task.
# Shape C: graded, all three bands available.
SHAPE_BANDS: dict[Shape, frozenset[str]] = {
    "A": frozenset({"fail", "clean", "not_evaluated"}),
    "B": frozenset({"non_fail", "clean", "not_evaluated"}),
    "C": frozenset({"fail", "non_fail", "clean", "not_evaluated"}),
}


class ShapeViolation(ValueError):
    pass


def score_for_band(band: Band, policy: Policy = DEFAULT_POLICY) -> int | None:
    if band == "fail":
        return policy.fail_score
    if band == "non_fail":
        return policy.non_fail_score
    if band == "clean":
        return policy.clean_score
    return None


@dataclass
class Measurement:
    """The arithmetic behind a gated score, kept for auditability."""

    numerator: float | None = None
    denominator: float | None = None
    rate: float | None = None
    threshold: float | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "threshold": self.threshold,
            "counts": dict(self.counts),
            "notes": self.notes,
        }


@dataclass
class CheckVerdict:
    task_id: str
    check_id: int
    dimension: str
    sub_dimension: str
    shape: Shape
    band: Band
    score: int | None
    error_code: str | None
    measurement: Measurement = field(default_factory=Measurement)
    contributing_items: list[str] = field(default_factory=list)
    confidence: str = "high"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "check_id": self.check_id,
            "dimension": self.dimension,
            "sub_dimension": self.sub_dimension,
            "score": self.score,
            "band": self.band,
            "error_code": self.error_code,
            "measurement": self.measurement.to_dict(),
            "contributing_items": list(self.contributing_items),
            "confidence": self.confidence,
        }


def build_verdict(
    *,
    task_id: str,
    check_id: int,
    band: Band,
    measurement: Measurement | None = None,
    contributing_items: list[str] | None = None,
    confidence: str = "high",
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Construct a verdict, refusing any band the check's shape cannot express."""
    from .registry import REGISTRY

    spec = REGISTRY[check_id]
    if band not in SHAPE_BANDS[spec.shape]:
        raise ShapeViolation(
            f"check {check_id} is shape {spec.shape}; band {band!r} is not available "
            f"(allowed: {sorted(SHAPE_BANDS[spec.shape])})"
        )
    score = score_for_band(band, policy)
    if score == 2:
        raise ShapeViolation("score 2 is unused and must never be emitted")
    return CheckVerdict(
        task_id=task_id,
        check_id=check_id,
        dimension=spec.dimension,
        sub_dimension=spec.sub_dimension,
        shape=spec.shape,
        band=band,
        score=score,
        error_code=error_code(check_id, band),
        measurement=measurement or Measurement(),
        contributing_items=contributing_items or [],
        confidence=confidence,
    )
