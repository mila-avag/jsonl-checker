"""Fixture builders. No model calls anywhere in the test suite."""

from __future__ import annotations

from honeybee_qc.findings import (
    CriterionFinding,
    CriterionRatingFinding,
    DimensionRatingFinding,
    Issue,
    L1Finding,
    WeightFinding,
)
from honeybee_qc.models import (
    CriterionRating,
    DimensionRating,
    KeyTurn,
    ModelSubmission,
    RubricCriterion,
    Sxs,
    Task,
    Turn,
    TurnLink,
)
from honeybee_qc.taxonomies import RATING_DIMENSIONS

GEMINI = "https://gemini.google.com/share/d64e0a300ec0"
GPT = "https://chatgpt.com/share/8f2c1d4e9a0b"
CLAUDE = "https://claude.ai/share/1a2b3c4d5e6f"


def make_conversation(exchanges: int = 20, prefix: str = "") -> list[Turn]:
    turns: list[Turn] = []
    for i in range(1, exchanges + 1):
        turns.append(Turn(index=i, role="user", text=f"{prefix}user message {i}"))
        turns.append(Turn(index=i, role="assistant", text=f"{prefix}model reply {i}"))
    return turns


def make_submission(
    slot: str = "A",
    final_link: str = GEMINI,
    exchanges: int = 20,
    attachments: list[str] | None = None,
    per_turn_links: bool = True,
) -> ModelSubmission:
    return ModelSubmission(
        model=slot,  # type: ignore[arg-type]
        final_link=final_link,
        turn_links=(
            [TurnLink(turn=i, url=final_link) for i in range(1, exchanges + 1)]
            if per_turn_links
            else []
        ),
        transcript_pdf=f"model_{slot}_transcript.pdf",
        attachments=list(attachments or [f"model_{slot}_output.mp4"]),
        conversation=make_conversation(exchanges, prefix=f"{slot}: "),
    )


def make_rubric(n: int = 20, weight: int = 3, l1: str = "Outcome Quality") -> list[RubricCriterion]:
    return [
        RubricCriterion(
            criterion_id=f"C{i}",
            text=f"The response satisfies requirement {i}.",
            l1_label=l1,
            l2_label=None,
            weight=weight,
        )
        for i in range(1, n + 1)
    ]


def make_task(
    task_id: str = "task-1",
    n_criteria: int = 20,
    exchanges: int = 20,
    likert: int = 4,
    key_turn: int | None = 5,
    dimension_rating_a: int = 4,
    dimension_rating_b: int = 4,
) -> Task:
    rubric = make_rubric(n_criteria)
    dimension_ratings: list[DimensionRating] = []
    for dim in RATING_DIMENSIONS:
        for slot, rating in (("A", dimension_rating_a), ("B", dimension_rating_b)):
            dimension_ratings.append(
                DimensionRating(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dim,
                    rating=rating,
                    not_applicable=False,
                    justification=f"{slot} scored {rating} on {dim} because of X.",
                    relevant_turns=[3],
                )
            )
    return Task(
        task_id=task_id,
        seeded_prompt="Make a short animation explaining how a gyroscope works.",
        user_goal="Produce a classroom-ready teaching aid.",
        target_deliverables=["an 11-second clip", "4K resolution", "consistent colour coding"],
        prompts=[
            Turn(index=i, role="user", text=f"prompt turn {i}") for i in range(1, exchanges + 1)
        ],
        target_outcome=["an 11-second clip", "4K resolution"],
        model_a=make_submission("A", GEMINI, exchanges),
        model_b=make_submission("B", GPT, exchanges),
        key_turn=KeyTurn(turn_index=key_turn, justification="This turn delivers the video."),
        rubric=rubric,
        criterion_ratings=[
            CriterionRating(criterion_id=c.criterion_id, model=slot, score=1)  # type: ignore[arg-type]
            for c in rubric
            for slot in ("A", "B")
        ],
        dimension_ratings=dimension_ratings,
        sxs=Sxs(likert=likert, justification="I prefer Model A because it rendered the clip."),
    )


# ---------------------------------------------------------------------------
# Finding builders
# ---------------------------------------------------------------------------


def issue(category: str = "inaccurate") -> Issue:
    return Issue(
        category=category,
        description=f"criterion has a {category} problem",
        evidence="verbatim quote from the criterion",
    )


def criterion_findings(
    n_criteria: int = 20, categories_by_index: dict[int, list[str]] | None = None
) -> list[CriterionFinding]:
    """One finding per criterion; only the indices named carry issues."""
    categories_by_index = categories_by_index or {}
    out: list[CriterionFinding] = []
    for i in range(1, n_criteria + 1):
        cats = categories_by_index.get(i, [])
        out.append(
            CriterionFinding(
                criterion_id=f"C{i}", issues=[issue(c) for c in cats]
            )
        )
    return out


def l1_findings(n: int = 20, n_incorrect: int = 0) -> list[L1Finding]:
    out: list[L1Finding] = []
    for i in range(1, n + 1):
        wrong = i <= n_incorrect
        out.append(
            L1Finding(
                criterion_id=f"C{i}",
                contributor_label="Outcome Quality",
                auditor_label="Safety" if wrong else "Outcome Quality",
            )
        )
    return out


def weight_findings(
    n: int = 20,
    two_level: int = 0,
    one_level: int = 0,
    same_bucket_delta: int = 0,
    inside_band: int = 0,
) -> list[WeightFinding]:
    """Weight findings by how far the recorded weight sits outside the auditor's band.

    `inside_band` items are the case the point comparison used to flag: the recorded
    weight differs from the weight the auditor would have picked but is inside the
    range the auditor calls defensible, so it is no finding at all.
    """
    out: list[WeightFinding] = []
    for i in range(1, n + 1):
        low, high = 5, 5
        if i <= two_level:
            cb, auditor = 1, 5           # low -> high band, two levels outside
        elif i <= two_level + one_level:
            cb, auditor = 3, 5           # medium -> high band, one level outside
        elif i <= two_level + one_level + same_bucket_delta:
            cb, auditor = 4, 5           # outside the band, zero levels
        elif i <= two_level + one_level + same_bucket_delta + inside_band:
            cb, auditor, low, high = 3, 5, 2, 5   # inside a wide band
        else:
            cb, auditor, low, high = 3, 3, 3, 3
        out.append(
            WeightFinding(
                criterion_id=f"C{i}",
                contributor_weight=cb,
                auditor_weight=auditor,
                defensible_low=low,
                defensible_high=high,
            )
        )
    return out


def criterion_rating_findings(
    n_criteria: int = 20, n_disagreements: int = 0
) -> list[CriterionRatingFinding]:
    out: list[CriterionRatingFinding] = []
    made = 0
    for i in range(1, n_criteria + 1):
        for slot in ("A", "B"):
            disagree = made < n_disagreements
            out.append(
                CriterionRatingFinding(
                    criterion_id=f"C{i}",
                    model=slot,  # type: ignore[arg-type]
                    contributor_score=1,
                    auditor_score=0 if disagree else 1,
                )
            )
            if disagree:
                made += 1
    return out


def dimension_rating_findings(
    n_major: int = 0, n_minor: int = 0, n_both_na: int = 0
) -> list[DimensionRatingFinding]:
    """Builds up to 12 (model, dimension) judgments with controlled deltas."""
    out: list[DimensionRatingFinding] = []
    slots = ("A", "B")
    pairs = [(s, d) for d in RATING_DIMENSIONS for s in slots]
    major, minor, both_na = n_major, n_minor, n_both_na
    for slot, dim in pairs:
        if both_na > 0:
            out.append(
                DimensionRatingFinding(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dim,
                    contributor_na=True,
                    auditor_na=True,
                )
            )
            both_na -= 1
        elif major > 0:
            out.append(
                DimensionRatingFinding(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dim,
                    contributor_rating=2,
                    auditor_rating=5,  # delta 3 -> major
                )
            )
            major -= 1
        elif minor > 0:
            out.append(
                DimensionRatingFinding(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dim,
                    contributor_rating=4,
                    auditor_rating=5,  # delta 1 -> minor
                )
            )
            minor -= 1
        else:
            out.append(
                DimensionRatingFinding(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dim,
                    contributor_rating=5,
                    auditor_rating=5,
                )
            )
    return out
