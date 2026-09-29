"""The blind rating pass: checks 270, 300, and 400.

Volume is the defining property. A 20-criterion task is 40 calls for 270, 12 for
300, and 1 for 400 -- 53 on top of the rubric stage's 21. Nothing is batched:
sharing a call between two judgments lets the first colour the second, and these
are the checks where that matters most.

Abstentions are first-class. `cannot_determine` means the transcript could not
support a judgment, which happens whenever the requirement concerns an artifact
the transcript does not contain. Abstentions leave the denominator and are
reported; if too few judgments survive, the check returns not_evaluated rather
than publishing a rate computed over whichever items happened to be text-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_POLICY, Policy
from .context import EvidenceProfile, profile_evidence
from .findings import CriterionRatingFinding, DimensionRatingFinding, LikertFinding
from .gates import evaluate_check_270, evaluate_check_271, evaluate_check_300, evaluate_check_305, evaluate_check_400
from .llm import ModelClient, ModelRequest, ModelResponse, run_requests
from .models import ModelSlot, ModelSubmission, Task
from .projects import drop_disabled
from .rating_prompts import (
    ADJUDICATION_SCHEMA,
    ADJUDICATION_SYSTEM_PROMPT,
    COLLABORATION_RATING_SCHEMA,
    CRITERION_RATING_SCHEMA,
    DIMENSION_RATING_SCHEMA,
    LIKERT_SCHEMA,
    RATING_SYSTEM_PROMPT,
    assert_blind,
    build_criterion_rating_prompt_with_evidence,
    build_dimension_adjudication_prompt,
    build_dimension_rating_prompt_with_evidence,
    build_likert_prompt_with_evidence,
    build_ranking_adjudication_prompt,
)
from .scoring import CheckVerdict, Measurement, build_verdict
from .taxonomies import RATING_DIMENSIONS

CONFIDENCES = ("low", "medium", "high")
RATING_CHECKS = (270, 271, 300, 305, 400)


@dataclass
class Abstention:
    check_id: int
    item_id: str
    reason: str


@dataclass
class RatingStageResult:
    task_id: str
    verdicts: list[CheckVerdict] = field(default_factory=list)
    criterion_findings: list[CriterionRatingFinding] = field(default_factory=list)
    simplified_criterion_findings: list[CriterionRatingFinding] = field(
        default_factory=list
    )
    dimension_findings: list[DimensionRatingFinding] = field(default_factory=list)
    collaboration_findings: list[DimensionRatingFinding] = field(default_factory=list)
    # One per comparison the task recorded. A task with two comparisons yields
    # two independent Likert judgments, and each is adjudicated and scored on
    # its own.
    likerts: list[LikertFinding] = field(default_factory=list)
    abstentions: list[Abstention] = field(default_factory=list)
    evidence: list[EvidenceProfile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0

    @property
    def likert(self) -> LikertFinding | None:
        """The primary comparison's Likert finding.

        Kept so that readers written when a task had exactly one comparison
        keep working; anything that needs to see every comparison reads
        `likerts`.
        """
        return self.likerts[0] if self.likerts else None

    def likert_for(self, pair: str) -> LikertFinding | None:
        return next((f for f in self.likerts if f.pair == pair), None)

    def abstained(self, check_id: int) -> int:
        return sum(1 for a in self.abstentions if a.check_id == check_id)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "cost_usd": round(self.cost_usd, 4),
            "criterion_judgments": len(self.criterion_findings),
            "dimension_judgments": len(self.dimension_findings),
            "likert_judged": bool(self.likerts),
            "likerts_judged": [f.pair for f in self.likerts],
            "abstentions": {
                "270": self.abstained(270),
                "300": self.abstained(300),
                "400": self.abstained(400),
            },
            "abstention_detail": [
                {"check_id": a.check_id, "item": a.item_id, "reason": a.reason}
                for a in self.abstentions
            ],
            "evidence": [e.to_dict() for e in self.evidence],
            "dropped": list(self.dropped),
            "errors": list(self.errors),
        }


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _confidence(data: dict) -> str:
    c = data.get("confidence")
    return c if c in CONFIDENCES else "low"


def _why_unverifiable(data: dict) -> str:
    """What the judge says it was not shown, preferred over its general reasoning.

    The prompts ask for the missing turn or deliverable to be named in its own
    field, because an abstention whose reason is buried in prose about the item is
    one a reviewer cannot act on -- and acting on it, by raising the render budget
    for the conversations that need it, is the only way the abstention rate comes
    down. `reasoning` remains the fallback for a response that leaves it empty.
    """
    named = _clean(data.get("why_unverifiable"))
    return (named or _clean(data.get("reasoning")))[:200]


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def build_rating_requests(
    task: Task, policy: Policy = DEFAULT_POLICY, guard: bool = True
) -> list[ModelRequest]:
    """All blind-pass requests for a task, in one flat list.

    Every prompt passes the blindness guard before it can be sent. A leak makes
    the audit agree with the contributor while still producing plausible numbers,
    so this fails the run rather than warning.
    """
    requests: list[ModelRequest] = []

    def add(key: str, prompt: str, schema: dict, metadata: dict) -> None:
        if guard:
            report = assert_blind(prompt, task)
            if not report.clean:
                raise RuntimeError(
                    f"blind prompt {key} exposes contributor judgments: {report.leaked}"
                )
        requests.append(
            ModelRequest(
                key=key,
                prompt=prompt,
                schema=schema,
                system=RATING_SYSTEM_PROMPT,
                metadata=metadata,
            )
        )

    for sub in task.submissions_in_scope():
        for criterion in task.rubric:
            prompt, profile = build_criterion_rating_prompt_with_evidence(
                task, criterion, sub, policy
            )
            add(
                f"{task.task_id}::rate::{sub.model}::{criterion.criterion_id}",
                prompt,
                CRITERION_RATING_SCHEMA,
                {
                    "check_id": 270,
                    "model": sub.model,
                    "criterion_id": criterion.criterion_id,
                    # Attached here because this is the only place that knows it.
                    # The judgment and the render that conditioned it are separated
                    # by a queue, a cache, and a worker pool; the request is what
                    # survives all three.
                    "truncated": profile.tail_truncated,
                    "elided_turns": profile.elided_turns,
                },
            )

    if task.simplified is not None:
        for criterion in task.rubric:
            prompt, profile = build_criterion_rating_prompt_with_evidence(
                task, criterion, task.simplified, policy
            )
            add(
                f"{task.task_id}::rate::S::{criterion.criterion_id}",
                prompt,
                CRITERION_RATING_SCHEMA,
                {
                    "check_id": 271,
                    "model": "S",
                    "criterion_id": criterion.criterion_id,
                    "truncated": profile.tail_truncated,
                    "elided_turns": profile.elided_turns,
                },
            )

    for sub in task.submissions_in_scope():
        # No contributor 1-5s for this slot means check 300 has nothing to
        # disagree with. Skip the spend; a separate fill writes the missing
        # ratings rather than pretending they were audited.
        if not any(d.model == sub.model for d in task.dimension_ratings):
            continue
        for dimension in RATING_DIMENSIONS:
            prompt, profile = build_dimension_rating_prompt_with_evidence(
                task, dimension, sub, policy
            )
            add(
                f"{task.task_id}::dimension::{sub.model}::{dimension}",
                prompt,
                DIMENSION_RATING_SCHEMA,
                {
                    "check_id": 300,
                    "model": sub.model,
                    "dimension": dimension,
                    "truncated": profile.tail_truncated,
                    "elided_turns": profile.elided_turns,
                },
            )

    for entry in task.dimension_ratings:
        if entry.dimension != "Collaboration quality":
            continue
        if not task.in_audit_scope(entry.model):
            continue
        sub = task.submission(entry.model)  # type: ignore[arg-type]
        if sub is None:
            continue
        prompt, profile = build_dimension_rating_prompt_with_evidence(
            task, "Collaboration quality", sub, policy
        )
        add(
            f"{task.task_id}::dimension::{sub.model}::Collaboration quality",
            prompt,
            COLLABORATION_RATING_SCHEMA,
            {
                "check_id": 305,
                "model": sub.model,
                "dimension": "Collaboration quality",
                "truncated": profile.tail_truncated,
                "elided_turns": profile.elided_turns,
            },
        )

    # One independent Likert per recorded comparison. Keyed by the pair so that
    # A-vs-B and A-vs-C get their own cached judgment: they are different
    # questions, and a single cache key would serve the first answer to both.
    for comparison in task.comparison_list():
        left = task.submission(comparison.left)
        right = task.submission(comparison.right)
        if left is None or right is None:
            continue
        prompt, profiles = build_likert_prompt_with_evidence(task, policy, comparison)
        by_slot = {p.model: p for p in profiles}
        lp, rp = by_slot.get(comparison.left), by_slot.get(comparison.right)
        add(
            f"{task.task_id}::likert::{comparison.pair_label}",
            prompt,
            LIKERT_SCHEMA,
            {
                "check_id": 400,
                "pair": comparison.pair_label,
                "left": comparison.left,
                "right": comparison.right,
                # Named left/right rather than a/b because the right-hand side
                # is C in half the comparisons this now covers.
                "truncated_a": bool(lp and lp.tail_truncated),
                "truncated_b": bool(rp and rp.tail_truncated),
                "turns_shown_a": lp.turns_rendered if lp else 0,
                "turns_shown_b": rp.turns_rendered if rp else 0,
                "turns_total_a": lp.turns_available if lp else 0,
                "turns_total_b": rp.turns_available if rp else 0,
            },
        )
    return requests


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_criterion_rating(
    response: ModelResponse,
    criterion_id: str,
    model: ModelSlot,
    contributor_score: int | None,
    truncated: bool = False,
) -> tuple[CriterionRatingFinding | None, Abstention | None, str]:
    item = f"{model}::{criterion_id}"
    if not response.ok or response.data is None:
        return None, None, response.error or "no data returned"

    verdict = _clean(response.data.get("verdict"))
    if verdict == "cannot_determine":
        return None, Abstention(270, item, _why_unverifiable(response.data)), ""
    if verdict not in ("pass", "fail"):
        return None, None, f"unknown verdict {verdict!r}"
    if contributor_score is None:
        # No contributor rating means there is nothing to disagree with.
        return None, None, ""

    return (
        CriterionRatingFinding(
            criterion_id=criterion_id,
            model=model,
            contributor_score=contributor_score,
            auditor_score=1 if verdict == "pass" else 0,
            confidence=_confidence(response.data),
            truncated=truncated,
        ),
        None,
        "",
    )


def parse_dimension_rating(
    response: ModelResponse,
    dimension: str,
    model: ModelSlot,
    contributor_rating: int | None,
    contributor_na: bool,
    has_contributor_entry: bool,
    policy: Policy = DEFAULT_POLICY,
    truncated: bool = False,
) -> tuple[DimensionRatingFinding | None, Abstention | None, str]:
    item = f"{model}::{dimension}"
    if not response.ok or response.data is None:
        return None, None, response.error or "no data returned"

    data = response.data
    if data.get("cannot_determine"):
        return None, Abstention(300, item, _why_unverifiable(data)), ""
    if not has_contributor_entry:
        return None, None, ""

    auditor_na = bool(data.get("not_applicable"))
    rating = data.get("rating")
    lo, hi = policy.dimension_rating_scale

    if not auditor_na:
        if not isinstance(rating, int) or not (lo <= rating <= hi):
            return None, None, f"rating {rating!r} outside the {lo}-{hi} scale"
    else:
        rating = None

    return (
        DimensionRatingFinding(
            model=model,
            dimension=dimension,
            contributor_rating=contributor_rating,
            auditor_rating=rating,
            contributor_na=contributor_na,
            auditor_na=auditor_na,
            confidence=_confidence(data),
            truncated=truncated,
        ),
        None,
        "",
    )


def parse_likert(
    response: ModelResponse,
    contributor_likert: int | None,
    policy: Policy = DEFAULT_POLICY,
    render: dict | None = None,
    pair: str = "AB",
) -> tuple[LikertFinding | None, Abstention | None, str]:
    if not response.ok or response.data is None:
        return None, None, response.error or "no data returned"
    data = response.data
    if data.get("cannot_determine"):
        # Abstention item ids are namespaced by pair so a task that abstains on
        # one comparison and judges the other is not read as abstaining wholesale.
        return None, Abstention(400, pair, _why_unverifiable(data)), ""
    if contributor_likert is None:
        return None, None, ""

    value = data.get("likert")
    lo, hi = policy.likert_scale
    if not isinstance(value, int) or not (lo <= value <= hi):
        return None, None, f"likert {value!r} outside the {lo}-{hi} scale"
    meta = render or {}
    return (
        LikertFinding(
            contributor_likert=contributor_likert,
            auditor_likert=value,
            confidence=_confidence(data),
            truncated_a=bool(meta.get("truncated_a")),
            truncated_b=bool(meta.get("truncated_b")),
            turns_shown_a=int(meta.get("turns_shown_a") or 0),
            turns_shown_b=int(meta.get("turns_shown_b") or 0),
            turns_total_a=int(meta.get("turns_total_a") or 0),
            turns_total_b=int(meta.get("turns_total_b") or 0),
            pair=pair,
        ),
        None,
        "",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _thin_evidence_verdict(
    task: Task,
    check_id: int,
    judged: int,
    abstained: int,
    policy: Policy,
) -> CheckVerdict | None:
    """Refuse to publish a rate computed over too few surviving judgments."""
    total = judged + abstained
    if total == 0:
        return build_verdict(
            task_id=task.task_id,
            check_id=check_id,
            band="not_evaluated",
            measurement=Measurement(
                counts={"judged": 0, "abstained": abstained},
                notes="No judgments were returned for this check.",
            ),
            policy=policy,
        )
    ratio = judged / total
    if ratio < policy.min_judged_ratio:
        return build_verdict(
            task_id=task.task_id,
            check_id=check_id,
            band="not_evaluated",
            measurement=Measurement(
                numerator=judged,
                denominator=total,
                rate=ratio,
                threshold=policy.min_judged_ratio,
                counts={"judged": judged, "abstained": abstained},
                notes=(
                    f"Only {judged} of {total} judgments could be made from the "
                    f"transcript ({ratio:.0%}), below the {policy.min_judged_ratio:.0%} "
                    "floor. Usually the deliverables are not present in the transcript. "
                    "A rate over this base would describe only the text-only items."
                ),
            ),
            confidence="low",
            policy=policy,
        )
    return None


def run_rating_stage(
    task: Task,
    client: ModelClient,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 6,
    checks: set[int] | None = None,
) -> RatingStageResult:
    """Run the blind pass.

    `checks`, when given, restricts which of 270/300/400 spend a model call at
    all -- not just which are reported. A check left out gets no requests, so
    its verdict falls out of `_build_rating_verdicts`'s existing thin-evidence
    path as `not_evaluated`, the same as a task with no evidence for it; no
    separate "skipped by policy" case was needed.
    """
    result = RatingStageResult(task_id=task.task_id)
    result.evidence = [profile_evidence(s) for s in task.submissions()]

    if not task.rubric:
        result.errors.append("rubric is empty; no criterion ratings to audit")
    if not task.submissions():
        result.errors.append("no model submissions; nothing to rate")
        result.verdicts = not_evaluated_rating_verdicts(
            task, "no model submissions", policy
        )
        return result

    requests = build_rating_requests(task, policy)
    if checks is not None:
        requests = [r for r in requests if r.metadata.get("check_id") in checks]
    responses = run_requests(client, requests, workers=workers)
    result.calls = len(responses)
    result.cost_usd = sum(r.cost_usd for r in responses)
    result.cached_calls = sum(1 for r in responses if r.cached)

    scores = {(r.criterion_id, r.model): r.score for r in task.criterion_ratings}
    dims = {(d.model, d.dimension): d for d in task.dimension_ratings}

    # Paired positionally: an error response need not echo metadata back.
    for request, response in zip(requests, responses):
        meta = request.metadata
        check_id = meta.get("check_id")

        if check_id == 270:
            cid, model = meta["criterion_id"], meta["model"]
            finding, abstention, error = parse_criterion_rating(
                response,
                cid,
                model,
                scores.get((cid, model)),
                truncated=bool(meta.get("truncated")),
            )
        elif check_id == 271:
            cid, model = meta["criterion_id"], meta["model"]
            finding, abstention, error = parse_criterion_rating(
                response,
                cid,
                model,
                scores.get((cid, model)),
                truncated=bool(meta.get("truncated")),
            )
        elif check_id == 300:
            dimension, model = meta["dimension"], meta["model"]
            entry = dims.get((model, dimension))
            finding, abstention, error = parse_dimension_rating(
                response,
                dimension,
                model,
                entry.rating if entry else None,
                bool(entry.not_applicable) if entry else False,
                entry is not None,
                policy,
                truncated=bool(meta.get("truncated")),
            )
        elif check_id == 305:
            dimension, model = meta["dimension"], meta["model"]
            entry = dims.get((model, dimension))
            finding, abstention, error = parse_dimension_rating(
                response,
                dimension,
                model,
                entry.rating if entry else None,
                bool(entry.not_applicable) if entry else False,
                entry is not None,
                policy,
                truncated=bool(meta.get("truncated")),
            )
        else:
            pair_label = str(meta.get("pair") or "AB")
            recorded = next(
                (c for c in task.comparison_list() if c.pair_label == pair_label),
                None,
            )
            finding, abstention, error = parse_likert(
                response,
                recorded.likert if recorded else None,
                policy,
                render=meta,
                pair=pair_label,
            )

        if error:
            result.errors.append(f"{request.key}: {error}")
            continue
        if abstention is not None:
            result.abstentions.append(abstention)
            continue
        if finding is None:
            result.dropped.append(request.key)
            continue

        if check_id == 270:
            result.criterion_findings.append(finding)
        elif check_id == 271:
            result.simplified_criterion_findings.append(finding)
        elif check_id == 300:
            result.dimension_findings.append(finding)
        elif check_id == 305:
            result.collaboration_findings.append(finding)
        else:
            result.likerts.append(finding)

    run_adjudication(task, result, client, policy, workers=workers)

    result.verdicts = _build_rating_verdicts(task, result, policy)
    return result


# ---------------------------------------------------------------------------
# Adjudication: the informed second pass over what the blind pass disagreed on
# ---------------------------------------------------------------------------


def _adjudication_targets(
    result: RatingStageResult, policy: Policy
) -> list[DimensionRatingFinding]:
    """Dimensions worth spending a confirming call on.

    Only disagreements, because a dimension the two sides rated the same has
    nothing to adjudicate; and not N/A mismatches, which 300 already declines to
    count as defects and which are an applicability question rather than a rating
    one. That keeps the second pass at roughly two to six calls a task against
    the blind pass's twelve.
    """
    if not policy.dimension_disagreement_requires_informed_confirmation:
        return []
    return [
        f
        for f in result.dimension_findings
        if f.comparable and (f.delta or 0) > 0
    ]


def build_adjudication_requests(
    task: Task,
    result: RatingStageResult,
    policy: Policy = DEFAULT_POLICY,
) -> list[ModelRequest]:
    """The informed pass, built outside the blindness guard on purpose.

    Every prompt here shows the contributor's rating and their justification,
    which is exactly what `assert_blind` exists to prevent -- so these requests
    never pass through `build_rating_requests`'s `add`, and the guard is not
    weakened to accommodate them. The two passes stay separable: the blind
    judgment is already recorded on the finding before any of this runs.
    """
    requests: list[ModelRequest] = []
    subs = {s.model: s for s in task.submissions()}
    justifications = {
        (d.model, d.dimension): d.justification for d in task.dimension_ratings
    }

    for f in _adjudication_targets(result, policy):
        sub = subs.get(f.model)
        if sub is None:
            continue
        prompt, profile = build_dimension_adjudication_prompt(
            task,
            f.dimension,
            sub,
            f.contributor_rating,
            justifications.get((f.model, f.dimension), ""),
            f.auditor_rating,
            policy,
        )
        requests.append(
            ModelRequest(
                key=f"{task.task_id}::adjudicate::{f.model}::{f.dimension}",
                prompt=prompt,
                schema=ADJUDICATION_SCHEMA,
                system=ADJUDICATION_SYSTEM_PROMPT,
                metadata={
                    "check_id": 300,
                    "model": f.model,
                    "dimension": f.dimension,
                    # Whether even the wide render left turns out. This is what
                    # decides whether the verdict below is countable, so it has to
                    # travel with the request the same way the blind pass's does.
                    "adjudication_tail_truncated": profile.tail_truncated,
                },
            )
        )

    if policy.ranking_disagreement_requires_informed_confirmation:
        by_pair = {c.pair_label: c for c in task.comparison_list()}
        for likert in result.likerts:
            if likert.delta <= 0:
                continue
            prompt, profiles = build_ranking_adjudication_prompt(
                task,
                likert.contributor_likert,
                likert.auditor_likert,
                policy,
                by_pair.get(likert.pair),
            )
            requests.append(
                ModelRequest(
                    # Keyed by pair: A-vs-B and A-vs-C are different questions
                    # and a shared key would serve the first answer to both.
                    key=f"{task.task_id}::adjudicate::likert::{likert.pair}",
                    prompt=prompt,
                    schema=ADJUDICATION_SCHEMA,
                    system=ADJUDICATION_SYSTEM_PROMPT,
                    metadata={
                        "check_id": 400,
                        "pair": likert.pair,
                        "adjudication_tail_truncated": any(
                            p.tail_truncated for p in profiles
                        ),
                    },
                )
            )
    return requests


def apply_adjudication(finding, data: dict) -> None:
    """Write one adjudication onto the finding it settles.

    An unreadable or missing verdict leaves the finding unadjudicated rather than
    defensible: a call that failed is not evidence the rating was fine, and the
    policy flag decides what an unconfirmed disagreement is worth.
    """
    verdict = _clean(data.get("verdict"))
    if verdict == "cannot_determine":
        finding.adjudication_abstained = True
        finding.why_unadjudicated = _why_unverifiable(data)
        return
    if verdict not in ("defensible", "indefensible"):
        finding.why_unadjudicated = f"unreadable adjudication verdict {verdict!r}"
        return
    finding.adjudication = verdict
    finding.adjudication_reasoning = _clean(data.get("why"))[:400]


def run_adjudication(
    task: Task,
    result: RatingStageResult,
    client: ModelClient,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 6,
) -> None:
    """Confirm, or decline to confirm, every disagreement the blind pass produced.

    Runs in place on `result` before the verdicts are built, so the gates see
    findings that already know whether anyone informed agreed the rating was
    wrong. A task with no disagreements spends nothing here, which is the point:
    the cost is proportional to what is actually in dispute.
    """
    requests = build_adjudication_requests(task, result, policy)
    if not requests:
        return

    responses = run_requests(client, requests, workers=workers)
    result.calls += len(responses)
    result.cost_usd += sum(r.cost_usd for r in responses)
    result.cached_calls += sum(1 for r in responses if r.cached)

    by_dimension = {(f.model, f.dimension): f for f in result.dimension_findings}
    for request, response in zip(requests, responses):
        meta = request.metadata
        target = (
            result.likert_for(str(meta.get("pair") or "AB"))
            if meta.get("check_id") == 400
            else by_dimension.get((meta.get("model"), meta.get("dimension")))
        )
        if target is None:
            continue
        target.adjudication_tail_truncated = bool(
            meta.get("adjudication_tail_truncated")
        )
        if not response.ok or response.data is None:
            target.why_unadjudicated = (
                response.error or "adjudication returned no data"
            )[:200]
            result.errors.append(
                f"{request.key}: {response.error or 'no data returned'}"
            )
            continue
        apply_adjudication(target, response.data)


def _build_rating_verdicts(
    task: Task, result: RatingStageResult, policy: Policy
) -> list[CheckVerdict]:
    verdicts: list[CheckVerdict] = []

    thin = _thin_evidence_verdict(
        task, 270, len(result.criterion_findings), result.abstained(270), policy
    )
    verdicts.append(
        thin
        if thin is not None
        else evaluate_check_270(
            task, result.criterion_findings, policy, abstained=result.abstained(270)
        )
    )

    thin = _thin_evidence_verdict(
        task, 271, len(result.simplified_criterion_findings), result.abstained(271), policy
    )
    verdicts.append(
        thin
        if thin is not None
        else evaluate_check_271(
            task,
            result.simplified_criterion_findings,
            policy,
            abstained=result.abstained(271),
        )
    )

    thin = _thin_evidence_verdict(
        task, 300, len(result.dimension_findings), result.abstained(300), policy
    )
    verdicts.append(
        thin
        if thin is not None
        else evaluate_check_300(
            task, result.dimension_findings, policy, abstained=result.abstained(300)
        )
    )

    thin = _thin_evidence_verdict(
        task, 305, len(result.collaboration_findings), result.abstained(305), policy
    )
    verdicts.append(
        thin
        if thin is not None
        else evaluate_check_305(
            task,
            result.collaboration_findings,
            policy,
            abstained=result.abstained(305),
        )
    )

    if result.likerts:
        verdicts.append(evaluate_check_400(task, result.likerts, policy))
    else:
        reason = (
            "auditor abstained from the comparison"
            if result.abstained(400)
            else "no comparable Likert judgment was produced"
        )
        verdicts.append(
            build_verdict(
                task_id=task.task_id,
                check_id=400,
                band="not_evaluated",
                measurement=Measurement(notes=reason),
                confidence="low",
                policy=policy,
            )
        )
    return drop_disabled(verdicts, task.project_id)


def split_check_270_by_model(
    task: Task, result: RatingStageResult, policy: Policy = DEFAULT_POLICY
) -> dict[ModelSlot, CheckVerdict]:
    """270's per-model breakdown, alongside the pooled verdict `_build_rating_verdicts`
    already publishes.

    Reuses `evaluate_check_270` unchanged, once per model slot, over that
    model's own findings -- the same numbers the pooled verdict's
    `disagreeing_judgments` count already carries, just not split out by slot.
    Abstentions are split the same way `criterion_findings` are, by the model
    named on the finding/abstention rather than by re-deriving it.
    """
    by_model: dict[ModelSlot, CheckVerdict] = {}
    # The competing slots. The simplified trajectory's ratings are audited by
    # their own check, so they are not split out here.
    for slot in (sub.model for sub in task.submissions_in_scope()):
        findings = [f for f in result.criterion_findings if f.model == slot]
        abstained = sum(
            1 for a in result.abstentions if a.check_id == 270 and a.item_id.startswith(f"{slot}::")
        )
        # Same thin-evidence floor the pooled verdict applies: without it, a
        # model with zero surviving judgments and any abstentions reports
        # `clean` here even though nothing was actually checked.
        thin = _thin_evidence_verdict(task, 270, len(findings), abstained, policy)
        by_model[slot] = thin or evaluate_check_270(  # type: ignore[index]
            task, findings, policy, abstained=abstained
        )
    return by_model


def not_evaluated_rating_verdicts(
    task: Task, reason: str, policy: Policy = DEFAULT_POLICY
) -> list[CheckVerdict]:
    return [
        build_verdict(
            task_id=task.task_id,
            check_id=check_id,
            band="not_evaluated",
            measurement=Measurement(notes=reason),
            policy=policy,
        )
        for check_id in RATING_CHECKS
    ]


def estimate_rating_calls(task: Task) -> int:
    """The blind pass: one call per judgment, nothing batched.

    Deliberately still only the blind pass. The confirming round's size depends on
    what the blind pass disagreed about, which is not knowable in advance, so it is
    estimated separately rather than folded in here where it would turn an exact
    count into a guess.
    """
    models = len(task.submissions_in_scope())
    dim_models = {
        s.model
        for s in task.submissions_in_scope()
        if any(d.model == s.model for d in task.dimension_ratings)
    }
    n = models * len(task.rubric) + len(dim_models) * len(RATING_DIMENSIONS)
    n += sum(
        1
        for c in task.comparison_list()
        if task.submission(c.left) and task.submission(c.right)
    )
    if task.simplified is not None:
        n += len(task.rubric)
    n += sum(
        1
        for d in task.dimension_ratings
        if d.dimension == "Collaboration quality" and task.in_audit_scope(d.model)
    )
    return n


def estimate_adjudication_calls(task: Task, policy: Policy = DEFAULT_POLICY) -> int:
    """An upper bound on the confirming pass: every dimension in dispute.

    The real number is one call per disagreement, and historically about a third of
    dimensions disagree, so this over-states the bill rather than surprising the
    operator with one. A task where the two sides agree everywhere spends nothing
    here.
    """
    models = len(task.submissions_in_scope())
    calls = 0
    if policy.dimension_disagreement_requires_informed_confirmation:
        calls += models * len(RATING_DIMENSIONS)
    if policy.ranking_disagreement_requires_informed_confirmation:
        calls += sum(
            1
            for c in task.comparison_list()
            if task.submission(c.left) and task.submission(c.right)
        )
    return calls
