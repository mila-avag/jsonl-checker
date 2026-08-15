"""The rubric-authoring stage: one model call per criterion, plus one coverage pass.

Produces checks 200, 230, 240, 250, and 260.

Response validation is strict and one-directional: anything the model returns
that is not in the closed set, or not backed by a verbatim quote, is dropped and
counted. Dropping under-reports defects, which is the safe direction. Silently
accepting an invented category would put an unexplainable string in a customer's
error-code column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_POLICY, POLICY_VERSION, Policy
from .findings import (
    CoverageFinding,
    CriterionFinding,
    Issue,
    L1Finding,
    L2Finding,
    WeightFinding,
)
from .gates import Census, build_census, evaluate_check_200, evaluate_check_210, \
    evaluate_check_260, evaluate_rubric_quality_gates
from .llm import ModelClient, ModelRequest, ModelResponse, run_requests
from .models import RubricCriterion, Task
from .prompts import (
    COVERAGE_SCHEMA,
    CRITERION_SCHEMA,
    PER_CRITERION_CATEGORIES,
    SYSTEM_PROMPT,
    assert_independent,
    build_coverage_prompt,
    build_criterion_prompt,
)
from .scoring import CheckVerdict, Measurement, build_verdict
from .taxonomies import (
    L1_LABELS,
    L2_LABELS,
    WEIGHT_CONDITIONED_CATEGORIES,
    weight_confirms_double_negative,
)

CONFIDENCES = ("low", "medium", "high")


@dataclass
class DroppedItem:
    criterion_id: str
    reason: str
    detail: str = ""


@dataclass
class CriterionAudit:
    criterion_id: str
    finding: CriterionFinding | None = None
    l1: L1Finding | None = None
    l2: L2Finding | None = None
    weight: WeightFinding | None = None
    error: str = ""
    dropped: list[DroppedItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.finding is not None and not self.error


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


MIN_EVIDENCE_SPAN = 12

_QUOTED_SPAN = re.compile(r"[\"\u201c]([^\"\u201d]{%d,})[\"\u201d]" % MIN_EVIDENCE_SPAN)
_SPAN_SEPARATORS = re.compile(r"\s*(?:/|\||;|\bvs\.?\b|\band\b)\s*", re.IGNORECASE)


def _evidence_spans(evidence: str) -> list[str]:
    """Candidate quotes inside an evidence string.

    `overlapping` is naturally evidenced by citing both criteria, so evidence
    arrives as something like `C5: "..." / C6: "..."`. Requiring the whole string
    to be a substring of one criterion rejects exactly the issue type that needs
    two quotes, so quoted spans and separator-delimited fragments are each tried
    on their own.
    """
    spans = [evidence]
    spans += [m.group(1) for m in _QUOTED_SPAN.finditer(evidence)]
    spans += _SPAN_SEPARATORS.split(evidence)
    return [s for s in spans if s and s.strip()]


def _evidence_supports(evidence: str, criterion: RubricCriterion, task: Task) -> bool:
    """An issue is only real if its quote actually appears in the rubric.

    Guards against a fabricated quote presented as evidence. Matching is
    whitespace- and case-insensitive, and the whole rubric is searched because
    `overlapping` legitimately quotes a sibling criterion.
    """
    if not evidence.strip():
        return False
    haystacks = [" ".join(c.text.lower().split()) for c in task.rubric]
    haystacks.append(" ".join(criterion.text.lower().split()))

    for span in _evidence_spans(evidence):
        needle = " ".join(span.lower().split()).strip(" .,:;\"'")
        if len(needle) < MIN_EVIDENCE_SPAN:
            continue
        if any(needle in h for h in haystacks):
            return True
    return False


def parse_criterion_response(
    response: ModelResponse,
    criterion: RubricCriterion,
    task: Task,
    policy: Policy = DEFAULT_POLICY,
) -> CriterionAudit:
    audit = CriterionAudit(criterion_id=criterion.criterion_id)
    if not response.ok or response.data is None:
        audit.error = response.error or "no data returned"
        return audit

    data = response.data
    confidence = data.get("confidence")
    if confidence not in CONFIDENCES:
        confidence = "low"

    issues: list[Issue] = []
    unverifiable: list[Issue] = []
    for raw in data.get("issues") or []:
        if not isinstance(raw, dict):
            audit.dropped.append(DroppedItem(criterion.criterion_id, "malformed_issue"))
            continue
        category = _clean_text(raw.get("category"))
        evidence = _clean_text(raw.get("evidence"))
        description = _clean_text(raw.get("description"))

        if category not in PER_CRITERION_CATEGORIES:
            audit.dropped.append(
                DroppedItem(criterion.criterion_id, "unknown_category", category)
            )
            continue
        # An issue the auditor says it could not confirm is routed out of the
        # counted list here, before the evidence gates, because its evidence is by
        # definition the thing it was not shown: a value inside an input file the
        # prompt only names, or an instruction in a prompt turn the context block
        # truncated. Sent through the quote check it would be dropped as fabricated
        # evidence, which reads in the report as a judge that invented a quote
        # rather than one that told us what it was missing.
        if raw.get("unverifiable"):
            unverifiable.append(
                Issue(
                    category=category,
                    description=description,
                    evidence=evidence,
                    why_unverifiable=_clean_text(raw.get("why_unverifiable")),
                    confidence=confidence,
                )
            )
            continue
        # The spec's framing/double negative needs a negative weight as well as
        # negative phrasing, and the prompt withholds the weight so checks 200 and
        # 260 stay independent. The model reports the phrasing; the weight half is
        # decided here, where the contributor's own value is available.
        if category in WEIGHT_CONDITIONED_CATEGORIES and not (
            weight_confirms_double_negative(criterion.weight)
        ):
            audit.dropped.append(
                DroppedItem(
                    criterion.criterion_id,
                    "weight_not_negative",
                    f"{category} (recorded weight {criterion.weight})",
                )
            )
            continue
        if not evidence:
            audit.dropped.append(
                DroppedItem(criterion.criterion_id, "no_evidence", category)
            )
            continue
        if not _evidence_supports(evidence, criterion, task):
            audit.dropped.append(
                DroppedItem(criterion.criterion_id, "unverifiable_quote", evidence[:80])
            )
            continue
        overlaps_with: list[str] = []
        if category == "overlapping":
            known = task.criterion_ids()
            overlaps_with = sorted(
                {
                    _clean_text(o)
                    for o in (raw.get("overlaps_with") or [])
                    if _clean_text(o) in known and _clean_text(o) != criterion.criterion_id
                }
            )
        issues.append(
            Issue(
                category=category,
                description=description,
                evidence=evidence,
                confidence=confidence,
                overlaps_with=overlaps_with,
            )
        )

    # Deduplicate: two issues of the same category on one criterion still describe
    # one defect, and the census counts criteria anyway.
    seen: set[str] = set()
    deduped: list[Issue] = []
    for issue in issues:
        if issue.category in seen:
            audit.dropped.append(
                DroppedItem(criterion.criterion_id, "duplicate_category", issue.category)
            )
            continue
        seen.add(issue.category)
        deduped.append(issue)

    audit.finding = CriterionFinding(
        criterion_id=criterion.criterion_id,
        issues=deduped,
        unverifiable_issues=unverifiable,
        confidence=confidence,
    )

    auditor_label = _clean_text(data.get("l1_label"))
    if auditor_label in L1_LABELS and criterion.l1_label:
        audit.l1 = L1Finding(
            criterion_id=criterion.criterion_id,
            contributor_label=criterion.l1_label,
            auditor_label=auditor_label,
            confidence=confidence,
        )
    elif criterion.l1_label and auditor_label not in L1_LABELS:
        audit.dropped.append(
            DroppedItem(criterion.criterion_id, "unknown_l1_label", auditor_label)
        )

    auditor_l2 = _clean_text(data.get("l2_label"))
    if auditor_l2 in L2_LABELS and criterion.l2_label:
        audit.l2 = L2Finding(
            criterion_id=criterion.criterion_id,
            contributor_label=criterion.l2_label,
            auditor_label=auditor_l2,
            confidence=confidence,
        )
    elif criterion.l2_label and auditor_l2 not in L2_LABELS:
        audit.dropped.append(
            DroppedItem(criterion.criterion_id, "unknown_l2_label", auditor_l2)
        )

    lo, hi = policy.weight_scale

    def _weight_in_scale(value: Any) -> bool:
        return isinstance(value, int) and lo <= value <= hi

    auditor_weight = data.get("weight")
    band_low = data.get("weight_defensible_low")
    band_high = data.get("weight_defensible_high")
    if criterion.weight is not None:
        if not _weight_in_scale(auditor_weight):
            audit.dropped.append(
                DroppedItem(criterion.criterion_id, "invalid_weight", str(auditor_weight))
            )
        elif not (_weight_in_scale(band_low) and _weight_in_scale(band_high)):
            # Falling back to the auditor's point weight is exactly the comparison
            # that made check 260 flag every task, so a response that does not state
            # a defensible range leaves the check's denominator instead.
            audit.dropped.append(
                DroppedItem(
                    criterion.criterion_id,
                    "invalid_weight_band",
                    f"{band_low}-{band_high}",
                )
            )
        else:
            audit.weight = WeightFinding(
                criterion_id=criterion.criterion_id,
                contributor_weight=criterion.weight,
                auditor_weight=auditor_weight,
                defensible_low=band_low,
                defensible_high=band_high,
                confidence=confidence,
            )
    return audit


def parse_coverage_response(
    response: ModelResponse, task: Task
) -> tuple[CoverageFinding, list[DroppedItem], str]:
    coverage = CoverageFinding()
    dropped: list[DroppedItem] = []
    if not response.ok or response.data is None:
        return coverage, dropped, response.error or "no data returned"

    for raw in response.data.get("missing") or []:
        if not isinstance(raw, dict):
            dropped.append(DroppedItem("", "malformed_gap"))
            continue
        requirement = _clean_text(raw.get("requirement"))
        criticality = _clean_text(raw.get("criticality"))
        evidence = _clean_text(raw.get("evidence"))
        if not requirement:
            dropped.append(DroppedItem("", "empty_requirement"))
            continue
        if criticality not in ("critical", "non_critical"):
            dropped.append(DroppedItem("", "unknown_criticality", criticality))
            continue
        if not evidence:
            dropped.append(DroppedItem("", "no_evidence", requirement[:80]))
            continue
        if criticality == "critical":
            coverage.missing_critical.append(requirement)
        else:
            coverage.missing_non_critical.append(requirement)
    return coverage, dropped, ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_criterion_requests(
    task: Task, policy: Policy = DEFAULT_POLICY, guard: bool = True
) -> list[ModelRequest]:
    requests: list[ModelRequest] = []
    for criterion in task.rubric:
        prompt = build_criterion_prompt(task, criterion, policy)
        if guard:
            leak = assert_independent(prompt, task)
            if not leak.clean:
                raise RuntimeError(
                    f"prompt for {criterion.criterion_id} leaks contributor values: "
                    f"{leak.leaked}"
                )
        requests.append(
            ModelRequest(
                key=f"{task.task_id}::criterion::{criterion.criterion_id}",
                prompt=prompt,
                schema=CRITERION_SCHEMA,
                system=SYSTEM_PROMPT,
                metadata={"task_id": task.task_id, "criterion_id": criterion.criterion_id},
            )
        )
    return requests


def build_coverage_request(task: Task, policy: Policy = DEFAULT_POLICY) -> ModelRequest:
    return ModelRequest(
        key=f"{task.task_id}::coverage",
        prompt=build_coverage_prompt(task, policy),
        schema=COVERAGE_SCHEMA,
        system=SYSTEM_PROMPT,
        metadata={"task_id": task.task_id},
    )


@dataclass
class RubricStageResult:
    task_id: str
    verdicts: list[CheckVerdict] = field(default_factory=list)
    census: Census | None = None
    audits: list[CriterionAudit] = field(default_factory=list)
    coverage: CoverageFinding = field(default_factory=CoverageFinding)
    dropped: list[DroppedItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0

    @property
    def audited_ids(self) -> list[str]:
        return [a.criterion_id for a in self.audits if a.ok]

    @property
    def failed_ids(self) -> list[str]:
        return [a.criterion_id for a in self.audits if not a.ok]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "criteria_audited": len(self.audited_ids),
            "criteria_failed": self.failed_ids,
            "coverage": {
                "missing_critical": list(self.coverage.missing_critical),
                "missing_non_critical": list(self.coverage.missing_non_critical),
            },
            "census": (
                self.census.severity_counts() if self.census else {}
            ),
            "issue_categories": (
                dict(self.census.category_counts) if self.census else {}
            ),
            # The per-criterion detail is what makes a gate's percentage
            # reviewable: without the quote behind each issue, a reader cannot
            # tell a real defect from a miscount.
            "findings": [
                {
                    "criterion_id": audit.criterion_id,
                    "issues": [
                        {
                            "category": issue.category,
                            "severity": issue.severity,
                            "description": issue.description,
                            "evidence": issue.evidence,
                            "confidence": issue.confidence,
                        }
                        for issue in (audit.finding.issues if audit.finding else [])
                    ],
                    # Recorded beside the counted issues rather than inside them:
                    # these are the allegations the auditor withdrew for lack of the
                    # prompt turn or input file they turn on, and a rubric that
                    # collects several of them is telling us the context block, not
                    # the contributor, is what needs fixing.
                    "unverifiable_issues": [
                        {
                            "category": issue.category,
                            "severity": issue.severity,
                            "description": issue.description,
                            "why_unverifiable": issue.why_unverifiable,
                        }
                        for issue in (
                            audit.finding.unverifiable_issues if audit.finding else []
                        )
                    ],
                    "l1": (
                        {
                            "contributor": audit.l1.contributor_label,
                            "auditor": audit.l1.auditor_label,
                            "incorrect": audit.l1.incorrect,
                        }
                        if audit.l1
                        else None
                    ),
                    "l2": (
                        {
                            "contributor": audit.l2.contributor_label,
                            "auditor": audit.l2.auditor_label,
                            "incorrect": audit.l2.incorrect,
                        }
                        if audit.l2
                        else None
                    ),
                    "weight": (
                        {
                            "contributor": audit.weight.contributor_weight,
                            "auditor": audit.weight.auditor_weight,
                            "defensible_low": audit.weight.band[0],
                            "defensible_high": audit.weight.band[1],
                            "in_band": audit.weight.in_band,
                            "delta": audit.weight.raw_delta,
                        }
                        if audit.weight
                        else None
                    ),
                    "error": audit.error,
                }
                for audit in self.audits
            ],
            "dropped": [
                {"criterion_id": d.criterion_id, "reason": d.reason, "detail": d.detail}
                for d in self.dropped
            ],
            "errors": list(self.errors),
            "cost_usd": round(self.cost_usd, 4),
            "calls": self.calls,
            "cached_calls": self.cached_calls,
        }


def run_rubric_stage(
    task: Task,
    client: ModelClient,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 4,
    run_coverage: bool = True,
) -> RubricStageResult:
    result = RubricStageResult(task_id=task.task_id)
    if not task.rubric:
        result.errors.append("rubric is empty; nothing to audit")
        return result

    by_id = {c.criterion_id: c for c in task.rubric}
    requests = build_criterion_requests(task, policy)
    if run_coverage:
        requests.append(build_coverage_request(task, policy))

    responses = run_requests(client, requests, workers=workers)
    result.calls = len(responses)
    result.cost_usd = sum(r.cost_usd for r in responses)
    result.cached_calls = sum(1 for r in responses if r.cached)

    coverage_response: ModelResponse | None = None
    # Paired positionally against the requests, never by reading metadata off the
    # response: a client returning an error need not echo metadata back, and
    # routing on it silently misfiles every failed call.
    for request, response in zip(requests, responses):
        criterion_id = request.metadata.get("criterion_id")
        if criterion_id is None:
            coverage_response = response
            continue
        audit = parse_criterion_response(response, by_id[criterion_id], task, policy)
        result.audits.append(audit)
        result.dropped.extend(audit.dropped)
        if audit.error:
            result.errors.append(f"{criterion_id}: {audit.error}")

    if coverage_response is not None:
        coverage, dropped, error = parse_coverage_response(coverage_response, task)
        result.coverage = coverage
        result.dropped.extend(dropped)
        if error:
            result.errors.append(f"coverage: {error}")

    findings = [a.finding for a in result.audits if a.finding is not None]
    audited = set(result.audited_ids)

    # A criterion whose call failed was not audited. Leaving it in the denominator
    # would dilute the rate with criteria nobody looked at, so it is excluded and
    # reported.
    result.census = build_census(
        task, findings, result.coverage, audited_ids=audited or None
    )

    result.verdicts = list(evaluate_rubric_quality_gates(task, result.census, policy))
    result.verdicts.append(
        evaluate_check_200(
            task,
            [a.l1 for a in result.audits if a.l1 is not None],
            policy,
        )
    )
    result.verdicts.append(
        evaluate_check_210(
            task,
            [a.l2.criterion_id for a in result.audits if a.l2 is not None and a.l2.incorrect],
            policy,
        )
    )
    result.verdicts.append(
        evaluate_check_260(
            task,
            [a.weight for a in result.audits if a.weight is not None],
            policy,
        )
    )

    if result.failed_ids:
        note = (
            f"{len(result.failed_ids)} of {len(task.rubric)} criteria could not be "
            f"audited and are excluded from the denominator"
        )
        for verdict in result.verdicts:
            verdict.measurement.notes = (
                f"{verdict.measurement.notes} {note}".strip()
            )
            verdict.confidence = "low"

    return result


def not_evaluated_rubric_verdicts(
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
        for check_id in (200, 210, 230, 240, 250, 260)
    ]


def stage_policy_version(policy: Policy) -> str:
    return f"{POLICY_VERSION}:rubric-stage-1"
