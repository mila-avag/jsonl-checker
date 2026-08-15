"""Deterministic aggregation: every count, percentage, and threshold.

Two rules govern this module.

Counting: 230/240/250 count *criteria*, not issues. A criterion with three major
issues contributes 1 to the numerator, not 3. A criterion with one major and one
minor issue counts once in all three gates.

Monotonicity: because inclusion widens across 230 -> 240 -> 250, the numerator is
non-decreasing. It is asserted, not assumed. If it regresses the census is
corrupted and the run should abort rather than publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Literal

from .config import DEFAULT_POLICY, Policy
from .env_context import FileReferenceScan
from .findings import (
    ArtifactFinding,
    AutofailFinding,
    CoverageFinding,
    CriterionFinding,
    CriterionRatingFinding,
    DimensionRatingFinding,
    DomainRelevanceFinding,
    EnvironmentContextFinding,
    InversionFinding,
    JustificationFinding,
    KeyTurnJustificationFinding,
    L1Finding,
    LikertFinding,
    PromptConsistencyFinding,
    RelevantTurnFinding,
    TargetOutcomeFinding,
    VerdictFinding,
    WeightFinding,
)
from .models import Task
from .scoring import Band, CheckVerdict, Measurement, build_verdict
from .taxonomies import SEVERITY_RANK, severity_of


class CensusCorrupted(RuntimeError):
    pass


def _lowest_confidence(values) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    vals = [v for v in values if v in order]
    if not vals:
        return "high"
    return min(vals, key=lambda v: order[v])


def _nonzero_band(rate: float, fail_threshold: float) -> Band:
    """Shared band logic for a rate gate.

    Fail thresholds are inclusive on the fail side. The non-fail band is any
    nonzero rate below the threshold: 240 reads "between 1% and 15%" while 230 and
    250 read "more than 0%", and a literal 1% floor would leave a single issue in
    a small rubric with no representable score.
    """
    if rate >= fail_threshold:
        return "fail"
    if rate > 0:
        return "non_fail"
    return "clean"


# ---------------------------------------------------------------------------
# Census (feeds 230 / 240 / 250)
# ---------------------------------------------------------------------------


@dataclass
class Census:
    denominator: int
    criteria_with_major: list[str] = field(default_factory=list)
    criteria_with_moderate: list[str] = field(default_factory=list)
    criteria_with_minor: list[str] = field(default_factory=list)
    missing_critical: list[str] = field(default_factory=list)
    missing_non_critical: list[str] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    # `category_counts` pools every issue category over the whole rubric, so it can
    # say that some criterion carried "unverifiable" and never which one. Beside a
    # flat list of counted criteria that invites a reader to pair the two off and
    # attribute a category to a criterion that never had it, which is the same
    # misattribution 450 made with its conditions.
    categories_by_criterion: dict[str, list[str]] = field(default_factory=dict)
    confidence: str = "high"

    def numerator_items(self, gate: Literal[230, 240, 250]) -> list[str]:
        """Criteria counted by a gate, plus the absent criteria it includes."""
        if gate == 230:
            items = set(self.criteria_with_major)
            absent = list(self.missing_critical)
        elif gate == 240:
            items = set(self.criteria_with_major) | set(self.criteria_with_moderate)
            absent = list(self.missing_critical) + list(self.missing_non_critical)
        else:
            items = (
                set(self.criteria_with_major)
                | set(self.criteria_with_moderate)
                | set(self.criteria_with_minor)
            )
            absent = list(self.missing_critical) + list(self.missing_non_critical)
        return sorted(items) + [f"missing::{m}" for m in absent]

    def numerator(self, gate: Literal[230, 240, 250]) -> int:
        return len(self.numerator_items(gate))

    def absent_counted_by(self, gate: Literal[230, 240, 250]) -> list[str]:
        """Criteria that do not exist and are counted by this gate."""
        if gate == 230:
            return list(self.missing_critical)
        return list(self.missing_critical) + list(self.missing_non_critical)

    def severity_counts(self) -> dict[str, int]:
        return {
            "major": len(self.criteria_with_major),
            "moderate": len(self.criteria_with_moderate),
            "minor": len(self.criteria_with_minor),
            "missing_critical": len(self.missing_critical),
            "missing_non_critical": len(self.missing_non_critical),
        }

    def severities_by_item(
        self, gate: Literal[230, 240, 250]
    ) -> dict[str, list[str]]:
        """Which severity put each counted item in this gate's numerator.

        `severity_counts` is a task-wide tally and `numerator_items` a flat list, so
        neither can say that the one major sits on this criterion and the twelve
        moderates on those. Only the pairing can, and only the pairing is safe to
        show next to a criterion's name.
        """
        counted = {
            230: ("major",),
            240: ("major", "moderate"),
            250: ("major", "moderate", "minor"),
        }[gate]
        by_severity = {
            "major": self.criteria_with_major,
            "moderate": self.criteria_with_moderate,
            "minor": self.criteria_with_minor,
        }
        out: dict[str, list[str]] = {}
        for severity in counted:
            for criterion_id in by_severity[severity]:
                reasons = out.setdefault(criterion_id, [])
                if severity not in reasons:
                    reasons.append(severity)
        for criterion_id in self.absent_counted_by(gate):
            label = (
                "missing_critical"
                if criterion_id in self.missing_critical
                else "missing_non_critical"
            )
            out[f"missing::{criterion_id}"] = [label]
        return {k: out[k] for k in sorted(out)}


def _overlap_links(findings: list[CriterionFinding]) -> dict[str, list[str]]:
    """Criteria whose only issue at or above `overlapping`'s severity tier is the
    overlap itself, mapped to the sibling ids they name.

    A criterion carrying an independent moderate or major issue alongside the
    overlap is excluded here and counted on its own, per tab 2 row 65: only a
    *pure* overlap collapses into the shared error.
    """
    tier = SEVERITY_RANK[severity_of("overlapping")]
    links: dict[str, list[str]] = {}
    for f in findings:
        at_or_above = {
            i.category for i in f.issues if SEVERITY_RANK[severity_of(i.category)] >= tier
        }
        if at_or_above != {"overlapping"}:
            continue
        overlap_issue = next(i for i in f.issues if i.category == "overlapping")
        links[f.criterion_id] = list(overlap_issue.overlaps_with)
    return links


def _overlap_components(links: dict[str, list[str]]) -> list[list[str]]:
    """Connected components of `links`, read as undirected: a one-directional
    report is enough to join two criteria into the same set."""
    visited: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(links):
        if start in visited:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbour in links.get(node, ()):
                if neighbour in links and neighbour not in visited:
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def build_census(
    task: Task,
    findings: list[CriterionFinding],
    coverage: CoverageFinding | None = None,
    audited_ids: set[str] | None = None,
) -> Census:
    """Build the shared per-criterion census.

    The denominator is the criteria the contributor wrote. `audited_ids` narrows
    it when some criteria could not be audited: leaving those in would dilute the
    rate with criteria nobody looked at, biasing every gate toward passing.
    """
    coverage = coverage or CoverageFinding()
    denominator = len(audited_ids) if audited_ids is not None else len(task.rubric)
    census = Census(denominator=denominator)

    known = task.criterion_ids()
    for f in findings:
        if f.criterion_id not in known:
            raise CensusCorrupted(
                f"finding references unknown criterion_id {f.criterion_id!r}"
            )
        sevs = f.severities()
        if "major" in sevs:
            census.criteria_with_major.append(f.criterion_id)
        if "minor" in sevs:
            census.criteria_with_minor.append(f.criterion_id)
        for issue in f.issues:
            census.category_counts[issue.category] = (
                census.category_counts.get(issue.category, 0) + 1
            )
            categories = census.categories_by_criterion.setdefault(f.criterion_id, [])
            if issue.category not in categories:
                categories.append(issue.category)

    # Overlap is the one category the spec counts by set rather than by
    # criterion: "each set of overlapping criteria counts as one error." Pure
    # overlap criteria are pooled into one entry per connected component before
    # anything else joins the moderate bucket individually.
    pure_overlap = _overlap_links(findings)
    pooled_ids: set[str] = set()
    for component in _overlap_components(pure_overlap):
        pooled_ids.update(component)
        entry = component[0] if len(component) == 1 else f"overlap::{'+'.join(component)}"
        census.criteria_with_moderate.append(entry)
        if entry not in census.categories_by_criterion:
            census.categories_by_criterion[entry] = ["overlapping"]
    for f in findings:
        if f.criterion_id in pooled_ids:
            continue
        if "moderate" in f.severities():
            census.criteria_with_moderate.append(f.criterion_id)

    census.missing_critical = list(coverage.missing_critical)
    census.missing_non_critical = list(coverage.missing_non_critical)
    census.confidence = _lowest_confidence([f.confidence for f in findings])

    n230, n240, n250 = (census.numerator(g) for g in (230, 240, 250))
    if not (n230 <= n240 <= n250):
        raise CensusCorrupted(
            f"monotonicity violated: 230={n230}, 240={n240}, 250={n250}"
        )
    return census


# ---------------------------------------------------------------------------
# 200 -- L1 labels
# ---------------------------------------------------------------------------


def evaluate_check_200(
    task: Task, findings: list[L1Finding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    labeled = len(findings)
    unlabeled = len(task.rubric) - labeled
    incorrect = [f.criterion_id for f in findings if f.incorrect]
    rate = (len(incorrect) / labeled) if labeled else 0.0
    band = _nonzero_band(rate, policy.l1_fail_rate)
    return build_verdict(
        task_id=task.task_id,
        check_id=200,
        band=band,
        measurement=Measurement(
            numerator=len(incorrect),
            denominator=labeled,
            rate=rate,
            threshold=policy.l1_fail_rate,
            counts={"labeled_criteria": labeled, "unlabeled_excluded": unlabeled},
            notes="Unlabeled criteria are excluded from both numerator and denominator.",
        ),
        contributing_items=incorrect,
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 210 -- L2 labels
# ---------------------------------------------------------------------------


def evaluate_check_210(
    task: Task,
    incorrect_labels: list[str] | None = None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not policy.l2_labels_configured:
        return build_verdict(
            task_id=task.task_id,
            check_id=210,
            band="not_evaluated",
            measurement=Measurement(
                notes="L2 taxonomy unconfigured. Scoring clean here would understate "
                "defect rates."
            ),
            policy=policy,
        )
    incorrect = list(incorrect_labels or [])
    # Zero tolerance but never fatal: one incorrect label anywhere is a non-fail.
    band: Band = "non_fail" if incorrect else "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=210,
        band=band,
        measurement=Measurement(numerator=len(incorrect), counts={"incorrect": len(incorrect)}),
        contributing_items=incorrect,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 230 / 240 / 250 -- rubric quality gates
# ---------------------------------------------------------------------------

_CENSUS_THRESHOLD_FIELD = {
    230: "rubric_major_fail_rate",
    240: "rubric_major_moderate_fail_rate",
    250: "rubric_all_fail_rate",
}


def evaluate_census_gate(
    task: Task,
    census: Census,
    gate: Literal[230, 240, 250],
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    threshold = getattr(policy, _CENSUS_THRESHOLD_FIELD[gate])
    numerator = census.numerator(gate)
    absent = len(census.absent_counted_by(gate))
    denominator = census.denominator + (absent if policy.coverage_in_denominator else 0)
    rate = (numerator / denominator) if denominator else 0.0
    band = _nonzero_band(rate, threshold)

    notes = (
        "Denominator is the number of criteria the contributor wrote. "
        "Numerator counts criteria, not issues."
    )
    counted_items = census.numerator_items(gate)
    counts = census.severity_counts()
    counts["absent_criteria_counted"] = absent
    counts["coverage_in_denominator"] = policy.coverage_in_denominator
    # The tallies above are task-wide and `contributing_items` is a flat list, so a
    # reader lining the two up can hang the major on a criterion that only carried a
    # moderate. These maps are what should be shown against a criterion's name.
    counts["severities_by_item"] = census.severities_by_item(gate)
    counts["categories_by_item"] = {
        item: list(census.categories_by_criterion[item])
        for item in counted_items
        if item in census.categories_by_criterion
    }
    if rate > 1.0:
        # Publishing "110% of criteria are defective" is indefensible even when
        # the band is correct, so it is surfaced rather than quietly emitted.
        counts["rate_exceeds_100_percent"] = True
        notes += (
            f" Rate exceeds 100% because {absent} absent criteria raise the "
            "numerator without raising the denominator; report the counts, not the rate."
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=gate,
        band=band,
        measurement=Measurement(
            numerator=numerator,
            denominator=denominator,
            rate=rate,
            threshold=threshold,
            counts=counts,
            notes=notes,
        ),
        contributing_items=counted_items,
        confidence=census.confidence,
        policy=policy,
    )


def evaluate_rubric_quality_gates(
    task: Task, census: Census, policy: Policy = DEFAULT_POLICY
) -> list[CheckVerdict]:
    verdicts = [evaluate_census_gate(task, census, g, policy) for g in (230, 240, 250)]
    nums = [v.measurement.numerator or 0 for v in verdicts]
    if not (nums[0] <= nums[1] <= nums[2]):
        raise CensusCorrupted(f"gate numerators are not monotonic: {nums}")
    return verdicts


# ---------------------------------------------------------------------------
# 260 -- weights
# ---------------------------------------------------------------------------


def evaluate_check_260(
    task: Task, findings: list[WeightFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Count weights the customer's definitions cannot support.

    A weight inside the auditor's defensible range is not counted at all, however
    far it sits from the weight the auditor would itself have picked. Only a weight
    outside the range is a finding, and the distance *outside the range* -- not the
    distance from a point -- decides whether it counts toward the two-level gate or
    only the any-level one.
    """
    buckets = policy.weight_buckets
    denominator = len(findings)

    two_level: list[str] = []
    any_level: list[str] = []
    out_of_band: list[str] = []
    reasons: dict[str, list[str]] = {}
    point_gap = 0

    for f in findings:
        if f.raw_delta != 0:
            point_gap += 1
        if f.in_band:
            continue
        out_of_band.append(f.criterion_id)
        reasons.setdefault(f.criterion_id, []).append("out_of_band")
        distance = buckets.level_distance(f.contributor_weight, f.nearest_defensible)
        if distance >= 2:
            two_level.append(f.criterion_id)
            reasons[f.criterion_id].append("two_level_off")
        if distance >= 1:
            any_level.append(f.criterion_id)
            reasons[f.criterion_id].append("any_level_off")

    two_rate = (len(two_level) / denominator) if denominator else 0.0
    any_rate = (len(any_level) / denominator) if denominator else 0.0

    if (
        two_rate >= policy.weights_two_level_fail_rate
        or any_rate >= policy.weights_any_level_fail_rate
    ):
        band: Band = "fail"
    else:
        inaccurate = out_of_band if policy.weights_non_fail_on_out_of_band else any_level
        band = "non_fail" if inaccurate else "clean"

    contributing = sorted(
        set(out_of_band if policy.weights_non_fail_on_out_of_band else any_level)
    )
    return build_verdict(
        task_id=task.task_id,
        check_id=260,
        band=band,
        measurement=Measurement(
            numerator=len(any_level),
            denominator=denominator,
            rate=any_rate,
            threshold=policy.weights_any_level_fail_rate,
            counts={
                "two_level_off": len(two_level),
                "two_level_rate": two_rate,
                "two_level_threshold": policy.weights_two_level_fail_rate,
                "any_level_off": len(any_level),
                "out_of_band": len(out_of_band),
                # Reported so a reader can see how much of the disagreement the
                # band absorbs, which is the whole point of measuring one.
                "differs_from_auditor_pick": point_gap,
                # One criterion can be counted by the two-level gate, the any-level
                # gate and the out-of-band list at once, and `contributing_items`
                # flattens all three into one name per criterion. Without this a
                # reader cannot tell which four of the seventeen were two levels
                # off, and will guess.
                "reasons_by_item": {k: list(reasons[k]) for k in sorted(reasons)},
            },
            notes=(
                "Counts criteria whose recorded weight falls outside the range the "
                "weight definitions can support; a weight inside the range is not a "
                f"finding. Buckets low={list(buckets.low)}, medium={list(buckets.medium)}, "
                f"high={list(buckets.high)} from the customer weight guide."
            ),
        ),
        contributing_items=contributing,
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 270 -- rubric evaluation
# ---------------------------------------------------------------------------


def evaluate_check_270(
    task: Task,
    findings: list[CriterionRatingFinding],
    policy: Policy = DEFAULT_POLICY,
    abstained: int = 0,
) -> CheckVerdict:
    """Count criteria the auditor rates differently, over the criteria it judged.

    Both of the spec's fail legs are stated in criteria -- "You disagree with the
    contributor's ratings of 5 or more criteria, or 20% or more of the criteria"
    -- while the notes column asks the auditor to "count the disagreements across
    all models". A criterion the auditor disagrees on for both models therefore
    counts once: "across all models" fixes what the auditor looks at, and the
    unit counted is the criterion. Two further reasons settle the ambiguity the
    same way. It is the only reading under which the rate cannot exceed 100%, so
    "20% or more of the criteria" stays a statement about criteria. And it is how
    the other census checks in this build count: 230/240/250 credit a criterion
    once however many issues it carries.

    The judgment-level count is reported beside it, so nothing a per-model tally
    would have shown is lost.
    """
    disagreeing_judgments = sorted(f.item_id for f in findings if f.disagrees)
    disagreements = sorted({f.criterion_id for f in findings if f.disagrees})
    count = len(disagreements)

    denominator = len({f.criterion_id for f in findings})
    rate = (count / denominator) if denominator else 0.0

    if count >= policy.rubric_eval_fail_count or rate >= policy.rubric_eval_fail_rate:
        band: Band = "fail"
    elif count > 0:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=270,
        band=band,
        measurement=Measurement(
            numerator=count,
            denominator=denominator,
            rate=rate,
            threshold=policy.rubric_eval_fail_rate,
            counts={
                "disagreeing_criteria": count,
                "criteria_judged": denominator,
                # A criterion disagreed on for both models raises this and not the
                # numerator, so the gap between the two is the two-model overlap.
                "disagreeing_judgments": len(disagreeing_judgments),
                "judgments": len(findings),
                "disagreeing_items": disagreeing_judgments,
                "absolute_threshold": policy.rubric_eval_fail_count,
                "denominator_basis": "criterion count",
                "abstained": abstained,
                "judged_ratio": (
                    round(len(findings) / (len(findings) + abstained), 3)
                    if (len(findings) + abstained)
                    else 0.0
                ),
            },
            notes="Numerator and denominator are both criteria: a criterion the "
            "auditor rates differently on both models counts once. Disagreements "
            "are looked for across all models. The absolute threshold of 5 "
            "criteria binds before the rate for any rubric over 25 criteria."
            + (
                f" {abstained} judgments abstained for lack of auditable "
                "evidence; a criterion leaves the denominator only when every "
                "model's judgment on it abstained."
                if abstained
                else ""
            ),
        ),
        contributing_items=disagreements,
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 280 / 310 -- relevant turns
# ---------------------------------------------------------------------------


def evaluate_relevant_turns(
    task: Task,
    check_id: Literal[280, 310],
    findings: list[RelevantTurnFinding],
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    scoped = [f for f in findings if f.check_id == check_id]

    incorrect = 0
    missing = 0
    unverifiable = 0
    contributing: list[str] = []
    reasons: dict[str, list[str]] = {}
    turns_by_item: dict[str, list[int]] = {}
    unverifiable_by_item: dict[str, list[int]] = {}
    for f in scoped:
        # Namespaced by model as well as item: both submissions rate the same
        # dimension names ("Communication quality" exists for A and for B), and
        # keying on the bare item alone had a second model's citation overwrite
        # the first's here, silently dropping it from the report even though the
        # count above it was never wrong -- the gate summed every finding's
        # length regardless of this key, so only the per-item breakdown was lossy.
        item = f"{check_id}::{f.model}::{f.item_id}"
        if f.incorrect_turns:
            incorrect += len(f.incorrect_turns)
            contributing.append(f"{check_id}::{f.item_id}")
            reasons.setdefault(item, []).append("incorrect_turns")
            turns_by_item[item] = list(f.incorrect_turns)
        if f.missing_turn:
            missing += 1
            reasons.setdefault(item, []).append("missing_turn")
            if policy.count_missing_turns_as_incorrect:
                contributing.append(f"{check_id}::{f.item_id}::missing_turn")
        unverifiable += len(f.unverifiable_turns)
        if f.unverifiable_turns:
            reasons.setdefault(item, []).append("unverifiable_turns")
            unverifiable_by_item[item] = list(f.unverifiable_turns)

    counted = incorrect + (missing if policy.count_missing_turns_as_incorrect else 0)

    if counted >= policy.relevant_turns_fail_count:
        band: Band = "fail"
    elif counted > 0:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=check_id,
        band=band,
        measurement=Measurement(
            numerator=counted,
            threshold=policy.relevant_turns_fail_count,
            counts={
                "incorrect_turns": incorrect,
                "missing_turns": missing,
                "missing_counted_as_incorrect": policy.count_missing_turns_as_incorrect,
                "unverifiable_turns": unverifiable,
                "items_audited": len(scoped),
                # The three tallies above are task-wide and the item list mixes
                # wrongly cited turns with citations that are absent altogether, so
                # a reader cannot tell which item was faulted for what -- nor which
                # turn numbers were wrong, which the counts drop entirely. An item
                # appears here with only `unverifiable_turns` when nothing about it
                # was counted, because a turn we could not fetch is our failure and
                # must not read as a defect.
                "reasons_by_item": {k: list(reasons[k]) for k in sorted(reasons)},
                "turns_by_item": {k: turns_by_item[k] for k in sorted(turns_by_item)},
                "unverifiable_turns_by_item": {
                    k: unverifiable_by_item[k] for k in sorted(unverifiable_by_item)
                },
            },
            notes="Absolute counts, no percentages. Findings are namespaced by check "
            "ID so a turn is never counted toward both 280 and 310. Turns cited "
            "beyond the end of the conversation the audit could fetch are reported "
            "as unverifiable and never counted.",
        ),
        contributing_items=contributing,
        confidence=_lowest_confidence([f.confidence for f in scoped]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 300 -- dimension ratings
# ---------------------------------------------------------------------------


def _check_300_basis_note(
    counted: int,
    blind_only: list[str],
    truncated: list[str],
    unadjudicated: list[str],
    policy: Policy,
) -> str:
    """Say, in the first sentence of the notes, what this verdict rests on.

    A reader who reads nothing else has to come away knowing whether a rating was
    wrong or whether our rater merely differed, because the two were
    indistinguishable in the recorded numbers and the second one is not actionable.
    """
    dropped = len(blind_only) + len(truncated) + len(unadjudicated)
    if not policy.dimension_disagreement_requires_informed_confirmation:
        return (
            "BASIS: blind rating deltas, unconfirmed -- informed confirmation is "
            "switched off by policy, so every gap between the contributor's rating "
            "and an independent blind rating is counted as the spec reads it. "
        )
    if counted and not dropped:
        return (
            f"BASIS: {counted} rating gap(s), every one of them confirmed by an "
            "informed reader as a rating this conversation does not support. "
        )
    if counted:
        return (
            f"BASIS: {counted} confirmed rating gap(s); a further {dropped} gap(s) "
            "were found and are not counted -- see blind_only_items (a defensible "
            "difference of judgment), truncated_evidence_items (the blind rating "
            "read a conversation that had been cut), and unadjudicated_items. "
        )
    if dropped:
        return (
            f"BASIS: NO confirmed rating gaps. The blind pass disagreed on "
            f"{dropped} dimension(s) and no informed reading supported treating any "
            "of them as a defect, so this check reports nothing countable: "
            f"{len(blind_only)} defensible difference(s) of judgment, "
            f"{len(truncated)} made from a truncated conversation, "
            f"{len(unadjudicated)} where confirmation could not be obtained. "
        )
    return "BASIS: no rating gaps between the contributor and the blind rating. "


def evaluate_check_300(
    task: Task,
    findings: list[DimensionRatingFinding],
    policy: Policy = DEFAULT_POLICY,
    abstained: int = 0,
) -> CheckVerdict:
    """Count the disagreements an informed reader confirmed, not the blind deltas.

    The spec's fail cell is a count of rating gaps, and this check used to publish
    exactly that: a blind judge's number against the contributor's, three points
    apart, twice, is a fail. A manual audit of twelve of those fails found seven
    were not defects. Two kinds of thing were being counted as one. Some gaps were
    two defensible readings of the same conversation landing apart on a five-point
    scale, which is what a subjective rating does. Others were the judge rating a
    conversation it had only been shown half of, because the render budget cut it
    and nothing carried that fact to the gate.

    So a gap is now necessary but not sufficient. `counts_as_disagreement` holds
    the informed pass's answer to whether the contributor's rating was defensible,
    and a gap nobody informed would call indefensible is reported as `blind_only`
    and left out of the bands. A gap whose blind judgment came from a truncated
    render is reported as `truncated_evidence` and, by default, also left out --
    both passes read the same cut conversation, so the second one confirms the
    first's blind spot rather than checking it.

    Every excluded gap stays in the output under its own reason. The point of the
    labels is that a reader can tell a rating the transcript contradicts from an
    artifact of how this pipeline rendered it, which is the distinction the
    recorded fail rate was hiding.
    """
    major: list[str] = []
    total: list[str] = []
    na_mismatches: list[str] = []
    reasons: dict[str, list[str]] = {}
    considered = 0
    # Gaps that exist but are not counted, each under the reason it was dropped.
    blind_only: list[str] = []
    truncated_excluded: list[str] = []
    unadjudicated: list[str] = []

    for f in findings:
        # A dimension both sides agree is not applicable leaves the denominator.
        if f.both_na:
            continue
        considered += 1

        if f.na_mismatch:
            na_mismatches.append(f.item_id)
            reasons.setdefault(f.item_id, []).append("na_mismatch")
            if policy.na_mismatch_counts_as == "major":
                major.append(f.item_id)
                total.append(f.item_id)
            elif policy.na_mismatch_counts_as == "minor":
                total.append(f.item_id)
            continue

        delta = f.delta
        if delta is None or delta == 0:
            continue

        reasons.setdefault(f.item_id, []).append("rating_delta")
        if delta >= policy.dimension_major_delta:
            reasons[f.item_id].append("major_delta")
        if f.truncated:
            reasons[f.item_id].append("truncated_evidence")
        if f.adjudication is not None:
            reasons[f.item_id].append(f"adjudicated_{f.adjudication}")
        elif f.adjudication_abstained:
            reasons[f.item_id].append("adjudication_abstained")

        if f.truncated and policy.exclude_truncated_evidence_disagreements:
            truncated_excluded.append(f.item_id)
            continue
        if not f.counts_as_disagreement:
            if f.adjudication == "defensible":
                blind_only.append(f.item_id)
            else:
                unadjudicated.append(f.item_id)
            continue

        total.append(f.item_id)
        if delta >= policy.dimension_major_delta:
            major.append(f.item_id)

    if (
        len(major) >= policy.dimension_fail_major_count
        or len(total) >= policy.dimension_fail_total_count
    ):
        band: Band = "fail"
    elif total:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=300,
        band=band,
        measurement=Measurement(
            numerator=len(total),
            denominator=considered,
            threshold=policy.dimension_fail_total_count,
            counts={
                "major_disagreements": len(major),
                "total_disagreements": len(total),
                "major_threshold": policy.dimension_fail_major_count,
                "major_delta": policy.dimension_major_delta,
                "na_mismatches": len(na_mismatches),
                "na_mismatch_counts_as": policy.na_mismatch_counts_as,
                "rating_scale": list(policy.dimension_rating_scale),
                "abstained": abstained,
                # Every rating gap the blind pass found, and what happened to it.
                # A reader asking "did this task fail because a rating is wrong, or
                # because our rater picked a different number" answers it from
                # these four numbers without opening the findings.
                "rating_gaps_found": (
                    len(total)
                    + len(blind_only)
                    + len(truncated_excluded)
                    + len(unadjudicated)
                ),
                "counted_disagreements": len(total),
                "blind_only_not_counted": len(blind_only),
                "blind_only_items": sorted(blind_only),
                "truncated_evidence_not_counted": len(truncated_excluded),
                "truncated_evidence_items": sorted(truncated_excluded),
                "unadjudicated_not_counted": len(unadjudicated),
                "unadjudicated_items": sorted(unadjudicated),
                "informed_confirmation_required": (
                    policy.dimension_disagreement_requires_informed_confirmation
                ),
                # True when nothing survived confirmation: the blind pass
                # disagreed and no informed reading backed it. The plainest form
                # of the "failed purely because of the blind rater" question.
                "blind_disagreement_only": bool(
                    not total and (blind_only or truncated_excluded or unadjudicated)
                ),
                # An N/A mismatch and a two-point rating gap are different accusations
                # and the item list merges them, so the task-level `na_mismatches` and
                # `major_disagreements` counts can only be attached to a dimension by
                # guesswork. A dimension can also appear here on `na_mismatch` alone
                # while the policy declines to count it, which is why the reason is
                # named rather than implied by membership.
                "reasons_by_item": {k: list(reasons[k]) for k in sorted(reasons)},
            },
            notes=_check_300_basis_note(
                len(total), blind_only, truncated_excluded, unadjudicated, policy
            )
            + "Dimensions both sides marked N/A are excluded from the "
            "rating-comparison denominator, but the check still reports a band, and "
            "with nothing else in disagreement that band is clean. The spec's "
            "general grading instruction 4 asks for the 'no issues' option where a "
            "field does not apply to the task type, so an inapplicable dimension is "
            "a recorded clean and stays in the published denominator; it is not "
            "not_evaluated, which is reserved for dimensions we could not judge."
            + (
                f" {abstained} dimensions abstained for lack of auditable evidence and "
                "are also excluded."
                if abstained
                else ""
            ),
        ),
        contributing_items=sorted(set(total)),
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 400 -- ranking disagreement
# ---------------------------------------------------------------------------


def evaluate_check_400(
    task: Task, finding: LikertFinding, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """A ranking gap, confirmed against the transcripts and against the render.

    Two conditions now stand between a Likert delta and a fail, both from the same
    manual audit that produced check 300's: five of twelve sampled fails were the
    render rather than the ranking.

    The first is the render itself. The two conversations share one budget, so a
    long side is cut sooner here than it would be alone, and one sampled task
    ranked a Model B it had seen one exchange of against a Model A it had seen
    eight. A preference between conversations shown to incomparable depths is a
    preference about the budget, and `asymmetric_render` takes it out of the count
    rather than publishing it.

    The second is the same informed confirmation check 300 now applies: the blind
    Likert says the two evaluators placed the pair differently, which on a
    seven-point scale two careful readers regularly do. Only a preference an
    informed reader calls indefensible against the transcripts is counted.
    """
    delta = finding.delta
    asymmetric = finding.asymmetric_render
    confirmation_on = policy.ranking_disagreement_requires_informed_confirmation

    excluded_reason = ""
    if delta > 0:
        if asymmetric:
            excluded_reason = "asymmetric_render"
        elif finding.truncated and policy.exclude_truncated_evidence_disagreements:
            excluded_reason = "truncated_evidence"
        elif confirmation_on and not finding.counts_as_disagreement:
            excluded_reason = (
                "blind_only"
                if finding.adjudication == "defensible"
                else "unadjudicated"
            )

    counted = delta if not excluded_reason else 0

    if counted >= policy.likert_fail_delta:
        band: Band = "fail"
    elif counted > 0:
        band = "non_fail"
    else:
        band = "clean"

    if not excluded_reason:
        note = (
            "Raw point deltas per tab 2, which overrides tab 1's bucket agreement "
            "and is stricter than it."
            if delta == 0
            else (
                f"BASIS: a {delta}-point ranking gap, confirmed against both "
                "transcripts by an informed reader as a preference they do not "
                "support. Raw point deltas per tab 2, which overrides tab 1's "
                "bucket agreement and is stricter than it."
            )
        )
    elif excluded_reason == "asymmetric_render":
        note = (
            f"BASIS: a {delta}-point ranking gap, NOT counted. The two "
            f"conversations were not rendered to comparable depth "
            f"(Model A: {finding.turns_shown_a} of {finding.turns_total_a} turns; "
            f"Model B: {finding.turns_shown_b} of {finding.turns_total_b}), so the "
            "blind comparison ranked how much of each side it was shown as much as "
            "the conversations themselves."
        )
    elif excluded_reason == "truncated_evidence":
        note = (
            f"BASIS: a {delta}-point ranking gap, NOT counted: the blind comparison "
            "read at least one conversation that had been cut to fit the render "
            "budget, so the gap may sit in the part it was never shown."
        )
    elif excluded_reason == "blind_only":
        note = (
            f"BASIS: a {delta}-point ranking gap, NOT counted. An informed reader "
            "shown the contributor's preference, their reasoning, and both "
            "transcripts found the preference defensible: the two evaluators "
            "weighed the same evidence differently, which is not a defect."
        )
    else:
        note = (
            f"BASIS: a {delta}-point ranking gap, NOT counted: informed "
            "confirmation could not be obtained"
            + (f" ({finding.why_unadjudicated})." if finding.why_unadjudicated else ".")
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=400,
        band=band,
        measurement=Measurement(
            numerator=counted,
            threshold=policy.likert_fail_delta,
            counts={
                "contributor_likert": finding.contributor_likert,
                "auditor_likert": finding.auditor_likert,
                "delta": delta,
                "counted_delta": counted,
                "not_counted_because": excluded_reason,
                "blind_disagreement_only": bool(delta > 0 and excluded_reason),
                "informed_confirmation_required": confirmation_on,
                "adjudication": finding.adjudication or "",
                "adjudication_reasoning": finding.adjudication_reasoning,
                # The render's own fairness, which for a comparison is part of the
                # measurement rather than a footnote about it.
                "render_asymmetric": asymmetric,
                "turns_shown_a": finding.turns_shown_a,
                "turns_shown_b": finding.turns_shown_b,
                "turns_total_a": finding.turns_total_a,
                "turns_total_b": finding.turns_total_b,
                "truncated_a": finding.truncated_a,
                "truncated_b": finding.truncated_b,
            },
            notes=note,
        ),
        confidence=("low" if excluded_reason else finding.confidence),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 450 -- justifications
# ---------------------------------------------------------------------------


def evaluate_check_450(
    task: Task, findings: list[JustificationFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Fail on the share of justifications that trip a condition, not on the first.

    The spec's conditions are per-argument -- "at least 1 supporting claim does
    not logically defend the verdict", two claims lacking evidence, two
    misconstrued quotes -- and `justification_scope` keeps them that way. It says
    nothing about how many of the roughly fifteen justifications a task carries
    have to trip before the task fails. Reading that silence as "one" made the
    fail band unreachable in the contributor's favour: on the first live run the
    per-justification trip rate ran from 7% to 80% and the check fired on 17 of
    17 tasks, so the band stopped distinguishing a contributor with one weak
    argument from one whose arguments fail throughout.

    `policy.justification_fail_rate` is therefore a proportion, and it is ours
    rather than the customer's -- see the field's own comment for why 30% and not
    one of the spec's other percentages.

    The non-fail band is deliberately left at any justification carrying any
    issue at all. That is the spec's own convention for a shape-C middle band
    (230's "< 10% Major Rubric Errors" is exactly "some, but not enough to
    fail"), and sub-threshold defects are what it exists to hold. It does mean
    this check reports non-clean on essentially every task, which is a true
    statement about the data and not something a fail threshold can fix.

    Pooled scope keeps the any-trigger fail. Pooling collapses the task into one
    synthetic justification, so there is no population left to take a share of;
    a rate of one-over-fifteen would silence the check entirely.
    """
    triggered: dict[str, list[str]] = {}
    with_issues: list[str] = []

    if policy.justification_scope == "per_justification":
        for f in findings:
            conditions = f.triggered_conditions()
            if conditions:
                triggered[f.item_id] = conditions
            if f.has_any_issue():
                with_issues.append(f.item_id)
    else:
        pooled = JustificationFinding(item_id="pooled")
        for f in findings:
            pooled.contradicts_verdict_claims += f.contradicts_verdict_claims
            pooled.contradicting_quotes += f.contradicting_quotes
            pooled.is_generic = pooled.is_generic or f.generic
            pooled.is_skewed = pooled.is_skewed or f.is_skewed
            pooled.inaccurate_primary_claims += f.inaccurate_primary_claims
            pooled.inaccurate_secondary_claims += f.inaccurate_secondary_claims
            pooled.unsupported_claims += f.unsupported_claims
            pooled.inaccurate_evidence += f.inaccurate_evidence
            pooled.misconstrued_evidence += f.misconstrued_evidence
            if f.has_any_issue():
                with_issues.append(f.item_id)
        conditions = pooled.triggered_conditions()
        if conditions:
            triggered["pooled"] = conditions

    proportional = policy.justification_scope == "per_justification"
    denominator = len(findings)
    rate = (len(triggered) / denominator) if denominator else 0.0
    threshold = policy.justification_fail_rate if proportional else None

    if triggered and (not proportional or rate >= policy.justification_fail_rate):
        band: Band = "fail"
    elif triggered or with_issues:
        band = "non_fail"
    else:
        band = "clean"

    all_conditions = sorted({c for cs in triggered.values() for c in cs})
    return build_verdict(
        task_id=task.task_id,
        check_id=450,
        band=band,
        measurement=Measurement(
            numerator=len(triggered),
            denominator=denominator,
            rate=rate,
            threshold=threshold,
            counts={
                "justifications_audited": len(findings),
                "justifications_with_issues": len(with_issues),
                "failing_justifications": len(triggered),
                "failing_rate": rate,
                "fail_threshold_is_ours": proportional,
                "conditions_triggered": all_conditions,
                # The flat list above is a union over the task, so on its own it
                # accuses all fifteen justifications of every condition any one
                # of them tripped. The map is what a reader should be shown.
                "conditions_by_item": {k: list(v) for k, v in sorted(triggered.items())},
                "unverifiable_claims": sum(f.unverifiable_claims for f in findings),
                "scope": policy.justification_scope,
            },
            notes="Population is every dimension justification for every model plus "
            "the ranking justification. Each condition's own floor is the spec's and "
            "is applied within a single justification; the share of justifications "
            "that must trip before the task fails is not stated by the spec and the "
            "threshold reported here is this build's own."
            if proportional
            else "Population is every dimension justification for every model plus "
            "the ranking justification. Counts are pooled across the task, so any "
            "triggered condition fails and no proportion is computed.",
        ),
        contributing_items=sorted(triggered) or sorted(set(with_issues)),
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 80 -- target outcome, 95 -- artifacts, 110 -- key turn justification,
# 220 -- rubric autofail, 470 -- verdict
# ---------------------------------------------------------------------------


def evaluate_check_70(
    task: Task,
    finding: DomainRelevanceFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape C, and the whole point of the middle band.

    "Egregious domain misalignment" is the fail wording, so only `unrelated`
    fails; `partial` -- a prompt that leans away from its domain without leaving
    it -- lands in non-fail. An unevidenced allegation is recorded in the
    measurement and scored clean, because a misalignment nobody can quote is not
    one.

    Reports not_evaluated when no domain was assigned, which is the honest answer
    rather than grading the prompt against nothing.
    """
    if not (task.assigned_domain or "").strip():
        return build_verdict(
            task_id=task.task_id,
            check_id=70,
            band="not_evaluated",
            measurement=Measurement(
                notes="no assigned domain was recorded for this task, so there is "
                "nothing to measure the prompt against"
            ),
            confidence="low",
            policy=policy,
        )

    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=70,
            band="not_evaluated",
            measurement=Measurement(notes="the prompt was not audited for domain relevance"),
            confidence="low",
            policy=policy,
        )

    if finding.is_egregious:
        band: Band = "fail"
    elif finding.is_issue:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=70,
        band=band,
        measurement=Measurement(
            numerator=1 if band != "clean" else 0,
            denominator=1,
            counts={
                "assessment": finding.assessment,
                "assigned_domain": task.assigned_domain,
                "assigned_category": task.prompt_category,
                "evidenced": finding.is_evidenced,
            },
            notes="Only an egregious mismatch fails; a prompt that drifts in "
            "emphasis without leaving its domain is a non-fail. A finding that "
            "quotes neither the prompt nor the assignment is discarded."
            + (
                " Discarded here: the assessment was "
                f"{finding.assessment!r} with no quoted evidence."
                if finding.assessment != "aligned" and not finding.is_evidenced
                else ""
            ),
        ),
        contributing_items=(
            [finding.prompt_quote, finding.domain_basis] if finding.is_issue else []
        ),
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_75(
    task: Task,
    finding: PromptConsistencyFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape C, the same shape as 70 and for the same reason: the middle band
    holds a reworded opening that keeps the same underlying request, which is
    the expected outcome whenever a prompt was naturalised rather than copied.

    Only "different" -- a genuinely different subject or request -- fails.
    An unevidenced allegation is recorded in the measurement and scored clean,
    because a mismatch nobody can quote from both sides is not one.

    Reports not_evaluated when no pre-seeded prompt was recorded, which is the
    honest answer rather than grading the submission against nothing: the field
    is absent on most tasks project-wide, and this check must abstain rather
    than guess at what the platform would have prepared.
    """
    if not (task.pre_seeded_prompt or "").strip():
        return build_verdict(
            task_id=task.task_id,
            check_id=75,
            band="not_evaluated",
            measurement=Measurement(
                notes="no pre-seeded prompt was recorded for this task, so there "
                "is nothing to compare the submitted prompt against"
            ),
            confidence="low",
            policy=policy,
        )

    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=75,
            band="not_evaluated",
            measurement=Measurement(
                notes="the submitted prompt was not audited for pre-seed consistency"
            ),
            confidence="low",
            policy=policy,
        )

    if finding.is_fail:
        band: Band = "fail"
    elif finding.is_issue:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=75,
        band=band,
        measurement=Measurement(
            numerator=1 if band != "clean" else 0,
            denominator=1,
            counts={
                "assessment": finding.assessment,
                "evidenced": finding.is_evidenced,
            },
            notes="Judged on underlying intent, not wording: paraphrasing and "
            "added context are expected and land in the non-fail band, not the "
            "fail band. Only a genuinely different subject or request fails. A "
            "finding that quotes neither the pre-seeded prompt nor the "
            "submitted prompt is discarded."
            + (
                " Discarded here: the assessment was "
                f"{finding.assessment!r} with no quoted evidence."
                if finding.assessment != "same" and not finding.is_evidenced
                else ""
            ),
        ),
        contributing_items=(
            [finding.pre_seeded_quote, finding.submitted_quote]
            if finding.is_issue
            else []
        ),
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_80(
    task: Task, findings: list[TargetOutcomeFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Shape B: the worst available band is non-fail, so this can never fail a task.

    The population is the contributor's own list and nothing else. Tab 2 defines
    this check as three predicates on entries that were written down -- the list
    "is misaligned with the prompt(s); it's incorrect, lists outcomes which are not
    necessary or expected by the prompt(s), or contradicts one or more prompts" --
    and step 6 of the audit workflow tab says to "Determine if the target outcomes
    listed are correct and align with the prompt(s)". Neither asks whether the
    list is complete, so the judge is never asked to derive what it omits, and
    both the numerator and the denominator are just the entries it classified.

    Which classifications are a defect is read from
    `policy.target_outcome_defect_classifications` rather than fixed here, so a
    later ruling on scope is a one-field change and never a gate rewrite. The
    classifications the judge assigned but the active scope does not count are
    reported under `recorded_not_counted`, so narrowing the scope hides nothing.
    """
    scope = tuple(policy.target_outcome_defect_classifications)
    entries = list(findings)
    counted = [f for f in findings if f.counts_as_defect(scope)]

    by_class: dict[str, int] = {}
    for f in findings:
        by_class[f.classification] = by_class.get(f.classification, 0) + 1
    recorded_not_counted = {
        name: n
        for name, n in by_class.items()
        if name != "supported" and name not in scope
    }

    return build_verdict(
        task_id=task.task_id,
        check_id=80,
        band="non_fail" if counted else "clean",
        measurement=Measurement(
            numerator=len(counted),
            denominator=len(entries),
            counts={
                "entries_audited": len(entries),
                **by_class,
                "defect_classifications": list(scope),
                # Judged, recorded, and deliberately outside the active scope.
                "recorded_not_counted": recorded_not_counted,
                # The per-classification tallies are task-wide, so beside the flat
                # entry list they cannot say which entry merely was not required and
                # which one contradicts the prompt outright.
                "classifications_by_item": {f.entry: f.classification for f in counted},
            },
            notes="Final-state semantics: requirements are replayed across all "
            "turns before comparison, never diffed against turn 1 alone. Scored "
            "on the contributor's listed entries only; the spec asks whether the "
            "outcomes listed are correct, never whether the list is exhaustive, "
            "so the judge is never asked what it omits.",
        ),
        contributing_items=[f.entry for f in counted],
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


def evaluate_check_85(
    task: Task,
    findings: list[EnvironmentContextFinding],
    scan: FileReferenceScan,
    entity_half_ran: bool = False,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape A, and composed of two halves that can succeed independently.

    A file-kind finding counts only when the file half actually ran, which needs a
    manifest that names something. Reporting a named file as absent from a manifest
    of opaque CDN paths would fail every task that attached one, and reporting it
    absent from a manifest we never ingested would fail tasks whose input files are
    sitting in the export unread. Either way the reference is unverified, not
    violated, and unverified never fails.

    The task is clean when at least one half ran and neither found a violation, and
    not_evaluated only when neither half could run at all -- which is the honest
    answer, not a pass.
    """
    unverified = [f for f in findings if f.kind == "file"] if not scan.ran else []
    considered = [f for f in findings if f not in unverified]
    violations: list[EnvironmentContextFinding] = []
    seen: set[str] = set()
    for f in considered:
        if not f.is_violation or f.item_id.lower() in seen:
            continue
        seen.add(f.item_id.lower())
        violations.append(f)

    discarded = [f for f in considered if not f.evidenced]
    evaluated = scan.ran or entity_half_ran

    if violations:
        band: Band = "fail"
    elif evaluated:
        band = "clean"
    else:
        band = "not_evaluated"

    notes = (
        "The bar is the audit workflow tab's: the prompt must reference only the "
        "entities and events in the task's universe. A reference the prompt merely "
        "proposes or asks about, one the work does not need, and any public "
        "real-world entity are all excluded. Every finding must quote the prompt and "
        "name what it was checked against; one that does neither is discarded."
    )
    if not scan.ran:
        notes += f" File half not run: {scan.blocked_reason}."
    if not entity_half_ran:
        notes += " Entity half not run: no model judgment was recorded."
    if discarded:
        notes += f" {len(discarded)} unevidenced allegation(s) discarded."

    return build_verdict(
        task_id=task.task_id,
        check_id=85,
        band=band,
        measurement=Measurement(
            numerator=len(violations),
            denominator=len(considered),
            counts={
                "violations": len(violations),
                "file_violations": sum(1 for f in violations if f.kind == "file"),
                "entity_violations": sum(1 for f in violations if f.kind != "file"),
                "file_half_ran": scan.ran,
                "entity_half_ran": entity_half_ran,
                "references_unverifiable": len(unverified),
                "unevidenced_discarded": len(discarded),
                **scan.to_dict(),
            },
            notes=notes,
        ),
        contributing_items=[f"{f.item_id}: {f.quote}" for f in violations],
        confidence=(
            "low" if not evaluated else _lowest_confidence([f.confidence for f in violations])
        ),
        policy=policy,
    )


def evaluate_check_95(
    task: Task, findings: list[ArtifactFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Shape A: fail if any claimed output file was never uploaded.

    A response that produced no files at all is an explicit clean, so an empty
    extraction must never fail.

    What is counted is the names a confirming pass agreed were absent, not the
    names a string comparison could not match. The two are different populations:
    on a manual audit of nine fails here, five were the comparison's own limits --
    a screenshot whose capture-tool name no distance metric ties to the
    contributor's name for the same image, an inline code block read as an
    undelivered file. This check has no middle band, so each of those was a
    task-level fail with nothing to soften it. Every name the confirmation clears
    is still published, with the reason, under `cleared_by_confirmation`.
    """
    named = [f"{f.model}:{name}" for f in findings for name in f.confirmed_missing]
    shortfall = sum(f.shortfall for f in findings if f.basis == "count")
    counted = [
        f"{f.model}: {f.shortfall} of {len(f.claimed_files)} files not accounted for"
        for f in findings
        if f.basis == "count" and f.shortfall
    ]
    claimed = sum(len(f.claimed_files) for f in findings)
    missing = len(named) + shortfall
    fuzzy_matched = [
        f"{f.model}:{claimed_name}~{uploaded_name}"
        for f in findings
        for claimed_name, uploaded_name in f.fuzzy_matched_files
    ]

    notes = (
        "Embedded link liveness is never tested; expired links are "
        "expected. A response mentioning no output files is clean. Where the "
        "upload manifest carries no filenames, only the shortfall in count is "
        "reported and no individual file is named."
    )
    if fuzzy_matched:
        notes += (
            f" {len(fuzzy_matched)} claimed filename(s) matched the upload "
            f"manifest only above the {policy.filename_fuzzy_match_threshold:.0%} "
            "similarity threshold, not exactly -- a duplicate-upload or version "
            "suffix rather than a missing file -- and count as present; see "
            "fuzzy_matched below."
        )

    cleared = [
        f"{f.model}:{name} ({assessment})"
        for f in findings
        for name, assessment in f.cleared_by_confirmation
    ]
    confirmation_reasons = {
        f"{f.model}:{name}": reason
        for f in findings
        for name, reason in f.confirmation_reasons.items()
    }
    if cleared:
        notes += (
            f" {len(cleared)} name(s) the filename comparison read as missing were "
            "put to a confirming model call and cleared: either the same file under "
            "a name string distance cannot bridge, or content the model printed "
            "inline rather than a file it claimed to attach. See "
            "cleared_by_confirmation, with each reason."
        )
    unconfirmed = [f.model for f in findings if f.missing_files and not f.confirmation_ran]
    if unconfirmed and policy.artifact_miss_requires_llm_confirmation:
        notes += (
            " Confirmation did not run for "
            f"{', '.join(sorted(set(unconfirmed)))}; those names stay counted, "
            "because a call that failed is not evidence the file was there."
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=95,
        band="fail" if missing else "clean",
        measurement=Measurement(
            numerator=missing,
            denominator=claimed,
            counts={
                "files_claimed": claimed,
                "files_uploaded": sum(len(f.uploaded_files) for f in findings),
                "files_missing": missing,
                "matched_by": sorted({f.basis for f in findings}),
                # `matched_by` pools both models, so a report where one model was
                # matched by filename and the other only by count reads as though
                # every named file had been checked by name.
                "basis_by_model": {f.model: f.basis for f in findings},
                "fuzzy_matched": fuzzy_matched,
                # What the filename comparison alone would have counted, beside
                # what survived confirmation. A reader asking whether this fail is
                # real answers it from the gap between these two numbers.
                "files_missing_before_confirmation": sum(
                    len(f.missing_files) for f in findings
                )
                + shortfall,
                "cleared_by_confirmation": cleared,
                "confirmation_reasons": confirmation_reasons,
                "confirmation_ran_for": sorted(
                    f.model for f in findings if f.confirmation_ran
                ),
                "confirmation_required": policy.artifact_miss_requires_llm_confirmation,
                "fuzzy_threshold": policy.filename_fuzzy_match_threshold,
            },
            notes=notes,
        ),
        contributing_items=named + counted,
        confidence=_lowest_confidence([f.confidence for f in findings]),
        policy=policy,
    )


def evaluate_check_110(
    task: Task,
    finding: KeyTurnJustificationFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape B, and an unusually low bar: 'or contains any issues'.

    Scored independently of 100 -- a correctly chosen turn can carry a bad
    justification, and both combinations occur.
    """
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=110,
            band="not_evaluated",
            measurement=Measurement(notes="no key turn justification was audited"),
            confidence="low",
            policy=policy,
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=110,
        band="non_fail" if finding.is_issue else "clean",
        measurement=Measurement(
            counts={
                "describes_selected_turn": finding.describes_selected_turn,
                "claims_are_accurate": finding.claims_are_accurate,
                "connects_to_core_value": finding.connects_to_core_value,
                "issues": len(finding.issues),
            },
            notes="Any defect qualifies; this is broader than 450's enumerated "
            "conditions. Scored independently of check 100.",
        ),
        contributing_items=list(finding.issues),
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_220(
    task: Task, findings: list[AutofailFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Shape A, and the highest-stakes check in the spec: one finding fails the
    task outright with no middle band.

    Only a finding both passes agree on can fail a task. An allegation the second
    opinion rejects is still reported, because a reviewer should see a near miss
    on a check this severe, but it cannot decide the verdict alone.
    """
    confirmed = [f for f in findings if f.confirmed]
    unconfirmed = [f for f in findings if not f.confirmed]

    return build_verdict(
        task_id=task.task_id,
        check_id=220,
        band="fail" if confirmed else "clean",
        measurement=Measurement(
            numerator=len(confirmed),
            denominator=len(task.rubric),
            counts={
                "confirmed": len(confirmed),
                "alleged_but_not_confirmed": len(unconfirmed),
                # `contributing_items` names only the confirmed criteria while the
                # counts admit to allegations that failed review, so an unconfirmed
                # allegation has no name and a confirmed one can inherit its blame.
                "status_by_item": {
                    **{f.criterion_id: "alleged_not_confirmed" for f in unconfirmed},
                    **{f.criterion_id: "confirmed" for f in confirmed},
                },
            },
            notes="The bar is outcome-destroying, not merely wrong. Every finding "
            "passes a mandatory second opinion before it can fail the task.",
        ),
        contributing_items=[f.criterion_id for f in confirmed],
        confidence=_lowest_confidence([f.confidence for f in confirmed]),
        policy=policy,
    )


def evaluate_check_470(
    task: Task, finding: VerdictFinding | None, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Shape A, and the narrowest check in the spec.

    Only whether a preference is stated -- not whether it is correct (400),
    consistent (460), or well argued (450). A reasoned tie is a stated preference.
    """
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=470,
            band="not_evaluated",
            measurement=Measurement(notes="no comparison justification was audited"),
            confidence="low",
            policy=policy,
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=470,
        band="clean" if finding.states_preference else "fail",
        measurement=Measurement(
            counts={"states_preference": finding.states_preference},
            notes="Scope is the model comparison justification only; a reasoned "
            "tie counts as a stated preference.",
        ),
        contributing_items=[finding.quote] if finding.quote else [],
        confidence=finding.confidence,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 460 -- inconsistent ranking
# ---------------------------------------------------------------------------


def signed_preference(
    likert: int | None, winner_index: int | None = None, policy: Policy = DEFAULT_POLICY
) -> int | None:
    """Normalise either preference encoding onto one signed axis.

    Negative favors model A, positive favors model B, zero is neutral. This is
    what lets check 400's "differ by 3 or more points" mean the same thing under
    both forms: under direction_magnitude a flipped winner separates the two
    sides of the axis, so an A-by-2 against a B-by-2 reads as 4 points apart
    rather than as agreement on the number 2.
    """
    if likert is None:
        return None
    if policy.preference_encoding == "bipolar_7":
        lo, hi = policy.likert_scale
        return likert - (lo + hi) // 2
    if winner_index is None:
        return None
    return -likert if winner_index == 0 else likert


def likert_direction(
    likert: int | None,
    policy: Policy = DEFAULT_POLICY,
    winner_index: int | None = None,
) -> Literal["A", "B", "neutral"]:
    """Which model the contributor preferred.

    The recorded winner wins over the spec's buckets whenever it is present,
    because the buckets are demonstrably wrong about this product. Across 73 real
    tasks the tool records Likert 1-3 as "A wins" and 5-7 as "B wins", never
    emitting 4 at all; the spec calls 3, 4 and 5 neutral. Trusting the buckets
    silences check 460 on the 27 of 73 tasks sitting at 3 or 5 -- a third of the
    graded set -- because a "neutral" preference can never contradict anything.
    """
    if winner_index is not None:
        return "A" if winner_index == 0 else "B"
    if likert is None:
        return "neutral"
    if policy.preference_encoding == "bipolar_7":
        if likert in policy.likert_favors_a:
            return "A"
        if likert in policy.likert_favors_b:
            return "B"
        return "neutral"
    return "neutral"


def dimension_direction(
    findings_or_task, policy: Policy = DEFAULT_POLICY
) -> Literal["A", "B", "neutral"]:
    """Direction implied by the contributor's own per-dimension ratings.

    Under `dominance` a side counts as favoured only when it wins at least one
    dimension and loses none. A mixed profile -- each model ahead somewhere --
    genuinely supports either verdict, so the Likert cannot contradict it and
    the check stays silent. Averaging instead turns a split decision into a
    direction, and a fraction of a point of separation is then enough to accuse
    a contributor of contradicting themselves.
    """
    ratings = getattr(findings_or_task, "dimension_ratings", findings_or_task)
    usable = [
        r for r in ratings if not r.not_applicable and r.rating is not None
    ]
    a = [r.rating for r in usable if r.model == "A"]
    b = [r.rating for r in usable if r.model == "B"]
    if not a or not b:
        return "neutral"

    if policy.ranking_direction_basis == "dominance":
        by_dimension: dict[str, dict[str, float]] = {}
        for r in usable:
            by_dimension.setdefault(r.dimension, {})[r.model] = r.rating
        a_wins = b_wins = 0
        for scores in by_dimension.values():
            if "A" not in scores or "B" not in scores:
                continue
            if scores["A"] > scores["B"]:
                a_wins += 1
            elif scores["B"] > scores["A"]:
                b_wins += 1
        if a_wins and not b_wins:
            return "A"
        if b_wins and not a_wins:
            return "B"
        return "neutral"

    ma, mb = mean(a), mean(b)
    if ma > mb:
        return "A"
    if mb > ma:
        return "B"
    return "neutral"


def evaluate_check_460(
    task: Task, finding: InversionFinding, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    # Fail only when the directions are opposite AND the justification does not
    # adequately support the inversion. A contributor may rationally prefer the
    # model with lower dimension scores if one severe failure outweighs several
    # small wins, and saying so makes the ranking consistent.
    contradicts = finding.contradicts
    band: Band = (
        "fail" if contradicts and not finding.justification_explains_inversion else "clean"
    )
    return build_verdict(
        task_id=task.task_id,
        check_id=460,
        band=band,
        measurement=Measurement(
            counts={
                "dimension_direction": finding.dimension_direction,
                "likert_direction": finding.likert_direction,
                "contradicts": contradicts,
                "justification_explains_inversion": finding.justification_explains_inversion,
                "direction_basis": policy.ranking_direction_basis,
            },
            notes="Only opposite directions qualify; a neutral Likert with a mild "
            "dimension lean is not a contradiction.",
        ),
        confidence=finding.confidence,
        policy=policy,
    )


def build_inversion_finding(
    task: Task,
    justification_explains_inversion: bool = False,
    policy: Policy = DEFAULT_POLICY,
) -> InversionFinding:
    return InversionFinding(
        dimension_direction=dimension_direction(task, policy),
        likert_direction=likert_direction(
            task.sxs.likert, policy, winner_index=task.sxs.winner_index
        ),
        justification_explains_inversion=justification_explains_inversion,
    )


# ---------------------------------------------------------------------------
# 100 -- key turn identification
# ---------------------------------------------------------------------------


def evaluate_check_100(
    task: Task, auditor_key_turn: int | None = None, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Two-step, and the order matters.

    Turn-1 selection is a structural fail that needs no judgment, so it is decided
    first and short-circuits. Only if the selection is not turn 1 is the auditor's
    independent identification compared.
    """
    cb_turn = task.key_turn.turn_index

    if cb_turn == 1:
        return build_verdict(
            task_id=task.task_id,
            check_id=100,
            band="fail",
            measurement=Measurement(
                counts={"contributor_key_turn": cb_turn},
                notes="The key turn is never turn 1; structural fail, no judgment applied.",
            ),
            policy=policy,
        )

    if auditor_key_turn is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=100,
            band="not_evaluated",
            measurement=Measurement(
                counts={"contributor_key_turn": cb_turn},
                notes="Not turn 1; awaiting the auditor's independent identification.",
            ),
            policy=policy,
        )

    band: Band = "clean" if auditor_key_turn == cb_turn else "non_fail"
    return build_verdict(
        task_id=task.task_id,
        check_id=100,
        band=band,
        measurement=Measurement(
            counts={
                "contributor_key_turn": cb_turn,
                "auditor_key_turn": auditor_key_turn,
            }
        ),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 96 -- minimum turns
# ---------------------------------------------------------------------------


def cited_turns(task: Task, model: str) -> set[int]:
    """Every turn the contributor pointed at for one model.

    The key turn is recorded once for the task rather than per model, so it
    counts for both.
    """
    cited: set[int] = set()
    if task.key_turn.turn_index is not None:
        cited.add(task.key_turn.turn_index)
    for rating in task.criterion_ratings:
        if rating.model == model:
            cited.update(rating.relevant_turns)
    for rating in task.dimension_ratings:
        if rating.model == model:
            cited.update(rating.relevant_turns)
    return {turn for turn in cited if isinstance(turn, int) and turn > 0}


def evaluate_check_96(task: Task, policy: Policy = DEFAULT_POLICY) -> CheckVerdict:
    """Shape A, fully deterministic, and the one check that must not trust its
    own inputs.

    The requirement is the audit workflow tab's: "Both the models require at
    least 7 turns." Gating on hydrated turn counts alone would fail our own fetch
    failures -- a share page that renders a login wall parses as one exchange
    whatever the conversation behind it holds -- so a submission counts only when
    the count can be believed.

    Two facts make that decidable without a model call. First, an incomplete
    fetch can only *under*count, so a submission already at the minimum settles
    the question and no amount of missing tail can change it. Second, below the
    minimum, the contributor's own citations say whether the conversation
    continues past where the fetch stopped -- but only up to the longest
    conversation retrieved for the task, because both models replay one shared
    prompt script and so should run to the same length. A citation above that
    ceiling is corroborated by nothing: three tasks in the live batch cite a
    "Turn 35" against conversations of 21, 19 and 6 turns, and reading it as
    evidence would abstain on a genuine six-turn violation.
    """
    minimum = policy.min_turns_per_model
    strict = policy.turn_count_requires_complete_conversation
    # The best fetch in the task is the only corroboration available for a
    # citation past the end of the worst one.
    ceiling = max((sub.exchange_count() for sub in task.submissions()), default=0)

    counts: dict[str, int] = {}
    skipped: dict[str, str] = {}
    verifiable: dict[str, int] = {}

    for sub in task.submissions():
        count = sub.exchange_count()
        counts[sub.model] = count
        if not sub.conversation:
            skipped[sub.model] = "conversation was never fetched"
            continue
        if strict and count < minimum:
            unfetched = sorted(
                turn for turn in cited_turns(task, sub.model) if count < turn <= ceiling
            )
            if unfetched:
                skipped[sub.model] = (
                    f"contributor cites turns {unfetched} beyond the {count} fetched, "
                    f"so the conversation is longer than the fetch retrieved"
                )
                continue
        verifiable[sub.model] = count

    short = sorted(m for m, count in verifiable.items() if count < minimum)
    if short:
        band: Band = "fail"
    elif verifiable and not skipped:
        band = "clean"
    else:
        # Either nothing was verifiable, or a submission we could not verify
        # might be the one that fails. Neither supports a clean.
        band = "not_evaluated"

    notes = (
        f"Audit workflow step 3: both models require at least {minimum} turns. A "
        "turn is one exchange -- a user message and the reply that answers it "
        "share an index. A submission whose citations run past the last turn "
        "fetched is set aside, because a short count there is our fetch failing "
        "rather than the contributor's work."
    )
    if skipped:
        notes += (
            f" {len(skipped)} of {len(counts)} submission(s) could not be verified: "
            + "; ".join(f"model {m}: {why}" for m, why in sorted(skipped.items()))
            + "."
        )
    if not counts:
        notes += " No submission was filed at all."

    return build_verdict(
        task_id=task.task_id,
        check_id=96,
        band=band,
        measurement=Measurement(
            numerator=len(short),
            denominator=len(verifiable),
            threshold=minimum,
            counts={
                "minimum_turns": minimum,
                # Per model rather than pooled: a reader has to be able to see
                # which model was short and which one was set aside, and why.
                "turns_by_model": dict(sorted(counts.items())),
                "verified_by_model": dict(sorted(verifiable.items())),
                "unverifiable_by_model": dict(sorted(skipped.items())),
                "submissions": len(counts),
                "submissions_verified": len(verifiable),
                "longest_fetched": ceiling,
                "requires_complete_conversation": strict,
            },
            notes=notes,
        ),
        contributing_items=[f"model {m}: {verifiable[m]} turns" for m in short],
        # A fail rests on a count we could verify, so setting another submission
        # aside does not weaken it: the spec needs both models over the bar and
        # one is demonstrably under it.
        confidence="low" if band == "not_evaluated" else "high",
        policy=policy,
    )
