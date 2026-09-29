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

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Literal

from .config import DEFAULT_POLICY, Policy
from .env_context import FileReferenceScan
from .findings import (
    COUNTED_CONDITION_FLOORS,
    ArtifactFinding,
    AutofailFinding,
    CoverageFinding,
    CriterionFinding,
    CriterionRatingFinding,
    DimensionRatingFinding,
    Direction,
    DomainRelevanceFinding,
    EnvironmentContextFinding,
    InversionFinding,
    JustificationCoverage,
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
from .models import KeyTurn, ModelSlot, Sxs, Task
from .projects import profile_for_project
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


def _nonzero_band(rate: float, fail_threshold: float, inclusive: bool = True) -> Band:
    """Shared band logic for a rate gate.

    Fail thresholds are inclusive on the fail side by default. The non-fail band
    is any nonzero rate below the threshold: 240 reads "between 1% and 15%" while
    230 and 250 read "more than 0%", and a literal 1% floor would leave a single
    issue in a small rubric with no representable score.

    `inclusive` exists because a spec can say either, and which one it says is
    not decoration. Honeybee Aspirational's V5 rubric fails 250 at "20% or more"
    and 230 at "10% or more" but 240 at "More than 15% (>15%)", with its non-fail
    band spelled "Up to 15% (<=15%)". A rubric of 20 criteria with 3 carrying
    moderate issues sits exactly on 15%, and reading that as a fail convicts a
    task the customer's own QC passes. `policy.exclusive_fail_thresholds` names
    the gates that read this way.
    """
    failed = rate >= fail_threshold if inclusive else rate > fail_threshold
    if failed:
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
    band = _nonzero_band(
        rate, threshold, inclusive=gate not in policy.exclusive_fail_thresholds
    )

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

    # V5 words this one "over 20%", so 270 can be one of the exclusive gates; the
    # absolute-count leg is unaffected either way.
    over_rate = (
        rate > policy.rubric_eval_fail_rate
        if 270 in policy.exclusive_fail_thresholds
        else rate >= policy.rubric_eval_fail_rate
    )
    if count >= policy.rubric_eval_fail_count or over_rate:
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
    check_id: int,
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

    # 280's item population is the rubric, which runs from 0 to 54 criteria on
    # this data, so an absolute count of wrong citations is partly a measure of
    # how big the rubric was. The rate floor applies to 280 only: 310's
    # population is the dimension ratings, always the same size, so it has no
    # size to correlate with and gets no second leg. See
    # `policy.rubric_relevant_turns_fail_rate`.
    items = len(scoped)
    rate_floor = policy.rubric_relevant_turns_fail_rate if check_id == 280 else 0.0
    rate = (counted / items) if items else 0.0
    over_count = counted >= policy.relevant_turns_fail_count
    # A task with no audited items cannot clear a share of them, and it has no
    # denominator to have got a rate wrong with either; the count leg alone
    # decides it, exactly as it did before this floor existed.
    over_rate = (not rate_floor) or (not items) or rate >= rate_floor

    if over_count and over_rate:
        band: Band = "fail"
    elif counted > 0:
        band = "non_fail"
    else:
        band = "clean"

    # 281 is report-only: V5 does not fail tasks on passing-criteria turn
    # citations, so a count that would fail 280 is published as non_fail here.
    if check_id == 281 and band == "fail":
        band = "non_fail"

    return build_verdict(
        task_id=task.task_id,
        check_id=check_id,
        band=band,
        measurement=Measurement(
            numerator=counted,
            denominator=items or None,
            rate=rate if rate_floor and items else None,
            threshold=policy.relevant_turns_fail_count,
            counts={
                "incorrect_turns": incorrect,
                # Both legs of 280's fail condition, and which of them held. On
                # 310 the rate leg is absent and `rate_threshold` is None.
                "rate_threshold": rate_floor or None,
                "incorrect_turn_rate": round(rate, 3) if items else None,
                "over_count_threshold": over_count,
                "over_rate_threshold": bool(rate_floor) and over_rate,
                # True where the absolute count was reached and the share was
                # not: the population was large enough to accumulate three wrong
                # citations without the citations being systematically wrong.
                "held_below_rate_floor": over_count and not over_rate,
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
            notes=(
                "Findings are namespaced by check ID so a turn is never counted "
                "toward both 280 and 310. Turns cited beyond the end of the "
                "conversation the audit could fetch are reported as unverifiable "
                "and never counted."
                + (
                    f" Two legs, both required: at least "
                    f"{policy.relevant_turns_fail_count} incorrectly cited turn(s) "
                    f"AND at least {rate_floor:.0%} of the "
                    f"{items} audited item(s). The share leg exists because this "
                    "check's population is the rubric, which ranges from 0 to 54 "
                    "criteria, so a bare count rises with rubric size; the "
                    "absolute count is kept beside it so a small rubric with "
                    "three fabricated citations still fails."
                    if rate_floor
                    else " Absolute count, no percentage: this check's population "
                    "is the task's dimension ratings, which is the same size on "
                    "every task, so there is no denominator effect to correct for."
                )
            ),
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
    points: int,
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
    # Stated wherever a gap was counted, because the alternative is a reader
    # inferring the rule from `dimension_fail_major_count` and getting it wrong in
    # the direction of believing the bar is higher than it is.
    inert = ""
    if (
        policy.dimension_fail_major_count * policy.dimension_major_delta
        >= policy.dimension_fail_total_points
        and policy.na_mismatch_counts_as != "major"
    ):
        inert = (
            f"The major leg ({policy.dimension_fail_major_count} gap(s) of "
            f"{policy.dimension_major_delta}+ points) cannot fire on its own at "
            "these numbers, because that many major gaps already reach the summed-"
            "point bar; it is reported as major_leg_can_fire=false. "
        )
    if not policy.dimension_disagreement_requires_informed_confirmation:
        return (
            "BASIS: blind rating deltas, unconfirmed -- informed confirmation is "
            "switched off by policy, so every gap between the contributor's rating "
            "and an independent blind rating is counted as the spec reads it. "
        )
    if counted and not dropped:
        return (
            f"BASIS: {counted} rating gap(s) totalling {points} point(s), every "
            "one of them confirmed by an informed reader as a rating this "
            f"conversation does not support. The fail leg is the {points}, not "
            "the number of dimensions: the bar is "
            f"{policy.dimension_fail_total_points} summed points. " + inert
        )
    if counted:
        return (
            f"BASIS: {counted} confirmed rating gap(s) totalling {points} "
            f"point(s) against a bar of {policy.dimension_fail_total_points} "
            f"summed points; a further {dropped} gap(s) "
            "were found and are not counted -- see blind_only_items (a defensible "
            "difference of judgment), truncated_evidence_items (the blind rating "
            "read a conversation that had been cut), and unadjudicated_items. "
            + inert
        )
    if truncated or unadjudicated:
        return (
            f"BASIS: NOT EVALUATED. {len(truncated) + len(unadjudicated)} of "
            f"{dropped} rating gap(s) could not be judged either way -- "
            f"{len(truncated)} where the render was still cut when an informed "
            f"reader went to check it, {len(unadjudicated)} where confirmation "
            "could not be obtained -- and nothing was confirmed against the "
            "contributor. This check has judged neither for nor against them: the "
            "gaps are open questions, not a clean result. "
        )
    if dropped:
        return (
            f"BASIS: NO confirmed rating gaps. The blind pass disagreed on "
            f"{dropped} dimension(s) and an informed reader, shown the "
            "conversation and the reasoning given for each rating, found every one "
            "of them defensible on the evidence. Every gap was judged; none was a "
            "defect. "
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

    The fail leg is a SUM OF POINTS over the confirmed gaps, not a count of them,
    which is the spec's own unit and the one distinction no count leg can draw:
    three 1-point disagreements and three 3-point ones are the same number of
    items and very different findings. Swept over 171 tasks with every magnitude
    recovered from the stored draws, no pure count threshold separates this
    check's true fails from its false ones -- one QC-corroborated fail sits on two
    1-point gaps while two unambiguous false fails sit on 2-point gaps -- and the
    false fails concentrate on the 1-point gap. See
    `policy.dimension_fail_total_points`, including why the major leg cannot fire
    on its own at the current numbers.
    """
    major: list[str] = []
    total: list[str] = []
    # The magnitude of every counted gap, keyed by item. The gate's fail leg is a
    # sum over these, and publishing them is also what lets a completed run be
    # re-scored at a different threshold without reopening the response cache to
    # recover the deltas by inference.
    points_by_item: dict[str, int] = {}
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
            reasons[f.item_id].append("blind_render_truncated")
        if f.evidence_incomplete:
            reasons[f.item_id].append("truncated_evidence")
        if f.adjudication is not None:
            reasons[f.item_id].append(f"adjudicated_{f.adjudication}")
        elif f.adjudication_abstained:
            reasons[f.item_id].append("adjudication_abstained")

        if f.evidence_incomplete and policy.exclude_truncated_evidence_disagreements:
            truncated_excluded.append(f.item_id)
            continue
        if not f.counts_as_disagreement:
            if f.adjudication == "defensible":
                blind_only.append(f.item_id)
            else:
                unadjudicated.append(f.item_id)
            continue

        total.append(f.item_id)
        points_by_item[f.item_id] = delta
        if delta >= policy.dimension_major_delta:
            major.append(f.item_id)

    # Gaps that exist and that nobody managed to judge either way. A gap an
    # informed reader called defensible is judged -- someone read the conversation
    # and answered the question -- but one dropped because the render was cut, or
    # because confirmation never came back, is an open question, and an open
    # question must not read as an answer of "clean".
    unjudged = truncated_excluded + unadjudicated

    # The spec's unit: total point-deltas, summed over the confirmed gaps only.
    # An N/A mismatch that policy counts carries no numeric delta, so it lands in
    # `total` with no points and can only reach the band through the major leg.
    total_points = sum(points_by_item.values())

    # V5 states this leg as a count of disagreements rather than a sum of their
    # magnitudes -- "5 or more major or minor disagreements" -- which is a
    # different population from the points leg in both directions: five 1-point
    # gaps are 5 points and reach the count bar without reaching the points bar,
    # while two 3-point gaps are the reverse. Off unless a project sets it, and
    # unlike the points bar its false-fail rate here is unmeasured, because no
    # scored corpus exists for the project that states it.
    over_count = (
        policy.dimension_fail_total_count > 0
        and len(total) >= policy.dimension_fail_total_count
    )
    if (
        len(major) >= policy.dimension_fail_major_count
        or total_points >= policy.dimension_fail_total_points
        or over_count
    ):
        band: Band = "fail"
    elif total:
        band = "non_fail"
    elif unjudged:
        # The failure mode this exists for: on the scored run one task had all six
        # of its rating gaps dropped for truncation, so not one piece of evidence
        # was considered, and the check published `clean`. A check that judged
        # nothing has to say so.
        band = "not_evaluated"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=300,
        band=band,
        measurement=Measurement(
            # Points, matching `threshold`, so `numerator >= threshold` is the
            # fail leg as written. The item count is `total_disagreements` below;
            # it was the numerator until the leg changed unit, and every consumer
            # that wants "how many dimensions" should read that instead.
            numerator=total_points,
            denominator=considered,
            threshold=policy.dimension_fail_total_points,
            counts={
                "major_disagreements": len(major),
                "total_disagreements": len(total),
                # Summed |delta| over the confirmed gaps, and the per-item
                # magnitudes behind it. An N/A mismatch counted by policy appears
                # in `total_disagreements` and not here, because it has no
                # numeric delta to contribute.
                "total_points": total_points,
                "points_threshold": policy.dimension_fail_total_points,
                "points_by_item": {k: points_by_item[k] for k in sorted(points_by_item)},
                # Whether the major leg is reachable at all at the configured
                # numbers. It is not while `major_count * major_delta` is at or
                # above the points threshold, because every set of gaps that
                # trips it already trips the points leg; the one exception is an
                # applicability divergence under
                # `na_mismatch_counts_as == "major"`, which carries no points.
                # Published rather than left to a comment so that a future change
                # to any of the three numbers shows up in the output.
                "major_leg_can_fire": (
                    policy.dimension_fail_major_count * policy.dimension_major_delta
                    < policy.dimension_fail_total_points
                    or policy.na_mismatch_counts_as == "major"
                ),
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
                # Gaps left open rather than answered, and the reason this check
                # can report not_evaluated with findings in hand.
                "unjudged_gaps": len(unjudged),
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
                len(total),
                total_points,
                blind_only,
                truncated_excluded,
                unadjudicated,
                policy,
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


def evaluate_check_271(
    task: Task,
    findings: list[CriterionRatingFinding],
    policy: Policy = DEFAULT_POLICY,
    abstained: int = 0,
) -> CheckVerdict:
    """Simplified Rubric Evaluation. Same disagreement math as 270 over slot S,
    published as report-only because V5's Rubric Evaluation covers A/B/C only."""
    if task.simplified is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=271,
            band="not_evaluated",
            measurement=Measurement(notes="no simplified trajectory was filed"),
            confidence="low",
            policy=policy,
        )
    inner = evaluate_check_270(task, findings, policy, abstained=abstained)
    band: Band = "non_fail" if inner.band == "fail" else inner.band
    return build_verdict(
        task_id=task.task_id,
        check_id=271,
        band=band,
        measurement=inner.measurement,
        contributing_items=list(inner.contributing_items),
        confidence=inner.confidence,
        policy=policy,
    )


def evaluate_check_305(
    task: Task,
    findings: list[DimensionRatingFinding],
    policy: Policy = DEFAULT_POLICY,
    abstained: int = 0,
) -> CheckVerdict:
    """Collaboration Quality Rating. Report-only: V5's Major Rating lists six
    dimensions and collaboration is not among them."""
    collab = [
        f for f in findings if f.dimension.lower() == "collaboration quality"
    ]
    if not collab and not any(
        d.dimension == "Collaboration quality" for d in task.dimension_ratings
    ):
        return build_verdict(
            task_id=task.task_id,
            check_id=305,
            band="not_evaluated",
            measurement=Measurement(notes="no collaboration quality rating was recorded"),
            confidence="low",
            policy=policy,
        )
    inner = evaluate_check_300(task, collab or findings, policy, abstained=abstained)
    band: Band = "non_fail" if inner.band == "fail" else inner.band
    return build_verdict(
        task_id=task.task_id,
        check_id=305,
        band=band,
        measurement=inner.measurement,
        contributing_items=list(inner.contributing_items),
        confidence=inner.confidence,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 400 -- ranking disagreement
# ---------------------------------------------------------------------------


# Worst-first ranking, for the checks that summarise several sub-judgments into
# one verdict. `not_evaluated` outranks `clean` deliberately: an unanswered
# question is a weaker result than an answered one, and collapsing the two would
# let one unjudgeable comparison hide behind another that came out even.
_BAND_RANK: dict[str, int] = {
    "clean": 0,
    "not_evaluated": 1,
    "non_fail": 2,
    "fail": 3,
}


def evaluate_check_400(
    task: Task,
    finding: LikertFinding | list[LikertFinding],
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """One verdict covering every comparison the task recorded.

    Each pair is evaluated independently by `_evaluate_one_ranking` and the
    task takes the worst result, because a contributor who ranked A-vs-B well
    and A-vs-C indefensibly has still misranked a comparison. Every pair's delta
    and band is reported alongside, so which one drove the verdict is visible
    rather than inferred.
    """
    findings = finding if isinstance(finding, list) else [finding]
    if not findings:
        return build_verdict(
            task_id=task.task_id,
            check_id=400,
            band="not_evaluated",
            measurement=Measurement(notes="No comparable Likert judgment was produced."),
            confidence="low",
            policy=policy,
        )

    judged = [(f, _evaluate_one_ranking(task, f, policy)) for f in findings]
    if len(judged) == 1:
        return judged[0][1]

    worst_finding, worst = max(judged, key=lambda pair: _BAND_RANK[pair[1].band])
    counts = dict(worst.measurement.counts)
    counts["pair"] = worst_finding.pair
    counts["pairs_judged"] = ", ".join(f.pair for f, _ in judged)
    for f, v in judged:
        counts[f"delta_{f.pair}"] = f.delta
        counts[f"band_{f.pair}"] = v.band
    return build_verdict(
        task_id=task.task_id,
        check_id=400,
        band=worst.band,
        measurement=Measurement(
            numerator=worst.measurement.numerator,
            denominator=worst.measurement.denominator,
            rate=worst.measurement.rate,
            threshold=worst.measurement.threshold,
            counts=counts,
            notes=(
                f"Worst of {len(judged)} comparisons ({worst_finding.pair}). "
                + worst.measurement.notes
            ),
        ),
        confidence=min((f.confidence for f in findings), key=_CONFIDENCE_ORDER.get),
        policy=policy,
    )


def _evaluate_one_ranking(
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
        elif (
            finding.evidence_incomplete
            and policy.exclude_truncated_evidence_disagreements
        ):
            excluded_reason = "truncated_evidence"
        elif confirmation_on and not finding.counts_as_disagreement:
            excluded_reason = (
                "blind_only"
                if finding.adjudication == "defensible"
                else "unadjudicated"
            )

    counted = delta if not excluded_reason else 0

    # A gap dropped because nobody could judge it is an open question, and this
    # check has no band below clean to put one in -- so it says so instead. Only
    # `blind_only` is an actual answer: an informed reader read both transcripts
    # and the contributor's reasoning and found the preference defensible.
    unjudged = excluded_reason in (
        "asymmetric_render",
        "truncated_evidence",
        "unadjudicated",
    )

    if counted >= policy.likert_fail_delta:
        band: Band = "fail"
    elif counted > 0:
        band = "non_fail"
    elif unjudged:
        band = "not_evaluated"
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
            "the conversations themselves. Reported not_evaluated rather than "
            "clean: a comparison nobody could make fairly is not a comparison that "
            "came out even."
        )
    elif excluded_reason == "truncated_evidence":
        note = (
            f"BASIS: a {delta}-point ranking gap, NOT judged: at least one "
            "conversation was still cut to fit the render budget when an informed "
            "reader went to check the preference, so the grounds for it may sit in "
            "the part nobody was shown. Reported not_evaluated rather than clean -- "
            "the question is open, not answered."
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
            f"BASIS: a {delta}-point ranking gap, NOT judged: informed "
            "confirmation could not be obtained"
            + (f" ({finding.why_unadjudicated})" if finding.why_unadjudicated else "")
            + ", so this is reported not_evaluated rather than clean."
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=400,
        band=band,
        measurement=Measurement(
            numerator=counted,
            threshold=policy.likert_fail_delta,
            counts={
                "pair": finding.pair,
                "contributor_likert": finding.contributor_likert,
                "auditor_likert": finding.auditor_likert,
                "delta": delta,
                "counted_delta": counted,
                "not_counted_because": excluded_reason,
                "unjudged": unjudged,
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


def aggregate_repeated_conditions(
    findings: list[JustificationFinding], triggered: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Credit a defect repeated across justifications that trips no floor alone.

    Every counted condition's floor is read inside one justification, which is
    right for the thing the spec describes -- an argument that leans twice on
    nothing. It is wrong for the contributor who leans once on nothing in each
    of five arguments: five instances of the same defect, and the check sees a
    clean task. The observed case is a task where two justifications each rested
    on the same unexplained "all five target deliverables" claim; the judge
    named the repetition itself and still could not count it.

    Only justifications already carrying at least one instance are eligible, so
    this reaches a pattern rather than manufacturing one, and the credit is
    proportionate: `total // floor` justifications are marked, not all of them.
    Crediting every eligible justification was tried first and is far too blunt
    -- nine arguments each leaning once on nothing became nine failures rather
    than the four they are worth, and replaying that rule over 505 cached tasks
    turned 26 of them into fails, two of the three with a human verdict being
    ones the customer passed. Two half-defects make one defect, and the marked
    justifications are the heaviest, so a task needs a real spread of them
    before the share moves.

    Conditions credited here must not reach the severity escape: "one is enough"
    is the spec's judgement about a single argument resting on something false,
    and a defect diffuse enough to need aggregating is by construction not that.
    They count toward the share and nothing else.
    """
    aggregated: dict[str, list[str]] = {}
    for condition, (attr, floor) in COUNTED_CONDITION_FLOORS.items():
        eligible = [
            f
            for f in findings
            if 0 < getattr(f, attr) < floor
            and condition not in triggered.get(f.item_id, ())
        ]
        credits = sum(getattr(f, attr) for f in eligible) // floor
        if len(eligible) < 2 or not credits:
            continue
        eligible.sort(key=lambda f: (-getattr(f, attr), f.item_id))
        for f in eligible[:credits]:
            aggregated.setdefault(f.item_id, []).append(condition)
    return aggregated


def _justification_group(item_id: str) -> str:
    """The model call an item belongs to: a model slot, or the ranking on its own.

    Mirrors how `justification_coverage` groups the same ids. Kept as one function
    because two readings of "which side of the task is this" that can disagree are
    worse than either.
    """
    head = item_id.split("::", 1)[0].strip()
    return head or "ranking"


def _justification_band_v5(
    task: Task,
    triggered: dict[str, list[str]],
    severe: dict[str, list[str]],
    with_issues: set[str] | dict[str, object],
) -> tuple[Band, dict[str, object]]:
    """The V5 rubric's band for check 450, which is structural rather than a rate.

    Quoted from its fail cell: "Every model has at least 1 Major Issue, or every
    comparison has at least 1 Major Issue", and from its non-fail cell: "At least
    one model and at least one comparison is free of Major Issues, or the task
    justifications contain only Minor Issues".

    So a major issue on its own convicts nothing here. A task whose Model A
    justification fabricates a citation is non-fail as long as B and C are clean,
    which is a far more forgiving rule than the share-and-severe one this build
    was tuned to, and the gap between them is the largest single strictness
    difference between the two specs. Minor issues can only ever reach non-fail.

    "Major" maps to `policy.justification_severe_conditions`, which is the same
    population the severe escape uses: a fabricated or misattributed claim, or
    supporting claims that contradict the verdict they defend. The evidence-
    quality filters upstream still apply -- a condition only one draw named has
    already been dropped before this sees it -- but the share floor deliberately
    does not, because this rule asks where the majors fall and not how many.
    """
    models = [
        slot
        for slot in (task.audit_slots or task.expected_slots)
        if slot != "S"
    ]
    pairs = [c.pair_label for c in task.comparison_list()]

    majors_by_model = {m: False for m in models}
    majors_by_pair = {p: False for p in pairs}
    for item in severe:
        head, _, tail = item.partition("::")
        if head == "ranking":
            # A single-comparison task files its ranking justification as plain
            # "ranking" with no pair in the id.
            label = tail or (pairs[0] if len(pairs) == 1 else "")
            if label in majors_by_pair:
                majors_by_pair[label] = True
        elif head in majors_by_model:
            majors_by_model[head] = True

    every_model = bool(majors_by_model) and all(majors_by_model.values())
    every_pair = bool(majors_by_pair) and all(majors_by_pair.values())

    counts: dict[str, object] = {
        "models_with_a_major_issue": sorted(m for m, hit in majors_by_model.items() if hit),
        "comparisons_with_a_major_issue": sorted(
            p for p, hit in majors_by_pair.items() if hit
        ),
        "every_model_has_a_major_issue": every_model,
        "every_comparison_has_a_major_issue": every_pair,
    }

    if every_model or every_pair:
        return "fail", counts
    if triggered or with_issues:
        return "non_fail", counts
    return "clean", counts


def evaluate_check_450(
    task: Task,
    findings: list[JustificationFinding],
    policy: Policy = DEFAULT_POLICY,
    coverage: JustificationCoverage | None = None,
    evidence_by_group: dict[str, bool] | None = None,
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

    The share does not govern every condition. Two of them -- a fabricated or
    misattributed citation, and a false claim the rating rests on -- carry a
    floor of one in the spec's own words, and a proportion applied on top of
    that floor overrode it: one instance of fifteen is 6.7%, so the two
    strictest numerators in the audit were the two the gate could not see.
    `policy.justification_severe_conditions` names them, and a justification
    tripping one fails the task at any share. The rate still decides everything
    else, which is what keeps the escape from collapsing back into the
    any-trigger gate: it is a second route to fail on a defect that needs no
    corroboration, not a lower bar for the ones that do.

    "Needs no corroboration" is about the other fourteen justifications, not
    about the judge. The findings reaching this gate are aggregated across
    independent draws, so the escape has always required a majority of them --
    and only where there were draws to combine. A group that lost four of five
    calls arrives as one draw's opinion, and two sheet22 tasks failed on a
    severe condition exactly one stored draw named.
    `policy.justification_severe_min_draws` is the floor on that, read off the
    finding's own draw record, and it exempts findings nothing sampled rather
    than denying them.

    `policy.justification_severe_min_share` is the separate and larger question
    of whether one bad justification of fifteen should fail a task at all. It is
    off by default because the spec's answer is yes and this build is not
    entitled to overrule it quietly; the field exists so that overruling it is a
    decision someone takes. Either way a withheld escape keeps its place in the
    share's numerator: the defect is still a defect.

    Pooled scope keeps the any-trigger fail. Pooling collapses the task into one
    synthetic justification, so there is no population left to take a share of;
    a rate of one-over-fifteen would silence the check entirely.

    `coverage` is what the population was supposed to be. Without it the
    denominator is simply however many findings arrived, which is why a task
    whose model B set never came back reported clean over model A's eight, with
    an empty condition map and no sign anything was missing; the check could not
    tell a short task from a lost call, and QC had failed that task for this very
    check. Supplied, a model whose judged share falls under
    `policy.min_judged_ratio` costs the check its licence to say the task did not
    fail. Callers replaying stored findings against a bare `Task` pass nothing
    and keep the old behaviour, which is correct for them -- they hold no
    dimension ratings to derive an expectation from.

    `evidence_by_group` closes the one hole `coverage` cannot see. Coverage counts
    how many justifications came back, not whether there was anything to judge them
    against: a submission whose share page died, whose turn manifest was empty and
    which carried no exported PDF renders a blank conversation block, and the judge
    is then asked to audit fifteen arguments about a transcript it was never shown.
    Every observed instance abstained, all counters zero, which coverage catches --
    but nothing in the code requires that, and a judge that reports a defect instead
    of abstaining makes its items `audited`, leaves coverage complete, and carries a
    fail on evidence that does not exist. So a fail whose every triggering item
    belongs to a model with no rendered conversation is reported `not_evaluated`.

    Deliberately narrow. One triggering item from a model whose transcript was
    present keeps the fail, for the reason the paragraph above gives: a defect read
    from evidence survives a short population. Twelve submissions across three
    corpora failed to hydrate and ten still reached the judge with 9.5k to 114k
    characters, so this is a guard on the two that did not, not on hydration
    trouble in general.
    """
    triggered: dict[str, list[str]] = {}
    aggregated: dict[str, list[str]] = {}
    with_issues: list[str] = []

    if policy.justification_scope == "per_justification":
        for f in findings:
            conditions = f.triggered_conditions()
            if conditions:
                triggered[f.item_id] = conditions
            if f.has_any_issue():
                with_issues.append(f.item_id)
        aggregated = aggregate_repeated_conditions(findings, triggered)
        for item, conds in aggregated.items():
            triggered.setdefault(item, []).extend(conds)
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

    by_item = {f.item_id: f for f in findings}
    severe = {
        item: [
            c
            for c in conds
            if c in policy.justification_severe_conditions
            and c not in aggregated.get(item, ())
        ]
        for item, conds in triggered.items()
    }
    severe = {item: conds for item, conds in severe.items() if conds}

    # How much of the judge is behind each escape, and what the bar rejected.
    # Published because the escape is the one route in this check that asserts a
    # task-level fail from a single justification, and a reader cannot weigh that
    # without knowing whether one draw said it or five did.
    support = {
        f"{item}::{c}": by_item[item].condition_support(c)
        for item, conds in severe.items()
        for c in conds
        if item in by_item
    }
    suppressed = {
        item: [
            c
            for c in conds
            if (s := support.get(f"{item}::{c}")) is not None
            and s < policy.justification_severe_min_draws
        ]
        for item, conds in severe.items()
    }
    suppressed = {item: conds for item, conds in suppressed.items() if conds}
    severe = {
        item: kept
        for item, conds in severe.items()
        if (kept := [c for c in conds if c not in suppressed.get(item, ())])
    }

    # The "one of many" floor. A severe condition still counts toward the share
    # when the floor withholds the escape -- the finding is not in doubt, only
    # whether one justification of fifteen should fail a task on its own.
    severe_share = (len(severe) / denominator) if denominator else 0.0
    below_floor = bool(severe) and severe_share < policy.justification_severe_min_share
    escapes = bool(severe) and not below_floor
    over_share = bool(triggered) and rate >= policy.justification_fail_rate

    coverage_of_majors: dict[str, object] = {}
    if policy.justification_fail_rule == "all_models_or_all_comparisons":
        band, coverage_of_majors = _justification_band_v5(
            task, triggered, severe, with_issues
        )
    elif triggered and (not proportional or escapes or over_share):
        band: Band = "fail"
    elif triggered or with_issues:
        band = "non_fail"
    else:
        band = "clean"

    # A fail survives a short population. The defect is in a justification that
    # was read, and finding it in eight of fifteen is still finding it -- the
    # existing rule that one abstention must not take the remaining
    # justifications out of the check is the same principle. Only the two bands
    # that assert an absence depend on having read the population: clean and
    # non_fail both say the task did not reach the bar, and neither is a
    # statement anyone can make about justifications that never arrived.
    undercovered = coverage.undercovered(policy.min_judged_ratio) if coverage else {}
    if undercovered and band != "fail":
        band = "not_evaluated"

    # The evidence half of the same question. `evidence_by_group` is keyed the way
    # `justification_coverage` groups items; the ranking justification is judged
    # against both conversations, so it is unattributable only when neither model
    # rendered one.
    blind_groups: list[str] = []
    if evidence_by_group is not None and triggered and band == "fail":
        any_model_evidence = any(
            present for group, present in evidence_by_group.items() if group != "ranking"
        )
        for item in sorted(triggered):
            group = _justification_group(item)
            present = (
                any_model_evidence
                if group == "ranking"
                else evidence_by_group.get(group, True)
            )
            if not present:
                blind_groups.append(group)
        if len(blind_groups) == len(triggered):
            band = "not_evaluated"
    # Recorded either way, because a fail that stands on one model's transcript
    # while another's was blank is a fail a reader should be able to see the shape
    # of.
    blind_groups = sorted(set(blind_groups))

    all_conditions = sorted({c for cs in triggered.values() for c in cs})
    notes = (
        "Population is every dimension justification for every model plus the "
        "ranking justification. Each condition's own floor is the spec's and is "
        "applied within a single justification; the share of justifications that "
        "must trip before the task fails is not stated by the spec and the "
        "threshold reported here is this build's own. Conditions the spec calls "
        "one-is-enough -- a fabricated or misattributed citation, a false claim "
        "the rating rests on -- fail the task at any share."
        if proportional
        else "Population is every dimension justification for every model plus "
        "the ranking justification. Counts are pooled across the task, so any "
        "triggered condition fails and no proportion is computed."
    )
    if blind_groups and band == "not_evaluated":
        notes += (
            " NOT judged: every justification that tripped a condition belongs to "
            + ", ".join(f"model {g}" for g in blind_groups)
            + ", whose conversation could not be rendered at all -- no turn of it "
            "reached the judge. A defect reported against a transcript nobody was "
            "shown is not a defect this check can assert, so the question is "
            "reported open rather than answered."
        )
    elif blind_groups:
        notes += (
            " Reported despite missing evidence for "
            + ", ".join(f"model {g}" for g in blind_groups)
            + ": at least one triggering justification belongs to a model whose "
            "conversation was rendered in full, and a defect read from evidence "
            "stands on the evidence it was read from."
        )
    if undercovered and band == "not_evaluated":
        notes += (
            " NOT judged: "
            + ", ".join(
                f"model {group} reached {share:.0%} of the justifications its own "
                "dimension ratings carry"
                for group, share in sorted(undercovered.items())
            )
            + f", under the {policy.min_judged_ratio:.0%} floor. The share this "
            "check gates on would be taken over whichever justifications happened "
            "to come back, so this is reported not_evaluated rather than clean -- "
            "the question is open, not answered."
        )
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
                "fail_rule": policy.justification_fail_rule,
                # Empty unless the V5 rule decided the band, in which case these
                # are what it decided on: which models and which comparisons
                # carry a major issue, and whether that covers all of either.
                **coverage_of_majors,
                "severe_conditions_by_item": {k: list(v) for k, v in sorted(severe.items())},
                # How many draws named each surviving escape, and the ones the
                # draw bar and the share floor withheld. None means the finding
                # was never sampled, so there was no draw record to judge.
                "severe_draws_by_condition": dict(sorted(support.items())),
                "severe_withheld_too_few_draws": {
                    k: list(v) for k, v in sorted(suppressed.items())
                },
                "severe_withheld_below_share_floor": (
                    {k: list(v) for k, v in sorted(severe.items())}
                    if below_floor
                    else {}
                ),
                # True only where the share alone would have passed the task, so
                # a reader can tell a pattern of weak arguments from a single
                # fabricated citation without recomputing either.
                "failed_on_severity": escapes and proportional and not over_share,
                "conditions_triggered": all_conditions,
                # The flat list above is a union over the task, so on its own it
                # accuses all fifteen justifications of every condition any one
                # of them tripped. The map is what a reader should be shown.
                "conditions_by_item": {k: list(v) for k, v in sorted(triggered.items())},
                # Credited to a justification only because the same defect
                # recurs elsewhere on the task. Separated so a reader can see
                # which failures rest on a pattern rather than on the
                # justification's own counts, and excluded from the escape.
                "aggregated_conditions_by_item": {
                    k: list(v) for k, v in sorted(aggregated.items())
                },
                "unverifiable_claims": sum(f.unverifiable_claims for f in findings),
                "scope": policy.justification_scope,
                # What the population was supposed to be, beside what it was.
                # Published even when coverage is complete, because "15 of 15"
                # is the only thing that makes a denominator of 15 meaningful.
                "coverage": coverage.to_dict() if coverage else None,
                "undercovered_groups": undercovered,
                # Whether each model's conversation reached the judge at all, and
                # which of the triggering items were read against one that did not.
                # None where the caller could not say, which is every replay of
                # stored findings against a bare `Task`.
                "evidence_by_group": (
                    dict(sorted(evidence_by_group.items()))
                    if evidence_by_group is not None
                    else None
                ),
                "unevidenced_groups": blind_groups,
            },
            notes=notes,
        ),
        contributing_items=sorted(triggered) or sorted(set(with_issues)),
        confidence=(
            "low"
            if undercovered
            else _lowest_confidence([f.confidence for f in findings])
        ),
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
    """Shape B for now, and composed of two halves that can succeed independently.

    A file-kind finding counts only when the file half actually ran, which needs a
    manifest that names something. Reporting a named file as absent from a manifest
    of opaque CDN paths would fail every task that attached one, and reporting it
    absent from a manifest we never ingested would fail tasks whose input files are
    sitting in the export unread. Either way the reference is unverified, not
    violated, and unverified never fails.

    The task is clean when at least one half ran and neither found a violation, and
    not_evaluated only when neither half could run at all -- which is the honest
    answer, not a pass.

    A violation reports `non_fail`, not `fail`, and that is a temporary demotion
    rather than a reading of the requirement -- the audit workflow tab states this
    one flatly and gives it no middle band. Three defects were measured behind 11
    of 13 escalations across three corpora: the judge was handed the models' own
    output files under the heading "files the task declares were supplied", the
    parser hardcoded the two questions that decide whether a missing file is a
    violation at all, and a judge that searched for a reference and FOUND it had
    nowhere to say so. All three are fixed in the same change as this line, which
    is exactly why the band moves: the fix alters what the judge is asked, so its
    output cannot be trusted on the strength of code review, and a wrong finding
    should cost a note rather than a $6 task until a scored run says otherwise.
    Restoring the fail band is a one-word edit in `registry` and this branch.
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
        band: Band = "non_fail"
    elif evaluated:
        band = "clean"
    else:
        band = "not_evaluated"

    notes = (
        "The bar is the audit workflow tab's: the prompt must reference only the "
        "entities and events in the task's universe. A reference the prompt merely "
        "proposes or asks about, one the work does not need, and any public "
        "real-world entity are all excluded. Every finding must quote the prompt and "
        "name what it was checked against; one that does neither is discarded. "
        "Reported rather than failed: the requirement is stated as a fail, but this "
        "check is running one band down while the fixes to what its judge is asked "
        "await validation on a scored run."
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


def _empty_final_outputs(task: Task, policy: Policy) -> list[str]:
    """Submissions that delivered a file and uploaded nothing to the final section.

    The named-file comparison cannot see this. `attachments` is the union of the
    final-outputs upload and the per-turn uploads, so a task that attached every
    file turn by turn and left the final-outcome section empty matches all of its
    own claims and scores clean -- which is what happened on a task the customer
    failed for exactly this, with 24 files uploaded and the section blank.

    Deliveries are counted from the transcript's own artifact signals rather than
    from `claimed_files`, because the case that matters most is an inline image,
    which has no readable name to claim.
    """
    if not policy.final_outputs_section_required:
        return []
    from .context import detect_artifact_signals

    gaps: list[str] = []
    for sub in task.submissions():
        # Absent on taxonomies with no such section, where empty means "not
        # asked for" and failing a contributor over it would be inventing a rule.
        #
        # On this customer's data the key is usually absent even where the
        # section exists, which makes this leg nearly unreachable -- 54 of 242
        # submissions had no key and exactly one was present-and-empty. Widening
        # it is deliberately not done here: the customer runs their own check
        # for unpreserved artifacts, and a second one firing from this pipeline
        # would double-count against it.
        if not sub.final_outputs_declared or sub.final_output_uploads:
            continue
        delivered = sum(1 for t in sub.assistant_turns() if detect_artifact_signals(t))
        if delivered:
            gaps.append(
                f"{sub.model}: final-outcome upload section is empty though "
                f"{delivered} turn(s) delivered a file"
            )
    return gaps


def _nothing_was_compared(
    task: Task, findings: list[ArtifactFinding], policy: Policy = DEFAULT_POLICY
) -> list[str]:
    """Submissions whose comparison had nothing in it to compare.

    "No artifacts mentioned at all is a clean 5" is right about the conversation
    it was written for -- most produce no files, and failing them would be
    inventing a defect. What it cannot distinguish is a conversation that
    produced nothing from an extraction that read nothing, and it published the
    same clean for both. On the 60-task sheet22 run, 26 of 60 tasks reported
    `files_claimed: 0` and 95 was the worst check in the run at 2 true positives
    against 8 misses; the auditors' own text on the missed ones cites "the pdf
    trajectory" and "the pdf file of the model's chat", which is to say the
    deliveries were made in the exported chat PDF that this check's prompt does
    not render.

    Two states are separated here, both per submission, and both are about what
    the subtraction had in its hands rather than about what the conversation
    looked like:

      * the extraction read no filename at all while the contributor uploaded
        files for that same submission. Zero claimed against a non-empty
        manifest is not a comparison -- `_match_uploads` subtracts an empty set
        and every upload comes back unaccounted for -- and the contributor's own
        uploads are the evidence that this submission produced something.
      * no model replies were fetched for the submission, and no exported
        transcript stood in for them. "This response produced no files" is then a
        fact about the fetch, not about the response, and the extraction was
        shown the user's half of a conversation. The PDF fallback is checked
        because a submission it rescued did have the model's words in front of
        the extraction, even though they never became numbered turns.

    Deliberately not used as signals: deliverable markers and download-link
    patterns in the conversation text. `detect_artifact_signals` fires on
    "I've created ..." and matches ordinary prose, so keying an abstention off it
    would take file-free conversations -- the majority, and the population the
    clean-5 rule exists to protect -- out of the clean band wholesale. Nor
    truncation, for the same reason: long conversations are common and most of
    them still produce nothing.

    Replayed against the sheet22 run this reaches 25 of the 60 tasks, 24 of them
    on the uploads leg. Every one of those 24 holds a submission the contributor
    uploaded files for, so none is a file-free conversation losing its clean 5:
    the 15 tasks whose zero-name extraction sits against an empty manifest keep
    it. What the check gives up is the claim that it checked those 25, which it
    had not.
    """
    subs = {s.model: s for s in task.submissions()}
    out: list[str] = []
    for f in findings:
        if f.claimed_files:
            continue
        if f.uploaded_files:
            out.append(
                f"{f.model}: no filename was read out of the conversation, while "
                f"{len(f.uploaded_files)} file(s) were uploaded for this submission, "
                "so nothing was compared against the manifest"
            )
            continue
        sub = subs.get(f.model)
        if sub is None or sub.assistant_turns():
            continue
        rescued = bool(
            policy.fetch_attachment_text
            and policy.transcript_pdf_fallback
            and getattr(sub, "transcript_pdf", "")
        )
        if not rescued:
            out.append(
                f"{f.model}: no model replies were fetched, so an empty extraction "
                "says nothing about whether this response produced files"
            )
    return out


def evaluate_check_95(
    task: Task, findings: list[ArtifactFinding], policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Shape A: fail if any claimed output file was never uploaded.

    A response that produced no files at all is an explicit clean, so an empty
    extraction must never fail. An extraction that read nothing out of a
    submission whose files *were* uploaded is a third state and reports
    not_evaluated -- see `_nothing_was_compared`. Order matters: a confirmed
    missing file or an empty final-outcome section is a finding made on evidence
    and outranks an abstention made for want of it.

    What is counted is the names a confirming pass agreed were absent, not the
    names a string comparison could not match. The two are different populations:
    on a manual audit of nine fails here, five were the comparison's own limits --
    a screenshot whose capture-tool name no distance metric ties to the
    contributor's name for the same image, an inline code block read as an
    undelivered file. This check has no middle band, so each of those was a
    task-level fail with nothing to soften it. Every name the confirmation clears
    is still published, with the reason, under `cleared_by_confirmation`.

    That sentence had an exception, and the exception was where this check's false
    fails lived. On the *count* basis -- reached precisely when the upload
    manifest carries no readable filename -- there is no name to put to the
    confirming call, so the call never runs, and a shortfall reached that way was
    published as a task-level fail with no file named and the one safeguard
    designed to catch exactly this bypassed. Six of twelve escalations on the
    scored corpus came this way, four on tasks human QC passed, three of those as
    the sole cause of the task's fail. "16 files missing of 0 claimed, 0 uploaded"
    is not a finding anyone can act on.

    So a shortfall no confirmation could reach now reports `not_evaluated`
    instead: the counts, the shortfall and the reason are all still published, and
    what is withdrawn is only the claim that the check established a defect. It
    costs no recall -- both QC-failed tasks it also covered fail on other checks
    -- and it restores this check's own stated design to the one path that
    routed around it. A shortfall on a submission whose confirmation *did* run
    keeps failing, and so does a named `confirmed_missing`, whatever basis its
    sibling submission used.
    """
    named = [f"{f.model}:{name}" for f in findings for name in f.confirmed_missing]
    # A count-basis shortfall is unconfirmable by construction:
    # `_confirm_missing_artifacts` only takes `basis == "filename"` findings with
    # names in `missing_files`, and the count basis has neither. `confirmation_ran`
    # is tested too so that the discriminator is the fact that matters -- no
    # second opinion was obtained -- rather than a basis label that could later
    # acquire a confirming path of its own.
    def is_unconfirmable(f) -> bool:
        return bool(f.basis == "count" and f.shortfall and not f.confirmation_ran)

    unconfirmable = [f for f in findings if is_unconfirmable(f)]
    shortfall = sum(
        f.shortfall for f in findings if f.basis == "count" and not is_unconfirmable(f)
    )
    unconfirmable_shortfall = sum(f.shortfall for f in unconfirmable)
    # The denominator carries the unverifiable deliveries as well as the named
    # ones, because the count basis is reached precisely when names could not be
    # read: a session that delivered eighteen images and preserved none has an
    # empty `claimed_files`, and reporting "18 of 0" states a true number inside
    # a sentence that reads as nonsense.
    counted = [
        f"{f.model}: {f.shortfall} of "
        f"{len(f.claimed_files) + len(f.unverifiable_files)} files not accounted for"
        for f in findings
        if f.basis == "count" and f.shortfall and not is_unconfirmable(f)
    ]
    unconfirmed_counts = [
        f"{f.model}: {f.shortfall} of "
        f"{len(f.claimed_files) + len(f.unverifiable_files)} files not accounted "
        "for, on a manifest carrying no readable filename -- nothing to put to a "
        "confirming call, so this is reported and not counted"
        for f in unconfirmable
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

    empty_section = _empty_final_outputs(task, policy)
    if empty_section:
        notes += (
            " A submission delivered files and left the final-outcome upload "
            "section empty. That is the customer's own fail condition and it is "
            "structural, not a name comparison: the count above can read 'all "
            "files uploaded' from per-turn attachments while the section the "
            "auditor opens holds nothing."
        )

    if unconfirmable:
        notes += (
            " Reported not_evaluated rather than fail for "
            + "; ".join(sorted(f.model for f in unconfirmable))
            + f": {unconfirmable_shortfall} delivery(ies) are unaccounted for on a "
            "count of files, with no filename on either side to name them and so "
            "nothing the confirming pass could be asked about. The shortfall is "
            "real arithmetic and is published above; what is withheld is the claim "
            "that a file was established missing, because this check's own rule is "
            "that what counts is the names a confirming pass agreed were absent, "
            "and on this path that pass cannot run at all."
        )

    uncompared = _nothing_was_compared(task, findings, policy)
    if uncompared and not (missing or empty_section):
        notes += (
            " Reported not_evaluated rather than clean: " + "; ".join(uncompared) + ". "
            "An empty extraction is a clean 5 when the conversation produced no "
            "files, which is most of them, but it cannot also stand for an "
            "extraction that read nothing out of a submission whose files were "
            "uploaded -- the deliveries in that case are typically inside the "
            "exported chat PDF, which this check's prompt does not render."
        )

    return build_verdict(
        task_id=task.task_id,
        check_id=95,
        band=(
            "fail"
            if missing or empty_section
            else (
                "not_evaluated"
                if uncompared or unconfirmable
                else "clean"
            )
        ),
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
                + shortfall
                + unconfirmable_shortfall,
                # A shortfall found on a count of files, where neither side of
                # the comparison carried a name to put to the confirming pass.
                # Published in full and deliberately not counted: this is the
                # path that produced "16 files missing of 0 claimed" as a
                # task-level fail with no file named.
                "unconfirmable_count_shortfall": unconfirmable_shortfall,
                "unconfirmable_count_by_model": {
                    f.model: f.shortfall for f in unconfirmable
                },
                "unconfirmable_count_detail": unconfirmed_counts,
                "cleared_by_confirmation": cleared,
                "confirmation_reasons": confirmation_reasons,
                "confirmation_ran_for": sorted(
                    f.model for f in findings if f.confirmation_ran
                ),
                "confirmation_required": policy.artifact_miss_requires_llm_confirmation,
                "fuzzy_threshold": policy.filename_fuzzy_match_threshold,
                "empty_final_outputs_section": empty_section,
                # Published whether or not it changed the band, so a reader can
                # see that a fail was reached on one submission while the other
                # was never compared.
                "uncompared_submissions": uncompared,
            },
            notes=notes,
        ),
        contributing_items=named + counted + empty_section,
        confidence=(
            "low"
            if (uncompared or unconfirmable) and not (missing or empty_section)
            else _lowest_confidence([f.confidence for f in findings])
        ),
        policy=policy,
    )


def evaluate_check_110(
    task: Task,
    finding: KeyTurnJustificationFinding | list[KeyTurnJustificationFinding] | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape B, and an unusually low bar: 'or contains any issues'.

    Scored independently of 100 -- a correctly chosen turn can carry a bad
    justification, and both combinations occur. V5 grades every Full Trajectory
    model; a list of findings is worst-of (any slot with an issue → non_fail).
    A single finding keeps the counts and notes the original callers expect.
    """
    findings = (
        list(finding)
        if isinstance(finding, list)
        else ([finding] if finding is not None else [])
    )
    findings = [f for f in findings if f is not None]
    if not findings:
        return build_verdict(
            task_id=task.task_id,
            check_id=110,
            band="not_evaluated",
            measurement=Measurement(notes="no key turn justification was audited"),
            confidence="low",
            policy=policy,
        )

    if len(findings) == 1:
        f = findings[0]
        return build_verdict(
            task_id=task.task_id,
            check_id=110,
            band="non_fail" if f.is_issue else "clean",
            measurement=Measurement(
                counts={
                    "describes_selected_turn": f.describes_selected_turn,
                    "claims_are_accurate": f.claims_are_accurate,
                    "connects_to_core_value": f.connects_to_core_value,
                    "issues": len(f.issues),
                },
                notes="Any defect qualifies; this is broader than 450's enumerated "
                "conditions. Scored independently of check 100.",
            ),
            contributing_items=list(f.issues),
            confidence=f.confidence,
            policy=policy,
        )

    issues = [f for f in findings if f.is_issue]
    counts: dict[str, object] = {
        "slots_audited": len(findings),
        "slots_with_issues": len(issues),
    }
    contributing: list[str] = []
    for f in findings:
        label = f.model or "task"
        counts[f"describes_selected_turn_{label}"] = f.describes_selected_turn
        counts[f"claims_are_accurate_{label}"] = f.claims_are_accurate
        counts[f"connects_to_core_value_{label}"] = f.connects_to_core_value
        counts[f"issues_{label}"] = len(f.issues)
        if f.is_issue:
            prefix = f"{f.model}:" if f.model else ""
            if f.issues:
                contributing.extend(f"{prefix}{i}" for i in f.issues)
            else:
                contributing.append(f"{prefix}justification defect")

    return build_verdict(
        task_id=task.task_id,
        check_id=110,
        band="non_fail" if issues else "clean",
        measurement=Measurement(
            counts=counts,
            notes="Worst-of across Full Trajectory models; S excluded. Any defect "
            "on any slot qualifies. Scored independently of check 100.",
        ),
        contributing_items=contributing,
        confidence=_lowest_confidence([f.confidence for f in findings]),
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
    task: Task,
    finding: VerdictFinding | list[VerdictFinding] | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Shape A, and the narrowest check in the spec.

    Only whether a preference is stated -- not whether it is correct (400),
    consistent (460), or well argued (450). A reasoned tie is a stated
    preference.

    On a task with several comparisons the fail band needs ALL of them to be
    silent; some stated and some not is the non-fail band. Quoted from the V5
    spec, whose fail option is "No comparison justification mentions preference"
    and whose non-fail is "At least one comparison justification does not mention
    preference, but at least one does". Failing on any single silent comparison
    -- the reading this had first -- fails tasks the customer's QC passes, and on
    the narrowest check in the spec that is the wrong direction to err in.
    """
    findings = (
        finding
        if isinstance(finding, list)
        else ([] if finding is None else [finding])
    )
    if not findings:
        return build_verdict(
            task_id=task.task_id,
            check_id=470,
            band="not_evaluated",
            measurement=Measurement(notes="no comparison justification was audited"),
            confidence="low",
            policy=policy,
        )

    silent = [f for f in findings if not f.states_preference]
    counts: dict[str, object] = {}
    for f in findings:
        suffix = f"_{f.pair}" if len(findings) > 1 else ""
        counts[f"states_preference{suffix}"] = f.states_preference
    if silent and len(findings) > 1:
        counts["pairs_without_a_stated_preference"] = ", ".join(f.pair for f in silent)
    if len(silent) == len(findings):
        band: Band = "fail"
    elif silent:
        band = "non_fail"
    else:
        band = "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=470,
        band=band,
        measurement=Measurement(
            counts=counts,
            notes="Scope is the model comparison justification only; a reasoned "
            "tie counts as a stated preference.",
        ),
        contributing_items=[f.quote for f in findings if f.quote],
        confidence=min((f.confidence for f in findings), key=_CONFIDENCE_ORDER.get),
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
    pair: tuple[ModelSlot, ModelSlot] = ("A", "B"),
) -> Direction:
    """Which model the contributor preferred, named by slot.

    `pair` says which two slots the scale runs between; the low end is its left
    and the high end its right. Defaults to A-vs-B, which is every
    single-comparison task.

    The recorded winner wins over the spec's buckets whenever it is present,
    because the buckets are demonstrably wrong about this product. Across 73 real
    tasks the tool records Likert 1-3 as "the left model wins" and 5-7 as "the
    right one wins", never emitting 4 at all; the spec calls 3, 4 and 5 neutral.
    Trusting the buckets silences check 460 on the 27 of 73 tasks sitting at 3 or
    5 -- a third of the graded set -- because a "neutral" preference can never
    contradict anything.
    """
    left, right = pair
    if winner_index is not None:
        return left if winner_index == 0 else right
    if likert is None:
        return "neutral"
    if policy.preference_encoding == "bipolar_7":
        if likert in policy.likert_favors_a:
            return left
        if likert in policy.likert_favors_b:
            return right
        return "neutral"
    return "neutral"


def dimension_direction(
    findings_or_task,
    policy: Policy = DEFAULT_POLICY,
    pair: tuple[ModelSlot, ModelSlot] = ("A", "B"),
) -> Direction:
    """Direction implied by the contributor's own per-dimension ratings.

    Only the two slots named by `pair` are considered. On a three-model task the
    dimension ratings cover all three, and folding a slot that is not in this
    comparison into the comparison's direction would have C's scores decide
    whether A-vs-B contradicts itself.

    Under `dominance` a side counts as favoured only when it wins at least one
    dimension and loses none. A mixed profile -- each model ahead somewhere --
    genuinely supports either verdict, so the Likert cannot contradict it and
    the check stays silent. Averaging instead turns a split decision into a
    direction, and a fraction of a point of separation is then enough to accuse
    a contributor of contradicting themselves.
    """
    left, right = pair
    ratings = getattr(findings_or_task, "dimension_ratings", findings_or_task)
    usable = [
        r
        for r in ratings
        if not r.not_applicable and r.rating is not None and r.model in (left, right)
    ]
    a = [r.rating for r in usable if r.model == left]
    b = [r.rating for r in usable if r.model == right]
    if not a or not b:
        return "neutral"

    if policy.ranking_direction_basis == "dominance":
        by_dimension: dict[str, dict[str, float]] = {}
        for r in usable:
            by_dimension.setdefault(r.dimension, {})[r.model] = r.rating
        a_wins = b_wins = 0
        for scores in by_dimension.values():
            if left not in scores or right not in scores:
                continue
            if scores[left] > scores[right]:
                a_wins += 1
            elif scores[right] > scores[left]:
                b_wins += 1
        if a_wins and not b_wins:
            return left
        if b_wins and not a_wins:
            return right
        return "neutral"

    ma, mb = mean(a), mean(b)
    if ma > mb:
        return left
    if mb > ma:
        return right
    return "neutral"


# Ordered worst-first, so `min` over several findings' confidences yields the
# least confident of them -- a verdict summarising two comparisons is only as
# trustworthy as its weaker half.
_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def evaluate_check_460(
    task: Task,
    finding: InversionFinding | list[InversionFinding],
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """One verdict covering every comparison the task recorded.

    Accepts a single finding or a list of them, one per pair. A task with two
    comparisons has two chances to contradict itself and fails on either, the
    same way minimum turns fails if any one model is short -- so the verdict is
    the worst pair's, with each pair's directions reported so the reader can see
    which one it was.

    Fail only when the directions are opposite AND the justification does not
    adequately support the inversion. A contributor may rationally prefer the
    model with lower dimension scores if one severe failure outweighs several
    small wins, and saying so makes the ranking consistent.
    """
    findings = finding if isinstance(finding, list) else [finding]
    if not findings:
        return build_verdict(
            task_id=task.task_id,
            check_id=460,
            band="not_evaluated",
            measurement=Measurement(notes="No comparison was recorded."),
            policy=policy,
        )

    failing = [
        f for f in findings if f.contradicts and not f.justification_explains_inversion
    ]
    band: Band = "fail" if failing else "clean"
    counts: dict[str, object] = {"direction_basis": policy.ranking_direction_basis}
    for f in findings:
        # Suffixed per pair rather than nested so the counts stay a flat mapping,
        # which is what every consumer of `Measurement.counts` expects.
        suffix = f"_{f.pair}" if len(findings) > 1 else ""
        counts[f"dimension_direction{suffix}"] = f.dimension_direction
        counts[f"likert_direction{suffix}"] = f.likert_direction
        counts[f"contradicts{suffix}"] = f.contradicts
        counts[f"justification_explains_inversion{suffix}"] = (
            f.justification_explains_inversion
        )
    if failing and len(findings) > 1:
        counts["failing_pairs"] = ", ".join(f.pair for f in failing)
    return build_verdict(
        task_id=task.task_id,
        check_id=460,
        band=band,
        measurement=Measurement(
            counts=counts,
            notes="Only opposite directions qualify; a neutral Likert with a mild "
            "dimension lean is not a contradiction.",
        ),
        confidence=min((f.confidence for f in findings), key=_CONFIDENCE_ORDER.get),
        policy=policy,
    )


def build_inversion_finding(
    task: Task,
    justification_explains_inversion: bool = False,
    policy: Policy = DEFAULT_POLICY,
    comparison: Sxs | None = None,
) -> InversionFinding:
    """The inversion finding for one comparison, defaulting to the primary one."""
    pair = comparison or task.comparison_list()[0]
    return InversionFinding(
        dimension_direction=dimension_direction(task, policy, pair.pair),
        likert_direction=likert_direction(
            pair.likert, policy, winner_index=pair.winner_index, pair=pair.pair
        ),
        justification_explains_inversion=justification_explains_inversion,
        pair=pair.pair_label,
    )


def build_inversion_findings(
    task: Task,
    justification_explains_inversion: bool = False,
    policy: Policy = DEFAULT_POLICY,
) -> list[InversionFinding]:
    """One inversion finding per comparison the task recorded."""
    return [
        build_inversion_finding(task, justification_explains_inversion, policy, pair)
        for pair in task.comparison_list()
    ]


# ---------------------------------------------------------------------------
# 100 -- key turn identification
# ---------------------------------------------------------------------------


def _key_turn_worst_of_slots(
    task: Task,
    per_slot: dict[ModelSlot, KeyTurn],
    auditor_key_turn: int | dict[ModelSlot, int | None] | None,
    policy: Policy,
) -> CheckVerdict:
    """One verdict over several models' key turn selections, taking the worst.

    Worst-of rather than a count, because the spec's bands are themselves
    worst-of: turn 1 on *any* model fails, misidentification on any is the middle
    band, and clean requires all of them right. What each slot contributed is
    published in the counts so a reader can see which model was the problem
    without re-deriving it.
    """
    auditor = (
        auditor_key_turn
        if isinstance(auditor_key_turn, dict)
        else {"A": auditor_key_turn}
    )

    counts: dict[str, object] = {}
    over_flagged: list[str] = []
    turn_one: list[str] = []
    misidentified: list[str] = []
    unjudged: list[str] = []

    for slot in sorted(per_slot):
        kt = per_slot[slot]
        counts[f"contributor_key_turn_{slot}"] = (
            ", ".join(str(t) for t in kt.flagged_turns)
            if kt.over_flagged
            else kt.turn_index
        )
        if kt.over_flagged:
            over_flagged.append(f"{slot}:{len(kt.flagged_turns)} turns flagged")
            continue
        if kt.turn_index == 1:
            turn_one.append(slot)
            continue
        theirs = auditor.get(slot)
        if theirs is None:
            unjudged.append(slot)
            continue
        counts[f"auditor_key_turn_{slot}"] = theirs
        if theirs != kt.turn_index:
            misidentified.append(f"{slot}:{kt.turn_index} vs {theirs}")

    notes: list[str] = []
    if over_flagged:
        notes.append(
            "several turns flagged as the key turn (" + "; ".join(over_flagged) + ")"
        )
    if turn_one:
        notes.append("key turn is turn 1 on " + ", ".join(turn_one))
    if misidentified:
        notes.append("misidentified on " + "; ".join(misidentified))
    if unjudged:
        notes.append(
            "no independent identification yet for " + ", ".join(unjudged)
        )

    if over_flagged or turn_one:
        band: Band = "fail"
    elif misidentified:
        band = "non_fail"
    elif unjudged:
        # Nothing convicting found, but not every model was actually looked at.
        band = "not_evaluated"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=100,
        band=band,
        measurement=Measurement(counts=counts, notes="; ".join(notes)),
        contributing_items=[f"key_turn_{s}" for s in turn_one]
        + [f"key_turn_{o.split(':')[0]}" for o in over_flagged],
        policy=policy,
    )


def evaluate_check_100(
    task: Task,
    auditor_key_turn: int | dict[ModelSlot, int | None] | None = None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    """Three-step, and the order matters.

    Two structural defects need no judgment and so are decided first and
    short-circuit: naming several key turns, and naming turn 1. Only a selection
    that is a single turn other than 1 is compared against the auditor's
    independent identification.

    On a project that records a key turn per model, every competing model's
    selection is graded and the task takes the worst of them. The V5 spec's fail
    option is "the Key Turn is marked as turn 1 for any of Model A, B, and C"
    and its clean band is "correctly identified in all 3", so reading only the
    model under test's would let a turn-1 selection on B or C through. The
    simplified trajectory is excluded, which the spec states outright: it has no
    key turn.

    `auditor_key_turn` therefore takes a per-slot mapping as well as a bare int.
    Both structural defects are decided without it, so the fail band is fully
    graded on every slot either way; a bare int is compared against the model
    under test only, and slots with no auditor answer are left unjudged rather
    than assumed correct.
    """
    per_slot = {
        slot: kt for slot, kt in task.key_turns().items() if slot != "S"
    }
    if len(per_slot) > 1:
        return _key_turn_worst_of_slots(task, per_slot, auditor_key_turn, policy)

    if isinstance(auditor_key_turn, dict):
        auditor_key_turn = auditor_key_turn.get("A")
    cb_turn = task.key_turn.turn_index

    # A key turn is a single freeze point. Flagging several is the same defect as
    # flagging none -- there is no turn to freeze at -- and it is not a thin-
    # evidence abstention, because the contributor did answer, just not with an
    # answer the field can hold. Seen on 3 of 73 aspirational tasks, one of them
    # flagging 14 of its 18 turns.
    if task.key_turn.over_flagged:
        flagged = task.key_turn.flagged_turns
        return build_verdict(
            task_id=task.task_id,
            check_id=100,
            band="fail",
            measurement=Measurement(
                counts={"key_turns_flagged": len(flagged)},
                notes=(
                    f"{len(flagged)} turns are flagged as the key turn "
                    f"({', '.join(str(t) for t in flagged)}); exactly one is required."
                ),
            ),
            policy=policy,
        )

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

    Kept as a module-level name because callers here and in `build_context`
    import it, but the logic lives on `Task` -- two copies drifted once already
    over which key turn counts for which slot, which is exactly the kind of
    disagreement that manufactures a "cited turn is past the end of the
    conversation" fail out of nothing.
    """
    return task.cited_turns(model)


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

    The count is the best contributor-side record, not whichever source wrote
    `conversation` last. `sub.conversation` starts as the contributor's own
    per-turn manifest and hydration overwrites it unconditionally on any fetch
    that parsed at least one user turn, so a submission whose manifest records 22
    turns and whose share page renders 3 was being failed on 3. The overwrite is
    right for the judges -- a short fetch carries the model's actual replies and
    the manifest carries only the user's side -- so the fix belongs here, in the
    one check that gates on the length rather than on the content. Taking the
    maximum cannot manufacture a fail: it can only raise a count, and this check
    only ever fails a submission for being low. The reverse direction is on
    record too (`6a800beb4c1c25ccc73e4a91` slot A has a manifest of 1 and
    hydrated to 21), which is exactly why neither record can be trusted alone.

    Zero cases in the 80 submissions scanned so far have a manifest at or above
    the minimum against a hydrated count below it, so this is latent rather than
    live -- but the two records are on record disagreeing by twenty turns, and
    the direction this guards is the one that costs a contributor a task.
    """
    minimum = policy.min_turns_per_model
    strict = policy.turn_count_requires_complete_conversation

    def best_record(sub) -> int:
        """The longest length the contributor's own side of the record supports."""
        return max(sub.exchange_count(), len(sub.turn_manifest_indices()))

    # The best fetch in the task is the only corroboration available for a
    # citation past the end of the worst one. Deliberately still the *fetched*
    # length rather than `best_record`: the ceiling exists to bound a citation
    # against something the audit retrieved itself, and raising it with a
    # contributor-authored manifest would let the same document both make a
    # citation and corroborate it.
    ceiling = max((sub.exchange_count() for sub in task.submissions()), default=0)

    counts: dict[str, int] = {}
    fetched_counts: dict[str, int] = {}
    manifest_counts: dict[str, int] = {}
    skipped: dict[str, str] = {}
    verifiable: dict[str, int] = {}

    for sub in task.submissions():
        count = best_record(sub)
        counts[sub.model] = count
        fetched_counts[sub.model] = sub.exchange_count()
        manifest_counts[sub.model] = len(sub.turn_manifest_indices())
        if not sub.conversation and not sub.turn_manifest:
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
        "rather than the contributor's work. The count compared against the "
        "minimum is the greater of what the fetch retrieved and what the "
        "contributor's own per-turn manifest records, so which of the two "
        "happened to write the conversation last cannot decide the verdict."
    )
    outranked = sorted(
        m for m in counts if manifest_counts[m] > fetched_counts[m]
    )
    if outranked:
        notes += (
            " The manifest is the longer record for "
            + ", ".join(
                f"model {m} ({manifest_counts[m]} vs {fetched_counts[m]} fetched)"
                for m in outranked
            )
            + "."
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
                # The two records behind the count above, so a reader can see
                # which one carried it. They disagree by twenty turns on one
                # submission in the live data.
                "fetched_turns_by_model": dict(sorted(fetched_counts.items())),
                "manifest_turns_by_model": dict(sorted(manifest_counts.items())),
                "counted_from_manifest": outranked,
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


def _is_gemini_slot(sub) -> bool:
    from .links import classify_link

    if classify_link(sub.final_link).provider == "gemini":
        return True
    return "gemini" in (sub.declared_provider or "").lower()


def html_export_without_share_manifest(sub) -> bool:
    """Spark HTML/MHTML chat export with no per-turn Gemini share URLs.

    Unique Links (91) judges Gemini share-link uniqueness. An HTML dump has
    no share-URL manifest to judge, so 91 is N/A on that slot. A slot that
    also filed valid Gemini share links is still in scope — this is not a
    blanket ignore on share-link B.
    """
    from .links import classify_link
    from .sources import looks_like_html_chat_export

    final = classify_link(getattr(sub, "final_link", "") or "")
    if final.valid and final.provider == "gemini":
        return False
    for entry in getattr(sub, "turn_manifest", None) or []:
        check = classify_link(getattr(entry, "share_link", "") or "")
        if check.valid and check.provider == "gemini":
            return False

    export = (getattr(sub, "transcript_pdf", "") or "").strip()
    if not export:
        return False
    names = getattr(sub, "attachment_names", None) or {}
    mimes = getattr(sub, "attachment_mime_types", None) or {}
    export_name = str(names.get(export) or "")
    export_mime = str(mimes.get(export) or "").lower()
    if looks_like_html_chat_export(export) or looks_like_html_chat_export(export_name):
        return True
    if "html" in export_mime or "multipart/related" in export_mime:
        return True
    # A named PDF is not an HTML/MHTML dump. Unique Links still applies.
    export_path = export.split("?")[0].lower()
    if export_path.endswith(".pdf") or export_name.lower().endswith(".pdf"):
        return False
    if "pdf" in export_mime:
        return False
    # Scale CDS chat-download URLs have no .html/.mhtml suffix. A download
    # with no Gemini share-URL manifest is still Unique Links N/A.
    return export.startswith(("http://", "https://"))


def evaluate_check_91(
    task: Task, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Unique Links. V5 fails at "4+ turns for Gemini that do not contain a
    unique link" and non-fails at "1-3".

    Source is the criteria-builder turn manifest, not `turn_links`. An empty
    manifest is `not_evaluated` so a fixture that reuses one Gemini URL for
    every turn (the default `make_task`) is not convicted of a uniqueness
    defect it never recorded. Counted across every Gemini slot, including S.
    HTML/MHTML chat exports with no share-URL manifest are N/A, not a fail.
    """
    from .links import classify_link

    offenders: list[str] = []
    gemini_with_manifest = 0
    html_export_slots = 0
    for sub in task.trajectories():
        if not _is_gemini_slot(sub):
            continue
        if html_export_without_share_manifest(sub):
            html_export_slots += 1
            continue
        if not sub.turn_manifest:
            continue
        gemini_with_manifest += 1
        checks = [
            (entry.turn_number, classify_link(entry.share_link))
            for entry in sub.turn_manifest
        ]
        share_counts = Counter(
            check.share_id for _, check in checks if check.valid and check.share_id
        )
        for turn, check in checks:
            if not check.valid:
                offenders.append(
                    f"{sub.model}:turn {turn}:{check.reason or 'missing'}"
                )
            elif share_counts[check.share_id] > 1:
                offenders.append(
                    f"{sub.model}:turn {turn}:duplicate {check.share_id}"
                )

    if not gemini_with_manifest:
        notes = (
            "no Gemini turn manifest was recorded, so uniqueness "
            "could not be judged"
        )
        if html_export_slots:
            notes = (
                "Gemini slot(s) are HTML/MHTML chat exports with no per-turn "
                "share-URL manifest; Unique Links is N/A"
            )
        return build_verdict(
            task_id=task.task_id,
            check_id=91,
            band="not_evaluated",
            measurement=Measurement(
                notes=notes,
                counts={"html_export_slots_without_share_manifest": html_export_slots},
            ),
            confidence="low",
            policy=policy,
        )

    n = len(offenders)
    if n >= policy.unique_links_fail_count:
        band: Band = "fail"
    elif n > 0:
        band = "non_fail"
    else:
        band = "clean"

    return build_verdict(
        task_id=task.task_id,
        check_id=91,
        band=band,
        measurement=Measurement(
            numerator=n,
            threshold=policy.unique_links_fail_count,
            counts={
                "gemini_slots_with_manifest": gemini_with_manifest,
                "non_unique_or_missing": n,
            },
            notes='V5: fail at "4+ turns for Gemini that do not contain a unique '
            'link"; non-fail at "1-3".',
        ),
        contributing_items=offenders,
        policy=policy,
    )


def evaluate_check_97(
    task: Task, policy: Policy = DEFAULT_POLICY
) -> CheckVerdict:
    """Valid Model Failure / Simplified Difficulty. Hard fail.

    The simplified trajectory must fail: weighted pass rate must not exceed
    the project's ceiling (50% on Aspirational, customer-confirmed against
    V5's 40% text), and its turn count must sit in the project's bounds (1-3).
    """
    if task.simplified is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=97,
            band="not_evaluated",
            measurement=Measurement(notes="no simplified trajectory was filed"),
            confidence="low",
            policy=policy,
        )

    s_ratings = [r for r in task.criterion_ratings if r.model == "S"]
    if not s_ratings:
        return build_verdict(
            task_id=task.task_id,
            check_id=97,
            band="not_evaluated",
            measurement=Measurement(
                notes="simplified trajectory has no criterion ratings"
            ),
            confidence="low",
            policy=policy,
        )

    profile = profile_for_project(task.project_id)
    ceiling = (
        profile.simplified_max_pass_rate
        if profile and profile.simplified_max_pass_rate is not None
        else 0.50
    )
    bounds = (
        profile.simplified_turn_bounds
        if profile and profile.simplified_turn_bounds is not None
        else (1, 3)
    )

    weights = {c.criterion_id: c.weight for c in task.rubric}
    total = sum(weights.get(r.criterion_id, 0) for r in s_ratings)
    passed = sum(
        weights.get(r.criterion_id, 0) for r in s_ratings if r.score == 1
    )
    rate = (passed / total) if total else 0.0

    sub = task.simplified
    turns = sub.exchange_count() or len(sub.turn_manifest)
    lo, hi = bounds
    over_ceiling = rate > ceiling
    out_of_bounds = turns < lo or turns > hi

    if over_ceiling or out_of_bounds:
        band = "fail"
    else:
        band = "clean"

    reasons = []
    if over_ceiling:
        reasons.append(
            f"weighted pass rate {rate:.1%} exceeds the {ceiling:.0%} ceiling"
        )
    if out_of_bounds:
        reasons.append(f"{turns} turns is outside the {lo}-{hi} bound")

    return build_verdict(
        task_id=task.task_id,
        check_id=97,
        band=band,
        measurement=Measurement(
            numerator=passed,
            denominator=total or None,
            rate=rate,
            threshold=ceiling,
            counts={
                "simplified_turns": turns,
                "turn_bounds": list(bounds),
                "over_ceiling": over_ceiling,
                "out_of_bounds": out_of_bounds,
            },
            notes="; ".join(reasons)
            or f"weighted pass rate {rate:.1%} at or under {ceiling:.0%}, "
            f"{turns} turns in {lo}-{hi}",
        ),
        contributing_items=reasons,
        policy=policy,
    )
