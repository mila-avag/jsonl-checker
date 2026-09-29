"""Per-item findings produced by the model stages and consumed by the gates.

Models find issues; Python counts them. Every threshold in the spec is a count or
a percentage, and none of them is ever produced by a language model. Keeping the
findings as plain data makes every gate a pure function that tests without any
model call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .models import ModelSlot
from .taxonomies import severity_of


@dataclass
class Issue:
    """One defect found in one rubric criterion. An issue without a verbatim quote
    is not an issue."""

    category: str
    description: str
    evidence: str
    # What the audit was not shown and would have needed in order to decide.
    # Set only on an issue routed to `CriterionFinding.unverifiable_issues`.
    why_unverifiable: str = ""
    confidence: str = "high"
    # `overlapping` only: the sibling criterion id(s) this one fully duplicates.
    # Lets the census collapse a mutually-reported overlap into the one error the
    # spec counts instead of one per criterion in the set.
    overlaps_with: list[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        return severity_of(self.category)


@dataclass
class CriterionFinding:
    """Result of the per-criterion audit for one rubric criterion."""

    criterion_id: str
    issues: list[Issue] = field(default_factory=list)
    # Issues the auditor could only have confirmed against context it was not
    # given: a prompt turn the context block truncated, or the contents of an
    # input file the audit holds only the name of. `inaccurate`,
    # `counterproductive` and `not_self_contained` are all judged against that
    # material, so an auditor reading a truncated prompt can find every one of
    # them in a criterion that is in fact correct. Reported, never counted --
    # `severities()` reads `issues` alone, so the census numerator never sees
    # these and no gate had to change to exclude them.
    unverifiable_issues: list[Issue] = field(default_factory=list)
    confidence: str = "high"

    def severities(self) -> set[str]:
        return {i.severity for i in self.issues}

    def has_at_least(self, *severities: str) -> bool:
        return bool(self.severities() & set(severities))


@dataclass
class CoverageFinding:
    """Task-level pass for criteria that do not exist.

    Missing criteria raise the numerator of the relevant gates without raising the
    denominator, which is what makes coverage failures bite.
    """

    missing_critical: list[str] = field(default_factory=list)
    missing_non_critical: list[str] = field(default_factory=list)


@dataclass
class L1Finding:
    criterion_id: str
    contributor_label: str
    auditor_label: str
    confidence: str = "high"

    @property
    def incorrect(self) -> bool:
        return self.contributor_label != self.auditor_label


@dataclass
class L2Finding:
    """Check 210. Same shape as L1, but a leaf parented under the wrong L1 counts
    as incorrect even when the leaf name itself is one the contributor could pick."""

    criterion_id: str
    contributor_label: str
    auditor_label: str
    confidence: str = "high"

    @property
    def incorrect(self) -> bool:
        return self.contributor_label != self.auditor_label


@dataclass
class WeightFinding:
    """Check 260. A weight is a judgement call, so the auditor returns the range the
    customer's weight definitions can support and only a weight outside that range
    is a defect.

    Comparing point to point measured the wrong thing: the auditor's own taste sits
    a level below the contributors' on this scale, so a contributor who picked a
    perfectly defensible weight was recorded as wrong whenever the model would have
    picked its neighbour. `defensible_low`/`defensible_high` are the gate's input;
    `auditor_weight` is the auditor's own pick, kept for the report so a reader can
    see what it would have chosen, and never counted on its own.
    """

    criterion_id: str
    contributor_weight: int
    auditor_weight: int
    defensible_low: int | None = None
    defensible_high: int | None = None
    confidence: str = "high"

    @property
    def band(self) -> tuple[int, int]:
        """The defensible range, widened to include the auditor's own pick.

        A response naming a preferred weight outside the range it just called
        defensible has contradicted itself, and the union is the reading that
        cannot invent a defect. With no range recorded at all the band collapses to
        the point, which is the old behaviour and is why the parser drops a
        response that omits it rather than gating on one.
        """
        if self.defensible_low is None or self.defensible_high is None:
            return (self.auditor_weight, self.auditor_weight)
        values = (self.defensible_low, self.defensible_high, self.auditor_weight)
        return (min(values), max(values))

    @property
    def in_band(self) -> bool:
        low, high = self.band
        return low <= self.contributor_weight <= high

    @property
    def nearest_defensible(self) -> int:
        """The end of the band closest to the recorded weight."""
        low, high = self.band
        if self.contributor_weight < low:
            return low
        if self.contributor_weight > high:
            return high
        return self.contributor_weight

    @property
    def points_outside(self) -> int:
        return abs(self.contributor_weight - self.nearest_defensible)

    @property
    def raw_delta(self) -> int:
        """Signed gap to the auditor's own pick. Reported, never gated on."""
        return self.auditor_weight - self.contributor_weight


@dataclass
class CriterionRatingFinding:
    """One (criterion, model) judgment for check 270."""

    criterion_id: str
    model: ModelSlot
    contributor_score: int
    auditor_score: int
    confidence: str = "high"
    # Whether the conversation this judgment read was cut before its final turn.
    # Reported rather than acted on for 270, whose unit is a pass/fail against one
    # stated requirement and whose judge abstains on a requirement it cannot see;
    # 300's numeric delta is the one this build stopped counting on truncation.
    truncated: bool = False

    @property
    def disagrees(self) -> bool:
        return self.contributor_score != self.auditor_score

    @property
    def item_id(self) -> str:
        return f"{self.model}::{self.criterion_id}"


@dataclass
class RelevantTurnFinding:
    """Turn-citation judgments for checks 280 and 310.

    Namespaced by check_id so an incorrect turn is never counted toward both.
    """

    check_id: Literal[280, 310]
    item_id: str
    model: ModelSlot
    incorrect_turns: list[int] = field(default_factory=list)
    missing_turn: bool = False
    # Cited turns past the end of the conversation the audit could fetch. The
    # citation may well be correct against the full conversation the contributor
    # saw, so these are reported and never counted -- a truncated fetch is our
    # failure, not theirs.
    unverifiable_turns: list[int] = field(default_factory=list)
    # What the judge says it was missing, in its own words. Two things reach
    # `unverifiable_turns`: a citation past the end of the fetched conversation,
    # which Python detects, and a citation into a span the render budget cut or
    # into a deliverable whose content the transcript does not hold, which only
    # the judge can see. The second is why the judge is asked to name it.
    why_unverifiable: str = ""
    confidence: str = "high"


@dataclass
class DimensionRatingFinding:
    """One (model, dimension) judgment for check 300.

    Two fields decide whether a disagreement here is publishable as a defect, and
    both exist because the blind delta on its own was not.

    `truncated` records that the conversation this judgment was made from was cut
    before its last turn. `render_conversation` has always known this and the
    prompt builders threw it away, so a judgment made from half a conversation
    arrived at the gate indistinguishable from one made from all of it.

    `adjudication` carries the second, informed pass's answer to the only question
    that makes a delta actionable: not "would I have rated it the same" but "is
    the contributor's rating defensible against this conversation". `defensible`
    means the disagreement is between two readings a reviewer could hold, which is
    not a defect; `indefensible` means the transcript does not support the rating
    the contributor gave. `None` means no adjudication was run -- either the
    dimension carried no disagreement to adjudicate, or the policy flag is off.
    """

    model: ModelSlot
    dimension: str
    contributor_rating: int | None = None
    auditor_rating: int | None = None
    contributor_na: bool = False
    auditor_na: bool = False
    confidence: str = "high"
    # Whether whole turns were missing from the conversation this blind judgment
    # read. A turn whose middle was elided is not this: the turn was shown, the cut
    # was labelled in place, and treating that as evidence loss is what made three
    # tasks report clean on conversations the pipeline had rendered end to end.
    truncated: bool = False
    adjudication: Literal["defensible", "indefensible"] | None = None
    adjudication_reasoning: str = ""
    # The same question about the *adjudication's* render, which runs at a far
    # wider budget. This is the one the gate acts on, because the adjudication
    # verdict is what gets counted: a blind rating made from a cut conversation is
    # still checkable, provided the reader that checked it saw the whole thing.
    adjudication_tail_truncated: bool = False
    # What the informed pass would have to have been shown to decide, set only
    # when it declined to. An adjudication that abstains leaves the disagreement
    # unconfirmed, which under the default policy means uncounted.
    adjudication_abstained: bool = False
    why_unadjudicated: str = ""

    @property
    def counts_as_disagreement(self) -> bool:
        """Whether this disagreement is publishable as the contributor's defect.

        Read by the gate, which owns the policy flags; this property answers only
        the part that is a property of the finding: an adjudication that ran and
        came back `defensible` (or abstained) is not a defect, whatever the blind
        delta was.
        """
        if self.adjudication is None:
            return not self.adjudication_abstained
        return self.adjudication == "indefensible"

    @property
    def evidence_incomplete(self) -> bool:
        """Whether the judgment that decides this item was made from a cut render.

        Which render that is depends on which judgment decides the item. Once an
        adjudication has run, its verdict is what the gate counts, so its render is
        the one that has to have been complete -- the blind pass's cut is recorded
        and reported, but it is no longer what the count rests on.
        """
        if self.adjudication is not None or self.adjudication_abstained:
            return self.adjudication_tail_truncated
        return self.truncated

    def __post_init__(self) -> None:
        # N/A and a rating are mutually exclusive: a dimension that does not apply
        # carries no rating at all. Upstream data sometimes arrives with both, and a
        # surviving number is worse than no number because every delta and mean
        # downstream will consume it silently.
        if self.contributor_na:
            self.contributor_rating = None
        if self.auditor_na:
            self.auditor_rating = None

    @property
    def item_id(self) -> str:
        return f"{self.model}::{self.dimension}"

    @property
    def comparable(self) -> bool:
        """Whether this dimension belongs in a rating-comparison denominator.

        Either side calling it N/A takes it out of the rating comparison: what is
        left is an applicability disagreement, which check 300 counts separately.
        """
        return not self.contributor_na and not self.auditor_na

    @property
    def na_mismatch(self) -> bool:
        return self.contributor_na != self.auditor_na

    @property
    def both_na(self) -> bool:
        return self.contributor_na and self.auditor_na

    @property
    def delta(self) -> int | None:
        if self.contributor_rating is None or self.auditor_rating is None:
            return None
        return abs(self.auditor_rating - self.contributor_rating)


# How many wrong subordinate claims in one justification trip `is_inaccurate`.
#
# THIS NUMBER IS OURS, NOT THE SPEC'S, and it is the one floor in
# `triggered_conditions` that departs from the auditor instructions. The spec
# says two, on the stated equivalence that two subordinate errors are worth one
# load-bearing one, and this build ran at two until a QC comparison run argued
# otherwise: over 14 human-labelled tasks, secondary inaccuracies were the modal
# defect the judge reported and the modal defect a human failed a task for, and
# pairing them inside a single justification is what the two-floor requires.
# Nothing about the defect gets less serious for arriving alone.
#
# Set back to 2 to restore the spec's reading. `COUNTED_CONDITION_FLOORS` reads
# this, so the cross-justification aggregation follows automatically: at 1 there
# is no sub-floor residue left to aggregate and that branch goes inert on its
# own, which is correct rather than a gap.
#
# What this does not change is who decides. The judge is told to bucket by how
# much the rating leans on a claim and is deliberately not told what either
# bucket costs, so this floor moves the consequence without moving the counting.
SECONDARY_INACCURACY_FLOOR = 1


@dataclass
class JustificationFinding:
    """Per-condition counts for one justification, for check 450.

    Thresholds are per-condition and counted within a single justification, so the
    counts stay attached to their justification rather than being pooled.
    """

    item_id: str
    rated_value: int | None = None
    contradicts_verdict_claims: int = 0
    is_generic: bool = False
    is_skewed: bool = False
    inaccurate_primary_claims: int = 0
    inaccurate_secondary_claims: int = 0
    unsupported_claims: int = 0
    inaccurate_evidence: int = 0
    misconstrued_evidence: int = 0
    # Verbatim spans of the justification that name something only this
    # conversation contains. Generic is the claim that there is nothing here to
    # quote, so a single span refutes it and the model's boolean is overridden.
    specifics_quoted: list[str] = field(default_factory=list)
    # Verbatim spans alleged to argue against the rating they defend. The spec's
    # condition is about a *supporting claim* that fails to support, so the claim
    # has to be exhibited before it can be counted; an unquoted count is the
    # auditor's impression of the argument as a whole, which is a different and
    # much broader complaint than the one 450 enumerates.
    contradicting_quotes: list[str] = field(default_factory=list)
    # Claims the audit could not check because the turn or artifact they name
    # never reached the judge's context, not because the contributor left them
    # bare. Reported, never counted: a truncated fetch is our failure, not theirs.
    unverifiable_claims: int = 0
    # The whole justification defends a rating of something this transcript does
    # not contain -- the deliverable itself, or a span the render budget cut --
    # so none of the counts above can be formed honestly. Set here, the parser
    # zeroes every count and the stage keeps the finding out of the list the gate
    # scores, so the justification leaves 450's denominator as well as its
    # numerator. Counting it clean would be just as wrong: it would report a
    # justification as audited that nobody could audit.
    unverifiable: bool = False
    why_unverifiable: str = ""
    confidence: str = "high"
    # How many independent draws named each condition on this justification, and
    # how many draws were combined to reach the counts above.
    #
    # 450's severe route -- `policy.justification_severe_conditions`, where one
    # justification fails the whole task at any share -- reads a merged finding
    # and so has no way to ask how much of the judge was behind it. That was
    # invisible rather than absent: the aggregator's own threshold bounds it, so
    # the escape has always needed a majority of the draws. It bounds it only
    # where there were draws to combine. `aggregate_justifications` returns a
    # lone draw verbatim when a group lost the rest, and on the sheet22 corpus
    # two tasks failed on a severe condition exactly one stored draw ever named.
    #
    # `draws_combined` of 0 means nothing sampled this finding -- a synthesised
    # empty-justification finding, a single-sample run, a replay building
    # findings by hand -- and is the signal for the gate to leave the severe
    # route alone rather than deny an escape for want of a record.
    condition_draws: dict[str, int] = field(default_factory=dict)
    draws_combined: int = 0

    def condition_support(self, condition: str) -> int | None:
        """Draws that named `condition`, or None where no draw record was kept.

        None and 0 are different answers and the gate treats them differently:
        None is "this finding was never sampled, do not judge it on draws", 0 is
        "it was sampled and no draw named this", which for a condition the merged
        counts do trigger can only mean the condition came from somewhere other
        than a draw's own judgment.
        """
        if not self.draws_combined:
            return None
        return self.condition_draws.get(condition, 0)

    @property
    def generic(self) -> bool:
        return self.is_generic and not any(s.strip() for s in self.specifics_quoted)

    @property
    def contradicts_verdict(self) -> bool:
        return self.contradicts_verdict_claims >= 1 and any(
            q.strip() for q in self.contradicting_quotes
        )

    def triggered_conditions(self) -> list[str]:
        out: list[str] = []
        if self.contradicts_verdict:
            out.append("contradicts_verdict")
        if self.generic:
            out.append("is_generic")
        if self.is_skewed:
            out.append("is_skewed")
        if (
            self.inaccurate_primary_claims >= 1
            or self.inaccurate_secondary_claims >= SECONDARY_INACCURACY_FLOOR
        ):
            out.append("is_inaccurate")
        if self.unsupported_claims >= 2:
            out.append("lacks_evidence")
        if self.inaccurate_evidence >= 1:
            out.append("cites_incorrect_evidence")
        if self.misconstrued_evidence >= 2:
            out.append("misconstrues_evidence")
        return out

    def has_any_issue(self) -> bool:
        return any(
            (
                self.contradicts_verdict,
                self.generic,
                self.is_skewed,
                self.inaccurate_primary_claims,
                self.inaccurate_secondary_claims,
                self.unsupported_claims,
                self.inaccurate_evidence,
                self.misconstrued_evidence,
            )
        )


@dataclass
class JustificationCoverage:
    """How much of check 450's intended population actually reached a judgment.

    450 is the one check whose population is invisible in its own output: the
    denominator is however many findings came back, so a call that returned
    nothing for a model looks identical to a task with fewer dimensions. On the
    sheet22 run three tasks audited 8 justifications against the 15 their own
    `dimension_ratings` carry, all three reported zero failing justifications,
    and one of them (6a7d3a48a9485b6a031cb305) was a task human QC had failed
    for Bad Justifications -- replayed with model B's set present it fails at
    3/15 with three severe items.

    Counted by group, where a group is the unit of one model call: each model's
    batched dimension justifications, and the ranking justification on its own.
    A per-task ratio does not see the failure -- 8 of 15 is 53%, over
    `min_judged_ratio` -- because the loss is a whole side of the task rather
    than a thin scattering across it.

    `abstained` is held apart from `audited` deliberately. An abstention is a
    judgment ("the transcript does not contain what this defends") and is
    already excluded from the gate's numerator and denominator by design, but it
    is not evidence that the contributor's argument holds, so it cannot count
    toward coverage either. `missing` is the third thing: expected, neither
    judged nor abstained on, which is silent loss.
    """

    expected: dict[str, list[str]] = field(default_factory=dict)
    audited: dict[str, list[str]] = field(default_factory=dict)
    abstained: dict[str, list[str]] = field(default_factory=dict)

    @property
    def missing(self) -> list[str]:
        seen = {i for ids in self.audited.values() for i in ids} | {
            i for ids in self.abstained.values() for i in ids
        }
        return sorted(
            i for ids in self.expected.values() for i in ids if i not in seen
        )

    def undercovered(self, min_ratio: float) -> dict[str, float]:
        """Groups whose judged share falls below the floor, with the share.

        The ranking justification is one item and is left out: a group of one
        is either judged or not, and abstaining a whole check because a single
        justification of fifteen could not be read would trade a rare silent
        pass for a common silent gap.
        """
        out: dict[str, float] = {}
        for group, expected in self.expected.items():
            if group == "ranking" or not expected:
                continue
            judged = len(set(expected) & set(self.audited.get(group, [])))
            ratio = judged / len(expected)
            if ratio < min_ratio:
                out[group] = round(ratio, 3)
        return out

    def to_dict(self) -> dict:
        # Keyed on `expected` throughout, so a group nothing came back for
        # reports a zero rather than dropping out of the record. An absent key is
        # the shape of this bug, not a way to report it.
        return {
            "expected": {k: len(v) for k, v in sorted(self.expected.items())},
            "audited": {
                k: len(set(v) & set(self.audited.get(k, [])))
                for k, v in sorted(self.expected.items())
            },
            "abstained": {
                k: len(set(v) & set(self.abstained.get(k, [])))
                for k, v in sorted(self.expected.items())
            },
            "missing": self.missing,
        }


# The 450 conditions whose floors are counted, and the count each reads. A
# contributor who makes the same bad move once in each of five justifications
# trips none of these, because every floor is counted inside a single
# justification; see `aggregate_repeated_conditions`.
#
# An entry whose floor is 1 has nothing to aggregate -- one instance already
# trips it where it stands -- and drops out of that pass without needing to be
# removed from here. `is_inaccurate` is in that position today.
COUNTED_CONDITION_FLOORS: dict[str, tuple[str, int]] = {
    "is_inaccurate": ("inaccurate_secondary_claims", SECONDARY_INACCURACY_FLOOR),
    "lacks_evidence": ("unsupported_claims", 2),
    "misconstrues_evidence": ("misconstrued_evidence", 2),
}


@dataclass
class LikertFinding:
    contributor_likert: int
    auditor_likert: int
    confidence: str = "high"

    # Which comparison this is about, e.g. "AB" or "AC". A task may record more
    # than one, and they are separate judgments: the same contributor can be
    # right about A-vs-B and wrong about A-vs-C. Defaults to the single-
    # comparison case so existing callers and fixtures are unaffected.
    pair: str = "AB"

    # Populated only under the direction_magnitude encoding, where the winner is
    # a separate field and the Likert is a bare margin. Left None under the
    # spec's bipolar 1-7 form, on which the raw delta is already signed-correct
    # because the constant centre cancels.
    contributor_winner_index: int | None = None
    auditor_winner_index: int | None = None

    # Per-side render state, which for a comparison is a fairness property rather
    # than a footnote. The two conversations share one budget, so a long Model B
    # is cut sooner here than it would be on its own, and a ranking made across an
    # eight-exchange A and a one-exchange B is a ranking of the render.
    truncated_a: bool = False
    truncated_b: bool = False
    turns_shown_a: int = 0
    turns_shown_b: int = 0
    turns_total_a: int = 0
    turns_total_b: int = 0

    adjudication: Literal["defensible", "indefensible"] | None = None
    adjudication_reasoning: str = ""
    adjudication_abstained: bool = False
    why_unadjudicated: str = ""
    # Whether the adjudication's own, much wider render still could not show both
    # conversations in full. See `DimensionRatingFinding.evidence_incomplete`.
    adjudication_tail_truncated: bool = False

    @property
    def truncated(self) -> bool:
        return self.truncated_a or self.truncated_b

    @property
    def evidence_incomplete(self) -> bool:
        if self.adjudication is not None or self.adjudication_abstained:
            return self.adjudication_tail_truncated
        return self.truncated

    @property
    def asymmetric_render(self) -> bool:
        """Whether the two sides were shown to materially different depths.

        A ranking is a comparison, so what matters is not that a side was cut but
        that the sides were cut unequally: half of one conversation against all of
        the other is a comparison of the budget. The bar is half -- one side
        showing under 50% of the exchanges the other did.
        """
        shown = (self.turns_shown_a, self.turns_shown_b)
        if not all(shown):
            return any(shown)
        return min(shown) / max(shown) < 0.5

    @property
    def counts_as_disagreement(self) -> bool:
        if self.adjudication is None:
            return not self.adjudication_abstained
        return self.adjudication == "indefensible"

    @property
    def delta(self) -> int:
        if self.contributor_winner_index is None or self.auditor_winner_index is None:
            return abs(self.auditor_likert - self.contributor_likert)
        contributor = (
            -self.contributor_likert
            if self.contributor_winner_index == 0
            else self.contributor_likert
        )
        auditor = (
            -self.auditor_likert if self.auditor_winner_index == 0 else self.auditor_likert
        )
        return abs(auditor - contributor)


# Check 80's default scope. Lives here rather than as a literal in the gate so
# `Policy.target_outcome_defect_classifications` and `TargetOutcomeFinding.is_issue`
# cannot drift apart; the policy field's comment carries the reasoning.
DEFAULT_TARGET_OUTCOME_DEFECTS: tuple[str, ...] = ("contradicts_prompt",)


@dataclass
class TargetOutcomeFinding:
    """Check 80. One classified entry of the contributor's target outcome list.

    Tab 2 defines 80 entirely as a predicate on entries the contributor listed --
    "it's incorrect, lists outcomes which are not necessary or expected by the
    prompt(s), or contradicts one or more prompts" -- and the audit workflow
    tab's step 6 says to "Determine if the target outcomes listed are correct and
    align with the prompt(s)". Neither asks what the list leaves out, so the
    judge is never asked to derive a requirement the list omits: a target
    outcome list was never contracted to be an exhaustive index of a
    conversation, and the longer the conversation, the more asks a fixed-length
    list must fail to enumerate. On the first live run, counting omissions
    manufactured 113 defects across 17 tasks and denied every task a clean.

    Which of the remaining classifications is a defect is a policy call, not a
    property of the data, so it lives in
    `Policy.target_outcome_defect_classifications` and the gate applies it.
    `is_issue` answers only for the default scope, for the report summaries that
    have no policy in hand.
    """

    entry: str
    classification: Literal["supported", "not_required", "contradicts_prompt"]
    established_turn: int | None = None
    reasoning: str = ""
    confidence: str = "high"

    def counts_as_defect(self, defect_classifications: tuple[str, ...]) -> bool:
        return self.classification in defect_classifications

    @property
    def is_issue(self) -> bool:
        return self.counts_as_defect(DEFAULT_TARGET_OUTCOME_DEFECTS)


@dataclass
class DomainRelevanceFinding:
    """Check 70. How far the submitted prompt sits from the domain it was
    commissioned for.

    Three verdicts rather than a boolean, because the spec's fail wording is
    "egregious" misalignment: a prompt that leans away from its domain while
    still belonging to it is the non-fail band, not a fail. Collapsing the two
    would fail the majority of real prompts, which drift as a matter of course.

    Both quotes are load-bearing. `prompt_quote` must be verbatim from the
    submitted prompt and `domain_basis` verbatim from the assignment, so a
    misalignment claim names the text on each side that disagrees. An allegation
    that quotes neither is not evidence of anything and `is_evidenced` withholds
    it.
    """

    assessment: Literal["aligned", "partial", "unrelated"]
    assigned_domain: str = ""
    prompt_quote: str = ""
    domain_basis: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.prompt_quote.strip() and self.domain_basis.strip())

    @property
    def is_issue(self) -> bool:
        return self.assessment != "aligned" and self.is_evidenced

    @property
    def is_egregious(self) -> bool:
        return self.assessment == "unrelated" and self.is_evidenced


@dataclass
class PromptConsistencyFinding:
    """Check 75. Whether the contributor's submitted opening prompt is the same
    underlying request as the pre-seeded prompt the platform prepared for them.

    Three verdicts, the same shape as 70's `DomainRelevanceFinding` and for the
    same reason: the bar is the *underlying request*, not the wording, so a
    reworded opening that keeps the same core intent is the non-fail band, not a
    fail. Paraphrasing and added context (a persona, a reason for asking) are
    expected and must not read as a mismatch. Only a genuinely different subject
    reaches "different".

    Both quotes are load-bearing, the same discipline check 450 applies to
    `contradicts_verdict`: a verdict of anything but "same" must cite an actual
    quote from each side's core ask, not assert a match or a mismatch without
    one. `is_evidenced` withholds an allegation that quotes neither.
    """

    assessment: Literal["same", "reworded_same_intent", "different"]
    pre_seeded_quote: str = ""
    submitted_quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.pre_seeded_quote.strip() and self.submitted_quote.strip())

    @property
    def is_issue(self) -> bool:
        return self.assessment != "same" and self.is_evidenced

    @property
    def is_fail(self) -> bool:
        return self.assessment == "different" and self.is_evidenced


@dataclass
class ArtifactFinding:
    """Check 95. Files one model claimed to produce, against what was uploaded.

    `basis` records how the comparison was made. Uploads arrive as CDN URLs whose
    last path segment is a content ID, not a filename, and the objects carry no
    Content-Disposition, so the name is unrecoverable. When that happens the check
    can still tell that three files were claimed and two uploaded, but it cannot
    tell *which* is absent -- so it counts the shortfall and names nothing.
    """

    model: ModelSlot
    claimed_files: list[str] = field(default_factory=list)
    uploaded_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    basis: Literal["filename", "count"] = "filename"
    shortfall: int = 0
    # Claimed files matched an uploaded file only above the fuzzy threshold, not
    # exactly -- e.g. "report.md" claimed against an uploaded "report (2).md".
    # Not in `missing_files` (the match counts as present), kept here so the
    # near-miss stays visible instead of disappearing silently into a clean
    # verdict.
    fuzzy_matched_files: list[tuple[str, str]] = field(default_factory=list)
    # Uploads no claimed file was assigned to. Each upload accounts for one claimed
    # file at most, so this is what a rename still has left to point at -- and an
    # empty list means every upload is already spoken for, which is what makes "the
    # same file under another name" impossible rather than merely unlikely.
    unmatched_uploads: list[str] = field(default_factory=list)
    # Deliveries the transcript records without recording what was delivered: the
    # turn is labelled as producing a file whose content is absent, or it sits
    # past the render budget's cut. The name is then unreadable, so comparing it
    # against the manifest can only invent a missing file -- and this check has no
    # middle band to absorb that. Held out of `claimed_files` by the parser, so
    # neither the filename subtraction nor the count shortfall can see them.
    unverifiable_files: list[str] = field(default_factory=list)
    why_unverifiable: str = ""
    confidence: str = "high"

    # What the confirming pass made of each name the subtraction called missing.
    # Only fails are confirmed, so an empty dict on a finding with no
    # `missing_files` means nothing was in doubt, not that nothing was checked;
    # `confirmation_ran` separates those two.
    #
    #   `missing`   -- genuinely absent; stays counted.
    #   `renamed`   -- the same file under another name, which string distance
    #                  could not bridge (a capture tool's screenshot name against
    #                  the contributor's own name for the same image). Only
    #                  accepted against an upload no other claimed file was
    #                  assigned to: one upload cannot be two deliverables.
    #   `superseded`-- an earlier state of a deliverable the model went on to
    #                  revise, whose final state *is* in the manifest. What the
    #                  check asks is whether the work product was preserved, and an
    #                  iterative session's intermediate revisions are not each a
    #                  separate work product. Counting them made a contributor who
    #                  uploaded the final image of a twelve-round editing session
    #                  responsible for eleven files nobody wanted.
    #   `not_a_file`-- never a claimed attachment at all: a fenced code block, an
    #                  inline table, a path the model proposed rather than
    #                  delivered, or an image the chat UI rendered inline with a
    #                  download button of its own.
    #   `unverifiable` -- the judge could not tell, which leaves it uncounted for
    #                  the same reason every other abstention in this build does.
    confirmations: dict[str, str] = field(default_factory=dict)
    confirmation_reasons: dict[str, str] = field(default_factory=dict)
    confirmation_ran: bool = False
    # Files the extraction listed from an earlier turn, held out of
    # `claimed_files` when `policy.artifact_scope == "final_response"`. V5
    # excuses them; they are published so a reviewer can see what was dropped.
    out_of_scope_files: list[str] = field(default_factory=list)
    # The turn treated as the final response when the scope is last-turn-only.
    final_response_turn: int | None = None

    @property
    def confirmed_missing(self) -> list[str]:
        """The names still counted after confirmation.

        With no confirming pass this is `missing_files` unchanged, which is what
        keeps the policy flag's "off" branch byte-for-byte the old behaviour.
        """
        if not self.confirmation_ran:
            return list(self.missing_files)
        return [
            name
            for name in self.missing_files
            if self.confirmations.get(name, "missing") == "missing"
        ]

    @property
    def cleared_by_confirmation(self) -> list[tuple[str, str]]:
        """Names the confirming pass took out of the count, with its verdict."""
        return [
            (name, self.confirmations[name])
            for name in self.missing_files
            if self.confirmations.get(name, "missing") != "missing"
        ]

    @property
    def missing_count(self) -> int:
        if self.basis == "count":
            return self.shortfall
        return len(self.confirmed_missing)


@dataclass
class EnvironmentContextFinding:
    """Check 85. One thing the prompt references that the task's universe lacks.

    The spec's bar is that the prompt "references only the entities and events in
    the task's universe", which is narrower than any reference the audit cannot
    immediately place. Three conditions separate a violation from a loose mention,
    and all of them have to hold:

      * `presented_as_existing` -- the prompt asserts the thing is already there,
        rather than proposing, hypothesising, or asking about it.
      * `required_to_complete` -- the work cannot be done without it, so its
        absence breaks the task rather than decorating it.
      * not `real_world_public_entity` -- a task about MikroTik switches names
        MikroTik legitimately, and public people, products and events are part of
        any task's world without being supplied by it.

    `evidenced` is the separate, harder gate: a finding with no verbatim quote, or
    that does not say what the reference was checked against, is unfalsifiable and
    is discarded rather than scored.
    """

    reference: str
    kind: Literal["file", "person", "organisation", "event", "system", "place", "other"]
    quote: str = ""
    checked_against: list[str] = field(default_factory=list)
    why_outside_universe: str = ""
    presented_as_existing: bool = True
    required_to_complete: bool = True
    real_world_public_entity: bool = False
    basis: Literal["deterministic", "model"] = "model"
    confidence: str = "high"

    @property
    def evidenced(self) -> bool:
        return bool(self.quote.strip()) and any(
            c.strip() for c in self.checked_against
        )

    @property
    def is_violation(self) -> bool:
        return (
            self.evidenced
            and self.presented_as_existing
            and self.required_to_complete
            and not self.real_world_public_entity
        )

    @property
    def item_id(self) -> str:
        return f"{self.kind}::{self.reference}"


@dataclass
class KeyTurnJustificationFinding:
    """Check 110. The bar is 'contains any issues', so one flag is enough."""

    # Which model's justification this is. Empty on the L1 single-call path.
    model: str = ""
    describes_selected_turn: bool = True
    claims_are_accurate: bool = True
    connects_to_core_value: bool = True
    issues: list[str] = field(default_factory=list)
    # The selected turn was not in what the judge was shown -- cut by the render
    # budget, or delivered as a file whose content the transcript omits. All three
    # questions above are asked *of that turn*, so a judge that cannot read it
    # cannot answer any of them, and its guesses are what the parser drops: the
    # three flags come back True and the alleged defects land here instead.
    # Reported, never counted, because the bar is "contains any issues" and there
    # is no band below non_fail to put a mistake in.
    unverifiable: bool = False
    why_unverifiable: str = ""
    unverifiable_issues: list[str] = field(default_factory=list)
    confidence: str = "high"

    @property
    def is_issue(self) -> bool:
        return bool(self.issues) or not (
            self.describes_selected_turn
            and self.claims_are_accurate
            and self.connects_to_core_value
        )


@dataclass
class AutofailFinding:
    """Check 220. One criterion alleged to render the whole outcome worthless.

    `confirmed` is set by the mandatory second opinion. An allegation only one
    pass believes is recorded but never fails the task, because this check has
    no non-fail band to absorb a mistake.
    """

    criterion_id: str
    quote: str
    why_outcome_is_useless: str
    confirmed: bool = False
    confidence: str = "high"


@dataclass
class VerdictFinding:
    """Check 470. Whether the comparison justification states a preference."""

    states_preference: bool
    quote: str = ""
    confidence: str = "high"
    # Which comparison's justification this is about. See `LikertFinding.pair`.
    pair: str = "AB"


# A direction names a slot or declines to name one. Widened past A/B because a
# comparison's right-hand side is C in half of Honeybee Aspirational's pairs.
Direction = Literal["A", "B", "C", "neutral"]


@dataclass
class InversionFinding:
    """Check 460. The contradiction is detected deterministically; only the
    adequacy of the justification is a model judgment."""

    dimension_direction: Direction
    likert_direction: Direction
    justification_explains_inversion: bool = False
    confidence: str = "high"
    # Which comparison this is about. See `LikertFinding.pair`.
    pair: str = "AB"

    @property
    def contradicts(self) -> bool:
        if "neutral" in (self.dimension_direction, self.likert_direction):
            return False
        return self.dimension_direction != self.likert_direction
