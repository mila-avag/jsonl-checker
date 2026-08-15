"""Repeated sampling: draw a judgment more than once and keep only what survives.

Three agents each fixed a check definition this morning, measured honestly, and
each found the flags dropped while the task verdicts did not move. The reason is
upstream of every definition. Running the identical check-280 prompt four times
against the same submission rejected 2, 1, 2 and 1 of the same 10 turn citations,
and *not one citation was rejected by all four runs*. Every finding that pass
produced was one a rerun disagreed with, and a single draw of that judge was
deciding whether a task failed.

So the unit of belief here is not a response, it is an *item*: one cited turn, one
justification condition, one boolean about one key-turn justification. A judgment
counts only if a majority of independent draws agree on it.

Two design decisions carry the module.

**Aggregate on parsed findings, not on payloads.** One check-280 response carries a
dozen per-item judgments, and the samples disagree *within* a response rather than
about it wholesale. Majority-voting whole responses would throw away the very
resolution the sampling was bought for. So each sample is parsed by the existing
parser and the findings are combined here, which also means every definitional fix
another agent lands in a parser applies to all samples for free.

**Count fields aggregate by median, not by an explicit vote.** For a threshold
predicate `value >= t`, the median satisfies it exactly when a majority of samples
do, so one statistic implements majority agreement on *every* threshold at once --
including thresholds this module has never heard of, which is what keeps 450's
seven conditions from having to be enumerated here and re-enumerated whenever the
spec audit moves one. The low median is used so that an even sample count resolves
against the finding rather than for it: with two draws, both must agree.

Disagreement is recorded rather than discarded. A judge that splits 2-1 on an item
is not producing noise to be filtered, it is telling us the item is marginal, and
that belongs in the report next to the item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median_low

from .config import DEFAULT_POLICY, Policy
from .findings import (
    JustificationFinding,
    KeyTurnJustificationFinding,
    RelevantTurnFinding,
)
from .llm import ModelRequest, ModelResponse

# Checks with an aggregator below. Asking for repeats of anything else would pay
# for samples nothing combines, so `samples_for` raises instead of quietly
# charging for them.
SUPPORTED_SAMPLED_CHECKS = frozenset({110, 280, 310, 450})


class SamplingMisconfigured(ValueError):
    pass


def samples_for(check_id: int | None, policy: Policy = DEFAULT_POLICY) -> int:
    """How many independent draws this check's judgments get."""
    if check_id is None:
        return 1
    count = int(policy.samples_by_check.get(check_id, 1))
    if count < 1:
        raise SamplingMisconfigured(
            f"check {check_id} is configured for {count} samples; the minimum is 1"
        )
    if count > 1 and check_id not in SUPPORTED_SAMPLED_CHECKS:
        raise SamplingMisconfigured(
            f"check {check_id} is configured for {count} samples but has no "
            f"aggregator; sampled checks are {sorted(SUPPORTED_SAMPLED_CHECKS)}. "
            "Repeats would cost money and then be discarded."
        )
    return count


def _kept(votes: int, samples: int, policy: Policy) -> bool:
    """Strict majority. An item exactly half the draws flagged is marginal, not
    defective, and resolving it in the contributor's favour is the only direction
    that cannot invent a defect out of a coin flip."""
    if samples <= 0:
        return False
    return votes > samples * policy.sample_majority_ratio


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


@dataclass
class SampleFanout:
    """The expanded request list, and where each logical request's draws landed.

    `groups[i]` holds the positions in `requests` belonging to logical request i,
    so a caller can run one flat batch -- which is what keeps the worker pool full
    and the global throttle meaningful -- and still reassemble by judgment.
    """

    requests: list[ModelRequest] = field(default_factory=list)
    groups: list[list[int]] = field(default_factory=list)

    def responses_for(
        self, index: int, responses: list[ModelResponse]
    ) -> list[ModelResponse]:
        return [responses[p] for p in self.groups[index] if p < len(responses)]


def expand_requests(
    requests: list[ModelRequest], policy: Policy = DEFAULT_POLICY
) -> SampleFanout:
    """Duplicate each request once per configured sample.

    Sample 0 keeps the original request key so a half-finished single-sample run
    still resumes off its own cached rows, and so `hbstatus`, which counts
    distinct `request_key` values to report progress, keeps counting judgments
    rather than draws.
    """
    fanout = SampleFanout()
    for request in requests:
        count = samples_for(request.metadata.get("check_id"), policy)
        positions: list[int] = []
        for index in range(count):
            positions.append(len(fanout.requests))
            if index == 0:
                fanout.requests.append(request)
                continue
            fanout.requests.append(
                ModelRequest(
                    key=f"{request.key}::sample{index}",
                    prompt=request.prompt,
                    schema=request.schema,
                    system=request.system,
                    metadata=dict(request.metadata),
                    sample_index=index,
                )
            )
        fanout.groups.append(positions)
    return fanout


# ---------------------------------------------------------------------------
# Agreement records
# ---------------------------------------------------------------------------


@dataclass
class Agreement:
    """One claim, how many draws made it, and whether it survived."""

    check_id: int
    item_id: str
    claim: str
    votes: int
    samples: int
    kept: bool

    @property
    def split(self) -> bool:
        """Some draws made this claim and some did not."""
        return 0 < self.votes < self.samples

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "item": self.item_id,
            "claim": self.claim,
            "votes": self.votes,
            "samples": self.samples,
            "kept": self.kept,
        }


@dataclass
class AgreementLog:
    """Every sampled claim in one stage, and the disagreement rate over them.

    The rate is the headline number: it is the share of claims the judge did not
    make consistently, which is the direct measure of how much of a single-sample
    run was noise.
    """

    records: list[Agreement] = field(default_factory=list)

    def add(self, record: Agreement, policy: Policy = DEFAULT_POLICY) -> None:
        if policy.record_sample_disagreement:
            self.records.append(record)

    def extend(self, records: list[Agreement], policy: Policy = DEFAULT_POLICY) -> None:
        for record in records:
            self.add(record, policy)

    @property
    def sampled_claims(self) -> list[Agreement]:
        return [r for r in self.records if r.samples > 1]

    @property
    def split_claims(self) -> list[Agreement]:
        return [r for r in self.sampled_claims if r.split]

    @property
    def dropped_claims(self) -> list[Agreement]:
        """Claims at least one draw made that no majority upheld. These are the
        findings a single-sample run had a chance of publishing."""
        return [r for r in self.sampled_claims if r.votes and not r.kept]

    @property
    def disagreement_rate(self) -> float:
        claims = self.sampled_claims
        if not claims:
            return 0.0
        return len(self.split_claims) / len(claims)

    def to_dict(self) -> dict:
        return {
            "sampled_claims": len(self.sampled_claims),
            "split_claims": len(self.split_claims),
            "dropped_claims": len(self.dropped_claims),
            "disagreement_rate": round(self.disagreement_rate, 3),
            "splits": [r.to_dict() for r in self.split_claims],
        }


def _lowest_confidence(values) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    vals = [v for v in values if v in order]
    return min(vals, key=lambda v: order[v]) if vals else "high"


# ---------------------------------------------------------------------------
# 280 / 310 -- turn citations
# ---------------------------------------------------------------------------


def aggregate_relevant_turns(
    samples: list[list[RelevantTurnFinding]],
    check_id: int,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[list[RelevantTurnFinding], list[Agreement]]:
    """Keep a cited turn only if a majority of draws called it incorrect.

    The vote is per (item, turn), because that is the resolution at which the
    judge actually disagreed with itself: across four draws of one submission no
    single turn was rejected every time, so a whole-response vote would have
    resolved nothing.

    A draw that returned an item without flagging a turn is a vote against that
    turn, not an absence of evidence -- the judge was shown every citation and
    declined to fault this one. The denominator is therefore the number of draws
    that answered at all, so a draw lost to a timeout cannot make a 1-of-3 claim
    look like 1-of-1.

    `unverifiable_turns` is unioned rather than voted. It is not a judgment: the
    parser derives it from the end of the conversation the audit could fetch, so
    every draw sees the same truncation, and any draw noticing a turn is
    unfetchable is enough. It is reported and never counted either way.
    """
    usable = [s for s in samples if s is not None]
    total = len(usable)
    if total <= 1:
        return (usable[0] if usable else []), []

    items: list[str] = []
    votes: dict[tuple[str, int], int] = {}
    models: dict[str, object] = {}
    unverifiable: dict[str, list[int]] = {}
    confidences: dict[str, list[str]] = {}
    answered: dict[str, int] = {}

    for findings in usable:
        for f in findings:
            if f.check_id != check_id:
                continue
            if f.item_id not in models:
                items.append(f.item_id)
                models[f.item_id] = f.model
            answered[f.item_id] = answered.get(f.item_id, 0) + 1
            confidences.setdefault(f.item_id, []).append(f.confidence)
            for turn in f.incorrect_turns:
                votes[(f.item_id, turn)] = votes.get((f.item_id, turn), 0) + 1
            for turn in f.unverifiable_turns:
                seen = unverifiable.setdefault(f.item_id, [])
                if turn not in seen:
                    seen.append(turn)

    records: list[Agreement] = []
    kept_turns: dict[str, list[int]] = {}
    for (item_id, turn), count in votes.items():
        # The denominator is how many draws answered for THIS item, not how many
        # draws were usable overall: a draw that never mentioned this item_id
        # (omitted it, errored on it, or was retried) must not dilute the vote
        # of the draws that did answer.
        item_total = answered.get(item_id, total)
        keep = _kept(count, item_total, policy)
        records.append(
            Agreement(
                check_id=check_id,
                item_id=item_id,
                claim=f"turn {turn} incorrect",
                votes=count,
                samples=item_total,
                kept=keep,
            )
        )
        if keep:
            kept_turns.setdefault(item_id, []).append(turn)

    aggregated = [
        RelevantTurnFinding(
            check_id=check_id,  # type: ignore[arg-type]
            item_id=item_id,
            model=models[item_id],  # type: ignore[arg-type]
            incorrect_turns=sorted(kept_turns.get(item_id, [])),
            unverifiable_turns=sorted(unverifiable.get(item_id, [])),
            confidence=_lowest_confidence(confidences.get(item_id, [])),
        )
        for item_id in items
    ]
    records.sort(key=lambda r: (r.item_id, r.claim))
    return aggregated, records


# ---------------------------------------------------------------------------
# 450 -- justification conditions
# ---------------------------------------------------------------------------

# Which numeric fields feed which of 450's conditions, used only to name a claim
# in the agreement log. The aggregation itself never consults this: the median
# already resolves every threshold, so a condition added or retuned by the spec
# audit needs no change here.
_JUSTIFICATION_COUNT_FIELDS = (
    "contradicts_verdict_claims",
    "inaccurate_primary_claims",
    "inaccurate_secondary_claims",
    "unsupported_claims",
    "inaccurate_evidence",
    "misconstrued_evidence",
    "unverifiable_claims",
)


def aggregate_justifications(
    samples: list[list[JustificationFinding]], policy: Policy = DEFAULT_POLICY
) -> tuple[list[JustificationFinding], list[Agreement]]:
    """Combine per-justification condition counts across draws.

    Counts take the low median, which is what makes this robust to a moving
    definition: `median >= t` holds exactly when a majority of draws have
    `value >= t`, so every one of 450's thresholds -- "2 or more claims lack
    evidence", "1 or more inaccurate primary claim", and whichever ones the spec
    audit lands next -- gets majority agreement from a single statistic, with no
    list of conditions in this module to fall out of date.

    Booleans take a strict majority. Evidence spans are unioned, because both of
    them are used to *withhold* a finding rather than to make one: a single draw
    quoting something specific refutes the claim that a justification is generic,
    and 450 requires a contradiction to be exhibited before it can be counted. A
    union is the reading that cannot manufacture a defect.
    """
    usable = [s for s in samples if s is not None]
    total = len(usable)
    if total <= 1:
        return (usable[0] if usable else []), []

    order: list[str] = []
    by_item: dict[str, list[JustificationFinding]] = {}
    for findings in usable:
        for f in findings:
            if f.item_id not in by_item:
                order.append(f.item_id)
                by_item[f.item_id] = []
            by_item[f.item_id].append(f)

    aggregated: list[JustificationFinding] = []
    records: list[Agreement] = []

    for item_id in order:
        drawn = by_item[item_id]
        # Draws that never mentioned this justification are votes that it carried
        # nothing, so they pad the count list with zeroes rather than shrinking
        # the denominator.
        missing = [0] * (total - len(drawn))

        def count(name: str) -> int:
            return int(median_low([getattr(f, name) for f in drawn] + missing))

        def vote(name: str) -> int:
            return sum(1 for f in drawn if getattr(f, name))

        merged = JustificationFinding(
            item_id=item_id,
            rated_value=next(
                (f.rated_value for f in drawn if f.rated_value is not None), None
            ),
            contradicts_verdict_claims=count("contradicts_verdict_claims"),
            is_generic=_kept(vote("is_generic"), total, policy),
            is_skewed=_kept(vote("is_skewed"), total, policy),
            inaccurate_primary_claims=count("inaccurate_primary_claims"),
            inaccurate_secondary_claims=count("inaccurate_secondary_claims"),
            unsupported_claims=count("unsupported_claims"),
            inaccurate_evidence=count("inaccurate_evidence"),
            misconstrued_evidence=count("misconstrued_evidence"),
            unverifiable_claims=count("unverifiable_claims"),
            specifics_quoted=sorted(
                {s for f in drawn for s in f.specifics_quoted if s.strip()}
            ),
            contradicting_quotes=sorted(
                {q for f in drawn for q in f.contradicting_quotes if q.strip()}
            ),
            confidence=_lowest_confidence([f.confidence for f in drawn]),
        )
        aggregated.append(merged)

        surviving = set(merged.triggered_conditions())
        alleged = {c for f in drawn for c in f.triggered_conditions()}
        for condition in sorted(alleged | surviving):
            records.append(
                Agreement(
                    check_id=450,
                    item_id=item_id,
                    claim=condition,
                    votes=sum(
                        1 for f in drawn if condition in f.triggered_conditions()
                    ),
                    samples=total,
                    kept=condition in surviving,
                )
            )
        for name in _JUSTIFICATION_COUNT_FIELDS:
            values = [getattr(f, name) for f in drawn]
            if len(set(values)) > 1:
                records.append(
                    Agreement(
                        check_id=450,
                        item_id=item_id,
                        claim=f"{name} spread {min(values)}-{max(values)}",
                        votes=sum(1 for v in values if v),
                        samples=total,
                        kept=bool(getattr(merged, name)),
                    )
                )

    return aggregated, records


# ---------------------------------------------------------------------------
# 110 -- key turn justification
# ---------------------------------------------------------------------------

_KEY_TURN_BOOLEANS = (
    "describes_selected_turn",
    "claims_are_accurate",
    "connects_to_core_value",
)


def aggregate_key_turn_justification(
    samples: list[KeyTurnJustificationFinding | None],
    policy: Policy = DEFAULT_POLICY,
) -> tuple[KeyTurnJustificationFinding | None, list[Agreement]]:
    """Majority-vote 110's three booleans and the existence of a free-text issue.

    110's bar is "contains any issues", so one flag from one draw fails a task
    outright on a check with no fail band to absorb it. That makes it the shape
    most exposed to a noisy draw, which is why it is sampled by default despite
    being a single cheap call.

    Free-text issues cannot be matched across draws -- two draws describe the same
    defect in different words -- so the vote is on whether a draw reported any
    issue at all. When no majority did, the minority's text is still carried into
    the agreement log rather than deleted, because a near miss on this check is
    something a reviewer should see.
    """
    usable = [s for s in samples if s is not None]
    total = len(usable)
    if total <= 1:
        return (usable[0] if usable else None), []

    records: list[Agreement] = []
    values: dict[str, bool] = {}
    for name in _KEY_TURN_BOOLEANS:
        # These default True for "no problem", so the claim being voted on is the
        # negation: how many draws said this leg was NOT satisfied.
        against = sum(1 for f in usable if not getattr(f, name))
        failed = _kept(against, total, policy)
        values[name] = not failed
        if against:
            records.append(
                Agreement(
                    check_id=110,
                    item_id="key_turn_justification",
                    claim=f"not {name}",
                    votes=against,
                    samples=total,
                    kept=failed,
                )
            )

    with_issues = sum(1 for f in usable if f.issues)
    keep_issues = _kept(with_issues, total, policy)
    if with_issues:
        records.append(
            Agreement(
                check_id=110,
                item_id="key_turn_justification",
                claim="reports a free-text issue",
                votes=with_issues,
                samples=total,
                kept=keep_issues,
            )
        )

    issue_texts = sorted({i for f in usable for i in f.issues if i.strip()})
    if not keep_issues:
        for text in issue_texts:
            records.append(
                Agreement(
                    check_id=110,
                    item_id="key_turn_justification",
                    claim=f"issue not upheld: {text}",
                    votes=sum(1 for f in usable if text in f.issues),
                    samples=total,
                    kept=False,
                )
            )

    return (
        KeyTurnJustificationFinding(
            describes_selected_turn=values["describes_selected_turn"],
            claims_are_accurate=values["claims_are_accurate"],
            connects_to_core_value=values["connects_to_core_value"],
            issues=issue_texts if keep_issues else [],
            confidence=_lowest_confidence([f.confidence for f in usable]),
        ),
        records,
    )


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


@dataclass
class SamplingCost:
    """Projected calls and dollars for one sample configuration.

    Reported rather than assumed, because this is the user's money and three
    samples of everything is roughly triple the bill.
    """

    calls_by_check: dict[int, int] = field(default_factory=dict)
    samples_by_check: dict[int, int] = field(default_factory=dict)
    unit_cost_by_check: dict[int, float] = field(default_factory=dict)

    @property
    def single_sample_calls(self) -> int:
        return sum(self.calls_by_check.values())

    @property
    def sampled_calls(self) -> int:
        return sum(
            n * self.samples_by_check.get(check_id, 1)
            for check_id, n in self.calls_by_check.items()
        )

    @property
    def single_sample_usd(self) -> float:
        return sum(
            n * self.unit_cost_by_check.get(check_id, 0.0)
            for check_id, n in self.calls_by_check.items()
        )

    @property
    def sampled_usd(self) -> float:
        return sum(
            n * self.samples_by_check.get(check_id, 1)
            * self.unit_cost_by_check.get(check_id, 0.0)
            for check_id, n in self.calls_by_check.items()
        )

    def to_dict(self) -> dict:
        return {
            "single_sample_calls": self.single_sample_calls,
            "sampled_calls": self.sampled_calls,
            "extra_calls": self.sampled_calls - self.single_sample_calls,
            "single_sample_usd": round(self.single_sample_usd, 2),
            "sampled_usd": round(self.sampled_usd, 2),
            "extra_usd": round(self.sampled_usd - self.single_sample_usd, 2),
            "samples_by_check": dict(sorted(self.samples_by_check.items())),
        }


def project_cost(
    calls_by_check: dict[int, int],
    unit_cost_by_check: dict[int, float],
    policy: Policy = DEFAULT_POLICY,
) -> SamplingCost:
    return SamplingCost(
        calls_by_check=dict(calls_by_check),
        samples_by_check={
            check_id: samples_for(check_id, policy) for check_id in calls_by_check
        },
        unit_cost_by_check=dict(unit_cost_by_check),
    )
