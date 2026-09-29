"""Repeated sampling: cache identity, majority aggregation, and cost projection.

The property the whole change rests on is tested first and directly: N samples
must be N real calls. If the cache collapses them the aggregation still runs, still
reports unanimity, and is measuring one draw N times -- which is exactly the
failure the sampling was bought to fix, and it would be invisible in the output.
"""

from __future__ import annotations

import dataclasses
import random
from statistics import median_low

import pytest

from honeybee_qc.cache import ResponseCache, compute_cache_key
from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.findings import (
    JustificationFinding,
    KeyTurnJustificationFinding,
    RelevantTurnFinding,
)
from honeybee_qc.gates import evaluate_check_450, evaluate_relevant_turns
from honeybee_qc.informed_stages import estimate_informed_calls, run_informed_stage
from honeybee_qc.llm import CachedClient, FakeModelClient, ModelRequest
from honeybee_qc.sampling import (
    SamplingMisconfigured,
    _justification_votes,
    _kept,
    _kth_largest,
    aggregate_justifications,
    aggregate_key_turn_justification,
    aggregate_relevant_turns,
    expand_requests,
    project_cost,
    samples_for,
)
from honeybee_qc.tests.fixtures import make_task

TASK = make_task()

SAMPLED = dataclasses.replace(
    DEFAULT_POLICY, samples_by_check={110: 3, 280: 3, 310: 3, 450: 3}
)
SINGLE = dataclasses.replace(DEFAULT_POLICY, samples_by_check={})


def _turn_finding(item: str, turns: list[int], unverifiable: list[int] | None = None):
    return RelevantTurnFinding(
        check_id=280,
        item_id=item,
        model="A",
        incorrect_turns=list(turns),
        unverifiable_turns=list(unverifiable or []),
    )


# ---------------------------------------------------------------------------
# Cache identity: the reason repeated sampling was impossible before
# ---------------------------------------------------------------------------


def _key(**overrides):
    base = dict(
        prompt="judge this",
        schema={"type": "object"},
        model="claude-sonnet-4-6",
        effort="medium",
        policy_version="v1",
        system="sys",
    )
    base.update(overrides)
    return compute_cache_key(**base)


def test_sample_zero_keys_exactly_as_it_did_before_sampling_existed():
    """Roughly $95 of paid-for responses are keyed without a sample index. Folding
    one in unconditionally would orphan every one of them."""
    assert _key() == _key(sample_index=0)


def test_each_sample_gets_its_own_cache_identity():
    keys = {_key(sample_index=i) for i in range(4)}
    assert len(keys) == 4


def test_repeated_samples_are_distinct_calls_not_cache_hits(tmp_path):
    """The load-bearing property. Everything else here is arithmetic over draws
    that have to actually differ, and they cannot differ if the second draw reads
    the first one's row."""
    inner = FakeModelClient(lambda r: {"judgments": [], "sample": r.sample_index})
    cache = ResponseCache(tmp_path / "c.db")
    client = CachedClient(
        inner, cache, model="m", effort="medium", policy_version="v1"
    )

    request = ModelRequest(key="k", prompt="p", schema={"type": "object"})
    for index in range(3):
        client.complete(dataclasses.replace(request, sample_index=index))

    assert len(inner.calls) == 3, "samples collapsed into one cached answer"
    assert cache.hits == 0

    # And each is independently resumable: a second pass over the same three
    # samples costs nothing.
    for index in range(3):
        client.complete(dataclasses.replace(request, sample_index=index))
    assert len(inner.calls) == 3
    assert cache.hits == 3
    cache.close()


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def test_unsampled_check_is_untouched_by_fanout():
    request = ModelRequest(
        key="t::verdict", prompt="p", schema={}, metadata={"check_id": 470}
    )
    fanout = expand_requests([request], SAMPLED)
    assert fanout.requests == [request]
    assert fanout.groups == [[0]]


def test_sampled_check_fans_out_with_the_first_draw_keyed_unchanged():
    request = ModelRequest(
        key="t::criterion_turns::A", prompt="p", schema={}, metadata={"check_id": 280}
    )
    fanout = expand_requests([request], SAMPLED)
    assert len(fanout.requests) == 3
    assert fanout.groups == [[0, 1, 2]]
    assert [r.sample_index for r in fanout.requests] == [0, 1, 2]
    # Draw 0 keeps the bare key so `hbstatus`, which counts distinct request keys
    # to report progress, keeps counting judgments rather than draws.
    assert fanout.requests[0].key == "t::criterion_turns::A"
    assert len({r.key for r in fanout.requests}) == 3
    assert all(r.prompt == "p" for r in fanout.requests)


def test_fanout_groups_track_each_judgment_across_mixed_sample_counts():
    requests = [
        ModelRequest(key="a", prompt="p", schema={}, metadata={"check_id": 470}),
        ModelRequest(key="b", prompt="p", schema={}, metadata={"check_id": 280}),
        ModelRequest(key="c", prompt="p", schema={}, metadata={"check_id": 70}),
    ]
    fanout = expand_requests(requests, SAMPLED)
    assert fanout.groups == [[0], [1, 2, 3], [4]]
    assert len(fanout.requests) == 5


def test_sampling_an_unaggregatable_check_is_refused_rather_than_billed():
    """Paying for draws nothing combines is worse than not sampling: the cost is
    real and the output is unchanged."""
    policy = dataclasses.replace(DEFAULT_POLICY, samples_by_check={230: 3})
    with pytest.raises(SamplingMisconfigured):
        samples_for(230, policy)


def test_zero_samples_is_refused():
    policy = dataclasses.replace(DEFAULT_POLICY, samples_by_check={280: 0})
    with pytest.raises(SamplingMisconfigured):
        samples_for(280, policy)


# ---------------------------------------------------------------------------
# 280 / 310 aggregation
# ---------------------------------------------------------------------------


def test_a_turn_two_of_three_draws_flag_survives_and_one_of_three_does_not():
    samples = [
        [_turn_finding("C1", [4, 9])],
        [_turn_finding("C1", [4])],
        [_turn_finding("C1", [4, 12])],
    ]
    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert len(findings) == 1
    assert findings[0].incorrect_turns == [4]

    kept = {r.claim: r for r in records}
    assert kept["turn 4 incorrect"].votes == 3 and kept["turn 4 incorrect"].kept
    assert kept["turn 9 incorrect"].votes == 1 and not kept["turn 9 incorrect"].kept
    assert kept["turn 12 incorrect"].votes == 1


def test_a_turn_flagged_by_every_draw_but_one_still_survives():
    samples = [
        [_turn_finding("C1", [4])],
        [_turn_finding("C1", [4])],
        [_turn_finding("C1", [])],
    ]
    findings, _ = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == [4]


def test_an_even_split_resolves_in_the_contributors_favour():
    """Strict majority. A judge split down the middle is saying the item is
    marginal, and resolving that as a defect invents one out of a coin flip."""
    samples = [[_turn_finding("C1", [4])], [_turn_finding("C1", [])]]
    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == []
    assert [r.kept for r in records] == [False]


def test_unverifiable_turns_are_unioned_and_never_counted():
    """The parser derives these from the end of the conversation the audit could
    fetch, so they are not a judgment and every draw sees the same truncation."""
    samples = [
        [_turn_finding("C1", [], unverifiable=[99])],
        [_turn_finding("C1", [], unverifiable=[])],
        [_turn_finding("C1", [], unverifiable=[99, 100])],
    ]
    findings, _ = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].unverifiable_turns == [99, 100]
    assert findings[0].incorrect_turns == []
    verdict = evaluate_relevant_turns(TASK, 280, findings, SAMPLED)
    assert verdict.measurement.numerator == 0


def test_a_failed_draw_costs_resolution_and_never_invents_a_verdict():
    """Two usable draws out of three means the majority is over two, not over
    three; a timeout must not turn a 1-of-3 claim into a 1-of-1 finding."""
    samples = [[_turn_finding("C1", [4])], None, [_turn_finding("C1", [4])]]
    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == [4]
    assert all(r.samples == 2 for r in records)


def test_three_draws_drop_a_finding_that_would_have_failed_the_task():
    """The whole point, at the gate, on the shape the live run actually produced:
    across four draws of one submission no single citation was rejected every
    time. 280 fails at three counted turns, so one draw flagging three fails the
    task by itself, and nothing it flagged survives a majority."""
    samples = [
        [_turn_finding("C1", [2, 5, 8])],
        [_turn_finding("C1", [11])],
        [_turn_finding("C1", [14])],
    ]
    single = evaluate_relevant_turns(TASK, 280, samples[0], SAMPLED)
    assert single.band == "fail"

    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == []
    assert evaluate_relevant_turns(TASK, 280, findings, SAMPLED).band == "clean"
    assert all(not r.kept for r in records)


def test_turns_a_majority_agree_on_survive_and_only_demote_the_band():
    """Sampling is not a blanket amnesty. Two of three draws faulting a turn is
    agreement, and two surviving turns sit under 280's threshold of three, so the
    task lands in non_fail rather than clean.

    Worth pinning because of how the QC-sheet scorer reads a band: it counts
    non_fail as a flag exactly like fail, so a check that only moves from fail to
    non_fail changes the task verdict and leaves check-level precision untouched.
    """
    samples = [
        [_turn_finding("C1", [2, 5, 8])],
        [_turn_finding("C1", [2])],
        [_turn_finding("C1", [5])],
    ]
    findings, _ = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == [2, 5]
    assert evaluate_relevant_turns(TASK, 280, findings, SAMPLED).band == "non_fail"


def test_single_sample_aggregation_is_the_identity():
    findings = [_turn_finding("C1", [4, 9])]
    out, records = aggregate_relevant_turns([findings], 280, SINGLE)
    assert out == findings
    assert records == []


def test_an_item_only_one_draw_answered_for_is_judged_against_that_one_draw():
    """The docstring's own promise: the denominator is how many draws answered
    for THIS item, not how many draws were usable overall. Here two of the three
    draws never returned a finding for "C2" at all (as if the model omitted it,
    or that item's judgment errored and was dropped) -- only draw 0 answered for
    it, and that lone draw flagged turn 6 as incorrect.

    Fixed behavior: 1 vote out of 1 answering draw is a unanimous majority, so
    the turn is kept. Under the old bug, `_kept` was called against the
    batch-wide denominator of 3 regardless of how many draws answered for this
    item, so the same vote would have been judged as 1-of-3 (1 > 3*0.5=1.5 is
    False) and silently dropped -- exactly the mismatch the manual audit found
    between the raw per-draw judgment and the aggregated measurement.
    """
    samples = [
        [_turn_finding("C1", [4]), _turn_finding("C2", [6])],
        [_turn_finding("C1", [4])],
        [_turn_finding("C1", [4])],
    ]
    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)

    by_item = {f.item_id: f for f in findings}
    # Fixed behavior: 1 vote out of 1 answering draw is unanimous, so it is kept.
    assert by_item["C2"].incorrect_turns == [6]

    record = next(
        r for r in records if r.item_id == "C2" and r.claim == "turn 6 incorrect"
    )
    assert record.votes == 1
    assert record.samples == 1, "denominator must be draws that answered for C2, not 3"
    assert record.kept is True


def test_an_item_every_draw_answers_for_still_uses_the_full_sample_count():
    """Confirms the fix does not disturb ordinary majority voting when all draws
    answer for an item: the per-item denominator equals the batch-wide total, so
    behavior over a fully-answered item is unchanged."""
    samples = [
        [_turn_finding("C1", [4, 9])],
        [_turn_finding("C1", [4])],
        [_turn_finding("C1", [4, 12])],
    ]
    findings, records = aggregate_relevant_turns(samples, 280, SAMPLED)
    assert findings[0].incorrect_turns == [4]

    kept = {r.claim: r for r in records}
    assert kept["turn 4 incorrect"].votes == 3 and kept["turn 4 incorrect"].samples == 3
    assert kept["turn 4 incorrect"].kept
    assert kept["turn 9 incorrect"].samples == 3 and not kept["turn 9 incorrect"].kept
    assert kept["turn 12 incorrect"].samples == 3


# ---------------------------------------------------------------------------
# 450 aggregation
# ---------------------------------------------------------------------------


def _just(item: str = "A::helpfulness", **fields):
    return JustificationFinding(item_id=item, **fields)


def test_a_condition_a_bare_majority_of_draws_trip_no_longer_survives():
    """The 450 aggregation change: a majority is not enough, unanimity is required.

    Two of three draws saying a justification leaves claims unevidenced used to be
    the finding. Measured over 400 five-draw groups, only 17% of flagged items were
    unanimous and 56% were the opinion of one or two draws -- and both figures
    replicated out of sample once corpus composition was controlled -- so a
    majority here is close to a coin flip, and one severe item fails the task.
    """
    samples = [
        [_just(unsupported_claims=3)],
        [_just(unsupported_claims=2)],
        [_just(unsupported_claims=0)],
    ]
    findings, records = aggregate_justifications(samples, SAMPLED)
    assert findings[0].unsupported_claims == 0
    assert findings[0].triggered_conditions() == []

    # The dissent is recorded rather than discarded: the two draws that alleged it
    # are still in the agreement log, marked as not kept.
    dropped = [r for r in records if r.claim == "lacks_evidence"]
    assert dropped and not dropped[0].kept
    assert (dropped[0].votes, dropped[0].samples) == (2, 3)


def test_the_agreement_bar_is_one_named_field_and_3_restores_the_old_median():
    """`justification_draw_agreement` is settled at 5, and 3 replays a prior run.

    The field stays named rather than inlined for two reasons: 3 is what reproduces
    every run scored before it existed, and the severe route is still being
    measured separately, so a second agreement bar may be added beside this one.
    """
    # The live configuration: 450 draws five samples.
    five = dataclasses.replace(DEFAULT_POLICY, samples_by_check={450: 5})
    samples = [
        [_just(unsupported_claims=3)],
        [_just(unsupported_claims=2)],
        [_just(unsupported_claims=2)],
        [_just(unsupported_claims=0)],
        [_just(unsupported_claims=0)],
    ]
    findings, _ = aggregate_justifications(samples, five)
    assert findings[0].unsupported_claims == 0, "3 of 5 must not survive at k=5"

    as_median = dataclasses.replace(five, justification_draw_agreement=3)
    findings, _ = aggregate_justifications(samples, as_median)
    assert findings[0].unsupported_claims == 2
    assert "lacks_evidence" in findings[0].triggered_conditions()

    # And the setting rejected on mechanical grounds behaves as measured: four
    # draws must reach the value, and only three did.
    as_four = dataclasses.replace(five, justification_draw_agreement=4)
    findings, _ = aggregate_justifications(samples, as_four)
    assert findings[0].unsupported_claims == 0


def test_agreement_3_is_byte_identical_to_the_median_low_it_replaced():
    """The evidence that the refactor is behaviour-preserving, as a property.

    The patch this came from was verified by replaying 68 stored tasks at
    `agreement=3` and finding zero differences in any aggregated field. That is a
    fact about one corpus; this is the same claim over every shape the aggregator
    can be handed -- group sizes 2 to 5, counts and booleans, items some draws
    never mention -- checked against a transcription of the pre-patch code. If
    either is ever wrong, a pre-patch run stops being reproducible and every
    number measured against one becomes unreadable.
    """
    rng = random.Random(4507)
    fields = (
        "contradicts_verdict_claims",
        "inaccurate_primary_claims",
        "inaccurate_secondary_claims",
        "unsupported_claims",
        "inaccurate_evidence",
        "misconstrued_evidence",
        "unverifiable_claims",
    )
    policy = dataclasses.replace(
        DEFAULT_POLICY, samples_by_check={450: 5}, justification_draw_agreement=3
    )
    for _ in range(400):
        samples = []
        for _draw in range(rng.randint(2, 5)):
            drawn = []
            for item in ("j0", "j1"):
                # A draw that never mentions an item is the case the zero padding
                # exists for, and the one an order statistic gets wrong if the
                # denominator shrinks instead.
                if rng.random() < 0.2:
                    continue
                drawn.append(
                    _just(
                        item,
                        is_generic=rng.random() < 0.4,
                        is_skewed=rng.random() < 0.3,
                        **{f: rng.randint(0, 3) for f in fields},
                    )
                )
            samples.append(drawn)

        got, _ = aggregate_justifications(samples, policy)

        total = len(samples)
        by_item: dict[str, list] = {}
        order: list[str] = []
        for drawn in samples:
            for f in drawn:
                if f.item_id not in by_item:
                    order.append(f.item_id)
                    by_item[f.item_id] = []
                by_item[f.item_id].append(f)
        if total <= 1:
            continue

        assert [f.item_id for f in got] == order
        for merged in got:
            drawn = by_item[merged.item_id]
            missing = [0] * (total - len(drawn))
            for name in fields:
                # The pre-patch aggregation, transcribed: `median_low` on the
                # counts and a strict majority on the booleans.
                expected = int(median_low([getattr(f, name) for f in drawn] + missing))
                assert getattr(merged, name) == expected, (merged.item_id, name, total)
            for name in ("is_generic", "is_skewed"):
                votes = sum(1 for f in drawn if getattr(f, name))
                assert getattr(merged, name) == _kept(votes, total, policy), (
                    merged.item_id,
                    name,
                    total,
                )


def test_a_condition_only_one_draw_trips_does_not_survive():
    samples = [
        [_just(unsupported_claims=4)],
        [_just(unsupported_claims=0)],
        [_just(unsupported_claims=1)],
    ]
    findings, records = aggregate_justifications(samples, SAMPLED)
    assert findings[0].unsupported_claims == 0
    assert findings[0].triggered_conditions() == []
    dropped = [r for r in records if r.claim == "lacks_evidence"]
    assert dropped and not dropped[0].kept and dropped[0].votes == 1


def test_the_order_statistic_resolves_a_threshold_this_module_never_names():
    """`kth_largest >= t` holds exactly when at least k draws do, for every t. That
    is what keeps 450's seven conditions -- and whichever ones the spec audit
    retunes next -- from having to be enumerated in the aggregator."""
    unanimous = [
        [_just(inaccurate_secondary_claims=2)],
        [_just(inaccurate_secondary_claims=2)],
        [_just(inaccurate_secondary_claims=3)],
    ]
    findings, _ = aggregate_justifications(unanimous, SAMPLED)
    # The bar is "at least k draws reached this value", not "k draws agreed on a
    # number": the third draw counting one more does not reduce the other two.
    assert findings[0].inaccurate_secondary_claims == 2
    assert "is_inaccurate" in findings[0].triggered_conditions()

    split = [
        [_just(inaccurate_secondary_claims=2)],
        [_just(inaccurate_secondary_claims=2)],
        [_just(inaccurate_secondary_claims=0)],
    ]
    findings, _ = aggregate_justifications(split, SAMPLED)
    assert findings[0].triggered_conditions() == []


def test_a_single_quoted_specific_refutes_genericness_across_draws():
    """Evidence spans are unioned because both of 450's span fields withhold a
    finding rather than make one -- and the union must not tighten with k."""
    samples = [
        [_just(is_generic=True)],
        [_just(is_generic=True)],
        [_just(is_generic=False, specifics_quoted=["the 11-second clip"])],
    ]
    findings, _ = aggregate_justifications(samples, SAMPLED)
    # Two of three is no longer enough to carry the boolean either: booleans take
    # the same bar as the counts, or a condition could survive at one level of
    # agreement and a count at another.
    assert findings[0].is_generic is False
    assert findings[0].specifics_quoted == ["the 11-second clip"]
    assert findings[0].generic is False
    assert "is_generic" not in findings[0].triggered_conditions()

    # The union is what withholds the finding even where the boolean does survive:
    # at the old majority bar two draws carry `is_generic`, and the one draw that
    # quoted a specific still refutes them.
    as_median = dataclasses.replace(
        DEFAULT_POLICY, samples_by_check={450: 3}, justification_draw_agreement=2
    )
    findings, _ = aggregate_justifications(samples, as_median)
    assert findings[0].is_generic is True
    assert findings[0].generic is False
    assert "is_generic" not in findings[0].triggered_conditions()


def test_the_agreement_bar_rescales_a_lost_draw_instead_of_becoming_unsatisfiable():
    """A group that lost a draw must not silently stop being able to fail.

    The bar is stated against the nominal draw count, so "5 of 5" applied literally
    to a four-draw group would be unsatisfiable -- and a bar nothing can clear
    empties the check rather than tightening it. Fourteen of the 238 groups stored
    by the live run came back with fewer than five draws, so this is the observed
    case and not a defensive one.
    """
    five = dataclasses.replace(DEFAULT_POLICY, samples_by_check={450: 5})
    assert [_justification_votes(n, five) for n in (5, 4, 3, 2, 1)] == [5, 4, 3, 2, 1]

    # 3 of the nominal 5 is `median_low` on a full group and on every group that
    # lost draws, which is the property that makes a pre-patch run replayable:
    # 3 of 5, 3 of 4, 2 of 3, 2 of 2.
    at_three = dataclasses.replace(five, justification_draw_agreement=3)
    assert [_justification_votes(n, at_three) for n in (5, 4, 3, 2)] == [3, 3, 2, 2]
    for n in (2, 3, 4, 5):
        values = list(range(n))
        assert _kth_largest(values, _justification_votes(n, at_three)) == median_low(
            values
        )

    # A bar above the nominal count clamps to unanimity rather than to nothing.
    three_draws = dataclasses.replace(five, samples_by_check={450: 3})
    assert _justification_votes(3, three_draws) == 3


def test_an_unquoted_contradiction_a_minority_alleged_is_dropped():
    samples = [
        [_just(contradicts_verdict_claims=2, contradicting_quotes=["it was slow"])],
        [_just()],
        [_just()],
    ]
    findings, _ = aggregate_justifications(samples, SAMPLED)
    assert findings[0].contradicts_verdict_claims == 0
    assert findings[0].contradicts_verdict is False


def test_three_draws_move_450_off_a_fail_the_first_draw_alone_would_have_scored():
    samples = [
        [_just(unsupported_claims=2), _just("A::speed", inaccurate_evidence=1)],
        [_just(unsupported_claims=0), _just("A::speed", inaccurate_evidence=0)],
        [_just(unsupported_claims=0), _just("A::speed", inaccurate_evidence=0)],
    ]
    assert evaluate_check_450(TASK, samples[0], SAMPLED).band == "fail"
    findings, _ = aggregate_justifications(samples, SAMPLED)
    assert evaluate_check_450(TASK, findings, SAMPLED).band == "clean"


def test_a_surviving_sub_threshold_count_still_denies_450_a_clean():
    """A count that survives the median but sits under its condition's threshold
    leaves 450 in non_fail, because `has_any_issue` reads the raw counts rather
    than the triggered conditions.

    This is the ceiling on what sampling alone can do for precision against the
    human sheet: the scorer counts non_fail as a flag, so a check that lands here
    is still a flag however much of the noise was removed.
    """
    samples = [
        [_just(unsupported_claims=2)],
        [_just(unsupported_claims=1)],
        [_just(unsupported_claims=1)],
    ]
    findings, _ = aggregate_justifications(samples, SAMPLED)
    assert findings[0].triggered_conditions() == []
    assert findings[0].has_any_issue() is True
    assert evaluate_check_450(TASK, findings, SAMPLED).band == "non_fail"


def test_a_justification_some_draws_never_mention_is_padded_with_zeroes():
    """Silence about a justification the judge was shown is a judgment that it
    carried nothing, so it must not shrink the denominator into a majority."""
    samples = [[_just(unsupported_claims=5)], [], []]
    findings, _ = aggregate_justifications(samples, SAMPLED)
    assert findings[0].unsupported_claims == 0


def test_even_sample_counts_take_the_low_median():
    """With two draws both must agree; the second sample buys strictness rather
    than accuracy, which is why odd counts are what majority agreement is for."""
    policy = dataclasses.replace(DEFAULT_POLICY, samples_by_check={450: 2})
    findings, _ = aggregate_justifications(
        [[_just(unsupported_claims=4)], [_just(unsupported_claims=0)]], policy
    )
    assert findings[0].unsupported_claims == 0


# ---------------------------------------------------------------------------
# 110 aggregation
# ---------------------------------------------------------------------------


def test_110_drops_an_issue_only_one_draw_reported():
    """110's bar is 'contains any issues' and it has no fail band to absorb a
    mistake, which makes it the shape most exposed to a single noisy draw."""
    samples = [
        KeyTurnJustificationFinding(issues=["overstates the outcome"]),
        KeyTurnJustificationFinding(),
        KeyTurnJustificationFinding(),
    ]
    finding, records = aggregate_key_turn_justification(samples, SAMPLED)
    assert finding.issues == []
    assert finding.is_issue is False
    # The near miss is still visible to a reviewer rather than deleted.
    assert any("overstates the outcome" in r.claim for r in records)


def test_110_keeps_an_issue_a_majority_reported():
    samples = [
        KeyTurnJustificationFinding(issues=["misreads the turn"]),
        KeyTurnJustificationFinding(issues=["describes the wrong turn"]),
        KeyTurnJustificationFinding(),
    ]
    finding, _ = aggregate_key_turn_justification(samples, SAMPLED)
    assert finding.is_issue is True
    assert len(finding.issues) == 2


def test_110_booleans_take_a_majority():
    samples = [
        KeyTurnJustificationFinding(claims_are_accurate=False),
        KeyTurnJustificationFinding(claims_are_accurate=False),
        KeyTurnJustificationFinding(describes_selected_turn=False),
    ]
    finding, _ = aggregate_key_turn_justification(samples, SAMPLED)
    assert finding.claims_are_accurate is False
    assert finding.describes_selected_turn is True


# ---------------------------------------------------------------------------
# Disagreement reporting
# ---------------------------------------------------------------------------


def test_the_stage_reports_which_claims_the_draws_split_on():
    """A judge that splits 2-1 is saying the item is marginal. That is information
    about the evidence and the report carries it rather than discarding it."""

    def responder(request):
        if request.metadata.get("check_id") != 280:
            return {"judgments": [], "confidence": "high"}
        # Only the first draw faults turn 7.
        turns = [7] if request.sample_index == 0 else []
        return {
            "judgments": [{"item_id": "C1", "incorrect_turns": turns}],
            "confidence": "high",
        }

    task = make_task()
    task.criterion_ratings[0].score = 0
    task.criterion_ratings[0].relevant_turns = [7]

    result = run_informed_stage(task, FakeModelClient(responder), SAMPLED, workers=1)
    log = result.agreement
    assert log.dropped_claims, "a claim only one draw made should be recorded as dropped"
    assert log.disagreement_rate > 0
    assert result.to_dict()["agreement"]["dropped_claims"] >= 1
    flagged = [
        f for f in result.relevant_turns if f.check_id == 280 and f.incorrect_turns
    ]
    assert flagged == []


def test_single_sample_stage_reports_no_agreement_data_and_makes_one_call_each():
    client = FakeModelClient(lambda r: {"judgments": [], "confidence": "high"})
    result = run_informed_stage(TASK, client, SINGLE, workers=1)
    assert result.agreement.records == []
    assert result.calls == estimate_informed_calls(TASK, SINGLE)
    assert result.calls == estimate_informed_calls(TASK)


def test_sampled_stage_bills_the_sampled_checks_at_their_sample_count():
    client = FakeModelClient(lambda r: {"judgments": [], "confidence": "high"})
    sampled = run_informed_stage(TASK, client, SAMPLED, workers=1)
    assert sampled.calls == estimate_informed_calls(TASK, SAMPLED)
    assert sampled.calls > estimate_informed_calls(TASK, SINGLE)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_projection_prices_each_check_at_its_own_sample_count():
    """Measured unit costs from the completed run's cache, so the projection is a
    restatement of a real bill rather than a guess."""
    calls = {280: 50, 310: 50, 450: 78, 110: 26, 70: 17}
    unit = {280: 0.1623, 310: 0.1550, 450: 0.1783, 110: 0.0875, 70: 0.0209}
    cost = project_cost(calls, unit, SAMPLED)

    assert cost.single_sample_calls == 221
    # Only the four sampled checks triple; 70 stays at one draw.
    assert cost.sampled_calls == 221 + 2 * (50 + 50 + 78 + 26)
    assert cost.sampled_usd > cost.single_sample_usd
    payload = cost.to_dict()
    assert payload["extra_calls"] == 408
    assert payload["samples_by_check"][70] == 1
    assert payload["samples_by_check"][280] == 3


def test_a_single_sample_configuration_costs_exactly_what_it_did_before():
    calls = {280: 50, 450: 78}
    unit = {280: 0.1623, 450: 0.1783}
    cost = project_cost(calls, unit, SINGLE)
    assert cost.sampled_calls == cost.single_sample_calls
    assert cost.to_dict()["extra_usd"] == 0.0
