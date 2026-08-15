"""Threshold boundaries, counting semantics, denominators, and monotonicity."""

from __future__ import annotations

import random

import pytest

from honeybee_qc.config import DEFAULT_POLICY, Policy
from honeybee_qc.findings import (
    CoverageFinding,
    CriterionFinding,
    CriterionRatingFinding,
    DimensionRatingFinding,
    InversionFinding,
    Issue,
    JustificationFinding,
    LikertFinding,
    RelevantTurnFinding,
    WeightFinding,
)
from honeybee_qc.gates import (
    CensusCorrupted,
    build_census,
    build_inversion_finding,
    dimension_direction,
    evaluate_census_gate,
    evaluate_check_100,
    evaluate_check_200,
    evaluate_check_210,
    evaluate_check_260,
    evaluate_check_270,
    evaluate_check_300,
    evaluate_check_400,
    evaluate_check_450,
    evaluate_check_460,
    evaluate_relevant_turns,
    evaluate_rubric_quality_gates,
    likert_direction,
)
from honeybee_qc.taxonomies import (
    MAJOR_CATEGORIES,
    MINOR_CATEGORIES,
    MODERATE_CATEGORIES,
    severity_of,
)
from honeybee_qc.tests.fixtures import (
    criterion_findings,
    dimension_rating_findings,
    l1_findings,
    make_task,
    weight_findings,
)


# ---------------------------------------------------------------------------
# 200 -- L1 labels
# ---------------------------------------------------------------------------


def test_200_at_exactly_30_percent_is_a_fail():
    task = make_task(n_criteria=10)
    v = evaluate_check_200(task, l1_findings(10, n_incorrect=3))
    assert v.measurement.rate == pytest.approx(0.30)
    assert v.band == "fail"


def test_200_just_below_30_percent_is_a_non_fail():
    task = make_task(n_criteria=20)
    v = evaluate_check_200(task, l1_findings(20, n_incorrect=5))
    assert v.measurement.rate == pytest.approx(0.25)
    assert v.band == "non_fail"


def test_200_single_wrong_label_is_a_non_fail_not_a_dead_zone():
    # 1 of 20 is 5%; a literal "between 1% and 30%" floor must not swallow it.
    task = make_task(n_criteria=20)
    v = evaluate_check_200(task, l1_findings(20, n_incorrect=1))
    assert v.band == "non_fail"


def test_200_all_correct_is_clean():
    task = make_task(n_criteria=20)
    assert evaluate_check_200(task, l1_findings(20, 0)).band == "clean"


def test_200_denominator_excludes_unlabeled_criteria():
    task = make_task(n_criteria=20)
    for c in task.rubric[10:]:
        c.l1_label = None
    # Only the 10 labeled criteria are audited; 3 wrong of 10 is a fail at 30%.
    v = evaluate_check_200(task, l1_findings(10, n_incorrect=3))
    assert v.measurement.denominator == 10
    assert v.measurement.counts["unlabeled_excluded"] == 10
    assert v.band == "fail"


# ---------------------------------------------------------------------------
# 210 -- blocked on the missing appendix
# ---------------------------------------------------------------------------


def test_210_returns_not_evaluated_when_the_l2_enum_is_switched_off():
    """The taxonomy now ships, but the escape hatch has to keep working: scoring
    clean against an enum we cannot see would understate the defect rate."""
    task = make_task()
    v = evaluate_check_210(task, policy=Policy(l2_labels_configured=False))
    assert v.band == "not_evaluated"
    assert v.score is None
    assert v.error_code is None


def test_210_scores_once_the_enum_is_configured():
    task = make_task()
    policy = Policy(l2_labels_configured=True)
    assert evaluate_check_210(task, [], policy).band == "clean"
    assert evaluate_check_210(task, ["C1"], policy).band == "non_fail"


# ---------------------------------------------------------------------------
# 230 / 240 / 250 -- census gates
# ---------------------------------------------------------------------------


def test_230_at_exactly_10_percent_is_a_fail():
    task = make_task(n_criteria=10)
    census = build_census(task, criterion_findings(10, {1: ["inaccurate"]}))
    v = evaluate_census_gate(task, census, 230)
    assert v.measurement.rate == pytest.approx(0.10)
    assert v.band == "fail"


def test_230_just_below_10_percent_is_a_non_fail():
    task = make_task(n_criteria=20)
    census = build_census(task, criterion_findings(20, {1: ["inaccurate"]}))
    v = evaluate_census_gate(task, census, 230)
    assert v.measurement.rate == pytest.approx(0.05)
    assert v.band == "non_fail"


def test_240_at_exactly_15_percent_is_a_fail():
    task = make_task(n_criteria=20)
    census = build_census(
        task,
        criterion_findings(20, {1: ["inaccurate"], 2: ["overlapping"], 3: ["non_atomic"]}),
    )
    v = evaluate_census_gate(task, census, 240)
    assert v.measurement.rate == pytest.approx(0.15)
    assert v.band == "fail"


def test_250_at_exactly_20_percent_is_a_fail():
    task = make_task(n_criteria=20)
    census = build_census(
        task,
        criterion_findings(
            20,
            {
                1: ["inaccurate"],
                2: ["overlapping"],
                3: ["non_atomic"],
                4: ["unnecessary"],
            },
        ),
    )
    v = evaluate_census_gate(task, census, 250)
    assert v.measurement.rate == pytest.approx(0.20)
    assert v.band == "fail"


def test_250_one_increment_below_is_a_non_fail():
    task = make_task(n_criteria=20)
    census = build_census(
        task, criterion_findings(20, {1: ["inaccurate"], 2: ["overlapping"], 3: ["unnecessary"]})
    )
    v = evaluate_census_gate(task, census, 250)
    assert v.measurement.rate == pytest.approx(0.15)
    assert v.band == "non_fail"


def test_clean_census_is_clean_on_all_three_gates():
    task = make_task(n_criteria=20)
    census = build_census(task, criterion_findings(20))
    for v in evaluate_rubric_quality_gates(task, census):
        assert v.band == "clean"
        assert v.error_code is None


def test_criterion_with_three_major_issues_counts_once():
    task = make_task(n_criteria=20)
    findings = criterion_findings(
        20, {1: ["inaccurate", "counterproductive", "missing_critical"]}
    )
    census = build_census(task, findings)
    assert census.numerator(230) == 1
    assert census.criteria_with_major == ["C1"]


def test_criterion_with_major_and_minor_counts_once_in_all_three_gates():
    task = make_task(n_criteria=20)
    census = build_census(task, criterion_findings(20, {1: ["inaccurate", "unnecessary"]}))
    assert census.numerator(230) == 1
    assert census.numerator(240) == 1
    assert census.numerator(250) == 1


def test_mutual_overlap_counts_as_one_error_not_two():
    """Tab 2 row 65: 'Each set of overlapping criteria counts as one error.' Two
    criteria that only report each other must not double the moderate count."""
    task = make_task(n_criteria=20)
    findings = criterion_findings(20)
    findings[0] = CriterionFinding(
        criterion_id="C1",
        issues=[Issue("overlapping", "d", "e", overlaps_with=["C2"])],
    )
    findings[1] = CriterionFinding(
        criterion_id="C2",
        issues=[Issue("overlapping", "d", "e", overlaps_with=["C1"])],
    )
    census = build_census(task, findings)
    assert census.numerator(240) == 1
    assert census.criteria_with_moderate == ["overlap::C1+C2"]


def test_a_criterion_with_an_independent_issue_alongside_overlap_still_counts_on_its_own():
    """The pooling only applies to a *pure* overlap. A criterion that also carries
    its own moderate issue is not folded into the overlap set."""
    task = make_task(n_criteria=20)
    findings = criterion_findings(20)
    findings[0] = CriterionFinding(
        criterion_id="C1",
        issues=[
            Issue("overlapping", "d", "e", overlaps_with=["C2"]),
            Issue("vague_subjective", "d2", "e2"),
        ],
    )
    findings[1] = CriterionFinding(
        criterion_id="C2",
        issues=[Issue("overlapping", "d", "e", overlaps_with=["C1"])],
    )
    census = build_census(task, findings)
    # C1 is counted on its own (it has an independent moderate issue); C2 is a
    # pure overlap with nothing left to pool with once C1 opts out.
    assert sorted(census.criteria_with_moderate) == ["C1", "C2"]
    assert census.numerator(240) == 2


def test_missing_criteria_raise_the_numerator_but_not_the_denominator():
    task = make_task(n_criteria=20)
    census = build_census(
        task,
        criterion_findings(20),
        CoverageFinding(missing_critical=["export as 4K"], missing_non_critical=["loop cleanly"]),
    )
    v230 = evaluate_census_gate(task, census, 230)
    v240 = evaluate_census_gate(task, census, 240)
    assert v230.measurement.denominator == 20
    assert v230.measurement.numerator == 1        # missing_critical is major
    assert v240.measurement.numerator == 2        # plus missing_non_critical
    assert any(i.startswith("missing::") for i in v230.contributing_items)


def test_a_rate_above_100_percent_is_flagged_rather_than_quietly_published():
    # Every criterion defective plus an absent one: the spec's numerator-only rule
    # for missing criteria makes this arithmetically possible.
    task = make_task(n_criteria=3)
    census = build_census(
        task,
        criterion_findings(3, {1: ["inaccurate"], 2: ["inaccurate"], 3: ["inaccurate"]}),
        CoverageFinding(missing_critical=["export at 4K"]),
    )
    v = evaluate_census_gate(task, census, 230)
    assert v.measurement.numerator == 4
    assert v.measurement.denominator == 3
    assert v.measurement.counts["rate_exceeds_100_percent"] is True
    assert "report the counts, not the rate" in v.measurement.notes
    assert v.band == "fail"


def test_counting_absent_criteria_in_the_denominator_keeps_the_rate_in_range():
    task = make_task(n_criteria=3)
    census = build_census(
        task,
        criterion_findings(3, {1: ["inaccurate"], 2: ["inaccurate"], 3: ["inaccurate"]}),
        CoverageFinding(missing_critical=["export at 4K"]),
    )
    v = evaluate_census_gate(task, census, 230, Policy(coverage_in_denominator=True))
    assert v.measurement.denominator == 4
    assert v.measurement.rate == pytest.approx(1.0)
    assert "rate_exceeds_100_percent" not in v.measurement.counts


def test_the_two_denominator_conventions_can_disagree_at_the_boundary():
    # 10 clean criteria and one missing critical: 1/10 fails 230 exactly, while
    # 1/11 does not. Worth knowing before publishing either number.
    task = make_task(n_criteria=10)
    census = build_census(
        task, criterion_findings(10), CoverageFinding(missing_critical=["export at 4K"])
    )
    assert evaluate_census_gate(task, census, 230).band == "fail"
    lenient = evaluate_census_gate(task, census, 230, Policy(coverage_in_denominator=True))
    assert lenient.band == "non_fail"


def test_monotonicity_holds_across_random_censuses():
    rng = random.Random(1234)
    categories = sorted(MAJOR_CATEGORIES | MODERATE_CATEGORIES | MINOR_CATEGORIES)
    for _ in range(300):
        n = rng.randint(1, 30)
        task = make_task(n_criteria=n)
        assignment = {
            i: rng.sample(categories, rng.randint(1, 3))
            for i in range(1, n + 1)
            if rng.random() < 0.5
        }
        census = build_census(task, criterion_findings(n, assignment))
        n230, n240, n250 = (census.numerator(g) for g in (230, 240, 250))
        assert n230 <= n240 <= n250
        nums = [v.measurement.numerator for v in evaluate_rubric_quality_gates(task, census)]
        assert nums == sorted(nums)


def test_census_rejects_a_finding_for_an_unknown_criterion():
    task = make_task(n_criteria=3)
    bogus = [CriterionFinding(criterion_id="C99", issues=[Issue("inaccurate", "d", "e")])]
    with pytest.raises(CensusCorrupted):
        build_census(task, bogus)


def test_severity_membership_matches_the_spec():
    assert severity_of("inaccurate") == "major"
    assert severity_of("counterproductive") == "major"
    assert severity_of("missing_critical") == "major"
    assert severity_of("not_self_contained") == "moderate"
    assert severity_of("framing_double_negative") == "moderate"
    assert severity_of("missing_non_critical") == "moderate"
    assert severity_of("unnecessary") == "minor"
    with pytest.raises(ValueError):
        severity_of("overfitted")  # TARA category with no home in this spec


# ---------------------------------------------------------------------------
# 260 -- weights
# ---------------------------------------------------------------------------


def test_260_two_level_at_exactly_5_percent_is_a_fail():
    task = make_task(n_criteria=20)
    v = evaluate_check_260(task, weight_findings(20, two_level=1))
    assert v.measurement.counts["two_level_rate"] == pytest.approx(0.05)
    assert v.band == "fail"


def test_260_any_level_at_exactly_30_percent_is_a_fail():
    task = make_task(n_criteria=10)
    v = evaluate_check_260(task, weight_findings(10, one_level=3))
    assert v.measurement.rate == pytest.approx(0.30)
    assert v.measurement.counts["two_level_off"] == 0
    assert v.band == "fail"


def test_260_a_few_one_level_misses_is_a_non_fail():
    task = make_task(n_criteria=20)
    v = evaluate_check_260(task, weight_findings(20, one_level=2))
    assert v.measurement.rate == pytest.approx(0.10)
    assert v.band == "non_fail"


def test_260_accurate_weights_are_clean():
    task = make_task(n_criteria=20)
    assert evaluate_check_260(task, weight_findings(20)).band == "clean"


def test_260_same_bucket_disagreement_is_non_fail_but_never_a_fail():
    # A 4 against a band of 5 is outside the band but zero levels under the bucket
    # map, so it cannot reach a fail while still being a weight the definitions do
    # not support.
    task = make_task(n_criteria=20)
    v = evaluate_check_260(task, weight_findings(20, same_bucket_delta=8))
    assert v.measurement.counts["any_level_off"] == 0
    assert v.measurement.counts["out_of_band"] == 8
    assert v.band == "non_fail"


def test_260_ignores_a_weight_inside_the_defensible_band():
    """The complaint that prompted the band: a contributor who records 3 where the
    auditor's own pick is 5 but 2-5 is defensible has done nothing wrong."""
    task = make_task(n_criteria=20)
    v = evaluate_check_260(task, weight_findings(20, inside_band=20))
    assert v.measurement.counts["out_of_band"] == 0
    assert v.measurement.counts["any_level_off"] == 0
    # The point disagreement is still reported, so the band's effect is visible.
    assert v.measurement.counts["differs_from_auditor_pick"] == 20
    assert v.band == "clean"


def test_260_counts_a_weight_outside_the_band_even_when_the_band_is_wide():
    task = make_task(n_criteria=20)
    findings = weight_findings(20, inside_band=20)
    findings[0].contributor_weight = 1   # band is 2-5
    v = evaluate_check_260(task, findings)
    assert v.measurement.counts["out_of_band"] == 1
    assert v.contributing_items == ["C1"]


def test_260_distance_outside_the_band_not_from_the_pick_sets_the_level():
    """One level outside a band whose auditor pick is two levels away still counts
    once: the gate measures the distance the definitions cannot cover."""
    task = make_task(n_criteria=10)
    one_off = WeightFinding(
        criterion_id="C1",
        contributor_weight=1,
        auditor_weight=5,
        defensible_low=3,
        defensible_high=5,
    )
    two_off = WeightFinding(
        criterion_id="C2",
        contributor_weight=1,
        auditor_weight=5,
        defensible_low=4,
        defensible_high=5,
    )
    assert one_off.points_outside == 2 and one_off.nearest_defensible == 3
    v = evaluate_check_260(task, [one_off] + weight_findings(9))
    assert v.measurement.counts["any_level_off"] == 1
    assert v.measurement.counts["two_level_off"] == 0

    v = evaluate_check_260(task, [two_off] + weight_findings(9))
    assert v.measurement.counts["two_level_off"] == 1


def test_260_an_out_of_scale_weight_is_still_maximally_wrong():
    """Live rubrics carry negative weights; no band can support one."""
    task = make_task(n_criteria=20)
    findings = weight_findings(20, inside_band=20)
    findings[0].contributor_weight = -5
    v = evaluate_check_260(task, findings)
    assert v.measurement.counts["two_level_off"] == 1
    assert v.band == "fail"


def test_260_an_auditor_pick_outside_its_own_band_widens_the_band():
    """A response that contradicts itself must not manufacture a defect."""
    finding = WeightFinding(
        criterion_id="C1",
        contributor_weight=5,
        auditor_weight=5,
        defensible_low=1,
        defensible_high=3,
    )
    assert finding.band == (1, 5)
    assert finding.in_band
    task = make_task(n_criteria=1)
    assert evaluate_check_260(task, [finding]).band == "clean"


def test_260_bucket_map_comes_from_the_customer_weight_guide():
    task = make_task(n_criteria=20)
    v = evaluate_check_260(task, weight_findings(20))
    assert "low=[1, 2]" in v.measurement.notes
    assert "medium=[3]" in v.measurement.notes
    assert "high=[4, 5]" in v.measurement.notes


# ---------------------------------------------------------------------------
# 270 -- rubric evaluation
# ---------------------------------------------------------------------------


def _ratings(
    n_criteria: int, n_disagreeing: int = 0, models_disagreeing: int = 1
) -> list[CriterionRatingFinding]:
    """Both models rate every criterion. The first `n_disagreeing` criteria are
    rated differently by `models_disagreeing` of the two."""
    out: list[CriterionRatingFinding] = []
    for i in range(1, n_criteria + 1):
        for position, slot in enumerate(("A", "B")):
            wrong = i <= n_disagreeing and position < models_disagreeing
            out.append(
                CriterionRatingFinding(
                    criterion_id=f"C{i}",
                    model=slot,  # type: ignore[arg-type]
                    contributor_score=1,
                    auditor_score=0 if wrong else 1,
                )
            )
    return out


def test_270_at_exactly_the_specs_twenty_percent_of_criteria_is_a_fail():
    """"20% or more of the criteria". 4 of 20 criteria, with the absolute leg
    silent at 4 < 5, so this isolates the rate."""
    task = make_task(n_criteria=20)
    v = evaluate_check_270(task, _ratings(20, n_disagreeing=4))
    assert (v.measurement.numerator, v.measurement.denominator) == (4, 20)
    assert v.measurement.rate == pytest.approx(0.20)
    assert v.band == "fail"


def test_270_just_under_twenty_percent_of_criteria_is_a_non_fail():
    task = make_task(n_criteria=25)
    v = evaluate_check_270(task, _ratings(25, n_disagreeing=4))
    assert v.measurement.rate == pytest.approx(0.16)
    assert v.band == "non_fail"


def test_270_just_over_twenty_percent_of_criteria_is_a_fail():
    task = make_task(n_criteria=19)
    v = evaluate_check_270(task, _ratings(19, n_disagreeing=4))
    assert v.measurement.rate == pytest.approx(4 / 19)
    assert v.band == "fail"


def test_270_a_criterion_both_models_disagree_on_counts_once():
    """The spec's fail legs are stated in criteria while its notes say to count
    "across all models", so the criterion is the unit and the second model's
    judgment cannot double it. Reported alongside, never counted twice."""
    task = make_task(n_criteria=20)
    v = evaluate_check_270(task, _ratings(20, n_disagreeing=4, models_disagreeing=2))
    assert v.measurement.numerator == 4
    assert v.measurement.denominator == 20
    assert v.measurement.counts["disagreeing_judgments"] == 8
    assert v.measurement.counts["judgments"] == 40
    assert v.measurement.rate == pytest.approx(0.20)
    assert v.band == "fail"


def test_270_denominator_is_criteria_not_judgments():
    """The regression this pins: with judgments (criteria x models) underneath,
    4 disagreements in a 20-criterion rubric read as 10% and landed in the
    non-fail band, where the spec puts them at 20% and fails the task."""
    task = make_task(n_criteria=20)
    v = evaluate_check_270(task, _ratings(20, n_disagreeing=4))
    assert v.measurement.denominator == 20
    assert v.measurement.rate == pytest.approx(0.20)
    assert v.band == "fail"


def test_270_the_absolute_leg_binds_where_the_rate_does_not():
    """"5 or more criteria" -- 5 of 40 is 12.5%, below the rate threshold."""
    task = make_task(n_criteria=40)
    v = evaluate_check_270(task, _ratings(40, n_disagreeing=5))
    assert v.measurement.numerator == 5
    assert v.measurement.rate == pytest.approx(0.125)
    assert v.band == "fail"


def test_270_one_disagreement_in_a_large_rubric_is_a_non_fail():
    task = make_task(n_criteria=40)
    v = evaluate_check_270(task, _ratings(40, n_disagreeing=1))
    assert v.band == "non_fail"


def test_270_no_disagreements_is_clean():
    task = make_task(n_criteria=20)
    v = evaluate_check_270(task, _ratings(20))
    assert v.measurement.denominator == 20
    assert v.band == "clean"


# ---------------------------------------------------------------------------
# 280 / 310 -- relevant turns
# ---------------------------------------------------------------------------


def _turn_findings(check_id, n_items, incorrect_per_item=1, missing=0):
    out = []
    for i in range(n_items):
        out.append(
            RelevantTurnFinding(
                check_id=check_id,
                item_id=f"item{i}",
                model="A",
                incorrect_turns=list(range(incorrect_per_item)),
            )
        )
    for i in range(missing):
        out.append(
            RelevantTurnFinding(
                check_id=check_id, item_id=f"missing{i}", model="A", missing_turn=True
            )
        )
    return out


@pytest.mark.parametrize("check_id", [280, 310])
def test_relevant_turns_at_exactly_three_is_a_fail(check_id):
    task = make_task()
    v = evaluate_relevant_turns(task, check_id, _turn_findings(check_id, 3))
    assert v.measurement.numerator == 3
    assert v.band == "fail"


@pytest.mark.parametrize("check_id", [280, 310])
def test_relevant_turns_at_two_is_a_non_fail(check_id):
    task = make_task()
    v = evaluate_relevant_turns(task, check_id, _turn_findings(check_id, 2))
    assert v.band == "non_fail"


@pytest.mark.parametrize("check_id", [280, 310])
def test_relevant_turns_all_correct_is_clean(check_id):
    task = make_task()
    assert evaluate_relevant_turns(task, check_id, []).band == "clean"


def test_280_and_310_findings_never_leak_into_each_other():
    task = make_task()
    mixed = _turn_findings(280, 3) + _turn_findings(310, 1)
    v280 = evaluate_relevant_turns(task, 280, mixed)
    v310 = evaluate_relevant_turns(task, 310, mixed)
    assert (v280.measurement.numerator, v280.band) == (3, "fail")
    assert (v310.measurement.numerator, v310.band) == (1, "non_fail")
    assert all(i.startswith("280::") for i in v280.contributing_items)
    assert all(i.startswith("310::") for i in v310.contributing_items)


def test_both_models_incorrect_turns_survive_in_the_per_item_breakdown():
    """Model A and Model B both rate a dimension called "Communication quality",
    so the two findings share an `item_id`. The count must sum both (it always
    did), and the breakdown dict must keep both too, rather than the second
    model silently overwriting the first's turn numbers under the same key."""
    task = make_task()
    findings = [
        RelevantTurnFinding(check_id=310, item_id="Communication quality", model="A",
                             incorrect_turns=[18]),
        RelevantTurnFinding(check_id=310, item_id="Communication quality", model="B",
                             incorrect_turns=[1]),
    ]
    v = evaluate_relevant_turns(task, 310, findings)
    assert v.measurement.numerator == 2
    turns_by_item = v.measurement.counts["turns_by_item"]
    all_turns = [t for turns in turns_by_item.values() for t in turns]
    assert sorted(all_turns) == [1, 18]


def test_missing_turns_are_counted_as_incorrect_under_their_own_category():
    task = make_task()
    findings = _turn_findings(280, 0, missing=3)
    v = evaluate_relevant_turns(task, 280, findings)
    assert v.measurement.counts["missing_turns"] == 3
    assert v.band == "fail"
    assert all("missing_turn" in i for i in v.contributing_items)


def test_missing_turns_can_be_excluded_by_policy_without_a_rerun():
    task = make_task()
    policy = Policy(count_missing_turns_as_incorrect=False)
    v = evaluate_relevant_turns(task, 280, _turn_findings(280, 0, missing=3), policy)
    assert v.measurement.counts["missing_turns"] == 3
    assert v.band == "clean"


# ---------------------------------------------------------------------------
# 300 -- dimension ratings
# ---------------------------------------------------------------------------


def test_300_at_exactly_two_major_is_a_fail():
    task = make_task()
    v = evaluate_check_300(task, dimension_rating_findings(n_major=2))
    assert v.measurement.counts["major_disagreements"] == 2
    assert v.band == "fail"


def test_300_one_major_plus_three_minor_is_a_non_fail():
    task = make_task()
    v = evaluate_check_300(task, dimension_rating_findings(n_major=1, n_minor=3))
    assert v.measurement.counts["major_disagreements"] == 1
    assert v.measurement.counts["total_disagreements"] == 4
    assert v.band == "non_fail"


def test_300_five_total_minor_is_a_fail():
    task = make_task()
    v = evaluate_check_300(task, dimension_rating_findings(n_minor=5))
    assert v.measurement.counts["major_disagreements"] == 0
    assert v.measurement.counts["total_disagreements"] == 5
    assert v.band == "fail"


def test_300_bands_are_exhaustive_with_no_gap_or_overlap():
    task = make_task()
    for major in range(0, 3):
        for minor in range(0, 6):
            if major + minor > 12:
                continue
            v = evaluate_check_300(task, dimension_rating_findings(n_major=major, n_minor=minor))
            total = major + minor
            if major >= 2 or total >= 5:
                assert v.band == "fail", (major, minor)
            elif total >= 1:
                assert v.band == "non_fail", (major, minor)
            else:
                assert v.band == "clean", (major, minor)


def test_300_matching_ratings_are_clean():
    task = make_task()
    assert evaluate_check_300(task, dimension_rating_findings()).band == "clean"


def test_300_denominator_excludes_dimensions_both_sides_marked_na():
    task = make_task()
    v = evaluate_check_300(task, dimension_rating_findings(n_both_na=4))
    assert v.measurement.denominator == 8
    assert v.band == "clean"


def test_300_na_mismatch_is_a_disagreement_not_a_numeric_delta():
    task = make_task()
    findings = [
        DimensionRatingFinding(
            model="A",
            dimension="Memory & personalization",
            contributor_na=True,
            auditor_na=False,
            auditor_rating=6,
        )
    ]
    v = evaluate_check_300(task, findings)
    assert v.measurement.counts["na_mismatches"] == 1
    # Reported but not counted: the spec puts an applicability divergence in the
    # "no issues" cell, so it is not a defect under the default policy.
    assert v.measurement.counts["total_disagreements"] == 0
    assert v.band == "clean"

    lenient = evaluate_check_300(task, findings, Policy(na_mismatch_counts_as="minor"))
    assert lenient.measurement.counts["total_disagreements"] == 1
    assert lenient.band == "non_fail"

    strict = evaluate_check_300(task, findings, Policy(na_mismatch_counts_as="major"))
    assert strict.measurement.counts["major_disagreements"] == 1


# ---------------------------------------------------------------------------
# 400 -- ranking disagreement
# ---------------------------------------------------------------------------


def test_400_delta_three_is_a_fail_and_delta_two_is_not():
    task = make_task()
    assert evaluate_check_400(task, LikertFinding(2, 5)).band == "fail"
    assert evaluate_check_400(task, LikertFinding(2, 4)).band == "non_fail"


def test_400_agreement_is_clean():
    task = make_task()
    v = evaluate_check_400(task, LikertFinding(4, 4))
    assert v.band == "clean"
    assert v.measurement.numerator == 0


def test_400_uses_raw_deltas_not_bucket_agreement():
    # 2 and 3 sit in different tab-1 buckets but are only one point apart, so the
    # stricter raw-delta reading from tab 2 gives a non-fail rather than agreement.
    task = make_task()
    assert evaluate_check_400(task, LikertFinding(2, 3)).band == "non_fail"


# ---------------------------------------------------------------------------
# 450 -- justifications
# ---------------------------------------------------------------------------


def test_450_one_bad_primary_claim_fails_but_one_bad_secondary_does_not():
    task = make_task()
    primary = JustificationFinding(item_id="A::Outcome quality", inaccurate_primary_claims=1)
    secondary = JustificationFinding(item_id="A::Outcome quality", inaccurate_secondary_claims=1)
    assert evaluate_check_450(task, [primary]).band == "fail"
    assert evaluate_check_450(task, [secondary]).band == "non_fail"


def test_450_two_bad_secondary_claims_fail():
    task = make_task()
    f = JustificationFinding(item_id="x", inaccurate_secondary_claims=2)
    assert evaluate_check_450(task, [f]).band == "fail"


def test_450_one_inaccurate_quote_fails_but_one_misread_quote_does_not():
    task = make_task()
    fabricated = JustificationFinding(item_id="x", inaccurate_evidence=1)
    misread = JustificationFinding(item_id="x", misconstrued_evidence=1)
    assert evaluate_check_450(task, [fabricated]).band == "fail"
    assert evaluate_check_450(task, [misread]).band == "non_fail"


def test_450_thresholds_are_counted_within_a_single_justification():
    task = make_task()
    one_bad = JustificationFinding(item_id="one", unsupported_claims=2)
    assert evaluate_check_450(task, [one_bad]).band == "fail"

    spread = [
        JustificationFinding(item_id="one", unsupported_claims=1),
        JustificationFinding(item_id="two", unsupported_claims=1),
    ]
    assert evaluate_check_450(task, spread).band == "non_fail"


def test_450_pooled_scope_changes_the_outcome():
    task = make_task()
    spread = [
        JustificationFinding(item_id="one", unsupported_claims=1),
        JustificationFinding(item_id="two", unsupported_claims=1),
    ]
    pooled = evaluate_check_450(task, spread, Policy(justification_scope="pooled"))
    assert pooled.band == "fail"


def test_450_generic_praise_fails_regardless_of_the_rating():
    task = make_task()
    f = JustificationFinding(item_id="A::Outcome quality", rated_value=5, is_generic=True)
    v = evaluate_check_450(task, [f])
    assert v.band == "fail"
    assert "is_generic" in v.measurement.counts["conditions_triggered"]


def test_450_clean_justifications_are_clean():
    task = make_task()
    findings = [JustificationFinding(item_id=f"j{i}") for i in range(13)]
    v = evaluate_check_450(task, findings)
    assert v.band == "clean"
    assert v.measurement.counts["justifications_audited"] == 13


# The Trust & grounding justification from task 6a7190c542368ece68aa26a0, which
# the informed pass reported on while another justification on the same task
# tripped is_generic. It names two real platforms and a fabricated domain, so
# whatever else is wrong with it, there is something here to quote.
TRUST_AND_GROUNDING = (
    "There were many mistakes where it claimed apps were for students and then had "
    "to correct itself. It also created a dead link. For example it said Athletes "
    "Untapped and TeachMeTo were for 18+. It also created the fake link for the "
    '"debatemarket.com" which does not exist.'
)


def test_450_a_justification_naming_specifics_is_never_generic():
    task = make_task()
    f = JustificationFinding(
        item_id="A::Trust & grounding",
        rated_value=2,
        is_generic=True,
        specifics_quoted=["Athletes Untapped and TeachMeTo", '"debatemarket.com"'],
    )
    assert f.generic is False
    assert "is_generic" not in f.triggered_conditions()
    assert evaluate_check_450(task, [f]).band == "clean"


def test_450_vacuous_praise_with_nothing_to_quote_is_generic():
    task = make_task()
    f = JustificationFinding(
        item_id="A::Outcome quality",
        rated_value=5,
        is_generic=True,
        specifics_quoted=[],
    )
    assert f.generic is True
    v = evaluate_check_450(task, [f])
    assert v.band == "fail"
    assert v.measurement.counts["conditions_by_item"] == {
        "A::Outcome quality": ["is_generic"]
    }


def test_450_blank_specifics_do_not_rescue_a_generic_justification():
    f = JustificationFinding(item_id="x", is_generic=True, specifics_quoted=["", "  "])
    assert f.generic is True


def test_450_brevity_alone_is_not_an_issue():
    task = make_task()
    f = JustificationFinding(
        item_id="A::Trust & grounding", specifics_quoted=['"debatemarket.com"']
    )
    v = evaluate_check_450(task, [f])
    assert v.band == "clean"
    assert v.measurement.counts["justifications_with_issues"] == 0


def test_450_unverifiable_claims_are_reported_but_never_counted():
    task = make_task()
    f = JustificationFinding(item_id="A::Outcome quality", unverifiable_claims=3)
    v = evaluate_check_450(task, [f])
    assert v.band == "clean"
    assert v.measurement.counts["unverifiable_claims"] == 3


def test_450_conditions_are_attributed_to_the_justification_that_tripped_them():
    task = make_task()
    findings = [
        JustificationFinding(item_id="A::Communication quality", is_generic=True),
        JustificationFinding(
            item_id="A::Trust & grounding",
            specifics_quoted=["Athletes Untapped"],
            misconstrued_evidence=1,
        ),
    ]
    v = evaluate_check_450(task, findings)
    counts = v.measurement.counts
    assert counts["conditions_by_item"] == {"A::Communication quality": ["is_generic"]}
    assert "A::Trust & grounding" not in counts["conditions_by_item"]


def _justifications(total: int, failing: int) -> list[JustificationFinding]:
    """`failing` justifications tripping is_generic, the rest clean."""
    return [
        JustificationFinding(item_id=f"j{i}", is_generic=i < failing)
        for i in range(total)
    ]


def test_450_one_bad_justification_among_fifteen_does_not_fail_the_task():
    """The mechanism this gate was fixed for.

    Task 6a7190c542368ece68aa2723 had exactly one of fifteen justifications trip
    a condition and was failed for it, alongside 6a7190c542368ece68aa2694 whose
    twelve of fifteen failed. One bad argument in fifteen is the middle band.
    """
    task = make_task()
    v = evaluate_check_450(task, _justifications(15, 1))
    assert v.band == "non_fail"
    assert v.measurement.counts["failing_justifications"] == 1
    assert v.measurement.rate == pytest.approx(1 / 15)


def test_450_fails_once_the_share_reaches_the_threshold():
    task = make_task()
    # 5 of 15 is exactly 33%, the first reachable rate at or above 30%.
    assert evaluate_check_450(task, _justifications(15, 4)).band == "non_fail"
    assert evaluate_check_450(task, _justifications(15, 5)).band == "fail"


def test_450_the_threshold_is_inclusive_like_every_other_rate_gate():
    task = make_task()
    assert evaluate_check_450(task, _justifications(10, 3)).band == "fail"


def test_450_the_share_is_taken_over_the_population_not_a_fixed_count():
    """Proportional, not a disguised absolute: the same three failing
    justifications decide differently in a small population and a large one."""
    task = make_task()
    assert evaluate_check_450(task, _justifications(6, 3)).band == "fail"
    assert evaluate_check_450(task, _justifications(15, 3)).band == "non_fail"


def test_450_sub_threshold_triggers_are_still_reported_as_non_clean():
    """A demotion from fail to non-fail must not become a demotion to clean.

    The QC scorer counts a non-fail as a flag exactly like a fail, so this is
    what keeps the check's recall against the human sheet where it was.
    """
    task = make_task()
    v = evaluate_check_450(task, _justifications(15, 2))
    assert v.band == "non_fail"
    assert v.measurement.counts["conditions_by_item"] == {
        "j0": ["is_generic"],
        "j1": ["is_generic"],
    }
    assert v.contributing_items == ["j0", "j1"]


def test_450_reports_the_threshold_as_ours_rather_than_the_specs():
    """Every other threshold in this module is quoted from the spec. This one is
    not, and a published measurement has to say so on its face."""
    task = make_task()
    v = evaluate_check_450(task, _justifications(15, 5))
    assert v.measurement.threshold == DEFAULT_POLICY.justification_fail_rate
    assert v.measurement.counts["fail_threshold_is_ours"] is True
    assert "not stated by the spec" in v.measurement.notes


def test_450_pooled_scope_keeps_its_any_trigger_fail():
    """Pooling collapses the task to one synthetic justification, so a share of
    the population is not available to gate on and one-over-fifteen would
    silence the check rather than calibrate it."""
    task = make_task()
    spread = [
        JustificationFinding(item_id="one", unsupported_claims=1),
        *(JustificationFinding(item_id=f"j{i}") for i in range(13)),
        JustificationFinding(item_id="two", unsupported_claims=1),
    ]
    v = evaluate_check_450(task, spread, Policy(justification_scope="pooled"))
    assert v.band == "fail"
    assert v.measurement.counts["fail_threshold_is_ours"] is False
    assert v.measurement.threshold is None


def test_450_a_lone_justification_that_trips_is_all_of_the_population():
    """The rate is the whole point: one failing justification out of one is
    100%, not 6.7%, and the ranking justification audited alone must still
    be able to fail."""
    task = make_task()
    v = evaluate_check_450(task, [JustificationFinding(item_id="ranking", is_generic=True)])
    assert v.band == "fail"
    assert v.measurement.rate == 1.0


def test_450_an_empty_population_is_clean_and_does_not_divide_by_zero():
    task = make_task()
    v = evaluate_check_450(task, [])
    assert v.band == "clean"
    assert v.measurement.rate == 0.0


# ---------------------------------------------------------------------------
# 460 -- inconsistent ranking
# ---------------------------------------------------------------------------


def test_likert_direction_buckets():
    assert likert_direction(1) == "A"
    assert likert_direction(2) == "A"
    assert likert_direction(3) == "neutral"
    assert likert_direction(5) == "neutral"
    assert likert_direction(6) == "B"
    assert likert_direction(7) == "B"


def test_dimension_direction_from_contributor_ratings():
    assert dimension_direction(make_task(dimension_rating_a=8, dimension_rating_b=4)) == "A"
    assert dimension_direction(make_task(dimension_rating_a=4, dimension_rating_b=8)) == "B"
    assert dimension_direction(make_task(dimension_rating_a=6, dimension_rating_b=6)) == "neutral"


def test_460_fails_only_on_an_unexplained_inversion():
    task = make_task(dimension_rating_a=8, dimension_rating_b=4, likert=7)
    finding = build_inversion_finding(task)
    assert finding.dimension_direction == "A"
    assert finding.likert_direction == "B"
    assert finding.contradicts
    assert evaluate_check_460(task, finding).band == "fail"


def test_460_does_not_fail_when_the_justification_explains_the_inversion():
    task = make_task(dimension_rating_a=8, dimension_rating_b=4, likert=7)
    finding = build_inversion_finding(task, justification_explains_inversion=True)
    assert finding.contradicts
    assert evaluate_check_460(task, finding).band == "clean"


def test_460_neutral_likert_with_a_mild_lean_is_not_a_contradiction():
    task = make_task(dimension_rating_a=7, dimension_rating_b=6, likert=4)
    finding = build_inversion_finding(task)
    assert finding.dimension_direction == "A"
    assert finding.likert_direction == "neutral"
    assert not finding.contradicts
    assert evaluate_check_460(task, finding).band == "clean"


def test_460_agreeing_directions_are_clean():
    task = make_task(dimension_rating_a=8, dimension_rating_b=4, likert=1)
    assert evaluate_check_460(task, build_inversion_finding(task)).band == "clean"


def test_460_cannot_emit_a_middle_band():
    task = make_task()
    finding = InversionFinding(dimension_direction="A", likert_direction="B")
    v = evaluate_check_460(task, finding)
    assert v.band in ("fail", "clean")


# ---------------------------------------------------------------------------
# 100 -- key turn
# ---------------------------------------------------------------------------


def test_100_turn_one_fails_regardless_of_any_other_reasoning():
    task = make_task(key_turn=1)
    # Even when the auditor independently agrees turn 1 is where value lands.
    v = evaluate_check_100(task, auditor_key_turn=1)
    assert v.band == "fail"
    assert v.error_code == "[Fail - First Turn is Key Turn]"


def test_100_disagreement_is_only_a_non_fail():
    task = make_task(key_turn=5)
    v = evaluate_check_100(task, auditor_key_turn=9)
    assert v.band == "non_fail"
    assert v.error_code == "[Non-Fail - Misidentified Key Turn]"


def test_100_agreement_is_clean():
    task = make_task(key_turn=5)
    assert evaluate_check_100(task, auditor_key_turn=5).band == "clean"


def test_100_without_an_independent_judgment_is_not_evaluated():
    task = make_task(key_turn=5)
    assert evaluate_check_100(task).band == "not_evaluated"
