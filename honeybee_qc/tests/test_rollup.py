"""Task roll-up, output validation, and batch metrics."""

from __future__ import annotations

import pytest

from honeybee_qc.config import Policy
from honeybee_qc.rollup import build_batch_report, roll_up_task, validate_verdicts
from honeybee_qc.scoring import build_verdict


def v(check_id: int, band: str):
    return build_verdict(task_id="t", check_id=check_id, band=band)


def test_any_fail_fails_the_task():
    r = roll_up_task("t", [v(230, "fail"), v(240, "clean"), v(90, "clean")])
    assert r.verdict == "fail"
    assert r.fail_checks == [230]


def test_all_clean_is_clean():
    r = roll_up_task("t", [v(90, "clean"), v(230, "clean"), v(470, "clean")])
    assert r.verdict == "clean"


def test_non_fail_issues_give_pass_with_issues():
    r = roll_up_task("t", [v(90, "clean"), v(230, "non_fail")])
    assert r.verdict == "pass_with_issues"
    assert r.non_fail_checks == [230]


def test_unauditable_short_circuits_every_other_result():
    r = roll_up_task("t", [v(1000, "fail"), v(230, "fail"), v(90, "fail")])
    assert r.verdict == "unauditable"


def test_not_evaluated_does_not_deny_clean_by_default():
    r = roll_up_task("t", [v(90, "clean"), v(210, "not_evaluated")])
    assert r.verdict == "clean"
    assert r.not_evaluated_checks == [210]


def test_not_evaluated_can_be_made_to_block_clean():
    policy = Policy(not_evaluated_blocks_clean=True)
    r = roll_up_task("t", [v(90, "clean"), v(210, "not_evaluated")], policy)
    assert r.verdict == "pass_with_issues"


def test_missing_links_fail_90_without_marking_the_task_unauditable():
    r = roll_up_task("t", [v(90, "fail"), v(1000, "clean")])
    assert r.verdict == "fail"
    assert r.verdict != "unauditable"


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def test_a_clean_verdict_carrying_an_error_code_is_rejected():
    bad = v(230, "clean")
    bad.error_code = "[Fail - 10%+ Major Rubric Errors]"
    with pytest.raises(ValueError, match="clean but carries an error code"):
        validate_verdicts([bad])


def test_a_fail_without_an_error_code_is_rejected():
    bad = v(230, "fail")
    bad.error_code = None
    with pytest.raises(ValueError, match="has no error code"):
        validate_verdicts([bad])


def test_an_unrecognised_error_code_is_rejected():
    bad = v(230, "fail")
    bad.error_code = "[Fail - Ten Percent Major Errors]"
    with pytest.raises(ValueError, match="unknown error code"):
        validate_verdicts([bad])


def test_score_two_is_rejected():
    bad = v(230, "non_fail")
    bad.score = 2
    with pytest.raises(ValueError, match="score 2"):
        validate_verdicts([bad])


def test_an_unknown_check_id_is_rejected():
    bad = v(230, "clean")
    bad.check_id = 999
    with pytest.raises(ValueError, match="unknown check_id"):
        validate_verdicts([bad])


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------


def test_unauditable_tasks_still_exclude_their_own_not_evaluated_checks():
    # In practice a task made unauditable by an empty rubric never produces a
    # real 230 verdict -- the rubric stage has nothing to grade -- so it
    # reports not_evaluated same as any other task missing that stage.
    good = roll_up_task("good", [v(230, "fail"), v(1000, "clean")])
    spam = roll_up_task("spam", [v(1000, "fail"), v(230, "not_evaluated")])
    report = build_batch_report([good, spam])

    assert report.task_counts["total"] == 2
    assert report.task_counts["unauditable"] == 1
    assert report.task_counts["auditable"] == 1

    rate_230 = next(r for r in report.check_rates if r.check_id == 230)
    assert rate_230.denominator == 1
    assert rate_230.fails == 1
    assert rate_230.fail_rate == 1.0
    assert rate_230.not_evaluated == 1


def test_a_check_that_ran_despite_the_unauditable_flag_still_counts():
    # An empty rubric blocks the rubric-dependent checks, but not a check
    # like 400 (the blind SxS pick) that never reads the rubric at all -- it
    # still gets a real verdict and should still land in that check's
    # denominator, even though the task overall stays "unauditable".
    good = roll_up_task("good", [v(400, "clean"), v(1000, "clean")])
    spam = roll_up_task("spam", [v(400, "fail"), v(1000, "fail")])
    report = build_batch_report([good, spam])

    assert report.task_counts["unauditable"] == 1
    rate_400 = next(r for r in report.check_rates if r.check_id == 400)
    assert rate_400.denominator == 2
    assert rate_400.fails == 1
    assert rate_400.fail_rate == 0.5


def test_check_1000_is_kept_out_of_the_per_check_quality_rates():
    r = roll_up_task("t", [v(1000, "clean"), v(230, "clean")])
    report = build_batch_report([r])
    assert all(rate.check_id != 1000 for rate in report.check_rates)


def test_not_evaluated_is_excluded_from_a_check_denominator():
    a = roll_up_task("a", [v(210, "not_evaluated")])
    b = roll_up_task("b", [v(210, "not_evaluated")])
    report = build_batch_report([a, b])
    rate = next(r for r in report.check_rates if r.check_id == 210)
    assert rate.denominator == 0
    assert rate.not_evaluated == 2
    assert rate.fail_rate == 0.0


def test_error_code_distribution_is_reported():
    a = roll_up_task("a", [v(230, "fail")])
    b = roll_up_task("b", [v(230, "fail")])
    c = roll_up_task("c", [v(230, "non_fail")])
    report = build_batch_report([a, b, c])
    assert report.error_code_distribution["[Fail - 10%+ Major Rubric Errors]"] == 2
    assert report.error_code_distribution["[Non-Fail - < 10% Major Rubric Errors]"] == 1


def test_gate_rate_distribution_keeps_the_underlying_rates():
    # A batch failing the 10% gate at 11% is a different conversation from one
    # failing at 40%, so the rates are preserved, not just the pass/fail split.
    from honeybee_qc.scoring import Measurement

    barely = build_verdict(
        task_id="a", check_id=230, band="fail", measurement=Measurement(rate=0.11)
    )
    badly = build_verdict(
        task_id="b", check_id=230, band="fail", measurement=Measurement(rate=0.40)
    )
    report = build_batch_report(
        [roll_up_task("a", [barely]), roll_up_task("b", [badly])]
    )
    assert report.gate_rate_distribution[230] == [0.11, 0.40]


def test_report_records_the_active_policy_for_reproducibility():
    report = build_batch_report([roll_up_task("t", [v(230, "clean")])])
    assert report.policy["policy_version"]
    assert report.policy["dimension_rating_scale"] == (1, 5)
    assert report.to_dict()["policy"]["rubric_eval_fail_rate"] == 0.20


# ---------------------------------------------------------------------------
# Findings the confirming passes suppressed
# ---------------------------------------------------------------------------
#
# Checks 300, 400 and 95 each now require an informed second call before a finding
# is published. Without a batch-level tally of what that removed, a reader
# comparing this run's fail rate against an earlier one cannot tell a real
# improvement in the contributors' work from findings this build stopped counting.


def _measured(task_id: str, check_id: int, band: str, counts: dict):
    from honeybee_qc.scoring import Measurement

    return build_verdict(
        task_id=task_id,
        check_id=check_id,
        band=band,
        measurement=Measurement(counts=counts),
    )


def test_the_batch_reports_what_each_confirming_pass_took_out_of_the_fail_rate():
    verdicts = [
        _measured(
            "t1",
            300,
            "clean",
            {
                "blind_only_not_counted": 3,
                "truncated_evidence_not_counted": 2,
                "unadjudicated_not_counted": 1,
                "blind_disagreement_only": True,
            },
        ),
        _measured(
            "t1", 400, "clean",
            {"not_counted_because": "asymmetric_render", "blind_disagreement_only": True},
        ),
        _measured(
            "t1", 95, "clean",
            {"cleared_by_confirmation": ["A:shot.png (renamed)", "B:x.py (not_a_file)"]},
        ),
    ]
    report = build_batch_report([roll_up_task("t1", verdicts)]).to_dict()

    assert report["suppressed_findings"] == {
        "300_blind_only_rating_gaps": 3,
        "300_truncated_evidence_rating_gaps": 2,
        "300_unadjudicated_rating_gaps": 1,
        "400_ranking_gap_asymmetric_render": 1,
        "95_cleared_by_confirmation": 2,
    }
    assert report["blind_only_tasks"] == {300: ["t1"], 400: ["t1"]}


def test_a_batch_with_nothing_suppressed_reports_nothing():
    """The tally must not manufacture keys for checks that confirmed everything."""
    verdicts = [
        _measured("t1", 300, "fail", {"counted_disagreements": 4, "blind_disagreement_only": False}),
        _measured("t1", 400, "fail", {"not_counted_because": "", "blind_disagreement_only": False}),
        _measured("t1", 95, "fail", {"cleared_by_confirmation": []}),
    ]
    report = build_batch_report([roll_up_task("t1", verdicts)]).to_dict()

    assert report["suppressed_findings"] == {}
    assert report["blind_only_tasks"] == {}
