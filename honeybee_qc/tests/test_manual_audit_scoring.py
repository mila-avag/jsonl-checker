"""Scoring a re-run against a manual audit's per-(task, check) ground truth."""

from __future__ import annotations

from honeybee_qc.score_against_manual_audit import (
    compare,
    index_bands,
    scored_checks,
    unguarded_true_fails,
)

GT = {
    "_comment": "not a check id",
    "300": {
        "FALSE_FAIL": ["false-fail-task"],
        "TRUE_FAIL": ["true-fail-task"],
        "BORDERLINE": ["borderline-task"],
    },
}


def bands(*entries: tuple[str, int, str]) -> dict[tuple[str, int], dict]:
    """Index a report built from (task_id, check_id, band) triples."""
    tasks: dict[str, list[dict]] = {}
    for task_id, check_id, band in entries:
        tasks.setdefault(task_id, []).append({"check_id": check_id, "band": band})
    return index_bands(
        {"tasks": [{"task_id": t, "checks": c} for t, c in tasks.items()]}
    )


def outcomes(before, after, gt=GT, checks=None) -> dict[str, str]:
    return {r.task_id: r.outcome for r in compare(gt, before, after, checks)}


ALL_FAILING = bands(
    ("false-fail-task", 300, "fail"),
    ("true-fail-task", 300, "fail"),
    ("borderline-task", 300, "fail"),
)


def test_a_false_fail_that_stopped_failing_is_a_fix():
    after = bands(
        ("false-fail-task", 300, "clean"),
        ("true-fail-task", 300, "fail"),
        ("borderline-task", 300, "fail"),
    )
    assert outcomes(ALL_FAILING, after)["false-fail-task"] == "fixed"


def test_a_false_fail_that_kept_failing_is_not_a_fix():
    assert outcomes(ALL_FAILING, ALL_FAILING)["false-fail-task"] == "still_failing"


def test_a_true_fail_that_stopped_failing_is_a_regression():
    after = bands(
        ("false-fail-task", 300, "clean"),
        ("true-fail-task", 300, "clean"),
        ("borderline-task", 300, "fail"),
    )
    rows = compare(GT, ALL_FAILING, after)
    assert {r.task_id: r.outcome for r in rows}["true-fail-task"] == "regressed"
    assert [r.task_id for r in rows if r.outcome == "regressed"] == ["true-fail-task"]


def test_a_true_fail_still_failing_is_held():
    assert outcomes(ALL_FAILING, ALL_FAILING)["true-fail-task"] == "held"


def test_an_abstaining_check_counts_as_no_longer_failing():
    # not_evaluated is not a pass, but it is not this build catching the defect
    # either -- so it reads as a fix on a false fail and, more importantly, as a
    # regression on a true one rather than being quietly credited as held.
    after = bands(
        ("false-fail-task", 300, "not_evaluated"),
        ("true-fail-task", 300, "not_evaluated"),
        ("borderline-task", 300, "fail"),
    )
    scored = outcomes(ALL_FAILING, after)
    assert scored["false-fail-task"] == "fixed"
    assert scored["true-fail-task"] == "regressed"


def test_a_borderline_is_scored_in_neither_column_whichever_way_it_moved():
    # The audit judged these defensible either way; counting one as a fix or a
    # regression would read a preference into evidence that did not support one.
    for band in ("fail", "clean", "non_fail"):
        after = bands(
            ("false-fail-task", 300, "fail"),
            ("true-fail-task", 300, "fail"),
            ("borderline-task", 300, band),
        )
        assert outcomes(ALL_FAILING, after)["borderline-task"] == "unscored"


def test_a_task_the_rerun_did_not_cover_is_not_reported_as_fixed():
    # A narrowed or chunked re-run carries only some of the audited tasks. The
    # cheapest way to show every false fail fixed is to run none of them.
    after = bands(("true-fail-task", 300, "fail"))
    scored = outcomes(ALL_FAILING, after)
    assert scored["false-fail-task"] == "missing"
    assert scored["borderline-task"] == "missing"


def test_a_true_fail_the_rerun_skipped_leaves_the_regression_guard_incomplete():
    after = bands(("false-fail-task", 300, "clean"))
    rows = compare(GT, ALL_FAILING, after)
    assert not [r for r in rows if r.outcome == "regressed"]
    assert [r.task_id for r in unguarded_true_fails(rows)] == ["true-fail-task"]


def test_a_true_fail_that_was_not_failing_in_the_before_report_is_not_counted_as_held():
    # Ground truth written against a different run, or the wrong --before file:
    # there is no fail here to have held, and calling it held would manufacture
    # a passing regression guard out of a mismatch.
    before = bands(
        ("false-fail-task", 300, "fail"),
        ("true-fail-task", 300, "clean"),
        ("borderline-task", 300, "fail"),
    )
    rows = compare(GT, before, ALL_FAILING)
    scored = {r.task_id: r.outcome for r in rows}
    assert scored["true-fail-task"] == "not_failing_before"
    assert [r.task_id for r in unguarded_true_fails(rows)] == ["true-fail-task"]


def test_only_the_checks_asked_for_are_scored():
    gt = dict(GT, **{"400": {"TRUE_FAIL": ["true-fail-task"]}})
    rows = compare(gt, ALL_FAILING, ALL_FAILING, ["300"])
    assert {r.check_id for r in rows} == {"300"}


def test_ground_truth_keys_that_are_not_check_ids_are_ignored():
    assert scored_checks(GT) == ["300"]
    assert scored_checks({"_comment": "x", "400": {}, "85": {}}) == ["85", "400"]
