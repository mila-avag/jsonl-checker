"""Shape enforcement and error-code integrity."""

from __future__ import annotations

import pytest

from honeybee_qc.errors import ERROR_CODES, all_codes, error_code
from honeybee_qc.registry import ORDER, REGISTRY
from honeybee_qc.scoring import SHAPE_BANDS, ShapeViolation, build_verdict
from honeybee_qc.tests.test_spec_conformance import NET_NEW_CHECK_IDS, SPEC_CHECK_IDS

BINARY = (85, 90, 95, 96, 220, 460, 470, 1000)
NON_FAIL_ONLY = (80, 110, 210)
GRADED = (70, 75, 100, 200, 230, 240, 250, 260, 270, 280, 300, 310, 400, 450)


def test_registry_covers_the_spec_dimensions_plus_the_declared_net_new_ones():
    # Counted off the two declarations in test_spec_conformance rather than
    # restated, so adding a net-new dimension is a one-line change there.
    assert len(REGISTRY) == len(SPEC_CHECK_IDS) + len(NET_NEW_CHECK_IDS)
    assert ORDER == tuple(sorted(REGISTRY))


def test_shape_assignment_matches_spec():
    for cid in BINARY:
        assert REGISTRY[cid].shape == "A", cid
    for cid in NON_FAIL_ONLY:
        assert REGISTRY[cid].shape == "B", cid
    for cid in GRADED:
        assert REGISTRY[cid].shape == "C", cid
    # Every registered dimension is classified, so a new one cannot slip in
    # unshaped.
    assert set(BINARY) | set(NON_FAIL_ONLY) | set(GRADED) == set(ORDER)


@pytest.mark.parametrize("check_id", BINARY)
@pytest.mark.parametrize("band", ["non_fail"])
def test_binary_checks_cannot_emit_the_middle_band(check_id, band):
    with pytest.raises(ShapeViolation):
        build_verdict(task_id="t", check_id=check_id, band=band)


@pytest.mark.parametrize("check_id", NON_FAIL_ONLY)
def test_non_fail_only_checks_cannot_fail_a_task(check_id):
    with pytest.raises(ShapeViolation):
        build_verdict(task_id="t", check_id=check_id, band="fail")


@pytest.mark.parametrize("check_id", GRADED)
def test_graded_checks_support_all_three_bands(check_id):
    for band in ("fail", "non_fail", "clean"):
        v = build_verdict(task_id="t", check_id=check_id, band=band)
        assert v.band == band


def test_no_check_ever_emits_score_two():
    for cid in ORDER:
        for band in SHAPE_BANDS[REGISTRY[cid].shape]:
            v = build_verdict(task_id="t", check_id=cid, band=band)
            assert v.score != 2


def test_band_to_score_mapping():
    v_fail = build_verdict(task_id="t", check_id=230, band="fail")
    v_non = build_verdict(task_id="t", check_id=230, band="non_fail")
    v_clean = build_verdict(task_id="t", check_id=230, band="clean")
    v_ne = build_verdict(task_id="t", check_id=230, band="not_evaluated")
    assert (v_fail.score, v_non.score, v_clean.score, v_ne.score) == (1, 3, 5, None)


def test_clean_band_emits_no_error_code():
    assert error_code(230, "clean") is None
    assert build_verdict(task_id="t", check_id=230, band="clean").error_code is None


def test_not_evaluated_emits_no_error_code():
    assert error_code(210, "not_evaluated") is None


def test_every_emitted_code_is_in_the_table():
    known = set(all_codes())
    for cid in ORDER:
        for band in SHAPE_BANDS[REGISTRY[cid].shape]:
            v = build_verdict(task_id="t", check_id=cid, band=band)
            if v.error_code is not None:
                assert v.error_code in known


def test_error_code_table_shape_matches_registry():
    for cid, bands in ERROR_CODES.items():
        allowed = SHAPE_BANDS[REGISTRY[cid].shape]
        for band in bands:
            assert band in allowed, f"check {cid} has a code for unavailable band {band}"


def test_stray_plus_in_240_non_fail_code_is_preserved():
    # Malformed in the source spreadsheet. It is the downstream join key, so
    # "fixing" it would break matching against the customer's own tooling.
    assert ERROR_CODES[240]["non_fail"] == "[Non-Fail - < 15%+ Major/Moderate Rubric Errors]"
    assert "%+" in ERROR_CODES[240]["non_fail"]


def test_280_and_310_share_both_strings():
    assert ERROR_CODES[280] == ERROR_CODES[310]


def test_the_code_table_has_exactly_two_duplicate_strings():
    """280 and 310 share both of their strings and nothing else may. A third
    duplicate would mean two dimensions publishing the same label without the
    namespacing that keeps 280 and 310 apart."""
    codes = all_codes()
    assert len(codes) - len(set(codes)) == 2


def test_1000_code_is_a_sentence_not_a_bracketed_tag():
    assert ERROR_CODES[1000]["fail"] == "This task is unauditable"
    assert not ERROR_CODES[1000]["fail"].startswith("[")


def test_unknown_band_for_check_raises():
    with pytest.raises(ValueError):
        error_code(90, "non_fail")
