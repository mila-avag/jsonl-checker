"""Replaying a deterministic check from snapshots, offline."""

from __future__ import annotations

from pathlib import Path

from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.replay_deterministic import (
    NO_SNAPSHOT,
    REPLAYS,
    Replay,
    SnapshotOnlyFetcher,
    outcome_for,
    snapshot_name,
)
from honeybee_qc.sources import classify_link
from honeybee_qc.tests.fixtures import GEMINI, make_task
from honeybee_qc.transcripts import TranscriptTurn

NOWHERE = SnapshotOnlyFetcher(Path("/nonexistent-snapshot-dir"))


def test_every_replayable_check_reaches_a_band_without_a_model_call():
    # Each adapter calls into gates/links with a signature that has to keep
    # matching; a replay that raised here would otherwise only be discovered by
    # someone reaching for a free re-score during an expensive run.
    for check_id, replay_check in sorted(REPLAYS.items()):
        replay = replay_check(make_task(), DEFAULT_POLICY, NOWHERE)
        assert replay.band, f"check {check_id} produced no band"


def test_a_check_that_could_not_finish_offline_is_never_scored_as_a_fix():
    # 100 abstains for any key turn but 1, because the rest needs the auditor's
    # own pick. Reading that abstention as "no longer failing" would report a
    # fix for every task the replay could not actually judge.
    replay = REPLAYS[100](make_task(key_turn=5), DEFAULT_POLICY, NOWHERE)
    assert replay.band == "not_evaluated"
    assert outcome_for("FALSE_FAIL", replay) == "unreplayable"
    assert outcome_for("TRUE_FAIL", replay) == "unreplayable"


def test_the_structural_leg_of_100_still_replays():
    replay = REPLAYS[100](make_task(key_turn=1), DEFAULT_POLICY, NOWHERE)
    assert replay.band == "fail"
    assert outcome_for("TRUE_FAIL", replay) == "held"


def test_85_is_unreplayable_when_the_file_half_had_nothing_to_compare():
    task = make_task()
    for submission in (task.model_a, task.model_b):
        submission.attachments = []
    replay = REPLAYS[85](task, DEFAULT_POLICY, NOWHERE)

    assert replay.blocked
    # The band is "unverified, so not failing" rather than "checked and clean",
    # which is exactly the value that must not be counted as a false fail fixed.
    assert replay.band != "fail"
    assert outcome_for("FALSE_FAIL", replay) == "unreplayable"


def test_a_replay_scores_both_directions():
    assert outcome_for("FALSE_FAIL", Replay(band="clean")) == "fixed"
    assert outcome_for("FALSE_FAIL", Replay(band="fail")) == "still_failing"
    assert outcome_for("TRUE_FAIL", Replay(band="fail")) == "held"
    assert outcome_for("TRUE_FAIL", Replay(band="clean")) == "regressed"


def test_a_borderline_is_scored_in_neither_column():
    assert outcome_for("BORDERLINE", Replay(band="clean")) == "unscored"
    assert outcome_for("BORDERLINE", Replay(band="fail")) == "unscored"


def test_a_missing_snapshot_is_reported_rather_than_fetched():
    transcript = NOWHERE(GEMINI)
    assert transcript.error == NO_SNAPSHOT
    assert not transcript.ok
    assert not transcript.turns


def test_an_invalid_share_url_is_rejected_before_any_lookup():
    transcript = NOWHERE("not-a-share-url")
    assert "not a valid share URL" in transcript.error
    assert not transcript.ok


def test_a_snapshot_on_disk_is_read_from_the_name_the_pipeline_wrote(tmp_path, monkeypatch):
    provider = classify_link(GEMINI).provider
    (tmp_path / snapshot_name(provider, GEMINI)).write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        "honeybee_qc.replay_deterministic.parse_share_html",
        lambda html, prov: ([TranscriptTurn(index=1, role="user", text="hi")], "hi", False),
    )

    transcript = SnapshotOnlyFetcher(tmp_path)(GEMINI)

    assert transcript.turns
    assert transcript.snapshot_path.endswith(snapshot_name(provider, GEMINI))


def test_a_share_page_the_archive_shows_as_gone_becomes_a_check_90_reason(tmp_path, monkeypatch):
    task = make_task()
    url = task.model_a.final_link
    provider = classify_link(url).provider
    (tmp_path / snapshot_name(provider, url)).write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        "honeybee_qc.replay_deterministic.parse_share_html",
        lambda html, prov: ([], "", True),
    )

    fetcher = SnapshotOnlyFetcher(tmp_path)
    fetcher(url)

    assert fetcher.dead_links(task) == ["final_link_A:share page unavailable"]


def test_a_link_never_fetched_is_not_reported_dead():
    # Absence of evidence: a snapshot we never opened says nothing about whether
    # the page is gone, and check 90 fails a task outright on a dead link.
    assert NOWHERE.dead_links(make_task()) == []
