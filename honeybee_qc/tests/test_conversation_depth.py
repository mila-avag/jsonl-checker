"""Exchanges fetched against exchanges corroborated, and its three consumers.

Two tasks in the 53-task human-validated batch were failed by a QC auditor for a
share link that opens and serves one exchange of twenty, and passed by the
pipeline. On both, every rater honestly abstained on what it could not see, the
abstentions left the denominator, and nothing anywhere compared how much came
back against how much the rest of the task says was there.

The cases here are the two real ones and, at greater length, the cases that look
like them and must not be convicted: this audit's live precision is 30%, so the
boundary tests are the load-bearing half of the file.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.context import render_conversation
from honeybee_qc.hydrate import hydrate_conversations
from honeybee_qc.links import evaluate_check_90, measure_conversation_depth
from honeybee_qc.models import Turn
from honeybee_qc.transcripts import Transcript, TranscriptTurn

from .fixtures import CLAUDE, GEMINI, GPT, make_conversation, make_task


def _lopsided_task(
    *,
    short_link: str = CLAUDE,
    peer_exchanges: int = 21,
    fetched: int = 1,
    cited: tuple[int, ...] = (2, 8, 19, 20),
    key_turn: int | None = 5,
):
    """Task 6a7190c542368ece68aa26b8, in miniature.

    Model A's Gemini link returned 21 exchanges; Model B's Claude link returned
    one, while the contributor's Model B ratings cite content from turns 2
    through 20.
    """
    task = make_task(exchanges=peer_exchanges, key_turn=key_turn)
    task.model_a.final_link = GEMINI
    task.model_b.final_link = short_link
    task.model_b.conversation = make_conversation(fetched, prefix="B: ")
    for rating in task.dimension_ratings:
        if rating.model == "B":
            rating.relevant_turns = list(cited)
    return task


def _depth(task, slot: str = "B"):
    sub = task.model_a if slot == "A" else task.model_b
    return measure_conversation_depth(task, sub)


# ---------------------------------------------------------------------------
# Defect 1 -- check 90 had no concept of an incomplete link
# ---------------------------------------------------------------------------


def test_a_link_serving_one_exchange_of_twenty_one_is_measured_as_incomplete():
    depth = _depth(_lopsided_task())
    assert depth.fetched == 1
    assert depth.corroborated == 21
    assert depth.cited_beyond == [2, 5, 8, 19, 20]
    assert depth.link_incomplete


def test_90_records_the_shortfall_and_convicts_on_it_by_default():
    """Measuring and publishing stayed separable while the band was being
    validated. It fired on exactly the two submissions a human failed for this
    and nowhere else across the 53-task batch, so the default now convicts, and
    a caller that only wants the number passes `fail_on_incomplete=False`."""
    verdict = evaluate_check_90(_lopsided_task())
    assert verdict.band == "fail"
    assert verdict.measurement.counts["incomplete_links"] == 1
    assert verdict.measurement.counts["exchanges_fetched"] == {"A": 21, "B": 1}
    assert "1 of 21 exchanges retrieved" in verdict.measurement.notes


def test_90_fails_on_an_incomplete_link_once_the_caller_asks_it_to():
    verdict = evaluate_check_90(_lopsided_task(), fail_on_incomplete=True)
    assert verdict.band == "fail"
    assert verdict.error_code == "[Fail - Missing/Invalid Links]"
    assert any("final_link_B:incomplete" in item for item in verdict.contributing_items)


def test_a_structurally_valid_link_that_served_everything_is_marked_complete():
    from honeybee_qc.links import build_link_report

    report = build_link_report(make_task(exchanges=20))
    assert [c.completeness for c in (report.final_a, report.final_b)] == [
        "complete",
        "complete",
    ]


# ---------------------------------------------------------------------------
# Defect 2 -- the exported-PDF rescue was gated on zero replies
# ---------------------------------------------------------------------------


def _pdf_policy(**overrides):
    return dataclasses.replace(DEFAULT_POLICY, fetch_attachment_text=True, **overrides)


def _fake_pdf(monkeypatch, text: str = "the contributor's own export of all nineteen turns"):
    import honeybee_qc.context as context

    read: list[str] = []

    def fetch(url, policy, mime_type=""):
        read.append(url)
        return text if url.endswith(".pdf") else ""

    monkeypatch.setattr(context, "fetch_attachment_text", fetch)
    return read


def test_a_one_exchange_capture_of_a_twenty_turn_conversation_pulls_the_pdf(monkeypatch):
    """Task 6a7f7ca67216429b95c72149: one assistant turn was enough to disable
    the rescue, and the contributor's 19-turn export sat unread while four of
    that model's justifications were abstained on for want of it."""
    _fake_pdf(monkeypatch)
    sub = _lopsided_task().model_b
    sub.transcript_pdf = "https://example.test/chat.pdf"
    sub.attachments = []
    sub.corroborated_exchanges = 20

    rendered, profile = render_conversation(sub, _pdf_policy())

    assert profile.assistant_turns == 1
    assert profile.materially_short
    assert profile.pdf_fallback_used
    assert "all nineteen turns" in rendered


def test_the_rescue_still_fires_when_no_replies_came_back_at_all(monkeypatch):
    """The old trigger is a subset of the new one, not a case it drops."""
    _fake_pdf(monkeypatch)
    sub = _lopsided_task().model_b
    sub.conversation = [Turn(index=1, role="user", text="hello")]
    sub.transcript_pdf = "https://example.test/chat.pdf"
    sub.attachments = []

    _, profile = render_conversation(sub, _pdf_policy())

    assert profile.corroborated_exchanges == 0
    assert profile.pdf_fallback_used


# ---------------------------------------------------------------------------
# Defect 3 -- a short capture counted as hydrated and said nothing
# ---------------------------------------------------------------------------


def _transcript(exchanges: int) -> Transcript:
    turns: list[TranscriptTurn] = []
    for i in range(1, exchanges + 1):
        turns.append(TranscriptTurn(index=i, role="user", text=f"ask {i}"))
        turns.append(TranscriptTurn(index=i, role="assistant", text=f"answer {i}"))
    return Transcript(kind="link", source_id="s", provider="claude", turns=turns)


def test_hydration_reports_a_capture_that_came_back_a_twentieth_of_its_length():
    task = _lopsided_task()

    def fetch(url: str) -> Transcript:
        return _transcript(21 if url == GEMINI else 1)

    report = hydrate_conversations([task], fetch=fetch)

    assert report.hydrated == 2
    assert len(report.short_captures) == 1
    assert "task-1/B" in report.short_captures[0]
    assert "1 of 21 exchanges retrieved" in report.short_captures[0]


def test_hydration_stamps_the_corroborated_length_for_the_renderer_to_read():
    task = _lopsided_task()

    def fetch(url: str) -> Transcript:
        return _transcript(21 if url == GEMINI else 1)

    hydrate_conversations([task], fetch=fetch)

    assert task.model_b.corroborated_exchanges == 21
    assert task.model_a.corroborated_exchanges == 21


# ---------------------------------------------------------------------------
# The boundary: what must NOT be called an incomplete link
# ---------------------------------------------------------------------------


def test_a_shortfall_on_both_sides_is_our_fetcher_and_is_not_charged_to_anyone():
    """The single most important case in this file.

    Two short captures in one run have one thing in common and it is us. Check
    96 already declines to charge the contributor here; so must 90, or the fix
    for two misses buys a class of false fails that is larger than both.
    """
    task = _lopsided_task(peer_exchanges=1)
    task.estimated_turns = 20

    depth = _depth(task)

    assert depth.materially_short
    assert not depth.peer_healthy
    assert not depth.link_incomplete
    assert evaluate_check_90(task, fail_on_incomplete=True).band == "clean"


def test_a_short_chatgpt_capture_is_our_scraper_before_it_is_the_link():
    """`sources.py` records that ChatGPT virtualises its message list, so the
    served DOM holds only what is near the top of the viewport and the payload
    path that recovers the rest degrades to that same scrape when it will not
    decode. The ambiguity is real and resolves in the contributor's favour."""
    depth = _depth(_lopsided_task(short_link=GPT))
    assert depth.materially_short
    assert not depth.link_incomplete


def test_a_capture_with_no_replies_at_all_is_a_fetch_failure_not_a_link_defect():
    """A login wall, an interstitial, or a marketing page parses as a user turn
    and nothing else. That is our fetch hitting something, and its remedy is the
    exported transcript rather than a finding against the contributor."""
    task = _lopsided_task()
    task.model_b.conversation = [Turn(index=1, role="user", text="opening ask")]

    depth = _depth(task)

    assert depth.materially_short
    assert not depth.fetch_looks_healthy
    assert not depth.link_incomplete


def test_nothing_the_contributor_wrote_corroborates_a_longer_conversation():
    """Without a citation past the fetched end, the only claim that the
    conversation ran longer is our inference from the other slot. That is check
    96's question -- was the work too short -- and not this one's."""
    task = _lopsided_task(cited=(1,), key_turn=1)

    depth = _depth(task)

    assert depth.cited_beyond == []
    assert not depth.link_incomplete


def test_a_capture_that_is_merely_shorter_than_the_other_side_is_left_alone():
    """Turn counts disagree for honest reasons -- a reply still streaming when
    the page was archived, an exchange the pairing heuristic merged, a
    contributor numbering from a print. Those cost a turn or two."""
    task = _lopsided_task(fetched=18, cited=(19, 20))

    depth = _depth(task)

    assert depth.missing == 3
    assert not depth.materially_short
    assert not depth.link_incomplete


def test_a_capture_just_the_wrong_side_of_the_ratio_is_still_left_alone():
    task = _lopsided_task(peer_exchanges=20, fetched=6, cited=(7, 15))
    assert not _depth(task).materially_short

    convicted = _lopsided_task(peer_exchanges=20, fetched=4, cited=(7, 15))
    assert _depth(convicted).link_incomplete


def test_a_short_conversation_is_check_96s_business_and_not_a_link_finding():
    """One exchange of six is a shortfall worth reading the export over, and it
    is not a link finding: the other side is itself under the seven exchanges
    the spec calls a conversation, so it witnesses nothing about our fetcher.
    Failing the link here would charge the contributor twice for one fact, once
    at 96 for a short conversation and once at 90 for the link that shows it."""
    task = _lopsided_task(peer_exchanges=6, fetched=1, cited=(2, 6))
    depth = _depth(task)
    assert depth.missing == 5
    assert depth.materially_short
    assert not depth.peer_healthy
    assert not depth.link_incomplete


def test_the_declared_turn_count_alone_never_convicts_a_link():
    """`estimated_turns` is what the task was commissioned for, recorded before
    anyone ran it. It corroborates a length; it does not witness one."""
    task = _lopsided_task(peer_exchanges=1, cited=())
    task.estimated_turns = 20
    task.key_turn.turn_index = None

    depth = _depth(task)

    assert depth.corroborated == 20
    assert depth.materially_short
    assert not depth.link_incomplete


def test_a_modest_shortfall_does_not_re_buy_the_task_by_pulling_a_pdf(monkeypatch):
    """The rescue changes the conversation fed to every prompt on the task, so
    it re-buys every call the task makes. It has to stay rare."""
    read = _fake_pdf(monkeypatch)
    sub = _lopsided_task(fetched=18).model_b
    sub.transcript_pdf = "https://example.test/chat.pdf"
    sub.attachments = []
    sub.corroborated_exchanges = 21

    _, profile = render_conversation(sub, _pdf_policy())

    assert not profile.pdf_fallback_used
    assert read == []
