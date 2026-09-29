"""Regressions for the evidence-gathering path, all drawn from live data.

Every case here is something that silently produced an empty or truncated
conversation against real tasks, which is the worst failure mode this pipeline
has: the audit still returns a verdict, but it reached it without the evidence.
"""

from __future__ import annotations

import dataclasses
import json
import os

from ..config import DEFAULT_POLICY
from ..gates import dimension_direction
from ..hydrate import hydrate_conversations
from ..models import DimensionRating
from ..sources import is_challenge_page, parse_share_html
from ..transcripts import Transcript, TranscriptTurn
from .fixtures import make_task


# ---------------------------------------------------------------------------
# 460 -- which model the dimension ratings actually favour
# ---------------------------------------------------------------------------


def _task_with_dimension_scores(pairs: dict[str, tuple[int, int]]):
    """A task whose dimension ratings are exactly the supplied A/B pairs."""
    task = make_task()
    ratings = []
    for dimension, (a, b) in pairs.items():
        for slot, score in (("A", a), ("B", b)):
            ratings.append(
                DimensionRating(
                    model=slot,  # type: ignore[arg-type]
                    dimension=dimension,
                    rating=score,
                    not_applicable=False,
                    justification=f"{slot} scored {score}",
                    relevant_turns=[3],
                )
            )
    task.dimension_ratings = ratings
    return task


def test_dominance_ignores_a_split_decision():
    """Each model ahead somewhere supports either verdict, so neither direction wins.

    Task 6a7190c542368ece68aa2694 looks like this and the plain mean called it
    for A by 0.14 of a point, manufacturing a contradiction that QC did not find.
    """
    task = _task_with_dimension_scores(
        {
            "Outcome quality": (3, 4),
            "Communication quality": (3, 3),
            "Interaction efficiency": (4, 3),
            "Tool & connector reliability": (4, 3),
        }
    )
    assert dimension_direction(task) == "neutral"


def test_dominance_calls_a_clean_sweep():
    """Wins on three dimensions and loses none: this is 6a7190c542368ece68aa27f6."""
    task = _task_with_dimension_scores(
        {
            "Outcome quality": (4, 4),
            "Communication quality": (4, 3),
            "Tool & connector reliability": (4, 3),
            "Collaboration quality": (4, 3),
        }
    )
    assert dimension_direction(task) == "A"


def test_plain_mean_remains_available_and_differs():
    """The old basis is still selectable, and the split decision is why it changed."""
    task = _task_with_dimension_scores(
        {
            "Outcome quality": (3, 4),
            "Interaction efficiency": (4, 3),
            "Tool & connector reliability": (4, 3),
        }
    )
    mean_policy = dataclasses.replace(DEFAULT_POLICY, ranking_direction_basis="plain_mean")
    assert dimension_direction(task, mean_policy) == "A"
    assert dimension_direction(task, DEFAULT_POLICY) == "neutral"


def test_dominance_ignores_dimensions_only_one_side_was_rated_on():
    task = _task_with_dimension_scores({"Outcome quality": (4, 3)})
    task.dimension_ratings.append(
        DimensionRating(
            model="A",
            dimension="Memory & personalization",
            rating=5,
            not_applicable=False,
            justification="A only",
            relevant_turns=[1],
        )
    )
    assert dimension_direction(task) == "A"


# ---------------------------------------------------------------------------
# ChatGPT share pages: the DOM is virtualised, the router payload is complete
# ---------------------------------------------------------------------------


def _router_stream_html(messages: list[tuple[str, str]]) -> str:
    """Build a share page whose payload encodes `messages`, as ChatGPT serves it.

    The encoding is React Router's: one flat table where every value is an index
    into that table and object keys are themselves `_<index>` references.
    """
    flat: list = []

    def add(value) -> int:
        flat.append(value)
        return len(flat) - 1

    add(None)  # slot 0 is the root, patched at the end
    key_loader = add("loaderData")
    key_linear = add("linear_conversation")
    key_message = add("message")
    key_author = add("author")
    key_role = add("role")
    key_content = add("content")
    key_parts = add("parts")

    node_refs = []
    for role, text in messages:
        role_ref = add(role)
        text_ref = add(text)
        parts_ref = add([text_ref])
        author_ref = add({f"_{key_role}": role_ref})
        content_ref = add({f"_{key_parts}": parts_ref})
        message_ref = add({f"_{key_author}": author_ref, f"_{key_content}": content_ref})
        node_refs.append(add({f"_{key_message}": message_ref}))

    conversation_ref = add(node_refs)
    loader_ref = add({f"_{key_linear}": conversation_ref})
    flat[0] = {f"_{key_loader}": loader_ref}

    payload = json.dumps(json.dumps(flat))
    return (
        "<html><body><div>rendered chrome</div>"
        f"<script>window.__reactRouterContext.streamController.enqueue({payload})</script>"
        "</body></html>"
    )


def test_gpt_payload_recovers_messages_the_dom_never_rendered():
    html = _router_stream_html(
        [
            ("user", "Search common used marketplaces for Alveo FPGA cards."),
            ("assistant", "I found several credible listings."),
            ("user", "Narrow it to 1 million LUTs or more."),
            ("assistant", "Here is the filtered comparison."),
        ]
    )
    turns, _, dead = parse_share_html(html, "gpt")
    assert not dead
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0].text.startswith("Search common used marketplaces")
    assert [t.index for t in turns] == [1, 1, 2, 2]


def test_gpt_turn_numbers_mean_the_same_thing_as_every_other_providers():
    """The payload is the only source that arrives as a flat message list.

    Numbering it per message makes turn 7 mean one thing for a GPT submission and
    another for the Gemini submission it is compared against, which silently
    breaks every check that reads a cited turn number.
    """
    exchanges = [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]
    gpt, _, _ = parse_share_html(_router_stream_html(exchanges), "gpt")
    assert [(t.index, t.role) for t in gpt] == [
        (1, "user"), (1, "assistant"), (2, "user"), (2, "assistant")
    ]


def test_a_reply_delivered_in_several_parts_is_one_turn():
    """GPT splits a long answer across messages; numbering each one separately
    would inflate the turn count and shift every later citation."""
    html = _router_stream_html(
        [
            ("user", "explain it"),
            ("assistant", "part one of the answer"),
            ("assistant", "part two of the answer"),
            ("user", "thanks"),
            ("assistant", "you are welcome"),
        ]
    )
    turns, _, _ = parse_share_html(html, "gpt")
    assert [t.index for t in turns] == [1, 1, 1, 2, 2]


def test_gpt_payload_drops_tool_and_empty_messages():
    html = _router_stream_html(
        [
            ("user", "do the thing"),
            ("tool", "web.search({})"),
            ("assistant", ""),
            ("assistant", "done"),
        ]
    )
    turns, _, _ = parse_share_html(html, "gpt")
    assert [(t.role, t.text) for t in turns] == [("user", "do the thing"), ("assistant", "done")]


def test_gpt_falls_back_to_the_dom_when_there_is_no_payload():
    html = (
        '<div data-message-author-role="user">hello</div>'
        '<div data-message-author-role="assistant">hi there</div>'
    )
    turns, _, _ = parse_share_html(html, "gpt")
    assert len(turns) >= 2


# ---------------------------------------------------------------------------
# Bot challenges must never be mistaken for a conversation
# ---------------------------------------------------------------------------


def test_challenge_page_is_recognised():
    assert is_challenge_page(
        "<html><head><title>Just a moment...</title></head><body></body></html>"
    )
    assert is_challenge_page("<html><body>Enable JavaScript and cookies to continue</body></html>")
    assert not is_challenge_page("<html><head><title>Compare Alveo Deals</title></head></html>")


def test_geminis_deleted_share_dialog_is_recognised_as_a_dead_page():
    """Gemini renders a gone share as a dialog over an otherwise normal page.

    Observed on five links in one batch: each answered HTTP 200 and redirected to a
    plausible `/share/<id>` URL, so neither the status nor the URL betrays it. Left
    unrecognised the page parses as "text we could not turn into turns", which reads
    as our parser failing rather than as a share the contributor deleted -- and a
    deleted share is check 90's business, not a fetch problem to retry.
    """
    html = (
        "<html><body><h1 class='dialog-title-element'>Link doesn't exist</h1>"
        "<a class='dialog-text'>The link might have been deleted or not entered "
        "correctly.</a><button>OK</button></body></html>"
    )
    turns, full_text, dead = parse_share_html(html, "gemini")
    assert turns == []
    assert dead is True
    assert "might have been deleted" in full_text


def test_a_live_page_is_never_called_dead_just_because_a_dialog_word_appears():
    """The dead-page markers are only consulted when no turns were recovered.

    A conversation in hand outranks any phrase on the page, which is what keeps a
    marker list from turning a working share into a fail.
    """
    html = (
        "<html><body><user-query-content>is the link doesn't exist error normal?"
        "</user-query-content><message-content>Yes, that appears when a share is "
        "deleted.</message-content></body></html>"
    )
    turns, _, dead = parse_share_html(html, "gemini")
    assert len(turns) == 2
    assert dead is False


def test_claude_current_assistant_markup_parses():
    html = (
        '<div data-testid="user-message">what is the plan</div>'
        '<div class="font-claude-response relative leading-[1.65rem]">here is the plan</div>'
    )
    turns, _, _ = parse_share_html(html, "claude")
    assert [t.role for t in turns] == ["user", "assistant"]


def test_a_cited_source_reads_as_linked_rather_than_merely_named():
    """Anchor text survived extraction and the anchor did not, so a linked
    citation and a bare mention of the same name were the same string.

    On task `6a7d3a48a9485b6a031cb308` four of five draws stated positively that a
    turn offered "no link, no source" where the page carries a Sources section of
    embedded links, and QC failed the task for that inaccuracy. The host, not the
    URL: what the claim turns on is whether anything was linked at all, and a
    search-citation URL costs several times what its host does.
    """
    html = (
        '<div data-testid="user-message">where did that number come from</div>'
        '<div class="font-claude-response">Pricing is per million tokens. '
        '<a href="https://www.cloudzero.com/blog/llm-costs/">CloudZero</a> '
        'It has not changed since June.</div>'
    )
    turns, _, _ = parse_share_html(html, "claude")

    assert "CloudZero [link: cloudzero.com]" in turns[1].text


def test_in_page_navigation_is_not_reported_as_a_source():
    """A share page is full of anchors that go nowhere a citation could: the
    provider's own controls, fragment jumps, `javascript:` handlers. Marking
    those would put a link beside text nobody linked."""
    html = (
        '<div data-testid="user-message">go on</div>'
        '<div class="font-claude-response">See <a href="#section-2">section 2</a> '
        'or <a href="javascript:void(0)">expand</a>.</div>'
    )
    turns, _, _ = parse_share_html(html, "claude")

    assert "[link:" not in turns[1].text


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def _transcript(turns: list[tuple[str, str]]) -> Transcript:
    t = Transcript(kind="link", source_id="s", provider="claude")
    t.turns = [
        TranscriptTurn(index=i + 1, role=role, text=text)
        for i, (role, text) in enumerate(turns)
    ]
    t.full_text = " ".join(text for _, text in turns)
    return t


def test_hydration_populates_both_submissions():
    task = make_task()
    task.model_a.final_link = "https://share.gemini.google/aaaa"
    task.model_b.final_link = "https://chatgpt.com/share/bbbb"

    def fetch(url: str) -> Transcript:
        return _transcript([("user", f"ask {url}"), ("assistant", "answer")])

    report = hydrate_conversations([task], fetch=fetch)
    assert report.hydrated == 2
    assert report.turns == 4
    assert len(task.model_a.conversation) == 2
    assert task.model_b.conversation[0].role == "user"


def test_hydration_rejects_a_page_with_no_user_turns():
    """A share link that renders claude.ai's marketing page, seen on 6a7190c542368ece68aa27cb."""
    task = make_task()
    task.model_a.final_link = "https://claude.ai/share/cccc"
    task.model_b.final_link = "https://claude.ai/share/dddd"
    task.model_a.conversation = []
    task.model_b.conversation = []

    def fetch(url: str) -> Transcript:
        if url.endswith("cccc"):
            return _transcript([("assistant", "Your thinking partner"), ("assistant", "Pro")])
        return _transcript([("user", "real question"), ("assistant", "real answer")])

    report = hydrate_conversations([task], fetch=fetch)
    assert report.hydrated == 1
    assert task.model_a.conversation == []
    assert any("no user turns" in f for f in report.failures)


def test_hydration_reports_a_dead_page_separately_from_a_fetch_failure():
    task = make_task()
    task.model_a.final_link = "https://chatgpt.com/share/dead"
    task.model_b.final_link = ""

    def fetch(url: str) -> Transcript:
        t = Transcript(kind="link", source_id="dead", provider="gpt")
        t.dead_page = True
        return t

    report = hydrate_conversations([task], fetch=fetch)
    assert report.hydrated == 0
    assert len(report.dead_pages) == 1
    assert any("no final trajectory link" in f for f in report.failures)
    # The report line is for a human reading stderr; check 90 reads the flag.
    assert task.model_a.share_page_dead is True


def test_a_timed_out_fetch_is_not_stamped_as_a_dead_page():
    """The distinction check 90 rests on: our failure is not their bad link."""
    task = make_task()
    task.model_a.final_link = "https://chatgpt.com/share/slow"
    task.model_b.final_link = ""

    def fetch(url: str) -> Transcript:
        t = Transcript(kind="link", source_id="slow", provider="gpt")
        t.error = "browser produced an empty DOM (timed out)"
        return t

    report = hydrate_conversations([task], fetch=fetch)
    assert report.dead_pages == []
    assert task.model_a.share_page_dead is False


def test_check_90_fails_on_a_dead_page_found_by_hydration():
    """Hydration sees the dead page on every fetching run; provenance is opt-in."""
    from ..cli import run_deterministic

    task = make_task()
    task.model_a.final_link = "https://chatgpt.com/share/dead"
    task.model_a.share_page_dead = True

    verdict = next(v for v in run_deterministic(task) if v.check_id == 90)
    assert verdict.band == "fail"
    assert verdict.measurement.counts["dead_links"] == 1


def test_hydration_fetches_each_distinct_url_once():
    tasks = [make_task(task_id=f"t{i}") for i in range(3)]
    for task in tasks:
        task.model_a.final_link = "https://share.gemini.google/same"
        task.model_b.final_link = "https://share.gemini.google/same"

    calls: list[str] = []

    def fetch(url: str) -> Transcript:
        calls.append(url)
        return _transcript([("user", "q"), ("assistant", "a")])

    report = hydrate_conversations(tasks, fetch=fetch)
    assert calls == ["https://share.gemini.google/same"]
    assert report.hydrated == 6


# ---------------------------------------------------------------------------
# A full disk, which announces itself as a network timeout
# ---------------------------------------------------------------------------
#
# Drawn from a 34-task run that hydrated 5 of 68 submissions and reported
# "browser produced an empty DOM (timed out)" against every other link,
# including two that had rendered fine three hours earlier. Chrome cannot write
# its profile on a full disk, so it exits having produced nothing, which is
# byte-for-byte what a page that never rendered produces. The run cost $133 and
# the verdicts were scored against a judge that had no transcripts.


def test_a_full_disk_is_refused_before_a_browser_is_launched(monkeypatch):
    """The check has to come first, or the diagnosis is lost.

    Once Chrome has been launched and killed there is nothing left to
    distinguish this from the host timing out.
    """
    from .. import sources

    monkeypatch.setattr(sources, "find_chrome", lambda: "/fake/chrome")
    monkeypatch.setattr(sources, "free_disk_bytes", lambda path=None: 500 * 1024**2)

    def explode(*args, **kwargs):
        raise AssertionError("a browser must not be launched with no disk to write to")

    monkeypatch.setattr(sources.subprocess, "Popen", explode)

    result = sources.fetch_rendered_html("https://share.gemini.google/x")
    assert "refusing to launch a browser" in result.error
    assert "0.5 GiB free" in result.error
    assert not result.html


def test_an_empty_dom_on_a_full_disk_blames_the_disk_and_not_the_host(monkeypatch):
    """Triage reads a timeout as the host refusing us and marks the task for a
    refetch. A refetch cannot fix a full disk, so the message has to say which
    of the two it was."""
    from .. import sources

    monkeypatch.setattr(sources, "free_disk_bytes", lambda path=None: 100 * 1024**2)
    note = sources._disk_note()
    assert "0.1 GiB free" in note


def test_an_empty_dom_with_disk_to_spare_still_reads_as_a_timeout(monkeypatch):
    """The original diagnosis has to survive: most empty DOMs really are the page."""
    from .. import sources

    monkeypatch.setattr(sources, "free_disk_bytes", lambda path=None: 500 * 1024**3)
    assert sources.free_disk_bytes() > sources.MIN_FREE_DISK_BYTES


def test_the_sweep_removes_abandoned_profiles_but_spares_a_running_one(tmp_path, monkeypatch):
    """The fetch cleans up in a `finally`, which a SIGKILL skips. One interrupted
    run left 61 profiles behind holding 255 MiB, on a disk with 6.8 GiB free."""
    import time as _time

    from .. import sources

    monkeypatch.setattr(sources.tempfile, "gettempdir", lambda: str(tmp_path))
    stale = tmp_path / "hb_profile_old"
    stale.mkdir()
    (stale / "Default").write_text("cache")
    os.utime(stale, (_time.time() - 7200, _time.time() - 7200))
    live = tmp_path / "hb_profile_new"
    live.mkdir()
    unrelated = tmp_path / "someone_elses_dir"
    unrelated.mkdir()

    removed = sources.sweep_stale_browser_profiles(older_than_s=3600)

    assert removed == 1
    assert not stale.exists()
    assert live.exists(), "a profile a running fetch is using must survive"
    assert unrelated.exists(), "the sweep must only touch its own prefix"


# ---------------------------------------------------------------------------
# Office deliverables -- a .docx is a zip, and the generic GET rejected it
# ---------------------------------------------------------------------------


def _docx_bytes(paragraphs: list[str]) -> bytes:
    """The smallest thing that is genuinely a .docx as far as the reader cares."""
    import io
    import zipfile

    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", f"<w:document><w:body>{body}</w:body></w:document>")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"Content-Type": "application/octet-stream"}

    def read(self, size: int = -1) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_docx_deliverable_is_read_rather_than_dropped(monkeypatch, tmp_path):
    from .. import sources

    payload = _docx_bytes(["Deploy seven ThinkPad E14 units.", "Record every serial number."])
    monkeypatch.setattr(sources, "urlopen", lambda *a, **k: _FakeResponse(payload))
    policy = dataclasses.replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path))

    text = sources.fetch_attachment_text(
        "https://example.com/checklist.docx",
        policy,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert "Deploy seven ThinkPad E14 units." in text
    assert "Record every serial number." in text
    assert "<w:t>" not in text, "XML tags must not reach the auditor as content"


def test_an_unreadable_office_file_returns_empty_rather_than_garbage(monkeypatch, tmp_path):
    """Fail safe: an unreadable deliverable must not look like an empty one."""
    from .. import sources

    monkeypatch.setattr(sources, "urlopen", lambda *a, **k: _FakeResponse(b"not a zip at all"))
    policy = dataclasses.replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path))

    text = sources.fetch_attachment_text(
        "https://example.com/broken.docx", policy, mime_type="application/octet-stream"
    )

    assert text == ""
