"""Regressions for the evidence-gathering path, all drawn from live data.

Every case here is something that silently produced an empty or truncated
conversation against real tasks, which is the worst failure mode this pipeline
has: the audit still returns a verdict, but it reached it without the evidence.
"""

from __future__ import annotations

import dataclasses
import json

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


def test_claude_current_assistant_markup_parses():
    html = (
        '<div data-testid="user-message">what is the plan</div>'
        '<div class="font-claude-response relative leading-[1.65rem]">here is the plan</div>'
    )
    turns, _, _ = parse_share_html(html, "claude")
    assert [t.role for t in turns] == ["user", "assistant"]


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
