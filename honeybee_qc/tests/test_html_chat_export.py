"""Model B HTML / MHTML chat exports hydrate without a share URL."""

from __future__ import annotations

from pathlib import Path

from honeybee_qc.hydrate import hydrate_conversations
from honeybee_qc.sources import (
    extract_html_from_mhtml,
    infer_html_provider,
    looks_like_html_chat_export,
    parse_share_html,
    read_html_transcript,
)
from honeybee_qc.tests.fixtures import make_task

GEMINI_DUMP = """<!DOCTYPE html>
<html><body>
<user-query-content>Plan the Antarctic itinerary for the household.</user-query-content>
<message-content>Here is the PolarLuxe suite comparison and the four open items.</message-content>
<user-query-content>Now inspect the signed contract.</user-query-content>
<message-content>The contract lists two Explorer suites and one Owner suite.</message-content>
</body></html>
"""

MHTML = (
    "From: <Saved by Blink>\r\n"
    "MIME-Version: 1.0\r\n"
    "Content-Type: multipart/related; type=\"text/html\"; "
    "boundary=\"----Boundary\"\r\n"
    "\r\n"
    "------Boundary\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "\r\n"
    f"{GEMINI_DUMP}\r\n"
    "------Boundary--\r\n"
)


def test_parse_share_html_reads_gemini_dump_dom():
    turns, _, dead = parse_share_html(GEMINI_DUMP, "gemini")
    assert not dead
    assert len(turns) == 4
    assert turns[0].role == "user"
    assert "Antarctic" in turns[0].text
    assert turns[1].role == "assistant"


def test_looks_like_html_chat_export():
    assert looks_like_html_chat_export("https://s3.example.com/Gemini Spark.mhtml?x=1")
    assert looks_like_html_chat_export("/tmp/dump.html")
    assert not looks_like_html_chat_export("model_B_transcript.pdf")


def test_mhtml_extracts_html_part():
    html = extract_html_from_mhtml(MHTML.encode("utf-8"))
    assert "PolarLuxe" in html
    assert infer_html_provider(html) == "gemini"


def test_read_html_transcript_from_local_file(tmp_path: Path):
    path = tmp_path / "spark.html"
    path.write_text(GEMINI_DUMP, encoding="utf-8")
    from dataclasses import replace

    from honeybee_qc.config import DEFAULT_POLICY

    policy = replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path / "snap"))
    transcript = read_html_transcript(str(path), policy)
    assert not transcript.error
    assert len(transcript.turns) == 4
    assert any(t.role == "user" for t in transcript.turns)


def test_hydrate_uses_html_export_when_b_has_no_share_link(tmp_path: Path):
    path = tmp_path / "Gemini Spark.html"
    path.write_text(GEMINI_DUMP, encoding="utf-8")
    task = make_task()
    task.model_a.final_link = "https://gemini.google.com/share/aaaa"
    task.model_b.final_link = ""
    task.model_a.conversation = []
    task.model_b.conversation = []
    task.model_b.transcript_pdf = str(path)

    from dataclasses import replace
    from honeybee_qc.config import DEFAULT_POLICY

    policy = replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path / "snap"))

    def fetch(url: str):
        from honeybee_qc.transcripts import Transcript, TranscriptTurn

        t = Transcript(kind="link", source_id="aaaa", provider="gemini")
        t.turns = [
            TranscriptTurn(1, "user", "hello"),
            TranscriptTurn(1, "assistant", "hi"),
        ]
        return t

    report = hydrate_conversations([task], policy=policy, fetch=fetch)
    assert report.hydrated == 2
    assert len(task.model_b.conversation) == 4
    assert task.model_b.conversation[0].role == "user"
    assert not any("no final trajectory link" in f and "/B" in f for f in report.failures)
