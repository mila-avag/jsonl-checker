"""Link-vs-PDF provenance: scoring, verdicts, reuse detection, and check mapping.

The false-positive cases matter more than the true positives here. Wrongly
calling an honest contributor a spammer is the expensive error, so most of these
tests pin down what must *not* convict.
"""

from __future__ import annotations

import pytest

from honeybee_qc.config import Policy
from honeybee_qc.gates import evaluate_check_100
from honeybee_qc.links import evaluate_check_90
from honeybee_qc.provenance import (
    compare_transcripts,
    duplicates_convict,
    find_duplicates,
)
from honeybee_qc.sources import parse_share_html, read_pdf_transcript
from honeybee_qc.transcripts import (
    Transcript,
    TranscriptTurn,
    containment,
    informative_turns,
    normalize,
    shingles,
    tokens,
)
from honeybee_qc.tests.fixtures import make_task
from honeybee_qc.verify import (
    collect_duplicates,
    dead_link_reasons,
    unauditable_reasons,
    verify_task,
)

VERIFY = Policy(verify_provenance=True)

REAL_TURNS = [
    "I need an eleven second animation explaining how a gyroscope resists changes "
    "in orientation, aimed at high school physics students.",
    "Here is an eleven second animation showing a spinning gyroscope with angular "
    "momentum vectors drawn along the spin axis in blue.",
    "Can you make the precession arrow red instead of blue so it contrasts with "
    "the angular momentum vector on the spin axis.",
    "I have recoloured the precession arrow to red and kept the angular momentum "
    "vector in blue for contrast throughout the eleven second clip.",
    "Please export the final clip at four K resolution and make sure the loop is "
    "seamless when it repeats.",
    "The clip is exported at four K resolution and the final frame matches the "
    "first so the loop repeats seamlessly without a visible jump.",
]

UNRELATED_TURNS = [
    "Write me a spreadsheet formula that computes compound interest across a "
    "variable number of monthly deposit periods.",
    "You can use the future value function with a monthly rate and the number of "
    "deposit periods as arguments to compute that total.",
    "Now add a column that shows the running balance after each monthly deposit "
    "so I can chart the growth curve over time.",
    "I have added a running balance column that accumulates each monthly deposit "
    "plus interest so the growth curve can be charted directly.",
    "Format the currency column with two decimal places and a thousands separator "
    "for readability in the final report.",
    "The currency column is now formatted with two decimals and a thousands "
    "separator so the figures read cleanly in the report.",
]


def link_transcript(turn_texts=None, source_id="abc123", **kwargs) -> Transcript:
    turn_texts = REAL_TURNS if turn_texts is None else turn_texts
    turns = []
    for i, text in enumerate(turn_texts):
        turns.append(
            TranscriptTurn(
                index=i // 2 + 1, role="user" if i % 2 == 0 else "assistant", text=text
            )
        )
    return Transcript(
        kind="link",
        source_id=source_id,
        provider="gemini",
        turns=turns,
        full_text="\n".join(turn_texts),
        **kwargs,
    )


def pdf_transcript(turn_texts=None, extra="", **kwargs) -> Transcript:
    turn_texts = REAL_TURNS if turn_texts is None else turn_texts
    return Transcript(
        kind="pdf", source_id="t.pdf", full_text="\n".join(turn_texts) + "\n" + extra, **kwargs
    )


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def test_normalization_folds_pdf_print_artifacts():
    assert tokens("Hello\u00a0World") == ["hello", "world"]
    assert tokens("\u201cquoted\u201d") == ["quoted"]
    assert "page" not in tokens("real content\nPage 3 of 12\nmore content")


def test_containment_is_one_when_the_turn_is_inside_the_document():
    doc = shingles(tokens(" ".join(REAL_TURNS)))
    assert containment(REAL_TURNS[0], doc) == 1.0


def test_containment_is_zero_for_unrelated_text():
    doc = shingles(tokens(" ".join(REAL_TURNS)))
    assert containment(UNRELATED_TURNS[0], doc) == 0.0


def test_shingles_resist_coincidental_vocabulary_overlap():
    # Same words, different order: a bag-of-words measure would score this high.
    a = "the gyroscope resists changes in orientation because of angular momentum"
    b = "angular momentum orientation of because changes resists the in gyroscope"
    assert containment(a, shingles(tokens(b))) == 0.0


def test_short_filler_turns_are_excluded_from_scoring():
    turns = [
        TranscriptTurn(1, "user", "ok"),
        TranscriptTurn(1, "assistant", "thanks, continuing"),
        TranscriptTurn(2, "user", REAL_TURNS[0]),
    ]
    kept = informative_turns(turns, min_tokens=8)
    assert len(kept) == 1


def test_normalize_is_case_and_whitespace_insensitive():
    assert normalize("  HELLO   World  ").split() == ["hello", "world"]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_matching_pdf_is_verified():
    r = compare_transcripts("t", "A", link_transcript(), pdf_transcript(), VERIFY)
    assert r.verdict == "verified"
    assert r.forward_ratio == 1.0
    assert not r.convicts


def test_a_pdf_from_a_different_conversation_is_contradicted():
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(UNRELATED_TURNS), VERIFY
    )
    assert r.verdict == "contradicted"
    assert r.forward_ratio == 0.0
    assert r.convicts


def test_a_pdf_covering_only_the_opening_turns_is_contradicted():
    # The classic spam shape: a real link, a PDF padded out from the first exchange.
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(REAL_TURNS[:2]), VERIFY
    )
    assert r.forward_ratio == pytest.approx(2 / 6)
    assert r.verdict == "contradicted"


def test_partial_overlap_abstains_rather_than_convicting():
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(REAL_TURNS[:4]), VERIFY
    )
    assert r.forward_ratio == pytest.approx(4 / 6)
    assert r.verdict == "inconclusive"
    assert r.needs_review
    assert not r.convicts


def test_pdf_with_extra_appendix_material_is_not_called_fraud():
    # Distinct padding, because shingles are a set and repeated text collapses.
    padding = " ".join(
        f"appendix note {i} covering unrelated supplementary detail number {i}"
        for i in range(400)
    )
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(extra=padding), VERIFY
    )
    assert r.forward_ratio == 1.0
    assert r.reverse_ratio < VERIFY.provenance_reverse_min
    assert r.verdict == "inconclusive"
    assert not r.convicts


def _as_pdf_print(turns: list[str]) -> list[str]:
    """Simulate what a browser print actually does to text."""
    out = []
    for n, text in enumerate(turns, start=1):
        words = text.split()
        # Wrap at 9 words and hyphenate the break, as a justified column does.
        lines = []
        for i in range(0, len(words), 9):
            chunk = words[i : i + 9]
            if i + 9 < len(words) and len(chunk[-1]) > 4:
                chunk[-1] = chunk[-1][:3] + "-\n" + chunk[-1][3:]
            lines.append(" ".join(chunk))
        out.append(f"Page {n} of {len(turns)}\n" + "\n".join(lines).upper())
    return out


def test_realistic_pdf_print_artifacts_still_verify():
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(_as_pdf_print(REAL_TURNS)), VERIFY
    )
    assert r.verdict == "verified"
    assert not r.convicts


def test_line_break_hyphenation_does_not_destroy_a_match():
    assert tokens("orienta-\ntion") == ["orientation"]
    assert tokens("momen\u2011\n  tum") == ["momentum"]


# ---------------------------------------------------------------------------
# Failing soft
# ---------------------------------------------------------------------------


def test_a_fetch_timeout_never_convicts():
    dead = Transcript(kind="link", error="browser produced an empty DOM (timed out)")
    r = compare_transcripts("t", "A", dead, pdf_transcript(), VERIFY)
    assert r.verdict == "unverifiable"
    assert not r.convicts
    assert r.needs_review


def test_a_missing_browser_never_convicts():
    nodriver = Transcript(kind="link", error="no Chrome or Chromium binary found")
    r = compare_transcripts("t", "A", nodriver, pdf_transcript(), VERIFY)
    assert r.verdict == "unverifiable"
    assert not r.convicts


def test_a_missing_pdf_never_convicts():
    r = compare_transcripts(
        "t", "A", link_transcript(), Transcript(kind="pdf", error="no PDF supplied"), VERIFY
    )
    assert r.verdict == "unverifiable"
    assert not r.convicts


def test_a_dead_share_page_is_reported_separately_from_a_timeout():
    dead = Transcript(
        kind="link", error="share page reports the conversation as unavailable", dead_page=True
    )
    r = compare_transcripts("t", "A", dead, pdf_transcript(), VERIFY)
    assert r.verdict == "link_dead"
    assert r.convicts


def test_too_few_recovered_turns_abstains():
    r = compare_transcripts(
        "t", "A", link_transcript(REAL_TURNS[:2]), pdf_transcript(UNRELATED_TURNS), VERIFY
    )
    assert r.verdict == "unverifiable"
    assert "needed to judge" in " ".join(r.reasons)


def test_thresholds_are_policy_controlled():
    lenient = Policy(verify_provenance=True, provenance_contradicted_ratio=0.0)
    r = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(UNRELATED_TURNS), lenient
    )
    assert r.verdict == "contradicted"  # 0.0 <= 0.0 still convicts at the boundary

    r2 = compare_transcripts(
        "t", "A", link_transcript(), pdf_transcript(REAL_TURNS[:2]), lenient
    )
    assert r2.verdict == "inconclusive"


# ---------------------------------------------------------------------------
# Reuse across the batch
# ---------------------------------------------------------------------------


def test_the_same_share_link_under_two_tasks_is_flagged_and_convicts():
    rows = [
        ("t1", "A", "share123", "pdfA"),
        ("t2", "A", "share123", "pdfB"),
    ]
    findings = find_duplicates(rows)
    assert findings[0].kind == "link_across_tasks"
    assert findings[0].task_ids == ["t1", "t2"]
    assert duplicates_convict(findings)


def test_the_same_link_across_both_slots_of_one_task_is_milder():
    rows = [("t1", "A", "share123", "pdfA"), ("t1", "B", "share123", "pdfB")]
    findings = find_duplicates(rows)
    assert findings[0].kind == "link_within_task"
    assert not duplicates_convict(findings)


def test_identical_pdfs_across_tasks_convict():
    rows = [("t1", "A", "s1", "samehash"), ("t2", "A", "s2", "samehash")]
    findings = find_duplicates(rows)
    assert any(f.kind == "pdf_across_tasks" for f in findings)
    assert duplicates_convict(findings)


def test_distinct_submissions_produce_no_duplicate_findings():
    rows = [("t1", "A", "s1", "p1"), ("t1", "B", "s2", "p2"), ("t2", "A", "s3", "p3")]
    assert find_duplicates(rows) == []


def test_duplicate_conviction_can_be_disabled_by_policy():
    rows = [("t1", "A", "s1", "p1"), ("t2", "A", "s1", "p2")]
    findings = find_duplicates(rows)
    assert not duplicates_convict(findings, Policy(duplicate_across_tasks_is_unauditable=False))


# ---------------------------------------------------------------------------
# Wiring into checks
# ---------------------------------------------------------------------------


def fake_fetch(mapping):
    def _fetch(url, policy=VERIFY):
        return mapping.get(url, Transcript(kind="link", error="not stubbed"))

    return _fetch


def fake_pdf(mapping):
    def _read(path):
        return mapping.get(path, Transcript(kind="pdf", error="no PDF supplied"))

    return _read


def test_verify_task_compares_both_model_slots():
    task = make_task()
    tp = verify_task(
        task,
        VERIFY,
        fetch=fake_fetch(
            {
                task.model_a.final_link: link_transcript(source_id="sa"),
                task.model_b.final_link: link_transcript(source_id="sb"),
            }
        ),
        read_pdf=fake_pdf(
            {
                task.model_a.transcript_pdf: pdf_transcript(),
                task.model_b.transcript_pdf: pdf_transcript(),
            }
        ),
    )
    assert [r.verdict for r in tp.reports] == ["verified", "verified"]
    assert tp.by_model("B").verdict == "verified"


def test_a_contradicted_submission_makes_the_task_unauditable():
    task = make_task()
    tp = verify_task(
        task,
        VERIFY,
        fetch=fake_fetch(
            {
                task.model_a.final_link: link_transcript(source_id="sa"),
                task.model_b.final_link: link_transcript(source_id="sb"),
            }
        ),
        read_pdf=fake_pdf(
            {
                task.model_a.transcript_pdf: pdf_transcript(),
                task.model_b.transcript_pdf: pdf_transcript(UNRELATED_TURNS),
            }
        ),
    )
    reasons = unauditable_reasons(task.task_id, tp, [], VERIFY)
    assert reasons
    assert "does not match the live conversation" in reasons[0]


def test_a_dead_link_fails_check_90():
    task = make_task()
    dead = Transcript(kind="link", error="unavailable", dead_page=True)
    tp = verify_task(
        task,
        VERIFY,
        fetch=fake_fetch({task.model_a.final_link: dead, task.model_b.final_link: dead}),
        read_pdf=fake_pdf({}),
    )
    reasons = dead_link_reasons(tp)
    assert len(reasons) == 2
    v = evaluate_check_90(task, VERIFY, dead_links=reasons)
    assert v.band == "fail"
    assert v.error_code == "[Fail - Missing/Invalid Links]"
    assert v.measurement.counts["dead_links"] == 2


def test_an_unverifiable_fetch_leaves_check_90_clean():
    task = make_task()
    timed_out = Transcript(kind="link", error="timed out")
    tp = verify_task(
        task,
        VERIFY,
        fetch=fake_fetch(
            {task.model_a.final_link: timed_out, task.model_b.final_link: timed_out}
        ),
        read_pdf=fake_pdf({}),
    )
    assert dead_link_reasons(tp) == []
    assert evaluate_check_90(task, VERIFY, dead_links=[]).band == "clean"
    assert len(tp.needs_review) == 2


def test_collect_duplicates_falls_back_to_the_declared_link_when_fetching_failed():
    t1, t2 = make_task("t1"), make_task("t2")
    t2.model_a.final_link = t1.model_a.final_link
    dupes = collect_duplicates([t1, t2], [])
    assert any(d.kind == "link_across_tasks" for d in dupes)


def test_provenance_verdicts_do_not_disturb_the_key_turn_check():
    task = make_task(key_turn=1)
    assert evaluate_check_100(task).band == "fail"


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


GEMINI_HTML = """
<html><body>
<script>var x = "ignore me entirely";</script>
<user-query><div class="q">How does a gyroscope work?</div></user-query>
<response-container><div>It resists changes in orientation.</div></response-container>
<user-query><div>Make the arrow red.</div></user-query>
<response-container><div>The arrow is now red.</div></response-container>
</body></html>
"""

GPT_HTML = """
<html><body>
<div data-message-author-role="user">How does a gyroscope work?</div>
<div data-message-author-role="assistant">It resists changes in orientation.</div>
</body></html>
"""

CLAUDE_HTML = """
<html><body>
<div data-testid="user-message">How does a gyroscope work?</div>
<div class="font-claude-message">It resists changes in orientation.</div>
</body></html>
"""


@pytest.mark.parametrize(
    "html,provider", [(GEMINI_HTML, "gemini"), (GPT_HTML, "gpt"), (CLAUDE_HTML, "claude")]
)
def test_share_html_parses_into_alternating_roles(html, provider):
    turns, full_text, dead = parse_share_html(html, provider)
    assert [t.role for t in turns][:2] == ["user", "assistant"]
    assert "gyroscope" in full_text.lower()
    assert not dead


def test_gemini_html_pairs_turns_into_exchanges():
    turns, _, _ = parse_share_html(GEMINI_HTML, "gemini")
    assert [t.index for t in turns] == [1, 1, 2, 2]


def test_script_contents_are_not_treated_as_conversation():
    turns, full_text, _ = parse_share_html(GEMINI_HTML, "gemini")
    assert all("ignore me entirely" not in t.text for t in turns)
    assert "ignore me entirely" not in full_text


def test_void_tags_do_not_make_a_block_swallow_the_whole_page():
    # Serialized DOM has unclosed <br>, <img>, <input>. Tracking them as open
    # leaves the first block never closing, so one turn absorbs the transcript.
    html = """
    <html><body>
    <user-query-content>First question<br>with a line break<img src="x.png"></user-query-content>
    <message-content>First answer<hr></message-content>
    <user-query-content>Second question<input type="text"></user-query-content>
    <message-content>Second answer</message-content>
    </body></html>
    """
    turns, _, _ = parse_share_html(html, "gemini")
    assert len(turns) == 4
    assert [t.index for t in turns] == [1, 1, 2, 2]
    assert "Second" not in turns[0].text


def test_a_realistic_sixteen_exchange_page_parses_into_sixteen_exchanges():
    body = "".join(
        f"<user-query-content><p>Question number {i} about the animation.</p>"
        f"<br></user-query-content>"
        f"<message-content><div>Answer number {i}.</div><img src='a.png'></message-content>"
        for i in range(1, 17)
    )
    turns, _, _ = parse_share_html(f"<html><body>{body}</body></html>", "gemini")
    assert len({t.index for t in turns}) == 16
    assert len(turns) == 32
    assert [t.role for t in turns[:4]] == ["user", "assistant", "user", "assistant"]


def test_inner_content_markers_are_preferred_over_the_outer_container():
    html = """
    <html><body>
    <user-query><button>edit</button><user-query-content>Real question here</user-query-content>
    </user-query>
    <response-container><span>more_vert</span><message-content>Real answer here</message-content>
    </response-container>
    </html></body>
    """
    turns, _, _ = parse_share_html(html, "gemini")
    assert [t.text for t in turns] == ["Real question here", "Real answer here"]
    assert all("edit" not in t.text and "more_vert" not in t.text for t in turns)


def test_outer_container_is_used_when_the_inner_markers_are_gone():
    html = """
    <html><body>
    <user-query>Question survives a redesign</user-query>
    <response-container>Answer survives a redesign</response-container>
    </body></html>
    """
    turns, _, _ = parse_share_html(html, "gemini")
    assert len(turns) == 2
    assert turns[0].role == "user"


def test_consecutive_user_messages_do_not_collapse_into_one_exchange():
    html = """
    <html><body>
    <user-query-content>First question</user-query-content>
    <message-content>First answer</message-content>
    <user-query-content>Second question</user-query-content>
    <user-query-content>Third question</user-query-content>
    <message-content>Combined answer</message-content>
    </body></html>
    """
    turns, _, _ = parse_share_html(html, "gemini")
    assert [t.index for t in turns] == [1, 1, 2, 2, 2]


def test_an_image_only_reply_is_kept_as_a_turn():
    """Gemini answers an image request with <generated-image><img> and no prose.

    Dropping it for having no text loses the turn, and every later user message
    then looks consecutive, folding the rest of the conversation onto one index.
    On the Ultra Evals batch this cut a 20-turn conversation to 4 and made
    preflight declare the task unauditable.
    """
    html = """
    <html><body>
    <user-query-content>Draw me a diagram</user-query-content>
    <message-content><div class="attachment-container generated-images">
      <generated-image><single-image><img alt="a hypergraph"></single-image></generated-image>
    </div></message-content>
    <user-query-content>Now add labels</user-query-content>
    <message-content><generated-image><single-image><img></single-image></generated-image></message-content>
    <user-query-content>Perfect</user-query-content>
    <message-content>Glad it works.</message-content>
    </body></html>
    """
    turns, _, _ = parse_share_html(html, "gemini")
    assert [t.index for t in turns] == [1, 1, 2, 2, 3, 3]
    assert [t.role for t in turns] == [
        "user", "assistant", "user", "assistant", "user", "assistant"
    ]
    # The alt text is carried through where the page supplies one.
    assert "image" in turns[1].text
    assert "a hypergraph" in turns[1].text


def test_an_image_only_reply_is_treated_as_unauditable_evidence_not_as_prose():
    """The turn has to exist for numbering, but it carries no argument to judge,
    so the rating stage must abstain rather than score an empty reply."""
    from honeybee_qc.context import is_placeholder_turn
    from honeybee_qc.models import Turn

    assert is_placeholder_turn(Turn(index=2, role="assistant", text="[image]"))


def test_a_not_found_page_is_detected_as_dead():
    html = "<html><body><h1>Conversation not found</h1></body></html>"
    turns, _, dead = parse_share_html(html, "gemini")
    assert dead
    assert turns == []


def test_a_page_with_content_is_never_dead_even_if_it_offers_sign_in():
    html = GEMINI_HTML.replace("</body>", "<footer>Sign in to continue</footer></body>")
    _, _, dead = parse_share_html(html, "gemini")
    assert not dead


def test_unknown_provider_still_yields_full_text_for_comparison():
    turns, full_text, _ = parse_share_html(GEMINI_HTML, "unknown")
    assert turns == []
    assert "gyroscope" in full_text.lower()


def test_missing_pdf_path_reports_an_error_rather_than_raising():
    assert read_pdf_transcript(None).error == "no PDF supplied"
    assert "not found" in read_pdf_transcript("/nonexistent/x.pdf").error


def test_pdf_extraction_round_trips_real_text(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "t.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), REAL_TURNS[0][:80])
    doc.save(str(path))
    doc.close()

    t = read_pdf_transcript(str(path))
    assert t.ok
    assert "gyroscope" in t.text().lower()
