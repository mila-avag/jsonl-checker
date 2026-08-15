"""Preflight validation and check 90 link validation."""

from __future__ import annotations

import json

import pytest

from honeybee_qc.config import Policy
from honeybee_qc.links import build_link_report, classify_link, evaluate_check_90
from honeybee_qc.models import DimensionRating, Sxs
from honeybee_qc.preflight import load_tasks, parse_task, validate_batch, validate_task
from honeybee_qc.tests.fixtures import CLAUDE, GEMINI, GPT, make_submission, make_task


# ---------------------------------------------------------------------------
# Link classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,provider",
    [
        ("https://gemini.google.com/share/d64e0a300ec0", "gemini"),
        ("https://gemini.google.com/share/d64e0a300ec0?skid=86dc54c2-ae44", "gemini"),
        ("https://share.gemini.google/fpM8qqHSz7XG", "gemini"),
        ("https://chatgpt.com/share/8f2c1d4e9a0b", "gpt"),
        ("https://chat.openai.com/share/8f2c1d4e9a0b", "gpt"),
        ("https://claude.ai/share/1a2b3c4d5e6f", "claude"),
        ("https://www.claude.ai/share/1a2b3c4d5e6f", "claude"),
    ],
)
def test_valid_share_links_across_all_three_providers(url, provider):
    result = classify_link(url)
    assert result.valid
    assert result.provider == provider
    assert result.share_id


@pytest.mark.parametrize(
    "url,reason_fragment",
    [
        ("", "missing"),
        ("   ", "missing"),
        ("not-a-url", "unsupported scheme"),
        ("ftp://gemini.google.com/share/abc123", "unsupported scheme"),
        ("https://example.com/share/abc123", "unrecognised host"),
        ("https://gemini.google.com/app/d64e0a300ec0", "not a share link"),
        ("https://chatgpt.com/c/d64e0a300ec0", "not a share link"),
        ("https://gemini.google.com/share/", "share path does not match"),
        ("https://gemini.google.com/share/ab", "share path does not match"),
    ],
)
def test_invalid_links_are_rejected_with_a_reason(url, reason_fragment):
    result = classify_link(url)
    assert not result.valid
    assert reason_fragment in result.reason


def test_query_string_does_not_affect_validity():
    bare = classify_link("https://gemini.google.com/share/d64e0a300ec0")
    with_skid = classify_link("https://gemini.google.com/share/d64e0a300ec0?skid=x")
    assert bare.share_id == with_skid.share_id


# ---------------------------------------------------------------------------
# Check 90
# ---------------------------------------------------------------------------


def test_90_passes_when_both_final_links_are_valid():
    v = evaluate_check_90(make_task())
    assert v.band == "clean"
    assert v.error_code is None
    assert v.measurement.counts["final_links_valid"] == 2


def test_90_fails_when_a_final_link_is_missing():
    task = make_task()
    task.model_b.final_link = ""
    v = evaluate_check_90(task)
    assert v.band == "fail"
    assert v.error_code == "[Fail - Missing/Invalid Links]"
    assert any("final_link_B" in i for i in v.contributing_items)


def test_90_fails_when_a_final_link_is_a_private_conversation_url():
    task = make_task()
    task.model_a.final_link = "https://gemini.google.com/app/d64e0a300ec0"
    assert evaluate_check_90(task).band == "fail"


def test_90_fails_when_a_whole_submission_is_absent():
    task = make_task()
    task.model_b = None
    v = evaluate_check_90(task)
    assert v.band == "fail"
    assert "final_link_B:missing" in v.contributing_items


def test_90_is_binary_and_never_emits_a_middle_band():
    for task in (make_task(), make_task()):
        assert evaluate_check_90(task).band in ("fail", "clean")


def test_90_invalid_per_turn_links_warn_but_do_not_fail():
    task = make_task()
    task.model_a.turn_links[0].url = "https://example.com/nope"
    v = evaluate_check_90(task)
    assert v.band == "clean"
    assert v.measurement.counts["per_turn_invalid_a"] == 1
    assert "per-turn links are invalid" in v.measurement.notes


def test_same_conversation_filed_under_both_slots_is_flagged():
    task = make_task()
    task.model_b.final_link = task.model_a.final_link
    report = build_link_report(task)
    assert any("same shared conversation" in w for w in report.warnings)


def test_provider_mismatch_against_the_declared_model_is_flagged():
    task = make_task()
    task.model_a.declared_provider = "claude"
    report = build_link_report(task)
    assert any("declares claude" in w for w in report.warnings)


def test_missing_per_turn_links_warn_only_when_policy_requires_them():
    task = make_task()
    task.model_a = make_submission("A", GEMINI, 20, per_turn_links=False)
    assert not any("no per-turn links" in w for w in build_link_report(task).warnings)
    strict = build_link_report(task, Policy(require_per_turn_links=True))
    assert any("no per-turn links" in w for w in strict.warnings)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_a_well_formed_task_passes_preflight():
    res = validate_task(make_task())
    assert res.ok, res.hard_failures


def test_empty_rubric_hard_fails_rather_than_dividing_by_zero():
    task = make_task()
    task.rubric = []
    task.criterion_ratings = []
    res = validate_task(task)
    assert not res.ok
    assert any("rubric is empty" in f for f in res.hard_failures)


def test_missing_task_id_hard_fails():
    task = make_task()
    task.task_id = ""
    assert not validate_task(task).ok


def test_duplicate_task_ids_in_a_batch_hard_fail():
    results = validate_batch([make_task("dup"), make_task("dup")])
    assert all(not r.ok for r in results)
    assert all(any("duplicated" in f for f in r.hard_failures) for r in results)


def test_empty_prompts_warns_because_the_conversation_is_fetched_separately():
    """A task straight out of Snowflake has no turn text: it lives on the share
    pages. Rejecting it would void the twenty checks that need no conversation."""
    task = make_task()
    task.prompts = []
    res = validate_task(task)
    assert res.ok
    assert any("has not been fetched" in w for w in res.warnings)


def test_non_monotonic_turn_indices_hard_fail():
    task = make_task()
    task.prompts[3].index = 1
    res = validate_task(task)
    assert any("non-monotonic" in f for f in res.hard_failures)


def test_first_turn_must_be_a_user_turn():
    task = make_task()
    task.prompts[0].role = "assistant"
    res = validate_task(task)
    assert any("first turn must be a user turn" in f for f in res.hard_failures)


@pytest.mark.parametrize("weight", [0, -1, -5, 6, 99])
def test_out_of_range_weights_warn_and_leave_the_task_auditable(weight):
    """Live rubrics really do carry -3 to -5 against a spec that forbids negatives.

    Voiding the task would discard the twenty other checks over one weight, so the
    defect is reported and check 260 counts it as maximally wrong instead.
    """
    task = make_task()
    task.rubric[0].weight = weight
    res = validate_task(task)
    assert res.ok
    assert any("outside 1-5" in w for w in res.warnings)


def test_a_non_integer_weight_is_still_malformed_input():
    task = make_task()
    task.rubric[0].weight = "heavy"  # type: ignore[assignment]
    res = validate_task(task)
    assert any("not an integer" in f for f in res.hard_failures)


def test_an_out_of_scale_weight_counts_as_a_two_level_disagreement():
    from honeybee_qc.config import WeightBucketMap

    buckets = WeightBucketMap()
    assert buckets.level_distance(-5, 3) == 2
    assert buckets.level_distance(1, 2) == 0
    assert not buckets.in_scale(-5)


def test_unknown_l1_label_hard_fails():
    task = make_task()
    task.rubric[0].l1_label = "Vibes"
    assert not validate_task(task).ok


def test_criterion_rating_for_unknown_criterion_hard_fails():
    task = make_task()
    task.criterion_ratings[0].criterion_id = "C999"
    res = validate_task(task)
    assert any("unknown criterion_id" in f for f in res.hard_failures)


def test_duplicate_criterion_ids_hard_fail():
    task = make_task()
    task.rubric[1].criterion_id = task.rubric[0].criterion_id
    res = validate_task(task)
    assert any("duplicate criterion_id" in f for f in res.hard_failures)


def test_unknown_rating_dimension_hard_fails():
    task = make_task()
    task.dimension_ratings.append(
        DimensionRating(model="A", dimension="Vibe quality", rating=5)
    )
    assert not validate_task(task).ok


@pytest.mark.parametrize("likert", [0, 8, 99, -1])
def test_likert_outside_one_to_seven_hard_fails(likert):
    task = make_task()
    task.sxs = Sxs(likert=likert, justification="x")
    res = validate_task(task)
    assert any("outside the 1-7 scale" in f for f in res.hard_failures)


def test_dimension_rating_outside_the_configured_scale_hard_fails():
    task = make_task()
    task.dimension_ratings[0].rating = 11
    res = validate_task(task)
    assert any("outside the 1-5 scale" in f for f in res.hard_failures)


def test_key_turn_that_does_not_resolve_to_an_assistant_turn_hard_fails():
    task = make_task()
    task.model_a.conversation = [
        t for t in task.model_a.conversation if not (t.index == 5 and t.role == "assistant")
    ]
    res = validate_task(task)
    assert any("does not resolve to an assistant turn" in f for f in res.hard_failures)


def test_key_turn_beyond_the_fetched_conversation_warns_instead_of_failing():
    """Share pages routinely render only part of a long conversation. Failing the
    task hard on that discards good work over our own fetch shortfall, and it is
    what dropped 7 of 17 tasks in the first Ultra Evals batch."""
    task = make_task(key_turn=99)
    res = validate_task(task)
    assert res.ok
    assert not res.hard_failures
    assert any("is beyond the" in w and "incomplete" in w for w in res.warnings)


def test_a_gap_in_the_middle_of_the_conversation_still_hard_fails():
    """Truncation only ever removes a suffix. A missing turn with turns on both
    sides of it is real corruption, and nothing legitimate produces one."""
    task = make_task(key_turn=5)
    task.model_a.conversation = [
        t for t in task.model_a.conversation if t.index != 5
    ]
    res = validate_task(task)
    assert any("does not exist" in f for f in res.hard_failures)


def test_a_conversation_cut_off_before_the_reply_warns_instead_of_failing():
    task = make_task(key_turn=20, exchanges=20)
    task.model_a.conversation = [
        t for t in task.model_a.conversation if not (t.index == 20 and t.role == "assistant")
    ]
    res = validate_task(task)
    assert res.ok
    assert any("cut off mid-exchange" in w for w in res.warnings)


def test_l2_label_without_a_configured_enum_warns_only():
    task = make_task()
    task.rubric[0].l2_label = "Some L2"
    res = validate_task(task, Policy(l2_labels_configured=False))
    assert res.ok
    assert any("check 210 will report not_evaluated" in w for w in res.warnings)


def test_unknown_l2_label_warns_because_the_taxonomy_has_been_revised():
    """32 live criteria carry leaves no revision we hold defines. That is a leaf
    we cannot place, not a task we cannot audit."""
    task = make_task()
    task.rubric[0].l2_label = "Some L2"
    res = validate_task(task)
    assert res.ok
    assert any("check 210 will skip it" in w for w in res.warnings)


def test_l2_parented_under_the_wrong_l1_hard_fails():
    task = make_task()
    task.rubric[0].l1_label = "Safety"
    task.rubric[0].l2_label = "Goal Elicitation"
    res = validate_task(task)
    assert any("belongs under 'Collaboration Quality'" in f for f in res.hard_failures)


def test_score_zero_without_relevant_turns_warns_and_feeds_missing_turn():
    task = make_task()
    task.criterion_ratings[0].score = 0
    task.criterion_ratings[0].relevant_turns = []
    res = validate_task(task)
    assert res.ok
    assert any("missing_turn" in w for w in res.warnings)


def test_not_applicable_on_a_non_conditional_dimension_warns():
    task = make_task()
    task.dimension_ratings[0].not_applicable = True
    task.dimension_ratings[0].rating = None
    res = validate_task(task)
    assert any("is conditional" in w for w in res.warnings)


def test_conversation_below_seven_turns_warns():
    """Audit workflow step 3: both models require at least 7 turns."""
    task = make_task(exchanges=5, key_turn=3)
    res = validate_task(task)
    assert any("fewer than the 7 turns" in w for w in res.warnings)


def test_exactly_seven_turns_does_not_warn():
    task = make_task(exchanges=7, key_turn=3)
    res = validate_task(task)
    assert not any("turns both models are required" in w for w in res.warnings)


def test_missing_target_deliverables_warns_that_coverage_has_no_checklist():
    task = make_task()
    task.target_deliverables = []
    res = validate_task(task)
    assert any("no customer checklist" in w for w in res.warnings)


def test_missing_transcript_pdf_warns():
    task = make_task()
    task.model_a.transcript_pdf = ""
    assert any("no transcript PDF" in w for w in validate_task(task).warnings)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_task_round_trips_the_documented_contract():
    raw = {
        "task_id": "t1",
        "prompts": [{"turn": 1, "role": "user", "text": "hello"}],
        "target_deliverables": ["a chart"],
        "model_a": {
            "final_link": GEMINI,
            "turn_links": [{"turn": 1, "url": GEMINI}],
            "transcript_pdf": "a.pdf",
            "attachments": ["chart.png"],
            "conversation": [
                {"turn": 1, "role": "user", "text": "hi"},
                {"turn": 1, "role": "assistant", "text": "here is chart.png"},
            ],
        },
        "model_b": {"final_link": CLAUDE},
        "key_turn": {"turn_index": 1, "justification": "j"},
        "rubric": [
            {
                "criterion_id": "C1",
                "text": "produces the chart",
                "l1_label": "Outcome Quality",
                "weight": 4,
            }
        ],
        "criterion_ratings": [
            {"criterion_id": "C1", "model": "A", "score": 0, "relevant_turns": [1]}
        ],
        "dimension_ratings": [
            {
                "model": "A",
                "dimension": "Outcome quality",
                "rating": 7,
                "justification": "j",
                "relevant_turns": [1],
            }
        ],
        "sxs": {"likert": 6, "justification": "prefer B"},
    }
    task = parse_task(raw)
    assert task.task_id == "t1"
    assert task.model_a.attachments == ["chart.png"]
    assert task.model_a.exchange_count() == 1
    assert task.model_a.has_assistant_at(1)
    assert task.rubric[0].weight == 4
    assert task.sxs.likert == 6
    assert task.target_deliverables == ["a chart"]


def test_load_tasks_reports_the_line_of_a_malformed_record(tmp_path):
    p = tmp_path / "batch.jsonl"
    p.write_text('{"task_id": "ok"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tasks(p)
    assert ":2:" in str(exc.value)


def test_load_tasks_skips_blank_lines(tmp_path):
    p = tmp_path / "batch.jsonl"
    p.write_text(json.dumps({"task_id": "a"}) + "\n\n" + json.dumps({"task_id": "b"}) + "\n")
    assert [t.task_id for t in load_tasks(p)] == ["a", "b"]
