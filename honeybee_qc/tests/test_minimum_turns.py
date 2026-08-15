"""Check 96: both models require at least 7 turns.

The requirement is stated once, in the audit workflow tab's step 3, and tab 2
gives it no row. It is the first check in this build that was failing in the
direction of missed detections rather than false positives: `preflight` counted
the exchanges and warned, but a warning never reaches a verdict, so a
two-turn submission was scored clean on every dimension that mattered.

What most of this file tests is the abstention. The check is shape A, so a fail
is terminal, and the counts it gates on come out of a browser fetch that fails
for reasons that have nothing to do with the contributor: a share page behind a
login wall parses as one exchange whatever sits behind it. Failing that is our
bug, not a detection.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.cli import DETERMINISTIC_NOW, run_deterministic
from honeybee_qc.config import DEFAULT_POLICY, Policy
from honeybee_qc.errors import ERROR_CODES
from honeybee_qc.gates import cited_turns, evaluate_check_96
from honeybee_qc.models import CriterionRating, DimensionRating, KeyTurn, Turn
from honeybee_qc.preflight import MIN_EXPECTED_EXCHANGES, validate_task
from honeybee_qc.registry import BLOCKS, REGISTRY
from honeybee_qc.tests.fixtures import make_conversation, make_task


def _cite(task, model: str, turns: list[int]) -> None:
    """Attach the contributor's turn citations for one model.

    Both citation surfaces are used, because both are read: the per-criterion
    turn attribution and the per-dimension relevant turns.
    """
    task.criterion_ratings.append(
        CriterionRating(
            criterion_id=task.rubric[0].criterion_id,
            model=model,  # type: ignore[arg-type]
            score=0,
            relevant_turns=list(turns),
        )
    )
    task.dimension_ratings.append(
        DimensionRating(
            model=model,  # type: ignore[arg-type]
            dimension="Outcome quality",
            rating=3,
            justification="cited above",
            relevant_turns=list(turns),
        )
    )


# ---------------------------------------------------------------------------
# The requirement itself
# ---------------------------------------------------------------------------


def test_both_models_at_the_minimum_is_clean():
    task = make_task(exchanges=7, key_turn=7)
    verdict = evaluate_check_96(task)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["turns_by_model"] == {"A": 7, "B": 7}
    assert verdict.measurement.counts["unverifiable_by_model"] == {}


def test_comfortably_long_conversations_are_clean():
    verdict = evaluate_check_96(make_task(exchanges=20))
    assert verdict.band == "clean"
    assert verdict.error_code is None


def test_one_model_below_the_minimum_fails_the_task():
    """The spec requires *both* models to reach 7, so one short submission
    settles it however long the other one ran."""
    task = make_task(exchanges=20)
    task.model_a.conversation = make_conversation(6)
    _cite(task, "A", [1, 2, 6])
    task.key_turn = KeyTurn(turn_index=6)

    verdict = evaluate_check_96(task)
    assert verdict.band == "fail"
    assert verdict.error_code == "[Fail - Insufficient Turns]"
    assert verdict.contributing_items == ["model A: 6 turns"]
    assert verdict.measurement.counts["turns_by_model"] == {"A": 6, "B": 20}


def test_six_turns_fails_and_seven_does_not():
    """The threshold is inclusive on the passing side: "at least 7 turns"."""
    for exchanges, expected in ((6, "fail"), (7, "clean")):
        task = make_task(exchanges=exchanges, key_turn=exchanges)
        assert evaluate_check_96(task).band == expected, exchanges


def test_the_minimum_comes_from_policy_and_not_from_a_literal():
    task = make_task(exchanges=7, key_turn=7)
    strict = dataclasses.replace(DEFAULT_POLICY, min_turns_per_model=9)
    assert evaluate_check_96(task, DEFAULT_POLICY).band == "clean"
    assert evaluate_check_96(task, strict).band == "fail"


def test_preflight_reads_the_same_number_the_gate_does():
    """One statement of the requirement, in `Policy`. Two would drift."""
    assert MIN_EXPECTED_EXCHANGES == DEFAULT_POLICY.min_turns_per_model == 7


def test_preflight_still_warns_so_the_operator_sees_the_raw_count():
    task = make_task(exchanges=20)
    task.model_a.conversation = make_conversation(3)
    warnings = validate_task(task).warnings
    assert any("3 exchanges" in w and "check 96" in w for w in warnings)
    assert not validate_task(task).hard_failures


# ---------------------------------------------------------------------------
# Abstention: a short count we cannot believe
# ---------------------------------------------------------------------------


def test_a_login_wall_abstains_because_citations_run_past_the_fetch():
    """The live case: a ChatGPT share page rendered "Log in" and parsed as one
    exchange while the contributor cited turns up to 14 and the other model ran
    to 17. A fail there would be ours, not the contributor's."""
    task = make_task(exchanges=17)
    task.model_b.conversation = make_conversation(1)
    _cite(task, "B", [1, 2, 4, 9, 12, 14])
    _cite(task, "A", [1, 5, 12])
    task.key_turn = KeyTurn(turn_index=12)

    verdict = evaluate_check_96(task)
    assert verdict.band == "not_evaluated"
    assert verdict.error_code is None
    skipped = verdict.measurement.counts["unverifiable_by_model"]
    assert set(skipped) == {"B"}
    assert "beyond the 1 fetched" in skipped["B"]
    # The verified count for the other model is still reported, so a reader can
    # see the abstention rests on B alone.
    assert verdict.measurement.counts["verified_by_model"] == {"A": 17}


def test_a_submission_that_was_never_fetched_is_set_aside_not_counted_short():
    task = make_task(exchanges=20)
    task.model_b.conversation = []
    verdict = evaluate_check_96(task)
    assert verdict.band == "not_evaluated"
    assert (
        verdict.measurement.counts["unverifiable_by_model"]["B"]
        == "conversation was never fetched"
    )


def test_the_other_model_still_fails_on_its_own_when_one_is_unhydratable():
    """The second live case: model B's share link rendered a marketing shell with
    no user turns at all, so nothing was hydrated -- and model A genuinely ran
    two turns. B's blindness does not rescue A."""
    task = make_task(exchanges=2, key_turn=2)
    task.model_b.conversation = []
    _cite(task, "A", [1, 2])
    _cite(task, "B", [1, 2, 3])

    verdict = evaluate_check_96(task)
    assert verdict.band == "fail"
    assert verdict.contributing_items == ["model A: 2 turns"]
    assert set(verdict.measurement.counts["unverifiable_by_model"]) == {"B"}
    # The fail rests on a count that was verified, so B's blindness does not
    # discount it.
    assert verdict.confidence == "high"


def test_no_verifiable_submission_yields_not_evaluated():
    task = make_task(exchanges=20)
    task.model_a.conversation = []
    task.model_b.conversation = []
    verdict = evaluate_check_96(task)
    assert verdict.band == "not_evaluated"
    assert verdict.measurement.counts["submissions_verified"] == 0
    assert verdict.confidence == "low"


def test_a_task_with_no_submissions_at_all_is_not_evaluated():
    task = make_task()
    task.model_a = None
    task.model_b = None
    verdict = evaluate_check_96(task)
    assert verdict.band == "not_evaluated"
    assert verdict.measurement.counts["submissions"] == 0
    assert "No submission was filed" in verdict.measurement.notes


def test_a_count_already_at_the_minimum_needs_no_citation_check():
    """An incomplete fetch can only undercount, so citations past the end cannot
    turn a submission that already clears the bar into a question. Testing the
    other way round would abstain on five of the seventeen live tasks whose
    contributors cite a turn our fetch stopped short of."""
    task = make_task(exchanges=20)
    _cite(task, "A", [1, 26])
    _cite(task, "B", [1, 26])
    verdict = evaluate_check_96(task)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["unverifiable_by_model"] == {}


def test_a_citation_no_fetch_in_the_task_corroborates_is_not_evidence():
    """Both models replay one shared prompt script, so the longest conversation
    fetched for the task bounds how long the real one can be. Three live tasks
    cite a "Turn 35" against conversations of 21, 19 and 6 turns; reading that as
    a fetch failure abstains on a genuine six-turn violation."""
    task = make_task(exchanges=6, key_turn=6)
    _cite(task, "A", [1, 2, 3, 4, 5, 6, 35])
    _cite(task, "B", [1, 2, 3, 4, 5, 6, 35])

    verdict = evaluate_check_96(task)
    assert verdict.band == "fail"
    assert verdict.measurement.counts["longest_fetched"] == 6
    assert verdict.measurement.counts["unverifiable_by_model"] == {}


def test_the_abstention_can_be_switched_off_by_policy():
    """The flag exists so the naive behaviour is nameable, not because it is
    defensible: it fails every login wall."""
    task = make_task(exchanges=17)
    task.model_b.conversation = make_conversation(1)
    _cite(task, "B", [1, 14])
    naive = dataclasses.replace(
        DEFAULT_POLICY, turn_count_requires_complete_conversation=False
    )
    assert evaluate_check_96(task, DEFAULT_POLICY).band == "not_evaluated"
    assert evaluate_check_96(task, naive).band == "fail"


def test_the_key_turn_counts_as_a_citation_for_both_models():
    task = make_task(exchanges=20)
    task.criterion_ratings = []
    task.dimension_ratings = []
    task.key_turn = KeyTurn(turn_index=9)
    assert cited_turns(task, "A") == {9}
    assert cited_turns(task, "B") == {9}


def test_citations_are_read_from_both_rating_surfaces():
    task = make_task(exchanges=20)
    task.criterion_ratings = [
        CriterionRating(criterion_id="C1", model="A", score=0, relevant_turns=[4])
    ]
    task.dimension_ratings = [
        DimensionRating(
            model="A", dimension="Outcome quality", rating=3, relevant_turns=[11]
        )
    ]
    task.key_turn = KeyTurn(turn_index=None)
    assert cited_turns(task, "A") == {4, 11}
    assert cited_turns(task, "B") == set()


# ---------------------------------------------------------------------------
# The counting unit
# ---------------------------------------------------------------------------


def test_a_reply_that_is_only_an_image_still_carries_its_turn():
    """Gemini renders an image answer as markup that strips to no text, and the
    parser stands the turn up with a media marker instead of dropping it. If it
    were dropped the later user messages would look consecutive and the whole
    conversation would collapse onto fewer indices -- which on a seven-turn
    submission is the difference between clean and a fail nobody earned.
    """
    conversation = make_conversation(6)
    conversation += [
        Turn(index=7, role="user", text="render the diagram"),
        Turn(index=7, role="assistant", text="[image]"),
    ]
    task = make_task(exchanges=6, key_turn=6)
    task.model_a.conversation = conversation
    task.model_b.conversation = conversation

    assert task.model_a.exchange_count() == 7
    verdict = evaluate_check_96(task)
    assert verdict.band == "clean"


def test_the_exchange_count_agrees_with_counting_indices_that_carry_a_reply():
    """Our numbering preserves a media-only reply, a stricter count would take
    only indices carrying an assistant turn, and the two agree on all 34 live
    conversation snapshots. They must keep agreeing on the media case, which is
    the only place the two definitions could come apart.
    """
    conversation = make_conversation(6)
    conversation += [
        Turn(index=7, role="user", text="render the diagram"),
        Turn(index=7, role="assistant", text="[image: gyroscope]"),
    ]
    task = make_task(exchanges=7)
    task.model_a.conversation = conversation
    strict = len({t.index for t in conversation if t.role == "assistant"})
    assert task.model_a.exchange_count() == strict == 7


def test_a_trailing_user_turn_with_no_reply_counts_and_so_cannot_manufacture_a_fail():
    """The one place the two counting definitions could disagree in the other
    direction: a fetch that stopped after the user's message leaves an index with
    no reply, which our numbering counts and the strict reading does not. It
    counts, because the alternative is failing a task over where the fetch
    happened to stop. No live snapshot exhibits it.
    """
    conversation = make_conversation(6)
    conversation.append(Turn(index=7, role="user", text="one more thing"))
    task = make_task(exchanges=7)
    task.model_a.conversation = conversation
    task.model_b.conversation = conversation

    strict = len({t.index for t in conversation if t.role == "assistant"})
    assert task.model_a.exchange_count() == 7
    assert strict == 6
    assert evaluate_check_96(task).band == "clean"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_check_is_registered_as_a_net_new_deterministic_input_check():
    spec = REGISTRY[96]
    assert (spec.dimension, spec.sub_dimension) == ("Conversation", "Minimum Turns")
    assert (spec.shape, spec.stage, spec.deterministic) == ("A", 1, True)
    assert spec.evidence == ("conversations",)
    assert "Net-new" in spec.notes
    assert "tab 2 gives it no row" in spec.notes
    assert 96 in BLOCKS["setup_and_inputs"]


def test_the_error_label_is_the_only_one_flagged_provisional():
    """No use of this string was found in the human QC export, so it must not be
    joined on until the customer confirms it."""
    assert ERROR_CODES[96] == {"fail": "[Fail - Insufficient Turns]"}


def test_the_deterministic_path_reaches_a_verdict_with_no_model_call():
    task = make_task(exchanges=20)
    task.model_a.conversation = make_conversation(4)
    _cite(task, "A", [1, 2, 4])
    task.key_turn = KeyTurn(turn_index=4)

    verdicts = {v.check_id: v for v in run_deterministic(task)}
    assert verdicts[96].band == "fail"
    assert 96 in DETERMINISTIC_NOW


def test_the_gate_needs_nothing_but_the_task_and_the_policy():
    """Deterministic means deterministic: no findings argument, no client, and
    the same answer twice."""
    task = make_task(exchanges=5, key_turn=5)
    first = evaluate_check_96(task, Policy())
    second = evaluate_check_96(task, Policy())
    assert first.band == second.band == "fail"
    assert first.measurement.counts == second.measurement.counts
