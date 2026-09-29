"""Paste-safety families the fixer must re-check on every run."""

from __future__ import annotations

from honeybee_qc.fixer_invariants import (
    SUBAGENT_PROMPT,
    audit_task,
    dimension_is_na,
    empty_turns_ok,
    explains_inversion,
    foreign_slots,
    form_likert_side,
    inversion_is_unexplained,
    likert_magnitude_ok,
    mean_dimension_side,
    patched_task,
)
from honeybee_qc.tests.fixtures import make_task


def test_form_likert_side_only_four_is_a_tie():
    assert form_likert_side(1) == "left"
    assert form_likert_side(3) == "left"
    assert form_likert_side(4) == "tie"
    assert form_likert_side(5) == "right"
    assert form_likert_side(7) == "right"


def test_mean_side_follows_averages_not_dominance():
    task = make_task("t-mean", dimension_rating_a=4, dimension_rating_b=4, likert=7)
    for dim in task.dimension_ratings:
        if dim.model == "B" and dim.dimension == "Communication quality":
            dim.rating = 1
        if dim.model == "B" and dim.dimension == "Trust & grounding":
            dim.rating = 1
        if dim.model == "B" and dim.dimension == "Tool & connector reliability":
            dim.rating = 1
        if dim.model == "A" and dim.dimension == "Interaction efficiency":
            dim.rating = 3
    # A still loses Interaction efficiency, so dominance is mixed; mean favors A.
    assert mean_dimension_side(task.dimension_ratings, "A", "B") == "left"
    task.sxs.justification = "Model B is much better and had no issues."
    assert inversion_is_unexplained(task) is True
    task.sxs.justification = (
        "Despite lower dimension scores, Model B is much better because of "
        "one decisive failure on A."
    )
    assert inversion_is_unexplained(task) is False


def test_explains_inversion_requires_an_inversion_marker():
    assert explains_inversion("Model B is much better.") is False
    assert explains_inversion("Verdict: Model C is much better (Likert 7/7).") is False
    assert explains_inversion("Even though A wins the dimensions, B is better.") is True


def test_na_dimension_is_detected_from_justification():
    task = make_task("t-na")
    dim = next(
        d
        for d in task.dimension_ratings
        if d.dimension == "Tool & connector reliability" and d.model == "A"
    )
    dim.justification = "N/A — no tools were used."
    dim.not_applicable = False
    assert dimension_is_na(dim) is True


def test_audit_flags_empty_turns_pfo_and_sxs_tie_on_a_win():
    task = make_task("t-audit", likert=3)
    task.model_a.produced_final_outcome = False
    task.sxs.justification = "This is a tie between the two."
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.relevant_turns = []
        if dim.model == "B" and dim.dimension == "Memory & personalization":
            dim.not_applicable = True
            dim.justification = "N/A no memory used"
            dim.relevant_turns = [3]
    hits = audit_task(task)
    families = {h.family for h in hits}
    assert "produced_outcome" in families
    assert "dimension_turns" in families
    assert "sxs_scale" in families


def test_audit_does_not_flag_empty_cites_on_a_five():
    task = make_task("t-five-empty")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.rating = 5
            dim.relevant_turns = []
    hits = [
        h
        for h in audit_task(task)
        if h.family == "dimension_turns" and "Outcome quality" in h.detail
    ]
    assert hits == []
    dim = next(
        d
        for d in task.dimension_ratings
        if d.model == "A" and d.dimension == "Outcome quality"
    )
    assert empty_turns_ok(dim) is True


def test_audit_does_not_flag_empty_cites_on_a_four():
    task = make_task("t-four-empty")
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.rating = 4
            dim.relevant_turns = []
    hits = [
        h
        for h in audit_task(task)
        if h.family == "dimension_turns" and "Outcome quality" in h.detail
    ]
    assert hits == []
    dim = next(
        d
        for d in task.dimension_ratings
        if d.model == "A" and d.dimension == "Outcome quality"
    )
    assert empty_turns_ok(dim) is True


def test_patched_task_applies_dimension_and_ranking_patches():
    task = make_task("t-patch", likert=7)
    patched = patched_task(
        task,
        [
            {
                "kind": "dimension_rating",
                "slot": "B",
                "dimension": "Communication quality",
                "new_value": 1,
            },
            {
                "kind": "preference_justification",
                "comparison": "AB",
                "new_value": "Despite the scores, Model B is much better.",
            },
        ],
    )
    comm = next(
        d
        for d in patched.dimension_ratings
        if d.model == "B" and d.dimension == "Communication quality"
    )
    assert comm.rating == 1
    assert "Despite" in patched.sxs.justification


def test_subagent_prompt_names_every_family():
    text = SUBAGENT_PROMPT.lower()
    assert "produced outcome" in text or "check 82" in text
    assert "310" in text
    assert "only 4 is a tie" in text
    assert "inconsistent ranking" in text
    assert "transcript" in text
    assert "model b" in text
    assert "slightly better" in text
    assert "not-applicable" in text or "n/a vs numeric" in text


def test_foreign_slots_ignore_verdict_legend_and_plan_b():
    assert foreign_slots("Plan B was the backup.", "A", "C") == ()
    assert foreign_slots("The Response B is much better.", "A", "C") == ("B",)
    assert (
        foreign_slots(
            "Verdict: C is much better (1 = A much better, 7 = C much better).\n"
            "The Response B still wins the file.",
            "A",
            "C",
        )
        == ("B",)
    )


def test_verdict_legend_is_not_a_tie_writeup():
    task = make_task("t-legend-tie", likert=7)
    task.sxs.justification = (
        "Verdict: Model B is much better (1 = A much better, 4 = tie, 7 = B much better).\n"
        "Model B is much better than Model A on the only required file."
    )
    families = {h.family for h in audit_task(task)}
    assert "sxs_scale" not in families


def test_retarget_foreign_slots_rewrites_model_b_on_ac():
    from honeybee_qc.fixer_invariants import retarget_foreign_slots

    c_win = retarget_foreign_slots("The Response B is much better than Model A.", "A", "C", 7)
    assert "Response C" in c_win
    assert "Response B" not in c_win
    a_win = retarget_foreign_slots("Model A is much better than Model B.", "A", "C", 1)
    assert a_win == "Model A is much better than Model C."
    body = (
        "Verdict: C is slightly better (1 = A much better, 7 = C much better).\n"
        "Model C is slightly better on the only required file."
    )
    assert likert_magnitude_ok(5, body) is True
    assert likert_magnitude_ok(5, "Model C is much better on the file.") is False
    assert likert_magnitude_ok(7, "Model C is slightly better on the file.") is False


def test_user_only_manifest_conversation_is_not_a_transcript_hold():
    """Ingest plants user-only turns from the turn manifest. That is not hydrate."""
    task = make_task("t-manifest-only")
    task.model_a.conversation = [
        t for t in task.model_a.conversation if t.role == "user"
    ]
    task.model_b.conversation = [
        t for t in task.model_b.conversation if t.role == "user"
    ]
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.relevant_turns = [2, 4]
    families = {h.family for h in audit_task(task)}
    assert "transcript" not in families


def test_short_capture_is_a_transcript_hold():
    task = make_task("t-stub")
    task.model_a.conversation = [
        t for t in task.model_a.conversation if t.index <= 2
    ]
    task.model_a.corroborated_exchanges = 20
    families = {h.family for h in audit_task(task)}
    assert "transcript" in families


def test_hydrated_empty_assistant_cite_is_a_transcript_hold():
    task = make_task("t-empty-asst")
    for turn in task.model_a.conversation:
        if turn.index == 4 and turn.role == "assistant":
            turn.text = ""
    for dim in task.dimension_ratings:
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.relevant_turns = [4]
    families = {h.family for h in audit_task(task)}
    assert "transcript" in families


def test_audit_flags_model_b_on_ac_and_copied_justs():
    task = make_task("t-audit-ac", likert=7)
    task.sxs.left = "A"
    task.sxs.right = "C"
    task.sxs.justification = "The Response B is much better than Model A."
    blob = (
        "The assistant walked through the gyroscope animation in careful "
        "detail, named every constraint, and exported the classroom clip "
        "with consistent colour coding on the labeled diagram."
    )
    for dim in task.dimension_ratings:
        if dim.dimension == "Communication quality":
            dim.justification = blob
        if dim.model == "A" and dim.dimension == "Outcome quality":
            dim.relevant_turns = [1]
    families = {h.family for h in audit_task(task)}
    assert "sxs_slot" in families
    assert "copied_justification" in families
    assert "dimension_turns" in families


def test_patched_task_accepts_not_applicable_score():
    task = make_task("t-patch-na")
    patched = patched_task(
        task,
        [
            {
                "kind": "dimension_rating",
                "slot": "A",
                "dimension": "Tool & connector reliability",
                "new_value": "not-applicable",
            }
        ],
    )
    tool = next(
        d
        for d in patched.dimension_ratings
        if d.model == "A" and d.dimension == "Tool & connector reliability"
    )
    assert tool.rating is None
    assert tool.not_applicable is True
