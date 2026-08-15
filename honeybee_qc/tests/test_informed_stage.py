"""Tests for the informed pass: checks 80, 95, 110, 220, 280, 310, 450, 470.

Four properties carry the most risk and are tested hardest:

  1. Shape discipline. 80 and 110 can never fail a task; 95, 220, and 470 have no
     middle band. A gate that invents a band the spec does not have corrupts the
     task verdict, because the rollup takes the worst band across all checks.
  2. Check 220 cannot fail a task on one opinion. It is the only check where a
     single finding is fatal with no lesser band to absorb a mistake.
  3. Check 95 subtracts sets in Python. If a model's claim of what it uploaded
     could decide the score, the check would audit nothing.
  4. Check 80 replays the prompts. Diffing against turn 1 turns every correct
     target outcome in a revised conversation into a false issue.
"""

from __future__ import annotations

import dataclasses

from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.findings import (
    ArtifactFinding,
    AutofailFinding,
    JustificationFinding,
    KeyTurnJustificationFinding,
    TargetOutcomeFinding,
    VerdictFinding,
)
from honeybee_qc.gates import (
    evaluate_check_80,
    evaluate_check_95,
    evaluate_check_110,
    evaluate_check_220,
    evaluate_check_470,
    evaluate_relevant_turns,
)
from honeybee_qc.informed_prompts import (
    TARGET_OUTCOME_SCHEMA,
    build_artifact_prompt,
    build_autofail_prompt,
    build_criterion_turns_prompt,
    build_dimension_justification_prompt,
    build_dimension_turns_prompt,
    build_key_turn_justification_prompt,
    build_target_outcome_prompt,
    build_verdict_prompt,
)
from honeybee_qc.informed_stages import (
    INFORMED_CHECKS,
    apply_artifact_confirmation,
    build_informed_requests,
    empty_justification_findings,
    estimate_informed_calls,
    missing_turn_findings,
    parse_artifacts,
    parse_autofails,
    parse_justifications,
    parse_key_turn_justification,
    parse_relevant_turns,
    parse_target_outcome,
    run_informed_stage,
)
from honeybee_qc.llm import FakeModelClient, ModelResponse
from honeybee_qc.models import CriterionRating, KeyTurn, Sxs, Turn
from honeybee_qc.tests.fixtures import make_task

TASK = make_task()


def _response(data: dict) -> ModelResponse:
    return ModelResponse(key="k", data=data)


# ---------------------------------------------------------------------------
# Shape discipline
# ---------------------------------------------------------------------------


def test_80_cannot_fail_even_when_every_entry_is_wrong():
    findings = [
        TargetOutcomeFinding(entry=f"e{i}", classification="contradicts_prompt")
        for i in range(10)
    ]
    assert evaluate_check_80(TASK, findings).band == "non_fail"


def test_110_cannot_fail_even_when_every_signal_is_bad():
    finding = KeyTurnJustificationFinding(
        describes_selected_turn=False,
        claims_are_accurate=False,
        connects_to_core_value=False,
        issues=["wrong turn", "fabricated claim"],
    )
    assert evaluate_check_110(TASK, finding).band == "non_fail"


def test_95_and_470_have_no_middle_band():
    partial = ArtifactFinding(
        model="A", claimed_files=["a.md", "b.md"], uploaded_files=["a.md"],
        missing_files=["b.md"],
    )
    assert evaluate_check_95(TASK, [partial]).band == "fail"
    assert evaluate_check_470(TASK, VerdictFinding(states_preference=False)).band == "fail"
    assert evaluate_check_470(TASK, VerdictFinding(states_preference=True)).band == "clean"


# ---------------------------------------------------------------------------
# 95 -- artifacts
# ---------------------------------------------------------------------------


def test_95_is_clean_when_the_response_produced_no_files_at_all():
    """The spec makes this an explicit 5, so an empty extraction must not fail."""
    finding = ArtifactFinding(model="A", claimed_files=[], uploaded_files=[])
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_95_ignores_files_the_model_only_described():
    sub = dataclasses.replace(TASK.model_a, attachments=["report.md"])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "report.md", "turn": 4, "quote": "here it is",
                     "produced": True},
                    {"filename": "followup.md", "turn": 5,
                     "quote": "I could also draft", "produced": False},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.claimed_files == ["report.md"]
    assert finding.missing_files == []


def test_95_matches_on_the_bare_filename_across_path_and_case():
    sub = dataclasses.replace(TASK.model_a, attachments=["/uploads/Report.MD"])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "output/report.md", "turn": 4, "quote": "q",
                     "produced": True}
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == []


def test_95_does_not_name_files_when_the_manifest_has_no_filenames():
    """Scale uploads are CDN URLs ending in a content ID with no extension, and
    the objects serve no Content-Disposition. Treating that as a filename makes
    every claimed file look absent and fabricates a fail on every task."""
    sub = dataclasses.replace(
        TASK.model_a,
        attachments=[
            "https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/ZQjy98e0JqvrEIg",
            "https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/4L9Du-9aNBhH_7m",
        ],
    )
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "outline.txt", "turn": 3, "quote": "q", "produced": True},
                    {"filename": "catalog.txt", "turn": 5, "quote": "q", "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.basis == "count"
    assert finding.missing_files == []
    assert finding.shortfall == 0
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_95_still_catches_a_shortfall_when_it_cannot_name_the_missing_file():
    sub = dataclasses.replace(
        TASK.model_a,
        attachments=["https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/ZQjy98e"],
    )
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "a.txt", "turn": 3, "quote": "q", "produced": True},
                    {"filename": "b.txt", "turn": 4, "quote": "q", "produced": True},
                    {"filename": "c.txt", "turn": 5, "quote": "q", "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.shortfall == 2
    verdict = evaluate_check_95(TASK, [finding])
    assert verdict.band == "fail"
    assert verdict.measurement.numerator == 2
    # Nothing is named, because nothing can be.
    assert all("a.txt" not in item for item in verdict.contributing_items)


def test_95_treats_a_duplicate_upload_suffix_as_the_same_file():
    """A contributor re-uploading a same-named file gets it auto-suffixed
    " (2)" by the upload widget, not by the model -- that is not evidence the
    claimed file is missing."""
    sub = dataclasses.replace(
        TASK.model_a, attachments=["anti-abuse-security-directive (2).md"]
    )
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "anti-abuse-security-directive.md", "turn": 4,
                     "quote": "here it is", "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == []
    assert finding.fuzzy_matched_files == [
        ("anti-abuse-security-directive.md", "anti-abuse-security-directive (2).md")
    ]
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_95_treats_a_version_suffix_as_the_same_file():
    sub = dataclasses.replace(
        TASK.model_a, attachments=["technical_directive_anti_bot_defense_v2.md"]
    )
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "technical_directive_anti_bot_defense.md",
                     "turn": 4, "quote": "here it is", "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == []
    assert len(finding.fuzzy_matched_files) == 1


def test_95_still_fails_a_genuinely_different_filename():
    """Below the similarity threshold, a near-miss must not be waved through --
    "report.pdf" and "summary.pdf" are two different deliverables, not a
    renamed upload."""
    sub = dataclasses.replace(TASK.model_a, attachments=["summary.pdf"])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "report.pdf", "turn": 4, "quote": "here it is",
                     "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == ["report.pdf"]
    assert finding.fuzzy_matched_files == []
    assert evaluate_check_95(TASK, [finding]).band == "fail"


def test_95_fuzzy_threshold_is_configurable():
    sub = dataclasses.replace(TASK.model_a, attachments=["report_final.pdf"])
    strict_finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "report.pdf", "turn": 4, "quote": "here it is",
                     "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
        dataclasses.replace(DEFAULT_POLICY, filename_fuzzy_match_threshold=0.99),
    )
    assert strict_finding.missing_files == ["report.pdf"]

    lenient_finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "report.pdf", "turn": 4, "quote": "here it is",
                     "produced": True},
                ],
                "confidence": "high",
            }
        ),
        sub,
        dataclasses.replace(DEFAULT_POLICY, filename_fuzzy_match_threshold=0.7),
    )
    assert lenient_finding.missing_files == []


def test_95_decides_on_the_upload_manifest_not_on_what_the_model_claims():
    """The model extracts; Python subtracts. A model that says a file was uploaded
    cannot talk the check out of a finding."""
    sub = dataclasses.replace(TASK.model_a, attachments=[])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "deck.pdf", "turn": 6, "quote": "attached",
                     "produced": True}
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == ["deck.pdf"]
    assert evaluate_check_95(TASK, [finding]).band == "fail"


# ---------------------------------------------------------------------------
# 220 -- autofail and its second opinion
# ---------------------------------------------------------------------------


def test_220_never_fails_a_task_on_an_unconfirmed_allegation():
    alleged = AutofailFinding(
        criterion_id="C3", quote="send the draft", why_outcome_is_useless="irreversible",
        confirmed=False,
    )
    verdict = evaluate_check_220(TASK, [alleged])
    assert verdict.band == "clean"
    assert verdict.measurement.counts["alleged_but_not_confirmed"] == 1


def test_220_fails_only_once_the_second_opinion_agrees():
    confirmed = AutofailFinding(
        criterion_id="C3", quote="send the draft", why_outcome_is_useless="irreversible",
        confirmed=True,
    )
    assert evaluate_check_220(TASK, [confirmed]).band == "fail"


def test_220_second_opinion_runs_only_when_something_was_alleged():
    """Most rubrics have no autofail, and the confirmation round should cost
    nothing on those."""
    client = FakeModelClient(lambda r: {"autofail_criteria": [], "confidence": "high"})
    result = run_informed_stage(TASK, client, workers=1)
    assert result.autofails == []
    # The estimate has to be given the policy now that a judgment is not one
    # call: the sampled checks draw three times each, so the bill is no longer
    # the judgment count and a dry run that reported it would understate.
    assert result.calls == estimate_informed_calls(TASK, DEFAULT_POLICY)


# ---------------------------------------------------------------------------
# 280 / 310 -- relevant turns
# ---------------------------------------------------------------------------


def test_280_and_310_never_count_the_same_turn_twice():
    findings = parse_relevant_turns(
        _response(
            {"judgments": [{"item_id": "C1", "incorrect_turns": [3], "reasoning": ""}],
             "confidence": "high"}
        ),
        280,
        "A",
    ) + parse_relevant_turns(
        _response(
            {"judgments": [{"item_id": "Outcome quality", "incorrect_turns": [3],
                            "reasoning": ""}], "confidence": "high"}
        ),
        310,
        "A",
    )
    assert evaluate_relevant_turns(TASK, 280, findings).measurement.numerator == 1
    assert evaluate_relevant_turns(TASK, 310, findings).measurement.numerator == 1


def test_a_turn_cited_past_the_end_of_what_we_fetched_is_not_counted_against_anyone():
    """Hydration truncates some conversations. The contributor cited a turn we
    never retrieved, so the citation may be perfectly correct -- counting it as
    incorrect charges them for our failed fetch."""
    findings = parse_relevant_turns(
        _response(
            {
                "judgments": [
                    {"item_id": "C1", "incorrect_turns": [3, 35], "reasoning": ""}
                ],
                "confidence": "high",
            }
        ),
        280,
        "A",
        last_turn=20,
    )
    assert findings[0].incorrect_turns == [3]
    assert findings[0].unverifiable_turns == [35]

    verdict = evaluate_relevant_turns(TASK, 280, findings)
    assert verdict.measurement.numerator == 1
    assert verdict.measurement.counts["unverifiable_turns"] == 1


def test_280_population_is_the_contributors_own_score_zero_set():
    """Built from the ratings the contributor gave, not from corrected ones."""
    task = dataclasses.replace(
        TASK,
        criterion_ratings=[
            CriterionRating(criterion_id="C1", model="A", score=0, relevant_turns=[2]),
            CriterionRating(criterion_id="C2", model="A", score=1, relevant_turns=[]),
        ],
    )
    prompt = build_criterion_turns_prompt(task, task.model_a)
    assert "[C1]" in prompt
    assert "[C2]" not in prompt


def test_a_score_zero_criterion_citing_no_turn_is_recorded_separately():
    task = dataclasses.replace(
        TASK,
        criterion_ratings=[
            CriterionRating(criterion_id="C1", model="A", score=0, relevant_turns=[])
        ],
    )
    findings = missing_turn_findings(task, 280)
    assert len(findings) == 1 and findings[0].missing_turn

    counted = evaluate_relevant_turns(task, 280, findings)
    assert counted.measurement.counts["missing_turns"] == 1
    assert counted.measurement.counts["incorrect_turns"] == 0

    lenient = evaluate_relevant_turns(
        task, 280, findings,
        dataclasses.replace(DEFAULT_POLICY, count_missing_turns_as_incorrect=False),
    )
    assert lenient.band == "clean"


# ---------------------------------------------------------------------------
# 450 -- justifications
# ---------------------------------------------------------------------------


def test_450_thresholds_are_counted_within_one_justification_not_pooled():
    """Two justifications with one unsupported claim each are an issue but not a
    fail; one justification with two is a fail."""
    split = [
        JustificationFinding(item_id="A::Outcome quality", unsupported_claims=1),
        JustificationFinding(item_id="A::Trust & grounding", unsupported_claims=1),
    ]
    from honeybee_qc.gates import evaluate_check_450

    assert evaluate_check_450(TASK, split).band == "non_fail"

    together = [JustificationFinding(item_id="A::Outcome quality", unsupported_claims=2)]
    assert evaluate_check_450(TASK, together).band == "fail"


def test_450_population_includes_the_ranking_justification():
    requests = build_informed_requests(TASK)
    keys = [r.key for r in requests if r.metadata["check_id"] == 450]
    assert any(k.endswith("ranking_justification") for k in keys)
    assert sum(1 for k in keys if "dimension_justifications" in k) == 2


def test_a_dimension_rated_without_any_justification_is_generic():
    task = dataclasses.replace(
        TASK,
        dimension_ratings=[
            dataclasses.replace(TASK.dimension_ratings[0], justification="")
        ],
    )
    findings = empty_justification_findings(task)
    assert len(findings) == 1
    assert "is_generic" in findings[0].triggered_conditions()


def test_450_drops_a_judgment_with_no_item_id():
    assert parse_justifications(
        _response({"justifications": [{"item_id": "", "unsupported_claims": 5}],
                   "confidence": "high"})
    ) == []


def test_450_generic_needs_the_judge_to_find_nothing_quotable():
    """The Trust & grounding justification from 6a7190c542368ece68aa26a0.

    Two real platforms and a fabricated domain are named, so the judge can quote
    a specific back and its generic call is overridden. The vacuous companion has
    nothing to quote and stands.
    """
    specific, vacuous = parse_justifications(
        _response(
            {
                "justifications": [
                    {
                        "item_id": "A::Trust & grounding",
                        "rated_value": 2,
                        "is_generic": True,
                        "specifics_quoted": [
                            "Athletes Untapped and TeachMeTo",
                            '"debatemarket.com"',
                        ],
                    },
                    {
                        "item_id": "A::Communication quality",
                        "rated_value": 5,
                        "is_generic": True,
                        "specifics_quoted": [],
                    },
                ],
                "confidence": "high",
            }
        )
    )
    assert specific.generic is False
    assert specific.has_any_issue() is False
    assert vacuous.generic is True
    assert vacuous.triggered_conditions() == ["is_generic"]


def test_450_truncation_abstention_is_parsed_and_left_uncounted():
    (f,) = parse_justifications(
        _response(
            {
                "justifications": [
                    {"item_id": "A::Collaboration quality", "unverifiable_claims": 2}
                ],
                "confidence": "high",
            }
        )
    )
    assert f.unverifiable_claims == 2
    assert f.triggered_conditions() == []


def test_450_prompt_asks_for_specifics_before_the_generic_call():
    prompt = build_dimension_justification_prompt(TASK, TASK.model_a)
    assert "specifics_quoted" in prompt
    assert "2-3 sentences" in prompt
    assert "unverifiable_claims" in prompt


# ---------------------------------------------------------------------------
# 80 -- target outcome replay
# ---------------------------------------------------------------------------


def test_80_prompt_carries_every_user_turn_in_order():
    prompt = build_target_outcome_prompt(TASK)
    assert "### Turn 1" in prompt and "### Turn 20" in prompt
    assert prompt.index("### Turn 1") < prompt.index("### Turn 2")


def test_80_recovers_the_prompt_turns_from_the_conversation_when_the_form_omits_them():
    """Snowflake ingest records only the seeded prompt; the later turns exist only
    inside the hydrated conversation, and without them every revision is invisible."""
    task = dataclasses.replace(TASK, prompts=[])
    prompt = build_target_outcome_prompt(task)
    assert "### Turn 1" in prompt
    assert "A: user message 1" in prompt


def test_80_falls_back_to_the_seeded_prompt_when_nothing_was_hydrated():
    task = dataclasses.replace(
        TASK,
        prompts=[],
        model_a=dataclasses.replace(TASK.model_a, conversation=[]),
        model_b=dataclasses.replace(TASK.model_b, conversation=[]),
    )
    assert TASK.seeded_prompt in build_target_outcome_prompt(task)


def test_80_schema_never_asks_the_judge_to_invent_missing_requirements():
    """Tab 2 grades this dimension on whether the outcomes listed are correct,
    never on whether the list is exhaustive, so the schema has no slot for the
    judge to report requirements the list leaves out."""
    assert "missing_requirements" not in TARGET_OUTCOME_SCHEMA["properties"]
    assert "missing_requirements" not in TARGET_OUTCOME_SCHEMA["required"]
    entry_enum = TARGET_OUTCOME_SCHEMA["properties"]["entries"]["items"]["properties"][
        "classification"
    ]["enum"]
    assert "missing" not in entry_enum


def test_80_prompt_never_asks_for_missing_requirements():
    prompt = build_target_outcome_prompt(TASK)
    assert "missing_requirements" not in prompt


def test_80_parser_only_ever_produces_classified_entries():
    """No second pass over what the list leaves out: every finding the parser
    returns traces back to one of the contributor's own entries."""
    findings = parse_target_outcome(
        _response(
            {
                "entries": [
                    {
                        "entry": "an 11-second clip",
                        "classification": "supported",
                        "established_turn": 2,
                        "reasoning": "turn 2 asks for it",
                    }
                ],
                "confidence": "high",
            }
        )
    )
    assert len(findings) == 1
    assert evaluate_check_80(TASK, findings).band == "clean"


def test_80_does_not_count_an_entry_the_conversation_never_asked_for():
    """An anticipated ask the conversation did not end up making is a normal thing
    to have written down in advance. The judge sees the prompts and neither the
    input artifacts nor the responses, so "nothing in the prompts asks for it" is
    not yet evidence of a defect -- tab 2 says the list "may include additional
    components beyond those required by the initial prompt"."""
    findings = [
        TargetOutcomeFinding(entry="a spec table", classification="supported"),
        TargetOutcomeFinding(
            entry="The user might request a comparison between models.",
            classification="not_required",
        ),
    ]
    verdict = evaluate_check_80(TASK, findings)

    assert verdict.band == "clean"
    assert verdict.measurement.numerator == 0
    # Judged and recorded, just outside the active scope. Narrowing hides nothing.
    assert verdict.measurement.counts["recorded_not_counted"] == {"not_required": 1}
    assert verdict.measurement.counts["not_required"] == 1


def test_80_still_flags_an_entry_that_contradicts_the_prompts():
    """The one defect mode that survives. Tab 2's non-fail cell names an entry that
    "contradicts one or more prompts", and the judge can decide that from the
    prompts alone, which is all it is shown."""
    findings = [
        TargetOutcomeFinding(entry="a spec table", classification="supported"),
        TargetOutcomeFinding(
            entry="uses only household materials",
            classification="contradicts_prompt",
        ),
        TargetOutcomeFinding(entry="a supplier list", classification="not_required"),
    ]
    verdict = evaluate_check_80(TASK, findings)

    assert verdict.band == "non_fail"
    assert verdict.measurement.numerator == 1
    assert verdict.contributing_items == ["uses only household materials"]
    assert verdict.measurement.counts["classifications_by_item"] == {
        "uses only household materials": "contradicts_prompt"
    }


def test_80_scope_is_a_policy_field_and_not_a_hardcoded_gate_rule():
    """A definitional ruling from the spec drops in by widening one tuple. If this
    ever needs a gate edit, the scope has been hardcoded somewhere it should not
    be."""
    findings = [
        TargetOutcomeFinding(entry="a supplier list", classification="not_required"),
        TargetOutcomeFinding(entry="a spec table", classification="supported"),
    ]
    assert evaluate_check_80(TASK, findings).band == "clean"

    widened = dataclasses.replace(
        DEFAULT_POLICY,
        target_outcome_defect_classifications=("contradicts_prompt", "not_required"),
    )
    verdict = evaluate_check_80(TASK, findings, widened)
    assert verdict.band == "non_fail"
    assert verdict.measurement.numerator == 1


# ---------------------------------------------------------------------------
# Scope discipline: each prompt sees only what its check is allowed to judge
# ---------------------------------------------------------------------------


def test_the_target_outcome_prompt_carries_tab_2s_carve_out_for_extra_entries():
    """The judge was asked to classify entries against the prompts without being
    told the spec lets the list run beyond them, which is what turned deliverable
    content it could not trace into `not_required`."""
    prompt = build_target_outcome_prompt(TASK)
    assert "beyond those required by the initial prompt" in prompt
    assert "artifacts and/or final model responses" in prompt


def test_the_target_outcome_prompt_never_shows_a_model_response():
    """80 is a prompt-versus-list check; a response in context invites the model to
    score the conversation instead."""
    prompt = build_target_outcome_prompt(TASK)
    assert "model reply" not in prompt


def test_the_verdict_prompt_shows_only_the_comparison_justification():
    """470 is the narrowest check in the spec and drifts into re-judging quality
    the moment it can see anything else."""
    prompt = build_verdict_prompt(TASK)
    assert TASK.sxs.justification in prompt
    assert "model reply" not in prompt
    for dim in TASK.dimension_ratings:
        assert dim.justification not in prompt


def test_the_key_turn_prompt_carries_the_selected_turn_and_its_justification():
    prompt = build_key_turn_justification_prompt(TASK)
    assert f"Turn {TASK.key_turn.turn_index}" in prompt
    assert TASK.key_turn.justification in prompt


def test_the_autofail_prompt_shows_the_rubric_and_the_prompts_but_no_ratings():
    prompt = build_autofail_prompt(TASK)
    assert "[C1]" in prompt
    assert "Likert" not in prompt


def test_the_justification_prompt_pairs_each_rating_with_its_own_text():
    prompt = build_dimension_justification_prompt(TASK, TASK.model_a)
    for dim in [d for d in TASK.dimension_ratings if d.model == "A"]:
        assert dim.justification in prompt
    for dim in [d for d in TASK.dimension_ratings if d.model == "B"]:
        assert dim.justification not in prompt


# ---------------------------------------------------------------------------
# Stage plumbing
# ---------------------------------------------------------------------------


def test_estimate_matches_what_the_stage_actually_builds():
    assert len(build_informed_requests(TASK)) == estimate_informed_calls(TASK)


def test_no_requests_are_built_for_work_the_contributor_never_did():
    task = dataclasses.replace(
        TASK,
        target_outcome=[],
        rubric=[],
        criterion_ratings=[],
        dimension_ratings=[],
        key_turn=KeyTurn(turn_index=5, justification=""),
        sxs=Sxs(likert=4, justification=""),
    )
    checks = {r.metadata["check_id"] for r in build_informed_requests(task)}
    # Named individually rather than as an equality against {95}, so a check
    # whose evidence this task does still carry is not swept in by accident.
    assert checks & {70, 80, 110, 220, 280, 310, 450, 470} == set()
    assert 95 in checks


def test_a_failed_call_is_an_error_and_never_a_finding():
    client = FakeModelClient(
        lambda r: ModelResponse(key=r.key, error="upstream timeout")
    )
    result = run_informed_stage(TASK, client, workers=1)
    assert result.errors
    assert result.justifications == []
    assert result.autofails == []
    assert result.verdict is None
    assert {v.check_id for v in result.verdicts} == set(INFORMED_CHECKS)


def test_every_informed_check_reports_a_verdict_even_when_nothing_was_judged():
    client = FakeModelClient(lambda r: {"confidence": "high"})
    result = run_informed_stage(TASK, client, workers=1)
    assert len(result.verdicts) == len(INFORMED_CHECKS)
    assert all(v.band in {"clean", "non_fail", "fail", "not_evaluated"} for v in result.verdicts)


# ---------------------------------------------------------------------------
# Abstention: an item the judge could not see reaches neither the numerator nor
# the band.
#
# 18 of 33 conversations in the first live run exceeded the render budget, and 82
# of 305 informed judge calls named the cut or an absent deliverable in their own
# reasoning and then returned a finding anyway. Each test below is that shape: the
# judge says what it could not open, and the gate must come out unchanged.
# ---------------------------------------------------------------------------


def test_110_abstains_rather_than_faulting_a_turn_it_was_not_shown():
    """All three of 110's questions are about the selected turn, so a judge that
    could not read that turn has answered none of them and its answers are
    discarded."""
    finding = parse_key_turn_justification(
        _response(
            {
                "describes_selected_turn": False,
                "claims_are_accurate": False,
                "connects_to_core_value": False,
                "unverifiable": True,
                "why_unverifiable": "turn 5 sits past the truncation line",
                "issues": [
                    {"issue": "claims a video was delivered", "quote": "delivers the"}
                ],
                "confidence": "high",
            }
        )
    )
    assert finding.unverifiable
    assert finding.is_issue is False
    assert finding.issues == []
    assert finding.unverifiable_issues == ["claims a video was delivered"]
    assert "truncation line" in finding.why_unverifiable

    verdict = evaluate_check_110(TASK, finding)
    assert verdict.band == "clean"
    assert verdict.contributing_items == []


def test_110_can_withhold_one_claim_without_abstaining_from_the_whole_judgment():
    """The turn was readable; one allegation about it was not checkable. Only that
    allegation leaves, and the rest of the judgment still counts."""
    finding = parse_key_turn_justification(
        _response(
            {
                "describes_selected_turn": True,
                "claims_are_accurate": False,
                "connects_to_core_value": True,
                "unverifiable": False,
                "why_unverifiable": "",
                "issues": [
                    {"issue": "misnames the turn", "quote": "q", "unverifiable": False},
                    {
                        "issue": "says the clip is 11 seconds",
                        "quote": "q",
                        "unverifiable": True,
                        "why_unverifiable": "the clip's content is not in the transcript",
                    },
                ],
                "confidence": "high",
            }
        )
    )
    assert finding.issues == ["misnames the turn"]
    assert finding.unverifiable_issues == ["says the clip is 11 seconds"]
    assert finding.is_issue is True
    assert evaluate_check_110(TASK, finding).band == "non_fail"


def test_95_never_compares_a_delivery_whose_filename_it_could_not_read():
    """A turn labelled as delivering a file the transcript does not carry has no
    readable filename. Guessing one and subtracting it from the manifest is how a
    correctly uploaded file becomes a missing file, on a check with no middle
    band."""
    sub = dataclasses.replace(TASK.model_a, attachments=["report.md"])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "report.md", "turn": 3, "quote": "here it is",
                     "produced": True, "unverifiable": False},
                    {"filename": "(unnamed video)", "turn": 9, "quote": "0:00 / 0:10",
                     "produced": True, "unverifiable": True,
                     "why_unverifiable": "delivered past the truncation line"},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.claimed_files == ["report.md"]
    assert finding.unverifiable_files == ["(unnamed video)"]
    assert finding.missing_files == []
    assert "truncation line" in finding.why_unverifiable

    verdict = evaluate_check_95(TASK, [finding])
    assert verdict.band == "clean"
    assert verdict.measurement.numerator == 0
    # Out of the denominator as well: one file was compared, not two.
    assert verdict.measurement.counts["files_claimed"] == 1


def test_95_keeps_failing_on_a_named_file_that_was_never_uploaded():
    """The abstention must not become an amnesty for the deliveries the judge
    could read."""
    sub = dataclasses.replace(TASK.model_a, attachments=[])
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "deck.pdf", "turn": 6, "quote": "attached",
                     "produced": True, "unverifiable": False},
                    {"filename": "clip.mp4", "turn": 9, "quote": "0:00 / 0:10",
                     "produced": True, "unverifiable": True,
                     "why_unverifiable": "content absent from the transcript"},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.missing_files == ["deck.pdf"]
    assert evaluate_check_95(TASK, [finding]).band == "fail"


def test_95_abstention_also_leaves_the_count_basis_alone():
    """With an opaque upload manifest the check can only compare counts, and an
    unreadable delivery must not raise the claimed side of that subtraction."""
    sub = dataclasses.replace(
        TASK.model_a,
        attachments=["https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/ZQjy98e"],
    )
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": "a.txt", "turn": 3, "quote": "q", "produced": True,
                     "unverifiable": False},
                    {"filename": "b.txt", "turn": 4, "quote": "q", "produced": True,
                     "unverifiable": True, "why_unverifiable": "turn 4 was truncated"},
                ],
                "confidence": "high",
            }
        ),
        sub,
    )
    assert finding.basis == "count"
    assert finding.shortfall == 0
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_a_turn_the_judge_says_it_could_not_open_is_reported_and_never_counted():
    """Distinct from the turn-past-the-end case: turn 7 is inside the conversation
    the audit fetched, so Python cannot tell it was cut from the render. Only the
    judge can, which is why it has a field to say so in."""
    findings = parse_relevant_turns(
        _response(
            {
                "judgments": [
                    {
                        "item_id": "C1",
                        "incorrect_turns": [3, 7],
                        "unverifiable_turns": [7],
                        "why_unverifiable": "turn 7 is past the truncation marker",
                        "reasoning": "",
                    }
                ],
                "confidence": "high",
            }
        ),
        280,
        "A",
        last_turn=20,
    )
    assert findings[0].incorrect_turns == [3]
    assert findings[0].unverifiable_turns == [7]
    assert "truncation marker" in findings[0].why_unverifiable

    verdict = evaluate_relevant_turns(TASK, 280, findings)
    assert verdict.measurement.numerator == 1
    assert verdict.measurement.counts["unverifiable_turns"] == 1
    assert verdict.band == "non_fail"


def test_three_unverifiable_turns_do_not_reach_280s_fail_threshold():
    """Three counted turns fail 280. Three the judge could not open score nothing
    at all, which is the whole property."""
    findings = parse_relevant_turns(
        _response(
            {
                "judgments": [
                    {
                        "item_id": "C1",
                        "incorrect_turns": [4, 5, 6],
                        "unverifiable_turns": [4, 5, 6],
                        "why_unverifiable": "all three sit past the truncation marker",
                        "reasoning": "",
                    }
                ],
                "confidence": "high",
            }
        ),
        280,
        "A",
        last_turn=20,
    )
    verdict = evaluate_relevant_turns(TASK, 280, findings)
    assert verdict.measurement.numerator == 0
    assert verdict.band == "clean"
    assert verdict.measurement.counts["unverifiable_turns"] == 3
    assert verdict.contributing_items == []


def test_450_zeroes_the_counts_of_a_justification_the_judge_could_not_read():
    (f,) = parse_justifications(
        _response(
            {
                "justifications": [
                    {
                        "item_id": "A::Outcome quality",
                        "rated_value": 2,
                        "unsupported_claims": 3,
                        "inaccurate_primary_claims": 1,
                        "is_generic": True,
                        "contradicts_verdict_claims": 2,
                        "contradicting_quotes": ["it rendered nothing"],
                        "unverifiable": True,
                        "why_unverifiable": "the clip it rates is not in the transcript",
                    }
                ],
                "confidence": "high",
            }
        )
    )
    assert f.unverifiable
    assert f.triggered_conditions() == []
    assert f.has_any_issue() is False
    assert f.unsupported_claims == 0
    assert f.inaccurate_primary_claims == 0
    assert "not in the transcript" in f.why_unverifiable


def test_an_abstained_justification_leaves_450s_denominator_and_is_still_reported():
    """The rating stage's rule, applied to 450: an item nobody could audit is not a
    clean item. Counting it clean would report a justification as audited that the
    transcript never supported."""

    def respond(request):
        if request.metadata.get("check_id") != 450:
            return {"confidence": "high"}
        return {
            "justifications": [
                {
                    "item_id": "A::Outcome quality",
                    "rated_value": 4,
                    "unsupported_claims": 4,
                    "unverifiable": True,
                    "why_unverifiable": "rates a video whose content is absent",
                }
            ],
            "confidence": "high",
        }

    task = dataclasses.replace(
        TASK, dimension_ratings=[TASK.dimension_ratings[0]], sxs=Sxs(likert=4)
    )
    result = run_informed_stage(task, FakeModelClient(respond), workers=1)

    assert [f.item_id for f in result.unverifiable_justifications] == [
        "A::Outcome quality"
    ]
    assert result.justifications == []
    verdict = next(v for v in result.verdicts if v.check_id == 450)
    assert verdict.band == "not_evaluated"
    assert "does not contain" in verdict.measurement.notes
    assert result.to_dict()["justifications_unverifiable"] == [
        {"item": "A::Outcome quality", "why": "rates a video whose content is absent"}
    ]


def test_one_abstention_does_not_take_the_other_justifications_out_of_450():
    def respond(request):
        if request.metadata.get("check_id") != 450:
            return {"confidence": "high"}
        return {
            "justifications": [
                {
                    "item_id": "A::Outcome quality",
                    "unsupported_claims": 4,
                    "unverifiable": True,
                    "why_unverifiable": "rates a video whose content is absent",
                },
                {"item_id": "A::Trust & grounding", "unsupported_claims": 2},
            ],
            "confidence": "high",
        }

    task = dataclasses.replace(
        TASK, dimension_ratings=[TASK.dimension_ratings[0]], sxs=Sxs(likert=4)
    )
    result = run_informed_stage(task, FakeModelClient(respond), workers=1)

    assert [f.item_id for f in result.justifications] == ["A::Trust & grounding"]
    verdict = next(v for v in result.verdicts if v.check_id == 450)
    assert verdict.measurement.denominator == 1
    assert verdict.band == "fail"


# ---------------------------------------------------------------------------
# The prompts have to name the markers, not gesture at them
# ---------------------------------------------------------------------------

TRUNCATION_MARKER = "[... conversation truncated here;"
DELIVERABLE_MARKER = "[DELIVERABLE PRODUCED HERE"


def test_every_informed_prompt_that_renders_a_conversation_names_the_markers():
    """The system prompt already said "say so rather than guessing" while the
    judges guessed. What was missing was the four strings `render_conversation`
    actually emits, and a field to put the answer in."""
    prompts = [
        build_artifact_prompt(TASK, TASK.model_a),
        build_key_turn_justification_prompt(TASK),
        build_criterion_turns_prompt(TASK, TASK.model_a),
        build_dimension_turns_prompt(TASK, TASK.model_a),
        build_dimension_justification_prompt(TASK, TASK.model_a),
    ]
    for prompt in prompts:
        assert TRUNCATION_MARKER in prompt
        assert DELIVERABLE_MARKER in prompt
        assert "MUST NOT" in prompt or "and nowhere else" in prompt
        assert "costs you nothing" in prompt or "costs nothing" in prompt


def test_the_turn_prompts_point_the_abstention_at_its_own_field():
    for prompt in (
        build_criterion_turns_prompt(TASK, TASK.model_a),
        build_dimension_turns_prompt(TASK, TASK.model_a),
    ):
        assert "`unverifiable_turns`" in prompt
        assert "`why_unverifiable`" in prompt


def test_the_autofail_and_verdict_prompts_get_no_abstention_field():
    """Neither judge is shown a conversation: 220 reads the prompts and the rubric
    text, 470 reads one paragraph the contributor wrote. There is nothing for a
    truncated transcript to hide from either, and a field they cannot use would
    invite them to abstain from work they can do."""
    for prompt in (build_autofail_prompt(TASK), build_verdict_prompt(TASK)):
        assert TRUNCATION_MARKER not in prompt
        assert DELIVERABLE_MARKER not in prompt
        assert "unverifiable" not in prompt


# ---------------------------------------------------------------------------
# 95 -- confirming a missing file before publishing it as one
# ---------------------------------------------------------------------------
#
# 95 has no middle band, so every name the filename subtraction gets wrong is a
# task-level fail with nothing to absorb it. On a manual audit of nine of these
# fails, five were the comparison rather than the contributor: a screenshot whose
# capture-tool name no distance metric ties to the contributor's name for the same
# image, and inline code blocks read as undelivered files. The confirming pass runs
# only where the subtraction already found a shortfall.


def _artifact_finding(claimed: list[str], uploaded: list[str]):
    sub = dataclasses.replace(TASK.model_a, attachments=uploaded)
    return parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {"filename": name, "turn": 4, "quote": "here it is",
                     "produced": True}
                    for name in claimed
                ],
                "confidence": "high",
            }
        ),
        sub,
    )


def test_95_confirmation_clears_a_screenshot_under_the_capture_tools_own_name():
    finding = _artifact_finding(
        ["dashboard-after-fix.png"], ["Screenshot 2026-08-02 at 11.14.31.png"]
    )
    # String distance cannot bridge these, and should not be asked to.
    assert finding.missing_files == ["dashboard-after-fix.png"]

    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {
                    "claimed_filename": "dashboard-after-fix.png",
                    "assessment": "renamed",
                    "matched_upload": "Screenshot 2026-08-02 at 11.14.31.png",
                    "why": "turn 4 delivers the dashboard image and it is the only png uploaded",
                }
            ],
            "confidence": "high",
        },
    )

    assert finding.confirmed_missing == []
    assert finding.missing_count == 0
    verdict = evaluate_check_95(TASK, [finding])
    assert verdict.band == "clean"
    # The name is still published, with the reason, rather than vanishing.
    assert verdict.measurement.counts["files_missing_before_confirmation"] == 1
    assert verdict.measurement.counts["cleared_by_confirmation"] == [
        "A:dashboard-after-fix.png (renamed)"
    ]
    assert "Screenshot" in verdict.measurement.counts["confirmation_reasons"][
        "A:dashboard-after-fix.png"
    ]


def test_95_confirmation_clears_content_the_model_printed_inline():
    finding = _artifact_finding(["analysis.py"], [])
    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {
                    "claimed_filename": "analysis.py",
                    "assessment": "not_a_file",
                    "matched_upload": "",
                    "why": "turn 3 prints the script in a fenced code block; nothing was attached",
                }
            ],
            "confidence": "high",
        },
    )
    assert finding.confirmed_missing == []
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_95_confirmation_still_fails_a_genuinely_absent_file():
    """The pass must be able to say no, or it is just a way of never failing."""
    finding = _artifact_finding(["report.pdf"], ["summary.pdf"])
    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {
                    "claimed_filename": "report.pdf",
                    "assessment": "missing",
                    "matched_upload": "",
                    "why": "turn 4 hands over a report and only an unrelated summary was uploaded",
                }
            ],
            "confidence": "high",
        },
    )
    assert finding.confirmed_missing == ["report.pdf"]
    assert evaluate_check_95(TASK, [finding]).band == "fail"


def test_95_confirmation_that_abstains_does_not_count_the_file():
    finding = _artifact_finding(["report.pdf"], ["summary.pdf"])
    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {
                    "claimed_filename": "report.pdf",
                    "assessment": "unverifiable",
                    "matched_upload": "",
                    "why": "the delivering turn sits past the truncation line",
                }
            ],
            "confidence": "low",
        },
    )
    assert finding.confirmed_missing == []
    assert evaluate_check_95(TASK, [finding]).band == "clean"


def test_95_confirmation_cannot_add_a_file_the_subtraction_never_flagged():
    """A judge volunteering an opinion about some other file changes nothing."""
    finding = _artifact_finding(["report.pdf"], ["summary.pdf"])
    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {"claimed_filename": "invented.pdf", "assessment": "missing",
                 "matched_upload": "", "why": "not one of the flagged names"},
                {"claimed_filename": "report.pdf", "assessment": "renamed",
                 "matched_upload": "summary.pdf", "why": "same document"},
            ],
            "confidence": "high",
        },
    )
    assert set(finding.confirmations) == {"report.pdf"}
    assert finding.confirmed_missing == []


def test_95_an_unreadable_assessment_leaves_the_file_counted():
    finding = _artifact_finding(["report.pdf"], ["summary.pdf"])
    apply_artifact_confirmation(
        finding,
        {
            "files": [
                {"claimed_filename": "report.pdf", "assessment": "probably fine",
                 "matched_upload": "", "why": ""}
            ],
            "confidence": "high",
        },
    )
    # Conservative direction: an answer we cannot read is not a clearance.
    assert finding.confirmed_missing == ["report.pdf"]
    assert evaluate_check_95(TASK, [finding]).band == "fail"


def test_95_a_finding_with_no_confirmation_behaves_exactly_as_before():
    """The policy flag's off branch, and every task audited before this existed."""
    finding = _artifact_finding(["report.pdf"], ["summary.pdf"])
    assert not finding.confirmation_ran
    assert finding.confirmed_missing == ["report.pdf"]
    assert evaluate_check_95(TASK, [finding]).band == "fail"


def test_95_confirmation_only_runs_where_a_file_was_read_as_missing(monkeypatch):
    """A submission whose claimed files all matched must not spend a call."""
    calls: list[str] = []

    def respond(request):
        calls.append(request.key)
        if request.metadata.get("check_id") == 95 and "artifact_confirm" not in request.key:
            return {
                "claimed_files": [
                    {"filename": "report.pdf", "turn": 4, "quote": "here",
                     "produced": True, "unverifiable": False, "why_unverifiable": ""}
                ],
                "confidence": "high",
            }
        return {"confidence": "high"}

    task = dataclasses.replace(
        TASK,
        model_a=dataclasses.replace(TASK.model_a, attachments=["report.pdf"]),
        model_b=dataclasses.replace(TASK.model_b, attachments=["report.pdf"]),
    )
    run_informed_stage(task, FakeModelClient(respond), workers=1)

    assert not [key for key in calls if "artifact_confirm" in key]


def test_95_confirmation_fires_once_per_submission_with_a_missing_file():
    seen: list[str] = []

    def respond(request):
        if "artifact_confirm" in request.key:
            seen.append(request.key)
            return {
                "files": [
                    {"claimed_filename": "report.pdf", "assessment": "not_a_file",
                     "matched_upload": "", "why": "printed inline"}
                ],
                "confidence": "high",
            }
        if request.metadata.get("check_id") == 95:
            return {
                "claimed_files": [
                    {"filename": "report.pdf", "turn": 4, "quote": "here",
                     "produced": True, "unverifiable": False, "why_unverifiable": ""}
                ],
                "confidence": "high",
            }
        return {"confidence": "high"}

    task = dataclasses.replace(
        TASK,
        model_a=dataclasses.replace(TASK.model_a, attachments=["unrelated.csv"]),
        model_b=dataclasses.replace(TASK.model_b, attachments=["unrelated.csv"]),
    )
    result = run_informed_stage(task, FakeModelClient(respond), workers=1)

    assert len(seen) == 2  # one per submission, not one per file
    verdict = next(v for v in result.verdicts if v.check_id == 95)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["confirmation_ran_for"] == ["A", "B"]
