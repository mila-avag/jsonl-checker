"""V5 alignment: scoped artifacts, per-slot key-turn justifications, unique
links, simplified difficulty, prompt-quality checks, and disabled-check wiring.
"""

from honeybee_qc.config import DEFAULT_POLICY, Policy
from honeybee_qc.findings import ArtifactFinding, KeyTurnJustificationFinding
from honeybee_qc.gates import (
    evaluate_check_91,
    evaluate_check_95,
    evaluate_check_97,
    evaluate_check_110,
)
from honeybee_qc.informed_prompts import (
    build_artifact_prompt,
    build_key_turn_justification_prompt,
)
from honeybee_qc.informed_stages import parse_artifacts
from honeybee_qc.llm import ModelResponse
from honeybee_qc.models import KeyTurn, ModelSubmission, Turn, TurnManifestEntry
from honeybee_qc.projects import HONEYBEE_ASPIRATIONAL, HONEYBEE_L1, disabled_checks_for
from honeybee_qc.tests.fixtures import make_submission, make_task
from honeybee_qc.v5_prompt import (
    PromptClarityFinding,
    evaluate_check_60,
    parse_prompt_clarity,
)


def _response(data: dict) -> ModelResponse:
    return ModelResponse(key="k", data=data)


def test_95_default_prompt_does_not_mention_final_response_scope():
    prompt = build_artifact_prompt(make_task(), make_task().model_a)
    assert "final response" not in prompt.lower()


def test_95_scoped_prompt_asks_only_for_the_final_response():
    policy = Policy(artifact_scope="final_response")
    prompt = build_artifact_prompt(make_task(), make_task().model_a, policy)
    assert "final response" in prompt.lower()


def test_95_final_response_scope_drops_earlier_turn_files():
    """V5 excuses files named in earlier turns that are not in the final response."""
    task = make_task(exchanges=5)
    sub = task.model_a
    policy = Policy(artifact_scope="final_response")
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {
                        "filename": "draft.md",
                        "turn": 2,
                        "quote": "here is a draft",
                        "produced": True,
                    },
                    {
                        "filename": "final.md",
                        "turn": 5,
                        "quote": "here is the final",
                        "produced": True,
                    },
                ],
                "confidence": "high",
            }
        ),
        sub,
        policy,
    )
    assert finding.claimed_files == ["final.md"]
    assert "draft.md" in finding.out_of_scope_files
    assert finding.final_response_turn == 5
    assert finding.missing_files == ["final.md"] or "final.md" in finding.claimed_files


def test_95_conversation_scope_still_counts_an_early_turn_file():
    task = make_task(exchanges=5)
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {
                        "filename": "draft.md",
                        "turn": 2,
                        "quote": "here is a draft",
                        "produced": True,
                    }
                ],
                "confidence": "high",
            }
        ),
        task.model_a,
        DEFAULT_POLICY,
    )
    assert finding.claimed_files == ["draft.md"]
    assert finding.out_of_scope_files == []


def test_95_unplaced_files_stay_in_scope():
    policy = Policy(artifact_scope="final_response")
    finding = parse_artifacts(
        _response(
            {
                "claimed_files": [
                    {
                        "filename": "mystery.md",
                        "turn": None,
                        "quote": "attached",
                        "produced": True,
                    }
                ],
                "confidence": "high",
            }
        ),
        make_task().model_a,
        policy,
    )
    assert finding.claimed_files == ["mystery.md"]


def test_aspirational_does_not_fail_an_empty_final_outputs_section():
    policy = HONEYBEE_ASPIRATIONAL.policy()
    from honeybee_qc.models import ModelSubmission, Turn

    task = make_task()
    task.model_a = ModelSubmission(
        model="A",
        final_outputs_declared=True,
        final_output_uploads=[],
        attachments=["s3://turn-upload-1"],
        conversation=[
            Turn(index=1, role="user", text="make me the deck"),
            Turn(index=1, role="assistant", text="Here it is."),
        ],
    )
    task.model_b = ModelSubmission(
        model="B",
        final_outputs_declared=True,
        final_output_uploads=["s3://final-1"],
        attachments=["s3://final-1"],
        conversation=[
            Turn(index=1, role="assistant", text="Here it is."),
        ],
    )
    verdict = evaluate_check_95(
        task, [ArtifactFinding(model="A", claimed_files=[], uploaded_files=[])], policy
    )
    assert verdict.band != "fail"


def test_110_worst_of_any_slot_with_an_issue():
    clean = KeyTurnJustificationFinding(model="A")
    bad = KeyTurnJustificationFinding(
        model="B", describes_selected_turn=False, issues=["wrong turn"]
    )
    verdict = evaluate_check_110(make_task(), [clean, bad])
    assert verdict.band == "non_fail"
    assert any("B:" in item for item in verdict.contributing_items)


def test_110_per_slot_prompt_uses_that_models_justification():
    task = make_task()
    task.model_b.key_turn = KeyTurn(
        turn_index=9, justification="B delivers the chart here."
    )
    prompt = build_key_turn_justification_prompt(task, slot="B")
    assert "B delivers the chart here." in prompt
    assert "Turn 9" in prompt


def test_91_abstains_when_no_gemini_manifest_was_recorded():
    assert evaluate_check_91(make_task()).band == "not_evaluated"


def test_91_fails_at_four_non_unique_gemini_turns():
    task = make_task()
    url = "https://gemini.google.com/share/aaaaaaaa"
    task.model_a.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link=url)
        for i in range(1, 5)
    ]
    task.model_a.declared_provider = "gemini"
    verdict = evaluate_check_91(task)
    assert verdict.band == "fail"


def test_91_non_fails_at_one_to_three():
    task = make_task()
    url = "https://gemini.google.com/share/aaaaaaaa"
    task.model_a.turn_manifest = [
        TurnManifestEntry(turn_number=1, prompt_text="p", share_link=url),
        TurnManifestEntry(turn_number=2, prompt_text="p", share_link=url),
    ]
    task.model_a.declared_provider = "gemini"
    assert evaluate_check_91(task).band == "non_fail"


def test_91_html_export_without_share_urls_is_not_evaluated():
    """Spark B HTML/MHTML has no per-turn Gemini share URLs — 91 is N/A."""
    task = make_task()
    task.model_a.declared_provider = "gemini"
    task.model_a.turn_manifest = []
    task.model_b.final_link = ""
    task.model_b.declared_provider = "gemini"
    task.model_b.transcript_pdf = "https://example.com/Gemini%20Spark.mhtml"
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link="")
        for i in range(1, 8)
    ]
    verdict = evaluate_check_91(task)
    assert verdict.band == "not_evaluated"
    assert "HTML" in (verdict.measurement.notes or "")


def test_91_extensionless_s3_chat_download_is_not_evaluated():
    """Live HTML-B downloads are opaque Scale CDS URLs, not *.mhtml paths."""
    task = make_task()
    task.model_a.declared_provider = "gemini"
    task.model_a.turn_manifest = []
    task.model_b.final_link = ""
    task.model_b.declared_provider = "gemini"
    task.model_b.transcript_pdf = (
        "https://scale-cds-public-us-west-2.s3.amazonaws.com/"
        "6463e58315d308d12ddb5385/hB9StvK3fPC9WQZ"
    )
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link="")
        for i in range(1, 8)
    ]
    assert evaluate_check_91(task).band == "not_evaluated"


def test_91_named_pdf_export_without_share_urls_still_fails():
    """A PDF transcript is not HTML-B evidence — missing Gemini links still fail."""
    task = make_task()
    task.model_a.turn_manifest = []
    task.model_b.final_link = ""
    task.model_b.declared_provider = "gemini"
    task.model_b.transcript_pdf = "https://example.com/model_B_transcript.pdf"
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link="")
        for i in range(1, 5)
    ]
    assert evaluate_check_91(task).band == "fail"


def test_91_still_fails_share_link_gemini_even_if_html_also_present():
    """Do not blanket-ignore Unique Links when B filed real Gemini share URLs."""
    task = make_task()
    url = "https://gemini.google.com/share/aaaaaaaa"
    task.model_b.declared_provider = "gemini"
    task.model_b.final_link = url
    task.model_b.transcript_pdf = "https://example.com/Gemini%20Spark.mhtml"
    task.model_b.turn_manifest = [
        TurnManifestEntry(turn_number=i, prompt_text="p", share_link=url)
        for i in range(1, 5)
    ]
    assert evaluate_check_91(task).band == "fail"


def test_97_abstains_without_a_simplified_trajectory():
    assert evaluate_check_97(make_task()).band == "not_evaluated"


def test_97_fails_when_simplified_pass_rate_exceeds_the_ceiling():
    from honeybee_qc.models import CriterionRating

    task = make_task(n_criteria=4)
    task.project_id = HONEYBEE_ASPIRATIONAL.project_id
    task.simplified = make_submission("S", exchanges=2)
    task.criterion_ratings += [
        CriterionRating(criterion_id=c.criterion_id, model="S", score=1)
        for c in task.rubric
    ]
    verdict = evaluate_check_97(task, HONEYBEE_ASPIRATIONAL.policy())
    assert verdict.band == "fail"


def test_60_defaults_to_clean_when_the_judge_omits_an_assessment():
    finding = parse_prompt_clarity(_response({"confidence": "high"}))
    assert finding.assessment == "unique_ground_truth"
    assert evaluate_check_60(make_task(), finding).band == "clean"


def test_60_fails_a_subjective_prompt_when_quoted():
    finding = PromptClarityFinding(
        assessment="multiple_or_subjective",
        prompt_quote="make it nice",
    )
    assert evaluate_check_60(make_task(), finding).band == "fail"


def test_disabled_checks_are_dropped_from_an_aspirational_report():
    from honeybee_qc.cli import run_deterministic

    task = make_task()
    task.project_id = HONEYBEE_ASPIRATIONAL.project_id
    ids = {v.check_id for v in run_deterministic(task)}
    assert 96 not in ids
    assert 220 not in ids
    assert 460 not in ids
    assert 70 not in ids
    assert 90 in ids
    assert 91 in ids
    assert 97 in ids


def test_l1_keeps_the_failing_checks_v5_added():
    assert 96 not in HONEYBEE_L1.disabled_checks
    assert 60 in HONEYBEE_L1.disabled_checks
    assert disabled_checks_for(make_task().project_id) == frozenset()
