"""Check 85: prompt references against the task's universe.

The risk this check carries is almost entirely false positives, and the two things
it compares are both routinely unreadable on the live export: `Task.prompts` is
empty there and the file manifest arrives as opaque CDN paths. Most of what is
tested here is therefore the refusal to conclude anything -- an unreadable
manifest, an unquoted allegation, and a prompt nobody recorded all have to come
back `not_evaluated` or clean rather than as a violation.

The requirement is stated as a flat fail with no middle band, and the check is
nonetheless running as shape B. That is a temporary demotion, and the reason is
measured rather than cautious: across three corpora 11 of 13 escalations sat on
tasks whose entire "declared supplied" file list was model OUTPUT, described to
the judge as contributor input. Three defects behind that are fixed here -- the
supplied/produced split, the parser no longer hardcoding the two questions that
decide whether a missing file is a violation, and a `found_in_materials` field for
a judge that looked and found the thing -- but they change what the judge is asked,
so their output is unvalidated on a scored run. Until it is, a wrong finding costs
a note instead of a task. `test_the_demotion_is_temporary_and_says_so` pins the
restore condition.
"""

from __future__ import annotations

import dataclasses

import pytest

from honeybee_qc.env_context import (
    assistant_mentioned_filenames,
    file_references,
    input_manifest,
    prompt_texts,
    scan_file_references,
    universe_materials,
    unnamed_artifact_references,
)
from honeybee_qc.errors import ERROR_CODES
from honeybee_qc.findings import EnvironmentContextFinding
from honeybee_qc.gates import evaluate_check_85
from honeybee_qc.informed_prompts import build_environment_context_prompt
from honeybee_qc.informed_stages import (
    INFORMED_CHECKS,
    build_informed_requests,
    can_judge_environment_context,
    estimate_informed_calls,
    parse_environment_context,
    run_informed_stage,
)
from honeybee_qc.llm import FakeModelClient, ModelResponse
from honeybee_qc.models import InputArtifact, Turn
from honeybee_qc.registry import REGISTRY
from honeybee_qc.tests.fixtures import make_task

# What human QC writes, and what 85 emitted before the demotion.
SPEC_LABEL = "[Fail - Environment Context Violation]"
# What it emits now. Provisional, and ours: QC has no non-fail precedent here.
REPORTED_LABEL = "[Non-Fail - Environment Context Violation]"


def _task(prompt: str, attachments_a=None, attachments_b=None):
    """A task whose prompt and file manifest are the only things under test."""
    task = make_task("env")
    task.prompts = [Turn(index=1, role="user", text=prompt)]
    task.seeded_prompt = prompt
    for sub, attachments in ((task.model_a, attachments_a), (task.model_b, attachments_b)):
        sub.attachments = list(attachments if attachments is not None else [])
        sub.conversation = []
    return task


def _finding(**kw) -> EnvironmentContextFinding:
    base = dict(
        reference="Priya Nair",
        kind="person",
        quote="ask Priya Nair for the signed copy",
        checked_against=["input file manifest (no files recorded)"],
        why_outside_universe="No material names this person.",
    )
    base.update(kw)
    return EnvironmentContextFinding(**base)  # type: ignore[arg-type]


def _response(data: dict) -> ModelResponse:
    return ModelResponse(key="k", data=data)


# ---------------------------------------------------------------------------
# Registry and label contract
# ---------------------------------------------------------------------------


def test_85_is_registered_so_it_reports_rather_than_vanishing():
    spec = REGISTRY[85]
    assert (spec.dimension, spec.sub_dimension) == ("Prompt", "Environment Context")
    # Shape B, against a requirement the audit workflow tab states with no middle
    # band. See the module docstring and the test below.
    assert spec.shape == "B"
    assert 85 in INFORMED_CHECKS


def test_the_demotion_is_temporary_and_says_so():
    """A band that moved for a reason has to carry the reason and the way back.

    The failure mode this guards is a demotion becoming permanent by being
    forgotten: 85 reports instead of failing because the fixes to what its judge is
    asked are unvalidated, not because the requirement softened. Two edits restore
    it -- the registry shape and the gate's band -- and both are named in the notes
    so whoever validates the fixes can find them without reading the history.
    """
    notes = REGISTRY[85].notes
    assert "DEMOTED" in notes
    assert "Restore band 'A'" in notes
    assert "11 of 13" in notes, "the measurement that justifies it must travel too"

    task = _task("Use the numbers in q3_actuals.csv.", ["budget.xlsx"])
    scan = scan_file_references(task)
    verdict = evaluate_check_85(task, scan.findings, scan, entity_half_ran=False)
    assert "Reported rather than failed" in verdict.measurement.notes


def test_the_error_strings_keep_the_join_key_human_qc_writes():
    """The fail label is the join key against the QC sheet, and it must survive the
    demotion even though 85 no longer emits it. Dropping it would not fail loudly:
    a QC row citing this requirement would simply stop counting as ground truth for
    85, which reads as the customer never having raised it."""
    from honeybee_qc.errors import RETIRED_CODES

    assert ERROR_CODES[85] == {"non_fail": REPORTED_LABEL}
    assert RETIRED_CODES[85] == {"fail": SPEC_LABEL}


def test_a_violation_is_reported_under_the_spec_label():
    task = _task("Use the numbers in q3_actuals.csv.", ["budget.xlsx"])
    scan = scan_file_references(task)
    verdict = evaluate_check_85(task, scan.findings, scan, entity_half_ran=False)
    assert verdict.band == "non_fail"
    assert verdict.error_code == REPORTED_LABEL


# ---------------------------------------------------------------------------
# The deterministic half: files
# ---------------------------------------------------------------------------


def test_a_prompt_naming_a_file_that_was_provided_is_clean():
    task = _task(
        "Summarise the findings in Ridgewater_PortfolioPlaybook_v2.pdf.",
        ["Ridgewater_PortfolioPlaybook_v2.pdf"],
    )
    scan = scan_file_references(task)
    assert scan.ran
    assert scan.named_references == ["Ridgewater_PortfolioPlaybook_v2.pdf"]
    assert scan.findings == []
    assert evaluate_check_85(task, scan.findings, scan).band == "clean"


def test_a_prompt_naming_a_file_absent_from_the_manifest_fails():
    task = _task(
        "Reconcile market_sizing_assumptions.xlsx against the plan.",
        ["some_other_input.pdf"],
    )
    scan = scan_file_references(task)
    assert [f.reference for f in scan.findings] == ["market_sizing_assumptions.xlsx"]

    verdict = evaluate_check_85(task, scan.findings, scan)
    assert verdict.band == "non_fail"
    assert verdict.measurement.counts["file_violations"] == 1
    # The finding carries its own evidence: the prompt's words, and the manifest
    # it was compared against.
    assert "market_sizing_assumptions.xlsx" in verdict.contributing_items[0]
    assert "some_other_input.pdf" in scan.findings[0].checked_against[0]


def test_a_file_is_matched_on_its_basename_across_a_path():
    task = _task(
        "Open Matters/RidgewaterRefactor/Playbook/playbook_v2.pdf and check the fees.",
        ["https://cdn.example.com/uploads/PLAYBOOK_V2.PDF"],
    )
    scan = scan_file_references(task)
    assert scan.ran
    assert scan.findings == []


def test_an_opaque_manifest_abstains_instead_of_failing_every_attached_file():
    """Live uploads are CDN paths whose last segment is a content ID. Reading one
    as a filename makes every file the prompt names look absent, which would fail
    the task for attaching something."""
    task = _task(
        "Use the tables in q3_actuals.csv.",
        ["https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/ZQjy98e0JqvrEIg"],
    )
    scan = scan_file_references(task)
    assert not scan.ran
    assert scan.manifest.opaque and not scan.manifest.named
    assert scan.findings == []

    verdict = evaluate_check_85(task, scan.findings, scan, entity_half_ran=False)
    assert verdict.band == "not_evaluated"
    assert "no recoverable filename" in verdict.measurement.notes


def test_an_empty_manifest_abstains_because_input_artifacts_are_not_ingested():
    """`Task` exposes no input manifest of its own, so an empty attachment list
    does not mean no files were supplied -- the taskattempt export carries an
    input-artifact step this build does not read."""
    task = _task("Summarise the attached deck, exec_readout.pptx.")
    scan = scan_file_references(task)
    assert not scan.ran
    assert "not ingested" in scan.blocked_reason
    assert evaluate_check_85(task, scan.findings, scan).band == "not_evaluated"


def test_a_model_reported_file_never_fails_a_manifest_that_could_not_be_read():
    task = _task("Use q3_actuals.csv.")
    scan = scan_file_references(task)
    alleged = _finding(reference="q3_actuals.csv", kind="file")
    verdict = evaluate_check_85(task, [alleged], scan, entity_half_ran=True)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["references_unverifiable"] == 1


def test_a_linked_file_is_not_a_claim_that_it_was_attached():
    assert file_references("See https://example.com/reports/q3.pdf for context") == []
    assert file_references("See q3.pdf") == ["q3.pdf"]


def test_only_real_extensions_count_as_filenames():
    """An open `\\.\\w+` pattern reads domains and version numbers as files, and
    every one of those would be a fail this check cannot defend."""
    assert file_references("debatemarket.com went live in v2.0, per Acme Inc.") == []
    assert file_references("notes.txt and chart.png") == ["notes.txt", "chart.png"]


def test_an_unnamed_attachment_reference_is_reported_and_never_counted():
    """"Here is the report" names no file, so any file in the manifest satisfies
    it. It is recorded so a reader can see the prompt referred to an artifact."""
    task = _task("Here is the report. Tighten the summary.", ["deck.pptx"])
    scan = scan_file_references(task)
    assert scan.unnamed_references
    assert scan.findings == []
    assert evaluate_check_85(task, scan.findings, scan).band == "clean"


def test_the_artifact_detector_is_what_spots_an_unnamed_reference():
    assert unnamed_artifact_references("I have attached the numbers")
    assert not unnamed_artifact_references("What are the latest MikroTik switches?")


# ---------------------------------------------------------------------------
# The model-judged half: entities, people, events
# ---------------------------------------------------------------------------


def test_a_prompt_inventing_a_person_fails():
    task = _task("Send the signed copy to Priya Nair, who approved it last Tuesday.")
    scan = scan_file_references(task)
    verdict = evaluate_check_85(task, [_finding()], scan, entity_half_ran=True)
    assert verdict.band == "non_fail"
    assert verdict.error_code == REPORTED_LABEL
    assert verdict.measurement.counts["entity_violations"] == 1


def test_a_prompt_inventing_an_event_fails():
    task = _task("Follow up on the decisions from the March offsite.")
    scan = scan_file_references(task)
    finding = _finding(
        reference="the March offsite",
        kind="event",
        quote="the decisions from the March offsite",
    )
    verdict = evaluate_check_85(task, [finding], scan, entity_half_ran=True)
    assert verdict.band == "non_fail"


def test_a_real_world_entity_the_task_legitimately_involves_does_not_fail():
    """A prompt about MikroTik hardware names MikroTik. A prompt is not required
    to supply the outside world, and the spec's bar is the task's universe."""
    task = _task("What are the latest MikroTik access points and PoE switches?")
    scan = scan_file_references(task)
    finding = _finding(
        reference="MikroTik",
        kind="organisation",
        quote="the latest MikroTik access points",
        real_world_public_entity=True,
    )
    verdict = evaluate_check_85(task, [finding], scan, entity_half_ran=True)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["violations"] == 0


def test_a_reference_the_prompt_only_proposes_does_not_fail():
    task = _task("Draft a note to a hypothetical client considering the upgrade.")
    scan = scan_file_references(task)
    finding = _finding(
        reference="a hypothetical client",
        kind="person",
        quote="a hypothetical client considering the upgrade",
        presented_as_existing=False,
    )
    assert evaluate_check_85(task, [finding], scan, entity_half_ran=True).band == "clean"


def test_a_reference_the_work_does_not_need_does_not_fail():
    task = _task("Over coffee with Sam, I realised the pricing page needs work.")
    scan = scan_file_references(task)
    finding = _finding(
        reference="Sam",
        kind="person",
        quote="Over coffee with Sam",
        required_to_complete=False,
    )
    assert evaluate_check_85(task, [finding], scan, entity_half_ran=True).band == "clean"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_an_allegation_without_a_quote_cannot_stand():
    task = _task("Send the signed copy to Priya Nair.")
    scan = scan_file_references(task)
    verdict = evaluate_check_85(task, [_finding(quote="")], scan, entity_half_ran=True)
    assert verdict.band == "clean"
    assert verdict.measurement.counts["unevidenced_discarded"] == 1
    assert "discarded" in verdict.measurement.notes


def test_an_allegation_that_names_nothing_it_checked_cannot_stand():
    task = _task("Send the signed copy to Priya Nair.")
    scan = scan_file_references(task)
    verdict = evaluate_check_85(
        task, [_finding(checked_against=[])], scan, entity_half_ran=True
    )
    assert verdict.band == "clean"
    assert verdict.measurement.counts["unevidenced_discarded"] == 1


def test_the_same_reference_reported_twice_counts_once():
    task = _task("Ask Priya Nair, then confirm with Priya Nair.")
    scan = scan_file_references(task)
    verdict = evaluate_check_85(
        task, [_finding(), _finding()], scan, entity_half_ran=True
    )
    assert verdict.measurement.counts["violations"] == 1


def test_declared_file_checks_synthesises_a_finding_when_neither_signal_is_present():
    findings = parse_environment_context(
        _response(
            {
                "declared_file_checks": [
                    {
                        "filename": "AnalystMemo_v3.docx",
                        "prompt_quote": "go through AnalystMemo_v3.docx carefully",
                        "prompt_treats_as_available_to_read": True,
                        "conversation_shows_attachment_or_upload": False,
                        "conversation_shows_independent_content": False,
                        "evidence_summary": "No attachment marker and no detail "
                        "beyond what the user typed.",
                    }
                ],
                "references": [],
                "confidence": "high",
            }
        )
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.reference == "AnalystMemo_v3.docx"
    assert f.kind == "file"
    assert f.basis == "model"
    assert f.is_violation


def test_declared_file_checks_produces_nothing_when_either_signal_is_present():
    for signal in ("conversation_shows_attachment_or_upload", "conversation_shows_independent_content"):
        findings = parse_environment_context(
            _response(
                {
                    "declared_file_checks": [
                        {
                            "filename": "AnalystMemo_v3.docx",
                            "prompt_quote": "go through AnalystMemo_v3.docx",
                            "prompt_treats_as_available_to_read": True,
                            "conversation_shows_attachment_or_upload": False,
                            "conversation_shows_independent_content": False,
                            "evidence_summary": "",
                            signal: True,
                        }
                    ],
                    "references": [],
                    "confidence": "high",
                }
            )
        )
        assert findings == [], f"unexpected finding when {signal} is true"


def test_declared_file_checks_is_skipped_when_the_prompt_never_treats_it_as_available():
    findings = parse_environment_context(
        _response(
            {
                "declared_file_checks": [
                    {
                        "filename": "AnalystMemo_v3.docx",
                        "prompt_quote": "the file we discussed earlier",
                        "prompt_treats_as_available_to_read": False,
                        "conversation_shows_attachment_or_upload": False,
                        "conversation_shows_independent_content": False,
                        "evidence_summary": "Just a passing mention.",
                    }
                ],
                "references": [],
                "confidence": "high",
            }
        )
    )
    assert findings == []


def test_declared_file_checks_missing_fields_is_discarded_not_defaulted():
    findings = parse_environment_context(
        _response(
            {
                "declared_file_checks": [
                    {
                        "filename": "",
                        "prompt_quote": "go through the record",
                        "prompt_treats_as_available_to_read": True,
                        "conversation_shows_attachment_or_upload": False,
                        "conversation_shows_independent_content": False,
                        "evidence_summary": "No evidence.",
                    }
                ],
                "references": [],
                "confidence": "high",
            }
        )
    )
    assert findings == []


def test_the_parser_keeps_the_evidence_and_normalises_an_unknown_kind():
    findings = parse_environment_context(
        _response(
            {
                "references": [
                    {
                        "reference": "the Q3 board memo",
                        "kind": "artefact",
                        "quote": "as agreed in the Q3 board memo",
                        "checked_against": ["target outcome list (2 entries)", " "],
                        "why_outside_universe": "Nothing supplies it.",
                        "presented_as_existing": True,
                        "required_to_complete": True,
                        "real_world_public_entity": False,
                    },
                    {"reference": "", "kind": "person"},
                ],
                "confidence": "medium",
            }
        )
    )
    assert len(findings) == 1
    assert findings[0].kind == "other"
    assert findings[0].checked_against == ["target outcome list (2 entries)"]
    assert findings[0].confidence == "medium"
    assert findings[0].basis == "model"


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


def test_a_task_with_no_prompt_text_is_not_evaluated():
    task = _task("")
    task.prompts = []
    task.seeded_prompt = ""
    scan = scan_file_references(task)
    assert not scan.prompt_available
    verdict = evaluate_check_85(task, [], scan, entity_half_ran=False)
    assert verdict.band == "not_evaluated"
    assert verdict.score is None
    assert "no prompt text" in verdict.measurement.notes


def test_no_prompt_text_means_no_model_call_is_worth_making():
    task = _task("")
    task.prompts = []
    task.seeded_prompt = ""
    assert not can_judge_environment_context(task)


def test_nothing_to_check_against_means_no_model_call_either():
    """With only the prompt in hand every reference it makes is unfalsifiable, so
    the question is not asked rather than answered from nothing."""
    task = _task("Send the signed copy to Priya Nair.")
    task.target_outcome = []
    task.target_deliverables = []
    assert universe_materials(task) == []
    assert not can_judge_environment_context(task)


def test_the_seeded_prompt_is_used_when_per_turn_prompts_were_not_ingested():
    """On the Snowflake export `Task.prompts` is empty for every task and the
    seeded prompt is the only text there is. Ingesting the turns belongs to
    whoever owns prompt ingestion; this check must work without it."""
    task = _task("Reconcile q3_actuals.csv.", ["q3_actuals.csv"])
    task.prompts = []
    assert prompt_texts(task) == ["Reconcile q3_actuals.csv."]
    scan = scan_file_references(task)
    assert scan.prompt_source == "seeded prompt only"
    assert scan.ran and scan.findings == []


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------


def test_the_prompt_names_every_material_the_model_may_cite():
    task = _task("Reconcile market_sizing_assumptions.xlsx.", ["budget.xlsx"])
    prompt = build_environment_context_prompt(task)
    assert "budget.xlsx" in prompt
    assert "target outcome list" in prompt
    # The severity bar and the evidence requirement both have to reach the model,
    # not just the gate.
    assert "real_world_public_entity" in prompt
    assert "checked_against" in prompt


# ---------------------------------------------------------------------------
# The manifest is declared, not delivered: a named file the conversation never
# actually shows can still be a violation, and the model half needs the
# conversation in front of it to catch that.
# ---------------------------------------------------------------------------


def test_the_prompt_shows_the_conversation_not_just_the_manifest():
    """A declared file attached at prompt-collection time is not proof it ever
    reached the live conversation. The judge needs the transcript to tell the
    two apart, which means it has to be in the prompt at all."""
    task = _task("Go through AnalystMemo_v3.docx and tell me what it says.")
    task.model_a.attachments = ["AnalystMemo_v3.docx"]
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Go through AnalystMemo_v3.docx and tell me what it says."),
        Turn(index=1, role="assistant", text="Here is what the memo says..."),
    ]
    prompt = build_environment_context_prompt(task)
    assert "Conversation with Model A" in prompt
    assert "Here is what the memo says" in prompt
    assert "DECLARES were supplied" in prompt
    assert "does NOT say the file" in prompt


def test_a_declared_file_never_evidenced_in_conversation_can_still_fail():
    """This is the shape of the real miss: `AnalystSummaryMemo_v3.docx` is on the
    declared manifest, so the deterministic half is clean, but the conversation
    never shows the model was actually given it -- the model just answers from
    what the user narrated. The model half has to be free to flag that."""
    task = _task(
        "I found a problem in AnalystSummaryMemo_v3.docx from last week. "
        "Can you go through the record and tell me what actually happened?"
    )
    task.model_a.attachments = ["AnalystSummaryMemo_v3.docx"]
    task.model_a.conversation = [
        Turn(
            index=1,
            role="user",
            text=(
                "I found a problem in AnalystSummaryMemo_v3.docx from last week. "
                "Can you go through the record and tell me what actually happened?"
            ),
        ),
        Turn(index=1, role="assistant", text="Here is what happened, based on the record..."),
    ]
    scan = scan_file_references(task)
    # The deterministic half is satisfied: the name is on the declared manifest.
    assert scan.findings == []

    finding = EnvironmentContextFinding(
        reference="AnalystSummaryMemo_v3.docx",
        kind="file",
        quote="go through the record and tell me what actually happened",
        checked_against=[
            "input file manifest (analystsummarymemo_v3.docx)",
            "Model A conversation (1 exchanges)",
        ],
        why_outside_universe=(
            "The name is on the declared manifest, but Model A's conversation "
            "never shows the file was attached or its content shared -- the "
            "model answers only from what the user narrated."
        ),
        presented_as_existing=True,
        required_to_complete=True,
        real_world_public_entity=False,
        basis="model",
    )
    verdict = evaluate_check_85(task, [finding], scan, entity_half_ran=True)
    assert verdict.band == "non_fail"
    assert verdict.measurement.counts["file_violations"] == 1


def test_content_the_user_typed_directly_is_not_a_missing_file_violation():
    """The rule targets files presented as independently reviewable, not a user
    pasting the content themselves. Nothing here requires the model half to
    report this shape at all; it just must not be forced to."""
    task = _task("Here are last week's numbers: revenue $4.2M, costs $3.1M. Summarise them.")
    task.model_a.conversation = [
        Turn(
            index=1,
            role="user",
            text="Here are last week's numbers: revenue $4.2M, costs $3.1M. Summarise them.",
        ),
        Turn(index=1, role="assistant", text="Revenue was $4.2M against $3.1M in costs."),
    ]
    scan = scan_file_references(task)
    verdict = evaluate_check_85(task, [], scan, entity_half_ran=True)
    assert verdict.band == "clean"


def test_one_call_is_built_for_85_and_counted_in_the_dry_run_estimate():
    task = make_task("est")
    requests = build_informed_requests(task)
    assert sum(1 for r in requests if r.metadata.get("check_id") == 85) == 1
    assert estimate_informed_calls(task) == len(requests)


def test_no_call_is_built_when_there_is_nothing_to_judge():
    task = _task("")
    task.prompts = []
    task.seeded_prompt = ""
    requests = build_informed_requests(task)
    assert not any(r.metadata.get("check_id") == 85 for r in requests)
    assert estimate_informed_calls(task) == len(requests)


def test_the_stage_runs_both_halves_and_scores_them_as_one_dimension():
    task = _task("Reconcile market_sizing_assumptions.xlsx with Priya Nair's figures.")
    task.model_a.attachments = ["budget.xlsx"]

    def respond(request):
        if request.metadata.get("check_id") != 85:
            return ModelResponse(key=request.key, error="not under test")
        return {
            "references": [
                {
                    "reference": "Priya Nair",
                    "kind": "person",
                    "quote": "Priya Nair's figures",
                    "checked_against": ["target outcome list (2 entries)"],
                    "why_outside_universe": "No material names her.",
                    "presented_as_existing": True,
                    "required_to_complete": True,
                    "real_world_public_entity": False,
                }
            ],
            "confidence": "high",
        }

    result = run_informed_stage(task, FakeModelClient(respond), workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 85)

    assert result.entity_half_ran
    assert result.file_scan is not None and result.file_scan.ran
    kinds = sorted(f.kind for f in result.environment_context)
    assert kinds == ["file", "person"]
    assert verdict.band == "non_fail"
    assert verdict.measurement.counts["file_violations"] == 1
    assert verdict.measurement.counts["entity_violations"] == 1


def test_the_file_half_still_runs_when_the_model_call_fails():
    task = _task("Reconcile market_sizing_assumptions.xlsx.", ["budget.xlsx"])
    client = FakeModelClient(lambda request: None)
    result = run_informed_stage(task, client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 85)

    assert not result.entity_half_ran
    assert verdict.band == "non_fail"
    assert verdict.measurement.counts["file_half_ran"] is True


def test_a_clean_scan_with_no_model_answer_is_clean_not_not_evaluated():
    task = _task("Reconcile budget.xlsx.", ["budget.xlsx"])
    client = FakeModelClient(lambda request: None)
    result = run_informed_stage(task, client, workers=1)
    verdict = next(v for v in result.verdicts if v.check_id == 85)
    assert verdict.band == "clean"


def test_the_manifest_reads_both_slots():
    task = _task("x", ["a.pdf"], ["b.pdf"])
    manifest = input_manifest(task)
    assert manifest.basenames() == {"a.pdf", "b.pdf"}
    assert manifest.source == "ModelSubmission.attachments"


# ---------------------------------------------------------------------------
# Supplied files and produced files are not the same population. Conflating
# them accounted for 11 of the 13 escalations measured across three corpora.
# ---------------------------------------------------------------------------


def _artifact(name: str) -> InputArtifact:
    return InputArtifact(name=name, origin="upload")


def test_the_manifest_keeps_produced_files_apart_from_supplied_ones():
    """`named` stays the union and `produced` names the half that is output.

    Both are needed and for opposite reasons. The deterministic half only ever uses
    the list to CLEAR a reference, so it wants everything anyone has seen -- reading
    an output file as if it had never existed would fail a callback. The model half
    is told the same list was "supplied", and it reads that as the population to
    audit, so it needs the half that actually was.
    """
    task = _task("Summarise budget.xlsx.")
    task.input_artifacts = [_artifact("budget.xlsx")]
    task.model_a.attachments = ["turn 14.pdf"]
    task.model_b.attachments = ["gemini-code-1786727907414.txt"]

    manifest = input_manifest(task)
    assert manifest.basenames() == {
        "budget.xlsx",
        "turn 14.pdf",
        "gemini-code-1786727907414.txt",
    }
    assert manifest.supplied_basenames() == {"budget.xlsx"}
    assert manifest.produced_basenames() == {
        "turn 14.pdf",
        "gemini-code-1786727907414.txt",
    }
    # And the clearing set is still the union, so a callback to an output stays
    # clean.
    assert "turn 14.pdf" in manifest.known_basenames()


def test_the_judge_is_not_told_the_models_own_output_was_supplied_to_them():
    """The prompt-level half of the same fix, and the one that was doing the damage.

    On 134 of the 191 tasks carrying an 85 verdict, every name under "files the task
    DECLARES were supplied" was a model attachment. A judge told a file was supplied
    and then asked to find evidence it was uploaded finds none, because it never
    was: it came out of the conversation rather than into it.
    """
    task = _task("Review the attached deck and update it.")
    task.input_artifacts = [_artifact("deck.pptx")]
    task.model_a.attachments = ["turn 14.pdf"]

    prompt = build_environment_context_prompt(task)
    declared = prompt.split("DECLARES were supplied", 1)[1].split("Files the models", 1)[0]
    assert "deck.pptx" in declared
    assert "turn 14.pdf" not in declared, "an output must not be described as an input"

    produced = prompt.split("Files the models PRODUCED", 1)[1]
    assert "turn 14.pdf" in produced
    assert "They are listed so you can recognise them, not" in produced
    assert "never belongs in `declared_file_checks`" in produced


def test_a_task_with_no_supplied_files_says_so_rather_than_listing_outputs():
    task = _task("Build me a chart.")
    task.model_a.attachments = ["chart.png"]
    prompt = build_environment_context_prompt(task)
    declared = prompt.split("DECLARES were supplied", 1)[1].split("Files the models", 1)[0]
    assert "(none with a readable name)" in declared
    assert "chart.png" in prompt.split("Files the models PRODUCED", 1)[1]


def test_the_file_path_asks_the_two_violation_questions_instead_of_assuming_them():
    """The parser hardcoded `presented_as_existing` and `required_to_complete` to
    True on this path, so a file became a violation the moment the two upload
    questions came back false -- the expected answer for anything the model itself
    produced. The same judge, in one response, answered `required_to_complete=False`
    for a reference on the `references` path and had it forced to True here.
    """
    asked = _response(
        {
            "declared_file_checks": [
                {
                    "filename": "office_template.docx",
                    "prompt_quote": "use the formatting we use in office",
                    "prompt_treats_as_available_to_read": True,
                    "conversation_shows_attachment_or_upload": False,
                    "conversation_shows_independent_content": False,
                    "presented_as_existing": True,
                    "required_to_complete": False,
                    "evidence_summary": "The model can write around the template.",
                }
            ],
            "references": [],
            "confidence": "high",
        }
    )
    findings = parse_environment_context(asked)
    assert len(findings) == 1
    assert findings[0].required_to_complete is False
    assert not findings[0].is_violation, "an unnecessary file is not a violation"

    # A judge that answers both questions true still gets its violation, so the
    # change is a question asked rather than a leg removed.
    data = dict(asked.data)
    data["declared_file_checks"][0]["required_to_complete"] = True
    findings = parse_environment_context(_response(data))
    assert findings[0].is_violation

    # And an older cached response with neither field keeps the old reading, so a
    # replay of a run scored before this is unchanged.
    legacy = dict(asked.data)
    del legacy["declared_file_checks"][0]["presented_as_existing"]
    del legacy["declared_file_checks"][0]["required_to_complete"]
    findings = parse_environment_context(_response(legacy))
    assert findings[0].is_violation


def test_a_reference_the_judge_looked_for_and_found_is_not_an_allegation():
    """The judge is asked to show its work, so `references` carries rows for things
    it searched for and located. Answered literally about an in-universe reference,
    the three booleans spell violation -- it exists, the work needs it, it is not a
    public entity -- and the only place the judge could say "I found it" was
    free-text prose nothing reads. One live row said "Nothing about Straxon requires
    consulting an external, undisclosed source" and failed the task.
    """
    found = _response(
        {
            "declared_file_checks": [],
            "references": [
                {
                    "reference": "Straxon",
                    "kind": "organisation",
                    "quote": "our Straxon deployment",
                    "checked_against": ["the model conversation(s)"],
                    "found_in_materials": True,
                    "why_outside_universe": (
                        "Nothing about Straxon requires consulting an external, "
                        "undisclosed source."
                    ),
                    "presented_as_existing": True,
                    "required_to_complete": True,
                    "real_world_public_entity": False,
                }
            ],
            "confidence": "high",
        }
    )
    assert parse_environment_context(found) == []

    # The same row with the field false is still an allegation, so the fix does not
    # disarm the check.
    data = dict(found.data)
    data["references"][0]["found_in_materials"] = False
    findings = parse_environment_context(_response(data))
    assert len(findings) == 1 and findings[0].is_violation

    # And a cached response from before the field existed reads as before.
    legacy = dict(found.data)
    del legacy["references"][0]["found_in_materials"]
    assert len(parse_environment_context(_response(legacy))) == 1


def test_the_prompt_tells_the_judge_that_found_is_the_common_answer():
    """A field the judge does not know to set true is a field that does nothing.
    The instruction has to say so explicitly, because the judge's default posture
    is to write rows only for things it is alleging."""
    prompt = build_environment_context_prompt(_task("Check our Straxon deployment."))
    assert "`found_in_materials`" in prompt
    assert "Answer TRUE whenever you did" in prompt
    assert "is not a violation no matter how the other three answers come out" in prompt
    assert "answer four questions about it" in prompt


# ---------------------------------------------------------------------------
# A filename the conversation itself already established -- either as the
# model's own earlier deliverable, or as a name the model only ever typed --
# is a callback, not a claim about something outside the task's universe.
# ---------------------------------------------------------------------------


def test_assistant_mentioned_filenames_reads_only_assistant_turns_of_both_slots():
    task = _task("x", ["a.pdf"])
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Draft it, then send me answer.docx."),
        Turn(index=1, role="assistant", text="Here's draft.docx to start."),
    ]
    task.model_b.conversation = [
        Turn(index=1, role="assistant", text="See summary.pdf for the recap."),
    ]
    mentioned = assistant_mentioned_filenames(task)
    # Only the assistant's own turns are read -- the user turn's "answer.docx"
    # must not leak in even though it sits in the same conversation.
    assert sorted(m.lower() for m in mentioned) == ["draft.docx", "summary.pdf"]


def test_a_file_the_model_produced_earlier_and_the_user_later_asks_to_revise_is_not_a_violation():
    """This is the systemic false-fail from the manual audit: the model says
    'Here's your report.pdf' in one turn, and a later turn asks to revise
    report.pdf. That is a callback to the conversation's own earlier output,
    not a claim about a file the task never supplied."""
    task = _task("Draft the quarterly report.", ["budget.xlsx"])
    task.prompts = []  # force the fallback to conversation user turns
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Draft the quarterly report."),
        Turn(index=1, role="assistant", text="Here's your report.pdf."),
        Turn(index=2, role="user", text="Please revise report.pdf to add the Q3 numbers."),
        Turn(index=2, role="assistant", text="Updated report.pdf accordingly."),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert "report.pdf" in [r.lower() for r in scan.named_references]
    assert scan.findings == []


def test_a_filename_the_model_hallucinated_and_the_user_quotes_back_is_not_a_violation():
    """The model claims a file it never actually produced; the user later quotes
    that fabricated name back while complaining about it. The name is still on
    the table because the conversation itself already said it -- the defect is
    the model's fabrication, a different check's concern, not a prompt
    presupposing something external."""
    task = _task("Generate a chart of last month's sales.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Generate a chart of last month's sales."),
        Turn(index=1, role="assistant", text="Done -- see image_6.png for the chart."),
        Turn(
            index=2,
            role="user",
            text="You also mentioned image_6.png but never gave me a file called that.",
        ),
        Turn(index=2, role="assistant", text="Apologies, I did not actually produce image_6.png."),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert scan.findings == []


def test_a_genuinely_external_file_never_mentioned_by_either_side_still_fails():
    """Guards against over-broadening: a file nobody -- neither the manifest nor
    any assistant turn -- ever put on the table is still a violation."""
    task = _task("Reconcile the numbers in external_ledger.xlsx.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Reconcile the numbers in external_ledger.xlsx."),
        Turn(index=1, role="assistant", text="I don't see that file -- can you attach it?"),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference for f in scan.findings] == ["external_ledger.xlsx"]


@pytest.mark.parametrize(
    "refusal",
    [
        "I don't have access to external_ledger.xlsx -- can you attach it?",
        "Please upload external_ledger.xlsx so I can proceed.",
        "I could not find external_ledger.xlsx in the provided files.",
        "I am unable to access external_ledger.xlsx.",
        "external_ledger.xlsx was not provided in this task.",
        "There is no file named external_ledger.xlsx here.",
        "external_ledger.xlsx is missing from the uploads.",
        "I never received external_ledger.xlsx.",
    ],
)
def test_a_model_naming_the_missing_file_while_asking_for_it_does_not_excuse_the_prompt(refusal):
    """The exemption must not be satisfied by the model's own confirmation of
    the violation. Naming the file while reporting its absence is the most
    natural way to answer a prompt that presupposes a file the task never
    supplied, so treating that mention as "established" would silently retire
    the check for its single most common case."""
    task = _task("Reconcile the numbers in external_ledger.xlsx.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Reconcile the numbers in external_ledger.xlsx."),
        Turn(index=1, role="assistant", text=refusal),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference for f in scan.findings] == ["external_ledger.xlsx"]


def test_one_reply_can_establish_one_file_while_disclaiming_another():
    """Disclaiming is judged per sentence, not per turn: a reply that hands over
    one file and asks for another exempts only the file it handed over."""
    task = _task("Summarise the quarter.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Summarise the quarter using ledger.xlsx."),
        Turn(
            index=1,
            role="assistant",
            text=(
                "Here is summary.docx built from the budget. "
                "I don't have ledger.xlsx, so those rows are omitted."
            ),
        ),
        Turn(index=2, role="user", text="Tighten summary.docx and re-check ledger.xlsx."),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference.lower() for f in scan.findings] == ["ledger.xlsx"]


@pytest.mark.parametrize(
    "reply",
    [
        "Here is report.pdf -- note the appendix is missing.",
        "I created report.pdf, though I could not find the tax records.",
        "Here is report.pdf. Please upload ledger.xlsx for the rest.",
        "report.pdf is attached to this reply.",
    ],
)
def test_delivering_one_file_while_noting_another_is_absent_still_establishes_it(reply):
    """The guard must not swing back into the false fails it was added beside.
    A reply that hands over report.pdf and in the same sentence mentions
    something else missing has still handed over report.pdf, so the nearest cue
    to the name -- not the mere presence of absence-language -- decides."""
    task = _task("Update report.pdf with the totals.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Update report.pdf with the totals."),
        Turn(index=1, role="assistant", text=reply),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference.lower() for f in scan.findings if f.reference.lower() == "report.pdf"] == []


def test_a_disclaimer_beats_a_delivery_that_sits_further_from_the_name():
    """"I don't have report.pdf, but here is a draft" delivers something, just
    not report.pdf. The disclaimer is the closer cue, so the name stays
    unestablished."""
    task = _task("Update report.pdf with the totals.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Update report.pdf with the totals."),
        Turn(index=1, role="assistant", text="I don't have report.pdf, but here is a draft."),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference.lower() for f in scan.findings] == ["report.pdf"]


def test_a_file_disclaimed_once_but_delivered_later_is_still_established():
    """The union is taken across the whole trajectory, so an early "I couldn't
    find it" does not permanently poison a name the model later produces."""
    task = _task("Build the reconciliation.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Build the reconciliation."),
        Turn(index=1, role="assistant", text="I couldn't find ledger.xlsx to start from."),
        Turn(index=2, role="user", text="Construct it from the budget instead."),
        Turn(index=2, role="assistant", text="Done -- here is ledger.xlsx with the rows."),
        Turn(index=3, role="user", text="Now add totals to ledger.xlsx."),
    ]
    scan = scan_file_references(task)
    assert scan.ran
    assert scan.findings == []


def test_conversation_mentioned_filenames_are_visible_but_kept_out_of_named_manifest_reporting():
    """The new source of "known" filenames is reported distinctly, not folded
    into `named` as though it had been uploaded."""
    task = _task("Draft the quarterly report.", ["budget.xlsx"])
    task.prompts = []
    task.model_a.conversation = [
        Turn(index=1, role="user", text="Draft the quarterly report."),
        Turn(index=1, role="assistant", text="Here's your report.pdf."),
        Turn(index=2, role="user", text="Please revise report.pdf."),
    ]
    manifest = input_manifest(task)
    assert manifest.basenames() == {"budget.xlsx"}
    assert manifest.conversation_mentioned
    assert {m.lower() for m in manifest.conversation_mentioned} == {"report.pdf"}
    assert manifest.known_basenames() == {"budget.xlsx", "report.pdf"}

    scan = scan_file_references(task)
    payload = scan.to_dict()
    assert payload["manifest_named"] == 1
    assert payload["manifest_conversation_mentioned"] == 1


def test_the_verdict_survives_a_round_trip_to_the_report():
    task = _task("Reconcile q3.csv.", ["budget.xlsx"])
    scan = scan_file_references(task)
    payload = evaluate_check_85(task, scan.findings, scan).to_dict()
    assert payload["check_id"] == 85
    assert payload["error_code"] == REPORTED_LABEL
    assert dataclasses.asdict(scan.findings[0])["basis"] == "deterministic"
