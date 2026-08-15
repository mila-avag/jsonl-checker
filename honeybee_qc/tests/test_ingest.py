"""Snowflake ingestion.

Every fixture here is shaped after something the live project actually does. The
documented step map is a good starting point and a bad contract: on a 25-task
sample the step documented as "B scores" carried both products, the step
documented as "A scores" was absent, the B trajectory field appeared under five
different names, and the product-name prefillers were missing about half the
time. Reading by position yields a silently half-empty audit, so the adapter
reads by shape and these tests hold it to that.
"""

from __future__ import annotations

import json

from honeybee_qc.env_context import input_manifest, scan_file_references
from honeybee_qc.gates import evaluate_check_85
from honeybee_qc.ingest import (
    PREFILLER_ASSIGNED_DOMAIN,
    PREFILLER_PRESEEDED_PROMPT,
    PROJECT_ID,
    STEP_PREFERENCE,
    STEP_PROMPT,
    STEP_RUBRIC,
    STEP_RUN_A,
    STEP_RUN_B,
    ingest_attempt,
    ingest_rows,
)


def _step(output: dict) -> dict:
    return {"output": output}


def _row(before: dict, sources: dict | None = None, task: str = "t-1") -> dict:
    return {
        "TASK": task,
        "RESPONSE": json.dumps({
            "before": before,
            "turns": [],
            "after": {},
            "metrics": {},
            "dataSourceResults": sources or {},
        }),
    }


def _criterion(cid: str, weight: int, category: str | None = None) -> dict:
    return {
        "id": cid,
        "title": f"criterion {cid}",
        "weight": weight,
        "annotations": {"criterion_category": category} if category else {},
    }


# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------


def test_both_products_are_found_when_one_step_carries_them_both():
    """The observed layout that broke reading by step ID: the "B scores" step
    holds response-a and response-b, and the "A scores" step is absent."""
    before = {
        STEP_RUBRIC: _step({"criteria": [_criterion("1", 3), _criterion("2", 5)]}),
        "step-RubricCriteriaRating-49853f6eba5a": _step({
            "responseRatings": {
                "response-a": {"1": {"score": 1}, "2": {"score": 0}},
                "response-b": {"1": {"score": 0}, "2": {"score": 1}},
            }
        }),
    }
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert sorted(r.model for r in task.criterion_ratings) == ["A", "A", "B", "B"]
    assert {(r.model, r.criterion_id): r.score for r in task.criterion_ratings} == {
        ("A", "1"): 1, ("A", "2"): 0, ("B", "1"): 0, ("B", "2"): 1,
    }


def test_turn_attribution_is_never_mistaken_for_a_score():
    """The attribution entries carry `title` and no `score`. Counting them as
    scores would invent ratings; ignoring them would drop the cited turns."""
    before = {
        STEP_RUBRIC: _step({"criteria": [_criterion("1", 3)]}),
        "step-RubricCriteriaRating-5665ecba649a": _step({
            "responseRatings": {"response-a": {"1": {"score": 0}}}
        }),
        "step-RubricCriteriaRating-e2c61a59aef7": _step({
            "responseRatings": {"response-a": {"1": {"id": "1", "title": "Turn 11"}}}
        }),
    }
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert len(task.criterion_ratings) == 1
    rating = task.criterion_ratings[0]
    assert rating.score == 0
    assert rating.relevant_turns == [11]


def test_a_task_with_no_ratings_yields_none_rather_than_zeros():
    before = {STEP_RUBRIC: _step({"criteria": [_criterion("1", 3)]})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.criterion_ratings == []


# ---------------------------------------------------------------------------
# Trajectory links and product names
# ---------------------------------------------------------------------------


def test_the_b_trajectory_is_found_under_any_of_its_observed_names():
    for field in (
        "claude_trajectory_b",
        "claude_trajectory",
        "gemini_final_trajectory_b",
        "gpt_trajectory",
        "gpt_trajectory_b",
    ):
        before = {STEP_RUN_B: _step({field: "https://example.test/b"})}
        task = ingest_attempt(_row(before)).task
        assert task is not None
        assert task.model_b.final_link == "https://example.test/b", field


def test_product_a_is_not_assumed_to_be_gemini():
    """The skill states product A is always Gemini. Four of 25 sampled tasks
    record `claude_trajectory_a`."""
    before = {STEP_RUN_A: _step({"claude_trajectory_a": "https://claude.ai/share/x"})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.model_a.final_link == "https://claude.ai/share/x"
    assert task.model_a.declared_provider == "claude"


def test_a_final_link_field_holding_a_turn_list_yields_the_last_url():
    """One sampled task files five newline-separated URLs in a single trajectory
    field, the last of which is that task's recorded final trajectory."""
    before = {STEP_RUN_A: _step({
        "claude_trajectory_a": (
            "https://share.gemini.google/one\n"
            "https://share.gemini.google/two\n"
            "https://share.gemini.google/final"
        )
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.model_a.final_link == "https://share.gemini.google/final"


def test_a_field_naming_itself_final_wins_over_a_bare_one():
    before = {STEP_RUN_A: _step({
        "claude_trajectory_a": "https://share.gemini.google/turnlist",
        "gemini_final_trajectory_a": "https://share.gemini.google/thefinal",
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.model_a.final_link == "https://share.gemini.google/thefinal"


def test_the_link_domain_beats_a_field_name_that_disagrees_with_it():
    """A sampled task files a `share.gemini.google` URL under `claude_trajectory_a`.
    Believing the name would declare the slot Claude and make check 90 fail the
    task for a provider mismatch we invented."""
    before = {STEP_RUN_A: _step({
        "claude_trajectory_a": "https://share.gemini.google/mRGTcVyQExjY"
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.model_a.declared_provider == "gemini"


def test_the_prefiller_wins_over_the_inferred_product_name():
    before = {STEP_RUN_A: _step({"gemini_trajectory": "https://share.gemini.google/x"})}
    sources = {
        "before:0:data-source-string-prefiller-646efa3f869f": {
            "content": "Gemini 3.6 Flash Extended Thinking (custom)"
        }
    }
    task = ingest_attempt(_row(before, sources)).task
    assert task is not None
    assert task.model_a.declared_provider == "Gemini 3.6 Flash Extended Thinking (custom)"


def test_per_turn_links_parse_under_either_observed_field_name():
    for field in ("every_gemini_custom_turn_a", "every_gemini_turn_a"):
        before = {STEP_RUN_A: _step({
            field: "Turn 1: https://example.test/1\nTurn 2: https://example.test/2"
        })}
        task = ingest_attempt(_row(before)).task
        assert task is not None
        assert [(t.turn, t.url) for t in task.model_a.turn_links] == [
            (1, "https://example.test/1"),
            (2, "https://example.test/2"),
        ], field


def test_attachments_come_through_as_plain_public_urls():
    before = {STEP_RUN_A: _step({
        "final_outputs_a": [{"s3Url": "https://s3.test/a.mp4"}, {"s3Url": "https://s3.test/b.png"}],
        "model_a_chat_download": [{"s3Url": "https://s3.test/chat.pdf"}],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.model_a.attachments == ["https://s3.test/a.mp4", "https://s3.test/b.png"]
    assert task.model_a.transcript_pdf == "https://s3.test/chat.pdf"


# ---------------------------------------------------------------------------
# The prompt assignment, for check 70
# ---------------------------------------------------------------------------


def test_the_assigned_domain_and_category_come_through():
    """Check 70 has nothing to measure against without these two. The persona
    lives in a string prefiller and the use-case category in the prompt step, so
    they arrive from two different places in the payload."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "What are the latest MikroTik PoE switches?",
        "prompt_CUJ": "shopping",
    })}
    sources = {
        PREFILLER_ASSIGNED_DOMAIN: {"content": "Backend software engineer"},
    }
    task = ingest_attempt(_row(before, sources)).task
    assert task is not None
    assert task.assigned_domain == "Backend software engineer"
    assert task.prompt_category == "shopping"


def test_a_missing_assignment_is_empty_rather_than_a_placeholder():
    """About a third of the corpus records no category. An empty string is what
    check 70's gate reads to decide not_evaluated, so a placeholder here would be
    graded as if it were a real assignment."""
    task = ingest_attempt(_row({STEP_PROMPT: _step({"prompt_CUJ": None})})).task
    assert task is not None
    assert task.assigned_domain == ""
    assert task.prompt_category == ""


# ---------------------------------------------------------------------------
# Input artifacts, for check 85's file half
# ---------------------------------------------------------------------------


def _upload(name: str, **kw) -> dict:
    base = {
        "name": name,
        "s3Url": f"https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/{name[:6]}",
        "url": "scale-cds://6463e58/xyz#s3/scale-cds-public-us-west-2",
        "mimeType": "application/pdf",
        "fileType": "FILE_TYPE_DOCUMENT",
        "fileSizeInBytes": 40054,
        "imageSubmissionMethod": "File Upload",
    }
    base.update(kw)
    return base


def _declared(path: str, ordinal: str = "1") -> dict:
    return {
        "fullText": f"{ordinal} {path}",
        "parts": [
            {"id": "Path to Artifacts", "value": path},
            {"id": "Number Input", "value": ordinal},
        ],
    }


def test_uploaded_input_files_arrive_with_their_names_and_types():
    """The names check 85 compares against. They are recoverable only from this
    field: the CDN URL beside each one ends in a content ID, and it is the URL
    alone that `ModelSubmission.attachments` carries."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "Summarise Ridgewater_PortfolioPlaybook_v2.pdf.",
        "input_artifacts_a": [_upload("Ridgewater_PortfolioPlaybook_v2.pdf")],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.input_artifacts is not None
    artifact = task.input_artifacts[0]
    assert artifact.name == "Ridgewater_PortfolioPlaybook_v2.pdf"
    assert artifact.origin == "upload"
    assert artifact.mime_type == "application/pdf"
    assert artifact.file_type == "FILE_TYPE_DOCUMENT"
    assert artifact.size_bytes == 40054
    assert artifact.url.startswith("https://scale-cds-public-us-west-2")


def test_input_files_are_per_task_rather_than_per_model():
    """The `_a` in `input_artifacts_a` is part of the field name, not a slot. The
    field sits on the prompt step beside `final_hardened_prompt`, there is no
    `input_artifacts_b` anywhere in the export, and a pairwise comparison in which
    the two models were given different files would not be a comparison. So one
    manifest serves both slots, and a `_b` payload must not create a second."""
    before = {
        STEP_PROMPT: _step({"input_artifacts_a": [_upload("shared_input.pdf")]}),
        STEP_RUN_A: _step({"input_artifacts_a": [_upload("stray_a.pdf")]}),
        STEP_RUN_B: _step({"input_artifacts_b": [_upload("stray_b.pdf")]}),
    }
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert [a.name for a in task.input_artifacts] == ["shared_input.pdf"]


def test_a_declared_path_is_carried_verbatim_without_being_parsed():
    """Files the contributor located instead of uploading. The adapter does not
    decide whether one of these names a file -- half are plain web URLs -- so the
    string is transcribed and the reading is left to the manifest."""
    before = {STEP_PROMPT: _step({"input_artifacts_a_links": [
        _declared("Matters/RidgewaterRefactor/Playbook/Ridgewater_PortfolioPlaybook_v2.pdf"),
        _declared("https://drive.google.com/file/d/1Abs-3tzUSxmtG8JjcgrJ3AOqffp-mPfd/view", "2"),
    ]})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert [a.origin for a in task.input_artifacts] == ["declared_path", "declared_path"]
    assert task.input_artifacts[0].path.endswith("Ridgewater_PortfolioPlaybook_v2.pdf")
    assert task.input_artifacts[0].name == ""
    assert task.input_artifacts[1].path.startswith("https://drive.google.com")


def test_a_prompt_step_with_no_artifact_fields_is_an_empty_manifest_not_an_absent_one():
    """The contributor supplied nothing, which is a fact about the task. It has to
    be distinguishable from having read no manifest at all, because only one of
    the two licenses any conclusion about a file the prompt names."""
    before = {STEP_PROMPT: _step({"final_hardened_prompt": "no files here"})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.input_artifacts == []


def test_an_absent_prompt_step_leaves_the_manifest_unread_rather_than_empty():
    before = {STEP_RUBRIC: _step({"criteria": [_criterion("1", 3)]})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.input_artifacts is None


def test_the_manifest_reads_the_ingested_input_files():
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "Reconcile market_sizing_assumptions.xlsx.",
        "input_artifacts_a": [_upload("market_sizing_assumptions.xlsx")],
        "input_artifacts_a_links": [_declared("market_sizing_assumptions.xlsx")],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    manifest = input_manifest(task)
    assert manifest.usable
    assert manifest.basenames() == {"market_sizing_assumptions.xlsx"}
    assert "Task.input_artifacts" in manifest.source

    scan = scan_file_references(task)
    assert scan.ran
    assert scan.findings == []


def test_a_declared_path_is_matched_on_its_basename_however_it_was_typed():
    """Observed forms: a quoted Windows path, a repository-relative path, and a
    bare filename. A quote left on the end makes the extension `pdf"`, which is no
    extension, and the file the task really has goes missing from the manifest."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": (
            "Compare PublishedPaper-Review.pdf against newsletter_idea_backlog.txt."
        ),
        "input_artifacts_a_links": [
            _declared('"C:\\Users\\Alberto\\Downloads\\PublishedPaper-Review.pdf"'),
            _declared("05_knowledge_base/research_notes/newsletter_idea_backlog.txt", "2"),
        ],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    manifest = input_manifest(task)
    assert manifest.basenames() == {
        "publishedpaper-review.pdf",
        "newsletter_idea_backlog.txt",
    }
    assert scan_file_references(task).findings == []


def test_a_link_carrying_a_query_string_is_not_read_as_a_file():
    """A Drive URL ending `/edit?usp=drive_link&sd=true` ends in `true`, which
    passes for an extension. Admitting it would put a fictional filename in the
    manifest, and one name is all it takes to make the manifest `usable` -- so the
    half would run and report every real file the prompt names as absent."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "Summarise the attached deck, exec_readout.pptx.",
        "input_artifacts_a_links": [_declared(
            "https://docs.google.com/presentation/d/12Fv_k-lLfWuVVdLj7RstyiDL74do-CU8"
            "/edit?usp=drive_link&ouid=111819959194727097378&rtpof=true&sd=true"
        )],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    manifest = input_manifest(task)
    assert manifest.named == []
    assert manifest.opaque and not manifest.usable

    scan = scan_file_references(task)
    assert not scan.ran
    assert scan.findings == []


def test_an_opaque_cdn_upload_still_abstains_after_ingestion():
    """The case that made the half inert, and it must stay abstaining: an upload
    whose name never reached the export leaves the URL, whose last segment is a
    content ID. It proves a file exists and says nothing about which."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "Use the tables in q3_actuals.csv.",
        "input_artifacts_a": [_upload(
            "", s3Url="https://scale-cds-public-us-west-2.s3.amazonaws.com/6463e58/ZQjy98e0JqvrEIg"
        )],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.input_artifacts is not None and len(task.input_artifacts) == 1

    manifest = input_manifest(task)
    assert manifest.opaque and not manifest.named
    assert manifest.total == 1

    scan = scan_file_references(task)
    assert not scan.ran
    assert scan.findings == []
    assert evaluate_check_85(task, scan.findings, scan).band == "not_evaluated"


def test_a_file_the_prompt_names_and_the_task_does_not_have_fails_once_ingested():
    """The half's whole point, and it could not fire before: with a manifest that
    names something, a reference to a file nobody supplied is now a violation
    rather than an unverified claim."""
    before = {STEP_PROMPT: _step({
        "final_hardened_prompt": "Reconcile q3_actuals.csv against the plan.",
        "input_artifacts_a": [_upload("market_sizing_assumptions.xlsx")],
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    scan = scan_file_references(task)
    assert scan.ran
    assert [f.reference for f in scan.findings] == ["q3_actuals.csv"]

    verdict = evaluate_check_85(task, scan.findings, scan)
    assert verdict.band == "fail"
    assert verdict.measurement.counts["file_half_ran"] is True
    assert "market_sizing_assumptions.xlsx" in scan.findings[0].checked_against[0]


def test_the_hardened_prompt_is_the_submitted_prompt_not_a_pre_seed():
    """The prompt step's own `final_hardened_prompt` output field holds what the
    contributor submitted, so `seeded_prompt` is the submitted text despite its
    name -- distinct from `pre_seeded_prompt` below, which is a different field
    at a different path entirely."""
    before = {STEP_PROMPT: _step({"final_hardened_prompt": "the contributor's own wording"})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.seeded_prompt == "the contributor's own wording"
    assert task.pre_seeded_prompt is None


def test_the_pre_seeded_prompt_is_read_from_its_own_form_prefiller():
    """For check 75. A `data-source-form-prefiller`, not the `data-source-string-
    prefiller` shape `product_name` reads elsewhere, and it carries its own
    `final_hardened_prompt` key -- the same field name the prompt step's output
    uses for the submission, at an unrelated path in `dataSourceResults`."""
    before = {STEP_PROMPT: _step({"final_hardened_prompt": "the contributor's own wording"})}
    sources = {
        PREFILLER_PRESEEDED_PROMPT: {
            "final_hardened_prompt": "the platform's pre-seeded template"
        },
    }
    task = ingest_attempt(_row(before, sources)).task
    assert task is not None
    assert task.seeded_prompt == "the contributor's own wording"
    assert task.pre_seeded_prompt == "the platform's pre-seeded template"


def test_a_pre_seeded_prompt_absent_project_wide_ingests_as_none_not_blank():
    """The field is missing on most tasks project-wide. `None` is what check 75's
    gate reads to abstain with not_evaluated, so an empty string here would be
    graded as though the platform had prepared a blank pre-seed."""
    before = {STEP_PROMPT: _step({"final_hardened_prompt": "no prefiller on this task"})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert task.pre_seeded_prompt is None


def test_a_pre_seeded_prompt_prefiller_missing_its_own_field_is_also_none():
    before = {STEP_PROMPT: _step({"final_hardened_prompt": "x"})}
    sources = {PREFILLER_PRESEEDED_PROMPT: {"content": "wrong shape for this prefiller"}}
    task = ingest_attempt(_row(before, sources)).task
    assert task is not None
    assert task.pre_seeded_prompt is None


# ---------------------------------------------------------------------------
# Rubric, dimensions, preference
# ---------------------------------------------------------------------------


def test_the_category_string_is_split_into_l1_and_l2():
    before = {STEP_RUBRIC: _step({"criteria": [
        _criterion("1", 3, "outcome quality-artifact delivery: a usable artifact was actually produced."),
    ]})}
    result = ingest_attempt(_row(before))
    assert result.task is not None
    criterion = result.task.rubric[0]
    assert (criterion.l1_label, criterion.l2_label) == ("Outcome Quality", "Artifact Delivery")
    assert not result.notes


def test_an_unplaceable_leaf_is_noted_and_keeps_its_l1():
    before = {STEP_RUBRIC: _step({"criteria": [
        _criterion("1", 3, "interaction efficiency-clarification timing"),
    ]})}
    result = ingest_attempt(_row(before))
    assert result.task is not None
    criterion = result.task.rubric[0]
    assert criterion.l1_label == "Interaction Efficiency"
    assert criterion.l2_label is None
    assert any("unrecognised L2" in n for n in result.notes)


def test_negative_weights_survive_ingestion():
    """They exist -- 25 criteria across 14 tasks, from -3 to -5 -- against a spec
    that says 1-5 with no negatives. Dropping them would hide the defect."""
    before = {STEP_RUBRIC: _step({"criteria": [_criterion("1", -5), _criterion("2", 3)]})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert [c.weight for c in task.rubric] == [-5, 3]


def test_memory_is_marked_not_applicable_rather_than_missing():
    before = {"step-ResponseTextCollection-cb06a191fca3": _step({"annotations": {
        "outcome_quality_a": 4,
        "outcome_quality_justif_a": "solid",
        "outcome_quality_turns_a": [{"parts": [{"id": "turn_number", "value": "3"}]}],
        "memory_applicable_a": "No",
    }})}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    by_dim = {d.dimension: d for d in task.dimension_ratings}
    assert by_dim["Outcome quality"].rating == 4
    assert by_dim["Outcome quality"].relevant_turns == [3]
    memory = by_dim["Memory & personalization"]
    assert memory.not_applicable and memory.rating is None


def test_collaboration_quality_is_carried_even_though_nothing_audits_it():
    before = {"step-ResponseTextCollection-cb06a191fca3": _step({
        "annotations": {"collaboration_quality_a": 5}
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert [d.rating for d in task.dimension_ratings if d.dimension == "Collaboration quality"] == [5]


def test_preference_captures_the_winner_alongside_the_likert():
    before = {STEP_PREFERENCE: _step({
        "responseIdx": 1,
        "preferenceLikert": 6,
        "fieldResponses": {"preference_justification": "B held the thread"},
    })}
    task = ingest_attempt(_row(before)).task
    assert task is not None
    assert (task.sxs.likert, task.sxs.winner_index) == (6, 1)
    assert task.sxs.justification == "B held the thread"


def test_a_defaulted_response_index_without_a_likert_is_not_a_preference():
    """responseIdx defaults to 0, which is also a valid answer, so it cannot be
    used to decide whether a preference was ever recorded."""
    before = {STEP_PREFERENCE: _step({"responseIdx": 0})}
    result = ingest_attempt(_row(before))
    assert result.task is not None
    assert result.task.sxs.likert is None
    assert result.task.sxs.winner_index is None
    assert any("unrecorded" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Row handling
# ---------------------------------------------------------------------------


def test_column_case_and_variant_payloads_are_both_accepted():
    before = {STEP_RUBRIC: _step({"criteria": [_criterion("1", 3)]})}
    as_text = ingest_attempt({"TASK": "t", "RESPONSE": json.dumps({"before": before})})
    as_dict = ingest_attempt({"task": "t", "response": {"before": before}})
    assert as_text.ok and as_dict.ok
    assert as_text.task.project_id == PROJECT_ID


def test_a_malformed_row_is_reported_not_raised():
    assert ingest_attempt({"TASK": "t", "RESPONSE": "not json"}).error
    assert ingest_attempt({"TASK": "", "RESPONSE": "{}"}).error
    assert ingest_attempt({"TASK": "t", "RESPONSE": "{}"}).error


def test_a_bad_row_does_not_take_the_batch_down_with_it():
    good = _row({STEP_RUBRIC: _step({"criteria": [_criterion("1", 3)]})}, task="good")
    tasks, results = ingest_rows([good, {"TASK": "bad", "RESPONSE": "not json"}])
    assert [t.task_id for t in tasks] == ["good"]
    assert sum(1 for r in results if r.error) == 1
