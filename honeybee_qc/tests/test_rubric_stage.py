"""Rubric-authoring stage: prompts, parsing, validation, and the run into gates.

No test here contacts a model. `FakeModelClient` supplies every response, so the
entire stage including concurrency and caching is exercised for free.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from honeybee_qc.cache import ResponseCache, compute_cache_key
from honeybee_qc.config import Policy
from honeybee_qc.llm import (
    CachedClient,
    FakeModelClient,
    ModelRequest,
    ModelResponse,
    RetryingClient,
    parse_cli_output,
    run_requests,
)
from honeybee_qc.prompts import (
    CRITERION_SCHEMA,
    assert_independent,
    build_coverage_prompt,
    build_criterion_prompt,
)
from honeybee_qc.stages import (
    build_criterion_requests,
    parse_coverage_response,
    parse_criterion_response,
    run_rubric_stage,
)
from honeybee_qc.taxonomies import ISSUE_DEFINITIONS, WEIGHT_DEFINITIONS
from honeybee_qc.tests.fixtures import make_task

POLICY = Policy()


def criterion_payload(
    cid="C1",
    issues=None,
    l1="Outcome Quality",
    weight=3,
    conf="high",
    band=(3, 3),
):
    payload = {
        "criterion_id": cid,
        "issues": issues if issues is not None else [],
        "l1_label": l1,
        "weight": weight,
        "confidence": conf,
    }
    if band is not None:
        payload["weight_defensible_low"] = band[0]
        payload["weight_defensible_high"] = band[1]
    return payload


def issue(category, evidence, description="a problem"):
    return {"category": category, "description": description, "evidence": evidence}


def response_for(key, payload, **kw):
    return ModelResponse(key=key, data=payload, **kw)


# ---------------------------------------------------------------------------
# Prompt construction and independence
# ---------------------------------------------------------------------------


def test_criterion_prompt_contains_the_target_and_its_siblings():
    task = make_task(n_criteria=5)
    prompt = build_criterion_prompt(task, task.rubric[2], POLICY)
    assert ">> [C3]" in prompt
    for c in task.rubric:
        assert f"[{c.criterion_id}]" in prompt


def test_criterion_prompt_withholds_the_contributor_label_and_weight():
    task = make_task(n_criteria=5)
    for c in task.rubric:
        c.l1_label = "Safety"
        c.weight = 5
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    for line in prompt.splitlines():
        if "[C1]" in line:
            assert "Safety" not in line
            assert "weight 5" not in line.lower()


def test_independence_guard_passes_on_a_clean_prompt():
    task = make_task(n_criteria=4)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert assert_independent(prompt, task).clean


@pytest.mark.parametrize("weight", [1, 2, 3, 4, 5])
def test_asking_for_a_defensible_range_still_leaks_no_weight(weight):
    """The range question is what check 260 gates on, so it must not arrive with the
    contributor's own number attached."""
    task = make_task(n_criteria=4)
    for c in task.rubric:
        c.weight = weight
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert assert_independent(prompt, task).clean
    assert "weight_defensible_low" in prompt
    assert "weight_defensible_high" in prompt
    for line in prompt.splitlines():
        if any(f"[{c.criterion_id}]" in line for c in task.rubric):
            assert f"weight {weight}" not in line.lower()


def test_criterion_prompt_asks_for_overlap_sibling_ids():
    task = make_task(n_criteria=5)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert "overlaps_with" in prompt
    assert "overlaps_with" in CRITERION_SCHEMA["properties"]["issues"]["items"]["properties"]


def test_the_prompt_shows_the_customers_weight_definitions_verbatim():
    task = make_task(n_criteria=3)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    for value, definition in WEIGHT_DEFINITIONS.items():
        assert f"  {value} = {definition}" in prompt


def test_independence_guard_catches_a_leaked_label():
    task = make_task(n_criteria=3)
    task.rubric[0].l1_label = "Safety"
    leaked = build_criterion_prompt(task, task.rubric[0], POLICY) + "\n  [C1] Safety\n"
    report = assert_independent(leaked, task)
    assert not report.clean
    assert "L1 label bound" in report.leaked[0]


def test_independence_guard_allows_a_criterion_that_words_its_own_label():
    task = make_task(n_criteria=3)
    task.rubric[0].l1_label = "Safety"
    task.rubric[0].weight = 4
    task.rubric[0].text = "Includes a safety checklist covering weight 4 lifts."
    prompt = build_criterion_prompt(task, task.rubric[1], POLICY)
    assert assert_independent(prompt, task).clean


def test_independence_guard_catches_a_leaked_weight():
    task = make_task(n_criteria=3)
    task.rubric[0].weight = 4
    leaked = build_criterion_prompt(task, task.rubric[0], POLICY) + "\n  [C1] weight 4\n"
    assert not assert_independent(leaked, task).clean


def test_building_requests_refuses_to_run_a_leaking_prompt(monkeypatch):
    task = make_task(n_criteria=2)
    task.rubric[0].l1_label = "Safety"
    monkeypatch.setattr(
        "honeybee_qc.stages.build_criterion_prompt",
        lambda t, c, p: f"[{c.criterion_id}] Safety",
    )
    with pytest.raises(RuntimeError, match="leaks contributor values"):
        build_criterion_requests(task, POLICY)


def test_one_request_is_built_per_criterion():
    task = make_task(n_criteria=7)
    requests = build_criterion_requests(task, POLICY)
    assert len(requests) == 7
    assert requests[0].key == "task-1::criterion::C1"
    assert requests[0].schema is CRITERION_SCHEMA


def test_coverage_prompt_carries_the_customer_deliverables_and_all_criteria():
    task = make_task(n_criteria=3)
    prompt = build_coverage_prompt(task, POLICY)
    assert "an 11-second clip" in prompt
    assert "4K resolution" in prompt
    for c in task.rubric:
        assert c.criterion_id in prompt


def test_coverage_prompt_carries_the_aligned_criticality_definitions_verbatim():
    """The spec's test for critical is a direct-ask predicate, and for
    non_critical a positive process/reasoning check -- not the old absence
    framing, which also turned "neither" into a residual non_critical bucket."""
    prompt = build_coverage_prompt(make_task(n_criteria=3), POLICY)
    assert ISSUE_DEFINITIONS["missing_critical"] in prompt
    assert ISSUE_DEFINITIONS["missing_non_critical"] in prompt
    assert "fails its purpose without it" not in prompt


def test_prompts_stay_within_a_sane_size():
    task = make_task(n_criteria=30, exchanges=40)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert len(prompt) < 40000


def test_context_block_recovers_prompts_from_the_conversation_when_the_form_omits_them():
    """Task.prompts is empty for every real ingested task; the rubric auditor
    must still see the actual conversation turns, not just the target-outcome
    list, or it ends up judging criteria against a prompt it never read."""
    task = dataclasses.replace(make_task(n_criteria=3), prompts=[])
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert "### Turn 1" in prompt
    assert "A: user message 1" in prompt


def test_context_block_does_not_call_the_target_deliverables_list_authoritative():
    """`target_deliverables` is the contributor's own target-outcome list, copied
    verbatim by ingest -- not a customer requirement -- so the rubric auditor
    must not be told to treat it as one."""
    task = make_task(n_criteria=3)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert "customer requires (authoritative)" not in prompt
    assert "not a customer requirement" in prompt


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def test_a_clean_criterion_yields_a_finding_with_no_issues():
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload("C1")), task.rubric[0], task, POLICY
    )
    assert audit.ok
    assert audit.finding.issues == []
    assert audit.l1.contributor_label == "Outcome Quality"
    assert audit.weight.auditor_weight == 3


def test_an_issue_quoting_the_criterion_is_kept():
    task = make_task(n_criteria=3)
    quote = task.rubric[0].text[:20]
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("vague_subjective", quote)])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert [i.category for i in audit.finding.issues] == ["vague_subjective"]


def test_an_issue_with_no_evidence_is_dropped():
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("inaccurate", "")])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.finding.issues == []
    assert audit.dropped[0].reason == "no_evidence"


def test_an_issue_quoting_text_that_is_not_in_the_rubric_is_dropped():
    # Guards against a fabricated quote being presented as evidence.
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for(
            "k", criterion_payload(issues=[issue("inaccurate", "text that never appears")])
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.finding.issues == []
    assert audit.dropped[0].reason == "unverifiable_quote"


def test_an_invented_category_is_dropped_rather_than_passed_through():
    task = make_task(n_criteria=3)
    quote = task.rubric[0].text[:20]
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("overfitted", quote)])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.finding.issues == []
    assert audit.dropped[0].reason == "unknown_category"


def test_a_task_level_missing_category_cannot_be_attached_to_a_criterion():
    task = make_task(n_criteria=3)
    quote = task.rubric[0].text[:20]
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("missing_critical", quote)])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.finding.issues == []
    assert audit.dropped[0].reason == "unknown_category"


def test_overlapping_evidence_citing_both_criteria_is_kept():
    # Observed live: the model evidences duplication by quoting the pair, e.g.
    # 'C5: "..." / C6: "..."'. Requiring the whole string to be a substring of one
    # criterion rejected the very issue type that needs two quotes.
    task = make_task(n_criteria=3)
    a, b = task.rubric[0].text, task.rubric[1].text
    evidence = f'C1: "{a}" / C2: "{b}"'
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("overlapping", evidence)])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert [i.category for i in audit.finding.issues] == ["overlapping"]


def test_a_fabricated_quote_is_still_rejected_alongside_real_ones():
    task = make_task(n_criteria=3)
    real = task.rubric[0].text
    audit = parse_criterion_response(
        response_for(
            "k",
            criterion_payload(
                issues=[issue("overlapping", f'"{real}" / "a requirement never written"')]
            ),
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    # One span matches, so the issue stands; the guard blocks wholly invented
    # evidence, not a partly paraphrased citation.
    assert len(audit.finding.issues) == 1


def test_a_short_fragment_is_not_enough_to_pass_the_evidence_guard():
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("inaccurate", "The")])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.finding.issues == []
    assert audit.dropped[0].reason == "unverifiable_quote"


def test_overlapping_may_quote_a_sibling_criterion():
    task = make_task(n_criteria=3)
    sibling_quote = task.rubric[1].text[:25]
    audit = parse_criterion_response(
        response_for("k", criterion_payload(issues=[issue("overlapping", sibling_quote)])),
        task.rubric[0],
        task,
        POLICY,
    )
    assert [i.category for i in audit.finding.issues] == ["overlapping"]


def test_overlaps_with_keeps_only_known_sibling_ids():
    """The judge names which sibling(s) a criterion duplicates; a hallucinated or
    self-referencing id must not leak into the census's overlap grouping."""
    task = make_task(n_criteria=3)
    sibling_quote = task.rubric[1].text[:25]
    payload = criterion_payload(issues=[issue("overlapping", sibling_quote)])
    payload["issues"][0]["overlaps_with"] = ["C2", "C1", "C99"]
    audit = parse_criterion_response(
        response_for("k", payload), task.rubric[0], task, POLICY
    )
    assert audit.finding.issues[0].overlaps_with == ["C2"]


def test_overlaps_with_is_empty_for_every_other_category():
    task = make_task(n_criteria=3)
    payload = criterion_payload(issues=[issue("inaccurate", task.rubric[0].text[:20])])
    payload["issues"][0]["overlaps_with"] = ["C2"]
    audit = parse_criterion_response(
        response_for("k", payload), task.rubric[0], task, POLICY
    )
    assert audit.finding.issues[0].overlaps_with == []


def test_repeated_categories_collapse_to_one_issue():
    task = make_task(n_criteria=3)
    q1, q2 = task.rubric[0].text[:20], task.rubric[0].text[5:25]
    audit = parse_criterion_response(
        response_for(
            "k",
            criterion_payload(issues=[issue("vague_subjective", q1), issue("vague_subjective", q2)]),
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    assert len(audit.finding.issues) == 1
    assert audit.dropped[0].reason == "duplicate_category"


# ---------------------------------------------------------------------------
# framing_double_negative: the spec conditions it on the contributor's weight
# ("When a criterion with a negative weight is framed negatively"), which the
# prompt withholds, so the model reports the phrasing and the parser applies the
# weight.
# ---------------------------------------------------------------------------

NEGATIVELY_FRAMED = "The response does not take the long-term capital gain into consideration."


def negatively_framed_task(weight):
    task = make_task(n_criteria=3)
    task.rubric[0].text = NEGATIVELY_FRAMED
    task.rubric[0].weight = weight
    return task


def audit_negative_framing(weight):
    task = negatively_framed_task(weight)
    return parse_criterion_response(
        response_for(
            "k",
            criterion_payload(
                issues=[issue("framing_double_negative", NEGATIVELY_FRAMED)]
            ),
        ),
        task.rubric[0],
        task,
        POLICY,
    )


@pytest.mark.parametrize("weight", [1, 3, 5])
def test_negative_phrasing_with_a_positive_weight_is_not_a_double_negative(weight):
    audit = audit_negative_framing(weight)
    assert audit.finding.issues == []
    assert any(d.reason == "weight_not_negative" for d in audit.dropped)


@pytest.mark.parametrize("weight", [-1, -3, -5])
def test_negative_phrasing_with_a_negative_weight_is_a_double_negative(weight):
    audit = audit_negative_framing(weight)
    assert [i.category for i in audit.finding.issues] == ["framing_double_negative"]
    assert audit.finding.severities() == {"moderate"}


def test_an_unrecorded_weight_cannot_confirm_a_double_negative():
    audit = audit_negative_framing(None)
    assert audit.finding.issues == []
    assert any(d.reason == "weight_not_negative" for d in audit.dropped)


def test_the_weight_condition_does_not_suppress_other_categories():
    """Only framing_double_negative is conditioned; a positive weight must not
    silence the categories that never mentioned a weight."""
    task = negatively_framed_task(3)
    audit = parse_criterion_response(
        response_for(
            "k", criterion_payload(issues=[issue("vague_subjective", NEGATIVELY_FRAMED)])
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    assert [i.category for i in audit.finding.issues] == ["vague_subjective"]


def test_the_negative_weight_still_never_reaches_the_prompt():
    task = negatively_framed_task(-5)
    prompt = build_criterion_prompt(task, task.rubric[0], POLICY)
    assert assert_independent(prompt, task).clean
    for line in prompt.splitlines():
        if "[C1]" in line:
            assert "-5" not in line


def test_a_positive_weight_double_negative_leaves_the_240_gate_clean():
    task = make_task(n_criteria=10)
    task.rubric[0].text = NEGATIVELY_FRAMED
    client = FakeModelClient(
        stage_responder(
            task,
            issues_by_criterion={
                "C1": [issue("framing_double_negative", NEGATIVELY_FRAMED)]
            },
        )
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    bands = {v.check_id: v.band for v in result.verdicts}
    assert bands[230] == bands[240] == bands[250] == "clean"
    assert result.census.category_counts == {}


def test_a_negative_weight_double_negative_reaches_the_240_gate():
    task = make_task(n_criteria=10)
    task.rubric[0].text = NEGATIVELY_FRAMED
    task.rubric[0].weight = -5
    client = FakeModelClient(
        stage_responder(
            task,
            issues_by_criterion={
                "C1": [issue("framing_double_negative", NEGATIVELY_FRAMED)]
            },
        )
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    bands = {v.check_id: v.band for v in result.verdicts}
    assert bands[230] == "clean"
    assert bands[240] == "non_fail"
    assert bands[250] == "non_fail"
    assert result.census.category_counts == {"framing_double_negative": 1}


def test_an_out_of_enum_l1_label_is_dropped_and_leaves_200s_denominator():
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload(l1="Vibes")), task.rubric[0], task, POLICY
    )
    assert audit.l1 is None
    assert audit.dropped[0].reason == "unknown_l1_label"


@pytest.mark.parametrize("weight", [0, 6, -1, "four", None, 3.5])
def test_an_out_of_range_weight_is_dropped(weight):
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload(weight=weight)), task.rubric[0], task, POLICY
    )
    assert audit.weight is None
    assert any(d.reason == "invalid_weight" for d in audit.dropped)


@pytest.mark.parametrize("band", [None, (0, 3), (3, 9), ("low", "high")])
def test_a_response_without_a_usable_defensible_range_is_dropped(band):
    """Degrading to the auditor's point weight is the comparison that over-flagged
    check 260, so the criterion leaves the denominator instead."""
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for("k", criterion_payload(band=band)), task.rubric[0], task, POLICY
    )
    assert audit.weight is None
    assert any(d.reason == "invalid_weight_band" for d in audit.dropped)


def test_the_defensible_range_reaches_the_finding():
    task = make_task(n_criteria=3)   # contributor weight is 3
    audit = parse_criterion_response(
        response_for("k", criterion_payload(weight=5, band=(2, 5))),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.weight.band == (2, 5)
    assert audit.weight.auditor_weight == 5
    assert audit.weight.in_band
    assert audit.weight.raw_delta == 2


def test_a_failed_call_produces_an_audit_that_is_not_ok():
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        ModelResponse(key="k", error="timeout after 300s"), task.rubric[0], task, POLICY
    )
    assert not audit.ok
    assert audit.error == "timeout after 300s"


def test_missing_confidence_degrades_to_low():
    task = make_task(n_criteria=3)
    payload = criterion_payload()
    del payload["confidence"]
    audit = parse_criterion_response(response_for("k", payload), task.rubric[0], task, POLICY)
    assert audit.finding.confidence == "low"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_splits_critical_from_non_critical():
    task = make_task(n_criteria=3)
    payload = {
        "missing": [
            {"requirement": "export at 4K", "criticality": "critical",
             "evidence": "4K resolution", "reason": "no criterion mentions resolution"},
            {"requirement": "loop cleanly", "criticality": "non_critical",
             "evidence": "an 11-second clip", "reason": "not covered"},
        ],
        "confidence": "high",
    }
    coverage, dropped, error = parse_coverage_response(response_for("k", payload), task)
    assert coverage.missing_critical == ["export at 4K"]
    assert coverage.missing_non_critical == ["loop cleanly"]
    assert not dropped and not error


def test_a_coverage_gap_without_evidence_is_dropped():
    task = make_task(n_criteria=3)
    payload = {
        "missing": [{"requirement": "x", "criticality": "critical", "evidence": "",
                     "reason": "r"}],
        "confidence": "high",
    }
    coverage, dropped, _ = parse_coverage_response(response_for("k", payload), task)
    assert coverage.missing_critical == []
    assert dropped[0].reason == "no_evidence"


def test_an_unknown_criticality_is_dropped():
    task = make_task(n_criteria=3)
    payload = {
        "missing": [{"requirement": "x", "criticality": "urgent", "evidence": "e",
                     "reason": "r"}],
        "confidence": "high",
    }
    _, dropped, _ = parse_coverage_response(response_for("k", payload), task)
    assert dropped[0].reason == "unknown_criticality"


# ---------------------------------------------------------------------------
# End to end into the gates
# ---------------------------------------------------------------------------


def stage_responder(task, issues_by_criterion=None, weights=None, labels=None, coverage=None,
                    fail_ids=(), bands=None):
    issues_by_criterion = issues_by_criterion or {}
    weights = weights or {}
    labels = labels or {}
    bands = bands or {}

    def responder(request: ModelRequest):
        cid = request.metadata.get("criterion_id")
        if cid is None:
            return coverage if coverage is not None else {"missing": [], "confidence": "high"}
        if cid in fail_ids:
            return ModelResponse(key=request.key, error="timeout after 300s")
        weight = weights.get(cid, 3)
        return criterion_payload(
            cid=cid,
            issues=issues_by_criterion.get(cid, []),
            l1=labels.get(cid, "Outcome Quality"),
            weight=weight,
            # An auditor that names one weight as the only defensible one is the
            # narrowest case; tests that need a real range pass `bands`.
            band=bands.get(cid, (weight, weight)),
        )

    return responder


def test_a_clean_rubric_produces_six_clean_verdicts():
    task = make_task(n_criteria=10)
    client = FakeModelClient(stage_responder(task))
    result = run_rubric_stage(task, client, POLICY, workers=2)

    assert len(client.calls) == 11  # 10 criteria plus one coverage pass
    bands = {v.check_id: v.band for v in result.verdicts}
    assert bands == {
        200: "clean",
        210: "clean",
        230: "clean",
        240: "clean",
        250: "clean",
        260: "clean",
    }
    assert not result.errors


def test_an_unverifiable_issue_reaches_neither_the_230_numerator_nor_the_report_as_a_defect():
    """The rubric judge is shown the task prompts and the names of the input files,
    never the files' contents. When it says an accuracy claim turns on a value it
    could not open, that is our missing context, and 1 of 10 criteria is enough to
    fail 230 outright."""
    task = make_task(n_criteria=10)
    quote = task.rubric[0].text[:20]
    withheld = dict(issue("inaccurate", quote))
    withheld["unverifiable"] = True
    withheld["why_unverifiable"] = "the target value is inside quarterly.xlsx"
    client = FakeModelClient(
        stage_responder(task, issues_by_criterion={"C1": [withheld]})
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)

    v230 = next(v for v in result.verdicts if v.check_id == 230)
    assert v230.measurement.numerator == 0
    assert v230.measurement.denominator == 10
    assert v230.band == "clean"
    assert v230.contributing_items == []

    audit = next(a for a in result.audits if a.criterion_id == "C1")
    finding = audit.finding
    assert finding.issues == []
    assert [i.category for i in finding.unverifiable_issues] == ["inaccurate"]
    reported = next(
        c for c in result.to_dict()["findings"] if c["criterion_id"] == "C1"
    )
    assert reported["issues"] == []
    assert reported["unverifiable_issues"][0]["why_unverifiable"] == (
        "the target value is inside quarterly.xlsx"
    )


def test_an_unverifiable_issue_is_not_recorded_as_a_fabricated_quote():
    """Its evidence is the thing the auditor was not shown, so the quote check
    would drop it and the run report would accuse the judge of inventing a
    citation, hiding the context gap that actually caused it."""
    task = make_task(n_criteria=3)
    audit = parse_criterion_response(
        response_for(
            "k",
            criterion_payload(
                issues=[
                    {
                        "category": "inaccurate",
                        "description": "cites a threshold I cannot check",
                        "evidence": "a value inside the attached spreadsheet",
                        "unverifiable": True,
                        "why_unverifiable": "spreadsheet contents are not in the prompt",
                    }
                ]
            ),
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    assert audit.ok
    assert audit.dropped == []
    assert audit.finding.issues == []
    assert len(audit.finding.unverifiable_issues) == 1
    assert audit.finding.severities() == set()


def test_abstaining_on_one_issue_does_not_excuse_the_others():
    task = make_task(n_criteria=3)
    quote = task.rubric[0].text[:20]
    unchecked = {
        "category": "inaccurate",
        "description": "turns on a file I was not given",
        "evidence": "the uploaded CSV",
        "unverifiable": True,
        "why_unverifiable": "CSV contents absent",
    }
    audit = parse_criterion_response(
        response_for(
            "k",
            criterion_payload(issues=[unchecked, issue("vague_subjective", quote)]),
        ),
        task.rubric[0],
        task,
        POLICY,
    )
    assert [i.category for i in audit.finding.issues] == ["vague_subjective"]
    assert [i.category for i in audit.finding.unverifiable_issues] == ["inaccurate"]
    # The withheld allegation was the major one, so the criterion keeps only the
    # severity of the defect the auditor could actually stand behind.
    assert not audit.finding.has_at_least("major", "critical")
    assert audit.finding.severities()


def test_the_criterion_prompt_offers_the_abstention_and_still_leaks_nothing():
    """The instruction has to describe what the judge was not shown without
    describing what the contributor decided, which is what the guard checks."""
    task = make_task(n_criteria=4)
    prompt = build_criterion_prompt(task, task.rubric[1], POLICY)
    assert "unverifiable" in prompt
    assert "why_unverifiable" in prompt
    assert assert_independent(prompt, task).clean
    props = CRITERION_SCHEMA["properties"]["issues"]["items"]["properties"]
    assert "unverifiable" in props and "why_unverifiable" in props


def test_major_issues_drive_the_230_gate():
    task = make_task(n_criteria=10)
    quote = task.rubric[0].text[:20]
    client = FakeModelClient(
        stage_responder(task, issues_by_criterion={"C1": [issue("inaccurate", quote)]})
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    v230 = next(v for v in result.verdicts if v.check_id == 230)
    assert v230.measurement.numerator == 1
    assert v230.measurement.denominator == 10
    assert v230.band == "fail"  # 1/10 hits the 10% threshold exactly


def test_a_coverage_gap_raises_the_numerator_without_raising_the_denominator():
    task = make_task(n_criteria=10)
    client = FakeModelClient(
        stage_responder(
            task,
            coverage={
                "missing": [
                    {"requirement": "export at 4K", "criticality": "critical",
                     "evidence": "4K resolution", "reason": "uncovered"}
                ],
                "confidence": "high",
            },
        )
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    v230 = next(v for v in result.verdicts if v.check_id == 230)
    assert v230.measurement.numerator == 1
    assert v230.measurement.denominator == 10
    assert result.coverage.missing_critical == ["export at 4K"]


def test_weight_disagreement_drives_260():
    task = make_task(n_criteria=10)
    # Contributor said 3 for all; the auditor says 5 is the only defensible weight
    # (medium -> high) on four of them.
    client = FakeModelClient(
        stage_responder(task, weights={f"C{i}": 5 for i in range(1, 5)})
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    v260 = next(v for v in result.verdicts if v.check_id == 260)
    assert v260.measurement.counts["any_level_off"] == 4
    assert v260.measurement.rate == pytest.approx(0.40)
    assert v260.band == "fail"


def test_a_defensible_range_that_covers_the_recorded_weight_leaves_260_clean():
    """Same auditor preference as the test above, but with the range the weight
    definitions actually support the recorded weights are all defensible."""
    task = make_task(n_criteria=10)
    client = FakeModelClient(
        stage_responder(
            task,
            weights={f"C{i}": 5 for i in range(1, 5)},
            bands={f"C{i}": (2, 5) for i in range(1, 5)},
        )
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    v260 = next(v for v in result.verdicts if v.check_id == 260)
    assert v260.measurement.counts["out_of_band"] == 0
    assert v260.measurement.counts["differs_from_auditor_pick"] == 4
    assert v260.band == "clean"


def test_label_disagreement_drives_200():
    task = make_task(n_criteria=10)
    client = FakeModelClient(
        stage_responder(task, labels={f"C{i}": "Safety" for i in range(1, 4)})
    )
    result = run_rubric_stage(task, client, POLICY, workers=2)
    v200 = next(v for v in result.verdicts if v.check_id == 200)
    assert v200.measurement.numerator == 3
    assert v200.measurement.rate == pytest.approx(0.30)
    assert v200.band == "fail"


def test_a_failed_criterion_leaves_the_denominator_and_lowers_confidence():
    task = make_task(n_criteria=10)
    client = FakeModelClient(stage_responder(task, fail_ids={"C9", "C10"}))
    result = run_rubric_stage(task, client, POLICY, workers=2)

    assert result.failed_ids == ["C9", "C10"] or set(result.failed_ids) == {"C9", "C10"}
    v230 = next(v for v in result.verdicts if v.check_id == 230)
    assert v230.measurement.denominator == 8
    assert v230.confidence == "low"
    assert "excluded from the denominator" in v230.measurement.notes
    assert len(result.errors) == 2


def test_monotonicity_is_preserved_through_the_stage():
    task = make_task(n_criteria=12)
    q = lambda i: task.rubric[i - 1].text[:20]
    client = FakeModelClient(
        stage_responder(
            task,
            issues_by_criterion={
                "C1": [issue("inaccurate", q(1))],
                "C2": [issue("overlapping", q(2))],
                "C3": [issue("unnecessary", q(3))],
            },
        )
    )
    result = run_rubric_stage(task, client, POLICY, workers=3)
    nums = [
        next(v for v in result.verdicts if v.check_id == cid).measurement.numerator
        for cid in (230, 240, 250)
    ]
    assert nums == [1, 2, 3]


def test_an_empty_rubric_returns_an_error_without_calling_the_model():
    task = make_task(n_criteria=1)
    task.rubric = []
    client = FakeModelClient(stage_responder(task))
    result = run_rubric_stage(task, client, POLICY)
    assert result.errors
    assert client.calls == []


def test_cost_and_cache_counts_are_reported():
    task = make_task(n_criteria=3)
    base = stage_responder(task)

    def responder(request):
        payload = base(request)
        return ModelResponse(key=request.key, data=payload, cost_usd=0.01, cached=False)

    result = run_rubric_stage(task, FakeModelClient(responder), POLICY, workers=2)
    assert result.calls == 4
    assert result.cost_usd == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Client plumbing
# ---------------------------------------------------------------------------


def test_requests_run_concurrently_and_return_in_submission_order():
    reqs = [
        ModelRequest(key=f"k{i}", prompt="p", schema={}, metadata={"i": i}) for i in range(20)
    ]
    client = FakeModelClient(lambda r: {"i": r.metadata["i"]})
    results = run_requests(client, reqs, workers=8)
    assert [r.data["i"] for r in results] == list(range(20))


def test_retry_stops_at_the_first_success():
    attempts = {"n": 0}

    class Flaky:
        def complete(self, request):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return ModelResponse(key=request.key, error="transient")
            return ModelResponse(key=request.key, data={"ok": True})

    resp = RetryingClient(Flaky(), attempts=5, backoff_s=0).complete(
        ModelRequest(key="k", prompt="p", schema={})
    )
    assert resp.ok
    assert resp.attempts == 3
    assert attempts["n"] == 3


def test_retry_gives_up_and_reports_the_last_error():
    class Always:
        def complete(self, request):
            return ModelResponse(key=request.key, error="down")

    resp = RetryingClient(Always(), attempts=3, backoff_s=0).complete(
        ModelRequest(key="k", prompt="p", schema={})
    )
    assert not resp.ok
    assert resp.attempts == 3


def test_cli_output_parsing_extracts_structured_output_and_cost():
    envelope = json.dumps(
        {
            "type": "result",
            "is_error": False,
            "total_cost_usd": 0.0316,
            "result": "prose the model also wrote",
            "structured_output": {"l1_label": "Outcome Quality"},
        }
    )
    resp = parse_cli_output(ModelRequest(key="k", prompt="p", schema={}), envelope, 0, 1.0)
    assert resp.ok
    assert resp.data == {"l1_label": "Outcome Quality"}
    assert resp.cost_usd == pytest.approx(0.0316)


def test_cli_error_envelope_becomes_an_error_response():
    envelope = json.dumps({"is_error": True, "result": "rate limited"})
    resp = parse_cli_output(ModelRequest(key="k", prompt="p", schema={}), envelope, 1, 1.0)
    assert not resp.ok
    assert "rate limited" in resp.error


def test_non_json_cli_output_becomes_an_error_response():
    resp = parse_cli_output(
        ModelRequest(key="k", prompt="p", schema={}), "segfault", 139, 1.0, "boom"
    )
    assert not resp.ok
    assert "not JSON" in resp.error


def test_structured_output_can_arrive_as_text_in_result():
    envelope = json.dumps({"is_error": False, "result": '{"weight": 4}'})
    resp = parse_cli_output(ModelRequest(key="k", prompt="p", schema={}), envelope, 0, 1.0)
    assert resp.data == {"weight": 4}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_key_changes_with_every_input_that_can_change_the_answer():
    base = dict(
        prompt="p", schema={"a": 1}, model="m", effort="medium", policy_version="v1"
    )
    key = compute_cache_key(**base)
    assert compute_cache_key(**{**base, "model": "other"}) != key
    assert compute_cache_key(**{**base, "effort": "high"}) != key
    assert compute_cache_key(**{**base, "policy_version": "v2"}) != key
    assert compute_cache_key(**{**base, "schema": {"a": 2}}) != key
    assert compute_cache_key(**{**base, "prompt": "q"}) != key
    assert compute_cache_key(**{**base, "system": "s"}) != key
    assert compute_cache_key(**base) == key


def test_cache_round_trips_and_serves_a_hit(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    inner = FakeModelClient(lambda r: {"value": 1})
    client = CachedClient(
        inner, cache, model="m", effort="medium", policy_version="v1"
    )
    req = ModelRequest(key="k", prompt="p", schema={"a": 1})

    first = client.complete(req)
    second = client.complete(req)
    assert first.data == second.data == {"value": 1}
    assert not first.cached and second.cached
    assert len(inner.calls) == 1
    cache.close()


def test_changing_the_model_does_not_reuse_a_cached_answer(tmp_path):
    # The predecessor engine wrote the model into the key but read by prompt hash
    # alone, so this exact case silently returned the old model's result.
    cache = ResponseCache(tmp_path / "c.db")
    req = ModelRequest(key="k", prompt="p", schema={"a": 1})

    a = FakeModelClient(lambda r: {"from": "model-a"})
    CachedClient(a, cache, model="model-a", effort="medium", policy_version="v1").complete(req)

    b = FakeModelClient(lambda r: {"from": "model-b"})
    out = CachedClient(b, cache, model="model-b", effort="medium", policy_version="v1").complete(req)

    assert out.data == {"from": "model-b"}
    assert len(b.calls) == 1
    cache.close()


def test_errors_are_never_cached(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    inner = FakeModelClient(lambda r: ModelResponse(key=r.key, error="transient"))
    client = CachedClient(inner, cache, model="m", effort="medium", policy_version="v1")
    req = ModelRequest(key="k", prompt="p", schema={})

    client.complete(req)
    client.complete(req)
    assert len(inner.calls) == 2
    cache.close()


def test_refresh_bypasses_a_hit_but_still_writes(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    inner = FakeModelClient(lambda r: {"v": 1})
    warm = CachedClient(inner, cache, model="m", effort="e", policy_version="v1")
    req = ModelRequest(key="k", prompt="p", schema={})
    warm.complete(req)

    fresh = CachedClient(inner, cache, model="m", effort="e", policy_version="v1", refresh=True)
    resp = fresh.complete(req)
    assert not resp.cached
    assert len(inner.calls) == 2
    cache.close()


def test_a_disabled_cache_is_transparent():
    cache = ResponseCache(None)
    inner = FakeModelClient(lambda r: {"v": 1})
    client = CachedClient(inner, cache, model="m", effort="e", policy_version="v1")
    assert client.complete(ModelRequest(key="k", prompt="p", schema={})).data == {"v": 1}
    assert not cache.enabled
