"""Stage 4: the informed pass, covering checks 70, 75, 80, 85, 95, 110, 220, 280, 310, 450, 470.

Runs after the blind rating pass and never feeds it. The separation is the point:
these prompts contain the contributor's ratings and justifications, so letting a
result from here reach checks 270, 300, or 400 would anchor the independent
ratings those checks depend on.

Check 220 is the exception to the one-round shape. An autofail allegation fails
the task outright with no middle band, so every allegation goes to a second
opinion and only a confirmed one counts. That makes this stage two rounds: the
first is a single batch, the second runs only if something was alleged, which for
most tasks means it never runs at all.

Check 85 is half deterministic. Its file-reference comparison needs no model, so it
is computed here alongside the model half's entity judgment and the two are scored
as one dimension.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .config import DEFAULT_POLICY, Policy
from .env_context import (
    FileReferenceScan,
    prompt_texts,
    scan_file_references,
    universe_materials,
)
from .findings import (
    ArtifactFinding,
    AutofailFinding,
    DomainRelevanceFinding,
    EnvironmentContextFinding,
    JustificationFinding,
    KeyTurnJustificationFinding,
    PromptConsistencyFinding,
    RelevantTurnFinding,
    TargetOutcomeFinding,
    VerdictFinding,
)
from .gates import (
    evaluate_check_70,
    evaluate_check_75,
    evaluate_check_80,
    evaluate_check_85,
    evaluate_check_95,
    evaluate_check_110,
    evaluate_check_220,
    evaluate_check_450,
    evaluate_check_470,
    evaluate_relevant_turns,
)
from .informed_prompts import (
    ARTIFACT_CONFIRMATION_SCHEMA,
    ARTIFACT_SCHEMA,
    AUTOFAIL_SCHEMA,
    DOMAIN_RELEVANCE_SCHEMA,
    ENVIRONMENT_CONTEXT_SCHEMA,
    INFORMED_SYSTEM_PROMPT,
    JUSTIFICATION_SCHEMA,
    KEY_TURN_JUSTIFICATION_SCHEMA,
    PROMPT_CONSISTENCY_SCHEMA,
    RELEVANT_TURNS_SCHEMA,
    TARGET_OUTCOME_SCHEMA,
    VERDICT_SCHEMA,
    build_artifact_confirmation_prompt,
    build_artifact_prompt,
    build_autofail_prompt,
    build_autofail_second_opinion_prompt,
    build_criterion_turns_prompt,
    build_dimension_justification_prompt,
    build_dimension_turns_prompt,
    build_domain_relevance_prompt,
    build_environment_context_prompt,
    build_key_turn_justification_prompt,
    build_prompt_consistency_prompt,
    build_ranking_justification_prompt,
    build_target_outcome_prompt,
    build_verdict_prompt,
)
from .llm import ModelClient, ModelRequest, ModelResponse, run_requests
from .models import Task
from .sampling import (
    AgreementLog,
    aggregate_justifications,
    aggregate_key_turn_justification,
    aggregate_relevant_turns,
    expand_requests,
    samples_for,
)
from .scoring import CheckVerdict, Measurement, build_verdict

INFORMED_CHECKS = (70, 75, 80, 85, 95, 110, 220, 280, 310, 450, 470)


@dataclass
class InformedStageResult:
    task_id: str
    domain_relevance: DomainRelevanceFinding | None = None
    prompt_consistency: PromptConsistencyFinding | None = None
    environment_context: list[EnvironmentContextFinding] = field(default_factory=list)
    # None until the deterministic half has been computed, which distinguishes
    # "scanned and found nothing" from "never scanned".
    file_scan: FileReferenceScan | None = None
    entity_half_ran: bool = False
    target_outcome: list[TargetOutcomeFinding] = field(default_factory=list)
    artifacts: list[ArtifactFinding] = field(default_factory=list)
    key_turn_justification: KeyTurnJustificationFinding | None = None
    autofails: list[AutofailFinding] = field(default_factory=list)
    relevant_turns: list[RelevantTurnFinding] = field(default_factory=list)
    justifications: list[JustificationFinding] = field(default_factory=list)
    # Justifications the judge could not read at all, held apart from the list
    # above so they reach neither 450's numerator nor its denominator. Scoring
    # them clean would report a justification as audited that nobody audited, and
    # scoring them defective would charge the contributor for a transcript this
    # audit failed to render.
    unverifiable_justifications: list[JustificationFinding] = field(
        default_factory=list
    )
    verdict: VerdictFinding | None = None

    verdicts: list[CheckVerdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0
    cached_calls: int = 0
    cost_usd: float = 0.0
    # Which per-item claims the repeated draws split on. Empty when every check
    # ran at one sample, which is the single-sample configuration's whole report.
    agreement: AgreementLog = field(default_factory=AgreementLog)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "cost_usd": round(self.cost_usd, 4),
            "domain_relevance": (
                {
                    "assessment": self.domain_relevance.assessment,
                    "assigned_domain": self.domain_relevance.assigned_domain,
                    "prompt_quote": self.domain_relevance.prompt_quote,
                    "domain_basis": self.domain_relevance.domain_basis,
                    "evidenced": self.domain_relevance.is_evidenced,
                }
                if self.domain_relevance
                else None
            ),
            "prompt_consistency": (
                {
                    "assessment": self.prompt_consistency.assessment,
                    "pre_seeded_quote": self.prompt_consistency.pre_seeded_quote,
                    "submitted_quote": self.prompt_consistency.submitted_quote,
                    "evidenced": self.prompt_consistency.is_evidenced,
                }
                if self.prompt_consistency
                else None
            ),
            "environment_context": {
                "entity_half_ran": self.entity_half_ran,
                "file_scan": self.file_scan.to_dict() if self.file_scan else None,
                "references": [
                    {
                        "reference": f.reference,
                        "kind": f.kind,
                        "basis": f.basis,
                        "quote": f.quote,
                        "checked_against": list(f.checked_against),
                        "evidenced": f.evidenced,
                        "violation": f.is_violation,
                    }
                    for f in self.environment_context
                ],
            },
            "target_outcome_entries": len(self.target_outcome),
            "target_outcome_issues": sum(
                1 for f in self.target_outcome if f.is_issue
            ),
            "artifacts": [
                {
                    "model": a.model,
                    "claimed": a.claimed_files,
                    "uploaded": a.uploaded_files,
                    "missing": a.missing_files,
                    "basis": a.basis,
                    "shortfall": a.shortfall,
                    "unverifiable": a.unverifiable_files,
                    "why_unverifiable": a.why_unverifiable,
                }
                for a in self.artifacts
            ],
            "key_turn_justification_issue": (
                self.key_turn_justification.is_issue
                if self.key_turn_justification
                else None
            ),
            "key_turn_justification_unverifiable": (
                {
                    "unverifiable": self.key_turn_justification.unverifiable,
                    "why": self.key_turn_justification.why_unverifiable,
                    "issues_withheld": list(
                        self.key_turn_justification.unverifiable_issues
                    ),
                }
                if self.key_turn_justification
                else None
            ),
            "autofails": [
                {
                    "criterion_id": f.criterion_id,
                    "confirmed": f.confirmed,
                    "why": f.why_outcome_is_useless,
                }
                for f in self.autofails
            ],
            "relevant_turns": [
                {
                    "check_id": f.check_id,
                    "item": f.item_id,
                    "model": f.model,
                    "incorrect": f.incorrect_turns,
                    "missing_turn": f.missing_turn,
                    "unverifiable": f.unverifiable_turns,
                    "why_unverifiable": f.why_unverifiable,
                }
                for f in self.relevant_turns
                if f.incorrect_turns or f.missing_turn or f.unverifiable_turns
            ],
            "justifications": [
                {"item": f.item_id, "conditions": f.triggered_conditions()}
                for f in self.justifications
                if f.has_any_issue()
            ],
            "justifications_audited": len(self.justifications),
            "justifications_unverifiable": [
                {"item": f.item_id, "why": f.why_unverifiable}
                for f in self.unverifiable_justifications
            ],
            "states_preference": (
                self.verdict.states_preference if self.verdict else None
            ),
            # A judge that split 2-1 on an item is saying the item is marginal.
            # That is a fact about the evidence and belongs beside the finding,
            # not in the bin.
            "agreement": self.agreement.to_dict(),
            "errors": list(self.errors),
        }


def _norm_filename(name: str) -> str:
    """Compare on the bare filename, case-folded.

    The contributor uploads a file the model delivered behind a generated URL, so
    the two names agree on the basename and disagree on everything around it.
    """
    return name.strip().replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def _best_fuzzy_match(name: str, candidates: list[str]) -> tuple[str | None, float]:
    """The uploaded name closest to `name`, and how close, by similarity ratio.

    Both sides are already case-folded basenames by the time this is called.
    Ratio, not a fixed edit distance, because the suffixes that make an exact
    match fail -- " (2)", "_v2", " - Copy" -- vary in length with the filename
    they are appended to, so a fixed distance would need re-tuning per name
    length. `difflib`'s ratio is already length-normalised.
    """
    best_name: str | None = None
    best_ratio = 0.0
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, name, candidate).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, candidate
    return best_name, best_ratio


def _is_opaque(name: str) -> bool:
    """A CDN path segment that is a content ID rather than a filename.

    Scale's uploads look like `.../6463e58.../ZQjy98e0JqvrEIg`: no extension, and
    the object serves no Content-Disposition, so nothing recoverable names it.
    Treating that as a filename makes every claimed file look absent.
    """
    base = _norm_filename(name)
    stem, dot, ext = base.rpartition(".")
    return not dot or not stem or len(ext) > 5 or not ext.isalnum()


def _confidence(data: dict) -> str:
    value = str(data.get("confidence", "high")).lower()
    return value if value in {"low", "medium", "high"} else "high"


def can_judge_environment_context(task: Task) -> bool:
    """Whether check 85's entity half has anything to judge against."""
    return bool(
        any((t or "").strip() for t in prompt_texts(task)) and universe_materials(task)
    )


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def build_informed_requests(
    task: Task, policy: Policy = DEFAULT_POLICY
) -> list[ModelRequest]:
    requests: list[ModelRequest] = []

    def add(key: str, prompt: str, schema: dict, metadata: dict) -> None:
        requests.append(
            ModelRequest(
                key=f"{task.task_id}::{key}",
                prompt=prompt,
                schema=schema,
                system=INFORMED_SYSTEM_PROMPT,
                metadata=metadata,
            )
        )

    # No assigned domain means nothing to compare the prompt against, so the
    # call is skipped rather than asked to grade against a blank.
    if (task.assigned_domain or "").strip():
        add(
            "domain_relevance",
            build_domain_relevance_prompt(task, policy),
            DOMAIN_RELEVANCE_SCHEMA,
            {"check_id": 70},
        )

    # No pre-seeded prompt means nothing to compare the submission against; the
    # field is absent on most tasks project-wide, so this abstains far more often
    # than it runs.
    if (task.pre_seeded_prompt or "").strip():
        add(
            "prompt_consistency",
            build_prompt_consistency_prompt(task, policy),
            PROMPT_CONSISTENCY_SCHEMA,
            {"check_id": 75},
        )

    if task.target_outcome:
        add(
            "target_outcome",
            build_target_outcome_prompt(task, policy),
            TARGET_OUTCOME_SCHEMA,
            {"check_id": 80},
        )

    # Absence has to be establishable before it is worth asking about: with no
    # prompt text, or nothing at all to check a reference against, every answer
    # would be unfalsifiable.
    if can_judge_environment_context(task):
        add(
            "environment_context",
            build_environment_context_prompt(task, policy),
            ENVIRONMENT_CONTEXT_SCHEMA,
            {"check_id": 85},
        )

    for sub in task.submissions():
        add(
            f"artifacts::{sub.model}",
            build_artifact_prompt(task, sub, policy),
            ARTIFACT_SCHEMA,
            {"check_id": 95, "model": sub.model},
        )

    if (task.key_turn.justification or "").strip():
        add(
            "key_turn_justification",
            build_key_turn_justification_prompt(task, policy),
            KEY_TURN_JUSTIFICATION_SCHEMA,
            {"check_id": 110},
        )

    if task.rubric:
        add(
            "autofail",
            build_autofail_prompt(task, policy),
            AUTOFAIL_SCHEMA,
            {"check_id": 220},
        )

    for sub in task.submissions():
        # 280's population is the contributor's own score-0 set, which is often
        # empty for a model that passed everything.
        if any(r.model == sub.model and r.score == 0 for r in task.criterion_ratings):
            add(
                f"criterion_turns::{sub.model}",
                build_criterion_turns_prompt(task, sub, policy),
                RELEVANT_TURNS_SCHEMA,
                {"check_id": 280, "model": sub.model},
            )
        if any(d.model == sub.model for d in task.dimension_ratings):
            add(
                f"dimension_turns::{sub.model}",
                build_dimension_turns_prompt(task, sub, policy),
                RELEVANT_TURNS_SCHEMA,
                {"check_id": 310, "model": sub.model},
            )
            add(
                f"dimension_justifications::{sub.model}",
                build_dimension_justification_prompt(task, sub, policy),
                JUSTIFICATION_SCHEMA,
                {"check_id": 450, "model": sub.model},
            )

    if (task.sxs.justification or "").strip():
        add(
            "ranking_justification",
            build_ranking_justification_prompt(task, policy),
            JUSTIFICATION_SCHEMA,
            {"check_id": 450, "model": None},
        )
        add(
            "verdict",
            build_verdict_prompt(task, policy),
            VERDICT_SCHEMA,
            {"check_id": 470},
        )
    return requests


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_domain_relevance(
    response: ModelResponse, task: Task
) -> DomainRelevanceFinding:
    data = response.data or {}
    assessment = str(data.get("assessment", "aligned")).lower()
    if assessment not in {"aligned", "partial", "unrelated"}:
        assessment = "aligned"
    return DomainRelevanceFinding(
        assessment=assessment,  # type: ignore[arg-type]
        assigned_domain=task.assigned_domain,
        prompt_quote=str(data.get("prompt_quote", "")),
        domain_basis=str(data.get("domain_basis", "")),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_prompt_consistency(response: ModelResponse) -> PromptConsistencyFinding:
    data = response.data or {}
    assessment = str(data.get("assessment", "same")).lower()
    if assessment not in {"same", "reworded_same_intent", "different"}:
        assessment = "same"
    return PromptConsistencyFinding(
        assessment=assessment,  # type: ignore[arg-type]
        pre_seeded_quote=str(data.get("pre_seeded_quote", "")),
        submitted_quote=str(data.get("submitted_quote", "")),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


_ENVIRONMENT_KINDS = frozenset(
    {"file", "person", "organisation", "event", "system", "place", "other"}
)


def parse_environment_context(
    response: ModelResponse,
) -> list[EnvironmentContextFinding]:
    """Keep only references that name themselves and say what they were checked
    against. An allegation missing either is dropped here rather than carried into
    the gate, so the count the gate sees is already the evidenced one.

    `declared_file_checks` is parsed separately from `references` rather than
    trusting the judge to also transcribe a failing check into a `references`
    entry: asking for both the structured checklist and a free-form entry that
    has to agree with it is one more place for the two to drift, and the
    checklist alone already carries every field a finding needs.
    """
    data = response.data or {}
    conf = _confidence(data)
    findings: list[EnvironmentContextFinding] = []
    for entry in data.get("declared_file_checks", []):
        filename = str(entry.get("filename", "")).strip()
        quote = str(entry.get("prompt_quote", "")).strip()
        summary = str(entry.get("evidence_summary", "")).strip()
        if not filename or not quote or not summary:
            continue
        if not bool(entry.get("prompt_treats_as_available_to_read", False)):
            continue
        if bool(entry.get("conversation_shows_attachment_or_upload", False)):
            continue
        if bool(entry.get("conversation_shows_independent_content", False)):
            continue
        findings.append(
            EnvironmentContextFinding(
                reference=filename,
                kind="file",
                quote=quote,
                checked_against=[
                    "declared input file manifest",
                    "the model conversation(s) rendered to the judge",
                ],
                why_outside_universe=summary,
                presented_as_existing=True,
                required_to_complete=True,
                real_world_public_entity=False,
                basis="model",
                confidence=conf,
            )
        )
    for entry in data.get("references", []):
        reference = str(entry.get("reference", "")).strip()
        if not reference:
            continue
        kind = str(entry.get("kind", "other")).lower()
        findings.append(
            EnvironmentContextFinding(
                reference=reference,
                kind=kind if kind in _ENVIRONMENT_KINDS else "other",  # type: ignore[arg-type]
                quote=str(entry.get("quote", "")),
                checked_against=[
                    str(c) for c in (entry.get("checked_against") or []) if str(c).strip()
                ],
                why_outside_universe=str(entry.get("why_outside_universe", "")),
                presented_as_existing=bool(entry.get("presented_as_existing", True)),
                required_to_complete=bool(entry.get("required_to_complete", True)),
                real_world_public_entity=bool(
                    entry.get("real_world_public_entity", False)
                ),
                basis="model",
                confidence=conf,
            )
        )
    return findings


def parse_target_outcome(response: ModelResponse) -> list[TargetOutcomeFinding]:
    """Tab 2 grades this dimension on the entries the contributor wrote, never on
    whether the list is exhaustive, so the judge is only ever asked to classify
    those entries -- there is no separate "what the list leaves out" pass."""
    data = response.data or {}
    conf = _confidence(data)
    return [
        TargetOutcomeFinding(
            entry=str(e.get("entry", "")),
            classification=e.get("classification", "supported"),
            established_turn=e.get("established_turn"),
            reasoning=str(e.get("reasoning", "")),
            confidence=conf,
        )
        for e in data.get("entries", [])
    ]


def parse_artifacts(
    response: ModelResponse, sub, policy: Policy = DEFAULT_POLICY
) -> ArtifactFinding:
    """Extraction is the model's job; the set subtraction deciding the score is not.

    A delivery the judge marks unverifiable never reaches `claimed_files`, so it
    is absent from both the filename subtraction and the count shortfall. That is
    the whole abstention: the check can see that something was handed over inside
    a truncated span or behind a deliverable placeholder, but it cannot read the
    name, and a name it guesses at is a missing file it invented.
    """
    data = response.data or {}
    produced = [
        f
        for f in data.get("claimed_files", [])
        if f.get("produced") and str(f.get("filename", "")).strip()
    ]
    claimed = [
        str(f.get("filename", "")).strip()
        for f in produced
        if not f.get("unverifiable")
    ]
    unverifiable = [
        str(f.get("filename", "")).strip() for f in produced if f.get("unverifiable")
    ]
    why = "; ".join(
        str(f.get("why_unverifiable", "")).strip()
        for f in produced
        if f.get("unverifiable") and str(f.get("why_unverifiable", "")).strip()
    )
    # The manifest's own name for each upload when it recorded one. Before
    # this, `uploaded` held the opaque `.../WFx0N9ATP0DbySR` URL instead of
    # the real filename beside it in the same manifest entry, which
    # `_is_opaque` (correctly) flagged as unnamed on every attachment, every
    # task -- so this check could never take the `filename` branch below and
    # silently fell back to a raw count comparison for all of them.
    uploaded = [
        (sub.attachment_names.get(str(a)) or str(a)) for a in (sub.attachments or [])
    ]
    confidence = _confidence(data)

    if uploaded and all(_is_opaque(a) for a in uploaded):
        return ArtifactFinding(
            model=sub.model,
            claimed_files=claimed,
            uploaded_files=uploaded,
            basis="count",
            shortfall=max(0, len(claimed) - len(uploaded)),
            unverifiable_files=unverifiable,
            why_unverifiable=why,
            # Counting cannot see a swap: the right number of wrong files passes.
            confidence="low" if claimed else confidence,
        )

    uploaded_norm = [_norm_filename(a) for a in uploaded]
    uploaded_norm_set = set(uploaded_norm)
    missing: list[str] = []
    fuzzy_matched: list[tuple[str, str]] = []
    for c in claimed:
        c_norm = _norm_filename(c)
        if c_norm in uploaded_norm_set:
            continue
        best, ratio = _best_fuzzy_match(c_norm, uploaded_norm)
        if best is not None and ratio >= policy.filename_fuzzy_match_threshold:
            fuzzy_matched.append((c, best))
        else:
            missing.append(c)

    return ArtifactFinding(
        model=sub.model,
        claimed_files=claimed,
        uploaded_files=uploaded,
        missing_files=missing,
        basis="filename",
        fuzzy_matched_files=fuzzy_matched,
        unverifiable_files=unverifiable,
        why_unverifiable=why,
        confidence=confidence,
    )


def parse_key_turn_justification(
    response: ModelResponse,
) -> KeyTurnJustificationFinding:
    """All three questions are about the selected turn, so a judge that could not
    read that turn has answered none of them.

    Its three booleans are therefore reset to the no-defect reading and its
    allegations moved to `unverifiable_issues`, which `is_issue` does not consult.
    The gate then sees a clean finding and needed no change: 110 has no band below
    non_fail, so a mistake made from a marker in place of a turn has nowhere to go.
    """
    data = response.data or {}
    unverifiable = bool(data.get("unverifiable", False))
    issues = [str(i.get("issue", "")) for i in data.get("issues", []) if i.get("issue")]
    abstained = [
        str(i.get("issue", ""))
        for i in data.get("issues", [])
        if i.get("issue") and i.get("unverifiable")
    ]
    if unverifiable:
        abstained = issues
    return KeyTurnJustificationFinding(
        describes_selected_turn=(
            True if unverifiable else bool(data.get("describes_selected_turn", True))
        ),
        claims_are_accurate=(
            True if unverifiable else bool(data.get("claims_are_accurate", True))
        ),
        connects_to_core_value=(
            True if unverifiable else bool(data.get("connects_to_core_value", True))
        ),
        issues=[i for i in issues if i not in abstained],
        unverifiable=unverifiable,
        why_unverifiable=str(data.get("why_unverifiable", "")),
        unverifiable_issues=abstained,
        confidence=_confidence(data),
    )


def parse_autofails(response: ModelResponse) -> list[AutofailFinding]:
    data = response.data or {}
    conf = _confidence(data)
    return [
        AutofailFinding(
            criterion_id=str(a.get("criterion_id", "")),
            quote=str(a.get("quote", "")),
            why_outcome_is_useless=str(a.get("why_outcome_is_useless", "")),
            confidence=conf,
        )
        for a in data.get("autofail_criteria", [])
        if str(a.get("criterion_id", "")).strip()
    ]


def parse_relevant_turns(
    response: ModelResponse, check_id, model, last_turn: int | None = None
) -> list[RelevantTurnFinding]:
    """Route every turn the audit could not open out of the counted list.

    Two kinds arrive. A citation past the end of the fetched conversation is
    detected here from `last_turn` and needs no cooperation from the judge. A
    citation into a span the render budget cut, or into a deliverable whose content
    the transcript does not hold, is invisible to Python -- the turn number is in
    range and the text simply is not there -- so the judge reports it in
    `unverifiable_turns` and it is honoured even where it also appears among the
    incorrect ones. Both leave `incorrect_turns` before the gate sees the finding,
    which is why 280 and 310 needed no change to stop counting them.
    """
    data = response.data or {}
    conf = _confidence(data)

    findings: list[RelevantTurnFinding] = []
    for j in data.get("judgments", []):
        flagged = [int(t) for t in j.get("incorrect_turns", [])]
        declared = [int(t) for t in j.get("unverifiable_turns", [])]
        beyond = (
            [t for t in flagged if t > last_turn] if last_turn is not None else []
        )
        unverifiable = list(dict.fromkeys(beyond + declared))
        findings.append(
            RelevantTurnFinding(
                check_id=check_id,
                item_id=str(j.get("item_id", "")),
                model=model,
                incorrect_turns=[t for t in flagged if t not in unverifiable],
                unverifiable_turns=unverifiable,
                why_unverifiable=str(j.get("why_unverifiable", "")),
                confidence=conf,
            )
        )
    return findings


def _last_turn(sub) -> int | None:
    turns = getattr(sub, "conversation", None) or []
    return max((t.index for t in turns), default=None)


def missing_turn_findings(task: Task, check_id) -> list[RelevantTurnFinding]:
    """Score-0 criteria and dimension ratings that cite no turn at all.

    The spec counts incorrectly selected turns and is silent on absent ones, so
    these are recorded under their own category and counted only when policy says
    to. No model call is needed to see that a list is empty.
    """
    out: list[RelevantTurnFinding] = []
    if check_id == 280:
        for r in task.criterion_ratings:
            if r.score == 0 and not r.relevant_turns:
                out.append(
                    RelevantTurnFinding(
                        check_id=280,
                        item_id=r.criterion_id,
                        model=r.model,
                        missing_turn=True,
                    )
                )
    else:
        for d in task.dimension_ratings:
            if not d.not_applicable and not d.relevant_turns:
                out.append(
                    RelevantTurnFinding(
                        check_id=310,
                        item_id=d.dimension,
                        model=d.model,
                        missing_turn=True,
                    )
                )
    return out


def parse_justifications(response: ModelResponse) -> list[JustificationFinding]:
    """Build one finding per justification, zeroing the counts of any the judge
    could not read.

    A justification defending a rating of a deliverable the transcript omits, or of
    a stretch of conversation the render budget cut, has no honest count against
    any of 450's conditions. Zeroing them here rather than trusting the model to
    leave them empty is the same discipline `parse_relevant_turns` applies: the
    abstention flag is the claim, and everything it contradicts is discarded. The
    stage then keeps these findings out of the list the gate scores, so they leave
    450's denominator as well.
    """
    data = response.data or {}
    conf = _confidence(data)
    findings: list[JustificationFinding] = []
    for j in data.get("justifications", []):
        if not str(j.get("item_id", "")).strip():
            continue
        unverifiable = bool(j.get("unverifiable", False))
        findings.append(
            JustificationFinding(
                item_id=str(j.get("item_id", "")),
                rated_value=j.get("rated_value"),
                contradicts_verdict_claims=(
                    0 if unverifiable else int(j.get("contradicts_verdict_claims", 0))
                ),
                is_generic=(
                    False if unverifiable else bool(j.get("is_generic", False))
                ),
                is_skewed=False if unverifiable else bool(j.get("is_skewed", False)),
                inaccurate_primary_claims=(
                    0 if unverifiable else int(j.get("inaccurate_primary_claims", 0))
                ),
                inaccurate_secondary_claims=(
                    0 if unverifiable else int(j.get("inaccurate_secondary_claims", 0))
                ),
                unsupported_claims=(
                    0 if unverifiable else int(j.get("unsupported_claims", 0))
                ),
                inaccurate_evidence=(
                    0 if unverifiable else int(j.get("inaccurate_evidence", 0))
                ),
                misconstrued_evidence=(
                    0 if unverifiable else int(j.get("misconstrued_evidence", 0))
                ),
                unverifiable_claims=int(j.get("unverifiable_claims", 0)),
                unverifiable=unverifiable,
                why_unverifiable=str(j.get("why_unverifiable", "")),
                specifics_quoted=[
                    str(s) for s in (j.get("specifics_quoted") or []) if str(s).strip()
                ],
                contradicting_quotes=(
                    []
                    if unverifiable
                    else [
                        str(s)
                        for s in (j.get("contradicting_quotes") or [])
                        if str(s).strip()
                    ]
                ),
                confidence=conf,
            )
        )
    return findings


def split_unverifiable_justifications(
    samples: list[list[JustificationFinding]],
) -> tuple[list[list[JustificationFinding]], list[JustificationFinding]]:
    """Take abstained justifications out of the draws before they are aggregated.

    An abstention is not a judgment with a value to combine. Aggregation votes on
    counts and conditions, so a justification left in would be merged into a clean
    finding and then reported as audited -- which is the claim the abstention
    exists to deny. It is unioned rather than voted, the same way
    `unverifiable_turns` is: every draw is shown the identical rendered
    conversation, so a draw saying the deliverable is absent is not outvoted by two
    draws that judged it anyway.
    """
    abstained: dict[str, JustificationFinding] = {}
    for findings in samples:
        for f in findings:
            if f.unverifiable and f.item_id not in abstained:
                abstained[f.item_id] = f
    kept = [[f for f in s if f.item_id not in abstained] for s in samples]
    return kept, [abstained[k] for k in abstained]


def apply_key_turn_abstention(
    finding: KeyTurnJustificationFinding | None,
    samples: list[KeyTurnJustificationFinding],
) -> KeyTurnJustificationFinding | None:
    """Carry 110's abstention across aggregation.

    Aggregation majority-votes the three booleans and the existence of an issue,
    and has no field for the abstention, so the reason the judge gave for not being
    able to answer would be lost in the merge and a draw that abstained would be
    outvoted by two that guessed. Every draw reads the same rendered conversation:
    if any of them says the selected turn was not in it, that is a fact about the
    render rather than a judgment, and this check -- one flag, no fail band -- is
    the last place to resolve it against the contributor.
    """
    if finding is None:
        return None
    abstaining = [s for s in samples if s.unverifiable]
    if not abstaining:
        return finding
    finding.unverifiable = True
    finding.why_unverifiable = next(
        (s.why_unverifiable for s in abstaining if s.why_unverifiable.strip()), ""
    )
    finding.unverifiable_issues = list(
        dict.fromkeys(
            [i for s in samples for i in s.unverifiable_issues] + list(finding.issues)
        )
    )
    finding.issues = []
    finding.describes_selected_turn = True
    finding.claims_are_accurate = True
    finding.connects_to_core_value = True
    return finding


def carry_unverifiable_reasons(
    merged: list[RelevantTurnFinding], samples: list[list[RelevantTurnFinding]]
) -> list[RelevantTurnFinding]:
    """Re-attach what the judge said it could not open.

    Aggregation votes on turn numbers and unions the unverifiable ones, but the
    free-text reason has no field to survive the merge, and an abstention nobody
    can read the reason for is indistinguishable from silence in the report.
    """
    reasons: dict[str, str] = {}
    for findings in samples:
        for f in findings:
            if f.why_unverifiable.strip():
                reasons.setdefault(f.item_id, f.why_unverifiable)
    for f in merged:
        if not f.why_unverifiable.strip():
            f.why_unverifiable = reasons.get(f.item_id, "")
    return merged


def empty_justification_findings(task: Task) -> list[JustificationFinding]:
    """A dimension rated without any justification at all.

    An absent justification cannot be sent to a model, but it is the strongest
    form of the generic condition: no "why" whatsoever.
    """
    return [
        JustificationFinding(
            item_id=f"{d.model}::{d.dimension}",
            rated_value=d.rating,
            is_generic=True,
        )
        for d in task.dimension_ratings
        if not d.not_applicable and not (d.justification or "").strip()
    ]


def parse_verdict(response: ModelResponse) -> VerdictFinding:
    data = response.data or {}
    return VerdictFinding(
        states_preference=bool(data.get("states_preference", False)),
        quote=str(data.get("quote", "")),
        confidence=_confidence(data),
    )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def run_informed_stage(
    task: Task,
    client: ModelClient,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 6,
) -> InformedStageResult:
    """Run the informed pass.

    Unlike `run_rating_stage`, this always spends every one of its calls: several
    of its gates (70, 75) decide `not_evaluated` from the *task's own* data
    (an absent domain or pre-seed) rather than from whether a finding was
    produced, so silently withholding a request here -- the way `checks` does
    for the rating stage's 300 -- would hand `None` to a gate that still expects
    to see a real finding whenever that data is present, which is a crash
    waiting on the first task where it is. A caller wanting a restricted report
    (e.g. only 280/310/450/470) filters `result.verdicts` afterwards instead.
    """
    result = InformedStageResult(task_id=task.task_id)
    subs = {s.model: s for s in task.submissions()}

    requests = build_informed_requests(task, policy)
    # One flat batch of every draw of every judgment, so the global throttle still
    # sees the true number of calls in flight and the worker pool stays full;
    # `fanout.groups` puts the draws back together by judgment afterwards.
    fanout = expand_requests(requests, policy)
    responses = run_requests(client, fanout.requests, workers=workers)
    result.calls = len(responses)
    result.cost_usd = sum(r.cost_usd for r in responses)
    result.cached_calls = sum(1 for r in responses if r.cached)

    # Paired positionally: an error response need not echo metadata back. A draw
    # that failed is dropped rather than counted as agreement or disagreement, so
    # a timeout costs resolution and never invents a verdict.
    for index, request in enumerate(requests):
        meta = request.metadata
        check_id = meta.get("check_id")

        drawn: list[ModelResponse] = []
        for response in fanout.responses_for(index, responses):
            if not response.ok or response.data is None:
                result.errors.append(
                    f"{response.key or request.key}: {response.error or 'no data'}"
                )
                continue
            drawn.append(response)
        if not drawn:
            continue
        first = drawn[0]

        if check_id == 70:
            result.domain_relevance = parse_domain_relevance(first, task)
        elif check_id == 75:
            result.prompt_consistency = parse_prompt_consistency(first)
        elif check_id == 85:
            result.environment_context += parse_environment_context(first)
            result.entity_half_ran = True
        elif check_id == 80:
            result.target_outcome = parse_target_outcome(first)
        elif check_id == 95:
            result.artifacts.append(parse_artifacts(first, subs[meta["model"]], policy))
        elif check_id == 110:
            parsed = [parse_key_turn_justification(r) for r in drawn]
            finding, records = aggregate_key_turn_justification(parsed, policy)
            result.key_turn_justification = apply_key_turn_abstention(finding, parsed)
            result.agreement.extend(records, policy)
        elif check_id == 220:
            result.autofails = parse_autofails(first)
        elif check_id in (280, 310):
            last = _last_turn(subs[meta["model"]])
            parsed_turns = [
                parse_relevant_turns(r, check_id, meta["model"], last) for r in drawn
            ]
            findings, records = aggregate_relevant_turns(
                parsed_turns, check_id, policy
            )
            result.relevant_turns += carry_unverifiable_reasons(findings, parsed_turns)
            result.agreement.extend(records, policy)
        elif check_id == 450:
            kept, abstained = split_unverifiable_justifications(
                [parse_justifications(r) for r in drawn]
            )
            findings, records = aggregate_justifications(kept, policy)
            result.justifications += findings
            result.unverifiable_justifications += abstained
            result.agreement.extend(records, policy)
        elif check_id == 470:
            result.verdict = parse_verdict(first)

    result.relevant_turns += missing_turn_findings(task, 280)
    result.relevant_turns += missing_turn_findings(task, 310)
    result.justifications += empty_justification_findings(task)

    # Check 85's file half. No call, so it runs whether or not the entity half
    # was asked for, and its findings join the model's in one dimension. Its
    # gate (`_build_informed_verdicts` below) always publishes a verdict for it
    # regardless of `checks` -- restricting the report to a check subset is the
    # caller's job, same as every other check `checks` excludes from spending a
    # call but that still gets a (`not_evaluated`) verdict here.
    result.file_scan = scan_file_references(task)
    result.environment_context += result.file_scan.findings

    _confirm_autofails(task, client, result, workers)
    _confirm_missing_artifacts(task, client, result, policy, workers)
    result.verdicts = _build_informed_verdicts(task, result, policy)
    return result


def _confirm_missing_artifacts(
    task: Task,
    client: ModelClient,
    result: InformedStageResult,
    policy: Policy = DEFAULT_POLICY,
    workers: int = 6,
) -> None:
    """Check 95's second opinion, run only on the names already read as absent.

    Same shape as `_confirm_autofails` and for the same reason: 95 has no middle
    band, so a name the subtraction gets wrong is a task-level fail with nothing to
    absorb it. On a manual audit of nine of those fails, five were the comparison
    rather than the contributor -- screenshots whose capture-tool names could not be
    matched to the contributor's own, and inline code blocks read as undelivered
    files. None of those is visible to string distance, and all of them are obvious
    to a reader with the conversation in front of them.

    A submission whose claimed files all matched costs nothing here. The call
    happens per submission, not per file, so the worst case is two.
    """
    if not policy.artifact_miss_requires_llm_confirmation:
        return

    subs = {s.model: s for s in task.submissions()}
    targets = [
        f
        for f in result.artifacts
        if f.basis == "filename" and f.missing_files and f.model in subs
    ]
    if not targets:
        return

    requests = [
        ModelRequest(
            key=f"{task.task_id}::artifact_confirm::{f.model}",
            prompt=build_artifact_confirmation_prompt(
                task, subs[f.model], f.missing_files, f.uploaded_files, policy
            ),
            schema=ARTIFACT_CONFIRMATION_SCHEMA,
            system=INFORMED_SYSTEM_PROMPT,
            metadata={"check_id": 95, "model": f.model},
        )
        for f in targets
    ]
    responses = run_requests(client, requests, workers=workers)
    result.calls += len(responses)
    result.cost_usd += sum(r.cost_usd for r in responses)
    result.cached_calls += sum(1 for r in responses if r.cached)

    for finding, response in zip(targets, responses):
        if not response.ok or response.data is None:
            # A failed confirmation leaves the subtraction's answer standing: this
            # pass exists to clear false positives, and a call that never happened
            # is not evidence either way. The finding records that it did not run.
            result.errors.append(
                f"artifact confirmation for {finding.model}: "
                f"{response.error or 'no data'}"
            )
            continue
        apply_artifact_confirmation(finding, response.data)


def apply_artifact_confirmation(finding: ArtifactFinding, data: dict) -> None:
    """Record one confirming pass's per-file assessments on the finding.

    Only names the subtraction actually flagged are accepted, so a judge that
    volunteers an opinion about some other file cannot add to or subtract from the
    count. An unreadable assessment is left out entirely, which leaves that name
    counted -- the conservative direction, and the same one a failed call takes.
    """
    finding.confirmation_ran = True
    allowed = {name.strip().lower(): name for name in finding.missing_files}
    for entry in data.get("files", []):
        if not isinstance(entry, dict):
            continue
        name = allowed.get(str(entry.get("claimed_filename", "")).strip().lower())
        if name is None:
            continue
        assessment = str(entry.get("assessment", "")).strip()
        if assessment not in ("missing", "renamed", "not_a_file", "unverifiable"):
            continue
        finding.confirmations[name] = assessment
        reason = " ".join(str(entry.get("why", "")).split())[:300]
        matched = " ".join(str(entry.get("matched_upload", "")).split())
        if assessment == "renamed" and matched:
            reason = f"matched upload {matched}: {reason}" if reason else f"matched upload {matched}"
        if reason:
            finding.confirmation_reasons[name] = reason


def _confirm_autofails(
    task: Task, client: ModelClient, result: InformedStageResult, workers: int
) -> None:
    """Mandatory second opinion, run only when something was alleged."""
    if not result.autofails:
        return

    requests = [
        ModelRequest(
            key=f"{task.task_id}::autofail_confirm::{f.criterion_id}",
            prompt=build_autofail_second_opinion_prompt(
                task, f.criterion_id, f.quote, f.why_outcome_is_useless
            ),
            schema=AUTOFAIL_SCHEMA,
            system=INFORMED_SYSTEM_PROMPT,
            metadata={"check_id": 220, "criterion_id": f.criterion_id},
        )
        for f in result.autofails
    ]
    responses = run_requests(client, requests, workers=workers)
    result.calls += len(responses)
    result.cost_usd += sum(r.cost_usd for r in responses)
    result.cached_calls += sum(1 for r in responses if r.cached)

    for finding, response in zip(result.autofails, responses):
        if not response.ok or response.data is None:
            result.errors.append(
                f"autofail second opinion for {finding.criterion_id}: "
                f"{response.error or 'no data'}"
            )
            continue
        confirmed_ids = {
            str(a.get("criterion_id", ""))
            for a in (response.data or {}).get("autofail_criteria", [])
        }
        finding.confirmed = finding.criterion_id in confirmed_ids


def _build_informed_verdicts(
    task: Task, result: InformedStageResult, policy: Policy
) -> list[CheckVerdict]:
    verdicts = [
        # 70's own gate decides not_evaluated from the absent domain, so it is
        # called unconditionally rather than guarded here.
        evaluate_check_70(task, result.domain_relevance, policy),
        # 75's own gate decides not_evaluated from the absent pre-seed, the same
        # way 70's does.
        evaluate_check_75(task, result.prompt_consistency, policy),
        evaluate_check_80(task, result.target_outcome, policy)
        if result.target_outcome
        else _not_evaluated(task, 80, "no target outcome list was recorded", policy),
        # 85's own gate decides not_evaluated from which of its two halves ran.
        evaluate_check_85(
            task,
            result.environment_context,
            result.file_scan or scan_file_references(task),
            result.entity_half_ran,
            policy,
        ),
        evaluate_check_95(task, result.artifacts, policy)
        if result.artifacts
        else _not_evaluated(task, 95, "no model submissions to scan", policy),
        evaluate_check_110(task, result.key_turn_justification, policy),
        evaluate_check_220(task, result.autofails, policy)
        if task.rubric
        else _not_evaluated(task, 220, "rubric is empty", policy),
        evaluate_relevant_turns(task, 280, result.relevant_turns, policy)
        if any(r.score == 0 for r in task.criterion_ratings)
        else _not_evaluated(
            task, 280, "no criterion was scored 0, so no turns were required", policy
        ),
        evaluate_relevant_turns(task, 310, result.relevant_turns, policy)
        if task.dimension_ratings
        else _not_evaluated(task, 310, "no dimension ratings were recorded", policy),
        evaluate_check_450(task, result.justifications, policy)
        if result.justifications
        else _not_evaluated(
            task,
            450,
            "every justification the contributor wrote defends something this "
            "transcript does not contain, so none could be audited"
            if result.unverifiable_justifications
            else "no justifications were recorded",
            policy,
        ),
        evaluate_check_470(task, result.verdict, policy),
    ]
    return verdicts


def _not_evaluated(
    task: Task, check_id: int, reason: str, policy: Policy
) -> CheckVerdict:
    return build_verdict(
        task_id=task.task_id,
        check_id=check_id,
        band="not_evaluated",
        measurement=Measurement(notes=reason),
        confidence="low",
        policy=policy,
    )


def not_evaluated_informed_verdicts(
    task: Task, reason: str, policy: Policy = DEFAULT_POLICY
) -> list[CheckVerdict]:
    return [_not_evaluated(task, cid, reason, policy) for cid in INFORMED_CHECKS]


def informed_calls_by_check(task: Task) -> dict[int, int]:
    """Logical judgments this task needs, per check, before sampling.

    Split out from the total so a cost projection can price the sampled checks at
    their own sample count: three draws of 280 and one of everything else is a
    very different bill from three draws of the stage.

    Excludes the two confirming passes -- 220's second opinion and 95's
    missing-file confirmation -- because both fire only where a fail is already on
    the table, which is the exception. That is a different case from the rating
    stage's adjudication round, which fires on any rating gap at all and so is
    priced into `estimate_adjudication_calls`: a cost most tasks pay belongs in the
    estimate, one a few tasks pay does not.
    """
    counts: dict[int, int] = {}

    def add(check_id: int, n: int = 1) -> None:
        if n:
            counts[check_id] = counts.get(check_id, 0) + n

    add(70, 1 if (task.assigned_domain or "").strip() else 0)
    add(75, 1 if (task.pre_seeded_prompt or "").strip() else 0)
    add(80, 1 if task.target_outcome else 0)
    # 85's file half is free; only its entity half is a call, and only when there
    # is something to check a reference against.
    add(85, 1 if can_judge_environment_context(task) else 0)
    add(95, len(task.submissions()))
    add(110, 1 if (task.key_turn.justification or "").strip() else 0)
    add(220, 1 if task.rubric else 0)
    for sub in task.submissions():
        if any(r.model == sub.model and r.score == 0 for r in task.criterion_ratings):
            add(280)
        if any(d.model == sub.model for d in task.dimension_ratings):
            add(310)
            add(450)
    if (task.sxs.justification or "").strip():
        add(450)
        add(470)
    return counts


# Which check each informed-stage request key belongs to. The key names are what
# the response cache stores, so this is what lets a cost projection price a
# configuration off the real per-call spend of a completed run instead of a guess.
CALL_NAME_TO_CHECK = {
    "domain_relevance": 70,
    "prompt_consistency": 75,
    "target_outcome": 80,
    "environment_context": 85,
    "artifacts": 95,
    "key_turn_justification": 110,
    "autofail": 220,
    "criterion_turns": 280,
    "dimension_turns": 310,
    "dimension_justifications": 450,
    "ranking_justification": 450,
    "verdict": 470,
}


def estimate_informed_calls(task: Task, policy: Policy | None = None) -> int:
    """Model calls this task's informed stage will make.

    With a policy supplied the sampled checks are priced at their configured
    sample count, which is what a dry run has to show: the whole point of making
    the count per check is that the bill is no longer the judgment count.
    """
    counts = informed_calls_by_check(task)
    if policy is None:
        return sum(counts.values())
    return sum(n * samples_for(check_id, policy) for check_id, n in counts.items())
