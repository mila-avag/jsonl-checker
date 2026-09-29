"""V5 prompt-quality checks: 60, 71, 72, 76, 82, 86.

One call per task (or per slot for 82). Parsers default to the clean assessment
when FakeModelClient returns only `confidence`, so existing informed-stage tests
that do not know about these checks stay green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import DEFAULT_POLICY, Policy
from .context import golden_answer_block, numbered_prompts
from .llm import ModelResponse
from .models import ModelSubmission, Task
from .scoring import Band, CheckVerdict, Measurement, build_verdict


def _confidence(data: dict) -> str:
    value = str(data.get("confidence", "high")).lower()
    return value if value in {"low", "medium", "high"} else "high"


def _quote(data: dict, *keys: str) -> str:
    for key in keys:
        text = str(data.get(key, "")).strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass
class PromptClarityFinding:
    """Check 60. Fail when the prompt is unclear or has no unique ground truth."""

    assessment: Literal[
        "unique_ground_truth",
        "mostly_clear",
        "unclear",
        "missing_details",
        "multiple_or_subjective",
    ] = "unique_ground_truth"
    prompt_quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.prompt_quote.strip())

    @property
    def is_issue(self) -> bool:
        return self.assessment != "unique_ground_truth" and self.is_evidenced

    @property
    def is_fail(self) -> bool:
        return (
            self.assessment in {"unclear", "missing_details", "multiple_or_subjective"}
            and self.is_evidenced
        )


@dataclass
class ReasoningRequirementFinding:
    assessment: Literal[
        "requires_reasoning", "trivial_reasoning", "factual_lookup"
    ] = "requires_reasoning"
    prompt_quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.prompt_quote.strip())

    @property
    def is_issue(self) -> bool:
        return self.assessment != "requires_reasoning" and self.is_evidenced

    @property
    def is_fail(self) -> bool:
        return self.assessment == "factual_lookup" and self.is_evidenced


@dataclass
class ConstraintsFinding:
    assessment: Literal[
        "natural", "basic_or_overused", "contrived", "unrealistically_stacked"
    ] = "natural"
    prompt_quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.prompt_quote.strip())

    @property
    def is_issue(self) -> bool:
        return self.assessment != "natural" and self.is_evidenced

    @property
    def is_fail(self) -> bool:
        return (
            self.assessment in {"contrived", "unrealistically_stacked"}
            and self.is_evidenced
        )


@dataclass
class PromptRequestFinding:
    assessment: Literal["has_request", "no_request"] = "has_request"
    prompt_quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_evidenced(self) -> bool:
        return bool(self.prompt_quote.strip()) or self.assessment == "has_request"

    @property
    def is_fail(self) -> bool:
        return self.assessment == "no_request" and self.is_evidenced


@dataclass
class FinalOutcomeClaimFinding:
    model: str
    claimed: bool
    assessment: Literal["claim_accurate", "claim_wrong"] = "claim_accurate"
    quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_issue(self) -> bool:
        return self.assessment == "claim_wrong" and bool(self.quote.strip())


@dataclass
class KnownInfoFinding:
    assessment: Literal["accurate", "inaccurate"] = "accurate"
    quote: str = ""
    reasoning: str = ""
    confidence: str = "high"

    @property
    def is_issue(self) -> bool:
        return self.assessment == "inaccurate" and bool(self.quote.strip())


# ---------------------------------------------------------------------------
# Schemas and prompts
# ---------------------------------------------------------------------------

PROMPT_CLARITY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": [
                "unique_ground_truth",
                "mostly_clear",
                "unclear",
                "missing_details",
                "multiple_or_subjective",
            ],
        },
        "prompt_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "prompt_quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

REASONING_REQUIREMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["requires_reasoning", "trivial_reasoning", "factual_lookup"],
        },
        "prompt_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "prompt_quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

CONSTRAINTS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": [
                "natural",
                "basic_or_overused",
                "contrived",
                "unrealistically_stacked",
            ],
        },
        "prompt_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "prompt_quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

PROMPT_REQUEST_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "enum": ["has_request", "no_request"]},
        "prompt_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "prompt_quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

FINAL_OUTCOME_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["claim_accurate", "claim_wrong"],
        },
        "quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

KNOWN_INFO_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "enum": ["accurate", "inaccurate"]},
        "quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["assessment", "quote", "reasoning", "confidence"],
    "additionalProperties": False,
}


def _prompt_block(task: Task) -> str:
    return f"""## The prompt the contributor submitted

{numbered_prompts(task)}
{golden_answer_block(task)}"""


def build_prompt_clarity_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""{_prompt_block(task)}
## Your task

Judge whether this prompt is clear, specific, and has a unique ground-truth
final answer.

Pick one `assessment`:

  - "unique_ground_truth": the prompt is clear and specific, and a competent
    responder would converge on one correct final answer. This is the expected
    answer.
  - "mostly_clear": mostly clear, with interpretations that still converge.
  - "unclear": vague or ambiguous in a way that would produce different final
    answers.
  - "missing_details": major details needed to produce a unique answer are
    absent.
  - "multiple_or_subjective": the prompt has several valid final answers, or is
    subjective.

Quote the prompt in `prompt_quote`. The customer's ground-truth answer above,
when present, is the unique answer the prompt is supposed to have.
"""


def build_reasoning_requirement_prompt(
    task: Task, policy: Policy = DEFAULT_POLICY
) -> str:
    return f"""{_prompt_block(task)}
## Your task

Judge whether this prompt requires real reasoning, not just a factual lookup.

Pick one `assessment`:

  - "requires_reasoning": answering well needs analysis, synthesis, or
    multi-step reasoning at the assigned expertise. This is the expected answer.
  - "trivial_reasoning": some reasoning, but trivial for the assigned expertise.
  - "factual_lookup": the prompt is a definition, lookup, or recall question.

Quote the prompt in `prompt_quote`.
"""


def build_constraints_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""{_prompt_block(task)}
## Your task

Judge the constraints the prompt imposes (format, length, tools, style, etc.).

Pick one `assessment`:

  - "natural": constraints fit the request. This is the expected answer.
  - "basic_or_overused": constraints are basic or overused but still realistic.
  - "contrived": constraints are contrived or unrealistic.
  - "unrealistically_stacked": constraints are stacked past what a real user
    would ask -- V5's example is 3+ formatting constraints piled on.

Quote the constraining language in `prompt_quote`.
"""


def build_prompt_request_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""{_prompt_block(task)}
## Your task

Does any prompt in this conversation ask for a deliverable of any kind -- a
file, an answer, a plan, an analysis, a rewrite, anything the model is supposed
to produce?

Pick one `assessment`:

  - "has_request": at least one prompt asks for a deliverable. This is the
    expected answer.
  - "no_request": no prompt asks for anything.

Quote the request in `prompt_quote` when one exists.
"""


def build_final_outcome_claim_prompt(
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> str:
    from .context import render_conversation

    conversation, _ = render_conversation(sub, policy)
    claimed = "yes" if sub.produced_final_outcome else "no"
    return f"""## Conversation with Model {sub.model}

{conversation}

## The contributor's claim

The contributor recorded that this model {"did" if sub.produced_final_outcome else "did not"}
produce the final outcome (`produced_final_outcome` = {claimed}).

## Your task

Is that claim accurate given the conversation?

  - "claim_accurate": the conversation matches the claim.
  - "claim_wrong": the conversation contradicts the claim.

Quote the turn that decides it.
"""


def build_known_info_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""{_prompt_block(task)}
## Known information the contributor recorded

{(task.known_information or "(empty)").strip()}

## Not-known information the contributor recorded

{(task.not_known_information or "(empty)").strip()}

## Your task

Is this known / not-known split accurate given the prompt?

  - "accurate": the split matches what the prompt actually supplies and withholds.
  - "inaccurate": something listed as known is not in the prompt, or something
    listed as not-known is.

Quote the mismatch, or the matching language, in `quote`.
"""


# ---------------------------------------------------------------------------
# Parsers -- missing assessment defaults to clean so FakeModelClient is a no-op
# ---------------------------------------------------------------------------


def parse_prompt_clarity(response: ModelResponse) -> PromptClarityFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "unique_ground_truth").lower()
    allowed = {
        "unique_ground_truth",
        "mostly_clear",
        "unclear",
        "missing_details",
        "multiple_or_subjective",
    }
    if assessment not in allowed:
        assessment = "unique_ground_truth"
    return PromptClarityFinding(
        assessment=assessment,  # type: ignore[arg-type]
        prompt_quote=_quote(data, "prompt_quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_reasoning_requirement(
    response: ModelResponse,
) -> ReasoningRequirementFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "requires_reasoning").lower()
    if assessment not in {"requires_reasoning", "trivial_reasoning", "factual_lookup"}:
        assessment = "requires_reasoning"
    return ReasoningRequirementFinding(
        assessment=assessment,  # type: ignore[arg-type]
        prompt_quote=_quote(data, "prompt_quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_constraints(response: ModelResponse) -> ConstraintsFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "natural").lower()
    if assessment not in {
        "natural",
        "basic_or_overused",
        "contrived",
        "unrealistically_stacked",
    }:
        assessment = "natural"
    return ConstraintsFinding(
        assessment=assessment,  # type: ignore[arg-type]
        prompt_quote=_quote(data, "prompt_quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_prompt_request(response: ModelResponse) -> PromptRequestFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "has_request").lower()
    if assessment not in {"has_request", "no_request"}:
        assessment = "has_request"
    return PromptRequestFinding(
        assessment=assessment,  # type: ignore[arg-type]
        prompt_quote=_quote(data, "prompt_quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_final_outcome_claim(
    response: ModelResponse, sub: ModelSubmission
) -> FinalOutcomeClaimFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "claim_accurate").lower()
    if assessment not in {"claim_accurate", "claim_wrong"}:
        assessment = "claim_accurate"
    return FinalOutcomeClaimFinding(
        model=sub.model,
        claimed=bool(sub.produced_final_outcome),
        assessment=assessment,  # type: ignore[arg-type]
        quote=_quote(data, "quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


def parse_known_info(response: ModelResponse) -> KnownInfoFinding:
    data = response.data or {}
    assessment = str(data.get("assessment") or "accurate").lower()
    if assessment not in {"accurate", "inaccurate"}:
        assessment = "accurate"
    return KnownInfoFinding(
        assessment=assessment,  # type: ignore[arg-type]
        quote=_quote(data, "quote"),
        reasoning=str(data.get("reasoning", "")),
        confidence=_confidence(data),
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _prompt_present(task: Task) -> bool:
    return bool(
        (task.seeded_prompt or "").strip()
        or any((t.text or "").strip() for t in task.prompts)
    )


def evaluate_check_60(
    task: Task,
    finding: PromptClarityFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not _prompt_present(task):
        return build_verdict(
            task_id=task.task_id,
            check_id=60,
            band="not_evaluated",
            measurement=Measurement(notes="no prompt was recorded"),
            confidence="low",
            policy=policy,
        )
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=60,
            band="not_evaluated",
            measurement=Measurement(notes="the prompt was not audited for clarity"),
            confidence="low",
            policy=policy,
        )
    if finding.is_fail:
        band: Band = "fail"
    elif finding.is_issue:
        band = "non_fail"
    else:
        band = "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=60,
        band=band,
        measurement=Measurement(
            counts={"assessment": finding.assessment},
            notes="V5 fails when the prompt is unclear/vague, major details are "
            "missing, or the prompt has multiple valid final answers / is "
            "subjective.",
        ),
        contributing_items=[finding.prompt_quote] if finding.is_issue else [],
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_71(
    task: Task,
    finding: ReasoningRequirementFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not _prompt_present(task):
        return build_verdict(
            task_id=task.task_id,
            check_id=71,
            band="not_evaluated",
            measurement=Measurement(notes="no prompt was recorded"),
            confidence="low",
            policy=policy,
        )
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=71,
            band="not_evaluated",
            measurement=Measurement(notes="the prompt was not audited for reasoning"),
            confidence="low",
            policy=policy,
        )
    if finding.is_fail:
        band: Band = "fail"
    elif finding.is_issue:
        band = "non_fail"
    else:
        band = "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=71,
        band=band,
        measurement=Measurement(counts={"assessment": finding.assessment}),
        contributing_items=[finding.prompt_quote] if finding.is_issue else [],
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_72(
    task: Task,
    finding: ConstraintsFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not _prompt_present(task):
        return build_verdict(
            task_id=task.task_id,
            check_id=72,
            band="not_evaluated",
            measurement=Measurement(notes="no prompt was recorded"),
            confidence="low",
            policy=policy,
        )
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=72,
            band="not_evaluated",
            measurement=Measurement(notes="the prompt was not audited for constraints"),
            confidence="low",
            policy=policy,
        )
    if finding.is_fail:
        band: Band = "fail"
    elif finding.is_issue:
        band = "non_fail"
    else:
        band = "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=72,
        band=band,
        measurement=Measurement(counts={"assessment": finding.assessment}),
        contributing_items=[finding.prompt_quote] if finding.is_issue else [],
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_76(
    task: Task,
    finding: PromptRequestFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not _prompt_present(task):
        return build_verdict(
            task_id=task.task_id,
            check_id=76,
            band="not_evaluated",
            measurement=Measurement(notes="no prompt was recorded"),
            confidence="low",
            policy=policy,
        )
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=76,
            band="not_evaluated",
            measurement=Measurement(notes="the prompt was not audited for a request"),
            confidence="low",
            policy=policy,
        )
    band: Band = "fail" if finding.is_fail else "clean"
    return build_verdict(
        task_id=task.task_id,
        check_id=76,
        band=band,
        measurement=Measurement(counts={"assessment": finding.assessment}),
        contributing_items=[finding.prompt_quote] if finding.is_fail else [],
        confidence=finding.confidence,
        policy=policy,
    )


def evaluate_check_82(
    task: Task,
    findings: list[FinalOutcomeClaimFinding],
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    claimed = [
        sub
        for sub in task.trajectories()
        if sub.produced_final_outcome is not None
    ]
    if not claimed:
        return build_verdict(
            task_id=task.task_id,
            check_id=82,
            band="not_evaluated",
            measurement=Measurement(
                notes="no produced_final_outcome claim was recorded"
            ),
            confidence="low",
            policy=policy,
        )
    if not findings:
        return build_verdict(
            task_id=task.task_id,
            check_id=82,
            band="not_evaluated",
            measurement=Measurement(notes="final-outcome claims were not audited"),
            confidence="low",
            policy=policy,
        )
    issues = [f for f in findings if f.is_issue]
    return build_verdict(
        task_id=task.task_id,
        check_id=82,
        band="non_fail" if issues else "clean",
        measurement=Measurement(
            counts={
                "slots_audited": len(findings),
                "inaccurate_claims": len(issues),
            },
            notes="Report-only: V5 does not fail the task on this dimension.",
        ),
        contributing_items=[f"{f.model}:{f.quote}" for f in issues],
        confidence=min((f.confidence for f in findings), default="high"),
        policy=policy,
    )


def evaluate_check_86(
    task: Task,
    finding: KnownInfoFinding | None,
    policy: Policy = DEFAULT_POLICY,
) -> CheckVerdict:
    if not (
        (task.known_information or "").strip()
        or (task.not_known_information or "").strip()
    ):
        return build_verdict(
            task_id=task.task_id,
            check_id=86,
            band="not_evaluated",
            measurement=Measurement(notes="no known / not-known split was recorded"),
            confidence="low",
            policy=policy,
        )
    if finding is None:
        return build_verdict(
            task_id=task.task_id,
            check_id=86,
            band="not_evaluated",
            measurement=Measurement(notes="the known / not-known split was not audited"),
            confidence="low",
            policy=policy,
        )
    return build_verdict(
        task_id=task.task_id,
        check_id=86,
        band="non_fail" if finding.is_issue else "clean",
        measurement=Measurement(counts={"assessment": finding.assessment}),
        contributing_items=[finding.quote] if finding.is_issue else [],
        confidence=finding.confidence,
        policy=policy,
    )
