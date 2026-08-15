"""Blind-pass prompts for checks 270, 300, and 400.

These three checks measure how far the contributor's ratings are from an
independent rating, so the contributor's values must be absent from context.
Anchoring here does not add noise, it pulls every rate toward agreement and makes
a broken audit look like a passing one.

Each prompt therefore contains only the conversation and the thing being rated.
`assert_blind` re-checks that at runtime using the contributor's own justification
text as a canary: a distinctive long string that could only appear in the prompt
through a leak.

Every schema offers `cannot_determine`. When the deliverable is a video the
transcript does not contain, the honest answer is that the transcript cannot
support a rating, and abstaining is what keeps the disagreement rate meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_POLICY, Policy
from .context import EvidenceProfile, render_comparison, render_conversation
from .models import ModelSubmission, RubricCriterion, Task
from .taxonomies import L1_DEFINITIONS, RATING_DIMENSIONS

RATING_SYSTEM_PROMPT = (
    "You are an independent evaluator. You are shown a conversation and asked to "
    "judge it yourself, from scratch. You are never shown anyone else's rating, and "
    "you must not try to infer what rating would be expected. Cite the turn numbers "
    "that support your judgment. When the transcript does not contain what you would "
    "need in order to judge -- most often because the model produced a file, image, "
    "or video whose content is not in the text -- say so with cannot_determine "
    "instead of guessing. Abstaining is a correct and useful answer. "
    # QC spec general grading instructions.
    "Where a requirement admits more than one reasonable reading and the conversation "
    "satisfies one of them, judge it satisfied. Never penalise a conversation for "
    "doing something the prompt or the task instructions explicitly asked for."
)

# The markers are quoted as the literal strings `render_conversation` emits.
# "Turns where that happened are labelled above" was already here and was not
# enough: on a completed run 18 of 33 conversations exceeded the render budget and
# were visibly cut, and judges across both passes named the cut in their own
# reasoning and then scored the item anyway. A judge told to abstain "when the
# transcript cannot support a judgment" applies that to its confidence; a judge
# told which four strings to look for applies it to the page.
ABSTAIN_GUIDANCE = """
Use `cannot_determine` when the transcript does not contain what you would need in
order to judge. The conversation above says where it is incomplete, and these are
the exact markers:

  - `[... conversation truncated here; N exchanges total ...]` -- the conversation
    runs to N exchanges in total and everything past this line was cut. It
    happened; you were not shown it.
  - `[... N characters omitted ...]` -- the middle of that one turn was cut.
  - `[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its
    quality cannot be judged from this text]` -- a file, image, or video was
    delivered here. You can see that it arrived, not what was in it.
  - `Files accompanying this conversation (contents NOT available to you;
    filenames only)` -- names, never contents.

If what you would need to judge this item sits behind one of those markers, you
MUST abstain and you MUST NOT score the item low for it. Do not infer quality from
the model's own claim that it succeeded, and do not treat an absent deliverable as
a deliverable that was never produced.

Abstaining is a correct, expected answer and it costs you nothing. An abstention is
recorded and then leaves the comparison entirely -- it is not read as a low score,
not read as a high one, and no rate is computed over it. Say what was missing in
`why_unverifiable`: the turn number you needed, or the deliverable you would have
had to open. A judgment made from a marker instead of evidence is the one answer
this audit has no use for.
"""


# ---------------------------------------------------------------------------
# 270 - rubric evaluation, one call per (criterion, model)
# ---------------------------------------------------------------------------

CRITERION_RATING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "criterion_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail", "cannot_determine"]},
        "evidence_turns": {"type": "array", "items": {"type": "integer"}},
        "why_unverifiable": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "criterion_id",
        "verdict",
        "evidence_turns",
        "why_unverifiable",
        "reasoning",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_criterion_rating_prompt(
    task: Task,
    criterion: RubricCriterion,
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
) -> str:
    return build_criterion_rating_prompt_with_evidence(task, criterion, sub, policy)[0]


def build_criterion_rating_prompt_with_evidence(
    task: Task,
    criterion: RubricCriterion,
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[str, EvidenceProfile]:
    """The prompt, and what the render could actually support.

    The profile is the half this module used to throw away. `render_conversation`
    knows whether it cut the conversation and how deep it got, the request that
    carries the prompt is the only place that knowledge can be attached to the
    judgment it conditions, and discarding it here is what let a rating made from
    half a transcript reach a gate looking like any other.
    """
    conversation, profile = render_conversation(sub, policy)
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

## Conversation with {"Model " + sub.model}

{conversation}

## The single requirement you are judging

[{criterion.criterion_id}] {' '.join(criterion.text.split())}

## Your task

Decide whether this conversation satisfies that one requirement.

  - "pass": the conversation satisfies it, and you can point to the turns showing so.
  - "fail": the conversation does not satisfy it, and you can point to the turns showing so.
  - "cannot_determine": the transcript does not contain what you would need to tell.

{ABSTAIN_GUIDANCE}

List in `evidence_turns` the turn numbers you relied on. Judge only this
requirement; ignore every other quality of the conversation.

Return JSON matching the schema. Set `criterion_id` to "{criterion.criterion_id}".
""", profile


# ---------------------------------------------------------------------------
# 300 - dimension ratings, one call per (model, dimension)
# ---------------------------------------------------------------------------

DIMENSION_RATING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "dimension": {"type": "string", "enum": list(RATING_DIMENSIONS)},
        "rating": {"type": ["integer", "null"]},
        "not_applicable": {"type": "boolean"},
        "cannot_determine": {"type": "boolean"},
        "why_unverifiable": {"type": "string"},
        "evidence_turns": {"type": "array", "items": {"type": "integer"}},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "dimension", "rating", "not_applicable", "cannot_determine",
        "why_unverifiable", "evidence_turns", "reasoning", "confidence",
    ],
    "additionalProperties": False,
}

DIMENSION_GUIDANCE: dict[str, str] = {
    "Outcome quality": L1_DEFINITIONS["Outcome Quality"],
    "Communication quality": L1_DEFINITIONS["Communication Quality"],
    "Trust & grounding": L1_DEFINITIONS["Trust and Grounding"],
    "Interaction efficiency": L1_DEFINITIONS["Interaction Efficiency"],
    # The only dimension whose text is not the customer's. Tab 2 row 67 names it as
    # a rating dimension but row 58 does not define it, so there is no spec wording
    # to restore. This is a second, independently worded definition of the same
    # dimension -- L1_DEFINITIONS["Tool and Connector Reliability"] scopes it to the
    # four named connectors, this one to any tool or integration -- and the customer
    # has to settle which. Kept broader here deliberately: the rating dimension is
    # scored on conversations whose only tool use is web search or code execution,
    # and the narrower text would make those unrateable.
    "Tool & connector reliability": (
        "Whether tools, connectors, and integrations the model invoked actually "
        "worked: calls succeeded, results were used correctly, failures were "
        "recovered from rather than ignored."
    ),
    "Memory & personalization": L1_DEFINITIONS["Personalization and Memory"],
}


def build_dimension_rating_prompt(
    task: Task,
    dimension: str,
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
) -> str:
    return build_dimension_rating_prompt_with_evidence(task, dimension, sub, policy)[0]


def build_dimension_rating_prompt_with_evidence(
    task: Task,
    dimension: str,
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[str, EvidenceProfile]:
    lo, hi = policy.dimension_rating_scale
    conversation, profile = render_conversation(sub, policy)
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

## Conversation with {"Model " + sub.model}

{conversation}

## The dimension you are rating

{dimension}: {DIMENSION_GUIDANCE.get(dimension, '')}

## Your task

Rate this conversation on that one dimension, from {lo} (worst) to {hi} (best).

  - Set `rating` to your score and leave the two flags false.
  - Set `not_applicable` true, with `rating` null, if the dimension genuinely does
    not apply to this task (for example, memory and personalization in a task with
    nothing to carry across turns).
  - Set `cannot_determine` true, with `rating` null, if the transcript does not
    contain what you would need in order to rate it.

{ABSTAIN_GUIDANCE}

List in `evidence_turns` the turn numbers you relied on. Rate only this dimension.

Return JSON matching the schema. Set `dimension` to "{dimension}".
""", profile


# ---------------------------------------------------------------------------
# 400 - side-by-side Likert, one call per task
# ---------------------------------------------------------------------------

LIKERT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "likert": {"type": ["integer", "null"]},
        "cannot_determine": {"type": "boolean"},
        "why_unverifiable": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "likert",
        "cannot_determine",
        "why_unverifiable",
        "reasoning",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_likert_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return build_likert_prompt_with_evidence(task, policy)[0]


def build_likert_prompt_with_evidence(
    task: Task, policy: Policy = DEFAULT_POLICY
) -> tuple[str, list[EvidenceProfile]]:
    comparison, profiles = render_comparison(task.model_a, task.model_b, policy)
    lo, hi = policy.likert_scale
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

{comparison}

## Your task

Compare the two conversations and place your preference on a {lo}-{hi} scale:

  1 = Model A is much better
  2 = Model A is better
  3 = Model A is slightly better
  4 = the two are equivalent
  5 = Model B is slightly better
  6 = Model B is better
  7 = Model B is much better

Judge which conversation better served the user's goal overall. Set
`cannot_determine` true with `likert` null only if neither transcript contains
enough to compare at all -- a comparison you can make only weakly should still be
a rating, with `confidence` set to low. Truncation on one side alone is enough to
abstain when the difference you would be ranking on sits in the part of that side
you were not shown: the two conversations share one render budget, so each is cut
sooner here than it would be on its own.

{ABSTAIN_GUIDANCE}

Return JSON matching the schema.
""", profiles


# ---------------------------------------------------------------------------
# Informed adjudication -- the second pass for 300 and 400
# ---------------------------------------------------------------------------
#
# Everything above is blind, and has to be: a judge shown the contributor's
# number agrees with it, and then the disagreement rate measures nothing. What
# blindness cannot do is tell a defect from a difference of opinion. A blind
# rating of 2 against a contributor's 5 says the two disagree; it does not say
# the 5 was wrong, and on a manual audit of check-300 fails that gap was most of
# them.
#
# So these prompts deliberately do the thing the ones above forbid. They are
# shown the contributor's rating and their own justification, and they are not
# asked to rate anything. They are asked whether the rating the contributor gave
# is defensible against the conversation -- which is the question the customer's
# fail cell is actually about, and the only one whose answer a contributor could
# be shown. `assert_blind` must never run on these; `build_adjudication_requests`
# builds them outside the guarded path for that reason.

ADJUDICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["defensible", "indefensible", "cannot_determine"],
        },
        "why": {"type": "string"},
        "evidence_turns": {"type": "array", "items": {"type": "integer"}},
        "why_unverifiable": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["verdict", "why", "evidence_turns", "why_unverifiable", "confidence"],
    "additionalProperties": False,
}

ADJUDICATION_SYSTEM_PROMPT = (
    "You are adjudicating a disagreement between two evaluators who rated the same "
    "conversation. You are not being asked for your own rating and you must not "
    "produce one. You are being asked a narrower question: is the rating under "
    "review one a careful reviewer could defend against this conversation? "
    "Ratings on a 1-5 scale are judgments, and two careful reviewers routinely "
    "land a point or two apart on the same evidence -- that is a difference of "
    "opinion, not an error, and it is `defensible`. Reserve `indefensible` for a "
    "rating the conversation actually contradicts: it credits something that did "
    "not happen, penalises something that plainly did, or rests on a claim the "
    "transcript refutes. When the transcript does not contain what you would need "
    "in order to tell -- the deliverable's content, or a stretch of the "
    "conversation you were not shown -- answer `cannot_determine`. Abstaining is "
    "correct and useful; guessing is not."
)

_ADJUDICATION_TASK = """
## Your task

Decide whether the rating under review is defensible against this conversation.

  - "defensible": a careful reviewer could hold this rating on this evidence, even
    if you would have chosen a different number. A one- or two-point difference of
    emphasis is defensible by default.
  - "indefensible": the conversation contradicts the rating. Say in `why` what it
    credits that did not happen, or what it ignores that plainly did, and cite the
    turns in `evidence_turns`.
  - "cannot_determine": you were not shown what you would need in order to tell.

You are judging the rating, not the conversation, and not the other evaluator's
number. The other evaluator's rating is shown only so you know what the
disagreement was about; it carries no authority and it may be the wrong one.
"""


def build_dimension_adjudication_prompt(
    task: Task,
    dimension: str,
    sub: ModelSubmission,
    contributor_rating: int | None,
    contributor_justification: str,
    auditor_rating: int | None,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[str, EvidenceProfile]:
    lo, hi = policy.dimension_rating_scale
    conversation, profile = render_conversation(sub, policy)
    justification = " ".join((contributor_justification or "").split())
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

## Conversation with {"Model " + sub.model}

{conversation}

## The dimension

{dimension}: {DIMENSION_GUIDANCE.get(dimension, '')}

Rated on a {lo}-{hi} scale, {lo} worst and {hi} best.

## The rating under review

{contributor_rating if contributor_rating is not None else '(not applicable)'}

Reasoning given for it: {justification or '(none recorded)'}

## For context only: a second evaluator's rating of the same dimension

{auditor_rating if auditor_rating is not None else '(not applicable)'}
{_ADJUDICATION_TASK}
{ABSTAIN_GUIDANCE}

Return JSON matching the schema.
""", profile


def build_ranking_adjudication_prompt(
    task: Task,
    contributor_likert: int | None,
    auditor_likert: int | None,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[str, list[EvidenceProfile]]:
    comparison, profiles = render_comparison(task.model_a, task.model_b, policy)
    lo, hi = policy.likert_scale
    justification = " ".join((task.sxs.justification or "").split())
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

{comparison}

## The scale

  1 = Model A is much better
  2 = Model A is better
  3 = Model A is slightly better
  4 = the two are equivalent
  5 = Model B is slightly better
  6 = Model B is better
  7 = Model B is much better

## The preference under review

{contributor_likert if contributor_likert is not None else '(none recorded)'} on that {lo}-{hi} scale.

Reasoning given for it: {justification or '(none recorded)'}

## For context only: a second evaluator's placement on the same scale

{auditor_likert if auditor_likert is not None else '(none recorded)'}
{_ADJUDICATION_TASK}

One more rule specific to a comparison: if the two conversations were not shown to
you at comparable depth, you cannot adjudicate a preference between them. The
render budget is shared, so a long conversation is cut sooner here than it would be
alone, and the disclosure above says how much of each side you are seeing. If the
grounds for the preference could sit in the part of either side you were not shown,
answer `cannot_determine`.

{ABSTAIN_GUIDANCE}

Return JSON matching the schema.
""", profiles


# ---------------------------------------------------------------------------
# Blindness guard
# ---------------------------------------------------------------------------

CONTRIBUTOR_MARKERS = (
    "contributor rated",
    "contributor scored",
    "contributor's rating",
    "contributor's score",
    "contributor gave",
    "the contributor believes",
    "existing rating",
    "previous rating",
)

CANARY_LENGTH = 30


@dataclass
class BlindnessReport:
    leaked: list[str]

    @property
    def clean(self) -> bool:
        return not self.leaked


def assert_blind(prompt: str, task: Task) -> BlindnessReport:
    """Verify a blind-pass prompt exposes none of the contributor's judgments.

    The contributor's justification text is the canary. It is long and distinctive,
    so it cannot appear in a correctly built prompt by chance, and any path that
    starts including contributor context will drag it in.
    """
    leaked: list[str] = []
    lowered = prompt.lower()

    for marker in CONTRIBUTOR_MARKERS:
        if marker in lowered:
            leaked.append(f"contributor framing present: {marker!r}")

    canaries: list[tuple[str, str]] = []
    if task.sxs.justification:
        canaries.append(("ranking justification", task.sxs.justification))
    for d in task.dimension_ratings:
        if d.justification:
            canaries.append((f"{d.model} {d.dimension} justification", d.justification))
    if task.key_turn.justification:
        canaries.append(("key turn justification", task.key_turn.justification))

    for label, text in canaries:
        canary = " ".join(text.lower().split())[:CANARY_LENGTH]
        if len(canary) >= CANARY_LENGTH and canary in lowered:
            leaked.append(f"{label} leaked into a blind prompt")

    if task.sxs.likert is not None and f"likert {task.sxs.likert}" in lowered:
        leaked.append("contributor Likert value present")

    return BlindnessReport(leaked=leaked)
