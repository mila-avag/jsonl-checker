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
from .context import EvidenceProfile, golden_answer_block, render_comparison, render_conversation
from .models import ModelSubmission, RubricCriterion, Sxs, Task
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
    golden = golden_answer_block(task)
    return f"""## The user's goal

{task.user_goal or '(not stated)'}
{golden}
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

COLLABORATION_RATING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "dimension": {"type": "string", "enum": ["Collaboration quality"]},
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
    # to restore, and this build ran a deliberately broader definition -- any tool
    # or integration, rather than L1_DEFINITIONS' four named connectors -- while
    # noting the customer had to settle which.
    #
    # They since have, in their own QC. An auditor failing a task wrote, of both
    # models: "No Gmail, Drive, Calendar, Github or other external connector was
    # requested or invoked. Since no such request was ever required or failed, the
    # dimension should be a 5. There is no applicable model failure that would
    # reduce the rating in this dimension." That is the narrow reading, and it
    # settles both halves: scope is external connectors, and their absence is a 5
    # rather than a middling score. The worry that narrowing makes web-search and
    # code-execution conversations unrateable was misplaced -- they are rateable,
    # and the answer is 5.
    "Tool & connector reliability": (
        "Whether external connectors the model invoked -- Gmail, Drive, Calendar, "
        "GitHub and their like -- executed end to end and returned real results, "
        "rather than erroring, timing out, or coming back empty, and whether "
        "failures were recovered from rather than ignored. Scope is external "
        "connectors to systems of record. A model's own built-in facilities, such "
        "as web search, code execution, or file generation, are not connectors "
        "for this dimension; where they fail, the cost lands on Outcome quality "
        "or Trust & grounding instead.\n\n"
        "Where no external connector was requested, required, or invoked, there is "
        "no failure available to this dimension and the rating is 5. Absence of "
        "connector activity is not evidence of unreliability, and a middling score "
        "on that ground is wrong."
    ),
    "Memory & personalization": L1_DEFINITIONS["Personalization and Memory"],
    "Collaboration quality": L1_DEFINITIONS["Collaboration Quality"],
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


def build_likert_prompt(
    task: Task, policy: Policy = DEFAULT_POLICY, comparison: Sxs | None = None
) -> str:
    return build_likert_prompt_with_evidence(task, policy, comparison)[0]


def build_likert_prompt_with_evidence(
    task: Task, policy: Policy = DEFAULT_POLICY, comparison: Sxs | None = None
) -> tuple[str, list[EvidenceProfile]]:
    """The blind Likert prompt for one comparison.

    `comparison` names which pair to render; it defaults to the task's primary
    one so that a single-comparison task and every existing caller are
    unaffected. The scale anchors are written from the pair's own slots, because
    an A-vs-C comparison whose anchors say "Model B is much better" asks the
    judge about a model that is not in front of it.
    """
    pair = comparison or task.comparison_list()[0]
    left, right = pair.left, pair.right
    rendered, profiles = render_comparison(
        task.submission(left), task.submission(right), policy
    )
    lo, hi = policy.likert_scale
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

{rendered}

## Your task

Compare the two conversations and place your preference on a {lo}-{hi} scale:

  1 = Model {left} is much better
  2 = Model {left} is better
  3 = Model {left} is slightly better
  4 = the two are equivalent
  5 = Model {right} is slightly better
  6 = Model {right} is better
  7 = Model {right} is much better

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
    "produce one. You are being asked a narrower question: does this conversation "
    "support the rating under review? "
    "Answer it from the conversation, not from the distance between the two "
    "numbers. How far apart the evaluators landed is not evidence of anything -- a "
    "rating one point from another evaluator's can still rest on a claim the "
    "transcript refutes, and a rating three points away can still be exactly what "
    "the conversation shows. "
    "`defensible` means you can point to what in the conversation holds the rating "
    "up: the behaviour it credits happened, the shortcoming it penalises is there, "
    "and the reasoning given for it describes this conversation. "
    "`indefensible` means the conversation does not support it: it credits "
    "something that did not happen, ignores or denies something that plainly did, "
    "or the reasoning given for it rests on a claim the transcript refutes -- "
    "whether or not the number itself is close to the other evaluator's. "
    "When the transcript does not contain what you would need in order to tell -- "
    "the deliverable's content, or a stretch of the conversation you were not "
    "shown -- answer `cannot_determine`. Abstaining is correct and useful; "
    "guessing is not, and neither is defaulting to `defensible` because the "
    "question is hard."
)

_ADJUDICATION_TASK = """
## Your task

Decide whether this conversation supports the rating under review.

  - "defensible": you can name what in the conversation holds this rating up. Say
    in `why` what that is and cite the turns in `evidence_turns`. You do not have
    to agree with the number -- you have to be able to point at the evidence for
    it.
  - "indefensible": the conversation does not support it. Say in `why` what it
    credits that did not happen, what it ignores that plainly did, or which of the
    claims in its stated reasoning the transcript refutes, and cite the turns.
  - "cannot_determine": you were not shown what you would need in order to tell.

Two things are not reasons for either verdict. The size of the gap between the two
evaluators is not one: judge the rating against the conversation, and a rating one
point from the other evaluator's gets exactly the same scrutiny as one three points
away. And the other evaluator's number is not one -- it is shown only so you know
what the disagreement was about, it carries no authority, and it may be the wrong
one.

Both verdicts have to be earned from the transcript. `defensible` is not the safe
answer for a call you find difficult; if you cannot point to what supports the
rating and cannot point to what refutes it, the honest answer is
`cannot_determine`.
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
    # The wide budget, not the blind pass's. Adjudication is one call about one
    # dimension, so it can read the turns the blind pass had to elide -- and unless
    # it does, its verdict is formed from the same gap that caused the
    # disagreement and the gate has nothing independent to count.
    conversation, profile = render_conversation(
        sub,
        policy,
        policy.adjudication_max_conversation_chars,
        policy.adjudication_max_turn_chars,
    )
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
    comparison: Sxs | None = None,
) -> tuple[str, list[EvidenceProfile]]:
    """The informed second pass over one comparison's Likert.

    Defaults to the task's primary comparison, so a single-comparison task and
    every existing caller behave as before.
    """
    pair = comparison or task.comparison_list()[0]
    left, right = pair.left, pair.right
    rendered, profiles = render_comparison(
        task.submission(left),
        task.submission(right),
        policy,
        policy.adjudication_max_comparison_chars,
        policy.adjudication_max_turn_chars,
    )
    lo, hi = policy.likert_scale
    justification = " ".join((pair.justification or "").split())
    return f"""## The user's goal

{task.user_goal or '(not stated)'}

{rendered}

## The scale

  1 = Model {left} is much better
  2 = Model {left} is better
  3 = Model {left} is slightly better
  4 = the two are equivalent
  5 = Model {right} is slightly better
  6 = Model {right} is better
  7 = Model {right} is much better

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

# Long enough that contributor prose cannot collide with the transcript it
# describes. At 30 characters it did: a justification opening "The
# image-generation workflow was used consistently..." reduced to the canary
# "the image-generation workflow ", a phrase appearing seven times in that
# task's own conversation, and the false leak cost the task its whole rating
# stage. A real leak inserts the justification verbatim, so a longer prefix
# still catches it; only the collisions go away.
CANARY_LENGTH = 120

# The floor stays where it was. A justification shorter than the prefix is
# canaried whole, which is a fingerprint in its own right; raising this instead
# of the prefix would have switched the guard off for exactly the short
# justifications it already protected.
MIN_CANARY_LENGTH = 30


@dataclass
class BlindnessReport:
    leaked: list[str]

    @property
    def clean(self) -> bool:
        return not self.leaked


def assert_blind(prompt: str, task: Task) -> BlindnessReport:
    """Verify a blind-pass prompt exposes none of the contributor's judgments.

    The contributor's justification text is the canary, taken long enough to be a
    fingerprint rather than a phrase: contributors describe the conversation in
    the conversation's own vocabulary, so a short prefix collides with the
    transcript and fails the rating stage of a task that was never contaminated.
    Any path that starts including contributor context drags the whole
    justification in, which a long prefix still catches.
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
        if len(canary) >= MIN_CANARY_LENGTH and canary in lowered:
            leaked.append(f"{label} leaked into a blind prompt")

    if task.sxs.likert is not None and f"likert {task.sxs.likert}" in lowered:
        leaked.append("contributor Likert value present")

    return BlindnessReport(leaked=leaked)
