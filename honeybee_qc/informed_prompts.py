"""Informed-pass prompts for checks 70, 75, 80, 95, 110, 220, 280, 310, 450, and 470.

Most of these checks audit the contributor's *stated reasoning* -- their target
outcome list, their key turn justification, the turns they cited, the
justifications they wrote, and whether their comparison names a winner. You
cannot judge a justification without reading it, so unlike the blind pass in
`rating_prompts`, these prompts deliberately show the contributor's work. That is
the defining difference between the two stages, and the reason `assert_blind` is
never applied here.

The blindness rule still holds where it matters. None of these prompts asks the
auditor to produce a rating of its own, so there is no independent judgment for
the contributor's values to anchor. Checks 270, 300, and 400 remain the only
place a competing rating is formed, and they stay blind.

Calls are batched per model rather than per item -- one call covering all seven
dimension justifications instead of seven -- because each one needs the same
conversation in context. Batching cuts the token cost roughly sevenfold and lets
the auditor weigh a set of justifications against each other, which is how the
per-justification thresholds in 450 are meant to be applied.
"""

from __future__ import annotations

from .config import DEFAULT_POLICY, Policy
from .context import NO_CONVERSATION_MARKER_LEAD as _NO_CONVERSATION_LEAD
from .context import golden_answer_block as _golden_answer_block
from .context import numbered_prompts as _numbered_prompts
from .context import render_comparison, render_conversation
from .env_context import input_manifest, universe_materials
from .models import ModelSubmission, Sxs, Task

INFORMED_SYSTEM_PROMPT = (
    "You are auditing the work of a contributor who evaluated a model conversation. "
    "You are shown what they wrote and asked whether it holds up. Judge only what "
    "the check in front of you asks about; these checks are scored separately and "
    "deliberately overlap in subject matter, so straying is double-counting. "
    "Quote the contributor's own words as evidence for every issue you report. "
    "When the transcript cannot support a judgment -- most often because a "
    "deliverable was a file, image, or video whose content is not in the text -- "
    "say so rather than guessing. "
    # QC spec general grading instructions. You are shown the contributor's own
    # reasoning, so the charity rule is about how to treat that reasoning: the spec
    # says "if a contributor presents a reasonable argument, we should accept their
    # interpretation", and its example is a contributor reading "from 2012 until
    # 2014" as excluding 2014.
    "Where the contributor presents a reasonable argument for the reading they "
    "took, accept their interpretation and judge the work against it, even where "
    "you would have argued differently -- a defensible reading you disagree with "
    "is not an issue. Never penalise a contributor for doing something the prompt "
    "or the task instructions explicitly asked for."
)

# Shared by every prompt below that renders a conversation. The markers are quoted
# as the literal strings `render_conversation` emits, because "say so when the
# transcript cannot support a judgment" -- which the system prompt has always said
# -- is an instruction a judge applies to its own confidence rather than to what is
# on the page. Measured on a completed run: 82 of 305 informed judge calls named a
# truncated transcript or an absent deliverable in their own reasoning and returned
# a finding anyway, against 18 of 33 conversations that were visibly cut. The judge
# had the signal and no field to put it in, so it guessed, and the guess was
# charged to a contributor who had read the whole conversation.
UNVERIFIABLE_GUIDANCE = f"""
## When you were not shown the evidence

The conversation above may be incomplete, and where it is, it says so. These are
the exact markers, and they are the only warning you get:

  - `[... conversation truncated here; N exchanges total ...]` -- the conversation
    continues past this line for N exchanges in total and you were shown none of
    them.
  - `[... N characters omitted ...]` -- the middle of that one turn was cut.
  - `[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its
    quality cannot be judged from this text]` -- a file, image, or video was handed
    to the user here. You are reading the delivery, not the thing delivered.
  - `Files accompanying this conversation (contents NOT available to you;
    filenames only)` -- you have names, never contents.
  - `{_NO_CONVERSATION_LEAD}` -- nothing at all came back for
    that side, and the heading above it is followed by no turns whatsoever. This is
    the only marker that can appear with an empty conversation beneath it, and it
    means every claim about what either speaker said there is unverifiable. An
    empty block is never evidence that a model said nothing.

If deciding an item would need a turn, a span of a turn, or the content of a
deliverable that one of those markers stands in for, that item is unverifiable.
You MUST mark it unverifiable using the field named in your task below, and you
MUST NOT report a defect for it.

Abstaining is a correct, expected answer and it costs you nothing. An unverifiable
item is recorded and then counted neither for the contributor nor against them:
there is no penalty for the audit, none for you, and none for them. A defect
reported from evidence you were never shown is different -- it is charged to
someone who read the whole conversation, and it is the one outcome this audit
cannot tolerate. When in doubt about whether you saw enough, you did not.

Say what was missing whenever you abstain: the turn number you needed, the
deliverable whose content you would have had to open, or the marker you are
relying on. "Cannot tell" without naming the gap is not an abstention.
"""


# ---------------------------------------------------------------------------
# 70 - domain relevance, one call per task
# ---------------------------------------------------------------------------

DOMAIN_RELEVANCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["aligned", "partial", "unrelated"],
        },
        "domain_basis": {"type": "string"},
        "prompt_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "assessment",
        "domain_basis",
        "prompt_quote",
        "reasoning",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_domain_relevance_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    category = task.prompt_category or "(none assigned)"
    pre_seed = (task.pre_seeded_prompt or "").strip()
    pre_seed_block = (
        f"""

## The scenario the platform pre-seeded for this assignment

{pre_seed}

The contributor's job was to naturalise this specific scenario into the prompt
above in their own words -- rephrasing, adding colour, or extending it across
turns is expected and is not misalignment on its own. But the assigned domain
was commissioned *through* this scenario, not as an abstract label the
contributor could satisfy with any topic that happens to share it. If the
submitted prompt keeps the same persona-level label but abandons this scenario
for an unconnected fact pattern and a different concrete ask -- a different
matter, a different client, a different problem, sharing nothing but the
broad professional category -- the domain was never actually exercised, and
that counts as "unrelated" here even though a generic description of the new
topic would also sit inside the assigned persona's work. The bar is whether
this scenario survived, not whether some scenario in the category would fit."""
        if pre_seed
        else ""
    )
    return f"""## The domain this prompt was commissioned for

Assigned domain / persona: {task.assigned_domain}
Assigned use-case category: {category}
{pre_seed_block}

## The prompt the contributor submitted

{_numbered_prompts(task)}

## Your task

Decide whether this prompt is clearly related to the assigned domain above. The
standard for a clean task is the customer's: "The Prompt(s) are clearly related
to the assigned domain."

Pick one `assessment`:

  - "aligned": the prompt is work the assigned persona plausibly does, or a
    question they plausibly ask. This is the expected answer.
  - "partial": the prompt belongs to the domain but leans noticeably away from
    its centre -- it emphasises a neighbouring speciality, or the persona is a
    stretch rather than a natural fit.
  - "unrelated": egregious misalignment. The prompt is about a different field
    altogether and no part of it is work the assigned persona would do{
        ", OR it discards the pre-seeded scenario above wholesale for an "
        "unconnected one, per the instructions above" if pre_seed else ""
    }.

Hold the "unrelated" bar high. It is reserved for the case where naming the
assigned domain alongside this prompt looks like a clerical error{
    ", or where the pre-seeded scenario above was thrown out for something it "
    "shares no facts with," if pre_seed else ""
} and it fails the task outright. Reach for it only when you could not
construct a reasonable story connecting the two.

Three things that are not misalignment. A prompt at the edge of a domain, or
spanning it and an adjacent one, is still related to it -- breadth is normal and
belongs in "partial" at worst. A persona who is a plausible *user* of the
subject matter counts as related even where they are not a specialist in it: a
backend engineer buying hardware, or an educator asking about physics, is
working within their domain. And a generically worded prompt is not misaligned;
whether a prompt is specific enough is not this check's business.

Two quotes are required whenever you answer anything other than "aligned", and
the finding is discarded without them:

  - `prompt_quote`: the words from the submitted prompt, verbatim, that show what
    subject it is actually about.
  - `domain_basis`: the words from the assignment above, verbatim, that the
    prompt fails to match.

Quote both sides for "aligned" too, naming the part of the prompt that connects
it to the assignment.

Judge subject matter only. The prompt's clarity, difficulty, answerability, and
whether it has a unique ground truth are all outside this check.
"""


# ---------------------------------------------------------------------------
# 75 - pre-seeded/opening prompt consistency, one call per task
# ---------------------------------------------------------------------------

PROMPT_CONSISTENCY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["same", "reworded_same_intent", "different"],
        },
        "pre_seeded_quote": {"type": "string"},
        "submitted_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "assessment",
        "pre_seeded_quote",
        "submitted_quote",
        "reasoning",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_prompt_consistency_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    """Check 75. The submitted *opening* prompt against the platform's pre-seed.

    Shown the pre-seed once and the full numbered submission, because a later
    turn can still be the contributor's own follow-up rather than part of the
    opening prompt itself -- the judge is told which one this check is about
    rather than left to guess from position alone.
    """
    return f"""## The pre-seeded prompt the platform prepared for the contributor

{' '.join((task.pre_seeded_prompt or '').split())}

## The prompt the contributor actually submitted

{_numbered_prompts(task)}

## Your task

Decide whether the contributor's *opening* (first) submitted prompt represents
the same underlying request as the pre-seeded prompt above -- the same core ask
-- not whether the wording matches. Paraphrasing, a reworded opening, and added
context (a persona, a reason for asking, extra background) are all expected and
must not be read as a mismatch on their own.

Pick one `assessment`:

  - "same": the opening prompt is the pre-seeded prompt, verbatim or with only
    trivial rewording -- a typo fixed, punctuation, a word or two swapped for a
    synonym.
  - "reworded_same_intent": the wording differs, sometimes substantially -- a
    different opening framing, an added persona or context, a different
    narrative voice -- but the underlying request is the same: same subject,
    same core ask. This is the expected answer whenever the prompt was
    naturalised rather than copied.
  - "different": the two are about different subjects or make different
    requests. No reasonable reading connects them to the same underlying task.

Read the whole submitted prompt before deciding, not only its first sentence. A
prompt that opens by establishing context, a persona, or a reason for asking --
before restating the pre-seeded ask in its own words -- is "reworded_same_intent",
not "different"; the context is additional, not a substitution.

Two quotes are required whenever you answer anything other than "same", and the
finding is discarded without them:

  - `pre_seeded_quote`: the words from the pre-seeded prompt, verbatim, that
    state its core ask.
  - `submitted_quote`: the words from the submitted prompt, verbatim, that state
    its core ask.

Quote both sides for "same" too, naming the part of each that shows the match.

Judge underlying intent only. Whether the prompt is clear, specific, in the
right domain, or references anything outside the task's universe are all other
checks.
"""


# ---------------------------------------------------------------------------
# 80 - target outcome, one call per task
# ---------------------------------------------------------------------------

TARGET_OUTCOME_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["supported", "not_required", "contradicts_prompt"],
                    },
                    "established_turn": {"type": ["integer", "null"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["entry", "classification", "established_turn", "reasoning"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["entries", "confidence"],
    "additionalProperties": False,
}


def build_target_outcome_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    outcomes = "\n".join(f"  - {' '.join(o.split())}" for o in task.target_outcome)
    return f"""## The prompts the user sent, in order

{_numbered_prompts(task)}

## The target outcome list the contributor wrote

{outcomes or '(empty)'}

## Your task

Decide whether the target outcome list matches what the prompts actually ask for.

Build the requirement set by replaying every turn in order and applying each
revision and removal as you go. Compare the list against that final expected
state, never against turn 1 alone. A later turn can add a requirement, narrow
one, or withdraw one entirely; an entry that matches the revised form is correct
even where it contradicts the original request, and an entry that still matches
a withdrawn requirement is wrong.

A target outcome list is not a restatement of the prompts. It "lists the
necessary components for a successful response", it "can include components of
artifacts and/or final model responses", and it "may include additional
components beyond those required by the initial prompt". So an entry that
specifies what a correct deliverable contains -- a value, a filename, a
recipient, a structural requirement, a thing the response must not do -- is
legitimate even though no turn asked for it in those words. Withhold
"not_required" unless the entry is something the final expected state positively
does not call for; being unable to trace it to a turn is not enough on its own,
because the input files that would ground it are not in front of you.

Classify each entry:

  - "supported": the final expected state requires it. Name the turn that
    established its final form in `established_turn`.
  - "not_required": nothing in the prompts asks for it, and it is not implied.
    The list may include additional components beyond those required by the
    initial prompt in multiturn conversations -- check later prompts to ensure
    a requirement established early on was not later revised or removed before
    flagging it as extraneous.
  - "contradicts_prompt": it conflicts with the final expected state.

Judge the list against the prompts only. You are not shown the models' responses
and must not speculate about them. Do not treat a paraphrase as a mismatch: an
entry that captures a requirement in different words is supported.
"""


# ---------------------------------------------------------------------------
# 85 - environment context, one call per task
# ---------------------------------------------------------------------------

ENVIRONMENT_CONTEXT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "declared_file_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "prompt_quote": {"type": "string"},
                    "prompt_treats_as_available_to_read": {"type": "boolean"},
                    "conversation_shows_attachment_or_upload": {"type": "boolean"},
                    "conversation_shows_independent_content": {"type": "boolean"},
                    # Asked here for the same reason they are asked of a
                    # `references` row. The parser used to hardcode both to True on
                    # this path, so a file the prompt asked the model to CREATE was
                    # a violation the moment the two upload questions came back
                    # false -- which is the expected answer for anything the model
                    # produced itself.
                    "presented_as_existing": {"type": "boolean"},
                    "required_to_complete": {"type": "boolean"},
                    "evidence_summary": {"type": "string"},
                },
                "required": [
                    "filename",
                    "prompt_quote",
                    "prompt_treats_as_available_to_read",
                    "conversation_shows_attachment_or_upload",
                    "conversation_shows_independent_content",
                    "presented_as_existing",
                    "required_to_complete",
                    "evidence_summary",
                ],
                "additionalProperties": False,
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "file",
                            "person",
                            "organisation",
                            "event",
                            "system",
                            "place",
                            "other",
                        ],
                    },
                    "quote": {"type": "string"},
                    "checked_against": {"type": "array", "items": {"type": "string"}},
                    # The one answer that had nowhere to go. The judge is asked to
                    # show what it searched, so it writes rows for references it
                    # looked for and FOUND, and the only field that could carry
                    # that was free-text prose nothing reads.
                    "found_in_materials": {"type": "boolean"},
                    "why_outside_universe": {"type": "string"},
                    "presented_as_existing": {"type": "boolean"},
                    "required_to_complete": {"type": "boolean"},
                    "real_world_public_entity": {"type": "boolean"},
                },
                "required": [
                    "reference",
                    "kind",
                    "quote",
                    "checked_against",
                    "found_in_materials",
                    "why_outside_universe",
                    "presented_as_existing",
                    "required_to_complete",
                    "real_world_public_entity",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["declared_file_checks", "references", "confidence"],
    "additionalProperties": False,
}


def build_environment_context_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    """Check 85's entity half. The file half runs in `env_context` with no call.

    Everything the audit can check a reference against is listed by name in the
    prompt, because the model has to say what it compared a reference to before a
    finding counts, and it cannot name a material it was never shown.

    The file manifest is also named here, but only ever as a *declared* list: it
    comes from the prompt-collection step, which records what the contributor
    attached to the task on Scale's side, not what actually crossed into the
    external model's chat. A contributor can (and on at least one audited task,
    did) attach a document as source material for writing the prompt, describe
    its contents in prose, and never upload it into the live Gemini/Claude
    conversation at all -- the manifest still names the file, and the model plays
    along with the user's description well enough that nothing about its answer
    looks wrong. The conversations are rendered below for exactly this reason:
    the manifest can say a file was supplied, and the conversation can show it
    never was, and only the second fact is the one the "task's universe" bar
    cares about.
    """
    # The manifest entry is deliberately excluded from this list and surfaced
    # separately below instead: `universe_materials` names it the same way it
    # names a conversation or a target-outcome list, and that flat "available to
    # you" framing is exactly what let the model treat a declared-but-undelivered
    # file as satisfied without ever reading the conversation for it.
    materials = [
        m for m in universe_materials(task) if not m.startswith("input file manifest")
    ]
    manifest = input_manifest(task)
    # `basenames()` is the union of supplied and produced files, and handing that
    # to the judge under the "DECLARES were supplied" heading is what told it the
    # models' own outputs were the contributor's inputs.
    files = (
        ", ".join(sorted(manifest.supplied_basenames()))
        or "(none with a readable name)"
    )
    produced = ", ".join(sorted(manifest.produced_basenames())) or "(none recorded)"
    outcomes = (
        "\n".join(f"  - {' '.join(o.split())}" for o in task.target_outcome) or "  (empty)"
    )
    conversations = []
    for sub in task.submissions():
        if sub.conversation:
            rendered, _ = render_conversation(sub, policy)
            conversations.append(f"### Conversation with Model {sub.model}\n\n{rendered}")
    conversation_block = (
        "\n\n".join(conversations)
        if conversations
        else "(no conversation was fetched for either model)"
    )
    return f"""## The prompt the user sent

{_numbered_prompts(task)}

## Everything else this task supplies, and nothing more

Materials available to you: {'; '.join(materials) or '(none)'}

Files the task DECLARES were supplied (from the prompt-collection step -- this
says the contributor attached these on Scale's side; it does NOT say the file
ever reached the model in the conversation below): {files}

Files the models PRODUCED during the conversation and handed back to the user:
{produced}

Those produced files are OUTPUTS. They are listed so you can recognise them, not
so you can audit them. A later user turn that asks the model to review, update,
correct, or quote a file the model itself generated earlier in this same
conversation is referring to something the conversation established, which is
inside the task's universe by construction. It is never an out-of-universe
reference, it never belongs in `declared_file_checks`, and the absence of an
upload for it is expected rather than evidence of anything.
Files attached but unnamed: {len(manifest.opaque)}

Target outcome list the contributor wrote:

{outcomes}

## The conversation(s) actually held with the model

{conversation_block}
{UNVERIFIABLE_GUIDANCE}
## Your task

Decide whether the prompt references anything that is not part of this task's
universe. The customer's standard is that the prompt "references only the
entities and events in the task's universe".

A violation is a reference the prompt treats as an existing part of the world the
model is working in, which nothing above supplies. The clearest case is a prompt
that says to use an attached file when no such file was attached; the same fault
covers a colleague, a client, a meeting, a prior decision, or an internal system
the prompt speaks of as established fact and which appears nowhere else.

## Mandatory: `declared_file_checks`

A file's name appearing in the declared list above is not, by itself, evidence
that the reference is satisfied, and you must not let it substitute for actually
reading the conversation. Before you write anything else, find every file the
prompt speaks of as something to review, open, pull up, check, or verify --
anything past merely mentioning it in passing -- and give each one its own entry
in `declared_file_checks`. This step is required even when your answer to every
field ends up being the unremarkable one; an empty `declared_file_checks` list is
only correct when the prompt never treats any file this way.

For each such file, answer literally, not as a summary judgement:

  - `conversation_shows_attachment_or_upload`: can you point to specific text in
    a USER turn that describes handing the file to the model -- "I've attached
    X", "here is the document", an upload notice, or the model itself saying it
    opened a file the user gave it? This can only be true from something on the
    input side. A "Files accompanying this conversation (contents NOT available
    to you; filenames only)" header lists what a turn produced as *output* --
    the assistant's own deliverables going to the user -- and never counts as
    evidence here no matter how many files it lists or how closely a name in it
    resembles the file under review; do not let a coincidence of filenames
    substitute for reading which direction the file moved.
  - `conversation_shows_independent_content`: set this true only when something
    in the conversation *tests* the model's account of the file against a
    source other than the model's own say-so, and the model's account holds up.
    A model producing specific, confident, detailed claims about a file's
    contents is not that test passing -- confident fabrication looks exactly
    like accurate recall from the outside, which is the entire reason this
    field exists rather than just asking "does the model sound like it read
    it". Two patterns are decisive and you should actively look for them before
    answering:
      * The user contradicts, corrects, or expresses surprise at a specific
        detail the model attributed to the file ("wait, I don't think that
        figure is right", "that's not what I remember it saying"). This is
        affirmative evidence AGAINST access -- a model reading the real file
        would not need correcting on the real file's own contents -- and
        `conversation_shows_independent_content` must be false whenever you
        find this, regardless of how much other detail the model produced.
      * The user explicitly affirms a specific, checkable claim as correct
        against their own independent knowledge of the file, unprompted by the
        model asking them to confirm anything.
    Absent one of those two patterns, treat unverified specificity as
    unverified: if nothing in the conversation ever puts the model's account of
    the file to a real test, this field is false, even where the model never
    gets caught -- a claim nobody checked is not a claim that passed.

  - `presented_as_existing` / `required_to_complete`: the same two questions
    asked of a `references` row, asked of this file. False for a file the prompt
    asks the model to CREATE, and false for one the model can simply write
    around. Both must be true before a missing file is a violation, and both are
    read from your answer rather than assumed.

If both upload questions are false and `prompt_treats_as_available_to_read` is
true, the file exists in this conversation in name only: the user is narrating a
document and the model is going along with the narration, not reading anything.
This is not a
rare or exotic failure -- it is the whole point of asking you to check every
named file this way rather than only when something else already looks wrong.
Content the user pasted or typed directly into their own message does not count
as a missing file; the file's contents have to be presented as something the
model can independently consult, not something the user is telling it.

You do not need to also add a matching entry to `references` for a file that
fails this check -- the audit derives the finding from `declared_file_checks`
directly. Use `references` (below) for every other kind of out-of-universe
reference: people, organisations, events, systems, and places.

## `references`: everything that is not a file

Report each such reference and answer four questions about it. All four decide
whether it counts, so answer them literally rather than as a summary of your
overall impression:

  - `found_in_materials`: did you actually locate this reference in one of the
    materials you searched? Answer TRUE whenever you did. A reference you found
    is not a violation no matter how the other three answers come out, and TRUE
    is the right answer for most rows you will write, because you are asked to
    show your work on references you considered as well as ones you are
    alleging. Do not put "it is in the task universe" in `why_outside_universe`
    and leave this false -- that field is prose the audit does not read, and a
    row left false on those terms fails the task.
  - `presented_as_existing`: does the prompt assert the thing is already there?
    False for anything it proposes, hypothesises, asks about, or invites the model
    to invent -- "draft a note to a hypothetical client" invents nothing.
  - `required_to_complete`: is the work impossible without it? False when the
    reference is scene-setting the model can simply write around.
  - `real_world_public_entity`: is it a public person, company, product, place, or
    event? True for those. A task about MikroTik switches names MikroTik
    legitimately, and a prompt is not required to supply the outside world.

Two quotes are required for every reference, and one lacking either is discarded:

  - `quote`: the words from the prompt, verbatim, that make the reference.
  - `checked_against`: the materials from the list above that you searched, named
    as they are named there. Say what you looked at and did not find it in.

Absence of evidence is not evidence of absence here. The materials above are
often thin -- an unnamed attachment or an empty conversation means the audit could
not read something, not that the contributor invented it. Where a reference could
plausibly be satisfied by material you were not shown, leave it out. An empty
`references` list is the expected answer for almost every task, and a wrong entry
costs more than a missed one: this check fails a task outright with no middle band.

Judge the prompt's grounding only. Whether it is clear, specific, difficult, or in
the right domain are all other checks.
"""


# ---------------------------------------------------------------------------
# 95 - output artifacts preserved, one call per model
# ---------------------------------------------------------------------------

ARTIFACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claimed_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "turn": {"type": ["integer", "null"]},
                    "quote": {"type": "string"},
                    "produced": {"type": "boolean"},
                    "unverifiable": {"type": "boolean"},
                    "why_unverifiable": {"type": "string"},
                },
                "required": [
                    "filename",
                    "turn",
                    "quote",
                    "produced",
                    "unverifiable",
                    "why_unverifiable",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["claimed_files", "confidence"],
    "additionalProperties": False,
}


def build_artifact_prompt(
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> str:
    conversation, _ = render_conversation(sub, policy, list_attachment_names=False)
    # Default prompt is the L1 reading: every file produced across the session.
    # V5 scopes the same extraction to the last assistant turn; the extra
    # paragraph is appended only then so existing tests and cached L1 runs
    # keep the byte-exact prompt they were built against.
    scope = ""
    if policy.artifact_scope == "final_response":
        last = sub.final_response_turn()
        where = f"turn {last}" if last is not None else "the model's last assistant turn"
        scope = f"""
List only files mentioned in the final response ({where}). Intermediate versions
of a file edited across turns, and files delivered in earlier turns that are not
named again in the final response, are out of scope -- do not list them. The rest
of the conversation is shown so you can tell a final-response mention from an
earlier delivery of the same file.

"""
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## Your task

List every output file this model actually produced and handed to the user.
{scope}
For each one give the filename as it appears, the turn it was delivered in, and a
short verbatim quote showing the delivery. Set `produced` to true only when the
model created the file and gave it to the user in this conversation.

Set `produced` to false, but still list the file, when the model only talked
about it: a file it proposed making, recommended, described as a possible next
step, or named as something the user already had. The distinction is delivery,
not mention.

Do not test whether any link still opens. Download links in these conversations
expire as a matter of course, and a dead link tells you nothing about whether the
file was produced.

{UNVERIFIABLE_GUIDANCE}

Set `unverifiable` true on a delivery you can see happened but whose filename you
cannot read: a turn carrying the deliverable marker, or a delivery you can only
infer sits past the truncation line. Still list it, with whatever the transcript
does say in `filename`, and name the gap in `why_unverifiable`. An unverifiable
delivery is compared against nothing -- guessing its name instead is how a file
that was uploaded correctly gets reported as absent, and this check has no middle
band to soften that.

If the model produced no files, return an empty list. That is a normal and
correct answer -- most conversations produce none.
"""


# 95's confirming pass. Runs only over the names the set subtraction already
# called missing, which is what keeps it nearly free: a conversation whose
# claimed files all match spends nothing here.
#
# What it exists to catch is the three things a string comparison structurally
# cannot see. A capture tool's own name for a screenshot
# (`Screenshot 2026-08-02 at 11.14.31.png`) against the contributor's name for
# the same image scores near zero similarity while being the same file. A fenced
# code block or an inline table the model printed in the conversation reads as a
# claimed `.py` or `.csv` that was never an attachment at all. And, in the other
# direction, `report_final.pdf` against an uploaded `report_final_v2.pdf` scores
# 0.94 and is very likely two different documents -- which is why loosening the
# fuzzy threshold without this pass would have traded false fails for missed
# ones.

ARTIFACT_CONFIRMATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claimed_filename": {"type": "string"},
                    "assessment": {
                        "type": "string",
                        "enum": [
                            "missing",
                            "renamed",
                            "superseded",
                            "not_a_file",
                            "unverifiable",
                        ],
                    },
                    "matched_upload": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["claimed_filename", "assessment", "matched_upload", "why"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["files", "confidence"],
    "additionalProperties": False,
}


def build_artifact_confirmation_prompt(
    task: Task,
    sub: ModelSubmission,
    missing: list[str],
    uploaded: list[str],
    policy: Policy = DEFAULT_POLICY,
    unmatched_uploads: list[str] | None = None,
) -> str:
    conversation, _ = render_conversation(sub, policy)
    claimed_block = "\n".join(f"  - {name}" for name in missing)
    available = uploaded if unmatched_uploads is None else unmatched_uploads
    spoken_for = [name for name in uploaded if name not in set(available)]
    uploaded_block = (
        "\n".join(f"  - {name}" for name in available)
        if available
        else "  (none -- every upload is already accounted for by another claimed file)"
    )
    accounted_block = (
        "\n\n## Uploads already accounted for, and therefore not available here\n\n"
        + "\n".join(f"  - {name}" for name in spoken_for)
        + "\n\nEach of these has already been matched to a different file this model "
        "delivered. One upload is one file: it cannot also be the file you are "
        "judging below."
        if spoken_for
        else ""
    )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## Files this model was read as claiming to deliver, which no upload matched by name

{claimed_block}

## Uploads still unaccounted for, which are what these could be

{uploaded_block}{accounted_block}

## Your task

For each claimed filename above, decide which of five things is true. A name
comparison has already been run and found no match, so the question is not whether
the strings differ -- they do -- but whether that difference means a file the model
promised is actually absent.

  - "renamed": the same file is one of the *unaccounted-for* uploads above, under a
    different name. Name it in `matched_upload`. This is common and legitimate:
    capture tools name screenshots after the date and time, upload paths add a
    duplicate suffix, contributors rename a file before attaching it, and the upload
    path re-encodes some deliverables, so a file the model produced as one format
    can arrive as an export of it in another. Judge it on what the conversation says
    the file was and what it contained, not on string similarity and not on the
    extension alone. You may only use an upload that is still unaccounted for -- if
    the list above is empty, "renamed" is not available to you, however well the
    content matches.
  - "superseded": this is an earlier state of a deliverable the model went on to
    revise, and the final state of it *is* in the uploads. An iterative session --
    twelve rounds of edits to one image, three drafts of one document -- produces
    one work product, not twelve, and the contributor is asked to preserve the work
    product. Use this only when the later version is genuinely the same deliverable
    carried forward, not when the conversation delivered two documents that happen
    to be related.
  - "not_a_file": this was never a claimed attachment. The model printed the
    content inline -- a fenced code block, a table, a snippet -- named a path it
    proposed, recommended, or told the user to create themselves, or generated an
    image the chat interface displayed inline with a download button of its own,
    which is the interface offering a copy rather than the model handing over a
    file.
  - "missing": the model genuinely handed over, or said it was handing over, a
    distinct final deliverable that is not among the uploads under any name. Two
    similar names can still be two different files: `report.pdf` and `report_v2.pdf`
    are one document each, and if the conversation delivered both as separate
    deliverables, one being uploaded does not account for the other.
  - "unverifiable": you cannot tell from what you were shown.

Say why in `why`, in one sentence, citing the conversation where it helps. Leave
`matched_upload` empty for anything other than "renamed".

Only "missing" is counted against the contributor, so an assessment you are not
confident in should be "unverifiable" rather than a guess in either direction. But
"missing" is the right answer when it is the right answer: a final deliverable the
user was handed and that nobody uploaded is exactly what this check exists to find.
"""


# ---------------------------------------------------------------------------
# 110 - key turn justification, one call per task
# ---------------------------------------------------------------------------

KEY_TURN_JUSTIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "describes_selected_turn": {"type": "boolean"},
        "claims_are_accurate": {"type": "boolean"},
        "connects_to_core_value": {"type": "boolean"},
        "unverifiable": {"type": "boolean"},
        "why_unverifiable": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "quote": {"type": "string"},
                    "unverifiable": {"type": "boolean"},
                    "why_unverifiable": {"type": "string"},
                },
                "required": ["issue", "quote", "unverifiable", "why_unverifiable"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "describes_selected_turn",
        "claims_are_accurate",
        "connects_to_core_value",
        "unverifiable",
        "why_unverifiable",
        "issues",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_key_turn_justification_prompt(
    task: Task, policy: Policy = DEFAULT_POLICY, slot: str | None = None
) -> str:
    """Judge one key-turn justification.

    `slot=None` is the L1 single-call path: `task.key_turn` against model A's
    conversation. A named slot is the V5 path -- one call per competing model.
    Existing callers that omit `slot` keep the prompt they have always had.
    """
    if slot:
        sub = task.submission(slot)  # type: ignore[arg-type]
        kt = sub.key_turn if sub is not None else task.key_turn
    else:
        sub = task.model_a or task.model_b
        kt = task.key_turn
    conversation = render_conversation(sub, policy)[0] if sub else "(unavailable)"
    return f"""## The conversation

{conversation}

## The turn the contributor selected as the key turn

Turn {kt.turn_index}

## Their justification for that choice

{' '.join((kt.justification or '').split()) or '(empty)'}

## Your task

Judge the justification, not the choice.

A key turn is the turn where the user expects the model to deliver its core value
and produce the primary deliverable. Three questions:

  - `describes_selected_turn`: does the justification describe the turn actually
    selected, rather than some other turn?
  - `claims_are_accurate`: is everything it says about that turn factually true
    of the conversation?
  - `connects_to_core_value`: does its reasoning connect the turn to delivering
    core value or the primary deliverable, rather than only asserting importance?

List any other defect in `issues` with a quote. The bar here is deliberately low:
anything incorrect, misaligned, or otherwise wrong counts, even if minor.

{UNVERIFIABLE_GUIDANCE}

All three questions are asked about turn {kt.turn_index} specifically.
If that turn is not in the conversation above -- cut at the truncation line, or
present only as a deliverable whose content is absent -- set `unverifiable` true,
say which in `why_unverifiable`, and answer the three questions true. You cannot
tell whether a justification describes a turn you were not shown, and a false
answer there is a defect recorded against the contributor on evidence that reached
you as a marker. Where you can read the turn but one particular claim turns on
something you were not shown, list that claim in `issues` with `unverifiable` true
instead of leaving it out.

Do not judge whether the selected turn is the right one -- a different check
scores that. A justification can be sound reasoning about the wrong turn, and
that combination must score clean here.
"""


# ---------------------------------------------------------------------------
# 220 - rubric autofail, one call per task plus a second opinion
# ---------------------------------------------------------------------------

AUTOFAIL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "autofail_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "why_outcome_is_useless": {"type": "string"},
                },
                "required": ["criterion_id", "quote", "why_outcome_is_useless"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["autofail_criteria", "confidence"],
    "additionalProperties": False,
}


def _rubric_block(task: Task) -> str:
    return "\n".join(
        f"  [{c.criterion_id}] {' '.join(c.text.split())}" for c in task.rubric
    ) or "(empty rubric)"


def build_autofail_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""## The prompts the user sent, in order

{_numbered_prompts(task)}

## The rubric the contributor wrote

{_rubric_block(task)}

## Your task

Find any criterion that would make the outcome worthless.

The test is severe and specific: could a response satisfy every other criterion
perfectly and still leave the user worse off than if nothing had been done,
because it did this one thing? Typical cases are irreversible or misdirected side
effects -- sending to the wrong recipient, sending something that should have
stayed a draft, deleting or overwriting data the user needed, acting on the wrong
account.

This is not the place for criteria that are merely wrong, poorly worded, too
strict, unsupported by the prompt, or even actively unhelpful. Those are real
defects and other checks score them. Reserve this one for criteria that destroy
the outcome.

For each finding, quote the criterion verbatim and state plainly why a response
following it produces a useless result. If nothing meets the bar, return an empty
list. An empty list is the expected answer for almost every rubric.
"""


def build_autofail_second_opinion_prompt(
    task: Task, criterion_id: str, quote: str, rationale: str
) -> str:
    return f"""## The prompts the user sent, in order

{_numbered_prompts(task)}

## The full rubric, for context

{_rubric_block(task)}

## The criterion another auditor flagged as an autofail

[{criterion_id}] {' '.join(quote.split())}

## Their reasoning

{' '.join(rationale.split())}

## Your task

Decide independently whether this criterion really destroys the outcome.

Confirming it fails the entire task outright, and there is no lesser band to
absorb a mistake, so confirm only if you would stake the task's verdict on it.
The bar is that a response satisfying every other criterion perfectly would still
be worthless because of this one.

Reject the finding if the criterion is merely wrong, overly strict, unsupported
by the prompt, or unhelpful. Those are defects other checks already score, and
confirming one here punishes the same mistake twice.
"""


# ---------------------------------------------------------------------------
# 280 / 310 - relevant turns, one call per model per check
# ---------------------------------------------------------------------------

RELEVANT_TURNS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "incorrect_turns": {"type": "array", "items": {"type": "integer"}},
                    "unverifiable_turns": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "why_unverifiable": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "item_id",
                    "incorrect_turns",
                    "unverifiable_turns",
                    "why_unverifiable",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["judgments", "confidence"],
    "additionalProperties": False,
}

# Turn citations are the check where invisible evidence bites hardest: the cited
# turn number is often the very thing the render budget cut, and a turn the judge
# cannot open looks exactly like a turn that does not support the claim.
TURN_ABSTENTION_INSTRUCTION = """
Put a cited turn in `unverifiable_turns`, and not in `incorrect_turns`, whenever
you cannot open it: it sits past the truncation line, its middle was omitted, or
it is present only as a deliverable whose content the transcript does not carry.
Name which in `why_unverifiable`. A turn you were not shown is not a wrong
citation -- it is a turn this audit failed to fetch, and the citation may be
exactly right against the conversation the contributor actually read. Every turn
you list there is dropped from the count rather than scored, so nothing is lost by
being honest about what you could not see.
"""


def build_criterion_turns_prompt(
    task: Task,
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
    score: int = 0,
) -> str:
    """Checks 280 (score 0) and 281 (score 1). Population is the contributor's
    own ratings at that score, not a corrected set."""
    conversation, _ = render_conversation(sub, policy)
    text = {c.criterion_id: c.text for c in task.rubric}
    items = "\n\n".join(
        f"  [{r.criterion_id}] {' '.join(text.get(r.criterion_id, '(unknown)').split())}\n"
        f"      turns cited: {r.relevant_turns or '(none)'}"
        for r in task.criterion_ratings
        if r.model == sub.model and r.score == score
    )
    if score == 1:
        heading = "The criteria this model passed, and the turns the contributor cited"
        decide = (
            "For each criterion, decide whether each cited turn actually shows "
            "the criterion being met."
        )
        correct = (
            "A turn is correct when it contains the behaviour the criterion is "
            "about -- the place a reader would look to see the criterion met."
        )
        judge = (
            "Judge citation only. Whether the contributor was right to score the "
            "criterion 1 is a separate check, and these turns are being audited "
            "against the rating they actually gave. Even where you think the score "
            "is wrong, judge whether the turn supports the claim they made."
        )
    else:
        heading = "The criteria this model failed, and the turns the contributor cited"
        decide = (
            "For each criterion, decide whether each cited turn actually shows "
            "the failure."
        )
        correct = (
            "A turn is correct when it contains the behaviour the criterion is "
            "about -- the place a reader would look to see the criterion unmet."
        )
        judge = (
            "Judge citation only. Whether the contributor was right to score the "
            "criterion 0 is a separate check, and these turns are being audited "
            "against the rating they actually gave. Even where you think the score "
            "is wrong, judge whether the turn supports the claim they made."
        )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## {heading}

{items or '(none)'}

## Your task

{decide}

{correct} A turn is incorrect when it
is unrelated to the criterion, or when it belongs to the other model's
conversation.

List only the incorrect ones in `incorrect_turns`, by number. A criterion whose
cited turns are all correct gets an empty list.

{UNVERIFIABLE_GUIDANCE}
{TURN_ABSTENTION_INSTRUCTION}

{judge}
"""


def build_dimension_turns_prompt(
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> str:
    """Check 310. Judged against the rating *and* the justification it carries."""
    conversation, _ = render_conversation(sub, policy)
    items = "\n\n".join(
        f"  Dimension: {d.dimension}\n"
        f"      rating: {'N/A' if d.not_applicable else d.rating}\n"
        f"      justification: {' '.join((d.justification or '(empty)').split())}\n"
        f"      turns cited: {d.relevant_turns or '(none)'}"
        for d in task.dimension_ratings
        if d.model == sub.model
    )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## The contributor's dimension ratings, with their cited turns

{items or '(none)'}

## Your task

For each dimension, decide whether each cited turn supports the specific claim
the justification makes.

The bar is higher than topical relevance. A turn can be about the right subject
and still fail to show what the justification says it shows -- if the
justification cites a turn as evidence the model ignored a constraint, that turn
has to be where the constraint was ignored. Mark it incorrect when it does not
support that specific claim, when it is unrelated to the dimension, or when it
belongs to the other model's conversation.

Where the claim is about what the model did -- that it repeated itself, ignored
the question, dropped a constraint -- the support has to be the MODEL text at
that turn. The user side was written by the contributor whose citation you are
checking, so a user turn complaining the model repeated itself is not evidence
that it did. Read the model's own words before you accept the citation. Turn
numbers here count exchanges, so check the model turn immediately before the
cited one too, and treat an off-by-one citation of a claim the model text does
bear out as correct.

List only incorrect turns in `incorrect_turns`, by number, and set `item_id` to
the dimension name.

{UNVERIFIABLE_GUIDANCE}
{TURN_ABSTENTION_INSTRUCTION}

Judge citation only. Whether the rating itself is right is a separate check.
"""


# ---------------------------------------------------------------------------
# 450 - justifications, one call per model plus one for the ranking
# ---------------------------------------------------------------------------

JUSTIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "justifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "rated_value": {"type": ["integer", "null"]},
                    "contradicts_verdict_claims": {"type": "integer"},
                    "is_generic": {"type": "boolean"},
                    "is_skewed": {"type": "boolean"},
                    "inaccurate_primary_claims": {"type": "integer"},
                    "inaccurate_secondary_claims": {"type": "integer"},
                    "unsupported_claims": {"type": "integer"},
                    "inaccurate_evidence": {"type": "integer"},
                    "misconstrued_evidence": {"type": "integer"},
                    "unverifiable_claims": {"type": "integer"},
                    "unverifiable": {"type": "boolean"},
                    "why_unverifiable": {"type": "string"},
                    "specifics_quoted": {"type": "array", "items": {"type": "string"}},
                    "contradicting_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "item_id",
                    "rated_value",
                    "contradicts_verdict_claims",
                    "is_generic",
                    "is_skewed",
                    "inaccurate_primary_claims",
                    "inaccurate_secondary_claims",
                    "unsupported_claims",
                    "inaccurate_evidence",
                    "misconstrued_evidence",
                    "unverifiable_claims",
                    "unverifiable",
                    "why_unverifiable",
                    "specifics_quoted",
                    "contradicting_quotes",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["justifications", "confidence"],
    "additionalProperties": False,
}

# Every count in the 450 schema feeds a threshold that is tight enough for a
# definitional slip to become a false fail, so each term is defined with the
# distinction that matters and the spec's own floor example is quoted verbatim.
JUSTIFICATION_RUBRIC = """
Count these independently for each justification. They overlap by design; a
single sentence can be both inaccurate and unsupported, and you should count it
under both.

**contradicting_quotes** — fill this first, before counting
`contradicts_verdict_claims`. Copy out, verbatim, each supporting claim you say
argues against the rating it defends. Leave the list empty if you cannot point
at one.

**contradicts_verdict_claims** — the spec's wording is "at least 1 supporting
claims does not logically defend the verdict or contradict it". Count only
claims you copied into `contradicting_quotes`: a claim offered in support that
pulls the other way, such as a justification for a rating of 1 whose whole
argument is that the dimension never applied.

Three things are not this condition. A justification for a middling rating that
names only strengths, or one for a high rating that names only shortcomings, is
not contradicting itself — a mid rating is defended by either side, and the
question "why not a point higher?" is one the form never asks. A concession is
not a supporting claim: "although it produced substantial material, it described
the deliverable rather than building it" argues for the low rating, it does not
contradict it. And a claim that is simply false is `inaccurate_*`'s business;
counting it here as well punishes one mistake twice.

**specifics_quoted** — fill this first, before deciding `is_generic`. Copy out,
verbatim, every span of the justification that names something only this
conversation contains: a product, tool, file, link, or proper noun; a number,
date, or quantity; a turn reference; a quoted phrase; or a named behaviour the
model is said to have performed or bungled. Copy the contributor's words, not
your paraphrase. Leave the list empty only when there is genuinely nothing of
that kind to copy.

**is_generic** — the spec's wording is "does not clearly and specifically
explain why a rating/ranking was selected". Set it true only when
`specifics_quoted` came out empty, because a justification that names a
checkable particular has explained why. "The response was good and helpful" is
generic; "it invented the link debatemarket.com, which does not exist" is not,
and stays not generic even if you think the point is wrong, thin, or badly
argued — wrongness is `inaccurate_*`'s business, not this one.

Two things are not genericness. Brevity: the form asks for 2-3 sentences, so a
short justification that names one real particular is specific and a contributor
is never penalised for stopping there. And incompleteness: the absence of some
further observation you would have made, of a contrast with a hypothetical worse
response, or of an explanation of what made this conversation *stand out*, is
not a defect. Judge what is on the page, not what else could have been written.
The bar is lower still for a rating of 5, where less argument is needed, but
there must be some "why".

**is_skewed** — true when the weighting is wrong in a way that drives the wrong
rating: a trivial blemish treated as decisive, or a serious failure waved
through. A justification that misweighs something but still lands on a defensible
rating is not skewed.

**inaccurate_primary_claims** — the main assertion the rating rests on is
factually wrong about the conversation. One is enough to fail, so reserve this
for the load-bearing claim.

**inaccurate_secondary_claims** — a subordinate observation is factually wrong.

**unsupported_claims** — assertions the justification never backs up, with no
quote, turn reference, or specific detail. Judge support within the justification
itself, not by whether you could find support in the conversation. This is the
category for an assertion that is merely unbacked. If you check it against the
conversation and find it false, score it as an inaccurate claim instead, primary
or secondary by the rule below.

**inaccurate_evidence** — a *citation* that does not hold up: a quote whose
words do not appear in the conversation, or a turn reference that does not
contain what is attributed to it. One is enough to fail, and that severity is
why this condition is confined to citation. The defect it names is pointing at
the record and misreporting what is there, which no reader can catch without
going back to the turn, and which a contributor should never produce.

Because one instance fails the task, satisfy yourself that the citation is
actually fabricated before booking it, and rule out the two ways a sound one
can look broken.

The first is numbering. Turn numbers count exchanges, and a render can group a
multi-message reply differently from the interface the contributor read, which
shifts every later number by one or two. Before calling a turn reference wrong,
read the turns immediately either side of it. If the content described is
there, the citation is correct and you must not book it — a number that is off
by one against a claim the surrounding text bears out is a numbering artefact,
not a misreport. It is a defect only when the claim depends on that specific
turn being the one, as a claim about ordering or about which turn came first
does.

A citation past the last turn of this conversation is the same artefact, not a
worse one. The two models are rated side by side and usually ran to different
lengths, so a number that overshoots this model's final turn is nearly always
carried across from the other model's citation list. Book nothing for it on its
own. Read the model's last turns, and treat the citation as sound if they carry
what is attributed; only if the claim is found nowhere in the conversation does
it become a defect, and then it is an inaccurate claim rather than this.

A surplus citation is likewise not a defect. Where a justification cites several
turns and the remaining ones already carry the claim, an extra number that does
not is redundant, not fabricated: nothing has been misreported to a reader who
follows the other citations. Book this only when the unsupported turn is the
sole support offered for the point being made.

The second is wording. This condition asks whether the support is invented, not
whether it was transcribed exactly. A contributor who paraphrases, trims, or
tidies a quote while pointing at the right turn, and whose paraphrase carries
what that turn actually says, has cited accurately: book nothing. Reserve this
for a quote whose substance is not in the cited turn at all, words attributed
to a speaker who never said them, or a reference to a turn that does not
discuss the matter. If the turn says the thing and the contributor rendered it
loosely, that is not fabricated evidence.

A bare factual assertion about the conversation that the transcript contradicts
is false, but it is not this. Score it under `inaccurate_primary_claims` when
the rating rests on it and `inaccurate_secondary_claims` when it is
subordinate. The same false statement must not fail a task or pass it according
to whether the contributor happened to attach a quotation mark or a turn number
to it; what decides its weight is how much the rating leans on it. So a
justification whose central argument is sound and which gets one incidental
detail wrong is not a failing justification, while one whose case collapses
without the false claim is.

Decide that by deletion, not by tone. Strike the false claim out and read what
is left: if the rating as written still has its stated support, the claim was
secondary; if the reason for the score has gone with it, it was primary. Apply
this even when the claim is one sentence among several, and even when the
contributor sounds measured — a hedge around a false claim does not demote it.
Two shapes are primary almost by construction, and both have been filed as
secondary here: the only substantive claim a justification makes, and the
claim a justification gives as its reason for departing from full marks. If a
score of 4 rests on one asserted shortcoming and that shortcoming did not
happen, nothing supports the 4, and the finding is primary.

`inaccurate_secondary_claims` is for the genuinely incidental — a
misremembered turn number beside an accurate point, a detail the argument
never needed. Reaching for it because a claim feels minor, or because scoring
primary would fail the task, is the error to avoid: the weight of the
consequence is the gate's business, not yours.

Note what this rule does *not* license. It sharpens which bucket a defect goes
in; it never licenses leaving one out. Every inaccuracy you find is recorded
somewhere — primary if the rating rests on it, secondary if not — and a claim
you are unsure about is a secondary one, not an omitted one. The counts on this
schema are a record of what you saw, and a finding you decline to write down is
the one failure mode none of the thresholds below can recover from. Where you
are torn between two buckets, pick one and say why; do not resolve the tension
by reporting neither.

Before scoring an inaccuracy of any of these three kinds, find the turn the
claim would have to rest on and quote it, so the judgement rests on the record
rather than on your memory of it. Where no such turn exists, say so.

**misconstrued_evidence** — a real, accurately reproduced quote read to mean
something it does not. Two are needed to fail.

Where the quote admits more than one reading and the reading the contributor
offered is a reasonable one, it is not misconstrued, even if you would have read
it differently: the spec instructs that a contributor presenting a reasonable
argument has their interpretation accepted. Count this only where no sound
reading of the quote supports what is said about it.

The last two are the pair most often confused. Ask whether the citation itself is
wrong (inaccurate) or whether the citation is right and the reading of it is
wrong (misconstrued).

Three situations recur in this project and are settled by the record rather than
by charity. In each, the trap is to notice the shortfall, accept the
contributor's gloss on it, and score nothing.

*A claim that the rest of the conversation is clean.* Justifications routinely
carry a sweeping negative: "the model does not make any false claims", "apart
from a minor slip at turn 20 it was otherwise faithful", "everywhere else it was
unusually honest". This is a claim about every turn you were shown, and it is
where a generous rating is usually anchored, so it is almost always primary. It
is not softened by the concession in front of it — conceding one error is a
claim that there are no others, and it is false the moment you find a second.
Do not accept it because it is broad or because checking it is work: read the
turns and look for a counterexample. One clear instance of the thing the
contributor says never happened — a fabricated citation elsewhere, a fictional
figure treated as real, a second unfaithful passage — makes the claim inaccurate,
primary where the rating rests on it.

Two bounds keep this honest. The counterexample has to be in the transcript in
front of you and has to be the same kind of defect the claim denies; a different
weakness does not falsify a claim about faithfulness. And where the sweep is
decoration rather than support — the rating stands on specifics and would stand
without the flourish — it is secondary rather than primary. Both bounds move a
finding down a tier; neither deletes it.

*A deliverable the user asked for and did not get.* Where the user asked for a
file, document, export, or other artifact and the model returned the material as
chat text, a description of how to build it, or instructions for producing or
converting it themselves, the artifact was not delivered. A justification
asserting that the request was satisfied, that every deliverable was met, or that
the gap was resolved with a workable alternative is contradicted by the
transcript: score it an inaccurate claim, primary where the rating rests on it
and secondary where it does not. That the model was candid about the limitation, and that what
it returned would let a determined user assemble the artifact, do not make the
claim accurate. They bear on how serious the miss was, which is the rating's
business and not yours.

*A re-prompt the user should not have had to send.* Where the user restates,
narrows, or repeats a request because the previous turn did not do what was
asked, that is a correction loop, whatever its tone. A justification claiming
that no turn required a re-prompt, or that no correction was needed, is
contradicted by such an exchange. A follow-up opening genuinely new ground is not
a re-prompt; a follow-up asking again for something already requested is one,
even when the user is polite about it, and even when the earlier turn was useful
in other respects.

What a user turn establishes is bounded, and the bound matters here more than
anywhere else in this rubric. **The user side of this conversation was written by
the same contributor whose justification you are auditing.** So a user turn
proves that a correction turn was spent, and it proves what the user believed had
happened. It is never evidence of what the model actually did. A justification
claiming the model repeated itself, ignored the question, dropped a constraint or
produced nothing is a claim about the model's text, and only the model's text at
that turn can confirm or refute it — a user complaining "you just repeated
yourself" does not make it so, and treating the complaint as proof lets the
contributor corroborate their own claim.

So before you accept a claim of that shape, read the MODEL text at the turn it
rests on, and at the model turn immediately before it, since turn numbering here
counts exchanges and citations land one either side often. Where that text plainly
does the thing the justification says it failed to do, the claim is false: score
it primary where the rating leans on it, secondary where it does not. Where the
model's text is absent from this transcript, that is `unverifiable_claims` and not
an inaccuracy. And hold the wording charity you would extend anywhere else: a
model that recycled most of a previous answer, described as having repeated it
"verbatim", is a real defect loosely worded, not a false claim.

**unverifiable_claims** — the abstention, and mandatory rather than optional. The
conversation above may be cut short at
`[... conversation truncated here; N exchanges total ...]`, a single turn may have
its middle removed at `[... N characters omitted ...]`, and a turn that delivered
a file, image, or video appears as
`[DELIVERABLE PRODUCED HERE - its content is not in the transcript, so its quality
cannot be judged from this text]`. When a claim turns on a turn past the end of
what you were shown, on a span that was omitted, or on the contents of a
deliverable you were not shown, count it here and nowhere else. Do not call it
inaccurate, do not call it unsupported, do not call it misconstrued, and do not
read it as contradicting the verdict: you did not see the evidence, and
the contributor saw the whole conversation. Only what is in front of you can be
contradicted. This costs nothing — a claim counted here is reported and then
counted neither way — and naming the turn or deliverable you would have needed in
`why_unverifiable` is what makes it reviewable.

Read the boundary of this abstention as strictly as its licence. It is for
material the render does not contain, and the render says so in plain markers:
the truncation line, the omitted-characters note, the deliverable placeholder.
Absent one of those, the conversation in front of you is the whole conversation,
and a thing that is not in it is not there. So a claim that the model connected
a service, ran a test, produced a file or said something, where a complete
transcript contains no such thing, is **not** unverifiable — it is
`unsupported_claims`, or inaccurate where the transcript positively shows
otherwise. "I searched and could not find it" is a finding, not an abstention;
only "I was not shown the place it would be" is an abstention. This bucket costs
nothing precisely because it is narrow, and using it for claims you simply could
not corroborate empties the audit: on the last batch it was reached 395 times
against 279 justifications flagged with any issue at all.

**unverifiable** — the whole-justification form of the same abstention. Set it
true when the justification defends a rating of something this transcript does not
contain at all: the deliverable itself, or a stretch of conversation that sits
entirely past the truncation line. Then leave every count above at zero. A
justification you could not read is not a clean justification and not a defective
one, and this flag is how it leaves the audit's population instead of being
recorded as either.

Two habits to hold to throughout. Where a justification admits more than one
reading and one of them is coherent, take that one — the spec instructs that a
contributor presenting a reasonable argument has their interpretation accepted.
That latitude is over what the contributor's words mean, not over what happened:
where the transcript settles a question of fact, the fact governs, and a
sympathetic account of why the contributor might have seen it otherwise is not a
second reading. And judge the justification, not the rating or the conversation:
a point you would have argued differently, or would not have made at all, is not
an issue under any of these counts.
"""


def _weighted_rubric_block(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    """The task's own requirements, with weights, for the justification audits.

    A contributor rates against the rubric, and a judge holding only the
    conversation cannot tell a claim that quotes a requirement from one that
    invents a standard. It resolves that the wrong way: on task
    6a7d3a48a9485b6a031cb312, which human QC scored 5, the contributor faulted a
    model for returning no Google Drive link -- criterion 811b1f24, weight 4 --
    and all five draws called it an inaccurate primary claim, which is a severe
    condition and failed a clean task on its own. The rubric is what makes that
    claim readable, and it is missing from no other stage's prompts.

    Weights are printed because they are the task's ranking of its own
    requirements, and a justification that leans on a weight-5 requirement is
    making a stronger argument than one leaning on a weight-1.

    Over `policy.max_rubric_prompt_chars` the heaviest criteria are kept and the
    remainder dropped with a count. The cap is a guard and not a routine trim:
    across the 60-task sheet22 run the 15 tasks carrying a rubric held 18 to 40
    criteria and rendered at 3.6k to 7.8k characters, none of them near it.
    """
    if not task.rubric:
        return "(no rubric was recorded for this task)"

    lines = [
        f"  [{c.criterion_id}] (weight "
        f"{c.weight if c.weight is not None else 'unstated'}) "
        f"{' '.join(c.text.split())}"
        for c in task.rubric
    ]
    if sum(len(line) + 1 for line in lines) <= policy.max_rubric_prompt_chars:
        return "\n".join(lines)

    order = sorted(
        range(len(lines)),
        key=lambda i: (-(task.rubric[i].weight or 0), i),
    )
    keep: set[int] = set()
    used = 0
    for i in order:
        if used + len(lines[i]) + 1 > policy.max_rubric_prompt_chars:
            continue
        keep.add(i)
        used += len(lines[i]) + 1
    dropped = len(lines) - len(keep)
    return "\n".join(
        [lines[i] for i in sorted(keep)]
        + [f"  ({dropped} further criteria of lower weight omitted for length)"]
    )


RUBRIC_FRAMING = """Each line is a requirement the response was expected to meet, with the weight the
task placed on it. The contributor rated and ranked against this rubric, so a
justification that faults a model for missing one of these requirements is
grounded in the task's own standard -- even where the conversation never mentions
the requirement and the user never asked for it in so many words. That is not an
invented standard, and it is not an inaccurate claim.

What the rubric settles is what was required. Whether the model actually did it
is still a question for the transcript."""


def build_dimension_justification_prompt(
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> str:
    conversation, _ = render_conversation(sub, policy)
    items = "\n\n".join(
        f"  Dimension: {d.dimension}\n"
        f"      rating: {'N/A' if d.not_applicable else d.rating}\n"
        f"      turns cited: {d.relevant_turns or '(none)'}\n"
        f"      justification: {' '.join((d.justification or '(empty)').split())}"
        for d in task.dimension_ratings
        if d.model == sub.model
    )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## The rubric the contributor graded against

{_weighted_rubric_block(task, policy)}

{RUBRIC_FRAMING}
{_golden_answer_block(task)}
## The contributor's dimension ratings and justifications

{items or '(none)'}

## What to count

{JUSTIFICATION_RUBRIC}

{UNVERIFIABLE_GUIDANCE}

## Your task

Audit each justification separately and report counts per the definitions above.
Set `item_id` to "{sub.model}::" followed by the dimension name, and `rated_value`
to the rating it defends.

Judge each justification on its own. Counts are not pooled across dimensions, and
an issue in one says nothing about another.

Judge the justification, not the rating. A rating you would not have given can
still be defended accurately and specifically, and that is a clean justification
here. Explain your counts in `reasoning`, quoting the contributor.
"""


def _likert_scale_note(
    policy: Policy, left: str = "A", right: str = "B"
) -> str:
    """Spell the preference scale out for the auditor.

    Printing a bare number invites the auditor to assume the scale it knows best.
    On the spec's bipolar 1-7 form it read 5 as the top of a 1-5 scale and so as
    "the strongest possible preference", then charged contributors with
    contradicting their verdict for describing a 5 as a slight lean toward Model
    B -- which is exactly what a 5 is. Anchors must be the pair's own slots:
    D-vs-A is 1 = D, not 1 = A.
    """
    if policy.preference_encoding == "bipolar_7":
        lo, hi = policy.likert_scale
        centre = (lo + hi) // 2
        return (
            f"a single {lo}-{hi} scale carrying both direction and strength: {lo} is "
            f"the strongest preference for Model {left}, {hi} the strongest for "
            f"Model {right}, and {centre} is a tie. Values next to the centre are "
            f"slight preferences, so describing one as slight agrees with it"
        )
    lo, hi = policy.preference_margin_scale
    return (
        f"a margin of {lo}-{hi} with no direction of its own; the preferred model "
        f"index names the winner and the Likert says only how far ahead"
    )


def build_ranking_justification_prompt(
    task: Task, policy: Policy = DEFAULT_POLICY, comparison: Sxs | None = None
) -> str:
    """Audit one comparison's ranking justification.

    Defaults to the task's primary comparison. Each pair's justification is
    audited against that pair's own two conversations -- judging the A-vs-C
    write-up against A and B would mark accurate claims about C unsupported.
    """
    pair = comparison or task.comparison_list()[0]
    rendered = render_comparison(
        task.submission(pair.left), task.submission(pair.right), policy
    )[0]
    return f"""## The two conversations

{rendered}

## The rubric the contributor graded against

{_weighted_rubric_block(task, policy)}

{RUBRIC_FRAMING}
{_golden_answer_block(task)}

## The contributor's comparison

This is the comparison between Model {pair.left} and Model {pair.right}.

Likert rating: {pair.likert if pair.likert is not None else '(none)'}
  The Likert is {_likert_scale_note(policy, pair.left, pair.right)}.
Preferred model: {pair.winner_slot() or '(none)'}

Justification: {' '.join((pair.justification or '(empty)').split())}

## What to count

{JUSTIFICATION_RUBRIC}

{UNVERIFIABLE_GUIDANCE}

## Your task

Audit this single ranking justification and report counts per the definitions
above. Set `item_id` to "ranking" and `rated_value` to the Likert rating.

Both conversations are rendered under a shared budget, so each side is cut sooner
than it would be alone. A comparison claim that turns on a stretch either side had
truncated is unverifiable, not inaccurate.

Judge the justification, not the comparison. Whether you would have preferred the
other model is a separate check; the question here is whether the case they made
is specific, accurate, and supported.
"""


# ---------------------------------------------------------------------------
# 470 - verdict, one call per task
# ---------------------------------------------------------------------------

VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "states_preference": {"type": "boolean"},
        "quote": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["states_preference", "quote", "confidence"],
    "additionalProperties": False,
}


def build_verdict_prompt(
    task: Task, policy: Policy = DEFAULT_POLICY, comparison: Sxs | None = None
) -> str:
    pair = comparison or task.comparison_list()[0]
    return f"""## The contributor's model comparison justification

This is the comparison between Model {pair.left} and Model {pair.right}.

{' '.join((pair.justification or '(empty)').split())}

## Your task

One question only: does this justification state a preference between these two
models?

Set `states_preference` to true if it names a winner, says one model is better in
overall terms, or concludes they are equivalent. A reasoned tie is a stated
preference. Quote the words that state it.

Set it to false only when the text describes both models and never reaches a
comparative conclusion -- strengths and weaknesses listed side by side with
nothing said about which came out ahead.

Do not consider whether the preference is correct, whether it matches the rating,
or whether it is well argued. Those are three other checks. A badly reasoned
preference is still a stated preference, and it passes here.
"""
