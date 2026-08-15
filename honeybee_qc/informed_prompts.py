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
from .context import numbered_prompts as _numbered_prompts
from .context import render_comparison, render_conversation
from .env_context import input_manifest, universe_materials
from .models import ModelSubmission, Task

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
UNVERIFIABLE_GUIDANCE = """
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
                    "evidence_summary": {"type": "string"},
                },
                "required": [
                    "filename",
                    "prompt_quote",
                    "prompt_treats_as_available_to_read",
                    "conversation_shows_attachment_or_upload",
                    "conversation_shows_independent_content",
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
    files = ", ".join(sorted(manifest.basenames())) or "(none with a readable name)"
    outcomes = (
        "\n".join(f"  - {' '.join(o.split())}" for o in task.target_outcome) or "  (empty)"
    )
    conversations = []
    for sub in (task.model_a, task.model_b):
        if sub is not None and sub.conversation:
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

If both are false and `prompt_treats_as_available_to_read` is true, the file
exists in this conversation in name only: the user is narrating a document and
the model is going along with the narration, not reading anything. This is not a
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

Report each such reference and answer three questions about it. All three decide
whether it counts, so answer them literally rather than as a summary of your
overall impression:

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
    conversation, _ = render_conversation(sub, policy)
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## Your task

List every output file this model actually produced and handed to the user.

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
                        "enum": ["missing", "renamed", "not_a_file", "unverifiable"],
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
) -> str:
    conversation, _ = render_conversation(sub, policy)
    claimed_block = "\n".join(f"  - {name}" for name in missing)
    uploaded_block = (
        "\n".join(f"  - {name}" for name in uploaded)
        if uploaded
        else "  (the manifest records no uploads at all)"
    )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## Files this model was read as claiming to deliver, which no upload matched by name

{claimed_block}

## Every file actually uploaded with this submission

{uploaded_block}

## Your task

For each claimed filename above, decide which of four things is true. A name
comparison has already been run and found no match, so the question is not whether
the strings differ -- they do -- but whether that difference means a file the model
promised is actually absent.

  - "renamed": the same file is in the upload list under a different name. Name it
    in `matched_upload`. This is common and legitimate: capture tools name
    screenshots after the date and time, upload paths add a duplicate suffix, and
    contributors rename a file to something meaningful before attaching it. Judge
    it on type and on what the conversation says the file was, not on string
    similarity.
  - "not_a_file": this was never a claimed attachment. The model printed the
    content inline -- a fenced code block, a table, a snippet -- or named a path it
    proposed, recommended, or told the user to create themselves. Nothing was
    promised, so nothing is missing.
  - "missing": the model genuinely handed over, or said it was handing over, a file
    that is not in the upload list under any name. Note that two similar names can
    still be two different files: `report.pdf` and `report_v2.pdf` are one document
    each, and if the conversation delivered both, one being uploaded does not
    account for the other.
  - "unverifiable": you cannot tell from what you were shown.

Say why in `why`, in one sentence, citing the conversation where it helps. Leave
`matched_upload` empty for anything other than "renamed".

Only "missing" is counted against the contributor, so an assessment you are not
confident in should be "unverifiable" rather than a guess in either direction.
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
    task: Task, policy: Policy = DEFAULT_POLICY
) -> str:
    sub = task.model_a or task.model_b
    conversation = render_conversation(sub, policy)[0] if sub else "(unavailable)"
    return f"""## The conversation

{conversation}

## The turn the contributor selected as the key turn

Turn {task.key_turn.turn_index}

## Their justification for that choice

{' '.join((task.key_turn.justification or '').split()) or '(empty)'}

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

All three questions are asked about turn {task.key_turn.turn_index} specifically.
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
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> str:
    """Check 280. Population is the contributor's own score-0 criteria."""
    conversation, _ = render_conversation(sub, policy)
    text = {c.criterion_id: c.text for c in task.rubric}
    items = "\n\n".join(
        f"  [{r.criterion_id}] {' '.join(text.get(r.criterion_id, '(unknown)').split())}\n"
        f"      turns cited: {r.relevant_turns or '(none)'}"
        for r in task.criterion_ratings
        if r.model == sub.model and r.score == 0
    )
    return f"""## Conversation with {"Model " + sub.model}

{conversation}

## The criteria this model failed, and the turns the contributor cited

{items or '(none)'}

## Your task

For each criterion, decide whether each cited turn actually shows the failure.

A turn is correct when it contains the behaviour the criterion is about -- the
place a reader would look to see the criterion unmet. A turn is incorrect when it
is unrelated to the criterion, or when it belongs to the other model's
conversation.

List only the incorrect ones in `incorrect_turns`, by number. A criterion whose
cited turns are all correct gets an empty list.

{UNVERIFIABLE_GUIDANCE}
{TURN_ABSTENTION_INSTRUCTION}

Judge citation only. Whether the contributor was right to score the criterion 0
is a separate check, and these turns are being audited against the rating they
actually gave. Even where you think the score is wrong, judge whether the turn
supports the claim they made.
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
Two are needed to fail.

**unsupported_claims** — assertions the justification never backs up, with no
quote, turn reference, or specific detail. Judge support within the justification
itself, not by whether you could find support in the conversation.

**inaccurate_evidence** — a quote or turn reference that is fabricated or
factually wrong: words that do not appear, a turn that does not say what is
attributed to it. One is enough to fail.

**misconstrued_evidence** — a real, accurately reproduced quote read to mean
something it does not. Two are needed to fail.

Where the quote admits more than one reading and the reading the contributor
offered is a reasonable one, it is not misconstrued, even if you would have read
it differently: the spec instructs that a contributor presenting a reasonable
argument has their interpretation accepted. Count this only where no sound
reading of the quote supports what is said about it.

The last two are the pair most often confused. Ask whether the evidence itself is
wrong (inaccurate) or whether the evidence is right and the reading of it is
wrong (misconstrued).

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
And judge the justification, not the rating or the conversation: a point you
would have argued differently, or would not have made at all, is not an issue
under any of these counts.
"""


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


def _likert_scale_note(policy: Policy) -> str:
    """Spell the preference scale out for the auditor.

    Printing a bare number invites the auditor to assume the scale it knows best.
    On the spec's bipolar 1-7 form it read 5 as the top of a 1-5 scale and so as
    "the strongest possible preference", then charged contributors with
    contradicting their verdict for describing a 5 as a slight lean toward Model
    B -- which is exactly what a 5 is.
    """
    if policy.preference_encoding == "bipolar_7":
        lo, hi = policy.likert_scale
        centre = (lo + hi) // 2
        return (
            f"a single {lo}-{hi} scale carrying both direction and strength: {lo} is "
            f"the strongest preference for Model A, {hi} the strongest for Model B, "
            f"and {centre} is a tie. Values next to the centre are slight "
            f"preferences, so describing one as slight agrees with it"
        )
    lo, hi = policy.preference_margin_scale
    return (
        f"a margin of {lo}-{hi} with no direction of its own; the preferred model "
        f"index names the winner and the Likert says only how far ahead"
    )


def build_ranking_justification_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    comparison = render_comparison(task.model_a, task.model_b, policy)[0]
    return f"""## The two conversations

{comparison}

## The contributor's comparison

Likert rating: {task.sxs.likert if task.sxs.likert is not None else '(none)'}
  The Likert is {_likert_scale_note(policy)}.
Preferred model index: {task.sxs.winner_index if task.sxs.winner_index is not None else '(none)'}

Justification: {' '.join((task.sxs.justification or '(empty)').split())}

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


def build_verdict_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    return f"""## The contributor's model comparison justification

{' '.join((task.sxs.justification or '(empty)').split())}

## Your task

One question only: does this justification state a preference between the two
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
