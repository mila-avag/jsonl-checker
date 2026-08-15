"""Shared enums, defined once and injected into every prompt and gate.

The L1 list in check 200 and the L1 list referenced by check 230's reasoning must
never drift apart, so both read from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# L1 rubric criterion labels. Tab 2 row 58 of the QC spec enumerates seven; the
# authoring form contributors actually pick from offers eight, adding "Tool and
# Connector Reliability" (honeybee-l1-task-browsing, data-source-form-4607d2da02de).
#
# The eighth is included here deliberately. Auditing against the spec's seven
# would leave the auditor unable to express a label the contributor can select,
# so every criterion legitimately categorised as tool/connector reliability would
# read as an L1 disagreement on check 200 that no contributor could have avoided.
# SPEC_L1_LABELS keeps the spec's own list addressable for conformance tests.
# ---------------------------------------------------------------------------

SPEC_L1_LABELS: tuple[str, ...] = (
    "Outcome Quality",
    "Communication Quality",
    "Trust and Grounding",
    "Safety",
    "Personalization and Memory",
    "Interaction Efficiency",
    "Collaboration Quality",
)

L1_LABELS: tuple[str, ...] = SPEC_L1_LABELS + ("Tool and Connector Reliability",)

L1_DEFINITIONS: dict[str, str] = {
    "Outcome Quality": (
        "The right thing was produced or done: instructions and constraints honored, "
        "required actions landed, deliverables actually generated, and the result is "
        "correct and well-made. Includes visual and aesthetic quality where the "
        "artifact's purpose is visual. Explicitly calls out the distinction between a "
        "model-produced deliverable and a model description of how to produce the "
        "deliverable."
    ),
    "Communication Quality": (
        "How things are expressed, in the response and inside the deliverable: clarity, "
        "tone, verbosity, and format across the deliverable and the multi-turn "
        "interaction; pitched at the right level and length."
    ),
    "Trust and Grounding": (
        "Whether what the model says is true and honestly reported: claims and numbers "
        "verifiable, nothing invented, no claiming actions it did not perform, signals "
        "uncertainty when unsure."
    ),
    "Safety": (
        "Privacy protected, actions stay in-scope and authorized, access and recipients "
        "contained, no unauthorized actions."
    ),
    "Personalization and Memory": (
        "Prior context and preferences carried across turns and sessions without the "
        "user restating them."
    ),
    "Interaction Efficiency": (
        "Measurable user effort and time to reach a usable result: turns, calls, "
        "re-prompts, latency, and overhead spent recovering from errors."
    ),
    "Collaboration Quality": (
        # "This could include" is the spec's own wording and is load-bearing: the
        # three behaviours are illustrative, not the closed set a colon would imply.
        "How well the model partners with the user. This could include: eliciting "
        "intent, absorbing corrections and pivots, and taking useful initiative. "
        "Judged as usefulness toward the emergent goal rather than literal compliance "
        "with an itemized spec."
    ),
    "Tool and Connector Reliability": (
        "Calls across Gmail, Drive, Calendar, and GitHub executed end to end and "
        "returned real results, rather than erroring, timing out, or coming back empty."
    ),
}

# ---------------------------------------------------------------------------
# L2 labels, from the L1/L2 Unified Scoring Taxonomy behind the authoring form's
# `criterion_category` field (honeybee-l1-task-browsing reference.md, sourced from
# data-source-form-4607d2da02de). This previously blocked check 210, which
# returned not_evaluated rather than score a taxonomy it could not see.
#
# The source describes "25 L2 leaves" but enumerates 26. The discrepancy is the
# Tool and Connector Reliability L1, whose single L2 repeats its parent's name and
# was presumably not counted as a distinct leaf. All 26 are kept: dropping the
# duplicate-named leaf would make that L1 unlabelable at L2.
# ---------------------------------------------------------------------------

L2_BY_L1: dict[str, tuple[tuple[str, str], ...]] = {
    "Outcome Quality": (
        (
            "Instruction Following",
            "Constraints and specifications stated in the prompt are honored, "
            "including ones introduced mid-conversation.",
        ),
        (
            "Task Completion",
            "The required state change occurred in the environment (email sent, "
            "event created, file shared).",
        ),
        (
            "Artifact Delivery",
            "A usable artifact was actually produced. Known Gemini-specific loss "
            "bucket: the model returns step-by-step instructions for building the "
            "deliverable instead of generating it.",
        ),
        (
            "Artifact / Task Correctness",
            "The result is right for the task and well-made: logic sound, "
            "computations correct, artifact functional, visual quality appropriate "
            "where the artifact's purpose is visual.",
        ),
        (
            "Visual Quality",
            "Visual elements are complete and well-executed: appropriate resolution; "
            "diagrams and charts logical and easy to understand; UI navigable; "
            "generated images/videos free from defects (bad physics, distorted "
            "faces/limbs, flickering, artifacts).",
        ),
        (
            "Visual Appeal",
            "Visually appealing and tasteful; visuals included where they improve the "
            "response; artifacts add value for the persona (not \"AI slop\"); the "
            "persona would download or share this.",
        ),
    ),
    "Communication Quality": (
        ("Tone and Register", "Pitched correctly for the persona, audience, and context."),
        ("Structure and Polish", "How information is organized and presented."),
        (
            "Clarity and Conciseness",
            "Right length for the ask, no padding, preamble, or repetition.",
        ),
    ),
    "Trust and Grounding": (
        (
            "Source Faithfulness",
            "Sources and artifacts that exist are represented accurately, without "
            "distortion.",
        ),
        (
            "Anti-Hallucination",
            "Nothing invented: facts, figures, files, or citations with no underlying "
            "source.",
        ),
        (
            "Action Honesty",
            "Does not report completing actions it did not complete, and does not "
            "present placeholders as real output.",
        ),
        (
            "Uncertainty and Hedging",
            "Signals low confidence rather than guessing; calibrated rather than "
            "over-hedged.",
        ),
    ),
    "Safety": (
        (
            "Data Exposure",
            "Sensitive or personal content surfaced or included where it should not be.",
        ),
        (
            "Access and Sharing Control",
            "Permissions, visibility, and recipient lists are correct and minimal.",
        ),
        (
            "Scope and Authorization",
            "Actions stay within what was authorized, with confirmation before "
            "irreversible writes the user did not request.",
        ),
    ),
    "Personalization and Memory": (
        (
            "Profile Adherence",
            "Persona facts and standing preferences honored without the user restating "
            "them.",
        ),
        (
            "Context Awareness",
            "Information established earlier in the conversation or in a prior session "
            "recalled without repetition.",
        ),
    ),
    "Interaction Efficiency": (
        (
            "Turn and Call Economy",
            "Turns, re-prompts, and tool calls required to reach the outcome, scaled to "
            "task complexity.",
        ),
        (
            "Latency and Responsiveness",
            "Perceived latency and responsiveness of multi-turn interactions.",
        ),
        (
            "Recovery Overhead",
            "Extra turns and time spent getting back on track after an error or dead end.",
        ),
    ),
    "Collaboration Quality": (
        (
            "Goal Elicitation",
            "Questions and reframing that surface what the user actually wants when the "
            "opener is abstract.",
        ),
        (
            "Goal Verification",
            "For expensive or high-latency tasks, provide logical previews for the user "
            "to verify goal alignment before proceeding.",
        ),
        (
            "Correction and Pivot Handling",
            "Absorbs mid-task corrections and direction changes, and recovers the thread "
            "rather than stalling.",
        ),
        (
            "Initiative and Proactivity",
            "Surfaces useful options or next steps the user had not specified, without "
            "overstepping.",
        ),
    ),
    "Tool and Connector Reliability": (
        (
            "Tool and Connector Reliability",
            "Calls across Gmail, Drive, Calendar, and GitHub executed end to end and "
            "returned real results, rather than erroring, timing out, or coming back "
            "empty.",
        ),
    ),
}

L2_LABELS: tuple[str, ...] = tuple(
    name for leaves in L2_BY_L1.values() for name, _ in leaves
)

L2_DEFINITIONS: dict[str, str] = {
    name: description for leaves in L2_BY_L1.values() for name, description in leaves
}

# The L2 a criterion carries must sit under the L1 it carries, so check 210 can
# fail a leaf that exists but is parented wrong.
L1_OF_L2: dict[str, str] = {
    name: l1 for l1, leaves in L2_BY_L1.items() for name, _ in leaves
}

# ---------------------------------------------------------------------------
# Rating dimensions (6) for check 300. Tab 2 row 67.
#
# Deliberately NOT unified with L1_LABELS: this list adds "Tool & connector
# reliability" and omits Safety and Collaboration Quality. L1 labels tag rubric
# criteria; these score model behavior.
# ---------------------------------------------------------------------------

RATING_DIMENSIONS: tuple[str, ...] = (
    "Outcome quality",
    "Communication quality",
    "Trust & grounding",
    "Interaction efficiency",
    "Tool & connector reliability",
    "Memory & personalization",
)

CONDITIONAL_DIMENSIONS: frozenset[str] = frozenset({"Memory & personalization"})

# Contributors rate a seventh dimension on the same 1-5 scale, and check 300's
# list does not mention it. It is carried so the data survives ingestion intact
# and preflight does not reject a rating the form asked for, but nothing audits
# it: inventing a gate the spec never defined would publish a rate no one agreed
# to. Safety is different again -- a Yes/No concern flag with a category and a
# justification, never a 1-5 score, so there is no rating to disagree with.
UNAUDITED_DIMENSIONS: frozenset[str] = frozenset({"Collaboration quality"})

COLLECTED_DIMENSIONS: frozenset[str] = frozenset(RATING_DIMENSIONS) | UNAUDITED_DIMENSIONS

# ---------------------------------------------------------------------------
# Rubric criterion issue census (section 6.3). One value per listed issue so the
# census stays auditable rather than collapsing into three severity buckets.
# ---------------------------------------------------------------------------

Severity = Literal["major", "moderate", "minor"]

MAJOR_CATEGORIES: frozenset[str] = frozenset(
    {"inaccurate", "counterproductive", "missing_critical"}
)
MODERATE_CATEGORIES: frozenset[str] = frozenset(
    {
        "not_self_contained",
        "non_atomic",
        "overlapping",
        "vague_subjective",
        "missing_non_critical",
        "framing_double_negative",
    }
)
MINOR_CATEGORIES: frozenset[str] = frozenset({"unnecessary"})

ALL_ISSUE_CATEGORIES: frozenset[str] = (
    MAJOR_CATEGORIES | MODERATE_CATEGORIES | MINOR_CATEGORIES
)

# Categories describing criteria that do not exist. They cannot be found by a
# per-criterion loop and come from the task-level coverage pass instead.
ABSENT_CRITERION_CATEGORIES: frozenset[str] = frozenset(
    {"missing_critical", "missing_non_critical"}
)

SEVERITY_RANK: dict[str, int] = {"minor": 0, "moderate": 1, "major": 2}

# Taken from the QC spec's "Rubric Criteria Issue Classification" table, close to
# verbatim. The worked examples are load-bearing: an earlier, looser paraphrase of
# non_atomic and overlapping made the auditor flag a defensible moderate issue on
# nearly every criterion and drove check 240 to 100%. The spec's own edge cases are
# what pull it back.
#
# Two notes where the spec's table conflicts with this project:
#  * The absolute weight bands (8-10 critical, 4-7 non-critical) come from a
#    different project. Check 260 here fixes the scale at 1-5, so those numbers
#    are dropped and the definitions kept.
#  * framing/double negative is conditional in the spec -- "a criterion with a
#    negative weight is framed negatively" -- and both halves are load-bearing.
#    Check 260 saying weights *should* be 1-5 with no negatives is a statement
#    about what the contributor ought to write, not a guarantee about the data:
#    `WeightBucketMap.level_distance` exists precisely because 25 live criteria
#    across 14 tasks carry -3 to -5. The definition below therefore asks the model
#    for the phrasing half alone, and `weight_confirms_double_negative` applies
#    the weight half deterministically, because the criterion prompt withholds the
#    contributor's weight to keep checks 200 and 260 independent.
ISSUE_DEFINITIONS: dict[str, str] = {
    "inaccurate": (
        "Inaccurate/Unspecified. A criterion that includes a factually wrong claim or "
        "incorrectly calculated value. Where a criterion says \"approximately\" or "
        "equivalent, rounding or a 1% deviation from the precise value is permitted, "
        "unless precision is absolutely important in the context of the prompt's "
        "request. Ranges are allowed as long as they contain the correct value and are "
        "not unreasonably wide. Unspecified: it is unclear what aspect of the prompt is "
        "being evaluated, e.g. \"The response shows the company's cost is $500\" when "
        "there are three companies in the prompt, so it is not clear which company the "
        "criterion refers to."
    ),
    "counterproductive": (
        "A criterion that rewards something incorrect or harmful, or penalizes "
        "something correct or useful. E.g. a criterion that requires something the "
        "prompt forbade, or penalizes something the prompt asked for."
    ),
    "not_self_contained": (
        "A grader with no knowledge of the subject outside the provided prompt and "
        "input files cannot decide whether the criterion is present in the response. "
        "The grader sees only the raw information from the prompt and input files and "
        "cannot be expected to perform any analysis. "
        "E.g. \"The response calculates the tax return as $1000\" is self-contained. "
        "\"The response includes the tax return\" is self-contained: it only checks "
        "whether the topic is addressed, not the value (if the value is integral, a "
        "separate criterion must check it). \"The response calculates the correct tax "
        "return\" is NOT self-contained: the grader does not know the correct value and "
        "cannot compute it."
    ),
    "non_atomic": (
        "The criterion groups two or more UNRELATED constraints, leaving it with no "
        "clear focus on what aspect of the response it evaluates. A criterion with a "
        "single focus is atomic even if it mentions several details. "
        "E.g. \"The response must include two sections, one titled X and one titled Y\" "
        "IS atomic: its single focus is section structure. \"The response explains that "
        "the put is cheap because it is ~40% out-of-the-money, while the call is only "
        "~13% out-of-the-money\" IS atomic: its single focus is the put being cheaper."
    ),
    "overlapping": (
        "Two or more criteria fully overlap: they are duplicates phrased differently, "
        "such that when one is met the others are certainly met as well. If one "
        "criterion is a SUBSET of another, that is not an overlap. E.g. \"The response "
        "mentions X and Y\" and \"The response does not mention X\" are not fully "
        "overlapping. Each set of overlapping criteria counts as one error."
    ),
    "vague_subjective": (
        "The criterion cannot be objectively evaluated, or is phrased such that it "
        "cannot be made sense of. E.g. \"The response has a professional tone\"."
    ),
    "framing_double_negative": (
        "The criterion is framed negatively, checking for the absence of something "
        "instead of its presence. E.g. \"The response does not take the long-term "
        "capital gain into consideration\"; the positive framing is \"The response "
        "takes the long-term capital gain into consideration\". Report the framing "
        "and nothing else. This category also requires the criterion to carry a "
        "negative weight, and the contributor's weight is deliberately withheld "
        "from you; that half of the test is applied to your answer afterwards from "
        "the recorded weight, so do not try to guess it."
    ),
    "unnecessary": (
        "Criteria that are irrelevant to the topic and cannot be considered "
        "nice-to-haves either."
    ),
    "missing_critical": (
        "The rubric does not include criteria that are absolutely essential to satisfy "
        "a direct ask from the prompt. This includes criteria that contain a result or "
        "outcome in direct response to a prompt's ask, and criteria responding to "
        "implicit but objectively necessary asks. Task-level only; never attach this to "
        "an existing criterion."
    ),
    "missing_non_critical": (
        "Important criteria required to evaluate the process, reasoning, and logic of "
        "the content are missing from the rubric. Task-level only."
    ),
}

# The customer's own weight guide, verbatim from the spec's Weights/Definition
# table. The wording matters and is not paraphrasable: the customer defines 4 and
# 5 by what their *presence* contributes ("substantially elevates", "fundamental
# to task success"), where an earlier paraphrase here defined them by what their
# absence costs ("the response fails its purpose without it"). That is a stricter
# bar, and it pulled the auditor's own weights down the scale -- across the first
# live run it assigned 5 to 22 criteria where contributors assigned it to 148.
WEIGHT_DEFINITIONS: dict[int, str] = {
    1: (
        "Minor positive. A nice-to-have quality that provides a small improvement "
        "but is not essential."
    ),
    2: (
        "Moderately important. A valuable behavior or attribute that meaningfully "
        "improves the result."
    ),
    3: (
        "Important. A key quality that strongly contributes to task success and "
        "user satisfaction."
    ),
    4: (
        "Highly important. A critical success factor whose presence substantially "
        "elevates the quality of the outcome."
    ),
    5: (
        "Essential - a defining requirement. Doing this well is fundamental to task "
        "success and should heavily influence the final score."
    ),
}

# Categories the spec conditions on the contributor's weight as well as on the
# criterion's wording. The model supplies the wording half only.
WEIGHT_CONDITIONED_CATEGORIES: frozenset[str] = frozenset({"framing_double_negative"})


def weight_confirms_double_negative(weight: int | None) -> bool:
    """Whether a recorded weight meets the spec's negative-weight condition.

    A negatively framed criterion carrying a positive weight is simply a criterion
    that rewards an absence, which the spec does not call an error; the double
    negative is the combination of the two. An unrecorded weight cannot confirm the
    condition, and under-reporting is the safe direction: the alternative publishes
    a moderate error against a definition the customer did not write.
    """
    return weight is not None and weight < 0


# ---------------------------------------------------------------------------
# criterion_category parsing.
#
# The authoring form stores one lowercased string per criterion rather than two
# fields: "<l1>-<l2>: <l2 description>". Splitting on the first hyphen is not
# safe in general -- "trust and grounding-anti-hallucination: ..." carries a
# hyphen inside its L2 -- so the closed set of L1 names is matched as a prefix
# instead, longest first so no L1 that prefixes another can shadow it.
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Fold the spelling variants that carry no meaning.

    Live data contains "tool & connector reliability" beside "tool and connector
    reliability", "Outcome quality" beside "outcome quality", and two leaves
    prefixed with a stray tab. None of those are different selections.
    """
    return " ".join(text.replace("&", " and ").lower().split())


# L1 spellings that differ by more than case or an ampersand.
L1_ALIASES: dict[str, str] = {
    "memory and personalization": "Personalization and Memory",
}

# Leaves renamed between taxonomy revisions. Each of these arrives carrying the
# byte-identical description of the canonical leaf it replaces, which is what
# justifies treating them as the same selection rather than guessing from the
# name. `verify_alias_descriptions` re-checks that claim against the shipped
# definitions so a future rename cannot quietly ride in on an alias.
L2_ALIASES: dict[str, str] = {
    "task correctness and craft": "Artifact / Task Correctness",
    "artifact/task correctness and craft": "Artifact / Task Correctness",
    "action completion": "Task Completion",
    "context retention": "Context Awareness",
}

_L1_BY_KEY: dict[str, str] = {_normalise(label): label for label in L1_LABELS}
_L1_BY_KEY.update({_normalise(k): v for k, v in L1_ALIASES.items()})
_L2_BY_KEY: dict[str, str] = {_normalise(label): label for label in L2_LABELS}
_L2_BY_KEY.update({_normalise(k): v for k, v in L2_ALIASES.items()})


@dataclass(frozen=True)
class ParsedCategory:
    l1_label: str | None
    l2_label: str | None
    raw: str
    error: str | None = None

    @property
    def l2_selected(self) -> bool:
        """False when the contributor picked an L1 and stopped there.

        Distinct from an L2 this build cannot place: one is a narrower choice,
        the other is a taxonomy revision we do not hold. Neither gives check 210
        anything to compare, but only the second is worth reporting.
        """
        return self.l2_label is not None or bool(self.error and "L2" in self.error)


def parse_criterion_category(raw: str | None) -> ParsedCategory:
    """Split a form `criterion_category` into canonical L1 and L2 labels.

    The form stores `"<l1>-<l2>: <l2 description>"`, but roughly one criterion in
    fifteen departs from it: an L1 on its own, ampersands for "and", a stray tab,
    or a leaf from a taxonomy revision this build does not hold. Unrecognised
    values come back as None with an error rather than raising, because a label
    we cannot place is a finding for checks 200 and 210 to report, not a reason
    to drop the whole task.
    """
    if raw is None or not raw.strip():
        return ParsedCategory(None, None, raw or "", "empty criterion_category")

    text = " ".join(raw.split())
    key = _normalise(text)

    # An L1 on its own is a legitimate, narrower selection, not a malformed pair.
    if key in _L1_BY_KEY:
        return ParsedCategory(_L1_BY_KEY[key], None, text)

    # The closed L1 set is matched as a prefix rather than split on the first
    # hyphen, because "anti-hallucination" carries a hyphen of its own. Longest
    # first so no L1 that prefixes another can shadow it.
    for candidate in sorted(_L1_BY_KEY, key=len, reverse=True):
        if key.startswith(candidate + "-"):
            l1 = _L1_BY_KEY[candidate]
            remainder = key[len(candidate) + 1 :]
            break
    else:
        return ParsedCategory(None, None, text, f"unrecognised L1 in {text!r}")

    # The description after the colon is prose already held in L2_DEFINITIONS.
    l2_raw = remainder.split(":", 1)[0].strip()
    if not l2_raw:
        return ParsedCategory(l1, None, text)
    l2 = _L2_BY_KEY.get(l2_raw)
    if l2 is None:
        return ParsedCategory(l1, None, text, f"unrecognised L2 {l2_raw!r} under {l1!r}")
    if L1_OF_L2[l2] != l1:
        return ParsedCategory(l1, l2, text, f"L2 {l2!r} does not belong under L1 {l1!r}")
    return ParsedCategory(l1, l2, text)


def verify_alias_descriptions(samples: dict[str, str]) -> list[str]:
    """Check that each alias arrived describing the leaf it is mapped onto.

    `samples` maps an alias to the description text seen alongside it in the
    data. Returns the aliases whose description does not match the canonical
    leaf's, which are the ones where the rename assumption does not hold.
    """
    mismatched = []
    for alias, description in samples.items():
        canonical = L2_ALIASES.get(_normalise(alias))
        if canonical is None:
            mismatched.append(alias)
            continue
        if _normalise(description) != _normalise(L2_DEFINITIONS[canonical]):
            mismatched.append(alias)
    return mismatched


def severity_of(category: str) -> Severity:
    if category in MAJOR_CATEGORIES:
        return "major"
    if category in MODERATE_CATEGORIES:
        return "moderate"
    if category in MINOR_CATEGORIES:
        return "minor"
    raise ValueError(f"unknown rubric issue category: {category!r}")


# ---------------------------------------------------------------------------
# Check 450 justification issue conditions.
# ---------------------------------------------------------------------------

JUSTIFICATION_CONDITIONS: tuple[str, ...] = (
    "contradicts_verdict",
    "is_generic",
    "is_skewed",
    "is_inaccurate",
    "lacks_evidence",
    "cites_incorrect_evidence",
    "misconstrues_evidence",
)
