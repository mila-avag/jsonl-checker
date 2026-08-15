"""Prompt construction for the rubric-authoring stage.

Independence is structural here, not a request. The auditor is asked to assign a
label and a weight, and the contributor's label and weight are simply absent from
the prompt, so there is nothing to anchor on. `assert_independent` enforces that
at runtime, because a future edit that leaks them would bias every rate toward
agreement while still looking like it works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .config import DEFAULT_POLICY, Policy
from .context import numbered_prompts
from .models import RubricCriterion, Task
from .taxonomies import (
    ISSUE_DEFINITIONS,
    L1_DEFINITIONS,
    L1_LABELS,
    L2_BY_L1,
    L2_LABELS,
    WEIGHT_DEFINITIONS,
)

PER_CRITERION_CATEGORIES = tuple(
    c for c in ISSUE_DEFINITIONS if not c.startswith("missing_")
)

# The last two sentences are the QC spec's own general grading instructions, and
# they exist to stop exactly the failure mode measured on the first live run: the
# auditor found a defensible moderate issue on nearly every criterion, which drove
# check 240 to 100%. Both rules push the reading back toward the contributor.
SYSTEM_PROMPT = (
    "You are an independent quality auditor reviewing rubrics written by human "
    "contributors for a model-comparison task. You judge the rubric itself, not the "
    "models and not the prompt. Report only what you can support with a verbatim "
    "quote. An issue you cannot quote is not an issue. Prefer reporting nothing over "
    "reporting something you are unsure of. "
    "Where the criterion admits a reasonable interpretation under which it is "
    "sound, accept that interpretation and report no issue, even where you would "
    "have written the criterion differently. The wording is only one way a reading "
    "can be reasonable: what the criterion is plainly getting at counts too. "
    "Never penalise a criterion for something the prompt or the task instructions "
    "explicitly asked for."
)

MAX_PROMPT_CHARS = 12000


def _context_block(task: Task, policy: Policy) -> str:
    lines = ["## Task context", ""]
    if task.user_goal:
        lines.append(f"User's goal: {task.user_goal}")
    if task.target_deliverables:
        lines.append("")
        lines.append(
            "The contributor's own target outcome list (their work product, not "
            "a customer requirement -- do not treat as authoritative):"
        )
        lines += [f"  - {d}" for d in task.target_deliverables]
    lines.append("")
    lines.append("User prompts, in order:")
    prompts = numbered_prompts(task)
    if len(prompts) > MAX_PROMPT_CHARS:
        prompts = prompts[:MAX_PROMPT_CHARS] + "\n\n[... prompts truncated ...]"
    lines.append(prompts)
    return "\n".join(lines)


def _rubric_block(task: Task, target_id: str) -> str:
    """Every criterion's text, with no weights or labels.

    The sibling criteria are present only so `overlapping` can be detected; their
    weights and L1 labels are withheld so they cannot anchor the judgment.
    """
    lines = ["## All criteria in this rubric (for duplication checks only)", ""]
    for c in task.rubric:
        marker = "  >>" if c.criterion_id == target_id else "    "
        lines.append(f"{marker} [{c.criterion_id}] {' '.join(c.text.split())}")
    return "\n".join(lines)


def _enum_block() -> str:
    lines = ["## Issue categories (closed set - use these exact strings)", ""]
    for name in PER_CRITERION_CATEGORIES:
        lines.append(f"  - {name}: {ISSUE_DEFINITIONS[name]}")
    lines += ["", "## L1 categories (closed set), each with its L2 leaves", ""]
    for label in L1_LABELS:
        lines.append(f"  - {label}: {L1_DEFINITIONS[label]}")
        for leaf, definition in L2_BY_L1.get(label, ()):
            lines.append(f"      * {leaf}: {definition}")
    lines += ["", "## Weight scale", ""]
    for value, meaning in WEIGHT_DEFINITIONS.items():
        lines.append(f"  {value} = {meaning}")
    return "\n".join(lines)


WEIGHT_INSTRUCTION = """3. Weight this criterion. A weight is a judgement call, and the definitions above
   leave most criteria with more than one weight a careful author could defend, so
   report a range rather than a single number:
     - `weight_defensible_low` and `weight_defensible_high`: the lowest and the
       highest weight whose definition above genuinely fits this criterion, given
       the user's goal and the target deliverables. Widen the range to every weight
       that fits; narrow it only where the definitions themselves rule a weight out.
     - `weight`: the one weight you would assign yourself, inside that range.
   Do not narrow the range to your own preference. A criterion that is plainly
   decorative and one that a target deliverable names outright have narrow ranges;
   most criteria in between do not."""


CRITERION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "criterion_id": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(PER_CRITERION_CATEGORIES)},
                    "description": {"type": "string"},
                    "evidence": {"type": "string"},
                    "unverifiable": {"type": "boolean"},
                    "why_unverifiable": {"type": "string"},
                    "overlaps_with": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "category",
                    "description",
                    "evidence",
                    "unverifiable",
                    "why_unverifiable",
                    "overlaps_with",
                ],
                "additionalProperties": False,
            },
        },
        "l1_label": {"type": "string", "enum": list(L1_LABELS)},
        "l2_label": {"type": "string", "enum": list(L2_LABELS)},
        "weight": {"type": "integer", "minimum": 1, "maximum": 5},
        "weight_defensible_low": {"type": "integer", "minimum": 1, "maximum": 5},
        "weight_defensible_high": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "criterion_id",
        "issues",
        "l1_label",
        "l2_label",
        "weight",
        "weight_defensible_low",
        "weight_defensible_high",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_criterion_prompt(
    task: Task, criterion: RubricCriterion, policy: Policy = DEFAULT_POLICY
) -> str:
    return f"""{_context_block(task, policy)}

{_rubric_block(task, criterion.criterion_id)}

{_enum_block()}

## Your task

Audit exactly one criterion, marked with >> above:

  [{criterion.criterion_id}] {' '.join(criterion.text.split())}

Do three things, independently of each other:

1. List every issue this criterion has, using only the closed set of categories.
   Each issue needs `evidence`: a verbatim quote from the criterion text (or, for
   `overlapping`, from the criterion it duplicates). Quote exactly; do not
   paraphrase. For `overlapping`, also set `overlaps_with` to the id(s), from the
   list above, of every sibling criterion this one fully duplicates; leave it
   empty for every other category. If the criterion is sound, return an empty
   list.
2. Assign the single best L1 category from the closed set, based on what the
   criterion measures, then the single best L2 leaf from underneath that L1.
   The L2 must be one of the leaves listed under the L1 you chose.
{WEIGHT_INSTRUCTION}

## Calibration for not_self_contained and vague_subjective

These two categories are the easiest to over-report, because a grader can
always imagine a stricter, more explicit rewrite of any criterion. That is not
the bar. The bar is whether a competent grader holding the prompts above and
the input files can actually apply the criterion as written -- not whether it
is phrased as tightly as you would have phrased it.

Before flagging `not_self_contained`: reread every user turn above, not just
the one that introduced the topic, and reread the target deliverables. If the
value, object, or referent the criterion needs is stated anywhere in that
material, or follows from it by ordinary reading (not outside expertise), the
criterion is self-contained -- it is allowed to lean on the prompt instead of
restating the value itself. E.g. if turn 2 says "make the accent color green,"
a later criterion that says "the response uses the user-requested accent
color" is self-contained: the grader looks up two sentences, not a fact
outside the conversation. Reserve this category for when the needed fact is
genuinely absent from the prompts, the deliverables, and any input file
description alike.

Before flagging `vague_subjective`: ask whether two competent graders reading
the prompt and the criterion together would reach the same verdict on a given
response. If they would -- even though the wording is informal, could be
tightened, or leaves minor judgment calls a grader can resolve by reading the
response -- that is not vague. Flag this category only when the criterion's
own wording, read plainly and in context, does not pin down what is being
checked well enough for two graders to agree.

A criterion that points back at the prompt's own structure -- "each requested
deliverable," "the turn it is requested," "the format the user specified" --
is not vague just because it does not enumerate the set itself. Reread the
turns: if a competent grader can walk them and produce the same list of
"requested deliverables" (or formats, or constraints) another grader would,
the referent is pinned down by the conversation, not left to opinion. E.g.
"provides each requested deliverable within the turn it is requested" is
answered by checking, turn by turn, which ones ask for a concrete artifact
(a file, a table, a plan) versus an explanation or discussion -- that is a
comprehension task with a right answer, not a subjective one. Reserve
`vague_subjective` for wording where rereading the prompt does not settle it:
terms like "reasonable," "appropriate," or "good" quality that call for a
judgment call no amount of rereading resolves the same way twice.

## When you were not shown what an issue turns on

The prompts above are rendered under a budget. A trailing `[... prompts
truncated ...]` line means everything past it was cut, and you were shown none
of it. Input files the task attached are not reproduced here at all: you hold no
file contents, only whatever the prompts say about them.

An issue you could only confirm against material like that is unverifiable. Set
`unverifiable` true on it and name in `why_unverifiable` what you would have needed
-- the turn, or the file. That is the honest answer for `inaccurate` when the
correct value lives in a file you cannot open, for `counterproductive` when the
instruction it supposedly contradicts sits past the truncation line, and for
`not_self_contained` when the thing you cannot see is exactly what a grader holding
the input files would have had. Marking an issue unverifiable is expected and costs
nothing: it is recorded and then counted neither way. An issue reported from
material you were never shown is charged to somebody who had it in front of them,
and that is the outcome to avoid.

Judge only this criterion. Do not report issues belonging to other criteria, and
do not penalise it for something another criterion should cover.

Return JSON matching the schema. Set `criterion_id` to "{criterion.criterion_id}".
"""


COVERAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "missing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "criticality": {"type": "string", "enum": ["critical", "non_critical"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["requirement", "criticality", "evidence", "reason"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["missing", "confidence"],
    "additionalProperties": False,
}


def build_coverage_prompt(task: Task, policy: Policy = DEFAULT_POLICY) -> str:
    """Task-level pass for requirements no criterion covers.

    `target_deliverables` is the contributor's own list, not a customer-supplied
    checklist, so it is shown as context alongside the prompts rather than as an
    authoritative source of requirements on its own.
    """
    criteria = "\n".join(
        f"  [{c.criterion_id}] {' '.join(c.text.split())}" for c in task.rubric
    )
    return f"""{_context_block(task, policy)}

## The complete rubric

{criteria}

## Gap categories (closed set - use these exact strings)

  - critical: {ISSUE_DEFINITIONS["missing_critical"]}
  - non_critical: {ISSUE_DEFINITIONS["missing_non_critical"]}

## Your task

The rubric must reflect everything in the target deliverables. Identify
requirements that the prompts or the target deliverables clearly call for and
that NO criterion above covers.

For each gap:
  - `requirement`: the uncovered requirement, stated plainly.
  - `criticality`: "critical" if the requirement is a direct ask from the prompt
    -- an explicit result or outcome it asks for, or an implicit but objectively
    necessary ask -- such that a criterion covering it would be absolutely,
    100% essential. "non_critical" if a criterion covering it would instead be
    an important check on the process, reasoning, or logic of the content. If
    it is neither, the gap is not an error: do not report it.
  - `evidence`: a verbatim quote from the target deliverables or the prompts
    showing the requirement was asked for.
  - `reason`: why the existing criteria do not already cover it.

Be conservative. A requirement covered loosely, or covered by a criterion worded
differently, is covered. Report a gap only when no criterion addresses it at
all. A nice-to-have that is neither a direct ask nor a process or reasoning
check is not a gap at all, no matter how loosely it is covered. If the rubric
covers everything, return an empty list.

Return JSON matching the schema.
"""


# ---------------------------------------------------------------------------
# Independence guard
# ---------------------------------------------------------------------------


@dataclass
class LeakReport:
    leaked: list[str]

    @property
    def clean(self) -> bool:
        return not self.leaked


def assert_independent(prompt: str, task: Task) -> LeakReport:
    """Fail loudly if a prompt exposes the values the auditor must reproduce.

    Checks 200 and 260 measure disagreement with the contributor. If the
    contributor's label or weight reaches the prompt, agreement is manufactured
    and the resulting rates are meaningless while still looking plausible.
    """
    leaked: list[str] = []
    lowered = prompt.lower()

    for c in task.rubric:
        bound = f"[{c.criterion_id.lower()}]"
        # The criterion's own wording sits next to its id by design, and it may
        # coincide with a label name ("a safety checklist" under L1 Safety). Only
        # what the prompt adds around the wording can be a leak.
        own_text = " ".join(c.text.lower().split())
        annotations = [
            line.replace(own_text, " ") if own_text else line
            for line in lowered.splitlines()
            if bound in line
        ]
        if c.l1_label:
            # A bare label name is unavoidable: the enum itself lists all seven.
            # What must never appear is a label bound to a criterion id.
            if any(c.l1_label.lower() in line for line in annotations):
                leaked.append(f"{c.criterion_id}: L1 label bound to the criterion")
        if c.weight is not None:
            if any(
                f"weight {c.weight}" in line or f"weight: {c.weight}" in line
                for line in annotations
            ):
                leaked.append(f"{c.criterion_id}: weight bound to the criterion")

    if task.criterion_ratings:
        for marker in ("contributor rated", "contributor scored", "contributor's rating"):
            if marker in lowered:
                leaked.append(f"contributor ratings present ({marker})")
    return LeakReport(leaked=leaked)


def schema_json(schema: dict) -> str:
    return json.dumps(schema, separators=(",", ":"), sort_keys=True)
