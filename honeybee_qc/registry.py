"""Declarative registry of the active dimensions.

The contracted audit surface is the 21 custom dimensions in tab 2, rows 52-72,
and `Order of Appearance in Rubric` is the canonical check ID for those, so
reordering the report never breaks stored results.

Tab 1's `Applicable?` column cannot be used to rule a dimension out. Every one
of its 95 named standard rows reads FALSE, including `Un-auditable`, which tab 2
nonetheless grades as 1000 -- the column was never filled in, so FALSE carries
no signal either way. `Prompt / Domain Relevance` is in as 70 because the
customer's QC cites it in the field, `Prompt / Environment Context` as 85 and
`Conversation / Minimum Turns` as 96 because the audit workflow tab states both
requirements outright, and `Prompt / Pre-seeded/Opening Prompt Consistency` as
75 for the same reason as 70: the customer's QC cites it in the field too, under
its own bracketed label, with no row of its own in tab 2's custom block either.
75 sits ahead of 80 rather than beside 70 because it reads a second, distinct
source field (the pre-seed) that 70 never touches, and grouping it with the
other input-stage checks in `BLOCKS["setup_and_inputs"]` keeps every check that
runs before a model response exists together. Net-new dimensions carry a local
ID and say so in `notes`.

Prompt quality, tool-call syntax and ground-truth uniqueness are BLOCKED, not out
of scope. Tab 2's standard-dimension block was never exported: row 6 of the
Dimension Definition tab reads "(no task type and/or dimensions selected in the
previous step)" and every row beneath it is empty, so the tab omits all 95
standard dimensions rather than declining these three. Tab 1 does name them --
rows 9 and 11 give `Prompt / Unique Ground Truth` and `Prompt / Prompt Clarity
and Specificity / Unique Ground Truth` a "5 out of 5" definition each, and rows
99-103 do the same for the `Response / Tool Call *` family -- but tab 1 carries no
fail or non-fail band for any of them, and a band is what a check is built out
of. They stay unbuilt until the missing section arrives; inventing bands for them
is what produced the weight-definition and generic-justification drifts.

Known-uncovered requirements, recorded so a reader does not mistake silence for
coverage:

  * Prompt clarity and specificity (audit workflow step 2, "make sure it's
    specific and clear"). Named in tab 1 row 11, no bands anywhere.
  * Unique answer / unique ground truth (step 2, "has a unique answer"). Named in
    tab 1 rows 9 and 11, no bands anywhere.
  * The 07/31 instruction to check every prompt for LLM authorship through
    zerogpt, gptzero, quillbot and humanizeai. Four third-party web detectors,
    not implementable offline and not reproducible even online; this belongs to
    the human QC step and no check here should imply it ran.
  * "Information in the trajectories should be grounded in the universe" (step 3,
    beside the 7-turn rule). Check 85 does not cover it: 85 reads prompt text
    against the input manifest and never looks at model output, so a trajectory
    that invents an entity mid-conversation passes 85 untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scoring import Shape

Stage = int  # 1 preflight, 2 per-criterion, 3 per-rating, 4 task-level, 5 aggregation


@dataclass(frozen=True)
class CheckSpec:
    check_id: int
    dimension: str
    sub_dimension: str
    shape: Shape
    stage: Stage
    deterministic: bool
    evidence: tuple[str, ...]
    summary: str
    blocks_task: bool = True
    notes: str = ""


def _spec(*args, **kwargs) -> CheckSpec:
    return CheckSpec(*args, **kwargs)


REGISTRY: dict[int, CheckSpec] = {
    70: _spec(
        70, "Prompt", "Domain Relevance", "C", 4, False,
        ("assigned_domain", "prompts"),
        "Prompt is clearly related to the assigned domain.",
        notes="Net-new: tab 1 names the dimension and states the 5-of-5 bar, but "
              "tab 2's custom block assigns it no rubric position, so 70 is a "
              "local id chosen to sort ahead of 80. Fail requires egregious "
              "misalignment; drifting emphasis is the non-fail band.",
    ),
    75: _spec(
        75, "Prompt", "Pre-seeded/Opening Prompt Consistency", "C", 4, False,
        ("pre_seeded_prompt", "prompts"),
        "Submitted opening prompt is consistent with the pre-seeded prompt.",
        notes="Net-new: the customer's QC cites this dimension in the field under "
              "its own bracketed label, but tab 2's custom block assigns it no "
              "rubric position, so 75 is a local id chosen to sort between 70 and "
              "80 -- ahead of Target Outcome, alongside the other checks that "
              "read material fixed before either model ran. Judged on underlying "
              "intent, not wording: a reworded opening that keeps the same core "
              "ask is the non-fail band, not a fail, and only a genuinely "
              "different subject or request fails. Abstains with not_evaluated "
              "when no pre-seeded prompt was recorded, which most tasks "
              "project-wide will not have.",
    ),
    80: _spec(
        80, "Target Outcome", "", "B", 4, False,
        ("prompts", "target_outcome"),
        "Target outcome list aligned with the final state the prompts expect.",
        blocks_task=False,
        notes="Final-state semantics: replay all turns applying revisions and "
              "removals before comparing. Never diff against turn 1 alone.",
    ),
    85: _spec(
        85, "Prompt", "Environment Context", "A", 4, False,
        ("prompts", "input_artifacts", "target_outcome"),
        "Prompt references only entities, files, and events in the task's universe.",
        notes="Net-new: the audit workflow tab states the requirement ('references "
              "only the entities and events in the task's universe') but tab 2 gives "
              "it no row, so 85 is a local id placed with the other input checks. "
              "Two halves: named files are compared against the input manifest with "
              "no model call, and entities/people/events are judged by the informed "
              "stage. Fail-only, and both halves abstain rather than guess when the "
              "manifest or the prompt text is missing.",
    ),
    90: _spec(
        90, "User Input", "Valid Links", "A", 1, True,
        ("trajectory_links",),
        "Valid final trajectory links present for both Model A and Model B.",
        notes="Fully deterministic. Validate the FINAL link specifically.",
    ),
    95: _spec(
        95, "Conversation", "Output Artifacts Preserved", "A", 4, False,
        ("conversations", "uploaded_artifacts"),
        "Every output file mentioned in a response was uploaded.",
        notes="Never test embedded link liveness; expired links are expected. "
              "No artifacts mentioned at all is a clean 5.",
    ),
    96: _spec(
        96, "Conversation", "Minimum Turns", "A", 1, True,
        ("conversations",),
        "Both models ran at least 7 turns",
        notes="Net-new: the audit workflow tab states the requirement ('Both the "
              "models require at least 7 turns') but tab 2 gives it no row, so 96 "
              "is a local id placed with the other input checks. Fully "
              "deterministic and fail-only. A submission whose own cited turns run "
              "past the last turn the fetch retrieved is set aside rather than "
              "counted short, so an incomplete fetch abstains instead of "
              "manufacturing a fail.",
    ),
    100: _spec(
        100, "Key Turn", "Key Turn Identification", "C", 1, True,
        ("conversations", "key_turn", "target_outcome"),
        "Key turn correctly identified; turn 1 is a structural fail.",
        notes="Turn-1 selection is decided deterministically before any judgment.",
    ),
    110: _spec(
        110, "Key Turn", "Key Turn Justification", "B", 4, False,
        ("conversations", "key_turn"),
        "Key turn justification is valid.",
        blocks_task=False,
        notes="Unusually low bar: 'or contains any issues'. Scored independently "
              "of 100 -- a correct turn can carry a bad justification.",
    ),
    200: _spec(
        200, "Rubric Criteria", "L1 Labels/Annotations", "C", 2, False,
        ("rubric",),
        "L1 labels accurate against the 7-category enum.",
        notes="Denominator counts only criteria carrying an L1 label.",
    ),
    210: _spec(
        210, "Rubric Criteria", "L2 Labels/Annotations", "B", 2, False,
        ("rubric",),
        "L2 labels accurate.",
        blocks_task=False,
        notes="Evaluated in the rubric stage alongside 200. Returns "
              "not_evaluated only when the stage itself is skipped.",
    ),
    220: _spec(
        220, "Rubric", "Rubric Autofail", "A", 4, False,
        ("rubric", "prompts"),
        "Rubric contains no outcome-destroying criterion.",
        notes="Highest-stakes single check: one finding fails the task with no "
              "middle band. Requires a mandatory second-opinion pass.",
    ),
    230: _spec(
        230, "Rubric Criteria", "Overall Rubric Quality - 10%", "C", 5, True,
        ("rubric_census",),
        "Under 10% of criteria carry major issues.",
    ),
    240: _spec(
        240, "Rubric Criteria", "Overall Rubric Quality - 15%", "C", 5, True,
        ("rubric_census",),
        "Under 15% of criteria carry major or moderate issues.",
    ),
    250: _spec(
        250, "Rubric Criteria", "Overall Rubric Quality - 20%", "C", 5, True,
        ("rubric_census",),
        "Under 20% of criteria carry major, moderate, or minor issues.",
    ),
    260: _spec(
        260, "Rubric Criteria", "Weights", "C", 5, True,
        ("rubric",),
        "Criteria weights reflect their impact on the response.",
        notes="Weights are 1-5 with no negatives; out-of-range values are a "
              "structural preflight defect, not a silently bucketed one.",
    ),
    270: _spec(
        270, "Rating", "Rubric Evaluation", "C", 5, True,
        ("rubric", "criterion_ratings", "conversations"),
        "Contributor's per-criterion ratings agree with an independent rating.",
        notes="One call per (criterion, model). Batching destroys independence.",
    ),
    280: _spec(
        280, "Rubric Ratings", "Relevant Turns", "C", 5, True,
        ("criterion_ratings", "conversations"),
        "Turns cited for score-0 criteria are genuinely relevant.",
        notes="Work list is built from the CONTRIBUTOR's score-0 set, not from "
              "hypothetical corrected ratings.",
    ),
    300: _spec(
        300, "Rating", "Correctness/Accuracy - Major Rating", "C", 5, True,
        ("dimension_ratings", "conversations"),
        "Contributor's per-dimension ratings agree with an independent rating.",
        notes="N/A dimensions are excluded from the denominator.",
    ),
    310: _spec(
        310, "Dimension Ratings", "Relevant Turns", "C", 5, True,
        ("dimension_ratings", "conversations"),
        "Turns cited for dimension ratings are relevant to rating and justification.",
        notes="Shares 280's error strings but is a separate dimension; findings "
              "are namespaced by check ID so no turn counts toward both.",
    ),
    400: _spec(
        400, "SxS Rating", "Ranking Disagreement", "C", 5, True,
        ("sxs", "conversations"),
        "Independent Likert rating agrees with the contributor's.",
        notes="Most anchoring-prone check. The independent rating runs as its own "
              "cached call so the contributor's value is never in context.",
    ),
    450: _spec(
        450, "Dimension Rating / Model Ranking", "Justifications", "C", 5, True,
        ("dimension_ratings", "sxs", "conversations"),
        "Justifications are specific, accurate, and evidenced.",
        notes="Population is every dimension justification for every model PLUS "
              "the ranking justification. Evidence bar flexes for a rating of 5, "
              "but 'this convo is good' fails regardless.",
    ),
    460: _spec(
        460, "SxS Rating", "Inconsistent Ranking", "A", 5, True,
        ("dimension_ratings", "sxs"),
        "Likert ranking is consistent with the individual quality ratings.",
        notes="Only opposite directions qualify. An inversion the justification "
              "adequately explains is not a fail.",
    ),
    470: _spec(
        470, "SxS Rating", "Verdict", "A", 4, False,
        ("sxs",),
        "Ranking justification states a preference.",
        notes="Narrowest check in the spec. Not whether the preference is right, "
              "consistent, or well-argued -- only whether it is stated. A "
              "reasoned tie counts as a stated preference.",
    ),
    1000: _spec(
        1000, "Unauditable", "", "A", 1, False,
        ("all",),
        "Task is auditable rather than spam or unusable.",
        notes="Runs first as a gate. A task merely missing trajectory links is a "
              "90 fail, not unauditable.",
    ),
}

# Natural blocks worth preserving in the pipeline.
BLOCKS: dict[str, tuple[int, ...]] = {
    "setup_and_inputs": (75, 80, 85, 90, 95, 96, 100, 110),
    "rubric_authoring": (200, 210, 220, 230, 240, 250, 260),
    "rating_accuracy": (270, 280, 300, 310),
    "comparison": (400, 450, 460, 470),
    "escape_hatch": (1000,),
}

ORDER: tuple[int, ...] = tuple(sorted(REGISTRY))

# The three checks that read one shared per-criterion census, so their
# numerators are guaranteed monotonic.
CENSUS_GATES: tuple[int, ...] = (230, 240, 250)


def specs_for_stage(stage: Stage) -> list[CheckSpec]:
    return [REGISTRY[cid] for cid in ORDER if REGISTRY[cid].stage == stage]


def deterministic_checks() -> list[CheckSpec]:
    return [REGISTRY[cid] for cid in ORDER if REGISTRY[cid].deterministic]
