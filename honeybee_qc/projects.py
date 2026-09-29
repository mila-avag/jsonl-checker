"""Per-project taxonomy maps.

`ingest.py` used to hold one project's step IDs as module constants, which was
right while there was one project. There are now two, they disagree about which
step holds what, and they disagree about how many products a task compares --
so the step map becomes data and the adapter reads it.

Two profiles ship:

  * `HONEYBEE_L1` -- two products, one comparison. Exactly the constants
    `ingest.py` carried before this module existed, moved rather than changed.
  * `HONEYBEE_ASPIRATIONAL` -- three products plus a simplified trajectory, two
    comparisons, and no inputs for two of the prompt checks.

What a profile deliberately does *not* carry is any judgment: no thresholds a
check reasons with, no prompt text, no bands. It answers "where does this field
live in this project's export" and "which checks have inputs here", and nothing
else. The one exception is `simplified_max_pass_rate`, which is a property of
the project's task-construction contract rather than of the audit -- the
customer sets it per project the same way they set the step IDs.

Every step ID below was verified against live exports rather than read off a
taxonomy document: the aspirational map against 282 task responses (73 of them
complete), the L1 map against the batch it was built on. Where the two projects
share an ID they genuinely share it -- the prompt, target-outcome, and rubric
steps are byte-identical across both, as are all four of A and B's rating steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .config import DEFAULT_POLICY, THREE_POINT_WEIGHTS, Policy
from .models import ModelSlot

# How a slot's field names are built from its suffix. Holds for A, B, and C in
# both projects. The simplified trajectory follows none of it -- its run-step
# fields are `simplified_*` and its manifest fields `simp_*`, and it collects no
# key-turn call at all -- so it supplies `field_overrides` instead of a suffix
# these templates could be applied to.
FIELD_TEMPLATES: dict[str, str] = {
    "chat_download": "model_{suffix}_chat_download",
    "final_outputs": "final_outputs_{suffix}",
    "turn_number": "turn_number_{suffix}",
    "turn_link": "turn_link_{suffix}",
    "model_files": "model_generated_files_turn_{suffix}",
    "key_turn": "key_turn_index_{suffix}",
    "key_turn_justification": "key_turn_justification_{suffix}",
    "turn_attachment": "turn_attachment_{suffix}",
}


@dataclass(frozen=True)
class TrajectorySpec:
    """Where one slot's conversation, ratings, and scores live.

    `turns_step` and `dims_step` are optional because the simplified trajectory
    has neither: it is scored against the rubric but the contributor never
    attributes its criteria to turns and never rates it on the quality
    dimensions. `None` here is what makes the per-dimension checks skip it
    rather than report it empty.
    """

    slot: ModelSlot
    # Field-name suffix inside the step outputs: "a", "b", "c", or "simp".
    # Not always `slot.lower()` -- the simplified trajectory's fields are
    # prefixed `simp_`/`simplified_` rather than `_s`.
    suffix: str
    run_step: str
    manifest_step: str
    scores_step: str
    viewer_step: str
    # Key inside `responseRatings` / `evaluations`, e.g. "response-a". The
    # simplified trajectory's key is not derivable from its slot.
    rating_key: str
    turns_step: str | None = None
    dims_step: str | None = None
    # `dataSourceResults` key holding the product's display name. Absent for
    # product A, whose name no prefiller carries in either project, and for the
    # simplified trajectory, which is not a product.
    product_prefiller: str | None = None
    # Field on `run_step` holding the contributor's claim that this trajectory
    # actually delivered the final outcome.
    produced_outcome_field: str = ""
    # Field names that do not follow `FIELD_TEMPLATES`. An empty-string override
    # means the project does not collect that field for this slot, which readers
    # must treat as absent rather than as a lookup of the key "".
    field_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def in_comparisons(self) -> bool:
        return self.slot != "S"

    def field_name(self, logical: str) -> str:
        """Resolve a logical field to this slot's actual export key.

        Returns "" when the project does not collect it, so every caller has one
        thing to test rather than distinguishing a missing template from an
        overridden-to-nothing one.
        """
        if logical in self.field_overrides:
            return self.field_overrides[logical]
        template = FIELD_TEMPLATES.get(logical)
        return template.format(suffix=self.suffix) if template else ""


@dataclass(frozen=True)
class ComparisonSpec:
    """One head-to-head the project asks for.

    Aspirational asks for two, both anchored on A -- the product under test --
    so `left` is A in both and there is no B-vs-C. Nothing here assumes that;
    a project wanting the full round-robin adds a third entry.
    """

    left: ModelSlot
    right: ModelSlot
    selector_step: str


@dataclass(frozen=True)
class ProjectProfile:
    project_id: str
    # Short human name, used in reports and in the `--project` CLI value.
    key: str
    name: str

    prompt_step: str
    target_outcome_step: str
    rubric_step: str
    trajectories: tuple[TrajectorySpec, ...]
    comparisons: tuple[ComparisonSpec, ...]

    # `None` means the project's export has no such field, which is different
    # from "the field is present and empty". The checks that read these abstain
    # on the second and are disabled outright on the first.
    assigned_domain_prefiller: str | None = None
    preseeded_prompt_prefiller: str | None = None

    # Steps carrying material this project collects and the other does not.
    golden_answer_step: str | None = None
    known_information_step: str | None = None
    # Steps carrying `lint_acknowledged_{suffix}`, per slot. More than one step
    # can carry the same field across taxonomy revisions, so this is a tuple and
    # the adapter takes the first populated one.
    lint_ack_steps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # `dataSourceResults` keys for the platform's own file-manifest autorater,
    # per slot. Its payload carries `critical_error` plus expected/present/
    # missing filenames, and the contributor answers it via `lint_ack_steps`.
    lint_autorater_sources: dict[str, str] = field(default_factory=dict)

    # Whether the key turn is recorded once for the task (L1: an integer index
    # on product A's run step) or once per model (Aspirational: a boolean on
    # each slot's turn-manifest entries). Not cosmetic -- where all three
    # aspirational products name a key turn, 45 of 48 tasks name three
    # different turns, so auditing one of them as though it were the task's
    # answer would leave two unaudited and grade the wrong turn.
    key_turn_per_slot: bool = False

    # Checks with no inputs in this project. Dropped from the report entirely
    # rather than reported not_evaluated, because "we could not look" and "this
    # project does not collect the thing" are different statements and only the
    # first belongs in a task's result.
    disabled_checks: frozenset[int] = frozenset()

    # The simplified trajectory's weighted pass-rate ceiling, as a fraction.
    # `None` on projects with no simplified trajectory.
    simplified_max_pass_rate: float | None = None
    # Inclusive turn-count bounds the authoring form imposes on it.
    simplified_turn_bounds: tuple[int, int] | None = None

    # Policy fields this project sets differently, applied over the base policy
    # by `policy()`. Only for values the customer's own spec states for this
    # project -- a scale, a threshold it quotes -- and never for a judgment we
    # made ourselves, which belongs in `config.py` where it is measured and
    # argued once for everybody.
    policy_overrides: dict[str, Any] = field(default_factory=dict)

    def policy(self, base: Policy = DEFAULT_POLICY) -> Policy:
        """The base policy with this project's stated differences applied."""
        return replace(base, **self.policy_overrides) if self.policy_overrides else base

    def trajectory(self, slot: ModelSlot) -> TrajectorySpec | None:
        for spec in self.trajectories:
            if spec.slot == slot:
                return spec
        return None

    def comparison_slots(self) -> tuple[ModelSlot, ...]:
        return tuple(t.slot for t in self.trajectories if t.in_comparisons)

    @property
    def has_simplified(self) -> bool:
        return any(t.slot == "S" for t in self.trajectories)


# Fields shared by both projects. Named here so the two profiles below read as a
# diff against each other rather than as two independent tables.
_PROMPT_STEP = "step-TextCollection-e8ca527371fe"
_TARGET_OUTCOME_STEP = "step-1784053126103-0h5ldq"
_RUBRIC_STEP = "step-RubricCriteriaBuilder-rubric01"
_RUN_A = "step-TextCollection-29cf069962f8"
_RUN_B = "step-TextCollection-f7304e8fa411"
_SCORES_A = "step-RubricCriteriaRating-5665ecba649a"
_SCORES_B = "step-RubricCriteriaRating-49853f6eba5a"
_TURNS_A = "step-RubricCriteriaRating-e2c61a59aef7"
_TURNS_B = "step-RubricCriteriaRating-da2090d5dc04"
_DIMS_A = "step-ResponseTextCollection-cb06a191fca3"
_DIMS_B = "step-ResponseTextCollection-2a199aa081bb"
_VIEWER_A = "step-RubricEvaluationViewer-e8bc55c84da9"
_VIEWER_B = "step-RubricEvaluationViewer-48999ca811b6"
_MANIFEST_B = "step-RubricCriteriaBuilder-0e4891d4e48f"
_SELECTOR_AB = "step-QuantitativeModelResponseSelector-cd151dbf1b55"
_PREFILLER_PRODUCT_B = "before:0:data-source-string-prefiller-88fdb8d4174c"


HONEYBEE_L1 = ProjectProfile(
    project_id="6a567b80aaa4d140131eae6c",
    key="l1",
    name="Honeybee L1",
    prompt_step=_PROMPT_STEP,
    target_outcome_step=_TARGET_OUTCOME_STEP,
    rubric_step=_RUBRIC_STEP,
    trajectories=(
        TrajectorySpec(
            slot="A",
            suffix="a",
            run_step=_RUN_A,
            # L1 authored A's turn manifest under a different ID than
            # Aspirational does. This is the only step the two projects hold at
            # different keys while meaning the same thing.
            manifest_step="step-RubricCriteriaBuilder-bda5cf5e772d",
            scores_step=_SCORES_A,
            turns_step=_TURNS_A,
            dims_step=_DIMS_A,
            viewer_step=_VIEWER_A,
            rating_key="response-a",
            product_prefiller="before:0:data-source-string-prefiller-646efa3f869f",
        ),
        TrajectorySpec(
            slot="B",
            suffix="b",
            run_step=_RUN_B,
            manifest_step=_MANIFEST_B,
            scores_step=_SCORES_B,
            turns_step=_TURNS_B,
            dims_step=_DIMS_B,
            viewer_step=_VIEWER_B,
            rating_key="response-b",
            product_prefiller=_PREFILLER_PRODUCT_B,
        ),
    ),
    comparisons=(ComparisonSpec(left="A", right="B", selector_step=_SELECTOR_AB),),
    assigned_domain_prefiller="before:0:data-source-string-prefiller-07f6e8dd43f8",
    preseeded_prompt_prefiller="before:0:data-source-form-prefiller-41acb4777bed",
    # V5-only failing checks. Running them on L1 would fail tasks the L1 spec
    # never asked about. Shape-B report-only checks (82, 86, 271, 281, 305)
    # stay enabled: they cannot change the task verdict.
    disabled_checks=frozenset({60, 71, 72, 76, 91}),
)


# Same L1 project, later SxS-queue taxonomy. Ranking moved off
# `cd151dbf1b55` onto `15407fcb5060`, and A's turn manifest uses the
# Aspirational step id (`da9bb8c86eab`) instead of `bda5cf5e772d`. Unique
# Links is on this queue's QC form, so 91 stays enabled. Ingest still keys
# off `--project l1_sxs` rather than the shared project id, because the
# original L1 map has to keep serving the older attempts.
_L1_A = HONEYBEE_L1.trajectory("A")
_L1_B = HONEYBEE_L1.trajectory("B")
assert _L1_A is not None and _L1_B is not None
HONEYBEE_L1_SXS = replace(
    HONEYBEE_L1,
    key="l1_sxs",
    name="Honeybee L1 SxS queue",
    trajectories=(
        replace(
            _L1_A,
            manifest_step="step-RubricCriteriaBuilder-da9bb8c86eab",
        ),
        _L1_B,
    ),
    comparisons=(
        ComparisonSpec(
            left="A",
            right="B",
            selector_step="step-QuantitativeModelResponseSelector-15407fcb5060",
        ),
    ),
    key_turn_per_slot=True,
    golden_answer_step="step-TextCollection-7be9bf826e8c",
    disabled_checks=frozenset({60, 71, 72, 76}),
)


HONEYBEE_ASPIRATIONAL = ProjectProfile(
    project_id="6a84adb9124120a0ad6fa217",
    key="aspirational",
    name="Honeybee Aspirational",
    prompt_step=_PROMPT_STEP,
    target_outcome_step=_TARGET_OUTCOME_STEP,
    rubric_step=_RUBRIC_STEP,
    trajectories=(
        TrajectorySpec(
            slot="A",
            suffix="a",
            run_step=_RUN_A,
            manifest_step="step-RubricCriteriaBuilder-da9bb8c86eab",
            scores_step=_SCORES_A,
            turns_step=_TURNS_A,
            dims_step=_DIMS_A,
            viewer_step=_VIEWER_A,
            rating_key="response-a",
            # No prefiller carries A's name. A is the product under test and is
            # Gemini on every task in the project; the only "Product C"-style
            # prefiller here holds a form label, not a product name.
            product_prefiller=None,
            produced_outcome_field="model_produced_final_outcome_a",
        ),
        TrajectorySpec(
            slot="B",
            suffix="b",
            run_step=_RUN_B,
            manifest_step=_MANIFEST_B,
            scores_step=_SCORES_B,
            turns_step=_TURNS_B,
            dims_step=_DIMS_B,
            viewer_step=_VIEWER_B,
            rating_key="response-b",
            product_prefiller=_PREFILLER_PRODUCT_B,
            produced_outcome_field="model_produced_final_outcome_b",
        ),
        TrajectorySpec(
            slot="C",
            suffix="c",
            run_step="step-TextCollection-f573e56e0e5f",
            manifest_step="step-RubricCriteriaBuilder-85f9d3fc040e",
            scores_step="step-RubricCriteriaRating-b22893a182b1",
            turns_step="step-RubricCriteriaRating-e906c86a7d11",
            dims_step="step-ResponseTextCollection-6bc02462ff7a",
            viewer_step="step-RubricEvaluationViewer-a300dcaecf42",
            rating_key="response-c",
            product_prefiller="before:0:data-source-string-prefiller-d7505dde5188",
            produced_outcome_field="model_produced_final_outcome_c",
        ),
        TrajectorySpec(
            slot="S",
            # Its fields are `simp_*` on the manifest and `simplified_*` on the
            # run step, so neither `slot.lower()` nor a single suffix covers it;
            # the adapter special-cases the prefix off this value.
            suffix="simp",
            run_step="step-TextCollection-30a31298e5a9",
            manifest_step="step-RubricCriteriaBuilder-532ab3277d31",
            scores_step="step-RubricCriteriaRating-724814d00a68",
            # No turn attribution and no dimension ratings are collected for it.
            turns_step=None,
            dims_step=None,
            viewer_step="step-RubricEvaluationViewer-f253336085bf",
            rating_key="simplified-rubric01-response",
            product_prefiller=None,
            produced_outcome_field="simplified_model_produced_output",
            field_overrides={
                "chat_download": "simplified_chat_pdf",
                "final_outputs": "simplified_final_outputs",
                "turn_number": "simp_turn_number",
                "turn_link": "simp_turn_link",
                # Note the shape break: not `model_generated_files_turn_simp`.
                "model_files": "simp_model_generated_files",
                "turn_attachment": "simp_turn_attachment",
                # No key turn is collected for the simplified trajectory. It
                # exists to be scored, not to be frozen into a static-context
                # item, so there is nothing to name a freeze point for.
                "key_turn": "",
                "key_turn_justification": "",
            },
        ),
    ),
    comparisons=(
        ComparisonSpec(left="A", right="B", selector_step=_SELECTOR_AB),
        ComparisonSpec(
            left="A",
            right="C",
            selector_step="step-QuantitativeModelResponseSelector-ec730e661d1a",
        ),
    ),
    # Neither prefiller exists in this project's export -- checked as a literal
    # string across all 282 task responses. The two checks that read them are
    # disabled below rather than left to abstain on every task.
    assigned_domain_prefiller=None,
    preseeded_prompt_prefiller=None,
    golden_answer_step="step-TextCollection-7be9bf826e8c",
    known_information_step="step-TextCollection-0c0626b307df",
    lint_ack_steps={
        # Two steps carry `lint_acknowledged_a` across revisions of this
        # taxonomy: the second appears on 58 of the 73 complete tasks. Ordered
        # so the revision that is present on all of them is consulted first.
        "A": ("step-1787067837890-nr18ai", "step-ResponseTextCollection-9c4820d08478"),
        "B": ("step-1787003724568-dhkwmq",),
    },
    lint_autorater_sources={
        # Tied to the acknowledgement steps by the shared timestamp prefix in
        # the ID, which is how the platform pairs a data source to its step.
        "A": "before:0:data-source-1787067837890-472g8x",
        "B": "before:0:data-source-1787003724568-18egyr",
    },
    key_turn_per_slot=True,
    disabled_checks=frozenset(
        {
            # 70 Domain Relevance. Still a required dimension in the V5 spec, but
            # this project's export carries no assigned domain anywhere -- checked
            # as a literal string across all 282 task responses -- so there is
            # nothing to grade the prompt against. Disabled for want of an input,
            # not because the customer dropped the check, and the distinction
            # matters: if the field turns up, this comes straight back.
            70,
            # 75 Pre-seeded/Opening Prompt Consistency. Absent from the V5 spec
            # and no prefiller in the export either.
            75,
            # The three checks the V5 spec has no equivalent of. Ours would fail
            # tasks the customer's QC would pass, and the instruction is to match
            # the spec's fail behaviour exactly.
            #
            # 96 Minimum Turns. No turn-count floor appears in V5.
            96,
            # 220 Rubric Autofail. Our highest-stakes check, one finding failing a
            # task with no middle band, and V5 contains nothing like it.
            220,
            # 460 Inconsistent Ranking. Not a standalone dimension in V5, which
            # folds it into the Justifications check as one Major condition --
            # under a much more lenient bar than a direct fail.
            460,
        }
    ),
    # The customer confirmed 50% stands, against the 40% the V5 rubric text
    # states, and confirmed it is the weighted score rather than a count of
    # criteria met. Two of 73 tasks exceed it; at 40% it would be five, and by
    # criteria count seven.
    simplified_max_pass_rate=0.50,
    simplified_turn_bounds=(1, 3),
    policy_overrides={
        # "Criteria are weighted on a three-point scale: +1 (low), +2 (medium),
        # +3 (high)", quoted from the V5 rubric and confirmed against every
        # rubric in the project. See `WeightBucketMap`.
        "weight_scale": (1, 3),
        "weight_buckets": THREE_POINT_WEIGHTS,
        # V5 fails Overall Rubric Quality at "More than 15% (>15%)" and Rubric
        # Evaluation at "over 20%", where the base policy reads both as "at
        # least". The other two quality gates it does word inclusively, so only
        # these two are listed.
        "exclusive_fail_thresholds": frozenset({240, 270}),
        # "You have 5 or more major or minor disagreements with the contributor",
        # quoted from V5's Major Rating fail cell.
        "dimension_fail_total_count": 5,
        # "Every model has at least 1 Major Issue, or every comparison has at
        # least 1 Major Issue" -- V5's Justifications fail cell, which is a
        # structural rule rather than a share.
        "justification_fail_rule": "all_models_or_all_comparisons",
        # "Verify that every output file mentioned in the final response was
        # uploaded". V5 explicitly excuses intermediate versions and earlier-turn
        # files that are not part of the final response. The empty-final-section
        # structural leg is an L1 customer fail V5 does not state.
        "artifact_scope": "final_response",
        "final_outputs_section_required": False,
    },
)


# Same taxonomy as Aspirational — three products, simplified trajectory, V5
# policy overrides — on a later project id. Step IDs were confirmed identical
# by the operator; ingest still keys off the row's `project` column so reports
# keep the real id.
HONEYBEE_ASPIRATIONAL_6A90995D = replace(
    HONEYBEE_ASPIRATIONAL,
    project_id="6a90995deeeca5f9655d1fb6",
)

# Aspirational plus a fourth product (Model D) on the step that used to hold
# the simplified trajectory. No simplified run, three D-anchored SxS pairs
# (D-vs-A, D-vs-B, D-vs-C) instead of A-vs-B / A-vs-C. Dimension ratings for
# D live on a later ResponseTextCollection that is missing on some tasks --
# `_dimension_ratings` still finds `_d` annotations by suffix when the step
# is present, and returns nothing when it is not.
_ASP_A = HONEYBEE_ASPIRATIONAL.trajectory("A")
_ASP_B = HONEYBEE_ASPIRATIONAL.trajectory("B")
_ASP_C = HONEYBEE_ASPIRATIONAL.trajectory("C")
assert _ASP_A is not None and _ASP_B is not None and _ASP_C is not None

HONEYBEE_ASPIRATIONAL_D = replace(
    HONEYBEE_ASPIRATIONAL,
    project_id="6a95bca6e0021078893ae170",
    key="aspirational_d",
    name="Honeybee Aspirational + Model D",
    trajectories=(
        _ASP_A,
        _ASP_B,
        _ASP_C,
        TrajectorySpec(
            slot="D",
            suffix="d",
            run_step="step-TextCollection-30a31298e5a9",
            manifest_step="step-RubricCriteriaBuilder-532ab3277d31",
            scores_step="step-RubricCriteriaRating-f7f8de28b521",
            # This project does not collect per-criterion turn attribution for D.
            turns_step=None,
            dims_step="step-ResponseTextCollection-9bc91ae90a11",
            viewer_step="step-RubricEvaluationViewer-f253336085bf",
            rating_key="response-d",
            product_prefiller=None,
            produced_outcome_field="model_d_produced_output",
            field_overrides={
                "chat_download": "model_d_chat_pdf",
                "final_outputs": "model_d_final_outputs",
                # The D turn-list reused the simplified trajectory's field
                # names rather than growing a `_d` set.
                "turn_number": "simp_turn_number",
                "turn_link": "simp_turn_link",
                "model_files": "simp_model_generated_files",
                "turn_attachment": "simp_turn_attachment",
                "key_turn": "",
                "key_turn_justification": "",
            },
        ),
    ),
    comparisons=(
        ComparisonSpec(
            left="D",
            right="A",
            selector_step="step-QuantitativeModelResponseSelector-15407fcb5060",
        ),
        ComparisonSpec(
            left="D",
            right="B",
            selector_step="step-QuantitativeModelResponseSelector-f7fd6460a987",
        ),
        ComparisonSpec(
            left="D",
            right="C",
            selector_step="step-QuantitativeModelResponseSelector-1f67812a5209",
        ),
    ),
    disabled_checks=HONEYBEE_ASPIRATIONAL.disabled_checks | frozenset({97}),
    simplified_max_pass_rate=None,
    simplified_turn_bounds=None,
)


# L1 SxS tasks that also filed a Model C trajectory (same C step IDs as
# Aspirational). Forced via `--project l1_sxs_c` so A-vs-B is not re-graded.
# `--audit-slots C` still needs A in front of the judge for the AC pair.
_SXS_A = HONEYBEE_L1_SXS.trajectory("A")
assert _SXS_A is not None
HONEYBEE_L1_SXS_C = replace(
    HONEYBEE_L1_SXS,
    key="l1_sxs_c",
    name="Honeybee L1 SxS + Model C",
    trajectories=(
        # L1 SxS C attempts all carry `model_produced_final_outcome_a` on A's
        # run step (verified on the 223 A-vs-C pull). The base L1 A spec leaves
        # that field unmapped, which made every A claim ingest as None and hid
        # the "CB said no, conversation generated files" QC miss from check 82
        # and from the paste sheet.
        replace(_SXS_A, produced_outcome_field="model_produced_final_outcome_a"),
        _ASP_C,
    ),
    comparisons=(
        ComparisonSpec(
            left="A",
            right="C",
            # L1 C tasks use this selector (candidates response-a / response-c).
            # Aspirational's A vs C selector is ec730e661d1a and is not present here.
            selector_step="step-QuantitativeModelResponseSelector-f7fd6460a987",
        ),
    ),
)


PROFILES: tuple[ProjectProfile, ...] = (
    HONEYBEE_L1,
    HONEYBEE_L1_SXS,
    HONEYBEE_L1_SXS_C,
    HONEYBEE_ASPIRATIONAL,
    HONEYBEE_ASPIRATIONAL_6A90995D,
    HONEYBEE_ASPIRATIONAL_D,
)
DEFAULT_PROFILE = HONEYBEE_L1


def profile_for_project(project_id: str) -> ProjectProfile | None:
    for profile in PROFILES:
        if profile.project_id == project_id:
            return profile
    return None


def profile_by_key(key: str) -> ProjectProfile | None:
    for profile in PROFILES:
        if profile.key == key:
            return profile
    return None


def resolve_profile(
    project_id: str = "", key: str = "", default: ProjectProfile | None = None
) -> ProjectProfile:
    """Pick a profile from an explicit key, else the row's project id, else the
    default.

    Falls back rather than raising on an unrecognised project id: an unknown
    project is far more likely to be a new batch of an existing taxonomy than a
    new taxonomy, and the ingest adapter already reports every field it could
    not find. A hard failure here would turn "one step moved" into "nothing
    ingests".
    """
    if key:
        chosen = profile_by_key(key)
        if chosen is None:
            raise ValueError(
                f"unknown project profile {key!r}; "
                f"known: {', '.join(p.key for p in PROFILES)}"
            )
        return chosen
    if project_id:
        found = profile_for_project(project_id)
        if found is not None:
            return found
    return default or DEFAULT_PROFILE


def disabled_checks_for(project_id: str) -> frozenset[int]:
    """Checks this project's spec has no equivalent of, to drop from the report.

    An unknown project id returns empty: better to grade a check than to silently
    omit one because we could not place the row.
    """
    profile = profile_for_project(project_id)
    return profile.disabled_checks if profile else frozenset()


def drop_disabled(verdicts: list, project_id: str) -> list:
    """Strip checks this project's spec has no equivalent of from a report.

    Dropped entirely, not reported `not_evaluated`: "we could not look" and
    "this project does not collect the thing" are different statements.
    """
    disabled = disabled_checks_for(project_id)
    if not disabled:
        return verdicts
    return [v for v in verdicts if v.check_id not in disabled]
