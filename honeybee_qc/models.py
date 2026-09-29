"""The input contract.

Turn index convention, declared once and asserted in preflight:

  * Turn indices are 1-based.
  * A user message and the model response that answers it share one index, so an
    index identifies an exchange. This matches the rendered share pages, which
    expose exactly one user block and one response block per exchange.
  * The first turn is a user turn.

Check 100 hard-fails on "turn 1", so an off-by-one here manufactures fails. Any
adapter for a differently numbered source must renumber, not reinterpret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# "A", "B", "C", "D" are the products being compared against each other. "S" is
# the simplified trajectory: a 1-3 turn run of the same prompt, scored against
# the same rubric, whose purpose is to be *hard* -- it must land at or under a
# weighted pass-rate ceiling, because a simplified run that scores well means the
# task was too easy to discriminate between products. It is not a competitor and
# never appears in a comparison, which is why `Task.submissions()` excludes it
# and `Task.trajectories()` is the accessor that includes it.
ModelSlot = Literal["A", "B", "C", "D", "S"]
# The slots that compete. Anything ranking, preference, or inversion related is
# defined over these and never over "S".
COMPARISON_SLOTS: tuple[ModelSlot, ...] = ("A", "B", "C", "D")
Role = Literal["user", "assistant"]


@dataclass
class Turn:
    index: int
    role: Role
    text: str
    artifacts_mentioned: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnLink:
    turn: int
    url: str


@dataclass
class TurnManifestEntry:
    """One turn's contributor-recorded metadata, from the newer per-turn manifest
    steps (`ingest.py`'s `STEP_TURN_MANIFEST_A`/`_B`, a `RubricCriteriaBuilder`
    whose "criteria" are turns rather than rubric items).

    Unlike a hydrated transcript this carries only the user's side of the
    exchange -- the model's reply is never recorded here -- plus whatever the
    contributor annotated about that turn: a share link, whether the model
    produced a file there, and whether the contributor marked it the key turn.

    `share_link` is populated for product A on every turn and for the
    simplified trajectory on some; products B and C record only a single final
    link, so their manifest entries carry no per-turn link at all. Turn-level
    inspection of B and C therefore has to come from the exported transcript.
    """

    turn_number: int
    prompt_text: str
    share_link: str = ""
    has_model_files: bool = False
    is_key_turn: bool = False
    key_turn_justification: str = ""
    attachment_names: list[str] = field(default_factory=list)


@dataclass
class KeyTurn:
    """The turn before which the session would be frozen to build a static-
    context item downstream. A freeze point, not a failure marker.

    `turn_index` is None both when nothing was recorded and when more than one
    turn was flagged, because neither is a key turn and picking one of several
    would invent an answer. `flagged_turns` keeps what was actually recorded so
    the two cases stay distinguishable: empty means the contributor named none,
    and more than one means they named several, which is its own defect.
    """

    turn_index: int | None = None
    justification: str = ""
    flagged_turns: list[int] = field(default_factory=list)

    @property
    def over_flagged(self) -> bool:
        return len(self.flagged_turns) > 1


@dataclass
class ModelSubmission:
    """What the contributor filed for one model slot.

    Contributors supply a link per turn plus a final link, a PDF transcript, and
    any attachments the model generated. The attachments are the only way the
    audit ever sees a produced artifact: share pages render deliverables as
    players and previews with no retrievable file.
    """

    model: ModelSlot
    final_link: str = ""
    turn_links: list[TurnLink] = field(default_factory=list)
    transcript_pdf: str = ""
    attachments: list[str] = field(default_factory=list)
    # `attachments` stays a list of URLs -- the input contract's documented
    # shape, and the value everything that fetches a file's bytes keys off.
    # These two carry what the upload manifest recorded about each one,
    # looked up by that same URL, so a consumer that only wants a name or a
    # type never has to parse the opaque S3 key to get it. Absent for a URL
    # that was never in the manifest (e.g. the JSONL input contract, which has
    # no field for either), which is why every reader falls back to the URL
    # itself rather than treating a missing entry as an error.
    attachment_names: dict[str, str] = field(default_factory=dict)
    attachment_mime_types: dict[str, str] = field(default_factory=dict)
    # `attachments` is the union of the final-outputs upload and the per-turn
    # uploads, which is right for "did this file reach us at all" and wrong for
    # the one question check 95 is asked. The customer fails a task whose final
    # outcome section is empty even when every file was attached turn by turn,
    # and the union erases exactly that distinction: a task with 24 per-turn
    # uploads and nothing in the final-outcome field scored 24 claimed, 24
    # uploaded, 0 missing. These two keep the narrower fact recoverable.
    # `final_outputs_declared` says the form asked for the section at all, so
    # its absence on an older taxonomy reads as "not applicable" rather than as
    # a contributor who skipped it.
    final_output_uploads: list[str] = field(default_factory=list)
    final_outputs_declared: bool = False
    conversation: list[Turn] = field(default_factory=list)
    # How long this submission's conversation is believed to be, from evidence
    # that does not depend on our fetch of its link -- the other model's fetched
    # length, the task's declared `estimated_turns`, the contributor's own turn
    # citations. Stamped by hydration, because hydration is the only place that
    # knows what came back; left at 0 when nothing was fetched, which reads as
    # "no corroboration" and leaves every consumer at its pre-existing
    # behaviour. It exists as a field rather than a computed property because
    # the renderer is handed a submission and never the task it belongs to, and
    # the corroboration is a fact about the task.
    corroborated_exchanges: int = 0
    # Whether the share page loaded and reported the conversation unavailable.
    # Stamped by hydration for the same reason as `corroborated_exchanges`: it is
    # the only place that sees the rendered page. Deliberately narrower than "the
    # fetch failed" -- a timeout is our problem and must never reach check 90,
    # while a page that affirmatively says the conversation is gone is the
    # contributor's invalid link.
    share_page_dead: bool = False
    # Whether our own fetch of the share link failed -- a timeout, a transport
    # error, a page that returned nothing parseable. The counterpart to
    # `share_page_dead` and deliberately kept apart from it: that field is a
    # statement about the contributor's link and feeds check 90, this one is a
    # statement about our infrastructure and feeds nothing. No check reads it and
    # no check should; it exists so that a report can distinguish "we could not
    # look" from "there was nothing to see", which today are the same silence.
    # Wiring it to check 90 would fail contributors for our network.
    fetch_failed: bool = False
    # Present only on tasks authored against the turn-list taxonomy; empty on
    # every task ingested before it, which is the entire fallback -- nothing
    # downstream distinguishes "older schema" from "schema present but empty".
    turn_manifest: list[TurnManifestEntry] = field(default_factory=list)
    declared_provider: str = ""
    # This slot's own freeze point. Honeybee L1 asks for one key turn per task
    # and records it on product A's run step; Honeybee Aspirational asks for one
    # per model and records it on each slot's turn manifest, and the answers
    # genuinely differ -- where all three products name a turn, 45 of 48 tasks
    # name three different ones. So a task-level key turn cannot represent this
    # project, and a per-slot one cannot be dropped in favour of it.
    key_turn: KeyTurn = field(default_factory=KeyTurn)
    # The contributor's claim that this trajectory delivered the final outcome.
    # `None` where the project does not ask, which is not the same as False.
    produced_final_outcome: bool | None = None

    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.conversation if t.role == "assistant"]

    def final_response_turn(self) -> int | None:
        """The last assistant turn, which V5 calls the final response.

        Check 95's V5 rule only cares about files mentioned there. Falls back
        to the last turn in the contributor's manifest when the conversation
        has not been hydrated yet, because that is the same last exchange.
        """
        assistants = [t.index for t in self.conversation if t.role == "assistant"]
        if assistants:
            return max(assistants)
        if self.turn_manifest:
            return max(e.turn_number for e in self.turn_manifest)
        return None

    def exchange_count(self) -> int:
        return len({t.index for t in self.conversation})

    def turn_indices(self) -> set[int]:
        return {t.index for t in self.conversation}

    def has_assistant_at(self, index: int) -> bool:
        return any(t.index == index and t.role == "assistant" for t in self.conversation)

    def turn_manifest_indices(self) -> set[int]:
        return {e.turn_number for e in self.turn_manifest}


@dataclass
class InputArtifact:
    """A file the task supplied, recorded before either model ran.

    Inputs are per-task, not per-slot. The export carries them on the single
    prompt-collection step alongside the prompt itself, and there is no `_b`
    counterpart to the `input_artifacts_a` field it uses, so both models are
    given the same files. The `_a` there is part of the field's name, not a
    model slot.

    Two origins, which have to be kept apart because only one of them is a file
    we hold. An `upload` carries a real `name` from the upload widget. A
    `declared_path` is a string the contributor typed to say where the file
    lives -- a Drive URL, a Windows path, a repository-relative path -- and may
    name a file or may just be a link, so whether it identifies anything is for
    the reader to decide rather than the adapter.
    """

    name: str = ""
    path: str = ""
    url: str = ""
    mime_type: str = ""
    file_type: str = ""
    size_bytes: int | None = None
    origin: Literal["upload", "declared_path"] = "upload"


@dataclass
class RubricCriterion:
    criterion_id: str
    text: str
    l1_label: str | None = None
    l2_label: str | None = None
    weight: int | None = None
    is_process_criterion: bool = False


@dataclass
class CriterionRating:
    criterion_id: str
    model: ModelSlot
    score: int
    relevant_turns: list[int] = field(default_factory=list)


@dataclass
class DimensionRating:
    model: ModelSlot
    dimension: str
    rating: int | None = None
    not_applicable: bool = False
    justification: str = ""
    relevant_turns: list[int] = field(default_factory=list)


@dataclass
class Sxs:
    """One head-to-head comparison the contributor recorded.

    A task may carry more than one. Honeybee Aspirational asks for two, both
    anchored on the product under test: A-vs-B and A-vs-C, with no B-vs-C. So
    `left`/`right` name which slots this comparison is between rather than
    leaving it implicit, and `winner_index` is an index into (`left`, `right`)
    rather than into a fixed (A, B).
    """

    likert: int | None = None
    justification: str = ""
    # Set only under the direction_magnitude encoding, where the authoring form
    # records the winner separately (responseIdx: 0=left, 1=right) and `likert`
    # carries nothing but the margin. Left None under the spec's bipolar 1-7
    # form, whose value already encodes direction.
    winner_index: int | None = None
    # Default to the A-vs-B pair so every existing single-comparison task, test
    # fixture, and JSONL row constructs unchanged.
    left: ModelSlot = "A"
    right: ModelSlot = "B"

    @property
    def pair(self) -> tuple[ModelSlot, ModelSlot]:
        return (self.left, self.right)

    @property
    def pair_label(self) -> str:
        """Stable identifier for findings and item ids, e.g. "AB", "AC"."""
        return f"{self.left}{self.right}"

    def winner_slot(self) -> ModelSlot | None:
        if self.winner_index == 0:
            return self.left
        if self.winner_index == 1:
            return self.right
        return None


@dataclass
class Task:
    task_id: str
    project_id: str = "6a70ffe56999de9083413f7d"

    # Customer-seeded material. The contributor may naturalise the prompt but
    # must preserve its meaning.
    seeded_prompt: str = ""
    user_goal: str = ""
    # Contributor-authored, not customer-seeded: `ingest.py` copies this straight
    # from the contributor's own target-outcome list (the same list check 80
    # audits), so it is their work product and must not be treated as ground
    # truth against which the rubric or the response is judged.
    target_deliverables: list[str] = field(default_factory=list)

    # The assignment the prompt was written against: a persona and a use-case
    # category, both prefilled by the customer before the contributor typed
    # anything. Check 70 measures the prompt against these two and nothing else.
    #
    # `seeded_prompt` above holds the *submitted* text despite its name: the
    # taskattempt's `final_hardened_prompt` field on the prompt-collection step is
    # what the contributor typed. The actual pre-seed lives somewhere else
    # entirely -- a `data-source-form-prefiller` entry under
    # `dataSourceResults`, keyed by a form ID rather than a step ID -- which is
    # why it went unread for as long as it did. `pre_seeded_prompt` below is that
    # field, read by `ingest.py`'s `PREFILLER_PRESEEDED_PROMPT`. Check 75 is the
    # only consumer of the diff between the two.
    assigned_domain: str = ""
    prompt_category: str = ""
    # The template the platform prepared before the contributor wrote anything,
    # for check 75. `None` when the prefiller is absent, which the ingest adapter
    # expects on most tasks project-wide -- the field is present on every task in
    # the audited batch this check was built against, but nothing licenses
    # assuming that holds elsewhere. `""` and `None` are the same "abstain"
    # signal here; there is no meaningful empty-but-present case to distinguish,
    # unlike `input_artifacts`.
    pre_seeded_prompt: str | None = None

    # The files supplied to the task, shared by both models. `None` and `[]` are
    # different answers and the distinction is load-bearing: `None` means no
    # adapter read an input manifest, so nothing can be concluded from its
    # emptiness, while `[]` means a manifest was read and the task supplied no
    # files. Check 85's file half fails a prompt for naming a file the task does
    # not have, so mistaking the first for the second fails tasks whose files
    # were sitting in the export unread.
    input_artifacts: list[InputArtifact] | None = None

    prompts: list[Turn] = field(default_factory=list)
    target_outcome: list[str] = field(default_factory=list)

    # How many turns the task was commissioned for, recorded on the prompt step
    # before either model ran. `None` when no adapter read it, which is every
    # task today -- the field sits in the export unread. It is a declaration of
    # intent rather than a measurement of what happened, so nothing may fail a
    # task on it alone; it is carried because it corroborates a conversation's
    # length independently of whether our fetch succeeded.
    estimated_turns: int | None = None

    model_a: ModelSubmission | None = None
    model_b: ModelSubmission | None = None
    # Present only on projects that run a third product. `None` on every
    # two-model task, which is what keeps `submissions()` returning a pair there.
    model_c: ModelSubmission | None = None
    # Present only on projects that add a fourth product (Model D). `None`
    # everywhere else, so existing three- and two-model tasks stay a pair/triple.
    model_d: ModelSubmission | None = None
    # The simplified trajectory (slot "S"). Held in its own field rather than
    # alongside the others so that no per-model check can pick it up by
    # accident: it has no dimension ratings, no turn attributions, and a turn
    # count that would fail the minimum-turns check by design.
    simplified: ModelSubmission | None = None

    # The slots this task's project files, whether or not each was filled in.
    # Carried on the task rather than looked up from `project_id` so that
    # `models` stays free of a dependency on `projects` -- and so that a task
    # built from the JSONL contract, which has no project, still declares what
    # "missing" would mean. A consumer that wants "was this slot left empty"
    # needs the expected set; `submissions()` can only report what is there.
    expected_slots: tuple[ModelSlot, ...] = ("A", "B")
    # When set, per-model checks (270/280/300/310/90/95/450-dims) only look at
    # these slots. Comparisons still run in full -- SxS needs both sides even
    # when only one of them is in scope. `None` means every competing slot.
    audit_slots: tuple[ModelSlot, ...] | None = None

    key_turn: KeyTurn = field(default_factory=KeyTurn)
    rubric: list[RubricCriterion] = field(default_factory=list)
    criterion_ratings: list[CriterionRating] = field(default_factory=list)
    dimension_ratings: list[DimensionRating] = field(default_factory=list)
    # The primary comparison. Retained as the A-vs-B pair so every existing
    # reader of `task.sxs` keeps the value it has always had; on a multi-
    # comparison task it is the same object as `comparisons[0]`, not a copy.
    sxs: Sxs = field(default_factory=Sxs)
    # Every comparison the task recorded. Empty on single-comparison tasks,
    # where `comparison_list()` falls back to `sxs`.
    comparisons: list[Sxs] = field(default_factory=list)
    # Customer-authored ground-truth final answer, when the project collects
    # one. Empty on Honeybee L1. Fed into Rubric Evaluation, Justifications,
    # and Prompt Clarity so "does this prompt have a unique ground truth?"
    # is a question the judge can actually answer.
    golden_answer: str = ""
    # Contributor-recorded known / not-known split, when the project collects
    # one. Empty on Honeybee L1.
    known_information: str = ""
    not_known_information: str = ""

    def submissions(self) -> list[ModelSubmission]:
        """The competing products, in slot order. Excludes the simplified
        trajectory -- see `trajectories()` for the accessor that includes it."""
        return [
            s
            for s in (self.model_a, self.model_b, self.model_c, self.model_d)
            if s is not None
        ]

    def in_audit_scope(self, slot: ModelSlot | str) -> bool:
        if self.audit_slots is None:
            return slot != "S"
        return slot in self.audit_slots

    def submissions_in_scope(self) -> list[ModelSubmission]:
        """Competing products the current run is actually grading.

        Hydration and SxS still see every submission; this is the per-model
        check population when `--audit-slots` has narrowed the report.
        """
        return [s for s in self.submissions() if self.in_audit_scope(s.model)]

    def trajectories(self) -> list[ModelSubmission]:
        """Every conversation filed for this task, simplified one included.

        For the checks whose question is about a filed conversation rather than
        about a product's quality: does its link resolve, were the files it
        produced uploaded.
        """
        return [*self.submissions(), *([self.simplified] if self.simplified else [])]

    def submission(self, slot: ModelSlot) -> ModelSubmission | None:
        return {
            "A": self.model_a,
            "B": self.model_b,
            "C": self.model_c,
            "D": self.model_d,
            "S": self.simplified,
        }.get(slot)

    def slot_submissions(self) -> list[tuple[ModelSlot, ModelSubmission | None]]:
        """Every slot the project expects, paired with what was filed for it.

        `None` means the slot was expected and left empty, which is a finding.
        Distinct from `trajectories()`, which reports only what exists.
        """
        return [(slot, self.submission(slot)) for slot in self.expected_slots]

    def key_turns(self) -> dict[ModelSlot, KeyTurn]:
        """Each slot's own freeze point, for the projects that record one per
        model. Empty on a project that records a single task-level key turn,
        where `self.key_turn` is the answer instead."""
        return {
            sub.model: sub.key_turn
            for sub in self.trajectories()
            if sub.key_turn.turn_index is not None or sub.key_turn.flagged_turns
        }

    def comparison_list(self) -> list[Sxs]:
        """Every comparison, for the checks defined over a pair.

        Falls back to the single `sxs` when `comparisons` was never populated,
        so a two-model task and its fixtures need no migration.
        """
        return self.comparisons or [self.sxs]

    def cited_turns(self, model: str) -> set[int]:
        """Every turn the contributor pointed at for one model.

        This lives on the input contract because three different consumers ask
        the same question of it -- check 96's corroboration of a short fetch,
        check 90's completeness measurement, and the hand-audit export -- and
        three copies of it would drift.

        The key turn counts as a citation, but which key turn depends on the
        project. Where one is recorded per model, this slot's own is the only
        one that is a statement about this slot. Where one is recorded for the
        whole task, it counts for every slot. The simplified trajectory is
        excluded from that fallback: no key turn is collected for it, and
        inheriting the task's -- turn 20 of a 21-turn session -- would read as a
        citation 18 turns past the end of a 2-turn conversation.
        """
        cited: set[int] = set()
        sub = self.submission(model)  # type: ignore[arg-type]
        own = sub.key_turn if sub is not None else None
        if own is not None and own.turn_index is not None:
            cited.add(own.turn_index)
        elif model != "S" and not (own and own.flagged_turns):
            if self.key_turn.turn_index is not None:
                cited.add(self.key_turn.turn_index)
        for rating in self.criterion_ratings:
            if rating.model == model:
                cited.update(rating.relevant_turns or [])
        for rating in self.dimension_ratings:
            if rating.model == model:
                cited.update(rating.relevant_turns or [])
        return {turn for turn in cited if isinstance(turn, int) and turn > 0}

    def criterion_ids(self) -> set[str]:
        return {c.criterion_id for c in self.rubric}

    def labeled_criteria(self) -> list[RubricCriterion]:
        return [c for c in self.rubric if c.l1_label]

    def weighted_criteria(self) -> list[RubricCriterion]:
        return [c for c in self.rubric if c.weight is not None]

    def zero_scored_ratings(self) -> list[CriterionRating]:
        return [r for r in self.criterion_ratings if r.score == 0]
