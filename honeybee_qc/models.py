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

ModelSlot = Literal["A", "B"]
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
    exchange -- the model's reply is never recorded here, matching the known
    limitation that only product A's legacy per-turn links exist at all -- plus
    whatever the contributor annotated about that turn: a share link (A only),
    whether the model produced a file there, and whether the contributor marked
    it the key turn.
    """

    turn_number: int
    prompt_text: str
    share_link: str = ""
    has_model_files: bool = False
    is_key_turn: bool = False
    key_turn_justification: str = ""
    attachment_names: list[str] = field(default_factory=list)


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
    conversation: list[Turn] = field(default_factory=list)
    # Present only on tasks authored against the turn-list taxonomy; empty on
    # every task ingested before it, which is the entire fallback -- nothing
    # downstream distinguishes "older schema" from "schema present but empty".
    turn_manifest: list[TurnManifestEntry] = field(default_factory=list)
    declared_provider: str = ""

    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.conversation if t.role == "assistant"]

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
class KeyTurn:
    turn_index: int | None = None
    justification: str = ""


@dataclass
class Sxs:
    likert: int | None = None
    justification: str = ""
    # Set only under the direction_magnitude encoding, where the authoring form
    # records the winner separately (responseIdx: 0=A, 1=B) and `likert` carries
    # nothing but the margin. Left None under the spec's bipolar 1-7 form, whose
    # value already encodes direction.
    winner_index: int | None = None


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

    model_a: ModelSubmission | None = None
    model_b: ModelSubmission | None = None

    key_turn: KeyTurn = field(default_factory=KeyTurn)
    rubric: list[RubricCriterion] = field(default_factory=list)
    criterion_ratings: list[CriterionRating] = field(default_factory=list)
    dimension_ratings: list[DimensionRating] = field(default_factory=list)
    sxs: Sxs = field(default_factory=Sxs)

    def submissions(self) -> list[ModelSubmission]:
        return [s for s in (self.model_a, self.model_b) if s is not None]

    def criterion_ids(self) -> set[str]:
        return {c.criterion_id for c in self.rubric}

    def labeled_criteria(self) -> list[RubricCriterion]:
        return [c for c in self.rubric if c.l1_label]

    def weighted_criteria(self) -> list[RubricCriterion]:
        return [c for c in self.rubric if c.weight is not None]

    def zero_scored_ratings(self) -> list[CriterionRating]:
        return [r for r in self.criterion_ratings if r.score == 0]
