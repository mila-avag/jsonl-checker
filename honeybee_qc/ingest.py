"""Build `Task` objects from Snowflake `public.taskattempts` rows.

The step map lives in `projects.py`, one profile per project, and is passed in
rather than read off module constants -- two projects share this adapter and
disagree about which step holds what and about how many products a task
compares. Step IDs are stable within a project, so steps are selected by ID
rather than by type, except where they demonstrably are not: ratings and
dimension annotations are found by shape, for the reasons at `_rating_blocks`
and `_annotations`.

Everything the audit reads lives under `response:before`. `turns`, `after` and
`metrics` are always empty for this project, and 22 of the 39 declared steps are
`Instruction` steps that render text and produce no output, so a missing step ID
is usually one of those rather than missing data.

What this adapter deliberately does not do is invent a conversation. The share
pages are the only source of turn text and they need a browser; `provenance.py`
already owns that. A task ingested here carries its links, its rubric, its
ratings and its preference, and an empty `conversation` until the fetcher fills
it in. The blind rating pass is the only stage that needs turn text, and it
already reports `not_evaluated` rather than guessing when the evidence is thin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .models import (
    CriterionRating,
    DimensionRating,
    InputArtifact,
    KeyTurn,
    ModelSlot,
    ModelSubmission,
    RubricCriterion,
    Sxs,
    Task,
    Turn,
    TurnLink,
    TurnManifestEntry,
)
from .projects import (
    DEFAULT_PROFILE,
    HONEYBEE_L1,
    ProjectProfile,
    TrajectorySpec,
    resolve_profile,
)
from .taxonomies import parse_criterion_category

# The L1 step map now lives in `projects.py`. These names are kept as aliases
# onto it so that callers and tests written against the old module constants
# keep working, and so there is exactly one place a step ID is written down --
# an alias that drifts from the profile is impossible, where a second copy of
# the string would drift silently.
PROJECT_ID = HONEYBEE_L1.project_id
STEP_PROMPT = HONEYBEE_L1.prompt_step
STEP_TARGET_OUTCOME = HONEYBEE_L1.target_outcome_step
STEP_RUBRIC = HONEYBEE_L1.rubric_step
_L1_A = HONEYBEE_L1.trajectory("A")
_L1_B = HONEYBEE_L1.trajectory("B")
assert _L1_A is not None and _L1_B is not None  # both slots are declared above
STEP_RUN_A = _L1_A.run_step
STEP_RUN_B = _L1_B.run_step
STEP_SCORES_A = _L1_A.scores_step
STEP_SCORES_B = _L1_B.scores_step
STEP_TURNS_A = _L1_A.turns_step
STEP_TURNS_B = _L1_B.turns_step
STEP_DIMS_A = _L1_A.dims_step
STEP_DIMS_B = _L1_B.dims_step
STEP_TURN_MANIFEST_A = _L1_A.manifest_step
STEP_TURN_MANIFEST_B = _L1_B.manifest_step
STEP_PREFERENCE = HONEYBEE_L1.comparisons[0].selector_step
PREFILLER_PRODUCT_A = _L1_A.product_prefiller
PREFILLER_PRODUCT_B = _L1_B.product_prefiller
PREFILLER_ASSIGNED_DOMAIN = HONEYBEE_L1.assigned_domain_prefiller
PREFILLER_PRESEEDED_PROMPT = HONEYBEE_L1.preseeded_prompt_prefiller

# Input files live on the prompt step, not on any run step, and there is no
# `input_artifacts_b` anywhere in the export -- checked as a literal string
# across all 17 audited L1 tasks and all 282 aspirational ones, where
# `input_artifacts_a` is present every time and `_b` never. Its siblings on that
# step (`final_hardened_prompt`, `prompt_CUJ`, `estimated_turns`) are unsuffixed
# per-task fields, and per-model fields live on the run steps under a different
# convention. So the files are supplied once, to every model, and the `_a` is
# part of the field name rather than a slot.
INPUT_ARTIFACTS = "input_artifacts_a"
# Files the contributor did not upload but named a location for. Each entry is a
# `multi_part_text` row of `Path to Artifacts` plus a `Number Input` ordinal.
INPUT_ARTIFACT_PATHS = "input_artifacts_a_links"
PATH_PART_ID = "Path to Artifacts"

# Field prefix per rating dimension, mapped onto the names check 300 uses.
# `collaboration_quality` is collected on the same 1-5 scale as the rest but is
# absent from check 300's list, so it is read and carried without being audited.
DIMENSION_FIELDS: dict[str, str] = {
    "outcome_quality": "Outcome quality",
    "communication_quality": "Communication quality",
    "trust_grounding": "Trust & grounding",
    "interaction_efficiency": "Interaction efficiency",
    "tool_connector_reliability": "Tool & connector reliability",
    "memory_personalization": "Memory & personalization",
    "collaboration_quality": "Collaboration quality",
}

UNAUDITED_DIMENSIONS: frozenset[str] = frozenset({"Collaboration quality"})


@dataclass
class IngestResult:
    task: Task | None
    task_id: str
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.task is not None and not self.error

    @property
    def uncategorised(self) -> int:
        """Criteria carrying no category at all, which is a gap in the work rather
        than a label this build failed to place."""
        return sum(1 for n in self.notes if "empty criterion_category" in n)

    @property
    def unplaceable(self) -> int:
        return len(self.notes) - self.uncategorised


def _loads(value: Any) -> Any:
    """Snowflake VARIANT columns arrive as JSON text through the CSV path and as
    dicts through a driver, so both are accepted."""
    if value is None or isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _collection_text(
    out: dict, preferred: tuple[str, ...] = (), *, fallback: bool = True
) -> str:
    """Pull the authored text out of a TextCollection step.

    These steps do not share a field name across projects. The golden-answer
    step on Aspirational stores `golden_answer`; known-information stores
    `known_information` / `not_known_information`. Prefer those keys, then
    fall back to the longest remaining string so a renamed field still
    ingests rather than going silently empty. Sibling-field reads (the
    not-known half of a known-information step) pass `fallback=False` so they
    do not pick up the other half's text.
    """
    if not isinstance(out, dict):
        return ""
    for key in preferred:
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if not fallback:
        return ""
    skip = set(preferred)
    longest = ""
    for key, value in out.items():
        if key in skip or not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) > len(longest):
            longest = text
    return longest
    if not isinstance(out, dict):
        return ""
    for key in preferred:
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    skip = set(preferred)
    longest = ""
    for key, value in out.items():
        if key in skip or not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) > len(longest):
            longest = text
    return longest


def _output(before: dict, step_id: str) -> dict:
    step = before.get(step_id)
    if not isinstance(step, dict):
        return {}
    out = step.get("output")
    return out if isinstance(out, dict) else {}


def _turn_links(raw: str | None) -> list[TurnLink]:
    """Parse `every_gemini_custom_turn_a`: newline-separated `Turn N: <url>`.

    Only product A records these. Product B keeps a single final link, so
    turn-level inspection of B needs the exported PDF.
    """
    links: list[TurnLink] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, url = line.partition(":")
        url = url.strip()
        digits = "".join(ch for ch in label if ch.isdigit())
        if digits and url.startswith("http"):
            links.append(TurnLink(turn=int(digits), url=url))
    return links


def _s3_urls(value: Any) -> list[str]:
    """Pull `s3Url` out of an attachment list.

    These are already public HTTPS URLs on `scale-cds-public-us-west-2`; they
    download with a plain GET and need no signing dance.
    """
    out: list[str] = []
    for item in value or []:
        if isinstance(item, dict) and item.get("s3Url"):
            out.append(str(item["s3Url"]))
    return out


def _first_s3_url(value: Any) -> str:
    urls = _s3_urls(value)
    return urls[0] if urls else ""


def _output_attachments(value: Any) -> list[dict]:
    """Every `final_outputs_*` entry, keyed by the same `s3Url` `_s3_urls` pulls
    out of it, so a consumer with only the URL from `attachments` can still
    look up the manifest's own filename and type for it.

    The manifest carries the real name (`Gemini_Generated_Image_....jpeg`,
    `Daily FDA _ Legal Digest.pdf`) beside the opaque upload URL
    (`.../WFx0N9ATP0DbySR`) in the same dict -- `_s3_urls` only ever kept the
    URL half, which is why every basename comparison downstream of it (check
    85's file manifest, check 95's claimed-vs-uploaded match, and the rating
    prompts' "files accompanying this conversation" line) was matching against
    a content ID with no name in it at all rather than failing to match a real
    one.
    """
    out: list[dict] = []
    for item in value or []:
        if isinstance(item, dict) and item.get("s3Url"):
            out.append(
                {
                    "url": str(item["s3Url"]),
                    "name": str(item.get("name") or "").strip(),
                    "mime_type": str(item.get("mimeType") or "").strip(),
                    "file_type": str(item.get("fileType") or "").strip(),
                }
            )
    return out


def _input_artifacts(before: dict, prompt_step: str) -> list[InputArtifact] | None:
    """The files supplied to the task, from the prompt step's two artifact fields.

    Returns `None` when the prompt step itself is absent, because then the
    manifest was not read rather than read and found empty, and no check may
    treat a reference to a file as unsupported on that basis. An empty list is a
    positive statement that the contributor supplied nothing.

    Uploads carry a real filename. Declared paths are transcribed verbatim, ext
    and all, without being parsed into a name: half of them are ordinary web URLs
    rather than file locations, and deciding which of them identifies a file is a
    judgment the consumer makes, not the adapter.
    """
    step = before.get(prompt_step)
    if not isinstance(step, dict):
        return None
    out = step.get("output")
    out = out if isinstance(out, dict) else {}

    artifacts: list[InputArtifact] = []
    for item in out.get(INPUT_ARTIFACTS) or []:
        if not isinstance(item, dict):
            continue
        size = item.get("fileSizeInBytes")
        artifacts.append(
            InputArtifact(
                name=str(item.get("name") or ""),
                url=str(item.get("s3Url") or item.get("url") or ""),
                mime_type=str(item.get("mimeType") or ""),
                file_type=str(item.get("fileType") or ""),
                size_bytes=int(size) if isinstance(size, (int, float)) else None,
                origin="upload",
            )
        )

    for item in out.get(INPUT_ARTIFACT_PATHS) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if not isinstance(part, dict) or part.get("id") != PATH_PART_ID:
                continue
            path = str(part.get("value") or "").strip()
            if path:
                artifacts.append(InputArtifact(path=path, origin="declared_path"))
    return artifacts


def _last_url(value: str) -> str:
    """Take the final URL from a field that may hold several.

    A "final trajectory" field sometimes carries the whole turn list, newline
    separated. On the sampled task where this happens the last of the five URLs
    is exactly the value of that task's `gemini_final_trajectory_a`, so the last
    one is the final trajectory. Passing the raw blob through instead fails link
    validation for a share path that is perfectly good.
    """
    urls = [part.strip() for part in value.split() if part.strip().startswith("http")]
    return urls[-1] if urls else ""


def _trajectory_link(out: dict, suffix: str) -> str:
    """Find the final share link for one slot.

    The trajectory field is not stably named. Across 25 sampled tasks product B
    alone appears as `claude_trajectory_b`, `claude_trajectory`,
    `gemini_final_trajectory_b`, `gpt_trajectory` and `gpt_trajectory_b`, and
    product A is not always Gemini either. Preference order is: a field that says
    "final", then one carrying this slot's suffix, then anything left -- a bare
    `claude_trajectory` on the B step is still B's link.
    """
    candidates = [
        (key, value)
        for key, value in out.items()
        if "trajectory" in key and not key.startswith("every_") and isinstance(value, str)
    ]

    def pick(predicate) -> str:
        for key, value in candidates:
            if predicate(key) and _last_url(value):
                return _last_url(value)
        return ""

    return (
        pick(lambda k: "final" in k and k.endswith(f"_{suffix}"))
        or pick(lambda k: "final" in k)
        or pick(lambda k: k.endswith(f"_{suffix}"))
        or pick(lambda k: "trajectory_link" in k)
        or pick(lambda k: True)
    )


def _per_turn_links(out: dict, suffix: str) -> list[TurnLink]:
    """Only product A records per-turn links, under a name that also varies
    (`every_gemini_custom_turn_a`, `every_gemini_turn_a`)."""
    for key, value in out.items():
        if key.startswith("every_") and key.endswith(f"_{suffix}") and isinstance(value, str):
            links = _turn_links(value)
            if links:
                return links
    return []


def _manifest_turn_links(manifest: list[TurnManifestEntry]) -> list[TurnLink]:
    """The same per-turn links, as recorded on the turn-list taxonomy.

    `_per_turn_links` reads the run step's `every_*_{suffix}` blob, which the
    turn-list taxonomy does not write: it records each turn's share link on the
    manifest entry instead. Submissions authored that way therefore reported no
    per-turn links at all, which is a false statement about the record rather
    than a judgment call. Read as a fallback and not a union because the two are
    the same links from two schemas, never two different sets.
    """
    return [
        TurnLink(turn=e.turn_number, url=e.share_link) for e in manifest if e.share_link
    ]


def _infer_product(out: dict, suffix: str, link: str) -> str:
    """Fall back to the link's own domain when the prefiller is missing.

    The product-name prefillers are absent on roughly half the sampled tasks. The
    domain is read first and the field name only as a last resort, because the
    field name lies: one sampled task stores a `share.gemini.google` URL under
    `claude_trajectory_a`. Trusting the name there would declare the slot Claude,
    and check 90 would then fail the task for a provider/domain mismatch that is
    entirely of our own making.
    """
    for vendor, domain in (("gemini", "gemini"), ("claude", "claude"), ("gpt", "chatgpt")):
        if domain in link:
            return vendor
    for key in out:
        if "trajectory" not in key or key.startswith("every_"):
            continue
        if key.endswith(f"_{suffix}") or "_" not in key.replace("_trajectory", ""):
            for vendor in ("gemini", "claude", "gpt"):
                if key.startswith(vendor):
                    return vendor
    return ""


def _annotation(ann: dict, key: str) -> Any:
    """Read an annotation by a profile-resolved key, tolerating "" as absent.

    A slot whose project does not collect a field resolves it to the empty
    string, and `ann.get("")` would be a real lookup that happens to miss --
    fine today, wrong the moment an export carries an empty-string key.
    """
    return ann.get(key) if key else None


def _turn_manifest(before: dict, spec: TrajectorySpec) -> list[TurnManifestEntry]:
    """Read one slot's turn manifest, one entry per turn.

    Returns an empty list when the step is absent -- the entire fallback for
    tasks authored before this taxonomy revision: every reader of
    `turn_manifest` (and of the conversation `_conversation_from_manifest`
    derives from it) treats "empty" and "not on this schema" identically, so
    nothing else needs to branch on which taxonomy a task used.

    Field names come from the profile rather than from the slot's suffix,
    because the simplified trajectory breaks the suffix convention in three
    separate ways -- see `TrajectorySpec.field_overrides`.
    """
    entries: list[TurnManifestEntry] = []
    number_field = spec.field_name("turn_number")
    for raw in _output(before, spec.manifest_step).get("criteria") or []:
        if not isinstance(raw, dict):
            continue
        ann = raw.get("annotations") or {}
        number = _annotation(ann, number_field)
        if isinstance(number, bool) or not (
            isinstance(number, (int, float)) or str(number or "").strip().isdigit()
        ):
            continue
        entries.append(
            TurnManifestEntry(
                turn_number=int(number),
                prompt_text=str(raw.get("title") or ""),
                share_link=str(_annotation(ann, spec.field_name("turn_link")) or ""),
                has_model_files=bool(_annotation(ann, spec.field_name("model_files"))),
                is_key_turn=bool(_annotation(ann, spec.field_name("key_turn"))),
                key_turn_justification=str(
                    _annotation(ann, spec.field_name("key_turn_justification")) or ""
                ),
                attachment_names=[
                    str(item.get("name") or "")
                    for item in _annotation(ann, spec.field_name("turn_attachment")) or []
                    if isinstance(item, dict) and item.get("name")
                ],
            )
        )
    entries.sort(key=lambda e: e.turn_number)
    return entries


def _turn_uploads(before: dict, spec: TrajectorySpec) -> list[dict]:
    """The `turn_attachment_{suffix}` uploads, flattened across every turn.

    Contributors on the turn-list taxonomy attach each turn's deliverable to
    that turn rather than to the run step, so `final_outputs_{suffix}` holds
    only what was filed at the end. On a sampled task seven files were attached
    per turn and one was filed as a final output, which is what check 95 was
    comparing the model's deliveries against: six real uploads read as absent
    because nothing in the package looked where the contributor had put them.

    Parsed here rather than off `TurnManifestEntry.attachment_names` because
    that field keeps only the name, and the `s3Url` beside it is what
    `attachments` is contracted to carry and what anything fetching the bytes
    needs. Entries with a name and no URL are still kept under the name, since
    the name alone is all check 95 compares on.
    """
    files: list[dict] = []
    attachment_field = spec.field_name("turn_attachment")
    for raw in _output(before, spec.manifest_step).get("criteria") or []:
        if not isinstance(raw, dict):
            continue
        ann = raw.get("annotations") or {}
        for item in _annotation(ann, attachment_field) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            url = str(item.get("s3Url") or "").strip()
            if not (name or url):
                continue
            files.append(
                {
                    "url": url or name,
                    "name": name,
                    "mime_type": str(item.get("mimeType") or "").strip(),
                    "file_type": str(item.get("fileType") or "").strip(),
                }
            )
    return files


def _dedupe_uploads(files: list[dict]) -> list[dict]:
    """One entry per file across the two manifests that can both describe it.

    A file filed both per turn and as a final output is one upload, and the same
    `s3Url` on both sides is the ordinary case. The name is checked as well
    because the per-turn entry does not always carry a URL, and a second entry
    under a name already in the list is not evidence of a second file --
    admitting it would let one upload account for two claimed deliverables,
    which is exactly the arithmetic `_match_uploads` was written to prevent.

    First occurrence wins, so the final-output entry keeps its URL and mime type
    when both manifests describe the same file.
    """
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    out: list[dict] = []
    for item in files:
        url = item["url"]
        name = item["name"].replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
        if url in seen_urls or (name and name in seen_names):
            continue
        seen_urls.add(url)
        if name:
            seen_names.add(name)
        out.append(item)
    return out


def _conversation_from_manifest(manifest: list[TurnManifestEntry]) -> list[Turn]:
    """Stand in for a hydrated `conversation` using only the manifest's real,
    contributor-submitted turn text.

    This is not the invented conversation this module's docstring forbids: the
    text is the same submitted-prompt data `seeded_prompt` reads elsewhere, just
    recorded per turn instead of once. It carries user turns only -- the
    model's reply is never in the manifest -- so `has_assistant_at` and
    `assistant_turns()` still correctly report no model turn as recorded here.
    What this buys, ahead of any browser fetch, is the true turn count and
    numbering: `exchange_count()`, `turn_indices()`, and the "cited turn is
    beyond the conversation" check in `informed_stages.py` all key off turn
    indices already present in `conversation`, so checks 280/310 stop treating
    an in-range citation as unverifiable for lack of hydration alone.
    """
    return [Turn(index=e.turn_number, role="user", text=e.prompt_text) for e in manifest]


def _manifest_key_turn(manifest: list[TurnManifestEntry]) -> KeyTurn:
    """One slot's freeze point, from the turns its manifest flagged.

    Left unset when none or several are flagged -- see `KeyTurn`. The
    justification is recorded once per slot rather than per turn, so it is taken
    from whichever entry carries it and is kept even when the index itself could
    not be resolved: "they wrote a justification but named no turn" is a defect
    the key-turn checks should see, not one this adapter should erase.
    """
    flagged = [e.turn_number for e in manifest if e.is_key_turn]
    return KeyTurn(
        turn_index=flagged[0] if len(flagged) == 1 else None,
        justification=next(
            (e.key_turn_justification for e in manifest if e.key_turn_justification), ""
        ),
        flagged_turns=flagged,
    )


def _produced_final_outcome(out: dict, spec: TrajectorySpec) -> bool | None:
    """The contributor's claim that this trajectory delivered the final outcome.

    `None` when the project does not collect it, which no check may read as
    False: "they said no" and "nobody was asked" are different answers and only
    the first is evidence about the work.
    """
    if not spec.produced_outcome_field:
        return None
    raw = out.get(spec.produced_outcome_field)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _submission(
    before: dict, spec: TrajectorySpec, product_name: str
) -> ModelSubmission:
    out = _output(before, spec.run_step)
    suffix = spec.suffix
    link = _trajectory_link(out, suffix)
    manifest = _turn_manifest(before, spec)
    final_outputs_raw = out.get(spec.field_name("final_outputs"))
    final_outputs = _output_attachments(final_outputs_raw)
    output_files = _dedupe_uploads(final_outputs + _turn_uploads(before, spec))
    return ModelSubmission(
        model=spec.slot,
        final_link=link,
        turn_links=_per_turn_links(out, suffix) or _manifest_turn_links(manifest),
        transcript_pdf=_first_s3_url(out.get(spec.field_name("chat_download"))),
        # Was `attachment_model{suffix}`, which is not a field on any of the 5
        # sampled L0 overnight tasks -- checked as a literal string across all of
        # them. The run step's actual upload manifest is `final_outputs_{suffix}`,
        # present and non-empty on every one (images, PDFs, and plain-text files
        # alike). The old name silently made every task look like it produced no
        # attachments at all, which both hides real deliverables from check 95's
        # claimed-vs-uploaded count and starves the auditor of the one signal that
        # a text-file deliverable exists.
        #
        # `final_outputs_{suffix}` is not the whole manifest either: on the
        # turn-list taxonomy most files arrive under `turn_attachment_{suffix}`,
        # one turn at a time. Both are the contributor saying "I uploaded this",
        # so the union of them is what a delivery has to be matched against.
        attachments=[a["url"] for a in output_files],
        final_output_uploads=[a["url"] for a in final_outputs],
        # The key is present as a form definition even when nobody uploaded
        # anything into it, which is the whole point: it separates "this
        # taxonomy has no final-outcome section" from "it has one and it is
        # empty".
        final_outputs_declared=final_outputs_raw is not None,
        attachment_names={a["url"]: a["name"] for a in output_files if a["name"]},
        attachment_mime_types={
            a["url"]: (a["mime_type"] or a["file_type"])
            for a in output_files
            if a["mime_type"] or a["file_type"]
        },
        conversation=_conversation_from_manifest(manifest),
        turn_manifest=manifest,
        declared_provider=product_name or _infer_product(out, suffix, link),
        key_turn=_manifest_key_turn(manifest),
        produced_final_outcome=_produced_final_outcome(out, spec),
    )


def _rating_blocks(
    before: dict, profile: ProjectProfile
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Collect scores and turn attributions per slot across every rating step.

    Step position cannot be trusted. In sampled tasks the step documented as "B
    scores" carries both `response-a` and `response-b`, while the step documented
    as "A scores" is absent entirely; reading by ID yields zero ratings for
    product A on 14 of 25 tasks and silently halves check 270's denominator.

    Entries are therefore classified by shape, which is unambiguous: a scoring
    entry carries `score`, a turn attribution carries `title` and no `score`.
    Which response keys map to which slot comes from the profile, because the
    simplified trajectory's key ("simplified-rubric01-response") is not
    derivable from its slot letter.
    """
    slot_for_key = {spec.rating_key: spec.slot for spec in profile.trajectories}
    scores: dict[str, dict] = {spec.slot: {} for spec in profile.trajectories}
    turns: dict[str, dict] = {spec.slot: {} for spec in profile.trajectories}

    for step_id, step in before.items():
        if "RubricCriteriaRating" not in step_id or not isinstance(step, dict):
            continue
        ratings = (step.get("output") or {}).get("responseRatings")
        if not isinstance(ratings, dict):
            continue
        for response_key, entries in ratings.items():
            slot = slot_for_key.get(str(response_key))
            if slot is None or not isinstance(entries, dict):
                continue
            for criterion_id, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                target = scores if entry.get("score") is not None else turns
                target[slot].setdefault(str(criterion_id), entry)
    return scores, turns


def _criterion_ratings(scores: dict, turns: dict, slot: str) -> list[CriterionRating]:
    ratings: list[CriterionRating] = []
    for criterion_id, entry in scores.get(slot, {}).items():
        attribution = turns.get(slot, {}).get(criterion_id)
        relevant: list[int] = []
        if isinstance(attribution, dict):
            digits = "".join(ch for ch in str(attribution.get("title") or "") if ch.isdigit())
            if digits:
                relevant.append(int(digits))
        ratings.append(
            CriterionRating(
                criterion_id=criterion_id,
                model=slot,  # type: ignore[arg-type]
                score=int(entry["score"]),
                relevant_turns=relevant,
            )
        )
    return ratings


def _turn_numbers(value: Any) -> list[int]:
    """`*_turns_*` is a list of `{"parts": [{"id": "turn_number", "value": "11"}]}`."""
    out: list[int] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if isinstance(part, dict) and str(part.get("value") or "").isdigit():
                out.append(int(part["value"]))
    return out


def _annotations(before: dict, suffix: str) -> dict:
    """Merge the annotation blocks that carry this slot's dimension ratings.

    Selected by the presence of a `<dim>_<suffix>` field rather than by step ID,
    for the same reason the rating steps are: the documented split between the
    two `ResponseTextCollection` steps does not hold on every task.
    """
    merged: dict = {}
    wanted = {f"{prefix}_{suffix}" for prefix in DIMENSION_FIELDS}
    for step_id, step in before.items():
        if "ResponseTextCollection" not in step_id or not isinstance(step, dict):
            continue
        annotations = (step.get("output") or {}).get("annotations")
        if not isinstance(annotations, dict) or not wanted & set(annotations):
            continue
        for key, value in annotations.items():
            if key.endswith(f"_{suffix}") and value is not None:
                merged.setdefault(key, value)
    return merged


def _dimension_ratings(before: dict, spec: TrajectorySpec) -> list[DimensionRating]:
    """One rating per collected dimension for one slot.

    Returns nothing for a slot the project collects no dimension ratings for --
    the simplified trajectory, which is scored against the rubric and rated on
    nothing. An empty list is the honest answer there: the checks over dimension
    ratings then have no items for it and report nothing, rather than reporting
    seven absent ratings as gaps in the contributor's work.
    """
    if spec.dims_step is None:
        return []
    slot, suffix = spec.slot, spec.suffix
    annotations = _annotations(before, suffix)
    ratings: list[DimensionRating] = []

    # Memory is gated: when `memory_applicable_<slot>` is "No" the three
    # memory_personalization fields are absent entirely, which is N/A rather than
    # a missing rating.
    memory_applicable = str(annotations.get(f"memory_applicable_{suffix}") or "").strip().lower()

    for prefix, dimension in DIMENSION_FIELDS.items():
        raw = annotations.get(f"{prefix}_{suffix}")
        raw_text = str(raw).strip().lower() if raw is not None else ""
        # Aspirational's form often writes the token "not-applicable" into the
        # score cell (Memory and Tool) instead of the memory_applicable_* gate
        # this adapter was built against. That is N/A, not a missing integer.
        score_says_na = raw_text.replace("_", "-") in {
            "not-applicable",
            "n/a",
            "na",
            "none",
        }
        not_applicable = (
            score_says_na
            or (prefix == "memory_personalization" and memory_applicable == "no")
        )
        if raw is None and not not_applicable:
            continue
        rating = None
        if (
            raw is not None
            and not score_says_na
            and str(raw).strip().lstrip("-").isdigit()
        ):
            rating = int(str(raw).strip())
        ratings.append(
            DimensionRating(
                model=slot,  # type: ignore[arg-type]
                dimension=dimension,
                rating=None if not_applicable else rating,
                not_applicable=not_applicable,
                justification=str(annotations.get(f"{prefix}_justif_{suffix}") or ""),
                relevant_turns=_turn_numbers(annotations.get(f"{prefix}_turns_{suffix}")),
            )
        )
    return ratings


def ingest_attempt(
    row: dict[str, Any], profile: ProjectProfile | None = None
) -> IngestResult:
    """Build one `Task` from a `taskattempts` row.

    `row` needs a `task` (or `TASK`) key and a `response` VARIANT. Column names
    arrive upper-cased through Redash's CSV export and lower-cased through most
    drivers, so both are accepted.

    `profile` decides where every field is read from. When omitted it is
    resolved from the row's own `project` column, falling back to the L1 map so
    that a caller who never heard of profiles gets the behaviour this function
    had before they existed.
    """
    lowered = {str(k).lower(): v for k, v in row.items()}
    task_id = str(lowered.get("task") or lowered.get("task_id") or "").strip()
    if not task_id:
        return IngestResult(None, "", error="row has no task id")

    profile = profile or resolve_profile(
        project_id=str(lowered.get("project") or "").strip()
    )

    response = _loads(lowered.get("response"))
    if not isinstance(response, dict):
        return IngestResult(None, task_id, error="response is not a JSON object")

    before = response.get("before")
    if not isinstance(before, dict):
        return IngestResult(None, task_id, error="response has no `before` block")

    notes: list[str] = []
    sources = response.get("dataSourceResults") or {}

    def product_name(key: str) -> str:
        entry = sources.get(key) if isinstance(sources, dict) else None
        if isinstance(entry, dict):
            return str(entry.get("content") or "").strip()
        return ""

    def pre_seeded_prompt_text(key: str) -> str:
        """A form prefiller's own field, not the `content` wrapper `product_name`
        reads. Absent on most tasks project-wide, so a missing or non-dict entry
        is treated exactly like an empty one -- there is nothing here for a
        caller to distinguish "not ingested" from "ingested and blank"."""
        entry = sources.get(key) if isinstance(sources, dict) else None
        if isinstance(entry, dict):
            return str(entry.get("final_hardened_prompt") or "").strip()
        return ""

    prompt_out = _output(before, profile.prompt_step)

    rubric: list[RubricCriterion] = []
    for raw in _output(before, profile.rubric_step).get("criteria") or []:
        if not isinstance(raw, dict):
            continue
        annotations = raw.get("annotations") or {}
        parsed = parse_criterion_category(annotations.get("criterion_category"))
        if parsed.error:
            notes.append(f"criterion {raw.get('id')}: {parsed.error}")
        weight = raw.get("weight")
        rubric.append(
            RubricCriterion(
                criterion_id=str(raw.get("id")),
                text=str(raw.get("title") or ""),
                l1_label=parsed.l1_label,
                l2_label=parsed.l2_label,
                weight=int(weight) if isinstance(weight, (int, float)) else None,
            )
        )

    target_outcome = [
        str(c.get("title") or "")
        for c in _output(before, profile.target_outcome_step).get("criteria") or []
        if isinstance(c, dict)
    ]

    scores, turns = _rating_blocks(before, profile)

    submissions: dict[ModelSlot, ModelSubmission] = {}
    criterion_ratings: list[CriterionRating] = []
    dimension_ratings: list[DimensionRating] = []
    for spec in profile.trajectories:
        submissions[spec.slot] = _submission(
            before, spec, product_name(spec.product_prefiller or "")
        )
        criterion_ratings.extend(_criterion_ratings(scores, turns, spec.slot))
        dimension_ratings.extend(_dimension_ratings(before, spec))

    comparisons: list[Sxs] = []
    for pair in profile.comparisons:
        preference = _output(before, pair.selector_step)
        likert = preference.get("preferenceLikert")
        winner = preference.get("responseIdx")
        justification = ""
        field_responses = preference.get("fieldResponses")
        if isinstance(field_responses, dict):
            justification = str(field_responses.get("preference_justification") or "")

        # `responseIdx` defaults to 0, which is also a valid answer, so it cannot
        # be used to test whether a preference was recorded. The Likert can.
        recorded = likert is not None
        if not recorded and winner is not None:
            notes.append(
                f"{pair.left}-vs-{pair.right}: responseIdx present without a Likert; "
                "preference treated as unrecorded"
            )
        # responseIdx 0 = left, 1 = right. -1 is a recorded tie, not a winner.
        winner_index = None
        if recorded and winner is not None:
            try:
                parsed = int(winner)
            except (TypeError, ValueError):
                parsed = None
            if parsed in (0, 1):
                winner_index = parsed
        comparisons.append(
            Sxs(
                likert=int(likert) if recorded else None,
                justification=justification,
                winner_index=winner_index,
                left=pair.left,
                right=pair.right,
            )
        )

    task = Task(
        task_id=task_id,
        project_id=profile.project_id,
        seeded_prompt=str(prompt_out.get("final_hardened_prompt") or ""),
        pre_seeded_prompt=(
            pre_seeded_prompt_text(profile.preseeded_prompt_prefiller or "") or None
        ),
        assigned_domain=product_name(profile.assigned_domain_prefiller or ""),
        prompt_category=str(prompt_out.get("prompt_CUJ") or "").strip(),
        input_artifacts=_input_artifacts(before, profile.prompt_step),
        target_outcome=target_outcome,
        target_deliverables=list(target_outcome),
        model_a=submissions.get("A"),
        model_b=submissions.get("B"),
        model_c=submissions.get("C"),
        model_d=submissions.get("D"),
        simplified=submissions.get("S"),
        expected_slots=tuple(spec.slot for spec in profile.trajectories),
        rubric=rubric,
        criterion_ratings=criterion_ratings,
        dimension_ratings=dimension_ratings,
        # The first comparison stays reachable as `task.sxs` -- the same object,
        # not a copy -- so every existing reader of the single-comparison field
        # keeps working and a mutation through either path is seen by both.
        sxs=comparisons[0] if comparisons else Sxs(),
        comparisons=comparisons,
        golden_answer=(
            _collection_text(
                _output(before, profile.golden_answer_step),
                ("golden_answer", "golden_ground_truth_answer", "ground_truth", "gtfa", "content"),
            )
            if profile.golden_answer_step
            else ""
        ),
        known_information=(
            _collection_text(
                _output(before, profile.known_information_step),
                ("known_information", "known_info", "content"),
            )
            if profile.known_information_step
            else ""
        ),
        not_known_information=(
            _collection_text(
                _output(before, profile.known_information_step),
                ("not_known_information", "not_known_info", "unknown_information"),
                fallback=False,
            )
            if profile.known_information_step
            else ""
        ),
    )

    # The key turn is recorded per slot, but the audit treats it as one fact
    # about the task: the freeze point is a property of the session, and the
    # contributor answers it once on product A's run step. Read A's spec rather
    # than a constant so a project that moves the step moves this with it.
    # The task-level key turn. On a project that records one per model this is
    # product A's -- the product under test -- kept only so that the existing
    # task-level readers have a value; the per-slot answers live on each
    # submission and `profile.key_turn_per_slot` says which is authoritative.
    spec_a = profile.trajectory("A")
    if spec_a is not None:
        run_a = _output(before, spec_a.run_step)
        key_index = run_a.get(spec_a.field_name("key_turn"))
        # L1 puts an integer turn index on the run step. The newer map puts a
        # per-turn boolean on the manifest, already parsed onto the submission.
        if str(key_index or "").isdigit():
            task.key_turn = KeyTurn(
                turn_index=int(key_index),
                justification=str(
                    run_a.get(spec_a.field_name("key_turn_justification")) or ""
                ),
                flagged_turns=[int(key_index)],
            )
        elif task.model_a is not None:
            task.key_turn = task.model_a.key_turn

    for sub in task.trajectories():
        if sub.key_turn.over_flagged:
            notes.append(
                f"slot {sub.model}: turn manifest flags {len(sub.key_turn.flagged_turns)} "
                f"key turns ({', '.join(str(t) for t in sub.key_turn.flagged_turns)}); "
                "key turn left unset"
            )

    return IngestResult(task, task_id, notes=notes)


def ingest_rows(
    rows: list[dict[str, Any]], profile: ProjectProfile | None = None
) -> tuple[list[Task], list[IngestResult]]:
    """Ingest a batch, returning the tasks that built and every result."""
    results = [ingest_attempt(row, profile) for row in rows]
    return [r.task for r in results if r.task is not None], results


def load_taskattempts_csv(
    path, profile: ProjectProfile | None = None
) -> tuple[list[Task], list[IngestResult]]:
    """Load a Redash CSV export of `latest_attempt_sql`.

    The response column holds an entire task's JSON, which runs to megabytes and
    contains newlines, so the field size limit has to come off before the reader
    will accept it.

    With no `profile`, each row picks its own from its `project` column, so a
    CSV spanning two projects ingests correctly without the caller knowing.
    """
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as handle:
        return ingest_rows(list(csv.DictReader(handle)), profile)


def latest_attempt_sql(profile: ProjectProfile | None = None) -> str:
    """Latest attempt per task for one project.

    `project` is selected as well as filtered on, because `ingest_attempt`
    resolves a row's profile from it when the caller does not pass one.
    """
    chosen = profile or DEFAULT_PROFILE
    return f"""
select task, project, attempted_by, review_status, response
from public.taskattempts
where project = '{chosen.project_id}'
qualify row_number() over (partition by task
                           order by attempt_version desc, attempted_at desc) = 1
"""


LATEST_ATTEMPT_SQL = latest_attempt_sql(HONEYBEE_L1)
