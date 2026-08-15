"""Build `Task` objects from Snowflake `public.taskattempts` rows.

Source of the step map: the `honeybee-l1-task-browsing` skill. Step IDs are
stable across the project, so steps are selected by ID rather than by type.

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
    ModelSubmission,
    RubricCriterion,
    Sxs,
    Task,
    Turn,
    TurnLink,
    TurnManifestEntry,
)
from .taxonomies import parse_criterion_category

PROJECT_ID = "6a567b80aaa4d140131eae6c"

# Step IDs, from the skill's step map. Named rather than inlined so a taxonomy
# revision is a one-line change here instead of a search across the module.
STEP_PROMPT = "step-TextCollection-e8ca527371fe"
# Input files live on the prompt step, not on either run step, and there is no
# `input_artifacts_b` anywhere in the export -- checked as a literal string across
# all 17 audited tasks, where `input_artifacts_a` appears 60 times and `_b` never.
# Its siblings on that step (`final_hardened_prompt`, `prompt_CUJ`,
# `estimated_turns`) are unsuffixed per-task fields, and per-model fields live on
# the run steps under a different convention (`attachment_modela`). So the files
# are supplied once, to both models, and the `_a` is part of the field name.
INPUT_ARTIFACTS = "input_artifacts_a"
# Files the contributor did not upload but named a location for. Each entry is a
# `multi_part_text` row of `Path to Artifacts` plus a `Number Input` ordinal.
INPUT_ARTIFACT_PATHS = "input_artifacts_a_links"
PATH_PART_ID = "Path to Artifacts"
STEP_TARGET_OUTCOME = "step-1784053126103-0h5ldq"
STEP_RUBRIC = "step-RubricCriteriaBuilder-rubric01"
STEP_RUN_A = "step-TextCollection-29cf069962f8"
# Was "step-TextCollection-2f61d970dbe7", which does not exist in any of the 5
# sampled L0 overnight tasks -- checked as a literal string across all of them.
# B's actual final-outputs/trajectory step is f7304e8fa411 in every one; the old
# ID silently made every B submission's final link and per-model outputs empty,
# which is why hydration only ever attempted A's share link.
STEP_RUN_B = "step-TextCollection-f7304e8fa411"
STEP_SCORES_A = "step-RubricCriteriaRating-5665ecba649a"
STEP_SCORES_B = "step-RubricCriteriaRating-49853f6eba5a"
STEP_TURNS_A = "step-RubricCriteriaRating-e2c61a59aef7"
STEP_TURNS_B = "step-RubricCriteriaRating-da2090d5dc04"
STEP_DIMS_A = "step-ResponseTextCollection-cb06a191fca3"
STEP_DIMS_B = "step-ResponseTextCollection-2a199aa081bb"
STEP_PREFERENCE = "step-QuantitativeModelResponseSelector-cd151dbf1b55"
# The per-turn manifest, a newer taxonomy revision that authors turn citations
# directly instead of leaving them to be read off a hydrated transcript. A
# `RubricCriteriaBuilder` step whose "criteria" are turns: each entry's `title`
# is the user's actual prompt text for that turn, and its `annotations` carry
# the per-turn share link (A only), a generated-files flag, and the
# contributor's key-turn call. Confirmed stable across both the L0 and L1
# projects sharing this taxonomy, so it is hardcoded like every other step ID
# here rather than discovered.
STEP_TURN_MANIFEST_A = "step-RubricCriteriaBuilder-bda5cf5e772d"
STEP_TURN_MANIFEST_B = "step-RubricCriteriaBuilder-0e4891d4e48f"

PREFILLER_PRODUCT_A = "before:0:data-source-string-prefiller-646efa3f869f"
PREFILLER_PRODUCT_B = "before:0:data-source-string-prefiller-88fdb8d4174c"
# The persona the prompt was commissioned against, e.g. "Backend software
# engineer". This is the "assigned domain" check 70 grades the prompt against.
PREFILLER_ASSIGNED_DOMAIN = "before:0:data-source-string-prefiller-07f6e8dd43f8"
# The template the platform prepared before the contributor wrote anything, for
# check 75. A `data-source-form-prefiller`, not a `data-source-string-prefiller`
# like the two above, and its payload shape follows: `product_name()` below reads
# a string prefiller's `content` key, but this one carries its own
# `final_hardened_prompt` key directly -- the same field name the prompt step's
# *output* uses for the contributor's submission, at a different path entirely.
PREFILLER_PRESEEDED_PROMPT = "before:0:data-source-form-prefiller-41acb4777bed"

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


def _input_artifacts(before: dict) -> list[InputArtifact] | None:
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
    step = before.get(STEP_PROMPT)
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


def _turn_manifest(before: dict, suffix: str, step_id: str) -> list[TurnManifestEntry]:
    """Read `STEP_TURN_MANIFEST_A`/`_B`, one entry per turn.

    Returns an empty list when the step is absent -- the entire fallback for
    tasks authored before this taxonomy revision: every reader of
    `turn_manifest` (and of the conversation `_conversation_from_manifest`
    derives from it) treats "empty" and "not on this schema" identically, so
    nothing else needs to branch on which taxonomy a task used.
    """
    entries: list[TurnManifestEntry] = []
    for raw in _output(before, step_id).get("criteria") or []:
        if not isinstance(raw, dict):
            continue
        ann = raw.get("annotations") or {}
        number = ann.get(f"turn_number_{suffix}")
        if isinstance(number, bool) or not (
            isinstance(number, (int, float)) or str(number or "").strip().isdigit()
        ):
            continue
        entries.append(
            TurnManifestEntry(
                turn_number=int(number),
                prompt_text=str(raw.get("title") or ""),
                share_link=str(ann.get(f"turn_link_{suffix}") or ""),
                has_model_files=bool(ann.get(f"model_generated_files_turn_{suffix}")),
                is_key_turn=bool(ann.get(f"key_turn_index_{suffix}")),
                key_turn_justification=str(ann.get(f"key_turn_justification_{suffix}") or ""),
                attachment_names=[
                    str(item.get("name") or "")
                    for item in ann.get(f"turn_attachment_{suffix}") or []
                    if isinstance(item, dict) and item.get("name")
                ],
            )
        )
    entries.sort(key=lambda e: e.turn_number)
    return entries


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


def _submission(
    before: dict, slot: str, step_id: str, product_name: str, manifest_step_id: str
) -> ModelSubmission:
    out = _output(before, step_id)
    suffix = slot.lower()
    link = _trajectory_link(out, suffix)
    manifest = _turn_manifest(before, suffix, manifest_step_id)
    output_files = _output_attachments(out.get(f"final_outputs_{suffix}"))
    return ModelSubmission(
        model=slot,  # type: ignore[arg-type]
        final_link=link,
        turn_links=_per_turn_links(out, suffix),
        transcript_pdf=_first_s3_url(out.get(f"model_{suffix}_chat_download")),
        # Was `attachment_model{suffix}`, which is not a field on any of the 5
        # sampled L0 overnight tasks -- checked as a literal string across all of
        # them. The run step's actual upload manifest is `final_outputs_{suffix}`,
        # present and non-empty on every one (images, PDFs, and plain-text files
        # alike). The old name silently made every task look like it produced no
        # attachments at all, which both hides real deliverables from check 95's
        # claimed-vs-uploaded count and starves the auditor of the one signal that
        # a text-file deliverable exists.
        attachments=[a["url"] for a in output_files],
        attachment_names={a["url"]: a["name"] for a in output_files if a["name"]},
        attachment_mime_types={
            a["url"]: (a["mime_type"] or a["file_type"])
            for a in output_files
            if a["mime_type"] or a["file_type"]
        },
        conversation=_conversation_from_manifest(manifest),
        turn_manifest=manifest,
        declared_provider=product_name or _infer_product(out, suffix, link),
    )


def _rating_blocks(before: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    """Collect scores and turn attributions per slot across every rating step.

    Step position cannot be trusted. In sampled tasks the step documented as "B
    scores" carries both `response-a` and `response-b`, while the step documented
    as "A scores" is absent entirely; reading by ID yields zero ratings for
    product A on 14 of 25 tasks and silently halves check 270's denominator.

    Entries are therefore classified by shape, which is unambiguous: a scoring
    entry carries `score`, a turn attribution carries `title` and no `score`.
    """
    scores: dict[str, dict] = {"A": {}, "B": {}}
    turns: dict[str, dict] = {"A": {}, "B": {}}

    for step_id, step in before.items():
        if "RubricCriteriaRating" not in step_id or not isinstance(step, dict):
            continue
        ratings = (step.get("output") or {}).get("responseRatings")
        if not isinstance(ratings, dict):
            continue
        for response_key, entries in ratings.items():
            slot = {"response-a": "A", "response-b": "B"}.get(str(response_key))
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


def _dimension_ratings(before: dict, slot: str) -> list[DimensionRating]:
    suffix = slot.lower()
    annotations = _annotations(before, suffix)
    ratings: list[DimensionRating] = []

    # Memory is gated: when `memory_applicable_<slot>` is "No" the three
    # memory_personalization fields are absent entirely, which is N/A rather than
    # a missing rating.
    memory_applicable = str(annotations.get(f"memory_applicable_{suffix}") or "").strip().lower()

    for prefix, dimension in DIMENSION_FIELDS.items():
        raw = annotations.get(f"{prefix}_{suffix}")
        not_applicable = prefix == "memory_personalization" and memory_applicable == "no"
        if raw is None and not not_applicable:
            continue
        rating = None
        if raw is not None and str(raw).strip().lstrip("-").isdigit():
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


def ingest_attempt(row: dict[str, Any]) -> IngestResult:
    """Build one `Task` from a `taskattempts` row.

    `row` needs a `task` (or `TASK`) key and a `response` VARIANT. Column names
    arrive upper-cased through Redash's CSV export and lower-cased through most
    drivers, so both are accepted.
    """
    lowered = {str(k).lower(): v for k, v in row.items()}
    task_id = str(lowered.get("task") or lowered.get("task_id") or "").strip()
    if not task_id:
        return IngestResult(None, "", error="row has no task id")

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

    prompt_out = _output(before, STEP_PROMPT)

    rubric: list[RubricCriterion] = []
    for raw in _output(before, STEP_RUBRIC).get("criteria") or []:
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
        for c in _output(before, STEP_TARGET_OUTCOME).get("criteria") or []
        if isinstance(c, dict)
    ]

    scores, turns = _rating_blocks(before)
    preference = _output(before, STEP_PREFERENCE)
    likert = preference.get("preferenceLikert")
    winner = preference.get("responseIdx")
    justification = ""
    field_responses = preference.get("fieldResponses")
    if isinstance(field_responses, dict):
        justification = str(field_responses.get("preference_justification") or "")

    # `responseIdx` defaults to 0, which is also a valid answer, so it cannot be
    # used to test whether a preference was recorded. The Likert can.
    recorded = likert is not None
    if not recorded and winner is not None:
        notes.append("responseIdx present without a Likert; preference treated as unrecorded")

    task = Task(
        task_id=task_id,
        project_id=PROJECT_ID,
        seeded_prompt=str(prompt_out.get("final_hardened_prompt") or ""),
        pre_seeded_prompt=(pre_seeded_prompt_text(PREFILLER_PRESEEDED_PROMPT) or None),
        assigned_domain=product_name(PREFILLER_ASSIGNED_DOMAIN),
        prompt_category=str(prompt_out.get("prompt_CUJ") or "").strip(),
        input_artifacts=_input_artifacts(before),
        target_outcome=target_outcome,
        target_deliverables=list(target_outcome),
        model_a=_submission(
            before, "A", STEP_RUN_A, product_name(PREFILLER_PRODUCT_A), STEP_TURN_MANIFEST_A
        ),
        model_b=_submission(
            before, "B", STEP_RUN_B, product_name(PREFILLER_PRODUCT_B), STEP_TURN_MANIFEST_B
        ),
        rubric=rubric,
        criterion_ratings=(
            _criterion_ratings(scores, turns, "A") + _criterion_ratings(scores, turns, "B")
        ),
        dimension_ratings=(
            _dimension_ratings(before, "A") + _dimension_ratings(before, "B")
        ),
        sxs=Sxs(
            likert=int(likert) if recorded else None,
            justification=justification,
            winner_index=int(winner) if recorded and winner is not None else None,
        ),
    )

    run_a = _output(before, STEP_RUN_A)
    key_index = run_a.get("key_turn_index_a")
    task.key_turn = KeyTurn(
        turn_index=int(key_index) if str(key_index or "").isdigit() else None,
        justification=str(run_a.get("key_turn_justification_a") or ""),
    )

    return IngestResult(task, task_id, notes=notes)


def ingest_rows(rows: list[dict[str, Any]]) -> tuple[list[Task], list[IngestResult]]:
    """Ingest a batch, returning the tasks that built and every result."""
    results = [ingest_attempt(row) for row in rows]
    return [r.task for r in results if r.task is not None], results


def load_taskattempts_csv(path) -> tuple[list[Task], list[IngestResult]]:
    """Load a Redash CSV export of `LATEST_ATTEMPT_SQL`.

    The response column holds an entire task's JSON, which runs to megabytes and
    contains newlines, so the field size limit has to come off before the reader
    will accept it.
    """
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as handle:
        return ingest_rows(list(csv.DictReader(handle)))


LATEST_ATTEMPT_SQL = f"""
with latest as (
  select *,
         row_number() over (partition by task
                            order by attempt_version desc, attempted_at desc) as rn
  from public.taskattempts
  where project = '{PROJECT_ID}'
)
select task, attempted_by, review_status, response
from latest
where rn = 1
"""
