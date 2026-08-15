"""Rendering a conversation into prompt context, and measuring what it can support.

The audit reads a transcript, but a transcript is not the work product. When a
model produces a video, image, or document, the share page renders a player or a
preview and the turn's text collapses to something like "Your video is ready!
0:00 / 0:10". Observed on a real 16-exchange page, every delivering turn looked
exactly like that.

An auditor asked to rate outcome quality from that has nothing to work with. If it
guesses, the disagreement rate becomes noise on precisely the criteria the
customer cares most about. So placeholder turns are detected and labelled in the
prompt, and the evidence profile tells the caller how much of the conversation is
actually judgeable before a single call is spent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DEFAULT_POLICY, Policy
from .models import ModelSubmission, Task, Turn
from .sources import fetch_attachment_text
from .transcripts import Transcript

# Text a model emits in place of a deliverable it produced. The media-player
# timestamp is the strongest signal: it is chrome, not prose.
ARTIFACT_PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("media_player", re.compile(r"\b\d+:\d{2}\s*/\s*\d+:\d{2}\b")),
    ("ready", re.compile(r"\byour\s+(\w+\s+)?(video|image|audio|clip|file|document|"
                         r"presentation|slide\s?deck|spreadsheet|chart|report)\s+is\s+ready\b",
                         re.IGNORECASE)),
    ("here_is", re.compile(r"\bhere(?:'s| is)\s+(?:your|the)\s+(video|image|audio|clip|"
                           r"file|document|presentation|spreadsheet|chart|report)\b",
                           re.IGNORECASE)),
    ("produced", re.compile(r"\bi(?:'ve| have)\s+(created|generated|made|produced|rendered|"
                            r"exported|attached)\b", re.IGNORECASE)),
    ("generating", re.compile(r"\b(generating|rendering|creating)\s+(your|the)\s+\w+",
                              re.IGNORECASE)),
    # The marker the DOM extractor leaves where a reply was a rendered artifact
    # and carried no prose at all.
    ("rendered_media", re.compile(r"^\s*\[(image|video|audio|canvas)\b", re.IGNORECASE)),
)

# Below this, a turn carries no argument a reviewer could weigh.
THIN_TURN_TOKENS = 25


def _tokens(text: str) -> int:
    return len(text.split())


def detect_artifact_signals(turn: Turn) -> list[str]:
    """Named reasons to believe this turn delivered something not in the text."""
    signals = list(turn.artifacts_mentioned)
    for name, pattern in ARTIFACT_PLACEHOLDER_PATTERNS:
        if pattern.search(turn.text or ""):
            signals.append(name)
    return signals


def is_placeholder_turn(turn: Turn) -> bool:
    """A delivering turn whose text is too thin to audit."""
    if turn.role != "assistant":
        return False
    return bool(detect_artifact_signals(turn)) and _tokens(turn.text) < THIN_TURN_TOKENS


@dataclass
class EvidenceProfile:
    """What a conversation can and cannot support, computed before spending calls."""

    model: str = ""
    exchanges: int = 0
    assistant_turns: int = 0
    substantive_turns: int = 0
    placeholder_turns: int = 0
    attachments: list[str] = field(default_factory=list)
    truncated: bool = False
    # How many turns actually reached the prompt, against how many the submission
    # holds. `truncated` says a cut happened; these say how deep it was, which is
    # the number a comparison needs -- half of one conversation against all of the
    # other is a fairness problem the boolean cannot express.
    turns_rendered: int = 0
    turns_available: int = 0
    # The exported chat PDF stood in for a conversation the share page did not
    # yield. Reported because a judgment made from it is made from a print of the
    # page rather than the page: turn boundaries are not recoverable, so the text
    # is offered as one block and no turn number is derived from it.
    pdf_fallback_used: bool = False

    @property
    def artifact_dependent(self) -> bool:
        """The deliverables live outside the transcript."""
        return self.placeholder_turns > 0

    @property
    def substantive_ratio(self) -> float:
        if not self.assistant_turns:
            return 0.0
        return self.substantive_turns / self.assistant_turns

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "exchanges": self.exchanges,
            "assistant_turns": self.assistant_turns,
            "substantive_turns": self.substantive_turns,
            "placeholder_turns": self.placeholder_turns,
            "substantive_ratio": round(self.substantive_ratio, 3),
            "artifact_dependent": self.artifact_dependent,
            "attachments": list(self.attachments),
            "truncated": self.truncated,
            "turns_rendered": self.turns_rendered,
            "turns_available": self.turns_available,
            "pdf_fallback_used": self.pdf_fallback_used,
        }


def profile_evidence(sub: ModelSubmission) -> EvidenceProfile:
    names = [sub.attachment_names.get(url) or url for url in sub.attachments]
    profile = EvidenceProfile(model=sub.model, attachments=names)
    profile.exchanges = sub.exchange_count()
    for turn in sub.conversation:
        if turn.role != "assistant":
            continue
        profile.assistant_turns += 1
        if is_placeholder_turn(turn):
            profile.placeholder_turns += 1
        elif _tokens(turn.text) >= THIN_TURN_TOKENS:
            profile.substantive_turns += 1
    return profile


def conversation_from_transcript(transcript: Transcript) -> list[Turn]:
    """Adapt a scraped transcript into the input contract's turn shape."""
    return [Turn(index=t.index, role=t.role, text=t.text) for t in transcript.turns]


def numbered_prompts(task: Task) -> str:
    """Every user turn, in order, for the final-state replay rule in checks 80 and 220.

    The taskattempt form records only the seeded prompt; the follow-up turns exist
    solely inside the conversation. Falling back to the seeded prompt alone would
    make every later revision invisible and turn correct target outcomes into
    contradictions, which is exactly the trap check 80 is built to avoid. Model A
    is preferred only because both slots answer the same prompts.
    """
    prompts = [t.text for t in task.prompts if (t.text or "").strip()]

    if not prompts:
        for sub in (task.model_a, task.model_b):
            turns = [
                t.text
                for t in ((sub.conversation if sub else None) or [])
                if t.role == "user" and (t.text or "").strip()
            ]
            if turns:
                prompts = turns
                break

    if not prompts:
        return task.seeded_prompt or "(no prompts recorded)"

    return "\n\n".join(
        f"### Turn {i}\n\n{' '.join(p.split())}" for i, p in enumerate(prompts, 1)
    )


def _truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = max((limit - 40) // 2, 40)
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters omitted ...]\n{text[-half:]}", True


def render_conversation(
    sub: ModelSubmission,
    policy: Policy = DEFAULT_POLICY,
    max_chars: int | None = None,
) -> tuple[str, EvidenceProfile]:
    """Render a conversation with explicit turn numbers and artifact labels.

    Turn numbers are load-bearing: the auditor cites them, and checks 280 and 310
    compare those citations against the contributor's. They use the same 1-based
    exchange convention asserted in preflight.
    """
    profile = profile_evidence(sub)
    profile.turns_available = len(sub.conversation or [])
    budget = max_chars if max_chars is not None else policy.max_conversation_chars

    header = []
    # The model's replies are missing entirely and there is another copy of this
    # conversation on record. Reading it is the difference between a judge that
    # abstains on every dimension (or worse, rates the user's half of a
    # conversation) and one that has the model's actual words.
    pdf_text = ""
    if (
        policy.fetch_attachment_text
        and policy.transcript_pdf_fallback
        and profile.assistant_turns == 0
        and getattr(sub, "transcript_pdf", "")
    ):
        pdf_text = fetch_attachment_text(
            sub.transcript_pdf, policy, mime_type="application/pdf"
        ).strip()
        if pdf_text:
            profile.pdf_fallback_used = True
            body, _ = _truncate_middle(" ".join(pdf_text.split()), max(budget // 2, 2000))
            header.append(
                "Exported transcript of this same conversation (the share page did "
                "not yield the model's replies, so this print of it is the record "
                "of what the model actually said). It carries both speakers' text "
                "in order, but page furniture and lost role markers mean you must "
                "NOT read turn numbers off it -- use the numbered turns below for "
                f"any turn you cite:\n{body}"
            )
    label_of = {url: (sub.attachment_names.get(url) or url) for url in sub.attachments}
    unread_attachments = [label_of[url] for url in sub.attachments]
    if policy.fetch_attachment_text:
        readable: list[tuple[str, str]] = []
        unread_attachments = []
        for url in sub.attachments:
            text = fetch_attachment_text(
                url, policy, mime_type=sub.attachment_mime_types.get(url, "")
            )
            if text.strip():
                readable.append((label_of[url], text))
            else:
                unread_attachments.append(label_of[url])
        for label, text in readable:
            # Deliberately not attributed to anyone: these prompts must not
            # mention a contributor at all, so no framing can hint that prior
            # ratings exist. The file's own content carries no such framing.
            body, _ = _truncate_middle(" ".join(text.split()), policy.max_turn_chars)
            header.append(f"Deliverable file accompanying this conversation ({label}):\n{body}")
    if unread_attachments:
        header.append(
            "Files accompanying this conversation (contents NOT available to you; "
            "filenames only): " + ", ".join(unread_attachments)
        )
    if profile.artifact_dependent:
        header.append(
            f"NOTE: {profile.placeholder_turns} of {profile.assistant_turns} model turns "
            "delivered a file, image, or video whose content is not present in this "
            "transcript. Those turns are labelled below."
        )
    if sub.conversation and profile.assistant_turns == 0 and not pdf_text:
        header.append(
            "NOTE: only the user's side of each turn is available below; the "
            "model's replies were not fetched for this task. Judge turn numbering "
            "and topical relevance from the user turns shown, and treat any claim "
            "that depends on seeing what the model actually said as unverifiable."
        )

    lines: list[str] = []
    used = sum(len(h) for h in header)
    truncated = False

    for turn in sub.conversation:
        speaker = "USER" if turn.role == "user" else "MODEL"
        text = " ".join((turn.text or "").split())

        label = ""
        if turn.role == "assistant":
            signals = detect_artifact_signals(turn)
            if signals and is_placeholder_turn(turn):
                label = (
                    "  [DELIVERABLE PRODUCED HERE - its content is not in the transcript, "
                    "so its quality cannot be judged from this text]"
                )
            elif signals:
                label = "  [this turn also produced or referenced a file]"

        body, cut = _truncate_middle(text, policy.max_turn_chars)
        truncated = truncated or cut
        entry = f"[turn {turn.index}] {speaker}:{label}\n{body}"

        if used + len(entry) > budget:
            lines.append(
                f"[... conversation truncated here; {sub.exchange_count()} exchanges total ...]"
            )
            truncated = True
            break
        used += len(entry)
        profile.turns_rendered += 1
        lines.append(entry)

    profile.truncated = truncated
    rendered = "\n\n".join(header + lines) if header else "\n\n".join(lines)
    return rendered, profile


def render_comparison(
    task_model_a: ModelSubmission | None,
    task_model_b: ModelSubmission | None,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[str, list[EvidenceProfile]]:
    """Both conversations for the side-by-side Likert judgment.

    The budget is split evenly so a long Model A cannot crowd out Model B and bias
    the comparison toward whichever was rendered first. Even splitting is not the
    same as even rendering, though, and that gap is what made this check
    unreliable: a side whose turns are long is cut after one exchange while the
    other fits eight inside the identical allowance, and a Likert placed across
    that pair is a judgment about the render.

    Two things address it. Whatever one side does not spend is handed to the other,
    so a short Model A stops costing Model B the depth it did not need. And when
    the sides still land at materially different depths, the prompt says so in the
    text, above both conversations, in the terms the judge needs to abstain: how
    many exchanges each side holds and how many of them it is actually being
    shown.
    """
    per_side = max(policy.max_comparison_chars // 2, 1000)
    sides = [("Model A", task_model_a), ("Model B", task_model_b)]

    # First pass at the even split, to learn what each side actually costs.
    first: dict[str, tuple[str, EvidenceProfile] | None] = {}
    for label, sub in sides:
        first[label] = None if sub is None else render_conversation(sub, policy, per_side)

    spare = sum(
        max(0, per_side - len(rendered))
        for rendered, _ in (v for v in first.values() if v is not None)
    )

    blocks: list[str] = []
    profiles: list[EvidenceProfile] = []
    for label, sub in sides:
        if sub is None:
            blocks.append(f"## {label}\n\n(no submission)")
            continue
        rendered, profile = first[label]  # type: ignore[misc]
        # Re-render only the side that was cut, and only if the other left room.
        if profile.truncated and spare > 0:
            rendered, profile = render_conversation(sub, policy, per_side + spare)
        profiles.append(profile)
        blocks.append(f"## {label}\n\n{rendered}")

    note = _symmetry_note(profiles)
    return ("\n\n".join(([note] if note else []) + blocks), profiles)


def _symmetry_note(profiles: list[EvidenceProfile]) -> str:
    """Disclose unequal render depth, in the terms the abstention rule uses.

    Silent asymmetry is the failure mode this exists for: the judge cannot see
    that the side it is about to call worse is the side it was shown less of, and
    on a manual audit it ranked one anyway.
    """
    if len(profiles) != 2 or not any(p.truncated for p in profiles):
        return ""
    shown = [p.turns_rendered for p in profiles]
    depths = "; ".join(
        f"Model {p.model or label}: {p.turns_rendered} of {p.turns_available} turns shown"
        for label, p in zip(("A", "B"), profiles)
    )
    lopsided = (
        min(shown) / max(shown) < 0.5 if all(shown) else True
    )
    warning = (
        " These depths are not comparable: one side is being shown less than half "
        "the turns of the other. Any difference you would rank on that could sit "
        "in the part you were not shown makes this a `cannot_determine`, not a "
        "close call."
        if lopsided
        else " Weigh only what you were shown on both sides."
    )
    return (
        "## How much of each conversation you are being shown\n\n"
        f"At least one conversation below was cut to fit. {depths}.{warning}"
    )
