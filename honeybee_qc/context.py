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
from .links import is_materially_short
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

# Emitted where a submission's conversation is wholly absent. The lead sentence
# is a separate constant because `informed_prompts.UNVERIFIABLE_GUIDANCE` quotes
# it into a marker list it describes to the judge as exhaustive, and a marker the
# renderer emits but the guidance omits is worse than no marker at all: the judge
# has been told it may trust the list.
NO_CONVERSATION_MARKER_LEAD = "NOTE: none of this conversation was retrieved."
NO_CONVERSATION_MARKER = (
    NO_CONVERSATION_MARKER_LEAD
    + " The share page did not yield it and no exported transcript was "
    "available, so no turn of it appears below. Every claim that turns on what "
    "either speaker said here is unverifiable and MUST NOT be reported as a "
    "defect."
)


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
    # Two different losses used to share one boolean, and conflating them is what
    # turned a false fail into a false pass. `tail_truncated` means the render ran
    # out of budget and whole turns never reached the prompt -- the judge does not
    # know what it did not see. `elided_turns` counts turns whose *middle* was cut
    # to `max_turn_chars` with both ends kept and the cut labelled in place: the
    # turn is there, its boundaries are there, and a judge reading it can tell
    # exactly what it is missing.
    #
    # Only the first is evidence loss a per-item judgment can hide behind. On the
    # scored run, three of the four tasks whose check-300 gaps were all dropped
    # "for truncation" had every single turn rendered and were flagged solely
    # because one turn ran past 12000 characters -- so the check reported clean on
    # a conversation it had read end to end.
    tail_truncated: bool = False
    elided_turns: int = 0
    # How many turns actually reached the prompt, against how many the submission
    # holds. `truncated` says a cut happened; these say how deep it was, which is
    # the number a comparison needs -- half of one conversation against all of the
    # other is a fairness problem the boolean cannot express.
    turns_rendered: int = 0
    turns_available: int = 0
    # How long this conversation is corroborated to be by evidence outside our
    # fetch of it, stamped on the submission by hydration. 0 means nothing
    # corroborates it either way, not that the capture is whole.
    corroborated_exchanges: int = 0
    # The exported chat PDF stood in for a conversation the share page did not
    # yield. Reported because a judgment made from it is made from a print of the
    # page rather than the page: turn boundaries are not recoverable, so the text
    # is offered as one block and no turn number is derived from it.
    pdf_fallback_used: bool = False

    @property
    def truncated(self) -> bool:
        """Anything at all was cut, which is what the prompts' abstention markers
        describe and what the reports have always published."""
        return self.tail_truncated or self.elided_turns > 0

    @property
    def artifact_dependent(self) -> bool:
        """The deliverables live outside the transcript."""
        return self.placeholder_turns > 0

    @property
    def materially_short(self) -> bool:
        """The capture stands in for a conversation several times its length."""
        return is_materially_short(self.exchanges, self.corroborated_exchanges)

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
            "tail_truncated": self.tail_truncated,
            "elided_turns": self.elided_turns,
            "turns_rendered": self.turns_rendered,
            "turns_available": self.turns_available,
            "corroborated_exchanges": self.corroborated_exchanges,
            "materially_short": self.materially_short,
            "pdf_fallback_used": self.pdf_fallback_used,
        }


def profile_evidence(sub: ModelSubmission) -> EvidenceProfile:
    names = [sub.attachment_names.get(url) or url for url in sub.attachments]
    profile = EvidenceProfile(model=sub.model, attachments=names)
    profile.exchanges = sub.exchange_count()
    profile.corroborated_exchanges = sub.corroborated_exchanges
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


def golden_answer_block(task: Task) -> str:
    """The customer's seeded unique answer, when the project collected one.

    Fed into Rubric Evaluation, Justifications, and Prompt Clarity. It is
    customer-authored ground truth, not a contributor rating, so showing it
    to those judges does not break blindness: `assert_blind` still keys off
    the contributor's own justification text.
    """
    text = (task.golden_answer or "").strip()
    if not text:
        return ""
    return (
        "\n## Customer-authored ground-truth final answer\n\n"
        f"{text}\n\n"
        "This is the customer's seeded unique answer, not a contributor rating. "
        "Use it to judge whether the prompt has a unique ground truth and whether "
        "the conversation reached that answer.\n"
    )


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
        for sub in task.submissions():
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
    max_turn_chars: int | None = None,
    list_attachment_names: bool = True,
) -> tuple[str, EvidenceProfile]:
    """Render a conversation with explicit turn numbers and artifact labels.

    Turn numbers are load-bearing: the auditor cites them, and checks 280 and 310
    compare those citations against the contributor's. They use the same 1-based
    exchange convention asserted in preflight.

    Both budgets are overridable because the adjudication pass needs a wider one
    than the blind pass: it makes one call about one item rather than twelve about
    twelve, so it can afford to read what the blind pass had to skim.
    """
    profile = profile_evidence(sub)
    profile.turns_available = len(sub.conversation or [])
    budget = max_chars if max_chars is not None else policy.max_conversation_chars

    header = []
    # The model's replies are missing, wholly or near enough, and there is
    # another copy of this conversation on record. Reading it is the difference
    # between a judge that abstains on every dimension (or worse, rates the
    # user's half of a conversation) and one that has the model's actual words.
    #
    # The trigger used to be "no replies at all", which turned the rescue off
    # for the case it is most needed on: a capture holding exactly one of
    # nineteen exchanges has one assistant turn, so the test failed and a
    # contributor's own 19-turn export sat unread while four of that model's
    # justifications were abstained on for want of the turns inside it. A
    # zero-turn fetch at least announces itself in the run log; a one-turn fetch
    # reports success. Materially short means several times shorter than what
    # something other than our own fetch says the conversation was, so the
    # question of whose fault the shortfall is never arises: reading the
    # contributor's export costs one download and answers it either way.
    pdf_text = ""
    if (
        policy.fetch_attachment_text
        and policy.transcript_pdf_fallback
        and (profile.assistant_turns == 0 or profile.materially_short)
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
    if unread_attachments and list_attachment_names:
        # Withheld from the artifact extraction, which is asked to read the
        # delivered filenames out of the conversation and is then compared
        # against this very manifest. A turn rendering as "[image] [image]" has
        # no name in it, so the judge took the name from here and the comparison
        # matched what it had just been handed: 24 claimed, 24 uploaded, nothing
        # missing, on a task the customer failed for preserving nothing.
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
    if not sub.conversation and not pdf_text:
        # The case that most needs saying and was the only one that said
        # nothing. The partial-capture note below is guarded on `sub.conversation`,
        # so a submission whose share page yielded nothing at all rendered as a
        # bare heading followed by empty space, and the judge was separately told
        # -- by `UNVERIFIABLE_GUIDANCE` -- that the listed markers are the only
        # warning it gets. Two justifications in the live batch worked out from
        # first principles that they had been shown nothing; that they did so is
        # luck, not design.
        header.append(NO_CONVERSATION_MARKER)
    elif sub.conversation and profile.assistant_turns == 0 and not pdf_text:
        header.append(
            "NOTE: only the user's side of each turn is available below; the "
            "model's replies were not fetched for this task. Judge turn numbering "
            "and topical relevance from the user turns shown, and treat any claim "
            "that depends on seeing what the model actually said as unverifiable."
        )

    lines: list[str] = []
    used = sum(len(h) for h in header)
    turn_budget = max_turn_chars if max_turn_chars is not None else policy.max_turn_chars

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

        body, cut = _truncate_middle(text, turn_budget)
        entry = f"[turn {turn.index}] {speaker}:{label}\n{body}"

        if used + len(entry) > budget:
            lines.append(
                f"[... conversation truncated here; {sub.exchange_count()} exchanges total ...]"
            )
            profile.tail_truncated = True
            break
        used += len(entry)
        profile.turns_rendered += 1
        profile.elided_turns += 1 if cut else 0
        lines.append(entry)

    rendered = "\n\n".join(header + lines) if header else "\n\n".join(lines)
    return rendered, profile


def render_comparison(
    task_model_a: ModelSubmission | None,
    task_model_b: ModelSubmission | None,
    policy: Policy = DEFAULT_POLICY,
    max_chars: int | None = None,
    max_turn_chars: int | None = None,
) -> tuple[str, list[EvidenceProfile]]:
    """Both conversations for one side-by-side Likert judgment.

    The two arguments are the *pair being compared*, not slots A and B. A task
    may record several comparisons -- Honeybee Aspirational records A-vs-B and
    A-vs-C -- and each is rendered on its own so that neither inherits the
    other's budget or its asymmetry. Labels follow each submission's own slot,
    so an A-vs-C render says "Model C" rather than calling it B.

    The budget is split evenly so a long left side cannot crowd out the right
    and bias the comparison toward whichever was rendered first. Even splitting
    is not the same as even rendering, though, and that gap is what made this
    check unreliable: a side whose turns are long is cut after one exchange
    while the other fits eight inside the identical allowance, and a Likert
    placed across that pair is a judgment about the render.

    Two things address it. Whatever one side does not spend is handed to the other,
    so a short Model A stops costing Model B the depth it did not need. And when
    the sides still land at materially different depths, the prompt says so in the
    text, above both conversations, in the terms the judge needs to abstain: how
    many exchanges each side holds and how many of them it is actually being
    shown.
    """
    budget = max_chars if max_chars is not None else policy.max_comparison_chars
    per_side = max(budget // 2, 1000)
    # Positional defaults only cover the case where a side is absent and so has
    # no slot to name itself by; a present submission always labels itself.
    sides = [
        (f"Model {sub.model}" if sub is not None else fallback, sub)
        for fallback, sub in (("Model A", task_model_a), ("Model B", task_model_b))
    ]

    # First pass at the even split, to learn what each side actually costs.
    first: dict[str, tuple[str, EvidenceProfile] | None] = {}
    for label, sub in sides:
        first[label] = (
            None
            if sub is None
            else render_conversation(sub, policy, per_side, max_turn_chars)
        )

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
        # Re-render only the side whose turns were dropped for budget, and only if
        # the other left room. Spare budget cannot restore an elided turn's middle,
        # which is capped per turn rather than by what the conversation spends.
        if profile.tail_truncated and spare > 0:
            rendered, profile = render_conversation(
                sub, policy, per_side + spare, max_turn_chars
            )
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
    if len(profiles) != 2 or not any(p.tail_truncated for p in profiles):
        return ""
    shown = [p.turns_rendered for p in profiles]
    depths = "; ".join(
        f"Model {p.model or '?'}: {p.turns_rendered} of {p.turns_available} turns shown"
        for p in profiles
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
