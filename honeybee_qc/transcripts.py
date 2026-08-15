"""Transcript representation and text-overlap scoring. Pure, no IO.

Scoring uses n-gram shingle containment rather than a diff ratio. A contributor's
PDF is a print of the share page, so it carries page furniture, wrapped lines,
and lost role markers; an exact or order-sensitive comparison would fail on
honest submissions. Shingles are the standard plagiarism-detection measure and
are hard to satisfy by coincidence: matching a 5-word window means the same
words appeared in the same order, which random topical overlap does not produce.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Literal

Role = Literal["user", "assistant"]
SourceKind = Literal["link", "pdf"]

# Page furniture a browser print adds that is not conversation content.
_PDF_FURNITURE = re.compile(
    r"(?im)^\s*(page\s+\d+\s*(of\s+\d+)?|\d+\s*/\s*\d+"
    r"|https?://\S+"
    r"|gemini|chatgpt|claude(\.ai)?"
    r"|\d{1,2}/\d{1,2}/\d{2,4}(,?\s*\d{1,2}:\d{2}(\s*[ap]m)?)?)\s*$"
)

_WORD = re.compile(r"[a-z0-9]+")

_LINE_BREAK_HYPHEN = re.compile(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)")


def normalize(text: str) -> str:
    """Fold away everything a PDF print can legitimately change."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    # A word broken across a line break ("orienta-\ntion") is the most common PDF
    # extraction artifact. Left alone it destroys the shingles containing it.
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    # Smart quotes and dashes survive rendering inconsistently.
    for a, b in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u00a0", " ")):
        text = text.replace(a, b)
    text = _PDF_FURNITURE.sub(" ", text)
    return text.lower()


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def shingles(toks: list[str], n: int = 5) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def containment(needle: str, haystack_shingles: set[tuple[str, ...]], n: int = 5) -> float:
    """Fraction of the needle's shingles that appear in the haystack.

    Asymmetric on purpose. "Does this turn appear in that document" is the
    question, and the document is expected to be much larger.
    """
    needle_shingles = shingles(tokens(needle), n)
    if not needle_shingles:
        return 1.0
    hit = len(needle_shingles & haystack_shingles)
    return hit / len(needle_shingles)


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(tokens(text)).encode("utf-8")).hexdigest()[:16]


@dataclass
class TranscriptTurn:
    index: int
    role: Role
    text: str

    @property
    def token_count(self) -> int:
        return len(tokens(self.text))


@dataclass
class Transcript:
    """A conversation recovered from one source."""

    kind: SourceKind
    source_id: str = ""
    provider: str = ""
    turns: list[TranscriptTurn] = field(default_factory=list)
    full_text: str = ""
    fetched_at: str = ""
    snapshot_path: str = ""
    error: str = ""
    # True only when the page definitively reports the share as gone. A timeout or
    # a driver failure must never set this, because it is the one fetch outcome
    # allowed to fail a task.
    dead_page: bool = False

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.full_text.strip() or self.turns)

    def exchange_count(self) -> int:
        return len({t.index for t in self.turns})

    def user_turns(self) -> list[TranscriptTurn]:
        return [t for t in self.turns if t.role == "user"]

    def assistant_turns(self) -> list[TranscriptTurn]:
        return [t for t in self.turns if t.role == "assistant"]

    def text(self) -> str:
        if self.full_text.strip():
            return self.full_text
        return "\n".join(t.text for t in self.turns)

    def shingle_set(self, n: int = 5) -> set[tuple[str, ...]]:
        return shingles(tokens(self.text()), n)

    def digest(self) -> str:
        return content_hash(self.text())


def informative_turns(
    turns: Iterable[TranscriptTurn], min_tokens: int = 5
) -> list[TranscriptTurn]:
    """Turns long enough to carry a distinguishing fingerprint.

    "ok", "thanks", and "continue" appear in every conversation, so scoring them
    would let a fabricated PDF inherit credit for matching filler.
    """
    return [t for t in turns if t.token_count >= min_tokens]
