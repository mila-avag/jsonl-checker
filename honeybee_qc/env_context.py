"""Check 85: does the prompt reference anything outside the task's universe?

The dimension has two halves, and only one of them needs a model.

The file half is arithmetic. A prompt that names `Ridgewater_PortfolioPlaybook_v2.pdf`
either has that file in the task's input manifest or it does not, and comparing
two lists of basenames settles it without spending a call. Everything in this
module is that half, plus the shared reading of what the task's universe consists
of, which the model half's prompt also needs.

The entity half -- a person, organisation, or event the prompt treats as part of
the task's world that nothing in the task supplies -- is a semantic judgment and
lives in `informed_prompts`/`informed_stages`.

Two things this module refuses to do. It never concludes a file is absent from a
manifest it cannot read: uploads arrive as CDN paths whose last segment is a
content ID, so a manifest can be non-empty and still name nothing, and treating
that as "no files provided" would fail every task that attached one. And it never
reads a prompt field the ingest adapter does not populate; on the Snowflake export
`Task.prompts` is empty and `seeded_prompt` is the only prompt text there is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .context import detect_artifact_signals
from .findings import EnvironmentContextFinding
from .models import Task, Turn

# Extensions that make a token a filename rather than prose. Deliberately a
# closed list: an open `\.\w+` pattern reads `debatemarket.com` and `v2.0` as
# files, and this half's whole value is that it never guesses.
FILE_EXTENSIONS: tuple[str, ...] = (
    "csv", "tsv", "xls", "xlsx", "xlsm", "doc", "docx", "ppt", "pptx", "pdf",
    "txt", "md", "rtf", "json", "yaml", "yml", "xml", "html", "htm", "ics",
    "zip", "tar", "gz", "png", "jpg", "jpeg", "gif", "svg", "webp", "bmp",
    "mp3", "wav", "m4a", "mp4", "mov", "avi", "py", "ipynb", "sql", "r",
    "js", "ts", "sh", "log", "eml", "msg", "key", "numbers", "pages",
)

_FILE_REFERENCE = re.compile(
    r"(?<![\w./\\-])"
    r"(?:[A-Za-z]:[\\/])?"          # a Windows drive letter, as contributors paste them
    r"(?:[\w.\-]+[\\/])*"           # directory components
    r"[\w.\-]+\.(?:" + "|".join(FILE_EXTENSIONS) + r")"
    r"(?![\w])",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Phrasings that name a file to report its absence rather than to present it.
# Over-matching here is the safe direction: it only withholds one sentence's
# exemption, and a file that really was delivered is almost always also named
# in a sentence that presents it.
_DISCLAIMER = re.compile(
    r"\b(?:"
    r"do(?:es)?n['’]?t\s+(?:have|see|appear|exist|contain)"
    r"|do(?:es)?\s+not\s+(?:have|see|appear|exist|contain)"
    r"|did(?:n['’]?t|\s+not)\s+(?:actually\s+)?"
    r"(?:produce|provide|attach|upload|create|generate|receive|get|see|find|include)"
    r"|ca(?:n['’]?t|nnot)\s+(?:find|access|see|open|locate|read|retrieve)"
    r"|could(?:n['’]?t|\s+not)\s+(?:find|access|see|open|locate|read|retrieve|be\s+found)"
    r"|(?:un(?:able|available)|no\s+access)\b"
    r"|never\s+(?:actually\s+)?"
    r"(?:provided|produced|attached|uploaded|gave|given|received|shared|created|generated|existed)"
    r"|(?:was|were|is|are)(?:n['’]?t|\s+not)\s+"
    r"(?:actually\s+)?(?:attached|provided|uploaded|included|shared|available|present|found|there)"
    r"|no\s+(?:such\s+)?file(?:\s+named)?"
    r"|not\s+(?:among|in)\s+the"
    r"|missing"
    r"|please\s+(?:re-?)?(?:upload|attach|provide|share|send|resend)"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?(?:mind\s+)?"
    r"(?:re-?)?(?:upload|attach|provide|share|send|resend)"
    r")",
    re.IGNORECASE,
)


def _basename(name: str) -> str:
    return name.strip().replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def _names_a_file(entry: str) -> bool:
    """Whether a manifest entry carries a recoverable filename.

    Scale's uploads look like `.../6463e58.../ZQjy98e0JqvrEIg` and serve no
    Content-Disposition, so the last segment is a content ID. An entry like that
    proves a file exists and says nothing about which.
    """
    base = _basename(entry)
    stem, dot, ext = base.rpartition(".")
    return bool(dot and stem and len(ext) <= 5 and ext.isalnum())


@dataclass
class InputManifest:
    """The files the task supplied, as far as the `Task` contract exposes them.

    `named` is what the file half can compare against. `opaque` entries are
    counted and reported so a reader can see that files were present but
    unidentifiable, which is the difference between "nothing was attached" and
    "we cannot tell what was attached".
    """

    named: list[str] = field(default_factory=list)
    opaque: list[str] = field(default_factory=list)
    source: str = ""
    # Filenames neither supplied nor uploaded, but mentioned somewhere in an
    # assistant turn of either model's conversation -- the model's own earlier
    # output ("here's report.pdf") or its own earlier hallucination, quoted back
    # by the user later. Kept apart from `named` because these were never
    # actually delivered as files; conflating the two would report a name the
    # model only ever typed as though it had been uploaded. They still belong in
    # the set `scan_file_references` checks a reference against, because a later
    # turn calling back to something the conversation itself already
    # established is not a claim about something outside the task's universe.
    conversation_mentioned: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.named)

    @property
    def total(self) -> int:
        return len(self.named) + len(self.opaque)

    def basenames(self) -> set[str]:
        return {_basename(n) for n in self.named}

    def known_basenames(self) -> set[str]:
        """Every basename a reference can be cleared against: uploaded/supplied
        files plus filenames the conversation already mentioned in an assistant
        turn. Distinct from `basenames()`, which stays upload-manifest-only for
        `describe()`'s reporting."""
        return self.basenames() | {_basename(n) for n in self.conversation_mentioned}

    def describe(self) -> str:
        if self.named:
            return f"input file manifest ({', '.join(sorted(self.basenames()))})"
        if self.opaque:
            return (
                f"input file manifest ({self.total} files present, none carrying a "
                "recoverable filename)"
            )
        return "input file manifest (no files recorded)"


def _locator(path: str) -> str:
    """A declared path reduced to something a basename can be taken from.

    Contributors paste these by hand, so they arrive quoted, and the ones that are
    URLs carry a query string. Both defeat a plain basename: `"C:\\...\\memo.pdf"`
    ends in `pdf"`, which is not an extension, and a Drive link ending
    `/edit?usp=drive_link&sd=true` ends in `true`, which is. Left alone the first
    silently drops a file the task really has and the second invents one called
    `edit`, and inventing one is the worse failure -- a manifest holding a single
    fictional name is `usable`, so the half would run and report every file the
    prompt names as absent.
    """
    text = (path or "").strip().strip("\"'").strip()
    for cut in ("?", "#"):
        text = text.split(cut, 1)[0]
    return text.rstrip("/\\")


def input_manifest(task: Task) -> InputManifest:
    """Every file the task carries, across both manifests `Task` exposes.

    `Task.input_artifacts` is the real one: the files the prompt step recorded,
    supplied once to both models. `ModelSubmission.attachments` is kept beside it
    because on this export it is where produced files land, and this half only
    ever adds names to the list it checks a reference against, so reading it can
    turn a finding into a clean but never the reverse.

    A `None` on `input_artifacts` means no adapter read a manifest at all, which
    is why nothing here treats an empty result as proof that no files were
    supplied. That stays the caller's decision, via `usable`.
    """
    sources = ["Task.input_artifacts"] if task.input_artifacts is not None else []
    sources.append("ModelSubmission.attachments")
    manifest = InputManifest(source=" + ".join(sources))

    for artifact in task.input_artifacts or []:
        name = (artifact.name or "").strip()
        if name:
            (manifest.named if _names_a_file(name) else manifest.opaque).append(name)
            continue
        # No name: the entry still counts, because a file that exists and cannot
        # be identified is the one state this half must not read as an empty
        # manifest. Falling back to the URL puts it in `opaque`, where it says a
        # file was supplied without claiming which -- dropping it instead would
        # report the task as having been given nothing.
        raw = (artifact.path or artifact.url or "").strip()
        locator = _locator(raw)
        if not locator:
            continue
        if _names_a_file(locator):
            manifest.named.append(locator)
        else:
            manifest.opaque.append(raw)

    for sub in task.submissions():
        for entry in sub.attachments or []:
            url = str(entry).strip()
            if not url:
                continue
            # The manifest's own recorded name for this upload, when it has
            # one -- an opaque `.../WFx0N9ATP0DbySR` URL never names a file no
            # matter how it is parsed, so falling back to it here just carries
            # the old (wrong) behaviour forward for the tasks the manifest
            # itself left this blank on.
            text = sub.attachment_names.get(url) or url
            (manifest.named if _names_a_file(text) else manifest.opaque).append(text)

    manifest.conversation_mentioned = assistant_mentioned_filenames(task)
    return manifest


def prompt_texts(task: Task) -> list[str]:
    """The prompt text this check judges, newest source first.

    Same fallback chain the informed prompts use: the recorded per-turn prompts,
    then the user turns of whichever conversation was fetched, then the seeded
    prompt on its own. Nothing here writes to `Task`, so the ingestion owner can
    populate `prompts` without this changing.
    """
    turns = [t.text for t in task.prompts if (t.text or "").strip()]
    if turns:
        return turns
    for sub in (task.model_a, task.model_b):
        fetched = [
            t.text
            for t in ((sub.conversation if sub else None) or [])
            if t.role == "user" and (t.text or "").strip()
        ]
        if fetched:
            return fetched
    return [task.seeded_prompt] if (task.seeded_prompt or "").strip() else []


def prompt_source(task: Task) -> str:
    if any((t.text or "").strip() for t in task.prompts):
        return "recorded prompt turns"
    for sub in (task.model_a, task.model_b):
        if any(
            t.role == "user" and (t.text or "").strip()
            for t in ((sub.conversation if sub else None) or [])
        ):
            return f"user turns of the Model {sub.model} conversation"
    if (task.seeded_prompt or "").strip():
        return "seeded prompt only"
    return "none"


def universe_materials(task: Task) -> list[str]:
    """What a reference can be checked against, named for the finding's evidence.

    An empty list means absence cannot be established at all: with only the
    prompt in hand, every reference it makes is unfalsifiable, and the check must
    abstain rather than treat our own thin context as the contributor's error.
    """
    materials: list[str] = []
    manifest = input_manifest(task)
    if manifest.total:
        materials.append(manifest.describe())
    if task.target_outcome:
        materials.append(f"target outcome list ({len(task.target_outcome)} entries)")
    if task.target_deliverables and task.target_deliverables != task.target_outcome:
        materials.append(
            f"target deliverables ({len(task.target_deliverables)} entries)"
        )
    for sub in task.submissions():
        if sub.conversation:
            materials.append(
                f"Model {sub.model} conversation ({sub.exchange_count()} exchanges)"
            )
    return materials


def file_references(text: str) -> list[str]:
    """Filenames the text names, in order, deduplicated on the basename.

    URLs are stripped first. A linked file is fetched at read time rather than
    supplied in the manifest, so `https://example.com/report.pdf` is not a claim
    that `report.pdf` was provided.
    """
    stripped = _URL.sub(" ", text or "")
    return _dedupe_on_basename(m.group(0) for m in _FILE_REFERENCE.finditer(stripped))


def _dedupe_on_basename(references: Iterable[str]) -> list[str]:
    """First spelling of each basename wins, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for ref in references:
        key = _basename(ref)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _disclaims_file(sentence: str) -> bool:
    """The sentence names a file in order to say it is *not* here.

    Distinguishing this from a claim is the whole point of the caller: "here is
    report.pdf" puts a name on the table, while "I can't find report.pdf"
    names it only to report its absence. Judged per sentence rather than per
    turn because a single reply routinely does both -- delivering one file and
    asking for another.
    """
    return bool(_DISCLAIMER.search(sentence or ""))


def assistant_mentioned_filenames(task: Task) -> list[str]:
    """Filenames either model's own assistant turns put on the table as present.

    A later turn calling `report.pdf` by name is not a claim about something
    outside the task's universe when the model itself already established that
    name -- whether the file was really delivered (a genuine deliverable) or
    only ever typed (a name the model claimed and the user later quoted back
    while complaining it was never given). Either way the conversation, not the
    prompt, introduced it, so this half's job is only to notice it was said,
    not to judge whether it was true.

    Mentions that *disclaim* the file are excluded, and that exclusion is what
    keeps the check alive: the most natural way for a model to answer a prompt
    presupposing a missing file is to name it while asking for it ("I don't
    have external_ledger.xlsx -- can you attach it?"). Counting that as
    established would let the model's own confirmation of the violation excuse
    it. A name is exempted when at least one sentence somewhere in either
    trajectory asserts it without disclaiming, so a reply that delivers
    `report.pdf` and in the next breath asks for `ledger.xlsx` exempts only the
    first.
    """
    out: list[str] = []
    for sub in (task.model_a, task.model_b):
        if sub is None:
            continue
        for turn in sub.conversation:
            if turn.role != "assistant" or not (turn.text or "").strip():
                continue
            for sentence in _SENTENCE_SPLIT.split(turn.text or ""):
                if _disclaims_file(sentence):
                    continue
                out.extend(file_references(sentence))
    return _dedupe_on_basename(out)


def unnamed_artifact_references(text: str) -> list[str]:
    """Signals the text refers to a file without naming one.

    Reuses the artifact detector the transcript renderer uses, which already
    knows the phrasings that mean a file is in play ("here is the report",
    "I have attached"). These are reported, never counted: an unnamed reference
    is satisfied by any file in the manifest, so it can only be judged when the
    manifest is empty, and an empty manifest is exactly the case this module
    cannot distinguish from an unread one.
    """
    return detect_artifact_signals(Turn(index=1, role="assistant", text=text or ""))


def _quote_for(text: str, reference: str) -> str:
    """The sentence the reference sits in, so a finding carries its own evidence."""
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        if reference in sentence:
            return " ".join(sentence.split())[:400]
    return " ".join((reference or "").split())


@dataclass
class FileReferenceScan:
    """The deterministic half's result, including why it could not run."""

    prompt_available: bool = False
    prompt_source: str = "none"
    manifest: InputManifest = field(default_factory=InputManifest)
    named_references: list[str] = field(default_factory=list)
    unnamed_references: list[str] = field(default_factory=list)
    findings: list[EnvironmentContextFinding] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        """A comparison was actually possible.

        Both sides have to be readable. No prompt text means nothing to scan; a
        manifest naming nothing means there is no list to check a name against.
        """
        return self.prompt_available and self.manifest.usable

    @property
    def blocked_reason(self) -> str:
        if not self.prompt_available:
            return "no prompt text is recorded on this task"
        if not self.manifest.usable:
            if self.manifest.opaque:
                return (
                    f"{self.manifest.total} attached files carry no recoverable "
                    "filename, so a named reference cannot be matched against them"
                )
            return (
                "the task exposes no input file manifest; input artifacts are not "
                "ingested, so an empty manifest does not mean no files were supplied"
            )
        return ""

    def to_dict(self) -> dict:
        return {
            "prompt_source": self.prompt_source,
            "manifest_named": len(self.manifest.named),
            "manifest_opaque": len(self.manifest.opaque),
            "manifest_conversation_mentioned": len(self.manifest.conversation_mentioned),
            "named_references": list(self.named_references),
            "unnamed_references": list(self.unnamed_references),
            "ran": self.ran,
            "blocked_reason": self.blocked_reason,
        }


def scan_file_references(task: Task) -> FileReferenceScan:
    """Compare the files the prompt names against the files the task supplied."""
    texts = prompt_texts(task)
    joined = "\n\n".join(texts)
    scan = FileReferenceScan(
        prompt_available=bool(joined.strip()),
        prompt_source=prompt_source(task),
        manifest=input_manifest(task),
        named_references=file_references(joined),
        unnamed_references=unnamed_artifact_references(joined),
    )
    if not scan.ran:
        return scan

    provided = scan.manifest.known_basenames()
    checked_against = [scan.manifest.describe()]
    if scan.manifest.conversation_mentioned:
        checked_against.append(
            "conversation turns ("
            + ", ".join(sorted({_basename(n) for n in scan.manifest.conversation_mentioned}))
            + ")"
        )
    for reference in scan.named_references:
        if _basename(reference) in provided:
            continue
        scan.findings.append(
            EnvironmentContextFinding(
                reference=reference,
                kind="file",
                quote=_quote_for(joined, reference),
                checked_against=list(checked_against),
                why_outside_universe=(
                    f"The prompt refers to {reference}, which is not among the "
                    f"{len(provided)} files supplied to the task."
                ),
                basis="deterministic",
            )
        )
    return scan
