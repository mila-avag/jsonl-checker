"""IO adapters that turn a PDF file or a live share link into a `Transcript`.

Everything here can fail, and all of it fails soft: an adapter returns a
`Transcript` carrying an `error` string rather than raising, so a missing driver
or a slow network degrades the audit to "unverifiable" instead of taking the run
down or, worse, accusing a contributor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import DEFAULT_POLICY, Policy
from .links import classify_link
from .transcripts import Transcript, TranscriptTurn

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# S3 serves the transcripts without checking this, but a default urllib agent is
# a common thing for edge proxies to reject outright.
PDF_USER_AGENT = "honeybee-qc/1.0"


def _pdf_text_fitz(path: str) -> str | None:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return None


def _pdf_text_pypdf(path: str) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return None


def _pdf_text_ocr(path: str) -> str | None:
    """Last resort for transcripts printed with the glyphs converted to outlines.

    Chrome's print-to-PDF can emit the page as vector drawings with no text layer,
    which every parsing backend reads as an empty document. Rasterising and running
    Tesseract is the only way back to text, and it is slow enough to stay opt-in.
    """
    try:
        import fitz
    except Exception:
        return None
    if not shutil.which("tesseract"):
        return None
    try:
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pages.append(page.get_textpage_ocr(flags=0, full=True).extractText())
        return "\n".join(pages)
    except Exception:
        return None


def _pdf_text_binary(path: str) -> str | None:
    exe = shutil.which("pdftotext")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-layout", path, "-"], capture_output=True, timeout=120, check=False
        )
        return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else None
    except Exception:
        return None


def _image_text_ocr(path: str, policy: Policy = DEFAULT_POLICY) -> tuple[str | None, str]:
    """Extract visible text from an image via Tesseract. Returns (text, error).

    This is OCR -- reading pixels that happen to be text -- not general visual
    understanding. A photo, an unlabeled chart, or a diagram with no legible
    text will correctly come back with no text, because nothing here (or
    anywhere else in this pipeline's model calls, which are text-only `claude
    -p` invocations) can describe what an image *shows*. Every failure mode
    (no binary, missing Python bindings, a corrupt file, a timeout) returns
    `(None, reason)` rather than raising, so a single bad attachment degrades
    that one attachment to "no text available" instead of taking the task down.
    """
    if not shutil.which("tesseract"):
        return None, "tesseract binary not found on PATH"
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        return None, f"OCR dependencies not installed (pytesseract/Pillow): {exc}"
    try:
        with Image.open(path) as img:
            img.load()
            text = pytesseract.image_to_string(img, timeout=policy.image_ocr_timeout_s)
    except Exception as exc:
        # Covers pytesseract.TesseractNotFoundError, TesseractError, the
        # timeout's RuntimeError subclass, and a Pillow decode failure on a
        # truncated or non-image download alike -- all of them mean "no text
        # available", not "crash the run".
        return None, f"OCR failed: {exc}"
    return text, ""


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")


def _download_image(url: str, policy: Policy) -> tuple[str, str]:
    """Cache a remote image next to the DOM snapshots. Returns (path, error).

    Mirrors `_download_pdf`: the download lands on a temporary name and is
    renamed only once complete, so an interrupted run cannot leave a truncated
    file that later looks cached.
    """
    cache_dir = Path(policy.snapshot_dir)
    suffix = next((e for e in _IMAGE_EXTENSIONS if url.lower().split("?")[0].endswith(e)), ".img")
    dest = cache_dir / f"image_{hashlib.sha256(url.encode()).hexdigest()[:16]}{suffix}"
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest), ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": PDF_USER_AGENT})
        with urlopen(request, timeout=policy.pdf_download_timeout_s) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return "", f"could not download image from {url}: {exc}"

    partial.replace(dest)
    return str(dest), ""


def read_image_transcript(path: str | None, policy: Policy = DEFAULT_POLICY) -> Transcript:
    """OCR a contributor's image attachment into a `Transcript`, PDF-adapter style.

    `path` may be a local file or a remote URL (downloaded and cached first,
    same as `read_pdf_transcript`). Every failure -- no path, download error,
    missing tesseract, no legible text, a timeout -- lands in `t.error` rather
    than raising, so a task's audit degrades to "no text available" for this
    one attachment instead of aborting.
    """
    t = Transcript(kind="image", source_id=os.path.basename(urlsplit(path or "").path or path or ""))
    if not path:
        t.error = "no image supplied"
        return t

    local_path = path
    if path.startswith(("http://", "https://")):
        local_path, error = _download_image(path, policy)
        if error:
            t.error = error
            return t

    if not os.path.exists(local_path):
        t.error = f"image not found at {path}"
        return t

    if not policy.image_ocr_fallback:
        t.error = "image OCR is disabled by policy"
        return t

    text, error = _image_text_ocr(local_path, policy)
    if error:
        t.error = error
        return t
    if not text or not text.strip():
        t.error = "OCR found no legible text in the image (blank, a photo, or a diagram with no text)"
        return t

    t.full_text = text
    return t


def _download_pdf(url: str, policy: Policy) -> tuple[str, str]:
    """Cache a remote PDF next to the DOM snapshots. Returns (path, error).

    The download lands on a temporary name and is renamed only once it completes,
    so an interrupted run cannot leave a truncated file that later looks cached.
    """
    cache_dir = Path(policy.snapshot_dir)
    dest = cache_dir / f"pdf_{hashlib.sha256(url.encode()).hexdigest()[:16]}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest), ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    try:
        request = Request(url, headers={"User-Agent": PDF_USER_AGENT})
        with urlopen(request, timeout=policy.pdf_download_timeout_s) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return "", f"could not download PDF from {url}: {exc}"

    partial.replace(dest)
    return str(dest), ""


# ---------------------------------------------------------------------------
# Plain-text deliverables
# ---------------------------------------------------------------------------

# Generous but bounded: guards against a mislabeled multi-megabyte file eating
# the model's context budget silently.
ATTACHMENT_TEXT_MAX_BYTES = 200_000


# Substrings of either the manifest's `mimeType` (e.g. "video/mp4") or its
# `fileType` (e.g. "FILE_TYPE_VIDEO") that mean "no text layer to extract,
# don't even try a GET". Checked as a substring so one set covers both fields
# without needing to know which of them is populated. Images used to be in
# this set too -- there is no text *layer*, but there can be visible text
# *in* the pixels, which `_IMAGE_HINTS` below routes through OCR instead of
# skipping outright.
_UNREADABLE_MEDIA_HINTS = ("video", "audio")
_IMAGE_HINTS = ("image",)
_PDF_HINTS = ("pdf",)
# Office files are zip containers, so the generic GET below rejects them on
# Content-Type and every .docx deliverable read as "the model produced nothing".
# They are the deliverable on 22 of the 138 tasks in the first blind batch --
# exactly the tasks whose ratings turn on a document the auditor could not open.
_OOXML_HINTS = ("officedocument", "wordprocessingml", "spreadsheetml")
_OOXML_EXTENSIONS = (".docx", ".xlsx")
# Where the text lives inside each container. A .docx keeps the body in one
# part; a .xlsx keeps cell strings in a shared table plus the sheets themselves.
_OOXML_TEXT_PARTS = ("word/document.xml", "xl/sharedStrings.xml")


def fetch_attachment_text(
    url: str, policy: Policy = DEFAULT_POLICY, mime_type: str = "", file_type: str = ""
) -> str:
    """Fetch a deliverable's text and cache it next to the DOM snapshots.

    `mime_type`/`file_type` are the upload manifest's own say-so about the
    file, used only to route the request -- a PDF gets the same multi-backend
    extraction `read_pdf_transcript` already runs for a contributor's exported
    transcript, an image is OCR'd via `read_image_transcript` for whatever
    visible text Tesseract can find (not general image understanding -- a
    photo or an unlabeled chart still yields nothing), and a video/audio file
    is never fetched at all, since no backend here turns sound or motion into
    text. Everything else still goes through a live GET with the response's
    actual `Content-Type` checked before it is trusted, exactly as before this
    hint existed: a manifest label describes the contributor's upload dialog,
    not the bytes, so a mislabelled file still fails safe into "" rather than
    being decoded as whatever the label claimed. Anything binary, oversized, or
    unreachable returns "": a deliverable this cannot read must never look like
    one it correctly read as empty, which would let the auditor conclude the
    model produced nothing.
    """
    cache_dir = Path(policy.snapshot_dir)
    dest = cache_dir / f"attachment_{hashlib.sha256(url.encode()).hexdigest()[:16]}.txt"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")

    hint = f"{mime_type} {file_type}".lower()
    if any(h in hint for h in _UNREADABLE_MEDIA_HINTS):
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text("", encoding="utf-8")
        return ""

    is_image_ext = url.lower().split("?")[0].endswith(_IMAGE_EXTENSIONS)
    if any(h in hint for h in _IMAGE_HINTS) or is_image_ext:
        transcript = read_image_transcript(url, policy)
        text = transcript.full_text if not transcript.error else ""
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text

    is_html_ext = url.lower().split("?")[0].endswith(_HTML_NAME_SUFFIXES)
    if any(h in hint for h in _HTML_MIME_HINTS) or is_html_ext:
        transcript = read_html_transcript(url, policy)
        text = transcript.full_text if not transcript.error else ""
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text

    if any(h in hint for h in _PDF_HINTS) or url.lower().split("?")[0].endswith(".pdf"):
        transcript = read_pdf_transcript(url, policy)
        text = transcript.full_text if not transcript.error else ""
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text

    is_ooxml_ext = url.lower().split("?")[0].endswith(_OOXML_EXTENSIONS)
    if any(h in hint for h in _OOXML_HINTS) or is_ooxml_ext:
        text = _read_ooxml_text(url, policy)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return text

    try:
        request = Request(url, headers={"User-Agent": PDF_USER_AGENT})
        with urlopen(request, timeout=policy.pdf_download_timeout_s) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "pdf" in content_type:
                transcript = read_pdf_transcript(url, policy)
                text = transcript.full_text if not transcript.error else ""
                cache_dir.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
                return text
            if "image/" in content_type:
                transcript = read_image_transcript(url, policy)
                text = transcript.full_text if not transcript.error else ""
                cache_dir.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
                return text
            if not ("text/" in content_type or "json" in content_type):
                return ""
            raw = response.read(ATTACHMENT_TEXT_MAX_BYTES + 1)
    except Exception:
        return ""

    if len(raw) > ATTACHMENT_TEXT_MAX_BYTES:
        return ""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text


def _read_ooxml_text(url: str, policy: Policy = DEFAULT_POLICY) -> str:
    """Pull the visible text out of a .docx/.xlsx without a third-party reader.

    Both formats are zip archives of XML, so the body is recoverable with the
    standard library alone: strip the tags from the parts named in
    `_OOXML_TEXT_PARTS` and keep the character data. Paragraph and cell
    boundaries become spaces, which is all the auditor needs -- it is asked
    whether the document says a thing, never how it was laid out.

    Fails safe to "" on anything unexpected, for the same reason every other
    branch here does: a deliverable this cannot read must not be indistinguishable
    from one correctly read as empty, or the auditor concludes the model produced
    nothing and fails a contributor for our missing parser.
    """
    import io
    import zipfile

    try:
        request = Request(url, headers={"User-Agent": PDF_USER_AGENT})
        with urlopen(request, timeout=policy.pdf_download_timeout_s) as response:
            raw = response.read(ATTACHMENT_TEXT_MAX_BYTES + 1)
    except Exception:
        return ""
    if len(raw) > ATTACHMENT_TEXT_MAX_BYTES:
        return ""

    chunks: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = set(archive.namelist())
            parts = [p for p in _OOXML_TEXT_PARTS if p in names]
            # A spreadsheet with no repeated strings keeps its values inline on
            # the sheets instead of in the shared table.
            parts += sorted(n for n in names if n.startswith("xl/worksheets/"))
            for part in parts:
                xml = archive.read(part).decode("utf-8", errors="replace")
                chunks.append(re.sub(r"<[^>]+>", " ", xml))
    except Exception:
        return ""

    text = " ".join(" ".join(chunks).split())
    return text[:ATTACHMENT_TEXT_MAX_BYTES]


def _pdf_page_count(path: str) -> int:
    try:
        import fitz
    except Exception:
        return 0
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


def read_pdf_transcript(path: str | None, policy: Policy = DEFAULT_POLICY) -> Transcript:
    """Extract text from a contributor's PDF, trying each available backend.

    Contributors' transcripts arrive as S3 URLs rather than local files, so a
    remote path is downloaded to the snapshot cache before any backend runs.
    """
    t = Transcript(kind="pdf", source_id=os.path.basename(urlsplit(path or "").path or path or ""))
    if not path:
        t.error = "no PDF supplied"
        return t

    local_path = path
    if path.startswith(("http://", "https://")):
        local_path, error = _download_pdf(path, policy)
        if error:
            t.error = error
            return t

    if not os.path.exists(local_path):
        t.error = f"PDF not found at {path}"
        return t

    backends = [_pdf_text_fitz, _pdf_text_pypdf, _pdf_text_binary]
    if policy.pdf_ocr_fallback:
        backends.append(_pdf_text_ocr)
    for backend in backends:
        text = backend(local_path)
        if text and text.strip():
            t.full_text = text
            return t

    # Distinguishing these matters: one is a missing dependency the operator can
    # fix, the other is a PDF that simply carries no text to find.
    if _pdf_page_count(local_path):
        t.error = (
            "PDF has no extractable text layer (printed as vector outlines); "
            "enable pdf_ocr_fallback to read it"
        )
    else:
        t.error = "no PDF backend could extract text (install pymupdf or pypdf)"
    return t


# ---------------------------------------------------------------------------
# Uploaded HTML / MHTML chat exports (no share URL)
# ---------------------------------------------------------------------------
#
# The 6aad55e9371b00482f57f7* A-vs-B cluster files Model B as
# `model_b_chat_download` (Gemini Spark dump-dom `.html` / Blink `.mhtml`)
# with no `*_trajectory*` share link. Treating "no final trajectory link" as
# dark-B is wrong when that export parses into real turns.

_HTML_NAME_SUFFIXES = (".html", ".htm", ".mhtml", ".mht")
_HTML_MIME_HINTS = ("text/html", "application/xhtml", "multipart/related", "message/rfc822")


def looks_like_html_chat_export(path: str) -> bool:
    """True for a Gemini Spark dump-dom / Blink MHTML chat-download path or URL."""
    from urllib.parse import unquote

    raw = unquote((path or "").split("?")[0]).lower()
    return raw.endswith(_HTML_NAME_SUFFIXES)


def _looks_like_mhtml(raw: bytes) -> bool:
    head = raw[:800].lstrip()
    return (
        head.startswith(b"From:")
        or head.startswith(b"MIME-Version:")
        or b"multipart/related" in head[:400]
        or b"Content-Type: multipart/" in head[:400]
    )


def extract_html_from_mhtml(raw: bytes) -> str:
    """Return the first text/html part of a Blink/Safari MHTML archive."""
    from email import policy as email_policy
    from email.parser import BytesParser

    message = BytesParser(policy=email_policy.default).parsebytes(raw)
    parts = [message] if not message.is_multipart() else list(message.walk())
    for part in parts:
        if part.get_content_type() not in {"text/html", "application/xhtml+xml"}:
            continue
        content = part.get_content()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content or "")
    return ""


def infer_html_provider(html: str) -> str:
    """Guess the share-page parser from dump-dom markers, Gemini first."""
    if "user-query-content" in html or "<user-query" in html:
        return "gemini"
    if "data-message-author-role" in html:
        return "gpt"
    if 'data-testid="user-message"' in html or "font-claude-response" in html:
        return "claude"
    return "gemini"


def _download_bytes(url: str, dest: Path, policy: Policy) -> str:
    """Download `url` to `dest`. Empty string on success, error text otherwise."""
    if dest.exists() and dest.stat().st_size > 0:
        return ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": PDF_USER_AGENT})
        with urlopen(request, timeout=policy.pdf_download_timeout_s) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return f"could not download chat export from {url}: {exc}"
    partial.replace(dest)
    return ""


def read_html_transcript(
    path: str | None, policy: Policy = DEFAULT_POLICY
) -> Transcript:
    """Parse a contributor-uploaded HTML or MHTML conversation dump.

    Remote paths are cached next to the DOM snapshots. No Chrome launch: these
    files already *are* the rendered page.
    """
    t = Transcript(
        kind="html",
        source_id=os.path.basename(urlsplit(path or "").path or path or ""),
    )
    if not path:
        t.error = "no HTML transcript supplied"
        return t

    local_path = path
    cache_dir = Path(policy.snapshot_dir)
    if path.startswith(("http://", "https://")):
        dest = cache_dir / f"html_{hashlib.sha256(path.encode()).hexdigest()[:16]}.bin"
        error = _download_bytes(path, dest, policy)
        if error:
            t.error = error
            return t
        local_path = str(dest)

    if not os.path.exists(local_path):
        t.error = f"HTML transcript not found at {path}"
        return t

    raw = Path(local_path).read_bytes()
    if _looks_like_mhtml(raw):
        html = extract_html_from_mhtml(raw)
        if not html.strip():
            t.error = "MHTML archive contained no text/html part"
            return t
    else:
        html = raw.decode("utf-8", errors="replace")

    snapshot = cache_dir / f"html_{hashlib.sha256(path.encode()).hexdigest()[:16]}.html"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(html, encoding="utf-8")
    t.snapshot_path = str(snapshot)

    provider = infer_html_provider(html)
    t.provider = provider
    order = [provider] + [p for p in ("gemini", "gpt", "claude") if p != provider]
    turns: list = []
    full_text = ""
    dead = False
    for candidate in order:
        turns, full_text, dead = parse_share_html(html, candidate)
        if turns:
            t.provider = candidate
            break
    t.turns = turns
    t.full_text = full_text
    t.dead_page = dead
    if dead:
        t.error = "HTML export reports the conversation as unavailable"
    elif not turns:
        t.error = "" if full_text.strip() else "HTML export contained no conversation turns"
    return t


def read_chat_export_transcript(
    path: str | None, policy: Policy = DEFAULT_POLICY
) -> Transcript:
    """`model_*_chat_download` may be a PDF, an HTML dump, or an MHTML archive."""
    if not path:
        t = Transcript(kind="html", source_id="")
        t.error = "no chat export supplied"
        return t
    if path.startswith(("http://", "https://")):
        dest = Path(policy.snapshot_dir) / (
            f"export_{hashlib.sha256(path.encode()).hexdigest()[:16]}.bin"
        )
        error = _download_bytes(path, dest, policy)
        if error:
            t = Transcript(kind="html", source_id=os.path.basename(urlsplit(path).path))
            t.error = error
            return t
        local = dest
        source = path
    else:
        local = Path(path)
        source = path
        if not local.exists():
            t = Transcript(kind="html", source_id=local.name)
            t.error = f"chat export not found at {path}"
            return t
    raw = local.read_bytes()[:16]
    if raw.startswith(b"%PDF"):
        return read_pdf_transcript(str(local) if not path.startswith("http") else path, policy)
    return read_html_transcript(source, policy)


# ---------------------------------------------------------------------------
# Share page HTML -> turns
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}

# Serialized DOM contains void elements with no closing tag. Tracking them as
# open would make every enclosing block fail to close and swallow the page.
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link",
    "meta", "param", "source", "track", "wbr",
}

# Per provider, marker sets tried in order. The first set is the innermost
# content element, which carries the least UI chrome; the fallback is the outer
# turn container, which is more durable across vendor redesigns but drags in
# button labels and toolbars. Text noise here lowers containment and could
# manufacture a false fraud finding, so the clean extraction is preferred.
PROVIDER_MARKERS: dict[str, list[dict[str, list[tuple[str, str]]]]] = {
    "gemini": [
        {
            "user": [("tag", "user-query-content")],
            "assistant": [("tag", "message-content")],
        },
        {
            "user": [("tag", "user-query")],
            "assistant": [("tag", "response-container"), ("tag", "model-response")],
        },
    ],
    "gpt": [
        {
            "user": [("attr", "data-message-author-role=user")],
            "assistant": [("attr", "data-message-author-role=assistant")],
        }
    ],
    "claude": [
        {
            "user": [("attr", "data-testid=user-message")],
            "assistant": [
                ("attr", "data-testid=assistant-message"),
                # Current markup; the two below are earlier names kept so an
                # older snapshot on disk still parses.
                ("class", "font-claude-response"),
                ("class", "font-claude-message"),
            ],
        }
    ],
}

# A bot check is the network refusing us, not a contributor's broken link, and
# caching one would keep every later run from ever seeing the conversation.
CHALLENGE_MARKERS = (
    "just a moment...",
    "enable javascript and cookies to continue",
    "checking your browser before accessing",
    "cf-browser-verification",
    "verifying you are human",
)

DEAD_PAGE_MARKERS = (
    "conversation not found",
    "this shared chat is no longer available",
    "shared conversation is no longer available",
    "no longer available",
    "page not found",
    "unable to load conversation",
    "you don't have access",
    "sign in to continue",
    # Gemini's own copy for a share that no longer resolves, rendered as a dialog on
    # an otherwise normal-looking page. Observed on five links in one batch, each
    # answering 200 and redirecting to a real-looking `/share/<id>` URL, so neither
    # the status nor the URL gives this away -- without these two phrases the page
    # parsed as "rendered text we could not turn into turns", which reads as our
    # parser failing rather than as a deleted share.
    "the link might have been deleted",
    "link doesn't exist",
)


def is_challenge_page(html: str) -> bool:
    head = html[:4000].lower()
    return any(marker in head for marker in CHALLENGE_MARKERS)


# Elements that stand in for a reply with no prose. Gemini renders an image
# answer as <generated-image><single-image><img>, which strips to nothing.
_MEDIA_TAGS = {
    "img": "image",
    "generated-image": "image",
    "single-image": "image",
    "video": "video",
    "audio": "audio",
    "canvas": "canvas",
}


class _BlockExtractor(HTMLParser):
    """Collects the text of every element matching a provider marker."""

    def __init__(self, markers: dict[str, list[tuple[str, str]]]):
        super().__init__(convert_charrefs=True)
        self.markers = markers
        self.blocks: list[tuple[str, str]] = []
        self._stack: list[str] = []
        self._open: list[tuple[str, int, list[str]]] = []  # role, depth, parts
        self._skip_depth = 0
        self._links: list[tuple[int, str]] = []  # depth, marker to emit on close

    def _note_media(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record a turn whose whole reply is a rendered artifact.

        A reply that is only an image carries no text, and dropping it as empty
        loses the turn entirely -- which then collapses every later exchange onto
        one index, because a user message with no reply before it does not start
        a new one. Standing the turn up with a marker keeps the numbering honest
        and tells the rating stage to abstain rather than judge an empty string.
        """
        if not self._open or tag not in _MEDIA_TAGS:
            return
        alt = next((v for k, v in attrs if k.lower() == "alt" and v), "")
        self._open[-1][2].append(f"[{_MEDIA_TAGS[tag]}{': ' + alt if alt else ''}]")

    def _note_link(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Mark anchor text as anchored, so a cited source reads as one.

        The extractor keeps an anchor's text and drops its href, which makes a
        linked citation and a bare mention of the same name identical strings.
        Judges then read the missing markup as missing sourcing: on task
        6a7d3a48a9485b6a031cb308 four of five draws stated positively that a turn
        offered "no link, no source" where the page carries a Sources section of
        embedded links, and QC failed the task for exactly that inaccuracy.

        The host alone, not the URL. What the claim under audit turns on is
        whether anything was linked at all, and a search-citation URL runs past a
        hundred characters against a host of fifteen -- a turn citing twenty
        sources would spend two thousand characters of the conversation budget to
        answer a yes-or-no question. Only http(s) anchors with a host qualify, so
        the provider's own in-page navigation and `javascript:` handlers stay out.
        """
        if tag != "a" or not self._open:
            return
        href = next((v for k, v in attrs if k.lower() == "href" and v), "")
        parts = urlsplit(href.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return
        host = parts.netloc.split("@")[-1].split(":")[0]
        self._links.append((len(self._stack), f"[link: {host.removeprefix('www.')}]"))

    def _close_link(self) -> None:
        # A stray close unwinds several levels at once, so drain every anchor the
        # unwind passed rather than assuming one per pop.
        while self._links and self._links[-1][0] > len(self._stack):
            _, marker = self._links.pop()
            if self._open:
                self._open[-1][2].append(marker)

    def _role_for(self, tag: str, attrs: list[tuple[str, str | None]]) -> str | None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        classes = attr_map.get("class", "").split()
        for role, rules in self.markers.items():
            for kind, value in rules:
                if kind == "tag" and tag == value:
                    return role
                if kind == "class" and value in classes:
                    return role
                if kind == "attr":
                    name, _, want = value.partition("=")
                    if attr_map.get(name) == want:
                        return role
        return None

    def handle_starttag(self, tag, attrs):
        if self._skip_depth:
            if tag in _SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = 1
            return
        if tag in _VOID_TAGS:
            self._note_media(tag, attrs)
            return
        self._stack.append(tag)
        self._note_media(tag, attrs)
        self._note_link(tag, attrs)
        # Nested matches are ignored; the outermost block owns the text.
        if not self._open:
            role = self._role_for(tag, attrs)
            if role:
                self._open.append((role, len(self._stack), []))

    def handle_startendtag(self, tag, attrs):
        self._note_media(tag, attrs)

    def handle_endtag(self, tag):
        if self._skip_depth:
            if tag in _SKIP_TAGS:
                self._skip_depth -= 1
            return
        if tag in _VOID_TAGS or tag not in self._stack:
            return
        # Unwind to the matching tag so a stray close cannot desynchronise depth.
        while self._stack:
            popped = self._stack.pop()
            self._close_link()
            if self._open and self._open[-1][1] > len(self._stack):
                role, _, parts = self._open.pop()
                text = " ".join(" ".join(parts).split())
                if text:
                    self.blocks.append((role, text))
            if popped == tag:
                break

    def handle_data(self, data):
        if self._skip_depth or not self._open:
            return
        if data.strip():
            self._open[-1][2].append(data)

    def close_all(self) -> None:
        while self._open:
            role, _, parts = self._open.pop()
            text = " ".join(" ".join(parts).split())
            if text:
                self.blocks.append((role, text))


def _strip_tags(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return " ".join(text.split())


def _extract_blocks(html: str, markers: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    parser = _BlockExtractor(markers)
    try:
        parser.feed(html)
        parser.close_all()
    except Exception:
        pass
    return parser.blocks


def _pair_into_exchanges(blocks: list[tuple[str, str]]) -> list[TranscriptTurn]:
    """A user message and the response answering it share one 1-based index.

    Consecutive user messages stay in one exchange: the user really can send two
    messages before the model answers. That only holds because a reply is never
    dropped for having no text -- an image-only answer is kept as a marker. Were
    it dropped, every later user message would look consecutive and the whole
    conversation would renumber onto one index.
    """
    turns: list[TranscriptTurn] = []
    index = 0
    prev_role: str | None = None
    for role, text in blocks:
        if role == "user" and prev_role != "user":
            index += 1
        elif index == 0:
            index = 1
        turns.append(TranscriptTurn(index=index, role=role, text=text))
        prev_role = role
    return turns


# React Router serialises the loader data as a flat table where every value is
# an index into that table. Negative indices are the encoding's constants.
_STREAM_CONSTANTS = {-1: None, -2: None, -3: None, -4: None, -5: None, -6: None, -7: 0}
_ENQUEUE = re.compile(r'enqueue\((\"(?:[^\"\\]|\\.)*\")\)', re.S)


def _decode_router_stream(html: str):
    """Rebuild the router's loader data from the serialised stream, or None."""
    chunks = _ENQUEUE.findall(html)
    if not chunks:
        return None
    try:
        payload = "".join(json.loads(c) for c in chunks)
        flat = json.loads(payload.split("\n", 1)[0])
    except Exception:
        return None
    if not isinstance(flat, list) or not flat:
        return None

    memo: dict[int, object] = {}

    def resolve(ref):
        if not isinstance(ref, int) or isinstance(ref, bool):
            return ref
        if ref < 0:
            return _STREAM_CONSTANTS.get(ref)
        if ref in memo:
            return memo[ref]
        if ref >= len(flat):
            return None
        value = flat[ref]
        if isinstance(value, list):
            # A pending promise or an error placeholder carries no data here.
            if len(value) == 2 and value[0] in ("P", "E"):
                memo[ref] = None
                return None
            out: list = []
            memo[ref] = out
            out.extend(resolve(item) for item in value)
            return out
        if isinstance(value, dict):
            out_map: dict = {}
            memo[ref] = out_map
            for key, item in value.items():
                name = flat[int(key[1:])] if key[1:].isdigit() and key.startswith("_") else key
                if isinstance(name, str):
                    out_map[name] = resolve(item)
            return out_map
        memo[ref] = value
        return value

    try:
        return resolve(0)
    except RecursionError:
        return None


def _find_conversation(node, depth: int = 0):
    """Locate the first `linear_conversation` list anywhere in the loader data."""
    if depth > 30:
        return None
    if isinstance(node, dict):
        found = node.get("linear_conversation")
        if isinstance(found, list) and found:
            return found
        for value in node.values():
            hit = _find_conversation(value, depth + 1)
            if hit:
                return hit
    elif isinstance(node, list):
        for value in node:
            hit = _find_conversation(value, depth + 1)
            if hit:
                return hit
    return None


def _gpt_turns_from_payload(html: str) -> list[TranscriptTurn]:
    root = _decode_router_stream(html)
    if root is None:
        return []
    conversation = _find_conversation(root)
    if not conversation:
        return []

    blocks: list[tuple[str, str]] = []
    for node in conversation:
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        # Tool calls and system preambles are machinery, not conversation.
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        blocks.append((role, text))

    # Every other provider is paired into exchanges, and this payload is the only
    # source that arrives as a flat message list. Numbering it per message would
    # make turn 7 mean different things for Model A and Model B in the same task,
    # which silently breaks every check that compares cited turns. It also splits
    # one reply delivered in several parts across several numbers.
    return _pair_into_exchanges(blocks)


def parse_share_html(html: str, provider: str) -> tuple[list[TranscriptTurn], str, bool]:
    """Return (turns, full_text, dead_page) for a rendered share page."""
    full_text = _strip_tags(html)
    lowered = full_text.lower()

    turns: list[TranscriptTurn] = []

    # ChatGPT virtualises its message list, so the served DOM holds only the few
    # messages near the top of the viewport. Scraping it silently truncates long
    # conversations, which would have the audit judge a model on its opening
    # exchanges alone. The router payload carries every message.
    if provider == "gpt":
        turns = _gpt_turns_from_payload(html)

    if not turns:
        for markers in PROVIDER_MARKERS.get(provider, []):
            blocks = _extract_blocks(html, markers)
            if len(blocks) >= 2:
                turns = _pair_into_exchanges(blocks)
                break

    # A dead page is only dead if it also produced no conversation. Some renders
    # legitimately carry a "sign in" affordance alongside real content.
    dead = not turns and any(m in lowered for m in DEAD_PAGE_MARKERS)
    return turns, full_text, dead


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


@dataclass
class FetchResult:
    html: str = ""
    error: str = ""
    snapshot_path: str = ""


# A browser given nowhere to write its profile still starts, still accepts the
# URL, and still exits leaving an empty output file, which is indistinguishable
# from a page that never rendered. A full disk therefore reports itself as a
# network timeout on every link at once, and triage reads that as the host
# refusing us and marks the tasks for a refetch that cannot help. The floor is
# what one Chrome profile plus its cache needs with room to spare.
MIN_FREE_DISK_BYTES = 2 * 1024**3


def free_disk_bytes(path: str | None = None) -> int:
    try:
        return shutil.disk_usage(path or tempfile.gettempdir()).free
    except Exception:  # noqa: BLE001 - a diagnostic must never end the run
        return -1


def _disk_note() -> str:
    free = free_disk_bytes()
    if free < 0:
        return ""
    return f"{free / 1024**3:.1f} GiB free on {tempfile.gettempdir()}"


def sweep_stale_browser_profiles(older_than_s: float = 3600.0) -> int:
    """Delete profile directories abandoned by killed runs.

    The fetch cleans up its own profile in a `finally`, which a SIGKILL skips.
    Each survivor holds tens of megabytes, and enough of them will exhaust the
    disk that the next run needs.
    """
    removed = 0
    cutoff = time.time() - older_than_s
    for pattern in ("hb_profile_*", "hb_dom_*"):
        for path in Path(tempfile.gettempdir()).glob(pattern):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
            except Exception:  # noqa: BLE001 - best effort housekeeping
                continue
    return removed


def fetch_rendered_html(url: str, policy: Policy = DEFAULT_POLICY) -> FetchResult:
    """Render a share page with headless Chrome.

    Chrome's --dump-dom writes the serialized DOM and then does not always exit,
    because the page holds long-lived connections open. Waiting on the process
    would hang the run, so output goes to a file, the file is polled until it
    stops growing, and the process is killed. Whatever was written is the DOM.
    """
    chrome = find_chrome()
    if not chrome:
        return FetchResult(error="no Chrome or Chromium binary found")

    free = free_disk_bytes()
    if 0 <= free < MIN_FREE_DISK_BYTES:
        return FetchResult(
            error=f"refusing to launch a browser: only {_disk_note()}, "
            f"below the {MIN_FREE_DISK_BYTES / 1024**3:.0f} GiB a render needs"
        )

    tmp = tempfile.NamedTemporaryFile(prefix="hb_dom_", suffix=".html", delete=False)
    tmp.close()
    profile = tempfile.mkdtemp(prefix="hb_profile_")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--disable-extensions",
        f"--user-data-dir={profile}",
        # Without an explicit agent Chrome advertises HeadlessChrome, which the
        # bot protection in front of some share pages answers with a challenge
        # page instead of the conversation.
        f"--user-agent={BROWSER_USER_AGENT}",
        "--virtual-time-budget=20000",
        "--dump-dom",
        url,
    ]

    proc = None
    try:
        with open(tmp.name, "wb") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL)
            deadline = time.time() + policy.fetch_timeout_s
            stable_since = None
            last_size = -1
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                size = os.path.getsize(tmp.name)
                if size > 0 and size == last_size:
                    stable_since = stable_since or time.time()
                    if time.time() - stable_since >= 2.0:
                        break
                else:
                    stable_since = None
                last_size = size
                time.sleep(0.5)
    except Exception as exc:
        return FetchResult(error=f"browser launch failed: {exc}")
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        shutil.rmtree(profile, ignore_errors=True)

    try:
        html = Path(tmp.name).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return FetchResult(error=f"could not read rendered DOM: {exc}")
    finally:
        os.unlink(tmp.name)

    if not html.strip():
        free = free_disk_bytes()
        if 0 <= free < MIN_FREE_DISK_BYTES:
            return FetchResult(
                error=f"browser produced an empty DOM, and the disk is nearly "
                f"full ({_disk_note()}) -- treat this as a local failure, not the host"
            )
        return FetchResult(error="browser produced an empty DOM (timed out)")
    return FetchResult(html=html)


def fetch_link_transcript(
    url: str, policy: Policy = DEFAULT_POLICY, snapshot_dir: str | None = None
) -> Transcript:
    """Render a share link and archive the DOM so a finding stays reproducible."""
    link = classify_link(url)
    t = Transcript(kind="link", source_id=link.share_id, provider=link.provider)
    t.fetched_at = datetime.now(timezone.utc).isoformat()

    if not link.valid:
        t.error = f"link is not a valid share URL: {link.reason}"
        return t

    cache_dir = Path(snapshot_dir or policy.snapshot_dir)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    html_path = cache_dir / f"{link.provider}_{key}.html"

    html = ""
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="replace")
        # An earlier build cached challenge pages; drop them rather than
        # reporting an empty conversation forever.
        if is_challenge_page(html):
            html_path.unlink(missing_ok=True)
            html = ""

    if not html:
        result = fetch_rendered_html(url, policy)
        if result.error:
            t.error = result.error
            return t
        html = result.html
        if is_challenge_page(html):
            t.error = "share page returned a bot challenge instead of the conversation"
            return t
        cache_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")

    t.snapshot_path = str(html_path)
    turns, full_text, dead = parse_share_html(html, link.provider)
    t.turns = turns
    t.full_text = full_text
    t.dead_page = dead
    if dead:
        t.error = "share page reports the conversation as unavailable"
    elif not turns:
        # Text without recognisable turn structure still supports comparison; the
        # vendor DOM changed, which is a parser problem, not a contributor problem.
        t.error = "" if full_text.strip() else "rendered page contained no text"

    meta = html_path.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "url": url,
                "provider": link.provider,
                "share_id": link.share_id,
                "fetched_at": t.fetched_at,
                "turns": len(turns),
                "digest": t.digest(),
                "dead_page": dead,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return t
