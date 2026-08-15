"""OCR-based text extraction from image attachments.

Scope note, load-bearing for every test here: this is OCR -- reading visible
text out of pixels via Tesseract -- not general image understanding. The
model calls elsewhere in this pipeline are text-only (`claude -p`), so nothing
here can describe what an image *shows*; an image with no legible text (a
photo, an unlabeled chart, a blank screenshot) correctly yields nothing.

Every failure mode -- no tesseract binary, missing bindings, a corrupt file,
a timeout, no legible text -- must degrade to `Transcript.error` set and
`full_text` empty, never raise, so one bad attachment never takes a task's
audit down.
"""

from __future__ import annotations

import builtins
import dataclasses
import shutil

import pytest

from honeybee_qc import sources
from honeybee_qc.config import DEFAULT_POLICY
from honeybee_qc.sources import (
    _IMAGE_HINTS,
    _UNREADABLE_MEDIA_HINTS,
    _image_text_ocr,
    fetch_attachment_text,
    read_image_transcript,
)
from honeybee_qc.transcripts import Transcript

pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _make_text_image(path, text: str) -> None:
    img = Image.new("RGB", (700, 160), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 50), text, fill="black", font=font)
    img.save(path)


# ---------------------------------------------------------------------------
# Clear text OCRs correctly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_image_with_clear_text_ocrs_correctly(tmp_path):
    pytest.importorskip("pytesseract")
    path = tmp_path / "screenshot.png"
    _make_text_image(path, "Hello OCR World")

    t = read_image_transcript(str(path))
    assert t.ok
    assert "hello ocr world" in t.full_text.lower()


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_fetch_attachment_text_reads_ocr_text_from_a_local_style_path(tmp_path, monkeypatch):
    """The same public entry point PDFs go through routes images to OCR too."""
    pytest.importorskip("pytesseract")
    policy = dataclasses.replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path / "snaps"))
    img_path = tmp_path / "src.png"
    _make_text_image(img_path, "Approved by Manager")

    # fetch_attachment_text only accepts URLs; the image branch delegates
    # straight to read_image_transcript(url, policy), so swapping that in for
    # a local-path lookalike proves the routing without a network fetch.
    monkeypatch.setattr(
        sources, "read_image_transcript", lambda url, pol: read_image_transcript(str(img_path), pol)
    )
    text = fetch_attachment_text(
        "https://example.com/uploads/manager_approval.png", policy, mime_type="image/png"
    )
    assert "approved by manager" in text.lower()


# ---------------------------------------------------------------------------
# No legible text degrades gracefully
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_image_with_no_text_degrades_gracefully(tmp_path):
    pytest.importorskip("pytesseract")
    path = tmp_path / "blank.png"
    Image.new("RGB", (200, 120), color="white").save(path)

    t = read_image_transcript(str(path))
    assert not t.ok
    assert t.full_text == ""
    assert "no legible text" in t.error


def test_missing_image_path_reports_an_error_rather_than_raising():
    assert read_image_transcript(None).error == "no image supplied"
    assert "not found" in read_image_transcript("/nonexistent/x.png").error


# ---------------------------------------------------------------------------
# OCR failure / timeout / missing tesseract all degrade gracefully
# ---------------------------------------------------------------------------


def test_missing_tesseract_binary_degrades_gracefully(tmp_path, monkeypatch):
    path = tmp_path / "any.png"
    Image.new("RGB", (100, 60), color="white").save(path)

    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    text, error = _image_text_ocr(str(path))
    assert text is None
    assert "tesseract" in error.lower()

    t = read_image_transcript(str(path))
    assert not t.ok
    assert "tesseract" in t.error.lower()


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_ocr_timeout_degrades_gracefully(tmp_path, monkeypatch):
    """A Tesseract timeout (or any OCR-library exception) never crashes the run."""
    pytesseract = pytest.importorskip("pytesseract")
    path = tmp_path / "slow.png"
    _make_text_image(path, "Doesn't matter, the call is mocked to blow up")

    def _boom(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_string", _boom)

    text, error = _image_text_ocr(str(path))
    assert text is None
    assert "OCR failed" in error

    t = read_image_transcript(str(path))
    assert not t.ok
    assert t.full_text == ""


def test_ocr_dependency_missing_degrades_gracefully(tmp_path, monkeypatch):
    """If pytesseract/PIL cannot be imported at all, still no crash."""
    path = tmp_path / "any.png"
    Image.new("RGB", (100, 60), color="white").save(path)

    # Force the binary check to pass but the import to fail, exercising the
    # "bindings not installed" branch independent of what's actually on disk.
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/tesseract")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pytesseract",):
            raise ImportError("no module named pytesseract")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    text, error = _image_text_ocr(str(path))
    assert text is None
    assert "not installed" in error.lower()


# ---------------------------------------------------------------------------
# Routing constants: images are no longer lumped in with unreadable media
# ---------------------------------------------------------------------------


def test_image_hint_is_no_longer_in_the_unreadable_media_set():
    assert "image" not in _UNREADABLE_MEDIA_HINTS
    assert "image" in _IMAGE_HINTS


def test_video_and_audio_hints_still_skip_without_fetching(tmp_path):
    policy = dataclasses.replace(DEFAULT_POLICY, snapshot_dir=str(tmp_path / "snaps"))
    text = fetch_attachment_text(
        "https://example.com/clip.mp4", policy, mime_type="video/mp4"
    )
    assert text == ""
