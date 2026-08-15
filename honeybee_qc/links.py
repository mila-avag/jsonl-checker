"""Check 90 -- valid trajectory links. Fully deterministic, no model call.

Trajectory links point at public share pages from three providers: Gemini, GPT,
and Claude. Structural validation is the default. HTTP reachability is opt-in
because it introduces network flakiness and false fails on auth-gated links, and
a network timeout must never fail a task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .config import DEFAULT_POLICY, Policy
from .models import ModelSubmission, Task
from .scoring import CheckVerdict, Measurement, build_verdict

# host -> (provider, path regex). The share id is the captured group.
SHARE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("gemini.google.com", "gemini", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    ("share.gemini.google", "gemini", re.compile(r"^/([A-Za-z0-9_-]{6,})/?$")),
    ("chatgpt.com", "gpt", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    ("chat.openai.com", "gpt", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    ("claude.ai", "claude", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
]

KNOWN_HOSTS = {host for host, _, _ in SHARE_PATTERNS}
PROVIDERS = {"gemini", "gpt", "claude"}


@dataclass
class LinkCheck:
    url: str
    valid: bool
    provider: str = ""
    share_id: str = ""
    reason: str = ""


def classify_link(url: str) -> LinkCheck:
    raw = (url or "").strip()
    if not raw:
        return LinkCheck(raw, False, reason="missing")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        return LinkCheck(raw, False, reason=f"unsupported scheme {parsed.scheme or '(none)'}")

    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return LinkCheck(raw, False, reason="no host")
    if host not in KNOWN_HOSTS:
        return LinkCheck(raw, False, reason=f"unrecognised host {host}")

    path = parsed.path or "/"
    for known_host, provider, pattern in SHARE_PATTERNS:
        if host != known_host:
            continue
        m = pattern.match(path)
        if m:
            return LinkCheck(raw, True, provider=provider, share_id=m.group(1))

    # A private conversation URL is not a share link and will not open for a reviewer.
    if path.startswith("/app/") or path.startswith("/c/") or path.startswith("/chat/"):
        return LinkCheck(raw, False, reason="not a share link (private conversation URL)")
    return LinkCheck(raw, False, reason="share path does not match provider pattern")


@dataclass
class LinkReport:
    final_a: LinkCheck | None = None
    final_b: LinkCheck | None = None
    per_turn_a: list[LinkCheck] = field(default_factory=list)
    per_turn_b: list[LinkCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def final_links_valid(self) -> bool:
        return bool(self.final_a and self.final_a.valid and self.final_b and self.final_b.valid)

    def offenders(self) -> list[str]:
        out = []
        for slot, check in (("A", self.final_a), ("B", self.final_b)):
            if check is None:
                out.append(f"final_link_{slot}:missing")
            elif not check.valid:
                out.append(f"final_link_{slot}:{check.reason}")
        return out


def _check_submission(sub: ModelSubmission | None) -> tuple[LinkCheck | None, list[LinkCheck]]:
    if sub is None:
        return None, []
    return classify_link(sub.final_link), [classify_link(tl.url) for tl in sub.turn_links]


def build_link_report(task: Task, policy: Policy = DEFAULT_POLICY) -> LinkReport:
    report = LinkReport()
    report.final_a, report.per_turn_a = _check_submission(task.model_a)
    report.final_b, report.per_turn_b = _check_submission(task.model_b)

    for slot, per_turn in (("A", report.per_turn_a), ("B", report.per_turn_b)):
        bad = [c for c in per_turn if not c.valid]
        if bad:
            report.warnings.append(
                f"model {slot}: {len(bad)} of {len(per_turn)} per-turn links are invalid"
            )
        if policy.require_per_turn_links and not per_turn:
            report.warnings.append(f"model {slot}: no per-turn links supplied")

    # A single conversation filed under both slots is not a comparison.
    if (
        report.final_a
        and report.final_b
        and report.final_a.valid
        and report.final_b.valid
        and report.final_a.share_id == report.final_b.share_id
    ):
        report.warnings.append(
            "Model A and Model B final links point at the same shared conversation"
        )

    for slot, sub, check in (
        ("A", task.model_a, report.final_a),
        ("B", task.model_b, report.final_b),
    ):
        if sub and check and check.valid and sub.declared_provider:
            declared = sub.declared_provider.strip().lower()
            if declared in PROVIDERS and declared != check.provider:
                report.warnings.append(
                    f"model {slot}: link is a {check.provider} share page but the "
                    f"submission declares {declared}"
                )
    return report


def evaluate_check_90(
    task: Task,
    policy: Policy = DEFAULT_POLICY,
    dead_links: list[str] | None = None,
) -> CheckVerdict:
    """Structural validity, plus any link the renderer found to be gone.

    A share page that reports itself unavailable is an invalid link, which is
    what 90 measures. A fetch that merely timed out is not passed in here.
    """
    report = build_link_report(task, policy)
    offenders = report.offenders() + list(dead_links or [])
    band = "clean" if report.final_links_valid() and not dead_links else "fail"
    measurement = Measurement(
        counts={
            "dead_links": len(dead_links or []),
            "final_links_checked": sum(1 for c in (report.final_a, report.final_b) if c),
            "final_links_valid": sum(
                1 for c in (report.final_a, report.final_b) if c and c.valid
            ),
            "per_turn_links_a": len(report.per_turn_a),
            "per_turn_links_b": len(report.per_turn_b),
            "per_turn_invalid_a": sum(1 for c in report.per_turn_a if not c.valid),
            "per_turn_invalid_b": sum(1 for c in report.per_turn_b if not c.valid),
        },
        notes="; ".join(report.warnings),
    )
    return build_verdict(
        task_id=task.task_id,
        check_id=90,
        band=band,
        measurement=measurement,
        contributing_items=offenders,
        policy=policy,
    )
