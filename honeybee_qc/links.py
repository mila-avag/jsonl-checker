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
from .models import ModelSlot, ModelSubmission, Task
from .scoring import CheckVerdict, Measurement, build_verdict

# host -> (provider, path regex). The share id is the captured group.
SHARE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("gemini.google.com", "gemini", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    ("share.gemini.google", "gemini", re.compile(r"^/([A-Za-z0-9_-]{6,})/?$")),
    ("chatgpt.com", "gpt", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    ("chat.openai.com", "gpt", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
    # ChatGPT's newer short share path, `chatgpt.com/s/t_<id>`, which is what the
    # product's own share button now hands out. Verified live: three of these,
    # rejected by the `/share/` pattern alone, answer an unauthenticated GET with
    # 200 and the conversation's title. Missing it was not a cosmetic gap -- these
    # are public share pages, so check 90 was failing them for a bad link and
    # triage was ready to send the tasks back to their authors.
    ("chatgpt.com", "gpt", re.compile(r"^/s/([A-Za-z0-9_-]{6,})/?$")),
    ("chat.openai.com", "gpt", re.compile(r"^/s/([A-Za-z0-9_-]{6,})/?$")),
    ("claude.ai", "claude", re.compile(r"^/share/([A-Za-z0-9_-]{6,})/?$")),
]

KNOWN_HOSTS = {host for host, _, _ in SHARE_PATTERNS}
PROVIDERS = {"gemini", "gpt", "claude"}

# `cursor.com/dashboard/shared-chats?shareId=...` is deliberately NOT here, and
# the omission is a decision rather than a gap. Two Honeybee Aspirational tasks
# file their B and C final links on that host -- real shared agent runs, declared
# as "GPT 5.6 Terra Medium" and "Claude Sonnet 5 Medium - Thinking", with distinct
# share ids per slot -- and the customer's answer when shown them was that a
# trajectory link has to be a Gemini, ChatGPT or Claude share page. So those four
# links fail check 90 as an unrecognised host, which is the intended outcome.
# Those same two tasks are also the only ones in the corpus whose B and C carry no
# per-turn links at all.


@dataclass
class LinkCheck:
    url: str
    valid: bool
    provider: str = ""
    share_id: str = ""
    reason: str = ""
    # Structural validity and completeness are different questions and only the
    # first can be answered from the URL, so `classify_link` never sets this.
    # A page that opens perfectly and serves one exchange of twenty is `valid`
    # and useless to the reviewer the link exists for, and with one boolean the
    # two were indistinguishable: on two tasks in the 53-task validation batch
    # a human failed the link and the check reported clean.
    completeness: str = "not_measured"


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


# ---------------------------------------------------------------------------
# How much of the conversation the link actually served
# ---------------------------------------------------------------------------
#
# A fetch that comes back short has two possible authors and they carry opposite
# consequences. If our renderer fell short, the contributor did nothing wrong and
# charging them for it is a false fail -- which is why check 96 sets a short
# submission aside the moment the contributor's citations show the conversation
# ran past where the fetch stopped, and why it must keep doing so. If the
# submitted share link itself serves a fraction of the conversation, the
# contributor has filed evidence no reviewer can open, which is precisely what
# check 90 exists to catch and what a human QC auditor failed two tasks for while
# this check reported clean.
#
# Nothing we hold proves which of the two happened. What follows is therefore not
# a test for "the link is incomplete" but a test for "every alternative
# explanation we can check has been ruled out", and it is deliberately
# unbalanced: each condition below removes a way for our own tooling to be the
# author of the shortfall, and any one of them failing leaves the finding
# unmade. The audit's live precision is 30%, so a rule that fires where the
# evidence is merely suggestive costs more than the misses it recovers.

# The fetched conversation must be a small fraction of the corroborated one, not
# merely shorter than it. Turn counts disagree for honest reasons -- a reply
# still streaming when the page was archived, an exchange the pairing heuristic
# merged, a contributor numbering from a PDF -- and those disagreements are
# worth a turn or two, not three quarters of a conversation. Both known
# positives sit at one exchange out of twenty, a ratio of 0.05.
INCOMPLETE_DEPTH_RATIO = 0.25
# And the absolute gap must be large enough to matter. One exchange of five
# clears the ratio comfortably and is four exchanges of nothing much: how long
# a five-exchange conversation should have been is check 96's question, asked
# of the contributor's work rather than of the link that shows it.
INCOMPLETE_MIN_MISSING_EXCHANGES = 5

# Providers whose share page hands us the whole conversation in the DOM we
# archive. ChatGPT is excluded on the evidence of our own scraper: `sources.py`
# records that it virtualises its message list, so the served DOM holds only the
# messages near the top of the viewport, and the router-payload path written to
# recover the rest silently degrades to that same partial scrape whenever the
# payload will not decode. A short GPT capture is as likely to be our renderer
# as the link, and that ambiguity has to resolve in the contributor's favour.
DOM_COMPLETE_PROVIDERS = frozenset({"claude", "gemini"})


@dataclass
class ConversationDepth:
    """Exchanges fetched against the best corroborated length, per submission."""

    model: str
    provider: str
    fetched: int
    corroborated: int
    corroborated_by: list[str] = field(default_factory=list)
    # Turns the contributor cites that sit past the end of what we fetched and
    # inside what something else corroborates. These are the contributor's own
    # words about a conversation they ran, written before any of this, which is
    # what makes them evidence rather than inference.
    cited_beyond: list[int] = field(default_factory=list)
    peer_fetched: int = 0
    # The other submission came back from the same fetcher, in the same run,
    # minutes apart, at a length the spec calls a real conversation. It is the
    # only thing on record that says our tooling was working when this link came
    # back short, and without it the simplest explanation of the shortfall is
    # the tooling.
    peer_healthy: bool = False
    # The capture has both speakers in it. A page that yielded no model replies
    # is our fetch hitting a wall, an interstitial, or a login gate; it is not a
    # short conversation, and it belongs to the exported-PDF rescue rather than
    # to a link finding.
    fetch_looks_healthy: bool = False
    link_valid: bool = False

    @property
    def missing(self) -> int:
        return max(0, self.corroborated - self.fetched)

    @property
    def materially_short(self) -> bool:
        """Too little of the conversation came back to stand in for it.

        Says nothing about whose fault that is, which is why the two consumers
        that only spend a fetch or write a log line read this one and the band
        reads `link_incomplete` below.
        """
        return is_materially_short(self.fetched, self.corroborated)

    @property
    def link_incomplete(self) -> bool:
        return (
            self.link_valid
            and self.materially_short
            and self.fetch_looks_healthy
            and bool(self.cited_beyond)
            and self.peer_healthy
            and self.provider in DOM_COMPLETE_PROVIDERS
        )

    def describe(self) -> str:
        by = "; ".join(self.corroborated_by)
        return (
            f"{self.fetched} of {self.corroborated} exchanges retrieved from the "
            f"submitted link (corroborated by {by})"
        )


def is_materially_short(fetched: int, corroborated: int) -> bool:
    if corroborated <= 0 or fetched >= corroborated:
        return False
    if corroborated - fetched < INCOMPLETE_MIN_MISSING_EXCHANGES:
        return False
    return fetched < corroborated * INCOMPLETE_DEPTH_RATIO


def measure_conversation_depth(
    task: Task, sub: ModelSubmission, policy: Policy = DEFAULT_POLICY
) -> ConversationDepth:
    """Compare what one submission's link served against what the task corroborates.

    The corroborating sources are ranked by how little they depend on us. The
    other models' fetched length is the strongest: every competing slot replays
    one shared prompt script, so they should run to the same length, and it is a
    length we actually retrieved rather than one anybody asserted.
    `estimated_turns` is a declaration made before the work started. The
    contributor's citations are bounded by those two exactly as check 96 bounds
    them -- a citation past everything else on record is corroborated by
    nothing, and three tasks in the live batch cite a "Turn 35" against
    conversations of 21, 19 and 6.

    None of that applies to the simplified trajectory, which answers a
    deliberately shortened script in 1-3 turns. Its peers ran 20-odd turns and
    the task's `estimated_turns` describes the full session, so both would
    "corroborate" a length it was never meant to reach and report every
    simplified conversation as a truncated fetch. It is therefore corroborated
    by nothing but itself.
    """
    fetched = sub.exchange_count()
    is_simplified = sub.model == "S"
    peer_fetched = (
        0
        if is_simplified
        else max(
            (other.exchange_count() for other in task.submissions() if other is not sub),
            default=0,
        )
    )

    corroborated = fetched
    corroborated_by: list[str] = []
    if peer_fetched > corroborated:
        corroborated = peer_fetched
        corroborated_by.append(f"another model's fetched conversation ({peer_fetched})")
    declared = 0 if is_simplified else (task.estimated_turns or 0)
    if declared > corroborated:
        corroborated = declared
        corroborated_by.append(f"the task's declared estimated_turns ({declared})")

    cited_beyond = sorted(
        turn for turn in task.cited_turns(sub.model) if fetched < turn <= corroborated
    )
    if cited_beyond:
        corroborated_by.append(
            f"the contributor's own citations up to turn {cited_beyond[-1]}"
        )

    conversation = sub.conversation or []
    check = classify_link(sub.final_link)
    return ConversationDepth(
        model=sub.model,
        provider=check.provider,
        fetched=fetched,
        corroborated=corroborated,
        corroborated_by=corroborated_by,
        cited_beyond=cited_beyond,
        peer_fetched=peer_fetched,
        peer_healthy=peer_fetched > fetched and peer_fetched >= policy.min_turns_per_model,
        fetch_looks_healthy=(
            any(t.role == "user" for t in conversation)
            and any(t.role == "assistant" for t in conversation)
        ),
        link_valid=check.valid,
    )


@dataclass
class LinkReport:
    """Every slot's links, keyed by slot rather than by a fixed pair of fields.

    The V5 spec requires a valid final link for Model A, Model B, Model C *and*
    the simplified trajectory, and fails the task if any one of them is missing
    or invalid. Two named fields could only ever answer for two of the four, and
    would have reported a project whose C link was absent as fully linked.

    A key is present for every slot the task's project expects, with a `None`
    value where no submission was filed at all -- that distinction is what lets
    `offenders` say "missing" rather than falling silent.
    """

    finals: dict[ModelSlot, LinkCheck | None] = field(default_factory=dict)
    per_turn: dict[ModelSlot, list[LinkCheck]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    depths: list[ConversationDepth] = field(default_factory=list)

    # Kept as views onto the slot map so callers and tests written against the
    # two-model shape keep working.
    @property
    def final_a(self) -> LinkCheck | None:
        return self.finals.get("A")

    @property
    def final_b(self) -> LinkCheck | None:
        return self.finals.get("B")

    @property
    def per_turn_a(self) -> list[LinkCheck]:
        return self.per_turn.get("A", [])

    @property
    def per_turn_b(self) -> list[LinkCheck]:
        return self.per_turn.get("B", [])

    def final_links_valid(self) -> bool:
        return bool(self.finals) and all(
            check is not None and check.valid for check in self.finals.values()
        )

    def offenders(self) -> list[str]:
        out = []
        for slot, check in self.finals.items():
            if check is None:
                out.append(f"final_link_{slot}:missing")
            elif not check.valid:
                out.append(f"final_link_{slot}:{check.reason}")
        return out

    def incomplete_offenders(self) -> list[str]:
        """Links that opened and did not serve the conversation they stand for.

        Kept apart from `offenders` because these rest on a different kind of
        evidence -- what came back rather than how the URL is spelled -- and a
        caller has to be able to publish one without the other.
        """
        return [
            f"final_link_{d.model}:incomplete ({d.describe()})"
            for d in self.depths
            if d.link_incomplete
        ]


def _check_submission(sub: ModelSubmission | None) -> tuple[LinkCheck | None, list[LinkCheck]]:
    if sub is None:
        return None, []
    return classify_link(sub.final_link), [classify_link(tl.url) for tl in sub.turn_links]


def build_link_report(task: Task, policy: Policy = DEFAULT_POLICY) -> LinkReport:
    report = LinkReport()
    for slot, sub in task.slot_submissions():
        final, per_turn = _check_submission(sub)
        report.finals[slot] = final
        report.per_turn[slot] = per_turn

    for slot, sub in task.slot_submissions():
        check = report.finals.get(slot)
        if sub is None or check is None:
            continue
        depth = measure_conversation_depth(task, sub, policy)
        report.depths.append(depth)
        if depth.link_incomplete:
            check.completeness = "incomplete"
        elif check.valid and depth.corroborated and not depth.materially_short:
            check.completeness = "complete"
        if depth.materially_short and not depth.link_incomplete:
            # Said out loud even where it is not convicting: the run's own
            # record should show that a judgment about this submission was made
            # from a fraction of it, whoever lost the rest.
            report.warnings.append(
                f"model {sub.model}: {depth.describe()}, but the shortfall is not "
                f"attributable to the submitted link"
            )

    for slot, per_turn in report.per_turn.items():
        bad = [c for c in per_turn if not c.valid]
        if bad:
            report.warnings.append(
                f"model {slot}: {len(bad)} of {len(per_turn)} per-turn links are invalid"
            )
        if policy.require_per_turn_links and not per_turn:
            report.warnings.append(f"model {slot}: no per-turn links supplied")

    # One conversation filed under two slots is not a comparison. Checked across
    # every pair of slots rather than just A and B: with three products filed the
    # duplicate can as easily be B and C.
    valid_finals = [
        (slot, check)
        for slot, check in report.finals.items()
        if check is not None and check.valid and check.share_id
    ]
    for i, (left, left_check) in enumerate(valid_finals):
        for right, right_check in valid_finals[i + 1 :]:
            if left_check.share_id == right_check.share_id:
                report.warnings.append(
                    f"Model {left} and Model {right} final links point at the same "
                    "shared conversation"
                )

    for slot, sub in task.slot_submissions():
        check = report.finals.get(slot)
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
    fail_on_incomplete: bool = True,
) -> CheckVerdict:
    """Structural validity, plus any link the renderer found to be gone.

    A share page that reports itself unavailable is an invalid link, which is
    what 90 measures. A fetch that merely timed out is not passed in here.

    A third state, a link that opens and serves a fraction of the conversation,
    is measured on every run and published only when the caller asks for it.
    The measurement and the decision to convict on it are separated because the
    first is a fact about the run worth recording immediately and the second
    changes what the audit fails tasks for, at a moment when the audit's
    precision is the thing most in need of defending.
    """
    report = build_link_report(task, policy)
    incomplete = report.incomplete_offenders()
    offenders = report.offenders() + list(dead_links or [])
    if fail_on_incomplete:
        offenders += incomplete
    if task.audit_slots is not None:
        keep = []
        for item in offenders:
            marker = "final_link_"
            if marker in item:
                slot = item.split(marker, 1)[1].split(":", 1)[0]
                if not task.in_audit_scope(slot):
                    continue
            keep.append(item)
        offenders = keep
        incomplete = [i for i in incomplete if i in offenders]
    band = "clean" if not offenders else "fail"
    measurement = Measurement(
        counts={
            "dead_links": len(dead_links or []),
            "incomplete_links": len(incomplete),
            "exchanges_fetched": {d.model: d.fetched for d in report.depths},
            "exchanges_corroborated": {d.model: d.corroborated for d in report.depths},
            "final_links_checked": sum(1 for c in report.finals.values() if c),
            "final_links_valid": sum(
                1 for c in report.finals.values() if c and c.valid
            ),
            "per_turn_links": {
                slot: len(checks) for slot, checks in report.per_turn.items()
            },
            "per_turn_invalid": {
                slot: sum(1 for c in checks if not c.valid)
                for slot, checks in report.per_turn.items()
            },
            "per_turn_links_a": len(report.per_turn_a),
            "per_turn_links_b": len(report.per_turn_b),
            "per_turn_invalid_a": sum(1 for c in report.per_turn_a if not c.valid),
            "per_turn_invalid_b": sum(1 for c in report.per_turn_b if not c.valid),
        },
        notes="; ".join(report.warnings + incomplete),
    )
    return build_verdict(
        task_id=task.task_id,
        check_id=90,
        band=band,
        measurement=measurement,
        contributing_items=offenders,
        policy=policy,
    )
