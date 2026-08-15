"""Does the live share link actually match the uploaded PDF?

The PDF is what the audit reads, so nothing stops a contributor from pasting a
well-formed link next to a transcript that never happened. This module renders
the link and compares it to the PDF.

Three principles, because a false fraud accusation is far more expensive than a
missed one:

1. A fetch failure is never evidence of anything. Timeouts, missing drivers, and
   auth walls produce `unverifiable`, which is a review flag, not a fail.
2. Only the forward direction convicts. Low forward containment means the PDF is
   missing the conversation the link actually contains, which is the fabrication
   signal. A PDF holding extra material is merely `inconclusive` -- appendices,
   attachments, and both models in one file are all legitimate.
3. There is a deliberate abstention band between verified and contradicted. A
   task landing in it is routed to a human rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .config import DEFAULT_POLICY, Policy
from .transcripts import Transcript, containment, informative_turns, tokens

Verdict = Literal["verified", "inconclusive", "contradicted", "unverifiable", "link_dead"]

CONVICTING = ("contradicted", "link_dead")


@dataclass
class TurnMatch:
    index: int
    role: str
    token_count: int
    containment: float
    matched: bool
    excerpt: str = ""


@dataclass
class ProvenanceReport:
    task_id: str
    model: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    link_turns: int = 0
    link_turns_scored: int = 0
    matched_turns: int = 0
    forward_ratio: float = 0.0
    reverse_ratio: float = 0.0
    first_prompt_containment: float | None = None

    link_source_id: str = ""
    link_digest: str = ""
    pdf_digest: str = ""
    snapshot_path: str = ""
    turn_matches: list[TurnMatch] = field(default_factory=list)

    @property
    def convicts(self) -> bool:
        return self.verdict in CONVICTING

    @property
    def needs_review(self) -> bool:
        return self.verdict in ("inconclusive", "unverifiable")

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "link_turns": self.link_turns,
            "link_turns_scored": self.link_turns_scored,
            "matched_turns": self.matched_turns,
            "forward_ratio": round(self.forward_ratio, 4),
            "reverse_ratio": round(self.reverse_ratio, 4),
            "first_prompt_containment": (
                None
                if self.first_prompt_containment is None
                else round(self.first_prompt_containment, 4)
            ),
            "link_source_id": self.link_source_id,
            "link_digest": self.link_digest,
            "pdf_digest": self.pdf_digest,
            "snapshot_path": self.snapshot_path,
            "unmatched_turns": [
                {"index": m.index, "role": m.role, "containment": round(m.containment, 3),
                 "excerpt": m.excerpt}
                for m in self.turn_matches
                if not m.matched
            ],
        }


def compare_transcripts(
    task_id: str,
    model: str,
    link: Transcript,
    pdf: Transcript,
    policy: Policy = DEFAULT_POLICY,
) -> ProvenanceReport:
    n = policy.provenance_shingle_size
    report = ProvenanceReport(
        task_id=task_id,
        model=model,
        verdict="unverifiable",
        link_source_id=link.source_id,
        snapshot_path=link.snapshot_path,
    )

    if link.dead_page:
        report.verdict = "link_dead"
        report.reasons.append(
            f"the share page reports the conversation as unavailable: "
            f"{link.error or 'not found'}"
        )
        return report

    if link.error:
        report.reasons.append(f"link could not be read: {link.error}")
        return report

    if not link.ok:
        report.reasons.append("link rendered no conversation content")
        return report

    if not pdf.ok:
        report.reasons.append(
            f"no readable PDF transcript to compare against ({pdf.error or 'empty'})"
        )
        return report

    report.link_digest = link.digest()
    report.pdf_digest = pdf.digest()
    report.link_turns = len(link.turns)

    pdf_shingles = pdf.shingle_set(n)
    scored = informative_turns(link.turns, policy.provenance_min_turn_tokens)
    report.link_turns_scored = len(scored)

    for turn in scored:
        score = containment(turn.text, pdf_shingles, n)
        matched = score >= policy.provenance_turn_match_threshold
        report.turn_matches.append(
            TurnMatch(
                index=turn.index,
                role=turn.role,
                token_count=turn.token_count,
                containment=score,
                matched=matched,
                excerpt=" ".join(turn.text.split())[:160],
            )
        )
    report.matched_turns = sum(1 for m in report.turn_matches if m.matched)
    report.forward_ratio = (
        report.matched_turns / report.link_turns_scored if report.link_turns_scored else 0.0
    )

    link_shingles = link.shingle_set(n)
    pdf_tokens = tokens(pdf.text())
    if link_shingles and pdf_tokens:
        from .transcripts import shingles as _shingles

        pdf_only = _shingles(pdf_tokens, n)
        report.reverse_ratio = (
            len(pdf_only & link_shingles) / len(pdf_only) if pdf_only else 0.0
        )

    first_user = next((t for t in link.user_turns()), None)
    if first_user is not None and first_user.token_count >= policy.provenance_min_turn_tokens:
        report.first_prompt_containment = containment(first_user.text, pdf_shingles, n)

    report.verdict = _decide(report, policy)
    return report


def _decide(report: ProvenanceReport, policy: Policy) -> Verdict:
    # Too little signal to judge. Convicting on one or two short turns would make
    # the check a coin flip on brief conversations.
    if report.link_turns_scored < policy.provenance_min_scored_turns:
        report.reasons.append(
            f"only {report.link_turns_scored} informative turns recovered from the link; "
            f"{policy.provenance_min_scored_turns} needed to judge"
        )
        return "unverifiable"

    fwd = report.forward_ratio
    if fwd >= policy.provenance_verified_ratio:
        if report.reverse_ratio < policy.provenance_reverse_min:
            report.reasons.append(
                f"PDF matches the conversation ({fwd:.0%} of turns) but carries "
                f"substantial extra material (reverse overlap {report.reverse_ratio:.0%}); "
                "may be an appendix or both models in one file"
            )
            return "inconclusive"
        report.reasons.append(f"{fwd:.0%} of link turns are present in the PDF")
        return "verified"

    if fwd <= policy.provenance_contradicted_ratio:
        report.reasons.append(
            f"only {fwd:.0%} of the live conversation's turns appear in the PDF "
            f"({report.matched_turns}/{report.link_turns_scored})"
        )
        if report.first_prompt_containment is not None:
            report.reasons.append(
                f"opening prompt containment {report.first_prompt_containment:.0%}"
            )
        return "contradicted"

    report.reasons.append(
        f"{fwd:.0%} of link turns found in the PDF, between the "
        f"{policy.provenance_contradicted_ratio:.0%} and "
        f"{policy.provenance_verified_ratio:.0%} decision thresholds"
    )
    return "inconclusive"


# ---------------------------------------------------------------------------
# Batch-level reuse detection
# ---------------------------------------------------------------------------


@dataclass
class DuplicateFinding:
    kind: Literal["link_across_tasks", "pdf_across_tasks", "link_within_task", "pdf_within_task"]
    key: str
    task_ids: list[str]
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "key": self.key,
            "task_ids": list(self.task_ids),
            "detail": self.detail,
        }


def find_duplicates(
    submissions: list[tuple[str, str, str, str]],
) -> list[DuplicateFinding]:
    """Detect reuse of a link or PDF across the batch.

    Takes (task_id, model, share_id, pdf_digest). Every task is a distinct
    conversation, so the same share page or the same transcript appearing under
    two task IDs is copy-paste, not coincidence. Reuse across the two model slots
    of one task is a separate, milder defect: it means the comparison has only
    one conversation in it.
    """
    findings: list[DuplicateFinding] = []

    by_link: dict[str, list[tuple[str, str]]] = {}
    by_pdf: dict[str, list[tuple[str, str]]] = {}
    for task_id, model, share_id, pdf_digest in submissions:
        if share_id:
            by_link.setdefault(share_id, []).append((task_id, model))
        if pdf_digest:
            by_pdf.setdefault(pdf_digest, []).append((task_id, model))

    for key, uses in sorted(by_link.items()):
        task_ids = sorted({t for t, _ in uses})
        if len(task_ids) > 1:
            findings.append(
                DuplicateFinding(
                    "link_across_tasks", key, task_ids,
                    f"share page {key} is submitted under {len(task_ids)} different tasks",
                )
            )
        elif len(uses) > 1:
            findings.append(
                DuplicateFinding(
                    "link_within_task", key, task_ids,
                    f"share page {key} is submitted for both model slots of {task_ids[0]}",
                )
            )

    for key, uses in sorted(by_pdf.items()):
        task_ids = sorted({t for t, _ in uses})
        if len(task_ids) > 1:
            findings.append(
                DuplicateFinding(
                    "pdf_across_tasks", key, task_ids,
                    f"an identical PDF transcript is submitted under {len(task_ids)} tasks",
                )
            )
        elif len(uses) > 1:
            findings.append(
                DuplicateFinding(
                    "pdf_within_task", key, task_ids,
                    f"an identical PDF transcript is submitted for both model slots of "
                    f"{task_ids[0]}",
                )
            )
    return findings


def duplicates_convict(
    findings: list[DuplicateFinding], policy: Policy = DEFAULT_POLICY
) -> list[DuplicateFinding]:
    kinds = {"link_across_tasks", "pdf_across_tasks"}
    if not policy.duplicate_across_tasks_is_unauditable:
        return []
    return [f for f in findings if f.kind in kinds]
