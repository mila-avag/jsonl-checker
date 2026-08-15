"""Runs provenance verification for a task and maps the result onto checks.

The mapping is deliberately conservative:

  contradicted        -> 1000, unauditable. The PDF is not a transcript of the
                         link, so no finding drawn from either is trustworthy.
                         Unauditable also keeps the fraud out of quality rates,
                         which is the correct statistical treatment: a fabricated
                         task should not dilute a defect percentage.
  link_dead           -> 90 fail. A share page that reports itself unavailable is
                         an invalid link, which is exactly what 90 measures.
  inconclusive        -> review flag only.
  unverifiable        -> review flag only. Never a fail; this is the bucket every
                         network and tooling failure lands in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import DEFAULT_POLICY, Policy
from .models import ModelSubmission, Task
from .provenance import (
    DuplicateFinding,
    ProvenanceReport,
    compare_transcripts,
    duplicates_convict,
    find_duplicates,
)
from .transcripts import Transcript


class LinkFetcher(Protocol):
    def __call__(self, url: str, policy: Policy = ...) -> Transcript: ...


class PdfReader(Protocol):
    def __call__(self, path: str | None) -> Transcript: ...


@dataclass
class TaskProvenance:
    task_id: str
    reports: list[ProvenanceReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_model(self, model: str) -> ProvenanceReport | None:
        return next((r for r in self.reports if r.model == model), None)

    @property
    def contradicted(self) -> list[ProvenanceReport]:
        return [r for r in self.reports if r.verdict == "contradicted"]

    @property
    def dead_links(self) -> list[ProvenanceReport]:
        return [r for r in self.reports if r.verdict == "link_dead"]

    @property
    def needs_review(self) -> list[ProvenanceReport]:
        return [r for r in self.reports if r.needs_review]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "reports": [r.to_dict() for r in self.reports],
            "warnings": list(self.warnings),
        }


def verify_task(
    task: Task,
    policy: Policy = DEFAULT_POLICY,
    fetch: LinkFetcher | None = None,
    read_pdf: PdfReader | None = None,
) -> TaskProvenance:
    """Compare each model's live share page against its uploaded PDF."""
    if fetch is None or read_pdf is None:
        from .sources import fetch_link_transcript, read_pdf_transcript

        fetch = fetch or fetch_link_transcript
        read_pdf = read_pdf or read_pdf_transcript

    out = TaskProvenance(task_id=task.task_id)
    for sub in task.submissions():
        out.reports.append(_verify_submission(task, sub, policy, fetch, read_pdf))

    for r in out.reports:
        if r.verdict == "contradicted":
            out.warnings.append(
                f"model {r.model}: uploaded PDF does not match the live conversation"
            )
        elif r.needs_review:
            out.warnings.append(f"model {r.model}: provenance {r.verdict} ({'; '.join(r.reasons)})")
    return out


def _verify_submission(
    task: Task,
    sub: ModelSubmission,
    policy: Policy,
    fetch: LinkFetcher,
    read_pdf: PdfReader,
) -> ProvenanceReport:
    link = fetch(sub.final_link, policy)
    pdf = read_pdf(sub.transcript_pdf)
    return compare_transcripts(task.task_id, sub.model, link, pdf, policy)


def collect_duplicates(
    tasks: list[Task], provenance: list[TaskProvenance]
) -> list[DuplicateFinding]:
    """Cross-task reuse detection over whatever provenance managed to recover."""
    rows: list[tuple[str, str, str, str]] = []
    by_task = {p.task_id: p for p in provenance}
    for task in tasks:
        tp = by_task.get(task.task_id)
        for sub in task.submissions():
            report = tp.by_model(sub.model) if tp else None
            share_id = report.link_source_id if report else ""
            pdf_digest = report.pdf_digest if report else ""
            if not share_id:
                from .links import classify_link

                share_id = classify_link(sub.final_link).share_id
            rows.append((task.task_id, sub.model, share_id, pdf_digest))
    return find_duplicates(rows)


def unauditable_reasons(
    task_id: str,
    provenance: TaskProvenance | None,
    duplicates: list[DuplicateFinding],
    policy: Policy = DEFAULT_POLICY,
) -> list[str]:
    """Integrity reasons a task cannot be audited at all."""
    reasons: list[str] = []
    if provenance:
        for r in provenance.contradicted:
            reasons.append(
                f"model {r.model}: uploaded PDF does not match the live conversation "
                f"({r.matched_turns}/{r.link_turns_scored} turns present)"
            )
    for d in duplicates_convict(duplicates, policy):
        if task_id in d.task_ids:
            reasons.append(d.detail)
    return reasons


def dead_link_reasons(provenance: TaskProvenance | None) -> list[str]:
    if not provenance:
        return []
    return [
        f"final_link_{r.model}:share page unavailable" for r in provenance.dead_links
    ]
