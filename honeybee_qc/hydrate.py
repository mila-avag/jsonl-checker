"""Filling in conversation text from the contributors' share links.

A task ingested from Snowflake carries ratings, a rubric, and links, but no
conversation: the product transcripts live behind the share URLs. The rating
stage judges the models' behaviour, so without this step every one of its
judgments abstains for lack of evidence and the calls are wasted.

Fetching is deliberately separated from auditing. It is slow, it touches the
network, and it fails in ways that have nothing to do with task quality, so it
runs once up front and reports what it could not retrieve rather than letting a
fetch failure masquerade as a contributor's error.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import DEFAULT_POLICY, Policy
from .context import conversation_from_transcript
from .links import measure_conversation_depth
from .models import Task
from .transcripts import Transcript


@dataclass
class HydrationReport:
    submissions: int = 0
    hydrated: int = 0
    turns: int = 0
    dead_pages: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    # Submissions that came back, were counted as hydrated, and hold a fraction
    # of the conversation the rest of the task corroborates. A run that fetched
    # one exchange of twenty reported "hydrated 100/106" with that submission
    # inside the 100 and no operator could see it, which is the whole reason
    # nine downstream checks abstained in silence and the task passed.
    short_captures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "submissions": self.submissions,
            "hydrated": self.hydrated,
            "turns": self.turns,
            "dead_pages": self.dead_pages,
            "failures": self.failures,
            "short_captures": self.short_captures,
        }


def hydrate_conversations(
    tasks: list[Task],
    policy: Policy = DEFAULT_POLICY,
    fetch=None,
    workers: int = 4,
) -> HydrationReport:
    """Populate every submission's turns from its final share link, in place."""
    if fetch is None:
        from .sources import fetch_link_transcript

        def fetch(url: str) -> Transcript:
            return fetch_link_transcript(url, policy, policy.snapshot_dir)

    # Every filed conversation, the simplified trajectory included: its link has
    # to resolve and the files it produced have to be accounted for, so it needs
    # fetching for exactly the same reasons a product's does.
    pairs = [(task, sub) for task in tasks for sub in task.trajectories()]
    report = HydrationReport(submissions=len(pairs))

    # One fetch per distinct URL. The snapshot cache would collapse duplicates
    # anyway, but only after paying for a second browser launch.
    urls = {sub.final_link for _, sub in pairs if sub.final_link}
    transcripts: dict[str, Transcript] = {}
    if urls:
        ordered = sorted(urls)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for url, transcript in zip(ordered, pool.map(fetch, ordered)):
                transcripts[url] = transcript

    for task, sub in pairs:
        label = f"{task.task_id}/{sub.model}"
        if not sub.final_link:
            # No share URL is not "dark" when the contributor uploaded the
            # conversation as HTML / MHTML / PDF on model_*_chat_download.
            export = (getattr(sub, "transcript_pdf", "") or "").strip()
            from pathlib import Path as _Path

            export_usable = bool(export) and (
                export.startswith(("http://", "https://")) or _Path(export).exists()
            )
            if export_usable:
                from .sources import read_chat_export_transcript

                transcript = read_chat_export_transcript(export, policy)
                if transcript.turns and any(turn.role == "user" for turn in transcript.turns):
                    sub.conversation = conversation_from_transcript(transcript)
                    report.hydrated += 1
                    report.turns += len(sub.conversation)
                    continue
                reason = transcript.error or "chat export did not parse into conversation turns"
                report.failures.append(f"{label}: {reason}")
                sub.fetch_failed = True
                continue
            report.failures.append(f"{label}: no final trajectory link to fetch")
            continue

        transcript = transcripts.get(sub.final_link)
        if transcript is None:
            report.failures.append(f"{label}: link was never fetched")
            # There was a link and we came back with nothing. Recorded on the
            # submission so a report can say "we could not look" instead of
            # falling silent; deliberately not read by any check, least of all
            # 90, which would turn our transport errors into the contributor's
            # invalid link. See `ModelSubmission.fetch_failed`.
            sub.fetch_failed = True
            continue
        if transcript.dead_page:
            report.dead_pages.append(f"{label}: {sub.final_link}")
            sub.share_page_dead = True
            continue
        if not transcript.turns:
            reason = transcript.error or "no turns parsed from the rendered page"
            report.failures.append(f"{label}: {reason}")
            sub.fetch_failed = True
            continue

        # Every real conversation opens with a user prompt. Without one the
        # renderer served something else -- a marketing page, an interstitial --
        # and feeding that to the rating stage would have it judge a model on
        # text the model never produced.
        if not any(turn.role == "user" for turn in transcript.turns):
            report.failures.append(
                f"{label}: rendered page has no user turns, so it is not the "
                f"shared conversation ({sub.final_link})"
            )
            sub.fetch_failed = True
            continue

        sub.conversation = conversation_from_transcript(transcript)
        report.hydrated += 1
        report.turns += len(sub.conversation)

    # Second pass, because a submission's depth is only meaningful against the
    # other one's and the first pass has not filled that in yet while it runs.
    for task, sub in pairs:
        depth = measure_conversation_depth(task, sub, policy)
        sub.corroborated_exchanges = depth.corroborated
        if depth.materially_short and sub.conversation:
            report.short_captures.append(
                f"{task.task_id}/{sub.model}: {depth.describe()}"
            )

    return report
