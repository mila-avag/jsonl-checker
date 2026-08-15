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
from .models import Task
from .transcripts import Transcript


@dataclass
class HydrationReport:
    submissions: int = 0
    hydrated: int = 0
    turns: int = 0
    dead_pages: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "submissions": self.submissions,
            "hydrated": self.hydrated,
            "turns": self.turns,
            "dead_pages": self.dead_pages,
            "failures": self.failures,
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

    pairs = [(task, sub) for task in tasks for sub in (task.model_a, task.model_b)]
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
            report.failures.append(f"{label}: no final trajectory link to fetch")
            continue

        transcript = transcripts.get(sub.final_link)
        if transcript is None:
            report.failures.append(f"{label}: link was never fetched")
            continue
        if transcript.dead_page:
            report.dead_pages.append(f"{label}: {sub.final_link}")
            continue
        if not transcript.turns:
            reason = transcript.error or "no turns parsed from the rendered page"
            report.failures.append(f"{label}: {reason}")
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
            continue

        sub.conversation = conversation_from_transcript(transcript)
        report.hydrated += 1
        report.turns += len(sub.conversation)

    return report
