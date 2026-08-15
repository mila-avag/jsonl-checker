"""Live run of the blind rating pass against a real share link.

    python3 -m honeybee_qc.demo_rating_stage --link https://gemini.google.com/share/...

The question this answers is not "does the code run" but "how much of a real task
can actually be audited from a transcript". The rubric below is planted: half its
criteria are about things the transcript records (what was asked, what the model
said it would do, how many turns it took) and half are about the video itself
(resolution, colour consistency, visual pacing). If the abstention model works,
the second half abstains and the first half does not.

Contributor ratings are all 1, claiming every requirement was met, so any
criterion the auditor can judge and fails becomes a visible disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config import DEFAULT_POLICY
from .context import conversation_from_transcript, profile_evidence
from .llm import DEFAULT_EFFORT, DEFAULT_MODEL, build_client
from .models import (
    CriterionRating,
    DimensionRating,
    KeyTurn,
    ModelSubmission,
    RubricCriterion,
    Sxs,
    Task,
)
from .rating_stages import estimate_rating_calls, run_rating_stage
from .sources import fetch_link_transcript
from .taxonomies import RATING_DIMENSIONS

# (id, text, what the auditor should be able to do with it)
PLANTED_RUBRIC: list[tuple[str, str, str]] = [
    ("C1", "The model acknowledges the user's request and states what it will produce "
           "before producing it.", "text"),
    ("C2", "The model asks no more than one clarifying question before starting work.",
     "text"),
    ("C3", "The model does not claim to have performed an action it did not perform.",
     "text"),
    ("C4", "The user does not have to repeat any requirement more than once.", "text"),
    ("C5", "The model delivers a video file rather than describing how to make one.",
     "text"),
    ("C6", "The final video is rendered at 4K resolution.", "artifact"),
    ("C7", "Colour coding of the three axes is consistent across every shot of the "
           "video.", "artifact"),
    ("C8", "The video's pacing leaves each label on screen long enough to read.",
     "artifact"),
    ("C9", "The animation shows precession in the correct direction relative to the "
           "applied torque.", "artifact"),
    ("C10", "The video is between 10 and 12 seconds long.", "artifact"),
]


def build_task(link: str, transcript, task_id: str = "live-rating-1") -> Task:
    sub = ModelSubmission(
        model="A",
        final_link=link,
        transcript_pdf="",
        attachments=["gyroscope_animation.mp4"],
        conversation=conversation_from_transcript(transcript),
    )
    rubric = [
        RubricCriterion(criterion_id=cid, text=text, l1_label="Outcome Quality", weight=3)
        for cid, text, _ in PLANTED_RUBRIC
    ]
    return Task(
        task_id=task_id,
        seeded_prompt="Make a short animation explaining how a gyroscope works.",
        user_goal="Produce a classroom-ready teaching aid explaining gyroscopic "
        "precession to high-school students.",
        target_deliverables=["a 10-12 second clip", "4K resolution", "consistent axis colours"],
        model_a=sub,
        key_turn=KeyTurn(turn_index=1, justification=""),
        rubric=rubric,
        # The contributor claims every requirement was satisfied.
        criterion_ratings=[
            CriterionRating(criterion_id=c.criterion_id, model="A", score=1) for c in rubric
        ],
        dimension_ratings=[
            DimensionRating(model="A", dimension=d, rating=9, justification="Looked good.")
            for d in RATING_DIMENSIONS
        ],
        sxs=Sxs(likert=None, justification=""),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="live blind rating pass")
    ap.add_argument("--link", default="https://gemini.google.com/share/d64e0a300ec0")
    ap.add_argument("--snapshot-dir", default=".honeybee_snaps")
    ap.add_argument("--cache-db", default=".honeybee_cache/rating_demo.db")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    policy = replace(DEFAULT_POLICY, snapshot_dir=args.snapshot_dir)

    print(f"fetching {args.link}", file=sys.stderr)
    transcript = fetch_link_transcript(args.link, policy, args.snapshot_dir)
    if transcript.error:
        print(f"fetch failed: {transcript.error}", file=sys.stderr)
        return 1
    print(f"  {len(transcript.turns)} turns from {transcript.snapshot_path}", file=sys.stderr)

    task = build_task(args.link, transcript)
    profile = profile_evidence(task.model_a)
    print(f"  evidence: {json.dumps(profile.to_dict())}", file=sys.stderr)
    print(f"  will make {estimate_rating_calls(task)} model calls", file=sys.stderr)

    client, cache = build_client(
        model=args.model,
        effort=args.effort,
        cache_path=args.cache_db,
        policy_version="rating-demo-1",
    )
    result = run_rating_stage(task, client, policy, workers=args.workers)
    cache.close()

    kind_by_id = {cid: kind for cid, _, kind in PLANTED_RUBRIC}
    judged = {f.criterion_id: f for f in result.criterion_findings}
    abstained = {a.item_id.split("::")[1] for a in result.abstentions if a.check_id == 270}

    print("\n=== per criterion (Model A) ===")
    for cid, text, kind in PLANTED_RUBRIC:
        if cid in abstained:
            outcome = "ABSTAINED"
        elif cid in judged:
            f = judged[cid]
            outcome = f"auditor={'pass' if f.auditor_score else 'FAIL'}"
            if f.disagrees:
                outcome += "  <-- disagrees with contributor"
        else:
            outcome = "no result"
        print(f"  {cid:<4} [{kind:<8}] {outcome}")
        print(f"       {text[:88]}")

    artifact_ids = {c for c, k in kind_by_id.items() if k == "artifact"}
    text_ids = {c for c, k in kind_by_id.items() if k == "text"}
    print("\n=== abstention by criterion kind ===")
    print(f"  artifact-dependent: {len(abstained & artifact_ids)} of {len(artifact_ids)} abstained")
    print(f"  text-auditable:     {len(abstained & text_ids)} of {len(text_ids)} abstained")

    print("\n=== verdicts ===")
    for v in result.verdicts:
        print(f"  check {v.check_id}: {v.band} (score={v.score}) {v.measurement.notes[:120]}")
        print(f"    {json.dumps(v.measurement.counts)}")

    print(
        f"\ncalls={result.calls} cached={result.cached_calls} "
        f"cost=${result.cost_usd:.4f} errors={len(result.errors)}"
    )
    for e in result.errors[:5]:
        print(f"  error: {e}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "summary": result.to_dict(),
                    "verdicts": [v.to_dict() for v in result.verdicts],
                },
                fh,
                indent=2,
                default=str,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
