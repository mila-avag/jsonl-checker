"""Replay a deterministic check against archived snapshots, offline and free.

Some checks reach a band without a model call. Those can be re-scored against a
manual audit's ground truth for nothing, from snapshots already on disk, which
matters twice over: a full LLM re-run costs on the order of $690, and a narrowed
re-run carries only some of the audited tasks -- one of ten, for the check-85
audit this was first written for. A replay scores all of them.

Both directions are scored, for the same reason the report-to-report comparison
scores both: a false fail that stopped failing is a fix, a true fail that stopped
failing is a regression, and a tool that could only report the first would be
measuring whether the check got quieter.

SAFETY. This reads snapshots and a CSV, and writes nothing at all -- not a
report, not a cache, nothing inside any run directory. A share page missing from
the snapshot directory is reported unreplayable rather than fetched, so there is
no path from here to the network. Both properties are what make it safe to run
against a run that is still in flight.

WHAT A REPLAY CAN AND CANNOT SEE. Every check here is deterministic, but three
are only partly so, and the parts that need a model call or an auditor's own
judgment cannot be replayed. Those are reported unreplayable rather than scored,
because a half-run check that abstains is not the same fact as a check that
looked and found nothing:

    85   the file half only -- prompt filenames against the manifest and the
         names the conversation established. The entity half is informed.
    90   structural link validity, plus any share page the archived DOM itself
         reports as gone. A link that merely timed out live is invisible here.
    96   in full: minimum turns, from the hydrated snapshots.
    100  the turn-1 structural leg only. Any other key turn needs the auditor's
         independent pick, so it abstains.
    460  in full: ranking-vs-dimension inversion, read off the ratings.

Note that `not_evaluated` means something different here than it does in
`score_against_manual_audit`. There it is an observed outcome of the re-run and
counts as no longer failing; here it almost always means the replay lacked an
input, so it is never scored in either column.

Usage:
    python3 -m honeybee_qc.replay_deterministic --check 85 \\
        --gt honeybee_qc/audit_runs/revalidate_20260815/gt.json \\
        --rows honeybee_qc/audit_runs/revalidate_20260815/revalidate.csv \\
        --snapshots honeybee_qc/audit_runs/honeybee_l1_full_20260814_batch9/snapshots
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_POLICY, Policy
from .env_context import FileReferenceScan, scan_file_references
from .gates import (
    build_inversion_finding,
    evaluate_check_85,
    evaluate_check_96,
    evaluate_check_100,
    evaluate_check_460,
)
from .hydrate import hydrate_conversations
from .ingest import ingest_rows
from .links import evaluate_check_90
from .models import Task
from .score_against_manual_audit import MARKS as REPORT_MARKS
from .score_against_manual_audit import VERDICTS, classify
from .sources import classify_link, parse_share_html
from .transcripts import Transcript

MARKS = {**REPORT_MARKS, "unreplayable": "unreplayable"}

NO_SNAPSHOT = "no snapshot on disk (this replay never fetches)"


@dataclass
class Replay:
    """One check replayed on one task: the band, and why it may not count."""

    band: str
    blocked: str = ""
    note: str = ""


def snapshot_name(provider: str, url: str) -> str:
    """The archive filename the pipeline writes for a share URL."""
    return f"{provider}_{hashlib.sha256(url.encode()).hexdigest()[:16]}.html"


class SnapshotOnlyFetcher:
    """A `hydrate_conversations` fetch that can only ever read the archive.

    There is no fallback branch: a URL with no snapshot returns a transcript
    carrying an error, so an incomplete archive shows up as tasks that could not
    be replayed instead of quietly becoming a fetch.
    """

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = Path(snapshot_dir)
        self.seen: dict[str, Transcript] = {}

    def __call__(self, url: str) -> Transcript:
        link = classify_link(url)
        transcript = Transcript(kind="link", source_id=link.share_id, provider=link.provider)
        if not link.valid:
            transcript.error = f"link is not a valid share URL: {link.reason}"
        else:
            path = self.snapshot_dir / snapshot_name(link.provider, url)
            if not path.exists():
                transcript.error = NO_SNAPSHOT
            else:
                turns, full_text, dead = parse_share_html(
                    path.read_text(encoding="utf-8", errors="replace"), link.provider
                )
                transcript.snapshot_path = str(path)
                transcript.turns = turns
                transcript.full_text = full_text
                transcript.dead_page = dead
        self.seen[url] = transcript
        return transcript

    def dead_links(self, task: Task) -> list[str]:
        """Check 90's dead-link reasons, for pages the archive shows as gone."""
        out: list[str] = []
        for sub in (task.model_a, task.model_b):
            if sub is None or not sub.final_link:
                continue
            transcript = self.seen.get(sub.final_link)
            if transcript is not None and transcript.dead_page:
                out.append(f"final_link_{sub.model}:share page unavailable")
        return out


def _counts_note(verdict) -> str:
    counts = getattr(getattr(verdict, "measurement", None), "counts", None) or {}
    bits = [
        f"{key}={value}"
        for key, value in list(counts.items())[:4]
        if not isinstance(value, (list, dict))
    ]
    return ", ".join(bits)


def _note_85(scan: FileReferenceScan, band: str) -> str:
    """What the scan actually concluded, short enough to read in a table."""
    if band == "fail":
        return "still flagging: " + ", ".join(f.reference for f in scan.findings[:4])
    established = len(scan.manifest.conversation_mentioned)
    return f"refs={len(scan.named_references)}, established_by_conversation={established}"


def _replay_85(task: Task, policy: Policy, fetcher: SnapshotOnlyFetcher) -> Replay:
    scan = scan_file_references(task)
    verdict = evaluate_check_85(
        task, scan.findings, scan, entity_half_ran=False, policy=policy
    )
    if not scan.ran:
        # The band here is "unverified", not "checked and clean", so it must not
        # be read as a fix.
        return Replay(verdict.band, blocked=f"file half did not run: {scan.blocked_reason}")
    return Replay(verdict.band, note=_note_85(scan, verdict.band))


def _replay_90(task: Task, policy: Policy, fetcher: SnapshotOnlyFetcher) -> Replay:
    verdict = evaluate_check_90(task, policy, dead_links=fetcher.dead_links(task))
    return Replay(verdict.band, note=_counts_note(verdict))


def _replay_96(task: Task, policy: Policy, fetcher: SnapshotOnlyFetcher) -> Replay:
    verdict = evaluate_check_96(task, policy)
    return Replay(verdict.band, note=_counts_note(verdict))


def _replay_100(task: Task, policy: Policy, fetcher: SnapshotOnlyFetcher) -> Replay:
    verdict = evaluate_check_100(task, auditor_key_turn=None, policy=policy)
    return Replay(verdict.band, note=_counts_note(verdict))


def _replay_460(task: Task, policy: Policy, fetcher: SnapshotOnlyFetcher) -> Replay:
    verdict = evaluate_check_460(task, build_inversion_finding(task, policy=policy), policy)
    return Replay(verdict.band, note=_counts_note(verdict))


REPLAYS = {85: _replay_85, 90: _replay_90, 96: _replay_96, 100: _replay_100, 460: _replay_460}


def outcome_for(gt_verdict: str, replay: Replay) -> str:
    """How this replay scores against what the audit said, if it scores at all.

    Ground-truth membership is itself the assertion that the check was failing
    when the audit read it, which is why the before band handed to `classify` is
    always "fail": there is no before report here to look one up in.
    """
    if replay.blocked or replay.band == "not_evaluated":
        return "unreplayable"
    return classify(gt_verdict, "fail", replay.band)


def replay_tasks(
    tasks: list[Task],
    wanted: dict[str, str],
    check_id: int,
    fetcher: SnapshotOnlyFetcher,
    policy: Policy = DEFAULT_POLICY,
) -> list[tuple[Task, str, Replay, str]]:
    """(task, audit verdict, replay, outcome) for every task, audit order first."""
    replay_check = REPLAYS[check_id]
    rows = []
    for task in sorted(tasks, key=lambda t: VERDICTS.index(wanted[t.task_id])):
        gt_verdict = wanted[task.task_id]
        replay = replay_check(task, policy, fetcher)
        rows.append((task, gt_verdict, replay, outcome_for(gt_verdict, replay)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", required=True, type=int, choices=sorted(REPLAYS),
        help="the check to replay; only deterministic checks can be",
    )
    ap.add_argument("--gt", required=True, help="the manual audit's ground-truth JSON")
    ap.add_argument("--rows", required=True, help="the taskattempts CSV holding those tasks")
    ap.add_argument("--snapshots", required=True, help="the archived share pages to replay from")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    check_id = str(args.check)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    if check_id not in gt:
        print(f"check {check_id} is not in {args.gt}", file=sys.stderr)
        return 1
    wanted = {
        task_id: verdict
        for verdict in VERDICTS
        for task_id in gt[check_id].get(verdict, [])
    }

    snapshots = Path(args.snapshots)
    if not snapshots.is_dir():
        print(f"snapshot dir not found: {snapshots}", file=sys.stderr)
        return 1

    csv.field_size_limit(sys.maxsize)
    with open(args.rows, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("TASK") or "") in wanted]
    tasks, _ = ingest_rows(rows)

    fetcher = SnapshotOnlyFetcher(snapshots)
    hydrate_conversations(tasks, DEFAULT_POLICY, fetch=fetcher, workers=args.workers)
    replayed = replay_tasks(tasks, wanted, args.check, fetcher)

    print("=" * 78)
    print(f"CHECK {args.check} -- replayed from snapshots against the manual audit")
    print(f"{len(tasks)}/{len(wanted)} audited tasks ingested from {Path(args.rows).name}")
    print("=" * 78)

    for task, gt_verdict, replay, outcome in replayed:
        hydrated = sum(
            1 for s in (task.model_a, task.model_b) if s is not None and s.conversation
        )
        note = replay.blocked or replay.note
        print(
            f"  {task.task_id[-6:]}  {gt_verdict:10}  {replay.band:<13} "
            f"{MARKS[outcome]:18} hydrated={hydrated}/2  {note}"
        )

    totals = Counter(outcome for _, _, _, outcome in replayed)
    print("\n" + "=" * 78)
    print(
        f"false fails fixed:      {totals['fixed']}\n"
        f"false fails remaining:  {totals['still_failing']}\n"
        f"true fails held:        {totals['held']}\n"
        f"true fails REGRESSED:   {totals['regressed']}\n"
        f"unreplayable:           {totals['unreplayable']}"
    )

    absent = sorted(set(wanted) - {t.task_id for t in tasks})
    if absent:
        print(f"not in --rows:          {len(absent)}")

    regressed = [t.task_id for t, _, _, o in replayed if o == "regressed"]
    if regressed:
        print("\nREGRESSIONS -- a real defect this build stopped catching:")
        for task_id in regressed:
            print(f"  {task_id}")
        return 0

    unguarded = [t.task_id for t, v, _, o in replayed if v == "TRUE_FAIL" and o == "unreplayable"]
    unguarded += [t for t in absent if wanted[t] == "TRUE_FAIL"]
    if not any(v == "TRUE_FAIL" for v in wanted.values()):
        print(
            f"\nNo regression guard: the audit listed no true fails for check "
            f"{args.check}, so this replay can only confirm fixes. Read the count "
            "above as 'nothing to regress', not as 'nothing regressed'."
        )
    elif unguarded:
        print(
            f"\nNo regressions among the {totals['held']} true fail(s) replayed -- but "
            f"{len(unguarded)} could not be, so the guard is incomplete:"
        )
        for task_id in unguarded:
            print(f"  {task_id}")
    else:
        print("\nNo regressions: every fail the audit called real is still failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
