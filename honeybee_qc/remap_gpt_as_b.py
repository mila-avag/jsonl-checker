"""Promote GPT-on-C to slot B for an A-vs-GPT eval.

Aspirational files A vs B and A vs C. On a coin-flip batch, B is often Claude
and GPT sits on C. An `--audit-slots A,B` run then grades Claude and uses the
AB ranking writeup, which is A vs Claude -- not A vs GPT.

This remap is a post-ingest overlay, not a project-profile change. For listed
task IDs it:

  * moves the C submission onto B and relabels `submission.model` to ``B``
  * rematches criterion / dimension ratings C → B (drops the Claude-as-B ones)
  * keeps only the AC SxS pair and rewrites it to ``left=A, right=B`` so
    checks 400/450/470 judge A vs GPT against the AC likert and writeup
  * drops Claude (old B), the now-empty C, and the simplified trajectory so
    hydration does not fetch them

Ranking paste later belongs on the AC selector
``step-QuantitativeModelResponseSelector-ec730e661d1a``. The Sxs object does
not store a step id.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .models import Task

AC_SELECTOR_STEP = "step-QuantitativeModelResponseSelector-ec730e661d1a"


def load_task_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            ids.add(text.split()[0])
    return ids


def relabel_gpt_c_as_b(task: Task) -> None:
    """In-place: treat this task's C trajectory (GPT) as B, keep the AC pair."""
    gpt = task.model_c
    if gpt is None:
        raise ValueError(f"{task.task_id}: no C submission to promote to B")
    ac = next(
        (c for c in task.comparison_list() if c.left == "A" and c.right == "C"),
        None,
    )
    if ac is None:
        raise ValueError(
            f"{task.task_id}: no A-vs-C comparison; cannot keep the GPT ranking pair"
        )

    gpt.model = "B"
    task.model_b = gpt
    task.model_c = None
    task.simplified = None

    task.criterion_ratings = [
        replace(r, model="B") if r.model == "C" else r
        for r in task.criterion_ratings
        if r.model not in ("B", "S")
    ]
    task.dimension_ratings = [
        replace(r, model="B") if r.model == "C" else r
        for r in task.dimension_ratings
        if r.model not in ("B", "S")
    ]

    ac.right = "B"
    task.comparisons = [ac]
    task.sxs = ac
    task.expected_slots = tuple(s for s in task.expected_slots if s in ("A", "B")) or (
        "A",
        "B",
    )


def apply_relabel_c_to_b(tasks: list[Task], ids: set[str]) -> list[str]:
    """Relabel every task whose id is in `ids`. Returns the ids that were hit."""
    hit: list[str] = []
    for task in tasks:
        if task.task_id not in ids:
            continue
        relabel_gpt_c_as_b(task)
        hit.append(task.task_id)
    return hit


def narrow_to_ab_only(task: Task) -> None:
    """Keep A-vs-B (GPT) only: drop C/D/S so they are not hydrated or ranked.

    Used on the whole A-vs-GPT batch after the 28 are relabeled, so the 26
    that already had GPT on B do not still fetch Claude-on-C or spend 400/450
    on the AC pair. Not a default `--audit-slots` behaviour -- D-only still
    needs DA/DB/DC in front of the judge.
    """
    pairs = [c for c in task.comparison_list() if c.left == "A" and c.right == "B"]
    if pairs:
        task.comparisons = pairs
        task.sxs = pairs[0]
    task.model_c = None
    task.model_d = None
    task.simplified = None
    task.expected_slots = ("A", "B")
    task.criterion_ratings = [r for r in task.criterion_ratings if r.model in ("A", "B")]
    task.dimension_ratings = [r for r in task.dimension_ratings if r.model in ("A", "B")]
