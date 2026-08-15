"""Merge N per-chunk `cli.py` report JSONs (from a single batch split into
resilience chunks against a shared cache.db/snapshots dir) into one combined
report JSON with the exact same top-level schema `cli.py` itself writes.

`build_manual_audit_viewer.py` already has a `merge_reports()` used to feed
the viewer, but it only combines the four fields the viewer actually reads
(`tasks`, `rubric_stage`, `rating_stage`, `informed_stage`, `cost_usd`, and a
partial `hydration`). This script produces a complete, standalone combined
report -- every top-level key `cli.py` writes, including `batch` (recomputed
by summing each chunk's already-aggregated counts, not by re-deriving verdicts
from scratch), `preflight`, `provenance`, `duplicates`, `review_queue`,
`stage_errors`, and the full `cache`/`hydration` blocks -- so nothing from any
chunk is silently dropped when chunks are combined for archival or downstream
tooling.

Chunks are assumed disjoint (each task_id assigned to exactly one chunk, as
`cli.py` chunking scripts in this repo always do). Per-task-keyed lists
(`tasks`, `preflight`) are deduplicated by `task_id` on the rare chance a task
appears in more than one chunk's report (first occurrence wins); everything
else is concatenated or summed since there is nothing to key it on.

Usage:
    python3 -m honeybee_qc.merge_chunk_reports \
        --report audit_runs/.../chunks/chunk_00_report.json \
        --report audit_runs/.../chunks/chunk_01_report.json \
        ... \
        --out audit_runs/.../combined_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dedupe_by_task_id(items: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for it in items:
        tid = it.get("task_id")
        if tid is not None:
            if tid in seen:
                continue
            seen.add(tid)
        out.append(it)
    return out


def _merge_check_rates(chunks: list[dict]) -> list[dict]:
    """Check rates are themselves counts (fails/non_fails/clean/not_evaluated
    over a denominator), so they combine by summing the raw counts per
    check_id and re-deriving the rates -- not by averaging the per-chunk
    rates, which would silently mis-weight chunks of different sizes."""
    by_check: dict[int, dict] = {}
    for b in chunks:
        for cr in b.get("check_rates") or []:
            cid = cr["check_id"]
            agg = by_check.setdefault(
                cid,
                {
                    "check_id": cid,
                    "dimension": cr["dimension"],
                    "sub_dimension": cr["sub_dimension"],
                    "denominator": 0,
                    "fails": 0,
                    "non_fails": 0,
                    "clean": 0,
                    "not_evaluated": 0,
                },
            )
            agg["denominator"] += cr["denominator"]
            agg["fails"] += cr["fails"]
            agg["non_fails"] += cr["non_fails"]
            agg["clean"] += cr["clean"]
            agg["not_evaluated"] += cr["not_evaluated"]
    out = []
    for cid in sorted(by_check):
        agg = by_check[cid]
        denom = agg["denominator"]
        agg["fail_rate"] = round(agg["fails"] / denom, 4) if denom else 0.0
        agg["non_fail_rate"] = round(agg["non_fails"] / denom, 4) if denom else 0.0
        out.append(agg)
    return out


def _merge_batch(chunks: list[dict]) -> dict:
    task_counts: dict[str, int] = {}
    error_code_distribution: dict[str, int] = {}
    gate_rate_distribution: dict[str, list[float]] = {}
    gate_rate_by_task: dict[str, dict[str, float]] = {}
    policies = []

    for b in chunks:
        for k, v in (b.get("task_counts") or {}).items():
            task_counts[k] = task_counts.get(k, 0) + v
        for k, v in (b.get("error_code_distribution") or {}).items():
            error_code_distribution[k] = error_code_distribution.get(k, 0) + v
        for k, vals in (b.get("gate_rate_distribution") or {}).items():
            gate_rate_distribution.setdefault(k, []).extend(vals)
        for k, by_task in (b.get("gate_rate_by_task") or {}).items():
            gate_rate_by_task.setdefault(k, {}).update(by_task)
        if b.get("policy") is not None:
            policies.append(b["policy"])

    distinct_policies = [p for i, p in enumerate(policies) if p not in policies[:i]]
    if len(distinct_policies) > 1:
        print(
            f"warning: chunks were run under {len(distinct_policies)} different "
            "policy configs; using the first chunk's policy in the combined report",
        )

    return {
        "task_counts": task_counts,
        "check_rates": _merge_check_rates(chunks),
        "error_code_distribution": error_code_distribution,
        "gate_rate_distribution": gate_rate_distribution,
        "gate_rate_by_task": gate_rate_by_task,
        "policy": policies[0] if policies else {},
    }


def merge_chunk_reports(report_paths: list[Path]) -> dict:
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]

    seen_tasks: set[str] = set()
    seen_preflight: set[str] = set()
    merged = {
        "batch": _merge_batch([p.get("batch") or {} for p in payloads]),
        "stage_errors": [],
        "preflight": [],
        "provenance": [],
        "duplicates": [],
        "rubric_stage": [],
        "rating_stage": [],
        "informed_stage": [],
        "cost_usd": 0.0,
        "review_queue": [],
        "tasks": [],
        "hydration": {
            "submissions": 0, "hydrated": 0, "turns": 0,
            "dead_pages": [], "failures": [],
        },
        "cache": {"enabled": True, "hits": 0, "misses": 0},
    }

    for p in payloads:
        merged["tasks"].extend(_dedupe_by_task_id(p.get("tasks") or [], seen_tasks))
        merged["preflight"].extend(_dedupe_by_task_id(p.get("preflight") or [], seen_preflight))
        for key in ("stage_errors", "provenance", "duplicates", "rubric_stage",
                    "rating_stage", "informed_stage", "review_queue"):
            merged[key].extend(p.get(key) or [])
        merged["cost_usd"] += p.get("cost_usd", 0) or 0

        h = p.get("hydration") or {}
        merged["hydration"]["submissions"] += h.get("submissions") or 0
        merged["hydration"]["hydrated"] += h.get("hydrated") or 0
        merged["hydration"]["turns"] += h.get("turns") or 0
        merged["hydration"]["dead_pages"].extend(h.get("dead_pages") or [])
        merged["hydration"]["failures"].extend(h.get("failures") or [])

        c = p.get("cache") or {}
        merged["cache"]["enabled"] = merged["cache"]["enabled"] and bool(c.get("enabled", True))
        merged["cache"]["hits"] += c.get("hits") or 0
        merged["cache"]["misses"] += c.get("misses") or 0

    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="append", required=True, dest="reports", help="repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    report_paths = [Path(p) for p in args.reports]
    merged = merge_chunk_reports(report_paths)

    n_in = 0
    for p in report_paths:
        n_in += len(json.loads(p.read_text(encoding="utf-8")).get("tasks") or [])
    n_out = len(merged["tasks"])
    if n_out != n_in:
        print(f"warning: {n_in} task entries in, {n_out} after de-dup by task_id")

    Path(args.out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {n_out} tasks, ${merged['cost_usd']:.2f} total spend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
