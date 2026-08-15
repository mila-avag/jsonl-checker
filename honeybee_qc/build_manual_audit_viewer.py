"""A manual-audit HTML viewer for a set of tasks: every check's verdict and
justification *and* the exact model call(s) behind it -- prompt, system
prompt, and raw model answer, pulled straight from the response cache.

Built for the 30 batch7 tasks whose rubric is empty by design for this batch.
`cli.py` runs the rating and informed stages on them same as any other task,
so 300/400/450/470/70-110 carry real verdicts even though 200/230/240/
250/260/270 stay `not_evaluated` -- and check 1000 itself is `not_evaluated`
too, not a fail: an empty rubric alone is expected here and does not make a
task unauditable (see `rubric_empty_check_1000_verdict` in `cli.py`). This is
the tool for eyeballing whether those real verdicts are actually trustworthy
-- for each one, it regenerates
the same prompt the real run built (same builder functions, same task data)
and looks up the model's actual answer by its cache `request_key`, so what you
read here is what was actually sent and actually came back, not a
reconstruction.

Usage:
    python3 -m honeybee_qc.build_manual_audit_viewer \
        --report audit_runs/.../chunks/chunk_00_report.json \
        --report audit_runs/.../chunks/chunk_remaining_report.json \
        --tasks-csv audit_runs/.../pulled_30.csv \
        --cache-db audit_runs/honeybee_l1_full_20260814_batch7/cache.db \
        --out audit_runs/.../manual_audit.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

from .build_canvas import (
    _abstention,
    _attributed_evidence,
    _check_name,
    _evidence,
    _informed_evidence,
    _justification,
    _set_aside,
    _spec_label,
)
from .config import DEFAULT_POLICY, Policy
from .hydrate import hydrate_conversations
from .informed_stages import build_informed_requests
from .ingest import load_taskattempts_csv
from .models import Task
from .rating_stages import build_rating_requests

BAND_CLASS = {"fail": "bad", "non_fail": "warn", "clean": "good", "not_evaluated": "muted"}
BAND_LABEL = {
    "fail": "Fail",
    "non_fail": "Non-fail",
    "clean": "Clean",
    "not_evaluated": "Not evaluated",
}


def _e(text: object) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _pre(text: object) -> str:
    return f"<pre>{_e(text)}</pre>" if str(text or "").strip() else "<p class='muted'>&mdash;</p>"


# ---------------------------------------------------------------------------
# Merge chunked reports
# ---------------------------------------------------------------------------


def merge_reports(report_paths: list[Path]) -> dict:
    merged: dict = {
        "tasks": [], "rubric_stage": [], "rating_stage": [], "informed_stage": [],
        "cost_usd": 0.0, "hydration": {},
    }
    seen: set[str] = set()
    for p in report_paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        for t in payload.get("tasks") or []:
            if t["task_id"] in seen:
                continue
            seen.add(t["task_id"])
            merged["tasks"].append(t)
        for key in ("rubric_stage", "rating_stage", "informed_stage"):
            merged[key].extend(payload.get(key) or [])
        merged["cost_usd"] += payload.get("cost_usd", 0) or 0
        h = payload.get("hydration") or {}
        for k in ("hydrated", "submissions", "turns"):
            merged["hydration"][k] = merged["hydration"].get(k, 0) + (h.get(k) or 0)
    return merged


# ---------------------------------------------------------------------------
# Cache lookup + prompt regeneration
# ---------------------------------------------------------------------------


def _cache_rows(conn: sqlite3.Connection, request_key: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload, cost_usd, created_at FROM response_cache "
        "WHERE request_key = ? ORDER BY created_at",
        (request_key,),
    ).fetchall()
    out = []
    for payload, cost_usd, created_at in rows:
        try:
            data = json.loads(payload)
        except Exception:
            data = {"_unparsed": payload}
        out.append({"data": data, "cost_usd": cost_usd, "created_at": created_at})
    return out


def _calls_by_check(task: Task, policy: Policy, conn: sqlite3.Connection | None) -> dict[int, list[dict]]:
    """Every model call the real run would have made for this task, grouped by
    the check it feeds, with its cached answer attached if one is on file.

    Rebuilt from the same functions `cli.py` calls, on the same `Task` object
    ingested from the same CSV -- so the prompt text here is byte-identical to
    what was actually sent, not an approximation of it. `guard=False` on the
    rating builder only skips the blindness assertion (which raises rather than
    warns); it does not change what prompt gets built, so a task that tripped
    it live still renders here with a note, instead of vanishing from the
    viewer entirely.
    """
    requests = []
    errors: list[str] = []
    try:
        requests += build_informed_requests(task, policy)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"informed request build failed: {exc}")
    try:
        requests += build_rating_requests(task, policy, guard=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rating request build failed: {exc}")

    out: dict[int, list[dict]] = {}
    for req in requests:
        cid = req.metadata.get("check_id")
        if cid is None:
            continue
        out.setdefault(cid, []).append(
            {
                "key": req.key,
                "prompt": req.prompt,
                "system": req.system,
                "metadata": {k: v for k, v in req.metadata.items() if k != "check_id"},
                "responses": _cache_rows(conn, req.key) if conn is not None else [],
            }
        )
    if errors:
        out.setdefault(-1, []).append({"key": "", "prompt": "", "system": "", "metadata": {"errors": errors}, "responses": []})
    return out


# ---------------------------------------------------------------------------
# Per-task rendering
# ---------------------------------------------------------------------------


def _call_block_html(call: dict, idx: int, task_suffix: str, check_id: int) -> str:
    meta = call["metadata"]
    label = ", ".join(f"{k}={v}" for k, v in meta.items()) or call["key"].rsplit("::", 1)[-1]
    responses = call["responses"]
    if not responses:
        resp_html = "<p class='muted'>No cached answer on file for this exact call.</p>"
    else:
        blocks = []
        for i, r in enumerate(responses):
            draw = f" &mdash; draw {i + 1}/{len(responses)}" if len(responses) > 1 else ""
            cost_str = f"{r['cost_usd']:.4f}"
            blocks.append(
                f"<p class='meta-line'>Model's answer{draw} <span class='muted'>"
                f"({_e(r['created_at'])}, ${_e(cost_str)})</span></p>"
                + _pre(json.dumps(r["data"], indent=2))
            )
        resp_html = "".join(blocks)
    uid = f"call-{_e(check_id)}-{_e(task_suffix)}-{idx}"
    return f"""<details class="call">
  <summary>{_e(label)}</summary>
  <div class="call-body">
    <p class="meta-line">Request key <span class="mono">{_e(call['key'])}</span></p>
    {f"<p class='meta-line'>System prompt</p>{_pre(call['system'])}" if call['system'] else ''}
    <p class="meta-line">Prompt sent to the model (regenerated from the same task data, byte-identical to the live run)</p>
    {_pre(call['prompt'])}
    {resp_html}
  </div>
</details>"""


def _check_card_html(
    check: dict, calls: list[dict], findings: dict, informed: dict | None, task_suffix: str
) -> str:
    cid = check["check_id"]
    band = check["band"]
    name = _check_name(cid)
    code = check.get("error_code") or _spec_label(cid, band)
    why = _justification(check, findings, None)
    evidence = (
        _evidence(check, findings, {})
        or _attributed_evidence(check, {})
        or _informed_evidence(check, informed, {})
    )
    set_aside = _set_aside(check)
    contributing = check.get("contributing_items") or []

    ev_html = ""
    if evidence:
        items = "".join(
            f"<li><b>{_e(it['criterion'])}:</b> {_e(it['detail'])}"
            + (f" <span class='quote'>&ldquo;{_e(it['quote'])}&rdquo;</span>" if it.get("quote") else "")
            + "</li>"
            for it in evidence
        )
        ev_html = f"<ul class='evidence-list'>{items}</ul>"

    extras = []
    if contributing and not evidence:
        extras.append(f"<p class='meta-line'>Items: {_e(', '.join(str(c) for c in contributing))}</p>")
    if set_aside:
        extras.append(f"<p class='meta-line muted'>Set aside: {_e('; '.join(set_aside))}</p>")

    measurement = check.get("measurement") or {}
    meas_html = _pre(json.dumps(measurement, indent=2)) if measurement else ""

    calls_html = ""
    if calls:
        blocks = "".join(_call_block_html(c, i, task_suffix, cid) for i, c in enumerate(calls))
        calls_html = (
            f"<details class='calls-group'><summary>Model call(s) behind this check ({len(calls)})</summary>"
            f"{blocks}</details>"
        )
    else:
        calls_html = "<p class='muted small'>No model call for this check (deterministic, or gated off).</p>"

    return f"""<details class="check-card check-{BAND_CLASS.get(band, 'muted')}" open>
  <summary>
    <span class="pill pill-{BAND_CLASS.get(band, 'muted')}">{_e(BAND_LABEL.get(band, band))}</span>
    <span class="check-id mono">{_e(cid)}</span>
    <span class="check-name">{_e(name)}</span>
    {f"<span class='check-code'>{_e(code)}</span>" if code else ''}
  </summary>
  <div class="check-body">
    {f"<p>{_e(why)}</p>" if why else ''}
    {ev_html}
    {''.join(extras)}
    <details class="raw-measurement"><summary>Raw measurement</summary>{meas_html}</details>
    {calls_html}
  </div>
</details>"""


def _rubric_html(task: Task | None) -> str:
    if not task or not task.rubric:
        return "<p class='muted'>Rubric is empty on this task by design for this batch &mdash; not a reason the task is unauditable. Checks 200/230/240/250/260/270 (and 1000 itself) are `not_evaluated` for this reason alone; every other check below still ran for real.</p>"
    rows = "".join(
        f"<tr><td class='mono'>C{i}</td><td>{_e(c.text)}</td><td class='center'>{_e(c.weight if c.weight is not None else '\u2014')}</td>"
        f"<td>{_e(c.l1_label or '\u2014')}</td></tr>"
        for i, c in enumerate(task.rubric, start=1)
    )
    return f"<table class='striped'><thead><tr><th>#</th><th>Criterion</th><th>Weight</th><th>L1</th></tr></thead><tbody>{rows}</tbody></table>"


def _dimension_ratings_html(task: Task | None) -> str:
    if not task or not task.dimension_ratings:
        return "<p class='muted'>No dimension ratings on file.</p>"
    rows = []
    for r in sorted(task.dimension_ratings, key=lambda r: (r.model, r.dimension)):
        val = "N/A" if r.not_applicable else (r.rating if r.rating is not None else "\u2014")
        turns = ", ".join(str(t) for t in r.relevant_turns) or "\u2014"
        rows.append(
            f"<tr><td class='mono center'>{_e(r.model)}</td><td>{_e(r.dimension)}</td>"
            f"<td class='center'>{_e(val)}</td><td>{_e(turns)}</td>"
            f"<td>{_e(r.justification)}</td></tr>"
        )
    return (
        "<table class='striped'><thead><tr><th>Model</th><th>Dimension</th><th>Contributor rating</th>"
        "<th>Cited turns</th><th>Contributor justification</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _sxs_html(task: Task | None) -> str:
    if not task:
        return ""
    sxs = task.sxs
    parts = []
    if sxs.likert is not None:
        parts.append(f"<p class='meta-line'><b>Likert:</b> {_e(sxs.likert)}"
                     + (f" (winner index {_e(sxs.winner_index)})" if sxs.winner_index is not None else "") + "</p>")
    if sxs.justification:
        parts.append(f"<p class='meta-line'><b>Contributor's SxS justification:</b></p>{_pre(sxs.justification)}")
    if task.key_turn.turn_index is not None or task.key_turn.justification:
        parts.append(
            f"<p class='meta-line'><b>Key turn:</b> {_e(task.key_turn.turn_index)}</p>"
            + (_pre(task.key_turn.justification) if task.key_turn.justification else "")
        )
    return "".join(parts) or "<p class='muted'>No side-by-side data on file.</p>"


def _prompts_html(task: Task | None) -> str:
    if not task:
        return "<p class='muted'>No prompts on file for this task.</p>"
    from .context import numbered_prompts

    text = numbered_prompts(task)
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("### "):
            heading, _, rest = block.partition("\n")
            blocks.append(f"<p class='turn-label'>{_e(heading[4:])}</p><p>{_e(rest.strip())}</p>")
        else:
            blocks.append(f"<p>{_e(block)}</p>")
    body = "".join(blocks) if blocks else f"<p>{_e(text)}</p>"

    meta = []
    if task.assigned_domain:
        meta.append(f"<b>Assigned persona/domain:</b> {_e(task.assigned_domain)}")
    if task.prompt_category:
        meta.append(f"<b>Prompt category (CUJ):</b> {_e(task.prompt_category)}")
    meta_html = f"<p class='meta-line'>{' &nbsp;&middot;&nbsp; '.join(meta)}</p>" if meta else ""

    extra = ""
    if task.pre_seeded_prompt and task.pre_seeded_prompt.strip():
        extra = (
            "<p class='turn-label'>Pre-seeded prompt (platform template, before the contributor wrote anything)</p>"
            f"<p>{_e(task.pre_seeded_prompt)}</p>"
        )
    deliverables = ""
    if task.target_deliverables:
        items = "".join(f"<li>{_e(d)}</li>" for d in task.target_deliverables)
        deliverables = f"<h4>Target deliverables (contributor's own list)</h4><ul>{items}</ul>"
    outcome = ""
    if task.target_outcome:
        items = "".join(f"<li>{_e(d)}</li>" for d in task.target_outcome)
        outcome = f"<h4>Target outcome</h4><ul>{items}</ul>"

    return meta_html + extra + body + deliverables + outcome


def _task_panel_html(t_raw: dict, task: Task | None, rubric_findings: dict, informed: dict | None, calls_by_check: dict[int, list[dict]]) -> str:
    task_id = t_raw["task_id"]
    suffix = task_id[-4:]
    checks = sorted(t_raw["checks"], key=lambda c: ({"fail": 0, "non_fail": 1, "not_evaluated": 2, "clean": 3}.get(c["band"], 4), c["check_id"]))
    n_fail = sum(1 for c in checks if c["band"] == "fail")
    n_nonfail = sum(1 for c in checks if c["band"] == "non_fail")

    pills = [
        f"<span class='pill pill-neutral'>Verdict: {_e(t_raw.get('verdict', ''))}</span>",
        f"<span class='pill pill-{'bad' if n_fail else 'muted'}'>{n_fail} fail check(s)</span>",
        f"<span class='pill pill-{'warn' if n_nonfail else 'muted'}'>{n_nonfail} non-fail check(s)</span>",
    ]

    check_cards = "".join(
        _check_card_html(c, calls_by_check.get(c["check_id"], []), rubric_findings, informed, suffix)
        for c in checks
        if c["check_id"] != 1000
    )

    parts = [
        f"<section class='panel' id='panel-task-{_e(suffix)}' hidden>",
        "<div class='panel-header'>",
        f"<h2>Task \u2026{_e(suffix)} <span class='task-id-full mono'>{_e(task_id)}</span></h2>",
        f"<div class='pills'>{''.join(pills)}</div>",
        "</div>",
        f"<div class='card'><h4>Prompts (verbatim)</h4><div class='prompts'>{_prompts_html(task)}</div></div>",
        f"<div class='card'><h4>Rubric</h4>{_rubric_html(task)}</div>",
        f"<div class='card'><h4>Contributor's dimension ratings (what checks 300/310 blind-compare against)</h4>{_dimension_ratings_html(task)}</div>",
        f"<div class='card'><h4>Contributor's side-by-side pick (what checks 400/450/470 blind-compare against)</h4>{_sxs_html(task)}</div>",
        "<h3>Every check, its verdict, and the model call(s) behind it</h3>",
        check_cards,
        "</section>",
    ]
    return "".join(parts)


def _sidebar_html(tasks_out: list[dict]) -> str:
    rows = []
    for t in tasks_out:
        n_fail = t["n_fail"]
        bar_cls = "bar-bad" if n_fail else ("bar-warn" if t["n_nonfail"] else "bar-good")
        rows.append(
            f"<button class='task-row' data-target='panel-task-{_e(t['suffix'])}' "
            f"onclick=\"selectPanel('panel-task-{_e(t['suffix'])}', this)\">"
            f"<span class='row-bar {bar_cls}'></span>"
            f"<span class='row-id mono'>{_e(t['suffix'])}</span>"
            f"<span class='row-counts'>{t['n_fail']}F / {t['n_nonfail']}NF</span>"
            f"</button>"
        )
    return (
        "<nav class='sidebar'>"
        "<button class='nav-item selected' data-target='panel-overview' "
        "onclick=\"selectPanel('panel-overview', this)\">Overview</button>"
        "<div class='sidebar-section-header'>Tasks <span class='badge'>" + str(len(tasks_out)) + "</span></div>"
        "<div class='sidebar-list'>" + "".join(rows) + "</div>"
        "</nav>"
    )


SCRIPT = """
function selectPanel(id, btn) {
  document.querySelectorAll('.panel').forEach(function (el) { el.hidden = true; });
  document.getElementById(id).hidden = false;
  document.querySelectorAll('.nav-item, .task-row').forEach(function (el) { el.classList.remove('selected'); });
  btn.classList.add('selected');
  document.querySelector('.detail').scrollTop = 0;
}
"""

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b6b6b; --border: #e5e5e5;
  --stripe: #fafafa; --good: #2f7a4f; --good-bg: #eaf5ee;
  --bad: #b23b3b; --bad-bg: #fbebeb; --warn: #a3760a; --warn-bg: #fbf3e1;
  --accent: #2b5fb0; --sidebar-w: 250px; --topbar-h: 96px;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); line-height: 1.5; font-size: 14px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 16px; margin: 0 0 14px; font-weight: 600; }
h3 { font-size: 14px; margin: 20px 0 8px; }
h4 { font-size: 11.5px; margin: 0 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
p { margin: 0 0 10px; }

.topbar { padding: 16px 28px; border-bottom: 1px solid var(--border); height: var(--topbar-h); box-sizing: border-box; }
.subtitle { color: var(--muted); font-size: 12.5px; max-width: 1000px; }

.layout { display: flex; height: calc(100vh - var(--topbar-h)); }

.sidebar { width: var(--sidebar-w); flex-shrink: 0; border-right: 1px solid var(--border);
           overflow-y: auto; padding: 10px 0; background: #fcfcfc; }
.nav-item { display: block; width: 100%; text-align: left; border: none; background: none; cursor: pointer;
            padding: 10px 20px; font-size: 13.5px; font-weight: 600; color: var(--fg); }
.nav-item:hover { background: #f0f0f0; }
.nav-item.selected { background: #e8eef8; color: var(--accent); border-left: 3px solid var(--accent); padding-left: 17px; }
.sidebar-section-header { padding: 16px 20px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
                           color: var(--muted); font-weight: 600; display: flex; align-items: center; gap: 6px; }
.badge { background: #eee; color: var(--muted); border-radius: 10px; padding: 1px 7px; font-size: 11px; font-weight: 600; }
.sidebar-list { display: flex; flex-direction: column; }
.task-row { display: flex; align-items: center; gap: 8px; width: 100%; border: none; background: none; cursor: pointer;
            padding: 7px 20px 7px 17px; text-align: left; font-size: 13px; color: var(--fg); }
.task-row:hover { background: #f0f0f0; }
.task-row.selected { background: #e8eef8; }
.row-bar { width: 3px; height: 16px; border-radius: 2px; flex-shrink: 0; }
.row-bar.bar-bad { background: var(--bad); } .row-bar.bar-warn { background: var(--warn); } .row-bar.bar-good { background: #cfd6e0; }
.row-id { flex: 1; }
.row-counts { font-size: 10.5px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

.detail { flex: 1; overflow-y: auto; padding: 28px 36px 80px; }
.panel-header { margin-bottom: 22px; }
.panel-header h2 { display: flex; align-items: baseline; gap: 10px; }
.task-id-full { font-size: 12px; color: var(--muted); font-weight: 400; }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500;
        border: 1px solid var(--border); background: #f4f4f4; color: var(--fg); }
.pill-good { background: var(--good-bg); color: var(--good); border-color: transparent; }
.pill-bad { background: var(--bad-bg); color: var(--bad); border-color: transparent; }
.pill-warn { background: var(--warn-bg); color: var(--warn); border-color: transparent; }
.pill-neutral { background: #eef1f6; color: #35507a; border-color: transparent; }
.pill-muted { background: #f2f2f2; color: var(--muted); border-color: transparent; }

.card { border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; margin-bottom: 16px; background: #fff; }

.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 22px; }
.stat { border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 160px; flex: 1; background: #fff; }
.stat .v { font-size: 20px; font-weight: 700; }
.stat .l { font-size: 11.5px; color: var(--muted); margin-top: 3px; }
.stat.good .v { color: var(--good); } .stat.bad .v { color: var(--bad); } .stat.warn .v { color: var(--warn); }

table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 0; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.02em; }
table.striped tbody tr:nth-child(even) { background: var(--stripe); }
.center { text-align: center; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }

.prompts { border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; background: #fbfbfb; }
.turn-label { font-weight: 600; color: var(--accent); font-size: 12px; margin: 10px 0 2px; }
.turn-label:first-child { margin-top: 0; }
.meta-line { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.meta-line b { color: var(--fg); }
.muted { color: var(--muted); font-size: 13px; }
.small { font-size: 12px; }
.quote { color: var(--accent); font-style: italic; }

pre { white-space: pre-wrap; word-break: break-word; background: #f7f7f8; border: 1px solid var(--border);
      border-radius: 6px; padding: 10px 12px; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      max-height: 480px; overflow-y: auto; margin: 0 0 10px; }

.check-card { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; background: #fff; overflow: hidden; }
.check-card.check-bad { border-left: 4px solid var(--bad); }
.check-card.check-warn { border-left: 4px solid var(--warn); }
.check-card.check-good { border-left: 4px solid var(--good); }
.check-card.check-muted { border-left: 4px solid #ccc; }
.check-card > summary { cursor: pointer; padding: 10px 14px; display: flex; align-items: center; gap: 10px;
                        list-style: none; font-size: 13.5px; }
.check-card > summary::-webkit-details-marker { display: none; }
.check-id { color: var(--muted); }
.check-name { font-weight: 600; }
.check-code { color: var(--muted); font-size: 12px; margin-left: auto; }
.check-body { padding: 0 16px 14px 16px; }
.evidence-list { margin: 6px 0 10px; padding-left: 18px; font-size: 12.5px; }
.evidence-list li { margin-bottom: 4px; }

details.raw-measurement, details.calls-group { margin-top: 8px; }
details.raw-measurement > summary, details.calls-group > summary { cursor: pointer; font-size: 12px; color: var(--accent); }
details.call { border: 1px solid var(--border); border-radius: 6px; margin: 8px 0 0; padding: 0; background: #fbfbfb; }
details.call > summary { cursor: pointer; padding: 8px 12px; font-size: 12.5px; font-weight: 600; }
.call-body { padding: 4px 12px 12px; }

footer { color: var(--muted); font-size: 11.5px; margin-top: 30px; border-top: 1px solid var(--border); padding-top: 14px; }
"""


def build_report_html(
    report_paths: list[Path],
    tasks_csv: Path,
    cache_db: Path | None,
    title: str = "Manual audit \u2014 batch7 rubric-empty rerun",
    policy: Policy = DEFAULT_POLICY,
    hydrate_workers: int = 8,
) -> str:
    merged = merge_reports(report_paths)

    ingested, _ = load_taskattempts_csv(tasks_csv)

    # `load_taskattempts_csv` alone leaves every submission's `conversation`
    # exactly where raw ingestion put it: the turn-manifest's user-only
    # record, per `TurnManifestEntry`'s documented limitation that "the
    # model's reply is never recorded here". The live run this viewer is
    # reporting on always passed `--fetch-conversations`, which replaces that
    # with the real hydrated transcript -- user turns *and* the model's
    # actual replies -- before building a single prompt. Skipping that step
    # here would make every "regenerated, byte-identical" prompt silently
    # drop the model's side of the conversation, which is exactly the
    # evidence a manual audit of these checks needs to see. This hits the
    # snapshot cache the live run already populated, so it costs no new
    # fetches as long as `policy.snapshot_dir` points at it.
    report = hydrate_conversations(ingested, policy, workers=hydrate_workers)
    print(
        f"hydrated {report.hydrated}/{report.submissions} submissions "
        f"({report.turns} turns) from {policy.snapshot_dir} for display",
        file=sys.stderr,
    )
    for line in report.failures:
        print(f"  not fetched: {line}", file=sys.stderr)

    task_by_id = {t.task_id: t for t in ingested}

    rubric_by_task = {r["task_id"]: r for r in merged.get("rubric_stage") or []}
    informed_by_task = {r["task_id"]: r for r in merged.get("informed_stage") or []}

    conn = sqlite3.connect(str(cache_db)) if cache_db and Path(cache_db).exists() else None

    tasks_out = []
    for t_raw in merged["tasks"]:
        task_id = t_raw["task_id"]
        task = task_by_id.get(task_id)
        rubric = rubric_by_task.get(task_id) or {}
        rubric_findings = {f["criterion_id"]: f for f in rubric.get("findings", [])}
        informed = informed_by_task.get(task_id)
        calls_by_check = _calls_by_check(task, policy, conn) if task is not None else {}

        checks = t_raw["checks"]
        n_fail = sum(1 for c in checks if c["band"] == "fail")
        n_nonfail = sum(1 for c in checks if c["band"] == "non_fail")
        n_not_eval = sum(1 for c in checks if c["band"] == "not_evaluated")
        n_clean = sum(1 for c in checks if c["band"] == "clean")

        tasks_out.append(
            {
                "task_id": task_id,
                "suffix": task_id[-4:],
                "verdict": t_raw.get("verdict", ""),
                "n_fail": n_fail,
                "n_nonfail": n_nonfail,
                "n_not_eval": n_not_eval,
                "n_clean": n_clean,
                "panel": _task_panel_html(t_raw, task, rubric_findings, informed, calls_by_check),
            }
        )

    if conn is not None:
        conn.close()

    tasks_out.sort(key=lambda t: (-t["n_fail"], -t["n_nonfail"], t["suffix"]))

    total = len(tasks_out)
    total_fail = sum(t["n_fail"] for t in tasks_out)
    total_nonfail = sum(t["n_nonfail"] for t in tasks_out)
    tasks_with_fail = sum(1 for t in tasks_out if t["n_fail"])
    tasks_with_any_flag = sum(1 for t in tasks_out if t["n_fail"] or t["n_nonfail"])

    stat_cards = [
        (str(total), "Tasks (empty rubric by design \u2014 none of these are 'unauditable')", ""),
        (str(tasks_with_fail), "Now carry at least one real Fail-band check", "bad" if tasks_with_fail else "good"),
        (str(tasks_with_any_flag), "Now carry at least one Fail or Non-fail check", "warn" if tasks_with_any_flag else "good"),
        (str(total_fail), "Total Fail-band checks across all 30", "bad" if total_fail else "good"),
        (str(total_nonfail), "Total Non-fail-band checks across all 30", "warn" if total_nonfail else "good"),
    ]
    stats_html = "".join(
        f"<div class='stat {cls}'><div class='v'>{_e(v)}</div><div class='l'>{_e(l)}</div></div>"
        for v, l, cls in stat_cards
    )

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    sidebar_html = _sidebar_html(tasks_out)
    task_panels = "".join(t["panel"] for t in tasks_out)

    overview = f"""<section class="panel" id="panel-overview">
  <h2>30 rubric-empty tasks, re-run with rating + informed stages live</h2>
  <p class="subtitle">Every one of these tasks has an empty rubric by design for this batch -- expected, not a
  defect. `cli.py`'s gate withholds only the rubric-dependent checks (200/230/240/250/260/270) for that reason;
  300/400/450/470/70-110 all ran for real against the actual conversation and the contributor's own ratings, and
  check 1000 itself reports `not_evaluated` rather than a fail, so none of these 30 tasks roll up as
  "unauditable" on that basis alone. Click a task in the sidebar to see every check's verdict, its justification,
  and the exact model call(s) that produced it \u2014 prompt, system prompt, and raw JSON answer, pulled from the
  response cache by the same request key the live run wrote it under.</p>
  <div class="stats">{stats_html}</div>
  <footer>
    Sources: {'; '.join(_e(p) for p in report_paths)}, joined against {_e(tasks_csv)}
    {f'and {_e(cache_db)} for the model call detail' if cache_db else ''}.
    Sidebar is sorted worst-first: most Fail-band checks, then most Non-fail. Bar colour: red = has a Fail,
    amber = Non-fail only, grey = fully clean/not_evaluated.
  </footer>
</section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_e(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{CSS}</style>
</head>
<body>
<div class="topbar">
  <h1>{_e(title)}</h1>
  <p class="subtitle">Generated {_e(generated)} \u00b7 {total} tasks \u00b7 ${_e(f"{merged.get('cost_usd', 0):.2f}")} model spend.</p>
</div>
<div class="layout">
  {sidebar_html}
  <main class="detail">
    {overview}
    {task_panels}
  </main>
</div>
<script>{SCRIPT}</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="append", required=True, dest="reports", help="repeatable")
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Manual audit \u2014 batch7 rubric-empty rerun")
    ap.add_argument(
        "--snapshot-dir",
        default=None,
        help="the live run's --snapshot-dir, so regenerated prompts can be "
             "hydrated with the model's actual conversation turns (and "
             "deliverable attachment text) from its cached DOMs/PDFs, exactly "
             "as the live run saw them. Without this, every regenerated "
             "prompt shows the user's side of the conversation only -- the "
             "raw ingest's turn-manifest fallback never records the model's "
             "replies -- which misrepresents what the cached judgment was "
             "actually based on.",
    )
    ap.add_argument(
        "--hydrate-workers", type=int, default=8,
        help="parallel fetches while hydrating; irrelevant when --snapshot-dir "
             "already has everything cached, which is the expected case here",
    )
    args = ap.parse_args(argv)

    policy = DEFAULT_POLICY
    if args.snapshot_dir:
        policy = replace(policy, snapshot_dir=args.snapshot_dir, fetch_attachment_text=True)
    else:
        print(
            "warning: no --snapshot-dir given; regenerated prompts will show "
            "only the user's side of each conversation, not what the live run "
            "actually sent the model",
            file=sys.stderr,
        )

    html_text = build_report_html(
        [Path(p) for p in args.reports],
        Path(args.tasks_csv),
        Path(args.cache_db) if args.cache_db else None,
        args.title,
        policy,
        args.hydrate_workers,
    )
    Path(args.out).write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
