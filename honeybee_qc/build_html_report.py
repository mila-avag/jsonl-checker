"""Render an audit run into one self-contained, shareable HTML report.

Same per-check comparison `build_canvas.py` computes (QC vs. our band, why, and
the specific evidence behind it), plus the three things a Cursor canvas can't
carry end to end: the task's full verbatim prompts, its full rubric criteria,
and its target deliverables. One HTML file, no server, no JS dependency --
`<details>`/`<summary>` does the collapsing natively, so it opens and reads the
same in any browser it's shared to.

Usage:
    python3 -m honeybee_qc.build_html_report \
        --report audit_runs/report.json \
        --sheet audit_runs/qc_validations.csv \
        --tasks-csv audit_runs/tasks.csv \
        --out audit_runs/report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
from pathlib import Path

from . import build_canvas
from .ingest import load_taskattempts_csv
from .models import Task

MATCH_LABEL = {
    "agree": "Agree",
    "disagree": "Disagree \u2014 flagged, wrong band",
    "missed": "Missed \u2014 not flagged",
    "extra": "Extra \u2014 we flagged, QC silent",
}
MATCH_CLASS = {
    "agree": "good",
    "disagree": "bad",
    "missed": "bad",
    "extra": "warn",
}
FAIL_STATUS_LABEL = {
    "caught": "Caught \u2014 same check, fail band",
    "wrong_band": "Flagged, wrong band",
    "missed": "Missed entirely",
}
FAIL_STATUS_CLASS = {"caught": "good", "wrong_band": "warn", "missed": "bad"}


def _e(text: object) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _classify_row(qc_band: str | None, our_band: str) -> str:
    our_flagged = our_band in ("fail", "non_fail")
    if qc_band and our_flagged:
        return "disagree" if (qc_band == "fail" and our_band != "fail") else "agree"
    if qc_band and not our_flagged:
        return "missed"
    if not qc_band and our_flagged:
        return "extra"
    return "n/a"


def _rows_for_task(task_canvas: dict, raw_bands: dict[int, str]) -> list[dict]:
    """Unify QC-cited checks and our flagged checks into one comparison table."""
    qc_ids = {c["id"]: c for c in task_canvas["qc_checks"]}
    audit_ids = {c["id"]: c for c in task_canvas["checks"]}
    all_ids = sorted(
        set(qc_ids) | set(audit_ids) | {cid for cid, b in raw_bands.items() if b in ("fail", "non_fail")}
    )
    rows = []
    for cid in all_ids:
        qc = qc_ids.get(cid)
        au = audit_ids.get(cid)
        true_band = raw_bands.get(cid, "not_evaluated")
        qc_band = "fail" if (qc and qc["band"] == "fail") else ("non_fail" if qc else None)
        match = _classify_row(qc_band, true_band)
        if match == "n/a":
            continue
        evidence = (au or {}).get("evidence") or []
        ev_txt = "; ".join(f"{it['criterion']}: {it['detail']}" for it in evidence[:6])
        more = len(evidence) - 6
        if more > 0:
            ev_txt += f" (+{more} more)"
        rows.append(
            {
                "checkId": cid,
                "checkName": (au or {}).get("name") or "",
                "qcPolarity": qc_band,
                "qcCode": (qc or {}).get("code") or "",
                "ourBand": true_band,
                "why": (au or {}).get("why") or "",
                "evidence": ev_txt,
                "match": match,
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["match"] == "missed" else 1 if r["match"] == "disagree" else 2 if r["match"] == "agree" else 3,
            0 if r["qcPolarity"] == "fail" else 1,
            r["checkId"],
        )
    )
    return rows


def _fail_items(tasks: list[dict]) -> list[dict]:
    out = []
    for t in tasks:
        if not t["qc_verdict"]:
            continue
        for r in t["rows"]:
            if r["qcPolarity"] != "fail":
                continue
            status = "caught" if r["match"] == "agree" else ("wrong_band" if r["match"] == "disagree" else "missed")
            out.append({**r, "suffix": t["suffix"], "status": status})
    order = {"missed": 0, "wrong_band": 1, "caught": 2}
    out.sort(key=lambda f: (order[f["status"]], f["suffix"]))
    return out


def _check_breakdown(tasks: list[dict]) -> list[dict]:
    rollup: dict[int, dict] = {}
    for t in tasks:
        for r in t["rows"]:
            entry = rollup.setdefault(r["checkId"], {"name": "", "agree": 0, "disagree": 0, "missed": 0, "extra": 0})
            if r["checkName"]:
                entry["name"] = r["checkName"]
            entry[r["match"]] += 1
    out = [{"checkId": cid, **v} for cid, v in rollup.items()]
    out.sort(key=lambda c: -(c["disagree"] + c["missed"] + c["extra"]))
    return out


def _rubric_rows_html(task: Task | None) -> str:
    if not task or not task.rubric:
        return "<p class='muted'>No rubric on file for this task.</p>"
    rows = []
    for i, c in enumerate(task.rubric, start=1):
        weight = "\u2014" if c.weight is None else c.weight
        l1 = c.l1_label or "\u2014"
        l2 = c.l2_label or "\u2014"
        rows.append(
            f"<tr><td class='mono'>C{i}</td><td>{_e(c.text)}</td>"
            f"<td class='center'>{_e(weight)}</td><td>{_e(l1)}</td><td>{_e(l2)}</td></tr>"
        )
    return (
        "<table class='striped'><thead><tr>"
        "<th>#</th><th>Criterion (verbatim)</th><th>Weight</th><th>L1</th><th>L2</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _deliverables_html(task: Task | None) -> str:
    if not task or not task.target_deliverables:
        return ""
    items = "".join(f"<li>{_e(d)}</li>" for d in task.target_deliverables)
    return f"<h4>Target deliverables (contributor's own list)</h4><ul>{items}</ul>"


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

    extra = []
    if task.pre_seeded_prompt and task.pre_seeded_prompt.strip():
        extra.append(
            f"<p class='turn-label'>Pre-seeded prompt (platform template, before the contributor wrote anything)</p>"
            f"<p>{_e(task.pre_seeded_prompt)}</p>"
        )
    meta = []
    if task.assigned_domain:
        meta.append(f"<b>Assigned persona/domain:</b> {_e(task.assigned_domain)}")
    if task.prompt_category:
        meta.append(f"<b>Prompt category (CUJ):</b> {_e(task.prompt_category)}")
    meta_html = f"<p class='meta-line'>{' &nbsp;&middot;&nbsp; '.join(meta)}</p>" if meta else ""

    return meta_html + "".join(extra) + body


def _comparison_table_html(rows: list[dict], with_qc: bool) -> str:
    if not rows:
        return ""
    trs = []
    for r in rows:
        cls = MATCH_CLASS[r["match"]]
        cells = [f"<td class='mono center'>{_e(r['checkId'])}</td>", f"<td>{_e(r['checkName'] or '\u2014')}</td>"]
        if with_qc:
            cells.append(f"<td class='center'>{'Fail' if r['qcPolarity'] == 'fail' else 'Non-fail'}</td>")
        cells.append(f"<td class='center'>{_e(r['ourBand'])}</td>")
        if with_qc:
            cells.append(f"<td>{_e(MATCH_LABEL[r['match']])}</td>")
        reasoning = " ".join(x for x in (r["why"], (f"e.g. {r['evidence']}" if r["evidence"] else "")) if x)
        cells.append(f"<td>{_e(reasoning) or '&mdash;'}</td>")
        trs.append(f"<tr class='row-{cls}'>{''.join(cells)}</tr>")
    headers = ["Check", "Dimension"]
    if with_qc:
        headers += ["QC", "Our band", "Match"]
    else:
        headers += ["Our band"]
    headers.append("Rubric/evidence \u2014 why we flagged it" + (" (or didn't)" if with_qc else ""))
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table class='striped'><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _bar_chart_html(rows: list[dict]) -> str:
    if not rows:
        return ""
    top = rows[:12]
    max_total = max((r["agree"] + r["disagree"] + r["missed"] + r["extra"]) for r in top) or 1
    lines = []
    for r in top:
        total = r["agree"] + r["disagree"] + r["missed"] + r["extra"]
        segs = []
        for key, cls in (("agree", "good"), ("disagree", "bad"), ("missed", "bad2"), ("extra", "warn")):
            v = r[key]
            if v:
                pct = 100 * v / max_total
                segs.append(f"<div class='seg seg-{cls}' style='width:{pct:.2f}%' title='{key}: {v}'></div>")
        lines.append(
            f"<div class='bar-row'><div class='bar-label mono'>{_e(r['checkId'])}</div>"
            f"<div class='bar-track'>{''.join(segs)}</div>"
            f"<div class='bar-total'>{total}</div></div>"
        )
    legend = (
        "<div class='legend'>"
        "<span class='dot dot-good'></span>Agreed with QC "
        "<span class='dot dot-bad'></span>Disagreed/missed "
        "<span class='dot dot-warn'></span>Uncited extra flag"
        "</div>"
    )
    return f"<div class='barchart'>{''.join(lines)}</div>{legend}"


def _verdict_pill(label: str, kind: str) -> str:
    return f"<span class='pill pill-{kind}'>{_e(label)}</span>"


def _fail_category_stats(t: dict) -> dict:
    """Fail-severity check IDs QC cited vs. the ones we banded fail, as a Venn split.

    A "fail category" is a check (dimension), not a single rubric criterion --
    the same granularity QC's own citations use. `qc_total` and `eval_total` are
    each already unique-category counts (set sizes); `overlap` is what landed in
    both, `qc_only`/`eval_only` are each side's unique-to-it remainder.
    """
    qc_fail_ids = {r["checkId"] for r in t["rows"] if r["qcPolarity"] == "fail"}
    eval_fail_ids = {r["checkId"] for r in t["rows"] if r["ourBand"] == "fail"}
    overlap = qc_fail_ids & eval_fail_ids
    return {
        "qc_total": len(qc_fail_ids),
        "eval_total": len(eval_fail_ids),
        "overlap": len(overlap),
        "qc_only": len(qc_fail_ids - eval_fail_ids),
        "eval_only": len(eval_fail_ids - qc_fail_ids),
    }


def _task_panel_html(t: dict, task: Task | None) -> str:
    cited_rows = [r for r in t["rows"] if r["qcPolarity"] is not None]
    extra_rows = [r for r in t["rows"] if r["qcPolarity"] is None]
    n_problem = sum(1 for r in cited_rows if r["qcPolarity"] == "fail" and r["match"] != "agree")
    stats = _fail_category_stats(t)

    pills = []
    if t["qc_verdict"]:
        pills.append(_verdict_pill(f"QC: {t['qc_verdict']} (score {t['qc_score']})", "neutral"))
    else:
        pills.append(_verdict_pill("No QC ground truth on file", "muted"))
    pills.append(_verdict_pill(f"Our verdict: {t['audit_verdict']}", "bad" if t["audit_verdict"] == "fail" else "good"))
    if t.get("project_team_verdict"):
        pills.append(_verdict_pill(f"Project team: {t['project_team_verdict']}", "neutral"))
    if stats["qc_total"] or stats["eval_total"]:
        pills.append(_verdict_pill(f"QC unique fail categories: {stats['qc_total']}", "neutral"))
        pills.append(_verdict_pill(f"Eval unique fail categories: {stats['eval_total']}", "neutral"))
        overlap_kind = (
            "good"
            if stats["qc_total"] and stats["overlap"] == stats["qc_total"]
            else ("bad" if stats["qc_total"] and stats["overlap"] == 0 else "warn")
        )
        pills.append(_verdict_pill(f"Overlap: {stats['overlap']}", overlap_kind))
    if n_problem:
        pills.append(_verdict_pill(f"{n_problem} QC fail item(s) missed/wrong-band", "bad"))

    parts = [
        f"<section class='panel' id='panel-task-{_e(t['suffix'])}' hidden>",
        "<div class='panel-header'>",
        f"<h2>Task \u2026{_e(t['suffix'])} <span class='task-id-full mono'>{_e(t['task_id'])}</span></h2>",
        f"<div class='pills'>{''.join(pills)}</div>",
        "</div>",
    ]

    if t.get("qc_feedback"):
        parts.append(
            f"<div class='card'><h4>QC's feedback (verbatim)</h4>"
            f"<p class='qc-feedback'>{_e(t['qc_feedback'])}</p></div>"
        )

    parts.append(
        f"<div class='card'><h4>Prompts (verbatim)</h4><div class='prompts'>{_prompts_html(task)}</div>"
        f"{_deliverables_html(task)}</div>"
    )
    parts.append(f"<div class='card'><h4>Rubric (verbatim)</h4>{_rubric_rows_html(task)}</div>")

    if cited_rows:
        parts.append(
            "<div class='card'><h4>What QC cited, and our flag for the same check</h4>"
            f"{_comparison_table_html(cited_rows, with_qc=True)}</div>"
        )
    if extra_rows:
        parts.append(
            "<div class='card'><h4>Fails/non-fails we raised that QC's feedback never mentions</h4>"
            f"{_comparison_table_html(extra_rows, with_qc=False)}</div>"
        )

    parts.append("</section>")
    return "".join(parts)


def _sidebar_html(tasks_out: list[dict]) -> str:
    rows = []
    for t in tasks_out:
        n_problem = sum(
            1
            for r in t["rows"]
            if r["qcPolarity"] == "fail" and r["match"] != "agree"
        )
        stats = _fail_category_stats(t)
        bar_cls = "bar-bad" if n_problem else ("bar-none" if not t["qc_verdict"] else "bar-good")
        verdict_dot = "dot-bad" if t["audit_verdict"] == "fail" else "dot-good"
        tri_title = (
            f"QC flagged {stats['qc_total']} unique fail categor{'y' if stats['qc_total'] == 1 else 'ies'}; "
            f"eval flagged {stats['eval_total']}; {stats['overlap']} overlap"
        )

        rows.append(
            f"<button class='task-row' data-target='panel-task-{_e(t['suffix'])}' "
            f"onclick=\"selectPanel('panel-task-{_e(t['suffix'])}', this)\" title='{_e(tri_title)}'>"
            f"<span class='row-bar {bar_cls}'></span>"
            f"<span class='row-dot {verdict_dot}'></span>"
            f"<span class='row-id mono'>{_e(t['suffix'])}</span>"
            f"<span class='row-tri'>"
            f"<span class='tri tri-qc'>{stats['qc_total']}</span>"
            f"<span class='tri tri-overlap'>{stats['overlap']}</span>"
            f"<span class='tri tri-eval'>{stats['eval_total']}</span>"
            f"</span>"
            f"</button>"
        )
    legend = (
        "<div class='tri-legend'>"
        "<span><span class='tri tri-qc'>QC</span></span>"
        "<span><span class='tri tri-overlap'>&cap;</span></span>"
        "<span><span class='tri tri-eval'>Eval</span></span>"
        "</div>"
    )
    return (
        "<nav class='sidebar'>"
        "<button class='nav-item selected' data-target='panel-overview' "
        "onclick=\"selectPanel('panel-overview', this)\">Overview</button>"
        "<div class='sidebar-section-header'>Tasks <span class='badge'>" + str(len(tasks_out)) + "</span></div>"
        + legend
        + "<div class='sidebar-list'>" + "".join(rows) + "</div>"
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
  --accent: #2b5fb0; --sidebar-w: 240px; --topbar-h: 96px;
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
a { color: var(--accent); }

.topbar { padding: 16px 28px; border-bottom: 1px solid var(--border); height: var(--topbar-h); box-sizing: border-box; }
.subtitle { color: var(--muted); font-size: 12.5px; max-width: 1000px; }

.layout { display: flex; height: calc(100vh - var(--topbar-h)); }

.sidebar { width: var(--sidebar-w); flex-shrink: 0; border-right: 1px solid var(--border);
           overflow-y: auto; padding: 10px 0; background: #fcfcfc; }
.nav-item { display: block; width: 100%; text-align: left; border: none; background: none; cursor: pointer;
            padding: 10px 20px; font-size: 13.5px; font-weight: 600; color: var(--fg); border-radius: 0; }
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
.row-bar { width: 3px; height: 16px; border-radius: 2px; background: transparent; flex-shrink: 0; }
.row-bar.bar-bad { background: var(--bad); }
.row-bar.bar-good { background: #cfd6e0; }
.row-bar.bar-none { background: #e6d9b0; }
.row-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.row-dot.dot-good { background: var(--good); } .row-dot.dot-bad { background: var(--bad); }
.row-id { flex: 1; color: var(--fg); }
.row-tri { display: flex; align-items: center; gap: 3px; }
.tri { font-size: 10.5px; font-weight: 700; border-radius: 7px; padding: 1px 5px; min-width: 16px;
       text-align: center; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.tri-qc { background: #eaf0fb; color: var(--accent); }
.tri-overlap { background: var(--good-bg); color: var(--good); }
.tri-eval { background: #f3ecfa; color: #7a3fa0; }
.tri-legend { display: flex; align-items: center; gap: 10px; padding: 0 20px 10px; font-size: 10.5px; color: var(--muted); }
.tri-legend .tri { padding: 0 5px; }

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
tr.row-good td:first-child { border-left: 3px solid var(--good); }
tr.row-bad td:first-child { border-left: 3px solid var(--bad); }
tr.row-warn td:first-child { border-left: 3px solid var(--warn); }

.prompts { border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; background: #fbfbfb; }
.turn-label { font-weight: 600; color: var(--accent); font-size: 12px; margin: 10px 0 2px; }
.turn-label:first-child { margin-top: 0; }
.meta-line { color: var(--muted); font-size: 12px; margin-bottom: 10px; }
.qc-feedback { background: #fbfbfb; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; font-size: 13px; margin: 0; }
.muted { color: var(--muted); font-size: 13px; }

.barchart { margin: 4px 0 4px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
.bar-label { width: 40px; text-align: right; color: var(--muted); }
.bar-track { flex: 1; display: flex; height: 14px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }
.seg { height: 100%; }
.seg-good { background: var(--good); } .seg-bad { background: var(--bad); }
.seg-bad2 { background: #d98c8c; } .seg-warn { background: var(--warn); }
.bar-total { width: 28px; font-size: 12px; color: var(--muted); }
.legend { font-size: 12px; color: var(--muted); margin-top: 8px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 4px 0 14px; vertical-align: -1px; }
.dot:first-child { margin-left: 0; }
.dot-good { background: var(--good); } .dot-bad { background: var(--bad); } .dot-warn { background: var(--warn); }

footer { color: var(--muted); font-size: 11.5px; margin-top: 30px; border-top: 1px solid var(--border); padding-top: 14px; }
"""


def build_report_html(
    report_path: Path,
    sheet_path: Path,
    tasks_csv: Path,
    title: str = "QC Backtest",
) -> str:
    canvas_data = build_canvas.build(report_path, sheet_path, tasks_csv)
    raw_payload = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    raw_bands_by_task = {t["task_id"]: {c["check_id"]: c["band"] for c in t["checks"]} for t in raw_payload["tasks"]}

    ingested, _ = load_taskattempts_csv(tasks_csv)
    task_by_id = {t.task_id: t for t in ingested}

    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    qc_feedback: dict[str, dict] = {}
    if sheet_path.exists():
        import re

        def clean(s: str | None) -> str:
            return re.sub(r"\s+", " ", (s or "").strip())

        rows_by_task: dict[str, list[dict]] = {}
        with sheet_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows_by_task.setdefault(row.get("Task ID", ""), []).append(row)
        for tid, rws in rows_by_task.items():
            overall = "; ".join(sorted({clean(r.get("Overall Auditor feedback")) for r in rws if clean(r.get("Overall Auditor feedback"))}))
            rubric_fb = "; ".join(sorted({clean(r.get("Auditor feedback (rubric criteria)")) for r in rws if clean(r.get("Auditor feedback (rubric criteria)"))}))
            proj_verdict = next((r.get("Project Team Verdict") for r in rws if r.get("Project Team Verdict")), "")
            qc_feedback[tid] = {"text": rubric_fb or overall, "project_team_verdict": proj_verdict or ""}

    tasks_out = []
    for t in canvas_data["tasks"]:
        tid = t["task_id"]
        rows = _rows_for_task(t, raw_bands_by_task.get(tid, {}))
        fb = qc_feedback.get(tid, {})
        task_obj = task_by_id.get(tid)
        tasks_out.append(
            {
                "task_id": tid,
                "suffix": tid[-4:],
                "qc_verdict": t["qc_verdict"],
                "qc_score": t["qc_score"],
                "audit_verdict": t["audit_verdict"],
                "rows": rows,
                "qc_feedback": fb.get("text", ""),
                "project_team_verdict": fb.get("project_team_verdict", ""),
                "criteria_count": t.get("criteria") or (len(task_obj.rubric) if task_obj else 0),
            }
        )

    order = {"Fail": 0, "": 1, "Pass": 2}
    tasks_out.sort(key=lambda t: (order.get(t["qc_verdict"], 1), t["suffix"]))

    qc_tasks = [t for t in tasks_out if t["qc_verdict"]]
    fail_rows = [r for t in qc_tasks for r in t["rows"] if r["qcPolarity"] == "fail"]
    all_cited = [r for t in qc_tasks for r in t["rows"] if r["qcPolarity"]]
    n_agree = sum(1 for r in all_cited if r["match"] == "agree")
    n_disagree = sum(1 for r in all_cited if r["match"] == "disagree")
    n_missed = sum(1 for r in all_cited if r["match"] == "missed")
    n_extra = sum(1 for t in tasks_out for r in t["rows"] if r["match"] == "extra")
    fail_agree = sum(1 for r in fail_rows if r["match"] == "agree")
    fail_disagree = sum(1 for r in fail_rows if r["match"] == "disagree")
    fail_missed = sum(1 for r in fail_rows if r["match"] == "missed")
    n_fail_all = sum(1 for t in tasks_out if t["audit_verdict"] == "fail")

    fail_items = _fail_items(tasks_out)
    breakdown = _check_breakdown(tasks_out)

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    stat_cards = [
        (str(len(tasks_out)), "Total tasks audited", ""),
        (str(len(qc_tasks)), "With QC ground truth on file", ""),
        (f"{fail_agree} / {len(fail_rows)}", "QC fail-items caught, same band", "good"),
        (f"{fail_disagree} / {len(fail_rows)}", "QC fail-items, wrong band", "warn" if fail_disagree else ""),
        (f"{fail_missed} / {len(fail_rows)}", "QC fail-items missed entirely", "bad" if fail_missed else "good"),
        (str(n_extra), "Extra flags QC never mentioned", "warn" if n_extra else ""),
        (f"{n_fail_all} / {len(tasks_out)}", "Tasks our pipeline verdicts Fail", ""),
    ]
    stats_html = "".join(
        f"<div class='stat {cls}'><div class='v'>{_e(v)}</div><div class='l'>{_e(l)}</div></div>"
        for v, l, cls in stat_cards
    )

    fail_item_rows = "".join(
        f"<tr class='row-{FAIL_STATUS_CLASS[f['status']]}'>"
        f"<td class='mono center'>{_e(f['suffix'])}</td>"
        f"<td>{_e(f['checkId'])} \u2014 {_e(f['checkName'] or '\u2014')}</td>"
        f"<td>{_e(f['qcCode'])}</td>"
        f"<td class='center'>{_e(f['ourBand'])}</td>"
        f"<td>{_e(FAIL_STATUS_LABEL[f['status']])}</td>"
        f"<td>{_e(' '.join(x for x in (f['why'], f['evidence']) if x)) or '&mdash;'}</td>"
        f"</tr>"
        for f in fail_items
    )
    fail_table_html = (
        "<table class='striped'><thead><tr><th>Task</th><th>Check</th><th>QC's fail code</th>"
        "<th>Our band</th><th>Status</th><th>Why</th></tr></thead>"
        f"<tbody>{fail_item_rows}</tbody></table>"
        if fail_items
        else "<p class='muted'>No QC fail-severity items on file.</p>"
    )

    breakdown_rows = "".join(
        f"<tr><td class='mono center'>{_e(c['checkId'])}</td><td>{_e(c['name'] or '\u2014')}</td>"
        f"<td class='center'>{c['agree'] or ''}</td><td class='center'>{c['disagree'] or ''}</td>"
        f"<td class='center'>{c['missed'] or ''}</td><td class='center'>{c['extra'] or ''}</td></tr>"
        for c in breakdown
    )
    breakdown_table_html = (
        "<table class='striped'><thead><tr><th>Check</th><th>Dimension</th><th>Agreed</th>"
        "<th>Disagreed</th><th>Missed</th><th>Uncited extra flag</th></tr></thead>"
        f"<tbody>{breakdown_rows}</tbody></table>"
    )

    task_panels = "".join(_task_panel_html(t, task_by_id.get(t["task_id"])) for t in tasks_out)
    sidebar_html = _sidebar_html(tasks_out)

    hydration = canvas_data.get("hydration") or {}
    cost = canvas_data.get("cost_usd", 0)

    overview_panel = f"""<section class="panel" id="panel-overview">
  <h2>QC fail-severity items \u2014 caught vs. missed</h2>
  <div class="stats">{stats_html}</div>
  <div class="card">{fail_table_html}</div>

  <h2>Where disagreements and uncited extra flags concentrate, by check</h2>
  <div class="card">
  {_bar_chart_html(breakdown)}
  {breakdown_table_html}
  </div>

  <footer>
    Source: {_e(report_path)} joined against {_e(sheet_path)} and {_e(tasks_csv)}.
    Match logic: &quot;agree&quot; = same band as QC; &quot;disagree&quot; = we flagged it but at the wrong
    severity; &quot;missed&quot; = QC cited it, we didn't flag it at all; &quot;extra&quot; = we flagged it and
    QC's feedback never mentions it (assumes QC's silence means QC found nothing wrong there). Sidebar bar:
    red = task has a QC fail-item we missed or wrong-banded; grey = QC-validated and clean; tan = no QC
    ground truth on file yet. Sidebar numbers, left to right: QC = count of unique fail categories (checks)
    QC cited for that task; &cap; = overlap, the categories both sides flagged fail; Eval = count of unique
    fail categories our pipeline flagged fail for that task (independent of QC; may include categories QC
    never mentioned).
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
  <p class="subtitle">Generated {_e(generated)} \u00b7 {len(tasks_out)} tasks \u00b7 ${_e(f'{cost:.2f}')} model spend \u00b7
  {_e(hydration.get('hydrated', 0))}/{_e(hydration.get('submissions', 0))} submissions hydrated,
  {_e(hydration.get('turns', 0))} turns.</p>
</div>
<div class="layout">
  {sidebar_html}
  <main class="detail">
    {overview_panel}
    {task_panels}
  </main>
</div>
<script>{SCRIPT}</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="QC Backtest")
    args = ap.parse_args(argv)

    html_text = build_report_html(Path(args.report), Path(args.sheet), Path(args.tasks_csv), args.title)
    Path(args.out).write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
