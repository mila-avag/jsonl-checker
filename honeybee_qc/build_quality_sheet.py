"""Build one shareable HTML reference sheet: every check this pipeline
implements (definition, bands, error codes, which stage/CLI flag turns it on)
plus live fail/non-fail counts aggregated across every task audited so far,
split by project/run.

Usage:
    python3 -m honeybee_qc.build_quality_sheet --out audit_runs/quality_sheet.html
"""
from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

from .errors import ERROR_CODES
from .registry import BLOCKS, ORDER, REGISTRY
from .score_against_qc import DETERMINISTIC, INFORMED_STAGE, RATING_STAGE, RUBRIC_STAGE

RESTRICTED_CHECKS = (270, 280, 310, 400, 450, 460, 470)

RUN_DIRS = {
    "L0 -- restricted-scope": "honeybee_qc/audit_runs/l0_overnight_20260813",
    "L1 -- full mode": "honeybee_qc/audit_runs/honeybee_l1_full_20260813",
}

BLOCK_LABEL = {
    "setup_and_inputs": "Setup & inputs",
    "rubric_authoring": "Rubric authoring",
    "rating_accuracy": "Rating accuracy",
    "comparison": "SxS comparison",
    "escape_hatch": "Escape hatch",
}


def _e(text: object) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def stage_of(check_id: int) -> str:
    if check_id in DETERMINISTIC:
        return "deterministic"
    if check_id in RUBRIC_STAGE:
        return "rubric"
    if check_id in RATING_STAGE:
        return "rating"
    if check_id in INFORMED_STAGE:
        return "informed"
    return "preflight/other"


def block_of(check_id: int) -> str:
    for block, ids in BLOCKS.items():
        if check_id in ids:
            return BLOCK_LABEL.get(block, block)
    return ""


def collect_stats() -> dict:
    stats = {}
    for label, run_dir in RUN_DIRS.items():
        n_tasks = 0
        verdicts: dict[str, int] = {}
        counts: dict[int, dict[str, int]] = {}
        cost = 0.0
        for path in sorted(glob.glob(f"{run_dir}/single_*.json")):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            cost += data.get("cost_usd") or 0.0
            for t in data.get("tasks", []):
                n_tasks += 1
                verdicts[t["verdict"]] = verdicts.get(t["verdict"], 0) + 1
                for c in t["checks"]:
                    if c["band"] in ("fail", "non_fail"):
                        counts.setdefault(c["check_id"], {"fail": 0, "non_fail": 0})
                        counts[c["check_id"]][c["band"]] += 1
        stats[label] = {"n_tasks": n_tasks, "verdicts": verdicts, "counts": counts, "cost": cost}
    return stats


CSS = """
:root{--bg:#0f1117;--panel:#171a23;--panel2:#1e222d;--border:#2a2f3c;--text:#e6e8ee;
--muted:#9aa3b2;--accent:#6ea8fe;--fail:#ff7a7e;--nonfail:#e6bd63;--det:#5ed37a;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--text);line-height:1.5;font-size:14px}
header{padding:24px 32px;border-bottom:1px solid var(--border);background:linear-gradient(180deg,#1a1e29,#12141c)}
header h1{margin:0;font-size:20px}
header p{margin:6px 0 0;color:var(--muted);font-size:13px}
main{padding:24px 32px;max-width:1400px}
h2{font-size:16px;border-bottom:1px solid var(--border);padding-bottom:8px;margin-top:36px}
.stats-row{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}
.stat-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 18px;min-width:200px}
.stat-card .n{font-size:22px;font-weight:700}
.stat-card .l{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden;margin:12px 0}
th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
th{background:var(--panel2);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
tr:hover td{background:#1c2029}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.center{text-align:center}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.badge.det{background:#123321;color:var(--det)}
.badge.rubric{background:#2a230f;color:var(--nonfail)}
.badge.rating{background:#2a0f11;color:var(--fail)}
.badge.informed{background:#12203a;color:var(--accent)}
.badge.other{background:#22252e;color:var(--muted)}
.yes{color:var(--det);font-weight:700}
.no{color:var(--muted)}
.fail-n{color:var(--fail);font-weight:700}
.nonfail-n{color:var(--nonfail);font-weight:700}
.code{font-size:12px;color:var(--muted)}
.notes{font-size:12px;color:var(--muted);max-width:340px}
footer{padding:24px 32px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);margin-top:24px}
"""


def build(out_path: Path) -> None:
    stats = collect_stats()

    total_tasks = sum(s["n_tasks"] for s in stats.values())
    total_cost = sum(s["cost"] for s in stats.values())

    stat_cards = ""
    for label, s in stats.items():
        v = ", ".join(f"{k}: {n}" for k, n in sorted(s["verdicts"].items()))
        stat_cards += (
            f"<div class='stat-card'><div class='n'>{s['n_tasks']}</div>"
            f"<div class='l'>{_e(label)}<br>{_e(v)}<br>${s['cost']:.2f} spend</div></div>"
        )
    stat_cards += (
        f"<div class='stat-card'><div class='n'>{total_tasks}</div>"
        f"<div class='l'>Total tasks audited<br>${total_cost:.2f} total spend</div></div>"
    )

    rows = ""
    for cid in ORDER:
        spec = REGISTRY[cid]
        codes = ERROR_CODES.get(cid, {})
        stage = stage_of(cid)
        block = block_of(cid)
        in_restricted = cid in RESTRICTED_CHECKS
        blocks_task = spec.blocks_task

        count_cells = ""
        for label in RUN_DIRS:
            c = stats[label]["counts"].get(cid, {"fail": 0, "non_fail": 0})
            f_n = c["fail"]
            nf_n = c["non_fail"]
            count_cells += (
                f"<td class='center'>"
                f"<span class='fail-n'>{f_n or ''}</span>"
                f"{' / ' if (f_n or nf_n) else ''}"
                f"<span class='nonfail-n'>{nf_n or ''}</span>"
                f"</td>"
            )

        dim = spec.sub_dimension and f"{spec.dimension} / {spec.sub_dimension}" or spec.dimension
        rows += (
            "<tr>"
            f"<td class='mono center'>{cid}</td>"
            f"<td>{_e(dim)}</td>"
            f"<td>{_e(block)}</td>"
            f"<td><span class='badge {stage if stage in ('deterministic','rubric','rating','informed') else 'other'}'>{_e(stage)}</span></td>"
            f"<td class='center'>{'<span class=yes>&#10003;</span>' if in_restricted else '<span class=no>&mdash;</span>'}</td>"
            f"<td class='center'>{'<span class=yes>&#10003;</span>' if blocks_task else '<span class=no>&mdash;</span>'}</td>"
            f"<td>{_e(spec.summary)}</td>"
            f"<td class='code'>{_e(codes.get('fail', '&mdash;'))}</td>"
            f"<td class='code'>{_e(codes.get('non_fail', '&mdash;'))}</td>"
            f"{count_cells}"
            f"<td class='notes'>{_e(spec.notes[:280] + ('…' if len(spec.notes) > 280 else ''))}</td>"
            "</tr>"
        )

    count_headers = "".join(f"<th>{_e(label)}<br>fail / non-fail</th>" for label in RUN_DIRS)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Honeybee QC -- Check Registry &amp; Live Stats</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Honeybee QC -- Check Registry &amp; Live Stats</h1>
  <p>All 25 implemented checks (definition, bands, error codes) plus fail/non-fail counts aggregated
  across every task audited so far. "In L0 restricted scope" marks the 7 checks the L0 overnight loop
  actually grades (270, 280, 310, 400, 450, 460, 470) -- everything else runs full-mode only (L1, or
  L0 tasks audited outside the restricted loop).</p>
</header>
<main>
  <h2>Coverage so far</h2>
  <div class="stats-row">{stat_cards}</div>

  <h2>Check registry</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Dimension</th><th>Block</th><th>Stage</th>
      <th>L0 restricted?</th><th>Blocks task?</th><th>What it measures</th>
      <th>Fail code</th><th>Non-fail code</th>
      {count_headers}
      <th>Notes</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>Reading the numbers</h2>
  <p style="color:var(--muted);max-width:900px">
  Each check cell shows <span class="fail-n">fail count</span> / <span class="nonfail-n">non-fail count</span>
  across all tasks audited in that project so far. A blank cell means the check never fired in that
  band for that project -- either it's genuinely clean there, or (for L0) it's outside the restricted
  scope and abstains as not_evaluated instead of a true clean. Stage badges: <span class="badge det">deterministic</span>
  needs no model call; <span class="badge rubric">rubric</span> grades how the rubric itself was authored;
  <span class="badge rating">rating</span> grades the contributor's per-criterion/per-dimension/ranking
  judgments against an independent re-rating; <span class="badge informed">informed</span> covers
  everything read straight off the prompt/response/artifacts with no independent re-judgment needed.
  </p>
</main>
<footer>Generated from the honeybee_qc registry (registry.py, errors.py, score_against_qc.py) and every
single_*.json report on disk under honeybee_qc/audit_runs/ at build time.</footer>
</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    print(f"wrote {out_path}: {len(ORDER)} checks, {total_tasks} tasks aggregated, ${total_cost:.2f} total spend")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
