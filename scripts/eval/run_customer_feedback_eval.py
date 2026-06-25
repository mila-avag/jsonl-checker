#!/usr/bin/env python3
"""Run tara eval on customer-flagged rubric/test items from two JSONL deliveries.

Produces:
  - <output_dir>/item_results.csv   — per-item tara verdicts + customer feedback agree/disagree + reasoning
  - <output_dir>/task_summary.csv   — per-task counts: total rubrics/tests, flagged, tara_agree, tara_disagree

Usage:
    python run_customer_feedback_eval.py [--workers 40] [--output_dir data/outputs/customer_eval]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

csv.field_size_limit(10_000_000)

BASE = Path(__file__).parent

JSONL_FILES = [
    Path.home() / "Downloads" / "delivery_sender_preview_698318a45989d90bf44b9b53_Test_2026_06_12_00_18_PDT_Kl36FYn_multipart_Test_2026_06_12_00_18_PDT_Kl36FYn.jsonl",
    Path.home() / "Downloads" / "delivery_sender_preview_698318a45989d90bf44b9b53_Test_2026_06_12_00_21_PDT_DHY3CWR_multipart_Test_2026_06_12_00_21_PDT_DHY3CWR.jsonl",
]

FEEDBACK_CSVS = [
    Path.home() / "Downloads" / "Copy of Scale AI RL task feedback - 0518 delivery.csv",
    Path.home() / "Downloads" / "Copy of Scale AI RL task feedback - 0527 delivery.csv",
]

EVAL_SCRIPT = BASE.parent.parent / "apps" / "tara_eval_open_claw_rl_main_2" / "run_threshold_eval.py"

WHITELIST_JSON  = BASE / "customer_feedback_whitelist.json"
DETAIL_JSON     = BASE / "customer_feedback_detail.json"
COMBINED_JSONL  = BASE / "combined_for_customer_eval.jsonl"


REMAP_LOG_JSON     = BASE / "customer_feedback_remap_log.json"
VERSION_DRIFT_JSON = BASE / "customer_feedback_version_drift.json"

# Similarity thresholds for matching a customer-flagged criterion to a JSONL rubric.
SAME_ID_OK     = 0.60   # criterion at the customer's stated ID matches well enough -> keep
REMAP_MIN      = 0.70   # a different JSONL id matches at least this well -> consider remap
REMAP_MARGIN   = 0.15   # ...and beats the same-id score by this margin -> remap


def _normalize_criterion(s: str) -> str:
    """Lowercase, strip markdown formatting/links/punctuation for fuzzy comparison."""
    import re
    s = s.lower()
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)   # [text](url) -> text
    s = re.sub(r'[`*_#\\]', '', s)                    # markdown emphasis / escapes
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_whitelist_and_feedback():
    """Re-parse feedback CSVs and regenerate whitelist + detail JSON.

    Fixes applied vs the original buggy version:
      1. Per-item rationale: each flagged item gets ITS OWN criterion + rationale
         block from problematic_rubrics/problematic_tests — not the whole blob and
         not the generic task-level rubric_quality_rationale.
      2. ID remapping by criterion text: the customer's rubric numbering does not
         always match the JSONL numbering (partial off-by-one / version drift).
         We match the customer's flagged criterion TEXT to the JSONL criterion and
         use the best-matching JSONL id, so the verifier evaluates the criterion the
         customer actually flagged. Remaps and unresolved (version-drift) items are
         logged for audit.
    """
    import re
    from difflib import SequenceMatcher

    # Load all tasks (full objects) from both JSONLs keyed by task_id
    jsonl_tasks: dict[str, dict] = {}
    for path in JSONL_FILES:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                jsonl_tasks[t["task_id"]] = t

    def sim(a: str, b: str) -> float:
        return SequenceMatcher(None, _normalize_criterion(a), _normalize_criterion(b)).ratio()

    RUB_HEAD  = re.compile(r'^(R\d+)\s*\(([a-z][a-z0-9_]*)\)\s*:\s*(.*)$')
    TEST_HEAD = re.compile(r'^(test_[a-zA-Z0-9_]+)\s*\(([a-z][a-z0-9_]*)\)\s*$')
    # Older sheets store item-level feedback in the pass-rate columns instead of
    # problematic_rubrics/problematic_tests. Rubric rows have criterion text first;
    # pytest rows have the test function name first.
    OLD_RUB_HEAD = re.compile(r'^(.*?)\s*\(weight:\s*[^)]*?\bissue:\s*([A-Z_]+)\)\s*$', re.I)
    OLD_TEST_HEAD = re.compile(r'^(test_[A-Za-z0-9_]+)\s*\(weight:\s*[^)]*?\bissue:\s*([A-Z_]+)\)\s*$', re.I)

    def parse_blocks(text: str, kind: str) -> list[dict]:
        """Split on --- and parse each block's id, issue, criterion text, rationale.

        Only the FIRST line of a block is matched against the header pattern, so
        incidental R#/test_name mentions inside Rationale text are not treated as
        new flagged items.
        """
        out = []
        head_re = RUB_HEAD if kind == "rubric" else TEST_HEAD
        for block in text.strip().split("\n---\n"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            m = head_re.match(lines[0].strip())
            if not m:
                continue
            orig_id   = m.group(1)
            issue     = m.group(2).strip()
            crit_text = (m.group(3).strip() if kind == "rubric" else "")
            rationale = "\n".join(lines[1:]).strip()
            rationale = re.sub(r'^Rationale:\s*', '', rationale).strip()
            out.append({
                "orig_id":   orig_id,
                "issue":     issue,
                "criterion": crit_text,
                "rationale": rationale,
            })
        return out

    def parse_pass_rate_blocks(text: str, kind: str) -> list[dict]:
        """Parse older pass-rate issue columns into the same block shape.

        Examples:
          Rubric criterion text. (weight: 5.0, issue: RUBRIC_DESIGN)
            Evidence: ...
          test_name (weight: 4.0, issue: PYTEST_INCORRECT_ASSERTION)
            Evidence: ...
        """
        out = []
        head_re = OLD_RUB_HEAD if kind == "rubric" else OLD_TEST_HEAD
        for block in re.split(r"\n\s*---\s*\n", (text or "").strip()):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            header = re.sub(r"^\[[^\]]+\]\s*", "", lines[0].strip())
            m = head_re.match(header)
            if not m:
                continue
            issue = m.group(2).lower()
            issue = re.sub(r"^(rubric|pytest|env)_", "", issue)
            rationale = "\n".join(lines[1:]).strip()
            rationale = re.sub(r"^Evidence:\s*", "", rationale, flags=re.I).strip()
            if kind == "rubric":
                out.append({
                    "orig_id": "",
                    "issue": issue,
                    "criterion": m.group(1).strip(),
                    "rationale": rationale,
                })
            else:
                out.append({
                    "orig_id": m.group(1).strip(),
                    "issue": issue,
                    "criterion": "",
                    "rationale": rationale,
                })
        return out

    feedback: dict[str, dict] = {}
    remap_log: list[dict] = []
    version_drift: list[dict] = []

    for csv_path in FEEDBACK_CSVS:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("task_id", "").strip()
                if not tid or tid not in jsonl_tasks:
                    continue

                task = jsonl_tasks[tid]
                rubrics = task.get("rubrics", {})
                rub_crit = {}
                if isinstance(rubrics, dict):
                    for rid, rv in rubrics.items():
                        rub_crit[rid] = rv.get("criterion", "") if isinstance(rv, dict) else str(rv)
                test_names = {t["test_name"] for t in task.get("unit_tests", [])}

                items: dict[str, dict] = {}

                # ── Rubrics: remap by criterion text ──────────────────────────
                rubric_blocks = parse_blocks(row.get("problematic_rubrics", ""), "rubric")
                rubric_blocks += parse_pass_rate_blocks(row.get("rubric_issues_based_on_pass_rate", ""), "rubric")
                for blk in rubric_blocks:
                    orig_id = blk["orig_id"]
                    ctext   = blk["criterion"]

                    same_s = sim(ctext, rub_crit.get(orig_id, "")) if orig_id and orig_id in rub_crit else 0.0
                    best_id, best_s = None, 0.0
                    for rid, crit in rub_crit.items():
                        s = sim(ctext, crit)
                        if s > best_s:
                            best_s, best_id = s, rid

                    if orig_id and orig_id in rub_crit and same_s >= SAME_ID_OK:
                        use_id, status = orig_id, "aligned"
                    elif best_id and best_s >= REMAP_MIN and (best_s - same_s) >= REMAP_MARGIN:
                        use_id, status = best_id, "remapped"
                        remap_log.append({
                            "task_id": tid, "customer_id": orig_id, "mapped_to": best_id,
                            "same_id_sim": round(same_s, 2), "best_sim": round(best_s, 2),
                            "customer_criterion": ctext[:120],
                            "jsonl_criterion": rub_crit.get(best_id, "")[:120],
                        })
                    else:
                        # No confident match anywhere — likely a different rubric
                        # VERSION than what the customer reviewed. Keep the customer's
                        # id so we still evaluate something, but flag for manual review.
                        if not orig_id:
                            continue
                        use_id, status = orig_id, "version_drift"
                        version_drift.append({
                            "task_id": tid, "customer_id": orig_id,
                            "best_guess_id": best_id, "best_sim": round(best_s, 2),
                            "customer_criterion": ctext[:120],
                            "jsonl_at_customer_id": rub_crit.get(orig_id, "")[:120],
                        })

                    # Per-item rationale = customer's own criterion + rationale text.
                    rat = ctext
                    if blk["rationale"]:
                        rat = f"{ctext}\n  Rationale: {blk['rationale']}" if ctext else blk["rationale"]
                    items[use_id] = {
                        "item_type":               "rubric",
                        "issue_type":              blk["issue"],
                        "customer_rationale":      rat,
                        "orig_customer_id":        orig_id,
                        "match_status":            status,
                    }

                # ── Unit tests: matched by name, per-item rationale ───────────
                test_blocks = parse_blocks(row.get("problematic_tests", ""), "unit_test")
                test_blocks += parse_pass_rate_blocks(row.get("pytest_issues_based_on_pass_rate", ""), "unit_test")
                for blk in test_blocks:
                    tname = blk["orig_id"]
                    items[tname] = {
                        "item_type":          "unit_test",
                        "issue_type":         blk["issue"],
                        "customer_rationale": blk["rationale"] or tname,
                        "orig_customer_id":   tname,
                        "match_status":       ("aligned" if tname in test_names else "name_not_in_jsonl"),
                    }

                if items:
                    feedback.setdefault(tid, {}).update(items)

    whitelist = {tid: list(items.keys()) for tid, items in feedback.items()}

    with open(WHITELIST_JSON, "w") as f:
        json.dump(whitelist, f, indent=2)
    with open(DETAIL_JSON, "w") as f:
        json.dump(feedback, f, indent=2)
    with open(REMAP_LOG_JSON, "w") as f:
        json.dump(remap_log, f, indent=2)
    with open(VERSION_DRIFT_JSON, "w") as f:
        json.dump(version_drift, f, indent=2)

    total_items = sum(len(v) for v in whitelist.values())
    print(f"Whitelist: {len(whitelist)} tasks, {total_items} flagged items")
    print(f"  ID remaps (customer id -> matched JSONL id): {len(remap_log)}  (see {REMAP_LOG_JSON.name})")
    print(f"  Version-drift items (no confident match; flagged for review): {len(version_drift)}  (see {VERSION_DRIFT_JSON.name})")
    return whitelist, feedback


def build_combined_jsonl(whitelist: dict):
    """Merge both JSONLs, keeping only tasks present in whitelist."""
    seen: set[str] = set()
    count = 0
    with open(COMBINED_JSONL, "w") as out:
        for path in JSONL_FILES:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    tid = t["task_id"]
                    if tid in whitelist and tid not in seen:
                        out.write(json.dumps(t) + "\n")
                        seen.add(tid)
                        count += 1
    print(f"Combined JSONL: {count} tasks written to {COMBINED_JSONL}")


def run_eval(output_dir: str, workers: int, dl_workers: int, disable_cache: bool):
    """Invoke run_threshold_eval.py with whitelist + customer feedback args."""
    cache_folder = os.path.abspath(os.path.join(output_dir, "cache"))
    os.makedirs(cache_folder, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(EVAL_SCRIPT),
        "--jsonl",                   str(COMBINED_JSONL.resolve()),
        "--cache_folder",            cache_folder,
        "--output",                  "item_results_raw.csv",
        "--workers",                 str(workers),
        "--dl_workers",              str(dl_workers),
        "--item_whitelist",          str(WHITELIST_JSON.resolve()),
        "--customer_feedback_json",  str(DETAIL_JSON.resolve()),
    ]
    if disable_cache:
        cmd.append("--disable_cache")

    print("\n" + "=" * 70)
    print("Running tara eval on customer-flagged items …")
    print(" ".join(cmd))
    print("=" * 70 + "\n")

    proc = subprocess.run(cmd, cwd=str(EVAL_SCRIPT.parent))
    if proc.returncode != 0:
        print(f"[ERROR] run_threshold_eval.py exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    raw_csv = os.path.join(cache_folder, "item_results_raw.csv")
    return raw_csv


def build_summary_csvs(raw_csv: str, output_dir: str, feedback: dict):
    """Post-process raw eval CSV into item-level + task-level summary CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    # Customer-feedback stats only concern real verifiers (rubric / unit_test);
    # task-level check rows (coverage/intersection/redundancy) are handled separately.
    item_rows = [r for r in all_rows if r.get("item_type") in ("rubric", "unit_test")]

    # ── Item-level CSV ────────────────────────────────────────────────────────
    item_fieldnames = [
        "task_id", "item_type", "item_id", "criterion",
        "pass_rate_opus", "pass_rate_gemini", "avg_pass_rate",
        "verifier_verdict", "confidence", "reasoning",
        "customer_issue_type", "customer_rationale",
        "customer_feedback_verdict", "customer_feedback_reasoning",
        "error",
    ]
    item_csv_path = os.path.join(output_dir, "item_results.csv")
    with open(item_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=item_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(item_rows, key=lambda r: (r["task_id"], r["item_type"], r["item_id"])):
            writer.writerow(row)
    print(f"Item-level CSV → {item_csv_path}  ({len(item_rows)} rows)")

    # ── Task-level summary ────────────────────────────────────────────────────
    task_stats: dict[str, dict] = {}

    # Seed with whitelist totals (even tasks whose env failed to load)
    for tid, items in feedback.items():
        task_stats[tid] = {
            "task_id":             tid,
            "total_flagged":       len(items),
            "flagged_rubrics":     sum(1 for v in items.values() if v["item_type"] == "rubric"),
            "flagged_tests":       sum(1 for v in items.values() if v["item_type"] == "unit_test"),
            "eval_ran":            0,
            "tara_agree":          0,
            "tara_partially_agree": 0,
            "tara_disagree":       0,
            "tara_flagged":        0,
            "tara_correct":        0,
            "tara_error":          0,
            "agree_reasoning":     [],
            "disagree_reasoning":  [],
        }

    for row in item_rows:
        tid = row["task_id"]
        if tid not in task_stats:
            continue
        s = task_stats[tid]
        s["eval_ran"] += 1

        if row.get("error"):
            s["tara_error"] += 1
            continue

        vv = row.get("verifier_verdict", "")
        if vv == "flagged":
            s["tara_flagged"] += 1
        elif vv == "correct":
            s["tara_correct"] += 1

        cfv = row.get("customer_feedback_verdict", "")
        cfr = row.get("customer_feedback_reasoning", "").strip()
        iid = row.get("item_id", "")

        if cfv == "agree":
            s["tara_agree"] += 1
            if cfr:
                s["agree_reasoning"].append(f"{iid}: {cfr}")
        elif cfv == "partially_agree":
            s["tara_partially_agree"] += 1
            if cfr:
                s["agree_reasoning"].append(f"{iid} (partial): {cfr}")
        elif cfv == "disagree":
            s["tara_disagree"] += 1
            if cfr:
                s["disagree_reasoning"].append(f"{iid}: {cfr}")

    task_fieldnames = [
        "task_id",
        "total_flagged", "flagged_rubrics", "flagged_tests",
        "eval_ran", "tara_flagged", "tara_correct", "tara_error",
        "tara_agree", "tara_partially_agree", "tara_disagree",
        "valid_customer_flags",   # agree + partially_agree
        "agree_reasoning_summary",
        "disagree_reasoning_summary",
    ]
    task_csv_path = os.path.join(output_dir, "task_summary.csv")
    with open(task_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=task_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for tid in sorted(task_stats):
            s = task_stats[tid]
            s["valid_customer_flags"] = s["tara_agree"] + s["tara_partially_agree"]
            s["agree_reasoning_summary"]    = " | ".join(s["agree_reasoning"])
            s["disagree_reasoning_summary"] = " | ".join(s["disagree_reasoning"])
            writer.writerow(s)

    total_agree    = sum(s["tara_agree"]    for s in task_stats.values())
    total_partial  = sum(s["tara_partially_agree"] for s in task_stats.values())
    total_disagree = sum(s["tara_disagree"] for s in task_stats.values())
    total_errors   = sum(s["tara_error"]    for s in task_stats.values())
    evaled         = sum(s["eval_ran"]      for s in task_stats.values())

    print(f"Task summary CSV → {task_csv_path}  ({len(task_stats)} tasks)")
    print(textwrap.dedent(f"""
    ┌──────────────────────────────────────────────┐
    │  Customer Feedback Evaluation Summary         │
    ├──────────────────────────────────────────────┤
    │  Items evaluated:      {evaled:>6}              │
    │  Tara AGREES:          {total_agree:>6}  (valid flags) │
    │  Tara PARTIALLY agrees:{total_partial:>6}              │
    │  Tara DISAGREES:       {total_disagree:>6}              │
    │  Errors:               {total_errors:>6}              │
    └──────────────────────────────────────────────┘
    """))


def load_task_totals():
    """Per-task total verifier counts (rubrics + unit tests) from the combined JSONL."""
    totals: dict[str, dict] = {}
    if not COMBINED_JSONL.exists():
        return totals
    with open(COMBINED_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            n_rub = len(t.get("rubrics", {}) or {})
            n_test = len(t.get("unit_tests", []) or [])
            totals[t["task_id"]] = {
                "n_rubrics": n_rub,
                "n_tests": n_test,
                "total_verifiers": n_rub + n_test,
            }
    return totals


# Severity thresholds for the per-task quality gates.
MAJOR_GATE = 0.10   # at most 10% of verifiers may have major issues
MINOR_GATE = 0.15   # at most 15% of verifiers may have minor issues


def _safe_json(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def build_quality_outputs(raw_csv: str, output_dir: str, totals: dict):
    """Per-task & overall verifier-quality scoring against the major/minor taxonomy.

    - % major / % minor use TOTAL verifiers in the task (from JSONL) as denominator.
    - major/minor numerators come from quality_issues on the flagged items evaluated.
    - missing-critical / non-critical come from the coverage check (outcome-only).
    - intersection & redundancy gates come from their task-level checks.
    """
    from collections import Counter
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Seed per-task records from JSONL totals.
    tasks: dict[str, dict] = {}
    for tid, tot in totals.items():
        tasks[tid] = {
            "task_id": tid,
            "n_rubrics": tot["n_rubrics"],
            "n_tests": tot["n_tests"],
            "total_verifiers": tot["total_verifiers"],
            "flagged_evaluated": 0,
            "major_verifiers": 0,
            "minor_verifiers": 0,
            "missing_critical": 0,
            "missing_non_critical": 0,
            "intersection_overlap": None,
            "intersection_pass": None,
            "redundancy_groups": None,
            "redundancy_pass": None,
            "single_gtfa_pass": None,
            "single_gtfa_failure_mode": "",
            "quoted_rubric_pass": None,
            "quoted_offenders": 0,
            "_major_cats": Counter(),
            "_minor_cats": Counter(),
        }

    quality_item_rows = []   # per-issue rows for the detailed item CSV
    resolution_rows = []     # one row per verifier the model wants edited/deleted
    quoted_edit_candidates = []   # (task_id, rubric_id, criterion, comment) from quoted_rubric_check
    rubric_text_lookup = {}       # (task_id, item_id) -> criterion text (fallback for quoted rows)
    major_cat_total = Counter()
    minor_cat_total = Counter()

    for r in rows:
        tid = r.get("task_id", "")
        it = r.get("item_type", "")
        s = tasks.get(tid)
        if s is None:
            continue

        if it == "rubric":
            crit = (r.get("criterion", "") or "").strip()
            if crit:
                rubric_text_lookup[(tid, r.get("item_id", ""))] = crit

        if it in ("rubric", "unit_test"):
            qissues = _safe_json(r.get("quality_issues", "")) or []
            if not isinstance(qissues, list):
                qissues = []
            s["flagged_evaluated"] += 1
            has_major = any(q.get("severity") == "major" for q in qissues)
            has_minor = any(q.get("severity") == "minor" for q in qissues)
            if has_major:
                s["major_verifiers"] += 1
            if has_minor:
                s["minor_verifiers"] += 1
            resolution_action = (r.get("resolution_action", "") or "").strip()
            resolution_comment = (r.get("resolution_comment", "") or "").strip()
            for q in qissues:
                sev = q.get("severity", "")
                cat = q.get("category", "")
                if sev == "major":
                    s["_major_cats"][cat] += 1
                    major_cat_total[cat] += 1
                elif sev == "minor":
                    s["_minor_cats"][cat] += 1
                    minor_cat_total[cat] += 1
                quality_item_rows.append({
                    "task_id": tid,
                    "item_type": it,
                    "item_id": r.get("item_id", ""),
                    "severity": sev,
                    "category": cat,
                    "description": q.get("description", ""),
                    "evidence": q.get("evidence", ""),
                    "resolution_action": resolution_action,
                    "resolution_comment": resolution_comment,
                })
            # One actionable resolution row per verifier flagged for edit/delete/create.
            if resolution_action in ("edit", "delete", "delete_and_create_rubric"):
                sev_max = "major" if has_major else ("minor" if has_minor else "")
                cats = sorted({q.get("category", "") for q in qissues if q.get("category")})
                resolution_rows.append({
                    "task_id": tid,
                    "item_type": it,
                    "item_id": r.get("item_id", ""),
                    "resolution_action": resolution_action,
                    "severity": sev_max,
                    "categories": ", ".join(cats),
                    "criterion": (r.get("criterion", "") or "").strip(),
                    "resolution_comment": resolution_comment,
                })

        elif it == "coverage_check":
            payload = _safe_json(r.get("check_payload", "")) or {}
            for mc in payload.get("missing_criteria", []) or []:
                if mc.get("criticality") == "critical":
                    s["missing_critical"] += 1
                else:
                    s["missing_non_critical"] += 1

        elif it == "intersection_check":
            payload = _safe_json(r.get("check_payload", "")) or {}
            s["intersection_overlap"] = payload.get("overlap_count")
            s["intersection_pass"] = payload.get("gate_pass")

        elif it == "redundancy_check":
            payload = _safe_json(r.get("check_payload", "")) or {}
            groups = payload.get("redundant_groups", []) or []
            s["redundancy_groups"] = len(groups)
            s["redundancy_pass"] = payload.get("gate_pass")

        elif it == "single_gtfa_check":
            payload = _safe_json(r.get("check_payload", "")) or {}
            verdict = payload.get("verifier_verdict")
            s["single_gtfa_pass"] = (verdict == "single_gtfa") if verdict else None
            s["single_gtfa_failure_mode"] = payload.get("failure_mode", "")

        elif it == "quoted_rubric_check":
            payload = _safe_json(r.get("check_payload", "")) or {}
            verdict = payload.get("verifier_verdict")
            offs = payload.get("offenders", []) or []
            s["quoted_rubric_pass"] = (verdict == "clean") if verdict else None
            s["quoted_offenders"] = len(offs)
            for o in offs:
                rid = (o.get("rubric_id", "") or "").strip()
                crit = (o.get("criterion", "") or "").strip()
                qt = (o.get("quoted_text", "") or "").strip()
                why = (o.get("reason", "") or "").strip()
                comment = (
                    "QUOTED-RUBRIC FIX: this criterion uses literal quoted wording "
                    f"({qt or 'see quotes'}) but the prompt accepts variations. REMOVE every "
                    "single (') and double (\") quote character and the verbatim quoted phrase, "
                    "and restate the requirement so reasonable variations (paraphrase, synonyms, "
                    "reordering, casing/spacing/punctuation) all pass. The rewritten criterion "
                    "MUST NOT contain any ' or \" characters. " + (why and f"Why it varies: {why}")
                )
                quoted_edit_candidates.append((tid, rid, crit, comment))

    # ── Inject quoted-rubric edits into the worklist ──────────────────────────
    # Each flagged quoted rubric becomes an `edit` row instructing the rewrite
    # agent to strip the quotes. Skip any (task_id,item_id) already on the list.
    existing_keys = {(r["task_id"], r["item_id"]) for r in resolution_rows
                     if r["item_type"] == "rubric"}
    n_quoted_added = 0
    for tid, rid, crit, comment in quoted_edit_candidates:
        if not rid or (tid, rid) in existing_keys:
            continue
        existing_keys.add((tid, rid))
        if not crit:
            crit = rubric_text_lookup.get((tid, rid), "")
        resolution_rows.append({
            "task_id": tid,
            "item_type": "rubric",
            "item_id": rid,
            "resolution_action": "edit",
            "severity": "minor",
            "categories": "quoted_overspecific",
            "criterion": crit,
            "resolution_comment": comment,
        })
        n_quoted_added += 1

    # Finalize per-task derived metrics.
    for s in tasks.values():
        tv = s["total_verifiers"] or 0
        s["pct_major"] = (s["major_verifiers"] / tv) if tv else 0.0
        s["pct_minor"] = (s["minor_verifiers"] / tv) if tv else 0.0
        # combined view folds missing criteria into the severity counts
        s["pct_major_combined"] = ((s["major_verifiers"] + s["missing_critical"]) / tv) if tv else 0.0
        s["pct_minor_combined"] = ((s["minor_verifiers"] + s["missing_non_critical"]) / tv) if tv else 0.0
        s["major_gate_pass"] = s["pct_major"] <= MAJOR_GATE
        s["minor_gate_pass"] = s["pct_minor"] <= MINOR_GATE

    # ── Detailed per-issue CSV ────────────────────────────────────────────────
    qi_path = os.path.join(output_dir, "quality_item_results.csv")
    with open(qi_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "item_type", "item_id",
                                          "severity", "category", "description", "evidence",
                                          "resolution_action", "resolution_comment"],
                           extrasaction="ignore")
        w.writeheader()
        for row in sorted(quality_item_rows, key=lambda x: (x["task_id"], x["severity"], x["category"])):
            w.writerow(row)

    # ── Resolution worklist (one row per verifier to edit/delete) ─────────────
    res_path = os.path.join(output_dir, "resolution_results.csv")
    with open(res_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "item_type", "item_id",
                                          "resolution_action", "severity", "categories",
                                          "criterion", "resolution_comment"],
                           extrasaction="ignore")
        w.writeheader()
        action_order = {"delete": 0, "delete_and_create_rubric": 1, "edit": 2}
        for row in sorted(resolution_rows, key=lambda x: (x["task_id"],
                                                          action_order.get(x["resolution_action"], 9),
                                                          x["item_type"], x["item_id"])):
            w.writerow(row)
    n_edit = sum(1 for r in resolution_rows if r["resolution_action"] == "edit")
    n_del = sum(1 for r in resolution_rows if r["resolution_action"] == "delete")
    n_create = sum(1 for r in resolution_rows if r["resolution_action"] == "delete_and_create_rubric")
    print(f"Resolution worklist  → {res_path}  ({n_edit} edit, {n_del} delete, "
          f"{n_create} delete+create-rubric; "
          f"{n_quoted_added} from quoted-rubric check)")

    # ── Per-task summary CSV ──────────────────────────────────────────────────
    qt_fieldnames = [
        "task_id", "n_rubrics", "n_tests", "total_verifiers", "flagged_evaluated",
        "major_verifiers", "pct_major", "major_gate_pass",
        "minor_verifiers", "pct_minor", "minor_gate_pass",
        "missing_critical", "missing_non_critical",
        "pct_major_combined", "pct_minor_combined",
        "intersection_overlap", "intersection_pass",
        "redundancy_groups", "redundancy_pass",
        "single_gtfa_pass", "single_gtfa_failure_mode",
        "quoted_rubric_pass", "quoted_offenders",
    ]
    qt_path = os.path.join(output_dir, "quality_task_summary.csv")
    with open(qt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qt_fieldnames, extrasaction="ignore")
        w.writeheader()
        for tid in sorted(tasks):
            s = dict(tasks[tid])
            s["pct_major"] = round(s["pct_major"], 4)
            s["pct_minor"] = round(s["pct_minor"], 4)
            s["pct_major_combined"] = round(s["pct_major_combined"], 4)
            s["pct_minor_combined"] = round(s["pct_minor_combined"], 4)
            w.writerow(s)

    # ── HTML report ───────────────────────────────────────────────────────────
    html_path = os.path.join(output_dir, "quality_report.html")
    _write_quality_html(html_path, tasks, major_cat_total, minor_cat_total)

    # ── Console summary ───────────────────────────────────────────────────────
    n_tasks = len(tasks)
    tot_verifiers = sum(s["total_verifiers"] for s in tasks.values())
    tot_major = sum(s["major_verifiers"] for s in tasks.values())
    tot_minor = sum(s["minor_verifiers"] for s in tasks.values())
    fail_major = sum(1 for s in tasks.values() if not s["major_gate_pass"])
    fail_minor = sum(1 for s in tasks.values() if not s["minor_gate_pass"])
    fail_inter = sum(1 for s in tasks.values() if s["intersection_pass"] is False)
    fail_redun = sum(1 for s in tasks.values() if s["redundancy_pass"] is False)
    tasks_missing_crit = sum(1 for s in tasks.values() if s["missing_critical"] > 0)
    fail_gtfa = sum(1 for s in tasks.values() if s["single_gtfa_pass"] is False)
    fail_quoted = sum(1 for s in tasks.values() if s["quoted_rubric_pass"] is False)
    tot_quoted_off = sum(s["quoted_offenders"] for s in tasks.values())

    print(f"\nQuality item CSV    → {qi_path}  ({len(quality_item_rows)} issues)")
    print(f"Quality task CSV    → {qt_path}  ({n_tasks} tasks)")
    print(f"Quality HTML report → {html_path}")
    print(textwrap.dedent(f"""
    ┌────────────────────────────────────────────────────────────┐
    │  Verifier Quality Summary ({n_tasks} tasks, {tot_verifiers} verifiers)         
    ├────────────────────────────────────────────────────────────┤
    │  Overall MAJOR issue rate:  {tot_major}/{tot_verifiers} = {(tot_major/tot_verifiers*100 if tot_verifiers else 0):.1f}%
    │  Overall MINOR issue rate:  {tot_minor}/{tot_verifiers} = {(tot_minor/tot_verifiers*100 if tot_verifiers else 0):.1f}%
    ├────────────────────────────────────────────────────────────┤
    │  Tasks FAILING major gate (>10%):   {fail_major}
    │  Tasks FAILING minor gate (>15%):   {fail_minor}
    │  Tasks FAILING intersection (>1):   {fail_inter}
    │  Tasks FAILING redundancy (>2):     {fail_redun}
    │  Tasks WITHOUT a single GTFA:       {fail_gtfa}
    │  Tasks w/ QUOTED rubrics to fix:    {fail_quoted}  ({tot_quoted_off} rubrics)
    │  Tasks with missing CRITICAL crit.: {tasks_missing_crit}
    └────────────────────────────────────────────────────────────┘
    """))


def _write_quality_html(path, tasks, major_cat_total, minor_cat_total):
    import html as _html

    n_tasks = len(tasks)
    tot_verifiers = sum(s["total_verifiers"] for s in tasks.values())
    tot_major = sum(s["major_verifiers"] for s in tasks.values())
    tot_minor = sum(s["minor_verifiers"] for s in tasks.values())
    tot_miss_crit = sum(s["missing_critical"] for s in tasks.values())
    tot_miss_noncrit = sum(s["missing_non_critical"] for s in tasks.values())
    fail_major = sum(1 for s in tasks.values() if not s["major_gate_pass"])
    fail_minor = sum(1 for s in tasks.values() if not s["minor_gate_pass"])
    fail_inter = sum(1 for s in tasks.values() if s["intersection_pass"] is False)
    fail_redun = sum(1 for s in tasks.values() if s["redundancy_pass"] is False)

    pct_major = (tot_major / tot_verifiers * 100) if tot_verifiers else 0
    pct_minor = (tot_minor / tot_verifiers * 100) if tot_verifiers else 0

    CAT_LABEL = {
        "not_self_contained": "Not self-contained",
        "inaccurate_misaligned": "Inaccurate: misaligned with prompt",
        "inaccurate_factual_error": "Inaccurate: factual error / misleading",
        "inaccurate_makes_response_worse": "Inaccurate: not required & makes response worse",
        "inaccurate_unrelated": "Inaccurate: unrelated to prompt",
        "incorrect_justification": "Incorrect / weak justification",
        "overfitted": "Overfitted / rejects valid implementations",
        "overlapping": "Overlapping / double-counted verifier",
        "brittle": "Brittle unit test",
        "underfitted": "Underfitted (too lenient / passes invalid output)",
        "too_prescriptive": "Too prescriptive",
        "subjective": "Subjective / vague / immeasurable",
    }

    def badge(ok, label_pass, label_fail):
        if ok is None:
            return '<span class="b b-na">n/a</span>'
        return (f'<span class="b b-pass">{label_pass}</span>' if ok
                else f'<span class="b b-fail">{label_fail}</span>')

    def cat_rows(counter):
        if not counter:
            return '<tr><td colspan="2" class="muted">none</td></tr>'
        return "".join(
            f"<tr><td>{_html.escape(CAT_LABEL.get(c, c))}</td><td class='num'>{n}</td></tr>"
            for c, n in counter.most_common()
        )

    # per-task rows, worst first (by combined major then minor)
    ordered = sorted(tasks.values(),
                     key=lambda s: (s["pct_major_combined"], s["pct_minor_combined"]),
                     reverse=True)
    trows = []
    for s in ordered:
        maj_cls = "fail" if not s["major_gate_pass"] else ""
        min_cls = "fail" if not s["minor_gate_pass"] else ""
        trows.append(f"""<tr>
  <td><code>{s['task_id']}</code></td>
  <td class="num">{s['total_verifiers']}</td>
  <td class="num">{s['flagged_evaluated']}</td>
  <td class="num {maj_cls}">{s['major_verifiers']} ({s['pct_major']*100:.1f}%)</td>
  <td>{badge(s['major_gate_pass'], 'PASS', 'FAIL >10%')}</td>
  <td class="num {min_cls}">{s['minor_verifiers']} ({s['pct_minor']*100:.1f}%)</td>
  <td>{badge(s['minor_gate_pass'], 'PASS', 'FAIL >15%')}</td>
  <td class="num">{s['missing_critical']}/{s['missing_non_critical']}</td>
  <td class="num">{'' if s['intersection_overlap'] is None else s['intersection_overlap']}</td>
  <td>{badge(s['intersection_pass'], 'PASS', 'FAIL >1')}</td>
  <td class="num">{'' if s['redundancy_groups'] is None else s['redundancy_groups']}</td>
  <td>{badge(s['redundancy_pass'], 'PASS', 'FAIL >2')}</td>
</tr>""")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Verifier Quality Report</title>
<style>
body{{font-family:Arial,sans-serif;font-size:12px;color:#202124;margin:0;padding:24px 40px;line-height:1.5}}
h1{{font-size:22px;color:#1a73e8;margin-bottom:2px}}
h2{{font-size:15px;color:#1a73e8;margin-top:26px;border-bottom:1px solid #e0e0e0;padding-bottom:4px}}
.meta{{color:#5f6368;font-size:11px;margin-bottom:14px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
.card{{border:1px solid #dadce0;border-radius:10px;padding:12px 16px;min-width:150px;background:#fafafa}}
.card .big{{font-size:24px;font-weight:700}}
.card .lbl{{color:#5f6368;font-size:11px}}
.ok{{color:#188038}} .bad{{color:#c5221f}}
table{{border-collapse:collapse;width:100%;margin:6px 0 14px;font-size:11.5px}}
th{{background:#e8f0fe;color:#1a73e8;text-align:left;padding:7px 10px;border:1px solid #c5cae9;position:sticky;top:0}}
td{{padding:6px 10px;border:1px solid #e0e0e0;vertical-align:top}}
td.num{{text-align:right;white-space:nowrap}}
td.fail{{background:#fce8e6;color:#c5221f;font-weight:700}}
code{{background:#f1f3f4;padding:1px 5px;border-radius:4px;font-size:11px}}
.muted{{color:#9aa0a6}}
.b{{padding:2px 7px;border-radius:9px;font-size:10px;font-weight:700;white-space:nowrap}}
.b-pass{{background:#e6f4ea;color:#188038;border:1px solid #b7e1c2}}
.b-fail{{background:#fce8e6;color:#c5221f;border:1px solid #f5b5b0}}
.b-na{{background:#f1f3f4;color:#9aa0a6;border:1px solid #dadce0}}
.half{{display:inline-block;vertical-align:top;width:48%;margin-right:1%}}
</style></head><body>

<h1>Verifier Quality Report</h1>
<div class="meta">{n_tasks} tasks &middot; {tot_verifiers} total verifiers &middot; gates: major &le;10%, minor &le;15%, rubric/pytest intersection &le;1, redundancy groups (&gt;2 verifiers) = 0</div>

<div class="cards">
  <div class="card"><div class="big {'bad' if pct_major>10 else 'ok'}">{pct_major:.1f}%</div><div class="lbl">Overall MAJOR issue rate<br>({tot_major}/{tot_verifiers})</div></div>
  <div class="card"><div class="big {'bad' if pct_minor>15 else 'ok'}">{pct_minor:.1f}%</div><div class="lbl">Overall MINOR issue rate<br>({tot_minor}/{tot_verifiers})</div></div>
  <div class="card"><div class="big {'bad' if fail_major else 'ok'}">{fail_major}</div><div class="lbl">Tasks failing<br>major gate (&gt;10%)</div></div>
  <div class="card"><div class="big {'bad' if fail_minor else 'ok'}">{fail_minor}</div><div class="lbl">Tasks failing<br>minor gate (&gt;15%)</div></div>
  <div class="card"><div class="big {'bad' if fail_inter else 'ok'}">{fail_inter}</div><div class="lbl">Tasks failing<br>intersection (&gt;1)</div></div>
  <div class="card"><div class="big {'bad' if fail_redun else 'ok'}">{fail_redun}</div><div class="lbl">Tasks failing<br>redundancy (&gt;2)</div></div>
  <div class="card"><div class="big">{tot_miss_crit}/{tot_miss_noncrit}</div><div class="lbl">Missing criteria<br>critical / non-critical</div></div>
</div>

<h2>Issue category breakdown</h2>
<div class="half">
  <table><tr><th>Major category</th><th class="num">count</th></tr>{cat_rows(major_cat_total)}</table>
</div>
<div class="half">
  <table><tr><th>Minor category</th><th class="num">count</th></tr>{cat_rows(minor_cat_total)}</table>
</div>

<h2>Per-task quality scores (worst first)</h2>
<table>
<tr>
  <th>Task</th><th class="num">Total<br>verifiers</th><th class="num">Flagged<br>eval'd</th>
  <th class="num">Major</th><th>Major gate</th>
  <th class="num">Minor</th><th>Minor gate</th>
  <th class="num">Missing<br>crit/non</th>
  <th class="num">Inter<br>overlap</th><th>Inter gate</th>
  <th class="num">Redun<br>groups</th><th>Redun gate</th>
</tr>
{''.join(trows)}
</table>

</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    global JSONL_FILES, FEEDBACK_CSVS, EVAL_SCRIPT
    global WHITELIST_JSON, DETAIL_JSON, COMBINED_JSONL, REMAP_LOG_JSON, VERSION_DRIFT_JSON

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir",     default="data/outputs/customer_eval")
    parser.add_argument("--workers",        type=int, default=40)
    parser.add_argument("--dl_workers",     type=int, default=10)
    parser.add_argument("--jsonl", type=Path, action="append", default=None,
                        help="Delivery JSONL to evaluate. Repeatable. Defaults to the legacy hardcoded files.")
    parser.add_argument("--feedback_csv", type=Path, action="append", default=None,
                        help="Customer feedback CSV. Repeatable. Defaults to the legacy hardcoded files.")
    parser.add_argument("--eval_script", type=Path, default=None,
                        help="Path to run_threshold_eval.py.")
    parser.add_argument("--disable_cache",  action="store_true")
    parser.add_argument("--skip_eval",      action="store_true",
                        help="Skip eval, only regenerate summary CSVs from existing raw results")
    parser.add_argument("--rebuild_whitelist", action="store_true",
                        help="Re-parse feedback CSVs even if whitelist JSON already exists")
    parser.add_argument("--limit_tasks", type=int, default=0,
                        help="Only run on the first N tasks (validation batch). 0 = all.")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        JSONL_FILES = args.jsonl
    if args.feedback_csv:
        FEEDBACK_CSVS = args.feedback_csv
    if args.eval_script:
        EVAL_SCRIPT = args.eval_script

    # Keep generated driver files with the requested output, not next to this script.
    WHITELIST_JSON = out_root / "customer_feedback_whitelist.json"
    DETAIL_JSON = out_root / "customer_feedback_detail.json"
    COMBINED_JSONL = out_root / "combined_for_customer_eval.jsonl"
    REMAP_LOG_JSON = out_root / "customer_feedback_remap_log.json"
    VERSION_DRIFT_JSON = out_root / "customer_feedback_version_drift.json"

    if not EVAL_SCRIPT.exists():
        print(f"[ERROR] run_threshold_eval.py not found: {EVAL_SCRIPT}")
        sys.exit(1)

    # Step 1: build whitelist + feedback detail
    if args.rebuild_whitelist or not WHITELIST_JSON.exists():
        whitelist, feedback = build_whitelist_and_feedback()
    else:
        with open(WHITELIST_JSON) as f:
            whitelist = json.load(f)
        with open(DETAIL_JSON) as f:
            feedback = json.load(f)
        total = sum(len(v) for v in whitelist.values())
        print(f"Using existing whitelist: {len(whitelist)} tasks, {total} items")

    # Optional: trim to a validation batch of the first N tasks
    if args.limit_tasks and args.limit_tasks > 0:
        keep = list(whitelist.keys())[:args.limit_tasks]
        whitelist = {tid: whitelist[tid] for tid in keep}
        feedback  = {tid: feedback[tid]  for tid in keep}
        # rewrite the JSON the eval reads so it only sees the batch
        with open(WHITELIST_JSON, "w") as f:
            json.dump(whitelist, f, indent=2)
        with open(DETAIL_JSON, "w") as f:
            json.dump(feedback, f, indent=2)
        print(f"[limit_tasks] Validation batch: {len(whitelist)} tasks, "
              f"{sum(len(v) for v in whitelist.values())} items")

    # Step 2: build combined JSONL
    if not COMBINED_JSONL.exists() or args.rebuild_whitelist or args.limit_tasks:
        build_combined_jsonl(whitelist)
    else:
        print(f"Using existing combined JSONL: {COMBINED_JSONL}")

    # Step 3: run eval (or skip)
    cache_folder = os.path.abspath(os.path.join(args.output_dir, "cache"))
    raw_csv = os.path.join(cache_folder, "item_results_raw.csv")

    if not args.skip_eval:
        raw_csv = run_eval(args.output_dir, args.workers, args.dl_workers, args.disable_cache)
    else:
        print(f"Skipping eval, using existing results: {raw_csv}")

    if not os.path.exists(raw_csv):
        print(f"[ERROR] Raw results not found at {raw_csv}")
        sys.exit(1)

    # Step 4: build summary CSVs
    build_summary_csvs(raw_csv, args.output_dir, feedback)

    # Step 5: verifier-quality scoring (major/minor %, gates, missing criteria)
    totals = load_task_totals()
    build_quality_outputs(raw_csv, args.output_dir, totals)


if __name__ == "__main__":
    main()
