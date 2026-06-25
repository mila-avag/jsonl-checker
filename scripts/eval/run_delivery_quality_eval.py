#!/usr/bin/env python3
"""Run the verifier-quality eval on a delivery JSONL whose unit tests are NOT
embedded in the JSONL but referenced via `unit_tests_s3Url` in a separate sheet.

Steps:
  1. Load the delivery JSONL (rubrics present, unit_tests empty).
  2. For each task, look up unit_tests_s3Url from the sheet, convert scale-cds://
     -> https, download the test file, and parse its `def test_*` names.
  3. Augment the JSONL so each task's `unit_tests` lists those test names.
  4. Build a whitelist of ALL rubrics + ALL tests per task (full quality audit,
     neutral mode -- no customer feedback).
  5. Invoke run_threshold_eval.py (it re-downloads + places each test file into
     the extracted S3 env so the verifier can read it).
  6. Post-process into the quality report (% major / % minor + gates).

Usage:
    python run_delivery_quality_eval.py \
        --jsonl /path/to/delivery.jsonl \
        --csv   /path/to/unit_test_urls.csv \
        --output_dir data/outputs/delivery_quality \
        --workers 25
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

csv.field_size_limit(10_000_000)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
import run_customer_feedback_eval as rcf  # noqa: E402  (build_quality_outputs)

EVAL_SCRIPT = BASE.parent.parent / "apps" / "tara_eval_open_claw_rl_main_2" / "run_threshold_eval.py"

TEST_DEF_RE = re.compile(r"def\s+(test_[A-Za-z0-9_]+)\s*\(")


def cds_to_https(url: str) -> str:
    url = (url or "").strip()
    if not url or url.startswith("http"):
        return url
    if url.startswith("scale-cds://"):
        body = url[len("scale-cds://"):]
        left, _, right = body.partition("#s3/")
        bucket = right.strip().rstrip("/")
        if bucket and left:
            return f"https://{bucket}.s3.amazonaws.com/{left}"
    return url


def parse_test_names(text: str) -> list[str]:
    return sorted(set(TEST_DEF_RE.findall(text)))


def _avg_pass_rate(o, g):
    vals = [v for v in (o, g) if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def download_test_file(url: str) -> str | None:
    """Return the unit-test file text for a (possibly scale-cds) url, or None."""
    https = cds_to_https(url)
    if not https or https.startswith("https://."):
        return None
    try:
        resp = requests.get(https, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] download failed: {e}")
        return None
    if resp.content[:2] == b"PK":  # zip
        import io
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for m in zf.namelist():
                    if m.endswith(".py"):
                        return zf.read(m).decode("utf-8", "replace")
        except Exception as e:
            print(f"  [warn] zip parse failed: {e}")
            return None
    return resp.content.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--csv", default=None,
                    help="optional sheet with task_id + unit_tests_s3Url. Only needed when the JSONL "
                         "has empty unit_tests AND the env zips do not bundle tests/test_outputs.py.")
    ap.add_argument("--output_dir", default="data/outputs/delivery_quality")
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--dl_workers", type=int, default=10)
    ap.add_argument("--limit_tasks", type=int, default=0)
    ap.add_argument("--exclude_pass_rate_at_or_above", type=float, default=None,
                    help="Skip rubrics/tests whose avg pass rate is >= this value (e.g. 0.85). "
                         "Items with no pass-rate data are always kept.")
    ap.add_argument("--disable_cache", action="store_true")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Only (re)build the quality report from existing raw results")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache_folder = os.path.join(out_dir, "cache")
    os.makedirs(cache_folder, exist_ok=True)

    aug_jsonl = os.path.join(out_dir, "augmented.jsonl")
    whitelist_json = os.path.join(out_dir, "whitelist.json")
    unit_tests_csv = os.path.join(out_dir, "unit_test_urls.csv")
    raw_csv = os.path.join(cache_folder, "item_results_raw.csv")

    # ---- load tasks ----
    tasks = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    print(f"Loaded {len(tasks)} tasks from JSONL")

    # ---- load unit-test URLs from sheet (optional) ----
    url_by_task: dict[str, str] = {}
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tid = (r.get("task_id", "") or "").strip()
                u = (r.get("unit_tests_s3Url", "") or "").strip()
                if tid and u and tid not in url_by_task:
                    url_by_task[tid] = u
        print(f"Sheet has unit_tests_s3Url for {len(url_by_task)} tasks")
    else:
        print("No unit-test sheet provided -> using unit tests embedded in the JSONL")

    if not args.skip_eval:
        # ---- ensure each task has a unit_tests list ----
        # Prefer unit tests already embedded in the JSONL. Only fetch from the sheet
        # for tasks that have NO embedded tests but DO have a unit_tests_s3Url.
        n_embedded = 0
        n_fetched = 0
        total_tests = 0
        ut_rows = []
        for t in tasks:
            tid = t["task_id"]
            embedded = t.get("unit_tests") or []
            raw_url = url_by_task.get(tid, "")
            if embedded:
                n_embedded += 1
                total_tests += len(embedded)
                # The test NAMES/weights/pass-rates are embedded, but the actual test
                # CODE may not be bundled in the env zip. If we have an s3Url, still
                # pass it through so the inner eval can fetch the code (matched by name).
                if raw_url:
                    ut_rows.append({"task_id": tid, "unit_tests_s3Url": cds_to_https(raw_url)})
                continue
            names = []
            if raw_url:
                text = download_test_file(raw_url)
                if text:
                    names = parse_test_names(text)
                ut_rows.append({"task_id": tid, "unit_tests_s3Url": cds_to_https(raw_url)})
            t["unit_tests"] = [{"test_name": n, "weight": 1} for n in names]
            if names:
                n_fetched += 1
                total_tests += len(names)
        print(f"Unit tests: {n_embedded} tasks embedded, {n_fetched} fetched from sheet "
              f"({total_tests} tests total)")

        # ---- write augmented JSONL ----
        with open(aug_jsonl, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        # ---- unit-test url csv (https) for run_threshold_eval to place fetched files ----
        if ut_rows:
            with open(unit_tests_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task_id", "unit_tests_s3Url"])
                w.writeheader()
                for row in ut_rows:
                    w.writerow(row)

        # ---- whitelist = all rubrics + all tests per task (optionally excluding
        #      high-pass-rate items) ----
        cutoff = args.exclude_pass_rate_at_or_above
        whitelist = {}
        n_excluded = 0
        for t in tasks:
            rr = t.get("rubrics_rating", {}) or {}
            ids = []
            for rid, r in (t.get("rubrics", {}) or {}).items():
                if cutoff is not None:
                    o = rr.get(rid, {}).get("pass_rate_opus", r.get("pass_rate_opus"))
                    g = rr.get(rid, {}).get("pass_rate_gemini", r.get("pass_rate_gemini31"))
                    a = _avg_pass_rate(o, g)
                    if a is not None and a >= cutoff:
                        n_excluded += 1
                        continue
                ids.append(rid)
            for ut in t.get("unit_tests", []):
                if cutoff is not None:
                    a = _avg_pass_rate(ut.get("pass_rate_opus"), ut.get("pass_rate_gemini31"))
                    if a is not None and a >= cutoff:
                        n_excluded += 1
                        continue
                ids.append(ut["test_name"])
            if ids:
                whitelist[t["task_id"]] = ids
        with open(whitelist_json, "w") as f:
            json.dump(whitelist, f, indent=2)
        n_items = sum(len(v) for v in whitelist.values())
        if cutoff is not None:
            print(f"Excluded {n_excluded} items with avg pass rate >= {cutoff:.0%}")
        print(f"Whitelist: {len(whitelist)} tasks, {n_items} items "
              f"(+{len(whitelist)*3} task-level checks)")

        # ---- run the eval ----
        cmd = [
            sys.executable, "-u", str(EVAL_SCRIPT),
            "--jsonl", aug_jsonl,
            "--cache_folder", cache_folder,
            "--output", "item_results_raw.csv",
            "--workers", str(args.workers),
            "--dl_workers", str(args.dl_workers),
            "--item_whitelist", whitelist_json,
        ]
        if ut_rows:
            cmd += ["--unit_tests_csv", unit_tests_csv]
        if args.disable_cache:
            cmd.append("--disable_cache")
        print("\n" + "=" * 70)
        print("Running tara quality eval …")
        print(" ".join(cmd))
        print("=" * 70 + "\n")
        proc = subprocess.run(cmd, cwd=str(EVAL_SCRIPT.parent))
        if proc.returncode != 0:
            print(f"[ERROR] eval exited with code {proc.returncode}")
            sys.exit(proc.returncode)

    if not os.path.exists(raw_csv):
        print(f"[ERROR] raw results not found at {raw_csv}")
        sys.exit(1)

    # ---- totals (denominator = all rubrics + tests per task) ----
    totals = {}
    for t in tasks:
        n_rub = len(t.get("rubrics", {}) or {})
        n_test = len(t.get("unit_tests", []) or [])
        totals[t["task_id"]] = {
            "n_rubrics": n_rub,
            "n_tests": n_test,
            "total_verifiers": n_rub + n_test,
        }

    # ---- quality report ----
    rcf.build_quality_outputs(raw_csv, out_dir, totals)


if __name__ == "__main__":
    main()
