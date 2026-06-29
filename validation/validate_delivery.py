#!/usr/bin/env python3
"""
Validate OpenClaw / Skills delivery data for consistency across:
- Delivery JSONL (rubrics, unit_tests, env URLs)
- Pass@k metadata CSV (per-run results)
- Downloaded environment zips (rubric.json, test_weights.json, test_outputs.py)
- Customer CSV (FORMATTED_RESPONSE_J_S_O_N with embedded passAtKResults)

Checks performed:
  1.  Rubric text/weight/count: ENV vs delivery
  2.  Test name/weight/count: ENV vs delivery
  3.  Rubric pass rates: ENV vs delivery
  4.  Rubric count/weight/text: delivery vs pass@k
  5.  Test count/weight/name: delivery vs pass@k
  6.  Rubric pass rate: delivery vs computed from runs
  7.  Test pass rate: delivery vs computed from runs
  8.  pytest_ratio / rubrics_ratio correctness
  9.  test_outputs.py functions vs delivery tests
  10. Weight x10 bug detection
  11. Run count per model (expect 8 each)
  12. Justifications for 0/0 rubrics (delivery + env)
  13. Justifications for 0/0 unit tests (delivery)
  14. Missing pass@k on any task
  15. Duplicate test names in delivery
  16. Sentinel test entries in env
  17. Zero rubrics (every task must have rubrics)
  18. Missing environment URL
  19. Empty/null criterion text or test name
  20. Rubric key gaps (R1, R2, R5 → missing R3, R4)
  21. Pass rate out of range (< 0 or > 1)
  22. Weight sign vs is_positive mismatch
  23. Invalid rubric weight values (must be in {-5,-3,-1,1,3,5})
  24. Invalid test weight values (must be in {1,3,5})
  25. Duplicate rubric criteria text within a task
  26. model_a run detection in pass@k
  27. Required Opus/Claude and Gemini pass-rate fields on every rubric/test
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(50_000_000)

PASS_RATE_FIELDS = ["pass_rate_opus", "pass_rate_gemini31"]

ALL_JUSTIFICATION_FIELDS = [
    "why_rubric_is_correct",
    "why_rubric_is_present",
    "what_model_did_wrong",
    "incorrect_rubric_justification",
    "incorrect_unit_test_justification",
    "correct_answer_justification_rubric",
    "correct_answer_justification_unit_test",
    "model_mistake_justification_rubric",
    "model_mistake_justification_unit_test",
]

LEGACY_ALIAS = {"why_rubric_is_important": "why_rubric_is_present"}


# ── helpers ──────────────────────────────────────────────────────────

def is_pass(result):
    return result in ("PASS", "PASSED")


def is_opus(model):
    return model and ("claude" in model or "opus" in model.lower())


def is_gemini(model):
    return model and "gemini" in model


def parse_if_string(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
    return obj


def safe_get_criteria(run):
    rr = run.get("rubric_results")
    if not rr or not isinstance(rr, dict):
        return None
    inner = rr.get("rubric_results")
    if isinstance(inner, list):
        return inner
    return rr.get("criteria")


def safe_get_cases(run):
    tr = run.get("test_results")
    if not tr or not isinstance(tr, dict):
        return None
    inner = tr.get("test_results")
    if isinstance(inner, list):
        return inner
    return tr.get("cases")


def get_passk_runs(task, pak):
    if pak and isinstance(pak, dict) and pak.get("runs"):
        return pak["runs"]
    if task.get("passk_runs"):
        return task["passk_runs"]
    return None


def has_delivery_pass_rate(items):
    for item in items:
        if not isinstance(item, dict):
            continue
        if "pass_rate_opus" in item or "pass_rate_gemini31" in item:
            return True
    return False


def has_any_passk_result(task, pak, dl_rubrics, dl_tests):
    """Return True if delivery or pass@k runs contain rubric/test scoring.

    Some no-unit-test tasks have test_results={"error": "no verifier.py available"}.
    That is not missing pass@k when rubric_results are present.
    """
    if has_delivery_pass_rate(dl_rubrics.values()) or has_delivery_pass_rate(dl_tests):
        return True

    pk_runs = get_passk_runs(task, pak)
    if not pk_runs or not isinstance(pk_runs, dict):
        return False

    for run in pk_runs.values():
        criteria = safe_get_criteria(run)
        if criteria:
            return True
        cases = safe_get_cases(run)
        if cases:
            return True
    return False


def has_justification(item):
    """Return True if the item has at least 3 non-empty justification fields."""
    ann = item.get("annotations")
    if ann:
        ann = parse_if_string(ann)
        if isinstance(ann, dict):
            filled = 0
            for f in ALL_JUSTIFICATION_FIELDS:
                if str(ann.get(f, "")).strip():
                    filled += 1
            for old_key, new_key in LEGACY_ALIAS.items():
                if old_key in ann and str(ann[old_key]).strip() and not str(ann.get(new_key, "")).strip():
                    filled += 1
            if filled >= 3:
                return True

    for f in ALL_JUSTIFICATION_FIELDS:
        pass  # top-level fields checked below

    filled = 0
    for f in ALL_JUSTIFICATION_FIELDS:
        if str(item.get(f, "")).strip():
            filled += 1
    for old_key, new_key in LEGACY_ALIAS.items():
        if old_key in item and str(item[old_key]).strip() and not str(item.get(new_key, "")).strip():
            filled += 1
    return filled >= 3


def is_pass0(item):
    """Check if both opus and gemini pass rates are 0 (or gemini missing with opus=0)."""
    pr_o = item.get("pass_rate_opus")
    pr_g = item.get("pass_rate_gemini31")
    try:
        pr_o = float(pr_o) if pr_o is not None else None
    except (ValueError, TypeError):
        pr_o = None
    try:
        pr_g = float(pr_g) if pr_g is not None else None
    except (ValueError, TypeError):
        pr_g = None

    if pr_o is None:
        return False
    if pr_o != 0:
        return False
    if pr_g is not None and pr_g != 0:
        return False
    return True


# ── download ─────────────────────────────────────────────────────────

def download_env(tid, url, base_dir):
    td = os.path.join(base_dir, tid)
    os.makedirs(td, exist_ok=True)
    zp = os.path.join(td, "dl.zip")
    try:
        r = subprocess.run(
            ["curl", "-sS", "-o", zp, "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=120,
        )
        if r.stdout.strip() != "200":
            return tid, f"HTTP {r.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return tid, "TIMEOUT"

    extract_dir = os.path.join(td, "_raw")
    try:
        subprocess.run(
            ["unzip", "-o", "-q", zp, "-d", extract_dir],
            capture_output=True, text=True, timeout=300, errors="replace",
        )
    except subprocess.TimeoutExpired:
        return tid, "UNZIP_TIMEOUT"

    tests_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if os.path.basename(root) == "tests" and (
            "rubric.json" in files or "test_weights.json" in files
        ):
            tests_dir = root
            break

    if tests_dir:
        dest = os.path.join(td, "tests")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(tests_dir, dest)

    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(zp):
        os.remove(zp)
    return tid, "OK"


# ── loaders ──────────────────────────────────────────────────────────

def load_delivery_jsonl(path):
    data = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[obj["task_id"]] = obj
    return data


def load_passk_csv(path):
    with open(path, "r") as f:
        return {row["_ID"]: row for row in csv.DictReader(f)}


def load_customer_csv(path):
    tasks = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            tid = row["TASK_ID"]
            d = json.loads(row["FORMATTED_RESPONSE_J_S_O_N"])
            pak_raw = d.get("passAtKResults")
            pak = json.loads(pak_raw) if pak_raw and isinstance(pak_raw, str) else pak_raw
            tasks[tid] = {"data": d, "pak": pak, "name": row.get("NAME", "")}
    return tasks


# ── checks ───────────────────────────────────────────────────────────

def run_checks(tasks, base_dir, expected_runs=8):
    issues = []

    def add(tid, pair, cat, detail):
        issues.append({"task_id": tid, "pair": pair, "cat": cat, "detail": detail})

    def missing_pass_rate_fields(item):
        return [field for field in PASS_RATE_FIELDS if field not in item or item.get(field) is None]

    for tid in sorted(tasks.keys()):
        t = tasks[tid]
        d = t["data"]
        pak = t.get("pak")

        dl_rubrics_raw = d.get("rubrics", {})
        dl_rubrics = {}
        if isinstance(dl_rubrics_raw, dict):
            for rk, rv in dl_rubrics_raw.items():
                dl_rubrics[rk] = parse_if_string(rv) if isinstance(rv, (str, dict)) else rv
        elif isinstance(dl_rubrics_raw, str):
            try:
                dl_rubrics = json.loads(dl_rubrics_raw)
            except (json.JSONDecodeError, TypeError):
                dl_rubrics = {}

        dl_tests_raw = d.get("unit_tests", [])
        if isinstance(dl_tests_raw, str):
            try:
                dl_tests_raw = json.loads(dl_tests_raw)
            except (json.JSONDecodeError, TypeError):
                dl_tests_raw = []
        dl_tests = [parse_if_string(tt) for tt in dl_tests_raw if isinstance(parse_if_string(tt), dict)]
        dl_t_by = {tt["test_name"]: tt for tt in dl_tests if "test_name" in tt}

        is_rubric_only = len(dl_tests) == 0

        # ── Load env files ──
        env_rb_path = os.path.join(base_dir, tid, "tests", "rubric.json")
        env_wt_path = os.path.join(base_dir, tid, "tests", "test_weights.json")
        env_py_path = os.path.join(base_dir, tid, "tests", "test_outputs.py")
        env_sh_path = os.path.join(base_dir, tid, "tests", "test.sh")

        env_rubrics = []
        if os.path.exists(env_rb_path):
            try:
                env_rubrics = json.load(open(env_rb_path))
            except json.JSONDecodeError:
                add(tid, "ENV", "parse_error", "rubric.json is not valid JSON")
        elif dl_rubrics:
            add(tid, "ENV", "missing_file", "rubric.json missing but delivery has rubrics")

        env_tests = []
        env_pr = None
        env_rr = None
        if os.path.exists(env_wt_path):
            try:
                wd = json.load(open(env_wt_path))
                if isinstance(wd, dict):
                    env_tests = wd.get("tests", [])
                    env_pr = wd.get("pytest_ratio")
                    env_rr = wd.get("rubrics_ratio")
                elif isinstance(wd, list):
                    env_tests = wd
            except json.JSONDecodeError:
                add(tid, "ENV", "parse_error", "test_weights.json is not valid JSON")
        elif dl_tests and not is_rubric_only:
            add(tid, "ENV", "missing_file", "test_weights.json missing but delivery has unit_tests")

        env_t_by = {tt["test_name"]: tt for tt in env_tests if "test_name" in tt}
        env_by_c = {}
        for r in env_rubrics:
            c = r.get("criteria", r.get("criterion", ""))
            if c:
                env_by_c[c.strip().lower()] = r

        py_tests = set()
        if os.path.exists(env_py_path):
            with open(env_py_path) as f:
                py_tests = set(re.findall(r"def (test_\w+)", f.read()))

        has_test_file = os.path.exists(env_py_path) or os.path.exists(env_sh_path)

        # ── CHECK 14: Missing pass@k ──
        if not has_any_passk_result(t, pak, dl_rubrics, dl_tests):
            add(tid, "PassK", "missing", "Task has no pass@k results on any rubric or test")

        # ── CHECK 15: Duplicate test names ──
        test_names_list = [tt.get("test_name") for tt in dl_tests if tt.get("test_name")]
        name_counts = Counter(test_names_list)
        for tn, cnt in name_counts.items():
            if cnt > 1:
                add(tid, "Duplicates", "duplicate_test", f"{tn} appears {cnt} times in delivery unit_tests")

        # ── CHECK 16: Sentinel tests ──
        for tt in env_tests:
            tn = tt.get("test_name", "")
            if "_skipped_no_pass_at_k" in tn or "sentinel" in tn.lower():
                add(tid, "Sentinel", "sentinel_test", f"Sentinel entry in env: {tn}")

        # ── CHECK 17: Zero rubrics ──
        if not dl_rubrics:
            add(tid, "DataIntegrity", "zero_rubrics", "Task has 0 rubrics (every task should have rubrics)")

        # ── CHECK 18: Missing environment URL ──
        env_url = d.get("delivery_url") or d.get("environment_docker_file", "")
        if not env_url:
            add(tid, "DataIntegrity", "no_env_url", "Task has no delivery_url or environment_docker_file")

        # ── CHECK 19: Empty/null criterion text or test name ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            crit = rv.get("criterion", "")
            if not str(crit).strip():
                add(tid, "DataIntegrity", "empty_criterion", f"{rk} has empty/null criterion text")
        for tt in dl_tests:
            tn = tt.get("test_name", "")
            if not str(tn).strip():
                add(tid, "DataIntegrity", "empty_test_name", "A unit_test entry has empty/null test_name")

        # ── CHECK 20: Rubric key gaps (R1, R2, R5 → missing R3, R4) ──
        r_nums = []
        for rk in dl_rubrics.keys():
            m = re.match(r'^R(\d+)$', rk)
            if m:
                r_nums.append(int(m.group(1)))
        if r_nums:
            r_nums_sorted = sorted(r_nums)
            expected_seq = list(range(r_nums_sorted[0], r_nums_sorted[-1] + 1))
            missing_keys = set(expected_seq) - set(r_nums)
            if missing_keys:
                missing_str = ", ".join(f"R{n}" for n in sorted(missing_keys))
                add(tid, "DataIntegrity", "rubric_key_gap",
                    f"Rubric keys have gaps — missing: {missing_str}")

        # ── CHECK 27: Required Opus/Claude and Gemini pass-rate fields ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            missing = missing_pass_rate_fields(rv)
            if missing:
                add(tid, "PassRate", "delivery_rubric_missing",
                    f"{rk} missing {', '.join(missing)}")
        for tt in dl_tests:
            tn = tt.get("test_name", "??")
            missing = missing_pass_rate_fields(tt)
            if missing:
                add(tid, "PassRate", "delivery_test_missing",
                    f"{tn} missing {', '.join(missing)}")
        for idx, er in enumerate(env_rubrics, start=1):
            if not isinstance(er, dict):
                continue
            missing = missing_pass_rate_fields(er)
            if missing:
                label = (er.get("criteria") or er.get("criterion") or f"rubric #{idx}").strip()
                add(tid, "PassRate", "env_rubric_missing",
                    f"{label[:70]} missing {', '.join(missing)}")
        for et in env_tests:
            if not isinstance(et, dict):
                continue
            tn = et.get("test_name", "??")
            missing = missing_pass_rate_fields(et)
            if missing:
                add(tid, "PassRate", "env_test_missing",
                    f"{tn} missing {', '.join(missing)}")

        # ── CHECK 21: Pass rate out of range ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            for fld in PASS_RATE_FIELDS:
                val = rv.get(fld)
                if val is not None:
                    try:
                        vf = float(val)
                        if vf < 0 or vf > 1:
                            add(tid, "DataIntegrity", "pass_rate_range",
                                f"{rk} {fld}={vf} (must be 0-1)")
                    except (ValueError, TypeError):
                        add(tid, "DataIntegrity", "pass_rate_invalid",
                            f"{rk} {fld}={val!r} (not a number)")
        for tt in dl_tests:
            tn = tt.get("test_name", "??")
            for fld in PASS_RATE_FIELDS:
                val = tt.get(fld)
                if val is not None:
                    try:
                        vf = float(val)
                        if vf < 0 or vf > 1:
                            add(tid, "DataIntegrity", "pass_rate_range",
                                f"test {tn} {fld}={vf} (must be 0-1)")
                    except (ValueError, TypeError):
                        add(tid, "DataIntegrity", "pass_rate_invalid",
                            f"test {tn} {fld}={val!r} (not a number)")

        # ── CHECK 22: Weight sign vs is_positive mismatch ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            is_pos = rv.get("is_positive")
            w = rv.get("score", rv.get("weight"))
            if is_pos is not None and w is not None:
                try:
                    wf = float(w)
                    if is_pos is True and wf < 0:
                        add(tid, "DataIntegrity", "is_positive_mismatch",
                            f"{rk}: is_positive=true but weight={w}")
                    elif is_pos is False and wf > 0:
                        add(tid, "DataIntegrity", "is_positive_mismatch",
                            f"{rk}: is_positive=false but weight={w}")
                except (ValueError, TypeError):
                    pass

        # ── CHECK 23: Invalid rubric weight values ──
        VALID_RUBRIC_WEIGHTS = {-5, -3, -1, 1, 3, 5}
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            w = rv.get("score", rv.get("weight"))
            if w is not None:
                try:
                    wf = float(w)
                    if wf not in VALID_RUBRIC_WEIGHTS:
                        add(tid, "Weights", "invalid_rubric_weight",
                            f"{rk}: weight={w} (expected one of {sorted(VALID_RUBRIC_WEIGHTS)})")
                except (ValueError, TypeError):
                    add(tid, "Weights", "weight_not_numeric", f"{rk}: weight={w!r}")

        # ── CHECK 24: Invalid test weight values ──
        VALID_TEST_WEIGHTS = {1, 3, 5}
        for tt in dl_tests:
            tn = tt.get("test_name", "??")
            w = tt.get("weight")
            if w is not None:
                try:
                    wf = float(w)
                    if wf not in VALID_TEST_WEIGHTS:
                        add(tid, "Weights", "invalid_test_weight",
                            f"{tn}: weight={w} (expected one of {sorted(VALID_TEST_WEIGHTS)})")
                except (ValueError, TypeError):
                    add(tid, "Weights", "weight_not_numeric", f"test {tn}: weight={w!r}")

        # ── CHECK 25: Duplicate rubric criteria text ──
        crit_texts = {}
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            ct = rv.get("criterion", "").strip().lower()
            if ct:
                crit_texts.setdefault(ct, []).append(rk)
        for ct, keys in crit_texts.items():
            if len(keys) > 1:
                add(tid, "Duplicates", "duplicate_rubric",
                    f"Identical criterion text in {', '.join(keys)}: {ct[:70]}")

        # ── CHECK 1: ENV vs Delivery rubrics ──
        if env_rubrics and dl_rubrics:
            env_crit_set = set(env_by_c.keys())
            dl_crit_set = set()
            dl_by_crit = {}
            for rk, dr in dl_rubrics.items():
                if not isinstance(dr, dict):
                    continue
                cn = dr.get("criterion", "").strip().lower()
                dl_crit_set.add(cn)
                dl_by_crit[cn] = (rk, dr)

            for cn in sorted(dl_crit_set - env_crit_set):
                rk, _ = dl_by_crit[cn]
                add(tid, "ENV_vs_Data", "rubric_only_in_data", f"{rk}: only in delivery")
            for cn in sorted(env_crit_set - dl_crit_set):
                add(tid, "ENV_vs_Data", "rubric_only_in_env", f"only in env: {cn[:70]}")

            for cn in dl_crit_set & env_crit_set:
                rk, dr = dl_by_crit[cn]
                er = env_by_c[cn]
                dl_w = dr.get("score", dr.get("weight"))
                env_w = er.get("weight")
                if dl_w is not None and env_w is not None:
                    try:
                        dl_w_f, env_w_f = float(dl_w), float(env_w)
                        if dl_w_f != env_w_f:
                            add(tid, "ENV_vs_Data", "rubric_weight", f"{rk}: delivery={dl_w} env={env_w}")
                    except (ValueError, TypeError):
                        pass

                for fld in PASS_RATE_FIELDS:
                    dp, ep = dr.get(fld), er.get(fld)
                    if dp is not None and ep is not None:
                        try:
                            if abs(float(dp) - float(ep)) > 0.001:
                                add(tid, "ENV_vs_Data", f"rubric_{fld}",
                                    f"{rk}: delivery={dp} env={ep}")
                        except (ValueError, TypeError):
                            pass

                dl_type = dr.get("type")
                env_type = er.get("type")
                if dl_type and env_type and dl_type != env_type:
                    add(tid, "ENV_vs_Data", "rubric_type", f"{rk}: delivery={dl_type} env={env_type}")

        # ── CHECK 2: ENV vs Delivery tests ──
        if (env_tests or dl_tests) and not is_rubric_only:
            if len(env_tests) != len(dl_tests):
                add(tid, "ENV_vs_Data", "test_count", f"env={len(env_tests)} delivery={len(dl_tests)}")
            for tn in sorted(set(dl_t_by) - set(env_t_by)):
                add(tid, "ENV_vs_Data", "test_only_in_data", f"{tn}")
            for tn in sorted(set(env_t_by) - set(dl_t_by)):
                add(tid, "ENV_vs_Data", "test_only_in_env", f"{tn}")
            for tn in set(dl_t_by) & set(env_t_by):
                dt, et = dl_t_by[tn], env_t_by[tn]
                dw = dt.get("weight")
                ew = et.get("weight")
                if dw is not None and ew is not None:
                    try:
                        if float(dw) != float(ew):
                            add(tid, "ENV_vs_Data", "test_weight", f"{tn}: delivery={dw} env={ew}")
                    except (ValueError, TypeError):
                        pass
                for fld in PASS_RATE_FIELDS:
                    dp, ep = dt.get(fld), et.get(fld)
                    if dp is not None and ep is not None:
                        try:
                            if abs(float(dp) - float(ep)) > 0.001:
                                add(tid, "ENV_vs_Data", f"test_{fld}",
                                    f"{tn}: delivery={dp} env={ep}")
                        except (ValueError, TypeError):
                            pass

        # ── CHECK 8: Weight ratios ──
        if env_pr is not None and env_rubrics:
            ps = sum(float(tt.get("weight", 0)) for tt in env_tests)
            ra = sum(abs(float(r.get("weight", 0))) for r in env_rubrics)
            ta = ps + ra
            cp = ps / ta if ta > 0 else 0
            if abs(float(env_pr) - cp) > 0.0001:
                add(tid, "Ratios", "pytest_ratio", f"env={env_pr:.6f} computed={cp:.6f}")
            if float(env_pr) > 1.0:
                add(tid, "Ratios", "ratio_impossible", f"pytest_ratio={env_pr} > 1.0")
            if env_rr is not None and float(env_rr) < 0:
                add(tid, "Ratios", "ratio_impossible", f"rubrics_ratio={env_rr} < 0")

        # ── CHECK 10: Weight x10 bug ──
        all_abs_weights = []
        for rk, dr in dl_rubrics.items():
            if not isinstance(dr, dict):
                continue
            w = dr.get("score", dr.get("weight"))
            if w is not None:
                try:
                    all_abs_weights.append(abs(float(w)))
                except (ValueError, TypeError):
                    pass

            cn = dr.get("criterion", "").strip().lower()
            er = env_by_c.get(cn, {})
            env_w = er.get("weight")
            dl_w = dr.get("score", dr.get("weight"))
            if dl_w is not None and env_w is not None:
                try:
                    dl_wf, env_wf = abs(float(dl_w)), abs(float(env_w))
                    if dl_wf > 0 and env_wf > 0:
                        if abs(dl_wf - env_wf * 10) < 0.01 or abs(env_wf - dl_wf * 10) < 0.01:
                            add(tid, "Weights", "x10_mismatch",
                                f"{rk}: delivery={dl_w} env={env_w} (10x difference)")
                except (ValueError, TypeError):
                    pass

        for tt in dl_tests:
            w = tt.get("weight")
            if w is not None:
                try:
                    all_abs_weights.append(abs(float(w)))
                except (ValueError, TypeError):
                    pass

            tn = tt.get("test_name", "")
            et = env_t_by.get(tn, {})
            env_tw = et.get("weight")
            dl_tw = tt.get("weight")
            if dl_tw is not None and env_tw is not None:
                try:
                    dl_twf, env_twf = abs(float(dl_tw)), abs(float(env_tw))
                    if dl_twf > 0 and env_twf > 0:
                        if abs(dl_twf - env_twf * 10) < 0.01 or abs(env_twf - dl_twf * 10) < 0.01:
                            add(tid, "Weights", "x10_mismatch",
                                f"test {tn}: delivery={dl_tw} env={env_tw} (10x difference)")
                except (ValueError, TypeError):
                    pass

        if all_abs_weights:
            small = [w for w in all_abs_weights if 0 < w <= 5]
            large = [w for w in all_abs_weights if w >= 10]
            if small and large:
                add(tid, "Weights", "mixed_scale",
                    f"Task has weights in [1-5] range ({len(small)}) AND [10+] range ({len(large)})")

        # ── PASS@K run data ──
        pk_runs = get_passk_runs(t, pak)

        if pk_runs and isinstance(pk_runs, dict):
            # ── CHECK 11: Run count per model ──
            opus_runs = 0
            gemini_runs = 0
            for rk_r in pk_runs:
                parts = rk_r.split("::")
                model = parts[1] if len(parts) >= 2 else ""
                if is_opus(model):
                    opus_runs += 1
                elif is_gemini(model):
                    gemini_runs += 1

            if opus_runs != expected_runs:
                add(tid, "RunCount", "opus_runs",
                    f"Expected {expected_runs} Opus runs, found {opus_runs}")
            if gemini_runs != expected_runs:
                add(tid, "RunCount", "gemini_runs",
                    f"Expected {expected_runs} Gemini runs, found {gemini_runs}")

            # ── CHECK 26: model_a run detection ──
            model_a_count = 0
            for rk_r in pk_runs:
                parts = rk_r.split("::")
                model = parts[1] if len(parts) >= 2 else ""
                if model and "model_a" in model.lower():
                    model_a_count += 1
            if model_a_count > 0:
                add(tid, "RunCount", "model_a_present",
                    f"{model_a_count} model_a run(s) detected in pass@k (may need exclusion)")

            # ── CHECKs 4-7: Delivery vs pass@k ──
            pk_crit = None
            pk_cases = None
            for rk_r, rv_r in pk_runs.items():
                c = safe_get_criteria(rv_r)
                if c:
                    pk_crit = c
                    pk_cases = safe_get_cases(rv_r) or []
                    break

            if pk_crit:
                if len(dl_rubrics) != len(pk_crit):
                    add(tid, "Data_vs_PassK", "rubric_count",
                        f"delivery={len(dl_rubrics)} passk={len(pk_crit)}")

                pk_by_t = {c.get("title", "").strip().lower(): c for c in pk_crit}
                for rk, dr in dl_rubrics.items():
                    if not isinstance(dr, dict):
                        continue
                    cn = dr.get("criterion", "").strip().lower()
                    if cn in pk_by_t:
                        pk_w = pk_by_t[cn].get("weight")
                        dl_w = dr.get("score", dr.get("weight"))
                        if dl_w is not None and pk_w is not None:
                            try:
                                if float(dl_w) != float(pk_w):
                                    add(tid, "Data_vs_PassK", "rubric_weight",
                                        f"{rk}: delivery={dl_w} passk={pk_w}")
                            except (ValueError, TypeError):
                                pass
                    else:
                        add(tid, "Data_vs_PassK", "rubric_only_in_data", f"{rk}")

                # Compute pass rates from runs and compare
                csv_rr = {}
                for rk_r, run in pk_runs.items():
                    parts = rk_r.split("::")
                    model = parts[1] if len(parts) >= 2 else ""
                    cr = safe_get_criteria(run)
                    if not cr:
                        continue
                    for c in cr:
                        ct = c.get("title", "").strip().lower()
                        csv_rr.setdefault(ct, {"opus": [], "gemini": []})
                        if is_opus(model):
                            csv_rr[ct]["opus"].append(1 if is_pass(c.get("result", "")) else 0)
                        elif is_gemini(model):
                            csv_rr[ct]["gemini"].append(1 if is_pass(c.get("result", "")) else 0)

                for rk, dr in dl_rubrics.items():
                    if not isinstance(dr, dict):
                        continue
                    cn = dr.get("criterion", "").strip().lower()
                    if cn not in csv_rr:
                        continue
                    for mk, df in [("opus", "pass_rate_opus"), ("gemini", "pass_rate_gemini31")]:
                        if csv_rr[cn][mk]:
                            csv_p = sum(csv_rr[cn][mk]) / len(csv_rr[cn][mk])
                            dl_p = dr.get(df)
                            if dl_p is not None:
                                try:
                                    if abs(csv_p - float(dl_p)) > 0.001:
                                        add(tid, "Data_vs_PassK", f"rubric_pr_{mk}",
                                            f"{rk}: delivery={dl_p} computed={csv_p:.4f}")
                                except (ValueError, TypeError):
                                    pass

                # Tests vs pass@k
                if pk_cases is not None:
                    pk_tn = {c.get("title", "") for c in pk_cases}
                    dl_tn = set(dl_t_by.keys())
                    if len(dl_tests) != len(pk_cases):
                        add(tid, "Data_vs_PassK", "test_count",
                            f"delivery={len(dl_tests)} passk={len(pk_cases)}")
                    for tn in sorted(dl_tn - pk_tn):
                        add(tid, "Data_vs_PassK", "test_only_in_data", f"{tn}")
                    for tn in sorted(pk_tn - dl_tn):
                        add(tid, "Data_vs_PassK", "test_only_in_passk", f"{tn}")

                    pk_tb = {c.get("title", ""): c for c in pk_cases}
                    for tn in dl_tn & pk_tn:
                        dt, pt = dl_t_by[tn], pk_tb[tn]
                        dw = dt.get("weight")
                        pw = pt.get("weight")
                        if dw is not None and pw is not None:
                            try:
                                if float(dw) != float(pw):
                                    add(tid, "Data_vs_PassK", "test_weight",
                                        f"{tn}: delivery={dw} passk={pw}")
                            except (ValueError, TypeError):
                                pass

                    csv_tr = {}
                    for rk_r, run in pk_runs.items():
                        parts = rk_r.split("::")
                        model = parts[1] if len(parts) >= 2 else ""
                        cs = safe_get_cases(run)
                        if not cs:
                            continue
                        for c in cs:
                            csv_tr.setdefault(c.get("title", ""), {"opus": [], "gemini": []})
                            if is_opus(model):
                                csv_tr[c["title"]]["opus"].append(
                                    1 if is_pass(c.get("result", "")) else 0)
                            elif is_gemini(model):
                                csv_tr[c["title"]]["gemini"].append(
                                    1 if is_pass(c.get("result", "")) else 0)

                    for tn in dl_tn & pk_tn:
                        dt = dl_t_by[tn]
                        if tn not in csv_tr:
                            continue
                        for mk, df in [("opus", "pass_rate_opus"), ("gemini", "pass_rate_gemini31")]:
                            if csv_tr[tn][mk]:
                                csv_p = sum(csv_tr[tn][mk]) / len(csv_tr[tn][mk])
                                dl_p = dt.get(df)
                                if dl_p is not None:
                                    try:
                                        if abs(csv_p - float(dl_p)) > 0.001:
                                            add(tid, "Data_vs_PassK", f"test_pr_{mk}",
                                                f"{tn}: delivery={dl_p} computed={csv_p:.4f}")
                                    except (ValueError, TypeError):
                                        pass

        # ── CHECK 9: test_outputs.py vs delivery ──
        if py_tests and not is_rubric_only:
            for tn in sorted(py_tests - set(dl_t_by)):
                add(tid, "TestPy", "only_in_py", tn)
            for tn in sorted(set(dl_t_by) - py_tests):
                if has_test_file:
                    add(tid, "TestPy", "only_in_delivery", tn)
        elif dl_tests and not is_rubric_only and not has_test_file:
            add(tid, "TestPy", "no_test_file",
                f"Delivery has {len(dl_tests)} tests but no test_outputs.py or test.sh in env")

        # ── CHECK 12: Justifications for 0/0 rubrics ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            if is_pass0(rv):
                has_j = has_justification(rv)
                if not has_j:
                    cn = rv.get("criterion", "").strip().lower()
                    er = env_by_c.get(cn, {})
                    if er and has_justification(er):
                        has_j = True
                if not has_j:
                    add(tid, "Justification", "rubric_missing",
                        f'{rk}: {rv.get("criterion", "")[:70]}')

        # ── CHECK 13: Justifications for 0/0 unit tests ──
        for tt in dl_tests:
            if is_pass0(tt):
                if not has_justification(tt):
                    add(tid, "Justification", "test_missing",
                        f'{tt.get("test_name", "??")}')

    return issues


# ── reporting ────────────────────────────────────────────────────────

def print_report(issues):
    print(f"\n{'=' * 70}")
    print(f"TOTAL ISSUES: {len(issues)}")
    print(f"{'=' * 70}")

    if not issues:
        print("\nALL CLEAR — zero discrepancies!")
        return

    affected_tasks = len(set(i["task_id"] for i in issues))
    print(f"Affected tasks: {affected_tasks}")

    by_pair = {}
    for iss in issues:
        by_pair.setdefault(iss["pair"], {}).setdefault(iss["cat"], []).append(iss)

    for pair in sorted(by_pair):
        cats = by_pair[pair]
        total = sum(len(v) for v in cats.values())
        affected = len(set(i["task_id"] for v in cats.values() for i in v))
        print(f"\n{pair}: {total} issues, {affected} tasks")
        for cat, items in sorted(cats.items()):
            tc = len(set(i["task_id"] for i in items))
            print(f"  {cat}: {len(items)} ({tc} tasks)")
            for i in items[:5]:
                print(f'    [{i["task_id"][:12]}] {i["detail"]}')
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")

    print(f"\n{'=' * 70}")
    print("ISSUE SUMMARY BY CATEGORY:")
    print(f"{'=' * 70}")
    pair_totals = []
    for pair in sorted(by_pair):
        total = sum(len(v) for v in by_pair[pair].values())
        pair_totals.append((pair, total))
    for pair, total in sorted(pair_totals, key=lambda x: -x[1]):
        print(f"  {pair:<20} {total:>5}")


_HTML_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Delivery Validation — __DELIVERY_NAME_HTML__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;line-height:1.5}
h1{font-size:1.4rem;margin-bottom:4px;color:#f0f6fc}
h3{font-size:.95rem;margin:8px 0 10px;color:#f0f6fc}
.subtitle{color:#8b949e;margin-bottom:20px;font-size:.85rem;word-break:break-all}
.stats-bar{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}
.stat{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 20px;flex:1;min-width:150px;text-align:center}
.stat .num{font-size:1.8rem;font-weight:700}
.stat .label{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.stat.green .num{color:#3fb950}.stat.red .num{color:#f85149}.stat.blue .num{color:#58a6ff}.stat.yellow .num{color:#d29922}
.group{background:#161b22;border:1px solid #30363d;border-radius:10px;margin-bottom:16px;overflow:hidden}
.group-header{padding:14px 20px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none}
.group-header:hover{background:#1c2333}
.group-header h2{font-size:1rem;font-weight:600;display:flex;align-items:center;gap:8px}
.group-header .count{background:#30363d;color:#c9d1d9;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600}
.group-header .arrow{transition:transform .2s;color:#8b949e}
.group-header.open .arrow{transform:rotate(90deg)}
.group-body{display:none;border-top:1px solid #30363d}
.group-body.open{display:block}
.task-row{padding:10px 20px;border-bottom:1px solid #21262d;font-size:.82rem}
.task-row:last-child{border-bottom:none}
.task-row-header{display:flex;justify-content:space-between;align-items:center}
.task-detail{color:#8b949e;font-size:.78rem}
.task-expand{display:none;margin-top:8px}
.task-expand.open{display:block}
.query-box{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;font-family:'SF Mono',Monaco,monospace;font-size:.75rem;color:#8b949e;margin-top:8px;max-height:240px;overflow:auto;white-space:pre-wrap;word-break:break-all}
.copy-btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:.75rem}
.copy-btn:hover{background:#30363d;border-color:#58a6ff}
.copy-btn.copied{background:#238636;border-color:#2ea043;color:#fff}
.validation-actions{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
</style></head>
<body>
<h1>Delivery Validation Report</h1>
<p class="subtitle">__DELIVERY_NAME_HTML__</p>
<div id="app"></div>
<script>
const ISSUES = __ISSUES_JSON__;
const ALL_IDS = __ALL_IDS_JSON__;
const DELIVERY_NAME = __DELIVERY_NAME_JS__;

function issuePair(i){return i.pair||i.category||'';}
function issueCat(i){return i.cat||i.type||'';}
function escHtml(s){if(s===null||s===undefined)return '';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fmtPctNum(p,w){if(!w)return '0.0%';return ((p/w)*100).toFixed(1)+'%';}

function computeBreakdown(issues, allIds){
  const allSet=new Set(allIds), affectedSet=new Set(), cats={};
  for(const it of issues){
    const tid=it.task_id||'', pair=issuePair(it)||'Other', cat=issueCat(it)||'other';
    if(tid){affectedSet.add(tid);allSet.add(tid);}
    if(!cats[pair])cats[pair]={issues:0,tasks:new Set(),subs:{}};
    cats[pair].issues++; if(tid)cats[pair].tasks.add(tid);
    if(!cats[pair].subs[cat])cats[pair].subs[cat]={issues:0,tasks:new Set()};
    cats[pair].subs[cat].issues++; if(tid)cats[pair].subs[cat].tasks.add(tid);
  }
  const total=allSet.size;
  const affectedIds=[...affectedSet].sort();
  const cleanIds=[...allSet].filter(id=>!affectedSet.has(id)).sort();
  return {total,affected:affectedIds.length,clean:cleanIds.length,affectedIds,cleanIds,cats,totalIssues:issues.length};
}

const BD = computeBreakdown(ISSUES, ALL_IDS);
BD.fileName = DELIVERY_NAME;

function toggleGroup(h){h.classList.toggle('open');h.nextElementSibling.classList.toggle('open');}
function toggleExpand(id){const el=document.getElementById(id);if(el)el.classList.toggle('open');}
function flashCopied(btn,label){const o=label||btn.textContent;btn.textContent='Copied!';btn.classList.add('copied');setTimeout(()=>{btn.textContent=o;btn.classList.remove('copied');},1500);}
function copyIds(btn,pair,cat){const info=BD.cats[pair];if(!info||!info.subs[cat])return;navigator.clipboard.writeText([...info.subs[cat].tasks].sort().join('\n'));flashCopied(btn,'Copy IDs');}
function copyClean(btn){navigator.clipboard.writeText(BD.cleanIds.join('\n'));flashCopied(btn,'Copy clean IDs');}
function copyAffected(btn){navigator.clipboard.writeText(BD.affectedIds.join('\n'));flashCopied(btn,'Copy not-clean IDs');}
function baseName(){return (BD.fileName||'delivery').replace(/\.[^.]+$/,'');}
function downloadText(fn,c){const b=new Blob([c],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=fn;a.click();}
function summaryText(){
  const lines=[];
  lines.push('Total tasks: '+BD.total);
  lines.push('Clean tasks: '+BD.clean+' ('+fmtPctNum(BD.clean,BD.total)+')');
  lines.push('Not-clean tasks: '+BD.affected+' ('+fmtPctNum(BD.affected,BD.total)+')');
  lines.push('Total issues: '+BD.totalIssues); lines.push('');
  for(const [pair,info] of Object.entries(BD.cats).sort((a,b)=>b[1].issues-a[1].issues)){
    lines.push(pair+': '+info.issues+' issues, '+info.tasks.size+' tasks');
    for(const [cat,sub] of Object.entries(info.subs).sort((a,b)=>b[1].issues-a[1].issues))
      lines.push('    '+cat+': '+sub.issues+' ('+sub.tasks.size+' tasks)');
    lines.push('');
  }
  return lines.join('\n');
}
function categoryListsText(){
  const lines=[];
  for(const [pair,info] of Object.entries(BD.cats).sort((a,b)=>b[1].issues-a[1].issues)){
    lines.push('### '+pair+' ('+info.tasks.size+' tasks)');
    [...info.tasks].sort().forEach(id=>lines.push(id)); lines.push('');
    for(const [cat,sub] of Object.entries(info.subs).sort((a,b)=>b[1].issues-a[1].issues)){
      lines.push('--- '+pair+' / '+cat+' ('+sub.tasks.size+' tasks) ---');
      [...sub.tasks].sort().forEach(id=>lines.push(id)); lines.push('');
    }
  }
  return lines.join('\n');
}
function dlSummary(){downloadText(baseName()+'_validation_summary.txt',summaryText());}
function dlClean(){downloadText(baseName()+'_clean_tasks.txt','Clean tasks: '+BD.clean+' / '+BD.total+' ('+fmtPctNum(BD.clean,BD.total)+')\n\n'+BD.cleanIds.join('\n')+'\n');}
function dlAffected(){downloadText(baseName()+'_not_clean_tasks.txt','Not-clean tasks: '+BD.affected+' / '+BD.total+' ('+fmtPctNum(BD.affected,BD.total)+')\n\n'+BD.affectedIds.join('\n')+'\n');}
function dlCats(){downloadText(baseName()+'_category_task_ids.txt',categoryListsText());}
function dlAll(){dlSummary();dlClean();dlAffected();dlCats();}

function render(){
  const cats=Object.entries(BD.cats).sort((a,b)=>b[1].issues-a[1].issues);
  let html='<div class="stats-bar">'
    +'<div class="stat blue"><div class="num">'+BD.total+'</div><div class="label">Total Tasks</div></div>'
    +'<div class="stat green"><div class="num">'+BD.clean+'</div><div class="label">Clean — '+fmtPctNum(BD.clean,BD.total)+'</div></div>'
    +'<div class="stat red"><div class="num">'+BD.affected+'</div><div class="label">Not Clean — '+fmtPctNum(BD.affected,BD.total)+'</div></div>'
    +'<div class="stat yellow"><div class="num">'+BD.totalIssues+'</div><div class="label">Total Issues</div></div></div>';
  html+='<div class="validation-actions">'
    +'<button class="copy-btn" onclick="dlSummary()">Download summary .txt</button>'
    +'<button class="copy-btn" onclick="dlClean()">Download clean IDs .txt</button>'
    +'<button class="copy-btn" onclick="dlAffected()">Download not-clean IDs .txt</button>'
    +'<button class="copy-btn" onclick="dlCats()">Download category/subcategory ID lists .txt</button>'
    +'<button class="copy-btn" onclick="dlAll()">Download all .txt</button></div>';
  html+='<h3>Categories &amp; subcategories</h3>';
  for(const [pair,info] of cats){
    let subRows='';
    for(const [cat,sub] of Object.entries(info.subs).sort((a,b)=>b[1].issues-a[1].issues)){
      const ids=[...sub.tasks].sort();
      const expandId='vbd_'+(pair+'__'+cat).replace(/[^A-Za-z0-9_]/g,'_');
      subRows+='<div class="task-row"><div class="task-row-header"><div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        +'<span style="color:#484f58;cursor:pointer;font-size:.7rem" onclick="toggleExpand(\''+expandId+'\')">▶</span>'
        +'<span style="color:#c9d1d9;font-weight:600;cursor:pointer" onclick="toggleExpand(\''+expandId+'\')">'+escHtml(cat)+'</span>'
        +'<span class="task-detail">— '+sub.issues+' issue'+(sub.issues===1?'':'s')+', '+sub.tasks.size+' task'+(sub.tasks.size===1?'':'s')+'</span>'
        +'<button class="copy-btn" style="padding:2px 8px" onclick="copyIds(this,\''+pair+'\',\''+cat+'\')">Copy IDs</button>'
        +'</div></div><div class="task-expand" id="'+expandId+'"><div class="query-box">'+(ids.length?escHtml(ids.join('\n')):'(no task IDs)')+'</div></div></div>';
    }
    html+='<div class="group"><div class="group-header" onclick="toggleGroup(this)">'
      +'<h2><span style="color:#58a6ff">●</span> '+escHtml(pair)+' <span class="count">'+info.issues+' issues · '+info.tasks.size+' tasks</span></h2>'
      +'<span class="arrow">▶</span></div><div class="group-body">'+subRows+'</div></div>';
  }
  html+='<div class="group"><div class="group-header" onclick="toggleGroup(this)">'
    +'<h2><span style="color:#3fb950">●</span> Clean tasks <span class="count">'+BD.clean+' · '+fmtPctNum(BD.clean,BD.total)+'</span></h2>'
    +'<span class="arrow">▶</span></div><div class="group-body"><div class="task-row">'
    +'<button class="copy-btn" style="margin-bottom:8px" onclick="copyClean(this)">Copy clean IDs</button>'
    +'<div class="query-box">'+(BD.cleanIds.length?escHtml(BD.cleanIds.join('\n')):'(none)')+'</div></div></div></div>';
  html+='<div class="group"><div class="group-header" onclick="toggleGroup(this)">'
    +'<h2><span style="color:#f85149">●</span> Not-clean tasks <span class="count">'+BD.affected+' · '+fmtPctNum(BD.affected,BD.total)+'</span></h2>'
    +'<span class="arrow">▶</span></div><div class="group-body"><div class="task-row">'
    +'<button class="copy-btn" style="margin-bottom:8px" onclick="copyAffected(this)">Copy not-clean IDs</button>'
    +'<div class="query-box">'+(BD.affectedIds.length?escHtml(BD.affectedIds.join('\n')):'(none)')+'</div></div></div></div>';
  document.getElementById('app').innerHTML=html;
}
render();
</script>
</body></html>
"""


def write_html_report(html_path, issues, all_task_ids, delivery_name):
    """Write a self-contained, styled HTML breakdown report (no server needed)."""
    compact = [
        {"task_id": i.get("task_id", ""), "pair": i.get("pair", ""), "cat": i.get("cat", "")}
        for i in issues
    ]
    html = (
        _HTML_REPORT_TEMPLATE
        .replace("__ISSUES_JSON__", json.dumps(compact))
        .replace("__ALL_IDS_JSON__", json.dumps(sorted(set(all_task_ids))))
        .replace("__DELIVERY_NAME_JS__", json.dumps(delivery_name))
        .replace("__DELIVERY_NAME_HTML__", _esc_html(delivery_name))
    )
    with open(html_path, "w") as f:
        f.write(html)


def _esc_html(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main():
    parser = argparse.ArgumentParser(
        description="Validate OpenClaw/Skills delivery data — catches rubric/test "
                    "mismatches, pass rate drift, missing justifications, weight bugs, and more."
    )
    parser.add_argument("--customer-csv", help="Customer-facing CSV (New_Query format)")
    parser.add_argument("--delivery", nargs="+", help="Delivery JSONL file(s)")
    parser.add_argument("--passk-csv", help="Pass@k batches CSV")
    parser.add_argument("--work-dir", default="/tmp/delivery_validation",
                        help="Working directory for downloads")
    parser.add_argument("--output", default="validation_report.json",
                        help="Output report path")
    parser.add_argument("--html", default=None,
                        help="Path for a self-contained HTML breakdown report "
                             "(default: alongside --output as .html; use 'none' to skip)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Download parallelism")
    parser.add_argument("--expected-runs", type=int, default=8,
                        help="Expected number of pass@k runs per model (default: 8)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip env zip download (reuse existing work-dir)")
    args = parser.parse_args()

    base_dir = args.work_dir
    tasks = {}

    if args.customer_csv:
        print(f"Loading customer CSV: {args.customer_csv}")
        tasks = load_customer_csv(args.customer_csv)
    elif args.delivery:
        for df in args.delivery:
            print(f"Loading delivery: {df}")
            delivery = load_delivery_jsonl(df)
            for tid, obj in delivery.items():
                tasks[tid] = {"data": obj, "pak": None, "name": ""}

        if args.passk_csv:
            print(f"Loading pass@k CSV: {args.passk_csv}")
            passk = load_passk_csv(args.passk_csv)
            for tid in tasks:
                if tid in passk:
                    try:
                        tasks[tid]["passk_runs"] = json.loads(passk[tid]["RUNS"])
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass
    else:
        print("Error: provide --customer-csv or --delivery", file=sys.stderr)
        sys.exit(2)

    print(f"Tasks loaded: {len(tasks)}")

    if not args.skip_download:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)

        links = {}
        for tid, t in tasks.items():
            url = t["data"].get("delivery_url") or t["data"].get("environment_docker_file", "")
            if url:
                links[tid] = url

        print(f"Downloading {len(links)} environments...")
        done = 0
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_env, t, u, base_dir): t for t, u in links.items()}
            for fut in as_completed(futs):
                done += 1
                tid, st = fut.result()
                if st != "OK":
                    errors.append((tid, st))
                if done % 50 == 0 or done == len(links):
                    print(f"  [{done}/{len(links)}]")

        print(f"Downloads: {len(links) - len(errors)} OK, {len(errors)} errors")
        if errors:
            for tid, err in errors[:10]:
                print(f"  FAIL: {tid} — {err}")
    else:
        print(f"Skipping download, reusing {base_dir}")
        os.makedirs(base_dir, exist_ok=True)

    print("\nRunning checks...")
    issues = run_checks(tasks, base_dir, expected_runs=args.expected_runs)
    print_report(issues)

    with open(args.output, "w") as f:
        json.dump(issues, f, indent=2)
    print(f"\nReport saved to {args.output}")

    html_path = args.html
    if html_path is None:
        root, _ = os.path.splitext(args.output)
        html_path = root + ".html"
    if str(html_path).lower() != "none":
        delivery_name = os.path.basename(args.delivery[0]) if args.delivery else os.path.basename(args.output)
        write_html_report(html_path, issues, list(tasks.keys()), delivery_name)
        print(f"HTML report saved to {html_path}")
        print(f"  Open it in your browser: file://{os.path.abspath(html_path)}")

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
