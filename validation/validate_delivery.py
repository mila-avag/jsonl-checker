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
        has_pass_k = False
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            if "pass_rate_opus" in rv or "pass_rate_gemini31" in rv:
                has_pass_k = True
                break
        if not has_pass_k:
            for tt in dl_tests:
                if "pass_rate_opus" in tt or "pass_rate_gemini31" in tt:
                    has_pass_k = True
                    break
        if not has_pass_k:
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

        # ── CHECK 21: Pass rate out of range ──
        for rk, rv in dl_rubrics.items():
            if not isinstance(rv, dict):
                continue
            for fld in ["pass_rate_opus", "pass_rate_gemini31"]:
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
            for fld in ["pass_rate_opus", "pass_rate_gemini31"]:
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

                for fld in ["pass_rate_opus", "pass_rate_gemini31"]:
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
        pk_runs = None
        if pak and isinstance(pak, dict) and pak.get("runs"):
            pk_runs = pak["runs"]
        elif t.get("passk_runs"):
            pk_runs = t["passk_runs"]

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

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
