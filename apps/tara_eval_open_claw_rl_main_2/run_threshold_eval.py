#!/usr/bin/env python3
"""Run verifier eval on rubrics/tests below a pass-rate threshold.

Selects:
  - RUBRIC_FRACTION of rubrics where avg(opus, gemini) < THRESHOLD
  - All unit tests where avg(opus, gemini) < THRESHOLD

Supports two environment sources (auto-detected per task):
  1. S3 env zip  -- tasks that have environment_docker_file + trajectories_url in the JSONL
  2. Local universe zip -- tasks without those URLs; supply --artifact_csv and
     --unit_tests_csv; universe zips must be in --universe_zips_dir

Usage:
    uv run python run_threshold_eval.py \
        --jsonl /path/to/delivery.jsonl \
        --artifact_csv /path/to/artifact_ids.csv \
        --unit_tests_csv /path/to/unit_tests_s3urls.csv \
        --universe_zips_dir ~/Downloads \
        --cache_folder outputs/threshold_cache \
        --workers 25 \
        --threshold 0.62 \
        --rubric_fraction 0.5
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timezone
from typing import Optional

import requests
from rich.console import Console
from rich.progress import track

MODEL = "claude-opus-4-6"
EFFORT = "high"
TIMEOUT_SECONDS = 600
MAX_RETRIES = 3
MAX_TURNS = 35
RANDOM_SEED = 42

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "actions", "agent_pipeline", "eval_schema.json",
)
with open(SCHEMA_PATH) as _f:
    SCHEMA_JSON = json.dumps(json.load(_f), separators=(",", ":"))


def _load_schema_json(name: str) -> str:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "actions", "agent_pipeline", name)
    with open(p) as f:
        return json.dumps(json.load(f), separators=(",", ":"))


COVERAGE_SCHEMA_JSON     = _load_schema_json("schema_coverage.json")
INTERSECTION_SCHEMA_JSON = _load_schema_json("schema_intersection.json")
REDUNDANCY_SCHEMA_JSON   = _load_schema_json("schema_redundancy.json")
SINGLE_GTFA_SCHEMA_JSON  = _load_schema_json("schema_single_gtfa.json")
QUOTED_RUBRIC_SCHEMA_JSON = _load_schema_json("schema_quoted_rubric.json")

console = Console()


# ---------------------------------------------------------------------------
# 1. Select items based on pass-rate threshold
# ---------------------------------------------------------------------------

def _avg_pass_rate(opus, gemini):
    vals = [v for v in [opus, gemini] if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_eval_items(tasks: list[dict], threshold: float, rubric_fraction: float, seed: int,
                     item_whitelist: dict | None = None) -> list[dict]:
    rng = random.Random(seed)
    items = []

    for task in tasks:
        task_id = task["task_id"]
        rubrics = task.get("rubrics", {})
        rr = task.get("rubrics_rating", {})
        unit_tests = task.get("unit_tests", [])

        # --- rubrics ---
        whitelist_ids = set(item_whitelist.get(task_id, [])) if item_whitelist else None
        qualifying_rubrics = []
        for rid, r in rubrics.items():
            o = rr.get(rid, {}).get("pass_rate_opus",   r.get("pass_rate_opus"))
            g = rr.get(rid, {}).get("pass_rate_gemini", r.get("pass_rate_gemini31"))
            avg = _avg_pass_rate(o, g)
            if whitelist_ids is not None:
                if rid in whitelist_ids:
                    qualifying_rubrics.append((rid, r, o, g, avg))
            elif avg is None or avg < threshold:
                qualifying_rubrics.append((rid, r, o, g, avg))

        # sample the requested fraction (skip sampling in whitelist mode)
        if whitelist_ids is not None:
            sampled = qualifying_rubrics
        else:
            n_sample = math.ceil(len(qualifying_rubrics) * rubric_fraction)
            sampled = rng.sample(qualifying_rubrics, min(n_sample, len(qualifying_rubrics)))

        for rid, r, o, g, avg in sampled:
            items.append({
                "task_id":        task_id,
                "item_type":      "rubric",
                "item_id":        rid,
                "criterion":      r.get("criterion", ""),
                "is_positive":    r.get("is_positive", True),
                "importance":     r.get("importance", ""),
                "score":          r.get("score", 0),
                "pass_rate_opus": o,
                "pass_rate_gemini": g,
                "avg_pass_rate":  avg,
                "flag_type":      "low_pass_rate" if avg is not None else "no_pass_rate_data",
                "flag_evidence":  (
                    f"Avg pass rate {avg:.1%} (opus={o}, gemini={g}) below threshold {threshold:.0%}"
                    if avg is not None else
                    f"No pass rate data available (opus={o}, gemini={g})"
                ),
                "prompt":         task.get("prompt", ""),
            })

        # --- unit tests ---
        for t in unit_tests:
            o = t.get("pass_rate_opus")
            g = t.get("pass_rate_gemini31")
            avg = _avg_pass_rate(o, g)
            tname = t["test_name"]
            if whitelist_ids is not None:
                include = tname in whitelist_ids
            else:
                include = avg is not None and avg < threshold
            if include:
                items.append({
                    "task_id":        task_id,
                    "item_type":      "unit_test",
                    "item_id":        t["test_name"],
                    "criterion":      t["test_name"],
                    "is_positive":    True,
                    "importance":     "",
                    "score":          t.get("weight", 0),
                    "pass_rate_opus": o,
                    "pass_rate_gemini": g,
                    "avg_pass_rate":  avg,
                    "flag_type":      "low_pass_rate" if avg is not None else "no_pass_rate_data",
                    "flag_evidence":  (
                        f"Avg pass rate {avg:.1%} (opus={o}, gemini={g}) below threshold {threshold:.0%}"
                        if avg is not None else
                        f"No pass rate data available (opus={o}, gemini={g})"
                    ),
                    "prompt":         task.get("prompt", ""),
                })

    return items


# ---------------------------------------------------------------------------
# 2. Download & extract S3 environments (parallelised)
# ---------------------------------------------------------------------------

def download_and_extract(task_id: str, env_url: str, traj_url: str, cache_folder: str) -> Optional[str]:
    task_dir = os.path.join(cache_folder, "tasks", task_id)
    extracted_dir = os.path.join(task_dir, "extracted")
    traj_dir = os.path.join(task_dir, "trajectories")

    if os.path.isdir(extracted_dir):
        return extracted_dir

    os.makedirs(task_dir, exist_ok=True)

    for url, zip_name, dest in [
        (env_url,  "env.zip",  extracted_dir),
        (traj_url, "traj.zip", traj_dir),
    ]:
        if not url:
            continue
        zip_path = os.path.join(task_dir, zip_name)
        if not os.path.exists(zip_path):
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                console.log(f"[red]Download failed {task_id}/{zip_name}: {e}[/red]")
                continue

        if not os.path.isdir(dest):
            os.makedirs(dest, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.namelist():
                        if member.startswith("__MACOSX/") or member.startswith("."):
                            continue
                        zf.extract(member, dest)
            except Exception as e:
                console.log(f"[red]Extract failed {task_id}/{zip_name}: {e}[/red]")

    return extracted_dir if os.path.isdir(extracted_dir) else None


def cds_to_https(url: str) -> str:
    """Convert a scale-cds:// URI to a public https S3 URL.

    scale-cds://<project>/<hash>#s3/<bucket>  ->
        https://<bucket>.s3.amazonaws.com/<project>/<hash>
    Plain http(s) URLs are returned unchanged.
    """
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


def place_unit_tests_file(extracted_dir: str, unit_tests_url: str, task_id: str = "") -> bool:
    """Download a unit-test file (scale-cds or https) into <extracted_dir>/tests/test_outputs.py.

    Used for S3-env tasks whose env zips do NOT bundle the test file. No-op if already present
    or no URL given. Returns True if a test file is present afterwards.
    """
    if not extracted_dir:
        return False
    tests_dir = os.path.join(extracted_dir, "tests")
    test_file = os.path.join(tests_dir, "test_outputs.py")
    if os.path.exists(test_file):
        return True
    url = cds_to_https(unit_tests_url)
    if not url or url.startswith("https://."):
        return False
    os.makedirs(tests_dir, exist_ok=True)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "zip" in content_type or url.endswith(".zip") or resp.content[:2] == b"PK":
            import io
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for member in zf.namelist():
                    if member.endswith(".py"):
                        zf.extract(member, tests_dir)
            # normalize to test_outputs.py if a single .py landed elsewhere
            if not os.path.exists(test_file):
                pys = [p for p in os.listdir(tests_dir) if p.endswith(".py")]
                if len(pys) == 1:
                    os.rename(os.path.join(tests_dir, pys[0]), test_file)
        else:
            with open(test_file, "wb") as f:
                f.write(resp.content)
    except Exception as e:
        console.log(f"[yellow]Unit test download failed {task_id}: {e}[/yellow]")
    return os.path.exists(test_file)


# ---------------------------------------------------------------------------
# 2b. Local universe zip extractor (for tasks without S3 env URLs)
# ---------------------------------------------------------------------------

def extract_local_universe(
    task_id: str,
    artifact_id: str,
    universe_zips_dir: str,
    unit_tests_url: str,
    cache_folder: str,
) -> Optional[str]:
    """Extract a local universe zip and download the unit test file.

    Returns the extracted universe directory, or None on failure.
    """
    task_dir = os.path.join(cache_folder, "tasks", task_id)
    extracted_dir = os.path.join(task_dir, "extracted")

    if not os.path.isdir(extracted_dir):
        # find the local zip: try exact name first, then glob
        zip_path = os.path.join(universe_zips_dir, f"{artifact_id}.zip")
        if not os.path.exists(zip_path):
            import glob as _glob
            matches = _glob.glob(os.path.join(universe_zips_dir, f"{artifact_id}*.zip"))
            zip_path = matches[0] if matches else None

        if not zip_path or not os.path.exists(zip_path):
            console.log(f"[red]No local zip for {task_id} (artifact {artifact_id})[/red]")
            return None

        os.makedirs(extracted_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.startswith("__MACOSX/") or member.startswith("."):
                        continue
                    zf.extract(member, extracted_dir)
        except Exception as e:
            console.log(f"[red]Extract failed {task_id}: {e}[/red]")
            return None

    # download unit test file if not already present
    tests_dir = os.path.join(extracted_dir, "tests")
    test_file = os.path.join(tests_dir, "test_outputs.py")
    unit_tests_url = cds_to_https(unit_tests_url)
    if unit_tests_url.startswith("https://."):
        unit_tests_url = ""
    if unit_tests_url and not os.path.exists(test_file):
        os.makedirs(tests_dir, exist_ok=True)
        try:
            resp = requests.get(unit_tests_url, timeout=120)
            resp.raise_for_status()
            # unwrap zip if the s3 url is a zip archive
            content_type = resp.headers.get("content-type", "")
            if "zip" in content_type or unit_tests_url.endswith(".zip"):
                import io
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for member in zf.namelist():
                        if member.endswith(".py"):
                            zf.extract(member, tests_dir)
            else:
                with open(test_file, "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            console.log(f"[yellow]Unit test download failed {task_id}: {e}[/yellow]")

    return extracted_dir


# ---------------------------------------------------------------------------
# 3. Prompt builder
# ---------------------------------------------------------------------------

def build_verifier_prompt(item: dict, extract_dir: str,
                          customer_feedback: dict | None = None,
                          strict_customer_eval: bool = False,
                          adversarial_customer_eval: bool = False) -> str:
    d = extract_dir

    # detect environment type: old S3 env zip vs new local universe zip
    is_universe = os.path.isdir(os.path.join(d, "services")) and not os.path.isfile(os.path.join(d, "instruction.md"))

    if is_universe:
        # universe zip: find the top-level folder (artifact_id subdir)
        universe_subdirs = [
            os.path.join(d, name) for name in os.listdir(d)
            if os.path.isdir(os.path.join(d, name)) and name != "tests"
        ]
        universe_root = universe_subdirs[0] if universe_subdirs else d

        env_section = f"""THE TASK ENVIRONMENT IS EXTRACTED AT:
  {d}

Key paths to investigate:

  1. TASK PROMPT (from JSONL):
     {item.get('prompt', '(see prompt field above)')}

  2. UNIVERSE / SERVICE DATA
     - {universe_root}/services/*/data.json  — per-service records (calendar, email, contacts, fintrack, etc.)
     - {universe_root}/metadata/             — universe metadata

  3. UNIT TEST CODE
     - {d}/tests/test_outputs.py             — pytest assertions (if present)

You MUST read the relevant service data files and the unit test code to verify claims. Do not rely on pass rates alone.

IMPORTANT: Do NOT open image files (.jpg, .jpeg, .JPEG, .png, .heic, .HEIC, .gif)."""
    else:
        task_dir = os.path.dirname(d)
        traj_base = os.path.join(task_dir, "trajectories", "anthropic_claude-opus-4_6")
        t = traj_base if os.path.isdir(traj_base) else os.path.join(task_dir, "trajectories")

        env_section = f"""THE TASK ENVIRONMENT IS EXTRACTED AT:
  {d}

Key paths to investigate:

  1. TASK SETUP
     - {d}/instruction.md  — the user's original prompt
     - {d}/task.toml       — task metadata

  2. INPUT DATA
     - {d}/environment/artifacts/inputs/files/
     - {d}/environment/skills/*/SKILL.md
     - {d}/environment/skills/*/references/

  3. DATABASE RECORDS
     - {d}/environment/server/data/*/data.json
     - {d}/environment/server/databases/*.db

  4. VERIFIER CODE
     - {d}/tests/rubric.json       — all rubric criteria with pass rates
     - {d}/tests/test_outputs.py   — pytest assertions
     - {d}/tests/test_weights.json — test weights and pass rates

  5. MODEL TRAJECTORY
     - {d}/conversation_history/Model_A.json — full conversation history

  6. MODEL OUTPUT
     - {t}/workspace/         — files the model produced
     - {t}/verifier/test-stdout.txt — actual pytest output

You MUST read relevant files to verify claims. Do not rely on pass rates alone.

IMPORTANT: Do NOT open image files (.jpg, .jpeg, .JPEG, .png, .heic, .HEIC, .gif)."""

    section1 = f"""You are auditing the correctness of a verifier (rubric criterion or unit test) used to evaluate an AI agent's performance on a task.

This item has a low pass rate (avg {f"{item['avg_pass_rate']:.1%}" if item.get('avg_pass_rate') is not None else "unknown"} across models), suggesting it may be environment-impossible, incorrectly specified, or have a brittle test implementation.

Pass rates across models:
  Claude Opus: {item.get('pass_rate_opus', 'N/A') if item.get('pass_rate_opus') is not None else 'N/A'}
  Gemini 3.1:  {item.get('pass_rate_gemini', 'N/A') if item.get('pass_rate_gemini') is not None else 'N/A'}

{env_section}"""

    if item["item_type"] == "rubric":
        section2 = f"""THE ITEM UNDER REVIEW (rubric):
  Task ID:    {item['task_id']}
  Item ID:    {item['item_id']}
  Criterion:  {item['criterion']}
  Is positive (model should do this): {item['is_positive']}
  Importance: {item['importance']}
  Score:      {item['score']}"""
    else:
        section2 = f"""THE ITEM UNDER REVIEW (unit_test):
  Task ID:   {item['task_id']}
  Test name: {item['item_id']}
  Weight:    {item['score']}

  NOTE: Find the function named "{item['item_id']}" in {d}/tests/test_outputs.py
  and check whether its assertions are correct."""

    # --- Customer feedback section (only injected when running in whitelist mode) ---
    if customer_feedback:
        cf_issue = customer_feedback.get("issue_type", "").strip()
        cf_rationale = customer_feedback.get("customer_rationale", "").strip()
        if adversarial_customer_eval:
            section_customer = f"""CUSTOMER FEEDBACK (Scale AI reviewer flag):
  Issue type reported:  {cf_issue if cf_issue else "(not specified)"}
  Reviewer rationale:   {cf_rationale if cf_rationale else "(not provided)"}

YOUR JOB IS TO ARGUE AGAINST THIS FLAG. Act as a defense attorney for this criterion.
Actively search the environment, test code, and task prompt for ANY evidence that supports
the criterion being correct, fair, or reasonable — even if the customer has a point.
Apply the following logic:
  - If the criterion is defensible under ANY reasonable reading of the task prompt, set verdict = "disagree".
  - If the environment data actually supports the criterion (even partially), set verdict = "disagree".
  - If the customer's complaint is about wording, specificity, or style rather than a factual error, set verdict = "disagree".
  - Only set verdict = "agree" if the criterion is INDEFENSIBLY wrong — e.g., contradicted by environment data, or tests something entirely absent from the prompt with zero supporting justification.
  - Do NOT agree just because a stricter or more lenient version of the criterion might also be valid.
In your final JSON response, set:
  - customer_feedback_verdict: "agree" | "disagree"
  - customer_feedback_reasoning: 1-2 sentences citing specific evidence FROM THE ENVIRONMENT
    or test code that defends or refutes the criterion. Be concrete.
"""
        elif strict_customer_eval:
            section_customer = f"""CUSTOMER FEEDBACK (Scale AI reviewer flag):
  Issue type reported:  {cf_issue if cf_issue else "(not specified)"}
  Reviewer rationale:   {cf_rationale if cf_rationale else "(not provided)"}

IMPORTANT — apply a HIGH standard before agreeing with this flag.
The customer may have misunderstood the task, the environment, or the rubric intent.
Only set customer_feedback_verdict = "agree" if ALL of the following are true:
  1. You found CLEAR, CONCRETE evidence in the environment files or test code supporting the complaint.
  2. The criterion is objectively wrong — not just harder than ideal or worded differently than you'd prefer.
  3. No reasonable reading of the task prompt and environment could justify the criterion as written.
If there is any plausible justification for the criterion, set verdict = "disagree".
In your final JSON response, set:
  - customer_feedback_verdict: "agree" | "disagree"
  - customer_feedback_reasoning: 1-2 sentences citing the SPECIFIC evidence you found (or did not find)
    that determines your verdict.
"""
        else:
            section_customer = f"""CUSTOMER FEEDBACK (Scale AI reviewer flag):
  Issue type reported:  {cf_issue if cf_issue else "(not specified)"}
  Reviewer rationale:   {cf_rationale if cf_rationale else "(not provided)"}

A human reviewer on the Scale AI platform flagged this specific item as problematic.
You must explicitly evaluate whether their concern is valid based on the evidence you find.
In your final JSON response, set:
  - customer_feedback_verdict: "agree" | "disagree" | "partially_agree"
  - customer_feedback_reasoning: 1-2 sentences explaining WHY you agree or disagree with
    the customer's specific concern, citing concrete evidence from the environment or test code.
"""
    else:
        section_customer = ""

    section3 = """INVESTIGATION CHECKLIST:

Flag the verifier ONLY if it is objectively wrong — where no reasonable reading
of the evidence could support its correctness.

1. CRITERION vs TASK MISMATCH: Does it test something the prompt never asked for?
2. CRITERION vs ENVIRONMENT DATA: Does it expect an answer contradicted by actual data?
3. ENVIRONMENT IMPOSSIBILITY: Requires data/capabilities genuinely absent in the environment.
4. POLICY AMBIGUITY: Policy text is genuinely ambiguous and model's interpretation is equally defensible.
5. TEST-CRITERION MISMATCH (unit tests): The test doesn't test what it claims, or has brittle assertions.
6. REDUNDANCY: Fully covered by another criterion/test.
7. PROCESS CHECK (not outcome): The criterion checks HOW the agent worked (steps taken, tools used,
   intermediate actions) rather than WHAT it produced. Rubrics must only evaluate the final output/outcome,
   not the agent's process or reasoning steps."""

    section_quality = """VERIFIER QUALITY CLASSIFICATION (independent of the verdict above):

Separately from whether the low pass rate is justified, classify THIS verifier against the
quality taxonomy below. Populate "quality_issues" with one entry per issue that genuinely applies
(it may be empty if the verifier is clean). Use the EXACT category strings.

IMPORTANT 06/24 PRIORITY:
  - First check for overfitting, underfitting/brittleness, and overlap. These are the most important
    categories for this run.
  - Do NOT flatten an over-specific or too-permissive verifier into generic "incorrect". Use the
    more specific category whenever it applies.
  - Use inaccurate_* only when the verifier checks something misaligned with the prompt, factually
    wrong, harmful to satisfy, or unrelated.

MAJOR issues (category):
  - "not_self_contained": The verifier cannot be assessed against the trajectory/output without
    also having the prompt in hand (it references "the prompt", "as requested", etc. instead of
    stating the concrete expected outcome).
  - "inaccurate_misaligned": The verifier checks for something that does not align with prompt
    requirements. Use a more specific category instead if the root problem is overfitting,
    underfitting, brittleness, or overlap.
  - "inaccurate_factual_error": The verifier contains a factual error or misleading point.
  - "inaccurate_makes_response_worse": The verifier is not an explicit prompt requirement, and
    satisfying it would make the response worse.
  - "inaccurate_unrelated": The verifier is not related to the prompt request.
  - "incorrect_justification": The verifier's stated pass/fail justification is factually wrong,
    unsupported by the prompt, or relies on a requirement the prompt does not actually make.
  - "overfitted": The criterion/test is overly specific, inflexible, or too rigid and would reject
    some valid implementations. Criteria may mention specific answers as examples if phrased as
    examples ("for example", parentheses, "such as", etc.) rather than the only acceptable answer.
  - "underfitted": The verifier is too lenient/permissive and passes outputs that do not actually
    satisfy the requirement. For tests, this includes broad assertions that accept invalid outputs,
    checking only file existence when content matters, or using weak/hash-only checks that do not
    validate the deliverable.
  - "brittle": A unit test is fragile in implementation details: exact strings/format/order/regexes,
    hardcoded incidental values, or environment assumptions cause it to fail valid outputs or pass
    invalid ones. If it is too rigid, also consider "overfitted"; if too permissive, also consider
    "underfitted".
  - "overlapping": This verifier meaningfully duplicates another rubric/test by checking the same
    specific behavior, requirement, or assertion in a way that creates double-counting. Do NOT flag
    overlap merely because two verifiers reference the same task/output; distinct failure modes,
    edge cases, inputs, outputs, or evaluation dimensions are not overlap.

MINOR issues (category):
  - "too_prescriptive": The verifier requires one specific way of solving/presenting the task when
    other valid methods exist, but the problem is narrow and easy to relax.
  - "subjective": The criterion is subjective, vague, or immeasurable (e.g. "good formatting",
    "code must be optimal").

UNIT-TEST FOCUS: when the item under review is a unit_test, read its body from tests/test_outputs.py
and explicitly check for: verifier.py/test code that is overfit to one valid implementation,
too permissive/underfit, brittle exact matching or fragile environment assumptions, overlapping
with another rubric/test, or better represented as a human-judged rubric rather than executable code.

For each entry provide: severity ("major"|"minor"), category (from above), a short description, and
concrete evidence from the prompt/environment/test code. Do NOT use these categories for "missing
criteria" — missing verifiers are handled by a separate task-level check.

RESOLUTION (for the next-round fixer): decide how THIS verifier should be remediated and write a
concrete, self-contained comment a downstream fixer agent can act on without re-reading your reasoning.
  - "keep": the verifier is clean / valid as-is — no change needed. Leave "resolution_comment" empty.
  - "edit": the verifier is salvageable and can be rewritten to be correct while still testing a real
    requirement. Typical for underfitted tests (tighten assertions), overfitted/brittle tests (accept
    valid alternatives while still rejecting invalid outputs), or rubric wording that can be relaxed
    without losing the requirement. In "resolution_comment", give precise edit instructions.
  - "delete": the verifier cannot be salvaged — it tests something unrequested, targets the wrong task,
    asserts an impossible/unsourceable value, or is redundant/overlapping enough that it should simply
    be removed.
  - "delete_and_create_rubric": ONLY for unit_test items. Use this when the pytest is the wrong form
    for the requirement (too semantic, subjective, multi-artifact, or better judged by a human), but
    the underlying requirement is valid and should remain covered as a rubric. In "resolution_comment",
    state why the test should be deleted and provide the exact new rubric criterion to add.
Choose an action whenever there is any quality issue or the verifier is flagged; otherwise "keep"."""

    if customer_feedback:
        customer_fields = """
  "customer_feedback_verdict": "agree" | "disagree" | "partially_agree",
  "customer_feedback_reasoning": "1-2 sentences citing specific evidence for why you agree or disagree with the customer's concern","""
        customer_note = ('- "customer_feedback_verdict"/"customer_feedback_reasoning": '
                         'REQUIRED — explicitly evaluate the customer flag above.\n')
    else:
        customer_fields = """
  "customer_feedback_verdict": "disagree",
  "customer_feedback_reasoning": "No customer feedback was provided for this item.","""
        customer_note = ""

    section4 = f"""After your investigation, respond with ONLY a JSON object:

{{
  "item_id": "{item['item_id']}",
  "task_id": "{item['task_id']}",
  "item_type": "{item['item_type']}",

  "justification_verdict": "accurate",
  "justification_issues": [],

  "verifier_verdict": "correct" | "flagged",
  "verifier_issues": [
    {{
      "check_number": <1-7>,
      "issue_type": "model_did_not_fail" | "environment_impossibility" | "ambiguous_policy" | "test_criterion_mismatch" | "process_check",
      "description": "Specific description of the problem",
      "evidence": "What in the environment/code shows this is wrong"
    }}
  ],

  "quality_issues": [
    {{
      "severity": "major" | "minor",
      "category": "not_self_contained" | "inaccurate_misaligned" | "inaccurate_factual_error" | "inaccurate_makes_response_worse" | "inaccurate_unrelated" | "incorrect_justification" | "overfitted" | "overlapping" | "brittle" | "underfitted" | "too_prescriptive" | "subjective",
      "description": "Specific description of the quality issue",
      "evidence": "Concrete evidence from prompt/environment/test code"
    }}
  ],

  "resolution_action": "keep" | "edit" | "delete" | "delete_and_create_rubric",
  "resolution_comment": "Empty if keep. If edit: concrete instructions for how to rewrite this verifier so it is correct. If delete: why it cannot be salvaged by editing. If delete_and_create_rubric: why the test should be removed plus the exact new rubric criterion to add.",

  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary of your assessment",{customer_fields}
}}

- "justification_verdict" must always be "accurate", "justification_issues" always []
- "verifier_verdict": "flagged" if genuinely problematic, "correct" if the low pass rate is just hard but the verifier is valid
- "quality_issues": classify this verifier against the quality taxonomy; use [] if it is clean
- "resolution_action"/"resolution_comment": "keep" with empty comment if clean; otherwise use an actionable edit/delete/delete_and_create_rubric comment
{customer_note}- No markdown fences, no commentary — only the JSON object."""

    customer_section = f"\n\n{section_customer}" if section_customer else ""
    return f"{section1}\n\n{section2}{customer_section}\n\n{section3}\n\n{section_quality}\n\n{section4}"


# ---------------------------------------------------------------------------
# 3b. Task-level checks: coverage (missing criteria), rubric/pytest
#     intersection, and verifier redundancy. All are env-aware so the model
#     can read tests/test_outputs.py and rubric definitions.
# ---------------------------------------------------------------------------

def _rubric_and_test_lists(task: dict) -> tuple[str, str, int, int]:
    rubrics = task.get("rubrics", {})
    rubric_list = "\n".join(
        f"  {rid}: {r.get('criterion', '')}" for rid, r in rubrics.items()
    ) or "  (none)"
    tests = task.get("unit_tests", [])
    test_list = "\n".join(f"  {t['test_name']}" for t in tests) or "  (none)"
    return rubric_list, test_list, len(rubrics), len(tests)


def build_task_check_items(tasks: list[dict], check_type: str) -> list[dict]:
    """One item per task for a given task-level check type."""
    items = []
    for task in tasks:
        rubric_list, test_list, n_rub, n_test = _rubric_and_test_lists(task)
        items.append({
            "task_id":     task["task_id"],
            "item_type":   check_type,
            "item_id":     check_type,
            "criterion":   check_type,
            "prompt":      task.get("prompt", ""),
            "rubric_list": rubric_list,
            "test_list":   test_list,
            "n_rubrics":   n_rub,
            "n_tests":     n_test,
        })
    return items


def _env_read_hint(extract_dir: str) -> str:
    d = extract_dir
    return f"""You may use Read/Grep/Glob to inspect the actual verifier code:
  - {d}/tests/test_outputs.py   — pytest assertions (the unit tests)
  - {d}/tests/rubric.json       — rubric criteria (if present)
Read the unit test bodies before judging what each test actually checks."""


def build_coverage_prompt(item: dict, extract_dir: str) -> str:
    return f"""You are auditing rubric/test COVERAGE for a task given to an AI agent.

Your job: read the task prompt and the list of verifiers (rubrics + unit tests), then identify
explicit requirements stated in the prompt whose FINAL-OUTCOME is NOT covered by any verifier.

CRITICAL CONSTRAINTS:
  - OUTCOME ONLY. Only consider requirements about the FINAL OUTPUT / RESULT the agent must
    produce. Do NOT flag anything about the agent's process, steps taken, tools used, or
    intermediate actions — those must never be verifier-checked and are out of scope here.
  - Only flag EXPLICIT prompt requirements. Ignore implicit or stylistic expectations.
  - For each gap, decide criticality:
      "critical"     = the requirement is essential to completing the task; missing its verifier
                       means a wrong response could pass.
      "non_critical" = an explicit but non-essential requirement.

THE TASK PROMPT:
{item['prompt']}

THE RUBRIC CRITERIA:
{item['rubric_list']}

THE UNIT TESTS:
{item['test_list']}

{_env_read_hint(extract_dir)}

After your analysis, respond with ONLY a JSON object:

{{
  "item_id": "coverage_check",
  "task_id": "{item['task_id']}",
  "item_type": "coverage_check",
  "verifier_verdict": "complete" | "gaps_found",
  "missing_criteria": [
    {{
      "requirement": "exact quote or close paraphrase of the outcome requirement from the prompt",
      "criticality": "critical" | "non_critical",
      "description": "why no existing verifier covers this final outcome"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary of your outcome-coverage gap analysis"
}}

- "verifier_verdict": "gaps_found" if at least one clear outcome gap exists, otherwise "complete"
- "missing_criteria": empty list [] if verdict is "complete"
- No markdown fences, no commentary — only the JSON object."""


def build_intersection_prompt(item: dict, extract_dir: str) -> str:
    return f"""You are auditing the INTERSECTION between rubric criteria and unit tests (pytests) for a task.

Rubrics and pytests should have little-to-no overlap. An OVERLAP is a case where a rubric criterion
and a unit test check essentially the SAME behavior/outcome (redundant coverage across the two
verifier types). The acceptable limit is at most ONE overlapping case; more than one fails the gate.

THE RUBRIC CRITERIA:
{item['rubric_list']}

THE UNIT TESTS:
{item['test_list']}

{_env_read_hint(extract_dir)}

Compare what each rubric checks against what each unit test actually asserts (read the test bodies).
Count distinct rubric<->test overlapping cases.

Respond with ONLY a JSON object:

{{
  "item_id": "intersection_check",
  "task_id": "{item['task_id']}",
  "item_type": "intersection_check",
  "overlap_count": <integer>,
  "overlaps": [
    {{ "rubric_id": "R#", "test_name": "test_...", "behavior": "the shared behavior both check" }}
  ],
  "gate_pass": <true if overlap_count <= 1 else false>,
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary"
}}

- No markdown fences, no commentary — only the JSON object."""


def build_redundancy_prompt(item: dict, extract_dir: str) -> str:
    return f"""You are auditing VERIFIER REDUNDANCY for a task given to an AI agent.

A redundancy problem exists when MORE THAN TWO verifiers (rubrics and/or unit tests) check EXACTLY
the same behavior/outcome with no meaningful difference. Identify any group of 3 or more verifiers
that are mutually redundant (checking the same thing).

THE RUBRIC CRITERIA:
{item['rubric_list']}

THE UNIT TESTS:
{item['test_list']}

{_env_read_hint(extract_dir)}

Respond with ONLY a JSON object:

{{
  "item_id": "redundancy_check",
  "task_id": "{item['task_id']}",
  "item_type": "redundancy_check",
  "redundant_groups": [
    {{ "members": ["R#", "test_...", "..."], "behavior": "the single behavior all members check" }}
  ],
  "gate_pass": <true if NO group has more than 2 members else false>,
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary"
}}

- Only include a group if it has 3 or more mutually-redundant members.
- No markdown fences, no commentary — only the JSON object."""


def build_single_gtfa_prompt(item: dict, extract_dir: str) -> str:
    return f"""You are auditing whether a task has a SINGLE GROUND-TRUTH FINAL ANSWER (single GTFA).

A task has a single GTFA when there is exactly ONE valid final outcome that any correct
solution must arrive at, and the verifiers (rubrics + unit tests) accept that outcome without
either (a) demanding incidental specifics the prompt never required, or (b) rejecting other
equally-valid final outcomes the prompt allows.

It is NOT a single GTFA when either failure mode is present:
  - "over_constrained": one or more verifiers require specifics BEYOND what the prompt asks for
    (exact wording, formatting, ordering, intermediate steps, or arbitrary values), so a correct
    response that satisfies the prompt could still fail.
  - "multiple_valid_outcomes": the prompt genuinely allows more than one equally-correct final
    outcome, but the verifiers only accept one of them.
  - "both": both failure modes are present.

CRITICAL CONSTRAINTS:
  - OUTCOME ONLY. Judge the FINAL OUTPUT/RESULT, never the agent's process or intermediate steps.
  - Anchor every judgement in what the prompt EXPLICITLY requires. Do not invent requirements.

THE TASK PROMPT:
{item['prompt']}

THE RUBRIC CRITERIA:
{item['rubric_list']}

THE UNIT TESTS:
{item['test_list']}

{_env_read_hint(extract_dir)}

Read the unit test bodies and rubric criteria before judging what each verifier actually enforces.

Respond with ONLY a JSON object:

{{
  "item_id": "single_gtfa_check",
  "task_id": "{item['task_id']}",
  "item_type": "single_gtfa_check",
  "verifier_verdict": "single_gtfa" | "not_single_gtfa",
  "failure_mode": "none" | "over_constrained" | "multiple_valid_outcomes" | "both",
  "offenders": [
    {{ "verifier_id": "R# or test_...", "verifier_type": "rubric" | "unit_test",
       "issue": "why this verifier over-constrains or rejects a valid outcome" }}
  ],
  "alternative_outcomes": [
    {{ "approach": "a different valid way to satisfy the prompt",
       "final_outcome": "the equally-valid final result it produces" }}
  ],
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary"
}}

- "verifier_verdict": "not_single_gtfa" if either failure mode applies, otherwise "single_gtfa".
- "failure_mode": "none" when verdict is "single_gtfa".
- "offenders": [] when verdict is "single_gtfa".
- "alternative_outcomes": [] unless the failure mode involves multiple valid outcomes.
- No markdown fences, no commentary — only the JSON object."""


def build_quoted_prompt(item: dict, extract_dir: str) -> str:
    return f"""You are auditing rubric criteria for OVER-SPECIFIED QUOTED WORDING.

Some rubric criteria wrap text in single or double quotes. Quotes are only appropriate when the
task prompt requires that EXACT wording verbatim in the final output. A quote is a PROBLEM when a
reasonable paraphrase or variation of the quoted text would ALSO be a correct answer — i.e. the
prompt does not strictly require those exact words. Such criteria should be rewritten to drop the
quotes and accept variation.

THE TASK PROMPT:
{item['prompt']}

THE RUBRIC CRITERIA:
{item['rubric_list']}

{_env_read_hint(extract_dir)}

For each rubric criterion that contains quoted wording, decide: does the prompt REQUIRE that exact
text verbatim in the output? If yes, the quote is fine — do not flag it. If a paraphrase/variation
would also be acceptable, flag it as an offender.

Respond with ONLY a JSON object:

{{
  "item_id": "quoted_rubric_check",
  "task_id": "{item['task_id']}",
  "item_type": "quoted_rubric_check",
  "verifier_verdict": "clean" | "quotes_to_remove",
  "offenders": [
    {{ "rubric_id": "R#", "quoted_text": "the exact quoted span in the criterion",
       "reason": "why a paraphrase/variation would also be correct, so the quotes over-specify" }}
  ],
  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary"
}}

- "verifier_verdict": "quotes_to_remove" if at least one criterion has over-specified quotes,
  otherwise "clean".
- "offenders": [] when verdict is "clean".
- No markdown fences, no commentary — only the JSON object."""


_TASK_CHECK_CONF = {
    "coverage_check":      (build_coverage_prompt,     COVERAGE_SCHEMA_JSON,      "COV"),
    "intersection_check":  (build_intersection_prompt, INTERSECTION_SCHEMA_JSON,  "INT"),
    "redundancy_check":    (build_redundancy_prompt,   REDUNDANCY_SCHEMA_JSON,    "RED"),
    "single_gtfa_check":   (build_single_gtfa_prompt,  SINGLE_GTFA_SCHEMA_JSON,   "GTFA"),
    "quoted_rubric_check": (build_quoted_prompt,       QUOTED_RUBRIC_SCHEMA_JSON, "QUOT"),
}


def _build_task_check_result(item, parsed=None, error=None, duration=0.0, cost_usd=0.0, attempts=1):
    parsed = parsed or {}
    return {
        "task_id":            item["task_id"],
        "item_id":            item["item_id"],
        "item_type":          item["item_type"],
        "criterion":          item.get("criterion", ""),
        "pass_rate_opus":     None,
        "pass_rate_gemini":   None,
        "avg_pass_rate":      None,
        "justification_verdict": "accurate",
        "justification_issues":  [],
        "verifier_verdict":   parsed.get("verifier_verdict", parsed.get("gate_pass")),
        "verifier_issues":    [],
        "quality_issues":     [],
        # full structured payload for this task-level check, consumed by postprocessing
        "check_payload":      json.dumps(parsed) if parsed else "",
        "confidence":         parsed.get("confidence"),
        "reasoning":          parsed.get("reasoning"),
        "error":              error,
        "duration_seconds":   duration,
        "cost_usd":           cost_usd,
        "attempts":           attempts,
    }


def run_task_check(item: dict, extract_dir: str, cache_folder: str,
                   disable_cache: bool = False) -> dict:
    """Run a task-level check (coverage/intersection/redundancy), env-aware."""
    check_type = item["item_type"]
    task_id = item["task_id"]
    key = f"{task_id}:{check_type}"
    build_prompt, schema_json, label = _TASK_CHECK_CONF[check_type]
    prompt = build_prompt(item, extract_dir)

    db_path = os.path.join(cache_folder, "threshold_eval_cache.db")
    _ensure_cache_schema(db_path, cache_folder)
    prompt_hash = _sha256(prompt.encode("utf-8"))

    if not disable_cache:
        cached = cache_get(db_path, prompt_hash)
        if cached and not cached.get("error"):
            console.log(f"CACHE  {key}")
            return cached

    results_dir = os.path.join(cache_folder, "results", task_id)
    os.makedirs(results_dir, exist_ok=True)

    result: dict = {}
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            proc = subprocess.run(
                [
                    "claude",
                    "-p", prompt,
                    "--model", MODEL,
                    "--effort", EFFORT,
                    "--output-format", "stream-json",
                    "--verbose",
                    "--strict-mcp-config",
                    "--json-schema", schema_json,
                    "--max-turns", "8",
                    "--permission-mode", "auto",
                    "--tools", "Bash,Read,Glob,Grep,NotebookRead",
                ],
                cwd=extract_dir,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            result = _build_task_check_result(item, error="timeout", duration=duration, attempts=attempt)
            continue
        except Exception as e:
            result = _build_task_check_result(item, error=str(e), duration=time.time()-start, attempts=attempt)
            continue

        duration = time.time() - start
        trace_events, parsed, cost_usd = _extract_from_stream(proc.stdout or "")

        trace_path = os.path.join(results_dir, f"{check_type}.trace.json")
        with open(trace_path, "w") as tf:
            json.dump(trace_events, tf, indent=2)

        if proc.returncode != 0 or parsed is None:
            result = _build_task_check_result(item, error=f"exit {proc.returncode} / no JSON",
                                              duration=duration, cost_usd=cost_usd, attempts=attempt)
            if attempt < MAX_RETRIES:
                console.log(f"RETRY  [{duration:.0f}s] {key} attempt {attempt}")
            continue

        result = _build_task_check_result(item, parsed=parsed, duration=duration,
                                          cost_usd=cost_usd, attempts=attempt)
        console.log(f"{label:<3}    [{duration:.0f}s] {key} → {result['verifier_verdict']} conf={result['confidence']}")
        with open(os.path.join(results_dir, f"{check_type}.json"), "w") as rf:
            json.dump(result, rf, indent=2)
        cache_put(db_path, prompt_hash, result)
        return result

    console.log(f"FAIL   {key} after {MAX_RETRIES} attempts")
    return result


# ---------------------------------------------------------------------------
# 4. Cache
# ---------------------------------------------------------------------------

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS threshold_eval_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key     TEXT NOT NULL UNIQUE,
    agent         TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    context_hash  TEXT NOT NULL,
    result        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""
_cache_init_lock = threading.Lock()
_cache_initialized: set[str] = set()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_cache_schema(db_path: str, cache_folder: str):
    if db_path in _cache_initialized:
        return
    with _cache_init_lock:
        if db_path in _cache_initialized:
            return
        os.makedirs(cache_folder, exist_ok=True)
        with _cache_connect(db_path) as conn:
            conn.executescript(_CACHE_SCHEMA)
            conn.commit()
        _cache_initialized.add(db_path)


def cache_get(db_path: str, prompt_hash: str) -> Optional[dict]:
    try:
        with _cache_connect(db_path) as conn:
            row = conn.execute(
                "SELECT result FROM threshold_eval_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()
        if row:
            return json.loads(row["result"])
    except Exception:
        pass
    return None


def cache_put(db_path: str, prompt_hash: str, result: dict):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(result, separators=(",", ":"))
    cache_key = _sha256(f"claude-{MODEL}:{prompt_hash}".encode())
    for attempt in range(5):
        try:
            with _cache_connect(db_path) as conn:
                conn.execute(
                    """INSERT INTO threshold_eval_cache
                       (cache_key, agent, prompt_hash, context_hash, result, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(cache_key) DO UPDATE SET
                           result = excluded.result, created_at = excluded.created_at""",
                    (cache_key, f"claude-{MODEL}", prompt_hash, "", payload, created_at),
                )
                conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt == 4:
                raise
            time.sleep(0.1 * (2 ** attempt))


# ---------------------------------------------------------------------------
# 5. Claude Code CLI runner
# ---------------------------------------------------------------------------

def _try_parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(text, str):
        return None
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def _extract_from_stream(raw_stdout: str):
    trace_events = []
    structured_output = None
    result_text = None
    cost_usd = 0.0
    for line in raw_stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            trace_events.append(event)
            if event.get("type") == "result":
                if event.get("structured_output"):
                    structured_output = event["structured_output"]
                result_text = event.get("result", "")
                cost_usd = event.get("modelUsage", {}).get(MODEL, {}).get("costUSD", 0)
        except json.JSONDecodeError:
            trace_events.append({"type": "raw", "content": line})
    if structured_output:
        return trace_events, structured_output, cost_usd
    if result_text:
        parsed = _try_parse_json(result_text)
        if parsed:
            return trace_events, parsed, cost_usd
    return trace_events, None, cost_usd


def _build_result(item, parsed=None, error=None, duration=0.0, cost_usd=0.0, attempts=1):
    parsed = parsed or {}
    return {
        "task_id":            item["task_id"],
        "item_id":            item["item_id"],
        "item_type":          item["item_type"],
        "criterion":          item.get("criterion", ""),
        "pass_rate_opus":     item.get("pass_rate_opus"),
        "pass_rate_gemini":   item.get("pass_rate_gemini"),
        "avg_pass_rate":      item.get("avg_pass_rate"),
        "justification_verdict": parsed.get("justification_verdict", "accurate"),
        "justification_issues":  parsed.get("justification_issues", []),
        "verifier_verdict":   parsed.get("verifier_verdict"),
        "verifier_issues":    parsed.get("verifier_issues", []),
        "quality_issues":     parsed.get("quality_issues", []),
        "resolution_action":  parsed.get("resolution_action", "keep"),
        "resolution_comment": parsed.get("resolution_comment", ""),
        "confidence":         parsed.get("confidence"),
        "reasoning":          parsed.get("reasoning"),
        # customer feedback fields (present when run in whitelist/feedback mode)
        "customer_issue_type":          item.get("customer_issue_type", ""),
        "customer_rationale":           item.get("customer_rationale", ""),
        "customer_feedback_verdict":    parsed.get("customer_feedback_verdict", ""),
        "customer_feedback_reasoning":  parsed.get("customer_feedback_reasoning", ""),
        "error":              error,
        "duration_seconds":   duration,
        "cost_usd":           cost_usd,
        "attempts":           attempts,
    }


def run_single_eval(item: dict, prompt: str, extract_dir: str,
                    cache_folder: str, disable_cache: bool = False) -> dict:
    task_id = item["task_id"]
    item_id = item["item_id"]
    key = f"{task_id}:{item_id}"

    db_path = os.path.join(cache_folder, "threshold_eval_cache.db")
    _ensure_cache_schema(db_path, cache_folder)
    prompt_hash = _sha256(prompt.encode("utf-8"))

    if not disable_cache:
        cached = cache_get(db_path, prompt_hash)
        if cached and not cached.get("error"):
            console.log(f"CACHE  {key}")
            return cached

    results_dir = os.path.join(cache_folder, "results", task_id)
    os.makedirs(results_dir, exist_ok=True)

    result: dict = {}
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            proc = subprocess.run(
                [
                    "claude",
                    "-p", prompt,
                    "--model", MODEL,
                    "--effort", EFFORT,
                    "--output-format", "stream-json",
                    "--verbose",
                    "--strict-mcp-config",
                    "--json-schema", SCHEMA_JSON,
                    "--max-turns", str(MAX_TURNS),
                    "--permission-mode", "auto",
                    "--tools", "Bash,Read,Glob,Grep,NotebookRead",
                ],
                cwd=extract_dir,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            result = _build_result(item, error=f"timeout after {TIMEOUT_SECONDS}s",
                                   duration=duration, attempts=attempt)
            continue
        except Exception as e:
            duration = time.time() - start
            result = _build_result(item, error=str(e), duration=duration, attempts=attempt)
            continue

        duration = time.time() - start
        trace_events, parsed, cost_usd = _extract_from_stream(proc.stdout or "")

        # save trace
        trace_path = os.path.join(results_dir, f"{item_id}.trace.json")
        with open(trace_path, "w") as tf:
            json.dump(trace_events, tf, indent=2)

        # Salvage a valid structured output even if the process exited non-zero
        # (e.g. a rate-limit event can make `claude` return a non-zero code AFTER
        # it already emitted the StructuredOutput tool call). Re-running these is
        # expensive and pointless when we already have the verdict.
        if parsed is None:
            if proc.returncode != 0:
                result = _build_result(item, error=f"exit code {proc.returncode}",
                                       duration=duration, cost_usd=cost_usd, attempts=attempt)
                if attempt < MAX_RETRIES:
                    console.log(f"RETRY  [{duration:.0f}s] {key} attempt {attempt}: exit {proc.returncode}")
                continue
            result = _build_result(item, error="no structured output",
                                   duration=duration, cost_usd=cost_usd, attempts=attempt)
            if attempt < MAX_RETRIES:
                console.log(f"RETRY  [{duration:.0f}s] {key} attempt {attempt}: no JSON")
            continue

        result = _build_result(item, parsed=parsed, duration=duration,
                               cost_usd=cost_usd, attempts=attempt)
        console.log(
            f"OK     [{duration:.0f}s] {key} → ver={result['verifier_verdict']} conf={result['confidence']}"
        )
        with open(os.path.join(results_dir, f"{item_id}.json"), "w") as rf:
            json.dump(result, rf, indent=2)
        cache_put(db_path, prompt_hash, result)
        return result

    console.log(f"FAIL   {key} after {MAX_RETRIES} attempts")
    with open(os.path.join(results_dir, f"{item_id}.json"), "w") as rf:
        json.dump(result, rf, indent=2)
    return result


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",              required=True)
    parser.add_argument("--cache_folder",       default="outputs/threshold_cache")
    parser.add_argument("--output",             default="results_threshold.csv")
    parser.add_argument("--workers",            type=int,   default=25)
    parser.add_argument("--dl_workers",         type=int,   default=10)
    parser.add_argument("--threshold",          type=float, default=0.62)
    parser.add_argument("--rubric_fraction",    type=float, default=0.5)
    parser.add_argument("--seed",               type=int,   default=RANDOM_SEED)
    parser.add_argument("--disable_cache",      action="store_true")
    parser.add_argument("--skip_task_checks",   action="store_true",
                        help="skip Step 3a task-level checks (use cached ones); go straight to per-item")
    parser.add_argument("--dry_run",            action="store_true")
    # local universe zip inputs (for tasks without S3 env URLs)
    parser.add_argument("--artifact_csv",       default=None,
                        help="CSV with columns TASK_ID, ARTIFACT_ID mapping tasks to universe zips")
    parser.add_argument("--unit_tests_csv",     default=None,
                        help="CSV with columns task_id, unit_tests_s3Url")
    parser.add_argument("--universe_zips_dir",  default=os.path.expanduser("~/Downloads"),
                        help="Directory containing openclaw-*-universe-*.zip files")
    parser.add_argument("--item_whitelist",     default=None,
                        help="JSON file mapping task_id -> [item_ids] to evaluate (skips threshold filtering)")
    parser.add_argument("--customer_feedback_json", default=None,
                        help="JSON file mapping task_id -> {item_id -> {issue_type, customer_rationale}} for prompt injection")
    parser.add_argument("--strict_customer_eval", action="store_true",
                        help="Apply a high skeptical bar before agreeing with customer feedback flags")
    parser.add_argument("--adversarial_customer_eval", action="store_true",
                        help="Actively argue against customer flags — only agree if criterion is indefensibly wrong")
    args = parser.parse_args()

    console.rule("[bold]Step 1: Load JSONL & select items[/bold]")
    tasks = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    console.log(f"Loaded {len(tasks)} tasks")

    item_whitelist: dict | None = None
    if args.item_whitelist:
        with open(args.item_whitelist) as _wf:
            item_whitelist = json.load(_wf)
        console.log(f"Whitelist loaded: {sum(len(v) for v in item_whitelist.values())} items across {len(item_whitelist)} tasks")

    customer_feedback_map: dict | None = None
    if args.customer_feedback_json:
        with open(args.customer_feedback_json) as _cf:
            customer_feedback_map = json.load(_cf)
        console.log(f"Customer feedback loaded for {len(customer_feedback_map)} tasks")

        # Load feedback schema (has customer_feedback_verdict/reasoning fields)
        _feedback_schema_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "actions", "agent_pipeline", "eval_schema_with_feedback.json",
        )
        import importlib
        global SCHEMA_JSON
        with open(_feedback_schema_path) as _sf:
            SCHEMA_JSON = json.dumps(json.load(_sf), separators=(",", ":"))
        console.log("Using feedback-extended schema")

    items = build_eval_items(tasks, args.threshold, args.rubric_fraction, args.seed,
                             item_whitelist=item_whitelist)

    # Attach customer feedback fields to each item
    if customer_feedback_map:
        for it in items:
            tid = it["task_id"]
            iid = it["item_id"]
            fb = customer_feedback_map.get(tid, {}).get(iid, {})
            it["customer_issue_type"] = fb.get("issue_type", "")
            it["customer_rationale"]  = fb.get("customer_rationale", "")

    rubric_items = [i for i in items if i["item_type"] == "rubric"]
    test_items   = [i for i in items if i["item_type"] == "unit_test"]
    if item_whitelist:
        console.log(f"Selected {len(items)} items from whitelist: {len(rubric_items)} rubrics, {len(test_items)} unit tests")
    else:
        console.log(
            f"Selected {len(items)} items: {len(rubric_items)} rubrics "
            f"({args.rubric_fraction:.0%} of <{args.threshold:.0%}), "
            f"{len(test_items)} unit tests (<{args.threshold:.0%})"
        )

    if args.dry_run:
        by_task = {}
        for it in items:
            by_task.setdefault(it["task_id"], {"rubric": 0, "unit_test": 0})[it["item_type"]] += 1
        for tid, counts in sorted(by_task.items()):
            console.log(f"  {tid}: {counts['rubric']} rubrics, {counts['unit_test']} tests")
        console.log(f"\nTotal: {len(items)} items across {len(by_task)} tasks")
        return

    # load artifact_id and unit_tests_url lookups if provided
    artifact_id_by_task: dict[str, str] = {}
    unit_tests_url_by_task: dict[str, str] = {}
    if args.artifact_csv:
        csv.field_size_limit(10_000_000)
        with open(args.artifact_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # handle both column name styles
                tid = row.get("TASK_ID") or row.get("TASKID", "")
                aid = (row.get("ARTIFACT_ID") or row.get("L8.METADATA:ARTIFACT_ID", "")).strip('"')
                if tid and aid:
                    artifact_id_by_task[tid] = aid
        console.log(f"Loaded artifact IDs for {len(artifact_id_by_task)} tasks")
    if args.unit_tests_csv:
        with open(args.unit_tests_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                unit_tests_url_by_task[row["task_id"]] = row["unit_tests_s3Url"]
        console.log(f"Loaded unit test URLs for {len(unit_tests_url_by_task)} tasks")

    console.rule("[bold]Step 2: Resolve environments (parallel)[/bold]")
    task_ids_needed = sorted({it["task_id"] for it in items})
    task_by_id = {t["task_id"]: t for t in tasks}
    env_dirs: dict[str, Optional[str]] = {}

    def _resolve_env(tid):
        task = task_by_id[tid]
        env_url  = task.get("environment_docker_file", "")
        traj_url = task.get("trajectories_url", "")
        if env_url:
            ed = download_and_extract(tid, env_url, traj_url, args.cache_folder)
            # S3 env zips may not bundle the unit tests; fetch them separately if a URL was given
            ut_url = unit_tests_url_by_task.get(tid, "")
            if ed and ut_url:
                place_unit_tests_file(ed, ut_url, tid)
            return tid, ed
        # fall back to local universe zip
        artifact_id = artifact_id_by_task.get(tid)
        if not artifact_id:
            console.log(f"[yellow]No artifact_id for {tid}, skipping[/yellow]")
            return tid, None
        return tid, extract_local_universe(
            tid,
            artifact_id,
            args.universe_zips_dir,
            unit_tests_url_by_task.get(tid, ""),
            args.cache_folder,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.dl_workers) as pool:
        futs = {pool.submit(_resolve_env, tid): tid for tid in task_ids_needed}
        for fut in track(concurrent.futures.as_completed(futs),
                         total=len(futs), description="Resolving envs..."):
            tid, env_dir = fut.result()
            env_dirs[tid] = env_dir

    ok = sum(1 for v in env_dirs.values() if v)
    console.log(f"Environments ready: {ok}/{len(task_ids_needed)}")
    items = [it for it in items if env_dirs.get(it["task_id"])]
    console.log(f"Items with environments: {len(items)}")

    console.rule("[bold]Step 3a: Run task-level checks (coverage, intersection, redundancy, single-GTFA, quoted-rubric)[/bold]")
    # Only run task-level checks on tasks whose environment resolved.
    tasks_with_env = [t for t in tasks if env_dirs.get(t["task_id"])]
    task_check_items = []
    if not args.skip_task_checks:
        for check_type in ("coverage_check", "intersection_check", "redundancy_check",
                           "single_gtfa_check", "quoted_rubric_check"):
            task_check_items.extend(build_task_check_items(tasks_with_env, check_type))
    else:
        console.log("[yellow]--skip_task_checks: skipping Step 3a (using cached task-level results)[/yellow]")

    coverage_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        cov_futures = {
            pool.submit(run_task_check, it, env_dirs[it["task_id"]],
                        args.cache_folder, args.disable_cache): it
            for it in task_check_items
        }
        for fut in track(concurrent.futures.as_completed(cov_futures),
                         total=len(cov_futures), description="Task-level checks..."):
            try:
                coverage_results.append(fut.result())
            except Exception as e:
                it = cov_futures[fut]
                console.log(f"[red]Task-check error {it['task_id']}:{it['item_type']}: {e}[/red]")
                coverage_results.append(_build_task_check_result(it, error=str(e)))

    gaps = sum(1 for r in coverage_results
               if r.get("item_type") == "coverage_check" and r.get("verifier_verdict") == "gaps_found")
    console.log(f"Task-level checks done: {len(coverage_results)} checks across {len(tasks_with_env)} tasks "
                f"({gaps} with coverage gaps)")

    console.rule("[bold]Step 3b: Run per-rubric/test evals[/bold]")
    results = []
    start_time = time.time()

    def _eval(item):
        ed = env_dirs[item["task_id"]]
        cf = None
        if customer_feedback_map:
            tid = item["task_id"]
            iid = item["item_id"]
            cf = customer_feedback_map.get(tid, {}).get(iid) or None
        prompt = build_verifier_prompt(item, ed, customer_feedback=cf,
                                       strict_customer_eval=args.strict_customer_eval,
                                       adversarial_customer_eval=args.adversarial_customer_eval)
        return run_single_eval(item, prompt, ed, args.cache_folder, args.disable_cache)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval, it): it for it in items}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                item = futures[fut]
                console.log(f"[red]Error {item['task_id']}:{item['item_id']}: {e}[/red]")
                results.append(_build_result(item, error=str(e)))
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed * 60
                remaining = (len(items) - done) / (done / elapsed) if done else 0
                console.log(
                    f"Progress: {done}/{len(items)} | "
                    f"{rate:.1f}/min | "
                    f"~{remaining/3600:.1f}h remaining"
                )

    console.rule("[bold]Step 4: Write results[/bold]")
    output_path = os.path.join(args.cache_folder, args.output)
    all_results = coverage_results + results
    fieldnames = [
        "task_id", "item_type", "item_id", "criterion",
        "pass_rate_opus", "pass_rate_gemini", "avg_pass_rate",
        "verifier_verdict", "verifier_issues", "quality_issues",
        "resolution_action", "resolution_comment", "confidence", "reasoning",
        "justification_verdict", "justification_issues",
        "check_payload",
        "customer_issue_type", "customer_rationale",
        "customer_feedback_verdict", "customer_feedback_reasoning",
        "error", "duration_seconds", "cost_usd", "attempts",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x["task_id"], x["item_type"], x["item_id"])):
            row = dict(r)
            for k in ("verifier_issues", "justification_issues", "quality_issues"):
                if isinstance(row.get(k), list):
                    row[k] = json.dumps(row[k])
            writer.writerow(row)

    flagged = sum(1 for r in results if r.get("verifier_verdict") == "flagged")
    correct = sum(1 for r in results if r.get("verifier_verdict") == "correct")
    errors  = sum(1 for r in results if r.get("error"))
    total_cost = sum(r.get("cost_usd", 0) or 0 for r in all_results)

    console.rule("[bold]Summary[/bold]")
    console.log(f"Task-level checks: {len(coverage_results)} ({gaps} coverage gaps)")
    console.log(f"Rubric/test evals:{len(results)}")
    console.log(f"  Flagged:        {flagged}  ({flagged/len(results)*100:.1f}%)" if results else "")
    console.log(f"  Correct:        {correct}")
    console.log(f"  Errors:         {errors}")
    console.log(f"  Total cost:     ${total_cost:.2f}")
    console.log(f"Results → {output_path}")


if __name__ == "__main__":
    main()
