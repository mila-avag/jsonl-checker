#!/usr/bin/env python3
"""Rewrite agents — apply the verifier-quality resolution worklist.

Standalone (does NOT import or modify the tara eval). It consumes the
`resolution_results.csv` produced by the quality eval and, for every verifier the
eval marked `edit` or `delete`, produces concrete fixes:

  RUBRICS
    - edits   -> rubric_edits.csv     (1 row per edit: task_id, old_rubric, new_rubric)
    - deletes -> rubric_deletes.csv    (task_id, rubric_to_delete)   [no agent needed]

  UNIT TESTS (edits and/or deletes, batched per task)
    - one rewritten test file per task -> unit_tests/<task_id>.py
      (edited functions rewritten, deleted functions removed, everything else kept)

Each rewrite runs an agent (`claude -p ... --json-schema`) with read access to the
task environment that the eval already extracted under
  <cache_folder>/tasks/<task_id>/extracted
so the agent can ground its fix in the prompt, the data, and the original verifier code.

Usage:
    python run_rewrite_agents.py \
        --resolution_csv data/outputs/delivery_quality_0620/resolution_results.csv \
        --cache_folder   data/outputs/delivery_quality_0620/cache \
        --output_dir     data/outputs/delivery_quality_0620/rewrites \
        --workers 10
"""

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

from rich.console import Console

csv.field_size_limit(10_000_000)
console = Console()

# --- agent settings (kept local; mirrors the eval's settings) ---
MODEL = "claude-opus-4-6"
EFFORT = "high"
TIMEOUT_SECONDS = 600
MAX_RETRIES = 3
MAX_TURNS = 20
TOOLS = "Bash,Read,Glob,Grep,NotebookRead"

RUBRIC_SCHEMA = json.dumps({
    "type": "object",
    "required": ["new_rubric", "notes"],
    "additionalProperties": False,
    "properties": {
        "new_rubric": {"type": "string"},
        "notes": {"type": "string"},
    },
}, separators=(",", ":"))

TESTFILE_SCHEMA = json.dumps({
    "type": "object",
    "required": ["new_file_content", "summary"],
    "additionalProperties": False,
    "properties": {
        "new_file_content": {"type": "string"},
        "summary": {"type": "string"},
    },
}, separators=(",", ":"))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _extract_from_stream(raw_stdout: str):
    """Pull the structured JSON object (and cost) out of stream-json output."""
    structured = None
    result_text = ""
    cost = 0.0
    for line in (raw_stdout or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            if ev.get("structured_output"):
                structured = ev["structured_output"]
            result_text = ev.get("result", "") or result_text
            cost = ev.get("modelUsage", {}).get(MODEL, {}).get("costUSD", 0) or cost
    if structured:
        return structured, cost
    if result_text:
        try:
            return json.loads(result_text), cost
        except json.JSONDecodeError:
            # try to salvage a JSON object substring
            i, j = result_text.find("{"), result_text.rfind("}")
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(result_text[i:j + 1]), cost
                except json.JSONDecodeError:
                    pass
    return None, cost


def run_agent(prompt: str, schema_json: str, extract_dir: str, key: str,
              cache_folder: str, disable_cache: bool, cache_only: bool = False):
    """Spawn one rewrite agent; returns (parsed_dict_or_None, error, cost)."""
    db_dir = os.path.join(cache_folder, "rewrite_cache")
    os.makedirs(db_dir, exist_ok=True)
    cache_file = os.path.join(db_dir, _sha256(prompt) + ".json")
    if not disable_cache and os.path.exists(cache_file):
        try:
            c = json.load(open(cache_file))
            console.log(f"CACHE  {key}")
            return c.get("parsed"), c.get("error"), 0.0
        except Exception:
            pass

    if cache_only:
        console.log(f"SKIP   {key} (not cached)")
        return None, "skipped: not cached", 0.0

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt, "--model", MODEL, "--effort", EFFORT,
                 "--output-format", "stream-json", "--verbose", "--strict-mcp-config",
                 "--json-schema", schema_json, "--max-turns", str(MAX_TURNS),
                 "--permission-mode", "auto", "--tools", TOOLS],
                cwd=extract_dir, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {TIMEOUT_SECONDS}s"
            continue
        except Exception as e:
            last_err = str(e)
            continue
        dur = time.time() - start
        if proc.returncode != 0:
            last_err = f"exit code {proc.returncode}"
            if attempt < MAX_RETRIES:
                console.log(f"RETRY  [{dur:.0f}s] {key} attempt {attempt}: {last_err}")
            continue
        parsed, cost = _extract_from_stream(proc.stdout or "")
        if parsed is None:
            last_err = "no structured output"
            if attempt < MAX_RETRIES:
                console.log(f"RETRY  [{dur:.0f}s] {key} attempt {attempt}: {last_err}")
            continue
        if not disable_cache:
            json.dump({"parsed": parsed, "error": None}, open(cache_file, "w"))
        console.log(f"OK     [{dur:.0f}s] {key}")
        return parsed, None, cost
    console.log(f"FAIL   {key}: {last_err}")
    return None, last_err, 0.0


def extracted_dir(cache_folder: str, task_id: str) -> str | None:
    d = os.path.join(cache_folder, "tasks", task_id, "extracted")
    return d if os.path.isdir(d) else None


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def _quote_rule(categories: str) -> str:
    """Extra, IMPOSSIBLE-TO-MISS instructions when the flag is about over-specified quotes."""
    if "quoted_overspecific" in (categories or ""):
        return (
            "\n"
            "  *** QUOTED-RUBRIC FIX — THIS IS THE WHOLE POINT OF THIS EDIT ***\n"
            "  This criterion was flagged because it pins down exact wording inside single (') or\n"
            "  double (\") quotes, but the prompt accepts reasonable VARIATIONS of that wording.\n"
            "  You MUST:\n"
            "    1. DELETE every single (') and double (\") quote character from the criterion.\n"
            "    2. REMOVE the verbatim quoted phrase and instead DESCRIBE the required outcome so\n"
            "       that paraphrases, synonyms, reordering, and differences in casing/spacing/\n"
            "       punctuation ALL satisfy it.\n"
            "    3. Preserve the substantive requirement — only the brittle exact-string matching goes.\n"
            "  HARD CONSTRAINT: your `new_rubric` text MUST NOT contain any ' or \" character at all.\n"
            "  (Exception: if the exact string is genuinely mandatory — e.g. an exact filename or code —\n"
            "   it should not have been flagged; in that rare case keep the requirement but still avoid\n"
            "   quote characters by naming the value plainly.)\n"
        )
    # General hygiene for every other rewrite: do not introduce brittle quoted strings.
    return ("\n  - Avoid wrapping expected wording in single/double quotes unless that EXACT string is\n"
            "    strictly required; prefer describing the outcome so variations also pass.\n")


def rubric_edit_prompt(d: str, criterion: str, severity: str, categories: str,
                       comment: str) -> str:
    return f"""You are fixing a SINGLE rubric criterion used to grade an AI agent's output on a task.

The task environment is extracted at:
  {d}
Ground your fix by reading:
  - {d}/instruction.md            — the user's task prompt (the source of truth for what is required)
  - {d}/tests/rubric.json          — the full rubric (for context / avoiding duplication)
  - {d}/environment/                — input data, records, skills the agent had access to
  - {d}/conversation_history/       — what the model actually did (if helpful)

THE RUBRIC CRITERION TO FIX (verbatim):
  {criterion}

WHY IT WAS FLAGGED — severity={severity}, categories=[{categories}]:
  {comment}

Rewrite the criterion so it is correct and aligned with the prompt, while still testing a REAL,
prompt-required outcome. Requirements for the rewrite:
  - Keep it a single, self-contained, OUTCOME-based criterion (never check the agent's process/steps).
  - Fix exactly the flagged problem; do not broaden it to test things the prompt never asked for.
  - State the concrete expected outcome (values, artifacts) so it can be judged without the prompt.
  - If the criterion has a polarity/weight notion, keep its intent unless the flag is about polarity.
{_quote_rule(categories)}
Respond with ONLY a JSON object: {{"new_rubric": "<the corrected criterion text>", "notes": "<1-2 sentences on what you changed>"}}."""


def rubric_create_prompt(d: str, test_name: str, severity: str, categories: str,
                         comment: str) -> str:
    return f"""You are replacing a flawed unit test with a human-judged rubric criterion.

The task environment is extracted at:
  {d}
Ground your new rubric by reading:
  - {d}/instruction.md            — the user's task prompt (the source of truth)
  - {d}/tests/test_outputs.py     — the flawed pytest function to remove
  - {d}/tests/rubric.json         — existing rubrics, so you avoid duplication
  - {d}/environment/              — input data and records the agent had access to

UNIT TEST TO DELETE:
  {test_name}

WHY IT SHOULD BECOME A RUBRIC — severity={severity}, categories=[{categories}]:
  {comment}

Create ONE new rubric criterion that preserves the valid underlying requirement but is better
judged as a rubric than as executable pytest. Requirements:
  - It must be a single, self-contained, outcome-based criterion.
  - It must align only with explicit prompt requirements and available environment data.
  - It must not duplicate an existing rubric.
  - It must be flexible enough to accept valid implementations, while still rejecting invalid ones.
  - If exact values/examples are useful, phrase them as examples unless the prompt requires them exactly.

Respond with ONLY a JSON object: {{"new_rubric": "<the new rubric criterion to add>", "notes": "<1-2 sentences explaining why this replaces the test>"}}."""


def testfile_prompt(d: str, edits: list[dict], deletes: list[dict]) -> str:
    def fmt(items):
        return "\n".join(f"  - {it['item_id']}: {it['resolution_comment']}" for it in items) or "  (none)"
    return f"""You are fixing the unit-test file for a task used to grade an AI agent's output.

The task environment is extracted at:
  {d}
The test file to rewrite is:
  {d}/tests/test_outputs.py
Read that file IN FULL first. Also read {d}/instruction.md (the task prompt) and the relevant
{d}/environment/ data so your edits assert the correct values.

Apply EXACTLY the following changes and NOTHING else:

EDIT these test functions — rewrite each so it correctly tests its requirement per the note.
Keep the SAME function name and signature; fix the assertion/logic only:
{fmt(edits)}

DELETE these test functions entirely — remove the whole function (and any now-unused helper that
ONLY it used). Do not leave a stub:
{fmt(deletes)}

Rules:
  - Leave every OTHER test function in the file unchanged.
  - Preserve all imports, module-level helpers, and fixtures still in use.
  - The output must be a COMPLETE, runnable Python file (the entire file, not a diff).

Respond with ONLY a JSON object: {{"new_file_content": "<the entire corrected file>", "summary": "<what you changed>"}}."""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution_csv", required=True)
    ap.add_argument("--cache_folder", required=True,
                    help="the eval cache folder containing tasks/<id>/extracted")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N agent jobs (debug)")
    ap.add_argument("--disable_cache", action="store_true")
    ap.add_argument("--cache_only", action="store_true",
                    help="Only emit results already in cache; skip (don't spawn) uncached jobs.")
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    tests_out = os.path.join(out, "unit_tests")
    os.makedirs(tests_out, exist_ok=True)

    # ---- load worklist ----
    rows = list(csv.DictReader(open(args.resolution_csv, newline="", encoding="utf-8")))
    rubric_edits = [r for r in rows if r["item_type"] == "rubric" and r["resolution_action"] == "edit"]
    rubric_deletes = [r for r in rows if r["item_type"] == "rubric" and r["resolution_action"] == "delete"]
    rubric_creates = [r for r in rows if r["item_type"] == "unit_test" and r["resolution_action"] == "delete_and_create_rubric"]
    test_changes = [r for r in rows if r["item_type"] == "unit_test" and r["resolution_action"] in ("edit", "delete", "delete_and_create_rubric")]

    tests_by_task: dict[str, dict] = defaultdict(lambda: {"edit": [], "delete": []})
    for r in test_changes:
        if r["resolution_action"] == "delete_and_create_rubric":
            tests_by_task[r["task_id"]]["delete"].append(r)
        else:
            tests_by_task[r["task_id"]][r["resolution_action"]].append(r)

    console.rule("[bold]Rewrite worklist[/bold]")
    console.log(f"rubric edits: {len(rubric_edits)} | rubric deletes: {len(rubric_deletes)} | "
                f"rubric creates: {len(rubric_creates)} | "
                f"tasks with test changes: {len(tests_by_task)} "
                f"({len(test_changes)} test funcs)")

    # ---- rubric deletes: no agent, write directly (with justification) ----
    del_path = os.path.join(out, "rubric_deletes.csv")
    with open(del_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "rubric_to_delete", "severity",
                                          "categories", "deletion_reason"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rubric_deletes:
            w.writerow({
                "task_id": r["task_id"],
                "rubric_to_delete": r.get("criterion", ""),
                "severity": r.get("severity", ""),
                "categories": r.get("categories", ""),
                "deletion_reason": r.get("resolution_comment", ""),
            })
    console.log(f"Wrote {del_path}  ({len(rubric_deletes)} rows)")

    # ---- unit-test change manifests: no agent, built from the worklist ----
    test_del_path = os.path.join(out, "test_deletes.csv")
    with open(test_del_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "test_name", "severity",
                                          "categories", "deletion_reason"],
                           extrasaction="ignore")
        w.writeheader()
        for r in test_changes:
            if r["resolution_action"] not in ("delete", "delete_and_create_rubric"):
                continue
            w.writerow({
                "task_id": r["task_id"],
                "test_name": r["item_id"],
                "severity": r.get("severity", ""),
                "categories": r.get("categories", ""),
                "deletion_reason": r.get("resolution_comment", ""),
            })
    n_test_del = sum(1 for r in test_changes if r["resolution_action"] in ("delete", "delete_and_create_rubric"))
    console.log(f"Wrote {test_del_path}  ({n_test_del} rows)")

    test_edit_path = os.path.join(out, "test_edits.csv")
    with open(test_edit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "test_name", "severity",
                                          "categories", "edit_reason"],
                           extrasaction="ignore")
        w.writeheader()
        for r in test_changes:
            if r["resolution_action"] != "edit":
                continue
            w.writerow({
                "task_id": r["task_id"],
                "test_name": r["item_id"],
                "severity": r.get("severity", ""),
                "categories": r.get("categories", ""),
                "edit_reason": r.get("resolution_comment", ""),
            })
    n_test_edit = sum(1 for r in test_changes if r["resolution_action"] == "edit")
    console.log(f"Wrote {test_edit_path}  ({n_test_edit} rows)")

    # ---- build agent jobs ----
    jobs = []  # (kind, key, payload)
    skipped = []
    for r in rubric_edits:
        d = extracted_dir(args.cache_folder, r["task_id"])
        if not d:
            skipped.append(("rubric_edit", r["task_id"], r["item_id"]))
            continue
        jobs.append(("rubric_edit", f"{r['task_id']}:{r['item_id']}", {"d": d, "row": r}))
    for r in rubric_creates:
        d = extracted_dir(args.cache_folder, r["task_id"])
        if not d:
            skipped.append(("rubric_create", r["task_id"], r["item_id"]))
            continue
        jobs.append(("rubric_create", f"{r['task_id']}:{r['item_id']}:create_rubric", {"d": d, "row": r}))
    for tid, groups in tests_by_task.items():
        d = extracted_dir(args.cache_folder, tid)
        if not d or not os.path.exists(os.path.join(d, "tests", "test_outputs.py")):
            skipped.append(("test_file", tid, "no test_outputs.py"))
            continue
        jobs.append(("test_file", f"{tid}:test_file", {"d": d, "task_id": tid, "groups": groups}))

    if args.limit:
        jobs = jobs[:args.limit]
    if skipped:
        console.log(f"[yellow]Skipped {len(skipped)} jobs with no extracted env / test file[/yellow]")

    # ---- run agents in parallel ----
    rubric_edit_rows = []
    rubric_create_rows = []
    total_cost = 0.0

    def do_job(job):
        kind, key, payload = job
        d = payload["d"]
        if kind == "rubric_edit":
            r = payload["row"]
            prompt = rubric_edit_prompt(d, r.get("criterion", ""), r.get("severity", ""),
                                        r.get("categories", ""), r.get("resolution_comment", ""))
            parsed, err, cost = run_agent(prompt, RUBRIC_SCHEMA, d, key,
                                          args.cache_folder, args.disable_cache, args.cache_only)
            return ("rubric_edit", r, parsed, err, cost)
        if kind == "rubric_create":
            r = payload["row"]
            prompt = rubric_create_prompt(d, r.get("item_id", ""), r.get("severity", ""),
                                          r.get("categories", ""), r.get("resolution_comment", ""))
            parsed, err, cost = run_agent(prompt, RUBRIC_SCHEMA, d, key,
                                          args.cache_folder, args.disable_cache, args.cache_only)
            return ("rubric_create", r, parsed, err, cost)
        else:
            groups = payload["groups"]
            prompt = testfile_prompt(d, groups["edit"], groups["delete"])
            parsed, err, cost = run_agent(prompt, TESTFILE_SCHEMA, d, key,
                                          args.cache_folder, args.disable_cache, args.cache_only)
            return ("test_file", payload["task_id"], parsed, err, cost)

    console.rule(f"[bold]Running {len(jobs)} rewrite agents ({args.workers} workers)[/bold]")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(do_job, jobs):
            kind = res[0]
            if kind == "rubric_edit":
                _, r, parsed, err, cost = res
                total_cost += cost or 0
                if parsed:
                    new_rubric = parsed.get("new_rubric", "")
                    # Safety net: a quoted-rubric fix MUST be quote-free. Strip any
                    # straight or smart quotes the agent left behind.
                    if "quoted_overspecific" in (r.get("categories", "") or ""):
                        for qch in ("'", '"', "\u2018", "\u2019", "\u201c", "\u201d"):
                            new_rubric = new_rubric.replace(qch, "")
                        new_rubric = " ".join(new_rubric.split())
                    rubric_edit_rows.append({
                        "task_id": r["task_id"],
                        "old_rubric": r.get("criterion", ""),
                        "new_rubric": new_rubric,
                        "severity": r.get("severity", ""),
                        "categories": r.get("categories", ""),
                        "flagged_issue": r.get("resolution_comment", ""),
                        "edit_justification": parsed.get("notes", ""),
                    })
            elif kind == "rubric_create":
                _, r, parsed, err, cost = res
                total_cost += cost or 0
                if parsed:
                    rubric_create_rows.append({
                        "task_id": r["task_id"],
                        "source_test_name": r.get("item_id", ""),
                        "new_rubric": parsed.get("new_rubric", ""),
                        "severity": r.get("severity", ""),
                        "categories": r.get("categories", ""),
                        "creation_reason": r.get("resolution_comment", ""),
                        "creation_justification": parsed.get("notes", ""),
                    })
            else:
                _, tid, parsed, err, cost = res
                total_cost += cost or 0
                if parsed and parsed.get("new_file_content"):
                    with open(os.path.join(tests_out, f"{tid}.py"), "w", encoding="utf-8") as f:
                        f.write(parsed["new_file_content"])

    # ---- write rubric edits CSV ----
    edits_path = os.path.join(out, "rubric_edits.csv")
    with open(edits_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "old_rubric", "new_rubric",
                                          "severity", "categories", "flagged_issue",
                                          "edit_justification"],
                           extrasaction="ignore")
        w.writeheader()
        for row in rubric_edit_rows:
            w.writerow(row)

    creates_path = os.path.join(out, "rubric_creates.csv")
    with open(creates_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "source_test_name", "new_rubric",
                                          "severity", "categories", "creation_reason",
                                          "creation_justification"],
                           extrasaction="ignore")
        w.writeheader()
        for row in rubric_create_rows:
            w.writerow(row)

    n_test_files = len([n for n in os.listdir(tests_out) if n.endswith(".py")])

    # ---- write README for the audit team ----
    readme_path = os.path.join(out, "README.md")
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    readme = f"""# Verifier-fix output bundle

_Generated {generated} from `{os.path.basename(args.resolution_csv)}`._

This folder contains the concrete fixes derived from the verifier-quality
evaluation. Every flagged rubric / unit test in the resolution worklist was
assigned a **resolution action** (`edit`, `delete`, or `delete_and_create_rubric`); the files below are the
applied results of those actions, ready for review before they go back into the
delivery.

## Counts in this run
| Artifact | Rows / files |
|---|---|
| Rubric edits | {len(rubric_edit_rows)} |
| Rubric deletes | {len(rubric_deletes)} |
| Rubric creates | {len(rubric_create_rows)} |
| Unit-test edits | {n_test_edit} |
| Unit-test deletes | {n_test_del} |
| Rewritten test files | {n_test_files} |

---

## Files

### `rubric_edits.csv`
One row per rubric criterion that was **rewritten**. The new text is produced by
an LLM agent that saw the original criterion, the flagged issue, and the task
environment.

| Column | Meaning |
|---|---|
| `task_id` | Task the rubric belongs to. |
| `old_rubric` | The original (flagged) criterion text. |
| `new_rubric` | The rewritten criterion text to replace it with. |
| `severity` | `major` or `minor` — how serious the original issue was. |
| `categories` | Quality-issue category tags for the original problem. |
| `flagged_issue` | Why the verifier flagged the original criterion (the evidence/comment). |
| `edit_justification` | The rewrite agent's note explaining what it changed and why. |

### `rubric_deletes.csv`
One row per rubric criterion that should be **removed** entirely (no replacement).

| Column | Meaning |
|---|---|
| `task_id` | Task the rubric belongs to. |
| `rubric_to_delete` | Exact criterion text to delete. |
| `severity` | `major` or `minor`. |
| `categories` | Quality-issue category tags. |
| `deletion_reason` | Why this criterion should be deleted. |

### `rubric_creates.csv`
One row per unit test that should be **deleted and replaced with a new rubric**.

| Column | Meaning |
|---|---|
| `task_id` | Task the new rubric belongs to. |
| `source_test_name` | Unit test that was removed from `unit_tests/<task_id>.py`. |
| `new_rubric` | New rubric criterion to add. |
| `severity` | `major` or `minor`. |
| `categories` | Quality-issue category tags. |
| `creation_reason` | Why pytest was the wrong verifier form. |
| `creation_justification` | The rewrite agent's note explaining the new rubric. |

### `test_edits.csv`
Manifest of every unit test that was **edited**. The actual rewritten code lives
in `unit_tests/<task_id>.py` (a whole file is rewritten per task); this CSV is
the human-readable index of *which* tests changed and *why*.

| Column | Meaning |
|---|---|
| `task_id` | Task the test belongs to. |
| `test_name` | Name of the test function that was edited. |
| `severity` | `major` or `minor`. |
| `categories` | Quality-issue category tags. |
| `edit_reason` | Why the test was flagged for editing. |

### `test_deletes.csv`
Manifest of every unit test that was **deleted**. The deletion is already applied
in the rewritten `unit_tests/<task_id>.py` file; this CSV records what was removed
and why, for audit.

| Column | Meaning |
|---|---|
| `task_id` | Task the test belongs to. |
| `test_name` | Name of the test function that was removed. |
| `severity` | `major` or `minor`. |
| `categories` | Quality-issue category tags. |
| `deletion_reason` | Why this test was deleted. |

### `unit_tests/<task_id>.py`
The full rewritten unit-test file for a task, with **all** edits and deletions for
that task already applied. There is one file per task that had any test change.
Tests that were not flagged are preserved unchanged. Use `test_edits.csv` and
`test_deletes.csv` to see exactly which functions were touched in each file.

---

## How to use this for the fix-up
1. Apply `rubric_edits.csv`: replace each `old_rubric` with its `new_rubric` on the matching `task_id`.
2. Apply `rubric_deletes.csv`: remove each `rubric_to_delete` from its `task_id`.
3. Apply `rubric_creates.csv`: add each `new_rubric` to its `task_id`.
4. Replace each task's unit-test file with the corresponding `unit_tests/<task_id>.py`.
5. The two `test_*.csv` manifests are for review/audit only — no action needed beyond step 4.

> Note: severity/categories come straight from the verifier-quality evaluation.
> `major` issues are correctness problems; `minor` issues are clarity / style.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    console.rule("[bold]Done[/bold]")
    console.log(f"rubric_edits.csv   → {edits_path}  ({len(rubric_edit_rows)} rows)")
    console.log(f"rubric_deletes.csv → {del_path}  ({len(rubric_deletes)} rows)")
    console.log(f"rubric_creates.csv → {creates_path}  ({len(rubric_create_rows)} rows)")
    console.log(f"test_edits.csv     → {test_edit_path}  ({n_test_edit} rows)")
    console.log(f"test_deletes.csv   → {test_del_path}  ({n_test_del} rows)")
    console.log(f"unit_tests/*.py    → {tests_out}  ({n_test_files} files)")
    console.log(f"README.md          → {readme_path}")
    console.log(f"approx agent cost: ${total_cost:,.2f}")


if __name__ == "__main__":
    main()
