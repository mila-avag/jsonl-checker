#!/usr/bin/env python3
"""Tara Eval — contributor-facing web UI (self-contained, runs locally).

A contributor pastes their prompt, picks a universe, optionally uploads their
pytest ``test_outputs.py`` and/or pastes rubrics, then gets back live verifier
feedback:

  * per-rubric audit  (verdict / quality issues / resolution / reasoning)
  * per-test  audit   (verdict / quality issues / resolution / reasoning)
  * coverage / redundancy / single-GTFA checks
  * quoted-rubric + intersection checks (when rubrics are supplied)

It drives the real evaluation engine in ``engine/run_threshold_eval.py`` by
importing it directly and calling the same functions the batch pipeline uses.
The UI uses the bundled engine's prompt builders, then calls LiteLLM instead of
shelling out to the ``claude`` CLI.

Run:
    pip install -r requirements.txt
    uvicorn app:app --port 8000
then open http://localhost:8000
"""
import os
import re
import sys
import json
import time
import uuid
import glob
import shutil
import zipfile
import hashlib
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from litellm import completion
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------
# Wire up the bundled eval engine (everything lives inside this folder)
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "engine")
UNIVERSE_DIR = os.environ.get("TARA_UNIVERSE_DIR", os.path.join(HERE, "universes"))
CACHE = os.environ.get("TARA_UI_CACHE", os.path.join(HERE, ".cache"))
WORKERS = int(os.environ.get("TARA_UI_WORKERS", "6"))
LITELLM_MODEL = os.environ.get("TARA_LITELLM_MODEL", "claude-opus-4-6")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://litellm.ml.scaleinternal.com/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY") or os.environ.get("API_KEY")

sys.path.insert(0, EVAL_DIR)
import run_threshold_eval as rte  # noqa: E402

os.makedirs(CACHE, exist_ok=True)

# Task-level checks. Quoted-rubric + intersection only make sense once the
# contributor supplies rubrics, so they're added conditionally.
CHECKS_NO_RUBRIC = ["coverage_check", "redundancy_check", "single_gtfa_check"]
CHECKS_WITH_RUBRIC = ["coverage_check", "intersection_check", "redundancy_check",
                      "single_gtfa_check", "quoted_rubric_check"]

TEST_DEF_RE = re.compile(r"def\s+(test_[A-Za-z0-9_]+)\s*\(")


# --------------------------------------------------------------------------
# LiteLLM runner
# --------------------------------------------------------------------------
MAX_CONTEXT_CHARS = int(os.environ.get("TARA_LITELLM_CONTEXT_CHARS", "90000"))
MAX_FILE_CHARS = int(os.environ.get("TARA_LITELLM_FILE_CHARS", "8000"))


def _try_parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _safe_read_text(path: str, limit: int = MAX_FILE_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            data = data[:limit] + "\n...[truncated]..."
        return data
    except Exception as e:
        return f"[could not read {path}: {e}]"


def build_environment_context(extract_dir: str) -> str:
    """Compact file context for LiteLLM calls.

    Claude Code had Read/Glob/Grep/Bash tools. LiteLLM does not have those local
    tools, so we inline the high-signal files contributors need for this UI:
    tests, prompt files, metadata, and OpenClaw service data JSON.
    """
    chunks: list[str] = []
    include_names = {"instruction.md", "task.toml", "test_outputs.py", "data.json"}
    include_dirs = (os.sep + "metadata" + os.sep, os.sep + "tests" + os.sep)
    for root, dirs, files in os.walk(extract_dir):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, extract_dir)
            lower = name.lower()
            normalized = os.sep + rel
            should_include = (
                name in include_names
                or normalized.startswith(include_dirs)
                or lower.endswith(".json") and ("/services/" in normalized or "\\services\\" in normalized)
            )
            if not should_include or lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic")):
                continue
            chunks.append(f"\n--- FILE: {rel} ---\n{_safe_read_text(path)}")
            if sum(len(c) for c in chunks) >= MAX_CONTEXT_CHARS:
                chunks.append("\n...[environment context truncated]...")
                return "\n".join(chunks)
    return "\n".join(chunks) if chunks else "(no readable environment files found)"


def _schema_for_single_eval() -> dict:
    return json.loads(rte.SCHEMA_JSON)


def _lite_completion(prompt: str, schema: dict, extract_dir: str) -> tuple[dict | None, float, str | None]:
    if not LITELLM_API_KEY:
        return None, 0.0, "LITELLM_API_KEY is not set"

    context = build_environment_context(extract_dir)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous verifier auditing OpenClaw task artifacts. "
                "Use the provided environment context instead of tool calls. "
                "Return only a JSON object that matches the provided schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "==================== LOCAL ENVIRONMENT CONTEXT ====================\n"
                f"{context}\n"
                "==================== REQUIRED JSON SCHEMA ====================\n"
                f"{json.dumps(schema, separators=(',', ':'))}\n"
                "Return only valid JSON. Do not include markdown fences."
            ),
        },
    ]
    try:
        resp = completion(
            model=LITELLM_MODEL,
            messages=messages,
            api_base=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            temperature=0,
        )
    except Exception as e:
        return None, 0.0, f"{type(e).__name__}: {e}"

    content = resp.choices[0].message.content if resp.choices else ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    parsed = _try_parse_json(content)
    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        cost = float(usage.get("cost", 0.0) or 0.0)
    else:
        cost = float(getattr(usage, "cost", 0.0) or 0.0) if usage else 0.0
    return parsed, cost, None if parsed else "no JSON"


def _cache_key(prompt: str, schema: dict) -> str:
    return hashlib.sha256(
        (LITELLM_MODEL + "\n" + prompt + "\n" + json.dumps(schema, sort_keys=True)).encode("utf-8")
    ).hexdigest()


def run_single_eval_litellm(item: dict, prompt: str, extract_dir: str,
                            cache_folder: str, disable_cache: bool = False) -> dict:
    schema = _schema_for_single_eval()
    db_path = os.path.join(cache_folder, "threshold_eval_cache.db")
    rte._ensure_cache_schema(db_path, cache_folder)
    prompt_hash = _cache_key(prompt, schema)
    if not disable_cache:
        cached = rte.cache_get(db_path, prompt_hash)
        if cached and not cached.get("error"):
            return cached

    start = time.time()
    parsed, cost, error = _lite_completion(prompt, schema, extract_dir)
    duration = time.time() - start
    result = rte._build_result(item, parsed=parsed, error=error, duration=duration, cost_usd=cost, attempts=1)
    if not error:
        rte.cache_put(db_path, prompt_hash, result)
    return result


def run_task_check_litellm(item: dict, extract_dir: str, cache_folder: str,
                           disable_cache: bool = False) -> dict:
    build_prompt, schema_json, _label = rte._TASK_CHECK_CONF[item["item_type"]]
    schema = json.loads(schema_json)
    prompt = build_prompt(item, extract_dir)
    db_path = os.path.join(cache_folder, "threshold_eval_cache.db")
    rte._ensure_cache_schema(db_path, cache_folder)
    prompt_hash = _cache_key(prompt, schema)
    if not disable_cache:
        cached = rte.cache_get(db_path, prompt_hash)
        if cached and not cached.get("error"):
            return cached

    start = time.time()
    parsed, cost, error = _lite_completion(prompt, schema, extract_dir)
    duration = time.time() - start
    result = rte._build_task_check_result(item, parsed=parsed, error=error, duration=duration, cost_usd=cost, attempts=1)
    if not error:
        rte.cache_put(db_path, prompt_hash, result)
    return result


def parse_rubrics(text: str) -> dict:
    """Accept either a JSON rubrics dict or one-criterion-per-line plain text.

    Plain-text lines may be prefixed with markers like 'R1:', '1.', '- '.
    """
    text = (text or "").strip()
    if not text:
        return {}
    if text.lstrip().startswith("{"):
        raw = json.loads(text)
        out = {}
        for k, v in raw.items():
            if isinstance(v, str):
                v = {"criterion": v}
            v.setdefault("criterion", "")
            v.setdefault("is_positive", True)
            v.setdefault("importance", "important")
            v.setdefault("score", 1)
            out[str(k)] = v
        return out
    out, i = {}, 1
    for line in text.splitlines():
        line = re.sub(r"^\s*(R?\d+\s*[:.\)]\s*|[-*\u2022]\s*)", "", line).strip()
        if not line:
            continue
        out[f"R{i}"] = {"criterion": line, "is_positive": True,
                        "importance": "important", "score": 1}
        i += 1
    return out


app = FastAPI(title="Tara Eval")

# --------------------------------------------------------------------------
# In-memory job store
# --------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set(job_id: str, **kw):
    with _jobs_lock:
        _jobs[job_id].update(kw)


def _get(job_id: str) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


# --------------------------------------------------------------------------
# Universe discovery
# --------------------------------------------------------------------------
def list_universes() -> list[dict]:
    """Find openclaw-<person>-universe-<hash>.zip files; de-dupe ' (1)' copies."""
    out: dict[str, dict] = {}
    pat = re.compile(r"openclaw-(.+?)-universe-([a-z0-9]+)", re.I)
    for path in sorted(glob.glob(os.path.join(UNIVERSE_DIR, "openclaw-*-universe-*.zip"))):
        fname = os.path.basename(path)
        stem = fname[:-4]
        if "(" in stem:  # skip duplicate downloads like "... (1).zip"
            continue
        m = pat.search(fname)
        if not m:
            continue
        person, h = m.group(1), m.group(2)
        label = person.replace("_", " ").title() + f"  \u00b7  {h}"
        out[stem] = {"value": fname, "label": label, "person": person}
    return sorted(out.values(), key=lambda x: x["label"].lower())


# --------------------------------------------------------------------------
# Environment resolution
# --------------------------------------------------------------------------
def _extract_universe_flat(zip_path: str, extract_dir: str):
    """Extract a universe zip, stripping the single top-level folder so that
    ``extract_dir/services`` exists (this is what the eval's `is_universe`
    detection looks for)."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist()
                 if not n.startswith("__MACOSX/") and not os.path.basename(n).startswith(".")]
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        strip = len(tops) == 1  # single wrapper dir -> strip it
        for n in names:
            if n.endswith("/"):
                continue
            rel = n.split("/", 1)[1] if (strip and "/" in n) else n
            if not rel:
                continue
            dest = os.path.join(extract_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(n) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def resolve_env(job_id, task_id, universe_zip, env_url, prompt) -> str:
    extract_dir = os.path.join(CACHE, "tasks", task_id, "extracted")
    if env_url:
        ed = rte.download_and_extract(task_id, env_url, "", CACHE)
        if not ed:
            raise RuntimeError("Could not download / extract the env zip from that URL.")
        # ensure the contributor's prompt is what the auditor sees
        try:
            with open(os.path.join(ed, "instruction.md"), "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            pass
        return ed
    # local universe zip
    zip_path = os.path.join(UNIVERSE_DIR, universe_zip)
    if not os.path.exists(zip_path):
        raise RuntimeError(f"Universe zip not found: {universe_zip}")
    _extract_universe_flat(zip_path, extract_dir)
    return extract_dir


def place_test_file(extract_dir: str, code: str) -> list[str]:
    tests_dir = os.path.join(extract_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    with open(os.path.join(tests_dir, "test_outputs.py"), "w", encoding="utf-8") as f:
        f.write(code)
    return sorted(set(TEST_DEF_RE.findall(code)))


# --------------------------------------------------------------------------
# Job runner
# --------------------------------------------------------------------------
def run_job(job_id, prompt, universe_zip, env_url, test_code, rubrics):
    try:
        task_id = "ui_" + uuid.uuid4().hex[:16]
        _set(job_id, status="resolving", message="Resolving universe / environment\u2026", task_id=task_id)
        extract_dir = resolve_env(job_id, task_id, universe_zip, env_url, prompt)

        test_names = place_test_file(extract_dir, test_code)
        if not test_names and not rubrics:
            raise RuntimeError("No `def test_*` functions found and no rubrics provided.")

        task = {
            "task_id": task_id,
            "prompt": prompt,
            "rubrics": rubrics,
            "unit_tests": [{"test_name": t, "weight": 1} for t in test_names],
        }

        whitelist = list(rubrics.keys()) + test_names
        items = rte.build_eval_items([task], threshold=1.0, rubric_fraction=1.0,
                                     seed=0, item_whitelist={task_id: whitelist})

        checks = CHECKS_WITH_RUBRIC if rubrics else CHECKS_NO_RUBRIC
        n_rub = sum(1 for it in items if it["item_type"] == "rubric")
        n_test = sum(1 for it in items if it["item_type"] == "unit_test")
        total = len(items) + len(checks)
        _set(job_id, status="running", total=total, completed=0,
             message=f"Auditing {n_rub} rubrics + {n_test} tests + {len(checks)} task-level checks\u2026",
             rubrics=[], tests=[], checks=[])

        def do_item(item):
            p = rte.build_verifier_prompt(item, extract_dir)
            res = run_single_eval_litellm(item, p, extract_dir, CACHE)
            return (item["item_type"], res)

        def do_check(check_type):
            citems = rte.build_task_check_items([task], check_type)
            if not citems:
                return ("check", {"item_id": check_type, "error": "no check item"})
            return ("check", run_task_check_litellm(citems[0], extract_dir, CACHE))

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(do_item, it) for it in items]
            futs += [ex.submit(do_check, c) for c in checks]
            done = 0
            for fut in as_completed(futs):
                kind, res = fut.result()
                done += 1
                bucket = {"rubric": "rubrics", "unit_test": "tests"}.get(kind, "checks")
                with _jobs_lock:
                    j = _jobs[job_id]
                    j[bucket].append(res)
                    j["completed"] = done
                    j["message"] = f"Completed {done}/{total}\u2026"

        _set(job_id, status="done", message="Evaluation complete.")
    except Exception as e:
        _set(job_id, status="error",
             message=str(e), traceback=traceback.format_exc())


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/universes")
def api_universes():
    return {"universes": list_universes(), "universe_dir": UNIVERSE_DIR}


@app.post("/api/eval")
async def api_eval(prompt: str = Form(...),
                   universe: str = Form(""),
                   env_url: str = Form(""),
                   rubrics: str = Form(""),
                   test_file: UploadFile | None = File(None)):
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required.")
    if not universe and not env_url:
        raise HTTPException(400, "Select a universe or provide an env zip URL.")

    code = ""
    if test_file is not None:
        code = (await test_file.read()).decode("utf-8", errors="replace")

    try:
        rubric_dict = parse_rubrics(rubrics)
    except Exception as e:
        raise HTTPException(400, f"Could not parse rubrics: {e}")

    if not code.strip() and not rubric_dict:
        raise HTTPException(400, "Provide a test file, rubrics, or both.")

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "message": "Queued\u2026",
                         "rubrics": [], "tests": [], "checks": [],
                         "completed": 0, "total": 0,
                         "filename": test_file.filename if test_file else None}
    threading.Thread(target=run_job,
                     args=(job_id, prompt, universe, env_url.strip(), code, rubric_dict),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    j = _get(job_id)
    if not j:
        raise HTTPException(404, "No such job.")
    return JSONResponse(j)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
