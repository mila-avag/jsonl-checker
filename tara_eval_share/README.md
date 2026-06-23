# Tara Eval — self-serve verifier quality check

A small local web app for contributors to **sanity-check their own tasks before
delivery**. Paste your prompt, pick a universe, optionally add your rubrics
and/or your `test_outputs.py`, hit **Run**, and get back the same verifier
feedback the batch audit pipeline produces:

- **Per-rubric audit** — verdict, quality issues, suggested resolution, reasoning
- **Per-test audit** — verdict, quality issues, suggested resolution, reasoning
- **Coverage check** — are the prompt's requirements actually covered by the tests?
- **Redundancy check** — are any tests duplicative?
- **Single-GTFA check** — do the tests target one ground-truth final answer?
- **Quoted-rubric** & **intersection** checks — run automatically when rubrics are provided

Everything needed is bundled in this folder: the evaluation engine, the
universe zips, and the web UI. You only need Python and a LiteLLM API key.

---

## Prerequisites

1. **Python 3.10+**
   ```bash
   python3 --version
   ```

2. **A LiteLLM API key.** The UI calls the Scale LiteLLM endpoint directly and
   keeps using the Anthropic model configured by `TARA_LITELLM_MODEL`.
   ```bash
   export LITELLM_API_KEY="..."
   export LITELLM_BASE_URL="https://litellm.ml.scaleinternal.com/"
   export TARA_LITELLM_MODEL="claude-opus-4-6"
   ```
   Do not commit or paste real keys into files.

---

## Quick start

```bash
cd tara_eval_share
./run.sh
```

`run.sh` creates a virtual environment, installs the dependencies, and starts
the server. When it's up, open:

> **http://localhost:8000**

To stop it, press `Ctrl+C`.

### Manual start (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

---

## How to use it

1. **Paste your full prompt** — the exact task instruction the model receives.
2. **(Optional) Paste your rubrics** — one criterion per line, or a JSON rubric
   dict. Adding rubrics turns on the per-rubric audits plus the quoted-rubric
   and intersection checks.
3. **Pick a universe** from the dropdown (these are the bundled
   `openclaw-*-universe-*.zip` files). Or click *"use an env-zip URL instead"*
   to point at a `scale-cds://` / `https://` environment zip.
4. **(Optional) Upload your `test_outputs.py`** — drag-and-drop or click to browse.
   You need *a test file, rubrics, or both.*
5. Click **Run evaluation**. Results stream in as each item finishes; flagged
   items are sorted to the top and expanded automatically.

---

## What's in here

```
tara_eval_share/
├── app.py                 # FastAPI web server (the UI backend)
├── run.sh                 # one-shot launcher (venv + deps + start)
├── requirements.txt
├── static/
│   └── index.html         # the single-page UI
├── engine/                # the real evaluation engine (vendored)
│   ├── run_threshold_eval.py
│   └── actions/agent_pipeline/*.json   # the audit JSON schemas
└── universes/             # bundled openclaw-*-universe-*.zip files
```

The app imports `engine/run_threshold_eval.py` directly and calls the same
functions the nightly batch pipeline uses, so the verdicts match.

---

## Configuration (optional env vars)

| Variable             | Default            | Purpose                                              |
|----------------------|--------------------|------------------------------------------------------|
| `PORT`               | `8000`             | Port for the web server (`PORT=9000 ./run.sh`)       |
| `TARA_UI_WORKERS`    | `6`                | Parallel audits. Raise it if your LiteLLM rate limits allow.     |
| `TARA_UNIVERSE_DIR`  | `./universes`      | Where to look for universe zips. Point at `~/Downloads` to use your own. |
| `TARA_UI_CACHE`      | `./.cache`         | Extracted envs + the SQLite result cache.            |
| `LITELLM_API_KEY`    | required           | LiteLLM API key.                                    |
| `LITELLM_BASE_URL`   | Scale LiteLLM URL  | LiteLLM endpoint.                                   |
| `TARA_LITELLM_MODEL` | `claude-opus-4-6`  | Model sent to LiteLLM.                              |

Identical runs are cached, so re-running the same prompt/test is instant.

---

## Adding your own universes

Drop any `openclaw-<person>-universe-<hash>.zip` into `universes/` (or set
`TARA_UNIVERSE_DIR` to a folder that has them, e.g. your `~/Downloads`) and hit
the **↻ refresh** link next to the dropdown.

---

## Troubleshooting

- **Dropdown says "No universes found"** — the `universes/` folder is empty or
  `TARA_UNIVERSE_DIR` points somewhere without `openclaw-*-universe-*.zip` files.
- **Every audit errors out / "LITELLM_API_KEY is not set"** — export
  `LITELLM_API_KEY` before running `./run.sh`.
- **Every audit errors out with API/auth failures** — confirm the LiteLLM key,
  base URL, and model env vars are correct.
- **Port already in use** — start on another port: `PORT=8800 ./run.sh`.
- **Slow** — each audit is a real LiteLLM call. Raise `TARA_UI_WORKERS` if your
  rate limits allow, or evaluate fewer items at a time.
