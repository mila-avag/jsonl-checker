# Honeybee QC — from an empty Mac

This package audits a contributor's side-by-side work product (rubric scores,
dimension ratings, cited turns, SxS ranking) and then rewrites the fields a
fail actually named. It is the same laptop eval + fixer Mila's team runs.
You do not need the git repo. You do need Python, Chrome, your own Anthropic
auth, and a Snowflake `taskattempts` CSV (or Redash to pull one).

Deeper design notes live in [`README.md`](README.md) if you already care.
Do not start there.

---

## 1. What this is

`honeybee_qc` is a local Python package. One wrapper,
[`team_run.sh`](team_run.sh), runs four steps on your laptop: **eval**
(`report.json`), **fixer** (field rewrites), **invariants gate** (paste
safety), and a **paste sheet** (`backfill_steps.csv` / `backfill_dims.csv`).
Default checks: Valid Links (90), Unique Links (91), Output Artifacts (95),
Key Turn Identification (100), Key Turn Justification (110), Rubric
Evaluation (270), Correctness/Accuracy (300), Relevant Turns (310), Ranking
Disagreement (400), Justifications (450), Verdict (470). Always write full
24-hex task IDs.

---

## 2. What you must have before anything runs

An empty Mac is fine. Install these yourself. None of them are in the
archive (and they cannot be: we cannot ship API keys or your Chrome
session).

| Need | Why |
|---|---|
| **Homebrew Python 3.13** (`/opt/homebrew/bin/python3`) | The package uses 3.13 syntax. System `/usr/bin/python3` is often 3.9 and will fail. `brew install python@3.13` |
| **Google Chrome** | Hydration launches headless Chrome to render Gemini / Claude / GPT share pages. Log into those sites in your normal Chrome so *you* can open links; the fetcher itself uses a throwaway profile (public `/share/` URLs work, private `/app/` `/c/` `/chat/` URLs fail Valid Links). |
| **`claude` CLI** (Claude Code) on PATH | Model calls are `claude -p …`, not the Anthropic HTTP SDK. |
| **Anthropic auth** | `export ANTHROPIC_API_KEY=…` **or** `claude login`. Confirm `claude` already works in a terminal. |
| **~10 GB free disk** | Snapshots + `cache.db` grow fast. Temp disk also needs room for Chrome profiles. |
| A **taskattempts CSV** or Redash | Columns below. Not shipped. |

Optional: `brew install tesseract` for OCR on scanned PDF chat exports.

Confirm:

```bash
/opt/homebrew/bin/python3 --version    # 3.13.x
which claude && claude --version
test -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" && echo chrome_ok
```

Do not put keys, `.env`, or `~/.cursor/redash.env` inside this folder.

---

## 3. Unpack

You were sent `honeybee_qc_team.zip` and/or `honeybee_qc_team.tar.gz`. They
are the same tree. On a Mac, **double-click the zip**.

```bash
# or, if you only have the tar:
tar xzf honeybee_qc_team.tar.gz
```

`cd` to the folder that **contains** `honeybee_qc/` (usually
`honeybee_qc_team/` after extract). That directory is the working directory
for every command below. `python3 -m honeybee_qc.cli` needs it on
`sys.path`; `team_run.sh` sets `PYTHONPATH` for you.

A Finder copy of the `honeybee_qc/` folder is equivalent. You do not
need git.

---

## 4. Install Python deps

```bash
/opt/homebrew/bin/python3 -m pip install -r honeybee_qc/requirements.txt
```

That is pytest plus optional PDF/OCR backends (`pymupdf`, `pypdf`,
`pytesseract`). Missing PDF extras degrade an attachment to "no text"
instead of crashing. Runtime model calls go through `claude` and Chrome,
not pip.

---

## 5. Get data

`--from-snowflake` means a Redash / Snowflake export of
`public.taskattempts`, **not** a list of IDs. Required columns:

`task`, `project`, `attempted_by`, `review_status`, `response`

`response` is a large JSON blob per row. A file of 24-hex IDs alone will
not ingest.

### Paste a CSV you already have

Point `--csv` at it. Skip the pull.

### Pull from Redash

1. Put full 24-hex IDs, one per line, in `ids.txt`.
2. Print the SQL template (same query ingest uses, plus an ID filter):

```bash
./honeybee_qc/team_run.sh pull-sql --project l1_sxs --ids-file ids.txt > pull.sql
```

3. Run `pull.sql` in Redash (or the Redash CLI if you already have a key).
   Save the result as CSV.

`--project` picks the Snowflake `project` UUID and the step map:

| `--project` | Snowflake `project` | Pair |
|---|---|---|
| `l1_sxs` | `6a567b80aaa4d140131eae6c` | L1 SxS queue, A vs B |
| `l1_sxs_c` | `6a567b80aaa4d140131eae6c` | same project id, A vs C only |
| `aspirational` | `6a84adb9124120a0ad6fa217` | A / B / C + simplified |

`l1_sxs` and `l1_sxs_c` share one project id and **disagree** about step
IDs. Always pass `--project`. Typical flags:

- `--project l1_sxs --slots A,B`
- `--project l1_sxs_c --slots A,C`
- `--project aspirational --slots A,B` or `A,C`

Redash keys stay in **your** environment (`REDASH_API_KEY`, `REDASH_URL`,
prefer data source **30**). Do not copy someone else's env file here.

---

## 6. One command

From the folder that contains `honeybee_qc/`:

```bash
./honeybee_qc/team_run.sh all \
  --csv /path/to/pulled.csv \
  --project l1_sxs \
  --slots A,B \
  --out honeybee_qc/runs/my_batch
```

That is eval → fixer → invariants gate → paste sheet. The wrapper pins
`--hydrate-workers 8`, `--workers 60`, `--task-workers 20`,
`--model claude-haiku-4-5`, `--fetch-conversations`, and the default check
set. You can run `eval`, `fixer`, `gate`, or `sheet` alone with the same
flags (`gate` / `sheet` / `fixer` do not need `--slots`).

`--dry-run` is on the raw CLI only (`python3 -m honeybee_qc.cli … --dry-run`)
if you want a model-call count first.

---

## 7. What comes out

```
<out>/
  report.json                 eval verdicts (every check, every task)
  cache.db                    SQLite model-response cache — reuse this path
  snapshots/                  rendered DOMs / HTML exports (the fetch cache)
  fixer/
    fixer_results.json        per-task status + patches
    fixer_invariants.json     gate report (required; missing = incomplete)
    all_patches.csv           every field rewrite
    unfixable.csv             tasks the fixer will not edit
    backfill_steps.csv        paste-ready form steps
    backfill_dims.csv         paste-ready dimension annotations
    tasks.csv                 one row per task (status, fail names)
```

The gate writes `fixer_invariants.json`. Exit 1 / any `hold_tasks` means
those IDs are not paste-ready. Print the human checklist with
`python3 -m honeybee_qc.fixer_invariants --print-subagent-prompt`.

`cache.db` and `snapshots/` are safe to reuse on a rerun. Keep them on
the laptop that ran the job.

---

## 8. Gotchas

- **`--hydrate-workers 8`.** Never 60. Never "match workers". Each hydrate
  worker is a Chrome. If you omit `--hydrate-workers` on the raw eval CLI,
  it defaults to `--workers` and you get empty DOMs.
- **Do not kill Chrome.** Hydration starts its own headless processes.
  Killing the browser mid-run orphans `hb_profile_*` dirs in `$TMPDIR` and
  the next batch looks like a network failure. Do not close the lid without
  `caffeinate` (`team_run.sh` uses `caffeinate -ims` when it exists).
- **Unique Links (91) on HTML-B is `not_evaluated`, not unfixable.** Spark
  HTML/MHTML chat exports have no per-turn Gemini share URLs. Do not park
  the task or skip Relevant Turns / Ranking Disagreement / Justifications
  because of it. A **share-link** B that reuses URLs is still a real Unique
  Links fail.
- **SxS Likert is 1–7. Only 4 is a tie.** 3 and 5 are slight wins. Likert,
  writeup, and `responseIdx` (0 left, 1 right, −1 tie) must agree.
- **Projects:** `l1_sxs` (A,B), `l1_sxs_c` (A,C), `aspirational` (pick the
  pair). Without `--project`, L1-family rows ingest as old `l1` (Unique
  Links off, wrong ranking step).
- Headless Chrome is a **throwaway profile**, not your logged-in window.
  Login-wall / challenge pages hydrate empty and downstream ratings go
  `not_evaluated`. `not_evaluated` is not a pass.
- Reuse `--cache-db` and `--snapshot-dir` or you pay for every call and
  every share-page render twice.
- Prompt / trajectory fails (dead share link, Unique Links on a real
  share-link Gemini, missing artifacts) land on `unfixable.csv`. Other
  fields on that task can still be fixed.
- Speak in check **names**. Gate exit 1 is a hold, not a suggestion.
