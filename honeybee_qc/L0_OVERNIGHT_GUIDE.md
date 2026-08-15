# L0 overnight restricted-scope QC — coworker guide

## Quick start — just talk to the Cursor agent

You don't have to type any of the commands below yourself. Open this
project in Cursor and tell the agent one of these, in plain English:

- **"Give me a status update on the L0 overnight QC run."** → it'll run
  §6's health check and summarize what's processed, what's pending, and
  anything in the error log.
- **"Anything look wrong with the L0 overnight run? Check for me."** → it'll
  read `results.csv`/`tick_errors.log` and the Sheet, and flag anything
  worth a human look.
- **"Stop the L0 overnight loop."** → runs the kill switch (§5) for you.
- **"Restart the L0 overnight loop."** → runs `l0_loop.sh` again (§5).

Everything past this point is the reference material the agent (or you, by
hand) actually follows — read on if you want the details or need to run a
command yourself.

**What this is:** an unattended pipeline that watches the L0 project
(`6a7a2d34ac683589810bfa03`) for tasks **attempted from yesterday onward**
that just reached the rating/likert stage, marks them "WIP" in a Google
Sheet, audits each one on a **restricted set of checks only**, then fills in
the real result — logging everything to a CSV (the durable record) as well.

Run folder for tonight: `honeybee_qc/audit_runs/l0_overnight_20260813/`

Google Sheet: https://docs.google.com/spreadsheets/d/1HrzymwVhHnfalZnINFdRYJtCmYjkF8mtHiN80fqv30w (tab: `Tasks`)

---

## 1. Where to look first (no code required)

- **Google Sheet, `Tasks` tab** — the running tracker. Each row: Task ID,
  Status, task verdict, reasoning, issues. A task gets a row the moment it's
  picked up (`Status = WIP`, other columns blank) — **before** the several
  minutes of auditing happen, not after — so you can see what's currently
  running vs. what's actually done.
- **`honeybee_qc/audit_runs/l0_overnight_20260813/results.csv`** — same
  final data, plus the raw check IDs. Columns: `tick_at`, `task_id`,
  `task_verdict`, `fail_checks`, `non_fail_checks`, `not_evaluated_checks`,
  `check_270_a`, `check_270_b` (per-model rubric-rating-correctness bands),
  `reasoning`, `issues`, `status`. This is the source of truth if the Sheet
  and the CSV ever disagree — the Sheet is a transcription of this file, not
  the other way around. There is no "WIP" row in this CSV — WIP only exists
  in the Sheet, as a live status; the CSV only ever gets a row once a task
  is actually done.
- **`honeybee_qc/audit_runs/l0_overnight_20260813/processed_ids.txt`** —
  which task IDs have already been audited (so re-running doesn't
  double-charge for the same task).
- **`honeybee_qc/audit_runs/l0_overnight_20260813/tick_errors.log`** — every
  failure any tick has hit (Snowflake hiccup, CLI crash, timeout), with a
  timestamp. Empty/missing file = no errors yet. A tick failing here does
  **not** stop the loop — it just tries again next tick.
- **`honeybee_qc/audit_runs/l0_overnight_20260813/tick_report.json`** — the
  full detail (every check, every finding, confidences, cost) for the *most
  recent* tick only. Older ticks' detail isn't kept, only their CSV row.

## 2. How tasks are picked

A task is picked up by a tick only if **both**:

1. It has an attempt on project `6a7a2d34ac683589810bfa03` with
   `attempted_at` from **yesterday onward** (a sliding window recomputed
   fresh every tick — not a fixed date), and
2. It has actually reached the rating/likert stage — a real, non-zero
   `criterion_ratings` score or a recorded `sxs.likert`. Not just "the step
   exists," which is true from the moment the rubric is authored.

Anything older than yesterday, or still stuck in rubric-authoring, is
ignored entirely — this loop tracks fresh activity, not a backlog.

## 3. What "restricted scope" means

We are **only** checking:

| Check | What it means |
|---|---|
| 270 | Does the auditor agree with the contributor's pass/fail rubric scoring? (reported per model, A and B separately) |
| 280 / 310 | Is the cited conversation turn for a rubric/dimension rating actually correct? |
| 400 / 460 | Does the auditor's own head-to-head Likert disagree with the contributor's, or contradict their own per-dimension leanings? |
| 450 / 470 | Is the model-comparison justification well-supported, and was a verdict actually stated? |

**Explicitly NOT checked:** rubric-authoring quality, and check 300 (the
8-dimension quality ratings, e.g. Outcome quality) — that was a separate,
already-completed 42-task run earlier and is out of scope here.

A check can legitimately come back `not_evaluated` — that means the auditor
didn't have enough evidence to judge it (e.g. the model's share link is
dead), **not** that it passed. Don't read `not_evaluated` as "clean."

`task_verdict` in `results.csv` is one of:

| Verdict | Meaning |
|---|---|
| `clean` | Every restricted check ran and none disagreed with the contributor. |
| `pass_with_issues` | At least one check flagged a minor disagreement (`non_fail`), or too much went `not_evaluated` to call it clean outright. Worth a skim, not necessarily a real problem. |
| `fail` | At least one check flagged a real disagreement — look at `fail_checks` and `reasoning`. |
| `unauditable` | The pipeline itself couldn't run the checks it needed to (missing/corrupt data), separate from a quality judgment. |

## 4. How the overnight loop works

Two independent pieces, both already running as of this writing:

1. **`honeybee_qc/l0_loop.sh <run-dir> <interval-seconds>`** — a plain shell
   heartbeat. Every `interval` seconds (default 7200 = 2h) it prints a
   sentinel line; it does **no pulling, auditing, or Sheet-writing itself**.
   It checks for a stop file every ~60s (or every `interval` seconds if
   `interval` is under 60s) and is the **only** thing that can end the loop —
   it never exits on its own, never after N iterations, never because of a
   transient error, because there's nothing in it that can throw one.
2. **The Cursor agent turn that reacts to the sentinel** does the real work,
   in this order, every time a sentinel fires:
   1. `python3 -m honeybee_qc.l0_overnight_tick --run-dir <run-dir> --discover-only` — cheap, read-only, prints which task IDs are newly eligible (per §2) without auditing anything yet.
   2. For each newly-eligible task ID **not already a row in the Sheet**: append a row with `Status = WIP` and the rest blank. **This happens before step 3** — that's the whole point, so the Sheet shows "running" the moment a task is picked up, not just once it's done (which can be 5–10+ minutes later).
   3. `honeybee_qc/l0_run_tick.sh <run-dir> 30` — the actual pull+filter+scrape+audit+append-to-CSV, at **30 concurrent model calls** (see §6 on why 30).
   4. Reads whatever new rows step 3 added to `results.csv` and updates those same Sheet rows (by task ID) in place with the real verdict/reasoning/issues/status — replacing the WIP placeholder, never duplicating the row.
   5. If step 3 failed (nonzero exit), the affected row is left as `WIP`/`error` rather than guessing a result — check `tick_errors.log` for what happened, and it'll just be retried next tick since the task never got added to `processed_ids.txt` on a failed run.

Sheet writes need the Google Sheets MCP connection, which only exists inside
a Cursor chat (currently connected as `mila.avag@gmail.com`) — that's why
step 2 and 4 are agent-turn work and not baked into the shell scripts
themselves.

## 5. Stopping the loop — the kill switch

**The loop does not stop on its own.** It keeps running indefinitely — past
12 hours if nothing tells it otherwise — until you explicitly kill it:

```bash
honeybee_qc/l0_stop_loop.sh honeybee_qc/audit_runs/l0_overnight_20260813
```

This writes a stop file the loop checks every ~60s; if it hasn't exited
within 75s (it should, almost always within a minute), the script kills its
pid directly as a fallback. You'll see a confirmation either way. There is
no other way to stop it short of killing its process/terminal directly — a
single bad tick, a Snowflake outage, a `claude` CLI hiccup, none of that
brings the loop down; it just logs to `tick_errors.log` and tries again next
cycle.

**If the chat this loop was started from gets closed**, the loop process
dies with it (it's a background shell tied to that session) — but nothing
on disk is lost (CSV/cache/`processed_ids.txt`/logs are all safe), and
restarting is just:

```bash
honeybee_qc/l0_loop.sh honeybee_qc/audit_runs/l0_overnight_20260813 7200
```

...from a fresh Cursor chat with the Google Sheets MCP connected, so the
agent turns reacting to its ticks can still do the Sheet-write steps.

## 6. Running it yourself / by hand

**Prerequisites:**
- `python3` with this repo's `honeybee_qc/requirements.txt` installed.
- `claude` CLI on `PATH` and already authenticated (this is what actually
  pays for and makes the model calls — check with `claude --version`).
- Google Chrome installed (for `--fetch-conversations`'s headless scraping;
  check with `python3 -c "from honeybee_qc.sources import find_chrome; print(find_chrome())"`).
- `~/.cursor/redash.env` with a working Snowflake/Redash API key.

**Everything defaults to 30 concurrent model calls** (`--workers 30`) —
both `honeybee_qc.cli` and `honeybee_qc.l0_overnight_tick` default to it now.
That's a deliberate throughput/cost tradeoff for an unattended overnight
run finishing each new task in a few minutes instead of ~10; drop `--workers`
to something smaller (e.g. 6–10) if you start seeing rate-limit errors from
`claude` in `tick_errors.log`.

**See what's eligible without spending anything:**

```bash
cd "/Users/mila.avagimova/final unit tests"
python3 -m honeybee_qc.l0_overnight_tick \
  --run-dir honeybee_qc/audit_runs/l0_overnight_20260813 --discover-only
```

**Run one full tick manually** (safe to re-run — cached calls cost nothing,
already-processed tasks are skipped automatically):

```bash
honeybee_qc/l0_run_tick.sh honeybee_qc/audit_runs/l0_overnight_20260813 30
```

Takes seconds if nothing new is eligible; roughly 3–8 minutes and $2–4 per
new task if there is (scraping + ~80 model calls at 30-way concurrency).

**Re-audit one specific task from scratch** (e.g. if something looks wrong
and you want a clean re-check): remove its line from `processed_ids.txt`,
then re-run the tick command above — it'll be picked up as "new" again. The
model-call cache is keyed by prompt content, so this *will* spend money again
if the transcript changed, but will hit cache for anything unchanged.

**Check just the raw audit output for one task without touching the ledger:**

```bash
# Pull one task's row straight from Snowflake into a CSV honeybee_qc.cli can read
~/.cursor/skills/redash/scripts/redash run \
  -e "select task, attempted_by, review_status, response from public.taskattempts where task = '<TASK_ID>' order by attempt_version desc, attempted_at desc limit 1" \
  -o /tmp/one_task.csv

python3 -m honeybee_qc.cli /tmp/one_task.csv --from-snowflake \
  --fetch-conversations --rating-stage --informed-stage --restricted-checks \
  --cache-db honeybee_qc/audit_runs/l0_overnight_20260813/cache.db \
  --out /tmp/one_task_report.json --workers 30
```

Then read `/tmp/one_task_report.json` — `tasks[0]["checks"]` has every
check's band + full reasoning in `measurement.notes`; `check_270_by_model`
at the top level has the per-model 270 split.

**Quick health check anytime:**

```bash
honeybee_qc/l0_status.sh honeybee_qc/audit_runs/l0_overnight_20260813
```

Shows how many tasks are processed, the last 5 results, cached-call count,
and the tail of the most recent tick's log.

## 7. If something looks wrong

- **A task shows `fail` on a check:** open `tick_report.json` (if it's the
  most recent tick) or re-run the single-task command above and read that
  check's `measurement.notes` / `contributing_items` for the specific
  criteria/turns it disagreed on.
- **Everything is `not_evaluated`:** almost always means scraping failed or
  wasn't attempted — check `hydration` in the report JSON (`dead_pages`,
  `failures`) or just confirm `--fetch-conversations` was actually passed.
- **A row is stuck on `WIP` for a long time:** check `tick_errors.log` —
  most likely the audit for that task failed (Snowflake, CLI, or a timeout
  at 1 hour) and it'll retry on the next tick since it never got marked
  processed. It is not silently stuck forever; each tick re-discovers it
  until it succeeds.
- **The tick errors out entirely:** `tick_errors.log` has the full traceback
  and which phase (discover vs. audit) failed; the loop itself is unaffected
  and will just try again next cycle.
- **Chrome/scraping fails (bot-challenge page, dead share link):** that
  submission's conversation stays empty and its checks abstain honestly —
  this is expected occasionally, not a bug, but worth noting in the Sheet's
  issues column if it happens a lot.
- **Snowflake pull looks stale/wrong:** `~/.cursor/skills/redash/scripts/redash run -e "select count(*) from public.taskattempts where project = '6a7a2d34ac683589810bfa03' and attempted_at >= dateadd(day, -1, current_date())"` to sanity check the row count directly.
- **Rate-limit errors from `claude` in the logs:** lower `--workers` (both in
  `l0_run_tick.sh`'s second argument and if calling `honeybee_qc.cli`
  directly) from 30 to something smaller, like 10.
