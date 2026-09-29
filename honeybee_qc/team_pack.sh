#!/usr/bin/env bash
# Build a small source tar + zip for AirDrop / Slack.
# Excludes audit_runs, caches, snapshots, pulled CSVs, secrets, tests, leftovers.
# The archives exist only to send the Python package without that junk.
set -euo pipefail

# Do not pack macOS AppleDouble / resource-fork files.
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/hbqc_pack.XXXXXX")"
NAME="honeybee_qc_team"
DEST="$SCRIPT_DIR/dist"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$STAGING/$NAME/honeybee_qc/dist/run_layout/snapshots" \
         "$STAGING/$NAME/honeybee_qc/dist/run_layout/fixer" \
         "$DEST/run_layout/snapshots" \
         "$DEST/run_layout/fixer"

# Run-path modules only (cli → fixer → fixer_invariants → backfill_steps + deps).
KEEP_PY=(
  __init__.py
  backfill_steps.py
  cache.py
  cli.py
  config.py
  context.py
  env_context.py
  errors.py
  fill_dims.py
  findings.py
  fixer.py
  fixer_invariants.py
  gates.py
  hydrate.py
  informed_prompts.py
  informed_stages.py
  ingest.py
  links.py
  llm.py
  models.py
  preflight.py
  projects.py
  prompts.py
  provenance.py
  rating_prompts.py
  rating_stages.py
  registry.py
  remap_gpt_as_b.py
  rollup.py
  sampling.py
  scoring.py
  sources.py
  stages.py
  taxonomies.py
  transcripts.py
  v5_prompt.py
  verify.py
)

rsync -a \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='._*' \
  "${KEEP_PY[@]/#/$SCRIPT_DIR/}" \
  "$STAGING/$NAME/honeybee_qc/"

cp "$SCRIPT_DIR/requirements.txt" "$STAGING/$NAME/honeybee_qc/"
cp "$SCRIPT_DIR/TEAM_RUN.md" "$STAGING/$NAME/honeybee_qc/"
cp "$SCRIPT_DIR/TEAM_RUN.md" "$STAGING/$NAME/README.md"
cp "$SCRIPT_DIR/README.md" "$STAGING/$NAME/honeybee_qc/README.md"
cp "$SCRIPT_DIR/team_run.sh" "$STAGING/$NAME/honeybee_qc/"
cp "$SCRIPT_DIR/team_run.sh" "$STAGING/$NAME/team_run.sh"
chmod +x "$STAGING/$NAME/honeybee_qc/team_run.sh" "$STAGING/$NAME/team_run.sh"

cat > "$STAGING/$NAME/honeybee_qc/dist/run_layout/README.txt" <<'EOF'
Empty out-dir layout. Copy this folder, drop pulled.csv here, then:

  ./honeybee_qc/team_run.sh all \
    --csv /path/to/pulled.csv \
    --project l1_sxs \
    --slots A,B \
    --out honeybee_qc/dist/run_layout

Created by the run: report.json, cache.db, snapshots/, fixer/.
Do not commit those. Do not Slack cache.db or snapshots/.
EOF
cp "$STAGING/$NAME/honeybee_qc/dist/run_layout/README.txt" \
   "$DEST/run_layout/README.txt"

cat > "$STAGING/$NAME/START_HERE.txt" <<'EOF'
Honeybee QC teammate package.

You are in the folder that contains honeybee_qc/. That is the working directory.

1. Read honeybee_qc/TEAM_RUN.md (from-zero guide).
2. Need Homebrew Python 3.13, Chrome, claude CLI, ANTHROPIC_API_KEY or `claude login`, ~10 GB free.
3. /opt/homebrew/bin/python3 -m pip install -r honeybee_qc/requirements.txt
4. Get a taskattempts CSV (columns: task, project, attempted_by, review_status, response)
   or: ./honeybee_qc/team_run.sh pull-sql --project l1_sxs --ids-file ids.txt > pull.sql
5. ./honeybee_qc/team_run.sh all --csv pulled.csv --project l1_sxs --slots A,B --out honeybee_qc/runs/my_batch

Do not put keys in this folder. Do not kill Chrome. --hydrate-workers stays at 8.
The zip and the tar are the same tree.
EOF

mkdir -p "$DEST"
OUT_TAR="$DEST/honeybee_qc_team.tar.gz"
OUT_ZIP="$DEST/honeybee_qc_team.zip"
tar -C "$STAGING" -czf "$OUT_TAR" "$NAME"
# Finder double-click friendly. Same tree as the tar.
rm -f "$OUT_ZIP"
(cd "$STAGING" && zip -r -X -q "$OUT_ZIP" "$NAME")

echo "wrote $OUT_TAR ($(du -h "$OUT_TAR" | awk '{print $1}'))"
echo "wrote $OUT_ZIP ($(du -h "$OUT_ZIP" | awk '{print $1}'))"
echo "extract: double-click the zip, or: tar xzf honeybee_qc_team.tar.gz && cd $NAME"
