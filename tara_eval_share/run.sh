#!/usr/bin/env bash
# One-shot launcher: makes a venv, installs deps, starts the server.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://litellm.ml.scaleinternal.com/}"
export TARA_LITELLM_MODEL="${TARA_LITELLM_MODEL:-claude-opus-4-6}"

# 1. Check LiteLLM credentials are available.
if [ -z "${LITELLM_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
  echo "ERROR: LITELLM_API_KEY is not set."
  echo "Set it before starting, e.g.:"
  echo "  export LITELLM_API_KEY='...'"
  exit 1
fi

# 2. Virtual env + deps.
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 3. Launch.
echo ""
echo "Tara Eval is starting on http://localhost:${PORT}"
echo "LiteLLM base URL: ${LITELLM_BASE_URL}"
echo "Model: ${TARA_LITELLM_MODEL}"
echo "Press Ctrl+C to stop."
echo ""
exec uvicorn app:app --host 127.0.0.1 --port "${PORT}"
