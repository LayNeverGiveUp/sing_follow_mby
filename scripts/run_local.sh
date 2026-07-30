#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: bash scripts/setup_local.sh" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

.venv/bin/python tools/bootstrap_runtime.py
exec .venv/bin/python -m uvicorn app.main:app \
  --host "${HUM_HOST:-127.0.0.1}" \
  --port "${HUM_PORT:-8000}"
