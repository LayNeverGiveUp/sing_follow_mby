#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/bootstrap_runtime.py

if [[ "${INSTALL_QUERY_ASSETS:-1}" != "0" ]]; then
  QUERY_ASSET_DIR="data/queries/mao_buyi_v1"
  if [[ ! -f "$QUERY_ASSET_DIR/.asset-version.json" ]] && \
    .venv/bin/python -c 'from pathlib import Path; raise SystemExit(0 if any(Path("data/queries/mao_buyi_v1").rglob("*.wav")) else 1)'; then
    echo "Local query WAV files already exist; keeping them unchanged."
  else
    .venv/bin/python tools/install_query_assets.py
  fi
else
  echo "Skipping optional query WAV assets (INSTALL_QUERY_ASSETS=0)."
fi

echo
echo "Setup complete. Start the service with:"
echo "  bash scripts/run_local.sh"
