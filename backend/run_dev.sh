#!/usr/bin/env bash
set -euo pipefail

# Developer runner: try `uv` (PEP-723 runner), then `uvicorn`, then direct python
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"

if command -v uv >/dev/null 2>&1; then
  echo "Running with uv"
  uv run "$ROOT_DIR"
elif command -v uvicorn >/dev/null 2>&1; then
  echo "uv not found; running with uvicorn"
  # assume ASGI app is in main:app or mendxai.app
  if python -c "import importlib,sys
try:
    importlib.import_module('mendxai')
    print('FOUND')
except Exception:
    sys.exit(1)" >/dev/null 2>&1; then
    uvicorn main:app --reload --app-dir "$ROOT_DIR"
  else
    uvicorn mendxai.main:app --reload --app-dir "$ROOT_DIR"
  fi
else
  echo "No uv/uvicorn found; running direct python (CLI mode)"
  python "$ROOT_DIR/main.py"
fi
