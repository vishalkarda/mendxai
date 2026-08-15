UV / dev-runner setup

Purpose
- Provide instructions for using `uv` (PEP-723 runner) if you use it, and fallbacks using `uvicorn` for ASGI apps.

Install
- Preferred (if you use `uv`):

```bash
pip install uv
```

- Common fallback for FastAPI/ASGI apps:

```bash
pip install uvicorn[standard]
```

Running the backend (developer workflow)
- If your project exposes an ASGI `app` object (e.g., `mendxai.app` or in `main.py`):

With `uv` (PEP-723 aware runner):

```bash
uv run backend
```

With `uvicorn` (explicit):

```bash
PYTHONPATH=backend/src uvicorn main:app --reload --app-dir backend
```

Quick local smoke import test (no install):

```bash
PYTHONPATH=backend/src python -c "import mendxai; print('mendxai OK')"
```

Notes
- `uv run` is helpful for PEP-723 inline script resolution and for some editor debug integrations. If you don't use PEP-723 entrypoints, `uvicorn` is a safe alternative for ASGI apps.
- If your backend is not ASGI (CLI app using `main.py`), run:

```bash
PYTHONPATH=backend/src python backend/main.py
```

If you'd like, I can add a small `run_dev.sh` wrapper that picks `uv` when available and falls back to `uvicorn` or direct `python` runs. Let me know if you want that created and whether `backend/main.py` exposes an `app` object or is a CLI entrypoint.
