Notebooks usage

Run notebooks from the repository root (recommended) so the notebooks locate `data/` at the repo root.

Examples:

```bash
# open Jupyter Lab from repo root
pip install -e backend[dev]
cd /path/to/mendx.ai
jupyter lab
```

Notes
- Notebooks include a `find_data_root()` helper to discover the `data/` directory at either the repository root or `backend/data`.
- If you prefer to run from `backend/`, ensure the `data/` directory exists under `backend/` or set the environment/working directory appropriately.
