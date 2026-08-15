Notebooks usage

Run notebooks from the repository root (recommended) so the notebooks locate `data/` under `backend/data`.

Examples:

```bash
# install dev deps (optional for notebooks)
pip install -e backend[dev]
cd /path/to/mendx.ai
jupyter lab
```

Notes
- The project data is expected to live at `backend/data/` (specifically `backend/data/audio_lanzhou_2015`).
- Notebooks have been updated to use the fixed `backend/data` location; ensure you've copied the dataset there before running.
