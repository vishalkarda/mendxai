# mendx.ai

Project summary
-- Purpose: develop and evaluate voice-based methods for detecting depression and estimating severity from speech recordings. The repository contains exploratory notebooks, a Python backend (ML pipelines and service), and scaffolding for a future frontend.

Problem statement
-- Clinical problem: detect depressive disorder and estimate severity using non-invasive audio recordings. The goal is to provide reproducible baselines, rigorous leakage-aware evaluation (patient-level splits), and clear EDA-driven dataset decisions before model changes.

Key components
- Backend: FastAPI service and ML package under `backend/src/mendxai/` (models, data loading, training, evaluation).
- Notebooks: EDA and reproducibility notebooks under `backend/notebooks/` (dataset audit, metadata EDA, audio-signal EDA, feature baseline EDA).
- Data: primary dataset is the MODMA / Lanzhou audio collection; canonical location for runs is `backend/data/audio_lanzhou_2015/` (dataset files not committed to git).
- Frontend (planned): a modern JS app managed by `pnpm`. Options discussed: React, Svelte, SolidJS. Recommendation: Svelte for fast iteration or SolidJS for best runtime performance.

Repository layout

- `backend/` — Python backend application
  - `pyproject.toml` — project/dependency metadata
  - `src/mendxai/` — package code (core, ml, models)
  - `run_dev.sh`, `UV_SETUP.md` — developer run helpers
  - `data/` — runtime data directory (expected location for audio and metadata)
  - `notebooks/` — EDA notebooks
- `data/` — original dataset source (if present); for runs, copy dataset into `backend/data/`
- `.github/` — optional CI workflows such as smoke import checks
- `backend/dev/` — change logs, templates and notes about changes

Getting started — backend (developer)

Prerequisites: Python 3.10+, `pip`, and a virtual environment manager.

Install dev dependencies (recommended):

```bash
python3 -m pip install --upgrade pip
pip install -e backend[dev]
```

Run the backend in development (dev helper chooses `uv` / `uvicorn` or falls back to `python`):

```bash
./backend/run_dev.sh
```

Notebooks and EDA
- Run EDA notebooks via Jupyter Lab from the repository root:

```bash
pip install -e backend[dev]   # optional for notebook deps
jupyter lab
# open backend/notebooks/01_dataset_audit.ipynb
```

- The canonical dataset location for notebooks is `backend/data/audio_lanzhou_2015/` and the metadata workbook `subjects_information_audio_lanzhou_2015.xlsx` should be placed in `backend/data/`.

Data license & privacy
- The repository contains a copy of the MODMA dataset user license at `data/MODMADatasetUserLicenseAgreement (1).docx` and the dataset papers. Handle data according to that license; do not commit raw audio or protected metadata to the repository.

Change management
- Use the per-change template in `backend/dev/template_change.log` and create `backend/dev/change_<n>_YYYYMMDD.log` for each batch so reviews remain focused. Include `Change-ID` linking to PR/Issue and verification/rollback steps.

CI and verification
- A lightweight GitHub Action (optional) runs a smoke import test. Locally, run the import check:

```bash
PYTHONPATH=backend/src python3 -c "import mendxai; print('mendxai import OK')"
```

Frontend options (short guidance)
- React + Vite: familiar ecosystem, large community.
- Svelte / SvelteKit: minimal runtime, fast developer experience — recommended for rapid prototypes.
- SolidJS / SolidStart: React-like mental model with excellent runtime performance — choose if performance is paramount.

Contributing
- Open an issue for substantial changes. For code changes, include a `backend/dev/change_*.log` entry referencing the PR. Follow any repository contribution guidelines (TBD).

Contact
- For questions about dataset licensing or research details, consult the `backend/dev/` notes or reach out to the project lead (add contact info here).

---
This README is intended as the project canonical overview; tell me if you want a `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, or a frontend scaffold next.
