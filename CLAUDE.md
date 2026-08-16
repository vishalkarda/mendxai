# mendx.ai

Voice-based depression detection/severity estimation from the MODMA/Lanzhou dataset. See [README.md](README.md) for the full project overview and [backend/README.md](backend/README.md) for phase/roadmap.

## Environment

- Editable install: `pip install -e backend[dev]`
- venv: `backend/.venv` (uv-managed). Kernel registration for Jupyter/VS Code: see `backend/dev/change_7_2026-08-15.md`.
- Import check: `PYTHONPATH=backend/src python3 -c "import mendxai"`

## Data — read before touching anything in backend/notebooks or backend/data

- Dataset: MODMA / Lanzhou 2015. Raw audio lives at `backend/data/audio_lanzhou_2015/<8-digit-subject-id>/{01..29}.wav` — flat, 52 subjects × 29 files.
- The class label field is `HC` (Healthy Control), **not** `NC`. The clinical scale is **PHQ-9 only** — this dataset has **no HAMD column**, despite older top-level docs implying it.
- Full dataset facts (recording protocol, file-index → task-type mapping, xlsx schema, subject-ID join recipe, license constraints, external benchmark numbers): see [backend/notebooks/DATA_CONTEXT.md](backend/notebooks/DATA_CONTEXT.md) — **read this before any EDA or modeling work.**
- `backend/src/mendxai/core/config.py` and `backend/src/mendxai/ml/data_loader.py` were fixed to match the real flat/HC layout in `backend/dev/change_8_2026-08-16.md`; historical mismatch details are recorded in `DATA_CONTEXT.md` section 6.

## Change tracking

Every batch of changes gets an entry in `backend/dev/` following `backend/dev/template_change.md`, including a "Key Facts / Findings Recorded" section for anything newly learned about the data or codebase.

## Data handling

- Do not commit raw audio, the metadata workbook, or the source PDFs/docx (`backend/data/` is gitignored).
- Usage is governed by `backend/data/MODMADatasetUserLicenseAgreement (1).docx`.
