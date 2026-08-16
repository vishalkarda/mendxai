# mendx.ai

Multi-disease voice biomarker platform — voice-based detection/severity estimation across conditions from speech. Depression (MODMA/Lanzhou dataset) is the first disease module. See [README.md](README.md) for the full project overview and [backend/README.md](backend/README.md) for phase/roadmap.

## Environment — verified, don't re-check

- Editable install: `pip install -e backend[dev]`
- venv: `backend/.venv` (uv-managed). Kernel registration for Jupyter/VS Code: see `backend/dev/change_7_2026-08-15.md`.
- Import check: `PYTHONPATH=backend/src python3 -c "import mendxai"`
- All packages in `backend/pyproject.toml` are installed in `backend/.venv` and confirmed working — core: numpy, pandas, librosa, scikit-learn, xgboost, torch, matplotlib, seaborn, tqdm, pyyaml, openpyxl, ipykernel; dev extras: pytest, black, ruff, ipywidgets, jupyterlab, uv, uvicorn. This includes pandas and openpyxl specifically, which the EDA notebooks depend on.
- **Treat this file and `backend/pyproject.toml`/`uv.lock` as ground truth.** Do not re-run `import X` probes, `pip show`, or other dependency-existence checks before using a package listed above. Only re-verify if a command actually fails at runtime — then fix it and update this section.
- Subagents: this file loads automatically for any Claude Code session or spawned agent working in this repo. Read it (and `DATA_CONTEXT.md` for data work) before running discovery/setup commands — don't repeat checks it already answers.

## Data — read before touching anything in backend/notebooks or backend/data

- Dataset: MODMA / Lanzhou 2015. Raw audio lives at `backend/data/audio_lanzhou_2015/<8-digit-subject-id>/{01..29}.wav` — flat, 52 subjects × 29 files.
- The class label field is `HC` (Healthy Control), **not** `NC`. The clinical scale is **PHQ-9 only** — this dataset has **no HAMD column**, despite older top-level docs implying it.
- Full dataset facts (recording protocol, file-index → task-type mapping, xlsx schema, subject-ID join recipe, license constraints, external benchmark numbers): see [backend/notebooks/DATA_CONTEXT.md](backend/notebooks/DATA_CONTEXT.md) — **read this before any EDA or modeling work.**
- For the narrative deep-dive behind those facts (what each column means and why, the MDD/HC/NC terminology history, EDA findings recap, prior-art modeling methodology, license/compliance details) see [backend/docs/data_research/](backend/docs/data_research/README.md).
- `backend/src/mendxai/core/config.py` and `backend/src/mendxai/ml/data_loader.py` were fixed to match the real flat/HC layout in `backend/dev/change_8_2026-08-16.md`; historical mismatch details are recorded in `DATA_CONTEXT.md` section 6.

## Change tracking

Every batch of changes gets an entry in `backend/dev/` following `backend/dev/template_change.md`, including a "Key Facts / Findings Recorded" section for anything newly learned about the data or codebase. If that finding is a durable environment/dependency/data fact (not a one-off), mirror it back into this file (`CLAUDE.md`) or `DATA_CONTEXT.md` so future sessions don't rediscover it.

## Data handling

- Do not commit raw audio, the metadata workbook, or the source PDFs/docx (`backend/data/` is gitignored).
- Usage is governed by `backend/data/MODMADatasetUserLicenseAgreement (1).docx`.
