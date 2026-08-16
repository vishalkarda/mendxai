# Voice Biomarker Platform — Depression Module (Phase 1)

This repository is being developed as a notebook-first research pipeline for a multi-disease voice biomarker platform. Depression is the first disease module — patient-level detection and severity estimation from voice data — with a roadmap toward additional disorders such as schizophrenia.

## Current Direction

The agreed Phase 1 scope is:

- depression detection first
- patient-level prediction, not file-level prediction
- two parallel objectives:
  - binary classification
  - severity estimation
- handcrafted acoustic and prosodic features as the first baseline
- pretrained speech embeddings in a later phase
- eventual comparison of:
  - handcrafted features only
  - embeddings only
  - hybrid features plus embeddings

## Why The Workflow Starts With EDA

Before we restructure the package or retrain models, we need to understand:

- dataset shape and patient grouping
- class balance
- metadata quality
- clinical score availability
- leakage risks caused by repeated recordings per participant
- raw audio quality and outliers

That is why the first active workstream is under `notebooks/`.

## Planned Notebook Set

The repository now treats the following notebooks as the first execution layer:

- `notebooks/01_dataset_audit.ipynb`
- `notebooks/02_clinical_metadata_eda.ipynb`
- `notebooks/03_audio_signal_eda.ipynb`
- `notebooks/04_feature_baseline_eda.ipynb`

Each notebook is focused on one stage of exploratory analysis so we can keep the research process clear, modular, and reviewable.

## Repository Memory Files

The repository also keeps explicit project memory in:

- `decisions.md` for technical and research decisions
- `execution-flow.md` for the intended end-to-end system flow
- `dev/` for engineering change logs

## Current Codebase Status

The repository already contains baseline implementation files for:

- data loading
- feature extraction
- training
- evaluation
- Decision Tree
- XGBoost
- PyTorch MLP

The source layout is now organized under `src/mendxai/` with a minimal forward-compatible structure:

```text
src/mendxai/
├── core/
├── ml/
└── ml/models/
```

This keeps the repository ready for future additions such as an API layer without overbuilding too early.

## Dataset Assumption

This project is currently designed around the MODMA dataset structure:

```text
data/raw/MDD/participant_xxx/*.wav
data/raw/NC/participant_xxx/*.wav
data/metadata/
```

Expected metadata includes demographic information and clinical scores such as:

- PHQ-9
- HAMD

## Setup With `uv`

### 1. Create the environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -e ".[dev]"
```

### 3. Start notebook work

```bash
jupyter lab
```

## Immediate Execution Order

1. Run the dataset audit notebook.
2. Run metadata EDA.
3. Run audio signal EDA.
4. Run feature baseline EDA.
5. Review findings together.
6. Restructure the package.
7. Build the patient-level depression baseline.
8. Add the severity-estimation workflow.

## Baseline Modeling Recommendation

For the first trustworthy baseline:

- XGBoost is the primary candidate
- the PyTorch MLP is a secondary comparison
- the Decision Tree is a simple interpretability baseline

The main reason is that the current pipeline is feature-based tabular ML, where XGBoost is usually the strongest and most stable starting point for a dataset of this type and scale.

## Important Research Note

This project should be treated as a voice-based screening or risk-assessment system, not a standalone clinical diagnosis system.

## References

- Cai, H., et al. (2020). MODMA dataset: a Multi-modal Open Dataset for Mental-disorder Analysis.
- MODMA Dataset: http://modma.lzu.edu.cn/data/index/
