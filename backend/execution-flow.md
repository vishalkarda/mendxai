# Execution Flow

Last updated: 2026-08-15

## Objective

This document describes the intended end-to-end execution flow for the project as it evolves from its current state into a structured research pipeline.

## Planned High-Level Flow

### 1. Environment Setup

- Create and manage the environment with `uv`.
- Install runtime dependencies from `pyproject.toml`.
- Install notebook dependencies for exploratory work.

### 2. Dataset Intake

- Place raw voice data under `data/raw/`.
- Organize participants by class directory, such as `MDD/` and `NC/`.
- Place clinical metadata under `data/metadata/`.

### 3. Notebook-First EDA

- Run `notebooks/01_dataset_audit.ipynb` to validate dataset structure.
- Run `notebooks/02_clinical_metadata_eda.ipynb` to understand labels and severity information.
- Run `notebooks/03_audio_signal_eda.ipynb` to inspect raw audio quality.
- Run `notebooks/04_feature_baseline_eda.ipynb` after feature extraction is available.

### 4. Source Layout

- Keep application code under `src/mendxai/`.
- Use `core/` for shared configuration and future common infrastructure.
- Use `ml/` for the research and modeling pipeline.
- Keep the root `main.py` as the current CLI entrypoint until a fuller app/backend interface is introduced.

### 5. Feature Extraction

- Extract handcrafted acoustic and prosodic features from each recording.
- Save extracted features in a reproducible processed-data location.
- Introduce patient identifiers so downstream aggregation is explicit.

### 6. Patient-Level Aggregation and Splitting

- Split data at the participant level, not the file level.
- Aggregate or group features by patient for core prediction tasks.
- Prevent train/test leakage across recordings from the same participant.

### 7. Baseline Modeling

- Train the primary depression baseline with XGBoost.
- Compare against the Decision Tree and MLP baselines.
- Evaluate using patient-level metrics and clinically interpretable summaries.

### 8. Severity Workflow

- Build a separate pipeline for severity estimation.
- Decide whether the target should be:
  - regression
  - ordinal categories
  - severity bins
- Base that decision on the EDA of PHQ-9 and HAMD metadata.

### 9. Future Representation Learning

- Add pretrained speech embeddings in a dedicated later phase.
- Compare:
  - handcrafted features only
  - embeddings only
  - hybrid features plus embeddings

## Collaboration Rule

- Every material code or file change should be proposed first.
- The expected impact should be explained before implementation.
- Changes should be documented in `dev/` so the project keeps a running engineering memory.
