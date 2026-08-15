# Change 2: EDA Workstream Introduction

Date: 2026-08-15
Status: Planned and scaffolded

## Purpose

This change introduces the notebook-first workflow for the project. The goal is to understand the dataset, metadata, patient structure, and feature behavior before changing training logic or restructuring the package.

## Why EDA Comes First

- The project will make patient-level predictions, so we must understand participant structure before training.
- Severity estimation depends on metadata quality, especially clinical scores such as PHQ-9 and HAMD.
- Audio datasets often contain hidden data quality issues such as missing recordings, inconsistent sampling, silence-heavy files, or naming mismatches.
- A trustworthy baseline depends on leakage-aware exploratory analysis, not just running classifiers.

## Notebooks Introduced

### `01_dataset_audit.ipynb`

Focus:
- inspect raw folder layout
- count participants and recordings by class
- identify missing or inconsistent files
- confirm patient-level grouping assumptions

### `02_clinical_metadata_eda.ipynb`

Focus:
- inspect metadata columns
- study missingness
- review PHQ-9 and HAMD distributions
- evaluate whether severity estimation is feasible and how labels might be defined

### `03_audio_signal_eda.ipynb`

Focus:
- inspect sample rates, durations, silence, and basic signal quality
- visualize waveform and spectrogram examples
- spot obvious outliers before feature extraction

### `04_feature_baseline_eda.ipynb`

Focus:
- study extracted feature distributions
- compare class separation
- inspect redundancy and outliers
- compare file-level vs patient-level aggregation behavior

## Expected Output From This Workstream

- a validated understanding of the dataset shape
- a severity target design strategy
- a data leakage prevention strategy
- a clear decision about which baseline metrics and aggregation level to use

## Impact

- No training behavior is changed in this step.
- This change improves collaboration, traceability, and confidence before deeper implementation work begins.
