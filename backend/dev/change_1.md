# Change 1: Existing Project State

Date: 2026-08-15
Status: Documented baseline state before restructuring

## Purpose

This file captures the repository exactly as it existed before the first planned cleanup pass. It serves as the starting point for all future project changes.

## Current Repository State

- The repository contains a compact Python voice-analysis project aimed at depression detection from MODMA voice data.
- The codebase currently includes:
  - `main.py` as the CLI entrypoint
  - `config.py`, `data_loader.py`, `feature_extractor.py`, `trainer.py`, and `evaluator.py`
  - model implementations for:
    - Decision Tree
    - XGBoost
    - PyTorch MLP
- The code is written as if it belongs to a `src/depression_detector` package, but the files currently live in the repository root.

## Observed Structural Issues

- `main.py` imports `src.depression_detector`, but the `src/` directory does not exist in the repository.
- `pyproject.toml` is configured to package `src/depression_detector`, which also does not exist yet.
- `__init__-main.py` and `__init__-models.py` appear to be placeholder files intended to become package `__init__.py` files later.
- The project is syntactically valid, but not runnable as a package in its current layout.

## Modeling State

- The current implemented approach is feature-based tabular machine learning.
- Features are extracted from audio using `librosa`.
- The project currently focuses on binary depression classification.
- The project does not yet contain a dedicated severity-estimation pipeline.
- The project does not yet contain pretrained speech embeddings.

## Evaluation Risks Identified

- The current split logic is file-level, not patient-level.
- Because MODMA contains repeated recordings per participant, the same participant can appear in both train and test sets under the current logic.
- This introduces leakage risk and can inflate downstream evaluation metrics.

## Project Direction Agreed With User

- Start with depression detection only.
- Use patient-level prediction.
- Build both:
  - binary classification
  - severity estimation
- Establish a handcrafted-feature baseline first.
- Later add pretrained speech embeddings.
- Longer term, consider combining handcrafted features and embeddings.

## Immediate Next Step After This Document

- Introduce notebook-first exploratory analysis under `notebooks/`.
- Add project memory documents and a running change log.
- Delay package restructuring until after EDA scaffolding is in place and reviewed.
