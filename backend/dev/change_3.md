# Change 3: Source Layout Restructure

Date: 2026-08-15
Status: Implemented

## Purpose

This change restructures the repository into a proper `src/` layout using the product-aligned package name `mendxai`. The goal is to make the codebase cleaner now while also leaving room for future backend and frontend integration.

## What Changed

- Created:
  - `src/mendxai/`
  - `src/mendxai/core/`
  - `src/mendxai/ml/`
  - `src/mendxai/ml/models/`
- Moved configuration into `src/mendxai/core/config.py`
- Moved data loading, feature extraction, training, and evaluation into `src/mendxai/ml/`
- Moved model implementations into `src/mendxai/ml/models/`
- Converted placeholder init files into real package `__init__.py` files
- Updated imports across the codebase to use the new package layout
- Kept `main.py` at the repository root as the CLI entrypoint

## Why This Structure

- It follows a standard Python `src/` layout.
- It uses `mendxai` as the long-term package namespace.
- It keeps the ML pipeline isolated but ready to sit alongside future layers such as:
  - API
  - services
  - schemas
  - frontend integration support

## Additional Hygiene Changes

- Renamed `gitignore` to `.gitignore`
- Renamed `python-version` to `.python-version`
- Updated the ignore rules so notebooks can remain version-controlled

## What Did Not Change

- No model behavior was intentionally changed
- No evaluation logic was intentionally changed
- No notebook contents were filled in during this step

## Impact

- The repository now matches the intended package structure more closely
- Future EDA notebooks can import from `mendxai` cleanly
- Future FastAPI-oriented growth should require less structural rework
