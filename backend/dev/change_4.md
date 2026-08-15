# Change 4: Backend Re-rooting and Docker Base

Date: 2026-08-15
Status: Implemented

## Purpose

This change converts the repository into a root-level container with a dedicated `backend/` application directory. The goal is to make room for a future frontend and keep the current Python application isolated as the backend service.

## What Changed

- Moved the current Python application and project documents into `backend/`
- Kept `.git/` at the repository root
- Kept the `src/mendxai/` package layout inside `backend/`
- Added `backend/Dockerfile` for containerizing the backend application
- Added `backend/.gitignore` so ignore rules continue to apply correctly inside the backend subtree
- Simplified the root `.gitignore` to only handle root-level artifacts

## Why This Structure

- It makes the repository ready for a future `frontend/` sibling directory
- It keeps backend packaging, notebooks, and source code together
- It avoids another large directory move when the frontend is introduced

## What Did Not Change

- No model behavior was intentionally changed
- No EDA notebook logic was filled in during this step
- No API layer was added yet

## Impact

- Backend work should now happen from the `backend/` directory
- Future Docker, FastAPI, and service work can stay contained in the backend subtree
