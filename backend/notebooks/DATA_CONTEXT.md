# MODMA / Lanzhou 2015 — Data Context

Durable reference for everything about this dataset that isn't obvious from the code. Read this before writing or reviewing any EDA or modeling work. Facts here were established on 2026-08-16 by reading the dataset paper, the companion methods paper, and the metadata workbook directly (see `backend/dev/change_8_2026-08-16.md`).

## 1. Dataset identity & source

- Raw audio: `backend/data/audio_lanzhou_2015/<8-digit-subject-id>/01.wav`..`29.wav`
- Metadata workbook: `backend/data/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx` — note it lives **inside** the dataset subdirectory, not directly under `backend/data/`. (This was the bug in the original `01_dataset_audit.ipynb`.)
- Companion documents present locally (not committed to git — see `backend/.gitignore`):
  - `backend/data/audio_lanzhou_2015/MODMA dataset-a Multi-modal Open Dataset for Mental-disorder Analysis.pdf` — the primary dataset paper (Cai et al., Lanzhou University).
  - `backend/data/audio_lanzhou_2015/A Novel Decision Tree for Depression Recognition in Speech.pdf` — companion methods paper, same 52-subject cohort.
  - `backend/data/MODMADatasetUserLicenseAgreement (1).docx` — usage license.

## 2. Recording protocol (from the MODMA paper)

- Quiet/soundproof room, <60dB ambient noise; Neumann TLM102 microphone; RME Fireface UCX audio interface.
- 44.1kHz / 24-bit, uncompressed WAV.
- Diagnosis: structured MINI interview meeting DSM-IV criteria. **PHQ-9 ≥ 5 required for MDD inclusion.** No psychotropic medication in the prior two weeks.
- Ethics approval: Lanzhou University Second Hospital Ethics Committee; informed consent obtained; participants compensated.

## 3. File-index → task-type mapping

Not documented anywhere else in the repo. Verified against the paper's session structure and the on-disk file counts (18+1+6+4 = 29, matching every subject folder):

| File index | Task type |
|---|---|
| 01–18 | Interview questions (18 files) |
| 19 | Passage reading (1 file) |
| 20–25 | Word reading (6 files) |
| 26–29 | Picture description (4 files) |

Total: 29 files/subject × 52 subjects = 1508 files.

## 4. Metadata workbook schema

`subjects_information_audio_lanzhou_2015.xlsx`, `Sheet1` only (`Sheet2`/`Sheet3` empty), 52 rows × 13 columns:

- **11 real columns**: `subject id`, `type`, `age`, `gender`, `education（years）` (note: full-width parenthesis in the actual column name), `PHQ-9`, `CTQ-SF`, `LES`, `SSRS`, `GAD-7`, `PSQI`.
- **2 junk columns**: `Unnamed: 11` (blank), `Unnamed: 12` (holds an inline legend/remarks block, not data) — drop both on load.
- Zero missing values across the 11 real columns.
- Scale glossary (from the workbook's own legend column): PHQ-9 = Patient Health Questionnaire; CTQ-SF = Childhood Trauma Questionnaire; LES = Life Event Scale; SSRS = Social Support Research Scale; GAD-7 = Generalized Anxiety Disorder scale; PSQI = Pittsburgh Sleep Quality Index.
- **There is no HAMD scale in this dataset.** Only PHQ-9 is used for depression severity/inclusion. Any repo text mentioning HAMD (e.g. older top-level README wording) is stale for this specific dataset.

## 5. Class / label conventions

- Label column: `type`, values `MDD` / `HC` — **not** `MDD`/`NC`.
- Class balance: MDD = 23, HC = 29 (52 subjects total).
- Gender: M = 36, F = 16. Age: 18–52, mean ≈ 31.3.
- **Subject-ID join recipe** (verified 1:1 match, zero orphans on either side):
  ```python
  folder_id = subject_dir.name                    # e.g. "02010002"
  xlsx_id   = str(row["subject id"]).zfill(8)      # e.g. 2010002 -> "02010002"
  ```

## 6. Known repo mismatches (historical — fixed in change_8)

- **Before the fix**: `backend/src/mendxai/core/config.py`'s `DataConfig` assumed `data/raw/{MDD,NC}/<participant>/*.wav`; `backend/src/mendxai/ml/data_loader.py`'s `DataLoader` and `verify_data_structure()` were built entirely around that wrong assumption. Both wrong: the real layout is flat, and the label is `HC`, not `NC`.
- **Before the fix**: `Config.__post_init__` auto-created these wrong directories as an import-time side effect (`mkdir` on `data/raw/MDD`, `data/raw/NC`, etc., relative to CWD) any time `mendxai.core.config` was imported — a real, active bug, not hypothetical.
- **Status**: fixed as of `backend/dev/change_8_2026-08-16.md`. `DataLoader` now walks the flat structure and joins against the xlsx via the recipe in §5; `classify_task()` implements the §3 mapping; directory creation is now explicit (`Config.ensure_output_dirs()`), never an import side effect.

## 7. External benchmark reference

The companion methods paper ("A Novel Decision Tree for Depression Recognition in Speech", same 52-subject cohort) reports **75.8% (male) / 68.5% (female)** accuracy using a decision-tree speech-segment-fusion method. Useful as an external sanity target for any future baseline model in this repo — not reproduced here.

## 8. License / data-handling constraints

- Usage governed by `backend/data/MODMADatasetUserLicenseAgreement (1).docx`. Not yet deep-parsed for specific redistribution/publication clauses — read it before sharing derived features or writeups externally.
- Raw audio, the metadata xlsx, and the source PDFs/docx are gitignored (`backend/data/`) — never commit them.

## 9. Known data-quality issues

- Subject `02010004`, files `24.wav`–`28.wav` (5 files: word-reading tasks 24/25, picture-description tasks 26/27/28) are **unreadable** — `soundfile.info()` fails with "Format not recognised", and `file` reports them as generic `data` (no recognizable RIFF/WAVE header), despite plausible non-zero file sizes (2–4.5MB). Discovered by `01_dataset_audit.ipynb`'s corrupt-file check (not detected by the original pre-`change_8` version of that notebook, which never validated file readability). Downstream notebooks (03, 04) should exclude these 5 files rather than fail on them.
- No exact-duplicate files were found across the corpus (content-fingerprint check, `01_dataset_audit.ipynb`).
- **Recordings sit at a low overall level relative to digital full scale**: mean/median peak ≈ -40dBFS across the 1503 valid files (`03_audio_signal_eda.ipynb`). An absolute-dBFS silence threshold is therefore wrong for this corpus (it would flag ~100% of files as silent) — silence must be measured relative to each file's own peak. No clipping was detected anywhere in the corpus.

## 10. Cached processed artifacts

Produced by the EDA notebooks, consumed downstream. All under `backend/data/processed/`:

| File | Produced by | Contents |
|---|---|---|
| `subjects_labeled.csv` | `01_dataset_audit.ipynb` | Per-subject: id, type/label, demographics, file counts |
| `audio_qa_manifest.csv` | `03_audio_signal_eda.ipynb` | Per-file: subject_id, label, file_index, task_type, duration, sr, silence_ratio, clip_count, flagged |
| `features.csv` | `04_feature_baseline_eda.ipynb` (Part A) | Per-file extracted acoustic/prosodic features + label |
