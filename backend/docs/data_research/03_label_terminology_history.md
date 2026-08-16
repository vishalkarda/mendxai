# Label Terminology: MDD, HC, and NC

If you've read through this codebase's older files, you've probably seen the control group called both `HC` and `NC` in different places, with no obvious explanation of which one is "right." This file is the direct answer: where "NC" actually came from, how it ended up scattered across this repository, and what the corrected, verified truth is.

## The short answer

The real metadata workbook — `backend/data/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx`, the authoritative source for every subject's class label — uses a column called `type`, with exactly two values: `MDD` (Major Depressive Disorder) and `HC` (Healthy Control). That's it. There are 23 MDD subjects and 29 HC subjects, verified against all 52 folders on disk with zero mismatches. `NC` does not appear anywhere in the actual data. Wherever you see `NC` in this codebase, it's a leftover naming mistake, not an alternate-but-valid label.

## Where "NC" actually came from

This is the part worth understanding properly, because "NC" wasn't a random typo — it was copied from a real source, just the wrong part of that source. The MODMA dataset paper describes *two* modalities collected from overlapping subject pools: EEG (both a 128-electrode setup and a lightweight 3-electrode wearable) and audio. In the sections of that paper describing the EEG cohort, the authors write, verbatim: *"The normal controls (NC) were recruited by posters."* They go on to use "NC" repeatedly as shorthand throughout the EEG-related sections — for instance, describing how files with a `0201` subject-ID prefix belong to MDD patients while `0203`-prefixed files belong to NC subjects, in the context of the EEG data records.

So "NC" is a real abbreviation, explicitly spelled out as "Normal Control," in a real source document. It's just describing the *EEG* cohort's naming convention, not the audio cohort's. The companion methods paper that actually models the 52-subject *audio* cohort this repository uses never uses "NC" at all — it consistently writes "healthy controls" or "healthy group" in prose, and its result tables label the two classes as "Healthy controls" and "Depression patients." Even between the two source papers, there's a small terminology drift describing the same audio subjects.

## How it leaked into this codebase

The earliest committed version of this repository's data-loading code (`backend/src/mendxai/core/config.py`, from the initial "created the backend dir" commit) defined the class labels like this:

```python
mdd_label: str = "MDD"  # Major Depressive Disorder
nc_label: str = "NC"    # Normal Control
```

That inline comment is the smoking gun: whoever scaffolded this code read "normal controls (NC)" somewhere in the MODMA paper — almost certainly from the EEG-cohort description, since that's the only place in either paper that phrase appears — and carried the abbreviation over into the audio pipeline's label convention, without cross-checking it against the actual audio metadata workbook, which uses `HC`. From there, `nc_label` propagated everywhere the original `data_loader.py` touched: it built an `NC/` subdirectory path, printed `"Loading NC audio files..."`, and reported `"NC: {n} files"` in its summary output. It also assumed a folder layout (`data/raw/MDD/` and `data/raw/NC/`) that didn't match the dataset's real flat structure at all — a separate but related mismatch, covered in `DATA_CONTEXT.md` section 6 and fixed in the same round of changes as the label correction.

## Where the fix landed, and where it's still incomplete

On 2026-08-16 (`backend/dev/change_8_2026-08-16.md`), `config.py` and `data_loader.py` were rewritten to match reality: the label is now `hc_label = "HC"`, the folder-walk logic matches the real flat layout, and the fix was verified against the real counts (1508 files, 52 subjects, 667 MDD-labeled files, 841 HC-labeled files). `feature_extractor.py`'s stray `"NC samples"` print statement was caught and fixed in the same pass.

Two files were missed by that fix and, as of this writing, still print or label the control group as `NC`:

- **`backend/src/mendxai/ml/evaluator.py`** — its confusion-matrix and classification-report code still hardcodes `target_names=['NC (Healthy)', 'MDD (Depression)']` and axis tick labels `['NC', 'MDD']`.
- **`backend/src/mendxai/ml/trainer.py`** — still prints `"NC samples: {n}"` when reporting a prepared dataset's class balance.

Three documentation files also still describe the old, incorrect `MDD`/`NC` folder-split convention and haven't been updated to match the real flat/`HC` layout: `backend/README.md` (line 82, and its mention of a "HAMD" clinical field that also doesn't exist in this dataset — see `DATA_CONTEXT.md`), `backend/SETUP_GUIDE.md` (lines 37, 81, 97), and `backend/execution-flow.md` (line 20).

None of this affects the correctness of the EDA notebooks or the fixed `config.py`/`data_loader.py` — the actual data pipeline now uses `HC` correctly throughout. But if you're reading `evaluator.py`, `trainer.py`, or any of those three docs and see `NC`, this is why, and it's a known, narrowly-scoped cleanup item rather than a sign of a deeper data problem.

## The one-sentence summary

`MDD` and `HC` are the real, verified class labels in the actual audio metadata. `NC` is a same-meaning abbreviation borrowed from the wrong part of the source paper (the EEG cohort's convention, not the audio cohort's) early in this repo's history, fixed in the core data-loading path as of `change_8`, but still present in two model-evaluation files and three older docs as a not-yet-cleaned-up loose end.
