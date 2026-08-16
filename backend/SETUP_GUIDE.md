# SETUP GUIDE — MacBook M-Series

This guide covers running the **depression module** (Phase 1 of the mendx.ai multi-disease voice
biomarker platform) end to end via `backend/main.py`. Future disease modules will get their own
pipelines under `backend/src/mendxai/`; this guide is specific to depression for now.

For repo-wide orientation (dataset facts, environment, change tracking) read
[`CLAUDE.md`](../CLAUDE.md) and [`notebooks/DATA_CONTEXT.md`](notebooks/DATA_CONTEXT.md) first —
this doc assumes those facts and doesn't repeat them.

## Quick Start (5 minutes)

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart terminal, then verify:
```bash
uv --version
```

### 2. Set up the environment (from the repo root)
```bash
cd mendx.ai
uv venv backend/.venv
source backend/.venv/bin/activate
uv pip install -e "backend[dev]"
```

See [`CLAUDE.md`](../CLAUDE.md) — the full dependency set (pandas, librosa, xgboost, torch,
openpyxl, etc.) is already verified to install cleanly this way; no need to re-check individual
packages.

### 3. Repository structure

```
mendx.ai/
├── CLAUDE.md
├── README.md
└── backend/
    ├── pyproject.toml
    ├── main.py                  # depression-module CLI entrypoint
    ├── src/mendxai/
    │   ├── core/config.py
    │   └── ml/
    │       ├── data_loader.py
    │       ├── feature_extractor.py
    │       ├── trainer.py
    │       ├── evaluator.py
    │       └── models/
    │           ├── decision_tree.py
    │           ├── xgboost_model.py
    │           └── neural_net.py
    ├── notebooks/                # EDA — run before touching the pipeline below
    ├── data/                     # gitignored — see "Dataset placement" below
    └── results/
        ├── models/
        ├── logs/
        └── figures/
```

### 4. Dataset placement

The real on-disk layout is **flat** — there is no MDD/HC split in the filesystem. Class label
comes only from joining subject IDs against the metadata workbook's `type` column. Full details:
[`notebooks/DATA_CONTEXT.md`](notebooks/DATA_CONTEXT.md).

```bash
mkdir -p backend/data
cp -r /path/to/MODMA/audio_lanzhou_2015 backend/data/
```

Expected structure:
```
backend/data/audio_lanzhou_2015/
  02010002/
    01.wav ... 29.wav
  02010004/
    01.wav ... 29.wav   # note: files 24-28 for this subject are corrupt, see DATA_CONTEXT.md §9
  ... (52 subject folders total)
  subjects_information_audio_lanzhou_2015.xlsx   # lives inside this dataset subdirectory
```

52 subjects × 29 files = 1508 recordings (1503 readable). Class balance: MDD = 23, HC = 29.

### 5. Run the pipeline

**Full pipeline (extract + train + evaluate), from the repo root:**
```bash
python backend/main.py --stage all --model all
```

**Or step by step:**
```bash
# Step 1: Extract features
python backend/main.py --stage extract

# Step 2: Train models
python backend/main.py --stage train --model all

# Step 3: Evaluate
python backend/main.py --stage evaluate
```

---

## Troubleshooting

### Error: "No audio files found" / data structure verification failed
**Solution:** Check `backend/data/audio_lanzhou_2015/` matches the flat layout above.
```bash
python -c "from mendxai.ml.data_loader import verify_data_structure; verify_data_structure()"
```

### Error: "librosa installation failed"
**MacBook M-series specific:** Install soundfile separately:
```bash
uv pip install soundfile
uv pip install librosa
```

### Error: "torch not found" (for the neural net model)
**M-series Mac:** Install PyTorch with MPS support:
```bash
uv pip install torch torchvision torchaudio
```

### Slow feature extraction?
Normal — processing ~1,500 audio files takes roughly 10-30 minutes depending on your Mac.
A progress bar shows estimates.

---

## Expected Runtime (MacBook Pro M1/M2/M3)

| Stage | Time | Notes |
|-------|------|-------|
| Feature extraction | 15-30 min | ~1,500 audio files |
| Decision Tree training | <1 min | Fast |
| XGBoost training | 1-2 min | Medium speed |
| Neural Network training | 3-5 min | GPU accelerated on M-series |
| Evaluation | <1 min | All models |
| **Total (full pipeline)** | **~20-40 min** | First run |

Subsequent runs (training only): ~5-10 minutes

---

## Checking Results

After running, check:
```bash
# Extracted features
ls -lh backend/data/processed/features.csv

# Trained models
ls -lh backend/results/models/

# Evaluation results / plots
ls -lh backend/results/logs/
ls -lh backend/results/figures/
```

---

## Next Steps After First Run

1. **Analyze results:**
   - Check `backend/results/logs/` for accuracy and comparison output.
   - View confusion matrices and ROC curves in `backend/results/figures/`.

2. **Improve models:**
   ```bash
   # Enable hyperparameter tuning (slower but better)
   python backend/main.py --stage train --model all --tune
   ```

3. **Feature engineering:**
   - Modify `backend/src/mendxai/ml/feature_extractor.py`.
   - Add new audio features, then re-run extraction.

4. **Experiment with specific models:**
   ```bash
   # Train only XGBoost
   python backend/main.py --stage train --model xgboost

   # Train only the neural net
   python backend/main.py --stage train --model neural_net
   ```

---

## Development Tips

**Format code:**
```bash
black backend/src backend/main.py
ruff check backend/src backend/main.py
```

**Run notebooks:**
```bash
source backend/.venv/bin/activate
jupyter lab
# open backend/notebooks/01_dataset_audit.ipynb
```

---

## File Size Expectations

| File | Size | Notes |
|------|------|-------|
| Raw audio (1 file) | ~500KB | WAV 44.1kHz |
| Total raw audio | ~750MB | 1,508 files |
| `features.csv` | ~5-10MB | ~1,500 rows × features |
| Trained models | ~1-50MB | Varies by model |

Make sure you have at least **2GB free space** for the full pipeline.

---

## Success Indicators

You're ready to proceed when you see:
```
Data directories verified successfully!
Feature extraction complete!
Decision Tree training complete
XGBoost training complete
Neural Network training complete
All models trained successfully!
Model evaluation complete!
PIPELINE COMPLETED SUCCESSFULLY
```

Expected final accuracy: **70-85%** on this depression module, based on MODMA literature.

---

## Questions?

Check these first:
1. [`CLAUDE.md`](../CLAUDE.md) and [`notebooks/DATA_CONTEXT.md`](notebooks/DATA_CONTEXT.md) — environment and dataset facts.
2. `backend/src/mendxai/core/config.py` — all configurable parameters.
3. Error messages — usually self-explanatory with fix suggestions.
