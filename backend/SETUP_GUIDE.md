# SETUP GUIDE - MacBook M-Series

## Quick Start (5 minutes)

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart terminal, then verify:
```bash
uv --version
```

### 2. Create project directory
```bash
mkdir depression-voice-detection
cd depression-voice-detection
```

### 3. Copy all project files
Place all the generated files in this directory following the structure:
```
depression-voice-detection/
├── pyproject.toml
├── README.md
├── .gitignore
├── .python-version
├── main.py
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_clinical_metadata_eda.ipynb
│   ├── 03_audio_signal_eda.ipynb
│   └── 04_feature_baseline_eda.ipynb
├── data/
│   ├── raw/MDD/
│   ├── raw/NC/
│   ├── processed/
│   └── metadata/
├── src/mendxai/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   └── ml/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── feature_extractor.py
│       ├── trainer.py
│       ├── evaluator.py
│       └── models/
│           ├── __init__.py
│           ├── decision_tree.py
│           ├── xgboost_model.py
│           └── neural_net.py
└── results/
    ├── models/
    ├── logs/
    └── figures/
```

### 4. Setup Python environment
```bash
# Create virtual environment with Python 3.10+
uv venv

# Activate it
source .venv/bin/activate

# Install project
uv pip install -e ".[dev]"
```

This will install all dependencies automatically from pyproject.toml.

### 5. Organize MODMA data
Copy your MODMA audio files:
```bash
# Example structure (adjust paths to your actual data)
cp -r /path/to/MODMA/MDD/* data/raw/MDD/
cp -r /path/to/MODMA/NC/* data/raw/NC/
cp /path/to/MODMA/participant_info.xlsx data/metadata/
```

Expected structure:
```
data/raw/MDD/
  participant_001/
    1.wav
    2.wav
    ...
    29.wav
  participant_002/
    ...
  (23 participants total)

data/raw/NC/
  participant_024/
    1.wav
    ...
  (29 participants total)
```

### 6. Run the pipeline!

**Full pipeline (extract + train + evaluate):**
```bash
python main.py --stage all --model all
```

**Or step by step:**
```bash
# Step 1: Extract features
python main.py --stage extract

# Step 2: Train models
python main.py --stage train --model all

# Step 3: Evaluate
python main.py --stage evaluate
```

---

## Troubleshooting

### Error: "No audio files found"
**Solution:** Check data directory structure matches expected format.
```bash
python -c "from mendxai.ml.data_loader import verify_data_structure; verify_data_structure()"
```

### Error: "librosa installation failed"
**MacBook M-series specific:** Install soundfile separately:
```bash
uv pip install soundfile
uv pip install librosa
```

### Error: "torch not found" (for Neural Network)
**M-series Mac:** Install PyTorch with MPS support:
```bash
uv pip install torch torchvision torchaudio
```

### Slow feature extraction?
Normal! Processing 1,508 audio files takes ~10-30 minutes depending on your Mac.
Progress bar will show estimates.

---

## Expected Runtime (MacBook Pro M1/M2/M3)

| Stage | Time | Notes |
|-------|------|-------|
| Feature extraction | 15-30 min | 1,508 audio files |
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
ls -lh data/processed/features.csv

# Trained models
ls -lh results/models/

# Evaluation results
cat results/logs/model_comparison.csv

# Plots
open results/figures/
```

---

## Next Steps After First Run

1. **Analyze results:**
   - Check `results/logs/model_comparison.csv` for accuracy
   - View confusion matrices in `results/figures/`
   - Compare ROC curves

2. **Improve models:**
   ```bash
   # Enable hyperparameter tuning (slower but better)
   python main.py --stage train --model all --tune
   ```

3. **Feature engineering:**
   - Modify `src/mendxai/ml/feature_extractor.py`
   - Add new audio features
   - Re-run extraction

4. **Experiment with specific models:**
   ```bash
   # Train only XGBoost
   python main.py --stage train --model xgboost
   
   # Train only Neural Network
   python main.py --stage train --model neural_net
   ```

---

## Development Tips

**Format code:**
```bash
uv pip install -e ".[dev]"
black src/ main.py
ruff check src/ main.py
```

**Run in notebook:**
```bash
uv pip install ipykernel
jupyter notebook notebooks/
```

**Watch logs in real-time:**
```bash
tail -f results/logs/*.log
```

---

## File Size Expectations

| File | Size | Notes |
|------|------|-------|
| Raw audio (1 file) | ~500KB | WAV 44.1kHz |
| Total raw audio | ~750MB | 1,508 files |
| features.csv | ~5-10MB | 1,508 rows × 70 features |
| Trained models | ~1-50MB | Varies by model |

Make sure you have at least **2GB free space** for the full pipeline.

---

## Success Indicators

✅ You're ready to proceed when you see:
```
✅ All data directories verified successfully!
✅ Feature extraction complete!
✅ Decision Tree training complete
✅ XGBoost training complete  
✅ Neural Network training complete
✅ All models trained successfully!
✅ Model evaluation complete!
✅ PIPELINE COMPLETED SUCCESSFULLY
```

Expected final accuracy: **70-85%** (based on MODMA literature)

---

## Questions?

Check these first:
1. README.md - Full documentation
2. src/mendxai/core/config.py - All configurable parameters
3. Error messages - Usually self-explanatory with fix suggestions

Good luck with your depression detection model! 🚀
