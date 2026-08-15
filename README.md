# Depression Voice Detection

Voice-based depression detection using machine learning on the MODMA dataset.

## Project Overview

This project implements a binary classification system to detect Major Depressive Disorder (MDD) from voice recordings using acoustic and prosodic features.

**Dataset:** MODMA (Multi-Modal Open Dataset for Mental-disorder Analysis)
- 23 MDD patients + 29 healthy controls
- 1,508 audio recordings (29 per participant)
- WAV format, 44.1 kHz, 24-bit

**Models Implemented:**
- Decision Tree
- XGBoost
- Neural Network (PyTorch)

## Setup Instructions

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup project
```bash
cd depression-voice-detection
uv venv
source .venv/bin/activate  # On macOS/Linux
uv pip install -e .
```

### 3. Prepare data
Copy your MODMA dataset to:
```
data/raw/MDD/          # Depression patients audio files
data/raw/NC/           # Healthy controls audio files
data/metadata/         # participant_info.xlsx (demographics + PHQ-9/HAMD scores)
```

**Expected structure:**
```
data/raw/MDD/participant_001/
    ├── 1.wav  (positive interview)
    ├── 2.wav
    ...
    └── 29.wav (TAT picture)
data/raw/NC/participant_030/
    ├── 1.wav
    ...
```

### 4. Run the pipeline

**Extract features from audio:**
```bash
python main.py --stage extract
```

**Train all models:**
```bash
python main.py --stage train --model all
```

**Train specific model:**
```bash
python main.py --stage train --model decision_tree
python main.py --stage train --model xgboost
python main.py --stage train --model neural_net
```

**Evaluate models:**
```bash
python main.py --stage evaluate
```

**Full pipeline (extract + train + evaluate):**
```bash
python main.py --stage all --model all
```

## Project Structure

```
depression-voice-detection/
├── src/depression_detector/    # Main package
│   ├── config.py               # Configuration
│   ├── data_loader.py          # Load audio + labels
│   ├── feature_extractor.py    # Extract MFCC, pitch, etc.
│   ├── trainer.py              # Training logic
│   ├── evaluator.py            # Evaluation metrics
│   └── models/                 # Model implementations
│       ├── decision_tree.py
│       ├── xgboost_model.py
│       └── neural_net.py
├── data/                       # Data directory
├── results/                    # Outputs
│   ├── models/                 # Saved models
│   ├── logs/                   # Training logs
│   └── figures/                # Visualizations
└── main.py                     # CLI entry point
```

## Features Extracted

**Acoustic Features:**
- **MFCC:** 13 Mel-frequency cepstral coefficients (mean, std, min, max)
- **Pitch (F0):** Fundamental frequency (mean, std, min, max, range)
- **Energy:** RMS energy (mean, std)
- **Zero Crossing Rate:** Mean, std
- **Spectral:** Centroid, bandwidth, rolloff, contrast

**Prosodic Features:**
- **Speech Rate:** Words per minute (estimated from audio duration)
- **Pause Statistics:** Number, mean/max duration, ratio
- **Jitter:** Pitch variability
- **Shimmer:** Amplitude variability

**Total:** ~70 features per audio file

## Expected Performance

Based on literature and MODMA dataset:
- **Decision Tree:** 70-75% accuracy
- **XGBoost:** 80-85% accuracy
- **Neural Network:** 75-85% accuracy

## Development

**Run tests:**
```bash
uv pip install -e ".[dev]"
pytest tests/
```

**Format code:**
```bash
black src/ main.py
ruff check src/ main.py
```

## Clinical Scores Available

- **PHQ-9:** Patient Health Questionnaire (0-27, depression severity)
- **HAMD:** Hamilton Depression Rating Scale (0-52, clinical assessment)

Future work: Regression models to predict severity scores.

## References

- Cai, H., et al. (2020). MODMA dataset: a Multi-modal Open Dataset for Mental-disorder Analysis. arXiv:2002.09283
- MODMA Dataset: http://modma.lzu.edu.cn/data/index/

## License

Research use only. Follow MODMA EULA terms.
