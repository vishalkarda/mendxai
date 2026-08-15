# Project Decisions

Last updated: 2026-08-15

## Decision Log

### 1. Primary Phase 1 Problem

Decision:
- Focus on depression detection first.

Why:
- It keeps the first version constrained and measurable.
- It avoids mixing disorders before the baseline pipeline is trustworthy.
- It creates a clean foundation for later schizophrenia work.

### 2. Prediction Unit

Decision:
- Make the core prediction patient-level rather than recording-level.

Why:
- The same participant has multiple recordings.
- Patient-level output is closer to the intended screening use case.
- It reduces the risk of misleading evaluation caused by repeated within-subject recordings.

### 3. Modeling Objectives

Decision:
- Build two parallel tracks:
  - binary classification for depression vs healthy control
  - severity estimation in a separate workflow

Why:
- The binary task provides a simpler baseline.
- Severity estimation is valuable but depends more heavily on metadata quality and label design.
- Keeping them separate avoids conflating two distinct modeling problems.

### 4. Baseline Feature Strategy

Decision:
- Start with handcrafted acoustic and prosodic features.

Why:
- The existing repository is already organized around this approach.
- It is the fastest path to a trustworthy first baseline.
- It gives interpretable signals before moving to representation learning.

### 5. Longer-Term Representation Strategy

Decision:
- Add pretrained speech embeddings after the baseline is stable.
- Longer term, evaluate combining handcrafted features with embeddings.

Why:
- Pretrained models can capture richer hidden vocal patterns.
- Combining explicit features and learned representations may yield stronger results than either alone.

### 6. Baseline Model Priority

Decision:
- Use XGBoost as the primary baseline candidate.
- Keep the MLP as an experimental comparison.
- Keep the Decision Tree as a simple interpretability baseline.

Why:
- XGBoost usually performs well on small-to-medium tabular feature datasets.
- The MLP may be useful later but is more prone to overfitting at this scale.
- The Decision Tree is helpful as a sanity-check model even if it is unlikely to be the final choice.

### 7. Workflow Order

Decision:
- EDA comes before restructuring and before baseline retraining.

Why:
- The dataset and metadata need to be understood before modeling assumptions are locked in.
- It gives the team a shared understanding of data quality, labels, and leakage risks.
