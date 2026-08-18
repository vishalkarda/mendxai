"""Model training orchestration."""
import contextlib
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

from ..core.config import config
from ..core.logging_utils import get_logger
from .models.decision_tree import DecisionTreeModel
from .models.gradient_boosting_model import GradientBoostingModel
from .models.neural_net import NeuralNetModel

try:
    from .models.xgboost_model import XGBoostModel
except Exception:
    # xgboost isn't usable in this environment — either the package isn't
    # installed (ImportError) or its native library fails to load (raises
    # xgboost's own XGBoostError, not ImportError; this repo's dev machine hits
    # exactly that: missing libomp — see backend/dev/change_9_*.md). Either way,
    # decision-tree/gradient-boosting/neural-net training must not be blocked;
    # only code paths that actually request 'xgboost' will raise, with a clear message.
    XGBoostModel = None

logger = get_logger("trainer")


class ModelTrainer:
    """Train and manage depression detection models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare features and labels for a single file-level train/val/test split.

        NOTE: this splits at the file level with no subject grouping — for the
        MODMA/Lanzhou dataset (~29 correlated recordings per subject), this risks
        leaking a subject's recordings across train/test. It's kept as-is for
        backward compatibility with the CLI (main.py), but subject-level modeling
        should use `cross_validate()` with a subject-level DataFrame instead (see
        mendxai.ml.aggregation.build_subject_level_features and
        backend/notebooks/05_baseline_model.ipynb).

        Args:
            df: DataFrame with features and 'label' column

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['label', 'file_path']]
        X = df[feature_cols].values
        y = df['label'].values

        print(f"\nPreparing data...")
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  MDD samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"  HC samples: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

        # Split into train and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config.training.test_size,
            random_state=config.training.random_state,
            stratify=y
        )

        # Split train into train and validation
        val_size_adjusted = config.training.val_size / (1 - config.training.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=config.training.random_state,
            stratify=y_temp
        )

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"\nData split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_decision_tree(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = False
    ):
        """Train decision tree model."""
        model = DecisionTreeModel()

        if tune_hyperparams:
            model.hyperparameter_tuning(X_train, y_train)
        else:
            model.train(X_train, y_train)

        self.models['decision_tree'] = model

        # Save model
        model_path = config.training.models_dir / "decision_tree.pkl"
        model.save(model_path)

        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = False
    ):
        """Train XGBoost model."""
        if XGBoostModel is None:
            raise ImportError(
                "xgboost is not available in this environment (likely missing libomp — "
                "run `brew install libomp` on macOS, then retry). Use train_gradient_boosting() "
                "as a stand-in in the meantime; see backend/dev/change_9_*.md."
            )
        model = XGBoostModel()

        if tune_hyperparams:
            model.hyperparameter_tuning(X_train, y_train)
        else:
            model.train(X_train, y_train)

        self.models['xgboost'] = model

        # Save model
        model_path = config.training.models_dir / "xgboost.pkl"
        model.save(model_path)

        return model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = False,
        balanced: bool = True,
    ):
        """Train gradient boosting model (sklearn HistGradientBoostingClassifier —
        stand-in for XGBoost; see models/gradient_boosting_model.py)."""
        model = GradientBoostingModel()

        if tune_hyperparams:
            model.hyperparameter_tuning(X_train, y_train)
        else:
            sample_weight = compute_sample_weight('balanced', y_train) if balanced else None
            model.train(X_train, y_train, sample_weight=sample_weight)

        self.models['gradient_boosting'] = model

        # Save model
        model_path = config.training.models_dir / "gradient_boosting.pkl"
        model.save(model_path)

        return model

    def train_neural_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train neural network model."""
        input_dim = X_train.shape[1]
        model = NeuralNetModel(input_dim=input_dim)

        model.train(X_train, y_train, X_val, y_val)

        self.models['neural_net'] = model

        # Save model
        model_path = config.training.models_dir / "neural_net.pth"
        model.save(model_path)

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        tune_hyperparams: bool = False
    ):
        """Train all models."""
        print("\n" + "="*50)
        print("TRAINING ALL MODELS")
        print("="*50)

        # Decision Tree
        self.train_decision_tree(X_train, y_train, tune_hyperparams)

        # XGBoost (falls back to Gradient Boosting if xgboost isn't available)
        if XGBoostModel is not None:
            self.train_xgboost(X_train, y_train, tune_hyperparams)
        else:
            print("\nxgboost not available in this environment — training Gradient Boosting instead.")
            self.train_gradient_boosting(X_train, y_train, tune_hyperparams)

        # Neural Network
        self.train_neural_net(X_train, y_train, X_val, y_val)

        print("\n✅ All models trained successfully!")

        return self.models

    def load_model(self, model_name: str):
        """Load a saved model."""
        if model_name == 'decision_tree':
            model = DecisionTreeModel()
            model_path = config.training.models_dir / "decision_tree.pkl"
            model.load(model_path)
        elif model_name == 'xgboost':
            if XGBoostModel is None:
                raise ImportError("xgboost is not available in this environment (missing libomp).")
            model = XGBoostModel()
            model_path = config.training.models_dir / "xgboost.pkl"
            model.load(model_path)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingModel()
            model_path = config.training.models_dir / "gradient_boosting.pkl"
            model.load(model_path)
        elif model_name == 'neural_net':
            # input_dim is overwritten by load() from the saved checkpoint, so
            # the placeholder value here doesn't matter.
            model = NeuralNetModel(input_dim=1)
            model_path = config.training.models_dir / "neural_net.pth"
            model.load(model_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.models[model_name] = model
        return model

    # ------------------------------------------------------------------
    # Subject-level cross-validation (for small-n datasets like this one)
    # ------------------------------------------------------------------

    def _build_model(self, model_name: str, input_dim: int, balanced: bool = True):
        """Instantiate a fresh, untrained model by name."""
        if model_name == 'decision_tree':
            return DecisionTreeModel(class_weight='balanced' if balanced else None)
        elif model_name == 'gradient_boosting':
            return GradientBoostingModel()
        elif model_name == 'xgboost':
            if XGBoostModel is None:
                raise ImportError(
                    "xgboost is not available in this environment (likely missing libomp). "
                    "Use model_name='gradient_boosting' as a stand-in."
                )
            return XGBoostModel()
        elif model_name == 'neural_net':
            return NeuralNetModel(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _fit_model(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        balanced: bool = True,
    ):
        """Fit a model built by `_build_model`.

        - neural_net: trained with (X_val, y_val) when provided, enabling the
          early-stopping logic already built into NeuralNetModel.train() (via
          config.model.nn_early_stopping_patience). Without a val set it just
          trains the fixed number of epochs, same as before.
        - gradient_boosting: sample_weight balancing, since
          HistGradientBoostingClassifier has no class_weight param.
        - decision_tree: class_weight is set at construction (_build_model), not here.
        """
        if model_name == 'neural_net':
            model.train(X_train, y_train, X_val, y_val)
        elif model_name == 'gradient_boosting' and balanced:
            sample_weight = compute_sample_weight('balanced', y_train)
            model.train(X_train, y_train, sample_weight=sample_weight)
        else:
            model.train(X_train, y_train)

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def _find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> float:
        """Scan candidate thresholds and return the one maximizing F-beta.

        beta=2 (F2) weights recall roughly twice as heavily as precision —
        appropriate for a screening context where missing a true case (false
        negative) is costlier than a false alarm. Must only ever be called with
        y_true/y_proba from data the model wasn't evaluated against for the
        final reported metrics (an inner validation split, not the CV test
        fold), otherwise the threshold is tuned against the same data used to
        judge it — a leakage/overfitting risk, not a hard error but a silent
        source of over-optimistic numbers.
        """
        thresholds = np.linspace(0.05, 0.95, 19)
        best_t, best_score = 0.5, -1.0
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
            if score > best_score:
                best_score, best_t = score, t
        return best_t

    def cross_validate(
        self,
        df: pd.DataFrame,
        model_name: str,
        n_splits: int = 5,
        n_repeats: int = 20,
        exclude_cols: Optional[list] = None,
        balanced: bool = True,
        tune_threshold: bool = True,
        log_run_history: bool = True,
    ) -> pd.DataFrame:
        """
        Repeated stratified k-fold cross-validation.

        Intended for subject-level data — one row per independent subject, e.g.
        the output of `mendxai.ml.aggregation.build_subject_level_features` — so
        that plain StratifiedKFold is leakage-safe without needing group-aware
        splitting (each row already is one independent subject). Do not pass
        file-level data here; a subject's correlated recordings would be split
        across folds.

        Uses repeated k-fold (default 5-fold x 20 repeats = 100 fold evaluations)
        rather than a single train/test split, since a dataset this small (~52
        subjects) makes any single split noisy. This mirrors the companion
        paper's own repeated-CV design (see
        backend/docs/data_research/05_companion_paper_methodology.md, which used
        4-fold x 50 repeats).

        Within each fold, the training portion is further split into an inner
        train/validation set (config.training.cv_inner_val_size). The inner
        validation set is used for two things, neither of which ever touches
        the fold's held-out test data: (1) neural-net early stopping, and (2)
        tuning the classification threshold to maximize F2 (recall-weighted)
        rather than defaulting to 0.5. Both the default-threshold and
        tuned-threshold metrics are reported per fold, so you can see the
        actual effect of threshold tuning rather than just trusting it.

        Args:
            df: subject-level DataFrame with a 'label' column (1=MDD, 0=HC).
            model_name: 'decision_tree' | 'gradient_boosting' | 'xgboost' | 'neural_net'
            n_splits, n_repeats: passed to RepeatedStratifiedKFold.
            exclude_cols: non-feature columns to drop before building X. Defaults
                to ['subject_id', 'label', 'type'].
            balanced: apply class-weight/sample-weight balancing where supported.
            tune_threshold: if False, skip threshold tuning (tuned columns will
                just mirror the default-threshold ones at 0.5).
            log_run_history: append a summary row to
                backend/results/logs/run_history.csv (see _append_run_history).

        Returns:
            DataFrame, one row per fold, with default-threshold metrics
            (accuracy/precision/recall/specificity/f1/f2/auc_roc) and
            tuned-threshold metrics (tuned_threshold/accuracy_tuned/
            precision_tuned/recall_tuned/specificity_tuned/f1_tuned/f2_tuned).
        """
        exclude_cols = exclude_cols or ['subject_id', 'label', 'type']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        y = df['label'].values

        n_folds_total = n_splits * n_repeats
        logger.info(
            f"cross_validate start: model={model_name} n_splits={n_splits} n_repeats={n_repeats} "
            f"({n_folds_total} fold evaluations) samples={X.shape[0]} features={X.shape[1]} "
            f"balanced={balanced} tune_threshold={tune_threshold}"
        )
        logger.info(f"  MDD: {sum(y)} ({sum(y)/len(y)*100:.1f}%)  HC: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=config.training.random_state
        )

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)

            # Inner train/val split from the training fold only — used for NN
            # early stopping and threshold tuning. The test fold is never
            # touched by either.
            X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                X_train_fold, y_train_fold,
                test_size=config.training.cv_inner_val_size,
                random_state=config.training.random_state,
                stratify=y_train_fold,
            )

            # Suppress the per-fold model chatter (e.g. NeuralNetModel's
            # constructor printing its architecture, "Training Decision
            # Tree...", or 50 epochs of NN logs) — with 100 folds this would
            # otherwise flood stdout/the log file; the logger call above and
            # the summary below are what matter for the historical record.
            with contextlib.redirect_stdout(io.StringIO()):
                model = self._build_model(model_name, input_dim=X.shape[1], balanced=balanced)
                self._fit_model(
                    model, model_name, X_inner_train, y_inner_train,
                    X_val=X_inner_val, y_val=y_inner_val, balanced=balanced,
                )

                if tune_threshold:
                    y_inner_val_proba = model.predict_proba(X_inner_val)
                    y_inner_val_proba_pos = y_inner_val_proba[:, 1] if y_inner_val_proba.ndim > 1 else y_inner_val_proba
                    tuned_threshold = self._find_best_threshold(y_inner_val, y_inner_val_proba_pos, beta=2.0)
                else:
                    tuned_threshold = 0.5

            y_test_proba = model.predict_proba(X_test_fold)
            y_test_proba_pos = y_test_proba[:, 1] if y_test_proba.ndim > 1 else y_test_proba

            y_pred_default = (y_test_proba_pos >= 0.5).astype(int)
            y_pred_tuned = (y_test_proba_pos >= tuned_threshold).astype(int)

            auc = roc_auc_score(y_test_fold, y_test_proba_pos) if len(set(y_test_fold)) > 1 else float('nan')

            fold_results.append({
                'fold': fold_idx,
                'accuracy': accuracy_score(y_test_fold, y_pred_default),
                'precision': precision_score(y_test_fold, y_pred_default, zero_division=0),
                'recall': recall_score(y_test_fold, y_pred_default, zero_division=0),
                'specificity': self._specificity(y_test_fold, y_pred_default),
                'f1': f1_score(y_test_fold, y_pred_default, zero_division=0),
                'f2': fbeta_score(y_test_fold, y_pred_default, beta=2.0, zero_division=0),
                'auc_roc': auc,
                'tuned_threshold': tuned_threshold,
                'accuracy_tuned': accuracy_score(y_test_fold, y_pred_tuned),
                'precision_tuned': precision_score(y_test_fold, y_pred_tuned, zero_division=0),
                'recall_tuned': recall_score(y_test_fold, y_pred_tuned, zero_division=0),
                'specificity_tuned': self._specificity(y_test_fold, y_pred_tuned),
                'f1_tuned': f1_score(y_test_fold, y_pred_tuned, zero_division=0),
                'f2_tuned': fbeta_score(y_test_fold, y_pred_tuned, beta=2.0, zero_division=0),
            })

        results_df = pd.DataFrame(fold_results)
        summary = results_df.drop(columns=['fold']).agg(['mean', 'std'])

        logger.info(f"cross_validate results: model={model_name} ({n_folds_total} folds)")
        for metric in summary.columns:
            logger.info(f"  {metric:18s}: {summary.loc['mean', metric]:.4f} +/- {summary.loc['std', metric]:.4f}")

        if log_run_history:
            self._append_run_history(model_name, n_splits, n_repeats, results_df)

        return results_df

    def _append_run_history(self, model_name: str, n_splits: int, n_repeats: int, results_df: pd.DataFrame):
        """Append one summary row for this cross_validate() call to
        backend/results/logs/run_history.csv (created with a header on first
        use). Lets you compare runs over time instead of only ever seeing the
        latest one — unlike subject_level_cv_comparison.csv, which is
        overwritten every run."""
        config.ensure_output_dirs()
        history_path = config.training.logs_dir / "run_history.csv"

        summary = results_df.drop(columns=['fold']).agg(['mean', 'std'])
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'model': model_name,
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'n_folds_total': n_splits * n_repeats,
        }
        for metric in summary.columns:
            row[f'{metric}_mean'] = summary.loc['mean', metric]
            row[f'{metric}_std'] = summary.loc['std', metric]

        row_df = pd.DataFrame([row])
        if history_path.exists():
            row_df.to_csv(history_path, mode='a', header=False, index=False)
        else:
            row_df.to_csv(history_path, mode='w', header=True, index=False)

        logger.info(f"Appended run to {history_path}")

    def fit_final_model(self, df: pd.DataFrame, model_name: str, exclude_cols: Optional[list] = None, balanced: bool = True):
        """Fit one final model on the full subject-level dataset (no held-out
        split) — for saving/deployment after cross_validate() has already
        established expected performance. Returns (model, fitted_scaler).

        Note on threshold: this fits on 100% of the data, so there's no
        held-out portion left to tune a decision threshold against here. Use
        the tuned_threshold column from a prior cross_validate() call (e.g.
        its mean across folds) as the recommended operating threshold for this
        model rather than assuming 0.5 — see backend/notebooks/05_baseline_model.ipynb.
        """
        exclude_cols = exclude_cols or ['subject_id', 'label', 'type']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        y = df['label'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = self._build_model(model_name, input_dim=X.shape[1], balanced=balanced)
        self._fit_model(model, model_name, X_scaled, y, balanced=balanced)

        self.models[model_name] = model
        return model, scaler
