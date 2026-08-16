"""Gradient Boosting model for depression detection.

Stand-in for XGBoostModel when xgboost isn't importable (this project's dev
environment is missing libomp — see backend/dev/change_9_*.md). Uses sklearn's
HistGradientBoostingClassifier, the closest native sklearn equivalent to
XGBoost's histogram-based boosting. Same method surface as XGBoostModel/
DecisionTreeModel, so switching between them is a one-line change in
ModelTrainer's model dispatch.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any, Optional
import joblib
from pathlib import Path

from ...core.config import config


class GradientBoostingModel:
    """Gradient boosting classifier wrapper (sklearn HistGradientBoostingClassifier)."""

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=config.model.gb_max_iter,
            max_depth=config.model.gb_max_depth,
            learning_rate=config.model.gb_learning_rate,
            random_state=config.model.gb_random_state,
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Train the gradient boosting model. Pass `sample_weight` (e.g. from
        sklearn.utils.class_weight.compute_sample_weight) for imbalanced classes —
        HistGradientBoostingClassifier has no class_weight parameter."""
        print("\nTraining Gradient Boosting...")
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.is_trained = True
        print("✅ Gradient Boosting training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.

        HistGradientBoostingClassifier doesn't expose feature_importances_
        directly (unlike DecisionTreeClassifier/XGBoost); permutation importance
        is the standard substitute but requires held-out data, so callers should
        use sklearn.inspection.permutation_importance(model.model, X, y) directly
        rather than this method for this model class.
        """
        raise NotImplementedError(
            "HistGradientBoostingClassifier has no built-in feature_importances_. "
            "Use sklearn.inspection.permutation_importance(model.model, X_holdout, y_holdout) instead."
        )

    def save(self, filepath: Path):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")

    def load(self, filepath: Path):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from: {filepath}")

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.

        Returns:
            Best parameters found
        """
        param_grid = {
            'max_iter': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
        }

        print("\nPerforming hyperparameter tuning for Gradient Boosting...")
        grid_search = GridSearchCV(
            HistGradientBoostingClassifier(random_state=config.model.gb_random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        self.is_trained = True

        return grid_search.best_params_
