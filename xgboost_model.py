"""XGBoost model for depression detection."""
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any
import joblib
from pathlib import Path

from ..config import config


class XGBoostModel:
    """XGBoost classifier wrapper."""
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=config.model.xgb_n_estimators,
            max_depth=config.model.xgb_max_depth,
            learning_rate=config.model.xgb_learning_rate,
            random_state=config.model.xgb_random_state,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost model."""
        print("\nTraining XGBoost...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ XGBoost training complete")
    
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
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
    
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
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        
        print("\nPerforming hyperparameter tuning for XGBoost...")
        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                random_state=config.model.xgb_random_state,
                eval_metric='logloss',
                use_label_encoder=False,
            ),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return grid_search.best_params_
