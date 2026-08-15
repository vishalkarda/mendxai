"""Decision Tree model for depression detection."""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any
import joblib
from pathlib import Path

from ..config import config


class DecisionTreeModel:
    """Decision Tree classifier wrapper."""
    
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=config.model.dt_max_depth,
            min_samples_split=config.model.dt_min_samples_split,
            random_state=config.model.dt_random_state,
        )
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the decision tree model."""
        print("\nTraining Decision Tree...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Decision Tree training complete")
    
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
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
        }
        
        print("\nPerforming hyperparameter tuning for Decision Tree...")
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=config.model.dt_random_state),
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
