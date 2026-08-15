"""Model training orchestration."""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path

from ..core.config import config
from .models.decision_tree import DecisionTreeModel
from .models.xgboost_model import XGBoostModel
from .models.neural_net import NeuralNetModel


class ModelTrainer:
    """Train and manage depression detection models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare features and labels for training.
        
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
        print(f"  NC samples: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
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
        
        # XGBoost
        self.train_xgboost(X_train, y_train, tune_hyperparams)
        
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
            model = XGBoostModel()
            model_path = config.training.models_dir / "xgboost.pkl"
            model.load(model_path)
        elif model_name == 'neural_net':
            # Need to know input_dim to load neural net
            # This is a limitation - you'd need to save metadata
            raise NotImplementedError("Neural net loading requires input dimension")
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.models[model_name] = model
        return model
