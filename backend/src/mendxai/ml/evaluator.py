"""Model evaluation and metrics."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.config import config


class ModelEvaluator:
    """Evaluate trained models."""
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> dict:
        """
        Evaluate a single model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*50}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'='*50}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'specificity': self._calculate_specificity(y_test, y_pred),
        }
        
        # AUC-ROC (need probability for positive class)
        if y_pred_proba.ndim > 1:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f} (Sensitivity)")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=['NC (Healthy)', 'MDD (Depression)']
        ))
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    
    def compare_models(self, results: list) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: List of metric dictionaries from evaluate_model()
            
        Returns:
            DataFrame with comparison
        """
        df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        
        # Save comparison
        output_path = config.training.logs_dir / "model_comparison.csv"
        df.to_csv(output_path, index=False)
        print(f"\nComparison saved to: {output_path}")
        
        return df
    
    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NC', 'MDD'],
            yticklabels=['NC', 'MDD']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = config.training.figures_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {output_path}")
    
    def plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        output_path = config.training.figures_dir / f"roc_curve_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved: {output_path}")
    
    def plot_feature_importance(
        self,
        model,
        feature_names: list,
        model_name: str,
        top_n: int = 20
    ):
        """Plot feature importance for tree-based models."""
        if not hasattr(model, 'get_feature_importance'):
            print(f"Model {model_name} does not support feature importance")
            return
        
        importances = model.get_feature_importance()
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        
        output_path = config.training.figures_dir / f"feature_importance_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved: {output_path}")
