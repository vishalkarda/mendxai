"""Machine-learning pipeline package for MendXAI."""

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor

__all__ = [
    "DataLoader",
    "FeatureExtractor",
    "ModelTrainer",
    "ModelEvaluator",
]


def __getattr__(name):
    """Lazily import model-training classes (see mendxai/__init__.py for why)."""
    if name == "ModelTrainer":
        from .trainer import ModelTrainer
        return ModelTrainer
    if name == "ModelEvaluator":
        from .evaluator import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
