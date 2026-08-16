"""MendXAI package."""
__version__ = "0.1.0"

from .core.config import config
from .ml.data_loader import DataLoader
from .ml.feature_extractor import FeatureExtractor

__all__ = [
    "config",
    "DataLoader",
    "FeatureExtractor",
    "ModelTrainer",
    "ModelEvaluator",
]


def __getattr__(name):
    """Lazily import model-training classes so basic data/config access (e.g.
    from the EDA notebooks) doesn't require torch/xgboost to be importable —
    those are only needed once you actually train or evaluate a model."""
    if name == "ModelTrainer":
        from .ml.trainer import ModelTrainer
        return ModelTrainer
    if name == "ModelEvaluator":
        from .ml.evaluator import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
