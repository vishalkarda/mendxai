"""Models package."""
from .decision_tree import DecisionTreeModel
from .gradient_boosting_model import GradientBoostingModel
from .neural_net import NeuralNetModel

__all__ = [
    "DecisionTreeModel",
    "GradientBoostingModel",
    "NeuralNetModel",
    "XGBoostModel",
]


def __getattr__(name):
    """Lazily import XGBoostModel so this package is usable without xgboost/
    libomp installed (see backend/dev/change_9_*.md)."""
    if name == "XGBoostModel":
        from .xgboost_model import XGBoostModel
        return XGBoostModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
