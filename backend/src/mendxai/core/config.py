"""
Configuration for depression detection pipeline.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Data paths and parameters."""
    
    # Paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    metadata_dir: Path = Path("data/metadata")
    
    # Class labels
    mdd_label: str = "MDD"  # Major Depressive Disorder
    nc_label: str = "NC"    # Normal Control
    
    # Audio parameters
    sample_rate: int = 44100  # MODMA dataset is 44.1 kHz
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    
    # File naming (MODMA convention: 1-29)
    audio_files_per_participant: int = 29


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    
    # Decision Tree
    dt_max_depth: int = 10
    dt_min_samples_split: int = 5
    dt_random_state: int = 42
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_random_state: int = 42
    
    # Neural Network
    nn_hidden_dims: List[int] = None
    nn_dropout: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 16
    nn_epochs: int = 50
    nn_early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.nn_hidden_dims is None:
            self.nn_hidden_dims = [128, 64, 32]


@dataclass
class TrainingConfig:
    """Training parameters."""
    
    test_size: float = 0.2
    val_size: float = 0.1  # From training set
    random_state: int = 42
    n_folds: int = 5  # For cross-validation
    
    # Output paths
    models_dir: Path = Path("results/models")
    logs_dir: Path = Path("results/logs")
    figures_dir: Path = Path("results/figures")


@dataclass
class Config:
    """Main configuration object."""
    
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.raw_data_dir / "MDD",
            self.data.raw_data_dir / "NC",
            self.data.processed_data_dir,
            self.data.metadata_dir,
            self.training.models_dir,
            self.training.logs_dir,
            self.training.figures_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
