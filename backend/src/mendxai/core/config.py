"""
Configuration for depression detection pipeline.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import yaml

# backend/src/mendxai/core/config.py -> core -> mendxai -> src -> backend -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_YAML_PATH = PROJECT_ROOT / "backend" / "config" / "config.yaml"


def _load_yaml_overrides() -> dict:
    """Read backend/config/config.yaml if present. Missing file/keys fall back
    to the dataclass defaults below, so this stays optional."""
    if _CONFIG_YAML_PATH.exists():
        with open(_CONFIG_YAML_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class DataConfig:
    """Data paths and parameters.

    Paths are resolved relative to the repository root (via backend/config/config.yaml
    when present), not the process CWD, so behavior doesn't depend on where a
    notebook or script happens to be run from.

    Real on-disk layout (MODMA/Lanzhou 2015): raw audio is flat, one folder per
    zero-padded 8-digit subject id, with 01.wav..29.wav inside — there is no
    MDD/HC split at the filesystem level. Class label is only obtainable by
    joining subject ids against the metadata workbook's `type` column
    (values: MDD / HC). See backend/notebooks/DATA_CONTEXT.md for the full
    dataset context and the join recipe.
    """

    # Paths (resolved in __post_init__)
    raw_data_dir: Optional[Path] = None
    dataset_subdir: str = "audio_lanzhou_2015"
    metadata_workbook: str = "subjects_information_audio_lanzhou_2015.xlsx"
    processed_data_dir: Optional[Path] = None

    # Class labels (MODMA convention: MDD vs HC — not NC)
    mdd_label: str = "MDD"
    hc_label: str = "HC"  # Healthy Control

    # Audio parameters
    sample_rate: int = 44100  # MODMA dataset is 44.1 kHz
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512

    # File naming (MODMA convention: 01-29, zero-padded, per subject)
    audio_files_per_participant: int = 29

    def __post_init__(self):
        overrides = _load_yaml_overrides()

        data_dir = overrides.get("data_dir", "backend/data")
        self.dataset_subdir = overrides.get("dataset_subdir", self.dataset_subdir)
        self.metadata_workbook = overrides.get("metadata_workbook", self.metadata_workbook)

        if self.raw_data_dir is None:
            self.raw_data_dir = PROJECT_ROOT / data_dir
        if self.processed_data_dir is None:
            self.processed_data_dir = self.raw_data_dir / "processed"

    @property
    def dataset_dir(self) -> Path:
        """Directory containing per-subject audio folders and the metadata workbook."""
        return self.raw_data_dir / self.dataset_subdir

    @property
    def metadata_path(self) -> Path:
        """Full path to the subject metadata workbook."""
        return self.dataset_dir / self.metadata_workbook


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

    # Output paths (resolved in __post_init__, relative to repo root)
    models_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None

    def __post_init__(self):
        results_dir = PROJECT_ROOT / "backend" / "results"
        if self.models_dir is None:
            self.models_dir = results_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = results_dir / "logs"
        if self.figures_dir is None:
            self.figures_dir = results_dir / "figures"


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

    def ensure_output_dirs(self):
        """Create output (not raw data) directories. Called explicitly by code
        that's about to write to them — never as an import side effect, since
        raw_data_dir is externally provided and must never be silently created."""
        for directory in (
            self.data.processed_data_dir,
            self.training.models_dir,
            self.training.logs_dir,
            self.training.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
