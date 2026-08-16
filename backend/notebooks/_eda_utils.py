"""
Shared helpers for the backend/notebooks EDA notebooks.

Every notebook's first cell should do:

    import sys
    from pathlib import Path
    _nb_dir = Path("backend/notebooks") if Path("backend/notebooks").exists() else Path.cwd()
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))
    from _eda_utils import bootstrap_project_paths
    PROJECT_ROOT = bootstrap_project_paths()
    from mendxai.core.config import config
    from mendxai.ml.data_loader import DataLoader, classify_task

Importing this module also makes `import mendxai` work regardless of the
notebook's CWD (it resolves backend/src relative to this file's own location
on disk, not the notebook's CWD), which is what replaces the three different,
inconsistent path-resolution blocks the notebooks used to duplicate.
"""
import sys
import time
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent   # backend/notebooks
PROJECT_ROOT = _THIS_DIR.parent.parent         # repository root
_SRC_DIR = PROJECT_ROOT / "backend" / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats

from mendxai.core.config import config
from mendxai.ml.data_loader import DataLoader, classify_task  # re-exported for convenience


def bootstrap_project_paths() -> Path:
    """Return the repository root. Importing this module already configures
    sys.path for `import mendxai`; call this in each notebook's setup cell for
    an explicit PROJECT_ROOT reference and as a readable "setup done" marker."""
    return PROJECT_ROOT


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to backend/data/processed/<filename>, creating the
    directory if needed."""
    config.ensure_output_dirs()
    out_path = config.data.processed_data_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return out_path


def load_processed(filename: str) -> pd.DataFrame:
    """Load a previously saved DataFrame from backend/data/processed/<filename>.

    Forces `subject_id` to load as a zero-padded string: pandas' default type
    inference otherwise silently strips the leading zero (e.g. "02010001" ->
    2010001 as int64), which would break any join back against folder names.
    """
    in_path = config.data.processed_data_dir / filename
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found — run the notebook that produces it first "
            f"(see backend/notebooks/DATA_CONTEXT.md section 10)."
        )
    df = pd.read_csv(in_path, dtype={"subject_id": str})
    print(f"Loaded {len(df)} rows from {in_path}")
    return df


def build_file_manifest(loader: Optional[DataLoader] = None) -> pd.DataFrame:
    """Build the full per-file manifest: subject_id, label, type, file_index,
    task_type, path. This is the canonical file-level view of the dataset,
    used by notebooks 03 and 04."""
    loader = loader or DataLoader()
    audio_paths, labels = loader.load_audio_file_paths()

    rows = []
    for path, label in zip(audio_paths, labels):
        file_index = int(path.stem)
        rows.append({
            "subject_id": path.parent.name,
            "label": label,
            "type": config.data.mdd_label if label == 1 else config.data.hc_label,
            "file_index": file_index,
            "task_type": classify_task(file_index),
            "path": str(path),
        })
    return pd.DataFrame(rows)


def header_probe(path) -> dict:
    """Fast, header-only probe of an audio file (no full decode): sample rate,
    channel count, duration, and subtype (bit-depth encoding)."""
    info = sf.info(str(path))
    return {
        "sr": info.samplerate,
        "channels": info.channels,
        "duration": info.frames / info.samplerate if info.samplerate else 0.0,
        "subtype": info.subtype,
    }


def plot_by_label(df: pd.DataFrame, col: str, label_col: str = "type", ax=None, bins: int = 20):
    """Overlaid histogram of `col`, split by `label_col` (MDD vs HC)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for label_value, group in df.groupby(label_col):
        ax.hist(group[col].dropna(), bins=bins, alpha=0.5, label=str(label_value))
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.legend(title=label_col)
    ax.set_title(f"{col} by {label_col}")
    return ax


def mannwhitney_effect_size(
    df: pd.DataFrame,
    col: str,
    label_col: str = "type",
    group_a: str = "MDD",
    group_b: str = "HC",
) -> dict:
    """Mann-Whitney U test + Cohen's d between two groups for one numeric column."""
    a = df.loc[df[label_col] == group_a, col].dropna()
    b = df.loc[df[label_col] == group_b, col].dropna()

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")

    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan

    return {
        "feature": col,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "u_stat": u_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
    }


class elapsed_timer:
    """Context manager that prints elapsed wall-clock time on exit.

    with elapsed_timer("feature extraction"):
        ...
    """

    def __init__(self, label: str = "block"):
        self.label = label

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        print(f"[{self.label}] elapsed: {elapsed:.1f}s")
