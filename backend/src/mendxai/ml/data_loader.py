"""
Data loader for the MODMA/Lanzhou dataset.
Handles loading audio files and labels (MDD vs HC), joined against the
subject metadata workbook. See backend/notebooks/DATA_CONTEXT.md for the
full dataset context (file layout, task-type mapping, join recipe).
"""
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd

from ..core.config import config


# File-index -> task-type mapping (MODMA/Lanzhou 2015 session structure).
# See backend/notebooks/DATA_CONTEXT.md section 3.
_TASK_TYPE_RANGES = [
    (range(1, 19), "interview"),             # 01-18
    (range(19, 20), "passage_reading"),      # 19
    (range(20, 26), "word_reading"),         # 20-25
    (range(26, 30), "picture_description"),  # 26-29
]


def classify_task(file_index: int) -> str:
    """Map a MODMA audio file index (1-29) to its recording task type."""
    for index_range, task_type in _TASK_TYPE_RANGES:
        if file_index in index_range:
            return task_type
    raise ValueError(f"file_index {file_index} is outside the expected 1-29 range")


class DataLoader:
    """Load audio files and participant labels from the MODMA/Lanzhou dataset.

    Real layout: `dataset_dir/<8-digit-subject-id>/01.wav`..`29.wav` — flat,
    no MDD/HC split on disk. Labels come from joining subject ids (zero-padded
    to 8 digits) against the metadata workbook's `type` column.
    """

    def __init__(self):
        self.dataset_dir = config.data.dataset_dir
        self.metadata_path = config.data.metadata_path

    def load_metadata(self) -> pd.DataFrame:
        """
        Load participant metadata (demographics, PHQ-9 and other clinical scales).
        Drops the two junk `Unnamed` legend columns present in the source workbook.

        Returns:
            DataFrame with one row per subject. `subject id` is int (no leading
            zero) — use `str(id).zfill(8)` to match folder names.
        """
        if not self.metadata_path.exists():
            print(f"Warning: Metadata file {self.metadata_path} not found")
            return pd.DataFrame()

        df = pd.read_excel(self.metadata_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        print(f"\nLoaded metadata for {len(df)} subjects")
        print(f"Columns: {df.columns.tolist()}")

        return df

    def _subject_dirs(self) -> List[Path]:
        if not self.dataset_dir.exists():
            print(f"  Warning: Directory {self.dataset_dir} does not exist")
            return []
        return sorted(d for d in self.dataset_dir.iterdir() if d.is_dir())

    def load_audio_file_paths(self) -> Tuple[List[Path], List[int]]:
        """
        Load all audio file paths and their labels, joined against metadata.

        Returns:
            audio_paths: List of Path objects to audio files
            labels: List of labels (1 for MDD, 0 for HC)
        """
        metadata = self.load_metadata()
        if metadata.empty:
            print("  Cannot assign labels: metadata not available")
            return [], []

        label_by_folder_id = {
            str(int(row["subject id"])).zfill(8): row["type"]
            for _, row in metadata.iterrows()
        }

        subject_dirs = self._subject_dirs()
        audio_paths = []
        labels = []
        unmatched_folders = []

        for subject_dir in subject_dirs:
            subject_type = label_by_folder_id.get(subject_dir.name)
            if subject_type is None:
                unmatched_folders.append(subject_dir.name)
                continue

            label = 1 if subject_type == config.data.mdd_label else 0
            wav_files = sorted(subject_dir.glob("*.wav"))
            audio_paths.extend(wav_files)
            labels.extend([label] * len(wav_files))

        if unmatched_folders:
            print(f"  Warning: {len(unmatched_folders)} subject folder(s) had no metadata match: {unmatched_folders}")

        matched_subjects = len(subject_dirs) - len(unmatched_folders)
        print(f"\nTotal: {len(audio_paths)} audio files across {matched_subjects} subjects")
        print(f"  MDD: {sum(labels)} files")
        print(f"  HC:  {len(labels) - sum(labels)} files")

        return audio_paths, labels

    def get_dataset_summary(self) -> Dict[str, float]:
        """Get summary statistics of the dataset."""
        audio_paths, labels = self.load_audio_file_paths()

        if not labels:
            return {"total_files": 0, "mdd_files": 0, "hc_files": 0}

        return {
            "total_files": len(audio_paths),
            "mdd_files": sum(labels),
            "hc_files": len(labels) - sum(labels),
            "mdd_percentage": sum(labels) / len(labels) * 100,
            "hc_percentage": (len(labels) - sum(labels)) / len(labels) * 100,
        }


def verify_data_structure() -> bool:
    """
    Verify that the dataset directory and metadata workbook are present.
    Print helpful instructions if not.
    """
    dataset_dir = config.data.dataset_dir
    metadata_path = config.data.metadata_path

    print("\n=== Data Structure Verification ===\n")

    issues = []

    if not dataset_dir.exists():
        issues.append(f"Dataset directory does not exist: {dataset_dir}")
    else:
        subject_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        wav_count = len(list(dataset_dir.glob("*/*.wav")))
        print(f"Dataset directory exists: {dataset_dir}")
        print(f"  {len(subject_dirs)} subject folders, {wav_count} .wav files")

    if not metadata_path.exists():
        issues.append(f"Metadata workbook does not exist: {metadata_path}")
    else:
        print(f"Metadata workbook exists: {metadata_path}")

    if issues:
        print("\n" + "=" * 50)
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nExpected layout:")
        print(f"""
{dataset_dir}/
├── <8-digit-subject-id>/
│   ├── 01.wav
│   ├── 02.wav
│   └── ... (29 files total)
├── ...
└── {config.data.metadata_workbook}
        """)
        print("=" * 50)
    else:
        print("\nAll data directories verified successfully!")

    return len(issues) == 0
