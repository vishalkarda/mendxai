"""
Data loader for MODMA dataset.
Handles loading audio files and labels (MDD vs NC).
"""
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import config


class DataLoader:
    """Load audio files and participant labels from MODMA dataset."""
    
    def __init__(self):
        self.raw_dir = config.data.raw_data_dir
        self.mdd_dir = self.raw_dir / config.data.mdd_label
        self.nc_dir = self.raw_dir / config.data.nc_label
        
    def load_audio_file_paths(self) -> Tuple[List[Path], List[int]]:
        """
        Load all audio file paths and their labels.
        
        Returns:
            audio_paths: List of Path objects to audio files
            labels: List of labels (1 for MDD, 0 for NC)
        """
        audio_paths = []
        labels = []
        
        # Load MDD patients (label = 1)
        print(f"Loading MDD audio files from {self.mdd_dir}...")
        mdd_paths = self._get_audio_paths_from_dir(self.mdd_dir)
        audio_paths.extend(mdd_paths)
        labels.extend([1] * len(mdd_paths))
        print(f"  Found {len(mdd_paths)} MDD audio files")
        
        # Load NC controls (label = 0)
        print(f"Loading NC audio files from {self.nc_dir}...")
        nc_paths = self._get_audio_paths_from_dir(self.nc_dir)
        audio_paths.extend(nc_paths)
        labels.extend([0] * len(nc_paths))
        print(f"  Found {len(nc_paths)} NC audio files")
        
        print(f"\nTotal: {len(audio_paths)} audio files")
        print(f"  MDD: {sum(labels)} files")
        print(f"  NC: {len(labels) - sum(labels)} files")
        
        return audio_paths, labels
    
    def _get_audio_paths_from_dir(self, directory: Path) -> List[Path]:
        """
        Get all .wav file paths from a directory.
        Expects structure: directory/participant_xxx/1.wav, 2.wav, ..., 29.wav
        """
        audio_paths = []
        
        if not directory.exists():
            print(f"  Warning: Directory {directory} does not exist")
            return audio_paths
        
        # Find all participant directories
        participant_dirs = [d for d in directory.iterdir() if d.is_dir()]
        
        for participant_dir in sorted(participant_dirs):
            # Get all .wav files in this participant's directory
            wav_files = sorted(participant_dir.glob("*.wav"))
            audio_paths.extend(wav_files)
        
        return audio_paths
    
    def load_metadata(self, metadata_file: str = "participant_info.xlsx") -> pd.DataFrame:
        """
        Load participant metadata (demographics, PHQ-9, HAMD scores).
        
        Args:
            metadata_file: Name of Excel file containing metadata
            
        Returns:
            DataFrame with participant info
        """
        metadata_path = config.data.metadata_dir / metadata_file
        
        if not metadata_path.exists():
            print(f"Warning: Metadata file {metadata_path} not found")
            return pd.DataFrame()
        
        df = pd.read_excel(metadata_path)
        print(f"\nLoaded metadata for {len(df)} participants")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def get_dataset_summary(self) -> Dict[str, any]:
        """Get summary statistics of the dataset."""
        audio_paths, labels = self.load_audio_file_paths()
        
        summary = {
            "total_files": len(audio_paths),
            "mdd_files": sum(labels),
            "nc_files": len(labels) - sum(labels),
            "mdd_percentage": sum(labels) / len(labels) * 100,
            "nc_percentage": (len(labels) - sum(labels)) / len(labels) * 100,
        }
        
        return summary


def verify_data_structure():
    """
    Verify that the data directory structure is correct.
    Print helpful instructions if not.
    """
    raw_dir = config.data.raw_data_dir
    mdd_dir = raw_dir / "MDD"
    nc_dir = raw_dir / "NC"
    
    print("\n=== Data Structure Verification ===\n")
    
    issues = []
    
    # Check if directories exist
    if not raw_dir.exists():
        issues.append(f"❌ Raw data directory does not exist: {raw_dir}")
    else:
        print(f"✅ Raw data directory exists: {raw_dir}")
    
    if not mdd_dir.exists():
        issues.append(f"❌ MDD directory does not exist: {mdd_dir}")
    else:
        mdd_count = len(list(mdd_dir.glob("*/*.wav")))
        print(f"✅ MDD directory exists with {mdd_count} audio files")
    
    if not nc_dir.exists():
        issues.append(f"❌ NC directory does not exist: {nc_dir}")
    else:
        nc_count = len(list(nc_dir.glob("*/*.wav")))
        print(f"✅ NC directory exists with {nc_count} audio files")
    
    if issues:
        print("\n" + "=" * 50)
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease organize your MODMA data as follows:")
        print(f"""
{raw_dir}/
├── MDD/
│   ├── participant_001/
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   └── ... (29 files total)
│   ├── participant_002/
│   └── ... (23 participants total)
└── NC/
    ├── participant_024/
    │   ├── 1.wav
    │   └── ...
    └── ... (29 participants total)
        """)
        print("=" * 50)
    else:
        print("\n✅ All data directories verified successfully!")
    
    return len(issues) == 0
