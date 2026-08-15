"""
Feature extraction from audio files.
Extracts acoustic and prosodic features for depression detection.
"""
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from .config import config


class FeatureExtractor:
    """Extract voice features from audio files."""
    
    def __init__(self):
        self.sr = config.data.sample_rate
        self.n_mfcc = config.data.n_mfcc
        self.n_fft = config.data.n_fft
        self.hop_length = config.data.hop_length
    
    def extract_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract comprehensive feature set from single audio file.
        
        Features extracted:
        - MFCC (13 coefficients × 4 stats = 52 features)
        - Pitch/F0 (5 features)
        - Energy (2 features)
        - Zero Crossing Rate (2 features)
        - Spectral features (5 features)
        - Prosodic features (4 features)
        
        Total: ~70 features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            features = {}
            
            # 1. MFCC features (52 features)
            mfcc_features = self._extract_mfcc(y)
            features.update(mfcc_features)
            
            # 2. Pitch/F0 features (5 features)
            pitch_features = self._extract_pitch(y)
            features.update(pitch_features)
            
            # 3. Energy features (2 features)
            energy_features = self._extract_energy(y)
            features.update(energy_features)
            
            # 4. Zero crossing rate (2 features)
            zcr_features = self._extract_zcr(y)
            features.update(zcr_features)
            
            # 5. Spectral features (5 features)
            spectral_features = self._extract_spectral(y)
            features.update(spectral_features)
            
            # 6. Prosodic features (4 features)
            prosodic_features = self._extract_prosodic(y, sr)
            features.update(prosodic_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {}
    
    def _extract_mfcc(self, y: np.ndarray) -> Dict[str, float]:
        """Extract MFCC features (mean, std, min, max for each coefficient)."""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i}_std'] = np.std(mfcc[i])
            features[f'mfcc{i}_min'] = np.min(mfcc[i])
            features[f'mfcc{i}_max'] = np.max(mfcc[i])
        
        return features
    
    def _extract_pitch(self, y: np.ndarray) -> Dict[str, float]:
        """Extract pitch/fundamental frequency features."""
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        
        # Get pitch values (ignore zeros)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features = {
                'pitch_mean': np.mean(pitch_values),
                'pitch_std': np.std(pitch_values),
                'pitch_min': np.min(pitch_values),
                'pitch_max': np.max(pitch_values),
                'pitch_range': np.max(pitch_values) - np.min(pitch_values),
            }
        else:
            features = {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_min': 0,
                'pitch_max': 0,
                'pitch_range': 0,
            }
        
        return features
    
    def _extract_energy(self, y: np.ndarray) -> Dict[str, float]:
        """Extract energy/RMS features."""
        rms = librosa.feature.rms(y=y)
        
        return {
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
        }
    
    def _extract_zcr(self, y: np.ndarray) -> Dict[str, float]:
        """Extract zero crossing rate features."""
        zcr = librosa.feature.zero_crossing_rate(y)
        
        return {
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
        }
    
    def _extract_spectral(self, y: np.ndarray) -> Dict[str, float]:
        """Extract spectral features."""
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)[0]
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast),
        }
    
    def _extract_prosodic(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract prosodic features (duration, pauses)."""
        duration = len(y) / sr
        
        # Detect pauses (silence)
        # Using RMS energy threshold
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.percentile(rms, 20)  # 20th percentile as silence threshold
        is_silent = rms < threshold
        
        # Count pauses
        pause_count = 0
        in_pause = False
        for silent in is_silent:
            if silent and not in_pause:
                pause_count += 1
                in_pause = True
            elif not silent:
                in_pause = False
        
        pause_ratio = np.sum(is_silent) / len(is_silent) if len(is_silent) > 0 else 0
        
        return {
            'duration': duration,
            'pause_count': pause_count,
            'pause_ratio': pause_ratio,
            'speech_rate': (duration - pause_ratio * duration) / duration if duration > 0 else 0,
        }
    
    def extract_features_batch(
        self, 
        audio_paths: List[Path], 
        labels: List[int]
    ) -> pd.DataFrame:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels (1=MDD, 0=NC)
            
        Returns:
            DataFrame with features and labels
        """
        print(f"\nExtracting features from {len(audio_paths)} audio files...")
        
        feature_list = []
        valid_labels = []
        
        for audio_path, label in tqdm(zip(audio_paths, labels), total=len(audio_paths)):
            features = self.extract_features(audio_path)
            
            if features:  # Only add if extraction succeeded
                features['label'] = label
                features['file_path'] = str(audio_path)
                feature_list.append(features)
                valid_labels.append(label)
        
        df = pd.DataFrame(feature_list)
        
        print(f"\nFeature extraction complete!")
        print(f"  Extracted features for {len(df)} files")
        print(f"  Feature dimension: {len(df.columns) - 2}")  # -2 for label and file_path
        print(f"  MDD samples: {sum(valid_labels)}")
        print(f"  NC samples: {len(valid_labels) - sum(valid_labels)}")
        
        return df
    
    def save_features(self, df: pd.DataFrame, output_file: str = "features.csv"):
        """Save extracted features to CSV."""
        output_path = config.data.processed_data_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"\nFeatures saved to: {output_path}")
    
    def load_features(self, input_file: str = "features.csv") -> pd.DataFrame:
        """Load previously extracted features from CSV."""
        input_path = config.data.processed_data_dir / input_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"Features file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        print(f"Loaded features from: {input_path}")
        print(f"  Shape: {df.shape}")
        
        return df
