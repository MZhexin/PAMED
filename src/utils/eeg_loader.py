# src/utils/eeg_loader.py
import h5py
import mne
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import logging
mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.WARNING)

class EEGLoader:
    """
    EEG data loader

    Function: Process EEG feature data in HDF5 format
    """

    def __init__(self, config_path: str = "configs/feature_params.yaml"):
        """
        Initialize loader

        Params:
            config_path: Path to config file
        """
        self.config_path = Path(__file__).resolve().parents[3] / config_path
        self.feature_keys = [
            'alpha_power', 'beta_power', 'delta_power',
            'gamma_power', 'theta_power', 'mean',
            'std', 'wavelet_entropy'
        ]

    def load_features(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load HDF5 feature file

        Params:
            file_path:Path to .h5 file

        Returns:
            features: Feature data in dictionary format
        """
        features = {}
        try:
            with h5py.File(file_path, 'r') as f:
                for key in self.feature_keys:
                    if key in f:
                        features[key] = np.array(f[key])
                    else:
                        raise KeyError(f"Feature {key} not found in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

        return features

    def create_feature_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create feature vector
        (1) If frequency feature (with "power" in key, e.g. "alpha_power")
        then use logarithmic transformation to compress the dynamic range
        (2) Other feature don't need any extra procession

        Params:
            features: Raw feature dictionary

        Returns:
            feature_vector: Concatenated feature vector
        """
        vectors = []
        for key in self.feature_keys:
            if key in features:
                # Handle feature normalization
                if 'power' in key:
                    vectors.append(np.log10(features[key] + 1e-12))    # Logarithmic transformation
                else:
                    vectors.append(features[key])
        return np.concatenate(vectors)

    def load_fif_features(self, fif_path: str) -> Optional[torch.Tensor]:
        """
            Load EEG data from a .fif file.

            Params:
                fif_path: Path to the .fif file

            Returns:
                Numpy array of EEG data
    """
        try:
            # Read .fif file using mne
            raw = mne.io.read_raw_fif(fif_path, preload=True)

            # Get EEG data (data matrix) and sampling rate
            eeg_data = raw.get_data()  # Returns a NumPy array with shape [channels, samples]
            sampling_rate = raw.info['sfreq']

            # Convert to PyTorch tensor
            eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)

            return eeg_data_tensor

        except Exception as e:
            print(f"Error loading {fif_path}: {str(e)}")
            return None