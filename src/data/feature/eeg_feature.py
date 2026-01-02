# -*- coding: utf-8 -*-
"""
    Calculate EEG features.
"""

import mne
import numpy as np
import yaml
import h5py
from pathlib import Path

from tqdm import tqdm
import pywt

# get project root
def get_project_root(current_file=__file__):
    return Path(current_file).resolve().parents[3]

# consider average power spectral density (APSD) as band power
def compute_band_power(raw, band_range):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(),
        sfreq=raw.info['sfreq'],
        fmin=band_range[0],
        fmax=band_range[1]
    )
    return np.mean(psd, axis=1)

def compute_wavelet_entropy(data, wavelet='db4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    energy = [np.sum(np.square(c)) for c in coeffs]
    total_energy = np.sum(energy)
    return -np.sum([e / total_energy * np.log(e / total_energy) for e in energy])

def extract_eeg_features(raw, params):
    features = {}

    # time domain feature: mean and standard deviation
    data = raw.get_data()
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)

    # frequency domain feature: band power
    for band, freq_range in params['eeg']['frequency_bands'].items():
        features[f'{band}_power'] = compute_band_power(raw, freq_range)

    # time-frequency feature: wavelet entropy
    wavelet_entropy = []
    for ch_data in data:
        wavelet_entropy.append(compute_wavelet_entropy(
            ch_data,
            wavelet=params['eeg']['wavelet_type'],
            level=params['eeg']['wavelet_level']
        ))
    features['wavelet_entropy'] = np.array(wavelet_entropy)

    return features


def memomusic_process():
    processed_root = project_root / Path(paths['processed_data_dir'])

    # traverse each user
    for sub_id in tqdm(range(1, 30), desc="Processing Subjects"):
        sub_dir = f"sub_{sub_id:02d}"

        fragment_dir = processed_root / sub_dir / paths['user_data']['memomusic_eeg_fragment_dir']
        feature_dir = processed_root / sub_dir / paths['user_data']['memomusic_eeg_feature_dir']
        feature_dir.mkdir(parents=True, exist_ok=True)

        if any(feature_dir.iterdir()):
            for item in feature_dir.iterdir():
                item.unlink()

        for exp_num in [1, 2]:  # traverse each experiment round
            for music_num in range(1, 5):  # traverse each piece of music in this round
                input_file = fragment_dir / f"{sub_dir}_exp{exp_num}_music{music_num}_raw.fif"

                if not input_file.exists():
                    continue

                try:
                    # read EEG data
                    raw = mne.io.read_raw_fif(input_file, preload=True)

                    # extract feature
                    features = extract_eeg_features(raw, params)

                    # save as HDF5
                    output_file = feature_dir / f"{sub_dir}_exp_{exp_num}_music_{music_num}_features.h5"
                    with h5py.File(output_file, 'w') as hf:
                        for key, value in features.items():
                            hf.create_dataset(key, data=value)

                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
                    continue

def iads_process():
    processed_root = project_root / Path(paths['processed_data_dir'])

    # traverse each user
    for sub_id in tqdm(range(1, 30), desc="Processing Subjects"):
        sub_dir = f"sub_{sub_id:02d}"

        fragment_dir = processed_root / sub_dir / paths['user_data']['iads_eeg_fragment_dir']
        feature_dir = processed_root / sub_dir / paths['user_data']['iads_eeg_feature_dir']
        feature_dir.mkdir(parents=True, exist_ok=True)

        if any(feature_dir.iterdir()):
            for item in feature_dir.iterdir():
                item.unlink()

        for Q_group in [1, 2, 3, 4]:  # traverse each Q-group
            if Q_group != 4:
                music_pieces = 10
            else:
                music_pieces = 5

            for music_num in range(1, music_pieces + 1):  # traverse each piece of music in this group
                input_file = fragment_dir / f"{sub_dir}_Q{Q_group}_music{music_num:02d}_raw.fif"

                if not input_file.exists():
                    continue

                try:
                    # read EEG data
                    raw = mne.io.read_raw_fif(input_file, preload=True)

                    # extract feature
                    features = extract_eeg_features(raw, params)

                    # save as HDF5
                    output_file = feature_dir / f"{sub_dir}_Q_{Q_group}_music_{music_num:02d}_features.h5"
                    with h5py.File(output_file, 'w') as hf:
                        for key, value in features.items():
                            hf.create_dataset(key, data=value)

                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")
                    continue


if __name__ == "__main__":
    # get config path
    project_root = get_project_root()
    config_path = project_root / "configs/paths.yaml"
    config_param = project_root / "configs/feature_params.yaml"

    # load path config
    with open(config_path, "r") as f:
        paths = yaml.safe_load(f)

    # load parameter config
    with open(config_param) as f:
        params = yaml.safe_load(f)

    # MemoMusic
    memomusic_process()
    print("\n")
    print("========== MemoMusic Done ==========")
    print("\n")

    # IADS-E
    iads_process()
    print("\n")
    print("========== IADS-E Done ==========")
    print("\n")

    print("\n")
    print("========== All Done ==========")