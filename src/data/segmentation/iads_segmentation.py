# -*- coding: utf-8 -*-
"""
    Segment IDAS-E data based on experiment setup.
"""

import os
import mne
from pathlib import Path
import yaml

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")  # ignore warnings about date

# Experiment time parameters
EXPERIMENT_PARAMS = {
    'initial_rest': 60,                 # initial rest time
    'cue_delay': 10,                    # delay between cue and music
    'music_duration': 5,                # music length (seconds)
    'feedback_duration': 40,            # feedback duration
    'group_config': [10, 10, 10, 5]     # music num in each group (group 1-3: 10 pieces; group 4: 5 pieces)
}

# get project root
def get_project_root(current_file: str = __file__) -> Path:
    return Path(current_file).resolve().parents[3]

def generate_time_windows() -> list[dict[str, int]]:
    """
        Generate the time window list based on experiment setup.

        Parameters:

        Returns:
            windows: list of time windows (group, segment, start, end).
        """
    windows = []
    current_time = EXPERIMENT_PARAMS['initial_rest']

    for group_index, num_segments in enumerate(EXPERIMENT_PARAMS['group_config']):
        # when each experiment starts, set a rest-time (60 seconds)
        group_start = current_time + 60 if group_index > 0 else current_time

        for seg in range(num_segments):
            # calculate time step
            cue_start = group_start + seg * (EXPERIMENT_PARAMS['cue_delay'] +
                                             EXPERIMENT_PARAMS['music_duration'] +
                                             EXPERIMENT_PARAMS['feedback_duration'])

            music_start = cue_start + EXPERIMENT_PARAMS['cue_delay']
            music_end = music_start + EXPERIMENT_PARAMS['music_duration']

            windows.append({
                'group': group_index + 1,   # Group num, reflecting to Q1-Q4 VA category
                'segment': seg + 1,         # Music piece num
                'start': music_start,       # Start time of the piece
                'end': music_end            # End time of the piece
            })

        current_time = windows[group_index]['end']  # rest time between every two groups

    return windows

def process_subject(sub_dir: str, fragment_path: Path, paths: dict) -> int:
    """
    Process the experimental data of a single subject, including EEG signal segmentation and MIDI segment extraction.

    Parameters:
        sub_dir: The name of the subject's directory (in the format like "sub_01").
        fragment_path: The root path where the processed data will be saved.
        paths: Dictionary of paths to processed data (read from configuration file in the main function).

    Returns:
        No explicit return value. The processing results are saved in the specified output directory.
    """
    # path
    sub_id = Path(sub_dir).name
    eeg_path = os.path.join(sub_dir, 'EEG', f"{sub_id}_IADS.cnt")

    raw = mne.io.read_raw_cnt(eeg_path, preload=True)

    time_windows = generate_time_windows()

    fragment_dir = fragment_path / sub_id / paths['user_data']['iads_eeg_fragment_dir']
    fragment_dir.mkdir(parents=True, exist_ok=True)

    # cut and save
    for window in time_windows:
        raw_crop = raw.copy().crop(tmin=window['start'], tmax=window['end'])

        output_name = f"{sub_id}_Q{window['group']}_music{window['segment']:02d}_raw.fif"
        raw_crop.save(fragment_dir / output_name, overwrite=True)

    return len(time_windows)


if __name__ == "__main__":
    # get config path
    project_root = get_project_root()
    config_path = project_root / "configs/paths.yaml"

    # load path config
    with open(config_path, "r") as f:
        paths = yaml.safe_load(f)

    # path
    raw_path = project_root / paths['raw_user_data_dir']
    fragment_path = project_root / paths['processed_data_dir']

    for sub_dir in os.listdir(raw_path):
        if sub_dir.startswith('sub_'):
            processed = process_subject(sub_dir=os.path.join(raw_path, sub_dir), fragment_path=fragment_path, paths=paths)

    print("\n")
    print("========== All Done ==========")