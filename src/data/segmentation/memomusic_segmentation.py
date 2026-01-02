# -*- coding: utf-8 -*-
"""
    Segment MemoMusic data based on experiment setup.
"""

import os
import warnings

import pandas as pd
import mne
import pretty_midi
from pathlib import Path
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

# get project root
def get_project_root(current_file: str = __file__) -> Path:
    return Path(current_file).resolve().parents[3]


def parse_time(time_str:any) -> float:
    """
        Convert a time string in the format of "minutes:seconds.centiseconds" to total seconds.

        Parameters:
            time_str: The input time value (non-string types will be converted to strings first, and the format should be "min:ss.cs").

        Returns:
            The converted total seconds (a floating-point number, accurate to 0.01 seconds).
    """
    if not isinstance(time_str, str):
        time_str = str(time_str)

    if ':' not in time_str or '.' not in time_str:
        raise ValueError(f"Invalid time format: {time_str}. Expected 'mm:ss.cs'.")

    try:
        minutes, rest = time_str.split(':')
        seconds, centiseconds = rest.split('.')
        return int(minutes) * 60 + int(seconds) + int(centiseconds) / 100
    except Exception as e:
        raise ValueError(f"Failed to parse time '{time_str}': {str(e)}")


def process_subject(sub_dir: str, user_df: pd.DataFrame, raw_path: Path, fragment_path: Path) -> None:
    """
    Process the experimental data of a single subject, including EEG signal segmentation and MIDI segment extraction.

    Parameters:
        sub_dir: The name of the subject's directory (in the format like "sub_01").
        user_df: A DataFrame containing subject information and time annotations.
        raw_path: The root path of the raw dataset.
        fragment_path: The root path where the processed data will be saved.

    Returns:
        No explicit return value. The processing results are saved in the specified output directory.
    """
    sub_id = sub_dir.split('_')[-1].zfill(2)  # Format user ID as two digits (e.g., "01")
    sub_path = os.path.join(raw_path, sub_dir)
    fragment_path = os.path.join(fragment_path, sub_dir)

    sub_data = user_df[user_df['user_ID'] == sub_id]

    # Create output directories if they don't exist
    (Path(fragment_path) / 'MemoMusic_EEG_fragment').mkdir(parents=True, exist_ok=True)
    (Path(fragment_path) / 'MemoMusic_MIDI_fragment').mkdir(parents=True, exist_ok=True)

    for _, row in sub_data.iterrows():
        exp_num = row['exp_num']
        music_num = row['music_num']
        music_id = str(row['music_ID'])

        # Parse start and end times, skip on format error
        try:
            start_sec = parse_time(row['Start'])
            end_sec = parse_time(row['End'])
        except Exception:
            continue

        # Process EEG data
        eeg_file = f"{sub_dir}_music_{exp_num}.cnt"
        eeg_path = os.path.join(sub_path, 'EEG', eeg_file)
        if os.path.exists(eeg_path):
            try:
                raw = mne.io.read_raw_cnt(eeg_path, preload=True)
                max_time = raw.times[-1]

                # Skip if end time exceeds EEG data duration
                if end_sec > max_time:
                    continue

                raw_crop = raw.crop(tmin=start_sec, tmax=end_sec)
                output_name = f"{sub_dir}_exp{exp_num}_music{music_num}_raw.fif"
                raw_crop.save(Path(fragment_path) / 'MemoMusic_EEG_fragment' / output_name, overwrite=True)
            except Exception:
                continue

        # Process MIDI data
        mid_pattern = f"exp_{exp_num}_music_{music_num}_mid_{music_id}.*"
        midi_files = list(Path(os.path.join(sub_path, 'MIDI')).glob(mid_pattern))
        if midi_files:
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
                new_midi = pretty_midi.PrettyMIDI()

                for inst in midi.instruments:
                    new_inst = pretty_midi.Instrument(program=inst.program)
                    for note in inst.notes:
                        if start_sec <= note.start <= end_sec:
                            new_note = pretty_midi.Note(
                                velocity=note.velocity,
                                pitch=note.pitch,
                                start=note.start - start_sec,
                                end=min(note.end, end_sec) - start_sec
                            )
                            new_inst.notes.append(new_note)
                    if new_inst.notes:
                        new_midi.instruments.append(new_inst)

                output_name = f"{sub_dir}_exp{exp_num}_music{music_num}_mid{music_id}.mid"
                new_midi.write(Path(fragment_path) / 'MemoMusic_MIDI_fragment' / output_name)
            except Exception:
                continue


if __name__ == "__main__":
    project_root = get_project_root()
    config_path = "../../../configs/paths.yaml"

    with open(config_path, "r") as f:
        paths = yaml.safe_load(f)

    raw_path = Path(project_root) / paths["raw_user_data_dir"]
    fragment_path = Path(project_root) / paths["processed_data_dir"]
    user_excel_path = os.path.join(raw_path, 'user.xlsx')

    user_df = pd.read_excel(user_excel_path, sheet_name='MemoMusic')
    user_df['user_ID'] = user_df['user_ID'].astype(str).str.zfill(2)
    user_df['Start'] = user_df['Start'].astype(str)
    user_df['End'] = user_df['End'].astype(str)

    for sub_dir in os.listdir(raw_path):
        if sub_dir.startswith('sub_'):
            process_subject(sub_dir, user_df, raw_path, fragment_path)

    print("\n========== All Done ==========")