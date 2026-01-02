from sklearn.model_selection import train_test_split

# Dataset splitting
def split_data(subjects, val_ratio=0.13, test_ratio=0.05):
    # Divide the training set and the remaining set (validation set + test set)
    train_subjects, remaining_subjects = train_test_split(subjects, test_size=val_ratio + test_ratio, random_state=42)

    # Divide the validation set and the test set from the remaining set
    val_subjects, test_subjects = train_test_split(remaining_subjects, test_size=test_ratio / (val_ratio + test_ratio),
                                                   random_state=42)

    return train_subjects, val_subjects, test_subjects


# Dataset preparation utility for PAMED training
def prepare_dataset(data_dict: dict, subject_list: list[str], mu_data: str, stage: str = 'mu', by_subject: bool = False) -> tuple[dict, dict]:
    """
    Construct EEG and label dictionaries for training, validation, or testing.

    Parameters:
        data_dict (dict): Nested data dictionary with 'IADS-E' and 'MemoMusic'.
        subject_list (list): List of subject IDs to include in the current subset.
        mu_data (str): 'IADS-E' or 'IADS-E + MemoMusic'.
        stage (str): One of ['mu', 'delta', 'full'] indicating training phase type.
        by_subject (bool): Whether to return subject-level structure [sub][exp][music].

    Returns:
        eeg_data (dict): EEG data organized by [Q/exp][music_id] or [sub][exp][music].
        labels (dict): Corresponding VA labels in same structure.
    """
    eeg_data = {}  # could be [exp][music] or [sub][exp][music]
    labels = {}

    for sub in subject_list:
        if by_subject:
            eeg_data[sub] = {}
            labels[sub] = {}

        # === IADS-E always used for 'mu' and 'full'
        if stage in ['mu', 'full']:
            for q in data_dict['IADS-E'].get(sub, {}):
                if by_subject:
                    eeg_data[sub].setdefault(q, {})
                    labels[sub].setdefault(q, {})
                    for m in data_dict['IADS-E'][sub][q]:
                        eeg_data[sub][q][m] = data_dict['IADS-E'][sub][q][m]['EEG']
                        labels[sub][q][m] = data_dict['IADS-E'][sub][q][m]['label']
                else:
                    eeg_data.setdefault(q, {})
                    labels.setdefault(q, {})
                    for m in data_dict['IADS-E'][sub][q]:
                        eeg_data[q][m] = data_dict['IADS-E'][sub][q][m]['EEG']
                        labels[q][m] = data_dict['IADS-E'][sub][q][m]['label']

        # === MemoMusic: conditionally used
        if (stage == 'mu' and mu_data == 'IADS-E + MemoMusic') or stage in ['delta', 'full']:
            for exp in data_dict['MemoMusic'].get(sub, {}):
                if by_subject:
                    eeg_data[sub].setdefault(exp, {})
                    labels[sub].setdefault(exp, {})
                else:
                    eeg_data.setdefault(exp, {})
                    labels.setdefault(exp, {})

                for m in data_dict['MemoMusic'][sub][exp]:
                    eeg_tensor = data_dict['MemoMusic'][sub][exp][m]['EEG']
                    label_tensor = (
                        data_dict['MemoMusic'][sub][exp][m]['mu_label']
                        if stage == 'mu' else
                        data_dict['MemoMusic'][sub][exp][m]['delta_label']
                    )
                    if label_tensor is not None:
                        if by_subject:
                            eeg_data[sub][exp][m] = eeg_tensor
                            labels[sub][exp][m] = label_tensor
                        else:
                            eeg_data[exp][m] = eeg_tensor
                            labels[exp][m] = label_tensor

    return eeg_data, labels



