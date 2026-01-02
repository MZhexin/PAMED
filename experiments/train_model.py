# -*- coding: utf-8 -*-
"""
    Train-Validate-Test Python file for PAMED.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from typing import Tuple

from src.models.model import PAMED
from src.utils.checkpoint import save_checkpoint
from src.utils.data_loader import split_data, prepare_dataset
from src.utils.eeg_loader import EEGLoader
from src.utils.evaluate import evaluate_va_metrics
from src.utils.plotting import plot_loss_curve, plot_va_trajectory, plot_metrics_distribution
from src.utils.va_loader import VALoader
from src.utils.va_mapper import VAMapper

# Setting device for training (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """
        Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate the PAMED model.")

    # Path to data
    parser.add_argument('--data_root', type=str, default='../data/processed', help="Path to the processed data.")
    parser.add_argument('--va_path', type=str, default='../data/raw/User_Data/user.xlsx', help="Path to the VA labels.")
    parser.add_argument('--epochs', type=int, default=30, help="Number of total epochs to train the model.")
    parser.add_argument('--mu_epochs', type=int, default=None, help="Number of epochs to train the mu parameter (used only in train_mu_then_delta).")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument('--kl_weight', type=float, default=0.5, help="Weight for the KL divergence loss term.")
    parser.add_argument('--va_weight', type=float, default=0.5, help="Weight for the VA penalty loss term.")
    parser.add_argument('--save_dir', type=str, default='../models', help="Directory to save the trained models.")
    parser.add_argument('--val_ratio', type=float, default=0.13, help="The ratio of data to use for validation.")
    parser.add_argument('--entropy_method', type=str, default='MDistEn',
                        choices=['MDistEn', 'SampleEntropy', 'TimeFreqFeatures'],
                        help="Method for calculating entropy features.")
    parser.add_argument('--optimizer_method', type=str, default='Euler',
                        choices=['Euler', 'Runge-Kutta', 'AdaptiveStep'], help="Method for parameter updates.")
    parser.add_argument('--loss_method', type=str, default='MSE + VA Penalty + KL',
                        choices=['MSE', 'MSE + KL', 'MSE + VA Penalty', 'MSE + VA Penalty + KL'],
                        help="Loss function method.")
    parser.add_argument('--model_type', type=str, default='PAMED', choices=['PAMED', 'LSTM', 'Transformer'],
                        help="Type of model to use.")
    parser.add_argument('--train_method', type=str, default='train_mu_then_delta',
                        choices=['train_mu', 'train_mu_delta', 'train_mu_then_delta'],
                        help="Method for training the model.")
    parser.add_argument('--mu_data', type=str, default='IADS-E + MemoMusic', choices=['IADS-E', 'IADS-E + MemoMusic'],
                        help="The data source to initialize μ.")
    return parser.parse_args()


def load_data(data_root: str, va_path: str, mu_data: str, train_method: str) -> dict:
    """
    Load EEG and VA data.

    Parameters:
        data_root: Path to processed data (../data/processed/)
        va_path: Path to user.xlsx which records VA labels (../data/raw/User_Data/user.xlsx)
        mu_data: Dataset used to train parameter mu of PAMED ('IADS-E' or 'IADS-E + MemoMusic')
        train_method: Method for training the model ('train_mu', 'train_mu_delta' or 'train_mu_then_delta')

    Returns:
        data_dict: Nested dictionary structured as:
                   data_dict['IADS-E'][subject][Q][music] = {'EEG': ..., 'label': ...}
                   data_dict['MemoMusic'][subject][exp][music] = {'EEG': ..., 'mu_label': ..., 'delta_label': ...}
    """
    eeg_loader = EEGLoader()
    va_loader = VALoader(va_path)
    va_mapper = VAMapper()

    all_subjects = [f"sub_{i:02d}" for i in range(1, 30)]

    memo_va_labels = {}
    iads_va_labels = {}
    eeg_features = {}

    for subject_id in all_subjects:
        memo_va_labels[subject_id] = va_loader.get_memo_va(subject_id)
        iads_va_labels[subject_id] = va_loader.get_iads_va(subject_id)

    for subject_id in all_subjects:
        subject_data = {'IADS-E': {}, 'MemoMusic': {}}

        # Load IADS-E EEG
        music_num_per_q = [10, 10, 10, 5]
        for q_index, music_count in enumerate(music_num_per_q, 1):
            q_num = f'Q{q_index}'
            subject_data['IADS-E'][q_num] = {}
            for music_num in range(1, music_count + 1):
                music_id = f'music_{music_num:02d}'
                path = f"{data_root}/{subject_id}/IADS_EEG_fragment/{subject_id}_Q{q_index}_music{music_num:02}_raw.fif"
                eeg_data = eeg_loader.load_fif_features(path)
                if eeg_data is not None:
                    subject_data['IADS-E'][q_num][music_id] = {'EEG': eeg_data}

        # Load MemoMusic EEG
        for exp_num in range(1, 3):
            exp_id = f'exp_{exp_num}'
            subject_data['MemoMusic'][exp_id] = {}
            for music_num in range(1, 5):
                music_id = f'music_{music_num:02d}'
                path = f"{data_root}/{subject_id}/MemoMusic_EEG_fragment/{subject_id}_exp{exp_num}_music{music_num}_raw.fif"
                eeg_data = eeg_loader.load_fif_features(path)
                if eeg_data is not None:
                    subject_data['MemoMusic'][exp_id][music_id] = {'EEG': eeg_data}

        eeg_features[subject_id] = subject_data

    # Project IADS-E VA labels to MemoMusic scale
    for subject_id in iads_va_labels:
        for q in iads_va_labels[subject_id]:
            for music_id in iads_va_labels[subject_id][q]:
                v_iads = iads_va_labels[subject_id][q][music_id]["music_V"]
                a_iads = iads_va_labels[subject_id][q][music_id]["music_A"]
                v_memo, a_memo = va_mapper.map_to_memo(v_iads, a_iads)
                iads_va_labels[subject_id][q][music_id]["music_V"] = v_memo
                iads_va_labels[subject_id][q][music_id]["music_A"] = a_memo

    # Generate pseudo labels if needed
    mu_pseudo_labels = None
    delta_pseudo_labels = None

    if 'MemoMusic' in mu_data:
        mu_pseudo_labels = generate_pseudo_labels(
            memo_va_labels, eeg_features, base_label_type='music',
            window_size=5, delta_max=1.0, max_perturbation=3.0,
            v_std_init=1.5, a_std_init=2.25
        )

    if 'delta' in train_method:
        delta_pseudo_labels = generate_pseudo_labels(
            memo_va_labels, eeg_features, base_label_type='after',
            window_size=5, delta_max=1.0, max_perturbation=3.0,
            v_std_init=1.5, a_std_init=2.25
        )

    # Reorganize data into final nested structure
    data_dict = {'IADS-E': {}, 'MemoMusic': {}}

    for s in all_subjects:
        # IADS-E
        data_dict['IADS-E'][s] = {}
        for q in eeg_features[s]['IADS-E']:
            data_dict['IADS-E'][s][q] = {}
            for m in eeg_features[s]['IADS-E'][q]:
                data_dict['IADS-E'][s][q][m] = {
                    'EEG': eeg_features[s]['IADS-E'][q][m]['EEG'],
                    'label': iads_va_labels[s][q][m]
                }

        # MemoMusic
        data_dict['MemoMusic'][s] = {}
        for exp in eeg_features[s]['MemoMusic']:
            data_dict['MemoMusic'][s][exp] = {}
            for m in eeg_features[s]['MemoMusic'][exp]:
                data_dict['MemoMusic'][s][exp][m] = {
                    'EEG': eeg_features[s]['MemoMusic'][exp][m]['EEG'],
                    'mu_label': (
                        mu_pseudo_labels[s][exp][m]
                        if mu_pseudo_labels is not None and s in mu_pseudo_labels and exp in mu_pseudo_labels[s] and m in mu_pseudo_labels[s][exp]
                        else None
                    ),
                    'delta_label': (
                        delta_pseudo_labels[s][exp][m]
                        if delta_pseudo_labels is not None and s in delta_pseudo_labels and exp in delta_pseudo_labels[s] and m in delta_pseudo_labels[s][exp]
                        else None
                    )
                }

    return data_dict


def generate_pseudo_labels(memo_va_labels: dict, eeg_features: dict, window_size: int = 5, delta_max: float = 1.0,
                           max_perturbation: float = 3.0, base_label_type: str = 'music', v_std_init: float = 1.5,
                           a_std_init: float = 2.25, std_decay: float = 0.8, min_std: float = 0.5) -> dict:
    """
    Generate dynamic pseudo-labels (Valence-Arousal sequences) from static VA labels in MemoMusic.
    The labels are expanded over time using smooth Gaussian perturbations with exponential std decay,
    and aligned with EEG signal duration (5s per timestep).

    Parameters:
        memo_va_labels (dict): Nested VA label dictionary [subject][exp][music] → {V_music, A_after, ...}
        eeg_features (dict): EEG tensor data [subject]['MemoMusic'][exp][music]['EEG']
        window_size (int): Duration (in seconds) per time step (default: 5s)
        delta_max (float): Max change allowed between consecutive time steps
        max_perturbation (float): Global deviation limit from the static base label
        base_label_type (str): Use 'music' or 'after' as base label type
        v_std_init (float): Initial std for valence perturbation
        a_std_init (float): Initial std for arousal perturbation
        std_decay (float): Exponential decay rate for perturbation std (0~1)
        min_std (float): Minimum std to avoid overly flat sequences

    Returns:
        pseudo_labels (dict): Nested dict [subject][exp][music] → Tensor(2, T) where T is time steps
    """
    pseudo_labels = {}

    for subject_id in memo_va_labels:
        subject_labels = {}

        for exp_id in memo_va_labels[subject_id]:
            exp_labels = {}

            for music_id in memo_va_labels[subject_id][exp_id]:
                # Skip missing or corrupted EEG entries
                if music_id not in eeg_features[subject_id]['MemoMusic'][exp_id]:
                    continue

                # Retrieve base valence and arousal labels
                v_key = f"V_{base_label_type}"
                a_key = f"A_{base_label_type}"
                v_base = memo_va_labels[subject_id][exp_id][music_id].get(v_key, 0)
                a_base = memo_va_labels[subject_id][exp_id][music_id].get(a_key, 0)

                # Calculate number of time steps based on EEG duration
                eeg_tensor = eeg_features[subject_id]['MemoMusic'][exp_id][music_id]['EEG']
                music_duration = eeg_tensor.size(1) / 1000  # from milliseconds to seconds
                time_steps = int(np.ceil(music_duration / window_size))

                v_sequence = []
                a_sequence = []

                v_prev = v_base
                a_prev = a_base

                for t in range(time_steps):
                    if t == 0:
                        # First step: use static base label directly
                        v_new = v_base
                        a_new = a_base
                    else:
                        # Apply exponential decay to std, with floor
                        std_v = max(v_std_init * (std_decay ** t), min_std)
                        std_a = max(a_std_init * (std_decay ** t), min_std)

                        # Sample Gaussian noise
                        perturb_v = np.random.normal(0, std_v)
                        perturb_a = np.random.normal(0, std_a)

                        # Raw candidate update
                        v_candidate = v_base + perturb_v
                        a_candidate = a_base + perturb_a

                        # Global limit to base label range
                        v_candidate = np.clip(v_candidate, v_base - max_perturbation, v_base + max_perturbation)
                        a_candidate = np.clip(a_candidate, a_base - max_perturbation, a_base + max_perturbation)

                        # Local smoothness constraint to previous value
                        v_candidate = np.clip(v_candidate, v_prev - delta_max, v_prev + delta_max)
                        a_candidate = np.clip(a_candidate, a_prev - delta_max, a_prev + delta_max)

                        v_new = v_candidate
                        a_new = a_candidate

                    # Record current value
                    v_sequence.append(v_new)
                    a_sequence.append(a_new)

                    v_prev = v_new
                    a_prev = a_new

                # Save sequence as a (2, T) tensor: [V_seq, A_seq]
                music_tensor = torch.tensor([v_sequence, a_sequence], dtype=torch.float32)
                exp_labels[music_id] = music_tensor

            subject_labels[exp_id] = exp_labels
        pseudo_labels[subject_id] = subject_labels

    return pseudo_labels


# Training method definition
# Training function
def train(model, train_data: Tuple[dict, dict], epoch: int, optimizer, device) -> Tuple[dict, dict]:
    """
    Train the model for one epoch.

    Parameters:
        model: The PAMED model instance.
        train_data (tuple): Tuple (eeg_data, labels) as returned by prepare_dataset().
        epoch (int): Current training epoch index.
        optimizer: Optimizer used to update model parameters.
        device: Target device (CPU or CUDA).

    Returns:
        avg_loss (dict): Average loss across all samples.
        epoch_losses (dict): Per-instance loss values.
    """
    model.train()
    eeg_data, labels = train_data
    epoch_losses = {'MSE': [], 'VA Penalty': [], 'KL': [], 'Total': []}

    window_size = 5000       # 5-second windows
    min_window_size = 3500   # minimum length to retain last segment

    # Compute loss, backpropagate, and update weights (nested function in train)
    def compute_and_step(va_input, eeg_input, va_label):
        optimizer.zero_grad()
        output = model(va_input, eeg_input)
        total_loss, loss_dict = model.compute_loss(output, va_label)
        total_loss.backward()
        optimizer.step()
        loss_dict['Total'] = total_loss.item()
        for k in epoch_losses:
            epoch_losses[k].append(loss_dict.get(k, 0.0))

    for group in eeg_data:  # 'Q1' or 'exp_1'
        for music_id in eeg_data[group]:
            data = eeg_data[group][music_id].to(device)
            label = labels[group][music_id]

            if isinstance(label, dict):
                va_input = torch.tensor([label['Valence'], label['Arousal']], dtype=torch.float32, device=device)
                va_label = torch.tensor([label['music_V'], label['music_A']], dtype=torch.float32, device=device)
                eeg_input = data.float()
                compute_and_step(va_input, eeg_input, va_label)

            else:
                label = label.to(device)  # (2, T)
                sequence_length = data.shape[1]
                for i in range(1, label.shape[1]):
                    va_input = label[:, i - 1]
                    va_label = label[:, i]
                    end_idx = (i + 1) * window_size
                    if end_idx > sequence_length:
                        if sequence_length < min_window_size:
                            break
                        end_idx = sequence_length
                    eeg_input = data[:, 0:end_idx].float()
                    compute_and_step(va_input, eeg_input, va_label)

    avg_loss = {
        k: float(np.mean([x.cpu().item() if torch.is_tensor(x) else x for x in v])) if v else 0.0
        for k, v in epoch_losses.items()
    }
    return avg_loss, epoch_losses

# Validation function
def validate(model, val_data: tuple[dict, dict], epoch: int, device) -> tuple[dict, dict]:
    """
    Validate the model on held-out data.

    Parameters:
        model: PAMED model.
        val_data (tuple): Tuple of (eeg_data, labels).
        epoch (int): Current validation epoch.
        device: Target computation device.

    Returns:
        avg_loss (dict): Average validation losses.
        epoch_losses (dict): Individual step losses.
    """
    model.eval()
    eeg_data, labels = val_data
    epoch_losses = {'MSE': [], 'VA Penalty': [], 'KL': [], 'Total': []}
    window_size = 5000
    min_window_size = 3500

    # Forward pass and record loss (nested function in validate)
    def compute_and_track(va_input, eeg_input, va_label):
        output = model(va_input, eeg_input)
        total_loss, loss_dict = model.compute_loss(output, va_label)
        loss_dict['Total'] = total_loss.item()
        for k in epoch_losses:
            epoch_losses[k].append(loss_dict.get(k, 0.0))

    with torch.no_grad():
        for group in eeg_data:
            for music_id in eeg_data[group]:
                data = eeg_data[group][music_id].to(device)
                label = labels[group][music_id]

                if isinstance(label, dict):
                    va_input = torch.tensor([label['Valence'], label['Arousal']], dtype=torch.float32, device=device)
                    va_label = torch.tensor([label['music_V'], label['music_A']], dtype=torch.float32, device=device)
                    eeg_input = data.float()
                    compute_and_track(va_input, eeg_input, va_label)

                else:
                    label = label.to(device)
                    sequence_length = data.shape[1]
                    for i in range(1, label.shape[1]):
                        va_input = label[:, i - 1]
                        va_label = label[:, i]
                        end_idx = (i + 1) * window_size
                        if end_idx > sequence_length:
                            if sequence_length < min_window_size:
                                break
                            end_idx = sequence_length
                        eeg_input = data[:, 0:end_idx].float()
                        compute_and_track(va_input, eeg_input, va_label)

    avg_loss = {
        k: float(np.mean([x.cpu().item() if torch.is_tensor(x) else x for x in v])) if v else 0.0
        for k, v in epoch_losses.items()
    }
    return avg_loss, epoch_losses


# Testing function
def test(model: torch.nn.Module, test_data: tuple[dict, dict], device: torch.device, save_dir: str):
    """
    Test the model on MemoMusic data by recursively predicting emotional states over time.

    Parameters:
        model (torch.nn.Module): Trained PAMED model instance.
        test_data (tuple[dict, dict]): Tuple containing (EEG data, ground-truth VA sequences).
        device (torch.device): Computation device ('cuda' or 'cpu').
        save_dir (str): Directory to save results (.npy, .csv, metrics, and plots).

    Returns:
        tuple[list, list]: All predicted VA trajectories and true labels for each music segment.
    """
    model.eval()

    eeg_data, labels = test_data
    all_predictions = []
    all_true_labels = []
    all_metrics = []

    # Prepare directories
    npy_dir = os.path.join(save_dir, "results/npy")
    csv_dir = os.path.join(save_dir, "results/csv")
    plot_dir = os.path.join(save_dir, "plot")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    for sub_id in eeg_data:
        for exp_num in eeg_data[sub_id]:
            for music_id in eeg_data[sub_id][exp_num]:

                # Load EEG and label sequence
                data = eeg_data[sub_id][exp_num][music_id].to(device).float()
                label = labels[sub_id][exp_num][music_id].to(device).float()
                sequence_length = label.shape[1]

                predictions = []
                for t in range(1, sequence_length):
                    s_prev = label[:, t - 1]
                    eeg_input = data[:, :t * 1000]
                    pred = model(s_prev, eeg_input)
                    predictions.append(pred.unsqueeze(1))

                # Concatenate predictions
                va_pred = torch.cat(predictions, dim=1).detach().cpu().numpy()
                va_true = label[:, 1:].detach().cpu().numpy()

                base_name = f"{sub_id}_{exp_num}_{music_id}"

                # Save predictions and labels
                np.save(os.path.join(npy_dir, f"{base_name}_predicted.npy"), va_pred)
                np.save(os.path.join(npy_dir, f"{base_name}_true.npy"), va_true)

                df = pd.DataFrame({
                    'Pred_Valence': va_pred[0],
                    'Pred_Arousal': va_pred[1],
                    'True_Valence': va_true[0],
                    'True_Arousal': va_true[1],
                })
                df.to_csv(os.path.join(csv_dir, f"{base_name}_va.csv"), index=False)

                # Plot VA trajectories
                plot_va_trajectory(va_pred[0], va_true[0], sub_id=sub_id, exp_id=exp_num, music_id=music_id, label="Valence")
                plot_va_trajectory(va_pred[1], va_true[1], sub_id=sub_id, exp_id=exp_num, music_id=music_id, label="Arousal")

                # Evaluate advanced metrics
                metrics = evaluate_va_metrics(va_pred, va_true)
                metrics['ID'] = base_name
                all_metrics.append(metrics)

                all_predictions.append(va_pred)
                all_true_labels.append(va_true)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "results/metrics/test.csv"), index=False)

    # Save summary statistics
    stats_row = metrics_df.drop(columns=["ID"]).agg(['mean', 'std']).T
    stats_row.to_csv(os.path.join(save_dir, "results/metrics/summary.csv"))

    # Draw per-metric plots (Valence / Arousal)
    plot_metrics_distribution(metrics_df, os.path.join(save_dir, "plot/metrics"), group_by="Valence")
    plot_metrics_distribution(metrics_df, os.path.join(save_dir, "plot/metrics"), group_by="Arousal")


def save_all_va_trajectories(model: torch.nn.Module, full_data: dict, device: torch.device, output_dir: str):
    """
    Generate VA prediction trajectories for all subjects on MemoMusic data and save to disk.

    Parameters:
        model (torch.nn.Module): Trained model instance.
        full_data (dict): data_dict['MemoMusic'] from load_data().
        device (torch.device): 'cuda' or 'cpu'.
        output_dir (str): Base directory to save trajectories (will create model/trajectory).
    """
    model.eval()
    trajectory_dir = os.path.join(output_dir, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)

    for sub_id, subject_data in full_data.items():
        for exp_id, exp_data in subject_data.items():
            for music_id, music_data in exp_data.items():
                eeg = music_data['EEG']
                if eeg is None:
                    continue
                label = music_data['mu_label']
                if label is None:
                    continue

                data = eeg.to(device).float()
                label = label.to(device).float()
                sequence_length = label.shape[1]

                predictions = []
                for t in range(1, sequence_length):
                    s_prev = label[:, t - 1]
                    eeg_input = data[:, :t * 1000]
                    pred = model(s_prev, eeg_input)
                    predictions.append(pred.unsqueeze(1))

                va_pred = torch.cat(predictions, dim=1).detach().cpu().numpy()

                sub_num = int(sub_id.split("_")[1])
                exp_num = int(exp_id.split("_")[1])
                music_num = int(music_id.split("_")[1])
                filename = f"sub_{sub_num:02d}_exp_{exp_num}_music_{music_num}.npy"
                save_path = os.path.join(trajectory_dir, filename)

                np.save(save_path, va_pred)


def calculate_accuracy(model: torch.nn.Module, data: dict, device: torch.device, threshold: float = 1.0) -> tuple[float, float]:
    """
    Calculate prediction accuracy for Valence and Arousal on MemoMusic dataset.

    Parameters:
        model (torch.nn.Module): Trained model instance.
        data (dict): Nested dictionary with EEG and delta_label for each music.
        device (torch.device): Target computation device.
        threshold (float): Threshold for accepting prediction as correct (default: 0.5).

    Returns:
        tuple[float, float]: (Valence accuracy, Arousal accuracy) across all music samples.
    """
    model.eval()
    correct_V = 0
    correct_A = 0
    total_V = 0
    total_A = 0

    with torch.no_grad():
        for sub_id in data:
            for exp_id in data[sub_id]:
                for music_id in data[sub_id][exp_id]:
                    music_data = data[sub_id][exp_id][music_id]

                    # Get EEG signal and delta label
                    eeg = music_data.get('EEG')
                    label = music_data.get('mu_label')

                    if eeg is None or label is None:
                        continue

                    eeg = eeg.to(device).float()
                    label = label.to(device).float()
                    seq_len = label.shape[1]

                    # Predict the VA sequence recursively
                    for t in range(1, seq_len):
                        s_prev = label[:, t - 1]
                        va_true = label[:, t]
                        eeg_input = eeg[:, :t * 1000]

                        va_pred = model(s_prev, eeg_input)

                        # Compare predictions
                        error = torch.abs(va_pred - va_true)
                        if error[0] < threshold:
                            correct_V += 1
                        if error[1] < threshold:
                            correct_A += 1

                        total_V += 1
                        total_A += 1

    # Final accuracy calculation
    acc_V = correct_V / total_V if total_V > 0 else 0.0
    acc_A = correct_A / total_A if total_A > 0 else 0.0
    return acc_V, acc_A



def main():
    """
    Main training loop for PAMED model.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = args.save_dir
    checkpoint_dir = os.path.join(save_dir, "PAMED")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Start")

    # Load data
    data_dict = load_data(args.data_root, args.va_path, args.mu_data, args.train_method)
    print("Data Loaded")

    # Initialize model
    model = PAMED(device=str(device), entropy_method=args.entropy_method, optimizer_method=args.optimizer_method,
        train_method=args.train_method, mu_data=args.mu_data, loss_method=args.loss_method, kl_weight=args.kl_weight,
        va_weight=args.va_weight).to(device)

    # Subject split
    subjects = list(data_dict['IADS-E'].keys())
    train_subjects, val_subjects, test_subjects = split_data(subjects, val_ratio=args.val_ratio)
    print(f"Train/Val/Test sizes: {len(train_subjects)}, {len(val_subjects)}, {len(test_subjects)}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses = {'MSE': [], 'VA Penalty': [], 'KL': [], 'Total': []}
    val_losses = {'MSE': [], 'VA Penalty': [], 'KL': [], 'Total': []}

    # Run one training + validation step (nested function in main)
    def run_one_epoch(current_epoch, train_data, val_data):
        train_loss, _ = train(model, train_data, current_epoch, optimizer, device)
        val_loss, _ = validate(model, val_data, current_epoch, device)
        for k in train_losses:
            train_losses[k].append(train_loss[k])
            val_losses[k].append(val_loss[k])
        scheduler.step()
        save_checkpoint(model, optimizer, current_epoch, checkpoint_dir)

    if args.train_method == 'train_mu_then_delta':
        mu_epochs = args.mu_epochs if args.mu_epochs is not None else args.epochs // 2

        # Stage 1: train mu
        mu_train_batch = prepare_dataset(data_dict, train_subjects, args.mu_data, stage='mu')
        mu_val_batch = prepare_dataset(data_dict, val_subjects, args.mu_data, stage='mu')
        print('Start training mu')
        for ep in tqdm(range(1, mu_epochs + 1)):
            run_one_epoch(ep, mu_train_batch, mu_val_batch)

        # Stage 2: train delta
        delta_train_batch = prepare_dataset(data_dict, train_subjects, args.mu_data, stage='delta')
        delta_val_batch = prepare_dataset(data_dict, val_subjects, args.mu_data, stage='delta')
        print('Start training delta')
        for ep in tqdm(range(mu_epochs + 1, args.epochs + 1)):
            run_one_epoch(ep, delta_train_batch, delta_val_batch)

    elif args.train_method == 'train_mu_delta':
        # Joint training of mu and delta on IADS-E + MemoMusic
        full_train_batch = prepare_dataset(data_dict, train_subjects, args.mu_data, stage='full')
        full_val_batch = prepare_dataset(data_dict, val_subjects, args.mu_data, stage='full')
        print('Start joint training of mu and delta')
        for ep in tqdm(range(1, args.epochs + 1)):
            run_one_epoch(ep, full_train_batch, full_val_batch)

    else:  # args.train_method == 'train_mu'
        # Standard mu-only training
        mu_train_batch = prepare_dataset(data_dict, train_subjects, args.mu_data, stage='mu')
        mu_val_batch = prepare_dataset(data_dict, val_subjects, args.mu_data, stage='mu')
        print('Start standard mu-only training')
        for ep in tqdm(range(1, args.epochs + 1)):
            run_one_epoch(ep, mu_train_batch, mu_val_batch)

    # Plot loss
    plot_loss_curve(train_losses, val_losses)

    # Final test
    test_stage = 'delta' if args.train_method == 'train_mu_then_delta' else ('full' if args.train_method == 'train_mu_delta' else 'mu')

    test_batch = prepare_dataset(data_dict, test_subjects, args.mu_data, stage=test_stage, by_subject=True)

    test(model, test_batch, device, save_dir=args.save_dir)

    # Save VA trajectories
    save_all_va_trajectories(model, data_dict['MemoMusic'], device, save_dir)

    # Calculate ACC
    ACC_V, ACC_A = calculate_accuracy(model, data_dict['MemoMusic'], device)
    print(f"Valence Accuracy: {ACC_V:.4f}")
    print(f"Arousal Accuracy: {ACC_A:.4f}")

if __name__ == "__main__":
    main()
