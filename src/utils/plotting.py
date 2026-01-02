
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses):
    """
    Plot loss curves per epoch for different components: MSE, KL, VA Penalty, Total.
    Save one figure per loss type.

    Parameters:
        train_losses (dict): Dict of training losses per epoch.
        val_losses (dict): Dict of validation losses per epoch.
    """
    def convert_to_numpy(loss_list):
        return [loss.cpu().numpy() if isinstance(loss, torch.Tensor) and loss.is_cuda else
                loss.numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_list]

    loss_types = ['MSE', 'KL', 'VA Penalty', 'Total']
    for loss_type in loss_types:
        if loss_type in train_losses and loss_type in val_losses:
            plt.figure(figsize=(10, 6))
            if train_losses[loss_type]:
                train_loss = convert_to_numpy(train_losses[loss_type])
                plt.plot(train_loss, label=f'Train {loss_type}', color='blue')
            if val_losses[loss_type]:
                val_loss = convert_to_numpy(val_losses[loss_type])
                plt.plot(val_loss, label=f'Validation {loss_type}', color='red')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'{loss_type} Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'../models/plot/loss/{loss_type.lower()}_loss_curve.png')
            plt.close()


def plot_va_trajectory(predicted_va, true_va, sub_id, exp_id, music_id, label="VA"):
    """
    Plot predicted vs. true VA trajectory for a single music sample.

    Parameters:
        predicted_va (array): Predicted values, shape (T,)
        true_va (array): Ground truth values, shape (T,)
        sub_id, exp_id, music_id: identifiers used in the figure name
        label (str): 'Valence' or 'Arousal'
    """
    plt.figure(figsize=(10, 6))
    time_steps = list(range(1, len(predicted_va) + 1))
    plt.plot(time_steps, predicted_va, label=f"Predicted {label}", color='blue', marker='o')
    plt.plot(time_steps, true_va, label=f"True {label}", color='red', marker='x')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{label} Value')
    plt.title(f'{label} Trajectory - sub {sub_id}, exp {exp_id}, music {music_id}')
    plt.legend()
    plt.grid(True)
    if label.lower() == 'valence':
        plt.ylim(-5, 5)
    elif label.lower() == 'arousal':
        plt.ylim(0, 10)
    save_path = f'../models/plot/va_trajectory/{label.lower()}_{sub_id}_{exp_id}_{music_id}_trajectory.png'
    plt.savefig(save_path)
    plt.close()


def plot_metrics_distribution(df: pd.DataFrame, save_dir: str, group_by: str = 'Valence') -> None:
    """
    Plot all metrics related to one emotion dimension (Valence or Arousal) as separate figures.

    Parameters:
        df (pd.DataFrame): DataFrame containing metric results per sample.
        save_dir (str): Path to save generated plots.
        group_by (str): Emotion dimension to filter (either 'Valence' or 'Arousal').

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select metrics with appropriate suffix (e.g., _Valence)
    suffix = f'_{group_by}'
    metric_cols = [col for col in df.columns if col.endswith(suffix) and col != 'ID']

    # Plot each metric separately
    for metric in metric_cols:
        save_path = os.path.join(save_dir, f'{group_by}_metric_{metric}.png')
        plot_single_metric(df, metric, save_path)


def plot_single_metric(df: pd.DataFrame, metric_name: str, save_path: str) -> None:
    """
    Automatically select the appropriate plot type for a given metric.

    Parameters:
        df (pd.DataFrame): DataFrame containing metric values.
        metric_name (str): Name of the metric column to plot.
        save_path (str): Path to save the generated figure.

    Returns:
        None
    """
    data = df[metric_name].dropna()
    plt.figure(figsize=(8, 5))

    # Plot histograms for numeric distance metrics
    if any(key in metric_name for key in ['MSE', 'DTW', 'AreaBetweenCurves', 'FirstStepError']):
        sns.histplot(data, kde=True, bins=12, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of {metric_name}")
        plt.xlabel(metric_name)
        plt.ylabel("Count")

    # Plot sorted bar for Pearson correlations
    elif 'Pearson' in metric_name:
        sorted_vals = data.sort_values().reset_index(drop=True)
        sns.barplot(x=sorted_vals.index, y=sorted_vals, hue=sorted_vals.index, palette='coolwarm', legend=False)
        plt.title(f"Sorted Pearson Coefficients for {metric_name}")
        plt.ylabel("Pearson Correlation")
        plt.xlabel("Sample Index")

    # Plot bar + mean line for agreement metrics
    elif any(key in metric_name for key in ['TrendAgreement', 'TurningPointsMatchRate']):
        sns.barplot(x=np.arange(len(data)), y=data, color='orange')
        mean_val = data.mean()
        plt.axhline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
        plt.title(f"{metric_name} across samples")
        plt.xlabel("Sample")
        plt.ylabel("Agreement Rate")
        plt.legend()

    # Fallback: use boxplot
    else:
        sns.boxplot(y=data)
        plt.title(f"Boxplot of {metric_name}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
