# -*- coding: utf-8 -*-
"""
    Models.
"""

import torch
import torch.nn as nn

from src.utils.mdis_en import MDistEn
from src.utils.eeg_loader import EEGLoader
from src.utils.sample_en import SampleEn


class PAMED(nn.Module):
    """
    PAMED Model Class for Emotion Prediction.
    This model is used for predicting emotional states (valence and arousal),
    including training mu and delta, computing VA penalty, and other related tasks.

    Attributes:
        mu (torch.nn.Parameter): Parameter for Valence prediction.
        delta (torch.nn.Parameter): Parameter for Arousal prediction.
        alpha (torch.nn.Parameter): Alpha coefficient for controlling the influence of phi.
        phi (nn.Sequential): Neural network for processing entropy features.
        gamma (torch.nn.Parameter): Gamma matrix for VA Penalty (activated by loss_method).
        mdist_en (MDistEn): MDistEn object for calculating the entropy of EEG signals.
        eeg_loader (EEGLoader): EEG feature loader for processing EEG data.
    """

    def __init__(self, device='cuda', entropy_method='MDistEn', optimizer_method='Euler',
                 train_method='train_mu', mu_data='IADS-E', loss_method='MSE', kl_weight=0.5, va_weight=0.5):
        """
        Initialize the PAMED model.

        Args:
            device (str): Device type, default is 'cuda'. Options: 'cuda', 'cpu'.
            entropy_method (str): Entropy calculation method. Options: 'MDistEn', 'SampleEn', 'TimeFreqFeatures'.
            optimizer_method (str): Optimization method. Options: 'Euler', 'Runge-Kutta', 'Adaptive'.
            train_method (str): Training method. Options: 'train_mu', 'train_mu_and_delta', 'train_mu_then_delta'.
            mu_data (str): Data for training mu. Options: 'IADS-E', 'IADS-E + MemoMusic'.
            loss_method (str): Loss method. Options: 'MSE', 'MSE + KL', 'MSE + VA Penalty', 'MSE + VA Penalty + KL'.
        """
        super(PAMED, self).__init__()

        # Set device: choose 'cuda' if available, otherwise use 'cpu'
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_method = train_method
        self.mu_data = mu_data
        self.entropy_method = entropy_method
        self.optimizer_method = optimizer_method
        self.loss_method = loss_method
        self.kl_weight = kl_weight
        self.va_weight = va_weight

        # 初始化mu和delta，避免初始值过大
        if 'MemoMusic' in self.mu_data:
            valence_std = 2.00
            arousal_std = 2.05
        else:
            valence_std = 2.10
            arousal_std = 1.99

        memo_valence_std = 1.71
        memo_arousal_std = 1.82

        # 使用标准差初始化mu和delta，并缩放它们
        mu_init = torch.tensor([0.1 * valence_std, 0.1 * arousal_std], device=self.device)
        delta_init = torch.tensor([0.1 * memo_valence_std, 0.1 * memo_arousal_std], device=self.device)

        self.mu = nn.Parameter(mu_init)
        self.delta = nn.Parameter(delta_init)

        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device))
        self.beta = nn.Parameter(torch.tensor(0.5, device=self.device))

        # φ network for processing entropy features
        self.phi = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 2)
        ).to(self.device)

        # Initialize entropy calculation class
        if self.entropy_method == 'MDistEn':
            self.entropy_calculator = MDistEn()  # 使用MDistEn计算熵
        elif self.entropy_method == 'SampleEn':
            self.entropy_calculator = SampleEn()  # 使用SampleEn计算熵
        elif self.entropy_method == 'TimeFreqFeatures':
            self.entropy_calculator = None  # 不需要额外的计算方法，直接使用时频特征
        else:
            raise ValueError(f"Invalid entropy method: {self.entropy_method}")


        self.eeg_loader = EEGLoader()  # For loading EEG features

        # Initialize gamma matrix (only if loss_method requires it)
        self.gamma = nn.Parameter(torch.eye(2, device=self.device)) if 'VA Penalty' in loss_method else None

        # Set freeze flags for mu and delta based on the chosen training method
        self._set_freeze_flags()

    def _set_freeze_flags(self):
        """
        Set freeze flags for mu and delta based on the training method.

        - 'train_mu': Freeze delta, train mu.
        - 'train_mu_and_delta': Train both mu and delta.
        - 'train_mu_then_delta': Train mu first, then train delta.
        """
        if self.train_method == 'train_mu':
            self.freeze_mu = False
            self.freeze_delta = True  # Freeze delta, only train mu
        elif self.train_method == 'train_mu_and_delta':
            self.freeze_mu = False
            self.freeze_delta = False  # Train both mu and delta
        elif self.train_method == 'train_mu_then_delta':
            self.freeze_mu = False
            self.freeze_delta = True  # Freeze delta after training mu

    def forward(self, s_prev: torch.Tensor, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PAMED model.

        Args:
            s_prev (torch.Tensor): The previous emotional state (Valence, Arousal).
            eeg_data (torch.Tensor): The EEG data for the current time step, used to compute entropy.

        Returns:
            torch.Tensor: Predicted next emotional state (Valence, Arousal).
        """
        # devices
        s_prev = s_prev.to(self.device)
        eeg_data = eeg_data.to(self.device)

        # Freeze parameters based on training strategy
        if self.freeze_mu:
            self.mu.requires_grad = False
        if self.freeze_delta:
            self.delta.requires_grad = False

        # Compute entropy from EEG data
        entropy_value = self.compute_entropy(eeg_data).to(self.device)  # Now compute entropy based on EEG data
        phi_out = self.phi(entropy_value)  # Pass the entropy through the phi network

        # Calculate basic state change (dS)
        dS = (self.mu + self.delta).to(self.device) * s_prev + self.alpha * phi_out.to(self.device)

        # Calculate state change (dS) with VA penalty if applicable
        if 'VA Penalty' in self.loss_method:
            dS = dS + self.beta * torch.matmul(self.gamma, s_prev.to(torch.float32))

        # Select the optimizer method for state update
        if self.optimizer_method == 'Euler':
            s_next = self.euler_method(s_prev, dS)
        elif self.optimizer_method == 'Runge-Kutta':
            s_next = self.runge_kutta_method(s_prev, dS, phi_out)
        elif self.optimizer_method == 'AdaptiveStep':
            s_next = self.adaptive_step_method(s_prev, dS)
        else:
            raise ValueError(f"Invalid optimizer method: {self.optimizer_method}")

        # Ensure VA predictions are within valid range
        s_next[0] = 5 * torch.tanh(s_next[0])  # Valence range: [-5, 5]
        s_next[1] = 5 * torch.tanh(s_next[1]) + 5  # Arousal range: [0, 10]

        return s_next

    def compute_kl_loss(self, s_prev: torch.Tensor, diff_target: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度损失。

        Args:
            s_prev (torch.Tensor): 先前的情感状态。
            diff_target (torch.Tensor): 目标差异。

        Returns:
            torch.Tensor: KL散度损失。
        """
        # KL散度计算 (根据PAMED类中的描述进行)
        std = torch.tensor([1.0, 1.0]).to(self.device)  # 标准差
        p_pred = torch.distributions.Normal(s_prev, std)  # 伪代码，需调整根据具体实现
        p_true = torch.distributions.Normal(diff_target, std)
        kl_loss = torch.distributions.kl.kl_divergence(p_pred, p_true).mean()
        return kl_loss

    def va_penalty(self, outputs):
        """
        Calculate the VA Penalty loss based on the relationship between Valence and Arousal.
        """
        # Assume that outputs is a tensor of shape [2, N] representing [V, A] for each sample
        v = outputs[0]  # Valence
        a = outputs[1]  # Arousal

        coff = 1.0

        # Calculate the VA penalty using the gamma matrix
        gamma_vv = self.gamma[0, 0]  # Valence-V to Valence-V
        gamma_va = self.gamma[0, 1]  # Valence-V to Arousal-A
        gamma_av = self.gamma[1, 0]  # Arousal-A to Valence-V
        gamma_aa = self.gamma[1, 1]  # Arousal-A to Arousal-A

        # Compute VA penalty based on the relationship defined by the Gamma matrix
        penalty = (gamma_vv * v + gamma_va * a) + (gamma_av * v + gamma_aa * a)

        # Use ReLU to ensure penalty is always positive
        penalty = torch.nn.functional.softplus(penalty)

        # The penalty encourages the correct relationship between V and A.
        va_penalty_loss =  coff * penalty

        return va_penalty_loss

    def compute_entropy(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        计算EEG信号的熵，使用选定的熵计算方法。
        :param eeg_data: 输入EEG信号，形状为[batch_size, channels, samples]。
        :return: 熵值
        """
        if self.entropy_method == 'MDistEn' or self.entropy_method == 'SampleEn':
            return self.entropy_calculator(eeg_data)  # 调用MDistEn或SampleEn
        elif self.entropy_method == 'TimeFreqFeatures':
            return self.get_time_freq_features(eeg_data)  # 计算时频特征熵
        else:
            raise ValueError(f"Invalid entropy method: {self.entropy_method}")

    def get_time_freq_features(self, subject_id, data_root, mu_data):
        """
        根据给定的路径，加载被试的EEG特征，并拼接所有时域、频域、时频域特征。

        参数：
        - subject_id (str): 被试编号（例如 "sub_01"）
        - data_root (str): 数据根目录路径
        - mu_data (str): 数据源（"IADS-E" 或 "IADS-E + MemoMusic"）
        - sampling_rate (int): EEG信号的采样率，默认256Hz

        返回：
        - features (torch.Tensor): 拼接后的时间-频率特征
        """
        eeg_features = {}

        # IADS-E
        if 'IADS-E' in mu_data:
            # Q1-Q3 has 10 music, Q4 has 5 music
            music_count_per_q = [10, 10, 10, 5]

            for q, music_count in enumerate(music_count_per_q, 1):  # Q从1到4
                for music_num in range(1, music_count + 1):  # 遍历每个情感维度的音乐
                    iads_eeg_path = f"{data_root}/{subject_id}/IADS_EEG_feature/{subject_id}_Q_{q}_music_{music_num:02}_features.h5"
                    eeg_features.update(self.eeg_loader.load_features(iads_eeg_path))

        # MemoMusic
        if 'MemoMusic' in mu_data:
            for exp_num in range(1, 3):
                for music_num in range(1, 5):
                    memo_eeg_path = f"{data_root}/{subject_id}/MemoMusic_EEG_feature/{subject_id}_exp_{exp_num}_music_{music_num}_features.h5"
                    eeg_features.update(self.eeg_loader.load_features(memo_eeg_path))

        # Get EEG features
        features = self.eeg_loader.create_feature_vector(eeg_features)

        # Convert the features to torch.Tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        return features_tensor

    def euler_method(self, s_prev: torch.Tensor, dS: torch.Tensor) -> torch.Tensor:
        """
        Euler method for updating state.

        Args:
            s_prev (torch.Tensor): Previous state (Valence, Arousal).
            dS (torch.Tensor): The calculated state change.

        Returns:
            torch.Tensor: Updated state.
        """
        # Euler update: s_next = s_prev + dS
        s_next = s_prev + dS

        return s_next

    def runge_kutta_method(self, s_prev: torch.Tensor, dS: torch.Tensor, phi_out: torch.Tensor) -> torch.Tensor:
        """
        Runge-Kutta method (4th order) for updating state.

        Args:
            s_prev (torch.Tensor): Previous state (Valence, Arousal).
            dS (torch.Tensor): The calculated state change.
            phi_out (torch.Tensor): The output from the entropy-based network


        Returns:
            torch.Tensor: Updated state.
        """
        if 'VA Penalty' in self.loss_method:
            # Calculate k1, k2, k3, k4 using the differential equation
            k1 = dS  # Use the current state change as k1

            # For k2 and k3, simulate progression by updating s_prev using intermediate steps
            k2 = (self.mu + self.delta).to(self.device) * (s_prev + 0.5 * k1) + self.alpha * phi_out + self.beta * torch.matmul(self.gamma, k1)
            k3 = (self.mu + self.delta).to(self.device) * (s_prev + 0.5 * k2) + self.alpha * phi_out + self.beta * torch.matmul(self.gamma, k2)
            k4 = (self.mu + self.delta).to(self.device) * (s_prev + k3) + self.alpha * phi_out + self.beta * torch.matmul(self.gamma, k3)
        else:
            # Calculate k1, k2, k3, k4 using the differential equation
            k1 = dS  # Use the current state change as k1

            # For k2 and k3, simulate progression by updating s_prev using intermediate steps
            k2 = (self.mu + self.delta).to(self.device) * (s_prev + 0.5 * k1) + self.alpha * phi_out
            k3 = (self.mu + self.delta).to(self.device) * (s_prev + 0.5 * k2) + self.alpha * phi_out
            k4 = (self.mu + self.delta).to(self.device) * (s_prev + k3) + self.alpha * phi_out

        # Combine all k values to update the next state
        s_next = s_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return s_next

    def adaptive_step_method(self, s_prev: torch.Tensor, dS: torch.Tensor) -> torch.Tensor:
        """
        Adaptive step method for updating state.

        Args:
            s_prev (torch.Tensor): Previous state (Valence, Arousal).
            dS (torch.Tensor): The calculated state change.

        Returns:
            torch.Tensor: Updated state with adaptive step size.
        """
        # Calculate the error based on dS (can be refined)
        error_estimate = torch.norm(dS, p=2)  # L2 norm of the state change

        # Define an adaptive learning rate based on error (can use more sophisticated methods)
        learning_rate = 1 / (1 + error_estimate)  # Simple inverse error-based learning rate adjustment

        # Update state using the adaptive step
        s_next = s_prev + learning_rate * dS

        return s_next

    def compute_loss(self, predict, target):
        """
        Compute the value of the loss functions.

        Returns:
            total_loss: Combined loss value incorporating all active loss components
            loss_dict: Dictionary with individual loss values:
        """
        loss_dict = {}

        # MSE Loss (Mean Squared Error)
        mse_loss = torch.nn.MSELoss()(predict.to(self.device), target.to(self.device))
        loss_dict['MSE'] = mse_loss.item()

        # VA Penalty
        if 'VA Penalty' in self.loss_method:
            va_penalty_loss = self.va_penalty(predict.to(self.device))
            loss_dict['VA Penalty'] = va_penalty_loss
        else:
            loss_dict['VA Penalty'] = None

        # KL Divergence
        if 'KL' in self.loss_method:
            kl_loss = self.compute_kl_loss(predict.to(self.device), target.to(self.device))
            loss_dict['KL'] = kl_loss
        else:
            loss_dict['KL'] = None

        # Total Loss
        total_loss = mse_loss
        if loss_dict['VA Penalty'] is not None:
            total_loss += self.va_weight * va_penalty_loss
        if loss_dict['KL'] is not None:
            total_loss += self.kl_weight * kl_loss

        loss_dict['Total'] = total_loss.item()
        return total_loss, loss_dict