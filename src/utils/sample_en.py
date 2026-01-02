import torch
import numpy as np

class SampleEn:
    def __init__(self, embedding_dim=2, delay=1, r=0.2, window_size=1280):
        """
        Initialize the SampleEn class, set the embedding dimension, time delay, distance threshold r, and window size.

        Params:
            embedding_dim: Embedding dimension, default is 2.
            delay: Time delay, default is 1.
            r: Distance threshold used to judge the similarity of two time slices, default value is 0.2.
            window_size: Window size, default is 1280 (i.e., 5 seconds of EEG data).
        """
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.r = r
        self.window_size = window_size

    def _calculate_similarity(self, x1, x2):
        """
        Calculate the similarity of two time slices using maximum distance.

        Params:
            x1: The first time slice with shape [embedding_dim].
            x2: The second time slice with shape [embedding_dim].
        Returns:
            value: Maximum distance between the two time slices.
        """
        return np.max(np.abs(x1 - x2))

    def _sample_entropy(self, time_series: torch.Tensor, m: int, r: float) -> float:
        """
        Calculate the sample entropy of a given time series.

        Params:
            time_series: Input time series data with shape (num_samples,).
            m: Embedding dimension.
            r: Distance threshold.
        Returns:
            value: Calculated sample entropy.
        """
        N = len(time_series)
        X = np.array([time_series[i:i + m] for i in range(N - m)])  # Get all m-dimensional time slices
        similarities = []

        # Calculate the similarity between each pair of time slices
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                similarity = self._calculate_similarity(X[i], X[j])
                similarities.append(similarity)

        # Calculate sample entropy
        threshold = r * np.std(time_series)  # Calculate standard deviation based on threshold r
        num_similar_pairs = sum([sim <= threshold for sim in similarities])
        sampen = -np.log(num_similar_pairs / len(similarities)) if num_similar_pairs != 0 else float('inf')
        return sampen

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sample entropy of EEG signals (supports multichannel input).

        Params:
            eeg_data: Input signal with shape [channels, samples].
        Returns:
            value: Calculated sample entropy.
        """
        if eeg_data.ndimension() == 2:
            entropies = []
            for ch in range(eeg_data.shape[0]):
                channel_data = eeg_data[ch]
                entropy = self._sample_entropy(channel_data, self.embedding_dim, self.r)
                entropies.append(entropy)

            return torch.tensor(entropies, dtype=torch.float32)
        else:
            raise ValueError("Expected input signal to be a 2D array with shape [channels, samples].")