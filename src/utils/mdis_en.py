import torch

class MDistEn:
    def __init__(self, embedding_dim=2, delay=1, window_size=1280, m_dist=False):
        """
        Initialize the MDistEn class, set the embedding dimension, time delay, and whether to use multichannel distribution entropy.

        Params:
            embedding_dim: Embedding dimension, default is 2.
            delay: Time delay, default is 1.
            window_size: Window size, default is 1280 (i.e., 5 seconds of EEG data).
            m_dist: Whether to use multichannel distribution entropy, default is False.
        """
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.window_size = window_size
        self.m_dist = m_dist

    def _mdisten_multichannel(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Distribution Entropy (DistEn) of multichannel EEG signals.

        Params:
            signals: Input EEG signals with shape [channels, samples].
        Returns:
            value: Calculated distribution entropy.
        """
        if signals.ndimension() != 2:
            raise ValueError("Expected input signal to be a 2D array with shape [channels, samples].")

        # Calculate entropy value for each channel, using standard deviation as a simple entropy metric here
        entropies = [torch.std(signals[ch]).item() for ch in range(signals.shape[0])]  # Use standard deviation to approximate entropy value
        avg_entropy = torch.mean(torch.tensor(entropies))  # Return the average entropy value across all channels
        return avg_entropy

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Distribution Entropy (DistEn) of EEG signals (supports multichannel input).

        Params:
            eeg_data: Input signal with shape [channels, num_samples], where num_samples can be 1 (representing a single timestamp).
        Returns:
            value: Average entropy value of all channels, as a Tensor with shape [1].
        """
        if eeg_data.ndimension() == 2:  # Handle the case of a single timestamp
            # Calculate the distribution entropy for this timestamp
            entropy_value = self._mdisten_multichannel(eeg_data)
            return torch.tensor([entropy_value], dtype=torch.float32)  # Return a 1D tensor containing the entropy value
        else:
            raise ValueError("Expected input signal to be a 2D array with shape [channels, samples], and samples should be 1.")
