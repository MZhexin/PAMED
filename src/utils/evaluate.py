import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
from scipy.signal import find_peaks

def evaluate_va_metrics(pred_va: np.ndarray, true_va: np.ndarray, smoothing_window: int = 5) -> dict:
    """
    Compute a set of advanced evaluation metrics between predicted and ground truth VA sequences.

    Parameters:
        pred_va (np.ndarray): Predicted Valence-Arousal sequence of shape (2, T).
        true_va (np.ndarray): Ground truth Valence-Arousal sequence of shape (2, T).
        smoothing_window (int): Window size for smoothing trend comparison (default: 5).

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    results = {}

    # Compute Mean Squared Error
    results['MSE_Valence'] = np.mean((pred_va[0] - true_va[0]) ** 2)
    results['MSE_Arousal'] = np.mean((pred_va[1] - true_va[1]) ** 2)

    # Compute Pearson correlation
    results['Pearson_Valence'], _ = pearsonr(pred_va[0], true_va[0])
    results['Pearson_Arousal'], _ = pearsonr(pred_va[1], true_va[1])

    # Compute First-Step Error
    results['FirstStepError_Valence'] = (pred_va[0, 0] - true_va[0, 0]) ** 2
    results['FirstStepError_Arousal'] = (pred_va[1, 0] - true_va[1, 0]) ** 2

    # Compute Dynamic Time Warping distance
    results['DTW_Valence'] = dtw.distance(pred_va[0], true_va[0])
    results['DTW_Arousal'] = dtw.distance(pred_va[1], true_va[1])

    # Compute Area Between Curves
    results['AreaBetweenCurves_Valence'] = np.trapz(np.abs(pred_va[0] - true_va[0]))
    results['AreaBetweenCurves_Arousal'] = np.trapz(np.abs(pred_va[1] - true_va[1]))

    # Compute smoothed trend agreement
    def trend(x: np.ndarray) -> np.ndarray:
        return np.sign(np.convolve(np.diff(x), np.ones(smoothing_window) / smoothing_window, mode='valid'))

    trend_pred_v = trend(pred_va[0])
    trend_true_v = trend(true_va[0])
    trend_pred_a = trend(pred_va[1])
    trend_true_a = trend(true_va[1])

    results['TrendAgreement_Valence'] = np.mean(trend_pred_v == trend_true_v)
    results['TrendAgreement_Arousal'] = np.mean(trend_pred_a == trend_true_a)

    # Compute turning point (peak) matching rate
    def turning_match(pred_peaks: np.ndarray, true_peaks: np.ndarray, tol: int = 5) -> float:
        if len(true_peaks) == 0:
            return 1.0
        return np.mean([np.any(np.abs(pred_peaks - tp) <= tol) for tp in true_peaks])

    peaks_pred_v, _ = find_peaks(pred_va[0])
    peaks_true_v, _ = find_peaks(true_va[0])
    peaks_pred_a, _ = find_peaks(pred_va[1])
    peaks_true_a, _ = find_peaks(true_va[1])

    results['TurningPointsMatchRate_Valence'] = turning_match(peaks_pred_v, peaks_true_v)
    results['TurningPointsMatchRate_Arousal'] = turning_match(peaks_pred_a, peaks_true_a)

    return results

