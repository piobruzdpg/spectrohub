"""
algorithms/statistics.py
Czysty silnik matematyczny statystyki spektralnej, szacowania szumu (ASTM) i replikatów.
NumPy / SciPy — zero kodu GUI.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np


def estimate_noise_metrics(
    x: np.ndarray,
    y: np.ndarray,
    nu_min: float,
    nu_max: float,
    detrend: bool = True
) -> Dict[str, float]:
    """
    Szacuje poziom szumu w wybranym oknie [nu_min, nu_max].
    Zgodnie ze standardem ASTM E1421: opcjonalne odjęcie trendu liniowego
    zapobiega zawyżaniu szumu przez nachylenie linii bazowej.
    """
    low, high = min(nu_min, nu_max), max(nu_min, nu_max)
    mask = (x >= low) & (x <= high)

    if np.count_nonzero(mask) < 4:
        return {
            "rms_noise": 0.0,
            "peak_to_peak": 0.0,
            "mean_level": 0.0,
            "std_raw": 0.0,
            "n_points": float(np.count_nonzero(mask))
        }

    x_win = x[mask]
    y_win = y[mask]
    n_pts = len(y_win)

    mean_lvl = float(np.mean(y_win))
    std_raw = float(np.std(y_win, ddof=1))
    ptp = float(np.ptp(y_win))

    if detrend:
        # Dopasowanie linii prostej y = ax + b (usunięcie nachylenia bazy)
        poly = np.polyfit(x_win, y_win, deg=1)
        fitted_line = np.polyval(poly, x_win)
        residuals = y_win - fitted_line
        dof = max(1, n_pts - 2)
        rms = float(np.sqrt(np.sum(residuals ** 2) / dof))
    else:
        rms = std_raw

    return {
        "rms_noise": rms,
        "peak_to_peak": ptp,
        "mean_level": mean_lvl,
        "std_raw": std_raw,
        "n_points": float(n_pts)
    }


def calculate_snr(
    x: np.ndarray,
    y: np.ndarray,
    signal_wn_min: float,
    signal_wn_max: float,
    noise_wn_min: float,
    noise_wn_max: float,
    detrend_noise: bool = True
) -> Dict[str, float]:
    """
    Wyznacza stosunek sygnału do szumu (SNR):
    Maksymalna absorbancja w oknie sygnału podzielona przez RMS szumu.
    """
    noise_res = estimate_noise_metrics(x, y, noise_wn_min, noise_wn_max, detrend=detrend_noise)
    rms = noise_res["rms_noise"]

    s_low, s_high = min(signal_wn_min, signal_wn_max), max(signal_wn_min, signal_wn_max)
    s_mask = (x >= s_low) & (x <= s_high)

    if not np.any(s_mask) or rms < 1e-12:
        return {
            "snr": 0.0,
            "peak_signal": 0.0,
            "rms_noise": rms
        }

    peak_sig = float(np.max(y[s_mask]))
    snr_val = peak_sig / rms

    return {
        "snr": snr_val,
        "peak_signal": peak_sig,
        "rms_noise": rms
    }


def aggregate_spectral_ensemble(
    matrices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Dokonuje agregacji zespołu widm (macierz W x N):
    Zwraca: mean, median, std, sem (błąd standardowy), min, max.
    """
    if matrices.ndim != 2 or matrices.shape[1] == 0:
        raise ValueError("Wymagana macierz 2D (Wavenumbers x Spectra).")

    n_samples = matrices.shape[1]
    mean_y = np.mean(matrices, axis=1)
    median_y = np.median(matrices, axis=1)

    if n_samples > 1:
        std_y = np.std(matrices, axis=1, ddof=1)
        sem_y = std_y / np.sqrt(n_samples)
    else:
        std_y = np.zeros_like(mean_y)
        sem_y = np.zeros_like(mean_y)

    min_y = np.min(matrices, axis=1)
    max_y = np.max(matrices, axis=1)

    return {
        "mean": mean_y,
        "median": median_y,
        "std": std_y,
        "sem": sem_y,
        "min": min_y,
        "max": max_y
    }


def compute_replicate_correlations(matrices: np.ndarray) -> np.ndarray:
    """
    Oblicza symetryczną macierz korelacji Pearsona (N x N) między replikatami.
    """
    if matrices.shape[1] < 2:
        return np.ones((matrices.shape[1], matrices.shape[1]), dtype=np.float64)

    return np.corrcoef(matrices, rowvar=False)