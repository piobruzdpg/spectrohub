"""
algorithms/atmospheric.py
Silnik numeryczny kompensacji pasm atmosferycznych (para wodna i CO2) metodą VaporFit.
Analityczne OLS na składowych wysokoczęstotliwościowych + Auto-Walidator znaku SDV.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger("SpectroHub")


class AtmosphericCorrectionEngine:
    """Silnik dopasowywania i kompensacji widm atmosferycznych metodą minimalizacji residuum."""

    @staticmethod
    def calculate_ssi(spectrum: np.ndarray) -> float:
        """Spectral Smoothness Index (SSI). Mniejsza wartość = gładsze widmo."""
        diff_sq = np.sum(np.diff(spectrum) ** 2)
        energy = np.sum(spectrum ** 2)
        return float(diff_sq / energy) if energy > 0 else 0.0

    @staticmethod
    def calculate_sdv(wavenumbers: np.ndarray, spectrum: np.ndarray) -> float:
        """Smoothness Derivative Variance (SDV). Wariancja 2. pochodnej."""
        d2y = np.gradient(np.gradient(spectrum, wavenumbers), wavenumbers)
        return float(np.var(d2y))

    @staticmethod
    def calculate_std_sg(spectrum: np.ndarray, window_length: int = 11, polyorder: int = 3) -> float:
        """Odchylenie standardowe szumu po odjęciu sygnału wygładzonego SG."""
        w = window_length if window_length % 2 != 0 else window_length + 1
        if len(spectrum) <= w:
            return 0.0
        smoothed = savgol_filter(spectrum, window_length=w, polyorder=polyorder)
        return float(np.std(spectrum - smoothed))

    @classmethod
    def fit_and_subtract(
            cls,
            wavenumbers: np.ndarray,
            sample_spectrum: np.ndarray,
            atm_spectra: np.ndarray,
            sg_points: int = 11,
            sg_poly: int = 3,
            sign_mode: Optional[str] = None  # zachowane dla wstecznej kompatybilności API
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Analityczne dopasowanie wag widm atmosfery za pomocą liniowej regresji składowych ostrych (High-Pass)
        oraz bezpośrednia kompensacja S_corr = S - sum(c_i * A_i).
        """
        wavenb = np.asarray(wavenumbers, dtype=np.float64)
        sample = np.asarray(sample_spectrum, dtype=np.float64)
        atm = np.asarray(atm_spectra, dtype=np.float64)

        if atm.ndim == 1:
            atm = atm[:, np.newaxis]

        if atm.shape[0] != len(sample):
            raise ValueError(
                f"Niezgodność wymiarów: próbka ma {len(sample)} pkt, a widma atmosfery {atm.shape[0]} pkt."
            )

        w = sg_points if sg_points % 2 != 0 else sg_points + 1
        poly = sg_poly if w > sg_poly else max(1, w - 2)

        # 1. Wyodrębnienie ostrych struktur (High-Pass: Sygnał surowy minus SG)
        y_sharp = sample - savgol_filter(sample, window_length=w, polyorder=poly)

        X_sharp = np.empty_like(atm)
        for col in range(atm.shape[1]):
            X_sharp[:, col] = atm[:, col] - savgol_filter(atm[:, col], window_length=w, polyorder=poly)

        # 2. Analityczne rozwiązanie OLS: minimalizacja ||X_sharp * c - y_sharp||^2
        # Dokładny odpowiednik dopasowania z oryginalnego VaporFit
        coeffs, _, _, _ = np.linalg.lstsq(X_sharp, y_sharp, rcond=None)

        # 3. Bezpośrednia kompensacja widma: S_corr = S - (A @ c)
        # Znak c_i wynika wprost z algebry liniowej:
        # - nadmiar pasm w próbce -> c > 0 -> piki są odejmowane
        # - deficyt pasm (dołki) w próbce -> c < 0 -> dołki są kompensowane (odejmowanie wartości ujemnej = dodawanie)
        atm_total = atm @ coeffs
        corrected_spectrum = sample - atm_total

        metrics = {
            'ssi_raw': cls.calculate_ssi(sample),
            'ssi_corr': cls.calculate_ssi(corrected_spectrum),
            'sdv_raw': cls.calculate_sdv(wavenb, sample),
            'sdv_corr': cls.calculate_sdv(wavenb, corrected_spectrum),
            'std_raw': cls.calculate_std_sg(sample, w, poly),
            'std_corr': cls.calculate_std_sg(corrected_spectrum, w, poly)
        }

        return corrected_spectrum, coeffs, metrics

    @classmethod
    def find_optimal_sg_window(
        cls,
        wavenumbers: np.ndarray,
        sample_spectrum: np.ndarray,
        atm_spectra: np.ndarray,
        sg_poly: int = 3,
        window_candidates: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Przeszukuje okna Savitzky-Golay w poszukiwaniu minimum wskaźników gładkości."""
        if window_candidates is None:
            window_candidates = [7, 9, 11, 13, 15, 17, 19, 21, 25, 29]

        min_valid = sg_poly + 2
        valid_windows = [w for w in window_candidates if w >= min_valid and w % 2 != 0]

        ssi_list, sdv_list, std_list = [], [], []

        for win in valid_windows:
            _, _, metrics = cls.fit_and_subtract(
                wavenumbers, sample_spectrum, atm_spectra,
                sg_points=win, sg_poly=sg_poly
            )
            ssi_list.append(metrics['ssi_corr'])
            sdv_list.append(metrics['sdv_corr'])
            std_list.append(metrics['std_corr'])

        return {
            'windows': valid_windows,
            'best_window_sdv': valid_windows[int(np.argmin(sdv_list))],
            'best_window_ssi': valid_windows[int(np.argmin(ssi_list))]
        }