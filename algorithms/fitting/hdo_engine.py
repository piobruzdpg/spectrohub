"""
algorithms/fitting/hdo_engine.py
Wieloprocesowy silnik dopasowywania widm składowych i izolacji residuum (HDO Fit).
Przystosowany do puli ProcessPoolExecutor / concurrent.futures.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy import integrate
from concurrent.futures import ProcessPoolExecutor
import os
from typing import Tuple, List, Dict, Any


def gxl_peak(wavenb: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Profil pasma syntetycznego: iloczyn Gaussa i Lorentza (GxL)."""
    diff_sq = ((wavenb - b) / d) ** 2
    numerator = np.exp(-((1.0 - c) / 2.0) * diff_sq)
    denominator = 1.0 + c * diff_sq
    return a * (numerator / denominator)


def evaluate_total_fit(param: np.ndarray, wavenb: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Oblicza sumę: Linia bazowa (kwadratowa) + c_A*W1 + c_B*W2 + pasma GxL."""
    baseline = param[0] * (wavenb ** 2) + param[1] * wavenb + param[2]
    fit_y = baseline + param[3] * w1 + param[4] * w2

    n_extra_peaks = (len(param) - 5) // 4
    for i in range(n_extra_peaks):
        pos = 5 + i * 4
        fit_y += gxl_peak(wavenb, param[pos], param[pos + 1], param[pos + 2], param[pos + 3])

    return fit_y


def hdo_residuals(param: np.ndarray, target: np.ndarray, wavenb: np.ndarray,
                  w1: np.ndarray, w2: np.ndarray, penalty_weight: float) -> np.ndarray:
    """Funkcja celu z asymetryczną karą za przekraczanie eksperymentalnego widma."""
    fit_y = evaluate_total_fit(param, wavenb, w1, w2)
    diff = fit_y - target
    # Wektorowa kara
    penalty = np.where(diff > 0, diff * penalty_weight, 0.0)
    return target - (fit_y + penalty)


def fit_single_initial_guess(args) -> Tuple[float, np.ndarray]:
    """Pojedyncza optymalizacja z losowego punktu startowego (Worker wieloprocesowy)."""
    param_guess, target, wavenb, w1, w2, penalty_weight, bounds = args
    try:
        res = least_squares(
            hdo_residuals, param_guess,
            args=(target, wavenb, w1, w2, penalty_weight),
            bounds=bounds, ftol=1e-5, xtol=1e-5
        )
        fit_y = evaluate_total_fit(res.x, wavenb, w1, w2)
        rss = float(np.sum((target - fit_y) ** 2))
        return rss, res.x
    except Exception:
        return np.inf, param_guess


class HDOFitEngine:
    """Orkiestrator procedury Monte Carlo dla dekompozycji HDO / 2 widm."""

    @staticmethod
    def generate_random_guesses(wavenb: np.ndarray, target: np.ndarray, n_peaks: int,
                                n_samples: int = 2000, bounds_w1=(0.0, 1.0), bounds_w2=(0.0, 1.0)) -> np.ndarray:
        """Generuje losowe wektory startowe w przestrzeni parametrów."""
        total_params = 5 + 4 * n_peaks
        guesses = np.zeros((n_samples, total_params), dtype=np.float64)

        # Baseline start: 0
        guesses[:, 0:3] = 0.0
        # Udziały W1 i W2
        guesses[:, 3] = np.random.uniform(bounds_w1[0], bounds_w1[1], n_samples)
        guesses[:, 4] = np.random.uniform(bounds_w2[0], bounds_w2[1], n_samples)

        max_amp = np.max(target)
        wn_min, wn_max = np.min(wavenb), np.max(wavenb)

        for i in range(n_peaks):
            pos = 5 + i * 4
            guesses[:, pos] = np.random.uniform(0.1 * max_amp, 0.8 * max_amp, n_samples)  # Amplituda
            guesses[:, pos + 1] = np.random.uniform(wn_min, wn_max, n_samples)             # Pozycja
            guesses[:, pos + 2] = np.random.uniform(0.01, 0.99, n_samples)                 # Kształt GxL
            guesses[:, pos + 3] = np.random.uniform(10.0, 80.0, n_samples)                 # Szerokość

        return guesses

    @classmethod
    def run_monte_carlo(cls, wavenb: np.ndarray, target: np.ndarray, w1: np.ndarray, w2: np.ndarray,
                        n_peaks: int = 0, n_iterations: int = 1000, penalty_weight: float = 10.0,
                        max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Równoległe przeszukiwanie przestrzeni rozwiązań na wielu rdzeniach CPU."""
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 2) - 1)

        guesses = cls.generate_random_guesses(wavenb, target, n_peaks, n_samples=n_iterations)

        # Ustalenie granic (Bounds)
        total_p = 5 + 4 * n_peaks
        lower = np.full(total_p, -np.inf)
        upper = np.full(total_p, np.inf)

        lower[0:3], upper[0:3] = -1e-8, 1e-8  # Blokada linii bazowej wokół 0
        lower[3], upper[3] = 0.0, 1.2          # W1
        lower[4], upper[4] = 0.0, 1.2          # W2

        if n_peaks > 0:
            wn_span = np.max(wavenb) - np.min(wavenb)
            for i in range(n_peaks):
                pos = 5 + i * 4
                lower[pos], upper[pos] = 0.0, 2.0 * np.max(target)
                lower[pos + 1], upper[pos + 1] = np.min(wavenb) - 50.0, np.max(wavenb) + 50.0
                lower[pos + 2], upper[pos + 2] = 0.0, 1.0
                lower[pos + 3], upper[pos + 3] = 2.0, 0.5 * wn_span

        bounds = (lower, upper)
        tasks = [(guesses[i], target, wavenb, w1, w2, penalty_weight, bounds) for i in range(n_iterations)]

        best_rss = np.inf
        best_params = guesses[0]

        # Wykonanie równoległe w osobnej puli procesów (brak GIL)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(fit_single_initial_guess, tasks, chunksize=max(1, n_iterations // (max_workers * 4)))
            for rss, params in results:
                if rss < best_rss:
                    best_rss = rss
                    best_params = params

        # Wyznaczenie czystego residuum: Residuum = Target - (c_A*W1 + c_B*W2)
        fitted_curve = evaluate_total_fit(best_params, wavenb, w1, w2)
        pure_excess_residual = target - (best_params[3] * w1 + best_params[4] * w2)

        return {
            'best_params': best_params,
            'c_A': float(best_params[3]),
            'c_B': float(best_params[4]),
            'rss': best_rss,
            'fitted_curve': fitted_curve,
            'excess_residual': pure_excess_residual
        }