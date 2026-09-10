"""
algorithms/fitting/hdo_engine.py
Wieloprocesowy, uogólniony silnik dekompozycji widmowej i HDO Fit.
Czysta matematyka (NumPy / SciPy) – zero zależności od GUI.
Obsługuje k widm bazowych, m pasm syntetycznych i asymetryczną funkcję kary.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy import integrate
import concurrent.futures
import multiprocessing
import os
from typing import Tuple, List, Dict, Any, Optional


def gxl_peak(wavenb: np.ndarray, amp: float, center: float, shape: float, width: float) -> np.ndarray:
    """
    Pasmo będące iloczynem Gaussa i Lorentza (GxL).
    shape in [0.0, 1.0]: 0.0 -> czysty Gauss, 1.0 -> czysty Lorentz.
    Zabezpieczone numerycznie przed overflow w np.exp.
    """
    diff_sq = ((wavenb - center) / max(width, 1e-4)) ** 2
    exponent = -((1.0 - shape) / 2.0) * diff_sq
    # Zabezpieczenie przed overflow w float64 (exp poniżej -100 to numeryczne 0.0)
    exponent_safe = np.clip(exponent, -100.0, 0.0)
    numerator = np.exp(exponent_safe)
    denominator = 1.0 + shape * diff_sq
    return amp * (numerator / denominator)


def evaluate_total_fit(param: np.ndarray, wavenb: np.ndarray,
                       base_matrix: np.ndarray, n_peaks: int,
                       baseline_active: bool = False) -> np.ndarray:
    """
    Oblicza rekonstruowane widmo:
    fit_y = Baseline(a*x^2 + b*x + c) + sum_i(c_i * W_i) + sum_j(Peak_j)
    """
    k_components = base_matrix.shape[0] if base_matrix.ndim == 2 else 0

    if baseline_active:
        fit_y = param[0] * (wavenb ** 2) + param[1] * wavenb + param[2]
    else:
        fit_y = np.zeros_like(wavenb, dtype=np.float64)

    # Suma ważonych widm bazowych: sum(c_i * W_i)
    if k_components > 0:
        coeffs = param[3: 3 + k_components]
        fit_y += np.dot(coeffs, base_matrix)

    # Suma pików syntetycznych GxL
    start_peaks = 3 + k_components
    for i in range(n_peaks):
        pos = start_peaks + i * 4
        fit_y += gxl_peak(wavenb, param[pos], param[pos + 1], param[pos + 2], param[pos + 3])

    return fit_y


def hdo_residuals(param: np.ndarray, target: np.ndarray, wavenb: np.ndarray,
                  base_matrix: np.ndarray, n_peaks: int,
                  penalty_weight: float, baseline_active: bool) -> np.ndarray:
    """
    Wektorowa funkcja celu z asymetryczną karą za przekraczanie widma badanego.
    diff = fit_y - target; jeśli diff > 0 -> kara = diff * penalty_weight.
    """
    fit_y = evaluate_total_fit(param, wavenb, base_matrix, n_peaks, baseline_active)
    diff = fit_y - target
    penalty = np.where(diff > 0, diff * penalty_weight, 0.0)
    return target - (fit_y + penalty)


def _mc_fit_worker(args) -> Tuple[float, np.ndarray]:
    """Top-level worker dla ProcessPoolExecutor (bezpieczny pod macOS Apple Silicon / spawn)."""
    param_guess, target, wavenb, base_matrix, n_peaks, penalty_weight, baseline_active, bounds = args
    try:
        res = least_squares(
            hdo_residuals, param_guess,
            args=(target, wavenb, base_matrix, n_peaks, penalty_weight, baseline_active),
            bounds=bounds, ftol=1.49e-5, xtol=1.49e-5, max_nfev=250
        )
        fit_y = evaluate_total_fit(res.x, wavenb, base_matrix, n_peaks, baseline_active)
        rss = float(np.sum((target - fit_y) ** 2))
        return rss, res.x
    except Exception:
        return np.inf, param_guess


class HDOFitEngine:
    """Główny silnik Monte Carlo dla dekompozycji k widm bazowych i m pasm syntetycznych."""

    @staticmethod
    def generate_random_guesses(wavenb: np.ndarray, target: np.ndarray,
                                k_components: int, n_peaks: int,
                                n_samples: int,
                                comp_bounds: List[Tuple[float, float]],
                                baseline_active: bool = False) -> np.ndarray:
        total_p = 3 + k_components + 4 * n_peaks
        guesses = np.zeros((n_samples, total_p), dtype=np.float64)

        # Baseline: start wokół 0
        guesses[:, 0:3] = 0.0

        # Losowanie wag dla każdego z k widm bazowych
        for i in range(k_components):
            low, high = comp_bounds[i]
            guesses[:, 3 + i] = np.random.uniform(low, high, n_samples)

        # Losowanie parametrów pików syntetycznych GxL
        max_amp = max(np.max(target), 1e-3)
        wn_min, wn_max = np.min(wavenb), np.max(wavenb)
        start_peaks = 3 + k_components

        for j in range(n_peaks):
            pos = start_peaks + j * 4
            guesses[:, pos] = np.random.uniform(0.05 * max_amp, 0.8 * max_amp, n_samples)
            guesses[:, pos + 1] = np.random.uniform(wn_min, wn_max, n_samples)
            guesses[:, pos + 2] = np.random.uniform(0.01, 0.99, n_samples)
            guesses[:, pos + 3] = np.random.uniform(10.0, 80.0, n_samples)

        for col in range(total_p):
            guesses[:, col] = np.random.permutation(guesses[:, col])

        return guesses

    @classmethod
    def run_monte_carlo(cls, wavenb: np.ndarray, target: np.ndarray,
                        base_spectra_list: List[np.ndarray],
                        comp_bounds: List[Tuple[float, float]],
                        n_peaks: int = 0,
                        n_iterations: int = 2000,
                        penalty_weight: float = 10.0,
                        st_dev: float = 0.01,
                        baseline_active: bool = False,
                        max_workers: Optional[int] = None,
                        progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Główna metoda obliczeniowa dekompozycji.
        Zwraca pełen zestaw parametrów, dopasowane widmo, krzywe składowe,
        czyste residuum nadmiarowe oraz statystyki całkowe.
        """
        k_components = len(base_spectra_list)
        if k_components > 0:
            base_matrix = np.vstack(base_spectra_list)
        else:
            base_matrix = np.empty((0, len(wavenb)))

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 2) - 1)

        guesses = cls.generate_random_guesses(
            wavenb, target, k_components, n_peaks, n_iterations, comp_bounds, baseline_active
        )

        total_p = 3 + k_components + 4 * n_peaks
        lower = np.full(total_p, -np.inf)
        upper = np.full(total_p, np.inf)

        if not baseline_active:
            lower[0:3], upper[0:3] = -1e-9, 1e-9
        else:
            lower[0:3], upper[0:3] = -0.1, 0.1

        for i in range(k_components):
            lower[3 + i], upper[3 + i] = comp_bounds[i][0], comp_bounds[i][1]

        start_peaks = 3 + k_components
        wn_span = np.max(wavenb) - np.min(wavenb)
        for j in range(n_peaks):
            pos = start_peaks + j * 4
            lower[pos], upper[pos] = 0.0, 2.0 * max(np.max(target), 1e-2)
            lower[pos + 1], upper[pos + 1] = np.min(wavenb) - 20.0, np.max(wavenb) + 20.0
            lower[pos + 2], upper[pos + 2] = 0.0, 1.0
            lower[pos + 3], upper[pos + 3] = 2.0, 0.6 * wn_span

        bounds = (lower, upper)
        tasks = [
            (guesses[i], target, wavenb, base_matrix, n_peaks, penalty_weight, baseline_active, bounds)
            for i in range(n_iterations)
        ]

        best_rss = np.inf
        best_params = guesses[0]

        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_mc_fit_worker, t) for t in tasks]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures)):
                rss, params = fut.result()
                if rss < best_rss:
                    best_rss = rss
                    best_params = params

                if progress_callback and (idx % max(1, n_iterations // 50) == 0 or idx == n_iterations - 1):
                    # Callback zwraca True, jeśli użytkownik kliknął Anuluj
                    should_cancel = progress_callback(int((idx + 1) / n_iterations * 100), best_rss)
                    if should_cancel:
                        for f in futures:
                            f.cancel()
                        break

        # 1. Obliczenie krzywej dopasowania (Total Fit)
        fitted_curve = evaluate_total_fit(best_params, wavenb, base_matrix, n_peaks, baseline_active)

        # 2. Składowe ważone: c_i * W_i
        scaled_components = []
        sum_components = np.zeros_like(wavenb)
        extracted_coeffs = []
        for i in range(k_components):
            c_val = float(best_params[3 + i])
            extracted_coeffs.append(c_val)
            comp_curve = c_val * base_matrix[i]
            scaled_components.append(comp_curve)
            sum_components += comp_curve

        # 3. Piki syntetyczne
        peaks_curves = []
        sum_peaks = np.zeros_like(wavenb)
        start_peaks = 3 + k_components
        peaks_params = []
        for j in range(n_peaks):
            pos = start_peaks + j * 4
            p_amp = float(best_params[pos])
            p_center = float(best_params[pos + 1])
            p_shape = float(best_params[pos + 2])
            p_width = float(best_params[pos + 3])
            peaks_params.append({'amp': p_amp, 'center': p_center, 'shape': p_shape, 'width': p_width})

            p_y = gxl_peak(wavenb, p_amp, p_center, p_shape, p_width)
            peaks_curves.append(p_y)
            sum_peaks += p_y

        # 4. Czyste residuum nadmiarowe
        excess_residual = target - sum_components

        # 5. Linia bazowa
        if baseline_active:
            baseline_curve = best_params[0] * (wavenb ** 2) + best_params[1] * wavenb + best_params[2]
        else:
            baseline_curve = np.zeros_like(wavenb)

        # 6. Analiza całkowa (Simpsona)
        comp_areas = [abs(float(integrate.simpson(c_curve, wavenb))) for c_curve in scaled_components]
        peaks_area = abs(float(integrate.simpson(sum_peaks, wavenb))) if n_peaks > 0 else 0.0
        baseline_area = abs(float(integrate.simpson(baseline_curve, wavenb))) if baseline_active else 0.0

        total_area = sum(comp_areas) + peaks_area + baseline_area
        area_percentages = [
            (a / total_area * 100.0) if total_area > 0 else 0.0 for a in comp_areas
        ]

        # 7. Metryki GoF
        ss_res = float(np.sum((target - fitted_curve) ** 2))
        ss_tot = float(np.sum((target - np.mean(target)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((target - fitted_curve) ** 2)))
        dof = max(1, len(wavenb) - total_p - 1)
        chi2_red = (ss_res / max(st_dev ** 2, 1e-8)) / dof

        return {
            'best_params': best_params,
            'coefficients': extracted_coeffs,
            'comp_areas': comp_areas,
            'area_percentages': area_percentages,
            'peaks_params': peaks_params,
            'peaks_area': peaks_area,
            'baseline_area': baseline_area,
            'fitted_curve': fitted_curve,
            'scaled_components': scaled_components,
            'sum_peaks': sum_peaks,
            'peaks_curves': peaks_curves,
            'baseline_curve': baseline_curve,
            'excess_residual': excess_residual,
            'rss': ss_res,
            'r2': r2,
            'rmse': rmse,
            'chi2_red': chi2_red
        }