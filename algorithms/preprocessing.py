"""
algorithms/preprocessing.py
Zunifikowane algorytmy przetwarzania sygnału widmowego (1D oraz 2D).
Brak zależności od Qt - czysty NumPy / SciPy.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional


# --- 1. LINIE BAZOWE ---

def baseline_asls(y: np.ndarray, lam: float = 1e5, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    """
    Korekcja linii bazowej AsLS (Eilers & Boelens) na macierzach rzadkich.
    Działa na pojedynczym wektorze 1D float64.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    L = len(y_arr)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format='csc', dtype=np.float64)
    D_sq = D.T.dot(D)

    w = np.ones(L, dtype=np.float64)
    baseline = np.zeros(L, dtype=np.float64)

    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L, format='csc')
        Z = W + lam * D_sq
        try:
            baseline = spsolve(Z, w * y_arr)
        except Exception:
            break
        w = p * (y_arr > baseline) + (1.0 - p) * (y_arr < baseline)

    return baseline


def baseline_linear(x: np.ndarray, y: np.ndarray, x1: float, x2: float) -> np.ndarray:
    """Odejmuje linię bazową rozpiętą między punktami x1 i x2."""
    idx1 = np.abs(x - x1).argmin()
    idx2 = np.abs(x - x2).argmin()

    denom = x[idx2] - x[idx1]
    if abs(denom) < 1e-12:
        return np.zeros_like(y)

    slope = (y[idx2] - y[idx1]) / denom
    intercept = y[idx1] - slope * x[idx1]
    return slope * x + intercept


# --- 2. FILTRY, GŁADZENIE I POCHODNE ---

def apply_savgol(y: np.ndarray, window: int = 11, poly: int = 3, deriv: int = 0) -> np.ndarray:
    """
    Filtr Savitzky-Golay z automatyczną walidacją poprawności okna.
    Obsługuje wektory 1D oraz kolumny macierzy 2D.
    """
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 2 if (poly + 2) % 2 != 0 else poly + 3

    if y.ndim == 1:
        return savgol_filter(y, window_length=window, polyorder=poly, deriv=deriv)
    elif y.ndim == 2:
        return savgol_filter(y, window_length=window, polyorder=poly, deriv=deriv, axis=0)
    raise ValueError("Obsługiwane są wyłącznie tablice 1D oraz 2D.")


def get_first_derivative(x: np.ndarray, y: np.ndarray, window: int = 11, poly: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Pierwsza pochodna sygnału (d1y/dx1) za pomocą Savitzky-Golay."""
    deriv1 = apply_savgol(y, window=window, poly=poly, deriv=1)
    return x, deriv1


def get_second_derivative_inverted(x: np.ndarray, y: np.ndarray, window: int = 11, poly: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Druga pochodna z odwróconym znakiem (-1 * d2y/dx2).
    Minima pasm FTIR stają się dodatnimi maksimami ułatwiającymi lokalizację pasm ukrytych.
    """
    deriv2 = apply_savgol(y, window=window, poly=poly, deriv=2)
    return x, -1.0 * deriv2


# --- 3. INTERPOLACJA I PRZYCINANIE ---

def apply_cubic_spline_densify(x: np.ndarray, y: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Zagęszcza siatkę punktów za pomocą splajnu sześciennego, zachowując kierunek osi X."""
    num_pts = len(x)
    new_num_pts = int(num_pts * factor) - (factor - 1)

    is_inverted = (x[0] > x[-1])
    x_sort = x[::-1] if is_inverted else x
    y_sort = y[::-1] if is_inverted else y

    cs = CubicSpline(x_sort, y_sort)
    new_x = np.linspace(x_sort[0], x_sort[-1], new_num_pts)
    new_y = cs(new_x)

    if is_inverted:
        return new_x[::-1], new_y[::-1]
    return new_x, new_y


def crop_range(x: np.ndarray, y: np.ndarray, min_wn: float, max_wn: float) -> Tuple[np.ndarray, np.ndarray]:
    """Wycina podany zakres liczb falowych."""
    mask = (x >= min(min_wn, max_wn)) & (x <= max(min_wn, max_wn))
    if not np.any(mask):
        raise ValueError(f"Zakres [{min_wn}, {max_wn}] nie zawiera żadnych punktów widmowych.")

    if y.ndim == 1:
        return x[mask], y[mask]
    elif y.ndim == 2:
        return x[mask], y[mask, :]
    raise ValueError("Nieobsługiwany wymiar macierzy.")


# --- 4. TRANSFORMACJE SKALI I NORMALIZACJA ---

def apply_offset(y: np.ndarray, x: np.ndarray, target_wn: float, target_val: float = 0.0) -> np.ndarray:
    """Przesuwa widmo tak, aby w punkcie target_wn miało wartość target_val."""
    idx = np.abs(x - target_wn).argmin()
    if y.ndim == 1:
        shift = target_val - y[idx]
        return y + shift
    elif y.ndim == 2:
        shift = target_val - y[idx, :]
        return y + shift
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_snv(y: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (centrowanie do średniej i skalowanie do odchylenia standardowego)."""
    if y.ndim == 1:
        std = np.std(y)
        return (y - np.mean(y)) / std if std > 1e-8 else y - np.mean(y)
    elif y.ndim == 2:
        mean = np.mean(y, axis=0, keepdims=True)
        std = np.std(y, axis=0, keepdims=True)
        std[std <= 1e-8] = 1.0
        return (y - mean) / std
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_min_max(y: np.ndarray) -> np.ndarray:
    """Skalowanie do zakresu 0.0 - 1.0."""
    if y.ndim == 1:
        mn, mx = np.min(y), np.max(y)
        return (y - mn) / (mx - mn) if (mx - mn) > 1e-8 else np.zeros_like(y)
    elif y.ndim == 2:
        mn = np.min(y, axis=0, keepdims=True)
        mx = np.max(y, axis=0, keepdims=True)
        diff = mx - mn
        diff[diff <= 1e-8] = 1.0
        return (y - mn) / diff
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_vector_norm_l2(y: np.ndarray) -> np.ndarray:
    """Normalizacja wektorowa Euklidesa (L2)."""
    if y.ndim == 1:
        norm = np.linalg.norm(y)
        return y / norm if norm > 1e-8 else y
    elif y.ndim == 2:
        norms = np.linalg.norm(y, axis=0, keepdims=True)
        norms[norms <= 1e-8] = 1.0
        return y / norms
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_msc(matrix_2d: np.ndarray) -> np.ndarray:
    """Multiplicative Scatter Correction względem widma średniego serii."""
    if matrix_2d.ndim != 2:
        raise ValueError("MSC wymaga macierzy dwuwymiarowej (serii widm).")

    mean_spectrum = np.mean(matrix_2d, axis=1, keepdims=True)
    corrected = np.empty_like(matrix_2d)
    model = LinearRegression()

    for col in range(matrix_2d.shape[1]):
        y = matrix_2d[:, col]
        model.fit(mean_spectrum, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        if abs(slope) > 1e-8:
            corrected[:, col] = (y - intercept) / slope
        else:
            corrected[:, col] = y - intercept

    return corrected


# --- 5. ALIASY ZGODNOŚCI WSTECZNEJ ---
min_max_scale = apply_min_max
snv = apply_snv
smooth_savitzky_golay = apply_savgol
vector_norm_l2 = apply_vector_norm_l2