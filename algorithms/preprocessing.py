"""
algorithms/preprocessing.py
Zunifikowane algorytmy przetwarzania sygnału widmowego (1D, 2D oraz tensory 3D).
Czysty rdzeń numeryczny (NumPy / SciPy / scikit-learn) — zero kodu GUI.
Przygotowany pod architekturę potoków i węzłów (Node-based pipeline).
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any


# =====================================================================
# 1. LINIE BAZOWE (AsLS, airPLS, Liniowa)
# =====================================================================

def baseline_asls(y: np.ndarray, lam: float = 1e5, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    """
    Korekcja linii bazowej AsLS (Eilers & Boelens) na macierzach rzadkich.
    Działa na wektorze 1D float64.
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


def baseline_airpls(y: np.ndarray, lam: float = 1e5, n_iter: int = 15, tol: float = 1e-4) -> np.ndarray:
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).
    Nie wymaga deklarowania parametru asymetrii p — wagi są dobierane adaptacyjnie.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    L = len(y_arr)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format='csc', dtype=np.float64)
    D_sq = D.T.dot(D)

    w = np.ones(L, dtype=np.float64)
    baseline = np.zeros(L, dtype=np.float64)

    for i in range(1, n_iter + 1):
        W = sparse.spdiags(w, 0, L, L, format='csc')
        Z = W + lam * D_sq
        try:
            baseline = spsolve(Z, w * y_arr)
        except Exception:
            break

        d = y_arr - baseline
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break

        sum_neg = np.sum(np.abs(d_neg))
        if sum_neg < 1e-10:
            break

        # Adaptacyjne wagi wykładnicze
        w_new = np.zeros(L, dtype=np.float64)
        mask_neg = (d < 0)
        w_new[mask_neg] = np.exp(i * np.abs(d[mask_neg]) / sum_neg)
        w_new[~mask_neg] = 0.0

        if np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-10) < tol:
            w = w_new
            break
        w = w_new

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


# =====================================================================
# 2. FOURIER SELF-DECONVOLUTION (FSD)
# =====================================================================

def fourier_self_deconvolution(wavenumbers: np.ndarray, y: np.ndarray,
                               fwhm: float = 18.0, k_factor: float = 2.0) -> np.ndarray:
    """
    Fourier Self-Deconvolution (FSD) z apodyzacją trójkątną (standard OMNIC / Kauppinen).
    Rozdziela nakładające się pasma (np. struktury w Amidzie I).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    N = len(y_arr)
    dw = np.abs(wavenumbers[1] - wavenumbers[0])
    if dw < 1e-8:
        return y_arr

    x = np.fft.fftfreq(N, d=dw)
    A_fourier = fft(y_arr)

    # Odwrotny filtr Lorentza
    deconv_filter = np.exp(np.pi * fwhm * np.abs(x))

    # Apodyzacja trójkątna (relacja Kauppinena)
    L = 0.88589 * k_factor / fwhm
    apod_filter = np.clip(1.0 - np.abs(x) / L, 0.0, 1.0)

    total_filter = deconv_filter * apod_filter
    deconvolved = np.real(ifft(A_fourier * total_filter))
    return deconvolved


# =====================================================================
# 3. FILTRY, GŁADZENIE I POCHODNE (Savitzky-Golay)
# =====================================================================

def apply_savgol(y: np.ndarray, window: int = 11, poly: int = 3, deriv: int = 0) -> np.ndarray:
    """
    Filtr Savitzky-Golay z automatyczną walidacją nieparzystości okna.
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
    """Pierwsza pochodna sygnału (d1y/dx1)."""
    deriv1 = apply_savgol(y, window=window, poly=poly, deriv=1)
    return x, deriv1


def get_second_derivative_inverted(x: np.ndarray, y: np.ndarray, window: int = 11, poly: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Druga pochodna z odwróconym znakiem (-d2y/dx2). Minima FTIR stają się maksimami."""
    deriv2 = apply_savgol(y, window=window, poly=poly, deriv=2)
    return x, -1.0 * deriv2


# =====================================================================
# 4. INTERPOLACJA, PRZYCINANIE I PUNKT KOTWICZENIA (OFFSET)
# =====================================================================

def apply_cubic_spline_densify(x: np.ndarray, y: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Zagęszcza siatkę punktów za pomocą splajnu sześciennego."""
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
    """Wycina zadany zakres liczb falowych."""
    mask = (x >= min(min_wn, max_wn)) & (x <= max(min_wn, max_wn))
    if not np.any(mask):
        raise ValueError(f"Zakres [{min_wn}, {max_wn}] nie zawiera żadnych punktów widmowych.")

    if y.ndim == 1:
        return x[mask], y[mask]
    elif y.ndim == 2:
        return x[mask], y[mask, :]
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_offset(y: np.ndarray, x: np.ndarray, target_wn: float, target_val: float = 0.0) -> np.ndarray:
    """Kotwiczy widmo: przesuwa absorbancję tak, aby w punkcie target_wn miało wartość target_val."""
    idx = np.abs(x - target_wn).argmin()
    if y.ndim == 1:
        return y + (target_val - y[idx])
    elif y.ndim == 2:
        return y + (target_val - y[idx, :])
    raise ValueError("Nieobsługiwany wymiar macierzy.")


# =====================================================================
# 5. NORMALIZACJE I SKALOWANIA CHEMOMETRYCZNE
# =====================================================================

def apply_snv(y: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (centrowanie do średniej i skalowanie do std=1)."""
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
    """Skalowanie do zakresu 0.0 – 1.0."""
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


def apply_norm_l1(y: np.ndarray) -> np.ndarray:
    """Normalizacja polowa L1 (suma modułów / pole powierzchni pod widmem = 1.0)."""
    if y.ndim == 1:
        area = np.sum(np.abs(y))
        return y / area if area > 1e-8 else y
    elif y.ndim == 2:
        areas = np.sum(np.abs(y), axis=0, keepdims=True)
        areas[areas <= 1e-8] = 1.0
        return y / areas
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_vector_norm_l2(y: np.ndarray) -> np.ndarray:
    """Normalizacja wektorowa Euklidesa (L2 = 1.0)."""
    if y.ndim == 1:
        norm = np.linalg.norm(y)
        return y / norm if norm > 1e-8 else y
    elif y.ndim == 2:
        norms = np.linalg.norm(y, axis=0, keepdims=True)
        norms[norms <= 1e-8] = 1.0
        return y / norms
    raise ValueError("Nieobsługiwany wymiar macierzy.")


def apply_mean_center(matrix_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centrowanie macierzy widm do średniej (wzdłuż próbek)."""
    mean_spec = np.mean(matrix_2d, axis=1, keepdims=True)
    return matrix_2d - mean_spec, mean_spec.flatten()


def apply_autoscaling(matrix_2d: np.ndarray) -> np.ndarray:
    """Autoskalowanie (Unit Variance scaling: centrowanie i dzielenie przez std każdej zmiennej)."""
    mean_spec = np.mean(matrix_2d, axis=1, keepdims=True)
    std_spec = np.std(matrix_2d, axis=1, keepdims=True)
    std_spec[std_spec <= 1e-8] = 1.0
    return (matrix_2d - mean_spec) / std_spec


def apply_pareto_scaling(matrix_2d: np.ndarray) -> np.ndarray:
    """Skalowanie Pareto (centrowanie i dzielenie przez sqrt(std) każdej zmiennej)."""
    mean_spec = np.mean(matrix_2d, axis=1, keepdims=True)
    std_spec = np.std(matrix_2d, axis=1, keepdims=True)
    sqrt_std = np.sqrt(std_spec)
    sqrt_std[sqrt_std <= 1e-8] = 1.0
    return (matrix_2d - mean_spec) / sqrt_std


def apply_msc(matrix_2d: np.ndarray, ref_spectrum: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Multiplicative Scatter Correction (MSC).
    Jeśli ref_spectrum=None, jako referencję przyjmuje widmo średnie serii.
    """
    if matrix_2d.ndim != 2:
        raise ValueError("MSC wymaga macierzy dwuwymiarowej (serii widm).")

    if ref_spectrum is None:
        ref_spectrum = np.mean(matrix_2d, axis=1, keepdims=True)
    else:
        ref_spectrum = ref_spectrum.reshape(-1, 1)

    corrected = np.empty_like(matrix_2d)
    model = LinearRegression()

    for col in range(matrix_2d.shape[1]):
        y = matrix_2d[:, col]
        model.fit(ref_spectrum, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        if abs(slope) > 1e-8:
            corrected[:, col] = (y - intercept) / slope
        else:
            corrected[:, col] = y - intercept

    return corrected


# =====================================================================
# 6. METRYKI GŁADKOŚCI (z VaporFit)
# =====================================================================

def calculate_spectral_smoothness(spectrum: np.ndarray) -> float:
    """Spectral Smoothness Index (SSI). Mniejsza wartość oznacza gładsze widmo."""
    diff_sq = np.sum(np.diff(spectrum) ** 2)
    energy = np.sum(spectrum ** 2)
    return float(diff_sq / energy) if energy > 1e-12 else 0.0


def calculate_smoothness_variance(wavenumbers: np.ndarray, spectrum: np.ndarray) -> float:
    """Wariancja drugiej pochodnej (SDV / Curvature smoothness). Mniejsza wartość = brak ostrych ząbków."""
    d2 = np.gradient(np.gradient(spectrum, wavenumbers), wavenumbers)
    return float(np.var(d2))


def calculate_std_after_smoothing(spectrum: np.ndarray, window: int = 11, poly: int = 3) -> float:
    """Odchylenie standardowe szumu po odjęciu wygładzenia Savitzky-Golay."""
    smoothed = savgol_filter(spectrum, window_length=window, polyorder=poly)
    return float(np.std(spectrum - smoothed))


# =====================================================================
# 7. METADANE DLA PIPELINE & PRZYSZŁEGO SILNIKA WĘZŁOWEGO (Node System)
# =====================================================================

PREPROCESSING_REGISTRY = {
    "crop": {
        "label": "Przytnij zakres (Crop)",
        "params": {"min_wn": {"type": "float", "default": 900.0}, "max_wn": {"type": "float", "default": 1800.0}}
    },
    "asls": {
        "label": "Linia bazowa AsLS",
        "params": {"lam": {"type": "float", "default": 1e5}, "p": {"type": "float", "default": 0.001}, "n_iter": {"type": "int", "default": 10}}
    },
    "airpls": {
        "label": "Linia bazowa airPLS",
        "params": {"lam": {"type": "float", "default": 1e5}, "n_iter": {"type": "int", "default": 15}}
    },
    "fsd": {
        "label": "Dekonwolucja Fouriera (FSD)",
        "params": {"fwhm": {"type": "float", "default": 18.0}, "k_factor": {"type": "float", "default": 2.0}}
    },
    "savgol": {
        "label": "Savitzky-Golay (Gładzenie)",
        "params": {"window": {"type": "int", "default": 11}, "poly": {"type": "int", "default": 3}, "deriv": {"type": "int", "default": 0}}
    },
    "snv": {"label": "Normalizacja SNV", "params": {}},
    "min_max": {"label": "Skalowanie Min-Max (0-1)", "params": {}},
    "norm_l1": {"label": "Normalizacja polowa (L1)", "params": {}},
    "norm_l2": {"label": "Normalizacja wektorowa (L2)", "params": {}},
    "offset": {
        "label": "Kotwiczenie (Offset)",
        "params": {"target_wn": {"type": "float", "default": 2000.0}, "target_val": {"type": "float", "default": 0.0}}
    },
    "msc": {"label": "Korekcja rozproszenia (MSC)", "params": {}}
}

# Aliasy zgodności wstecznej
min_max_scale = apply_min_max
snv = apply_snv
smooth_savitzky_golay = apply_savgol
vector_norm_l2 = apply_vector_norm_l2