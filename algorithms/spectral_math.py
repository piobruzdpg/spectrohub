"""
algorithms/spectral_math.py
Zunifikowany silnik arytmetyki i transformacji fotometrycznych widm FTIR.
Czysty rdzeń numeryczny (NumPy / SciPy) — zero kodu GUI.
"""

import numpy as np
from typing import Tuple, Optional


def align_to_reference(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """
    Interpoluje wektor y_src do siatki liczb falowych x_ref.
    Odporny na malejący lub rosnący kierunek osi X.
    """
    if len(x_src) == len(x_ref) and np.array_equal(x_src, x_ref):
        return y_src.copy()

    if len(x_src) > 1 and x_src[0] > x_src[-1]:
        xp, fp = x_src[::-1], y_src[::-1]
    else:
        xp, fp = x_src, y_src

    return np.interp(x_ref, xp, fp)


def subtract_spectra(y1: np.ndarray, y2: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Różnica widm: S = S1 - k * S2."""
    return y1 - (k * y2)


def add_spectra(y1: np.ndarray, y2: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Suma widm: S = S1 + k * S2."""
    return y1 + (k * y2)


def multiply_spectra(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Mnożenie widm punkt po punkcie: S = S1 * S2."""
    return y1 * y2


def divide_spectra(y1: np.ndarray, y2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Dzielenie widm punkt po punkcie: S = S1 / S2.
    Zabezpieczone przed dzieleniem przez wartości bliskie zeru.
    """
    denom = y2.copy()
    mask = np.abs(denom) < eps
    denom[mask] = eps
    res = y1 / denom
    res[mask] = 0.0
    return res


def scalar_operation(y: np.ndarray, c: float, op: str = 'add') -> np.ndarray:
    """Operacja ze skalarem c: 'add', 'sub', 'mul', 'div'."""
    if op == 'add':
        return y + c
    elif op == 'sub':
        return y - c
    elif op == 'mul':
        return y * c
    elif op == 'div':
        if abs(c) < 1e-12:
            raise ZeroDivisionError("Dzielenie przez zero skalara.")
        return y / c
    raise ValueError(f"Nieznana operacja: {op}")


def absorbance_to_transmittance(y_abs: np.ndarray) -> np.ndarray:
    """
    Przelicza absorbancję na transmitancję w ułamku: T = 10^(-A).
    A = 0  -> T = 1.0
    A = 1  -> T = 0.1
    A = 2  -> T = 0.01
    """
    return np.power(10.0, -y_abs)


def transmittance_to_absorbance(y_trans: np.ndarray) -> np.ndarray:
    """
    Przelicza transmitancję (0–1) na absorbancję: A = -log10(T).
    Zabezpieczone przed zerem lub wartościami ujemnymi,
    ale bez sztucznego obcinania z góry przy T > 1.0 (fluktuacje szumu bazy).
    """
    safe_t = np.clip(y_trans, 1e-7, None)
    return -np.log10(safe_t)


def absorbance_to_percent_t(y_abs: np.ndarray) -> np.ndarray:
    """
    Przelicza absorbancję na % transmitancji: %T = 100 * 10^(-A).
    A = 0  -> %T = 100%
    A = 1  -> %T = 10%
    A = 2  -> %T = 1%
    """
    return 100.0 * np.power(10.0, -y_abs)


def percent_t_to_absorbance(y_pct_t: np.ndarray) -> np.ndarray:
    """
    Przelicza % transmitancji (0–100%) na absorbancję: A = 2 - log10(%T).
    Zabezpieczone przed niedozwolonymi argumentami logarytmu.
    """
    safe_pct = np.clip(y_pct_t, 1e-5, None)
    return 2.0 - np.log10(safe_pct)


def single_beam_ratio(sample_sb: np.ndarray, bg_sb: np.ndarray) -> np.ndarray:
    """Wyznacza absorbancję z widm jedno-wiązkowych: A = -log10(Sample / Background)."""
    transmittance = divide_spectra(sample_sb, bg_sb)
    return transmittance_to_absorbance(transmittance)


def kubelka_munk(r_inf: np.ndarray) -> np.ndarray:
    """Transformacja Kubelki-Munka dla DRIFTS: f(R) = (1 - R)^2 / (2 * R)."""
    safe_r = np.clip(r_inf, 1e-5, 1.0)
    return ((1.0 - safe_r) ** 2) / (2.0 * safe_r)


def normalize_to_band(x: np.ndarray, y: np.ndarray, target_wn: float) -> np.ndarray:
    """Dzieli widmo przez wartość absorbancji przy wskazanej liczbie falowej."""
    idx = int(np.abs(x - target_wn).argmin())
    val = y[idx]
    if abs(val) < 1e-8:
        return y.copy()
    return y / val

def estimate_subtraction_factor(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    nu_min: float, nu_max: float
) -> float:
    """
    Analitycznie estymuje współczynnik k metodą OLS w oknie [nu_min, nu_max],
    minimalizując ||S1 - k*S2||^2. Zabezpieczona przed różnymi siatkami i orientacją X.
    """
    low, high = min(nu_min, nu_max), max(nu_min, nu_max)

    # 1. Sprowadzenie S2 do siatki S1
    y2_aligned = align_to_reference(x2, y2, x1)

    # 2. Wycięcie okna spektralnego
    mask = (x1 >= low) & (x1 <= high)
    if not np.any(mask):
        return 0.0

    s1_win = y1[mask]
    s2_win = y2_aligned[mask]

    denom = float(np.sum(s2_win ** 2))
    if denom < 1e-12:
        return 0.0

    k = float(np.sum(s1_win * s2_win) / denom)
    return max(-999.0, min(999.0, k))