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
    if np.array_equal(x_src, x_ref):
        return y_src.copy()

    if x_src[0] > x_src[-1]:
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
    denom[mask] = eps if np.issubdtype(denom.dtype, np.floating) else eps
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
    """Przelicza absorbancję na transmitancję: T = 10^(-A)."""
    return np.power(10.0, -np.clip(y_abs, -5.0, 10.0))


def transmittance_to_absorbance(y_trans: np.ndarray) -> np.ndarray:
    """Przelicza transmitancję na absorbancję: A = -log10(T)."""
    safe_t = np.clip(y_trans, 1e-6, 1.0)
    return -np.log10(safe_t)


def percent_t_to_absorbance(y_pct_t: np.ndarray) -> np.ndarray:
    """Przelicza %T na absorbancję: A = 2 - log10(%T)."""
    safe_pct = np.clip(y_pct_t, 1e-4, 100.0)
    return 2.0 - np.log10(safe_pct)


def absorbance_to_percent_t(y_abs: np.ndarray) -> np.ndarray:
    """Przelicza absorbancję na %T: %T = 100 * 10^(-A)."""
    return 100.0 * absorbance_to_transmittance(y_abs)


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
    idx = np.abs(x - target_wn).argmin()
    val = y[idx]
    if abs(val) < 1e-8:
        return y.copy()
    return y / val