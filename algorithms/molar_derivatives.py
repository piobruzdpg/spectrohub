"""
algorithms/molar_derivatives.py
Zunifikowany silnik obliczeniowy widm molowych i różnicowych widm zmienności
dla układów dwuskładnikowych (Woda + X) oraz trójskładnikowych (Woda + X + Y).
Czysty moduł numeryczny (NumPy / SciPy) — zero kodu GUI.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Literal

M_H2O = 18.01528  # Masa molowa wody [g/mol]


def calculate_water_density(temp_c: float = 25.0) -> float:
    """Wyznacza gęstość czystej wody [g/cm³] w funkcji temperatury wg wzoru Tanaki (0-40 °C)."""
    t = float(temp_c)
    a1 = -3.983035
    a2 = 301.797
    a3 = 522528.9
    a4 = 69.34881
    a5 = 999.974950
    dens_kg_m3 = a5 * (1.0 - ((t + a1)**2 * (t + a2)) / (a3 * (t + a4)))
    return dens_kg_m3 / 1000.0


def compute_concentrations_and_wow(
    mass_solute_x: float,
    mass_water: float,
    density_solution: float,
    molar_mass_x: float,
    density_pure_water: Optional[float] = None,
    temp_c: float = 25.0,
    mass_solute_y: float = 0.0,
    molar_mass_y: float = 1.0
) -> Dict[str, float]:
    """Oblicza stężenia molowe składników [mol/dm³] oraz współczynnik odejmowania wody k_wow."""
    if density_pure_water is None or density_pure_water <= 0:
        density_pure_water = calculate_water_density(temp_c)

    m_tot = mass_solute_x + mass_water + mass_solute_y
    if m_tot <= 0 or density_solution <= 0 or molar_mass_x <= 0:
        return {
            "C_X": 0.0, "C_water": 0.0, "C_Y": 0.0,
            "k_wow": 0.0, "density_pure_water": density_pure_water
        }

    v_solution_cm3 = m_tot / density_solution
    c_x = (mass_solute_x / molar_mass_x) / (v_solution_cm3 / 1000.0)
    c_water = (mass_water / M_H2O) / (v_solution_cm3 / 1000.0)
    c_y = (mass_solute_y / molar_mass_y) / (v_solution_cm3 / 1000.0) if mass_solute_y > 0 and molar_mass_y > 0 else 0.0

    c_pure_water = (density_pure_water / M_H2O) * 1000.0
    k_wow = c_water / c_pure_water if c_pure_water > 0 else 0.0

    return {
        "C_X": float(c_x),
        "C_water": float(c_water),
        "C_Y": float(c_y),
        "k_wow": float(k_wow),
        "density_pure_water": float(density_pure_water)
    }


def subtract_water(
    sample_wn: np.ndarray,
    sample_y: np.ndarray,
    water_wn: np.ndarray,
    water_y: np.ndarray,
    k_wow: float
) -> np.ndarray:
    """Odejmuje widmo wody: S_izol = S_probka - k_wow * S_woda."""
    if len(sample_wn) == len(water_wn) and np.allclose(sample_wn, water_wn):
        water_aligned = water_y
    else:
        if water_wn[0] > water_wn[-1]:
            w_x, w_y = water_wn[::-1], water_y[::-1]
        else:
            w_x, w_y = water_wn, water_y
        water_aligned = np.interp(sample_wn, w_x, w_y)

    return sample_y - (k_wow * water_aligned)


def interpolate_spectrum_at_concentration(
    ref_concentrations: np.ndarray,
    ref_spectra_matrix: np.ndarray,
    target_concentration: float,
    method: Literal["linear", "quadratic"] = "linear"
) -> np.ndarray:
    """
    Interpoluje nienormalizowane widmo absorbancji A_izol dla stężenia target_concentration
    na podstawie serii widm wyizolowanych (W x N) i wektora stężeń (N,).
    """
    concs = np.asarray(ref_concentrations, dtype=np.float64)
    mat = np.asarray(ref_spectra_matrix, dtype=np.float64)
    n_points = len(concs)

    if n_points == 0:
        return np.zeros(mat.shape[0], dtype=np.float64)

    if n_points == 1:
        ratio = target_concentration / concs[0] if concs[0] > 0 else 1.0
        return mat[:, 0] * ratio

    sort_idx = np.argsort(concs)
    concs_sorted = concs[sort_idx]
    mat_sorted = mat[:, sort_idx]
    w_len = mat.shape[0]

    out_spec = np.zeros(w_len, dtype=np.float64)
    if method == "quadratic" and n_points >= 3:
        x_poly = np.column_stack([concs_sorted**2, concs_sorted, np.ones(n_points)])
        coeffs, _, _, _ = np.linalg.lstsq(x_poly, mat_sorted.T, rcond=None)
        target_v = np.array([target_concentration**2, target_concentration, 1.0])
        out_spec = target_v @ coeffs
    else:
        for w_i in range(w_len):
            out_spec[w_i] = np.interp(target_concentration, concs_sorted, mat_sorted[w_i, :])

    return out_spec


def normalize_spectrum(
    y: np.ndarray,
    conc: float,
    method: Literal["molar", "area", "none"] = "molar",
    wn: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalizuje wyizolowane widmo (molowo lub polowo AUC)."""
    if method == "molar":
        return y / conc if conc > 1e-12 else y.copy()

    elif method == "area":
        if wn is not None:
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            area = np.abs(_trapz(y, wn))
        else:
            area = np.sum(np.abs(y))
        return y / area if area > 1e-12 else y.copy()

    return y.copy()


def compute_variation_spectrum_matrix(
    spectra_matrix: np.ndarray,
    concentrations: np.ndarray,
    model: Literal["linear", "quadratic"] = "linear"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wektorowe wyznaczanie różnicowego widma zmienności, widma granicznego oraz metryk dopasowania (RMSE, SE_p1, R²).
    """
    c_arr = np.asarray(concentrations, dtype=np.float64)
    y_mat = np.asarray(spectra_matrix, dtype=np.float64)
    w_len, n_samples = y_mat.shape

    if n_samples < 2:
        zeros = np.zeros(w_len, dtype=np.float64)
        return zeros, zeros, zeros, zeros, zeros

    if model == "quadratic" and n_samples >= 3:
        x_des = np.column_stack([c_arr**2, c_arr, np.ones(n_samples)])
        p_deg = 3
        sol, _, _, _ = np.linalg.lstsq(x_des, y_mat.T, rcond=None)
        p1 = sol[1, :]
        p0 = sol[2, :]
        y_pred = (x_des @ sol).T
        try:
            xtx_inv = np.linalg.inv(x_des.T @ x_des)
            var_factor_p1 = xtx_inv[1, 1]
        except np.linalg.LinAlgError:
            var_factor_p1 = 0.0
    else:
        x_des = np.column_stack([c_arr, np.ones(n_samples)])
        p_deg = 2
        sol, _, _, _ = np.linalg.lstsq(x_des, y_mat.T, rcond=None)
        p1 = sol[0, :]
        p0 = sol[1, :]
        y_pred = (x_des @ sol).T
        s_xx = np.sum((c_arr - np.mean(c_arr))**2)
        var_factor_p1 = 1.0 / s_xx if s_xx > 1e-12 else 0.0

    df = max(1, n_samples - p_deg)
    ss_res = np.sum((y_mat - y_pred)**2, axis=1)
    rmse = np.sqrt(ss_res / df)
    p1_err = rmse * np.sqrt(max(0.0, var_factor_p1))

    ss_tot = np.sum((y_mat - np.mean(y_mat, axis=1, keepdims=True))**2, axis=1)
    r2 = np.zeros(w_len, dtype=np.float64)
    valid_mask = ss_tot > 1e-10
    r2[valid_mask] = 1.0 - (ss_res[valid_mask] / ss_tot[valid_mask])
    r2 = np.clip(r2, 0.0, 1.0)

    return p1, p0, p1_err, rmse, r2