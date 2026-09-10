"""
gui/node_editor/evaluator.py
Zunifikowany silnik obliczeniowy grafu DAG:
- Routing wieloportowy: wyjścia profili stężeń MCR (C), ładunków PCA (Scores), residuów i całek.
- Dekompozycja HDO & Pasma Syntetyczne (Błyskawiczny OLS w podglądzie + Pełny Monte Carlo przy wykonaniu).
- Broadcasting operacji dwuwidmowych (1D z 2D, 2D z 2D).
- Wielowejściowy terminal okna roboczego (fan-in) oraz Mux (Merge Spectra).
- Wewnętrzny standard (IS) oraz integrator pasma (Kinetyka).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging
from scipy.optimize import least_squares

from core.session import session
from core.models import SpectrumData
from algorithms import preprocessing as prep
from algorithms.atmospheric import AtmosphericCorrectionEngine
from algorithms.chemometrics import multivariate as chem
from algorithms.fitting.hdo_engine import (
    HDOFitEngine, gxl_peak, evaluate_total_fit, hdo_residuals
)

logger = logging.getLogger("SpectroHub")

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _generate_synthetic_peak(x: np.ndarray, shape: str, center: float,
                             amp: float, width: float, shape_param: float = 0.5) -> np.ndarray:
    """Generuje profil pasma syntetycznego na zadanej siatce x."""
    if len(x) == 0:
        return np.zeros_like(x)

    w = max(width, 1e-4)
    if shape == "gxl":
        return gxl_peak(x, amp, center, shape_param, w)
    elif shape == "gaussian":
        sigma = w / 2.35482
        return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    elif shape == "lorentzian":
        gamma = w / 2.0
        return amp * (gamma ** 2 / ((x - center) ** 2 + gamma ** 2))
    elif shape == "voigt":
        sigma = w / 2.35482
        gamma = sigma
        diff = (x - center) / sigma
        return amp * np.exp(-0.5 * diff ** 2) / (1.0 + (diff * gamma / sigma) ** 2)
    elif shape == "pseudovoigt":
        eta = np.clip(shape_param, 0.0, 1.0)
        sigma = w / 2.35482
        gamma = w / 2.0
        g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        l = gamma ** 2 / ((x - center) ** 2 + gamma ** 2)
        return amp * (eta * l + (1.0 - eta) * g)

    return np.zeros_like(x)


def apply_single_step(x: np.ndarray, y: np.ndarray, step: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Aplikuje pojedynczy krok sekwencyjny na wektorze 1D lub wielokolumnowej macierzy 2D."""
    stype = step.get('type', '')
    curr_x = x.copy()
    curr_y = y.copy()

    try:
        if stype == 'crop':
            curr_x, curr_y = prep.crop_range(curr_x, curr_y, step.get('min_w', 1000.0), step.get('max_w', 1800.0))
        elif curr_y.ndim == 2:
            if stype == 'asls':
                for c in range(curr_y.shape[1]):
                    curr_y[:, c] -= prep.baseline_asls(curr_y[:, c], lam=step.get('lam', 1e5), p=step.get('p', 0.001))
            elif stype == 'airpls':
                for c in range(curr_y.shape[1]):
                    curr_y[:, c] -= prep.baseline_airpls(curr_y[:, c], lam=step.get('lam', 1e5), n_iter=step.get('n_iter', 15))
            elif stype == 'poly_bl':
                for c in range(curr_y.shape[1]):
                    curr_y[:, c] -= prep.baseline_polynomial(curr_y[:, c], curr_x, deg=step.get('deg', 3))
            elif stype == 'linear_bl':
                if len(curr_x) >= 2:
                    for c in range(curr_y.shape[1]):
                        curr_y[:, c] -= prep.baseline_linear(curr_x, curr_y[:, c], curr_x[0], curr_x[-1])
            elif stype == 'fsd':
                for c in range(curr_y.shape[1]):
                    curr_y[:, c] = prep.fourier_self_deconvolution(curr_x, curr_y[:, c], fwhm=step.get('fwhm', 18.0), k_factor=step.get('k_factor', 2.0))
            elif stype == 'sg_smooth':
                curr_y = prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=0)
            elif stype == 'deriv1':
                curr_y = prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=1)
            elif stype == 'deriv2':
                curr_y = -1.0 * prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=2)
            elif stype == 'snv':
                curr_y = prep.apply_snv(curr_y)
            elif stype == 'minmax':
                curr_y = prep.apply_min_max(curr_y)
            elif stype == 'l1':
                curr_y = prep.apply_norm_l1(curr_y)
            elif stype == 'l2':
                curr_y = prep.apply_vector_norm_l2(curr_y)
            elif stype == 'offset_min':
                curr_y -= np.min(curr_y, axis=0, keepdims=True)
            elif stype == 'offset_point':
                curr_y = prep.apply_offset(curr_y, curr_x, target_wn=step.get('target_wn', 2000.0), target_val=0.0)
            elif stype == 'scalar_math':
                op = step.get('op', 'mul')
                val = float(step.get('val', 1.0))
                if op == 'mul': curr_y *= val
                elif op == 'div': curr_y = np.where(abs(val) > 1e-12, curr_y / val, curr_y)
                elif op == 'add': curr_y += val
                elif op == 'sub': curr_y -= val
        else:
            if stype == 'asls':
                curr_y -= prep.baseline_asls(curr_y, lam=step.get('lam', 1e5), p=step.get('p', 0.001))
            elif stype == 'airpls':
                curr_y -= prep.baseline_airpls(curr_y, lam=step.get('lam', 1e5), n_iter=step.get('n_iter', 15))
            elif stype == 'poly_bl':
                curr_y -= prep.baseline_polynomial(curr_y, curr_x, deg=step.get('deg', 3))
            elif stype == 'linear_bl':
                if len(curr_x) >= 2:
                    curr_y -= prep.baseline_linear(curr_x, curr_y, curr_x[0], curr_x[-1])
            elif stype == 'fsd':
                curr_y = prep.fourier_self_deconvolution(curr_x, curr_y, fwhm=step.get('fwhm', 18.0), k_factor=step.get('k_factor', 2.0))
            elif stype == 'sg_smooth':
                curr_y = prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=0)
            elif stype == 'deriv1':
                curr_y = prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=1)
            elif stype == 'deriv2':
                curr_y = -1.0 * prep.apply_savgol(curr_y, window=step.get('win', 11), poly=step.get('poly', 3), deriv=2)
            elif stype == 'snv':
                curr_y = prep.apply_snv(curr_y)
            elif stype == 'minmax':
                curr_y = prep.apply_min_max(curr_y)
            elif stype == 'l1':
                curr_y = prep.apply_norm_l1(curr_y)
            elif stype == 'l2':
                curr_y = prep.apply_vector_norm_l2(curr_y)
            elif stype == 'offset_min':
                curr_y -= np.min(curr_y)
            elif stype == 'offset_point':
                curr_y = prep.apply_offset(curr_y, curr_x, target_wn=step.get('target_wn', 2000.0), target_val=0.0)
            elif stype == 'scalar_math':
                op = step.get('op', 'mul')
                val = float(step.get('val', 1.0))
                if op == 'mul': curr_y = curr_y * val
                elif op == 'div': curr_y = np.where(abs(val) > 1e-12, curr_y / val, curr_y)
                elif op == 'add': curr_y = curr_y + val
                elif op == 'sub': curr_y = curr_y - val
    except Exception as err:
        logger.warning(f"Błąd kalkulacji kroku {stype}: {err}")

    return curr_x, curr_y


def apply_photometry_conversion(y: np.ndarray, mode: str) -> np.ndarray:
    eps = 1e-7
    if mode == "a_to_t":
        return 10.0 ** (-y)
    elif mode == "t_to_a":
        return -np.log10(np.clip(y, eps, None))
    elif mode == "tpct_to_a":
        return 2.0 - np.log10(np.clip(y, 1e-5, None))
    elif mode == "a_to_tpct":
        return 10.0 ** (2.0 - y)
    elif mode == "kubelka_munk":
        r = np.clip(y, eps, 1.0)
        return ((1.0 - r) ** 2) / (2.0 * r)
    elif mode == "inv_kubelka_munk":
        f = np.clip(y, 0.0, None)
        return 1.0 + f - np.sqrt(np.clip(f ** 2 + 2.0 * f, 0.0, None))
    return y


def convolve_dft_sticks(freqs: np.ndarray, intens: np.ndarray, target_x: np.ndarray,
                        fwhm: float = 12.0, shape: str = "lorentzian") -> np.ndarray:
    if len(freqs) == 0 or len(target_x) == 0:
        return np.zeros_like(target_x)

    y_out = np.zeros_like(target_x, dtype=np.float64)
    gamma = fwhm / 2.0

    if shape == "lorentzian":
        for nu0, amp in zip(freqs, intens):
            y_out += amp * (gamma / (np.pi * ((target_x - nu0) ** 2 + gamma ** 2)))
    else:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        for nu0, amp in zip(freqs, intens):
            y_out += amp * (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((target_x - nu0) ** 2) / (2.0 * sigma ** 2))

    max_val = np.max(y_out)
    if max_val > 1e-9:
        y_out = (y_out / max_val) * (np.max(intens) if np.max(intens) < 2.5 else 1.0)

    return y_out


class DAGEvaluator:
    """Silnik ewaluacji potoku z routingiem wieloportowym, chemometrią macierzową i dekompozycją HDO."""

    @staticmethod
    def topological_sort(nodes: list) -> Tuple[Optional[list], bool]:
        adj = {n: [] for n in nodes}
        in_degree = {n: 0 for n in nodes}

        for n in nodes:
            for in_p in n.in_ports.values():
                for conn in in_p.connections:
                    if conn.out_port and conn.out_port.node_item:
                        src = conn.out_port.node_item
                        if src in adj:
                            adj[src].append(n)
                            in_degree[n] += 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(nodes):
            return None, True
        return ordered, False

    @staticmethod
    def evaluate_dag(ordered_nodes: list, default_x: np.ndarray, default_y: np.ndarray,
                     session_spectra: Dict[str, SpectrumData],
                     is_execution: bool = False) -> Dict[Tuple[Any, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        port_results: Dict[Tuple[Any, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

        def get_source_data(in_port) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if in_port and in_port.connections:
                conn = in_port.connections[0]
                if conn.out_port and conn.out_port.node_item:
                    key = (conn.out_port.node_item, conn.out_port.port_id)
                    return port_results.get(key, (None, None))
            return None, None

        for node in ordered_nodes:
            ntype = node.node_type

            # 1. ŹRÓDŁO DANYCH (KONFIGUROWALNY WĘZEŁ WEJŚCIOWY)
            if ntype == "input":
                source_mode = node.step_data.get('source_mode', 'active')

                if source_mode == 'specific':
                    spec_name = node.step_data.get('specific_spectrum', '')
                    spec = session.get_spectrum(spec_name)
                    if spec:
                        port_results[(node, "out")] = (spec.wavenumbers.copy(), spec.intensities.copy())
                    else:
                        port_results[(node, "out")] = (default_x.copy(), default_y.copy())

                elif source_mode == 'visible':
                    visible = session.get_pipeline_target_spectra()
                    if len(visible) >= 2:
                        ref_wn = visible[0].wavenumbers
                        cols = []
                        for s in visible:
                            xw, yw = (s.wavenumbers[::-1], s.intensities[::-1]) if s.wavenumbers[0] > s.wavenumbers[-1] else (s.wavenumbers, s.intensities)
                            cols.append(np.interp(ref_wn, xw, yw))
                        port_results[(node, "out")] = (ref_wn.copy(), np.column_stack(cols))
                    elif len(visible) == 1:
                        port_results[(node, "out")] = (visible[0].wavenumbers.copy(), visible[0].intensities.copy())
                    else:
                        port_results[(node, "out")] = (default_x.copy(), default_y.copy())

                elif source_mode == 'series':
                    if session.active_series:
                        port_results[(node, "out")] = (session.active_series.wavenumbers.copy(), session.active_series.get_matrix_2d().copy())
                    else:
                        port_results[(node, "out")] = (default_x.copy(), default_y.copy())

                else:
                    active = session.active_spectrum
                    if active and len(active.wavenumbers) > 2:
                        port_results[(node, "out")] = (active.wavenumbers.copy(), active.intensities.copy())
                    else:
                        port_results[(node, "out")] = (default_x.copy(), default_y.copy())

            # 2. WIDMA REFERENCYJNE Z PROJEKTU (OGÓLNE I ATM)
            elif ntype == "spectrum_select":
                selected = node.step_data.get('selected_spectra', [])
                if not selected and node.step_data.get('selected_spectrum'):
                    selected = [node.step_data['selected_spectrum']]

                valid_specs = [session.get_spectrum(name) for name in selected if session.get_spectrum(name)]

                if valid_specs:
                    ref_x = valid_specs[0].wavenumbers.copy()
                    cols = []
                    for s in valid_specs:
                        sx, sy = s.wavenumbers, s.intensities
                        sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
                        cols.append(np.interp(ref_x, sx_s, sy_s))

                    y_out = np.column_stack(cols) if len(cols) > 1 else cols[0]
                    port_results[(node, "out")] = (ref_x, y_out)
                else:
                    port_results[(node, "out")] = (default_x.copy(), np.zeros_like(default_x))

            # 3. HDO: WIDMO SKŁADOWE ZE SKALAREM (c · W_i)
            elif ntype == "hdo_spectrum_scaled":
                sel_name = node.step_data.get('selected_spectrum', '')
                scale = float(node.step_data.get('scale', 1.0))
                spec = session.get_spectrum(sel_name)

                if spec:
                    sx = spec.wavenumbers
                    sy = spec.intensities * scale
                    sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
                    y_interp = np.interp(default_x, sx_s, sy_s)
                    port_results[(node, "out")] = (default_x.copy(), y_interp)
                else:
                    port_results[(node, "out")] = (default_x.copy(), np.zeros_like(default_x))

            # 4. HDO: PASMO SYNTETYCZNE
            elif ntype == "synthetic_peak":
                sh = node.step_data.get('shape', 'gxl')
                c = float(node.step_data.get('center', 1650.0))
                a = float(node.step_data.get('amp', 0.5))
                w = float(node.step_data.get('width', 30.0))
                f = float(node.step_data.get('shape_param', 0.5))
                y_peak = _generate_synthetic_peak(default_x, shape=sh, center=c, amp=a, width=w, shape_param=f)
                port_results[(node, "out")] = (default_x.copy(), y_peak)

            # 5. HDO: PARAMETRY MONTE CARLO
            elif ntype == "mc_params":
                port_results[(node, "out_mc")] = (np.array([1.0]), np.array([1.0]))

            # 6. HDO: SOLWER DEKOMPOZYCJI HDO FIT
            elif ntype == "hdo_decomposition":
                data_tar = get_source_data(node.in_ports.get("in_target"))
                target_x, target_y = data_tar if data_tar[0] is not None else (default_x.copy(), default_y.copy())
                target_y_1d = target_y[:, 0] if target_y.ndim == 2 else target_y

                n_multiplier = float(node.step_data.get('n_multiplier', 1.0))
                w0_target = target_y_1d * n_multiplier

                # Pobranie widm bazowych W_k
                p_bases = node.in_ports.get("in_bases")
                bases_list, bounds_list, base_names = [], [], []
                if p_bases and p_bases.connections:
                    for conn in p_bases.connections:
                        if conn.out_port and conn.out_port.node_item:
                            src_node = conn.out_port.node_item
                            src_k = (src_node, conn.out_port.port_id)
                            ds = port_results.get(src_k, (None, None))
                            if ds[0] is not None and ds[1] is not None:
                                bx, by = ds
                                by_col = by[:, 0] if by.ndim == 2 else by
                                bx_s, by_s = (bx[::-1], by[::-1]) if bx[0] > bx[-1] else (bx, by)
                                by_interp = np.interp(target_x, bx_s, by_s)
                                bases_list.append(by_interp)

                                w_min = float(src_node.step_data.get('w_min', 0.0))
                                w_max = float(src_node.step_data.get('w_max', 1.5))
                                bounds_list.append((w_min, w_max))
                                b_name = src_node.step_data.get('selected_spectrum') or src_node.title
                                base_names.append(b_name)

                # Pobranie pików syntetycznych
                p_peaks = node.in_ports.get("in_peaks")
                synth_peak_nodes = []
                if p_peaks and p_peaks.connections:
                    for conn in p_peaks.connections:
                        if conn.out_port and conn.out_port.node_item:
                            synth_peak_nodes.append(conn.out_port.node_item)
                n_peaks = len(synth_peak_nodes)

                # Parametry MC z portu in_mc
                mc_node = None
                p_mc = node.in_ports.get("in_mc")
                if p_mc and p_mc.connections and p_mc.connections[0].out_port:
                    mc_node = p_mc.connections[0].out_port.node_item

                if mc_node:
                    n_iterations = int(mc_node.step_data.get('n_iter', 2000))
                    penalty_weight = float(mc_node.step_data.get('penalty_weight', 10.0))
                    baseline_active = bool(mc_node.step_data.get('baseline_active', False))
                else:
                    n_iterations = int(node.step_data.get('n_iter', 2000))
                    penalty_weight = float(node.step_data.get('penalty_weight', 10.0))
                    baseline_active = bool(node.step_data.get('baseline_active', False))

                if len(bases_list) > 0 or n_peaks > 0:
                    try:
                        if is_execution:
                            # 1. PEŁNE WYKONANIE: Wieloprocesowy Monte Carlo
                            results = HDOFitEngine.run_monte_carlo(
                                wavenb=target_x,
                                target=w0_target,
                                base_spectra_list=bases_list,
                                comp_bounds=bounds_list,
                                n_peaks=n_peaks,
                                n_iterations=n_iterations,
                                penalty_weight=penalty_weight,
                                st_dev=0.01,
                                baseline_active=baseline_active
                            )
                            fit_y = results['fitted_curve']
                            excess_res = results['excess_residual']
                            comps_y = np.column_stack(results['scaled_components']) if results['scaled_components'] else np.zeros_like(target_x)
                            peaks_y = np.column_stack(results['peaks_curves']) if results['peaks_curves'] else np.zeros_like(target_x)

                            # Zapis metryk do kroku noda (dla raportu i nazewnictwa)
                            node.step_data['fit_metrics'] = {
                                'coefficients': results['coefficients'],
                                'base_names': base_names,
                                'r2': results['r2'],
                                'rmse': results['rmse'],
                                'chi2_red': results['chi2_red']
                            }
                        else:
                            # 2. BŁYSKAWICZNY PODGLĄD NA ŻYWO (Preview): OLS w ułamku ms
                            k_comps = len(bases_list)
                            base_mat = np.vstack(bases_list) if k_comps > 0 else np.empty((0, len(target_x)))
                            total_p = 3 + k_comps + 4 * n_peaks
                            param_init = np.zeros(total_p, dtype=np.float64)

                            for i in range(k_comps):
                                param_init[3 + i] = 0.5 * (bounds_list[i][0] + bounds_list[i][1])

                            max_amp = max(np.max(w0_target), 1e-3)
                            for j in range(n_peaks):
                                pos = 3 + k_comps + j * 4
                                pn = synth_peak_nodes[j]
                                param_init[pos] = float(pn.step_data.get('amp', 0.5 * max_amp))
                                param_init[pos + 1] = float(pn.step_data.get('center', 1650.0))
                                param_init[pos + 2] = float(pn.step_data.get('shape_param', 0.5))
                                param_init[pos + 3] = float(pn.step_data.get('width', 30.0))

                            res_fast = least_squares(
                                hdo_residuals, param_init,
                                args=(w0_target, target_x, base_mat, n_peaks, penalty_weight, baseline_active),
                                max_nfev=35, ftol=1e-3, xtol=1e-3
                            )
                            fit_y = evaluate_total_fit(res_fast.x, target_x, base_mat, n_peaks, baseline_active)

                            sum_comps = np.zeros_like(target_x)
                            scaled_c = []
                            extracted_c = []
                            for i in range(k_comps):
                                c_val = float(res_fast.x[3 + i])
                                extracted_c.append(c_val)
                                c_curve = c_val * base_mat[i]
                                scaled_c.append(c_curve)
                                sum_comps += c_curve
                            excess_res = w0_target - sum_comps

                            p_curves = []
                            for j in range(n_peaks):
                                pos = 3 + k_comps + j * 4
                                p_curves.append(gxl_peak(target_x, res_fast.x[pos], res_fast.x[pos + 1], res_fast.x[pos + 2], res_fast.x[pos + 3]))

                            comps_y = np.column_stack(scaled_c) if scaled_c else np.zeros_like(target_x)
                            peaks_y = np.column_stack(p_curves) if p_curves else np.zeros_like(target_x)

                            ss_res = float(np.sum((w0_target - fit_y) ** 2))
                            ss_tot = float(np.sum((w0_target - np.mean(w0_target)) ** 2))
                            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            rmse_val = float(np.sqrt(np.mean((w0_target - fit_y) ** 2)))

                            node.step_data['fit_metrics'] = {
                                'coefficients': extracted_c,
                                'base_names': base_names,
                                'r2': r2_val,
                                'rmse': rmse_val
                            }

                        port_results[(node, "out_fit")] = (target_x.copy(), fit_y)
                        port_results[(node, "out_res")] = (target_x.copy(), excess_res)
                        port_results[(node, "out_fit_res")] = (target_x.copy(), w0_target - fit_y)
                        port_results[(node, "out_comps")] = (target_x.copy(), comps_y)
                        port_results[(node, "out_peaks")] = (target_x.copy(), peaks_y)

                    except Exception as e:
                        logger.warning(f"Błąd ewaluacji HDO Fit w DAG: {e}")
                        port_results[(node, "out_fit")] = (target_x.copy(), w0_target)
                        port_results[(node, "out_res")] = (target_x.copy(), np.zeros_like(target_x))
                        port_results[(node, "out_fit_res")] = (target_x.copy(), np.zeros_like(target_x))
                        port_results[(node, "out_comps")] = (target_x.copy(), np.zeros_like(target_x))
                        port_results[(node, "out_peaks")] = (target_x.copy(), np.zeros_like(target_x))
                else:
                    port_results[(node, "out_fit")] = (target_x.copy(), w0_target)
                    port_results[(node, "out_res")] = (target_x.copy(), np.zeros_like(target_x))
                    port_results[(node, "out_fit_res")] = (target_x.copy(), np.zeros_like(target_x))
                    port_results[(node, "out_comps")] = (target_x.copy(), np.zeros_like(target_x))
                    port_results[(node, "out_peaks")] = (target_x.copy(), np.zeros_like(target_x))

            # 7. ZBIERACZ / MUX (MERGE SPECTRA W MACIERZ 2D)
            elif ntype == "merge_spectra":
                in_p = node.in_ports.get("in")
                sources = []
                if in_p and in_p.connections:
                    for conn in in_p.connections:
                        if conn.out_port and conn.out_port.node_item:
                            src_k = (conn.out_port.node_item, conn.out_port.port_id)
                            ds = port_results.get(src_k, (None, None))
                            if ds[0] is not None and ds[1] is not None:
                                sources.append(ds)

                if sources:
                    ref_x = sources[0][0]
                    cols = []
                    for sx, sy in sources:
                        sx_s = sx[::-1] if sx[0] > sx[-1] else sx
                        if sy.ndim == 2:
                            for c in range(sy.shape[1]):
                                col_data = sy[::-1, c] if sx[0] > sx[-1] else sy[:, c]
                                cols.append(np.interp(ref_x, sx_s, col_data))
                        else:
                            col_data = sy[::-1] if sx[0] > sx[-1] else sy
                            cols.append(np.interp(ref_x, sx_s, col_data))

                    combined_y = np.column_stack(cols)
                    port_results[(node, "out")] = (ref_x.copy(), combined_y)
                else:
                    port_results[(node, "out")] = (default_x.copy(), default_y.copy())

            # 8. CHEMOMETRIA: ODSZUMIANIE SVD / PCA RANK-K
            elif ntype == "svd_denoise":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    k = int(node.step_data.get('k', 2))
                    center = bool(node.step_data.get('center', False))

                    if curr_y.ndim == 2 and curr_y.shape[1] >= 2:
                        k_eff = min(k, curr_y.shape[0], curr_y.shape[1])
                        res = chem.run_pca(curr_y, n_components=k_eff, center=center)
                        scores = res['scores']
                        x_samples = np.arange(1, scores.shape[0] + 1, dtype=np.float64)

                        port_results[(node, "out")] = (curr_x.copy(), res['dr'])
                        port_results[(node, "out_res")] = (curr_x.copy(), res['residuals'])
                        port_results[(node, "out_scores")] = (x_samples, scores)
                    else:
                        ref_mat = None
                        if session.active_series:
                            ref_mat = session.active_series.get_matrix_2d()
                        else:
                            vis = session.get_pipeline_target_spectra()
                            if len(vis) >= 2:
                                ref_mat = np.column_stack([np.interp(curr_x, s.wavenumbers[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.wavenumbers, s.intensities[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.intensities) for s in vis])

                        if ref_mat is not None and ref_mat.shape[1] >= 2:
                            k_eff = min(k, ref_mat.shape[0], ref_mat.shape[1])
                            res = chem.run_pca(ref_mat, n_components=k_eff, center=center)
                            U_k = res['loadings']
                            if center:
                                mu = res['mean_spectrum']
                                y_c = curr_y - mu
                                y_recon = mu + (U_k @ (U_k.T @ y_c))
                            else:
                                y_recon = U_k @ (U_k.T @ curr_y)
                            port_results[(node, "out")] = (curr_x.copy(), y_recon)
                            port_results[(node, "out_res")] = (curr_x.copy(), curr_y - y_recon)
                            port_results[(node, "out_scores")] = (np.array([1.0]), np.array([1.0]))
                        else:
                            port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                            port_results[(node, "out_scores")] = (np.array([1.0]), np.array([1.0]))
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_res")] = (None, None)
                    port_results[(node, "out_scores")] = (None, None)

            # 9. CHEMOMETRIA: DEKOMPOZYCJA MCR-ALS / SPEXFA
            elif ntype == "mcr_decompose":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    k = int(node.step_data.get('k', 2))
                    method = node.step_data.get('method', 'mcr_als')

                    if curr_y.ndim == 2 and curr_y.shape[1] >= 2:
                        target_mat = curr_y
                    elif session.active_series:
                        target_mat = session.active_series.get_matrix_2d()
                    else:
                        vis = session.get_pipeline_target_spectra()
                        if len(vis) >= 2:
                            target_mat = np.column_stack([np.interp(curr_x, s.wavenumbers[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.wavenumbers, s.intensities[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.intensities) for s in vis])
                        else:
                            target_mat = None

                    if target_mat is not None and target_mat.shape[1] >= 2:
                        k_eff = min(k, target_mat.shape[0], target_mat.shape[1])
                        try:
                            if method == 'spexfa':
                                res = chem.run_spexfa(target_mat, n=k_eff)
                            else:
                                res = chem.run_mcr_als(
                                    target_mat, rank=k_eff,
                                    non_negative=node.step_data.get('non_negative', True),
                                    closure=node.step_data.get('closure', False),
                                    unimodal=node.step_data.get('unimodal', False),
                                    max_iter=node.step_data.get('max_iter', 500)
                                )
                            pure_spec = res['pure_spectra']
                            pure_conc = res['pure_concentrations']
                            dr = res['dr']
                            residuals = res['residuals']

                            x_samples = np.arange(1, pure_conc.shape[0] + 1, dtype=np.float64)
                            spark_y = pure_spec[:, 0] if pure_spec.ndim == 2 else pure_spec

                            port_results[(node, "out")] = (curr_x.copy(), pure_spec if curr_y.ndim == 2 else spark_y)
                            port_results[(node, "out_conc")] = (x_samples, pure_conc)
                            port_results[(node, "out_recon")] = (curr_x.copy(), dr if curr_y.ndim == 2 else (dr[:, 0] if dr.ndim == 2 else dr))
                            port_results[(node, "out_res")] = (curr_x.copy(), residuals if curr_y.ndim == 2 else (residuals[:, 0] if residuals.ndim == 2 else residuals))
                        except Exception as e:
                            logger.warning(f"Błąd kalkulacji MCR: {e}")
                            port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_conc")] = (np.array([1.0]), np.array([1.0]))
                            port_results[(node, "out_recon")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                    else:
                        port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                        port_results[(node, "out_conc")] = (np.array([1.0]), np.array([1.0]))
                        port_results[(node, "out_recon")] = (curr_x.copy(), curr_y.copy())
                        port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_conc")] = (None, None)
                    port_results[(node, "out_recon")] = (None, None)
                    port_results[(node, "out_res")] = (None, None)

            # 10. CHEMOMETRIA: SKALOWANIA
            elif ntype == "matrix_scaling":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    st = node.step_data.get('scale_type', 'mean_center')

                    if curr_y.ndim == 2 and curr_y.shape[1] > 1:
                        if st == 'mean_center':
                            scaled_y, _ = prep.apply_mean_center(curr_y)
                        elif st == 'autoscaling':
                            scaled_y = prep.apply_autoscaling(curr_y)
                        elif st == 'pareto':
                            scaled_y = prep.apply_pareto_scaling(curr_y)
                        elif st == 'msc':
                            scaled_y = prep.apply_msc(curr_y)
                        else:
                            scaled_y = curr_y
                        port_results[(node, "out")] = (curr_x.copy(), scaled_y)
                    else:
                        port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                else:
                    port_results[(node, "out")] = (None, None)

            # 11. SPLOT PASM DFT
            elif ntype == "dft_broadener":
                dft_name = node.step_data.get('dft_name', '')
                scale = float(node.step_data.get('scale_factor', 0.965))
                fwhm = float(node.step_data.get('fwhm', 12.0))
                shape = node.step_data.get('shape', 'lorentzian')

                spec = session.get_spectrum(dft_name)
                in_x, _ = get_source_data(node.in_ports.get("in"))
                grid_x = in_x if in_x is not None else default_x

                if spec and hasattr(spec, 'stick_freqs') and len(spec.stick_freqs) > 0:
                    scaled_freqs = spec.stick_freqs * scale
                    y_conv = convolve_dft_sticks(scaled_freqs, spec.stick_intens, grid_x, fwhm=fwhm, shape=shape)
                    port_results[(node, "out")] = (grid_x.copy(), y_conv)
                else:
                    port_results[(node, "out")] = (grid_x.copy(), np.zeros_like(grid_x))

            # 12. FOTOMETRIA
            elif ntype == "photometry":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    mode = node.step_data.get('photo_mode', 'tpct_to_a')
                    res_y = apply_photometry_conversion(data_in[1], mode)
                    port_results[(node, "out")] = (data_in[0].copy(), res_y)
                else:
                    port_results[(node, "out")] = (None, None)

            # 13. LINIE BAZOWE
            elif ntype in ("asls", "airpls", "poly_bl", "linear_bl"):
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    bl = np.zeros_like(curr_y)

                    if curr_y.ndim == 2:
                        for c in range(curr_y.shape[1]):
                            if ntype == "asls":
                                bl[:, c] = prep.baseline_asls(curr_y[:, c], lam=node.step_data.get('lam', 1e5), p=node.step_data.get('p', 0.001))
                            elif ntype == "airpls":
                                bl[:, c] = prep.baseline_airpls(curr_y[:, c], lam=node.step_data.get('lam', 1e5), n_iter=node.step_data.get('n_iter', 15))
                            elif ntype == "poly_bl":
                                bl[:, c] = prep.baseline_polynomial(curr_y[:, c], curr_x, deg=node.step_data.get('deg', 3))
                            elif ntype == "linear_bl" and len(curr_x) >= 2:
                                bl[:, c] = prep.baseline_linear(curr_x, curr_y[:, c], curr_x[0], curr_x[-1])
                    else:
                        if ntype == "asls":
                            bl = prep.baseline_asls(curr_y, lam=node.step_data.get('lam', 1e5), p=node.step_data.get('p', 0.001))
                        elif ntype == "airpls":
                            bl = prep.baseline_airpls(curr_y, lam=node.step_data.get('lam', 1e5), n_iter=node.step_data.get('n_iter', 15))
                        elif ntype == "poly_bl":
                            bl = prep.baseline_polynomial(curr_y, curr_x, deg=node.step_data.get('deg', 3))
                        elif ntype == "linear_bl" and len(curr_x) >= 2:
                            bl = prep.baseline_linear(curr_x, curr_y, curr_x[0], curr_x[-1])

                    corr_y = curr_y - bl
                    port_results[(node, "out")] = (curr_x.copy(), corr_y)
                    port_results[(node, "out_bl")] = (curr_x.copy(), bl)
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_bl")] = (None, None)

            # 14. VAPORFIT
            elif ntype == "vaporfit":
                data_sig = get_source_data(node.in_ports.get("in_signal"))
                p_atm = node.in_ports.get("in_atm")

                atm_sources = [c.out_port for c in p_atm.connections if
                               c.out_port and c.out_port.node_item] if p_atm else []
                atm_datasets = [port_results.get((p.node_item, p.port_id), (None, None)) for p in atm_sources]
                atm_datasets = [ds for ds in atm_datasets if ds[0] is not None]

                if data_sig[0] is not None and atm_datasets:
                    xs, ys = data_sig
                    atm_cols = []
                    for xa, ya in atm_datasets:
                        xa_s = xa[::-1] if xa[0] > xa[-1] else xa
                        if ya.ndim == 2:
                            for c in range(ya.shape[1]):
                                ya_col = ya[::-1, c] if xa[0] > xa[-1] else ya[:, c]
                                atm_cols.append(np.interp(xs, xa_s, ya_col))
                        else:
                            ya_s = ya[::-1] if xa[0] > xa[-1] else ya
                            atm_cols.append(np.interp(xs, xa_s, ya_s))
                    atm_matrix = np.column_stack(atm_cols)
                elif data_sig[0] is not None and len(session.atm_spectra) > 0:
                    xs, ys = data_sig
                    atm_matrix = session.get_atm_matrix(xs, only_visible=True)
                    if atm_matrix is None or atm_matrix.shape[1] == 0:
                        atm_matrix = session.get_atm_matrix(xs, only_visible=False)
                else:
                    atm_matrix = None

                if data_sig[0] is not None and atm_matrix is not None:
                    xs, ys = data_sig
                    sg_pts = int(node.step_data.get('sg_points', 11))
                    sg_poly = int(node.step_data.get('sg_poly', 3))

                    try:
                        if ys.ndim == 2:
                            corr_cols, atm_fit_cols = [], []
                            for c in range(ys.shape[1]):
                                cy, coeffs, _ = AtmosphericCorrectionEngine.fit_and_subtract(
                                    xs, ys[:, c], atm_matrix, sg_points=sg_pts, sg_poly=sg_poly
                                )
                                corr_cols.append(cy)
                                atm_fit_cols.append(atm_matrix @ coeffs)
                            port_results[(node, "out")] = (xs.copy(), np.column_stack(corr_cols))
                            port_results[(node, "out_atm")] = (xs.copy(), np.column_stack(atm_fit_cols))
                        else:
                            corr_y, coeffs, metrics = AtmosphericCorrectionEngine.fit_and_subtract(
                                xs, ys, atm_matrix, sg_points=sg_pts, sg_poly=sg_poly
                            )
                            node.step_data['metrics'] = metrics
                            node.step_data['coeffs'] = coeffs.tolist()
                            fitted_atm = atm_matrix @ coeffs
                            port_results[(node, "out")] = (xs.copy(), corr_y)
                            port_results[(node, "out_atm")] = (xs.copy(), fitted_atm)
                    except Exception as e:
                        logger.warning(f"Błąd kalkulacji VaporFit: {e}")
                        port_results[(node, "out")] = (xs.copy(), ys.copy())
                        port_results[(node, "out_atm")] = (xs.copy(), np.zeros_like(ys))
                elif data_sig[0] is not None:
                    port_results[(node, "out")] = (data_sig[0].copy(), data_sig[1].copy())
                    port_results[(node, "out_atm")] = (data_sig[0].copy(), np.zeros_like(data_sig[0]))
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_atm")] = (None, None)

            # 15. WEWNĘTRZNY STANDARD (IS)
            elif ntype == "internal_standard":
                data_sig = get_source_data(node.in_ports.get("in_signal"))
                data_ref = get_source_data(node.in_ports.get("in_ref"))

                if data_sig[0] is not None and data_ref[0] is not None:
                    xs, ys = data_sig
                    xr, yr = data_ref
                    target_wn = float(node.step_data.get('target_wn', 2060.0))
                    half_win = float(node.step_data.get('half_win', 15.0))
                    calc_mode = node.step_data.get('calc_mode', 'peak_height')

                    xr_s = xr[::-1] if xr[0] > xr[-1] else xr
                    yr_col = yr[:, 0] if yr.ndim == 2 else yr
                    yr_s = yr_col[::-1] if xr[0] > xr[-1] else yr_col
                    yr_interp = np.interp(xs, xr_s, yr_s)

                    mask_is = (xs >= target_wn - half_win) & (xs <= target_wn + half_win)
                    if np.any(mask_is):
                        sub_y = yr_interp[mask_is]
                        sub_x = xs[mask_is]
                        if calc_mode == 'peak_area':
                            factor = float(np.abs(_trapezoid(sub_y, sub_x)))
                        else:
                            factor = float(np.max(sub_y) - np.min(sub_y))
                        if factor < 1e-8:
                            factor = 1.0
                    else:
                        factor = 1.0

                    corr_y = ys / factor
                    port_results[(node, "out")] = (xs.copy(), corr_y)
                elif data_sig[0] is not None:
                    port_results[(node, "out")] = (data_sig[0].copy(), data_sig[1].copy())
                else:
                    port_results[(node, "out")] = (None, None)

            # 16. CAŁKOWANIE PASMA (INTEGRAL / KINETYKA)
            elif ntype == "band_integrate":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    min_w = float(node.step_data.get('min_wn', 1600.0))
                    max_w = float(node.step_data.get('max_wn', 1700.0))
                    bl_mode = node.step_data.get('bl_mode', 'two_point')

                    mask = (curr_x >= min_w) & (curr_x <= max_w)
                    if not np.any(mask):
                        mask = np.ones(len(curr_x), dtype=bool)

                    sub_x = curr_x[mask]
                    if curr_y.ndim == 2:
                        areas = []
                        n_c = curr_y.shape[1]
                        for c in range(n_c):
                            sy = curr_y[mask, c].copy()
                            if bl_mode == 'two_point' and len(sy) >= 2:
                                sy -= np.linspace(sy[0], sy[-1], len(sy))
                            elif bl_mode == 'min_offset':
                                sy -= np.min(sy)
                            areas.append(float(np.abs(_trapezoid(sy, sub_x))))
                        x_samples = np.arange(1, n_c + 1, dtype=np.float64)
                        port_results[(node, "out")] = (x_samples, np.array(areas, dtype=np.float64))
                    else:
                        sy = curr_y[mask].copy()
                        if bl_mode == 'two_point' and len(sy) >= 2:
                            sy -= np.linspace(sy[0], sy[-1], len(sy))
                        elif bl_mode == 'min_offset':
                            sy -= np.min(sy)
                        val = float(np.abs(_trapezoid(sy, sub_x)))
                        port_results[(node, "out")] = (np.array([min_w, max_w]), np.array([val, val]))
                else:
                    port_results[(node, "out")] = (None, None)

            # 17. OPERACJA DWUWIDMOWA (Z BROADCASTINGIEM 1D / 2D)
            elif ntype == "math_op":
                data_a = get_source_data(node.in_ports.get("in_a"))
                data_b = get_source_data(node.in_ports.get("in_b"))

                if data_a[0] is not None and data_b[0] is not None:
                    xa, ya = data_a
                    xb, yb = data_b
                    xb_s = xb[::-1] if xb[0] > xb[-1] else xb

                    if yb.ndim == 2:
                        yb_cols = []
                        for c in range(yb.shape[1]):
                            col_data = yb[::-1, c] if xb[0] > xb[-1] else yb[:, c]
                            yb_cols.append(np.interp(xa, xb_s, col_data))
                        yb_interp = np.column_stack(yb_cols)
                    else:
                        yb_col = yb[::-1] if xb[0] > xb[-1] else yb
                        yb_interp = np.interp(xa, xb_s, yb_col)

                    op = node.step_data.get('op', 'sub')
                    k = float(node.step_data.get('k', 1.0))

                    def _calc_op(val_a, val_b):
                        if op == 'sub': return val_a - k * val_b
                        elif op == 'add': return val_a + k * val_b
                        elif op == 'mul': return val_a * val_b
                        elif op == 'div': return np.where(np.abs(val_b) > 1e-12, val_a / val_b, 0.0)
                        return val_a - k * val_b

                    if ya.ndim == 2 and yb_interp.ndim == 1:
                        y_res = _calc_op(ya, yb_interp[:, np.newaxis])
                    elif ya.ndim == 1 and yb_interp.ndim == 2:
                        y_res = _calc_op(ya[:, np.newaxis], yb_interp)
                    elif ya.ndim == 2 and yb_interp.ndim == 2:
                        if ya.shape[1] == yb_interp.shape[1]:
                            y_res = _calc_op(ya, yb_interp)
                        elif yb_interp.shape[1] == 1:
                            y_res = _calc_op(ya, yb_interp)
                        elif ya.shape[1] == 1:
                            y_res = _calc_op(ya, yb_interp)
                        else:
                            min_c = min(ya.shape[1], yb_interp.shape[1])
                            y_res = _calc_op(ya[:, :min_c], yb_interp[:, :min_c])
                    else:
                        y_res = _calc_op(ya, yb_interp)

                    port_results[(node, "out")] = (xa.copy(), y_res)
                elif data_a[0] is not None:
                    port_results[(node, "out")] = (data_a[0].copy(), data_a[1].copy())
                else:
                    port_results[(node, "out")] = (None, None)

            # 18. TERMINAL STANDARDOWY (OUTPUT)
            elif ntype == "output":
                data_in = get_source_data(node.in_ports.get("in"))
                port_results[(node, "in")] = data_in

            # 19. TERMINAL OKNA ROBOCZEGO (OUTPUT_WINDOW) — FAN-IN
            elif ntype == "output_window":
                in_p = node.in_ports.get("in")
                all_sources = []
                if in_p and in_p.connections:
                    for conn in in_p.connections:
                        if conn.out_port and conn.out_port.node_item:
                            src_k = (conn.out_port.node_item, conn.out_port.port_id)
                            ds = port_results.get(src_k, (None, None))
                            if ds[0] is not None and ds[1] is not None:
                                all_sources.append(ds)

                if not all_sources:
                    port_results[(node, "in")] = (None, None)
                    port_results[(node, "out")] = (None, None)
                elif len(all_sources) == 1:
                    port_results[(node, "in")] = all_sources[0]
                    port_results[(node, "out")] = all_sources[0]
                else:
                    ref_x = all_sources[0][0]
                    cols = []
                    for sx, sy in all_sources:
                        sx_s = sx[::-1] if sx[0] > sx[-1] else sx
                        if sy.ndim == 2:
                            for c in range(sy.shape[1]):
                                sy_col = sy[::-1, c] if sx[0] > sx[-1] else sy[:, c]
                                cols.append(np.interp(ref_x, sx_s, sy_col))
                        else:
                            sy_s = sy[::-1] if sx[0] > sx[-1] else sy
                            cols.append(np.interp(ref_x, sx_s, sy_s))
                    combined_y = np.column_stack(cols)
                    port_results[(node, "in")] = (ref_x.copy(), combined_y)
                    port_results[(node, "out")] = (ref_x.copy(), combined_y)

            # 20. POZOSTAŁE FILTRY
            else:
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    nx, ny = apply_single_step(curr_x, curr_y, node.step_data)
                    port_results[(node, "out")] = (nx, ny)
                else:
                    port_results[(node, "out")] = (None, None)

        return port_results