"""
gui/node_editor/evaluator.py
Zunifikowany silnik obliczeniowy grafu DAG dla operacji 1D oraz macierzowych (W x N).
Węzeł 'output_window' przekazuje sygnał dalej na port wyjściowy 'out' (Pass-through / Tap).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

from core.session import session
from core.models import SpectrumData, GaussianData
from algorithms import preprocessing as prep
from algorithms.atmospheric import AtmosphericCorrectionEngine
from algorithms.chemometrics import multivariate as chem

logger = logging.getLogger("SpectroHub")


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
    """Silnik ewaluacji potoku z routingiem wieloportowym i chemometrią macierzową."""

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
    def evaluate_dag(ordered_nodes: list, input_x: np.ndarray, input_y: np.ndarray,
                     session_spectra: Dict[str, SpectrumData]) -> Dict[Tuple[Any, str], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
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

            # 1. ŹRÓDŁO DANYCH
            if ntype == "input":
                port_results[(node, "out")] = (input_x.copy(), input_y.copy())

            # 2. WIDMO REFERENCYJNE
            elif ntype == "spectrum_select":
                # Obsługa zarówno pojedynczego wyboru, jak i listy wielu wzorców
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
                    port_results[(node, "out")] = (input_x.copy(), np.zeros_like(input_x))

            # 3. CHEMOMETRIA: ODSZUMIANIE SVD / PCA RANK-K
            elif ntype == "svd_denoise":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    k = int(node.step_data.get('k', 2))
                    center = bool(node.step_data.get('center', False))

                    if curr_y.ndim == 2 and curr_y.shape[1] >= 2:
                        k_eff = min(k, curr_y.shape[0], curr_y.shape[1])
                        res = chem.run_pca(curr_y, n_components=k_eff, center=center)
                        port_results[(node, "out")] = (curr_x.copy(), res['dr'])
                        port_results[(node, "out_res")] = (curr_x.copy(), res['residuals'])
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
                        else:
                            port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_res")] = (None, None)

            # 4. CHEMOMETRIA: DEKOMPOZYCJA MCR-ALS / SPEXFA
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
                            dr = res['dr']
                            residuals = res['residuals']

                            spark_y = pure_spec[:, 0] if pure_spec.ndim == 2 else pure_spec
                            port_results[(node, "out")] = (curr_x.copy(), pure_spec if curr_y.ndim == 2 else spark_y)
                            port_results[(node, "out_recon")] = (curr_x.copy(), dr if curr_y.ndim == 2 else (dr[:, 0] if dr.ndim == 2 else dr))
                            port_results[(node, "out_res")] = (curr_x.copy(), residuals if curr_y.ndim == 2 else (residuals[:, 0] if residuals.ndim == 2 else residuals))
                        except Exception as e:
                            logger.warning(f"Błąd kalkulacji MCR: {e}")
                            port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_recon")] = (curr_x.copy(), curr_y.copy())
                            port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                    else:
                        port_results[(node, "out")] = (curr_x.copy(), curr_y.copy())
                        port_results[(node, "out_recon")] = (curr_x.copy(), curr_y.copy())
                        port_results[(node, "out_res")] = (curr_x.copy(), np.zeros_like(curr_y))
                else:
                    port_results[(node, "out")] = (None, None)
                    port_results[(node, "out_recon")] = (None, None)
                    port_results[(node, "out_res")] = (None, None)

            # 5. CHEMOMETRIA: SKALOWANIA
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

            # 6. SPLOT PASM DFT
            elif ntype == "dft_broadener":
                dft_name = node.step_data.get('dft_name', '')
                scale = float(node.step_data.get('scale_factor', 0.965))
                fwhm = float(node.step_data.get('fwhm', 12.0))
                shape = node.step_data.get('shape', 'lorentzian')

                spec = session.get_spectrum(dft_name)
                in_x, _ = get_source_data(node.in_ports.get("in"))
                grid_x = in_x if in_x is not None else input_x

                if spec and hasattr(spec, 'stick_freqs') and len(spec.stick_freqs) > 0:
                    scaled_freqs = spec.stick_freqs * scale
                    y_conv = convolve_dft_sticks(scaled_freqs, spec.stick_intens, grid_x, fwhm=fwhm, shape=shape)
                    port_results[(node, "out")] = (grid_x.copy(), y_conv)
                else:
                    port_results[(node, "out")] = (grid_x.copy(), np.zeros_like(grid_x))

            # 7. FOTOMETRIA
            elif ntype == "photometry":
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    mode = node.step_data.get('photo_mode', 'tpct_to_a')
                    res_y = apply_photometry_conversion(data_in[1], mode)
                    port_results[(node, "out")] = (data_in[0].copy(), res_y)
                else:
                    port_results[(node, "out")] = (None, None)

            # 8. LINIE BAZOWE (MULTI-OUTPUT: Syg + Baseline)
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

            # 9. VAPORFIT (MULTI-OUTPUT: Syg + Atm Fit)
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
                        # Jeśli węzeł przekazuje macierz wielu widm (2D)
                        if ya.ndim == 2:
                            for c in range(ya.shape[1]):
                                ya_col = ya[::-1, c] if xa[0] > xa[-1] else ya[:, c]
                                atm_cols.append(np.interp(xs, xa_s, ya_col))
                        else:
                            # Pojedyncze widmo (1D)
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

            # 10. OPERACJA DWUWIDMOWA (A ± k·B)
            elif ntype == "math_op":
                data_a = get_source_data(node.in_ports.get("in_a"))
                data_b = get_source_data(node.in_ports.get("in_b"))

                if data_a[0] is not None and data_b[0] is not None:
                    xa, ya = data_a
                    xb, yb = data_b
                    xb_s, yb_s = (xb[::-1], yb[::-1]) if xb[0] > xb[-1] else (xb, yb)
                    yb_interp = np.interp(xa, xb_s, yb_s)

                    op = node.step_data.get('op', 'sub')
                    k = float(node.step_data.get('k', 1.0))

                    if ya.ndim == 2:
                        yb_broad = yb_interp[:, np.newaxis]
                        if op == 'sub': y_res = ya - k * yb_broad
                        elif op == 'add': y_res = ya + k * yb_broad
                        elif op == 'mul': y_res = ya * yb_broad
                        elif op == 'div': y_res = np.where(np.abs(yb_broad) > 1e-12, ya / yb_broad, 0.0)
                        else: y_res = ya - k * yb_broad
                    else:
                        if op == 'sub': y_res = ya - k * yb_interp
                        elif op == 'add': y_res = ya + k * yb_interp
                        elif op == 'mul': y_res = ya * yb_interp
                        elif op == 'div': y_res = np.where(np.abs(yb_interp) > 1e-12, ya / yb_interp, 0.0)
                        else: y_res = ya - k * yb_interp

                    port_results[(node, "out")] = (xa.copy(), y_res)
                elif data_a[0] is not None:
                    port_results[(node, "out")] = (data_a[0].copy(), data_a[1].copy())
                else:
                    port_results[(node, "out")] = (None, None)

            # 11. TERMINALE: 'output' ORAZ 'output_window' (Z PRZEKAZYWANIEM NA WYJŚCIE 'out')
            elif ntype == "output":
                data_in = get_source_data(node.in_ports.get("in"))
                port_results[(node, "in")] = data_in

            elif ntype == "output_window":
                data_in = get_source_data(node.in_ports.get("in"))
                port_results[(node, "in")] = data_in
                port_results[(node, "out")] = data_in  # Pass-through / Checkpoint!

            # 12. POZOSTAŁE KROKI FILTRÓW
            else:
                data_in = get_source_data(node.in_ports.get("in"))
                if data_in[0] is not None:
                    curr_x, curr_y = data_in
                    nx, ny = apply_single_step(curr_x, curr_y, node.step_data)
                    port_results[(node, "out")] = (nx, ny)
                else:
                    port_results[(node, "out")] = (None, None)

        return port_results