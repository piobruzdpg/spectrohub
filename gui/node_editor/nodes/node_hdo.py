"""
gui/node_editor/nodes/node_hdo.py
Węzły dekompozycji HDO / MC i pików syntetycznych z autonomicznymi dialogami.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
from scipy import integrate
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLabel
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from core.session import session
from algorithms.fitting.hdo_engine import (
    HDOFitEngine, gxl_peak, evaluate_total_fit, hdo_residuals
)


def _generate_synthetic_peak(x: np.ndarray, shape: str, center: float,
                             amp: float, width: float, shape_param: float = 0.5) -> np.ndarray:
    if len(x) == 0: return np.zeros_like(x)
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


class HdoSpectrumScaledDialog(QDialog):
    def __init__(self, node: 'HdoSpectrumScaledNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Widmo składowe z wagami")
        self.resize(400, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_hdo_spec = QComboBox()
        for name in session.spectra.keys():
            self.combo_hdo_spec.addItem(f"📈 {name}", name)
        cur_s = self.step.get('selected_spectrum', '')
        idx = self.combo_hdo_spec.findData(cur_s)
        if idx >= 0:
            self.combo_hdo_spec.setCurrentIndex(idx)
        form.addRow("Widmo z projektu:", self.combo_hdo_spec)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(-1000.0, 1000.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setValue(float(self.step.get('scale', 1.0)))
        form.addRow("Wstępny mnożnik skali (c):", self.spin_scale)

        self.spin_w_min = QDoubleSpinBox()
        self.spin_w_min.setRange(0.0, 50.0)
        self.spin_w_min.setDecimals(3)
        self.spin_w_min.setValue(float(self.step.get('w_min', 0.0)))
        form.addRow("Waga minimalna (Min w MC):", self.spin_w_min)

        self.spin_w_max = QDoubleSpinBox()
        self.spin_w_max.setRange(0.0, 50.0)
        self.spin_w_max.setDecimals(3)
        self.spin_w_max.setValue(float(self.step.get('w_max', 1.5)))
        form.addRow("Waga maksymalna (Max w MC):", self.spin_w_max)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        sel_s = self.combo_hdo_spec.currentData() or ""
        scale = float(self.spin_scale.value())
        w_min = float(self.spin_w_min.value())
        w_max = float(self.spin_w_max.value())
        self.step['selected_spectrum'] = sel_s
        self.step['scale'] = scale
        self.step['w_min'] = min(w_min, w_max)
        self.step['w_max'] = max(w_min, w_max)
        short_n = sel_s[:16] + "…" if len(sel_s) > 16 else sel_s
        self.step['name'] = f"{short_n} · {scale:.2f}" if sel_s else "Widmo Skalowane"
        self.accept()


class SyntheticPeakDialog(QDialog):
    def __init__(self, node: 'SyntheticPeakNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Pasmo syntetyczne")
        self.resize(400, 240)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_shape = QComboBox()
        self.combo_shape.addItem("Gauss x Lorentz (GxL)", "gxl")
        self.combo_shape.addItem("Gaussian", "gaussian")
        self.combo_shape.addItem("Lorentzian", "lorentzian")
        self.combo_shape.addItem("Voigt", "voigt")
        self.combo_shape.addItem("Pseudo-Voigt", "pseudovoigt")
        cur_shape = self.step.get('shape', 'gxl')
        idx = self.combo_shape.findData(cur_shape)
        if idx >= 0:
            self.combo_shape.setCurrentIndex(idx)
        form.addRow("Kształt pasma:", self.combo_shape)

        self.spin_center = QDoubleSpinBox()
        self.spin_center.setRange(100.0, 5000.0)
        self.spin_center.setValue(float(self.step.get('center', 1650.0)))
        form.addRow("Położenie ν₀ [cm⁻¹]:", self.spin_center)

        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.0, 100.0)
        self.spin_amp.setDecimals(4)
        self.spin_amp.setValue(float(self.step.get('amp', 0.5)))
        form.addRow("Amplituda / Wysokość:", self.spin_amp)

        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(0.5, 500.0)
        self.spin_width.setValue(float(self.step.get('width', 30.0)))
        form.addRow("Szerokość (FWHM):", self.spin_width)

        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.0, 1.0)
        self.spin_fraction.setSingleStep(0.05)
        self.spin_fraction.setValue(float(self.step.get('shape_param', 0.5)))
        form.addRow("Wsp. kształtu (0=G, 1=L):", self.spin_fraction)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        sh = self.combo_shape.currentData()
        c = float(self.spin_center.value())
        self.step['shape'] = sh
        self.step['center'] = c
        self.step['amp'] = float(self.spin_amp.value())
        self.step['width'] = float(self.spin_width.value())
        self.step['shape_param'] = float(self.spin_fraction.value())
        self.step['name'] = f"Pik {sh.upper()} @ {c:.1f}"
        self.accept()


class McParamsDialog(QDialog):
    def __init__(self, node: 'McParamsNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Parametry Monte Carlo")
        self.resize(380, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_mc_iter = QSpinBox()
        self.spin_mc_iter.setRange(100, 50000)
        self.spin_mc_iter.setSingleStep(500)
        self.spin_mc_iter.setValue(int(self.step.get('n_iter', 3000)))
        form.addRow("Liczba wariantów MC:", self.spin_mc_iter)

        self.spin_mc_pen = QDoubleSpinBox()
        self.spin_mc_pen.setRange(0.0, 500.0)
        self.spin_mc_pen.setValue(float(self.step.get('penalty_weight', 10.0)))
        form.addRow("Współczynnik asymetrycznej kary:", self.spin_mc_pen)

        self.chk_bl = QCheckBox("Linia bazowa (kwadratowa ax² + bx + c)")
        self.chk_bl.setChecked(bool(self.step.get('baseline_active', False)))
        form.addRow("Linia bazowa:", self.chk_bl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        it = int(self.spin_mc_iter.value())
        self.step['n_iter'] = it
        self.step['penalty_weight'] = float(self.spin_mc_pen.value())
        self.step['baseline_active'] = bool(self.chk_bl.isChecked())
        self.step['name'] = f"MC Params (iter={it})"
        self.accept()


class HdoDecompositionDialog(QDialog):
    def __init__(self, node: 'HdoDecompositionNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Solwer dekompozycji HDO / Monte Carlo")
        self.resize(440, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_mult_n = QDoubleSpinBox()
        self.spin_mult_n.setRange(0.01, 100.0)
        self.spin_mult_n.setValue(float(self.step.get('n_multiplier', 1.0)))
        form.addRow("Mnożnik widma badanego N:", self.spin_mult_n)

        lbl_info = QLabel(
            "🎲 Parametry optymalizacji Monte Carlo pobierane są\n"
            "z podłączonego węzła 'Parametry MC'.\n\n"
            "• W₀: Widmo badane\n"
            "• Wₖ: Widma składowe z wagami (multi-cable)\n"
            "• Pik: Piki syntetyczne GxL (multi-cable)\n"
            "• MC: Parametry Monte Carlo"
        )
        lbl_info.setStyleSheet("color: #a8dadc; font-size: 11px; padding: 6px;")
        form.addRow(lbl_info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        n_mult = float(self.spin_mult_n.value())
        self.step['n_multiplier'] = n_mult
        self.step['name'] = f"HDO Fit (N={n_mult:.2f})"
        self.accept()


@register_node
class HdoSpectrumScaledNode(BaseNode):
    node_type = "hdo_spectrum_scaled"
    title = "📈 Widmo Składowe (Wk · c)"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#1d3557", "#457b9d")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "Widmo Skalowane", "selected_spectrum": "",
            "scale": 1.0, "w_min": 0.0, "w_max": 1.5
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#457b9d"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return HdoSpectrumScaledDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sel_name = self.step_data.get('selected_spectrum', '')
        scale = float(self.step_data.get('scale', 1.0))
        spec = session.get_spectrum(sel_name)
        def_x = np.linspace(4000, 400, 1000)

        if spec:
            sx = spec.wavenumbers
            sy = spec.intensities * scale
            sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
            y_interp = np.interp(def_x, sx_s, sy_s)
            return {"out": (def_x.copy(), y_interp)}, {"out": [sel_name]}

        return {"out": (def_x.copy(), np.zeros_like(def_x))}, {"out": ["Widmo Skalowane"]}


@register_node
class SyntheticPeakNode(BaseNode):
    node_type = "synthetic_peak"
    title = "🔔 Pasmo Syntetyczne (GxL)"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#720026", "#e76f51")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "Pik GxL @ 1650", "shape": "gxl",
            "center": 1650.0, "amp": 0.5, "width": 30.0, "shape_param": 0.5
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#e76f51"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return SyntheticPeakDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        def_x = np.linspace(4000, 400, 1000)
        sh = self.step_data.get('shape', 'gxl')
        c = float(self.step_data.get('center', 1650.0))
        a = float(self.step_data.get('amp', 0.5))
        w = float(self.step_data.get('width', 30.0))
        f = float(self.step_data.get('shape_param', 0.5))
        y_peak = _generate_synthetic_peak(def_x, shape=sh, center=c, amp=a, width=w, shape_param=f)
        return {"out": (def_x.copy(), y_peak)}, {"out": [self.step_data.get('name', f"Pik {sh.upper()}")]}


@register_node
class McParamsNode(BaseNode):
    node_type = "mc_params"
    title = "🎲 Parametry Monte Carlo"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#4f000b", "#ff9f1c")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "MC Params (iter=3000)",
            "n_iter": 3000, "penalty_weight": 10.0, "baseline_active": False
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out_mc", label="MC", is_output=True, color=QColor("#ff9f1c"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return McParamsDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        return {"out_mc": (np.array([1.0]), np.array([1.0]))}, {"out_mc": ["MC"]}

    def draw_custom_overlay(self, painter, box_rect):
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor("#ff9f1c"))
        it = self.step_data.get('n_iter', 3000)
        pen = self.step_data.get('penalty_weight', 10.0)
        painter.drawText(box_rect, Qt.AlignCenter, f"MC Config\nIter: {it}\nPenalty: {pen:.1f}")


@register_node
class HdoDecompositionNode(BaseNode):
    node_type = "hdo_decomposition"
    title = "🧪 Solwer Dekompozycji HDO / MC"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#14213d", "#2a9d8f")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "HDO Fit (N=1.0)", "n_multiplier": 1.0}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in_target", label="W₀", is_output=False, color=QColor("#ffb703"), pos_y=44),
            PortDef(port_id="in_bases", label="Wₖ", is_output=False, color=QColor("#fb8500"), allow_multiple=True, pos_y=68),
            PortDef(port_id="in_peaks", label="Pik", is_output=False, color=QColor("#ff006e"), allow_multiple=True, pos_y=94),
            PortDef(port_id="in_mc", label="MC", is_output=False, color=QColor("#ff9f1c"), pos_y=120),
            PortDef(port_id="out_fit", label="Ź+Fit", is_output=True, color=QColor("#f4a261"), pos_y=34),
            PortDef(port_id="out_res", label="Res", is_output=True, color=QColor("#e63946"), pos_y=54),
            PortDef(port_id="out_fit_res", label="ΔFit", is_output=True, color=QColor("#ffd166"), pos_y=74),
            PortDef(port_id="out_comps", label="W_k", is_output=True, color=QColor("#06d6a0"), pos_y=94),
            PortDef(port_id="out_peaks", label="Pks", is_output=True, color=QColor("#ff006e"), pos_y=114),
            PortDef(port_id="out_params", label="Tab", is_output=True, color=QColor("#52b788"), pos_y=134)
        ]

    def create_config_dialog(self, parent=None):
        return HdoDecompositionDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_tar = inputs.get("in_target", (None, None))
        tar_names = input_names.get("in_target", [])
        target_name = tar_names[0] if tar_names else "W0"

        def_x = np.linspace(4000, 400, 1000)
        target_x = data_tar[0] if data_tar[0] is not None else def_x
        target_y = data_tar[1] if data_tar[1] is not None else np.zeros_like(target_x)
        target_y_1d = target_y[:, 0] if target_y.ndim == 2 else target_y

        n_multiplier = float(self.step_data.get('n_multiplier', 1.0))
        w0_target = target_y_1d * n_multiplier

        bases_sources = inputs.get("in_bases_multi", [])
        base_names_in = input_names.get("in_bases_multi", [])
        bounds_list = inputs.get("in_bases_bounds", [])

        bases_list = []
        base_names = []
        for idx, (bx, by) in enumerate(bases_sources):
            if bx is None or by is None: continue
            by_col = by[:, 0] if by.ndim == 2 else by
            bx_s, by_s = (bx[::-1], by[::-1]) if bx[0] > bx[-1] else (bx, by)
            bases_list.append(np.interp(target_x, bx_s, by_s))
            b_n = base_names_in[idx][0] if (idx < len(base_names_in) and base_names_in[idx]) else f"Comp_{idx + 1}"
            base_names.append(b_n)

        if not bounds_list or len(bounds_list) != len(bases_list):
            bounds_list = [(0.0, 1.5)] * len(bases_list)

        peaks_sources = inputs.get("in_peaks_multi", [])
        peak_params_nodes = inputs.get("in_peaks_params", [])
        n_peaks = len(peaks_sources)

        mc_cfg = inputs.get("in_mc_config", {})
        n_iterations = int(mc_cfg.get('n_iter', self.step_data.get('n_iter', 3000)))
        penalty_weight = float(mc_cfg.get('penalty_weight', self.step_data.get('penalty_weight', 10.0)))
        baseline_active = bool(mc_cfg.get('baseline_active', self.step_data.get('baseline_active', False)))

        if len(bases_list) > 0 or n_peaks > 0:
            try:
                if is_execution:
                    results = HDOFitEngine.run_monte_carlo(
                        wavenb=target_x, target=w0_target, base_spectra_list=bases_list,
                        comp_bounds=bounds_list, n_peaks=n_peaks, n_iterations=n_iterations,
                        penalty_weight=penalty_weight, st_dev=0.01, baseline_active=baseline_active,
                        progress_callback=progress_callback
                    )
                    fit_y = results['fitted_curve']
                    excess_res = results['excess_residual']
                    comps_y = np.column_stack(results['scaled_components']) if results['scaled_components'] else np.zeros_like(target_x)
                    peaks_y = np.column_stack(results['peaks_curves']) if results['peaks_curves'] else np.zeros_like(target_x)

                    full_metrics = {
                        'n_multiplier': n_multiplier, 'coefficients': results['coefficients'], 'base_names': base_names,
                        'comp_areas': results.get('comp_areas', []), 'peaks_params': results.get('peaks_params', []),
                        'peaks_area': results.get('peaks_area', 0.0), 'r2': results['r2'], 'rmse': results['rmse'],
                        'rss': results.get('rss', 0.0), 'mc_iterations': n_iterations, 'mc_penalty': penalty_weight,
                        'baseline_active': baseline_active
                    }
                    self.step_data['fit_metrics'] = full_metrics
                else:
                    k_comps = len(bases_list)
                    base_mat = np.vstack(bases_list) if k_comps > 0 else np.empty((0, len(target_x)))
                    total_p = 3 + k_comps + 4 * n_peaks
                    param_init = np.zeros(total_p, dtype=np.float64)
                    lower_fast = np.full(total_p, -np.inf)
                    upper_fast = np.full(total_p, np.inf)

                    if not baseline_active: lower_fast[0:3], upper_fast[0:3] = -1e-9, 1e-9
                    else: lower_fast[0:3], upper_fast[0:3] = -0.05, 0.05

                    for i in range(k_comps):
                        b_low, b_high = bounds_list[i]
                        b_low_clamped = max(0.0, float(b_low))
                        b_high_clamped = max(b_low_clamped + 1e-6, float(b_high))
                        lower_fast[3 + i], upper_fast[3 + i] = b_low_clamped, b_high_clamped
                        param_init[3 + i] = 0.5 * (b_low_clamped + b_high_clamped)

                    max_amp = max(float(np.max(w0_target)), 1e-3)
                    wn_min, wn_max = float(np.min(target_x)), float(np.max(target_x))
                    wn_span = max(wn_max - wn_min, 10.0)

                    start_p = 3 + k_comps
                    for j in range(n_peaks):
                        pos = start_p + j * 4
                        pn = peak_params_nodes[j] if j < len(peak_params_nodes) else {}
                        init_amp = float(pn.get('amp', 0.4 * max_amp))
                        init_center = float(pn.get('center', 0.5 * (wn_min + wn_max)))
                        init_shape = float(pn.get('shape_param', 0.5))
                        init_width = float(pn.get('width', 30.0))

                        param_init[pos] = np.clip(init_amp, 0.0, 2.5 * max_amp)
                        param_init[pos + 1] = np.clip(init_center, wn_min, wn_max)
                        param_init[pos + 2] = np.clip(init_shape, 0.0, 1.0)
                        param_init[pos + 3] = np.clip(init_width, 2.0, 0.8 * wn_span)

                        lower_fast[pos], upper_fast[pos] = 0.0, 2.5 * max(max_amp, 1e-2)
                        lower_fast[pos + 1], upper_fast[pos + 1] = wn_min, wn_max
                        lower_fast[pos + 2], upper_fast[pos + 2] = 0.0, 1.0
                        lower_fast[pos + 3], upper_fast[pos + 3] = 2.0, 0.8 * wn_span

                    res_fast = least_squares(
                        hdo_residuals, param_init,
                        args=(w0_target, target_x, base_mat, n_peaks, penalty_weight, baseline_active),
                        bounds=(lower_fast, upper_fast), max_nfev=45, ftol=1e-3, xtol=1e-3
                    )
                    fit_y = evaluate_total_fit(res_fast.x, target_x, base_mat, n_peaks, baseline_active)

                    sum_comps = np.zeros_like(target_x)
                    scaled_c, extracted_c = [], []
                    for i in range(k_comps):
                        c_val = float(res_fast.x[3 + i])
                        extracted_c.append(c_val)
                        c_curve = c_val * base_mat[i]
                        scaled_c.append(c_curve)
                        sum_comps += c_curve
                    excess_res = w0_target - sum_comps

                    p_curves, p_params = [], []
                    for j in range(n_peaks):
                        pos = 3 + k_comps + j * 4
                        p_a, p_c, p_s, p_w = float(res_fast.x[pos]), float(res_fast.x[pos + 1]), float(res_fast.x[pos + 2]), float(res_fast.x[pos + 3])
                        p_params.append({'amp': p_a, 'center': p_c, 'shape': p_s, 'width': p_w})
                        p_curves.append(gxl_peak(target_x, p_a, p_c, p_s, p_w))

                    comps_y = np.column_stack(scaled_c) if scaled_c else np.zeros_like(target_x)
                    peaks_y = np.column_stack(p_curves) if p_curves else np.zeros_like(target_x)

                    ss_res = float(np.sum((w0_target - fit_y) ** 2))
                    ss_tot = float(np.sum((w0_target - np.mean(w0_target)) ** 2))
                    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    rmse_val = float(np.sqrt(np.mean((w0_target - fit_y) ** 2)))

                    comp_areas = [abs(float(integrate.simpson(c_crv, target_x))) for c_crv in scaled_c]
                    sum_pks = np.sum(p_curves, axis=0) if p_curves else np.zeros_like(target_x)
                    peaks_area = abs(float(integrate.simpson(sum_pks, target_x))) if p_curves else 0.0

                    full_metrics = {
                        'n_multiplier': n_multiplier, 'coefficients': extracted_c, 'base_names': base_names,
                        'comp_areas': comp_areas, 'peaks_params': p_params, 'peaks_area': peaks_area,
                        'r2': r2_val, 'rmse': rmse_val, 'rss': ss_res, 'mc_iterations': n_iterations,
                        'mc_penalty': penalty_weight, 'baseline_active': baseline_active
                    }
                    self.step_data['fit_metrics'] = full_metrics

                combined_fit_y = np.column_stack([w0_target, fit_y])

                p_labels = ['MC_Iterations', 'MC_Penalty', 'Baseline_Active', 'N_multiplier', 'R2', 'RMSE', 'RSS', 'Peaks_Area']
                p_values = [
                    float(full_metrics.get('mc_iterations', n_iterations)), float(full_metrics.get('mc_penalty', penalty_weight)),
                    1.0 if full_metrics.get('baseline_active', False) else 0.0, float(full_metrics.get('n_multiplier', 1.0)),
                    float(full_metrics.get('r2', 0.0)), float(full_metrics.get('rmse', 0.0)), float(full_metrics.get('rss', 0.0)),
                    float(full_metrics.get('peaks_area', 0.0))
                ]
                for idx_c, c_val in enumerate(full_metrics.get('coefficients', [])):
                    b_lbl = base_names[idx_c] if idx_c < len(base_names) else f"Comp_{idx_c + 1}"
                    p_labels.append(f"c_{b_lbl}")
                    p_values.append(float(c_val))
                    if idx_c < len(full_metrics.get('comp_areas', [])):
                        p_labels.append(f"Area_{b_lbl}")
                        p_values.append(float(full_metrics['comp_areas'][idx_c]))

                for idx_p, p_dict in enumerate(full_metrics.get('peaks_params', [])):
                    p_labels.extend([f"Pik{idx_p + 1}_Nu0", f"Pik{idx_p + 1}_Amp", f"Pik{idx_p + 1}_Width", f"Pik{idx_p + 1}_Shape"])
                    p_values.extend([float(p_dict['center']), float(p_dict['amp']), float(p_dict['width']), float(p_dict['shape'])])

                tbl_x = np.arange(1, len(p_values) + 1, dtype=np.float64)
                tbl_y = np.array(p_values, dtype=np.float64)

                self.step_data['param_labels'] = p_labels

                return {
                    "out_fit": (target_x.copy(), combined_fit_y),
                    "out_res": (target_x.copy(), excess_res),
                    "out_fit_res": (target_x.copy(), w0_target - fit_y),
                    "out_comps": (target_x.copy(), comps_y),
                    "out_peaks": (target_x.copy(), peaks_y),
                    "out_params": (tbl_x, tbl_y)
                }, {
                    "out_fit": [f"N_{target_name}", f"Fit_{target_name}"],
                    "out_res": [f"Res_{target_name}"],
                    "out_fit_res": [f"ΔFit_{target_name}"],
                    "out_comps": base_names if base_names else [f"Comp_{i + 1}" for i in range(len(bases_list))],
                    "out_peaks": [f"Pik_{i + 1}" for i in range(n_peaks)],
                    "out_params": ["Wartosc"]
                }
            except Exception:
                pass

        comb_zero = np.column_stack([w0_target, w0_target])
        return {
            "out_fit": (target_x.copy(), comb_zero),
            "out_res": (target_x.copy(), np.zeros_like(target_x)),
            "out_fit_res": (target_x.copy(), np.zeros_like(target_x)),
            "out_comps": (target_x.copy(), np.zeros_like(target_x)),
            "out_peaks": (target_x.copy(), np.zeros_like(target_x)),
            "out_params": (np.array([1.0]), np.array([0.0]))
        }, {
            "out_fit": [f"N_{target_name}", f"Fit_{target_name}"],
            "out_res": [f"Res_{target_name}"],
            "out_fit_res": [f"ΔFit_{target_name}"],
            "out_comps": ["Comp_1"],
            "out_peaks": ["Pik_1"],
            "out_params": ["Param"]
        }

    def draw_custom_overlay(self, painter, box_rect):
        fit_metrics = self.step_data.get('fit_metrics', {})
        coeffs = fit_metrics.get('coefficients', [])
        r2 = fit_metrics.get('r2', None)
        if coeffs or r2 is not None:
            txt_parts = []
            if r2 is not None: txt_parts.append(f"R²={r2:.4f}")
            if coeffs:
                c_str = ", ".join([f"c{i+1}={c:.3f}" for i, c in enumerate(coeffs[:2])])
                if len(coeffs) > 2: c_str += "…"
                txt_parts.append(c_str)

            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#2a9d8f"))
            badge_rect = QRectF(box_rect.left() + 4, box_rect.top() + 2, box_rect.width() - 8, 26)
            painter.drawText(badge_rect, Qt.AlignLeft | Qt.AlignTop, "\n".join(txt_parts))