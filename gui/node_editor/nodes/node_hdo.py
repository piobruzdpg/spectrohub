"""
gui/node_editor/nodes/node_hdo.py
Zintegrowana gałąź spektralnego rozkładu pasm (curve fitting)/ Monte Carlo:
- HdoSourceSpectrumNode: Widmo źródłowe W₀ z mnożnikiem N
- HdoSpectrumScaledNode: Widmo składowe Wₖ ze skalarem c i wagami w_min/w_max
- SyntheticPeakNode: Generator pasma syntetycznego (GxL/Gauss/Lorentz)
- HdoConfigNode: Konfiguracja HDO (nazewnictwo, MC iteracje, kara, linia bazowa)
- HdoDecompositionNode: Solwer z 3 wyjściami metod (out_x, out_y, out_z)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
from scipy import integrate
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLabel, QLineEdit
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


# =============================================================================
# 1. WIDMO ŹRÓDŁOWE (W₀ · N)
# =============================================================================

class HdoSourceSpectrumDialog(QDialog):
    def __init__(self, node: 'HdoSourceSpectrumNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Widmo źródłowe W₀")
        self.resize(420, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_spec = QComboBox()
        for name in session.spectra.keys():
            self.combo_spec.addItem(f"📈 {name}", name)
        cur_s = self.step.get('selected_spectrum', '')
        idx = self.combo_spec.findData(cur_s)
        if idx >= 0:
            self.combo_spec.setCurrentIndex(idx)
        form.addRow("Widmo badane z projektu:", self.combo_spec)

        self.spin_mult_n = QDoubleSpinBox()
        self.spin_mult_n.setRange(0.01, 100.0)
        self.spin_mult_n.setDecimals(3)
        self.spin_mult_n.setSingleStep(0.05)
        self.spin_mult_n.setValue(float(self.step.get('n_multiplier', 1.0)))
        form.addRow("Mnożnik widma (N):", self.spin_mult_n)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        sel_s = self.combo_spec.currentData() or ""
        n_val = float(self.spin_mult_n.value())
        self.step['selected_spectrum'] = sel_s
        self.step['n_multiplier'] = n_val
        short_n = sel_s[:16] + "…" if len(sel_s) > 16 else sel_s
        self.step['name'] = f"W₀: {short_n} · {n_val:.2f}" if sel_s else "Widmo Źródłowe W₀"
        self.accept()


@register_node
class HdoSourceSpectrumNode(BaseNode):
    node_type = "hdo_source_spectrum"
    title = "📥 Widmo Źródłowe (W₀ · N)"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#1b4332", "#2d6a4f")
    preview_port_id = "out"  # Sparkline i canvas rysują widmo, a nie liczbę N

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Widmo Źródłowe W₀",
            "selected_spectrum": "",
            "n_multiplier": 1.0
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="W₀", is_output=True, color=QColor("#52b788"), pos_y=65),
            PortDef(port_id="out_n", label="N", is_output=True, color=QColor("#06d6a0"), pos_y=105)
        ]

    def create_config_dialog(self, parent=None):
        return HdoSourceSpectrumDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sel_name = self.step_data.get('selected_spectrum', '')
        n_mult = float(self.step_data.get('n_multiplier', 1.0))
        spec = session.get_spectrum(sel_name)

        if not spec and session.active_spectrum:
            spec = session.active_spectrum
            sel_name = spec.name

        scalar_n = (np.array([n_mult], dtype=np.float64), np.array([n_mult], dtype=np.float64))

        if spec:
            sx = spec.wavenumbers.copy()
            sy = spec.intensities.copy() * n_mult
            return {
                "out": (sx, sy),
                "out_n": scalar_n
            }, {
                "out": [sel_name],
                "out_n": [f"N={n_mult:.4g}"]
            }

        def_x = np.linspace(4000, 400, 1000)
        return {
            "out": (def_x, np.zeros_like(def_x)),
            "out_n": scalar_n
        }, {
            "out": ["W₀"],
            "out_n": [f"N={n_mult:.4g}"]
        }
# =============================================================================
# 2. WIDMO SKŁADOWE (Wₖ · c)
# =============================================================================

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
        self.spin_scale.setEnabled(False)  # <-- Blokuje edycję w oknie dialogowym
        form.addRow("Mnożnik skali (zamrożony):", self.spin_scale)

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
        scale = 1.0  # <-- Wymuszone 1.0
        w_min = float(self.spin_w_min.value())
        w_max = float(self.spin_w_max.value())
        self.step['selected_spectrum'] = sel_s
        self.step['scale'] = scale
        self.step['w_min'] = min(w_min, w_max)
        self.step['w_max'] = max(w_min, w_max)
        short_n = sel_s[:16] + "…" if len(sel_s) > 16 else sel_s
        self.step['name'] = f"{short_n} [0–{self.step['w_max']:.1f}]" if sel_s else "Widmo Skalowane"
        self.accept()


@register_node
class HdoSpectrumScaledNode(BaseNode):
    node_type = "hdo_spectrum_scaled"
    title = "📈 Widmo Składowe (Wk · c)"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#1d3557", "#457b9d")
    preview_port_id = "out"  # Sparkline i canvas rysują widmo, a nie w_max

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "Widmo Skalowane", "selected_spectrum": "",
            "scale": 1.0, "w_min": 0.0, "w_max": 15.0
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Wₖ", is_output=True, color=QColor("#457b9d"), pos_y=65),
            PortDef(port_id="out_w_max", label="w_max", is_output=True, color=QColor("#f4a261"), pos_y=105)
        ]

    def create_config_dialog(self, parent=None):
        return HdoSpectrumScaledDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sel_name = self.step_data.get('selected_spectrum', '')
        scale = float(self.step_data.get('scale', 1.0))
        w_max = float(self.step_data.get('w_max', 1.5))
        spec = session.get_spectrum(sel_name)

        scalar_wmax = (np.array([w_max], dtype=np.float64), np.array([w_max], dtype=np.float64))

        if spec:
            sx = spec.wavenumbers.copy()
            sy = spec.intensities.copy() * scale
            return {
                "out": (sx, sy),
                "out_w_max": scalar_wmax
            }, {
                "out": [sel_name],
                "out_w_max": [f"w_max={w_max:.4g}"]
            }

        def_x = np.linspace(4000, 400, 1000)
        return {
            "out": (def_x, np.zeros_like(def_x)),
            "out_w_max": scalar_wmax
        }, {
            "out": ["Widmo Skalowane"],
            "out_w_max": [f"w_max={w_max:.4g}"]
        }


# =============================================================================
# 3. PASMO SYNTETYCZNE
# =============================================================================

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
        sh = self.step_data.get('shape', 'gxl')
        c = float(self.step_data.get('center', 1650.0))
        a = float(self.step_data.get('amp', 0.5))
        w = float(self.step_data.get('width', 30.0))
        f = float(self.step_data.get('shape_param', 0.5))

        # W podglądzie rysujemy pasmo w jego własnym, naturalnym oknie (rozciągnięty dzwon)
        span = max(w * 3.5, 25.0)
        local_x = np.linspace(c + span, c - span, 200)
        y_peak = _generate_synthetic_peak(local_x, shape=sh, center=c, amp=a, width=w, shape_param=f)
        return {"out": (local_x, y_peak)}, {"out": [self.step_data.get('name', f"Pik {sh.upper()}")]}


# =============================================================================
# 4. KONFIGURACJA HDO
# =============================================================================

class HdoConfigDialog(QDialog):
    def __init__(self, node: 'HdoConfigNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Konfiguracja HDO")
        self.resize(440, 260)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.ent_base_name = QLineEdit(self.step.get('base_name', ''))
        self.ent_base_name.setPlaceholderText("Pozostaw puste dla nazwy widma źródłowego")
        form.addRow("Nazwa bazowa:", self.ent_base_name)

        self.ent_prefix = QLineEdit(self.step.get('prefix', ''))
        self.ent_prefix.setPlaceholderText("np. HDO_ (opcjonalnie)")
        form.addRow("Prefiks:", self.ent_prefix)

        self.ent_suffix = QLineEdit(self.step.get('suffix', ''))
        self.ent_suffix.setPlaceholderText("np. _MC (opcjonalnie)")
        form.addRow("Sufiks:", self.ent_suffix)

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
        b_name = self.ent_base_name.text().strip()
        self.step['base_name'] = b_name
        self.step['prefix'] = self.ent_prefix.text().strip()
        self.step['suffix'] = self.ent_suffix.text().strip()
        self.step['n_iter'] = int(self.spin_mc_iter.value())
        self.step['penalty_weight'] = float(self.spin_mc_pen.value())
        self.step['baseline_active'] = bool(self.chk_bl.isChecked())
        lbl_b = b_name if b_name else "Auto"
        self.step['name'] = f"Cfg: {lbl_b} (iter={self.step['n_iter']})"
        self.accept()


@register_node
class HdoConfigNode(BaseNode):
    node_type = "hdo_config"
    title = "⚙️ Konfiguracja HDO"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#4f000b", "#ff9f1c")
    has_preview = False  # Blokuje sparkline i napis "Brak sygnału"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Cfg: Auto (iter=3000)",
            "base_name": "",
            "prefix": "",
            "suffix": "",
            "n_iter": 3000,
            "penalty_weight": 10.0,
            "baseline_active": False
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out_cfg", label="Cfg", is_output=True, color=QColor("#ff9f1c"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return HdoConfigDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        return {"out_cfg": (np.array([1.0]), np.array([1.0]))}, {"out_cfg": ["Cfg"]}

    def draw_custom_overlay(self, painter, box_rect):
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor("#ff9f1c"))
        b_name = self.step_data.get('base_name', '') or 'Auto'
        it = self.step_data.get('n_iter', 3000)
        pen = self.step_data.get('penalty_weight', 10.0)
        painter.drawText(box_rect, Qt.AlignCenter, f"HDO Config\nBaza: {b_name}\nIter: {it} | Kara: {pen:.1f}")


# =============================================================================
# 5. GŁÓWNY SOLWER HDO (Z 3 WYJŚCIAMI METOD: OUT_X, OUT_Y, OUT_Z)
# =============================================================================

@register_node
class HdoDecompositionNode(BaseNode):
    node_type = "hdo_decomposition"
    title = "🧪 Solwer Dekompozycji HDO / MC"
    category = "Dekompozycja HDO & Pasma Syntetyczne"
    header_gradient = ("#14213d", "#2a9d8f")
    preview_port_id = "out_x"  # Miniaturka i Live Preview pobierają dane z Metody X

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Solwer HDO"}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in_target", label="W₀", is_output=False, color=QColor("#52b788"), pos_y=46),
            PortDef(port_id="in_bases", label="Wₖ", is_output=False, color=QColor("#fb8500"), allow_multiple=True, pos_y=76),
            PortDef(port_id="in_peaks", label="Pik", is_output=False, color=QColor("#ff006e"), allow_multiple=True, pos_y=104),
            PortDef(port_id="in_cfg", label="Cfg", is_output=False, color=QColor("#ff9f1c"), pos_y=130),

            PortDef(port_id="out_x", label="Met. X", is_output=True, color=QColor("#2a9d8f"), pos_y=50),
            PortDef(port_id="out_y", label="Met. Y", is_output=True, color=QColor("#48cae4"), pos_y=90),
            PortDef(port_id="out_z", label="Met. Z", is_output=True, color=QColor("#f4a261"), pos_y=130)
        ]

    def create_config_dialog(self, parent=None):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Solwer Dekompozycji HDO")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            "<b>🧪 Solwer Dekompozycji HDO / MC</b><br><br>"
            "• Parametry mnożnika <i>N</i> ustawiasz w bloczku <b>Widmo Źródłowe</b>.<br>"
            "• Nazwę bazową, iteracje MC i kary ustawiasz w bloczku <b>Konfiguracja HDO</b>.<br>"
            "• Wyjścia <b>Met. X</b>, <b>Met. Y</b> oraz <b>Met. Z</b> połącz kablem<br>"
            "  z węzłem <b>📁 Zapis do Folderu</b>, który utworzy folder i wszystkie obiekty w drzewie."
        )
        lbl.setStyleSheet("color: #e0e0e0; font-size: 11px; padding: 10px;")
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        return dlg

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_tar = inputs.get("in_target", (None, None))
        tar_names = input_names.get("in_target", [])
        target_name = tar_names[0] if tar_names else "W0"

        def_x = np.linspace(4000, 400, 1000)
        target_x = data_tar[0] if data_tar[0] is not None else def_x
        target_y = data_tar[1] if data_tar[1] is not None else np.zeros_like(target_x)
        w0_target = target_y[:, 0] if target_y.ndim == 2 else target_y

        # Konfiguracja
        cfg = inputs.get("in_cfg_config", {})
        base_name = cfg.get("base_name", "").strip() or target_name
        prefix = cfg.get("prefix", "")
        suffix = cfg.get("suffix", "")
        n_iterations = int(cfg.get("n_iter", 3000))
        penalty_weight = float(cfg.get("penalty_weight", 10.0))
        baseline_active = bool(cfg.get("baseline_active", False))

        # Składowe
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

        # Piki
        peaks_sources = inputs.get("in_peaks_multi", [])
        peak_params_nodes = inputs.get("in_peaks_params", [])
        n_peaks = len(peaks_sources)

        # Nazwa bazowa folderu: {prefix}{base_name}{suffix}_{n_peaks}p_{method}
        def _build_folder_name(method_tag: str) -> str:
            return f"{prefix}{base_name}{suffix}_{n_peaks}p_{method_tag}"

        folder_name_x = _build_folder_name("x")
        folder_name_y = _build_folder_name("y")
        folder_name_z = _build_folder_name("z")

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
                        'coefficients': results['coefficients'], 'base_names': base_names,
                        'comp_areas': results.get('comp_areas', []), 'peaks_params': results.get('peaks_params', []),
                        'peaks_area': results.get('peaks_area', 0.0), 'r2': results['r2'], 'rmse': results['rmse'],
                        'rss': results.get('rss', 0.0), 'mc_iterations': n_iterations, 'mc_penalty': penalty_weight,
                        'baseline_active': baseline_active
                    }
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
                        'coefficients': extracted_c, 'base_names': base_names,
                        'comp_areas': comp_areas, 'peaks_params': p_params, 'peaks_area': peaks_area,
                        'r2': r2_val, 'rmse': rmse_val, 'rss': ss_res, 'mc_iterations': n_iterations,
                        'mc_penalty': penalty_weight, 'baseline_active': baseline_active
                    }

                self.step_data['fit_metrics'] = full_metrics
                fit_res = w0_target - fit_y

                # Tabela parametrów
                p_labels = ['MC_Iterations', 'MC_Penalty', 'Baseline_Active', 'R2', 'RMSE', 'RSS', 'Peaks_Area']
                p_values = [
                    float(full_metrics.get('mc_iterations', n_iterations)), float(full_metrics.get('mc_penalty', penalty_weight)),
                    1.0 if full_metrics.get('baseline_active', False) else 0.0,
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

                # Pakujemy kompletny bundle wyników dla każdej metody
                def _create_bundle(method_key, f_name):
                    return {
                        'method': method_key,
                        'folder_name': f_name,
                        'target_name': target_name,
                        'wavenumbers': target_x.copy(),
                        'w0_target': w0_target.copy(),
                        'fit_y': fit_y.copy(),
                        'excess_res': excess_res.copy(),
                        'fit_res': fit_res.copy(),
                        'comps_y': comps_y.copy(),
                        'base_names': list(base_names),
                        'coeffs': list(full_metrics.get('coefficients', [])),
                        'peaks_y': peaks_y.copy(),
                        'n_peaks': n_peaks,
                        'metrics': full_metrics,
                        'table_x': tbl_x.copy(),
                        'table_y': tbl_y.copy(),
                        'param_labels': list(p_labels)
                    }

                bundle_x = _create_bundle("x", folder_name_x)
                bundle_y = _create_bundle("y", folder_name_y)
                bundle_z = _create_bundle("z", folder_name_z)

                self.runtime_bundles = {
                    "out_x": bundle_x,
                    "out_y": bundle_y,
                    "out_z": bundle_z
                }

                spark_y = np.column_stack([w0_target, fit_y])
                return {
                    "out_x": (target_x.copy(), spark_y),
                    "out_y": (target_x.copy(), spark_y),
                    "out_z": (target_x.copy(), spark_y)
                }, {
                    "out_x": [f"{folder_name_x} (W0)", f"{folder_name_x} (Fit)"],
                    "out_y": [f"{folder_name_y} (W0)", f"{folder_name_y} (Fit)"],
                    "out_z": [f"{folder_name_z} (W0)", f"{folder_name_z} (Fit)"]
                }
            except Exception:
                pass

        comb_zero = np.column_stack([w0_target, w0_target])
        return {
            "out_x": (target_x.copy(), comb_zero),
            "out_y": (target_x.copy(), comb_zero),
            "out_z": (target_x.copy(), comb_zero)
        }, {
            "out_x": [f"{folder_name_x}"],
            "out_y": [f"{folder_name_y}"],
            "out_z": [f"{folder_name_z}"]
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