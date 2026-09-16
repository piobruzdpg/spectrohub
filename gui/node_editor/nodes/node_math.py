"""
gui/node_editor/nodes/node_math.py
Węzły arytmetyczne, skalarne, IS oraz całkowania z autonomicznymi oknami dialogowymi,
dynamiczną aktualizacją etykiet portów i propagacją paczek danych.
"""

from typing import List, Dict, Any, Tuple, Optional
import copy
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _extract_scalar(data_tuple: Any, default_val: float) -> float:
    if data_tuple is None:
        return default_val
    if isinstance(data_tuple, (int, float, np.number)):
        return float(data_tuple)
    if isinstance(data_tuple, tuple) and len(data_tuple) >= 2:
        y = data_tuple[1]
        if y is not None:
            arr = np.asarray(y).ravel()
            if len(arr) > 0 and not np.isnan(arr[0]):
                return float(arr[0])
    return default_val


class MathOpDialog(QDialog):
    def __init__(self, node: 'MathOpNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Operacja dwuwidmowa")
        self.resize(380, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_op = QComboBox()
        self.combo_op.addItem("Odejmowanie: A - k · B", "sub")
        self.combo_op.addItem("Dodawanie: A + k · B", "add")
        self.combo_op.addItem("Mnożenie: A · B", "mul")
        self.combo_op.addItem("Dzielenie: A / B", "div")
        cur_op = self.step.get('op', 'sub')
        idx = self.combo_op.findData(cur_op)
        if idx >= 0:
            self.combo_op.setCurrentIndex(idx)

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setRange(-1000.0, 1000.0)
        self.spin_k.setDecimals(4)
        self.spin_k.setSingleStep(0.01)
        self.spin_k.setValue(float(self.step.get('k', 1.0)))

        form.addRow("Typ operacji:", self.combo_op)
        form.addRow("Współczynnik skali k (domyślny):", self.spin_k)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        op = self.combo_op.currentData()
        k = float(self.spin_k.value())
        self.step['op'] = op
        self.step['k'] = k
        op_sym = "-" if op == "sub" else ("+" if op == "add" else ("·" if op == "mul" else "/"))
        self.step['name'] = f"A {op_sym} {k:.3f}·B" if op in ('sub', 'add') else f"A {op_sym} B"
        self.accept()


class ScalarMathDialog(QDialog):
    def __init__(self, node: 'ScalarMathNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Działanie ze skalarem")
        self.resize(380, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_op = QComboBox()
        self.combo_op.addItem("Mnożenie: S · c", "mul")
        self.combo_op.addItem("Dzielenie: S / c", "div")
        self.combo_op.addItem("Dodawanie: S + c", "add")
        self.combo_op.addItem("Odejmowanie: S - c", "sub")
        cur_op = self.step.get('op', 'mul')
        idx = self.combo_op.findData(cur_op)
        if idx >= 0:
            self.combo_op.setCurrentIndex(idx)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(-1e6, 1e6)
        self.spin_val.setDecimals(4)
        self.spin_val.setSingleStep(0.1)
        self.spin_val.setValue(float(self.step.get('val', 1.0)))

        form.addRow("Działanie:", self.combo_op)
        form.addRow("Wartość skalara c (domyślna):", self.spin_val)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        op = self.combo_op.currentData()
        val = float(self.spin_val.value())
        self.step['op'] = op
        self.step['val'] = val
        op_sym = "·" if op == "mul" else ("/" if op == "div" else ("+" if op == "add" else "-"))
        self.step['name'] = f"S {op_sym} {val:.4g}"
        self.accept()


class InternalStandardDialog(QDialog):
    def __init__(self, node: 'InternalStandardNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Wewnętrzny standard (IS)")
        self.resize(400, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_target_wn = QDoubleSpinBox()
        self.spin_target_wn.setRange(100.0, 5000.0)
        self.spin_target_wn.setValue(float(self.step.get('target_wn', 2060.0)))
        form.addRow("Położenie pasma IS [cm⁻¹]:", self.spin_target_wn)

        self.spin_half_win = QDoubleSpinBox()
        self.spin_half_win.setRange(1.0, 100.0)
        self.spin_half_win.setValue(float(self.step.get('half_win', 15.0)))
        form.addRow("Tolerancja szukania (± cm⁻¹):", self.spin_half_win)

        self.combo_is_mode = QComboBox()
        self.combo_is_mode.addItem("Maksimum pasma (Wysokość piku)", "peak_height")
        self.combo_is_mode.addItem("Pole powierzchni (Całka pod pasmem)", "peak_area")
        idx = self.combo_is_mode.findData(self.step.get('calc_mode', 'peak_height'))
        if idx >= 0:
            self.combo_is_mode.setCurrentIndex(idx)
        form.addRow("Metoda wyznaczenia IS:", self.combo_is_mode)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        wn = float(self.spin_target_wn.value())
        self.step['target_wn'] = wn
        self.step['half_win'] = float(self.spin_half_win.value())
        self.step['calc_mode'] = self.combo_is_mode.currentData()
        self.step['name'] = f"IS @ {wn:.1f} cm⁻¹"
        self.accept()


class BandIntegrateDialog(QDialog):
    def __init__(self, node: 'BandIntegrateNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Całkowanie pasma")
        self.resize(400, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_int_min = QDoubleSpinBox()
        self.spin_int_min.setRange(100.0, 5000.0)
        self.spin_int_min.setValue(float(self.step.get('min_wn', 1600.0)))
        form.addRow("Dolna granica [cm⁻¹]:", self.spin_int_min)

        self.spin_int_max = QDoubleSpinBox()
        self.spin_int_max.setRange(100.0, 5000.0)
        self.spin_int_max.setValue(float(self.step.get('max_wn', 1700.0)))
        form.addRow("Górna granica [cm⁻¹]:", self.spin_int_max)

        self.combo_int_bl = QComboBox()
        self.combo_int_bl.addItem("Lokalna linia bazowa (dwa krańce)", "two_point")
        self.combo_int_bl.addItem("Całka od zera (bez korekcji)", "zero")
        self.combo_int_bl.addItem("Offset do minimum w rejonie", "min_offset")
        idx = self.combo_int_bl.findData(self.step.get('bl_mode', 'two_point'))
        if idx >= 0:
            self.combo_int_bl.setCurrentIndex(idx)
        form.addRow("Korekcja tła pasma:", self.combo_int_bl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        w1 = float(self.spin_int_min.value())
        w2 = float(self.spin_int_max.value())
        self.step['min_wn'] = min(w1, w2)
        self.step['max_wn'] = max(w1, w2)
        self.step['bl_mode'] = self.combo_int_bl.currentData()
        self.step['name'] = f"Całka [{self.step['min_wn']:.0f}–{self.step['max_wn']:.0f}]"
        self.accept()


@register_node
class MathOpNode(BaseNode):
    node_type = "math_op"
    title = "Operacja Dwuwidmowa (A ± k·B)"
    category = "Matematyka Spektralna i Skalary"
    header_gradient = ("#5a189a", "#9d4edd")
    preview_port_id = "out"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "A - 1.000·B", "op": "sub", "k": 1.0}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in_a", label="A", is_output=False, color=QColor("#ffb703"), pos_y=55),
            PortDef(port_id="in_b", label="B", is_output=False, color=QColor("#fb8500"), pos_y=88),
            PortDef(port_id="in_k", label="k", is_output=False, color=QColor("#06d6a0"), pos_y=120),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#ff70a6"), pos_y=88)
        ]

    def create_config_dialog(self, parent=None):
        return MathOpDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_a = inputs.get("in_a", (None, None))
        data_b = inputs.get("in_b", (None, None))
        data_k = inputs.get("in_k", (None, None))
        names_a = input_names.get("in_a", [])

        if data_a[0] is None:
            return {"out": (None, None)}, {"out": []}
        if data_b[0] is None:
            return {"out": (data_a[0].copy(), data_a[1].copy())}, {"out": names_a}

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

        op = self.step_data.get('op', 'sub')
        default_k = float(self.step_data.get('k', 1.0))
        k = _extract_scalar(data_k, default_k)

        # Dynamiczna aktualizacja napisu na węźle
        op_sym = "-" if op == "sub" else ("+" if op == "add" else ("·" if op == "mul" else "/"))
        if data_k[0] is not None:
            self.step_data['name'] = f"A {op_sym} {k:.3f}·B [k={k:.3f}]" if op in ('sub', 'add') else f"A {op_sym} B"
        else:
            self.step_data['name'] = f"A {op_sym} {k:.3f}·B" if op in ('sub', 'add') else f"A {op_sym} B"

        def _calc(a, b):
            if op == 'sub': return a - k * b
            if op == 'add': return a + k * b
            if op == 'mul': return a * b
            if op == 'div': return np.where(np.abs(b) > 1e-12, a / b, 0.0)
            return a - k * b

        if ya.ndim == 2 and yb_interp.ndim == 1:
            y_res = _calc(ya, yb_interp[:, np.newaxis])
        elif ya.ndim == 1 and yb_interp.ndim == 2:
            y_res = _calc(ya[:, np.newaxis], yb_interp)
        elif ya.ndim == 2 and yb_interp.ndim == 2:
            min_c = min(ya.shape[1], yb_interp.shape[1])
            y_res = _calc(ya[:, :min_c], yb_interp[:, :min_c])
        else:
            y_res = _calc(ya, yb_interp)

        return {"out": (xa.copy(), y_res)}, {"out": names_a}


@register_node
class ScalarMathNode(BaseNode):
    node_type = "scalar_math"
    title = "Działanie ze Skalarem (S · c)"
    category = "Matematyka Spektralna i Skalary"
    header_gradient = ("#4a148c", "#8e24aa")
    preview_port_id = "out"

    def __init__(self, step_data=None):
        super().__init__(step_data)
        self.runtime_bundles: Dict[str, Any] = {}

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "S · 1.0", "op": "mul", "val": 1.0}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=65),
            PortDef(port_id="in_c", label="c", is_output=False, color=QColor("#06d6a0"), pos_y=105),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#b388ff"), pos_y=85)
        ]

    def create_config_dialog(self, parent=None):
        return ScalarMathDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        data_c = inputs.get("in_c", (None, None))
        names_in = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            self.runtime_bundles = {}
            return {"out": (None, None)}, {"out": []}

        x, y = data_in
        op = self.step_data.get('op', 'mul')
        default_val = float(self.step_data.get('val', 1.0))
        val = _extract_scalar(data_c, default_val)

        # Dynamiczna aktualizacja napisu na węźle
        op_sym = "·" if op == "mul" else ("/" if op == "div" else ("+" if op == "add" else "-"))
        if data_c[0] is not None:
            self.step_data['name'] = f"S {op_sym} {val:.4g} [c={val:.4g}]"
        else:
            self.step_data['name'] = f"S {op_sym} {val:.4g}"

        def _apply_op(arr):
            if arr is None: return None
            if op == 'mul': return arr * val
            elif op == 'div': return np.where(abs(val) > 1e-12, arr / val, arr)
            elif op == 'add': return arr + val
            elif op == 'sub': return arr - val
            return arr

        y_c = _apply_op(y.copy())

        # Propagacja i skalowanie paczki HDO jeśli nadeszła z poprzednika
        in_bundle = inputs.get("in_bundle")
        if in_bundle:
            b = copy.deepcopy(in_bundle)
            b['w0_target'] = _apply_op(b.get('w0_target'))
            b['fit_y'] = _apply_op(b.get('fit_y'))
            b['excess_res'] = _apply_op(b.get('excess_res'))
            b['fit_res'] = _apply_op(b.get('fit_res'))
            b['comps_y'] = _apply_op(b.get('comps_y'))
            b['peaks_y'] = _apply_op(b.get('peaks_y'))
            tag = f"x{val:.3g}" if op == 'mul' else (f"div{val:.3g}" if op == 'div' else f"{op}{val:.3g}")
            b['folder_name'] = f"{b.get('folder_name', 'HDO')}_{tag}"
            self.runtime_bundles = {"out": b}
        else:
            self.runtime_bundles = {}

        return {"out": (x.copy(), y_c)}, {"out": list(names_in)}


@register_node
class InternalStandardNode(BaseNode):
    node_type = "internal_standard"
    title = "Wewnętrzny Standard (IS)"
    category = "Matematyka Spektralna i Skalary"
    header_gradient = ("#7b2cbf", "#9d4edd")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "IS @ 2060 cm⁻¹", "target_wn": 2060.0, "half_win": 15.0, "calc_mode": "peak_height"}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in_signal", label="Syg", is_output=False, color=QColor("#ffb703"), pos_y=70),
            PortDef(port_id="in_ref", label="IS", is_output=False, color=QColor("#fb8500"), pos_y=112),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#e0aaff"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return InternalStandardDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_sig = inputs.get("in_signal", (None, None))
        data_ref = inputs.get("in_ref", (None, None))
        names = input_names.get("in_signal", [])

        if data_sig[0] is None:
            return {"out": (None, None)}, {"out": []}
        if data_ref[0] is None:
            return {"out": (data_sig[0].copy(), data_sig[1].copy())}, {"out": names}

        xs, ys = data_sig
        xr, yr = data_ref
        t_wn = float(self.step_data.get('target_wn', 2060.0))
        hw = float(self.step_data.get('half_win', 15.0))
        mode = self.step_data.get('calc_mode', 'peak_height')

        xr_s = xr[::-1] if xr[0] > xr[-1] else xr
        yr_col = yr[:, 0] if yr.ndim == 2 else yr
        yr_s = yr_col[::-1] if xr[0] > xr[-1] else yr_col
        yr_interp = np.interp(xs, xr_s, yr_s)

        mask = (xs >= t_wn - hw) & (xs <= t_wn + hw)
        factor = 1.0
        if np.any(mask):
            sub_y, sub_x = yr_interp[mask], xs[mask]
            factor = float(np.abs(_trapezoid(sub_y, sub_x))) if mode == 'peak_area' else float(np.max(sub_y) - np.min(sub_y))
            if factor < 1e-8: factor = 1.0

        return {"out": (xs.copy(), ys / factor)}, {"out": names}


@register_node
class BandIntegrateNode(BaseNode):
    node_type = "band_integrate"
    title = "Całkowanie Pasma / Kinetyka"
    category = "Matematyka Spektralna i Skalary"
    header_gradient = ("#c1121f", "#780000")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Całka [1600–1700]", "min_wn": 1600.0, "max_wn": 1700.0, "bl_mode": "two_point"}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Area", is_output=True, color=QColor("#52b788"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return BandIntegrateDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None)}, {"out": []}

        curr_x, curr_y = data_in
        min_w = float(self.step_data.get('min_wn', 1600.0))
        max_w = float(self.step_data.get('max_wn', 1700.0))
        bl_mode = self.step_data.get('bl_mode', 'two_point')

        mask = (curr_x >= min(min_w, max_w)) & (curr_x <= max(min_w, max_w))
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
            return {"out": (x_samples, np.array(areas, dtype=np.float64))}, {"out": names}
        else:
            sy = curr_y[mask].copy()
            if bl_mode == 'two_point' and len(sy) >= 2:
                sy -= np.linspace(sy[0], sy[-1], len(sy))
            elif bl_mode == 'min_offset':
                sy -= np.min(sy)
            val = float(np.abs(_trapezoid(sy, sub_x)))
            return {"out": (np.array([min_w, max_w]), np.array([val, val]))}, {"out": names}