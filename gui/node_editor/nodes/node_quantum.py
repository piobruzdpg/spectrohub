"""
gui/node_editor/nodes/node_quantum.py
Węzeł splotu drgań teoretycznych DFT z autonomicznym oknem dialogowym.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from core.session import session


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


class DftBroadenerDialog(QDialog):
    def __init__(self, node: 'DftBroadenerNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Splot pasm obliczeń DFT")
        self.resize(420, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_dft_spec = QComboBox()
        for name, spec in session.spectra.items():
            if hasattr(spec, 'stick_freqs') and len(getattr(spec, 'stick_freqs', [])) > 0:
                self.combo_dft_spec.addItem(f"⚛️ {name}", name)

        if self.combo_dft_spec.count() == 0:
            self.combo_dft_spec.addItem("(Brak obliczeń DFT w sesji)", "")

        sel_dft = self.step.get('dft_name', '')
        idx = self.combo_dft_spec.findData(sel_dft)
        if idx >= 0:
            self.combo_dft_spec.setCurrentIndex(idx)
        form.addRow("Obliczenie DFT:", self.combo_dft_spec)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.5, 1.5)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setSingleStep(0.005)
        self.spin_scale.setValue(float(self.step.get('scale_factor', 0.965)))
        form.addRow("Czynnik skalujący częstości (σ):", self.spin_scale)

        self.spin_fwhm = QDoubleSpinBox()
        self.spin_fwhm.setRange(1.0, 100.0)
        self.spin_fwhm.setValue(float(self.step.get('fwhm', 12.0)))
        form.addRow("Szerokość połówkowa FWHM [cm⁻¹]:", self.spin_fwhm)

        self.combo_shape = QComboBox()
        self.combo_shape.addItem("Lorentz (domyślny FTIR)", "lorentzian")
        self.combo_shape.addItem("Gauss", "gaussian")
        idx_s = self.combo_shape.findData(self.step.get('shape', 'lorentzian'))
        if idx_s >= 0:
            self.combo_shape.setCurrentIndex(idx_s)
        form.addRow("Profil kształtu pasma:", self.combo_shape)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        sel_name = self.combo_dft_spec.currentData()
        scale = float(self.spin_scale.value())
        self.step['dft_name'] = sel_name
        self.step['scale_factor'] = scale
        self.step['fwhm'] = float(self.spin_fwhm.value())
        self.step['shape'] = self.combo_shape.currentData()
        self.step['name'] = f"DFT: {sel_name} (σ={scale:.3f})" if sel_name else "DFT Broadener"
        self.accept()


@register_node
class DftBroadenerNode(BaseNode):
    node_type = "dft_broadener"
    title = "⚛️ Splot pasm DFT"
    category = "Chemia Kwantowa (DFT)"
    header_gradient = ("#4361ee", "#4cc9f0")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "DFT Broadener", "dft_name": "",
            "scale_factor": 0.965, "fwhm": 12.0, "shape": "lorentzian"
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#4cc9f0"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return DftBroadenerDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        dft_name = self.step_data.get('dft_name', '')
        scale = float(self.step_data.get('scale_factor', 0.965))
        fwhm = float(self.step_data.get('fwhm', 12.0))
        shape = self.step_data.get('shape', 'lorentzian')

        spec = session.get_spectrum(dft_name)
        data_in = inputs.get("in", (None, None))
        grid_x = data_in[0] if data_in[0] is not None else np.linspace(4000, 400, 1200)

        if spec and hasattr(spec, 'stick_freqs') and len(spec.stick_freqs) > 0:
            scaled_freqs = spec.stick_freqs * scale
            y_conv = convolve_dft_sticks(scaled_freqs, spec.stick_intens, grid_x, fwhm=fwhm, shape=shape)
            return {"out": (grid_x.copy(), y_conv)}, {"out": [dft_name if dft_name else "DFT"]}

        return {"out": (grid_x.copy(), np.zeros_like(grid_x))}, {"out": ["DFT"]}