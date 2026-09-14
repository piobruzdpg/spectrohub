"""
gui/node_editor/nodes/node_photometry.py
Węzeł przeliczeń fotometrycznych (A, T, %T, Kubelka-Munk) z autonomicznym dialogiem.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QComboBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node


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


class PhotometryDialog(QDialog):
    def __init__(self, node: 'PhotometryNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Transformacja fotometryczna")
        self.resize(380, 150)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_photo = QComboBox()
        self.combo_photo.addItem("Absorbancja ➔ Transmisja [T]", "a_to_t")
        self.combo_photo.addItem("Transmisja [T] ➔ Absorbancja", "t_to_a")
        self.combo_photo.addItem("%Transmisji [%T] ➔ Absorbancja", "tpct_to_a")
        self.combo_photo.addItem("Absorbancja ➔ %Transmisji [%T]", "a_to_tpct")
        self.combo_photo.addItem("Kubelka-Munk DRIFTS f(R∞)", "kubelka_munk")
        self.combo_photo.addItem("Odwróć Kubelka-Munk ➔ R∞", "inv_kubelka_munk")

        cur_op = self.step.get('photo_mode', 'tpct_to_a')
        idx = self.combo_photo.findData(cur_op)
        if idx >= 0:
            self.combo_photo.setCurrentIndex(idx)
        form.addRow("Transformacja fotometryczna:", self.combo_photo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        mode = self.combo_photo.currentData()
        self.step['photo_mode'] = mode
        names = {
            'a_to_t': "A ➔ T", 't_to_a': "T ➔ A", 'tpct_to_a': "%T ➔ A",
            'a_to_tpct': "A ➔ %T", 'kubelka_munk': "Kubelka-Munk f(R∞)",
            'inv_kubelka_munk': "f(R∞) ➔ R∞"
        }
        self.step['name'] = f"Fotometria: {names.get(mode, mode)}"
        self.accept()


@register_node
class PhotometryNode(BaseNode):
    node_type = "photometry"
    title = "🔄 Konwerter Fotometryczny"
    category = "Fotometria & Konwersje"
    header_gradient = ("#007f5f", "#55a630")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Fotometria: %T ➔ A", "photo_mode": "tpct_to_a"}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#a7c957"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return PhotometryDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None)}, {"out": []}

        mode = self.step_data.get('photo_mode', 'tpct_to_a')
        res_y = apply_photometry_conversion(data_in[1], mode)
        return {"out": (data_in[0].copy(), res_y)}, {"out": list(names_in)}