"""
gui/node_editor/nodes/node_atmospheric.py
Węzeł kompensacji atmosferycznej (VaporFit) z autonomicznym dialogiem i autotuningiem SG.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QSpinBox, QPushButton, QLabel, QMessageBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from algorithms.atmospheric import AtmosphericCorrectionEngine
from core.session import session


class VaporFitNodeDialog(QDialog):
    def __init__(self, node: 'VaporFitNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Parametry kompensacji atmosferycznej (VaporFit)")
        self.resize(420, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_sg = QSpinBox()
        self.spin_sg.setRange(5, 101)
        self.spin_sg.setSingleStep(2)
        self.spin_sg.setValue(int(self.step.get('sg_points', 11)))

        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(1, 6)
        self.spin_poly.setValue(int(self.step.get('sg_poly', 3)))

        form.addRow("Szerokość okna SG (pkt):", self.spin_sg)
        form.addRow("Stopień wielomianu:", self.spin_poly)

        self.btn_opt_sg = QPushButton("🔍 Szukaj optymalnego okna SG")
        self.btn_opt_sg.clicked.connect(self._on_find_optimal_sg)
        form.addRow("", self.btn_opt_sg)

        metrics = self.step.get('metrics', {})
        if metrics:
            lbl_met = QLabel(
                f"SDV: {metrics.get('sdv_raw', 0):.2e} ➔ {metrics.get('sdv_corr', 0):.2e}\n"
                f"SSI: {metrics.get('ssi_raw', 0):.2e} ➔ {metrics.get('ssi_corr', 0):.2e}"
            )
            lbl_met.setStyleSheet("color: #a8dadc; font-size: 11px;")
            form.addRow("Metryki gładkości:", lbl_met)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_find_optimal_sg(self):
        active = session.active_spectrum
        if not active or len(active.wavenumbers) < 2:
            QMessageBox.warning(self, "Uwaga", "Wymagane jest aktywne widmo badane w projekcie.")
            return

        xs, ys = active.wavenumbers, active.intensities
        atm_matrix = session.get_atm_matrix(xs, only_visible=True)
        if atm_matrix is None or atm_matrix.shape[1] == 0:
            atm_matrix = session.get_atm_matrix(xs, only_visible=False)

        if atm_matrix is None:
            QMessageBox.warning(self, "Uwaga", "Brak wzorców atmosfery w projekcie (gałąź Wzorce ATM).")
            return

        poly = int(self.spin_poly.value())
        res = AtmosphericCorrectionEngine.find_optimal_sg_window(xs, ys, atm_matrix, sg_poly=poly)
        opt_win = res['best_window_sdv']
        self.spin_sg.setValue(opt_win)

        QMessageBox.information(
            self, "Optymalizacja SG",
            f"Wyznaczono optymalne okno SG:\n"
            f"• Wg krzywizny SDV: {res['best_window_sdv']} pkt\n"
            f"• Wg wskaźnika SSI: {res['best_window_ssi']} pkt\n\n"
            f"Wartość {opt_win} pkt została wprowadzona."
        )

    def _save_and_accept(self):
        self.step['sg_points'] = int(self.spin_sg.value())
        self.step['sg_poly'] = int(self.spin_poly.value())
        self.step['name'] = f"VaporFit (SG={self.step['sg_points']})"
        self.accept()


@register_node
class VaporFitNode(BaseNode):
    node_type = "vaporfit"
    title = "☁️ VaporFit (Para wodna i CO₂)"
    category = "Kompensacja Atmosferyczna"
    header_gradient = ("#0077b6", "#0096c7")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "VaporFit (SG=11)", "sg_points": 11, "sg_poly": 3}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in_signal", label="Syg", is_output=False, color=QColor("#ffb703"), pos_y=70),
            PortDef(port_id="in_atm", label="Atm", is_output=False, color=QColor("#fb8500"), allow_multiple=True, pos_y=112),
            PortDef(port_id="out", label="Syg", is_output=True, color=QColor("#48cae4"), pos_y=70),
            PortDef(port_id="out_atm", label="Atm", is_output=True, color=QColor("#00b4d8"), pos_y=112)
        ]

    def create_config_dialog(self, parent=None):
        return VaporFitNodeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_sig = inputs.get("in_signal", (None, None))
        in_names = input_names.get("in_signal", [])
        if data_sig[0] is None or data_sig[1] is None:
            return {"out": (None, None), "out_atm": (None, None)}, {"out": [], "out_atm": []}

        xs, ys = data_sig
        atm_sources = inputs.get("in_atm_multi", [])
        if not atm_sources and inputs.get("in_atm", (None, None))[0] is not None:
            atm_sources = [inputs["in_atm"]]

        if atm_sources:
            atm_cols = []
            for xa, ya in atm_sources:
                if xa is None or ya is None: continue
                xa_s = xa[::-1] if xa[0] > xa[-1] else xa
                if ya.ndim == 2:
                    for c in range(ya.shape[1]):
                        ya_col = ya[::-1, c] if xa[0] > xa[-1] else ya[:, c]
                        atm_cols.append(np.interp(xs, xa_s, ya_col))
                else:
                    ya_s = ya[::-1] if xa[0] > xa[-1] else ya
                    atm_cols.append(np.interp(xs, xa_s, ya_s))
            atm_matrix = np.column_stack(atm_cols) if atm_cols else None
        elif len(session.atm_spectra) > 0:
            atm_matrix = session.get_atm_matrix(xs, only_visible=True)
            if atm_matrix is None or atm_matrix.shape[1] == 0:
                atm_matrix = session.get_atm_matrix(xs, only_visible=False)
        else:
            atm_matrix = None

        if atm_matrix is None:
            return {"out": (xs.copy(), ys.copy()), "out_atm": (xs.copy(), np.zeros_like(ys))}, {
                "out": in_names, "out_atm": in_names
            }

        sg_pts = int(self.step_data.get('sg_points', 11))
        sg_poly = int(self.step_data.get('sg_poly', 3))

        try:
            if ys.ndim == 2:
                corr_cols, atm_fit_cols = [], []
                for c in range(ys.shape[1]):
                    cy, coeffs, _ = AtmosphericCorrectionEngine.fit_and_subtract(
                        xs, ys[:, c], atm_matrix, sg_points=sg_pts, sg_poly=sg_poly
                    )
                    corr_cols.append(cy)
                    atm_fit_cols.append(atm_matrix @ coeffs)
                return {
                    "out": (xs.copy(), np.column_stack(corr_cols)),
                    "out_atm": (xs.copy(), np.column_stack(atm_fit_cols))
                }, {"out": in_names, "out_atm": in_names}
            else:
                corr_y, coeffs, metrics = AtmosphericCorrectionEngine.fit_and_subtract(
                    xs, ys, atm_matrix, sg_points=sg_pts, sg_poly=sg_poly
                )
                self.step_data['metrics'] = metrics
                self.step_data['coeffs'] = coeffs.tolist()
                return {
                    "out": (xs.copy(), corr_y),
                    "out_atm": (xs.copy(), atm_matrix @ coeffs)
                }, {"out": in_names, "out_atm": in_names}
        except Exception:
            return {"out": (xs.copy(), ys.copy()), "out_atm": (xs.copy(), np.zeros_like(ys))}, {
                "out": in_names, "out_atm": in_names
            }