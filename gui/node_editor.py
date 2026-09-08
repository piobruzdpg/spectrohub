"""
gui/node_editor.py
Wizualny edytor potoku przetwarzania opartego na grafach DAG (Node-based Pipeline).
Zawiera:
- Węzeł VaporFit z analitycznym OLS i auto-walidatorem SDV (obsługuje wiele widm atm na jednym porcie)
- Operacje wielowejściowe (A ± k·B) oraz jednowejściowe ze skalarami (S ± c, S · c, S / c)
- Reaktywne sparklines w standardzie FTIR (oś X malejąca w prawo)
- Klonowanie węzłów (Cmd/Ctrl + Drag, Cmd+D / Ctrl+D, PPM)
- Podgląd na żywo na SpectroCanvas i pełne wsparcie QUndoStack
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy
import logging

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPathItem,
    QMenu, QLabel, QComboBox, QCheckBox, QMessageBox, QToolBar,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPainterPath, QPainterPathStroker, QPen, QBrush,
    QColor, QFont, QLinearGradient
)

from core.session import session
from core.models import SpectrumData, SpectralSeries
from algorithms import preprocessing as prep
from algorithms.atmospheric import AtmosphericCorrectionEngine
from gui.workspaces.ws_preprocessing import (
    ModifySpectrumCommand, BatchPipelineCommand,
    ModifySeriesCommand, AddBatchSpectraCommand, StepParamDialog
)

logger = logging.getLogger("SpectroHub")


# ==============================================================================
# FUNKCJA POMOCNICZA: APLIKOWANIE POJEDYNCZEGO KROKU 1D
# ==============================================================================

def apply_single_step(x: np.ndarray, y: np.ndarray, step: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Aplikuje pojedynczy krok przetwarzania sekwencyjnego lub skalarnego."""
    stype = step.get('type', '')
    curr_x = x.copy()
    curr_y = y.copy()

    try:
        if stype == 'crop':
            curr_x, curr_y = prep.crop_range(curr_x, curr_y, step.get('min_w', 1000.0), step.get('max_w', 1800.0))
        elif stype == 'asls':
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
            if op == 'mul':
                curr_y = curr_y * val
            elif op == 'div':
                curr_y = np.where(abs(val) > 1e-12, curr_y / val, curr_y)
            elif op == 'add':
                curr_y = curr_y + val
            elif op == 'sub':
                curr_y = curr_y - val
    except Exception as err:
        logger.warning(f"Błąd podczas kalkulacji kroku {stype}: {err}")

    return curr_x, curr_y


# ==============================================================================
# DIALOG PARAMETRÓW WĘZŁA
# ==============================================================================

class NodeParamDialog(QDialog):
    """Uniwersalny dialog konfiguracji węzłów matematycznych, VaporFit i referencyjnych."""

    def __init__(self, node_item: 'NodeItem', parent=None):
        super().__init__(parent)
        self.node = node_item
        self.step = node_item.step_data
        self.setWindowTitle(f"Parametry węzła: {node_item.title}")
        self.resize(380, 220)

        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        stype = self.node.node_type

        if stype == 'math_op':
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

            self.form.addRow("Typ operacji:", self.combo_op)
            self.form.addRow("Współczynnik skali k:", self.spin_k)

        elif stype == 'scalar_math':
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

            self.form.addRow("Działanie:", self.combo_op)
            self.form.addRow("Wartość skalara (c):", self.spin_val)

        elif stype == 'vaporfit':
            self.spin_sg = QSpinBox()
            self.spin_sg.setRange(5, 101)
            self.spin_sg.setSingleStep(2)
            self.spin_sg.setValue(int(self.step.get('sg_points', 11)))

            self.spin_poly = QSpinBox()
            self.spin_poly.setRange(1, 6)
            self.spin_poly.setValue(int(self.step.get('sg_poly', 3)))

            self.combo_sign = QComboBox()
            self.combo_sign.addItem("Auto (Wygładź)", "auto")
            self.combo_sign.addItem("Odejmij (-)", "subtract")
            self.combo_sign.addItem("Dodaj (+)", "add")
            cur_sign = self.step.get('sign_mode', 'auto')
            idx = self.combo_sign.findData(cur_sign)
            if idx >= 0:
                self.combo_sign.setCurrentIndex(idx)

            self.form.addRow("Szerokość okna SG (pkt):", self.spin_sg)
            self.form.addRow("Stopień wielomianu:", self.spin_poly)
            self.form.addRow("Kierunek korekty:", self.combo_sign)

            self.btn_opt_sg = QPushButton("🔍 Szukaj optymalnego okna SG")
            self.btn_opt_sg.clicked.connect(self._on_find_optimal_sg)
            self.form.addRow("", self.btn_opt_sg)

            metrics = self.step.get('metrics', {})
            if metrics:
                lbl_met = QLabel(
                    f"SDV: {metrics.get('sdv_raw', 0):.2e} ➔ {metrics.get('sdv_corr', 0):.2e}\n"
                    f"SSI: {metrics.get('ssi_raw', 0):.2e} ➔ {metrics.get('ssi_corr', 0):.2e}"
                )
                lbl_met.setStyleSheet("color: #a8dadc; font-size: 11px;")
                self.form.addRow("Metryki gładkości:", lbl_met)

        elif stype == 'spectrum_select':
            self.combo_spec = QComboBox()

            # 1. Wzorce Atmosfery (H2O / CO2)
            if session.atm_spectra:
                self.combo_spec.addItem("── WZORCE ATMOSFERY (ATM / CO₂) ──", "")
                self.combo_spec.model().item(self.combo_spec.count() - 1).setEnabled(False)
                for name in session.atm_spectra.keys():
                    self.combo_spec.addItem(f"☁️ {name}", name)

            # 2. Pomiary Eksperymentalne
            if session.spectra:
                self.combo_spec.addItem("── POMIARY EKSPERYMENTALNE ──", "")
                self.combo_spec.model().item(self.combo_spec.count() - 1).setEnabled(False)
                for name in session.spectra.keys():
                    self.combo_spec.addItem(f"📈 {name}", name)

            if self.combo_spec.count() == 0:
                self.combo_spec.addItem("(Brak widm w projekcie)", "")

            cur_sel = self.step.get('selected_spectrum', '')
            idx = self.combo_spec.findData(cur_sel)
            if idx >= 0:
                self.combo_spec.setCurrentIndex(idx)

            self.form.addRow("Widmo odniesienia:", self.combo_spec)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_find_optimal_sg(self):
        p_sig = self.node.in_ports.get("in_signal")
        p_atm = self.node.in_ports.get("in_atm")

        src_sig = p_sig.connections[0].out_port.node_item if (p_sig and p_sig.connections) else None
        atm_sources = [c.out_port.node_item for c in p_atm.connections if c.out_port and c.out_port.node_item] if p_atm else []

        if not src_sig or src_sig.preview_x is None or not atm_sources:
            QMessageBox.warning(self, "Uwaga", "Podłącz najpierw widmo badane (Syg) i co najmniej jedno widmo atmosfery (Atm).")
            return

        xs, ys = src_sig.preview_x, src_sig.preview_y
        atm_cols = []
        for s in atm_sources:
            if s.preview_x is not None and s.preview_y is not None:
                xa, ya = s.preview_x, s.preview_y
                if xa[0] > xa[-1]:
                    xa, ya = xa[::-1], ya[::-1]
                atm_cols.append(np.interp(xs, xa, ya))

        if not atm_cols and len(session.atm_spectra) > 0:
            atm_matrix = session.get_atm_matrix(xs, only_visible=True)
            if atm_matrix is None:
                atm_matrix = session.get_atm_matrix(xs, only_visible=False)
        elif atm_cols:
            atm_matrix = np.column_stack(atm_cols)
        else:
            atm_matrix = None

        if atm_matrix is None:
            QMessageBox.warning(self, "Uwaga", "Brak wzorców atmosfery w projekcie lub na wejściu węzła.")
            return

        atm_matrix = np.column_stack(atm_cols)
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
        stype = self.node.node_type
        if stype == 'math_op':
            op = self.combo_op.currentData()
            k = float(self.spin_k.value())
            self.step['op'] = op
            self.step['k'] = k
            op_sym = "-" if op == "sub" else ("+" if op == "add" else ("·" if op == "mul" else "/"))
            self.step['name'] = f"A {op_sym} {k:.3f}·B" if op in ('sub', 'add') else f"A {op_sym} B"
        elif stype == 'scalar_math':
            op = self.combo_op.currentData()
            val = float(self.spin_val.value())
            self.step['op'] = op
            self.step['val'] = val
            op_sym = "·" if op == "mul" else ("/" if op == "div" else ("+" if op == "add" else "-"))
            self.step['name'] = f"S {op_sym} {val:.4g}"
        elif stype == 'vaporfit':
            self.step['sg_points'] = int(self.spin_sg.value())
            self.step['sg_poly'] = int(self.spin_poly.value())
            self.step['sign_mode'] = self.combo_sign.currentData()
            self.step['name'] = f"VaporFit (SG={self.step['sg_points']})"
        elif stype == 'spectrum_select':
            sel_name = self.combo_spec.currentData()
            self.step['selected_spectrum'] = sel_name
            if sel_name:
                icon = "☁️" if sel_name in session.atm_spectra else "📈"
                self.step['name'] = f"{icon} {sel_name}"
            else:
                self.step['name'] = "Ref: (brak)"
        self.accept()


# ==============================================================================
# WARSTWA PREZENTACJI GRAPHICS VIEW: PORTY, POŁĄCZENIA I WĘZŁY
# ==============================================================================

class PortItem(QGraphicsItem):
    """Pojedynczy port węzła z identyfikatorem nazwy."""
    RADIUS = 6.0

    def __init__(self, is_output: bool, node_item: 'NodeItem', port_id: str, label: str = ""):
        super().__init__(node_item)
        self.is_output = is_output
        self.node_item = node_item
        self.port_id = port_id
        self.label = label
        self.connections: List['ConnectionItem'] = []
        self.setAcceptHoverEvents(True)
        self.is_hovered = False

    def boundingRect(self) -> QRectF:
        r = self.RADIUS + 3
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        if self.is_output:
            col = QColor("#48cae4")
        elif "atm" in self.port_id or "b" in self.port_id:
            col = QColor("#fb8500")
        else:
            col = QColor("#ffb703")

        if self.is_hovered:
            col = col.lighter(130)

        painter.setBrush(QBrush(col))
        painter.setPen(QPen(QColor("#1e1e24"), 1.5))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def get_scene_center(self) -> QPointF:
        return self.scenePos()


class ConnectionItem(QGraphicsPathItem):
    """Krzywa Béziera łącząca port wyjściowy z portem wejściowym."""

    def __init__(self, out_port: PortItem, in_port: Optional[PortItem] = None):
        super().__init__()
        self.out_port = out_port
        self.in_port = in_port
        self.end_point: Optional[QPointF] = None

        self.setZValue(-1)
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.is_hovered = False

        out_port.connections.append(self)
        if in_port:
            in_port.connections.append(self)
        self.update_path()

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        stroker.setCapStyle(Qt.RoundCap)
        return stroker.createStroke(self.path())

    def update_path(self):
        p1 = self.out_port.get_scene_center()
        p2 = self.in_port.get_scene_center() if self.in_port else self.end_point

        if not p2:
            return

        dx = max(abs(p2.x() - p1.x()) * 0.5, 35.0)
        c1 = QPointF(p1.x() + dx, p1.y())
        c2 = QPointF(p2.x() - dx, p2.y())

        path = QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        if self.isSelected():
            pen = QPen(QColor("#ffb703"), 3.5, Qt.SolidLine, Qt.RoundCap)
        elif self.is_hovered:
            pen = QPen(QColor("#48cae4"), 3.0, Qt.SolidLine, Qt.RoundCap)
        else:
            pen = QPen(QColor("#00b4d8"), 2.2, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        act_del = menu.addAction("✕ Rozłącz połączenie")
        action = menu.exec(event.screenPos())
        if action == act_del:
            sc = self.scene()
            self.disconnect()
            if sc and hasattr(sc, 'dialog_owner'):
                sc.dialog_owner.recalculate_graph_preview()

    def disconnect(self):
        if self.out_port and self in self.out_port.connections:
            self.out_port.connections.remove(self)
        if self.in_port and self in self.in_port.connections:
            self.in_port.connections.remove(self)
        if self.scene():
            self.scene().removeItem(self)


class NodeItem(QGraphicsItem):
    """Wizualna karta węzła ze sparkline'em FTIR i obsługą wielu portów."""
    WIDTH = 195
    HEIGHT = 130

    def __init__(self, node_type: str, title: str, step_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.node_type = node_type
        self.title = title
        self.step_data = step_data or {'type': node_type, 'name': title}

        self.preview_x: Optional[np.ndarray] = None
        self.preview_y: Optional[np.ndarray] = None

        self.in_ports: Dict[str, PortItem] = {}
        self.out_ports: Dict[str, PortItem] = {}

        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )

        self._setup_ports()

    def _setup_ports(self):
        if self.node_type == "math_op":
            p_a = PortItem(is_output=False, node_item=self, port_id="in_a", label="A")
            p_a.setPos(0, 62)
            self.in_ports["in_a"] = p_a

            p_b = PortItem(is_output=False, node_item=self, port_id="in_b", label="B")
            p_b.setPos(0, 98)
            self.in_ports["in_b"] = p_b
        elif self.node_type == "vaporfit":
            p_sig = PortItem(is_output=False, node_item=self, port_id="in_signal", label="Syg")
            p_sig.setPos(0, 62)
            self.in_ports["in_signal"] = p_sig

            p_atm = PortItem(is_output=False, node_item=self, port_id="in_atm", label="Atm")
            p_atm.setPos(0, 98)
            self.in_ports["in_atm"] = p_atm
        elif self.node_type not in ("input", "spectrum_select"):
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In")
            p_in.setPos(0, 82)
            self.in_ports["in"] = p_in

        if self.node_type != "output":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Out")
            p_out.setPos(self.WIDTH, 82)
            self.out_ports["out"] = p_out

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, self.HEIGHT)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.boundingRect()
        bg_color = QColor("#22222b")
        border_color = QColor("#f77f00") if self.isSelected() else QColor("#3d5a80")

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 2 if self.isSelected() else 1))
        painter.drawRoundedRect(rect, 8, 8)

        header_rect = QRectF(0, 0, self.WIDTH, 24)
        grad = QLinearGradient(0, 0, self.WIDTH, 0)
        if self.node_type == "input":
            grad.setColorAt(0, QColor("#1b4332"))
            grad.setColorAt(1, QColor("#2d6a4f"))
        elif self.node_type == "output":
            grad.setColorAt(0, QColor("#6a040f"))
            grad.setColorAt(1, QColor("#9d0208"))
        elif self.node_type == "math_op":
            grad.setColorAt(0, QColor("#5a189a"))
            grad.setColorAt(1, QColor("#9d4edd"))
        elif self.node_type == "scalar_math":
            grad.setColorAt(0, QColor("#4a148c"))
            grad.setColorAt(1, QColor("#8e24aa"))
        elif self.node_type == "vaporfit":
            grad.setColorAt(0, QColor("#0077b6"))
            grad.setColorAt(1, QColor("#0096c7"))
        elif self.node_type == "spectrum_select":
            grad.setColorAt(0, QColor("#0f4c5c"))
            grad.setColorAt(1, QColor("#e36414"))
        else:
            grad.setColorAt(0, QColor("#1e3d59"))
            grad.setColorAt(1, QColor("#17b978"))

        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 8, 8)
        painter.fillPath(header_path, QBrush(grad))

        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(8, 2, self.WIDTH - 16, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor("#a8dadc"))
        desc = self.step_data.get('name', '')
        if len(desc) > 30:
            desc = desc[:28] + "…"
        painter.drawText(QRectF(8, 26, self.WIDTH - 16, 16), Qt.AlignLeft | Qt.AlignVCenter, desc)

        if self.node_type == "math_op":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(10, 52, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "A")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(10, 88, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "B")
        elif self.node_type == "vaporfit":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(9, 52, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(9, 88, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Atm")

        box_left = 32 if self.node_type in ("math_op", "vaporfit") else 10
        box_w = self.WIDTH - box_left - 10
        box_rect = QRectF(box_left, 46, box_w, 72)

        painter.setBrush(QBrush(QColor("#141419")))
        painter.setPen(QPen(QColor("#343a40"), 1))
        painter.drawRoundedRect(box_rect, 4, 4)

        if self.preview_y is not None and len(self.preview_y) > 1:
            y_data = self.preview_y.copy()

            if self.preview_x is not None and len(self.preview_x) == len(y_data):
                if self.preview_x[0] < self.preview_x[-1]:
                    y_data = y_data[::-1]

            y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
            y_rng = y_max - y_min
            if y_rng < 1e-12:
                y_rng = 1.0

            n_pts = len(y_data)
            step = max(1, n_pts // 140)
            sampled_indices = np.arange(0, n_pts, step)

            x_coords = box_left + 3 + (sampled_indices / (n_pts - 1)) * (box_w - 6)
            y_norm = (y_data[sampled_indices] - y_min) / y_rng
            y_coords = (46 + 72 - 4) - y_norm * (72 - 8)

            path = QPainterPath()
            path.moveTo(x_coords[0], y_coords[0])
            for i in range(1, len(sampled_indices)):
                path.lineTo(x_coords[i], y_coords[i])

            if self.node_type == "input":
                curve_col = QColor("#52b788")
            elif self.node_type == "output":
                curve_col = QColor("#ff595e")
            elif self.node_type == "math_op":
                curve_col = QColor("#ff70a6")
            elif self.node_type == "scalar_math":
                curve_col = QColor("#b388ff")
            elif self.node_type == "vaporfit":
                curve_col = QColor("#48cae4")
            else:
                curve_col = QColor("#00f5d4")

            painter.setPen(QPen(curve_col, 1.3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        else:
            font_italic = QFont("Arial", 8)
            font_italic.setItalic(True)
            painter.setFont(font_italic)
            painter.setPen(QColor("#6c757d"))
            painter.drawText(box_rect, Qt.AlignCenter, "Brak sygnału")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            all_ports = list(self.in_ports.values()) + list(self.out_ports.values())
            for p in all_ports:
                for conn in p.connections:
                    conn.update_path()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if self.node_type in ("input", "output"):
            super().mouseDoubleClickEvent(event)
            return

        if self.node_type in ("math_op", "scalar_math", "vaporfit", "spectrum_select"):
            dlg = NodeParamDialog(self)
        else:
            dlg = StepParamDialog(self.step_data)

        if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
            self.scene().dialog_owner.recalculate_graph_preview()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        if self.node_type in ("input", "output"):
            return
        menu = QMenu()
        act_dup = menu.addAction("📋 Duplikuj węzeł (⌘D / Ctrl+D)")
        act_edit = menu.addAction("✏️ Edytuj parametry...")
        act_del = menu.addAction("✕ Usuń węzeł (Del)")

        action = menu.exec(event.screenPos())
        if action == act_dup:
            if self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.duplicate_node(self)
        elif action == act_edit:
            if self.node_type in ("math_op", "scalar_math", "vaporfit", "spectrum_select"):
                dlg = NodeParamDialog(self)
            else:
                dlg = StepParamDialog(self.step_data)
            if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.recalculate_graph_preview()
        elif action == act_del:
            sc = self.scene()
            for p in list(self.in_ports.values()) + list(self.out_ports.values()):
                for c in list(p.connections):
                    c.disconnect()
            sc.removeItem(self)
            if hasattr(sc, 'dialog_owner'):
                sc.dialog_owner.recalculate_graph_preview()


# ==============================================================================
# SCENA I WIDOK GRAFU (Z OBSŁUGĄ CMD/CTRL + DRAG DUPLICATION)
# ==============================================================================

class NodeScene(QGraphicsScene):
    """Scena z obsługą przeciągania, menu pod PPM, skrótu Del i klonowania Cmd/Ctrl."""

    def __init__(self, dialog_owner: 'NodePipelineDialog', parent=None):
        super().__init__(parent)
        self.dialog_owner = dialog_owner
        self.temp_connection: Optional[ConnectionItem] = None
        self.setSceneRect(-2000, -2000, 4000, 4000)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())

        if isinstance(item, PortItem) and item.is_output:
            self.temp_connection = ConnectionItem(out_port=item)
            self.temp_connection.end_point = event.scenePos()
            self.addItem(self.temp_connection)
            return

        if event.button() == Qt.LeftButton and (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            node = item if isinstance(item, NodeItem) else (item.parentItem() if (item and isinstance(item.parentItem(), NodeItem)) else None)
            if isinstance(node, NodeItem) and node.node_type not in ("input", "output") and not isinstance(item, PortItem):
                clone = NodeItem(node.node_type, node.title, copy.deepcopy(node.step_data))
                clone.setPos(node.pos())
                self.addItem(clone)
                self.clearSelection()
                clone.setSelected(True)
                clone.setZValue(node.zValue() + 1.0)
                super().mousePressEvent(event)
                self.dialog_owner.recalculate_graph_preview()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_connection:
            self.temp_connection.end_point = event.scenePos()
            self.temp_connection.update_path()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.temp_connection:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if (isinstance(item, PortItem) and not item.is_output and
                    item.node_item != self.temp_connection.out_port.node_item):
                # Zezwalamy na wiele połączeń tylko dla portu in_atm w VaporFit
                if item.connections and item.port_id != "in_atm":
                    item.connections[0].disconnect()
                self.temp_connection.in_port = item
                item.connections.append(self.temp_connection)
                self.temp_connection.update_path()
                self.temp_connection = None
                self.dialog_owner.recalculate_graph_preview()
            else:
                self.temp_connection.disconnect()
                self.temp_connection = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.dialog_owner._delete_selected()
            event.accept()
            return

        if event.key() == Qt.Key_D and (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            selected_nodes = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
            for node in selected_nodes:
                self.dialog_owner.duplicate_node(node)
            event.accept()
            return

        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if item is None:
            menu = QMenu()
            menu_title = menu.addAction("➕ Dodaj węzeł:")
            menu_title.setEnabled(False)
            menu.addSeparator()
            self.dialog_owner._populate_node_menu(menu, pos=event.scenePos())
            menu.exec(event.screenPos())
            event.accept()
            return
        super().contextMenuEvent(event)


class NodeView(QGraphicsView):
    """Widok grafu ze skalowaniem kółkiem myszy."""

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor("#18181e")))

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor("#252530"), 1))

        grid_step = 40
        left = int(rect.left()) - (int(rect.left()) % grid_step)
        top = int(rect.top()) - (int(rect.top()) % grid_step)

        lines = []
        for x in range(left, int(rect.right()), grid_step):
            lines.append((QPointF(x, rect.top()), QPointF(x, rect.bottom())))
        for y in range(top, int(rect.bottom()), grid_step):
            lines.append((QPointF(rect.left(), y), QPointF(rect.right(), y)))

        for p1, p2 in lines:
            painter.drawLine(p1, p2)


# ==============================================================================
# GŁÓWNE OKNO DIALOGOWE GRAFU (NODE PIPELINE DIALOG)
# ==============================================================================

class NodePipelineDialog(QDialog):
    """Kompletny edytor potoku węzłowego w architekturze DAG."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Potok Przetwarzania Węzłowego (DAG Node Engine) — Wersja Beta")
        self.resize(1200, 750)

        self.scene = NodeScene(dialog_owner=self, parent=self)
        self.view = NodeView(self.scene, self)
        self.scene.selectionChanged.connect(self._on_selection_changed)

        self._init_ui()
        self._spawn_default_pipeline()
        self.recalculate_graph_preview()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QToolBar(self)
        toolbar.setStyleSheet("background-color: #252530; spacing: 8px; padding: 4px;")

        btn_add = QPushButton("➕ Dodaj Węzeł...")
        btn_add.setStyleSheet("background-color: #3d5a80; padding: 5px 12px;")
        menu_nodes = QMenu(self)
        self._populate_node_menu(menu_nodes)
        btn_add.setMenu(menu_nodes)
        toolbar.addWidget(btn_add)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Cel potoku: "))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems([
            "Tylko aktywne widmo",
            "Wszystkie widoczne widma w drzewie",
            "Aktywna seria 2D / Tensor"
        ])
        self.combo_scope.currentIndexChanged.connect(lambda: self.recalculate_graph_preview())
        toolbar.addWidget(self.combo_scope)

        self.chk_inplace = QCheckBox("In-place (z Undo)")
        self.chk_inplace.setChecked(True)
        toolbar.addWidget(self.chk_inplace)

        toolbar.addSeparator()

        self.chk_live_preview = QCheckBox("👁 Live Preview na Canvasie")
        self.chk_live_preview.setChecked(True)
        self.chk_live_preview.setStyleSheet("color: #ffb703; font-weight: bold;")
        self.chk_live_preview.toggled.connect(self._toggle_live_preview)
        toolbar.addWidget(self.chk_live_preview)

        toolbar.addSeparator()

        btn_del = QPushButton("✕ Usuń (Del)")
        btn_del.clicked.connect(self._delete_selected)
        toolbar.addWidget(btn_del)

        btn_clear = QPushButton("Reset grafu")
        btn_clear.clicked.connect(self._clear_graph)
        toolbar.addWidget(btn_clear)

        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        toolbar.addWidget(spacer)

        btn_run = QPushButton("⚡ Uruchom Potok Grafu")
        btn_run.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 6px 14px;")
        btn_run.clicked.connect(self._execute_pipeline)
        toolbar.addWidget(btn_run)

        layout.addWidget(toolbar)
        layout.addWidget(self.view)

    def _populate_node_menu(self, menu: QMenu, pos: Optional[QPointF] = None):
        categories = {
            "Kompensacja Atmosferyczna": [
                ("VaporFit (Para wodna i CO₂)", {'type': 'vaporfit', 'name': "VaporFit (SG=11)", 'sg_points': 11, 'sg_poly': 3, 'sign_mode': "auto"}),
                ("Widmo Odniesienia z Projektu", {'type': 'spectrum_select', 'name': "Ref: (wybierz)", 'selected_spectrum': ""})
            ],
            "Matematyka Spektralna i Skalary": [
                ("Operacja Dwuwidmowa (A ± k·B)", {'type': 'math_op', 'name': "A - 1.000·B", 'op': 'sub', 'k': 1.0}),
                ("Mnożenie przez skalar (S · c)", {'type': 'scalar_math', 'name': "S · 2", 'op': 'mul', 'val': 2.0}),
                ("Dzielenie przez skalar (S / c)", {'type': 'scalar_math', 'name': "S / 2", 'op': 'div', 'val': 2.0}),
                ("Dodawanie stałej (S + c)", {'type': 'scalar_math', 'name': "S + 0.1", 'op': 'add', 'val': 0.1}),
                ("Odejmowanie stałej (S - c)", {'type': 'scalar_math', 'name': "S - 0.1", 'op': 'sub', 'val': 0.1})
            ],
            "Korekcja Linii Bazowej": [
                ("AsLS", {'type': 'asls', 'name': "AsLS (λ=1e5, p=0.001)", 'lam': 1e5, 'p': 0.001}),
                ("airPLS / airALS", {'type': 'airpls', 'name': "airPLS (λ=1e5)", 'lam': 1e5, 'n_iter': 15}),
                ("Wielomianowa", {'type': 'poly_bl', 'name': "Wielomian st. 3", 'deg': 3}),
                ("Dwa krańce", {'type': 'linear_bl', 'name': "Linia bazowa (krańce)"})
            ],
            "Filtry i Pochodne": [
                ("Wygładzanie SG", {'type': 'sg_smooth', 'name': "SG Wygładzanie (okno=11, st=3)", 'win': 11, 'poly': 3}),
                ("1. Pochodna SG", {'type': 'deriv1', 'name': "1. Pochodna SG (okno=11)", 'win': 11, 'poly': 3}),
                ("2. Pochodna (-) SG", {'type': 'deriv2', 'name': "2. Pochodna SG (-) (okno=11)", 'win': 11, 'poly': 3}),
                ("FSD (Kauppinen)", {'type': 'fsd', 'name': "FSD (FWHM=18.0, K=2.0)", 'fwhm': 18.0, 'k_factor': 2.0})
            ],
            "Przycinanie i Zakres": [
                ("Przycięcie (Crop)", {'type': 'crop', 'name': "Przycięcie [1000–1800]", 'min_w': 1000.0, 'max_w': 1800.0})
            ],
            "Normalizacje i Skalowania": [
                ("SNV", {'type': 'snv', 'name': "Normalizacja SNV"}),
                ("Min-Max [0, 1]", {'type': 'minmax', 'name': "Normalizacja Min-Max"}),
                ("Norma L₁ (Pole)", {'type': 'l1', 'name': "Normalizacja L₁ (Pole)"}),
                ("Norma L₂", {'type': 'l2', 'name': "Normalizacja L₂"}),
                ("Offset (Min ➔ 0.0)", {'type': 'offset_min', 'name': "Offset (Min ➔ 0.0)"}),
                ("Offset w punkcie ν", {'type': 'offset_point', 'name': "Offset w punkcie 2000.0 cm⁻¹", 'target_wn': 2000.0})
            ]
        }

        for cat_name, items in categories.items():
            sub = menu.addMenu(cat_name)
            for label, data in items:
                act = sub.addAction(label)
                act.triggered.connect(lambda chk=False, d=data, l=label, p=pos: self._add_processing_node(l, d, p))

    def _add_processing_node(self, title: str, step_data: Dict[str, Any], pos: Optional[QPointF] = None):
        node = NodeItem(node_type=step_data['type'], title=title, step_data=copy.deepcopy(step_data))
        if pos is not None:
            node.setPos(pos)
        else:
            node.setPos(self.view.mapToScene(self.view.viewport().rect().center()) + QPointF(np.random.randint(-40, 40), np.random.randint(-40, 40)))
        self.scene.addItem(node)
        self.recalculate_graph_preview()

    def duplicate_node(self, node: NodeItem, offset: QPointF = QPointF(35, 35)) -> Optional[NodeItem]:
        if node.node_type in ("input", "output"):
            return None
        clone = NodeItem(node.node_type, node.title, copy.deepcopy(node.step_data))
        clone.setPos(node.pos() + offset)
        self.scene.addItem(clone)
        self.scene.clearSelection()
        clone.setSelected(True)
        self.recalculate_graph_preview()
        return clone

    def _spawn_default_pipeline(self):
        in_node = NodeItem(node_type="input", title="📥 ŹRÓDŁO DANYCH")
        in_node.setPos(40, 120)
        self.scene.addItem(in_node)

        crop_node = NodeItem(node_type="crop", title="Przycięcie (Crop)",
                             step_data={'type': 'crop', 'name': "Przycięcie [1000–1800]", 'min_w': 1000.0, 'max_w': 1800.0})
        crop_node.setPos(280, 120)
        self.scene.addItem(crop_node)

        asls_node = NodeItem(node_type="asls", title="Linia Bazowa AsLS",
                             step_data={'type': 'asls', 'name': "AsLS (λ=1e5, p=0.001)", 'lam': 1e5, 'p': 0.001})
        asls_node.setPos(520, 120)
        self.scene.addItem(asls_node)

        out_node = NodeItem(node_type="output", title="💾 ZAPIS WYNIKU")
        out_node.setPos(760, 120)
        self.scene.addItem(out_node)

        self.scene.addItem(ConnectionItem(in_node.out_ports["out"], crop_node.in_ports["in"]))
        self.scene.addItem(ConnectionItem(crop_node.out_ports["out"], asls_node.in_ports["in"]))
        self.scene.addItem(ConnectionItem(asls_node.out_ports["out"], out_node.in_ports["in"]))

    def _delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, NodeItem):
                if item.node_type in ("input", "output"):
                    continue
                for p in list(item.in_ports.values()) + list(item.out_ports.values()):
                    for c in list(p.connections):
                        c.disconnect()
                self.scene.removeItem(item)
            elif isinstance(item, ConnectionItem):
                item.disconnect()
        self.recalculate_graph_preview()

    def _clear_graph(self):
        self.scene.clear()
        self._spawn_default_pipeline()
        self.recalculate_graph_preview()

    # ==========================================================================
    # SILNIK DAG (TOPOLOGICAL SORT & EVALUATION)
    # ==========================================================================

    def _topological_sort(self, nodes: List[NodeItem]) -> Tuple[Optional[List[NodeItem]], bool]:
        adj: Dict[NodeItem, List[NodeItem]] = {n: [] for n in nodes}
        in_degree: Dict[NodeItem, int] = {n: 0 for n in nodes}

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

    def _evaluate_dag(self, ordered_nodes: List[NodeItem], input_x: np.ndarray, input_y: np.ndarray,
                      session_spectra: Dict[str, SpectrumData]) -> Dict[NodeItem, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        results: Dict[NodeItem, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

        for node in ordered_nodes:
            ntype = node.node_type

            if ntype == "input":
                results[node] = (input_x.copy(), input_y.copy())

            elif ntype == "spectrum_select":
                spec_name = node.step_data.get('selected_spectrum', '')
                # session.get_spectrum automatycznie sprawdza zarówno _spectra, jak i _atm_spectra
                s = session.get_spectrum(spec_name)
                if s is not None:
                    results[node] = (s.wavenumbers.copy(), s.intensities.copy())
                else:
                    results[node] = (input_x.copy(), np.zeros_like(input_x))

            elif ntype == "vaporfit":
                p_sig = node.in_ports.get("in_signal")
                p_atm = node.in_ports.get("in_atm")

                src_sig = p_sig.connections[0].out_port.node_item if (p_sig and p_sig.connections) else None
                data_sig = results.get(src_sig, (None, None)) if src_sig else (None, None)

                atm_sources = [c.out_port.node_item for c in p_atm.connections if c.out_port and c.out_port.node_item] if p_atm else []
                atm_datasets = [results.get(s, (None, None)) for s in atm_sources if s in results and results[s][0] is not None]

                # 1. Jeśli użytkownik podpiął kable do portu Atm -> używamy wskazanych węzłów
                if data_sig[0] is not None and atm_datasets:
                    xs, ys = data_sig
                    atm_cols = []
                    for xa, ya in atm_datasets:
                        if xa[0] > xa[-1]:
                            xa_s, ya_s = xa[::-1], ya[::-1]
                        else:
                            xa_s, ya_s = xa, ya
                        atm_cols.append(np.interp(xs, xa_s, ya_s))
                    atm_matrix = np.column_stack(atm_cols)

                # 2. Jeśli port Atm jest wolny -> automatycznie pobieramy widma z gałęzi Wzorców Atmosfery
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
                    sign = node.step_data.get('sign_mode', 'auto')

                    try:
                        corr_y, coeffs, metrics = AtmosphericCorrectionEngine.fit_and_subtract(
                            xs, ys, atm_matrix, sg_points=sg_pts, sg_poly=sg_poly, sign_mode=sign
                        )
                        node.step_data['metrics'] = metrics
                        node.step_data['coeffs'] = coeffs.tolist()
                        results[node] = (xs.copy(), corr_y)
                    except Exception as e:
                        logger.warning(f"Błąd kalkulacji VaporFit w węźle: {e}")
                        results[node] = (xs.copy(), ys.copy())
                elif data_sig[0] is not None:
                    results[node] = (data_sig[0].copy(), data_sig[1].copy())
                else:
                    results[node] = (None, None)

            elif ntype == "math_op":
                p_a = node.in_ports.get("in_a")
                p_b = node.in_ports.get("in_b")
                src_a = p_a.connections[0].out_port.node_item if (p_a and p_a.connections) else None
                src_b = p_b.connections[0].out_port.node_item if (p_b and p_b.connections) else None

                data_a = results.get(src_a, (None, None)) if src_a else (None, None)
                data_b = results.get(src_b, (None, None)) if src_b else (None, None)

                if data_a[0] is not None and data_b[0] is not None:
                    xa, ya = data_a
                    xb, yb = data_b

                    if xb[0] > xb[-1]:
                        xb_s, yb_s = xb[::-1], yb[::-1]
                    else:
                        xb_s, yb_s = xb, yb
                    yb_interp = np.interp(xa, xb_s, yb_s)

                    op = node.step_data.get('op', 'sub')
                    k = float(node.step_data.get('k', 1.0))

                    if op == 'sub':
                        y_res = ya - k * yb_interp
                    elif op == 'add':
                        y_res = ya + k * yb_interp
                    elif op == 'mul':
                        y_res = ya * yb_interp
                    elif op == 'div':
                        y_res = np.where(np.abs(yb_interp) > 1e-12, ya / yb_interp, 0.0)
                    else:
                        y_res = ya - k * yb_interp

                    results[node] = (xa.copy(), y_res)
                elif data_a[0] is not None:
                    results[node] = (data_a[0].copy(), data_a[1].copy())
                else:
                    results[node] = (None, None)

            elif ntype == "output":
                p_in = node.in_ports.get("in")
                src = p_in.connections[0].out_port.node_item if (p_in and p_in.connections) else None
                results[node] = results.get(src, (None, None)) if src else (None, None)

            else:
                p_in = node.in_ports.get("in")
                src = p_in.connections[0].out_port.node_item if (p_in and p_in.connections) else None
                if src and src in results and results[src][0] is not None:
                    curr_x, curr_y = results[src]
                    nx, ny = apply_single_step(curr_x, curr_y, node.step_data)
                    results[node] = (nx, ny)
                else:
                    results[node] = (None, None)

        return results

    def _get_base_preview_data(self) -> Tuple[np.ndarray, np.ndarray]:
        active = session.active_spectrum
        if active is not None and len(active.wavenumbers) > 2:
            return active.wavenumbers.copy(), active.intensities.copy()

        wn = np.linspace(4000, 400, 1200)
        y = 0.8 * np.exp(-((wn - 1650) / 35) ** 2) + 0.4 * np.exp(-((wn - 1540) / 40) ** 2) + 0.05 * np.random.normal(0, 0.02, len(wn))
        return wn, y

    def recalculate_graph_preview(self):
        all_nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        ordered_nodes, has_cycle = self._topological_sort(all_nodes)

        if has_cycle:
            return

        base_x, base_y = self._get_base_preview_data()
        eval_results = self._evaluate_dag(ordered_nodes, base_x, base_y, session.spectra)

        for node, (res_x, res_y) in eval_results.items():
            node.preview_x = res_x
            node.preview_y = res_y
            node.update()

        self._update_live_canvas_preview()

    # ==========================================================================
    # LIVE PREVIEW NA SPECTROCANVAS
    # ==========================================================================

    def _get_canvas(self):
        main_win = self.parent()
        return getattr(main_win, 'canvas_2d', None)

    def _get_active_axes(self):
        canvas = self._get_canvas()
        if not canvas:
            return None
        if hasattr(canvas, 'ax') and canvas.ax is not None:
            return canvas.ax
        if hasattr(canvas, 'axes') and len(canvas.axes) > 0:
            return canvas.axes[0]
        if hasattr(canvas, 'fig') and len(canvas.fig.axes) > 0:
            return canvas.fig.axes[0]
        return None

    def _trigger_canvas_draw(self, canvas, ax=None):
        if ax is not None and getattr(ax, 'figure', None) and getattr(ax.figure, 'canvas', None):
            ax.figure.canvas.draw_idle()
        elif hasattr(canvas, 'canvas') and hasattr(canvas.canvas, 'draw_idle'):
            canvas.canvas.draw_idle()
        elif hasattr(canvas, 'draw_idle'):
            canvas.draw_idle()
        elif hasattr(canvas, 'redraw_all'):
            canvas.redraw_all()

    def _on_selection_changed(self):
        self.view.viewport().update()
        self._update_live_canvas_preview()

    def _toggle_live_preview(self, enabled: bool):
        if not enabled:
            self._clear_canvas_preview()
        else:
            self._update_live_canvas_preview()

    def _clear_canvas_preview(self):
        canvas = self._get_canvas()
        if not canvas:
            return
        line = getattr(canvas, '_node_preview_line', None)
        if line is not None:
            try:
                line.remove()
            except Exception:
                pass
            canvas._node_preview_line = None
            ax = self._get_active_axes()
            self._trigger_canvas_draw(canvas, ax)

    def _update_live_canvas_preview(self):
        if not self.chk_live_preview.isChecked():
            return

        canvas = self._get_canvas()
        ax = self._get_active_axes()
        if not canvas or not ax:
            return

        target_node = None
        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        if selected:
            target_node = selected[0]
        else:
            for item in self.scene.items():
                if isinstance(item, NodeItem) and item.node_type == "output":
                    target_node = item
                    break

        if not target_node or target_node.preview_x is None or target_node.preview_y is None:
            self._clear_canvas_preview()
            return

        x = target_node.preview_x
        y = target_node.preview_y

        line = getattr(canvas, '_node_preview_line', None)
        label_text = f"👁 Podgląd: {target_node.title}"

        if line is not None and line in ax.lines:
            line.set_data(x, y)
            line.set_label(label_text)
            line.set_visible(True)
        else:
            line, = ax.plot(
                x, y, color="#ffb703", linestyle="--", linewidth=2.0,
                alpha=0.9, zorder=999, label=label_text
            )
            canvas._node_preview_line = line

        self._trigger_canvas_draw(canvas, ax)

    def closeEvent(self, event):
        self._clear_canvas_preview()
        super().closeEvent(event)

    def reject(self):
        self._clear_canvas_preview()
        super().reject()

    # ==========================================================================
    # WYKONANIE POTOKU DAG Z HISTORIĄ QUNDOSTACK
    # ==========================================================================

    def _execute_pipeline(self):
        all_nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        ordered_nodes, has_cycle = self._topological_sort(all_nodes)

        if has_cycle:
            QMessageBox.warning(self, "Błąd", "Wykryto zapętlenie w grafie (cykl). Potok musi być acykliczny.")
            return

        out_node = next((n for n in all_nodes if n.node_type == "output"), None)
        if not out_node:
            QMessageBox.warning(self, "Błąd", "Brak węzła końcowego 'Zapis Wyniku'.")
            return

        self._clear_canvas_preview()

        main_win = self.parent()
        canvas = getattr(main_win, 'canvas_2d', None)
        target_win = session.active_window_id if session.active_context == "window" else None

        scope = self.combo_scope.currentIndex()
        in_place = self.chk_inplace.isChecked()

        try:
            # 1. TYLKO AKTYWNE WIDMO
            if scope == 0:
                active = session.active_spectrum
                if not active:
                    QMessageBox.warning(self, "Błąd", "Brak aktywnego widma.")
                    return

                res_map = self._evaluate_dag(ordered_nodes, active.wavenumbers, active.intensities, session.spectra)
                new_x, new_y = res_map.get(out_node, (None, None))
                if new_x is None or new_y is None:
                    QMessageBox.warning(self, "Błąd", "Węzeł wyjściowy nie otrzymał danych.")
                    return

                new_meta = copy.deepcopy(active.metadata)
                new_meta["dag_pipeline"] = [n.title for n in ordered_nodes if n.node_type not in ("input", "output")]

                if in_place:
                    cmd = ModifySpectrumCommand(
                        active, f"DAG: {active.name}", active.intensities, new_y,
                        active.wavenumbers, new_x, old_meta=active.metadata, new_meta=new_meta, canvas=canvas
                    )
                    session.undo_stack.push(cmd)
                else:
                    new_spec = SpectrumData(
                        wavenumbers=new_x, intensities=new_y, name=f"dag_{active.name}",
                        file_path=active.file_path, file_type="processed", metadata=new_meta
                    )
                    cmd = AddBatchSpectraCommand([new_spec], f"DAG: {active.name}", canvas=canvas, window_id=target_win)
                    session.undo_stack.push(cmd)

            # 2. WSZYSTKIE WIDOCZNE WIDMA (BATCH)
            elif scope == 1:
                visible = session.get_display_spectra()
                if not visible:
                    QMessageBox.warning(self, "Błąd", "Brak widocznych widm.")
                    return

                batch_records = []
                new_created = []

                for spec in visible:
                    res_map = self._evaluate_dag(ordered_nodes, spec.wavenumbers, spec.intensities, session.spectra)
                    new_x, new_y = res_map.get(out_node, (None, None))
                    if new_x is None or new_y is None:
                        continue

                    old_x, old_y = spec.wavenumbers.copy(), spec.intensities.copy()
                    new_meta = copy.deepcopy(spec.metadata)
                    new_meta["dag_pipeline"] = [n.title for n in ordered_nodes if n.node_type not in ("input", "output")]

                    if in_place:
                        batch_records.append((spec, old_x, old_y, new_x, new_y, spec.metadata, new_meta))
                    else:
                        new_created.append(SpectrumData(
                            wavenumbers=new_x.copy(), intensities=new_y.copy(),
                            name=f"dag_{spec.name}", file_path=spec.file_path,
                            file_type="processed", metadata=new_meta
                        ))

                if in_place and batch_records:
                    session.undo_stack.push(BatchPipelineCommand(batch_records, f"DAG Batch ({len(batch_records)} widm)", canvas=canvas))
                elif not in_place and new_created:
                    session.undo_stack.push(AddBatchSpectraCommand(new_created, f"DAG Batch ({len(new_created)} widm)", canvas=canvas, window_id=target_win))

            # 3. SERIA 2D / TENSOR
            elif scope == 2:
                active_s = session.active_series
                if not active_s:
                    QMessageBox.warning(self, "Błąd", "Brak aktywnej serii 2D.")
                    return

                mat = active_s.get_matrix_2d()
                cols = mat.shape[1]
                processed_cols = []
                out_wn = None

                for c in range(cols):
                    res_map = self._evaluate_dag(ordered_nodes, active_s.wavenumbers, mat[:, c], session.spectra)
                    new_x, new_y = res_map.get(out_node, (None, None))
                    if new_x is None or new_y is None:
                        continue
                    out_wn = new_x
                    processed_cols.append(new_y)

                if not processed_cols:
                    QMessageBox.warning(self, "Błąd", "Nie udało się przetworzyć żadnego profilu serii.")
                    return

                new_mat = np.column_stack(processed_cols)

                if in_place:
                    new_data = new_mat.reshape(len(out_wn), active_s.data.shape[1], active_s.data.shape[2]) if active_s.is_3d else new_mat
                    cmd = ModifySeriesCommand(active_s, f"DAG Serii: {active_s.name}",
                                              active_s.wavenumbers, out_wn, active_s.data, new_data, canvas=canvas)
                    session.undo_stack.push(cmd)
                else:
                    out_data = new_mat.reshape(len(out_wn), active_s.data.shape[1], active_s.data.shape[2]) if active_s.is_3d else new_mat
                    session.add_series(SpectralSeries(
                        name=f"dag_{active_s.name}", wavenumbers=out_wn, data=out_data,
                        axis_n_name=active_s.axis_n_name, axis_n_values=active_s.axis_n_values,
                        axis_m_name=active_s.axis_m_name, axis_m_values=active_s.axis_m_values
                    ), make_active=True)

            QMessageBox.information(self, "Sukces", f"Pomyślnie wykonano potok DAG!")
            self.recalculate_graph_preview()

        except Exception as e:
            logger.error(f"Błąd wykonania potoku DAG: {e}")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas obliczeń:\n{e}")