"""
gui/node_editor/items.py
Elementy wizualne grafu DAG: PortItem, ConnectionItem oraz NodeItem.
Węzeł 'output_window' posiada teraz port wyjściowy (funkcja Checkpoint / Tap).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import copy
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QMenu, QDialog
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPainterPath, QPainterPathStroker, QPen, QBrush,
    QColor, QFont, QLinearGradient
)

from gui.workspaces.ws_preprocessing import StepParamDialog
from gui.node_editor.dialogs import NodeParamDialog


class PortItem(QGraphicsItem):
    """Pojedynczy port węzła z hit-boxem i rozróżnieniem ról kolorystycznych."""
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
            if "res" in self.port_id:
                col = QColor("#e63946")
            elif "recon" in self.port_id:
                col = QColor("#f4a261")
            elif "bl" in self.port_id:
                col = QColor("#9d4edd")
            elif "atm" in self.port_id:
                col = QColor("#00b4d8")
            else:
                col = QColor("#48cae4")
        elif "atm" in self.port_id or "b" in self.port_id:
            col = QColor("#fb8500")
        else:
            col = QColor("#ffb703")

        if self.is_hovered:
            col = col.lighter(135)

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
    """Krzywa Béziera łącząca porty."""

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
                sc.dialog_owner.sync_state_to_session()

    def disconnect(self):
        if self.out_port and self in self.out_port.connections:
            self.out_port.connections.remove(self)
        if self.in_port and self in self.in_port.connections:
            self.in_port.connections.remove(self)
        if self.scene():
            self.scene().removeItem(self)


class NodeItem(QGraphicsItem):
    """Wizualny węzeł grafu z obsługą sparkline FTIR oraz portów chemometrycznych."""
    WIDTH = 215
    HEIGHT = 145

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
        # 1. Wejścia (In ports)
        if self.node_type == "math_op":
            p_a = PortItem(is_output=False, node_item=self, port_id="in_a", label="A")
            p_a.setPos(0, 68)
            self.in_ports["in_a"] = p_a

            p_b = PortItem(is_output=False, node_item=self, port_id="in_b", label="B")
            p_b.setPos(0, 108)
            self.in_ports["in_b"] = p_b

        elif self.node_type == "vaporfit":
            p_sig = PortItem(is_output=False, node_item=self, port_id="in_signal", label="Syg")
            p_sig.setPos(0, 68)
            self.in_ports["in_signal"] = p_sig

            p_atm = PortItem(is_output=False, node_item=self, port_id="in_atm", label="Atm")
            p_atm.setPos(0, 108)
            self.in_ports["in_atm"] = p_atm

        elif self.node_type not in ("input", "spectrum_select", "dft_broadener"):
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In")
            p_in.setPos(0, 88)
            self.in_ports["in"] = p_in

        # 2. Wyjścia (Out ports)
        # Wyłącznie czysty 'output' nie ma wyjścia; 'output_window' działa jako tap/checkpoint
        if self.node_type == "output":
            pass

        elif self.node_type == "svd_denoise":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="D_k")
            p_out.setPos(self.WIDTH, 68)
            self.out_ports["out"] = p_out

            p_res = PortItem(is_output=True, node_item=self, port_id="out_res", label="Res")
            p_res.setPos(self.WIDTH, 108)
            self.out_ports["out_res"] = p_res

        elif self.node_type == "mcr_decompose":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="S^T")
            p_out.setPos(self.WIDTH, 56)
            self.out_ports["out"] = p_out

            p_rec = PortItem(is_output=True, node_item=self, port_id="out_recon", label="Rec")
            p_rec.setPos(self.WIDTH, 88)
            self.out_ports["out_recon"] = p_rec

            p_res = PortItem(is_output=True, node_item=self, port_id="out_res", label="Res")
            p_res.setPos(self.WIDTH, 120)
            self.out_ports["out_res"] = p_res

        elif self.node_type in ("asls", "airpls", "poly_bl", "linear_bl"):
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Syg")
            p_out.setPos(self.WIDTH, 68)
            self.out_ports["out"] = p_out

            p_bl = PortItem(is_output=True, node_item=self, port_id="out_bl", label="BL")
            p_bl.setPos(self.WIDTH, 108)
            self.out_ports["out_bl"] = p_bl

        elif self.node_type == "vaporfit":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Syg")
            p_out.setPos(self.WIDTH, 68)
            self.out_ports["out"] = p_out

            p_atm_fit = PortItem(is_output=True, node_item=self, port_id="out_atm", label="Atm")
            p_atm_fit.setPos(self.WIDTH, 108)
            self.out_ports["out_atm"] = p_atm_fit

        else:
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Out")
            p_out.setPos(self.WIDTH, 88)
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
        elif self.node_type == "output_window":
            grad.setColorAt(0, QColor("#7209b7"))
            grad.setColorAt(1, QColor("#b5179e"))
        elif self.node_type == "svd_denoise":
            grad.setColorAt(0, QColor("#240046"))
            grad.setColorAt(1, QColor("#5a189a"))
        elif self.node_type == "mcr_decompose":
            grad.setColorAt(0, QColor("#9d0208"))
            grad.setColorAt(1, QColor("#d90429"))
        elif self.node_type == "matrix_scaling":
            grad.setColorAt(0, QColor("#0077b6"))
            grad.setColorAt(1, QColor("#0096c7"))
        elif self.node_type == "photometry":
            grad.setColorAt(0, QColor("#007f5f"))
            grad.setColorAt(1, QColor("#55a630"))
        elif self.node_type == "dft_broadener":
            grad.setColorAt(0, QColor("#4361ee"))
            grad.setColorAt(1, QColor("#4cc9f0"))
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
        if len(desc) > 32:
            desc = desc[:30] + "…"
        painter.drawText(QRectF(8, 26, self.WIDTH - 16, 16), Qt.AlignLeft | Qt.AlignVCenter, desc)

        if self.node_type == "math_op":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(10, 58, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "A")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(10, 98, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "B")
        elif self.node_type == "vaporfit":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(9, 58, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(9, 98, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Atm")

        if self.node_type == "svd_denoise":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 58, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "D_k")
            painter.setPen(QColor("#e63946"))
            painter.drawText(QRectF(self.WIDTH - 36, 98, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Res")
        elif self.node_type == "mcr_decompose":
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 46, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "S^T")
            painter.setPen(QColor("#f4a261"))
            painter.drawText(QRectF(self.WIDTH - 36, 78, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Rec")
            painter.setPen(QColor("#e63946"))
            painter.drawText(QRectF(self.WIDTH - 36, 110, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Res")
        elif len(self.out_ports) > 1:
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 58, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#9d4edd") if "out_bl" in self.out_ports else QColor("#00b4d8"))
            lbl_b = "BL" if "out_bl" in self.out_ports else "Atm"
            painter.drawText(QRectF(self.WIDTH - 36, 98, 26, 20), Qt.AlignRight | Qt.AlignVCenter, lbl_b)

        box_left = 32 if len(self.in_ports) > 1 else 10
        box_right = self.WIDTH - (38 if len(self.out_ports) > 1 else 10)
        box_w = box_right - box_left
        box_rect = QRectF(box_left, 46, box_w, 88)

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
            y_coords = (46 + 88 - 4) - y_norm * (88 - 8)

            path = QPainterPath()
            path.moveTo(x_coords[0], y_coords[0])
            for i in range(1, len(sampled_indices)):
                path.lineTo(x_coords[i], y_coords[i])

            if self.node_type == "input":
                curve_col = QColor("#52b788")
            elif self.node_type == "output":
                curve_col = QColor("#ff595e")
            elif self.node_type == "output_window":
                curve_col = QColor("#d946ef")
            elif self.node_type == "svd_denoise":
                curve_col = QColor("#b5179e")
            elif self.node_type == "mcr_decompose":
                curve_col = QColor("#ff4d6d")
            elif self.node_type == "matrix_scaling":
                curve_col = QColor("#48cae4")
            elif self.node_type == "photometry":
                curve_col = QColor("#a7c957")
            elif self.node_type == "dft_broadener":
                curve_col = QColor("#4cc9f0")
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
            if self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.sync_state_to_session()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if self.node_type == "input":
            super().mouseDoubleClickEvent(event)
            return

        if self.node_type in ("math_op", "scalar_math", "vaporfit", "spectrum_select",
                              "photometry", "dft_broadener", "output_window",
                              "svd_denoise", "mcr_decompose", "matrix_scaling"):
            dlg = NodeParamDialog(self)
        else:
            dlg = StepParamDialog(self.step_data)

        if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
            self.scene().dialog_owner.recalculate_graph_preview()
            self.scene().dialog_owner.sync_state_to_session()
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
            if self.node_type in ("math_op", "scalar_math", "vaporfit", "spectrum_select",
                                  "photometry", "dft_broadener", "output_window",
                                  "svd_denoise", "mcr_decompose", "matrix_scaling"):
                dlg = NodeParamDialog(self)
            else:
                dlg = StepParamDialog(self.step_data)
            if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.recalculate_graph_preview()
                self.scene().dialog_owner.sync_state_to_session()
        elif action == act_del:
            sc = self.scene()
            for p in list(self.in_ports.values()) + list(self.out_ports.values()):
                for c in list(p.connections):
                    c.disconnect()
            sc.removeItem(self)
            if hasattr(sc, 'dialog_owner'):
                sc.dialog_owner.recalculate_graph_preview()
                sc.dialog_owner.sync_state_to_session()