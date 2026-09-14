"""
gui/node_editor/items.py
Czyste elementy wizualne grafu DAG: PortItem, ConnectionItem oraz NodeItem.
Porty, geometria, kolory nagłówków i specyficzne nakładki wizualne
są dynamicznie pobierane z instancji BaseNode.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QMenu, QDialog
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPainterPath, QPainterPathStroker, QPen, QBrush,
    QColor, QFont, QLinearGradient
)

from gui.node_editor.registry import get_node_class
from gui.node_editor.base import BaseNode


class PortItem(QGraphicsItem):
    """Pojedynczy port węzła z hit-boxem, flagą wielowejściowości i kolorystyką ról."""
    RADIUS = 6.0

    def __init__(self, is_output: bool, node_item: 'NodeItem', port_id: str, label: str = "", allow_multiple: bool = False, color: Optional[QColor] = None):
        super().__init__(node_item)
        self.is_output = is_output
        self.node_item = node_item
        self.port_id = port_id
        self.label = label
        self.allow_multiple = allow_multiple
        self.color = color or (QColor("#48cae4") if is_output else QColor("#ffb703"))
        self.connections: List['ConnectionItem'] = []
        self.setAcceptHoverEvents(True)
        self.is_hovered = False

    def boundingRect(self) -> QRectF:
        r = self.RADIUS + 3
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        col = self.color.lighter(135) if self.is_hovered else self.color
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
    """Wizualny węzeł grafu potoku DAG."""

    def __init__(self, node_type: str, title: str, step_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.node_type = node_type
        self.step_data = step_data or {'type': node_type, 'name': title}

        node_cls = get_node_class(node_type)
        if node_cls:
            self.node_obj: Optional[BaseNode] = node_cls(self.step_data)
            self.title = title or self.node_obj.title
            self.WIDTH = self.node_obj.width
            self.HEIGHT = self.node_obj.height
        else:
            self.node_obj = None
            self.title = title
            self.WIDTH = 220
            self.HEIGHT = 150

        self.preview_x: Optional[np.ndarray] = None
        self.preview_y: Optional[np.ndarray] = None
        self.preview_total_count: int = 1

        self.in_ports: Dict[str, PortItem] = {}
        self.out_ports: Dict[str, PortItem] = {}

        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )

        self._setup_ports()

    def _setup_ports(self):
        if self.node_obj:
            for pdef in self.node_obj.get_ports():
                target_dict = self.out_ports if pdef.is_output else self.in_ports
                p_item = PortItem(
                    is_output=pdef.is_output,
                    node_item=self,
                    port_id=pdef.port_id,
                    label=pdef.label,
                    allow_multiple=pdef.allow_multiple,
                    color=pdef.color
                )
                pos_x = self.WIDTH if pdef.is_output else 0
                pos_y = pdef.pos_y if pdef.pos_y is not None else 90
                p_item.setPos(pos_x, pos_y)
                target_dict[pdef.port_id] = p_item
        else:
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In")
            p_in.setPos(0, 90)
            self.in_ports["in"] = p_in

            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Out")
            p_out.setPos(self.WIDTH, 90)
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

        if self.node_obj and hasattr(self.node_obj, 'header_gradient'):
            grad.setColorAt(0, QColor(self.node_obj.header_gradient[0]))
            grad.setColorAt(1, QColor(self.node_obj.header_gradient[1]))
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

        painter.setFont(QFont("Arial", 8, QFont.Bold))
        for p in self.in_ports.values():
            if p.label and p.label != "In":
                painter.setPen(QPen(p.color.lighter(120)))
                painter.drawText(QRectF(10, p.pos().y() - 10, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, p.label)

        for p in self.out_ports.values():
            if p.label and p.label != "Out":
                painter.setPen(p.color.lighter(120))
                painter.drawText(QRectF(self.WIDTH - 36, p.pos().y() - 10, 26, 20), Qt.AlignRight | Qt.AlignVCenter, p.label)

        # Sparklines
        box_left = 32 if len(self.in_ports) > 1 else 10
        box_right = self.WIDTH - (
            42 if self.node_type == "hdo_decomposition" else (38 if len(self.out_ports) > 1 else 10))
        box_w = box_right - box_left
        box_rect = QRectF(box_left, 46, box_w, 92)

        painter.setBrush(QBrush(QColor("#141419")))
        painter.setPen(QPen(QColor("#343a40"), 1))
        painter.drawRoundedRect(box_rect, 4, 4)

        if self.node_obj:
            self.node_obj.draw_custom_overlay(painter, box_rect)

        # Jeśli węzeł nie posiada podglądu wykresu (np. Konfiguracja HDO), nie rysujemy krzywych ani "Brak sygnału"
        has_preview = getattr(self.node_obj, 'has_preview', True) if self.node_obj else (
                    self.node_type not in ("mc_params", "hdo_config"))
        if not has_preview:
            return

        curves = []
        if self.preview_y is not None:
            y_raw = self.preview_y
            if y_raw.ndim == 1 and len(y_raw) > 1:
                curves = [y_raw.copy()]
            elif y_raw.ndim == 2 and y_raw.shape[0] > 1:
                n_c = y_raw.shape[1]
                if n_c == 1:
                    curves = [y_raw[:, 0].copy()]
                elif n_c == 2:
                    curves = [y_raw[:, 0].copy(), y_raw[:, 1].copy()]
                elif n_c == 3:
                    curves = [y_raw[:, 0].copy(), y_raw[:, 1].copy(), y_raw[:, 2].copy()]
                elif n_c > 3:
                    curves = [y_raw[:, 0].copy(), y_raw[:, n_c // 2].copy(), y_raw[:, -1].copy()]

        if curves and len(curves[0]) > 1:
            if self.preview_x is not None and len(self.preview_x) == len(curves[0]):
                if self.preview_x[0] < self.preview_x[-1]:
                    curves = [c[::-1] for c in curves]

            all_vals = np.concatenate(curves)
            y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            y_rng = y_max - y_min
            if y_rng < 1e-12 or np.isnan(y_rng): y_rng = 1.0

            n_pts = len(curves[0])
            step = max(1, n_pts // 140)
            sampled_indices = np.arange(0, n_pts, step)
            if sampled_indices[-1] != n_pts - 1:
                sampled_indices = np.append(sampled_indices, n_pts - 1)

            x_coords = box_left + 3 + (sampled_indices / (n_pts - 1)) * (box_w - 6)

            # Dwukolorowa reprezentacja dopasowania (Źródło = błękit, Fit = pomarańcz)
            is_fit_display = self.node_type in ("hdo_decomposition", "output_folder") or getattr(self.node_obj,
                                                                                                 'node_type',
                                                                                                 '') in (
                                 "hdo_decomposition", "output_folder")
            if is_fit_display and len(curves) == 2:
                palette = [QColor("#48cae4"), QColor("#f4a261")]
            elif len(curves) == 1:
                palette = [QColor("#00f5d4")]
            elif len(curves) == 2:
                palette = [QColor("#48cae4"), QColor("#ffd166")]
            else:
                palette = [QColor("#48cae4"), QColor("#ffd166"), QColor("#ff4d6d")]

            for idx, c_data in enumerate(curves):
                y_norm = (c_data[sampled_indices] - y_min) / y_rng
                y_norm = np.clip(np.nan_to_num(y_norm, nan=0.0), 0.0, 1.0)
                y_coords = (46 + 92 - 4) - y_norm * (92 - 8)

                path = QPainterPath()
                path.moveTo(x_coords[0], y_coords[0])
                for i in range(1, len(sampled_indices)):
                    path.lineTo(x_coords[i], y_coords[i])

                painter.setPen(QPen(palette[idx], 1.25))
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
        dlg = None
        if self.node_obj:
            dlg = self.node_obj.create_config_dialog(parent=None)

        if dlg and dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
            self.scene().dialog_owner.recalculate_graph_preview()
            self.scene().dialog_owner.sync_state_to_session()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        act_dup = menu.addAction("📋 Duplikuj węzeł (⌘D / Ctrl+D)")
        act_edit = menu.addAction("✏️ Edytuj parametry...")
        act_del = menu.addAction("✕ Usuń węzeł (Del)")

        action = menu.exec(event.screenPos())
        if action == act_dup:
            if self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.duplicate_node(self)
        elif action == act_edit:
            self.mouseDoubleClickEvent(event)
        elif action == act_del:
            sc = self.scene()
            for p in list(self.in_ports.values()) + list(self.out_ports.values()):
                for c in list(p.connections):
                    c.disconnect()
            sc.removeItem(self)
            if hasattr(sc, 'dialog_owner'):
                sc.dialog_owner.recalculate_graph_preview()
                sc.dialog_owner.sync_state_to_session()