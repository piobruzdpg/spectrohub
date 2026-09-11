"""
gui/node_editor/items.py
Elementy wizualne grafu DAG: PortItem, ConnectionItem oraz NodeItem.
Wsparcie dla wielowidmowych sparklines, fan-in (Mux/Terminale), profili stężeń MCR/PCA,
mini-raportów na kafelkach (HDO) oraz dekompozycji HDO / pasm syntetycznych.
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
    """Pojedynczy port węzła z hit-boxem, flagą wielowejściowości i kolorystyką ról."""
    RADIUS = 6.0

    def __init__(self, is_output: bool, node_item: 'NodeItem', port_id: str, label: str = "", allow_multiple: bool = False):
        super().__init__(node_item)
        self.is_output = is_output
        self.node_item = node_item
        self.port_id = port_id
        self.label = label
        self.allow_multiple = allow_multiple
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
            elif "recon" in self.port_id or "fit" in self.port_id:
                col = QColor("#f4a261")
            elif "conc" in self.port_id or "scores" in self.port_id or "area" in self.port_id:
                col = QColor("#52b788")
            elif "bl" in self.port_id:
                col = QColor("#9d4edd")
            elif "atm" in self.port_id:
                col = QColor("#00b4d8")
            elif "mat" in self.port_id or "comps" in self.port_id or "peaks" in self.port_id:
                col = QColor("#06d6a0")
            elif "mc" in self.port_id:
                col = QColor("#ff9f1c")
            else:
                col = QColor("#48cae4")
        elif "atm" in self.port_id or "b" in self.port_id or "ref" in self.port_id or "bases" in self.port_id:
            col = QColor("#fb8500")
        elif "peaks" in self.port_id:
            col = QColor("#ff006e")
        elif "mc" in self.port_id:
            col = QColor("#ff9f1c")
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
    """Wizualny węzeł grafu z obsługą 1-3 widm sparklines, mini-raportów i granularnych portów."""
    WIDTH = 215
    HEIGHT = 145

    def __init__(self, node_type: str, title: str, step_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.node_type = node_type
        self.title = title
        self.step_data = step_data or {'type': node_type, 'name': title}

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
        # 1. Wejścia (In ports)
        if self.node_type == "math_op":
            p_a = PortItem(is_output=False, node_item=self, port_id="in_a", label="A")
            p_a.setPos(0, 68)
            self.in_ports["in_a"] = p_a

            p_b = PortItem(is_output=False, node_item=self, port_id="in_b", label="B")
            p_b.setPos(0, 108)
            self.in_ports["in_b"] = p_b

        elif self.node_type == "internal_standard":
            p_sig = PortItem(is_output=False, node_item=self, port_id="in_signal", label="Syg")
            p_sig.setPos(0, 68)
            self.in_ports["in_signal"] = p_sig

            p_ref = PortItem(is_output=False, node_item=self, port_id="in_ref", label="IS")
            p_ref.setPos(0, 108)
            self.in_ports["in_ref"] = p_ref

        elif self.node_type == "vaporfit":
            p_sig = PortItem(is_output=False, node_item=self, port_id="in_signal", label="Syg")
            p_sig.setPos(0, 68)
            self.in_ports["in_signal"] = p_sig

            p_atm = PortItem(is_output=False, node_item=self, port_id="in_atm", label="Atm", allow_multiple=True)
            p_atm.setPos(0, 108)
            self.in_ports["in_atm"] = p_atm

        elif self.node_type == "hdo_decomposition":
            p_tar = PortItem(is_output=False, node_item=self, port_id="in_target", label="W₀")
            p_tar.setPos(0, 44)
            self.in_ports["in_target"] = p_tar

            p_bas = PortItem(is_output=False, node_item=self, port_id="in_bases", label="Wₖ", allow_multiple=True)
            p_bas.setPos(0, 68)
            self.in_ports["in_bases"] = p_bas

            p_pk = PortItem(is_output=False, node_item=self, port_id="in_peaks", label="Pik", allow_multiple=True)
            p_pk.setPos(0, 94)
            self.in_ports["in_peaks"] = p_pk

            p_mc = PortItem(is_output=False, node_item=self, port_id="in_mc", label="MC")
            p_mc.setPos(0, 120)
            self.in_ports["in_mc"] = p_mc

        elif self.node_type in ("output_window", "merge_spectra"):
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In", allow_multiple=True)
            p_in.setPos(0, 88)
            self.in_ports["in"] = p_in

        elif self.node_type not in ("input", "spectrum_select", "dft_broadener",
                                    "hdo_spectrum_scaled", "synthetic_peak", "mc_params"):
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In")
            p_in.setPos(0, 88)
            self.in_ports["in"] = p_in

        elif self.node_type in ("output_window", "output_table", "merge_spectra"):
            p_in = PortItem(is_output=False, node_item=self, port_id="in", label="In", allow_multiple=True)
            p_in.setPos(0, 88)
            self.in_ports["in"] = p_in

        # 2. Wyjścia (Out ports)
        if self.node_type == "output":
            pass

        elif self.node_type == "svd_denoise":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="D_k")
            p_out.setPos(self.WIDTH, 56)
            self.out_ports["out"] = p_out

            p_res = PortItem(is_output=True, node_item=self, port_id="out_res", label="Res")
            p_res.setPos(self.WIDTH, 88)
            self.out_ports["out_res"] = p_res

            p_scr = PortItem(is_output=True, node_item=self, port_id="out_scores", label="Scr")
            p_scr.setPos(self.WIDTH, 120)
            self.out_ports["out_scores"] = p_scr

        elif self.node_type == "mcr_decompose":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="S^T")
            p_out.setPos(self.WIDTH, 44)
            self.out_ports["out"] = p_out

            p_conc = PortItem(is_output=True, node_item=self, port_id="out_conc", label="C")
            p_conc.setPos(self.WIDTH, 68)
            self.out_ports["out_conc"] = p_conc

            p_rec = PortItem(is_output=True, node_item=self, port_id="out_recon", label="Rec")
            p_rec.setPos(self.WIDTH, 94)
            self.out_ports["out_recon"] = p_rec

            p_res = PortItem(is_output=True, node_item=self, port_id="out_res", label="Res")
            p_res.setPos(self.WIDTH, 120)
            self.out_ports["out_res"] = p_res

        elif self.node_type == "hdo_decomposition":
            p_fit = PortItem(is_output=True, node_item=self, port_id="out_fit", label="Fit")
            p_fit.setPos(self.WIDTH, 38)
            self.out_ports["out_fit"] = p_fit

            p_res = PortItem(is_output=True, node_item=self, port_id="out_res", label="Res")
            p_res.setPos(self.WIDTH, 62)
            self.out_ports["out_res"] = p_res

            p_fres = PortItem(is_output=True, node_item=self, port_id="out_fit_res", label="ΔFit")
            p_fres.setPos(self.WIDTH, 86)
            self.out_ports["out_fit_res"] = p_fres

            p_cmp = PortItem(is_output=True, node_item=self, port_id="out_comps", label="W_k")
            p_cmp.setPos(self.WIDTH, 108)
            self.out_ports["out_comps"] = p_cmp

            p_pks = PortItem(is_output=True, node_item=self, port_id="out_peaks", label="Pks")
            p_pks.setPos(self.WIDTH, 130)
            self.out_ports["out_peaks"] = p_pks

        elif self.node_type == "mc_params":
            p_out = PortItem(is_output=True, node_item=self, port_id="out_mc", label="MC")
            p_out.setPos(self.WIDTH, 88)
            self.out_ports["out_mc"] = p_out

        elif self.node_type in ("hdo_spectrum_scaled", "synthetic_peak"):
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Out")
            p_out.setPos(self.WIDTH, 88)
            self.out_ports["out"] = p_out

        elif self.node_type == "band_integrate":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Area")
            p_out.setPos(self.WIDTH, 88)
            self.out_ports["out"] = p_out

        elif self.node_type == "merge_spectra":
            p_out = PortItem(is_output=True, node_item=self, port_id="out", label="Mat")
            p_out.setPos(self.WIDTH, 88)
            self.out_ports["out"] = p_out

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
        elif self.node_type == "hdo_decomposition":
            grad.setColorAt(0, QColor("#14213d"))
            grad.setColorAt(1, QColor("#2a9d8f"))
        elif self.node_type == "hdo_spectrum_scaled":
            grad.setColorAt(0, QColor("#1d3557"))
            grad.setColorAt(1, QColor("#457b9d"))
        elif self.node_type == "synthetic_peak":
            grad.setColorAt(0, QColor("#720026"))
            grad.setColorAt(1, QColor("#e76f51"))
        elif self.node_type == "mc_params":
            grad.setColorAt(0, QColor("#4f000b"))
            grad.setColorAt(1, QColor("#ff9f1c"))
        elif self.node_type == "merge_spectra":
            grad.setColorAt(0, QColor("#028090"))
            grad.setColorAt(1, QColor("#00a896"))
        elif self.node_type == "internal_standard":
            grad.setColorAt(0, QColor("#7b2cbf"))
            grad.setColorAt(1, QColor("#9d4edd"))
        elif self.node_type == "band_integrate":
            grad.setColorAt(0, QColor("#c1121f"))
            grad.setColorAt(1, QColor("#780000"))
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
        elif self.node_type == "output_table":
            grad.setColorAt(0, QColor("#0077b6"))
            grad.setColorAt(1, QColor("#0096c7"))
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

        # Porty In
        if self.node_type == "math_op":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(10, 58, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "A")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(10, 98, 20, 20), Qt.AlignLeft | Qt.AlignVCenter, "B")
        elif self.node_type == "internal_standard":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(9, 58, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(9, 98, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "IS")
        elif self.node_type == "vaporfit":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(9, 58, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(9, 98, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Atm")
        elif self.node_type == "hdo_decomposition":
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#ffb703"))
            painter.drawText(QRectF(9, 34, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "W₀")
            painter.setPen(QColor("#fb8500"))
            painter.drawText(QRectF(9, 58, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Wₖ")
            painter.setPen(QColor("#ff006e"))
            painter.drawText(QRectF(9, 84, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "Pik")
            painter.setPen(QColor("#ff9f1c"))
            painter.drawText(QRectF(9, 110, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "MC")

        # Porty Out
        if self.node_type == "svd_denoise":
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 46, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "D_k")
            painter.setPen(QColor("#e63946"))
            painter.drawText(QRectF(self.WIDTH - 36, 78, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Res")
            painter.setPen(QColor("#52b788"))
            painter.drawText(QRectF(self.WIDTH - 36, 110, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Scr")
        elif self.node_type == "mcr_decompose":
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 34, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "S^T")
            painter.setPen(QColor("#52b788"))
            painter.drawText(QRectF(self.WIDTH - 36, 58, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "C")
            painter.setPen(QColor("#f4a261"))
            painter.drawText(QRectF(self.WIDTH - 36, 84, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Rec")
            painter.setPen(QColor("#e63946"))
            painter.drawText(QRectF(self.WIDTH - 36, 110, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Res")
        elif self.node_type == "hdo_decomposition":
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#f4a261"))
            painter.drawText(QRectF(self.WIDTH - 36, 28, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Fit")
            painter.setPen(QColor("#e63946"))
            painter.drawText(QRectF(self.WIDTH - 36, 52, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Res")
            painter.setPen(QColor("#ffd166"))
            painter.drawText(QRectF(self.WIDTH - 36, 76, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "ΔFit")
            painter.setPen(QColor("#06d6a0"))
            painter.drawText(QRectF(self.WIDTH - 36, 98, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "W_k")
            painter.setPen(QColor("#ff006e"))
            painter.drawText(QRectF(self.WIDTH - 36, 120, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Pks")
        elif self.node_type == "mc_params":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ff9f1c"))
            painter.drawText(QRectF(self.WIDTH - 36, 78, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "MC")
        elif self.node_type == "band_integrate":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#52b788"))
            painter.drawText(QRectF(self.WIDTH - 40, 78, 30, 20), Qt.AlignRight | Qt.AlignVCenter, "Area")
        elif self.node_type == "merge_spectra":
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#06d6a0"))
            painter.drawText(QRectF(self.WIDTH - 40, 78, 30, 20), Qt.AlignRight | Qt.AlignVCenter, "Mat")
        elif len(self.out_ports) > 1:
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#48cae4"))
            painter.drawText(QRectF(self.WIDTH - 36, 58, 26, 20), Qt.AlignRight | Qt.AlignVCenter, "Syg")
            painter.setPen(QColor("#9d4edd") if "out_bl" in self.out_ports else QColor("#00b4d8"))
            lbl_b = "BL" if "out_bl" in self.out_ports else "Atm"
            painter.drawText(QRectF(self.WIDTH - 36, 98, 26, 20), Qt.AlignRight | Qt.AlignVCenter, lbl_b)

        # Sparklines
        box_left = 32 if len(self.in_ports) > 1 else 10
        box_right = self.WIDTH - (38 if len(self.out_ports) > 1 else 10)
        box_w = box_right - box_left
        box_rect = QRectF(box_left, 46, box_w, 88)

        painter.setBrush(QBrush(QColor("#141419")))
        painter.setPen(QPen(QColor("#343a40"), 1))
        painter.drawRoundedRect(box_rect, 4, 4)

        if self.node_type == "mc_params":
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QColor("#ff9f1c"))
            it = self.step_data.get('n_iter', 3000)
            pen = self.step_data.get('penalty_weight', 10.0)
            painter.drawText(box_rect, Qt.AlignCenter, f"MC Config\nIter: {it}\nPenalty: {pen:.1f}")
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
            y_min = float(np.nanmin(all_vals))
            y_max = float(np.nanmax(all_vals))
            y_rng = y_max - y_min
            if y_rng < 1e-12 or np.isnan(y_rng):
                y_rng = 1.0

            n_pts = len(curves[0])
            step = max(1, n_pts // 140)
            sampled_indices = np.arange(0, n_pts, step)
            if sampled_indices[-1] != n_pts - 1:
                sampled_indices = np.append(sampled_indices, n_pts - 1)

            x_coords = box_left + 3 + (sampled_indices / (n_pts - 1)) * (box_w - 6)

            if self.node_type == "input": node_col = QColor("#52b788")
            elif self.node_type == "output": node_col = QColor("#ff595e")
            elif self.node_type == "output_window": node_col = QColor("#d946ef")
            elif self.node_type == "svd_denoise": node_col = QColor("#b5179e")
            elif self.node_type == "mcr_decompose": node_col = QColor("#ff4d6d")
            elif self.node_type == "hdo_decomposition": node_col = QColor("#2a9d8f")
            elif self.node_type == "hdo_spectrum_scaled": node_col = QColor("#457b9d")
            elif self.node_type == "synthetic_peak": node_col = QColor("#e76f51")
            elif self.node_type == "merge_spectra": node_col = QColor("#06d6a0")
            elif self.node_type == "internal_standard": node_col = QColor("#e0aaff")
            elif self.node_type == "band_integrate": node_col = QColor("#ffba08")
            elif self.node_type == "matrix_scaling": node_col = QColor("#48cae4")
            elif self.node_type == "photometry": node_col = QColor("#a7c957")
            elif self.node_type == "dft_broadener": node_col = QColor("#4cc9f0")
            elif self.node_type == "math_op": node_col = QColor("#ff70a6")
            elif self.node_type == "scalar_math": node_col = QColor("#b388ff")
            elif self.node_type == "vaporfit": node_col = QColor("#48cae4")
            else: node_col = QColor("#00f5d4")

            if len(curves) == 1:
                palette = [node_col]
            elif len(curves) == 2:
                palette = [QColor("#48cae4"), QColor("#ffd166")]
            else:
                palette = [QColor("#48cae4"), QColor("#ffd166"), QColor("#ff4d6d")]

            for idx, c_data in enumerate(curves):
                y_norm = (c_data[sampled_indices] - y_min) / y_rng
                y_norm = np.clip(np.nan_to_num(y_norm, nan=0.0), 0.0, 1.0)
                y_coords = (46 + 88 - 4) - y_norm * (88 - 8)

                path = QPainterPath()
                path.moveTo(x_coords[0], y_coords[0])
                for i in range(1, len(sampled_indices)):
                    path.lineTo(x_coords[i], y_coords[i])

                painter.setPen(QPen(palette[idx], 1.25))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)

            # Mini-raport na kafelku HDO Solwera
            if self.node_type == "hdo_decomposition":
                fit_metrics = self.step_data.get('fit_metrics', {})
                coeffs = fit_metrics.get('coefficients', [])
                r2 = fit_metrics.get('r2', None)
                if coeffs or r2 is not None:
                    txt_parts = []
                    if r2 is not None:
                        txt_parts.append(f"R²={r2:.4f}")
                    if coeffs:
                        c_str = ", ".join([f"c{i+1}={c:.3f}" for i, c in enumerate(coeffs[:2])])
                        if len(coeffs) > 2:
                            c_str += "…"
                        txt_parts.append(c_str)
                    report_text = "\n".join(txt_parts)

                    painter.setFont(QFont("Arial", 7, QFont.Bold))
                    painter.setPen(QColor("#2a9d8f"))
                    badge_rect = QRectF(box_left + 4, 48, box_w - 8, 26)
                    painter.drawText(badge_rect, Qt.AlignLeft | Qt.AlignTop, report_text)

            total_cnt = getattr(self, 'preview_total_count', len(curves))
            if total_cnt > 1 and self.node_type != "hdo_decomposition":
                badge_text = f"3/{total_cnt}" if total_cnt > 3 else f"{len(curves)} widma"
                painter.setFont(QFont("Arial", 7, QFont.Bold))
                painter.setPen(QColor("#788597"))
                badge_rect = QRectF(box_left + box_w - 44, 48, 40, 12)
                painter.drawText(badge_rect, Qt.AlignRight | Qt.AlignVCenter, badge_text)
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
        if self.node_type in ("input", "output", "output_window", "math_op", "scalar_math", "vaporfit",
                              "spectrum_select", "output_table",
                              "photometry", "dft_broadener",
                              "svd_denoise", "mcr_decompose", "matrix_scaling",
                              "internal_standard", "band_integrate", "merge_spectra",
                              "hdo_spectrum_scaled", "synthetic_peak", "mc_params", "hdo_decomposition"):
            dlg = NodeParamDialog(self)
        else:
            dlg = StepParamDialog(self.step_data)

        if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
            self.scene().dialog_owner.recalculate_graph_preview()
            self.scene().dialog_owner.sync_state_to_session()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        act_dup = menu.addAction("📋 Duplikuj węzeł (⌘D / Ctrl+D)")
        act_edit = menu.addAction("✏️ Edytuj parametry...")
        if self.node_type != "input":
            act_del = menu.addAction("✕ Usuń węzeł (Del)")
        else:
            act_del = None

        action = menu.exec(event.screenPos())
        if action == act_dup:
            if self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.duplicate_node(self)
            elif action == act_edit:
                if self.node_type in ("input", "output", "output_window", "math_op", "scalar_math", "vaporfit",
                                      "spectrum_select", "output_table",
                                      "photometry", "dft_broadener",
                                      "svd_denoise", "mcr_decompose", "matrix_scaling",
                                      "internal_standard", "band_integrate", "merge_spectra",
                                      "hdo_spectrum_scaled", "synthetic_peak", "mc_params", "hdo_decomposition"):
                    dlg = NodeParamDialog(self)
                else:
                    dlg = StepParamDialog(self.step_data)
            if dlg.exec() == QDialog.Accepted and self.scene() and hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner.recalculate_graph_preview()
                self.scene().dialog_owner.sync_state_to_session()
        elif act_del and action == act_del:
            sc = self.scene()
            for p in list(self.in_ports.values()) + list(self.out_ports.values()):
                for c in list(p.connections):
                    c.disconnect()
            sc.removeItem(self)
            if hasattr(sc, 'dialog_owner'):
                sc.dialog_owner.recalculate_graph_preview()
                sc.dialog_owner.sync_state_to_session()