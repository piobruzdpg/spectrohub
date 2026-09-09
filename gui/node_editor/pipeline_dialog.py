"""
gui/node_editor/pipeline_dialog.py
Główne okno dialogowe edytora węzłów (Niemodalne, Hybrydowe).
Domyślnym celem jest 'Wszystkie widoczne w aktywnym oknie / gałęzi' z pełną izolacją magazynu.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy
import json
import logging

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QPushButton, QGraphicsView,
    QGraphicsScene, QMenu, QLabel, QComboBox, QCheckBox,
    QMessageBox, QToolBar, QFileDialog
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QBrush, QColor, QPen

from core.session import session
from core.models import SpectrumData, SpectralSeries
from gui.workspaces.ws_preprocessing import (
    ModifySpectrumCommand, BatchPipelineCommand,
    ModifySeriesCommand, AddBatchSpectraCommand
)
from gui.node_editor.items import NodeItem, PortItem, ConnectionItem
from gui.node_editor.evaluator import DAGEvaluator

logger = logging.getLogger("SpectroHub")


class NodeScene(QGraphicsScene):
    """Scena grafowa z obsługą przeciągania kabli, duplikacji i menu kontekstowego."""

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
                self.dialog_owner.sync_state_to_session()
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
                if item.connections and item.port_id != "in_atm":
                    item.connections[0].disconnect()
                self.temp_connection.in_port = item
                item.connections.append(self.temp_connection)
                self.temp_connection.update_path()
                self.temp_connection = None
                self.dialog_owner.recalculate_graph_preview()
                self.dialog_owner.sync_state_to_session()
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
    """Widok z siatką współrzędnych i zoomem kółkiem myszy."""

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


class NodePipelineDialog(QDialog):
    """Niemodalny edytor potoku DAG z pełną obsługą chemometrii i terminali okien."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Potok Przetwarzania Węzłowego (DAG Node Engine)")
        self.resize(1280, 800)

        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.scene = NodeScene(dialog_owner=self, parent=self)
        self.view = NodeView(self.scene, self)
        self.scene.selectionChanged.connect(self._on_selection_changed)

        self._init_ui()

        if session.dag_state and "nodes" in session.dag_state and session.dag_state["nodes"]:
            self.import_graph_state(session.dag_state)
        else:
            self._spawn_default_pipeline()
            self.sync_state_to_session()

        session.active_spectrum_changed.connect(self._on_session_active_spectrum_changed)
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

        btn_presets = QPushButton("📑 Szablony Potoków...")
        btn_presets.setStyleSheet("background-color: #4a4e69; font-weight: bold; padding: 5px 12px;")
        menu_presets = QMenu(self)
        self._populate_presets_menu(menu_presets)
        btn_presets.setMenu(menu_presets)
        toolbar.addWidget(btn_presets)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Cel potoku: "))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems([
            "Wszystkie widoczne w aktywnym oknie / gałęzi",  # Indeks 0 (DOMYŚLNY)
            "Tylko aktywne widmo",                         # Indeks 1
            "Aktywna seria 2D / Tensor"                    # Indeks 2
        ])
        self.combo_scope.setCurrentIndex(0)
        self.combo_scope.currentIndexChanged.connect(lambda: [self.recalculate_graph_preview(), self.sync_state_to_session()])
        toolbar.addWidget(self.combo_scope)

        self.chk_inplace = QCheckBox("In-place (z Undo)")
        self.chk_inplace.setChecked(True)
        self.chk_inplace.toggled.connect(lambda: self.sync_state_to_session())
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

    def sync_state_to_session(self):
        session.dag_state = self.export_graph_state()

    def _on_session_active_spectrum_changed(self, spec):
        if self.isVisible():
            self.recalculate_graph_preview()

    # ==========================================================================
    # MENU WĘZŁÓW
    # ==========================================================================
    def _populate_node_menu(self, menu: QMenu, pos: Optional[QPointF] = None):
        categories = {
            "Chemometria & Analiza Wielowymiarowa": [
                ("📊 Odszumiacz SVD / PCA (Rank-k)", {'type': 'svd_denoise', 'name': "SVD Denoise (k=2)", 'k': 2, 'center': False}),
                ("🔬 Dekompozycja MCR-ALS / SPEXFA", {'type': 'mcr_decompose', 'name': "MCR-ALS (k=2)", 'method': "mcr_als", 'k': 2, 'non_negative': True, 'closure': False, 'unimodal': False, 'max_iter': 500}),
                ("📐 Skalowania Populacyjne (Mean/UV/Pareto/MSC)", {'type': 'matrix_scaling', 'name': "Skalowanie: Mean-center", 'scale_type': "mean_center"})
            ],
            "Węzły Końcowe (Terminale)": [
                ("💾 Zapis Wyniku (In-place / proc_*)", {'type': 'output', 'name': "Zapis standardowy"}),
                ("🗂️ Generuj w nowym oknie roboczym", {'type': 'output_window', 'name': "🗂️ Okno: Skorygowane", 'window_name': "Skorygowane", 'mode': "create_new"})
            ],
            "Chemia Kwantowa (DFT)": [
                ("⚛️ Splot pasm DFT (Gauss / Lorentz)", {'type': 'dft_broadener', 'name': "DFT Broadener", 'dft_name': "", 'scale_factor': 0.965, 'fwhm': 12.0, 'shape': "lorentzian"})
            ],
            "Fotometria & Konwersje": [
                ("🔄 Konwerter Fotometryczny (A ↔ T ↔ %T ↔ KM)", {'type': 'photometry', 'name': "Fotometria: %T ➔ A", 'photo_mode': "tpct_to_a"})
            ],
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
                ("AsLS (Syg + Baza)", {'type': 'asls', 'name': "AsLS (λ=1e5, p=0.001)", 'lam': 1e5, 'p': 0.001}),
                ("airPLS / airALS (Syg + Baza)", {'type': 'airpls', 'name': "airPLS (λ=1e5)", 'lam': 1e5, 'n_iter': 15}),
                ("Wielomianowa (Syg + Baza)", {'type': 'poly_bl', 'name': "Wielomian st. 3", 'deg': 3}),
                ("Dwa krańce (Syg + Baza)", {'type': 'linear_bl', 'name': "Linia bazowa (krańce)"})
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
        self.sync_state_to_session()

    def duplicate_node(self, node: NodeItem, offset: QPointF = QPointF(35, 35)) -> Optional[NodeItem]:
        if node.node_type in ("input", "output"):
            return None
        clone = NodeItem(node.node_type, node.title, copy.deepcopy(node.step_data))
        clone.setPos(node.pos() + offset)
        self.scene.addItem(clone)
        self.scene.clearSelection()
        clone.setSelected(True)
        self.recalculate_graph_preview()
        self.sync_state_to_session()
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
        self.sync_state_to_session()

    def _clear_graph(self):
        self.scene.clear()
        self._spawn_default_pipeline()
        self.recalculate_graph_preview()
        self.sync_state_to_session()

    # ==========================================================================
    # MENU SZABLONÓW (PRESETS)
    # ==========================================================================
    def _populate_presets_menu(self, menu: QMenu):
        sub_factory = menu.addMenu("🧪 Wbudowane procedury laboratoryjne")

        act_svd = sub_factory.addAction("📊 Odszumianie SVD Rank-k + Linia Bazowa ➔ Nowe Okno")
        act_svd.triggered.connect(self._load_preset_svd_denoise)

        act_mcr = sub_factory.addAction("🔬 Rozdział MCR-ALS (Czyste składniki) ➔ Nowe Okno")
        act_mcr.triggered.connect(self._load_preset_mcr_pipeline)

        act_atm = sub_factory.addAction("☁️ Kompensacja Atmosferyczna + AsLS")
        act_atm.triggered.connect(self._load_preset_atmospheric_asls)

        act_amide = sub_factory.addAction("🧬 Analiza Amidu I (Białka 1720–1580 cm⁻¹)")
        act_amide.triggered.connect(self._load_preset_amide_i)

        act_deriv2 = sub_factory.addAction("🔍 Detekcja Składowych (2. Pochodna SG + Offset)")
        act_deriv2.triggered.connect(self._load_preset_second_derivative)

        act_diff = sub_factory.addAction("🧪 Spektroskopia Różnicowa (A - k·B)")
        act_diff.triggered.connect(self._load_preset_differential_math)

        act_dft = sub_factory.addAction("⚛️ Porównanie DFT z Eksperymentem (Splot + Odejmowanie)")
        act_dft.triggered.connect(self._load_preset_dft_comparison)

        menu.addSeparator()

        act_save_file = menu.addAction("💾 Zapisz potok jako szablon (*.json)...")
        act_save_file.triggered.connect(self._save_preset_to_json)

        act_load_file = menu.addAction("📂 Wczytaj szablon z pliku (*.json)...")
        act_load_file.triggered.connect(self._load_preset_from_json)

    def _load_preset_svd_denoise(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "svd_denoise", "title": "Odszumiacz SVD (k=3)", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "svd_denoise", "name": "SVD Denoise (k=3)", "k": 3, "center": False}},
                {"id": "n2", "type": "asls", "title": "Linia Bazowa AsLS", "pos_x": 540.0, "pos_y": 140.0, "step_data": {"type": "asls", "name": "AsLS (λ=1e5, p=0.001)", "lam": 1e5, "p": 0.001}},
                {"id": "n3", "type": "output_window", "title": "🗂️ Nowe Okno: Odszumione", "pos_x": 800.0, "pos_y": 140.0, "step_data": {"type": "output_window", "name": "🗂️ Okno: Odszumione SVD", "window_name": "Odszumione SVD", "mode": "create_new"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n1", "to_port": "in"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_mcr_pipeline(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "asls", "title": "Linia Bazowa AsLS", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "asls", "name": "AsLS (λ=1e5, p=0.001)", "lam": 1e5, "p": 0.001}},
                {"id": "n2", "type": "mcr_decompose", "title": "Dekompozycja MCR-ALS", "pos_x": 540.0, "pos_y": 140.0, "step_data": {"type": "mcr_decompose", "name": "MCR-ALS (k=2)", "method": "mcr_als", "k": 2, "non_negative": True, "closure": False, "unimodal": False, "max_iter": 500}},
                {"id": "n3", "type": "output_window", "title": "🗂️ Nowe Okno: Czyste Widma", "pos_x": 800.0, "pos_y": 140.0, "step_data": {"type": "output_window", "name": "🗂️ Okno: MCR Składniki", "window_name": "MCR Składniki", "mode": "create_new"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n1", "to_port": "in"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_atmospheric_asls(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "vaporfit", "title": "VaporFit (Para wodna i CO₂)", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "vaporfit", "name": "VaporFit (SG=11)", "sg_points": 11, "sg_poly": 3, "sign_mode": "auto"}},
                {"id": "n2", "type": "crop", "title": "Przycięcie (Crop)", "pos_x": 520.0, "pos_y": 140.0, "step_data": {"type": "crop", "name": "Przycięcie [1000–1800]", "min_w": 1000.0, "max_w": 1800.0}},
                {"id": "n3", "type": "asls", "title": "Linia Bazowa AsLS", "pos_x": 760.0, "pos_y": 140.0, "step_data": {"type": "asls", "name": "AsLS (λ=1e5, p=0.001)", "lam": 1e5, "p": 0.001}},
                {"id": "n4", "type": "output", "title": "💾 ZAPIS WYNIKU", "pos_x": 1000.0, "pos_y": 140.0, "step_data": {"type": "output", "name": "Zapis"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n1", "to_port": "in_signal"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"},
                {"from_node": "n3", "from_port": "out", "to_node": "n4", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_amide_i(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "crop", "title": "Przycięcie Amidu I", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "crop", "name": "Przycięcie [1580–1720]", "min_w": 1580.0, "max_w": 1720.0}},
                {"id": "n2", "type": "linear_bl", "title": "Linia bazowa (krańce)", "pos_x": 520.0, "pos_y": 140.0, "step_data": {"type": "linear_bl", "name": "Linia krańcowa"}},
                {"id": "n3", "type": "minmax", "title": "Normalizacja Min-Max", "pos_x": 760.0, "pos_y": 140.0, "step_data": {"type": "minmax", "name": "Min-Max [0, 1]"}},
                {"id": "n4", "type": "output", "title": "💾 ZAPIS WYNIKU", "pos_x": 1000.0, "pos_y": 140.0, "step_data": {"type": "output", "name": "Zapis"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n1", "to_port": "in"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"},
                {"from_node": "n3", "from_port": "out", "to_node": "n4", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_second_derivative(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "crop", "title": "Przycięcie Pasma", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "crop", "name": "Przycięcie [1500–1750]", "min_w": 1500.0, "max_w": 1750.0}},
                {"id": "n2", "type": "sg_smooth", "title": "Wygładzanie SG", "pos_x": 520.0, "pos_y": 140.0, "step_data": {"type": "sg_smooth", "name": "SG (okno=13, st=3)", "win": 13, "poly": 3}},
                {"id": "n3", "type": "deriv2", "title": "2. Pochodna (-) SG", "pos_x": 760.0, "pos_y": 140.0, "step_data": {"type": "deriv2", "name": "2. Pochodna SG (-) (okno=13)", "win": 13, "poly": 3}},
                {"id": "n4", "type": "offset_min", "title": "Offset (Min ➔ 0.0)", "pos_x": 1000.0, "pos_y": 140.0, "step_data": {"type": "offset_min", "name": "Offset Min"}},
                {"id": "n5", "type": "output", "title": "💾 ZAPIS WYNIKU", "pos_x": 1240.0, "pos_y": 140.0, "step_data": {"type": "output", "name": "Zapis"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n1", "to_port": "in"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"},
                {"from_node": "n3", "from_port": "out", "to_node": "n4", "to_port": "in"},
                {"from_node": "n4", "from_port": "out", "to_node": "n5", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_differential_math(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📈 Próbka Badana (A)", "pos_x": 40.0, "pos_y": 80.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "spectrum_select", "title": "Widmo Odniesienia (B)", "pos_x": 40.0, "pos_y": 240.0, "step_data": {"type": "spectrum_select", "name": "Ref: (wybierz)", "selected_spectrum": ""}},
                {"id": "n2", "type": "math_op", "title": "Operacja Dwuwidmowa", "pos_x": 340.0, "pos_y": 160.0, "step_data": {"type": "math_op", "name": "A - 1.000·B", "op": "sub", "k": 1.0}},
                {"id": "n3", "type": "linear_bl", "title": "Korekcja Linii Krańcowej", "pos_x": 580.0, "pos_y": 160.0, "step_data": {"type": "linear_bl", "name": "Linia krańcowa"}},
                {"id": "n4", "type": "output", "title": "💾 ZAPIS WYNIKU", "pos_x": 820.0, "pos_y": 160.0, "step_data": {"type": "output", "name": "Zapis"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n2", "to_port": "in_a"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in_b"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"},
                {"from_node": "n3", "from_port": "out", "to_node": "n4", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _load_preset_dft_comparison(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📈 Pomiary (A)", "pos_x": 40.0, "pos_y": 80.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "dft_broadener", "title": "⚛️ Splot DFT (B)", "pos_x": 40.0, "pos_y": 240.0, "step_data": {"type": "dft_broadener", "name": "DFT Broadener", "scale_factor": 0.965, "fwhm": 12.0, "shape": "lorentzian"}},
                {"id": "n2", "type": "math_op", "title": "Różnica A - k·B", "pos_x": 340.0, "pos_y": 160.0, "step_data": {"type": "math_op", "name": "A - 1.000·B", "op": "sub", "k": 1.0}},
                {"id": "n3", "type": "output_window", "title": "🗂️ Nowe Okno: Residua DFT", "pos_x": 620.0, "pos_y": 160.0, "step_data": {"type": "output_window", "name": "🗂️ Okno: Residua DFT", "window_name": "Residua DFT", "mode": "create_new"}}
            ],
            "connections": [
                {"from_node": "n0", "from_port": "out", "to_node": "n2", "to_port": "in_a"},
                {"from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in_b"},
                {"from_node": "n2", "from_port": "out", "to_node": "n3", "to_port": "in"}
            ]
        }
        self.import_graph_state(data)
        self.sync_state_to_session()

    def _save_preset_to_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz szablon potoku", "szablon_potoku.json", "Szablon Potoku SpectroHub (*.json)"
        )
        if path:
            try:
                state = self.export_graph_state()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Zapisano", f"Pomyślnie zapisano szablon potoku do pliku:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać szablonu:\n{e}")

    def _load_preset_from_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wczytaj szablon potoku", "", "Szablon Potoku SpectroHub (*.json)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.import_graph_state(state)
                self.sync_state_to_session()
                QMessageBox.information(self, "Załadowano", "Pomyślnie wczytano szablon potoku.")
            except Exception as e:
                QMessageBox.critical(self, "Błąd odczytu", f"Nie udało się wczytać szablonu:\n{e}")

    # ==========================================================================
    # SERIALIZACJA GRAFU DAG (DLA PROJEKTU .SHUB)
    # ==========================================================================
    def export_graph_state(self) -> Dict[str, Any]:
        """Eksportuje pełną topologię węzłów, współrzędne i kable grafu DAG."""
        nodes_list = []
        conns_list = []

        all_nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        node_to_id = {node: f"node_{idx}" for idx, node in enumerate(all_nodes)}

        for node, n_id in node_to_id.items():
            nodes_list.append({
                "id": n_id,
                "type": node.node_type,
                "title": node.title,
                "pos_x": float(node.pos().x()),
                "pos_y": float(node.pos().y()),
                "step_data": copy.deepcopy(node.step_data)
            })

        for node, n_id in node_to_id.items():
            for p_name, out_p in node.out_ports.items():
                for conn in out_p.connections:
                    if conn.in_port and conn.in_port.node_item in node_to_id:
                        in_n_id = node_to_id[conn.in_port.node_item]
                        conns_list.append({
                            "from_node": n_id,
                            "from_port": p_name,
                            "to_node": in_n_id,
                            "to_port": conn.in_port.port_id
                        })

        return {
            "nodes": nodes_list,
            "connections": conns_list,
            "scope": self.combo_scope.currentIndex(),
            "in_place": self.chk_inplace.isChecked()
        }

    def import_graph_state(self, data: Dict[str, Any]):
        """Odtwarza graf z zapisanego stanu manifestu projektu."""
        if not data or "nodes" not in data:
            return

        self.scene.clear()
        id_to_node: Dict[str, NodeItem] = {}

        for n_dict in data["nodes"]:
            node = NodeItem(
                node_type=n_dict["type"],
                title=n_dict["title"],
                step_data=copy.deepcopy(n_dict["step_data"])
            )
            node.setPos(n_dict["pos_x"], n_dict["pos_y"])
            self.scene.addItem(node)
            id_to_node[n_dict["id"]] = node

        for c_dict in data.get("connections", []):
            src_node = id_to_node.get(c_dict["from_node"])
            dst_node = id_to_node.get(c_dict["to_node"])
            if src_node and dst_node:
                out_p = src_node.out_ports.get(c_dict["from_port"])
                in_p = dst_node.in_ports.get(c_dict["to_port"])
                if out_p and in_p:
                    self.scene.addItem(ConnectionItem(out_p, in_p))

        if "scope" in data:
            self.combo_scope.setCurrentIndex(data["scope"])
        if "in_place" in data:
            self.chk_inplace.setChecked(data["in_place"])

        self.recalculate_graph_preview()

    # ==========================================================================
    # OBLICZENIA I LIVE PREVIEW
    # ==========================================================================
    def _get_base_preview_data(self) -> Tuple[np.ndarray, np.ndarray]:
        active = session.active_spectrum
        if active is not None and len(active.wavenumbers) > 2:
            return active.wavenumbers.copy(), active.intensities.copy()

        wn = np.linspace(4000, 400, 1200)
        y = 0.8 * np.exp(-((wn - 1650) / 35) ** 2) + 0.4 * np.exp(-((wn - 1540) / 40) ** 2) + 0.05 * np.random.normal(0, 0.02, len(wn))
        return wn, y

    def recalculate_graph_preview(self):
        all_nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        ordered_nodes, has_cycle = DAGEvaluator.topological_sort(all_nodes)

        if has_cycle:
            return

        base_x, base_y = self._get_base_preview_data()
        eval_results = DAGEvaluator.evaluate_dag(ordered_nodes, base_x, base_y, session.spectra)

        for node in all_nodes:
            main_key = (node, "out") if node.node_type not in ("output", "output_window") else (node, "in")
            res_x, res_y = eval_results.get(main_key, (None, None))

            if res_y is not None and res_y.ndim == 2:
                res_y = res_y[:, 0]

            node.preview_x = res_x
            node.preview_y = res_y
            node.update()

        self._update_live_canvas_preview()

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
                if isinstance(item, NodeItem) and item.node_type in ("output", "output_window"):
                    target_node = item
                    break

        if not target_node or target_node.preview_x is None or target_node.preview_y is None:
            self._clear_canvas_preview()
            return

        x = target_node.preview_x
        y = target_node.preview_y
        if y.ndim == 2:
            y = y[:, 0]

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
        self.sync_state_to_session()
        super().closeEvent(event)

    def reject(self):
        self._clear_canvas_preview()
        self.sync_state_to_session()
        super().reject()

    # ==========================================================================
    # WYKONANIE POTOKU DAG (EXECUTION)
    # ==========================================================================
    def _execute_pipeline(self):
        all_nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        ordered_nodes, has_cycle = DAGEvaluator.topological_sort(all_nodes)

        if has_cycle:
            QMessageBox.warning(self, "Błąd", "Wykryto zapętlenie w grafie (cykl). Potok musi być acykliczny.")
            return

        terminals = [n for n in all_nodes if n.node_type in ("output", "output_window")]
        if not terminals:
            QMessageBox.warning(self, "Błąd", "Brak węzła końcowego ('Zapis Wyniku' lub 'Generuj w nowym oknie').")
            return

        self._clear_canvas_preview()

        main_win = self.parent()
        canvas = getattr(main_win, 'canvas_2d', None)

        scope = self.combo_scope.currentIndex()
        in_place = self.chk_inplace.isChecked()

        try:
            for term_node in terminals:
                target_window_id = None
                if term_node.node_type == "output_window":
                    win_name = term_node.step_data.get("window_name", "Skorygowane")
                    mode = term_node.step_data.get("mode", "create_new")

                    if mode == "append_existing":
                        for wid, win in session.windows.items():
                            if win.name.lower() == win_name.lower():
                                target_window_id = wid
                                break
                    if not target_window_id:
                        target_window_id = session.add_window(name=win_name)

                # 1. WSZYSTKIE WIDOCZNE W AKTYWNYM OKNIE / GAŁĘZI (SCOPE 0 - DOMYŚLNY)
                if scope == 0:
                    visible = session.get_pipeline_target_spectra()
                    if not visible:
                        QMessageBox.warning(self, "Błąd", "Brak widocznych widm w aktywnym oknie lub gałęzi.")
                        return

                    ref_wn = visible[0].wavenumbers
                    mat = np.zeros((len(ref_wn), len(visible)), dtype=np.float64)
                    for i, s in enumerate(visible):
                        xw, yw = (s.wavenumbers[::-1], s.intensities[::-1]) if s.wavenumbers[0] > s.wavenumbers[-1] else (s.wavenumbers, s.intensities)
                        mat[:, i] = np.interp(ref_wn, xw, yw)

                    res_map = DAGEvaluator.evaluate_dag(ordered_nodes, ref_wn, mat, session.spectra)
                    new_x, new_y = res_map.get((term_node, "in"), (None, None))
                    if new_x is None or new_y is None:
                        continue

                    batch_records = []
                    new_created = []

                    # Przypadek A: Wyjście MCR (k czystych składników)
                    if new_y.ndim == 2 and new_y.shape[1] != len(visible):
                        for col in range(new_y.shape[1]):
                            new_created.append(SpectrumData(
                                wavenumbers=new_x.copy(), intensities=new_y[:, col].copy(),
                                name=f"MCR_Comp_{col + 1}", file_path="",
                                file_type="processed", metadata={"dag_pipeline": [n.title for n in ordered_nodes]}
                            ))
                        session.undo_stack.push(AddBatchSpectraCommand(
                            new_created, f"DAG MCR ({len(new_created)} składników)",
                            canvas=canvas, window_id=target_window_id
                        ))
                    # Przypadek B: Przetworzone widma
                    else:
                        for idx, spec in enumerate(visible):
                            old_x, old_y = spec.wavenumbers.copy(), spec.intensities.copy()
                            proc_y = new_y[:, idx] if new_y.ndim == 2 else new_y
                            new_meta = copy.deepcopy(spec.metadata)
                            new_meta["dag_pipeline"] = [n.title for n in ordered_nodes if n.node_type not in ("input", "output", "output_window")]

                            if term_node.node_type == "output_window":
                                new_created.append(SpectrumData(
                                    wavenumbers=new_x.copy(), intensities=proc_y.copy(),
                                    name=f"{spec.name}", file_path=spec.file_path,
                                    file_type="processed", metadata=new_meta
                                ))
                            elif in_place:
                                batch_records.append((spec, old_x, old_y, new_x.copy(), proc_y.copy(), spec.metadata, new_meta))
                            else:
                                new_created.append(SpectrumData(
                                    wavenumbers=new_x.copy(), intensities=proc_y.copy(),
                                    name=f"dag_{spec.name}", file_path=spec.file_path,
                                    file_type="processed", metadata=new_meta
                                ))

                        if term_node.node_type == "output_window" and new_created:
                            session.undo_stack.push(AddBatchSpectraCommand(
                                new_created, f"DAG do okna '{term_node.step_data.get('window_name')}'",
                                canvas=canvas, window_id=target_window_id
                            ))
                        elif in_place and batch_records:
                            session.undo_stack.push(BatchPipelineCommand(batch_records, f"DAG Batch ({len(batch_records)} widm)", canvas=canvas))
                        elif not in_place and new_created:
                            target_win = session.active_window_id if session.active_context == "window" else None
                            session.undo_stack.push(AddBatchSpectraCommand(new_created, f"DAG Batch ({len(new_created)} widm)", canvas=canvas, window_id=target_win))

                # 2. TYLKO AKTYWNE WIDMO (SCOPE 1)
                elif scope == 1:
                    active = session.active_spectrum
                    if not active:
                        QMessageBox.warning(self, "Błąd", "Brak aktywnego widma.")
                        return

                    res_map = DAGEvaluator.evaluate_dag(ordered_nodes, active.wavenumbers, active.intensities, session.spectra)
                    new_x, new_y = res_map.get((term_node, "in"), (None, None))
                    if new_x is None or new_y is None:
                        continue

                    new_meta = copy.deepcopy(active.metadata)
                    new_meta["dag_pipeline"] = [n.title for n in ordered_nodes if n.node_type not in ("input", "output", "output_window")]

                    if new_y.ndim == 2:
                        spectra_to_add = []
                        for col in range(new_y.shape[1]):
                            spectra_to_add.append(SpectrumData(
                                wavenumbers=new_x.copy(), intensities=new_y[:, col].copy(),
                                name=f"{active.name}_comp_{col + 1}", file_path=active.file_path,
                                file_type="processed", metadata=new_meta
                            ))
                        cmd = AddBatchSpectraCommand(spectra_to_add, f"DAG: {active.name} składowe", canvas=canvas, window_id=target_window_id)
                        session.undo_stack.push(cmd)
                    elif term_node.node_type == "output_window":
                        new_spec = SpectrumData(
                            wavenumbers=new_x, intensities=new_y, name=f"{active.name}",
                            file_path=active.file_path, file_type="processed", metadata=new_meta
                        )
                        cmd = AddBatchSpectraCommand([new_spec], f"DAG do okna '{term_node.step_data.get('window_name')}'",
                                                     canvas=canvas, window_id=target_window_id)
                        session.undo_stack.push(cmd)
                    elif in_place:
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
                        target_win = session.active_window_id if session.active_context == "window" else None
                        cmd = AddBatchSpectraCommand([new_spec], f"DAG: {active.name}", canvas=canvas, window_id=target_win)
                        session.undo_stack.push(cmd)

                # 3. SERIA 2D / TENSOR 3D (SCOPE 2)
                elif scope == 2:
                    active_s = session.active_series
                    if not active_s:
                        QMessageBox.warning(self, "Błąd", "Brak aktywnej serii 2D.")
                        return

                    mat = active_s.get_matrix_2d()
                    res_map = DAGEvaluator.evaluate_dag(ordered_nodes, active_s.wavenumbers, mat, session.spectra)
                    new_x, new_y = res_map.get((term_node, "in"), (None, None))
                    if new_x is None or new_y is None:
                        continue

                    if new_y.ndim == 2 and new_y.shape[1] != mat.shape[1]:
                        frames = [SpectrumData(
                            wavenumbers=new_x.copy(), intensities=new_y[:, c].copy(),
                            name=f"{active_s.name}_Comp_{c + 1}", file_path="",
                            file_type="processed", metadata={"parent_series": active_s.name}
                        ) for c in range(new_y.shape[1])]
                        session.undo_stack.push(AddBatchSpectraCommand(
                            frames, f"Składniki MCR do okna '{term_node.step_data.get('window_name')}'",
                            canvas=canvas, window_id=target_window_id
                        ))
                    elif term_node.node_type == "output_window":
                        frames = [SpectrumData(
                            wavenumbers=new_x.copy(), intensities=new_y[:, c].copy(),
                            name=f"{active_s.name}_col{c + 1}", file_path="",
                            file_type="series_frame", metadata={"parent_series": active_s.name}
                        ) for c in range(new_y.shape[1])]
                        session.undo_stack.push(AddBatchSpectraCommand(
                            frames, f"Seria do okna '{term_node.step_data.get('window_name')}'",
                            canvas=canvas, window_id=target_window_id
                        ))
                    elif in_place:
                        new_data = new_y.reshape(len(new_x), active_s.data.shape[1], active_s.data.shape[2]) if active_s.is_3d else new_y
                        cmd = ModifySeriesCommand(active_s, f"DAG Serii: {active_s.name}",
                                                  active_s.wavenumbers, new_x, active_s.data, new_data, canvas=canvas)
                        session.undo_stack.push(cmd)
                    else:
                        out_data = new_y.reshape(len(new_x), active_s.data.shape[1], active_s.data.shape[2]) if active_s.is_3d else new_y
                        session.add_series(SpectralSeries(
                            name=f"dag_{active_s.name}", wavenumbers=new_x, data=out_data,
                            axis_n_name=active_s.axis_n_name, axis_n_values=active_s.axis_n_values,
                            axis_m_name=active_s.axis_m_name, axis_m_values=active_s.axis_m_values
                        ), make_active=True)

            QMessageBox.information(self, "Sukces", f"Pomyślnie wykonano potok DAG!")
            self.recalculate_graph_preview()
            self.sync_state_to_session()

        except Exception as e:
            logger.error(f"Błąd wykonania potoku DAG: {e}")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas obliczeń:\n{e}")