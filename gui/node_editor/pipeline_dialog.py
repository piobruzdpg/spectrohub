"""
gui/node_editor/pipeline_dialog.py
Główne okno edytora węzłów (QMainWindow) z architekturą wielozakładkową (Multi-Tab),
przypięciem do lewej, wydzielaniem przez przeciągnięcie (Tear-off) lub menu PPM oraz dokowalną paletą.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy
import json
import logging

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QMenu, QLabel,
    QComboBox, QCheckBox, QMessageBox, QToolBar, QFileDialog,
    QTabWidget, QTabBar, QInputDialog, QProxyStyle, QStyle
)

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, QSettings, Signal
from PySide6.QtGui import QPainter, QBrush, QColor, QPen

from core.session import session
from core.models import SpectrumData, SpectralSeries
from gui.workspaces.ws_preprocessing import (
    ModifySpectrumCommand, BatchPipelineCommand,
    ModifySeriesCommand, AddBatchSpectraCommand
)
from gui.node_editor.items import NodeItem, PortItem, ConnectionItem
from gui.node_editor.evaluator import DAGEvaluator
from gui.node_editor.palette import NodeLibraryWidget, MIME_NODE_TYPE

logger = logging.getLogger("SpectroHub")


class PipelineTabInfo:
    """Kontener metadanych pojedynczej zakładki potoku."""
    def __init__(self, title: str, view: 'NodeView', scene: 'NodeScene',
                 scope: int = 0, in_place: bool = True):
        self.title = title
        self.view = view
        self.scene = scene
        self.scope = scope
        self.in_place = in_place
        self.floating_window: Optional['FloatingPipelineWindow'] = None


class LeftAlignedTabStyle(QProxyStyle):
    """Wymusza wyrównanie zakładek do lewej krawędzi (omija domyślne centrowanie na macOS)."""
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        hint_val = int(hint.value) if hasattr(hint, 'value') else int(hint)
        target_val = int(QStyle.StyleHint.SH_TabBar_Alignment.value) if hasattr(QStyle.StyleHint.SH_TabBar_Alignment, 'value') else int(QStyle.StyleHint.SH_TabBar_Alignment)
        if hint_val == target_val:
            return int(Qt.AlignmentFlag.AlignLeft)
        return super().styleHint(hint, option, widget, returnData)

class DetachableTabBar(QTabBar):
    """Pasek zakładek z obsługą odrywania przez przeciągnięcie poza pasek oraz lewostronnego układu."""
    tabDetached = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setExpanding(False)
        self._press_pos = QPointF()
        self._drag_tab_idx = -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position()
            self._drag_tab_idx = self.tabAt(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_tab_idx >= 0:
            drop_pos = event.position().toPoint()
            # Jeśli kursor został upuszczony poza paskiem zakładek
            if not self.rect().adjusted(-30, -30, 30, 45).contains(drop_pos):
                idx = self._drag_tab_idx
                self._drag_tab_idx = -1
                self.tabDetached.emit(idx)
                event.accept()
                return
        self._drag_tab_idx = -1
        super().mouseReleaseEvent(event)


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
                self.dialog_owner.recalculate_graph_preview(target_scene=self)
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
                self.dialog_owner.recalculate_graph_preview(target_scene=self)
                self.dialog_owner.sync_state_to_session()
            else:
                self.temp_connection.disconnect()
                self.temp_connection = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.dialog_owner._delete_selected(scene=self)
            event.accept()
            return

        if event.key() == Qt.Key_D and (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            selected_nodes = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
            for node in selected_nodes:
                self.dialog_owner.duplicate_node(node, scene=self)
            event.accept()
            return

        if event.key() == Qt.Key_F and not (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            if self.views():
                self.views()[0].fit_to_content()
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
            self.dialog_owner._populate_node_menu(menu, pos=event.scenePos(), target_scene=self)
            menu.exec(event.screenPos())
            event.accept()
            return
        super().contextMenuEvent(event)


class NodeView(QGraphicsView):
    """Widok z siatką współrzędnych, zoomem, panoramowaniem i obsługą upuszczania węzłów."""

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor("#18181e")))
        self.setAcceptDrops(True)
        self._is_panning = False
        self._pan_start = QPointF()

    def fit_to_content(self, margin: float = 60.0):
        """Centruje i skaluje widok tak, aby objąć wszystkie bloczki węzłów."""
        nodes = [item for item in self.scene().items() if isinstance(item, NodeItem)]
        if not nodes:
            return

        rect = nodes[0].sceneBoundingRect()
        for node in nodes[1:]:
            rect = rect.united(node.sceneBoundingRect())

        if rect.width() < 700:
            cx = rect.center().x()
            rect.setLeft(cx - 350)
            rect.setRight(cx + 350)
        if rect.height() < 400:
            cy = rect.center().y()
            rect.setTop(cy - 200)
            rect.setBottom(cy + 200)

        rect.adjust(-margin, -margin, margin, margin)
        self.fitInView(rect, Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F and not (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            self.fit_to_content()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier)):
            self._is_panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and (event.button() in (Qt.MiddleButton, Qt.LeftButton)):
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_NODE_TYPE):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_NODE_TYPE):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(MIME_NODE_TYPE):
            raw_data = event.mimeData().data(MIME_NODE_TYPE).data().decode("utf-8")
            payload = json.loads(raw_data)
            scene_pos = self.mapToScene(event.position().toPoint())

            if hasattr(self.scene(), 'dialog_owner'):
                self.scene().dialog_owner._add_processing_node(
                    title=payload["title"],
                    step_data=payload["step_data"],
                    pos=scene_pos,
                    scene=self.scene()
                )
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

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


class FloatingPipelineWindow(QMainWindow):
    """Niezależne okno pływające dla wydzielonej zakładki potoku."""

    def __init__(self, tab_info: PipelineTabInfo, dialog_owner: 'NodePipelineDialog'):
        super().__init__(dialog_owner)
        self.tab_info = tab_info
        self.dialog_owner = dialog_owner
        self.setWindowTitle(f"Potok: {tab_info.title} — SpectroHub")
        self.resize(1020, 720)
        self.setWindowFlags(Qt.Window)

        toolbar = QToolBar("Narzędzia", self)
        toolbar.setStyleSheet("background-color: #252530; spacing: 8px; padding: 4px;")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        btn_redock = QPushButton("📌 Przypnij z powrotem do zakładek")
        btn_redock.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 5px 12px;")
        btn_redock.clicked.connect(self.redock)
        toolbar.addWidget(btn_redock)

        btn_fit = QPushButton("⛶ Dopasuj (F)")
        btn_fit.clicked.connect(self.tab_info.view.fit_to_content)
        toolbar.addWidget(btn_fit)

        toolbar.addSeparator()

        btn_run = QPushButton("⚡ Uruchom ten potok")
        btn_run.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 5px 14px;")
        btn_run.clicked.connect(lambda: self.dialog_owner.execute_tab_pipeline(self.tab_info))
        toolbar.addWidget(btn_run)

        # Kluczowe: przywrócenie widoczności widoku po wyjęciu z QTabWidget
        self.tab_info.view.setVisible(True)
        self.tab_info.view.show()
        self.setCentralWidget(self.tab_info.view)

        QTimer.singleShot(60, self._on_initial_show)

    def _on_initial_show(self):
        self.tab_info.view.setVisible(True)
        self.tab_info.view.show()
        self.tab_info.view.viewport().update()
        self.tab_info.view.fit_to_content()

    def redock(self):
        self.dialog_owner.reattach_tab(self.tab_info)
        self.close()

    def closeEvent(self, event):
        self.dialog_owner.reattach_tab(self.tab_info)
        super().closeEvent(event)


class NodePipelineDialog(QMainWindow):
    """Główne okno edytora potoku DAG z architekturą wielozakładkową i wydzielanymi oknami."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Potok Przetwarzania Węzłowego (DAG Node Engine)")
        self.resize(1360, 840)
        self.setWindowFlags(Qt.Window)

        self.tabs: List[PipelineTabInfo] = []
        self._tab_counter = 0

        # Centralny widget zakładek z własnym paskiem
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setDocumentMode(False)  # Wyłączone: zapobiega natywnemu centrowaniu macOS
        self.tab_bar = DetachableTabBar(self)
        self.tab_widget.setTabBar(self.tab_bar)

        self.tab_widget.setTabsClosable(True)
        self.tab_bar.tabDetached.connect(self._detach_tab_by_index)

        self.tab_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tab_bar.customContextMenuRequested.connect(self._on_tab_bar_context_menu)

        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.tab_widget.tabBarDoubleClicked.connect(self._on_tab_double_clicked)

        # Arkusz stylów zakładek
        self.tab_widget.setStyleSheet("""
                    QTabWidget::pane {
                        border-top: 1px solid #2d2d3a;
                        background-color: #18181e;
                    }
                    QTabWidget::tab-bar {
                        left: 0px;
                    }
                    QTabBar {
                        qproperty-expanding: false;
                        background-color: #18181e;
                    }
                    QTabBar::tab {
                        background-color: #22222f;
                        color: #b0b0c0;
                        padding: 7px 16px;
                        margin-right: 2px;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                        border: 1px solid #2d2d3a;
                        border-bottom: none;
                        font-size: 12px;
                    }
                    QTabBar::tab:selected {
                        background-color: #18181e;
                        color: #48cae4;
                        border: 1px solid #3d3d4e;
                        border-bottom: 1px solid #18181e;
                        font-weight: bold;
                    }
                    QTabBar::tab:hover:!selected {
                        background-color: #2b2b3d;
                        color: #ffffff;
                    }
                """)

        # KLUCZOWY KROK:
        # Opakowujemy styl QStyleSheetStyle stworzony przez powyższy setStyleSheet
        # i trzymamy referencję w self._tab_bar_style, aby zapobiec usunięciu przez GC.
        self._tab_bar_style = LeftAlignedTabStyle(self.tab_bar.style())
        self.tab_bar.setStyle(self._tab_bar_style)
        self.tab_bar.setExpanding(False)

        # Przycisk [+] po prawej stronie
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 6, 0)
        btn_new_tab = QPushButton("➕ Nowy potok")
        btn_new_tab.setStyleSheet("""
                    QPushButton {
                        background-color: #2a2a38;
                        color: #48cae4;
                        border: 1px solid #3d3d4e;
                        border-radius: 3px;
                        padding: 4px 10px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #3d5a80;
                        color: #ffffff;
                    }
                """)
        btn_new_tab.clicked.connect(lambda: self.add_pipeline_tab())
        corner_layout.addWidget(btn_new_tab)
        self.tab_widget.setCornerWidget(corner_widget, Qt.TopRightCorner)

        self.setCentralWidget(self.tab_widget)

        self._setup_docks()
        self._init_toolbar()
        self._restore_ui_state()

        if session.dag_state:
            self.import_graph_state(session.dag_state)
        else:
            self.add_pipeline_tab(title="Główny potok")
            self.sync_state_to_session()

        session.active_spectrum_changed.connect(self._on_session_active_spectrum_changed)

    # --- WŁAŚCIWOŚCI DLA AKTYWNEJ ZAKŁADKI ---
    @property
    def current_tab(self) -> Optional[PipelineTabInfo]:
        curr_widget = self.tab_widget.currentWidget()
        if not curr_widget:
            return None
        return self._get_tab_info_by_view(curr_widget)

    @property
    def scene(self) -> Optional[NodeScene]:
        tab = self.current_tab
        return tab.scene if tab else None

    @property
    def view(self) -> Optional[NodeView]:
        tab = self.current_tab
        return tab.view if tab else None

    def _get_tab_info_by_view(self, view: QWidget) -> Optional[PipelineTabInfo]:
        for t in self.tabs:
            if t.view == view:
                return t
        return None

    # --- ZARZĄDZANIE ZAKŁADKAMI ---
    def add_pipeline_tab(self, title: Optional[str] = None, graph_data: Optional[Dict[str, Any]] = None,
                         make_active: bool = True) -> PipelineTabInfo:
        self._tab_counter += 1
        t_name = title or f"Potok {self._tab_counter}"

        scene = NodeScene(dialog_owner=self, parent=self)
        view = NodeView(scene, self)
        scene.selectionChanged.connect(self._on_selection_changed)

        scope = graph_data.get("scope", 0) if graph_data else 0
        in_place = graph_data.get("in_place", True) if graph_data else True

        if graph_data and "nodes" in graph_data:
            self._deserialize_to_scene(scene, graph_data)
        else:
            self._spawn_default_pipeline(scene)

        tab_info = PipelineTabInfo(t_name, view, scene, scope=scope, in_place=in_place)
        self.tabs.append(tab_info)

        idx = self.tab_widget.addTab(view, t_name)
        if make_active:
            self.tab_widget.setCurrentIndex(idx)

        QTimer.singleShot(60, view.fit_to_content)
        self.recalculate_graph_preview(target_scene=scene)
        self.sync_state_to_session()
        return tab_info

    def _on_tab_changed(self, index: int):
        tab = self.current_tab
        if not tab:
            return

        self.combo_scope.blockSignals(True)
        self.chk_inplace.blockSignals(True)
        self.combo_scope.setCurrentIndex(tab.scope)
        self.chk_inplace.setChecked(tab.in_place)
        self.combo_scope.blockSignals(False)
        self.chk_inplace.blockSignals(False)

        self.recalculate_graph_preview()
        self._update_live_canvas_preview()
        self.sync_state_to_session()

    def _on_tab_double_clicked(self, index: int):
        if index < 0 or index >= self.tab_widget.count():
            return
        cur_title = self.tab_widget.tabText(index)
        new_title, ok = QInputDialog.getText(self, "Zmień nazwę potoku", "Wprowadź nową nazwę zakładki:", text=cur_title)
        if ok and new_title.strip():
            new_title = new_title.strip()
            self.tab_widget.setTabText(index, new_title)
            tab = self._get_tab_info_by_view(self.tab_widget.widget(index))
            if tab:
                tab.title = new_title
                if tab.floating_window:
                    tab.floating_window.setWindowTitle(f"Potok: {new_title} — SpectroHub")
            self.sync_state_to_session()

    def _on_tab_close_requested(self, index: int):
        if index < 0 or index >= self.tab_widget.count():
            return

        if self.tab_widget.count() == 1:
            reply = QMessageBox.question(
                self, "Resetuj potok",
                "To jest jedyna otwarta zakładka. Czy chcesz zresetować potok do stanu domyślnego?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._clear_graph()
            return

        view = self.tab_widget.widget(index)
        tab_info = self._get_tab_info_by_view(view)
        if tab_info and tab_info.floating_window:
            tab_info.floating_window.close()

        self.tab_widget.removeTab(index)
        if tab_info in self.tabs:
            self.tabs.remove(tab_info)

        self.recalculate_graph_preview()
        self.sync_state_to_session()

    def _on_tab_bar_context_menu(self, point):
        tab_idx = self.tab_bar.tabAt(point)
        if tab_idx < 0:
            menu = QMenu(self)
            act_new = menu.addAction("➕ Nowy potok")
            act_new.triggered.connect(lambda: self.add_pipeline_tab())
            menu.exec(self.tab_bar.mapToGlobal(point))
            return

        menu = QMenu(self)
        act_detach = menu.addAction("⧉ Wydziel do osobnego okna")
        act_detach.triggered.connect(lambda: self._detach_tab_by_index(tab_idx))

        act_rename = menu.addAction("✏️ Zmień nazwę zakładki...")
        act_rename.triggered.connect(lambda: self._on_tab_double_clicked(tab_idx))

        act_duplicate = menu.addAction("📋 Duplikuj potok")
        act_duplicate.triggered.connect(lambda: self._duplicate_tab_by_index(tab_idx))

        menu.addSeparator()
        act_close = menu.addAction("✕ Zamknij zakładkę")
        act_close.triggered.connect(lambda: self._on_tab_close_requested(tab_idx))

        menu.exec(self.tab_bar.mapToGlobal(point))

    def _duplicate_tab_by_index(self, index: int):
        view = self.tab_widget.widget(index)
        tab = self._get_tab_info_by_view(view)
        if not tab:
            return

        nodes_list, conns_list = self._serialize_scene(tab.scene)
        data = {
            "nodes": nodes_list,
            "connections": conns_list,
            "scope": tab.scope,
            "in_place": tab.in_place
        }
        self.add_pipeline_tab(title=f"{tab.title} (Kopia)", graph_data=data, make_active=True)

    # --- WYDZIELANIE I PRZYPINANIE OKIEN ---
    def _detach_tab_by_index(self, index: int):
        if index < 0 or index >= self.tab_widget.count():
            return
        view = self.tab_widget.widget(index)
        tab = self._get_tab_info_by_view(view)
        if not tab or tab.floating_window is not None:
            return

        self.tab_widget.removeTab(index)

        floating_win = FloatingPipelineWindow(tab, self)
        tab.floating_window = floating_win
        floating_win.show()
        floating_win.raise_()
        floating_win.activateWindow()

        if self.tab_widget.count() == 0:
            self.add_pipeline_tab(title="Nowy potok", make_active=True)

    def reattach_tab(self, tab_info: PipelineTabInfo):
        tab_info.floating_window = None
        tab_info.view.setVisible(True)
        tab_info.view.show()
        if self.tab_widget.indexOf(tab_info.view) == -1:
            idx = self.tab_widget.addTab(tab_info.view, tab_info.title)
            self.tab_widget.setCurrentIndex(idx)
            QTimer.singleShot(60, tab_info.view.fit_to_content)
            self.recalculate_graph_preview()
            self.sync_state_to_session()

    # --- CYKL ŻYCIA I ZAPIS UKŁADU ---
    def closeEvent(self, event):
        for tab in list(self.tabs):
            if tab.floating_window:
                tab.floating_window.close()

        self._save_ui_state()
        self._clear_canvas_preview()
        self.sync_state_to_session()
        super().closeEvent(event)

    def reject(self):
        self.close()

    def _save_ui_state(self):
        settings = QSettings("SpectroHub", "NodeEditor")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def _restore_ui_state(self):
        settings = QSettings("SpectroHub", "NodeEditor")
        geo = settings.value("geometry")
        state = settings.value("windowState")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)

    def _init_toolbar(self):
        toolbar = QToolBar("Główny pasek narzędzi", self)
        toolbar.setStyleSheet("background-color: #252530; spacing: 8px; padding: 4px;")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addAction(self.dock_library.toggleViewAction())

        btn_fit = QPushButton("⛶ Dopasuj widok (F)")
        btn_fit.setToolTip("Centruje i dopasowuje skalę do wszystkich węzłów (Skrót: F)")
        btn_fit.clicked.connect(lambda: self.view.fit_to_content() if self.view else None)
        toolbar.addWidget(btn_fit)

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
            "Wszystkie widoczne w aktywnym oknie / gałęzi",
            "Tylko aktywne widmo",
            "Aktywna seria 2D / Tensor"
        ])
        self.combo_scope.currentIndexChanged.connect(self._on_scope_changed)
        toolbar.addWidget(self.combo_scope)

        self.chk_inplace = QCheckBox("In-place (z Undo)")
        self.chk_inplace.setChecked(True)
        self.chk_inplace.toggled.connect(self._on_inplace_toggled)
        toolbar.addWidget(self.chk_inplace)

        toolbar.addSeparator()

        self.chk_live_preview = QCheckBox("👁 Live Preview na Canvasie")
        self.chk_live_preview.setChecked(True)
        self.chk_live_preview.setStyleSheet("color: #ffb703; font-weight: bold;")
        self.chk_live_preview.toggled.connect(self._toggle_live_preview)
        toolbar.addWidget(self.chk_live_preview)

        toolbar.addSeparator()

        btn_del = QPushButton("✕ Usuń (Del)")
        btn_del.clicked.connect(lambda: self._delete_selected())
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

    def _on_scope_changed(self, idx: int):
        if self.current_tab:
            self.current_tab.scope = idx
            self.recalculate_graph_preview()
            self.sync_state_to_session()

    def _on_inplace_toggled(self, checked: bool):
        if self.current_tab:
            self.current_tab.in_place = checked
            self.sync_state_to_session()

    def _setup_docks(self):
        self.dock_library = QDockWidget("📚 Biblioteka Węzłów", self)
        self.dock_library.setObjectName("DockNodeLibrary")
        self.dock_library.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        categories = self._get_node_categories()
        self.palette_widget = NodeLibraryWidget(
            categories_dict=categories,
            on_node_double_clicked=self._on_palette_node_double_clicked,
            parent=self.dock_library
        )
        self.dock_library.setWidget(self.palette_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_library)

    def _on_palette_node_double_clicked(self, title: str, step_data: dict):
        if self.view:
            center_scene = self.view.mapToScene(self.view.viewport().rect().center())
            self._add_processing_node(title, step_data, pos=center_scene)

    def _get_node_categories(self) -> Dict[str, list]:
        return {
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
                ("☁️ VaporFit (Para wodna i CO₂)", {'type': 'vaporfit', 'name': "VaporFit (SG=11)", 'sg_points': 11, 'sg_poly': 3}),
                ("📌 Widmo Odniesienia z Projektu", {'type': 'spectrum_select', 'name': "Ref: (wybierz)", 'selected_spectrum': ""})
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

    def _populate_node_menu(self, menu: QMenu, pos: Optional[QPointF] = None, target_scene: Optional[NodeScene] = None):
        categories = self._get_node_categories()
        for cat_name, items in categories.items():
            sub = menu.addMenu(cat_name)
            for label, data in items:
                act = sub.addAction(label)
                act.triggered.connect(lambda chk=False, d=data, l=label, p=pos, s=target_scene: self._add_processing_node(l, d, p, scene=s))

    def _add_processing_node(self, title: str, step_data: Dict[str, Any],
                             pos: Optional[QPointF] = None, scene: Optional[NodeScene] = None):
        target_scene = scene or self.scene
        if not target_scene:
            return

        node = NodeItem(node_type=step_data['type'], title=title, step_data=copy.deepcopy(step_data))
        if pos is not None:
            node.setPos(pos)
        else:
            v = self.view
            center = v.mapToScene(v.viewport().rect().center()) if v else QPointF(100, 100)
            node.setPos(center + QPointF(np.random.randint(-40, 40), np.random.randint(-40, 40)))

        target_scene.addItem(node)
        self.recalculate_graph_preview(target_scene=target_scene)
        self.sync_state_to_session()

    def duplicate_node(self, node: NodeItem, offset: QPointF = QPointF(35, 35),
                       scene: Optional[NodeScene] = None) -> Optional[NodeItem]:
        target_scene = scene or self.scene
        if not target_scene or node.node_type in ("input", "output"):
            return None

        clone = NodeItem(node.node_type, node.title, copy.deepcopy(node.step_data))
        clone.setPos(node.pos() + offset)
        target_scene.addItem(clone)
        target_scene.clearSelection()
        clone.setSelected(True)

        self.recalculate_graph_preview(target_scene=target_scene)
        self.sync_state_to_session()
        return clone

    def _spawn_default_pipeline(self, scene: Optional[NodeScene] = None):
        target_scene = scene or self.scene
        if not target_scene:
            return

        in_node = NodeItem(node_type="input", title="📥 ŹRÓDŁO DANYCH")
        in_node.setPos(40, 120)
        target_scene.addItem(in_node)

        crop_node = NodeItem(node_type="crop", title="Przycięcie (Crop)",
                             step_data={'type': 'crop', 'name': "Przycięcie [1000–1800]", 'min_w': 1000.0, 'max_w': 1800.0})
        crop_node.setPos(280, 120)
        target_scene.addItem(crop_node)

        asls_node = NodeItem(node_type="asls", title="Linia Bazowa AsLS",
                             step_data={'type': 'asls', 'name': "AsLS (λ=1e5, p=0.001)", 'lam': 1e5, 'p': 0.001})
        asls_node.setPos(520, 120)
        target_scene.addItem(asls_node)

        out_node = NodeItem(node_type="output", title="💾 ZAPIS WYNIKU")
        out_node.setPos(760, 120)
        target_scene.addItem(out_node)

        target_scene.addItem(ConnectionItem(in_node.out_ports["out"], crop_node.in_ports["in"]))
        target_scene.addItem(ConnectionItem(crop_node.out_ports["out"], asls_node.in_ports["in"]))
        target_scene.addItem(ConnectionItem(asls_node.out_ports["out"], out_node.in_ports["in"]))

    def _delete_selected(self, scene: Optional[NodeScene] = None):
        target_scene = scene or self.scene
        if not target_scene:
            return

        for item in target_scene.selectedItems():
            if isinstance(item, NodeItem):
                if item.node_type in ("input", "output"):
                    continue
                for p in list(item.in_ports.values()) + list(item.out_ports.values()):
                    for c in list(p.connections):
                        c.disconnect()
                target_scene.removeItem(item)
            elif isinstance(item, ConnectionItem):
                item.disconnect()

        self.recalculate_graph_preview(target_scene=target_scene)
        self.sync_state_to_session()

    def _clear_graph(self):
        if not self.scene:
            return
        self.scene.clear()
        self._spawn_default_pipeline(self.scene)
        self.recalculate_graph_preview(target_scene=self.scene)
        self.sync_state_to_session()

    # --- PROCEDURY LABORATORYJNE (PRESETS) ---
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
        self.add_pipeline_tab(title="Odszumianie SVD", graph_data=data, make_active=True)

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
        self.add_pipeline_tab(title="MCR-ALS", graph_data=data, make_active=True)

    def _load_preset_atmospheric_asls(self):
        data = {
            "nodes": [
                {"id": "n0", "type": "input", "title": "📥 ŹRÓDŁO DANYCH", "pos_x": 40.0, "pos_y": 140.0, "step_data": {"type": "input", "name": "Źródło"}},
                {"id": "n1", "type": "vaporfit", "title": "VaporFit (Para wodna i CO₂)", "pos_x": 280.0, "pos_y": 140.0, "step_data": {"type": "vaporfit", "name": "VaporFit (SG=11)", "sg_points": 11, "sg_poly": 3}},
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
        self.add_pipeline_tab(title="VaporFit + AsLS", graph_data=data, make_active=True)

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
        self.add_pipeline_tab(title="Amid I", graph_data=data, make_active=True)

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
        self.add_pipeline_tab(title="2. Pochodna", graph_data=data, make_active=True)

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
        self.add_pipeline_tab(title="Różnicowa", graph_data=data, make_active=True)

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
        self.add_pipeline_tab(title="Porównanie DFT", graph_data=data, make_active=True)

    def _save_preset_to_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz szablon potoku", "szablon_potoku.json", "Szablon Potoku SpectroHub (*.json)"
        )
        if path:
            try:
                curr = self.current_tab
                if not curr:
                    return
                nodes_list, conns_list = self._serialize_scene(curr.scene)
                state = {
                    "title": curr.title,
                    "nodes": nodes_list,
                    "connections": conns_list,
                    "scope": curr.scope,
                    "in_place": curr.in_place
                }
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
                    data = json.load(f)
                title = data.get("title", "Wczytany potok")
                self.add_pipeline_tab(title=title, graph_data=data, make_active=True)
                QMessageBox.information(self, "Załadowano", f"Wczytano szablon potoku jako nową zakładkę:\n'{title}'.")
            except Exception as e:
                QMessageBox.critical(self, "Błąd odczytu", f"Nie udało się wczytać szablonu:\n{e}")

    # --- SERIALIZACJA GRAFU ---
    def _serialize_scene(self, scene: NodeScene) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes_list, conns_list = [], []
        all_nodes = [item for item in scene.items() if isinstance(item, NodeItem)]
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

        return nodes_list, conns_list

    def _deserialize_to_scene(self, scene: NodeScene, data: Dict[str, Any]):
        scene.clear()
        if not data or "nodes" not in data:
            return

        id_to_node: Dict[str, NodeItem] = {}
        for n_dict in data["nodes"]:
            node = NodeItem(
                node_type=n_dict["type"],
                title=n_dict["title"],
                step_data=copy.deepcopy(n_dict["step_data"])
            )
            node.setPos(n_dict["pos_x"], n_dict["pos_y"])
            scene.addItem(node)
            id_to_node[n_dict["id"]] = node

        for c_dict in data.get("connections", []):
            src_node = id_to_node.get(c_dict["from_node"])
            dst_node = id_to_node.get(c_dict["to_node"])
            if src_node and dst_node:
                out_p = src_node.out_ports.get(c_dict["from_port"])
                in_p = dst_node.in_ports.get(c_dict["to_port"])
                if out_p and in_p:
                    scene.addItem(ConnectionItem(out_p, in_p))

    def export_graph_state(self) -> Dict[str, Any]:
        tabs_data = []
        for tab in self.tabs:
            n_list, c_list = self._serialize_scene(tab.scene)
            tabs_data.append({
                "title": tab.title,
                "nodes": n_list,
                "connections": c_list,
                "scope": tab.scope,
                "in_place": tab.in_place
            })

        active_idx = self.tab_widget.currentIndex()
        if active_idx < 0 and tabs_data:
            active_idx = 0

        active_dict = tabs_data[active_idx] if (0 <= active_idx < len(tabs_data)) else (tabs_data[0] if tabs_data else {})

        return {
            "version": 2,
            "active_tab": active_idx,
            "tabs": tabs_data,
            "nodes": active_dict.get("nodes", []),
            "connections": active_dict.get("connections", []),
            "scope": active_dict.get("scope", 0),
            "in_place": active_dict.get("in_place", True)
        }

    def import_graph_state(self, data: Dict[str, Any]):
        if not data:
            return

        for tab in list(self.tabs):
            if tab.floating_window:
                tab.floating_window.close()

        self.tab_widget.blockSignals(True)
        self.tab_widget.clear()
        self.tabs.clear()

        if "tabs" in data and isinstance(data["tabs"], list) and len(data["tabs"]) > 0:
            for t_dict in data["tabs"]:
                t_title = t_dict.get("title", "Potok")
                self.add_pipeline_tab(title=t_title, graph_data=t_dict, make_active=False)
            act_idx = data.get("active_tab", 0)
            if 0 <= act_idx < self.tab_widget.count():
                self.tab_widget.setCurrentIndex(act_idx)
        elif "nodes" in data:
            self.add_pipeline_tab(title="Główny potok", graph_data=data, make_active=True)
        else:
            self.add_pipeline_tab(title="Główny potok", make_active=True)

        self.tab_widget.blockSignals(False)
        self._on_tab_changed(self.tab_widget.currentIndex())

    def sync_state_to_session(self):
        session.dag_state = self.export_graph_state()

    def _on_session_active_spectrum_changed(self, spec):
        if self.isVisible():
            self.recalculate_graph_preview()

    # --- OBLICZENIA I PODGLĄD NA ŻYWO ---
    def _get_base_preview_data(self) -> Tuple[np.ndarray, np.ndarray]:
        active = session.active_spectrum
        if active is not None and len(active.wavenumbers) > 2:
            return active.wavenumbers.copy(), active.intensities.copy()

        wn = np.linspace(4000, 400, 1200)
        y = 0.8 * np.exp(-((wn - 1650) / 35) ** 2) + 0.4 * np.exp(-((wn - 1540) / 40) ** 2) + 0.05 * np.random.normal(0, 0.02, len(wn))
        return wn, y

    def recalculate_graph_preview(self, target_scene: Optional[NodeScene] = None):
        scene = target_scene or self.scene
        if not scene:
            return

        all_nodes = [item for item in scene.items() if isinstance(item, NodeItem)]
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

        if scene == self.scene:
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
        if self.view:
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
        if not self.chk_live_preview.isChecked() or not self.scene:
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

    # --- WYKONANIE POTOKU (EXECUTE) ---
    def _execute_pipeline(self):
        if not self.current_tab:
            return
        self.execute_tab_pipeline(self.current_tab)

    def execute_tab_pipeline(self, tab: PipelineTabInfo):
        scene = tab.scene
        all_nodes = [item for item in scene.items() if isinstance(item, NodeItem)]
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

        scope = tab.scope
        in_place = tab.in_place

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

                # 1. Scope 0: Wszystkie widoczne w aktywnym oknie / gałęzi
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

                # 2. Scope 1: Tylko aktywne widmo
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

                # 3. Scope 2: Aktywna seria 2D / Tensor 3D
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

            QMessageBox.information(self, "Sukces", f"Pomyślnie wykonano potok z zakładki '{tab.title}'!")
            self.recalculate_graph_preview(target_scene=scene)
            self.sync_state_to_session()

        except Exception as e:
            logger.error(f"Błąd wykonania potoku DAG: {e}")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas obliczeń:\n{e}")