"""
gui/main_window.py
Główne okno aplikacji SpectroHub w technologii PySide6 (Qt6).
Synchronizuje SpectroCanvas ze zmianą aktywnej gałęzi magazynu.
"""

import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QStackedWidget,
    QToolBar, QButtonGroup, QPushButton, QLabel, QFileDialog,
    QMessageBox, QStatusBar, QSplitter, QVBoxLayout, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
import logging

from core.i18n import tr, i18n
from core.session import session
from gui.canvas import SpectroCanvas
from gui.project_tree import ProjectTreeWidget
from gui.project_metadata_widget import ProjectMetadataWidget
from io_hub.project_io import ProjectBundleIO
from io_hub.dispatcher import SpectrumImportDispatcher

# Panele robocze
from gui.workspaces.ws_preprocessing import PreprocessingPanel
from gui.workspaces.ws_deconvolution import DeconvolutionPanel
from gui.workspaces.ws_chemometrics import ChemometricsPanel
from gui.workspaces.ws_quantum import QuantumPanel
from gui.workspaces.ws_spectral_math import SpectralMathPanel

from io_hub.exporters.excel_exporter import ExcelProjectExporter
from io_hub.exporters.ascii_exporter import ASCIIExporter
from io_hub.exporters.spc_exporter import SPCExporter
from io_hub.exporters.jcamp_exporter import JCAMPExporter

logger = logging.getLogger("SpectroHub")

DARK_STYLE = """
QMainWindow { background-color: #1e1e24; }
QDockWidget { color: #e0e0e0; font-weight: bold; }
QDockWidget::title { background: #2b2b36; padding: 6px; border-radius: 4px; }
QTreeWidget { background-color: #252530; color: #dcdcdc; border: 1px solid #383848; border-radius: 4px; }
QTreeWidget::item:selected { background-color: #3d5a80; }
QToolBar { background: #252530; border-bottom: 1px solid #383848; spacing: 6px; padding: 4px; }
QPushButton { background-color: #383848; color: #ffffff; border: 1px solid #4a4a5a; 
              padding: 5px 12px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #4a4a5a; }
QPushButton:checked { background-color: #3d5a80; border: 1px solid #98c1d9; }
QTableWidget { background-color: #252530; color: #ffffff; gridline-color: #383848; border: 1px solid #383848; }
QHeaderView::section { background-color: #2b2b36; color: #ffffff; padding: 4px; border: 1px solid #383848; }
QLabel { color: #dcdcdc; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QTextEdit { 
    background-color: #2b2b36; color: #ffffff; border: 1px solid #4a4a5a; padding: 4px; border-radius: 3px; 
}
QCheckBox { color: #dcdcdc; }
QSplitter::handle { background-color: #383848; height: 3px; }
"""


class MainWindow(QMainWindow):
    """Główne okno platformy SpectroHub."""

    def __init__(self):
        super().__init__()
        self.resize(1400, 920)
        self.setStyleSheet(DARK_STYLE)

        self.cached_dag_state = {}
        self._node_dialog = None

        self._setup_ui()
        i18n.language_changed.connect(self.retranslate_ui)
        session.metadata_changed.connect(self._update_window_title)

        # Reaktywność canvasu na zmianę gałęzi magazynu i kontekstu okna
        session.storage_branch_changed.connect(lambda _: self.canvas_2d.redraw_all())
        session.context_changed.connect(lambda _: self.canvas_2d.redraw_all())

        self.retranslate_ui()

    def _setup_ui(self):
        self._create_menu_bar()
        self._create_workspace_toolbar()

        # Lewy dok: Drzewo + Metadane
        self.dock_tree = QDockWidget(self)
        self.dock_tree.setObjectName("DockProjectTree")

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(0)

        self.left_splitter = QSplitter(Qt.Vertical)
        self.project_tree = ProjectTreeWidget(self)
        self.project_meta_widget = ProjectMetadataWidget(self)

        self.left_splitter.addWidget(self.project_tree)
        self.left_splitter.addWidget(self.project_meta_widget)
        self.left_splitter.setStretchFactor(0, 4)
        self.left_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(self.left_splitter)
        self.dock_tree.setWidget(left_container)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tree)

        # Centralny Canvas
        self.canvas_2d = SpectroCanvas(self)
        self.setCentralWidget(self.canvas_2d)

        # Prawy dok: Workspace Inspector
        self.dock_inspector = QDockWidget(self)
        self.dock_inspector.setObjectName("DockInspector")
        self.inspector_stack = QStackedWidget(self)

        self.panel_prep = PreprocessingPanel(self)
        self.panel_math = SpectralMathPanel(self)
        self.panel_deconv = DeconvolutionPanel(self)
        self.panel_chem = ChemometricsPanel(self)
        self.panel_dft = QuantumPanel(self)

        self.inspector_stack.addWidget(self.panel_prep)    # 0
        self.inspector_stack.addWidget(self.panel_math)    # 1
        self.inspector_stack.addWidget(self.panel_deconv)  # 2
        self.inspector_stack.addWidget(self.panel_chem)    # 3
        self.inspector_stack.addWidget(self.panel_dft)     # 4

        self.dock_inspector.setWidget(self.inspector_stack)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_inspector)

        # Zoptymalizowany podział przestrzeni na start
        self.resizeDocks([self.dock_tree, self.dock_inspector], [260, 360], Qt.Horizontal)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _create_menu_bar(self):
        menu = self.menuBar()

        self.menu_file = menu.addMenu("")

        self.act_new_proj = QAction("Nowy projekt", self)
        self.act_new_proj.setShortcut(QKeySequence.New)
        self.act_new_proj.triggered.connect(self._on_new_project)
        self.menu_file.addAction(self.act_new_proj)

        self.act_open_proj = QAction("Otwórz projekt (.shub)...", self)
        self.act_open_proj.setShortcut(QKeySequence.Open)
        self.act_open_proj.triggered.connect(self._on_open_project)
        self.menu_file.addAction(self.act_open_proj)

        self.act_save_proj = QAction("Zapisz projekt", self)
        self.act_save_proj.setShortcut(QKeySequence.Save)
        self.act_save_proj.triggered.connect(self._on_save_project)
        self.menu_file.addAction(self.act_save_proj)

        self.act_save_proj_as = QAction("Zapisz projekt jako...", self)
        self.act_save_proj_as.setShortcut(QKeySequence.SaveAs)
        self.act_save_proj_as.triggered.connect(self._on_save_project_as)
        self.menu_file.addAction(self.act_save_proj_as)

        self.menu_file.addSeparator()

        self.act_open = QAction(self)
        self.act_open.triggered.connect(self._on_open_files)
        self.menu_file.addAction(self.act_open)

        self.menu_file.addSeparator()

        self.act_export_excel = QAction(self)
        self.act_export_excel.triggered.connect(self._on_export_excel)
        self.menu_file.addAction(self.act_export_excel)

        self.act_export_active = QAction(self)
        self.act_export_active.triggered.connect(self._on_export_active_spectrum)
        self.menu_file.addAction(self.act_export_active)

        self.menu_file.addSeparator()

        self.act_exit = QAction(self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)
        self.menu_file.addAction(self.act_exit)

        # Edycja
        self.menu_edit = menu.addMenu("Edycja")
        self.act_undo = session.undo_stack.createUndoAction(self, "Cofnij")
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.menu_edit.addAction(self.act_undo)

        self.act_redo = session.undo_stack.createRedoAction(self, "Ponów")
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.menu_edit.addAction(self.act_redo)

        # Narzędzia
        self.menu_tools = menu.addMenu("")
        self.act_clear = QAction(self)
        self.act_clear.triggered.connect(session.clear_all)
        self.menu_tools.addAction(self.act_clear)

        self.act_tensor_builder = QAction(tr("menu.tensor_builder", "Kreator Serii & Tensora 3D..."), self)
        self.act_tensor_builder.triggered.connect(self._open_tensor_builder)
        self.menu_tools.addAction(self.act_tensor_builder)

    def _create_workspace_toolbar(self):
        toolbar = QToolBar("Workspaces", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.lbl_ws_header = QLabel()
        self.lbl_ws_header.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar.addWidget(self.lbl_ws_header)

        self.ws_group = QButtonGroup(self)
        self.ws_group.setExclusive(True)

        self.ws_buttons = []
        workspaces_keys = [
            ("workspaces.preprocessing", 0),
            ("workspaces.spectral_math", 1),
            ("workspaces.deconvolution", 2),
            ("workspaces.chemometrics", 3),
            ("workspaces.quantum", 4)
        ]

        for key, idx in workspaces_keys:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setProperty("i18n_key", key)
            if idx == 0:
                btn.setChecked(True)
            self.ws_group.addButton(btn, idx)
            toolbar.addWidget(btn)
            self.ws_buttons.append(btn)

        self.ws_group.idClicked.connect(self._on_workspace_selected)

        # Rozciągający się spacer dociągający prawe elementy do prawej krawędzi
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Przycisk Okna Grafów (Potoku Węzłowego)
        self.btn_node_graph = QPushButton("⑂ Okno grafów")
        self.btn_node_graph.setToolTip("Otwórz wizualny potok węzłowy (DAG)")
        self.btn_node_graph.setStyleSheet("""
            QPushButton {
                background-color: #3a0ca3; 
                color: #ffffff; 
                font-weight: bold; 
                padding: 5px 12px; 
                border: 1px solid #7209b7;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4361ee; }
        """)
        self.btn_node_graph.clicked.connect(self.open_node_pipeline_dialog)
        toolbar.addWidget(self.btn_node_graph)

        # Przycisk zmiany języka (zawsze skrajnie po prawej)
        self.btn_lang = QPushButton("🇵🇱 PL / 🇬🇧 EN")
        self.btn_lang.setToolTip("Zmień język interfejsu (Language)")
        self.btn_lang.clicked.connect(self._toggle_language)
        toolbar.addWidget(self.btn_lang)

    def open_node_pipeline_dialog(self):
        """Niemodalne otwarcie okna potoku węzłowego."""
        if self._node_dialog is None:
            from gui.node_editor import NodePipelineDialog
            self._node_dialog = NodePipelineDialog(parent=self)
            if self.cached_dag_state:
                self._node_dialog.import_graph_state(self.cached_dag_state)

        self._node_dialog.show()
        self._node_dialog.raise_()
        self._node_dialog.activateWindow()

    def _on_workspace_selected(self, idx: int):
        self.inspector_stack.setCurrentIndex(idx)
        self.status_bar.showMessage(f"{tr('app.ready')}: Workspace {idx + 1}")

        if idx in (0, 3):
            self.canvas_2d.prepare_axes(shared_x=True)
            self.canvas_2d.redraw_all()

    def _toggle_language(self):
        new_lang = "en" if i18n.current_language == "pl" else "pl"
        i18n.set_language(new_lang)

    def _update_window_title(self, meta=None):
        meta = meta or session.metadata
        proj_name = meta.title or "Bez tytułu"
        fpath = f" — [{os.path.basename(session.project_file_path)}]" if session.project_file_path else ""
        self.setWindowTitle(f"SpectroHub - {proj_name}{fpath}")

    def _on_new_project(self):
        reply = QMessageBox.question(
            self, "Nowy Projekt",
            "Czy na pewno chcesz utworzyć nowy projekt?\nWszystkie niezapisane zmiany zostaną utracone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            session.clear_all()
            self.cached_dag_state = {}
            self.panel_prep.set_pipeline_state([])
            self.canvas_2d.redraw_all()
            self.status_bar.showMessage("Utworzono nowy, czysty projekt.")

    def _on_save_project(self):
        if session.project_file_path:
            self._save_to_path(session.project_file_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        def_name = f"{session.metadata.title.replace(' ', '_')}.shub" if session.metadata.title else "Projekt_SpectroHub.shub"
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz projekt SpectroHub", def_name, "SpectroHub Project Bundle (*.shub *.SHUB);;Wszystkie pliki (*.*)"
        )
        if path:
            if not path.lower().endswith(".shub"):
                path += ".shub"
            self._save_to_path(path)

    def _save_to_path(self, path: str):
        try:
            linear_pipe = self.panel_prep.get_pipeline_state()
            ProjectBundleIO.save_project(
                file_path=path,
                session=session,
                dag_state=self.cached_dag_state,
                linear_pipeline=linear_pipe
            )
            self._update_window_title()
            self.status_bar.showMessage(f"Projekt został pomyślnie zapisany: {path}", 5000)
            QMessageBox.information(self, "Zapisano", f"Projekt pomyślnie zapisany do:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd zapisu", f"Nie udało się zapisać projektu:\n{e}")

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Otwórz projekt SpectroHub", "", "SpectroHub Project Bundle (*.shub *.SHUB);;Wszystkie pliki (*.*)"
        )
        if not path:
            return

        try:
            dag_state, linear_pipe = ProjectBundleIO.load_project(path, session)
            self.cached_dag_state = dag_state
            self.panel_prep.set_pipeline_state(linear_pipe)

            self.canvas_2d.prepare_axes(shared_x=True)
            self.canvas_2d.redraw_all()
            self._update_window_title()

            self.status_bar.showMessage(f"Załadowano projekt: {path}", 5000)
            QMessageBox.information(self, "Wczytano projekt", f"Pomyślnie załadowano projekt:\n{session.metadata.title}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd odczytu", f"Nie udało się wczytać pliku projektu:\n{e}")

    def _on_open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("menu.open_spectra", "Dodaj pliki widm..."), "",
            SpectrumImportDispatcher.get_open_file_filter()
        )
        if paths:
            from PySide6.QtCore import QThreadPool
            from gui.project_tree import FileImportWorker
            worker = FileImportWorker(paths)
            QThreadPool.globalInstance().start(worker)

    def _open_tensor_builder(self):
        from gui.tensor_builder_dialog import TensorBuilderDialog
        dialog = TensorBuilderDialog(self)
        dialog.exec()

    def _on_export_excel(self):
        spectra = list(session.spectra.values())
        if not spectra:
            QMessageBox.information(self, tr("app.info"), tr("export.no_spectra", "Brak widm do eksportu."))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, tr("export.save_excel", "Zapisz projekt do Excela"),
            "SpectroHub_Projekt.xlsx", "Skoroszyt Excel (*.xlsx)"
        )
        if not path:
            return

        try:
            peaks = getattr(self.panel_deconv.panel_deconv, 'peaks', [])
            ExcelProjectExporter.export(path, spectra, deconv_peaks=peaks)
            QMessageBox.information(self, tr("app.info"),
                                    tr("export.excel_success", "Pomyślnie wyeksportowano raport Excel!"))
        except Exception as e:
            logger.error(f"Błąd eksportu Excel: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd zapisu pliku Excel:\n{e}")

    def _on_export_active_spectrum(self):
        active = session.active_spectrum
        if not active:
            QMessageBox.warning(self, tr("app.warning"), tr("export.no_active", "Brak aktywnego widma."))
            return

        filters = (
            "CSV - Międzynarodowy (*.csv);;"
            "CSV - Polski format (*.csv);;"
            "Thermo Grams SPC (*.spc);;"
            "JCAMP-DX (*.dx);;"
            "Plik tekstowy DPT (*.dpt)"
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self, tr("export.save_spectrum", "Eksportuj aktywne widmo"),
            f"{active.name}.csv", filters
        )
        if not path:
            return

        try:
            if "Polski" in selected_filter:
                ASCIIExporter.export_spectrum_data(active, path, style="PL")
            elif "SPC" in selected_filter:
                SPCExporter.export_spectrum(active, path)
            elif "JCAMP" in selected_filter:
                JCAMPExporter.export_spectrum(active, path)
            else:
                ASCIIExporter.export_spectrum_data(active, path, style="EU")

            QMessageBox.information(self, tr("app.info"), tr("export.single_success", "Widmo zostało wyeksportowane."))
        except Exception as e:
            logger.error(f"Błąd eksportu widma: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd zapisu widma:\n{e}")

    def retranslate_ui(self):
        self._update_window_title()
        self.menu_file.setTitle(tr("menu.file", "Plik"))
        self.act_open.setText(tr("menu.open_spectra", "Dodaj pliki widm..."))
        self.act_export_excel.setText(tr("menu.export_excel", "Eksportuj projekt do Excela (.xlsx)..."))
        self.act_export_active.setText(tr("menu.export_active", "Eksportuj aktywne widmo jako..."))
        self.act_exit.setText(tr("menu.exit", "Wyjdź"))
        self.menu_tools.setTitle(tr("menu.tools", "Narzędzia"))
        self.act_clear.setText(tr("menu.clear", "Wyczyść sesję"))

        self.dock_tree.setWindowTitle(tr("project_tree.header", "Eksplorator Projektu"))
        self.dock_inspector.setWindowTitle(tr("inspector.header", "Inspektor Parametrów"))
        self.lbl_ws_header.setText(f"  {tr('workspaces.header', 'PERSPEKTYWY')}:  ")

        for btn in self.ws_buttons:
            key = btn.property("i18n_key")
            if key:
                btn.setText(tr(key))

        self.status_bar.showMessage(tr("app.ready", "Gotowy"))