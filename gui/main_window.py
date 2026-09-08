"""
gui/main_window.py
Główne okno aplikacji SpectroHub w technologii PySide6 (Qt6).
"""

import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QStackedWidget,
    QToolBar, QButtonGroup, QPushButton, QLabel, QFileDialog,
    QMessageBox, QStatusBar
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QFont, QKeySequence
import logging

from core.i18n import tr, i18n
from core.session import session
from gui.canvas import SpectroCanvas
from gui.project_tree import ProjectTreeWidget
from io_hub.dispatcher import SpectrumImportDispatcher

# Import paneli roboczych
from gui.workspaces.ws_preprocessing import PreprocessingPanel
from gui.workspaces.ws_deconvolution import DeconvolutionPanel
from gui.workspaces.ws_chemometrics import ChemometricsPanel
from gui.workspaces.ws_quantum import QuantumPanel

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
QSpinBox, QDoubleSpinBox, QComboBox { background-color: #2b2b36; color: #ffffff; border: 1px solid #4a4a5a; padding: 4px; border-radius: 3px; }
QCheckBox { color: #dcdcdc; }
"""


class MainWindow(QMainWindow):
    """Główne okno platformy SpectroHub."""

    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)

        self._setup_ui()
        i18n.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def _setup_ui(self):
        # 1. Pasek menu
        self._create_menu_bar()

        # 2. Górny pasek perspektyw roboczych (Workspaces)
        self._create_workspace_toolbar()

        # 3. Lewy dok: Drzewo projektu
        self.dock_tree = QDockWidget(self)
        self.dock_tree.setObjectName("DockProjectTree")
        self.project_tree = ProjectTreeWidget(self)
        self.dock_tree.setWidget(self.project_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tree)

        # 4. Centralny widget: Wykres 2D FTIR (Canvas)
        self.canvas_2d = SpectroCanvas(self)
        self.setCentralWidget(self.canvas_2d)

        # 5. Prawy dok: Dynamiczny Inspektor Parametrów
        self.dock_inspector = QDockWidget(self)
        self.dock_inspector.setObjectName("DockInspector")
        self.inspector_stack = QStackedWidget(self)

        self.panel_prep = PreprocessingPanel(self)
        self.panel_deconv = DeconvolutionPanel(self)
        self.panel_chem = ChemometricsPanel(self)
        self.panel_dft = QuantumPanel(self)

        self.inspector_stack.addWidget(self.panel_prep)    # Indeks 0
        self.inspector_stack.addWidget(self.panel_deconv)  # Indeks 1
        self.inspector_stack.addWidget(self.panel_chem)    # Indeks 2
        self.inspector_stack.addWidget(self.panel_dft)     # Indeks 3

        self.dock_inspector.setWidget(self.inspector_stack)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_inspector)

        # 6. Pasek stanu
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _create_menu_bar(self):
        menu = self.menuBar()

        # 1. Menu Plik
        self.menu_file = menu.addMenu("")
        self.act_open = QAction(self)
        self.act_open.setShortcut(QKeySequence.Open)
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

        # 2. Menu Edycja (Globalne Undo / Redo)
        self.menu_edit = menu.addMenu("Edycja")

        self.act_undo = session.undo_stack.createUndoAction(self, "Cofnij")
        self.act_undo.setShortcut(QKeySequence.Undo)  # Ctrl+Z / Cmd+Z
        self.menu_edit.addAction(self.act_undo)

        self.act_redo = session.undo_stack.createRedoAction(self, "Ponów")
        self.act_redo.setShortcut(QKeySequence.Redo)  # Ctrl+Y / Cmd+Shift+Z
        self.menu_edit.addAction(self.act_redo)

        # 3. Menu Narzędzia
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
            ("workspaces.deconvolution", 1),
            ("workspaces.chemometrics", 2),
            ("workspaces.quantum", 3)
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

        # Prawy narożnik paska: Przełącznik języka (PL / EN)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        toolbar.addWidget(spacer)

        self.btn_lang = QPushButton("🇵🇱 PL / 🇬🇧 EN")
        self.btn_lang.setToolTip("Zmień język interfejsu (Language)")
        self.btn_lang.clicked.connect(self._toggle_language)
        toolbar.addWidget(self.btn_lang)

    def _on_workspace_selected(self, idx: int):
        self.inspector_stack.setCurrentIndex(idx)
        self.status_bar.showMessage(f"{tr('app.ready')}: Workspace {idx + 1}")

        # Jeśli wracamy do Preprocessingu (lub DFT), przywracamy domyślny podgląd projektu
        if idx in (0, 3):
            self.canvas_2d.prepare_axes(shared_x=True)
            self.canvas_2d.redraw_all()

    def _toggle_language(self):
        new_lang = "en" if i18n.current_language == "pl" else "pl"
        i18n.set_language(new_lang)

    def _on_open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("menu.open_spectra"), "",
            "Obsługiwane widma (*.0 *.[0-9]* *.opus *.spa *.spg *.spc *.dx *.jdx *.csv *.dpt *.txt *.log);;Wszystkie (*.*)"
        )
        if paths:
            # Uruchamiamy FileImportWorker w tle przez globalny QThreadPool,
            # żeby nie blokować interfejsu i załadować wszystko jednym wsadem!
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
            elif "DPT" in selected_filter:
                ASCIIExporter.export_spectrum_data(active, path, style="EU")
            else:
                ASCIIExporter.export_spectrum_data(active, path, style="EU")

            QMessageBox.information(self, tr("app.info"), tr("export.single_success", "Widmo zostało wyeksportowane."))
        except Exception as e:
            logger.error(f"Błąd eksportu widma: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd zapisu widma:\n{e}")

    def retranslate_ui(self):
        self.setWindowTitle(tr("app.title", "SpectroHub - Platforma Analizy FTIR & DFT"))
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