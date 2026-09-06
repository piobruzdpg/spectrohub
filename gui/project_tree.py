"""
gui/project_tree.py
Drzewo obiektów projektu z obsługą Drag & Drop i wielowątkowością I/O.
"""

import os
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu
from PySide6.QtCore import Qt, QThreadPool, QRunnable, Slot
import logging

from core.session import session
from io_hub.dispatcher import SpectrumImportDispatcher
from core.i18n import tr

logger = logging.getLogger("SpectroHub")


class FileImportWorker(QRunnable):
    """Zadanie w tle: odczyt plików spektroskopowych bez blokowania GUI."""
    def __init__(self, file_paths: list):
        super().__init__()
        self.file_paths = file_paths

    @Slot()
    def run(self):
        for path in self.file_paths:
            try:
                spectra = SpectrumImportDispatcher.load_file(path)
                session.add_spectra_batch(spectra)
            except Exception as e:
                logger.error(f"Nie udało się zaimportować pliku {path}: {e}")


class ProjectTreeWidget(QTreeWidget):
    """Drzewo plików i obiektów roboczych projektu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DropOnly)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        self.thread_pool = QThreadPool.globalInstance()

        # Grupy nadrzędne
        self.root_exp = QTreeWidgetItem(self, [tr("project_tree.group_exp", "Pomiary Eksperymentalne")])
        self.root_series = QTreeWidgetItem(self, [tr("project_tree.group_series", "Serie Ewolucyjne (2D)")])
        self.root_dft = QTreeWidgetItem(self, [tr("project_tree.group_dft", "Obliczenia Kwantowe (DFT)")])

        for root in (self.root_exp, self.root_series, self.root_dft):
            root.setExpanded(True)

        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_selection_changed)
        self.customContextMenuRequested.connect(self._show_context_menu)

        session.spectrum_added.connect(self._on_spectrum_added)
        session.spectrum_removed.connect(self._on_spectrum_removed)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            logger.info(f"Upuszczono {len(paths)} plików do aplikacji. Uruchamiam import...")
            worker = FileImportWorker(paths)
            self.thread_pool.start(worker)
            event.acceptProposedAction()

    def _on_spectrum_added(self, spectrum):
        # Sprawdzenie czy element już istnieje
        target_root = self.root_exp
        item = QTreeWidgetItem(target_root, [f"📈 {spectrum.name}"])
        item.setCheckState(0, Qt.Checked)
        item.setData(0, Qt.UserRole, spectrum.name)

    def _on_spectrum_removed(self, name: str):
        for i in range(self.root_exp.childCount()):
            child = self.root_exp.child(i)
            if child and child.data(0, Qt.UserRole) == name:
                self.root_exp.removeChild(child)
                break

    def _on_item_changed(self, item, column):
        name = item.data(0, Qt.UserRole)
        if name:
            is_visible = (item.checkState(0) == Qt.Checked)
            session.set_visibility(name, is_visible)

    def _on_selection_changed(self, current, previous):
        if current:
            name = current.data(0, Qt.UserRole)
            if name:
                session.set_active_spectrum(name)

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item or not item.data(0, Qt.UserRole):
            return

        name = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        act_remove = menu.addAction("🗑 Usuń z projektu")
        action = menu.exec(self.viewport().mapToGlobal(pos))

        if action == act_remove:
            session.remove_spectrum(name)