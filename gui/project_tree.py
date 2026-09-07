"""
gui/project_tree.py
Drzewo obiektów projektu z automatyczną kategoryzacją (Exp / Serie / DFT) i Drag & Drop.
"""

from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu
from PySide6.QtCore import Qt, QThreadPool, QRunnable, Slot
import logging

from core.session import session
from core.models import GaussianData
from io_hub.dispatcher import SpectrumImportDispatcher
from core.i18n import tr

logger = logging.getLogger("SpectroHub")


class FileImportWorker(QRunnable):
    """Zadanie w tle: asynchroniczny odczyt plików spektroskopowych."""
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

        session.series_added.connect(self._on_series_added)
        session.series_removed.connect(self._on_series_removed)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            logger.info(f"Upuszczono {len(paths)} plików. Uruchamiam import...")
            worker = FileImportWorker(paths)
            self.thread_pool.start(worker)
            event.acceptProposedAction()

    def _on_spectrum_added(self, spectrum):
        # Automatyczne rozpoznanie kategorii obiektu
        if isinstance(spectrum, GaussianData) or getattr(spectrum, 'file_type', '') in ['gaussian', 'orca', 'dft']:
            target_root = self.root_dft
            icon = "⚛️"
        elif getattr(spectrum, 'file_type', '') in ['series', 'tensor']:
            target_root = self.root_series
            icon = "📊"
        else:
            target_root = self.root_exp
            icon = "📈"

        item = QTreeWidgetItem(target_root, [f"{icon} {spectrum.name}"])
        item.setCheckState(0, Qt.Checked)
        item.setData(0, Qt.UserRole, spectrum.name)

    def _on_spectrum_removed(self, name: str):
        for root in (self.root_exp, self.root_series, self.root_dft):
            for i in range(root.childCount()):
                child = root.child(i)
                if child and child.data(0, Qt.UserRole) == name:
                    root.removeChild(child)
                    return

    def _on_series_added(self, series_obj):
        dim_tag = "3D Tensor" if series_obj.is_3d else "2D Seria"
        item = QTreeWidgetItem(self.root_series, [f"📊 {series_obj.name} [{dim_tag}]"])
        item.setCheckState(0, Qt.Checked)
        item.setData(0, Qt.UserRole, series_obj.name)
        item.setData(0, Qt.UserRole + 1, "series")

    def _on_series_removed(self, name: str):
        for i in range(self.root_series.childCount()):
            child = self.root_series.child(i)
            if child and child.data(0, Qt.UserRole) == name:
                self.root_series.removeChild(child)
                return

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
        spec = session.spectra.get(name)
        if not spec:
            return

        from io_hub.exporters.ascii_exporter import ASCIIExporter
        from io_hub.exporters.spc_exporter import SPCExporter
        from io_hub.exporters.jcamp_exporter import JCAMPExporter
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        menu = QMenu(self)
        act_export = menu.addAction(f"💾 Eksportuj '{name}' jako...")
        act_remove = menu.addAction("🗑 Usuń z projektu")

        action = menu.exec(self.viewport().mapToGlobal(pos))

        if action == act_remove:
            session.remove_spectrum(name)
        elif action == act_export:
            filters = (
                "CSV - Międzynarodowy (*.csv);;"
                "CSV - Polski format (*.csv);;"
                "Thermo Grams SPC (*.spc);;"
                "JCAMP-DX (*.dx);;"
                "Plik tekstowy DPT (*.dpt)"
            )
            path, sel_filter = QFileDialog.getSaveFileName(
                self, f"Eksportuj {name}", f"{name}.csv", filters
            )
            if path:
                try:
                    if "Polski" in sel_filter:
                        ASCIIExporter.export_spectrum_data(spec, path, style="PL")
                    elif "SPC" in sel_filter:
                        SPCExporter.export_spectrum(spec, path)
                    elif "JCAMP" in sel_filter:
                        JCAMPExporter.export_spectrum(spec, path)
                    else:
                        ASCIIExporter.export_spectrum_data(spec, path, style="EU")
                    QMessageBox.information(self, tr("app.info"), "Plik został pomyślnie zapisany.")
                except Exception as e:
                    QMessageBox.critical(self, tr("app.error"), f"Błąd zapisu:\n{e}")