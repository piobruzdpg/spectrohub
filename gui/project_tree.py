"""
gui/project_tree.py
Drzewo obiektów projektu z obsługą folderów okien roboczych, Drag & Drop oraz zaawansowanym eksportem.
"""

import os
import logging
from typing import List, Tuple, Optional
from PySide6.QtCore import Qt, QThreadPool, QRunnable, Slot
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QAbstractItemView,
    QInputDialog, QFileDialog, QMessageBox
)

from core.session import session
from core.models import GaussianData, SpectrumData, WorkspaceWindow
from io_hub.dispatcher import SpectrumImportDispatcher
from io_hub.export_manager import ExportManager
from core.i18n import tr

logger = logging.getLogger("SpectroHub")


class FileImportWorker(QRunnable):
    """Zadanie w tle: importuje pliki do magazynu danych w jednym strzale."""

    def __init__(self, file_paths: list):
        super().__init__()
        self.file_paths = file_paths

    @Slot()
    def run(self):
        all_loaded = []
        for path in self.file_paths:
            try:
                spectra = SpectrumImportDispatcher.load_file(path)
                if isinstance(spectra, list):
                    all_loaded.extend(spectra)
                elif spectra:
                    all_loaded.append(spectra)
            except Exception as e:
                logger.error(f"Nie udało się zaimportować pliku {path}: {e}")

        if all_loaded:
            session.add_spectra_batch(all_loaded)


class ProjectTreeWidget(QTreeWidget):
    """Eksplorator z podziałem na Gałęzie Magazynu oraz Foldery i Okna Robocze."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        self.thread_pool = QThreadPool.globalInstance()
        self._is_internal_update = False

        self._build_root_structure()
        self._connect_signals()

    def _build_root_structure(self):
        # 1. SEKCYJNY KORZEŃ: MAGAZYN DANYCH
        self.root_storage = QTreeWidgetItem(self, ["📦 MAGAZYN DANYCH (Referencja)"])
        self.root_storage.setExpanded(True)
        self.root_storage.setData(0, Qt.UserRole + 1, "section_header")

        self.root_exp = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_exp", "Pomiary Eksperymentalne")])
        self.root_atm = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_atm", "☁️ Wzorce Atmosfery (ATM / CO₂)")])
        self.root_series = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_series", "Serie Ewolucyjne (2D)")])
        self.root_dft = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_dft", "Obliczenia Kwantowe (DFT)")])

        self.root_exp.setData(0, Qt.UserRole + 1, "category_exp")
        self.root_atm.setData(0, Qt.UserRole + 1, "category_atm")
        self.root_series.setData(0, Qt.UserRole + 1, "category_series")
        self.root_dft.setData(0, Qt.UserRole + 1, "category_dft")

        for sub in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            sub.setExpanded(True)
            sub.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            sub.setCheckState(0, Qt.CheckState.Checked)

        # 2. SEKCYJNY KORZEŃ: PRZESTRZENIE ROBOCZE
        self.root_workspaces = QTreeWidgetItem(self, ["🖥️ OKNA WYKRESÓW / ROBOCZE"])
        self.root_workspaces.setExpanded(True)
        self.root_workspaces.setData(0, Qt.UserRole + 1, "section_header")

        # Inicjalizacja folderów i okien z sesji
        for f_id, folder in session.folders.items():
            self._create_folder_item(f_id, folder.name, folder.parent_folder_id)

        for win_id, win in session.windows.items():
            self._create_window_item(win_id, win.name, win.folder_id)

    def _connect_signals(self):
        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_selection_changed)
        self.itemClicked.connect(self._on_selection_changed)
        self.customContextMenuRequested.connect(self._show_context_menu)

        session.session_cleared.connect(self._on_session_cleared)

        session.spectrum_added.connect(self._on_spectrum_added)
        session.spectra_batch_added.connect(self._on_spectra_batch_added)
        session.spectrum_removed.connect(self._on_spectrum_removed)

        session.atm_spectrum_added.connect(self._on_atm_spectrum_added)
        session.atm_spectra_batch_added.connect(self._on_atm_spectra_batch_added)
        session.atm_spectrum_removed.connect(self._on_atm_spectrum_removed)

        session.series_added.connect(self._on_series_added)
        session.series_removed.connect(self._on_series_removed)

        # Sygnały okien i folderów
        session.folder_added.connect(self._on_folder_added)
        session.folder_removed.connect(self._on_folder_removed)
        session.folder_renamed.connect(self._on_folder_renamed)
        session.window_folder_changed.connect(self._on_window_folder_changed)

        session.window_added.connect(self._on_window_added)
        session.window_removed.connect(self._on_window_removed)
        session.window_spectrum_added.connect(self._on_window_spectrum_added)
        session.window_spectrum_removed.connect(self._on_window_spectrum_removed)

    def _on_session_cleared(self):
        self._is_internal_update = True
        for cat in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            cat.takeChildren()
        self.root_workspaces.takeChildren()
        self._is_internal_update = False

    # --- ZARZĄDZANIE FOLDERAMI I OKNAMI W DRZEWIE ---
    def _find_folder_item(self, folder_id: str, parent: Optional[QTreeWidgetItem] = None) -> Optional[QTreeWidgetItem]:
        if parent is None:
            parent = self.root_workspaces

        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.UserRole + 1) == "window_folder" and child.data(0, Qt.UserRole) == folder_id:
                return child
            if child.data(0, Qt.UserRole + 1) == "window_folder":
                sub = self._find_folder_item(folder_id, parent=child)
                if sub:
                    return sub
        return None

    def _find_window_item(self, window_id: str, parent: Optional[QTreeWidgetItem] = None) -> Optional[QTreeWidgetItem]:
        if parent is None:
            parent = self.root_workspaces

        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.UserRole + 1) == "workspace_window" and child.data(0, Qt.UserRole) == window_id:
                return child
            if child.data(0, Qt.UserRole + 1) == "window_folder":
                sub = self._find_window_item(window_id, parent=child)
                if sub:
                    return sub
        return None

    def _create_folder_item(self, folder_id: str, name: str, parent_folder_id: Optional[str] = None) -> QTreeWidgetItem:
        parent_item = self.root_workspaces
        if parent_folder_id:
            found_p = self._find_folder_item(parent_folder_id)
            if found_p:
                parent_item = found_p

        folder_item = QTreeWidgetItem(parent_item, [f"📁 {name}"])
        folder_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDropEnabled
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        folder_item.setCheckState(0, Qt.CheckState.Checked)
        folder_item.setData(0, Qt.UserRole, folder_id)
        folder_item.setData(0, Qt.UserRole + 1, "window_folder")
        folder_item.setExpanded(True)
        return folder_item

    def _create_window_item(self, window_id: str, name: str, folder_id: Optional[str] = None) -> QTreeWidgetItem:
        parent_item = self.root_workspaces
        win = session.windows.get(window_id)
        f_id = folder_id if folder_id is not None else (win.folder_id if win else None)

        if f_id:
            found_folder = self._find_folder_item(f_id)
            if found_folder:
                parent_item = found_folder

        win_item = QTreeWidgetItem(parent_item, [f"🗂️ {name}"])
        win_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDropEnabled
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        win_item.setCheckState(0, Qt.CheckState.Checked)
        win_item.setData(0, Qt.UserRole, window_id)
        win_item.setData(0, Qt.UserRole + 1, "workspace_window")
        win_item.setExpanded(True)
        return win_item

    def _on_folder_added(self, folder_id: str, name: str, parent_folder_id: Optional[str]):
        if not self._find_folder_item(folder_id):
            self._create_folder_item(folder_id, name, parent_folder_id)

    def _on_folder_removed(self, folder_id: str):
        folder_node = self._find_folder_item(folder_id)
        if folder_node:
            p = folder_node.parent() or self.root_workspaces
            p.removeChild(folder_node)
            self._refresh_parent_indicator(p)

    def _on_folder_renamed(self, folder_id: str, new_name: str):
        folder_node = self._find_folder_item(folder_id)
        if folder_node:
            folder_node.setText(0, f"📁 {new_name}")

    def _on_window_folder_changed(self, window_id: str, folder_id: Optional[str]):
        win_node = self._find_window_item(window_id)
        if not win_node:
            return

        old_parent = win_node.parent() or self.root_workspaces
        old_parent.removeChild(win_node)

        new_parent = self.root_workspaces
        if folder_id:
            target_f = self._find_folder_item(folder_id)
            if target_f:
                new_parent = target_f

        new_parent.addChild(win_node)
        win_node.setExpanded(True)
        self._refresh_parent_indicator(new_parent)
        self._refresh_parent_indicator(old_parent)

    def _on_window_added(self, window_id: str, name: str):
        if not self._find_window_item(window_id):
            win = session.windows.get(window_id)
            f_id = win.folder_id if win else None
            self._create_window_item(window_id, name, f_id)

    def _on_window_removed(self, window_id: str):
        win_node = self._find_window_item(window_id)
        if win_node:
            p = win_node.parent() or self.root_workspaces
            p.removeChild(win_node)
            self._refresh_parent_indicator(p)

    def _on_window_spectrum_added(self, window_id: str, spec):
        win_node = self._find_window_item(window_id)
        if win_node:
            item = QTreeWidgetItem(win_node, [f"📈 {spec.name}"])
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.UserRole, spec.name)
            item.setData(0, Qt.UserRole + 1, "window_spectrum")
            item.setData(0, Qt.UserRole + 2, window_id)
            win_node.setExpanded(True)
            self._refresh_parent_indicator(win_node)

    def _on_window_spectrum_removed(self, window_id: str, name: str):
        win_node = self._find_window_item(window_id)
        if win_node:
            for j in range(win_node.childCount()):
                s_item = win_node.child(j)
                if s_item and s_item.data(0, Qt.UserRole) == name:
                    win_node.removeChild(s_item)
                    self._refresh_parent_indicator(win_node)
                    return

    # --- DRAG & DROP ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                worker = FileImportWorker(paths)
                self.thread_pool.start(worker)
                event.acceptProposedAction()
                return

        target_item = self.itemAt(event.position().toPoint())
        if not target_item:
            event.ignore()
            return

        target_type = target_item.data(0, Qt.UserRole + 1)
        target_id = target_item.data(0, Qt.UserRole)
        selected = self.selectedItems()
        if not selected:
            event.ignore()
            return

        first_sel = selected[0]
        first_type = first_sel.data(0, Qt.UserRole + 1)
        first_id = first_sel.data(0, Qt.UserRole)

        # 1. Przeciąganie OKNA ROBOCZEGO (workspace_window)
        if first_type == "workspace_window":
            if target_type == "window_folder":
                session.move_window_to_folder(first_id, target_id)
                event.acceptProposedAction()
                return
            elif target_type == "section_header" and target_item == self.root_workspaces:
                session.move_window_to_folder(first_id, None)
                event.acceptProposedAction()
                return
            elif target_type == "workspace_window":
                other_win = session.windows.get(target_id)
                session.move_window_to_folder(first_id, other_win.folder_id if other_win else None)
                event.acceptProposedAction()
                return

        # 2. Przeciąganie FOLDERU (window_folder)
        if first_type == "window_folder":
            if target_type == "window_folder" and target_id != first_id:
                f = session.get_folder(first_id)
                if f:
                    f.parent_folder_id = target_id
                    self._on_window_folder_changed_refresh()
                event.acceptProposedAction()
                return
            elif target_type == "section_header" and target_item == self.root_workspaces:
                f = session.get_folder(first_id)
                if f:
                    f.parent_folder_id = None
                    self._on_window_folder_changed_refresh()
                event.acceptProposedAction()
                return

        # 3. Przeciąganie WIDM do okna lub folderu
        target_win_id = None
        if target_type == "workspace_window":
            target_win_id = target_id
        elif target_type == "window_spectrum":
            target_win_id = target_item.data(0, Qt.UserRole + 2)
        elif target_type == "window_folder":
            folder_windows = [w for w in session.windows.values() if w.folder_id == target_id]
            if folder_windows:
                target_win_id = folder_windows[0].window_id
            else:
                target_win_id = session.add_window(name="Nowe Okno", folder_id=target_id)

        if target_win_id:
            def collect_specs(tree_item) -> list:
                specs = []
                itype = tree_item.data(0, Qt.UserRole + 1)
                iname = tree_item.data(0, Qt.UserRole)
                if itype in ["spectrum", "dft", "atm", "window_spectrum"] and iname:
                    specs.append(iname)
                for i in range(tree_item.childCount()):
                    specs.extend(collect_specs(tree_item.child(i)))
                return specs

            names_to_add = []
            for itm in selected:
                for s_name in collect_specs(itm):
                    if s_name not in names_to_add:
                        names_to_add.append(s_name)

            for s_name in names_to_add:
                src_spec = session.get_spectrum(s_name)
                if src_spec:
                    session.add_spectrum_to_window(target_win_id, src_spec)

            event.acceptProposedAction()
            return

        event.ignore()

    def _on_window_folder_changed_refresh(self):
        # Przebudowa struktury okien przy zagnieżdżaniu folderów
        self.root_workspaces.takeChildren()
        for f_id, folder in session.folders.items():
            self._create_folder_item(f_id, folder.name, folder.parent_folder_id)
        for win_id, win in session.windows.items():
            win_node = self._create_window_item(win_id, win.name, win.folder_id)
            for s_name, spec in win.spectra.items():
                item = QTreeWidgetItem(win_node, [f"📈 {spec.name}"])
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if win.visibility.get(s_name, True) else Qt.CheckState.Unchecked)
                item.setData(0, Qt.UserRole, spec.name)
                item.setData(0, Qt.UserRole + 1, "window_spectrum")
                item.setData(0, Qt.UserRole + 2, win_id)

    # --- SYNCHRONIZACJA ELEMENTÓW MAGAZYNU ---
    def _create_spectrum_item(self, spectrum, target_root=None, item_type=None) -> tuple[QTreeWidgetItem, QTreeWidgetItem]:
        if target_root is None:
            if isinstance(spectrum, GaussianData) or getattr(spectrum, 'file_type', '') in ['gaussian', 'orca', 'dft']:
                target_root = self.root_dft
                icon = "⚛️"
                item_type = "dft"
            elif item_type == "atm":
                target_root = self.root_atm
                icon = "☁️"
                item_type = "atm"
            else:
                target_root = self.root_exp
                icon = "📈"
                item_type = "spectrum"
        else:
            icon = "☁️" if item_type == "atm" else ("⚛️" if item_type == "dft" else "📈")

        item = QTreeWidgetItem(target_root, [f"{icon} {spectrum.name}"])
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        item.setCheckState(0, Qt.CheckState.Checked)
        item.setData(0, Qt.UserRole, spectrum.name)
        item.setData(0, Qt.UserRole + 1, item_type)
        return item, target_root

    def _on_spectrum_added(self, spectrum):
        self._is_internal_update = True
        _, target_root = self._create_spectrum_item(spectrum)
        self._refresh_parent_indicator(target_root)
        self._is_internal_update = False

    def _on_spectra_batch_added(self, spectra_list: list):
        self._is_internal_update = True
        self.setUpdatesEnabled(False)
        roots_affected = set()
        for spec in spectra_list:
            _, target_root = self._create_spectrum_item(spec)
            roots_affected.add(target_root)
        for root in roots_affected:
            self._refresh_parent_indicator(root)
        self.setUpdatesEnabled(True)
        self._is_internal_update = False

    def _on_atm_spectrum_added(self, spectrum):
        self._is_internal_update = True
        _, target_root = self._create_spectrum_item(spectrum, target_root=self.root_atm, item_type="atm")
        self._refresh_parent_indicator(target_root)
        self._is_internal_update = False

    def _on_atm_spectra_batch_added(self, spectra_list: list):
        self._is_internal_update = True
        self.setUpdatesEnabled(False)
        for spec in spectra_list:
            self._create_spectrum_item(spec, target_root=self.root_atm, item_type="atm")
        self._refresh_parent_indicator(self.root_atm)
        self.setUpdatesEnabled(True)
        self._is_internal_update = False

    def _on_spectrum_removed(self, name: str):
        self._is_internal_update = True
        for root in (self.root_exp, self.root_series, self.root_dft):
            for i in range(root.childCount()):
                child = root.child(i)
                if child and child.data(0, Qt.UserRole) == name:
                    root.removeChild(child)
                    self._refresh_parent_indicator(root)
                    self._is_internal_update = False
                    return
        self._is_internal_update = False

    def _on_atm_spectrum_removed(self, name: str):
        self._is_internal_update = True
        for i in range(self.root_atm.childCount()):
            child = self.root_atm.child(i)
            if child and child.data(0, Qt.UserRole) == name:
                self.root_atm.removeChild(child)
                self._refresh_parent_indicator(self.root_atm)
                break
        self._is_internal_update = False

    def _on_series_added(self, series_obj):
        dim_tag = "3D Tensor" if getattr(series_obj, 'is_3d', False) else "2D Seria"
        n_spectra = series_obj.n_spectra if hasattr(series_obj, 'n_spectra') else series_obj.data.shape[1]
        label = f"📊 {series_obj.name} [{dim_tag}, {n_spectra} widm]"

        series_item = QTreeWidgetItem(self.root_series, [label])
        series_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        series_item.setCheckState(0, Qt.CheckState.Checked)
        series_item.setData(0, Qt.UserRole, series_obj.name)
        series_item.setData(0, Qt.UserRole + 1, "series")
        self._refresh_parent_indicator(self.root_series)

    def _on_series_removed(self, name: str):
        self._is_internal_update = True
        for i in range(self.root_series.childCount()):
            child = self.root_series.child(i)
            if child and child.data(0, Qt.UserRole) == name:
                self.root_series.removeChild(child)
                self._refresh_parent_indicator(self.root_series)
                break
        self._is_internal_update = False

    # --- KONTROLA CHECKBOXÓW ---
    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._is_internal_update or column != 0:
            return

        self._is_internal_update = True
        state = item.checkState(0)

        if item.childCount() > 0:
            target_state = Qt.CheckState.Checked if state != Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
            item.setCheckState(0, target_state)
            self._propagate_down(item, target_state)

        p = item.parent()
        while p and p != self.invisibleRootItem():
            self._refresh_parent_indicator(p)
            p = p.parent()

        self._is_internal_update = False
        self._sync_item_visibility_to_session(item)

    def _propagate_down(self, parent: QTreeWidgetItem, state: Qt.CheckState):
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                child.setCheckState(0, state)
                self._sync_item_visibility_to_session(child)
                if child.childCount() > 0:
                    self._propagate_down(child, state)

    def _refresh_parent_indicator(self, parent: QTreeWidgetItem):
        total = parent.childCount()
        if total == 0:
            return

        checked_count = sum(1 for i in range(total) if parent.child(i).checkState(0) == Qt.CheckState.Checked)
        partial_count = sum(1 for i in range(total) if parent.child(i).checkState(0) == Qt.CheckState.PartiallyChecked)

        if checked_count == total:
            parent.setCheckState(0, Qt.CheckState.Checked)
        elif checked_count == 0 and partial_count == 0:
            parent.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            parent.setCheckState(0, Qt.CheckState.PartiallyChecked)

    # --- PRZEŁĄCZANIE AKTYWNEJ GAŁĘZI MAGAZYNU I OKIEN ---
    def _on_selection_changed(self, current: QTreeWidgetItem, *args):
        if not current:
            return

        item_type = current.data(0, Qt.UserRole + 1)
        name = current.data(0, Qt.UserRole)

        if current == self.root_exp or (current.parent() == self.root_exp):
            session.set_storage_branch("experimental")
            session.set_context("storage")
            if item_type == "spectrum" and name:
                session.set_active_spectrum(name)

        elif current == self.root_atm or (current.parent() == self.root_atm):
            session.set_storage_branch("atm")
            session.set_context("storage")
            if item_type == "atm" and name:
                session.set_active_spectrum(name)

        elif current == self.root_series or (current.parent() == self.root_series):
            session.set_storage_branch("series")
            session.set_context("storage")
            if item_type == "series" and name:
                session.set_active_series(name)

        elif current == self.root_dft or (current.parent() == self.root_dft):
            session.set_storage_branch("dft")
            session.set_context("storage")
            if item_type == "dft" and name:
                session.set_active_spectrum(name)

        elif current == self.root_storage:
            session.set_storage_branch("experimental")
            session.set_context("storage")

        elif current == self.root_workspaces or item_type == "window_folder":
            session.set_context("window")
            # Jeśli kliknięto na folder, uaktywnij pierwsze okno w tym folderze
            if item_type == "window_folder":
                for i in range(current.childCount()):
                    c = current.child(i)
                    if c.data(0, Qt.UserRole + 1) == "workspace_window":
                        session.set_active_window(c.data(0, Qt.UserRole))
                        break

        elif item_type == "workspace_window":
            session.set_active_window(name)

        elif item_type == "window_spectrum":
            win_id = current.data(0, Qt.UserRole + 2)
            session.set_window_active_spectrum(win_id, name)

    def _sync_item_visibility_to_session(self, item: QTreeWidgetItem):
        name = item.data(0, Qt.UserRole)
        item_type = item.data(0, Qt.UserRole + 1)
        is_visible = (item.checkState(0) == Qt.CheckState.Checked)

        if item_type in ["spectrum", "dft", "atm"] and name:
            session.set_visibility(name, is_visible)
        elif item_type == "series" and name:
            session.set_series_visibility(name, is_visible)
        elif item_type == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            session.set_window_spectrum_visibility(win_id, name, is_visible)

    # =========================================================================
    # SILNIK EKSPORTU I DIALOGÓW
    # =========================================================================

    def _get_window_relative_path(self, win: WorkspaceWindow) -> str:
        """Zwraca relatywną ścieżkę katalogu okna z uwzględnieniem folderów nadrzędnych."""
        parts = [win.name]
        cur_fid = win.folder_id
        while cur_fid:
            f = session.get_folder(cur_fid)
            if f:
                parts.append(f.name)
                cur_fid = f.parent_folder_id
            else:
                break
        parts.reverse()
        clean_parts = [ExportManager.sanitize_filename(p) for p in parts]
        return os.path.join(*clean_parts)

    def _get_folder_relative_path(self, folder_id: str) -> str:
        """Zwraca relatywną ścieżkę folderu z uwzględnieniem podfolderów nadrzędnych."""
        parts = []
        cur_fid = folder_id
        while cur_fid:
            f = session.get_folder(cur_fid)
            if f:
                parts.append(f.name)
                cur_fid = f.parent_folder_id
            else:
                break
        parts.reverse()
        clean_parts = [ExportManager.sanitize_filename(p) for p in parts]
        return os.path.join(*clean_parts) if clean_parts else ""

    def _collect_export_targets(self, clicked_item: QTreeWidgetItem,
                                selected_items: List[QTreeWidgetItem]) -> Tuple[List[Tuple[SpectrumData, str]], str]:
        """
        Gromadzi obiekty widm do eksportu wraz z ich dziedziczoną ścieżką w strukturze folderów.
        Zwraca: (lista_krotek_(SpectrumData, rel_dir), sugerowana_nazwa_bazowa).
        """
        targets: List[Tuple[SpectrumData, str]] = []
        seen_keys = set()

        # Jeśli kliknięto w element będący częścią multiselekcji
        items_to_process = selected_items if (selected_items and clicked_item in selected_items and len(selected_items) > 1) else [clicked_item]

        suggested_name = "Eksport_Widm"
        if len(items_to_process) == 1:
            it = items_to_process[0]
            suggested_name = it.data(0, Qt.UserRole) or it.text(0).replace("📁 ", "").replace("🗂️ ", "").replace("📈 ", "")

        def add_spec(spec: Optional[SpectrumData], rel_dir: str):
            if spec is None:
                return
            key = (id(spec), rel_dir)
            if key not in seen_keys:
                seen_keys.add(key)
                targets.append((spec, rel_dir))

        for itm in items_to_process:
            itype = itm.data(0, Qt.UserRole + 1)
            iname = itm.data(0, Qt.UserRole)

            # 1. Pojedyncze widmo w oknie
            if itype == "window_spectrum":
                win_id = itm.data(0, Qt.UserRole + 2)
                win = session.windows.get(win_id)
                rel_dir = self._get_window_relative_path(win) if win else ""
                add_spec(win.spectra.get(iname) if win else None, rel_dir)

            # 2. Pojedyncze widmo w magazynie
            elif itype in ["spectrum", "dft"]:
                add_spec(session.get_spectrum(iname), "Pomiary_Eksperymentalne" if itype == "spectrum" else "DFT")

            elif itype == "atm":
                add_spec(session.get_spectrum(iname), "Wzorce_Atmosfery")

            # 3. Całe Okno Robocze
            elif itype == "workspace_window":
                win = session.windows.get(iname)
                if win:
                    rel_dir = self._get_window_relative_path(win)
                    for s in win.spectra.values():
                        add_spec(s, rel_dir)

            # 4. Cały Folder Okien
            elif itype == "window_folder":
                folder_id = iname

                def collect_folder_windows(fid: str) -> List[WorkspaceWindow]:
                    res = [w for w in session.windows.values() if w.folder_id == fid]
                    for sub_fid, f in session.folders.items():
                        if f.parent_folder_id == fid:
                            res.extend(collect_folder_windows(sub_fid))
                    return res

                for win in collect_folder_windows(folder_id):
                    rel_dir = self._get_window_relative_path(win)
                    for s in win.spectra.values():
                        add_spec(s, rel_dir)

            # 5. Korzeń sekcji Okien Roboczych
            elif itm == self.root_workspaces:
                for win in session.windows.values():
                    rel_dir = self._get_window_relative_path(win)
                    for s in win.spectra.values():
                        add_spec(s, rel_dir)

            # 6. Gałęzie Magazynu
            elif itm == self.root_exp:
                for s in session.spectra.values():
                    add_spec(s, "Pomiary_Eksperymentalne")
            elif itm == self.root_atm:
                for s in session.atm_spectra.values():
                    add_spec(s, "Wzorce_Atmosfery")

        return targets, ExportManager.sanitize_filename(suggested_name)

    def _export_single_spectrum_dialog(self, spec: SpectrumData, fmt_id: str):
        fmt = ExportManager.get_format_info(fmt_id)
        def_name = f"{ExportManager.sanitize_filename(spec.name)}{fmt['ext']}"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Zapisz widmo ({fmt['short_name']})", def_name, fmt['filter']
        )
        if path:
            try:
                ExportManager.export_single_spectrum(spec, path, fmt_id)
                QMessageBox.information(self, tr("app.info", "Informacja"), f"Pomyślnie zapisano widmo:\n{path}")
            except Exception as e:
                logger.error(f"Błąd zapisu widma: {e}")
                QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd zapisu widma:\n{e}")

    def _export_batch_dir_dialog(self, targets: List[Tuple[SpectrumData, str]], fmt_id: str):
        if not targets:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Brak widm do eksportu.")
            return

        fmt = ExportManager.get_format_info(fmt_id)
        dest_dir = QFileDialog.getExistingDirectory(
            self, f"Wybierz katalog docelowy do zapisu ({fmt['short_name']})"
        )
        if dest_dir:
            try:
                count = ExportManager.export_batch_to_directory(targets, dest_dir, fmt_id)
                QMessageBox.information(
                    self, tr("app.info", "Informacja"),
                    f"Pomyślnie wyeksportowano {count} widm w formacie {fmt['short_name']} do katalogu:\n{dest_dir}"
                )
            except Exception as e:
                logger.error(f"Błąd eksportu wsadowego: {e}")
                QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd zapisu widm:\n{e}")

    def _export_batch_zip_dialog(self, targets: List[Tuple[SpectrumData, str]], suggested_name: str, fmt_id: str):
        if not targets:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Brak widm do eksportu.")
            return

        fmt = ExportManager.get_format_info(fmt_id)
        def_zip = f"{ExportManager.sanitize_filename(suggested_name)}_{fmt['id']}.zip"
        zip_path, _ = QFileDialog.getSaveFileName(
            self, f"Zapisz archiwum ZIP ({fmt['short_name']})", def_zip, "Archiwum ZIP (*.zip)"
        )
        if zip_path:
            if not zip_path.lower().endswith(".zip"):
                zip_path += ".zip"
            try:
                count = ExportManager.export_batch_to_zip(targets, zip_path, fmt_id)
                QMessageBox.information(
                    self, tr("app.info", "Informacja"),
                    f"Pomyślnie spakowano {count} widm w formacie {fmt['short_name']} do archiwum ZIP:\n{zip_path}"
                )
            except Exception as e:
                logger.error(f"Błąd eksportu ZIP: {e}")
                QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd zapisu archiwum ZIP:\n{e}")

    def _populate_export_menu(self, menu: QMenu, item: QTreeWidgetItem, selected_items: List[QTreeWidgetItem]):
        """Wstrzykuje inteligentne opcje zapisu i eksportu do menu kontekstowego."""
        targets, suggested_name = self._collect_export_targets(item, selected_items)
        if not targets:
            return

        menu.addSeparator()
        last_fmt_id = ExportManager.get_last_format()
        last_fmt = ExportManager.get_format_info(last_fmt_id)
        all_fmts = ExportManager.get_all_formats()

        # Pojedyncze widmo (zapis bezpośredni)
        if len(targets) == 1 and (item.data(0, Qt.UserRole + 1) in ["spectrum", "dft", "atm", "window_spectrum"]):
            spec, _ = targets[0]

            act_quick = menu.addAction(f"⚡ Zapisz jako {last_fmt['short_name']}...")
            act_quick.triggered.connect(lambda: self._export_single_spectrum_dialog(spec, last_fmt_id))

            sub_export = menu.addMenu("💾 Zapisz jako...")
            for fmt in all_fmts:
                fid = fmt['id']
                act = sub_export.addAction(fmt['name'])
                act.triggered.connect(lambda checked=False, f=fid: self._export_single_spectrum_dialog(spec, f))

            act_zip = sub_export.addAction("📦 Spakuj do pojedynczego ZIP...")
            act_zip.triggered.connect(lambda checked=False, f=last_fmt_id: self._export_batch_zip_dialog(targets, spec.name, f))

        # Wiele widm / Okno / Folder
        else:
            act_quick_dir = menu.addAction(f"⚡ Zapisz zaznaczone ({len(targets)}) jako {last_fmt['short_name']} (Katalog)...")
            act_quick_dir.triggered.connect(lambda: self._export_batch_dir_dialog(targets, last_fmt_id))

            act_quick_zip = menu.addAction(f"⚡ Spakuj zaznaczone ({len(targets)}) jako {last_fmt['short_name']} (ZIP)...")
            act_quick_zip.triggered.connect(lambda: self._export_batch_zip_dialog(targets, suggested_name, last_fmt_id))

            sub_export = menu.addMenu(f"💾 Eksportuj zaznaczone ({len(targets)} widm)...")

            sub_dir = sub_export.addMenu("📁 Do folderu na dysku (z hierarchią)")
            for fmt in all_fmts:
                fid = fmt['id']
                act = sub_dir.addAction(fmt['name'])
                act.triggered.connect(lambda checked=False, f=fid: self._export_batch_dir_dialog(targets, f))

            sub_zip = sub_export.addMenu("📦 Do archiwum ZIP (z hierarchią)")
            for fmt in all_fmts:
                fid = fmt['id']
                act = sub_zip.addAction(f"{fmt['name']} [.zip]")
                act.triggered.connect(lambda checked=False, f=fid: self._export_batch_zip_dialog(targets, suggested_name, f))

    # --- MENU KONTEKSTOWE ---
    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        selected_items = self.selectedItems()

        # 1. KORZEŃ OKIEN ROBOCZYCH
        if item == self.root_workspaces:
            act_new_win = menu.addAction("➕ Dodaj nowe okno robocze...")
            act_new_win.triggered.connect(lambda: self._prompt_new_window())

            act_new_folder = menu.addAction("📁 Dodaj nowy folder okien...")
            act_new_folder.triggered.connect(lambda: self._prompt_new_folder())

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 2. FOLDER OKIEN ROBOCZYCH
        if item.data(0, Qt.UserRole + 1) == "window_folder":
            folder_id = item.data(0, Qt.UserRole)

            act_add_win = menu.addAction("➕ Dodaj okno do tego folderu...")
            act_add_win.triggered.connect(lambda: self._prompt_new_window(parent_folder_id=folder_id))

            act_add_subf = menu.addAction("📁 Dodaj podfolder...")
            act_add_subf.triggered.connect(lambda: self._prompt_new_folder(parent_folder_id=folder_id))

            act_rename = menu.addAction("✏️ Zmień nazwę folderu...")
            act_rename.triggered.connect(lambda: self._prompt_rename_folder(folder_id))

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tym folderze")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tym folderze")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            menu.addSeparator()
            act_del_keep = menu.addAction("🗑 Usuń folder (zachowaj okna)")
            act_del_keep.triggered.connect(lambda: session.remove_folder(folder_id, delete_windows=False))

            act_del_all = menu.addAction("🗑 Usuń folder wraz z oknami")
            act_del_all.triggered.connect(lambda: session.remove_folder(folder_id, delete_windows=True))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 3. OKNO ROBOCZE
        if item.data(0, Qt.UserRole + 1) == "workspace_window":
            win_id = item.data(0, Qt.UserRole)
            act_rename = menu.addAction("✏️ Zmień nazwę okna...")
            act_rename.triggered.connect(lambda: self._prompt_rename_window(win_id))

            # Przenoszenie do folderów
            sub_move = menu.addMenu("📁 Przenieś do folderu...")
            act_root = sub_move.addAction("Poziom główny (bez folderu)")
            act_root.triggered.connect(lambda: session.move_window_to_folder(win_id, None))
            sub_move.addSeparator()
            for fid, f in session.folders.items():
                act_f = sub_move.addAction(f"📁 {f.name}")
                act_f.triggered.connect(lambda checked=False, target_fid=fid: session.move_window_to_folder(win_id, target_fid))

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tym oknie")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tym oknie")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            if len(session.windows) > 1:
                menu.addSeparator()
                act_del = menu.addAction("🗑 Usuń okno")
                act_del.triggered.connect(lambda: session.remove_window(win_id))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 4. KATEGORIE MAGAZYNU
        if item in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tej gałęzi")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))

            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tej gałęzi")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            menu.addSeparator()
            act_clear_cat = menu.addAction("🗑 Wyczyść tę gałąź")
            act_clear_cat.triggered.connect(lambda: self._clear_category_items(item))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 5. WIDMO W MAGAZYNIE (EKSPERYMENT / DFT)
        if item.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]:
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]]
            if not names: names = [item.data(0, Qt.UserRole)]

            if len(names) > 1:
                act_move_atm = menu.addAction(f"☁️ Przenieś zaznaczone ({len(names)}) do Wzorców Atmosfery")
                act_move_atm.triggered.connect(lambda: [session.move_to_atm(n) for n in names])
                act_rem = menu.addAction(f"🗑 Usuń zaznaczone ({len(names)}) z magazynu")
                act_rem.triggered.connect(lambda: [session.remove_spectrum(n) for n in names])
            else:
                s_name = names[0]
                act_move_atm = menu.addAction("☁️ Przenieś do Wzorców Atmosfery (ATM / CO₂)")
                act_move_atm.triggered.connect(lambda: session.move_to_atm(s_name))
                act_rem = menu.addAction(f"🗑 Usuń '{s_name}' z magazynu")
                act_rem.triggered.connect(lambda: session.remove_spectrum(s_name))

            if session.active_window_id:
                menu.addSeparator()
                act_add_win = menu.addAction("➕ Dodaj do aktywnego okna roboczego")
                act_add_win.triggered.connect(lambda: [session.add_spectrum_to_window(session.active_window_id, session.get_spectrum(n)) for n in names if session.get_spectrum(n)])

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 6. WZORCE ATMOSFERY
        if item.data(0, Qt.UserRole + 1) == "atm":
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) == "atm"]
            if not names: names = [item.data(0, Qt.UserRole)]

            if len(names) > 1:
                act_move_exp = menu.addAction(f"📈 Przenieś zaznaczone ({len(names)}) do Pomiarów Eksperymentalnych")
                act_move_exp.triggered.connect(lambda: [session.move_to_experimental(n) for n in names])
                act_rem = menu.addAction(f"🗑 Usuń zaznaczone ({len(names)}) wzorce atmosfery")
                act_rem.triggered.connect(lambda: [session.remove_spectrum(n) for n in names])
            else:
                s_name = names[0]
                act_move_exp = menu.addAction("📈 Przenieś do Pomiarów Eksperymentalnych")
                act_move_exp.triggered.connect(lambda: session.move_to_experimental(s_name))
                act_rem = menu.addAction(f"🗑 Usuń '{s_name}'")
                act_rem.triggered.connect(lambda: session.remove_spectrum(s_name))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 7. WIDMO W OKNIE ROBOCZYM
        if item.data(0, Qt.UserRole + 1) == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) == "window_spectrum"]
            if not names: names = [item.data(0, Qt.UserRole)]

            if len(names) > 1:
                act_rem = menu.addAction(f"✕ Usuń zaznaczone ({len(names)}) z tego okna")
                act_rem.triggered.connect(lambda: [session.remove_spectrum_from_window(win_id, n) for n in names])
            else:
                spec_name = names[0]
                act_rem = menu.addAction(f"✕ Usuń '{spec_name}' z tego okna")
                act_rem.triggered.connect(lambda: session.remove_spectrum_from_window(win_id, spec_name))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 8. SERIE 2D / 3D
        if item.data(0, Qt.UserRole + 1) == "series":
            s_name = item.data(0, Qt.UserRole)
            act_rem_ser = menu.addAction(f"🗑 Usuń serię '{s_name}'")
            act_rem_ser.triggered.connect(lambda: session.remove_series(s_name))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

    def _set_category_check_state(self, cat_item: QTreeWidgetItem, state: Qt.CheckState):
        self._is_internal_update = True
        cat_item.setCheckState(0, state)
        self._propagate_down(cat_item, state)
        self._is_internal_update = False

    def _clear_category_items(self, cat_item: QTreeWidgetItem):
        names = [cat_item.child(i).data(0, Qt.UserRole) for i in range(cat_item.childCount())]
        for name in names:
            if cat_item == self.root_series:
                session.remove_series(name)
            else:
                session.remove_spectrum(name)

    def _prompt_new_folder(self, parent_folder_id: Optional[str] = None):
        name, ok = QInputDialog.getText(self, "Nowy Folder", "Podaj nazwę folderu okien roboczych:")
        if ok and name.strip():
            session.add_folder(name.strip(), parent_folder_id=parent_folder_id)

    def _prompt_rename_folder(self, folder_id: str):
        folder = session.get_folder(folder_id)
        if folder:
            name, ok = QInputDialog.getText(self, "Zmień nazwę", "Nowa nazwa folderu:", text=folder.name)
            if ok and name.strip():
                session.rename_folder(folder_id, name.strip())

    def _prompt_new_window(self, parent_folder_id: Optional[str] = None):
        name, ok = QInputDialog.getText(self, "Nowe Okno", "Podaj nazwę nowego okna roboczego:")
        if ok and name.strip():
            session.add_window(name.strip(), folder_id=parent_folder_id)

    def _prompt_rename_window(self, win_id: str):
        if win_id in session.windows:
            curr_name = session.windows[win_id].name
            name, ok = QInputDialog.getText(self, "Zmień nazwę", "Nowa nazwa okna:", text=curr_name)
            if ok and name.strip():
                session.windows[win_id].name = name.strip()
                win_item = self._find_window_item(win_id)
                if win_item:
                    win_item.setText(0, f"🗂️ {name.strip()}")