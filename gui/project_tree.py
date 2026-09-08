"""
gui/project_tree.py
Drzewo obiektów projektu rozdzielające Magazyn Danych (read-only / batch prep)
od Przestrzeni Roboczych (Okien Wykresów z automatycznym klonowaniem).
"""

import logging
from PySide6.QtCore import Qt, QThreadPool, QRunnable, Slot
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QAbstractItemView,
    QFileDialog, QMessageBox, QInputDialog
)

from core.session import session
from core.models import GaussianData
from io_hub.dispatcher import SpectrumImportDispatcher
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
    """Eksplorator z podziałem na Magazyn Danych i Izolowane Okna Wykresów."""

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
        self.root_series = QTreeWidgetItem(self.root_storage,
                                           [tr("project_tree.group_series", "Serie Ewolucyjne (2D)")])
        self.root_dft = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_dft", "Obliczenia Kwantowe (DFT)")])

        # Pozwalamy na przeciąganie (Drag) całych gałęzi kategorii
        for sub in (self.root_exp, self.root_series, self.root_dft):
            sub.setExpanded(True)
            sub.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            sub.setCheckState(0, Qt.CheckState.Checked)
            sub.setData(0, Qt.UserRole + 1, "category")

        # 2. SEKCYJNY KORZEŃ: PRZESTRZENIE ROBOCZE
        self.root_workspaces = QTreeWidgetItem(self, ["🖥️ OKNA WYKRESÓW / ROBOCZE"])
        self.root_workspaces.setExpanded(True)
        self.root_workspaces.setData(0, Qt.UserRole + 1, "section_header")

        for win_id, win in session.windows.items():
            self._create_window_item(win_id, win.name)

    def _connect_signals(self):
        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_selection_changed)
        self.itemClicked.connect(self._on_selection_changed)  # Reakcja także na kliknięcie w to samo okno
        self.customContextMenuRequested.connect(self._show_context_menu)

        session.spectrum_added.connect(self._on_spectrum_added)
        session.spectra_batch_added.connect(self._on_spectra_batch_added)
        session.spectrum_removed.connect(self._on_spectrum_removed)
        session.series_added.connect(self._on_series_added)
        session.series_removed.connect(self._on_series_removed)

        session.window_added.connect(self._on_window_added)
        session.window_removed.connect(self._on_window_removed)
        session.window_spectrum_added.connect(self._on_window_spectrum_added)
        session.window_spectrum_removed.connect(self._on_window_spectrum_removed)

    # --- ZARZĄDZANIE OKNAMI W DRZEWIE ---
    def _create_window_item(self, window_id: str, name: str) -> QTreeWidgetItem:
        win_item = QTreeWidgetItem(self.root_workspaces, [f"🗂️ {name}"])
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

    def _on_window_added(self, window_id: str, name: str):
        self._create_window_item(window_id, name)

    def _on_window_removed(self, window_id: str):
        for i in range(self.root_workspaces.childCount()):
            child = self.root_workspaces.child(i)
            if child and child.data(0, Qt.UserRole) == window_id:
                self.root_workspaces.removeChild(child)
                return

    def _on_window_spectrum_added(self, window_id: str, spec):
        for i in range(self.root_workspaces.childCount()):
            win_node = self.root_workspaces.child(i)
            if win_node and win_node.data(0, Qt.UserRole) == window_id:
                item = QTreeWidgetItem(win_node, [f"📈 {spec.name}"])
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)
                item.setData(0, Qt.UserRole, spec.name)
                item.setData(0, Qt.UserRole + 1, "window_spectrum")
                item.setData(0, Qt.UserRole + 2, window_id)
                win_node.setExpanded(True)
                break

    def _on_window_spectrum_removed(self, window_id: str, name: str):
        for i in range(self.root_workspaces.childCount()):
            win_node = self.root_workspaces.child(i)
            if win_node and win_node.data(0, Qt.UserRole) == window_id:
                for j in range(win_node.childCount()):
                    s_item = win_node.child(j)
                    if s_item and s_item.data(0, Qt.UserRole) == name:
                        win_node.removeChild(s_item)
                        return

    # --- DRAG & DROP: CZYSTE KLONOWANIE Z MAGAZYNU DO OKIEN ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        # 1. Przeciąganie plików z eksploratora systemowego (Finder / Explorer)
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                worker = FileImportWorker(paths)
                self.thread_pool.start(worker)
                event.acceptProposedAction()
                return

        # 2. Przeciąganie wewnątrz drzewa na Okno Robocze
        target_item = self.itemAt(event.position().toPoint())
        if not target_item:
            event.ignore()
            return

        target_win_id = None
        target_type = target_item.data(0, Qt.UserRole + 1)

        if target_type == "workspace_window":
            target_win_id = target_item.data(0, Qt.UserRole)
        elif target_type == "window_spectrum":
            target_win_id = target_item.data(0, Qt.UserRole + 2)

        if target_win_id:
            selected = self.selectedItems()

            # Rekurencyjny kolektor widm (zbiera zaznaczone pojedyncze widma oraz całe gałęzie)
            def collect_specs(tree_item) -> list:
                specs = []
                itype = tree_item.data(0, Qt.UserRole + 1)
                iname = tree_item.data(0, Qt.UserRole)
                if itype in ["spectrum", "dft"] and iname:
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
                if s_name in session.spectra:
                    src_spec = session.spectra[s_name]
                    session.add_spectrum_to_window(target_win_id, src_spec)

            event.acceptProposedAction()
            return

        event.ignore()

    # --- SYNCHRONIZACJA ELEMENTÓW MAGAZYNU ---
    def _create_spectrum_item(self, spectrum) -> tuple[QTreeWidgetItem, QTreeWidgetItem]:
        if isinstance(spectrum, GaussianData) or getattr(spectrum, 'file_type', '') in ['gaussian', 'orca', 'dft']:
            target_root = self.root_dft
            icon = "⚛️"
            item_type = "dft"
        else:
            target_root = self.root_exp
            icon = "📈"
            item_type = "spectrum"

        item = QTreeWidgetItem(target_root, [f"{icon} {spectrum.name}"])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
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

    def _on_spectrum_removed(self, name: str):
        for root in (self.root_exp, self.root_series, self.root_dft):
            for i in range(root.childCount()):
                child = root.child(i)
                if child and child.data(0, Qt.UserRole) == name:
                    root.removeChild(child)
                    self._refresh_parent_indicator(root)
                    return

    def _on_series_added(self, series_obj):
        dim_tag = "3D Tensor" if getattr(series_obj, 'is_3d', False) else "2D Seria"
        n_spectra = series_obj.n_spectra if hasattr(series_obj, 'n_spectra') else series_obj.data.shape[1]
        label = f"📊 {series_obj.name} [{dim_tag}, {n_spectra} widm]"

        series_item = QTreeWidgetItem(self.root_series, [label])
        series_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
        series_item.setCheckState(0, Qt.CheckState.Checked)
        series_item.setData(0, Qt.UserRole, series_obj.name)
        series_item.setData(0, Qt.UserRole + 1, "series")
        self._refresh_parent_indicator(self.root_series)

    def _on_series_removed(self, name: str):
        for i in range(self.root_series.childCount()):
            child = self.root_series.child(i)
            if child and child.data(0, Qt.UserRole) == name:
                self.root_series.removeChild(child)
                self._refresh_parent_indicator(self.root_series)
                return

    # --- KONTROLA CHECKBOXÓW I PROPAGACJA ---
    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._is_internal_update or column != 0:
            return

        self._is_internal_update = True
        state = item.checkState(0)

        if item.childCount() > 0:
            target_state = Qt.CheckState.Checked if state == Qt.CheckState.Checked else Qt.CheckState.Unchecked
            item.setCheckState(0, target_state)
            self._propagate_down(item, target_state)

        p = item.parent()
        while p and p not in (self.root_storage, self.root_workspaces):
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

    def _on_selection_changed(self, current: QTreeWidgetItem, *args):
        if not current:
            return

        item_type = current.data(0, Qt.UserRole + 1)
        name = current.data(0, Qt.UserRole)

        # 1. KLIKNIĘCIE W GAŁĄŹ MAGAZYNU (Nagłówek sekcji, Kategoria lub konkretne widmo)
        if current == self.root_storage or item_type == "category" or item_type in ["spectrum", "dft", "series"]:
            session.set_context("storage")

            if item_type in ["spectrum", "dft"]:
                session.set_active_spectrum(name)
            elif item_type == "series":
                session.set_active_series(name)
            elif item_type == "category" and current.childCount() > 0:
                # Jeśli kliknięto całą gałąź (np. "Pomiary Eksperymentalne"), wskaż jej pierwsze widmo
                first_child = current.child(0)
                first_name = first_child.data(0, Qt.UserRole)
                if first_name:
                    session.set_active_spectrum(first_name)

        # 2. KLIKNIĘCIE W SEKCJĘ OKIEN ROBOCZYCH
        elif current == self.root_workspaces:
            session.set_context("window")

        # 3. KLIKNIĘCIE W KONKRETNE OKNO ROBOCZE (np. "Okno 1")
        elif item_type == "workspace_window":
            session.set_active_window(name)

        # 4. KLIKNIĘCIE W WIDMO WEWNĄTRZ OKNA
        elif item_type == "window_spectrum":
            win_id = current.data(0, Qt.UserRole + 2)
            session.set_window_active_spectrum(win_id, name)

    def _sync_item_visibility_to_session(self, item: QTreeWidgetItem):
        name = item.data(0, Qt.UserRole)
        item_type = item.data(0, Qt.UserRole + 1)
        is_visible = (item.checkState(0) == Qt.CheckState.Checked)

        if item_type in ["spectrum", "dft"] and name:
            session.set_visibility(name, is_visible)
        elif item_type == "series" and name:
            session.set_series_visibility(name, is_visible)
        elif item_type == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            session.set_window_spectrum_visibility(win_id, name, is_visible)

    # --- MENU KONTEKSTOWE ---
    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        menu = QMenu(self)

        # 1. Menu dla sekcji Okien Wykresów
        if item and item == self.root_workspaces:
            act_new_win = menu.addAction("➕ Dodaj nowe okno robocze...")
            act_new_win.triggered.connect(self._prompt_new_window)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 2. Menu dla konkretnego Okna Roboczego
        if item and item.data(0, Qt.UserRole + 1) == "workspace_window":
            win_id = item.data(0, Qt.UserRole)
            act_rename = menu.addAction("✏️ Zmień nazwę okna...")
            act_rename.triggered.connect(lambda: self._prompt_rename_window(win_id))

            if len(session.windows) > 1:
                act_del = menu.addAction("🗑 Usuń okno")
                act_del.triggered.connect(lambda: session.remove_window(win_id))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 3. Menu dla widma w Oknie Roboczym
        if item and item.data(0, Qt.UserRole + 1) == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            spec_name = item.data(0, Qt.UserRole)
            act_rem = menu.addAction(f"✕ Usuń '{spec_name}' z tego okna")
            act_rem.triggered.connect(lambda: session.remove_spectrum_from_window(win_id, spec_name))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 4. Standardowe menu dla Magazynu Danych
        if item and item.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]:
            spec_name = item.data(0, Qt.UserRole)
            act_rem_mag = menu.addAction(f"🗑 Usuń '{spec_name}' z magazynu")
            act_rem_mag.triggered.connect(lambda: session.remove_spectrum(spec_name))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

    def _prompt_new_window(self):
        name, ok = QInputDialog.getText(self, "Nowe Okno", "Podaj nazwę nowego okna roboczego:")
        if ok and name.strip():
            session.add_window(name.strip())

    def _prompt_rename_window(self, win_id: str):
        if win_id in session.windows:
            curr_name = session.windows[win_id].name
            name, ok = QInputDialog.getText(self, "Zmień nazwę", "Nowa nazwa okna:", text=curr_name)
            if ok and name.strip():
                session.windows[win_id].name = name.strip()
                for i in range(self.root_workspaces.childCount()):
                    child = self.root_workspaces.child(i)
                    if child.data(0, Qt.UserRole) == win_id:
                        child.setText(0, f"🗂️ {name.strip()}")
                        break