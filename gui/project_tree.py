"""
gui/project_tree.py
Drzewo obiektów projektu z automatycznym przełączaniem aktywnej gałęzi magazynu.
"""

import logging
from PySide6.QtCore import Qt, QThreadPool, QRunnable, Slot
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QAbstractItemView,
    QInputDialog
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
    """Eksplorator z podziałem na Gałęzie Magazynu i Izolowane Okna Wykresów."""

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

        for win_id, win in session.windows.items():
            self._create_window_item(win_id, win.name)

    def _connect_signals(self):
        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_selection_changed)
        self.itemClicked.connect(self._on_selection_changed)
        self.customContextMenuRequested.connect(self._show_context_menu)

        session.spectrum_added.connect(self._on_spectrum_added)
        session.spectra_batch_added.connect(self._on_spectra_batch_added)
        session.spectrum_removed.connect(self._on_spectrum_removed)

        session.atm_spectrum_added.connect(self._on_atm_spectrum_added)
        session.atm_spectra_batch_added.connect(self._on_atm_spectra_batch_added)
        session.atm_spectrum_removed.connect(self._on_atm_spectrum_removed)

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

        target_win_id = None
        target_type = target_item.data(0, Qt.UserRole + 1)

        if target_type == "workspace_window":
            target_win_id = target_item.data(0, Qt.UserRole)
        elif target_type == "window_spectrum":
            target_win_id = target_item.data(0, Qt.UserRole + 2)

        if target_win_id:
            selected = self.selectedItems()

            def collect_specs(tree_item) -> list:
                specs = []
                itype = tree_item.data(0, Qt.UserRole + 1)
                iname = tree_item.data(0, Qt.UserRole)
                if itype in ["spectrum", "dft", "atm"] and iname:
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
            parent.setCheckState(0, Qt.CheckState.Unchecked)
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

        # 1. Pomiary Eksperymentalne
        if current == self.root_exp or (current.parent() == self.root_exp):
            session.set_storage_branch("experimental")
            session.set_context("storage")
            if item_type == "spectrum" and name:
                session.set_active_spectrum(name)
            elif current == self.root_exp and self.root_exp.childCount() > 0:
                fn = self.root_exp.child(0).data(0, Qt.UserRole)
                if fn: session.set_active_spectrum(fn)

        # 2. Wzorce Atmosfery
        elif current == self.root_atm or (current.parent() == self.root_atm):
            session.set_storage_branch("atm")
            session.set_context("storage")
            if item_type == "atm" and name:
                session.set_active_spectrum(name)
            elif current == self.root_atm and self.root_atm.childCount() > 0:
                fn = self.root_atm.child(0).data(0, Qt.UserRole)
                if fn: session.set_active_spectrum(fn)

        # 3. Serie 2D / 3D
        elif current == self.root_series or (current.parent() == self.root_series):
            session.set_storage_branch("series")
            session.set_context("storage")
            if item_type == "series" and name:
                session.set_active_series(name)

        # 4. Obliczenia Kwantowe DFT
        elif current == self.root_dft or (current.parent() == self.root_dft):
            session.set_storage_branch("dft")
            session.set_context("storage")
            if item_type == "dft" and name:
                session.set_active_spectrum(name)

        # 5. Cały magazyn (kliknięcie w główny korzeń)
        elif current == self.root_storage:
            session.set_storage_branch("experimental")
            session.set_context("storage")

        # 6. Okna Robocze
        elif current == self.root_workspaces:
            session.set_context("window")

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

    # --- MENU KONTEKSTOWE ---
    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        menu = QMenu(self)
        selected_items = self.selectedItems()

        if item and item == self.root_workspaces:
            act_new_win = menu.addAction("➕ Dodaj nowe okno robocze...")
            act_new_win.triggered.connect(self._prompt_new_window)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item.data(0, Qt.UserRole + 1) == "workspace_window":
            win_id = item.data(0, Qt.UserRole)
            act_rename = menu.addAction("✏️ Zmień nazwę okna...")
            act_rename.triggered.connect(lambda: self._prompt_rename_window(win_id))

            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tym oknie")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tym oknie")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            if len(session.windows) > 1:
                menu.addSeparator()
                act_del = menu.addAction("🗑 Usuń okno")
                act_del.triggered.connect(lambda: session.remove_window(win_id))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tej gałęzi")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))

            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tej gałęzi")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            menu.addSeparator()
            act_clear_cat = menu.addAction("🗑 Wyczyść tę gałąź")
            act_clear_cat.triggered.connect(lambda: self._clear_category_items(item))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]:
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
                act_add_win = menu.addAction(f"➕ Dodaj do aktywnego okna roboczego")
                act_add_win.triggered.connect(lambda: [session.add_spectrum_to_window(session.active_window_id, session.get_spectrum(n)) for n in names if session.get_spectrum(n)])

            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item.data(0, Qt.UserRole + 1) == "atm":
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

            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item.data(0, Qt.UserRole + 1) == "series":
            s_name = item.data(0, Qt.UserRole)
            act_rem_ser = menu.addAction(f"🗑 Usuń serię '{s_name}'")
            act_rem_ser.triggered.connect(lambda: session.remove_series(s_name))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if item and item.data(0, Qt.UserRole + 1) == "window_spectrum":
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