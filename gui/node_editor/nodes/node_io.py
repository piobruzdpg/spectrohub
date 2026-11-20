"""
gui/node_editor/nodes/node_io.py
Węzły wejściowe i referencyjne potoku DAG:
- InputNode ('input'): dynamiczne źródło kontekstowe (aktywne, widoczne, całe okno robocze, seria)
- SpectrumSelectNode ('spectrum_select'): eksplorator drzewiasty folderów, okien i magazynu z filtrem
- AtmSelectNode ('atm_select'): dedykowane źródło wzorców atmosfery (H2O / CO2)
- MergeSpectraNode ('merge_spectra'): multiplexer łączący wiele gałęzi w macierz 2D
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import fnmatch
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QCheckBox, QRadioButton, QButtonGroup
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from core.session import session
from core.models import SpectrumData


def _natural_sort_key(s: str) -> list:
    """Klucz sortowania naturalnego (np. seria_2 przed seria_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def _get_folder_breadcrumb(folder_id: Optional[str]) -> str:
    """Buduje czytelną ścieżkę folderów (np. 'Wyniki / AsLS') dla okna roboczego."""
    parts = []
    cur_fid = folder_id
    visited = set()
    while cur_fid and cur_fid not in visited:
        visited.add(cur_fid)
        folder = session.get_folder(cur_fid)
        if not folder:
            break
        parts.append(folder.name)
        cur_fid = folder.parent_folder_id
    parts.reverse()
    return " / ".join(parts)


# =============================================================================
# 1. WĘZEŁ STARTU KONTEKSTOWEGO: InputNode ('input')
# =============================================================================

class InputNodeDialog(QDialog):
    def __init__(self, node: 'InputNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle(f"Parametry węzła: {node.title}")
        self.resize(480, 260)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_source_mode = QComboBox()
        self.combo_source_mode.addItem("Tylko aktywne widmo (1D)", "active")
        self.combo_source_mode.addItem("Wszystkie widoczne w aktywnym oknie / gałęzi (Batch 2D)", "visible")
        self.combo_source_mode.addItem("Wszystkie widma ze wskazanego okna roboczego (Batch 2D)", "window")
        self.combo_source_mode.addItem("Aktywna seria kinetyczna / Tensor", "series")

        cur_mode = self.step.get('source_mode', 'active')
        if cur_mode == 'specific':
            cur_mode = 'active'
        idx_m = self.combo_source_mode.findData(cur_mode)
        if idx_m >= 0:
            self.combo_source_mode.setCurrentIndex(idx_m)
        form.addRow("Tryb źródła danych:", self.combo_source_mode)

        self.combo_target_window = QComboBox()
        sorted_wins = sorted(
            session.windows.values(),
            key=lambda w: (_natural_sort_key(_get_folder_breadcrumb(w.folder_id)), _natural_sort_key(w.name))
        )
        for win in sorted_wins:
            crumb = _get_folder_breadcrumb(win.folder_id)
            prefix = f"📁 {crumb} / " if crumb else ""
            label = f"{prefix}🖼️ {win.name} ({len(win.spectra)} widm)"
            self.combo_target_window.addItem(label, win.window_id)

        cur_win_id = self.step.get('target_window_id', '')
        idx_w = self.combo_target_window.findData(cur_win_id)
        if idx_w >= 0:
            self.combo_target_window.setCurrentIndex(idx_w)
        form.addRow("Docelowe okno robocze:", self.combo_target_window)

        def _toggle_controls(idx):
            mode = self.combo_source_mode.itemData(idx)
            self.combo_target_window.setEnabled(mode == "window")

        self.combo_source_mode.currentIndexChanged.connect(_toggle_controls)
        _toggle_controls(self.combo_source_mode.currentIndex())

        lbl_hint = QLabel(
            "💡 Tryb 'Wskazane okno robocze' pobiera wszystkie widma z wybranego okna\n"
            "bez konieczności przełączania na nie w głównym drzewie projektu.\n"
            "Do wyboru pojedynczych plików z folderów użyj węzła '🗂️ Wybór z Projektu'."
        )
        lbl_hint.setStyleSheet("color: #a8dadc; font-size: 11px; padding-top: 6px;")
        form.addRow(lbl_hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        mode = self.combo_source_mode.currentData()
        self.step['source_mode'] = mode

        if mode == 'window':
            win_id = self.combo_target_window.currentData() or ""
            win = session.windows.get(win_id)
            self.step['target_window_id'] = win_id
            self.step['target_window_name'] = win.name if win else ""
            w_label = win.name if win else "(brak okna)"
            short_w = w_label[:18] + "…" if len(w_label) > 18 else w_label
            self.step['name'] = f"Okno: {short_w}"
        elif mode == 'visible':
            self.step['name'] = "Źródło: Wszystkie widoczne"
        elif mode == 'series':
            self.step['name'] = "Źródło: Aktywna seria"
        else:
            self.step['name'] = "Źródło: Aktywne widmo"
        self.accept()


@register_node
class InputNode(BaseNode):
    node_type = "input"
    title = "📥 ŹRÓDŁO DANYCH"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#1b4332", "#2d6a4f")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Źródło: Aktywne widmo",
            "source_mode": "active",
            "target_window_id": "",
            "target_window_name": "",
            "specific_spectrum": ""
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#52b788"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return InputNodeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        source_mode = self.step_data.get('source_mode', 'active')

        if source_mode == 'window':
            win_id = self.step_data.get('target_window_id', '')
            win = session.windows.get(win_id)
            if not win and self.step_data.get('target_window_name'):
                target_n = self.step_data['target_window_name'].lower()
                for w in session.windows.values():
                    if w.name.lower() == target_n:
                        win = w
                        break

            if win and win.spectra:
                specs = sorted(win.spectra.values(), key=lambda s: _natural_sort_key(s.name))
                if len(specs) == 1:
                    return {"out": (specs[0].wavenumbers.copy(), specs[0].intensities.copy())}, {"out": [specs[0].name]}

                ref_wn = specs[0].wavenumbers
                cols, names = [], []
                for s in specs:
                    xw, yw = (s.wavenumbers[::-1], s.intensities[::-1]) if s.wavenumbers[0] > s.wavenumbers[-1] else (s.wavenumbers, s.intensities)
                    if len(xw) == len(ref_wn) and np.allclose(xw, ref_wn, atol=1e-4):
                        cols.append(yw.copy())
                    else:
                        cols.append(np.interp(ref_wn, xw, yw))
                    names.append(s.name)
                return {"out": (ref_wn.copy(), np.column_stack(cols))}, {"out": names}

        elif source_mode == 'specific':
            spec_name = self.step_data.get('specific_spectrum', '')
            spec = session.get_spectrum(spec_name)
            if spec:
                return {"out": (spec.wavenumbers.copy(), spec.intensities.copy())}, {"out": [spec.name]}

        elif source_mode == 'visible':
            visible = session.get_pipeline_target_spectra()
            if len(visible) >= 2:
                ref_wn = visible[0].wavenumbers
                cols, names = [], []
                for s in visible:
                    xw, yw = (s.wavenumbers[::-1], s.intensities[::-1]) if s.wavenumbers[0] > s.wavenumbers[-1] else (s.wavenumbers, s.intensities)
                    if len(xw) == len(ref_wn) and np.allclose(xw, ref_wn, atol=1e-4):
                        cols.append(yw.copy())
                    else:
                        cols.append(np.interp(ref_wn, xw, yw))
                    names.append(s.name)
                return {"out": (ref_wn.copy(), np.column_stack(cols))}, {"out": names}
            elif len(visible) == 1:
                return {"out": (visible[0].wavenumbers.copy(), visible[0].intensities.copy())}, {"out": [visible[0].name]}

        elif source_mode == 'series':
            if session.active_series:
                mat = session.active_series.get_matrix_2d()
                names = [f"{session.active_series.name}_f{i + 1}" for i in range(mat.shape[1])]
                return {"out": (session.active_series.wavenumbers.copy(), mat.copy())}, {"out": names}

        active = session.active_spectrum
        if active and len(active.wavenumbers) > 2:
            return {"out": (active.wavenumbers.copy(), active.intensities.copy())}, {"out": [active.name]}

        def_x = np.linspace(4000, 400, 1000)
        return {"out": (def_x, np.zeros_like(def_x))}, {"out": ["Brak danych"]}


# =============================================================================
# 2. WĘZEŁ EKSPLORATORA PROJEKTU: SpectrumSelectNode ('spectrum_select')
# =============================================================================

class SpectrumSelectNodeDialog(QDialog):
    """Drzewiasty selektor widm z folderów, okien roboczych i magazynu z filtrowaniem na żywo."""

    def __init__(self, node: 'SpectrumSelectNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Wybór widm z struktury projektu")
        self.resize(560, 520)
        self._is_updating_checks = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Pasek filtrowania
        filter_row = QHBoxLayout()
        self.ent_filter = QLineEdit()
        self.ent_filter.setPlaceholderText("🔍 Filtruj po nazwie (np. seria_04, *50C*, lub s01_1, s01_2)...")
        self.ent_filter.setClearButtonEnabled(True)
        self.ent_filter.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.ent_filter)
        layout.addLayout(filter_row)

        # Przyciski szybkiej selekcji
        quick_row = QHBoxLayout()
        btn_check_visible = QPushButton("☑ Zaznacz pasujące")
        btn_check_visible.setToolTip("Dodaje do zaznaczenia wszystkie widma widoczne po przefiltrowaniu")
        btn_check_visible.clicked.connect(self._check_all_visible_leaves)

        btn_isolate_visible = QPushButton("🎯 Tylko pasujące")
        btn_isolate_visible.setToolTip("Zaznacza wyłącznie widma pasujące do filtra, odznaczając resztę")
        btn_isolate_visible.clicked.connect(self._isolate_visible_leaves)

        btn_clear = QPushButton("☐ Wyczyść wszystko")
        btn_clear.clicked.connect(self._uncheck_all)

        quick_row.addWidget(btn_check_visible)
        quick_row.addWidget(btn_isolate_visible)
        quick_row.addWidget(btn_clear)
        layout.addLayout(quick_row)

        # Drzewo projektu
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1a1a24;
                alternate-background-color: #20202c;
                color: #e0e0e0;
                border: 1px solid #3d3d4e;
                border-radius: 4px;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 3px 2px;
            }
            QTreeWidget::item:selected {
                background-color: #2b4c6f;
            }
        """)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

        # Opcja wyrównania siatki X
        self.chk_interp = QCheckBox("Wyrównaj (interpoluj) różne siatki osi X do pierwszego wybranego widma")
        self.chk_interp.setChecked(bool(self.step.get('interpolate_x', False)))
        self.chk_interp.setToolTip(
            "Gdy odznaczone (zalecane do uśredniania), węzeł sprawdza ścisłą zgodność "
            "liczby punktów i zakresu liczb falowych."
        )
        layout.addWidget(self.chk_interp)

        # Pasek statusu + OK/Anuluj
        bottom_row = QHBoxLayout()
        self.lbl_count = QLabel("Wybrano: 0 widm")
        self.lbl_count.setStyleSheet("color: #48cae4; font-weight: bold; font-size: 12px;")
        bottom_row.addWidget(self.lbl_count)
        bottom_row.addStretch()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        bottom_row.addWidget(btns)
        layout.addLayout(bottom_row)

        self._populate_tree()
        self._update_selected_counter()

    def _get_saved_selection_keys(self) -> Tuple[set, set]:
        """Zwraca (zbiór (window_id, spec_name), zbiór storage_spec_name) na podstawie zapisu."""
        win_keys = set()
        storage_keys = set()

        items = self.step.get('selected_items', [])
        if items:
            for entry in items:
                scope = entry.get('scope', 'storage')
                sname = entry.get('name', '')
                if not sname:
                    continue
                if scope == 'window':
                    wid = entry.get('window_id', '')
                    win_keys.add((wid, sname))
                else:
                    storage_keys.add(sname)
        else:
            # Kompatybilność wsteczna ze starym formatem selected_spectra / selected_spectrum
            legacy = self.step.get('selected_spectra', [])
            if not legacy and self.step.get('selected_spectrum'):
                legacy = [self.step.get('selected_spectrum')]
            for sname in legacy:
                storage_keys.add(sname)

        return win_keys, storage_keys

    def _make_group_item(self, parent, label: str, expanded: bool = True) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label])
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Unchecked)
        item.setExpanded(expanded)
        return item

    def _make_leaf_item(self, parent, label: str, payload: Dict[str, Any], is_checked: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label])
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
        item.setData(0, Qt.UserRole, payload)
        return item

    def _populate_tree(self):
        self._is_updating_checks = True
        self.tree.clear()

        win_keys, storage_keys = self._get_saved_selection_keys()

        # 1. OKNA WYKRESÓW / ROBOCZE + FOLDERY (najczęściej używane do obrobionych widm)
        root_ws = self._make_group_item(self.tree, "🖥️ OKNA WYKRESÓW / ROBOCZE", expanded=True)

        folder_items: Dict[str, QTreeWidgetItem] = {}
        sorted_folders = sorted(session.folders.values(), key=lambda f: _natural_sort_key(f.name))

        # Tworzenie węzłów folderów
        for folder in sorted_folders:
            f_item = self._make_group_item(root_ws, f"📁 {folder.name}", expanded=True)
            folder_items[folder.folder_id] = f_item

        # Przypięcie podfolderów do folderów nadrzędnych
        for folder in sorted_folders:
            if folder.parent_folder_id and folder.parent_folder_id in folder_items:
                f_item = folder_items[folder.folder_id]
                root_ws.removeChild(f_item)
                folder_items[folder.parent_folder_id].addChild(f_item)

        # Wstawienie okien roboczych do odpowiednich folderów (lub pod root_ws)
        sorted_wins = sorted(session.windows.values(), key=lambda w: _natural_sort_key(w.name))
        for win in sorted_wins:
            parent_node = folder_items.get(win.folder_id, root_ws)
            w_item = self._make_group_item(
                parent_node,
                f"🖼️ {win.name} ({len(win.spectra)} widm)",
                expanded=True
            )
            sorted_specs = sorted(win.spectra.values(), key=lambda s: _natural_sort_key(s.name))
            for spec in sorted_specs:
                checked = (win.window_id, spec.name) in win_keys
                payload = {
                    "scope": "window",
                    "window_id": win.window_id,
                    "window_name": win.name,
                    "name": spec.name
                }
                self._make_leaf_item(w_item, f"📈 {spec.name}", payload, checked)

        # 2. MAGAZYN DANYCH (Pomiary Eksperymentalne i DFT)
        root_storage = self._make_group_item(self.tree, "📦 MAGAZYN DANYCH (Referencja)", expanded=False)

        exp_specs = [
            s for s in session.spectra.values()
            if getattr(s, 'file_type', '') not in ('gaussian', 'orca', 'dft')
        ]
        dft_specs = [
            s for s in session.spectra.values()
            if getattr(s, 'file_type', '') in ('gaussian', 'orca', 'dft')
        ]

        if exp_specs:
            cat_exp = self._make_group_item(
                root_storage, f"📈 Pomiary Eksperymentalne ({len(exp_specs)})", expanded=True
            )
            for spec in sorted(exp_specs, key=lambda s: _natural_sort_key(s.name)):
                checked = spec.name in storage_keys
                payload = {"scope": "storage", "name": spec.name}
                self._make_leaf_item(cat_exp, f"📈 {spec.name}", payload, checked)

        if dft_specs:
            cat_dft = self._make_group_item(
                root_storage, f"⚛️ Obliczenia Kwantowe DFT ({len(dft_specs)})", expanded=False
            )
            for spec in sorted(dft_specs, key=lambda s: _natural_sort_key(s.name)):
                checked = spec.name in storage_keys
                payload = {"scope": "storage", "name": spec.name}
                self._make_leaf_item(cat_dft, f"⚛️ {spec.name}", payload, checked)

        # Odświeżenie stanów pośrednich rodziców od dołu do góry
        self._refresh_all_parent_states(self.tree.invisibleRootItem())
        self._is_updating_checks = False

    def _refresh_all_parent_states(self, parent: QTreeWidgetItem):
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.childCount() > 0:
                self._refresh_all_parent_states(child)
                self._update_parent_check_indicator(child)

    def _update_parent_check_indicator(self, parent: QTreeWidgetItem):
        total = parent.childCount()
        if total == 0:
            return
        checked = sum(1 for i in range(total) if parent.child(i).checkState(0) == Qt.Checked)
        partial = sum(1 for i in range(total) if parent.child(i).checkState(0) == Qt.PartiallyChecked)
        if checked == total:
            parent.setCheckState(0, Qt.Checked)
        elif checked == 0 and partial == 0:
            parent.setCheckState(0, Qt.Unchecked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)

    def _propagate_check_down(self, parent: QTreeWidgetItem, state: Qt.CheckState, only_visible: bool = False):
        for i in range(parent.childCount()):
            child = parent.child(i)
            if only_visible and child.isHidden():
                continue
            if child.childCount() > 0:
                self._propagate_check_down(child, state, only_visible=only_visible)
                self._update_parent_check_indicator(child)
            else:
                child.setCheckState(0, state)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._is_updating_checks or column != 0:
            return

        self._is_updating_checks = True
        state = item.checkState(0)

        if item.childCount() > 0:
            target_state = Qt.Checked if state != Qt.Unchecked else Qt.Unchecked
            item.setCheckState(0, target_state)
            self._propagate_check_down(item, target_state, only_visible=True)

        p = item.parent()
        while p and p != self.tree.invisibleRootItem():
            self._update_parent_check_indicator(p)
            p = p.parent()

        self._is_updating_checks = False
        self._update_selected_counter()

    def _apply_filter(self, text: str):
        raw = text.strip().lower()
        # Obsługa wielu fraz rozdzielonych przecinkiem lub średnikiem
        tokens = [t.strip() for t in re.split(r'[,;]+', raw) if t.strip()]

        def filter_node(node: QTreeWidgetItem) -> bool:
            if node.childCount() == 0:
                if not tokens:
                    node.setHidden(False)
                    return True
                payload = node.data(0, Qt.UserRole) or {}
                s_name = str(payload.get('name', node.text(0))).lower()
                matched = any(
                    (tok in s_name) or fnmatch.fnmatch(s_name, tok)
                    for tok in tokens
                )
                node.setHidden(not matched)
                return matched
            else:
                any_child_visible = False
                for i in range(node.childCount()):
                    if filter_node(node.child(i)):
                        any_child_visible = True
                node.setHidden(not any_child_visible)
                if any_child_visible and tokens:
                    node.setExpanded(True)
                return any_child_visible

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            filter_node(root.child(i))

    def _iter_leaf_items(self, parent: Optional[QTreeWidgetItem] = None):
        if parent is None:
            parent = self.tree.invisibleRootItem()
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.childCount() == 0:
                yield child
            else:
                yield from self._iter_leaf_items(child)

    def _check_all_visible_leaves(self):
        self._is_updating_checks = True
        for leaf in self._iter_leaf_items():
            if not leaf.isHidden():
                leaf.setCheckState(0, Qt.Checked)
        self._refresh_all_parent_states(self.tree.invisibleRootItem())
        self._is_updating_checks = False
        self._update_selected_counter()

    def _isolate_visible_leaves(self):
        self._is_updating_checks = True
        for leaf in self._iter_leaf_items():
            leaf.setCheckState(0, Qt.Checked if not leaf.isHidden() else Qt.Unchecked)
        self._refresh_all_parent_states(self.tree.invisibleRootItem())
        self._is_updating_checks = False
        self._update_selected_counter()

    def _uncheck_all(self):
        self._is_updating_checks = True
        for leaf in self._iter_leaf_items():
            leaf.setCheckState(0, Qt.Unchecked)
        self._refresh_all_parent_states(self.tree.invisibleRootItem())
        self._is_updating_checks = False
        self._update_selected_counter()

    def _update_selected_counter(self):
        cnt = sum(1 for leaf in self._iter_leaf_items() if leaf.checkState(0) == Qt.Checked)
        self.lbl_count.setText(f"Wybrano: {cnt} widm")

    def _save_and_accept(self):
        selected_items = []
        for leaf in self._iter_leaf_items():
            if leaf.checkState(0) == Qt.Checked:
                payload = leaf.data(0, Qt.UserRole)
                if payload:
                    selected_items.append(dict(payload))

        self.step['selected_items'] = selected_items
        self.step['interpolate_x'] = self.chk_interp.isChecked()
        # Czyścimy przestarzałe klucze jednowymiarowe, aby nie nadpisywały nowej struktury
        self.step['selected_spectra'] = [it['name'] for it in selected_items]
        self.step['selected_spectrum'] = selected_items[0]['name'] if selected_items else ""

        if not selected_items:
            self.step['name'] = "Projekt: (brak wyboru)"
        elif len(selected_items) == 1:
            one = selected_items[0]
            icon = "🖼️" if one.get('scope') == 'window' else "📈"
            short_n = one['name'][:22] + "…" if len(one['name']) > 22 else one['name']
            self.step['name'] = f"{icon} {short_n}"
        else:
            win_names = {it.get('window_name') for it in selected_items if it.get('scope') == 'window' and it.get('window_name')}
            if len(win_names) == 1 and all(it.get('scope') == 'window' for it in selected_items):
                w_n = next(iter(win_names))
                short_w = w_n[:14] + "…" if len(w_n) > 14 else w_n
                self.step['name'] = f"🖼️ {short_w} ({len(selected_items)})"
            else:
                self.step['name'] = f"🗂️ Wybrano: {len(selected_items)} widm"

        self.accept()


@register_node
class SpectrumSelectNode(BaseNode):
    node_type = "spectrum_select"
    title = "🗂️ Wybór z Projektu (Drzewo)"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#0f4c5c", "#e36414")

    def __init__(self, step_data: Optional[Dict[str, Any]] = None):
        super().__init__(step_data)
        self.validation_error: Optional[str] = None
        self.last_count: int = 0

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Projekt: (wybierz)",
            "selected_items": [],
            "selected_spectra": [],
            "interpolate_x": False
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#48cae4"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return SpectrumSelectNodeDialog(self, parent=parent)

    def _resolve_selected_spectra(self) -> List[SpectrumData]:
        items = self.step_data.get('selected_items', [])
        resolved: List[SpectrumData] = []

        if items:
            for entry in items:
                scope = entry.get('scope', 'storage')
                sname = entry.get('name', '')
                if not sname:
                    continue

                spec: Optional[SpectrumData] = None
                if scope == 'window':
                    wid = entry.get('window_id', '')
                    win = session.windows.get(wid)
                    if not win and entry.get('window_name'):
                        w_target = entry['window_name'].lower()
                        for w in session.windows.values():
                            if w.name.lower() == w_target:
                                win = w
                                break
                    if win and sname in win.spectra:
                        spec = win.spectra[sname]
                else:
                    spec = session.get_spectrum(sname)

                if spec is not None:
                    resolved.append(spec)
        else:
            # Kompatybilność wsteczna ze starymi projektami/szablonami
            legacy = self.step_data.get('selected_spectra', [])
            if not legacy and self.step_data.get('selected_spectrum'):
                legacy = [self.step_data['selected_spectrum']]
            for sname in legacy:
                spec = session.get_spectrum(sname)
                if spec is not None:
                    resolved.append(spec)

        return resolved

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        valid_specs = self._resolve_selected_spectra()

        if not valid_specs:
            self.validation_error = None
            self.last_count = 0
            self.has_preview = True
            def_x = np.linspace(4000, 400, 1000)
            return {"out": (def_x, np.zeros_like(def_x))}, {"out": ["Ref"]}

        interp_x = bool(self.step_data.get('interpolate_x', False))
        ref_x = valid_specs[0].wavenumbers.copy()
        cols: List[np.ndarray] = []
        names: List[str] = []

        for s in valid_specs:
            sx, sy = s.wavenumbers, s.intensities
            if interp_x:
                sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
                cols.append(np.interp(ref_x, sx_s, sy_s))
                names.append(s.name)
            else:
                if len(sx) != len(ref_x):
                    self.validation_error = f"Różna liczba punktów:\n{len(ref_x)} ≠ {len(sx)} ({s.name})"
                    self.last_count = 0
                    self.has_preview = False
                    return {"out": (None, None)}, {"out": []}
                if (ref_x[0] > ref_x[-1]) != (sx[0] > sx[-1]) or not np.allclose(ref_x, sx, rtol=0.0, atol=1e-3):
                    self.validation_error = f"Niezgodna oś X w:\n{s.name}"
                    self.last_count = 0
                    self.has_preview = False
                    return {"out": (None, None)}, {"out": []}
                cols.append(sy.copy())
                names.append(s.name)

        self.validation_error = None
        self.has_preview = True
        self.last_count = len(cols)
        y_out = np.column_stack(cols) if len(cols) > 1 else cols[0]
        return {"out": (ref_x, y_out)}, {"out": names}

    def draw_custom_overlay(self, painter: QPainter, box_rect: QRectF):
        if self.validation_error:
            painter.save()
            painter.setBrush(QBrush(QColor(150, 25, 35, 70)))
            painter.setPen(QPen(QColor("#ff4d6d"), 1.2))
            painter.drawRoundedRect(box_rect, 4, 4)

            title_rect = QRectF(box_rect.left() + 6, box_rect.top() + 8, box_rect.width() - 12, 18)
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ff758f"))
            painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignVCenter, "⚠️ Niezgodne widma")

            msg_rect = QRectF(box_rect.left() + 6, box_rect.top() + 28, box_rect.width() - 12, box_rect.height() - 34)
            painter.setFont(QFont("Arial", 7))
            painter.setPen(QColor("#ffe5ec"))
            painter.drawText(msg_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.validation_error)
            painter.restore()
        elif self.last_count > 1:
            painter.save()
            badge_rect = QRectF(box_rect.right() - 46, box_rect.top() + 3, 42, 14)
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#8ecae6"))
            painter.drawText(badge_rect, Qt.AlignRight | Qt.AlignVCenter, f"n = {self.last_count}")
            painter.restore()


# =============================================================================
# 3. DEDYKOWANY WĘZEŁ WZORCÓW ATMOSFERY: AtmSelectNode ('atm_select')
# =============================================================================

class AtmSelectNodeDialog(QDialog):
    """Konfiguracja źródła wzorców pary wodnej i CO2 z dedykowanej bazy atmosfery."""

    def __init__(self, node: 'AtmSelectNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Wzorce Atmosfery (ATM / CO₂)")
        self.resize(450, 330)

        layout = QVBoxLayout(self)

        self.radio_all_visible = QRadioButton("Wszystkie widoczne wzorce z bazy ATM (automatycznie)")
        self.radio_selected = QRadioButton("Tylko wybrane wzorce z listy poniżej:")

        grp = QButtonGroup(self)
        grp.addButton(self.radio_all_visible)
        grp.addButton(self.radio_selected)

        mode = self.step.get('atm_mode', 'all_visible')
        if mode == 'selected':
            self.radio_selected.setChecked(True)
        else:
            self.radio_all_visible.setChecked(True)

        layout.addWidget(self.radio_all_visible)
        layout.addWidget(self.radio_selected)

        self.list_atm = QListWidget()
        self.list_atm.setFixedHeight(170)
        current_selected = set(self.step.get('selected_atm', []))

        for name in sorted(session.atm_spectra.keys(), key=_natural_sort_key):
            item = QListWidgetItem(f"☁️ {name}")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in current_selected else Qt.Unchecked)
            self.list_atm.addItem(item)

        layout.addWidget(self.list_atm)

        row_btns = QHBoxLayout()
        btn_all = QPushButton("Zaznacz wszystkie")
        btn_all.clicked.connect(lambda: self._set_all_checks(Qt.Checked))
        btn_none = QPushButton("Odznacz wszystkie")
        btn_none.clicked.connect(lambda: self._set_all_checks(Qt.Unchecked))
        row_btns.addWidget(btn_all)
        row_btns.addWidget(btn_none)
        layout.addLayout(row_btns)

        def _toggle_list():
            enabled = self.radio_selected.isChecked()
            self.list_atm.setEnabled(enabled)
            btn_all.setEnabled(enabled)
            btn_none.setEnabled(enabled)

        self.radio_all_visible.toggled.connect(_toggle_list)
        _toggle_list()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _set_all_checks(self, state: Qt.CheckState):
        for i in range(self.list_atm.count()):
            self.list_atm.item(i).setCheckState(state)

    def _save_and_accept(self):
        if self.radio_selected.isChecked():
            selected = []
            for i in range(self.list_atm.count()):
                it = self.list_atm.item(i)
                if it.checkState() == Qt.Checked:
                    selected.append(it.data(Qt.UserRole))
            self.step['atm_mode'] = 'selected'
            self.step['selected_atm'] = selected
            self.step['name'] = f"☁️ ATM ({len(selected)} wybranych)"
        else:
            self.step['atm_mode'] = 'all_visible'
            self.step['name'] = "☁️ ATM (wszystkie widoczne)"
        self.accept()


@register_node
class AtmSelectNode(BaseNode):
    node_type = "atm_select"
    title = "☁️ Wzorce Atmosfery (ATM / CO₂)"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#005f73", "#0a9396")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "☁️ ATM (wszystkie widoczne)",
            "atm_mode": "all_visible",
            "selected_atm": []
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Atm", is_output=True, color=QColor("#94d2bd"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return AtmSelectNodeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        mode = self.step_data.get('atm_mode', 'all_visible')

        if mode == 'selected':
            names_req = self.step_data.get('selected_atm', [])
            specs = [session.atm_spectra[n] for n in names_req if n in session.atm_spectra]
        else:
            specs = session.get_visible_atm_spectra()

        if specs:
            ref_x = specs[0].wavenumbers.copy()
            cols, names = [], []
            for s in specs:
                sx, sy = s.wavenumbers, s.intensities
                sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
                cols.append(np.interp(ref_x, sx_s, sy_s))
                names.append(s.name)
            y_out = np.column_stack(cols) if len(cols) > 1 else cols[0]
            return {"out": (ref_x, y_out)}, {"out": names}

        def_x = np.linspace(4000, 400, 1000)
        return {"out": (def_x, np.zeros_like(def_x))}, {"out": ["Brak ATM"]}


# =============================================================================
# 4. ZBIERACZ / MUX: MergeSpectraNode ('merge_spectra')
# =============================================================================

@register_node
class MergeSpectraNode(BaseNode):
    node_type = "merge_spectra"
    title = "🔀 Zbieracz / Mux (Merge 2D)"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#028090", "#00a896")

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), allow_multiple=True, pos_y=90),
            PortDef(port_id="out", label="Mat", is_output=True, color=QColor("#06d6a0"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Zbieracz / Mux")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            "Węzeł Zbieracza (Mux) łączy sygnały z wielu gałęzi i nodów\n"
            "w jedną wspólną macierz dwuwymiarową (W x N).\n\n"
            "Podłącz kable z dowolnych węzłów pod port wejściowy 'In'."
        )
        lbl.setStyleSheet("color: #a8dadc; font-size: 12px; padding: 10px;")
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        return dlg

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sources = inputs.get("in_multi", [])
        source_names = input_names.get("in_multi", [])

        if not sources:
            d_in = inputs.get("in", (None, None))
            if d_in[0] is not None:
                sources = [d_in]
                source_names = [input_names.get("in", [])]

        if not sources or sources[0][0] is None:
            return {"out": (None, None)}, {"out": []}

        ref_x = sources[0][0]
        cols, collected_names = [], []

        for idx, (sx, sy) in enumerate(sources):
            if sx is None or sy is None:
                continue
            names = source_names[idx] if idx < len(source_names) else []
            sx_s = sx[::-1] if sx[0] > sx[-1] else sx

            if sy.ndim == 2:
                for c in range(sy.shape[1]):
                    col_data = sy[::-1, c] if sx[0] > sx[-1] else sy[:, c]
                    cols.append(np.interp(ref_x, sx_s, col_data))
                    collected_names.append(names[c] if c < len(names) else f"Widmo_{len(collected_names) + 1}")
            else:
                col_data = sy[::-1] if sx[0] > sx[-1] else sy
                cols.append(np.interp(ref_x, sx_s, col_data))
                collected_names.append(names[0] if names else f"Widmo_{len(collected_names) + 1}")

        if cols:
            return {"out": (ref_x.copy(), np.column_stack(cols))}, {"out": collected_names}

        return {"out": (None, None)}, {"out": []}
