"""
gui/project_tree.py
Drzewo obiektów projektu SpectroHub.
Obsługuje:
- Dwukolumnowy widok z klikalnym próbnikiem barw (OPUS Style)
- Bogate toolpity spektroskopowe z parametrami fizykochemicznymi
- Skróty klawiszowe (Spacja = widoczność, Del = usuwanie, F2 = zmiana nazwy)
- Naturalne sortowanie alfanumeryczne serii (1, 2, ..., 10 zamiast 1, 10, 2)
- Narzędzia szybkiego przepływu: wysyłanie do nowego okna, duplikacja okna, selekcja wg wzorca
"""

import os
import re
import fnmatch
import logging
from typing import List, Tuple, Optional
from PySide6.QtCore import Qt, QThreadPool, QObject, Signal, QRunnable, Slot
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QAbstractItemView,
    QInputDialog, QFileDialog, QMessageBox, QApplication,
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QLabel, QColorDialog
)
import numpy as np

from core.session import session
from core.models import GaussianData, SpectrumData, WorkspaceWindow, DataTable
from io_hub.dispatcher import SpectrumImportDispatcher
from io_hub.export_manager import ExportManager
from core.i18n import tr

logger = logging.getLogger("SpectroHub")

# Laboratoryjna paleta wysokiego kontrastu dopasowana do ciemnego motywu
SPECTRAL_PALETTE = [
    "#4cc9f0",  # 1. Jasny cyjan
    "#f72585",  # 2. Karmazyn / magenta
    "#4ade80",  # 3. Mięta / zieleń
    "#fb8500",  # 4. Pomarańcz
    "#d946ef",  # 5. Fuksja
    "#ffb703",  # 6. Bursztynowy złoty
    "#06d6a0",  # 7. Turkus
    "#ff4d6d",  # 8. Koralowa czerwień
    "#70d6ff",  # 9. Błękit
    "#ffd166",  # 10. Żółty
    "#a78bfa",  # 11. Lawenda
    "#80ed99",  # 12. Limonka
]

SUPPORTED_EXTENSIONS = {
    # Małe litery
    '.spc', '.spa', '.spg', '.dx', '.jdx', '.opus',
    '.csv', '.dpt', '.txt', '.tsv', '.dat',
    '.log', '.out', '.fchk',
    # Wielkie litery (macOS / Linux case-sensitivity)
    '.SPC', '.SPA', '.SPG', '.DX', '.JDX', '.OPUS',
    '.CSV', '.DPT', '.TXT', '.TSV', '.DAT',
    '.LOG', '.OUT', '.FCHK'
}

def natural_sort_key(s: str) -> list:
    """Klucz sortowania naturalnego (np. widmo_2 przed widmo_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def scan_directory_for_spectra(dir_path: str) -> List[str]:
    """Skanuje folder i zwraca posortowane naturalnie ścieżki do obsługiwanych plików widm."""
    found_paths = []
    try:
        for entry in os.listdir(dir_path):
            if entry.startswith('.'):
                continue
            full_path = os.path.join(dir_path, entry)
            if os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in SUPPORTED_EXTENSIONS or re.match(r'^\.\d+$', ext):
                    found_paths.append(full_path)
    except Exception as e:
        logger.error(f"Błąd skanowania katalogu {dir_path}: {e}")

    return sorted(found_paths, key=natural_sort_key)

def make_color_swatch_icon(color_hex: str) -> QIcon:
    """Tworzy zaokrąglony kafelek koloru z dyskretną ramką dla kolumny 1 drzewka."""
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color_hex))
    painter.setPen(QPen(QColor("#666677"), 1))
    painter.drawRoundedRect(1, 1, 12, 12, 3, 3)
    painter.end()

    return QIcon(pixmap)


def make_spectrum_tooltip(spec: SpectrumData) -> str:
    """Generuje bogaty tooltip z kluczowymi parametrami spektroskopowymi widma."""
    try:
        n_pts = len(spec.wavenumbers)
        x_min = float(np.min(spec.wavenumbers))
        x_max = float(np.max(spec.wavenumbers))
        y_min = float(np.min(spec.intensities))
        y_max = float(np.max(spec.intensities))
        max_idx = int(np.argmax(spec.intensities))
        peak_pos = float(spec.wavenumbers[max_idx])

        time_info = spec.time_str if getattr(spec, 'time_str', None) and spec.time_str != "00:00:00" else ""
        time_row = f"<b>Czas rejestracji:</b> {time_info}<br>" if time_info else ""

        return (
            f"<div style='font-size: 11px; line-height: 1.3;'>"
            f"<b style='color: #4cc9f0; font-size: 12px;'>{spec.name}</b><br>"
            f"<hr style='border: 0; border-top: 1px solid #4a4a5a; margin: 4px 0;'>"
            f"{time_row}"
            f"<b>Liczba punktów:</b> {n_pts}<br>"
            f"<b>Zakres liczb falowych:</b> {x_max:.1f} — {x_min:.1f} cm⁻¹<br>"
            f"<b>Pik główny:</b> A = {y_max:.4f} (@ {peak_pos:.1f} cm⁻¹)<br>"
            f"<b>Zakres absorbancji:</b> [{y_min:.4f}, {y_max:.4f}]"
            f"</div>"
        )
    except Exception:
        return spec.name


def resolve_spectrum_color(spec: SpectrumData, index_in_group: int) -> str:
    """Zwraca zdefiniowany custom_color lub deterministyczny kolor z palety i utrwala go w metadanych."""
    color = spec.metadata.get("custom_color")
    if not color:
        color = spec.metadata.get("display_color")
    if not color:
        color = SPECTRAL_PALETTE[index_in_group % len(SPECTRAL_PALETTE)]
    spec.metadata["display_color"] = color
    return color


class TableViewerDialog(QDialog):
    """Eleganckie okno inspekcji i kopiowania danych tabelarycznych."""

    def __init__(self, table: DataTable, parent=None):
        super().__init__(parent)
        self.table = table
        self.setWindowTitle(f"Tabela danych: {table.name}")
        self.resize(680, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        row_labels = table.metadata.get("row_labels", [])
        has_custom_labels = bool(row_labels and len(row_labels) == table.n_rows)

        lbl_info = QLabel(f"📊 <b>{table.name}</b> &nbsp;|&nbsp; Wiersze: {table.n_rows}, Kolumny: {table.n_cols}")
        lbl_info.setStyleSheet("color: #48cae4; font-size: 12px;")
        layout.addWidget(lbl_info)

        col_keys = list(table.y_columns.keys())
        first_header = "Parametr / Cecha" if has_custom_labels else table.x_label
        headers = [first_header] + col_keys

        self.table_widget = QTableWidget(table.n_rows, len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col_idx in range(1, len(headers)):
            self.table_widget.horizontalHeader().setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a24;
                alternate-background-color: #22222f;
                gridline-color: #3d3d4e;
                color: #e0e0e0;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #2b2b3d;
                color: #a8dadc;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #3d3d4e;
            }
        """)

        for row_idx in range(table.n_rows):
            lbl_val = str(row_labels[row_idx]) if has_custom_labels else f"{table.x_values[row_idx]:.6g}"
            item_lbl = QTableWidgetItem(lbl_val)
            item_lbl.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table_widget.setItem(row_idx, 0, item_lbl)

            for col_i, c_name in enumerate(col_keys):
                val = table.y_columns[c_name][row_idx] if row_idx < len(table.y_columns[c_name]) else 0.0
                item_val = QTableWidgetItem(f"{val:.6g}")
                item_val.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item_val.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table_widget.setItem(row_idx, col_i + 1, item_val)

        layout.addWidget(self.table_widget)

        btns_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Kopiuj dane (TSV pod Origin / Excel)")
        btn_copy.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 6px 14px;")
        btn_copy.clicked.connect(self._copy_tsv)
        btns_row.addWidget(btn_copy)

        btn_csv = QPushButton("💾 Zapisz CSV...")
        btn_csv.clicked.connect(self._save_csv)
        btns_row.addWidget(btn_csv)

        btns_row.addStretch()

        btn_close = QPushButton("Zamknij")
        btn_close.clicked.connect(self.accept)
        btns_row.addWidget(btn_close)

        layout.addLayout(btns_row)

    def _copy_tsv(self):
        tsv_data = self.table.to_tsv(decimal_comma=True)
        QApplication.clipboard().setText(tsv_data)
        QMessageBox.information(self, "Schowek", "Pomyślnie skopiowano dane tabeli w formacie TSV!")

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz tabelę jako CSV", f"{ExportManager.sanitize_filename(self.table.name)}.csv", "Plik CSV (*.csv)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.table.to_tsv(decimal_comma=True).replace("\t", ";"))
                QMessageBox.information(self, "Zapisano", f"Pomyślnie zapisano tabelę do pliku:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Błąd zapisu tabeli:\n{e}")


class ImportWorkerSignals(QObject):
    """Sygnały bezpiecznie łączące wątek poboczny z pętlą zdarzeń GUI."""
    progress = Signal(int, int)               # (bieżący_plik, suma)
    batch_loaded = Signal(list, bool)         # (lista_SpectrumData, is_atm)
    error_occurred = Signal(str)              # komunikat o błędach


class FileImportWorker(QRunnable):
    """Zadanie w tle: wczytuje pliki i bezpiecznie przekazuje je do wątku głównego."""

    def __init__(self, file_paths: list, target_atm: bool = False):
        super().__init__()
        # Zabezpieczenie przed losową kolejnością z okna dialogowego
        self.file_paths = sorted(file_paths, key=natural_sort_key)
        self.target_atm = target_atm
        self.signals = ImportWorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        all_loaded = []
        errors = []
        total = len(self.file_paths)

        for idx, path in enumerate(self.file_paths, 1):
            try:
                spectra = SpectrumImportDispatcher.load_file(path)
                if isinstance(spectra, list):
                    all_loaded.extend(spectra)
                elif spectra:
                    all_loaded.append(spectra)
            except Exception as e:
                err_msg = f"{os.path.basename(path)}: {e}"
                logger.error(f"Nie udało się zaimportować pliku {path}: {e}")
                errors.append(err_msg)

            if idx % 25 == 0 or idx == total:
                self.signals.progress.emit(idx, total)

        # 1. Przekazanie załadowanych widm do slotu w wątku głównym
        if all_loaded:
            self.signals.batch_loaded.emit(all_loaded, self.target_atm)

        # 2. Raport o ewentualnych pominiętych plikach
        if errors:
            summary = "\n".join(errors[:10])
            if len(errors) > 10:
                summary += f"\n... i {len(errors) - 10} innych."
            self.signals.error_occurred.emit(summary)

class ProjectTreeWidget(QTreeWidget):
    """Eksplorator z dwukolumnowym widokiem, próbnikiem barw (OPUS Style) oraz pełną kontrolą selekcji."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.setColumnWidth(1, 28)

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
        self.root_storage = QTreeWidgetItem(self, ["📦 MAGAZYN DANYCH (Referencja)", ""])
        self.root_storage.setExpanded(True)
        self.root_storage.setData(0, Qt.UserRole + 1, "section_header_storage")

        self.root_exp = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_exp", "Pomiary Eksperymentalne"), ""])
        self.root_atm = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_atm", "☁️ Wzorce Atmosfery (ATM / CO₂)"), ""])
        self.root_series = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_series", "Serie Ewolucyjne (2D)"), ""])
        self.root_dft = QTreeWidgetItem(self.root_storage, [tr("project_tree.group_dft", "Obliczenia Kwantowe (DFT)"), ""])

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
        self.root_workspaces = QTreeWidgetItem(self, ["🖥️ OKNA WYKRESÓW / ROBOCZE", ""])
        self.root_workspaces.setExpanded(True)
        self.root_workspaces.setData(0, Qt.UserRole + 1, "section_header")

        for f_id, folder in session.folders.items():
            self._create_folder_item(f_id, folder.name, folder.parent_folder_id)

        for win_id, win in session.windows.items():
            self._create_window_item(win_id, win.name, win.folder_id)

        for t_id, tbl in session.tables.items():
            self._create_table_item(t_id, tbl.name, tbl.folder_id)

    def _connect_signals(self):
        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_selection_changed)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
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

        session.folder_added.connect(self._on_folder_added)
        session.folder_removed.connect(self._on_folder_removed)
        session.folder_renamed.connect(self._on_folder_renamed)
        session.window_folder_changed.connect(self._on_window_folder_changed)

        session.window_added.connect(self._on_window_added)
        session.window_removed.connect(self._on_window_removed)
        session.window_spectrum_added.connect(self._on_window_spectrum_added)
        session.window_spectra_batch_added.connect(self._on_window_spectra_batch_added)
        session.window_spectrum_removed.connect(self._on_window_spectrum_removed)

        session.table_added.connect(self._on_table_added)
        session.table_removed.connect(self._on_table_removed)
        session.table_renamed.connect(self._on_table_renamed)
        session.table_folder_changed.connect(self._on_table_folder_changed)

    # --- SKRÓTY KLAWISZOWE ---
    def keyPressEvent(self, event):
        """Obsługa laboratoryjnych skrótów klawiszowych: Spacja, Delete, F2."""
        if event.key() == Qt.Key_Space:
            selected = self.selectedItems()
            checkable = [it for it in selected if it.flags() & Qt.ItemFlag.ItemIsUserCheckable]
            if checkable:
                all_checked = all(it.checkState(0) == Qt.CheckState.Checked for it in checkable)
                target = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
                self._set_selected_items_check_state(checkable, target)
                event.accept()
                return

        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._handle_delete_key()
            event.accept()
            return

        elif event.key() == Qt.Key_F2:
            self._handle_f2_rename()
            event.accept()
            return

        super().keyPressEvent(event)

    def _handle_delete_key(self):
        selected = self.selectedItems()
        if not selected:
            return

        first = selected[0]
        itype = first.data(0, Qt.UserRole + 1)
        name = first.data(0, Qt.UserRole)

        # 1. Usuwanie widm z Okna Roboczego (błyskawiczne, bez pytania)
        if itype == "window_spectrum":
            win_id = first.data(0, Qt.UserRole + 2)
            names = [it.data(0, Qt.UserRole) for it in selected if it.data(0, Qt.UserRole + 1) == "window_spectrum"]
            for n in names:
                session.remove_spectrum_from_window(win_id, n)
            return

        # 2. Usuwanie okna roboczego
        if itype == "workspace_window":
            if QMessageBox.question(self, "Usuń okno", f"Czy na pewno usunąć okno '{name}'?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                session.remove_window(name)
            return

        # 3. Usuwanie folderu
        if itype == "window_folder":
            reply = QMessageBox.question(self, "Usuń folder", "Czy zachować okna robocze z tego folderu?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                session.remove_folder(name, delete_windows=False)
            elif reply == QMessageBox.No:
                session.remove_folder(name, delete_windows=True)
            return

        # 4. Usuwanie tabeli
        if itype == "data_table":
            if QMessageBox.question(self, "Usuń tabelę", f"Czy na pewno usunąć tabelę '{name}'?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                session.remove_table(name)
            return

        # 5. Bezpieczne usuwanie widm z Magazynu Referencyjnego
        if itype in ["spectrum", "dft", "atm"]:
            names = [it.data(0, Qt.UserRole) for it in selected if it.data(0, Qt.UserRole + 1) in ["spectrum", "dft", "atm"]]
            msg = f"Czy na pewno trwale usunąć {len(names)} widm z Magazynu Danych?" if len(names) > 1 else f"Czy na pewno trwale usunąć '{names[0]}' z Magazynu Danych?"
            if QMessageBox.question(self, "Usuń z magazynu", msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                for n in names:
                    session.remove_spectrum(n)

    def _handle_f2_rename(self):
        item = self.currentItem()
        if not item:
            return

        itype = item.data(0, Qt.UserRole + 1)
        name = item.data(0, Qt.UserRole)

        if itype == "workspace_window":
            self._prompt_rename_window(name)
        elif itype == "window_folder":
            self._prompt_rename_folder(name)
        elif itype == "data_table":
            self._prompt_rename_table(name)

    def _on_session_cleared(self):
        self._is_internal_update = True
        for cat in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            cat.takeChildren()
        self.root_workspaces.takeChildren()
        self._is_internal_update = False

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        if not item:
            return
        itype = item.data(0, Qt.UserRole + 1)
        iname = item.data(0, Qt.UserRole)
        if itype == "data_table" and iname:
            tbl = session.get_table(iname)
            if tbl:
                dlg = TableViewerDialog(tbl, parent=self)
                dlg.exec()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Kliknięcie w próbnik w kolumnie 1 otwiera paletę kolorów."""
        if not item:
            return

        if column == 1:
            itype = item.data(0, Qt.UserRole + 1)
            iname = item.data(0, Qt.UserRole)
            if itype in ["spectrum", "dft", "atm", "window_spectrum"]:
                self._open_color_picker_for_item(item, iname, itype)
                return

        self._on_selection_changed(item)

    def _open_color_picker_for_item(self, item: QTreeWidgetItem, name: str, itype: str):
        """Otwiera okno wyboru koloru i odświeża widok na żywo bez zbędnych sygnałów widoczności."""
        if itype == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            win = session.windows.get(win_id)
            spec = win.spectra.get(name) if win else None
        else:
            spec = session.get_spectrum(name)

        if not spec:
            return

        curr_hex = spec.metadata.get("custom_color", spec.metadata.get("display_color", "#4cc9f0"))
        chosen_color = QColorDialog.getColor(QColor(curr_hex), self, f"Wybierz kolor widma: {name}")

        if chosen_color.isValid():
            new_hex = chosen_color.name()
            spec.metadata["custom_color"] = new_hex
            spec.metadata["display_color"] = new_hex

            item.setIcon(1, make_color_swatch_icon(new_hex))

            # Bezpośrednie i bezpieczne przerysowanie wykresu
            main_win = self.window()
            if hasattr(main_win, "canvas_2d"):
                main_win.canvas_2d.redraw_all()
            else:
                session.context_changed.emit(session.active_context)

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

    def _prompt_import_directory(self, target_atm: bool = False):
        """Otwiera okno wyboru folderu i wczytuje z niego wszystkie widma w tle."""
        title = "Wybierz folder z wzorcami atmosfery..." if target_atm else "Wybierz folder z widmami..."
        dir_path = QFileDialog.getExistingDirectory(self, title, "")
        if not dir_path:
            return

        paths = scan_directory_for_spectra(dir_path)
        if not paths:
            QMessageBox.information(self, "Brak plików",
                                    "W wybranym folderze nie znaleziono obsługiwanych plików widm.")
            return

        worker = FileImportWorker(paths, target_atm=target_atm)
        worker.signals.batch_loaded.connect(self._on_import_batch_received)
        worker.signals.error_occurred.connect(
            lambda errs: QMessageBox.warning(self, "Błędy importu", f"Pominięto niektóre pliki:\n{errs}")
        )
        self.thread_pool.start(worker)

    def _create_folder_item(self, folder_id: str, name: str, parent_folder_id: Optional[str] = None) -> QTreeWidgetItem:
        parent_item = self.root_workspaces
        if parent_folder_id:
            found_p = self._find_folder_item(parent_folder_id)
            if found_p:
                parent_item = found_p

        folder_item = QTreeWidgetItem(parent_item, [f"📁 {name}", ""])
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

        win_item = QTreeWidgetItem(parent_item, [f"🖼️ {name}", ""])
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
            idx = win_node.childCount()
            spec_color = resolve_spectrum_color(spec, idx)

            item = QTreeWidgetItem(win_node, [f"📈 {spec.name}", ""])
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

            item.setIcon(1, make_color_swatch_icon(spec_color))
            item.setToolTip(1, "Kliknij, aby zmienić kolor widma (OPUS style)")
            item.setToolTip(0, make_spectrum_tooltip(spec))

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

    def _on_window_spectra_batch_added(self, window_id: str, spectra_list: list):
        win_node = self._find_window_item(window_id)
        if not win_node:
            return

        self._is_internal_update = True
        self.setUpdatesEnabled(False)
        items_to_add = []
        base_idx = win_node.childCount()

        # Naturalne sortowanie numeryczne
        sorted_specs = sorted(spectra_list, key=lambda s: natural_sort_key(s.name))

        for i, spec in enumerate(sorted_specs):
            idx = base_idx + i
            spec_color = resolve_spectrum_color(spec, idx)

            item = QTreeWidgetItem([f"📈 {spec.name}", ""])
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

            item.setIcon(1, make_color_swatch_icon(spec_color))
            item.setToolTip(1, "Kliknij, aby zmienić kolor widma (OPUS style)")
            item.setToolTip(0, make_spectrum_tooltip(spec))

            items_to_add.append(item)

        win_node.addChildren(items_to_add)
        win_node.setExpanded(True)
        self._refresh_parent_indicator(win_node)
        self.setUpdatesEnabled(True)
        self._is_internal_update = False

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

        if first_type == "data_table":
            if target_type == "window_folder":
                session.move_table_to_folder(first_id, target_id)
                event.acceptProposedAction()
                return
            elif target_type == "section_header" and target_item == self.root_workspaces:
                session.move_table_to_folder(first_id, None)
                event.acceptProposedAction()
                return

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
        self.root_workspaces.takeChildren()
        for f_id, folder in session.folders.items():
            self._create_folder_item(f_id, folder.name, folder.parent_folder_id)
        for win_id, win in session.windows.items():
            win_node = self._create_window_item(win_id, win.name, win.folder_id)
            for s_idx, (s_name, spec) in enumerate(win.spectra.items()):
                spec_color = resolve_spectrum_color(spec, s_idx)
                item = QTreeWidgetItem(win_node, [f"📈 {spec.name}", ""])
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
                item.setCheckState(0, Qt.CheckState.Checked if win.visibility.get(s_name, True) else Qt.CheckState.Unchecked)
                item.setData(0, Qt.UserRole, spec.name)
                item.setData(0, Qt.UserRole + 1, "window_spectrum")
                item.setData(0, Qt.UserRole + 2, win_id)
                item.setIcon(1, make_color_swatch_icon(spec_color))
                item.setToolTip(1, "Kliknij, aby zmienić kolor widma (OPUS style)")
                item.setToolTip(0, make_spectrum_tooltip(spec))

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

        item = QTreeWidgetItem(target_root, [f"{icon} {spectrum.name}", ""])
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        item.setCheckState(0, Qt.CheckState.Checked)
        item.setData(0, Qt.UserRole, spectrum.name)
        item.setData(0, Qt.UserRole + 1, item_type)

        idx = target_root.childCount() - 1
        spec_color = resolve_spectrum_color(spectrum, idx)

        item.setIcon(1, make_color_swatch_icon(spec_color))
        item.setToolTip(1, "Kliknij, aby zmienić kolor widma (OPUS style)")
        item.setToolTip(0, make_spectrum_tooltip(spectrum))

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

        # Naturalne sortowanie numeryczne
        sorted_specs = sorted(spectra_list, key=lambda s: natural_sort_key(s.name))

        for spec in sorted_specs:
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
        sorted_specs = sorted(spectra_list, key=lambda s: natural_sort_key(s.name))
        for spec in sorted_specs:
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

        series_item = QTreeWidgetItem(self.root_series, [label, ""])
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

    # --- METODY LOGICZNE ZARZĄDZANIA ZAZNACZENIEM ---
    def _set_category_check_state(self, cat_item: QTreeWidgetItem, state: Qt.CheckState):
        """Ustawia stan zaznaczenia dla gałęzi i kaskadowo dla wszystkich jej potomków."""
        self._is_internal_update = True
        cat_item.setCheckState(0, state)
        self._propagate_down(cat_item, state)

        p = cat_item.parent()
        while p and p != self.invisibleRootItem():
            self._refresh_parent_indicator(p)
            p = p.parent()

        self._is_internal_update = False

    def _invert_category_check_state(self, cat_item: QTreeWidgetItem):
        """Odwraca zaznaczenie wszystkich bezpośrednich lub zagnieżdżonych elementów w danej gałęzi."""
        self._is_internal_update = True

        def invert_recursive(parent: QTreeWidgetItem):
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.childCount() > 0:
                    invert_recursive(child)
                elif child.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    cur = child.checkState(0)
                    new_state = Qt.CheckState.Unchecked if cur == Qt.CheckState.Checked else Qt.CheckState.Checked
                    child.setCheckState(0, new_state)
                    self._sync_item_visibility_to_session(child)

        invert_recursive(cat_item)

        self._refresh_parent_indicator(cat_item)
        p = cat_item.parent()
        while p and p != self.invisibleRootItem():
            self._refresh_parent_indicator(p)
            p = p.parent()

        self._is_internal_update = False

    def _set_selected_items_check_state(self, items: List[QTreeWidgetItem], state: Qt.CheckState):
        """Zmienia stan checkboxów wyłącznie dla elementów zaznaczonych myszą."""
        self._is_internal_update = True
        parents_to_refresh = set()

        for itm in items:
            if itm.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                itm.setCheckState(0, state)
                self._sync_item_visibility_to_session(itm)
                if itm.parent():
                    parents_to_refresh.add(itm.parent())

        for p in parents_to_refresh:
            curr = p
            while curr and curr != self.invisibleRootItem():
                self._refresh_parent_indicator(curr)
                curr = curr.parent()

        self._is_internal_update = False

    def _isolate_selected_spectra(self, container_item: QTreeWidgetItem, selected_names: set):
        """Zaznacza tylko wskazane widma, odznaczając wszystkie pozostałe w tym samym kontenerze."""
        self._is_internal_update = True

        def isolate_recursive(parent: QTreeWidgetItem):
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.childCount() > 0:
                    isolate_recursive(child)
                elif child.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    s_name = child.data(0, Qt.UserRole)
                    new_state = Qt.CheckState.Checked if s_name in selected_names else Qt.CheckState.Unchecked
                    child.setCheckState(0, new_state)
                    self._sync_item_visibility_to_session(child)

        isolate_recursive(container_item)

        self._refresh_parent_indicator(container_item)
        p = container_item.parent()
        while p and p != self.invisibleRootItem():
            self._refresh_parent_indicator(p)
            p = p.parent()

        self._is_internal_update = False

    def _prompt_pattern_select(self, container_item: QTreeWidgetItem):
        """Dialog selekcji wg wzorca (np. *40C*, *TMAO*)."""
        pattern, ok = QInputDialog.getText(
            self, "Zaznacz według wzorca",
            "Wpisz szukaną frazę lub wzorzec (np. 50C, *rep*, pH7):"
        )
        if not ok or not pattern.strip():
            return

        patt = pattern.strip().lower()
        self._is_internal_update = True

        def match_recursive(parent: QTreeWidgetItem):
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.childCount() > 0:
                    match_recursive(child)
                elif child.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    s_name = str(child.data(0, Qt.UserRole)).lower()
                    is_match = (patt in s_name) or fnmatch.fnmatch(s_name, patt)
                    new_state = Qt.CheckState.Checked if is_match else Qt.CheckState.Unchecked
                    child.setCheckState(0, new_state)
                    self._sync_item_visibility_to_session(child)

        match_recursive(container_item)
        self._refresh_parent_indicator(container_item)
        self._is_internal_update = False

    # --- SZYBKIE PRZEPŁYWY PRACY (WORKFLOW SHORTCUTS) ---
    def _send_to_new_window(self, names: List[str]):
        """Klonuje wybrane widma i wstrzykuje je do nowego okna roboczego."""
        if not names:
            return

        def_name = f"Okno_{names[0][:12]}" if len(names) == 1 else "Wybrane Widma"
        win_name, ok = QInputDialog.getText(self, "Nowe Okno Robocze", "Nazwa docelowego okna:", text=def_name)
        if not ok or not win_name.strip():
            return

        new_win_id = session.add_window(name=win_name.strip())
        for n in names:
            spec = session.get_spectrum(n)
            if spec:
                session.add_spectrum_to_window(new_win_id, spec)

        session.set_context("window")
        session.set_active_window(new_win_id)

    def _duplicate_workspace_window(self, win_id: str):
        """Klonuje okno robocze wraz z widmami i ich widocznością."""
        src_win = session.windows.get(win_id)
        if not src_win:
            return

        new_name = f"{src_win.name} (Kopia)"
        new_win_id = session.add_window(name=new_name, folder_id=src_win.folder_id)

        for s_name, spec in src_win.spectra.items():
            win_spec = session.add_spectrum_to_window(new_win_id, spec)
            vis = src_win.visibility.get(s_name, True)
            session.set_window_spectrum_visibility(new_win_id, win_spec.name, vis)

        session.set_context("window")
        session.set_active_window(new_win_id)

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

        elif item_type == "data_table":
            session.set_active_table(name)

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

    def _prompt_import_files(self, target_atm: bool = False):
        """Otwiera okno wyboru plików i importuje je w tle do magazynu lub bazy atmosfery."""
        title = "Dodaj wzorce atmosfery..." if target_atm else tr("menu.open_spectra", "Dodaj pliki widm...")
        paths, _ = QFileDialog.getOpenFileNames(
            self, title, "", SpectrumImportDispatcher.get_open_file_filter()
        )
        if not paths:
            return

        worker = FileImportWorker(paths, target_atm=target_atm)

        # Podpięcie bezpiecznego odbioru w wątku GUI
        worker.signals.batch_loaded.connect(self._on_import_batch_received)
        worker.signals.error_occurred.connect(
            lambda errs: QMessageBox.warning(self, "Błędy importu", f"Pominięto niektóre pliki:\n{errs}")
        )
        self.thread_pool.start(worker)

    @Slot(list, bool)
    def _on_import_batch_received(self, loaded_spectra: list, is_atm: bool):
        """Wykonywane w 100% w wątku głównym (GUI)."""
        if is_atm:
            session.add_atm_spectra_batch(loaded_spectra)
        else:
            session.add_spectra_batch(loaded_spectra)

    def _get_window_relative_path(self, win: WorkspaceWindow) -> str:
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

    def _collect_export_targets(self, clicked_item: QTreeWidgetItem,
                                selected_items: List[QTreeWidgetItem]) -> Tuple[List[Tuple[SpectrumData, str]], str]:
        targets: List[Tuple[SpectrumData, str]] = []
        seen_keys = set()

        items_to_process = selected_items if (selected_items and clicked_item in selected_items and len(selected_items) > 1) else [clicked_item]

        suggested_name = "Eksport_Widm"
        if len(items_to_process) == 1:
            it = items_to_process[0]
            suggested_name = it.data(0, Qt.UserRole) or it.text(0).replace("📁 ", "").replace("🖼️ ", "").replace("🗂️ ", "").replace("📈 ", "")

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

            if itype == "window_spectrum":
                win_id = itm.data(0, Qt.UserRole + 2)
                win = session.windows.get(win_id)
                rel_dir = self._get_window_relative_path(win) if win else ""
                add_spec(win.spectra.get(iname) if win else None, rel_dir)

            elif itype in ["spectrum", "dft"]:
                add_spec(session.get_spectrum(iname), "Pomiary_Eksperymentalne" if itype == "spectrum" else "DFT")

            elif itype == "atm":
                add_spec(session.get_spectrum(iname), "Wzorce_Atmosfery")

            elif itype == "workspace_window":
                win = session.windows.get(iname)
                if win:
                    rel_dir = self._get_window_relative_path(win)
                    for s in win.spectra.values():
                        add_spec(s, rel_dir)

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

            elif itm == self.root_workspaces:
                for win in session.windows.values():
                    rel_dir = self._get_window_relative_path(win)
                    for s in win.spectra.values():
                        add_spec(s, rel_dir)

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

    def _export_table_to_csv(self, tbl: DataTable):
        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz tabelę jako CSV", f"{ExportManager.sanitize_filename(tbl.name)}.csv", "Plik CSV (*.csv)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tbl.to_tsv(decimal_comma=True).replace("\t", ";"))
                QMessageBox.information(self, "Zapisano", f"Pomyślnie zapisano tabelę do pliku:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Błąd zapisu tabeli:\n{e}")

    def _prompt_rename_table(self, tbl_id: str):
        tbl = session.get_table(tbl_id)
        if tbl:
            name, ok = QInputDialog.getText(self, "Zmień nazwę", "Nowa nazwa tabeli:", text=tbl.name)
            if ok and name.strip():
                session.rename_table(tbl_id, name.strip())

    def _populate_export_menu(self, menu: QMenu, item: QTreeWidgetItem, selected_items: List[QTreeWidgetItem]):
        targets, suggested_name = self._collect_export_targets(item, selected_items)
        if not targets:
            return

        menu.addSeparator()
        last_fmt_id = ExportManager.get_last_format()
        last_fmt = ExportManager.get_format_info(last_fmt_id)
        all_fmts = ExportManager.get_all_formats()

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

        # 0. KORZEŃ MAGAZYNU DANYCH
        if item == self.root_storage:
            act_import = menu.addAction("📥 Dodaj pliki widm do magazynu...")
            act_import.triggered.connect(lambda: self._prompt_import_files(target_atm=False))

            act_import_dir = menu.addAction("📁 Dodaj cały folder widm...")
            act_import_dir.triggered.connect(lambda: self._prompt_import_directory(target_atm=False))

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystko w magazynie")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystko w magazynie")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))
            act_inv_all = menu.addAction("🔄 Odwróć zaznaczenie w magazynie")
            act_inv_all.triggered.connect(lambda: self._invert_category_check_state(item))

            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 1. KORZEŃ OKIEN ROBOCZYCH
        if item == self.root_workspaces:
            act_new_win = menu.addAction("➕ Dodaj nowe okno robocze...")
            act_new_win.triggered.connect(lambda: self._prompt_new_window())

            act_new_folder = menu.addAction("📁 Dodaj nowy folder okien...")
            act_new_folder.triggered.connect(lambda: self._prompt_new_folder())

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie okna robocze")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie okna robocze")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))
            act_inv_all = menu.addAction("🔄 Odwróć zaznaczenie we wszystkich oknach")
            act_inv_all.triggered.connect(lambda: self._invert_category_check_state(item))

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

            act_rename = menu.addAction("✏️ Zmień nazwę folderu (F2)...")
            act_rename.triggered.connect(lambda: self._prompt_rename_folder(folder_id))

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tym folderze")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tym folderze")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))
            act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tym folderze")
            act_inv.triggered.connect(lambda: self._invert_category_check_state(item))

            menu.addSeparator()
            act_del_keep = menu.addAction("🗑 Usuń folder (zachowaj okna)")
            act_del_keep.triggered.connect(lambda: session.remove_folder(folder_id, delete_windows=False))

            act_del_all = menu.addAction("🗑 Usuń folder wraz z oknami")
            act_del_all.triggered.connect(lambda: session.remove_folder(folder_id, delete_windows=True))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # MENU KONTEKSTOWE TABELI DANYCH
        if item.data(0, Qt.UserRole + 1) == "data_table":
            tbl_id = item.data(0, Qt.UserRole)
            tbl = session.get_table(tbl_id)
            if tbl:
                act_view = menu.addAction("🔍 Pokaż tabelę / parametry...")
                act_view.triggered.connect(lambda: TableViewerDialog(tbl, parent=self).exec())

                menu.addSeparator()

                act_copy_tsv = menu.addAction("📋 Kopiuj dane (TSV pod Origin / Excel)")
                act_copy_tsv.triggered.connect(
                    lambda: QApplication.clipboard().setText(tbl.to_tsv(decimal_comma=True)))

                act_save_csv = menu.addAction("💾 Zapisz jako CSV...")
                act_save_csv.triggered.connect(lambda: self._export_table_to_csv(tbl))

                menu.addSeparator()
                sub_move = menu.addMenu("📁 Przenieś do folderu...")
                act_root = sub_move.addAction("Poziom główny (bez folderu)")
                act_root.triggered.connect(lambda: session.move_table_to_folder(tbl_id, None))
                sub_move.addSeparator()
                for fid, f in session.folders.items():
                    act_f = sub_move.addAction(f"📁 {f.name}")
                    act_f.triggered.connect(
                        lambda checked=False, target_fid=fid: session.move_table_to_folder(tbl_id, target_fid))

                menu.addSeparator()
                act_rename = menu.addAction("✏️ Zmień nazwę tabeli (F2)...")
                act_rename.triggered.connect(lambda: self._prompt_rename_table(tbl_id))

                act_del = menu.addAction("🗑 Usuń tabelę (Del)")
                act_del.triggered.connect(lambda: session.remove_table(tbl_id))

            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 3. OKNO ROBOCZE
        if item.data(0, Qt.UserRole + 1) == "workspace_window":
            win_id = item.data(0, Qt.UserRole)
            act_rename = menu.addAction("✏️ Zmień nazwę okna (F2)...")
            act_rename.triggered.connect(lambda: self._prompt_rename_window(win_id))

            act_dup = menu.addAction("📑 Zduplikuj to okno...")
            act_dup.triggered.connect(lambda: self._duplicate_workspace_window(win_id))

            sub_move = menu.addMenu("📁 Przenieś do folderu...")
            act_root = sub_move.addAction("Poziom główny (bez folderu)")
            act_root.triggered.connect(lambda: session.move_window_to_folder(win_id, None))
            sub_move.addSeparator()
            for fid, f in session.folders.items():
                act_f = sub_move.addAction(f"📁 {f.name}")
                act_f.triggered.connect(
                    lambda checked=False, target_fid=fid: session.move_window_to_folder(win_id, target_fid))

            menu.addSeparator()
            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tym oknie")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))
            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tym oknie")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))
            act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tym oknie")
            act_inv.triggered.connect(lambda: self._invert_category_check_state(item))

            act_patt = menu.addAction("🔍 Zaznacz według wzorca...")
            act_patt.triggered.connect(lambda: self._prompt_pattern_select(item))

            menu.addSeparator()
            act_del = menu.addAction("🗑 Usuń okno (Del)")
            act_del.triggered.connect(lambda: session.remove_window(win_id))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 4. KATEGORIE MAGAZYNU
        if item in (self.root_exp, self.root_atm, self.root_series, self.root_dft):
            if item == self.root_exp:
                act_import = menu.addAction("📥 Dodaj pliki pomiarów...")
                act_import.triggered.connect(lambda: self._prompt_import_files(target_atm=False))

                act_import_dir = menu.addAction("📁 Dodaj folder pomiarów...")
                act_import_dir.triggered.connect(lambda: self._prompt_import_directory(target_atm=False))
                menu.addSeparator()

            elif item == self.root_atm:
                act_import = menu.addAction("☁️ Dodaj pliki wzorców atmosfery...")
                act_import.triggered.connect(lambda: self._prompt_import_files(target_atm=True))

                act_import_dir = menu.addAction("📁 Dodaj folder wzorców atmosfery...")
                act_import_dir.triggered.connect(lambda: self._prompt_import_directory(target_atm=True))
                menu.addSeparator()

            act_chk_all = menu.addAction("☑ Zaznacz wszystkie w tej gałęzi")
            act_chk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Checked))

            act_unchk_all = menu.addAction("☐ Odznacz wszystkie w tej gałęzi")
            act_unchk_all.triggered.connect(lambda: self._set_category_check_state(item, Qt.CheckState.Unchecked))

            act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tej gałęzi")
            act_inv.triggered.connect(lambda: self._invert_category_check_state(item))

            act_patt = menu.addAction("🔍 Zaznacz według wzorca...")
            act_patt.triggered.connect(lambda: self._prompt_pattern_select(item))

            menu.addSeparator()
            act_clear_cat = menu.addAction("🗑 Wyczyść tę gałąź")
            act_clear_cat.triggered.connect(lambda: self._clear_category_items(item))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 5. WIDMO W MAGAZYNIE (EKSPERYMENT / DFT)
        if item.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]:
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) in ["spectrum", "dft"]]
            if not names:
                names = [item.data(0, Qt.UserRole)]

            parent_cat = item.parent() or self.root_exp

            if len(names) > 1:
                act_chk_sel = menu.addAction(f"☑ Zaznacz wybrane ({len(names)}) [Spacja]")
                act_chk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Checked))
                act_unchk_sel = menu.addAction(f"☐ Odznacz wybrane ({len(names)}) [Spacja]")
                act_unchk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Unchecked))
                act_iso = menu.addAction(f"🎯 Zaznacz tylko wybrane ({len(names)})")
                act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_cat, set(names)))
            else:
                act_iso = menu.addAction("🎯 Zaznacz tylko to widmo")
                act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_cat, {names[0]}))

                act_color = menu.addAction("🎨 Zmień kolor widma...")
                act_color.triggered.connect(lambda: self._open_color_picker_for_item(item, names[0], item.data(0, Qt.UserRole + 1)))

            act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tej gałęzi")
            act_inv.triggered.connect(lambda: self._invert_category_check_state(parent_cat))

            menu.addSeparator()
            act_new_win = menu.addAction(f"🚀 Wyślij do nowego okna roboczego ({len(names)})...")
            act_new_win.triggered.connect(lambda: self._send_to_new_window(names))

            if session.active_window_id:
                act_add_win = menu.addAction("➕ Dodaj do aktywnego okna roboczego")
                act_add_win.triggered.connect(lambda: [session.add_spectrum_to_window(session.active_window_id, session.get_spectrum(n)) for n in names if session.get_spectrum(n)])

            menu.addSeparator()
            if len(names) > 1:
                act_move_atm = menu.addAction(f"☁️ Przenieś zaznaczone ({len(names)}) do Wzorców Atmosfery")
                act_move_atm.triggered.connect(lambda: [session.move_to_atm(n) for n in names])
                act_rem = menu.addAction(f"🗑 Usuń zaznaczone ({len(names)}) z magazynu (Del)")
                act_rem.triggered.connect(lambda: [session.remove_spectrum(n) for n in names])
            else:
                s_name = names[0]
                act_move_atm = menu.addAction("☁️ Przenieś do Wzorców Atmosfery (ATM / CO₂)")
                act_move_atm.triggered.connect(lambda: session.move_to_atm(s_name))
                act_rem = menu.addAction(f"🗑 Usuń '{s_name}' z magazynu (Del)")
                act_rem.triggered.connect(lambda: session.remove_spectrum(s_name))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 6. WZORCE ATMOSFERY
        if item.data(0, Qt.UserRole + 1) == "atm":
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) == "atm"]
            if not names:
                names = [item.data(0, Qt.UserRole)]

            parent_cat = item.parent() or self.root_atm

            if len(names) > 1:
                act_chk_sel = menu.addAction(f"☑ Zaznacz wybrane ({len(names)}) [Spacja]")
                act_chk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Checked))
                act_unchk_sel = menu.addAction(f"☐ Odznacz wybrane ({len(names)}) [Spacja]")
                act_unchk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Unchecked))
                act_iso = menu.addAction(f"🎯 Zaznacz tylko wybrane ({len(names)})")
                act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_cat, set(names)))
            else:
                act_iso = menu.addAction("🎯 Zaznacz tylko ten wzorzec")
                act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_cat, {names[0]}))

                act_color = menu.addAction("🎨 Zmień kolor wzorca...")
                act_color.triggered.connect(lambda: self._open_color_picker_for_item(item, names[0], "atm"))

            act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tej gałęzi")
            act_inv.triggered.connect(lambda: self._invert_category_check_state(parent_cat))

            menu.addSeparator()

            if len(names) > 1:
                act_move_exp = menu.addAction(f"📈 Przenieś zaznaczone ({len(names)}) do Pomiarów Eksperymentalnych")
                act_move_exp.triggered.connect(lambda: [session.move_to_experimental(n) for n in names])
                act_rem = menu.addAction(f"🗑 Usuń zaznaczone ({len(names)}) wzorce atmosfery (Del)")
                act_rem.triggered.connect(lambda: [session.remove_spectrum(n) for n in names])
            else:
                s_name = names[0]
                act_move_exp = menu.addAction("📈 Przenieś do Pomiarów Eksperymentalnych")
                act_move_exp.triggered.connect(lambda: session.move_to_experimental(s_name))
                act_rem = menu.addAction(f"🗑 Usuń '{s_name}' (Del)")
                act_rem.triggered.connect(lambda: session.remove_spectrum(s_name))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 7. WIDMO W OKNIE ROBOCZYM
        if item.data(0, Qt.UserRole + 1) == "window_spectrum":
            win_id = item.data(0, Qt.UserRole + 2)
            names = [it.data(0, Qt.UserRole) for it in selected_items if it.data(0, Qt.UserRole + 1) == "window_spectrum"]
            if not names:
                names = [item.data(0, Qt.UserRole)]

            parent_win_item = item.parent()

            if len(names) > 1:
                act_chk_sel = menu.addAction(f"☑ Zaznacz wybrane ({len(names)}) [Spacja]")
                act_chk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Checked))
                act_unchk_sel = menu.addAction(f"☐ Odznacz wybrane ({len(names)}) [Spacja]")
                act_unchk_sel.triggered.connect(lambda: self._set_selected_items_check_state(selected_items, Qt.CheckState.Unchecked))
                if parent_win_item:
                    act_iso = menu.addAction(f"🎯 Zaznacz tylko wybrane ({len(names)})")
                    act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_win_item, set(names)))
            else:
                if parent_win_item:
                    act_iso = menu.addAction("🎯 Zaznacz tylko to widmo")
                    act_iso.triggered.connect(lambda: self._isolate_selected_spectra(parent_win_item, {names[0]}))

                act_color = menu.addAction("🎨 Zmień kolor widma...")
                act_color.triggered.connect(lambda: self._open_color_picker_for_item(item, names[0], "window_spectrum"))

            if parent_win_item:
                act_inv = menu.addAction("🔄 Odwróć zaznaczenie w tym oknie")
                act_inv.triggered.connect(lambda: self._invert_category_check_state(parent_win_item))

            menu.addSeparator()
            act_new_win = menu.addAction(f"🚀 Wyślij do nowego okna ({len(names)})...")
            act_new_win.triggered.connect(lambda: self._send_to_new_window(names))

            menu.addSeparator()
            if len(names) > 1:
                act_rem = menu.addAction(f"✕ Usuń zaznaczone ({len(names)}) z tego okna (Del)")
                act_rem.triggered.connect(lambda: [session.remove_spectrum_from_window(win_id, n) for n in names])
            else:
                spec_name = names[0]
                act_rem = menu.addAction(f"✕ Usuń '{spec_name}' z tego okna (Del)")
                act_rem.triggered.connect(lambda: session.remove_spectrum_from_window(win_id, spec_name))

            self._populate_export_menu(menu, item, selected_items)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        # 8. SERIE 2D / 3D
        if item.data(0, Qt.UserRole + 1) == "series":
            s_name = item.data(0, Qt.UserRole)
            act_rem_ser = menu.addAction(f"🗑 Usuń serię '{s_name}' (Del)")
            act_rem_ser.triggered.connect(lambda: session.remove_series(s_name))
            menu.exec(self.viewport().mapToGlobal(pos))
            return

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
                    win_item.setText(0, f"🖼️ {name.strip()}")

    def _find_table_item(self, table_id: str, parent: Optional[QTreeWidgetItem] = None) -> Optional[QTreeWidgetItem]:
        if parent is None:
            parent = self.root_workspaces

        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.UserRole + 1) == "data_table" and child.data(0, Qt.UserRole) == table_id:
                return child
            if child.data(0, Qt.UserRole + 1) == "window_folder":
                sub = self._find_table_item(table_id, parent=child)
                if sub:
                    return sub
        return None

    def _create_table_item(self, table_id: str, name: str, folder_id: Optional[str] = None) -> QTreeWidgetItem:
        parent_item = self.root_workspaces
        if folder_id:
            found_folder = self._find_folder_item(folder_id)
            if found_folder:
                parent_item = found_folder

        tbl_item = QTreeWidgetItem(parent_item, [f"📊 {name}", ""])
        tbl_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        tbl_item.setData(0, Qt.UserRole, table_id)
        tbl_item.setData(0, Qt.UserRole + 1, "data_table")
        return tbl_item

    def _on_table_added(self, table):
        if not self._find_table_item(table.table_id):
            self._create_table_item(table.table_id, table.name, table.folder_id)

    def _on_table_removed(self, table_id: str):
        tbl_node = self._find_table_item(table_id)
        if tbl_node:
            p = tbl_node.parent() or self.root_workspaces
            p.removeChild(tbl_node)

    def _on_table_renamed(self, table_id: str, new_name: str):
        tbl_node = self._find_table_item(table_id)
        if tbl_node:
            tbl_node.setText(0, f"📊 {new_name}")

    def _on_table_folder_changed(self, table_id: str, folder_id: Optional[str]):
        tbl_node = self._find_table_item(table_id)
        if not tbl_node:
            return

        old_parent = tbl_node.parent() or self.root_workspaces
        old_parent.removeChild(tbl_node)

        new_parent = self.root_workspaces
        if folder_id:
            target_f = self._find_folder_item(folder_id)
            if target_f:
                new_parent = target_f

        new_parent.addChild(tbl_node)