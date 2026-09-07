"""
gui/tensor_builder_dialog.py
Interaktywny kreator serii ewolucyjnych 2D (kinetyka) oraz tensorów 3D.
"""

from typing import Tuple, Optional, Dict, Any, List
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox, QTabWidget,
    QGroupBox, QFrame, QAbstractItemView, QMenu
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import numpy as np
import os
import re
import logging

from core.session import session
from core.models import SpectralSeries, SpectrumData
from core.i18n import tr
from io_hub.dispatcher import SpectrumImportDispatcher

logger = logging.getLogger("SpectroHub")

STATUS_COLORS = {
    'EMPTY': QColor("#3a3a48"),
    'LOADED': QColor("#2a9d8f"),
    'MISSING': QColor("#6c757d"),
    'FILLED': QColor("#457b9d")
}


class TensorBuilderDialog(QDialog):
    """Kreator serii ewolucyjnych 2D oraz tensorów 3D z ulepszonym UX."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("tensor_builder.title", "Kreator Serii Ewolucyjnych i Tensorów 3D"))
        self.resize(950, 680)

        self.grid_data = {}       # (r, c) -> {'wn': np.ndarray, 'y': np.ndarray, 'name': str, 'status': str}
        self.common_wavenumbers = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        # Zakładka 1: Seria 2D (Kinetyka / Rampa temperaturowa)
        tab_2d = QWidget()
        self._setup_tab_2d(tab_2d)
        self.tabs.addTab(tab_2d, "📈 Seria Ewolucyjna 2D (Kinetyka)")

        # Zakładka 2: Tensor 3D (Siatka N x M)
        tab_3d = QWidget()
        self._setup_tab_3d(tab_3d)
        self.tabs.addTab(tab_3d, "🧊 Tensor 3D (Siatka N × M)")

        main_layout.addWidget(self.tabs)

    # -----------------------------------------------------------------
    # POMOCNICZA EKSTRAKCJA CZASU (Styl EvoChem)
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> float:
        """Konwertuje ciąg HH:MM:SS na liczbę sekund od początku doby."""
        parts = time_str.split(':')
        if len(parts) == 3:
            try:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            except ValueError:
                pass
        return 0.0

    @classmethod
    def _extract_spectrum_timestamp(cls, spec: SpectrumData) -> Tuple[str, float]:
        """
        Wyciąga czas pomiaru:
        1. Ze spec.time_str / metadanych OPUS (TIM, Time).
        2. Z wzorca nazwy pliku (HH-MM-SS lub HH_MM_SS).
        3. Ze stempla modyfikacji pliku (mtime) na dysku.
        Zwraca: (tekst_hh_mm_ss, sekundy_od_poczatku_doby).
        """
        time_str = getattr(spec, 'time_str', '00:00:00')

        # 1. Sprawdzenie metadanych
        if time_str == "00:00:00" and hasattr(spec, 'metadata'):
            meta_time = spec.metadata.get('TIM') or spec.metadata.get('Time')
            if meta_time:
                time_str = str(meta_time).strip()

        # 2. Wzorzec w nazwie pliku (np. 'pomiar_14-25-30.csv' lub 't_14_25_30')
        if time_str == "00:00:00":
            match = re.search(r'(\d{2})[-_:](\d{2})[-_:](\d{2})', spec.name)
            if match:
                time_str = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"

        # 3. Stempel modyfikacji pliku (mtime)
        if time_str == "00:00:00" and spec.file_path and os.path.exists(spec.file_path):
            try:
                mtime = os.path.getmtime(spec.file_path)
                import time as t_mod
                time_str = t_mod.strftime('%H:%M:%S', t_mod.localtime(mtime))
            except Exception:
                pass

        total_sec = cls._parse_time_to_seconds(time_str)
        return time_str, total_sec

    # -----------------------------------------------------------------
    # ZAKŁADKA 1: SERIA 2D (Zoptymalizowana pod czasy rzeczywiste)
    # -----------------------------------------------------------------
    def _setup_tab_2d(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(8)

        # 1. Konfiguracja osi i trybu czasu
        grp_cfg = QGroupBox("Parametry Serii Kinetycznej / Ewolucyjnej 2D")
        l_cfg = QVBoxLayout(grp_cfg)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Nazwa serii:"))
        self.ent_2d_name = QLineEdit("Kinetyka_1")
        row1.addWidget(self.ent_2d_name, 1)

        row1.addWidget(QLabel("Tryb wyznaczania osi ewolucyjnej:"))
        self.combo_2d_mode = QComboBox()
        self.combo_2d_mode.addItems([
            "Czas z plików (Δt w sekundach od startu)",
            "Czas z plików (Δt w minutach od startu)",
            "Czas bezwzględny (HH:MM:SS)",
            "Wartość liczbowa z nazwy pliku",
            "Stały krok zdefiniowany ręcznie (Δt)"
        ])
        self.combo_2d_mode.currentIndexChanged.connect(self._recalculate_2d_values)
        row1.addWidget(self.combo_2d_mode, 1)
        l_cfg.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Domyślny krok Δt (dla trybu ręcznego):"))
        self.spin_2d_dt = QDoubleSpinBox()
        self.spin_2d_dt.setRange(0.01, 86400.0)
        self.spin_2d_dt.setValue(60.0)
        self.spin_2d_dt.valueChanged.connect(self._recalculate_2d_values)
        row2.addWidget(self.spin_2d_dt)
        row2.addStretch()
        l_cfg.addLayout(row2)

        layout.addWidget(grp_cfg)

        # 2. Tabela widm
        self.table_2d = QTableWidget(0, 4)
        self.table_2d.setHorizontalHeaderLabels(
            ["Lp.", "Widmo / Plik", "Stempel (HH:MM:SS)", "Wartość na osi (do edycji)"])
        self.table_2d.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_2d.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table_2d)

        # 3. Przyciski zasilania i sortowania
        row_btns = QHBoxLayout()
        btn_add_proj = QPushButton("➕ Pobierz zaznaczone z projektu")
        btn_add_proj.clicked.connect(self._add_selected_project_spectra_2d)

        btn_add_files = QPushButton("📂 Importuj serię z dysku (OPUS / CSV)...")
        btn_add_files.clicked.connect(self._import_files_directly_2d)

        btn_sort_time = QPushButton("⏱ Sortuj wg czasu pomiaru")
        btn_sort_time.clicked.connect(self._sort_2d_by_time)

        btn_sort_name = QPushButton("🔤 Sortuj A-Z")
        btn_sort_name.clicked.connect(self._sort_2d_by_name)

        btn_clear = QPushButton("🗑 Wyczyść")
        btn_clear.clicked.connect(lambda: self.table_2d.setRowCount(0))

        row_btns.addWidget(btn_add_proj)
        row_btns.addWidget(btn_add_files)
        row_btns.addWidget(btn_sort_time)
        row_btns.addWidget(btn_sort_name)
        row_btns.addWidget(btn_clear)
        layout.addLayout(row_btns)

        # 4. Przycisk zatwierdzenia
        btn_commit_2d = QPushButton("🚀 Utwórz Serię 2D i Dodaj do Projektu")
        btn_commit_2d.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 10px;")
        btn_commit_2d.clicked.connect(self._commit_series_2d)
        layout.addWidget(btn_commit_2d)

    def _add_selected_project_spectra_2d(self):
        """Pobiera widma z sesji wraz z ich stemplami czasowymi."""
        for name, spec in session.spectra.items():
            if session.is_visible(name):
                time_str, total_sec = self._extract_spectrum_timestamp(spec)
                row = self.table_2d.rowCount()
                self.table_2d.insertRow(row)

                self.table_2d.setItem(row, 0, QTableWidgetItem(str(row + 1)))

                item_name = QTableWidgetItem(name)
                item_name.setData(Qt.UserRole, spec)
                item_name.setData(Qt.UserRole + 1, total_sec)
                self.table_2d.setItem(row, 1, item_name)

                self.table_2d.setItem(row, 2, QTableWidgetItem(time_str))
                self.table_2d.setItem(row, 3, QTableWidgetItem("0.00"))

        self._recalculate_2d_values()

    def _import_files_directly_2d(self):
        """Wczytuje serię plików bezpośrednio z dysku (wzorem EvoChem)."""
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz pliki serii (Bruker OPUS / CSV / SPC)", "",
            "Pliki widmowe (*.0 *.[0-9]* *.opus *.csv *.dpt *.spc);;Wszystkie (*.*)"
        )
        if not paths:
            return

        for p in paths:
            try:
                specs = SpectrumImportDispatcher.load_file(p)
                for s in specs:
                    session.add_spectrum(s, make_active=False)
                    time_str, total_sec = self._extract_spectrum_timestamp(s)

                    row = self.table_2d.rowCount()
                    self.table_2d.insertRow(row)
                    self.table_2d.setItem(row, 0, QTableWidgetItem(str(row + 1)))

                    item_name = QTableWidgetItem(s.name)
                    item_name.setData(Qt.UserRole, s)
                    item_name.setData(Qt.UserRole + 1, total_sec)
                    self.table_2d.setItem(row, 1, item_name)

                    self.table_2d.setItem(row, 2, QTableWidgetItem(time_str))
                    self.table_2d.setItem(row, 3, QTableWidgetItem("0.00"))
            except Exception as e:
                logger.error(f"Błąd importu serii z pliku {p}: {e}")

        self._sort_2d_by_time()

    def _sort_2d_by_time(self):
        """Sortuje wiersze tabeli chronologicznie według sekund od początku doby."""
        rows = []
        n_rows = self.table_2d.rowCount()
        for r in range(n_rows):
            item_name = self.table_2d.item(r, 1)
            time_str = self.table_2d.item(r, 2).text()
            spec = item_name.data(Qt.UserRole)
            sec = item_name.data(Qt.UserRole + 1) or self._parse_time_to_seconds(time_str)
            rows.append((spec, item_name.text(), time_str, sec))

        rows.sort(key=lambda x: x[3])

        self.table_2d.setRowCount(0)
        for idx, (spec, name, time_str, sec) in enumerate(rows):
            self.table_2d.insertRow(idx)
            self.table_2d.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))

            item = QTableWidgetItem(name)
            item.setData(Qt.UserRole, spec)
            item.setData(Qt.UserRole + 1, sec)
            self.table_2d.setItem(idx, 1, item)

            self.table_2d.setItem(idx, 2, QTableWidgetItem(time_str))
            self.table_2d.setItem(idx, 3, QTableWidgetItem("0.00"))

        self._recalculate_2d_values()

    def _sort_2d_by_name(self):
        """Sortuje alfabetycznie/numerycznie wg nazw."""
        rows = []
        n_rows = self.table_2d.rowCount()
        for r in range(n_rows):
            item_name = self.table_2d.item(r, 1)
            time_str = self.table_2d.item(r, 2).text()
            spec = item_name.data(Qt.UserRole)
            sec = item_name.data(Qt.UserRole + 1) or 0.0
            rows.append((spec, item_name.text(), time_str, sec))

        rows.sort(key=lambda x: x[1])

        self.table_2d.setRowCount(0)
        for idx, (spec, name, time_str, sec) in enumerate(rows):
            self.table_2d.insertRow(idx)
            self.table_2d.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))

            item = QTableWidgetItem(name)
            item.setData(Qt.UserRole, spec)
            item.setData(Qt.UserRole + 1, sec)
            self.table_2d.setItem(idx, 1, item)

            self.table_2d.setItem(idx, 2, QTableWidgetItem(time_str))
            self.table_2d.setItem(idx, 3, QTableWidgetItem("0.00"))

        self._recalculate_2d_values()

    def _recalculate_2d_values(self):
        """Przelicza wartości osi w kolumnie 3 w zależności od wybranego trybu."""
        n_rows = self.table_2d.rowCount()
        if n_rows == 0:
            return

        mode_idx = self.combo_2d_mode.currentIndex()

        # Pobieramy punkty czasowe w sekundach
        sec_list = []
        for r in range(n_rows):
            item_name = self.table_2d.item(r, 1)
            time_str = self.table_2d.item(r, 2).text()
            sec = item_name.data(Qt.UserRole + 1) if item_name else None
            if sec is None or sec == 0:
                sec = self._parse_time_to_seconds(time_str)
            sec_list.append(sec)

        t0 = sec_list[0] if sec_list else 0.0

        for r in range(n_rows):
            name = self.table_2d.item(r, 1).text()

            # 0: Sekundy od startu
            if mode_idx == 0:
                val = sec_list[r] - t0
                if val < 0:  # Przejście przez północ
                    val += 86400.0
                txt = f"{val:.1f}"

            # 1: Minuty od startu
            elif mode_idx == 1:
                val = (sec_list[r] - t0) / 60.0
                if val < 0:
                    val += 1440.0
                txt = f"{val:.2f}"

            # 2: Czas bezwzględny w godzinach ułamkowych
            elif mode_idx == 2:
                val = sec_list[r] / 3600.0
                txt = f"{val:.3f}"

            # 3: Wartość z nazwy pliku
            elif mode_idx == 3:
                nums = re.findall(r'[-+]?\d*\.?\d+', name)
                val = float(nums[-1]) if nums else float(r)
                txt = f"{val:.2f}"

            # 4: Stały krok zdefiniowany ręcznie
            else:
                dt = self.spin_2d_dt.value()
                val = float(r * dt)
                txt = f"{val:.2f}"

            item_val = self.table_2d.item(r, 3)
            if item_val:
                item_val.setText(txt)

    def _commit_series_2d(self):
        n_rows = self.table_2d.rowCount()
        if n_rows < 2:
            QMessageBox.warning(self, tr("app.warning"), "Seria musi zawierać co najmniej 2 widma.")
            return

        spec_names = [self.table_2d.item(r, 1).text() for r in range(n_rows)]
        time_labels = [self.table_2d.item(r, 2).text() for r in range(n_rows)]

        # Pobranie wartości numerycznych z kolumny 3 (z obsługą kropki i przecinka)
        try:
            var_values = np.array([
                float(self.table_2d.item(r, 3).text().replace(',', '.'))
                for r in range(n_rows)
            ], dtype=np.float64)
        except ValueError as e:
            QMessageBox.critical(
                self, tr("app.error"),
                f"Wszystkie wartości w kolumnie 'Wartość na osi' muszą być poprawnymi liczbami!\n{e}"
            )
            return

        # Pobieramy referencyjną siatkę liczb falowych z pierwszego widma
        ref_spec = session.spectra.get(spec_names[0])
        if not ref_spec:
            QMessageBox.critical(self, tr("app.error"), f"Nie znaleziono widma '{spec_names[0]}' w sesji.")
            return

        ref_wn = ref_spec.wavenumbers
        matrix_2d = np.zeros((len(ref_wn), n_rows), dtype=np.float64)

        for idx, name in enumerate(spec_names):
            s = session.spectra.get(name)
            if s is None:
                continue
            if len(s.wavenumbers) == len(ref_wn) and np.allclose(s.wavenumbers, ref_wn):
                matrix_2d[:, idx] = s.intensities
            else:
                # Automatyczna interpolacja w razie drobnych przesunięć siatki
                matrix_2d[:, idx] = np.interp(ref_wn, s.wavenumbers, s.intensities)

        # Zapisujemy serię z metadanymi stempli czasowych
        series_obj = SpectralSeries(
            name=self.ent_2d_name.text().strip() or "Seria_2D",
            wavenumbers=ref_wn,
            data=matrix_2d,
            axis_n_name=self.combo_2d_mode.currentText(),
            axis_n_values=var_values,
            metadata={
                "time_labels": time_labels,
                "spec_names": spec_names
            }
        )
        session.add_series(series_obj)
        QMessageBox.information(
            self, tr("app.info"),
            f"Pomyślnie utworzono serię 2D '{series_obj.name}' ({n_rows} widm)!"
        )
        self.accept()

    # -----------------------------------------------------------------
    # ZAKŁADKA 2: TENSOR 3D
    # -----------------------------------------------------------------
    def _setup_tab_3d(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(8)

        # 1. Górny pasek wymiarów
        row_dim = QHBoxLayout()
        row_dim.addWidget(QLabel("Nazwa tensora:"))
        self.ent_3d_name = QLineEdit("Tensor_3D_1")
        row_dim.addWidget(self.ent_3d_name)

        row_dim.addWidget(QLabel("Wiersze (N):"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 50)
        self.spin_n.setValue(4)
        row_dim.addWidget(self.spin_n)

        row_dim.addWidget(QLabel("Kolumny (M):"))
        self.spin_m = QSpinBox()
        self.spin_m.setRange(2, 50)
        self.spin_m.setValue(4)
        row_dim.addWidget(self.spin_m)

        btn_build_grid = QPushButton("🔲 Zbuduj Siatkę")
        btn_build_grid.clicked.connect(self._rebuild_grid_table)
        row_dim.addWidget(btn_build_grid)
        layout.addLayout(row_dim)

        # 2. Siatka matrycy tensora
        self.table_grid = QTableWidget()
        self.table_grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        layout.addWidget(self.table_grid)

        # 3. Pasek automatycznego zasilania siatki
        row_fill = QHBoxLayout()
        row_fill.addWidget(QLabel("Kierunek nawijania:"))
        self.combo_direction = QComboBox()
        self.combo_direction.addItems(["Wierszami (→)", "Kolumnami (↓)"])
        row_fill.addWidget(self.combo_direction)

        btn_fill_project = QPushButton("📥 Wypełnij widmami z projektu")
        btn_fill_project.clicked.connect(self._autofill_from_project)
        row_fill.addWidget(btn_fill_project)

        btn_impute = QPushButton("✨ Uzupełnij puste / braki...")
        btn_impute.clicked.connect(self._show_impute_menu)
        row_fill.addWidget(btn_impute)
        layout.addLayout(row_fill)

        # 4. Przycisk zatwierdzenia tensora
        btn_commit_3d = QPushButton("🚀 Utwórz Tensor 3D i Dodaj do Projektu")
        btn_commit_3d.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 10px;")
        btn_commit_3d.clicked.connect(self._commit_tensor_3d)
        layout.addWidget(btn_commit_3d)

        self._rebuild_grid_table()

    def _rebuild_grid_table(self):
        n = self.spin_n.value()
        m = self.spin_m.value()
        self.table_grid.setRowCount(n)
        self.table_grid.setColumnCount(m)

        self.table_grid.setHorizontalHeaderLabels([f"Kolumna {c+1}" for c in range(m)])
        self.table_grid.setVerticalHeaderLabels([f"Wiersz {r+1}" for r in range(n)])

        self.grid_data.clear()
        for r in range(n):
            for c in range(m):
                item = QTableWidgetItem("Puste")
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(STATUS_COLORS['EMPTY'])
                self.table_grid.setItem(r, c, item)
                self.grid_data[(r, c)] = {'status': 'EMPTY', 'name': '', 'y': None}

    def _autofill_from_project(self):
        spectra_names = [name for name, s in session.spectra.items() if session.is_visible(name)]
        if not spectra_names:
            QMessageBox.information(self, tr("app.info"), "Brak widocznych widm w projekcie.")
            return

        n, m = self.spin_n.value(), self.spin_m.value()
        direction = self.combo_direction.currentIndex()  # 0: rows, 1: cols

        coords_seq = []
        if direction == 0:
            for r in range(n):
                for c in range(m):
                    coords_seq.append((r, c))
        else:
            for c in range(m):
                for r in range(n):
                    coords_seq.append((r, c))

        for idx, (r, c) in enumerate(coords_seq):
            if idx < len(spectra_names):
                name = spectra_names[idx]
                spec = session.spectra[name]
                if self.common_wavenumbers is None:
                    self.common_wavenumbers = spec.wavenumbers.copy()

                y_interp = np.interp(self.common_wavenumbers, spec.wavenumbers, spec.intensities)
                self.grid_data[(r, c)] = {'status': 'LOADED', 'name': name, 'y': y_interp}

                item = self.table_grid.item(r, c)
                item.setText(f"({r},{c})\n{name[:10]}")
                item.setBackground(STATUS_COLORS['LOADED'])
            else:
                break

    def _on_grid_context_menu(self, pos):
        item = self.table_grid.itemAt(pos)
        if not item:
            return
        r, c = item.row(), item.column()

        menu = QMenu(self)
        act_missing = menu.addAction("Oznacz jako BRAK (Missing / NaN)")
        act_empty = menu.addAction("Oznacz jako Puste")
        menu.addSeparator()
        act_fill_zero = menu.addAction("Wypełnij zerami (0.0)")
        act_fill_row = menu.addAction("Wypełnij średnią wiersza (Row Mean)")
        act_fill_col = menu.addAction("Wypełnij średnią kolumny (Col Mean)")
        act_fill_neigh = menu.addAction("Wypełnij średnią z sąsiadów")

        action = menu.exec(self.table_grid.viewport().mapToGlobal(pos))
        if action == act_missing:
            self._set_cell_status(r, c, 'MISSING', "BRAK", None)
        elif action == act_empty:
            self._set_cell_status(r, c, 'EMPTY', "Puste", None)
        elif action == act_fill_zero:
            if self.common_wavenumbers is not None:
                self._set_cell_status(r, c, 'FILLED', "Zera", np.zeros_like(self.common_wavenumbers))
        elif action == act_fill_row:
            self._impute_cell(r, c, mode='row')
        elif action == act_fill_col:
            self._impute_cell(r, c, mode='col')
        elif action == act_fill_neigh:
            self._impute_cell(r, c, mode='neighbor')

    def _set_cell_status(self, r, c, status, text, y_vector):
        self.grid_data[(r, c)] = {'status': status, 'name': text, 'y': y_vector}
        item = self.table_grid.item(r, c)
        item.setText(f"({r},{c})\n{text}")
        item.setBackground(STATUS_COLORS.get(status, STATUS_COLORS['EMPTY']))

    def _impute_cell(self, r, c, mode='row'):
        n, m = self.spin_n.value(), self.spin_m.value()
        vectors = []

        if mode == 'row':
            for col in range(m):
                if col != c and self.grid_data.get((r, col), {}).get('y') is not None:
                    vectors.append(self.grid_data[(r, col)]['y'])
        elif mode == 'col':
            for row in range(n):
                if row != r and self.grid_data.get((row, c), {}).get('y') is not None:
                    vectors.append(self.grid_data[(row, c)]['y'])
        elif mode == 'neighbor':
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    if self.grid_data.get((nr, nc), {}).get('y') is not None:
                        vectors.append(self.grid_data[(nr, nc)]['y'])

        if vectors:
            mean_y = np.nanmean(vectors, axis=0)
            self._set_cell_status(r, c, 'FILLED', f"Śr. {mode[:4]}", mean_y)
        else:
            QMessageBox.warning(self, tr("app.warning"), "Brak poprawnych komórek referencyjnych do uśrednienia.")

    def _show_impute_menu(self):
        menu = QMenu(self)
        act_all_zeros = menu.addAction("Wypełnij wszystkie puste zerami (0.0)")
        act_all_row = menu.addAction("Wypełnij puste średnimi wierszy")
        act_all_col = menu.addAction("Wypełnij puste średnimi kolumn")

        action = menu.exec(self.cursor().pos())
        n, m = self.spin_n.value(), self.spin_m.value()
        if action == act_all_zeros and self.common_wavenumbers is not None:
            for r in range(n):
                for c in range(m):
                    if self.grid_data.get((r, c), {}).get('status') == 'EMPTY':
                        self._set_cell_status(r, c, 'FILLED', "Zera", np.zeros_like(self.common_wavenumbers))
        elif action == act_all_row:
            for r in range(n):
                for c in range(m):
                    if self.grid_data.get((r, c), {}).get('status') == 'EMPTY':
                        self._impute_cell(r, c, mode='row')
        elif action == act_all_col:
            for r in range(n):
                for c in range(m):
                    if self.grid_data.get((r, c), {}).get('status') == 'EMPTY':
                        self._impute_cell(r, c, mode='col')

    def _commit_tensor_3d(self):
        if self.common_wavenumbers is None:
            QMessageBox.warning(self, tr("app.warning"), "Siatka tensora nie zawiera żadnych załadowanych widm.")
            return

        n, m = self.spin_n.value(), self.spin_m.value()
        w = len(self.common_wavenumbers)
        tensor_3d = np.full((w, n, m), np.nan, dtype=np.float64)

        for r in range(n):
            for c in range(m):
                cell = self.grid_data.get((r, c), {})
                if cell.get('y') is not None:
                    tensor_3d[:, r, c] = cell['y']

        series_obj = SpectralSeries(
            name=self.ent_3d_name.text().strip() or "Tensor_3D",
            wavenumbers=self.common_wavenumbers,
            data=tensor_3d,
            axis_n_name="Rzędy (N)",
            axis_m_name="Kolumny (M)"
        )
        session.add_series(series_obj)
        QMessageBox.information(self, tr("app.info"), f"Pomyślnie utworzono tensor 3D '{series_obj.name}' ({w} × {n} × {m})!")
        self.accept()