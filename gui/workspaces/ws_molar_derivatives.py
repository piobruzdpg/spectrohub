"""
gui/workspaces/ws_molar_derivatives.py
Zintegrowana perspektywa: Widma Molowe i Różnicowe Widma Zmienności.
Obsługuje układy binarne (Woda + X) oraz trójskładnikowe (Woda + X + Y)
wraz z automatyczną generacją hierarchicznej struktury folderów, okien i tabel.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    QDoubleSpinBox, QCheckBox, QGroupBox, QMessageBox, QApplication,
    QTabWidget, QSplitter, QLineEdit, QGridLayout
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence

from core.session import session
from core.models import SpectrumData, DataTable
from algorithms.molar_derivatives import (
    calculate_water_density,
    compute_concentrations_and_wow,
    subtract_water,
    interpolate_spectrum_at_concentration,
    normalize_spectrum,
    compute_variation_spectrum_matrix,
    M_H2O
)

logger = logging.getLogger("SpectroHub")


class SeriesTableWidget(QTableWidget):
    """Tabela z ergonomiczną obsługą wklejania z Excela oraz selekcją wierszy."""

    data_pasted = Signal()

    def __init__(self, is_mixture: bool = False, parent=None):
        super().__init__(parent)
        self.is_mixture = is_mixture
        self._setup_table()

    def _setup_table(self):
        if self.is_mixture:
            headers = ["Użyj", "Widmo", "m(X) [g]", "m(Y) [g]", "m(H₂O) [g]", "d [g/cm³]", "C(X) [M]", "C(Y) [M]", "k_wow"]
        else:
            headers = ["Użyj", "Widmo", "m(subst) [g]", "m(H₂O) [g]", "d [g/cm³]", "C(subst) [M]", "k_wow"]

        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.setAlternatingRowColors(True)

        self.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a24;
                alternate-background-color: #22222f;
                gridline-color: #3d3d4e;
                color: #e0e0e0;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #2b2b3d;
                color: #48cae4;
                font-weight: bold;
                padding: 3px;
                border: 1px solid #3d3d4e;
            }
        """)

        self.setColumnWidth(0, 42)
        self.setColumnWidth(1, 160)
        for c in range(2, len(headers)):
            self.setColumnWidth(c, 78)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste) or (
            event.key() == Qt.Key_V and bool(event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        ):
            self.paste_from_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            return

        start_row = self.currentRow() if self.currentRow() >= 0 else 0
        start_col = self.currentColumn() if self.currentColumn() >= 0 else 2

        self.blockSignals(True)
        try:
            for r_offset, line in enumerate(lines):
                r = start_row + r_offset
                if r >= self.rowCount():
                    break

                cells = line.split("\t")
                if len(cells) == 1 and ";" in line:
                    cells = line.split(";")

                for c_offset, cell_val in enumerate(cells):
                    c = start_col + c_offset
                    if c >= self.columnCount():
                        break

                    if c in (0, 1) and start_col >= 2:
                        continue

                    val_clean = cell_val.strip().replace(" ", "").replace(",", ".")
                    item = self.item(r, c)
                    if item:
                        item.setText(val_clean)
                    else:
                        self.setItem(r, c, QTableWidgetItem(val_clean))
        finally:
            self.blockSignals(False)

        self.data_pasted.emit()


class MolarDerivativesPanel(QWidget):
    """Główny panel perspektywy: Widma Molowe i Różnicowe Widma Zmienności."""

    def __init__(self, main_window=None):
        super().__init__(main_window)
        self.main_window = main_window

        self.spectra_x: List[SpectrumData] = []
        self.spectra_y: List[SpectrumData] = []
        self.spectra_mix: List[SpectrumData] = []

        self.results: Dict[str, Any] = {}

        self._is_updating_table = False
        self._cursor_line = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. Konfiguracja Układu i Składników
        top_box = QGroupBox("1. Konfiguracja Układu i Składników")
        top_layout = QVBoxLayout(top_box)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(4)

        mode_row = QHBoxLayout()
        self.rb_binary = QRadioButton("Układ 2-składnikowy (Woda + X)")
        self.rb_ternary = QRadioButton("Układ 3-składnikowy (Woda + X + Y)")
        self.rb_binary.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_binary, 0)
        self.mode_group.addButton(self.rb_ternary, 1)
        mode_row.addWidget(self.rb_binary)
        mode_row.addWidget(self.rb_ternary)
        top_layout.addLayout(mode_row)

        phys_row = QHBoxLayout()
        phys_row.addWidget(QLabel("Widmo Wody:"))
        self.cb_water_spec = QComboBox()
        phys_row.addWidget(self.cb_water_spec, 2)

        phys_row.addWidget(QLabel("T [°C]:"))
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(0.0, 95.0)
        self.spn_temp.setValue(25.0)
        self.spn_temp.setDecimals(1)
        phys_row.addWidget(self.spn_temp, 1)

        phys_row.addWidget(QLabel("d₀ [g/cm³]:"))
        self.spn_d0 = QDoubleSpinBox()
        self.spn_d0.setRange(0.8, 1.2)
        self.spn_d0.setValue(calculate_water_density(25.0))
        self.spn_d0.setDecimals(6)
        phys_row.addWidget(self.spn_d0, 1)
        top_layout.addLayout(phys_row)

        # Siatka nazw i mas molowych
        comp_grid = QGridLayout()

        comp_grid.addWidget(QLabel("Nazwa składnika X:"), 0, 0)
        self.txt_name_x = QLineEdit("Substancja_X")
        comp_grid.addWidget(self.txt_name_x, 0, 1)

        comp_grid.addWidget(QLabel("M(X) [g/mol]:"), 0, 2)
        self.spn_mx = QDoubleSpinBox()
        self.spn_mx.setRange(1.0, 500000.0)
        self.spn_mx.setValue(60.06)
        self.spn_mx.setDecimals(3)
        comp_grid.addWidget(self.spn_mx, 0, 3)

        self.lbl_name_y = QLabel("Nazwa składnika Y:")
        self.txt_name_y = QLineEdit("Substancja_Y")
        self.lbl_my = QLabel("M(Y) [g/mol]:")
        self.spn_my = QDoubleSpinBox()
        self.spn_my.setRange(1.0, 500000.0)
        self.spn_my.setValue(75.07)
        self.spn_my.setDecimals(3)

        comp_grid.addWidget(self.lbl_name_y, 1, 0)
        comp_grid.addWidget(self.txt_name_y, 1, 1)
        comp_grid.addWidget(self.lbl_my, 1, 2)
        comp_grid.addWidget(self.spn_my, 1, 3)

        top_layout.addLayout(comp_grid)

        # Wybór składnika badanego w mieszaninie
        self.row_target = QHBoxLayout()
        self.lbl_target_substance = QLabel("Główny składnik badany:")
        self.cb_target_substance = QComboBox()
        self.row_target.addWidget(self.lbl_target_substance)
        self.row_target.addWidget(self.cb_target_substance, 1)
        top_layout.addLayout(self.row_target)

        main_layout.addWidget(top_box)

        # 2. Zakładki Serii Pomiarowych
        self.tabs_series = QTabWidget()

        # Tab X
        tab_x_widget = QWidget()
        tab_x_lay = QVBoxLayout(tab_x_widget)
        tab_x_lay.setContentsMargins(4, 4, 4, 4)
        self.table_x = SeriesTableWidget(is_mixture=False)
        btn_bar_x = self._create_series_toolbar("X", self.table_x)
        tab_x_lay.addLayout(btn_bar_x)
        tab_x_lay.addWidget(self.table_x)
        self.tabs_series.addTab(tab_x_widget, "🧪 Czysty X")

        # Tab Y
        self.tab_y_widget = QWidget()
        tab_y_lay = QVBoxLayout(self.tab_y_widget)
        tab_y_lay.setContentsMargins(4, 4, 4, 4)
        self.table_y = SeriesTableWidget(is_mixture=False)
        btn_bar_y = self._create_series_toolbar("Y", self.table_y)
        tab_y_lay.addLayout(btn_bar_y)
        tab_y_lay.addWidget(self.table_y)
        self.tabs_series.addTab(self.tab_y_widget, "🧪 Czysty Y")

        # Tab Mix
        self.tab_mix_widget = QWidget()
        tab_mix_lay = QVBoxLayout(self.tab_mix_widget)
        tab_mix_lay.setContentsMargins(4, 4, 4, 4)
        self.table_mix = SeriesTableWidget(is_mixture=True)
        btn_bar_mix = self._create_series_toolbar("MIX", self.table_mix)
        tab_mix_lay.addLayout(btn_bar_mix)
        tab_mix_lay.addWidget(self.table_mix)
        self.tabs_series.addTab(self.tab_mix_widget, "⚗️ Mieszanina X + Y")

        main_layout.addWidget(self.tabs_series, 3)

        # 3. Wybór podglądu etapu i modelu
        proc_box = QGroupBox("2. Inspekcja Etapów i Modelowanie")
        proc_layout = QVBoxLayout(proc_box)
        proc_layout.setContentsMargins(6, 6, 6, 6)

        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Podgląd etapu na wykresie:"))
        self.cb_view_stage = QComboBox()
        stage_row.addWidget(self.cb_view_stage, 3)

        stage_row.addWidget(QLabel("Model fitowania:"))
        self.cb_fit_model = QComboBox()
        self.cb_fit_model.addItems(["Liniowy (dε/dC = const)", "Kwadratowy (styczna C→0)"])
        stage_row.addWidget(self.cb_fit_model, 2)
        proc_layout.addLayout(stage_row)

        main_layout.addWidget(proc_box)

        # 4. Eksport zorganizowanej struktury
        export_box = QGroupBox("3. Eksport Struktury Wyników do Projektu")
        exp_layout = QVBoxLayout(export_box)
        exp_layout.setContentsMargins(6, 6, 6, 6)
        exp_layout.setSpacing(4)

        self.btn_export_all_structure = QPushButton("📁 Utwórz kompletną strukturę folderów i okien...")
        self.btn_export_all_structure.setStyleSheet("""
            QPushButton {
                background-color: #2b580c; 
                color: #ffffff; 
                font-weight: bold; 
                padding: 7px;
                border: 1px solid #639a24;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3e7a12; }
        """)
        exp_layout.addWidget(self.btn_export_all_structure)

        main_layout.addWidget(export_box)

        self.lbl_status = QLabel("Wprowadź dane masowe i gęstości dla serii.")
        self.lbl_status.setStyleSheet("color: #a8dadc; font-size: 11px;")
        main_layout.addWidget(self.lbl_status)

        self._update_component_names()
        self._update_mode_visibility()

    def _create_series_toolbar(self, series_key: str, table: SeriesTableWidget) -> QHBoxLayout:
        bar = QHBoxLayout()
        btn_tree = QPushButton("📥 Wstaw zaznaczone z drzewa")
        btn_tree.setToolTip("Wstawia widma zaznaczone aktualnie w drzewie projektu")
        btn_tree.setStyleSheet("background-color: #2b3a4a; font-weight: bold;")
        btn_tree.clicked.connect(lambda: self._add_selected_from_tree(series_key))
        bar.addWidget(btn_tree)

        btn_canvas = QPushButton("👁️ Wstaw widoczne")
        btn_canvas.setToolTip("Wstawia widma widoczne aktualnie na wykresie")
        btn_canvas.clicked.connect(lambda: self._add_visible_from_canvas(series_key))
        bar.addWidget(btn_canvas)

        btn_paste = QPushButton("📋 Wklej dane (Ctrl+V)")
        btn_paste.clicked.connect(table.paste_from_clipboard)
        bar.addWidget(btn_paste)

        btn_clear = QPushButton("🗑️ Wyczyść")
        btn_clear.clicked.connect(lambda: self._clear_series(series_key))
        bar.addWidget(btn_clear)

        bar.addStretch()
        return bar

    def _connect_signals(self):
        self.mode_group.idClicked.connect(self._on_mode_changed)
        self.spn_temp.valueChanged.connect(self._on_temp_changed)
        self.spn_mx.valueChanged.connect(self._recalculate_and_run)
        self.spn_my.valueChanged.connect(self._recalculate_and_run)

        self.txt_name_x.textChanged.connect(self._update_component_names)
        self.txt_name_y.textChanged.connect(self._update_component_names)

        self.cb_water_spec.currentIndexChanged.connect(self._recalculate_and_run)
        self.cb_target_substance.currentIndexChanged.connect(self._recalculate_and_run)
        self.cb_fit_model.currentIndexChanged.connect(self._recalculate_and_run)
        self.cb_view_stage.currentIndexChanged.connect(self._render_current_stage)

        for tbl in (self.table_x, self.table_y, self.table_mix):
            tbl.data_pasted.connect(self._recalculate_and_run)
            tbl.cellChanged.connect(self._on_table_cell_changed)

        self.btn_export_all_structure.clicked.connect(self._export_full_hierarchy)

        if self.main_window and hasattr(self.main_window, 'canvas_2d'):
            self.main_window.canvas_2d.canvas.mpl_connect('button_press_event', self._on_canvas_clicked)

    def _clear_cursor_line(self):
        if self._cursor_line is not None:
            try:
                self._cursor_line.remove()
            except Exception:
                pass
            self._cursor_line = None

    def _update_component_names(self):
        """Aktualizuje etykiety zakładek, podglądów i opcji na podstawie wpisanych nazw."""
        name_x = self.txt_name_x.text().strip() or "X"
        name_y = self.txt_name_y.text().strip() or "Y"

        self.tabs_series.setTabText(0, f"🧪 Czysty {name_x}")
        self.tabs_series.setTabText(1, f"🧪 Czysty {name_y}")
        self.tabs_series.setTabText(2, f"⚗️ Mieszanina {name_x} + {name_y}")

        self.cb_target_substance.blockSignals(True)
        cur_idx = self.cb_target_substance.currentIndex()
        self.cb_target_substance.clear()
        self.cb_target_substance.addItem(f"Składnik {name_x} (zaburzany przez {name_y})")
        self.cb_target_substance.addItem(f"Składnik {name_y} (zaburzany przez {name_x})")
        if cur_idx >= 0 and cur_idx < self.cb_target_substance.count():
            self.cb_target_substance.setCurrentIndex(cur_idx)
        self.cb_target_substance.blockSignals(False)

        self._populate_view_stages()

    def _on_mode_changed(self, mode_id: int):
        self._update_mode_visibility()
        self._update_component_names()
        self._recalculate_and_run()

    def _update_mode_visibility(self):
        is_ternary = self.rb_ternary.isChecked()
        self.tabs_series.setTabVisible(1, is_ternary)
        self.tabs_series.setTabVisible(2, is_ternary)

        self.lbl_name_y.setVisible(is_ternary)
        self.txt_name_y.setVisible(is_ternary)
        self.lbl_my.setVisible(is_ternary)
        self.spn_my.setVisible(is_ternary)

        self.lbl_target_substance.setVisible(is_ternary)
        self.cb_target_substance.setVisible(is_ternary)

    def _on_temp_changed(self, val: float):
        d0 = calculate_water_density(val)
        self.spn_d0.blockSignals(True)
        self.spn_d0.setValue(d0)
        self.spn_d0.blockSignals(False)
        self._recalculate_and_run()

    def refresh_water_list(self):
        cur_water = self.cb_water_spec.currentText()
        self.cb_water_spec.blockSignals(True)
        self.cb_water_spec.clear()

        all_specs = list(session.spectra.keys()) + list(session.atm_spectra.keys())
        if session.active_window:
            all_specs.extend(list(session.active_window.spectra.keys()))
        all_specs = sorted(list(set(all_specs)))

        for s_name in all_specs:
            self.cb_water_spec.addItem(s_name)

        if cur_water and cur_water in all_specs:
            self.cb_water_spec.setCurrentText(cur_water)
        else:
            for s_name in all_specs:
                nl = s_name.lower()
                if "h2o" in nl or "water" in nl or "woda" in nl:
                    self.cb_water_spec.setCurrentText(s_name)
                    break
        self.cb_water_spec.blockSignals(False)

    def refresh_view(self):
        if self.main_window and hasattr(self.main_window, 'canvas_2d'):
            self.main_window.canvas_2d.interaction_mode = "molar"
        self.refresh_water_list()
        self._render_current_stage()

    def hideEvent(self, event):
        if self.main_window and hasattr(self.main_window, 'canvas_2d'):
            self.main_window.canvas_2d.interaction_mode = "normal"
        super().hideEvent(event)

    def _populate_view_stages(self):
        cur_code = self.cb_view_stage.currentData()
        self.cb_view_stage.blockSignals(True)
        self.cb_view_stage.clear()

        is_ternary = self.rb_ternary.isChecked()
        nx = self.txt_name_x.text().strip() or "X"
        ny = self.txt_name_y.text().strip() or "Y"

        if not is_ternary:
            items = [
                (f"1. Surowy {nx}", "x_raw"),
                (f"2. Wyizolowany {nx} (-Woda)", "x_iso"),
                (f"3. Molowy {nx} (A/C)", "x_molar"),
                (f"4. Widmo Zmienności {nx} (dε/dC)", "x_variation")
            ]
        else:
            items = [
                (f"🧪 Czysty {nx}: Wyizolowany (-Woda)", "x_iso"),
                (f"🧪 Czysty {nx}: Molowy (A/C)", "x_molar"),
                (f"🧪 Czysty {nx}: Widmo Zmienności", "x_variation"),
                (f"🧪 Czysty {ny}: Wyizolowany (-Woda)", "y_iso"),
                (f"🧪 Czysty {ny}: Molowy (A/C)", "y_molar"),
                (f"🧪 Czysty {ny}: Widmo Zmienności", "y_variation"),
                (f"⚗️ Mieszanina: Surowa", "mix_raw"),
                (f"⚗️ Mieszanina: Po odjęciu Wody", "mix_no_water"),
                (f"⚗️ Mieszanina: Wyizolowany składnik (-Woda, -Zaburzacz)", "mix_iso"),
                (f"⚗️ Mieszanina: Molowy badany składnik", "mix_molar"),
                (f"🌟 MIESZANINA: Różnicowe Widmo Zmienności", "mix_variation")
            ]

        for label, code in items:
            self.cb_view_stage.addItem(label, code)

        idx = -1
        if cur_code:
            for i in range(self.cb_view_stage.count()):
                if self.cb_view_stage.itemData(i) == cur_code:
                    idx = i
                    break

        if idx >= 0:
            self.cb_view_stage.setCurrentIndex(idx)
        else:
            self.cb_view_stage.setCurrentIndex(self.cb_view_stage.count() - 1)

        self.cb_view_stage.blockSignals(False)

    def _add_selected_from_tree(self, series_key: str):
        if not self.main_window or not hasattr(self.main_window, 'project_tree'):
            return
        selected_items = self.main_window.project_tree.selectedItems()
        specs = []
        seen_names = set()

        for itm in selected_items:
            itype = itm.data(0, Qt.UserRole + 1)
            iname = itm.data(0, Qt.UserRole)
            if itype in ["spectrum", "window_spectrum"] and iname and iname not in seen_names:
                s = session.get_spectrum(iname)
                if s is not None:
                    seen_names.add(iname)
                    specs.append(s)

        if not specs:
            QMessageBox.information(self, "Wybór", "Zaznacz widma w drzewie projektu po lewej stronie.")
            return

        self._populate_series_table(series_key, specs)

    def _add_visible_from_canvas(self, series_key: str):
        specs = session.get_display_spectra()
        if not specs:
            QMessageBox.information(self, "Wybór", "Brak widocznych widm na wykresie.")
            return
        self._populate_series_table(series_key, specs)

    def _clear_series(self, series_key: str):
        if series_key == "X":
            self.table_x.setRowCount(0)
            self.spectra_x.clear()
        elif series_key == "Y":
            self.table_y.setRowCount(0)
            self.spectra_y.clear()
        else:
            self.table_mix.setRowCount(0)
            self.spectra_mix.clear()
        self._recalculate_and_run()

    def _populate_series_table(self, series_key: str, specs: List[SpectrumData]):
        if series_key == "X":
            tbl = self.table_x
            self.spectra_x = [s.clone() for s in specs]
        elif series_key == "Y":
            tbl = self.table_y
            self.spectra_y = [s.clone() for s in specs]
        else:
            tbl = self.table_mix
            self.spectra_mix = [s.clone() for s in specs]

        tbl.blockSignals(True)
        tbl.setRowCount(len(specs))
        for r, s in enumerate(specs):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked)
            tbl.setItem(r, 0, chk)

            name_item = QTableWidgetItem(s.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            name_item.setForeground(QColor("#48cae4"))
            tbl.setItem(r, 1, name_item)

            for c in range(2, tbl.columnCount()):
                it = tbl.item(r, c)
                if not it:
                    tbl.setItem(r, c, QTableWidgetItem("0.0"))

        tbl.blockSignals(False)
        self._recalculate_and_run()

    def _on_table_cell_changed(self, row: int, col: int):
        sender = self.sender()
        if self._is_updating_table or not isinstance(sender, SeriesTableWidget):
            return

        is_mix = sender.is_mixture
        calc_cols = (6, 7, 8) if is_mix else (5, 6)
        if col in calc_cols or col == 1:
            return

        self._recalculate_table_row(sender, row)
        self._run_calculations()

    def _recalculate_table_row(self, tbl: SeriesTableWidget, row: int):
        def val(c: int) -> float:
            it = tbl.item(row, c)
            if not it or not it.text().strip():
                return 0.0
            try:
                return float(it.text().strip().replace(' ', '').replace(',', '.'))
            except ValueError:
                return 0.0

        d0 = self.spn_d0.value()
        tbl.blockSignals(True)
        try:
            if not tbl.is_mixture:
                m_sub = val(2)
                m_w = val(3)
                d = val(4)
                m_mol = self.spn_mx.value() if tbl == self.table_x else self.spn_my.value()

                if (m_sub + m_w) > 0 and d > 0 and m_mol > 0:
                    res = compute_concentrations_and_wow(
                        mass_solute_x=m_sub, mass_water=m_w, density_solution=d,
                        molar_mass_x=m_mol, density_pure_water=d0
                    )
                    c_it = tbl.item(row, 5) or QTableWidgetItem()
                    c_it.setText(f"{res['C_X']:.4f}")
                    c_it.setForeground(QColor("#a8dadc"))
                    tbl.setItem(row, 5, c_it)

                    wow_it = tbl.item(row, 6) or QTableWidgetItem()
                    wow_it.setText(f"{res['k_wow']:.5f}")
                    wow_it.setForeground(QColor("#ffd166"))
                    tbl.setItem(row, 6, wow_it)
            else:
                m_x = val(2)
                m_y = val(3)
                m_w = val(4)
                d = val(5)
                mx_mol = self.spn_mx.value()
                my_mol = self.spn_my.value()

                if (m_x + m_y + m_w) > 0 and d > 0 and mx_mol > 0 and my_mol > 0:
                    res = compute_concentrations_and_wow(
                        mass_solute_x=m_x, mass_water=m_w, density_solution=d,
                        molar_mass_x=mx_mol, density_pure_water=d0,
                        mass_solute_y=m_y, molar_mass_y=my_mol
                    )
                    cx_it = tbl.item(row, 6) or QTableWidgetItem()
                    cx_it.setText(f"{res['C_X']:.4f}")
                    cx_it.setForeground(QColor("#a8dadc"))
                    tbl.setItem(row, 6, cx_it)

                    cy_it = tbl.item(row, 7) or QTableWidgetItem()
                    cy_it.setText(f"{res['C_Y']:.4f}")
                    cy_it.setForeground(QColor("#a8dadc"))
                    tbl.setItem(row, 7, cy_it)

                    wow_it = tbl.item(row, 8) or QTableWidgetItem()
                    wow_it.setText(f"{res['k_wow']:.5f}")
                    wow_it.setForeground(QColor("#ffd166"))
                    tbl.setItem(row, 8, wow_it)
        finally:
            tbl.blockSignals(False)

    def _recalculate_and_run(self):
        self._is_updating_table = True
        try:
            for tbl in (self.table_x, self.table_y, self.table_mix):
                for r in range(tbl.rowCount()):
                    self._recalculate_table_row(tbl, r)
        finally:
            self._is_updating_table = False
        self._run_calculations()

    def _run_calculations(self):
        water_name = self.cb_water_spec.currentText()
        water_spec = session.get_spectrum(water_name) if water_name else None
        fit_model = "quadratic" if self.cb_fit_model.currentIndex() == 1 else "linear"
        is_ternary = self.rb_ternary.isChecked()

        self.results.clear()

        def get_series_data(tbl: SeriesTableWidget, specs: List[SpectrumData], is_mix: bool = False):
            active_specs = []
            concs_x, concs_y, k_wows = [], [], []
            for r in range(tbl.rowCount()):
                chk = tbl.item(r, 0)
                if chk and chk.checkState() == Qt.CheckState.Checked and r < len(specs):
                    try:
                        if not is_mix:
                            c = float(tbl.item(r, 5).text().replace(',', '.'))
                            wow = float(tbl.item(r, 6).text().replace(',', '.'))
                            if c > 0 and wow > 0:
                                active_specs.append(specs[r])
                                concs_x.append(c)
                                k_wows.append(wow)
                        else:
                            cx = float(tbl.item(r, 6).text().replace(',', '.'))
                            cy = float(tbl.item(r, 7).text().replace(',', '.'))
                            wow = float(tbl.item(r, 8).text().replace(',', '.'))
                            if cx > 0 and cy > 0 and wow > 0:
                                active_specs.append(specs[r])
                                concs_x.append(cx)
                                concs_y.append(cy)
                                k_wows.append(wow)
                    except (ValueError, AttributeError):
                        continue
            return active_specs, np.array(concs_x), np.array(concs_y), np.array(k_wows)

        # 1. Czysty X
        act_x, c_x, _, wow_x = get_series_data(self.table_x, self.spectra_x, is_mix=False)
        if act_x and water_spec:
            ref_wn = act_x[0].wavenumbers
            x_iso = [subtract_water(s.wavenumbers, s.intensities, water_spec.wavenumbers, water_spec.intensities, wow_x[i])
                     for i, s in enumerate(act_x)]
            mat_x_iso = np.column_stack(x_iso)
            mat_x_molar = mat_x_iso / c_x

            p1_x, p0_x, p1_err_x, rmse_x, r2_x = compute_variation_spectrum_matrix(mat_x_molar, c_x, model=fit_model)

            self.results["x"] = {
                "wn": ref_wn, "concs": c_x, "raw": act_x,
                "iso_mat": mat_x_iso, "molar_mat": mat_x_molar,
                "p1": p1_x, "p0": p0_x, "p1_err": p1_err_x, "rmse": rmse_x, "r2": r2_x
            }

        # 2. Czysty Y
        if is_ternary:
            act_y, c_y, _, wow_y = get_series_data(self.table_y, self.spectra_y, is_mix=False)
            if act_y and water_spec:
                ref_wn = act_y[0].wavenumbers
                y_iso = [subtract_water(s.wavenumbers, s.intensities, water_spec.wavenumbers, water_spec.intensities, wow_y[i])
                         for i, s in enumerate(act_y)]
                mat_y_iso = np.column_stack(y_iso)
                mat_y_molar = mat_y_iso / c_y

                p1_y, p0_y, p1_err_y, rmse_y, r2_y = compute_variation_spectrum_matrix(mat_y_molar, c_y, model=fit_model)

                self.results["y"] = {
                    "wn": ref_wn, "concs": c_y, "raw": act_y,
                    "iso_mat": mat_y_iso, "molar_mat": mat_y_molar,
                    "p1": p1_y, "p0": p0_y, "p1_err": p1_err_y, "rmse": rmse_y, "r2": r2_y
                }

            # 3. Mieszanina X + Y
            act_mix, c_mix_x, c_mix_y, wow_mix = get_series_data(self.table_mix, self.spectra_mix, is_mix=True)
            if act_mix and water_spec and "x" in self.results and "y" in self.results:
                ref_wn = act_mix[0].wavenumbers
                w_len = len(ref_wn)
                n_mix = len(act_mix)

                mix_no_water = [subtract_water(s.wavenumbers, s.intensities, water_spec.wavenumbers, water_spec.intensities, wow_mix[i])
                                for i, s in enumerate(act_mix)]
                mat_mix_nw = np.column_stack(mix_no_water)

                target_is_x = (self.cb_target_substance.currentIndex() == 0)
                mat_mix_iso = np.zeros((w_len, n_mix), dtype=np.float64)
                mat_mix_molar = np.zeros((w_len, n_mix), dtype=np.float64)

                nx = self.txt_name_x.text().strip() or "X"
                ny = self.txt_name_y.text().strip() or "Y"

                if target_is_x:
                    ref_y_concs = self.results["y"]["concs"]
                    ref_y_iso_mat = self.results["y"]["iso_mat"]

                    for i in range(n_mix):
                        cy_target = c_mix_y[i]
                        y_interp = interpolate_spectrum_at_concentration(ref_y_concs, ref_y_iso_mat, cy_target, method="linear")
                        x_in_mix_iso = mat_mix_nw[:, i] - y_interp
                        mat_mix_iso[:, i] = x_in_mix_iso
                        mat_mix_molar[:, i] = x_in_mix_iso / c_mix_x[i]

                    pert_concs = c_mix_y
                    pert_label = f"C({ny})"
                    target_label = nx
                    pert_name = ny
                else:
                    ref_x_concs = self.results["x"]["concs"]
                    ref_x_iso_mat = self.results["x"]["iso_mat"]

                    for i in range(n_mix):
                        cx_target = c_mix_x[i]
                        x_interp = interpolate_spectrum_at_concentration(ref_x_concs, ref_x_iso_mat, cx_target, method="linear")
                        y_in_mix_iso = mat_mix_nw[:, i] - x_interp
                        mat_mix_iso[:, i] = y_in_mix_iso
                        mat_mix_molar[:, i] = y_in_mix_iso / c_mix_y[i]

                    pert_concs = c_mix_x
                    pert_label = f"C({nx})"
                    target_label = ny
                    pert_name = nx

                p1_m, p0_m, p1_err_m, rmse_m, r2_m = compute_variation_spectrum_matrix(mat_mix_molar, pert_concs, model=fit_model)

                self.results["mix"] = {
                    "wn": ref_wn, "concs_x": c_mix_x, "concs_y": c_mix_y, "pert_concs": pert_concs,
                    "pert_label": pert_label, "target_label": target_label, "pert_name": pert_name,
                    "raw": act_mix, "no_water_mat": mat_mix_nw,
                    "iso_mat": mat_mix_iso, "molar_mat": mat_mix_molar,
                    "p1": p1_m, "p0": p0_m, "p1_err": p1_err_m, "rmse": rmse_m, "r2": r2_m
                }

        self._render_current_stage()

    def _render_current_stage(self):
        if not self.main_window or not hasattr(self.main_window, 'canvas_2d'):
            return

        canvas = self.main_window.canvas_2d
        self._clear_cursor_line()
        canvas.ax_main.cla()
        canvas.ax_sub.cla()
        canvas.ax_sub_top.cla()
        canvas._apply_styling()

        stage = self.cb_view_stage.currentData()
        if not stage:
            return

        wn = None
        y_label = "Absorbancja"
        nx = self.txt_name_x.text().strip() or "X"
        ny = self.txt_name_y.text().strip() or "Y"

        if stage.startswith("x_") and "x" in self.results:
            d = self.results["x"]
            wn = d["wn"]
            if stage == "x_raw":
                for s in d["raw"]:
                    canvas.ax_main.plot(s.wavenumbers, s.intensities, label=s.name, lw=1.2, alpha=0.8)
                y_label = f"Absorbancja A (Surowy {nx})"
            elif stage == "x_iso":
                for i, c in enumerate(d["concs"]):
                    canvas.ax_main.plot(wn, d["iso_mat"][:, i], label=f"C={c:.3f}M", lw=1.2, alpha=0.85)
                y_label = f"Absorbancja A (Wyizolowany {nx})"
            elif stage == "x_molar":
                for i, c in enumerate(d["concs"]):
                    canvas.ax_main.plot(wn, d["molar_mat"][:, i], label=f"C={c:.3f}M", lw=1.2, alpha=0.85)
                y_label = f"Pozorna absorbancja molowa ε({nx})"
            elif stage == "x_variation":
                canvas.ax_main.plot(wn, d["p1"], color="#00ffcc", lw=2.0, label=f"Różnicowe Widmo Zmienności {nx} (dε/dC)")
                canvas.ax_main.plot(wn, d["p0"], color="#ffb703", lw=1.4, ls="--", label="Graniczne Widmo Molowe ε₀")
                canvas.ax_main.axhline(0, color="#ee6c4d", lw=0.8, ls=":")
                y_label = "Zmienność dε/dC / Graniczne ε₀"

        elif stage.startswith("y_") and "y" in self.results:
            d = self.results["y"]
            wn = d["wn"]
            if stage == "y_iso":
                for i, c in enumerate(d["concs"]):
                    canvas.ax_main.plot(wn, d["iso_mat"][:, i], label=f"C={c:.3f}M", lw=1.2, alpha=0.85)
                y_label = f"Absorbancja A (Wyizolowany {ny})"
            elif stage == "y_molar":
                for i, c in enumerate(d["concs"]):
                    canvas.ax_main.plot(wn, d["molar_mat"][:, i], label=f"C={c:.3f}M", lw=1.2, alpha=0.85)
                y_label = f"Pozorna absorbancja molowa ε({ny})"
            elif stage == "y_variation":
                canvas.ax_main.plot(wn, d["p1"], color="#00ffcc", lw=2.0, label=f"Różnicowe Widmo Zmienności {ny} (dε/dC)")
                canvas.ax_main.plot(wn, d["p0"], color="#ffb703", lw=1.4, ls="--", label="Graniczne Widmo Molowe ε₀")
                canvas.ax_main.axhline(0, color="#ee6c4d", lw=0.8, ls=":")
                y_label = "Zmienność dε/dC / Graniczne ε₀"

        elif stage.startswith("mix_") and "mix" in self.results:
            d = self.results["mix"]
            wn = d["wn"]
            if stage == "mix_raw":
                for s in d["raw"]:
                    canvas.ax_main.plot(s.wavenumbers, s.intensities, label=s.name, lw=1.2, alpha=0.8)
                y_label = "Absorbancja A (Surowa Mieszanina)"
            elif stage == "mix_no_water":
                for i in range(d["no_water_mat"].shape[1]):
                    canvas.ax_main.plot(wn, d["no_water_mat"][:, i], label=f"Próbka {i+1}", lw=1.2, alpha=0.85)
                y_label = "Absorbancja A (Mieszanina bez wody)"
            elif stage == "mix_iso":
                for i in range(d["iso_mat"].shape[1]):
                    canvas.ax_main.plot(wn, d["iso_mat"][:, i], label=f"Próbka {i+1}", lw=1.2, alpha=0.85)
                y_label = f"Absorbancja A (Wyizolowany {d['target_label']})"
            elif stage == "mix_molar":
                for i in range(d["molar_mat"].shape[1]):
                    canvas.ax_main.plot(wn, d["molar_mat"][:, i], label=f"{d['pert_label']}={d['pert_concs'][i]:.3f}M", lw=1.2, alpha=0.85)
                y_label = f"Pozorna absorbancja molowa ε({d['target_label']}) w mieszaninie"
            elif stage == "mix_variation":
                canvas.ax_main.plot(wn, d["p1"], color="#00ffcc", lw=2.0, label=f"Widmo Zmienności {d['target_label']} pod wpływem {d['pert_name']}")
                canvas.ax_main.plot(wn, d["p0"], color="#ffb703", lw=1.4, ls="--", label=f"Graniczne Widmo Molowe ({d['pert_label']}→0)")
                canvas.ax_main.axhline(0, color="#ee6c4d", lw=0.8, ls=":")
                y_label = "Zmienność dε/dC / Graniczne ε₀"

        canvas.ax_main.set_ylabel(y_label, color="#dcdcdc")
        if wn is not None and len(wn) > 0:
            canvas.ax_main.set_xlim(np.max(wn), np.min(wn))

        handles, labels = canvas.ax_main.get_legend_handles_labels()
        if handles:
            canvas.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                                  labelcolor='#ffffff', fontsize=8)
        canvas.canvas.draw_idle()

    def _on_canvas_clicked(self, event):
        if event.button != 1 or event.inaxes is None:
            return

        canvas = self.main_window.canvas_2d
        if event.inaxes != canvas.ax_main or event.xdata is None:
            return

        stage = self.cb_view_stage.currentData()
        if not stage:
            return

        chosen_wn = float(event.xdata)
        nx = self.txt_name_x.text().strip() or "X"
        ny = self.txt_name_y.text().strip() or "Y"

        if stage.startswith("x_") and "x" in self.results:
            d = self.results["x"]
            wn = d["wn"]
            c_points = d["concs"]
            c_label = f"C({nx}) [mol/dm³]"
            mat = d["molar_mat"] if "molar" in stage or "variation" in stage else d["iso_mat"]
            y_label_text = f"ε({nx})" if "molar" in stage or "variation" in stage else f"A({nx})"
        elif stage.startswith("y_") and "y" in self.results:
            d = self.results["y"]
            wn = d["wn"]
            c_points = d["concs"]
            c_label = f"C({ny}) [mol/dm³]"
            mat = d["molar_mat"] if "molar" in stage or "variation" in stage else d["iso_mat"]
            y_label_text = f"ε({ny})" if "molar" in stage or "variation" in stage else f"A({ny})"
        elif stage.startswith("mix_") and "mix" in self.results:
            d = self.results["mix"]
            wn = d["wn"]
            c_points = d["pert_concs"]
            c_label = f"{d['pert_label']} [mol/dm³]"
            mat = d["molar_mat"] if "molar" in stage or "variation" in stage else d["iso_mat"]
            y_label_text = f"ε({d['target_label']}) w mieszaninie" if "molar" in stage or "variation" in stage else "A"
        else:
            return

        if len(c_points) < 2 or mat is None:
            return

        idx = int(np.abs(wn - chosen_wn).argmin())
        exact_wn = float(wn[idx])
        y_points = mat[idx, :]

        self._clear_cursor_line()
        self._cursor_line = canvas.ax_main.axvline(exact_wn, color="#00e5ff", linestyle="--", lw=1.2, zorder=6)

        canvas.ax_sub.cla()
        canvas.ax_sub_top.cla()
        canvas._apply_styling()

        canvas.ax_sub.plot(c_points, y_points, 'o', color="#48cae4", markersize=6.0,
                           label=f"Punkty (@ {exact_wn:.1f} cm⁻¹)", zorder=4)

        max_c = np.max(c_points) if len(c_points) > 0 else 1.0
        c_dense = np.linspace(0.0, max_c * 1.15, 100)
        fit_model = self.cb_fit_model.currentIndex()

        if fit_model == 1 and len(c_points) >= 3:
            p, cov = np.polyfit(c_points, y_points, 2, cov=True)
            y_curve = np.polyval(p, c_dense)
            y_fit = np.polyval(p, c_points)
            ss_res = np.sum((y_points - y_fit)**2)
            rmse = np.sqrt(ss_res / max(1, len(c_points) - 3))
            ss_tot = np.sum((y_points - np.mean(y_points))**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

            slope = p[1]
            slope_err = np.sqrt(cov[1, 1]) if cov is not None else 0.0
            intercept = p[2]

            canvas.ax_sub.plot(c_dense, y_curve, '-', color="#f72585", lw=1.5,
                              label=f"Fit kwadratowy (RMSE = {rmse:.4f}, R² = {np.clip(r2, 0, 1):.3f})")
            y_tangent = slope * c_dense + intercept
            canvas.ax_sub.plot(c_dense, y_tangent, ':', color="#ffd166", lw=1.4,
                              label=f"Styczna C→0: dε/dC = {slope:.4f} ± {slope_err:.4f} | ε₀ = {intercept:.4f}")
        else:
            p, cov = np.polyfit(c_points, y_points, 1, cov=True)
            y_line = np.polyval(p, c_dense)
            y_fit = np.polyval(p, c_points)
            ss_res = np.sum((y_points - y_fit)**2)
            rmse = np.sqrt(ss_res / max(1, len(c_points) - 2))
            ss_tot = np.sum((y_points - np.mean(y_points))**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

            slope = p[0]
            slope_err = np.sqrt(cov[0, 0]) if cov is not None else 0.0
            intercept = p[1]

            canvas.ax_sub.plot(c_dense, y_line, '-', color="#00ffcc", lw=1.6,
                              label=f"Prosta: dε/dC = {slope:.4f} ± {slope_err:.4f} | ε₀ = {intercept:.4f} | RMSE = {rmse:.4f}")

        canvas.ax_sub.set_xlabel(c_label, color="#dcdcdc", fontsize=9)
        canvas.ax_sub.set_ylabel(y_label_text, color="#dcdcdc", fontsize=9)
        canvas.ax_sub.legend(loc="best", facecolor="#2b2b36", edgecolor="#4a4a5a",
                             labelcolor="#ffffff", fontsize=8)
        canvas.canvas.draw_idle()

    # --- ZAAWANSOWANY EKSPORT ZORGANIZOWANEJ HIERARCHII FOLDERÓW ---
    def _export_full_hierarchy(self):
        """Tworzy kompletną strukturę folderów, podfolderów, okien roboczych i tabel parametrów."""
        is_ternary = self.rb_ternary.isChecked()
        nx = self.txt_name_x.text().strip() or "X"
        ny = self.txt_name_y.text().strip() or "Y"

        if not is_ternary:
            if "x" not in self.results:
                QMessageBox.warning(self, "Brak danych", f"Najpierw uzupełnij dane i przelicz serię dla {nx}.")
                return

            root_folder_name = f"Analiza Molowa — {nx}"
            root_f_id = session.add_folder(name=root_folder_name)

            self._create_component_export(self.results["x"], nx, parent_folder_id=root_f_id)
            QMessageBox.information(self, "Sukces", f"Pomyślnie utworzono folder z wynikami:\n📁 {root_folder_name}")

        else:
            if "x" not in self.results or "y" not in self.results or "mix" not in self.results:
                QMessageBox.warning(self, "Brak danych", "Do pełnego eksportu wymagane są przeliczone serie X, Y oraz Mieszaniny.")
                return

            root_folder_name = f"Analiza Trójskładnikowa — {nx} + {ny}"
            root_f_id = session.add_folder(name=root_folder_name)

            # Podfolder 1: Czysty X
            f1_id = session.add_folder(name=f"1. Czysty {nx}", parent_folder_id=root_f_id)
            self._create_component_export(self.results["x"], nx, parent_folder_id=f1_id)

            # Podfolder 2: Czysty Y
            f2_id = session.add_folder(name=f"2. Czysty {ny}", parent_folder_id=root_f_id)
            self._create_component_export(self.results["y"], ny, parent_folder_id=f2_id)

            # Podfolder 3: Mieszanina
            f3_id = session.add_folder(name=f"3. Mieszanina {nx} + {ny}", parent_folder_id=root_f_id)
            self._create_mixture_export(self.results["mix"], nx, ny, parent_folder_id=f3_id)

            QMessageBox.information(self, "Sukces", f"Pomyślnie utworzono pełną strukturę trójskładnikową:\n📁 {root_folder_name}")

    def _create_component_export(self, d: Dict[str, Any], comp_name: str, parent_folder_id: str):
        wn = d["wn"]
        concs = d["concs"]

        # Okno 1: Wyizolowane
        w1_id = session.add_window(name=f"{comp_name} — Wyizolowane (-Woda)", folder_id=parent_folder_id)
        for i in range(d["iso_mat"].shape[1]):
            s = SpectrumData(
                wavenumbers=wn.copy(), intensities=d["iso_mat"][:, i].copy(),
                name=f"{comp_name}_izol_{i+1}_(C={concs[i]:.3f}M)",
                file_path="", file_type="processed"
            )
            session.add_spectrum_to_window(w1_id, s)

        # Okno 2: Molowe
        w2_id = session.add_window(name=f"{comp_name} — Widma Molowe (A/C)", folder_id=parent_folder_id)
        for i in range(d["molar_mat"].shape[1]):
            s = SpectrumData(
                wavenumbers=wn.copy(), intensities=d["molar_mat"][:, i].copy(),
                name=f"{comp_name}_molowe_{i+1}_(C={concs[i]:.3f}M)",
                file_path="", file_type="processed"
            )
            session.add_spectrum_to_window(w2_id, s)

        # Okno 3: Widmo Zmienności i Graniczne
        w3_id = session.add_window(name=f"{comp_name} — Zmienność i Graniczne", folder_id=parent_folder_id)
        s_var = SpectrumData(
            wavenumbers=wn.copy(), intensities=d["p1"].copy(),
            name=f"Widmo_Zmienności_{comp_name}_(dε/dC)",
            file_path="", file_type="processed"
        )
        session.add_spectrum_to_window(w3_id, s_var)
        s_int = SpectrumData(
            wavenumbers=wn.copy(), intensities=d["p0"].copy(),
            name=f"Graniczne_Molowe_{comp_name}_(ε₀)",
            file_path="", file_type="processed"
        )
        session.add_spectrum_to_window(w3_id, s_int)

        # Tabela parametrów fitu
        tbl = DataTable(
            table_id="",
            name=f"Fit — {comp_name}",
            x_values=wn,
            y_columns={
                "d_eps_dC": d["p1"],
                "d_eps_dC_err": d["p1_err"],
                "eps_0": d["p0"],
                "RMSE": d["rmse"],
                "R2": d["r2"]
            },
            x_label="Liczba falowa [cm⁻¹]",
            y_label="Wartość parametru",
            folder_id=parent_folder_id
        )
        session.add_table(tbl, make_active=False)

    def _create_mixture_export(self, d: Dict[str, Any], nx: str, ny: str, parent_folder_id: str):
        wn = d["wn"]
        target_name = d["target_label"]
        pert_name = d["pert_name"]
        pert_concs = d["pert_concs"]

        # Okno 1: Bez wody
        w1_id = session.add_window(name=f"Mieszanina — Bez Wody", folder_id=parent_folder_id)
        for i in range(d["no_water_mat"].shape[1]):
            s = SpectrumData(
                wavenumbers=wn.copy(), intensities=d["no_water_mat"][:, i].copy(),
                name=f"Mix_noH2O_{i+1}", file_path="", file_type="processed"
            )
            session.add_spectrum_to_window(w1_id, s)

        # Okno 2: Wyizolowany badany składnik (-Woda, -Perturbant)
        w2_id = session.add_window(name=f"{target_name} w Mix — Wyizolowany (-Woda, -{pert_name})", folder_id=parent_folder_id)
        for i in range(d["iso_mat"].shape[1]):
            s = SpectrumData(
                wavenumbers=wn.copy(), intensities=d["iso_mat"][:, i].copy(),
                name=f"{target_name}_in_mix_izol_{i+1}", file_path="", file_type="processed"
            )
            session.add_spectrum_to_window(w2_id, s)

        # Okno 3: Molowy badany składnik
        w3_id = session.add_window(name=f"{target_name} w Mix — Molowy", folder_id=parent_folder_id)
        for i in range(d["molar_mat"].shape[1]):
            s = SpectrumData(
                wavenumbers=wn.copy(), intensities=d["molar_mat"][:, i].copy(),
                name=f"{target_name}_molowy_{i+1}_({pert_name}={pert_concs[i]:.3f}M)",
                file_path="", file_type="processed"
            )
            session.add_spectrum_to_window(w3_id, s)

        # Okno 4: Widmo Zmienności w mieszaninie
        w4_id = session.add_window(name=f"Widmo Zmienności ({target_name} vs {pert_name})", folder_id=parent_folder_id)
        s_var = SpectrumData(
            wavenumbers=wn.copy(), intensities=d["p1"].copy(),
            name=f"Widmo_Zmienności_{target_name}_pod_wpływem_{pert_name}",
            file_path="", file_type="processed"
        )
        session.add_spectrum_to_window(w4_id, s_var)
        s_int = SpectrumData(
            wavenumbers=wn.copy(), intensities=d["p0"].copy(),
            name=f"Graniczne_Molowe_{target_name}_(C_{pert_name}→0)",
            file_path="", file_type="processed"
        )
        session.add_spectrum_to_window(w4_id, s_int)

        # Tabela parametrów fitu w mieszaninie
        tbl = DataTable(
            table_id="",
            name=f"Fit — Mieszanina ({target_name} vs {pert_name})",
            x_values=wn,
            y_columns={
                "d_eps_dC": d["p1"],
                "d_eps_dC_err": d["p1_err"],
                "eps_0": d["p0"],
                "RMSE": d["rmse"],
                "R2": d["r2"]
            },
            x_label="Liczba falowa [cm⁻¹]",
            y_label="Wartość parametru",
            folder_id=parent_folder_id
        )
        session.add_table(tbl, make_active=False)

# =========================================================================
    # SERIALIZACJA I DESERIALIZACJA STANU (.SHUB)
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Zwraca słownik ze stanem całej perspektywy gotowy do zapisu w project.json."""
        def _serialize_table(tbl: SeriesTableWidget) -> list:
            rows = []
            for r in range(tbl.rowCount()):
                chk = tbl.item(r, 0)
                name_it = tbl.item(r, 1)
                cells = []
                for c in range(2, tbl.columnCount()):
                    it = tbl.item(r, c)
                    cells.append(it.text().strip() if it else "0.0")
                rows.append({
                    "checked": (chk.checkState() == Qt.CheckState.Checked) if chk else True,
                    "name": name_it.text().strip() if name_it else "",
                    "cells": cells
                })
            return rows

        return {
            "mode_id": 1 if self.rb_ternary.isChecked() else 0,
            "water_spectrum": self.cb_water_spec.currentText(),
            "temp": self.spn_temp.value(),
            "d0": self.spn_d0.value(),
            "name_x": self.txt_name_x.text().strip(),
            "name_y": self.txt_name_y.text().strip(),
            "m_x": self.spn_mx.value(),
            "m_y": self.spn_my.value(),
            "target_substance_idx": self.cb_target_substance.currentIndex(),
            "fit_model_idx": self.cb_fit_model.currentIndex(),
            "view_stage_code": self.cb_view_stage.currentData(),
            "table_x_rows": _serialize_table(self.table_x),
            "table_y_rows": _serialize_table(self.table_y),
            "table_mix_rows": _serialize_table(self.table_mix)
        }

    def set_state(self, state: Dict[str, Any]):
        """Odtwarza stan perspektywy po załadowaniu projektu z pliku .shub."""
        if not state:
            return

        self._is_updating_table = True
        try:
            # 1. Tryb układu
            mode_id = state.get("mode_id", 0)
            if mode_id == 1:
                self.rb_ternary.setChecked(True)
            else:
                self.rb_binary.setChecked(True)
            self._update_mode_visibility()

            # 2. Nazwy i masy molowe
            self.txt_name_x.setText(state.get("name_x", "Substancja_X"))
            self.txt_name_y.setText(state.get("name_y", "Substancja_Y"))
            self.spn_mx.setValue(state.get("m_x", 60.06))
            self.spn_my.setValue(state.get("m_y", 75.07))
            self._update_component_names()

            # 3. Parametry fizykochemiczne
            self.spn_temp.setValue(state.get("temp", 25.0))
            self.spn_d0.setValue(state.get("d0", calculate_water_density(25.0)))

            # 4. Widmo wody
            self.refresh_water_list()
            water_name = state.get("water_spectrum", "")
            if water_name:
                self.cb_water_spec.setCurrentText(water_name)

            # 5. Składnik badany i model fitu
            target_idx = state.get("target_substance_idx", 0)
            if 0 <= target_idx < self.cb_target_substance.count():
                self.cb_target_substance.setCurrentIndex(target_idx)

            fit_idx = state.get("fit_model_idx", 0)
            if 0 <= fit_idx < self.cb_fit_model.count():
                self.cb_fit_model.setCurrentIndex(fit_idx)

            # 6. Odtworzenie tabel i powiązanie z widmami
            self._deserialize_table(self.table_x, state.get("table_x_rows", []), "X")
            self._deserialize_table(self.table_y, state.get("table_y_rows", []), "Y")
            self._deserialize_table(self.table_mix, state.get("table_mix_rows", []), "MIX")

            # 7. Aktywny podgląd etapu
            self._populate_view_stages()
            stage_code = state.get("view_stage_code", "")
            if stage_code:
                for i in range(self.cb_view_stage.count()):
                    if self.cb_view_stage.itemData(i) == stage_code:
                        self.cb_view_stage.setCurrentIndex(i)
                        break
        finally:
            self._is_updating_table = False

        # 8. Przeliczenie i narysowanie wyników
        self._recalculate_and_run()

    def _deserialize_table(self, tbl: SeriesTableWidget, rows_data: list, series_key: str):
        if not rows_data:
            tbl.setRowCount(0)
            if series_key == "X": self.spectra_x.clear()
            elif series_key == "Y": self.spectra_y.clear()
            else: self.spectra_mix.clear()
            return

        specs = []
        for r_info in rows_data:
            s_name = r_info.get("name", "")
            # Szukamy widma w magazynie lub oknach roboczych
            s = session.get_spectrum(s_name)
            if not s:
                for win in session.windows.values():
                    if s_name in win.spectra:
                        s = win.spectra[s_name]
                        break

            if s:
                specs.append(s.clone())
            else:
                # Rezerwowy obiekt, gdyby widmo nie było dostępne
                specs.append(SpectrumData(
                    wavenumbers=np.array([4000.0, 400.0]),
                    intensities=np.array([0.0, 0.0]),
                    name=s_name, file_path="", file_type="placeholder"
                ))

        if series_key == "X": self.spectra_x = specs
        elif series_key == "Y": self.spectra_y = specs
        else: self.spectra_mix = specs

        tbl.blockSignals(True)
        tbl.setRowCount(len(rows_data))
        for r, r_info in enumerate(rows_data):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked if r_info.get("checked", True) else Qt.CheckState.Unchecked)
            tbl.setItem(r, 0, chk)

            name_item = QTableWidgetItem(r_info.get("name", ""))
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            name_item.setForeground(QColor("#48cae4"))
            tbl.setItem(r, 1, name_item)

            cells = r_info.get("cells", [])
            for c_offset, cell_val in enumerate(cells):
                col = 2 + c_offset
                if col < tbl.columnCount():
                    tbl.setItem(r, col, QTableWidgetItem(str(cell_val)))
        tbl.blockSignals(False)

    def clear_state(self):
        """Czyści stan panelu przy tworzeniu nowego projektu."""
        self._is_updating_table = True
        try:
            self.table_x.setRowCount(0)
            self.spectra_x.clear()
            self.table_y.setRowCount(0)
            self.spectra_y.clear()
            self.table_mix.setRowCount(0)
            self.spectra_mix.clear()
            self.results.clear()
            self.rb_binary.setChecked(True)
            self._update_mode_visibility()
            self.txt_name_x.setText("Substancja_X")
            self.txt_name_y.setText("Substancja_Y")
            self._update_component_names()
            self._clear_cursor_line()
        finally:
            self._is_updating_table = False