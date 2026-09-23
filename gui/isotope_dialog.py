"""
gui/isotope_dialog.py
Dialog selektywnej wymiany izotopowej H -> D z interaktywnym podglądem 3D cząsteczki (PyVista).
Umożliwia zaznaczanie labilnych protonów, pojedynczych wodorów oraz przeliczanie częstości i intensywności IR.
"""

import os
import numpy as np
from typing import Dict, Set, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QGroupBox, QSplitter, QMessageBox, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from core.models import GaussianData
from algorithms.quantum.ped_engine import (
    PEDAnalyzer, COVALENT_RADII, ELEMENT_SYMBOLS
)

CPK_COLORS = {
    1: "#FFFFFF", 6: "#333333", 7: "#3050F8", 8: "#FF0D0D",
    9: "#90E050", 14: "#DAA520", 15: "#FF8000", 16: "#FFFF30",
    17: "#1FF01F", 35: "#A62929", 53: "#940094"
}

RGB_DEUTERIUM = (0.0, 0.9, 1.0)   # Cyjan dla D
RGB_HYDROGEN = (1.0, 1.0, 1.0)    # Biel CPK dla H


class IsotopeExchangeDialog(QDialog):
    """Okno konfiguracji i podglądu selektywnej wymiany izotopowej H -> D."""

    def __init__(self, parent=None, active_dft: Optional[GaussianData] = None):
        super().__init__(parent)
        self.setWindowTitle("SpectroHub — Selektywna Wymiana Izotopowa H ➔ D")
        self.resize(1120, 700)
        self.setMinimumSize(850, 550)

        self.dft = active_dft
        self.atomnos = np.asarray(self.dft.atomnos if self.dft else [], dtype=int)
        self.coords = np.asarray(self.dft.coords if self.dft else np.empty((0, 3)), dtype=float)

        # Identyfikacja wodorów i partnerów
        self.protons_info = PEDAnalyzer.identify_labile_protons(self.atomnos, self.coords)
        self.selected_h_indices: Set[int] = set()

        # Detekcja .fchk
        self.fchk_path = None
        if self.dft and self.dft.file_path:
            cand = os.path.splitext(self.dft.file_path)[0] + ".fchk"
            if os.path.exists(cand):
                self.fchk_path = cand

        self.atom_actors = {}
        self.bonds = []
        self._calculate_bonds()

        self._init_ui()
        self._populate_table()
        # Startujemy z czystym wyborem – użytkownik decyduje sam
        self._clear_selection()

    def _calculate_bonds(self):
        n_atoms = len(self.atomnos)
        for i in range(n_atoms):
            r_i = COVALENT_RADII.get(int(self.atomnos[i]), 0.8)
            for j in range(i + 1, n_atoms):
                r_j = COVALENT_RADII.get(int(self.atomnos[j]), 0.8)
                dist = float(np.linalg.norm(self.coords[i] - self.coords[j]))
                if dist <= (r_i + r_j + 0.35):
                    self.bonds.append((i, j))

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # ------------------- LEWY PANEL: Podgląd 3D -------------------
        left_widget = QWidget()
        l_vbox = QVBoxLayout(left_widget)
        l_vbox.setContentsMargins(0, 0, 0, 0)
        l_vbox.setSpacing(6)

        lbl_hint = QLabel("💡 Kliknij w atom wodoru w 3D lub zaznacz wiersz w tabeli, aby włączyć deuterowanie (D).")
        lbl_hint.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        l_vbox.addWidget(lbl_hint)

        if HAS_PYVISTA and len(self.atomnos) > 0:
            self.plotter = QtInteractor(left_widget)
            self.plotter.set_background('#141418')
            l_vbox.addWidget(self.plotter.interactor, 1)
            self._setup_3d_scene()
        else:
            lbl_fallback = QLabel("Wizualizator 3D wymaga PyVista / QtInteractor lub brak geometrii.")
            lbl_fallback.setAlignment(Qt.AlignCenter)
            l_vbox.addWidget(lbl_fallback, 1)

        btn_cam_reset = QPushButton("🔍 Reset Kamery")
        if HAS_PYVISTA and hasattr(self, 'plotter'):
            btn_cam_reset.clicked.connect(self.plotter.reset_camera)
        l_vbox.addWidget(btn_cam_reset)

        splitter.addWidget(left_widget)

        # ------------------- PRAWY PANEL: Kontrola i Tabela -------------------
        right_widget = QWidget()
        r_vbox = QVBoxLayout(right_widget)
        r_vbox.setContentsMargins(0, 0, 0, 0)
        r_vbox.setSpacing(8)

        group_quick = QGroupBox("Szybki Wybór Protonów")
        l_quick = QVBoxLayout(group_quick)

        row_btns = QHBoxLayout()
        self.btn_labile = QPushButton("💧 Zaznacz labilne (O/N/S/P–H)")
        self.btn_labile.setStyleSheet("font-weight: bold; background-color: #2b4c6f;")
        self.btn_labile.clicked.connect(self._select_all_labile)

        self.btn_all = QPushButton("⚛️ Zaznacz wszystkie H")
        self.btn_all.clicked.connect(self._select_all)

        self.btn_clear = QPushButton("🧹 Wyczyść")
        self.btn_clear.clicked.connect(self._clear_selection)

        self.btn_invert = QPushButton("🔄 Odwróć")
        self.btn_invert.clicked.connect(self._invert_selection)

        row_btns.addWidget(self.btn_labile)
        row_btns.addWidget(self.btn_all)
        row_btns.addWidget(self.btn_clear)
        row_btns.addWidget(self.btn_invert)
        l_quick.addLayout(row_btns)

        self.lbl_counts = QLabel("Zaznaczono wodorów: 0 / 0")
        self.lbl_counts.setStyleSheet("font-weight: bold; color: #00E5FF; font-size: 12px;")
        l_quick.addWidget(self.lbl_counts)

        r_vbox.addWidget(group_quick)

        # Tabela wodorów z obsługą klikania całych wierszy
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["D", "Atom", "Partner", "Typ Ugrupowania", "Odległość [Å]"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        r_vbox.addWidget(self.table, 1)

        # Parametry
        group_calc = QGroupBox("Parametry Obliczeniowe")
        l_calc = QVBoxLayout(group_calc)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Nazwa widma:"))
        self.edit_name = QLineEdit()
        base_name = self.dft.name if self.dft else "DFT"
        self.edit_name.setText(f"{base_name}_deuterated")
        row_name.addWidget(self.edit_name)
        l_calc.addLayout(row_name)

        fchk_status = "Dostępny plik .fchk (Precyzyjne tensory polarne APT)" if self.fchk_path else "Plik .log / .out (Rzutowanie modów Duschinsky'ego)"
        color_src = "#4BB543" if self.fchk_path else "#E0A96D"
        lbl_src = QLabel(f"Źródło intensywności: <b style='color: {color_src};'>{fchk_status}</b>")
        l_calc.addWidget(lbl_src)

        r_vbox.addWidget(group_calc)

        row_actions = QHBoxLayout()
        self.btn_cancel = QPushButton("Anuluj")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("✓ Przelicz i Utwórz Widmo Izotopomeru")
        self.btn_apply.setStyleSheet("background-color: #2a9d8f; font-weight: bold; font-size: 13px; padding: 6px;")
        self.btn_apply.clicked.connect(self._on_apply)

        row_actions.addWidget(self.btn_cancel)
        row_actions.addWidget(self.btn_apply, 1)
        r_vbox.addLayout(row_actions)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 620])
        main_layout.addWidget(splitter)

    def _setup_3d_scene(self):
        self.plotter.clear()
        self.atom_actors = {}

        for idx, (an, coord) in enumerate(zip(self.atomnos, self.coords)):
            an_int = int(an)
            color = CPK_COLORS.get(an_int, "#DD77DD")
            radius = float(COVALENT_RADII.get(an_int, 0.7) * 0.38)
            coord_arr = np.asarray(coord, dtype=float)
            sphere = pv.Sphere(radius=radius, center=coord_arr, theta_resolution=16, phi_resolution=16)

            is_hydrogen = bool(an_int == 1)
            pickable_val = 1 if is_hydrogen else 0

            actor = self.plotter.add_mesh(
                sphere, color=color, smooth_shading=True,
                specular=0.5, specular_power=20, pickable=pickable_val, render=False
            )
            self.atom_actors[idx] = actor

        tubes = []
        for a, b in self.bonds:
            tube = pv.Tube(pointa=self.coords[a], pointb=self.coords[b], radius=0.06, resolution=10)
            tubes.append(tube)

        if tubes:
            merged = pv.merge(tubes)
            self.plotter.add_mesh(
                merged, color='#888888', opacity=0.85,
                smooth_shading=True, pickable=0, render=False
            )

        self.plotter.enable_mesh_picking(
            callback=self._on_mesh_picked,
            show_message=False,
            left_clicking=True
        )

        self.plotter.reset_camera()
        self.plotter.render()

    def _on_mesh_picked(self, picked_mesh):
        if picked_mesh is None:
            return

        if hasattr(self.plotter, '_picked_actor') and self.plotter._picked_actor is not None:
            self.plotter.remove_actor(self.plotter._picked_actor, render=False)
            self.plotter._picked_actor = None

        atom_idx = None
        if hasattr(picked_mesh, 'center'):
            try:
                center = np.asarray(picked_mesh.center, dtype=float)
                dists = np.linalg.norm(self.coords - center, axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < 0.5:
                    atom_idx = best_idx
            except Exception:
                atom_idx = None

        if atom_idx is not None and atom_idx in self.protons_info:
            self._toggle_proton(atom_idx)

    def _toggle_proton(self, atom_idx: int):
        if atom_idx in self.selected_h_indices:
            self.selected_h_indices.remove(atom_idx)
        else:
            self.selected_h_indices.add(atom_idx)

        self._sync_table_row(atom_idx)
        self._update_atom_color_in_3d(atom_idx)
        self._update_counter_and_name()

    def _update_atom_color_in_3d(self, atom_idx: int):
        if not HAS_PYVISTA or atom_idx not in self.atom_actors:
            return

        actor = self.atom_actors[atom_idx]
        is_deut = atom_idx in self.selected_h_indices
        actor.prop.color = RGB_DEUTERIUM if is_deut else RGB_HYDROGEN
        self.plotter.render()

    def _populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.protons_info))

        for row, (h_idx, info) in enumerate(sorted(self.protons_info.items())):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setData(Qt.UserRole, int(h_idx))
            self.table.setItem(row, 0, chk_item)

            self.table.setItem(row, 1, QTableWidgetItem(f"H{h_idx + 1}"))

            p_lbl = f"{info['partner_sym']}{info['partner_idx'] + 1}" if info['partner_idx'] >= 0 else "—"
            self.table.setItem(row, 2, QTableWidgetItem(p_lbl))

            lbl_grp = info['group_name']
            it_grp = QTableWidgetItem(lbl_grp)
            if info['is_labile']:
                it_grp.setForeground(QColor("#00E5FF"))
            self.table.setItem(row, 3, it_grp)

            self.table.setItem(row, 4, QTableWidgetItem(f"{info['bond_dist']:.3f}"))

        self.table.blockSignals(False)

    def _on_cell_clicked(self, row: int, col: int):
        """Umożliwia przełączanie stanu zaznaczenia kliknięciem w dowolną komórkę wiersza."""
        if col > 0:
            item = self.table.item(row, 0)
            if item:
                new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)

    def _sync_table_row(self, atom_idx: int):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) == atom_idx:
                item.setCheckState(Qt.Checked if atom_idx in self.selected_h_indices else Qt.Unchecked)
                break
        self.table.blockSignals(False)

    def _on_table_item_changed(self, item):
        if item.column() != 0:
            return
        atom_idx = item.data(Qt.UserRole)
        if atom_idx is not None:
            atom_idx = int(atom_idx)
            if item.checkState() == Qt.Checked:
                self.selected_h_indices.add(atom_idx)
            else:
                self.selected_h_indices.discard(atom_idx)
            self._update_atom_color_in_3d(atom_idx)
            self._update_counter_and_name()

    def _select_all_labile(self):
        self.selected_h_indices = {idx for idx, p in self.protons_info.items() if p['is_labile']}
        self._refresh_all_visuals()

    def _select_all(self):
        self.selected_h_indices = set(self.protons_info.keys())
        self._refresh_all_visuals()

    def _clear_selection(self):
        self.selected_h_indices.clear()
        self._refresh_all_visuals()

    def _invert_selection(self):
        all_h = set(self.protons_info.keys())
        self.selected_h_indices = all_h - self.selected_h_indices
        self._refresh_all_visuals()

    def _refresh_all_visuals(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                h_idx = int(item.data(Qt.UserRole))
                item.setCheckState(Qt.Checked if h_idx in self.selected_h_indices else Qt.Unchecked)
        self.table.blockSignals(False)

        if HAS_PYVISTA:
            for idx in self.protons_info:
                if idx in self.atom_actors:
                    is_deut = idx in self.selected_h_indices
                    self.atom_actors[idx].prop.color = RGB_DEUTERIUM if is_deut else RGB_HYDROGEN
            self.plotter.render()

        self._update_counter_and_name()

    def _update_counter_and_name(self):
        total_h = len(self.protons_info)
        n_sel = len(self.selected_h_indices)
        n_labile = sum(1 for p in self.protons_info.values() if p['is_labile'])
        self.lbl_counts.setText(f"Zaznaczono wodorów: {n_sel} / {total_h} (dostępnych labilnych: {n_labile})")

        base_name = self.dft.name if self.dft else "DFT"
        if n_sel == 0:
            tag = "deuterated"
        elif n_sel == total_h and total_h > 0:
            tag = "dAll"
        elif n_sel == n_labile and n_labile > 0 and self.selected_h_indices == {i for i, p in self.protons_info.items() if p['is_labile']}:
            tag = "dLabile"
        else:
            tag = f"d{n_sel}H"

        self.edit_name.setText(f"{base_name}_{tag}")

    def _on_apply(self):
        if not self.selected_h_indices:
            QMessageBox.warning(self, "Brak wyboru", "Nie zaznaczono żadnego atomu wodoru do podstawienia deuterem.")
            return

        self.accept()

    def get_selected_replacements(self) -> Dict[int, float]:
        """Zwraca słownik podstawień masowych: {indeks_atomu: 2.01410178}."""
        return {int(idx): 2.01410178 for idx in self.selected_h_indices}

    def get_target_name(self) -> str:
        return self.edit_name.text().strip() or "DFT_deuterated"

    def closeEvent(self, event):
        if HAS_PYVISTA and hasattr(self, 'plotter'):
            self.plotter.close()
        super().closeEvent(event)