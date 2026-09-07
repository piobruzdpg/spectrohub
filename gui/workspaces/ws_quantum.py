"""
gui/workspaces/ws_quantum.py
Panel analizy obliczeń kwantowych (Gaussian/ORCA), PED i efektu izotopowego.
Wspiera otwieranie niezależnego okna wizualizacji 3D oraz interaktywne skalowanie.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QDoubleSpinBox, QGroupBox,
    QComboBox
)
from PySide6.QtCore import Qt, QTimer
import logging
from typing import Optional
import numpy as np
from core.session import session
from core.i18n import tr, i18n
from core.models import GaussianData
from algorithms.quantum.ped_engine import PEDAnalyzer
from gui.vibration_dialog import VibrationVisualizerDialog

logger = logging.getLogger("SpectroHub")


class QuantumPanel(QWidget):
    """Widok inspektora dla perspektywy DFT & Wibracje 3D z regulacją splotu i synchronizacją wykresu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vib_dialog: Optional[VibrationVisualizerDialog] = None
        self._canvas_connected = False

        self._init_ui()
        self._connect_signals()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 1. Metadane i parametry generowania widma DFT
        group_meta = QGroupBox("Parametry Widma Teoretycznego DFT")
        self.group_meta = group_meta
        l_meta = QVBoxLayout(group_meta)

        self.lbl_source = QLabel("Źródło: —")
        self.lbl_counts = QLabel("Liczba atomów: — | Liczba modów: —")
        l_meta.addWidget(self.lbl_source)
        l_meta.addWidget(self.lbl_counts)

        # Skalowanie częstości (s)
        row_scale = QHBoxLayout()
        self.lbl_scale = QLabel("Skalowanie (s):")
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.70, 1.30)
        self.spin_scale.setValue(1.000)
        self.spin_scale.setSingleStep(0.005)
        self.spin_scale.setDecimals(3)
        self.spin_scale.valueChanged.connect(self._on_params_changed)
        row_scale.addWidget(self.lbl_scale)
        row_scale.addWidget(self.spin_scale)
        l_meta.addLayout(row_scale)

        # Szerokość FWHM
        row_fwhm = QHBoxLayout()
        self.lbl_fwhm = QLabel("FWHM [cm⁻¹]:")
        self.spin_fwhm = QDoubleSpinBox()
        self.spin_fwhm.setRange(2.0, 50.0)
        self.spin_fwhm.setValue(14.0)
        self.spin_fwhm.setSingleStep(1.0)
        self.spin_fwhm.valueChanged.connect(self._on_params_changed)
        row_fwhm.addWidget(self.lbl_fwhm)
        row_fwhm.addWidget(self.spin_fwhm)
        l_meta.addLayout(row_fwhm)

        # Kształt pasma (Gauss / Lorentz / Pseudo-Voigt)
        row_shape = QHBoxLayout()
        self.lbl_shape = QLabel("Kształt pasm:")
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Gauss", "Lorentz", "Pseudo-Voigt"])
        self.combo_shape.currentTextChanged.connect(self._on_params_changed)
        row_shape.addWidget(self.lbl_shape)
        row_shape.addWidget(self.combo_shape)
        l_meta.addLayout(row_shape)

        layout.addWidget(group_meta)

        # 2. Tabela PED
        self.table_ped = QTableWidget(0, 4)
        self.table_ped.setHorizontalHeaderLabels(["Mod", "ν [cm⁻¹]", "I [km/mol]", "Dominujące Drganie (PED)"])
        self.table_ped.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_ped.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_ped.doubleClicked.connect(self._on_table_double_clicked)
        layout.addWidget(self.table_ped)

        # 3. Przyciski akcji
        self.btn_open_3d = QPushButton("🎥 Otwórz Podgląd 3D dla zaznaczonego")
        self.btn_open_3d.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 6px;")
        self.btn_open_3d.clicked.connect(self._on_open_3d_clicked)
        layout.addWidget(self.btn_open_3d)

        self.btn_isotope = QPushButton("🔄 Symuluj Deuterowanie (H ➔ D)")
        self.btn_isotope.clicked.connect(self._on_simulate_isotope)
        layout.addWidget(self.btn_isotope)

        self.retranslate_ui()

    def _connect_signals(self):
        session.active_spectrum_changed.connect(self._on_active_spectrum_changed)
        QTimer.singleShot(150, self._ensure_canvas_connected)

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _ensure_canvas_connected(self):
        if self._canvas_connected:
            return
        canvas = self._get_canvas()
        if canvas and hasattr(canvas, 'dft_mode_selected'):
            canvas.dft_mode_selected.connect(self.select_mode_from_canvas)
            self._canvas_connected = True

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_canvas_connected()

    def select_mode_from_canvas(self, mode_idx: int):
        """Zaznacza i przewija tabelę do modu wskazanego kliknięciem na wykresie."""
        if 0 <= mode_idx < self.table_ped.rowCount():
            self.table_ped.selectRow(mode_idx)
            item = self.table_ped.item(mode_idx, 0)
            if item:
                self.table_ped.scrollToItem(item)

    def _on_active_spectrum_changed(self, spectrum):
        self._ensure_canvas_connected()
        if isinstance(spectrum, GaussianData):
            self._load_dft_data(spectrum)
        else:
            self.table_ped.setRowCount(0)
            self.lbl_source.setText("Źródło: —")
            self.lbl_counts.setText("Wybierz obiekt DFT z drzewa projektu.")

    def _load_dft_data(self, dft: GaussianData):
        self.lbl_source.setText(f"Źródło: {dft.hessian_source} ({dft.file_type.upper()})")
        n_atoms = len(dft.atomnos)
        n_modes = len(dft.stick_freqs)
        self.lbl_counts.setText(f"Liczba atomów: {n_atoms} | Liczba modów: {n_modes}")

        self.spin_scale.blockSignals(True)
        self.spin_scale.setValue(dft.scale_factor)
        self.spin_scale.blockSignals(False)

        self.table_ped.setRowCount(n_modes)
        for row, (freq, inten) in enumerate(zip(dft.stick_freqs, dft.stick_intens)):
            ped_txt = "—"
            if dft.ped_results and row < len(dft.ped_results):
                ped_txt = dft.ped_results[row].get('summary', 'Inne')

            f_scaled = freq * dft.scale_factor
            self.table_ped.setItem(row, 0, QTableWidgetItem(f"#{row + 1}"))
            self.table_ped.setItem(row, 1, QTableWidgetItem(f"{f_scaled:.1f}"))
            self.table_ped.setItem(row, 2, QTableWidgetItem(f"{inten:.1f}"))
            self.table_ped.setItem(row, 3, QTableWidgetItem(ped_txt))

    def _on_params_changed(self):
        """Dynamicznie przelicza splot pasm i odświeża wykres."""
        active = session.active_spectrum
        if not isinstance(active, GaussianData) or len(active.stick_freqs) == 0:
            return

        s = self.spin_scale.value()
        fwhm = self.spin_fwhm.value()
        shape = self.combo_shape.currentText()

        active.scale_factor = s
        scaled_freqs = active.stick_freqs * s
        intens = active.stick_intens

        min_wn = max(200.0, np.min(scaled_freqs) - 150.0)
        max_wn = max(4000.0, np.max(scaled_freqs) + 150.0)
        wn_grid = np.arange(min_wn, max_wn + 1.0, 1.0)
        spectrum_y = np.zeros_like(wn_grid)

        c_gauss = 4.0 * np.log(2.0)
        gamma = fwhm / 2.0

        for f_val, i_val in zip(scaled_freqs, intens):
            diff = wn_grid - f_val
            if shape == "Gauss":
                spectrum_y += i_val * np.exp(-c_gauss * (diff / fwhm) ** 2)
            elif shape == "Lorentz":
                spectrum_y += i_val * (gamma ** 2) / (diff ** 2 + gamma ** 2)
            else:  # Pseudo-Voigt (50/50)
                y_g = np.exp(-c_gauss * (diff / fwhm) ** 2)
                y_l = (gamma ** 2) / (diff ** 2 + gamma ** 2)
                spectrum_y += i_val * (0.5 * y_g + 0.5 * y_l)

        # Aktualizacja wektorów aktywnego widma
        active.wavenumbers = wn_grid
        active.intensities = spectrum_y

        self._load_dft_data(active)
        canvas = self._get_canvas()
        if canvas:
            canvas.redraw_all()

    def _on_table_double_clicked(self, index):
        self._open_3d_viewer(index.row())

    def _on_open_3d_clicked(self):
        row = self.table_ped.currentRow()
        if row < 0:
            row = 0
        self._open_3d_viewer(row)

    def _open_3d_viewer(self, mode_idx: int):
        active = session.active_spectrum
        if not isinstance(active, GaussianData) or len(active.atomnos) == 0:
            QMessageBox.warning(self, tr("app.warning"), "Brak geometrii atomowej dla aktywnego widma.")
            return

        if self.vib_dialog is not None and self.vib_dialog.isVisible():
            self.vib_dialog.set_mode(mode_idx)
            self.vib_dialog.raise_()
            self.vib_dialog.activateWindow()
        else:
            self.vib_dialog = VibrationVisualizerDialog(
                self,
                atomnos=active.atomnos,
                coords=active.coords,
                freqs=active.stick_freqs,
                intens=active.stick_intens,
                Lx=active.Lx,
                initial_mode=mode_idx
            )
            self.vib_dialog.show()

    def _on_simulate_isotope(self):
        active = session.active_spectrum
        if not isinstance(active, GaussianData) or len(active.atomnos) == 0:
            return

        mass_rep = {i: 2.01410178 for i, an in enumerate(active.atomnos) if an == 1}
        if not mass_rep:
            QMessageBox.information(self, tr("app.info"), "Struktura nie zawiera atomów wodoru do deuterowania.")
            return

        iso_freqs, iso_intens, iso_Lx = PEDAnalyzer.simulate_isotopic_shift(
            active.atomnos, active.coords, list(active.stick_freqs), list(active.stick_intens), active.Lx, mass_rep
        )

        iso_data = GaussianData(
            wavenumbers=active.wavenumbers.copy(),
            intensities=active.intensities.copy(),
            name=f"{active.name}_deuterated",
            file_path=active.file_path,
            file_type="gaussian",
            stick_freqs=np.array(iso_freqs),
            stick_intens=np.array(iso_intens),
            atomnos=active.atomnos,
            coords=active.coords,
            Lx=iso_Lx,
            hessian_source="ISO_H->D"
        )
        session.add_spectrum(iso_data, make_active=True)
        QMessageBox.information(self, tr("app.info"), f"Utworzono widmo izotopomeru ({len(iso_freqs)} modów po deuterowaniu).")

    def retranslate_ui(self):
        self.group_meta.setTitle(tr("dft.meta_group", "Parametry Widma Teoretycznego DFT"))
        self.lbl_scale.setText(tr("dft.scale_factor", "Skalowanie (s):"))
        self.btn_open_3d.setText(tr("dft.open_3d", "🎥 Otwórz Podgląd 3D dla zaznaczonego"))
        self.btn_isotope.setText(tr("dft.isotope_btn", "🔄 Symuluj Deuterowanie (H ➔ D)"))
        self.table_ped.setHorizontalHeaderLabels([
            tr("dft.col_mode", "Mod"),
            tr("dft.col_freq", "ν [cm⁻¹]"),
            tr("dft.col_inten", "I [km/mol]"),
            tr("dft.col_ped", "Dominujące Drganie (PED)")
        ])