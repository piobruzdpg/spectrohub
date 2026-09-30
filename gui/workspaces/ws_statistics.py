"""
gui/workspaces/ws_statistics.py
Workspace Statystyki Spektralnej (Workspace 6).
Modułowa architektura z zakładkami narzędziowymi:
1. Szacowanie szumu (ASTM Detrended RMS, Peak-to-Peak, SNR).
2. Agregacja replikatów (Średnia, Mediana, Wstęgi błędu ±σ, CV%).
3. Jakość replikatów i detekcja pomiarów odstających (Korelacja Pearsona).
"""

import copy
import logging
import numpy as np
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QRadioButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QButtonGroup, QScrollArea, QFrame,
    QMessageBox, QTabWidget, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from core.session import session
from core.models import SpectrumData
from core.i18n import tr
from algorithms import statistics as stat_engine
from algorithms.spectral_math import align_to_reference
from gui.workspaces.ws_preprocessing import AddBatchSpectraCommand

logger = logging.getLogger("SpectroHub")


# =============================================================================
# KARTA 1: SZACOWANIE SZUMU I SNR (NOISE ESTIMATION TOOL)
# =============================================================================

class NoiseEstimationTab(QWidget):
    """Narzędzie precyzyjnego szacowania szumu (ASTM) i SNR w oknach liczb falowych."""

    def __init__(self, parent_workspace: 'StatisticsWorkspace'):
        super().__init__()
        self.ws = parent_workspace
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        # 1. Okno szumu
        box_noise = QGroupBox("1. Okno Pomiaru Szumu (Płaska Linia Bazowa)")
        box_noise.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        l_n = QGridLayout(box_noise)
        l_n.setSpacing(6)

        l_n.addWidget(QLabel("Zakres liczb falowych [cm⁻¹]:"), 0, 0, 1, 2)
        self.spin_n_min = QDoubleSpinBox()
        self.spin_n_min.setRange(350.0, 5000.0)
        self.spin_n_min.setValue(2000.0)
        self.spin_n_min.setDecimals(1)
        self.spin_n_min.setSuffix(" cm⁻¹")

        self.spin_n_max = QDoubleSpinBox()
        self.spin_n_max.setRange(350.0, 5000.0)
        self.spin_n_max.setValue(2200.0)
        self.spin_n_max.setDecimals(1)
        self.spin_n_max.setSuffix(" cm⁻¹")

        l_n.addWidget(self.spin_n_min, 1, 0)
        l_n.addWidget(self.spin_n_max, 1, 1)

        self.chk_detrend = QCheckBox("Odlicz trend liniowy bazy (ASTM E1421)")
        self.chk_detrend.setChecked(True)
        self.chk_detrend.setToolTip("Usuwa nachylenie linii bazowej, zapobiegając sztucznemu zawyżaniu szumu.")
        l_n.addWidget(self.chk_detrend, 2, 0, 1, 2)

        layout.addWidget(box_noise)

        # 2. Okno sygnału pod SNR (Opcjonalne)
        box_sig = QGroupBox("2. Pasmo Analityczne Sygnału (dla SNR)")
        box_sig.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        l_s = QGridLayout(box_sig)
        l_s.setSpacing(6)

        self.chk_calc_snr = QCheckBox("Obliczaj także SNR (Signal-to-Noise)")
        self.chk_calc_snr.setChecked(True)
        l_s.addWidget(self.chk_calc_snr, 0, 0, 1, 2)

        self.spin_s_min = QDoubleSpinBox()
        self.spin_s_min.setRange(350.0, 5000.0)
        self.spin_s_min.setValue(1600.0)
        self.spin_s_min.setDecimals(1)
        self.spin_s_min.setSuffix(" cm⁻¹")

        self.spin_s_max = QDoubleSpinBox()
        self.spin_s_max.setRange(350.0, 5000.0)
        self.spin_s_max.setValue(1700.0)
        self.spin_s_max.setDecimals(1)
        self.spin_s_max.setSuffix(" cm⁻¹")

        l_s.addWidget(self.spin_s_min, 1, 0)
        l_s.addWidget(self.spin_s_max, 1, 1)

        layout.addWidget(box_sig)

        # 3. Przycisk przeliczenia
        self.btn_calc_noise = QPushButton("⚡ Oblicz wskaźniki szumu dla widocznych widm")
        self.btn_calc_noise.setStyleSheet("background-color: #1b4332; color: #52b788; font-weight: bold; padding: 7px;")
        self.btn_calc_noise.clicked.connect(self.calculate_noise)
        layout.addWidget(self.btn_calc_noise)

        # 4. Tabela raportu wyników
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["Widmo", "RMS Szum", "Peak-to-Peak", "SNR"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_results.setStyleSheet("background-color: #1e1e24; color: #dcdcdc; font-size: 10px;")
        layout.addWidget(self.table_results)

    def calculate_noise(self):
        spectra = session.get_display_spectra()
        if not spectra:
            QMessageBox.warning(self, tr("app.warning"), "Brak widocznych widm w aktywnym oknie.")
            return

        n_min = self.spin_n_min.value()
        n_max = self.spin_n_max.value()
        detrend = self.chk_detrend.isChecked()
        calc_snr = self.chk_calc_snr.isChecked()
        s_min = self.spin_s_min.value()
        s_max = self.spin_s_max.value()

        self.table_results.setRowCount(len(spectra))

        for row, spec in enumerate(spectra):
            x, y = spec.wavenumbers, spec.intensities
            metrics = stat_engine.estimate_noise_metrics(x, y, n_min, n_max, detrend=detrend)
            rms = metrics["rms_noise"]
            ptp = metrics["peak_to_peak"]

            snr_str = "—"
            if calc_snr:
                snr_data = stat_engine.calculate_snr(x, y, s_min, s_max, n_min, n_max, detrend_noise=detrend)
                snr_str = f"{snr_data['snr']:.1f}"

            item_name = QTableWidgetItem(spec.name)
            item_rms = QTableWidgetItem(f"{rms:.4e}")
            item_ptp = QTableWidgetItem(f"{ptp:.4e}")
            item_snr = QTableWidgetItem(snr_str)

            for it in (item_name, item_rms, item_ptp, item_snr):
                it.setTextAlignment(Qt.AlignCenter)

            self.table_results.setItem(row, 0, item_name)
            self.table_results.setItem(row, 1, item_rms)
            self.table_results.setItem(row, 2, item_ptp)
            self.table_results.setItem(row, 3, item_snr)


# =============================================================================
# KARTA 2: AGREGACJA REPLIKATÓW (ŚREDNIA, MEDIANA, ODCHYLENIE STD)
# =============================================================================

class ReplicatesAggregationTab(QWidget):
    """Agregacja powtórzeń laboratoryjnych: widmo średnie, mediana, odchylenie standardowe."""

    def __init__(self, parent_workspace: 'StatisticsWorkspace'):
        super().__init__()
        self.ws = parent_workspace
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        box_method = QGroupBox("1. Metoda Centralna Agregacji")
        box_method.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        l_m = QVBoxLayout(box_method)

        self.bg_agg = QButtonGroup(self)
        self.rb_mean = QRadioButton("Średnia arytmetyczna (Mean)")
        self.rb_mean.setChecked(True)
        self.rb_median = QRadioButton("Mediana (Odporna na błędy docisku i pęcherzyki)")

        self.bg_agg.addButton(self.rb_mean, 0)
        self.bg_agg.addButton(self.rb_median, 1)

        l_m.addWidget(self.rb_mean)
        l_m.addWidget(self.rb_median)
        layout.addWidget(box_method)

        box_extras = QGroupBox("2. Generowane Widma Towarzyszące")
        box_extras.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        l_e = QVBoxLayout(box_extras)

        self.chk_gen_std = QCheckBox("Generuj widmo odchylenia standardowego (±σ)")
        self.chk_gen_std.setChecked(True)
        self.chk_gen_cv = QCheckBox("Generuj widmo współczynnika zmienności (CV%)")

        l_e.addWidget(self.chk_gen_std)
        l_e.addWidget(self.chk_gen_cv)
        layout.addWidget(box_extras)

        self.btn_commit_agg = QPushButton("➕ Wyznacz i dodaj do projektu")
        self.btn_commit_agg.setStyleSheet("background-color: #2a9d8f; color: #ffffff; font-weight: bold; padding: 8px;")
        self.btn_commit_agg.clicked.connect(self._on_commit_aggregation)
        layout.addWidget(self.btn_commit_agg)

        layout.addStretch()

    def _on_commit_aggregation(self):
        spectra = session.get_display_spectra()
        if len(spectra) < 2:
            QMessageBox.warning(self, tr("app.warning"), "Do agregacji potrzebujesz co najmniej 2 widm.")
            return

        ref_x = spectra[0].wavenumbers
        aligned_y = []
        for s in spectra:
            y_al = align_to_reference(s.wavenumbers, s.intensities, ref_x)
            aligned_y.append(y_al)

        matrix = np.column_stack(aligned_y)
        res = stat_engine.aggregate_spectral_ensemble(matrix)

        is_median = self.rb_median.isChecked()
        rep_y = res["median"] if is_median else res["mean"]
        tag = "mediana" if is_median else "srednia"

        base_name = spectra[0].name.split("_")[0]
        new_spectra = []

        # 1. Reprezentant główny
        spec_rep = SpectrumData(
            wavenumbers=ref_x.copy(),
            intensities=rep_y.copy(),
            name=f"{base_name}_{tag}",
            file_path="",
            file_type="stat_aggregated",
            metadata={"agg_type": tag, "n_replicates": len(spectra)}
        )
        new_spectra.append(spec_rep)

        # 2. Odchylenie standardowe
        if self.chk_gen_std.isChecked():
            spec_std = SpectrumData(
                wavenumbers=ref_x.copy(),
                intensities=res["std"].copy(),
                name=f"{base_name}_std",
                file_path="",
                file_type="stat_std",
                metadata={"agg_type": "std", "n_replicates": len(spectra)}
            )
            new_spectra.append(spec_std)

        # 3. CV%
        if self.chk_gen_cv.isChecked():
            cv_y = np.zeros_like(rep_y)
            mask = np.abs(rep_y) > 1e-4
            cv_y[mask] = (res["std"][mask] / rep_y[mask]) * 100.0

            spec_cv = SpectrumData(
                wavenumbers=ref_x.copy(),
                intensities=cv_y,
                name=f"{base_name}_CVpct",
                file_path="",
                file_type="stat_cv",
                metadata={"agg_type": "cv_percent"}
            )
            new_spectra.append(spec_cv)

        canvas = getattr(self.window(), 'canvas_2d', None)
        target_win = session.active_window_id if session.active_context == "window" else None
        cmd = AddBatchSpectraCommand(new_spectra, f"Agregacja statystyczna ({tag})", canvas=canvas, window_id=target_win)
        session.undo_stack.push(cmd)

        QMessageBox.information(self, tr("app.info"), f"Pomyślnie wyznaczono i dodano {len(new_spectra)} widm(a) statystyczne.")


# =============================================================================
# GŁÓWNY PANEL PERSPEKTYWY STATYSTYKI SPEKTRALNEJ
# =============================================================================

class StatisticsWorkspace(QWidget):
    """Główny panel Workspace Statystyka."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Zakładki narzędziowe
        self.tabs_tools = QTabWidget()

        self.tab_noise = NoiseEstimationTab(self)
        self.tab_agg = ReplicatesAggregationTab(self)

        self.tabs_tools.addTab(self.tab_noise, "🔍 Szum & SNR")
        self.tabs_tools.addTab(self.tab_agg, "📊 Replikaty & Średnia")

        main_layout.addWidget(self.tabs_tools)

    def refresh_view(self):
        """Metoda wywoływana przy przełączeniu na tę perspektywę."""
        pass