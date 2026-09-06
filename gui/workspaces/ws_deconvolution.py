"""
gui/workspaces/ws_deconvolution.py
Panel dekonwolucji pasma Amidu I i fitowania krzywych (lmfit).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QMessageBox, QFrame
)
from PySide6.QtCore import Qt
import numpy as np
import logging

from core.session import session
from core.i18n import tr, i18n
from algorithms.fitting.deconv_engine import DeconvolutionEngine
from algorithms import preprocessing as prep

logger = logging.getLogger("SpectroHub")


class DeconvolutionPanel(QWidget):
    """Widok inspektora dla perspektywy Dekonwolucja & Fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks = []
        self.peak_counter = 0
        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1. Konfiguracja środowiska
        row_cfg = QHBoxLayout()
        self.lbl_solvent = QLabel("Rozpuszczalnik:")
        self.combo_solvent = QComboBox()
        self.combo_solvent.addItems(["H2O", "D2O"])
        self.lbl_model = QLabel("Model:")
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Gaussian", "Voigt", "Lorentzian", "PseudoVoigt"])
        row_cfg.addWidget(self.lbl_solvent)
        row_cfg.addWidget(self.combo_solvent)
        row_cfg.addWidget(self.lbl_model)
        row_cfg.addWidget(self.combo_model)
        layout.addLayout(row_cfg)

        # 2. Tabela pasm
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            tr("deconv.table.peak", "Pik"),
            tr("deconv.table.center", "Pozycja"),
            tr("deconv.table.fwhm", "FWHM"),
            tr("deconv.table.area_pct", "Pole %"),
            tr("deconv.table.structure", "Struktura")
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # 3. Przyciski edycji pasm
        row_btns = QHBoxLayout()
        self.btn_add_peak = QPushButton("➕ Dodaj pasmo (1650 cm⁻¹)")
        self.btn_add_peak.clicked.connect(self._on_add_default_peak)
        self.btn_clear_peaks = QPushButton("🗑 Wyczyść")
        self.btn_clear_peaks.clicked.connect(self._on_clear_peaks)
        row_btns.addWidget(self.btn_add_peak)
        row_btns.addWidget(self.btn_clear_peaks)
        layout.addLayout(row_btns)

        # 4. Asystent 2. pochodnej
        self.btn_calc_deriv2 = QPushButton("📉 Wylicz 2. pochodną (Szukaj pasm)")
        self.btn_calc_deriv2.clicked.connect(self._on_calc_deriv2)
        layout.addWidget(self.btn_calc_deriv2)

        # 5. Silnik dopasowujący
        self.btn_fit = QPushButton("⚡ Dopasuj Pasma (Levenberg-Marquardt)")
        self.btn_fit.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 8px;")
        self.btn_fit.clicked.connect(self._on_run_fit)
        layout.addWidget(self.btn_fit)

        # 6. Panel metryk jakości (GoF)
        self.frame_gof = QFrame()
        self.frame_gof.setStyleSheet("background-color: #252530; border-radius: 4px; padding: 6px;")
        l_gof = QVBoxLayout(self.frame_gof)
        self.lbl_gof_title = QLabel("<b>Jakość dopasowania (GoF):</b>")
        self.lbl_r2 = QLabel("R²: —")
        self.lbl_rmse = QLabel("RMSE: —")
        self.lbl_chi = QLabel("χ² / Red-χ²: —")
        l_gof.addWidget(self.lbl_gof_title)
        l_gof.addWidget(self.lbl_r2)
        l_gof.addWidget(self.lbl_rmse)
        l_gof.addWidget(self.lbl_chi)
        layout.addWidget(self.frame_gof)

        self.retranslate_ui()

    def _on_add_default_peak(self):
        active = session.active_spectrum
        center = 1650.0
        if active:
            # Domyślnie w środku widocznego zakresu
            center = float(np.mean(active.wavenumbers))

        self.peak_counter += 1
        model_type = self.combo_model.currentText()
        solvent = self.combo_solvent.currentText()
        struct_guess = DeconvolutionEngine.guess_structure(center, solvent=solvent)

        peak = {
            'id': f"p{self.peak_counter}",
            'type': model_type,
            'center': center,
            'sigma': 10.0,
            'area': 1.0,
            'structure': struct_guess,
            'lock_center': False,
            'lock_sigma': False,
            'lock_area': False
        }
        self.peaks.append(peak)
        self._refresh_table()

    def _on_clear_peaks(self):
        self.peaks.clear()
        self._refresh_table()

    def _on_calc_deriv2(self):
        active = session.active_spectrum
        if not active:
            return
        x, d2y = prep.get_second_derivative_inverted(active.wavenumbers, active.intensities, window=13, poly=3)
        # Zastępujemy dolny wykres drugą pochodną
        canvas = self.window().canvas_2d
        canvas.ax_sub.clear()
        canvas.ax_sub.plot(x, d2y, color="#98c1d9", lw=1.0)
        canvas.ax_sub.axhline(0, color="#ee6c4d", lw=0.6, ls=":")
        canvas.ax_sub.set_ylabel("-d²A/dν²", color="#dcdcdc", fontsize=9)
        canvas.canvas.draw_idle()

    def _on_run_fit(self):
        active = session.active_spectrum
        if not active or not self.peaks:
            QMessageBox.warning(self, tr("app.warning"), "Wymagane jest aktywne widmo i co najmniej jedno pasmo.")
            return

        try:
            updated_peaks, gof = DeconvolutionEngine.fit_peaks(
                active.wavenumbers, active.intensities, self.peaks, method="leastsq"
            )
            self.peaks = updated_peaks
            self._refresh_table()

            # Aktualizacja statystyk
            self.lbl_r2.setText(f"R²: {gof['r2']:.5f}")
            self.lbl_rmse.setText(f"RMSE: {gof['rmse']:.6f}")
            self.lbl_chi.setText(f"χ²: {gof['chisqr']:.2e} | Red-χ²: {gof['redchi']:.2e}")

            # Wyrysowanie dopasowanych pasm na wykresie
            curves = DeconvolutionEngine.evaluate_model_curves(active.wavenumbers, active.intensities, self.peaks)
            canvas = self.window().canvas_2d
            canvas.ax_main.clear()
            canvas._apply_styling()

            canvas.ax_main.plot(active.wavenumbers, active.intensities, label="Eksperyment", color="#ffffff", lw=1.2)
            canvas.ax_main.plot(active.wavenumbers, curves['total'], label="Fit Sum", color="#ee6c4d", lw=1.4)

            palette = ['#f4a261', '#2a9d8f', '#e76f51', '#98c1d9', '#e9c46a']
            for idx, c in enumerate(curves['components']):
                canvas.ax_main.plot(active.wavenumbers, c['y'], lw=0.9, ls="--",
                                   color=palette[idx % len(palette)], label=c['id'])

            canvas.ax_sub.clear()
            canvas.ax_sub.plot(active.wavenumbers, curves['residuals'], color="#98c1d9", lw=0.8)
            canvas.ax_sub.axhline(0, color="#ee6c4d", lw=0.6, ls=":")
            canvas.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a', labelcolor='#ffffff', fontsize=7)
            canvas.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Niepowodzenie fitowania lmfit: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd procedury dopasowania:\n{e}")

    def _refresh_table(self):
        self.table.setRowCount(len(self.peaks))
        total_area = sum(p['area'] for p in self.peaks) or 1.0

        for row, p in enumerate(self.peaks):
            fwhm = p['sigma'] * (2.0 if p['type'] == 'Lorentzian' else 2.35482)
            area_pct = (p['area'] / total_area) * 100.0

            self.table.setItem(row, 0, QTableWidgetItem(p['id']))
            self.table.setItem(row, 1, QTableWidgetItem(f"{p['center']:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{fwhm:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{area_pct:.1f} %"))
            self.table.setItem(row, 4, QTableWidgetItem(p.get('structure', '—')))

    def retranslate_ui(self):
        self.lbl_solvent.setText(tr("deconv.solvent", "Rozpuszczalnik:"))
        self.btn_add_peak.setText(tr("deconv.add_peak", "➕ Dodaj pasmo"))
        self.btn_clear_peaks.setText(tr("deconv.clear_peaks", "🗑 Wyczyść"))
        self.btn_fit.setText(tr("deconv.fit_btn", "⚡ Dopasuj Pasma (lmfit)"))