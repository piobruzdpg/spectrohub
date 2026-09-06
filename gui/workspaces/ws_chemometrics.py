"""
gui/workspaces/ws_chemometrics.py
Panel analizy wielowymiarowej (MCR-ALS, PFA, SPEXFA, 2D-COS).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QMessageBox
)
import numpy as np
import logging

from core.session import session
from core.i18n import tr, i18n
from algorithms.chemometrics import multivariate as chem

logger = logging.getLogger("SpectroHub")


class ChemometricsPanel(QWidget):
    """Widok inspektora dla perspektywy Chemometria."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.lbl_method = QLabel("<b>Metoda Chemometryczna:</b>")
        layout.addWidget(self.lbl_method)

        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "MCR-ALS (Czyste składniki)",
            "PCA / PFA (Analiza rang)",
            "SPEXFA (Izolacja widm)",
            "2D-COS (Spektroskopia korelacyjna)"
        ])
        layout.addWidget(self.combo_method)

        row_k = QHBoxLayout()
        self.lbl_rank = QLabel("Liczba składników (k):")
        self.spin_rank = QSpinBox()
        self.spin_rank.setRange(1, 10)
        self.spin_rank.setValue(2)
        row_k.addWidget(self.lbl_rank)
        row_k.addWidget(self.spin_rank)
        layout.addLayout(row_k)

        # Restrykcje dla MCR
        self.chk_nnls = QCheckBox("Wymuś nieujemność (NNLS)")
        self.chk_nnls.setChecked(True)
        self.chk_closure = QCheckBox("Domknięcie bilansu (Closure)")
        self.chk_unimodal = QCheckBox("Jednomodalność (Unimodal)")
        layout.addWidget(self.chk_nnls)
        layout.addWidget(self.chk_closure)
        layout.addWidget(self.chk_unimodal)

        self.btn_run = QPushButton("🚀 Uruchom Dekompozycję")
        self.btn_run.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run_analysis)
        layout.addWidget(self.btn_run)

        layout.addStretch()
        self.retranslate_ui()

    def _on_run_analysis(self):
        visible_spectra = session.get_visible_spectra()
        if len(visible_spectra) < 2:
            QMessageBox.warning(self, tr("app.warning"), "Analiza wymaga co najmniej 2 zaznaczonych widm w drzewie.")
            return

        # Budowa macierzy 2D (Wiersze: liczby falowe, Kolumny: próbki)
        wavenumbers = visible_spectra[0].wavenumbers
        matrix_cols = []
        for s in visible_spectra:
            if len(s.wavenumbers) != len(wavenumbers) or not np.allclose(s.wavenumbers, wavenumbers, atol=1e-2):
                QMessageBox.critical(self, tr("app.error"), f"Widmo '{s.name}' ma inną siatkę liczb falowych.")
                return
            matrix_cols.append(s.intensities)

        D_matrix = np.column_stack(matrix_cols)
        k = self.spin_rank.value()
        method = self.combo_method.currentText()

        try:
            canvas = self.window().canvas_2d
            canvas.ax_main.clear()
            canvas.ax_sub.clear()
            canvas._apply_styling()

            if "MCR-ALS" in method:
                res = chem.run_mcr_als(
                    D_matrix, rank=k,
                    non_negative=self.chk_nnls.isChecked(),
                    closure=self.chk_closure.isChecked(),
                    unimodal=self.chk_unimodal.isChecked()
                )
                ST = res['ST']  # Czyste widma (liczby falowe x rank)
                C = res['C']    # Profile stężeń (próbki x rank)

                for comp_idx in range(k):
                    canvas.ax_main.plot(wavenumbers, ST[:, comp_idx], label=f"Składnik {comp_idx + 1}")
                    canvas.ax_sub.plot(C[:, comp_idx], 'o-', label=f"Profil C {comp_idx + 1}")

                canvas.ax_main.set_title("MCR-ALS: Czyste Widma (Pure Spectra)")
                canvas.ax_sub.set_title("MCR-ALS: Względne Stężenia (Concentrations)")
                canvas.ax_sub.set_xlabel("Numer widma w serii")

            elif "PCA" in method:
                res = chem.run_pca_uncentered(D_matrix.T, n_components=k)
                loadings = res['loadings']
                for comp_idx in range(k):
                    canvas.ax_main.plot(wavenumbers, loadings[:, comp_idx], label=f"Loading PC{comp_idx + 1}")
                canvas.ax_sub.bar(range(1, k + 1), res['variance'][:k] * 100.0, color="#ee6c4d")
                canvas.ax_sub.set_ylabel("Wariancja [%]")
                canvas.ax_sub.set_xlabel("Główna składowa (PC)")

            canvas.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a', labelcolor='#ffffff', fontsize=8)
            canvas.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Błąd chemometrii: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd dekompozycji:\n{e}")

    def retranslate_ui(self):
        self.lbl_method.setText(tr("chem.method_title", "<b>Metoda Chemometryczna:</b>"))
        self.btn_run.setText(tr("chem.run_btn", "🚀 Uruchom Dekompozycję"))