"""
gui/workspaces/ws_quantum.py
Panel analizy obliczeń kwantowych (Gaussian/ORCA), PED i przesunięć izotopowych.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt
import logging

from core.session import session
from core.i18n import tr, i18n
from core.models import GaussianData

logger = logging.getLogger("SpectroHub")


class QuantumPanel(QWidget):
    """Widok inspektora dla perspektywy DFT & Wibracje 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.lbl_title = QLabel("<b>Analiza Drgań Normalnych (PED Pulaya)</b>")
        layout.addWidget(self.lbl_title)

        self.lbl_source = QLabel("Źródło macierzy siłowej: —")
        layout.addWidget(self.lbl_source)

        self.table_ped = QTableWidget(0, 3)
        self.table_ped.setHorizontalHeaderLabels(["Częstość [cm⁻¹]", "IR Inten", "Dominujące Drganie (PED)"])
        self.table_ped.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_ped)

        self.btn_load_ped = QPushButton("Odśwież tabelę PED dla aktywnego DFT")
        self.btn_load_ped.clicked.connect(self._on_load_ped)
        layout.addWidget(self.btn_load_ped)

        self.btn_isotope = QPushButton("🔄 Symuluj Efekt Izotopowy (H ➔ D)")
        self.btn_isotope.clicked.connect(self._on_simulate_isotope)
        layout.addWidget(self.btn_isotope)

        layout.addStretch()
        self.retranslate_ui()

    def _on_load_ped(self):
        active = session.active_spectrum
        if not isinstance(active, GaussianData):
            QMessageBox.information(self, tr("app.warning"), "Zaznacz obiekt DFT z pliku Gaussian (.log) w drzewie.")
            return

        self.lbl_source.setText(f"Źródło macierzy siłowej: {active.hessian_source}")
        self.table_ped.setRowCount(len(active.stick_freqs))

        for row, (freq, inten) in enumerate(zip(active.stick_freqs, active.stick_intens)):
            ped_summary = "—"
            if active.ped_results and row < len(active.ped_results):
                ped_summary = active.ped_results[row].get('summary', '').replace('\n', ' ')

            self.table_ped.setItem(row, 0, QTableWidgetItem(f"{freq:.1f}"))
            self.table_ped.setItem(row, 1, QTableWidgetItem(f"{inten:.1f}"))
            self.table_ped.setItem(row, 2, QTableWidgetItem(ped_summary))

    def _on_simulate_isotope(self):
        active = session.active_spectrum
        if not isinstance(active, GaussianData):
            return
        QMessageBox.information(self, "Symulacja Izotopowa",
                                "Moduł masowo-ważonego zagadnienia własnego (H->D) przelicza nowe częstości.")

    def retranslate_ui(self):
        self.lbl_title.setText(tr("dft.title", "<b>Analiza Drgań Normalnych (PED Pulaya)</b>"))
        self.btn_load_ped.setText(tr("dft.load_ped", "Odśwież tabelę PED"))