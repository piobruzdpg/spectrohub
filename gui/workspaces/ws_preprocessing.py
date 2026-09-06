"""
gui/workspaces/ws_preprocessing.py
Panel sterowania obróbką danych widmowych (AsLS, SG, Normalizacja, Offset).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QFrame, QMessageBox
)
from PySide6.QtCore import Qt
import logging

from core.session import session
from core.i18n import tr, i18n
from algorithms import preprocessing as prep

logger = logging.getLogger("SpectroHub")


class PreprocessingPanel(QWidget):
    """Widok inspektora dla perspektywy Preprocessing & Czyszczenie."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- 1. Linia bazowa AsLS ---
        self.lbl_asls_group = QLabel("<b>Korekcja Linii Bazowej (AsLS)</b>")
        layout.addWidget(self.lbl_asls_group)

        f_asls = QFrame()
        f_asls.setStyleSheet("background-color: #252530; border-radius: 4px; padding: 6px;")
        l_asls = QVBoxLayout(f_asls)

        row_lam = QHBoxLayout()
        self.lbl_lam = QLabel("Lambda (gładkość):")
        self.spin_lam = QDoubleSpinBox()
        self.spin_lam.setRange(1e2, 1e9)
        self.spin_lam.setValue(1e5)
        self.spin_lam.setDecimals(0)
        row_lam.addWidget(self.lbl_lam)
        row_lam.addWidget(self.spin_lam)
        l_asls.addLayout(row_lam)

        row_p = QHBoxLayout()
        self.lbl_p = QLabel("Asymetria (p):")
        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(1e-5, 0.5)
        self.spin_p.setValue(0.001)
        self.spin_p.setDecimals(4)
        row_p.addWidget(self.lbl_p)
        row_p.addWidget(self.spin_p)
        l_asls.addLayout(row_p)

        self.btn_apply_asls = QPushButton("Odejmij linię bazową AsLS")
        self.btn_apply_asls.clicked.connect(self._on_apply_asls)
        l_asls.addWidget(self.btn_apply_asls)
        layout.addWidget(f_asls)

        # --- 2. Wygładzanie Savitzky-Golay ---
        self.lbl_sg_group = QLabel("<b>Wygładzanie (Savitzky-Golay)</b>")
        layout.addWidget(self.lbl_sg_group)

        f_sg = QFrame()
        f_sg.setStyleSheet("background-color: #252530; border-radius: 4px; padding: 6px;")
        l_sg = QVBoxLayout(f_sg)

        row_sg = QHBoxLayout()
        self.lbl_sg_w = QLabel("Okno:")
        self.spin_sg_w = QSpinBox()
        self.spin_sg_w.setRange(5, 101)
        self.spin_sg_w.setSingleStep(2)
        self.spin_sg_w.setValue(11)
        self.lbl_sg_p = QLabel("Wielomian:")
        self.spin_sg_p = QSpinBox()
        self.spin_sg_p.setRange(1, 6)
        self.spin_sg_p.setValue(3)
        row_sg.addWidget(self.lbl_sg_w)
        row_sg.addWidget(self.spin_sg_w)
        row_sg.addWidget(self.lbl_sg_p)
        row_sg.addWidget(self.spin_sg_p)
        l_sg.addLayout(row_sg)

        self.btn_apply_sg = QPushButton("Zastosuj filtr SG")
        self.btn_apply_sg.clicked.connect(self._on_apply_sg)
        l_sg.addWidget(self.btn_apply_sg)
        layout.addWidget(f_sg)

        # --- 3. Szybkie normalizacje ---
        self.lbl_norm_group = QLabel("<b>Transformacje i Normalizacje</b>")
        layout.addWidget(self.lbl_norm_group)

        self.btn_snv = QPushButton("Normalizacja SNV")
        self.btn_snv.clicked.connect(lambda: self._on_apply_norm("snv"))
        layout.addWidget(self.btn_snv)

        self.btn_minmax = QPushButton("Skalowanie Min-Max (0-1)")
        self.btn_minmax.clicked.connect(lambda: self._on_apply_norm("minmax"))
        layout.addWidget(self.btn_minmax)

        self.btn_l2 = QPushButton("Normalizacja wektorowa (L2)")
        self.btn_l2.clicked.connect(lambda: self._on_apply_norm("l2"))
        layout.addWidget(self.btn_l2)

        # --- 4. Offset (Zerowanie w punkcie) ---
        self.lbl_offset_group = QLabel("<b>Offset (Punkt Zerowy)</b>")
        layout.addWidget(self.lbl_offset_group)

        row_off = QHBoxLayout()
        self.spin_offset_wn = QDoubleSpinBox()
        self.spin_offset_wn.setRange(400.0, 4000.0)
        self.spin_offset_wn.setValue(2000.0)
        self.btn_apply_offset = QPushButton("Zeruj w cm⁻¹")
        self.btn_apply_offset.clicked.connect(self._on_apply_offset)
        row_off.addWidget(self.spin_offset_wn)
        row_off.addWidget(self.btn_apply_offset)
        layout.addLayout(row_off)

        layout.addStretch()
        self.retranslate_ui()

    def _on_apply_asls(self):
        active = session.active_spectrum
        if not active:
            QMessageBox.warning(self, tr("app.warning"), "Wybierz widmo z drzewa projektu.")
            return

        lam = self.spin_lam.value()
        p = self.spin_p.value()
        baseline = prep.baseline_asls(active.intensities, lam=lam, p=p)
        active.intensities = active.intensities - baseline
        session.active_spectrum_changed.emit(active)

    def _on_apply_sg(self):
        active = session.active_spectrum
        if not active:
            return
        w = self.spin_sg_w.value()
        p = self.spin_sg_p.value()
        active.intensities = prep.apply_savgol(active.intensities, window=w, poly=p, deriv=0)
        session.active_spectrum_changed.emit(active)

    def _on_apply_norm(self, method: str):
        active = session.active_spectrum
        if not active:
            return
        if method == "snv":
            active.intensities = prep.apply_snv(active.intensities)
        elif method == "minmax":
            active.intensities = prep.apply_min_max(active.intensities)
        elif method == "l2":
            active.intensities = prep.apply_vector_norm_l2(active.intensities)
        session.active_spectrum_changed.emit(active)

    def _on_apply_offset(self):
        active = session.active_spectrum
        if not active:
            return
        wn_target = self.spin_offset_wn.value()
        active.intensities = prep.apply_offset(active.intensities, active.wavenumbers, wn_target, 0.0)
        session.active_spectrum_changed.emit(active)

    def retranslate_ui(self):
        self.lbl_asls_group.setText(tr("prep.asls_title", "<b>Korekcja Linii Bazowej (AsLS)</b>"))
        self.btn_apply_asls.setText(tr("prep.asls_btn", "Odejmij linię bazową AsLS"))
        self.lbl_sg_group.setText(tr("prep.sg_title", "<b>Wygładzanie (Savitzky-Golay)</b>"))
        self.btn_apply_sg.setText(tr("prep.sg_btn", "Zastosuj filtr SG"))
        self.lbl_norm_group.setText(tr("prep.norm_title", "<b>Transformacje i Normalizacje</b>"))
        self.btn_snv.setText(tr("prep.snv_btn", "Normalizacja SNV"))
        self.btn_minmax.setText(tr("prep.minmax_btn", "Skalowanie Min-Max (0-1)"))
        self.btn_l2.setText(tr("prep.l2_btn", "Normalizacja wektorowa (L2)"))
        self.btn_apply_offset.setText(tr("prep.offset_btn", "Zeruj w cm⁻¹"))