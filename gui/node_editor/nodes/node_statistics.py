"""
gui/node_editor/nodes/node_statistics.py
Węzły statystyczne potoku DAG: obliczanie widma średniego (Mean) i odchylenia
standardowego (SD) z wielokablowych i macierzowych źródeł z walidacją zgodności osi X.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QLineEdit, QCheckBox, QDoubleSpinBox, QLabel
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from algorithms.statistics import aggregate_spectral_ensemble


class SpectralMeanDialog(QDialog):
    """Okno konfiguracji węzła średniej arytmetycznej widm."""

    def __init__(self, node: 'SpectralMeanNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Parametry węzła: Średnia Arytmetyczna")
        self.resize(450, 290)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.ent_mean_name = QLineEdit(self.step.get('mean_name', 'Średnia'))
        self.ent_mean_name.setPlaceholderText("np. Średnia, Mean_Bufor")
        form.addRow("Nazwa widma średniego (Mean):", self.ent_mean_name)

        self.ent_sd_name = QLineEdit(self.step.get('sd_name', 'SD'))
        self.ent_sd_name.setPlaceholderText("np. SD, Odchylenie_SD")
        form.addRow("Nazwa odchylenia (SD):", self.ent_sd_name)

        self.chk_append_n = QCheckBox("Dopisz liczbę uśrednionych widm do nazwy, np. (n=5)")
        self.chk_append_n.setChecked(bool(self.step.get('append_count', True)))
        form.addRow("", self.chk_append_n)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-6, 10.0)
        self.spin_tol.setDecimals(5)
        self.spin_tol.setSingleStep(0.001)
        self.spin_tol.setValue(float(self.step.get('grid_tol', 1e-3)))
        form.addRow("Tolerancja zgodności osi X [cm⁻¹]:", self.spin_tol)

        self.chk_allow_single = QCheckBox("Zezwól na 1 widmo wejściowe (wówczas SD = 0)")
        self.chk_allow_single.setChecked(bool(self.step.get('allow_single', False)))
        form.addRow("", self.chk_allow_single)

        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet("color: #48cae4; font-weight: bold; font-size: 11px;")
        form.addRow("Podgląd nazw wyjściowych:", self.lbl_preview)

        lbl_info = QLabel(
            "💡 Węzeł weryfikuje zgodność liczby punktów, zakresu liczb falowych\n"
            "oraz położenia węzłów osi X wszystkich podłączonych widm."
        )
        lbl_info.setStyleSheet("color: #a8dadc; font-size: 11px; padding-top: 4px;")
        form.addRow(lbl_info)

        def _update_preview():
            m_base = self.ent_mean_name.text().strip() or "Średnia"
            s_base = self.ent_sd_name.text().strip() or "SD"
            suffix = " (n=5)" if self.chk_append_n.isChecked() else ""
            self.lbl_preview.setText(f"Mean: {m_base}{suffix}   |   SD: {s_base}{suffix}")

        self.ent_mean_name.textChanged.connect(_update_preview)
        self.ent_sd_name.textChanged.connect(_update_preview)
        self.chk_append_n.toggled.connect(_update_preview)
        _update_preview()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        m_name = self.ent_mean_name.text().strip() or "Średnia"
        s_name = self.ent_sd_name.text().strip() or "SD"
        self.step['mean_name'] = m_name
        self.step['sd_name'] = s_name
        self.step['append_count'] = self.chk_append_n.isChecked()
        self.step['grid_tol'] = float(self.spin_tol.value())
        self.step['allow_single'] = self.chk_allow_single.isChecked()
        self.step['name'] = f"Średnia ({m_name})"
        self.accept()


@register_node
class SpectralMeanNode(BaseNode):
    """Węzeł obliczający średnią arytmetyczną oraz odchylenie standardowe zgodnych widm."""

    node_type = "spectral_mean"
    title = "📊 Średnia Arytmetyczna"
    category = "Statystyka i Replikaty"
    header_gradient = ("#1d3557", "#2a9d8f")
    preview_port_id = "out"

    def __init__(self, step_data: Optional[Dict[str, Any]] = None):
        super().__init__(step_data)
        self.validation_error: Optional[str] = None
        self.last_count: int = 0

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Średnia (Mean + SD)",
            "mean_name": "Średnia",
            "sd_name": "SD",
            "append_count": True,
            "grid_tol": 1e-3,
            "allow_single": False
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(
                port_id="in",
                label="In",
                is_output=False,
                color=QColor("#ffb703"),
                allow_multiple=True,
                pos_y=90
            ),
            PortDef(
                port_id="out",
                label="Mean",
                is_output=True,
                color=QColor("#48cae4"),
                pos_y=70
            ),
            PortDef(
                port_id="out_sd",
                label="SD",
                is_output=True,
                color=QColor("#f4a261"),
                pos_y=112
            )
        ]

    def create_config_dialog(self, parent=None):
        return SpectralMeanDialog(self, parent=parent)

    @staticmethod
    def _validate_and_collect(
        sources: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        tol: float,
        allow_single: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Weryfikuje zgodność siatek X wszystkich wejściowych widm i łączy je w macierz (W x N).
        Zwraca: (ref_x, matrix_2d, error_message).
        """
        ref_x: Optional[np.ndarray] = None
        cols: List[np.ndarray] = []

        for src_idx, (sx, sy) in enumerate(sources):
            if sx is None or sy is None:
                continue

            cur_x = np.asarray(sx, dtype=np.float64).ravel()
            cur_y = np.asarray(sy, dtype=np.float64)

            if cur_x.size < 2 or cur_y.size < 2:
                return None, None, "Puste lub zbyt krótkie widmo wejściowe."

            if cur_y.shape[0] != cur_x.size:
                return None, None, f"Niezgodność wymiarów X/Y w źródle #{src_idx + 1}."

            if not (np.all(np.isfinite(cur_x)) and np.all(np.isfinite(cur_y))):
                return None, None, f"Wartości NaN/Inf w źródle #{src_idx + 1}."

            if ref_x is None:
                ref_x = cur_x.copy()
            else:
                # 1. Kontrola liczby punktów
                if cur_x.size != ref_x.size:
                    return (
                        None,
                        None,
                        f"Różna liczba punktów:\n{ref_x.size} ≠ {cur_x.size} pkt"
                    )

                # 2. Kontrola kierunku osi X (rosnąca vs malejąca)
                if (ref_x[0] > ref_x[-1]) != (cur_x[0] > cur_x[-1]):
                    return (
                        None,
                        None,
                        "Odwrócona kolejność\nosi liczb falowych X"
                    )

                # 3. Kontrola granic zakresu liczb falowych
                if abs(cur_x[0] - ref_x[0]) > tol or abs(cur_x[-1] - ref_x[-1]) > tol:
                    return (
                        None,
                        None,
                        f"Różny zakres osi X:\n[{ref_x[0]:.1f}..{ref_x[-1]:.1f}] vs\n[{cur_x[0]:.1f}..{cur_x[-1]:.1f}]"
                    )

                # 4. Kontrola zgodności każdego węzła siatki X
                if not np.allclose(ref_x, cur_x, rtol=0.0, atol=tol):
                    max_diff = float(np.max(np.abs(ref_x - cur_x)))
                    return (
                        None,
                        None,
                        f"Niezgodna siatka X\n(max Δ = {max_diff:.4g} cm⁻¹)"
                    )

            if cur_y.ndim == 2:
                for c in range(cur_y.shape[1]):
                    cols.append(cur_y[:, c])
            else:
                cols.append(cur_y.ravel())

        if ref_x is None or len(cols) == 0:
            return None, None, None  # Brak podłączonych danych (stan spoczynkowy)

        if len(cols) < 2 and not allow_single:
            return (
                None,
                None,
                f"Wymagane min. 2 widma\n(podłączono obecnie: {len(cols)})"
            )

        return ref_x, np.column_stack(cols), None

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sources = inputs.get("in_multi", [])
        if not sources:
            d_in = inputs.get("in", (None, None))
            if d_in[0] is not None:
                sources = [d_in]

        if not sources or all(s[0] is None for s in sources):
            self.validation_error = None
            self.last_count = 0
            self.has_preview = True
            self.step_data['name'] = f"Średnia ({self.step_data.get('mean_name', 'Średnia')})"
            return {"out": (None, None), "out_sd": (None, None)}, {"out": [], "out_sd": []}

        tol = float(self.step_data.get('grid_tol', 1e-3))
        allow_single = bool(self.step_data.get('allow_single', False))

        ref_x, matrix_2d, err_msg = self._validate_and_collect(sources, tol=tol, allow_single=allow_single)

        if err_msg is not None or ref_x is None or matrix_2d is None:
            self.validation_error = err_msg
            self.last_count = 0
            self.has_preview = False  # Ukrywa "Brak sygnału" i eksponuje komunikat błędu na kafelku
            self.step_data['name'] = "⚠️ Niezgodne widma"
            return {"out": (None, None), "out_sd": (None, None)}, {"out": [], "out_sd": []}

        self.validation_error = None
        self.has_preview = True
        n_spectra = matrix_2d.shape[1]
        self.last_count = n_spectra

        stats = aggregate_spectral_ensemble(matrix_2d)
        mean_y = stats["mean"]
        sd_y = stats["std"]

        m_base = self.step_data.get('mean_name', 'Średnia').strip() or 'Średnia'
        s_base = self.step_data.get('sd_name', 'SD').strip() or 'SD'
        n_tag = f" (n={n_spectra})" if self.step_data.get('append_count', True) else ""

        mean_label = f"{m_base}{n_tag}"
        sd_label = f"{s_base}{n_tag}"
        self.step_data['name'] = f"{m_base}{n_tag}"

        return (
            {
                "out": (ref_x.copy(), mean_y),
                "out_sd": (ref_x.copy(), sd_y)
            },
            {
                "out": [mean_label],
                "out_sd": [sd_label]
            }
        )

    def draw_custom_overlay(self, painter: QPainter, box_rect: QRectF):
        """Rysuje komunikat błędu walidacji lub dyskretną odznakę liczby uśrednianych widm."""
        if self.validation_error:
            painter.save()
            painter.setBrush(QBrush(QColor(150, 25, 35, 70)))
            painter.setPen(QPen(QColor("#ff4d6d"), 1.2))
            painter.drawRoundedRect(box_rect, 4, 4)

            title_rect = QRectF(box_rect.left() + 6, box_rect.top() + 8, box_rect.width() - 12, 18)
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.setPen(QColor("#ff758f"))
            painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignVCenter, "⚠️ Niezgodność widm")

            msg_rect = QRectF(box_rect.left() + 6, box_rect.top() + 28, box_rect.width() - 12, box_rect.height() - 34)
            painter.setFont(QFont("Arial", 7))
            painter.setPen(QColor("#ffe5ec"))
            painter.drawText(msg_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.validation_error)
            painter.restore()

        elif self.last_count > 0:
            painter.save()
            badge_rect = QRectF(box_rect.right() - 46, box_rect.top() + 3, 42, 14)
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.setPen(QColor("#8ecae6"))
            painter.drawText(badge_rect, Qt.AlignRight | Qt.AlignVCenter, f"n = {self.last_count}")
            painter.restore()
