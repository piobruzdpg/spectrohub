"""
gui/workspaces/ws_deconvolution.py
Zintegrowany panel perspektywy 2: Dekonwolucja pasm & Dekompozycja HDO.
Wielozakładkowy interfejs:
  - Zakładka 1: Dekonwolucja krzywych (Tryb Amid I / Tryb Ogólny)
  - Zakładka 2: Dekompozycja wieloskładnikowa / HDO Monte Carlo
"""

from typing import List, Dict, Any, Tuple, Optional
import copy
import logging
import numpy as np
from scipy.signal import find_peaks

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QMessageBox,
    QFrame, QCheckBox, QAbstractItemView, QTabWidget, QSpinBox,
    QDoubleSpinBox, QProgressBar, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QUndoCommand, QColor

from core.session import session
from core.models import SpectrumData
from core.i18n import tr, i18n
from algorithms.fitting.deconv_engine import DeconvolutionEngine
from algorithms.fitting.hdo_engine import HDOFitEngine
from algorithms import preprocessing as prep

logger = logging.getLogger("SpectroHub")

PEAK_COLORS = [
    '#FF00FF', '#228B22', '#DC143C', '#FFA500', '#0000CD',
    '#8B008B', '#FF1493', '#008080', '#FF4500', '#1E90FF', '#00CED1'
]

STRUCTURE_OPTIONS = [
    "Alpha-Helix", "Beta-Sheet", "Beta-Sheet (Anti)", "Random Coil",
    "Turns", "3_10 Helix", "Aggregates", "Aggregates (Anti)",
    "Side Chains", "Other"
]


# =====================================================================
# ZAKŁADKA 1: DEKONWOLUCJA PASMA (AMID I / OGÓLNA)
# =====================================================================

class DeconvChangeCommand(QUndoCommand):
    def __init__(self, panel: 'AmideDeconvPanel', description: str,
                 old_peaks: list, new_peaks: list):
        super().__init__(description)
        self.panel = panel
        self.old_peaks = copy.deepcopy(old_peaks)
        self.new_peaks = copy.deepcopy(new_peaks)

    def undo(self):
        self.panel.apply_peaks_state(self.old_peaks)

    def redo(self):
        self.panel.apply_peaks_state(self.new_peaks)


class AmideDeconvPanel(QWidget):
    """Inspektor dekonwolucji pasmowej z przełącznikiem Amid I / Ogólny."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks: List[Dict[str, Any]] = []
        self.peak_counter = 0
        self.show_deriv = False
        self._is_updating_table = False
        self._canvas_connected = False

        self._init_ui()
        self._connect_signals()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Wybór trybu działania
        row_mode = QHBoxLayout()
        self.lbl_mode = QLabel("Tryb:")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Amid I (Białka)", "Ogólny (Custom)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.lbl_mode)
        row_mode.addWidget(self.combo_mode)
        layout.addLayout(row_mode)

        # 2. Konfiguracja środowiska
        row_cfg1 = QHBoxLayout()
        self.lbl_solvent = QLabel("Rozpuszczalnik:")
        self.combo_solvent = QComboBox()
        self.combo_solvent.addItems(["H2O", "D2O"])
        self.combo_solvent.currentTextChanged.connect(self._on_solvent_changed)

        self.lbl_model = QLabel("Kształt:")
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Gaussian", "PseudoVoigt", "Voigt", "Lorentzian"])

        row_cfg1.addWidget(self.lbl_solvent)
        row_cfg1.addWidget(self.combo_solvent)
        row_cfg1.addWidget(self.lbl_model)
        row_cfg1.addWidget(self.combo_model)
        layout.addLayout(row_cfg1)

        row_cfg2 = QHBoxLayout()
        self.lbl_algo = QLabel("Algorytm:")
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Levenberg-Marquardt", "Nelder-Mead", "Powell"])
        row_cfg2.addWidget(self.lbl_algo)
        row_cfg2.addWidget(self.combo_algo)
        layout.addLayout(row_cfg2)

        # 3. Tabela pasm
        self.table = QTableWidget(0, 9)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table)

        # 4. Przyciski edycji i Undo/Redo
        row_btns = QHBoxLayout()
        self.btn_add_interactive = QPushButton("➕ Dodaj (Klik)")
        self.btn_add_interactive.setCheckable(True)
        self.btn_add_interactive.clicked.connect(self._toggle_interactive_add)

        self.btn_undo = QPushButton("↶ Cofnij")
        self.btn_undo.clicked.connect(session.undo_stack.undo)
        session.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.btn_undo.setEnabled(False)

        self.btn_redo = QPushButton("↷ Ponów")
        self.btn_redo.clicked.connect(session.undo_stack.redo)
        session.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_redo.setEnabled(False)

        self.btn_clear_peaks = QPushButton("🗑 Wyczyść")
        self.btn_clear_peaks.clicked.connect(self._on_clear_peaks)

        row_btns.addWidget(self.btn_add_interactive)
        row_btns.addWidget(self.btn_undo)
        row_btns.addWidget(self.btn_redo)
        row_btns.addWidget(self.btn_clear_peaks)
        layout.addLayout(row_btns)

        # 5. Asystent 2. pochodnej
        row_deriv = QHBoxLayout()
        self.btn_toggle_deriv = QPushButton("📉 2. Pochodna")
        self.btn_toggle_deriv.setCheckable(True)
        self.btn_toggle_deriv.clicked.connect(self._on_toggle_deriv)

        self.btn_auto_peaks = QPushButton("🎯 Szukaj z 2. pochodnej")
        self.btn_auto_peaks.clicked.connect(self._on_auto_find_peaks)

        row_deriv.addWidget(self.btn_toggle_deriv)
        row_deriv.addWidget(self.btn_auto_peaks)
        layout.addLayout(row_deriv)

        # 6. Silnik dopasowujący
        self.btn_fit = QPushButton("⚡ Dopasuj Pasma")
        self.btn_fit.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 8px;")
        self.btn_fit.clicked.connect(self._on_run_fit)
        layout.addWidget(self.btn_fit)

        # 7. Panel metryk jakości (GoF)
        self.frame_gof = QFrame()
        self.frame_gof.setStyleSheet("background-color: #252530; border-radius: 4px; padding: 6px;")
        l_gof = QVBoxLayout(self.frame_gof)
        l_gof.setSpacing(2)
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

    def _connect_signals(self):
        session.active_spectrum_changed.connect(self._on_active_spectrum_changed)
        QTimer.singleShot(100, self._ensure_canvas_connected)

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _ensure_canvas_connected(self):
        if self._canvas_connected:
            return
        canvas = self._get_canvas()
        if canvas:
            canvas.peak_add_requested.connect(self.on_canvas_add_peak)
            self._canvas_connected = True

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_canvas_connected()
        if self.peaks or self.show_deriv:
            self._update_curves_and_canvas()
        else:
            canvas = self._get_canvas()
            if canvas:
                canvas.redraw_all()

    def hideEvent(self, event):
        super().hideEvent(event)
        canvas = self._get_canvas()
        if canvas:
            canvas.redraw_all()

    def _on_active_spectrum_changed(self, spectrum):
        self._ensure_canvas_connected()
        # Bezpiecznik: rysujemy nakładkę tylko jeśli Workspace 2 jest widoczny
        # i faktycznie mamy zdefiniowane pasma lub włączoną pochodną!
        if not self.isVisible():
            return
        if self.peaks or self.show_deriv:
            self._update_curves_and_canvas()

    def _is_general_mode(self) -> bool:
        return self.combo_mode.currentIndex() == 1

    def _on_mode_changed(self, idx: int):
        is_general = (idx == 1)
        self.lbl_solvent.setVisible(not is_general)
        self.combo_solvent.setVisible(not is_general)
        self.retranslate_ui()
        self._refresh_table()

    def _toggle_interactive_add(self, checked: bool):
        canvas = self._get_canvas()
        if canvas:
            canvas.interaction_mode = "add_peak" if checked else "normal"
            if checked:
                self.btn_add_interactive.setStyleSheet("background-color: #e76f51; font-weight: bold;")
                self.btn_add_interactive.setText(tr("deconv.click_plot", "Kliknij na wykresie..."))
            else:
                self.btn_add_interactive.setStyleSheet("")
                self.btn_add_interactive.setText(tr("deconv.add_interactive", "➕ Dodaj (Klik)"))

    def on_canvas_add_peak(self, x: float, y: float):
        active = session.active_spectrum
        if not active:
            return

        idx_closest = int((np.abs(active.wavenumbers - x)).argmin())
        y_val = float(active.intensities[idx_closest]) if y <= 0 else y

        self.peak_counter += 1
        model_type = self.combo_model.currentText()
        solvent = self.combo_solvent.currentText()

        if self._is_general_mode():
            struct_guess = f"Peak {self.peak_counter}"
        else:
            struct_guess = DeconvolutionEngine.guess_structure(x, solvent=solvent)

        sigma = 10.0
        if model_type == 'Lorentzian':
            area = float(y_val * sigma * np.pi)
        else:
            area = float(y_val * sigma * np.sqrt(2.0 * np.pi))

        color = PEAK_COLORS[(self.peak_counter - 1) % len(PEAK_COLORS)]

        new_peak = {
            'id': f"p{self.peak_counter}",
            'type': model_type,
            'center': float(x),
            'sigma': float(sigma),
            'area': max(area, 1e-4),
            'structure': struct_guess,
            'color': color,
            'lock_center': False,
            'lock_sigma': False,
            'lock_area': False,
            'gamma': float(sigma),
            'fraction': 0.5,
            'lock_extra': False
        }

        old_peaks = copy.deepcopy(self.peaks)
        new_peaks = copy.deepcopy(self.peaks) + [new_peak]

        cmd = DeconvChangeCommand(self, f"Dodaj pasmo {new_peak['id']}", old_peaks, new_peaks)
        session.undo_stack.push(cmd)

    def apply_peaks_state(self, peaks: list):
        self.peaks = copy.deepcopy(peaks)
        self._refresh_table()
        self._update_curves_and_canvas()

    def _on_clear_peaks(self):
        if not self.peaks:
            return
        cmd = DeconvChangeCommand(self, "Wyczyść pasma", self.peaks, [])
        session.undo_stack.push(cmd)

    def _on_delete_peak(self, row: int):
        if 0 <= row < len(self.peaks):
            old_peaks = copy.deepcopy(self.peaks)
            new_peaks = copy.deepcopy(self.peaks)
            deleted = new_peaks.pop(row)
            cmd = DeconvChangeCommand(self, f"Usuń pasmo {deleted['id']}", old_peaks, new_peaks)
            session.undo_stack.push(cmd)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if self._is_updating_table:
            return

        row = item.row()
        col = item.column()
        if not (0 <= row < len(self.peaks)):
            return

        peak = self.peaks[row]
        old_peaks = copy.deepcopy(self.peaks)

        try:
            if col == 6 and self._is_general_mode():
                peak['structure'] = item.text()
                cmd = DeconvChangeCommand(self, f"Etykieta {peak['id']}", old_peaks, self.peaks)
                session.undo_stack.push(cmd)
                return

            val = float(item.text().replace(',', '.'))
            if col == 2:
                peak['center'] = val
                if not self._is_general_mode():
                    peak['structure'] = DeconvolutionEngine.guess_structure(
                        val, solvent=self.combo_solvent.currentText()
                    )
            elif col == 3:
                factor = 2.0 if peak['type'] == 'Lorentzian' else 2.35482
                peak['sigma'] = max(1e-4, val / factor)
            elif col == 4:
                peak['area'] = max(1e-6, val)

            cmd = DeconvChangeCommand(self, f"Edycja {peak['id']}", old_peaks, self.peaks)
            session.undo_stack.push(cmd)

        except ValueError:
            self._refresh_table()

    def _get_widget_row(self, widget: QWidget) -> int:
        curr = widget
        while curr and curr.parentWidget() != self.table.viewport():
            curr = curr.parentWidget()
        if curr:
            return self.table.indexAt(curr.pos()).row()
        return -1

    def _refresh_table(self):
        self._is_updating_table = True
        self.table.blockSignals(True)

        n_peaks = len(self.peaks)
        curr_rows = self.table.rowCount()

        if curr_rows > n_peaks:
            for r in range(curr_rows - 1, n_peaks - 1, -1):
                self.table.removeRow(r)
        elif curr_rows < n_peaks:
            self.table.setRowCount(n_peaks)

        is_general = self._is_general_mode()
        if is_general:
            valid_area = sum(p['area'] for p in self.peaks)
        else:
            valid_area = sum(
                p['area'] for p in self.peaks
                if 1600 <= p['center'] <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]
            )

        for row, p in enumerate(self.peaks):
            fwhm = p['sigma'] * (2.0 if p['type'] == 'Lorentzian' else 2.35482)

            if is_general:
                perc = (p['area'] / valid_area * 100.0) if valid_area > 0 else 0.0
                perc_str = f"{perc:.1f} %"
            else:
                if 1600 <= p['center'] <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]:
                    perc = (p['area'] / valid_area * 100.0) if valid_area > 0 else 0.0
                    perc_str = f"{perc:.1f} %"
                else:
                    perc_str = "—"

            # Col 0: Pik ID
            item_id = self.table.item(row, 0)
            if not item_id:
                item_id = QTableWidgetItem()
                item_id.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(row, 0, item_id)
            item_id.setText(f"■ {p['id']}")
            item_id.setForeground(QColor(p.get('color', '#ffffff')))

            # Col 1: Typ
            combo_t = self.table.cellWidget(row, 1)
            if not combo_t:
                combo_t = QComboBox()
                combo_t.addItems(["Gaussian", "PseudoVoigt", "Voigt", "Lorentzian"])
                combo_t.currentTextChanged.connect(self._on_cell_type_changed)
                self.table.setCellWidget(row, 1, combo_t)
            combo_t.blockSignals(True)
            combo_t.setCurrentText(p['type'])
            combo_t.blockSignals(False)

            self._set_or_update_item(row, 2, f"{p['center']:.1f}")
            self._set_or_update_item(row, 3, f"{fwhm:.1f}")
            self._set_or_update_item(row, 4, f"{p['area']:.2f}")

            # Col 5: %
            item_pct = self.table.item(row, 5)
            if not item_pct:
                item_pct = QTableWidgetItem()
                item_pct.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(row, 5, item_pct)
            item_pct.setText(perc_str)

            # Col 6: Struktura / Etykieta
            if is_general:
                self.table.removeCellWidget(row, 6)
                self._set_or_update_item(row, 6, p.get('structure', f"Peak {row + 1}"))
            else:
                item_s = self.table.item(row, 6)
                if item_s:
                    self.table.takeItem(row, 6)
                combo_s = self.table.cellWidget(row, 6)
                if not combo_s:
                    combo_s = QComboBox()
                    combo_s.addItems(STRUCTURE_OPTIONS)
                    combo_s.currentTextChanged.connect(self._on_cell_struct_changed)
                    self.table.setCellWidget(row, 6, combo_s)
                combo_s.blockSignals(True)
                combo_s.setCurrentText(p.get('structure', 'Other'))
                combo_s.blockSignals(False)

            # Col 7: Blokady
            frame_locks = self.table.cellWidget(row, 7)
            if not frame_locks:
                frame_locks = QWidget()
                l_locks = QHBoxLayout(frame_locks)
                l_locks.setContentsMargins(2, 0, 2, 0)
                l_locks.setSpacing(4)
                chk_c = QCheckBox("C")
                chk_c.setProperty("lock_key", "lock_center")
                chk_c.toggled.connect(self._on_cell_lock_toggled)

                chk_w = QCheckBox("W")
                chk_w.setProperty("lock_key", "lock_sigma")
                chk_w.toggled.connect(self._on_cell_lock_toggled)

                chk_a = QCheckBox("A")
                chk_a.setProperty("lock_key", "lock_area")
                chk_a.toggled.connect(self._on_cell_lock_toggled)

                l_locks.addWidget(chk_c)
                l_locks.addWidget(chk_w)
                l_locks.addWidget(chk_a)
                self.table.setCellWidget(row, 7, frame_locks)

            chk_c, chk_w, chk_a = frame_locks.findChildren(QCheckBox)
            for chk in (chk_c, chk_w, chk_a):
                chk.blockSignals(True)
            chk_c.setChecked(p.get('lock_center', False))
            chk_w.setChecked(p.get('lock_sigma', False))
            chk_a.setChecked(p.get('lock_area', False))
            for chk in (chk_c, chk_w, chk_a):
                chk.blockSignals(False)

            # Col 8: Usuń ✕
            btn_del = self.table.cellWidget(row, 8)
            if not btn_del:
                btn_del = QPushButton("✕")
                btn_del.setStyleSheet("color: #ff6b6b; font-weight: bold; max-width: 24px;")
                btn_del.clicked.connect(self._on_cell_delete_clicked)
                self.table.setCellWidget(row, 8, btn_del)

        self.table.blockSignals(False)
        self._is_updating_table = False

    def _set_or_update_item(self, row: int, col: int, text: str):
        item = self.table.item(row, col)
        if not item:
            item = QTableWidgetItem(text)
            self.table.setItem(row, col, item)
        else:
            item.setText(text)

    def _on_cell_type_changed(self, text: str):
        combo = self.sender()
        if not combo or self._is_updating_table:
            return
        row = self._get_widget_row(combo)
        if 0 <= row < len(self.peaks):
            self.peaks[row]['type'] = text
            self._update_curves_and_canvas()

    def _on_cell_struct_changed(self, text: str):
        combo = self.sender()
        if not combo or self._is_updating_table:
            return
        row = self._get_widget_row(combo)
        if 0 <= row < len(self.peaks):
            self.peaks[row]['structure'] = text
            self._update_curves_and_canvas()

    def _on_cell_lock_toggled(self, checked: bool):
        chk = self.sender()
        if not chk or self._is_updating_table:
            return
        row = self._get_widget_row(chk)
        lock_key = chk.property("lock_key")
        if 0 <= row < len(self.peaks) and lock_key:
            self.peaks[row][lock_key] = checked

    def _on_cell_delete_clicked(self):
        btn = self.sender()
        if not btn:
            return
        row = self._get_widget_row(btn)
        if 0 <= row < len(self.peaks):
            self._on_delete_peak(row)

    def _on_solvent_changed(self, new_solvent: str):
        if not self._is_general_mode():
            for p in self.peaks:
                p['structure'] = DeconvolutionEngine.guess_structure(p['center'], solvent=new_solvent)
            self._refresh_table()
            self._update_curves_and_canvas()

    def _update_curves_and_canvas(self):
        active = session.active_spectrum
        canvas = self._get_canvas()
        if not canvas:
            return

        if not active or not self.peaks:
            canvas.update_deconv_overlay(active, None, [], self.show_deriv, self._get_deriv_data())
            return

        curves = DeconvolutionEngine.evaluate_model_curves(
            active.wavenumbers, active.intensities, self.peaks
        )
        canvas.update_deconv_overlay(active, curves, self.peaks, self.show_deriv, self._get_deriv_data())

    def _get_deriv_data(self):
        active = session.active_spectrum
        if not active or not self.show_deriv:
            return None
        return prep.get_second_derivative_inverted(active.wavenumbers, active.intensities, window=13, poly=3)

    def _on_toggle_deriv(self, checked: bool):
        self.show_deriv = checked
        self._update_curves_and_canvas()

    def _on_auto_find_peaks(self):
        active = session.active_spectrum
        if not active:
            return

        x, d2y = prep.get_second_derivative_inverted(active.wavenumbers, active.intensities, window=13, poly=3)

        if not self._is_general_mode():
            mask = (x >= 1600.0) & (x <= 1700.0)
            x_sub, d2y_sub = x[mask], d2y[mask]
        else:
            x_sub, d2y_sub = x, d2y

        if len(d2y_sub) == 0:
            return

        prominence = 0.08 * (np.max(d2y_sub) - np.min(d2y_sub))
        peak_indices, _ = find_peaks(d2y_sub, distance=6, prominence=prominence)

        if len(peak_indices) == 0:
            QMessageBox.information(self, tr("app.info"), tr("deconv.no_peaks_found", "Nie wykryto wyraźnych pików w 2. pochodnej."))
            return

        old_peaks = copy.deepcopy(self.peaks)
        new_peaks = copy.deepcopy(self.peaks)
        model_type = self.combo_model.currentText()
        solvent = self.combo_solvent.currentText()

        for idx in peak_indices:
            peak_x = float(x_sub[idx])
            idx_closest = int((np.abs(active.wavenumbers - peak_x)).argmin())
            y_val = float(active.intensities[idx_closest])

            self.peak_counter += 1
            if self._is_general_mode():
                struct = f"Peak {self.peak_counter}"
            else:
                struct = DeconvolutionEngine.guess_structure(peak_x, solvent=solvent)

            sigma = 10.0
            area = float(y_val * sigma * np.sqrt(2.0 * np.pi))
            color = PEAK_COLORS[(self.peak_counter - 1) % len(PEAK_COLORS)]

            new_peaks.append({
                'id': f"p{self.peak_counter}",
                'type': model_type,
                'center': peak_x,
                'sigma': sigma,
                'area': max(area, 1e-4),
                'structure': struct,
                'color': color,
                'lock_center': False,
                'lock_sigma': False,
                'lock_area': False,
                'gamma': sigma,
                'fraction': 0.5,
                'lock_extra': False
            })

        cmd = DeconvChangeCommand(self, f"Auto-detekcja ({len(peak_indices)} pików)", old_peaks, new_peaks)
        session.undo_stack.push(cmd)

    def _on_run_fit(self):
        active = session.active_spectrum
        if not active or not self.peaks:
            QMessageBox.warning(self, tr("app.warning"), tr("deconv.warn_need_peaks", "Wymagane jest aktywne widmo i co najmniej jedno pasmo."))
            return

        method_map = {
            "Levenberg-Marquardt": "leastsq",
            "Nelder-Mead": "nelder",
            "Powell": "powell"
        }
        method = method_map.get(self.combo_algo.currentText(), "leastsq")

        try:
            centers = [p['center'] for p in self.peaks]
            min_c, max_c = min(centers), max(centers)
            fit_mask = (active.wavenumbers >= min_c - 40.0) & (active.wavenumbers <= max_c + 40.0)

            if np.sum(fit_mask) > len(self.peaks) * 3:
                x_fit = active.wavenumbers[fit_mask]
                y_fit = active.intensities[fit_mask]
            else:
                x_fit = active.wavenumbers
                y_fit = active.intensities

            old_peaks = copy.deepcopy(self.peaks)
            updated_peaks, gof = DeconvolutionEngine.fit_peaks(x_fit, y_fit, self.peaks, method=method)

            cmd = DeconvChangeCommand(self, f"Dopasowanie ({method})", old_peaks, updated_peaks)
            session.undo_stack.push(cmd)

            self.lbl_r2.setText(f"R²: {gof['r2']:.5f}")
            self.lbl_rmse.setText(f"RMSE: {gof['rmse']:.6f}")
            self.lbl_chi.setText(f"χ²: {gof['chisqr']:.2e} | Red-χ²: {gof['redchi']:.2e}")

        except Exception as e:
            logger.error(f"Błąd dopasowania lmfit ({method}): {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd procedury dopasowania:\n{e}")

    def retranslate_ui(self):
        self.lbl_mode.setText(tr("deconv.mode", "Tryb:"))
        self.lbl_solvent.setText(tr("deconv.solvent", "Rozpuszczalnik:"))
        self.lbl_model.setText(tr("deconv.model", "Kształt:"))
        self.lbl_algo.setText(tr("deconv.algo", "Algorytm:"))
        self.btn_clear_peaks.setText(tr("deconv.clear_peaks", "🗑 Wyczyść"))
        self.btn_fit.setText(tr("deconv.fit_btn", "⚡ Dopasuj Pasma"))
        self.btn_undo.setText(tr("app.undo", "↶ Cofnij"))
        self.btn_redo.setText(tr("app.redo", "↷ Ponów"))
        self.btn_toggle_deriv.setText(tr("deconv.deriv_btn", "📉 2. Pochodna"))
        self.btn_auto_peaks.setText(tr("deconv.auto_peaks", "🎯 Szukaj z 2. pochodnej"))
        self.lbl_gof_title.setText(f"<b>{tr('deconv.gof_title', 'Jakość dopasowania (GoF):')}</b>")

        col5_txt = tr("deconv.table.area_pct", "% Całkowite") if self._is_general_mode() else tr("deconv.table.amide_pct", "% Amid I")
        col6_txt = tr("deconv.table.label", "Etykieta") if self._is_general_mode() else tr("deconv.table.structure", "Struktura")

        self.table.setHorizontalHeaderLabels([
            tr("deconv.table.peak", "Pik"),
            tr("deconv.table.type", "Typ"),
            tr("deconv.table.center", "Pozycja"),
            tr("deconv.table.fwhm", "FWHM"),
            tr("deconv.table.area", "Pole"),
            col5_txt, col6_txt,
            tr("deconv.table.locks", "Lock (C/W/A)"),
            "✕"
        ])


# =====================================================================
# ZAKŁADKA 2: UOGÓLNIONA DEKOMPOZYCJA WIDM / HDO FIT (MONTE CARLO)
# =====================================================================

class DecompositionWorkerThread(QThread):
    """Wątek tła dla wieloprocesowych obliczeń Monte Carlo w HDO Engine."""
    progress = Signal(int, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, wavenb: np.ndarray, target: np.ndarray,
                 base_spectra: List[np.ndarray], comp_bounds: List[Tuple[float, float]],
                 n_peaks: int, n_iterations: int, penalty_weight: float,
                 st_dev: float, baseline_active: bool):
        super().__init__()
        self.wavenb = wavenb
        self.target = target
        self.base_spectra = base_spectra
        self.comp_bounds = comp_bounds
        self.n_peaks = n_peaks
        self.n_iterations = n_iterations
        self.penalty_weight = penalty_weight
        self.st_dev = st_dev
        self.baseline_active = baseline_active

    def run(self):
        try:
            results = HDOFitEngine.run_monte_carlo(
                wavenb=self.wavenb,
                target=self.target,
                base_spectra_list=self.base_spectra,
                comp_bounds=self.comp_bounds,
                n_peaks=self.n_peaks,
                n_iterations=self.n_iterations,
                penalty_weight=self.penalty_weight,
                st_dev=self.st_dev,
                baseline_active=self.baseline_active,
                progress_callback=self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            logger.error(f"Błąd obliczeń HDO Worker: {e}")
            self.error.emit(str(e))


class GeneralizedDecompositionPanel(QWidget):
    """Inspektor dla uniwersalnej dekompozycji k widm czystych i m pasm syntetycznych."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[DecompositionWorkerThread] = None
        self.last_results: Optional[Dict[str, Any]] = None
        self.selected_target_name: Optional[str] = None

        self._init_ui()
        self._connect_signals()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. Widmo badane (W0)
        group_target = QGroupBox("1. Widmo Badane (W₀)")
        self.group_target = group_target
        l_target = QVBoxLayout(group_target)

        row_w0 = QHBoxLayout()
        self.lbl_target = QLabel("Widmo:")
        self.combo_target = QComboBox()
        self.lbl_n = QLabel("Mnożnik N:")
        self.spin_n = QDoubleSpinBox()
        self.spin_n.setRange(0.01, 100.0)
        self.spin_n.setValue(1.0)
        self.spin_n.setSingleStep(0.1)

        row_w0.addWidget(self.lbl_target)
        row_w0.addWidget(self.combo_target, 1)
        row_w0.addWidget(self.lbl_n)
        row_w0.addWidget(self.spin_n)
        l_target.addLayout(row_w0)
        layout.addWidget(group_target)

        # 2. Lista widm bazowych (W1, ..., Wk)
        group_bases = QGroupBox("2. Widma Składowe (W₁, W₂, ..., Wₖ)")
        self.group_bases = group_bases
        l_bases = QVBoxLayout(group_bases)

        self.table_bases = QTableWidget(0, 4)
        self.table_bases.setHorizontalHeaderLabels(["Widmo z sesji", "Waga Min", "Waga Max", "✕"])
        self.table_bases.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_bases.setSelectionBehavior(QAbstractItemView.SelectRows)
        l_bases.addWidget(self.table_bases)

        row_add_base = QHBoxLayout()
        self.btn_add_base = QPushButton("➕ Dodaj widmo bazowe")
        self.btn_add_base.clicked.connect(self._on_add_base_clicked)
        row_add_base.addWidget(self.btn_add_base)
        l_bases.addLayout(row_add_base)
        layout.addWidget(group_bases)

        # 3. Parametry pasm syntetycznych i linii bazowej
        group_synth = QGroupBox("3. Piki Syntetyczne & Linia Bazowa")
        self.group_synth = group_synth
        l_synth = QVBoxLayout(group_synth)

        row_s1 = QHBoxLayout()
        self.lbl_synth_peaks = QLabel("Liczba pików G×L:")
        self.spin_synth_peaks = QSpinBox()
        self.spin_synth_peaks.setRange(0, 10)
        self.spin_synth_peaks.setValue(0)

        self.chk_baseline = QCheckBox("Linia bazowa (kwadratowa)")
        self.chk_baseline.setChecked(False)

        row_s1.addWidget(self.lbl_synth_peaks)
        row_s1.addWidget(self.spin_synth_peaks)
        row_s1.addWidget(self.chk_baseline)
        l_synth.addLayout(row_s1)
        layout.addWidget(group_synth)

        # 4. Parametry Monte Carlo
        group_mc = QGroupBox("4. Parametry Monte Carlo")
        self.group_mc = group_mc
        l_mc = QVBoxLayout(group_mc)

        row_mc1 = QHBoxLayout()
        self.lbl_iter = QLabel("L. wariantów (MC):")
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(500, 50000)
        self.spin_iter.setValue(3000)
        self.spin_iter.setSingleStep(500)

        self.lbl_penalty = QLabel("Wsp. kary:")
        self.spin_penalty = QDoubleSpinBox()
        self.spin_penalty.setRange(0.0, 500.0)
        self.spin_penalty.setValue(10.0)

        row_mc1.addWidget(self.lbl_iter)
        row_mc1.addWidget(self.spin_iter)
        row_mc1.addWidget(self.lbl_penalty)
        row_mc1.addWidget(self.spin_penalty)
        l_mc.addLayout(row_mc1)
        layout.addWidget(group_mc)

        # 5. Przyciski startu i pasek postępu
        self.btn_run_decomp = QPushButton("⚡ Uruchom Dekompozycję (Monte Carlo)")
        self.btn_run_decomp.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 8px;")
        self.btn_run_decomp.clicked.connect(self._on_run_decomp_clicked)
        layout.addWidget(self.btn_run_decomp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Status: Gotowy")
        self.lbl_status.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # 6. Wyniki & Eksport
        group_results = QGroupBox("5. Wyniki & Eksport")
        self.group_results = group_results
        l_res = QVBoxLayout(group_results)

        self.table_results = QTableWidget(0, 3)
        self.table_results.setHorizontalHeaderLabels(["Składnik", "Waga (c)", "Udział [%]"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l_res.addWidget(self.table_results)

        self.lbl_decomp_gof = QLabel("R²: — | RMSE: — | χ²: —")
        self.lbl_decomp_gof.setStyleSheet("font-weight: bold; color: #48cae4;")
        l_res.addWidget(self.lbl_decomp_gof)

        self.btn_add_residual_to_session = QPushButton("📥 Dodaj Czyste Residuum do Projektu")
        self.btn_add_residual_to_session.setStyleSheet("background-color: #e76f51; font-weight: bold;")
        self.btn_add_residual_to_session.setEnabled(False)
        self.btn_add_residual_to_session.clicked.connect(self._on_add_residual_to_session)

        self.btn_add_fit_to_session = QPushButton("➕ Dodaj Krzywą Dopasowaną do Projektu")
        self.btn_add_fit_to_session.setEnabled(False)
        self.btn_add_fit_to_session.clicked.connect(self._on_add_fit_to_session)

        l_res.addWidget(self.btn_add_residual_to_session)
        l_res.addWidget(self.btn_add_fit_to_session)
        layout.addWidget(group_results)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        self.retranslate_ui()

    def _connect_signals(self):
        session.spectrum_added.connect(self._refresh_combos)
        session.spectrum_removed.connect(self._refresh_combos)
        session.active_spectrum_changed.connect(self._on_active_changed)

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_combos()

    def _on_active_changed(self, spec: Optional[SpectrumData]):
        if spec:
            idx = self.combo_target.findText(spec.name)
            if idx >= 0:
                self.combo_target.setCurrentIndex(idx)

    @staticmethod
    def _sync_combo_items(combo: QComboBox, new_items: list):
        current_items = [combo.itemText(i) for i in range(combo.count())]
        if current_items == new_items:
            return

        current_val = combo.currentText()
        combo.blockSignals(True)

        for item in current_items:
            if item not in new_items:
                idx = combo.findText(item)
                if idx >= 0:
                    combo.removeItem(idx)

        for item in new_items:
            if combo.findText(item) < 0:
                combo.addItem(item)

        if current_val in new_items:
            combo.setCurrentText(current_val)

        combo.blockSignals(False)

    def _refresh_combos(self):
        all_names = list(session.spectra.keys())
        self._sync_combo_items(self.combo_target, all_names)
        if not self.combo_target.currentText() and session.active_spectrum:
            self.combo_target.setCurrentText(session.active_spectrum.name)

        for row in range(self.table_bases.rowCount()):
            combo = self.table_bases.cellWidget(row, 0)
            if isinstance(combo, QComboBox):
                try:
                    self._sync_combo_items(combo, all_names)
                except RuntimeError:
                    pass

    def _on_add_base_clicked(self):
        all_names = list(session.spectra.keys())
        if not all_names:
            QMessageBox.information(self, tr("app.info"), tr("decomp.no_spectra", "Brak załadowanych widm w projekcie."))
            return

        row = self.table_bases.rowCount()
        self.table_bases.insertRow(row)

        combo = QComboBox()
        combo.addItems(all_names)
        if row < len(all_names):
            combo.setCurrentIndex(row)
        self.table_bases.setCellWidget(row, 0, combo)

        spin_min = QDoubleSpinBox()
        spin_min.setRange(0.0, 50.0)
        spin_min.setValue(0.0)
        spin_min.setSingleStep(0.05)
        self.table_bases.setCellWidget(row, 1, spin_min)

        spin_max = QDoubleSpinBox()
        spin_max.setRange(0.0, 50.0)
        spin_max.setValue(1.2)
        spin_max.setSingleStep(0.05)
        self.table_bases.setCellWidget(row, 2, spin_max)

        btn_del = QPushButton("✕")
        btn_del.setStyleSheet("color: #ff6b6b; font-weight: bold; max-width: 24px;")
        btn_del.clicked.connect(self._on_del_base_row)
        self.table_bases.setCellWidget(row, 3, btn_del)

    def _on_del_base_row(self):
        btn = self.sender()
        for r in range(self.table_bases.rowCount()):
            if self.table_bases.cellWidget(r, 3) == btn:
                self.table_bases.removeRow(r)
                break

    def _on_run_decomp_clicked(self):
        target_name = self.combo_target.currentText()
        if not target_name or target_name not in session.spectra:
            QMessageBox.warning(self, tr("app.warning"), tr("decomp.select_target", "Wybierz poprawne widmo badane (W₀)."))
            return

        target_spec = session.spectra[target_name]
        wavenb = target_spec.wavenumbers
        N_val = self.spin_n.value()
        target_y = N_val * target_spec.intensities

        base_spectra = []
        comp_bounds = []
        n_bases = self.table_bases.rowCount()

        if n_bases == 0:
            QMessageBox.warning(self, tr("app.warning"), tr("decomp.need_bases", "Dodaj co najmniej jedno widmo bazowe."))
            return

        for r in range(n_bases):
            combo = self.table_bases.cellWidget(r, 0)
            spin_min = self.table_bases.cellWidget(r, 1)
            spin_max = self.table_bases.cellWidget(r, 2)

            name = combo.currentText()
            if name not in session.spectra:
                continue

            spec = session.spectra[name]
            y_interp = np.interp(wavenb, spec.wavenumbers, spec.intensities)
            base_spectra.append(y_interp)
            comp_bounds.append((spin_min.value(), spin_max.value()))

        self.btn_run_decomp.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText(tr("decomp.status_calc", "Status: Trwają równoległe obliczenia Monte Carlo..."))
        self.lbl_status.setStyleSheet("color: #f4a261; font-weight: bold;")

        self.worker = DecompositionWorkerThread(
            wavenb=wavenb,
            target=target_y,
            base_spectra=base_spectra,
            comp_bounds=comp_bounds,
            n_peaks=self.spin_synth_peaks.value(),
            n_iterations=self.spin_iter.value(),
            penalty_weight=self.spin_penalty.value(),
            st_dev=0.01,
            baseline_active=self.chk_baseline.isChecked()
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, pct: int, best_rss: float):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"{tr('decomp.status_calc', 'Obliczenia')}: {pct}% (Najlepsze RSS: {best_rss:.4e})")

    def _on_worker_finished(self, results: Dict[str, Any]):
        self.last_results = results
        self.btn_run_decomp.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(tr("decomp.status_done", "Status: Zakończono sukcesem!"))
        self.lbl_status.setStyleSheet("color: #2a9d8f; font-weight: bold;")

        self.btn_add_residual_to_session.setEnabled(True)
        self.btn_add_fit_to_session.setEnabled(True)

        self.table_results.setRowCount(0)
        coeffs = results['coefficients']
        percentages = results['area_percentages']

        for r in range(len(coeffs)):
            combo = self.table_bases.cellWidget(r, 0)
            comp_name = combo.currentText() if combo else f"W{r + 1}"
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(row, 0, QTableWidgetItem(f"{comp_name}"))
            self.table_results.setItem(row, 1, QTableWidgetItem(f"{coeffs[r]:.4f}"))
            self.table_results.setItem(row, 2, QTableWidgetItem(f"{percentages[r]:.1f} %"))

        for idx, p in enumerate(results.get('peaks_params', [])):
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(row, 0, QTableWidgetItem(f"Pik GxL {idx + 1} ({p['center']:.1f} cm⁻¹)"))
            self.table_results.setItem(row, 1, QTableWidgetItem(f"Amp: {p['amp']:.3f}"))
            self.table_results.setItem(row, 2, QTableWidgetItem(f"W: {p['width']:.1f}"))

        self.lbl_decomp_gof.setText(
            f"R²: {results['r2']:.5f} | RMSE: {results['rmse']:.5f} | Red-χ²: {results['chi2_red']:.2e}"
        )

        self._update_canvas_plot()

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _update_canvas_plot(self):
        canvas = self._get_canvas()
        if not canvas or not self.last_results:
            return

        target_name = self.combo_target.currentText()
        target_spec = session.spectra.get(target_name)
        if not target_spec:
            return

        comp_names = [
            self.table_bases.cellWidget(r, 0).currentText()
            for r in range(self.table_bases.rowCount())
            if self.table_bases.cellWidget(r, 0)
        ]

        scaled_target = SpectrumData(
            wavenumbers=target_spec.wavenumbers,
            intensities=target_spec.intensities * self.spin_n.value(),
            name=target_spec.name,
            file_path=target_spec.file_path,
            file_type=target_spec.file_type
        )

        if hasattr(canvas, "update_decomposition_overlay"):
            canvas.update_decomposition_overlay(
                target_spec=scaled_target,
                fitted_y=self.last_results['fitted_curve'],
                components=self.last_results['scaled_components'],
                comp_names=comp_names,
                peaks_curves=self.last_results.get('peaks_curves', []),
                excess_residual=self.last_results['excess_residual'],
                fit_residual=scaled_target.intensities - self.last_results['fitted_curve']
            )

    def _on_worker_error(self, err: str):
        self.btn_run_decomp.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(tr("decomp.status_error", "Status: Błąd obliczeń!"))
        self.lbl_status.setStyleSheet("color: #e76f51; font-weight: bold;")
        QMessageBox.critical(self, tr("app.error"), f"Błąd procedury Monte Carlo:\n{err}")

    def _on_add_residual_to_session(self):
        if not self.last_results:
            return
        target_name = self.combo_target.currentText()
        target_spec = session.spectra[target_name]

        res_spec = SpectrumData(
            wavenumbers=target_spec.wavenumbers.copy(),
            intensities=self.last_results['excess_residual'].copy(),
            name=f"{target_name}_residuum",
            file_path=target_spec.file_path,
            file_type="residual",
            units_x=target_spec.units_x,
            units_y=target_spec.units_y,
            metadata={
                "source": "HDO Monte Carlo Decomposition",
                "parent_target": target_name,
                "r2": self.last_results['r2'],
                "rmse": self.last_results['rmse']
            }
        )
        session.add_spectrum(res_spec, make_active=True)
        QMessageBox.information(self, tr("app.info"),
                                tr("decomp.residual_added", "Residuum zostało pomyślnie dodane do drzewa projektu."))

    def _on_add_fit_to_session(self):
        if not self.last_results:
            return
        target_name = self.combo_target.currentText()
        target_spec = session.spectra[target_name]

        fit_spec = SpectrumData(
            wavenumbers=target_spec.wavenumbers.copy(),
            intensities=self.last_results['fitted_curve'].copy(),
            name=f"{target_name}_fit",
            file_path=target_spec.file_path,
            file_type="fit",
            units_x=target_spec.units_x,
            units_y=target_spec.units_y,
            metadata={"source": "HDO Total Fit"}
        )
        session.add_spectrum(fit_spec, make_active=True)
        QMessageBox.information(self, tr("app.info"),
                                tr("decomp.fit_added", "Krzywa dopasowania została dodana do drzewa projektu."))

    def retranslate_ui(self):
        self.group_target.setTitle(tr("decomp.group_target", "1. Widmo Badane (W₀)"))
        self.lbl_target.setText(tr("decomp.target", "Widmo:"))
        self.lbl_n.setText(tr("decomp.n_multiplier", "Mnożnik N:"))
        self.group_bases.setTitle(tr("decomp.group_bases", "2. Widma Składowe (W₁, W₂, ..., Wₖ)"))
        self.table_bases.setHorizontalHeaderLabels([
            tr("decomp.table.spectrum", "Widmo z sesji"),
            tr("decomp.table.min_w", "Waga Min"),
            tr("decomp.table.max_w", "Waga Max"),
            "✕"
        ])
        self.btn_add_base.setText(tr("decomp.add_base", "➕ Dodaj widmo bazowe"))
        self.group_synth.setTitle(tr("decomp.group_synth", "3. Piki Syntetyczne & Linia Bazowa"))
        self.lbl_synth_peaks.setText(tr("decomp.synth_peaks", "Liczba pików G×L:"))
        self.chk_baseline.setText(tr("decomp.baseline", "Linia bazowa (kwadratowa)"))
        self.group_mc.setTitle(tr("decomp.group_mc", "4. Parametry Monte Carlo"))
        self.lbl_iter.setText(tr("decomp.iterations", "L. wariantów (MC):"))
        self.lbl_penalty.setText(tr("decomp.penalty", "Wsp. kary:"))
        self.btn_run_decomp.setText(tr("decomp.run_btn", "⚡ Uruchom Dekompozycję (Monte Carlo)"))
        self.group_results.setTitle(tr("decomp.group_results", "5. Wyniki & Eksport"))
        self.table_results.setHorizontalHeaderLabels([
            tr("decomp.res_table.comp", "Składnik"),
            tr("decomp.res_table.weight", "Waga (c)"),
            tr("decomp.res_table.pct", "Udział [%]")
        ])
        self.btn_add_residual_to_session.setText(tr("decomp.add_residual", "📥 Dodaj Czyste Residuum do Projektu"))
        self.btn_add_fit_to_session.setText(tr("decomp.add_fit", "➕ Dodaj Krzywą Dopasowaną do Projektu"))


# =====================================================================
# GŁÓWNY WIDOK PERSPEKTYWY ROBOCZEJ
# =====================================================================

class DeconvolutionWorkspace(QWidget):
    """Główny kontener doku dla perspektywy Dekonwolucji z dwiema podzakładkami."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.panel_deconv = AmideDeconvPanel()
        self.panel_decomp = GeneralizedDecompositionPanel()

        self.tabs.addTab(self.panel_deconv, "📈 Dekonwolucja Pasma")
        self.tabs.addTab(self.panel_decomp, "🧪 Dekompozycja HDO / Widm")
        layout.addWidget(self.tabs)

        i18n.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.tabs.setTabText(0, tr("deconv.tab_deconv", "📈 Dekonwolucja Pasma"))
        self.tabs.setTabText(1, tr("deconv.tab_decomp", "🧪 Dekompozycja HDO / Widm"))


DeconvolutionPanel = DeconvolutionWorkspace