"""
gui/workspaces/ws_deconvolution.py
Panel dekonwolucji pasma Amidu I i fitowania krzywych (lmfit + QUndoStack + Live Canvas).
Wersja zoptymalizowana pod macOS Cocoa i stabilne sygnały Qt.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QMessageBox,
    QFrame, QCheckBox, QAbstractItemView
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QUndoCommand, QColor
import numpy as np
import copy
import logging
from scipy.signal import find_peaks

from core.session import session
from core.i18n import tr, i18n
from algorithms.fitting.deconv_engine import DeconvolutionEngine
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


class DeconvChangeCommand(QUndoCommand):
    def __init__(self, panel: 'DeconvolutionPanel', description: str,
                 old_peaks: list, new_peaks: list):
        super().__init__(description)
        self.panel = panel
        self.old_peaks = copy.deepcopy(old_peaks)
        self.new_peaks = copy.deepcopy(new_peaks)

    def undo(self):
        self.panel.apply_peaks_state(self.old_peaks)

    def redo(self):
        self.panel.apply_peaks_state(self.new_peaks)


class DeconvolutionPanel(QWidget):
    """Widok inspektora dla perspektywy Dekonwolucja & Fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks = []
        self.peak_counter = 0
        self.show_deriv = False
        self._is_updating_table = False
        self._canvas_connected = False

        self._init_ui()
        self._connect_signals()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Konfiguracja środowiska
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

        # 2. Tabela pasm
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Pik", "Typ", "Pozycja", "FWHM",
            "Pole", "% Amid I", "Struktura", "Lock (C/W/A)", "✕"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table)

        # 3. Przyciski edycji i Undo/Redo
        row_btns = QHBoxLayout()
        self.btn_add_interactive = QPushButton("➕ Dodaj (Klik na wykresie)")
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

        # 4. Asystent 2. pochodnej
        row_deriv = QHBoxLayout()
        self.btn_toggle_deriv = QPushButton("📉 2. Pochodna")
        self.btn_toggle_deriv.setCheckable(True)
        self.btn_toggle_deriv.clicked.connect(self._on_toggle_deriv)

        self.btn_auto_peaks = QPushButton("🎯 Szukaj z 2. pochodnej")
        self.btn_auto_peaks.clicked.connect(self._on_auto_find_peaks)

        row_deriv.addWidget(self.btn_toggle_deriv)
        row_deriv.addWidget(self.btn_auto_peaks)
        layout.addLayout(row_deriv)

        # 5. Silnik dopasowujący
        self.btn_fit = QPushButton("⚡ Dopasuj Pasma")
        self.btn_fit.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 8px;")
        self.btn_fit.clicked.connect(self._on_run_fit)
        layout.addWidget(self.btn_fit)

        # 6. Panel metryk jakości (GoF)
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
        self._update_curves_and_canvas()

    def _on_active_spectrum_changed(self, spectrum):
        self._ensure_canvas_connected()
        self._update_curves_and_canvas()

    def _toggle_interactive_add(self, checked: bool):
        canvas = self._get_canvas()
        if canvas:
            canvas.interaction_mode = "add_peak" if checked else "normal"
            if checked:
                self.btn_add_interactive.setStyleSheet("background-color: #e76f51; font-weight: bold;")
                self.btn_add_interactive.setText("Kliknij na wykresie...")
            else:
                self.btn_add_interactive.setStyleSheet("")
                self.btn_add_interactive.setText("➕ Dodaj (Klik na wykresie)")

    def on_canvas_add_peak(self, x: float, y: float):
        active = session.active_spectrum
        if not active:
            return

        idx_closest = int((np.abs(active.wavenumbers - x)).argmin())
        y_val = float(active.intensities[idx_closest]) if y <= 0 else y

        self.peak_counter += 1
        model_type = self.combo_model.currentText()
        solvent = self.combo_solvent.currentText()
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
            val = float(item.text().replace(',', '.'))
            if col == 2:
                peak['center'] = val
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

    # --- Pancerne zarządzanie wierszami tabeli ---
    def _get_widget_row(self, widget: QWidget) -> int:
        """Ustala wiersz tabeli dla kontrolki osadzonej w komórce."""
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

        valid_area_amide = sum(
            p['area'] for p in self.peaks
            if 1600 <= p['center'] <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]
        )

        for row, p in enumerate(self.peaks):
            fwhm = p['sigma'] * (2.0 if p['type'] == 'Lorentzian' else 2.35482)
            if 1600 <= p['center'] <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]:
                perc_amide = (p['area'] / valid_area_amide * 100.0) if valid_area_amide > 0 else 0.0
                perc_str = f"{perc_amide:.1f} %"
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

            # Col 1: Typ (Kształt)
            combo_t = self.table.cellWidget(row, 1)
            if not combo_t:
                combo_t = QComboBox()
                combo_t.addItems(["Gaussian", "PseudoVoigt", "Voigt", "Lorentzian"])
                combo_t.currentTextChanged.connect(self._on_cell_type_changed)
                self.table.setCellWidget(row, 1, combo_t)

            combo_t.blockSignals(True)
            combo_t.setCurrentText(p['type'])
            combo_t.blockSignals(False)

            # Col 2: Pozycja
            self._set_or_update_item(row, 2, f"{p['center']:.1f}")

            # Col 3: FWHM
            self._set_or_update_item(row, 3, f"{fwhm:.1f}")

            # Col 4: Pole
            self._set_or_update_item(row, 4, f"{p['area']:.2f}")

            # Col 5: % Amid I
            item_pct = self.table.item(row, 5)
            if not item_pct:
                item_pct = QTableWidgetItem()
                item_pct.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(row, 5, item_pct)
            item_pct.setText(perc_str)

            # Col 6: Struktura
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

    # Sloty komórek (obsługiwane bez rozłączania)
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
        mask = (x >= 1600.0) & (x <= 1700.0)
        x_sub, d2y_sub = x[mask], d2y[mask]

        if len(d2y_sub) == 0:
            return

        prominence = 0.1 * (np.max(d2y_sub) - np.min(d2y_sub))
        peak_indices, _ = find_peaks(d2y_sub, distance=5, prominence=prominence)

        if len(peak_indices) == 0:
            QMessageBox.information(self, tr("app.info"), "Nie wykryto wyraźnych minimów w 2. pochodnej.")
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
            QMessageBox.warning(self, tr("app.warning"), "Wymagane jest aktywne widmo i co najmniej jedno pasmo.")
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
            updated_peaks, gof = DeconvolutionEngine.fit_peaks(
                x_fit, y_fit, self.peaks, method=method
            )

            cmd = DeconvChangeCommand(self, f"Dopasowanie ({method})", old_peaks, updated_peaks)
            session.undo_stack.push(cmd)

            self.lbl_r2.setText(f"R²: {gof['r2']:.5f}")
            self.lbl_rmse.setText(f"RMSE: {gof['rmse']:.6f}")
            self.lbl_chi.setText(f"χ²: {gof['chisqr']:.2e} | Red-χ²: {gof['redchi']:.2e}")

        except Exception as e:
            logger.error(f"Błąd dopasowania lmfit ({method}): {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd procedury dopasowania:\n{e}")

    def retranslate_ui(self):
        self.lbl_solvent.setText(tr("deconv.solvent", "Rozpuszczalnik:"))
        self.lbl_model.setText(tr("deconv.model", "Kształt:"))
        self.lbl_algo.setText(tr("deconv.algo", "Algorytm:"))
        self.btn_clear_peaks.setText(tr("deconv.clear_peaks", "🗑 Wyczyść"))
        self.btn_fit.setText(tr("deconv.fit_btn", "⚡ Dopasuj Pasma"))
        self.btn_undo.setText(tr("app.undo", "↶ Cofnij"))
        self.btn_redo.setText(tr("app.redo", "↷ Ponów"))