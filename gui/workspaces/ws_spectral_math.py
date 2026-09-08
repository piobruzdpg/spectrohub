"""
gui/workspaces/ws_spectral_math.py
Perspektywa Matematyki Spektralnej (Spectral Math Workspace).
Zawiera parametryczny korektor suwakowy dla współczynnika k,
operacje algebraiczne na widmach oraz przeliczanie fotometryczne.
Pełna izolacja kontekstowa (Okna robocze vs Magazyn) i wsparcie wsadowe.
"""

import copy
import logging
import numpy as np
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QRadioButton,
    QButtonGroup, QScrollArea, QFrame, QSlider, QMessageBox,
    QTabWidget, QLineEdit, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QUndoCommand

from core.session import session
from core.models import SpectrumData
from core.i18n import tr, i18n
from algorithms import spectral_math as smath
from gui.workspaces.ws_preprocessing import AddBatchSpectraCommand, BatchPipelineCommand

logger = logging.getLogger("SpectroHub")


# =============================================================================
# WIDGET EQUALIZERA PARAMETRYCZNEGO DLA WSPÓŁCZYNNIKA K
# =============================================================================

class ParametricKWidget(QGroupBox):
    """Korektor suwakowy rozbijający współczynnik k na sumę rzędów wielkości."""
    k_changed = Signal(float)

    BANDS = [
        ("±10", 1.0, 10),        # Krok 1.0, zakres [-10, 10]
        ("±1.0", 0.1, 10),       # Krok 0.1, zakres [-1.0, 1.0]
        ("±0.1", 0.01, 10),      # Krok 0.01, zakres [-0.1, 0.1]
        ("±0.01", 0.001, 10),    # Krok 0.001, zakres [-0.01, 0.01]
        ("±0.001", 0.0001, 10),  # Krok 0.0001, zakres [-0.001, 0.001]
        ("±0.0001", 0.00001, 10) # Krok 0.00001, zakres [-0.0001, 0.0001]
    ]

    def __init__(self, parent=None):
        super().__init__("🎚️ Parametryczny Dobór Współczynnika (k)", parent)
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #48cae4; border: 1px solid #383848; border-radius: 4px; margin-top: 6px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        self._is_updating = False
        self._sliders: List[QSlider] = []
        self._val_labels: List[QLabel] = []

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 8, 6, 6)
        main_layout.setSpacing(6)

        row_master = QHBoxLayout()
        row_master.addWidget(QLabel("Współczynnik całkowity (k):"))
        self.spin_total_k = QDoubleSpinBox()
        self.spin_total_k.setRange(-99.0, 99.0)
        self.spin_total_k.setDecimals(5)
        self.spin_total_k.setSingleStep(0.01)
        self.spin_total_k.setValue(0.0)
        self.spin_total_k.valueChanged.connect(self._on_spin_k_changed)
        row_master.addWidget(self.spin_total_k, 1)

        btn_reset_all = QPushButton("Zeruj k")
        btn_reset_all.clicked.connect(self.reset_all)
        row_master.addWidget(btn_reset_all)
        main_layout.addLayout(row_master)

        sliders_frame = QFrame()
        sliders_frame.setStyleSheet("background-color: #1e1e24; border-radius: 4px; padding: 4px;")
        l_sliders = QHBoxLayout(sliders_frame)
        l_sliders.setSpacing(4)
        l_sliders.setContentsMargins(4, 4, 4, 4)

        for label_text, step_mult, max_steps in self.BANDS:
            col_box = QVBoxLayout()
            col_box.setAlignment(Qt.AlignCenter)
            col_box.setSpacing(4)

            lbl_val = QLabel("0.0")
            lbl_val.setStyleSheet("font-size: 10px; color: #a0a0a0;")
            lbl_val.setAlignment(Qt.AlignCenter)
            self._val_labels.append(lbl_val)
            col_box.addWidget(lbl_val)

            slider = QSlider(Qt.Vertical)
            slider.setRange(-max_steps, max_steps)
            slider.setValue(0)
            slider.setFixedHeight(95)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(max_steps // 2)
            slider.valueChanged.connect(self._on_slider_moved)
            self._sliders.append(slider)
            col_box.addWidget(slider, alignment=Qt.AlignCenter)

            btn_col_zero = QPushButton("0")
            btn_col_zero.setFixedSize(22, 18)
            btn_col_zero.setStyleSheet("font-size: 9px; padding: 0;")
            btn_col_zero.clicked.connect(lambda _, s=slider: s.setValue(0))
            col_box.addWidget(btn_col_zero, alignment=Qt.AlignCenter)

            lbl_band = QLabel(label_text)
            lbl_band.setStyleSheet("font-size: 9px; font-weight: bold; color: #dcdcdc;")
            lbl_band.setAlignment(Qt.AlignCenter)
            col_box.addWidget(lbl_band)

            l_sliders.addLayout(col_box)

        main_layout.addWidget(sliders_frame)
        self.set_k(0.0)

    def _on_slider_moved(self):
        if self._is_updating:
            return
        total = 0.0
        for i, (name, step_mult, _) in enumerate(self.BANDS):
            c_val = self._sliders[i].value() * step_mult
            self._val_labels[i].setText(f"{c_val:+.4f}".rstrip('0').rstrip('.'))
            total += c_val

        self._is_updating = True
        self.spin_total_k.setValue(total)
        self._is_updating = False
        self.k_changed.emit(total)

    def _on_spin_k_changed(self, val: float):
        if self._is_updating:
            return
        self.set_k(val)
        self.k_changed.emit(val)

    def set_k(self, total: float):
        self._is_updating = True
        self.spin_total_k.setValue(total)

        rem = total
        for i, (name, step_mult, max_steps) in enumerate(self.BANDS):
            steps = int(round(rem / step_mult))
            steps = max(-max_steps, min(max_steps, steps))
            self._sliders[i].setValue(steps)
            contrib = steps * step_mult
            self._val_labels[i].setText(f"{contrib:+.4f}".rstrip('0').rstrip('.'))
            rem -= contrib

        self._is_updating = False

    def reset_all(self):
        self._is_updating = True
        for s in self._sliders:
            s.setValue(0)
        for lbl in self._val_labels:
            lbl.setText("0.0")
        self.spin_total_k.setValue(0.0)
        self._is_updating = False
        self.k_changed.emit(0.0)

    def get_k(self) -> float:
        return float(self.spin_total_k.value())


# =============================================================================
# KOMENDA HISTORII DLA ARYTMETYKI IN-PLACE
# =============================================================================

class ApplyMathSpectrumCommand(QUndoCommand):
    """Modyfikuje pojedyncze widmo w miejscu w wyniku operacji arytmetycznej."""
    def __init__(self, spec: SpectrumData, desc: str, old_y: np.ndarray, new_y: np.ndarray, canvas=None):
        super().__init__(desc)
        self.spec = spec
        self.old_y = old_y.copy()
        self.new_y = new_y.copy()
        self.canvas = canvas

    def undo(self):
        self.spec.intensities = self.old_y.copy()
        if hasattr(session, "spectrum_data_changed"):
            session.spectrum_data_changed.emit(self.spec)
        if self.canvas:
            self.canvas.redraw_all()

    def redo(self):
        self.spec.intensities = self.new_y.copy()
        if hasattr(session, "spectrum_data_changed"):
            session.spectrum_data_changed.emit(self.spec)
        if self.canvas:
            self.canvas.redraw_all()


# =============================================================================
# GŁÓWNY PANEL PERSPEKTYWY MATEMATYKI SPEKTRALNEJ
# =============================================================================

class SpectralMathPanel(QWidget):
    """Inspektor roboczy dla perspektywy Matematyka Spektralna."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._preview_line = None
        self._last_result_y: Optional[np.ndarray] = None
        self._last_result_x: Optional[np.ndarray] = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. ŹRÓDŁA WIDM (S1 i S2)
        box_src = QGroupBox("1. Wybór Danych Wejściowych")
        l_src = QVBoxLayout(box_src)

        row_s1 = QHBoxLayout()
        self.lbl_s1 = QLabel("Widmo badane (S₁):")
        self.combo_s1 = QComboBox()
        self.combo_s1.currentIndexChanged.connect(self._on_selection_changed)
        row_s1.addWidget(self.lbl_s1)
        row_s1.addWidget(self.combo_s1, 1)
        l_src.addLayout(row_s1)

        row_s2 = QHBoxLayout()
        self.lbl_s2 = QLabel("Widmo odniesienia (S₂):")
        self.combo_s2 = QComboBox()
        self.combo_s2.currentIndexChanged.connect(self._on_selection_changed)
        row_s2.addWidget(self.lbl_s2)
        row_s2.addWidget(self.combo_s2, 1)
        l_src.addLayout(row_s2)

        # Tryb wsadowy i informacja o kontekście
        self.chk_batch = QCheckBox("Tryb wsadowy (Batch): zastosuj operację do serii")
        self.chk_batch.setStyleSheet("color: #48cae4; font-weight: bold;")
        self.chk_batch.toggled.connect(self._on_batch_toggled)
        l_src.addWidget(self.chk_batch)

        self.lbl_batch_scope = QLabel("Zakres: —")
        self.lbl_batch_scope.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        l_src.addWidget(self.lbl_batch_scope)

        layout.addWidget(box_src)

        # 2. EQUALIZER SUWAKOWY DLA K
        self.equalizer = ParametricKWidget()
        self.equalizer.k_changed.connect(self._on_k_changed)
        layout.addWidget(self.equalizer)

        # 3. TYP OPERACJI
        self.tabs_ops = QTabWidget()

        # Tab 1: Operacje dwuwidmowe
        tab_spec = QWidget()
        l_tab_spec = QVBoxLayout(tab_spec)
        self.bg_ops = QButtonGroup(self)

        self.rb_sub = QRadioButton("Różnica: S₁ − k·S₂")
        self.rb_sub.setChecked(True)
        self.rb_add = QRadioButton("Suma: S₁ + k·S₂")
        self.rb_mul = QRadioButton("Mnożenie: S₁ × S₂")
        self.rb_div = QRadioButton("Dzielenie: S₁ / S₂")

        for idx, rb in enumerate([self.rb_sub, self.rb_add, self.rb_mul, self.rb_div]):
            self.bg_ops.addButton(rb, idx)
            l_tab_spec.addWidget(rb)
            rb.toggled.connect(self._on_operation_changed)

        self.tabs_ops.addTab(tab_spec, "Widmo ↔ Widmo")

        # Tab 2: Operacje ze stałą (Skalar)
        tab_scalar = QWidget()
        l_tab_scalar = QVBoxLayout(tab_scalar)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Wartość stałej (c):"))
        self.spin_c = QDoubleSpinBox()
        self.spin_c.setRange(-1000.0, 1000.0)
        self.spin_c.setValue(1.0)
        self.spin_c.valueChanged.connect(self._recalculate)
        row_c.addWidget(self.spin_c)
        l_tab_scalar.addLayout(row_c)

        self.bg_scalar = QButtonGroup(self)
        self.rb_sc_add = QRadioButton("Dodaj: S + c")
        self.rb_sc_sub = QRadioButton("Odejmij: S − c")
        self.rb_sc_mul = QRadioButton("Pomnóż: S × c")
        self.rb_sc_div = QRadioButton("Podziel: S / c")
        self.rb_sc_add.setChecked(True)

        for idx, rb in enumerate([self.rb_sc_add, self.rb_sc_sub, self.rb_sc_mul, self.rb_sc_div]):
            self.bg_scalar.addButton(rb, idx)
            l_tab_scalar.addWidget(rb)
            rb.toggled.connect(self._recalculate)

        self.tabs_ops.addTab(tab_scalar, "Skalar (c)")

        # Tab 3: Przeliczenia fizyczne i fotometryczne
        tab_trans = QWidget()
        l_tab_trans = QVBoxLayout(tab_trans)

        btn_a_to_t = QPushButton("Absorbancja ➔ Transmitancja [0–1]")
        btn_a_to_t.clicked.connect(lambda: self._apply_photometric('a_to_t'))
        btn_a_to_pct = QPushButton("Absorbancja ➔ %T [0–100%]")
        btn_a_to_pct.clicked.connect(lambda: self._apply_photometric('a_to_pct'))
        btn_pct_to_a = QPushButton("%T ➔ Absorbancja")
        btn_pct_to_a.clicked.connect(lambda: self._apply_photometric('pct_to_a'))
        btn_km = QPushButton("Kubelka-Munk: f(R∞)")
        btn_km.clicked.connect(lambda: self._apply_photometric('km'))

        l_tab_trans.addWidget(btn_a_to_t)
        l_tab_trans.addWidget(btn_a_to_pct)
        l_tab_trans.addWidget(btn_pct_to_a)
        l_tab_trans.addWidget(btn_km)
        self.tabs_ops.addTab(tab_trans, "Fotometria")

        self.tabs_ops.currentChanged.connect(self._on_operation_changed)
        layout.addWidget(self.tabs_ops)

        # 4. ZATWIERDZANIE
        box_commit = QGroupBox("3. Zatwierdzenie")
        l_commit = QVBoxLayout(box_commit)

        self.btn_commit_new = QPushButton("➕ Dodaj wynik jako nowe widmo")
        self.btn_commit_new.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 7px;")
        self.btn_commit_new.clicked.connect(self._on_commit_new)
        l_commit.addWidget(self.btn_commit_new)

        self.btn_commit_replace = QPushButton("💾 Zastąp aktywne widmo (In-place z Undo)")
        self.btn_commit_replace.clicked.connect(self._on_commit_replace)
        l_commit.addWidget(self.btn_commit_replace)

        layout.addWidget(box_commit)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        session.spectrum_added.connect(self._refresh_combos)
        session.spectra_batch_added.connect(self._refresh_combos)
        session.spectrum_removed.connect(self._refresh_combos)
        session.visibility_changed.connect(self._refresh_combos)
        session.window_spectrum_added.connect(self._refresh_combos)
        session.window_spectrum_removed.connect(self._refresh_combos)
        session.active_window_changed.connect(self._refresh_combos)
        session.context_changed.connect(self._refresh_combos)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_combos()
        self._recalculate()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._remove_preview_line()

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _refresh_combos(self):
        self.combo_s1.blockSignals(True)
        self.combo_s2.blockSignals(True)

        prev_s1 = self.combo_s1.currentData()
        prev_s2 = self.combo_s2.currentData()

        self.combo_s1.clear()
        self.combo_s2.clear()

        # Domyślnie puste pozycje placeholder
        self.combo_s1.addItem("-- Wybierz widmo badane (S₁) --", None)
        self.combo_s2.addItem("-- Brak / Wybierz odniesienie (S₂) --", None)

        spectra = session.get_display_spectra()
        for s in spectra:
            self.combo_s1.addItem(f"📈 {s.name}", s.name)
            self.combo_s2.addItem(f"📈 {s.name}", s.name)

        if prev_s1 is not None:
            idx1 = self.combo_s1.findData(prev_s1)
            self.combo_s1.setCurrentIndex(idx1 if idx1 >= 0 else 0)
        else:
            self.combo_s1.setCurrentIndex(0)

        if prev_s2 is not None:
            idx2 = self.combo_s2.findData(prev_s2)
            self.combo_s2.setCurrentIndex(idx2 if idx2 >= 0 else 0)
        else:
            self.combo_s2.setCurrentIndex(0)

        self.combo_s1.blockSignals(False)
        self.combo_s2.blockSignals(False)

        # Precyzyjna informacja o kontekście operacji
        is_window = (session.active_context == "window" and session.active_window is not None)
        if is_window:
            win_name = session.active_window.name
            cnt = len([s for s in session.active_window.spectra.values() if session.active_window.visibility.get(s.name, True)])
            self.lbl_batch_scope.setText(f"🎯 Zakres: Okno robocze '{win_name}' ({cnt} widocznych widm)")
        else:
            cnt = len(session.get_visible_spectra())
            self.lbl_batch_scope.setText(f"📦 Zakres: Magazyn Danych ({cnt} widocznych pomiarów)")

    def _on_batch_toggled(self, checked: bool):
        if checked:
            self.lbl_s1.setText("Widmo do podglądu (opcjonalnie):")
            self.lbl_s2.setText("Widmo odniesienia (S₂) [do serii]:")
            self.btn_commit_new.setText("➕ Zastosuj i dodaj jako nowe widma (Batch)")
            self.btn_commit_replace.setText("💾 Zastosuj i zastąp w miejscu (Batch z Undo)")
        else:
            self.lbl_s1.setText("Widmo badane (S₁):")
            self.lbl_s2.setText("Widmo odniesienia (S₂):")
            self.btn_commit_new.setText("➕ Dodaj wynik jako nowe widmo")
            self.btn_commit_replace.setText("💾 Zastąp aktywne widmo (In-place z Undo)")

        self._recalculate()

    def _on_selection_changed(self):
        self._recalculate()

    def _on_operation_changed(self):
        self.equalizer.setVisible(self.tabs_ops.currentIndex() == 0 and self.bg_ops.checkedId() in [0, 1])
        self._recalculate()

    def _on_k_changed(self, k: float):
        self._recalculate()

    # =========================================================================
    # SZYBKI PODGLĄD NA ŻYWO (60 FPS, TYLKO GDY PERSPEKTYWA JEST WIDOCZNA)
    # =========================================================================

    def _recalculate(self):
        if not self.isVisible():
            self._remove_preview_line()
            return

        canvas = self._get_canvas()
        if not canvas:
            return

        s1_name = self.combo_s1.currentData()
        s2_name = self.combo_s2.currentData()
        tab_idx = self.tabs_ops.currentIndex()

        # Dopóki brak wyboru S1 lub S2 (dla operacji dwuwidmowych), usuwamy linię podglądu
        if not s1_name or (tab_idx == 0 and not s2_name):
            self._remove_preview_line()
            return

        spectra_dict = {s.name: s for s in session.get_display_spectra()}
        s1 = spectra_dict.get(s1_name)
        s2 = spectra_dict.get(s2_name) if s2_name else None

        if not s1:
            self._remove_preview_line()
            return

        x1, y1 = s1.wavenumbers, s1.intensities
        k = self.equalizer.get_k()

        try:
            if tab_idx == 0 and s2:
                y2_aligned = smath.align_to_reference(s2.wavenumbers, s2.intensities, x1)
                op_id = self.bg_ops.checkedId()
                if op_id == 0:
                    res_y = smath.subtract_spectra(y1, y2_aligned, k)
                elif op_id == 1:
                    res_y = smath.add_spectra(y1, y2_aligned, k)
                elif op_id == 2:
                    res_y = smath.multiply_spectra(y1, y2_aligned)
                elif op_id == 3:
                    res_y = smath.divide_spectra(y1, y2_aligned)
                else:
                    res_y = y1

            elif tab_idx == 1:
                c = self.spin_c.value()
                op_map = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div'}
                res_y = smath.scalar_operation(y1, c, op_map.get(self.bg_scalar.checkedId(), 'add'))
            else:
                res_y = y1

            self._last_result_x = x1
            self._last_result_y = res_y
            self._update_preview_line(canvas, x1, res_y)

        except Exception as e:
            logger.debug(f"Błąd podglądu matematyki: {e}")
            self._remove_preview_line()

    def _update_preview_line(self, canvas, x: np.ndarray, y: np.ndarray):
        if self._preview_line is None:
            self._preview_line, = canvas.ax_main.plot(
                x, y, color="#ffb703", lw=1.8, ls="-", zorder=7, label="Wynik (Podgląd)"
            )
            canvas.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a', labelcolor='#ffffff', fontsize=8)
        else:
            self._preview_line.set_xdata(x)
            self._preview_line.set_ydata(y)

        canvas.canvas.draw_idle()

    def _remove_preview_line(self):
        if self._preview_line is not None:
            try:
                self._preview_line.remove()
            except Exception:
                pass
            self._preview_line = None
            canvas = self._get_canvas()
            if canvas:
                canvas.canvas.draw_idle()

    # =========================================================================
    # ZATWIERDZANIE WYNIKÓW (POJEDYNCZE ORAZ HURTOWE Z UNDO / REDO)
    # =========================================================================

    def _on_commit_new(self):
        canvas = self._get_canvas()
        is_batch = self.chk_batch.isChecked()
        target_win = session.active_window_id if session.active_context == "window" else None
        visible_spectra = session.get_display_spectra()

        # 1. TRYB POJEDYNCZY
        if not is_batch:
            s1_name = self.combo_s1.currentData()
            if not s1_name or self._last_result_y is None or self._last_result_x is None:
                QMessageBox.warning(self, tr("app.warning"), "Wybierz widma wejściowe przed zatwierdzeniem.")
                return

            new_name = f"{s1_name}_math"
            new_spec = SpectrumData(
                wavenumbers=self._last_result_x.copy(),
                intensities=self._last_result_y.copy(),
                name=new_name,
                file_path="",
                file_type="math_result",
                metadata={"source_s1": s1_name, "k_factor": self.equalizer.get_k()}
            )
            self._remove_preview_line()
            cmd = AddBatchSpectraCommand([new_spec], f"Matematyka: {new_name}", canvas=canvas, window_id=target_win)
            session.undo_stack.push(cmd)
            QMessageBox.information(self, tr("app.info"), f"Pomyślnie dodano widmo '{new_name}' do projektu.")

        # 2. TRYB WSADOWY (BATCH)
        else:
            if not visible_spectra:
                QMessageBox.warning(self, tr("app.warning"), "Brak widocznych widm w aktywnym widoku.")
                return

            tab_idx = self.tabs_ops.currentIndex()

            if tab_idx == 0:  # Widmo ↔ Widmo
                s2_name = self.combo_s2.currentData()
                if not s2_name:
                    QMessageBox.warning(self, tr("app.warning"), "Wybierz widmo odniesienia (S₂) do operacji wsadowej.")
                    return

                s2 = next((s for s in visible_spectra if s.name == s2_name), None) or session.spectra.get(s2_name)
                if not s2:
                    QMessageBox.warning(self, tr("app.warning"), f"Nie odnaleziono widma odniesienia '{s2_name}'.")
                    return

                k = self.equalizer.get_k()
                op_id = self.bg_ops.checkedId()
                op_tags = {0: "sub", 1: "add", 2: "mul", 3: "div"}
                op_tag = op_tags.get(op_id, "math")

                new_list = []
                for s in visible_spectra:
                    # Wykonujemy operację na KAŻDYM widmie (włącznie z odniesieniem, dając np. linię 0 dla k=1)
                    y2_aligned = smath.align_to_reference(s2.wavenumbers, s2.intensities, s.wavenumbers)
                    if op_id == 0:
                        y_out = smath.subtract_spectra(s.intensities, y2_aligned, k)
                    elif op_id == 1:
                        y_out = smath.add_spectra(s.intensities, y2_aligned, k)
                    elif op_id == 2:
                        y_out = smath.multiply_spectra(s.intensities, y2_aligned)
                    elif op_id == 3:
                        y_out = smath.divide_spectra(s.intensities, y2_aligned)
                    else:
                        y_out = s.intensities.copy()

                    new_list.append(SpectrumData(
                        wavenumbers=s.wavenumbers.copy(),
                        intensities=y_out,
                        name=f"{op_tag}_{s.name}",
                        file_path=s.file_path,
                        file_type="math_batch",
                        metadata={"math_source": s.name, "math_ref": s2_name, "k": k, "op": op_tag}
                    ))

                if not new_list:
                    QMessageBox.information(self, tr("app.info"), "Brak widm do przetworzenia.")
                    return

                self._remove_preview_line()
                cmd = AddBatchSpectraCommand(new_list, f"Matematyka wsadowa ({len(new_list)} widm)", canvas=canvas,
                                             window_id=target_win)
                session.undo_stack.push(cmd)
                QMessageBox.information(self, tr("app.info"), f"Pomyślnie utworzono {len(new_list)} nowych widm.")

            elif tab_idx == 1:  # Skalar
                c = self.spin_c.value()
                op_id = self.bg_scalar.checkedId()
                op_map = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div'}
                op_str = op_map.get(op_id, 'add')

                new_list = []
                for s in visible_spectra:
                    y_out = smath.scalar_operation(s.intensities, c, op_str)
                    new_list.append(SpectrumData(
                        wavenumbers=s.wavenumbers.copy(),
                        intensities=y_out,
                        name=f"{op_str}_{s.name}",
                        file_path=s.file_path,
                        file_type="math_scalar",
                        metadata={"math_source": s.name, "scalar_c": c, "op": op_str}
                    ))

                self._remove_preview_line()
                cmd = AddBatchSpectraCommand(new_list, f"Skalar wsadowo ({len(new_list)} widm)", canvas=canvas,
                                             window_id=target_win)
                session.undo_stack.push(cmd)
                QMessageBox.information(self, tr("app.info"), f"Pomyślnie utworzono {len(new_list)} nowych widm.")

    def _on_commit_replace(self):
        canvas = self._get_canvas()
        is_batch = self.chk_batch.isChecked()
        visible_spectra = session.get_display_spectra()

        # 1. POJEDYNCZE ZASTĄPIENIE
        if not is_batch:
            s1_name = self.combo_s1.currentData()
            s1 = next((s for s in visible_spectra if s.name == s1_name), None)
            if not s1 or self._last_result_y is None:
                QMessageBox.warning(self, tr("app.warning"), "Brak obliczonego wyniku dla widma S₁.")
                return

            self._remove_preview_line()
            cmd = ApplyMathSpectrumCommand(s1, f"Matematyka: {s1.name}", s1.intensities, self._last_result_y,
                                           canvas=canvas)
            session.undo_stack.push(cmd)
            QMessageBox.information(self, tr("app.info"), f"Zastąpiono widmo '{s1.name}' nowymi danymi.")

        # 2. ZASTĄPIENIE WSADOWE (IN-PLACE Z UNDO)
        else:
            if not visible_spectra:
                QMessageBox.warning(self, tr("app.warning"), "Brak widocznych widm w aktywnym widoku.")
                return

            tab_idx = self.tabs_ops.currentIndex()
            batch_records = []

            if tab_idx == 0:  # Widmo ↔ Widmo
                s2_name = self.combo_s2.currentData()
                if not s2_name:
                    QMessageBox.warning(self, tr("app.warning"), "Wybierz widmo odniesienia (S₂) do operacji wsadowej.")
                    return

                s2 = next((s for s in visible_spectra if s.name == s2_name), None) or session.spectra.get(s2_name)
                if not s2:
                    return

                k = self.equalizer.get_k()
                op_id = self.bg_ops.checkedId()

                for s in visible_spectra:
                    # Przetwarzamy każde widmo w serii, włącznie z t0
                    y2_aligned = smath.align_to_reference(s2.wavenumbers, s2.intensities, s.wavenumbers)
                    if op_id == 0:
                        y_out = smath.subtract_spectra(s.intensities, y2_aligned, k)
                    elif op_id == 1:
                        y_out = smath.add_spectra(s.intensities, y2_aligned, k)
                    elif op_id == 2:
                        y_out = smath.multiply_spectra(s.intensities, y2_aligned)
                    elif op_id == 3:
                        y_out = smath.divide_spectra(s.intensities, y2_aligned)
                    else:
                        y_out = s.intensities.copy()

                    batch_records.append((s, s.wavenumbers.copy(), s.intensities.copy(),
                                          s.wavenumbers.copy(), y_out,
                                          copy.deepcopy(s.metadata), copy.deepcopy(s.metadata)))

            elif tab_idx == 1:  # Skalar
                c = self.spin_c.value()
                op_id = self.bg_scalar.checkedId()
                op_map = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div'}
                op_str = op_map.get(op_id, 'add')

                for s in visible_spectra:
                    y_out = smath.scalar_operation(s.intensities, c, op_str)
                    batch_records.append((s, s.wavenumbers.copy(), s.intensities.copy(),
                                          s.wavenumbers.copy(), y_out,
                                          copy.deepcopy(s.metadata), copy.deepcopy(s.metadata)))

            if not batch_records:
                return

            self._remove_preview_line()
            cmd = BatchPipelineCommand(batch_records, f"Matematyka wsadowa in-place ({len(batch_records)} widm)",
                                       canvas=canvas)
            session.undo_stack.push(cmd)
            QMessageBox.information(self, tr("app.info"), f"Zmodyfikowano {len(batch_records)} widm w miejscu.")

    def _apply_photometric(self, mode: str):
        canvas = self._get_canvas()
        is_batch = self.chk_batch.isChecked()
        visible_spectra = session.get_display_spectra()

        op_names = {'a_to_t': "A ➔ T", 'a_to_pct': "A ➔ %T", 'pct_to_a': "%T ➔ A", 'km': "Kubelka-Munk"}
        desc = op_names.get(mode, mode)

        def transform(y):
            if mode == 'a_to_t': return smath.absorbance_to_transmittance(y)
            elif mode == 'a_to_pct': return smath.absorbance_to_percent_t(y)
            elif mode == 'pct_to_a': return smath.percent_t_to_absorbance(y)
            elif mode == 'km': return smath.kubelka_munk(y)
            return y

        if not is_batch:
            s1_name = self.combo_s1.currentData()
            s1 = next((s for s in visible_spectra if s.name == s1_name), None)
            if not s1:
                QMessageBox.warning(self, tr("app.warning"), "Wybierz widmo badane (S₁) do przeliczenia fotometrycznego.")
                return

            new_y = transform(s1.intensities)
            self._remove_preview_line()
            cmd = ApplyMathSpectrumCommand(s1, f"Fotometria ({desc}): {s1.name}", s1.intensities, new_y, canvas=canvas)
            session.undo_stack.push(cmd)
            QMessageBox.information(self, tr("app.info"), f"Zastosowano transformację fotometryczną ({desc}) dla '{s1.name}'.")
        else:
            if not visible_spectra:
                QMessageBox.warning(self, tr("app.warning"), "Brak widocznych widm w aktywnym widoku.")
                return

            batch_records = []
            for s in visible_spectra:
                new_y = transform(s.intensities)
                batch_records.append((s, s.wavenumbers.copy(), s.intensities.copy(),
                                      s.wavenumbers.copy(), new_y,
                                      copy.deepcopy(s.metadata), copy.deepcopy(s.metadata)))

            self._remove_preview_line()
            cmd = BatchPipelineCommand(batch_records, f"Fotometria wsadowa ({desc}) - {len(batch_records)} widm", canvas=canvas)
            session.undo_stack.push(cmd)
            QMessageBox.information(self, tr("app.info"), f"Zastosowano transformację fotometryczną ({desc}) dla {len(batch_records)} widm.")