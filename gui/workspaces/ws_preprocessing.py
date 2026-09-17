"""
gui/workspaces/ws_preprocessing.py
Panel preprocessingu i przygotowania danych spektroskopowych (Workspace 1).
Zintegrowany z silnikiem Pipeline Managera (edycja parametrów, reordering, batch, in-place z Undo),
korekcją offsetu w wybranym punkcie oraz pełną obsługą historii QUndoStack.
Okna narzędziowe działają w trybie niemodalnym z izolacją celów potoku.
"""

import copy
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QMessageBox, QDialog, QComboBox,
    QListWidget, QListWidgetItem, QScrollArea, QFrame,
    QProgressBar, QApplication, QCheckBox, QFormLayout,
    QDialogButtonBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoCommand

from core.session import session
from core.models import SpectrumData, SpectralSeries
from core.i18n import tr, i18n
from algorithms import preprocessing as prep
from algorithms.atmospheric import AtmosphericCorrectionEngine

logger = logging.getLogger("SpectroHub")


# ==============================================================================
# DIALOG EDYCJI PARAMETRÓW KROKU POTOKU
# ==============================================================================
class StepParamDialog(QDialog):
    """Modalne okno pozwalające edytować parametry dowolnego kroku potoku."""

    def __init__(self, step: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.step = step
        self.setWindowTitle(f"Parametry kroku: {step.get('type', '')}")
        self.resize(340, 220)
        self.setModal(True)

        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.layout.addLayout(self.form)

        stype = step.get('type', '')

        if stype == 'crop':
            self.spin_min = QDoubleSpinBox()
            self.spin_min.setRange(0.0, 10000.0)
            self.spin_min.setValue(step.get('min_w', 1000.0))
            self.spin_max = QDoubleSpinBox()
            self.spin_max.setRange(0.0, 10000.0)
            self.spin_max.setValue(step.get('max_w', 1800.0))
            self.form.addRow("Liczba falowa Min [cm⁻¹]:", self.spin_min)
            self.form.addRow("Liczba falowa Max [cm⁻¹]:", self.spin_max)

        elif stype == 'asls':
            self.ent_lam = QLineEdit(f"{step.get('lam', 1e5):.0e}")
            self.spin_p = QDoubleSpinBox()
            self.spin_p.setRange(0.0001, 0.5)
            self.spin_p.setDecimals(4)
            self.spin_p.setSingleStep(0.001)
            self.spin_p.setValue(step.get('p', 0.001))
            self.form.addRow("Gładkość (λ):", self.ent_lam)
            self.form.addRow("Asymetria (p):", self.spin_p)

        elif stype == 'airpls':
            self.ent_lam = QLineEdit(f"{step.get('lam', 1e5):.0e}")
            self.spin_iter = QSpinBox()
            self.spin_iter.setRange(5, 100)
            self.spin_iter.setValue(step.get('n_iter', 15))
            self.form.addRow("Gładkość (λ):", self.ent_lam)
            self.form.addRow("Maks. iteracji:", self.spin_iter)

        elif stype == 'poly_bl':
            self.spin_deg = QSpinBox()
            self.spin_deg.setRange(1, 10)
            self.spin_deg.setValue(step.get('deg', 3))
            self.form.addRow("Stopień wielomianu:", self.spin_deg)

        elif stype == 'fsd':
            self.spin_fwhm = QDoubleSpinBox()
            self.spin_fwhm.setRange(1.0, 100.0)
            self.spin_fwhm.setValue(step.get('fwhm', 18.0))
            self.spin_k = QDoubleSpinBox()
            self.spin_k.setRange(0.1, 10.0)
            self.spin_k.setValue(step.get('k_factor', 2.0))
            self.form.addRow("FWHM filtra [cm⁻¹]:", self.spin_fwhm)
            self.form.addRow("Współczynnik K (Kauppinen):", self.spin_k)

        elif stype in ['sg_smooth', 'deriv1', 'deriv2']:
            self.spin_win = QSpinBox()
            self.spin_win.setRange(3, 101)
            self.spin_win.setSingleStep(2)
            self.spin_win.setValue(step.get('win', 11))
            self.spin_poly = QSpinBox()
            self.spin_poly.setRange(1, 6)
            self.spin_poly.setValue(step.get('poly', 3))
            self.form.addRow("Szerokość okna (pkt):", self.spin_win)
            self.form.addRow("Stopień wielomianu:", self.spin_poly)

        elif stype == 'offset_point':
            self.spin_target_wn = QDoubleSpinBox()
            self.spin_target_wn.setRange(0.0, 10000.0)
            self.spin_target_wn.setValue(step.get('target_wn', 2000.0))
            self.form.addRow("Punkt zerowania ν [cm⁻¹]:", self.spin_target_wn)

        else:
            self.form.addRow(QLabel("Ten krok nie wymaga dodatkowych parametrów."))

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        self.layout.addWidget(btns)

    def _save_and_accept(self):
        stype = self.step.get('type', '')
        try:
            if stype == 'crop':
                self.step['min_w'] = float(self.spin_min.value())
                self.step['max_w'] = float(self.spin_max.value())
                self.step['name'] = f"Przycięcie [{self.step['min_w']:.0f}–{self.step['max_w']:.0f}]"

            elif stype == 'asls':
                self.step['lam'] = float(self.ent_lam.text())
                self.step['p'] = float(self.spin_p.value())
                self.step['name'] = f"AsLS (λ={self.step['lam']:.0e}, p={self.step['p']})"

            elif stype == 'airpls':
                self.step['lam'] = float(self.ent_lam.text())
                self.step['n_iter'] = int(self.spin_iter.value())
                self.step['name'] = f"airPLS / airALS (λ={self.step['lam']:.0e})"

            elif stype == 'poly_bl':
                self.step['deg'] = int(self.spin_deg.value())
                self.step['name'] = f"Wielomian st. {self.step['deg']}"

            elif stype == 'fsd':
                self.step['fwhm'] = float(self.spin_fwhm.value())
                self.step['k_factor'] = float(self.spin_k.value())
                self.step['name'] = f"FSD (FWHM={self.step['fwhm']}, K={self.step['k_factor']})"

            elif stype == 'sg_smooth':
                self.step['win'] = int(self.spin_win.value())
                self.step['poly'] = int(self.spin_poly.value())
                self.step['name'] = f"SG Wygładzanie (okno={self.step['win']}, st={self.step['poly']})"

            elif stype == 'deriv1':
                self.step['win'] = int(self.spin_win.value())
                self.step['poly'] = int(self.spin_poly.value())
                self.step['name'] = f"1. Pochodna SG (okno={self.step['win']})"

            elif stype == 'deriv2':
                self.step['win'] = int(self.spin_win.value())
                self.step['poly'] = int(self.spin_poly.value())
                self.step['name'] = f"2. Pochodna SG (-) (okno={self.step['win']})"

            elif stype == 'offset_point':
                self.step['target_wn'] = float(self.spin_target_wn.value())
                self.step['name'] = f"Offset w punkcie {self.step['target_wn']:.1f} cm⁻¹ ➔ 0"

            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Błąd", "Wprowadzono niepoprawny format liczby.")


# ==============================================================================
# KOMENDY HISTORII (QUndoStack)
# ==============================================================================
class ModifySpectrumCommand(QUndoCommand):
    """Cofa lub ponawia modyfikację pojedynczego widma in-place."""

    def __init__(self, spec: SpectrumData, description: str,
                 old_y: np.ndarray, new_y: np.ndarray,
                 old_x: Optional[np.ndarray] = None, new_x: Optional[np.ndarray] = None,
                 old_meta: Optional[dict] = None, new_meta: Optional[dict] = None,
                 canvas=None):
        super().__init__(description)
        self.spec = spec
        self.old_y = old_y.copy()
        self.new_y = new_y.copy()
        self.old_x = old_x.copy() if old_x is not None else spec.wavenumbers.copy()
        self.new_x = new_x.copy() if new_x is not None else spec.wavenumbers.copy()
        self.old_meta = copy.deepcopy(old_meta) if old_meta is not None else copy.deepcopy(spec.metadata)
        self.new_meta = copy.deepcopy(new_meta) if new_meta is not None else copy.deepcopy(spec.metadata)
        self.canvas = canvas

    def undo(self):
        self.spec.wavenumbers = self.old_x.copy()
        self.spec.intensities = self.old_y.copy()
        self.spec.metadata = copy.deepcopy(self.old_meta)
        if hasattr(session, "spectrum_data_changed"):
            session.spectrum_data_changed.emit(self.spec)
        if self.canvas:
            self.canvas.redraw_all()

    def redo(self):
        self.spec.wavenumbers = self.new_x.copy()
        self.spec.intensities = self.new_y.copy()
        self.spec.metadata = copy.deepcopy(self.new_meta)
        if hasattr(session, "spectrum_data_changed"):
            session.spectrum_data_changed.emit(self.spec)
        if self.canvas:
            self.canvas.redraw_all()


class BatchPipelineCommand(QUndoCommand):
    """Cofa lub ponawia wykonanie potoku wsadowego na wielu widmach jednocześnie."""

    def __init__(self, items: List[Tuple[SpectrumData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]],
                 description: str, canvas=None):
        super().__init__(description)
        self.items = items
        self.canvas = canvas

    def undo(self):
        modified_specs = []
        for spec, old_x, old_y, _, _, old_meta, _ in self.items:
            spec.wavenumbers = old_x.copy()
            spec.intensities = old_y.copy()
            spec.metadata = copy.deepcopy(old_meta)
            modified_specs.append(spec)

        if hasattr(session, "batch_spectra_changed"):
            session.batch_spectra_changed.emit(modified_specs)
        if self.canvas:
            self.canvas.redraw_all()

    def redo(self):
        modified_specs = []
        for spec, _, _, new_x, new_y, _, new_meta in self.items:
            spec.wavenumbers = new_x.copy()
            spec.intensities = new_y.copy()
            spec.metadata = copy.deepcopy(new_meta)
            modified_specs.append(spec)

        if hasattr(session, "batch_spectra_changed"):
            session.batch_spectra_changed.emit(modified_specs)
        if self.canvas:
            self.canvas.redraw_all()


class ModifySeriesCommand(QUndoCommand):
    """Cofa lub ponawia operacje in-place na serii 2D lub tensorze 3D."""

    def __init__(self, series_obj: SpectralSeries, description: str,
                 old_wn: np.ndarray, new_wn: np.ndarray,
                 old_data: np.ndarray, new_data: np.ndarray,
                 canvas=None):
        super().__init__(description)
        self.series_obj = series_obj
        self.old_wn = old_wn.copy()
        self.new_wn = new_wn.copy()
        self.old_data = old_data.copy()
        self.new_data = new_data.copy()
        self.canvas = canvas

    def undo(self):
        self.series_obj.wavenumbers = self.old_wn.copy()
        self.series_obj.data = self.old_data.copy()
        if hasattr(session, "series_data_changed"):
            session.series_data_changed.emit(self.series_obj)
        if self.canvas:
            self.canvas.redraw_all()

    def redo(self):
        self.series_obj.wavenumbers = self.new_wn.copy()
        self.series_obj.data = self.new_data.copy()
        if hasattr(session, "series_data_changed"):
            session.series_data_changed.emit(self.series_obj)
        if self.canvas:
            self.canvas.redraw_all()


class AddBatchSpectraCommand(QUndoCommand):
    """Cofa lub ponawia dodanie wygenerowanych widm do sesji lub okna roboczego."""

    def __init__(self, spectra_list: List[SpectrumData], description: str,
                 canvas=None, window_id: Optional[str] = None):
        super().__init__(description)
        self.spectra_list = spectra_list
        self.window_id = window_id
        self.registered_names: List[str] = []
        self.canvas = canvas

    def redo(self):
        if self.window_id and self.window_id in session.windows:
            self.registered_names = []
            for spec in self.spectra_list:
                added = session.add_spectrum_to_window(self.window_id, spec)
                self.registered_names.append(added.name)
        else:
            session.add_spectra_batch(self.spectra_list)
            self.registered_names = [s.name for s in self.spectra_list]

        if self.canvas:
            self.canvas.redraw_all()

    def undo(self):
        if self.window_id and self.window_id in session.windows:
            for name in self.registered_names:
                session.remove_spectrum_from_window(self.window_id, name)
        else:
            for name in self.registered_names:
                session.remove_spectrum(name)

        if self.canvas:
            self.canvas.redraw_all()


# ==============================================================================
# DIALOG HURTOWEJ KOMPENSACJI ATMOSFERYCZNEJ (BATCH VAPORFIT)
# ==============================================================================
class VaporFitDialog(QDialog):
    """Okno konfiguracji i uruchomienia kompensacji par wody i CO2 z bazy atmosfery."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("vapor.title", "VaporFit — Hurtowa Kompensacja Atmosferyczna"))
        self.resize(760, 620)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self._init_ui()
        self._load_available_spectra()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        row_lists = QHBoxLayout()

        # 1. Próbki badane
        box_samples = QGroupBox(tr("vapor.samples_group", "1. Widma badane do skorygowania"))
        l_samples = QVBoxLayout(box_samples)
        row_sample_tools = QHBoxLayout()
        btn_sel_all = QPushButton("Wszystkie")
        btn_sel_all.clicked.connect(lambda: self._set_all_checks(self.list_samples, Qt.Checked))
        btn_sel_active = QPushButton("Tylko aktywne")
        btn_sel_active.clicked.connect(self._select_only_active_sample)
        btn_sel_none = QPushButton("Odznacz")
        btn_sel_none.clicked.connect(lambda: self._set_all_checks(self.list_samples, Qt.Unchecked))

        row_sample_tools.addWidget(btn_sel_all)
        row_sample_tools.addWidget(btn_sel_active)
        row_sample_tools.addWidget(btn_sel_none)
        l_samples.addLayout(row_sample_tools)

        self.list_samples = QListWidget()
        l_samples.addWidget(self.list_samples)
        row_lists.addWidget(box_samples, stretch=1)

        # 2. Baza wzorców atmosfery
        box_atm = QGroupBox(tr("vapor.atm_group", "2. Wzorce atmosfery (H₂O / CO₂)"))
        l_atm = QVBoxLayout(box_atm)
        row_atm_tools = QHBoxLayout()
        btn_atm_all = QPushButton("Zaznacz wszystkie")
        btn_atm_all.clicked.connect(lambda: self._set_all_checks(self.list_atm, Qt.Checked))
        btn_atm_none = QPushButton("Odznacz")
        btn_atm_none.clicked.connect(lambda: self._set_all_checks(self.list_atm, Qt.Unchecked))
        row_atm_tools.addWidget(btn_atm_all)
        row_atm_tools.addWidget(btn_atm_none)
        row_atm_tools.addStretch()
        l_atm.addLayout(row_atm_tools)

        self.list_atm = QListWidget()
        l_atm.addWidget(self.list_atm)
        row_lists.addWidget(box_atm, stretch=1)
        layout.addLayout(row_lists)

        # 3. Parametry
        box_params = QGroupBox(tr("vapor.params_group", "3. Parametry algorytmu optymalizacyjnego"))
        l_params = QVBoxLayout(box_params)

        row_inputs = QHBoxLayout()
        row_inputs.addWidget(QLabel("Zakres [cm⁻¹]:"))
        self.ent_min_wn = QLineEdit("1000")
        self.ent_max_wn = QLineEdit("4000")
        row_inputs.addWidget(self.ent_min_wn)
        row_inputs.addWidget(QLabel("—"))
        row_inputs.addWidget(self.ent_max_wn)

        row_inputs.addWidget(QLabel("SG Okno:"))
        self.ent_sg_points = QLineEdit("11")
        self.ent_sg_points.setFixedWidth(40)
        row_inputs.addWidget(self.ent_sg_points)

        row_inputs.addWidget(QLabel("Wielomian:"))
        self.ent_sg_poly = QLineEdit("3")
        self.ent_sg_poly.setFixedWidth(30)
        row_inputs.addWidget(self.ent_sg_poly)

        self.btn_find_opt = QPushButton("🔍 Szukaj SG")
        self.btn_find_opt.clicked.connect(self._on_find_optimal_window)
        row_inputs.addWidget(self.btn_find_opt)
        l_params.addLayout(row_inputs)
        layout.addWidget(box_params)

        # Status
        self.frame_status = QFrame()
        self.frame_status.setStyleSheet("background-color: #252530; border-radius: 4px; padding: 6px;")
        l_status = QVBoxLayout(self.frame_status)
        l_status.setSpacing(4)

        self.lbl_status = QLabel("Wybierz widma i kliknij 'Uruchom korekcję hurtową'.")
        self.lbl_status.setStyleSheet("color: #dcdcdc; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)

        l_status.addWidget(self.lbl_status)
        l_status.addWidget(self.progress_bar)
        layout.addWidget(self.frame_status)

        # Przyciski
        row_actions = QHBoxLayout()
        self.btn_apply = QPushButton("⚡ Uruchom korekcję hurtową (Batch)")
        self.btn_apply.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 8px;")
        self.btn_apply.clicked.connect(self._on_apply_batch_correction)

        self.btn_cancel = QPushButton("Zamknij")
        self.btn_cancel.clicked.connect(self.close)

        row_actions.addWidget(self.btn_apply)
        row_actions.addWidget(self.btn_cancel)
        layout.addLayout(row_actions)

    def _set_all_checks(self, list_widget: QListWidget, state: Qt.CheckState):
        list_widget.blockSignals(True)
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(state)
        list_widget.blockSignals(False)

    def _select_only_active_sample(self):
        active = session.active_spectrum
        if not active:
            return
        self.list_samples.blockSignals(True)
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) == active.name else Qt.Unchecked)
        self.list_samples.blockSignals(False)

    def _load_available_spectra(self):
        active = session.active_spectrum
        samples = session.spectra
        atm_specs = session.atm_spectra

        self.list_samples.setUpdatesEnabled(False)
        self.list_atm.setUpdatesEnabled(False)
        self.list_samples.blockSignals(True)
        self.list_atm.blockSignals(True)

        self.list_samples.clear()
        self.list_atm.clear()

        for name, spec in samples.items():
            item_s = QListWidgetItem(f"📈 {name} ({spec.n_points} pkt)")
            item_s.setData(Qt.UserRole, name)
            item_s.setFlags(item_s.flags() | Qt.ItemIsUserCheckable)
            is_active = (active and name == active.name)
            item_s.setCheckState(Qt.Checked if is_active else Qt.Unchecked)
            self.list_samples.addItem(item_s)

        for name, spec in atm_specs.items():
            item_a = QListWidgetItem(f"☁️ {name} ({spec.n_points} pkt)")
            item_a.setData(Qt.UserRole, name)
            item_a.setFlags(item_a.flags() | Qt.ItemIsUserCheckable)
            item_a.setCheckState(Qt.Checked)
            self.list_atm.addItem(item_a)

        if not atm_specs:
            placeholder = QListWidgetItem("(Brak wzorców w gałęzi Atmosfery)")
            placeholder.setFlags(Qt.NoItemFlags)
            self.list_atm.addItem(placeholder)

        self.list_samples.blockSignals(False)
        self.list_atm.blockSignals(False)
        self.list_samples.setUpdatesEnabled(True)
        self.list_atm.setUpdatesEnabled(True)

        if active:
            self.ent_min_wn.setText(f"{np.min(active.wavenumbers):.1f}")
            self.ent_max_wn.setText(f"{np.max(active.wavenumbers):.1f}")
        elif samples:
            first_s = next(iter(samples.values()))
            self.ent_min_wn.setText(f"{np.min(first_s.wavenumbers):.1f}")
            self.ent_max_wn.setText(f"{np.max(first_s.wavenumbers):.1f}")

    def _get_checked_names(self, list_widget: QListWidget):
        selected = []
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.checkState() == Qt.Checked and item.data(Qt.UserRole):
                selected.append(item.data(Qt.UserRole))
        return selected

    def _on_find_optimal_window(self):
        sample_names = self._get_checked_names(self.list_samples)
        atm_names = self._get_checked_names(self.list_atm)

        if not sample_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Zaznacz co najmniej jedno widmo badane.")
            return
        if not atm_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Zaznacz co najmniej jedno widmo wzorca atmosfery.")
            return

        sample_spec = session.get_spectrum(sample_names[0])
        try:
            poly = int(self.ent_sg_poly.text())
            min_w = float(self.ent_min_wn.text())
            max_w = float(self.ent_max_wn.text())
        except ValueError:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nieprawidłowe parametry numeryczne.")
            return

        mask_s = (sample_spec.wavenumbers >= min_w) & (sample_spec.wavenumbers <= max_w)
        wn_sub = sample_spec.wavenumbers[mask_s]
        y_sub = sample_spec.intensities[mask_s]

        atm_matrix = []
        for a_name in atm_names:
            a_spec = session.get_spectrum(a_name)
            if a_spec.wavenumbers[0] > a_spec.wavenumbers[-1]:
                x_ref, y_ref = a_spec.wavenumbers[::-1], a_spec.intensities[::-1]
            else:
                x_ref, y_ref = a_spec.wavenumbers, a_spec.intensities
            y_interp = np.interp(wn_sub, x_ref, y_ref)
            atm_matrix.append(y_interp)

        atm_matrix = np.column_stack(atm_matrix)

        res = AtmosphericCorrectionEngine.find_optimal_sg_window(
            wn_sub, y_sub, atm_matrix, sg_poly=poly
        )

        opt_win = res['best_window_sdv']
        self.ent_sg_points.setText(str(opt_win))
        QMessageBox.information(
            self, tr("app.info", "Informacja"),
            f"Sugerowane optymalne okno SG (próbka: {sample_spec.name}):\n"
            f"• Wg krzywizny SDV: {res['best_window_sdv']} pkt\n"
            f"• Wg wskaźnika SSI: {res['best_window_ssi']} pkt\n\n"
            f"Wartość {opt_win} została wstawiona do konfiguracji."
        )

    def _on_apply_batch_correction(self):
        sample_names = self._get_checked_names(self.list_samples)
        atm_names = self._get_checked_names(self.list_atm)

        if not sample_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Zaznacz co najmniej jedno widmo badane.")
            return
        if not atm_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Zaznacz co najmniej jedno widmo wzorca atmosfery.")
            return

        try:
            poly = int(self.ent_sg_poly.text())
            points = int(self.ent_sg_points.text())
            min_w = float(self.ent_min_wn.text())
            max_w = float(self.ent_max_wn.text())
        except ValueError:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nieprawidłowe wartości parametrów numerycznych.")
            return

        total_count = len(sample_names)
        self.btn_apply.setEnabled(False)
        self.progress_bar.setMaximum(total_count)
        self.progress_bar.setValue(0)

        created_spectra = []
        errors = []

        prepared_atm = []
        for a_name in atm_names:
            a_spec = session.get_spectrum(a_name)
            if a_spec.wavenumbers[0] > a_spec.wavenumbers[-1]:
                prepared_atm.append((a_spec.wavenumbers[::-1], a_spec.intensities[::-1]))
            else:
                prepared_atm.append((a_spec.wavenumbers, a_spec.intensities))

        for idx, s_name in enumerate(sample_names):
            self.lbl_status.setText(f"Korygowanie ({idx + 1}/{total_count}): {s_name}...")
            QApplication.processEvents()

            try:
                sample_spec = session.get_spectrum(s_name)
                mask = (sample_spec.wavenumbers >= min_w) & (sample_spec.wavenumbers <= max_w)
                wn_crop = sample_spec.wavenumbers[mask]
                y_crop = sample_spec.intensities[mask]

                if len(wn_crop) < points + 2:
                    errors.append(f"{s_name}: Za mało punktów w wybranym zakresie.")
                    continue

                atm_cols = [np.interp(wn_crop, x_ref, y_ref) for x_ref, y_ref in prepared_atm]
                atm_matrix = np.column_stack(atm_cols)

                corr_y, coeffs, metrics = AtmosphericCorrectionEngine.fit_and_subtract(
                    wn_crop, y_crop, atm_matrix, sg_points=points, sg_poly=poly
                )

                new_name = f"corr_{sample_spec.name}"
                new_spec = SpectrumData(
                    wavenumbers=wn_crop,
                    intensities=corr_y,
                    name=new_name,
                    file_path=sample_spec.file_path,
                    file_type="vaporfit",
                    metadata={
                        "parent_spectrum": sample_spec.name,
                        "atm_references": atm_names,
                        "coeffs": coeffs.tolist(),
                        "metrics": metrics
                    }
                )
                created_spectra.append(new_spec)

            except Exception as e:
                errors.append(f"{s_name}: {str(e)}")

            self.progress_bar.setValue(idx + 1)

        if created_spectra:
            cmd = AddBatchSpectraCommand(created_spectra, f"VaporFit Batch ({len(created_spectra)} widm)")
            session.undo_stack.push(cmd)

        self.btn_apply.setEnabled(True)
        self.lbl_status.setText(f"Ukończono! Skorygowano pomyślnie {len(created_spectra)}/{total_count} widm.")

        msg = f"Pomyślnie utworzono i dodano {len(created_spectra)} skorygowanych widm do sesji."
        if errors:
            msg += f"\n\nBłędy w {len(errors)} plikach:\n" + "\n".join(errors[:5])

        QMessageBox.information(self, tr("app.info", "Informacja"), msg)


# ==============================================================================
# GŁÓWNY PANEL INSPEKTORA PREPROCESSINGU
# ==============================================================================
class PreprocessingPanel(QWidget):
    """Widok narzędzi obróbki i przygotowania danych z wbudowanym Pipeline Managerem."""

    def _verify_single_tool_permission(self) -> bool:
        if getattr(session, 'active_context', 'storage') == 'storage':
            QMessageBox.information(
                self, tr("app.info", "Informacja"),
                "Jesteś w Magazynie Danych (Referencji).\n\n"
                "Pojedyncze manipulacje są zablokowane dla magazynu, aby chronić oryginalne pomiary.\n"
                "• Przeciągnij widmo do Okna Wykresu, aby dowolnie je modyfikować.\n"
                "• W Magazynie Danych możesz wykonywać pełną obróbkę wsadową (Potok Przetwarzania, Batch VaporFit)."
            )
            return False
        return True

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline_steps: List[Dict[str, Any]] = []
        self._vaporfit_dialog: Optional[VaporFitDialog] = None

        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def get_pipeline_state(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.pipeline_steps)

    def set_pipeline_state(self, steps: List[Dict[str, Any]]):
        self.pipeline_steps = copy.deepcopy(steps) if steps else []
        self._refresh_pipeline_list()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(2, 2, 2, 2)

        # SEKCJA 1: PIPELINE MANAGER
        box_pipeline = QGroupBox("⚙️ Potok Przetwarzania (Pipeline Manager)")
        box_pipeline.setStyleSheet("QGroupBox { font-weight: bold; color: #48cae4; }")
        l_pipe = QVBoxLayout(box_pipeline)
        l_pipe.setSpacing(6)

        # Lista kroków potoku
        self.list_pipeline = QListWidget()
        self.list_pipeline.setFixedHeight(115)
        self.list_pipeline.setStyleSheet("background-color: #1e1e24; border: 1px solid #383848;")
        self.list_pipeline.itemDoubleClicked.connect(self._on_edit_pipeline_step)
        l_pipe.addWidget(self.list_pipeline)

        # Wiersz 1 narzędzi potoku: Typ kroku + Przycisk Dodaj
        row_add_step = QHBoxLayout()
        row_add_step.setSpacing(4)
        self.combo_step_type = QComboBox()
        self.combo_step_type.addItems([
            "✂ Przycięcie (Crop)",
            "📏 Linia Bazowa (AsLS)",
            "📏 Linia Bazowa (airPLS / airALS)",
            "📏 Linia Bazowa (Wielomianowa)",
            "📏 Linia Bazowa (Dwa krańce)",
            "🌊 Dekonwolucja Fouriera (FSD)",
            "〰 Wygładzanie (SG)",
            "📉 1. Pochodna (SG)",
            "📉 2. Pochodna (-) (SG)",
            "⚖ Normalizacja SNV",
            "⚖ Normalizacja Min-Max",
            "⚖ Normalizacja L₁ (Pole)",
            "⚖ Normalizacja L₂",
            "⚖ Offset (Min ➔ 0.0)",
            "⚖ Offset (w punkcie ν)",
            "📐 Centrowanie do średniej (Mean-center)",
            "📐 Autoskalowanie (Unit Variance)",
            "📐 Skalowanie Pareto",
            "📐 Korekcja rozproszenia (MSC)"
        ])
        btn_add_step = QPushButton("➕ Dodaj")
        btn_add_step.clicked.connect(self._on_add_pipeline_step)
        row_add_step.addWidget(self.combo_step_type, 1)
        row_add_step.addWidget(btn_add_step)
        l_pipe.addLayout(row_add_step)

        # Wiersz 2 narzędzi potoku: Edycja, kolejność, usuwanie, czyszczenie
        row_step_actions = QHBoxLayout()
        row_step_actions.setSpacing(4)

        btn_edit_step = QPushButton("✏️ Edytuj")
        btn_edit_step.setToolTip("Edytuj parametry zaznaczonego kroku")
        btn_edit_step.clicked.connect(self._on_edit_pipeline_step)

        btn_up = QPushButton("▲")
        btn_up.setFixedWidth(26)
        btn_up.setToolTip("Przesuń krok w górę")
        btn_up.clicked.connect(self._on_move_step_up)

        btn_down = QPushButton("▼")
        btn_down.setFixedWidth(26)
        btn_down.setToolTip("Przesuń krok w dół")
        btn_down.clicked.connect(self._on_move_step_down)

        btn_del_step = QPushButton("✕")
        btn_del_step.setFixedWidth(26)
        btn_del_step.setToolTip("Usuń zaznaczony krok")
        btn_del_step.clicked.connect(self._on_remove_pipeline_step)

        btn_clear_pipe = QPushButton("Wyczyść")
        btn_clear_pipe.clicked.connect(self._on_clear_pipeline)

        row_step_actions.addWidget(btn_edit_step)
        row_step_actions.addWidget(btn_up)
        row_step_actions.addWidget(btn_down)
        row_step_actions.addWidget(btn_del_step)
        row_step_actions.addStretch()
        row_step_actions.addWidget(btn_clear_pipe)
        l_pipe.addLayout(row_step_actions)

        # Zakres docelowy
        row_target = QHBoxLayout()
        row_target.addWidget(QLabel("Cel:"))
        self.combo_target_scope = QComboBox()
        self.combo_target_scope.addItems([
            "Wszystkie widoczne w aktywnym oknie / gałęzi",
            "Tylko aktywne widmo",
            "Aktywna seria 2D / Tensor"
        ])
        self.combo_target_scope.setCurrentIndex(0)
        row_target.addWidget(self.combo_target_scope, 1)
        l_pipe.addLayout(row_target)

        self.chk_inplace = QCheckBox("Modyfikuj w miejscu (In-place z Undo)")
        self.chk_inplace.setChecked(True)
        l_pipe.addWidget(self.chk_inplace)

        self.pipe_progress = QProgressBar()
        self.pipe_progress.setValue(0)
        self.pipe_progress.setFixedHeight(14)
        self.pipe_progress.setVisible(False)
        l_pipe.addWidget(self.pipe_progress)

        self.btn_run_pipeline = QPushButton("⚡ Uruchom Potok na Wybranych Danych")
        self.btn_run_pipeline.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 7px;")
        self.btn_run_pipeline.clicked.connect(self._on_run_pipeline)
        l_pipe.addWidget(self.btn_run_pipeline)

        layout.addWidget(box_pipeline)

        # SEKCJA 2: POJEDYNCZE NARZĘDZIA
        box_vapor = QGroupBox("💨 Kompensacja Atmosferyczna (VaporFit)")
        l_vapor = QVBoxLayout(box_vapor)
        self.btn_vaporfit = QPushButton("Otwórz VaporFit (Batch)...")
        self.btn_vaporfit.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 6px;")
        self.btn_vaporfit.clicked.connect(self._open_vaporfit_dialog)
        l_vapor.addWidget(self.btn_vaporfit)
        layout.addWidget(box_vapor)

        box_crop = QGroupBox("✂ Przycinanie Zakresu (Crop)")
        l_crop = QVBoxLayout(box_crop)
        row_crop = QHBoxLayout()
        row_crop.addWidget(QLabel("Zakres:"))
        self.ent_crop_min = QLineEdit("1000")
        self.ent_crop_max = QLineEdit("1800")
        row_crop.addWidget(self.ent_crop_min)
        row_crop.addWidget(QLabel("—"))
        row_crop.addWidget(self.ent_crop_max)
        l_crop.addLayout(row_crop)

        self.btn_crop = QPushButton("Przytnij aktywne widmo")
        self.btn_crop.clicked.connect(self._apply_crop)
        l_crop.addWidget(self.btn_crop)
        layout.addWidget(box_crop)

        box_baseline = QGroupBox("📏 Linia Bazowa")
        l_baseline = QVBoxLayout(box_baseline)

        row_asls = QHBoxLayout()
        row_asls.addWidget(QLabel("λ:"))
        self.ent_asls_lam = QLineEdit("100000")
        row_asls.addWidget(self.ent_asls_lam)
        row_asls.addWidget(QLabel("p:"))
        self.ent_asls_p = QLineEdit("0.001")
        row_asls.addWidget(self.ent_asls_p)
        l_baseline.addLayout(row_asls)

        row_bl_btns = QHBoxLayout()
        self.btn_asls = QPushButton("AsLS")
        self.btn_asls.clicked.connect(self._apply_asls)
        self.btn_airpls = QPushButton("airPLS / airALS")
        self.btn_airpls.clicked.connect(self._apply_airpls)
        row_bl_btns.addWidget(self.btn_asls)
        row_bl_btns.addWidget(self.btn_airpls)
        l_baseline.addLayout(row_bl_btns)

        row_bl_btns2 = QHBoxLayout()
        self.btn_poly_bl = QPushButton("Wielomianowa (st. 3)")
        self.btn_poly_bl.clicked.connect(self._apply_poly_bl)
        self.btn_linear_bl = QPushButton("Dwa krańce")
        self.btn_linear_bl.clicked.connect(self._apply_linear_endpoints_bl)
        row_bl_btns2.addWidget(self.btn_poly_bl)
        row_bl_btns2.addWidget(self.btn_linear_bl)
        l_baseline.addLayout(row_bl_btns2)
        layout.addWidget(box_baseline)

        box_filters = QGroupBox("〰 Filtr Savitzky-Golay & Pochodne")
        l_filters = QVBoxLayout(box_filters)

        row_sg = QHBoxLayout()
        row_sg.addWidget(QLabel("Okno (pkt):"))
        self.ent_sg_win = QLineEdit("11")
        row_sg.addWidget(self.ent_sg_win)

        row_sg.addWidget(QLabel("Wielomian:"))
        self.ent_sg_poly = QLineEdit("3")
        row_sg.addWidget(self.ent_sg_poly)
        l_filters.addLayout(row_sg)

        self.btn_sg_smooth = QPushButton("Wygładź (SG)")
        self.btn_sg_smooth.clicked.connect(self._apply_sg_smoothing)
        l_filters.addWidget(self.btn_sg_smooth)

        row_derivs = QHBoxLayout()
        self.btn_deriv1 = QPushButton("1. Pochodna")
        self.btn_deriv1.clicked.connect(lambda: self._apply_derivative(1))
        self.btn_deriv2 = QPushButton("2. Pochodna (-)")
        self.btn_deriv2.clicked.connect(lambda: self._apply_derivative(2))
        row_derivs.addWidget(self.btn_deriv1)
        row_derivs.addWidget(self.btn_deriv2)
        l_filters.addLayout(row_derivs)
        layout.addWidget(box_filters)

        self.btn_fsd = QPushButton("Dekonwolucja Fouriera (FSD)")
        self.btn_fsd.clicked.connect(self._apply_fsd)
        l_filters.addWidget(self.btn_fsd)

        box_norm = QGroupBox("⚖ Normalizacje & Offset")
        l_norm = QVBoxLayout(box_norm)

        row_norm1 = QHBoxLayout()
        self.btn_snv = QPushButton("SNV")
        self.btn_snv.clicked.connect(self._apply_snv)
        self.btn_minmax = QPushButton("Min-Max [0, 1]")
        self.btn_minmax.clicked.connect(self._apply_minmax)
        row_norm1.addWidget(self.btn_snv)
        row_norm1.addWidget(self.btn_minmax)
        l_norm.addLayout(row_norm1)

        row_norm2 = QHBoxLayout()
        self.btn_l2 = QPushButton("Norma L₂")
        self.btn_l2.clicked.connect(self._apply_l2)
        self.btn_offset_zero = QPushButton("Offset (Min ➔ 0)")
        self.btn_offset_zero.clicked.connect(self._apply_offset_min)
        row_norm2.addWidget(self.btn_l2)
        row_norm2.addWidget(self.btn_offset_zero)
        l_norm.addLayout(row_norm2)

        row_offset_pt = QHBoxLayout()
        row_offset_pt.addWidget(QLabel("Offset przy:"))
        self.ent_offset_wn = QLineEdit("2000")
        self.ent_offset_wn.setFixedWidth(50)
        row_offset_pt.addWidget(self.ent_offset_wn)
        row_offset_pt.addWidget(QLabel("cm⁻¹"))

        self.btn_offset_point = QPushButton("A(ν) ➔ 0.0")
        self.btn_offset_point.clicked.connect(self._apply_offset_at_point)
        row_offset_pt.addWidget(self.btn_offset_point)
        l_norm.addLayout(row_offset_pt)

        layout.addWidget(box_norm)

        # Historia
        box_history = QGroupBox("↺ Historia Operacji")
        l_hist = QHBoxLayout(box_history)
        self.btn_undo = QPushButton("↶ Cofnij")
        self.btn_undo.setToolTip("Cofnij operację (⌘Z / Ctrl+Z)")
        self.btn_undo.clicked.connect(session.undo_stack.undo)
        session.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.btn_undo.setEnabled(session.undo_stack.canUndo())

        self.btn_redo = QPushButton("↷ Ponów")
        self.btn_redo.setToolTip("Ponów operację (⌘⇧Z / Ctrl+Y)")
        self.btn_redo.clicked.connect(session.undo_stack.redo)
        session.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_redo.setEnabled(session.undo_stack.canRedo())

        l_hist.addWidget(self.btn_undo)
        l_hist.addWidget(self.btn_redo)
        layout.addWidget(box_history)

        layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _open_vaporfit_dialog(self):
        if self._vaporfit_dialog is None:
            self._vaporfit_dialog = VaporFitDialog(parent=self.window())
        self._vaporfit_dialog.show()
        self._vaporfit_dialog.raise_()
        self._vaporfit_dialog.activateWindow()

    def _on_add_pipeline_step(self):
        choice = self.combo_step_type.currentText()
        step = {}

        try:
            if "Przycięcie" in choice:
                min_w = float(self.ent_crop_min.text())
                max_w = float(self.ent_crop_max.text())
                step = {'type': 'crop', 'name': f"Przycięcie [{min_w:.0f}–{max_w:.0f}]", 'min_w': min_w, 'max_w': max_w}

            elif "AsLS" in choice:
                lam = float(self.ent_asls_lam.text())
                p = float(self.ent_asls_p.text())
                step = {'type': 'asls', 'name': f"AsLS (λ={lam:.0e}, p={p})", 'lam': lam, 'p': p}

            elif "airPLS" in choice or "airALS" in choice:
                lam = float(self.ent_asls_lam.text())
                step = {'type': 'airpls', 'name': f"airPLS / airALS (λ={lam:.0e})", 'lam': lam, 'n_iter': 15}

            elif "Wielomianowa" in choice:
                step = {'type': 'poly_bl', 'name': "Wielomian st. 3", 'deg': 3}

            elif "Dwa krańce" in choice:
                step = {'type': 'linear_bl', 'name': "Linia bazowa (krańce)"}

            elif "FSD" in choice or "Dekonwolucja" in choice:
                step = {'type': 'fsd', 'name': "FSD (FWHM=18.0, K=2.0)", 'fwhm': 18.0, 'k_factor': 2.0}

            elif "Wygładzanie" in choice:
                win = int(self.ent_sg_win.text())
                poly = int(self.ent_sg_poly.text())
                step = {'type': 'sg_smooth', 'name': f"SG Wygładzanie (okno={win}, st={poly})", 'win': win, 'poly': poly}

            elif "1. Pochodna" in choice:
                win = int(self.ent_sg_win.text())
                poly = int(self.ent_sg_poly.text())
                step = {'type': 'deriv1', 'name': f"1. Pochodna SG (okno={win})", 'win': win, 'poly': poly}

            elif "2. Pochodna" in choice:
                win = int(self.ent_sg_win.text())
                poly = int(self.ent_sg_poly.text())
                step = {'type': 'deriv2', 'name': f"2. Pochodna SG (-) (okno={win})", 'win': win, 'poly': poly}

            elif "SNV" in choice:
                step = {'type': 'snv', 'name': "Normalizacja SNV"}

            elif "Min-Max" in choice:
                step = {'type': 'minmax', 'name': "Normalizacja Min-Max"}

            elif "Norma L₁" in choice:
                step = {'type': 'l1', 'name': "Normalizacja L₁ (Pole)"}

            elif "Norma L₂" in choice:
                step = {'type': 'l2', 'name': "Normalizacja L₂"}

            elif "Min ➔ 0" in choice:
                step = {'type': 'offset_min', 'name': "Offset (Min ➔ 0.0)"}

            elif "punkcie ν" in choice:
                t_wn = float(self.ent_offset_wn.text())
                step = {'type': 'offset_point', 'name': f"Offset w punkcie {t_wn:.1f} cm⁻¹ ➔ 0", 'target_wn': t_wn}

            elif "Centrowanie" in choice:
                step = {'type': 'mean_center', 'name': "Centrowanie do średniej"}

            elif "Autoskalowanie" in choice:
                step = {'type': 'autoscaling', 'name': "Autoskalowanie (Unit Variance)"}

            elif "Pareto" in choice:
                step = {'type': 'pareto', 'name': "Skalowanie Pareto"}

            elif "MSC" in choice:
                step = {'type': 'msc', 'name': "Korekcja rozproszenia (MSC)"}

        except ValueError:
            QMessageBox.warning(self, tr("app.warning"), "Nieprawidłowe wartości w polach parametrów.")
            return

        if step:
            self.pipeline_steps.append(step)
            self.list_pipeline.addItem(f"{len(self.pipeline_steps)}. {step['name']}")

    def _on_edit_pipeline_step(self):
        row = self.list_pipeline.currentRow()
        if not (0 <= row < len(self.pipeline_steps)):
            return

        step = self.pipeline_steps[row]
        dlg = StepParamDialog(step, self)
        if dlg.exec() == QDialog.Accepted:
            self._refresh_pipeline_list()
            self.list_pipeline.setCurrentRow(row)

    def _on_move_step_up(self):
        row = self.list_pipeline.currentRow()
        if row > 0:
            self.pipeline_steps[row - 1], self.pipeline_steps[row] = self.pipeline_steps[row], self.pipeline_steps[row - 1]
            self._refresh_pipeline_list()
            self.list_pipeline.setCurrentRow(row - 1)

    def _on_move_step_down(self):
        row = self.list_pipeline.currentRow()
        if 0 <= row < len(self.pipeline_steps) - 1:
            self.pipeline_steps[row + 1], self.pipeline_steps[row] = self.pipeline_steps[row], self.pipeline_steps[row + 1]
            self._refresh_pipeline_list()
            self.list_pipeline.setCurrentRow(row + 1)

    def _on_remove_pipeline_step(self):
        row = self.list_pipeline.currentRow()
        if 0 <= row < len(self.pipeline_steps):
            self.pipeline_steps.pop(row)
            self._refresh_pipeline_list()

    def _on_clear_pipeline(self):
        self.pipeline_steps.clear()
        self.list_pipeline.clear()

    def _refresh_pipeline_list(self):
        self.list_pipeline.clear()
        for idx, step in enumerate(self.pipeline_steps):
            self.list_pipeline.addItem(f"{idx + 1}. {step['name']}")

    def _apply_pipeline_to_matrix(self, x: np.ndarray, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        curr_x = x.copy()
        curr_mat = mat.copy()

        for step in self.pipeline_steps:
            stype = step['type']

            if stype == 'crop':
                curr_x, curr_mat = prep.crop_range(curr_x, curr_mat, step['min_w'], step['max_w'])

            elif stype == 'asls':
                for c in range(curr_mat.shape[1]):
                    baseline = prep.baseline_asls(curr_mat[:, c], lam=step['lam'], p=step['p'])
                    curr_mat[:, c] -= baseline

            elif stype == 'airpls':
                for c in range(curr_mat.shape[1]):
                    baseline = prep.baseline_airpls(curr_mat[:, c], lam=step.get('lam', 1e5),
                                                    n_iter=step.get('n_iter', 15))
                    curr_mat[:, c] -= baseline

            elif stype == 'poly_bl':
                for c in range(curr_mat.shape[1]):
                    baseline = prep.baseline_polynomial(curr_mat[:, c], curr_x, deg=step.get('deg', 3))
                    curr_mat[:, c] -= baseline

            elif stype == 'linear_bl':
                if len(curr_x) >= 2:
                    for c in range(curr_mat.shape[1]):
                        baseline = prep.baseline_linear(curr_x, curr_mat[:, c], curr_x[0], curr_x[-1])
                        curr_mat[:, c] -= baseline

            elif stype == 'fsd':
                for c in range(curr_mat.shape[1]):
                    curr_mat[:, c] = prep.fourier_self_deconvolution(
                        curr_x, curr_mat[:, c], fwhm=step.get('fwhm', 18.0), k_factor=step.get('k_factor', 2.0)
                    )

            elif stype == 'sg_smooth':
                curr_mat = prep.apply_savgol(curr_mat, window=step['win'], poly=step['poly'], deriv=0)

            elif stype == 'deriv1':
                curr_mat = prep.apply_savgol(curr_mat, window=step['win'], poly=step['poly'], deriv=1)

            elif stype == 'deriv2':
                curr_mat = -1.0 * prep.apply_savgol(curr_mat, window=step['win'], poly=step['poly'], deriv=2)

            elif stype == 'snv':
                curr_mat = prep.apply_snv(curr_mat)

            elif stype == 'minmax':
                curr_mat = prep.apply_min_max(curr_mat)

            elif stype == 'l1':
                curr_mat = prep.apply_norm_l1(curr_mat)

            elif stype == 'l2':
                curr_mat = prep.apply_vector_norm_l2(curr_mat)

            elif stype == 'offset_min':
                curr_mat -= np.min(curr_mat, axis=0, keepdims=True)

            elif stype == 'offset_point':
                curr_mat = prep.apply_offset(curr_mat, curr_x, target_wn=step['target_wn'], target_val=0.0)

            elif stype == 'mean_center':
                if curr_mat.shape[1] > 1:
                    curr_mat, _ = prep.apply_mean_center(curr_mat)

            elif stype == 'autoscaling':
                if curr_mat.shape[1] > 1:
                    curr_mat = prep.apply_autoscaling(curr_mat)

            elif stype == 'pareto':
                if curr_mat.shape[1] > 1:
                    curr_mat = prep.apply_pareto_scaling(curr_mat)

            elif stype == 'msc':
                if curr_mat.shape[1] > 1:
                    curr_mat = prep.apply_msc(curr_mat)

        return curr_x, curr_mat

    def _apply_pipeline_to_vectors(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_x, new_mat = self._apply_pipeline_to_matrix(x, y.reshape(-1, 1))
        return new_x, new_mat.flatten()

    def _on_run_pipeline(self):
        if not self.pipeline_steps:
            QMessageBox.warning(self, tr("app.warning"), "Potok jest pusty. Dodaj co najmniej jeden krok.")
            return

        scope = self.combo_target_scope.currentIndex()
        in_place = self.chk_inplace.isChecked()
        canvas = self._get_canvas()
        target_win = session.active_window_id if session.active_context == "window" else None

        # 1. WSZYSTKIE WIDOCZNE W AKTYWNYM WIDOKU / GAŁĘZI (SCOPE 0 - DOMYŚLNY)
        if scope == 0:
            visible_spectra = session.get_pipeline_target_spectra()
            if not visible_spectra:
                QMessageBox.warning(self, tr("app.warning"), "Brak widocznych widm w aktywnym oknie lub gałęzi.")
                return

            total = len(visible_spectra)
            self.pipe_progress.setMaximum(total)
            self.pipe_progress.setValue(0)
            self.pipe_progress.setVisible(True)
            self.btn_run_pipeline.setEnabled(False)

            ref_wn = visible_spectra[0].wavenumbers
            mat = np.zeros((len(ref_wn), total), dtype=np.float64)
            for i, spec in enumerate(visible_spectra):
                if spec.wavenumbers[0] > spec.wavenumbers[-1]:
                    mat[:, i] = np.interp(ref_wn, spec.wavenumbers[::-1], spec.intensities[::-1])
                else:
                    mat[:, i] = np.interp(ref_wn, spec.wavenumbers, spec.intensities)

            new_x, new_mat = self._apply_pipeline_to_matrix(ref_wn, mat)

            batch_records = []
            new_created_spectra = []

            for idx, spec in enumerate(visible_spectra):
                old_x = spec.wavenumbers.copy()
                old_y = spec.intensities.copy()
                new_y = new_mat[:, idx]

                old_meta = copy.deepcopy(spec.metadata)
                new_meta = copy.deepcopy(spec.metadata)
                new_meta["last_pipeline"] = [s['name'] for s in self.pipeline_steps]

                if in_place:
                    batch_records.append((spec, old_x, old_y, new_x, new_y, old_meta, new_meta))
                else:
                    new_spec = SpectrumData(
                        wavenumbers=new_x.copy(), intensities=new_y.copy(),
                        name=f"proc_{spec.name}", file_path=spec.file_path,
                        file_type="processed", metadata=new_meta
                    )
                    new_created_spectra.append(new_spec)

                self.pipe_progress.setValue(idx + 1)
                QApplication.processEvents()

            if in_place and batch_records:
                cmd = BatchPipelineCommand(batch_records, f"Potok wsadowy ({len(self.pipeline_steps)} kroków)",
                                           canvas=canvas)
                session.undo_stack.push(cmd)
            elif not in_place and new_created_spectra:
                cmd = AddBatchSpectraCommand(new_created_spectra, f"Potok wsadowy ({len(new_created_spectra)} widm)",
                                             canvas=canvas, window_id=target_win)
                session.undo_stack.push(cmd)

            self.pipe_progress.setVisible(False)
            self.btn_run_pipeline.setEnabled(True)
            QMessageBox.information(self, tr("app.info"), f"Przetworzono {total} widm potokiem.")

        # 2. TYLKO AKTYWNE WIDMO (SCOPE 1)
        elif scope == 1:
            active = session.active_spectrum
            if not active:
                QMessageBox.warning(self, tr("app.warning"), "Brak aktywnego widma.")
                return

            has_matrix_step = any(
                s['type'] in ['mean_center', 'autoscaling', 'pareto', 'msc'] for s in self.pipeline_steps)
            if has_matrix_step:
                QMessageBox.warning(
                    self, tr("app.warning"),
                    "Potok zawiera skalowania macierzowe (Mean-center / Autoscaling / Pareto / MSC),\n"
                    "które wymagają serii widm (min. 2). Dla pojedynczego widma kroki te zostaną pominięte."
                )

            new_x, new_y = self._apply_pipeline_to_vectors(active.wavenumbers, active.intensities)
            new_meta = copy.deepcopy(active.metadata)
            new_meta["last_pipeline"] = [s['name'] for s in self.pipeline_steps]

            if in_place:
                cmd = ModifySpectrumCommand(
                    active, f"Potok: {active.name}",
                    active.intensities, new_y, active.wavenumbers, new_x,
                    old_meta=active.metadata, new_meta=new_meta, canvas=canvas
                )
                session.undo_stack.push(cmd)
            else:
                new_spec = SpectrumData(
                    wavenumbers=new_x, intensities=new_y,
                    name=f"proc_{active.name}", file_path=active.file_path,
                    file_type="processed", metadata=new_meta
                )
                cmd = AddBatchSpectraCommand([new_spec], f"Potok: proc_{active.name}", canvas=canvas,
                                             window_id=target_win)
                session.undo_stack.push(cmd)

        # 3. AKTYWNA SERIA 2D / TENSOR 3D (SCOPE 2)
        elif scope == 2:
            active_s = session.active_series
            if not active_s:
                QMessageBox.warning(self, tr("app.warning"), "Brak aktywnej serii 2D.")
                return

            mat = active_s.get_matrix_2d()
            wn = active_s.wavenumbers

            self.pipe_progress.setMaximum(mat.shape[1])
            self.pipe_progress.setValue(0)
            self.pipe_progress.setVisible(True)
            self.btn_run_pipeline.setEnabled(False)

            new_x, new_mat = self._apply_pipeline_to_matrix(wn, mat)

            if in_place:
                old_wn = active_s.wavenumbers.copy()
                old_data = active_s.data.copy()

                if active_s.is_3d:
                    new_data = new_mat.reshape(len(new_x), active_s.data.shape[1], active_s.data.shape[2])
                else:
                    new_data = new_mat

                cmd = ModifySeriesCommand(active_s, f"Potok serii: {active_s.name}",
                                          old_wn, new_x, old_data, new_data, canvas=canvas)
                session.undo_stack.push(cmd)
            else:
                out_data = new_mat.reshape(len(new_x), active_s.data.shape[1],
                                           active_s.data.shape[2]) if active_s.is_3d else new_mat
                new_series = SpectralSeries(
                    name=f"proc_{active_s.name}",
                    wavenumbers=new_x,
                    data=out_data,
                    axis_n_name=active_s.axis_n_name,
                    axis_n_values=active_s.axis_n_values,
                    axis_m_name=active_s.axis_m_name,
                    axis_m_values=active_s.axis_m_values
                )
                session.add_series(new_series, make_active=True)

            self.pipe_progress.setVisible(False)
            self.btn_run_pipeline.setEnabled(True)
            QMessageBox.information(self, tr("app.info"),
                                    f"Przetworzono serię '{active_s.name}' ({mat.shape[1]} widm)!")

    # ======================================================================
    # POJEDYNCZE OPERACJE MANUALNE
    # ======================================================================
    def _apply_airpls(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            lam = float(self.ent_asls_lam.text())
            baseline = prep.baseline_airpls(spec.intensities, lam=lam)
            new_y = spec.intensities - baseline
            cmd = ModifySpectrumCommand(spec, "airPLS Baseline", spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error"), f"Błąd airPLS: {e}")

    def _apply_poly_bl(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            baseline = prep.baseline_polynomial(spec.intensities, spec.wavenumbers, deg=3)
            new_y = spec.intensities - baseline
            cmd = ModifySpectrumCommand(spec, "Wielomianowa st. 3", spec.intensities, new_y,
                                        canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error"), f"Błąd linii wielomianowej: {e}")

    def _apply_fsd(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            new_y = prep.fourier_self_deconvolution(spec.wavenumbers, spec.intensities, fwhm=18.0, k_factor=2.0)
            cmd = ModifySpectrumCommand(spec, "FSD (FWHM=18, K=2)", spec.intensities, new_y,
                                        canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error"), f"Błąd FSD: {e}")

    def _apply_asls(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            lam = float(self.ent_asls_lam.text())
            p = float(self.ent_asls_p.text())
            baseline = prep.baseline_asls(spec.intensities, lam=lam, p=p)
            new_y = spec.intensities - baseline

            cmd = ModifySpectrumCommand(spec, "AsLS Baseline", spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd AsLS: {e}")

    def _apply_linear_endpoints_bl(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec or len(spec.intensities) < 2:
            return
        baseline = prep.baseline_linear(spec.wavenumbers, spec.intensities, spec.wavenumbers[0], spec.wavenumbers[-1])
        new_y = spec.intensities - baseline

        cmd = ModifySpectrumCommand(spec, "Linia bazowa (krańce)", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_sg_smoothing(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            win = int(self.ent_sg_win.text())
            poly = int(self.ent_sg_poly.text())
            new_y = prep.apply_savgol(spec.intensities, window=win, poly=poly, deriv=0)

            cmd = ModifySpectrumCommand(spec, "Wygładzanie SG", spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd filtru SG: {e}")

    def _apply_derivative(self, order: int):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            win = int(self.ent_sg_win.text())
            poly = int(self.ent_sg_poly.text())
            if order == 1:
                _, new_y = prep.get_first_derivative(spec.wavenumbers, spec.intensities, window=win, poly=poly)
                desc = "1. Pochodna SG"
            else:
                _, new_y = prep.get_second_derivative_inverted(spec.wavenumbers, spec.intensities, window=win, poly=poly)
                desc = "2. Pochodna SG (-)"

            cmd = ModifySpectrumCommand(spec, desc, spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd pochodnej: {e}")

    def _apply_snv(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        new_y = prep.apply_snv(spec.intensities)
        cmd = ModifySpectrumCommand(spec, "Normalizacja SNV", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_minmax(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        new_y = prep.apply_min_max(spec.intensities)
        cmd = ModifySpectrumCommand(spec, "Normalizacja Min-Max", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_l2(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        new_y = prep.apply_vector_norm_l2(spec.intensities)
        cmd = ModifySpectrumCommand(spec, "Normalizacja L2", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_offset_min(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        min_val = np.min(spec.intensities)
        new_y = spec.intensities - min_val
        cmd = ModifySpectrumCommand(spec, "Offset do 0.0", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_offset_at_point(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            target_wn = float(self.ent_offset_wn.text())
            new_y = prep.apply_offset(spec.intensities, spec.wavenumbers, target_wn=target_wn, target_val=0.0)
            cmd = ModifySpectrumCommand(spec, f"Offset przy {target_wn:.1f} cm⁻¹", spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except ValueError:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nieprawidłowa liczba falowa punktu offsetu.")

    def _apply_crop(self):
        if not self._verify_single_tool_permission():
            return
        spec = session.active_spectrum
        if not spec:
            return
        try:
            min_w = float(self.ent_crop_min.text())
            max_w = float(self.ent_crop_max.text())
            new_x, new_y = prep.crop_range(spec.wavenumbers, spec.intensities, min_w, max_w)

            cmd = ModifySpectrumCommand(
                spec, f"Przycięcie [{min_w:.0f}-{max_w:.0f}]",
                spec.intensities, new_y,
                spec.wavenumbers, new_x,
                canvas=self._get_canvas()
            )
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), f"Błąd przycinania: {e}")

    def retranslate_ui(self):
        pass