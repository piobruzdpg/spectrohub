"""
gui/workspaces/ws_preprocessing.py
Panel preprocessingu i przygotowania danych spektroskopowych (Workspace 1).
Zintegrowany z silnikiem hurtowej kompensacji VaporFit (Batch), AsLS, SG i QUndoStack.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QMessageBox, QDialog, QComboBox,
    QListWidget, QListWidgetItem, QScrollArea, QFrame,
    QProgressBar, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoCommand
import numpy as np
import logging

from core.session import session
from core.models import SpectrumData
from core.i18n import tr, i18n
from algorithms import preprocessing as prep
from algorithms.atmospheric import AtmosphericCorrectionEngine

logger = logging.getLogger("SpectroHub")


# ==============================================================================
# KOMENDY HISTORII DLA OPERACJI PREPROCESSINGU (QUndoStack)
# ==============================================================================
class ModifySpectrumCommand(QUndoCommand):
    """Cofa lub ponawia operacje modyfikujące intensywności widma in-place."""

    def __init__(self, spec: SpectrumData, description: str,
                 old_y: np.ndarray, new_y: np.ndarray,
                 old_x: np.ndarray = None, new_x: np.ndarray = None,
                 canvas=None):
        super().__init__(description)
        self.spec = spec
        self.old_y = old_y.copy()
        self.new_y = new_y.copy()
        self.old_x = old_x.copy() if old_x is not None else None
        self.new_x = new_x.copy() if new_x is not None else None
        self.canvas = canvas

    def undo(self):
        self.spec.intensities = self.old_y.copy()
        if self.old_x is not None:
            self.spec.wavenumbers = self.old_x.copy()
        if self.canvas:
            self.canvas.redraw_all()

    def redo(self):
        self.spec.intensities = self.new_y.copy()
        if self.new_x is not None:
            self.spec.wavenumbers = self.new_x.copy()
        if self.canvas:
            self.canvas.redraw_all()


# ==============================================================================
# DIALOG HURTOWEJ KOMPENSACJI ATMOSFERYCZNEJ (BATCH VAPORFIT)
# ==============================================================================
class VaporFitDialog(QDialog):
    """Okno konfiguracji i uruchomienia kompensacji par wody i CO2 (pojedynczo lub hurtowo)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("vapor.title", "VaporFit — Hurtowa Kompensacja Atmosferyczna"))
        self.resize(750, 620)
        self.setModal(True)

        self._init_ui()
        self._load_available_spectra()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 1. Górny panel z dwoma listami: Próbki (lewa) oraz Atmosfera (prawa)
        row_lists = QHBoxLayout()

        # Lewa kolumna: Próbki do skorygowania
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

        # Prawa kolumna: Widma referencyjne atmosfery
        box_atm = QGroupBox(tr("vapor.atm_group", "2. Widma odniesienia atmosfery (H₂O / CO₂)"))
        l_atm = QVBoxLayout(box_atm)

        row_atm_tools = QHBoxLayout()
        btn_atm_none = QPushButton("Odznacz wszystko")
        btn_atm_none.clicked.connect(lambda: self._set_all_checks(self.list_atm, Qt.Unchecked))
        row_atm_tools.addWidget(btn_atm_none)
        row_atm_tools.addStretch()
        l_atm.addLayout(row_atm_tools)

        self.list_atm = QListWidget()
        l_atm.addWidget(self.list_atm)
        row_lists.addWidget(box_atm, stretch=1)

        layout.addLayout(row_lists)

        # 2. Parametry dopasowania
        box_params = QGroupBox(tr("vapor.params_group", "3. Parametry algorytmu optymalizacyjnego"))
        l_params = QVBoxLayout(box_params)

        # Zakres, SG i Wybór trybu znaku
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

        # Kontrolka znaku
        row_inputs.addWidget(QLabel("Kierunek:"))
        self.combo_sign = QComboBox()
        self.combo_sign.addItems(["Auto (Wygładź)", "Odejmij (-)", "Dodaj (+)"])
        row_inputs.addWidget(self.combo_sign)

        self.btn_find_opt = QPushButton("🔍 Szukaj SG")
        self.btn_find_opt.clicked.connect(self._on_find_optimal_window)
        row_inputs.addWidget(self.btn_find_opt)

        l_params.addLayout(row_inputs)

        layout.addWidget(box_params)

        # 3. Pasek postępu i diagnostyka
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

        # 4. Przyciski akcji
        row_actions = QHBoxLayout()
        self.btn_apply = QPushButton("⚡ Uruchom korekcję hurtową (Batch)")
        self.btn_apply.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 8px;")
        self.btn_apply.clicked.connect(self._on_apply_batch_correction)

        self.btn_cancel = QPushButton("Zamknij")
        self.btn_cancel.clicked.connect(self.reject)

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
        all_spectra = session.spectra

        # Blokada zdarzeń i odświeżania w celu wyeliminowania ostrzeżeń Cocoa Accessibility na macOS
        self.list_samples.setUpdatesEnabled(False)
        self.list_atm.setUpdatesEnabled(False)
        self.list_samples.blockSignals(True)
        self.list_atm.blockSignals(True)

        self.list_samples.clear()
        self.list_atm.clear()

        for name, spec in all_spectra.items():
            # Pozycja na liście próbek
            item_s = QListWidgetItem(f"{name} ({spec.n_points} pkt)")
            item_s.setData(Qt.UserRole, name)
            item_s.setFlags(item_s.flags() | Qt.ItemIsUserCheckable)
            is_active = (active and name == active.name)
            item_s.setCheckState(Qt.Checked if is_active else Qt.Unchecked)
            self.list_samples.addItem(item_s)

            # Pozycja na liście atmosfery
            item_a = QListWidgetItem(f"{name} ({spec.n_points} pkt)")
            item_a.setData(Qt.UserRole, name)
            item_a.setFlags(item_a.flags() | Qt.ItemIsUserCheckable)
            # Domyślnie zaznacz, jeśli nazwa sugeruje parę/atmosferę
            is_atm_name = any(k in name.lower() for k in ['vap', 'atm', 'h2o', 'co2', 'para'])
            item_a.setCheckState(Qt.Checked if is_atm_name and not is_active else Qt.Unchecked)
            self.list_atm.addItem(item_a)

        self.list_samples.blockSignals(False)
        self.list_atm.blockSignals(False)
        self.list_samples.setUpdatesEnabled(True)
        self.list_atm.setUpdatesEnabled(True)

        if active:
            self.ent_min_wn.setText(f"{np.min(active.wavenumbers):.1f}")
            self.ent_max_wn.setText(f"{np.max(active.wavenumbers):.1f}")

    def _get_checked_names(self, list_widget: QListWidget):
        selected = []
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def _on_find_optimal_window(self):
        sample_names = self._get_checked_names(self.list_samples)
        atm_names = self._get_checked_names(self.list_atm)

        if not sample_names or not atm_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"),
                                "Zaznacz co najmniej jedno widmo próbki i jedno widmo atmosfery.")
            return

        # Do estymacji bierzemy pierwszą zaznaczoną próbkę
        sample_spec = session.spectra[sample_names[0]]
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
            a_spec = session.spectra[a_name]
            # Uporządkowanie monotoniczne do np.interp
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
            self,
            tr("app.info", "Informacja"),
            f"Sugerowane optymalne okno SG (próbka: {sample_spec.name}):\n"
            f"• Wg krzywizny SDV: {res['best_window_sdv']} pkt\n"
            f"• Wg wskaźnika SSI: {res['best_window_ssi']} pkt\n\n"
            f"Wartość {opt_win} została wstawiona do konfiguracji."
        )

    def _on_apply_batch_correction(self):
        sample_names = self._get_checked_names(self.list_samples)
        atm_names = self._get_checked_names(self.list_atm)

        if not sample_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nie zaznaczono żadnego widma do korekcji.")
            return

        if not atm_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Wybierz co najmniej jedno widmo odniesienia atmosfery.")
            return

        # Wykluczenie widm atmosfery z listy próbek, jeśli zostały przypadkowo zaznaczone po obu stronach
        final_sample_names = [s for s in sample_names if s not in atm_names]
        if not final_sample_names:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"),
                                "Widma badane nie mogą być jednocześnie widmami odniesienia atmosfery.")
            return

        try:
            poly = int(self.ent_sg_poly.text())
            points = int(self.ent_sg_points.text())
            min_w = float(self.ent_min_wn.text())
            max_w = float(self.ent_max_wn.text())
        except ValueError:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nieprawidłowe wartości parametrów numerycznych.")
            return

        total_count = len(final_sample_names)
        self.btn_apply.setEnabled(False)
        self.progress_bar.setMaximum(total_count)
        self.progress_bar.setValue(0)

        created_spectra = []
        errors = []

        # Przygotowanie posortowanych baz atmosfery do szybkiej interpolacji
        prepared_atm = []
        for a_name in atm_names:
            a_spec = session.spectra[a_name]
            if a_spec.wavenumbers[0] > a_spec.wavenumbers[-1]:
                prepared_atm.append((a_spec.wavenumbers[::-1], a_spec.intensities[::-1]))
            else:
                prepared_atm.append((a_spec.wavenumbers, a_spec.intensities))

        sign_map = {
            "Auto (Wygładź)": "auto",
            "Odejmij (-)": "subtract",
            "Dodaj (+)": "add"
        }
        chosen_mode = sign_map.get(self.combo_sign.currentText(), "auto")

        for idx, s_name in enumerate(final_sample_names):
            self.lbl_status.setText(f"Korygowanie ({idx + 1}/{total_count}): {s_name}...")
            QApplication.processEvents()

            try:
                sample_spec = session.spectra[s_name]
                mask = (sample_spec.wavenumbers >= min_w) & (sample_spec.wavenumbers <= max_w)
                wn_crop = sample_spec.wavenumbers[mask]
                y_crop = sample_spec.intensities[mask]

                if len(wn_crop) < points + 2:
                    errors.append(f"{s_name}: Za mało punktów pomiarowych w wybranym zakresie.")
                    continue

                # Rzutowanie widm atmosfery na siatkę aktualnej próbki
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

        # Hurtowe dodanie skorygowanych widm do sesji
        if created_spectra:
            session.add_spectra_batch(created_spectra)

        self.btn_apply.setEnabled(True)
        self.lbl_status.setText(f"Ukończono! Skorygowano pomyślnie {len(created_spectra)}/{total_count} widm.")

        msg = f"Pomyślnie utworzono i dodano {len(created_spectra)} skorygowanych widm do sesji."
        if errors:
            msg += f"\n\nBłędy w {len(errors)} plikach:\n" + "\n".join(errors[:5])

        QMessageBox.information(self, tr("app.info", "Informacja"), msg)
        self.accept()


# ==============================================================================
# GŁÓWNY PANEL INSPEKTORA PREPROCESSINGU
# ==============================================================================
class PreprocessingPanel(QWidget):
    """Widok narzędzi obróbki i przygotowania danych (Workspace 1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        i18n.language_changed.connect(self.retranslate_ui)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)

        # 1. Narzędzie atmosferyczne VaporFit
        box_vapor = QGroupBox("💨 Kompensacja Atmosferyczna (VaporFit)")
        l_vapor = QVBoxLayout(box_vapor)
        self.btn_vaporfit = QPushButton("Otwórz VaporFit (Batch)...")
        self.btn_vaporfit.setStyleSheet("background-color: #2a9d8f; font-weight: bold; padding: 6px;")
        self.btn_vaporfit.clicked.connect(self._open_vaporfit_dialog)
        l_vapor.addWidget(self.btn_vaporfit)
        layout.addWidget(box_vapor)

        # 2. Korekcja linii bazowej
        box_baseline = QGroupBox("📏 Linia Bazowa")
        l_baseline = QVBoxLayout(box_baseline)

        row_asls = QHBoxLayout()
        row_asls.addWidget(QLabel("AsLS λ:"))
        self.ent_asls_lam = QLineEdit("100000")
        row_asls.addWidget(self.ent_asls_lam)

        row_asls.addWidget(QLabel("p:"))
        self.ent_asls_p = QLineEdit("0.001")
        row_asls.addWidget(self.ent_asls_p)
        l_baseline.addLayout(row_asls)

        self.btn_asls = QPushButton("Odejmij tło (AsLS)")
        self.btn_asls.clicked.connect(self._apply_asls)
        l_baseline.addWidget(self.btn_asls)

        self.btn_linear_bl = QPushButton("Odejmij linię (Dwa krańce)")
        self.btn_linear_bl.clicked.connect(self._apply_linear_endpoints_bl)
        l_baseline.addWidget(self.btn_linear_bl)

        layout.addWidget(box_baseline)

        # 3. Filtry i pochodne (Savitzky-Golay)
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

        # 4. Normalizacje i Offset
        box_norm = QGroupBox("⚖ Normalizacje & Skalowanie")
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
        self.btn_offset_zero = QPushButton("Offset (Min ➔ 0.0)")
        self.btn_offset_zero.clicked.connect(self._apply_offset_min)
        row_norm2.addWidget(self.btn_l2)
        row_norm2.addWidget(self.btn_offset_zero)
        l_norm.addLayout(row_norm2)

        layout.addWidget(box_norm)

        # 5. Przycinanie widma (Crop)
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

        # 6. Globalne Undo / Redo
        box_history = QGroupBox("↺ Historia Operacji")
        l_hist = QHBoxLayout(box_history)
        self.btn_undo = QPushButton("↶ Cofnij")
        self.btn_undo.clicked.connect(session.undo_stack.undo)
        session.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.btn_undo.setEnabled(False)

        self.btn_redo = QPushButton("↷ Ponów")
        self.btn_redo.clicked.connect(session.undo_stack.redo)
        session.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_redo.setEnabled(False)

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
        dlg = VaporFitDialog(self)
        dlg.exec()

    def _apply_asls(self):
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
        spec = session.active_spectrum
        if not spec or len(spec.intensities) < 2:
            return
        x1, y1 = spec.wavenumbers[0], spec.intensities[0]
        x2, y2 = spec.wavenumbers[-1], spec.intensities[-1]
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        baseline = m * spec.wavenumbers + c
        new_y = spec.intensities - baseline

        cmd = ModifySpectrumCommand(spec, "Linia bazowa (krańce)", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_sg_smoothing(self):
        spec = session.active_spectrum
        if not spec:
            return
        try:
            win = int(self.ent_sg_win.text())
            poly = int(self.ent_sg_poly.text())
            new_y = prep.smooth_savitzky_golay(spec.intensities, window_length=win, polyorder=poly)

            cmd = ModifySpectrumCommand(spec, "Wygładzanie SG", spec.intensities, new_y, canvas=self._get_canvas())
            session.undo_stack.push(cmd)
        except Exception as e:
            QMessageBox.critical(self, tr("app.error", "Błąd"), f"Błąd filtru SG: {e}")

    def _apply_derivative(self, order: int):
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
        spec = session.active_spectrum
        if not spec:
            return
        new_y = prep.snv(spec.intensities)
        cmd = ModifySpectrumCommand(spec, "Normalizacja SNV", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_minmax(self):
        spec = session.active_spectrum
        if not spec:
            return
        new_y = prep.min_max_scale(spec.intensities)
        cmd = ModifySpectrumCommand(spec, "Normalizacja Min-Max", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_l2(self):
        spec = session.active_spectrum
        if not spec:
            return
        norm = np.linalg.norm(spec.intensities)
        new_y = spec.intensities / norm if norm > 0 else spec.intensities
        cmd = ModifySpectrumCommand(spec, "Normalizacja L2", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_offset_min(self):
        spec = session.active_spectrum
        if not spec:
            return
        min_val = np.min(spec.intensities)
        new_y = spec.intensities - min_val
        cmd = ModifySpectrumCommand(spec, "Offset do 0.0", spec.intensities, new_y, canvas=self._get_canvas())
        session.undo_stack.push(cmd)

    def _apply_crop(self):
        spec = session.active_spectrum
        if not spec:
            return
        try:
            min_w = float(self.ent_crop_min.text())
            max_w = float(self.ent_crop_max.text())
            mask = (spec.wavenumbers >= min_w) & (spec.wavenumbers <= max_w)
            if not np.any(mask):
                QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Brak punktów pomiarowych w podanym zakresie.")
                return

            new_x = spec.wavenumbers[mask]
            new_y = spec.intensities[mask]

            cmd = ModifySpectrumCommand(
                spec, f"Przycięcie [{min_w:.0f}-{max_w:.0f}]",
                spec.intensities, new_y,
                spec.wavenumbers, new_x,
                canvas=self._get_canvas()
            )
            session.undo_stack.push(cmd)
        except ValueError:
            QMessageBox.warning(self, tr("app.warning", "Ostrzeżenie"), "Nieprawidłowe liczby falowe zakresu.")

    def retranslate_ui(self):
        pass