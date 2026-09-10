"""
gui/node_editor/dialogs.py
Dialogi konfiguracyjne węzłów DAG:
- Źródła danych: Węzeł Startowy (Input Source) z wyborem trybu i widma
- Odniesienia i Mux: Multi-Select, Mux / Zbieracz
- Chemometria: SVD/PCA, MCR-ALS / SPEXFA, Skalowania
- Dekompozycja HDO: Widmo Skalowane, Pasmo GxL/Gauss/Lorentz, Parametry MC, Solwer HDO
- Fotometria, DFT, VaporFit, Arytmetyka i Terminale
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QLabel,
    QPushButton, QMessageBox, QCheckBox, QListWidget, QListWidgetItem
)
from core.session import session
from algorithms.atmospheric import AtmosphericCorrectionEngine


class NodeParamDialog(QDialog):
    """Uniwersalny dialog konfiguracji węzłów analitycznych i chemometrycznych."""

    def __init__(self, node_item, parent=None):
        super().__init__(parent)
        self.node = node_item
        self.step = node_item.step_data
        self.setWindowTitle(f"Parametry węzła: {node_item.title}")
        self.resize(470, 420)

        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        stype = self.node.node_type

        # =====================================================================
        # 0. ŹRÓDŁO DANYCH (WĘZEŁ STARTOWY)
        # =====================================================================
        if stype == 'input':
            self.combo_source_mode = QComboBox()
            self.combo_source_mode.addItem("Tylko aktywne widmo (1D)", "active")
            self.combo_source_mode.addItem("Konkretne widmo z projektu (1D)", "specific")
            self.combo_source_mode.addItem("Wszystkie widoczne w oknie / gałęzi (Batch 2D)", "visible")
            self.combo_source_mode.addItem("Aktywna seria kinetyczna / Tensor", "series")

            cur_mode = self.step.get('source_mode', 'active')
            idx_m = self.combo_source_mode.findData(cur_mode)
            if idx_m >= 0:
                self.combo_source_mode.setCurrentIndex(idx_m)
            self.form.addRow("Zakres danych wejściowych:", self.combo_source_mode)

            self.combo_specific_spec = QComboBox()
            all_specs = list(session.spectra.keys())
            for name in all_specs:
                self.combo_specific_spec.addItem(f"📈 {name}", name)
            cur_s = self.step.get('specific_spectrum', '')
            idx_s = self.combo_specific_spec.findData(cur_s)
            if idx_s >= 0:
                self.combo_specific_spec.setCurrentIndex(idx_s)
            self.form.addRow("Wybrane widmo:", self.combo_specific_spec)

            def _toggle_specific(idx):
                mode = self.combo_source_mode.itemData(idx)
                self.combo_specific_spec.setEnabled(mode == "specific")

            self.combo_source_mode.currentIndexChanged.connect(_toggle_specific)
            _toggle_specific(self.combo_source_mode.currentIndex())

            lbl_hint = QLabel(
                "💡 Wybierz 'Tylko aktywne' lub 'Konkretne widmo' do pojedynczych analiz (np. HDO, dekonwolucja).\n"
                "Wybierz 'Wszystkie widoczne' do hurtowego przetwarzania wsadowego (Batch)."
            )
            lbl_hint.setStyleSheet("color: #a8dadc; font-size: 11px;")
            self.form.addRow(lbl_hint)

        # =====================================================================
        # 1. CHEMOMETRIA: ODSZUMIANIE SVD / PCA RANK-K
        # =====================================================================
        elif stype == 'svd_denoise':
            self.spin_k = QSpinBox()
            self.spin_k.setRange(1, 40)
            self.spin_k.setValue(int(self.step.get('k', 2)))
            self.form.addRow("Liczba składowych (Rank k):", self.spin_k)

            self.chk_center = QCheckBox("Centruj do średniej (Klasyczne PCA)")
            self.chk_center.setToolTip("Wyłączone = czyste SVD (rekonstrukcja zachowuje cechy fizycznych pasm absorbancji).")
            self.chk_center.setChecked(bool(self.step.get('center', False)))
            self.form.addRow("Tryb dekompozycji:", self.chk_center)

        # =====================================================================
        # 2. CHEMOMETRIA: DEKOMPOZYCJA MCR-ALS / SPEXFA
        # =====================================================================
        elif stype == 'mcr_decompose':
            self.combo_mcr_method = QComboBox()
            self.combo_mcr_method.addItem("MCR-ALS (Optymalizacja naprzemienna)", "mcr_als")
            self.combo_mcr_method.addItem("SPEXFA (Ekstrakcja kluczowa IKSFA)", "spexfa")
            idx = self.combo_mcr_method.findData(self.step.get('method', 'mcr_als'))
            if idx >= 0:
                self.combo_mcr_method.setCurrentIndex(idx)
            self.form.addRow("Algorytm dekompozycji:", self.combo_mcr_method)

            self.spin_mcr_k = QSpinBox()
            self.spin_mcr_k.setRange(1, 20)
            self.spin_mcr_k.setValue(int(self.step.get('k', 2)))
            self.form.addRow("Liczba czystych składników (k):", self.spin_mcr_k)

            self.chk_nnls = QCheckBox("Wymuś nieujemność widm i stężeń (NNLS)")
            self.chk_nnls.setChecked(bool(self.step.get('non_negative', True)))
            self.form.addRow("Więzy nieujemności:", self.chk_nnls)

            self.chk_closure = QCheckBox("Domknięcie bilansu (Suma stężeń = 1.0)")
            self.chk_closure.setChecked(bool(self.step.get('closure', False)))
            self.form.addRow("Więzy stężeń:", self.chk_closure)

            self.chk_unimodal = QCheckBox("Jednomodalność profili (Unimodal)")
            self.chk_unimodal.setChecked(bool(self.step.get('unimodal', False)))
            self.form.addRow("Kształt profilu:", self.chk_unimodal)

            self.spin_max_iter = QSpinBox()
            self.spin_max_iter.setRange(20, 2000)
            self.spin_max_iter.setValue(int(self.step.get('max_iter', 500)))
            self.form.addRow("Maksymalna liczba iteracji:", self.spin_max_iter)

        # =====================================================================
        # 3. CHEMOMETRIA: SKALOWANIA POPULACYJNE
        # =====================================================================
        elif stype == 'matrix_scaling':
            self.combo_scale = QComboBox()
            self.combo_scale.addItem("Centrowanie do średniej (Mean-center)", "mean_center")
            self.combo_scale.addItem("Autoskalowanie (Unit Variance)", "autoscaling")
            self.combo_scale.addItem("Skalowanie Pareto", "pareto")
            self.combo_scale.addItem("Korekcja rozproszenia (MSC)", "msc")
            idx = self.combo_scale.findData(self.step.get('scale_type', 'mean_center'))
            if idx >= 0:
                self.combo_scale.setCurrentIndex(idx)
            self.form.addRow("Transformacja macierzowa:", self.combo_scale)

        # =====================================================================
        # 4. DEKOMPOZYCJA HDO: WIDMO SKŁADOWE ZE SKALAREM I WAGAMI
        # =====================================================================
        elif stype == 'hdo_spectrum_scaled':
            self.combo_hdo_spec = QComboBox()
            all_specs = list(session.spectra.keys())
            for name in all_specs:
                self.combo_hdo_spec.addItem(f"📈 {name}", name)
            cur_s = self.step.get('selected_spectrum', '')
            idx = self.combo_hdo_spec.findData(cur_s)
            if idx >= 0:
                self.combo_hdo_spec.setCurrentIndex(idx)
            self.form.addRow("Widmo z projektu:", self.combo_hdo_spec)

            self.spin_scale = QDoubleSpinBox()
            self.spin_scale.setRange(-1000.0, 1000.0)
            self.spin_scale.setDecimals(4)
            self.spin_scale.setValue(float(self.step.get('scale', 1.0)))
            self.form.addRow("Wstępny mnożnik skali (c):", self.spin_scale)

            self.spin_w_min = QDoubleSpinBox()
            self.spin_w_min.setRange(0.0, 50.0)
            self.spin_w_min.setDecimals(3)
            self.spin_w_min.setValue(float(self.step.get('w_min', 0.0)))
            self.form.addRow("Waga minimalna (Min w MC):", self.spin_w_min)

            self.spin_w_max = QDoubleSpinBox()
            self.spin_w_max.setRange(0.0, 50.0)
            self.spin_w_max.setDecimals(3)
            self.spin_w_max.setValue(float(self.step.get('w_max', 1.5)))
            self.form.addRow("Waga maksymalna (Max w MC):", self.spin_w_max)

        # =====================================================================
        # 5. DEKOMPOZYCJA HDO: PASMO SYNTETYCZNE
        # =====================================================================
        elif stype == 'synthetic_peak':
            self.combo_shape = QComboBox()
            self.combo_shape.addItem("Gauss x Lorentz (GxL)", "gxl")
            self.combo_shape.addItem("Gaussian", "gaussian")
            self.combo_shape.addItem("Lorentzian", "lorentzian")
            self.combo_shape.addItem("Voigt", "voigt")
            self.combo_shape.addItem("Pseudo-Voigt", "pseudovoigt")
            cur_shape = self.step.get('shape', 'gxl')
            idx = self.combo_shape.findData(cur_shape)
            if idx >= 0:
                self.combo_shape.setCurrentIndex(idx)
            self.form.addRow("Kształt pasma:", self.combo_shape)

            self.spin_center = QDoubleSpinBox()
            self.spin_center.setRange(100.0, 5000.0)
            self.spin_center.setValue(float(self.step.get('center', 1650.0)))
            self.form.addRow("Położenie ν₀ [cm⁻¹]:", self.spin_center)

            self.spin_amp = QDoubleSpinBox()
            self.spin_amp.setRange(0.0, 100.0)
            self.spin_amp.setDecimals(4)
            self.spin_amp.setValue(float(self.step.get('amp', 0.5)))
            self.form.addRow("Amplituda / Wysokość:", self.spin_amp)

            self.spin_width = QDoubleSpinBox()
            self.spin_width.setRange(0.5, 500.0)
            self.spin_width.setValue(float(self.step.get('width', 30.0)))
            self.form.addRow("Szerokość (FWHM/Sigma):", self.spin_width)

            self.spin_fraction = QDoubleSpinBox()
            self.spin_fraction.setRange(0.0, 1.0)
            self.spin_fraction.setSingleStep(0.05)
            self.spin_fraction.setValue(float(self.step.get('shape_param', 0.5)))
            self.form.addRow("Wsp. kształtu (0=G, 1=L):", self.spin_fraction)

        # =====================================================================
        # 6. DEKOMPOZYCJA HDO: PARAMETRY MONTE CARLO
        # =====================================================================
        elif stype == 'mc_params':
            self.spin_mc_iter = QSpinBox()
            self.spin_mc_iter.setRange(100, 50000)
            self.spin_mc_iter.setSingleStep(500)
            self.spin_mc_iter.setValue(int(self.step.get('n_iter', 3000)))
            self.form.addRow("Liczba wariantów MC:", self.spin_mc_iter)

            self.spin_mc_pen = QDoubleSpinBox()
            self.spin_mc_pen.setRange(0.0, 500.0)
            self.spin_mc_pen.setValue(float(self.step.get('penalty_weight', 10.0)))
            self.form.addRow("Współczynnik asymetrycznej kary:", self.spin_mc_pen)

            self.chk_bl = QCheckBox("Linia bazowa (kwadratowa ax² + bx + c)")
            self.chk_bl.setChecked(bool(self.step.get('baseline_active', False)))
            self.form.addRow("Linia bazowa:", self.chk_bl)

        # =====================================================================
        # 7. DEKOMPOZYCJA HDO: GŁÓWNY WĘZEŁ SOLWERA
        # =====================================================================
        elif stype == 'hdo_decomposition':
            self.spin_mult_n = QDoubleSpinBox()
            self.spin_mult_n.setRange(0.01, 100.0)
            self.spin_mult_n.setValue(float(self.step.get('n_multiplier', 1.0)))
            self.form.addRow("Mnożnik widma badanego N:", self.spin_mult_n)

            lbl_info = QLabel(
                "🎲 Parametry optymalizacji Monte Carlo (liczba prób, kara, linia bazowa)\n"
                "są pobierane z podłączonego kablem węzła 'Parametry MC'.\n\n"
                "• W₀: Widmo badane\n"
                "• Wₖ: Widma składowe z wagami (akceptuje wiele kabli)\n"
                "• Pik: Piki syntetyczne GxL (akceptuje wiele kabli)\n"
                "• MC: Węzeł parametrów Monte Carlo"
            )
            lbl_info.setStyleSheet("color: #a8dadc; font-size: 11px; padding: 6px;")
            self.form.addRow(lbl_info)

        # =====================================================================
        # 8. TERMINAL: GENERUJ W NOWYM OKNIE ROBOCZYM
        # =====================================================================
        elif stype == 'output_window':
            self.ent_win_name = QLineEdit(self.step.get('window_name', 'Skorygowane'))
            self.form.addRow("Nazwa docelowego okna:", self.ent_win_name)

            self.combo_mode = QComboBox()
            self.combo_mode.addItem("Utwórz nowe okno robocze", "create_new")
            self.combo_mode.addItem("Dodaj do istniejącego (o tej nazwie)", "append_existing")
            idx = self.combo_mode.findData(self.step.get('mode', 'create_new'))
            if idx >= 0:
                self.combo_mode.setCurrentIndex(idx)
            self.form.addRow("Tryb przypisania:", self.combo_mode)

            lbl_info = QLabel("💡 Port wejściowy 'In' akceptuje kable z wielu gałęzi jednocześnie.")
            lbl_info.setStyleSheet("color: #a8dadc; font-size: 11px;")
            self.form.addRow("", lbl_info)

        # =====================================================================
        # 9. PRZELICZENIA FOTOMETRYCZNE
        # =====================================================================
        elif stype == 'photometry':
            self.combo_photo = QComboBox()
            self.combo_photo.addItem("Absorbancja ➔ Transmisja [T]", "a_to_t")
            self.combo_photo.addItem("Transmisja [T] ➔ Absorbancja", "t_to_a")
            self.combo_photo.addItem("%Transmisji [%T] ➔ Absorbancja", "tpct_to_a")
            self.combo_photo.addItem("Absorbancja ➔ %Transmisji [%T]", "a_to_tpct")
            self.combo_photo.addItem("Kubelka-Munk DRIFTS f(R∞)", "kubelka_munk")
            self.combo_photo.addItem("Odwróć Kubelka-Munk ➔ R∞", "inv_kubelka_munk")

            cur_op = self.step.get('photo_mode', 'tpct_to_a')
            idx = self.combo_photo.findData(cur_op)
            if idx >= 0:
                self.combo_photo.setCurrentIndex(idx)
            self.form.addRow("Transformacja fotometryczna:", self.combo_photo)

        # =====================================================================
        # 10. SPLOT PASM DFT
        # =====================================================================
        elif stype == 'dft_broadener':
            self.combo_dft_spec = QComboBox()
            for name, spec in session.spectra.items():
                if hasattr(spec, 'stick_freqs') and len(getattr(spec, 'stick_freqs', [])) > 0:
                    self.combo_dft_spec.addItem(f"⚛️ {name}", name)

            if self.combo_dft_spec.count() == 0:
                self.combo_dft_spec.addItem("(Brak obliczeń DFT w sesji)", "")

            sel_dft = self.step.get('dft_name', '')
            idx = self.combo_dft_spec.findData(sel_dft)
            if idx >= 0:
                self.combo_dft_spec.setCurrentIndex(idx)
            self.form.addRow("Obliczenie DFT:", self.combo_dft_spec)

            self.spin_scale = QDoubleSpinBox()
            self.spin_scale.setRange(0.5, 1.5)
            self.spin_scale.setDecimals(4)
            self.spin_scale.setSingleStep(0.005)
            self.spin_scale.setValue(float(self.step.get('scale_factor', 0.965)))
            self.form.addRow("Czynnik skalujący częstości (σ):", self.spin_scale)

            self.spin_fwhm = QDoubleSpinBox()
            self.spin_fwhm.setRange(1.0, 100.0)
            self.spin_fwhm.setValue(float(self.step.get('fwhm', 12.0)))
            self.form.addRow("Szerokość połówkowa FWHM [cm⁻¹]:", self.spin_fwhm)

            self.combo_shape = QComboBox()
            self.combo_shape.addItem("Lorentz (domyślny FTIR)", "lorentzian")
            self.combo_shape.addItem("Gauss", "gaussian")
            idx = self.combo_shape.findData(self.step.get('shape', 'lorentzian'))
            if idx >= 0:
                self.combo_shape.setCurrentIndex(idx)
            self.form.addRow("Profil kształtu pasma:", self.combo_shape)

        # =====================================================================
        # 11. OPERACJE DWUWIDMOWE
        # =====================================================================
        elif stype == 'math_op':
            self.combo_op = QComboBox()
            self.combo_op.addItem("Odejmowanie: A - k · B", "sub")
            self.combo_op.addItem("Dodawanie: A + k · B", "add")
            self.combo_op.addItem("Mnożenie: A · B", "mul")
            self.combo_op.addItem("Dzielenie: A / B", "div")
            cur_op = self.step.get('op', 'sub')
            idx = self.combo_op.findData(cur_op)
            if idx >= 0:
                self.combo_op.setCurrentIndex(idx)

            self.spin_k = QDoubleSpinBox()
            self.spin_k.setRange(-1000.0, 1000.0)
            self.spin_k.setDecimals(4)
            self.spin_k.setSingleStep(0.01)
            self.spin_k.setValue(float(self.step.get('k', 1.0)))

            self.form.addRow("Typ operacji:", self.combo_op)
            self.form.addRow("Współczynnik skali k:", self.spin_k)

        # =====================================================================
        # 12. SKALARY
        # =====================================================================
        elif stype == 'scalar_math':
            self.combo_op = QComboBox()
            self.combo_op.addItem("Mnożenie: S · c", "mul")
            self.combo_op.addItem("Dzielenie: S / c", "div")
            self.combo_op.addItem("Dodawanie: S + c", "add")
            self.combo_op.addItem("Odejmowanie: S - c", "sub")
            cur_op = self.step.get('op', 'mul')
            idx = self.combo_op.findData(cur_op)
            if idx >= 0:
                self.combo_op.setCurrentIndex(idx)

            self.spin_val = QDoubleSpinBox()
            self.spin_val.setRange(-1e6, 1e6)
            self.spin_val.setDecimals(4)
            self.spin_val.setSingleStep(0.1)
            self.spin_val.setValue(float(self.step.get('val', 1.0)))

            self.form.addRow("Działanie:", self.combo_op)
            self.form.addRow("Wartość skalara (c):", self.spin_val)

        # =====================================================================
        # 13. VAPORFIT
        # =====================================================================
        elif stype == 'vaporfit':
            self.spin_sg = QSpinBox()
            self.spin_sg.setRange(5, 101)
            self.spin_sg.setSingleStep(2)
            self.spin_sg.setValue(int(self.step.get('sg_points', 11)))

            self.spin_poly = QSpinBox()
            self.spin_poly.setRange(1, 6)
            self.spin_poly.setValue(int(self.step.get('sg_poly', 3)))

            self.form.addRow("Szerokość okna SG (pkt):", self.spin_sg)
            self.form.addRow("Stopień wielomianu:", self.spin_poly)

            self.btn_opt_sg = QPushButton("🔍 Szukaj optymalnego okna SG")
            self.btn_opt_sg.clicked.connect(self._on_find_optimal_sg)
            self.form.addRow("", self.btn_opt_sg)

            metrics = self.step.get('metrics', {})
            if metrics:
                lbl_met = QLabel(
                    f"SDV: {metrics.get('sdv_raw', 0):.2e} ➔ {metrics.get('sdv_corr', 0):.2e}\n"
                    f"SSI: {metrics.get('ssi_raw', 0):.2e} ➔ {metrics.get('ssi_corr', 0):.2e}"
                )
                lbl_met.setStyleSheet("color: #a8dadc; font-size: 11px;")
                self.form.addRow("Metryki gładkości:", lbl_met)

        # =====================================================================
        # 14. WZORCE ODNIESIENIA (MULTI-SELECT Z PROJEKTU)
        # =====================================================================
        elif stype == 'spectrum_select':
            self.list_specs = QListWidget()
            self.list_specs.setFixedHeight(140)

            current_selected = self.step.get('selected_spectra', [])
            if not current_selected and self.step.get('selected_spectrum'):
                current_selected = [self.step.get('selected_spectrum')]

            if session.atm_spectra:
                sep_atm = QListWidgetItem("── WZORCE ATMOSFERY (ATM / CO₂) ──")
                sep_atm.setFlags(Qt.NoItemFlags)
                self.list_specs.addItem(sep_atm)
                for name in session.atm_spectra.keys():
                    item = QListWidgetItem(f"☁️ {name}")
                    item.setData(Qt.UserRole, name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if name in current_selected else Qt.Unchecked)
                    self.list_specs.addItem(item)

            if session.spectra:
                sep_exp = QListWidgetItem("── POMIARY EKSPERYMENTALNE ──")
                sep_exp.setFlags(Qt.NoItemFlags)
                self.list_specs.addItem(sep_exp)
                for name in session.spectra.keys():
                    item = QListWidgetItem(f"📈 {name}")
                    item.setData(Qt.UserRole, name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if name in current_selected else Qt.Unchecked)
                    self.list_specs.addItem(item)

            if self.list_specs.count() == 0:
                empty_item = QListWidgetItem("(Brak widm w projekcie)")
                empty_item.setFlags(Qt.NoItemFlags)
                self.list_specs.addItem(empty_item)

            row_btns = QHBoxLayout()
            btn_sel_atm = QPushButton("Zaznacz ATM")
            btn_sel_atm.clicked.connect(self._select_all_atm_specs)
            btn_clear = QPushButton("Odznacz")
            btn_clear.clicked.connect(self._clear_all_specs)
            row_btns.addWidget(btn_sel_atm)
            row_btns.addWidget(btn_clear)

            self.form.addRow("Wybierz widma z projektu:", self.list_specs)
            self.form.addRow("", row_btns)

        # =====================================================================
        # 15. WEWNĘTRZNY STANDARD (IS)
        # =====================================================================
        elif stype == 'internal_standard':
            self.spin_target_wn = QDoubleSpinBox()
            self.spin_target_wn.setRange(100.0, 5000.0)
            self.spin_target_wn.setValue(float(self.step.get('target_wn', 2060.0)))
            self.form.addRow("Położenie pasma IS [cm⁻¹]:", self.spin_target_wn)

            self.spin_half_win = QDoubleSpinBox()
            self.spin_half_win.setRange(1.0, 100.0)
            self.spin_half_win.setValue(float(self.step.get('half_win', 15.0)))
            self.form.addRow("Tolerancja szukania (± cm⁻¹):", self.spin_half_win)

            self.combo_is_mode = QComboBox()
            self.combo_is_mode.addItem("Maksimum pasma (Wysokość piku)", "peak_height")
            self.combo_is_mode.addItem("Pole powierzchni (Całka pod pasmem)", "peak_area")
            idx = self.combo_is_mode.findData(self.step.get('calc_mode', 'peak_height'))
            if idx >= 0:
                self.combo_is_mode.setCurrentIndex(idx)
            self.form.addRow("Metoda wyznaczenia IS:", self.combo_is_mode)

        # =====================================================================
        # 16. CAŁKOWANIE PASMA / KINETYKA
        # =====================================================================
        elif stype == 'band_integrate':
            self.spin_int_min = QDoubleSpinBox()
            self.spin_int_min.setRange(100.0, 5000.0)
            self.spin_int_min.setValue(float(self.step.get('min_wn', 1600.0)))
            self.form.addRow("Dolna granica [cm⁻¹]:", self.spin_int_min)

            self.spin_int_max = QDoubleSpinBox()
            self.spin_int_max.setRange(100.0, 5000.0)
            self.spin_int_max.setValue(float(self.step.get('max_wn', 1700.0)))
            self.form.addRow("Górna granica [cm⁻¹]:", self.spin_int_max)

            self.combo_int_bl = QComboBox()
            self.combo_int_bl.addItem("Lokalna linia bazowa (dwa krańce)", "two_point")
            self.combo_int_bl.addItem("Całka od zera (bez korekcji)", "zero")
            self.combo_int_bl.addItem("Offset do minimum w rejonie", "min_offset")
            idx = self.combo_int_bl.findData(self.step.get('bl_mode', 'two_point'))
            if idx >= 0:
                self.combo_int_bl.setCurrentIndex(idx)
            self.form.addRow("Korekcja tła pasma:", self.combo_int_bl)

        # =====================================================================
        # 17. ZBIERACZ / MUX (MERGE SPECTRA)
        # =====================================================================
        elif stype == 'merge_spectra':
            lbl = QLabel(
                "Węzeł Zbieracza (Mux) łączy sygnały z wielu gałęzi i nodów\n"
                "w jedną wspólną macierz dwuwymiarową (W x N).\n\n"
                "Podłącz kable z dowolnych węzłów pod port wejściowy 'In'."
            )
            lbl.setStyleSheet("color: #a8dadc; font-size: 12px; padding: 10px;")
            self.form.addRow(lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _select_all_atm_specs(self):
        for i in range(self.list_specs.count()):
            item = self.list_specs.item(i)
            name = item.data(Qt.UserRole)
            if name and name in session.atm_spectra:
                item.setCheckState(Qt.Checked)

    def _clear_all_specs(self):
        for i in range(self.list_specs.count()):
            item = self.list_specs.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)

    def _on_find_optimal_sg(self):
        p_sig = self.node.in_ports.get("in_signal")
        p_atm = self.node.in_ports.get("in_atm")

        src_sig = p_sig.connections[0].out_port.node_item if (p_sig and p_sig.connections) else None
        atm_sources = [c.out_port.node_item for c in p_atm.connections if c.out_port and c.out_port.node_item] if p_atm else []

        if not src_sig or src_sig.preview_x is None or not atm_sources:
            QMessageBox.warning(self, "Uwaga", "Podłącz najpierw widmo badane (Syg) i co najmniej jedno widmo atmosfery (Atm).")
            return

        xs, ys = src_sig.preview_x, src_sig.preview_y
        if ys.ndim == 2:
            ys = ys[:, 0]

        atm_cols = []
        for s in atm_sources:
            if s.preview_x is not None and s.preview_y is not None:
                xa, ya = s.preview_x, s.preview_y
                if xa[0] > xa[-1]:
                    xa = xa[::-1]
                    ya = ya[::-1] if ya.ndim == 1 else ya[::-1, :]
                if ya.ndim == 2:
                    for col in range(ya.shape[1]):
                        atm_cols.append(np.interp(xs, xa, ya[:, col]))
                else:
                    atm_cols.append(np.interp(xs, xa, ya))

        if not atm_cols and len(session.atm_spectra) > 0:
            atm_matrix = session.get_atm_matrix(xs, only_visible=True)
            if atm_matrix is None:
                atm_matrix = session.get_atm_matrix(xs, only_visible=False)
        elif atm_cols:
            atm_matrix = np.column_stack(atm_cols)
        else:
            atm_matrix = None

        if atm_matrix is None:
            QMessageBox.warning(self, "Uwaga", "Brak wzorców atmosfery w projekcie lub na wejściu węzła.")
            return

        poly = int(self.spin_poly.value())
        res = AtmosphericCorrectionEngine.find_optimal_sg_window(xs, ys, atm_matrix, sg_poly=poly)
        opt_win = res['best_window_sdv']
        self.spin_sg.setValue(opt_win)

        QMessageBox.information(
            self, "Optymalizacja SG",
            f"Wyznaczono optymalne okno SG:\n"
            f"• Wg krzywizny SDV: {res['best_window_sdv']} pkt\n"
            f"• Wg wskaźnika SSI: {res['best_window_ssi']} pkt\n\n"
            f"Wartość {opt_win} pkt została wprowadzona."
        )

    def _save_and_accept(self):
        stype = self.node.node_type

        if stype == 'input':
            mode = self.combo_source_mode.currentData()
            self.step['source_mode'] = mode
            if mode == 'specific':
                spec_name = self.combo_specific_spec.currentData() or ""
                self.step['specific_spectrum'] = spec_name
                short_n = spec_name[:16] + "…" if len(spec_name) > 16 else spec_name
                self.step['name'] = f"Źródło: {short_n}"
            elif mode == 'visible':
                self.step['name'] = "Źródło: Wszystkie widoczne"
            elif mode == 'series':
                self.step['name'] = "Źródło: Aktywna seria"
            else:
                self.step['name'] = "Źródło: Aktywne widmo"

        elif stype == 'svd_denoise':
            k = int(self.spin_k.value())
            center = bool(self.chk_center.isChecked())
            self.step['k'] = k
            self.step['center'] = center
            tag = "PCA" if center else "SVD"
            self.step['name'] = f"{tag} Denoise (k={k})"

        elif stype == 'mcr_decompose':
            method = self.combo_mcr_method.currentData()
            k = int(self.spin_mcr_k.value())
            self.step['method'] = method
            self.step['k'] = k
            self.step['non_negative'] = bool(self.chk_nnls.isChecked())
            self.step['closure'] = bool(self.chk_closure.isChecked())
            self.step['unimodal'] = bool(self.chk_unimodal.isChecked())
            self.step['max_iter'] = int(self.spin_max_iter.value())
            m_label = "MCR-ALS" if method == "mcr_als" else "SPEXFA"
            self.step['name'] = f"{m_label} (k={k})"

        elif stype == 'matrix_scaling':
            st = self.combo_scale.currentData()
            self.step['scale_type'] = st
            names = {
                'mean_center': "Mean-center",
                'autoscaling': "Autoscaling (UV)",
                'pareto': "Skalowanie Pareto",
                'msc': "Korekcja MSC"
            }
            self.step['name'] = f"Skalowanie: {names.get(st, st)}"

        elif stype == 'hdo_spectrum_scaled':
            sel_s = self.combo_hdo_spec.currentData() or ""
            scale = float(self.spin_scale.value())
            w_min = float(self.spin_w_min.value())
            w_max = float(self.spin_w_max.value())
            self.step['selected_spectrum'] = sel_s
            self.step['scale'] = scale
            self.step['w_min'] = min(w_min, w_max)
            self.step['w_max'] = max(w_min, w_max)
            short_n = sel_s[:16] + "…" if len(sel_s) > 16 else sel_s
            self.step['name'] = f"{short_n} · {scale:.2f}" if sel_s else "Widmo Skalowane"

        elif stype == 'synthetic_peak':
            sh = self.combo_shape.currentData()
            c = float(self.spin_center.value())
            a = float(self.spin_amp.value())
            w = float(self.spin_width.value())
            f = float(self.spin_fraction.value())
            self.step['shape'] = sh
            self.step['center'] = c
            self.step['amp'] = a
            self.step['width'] = w
            self.step['shape_param'] = f
            self.step['name'] = f"Pik {sh.upper()} @ {c:.1f}"

        elif stype == 'mc_params':
            it = int(self.spin_mc_iter.value())
            pen = float(self.spin_mc_pen.value())
            bl = bool(self.chk_bl.isChecked())
            self.step['n_iter'] = it
            self.step['penalty_weight'] = pen
            self.step['baseline_active'] = bl
            self.step['name'] = f"MC Params (iter={it})"

        elif stype == 'hdo_decomposition':
            n_mult = float(self.spin_mult_n.value())
            self.step['n_multiplier'] = n_mult
            self.step['name'] = f"HDO Fit (N={n_mult:.2f})"

        elif stype == 'output_window':
            wname = self.ent_win_name.text().strip() or "Skorygowane"
            self.step['window_name'] = wname
            self.step['mode'] = self.combo_mode.currentData()
            self.step['name'] = f"🗂️ Okno: {wname}"

        elif stype == 'photometry':
            mode = self.combo_photo.currentData()
            self.step['photo_mode'] = mode
            names = {
                'a_to_t': "A ➔ T",
                't_to_a': "T ➔ A",
                'tpct_to_a': "%T ➔ A",
                'a_to_tpct': "A ➔ %T",
                'kubelka_munk': "Kubelka-Munk f(R∞)",
                'inv_kubelka_munk': "f(R∞) ➔ R∞"
            }
            self.step['name'] = f"Fotometria: {names.get(mode, mode)}"

        elif stype == 'dft_broadener':
            sel_name = self.combo_dft_spec.currentData()
            scale = float(self.spin_scale.value())
            fwhm = float(self.spin_fwhm.value())
            shape = self.combo_shape.currentData()

            self.step['dft_name'] = sel_name
            self.step['scale_factor'] = scale
            self.step['fwhm'] = fwhm
            self.step['shape'] = shape
            self.step['name'] = f"DFT: {sel_name} (σ={scale:.3f})" if sel_name else "DFT Broadener"

        elif stype == 'math_op':
            op = self.combo_op.currentData()
            k = float(self.spin_k.value())
            self.step['op'] = op
            self.step['k'] = k
            op_sym = "-" if op == "sub" else ("+" if op == "add" else ("·" if op == "mul" else "/"))
            self.step['name'] = f"A {op_sym} {k:.3f}·B" if op in ('sub', 'add') else f"A {op_sym} B"

        elif stype == 'scalar_math':
            op = self.combo_op.currentData()
            val = float(self.spin_val.value())
            self.step['op'] = op
            self.step['val'] = val
            op_sym = "·" if op == "mul" else ("/" if op == "div" else ("+" if op == "add" else "-"))
            self.step['name'] = f"S {op_sym} {val:.4g}"

        elif stype == 'vaporfit':
            self.step['sg_points'] = int(self.spin_sg.value())
            self.step['sg_poly'] = int(self.spin_poly.value())
            self.step['name'] = f"VaporFit (SG={self.step['sg_points']})"

        elif stype == 'spectrum_select':
            selected = []
            for i in range(self.list_specs.count()):
                item = self.list_specs.item(i)
                if (item.flags() & Qt.ItemIsUserCheckable) and item.checkState() == Qt.Checked:
                    val = item.data(Qt.UserRole)
                    if val:
                        selected.append(val)

            self.step['selected_spectra'] = selected
            if selected:
                self.step['selected_spectrum'] = selected[0]
                if len(selected) == 1:
                    icon = "☁️" if selected[0] in session.atm_spectra else "📈"
                    self.step['name'] = f"{icon} {selected[0]}"
                else:
                    self.step['name'] = f"📌 {len(selected)} widm"
            else:
                self.step['selected_spectrum'] = ""
                self.step['name'] = "Ref: (brak)"

        elif stype == 'internal_standard':
            wn = float(self.spin_target_wn.value())
            hw = float(self.spin_half_win.value())
            m = self.combo_is_mode.currentData()
            self.step['target_wn'] = wn
            self.step['half_win'] = hw
            self.step['calc_mode'] = m
            self.step['name'] = f"IS @ {wn:.1f} cm⁻¹"

        elif stype == 'band_integrate':
            w1 = float(self.spin_int_min.value())
            w2 = float(self.spin_int_max.value())
            bl = self.combo_int_bl.currentData()
            self.step['min_wn'] = min(w1, w2)
            self.step['max_wn'] = max(w1, w2)
            self.step['bl_mode'] = bl
            self.step['name'] = f"Całka [{self.step['min_wn']:.0f}–{self.step['max_wn']:.0f}]"

        elif stype == 'merge_spectra':
            self.step['name'] = "Mux: Zbieracz widm"

        self.accept()