"""
gui/workspaces/ws_chemometrics.py
Zintegrowany panel perspektywy 3: Chemometria, Analiza Ewolucyjna i Klasyfikacja.
Obsługuje: MCR-ALS, SPEXFA, SVD/PCA, PFA, Klasyfikację 2D, PARAFAC, AUTOWFA, FSMWEFA, HELP i 2D-COS.
Pełna integracja z Magazynem Danych, Oknami Roboczymi oraz Seriami 2D/3D.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QMessageBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from core.session import session
from core.models import SpectrumData, SpectralSeries, GaussianData
from core.i18n import tr, i18n
from algorithms.chemometrics import multivariate as chem

logger = logging.getLogger("SpectroHub")


class ChemometricsPanel(QWidget):
    """Widok inspektora dla perspektywy Chemometria."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_results: Optional[Dict[str, Any]] = None
        self.active_wn: Optional[np.ndarray] = None
        self.active_matrix: Optional[np.ndarray] = None
        self.active_tensor_3d: Optional[np.ndarray] = None
        self.sample_labels: list = []
        self.active_source_name: str = "Dane"

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
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. Źródło danych
        grp_source = QGroupBox("1. Źródło Danych do Analizy")
        self.grp_source = grp_source
        l_src = QVBoxLayout(grp_source)

        row_src = QHBoxLayout()
        self.lbl_source = QLabel("Wybierz zestaw:")
        self.combo_source = QComboBox()
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        row_src.addWidget(self.lbl_source)
        row_src.addWidget(self.combo_source, 1)
        l_src.addLayout(row_src)

        self.lbl_matrix_dims = QLabel("Wymiary: —")
        self.lbl_matrix_dims.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        l_src.addWidget(self.lbl_matrix_dims)
        layout.addWidget(grp_source)

        # 2. Wybór metody
        grp_method = QGroupBox("2. Metoda Chemometryczna")
        self.grp_method = grp_method
        l_meth = QVBoxLayout(grp_method)

        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "MCR-ALS (Czyste składniki & profile)",
            "SPEXFA (Izolacja widm i stężeń)",
            "SVD / PCA (Rozkład składowych & Rekonstrukcja)",
            "PFA / Malinowski (Analiza rang)",
            "Klasyfikacja 2D (Rzutowanie próbek)",
            "PARAFAC (Dekompozycja tensora 3D)",
            "AUTOWFA (Analiza z oknem)",
            "FSMWEFA (Rząd w oknie przesuwnym)",
            "HELP (Lokalny profil rzędu)",
            "2D-COS (Spektroskopia korelacyjna)"
        ])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        l_meth.addWidget(self.combo_method)
        layout.addWidget(grp_method)

        # 3. Parametry algorytmu
        grp_params = QGroupBox("3. Parametry i Opcje")
        self.grp_params = grp_params
        l_params = QVBoxLayout(grp_params)

        # Liczba składowych k
        self.row_k = QHBoxLayout()
        self.lbl_k = QLabel("Liczba składników (k):")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 40)
        self.spin_k.setValue(2)
        self.row_k.addWidget(self.lbl_k)
        self.row_k.addWidget(self.spin_k)
        l_params.addLayout(self.row_k)

        # Przełącznik podwidoku dla SVD / PCA / PFA
        self.row_subview = QHBoxLayout()
        self.lbl_subview = QLabel("Tryb widoku:")
        self.combo_subview = QComboBox()
        self.combo_subview.addItems([
            "1. Czyste składowe (Loadings & Scores)",
            "2. Scree Plot & Diagnostyka (EV / RE / IND)",
            "3. Rekonstrukcja (D_k) & Residua"
        ])
        self.combo_subview.currentIndexChanged.connect(self._on_subview_changed)
        self.row_subview.addWidget(self.lbl_subview)
        self.row_subview.addWidget(self.combo_subview, 1)
        l_params.addLayout(self.row_subview)

        # Opcje SVD vs PCA
        self.chk_pca_center = QCheckBox("Centruj do średniej (Klasyczne PCA)")
        self.chk_pca_center.setToolTip("Wyłączone = czyste SVD na danych surowych (ładunki zachowują cechy widm absorbancji).")
        self.chk_pca_center.setChecked(False)
        l_params.addWidget(self.chk_pca_center)

        # Opcje rzutowania / klasyfikacji 2D
        self.row_proj = QHBoxLayout()
        self.lbl_proj = QLabel("Algorytm rzutowania:")
        self.combo_proj = QComboBox()
        self.combo_proj.addItems(["SVD / PCA (PC1 vs PC2)", "UMAP (Nieliniowe)", "t-SNE (Rozkład t)"])
        self.row_proj.addWidget(self.lbl_proj)
        self.row_proj.addWidget(self.combo_proj, 1)
        l_params.addLayout(self.row_proj)

        # Opcje MCR
        self.chk_nnls = QCheckBox("Wymuś nieujemność (NNLS)")
        self.chk_nnls.setChecked(True)
        self.chk_mcr_spexfa = QCheckBox("Inicjalizacja widm przez SPEXFA")
        self.chk_mcr_spexfa.setChecked(True)
        self.chk_mcr_closure = QCheckBox("Domknięcie bilansu (Closure = 1.0)")
        self.chk_mcr_closure.setChecked(False)
        self.chk_mcr_unimodal = QCheckBox("Jednomodalność stężeń (Unimodal)")
        self.chk_mcr_unimodal.setChecked(False)

        # Parametry zbieżności MCR
        self.row_mcr_iter = QHBoxLayout()
        self.lbl_mcr_iter = QLabel("Maks. iteracji:")
        self.spin_mcr_iter = QSpinBox()
        self.spin_mcr_iter.setRange(20, 2000)
        self.spin_mcr_iter.setValue(500)
        self.row_mcr_iter.addWidget(self.lbl_mcr_iter)
        self.row_mcr_iter.addWidget(self.spin_mcr_iter)

        self.row_mcr_tol = QHBoxLayout()
        self.lbl_mcr_tol = QLabel("Tolerancja błędu:")
        self.spin_mcr_tol = QDoubleSpinBox()
        self.spin_mcr_tol.setDecimals(6)
        self.spin_mcr_tol.setRange(1e-7, 1e-2)
        self.spin_mcr_tol.setValue(1e-5)
        self.row_mcr_tol.addWidget(self.lbl_mcr_tol)
        self.row_mcr_tol.addWidget(self.spin_mcr_tol)

        l_params.addWidget(self.chk_nnls)
        l_params.addWidget(self.chk_mcr_spexfa)
        l_params.addWidget(self.chk_mcr_closure)
        l_params.addWidget(self.chk_mcr_unimodal)
        l_params.addLayout(self.row_mcr_iter)
        l_params.addLayout(self.row_mcr_tol)

        # Opcje FSMWEFA / HELP
        self.row_w = QHBoxLayout()
        self.lbl_window = QLabel("Szerokość okna (H):")
        self.spin_window = QSpinBox()
        self.spin_window.setRange(3, 99)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(7)
        self.row_w.addWidget(self.lbl_window)
        self.row_w.addWidget(self.spin_window)
        l_params.addLayout(self.row_w)

        self.row_noise = QHBoxLayout()
        self.lbl_noise = QLabel("Próg szumu (HELP):")
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setDecimals(4)
        self.spin_noise.setRange(0.0001, 1.0)
        self.spin_noise.setValue(0.05)
        self.row_noise.addWidget(self.lbl_noise)
        self.row_noise.addWidget(self.spin_noise)
        l_params.addLayout(self.row_noise)

        layout.addWidget(grp_params)

        # 4. Przycisk obliczeniowy
        self.btn_run = QPushButton("⚡ Uruchom Analizę")
        self.btn_run.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run_analysis)
        layout.addWidget(self.btn_run)

        # 5. Wyniki & Integracja z Projektem
        grp_results = QGroupBox("4. Wyniki & Integracja z Projektem")
        self.grp_results = grp_results
        l_res = QVBoxLayout(grp_results)

        self.lbl_res_summary = QLabel("Brak obliczonych wyników.")
        self.lbl_res_summary.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.lbl_res_summary.setWordWrap(True)
        l_res.addWidget(self.lbl_res_summary)

        self.btn_inject_spectra = QPushButton("📥 Dodaj Wyizolowane Widma do Projektu")
        self.btn_inject_spectra.setStyleSheet("background-color: #2a9d8f; font-weight: bold;")
        self.btn_inject_spectra.setEnabled(False)
        self.btn_inject_spectra.clicked.connect(self._on_inject_spectra_to_session)
        l_res.addWidget(self.btn_inject_spectra)

        self.btn_inject_recon = QPushButton("📥 Dodaj Odszumioną Serię (D_k) do Projektu")
        self.btn_inject_recon.setStyleSheet("background-color: #457b9d; font-weight: bold;")
        self.btn_inject_recon.setEnabled(False)
        self.btn_inject_recon.clicked.connect(self._on_inject_reconstruction_to_session)
        l_res.addWidget(self.btn_inject_recon)

        layout.addWidget(grp_results)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        self._update_method_ui()
        self.retranslate_ui()

    def _connect_signals(self):
        # Sygnały Magazynu i Serii
        session.series_added.connect(self._refresh_sources)
        session.series_removed.connect(self._refresh_sources)
        session.spectrum_added.connect(self._refresh_sources)
        session.spectra_batch_added.connect(self._refresh_sources)
        session.spectrum_removed.connect(self._refresh_sources)
        session.visibility_changed.connect(self._refresh_sources)

        # Sygnały Okien Roboczych
        session.window_added.connect(self._refresh_sources)
        session.window_removed.connect(self._refresh_sources)
        session.window_spectrum_added.connect(self._refresh_sources)
        session.window_spectrum_removed.connect(self._refresh_sources)
        session.window_visibility_changed.connect(self._refresh_sources)
        session.active_window_changed.connect(self._refresh_sources)
        session.context_changed.connect(self._refresh_sources)

        # Sygnały mutacji RAM
        session.spectrum_data_changed.connect(lambda _: self._on_source_changed())
        session.batch_spectra_changed.connect(lambda _: self._on_source_changed())
        session.series_data_changed.connect(lambda _: self._on_source_changed())

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_sources()

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    # =========================================================================
    # INTELIGENTNY DYSPOZYTOR ŹRÓDEŁ DANYCH
    # =========================================================================

    def _refresh_sources(self):
        """Przebudowuje listę combo_source, dynamicznie dobierając domyślny kontekst."""
        prev_data = self.combo_source.currentData()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()

        # 1. POZYCJA DOMYŚLNA (Zależna od aktywnego kontekstu sesji)
        is_window_ctx = (session.active_context == "window" and session.active_window is not None)
        if is_window_ctx:
            win = session.active_window
            vis_in_win = len([s for s in win.spectra.values() if win.visibility.get(s.name, True)])
            def_text = f"🎯 [DOMYŚLNE] Aktywne okno: {win.name} ({vis_in_win} widm)"
            def_payload = {"type": "active_display", "label": win.name}
        else:
            vis_in_mag = len(session.get_visible_spectra())
            def_text = f"🎯 [DOMYŚLNE] Magazyn: Widoczne pomiary ({vis_in_mag} widm)"
            def_payload = {"type": "active_display", "label": "Magazyn"}

        self.combo_source.addItem(def_text, def_payload)

        # 2. OKNA ROBOCZE (WORKSPACES)
        for win_id, win in session.windows.items():
            vis_cnt = len([s for s in win.spectra.values() if win.visibility.get(s.name, True)])
            tot_cnt = len(win.spectra)
            if tot_cnt > 0:
                self.combo_source.addItem(
                    f"🖥️ Okno: {win.name} (widoczne: {vis_cnt}/{tot_cnt})",
                    {"type": "window", "window_id": win_id, "only_visible": True, "label": win.name}
                )
                if vis_cnt != tot_cnt:
                    self.combo_source.addItem(
                        f"🖥️ Okno: {win.name} (wszystkie {tot_cnt} widm)",
                        {"type": "window", "window_id": win_id, "only_visible": False, "label": f"{win.name}_all"}
                    )

        # 3. MAGAZYN DANYCH (SEKCJE REFERENCYJNE)
        all_specs = list(session.spectra.values())
        exp_specs = [s for s in all_specs if not isinstance(s, GaussianData) and getattr(s, 'file_type', '') not in ['gaussian', 'orca', 'dft']]
        vis_specs = session.get_visible_spectra()

        if exp_specs:
            self.combo_source.addItem(
                f"📦 Magazyn: Wszystkie pomiary eksperymentalne ({len(exp_specs)})",
                {"type": "storage_exp", "label": "Magazyn_FTIR"}
            )
        if vis_specs:
            self.combo_source.addItem(
                f"📦 Magazyn: Aktualnie zaznaczone (widoczne: {len(vis_specs)})",
                {"type": "storage_visible", "label": "Magazyn_Widoczne"}
            )

        # 4. SERIE 2D ORAZ TENSORY 3D
        if hasattr(session, 'series'):
            for s_name, s_obj in session.series.items():
                icon = "🧊" if s_obj.is_3d else "📊"
                tag = "Tensor 3D" if s_obj.is_3d else "Seria 2D"
                n_specs = s_obj.n_spectra
                self.combo_source.addItem(
                    f"{icon} {tag}: {s_name} ({n_specs} widm)",
                    {"type": "series", "series_name": s_name, "label": s_name}
                )

        # Przywracanie zaznaczenia, jeśli nadal istnieje w opcjach
        restored = False
        if prev_data:
            for idx in range(self.combo_source.count()):
                d = self.combo_source.itemData(idx)
                if d and d.get('type') == prev_data.get('type') and d.get('window_id') == prev_data.get('window_id') and d.get('series_name') == prev_data.get('series_name'):
                    self.combo_source.setCurrentIndex(idx)
                    restored = True
                    break

        if not restored:
            self.combo_source.setCurrentIndex(0)

        self.combo_source.blockSignals(False)
        self._on_source_changed()

    def _on_source_changed(self):
        data = self.combo_source.currentData()
        self.active_tensor_3d = None
        self.active_matrix = None
        self.active_wn = None
        self.sample_labels = []

        if not data:
            self.lbl_matrix_dims.setText("Wymiary: —")
            return

        source_type = data.get('type')
        self.active_source_name = data.get('label', 'Dane')

        # A. SERIA 2D LUB TENSOR 3D
        if source_type == 'series':
            s_name = data.get('series_name')
            series_obj = session.series.get(s_name)
            if series_obj:
                self.active_wn = series_obj.wavenumbers
                self.active_matrix = series_obj.get_matrix_2d()
                if series_obj.is_3d:
                    self.active_tensor_3d = series_obj.data
                    self.lbl_matrix_dims.setText(
                        f"Tensor 3D: {series_obj.shape[0]} pkt. × {series_obj.shape[1]} (N) × {series_obj.shape[2]} (M)"
                    )
                else:
                    self.lbl_matrix_dims.setText(
                        f"Seria 2D: {self.active_matrix.shape[0]} pkt. × {self.active_matrix.shape[1]} próbek"
                    )

                if series_obj.axis_n_values is not None:
                    self.sample_labels = [f"{v:.2f}" for v in series_obj.axis_n_values]
                else:
                    self.sample_labels = [str(i + 1) for i in range(self.active_matrix.shape[1])]
            return

        # B. LISTA OBIEKTÓW SpectrumData (Okna robocze lub Magazyn)
        spectra_list: List[SpectrumData] = []

        if source_type == 'active_display':
            spectra_list = session.get_display_spectra()
        elif source_type == 'window':
            win_id = data.get('window_id')
            if win_id in session.windows:
                win = session.windows[win_id]
                if data.get('only_visible', True):
                    spectra_list = [s for s in win.spectra.values() if win.visibility.get(s.name, True)]
                else:
                    spectra_list = list(win.spectra.values())
        elif source_type == 'storage_exp':
            spectra_list = [
                s for s in session.spectra.values()
                if not isinstance(s, GaussianData) and getattr(s, 'file_type', '') not in ['gaussian', 'orca', 'dft']
            ]
        elif source_type == 'storage_visible':
            spectra_list = session.get_visible_spectra()

        if len(spectra_list) >= 2:
            ref_wn = spectra_list[0].wavenumbers
            mat = np.zeros((len(ref_wn), len(spectra_list)), dtype=np.float64)

            for i, s in enumerate(spectra_list):
                if s.wavenumbers[0] > s.wavenumbers[-1]:
                    xp, fp = s.wavenumbers[::-1], s.intensities[::-1]
                else:
                    xp, fp = s.wavenumbers, s.intensities
                mat[:, i] = np.interp(ref_wn, xp, fp)

            self.active_wn = ref_wn
            self.active_matrix = mat
            self.sample_labels = [s.name for s in spectra_list]
            self.lbl_matrix_dims.setText(f"Macierz 2D: {mat.shape[0]} pkt. × {mat.shape[1]} widm")
        else:
            self.active_matrix = None
            self.active_wn = None
            self.lbl_matrix_dims.setText(f"Wymiary: {len(spectra_list)} widm (wymagane min. 2)")

    def _on_method_changed(self):
        self._update_method_ui()

    def _on_subview_changed(self):
        if self.last_results is not None:
            method_idx = self.combo_method.currentIndex()
            canvas = self._get_canvas()
            if not canvas:
                return

            subview_idx = self.combo_subview.currentIndex()
            if method_idx in [2, 3]:  # SVD/PCA lub PFA
                if subview_idx == 0:
                    self._draw_loadings_scores(canvas, self.last_results)
                elif subview_idx == 1:
                    self._draw_scree_diagnostics(canvas, self.last_results)
                elif subview_idx == 2:
                    self._draw_reconstruction(canvas, self.last_results)

    def _update_method_ui(self):
        idx = self.combo_method.currentIndex()
        is_mcr = (idx == 0)
        is_spex = (idx == 1)
        is_svd_pca = (idx == 2)
        is_pfa = (idx == 3)
        is_classif = (idx == 4)
        is_parafac = (idx == 5)
        is_autowfa = (idx == 6)
        is_fsmw = (idx == 7)
        is_help = (idx == 8)
        is_cos = (idx == 9)

        # Widoczność kontrolek
        has_k = is_mcr or is_spex or is_svd_pca or is_pfa or is_parafac or is_autowfa
        self.lbl_k.setVisible(has_k)
        self.spin_k.setVisible(has_k)

        has_subviews = is_svd_pca or is_pfa
        self.lbl_subview.setVisible(has_subviews)
        self.combo_subview.setVisible(has_subviews)

        self.chk_pca_center.setVisible(is_svd_pca)
        self.lbl_proj.setVisible(is_classif)
        self.combo_proj.setVisible(is_classif)

        self.chk_nnls.setVisible(is_mcr or is_parafac)
        self.chk_mcr_spexfa.setVisible(is_mcr)
        self.chk_mcr_closure.setVisible(is_mcr)
        self.chk_mcr_unimodal.setVisible(is_mcr)
        self.lbl_mcr_iter.setVisible(is_mcr)
        self.spin_mcr_iter.setVisible(is_mcr)
        self.lbl_mcr_tol.setVisible(is_mcr)
        self.spin_mcr_tol.setVisible(is_mcr)

        self.lbl_window.setVisible(is_fsmw or is_help)
        self.spin_window.setVisible(is_fsmw or is_help)
        self.lbl_noise.setVisible(is_help)
        self.spin_noise.setVisible(is_help)

        has_pure = (self.last_results is not None and
                    ('pure_spectra' in self.last_results or 'loadings' in self.last_results))
        self.btn_inject_spectra.setEnabled(has_pure)

        has_recon = (self.last_results is not None and 'dr' in self.last_results)
        self.btn_inject_recon.setEnabled(has_recon)

    def _safe_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a', labelcolor='#ffffff', fontsize=8)

    def _on_run_analysis(self):
        if self.active_matrix is None or self.active_matrix.shape[1] < 2:
            QMessageBox.warning(self, tr("app.warning"), "Wybrane źródło zawiera mniej niż 2 widma. Zmień źródło danych.")
            return

        method_idx = self.combo_method.currentIndex()
        k = self.spin_k.value()
        k_max = min(self.active_matrix.shape)
        if k > k_max:
            k = k_max
            self.spin_k.setValue(k)

        canvas = self._get_canvas()
        if not canvas:
            return

        try:
            # 0. MCR-ALS
            if method_idx == 0:
                res = chem.run_mcr_als(
                    self.active_matrix, rank=k,
                    non_negative=self.chk_nnls.isChecked(),
                    closure=self.chk_mcr_closure.isChecked(),
                    unimodal=self.chk_mcr_unimodal.isChecked(),
                    use_spexfa_init=self.chk_mcr_spexfa.isChecked(),
                    max_iter=self.spin_mcr_iter.value(),
                    tol_err_change=self.spin_mcr_tol.value()
                )
                self.last_results = res
                self.lbl_res_summary.setText(
                    f"MCR-ALS (k={k}): LOF = {res['lof']:.2f}%, R² = {res['r2']:.2f}% "
                    f"({res['iterations']} iteracji, zbieżność: {res['converged']})."
                )
                self._draw_pure_components(canvas, res['pure_spectra'], res['pure_concentrations'], "MCR-ALS")

            # 1. SPEXFA
            elif method_idx == 1:
                res = chem.run_spexfa(self.active_matrix, n=k)
                self.last_results = res
                self.lbl_res_summary.setText(f"SPEXFA: Wyizolowano {k} czystych widm i profili stężeń z niepewnościami.")
                self._draw_pure_components(canvas, res['pure_spectra'], res['pure_concentrations'], "SPEXFA")

            # 2. SVD / PCA
            elif method_idx == 2:
                do_center = self.chk_pca_center.isChecked()
                res = chem.run_pca(self.active_matrix, n_components=k, center=do_center)
                self.last_results = res
                typ_str = "PCA (Centrowane)" if do_center else "SVD (Czyste ładunki widmowe)"
                var_sum = np.sum(res['explained_variance_ratio'])
                self.lbl_res_summary.setText(
                    f"{typ_str}: Wyjaśniona wariancja dla k={k}: {var_sum:.2f}%. "
                    f"Szum residuów (RMS): {res['residual_std']:.5f}."
                )
                self._on_subview_changed()

            # 3. PFA / Malinowski
            elif method_idx == 3:
                res = chem.run_pfa(self.active_matrix, n=k)
                self.last_results = res
                self.lbl_res_summary.setText(
                    f"PFA Malinowskiego: Sugerowany rząd matrycy wg minimum IND: k={res['suggested_k']}. "
                    f"Szum residuów dla k={k}: {res.get('residual_std', 0.0):.5f}."
                )
                self._on_subview_changed()

            # 4. Klasyfikacja 2D
            elif method_idx == 4:
                proj_name = self.combo_proj.currentText()
                if "PCA" in proj_name:
                    res_pca = chem.run_pca(self.active_matrix, n_components=2, center=True)
                    embedding = res_pca['scores'][:, :2]
                elif "UMAP" in proj_name:
                    embedding = chem.run_manifold_reduction(self.active_matrix, method="umap", n_components=2)
                else:
                    embedding = chem.run_manifold_reduction(self.active_matrix, method="tsne", n_components=2)

                self.last_results = {'embedding': embedding, 'method': proj_name}
                self.lbl_res_summary.setText(f"Klasyfikacja 2D: Zrzutowano {embedding.shape[0]} próbek metodą {proj_name}.")
                self._draw_classification_2d(canvas, embedding, proj_name)

            # 5. PARAFAC (Tensor 3D)
            elif method_idx == 5:
                if self.active_tensor_3d is None:
                    QMessageBox.warning(self, tr("app.warning"), "PARAFAC wymaga tensora 3D (W × N × M). Wybierz tensor z listy źródeł.")
                    return
                res = chem.run_parafac(self.active_tensor_3d, rank=k, non_negative=self.chk_nnls.isChecked())
                self.last_results = res
                self.lbl_res_summary.setText(f"PARAFAC: Rozkład tensora na {k} czynników.")
                self._draw_parafac(canvas, res)

            # 6. AUTOWFA
            elif method_idx == 6:
                res = chem.run_autowfa(self.active_matrix, n_chem=k)
                self.last_results = res
                self.lbl_res_summary.setText(f"AUTOWFA: Wyznaczono okna analityczne dla {k} składników.")
                self._draw_autowfa(canvas, res)

            # 7. FSMWEFA
            elif method_idx == 7:
                w_size = self.spin_window.value()
                res = chem.run_fsmwefa(self.active_matrix, window_size=w_size)
                self.last_results = res
                self.lbl_res_summary.setText(f"FSMWEFA: Profil rzędu dla okna H={w_size}.")
                self._draw_fsmwefa(canvas, res)

            # 8. HELP
            elif method_idx == 8:
                w_size = self.spin_window.value()
                noise_lvl = self.spin_noise.value()
                res = chem.run_help_rank_profile(self.active_matrix, window_size=w_size, noise_level=noise_lvl)
                self.last_results = res
                self.lbl_res_summary.setText(f"HELP: Profil rzędu lokalnego z progiem szumu {noise_lvl}.")
                self._draw_help(canvas, res)

            # 9. 2D-COS
            elif method_idx == 9:
                phi, psi = chem.calculate_2dcos(self.active_matrix)
                self.last_results = {'phi': phi, 'psi': psi}
                self.lbl_res_summary.setText("2D-COS: Wyznaczono mapy korelacji Φ oraz Ψ.")
                self._draw_2dcos(canvas, phi, psi)

            self._update_method_ui()

        except Exception as e:
            logger.error(f"Błąd analizy chemometrycznej: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd procedury chemometrycznej:\n{e}")

    # =========================================================================
    # PROCEDURY RENDEROWANIA NA CANVASIE
    # =========================================================================

    def _draw_pure_components(self, canvas, pure_spectra, pure_conc, method_name: str):
        canvas.prepare_axes(shared_x=False)
        k = pure_spectra.shape[1]
        palette = ['#e76f51', '#2a9d8f', '#e0aaff', '#f4a261', '#98c1d9', '#b5e48c']
        wn = self.active_wn

        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(wn, pure_spectra[:, i], color=c, lw=1.6, label=f"Składnik {i+1}")

        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.ax_main.set_title(f"{method_name}: Czyste Widma Składowe (Pure Spectra)")
        canvas.ax_main.set_ylabel("Absorbancja")
        self._safe_legend(canvas.ax_main)

        x_pts = np.arange(pure_conc.shape[0])
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_sub.plot(x_pts, pure_conc[:, i], 'o-', color=c, lw=1.3, markersize=4, label=f"Profil {i+1}")

        canvas.ax_sub.set_xlim(-0.5, len(x_pts) - 0.5)
        canvas.ax_sub.set_xlabel("Próbka / Krok procesu ewolucyjnego")
        canvas.ax_sub.set_ylabel("Względny udział / Stężenie")
        if len(self.sample_labels) == len(x_pts) and len(x_pts) <= 15:
            canvas.ax_sub.set_xticks(x_pts)
            canvas.ax_sub.set_xticklabels(self.sample_labels, rotation=15, fontsize=8)

        self._safe_legend(canvas.ax_sub)
        canvas.canvas.draw_idle()

    def _draw_loadings_scores(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        loadings = res['loadings']
        scores = res['scores']
        k = loadings.shape[1]
        wn = self.active_wn
        palette = ['#00E5FF', '#FF5252', '#76FF03', '#FFD700', '#E040FB', '#FF9100']

        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(wn, loadings[:, i], color=c, lw=1.5, label=f"Ładunek / Składowa {i+1}")

        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.ax_main.set_title("Widmowe Wektory Ładunków (Loadings / Spectral Components)")
        canvas.ax_main.set_ylabel("Amplituda ładunku")
        self._safe_legend(canvas.ax_main)

        x_pts = np.arange(scores.shape[0])
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_sub.plot(x_pts, scores[:, i], 'o-', color=c, lw=1.3, markersize=4, label=f"Wynik / Score {i+1}")

        canvas.ax_sub.set_xlim(-0.5, len(x_pts) - 0.5)
        canvas.ax_sub.set_xlabel("Numer widma w serii")
        canvas.ax_sub.set_ylabel("Wartość wyniku (Score)")
        self._safe_legend(canvas.ax_sub)
        canvas.canvas.draw_idle()

    def _draw_scree_diagnostics(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        ev = res['ev']
        factors = np.arange(1, len(ev) + 1)

        canvas.ax_main.semilogy(factors, ev, 'o-', color='#00E5FF', lw=1.5, label="Wartości własne (EV)")
        canvas.ax_main.set_title("Wykres Osypiska (Scree Plot / Log EV)")
        canvas.ax_main.set_ylabel("Log(EV)")
        canvas.ax_main.set_xlabel("Numer czynnika")
        self._safe_legend(canvas.ax_main)

        if 're' in res and 'ind' in res:
            re = res['re']
            ind = res['ind']
            valid = ~np.isnan(re)
            canvas.ax_sub.plot(factors[valid], re[valid], 's--', color='#f4a261', label="Błąd rzecz. (RE)")
            canvas.ax_sub.plot(factors[valid], ind[valid], '^:', color='#2a9d8f', label="Wskaźnik IND Malinowskiego")
            canvas.ax_sub.set_xlabel("Numer czynnika")
            canvas.ax_sub.set_ylabel("Wskaźniki błędu")
            self._safe_legend(canvas.ax_sub)
        elif 'explained_variance_ratio' in res:
            var = res['explained_variance_ratio']
            canvas.ax_sub.bar(np.arange(1, len(var) + 1), var, color='#457b9d', alpha=0.8, label="Wariancja [%]")
            canvas.ax_sub.set_xlabel("Numer składowej")
            canvas.ax_sub.set_ylabel("Wyjaśniona wariancja [%]")
            self._safe_legend(canvas.ax_sub)

        canvas.canvas.draw_idle()

    def _draw_reconstruction(self, canvas, res):
        canvas.prepare_axes(shared_x=True)
        wn = self.active_wn
        d_orig = self.active_matrix
        d_recon = res['dr']
        residuals = res['residuals']

        for j in range(d_orig.shape[1]):
            canvas.ax_main.plot(wn, d_orig[:, j], color='#555566', alpha=0.4, lw=0.8)
            canvas.ax_main.plot(wn, d_recon[:, j], lw=1.2, alpha=0.9)

        canvas.ax_main.set_title(f"Rekonstrukcja sygnału z k={res['k']} czynników (Odszumianie)")
        canvas.ax_main.set_ylabel("Absorbancja")

        for j in range(residuals.shape[1]):
            canvas.ax_sub.plot(wn, residuals[:, j], lw=0.8, alpha=0.7)

        canvas.ax_sub.axhline(0.0, color='red', linestyle='--', lw=0.8)
        canvas.ax_sub.set_xlabel(tr("canvas.wavenumber", "Liczba falowa [cm⁻¹]"))
        canvas.ax_sub.set_ylabel("Residuum (Szum)")
        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.canvas.draw_idle()

    def _draw_classification_2d(self, canvas, embedding, method_name: str):
        canvas.prepare_axes(shared_x=False)
        n_pts = embedding.shape[0]

        canvas.ax_main.plot(embedding[:, 0], embedding[:, 1], '-', color='#555566', alpha=0.5, lw=1.0)
        canvas.ax_main.scatter(
            embedding[:, 0], embedding[:, 1],
            c=np.arange(n_pts), cmap='viridis', s=60, edgecolors='#ffffff', lw=0.8
        )

        for i in range(n_pts):
            lbl = self.sample_labels[i] if i < len(self.sample_labels) else str(i + 1)
            canvas.ax_main.annotate(
                lbl, (embedding[i, 0], embedding[i, 1]),
                fontsize=7, color='#dddddd', xytext=(4, 4), textcoords='offset points'
            )

        canvas.ax_main.set_title(f"Klasyfikacja i Rzutowanie 2D ({method_name})")
        canvas.ax_main.set_xlabel("Wymiar 1")
        canvas.ax_main.set_ylabel("Wymiar 2")

        canvas.ax_sub.text(
            0.5, 0.5,
            f"Zrzutowano {n_pts} widm.\nKolor punktu odpowiada kolejności pomiaru w serii (początek -> fiolet, koniec -> żółty).",
            ha='center', va='center', color='#a0a0a0', fontsize=9
        )
        canvas.canvas.draw_idle()

    def _draw_parafac(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        pure_spectra = res['pure_spectra']
        m1 = res['mode1_trends']
        m2 = res['mode2_trends']
        k = res['rank']
        palette = ['#e76f51', '#2a9d8f', '#e0aaff', '#f4a261', '#98c1d9']
        wn = self.active_wn

        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(wn, pure_spectra[:, i], color=c, lw=1.6, label=f"Czynnik {i+1}")

        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.ax_main.set_title("PARAFAC: Składowe Widmowe (Mode 0)")
        canvas.ax_main.set_ylabel("Absorbancja")
        self._safe_legend(canvas.ax_main)

        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_sub.plot(np.arange(len(m1[:, i])), m1[:, i], 'o-', color=c, lw=1.3, label=f"Trend N ({i+1})")
            canvas.ax_sub.plot(np.arange(len(m2[:, i])), m2[:, i], 's--', color=c, alpha=0.7, lw=1.0, label=f"Trend M ({i+1})")

        canvas.ax_sub.set_xlabel("Indeks składowej tensora")
        canvas.ax_sub.set_ylabel("Waga / Profil")
        self._safe_legend(canvas.ax_sub)
        canvas.canvas.draw_idle()

    def _draw_autowfa(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        conc = res['conc']
        x_pts = np.arange(conc.shape[1])
        palette = ['#e76f51', '#2a9d8f', '#e0aaff', '#f4a261', '#98c1d9']

        for i in range(conc.shape[0]):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(x_pts, conc[i, :], 'o-', color=c, lw=1.5, label=f"Profil {i+1}")

        canvas.ax_main.set_title("AUTOWFA: Wyznaczone Profile Stężeń Ewolucyjnych")
        canvas.ax_main.set_ylabel("Względne stężenie")
        canvas.ax_main.set_xlabel("Indeks pomiaru w serii")
        self._safe_legend(canvas.ax_main)

        canvas.ax_sub.text(0.5, 0.5, "Okna analityczne WFAX zostały dopasowane.", ha='center', va='center', color='gray')
        canvas.canvas.draw_idle()

    def _draw_fsmwefa(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        efa_mat = np.copy(res['efa_matrix'])
        efa_mat[np.isnan(efa_mat) | (efa_mat <= 1e-10)] = 1e-10
        log_mat = np.log10(efa_mat)

        canvas.ax_main.imshow(log_mat, aspect='auto', cmap='viridis', origin='lower')
        canvas.ax_main.set_title("FSMWEFA: Logarytmiczna Mapa Rzędu (Wartości SVD w oknie)")
        canvas.ax_main.set_ylabel("Numer czynnika")
        canvas.ax_main.set_xlabel("Indeks widma w serii ewolucyjnej")

        canvas.ax_sub.text(0.5, 0.5, f"Analiza w oknie przesuwnym H={res['window_size']}.", ha='center', va='center', color='gray')
        canvas.canvas.draw_idle()

    def _draw_help(self, canvas, res):
        canvas.prepare_axes(shared_x=False)
        rank_profile = res['rank_profile']
        x_pts = np.arange(len(rank_profile))

        canvas.ax_main.step(x_pts, rank_profile, where='mid', color='#00E5FF', lw=1.8, label="Rząd lokalny")
        canvas.ax_main.set_title(f"HELP: Profil Rzędu Lokalnego (Okno H={res['window_size']})")
        canvas.ax_main.set_ylabel("Liczba czynników")
        canvas.ax_main.set_xlabel("Indeks widma w serii")
        canvas.ax_main.set_yticks(np.arange(0, np.max(rank_profile) + 2))
        self._safe_legend(canvas.ax_main)

        canvas.ax_sub.text(
            0.5, 0.5,
            "Wartość 1 wskazuje strefę czystego składnika (Rank-1 zone / Pure variable).\n"
            "Wartości > 1 oznaczają strefy nakładania się i ko-elucji.",
            ha='center', va='center', color='#a0a0a0'
        )
        canvas.canvas.draw_idle()

    def _draw_2dcos(self, canvas, phi, psi):
        canvas.prepare_axes(shared_x=False)
        wn = self.active_wn
        extent = [np.max(wn), np.min(wn), np.max(wn), np.min(wn)]

        vmax_phi = np.max(np.abs(phi))
        canvas.ax_main.imshow(phi, cmap='RdBu_r', vmin=-vmax_phi, vmax=vmax_phi, extent=extent, interpolation='nearest')
        canvas.ax_main.set_title("2D-COS: Mapa Synchroniczna (Φ)")
        canvas.ax_main.set_ylabel("ν₁ [cm⁻¹]")

        vmax_psi = np.max(np.abs(psi))
        canvas.ax_sub.imshow(psi, cmap='RdBu_r', vmin=-vmax_psi, vmax=vmax_psi, extent=extent, interpolation='nearest')
        canvas.ax_sub.set_title("2D-COS: Mapa Asynchroniczna (Ψ)")
        canvas.ax_sub.set_xlabel("ν₂ [cm⁻¹]")
        canvas.ax_sub.set_ylabel("ν₁ [cm⁻¹]")

        canvas.canvas.draw_idle()

    # =========================================================================
    # WSTRZYKIWANIE WYNIKÓW DO SESJI
    # =========================================================================

    def _on_inject_spectra_to_session(self):
        if not self.last_results:
            return

        pure_spectra = self.last_results.get('pure_spectra')
        if pure_spectra is None:
            pure_spectra = self.last_results.get('loadings')

        if pure_spectra is None:
            QMessageBox.warning(self, tr("app.warning"), "Brak wektorów widmowych do wyeksportowania.")
            return

        k = pure_spectra.shape[1]
        method_name = self.combo_method.currentText().split()[0].replace('/', '_')

        for i in range(k):
            spec_name = f"{method_name}_Comp_{i + 1}"
            spec_obj = SpectrumData(
                wavenumbers=self.active_wn.copy(),
                intensities=pure_spectra[:, i].copy(),
                name=spec_name,
                file_path="",
                file_type="pure_component",
                metadata={"source": method_name, "component_index": i + 1}
            )
            session.add_spectrum(spec_obj, make_active=(i == 0))

        QMessageBox.information(
            self, tr("app.info"),
            f"Pomyślnie dodano {k} wyizolowanych profili widmowych ({method_name}) do projektu!"
        )

    def _on_inject_reconstruction_to_session(self):
        if not self.last_results or 'dr' not in self.last_results:
            QMessageBox.warning(self, tr("app.warning"), "Brak danych rekonstrukcji macierzy (dr).")
            return

        dr = self.last_results['dr']
        k = self.last_results.get('k', self.spin_k.value())
        method_name = self.combo_method.currentText().split()[0].replace('/', '_')
        series_name = f"Recon_{self.active_source_name}_{method_name}_k{k}"

        recon_series = SpectralSeries(
            name=series_name,
            wavenumbers=self.active_wn.copy(),
            data=dr.copy(),
            axis_n_name="Próbka",
            axis_n_values=None,
            metadata={"source_matrix": self.active_source_name, "rank_k": k, "method": method_name}
        )

        session.add_series(recon_series)
        self._refresh_sources()

        QMessageBox.information(
            self, tr("app.info"),
            f"Dodano odszumioną serię rekonstrukcji '{series_name}' do drzewa projektu!"
        )

    def retranslate_ui(self):
        self.grp_source.setTitle(tr("chem.source_group", "1. Źródło Danych do Analizy"))
        self.lbl_source.setText(tr("chem.source", "Wybierz zestaw:"))
        self.grp_method.setTitle(tr("chem.method_group", "2. Metoda Chemometryczna"))
        self.grp_params.setTitle(tr("chem.params_group", "3. Parametry i Opcje"))
        self.lbl_k.setText(tr("chem.components_k", "Liczba składników (k):"))
        self.lbl_window.setText(tr("chem.window_h", "Szerokość okna (H):"))
        self.btn_run.setText(tr("chem.run_btn", "⚡ Uruchom Analizę"))
        self.grp_results.setTitle(tr("chem.results_group", "4. Wyniki & Integracja z Projektem"))
        self.btn_inject_spectra.setText(tr("chem.inject_btn", "📥 Dodaj Wyizolowane Widma do Projektu"))
        self.btn_inject_recon.setText(tr("chem.inject_recon_btn", "📥 Dodaj Odszumioną Serię (D_k) do Projektu"))