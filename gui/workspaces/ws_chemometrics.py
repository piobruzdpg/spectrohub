"""
gui/workspaces/ws_chemometrics.py
Zintegrowany panel perspektywy 3: Chemometria i Analiza Ewolucyjna.
Obsługuje serie 2D, tensory 3D, PARAFAC, MCR-ALS, SPEXFA, PFA, FSMWEFA i 2D-COS.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QMessageBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import numpy as np
import logging
from typing import Optional, Dict, Any

from core.session import session
from core.models import SpectrumData, SpectralSeries
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
        grp_source = QGroupBox("1. Źródło Danych")
        self.grp_source = grp_source
        l_src = QVBoxLayout(grp_source)

        row_src = QHBoxLayout()
        self.lbl_source = QLabel("Macierz / Tensor:")
        self.combo_source = QComboBox()
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        row_src.addWidget(self.lbl_source)
        row_src.addWidget(self.combo_source, 1)
        l_src.addLayout(row_src)

        self.lbl_matrix_dims = QLabel("Wymiary: —")
        self.lbl_matrix_dims.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        l_src.addWidget(self.lbl_matrix_dims)
        layout.addWidget(grp_source)

        # 2. Wybór metody chemometrycznej
        grp_method = QGroupBox("2. Metoda Chemometryczna")
        self.grp_method = grp_method
        l_meth = QVBoxLayout(grp_method)

        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "MCR-ALS (Czyste składniki)",
            "SPEXFA (Izolacja widm i stężeń)",
            "PARAFAC (Dekompozycja tensora 3D)",
            "PFA / Malinowski (Analiza rang)",
            "AUTOWFA (Analiza z oknem)",
            "FSMWEFA (Rząd w oknie przesuwnym)",
            "2D-COS (Spektroskopia korelacyjna)"
        ])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        l_meth.addWidget(self.combo_method)
        layout.addWidget(grp_method)

        # 3. Parametry algorytmu
        grp_params = QGroupBox("3. Parametry Algorytmu")
        self.grp_params = grp_params
        l_params = QVBoxLayout(grp_params)

        row_k = QHBoxLayout()
        self.lbl_k = QLabel("Liczba składników (k):")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 30)
        self.spin_k.setValue(2)
        row_k.addWidget(self.lbl_k)
        row_k.addWidget(self.spin_k)
        l_params.addLayout(row_k)

        # Opcje MCR i PARAFAC
        self.chk_nnls = QCheckBox("Wymuś nieujemność (NNLS / Non-negative)")
        self.chk_nnls.setChecked(True)
        self.chk_mcr_spexfa = QCheckBox("Inicjalizacja widm przez SPEXFA")
        self.chk_mcr_spexfa.setChecked(True)
        self.chk_mcr_closure = QCheckBox("Domknięcie bilansu (Closure)")
        self.chk_mcr_closure.setChecked(False)

        l_params.addWidget(self.chk_nnls)
        l_params.addWidget(self.chk_mcr_spexfa)
        l_params.addWidget(self.chk_mcr_closure)

        # Opcje FSMWEFA
        row_w = QHBoxLayout()
        self.lbl_window = QLabel("Szerokość okna (H):")
        self.spin_window = QSpinBox()
        self.spin_window.setRange(3, 99)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(7)
        row_w.addWidget(self.lbl_window)
        row_w.addWidget(self.spin_window)
        l_params.addLayout(row_w)

        layout.addWidget(grp_params)

        # 4. Przycisk obliczeniowy
        self.btn_run = QPushButton("⚡ Uruchom Analizę")
        self.btn_run.setStyleSheet("background-color: #3d5a80; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run_analysis)
        layout.addWidget(self.btn_run)

        # 5. Podsumowanie & Wstrzykiwanie wyników
        grp_results = QGroupBox("4. Wyniki & Eksport")
        self.grp_results = grp_results
        l_res = QVBoxLayout(grp_results)

        self.lbl_res_summary = QLabel("Brak obliczonych wyników.")
        self.lbl_res_summary.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        l_res.addWidget(self.lbl_res_summary)

        self.btn_inject_spectra = QPushButton("📥 Dodaj Wyizolowane Widma do Projektu")
        self.btn_inject_spectra.setStyleSheet("background-color: #2a9d8f; font-weight: bold;")
        self.btn_inject_spectra.setEnabled(False)
        self.btn_inject_spectra.clicked.connect(self._on_inject_spectra_to_session)
        l_res.addWidget(self.btn_inject_spectra)

        layout.addWidget(grp_results)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        self._update_method_ui()
        self.retranslate_ui()

    def _connect_signals(self):
        if hasattr(session, 'series_added'):
            session.series_added.connect(self._refresh_sources)
            session.series_removed.connect(self._refresh_sources)
        session.spectrum_added.connect(self._refresh_sources)
        session.spectrum_removed.connect(self._refresh_sources)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_sources()

    def _get_canvas(self):
        main_win = self.window()
        return getattr(main_win, 'canvas_2d', None)

    def _refresh_sources(self):
        current = self.combo_source.currentText()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()

        # 1. Serie ewolucyjne 2D i tensory 3D
        if hasattr(session, 'series'):
            for s_name in session.series.keys():
                self.combo_source.addItem(f"📊 {s_name}")

        # 2. Zaznaczone widma eksperymentalne
        vis_count = len(session.get_visible_spectra())
        self.combo_source.addItem(f"📈 Zaznaczone widma z projektu ({vis_count})")

        if current:
            idx = self.combo_source.findText(current)
            if idx >= 0:
                self.combo_source.setCurrentIndex(idx)

        self.combo_source.blockSignals(False)
        self._on_source_changed()

    def _on_source_changed(self):
        txt = self.combo_source.currentText()
        self.active_tensor_3d = None

        if txt.startswith("📊 ") and hasattr(session, 'series'):
            s_name = txt.replace("📊 ", "")
            series_obj = session.series.get(s_name)
            if series_obj:
                self.active_wn = series_obj.wavenumbers
                self.active_matrix = series_obj.get_matrix_2d()
                if series_obj.is_3d:
                    self.active_tensor_3d = series_obj.data
                    self.lbl_matrix_dims.setText(f"Tensor 3D: {series_obj.shape[0]} pkt. × {series_obj.shape[1]} (N) × {series_obj.shape[2]} (M)")
                else:
                    self.lbl_matrix_dims.setText(f"Seria 2D: {self.active_matrix.shape[0]} pkt. × {self.active_matrix.shape[1]} próbek")

                if series_obj.axis_n_values is not None:
                    self.sample_labels = [f"{v:.2f}" for v in series_obj.axis_n_values]
                else:
                    self.sample_labels = [str(i + 1) for i in range(self.active_matrix.shape[1])]
        else:
            spectra = session.get_visible_spectra()
            if len(spectra) >= 2:
                ref_wn = spectra[0].wavenumbers
                mat = np.zeros((len(ref_wn), len(spectra)), dtype=np.float64)
                for i, s in enumerate(spectra):
                    mat[:, i] = np.interp(ref_wn, s.wavenumbers, s.intensities)
                self.active_wn = ref_wn
                self.active_matrix = mat
                self.sample_labels = [s.name for s in spectra]
                self.lbl_matrix_dims.setText(f"Macierz 2D: {mat.shape[0]} pkt. × {mat.shape[1]} próbek")
            else:
                self.active_wn = None
                self.active_matrix = None
                self.lbl_matrix_dims.setText("Wymiary: < 2 widma (wymagane min. 2)")

    def _on_method_changed(self):
        self._update_method_ui()

    def _update_method_ui(self):
        method_idx = self.combo_method.currentIndex()
        is_mcr = (method_idx == 0)
        is_parafac = (method_idx == 2)
        is_fsmw = (method_idx == 5)
        is_cos = (method_idx == 6)

        self.chk_nnls.setVisible(is_mcr or is_parafac)
        self.chk_mcr_spexfa.setVisible(is_mcr)
        self.chk_mcr_closure.setVisible(is_mcr)

        self.lbl_k.setVisible(not is_cos and not is_fsmw)
        self.spin_k.setVisible(not is_cos and not is_fsmw)

        self.lbl_window.setVisible(is_fsmw)
        self.spin_window.setVisible(is_fsmw)

        # Aktualizacja statusu przycisku wstrzykiwania
        can_inject = method_idx in [0, 1, 2] and self.last_results is not None and 'pure_spectra' in self.last_results
        self.btn_inject_spectra.setEnabled(can_inject)
        if method_idx in [0, 1, 2]:
            self.btn_inject_spectra.setText(tr("chem.inject_btn", "📥 Dodaj Wyizolowane Widma do Projektu"))
        else:
            self.btn_inject_spectra.setText(tr("chem.no_pure_spectra", "(Wybrana metoda nie generuje widm czystych)"))

    def _safe_legend(self, ax):
        """Eliminuje ostrzeżenie Matplotlib przy pustej legendzie."""
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a', labelcolor='#ffffff', fontsize=8)

    def _on_run_analysis(self):
        if self.active_matrix is None or self.active_matrix.shape[1] < 2:
            QMessageBox.warning(self, tr("app.warning"), "Brak wystarczającej liczby próbek do analizy (min. 2).")
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
                    use_spexfa_init=self.chk_mcr_spexfa.isChecked()
                )
                self.last_results = res
                self.lbl_res_summary.setText(f"MCR-ALS: Zbieżność po {res['iterations']} iteracjach (k={k}).")
                self._update_method_ui()
                self._draw_pure_components(canvas, res['pure_spectra'], res['pure_concentrations'], "MCR-ALS")

            # 1. SPEXFA
            elif method_idx == 1:
                res = chem.run_spexfa(self.active_matrix, n=k)
                self.last_results = res
                self.lbl_res_summary.setText(f"SPEXFA: Wyizolowano {k} czystych widm i profili stężeń.")
                self._update_method_ui()
                self._draw_pure_components(canvas, res['pure_spectra'], res['pure_concentrations'], "SPEXFA")

            # 2. PARAFAC (Wymaga Tensora 3D)
            elif method_idx == 2:
                if self.active_tensor_3d is None:
                    QMessageBox.warning(self, tr("app.warning"), "PARAFAC wymaga tensora 3D (W × N × M). Wybierz tensor z listy źródeł.")
                    return
                res = chem.run_parafac(self.active_tensor_3d, rank=k, non_negative=self.chk_nnls.isChecked())
                self.last_results = res
                self.lbl_res_summary.setText(f"PARAFAC: Rozkład tensora na {k} czynników.")
                self._update_method_ui()
                self._draw_parafac(canvas, res)

            # 3. PFA / Malinowski
            elif method_idx == 3:
                res = chem.run_pfa(self.active_matrix, n=k)
                self.last_results = res
                self.lbl_res_summary.setText(f"PFA: Sugerowany rząd matrycy wg IND: k={res['suggested_k']}")
                self._update_method_ui()
                self._draw_pfa(canvas, res)

            # 4. AUTOWFA
            elif method_idx == 4:
                res = chem.run_autowfa(self.active_matrix, n_chem=k)
                self.last_results = res
                self.lbl_res_summary.setText(f"AUTOWFA: Wyznaczono okna analityczne dla {k} składników.")
                self._update_method_ui()
                self._draw_autowfa(canvas, res)

            # 5. FSMWEFA
            elif method_idx == 5:
                w_size = self.spin_window.value()
                res = chem.run_fsmwefa(self.active_matrix, window_size=w_size)
                self.last_results = res
                self.lbl_res_summary.setText(f"FSMWEFA: Wyznaczono profil rzędu dla okna H={w_size}.")
                self._update_method_ui()
                self._draw_fsmwefa(canvas, res)

            # 6. 2D-COS
            elif method_idx == 6:
                phi, psi = chem.calculate_2dcos(self.active_matrix)
                self.last_results = {'phi': phi, 'psi': psi}
                self.lbl_res_summary.setText("2D-COS: Wyznaczono mapy korelacji Phi i Psi.")
                self._update_method_ui()
                self._draw_2dcos(canvas, phi, psi)

        except Exception as e:
            logger.error(f"Błąd analizy chemometrycznej: {e}")
            QMessageBox.critical(self, tr("app.error"), f"Błąd procedury chemometrycznej:\n{e}")

    # --- PROCEDURY WIZUALIZACJI Z ROZŁĄCZONĄ OSIĄ X ---

    def _draw_pure_components(self, canvas, pure_spectra, pure_conc, method_name: str):
        # Rozłączamy oś dolnego panelu, aby stężenia skalowały się od 0 do N
        canvas.prepare_axes(shared_x=False)

        k = pure_spectra.shape[1]
        palette = ['#e76f51', '#2a9d8f', '#e0aaff', '#f4a261', '#98c1d9', '#b5e48c']
        wn = self.active_wn

        # Górny wykres: Czyste widma
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(wn, pure_spectra[:, i], color=c, lw=1.6, label=f"Składnik {i+1}")

        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.ax_main.set_title(f"{method_name}: Wyizolowane Czyste Widma (Pure Spectra)")
        canvas.ax_main.set_ylabel("Absorbancja")
        self._safe_legend(canvas.ax_main)

        # Dolny wykres: Profile stężeń
        x_pts = np.arange(pure_conc.shape[0])
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_sub.plot(x_pts, pure_conc[:, i], 'o-', color=c, lw=1.3, markersize=4, label=f"Stężenie {i+1}")

        canvas.ax_sub.set_xlim(-0.5, len(x_pts) - 0.5)
        canvas.ax_sub.set_xlabel("Próbka / Krok procesu ewolucyjnego")
        canvas.ax_sub.set_ylabel("Względny udział")
        if len(self.sample_labels) == len(x_pts) and len(x_pts) <= 15:
            canvas.ax_sub.set_xticks(x_pts)
            canvas.ax_sub.set_xticklabels(self.sample_labels, rotation=15, fontsize=8)

        self._safe_legend(canvas.ax_sub)
        canvas.canvas.draw_idle()

    def _draw_parafac(self, canvas, res):
        canvas.prepare_axes(shared_x=False)

        pure_spectra = res['pure_spectra']
        m1 = res['mode1_trends']
        m2 = res['mode2_trends']
        k = res['rank']
        palette = ['#e76f51', '#2a9d8f', '#e0aaff', '#f4a261', '#98c1d9']
        wn = self.active_wn

        # Góra: Widma PARAFAC
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_main.plot(wn, pure_spectra[:, i], color=c, lw=1.6, label=f"Czynnik {i+1}")

        canvas.ax_main.set_xlim(np.max(wn), np.min(wn))
        canvas.ax_main.set_title("PARAFAC: Składowe Widmowe (Mode 0)")
        canvas.ax_main.set_ylabel("Absorbancja")
        self._safe_legend(canvas.ax_main)

        # Dół: Trendy Mode 1 (linie ciągłe) i Mode 2 (linie przerywane)
        for i in range(k):
            c = palette[i % len(palette)]
            canvas.ax_sub.plot(np.arange(len(m1[:, i])), m1[:, i], 'o-', color=c, lw=1.3, label=f"Trend N (Czynnik {i+1})")
            canvas.ax_sub.plot(np.arange(len(m2[:, i])), m2[:, i], 's--', color=c, alpha=0.7, lw=1.0, label=f"Trend M (Czynnik {i+1})")

        canvas.ax_sub.set_xlabel("Indeks składowej tensora")
        canvas.ax_sub.set_ylabel("Waga / Profil")
        self._safe_legend(canvas.ax_sub)
        canvas.canvas.draw_idle()

    def _draw_pfa(self, canvas, res):
        canvas.prepare_axes(shared_x=False)

        factors = res['factors']
        ev = res['ev']
        re = res['re']
        ind = res['ind']

        canvas.ax_main.semilogy(factors, ev, 'o-', color='#00E5FF', lw=1.5, label="Wartości własne (EV)")
        canvas.ax_main.set_title("Wykres Osypiska (Scree Plot / Log EV)")
        canvas.ax_main.set_ylabel("Log(EV)")
        self._safe_legend(canvas.ax_main)

        valid = ~np.isnan(re)
        canvas.ax_sub.plot(factors[valid], re[valid], 's--', color='#f4a261', label="Błąd rzecz. (RE)")
        canvas.ax_sub.plot(factors[valid], ind[valid], '^:', color='#2a9d8f', label="Wskaźnik IND")
        canvas.ax_sub.set_xlabel("Numer czynnika")
        canvas.ax_sub.set_ylabel("Wskaźniki błędu")
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

        # Informacja na dolnym panelu
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

    def _on_inject_spectra_to_session(self):
        """Wstrzykuje wyizolowane czyste widma (SPEXFA / MCR-ALS / PARAFAC) do drzewa projektu."""
        if not self.last_results or 'pure_spectra' not in self.last_results:
            QMessageBox.information(
                self, tr("app.info"),
                "Ostatnio wykonana analiza nie generuje widm czystych składowych.\n"
                "Uruchom MCR-ALS, SPEXFA lub PARAFAC, aby wyizolować profile widmowe."
            )
            return

        pure_spectra = self.last_results['pure_spectra']
        k = pure_spectra.shape[1]
        method_name = self.combo_method.currentText().split()[0].replace('/', '_')

        for i in range(k):
            spec_name = f"{method_name}_Pure_Comp_{i + 1}"
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
            f"Pomyślnie dodano {k} wyizolowanych widm czystych ({method_name}) do projektu!"
        )

    def retranslate_ui(self):
        self.grp_source.setTitle(tr("chem.source_group", "1. Źródło Danych"))
        self.lbl_source.setText(tr("chem.source", "Macierz / Tensor:"))
        self.grp_method.setTitle(tr("chem.method_group", "2. Metoda Chemometryczna"))
        self.grp_params.setTitle(tr("chem.params_group", "3. Parametry Algorytmu"))
        self.lbl_k.setText(tr("chem.components_k", "Liczba składników (k):"))
        self.lbl_window.setText(tr("chem.window_h", "Szerokość okna (H):"))
        self.btn_run.setText(tr("chem.run_btn", "⚡ Uruchom Analizę"))
        self.grp_results.setTitle(tr("chem.results_group", "4. Wyniki & Eksport"))
        self.btn_inject_spectra.setText(tr("chem.inject_btn", "📥 Dodaj Wyizolowane Widma do Projektu"))