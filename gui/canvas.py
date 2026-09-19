"""
gui/canvas.py
Wysokowydajny dwupanelowy wykres Matplotlib (Góra: Widma i Serie 2D, Dół: Residua/Kinetyka).
Podwójny klik LPM = Reset zoomu, PPM = Menu kontekstowe, drag LPM = Zoom na obu panelach.
Inteligentna obsługa tabel profili i raportów parametrów (HDO Fit).
Deterministyczna paleta barw z pełną synchronizacją z próbnikiem w drzewie projektu.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication, QMenu, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from core.session import session
from core.models import SpectrumData, SpectralSeries, DataTable
from core.i18n import tr

logger = logging.getLogger("SpectroHub")

# Laboratoryjna paleta wysokiego kontrastu dopasowana do ciemnego tła SpectroHub
SPECTRAL_PALETTE = [
    "#4cc9f0",  # 1. Jasny cyjan
    "#f72585",  # 2. Karmazyn / neon magenta
    "#4ade80",  # 3. Jasna mięta / zieleń
    "#fb8500",  # 4. Ciepły pomarańcz
    "#d946ef",  # 5. Fuksja / jasny fiolet
    "#ffb703",  # 6. Bursztynowy złoty
    "#06d6a0",  # 7. Szmaragdowy turkus
    "#ff4d6d",  # 8. Koralowa czerwień
    "#70d6ff",  # 9. Błękit nieba
    "#ffd166",  # 10. Ciepły żółty
    "#a78bfa",  # 11. Lawendowy
    "#80ed99",  # 12. Jasna limonka
]

TRACE_PALETTE = ['#ee6c4d', '#00e5ff', '#2a9d8f', '#f4a261', '#e0aaff']


class SpectroCanvas(QWidget):
    """Zintegrowany wykres z obsługą Smart Zoom na obu panelach, akceleracją LineCollection i kinetyką."""

    peak_add_requested = Signal(float, float)
    dft_mode_selected = Signal(int)
    kinetic_point_selected = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 6), facecolor='#1e1e24')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout.addWidget(self.canvas)

        self.ax_main = self.fig.add_subplot(4, 1, (1, 3))
        self.ax_sub = self.fig.add_subplot(4, 1, 4)
        # Bliźniacza górna podziałka dla numerów widm serii (bez dzielenia obiektu XAxis)
        self.ax_sub_top = self.ax_sub.twiny()

        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.09, hspace=0.28)

        self.zoom_start = None
        self.zoom_rect = None
        self.zoom_target_ax = None
        self.initial_xlim = None
        self.initial_ylim = None
        self.ax_sub_initial_xlim = None
        self.ax_sub_initial_ylim = None

        self.interaction_mode = "normal"
        self._deconv_cache: Optional[Dict[str, Any]] = None

        self.kinetic_traces: List[Dict[str, Any]] = []
        self.last_kinetic_export_data: Optional[Dict[str, Any]] = None

        self._apply_styling()
        self._connect_events()

        session.spectrum_added.connect(self.redraw_all)
        session.spectra_batch_added.connect(self.redraw_all)
        session.spectrum_removed.connect(self.redraw_all)
        session.visibility_changed.connect(self.redraw_all)
        session.active_spectrum_changed.connect(self.redraw_all)
        session.session_cleared.connect(self.redraw_all)

        session.series_added.connect(self.redraw_all)
        session.series_removed.connect(self.redraw_all)
        session.series_visibility_changed.connect(self.redraw_all)
        session.active_series_changed.connect(self.redraw_all)

        session.active_window_changed.connect(self.redraw_all)
        session.window_spectrum_added.connect(self.redraw_all)
        session.window_spectra_batch_added.connect(self.redraw_all)
        session.window_spectrum_removed.connect(self.redraw_all)
        session.window_visibility_changed.connect(self.redraw_all)
        session.context_changed.connect(self.redraw_all)

        session.active_table_changed.connect(self._on_active_table_changed)

    def _apply_sub_top_styling(self):
        """Konfiguruje ax_sub_top jako czystą oś pomocniczą na górze dolnego panelu."""
        if self.ax_sub_top is None:
            return
        self.ax_sub_top.xaxis.tick_top()
        self.ax_sub_top.xaxis.set_label_position('top')
        self.ax_sub_top.tick_params(
            axis='x', which='both',
            top=True, labeltop=True,
            bottom=False, labelbottom=False,
            colors='#a0a0a0', labelsize=8
        )
        self.ax_sub_top.tick_params(
            axis='y', which='both',
            left=False, labelleft=False,
            right=False, labelright=False
        )
        for spine_name, spine in self.ax_sub_top.spines.items():
            if spine_name == 'top':
                spine.set_visible(True)
                spine.set_color('#4a4a5a')
            else:
                spine.set_visible(False)
        self.ax_sub_top.set_facecolor('none')

    def _apply_styling(self):
        for ax in (self.ax_main, self.ax_sub):
            ax.set_facecolor('#252530')
            ax.tick_params(colors='#dcdcdc', which='both', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#4a4a5a')
            ax.grid(True, linestyle='--', alpha=0.3, color='#888888')

        self._apply_sub_top_styling()

        self.ax_main.set_ylabel("Absorbancja", color="#dcdcdc", fontsize=10)
        self.ax_sub.set_ylabel("Residuum / Profil", color="#dcdcdc", fontsize=9)
        self.ax_sub.set_xlabel("Liczba falowa [cm⁻¹]", color="#dcdcdc", fontsize=10)
        self.ax_sub_top.set_visible(False)

    def _connect_events(self):
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def prepare_axes(self, shared_x: bool = True):
        self._deconv_cache = None
        self.ax_main.cla()
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

    def redraw_all(self, *args):
        self._deconv_cache = None
        self.kinetic_traces.clear()
        self.last_kinetic_export_data = None
        self.ax_main.cla()
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

        visible_spectra = session.get_display_spectra()
        visible_series = session.get_visible_series() if session.active_context == "storage" else []
        active_spec = session.active_spectrum
        active_series = session.active_series if session.active_context == "storage" else None

        if not visible_spectra and not visible_series:
            self.canvas.draw_idle()
            return

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        n_spectra = len(visible_spectra)
        if n_spectra > 25:
            # Akceleracja dla dużych zestawów danych
            segments = []
            colors = []

            if n_spectra > 200:
                stride = 4
            elif n_spectra > 60:
                stride = 2
            else:
                stride = 1

            indices_to_draw = list(range(0, n_spectra, stride))
            if (n_spectra - 1) not in indices_to_draw:
                indices_to_draw.append(n_spectra - 1)

            use_colormap = n_spectra > 50
            cmap = matplotlib.colormaps['viridis']

            for idx in indices_to_draw:
                spec = visible_spectra[idx]
                x = spec.wavenumbers
                y = spec.intensities
                segments.append(np.column_stack([x, y]))

                if use_colormap:
                    colors.append(cmap(idx / max(1, n_spectra - 1)))
                else:
                    spec_color = spec.metadata.get("custom_color") or spec.metadata.get("display_color") or SPECTRAL_PALETTE[idx % len(SPECTRAL_PALETTE)]
                    colors.append(spec_color)

                min_x = min(min_x, np.min(x))
                max_x = max(max_x, np.max(x))
                min_y = min(min_y, np.min(y))
                max_y = max(max_y, np.max(y))

            lc = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.75)
            self.ax_main.add_collection(lc)

            if active_spec:
                act_color = active_spec.metadata.get("custom_color") or active_spec.metadata.get("display_color") or "#ffffff"
                self.ax_main.plot(active_spec.wavenumbers, active_spec.intensities,
                                  color=act_color, lw=2.4, zorder=5, label=f"Aktywne: {active_spec.name}")

        else:
            for idx, spec in enumerate(visible_spectra):
                is_active = (active_spec and spec.name == active_spec.name)
                spec_color = spec.metadata.get("custom_color") or spec.metadata.get("display_color") or SPECTRAL_PALETTE[idx % len(SPECTRAL_PALETTE)]

                # Zachowujemy właściwy kolor widma z wyraźnym wyróżnieniem aktywności grubością i zorderem
                lw = 2.2 if is_active else 1.2
                alpha = 1.0 if is_active else 0.75
                zorder = 5.0 if is_active else 2.5

                is_dft = getattr(spec, 'file_type', '') in ['gaussian', 'orca', 'dft'] or hasattr(spec, 'stick_freqs')
                ls = "--" if is_dft else "-"

                self.ax_main.plot(spec.wavenumbers, spec.intensities, label=spec.name,
                                  color=spec_color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

                min_x = min(min_x, np.min(spec.wavenumbers))
                max_x = max(max_x, np.max(spec.wavenumbers))
                min_y = min(min_y, np.min(spec.intensities))
                max_y = max(max_y, np.max(spec.intensities))

                if hasattr(spec, 'stick_freqs') and spec.stick_freqs is not None and len(spec.stick_freqs) > 0:
                    s = getattr(spec, 'scale_factor', 1.0)
                    f_scaled = spec.stick_freqs * s
                    m_y = np.max(spec.intensities) if len(spec.intensities) > 0 else 1.0
                    m_i = np.max(spec.stick_intens) if len(spec.stick_intens) > 0 else 1.0
                    h = (spec.stick_intens / (m_i if m_i > 0 else 1.0)) * m_y * 0.95
                    self.ax_main.vlines(f_scaled, 0, h, color=spec_color, alpha=0.75, lw=1.2, ls='-', zorder=2.8)

        series_colormaps = ['viridis', 'plasma', 'cividis', 'inferno']
        for s_idx, s_obj in enumerate(visible_series):
            is_series_active = (active_series and s_obj.name == active_series.name)
            cmap_name = series_colormaps[s_idx % len(series_colormaps)]
            cmap = matplotlib.colormaps[cmap_name]

            matrix_2d = s_obj.get_matrix_2d()
            n_cols = matrix_2d.shape[1]
            x = s_obj.wavenumbers

            min_x = min(min_x, np.min(x))
            max_x = max(max_x, np.max(x))
            min_y = min(min_y, np.min(matrix_2d))
            max_y = max(max_y, np.max(matrix_2d))

            if n_cols > 10:
                segments = [np.column_stack([x, matrix_2d[:, col_i]]) for col_i in range(n_cols)]
                colors = [cmap(col_i / max(1, n_cols - 1)) for col_i in range(n_cols)]
                alpha = 0.85 if is_series_active else 0.45
                lw = 1.2 if is_series_active else 0.8
                lc = LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha)
                self.ax_main.add_collection(lc)
            else:
                for col_i in range(n_cols):
                    self.ax_main.plot(x, matrix_2d[:, col_i],
                                      color=cmap(col_i / max(1, n_cols - 1)),
                                      lw=1.2 if is_series_active else 0.9,
                                      alpha=0.85 if is_series_active else 0.45)

        if min_x != float('inf') and max_x != float('-inf'):
            self.ax_main.set_xlim(max_x, min_x)
            self.ax_sub.set_xlim(max_x, min_x)
        if min_y != float('inf') and max_y != float('-inf'):
            y_margin = max((max_y - min_y) * 0.08, 0.01)
            self.ax_main.set_ylim(min_y - y_margin, max_y + y_margin)

        if n_spectra <= 12 and not visible_series:
            handles, labels = self.ax_main.get_legend_handles_labels()
            if handles:
                self.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                                    labelcolor='#ffffff', fontsize=8)

        self.ax_sub.axhline(0, color="#ee6c4d", lw=0.8, ls=":")

        self.initial_xlim = self.ax_main.get_xlim()
        self.initial_ylim = self.ax_main.get_ylim()
        self.canvas.draw_idle()

    # =========================================================================
    # SILNIK PRZEKROJÓW KINETYCZNYCH
    # =========================================================================

    def _get_evolutionary_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]]:
        active_series = session.active_series
        if not active_series:
            vis_series = session.get_visible_series()
            if vis_series:
                active_series = vis_series[0]

        if active_series:
            mat = active_series.get_matrix_2d()
            wn = active_series.wavenumbers
            n_points = mat.shape[1]

            if active_series.axis_n_values is not None and len(active_series.axis_n_values) == n_points:
                x_vals = np.asarray(active_series.axis_n_values, dtype=np.float64)
                unit_label = getattr(active_series, 'axis_n_name', 'Czas')
                lower_lbl = unit_label.lower()
                if "[" in unit_label:
                    x_label = unit_label
                elif "minut" in lower_lbl:
                    x_label = f"{unit_label} [min]"
                elif "sekund" in lower_lbl:
                    x_label = f"{unit_label} [s]"
                elif "czas" in lower_lbl:
                    x_label = f"{unit_label} [min]"
                else:
                    x_label = unit_label
                is_time_series = True
            else:
                x_vals = np.arange(1, n_points + 1, dtype=np.float64)
                x_label = "Numer widma w serii"
                is_time_series = False

            return wn, mat, x_vals, x_label, is_time_series

        vis_specs = session.get_display_spectra()
        if len(vis_specs) > 1:
            ref_wn = vis_specs[0].wavenumbers
            n_points = len(vis_specs)
            mat = np.zeros((len(ref_wn), n_points), dtype=np.float64)
            for i, s in enumerate(vis_specs):
                mat[:, i] = np.interp(ref_wn, s.wavenumbers, s.intensities)

            x_vals = np.arange(1, n_points + 1, dtype=np.float64)
            x_label = "Numer widma"
            is_time_series = False
            return ref_wn, mat, x_vals, x_label, is_time_series

        return None

    def _update_kinetic_slice(self, chosen_wn: float, add_trace: bool = False):
        dataset = self._get_evolutionary_dataset()
        if not dataset:
            return

        wavelengths, d_matrix, x_vals, x_label, is_time_series = dataset
        idx = int(np.abs(wavelengths - chosen_wn).argmin())
        exact_wn = float(wavelengths[idx])
        profile = d_matrix[idx, :]

        if not add_trace:
            for trace in self.kinetic_traces:
                if 'v_line' in trace and trace['v_line']:
                    try:
                        trace['v_line'].remove()
                    except ValueError:
                        pass
            self.kinetic_traces.clear()

        color = TRACE_PALETTE[len(self.kinetic_traces) % len(TRACE_PALETTE)]
        v_line = self.ax_main.axvline(x=exact_wn, color=color, linestyle='--', linewidth=1.4, zorder=6)

        self.kinetic_traces.append({
            'wn': exact_wn,
            'color': color,
            'v_line': v_line,
            'profile': profile
        })

        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

        n_points = len(profile)

        for trace in self.kinetic_traces:
            lbl = f"ν = {trace['wn']:.1f} cm⁻¹"
            self.ax_sub.plot(x_vals, trace['profile'], 'o-', color=trace['color'],
                             markersize=3.5, lw=1.4, label=lbl)

        self.ax_sub.set_xlabel(x_label, color='#dcdcdc', fontsize=10)
        self.ax_sub.set_ylabel("Absorbancja", color='#dcdcdc', fontsize=9)
        self.ax_sub.grid(True, linestyle='--', alpha=0.3, color='#888888')

        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        dx = (x_max - x_min) * 0.03 if x_max > x_min else 0.5
        self.ax_sub.set_xlim(x_min - dx, x_max + dx)

        if is_time_series:
            self.ax_sub.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            self.ax_sub.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

            self.ax_sub_top.set_visible(True)
            self.ax_sub_top.set_xlim(self.ax_sub.get_xlim())

            num_ticks = min(10, n_points)
            idx_ticks = np.unique(np.linspace(0, n_points - 1, num=num_ticks, dtype=int))
            self.ax_sub_top.set_xticks(x_vals[idx_ticks])
            self.ax_sub_top.set_xticklabels([str(i + 1) for i in idx_ticks], fontsize=8, color='#a0a0a0')
            self.ax_sub_top.set_xlabel("")
        else:
            self.ax_sub.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
            self.ax_sub.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            self.ax_sub_top.set_visible(False)

        self.ax_sub.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                           labelcolor='#ffffff', fontsize=8)

        self.ax_sub_initial_xlim = self.ax_sub.get_xlim()
        self.ax_sub_initial_ylim = self.ax_sub.get_ylim()

        self.last_kinetic_export_data = {
            'x_vals': x_vals,
            'x_label': x_label,
            'traces': [(t['wn'], t['profile']) for t in self.kinetic_traces]
        }

        self.kinetic_point_selected.emit(exact_wn)
        self.canvas.draw_idle()

    # =========================================================================
    # MENU KONTEKSTOWE I EKSPORT (PPM)
    # =========================================================================

    def _show_main_context_menu(self):
        menu = QMenu(self)
        act_reset_main = menu.addAction("🔍 Resetuj powiększenie (Zoom)")
        act_reset_main.triggered.connect(self._reset_main_zoom)

        act_clear_tr = menu.addAction("🗑 Wyczyść wskaźniki kinetyczne")
        act_clear_tr.setEnabled(bool(self.kinetic_traces))
        act_clear_tr.triggered.connect(self._clear_kinetic_traces)

        menu.addSeparator()
        act_copy_img = menu.addAction("🖼 Kopiuj obraz wykresu do schowka")
        act_copy_img.triggered.connect(self._copy_figure_to_clipboard)

        menu.exec(QCursor.pos())

    def _show_sub_context_menu(self):
        menu = QMenu(self)
        has_data = bool(self.last_kinetic_export_data and self.kinetic_traces)

        act_copy = menu.addAction("📋 Kopiuj profil do schowka (Origin/Excel)")
        act_copy.setEnabled(has_data)
        act_copy.triggered.connect(self._copy_kinetic_to_clipboard)

        act_csv = menu.addAction("💾 Zapisz profil jako CSV...")
        act_csv.setEnabled(has_data)
        act_csv.triggered.connect(self._export_kinetic_to_csv)

        menu.addSeparator()
        act_reset_sub = menu.addAction("🔍 Resetuj powiększenie dolnego panelu")
        act_reset_sub.triggered.connect(self._reset_sub_zoom)

        act_clear_tr = menu.addAction("🗑 Wyczyść profile kinetyczne")
        act_clear_tr.setEnabled(bool(self.kinetic_traces))
        act_clear_tr.triggered.connect(self._clear_kinetic_traces)

        menu.exec(QCursor.pos())

    def _copy_figure_to_clipboard(self):
        pixmap = self.canvas.grab()
        QApplication.clipboard().setPixmap(pixmap)

    def _copy_kinetic_to_clipboard(self):
        if not self.last_kinetic_export_data:
            return

        x_vals = self.last_kinetic_export_data['x_vals']
        x_label = self.last_kinetic_export_data['x_label']
        traces = self.last_kinetic_export_data['traces']

        header = [x_label] + [f"A_{wn:.1f}" for wn, _ in traces]
        rows = ["\t".join(header)]

        for i, x in enumerate(x_vals):
            row = [f"{x:.4f}".replace('.', ',')]
            for _, prof in traces:
                row.append(f"{prof[i]:.6f}".replace('.', ','))
            rows.append("\t".join(row))

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(rows))

    def _export_kinetic_to_csv(self):
        if not self.last_kinetic_export_data:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz profile kinetyczne", "kinetyka.csv", "Plik CSV (*.csv);;Wszystkie (*.*)"
        )
        if not path:
            return

        try:
            x_vals = self.last_kinetic_export_data['x_vals']
            x_label = self.last_kinetic_export_data['x_label']
            traces = self.last_kinetic_export_data['traces']

            header = [x_label] + [f"A_{wn:.1f}cm-1" for wn, _ in traces]
            lines = [";".join(header)]

            for i, x in enumerate(x_vals):
                row = [f"{x:.4f}".replace('.', ',')]
                for _, prof in traces:
                    row.append(f"{prof[i]:.6f}".replace('.', ','))
                lines.append(";".join(row))

            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

            QMessageBox.information(self, tr("app.info"), "Profil kinetyczny został pomyślnie zapisany do CSV.")
        except Exception as e:
            QMessageBox.critical(self, tr("app.error"), f"Błąd zapisu CSV:\n{e}")

    def _reset_main_zoom(self):
        if self.initial_xlim and self.initial_ylim:
            self.ax_main.set_xlim(self.initial_xlim)
            self.ax_main.set_ylim(self.initial_ylim)
            self.canvas.draw_idle()

    def _reset_sub_zoom(self):
        if self.ax_sub_initial_xlim and self.ax_sub_initial_ylim:
            self.ax_sub.set_xlim(self.ax_sub_initial_xlim)
            self.ax_sub.set_ylim(self.ax_sub_initial_ylim)
            if self.ax_sub_top.get_visible():
                self.ax_sub_top.set_xlim(self.ax_sub_initial_xlim)
            self.canvas.draw_idle()

    def _clear_kinetic_traces(self):
        for trace in self.kinetic_traces:
            if 'v_line' in trace and trace['v_line']:
                try:
                    trace['v_line'].remove()
                except ValueError:
                    pass
        self.kinetic_traces.clear()
        self.last_kinetic_export_data = None
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()
        self.canvas.draw_idle()

    # =========================================================================
    # OBSŁUGA ZDARZEŃ MYSZY
    # =========================================================================

    def _on_press(self, event):
        if event.button in [2, 3]:
            if event.inaxes in (self.ax_sub, self.ax_sub_top):
                self._show_sub_context_menu()
                return
            elif event.inaxes == self.ax_main:
                self._show_main_context_menu()
                return

        if event.button == 1 and getattr(event, 'dblclick', False):
            if event.inaxes == self.ax_main:
                self._reset_main_zoom()
                return
            elif event.inaxes in (self.ax_sub, self.ax_sub_top):
                self._reset_sub_zoom()
                return

        if event.inaxes in (self.ax_sub, self.ax_sub_top) and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                self.zoom_target_ax = self.ax_sub
                self.zoom_start = (event.xdata, event.ydata)
                self.zoom_rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                           fill=True, facecolor='#00e5ff', alpha=0.2,
                                           edgecolor='#00e5ff', linestyle='--', lw=1)
                self.ax_sub.add_patch(self.zoom_rect)
                return

        if event.inaxes != self.ax_main:
            return

        modifiers = QApplication.keyboardModifiers()
        is_shift = bool(modifiers & Qt.ShiftModifier)
        is_other_mod = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier | Qt.AltModifier))

        if event.button == 1 and (is_other_mod or self.interaction_mode == "add_peak"):
            if event.xdata is not None and event.ydata is not None:
                self.peak_add_requested.emit(float(event.xdata), float(event.ydata))
            return

        if event.button == 1 and is_shift and event.xdata is not None and self._get_evolutionary_dataset():
            self._update_kinetic_slice(float(event.xdata), add_trace=True)
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.zoom_target_ax = self.ax_main
            self.zoom_start = (event.xdata, event.ydata)
            self.zoom_rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                       fill=True, facecolor='#ee6c4d', alpha=0.2,
                                       edgecolor='#ee6c4d', linestyle='--', lw=1)
            self.ax_main.add_patch(self.zoom_rect)

    def _on_motion(self, event):
        if event.button != 1 or self.zoom_start is None or self.zoom_target_ax is None:
            return

        valid_inaxes = (self.zoom_target_ax == self.ax_main and event.inaxes == self.ax_main) or \
                       (self.zoom_target_ax == self.ax_sub and event.inaxes in (self.ax_sub, self.ax_sub_top))
        if not valid_inaxes or event.xdata is None or event.ydata is None:
            return

        x0, y0 = self.zoom_start
        self.zoom_rect.set_width(event.xdata - x0)
        self.zoom_rect.set_height(event.ydata - y0)
        self.zoom_rect.set_xy((x0, y0))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self.zoom_rect is not None:
            self.zoom_rect.remove()
            self.zoom_rect = None

        if event.button != 1 or self.zoom_start is None or self.zoom_target_ax is None:
            self.zoom_start = None
            self.zoom_target_ax = None
            return

        x0, y0 = self.zoom_start
        x1, y1 = event.xdata, event.ydata
        target_ax = self.zoom_target_ax
        self.zoom_start = None
        self.zoom_target_ax = None

        if x1 is None or y1 is None:
            return

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if target_ax == self.ax_sub:
            span_x = abs(self.ax_sub.get_xlim()[1] - self.ax_sub.get_xlim()[0])
            span_y = abs(self.ax_sub.get_ylim()[1] - self.ax_sub.get_ylim()[0])
            if dx > (span_x * 0.02) or dy > (span_y * 0.02):
                self.ax_sub.set_xlim(min(x0, x1), max(x0, x1))
                self.ax_sub.set_ylim(min(y0, y1), max(y0, y1))
                if self.ax_sub_top.get_visible():
                    self.ax_sub_top.set_xlim(self.ax_sub.get_xlim())
                self.canvas.draw_idle()
            return

        if dx > 3.0 or dy > 0.01:
            current_xlim = self.ax_main.get_xlim()
            is_inverted = current_xlim[0] > current_xlim[1]
            if is_inverted:
                self.ax_main.set_xlim(max(x0, x1), min(x0, x1))
            else:
                self.ax_main.set_xlim(min(x0, x1), max(x0, x1))
            self.ax_main.set_ylim(min(y0, y1), max(y0, y1))
            self.canvas.draw_idle()
        else:
            active = session.active_spectrum
            if hasattr(active, 'stick_freqs') and active.stick_freqs is not None and len(active.stick_freqs) > 0:
                s = getattr(active, 'scale_factor', 1.0)
                scaled_freqs = active.stick_freqs * s
                dists = np.abs(scaled_freqs - x1)
                closest_idx = int(np.argmin(dists))
                if dists[closest_idx] <= 15.0:
                    self.dft_mode_selected.emit(closest_idx)
                    return

            if self._get_evolutionary_dataset():
                self._update_kinetic_slice(float(x1), add_trace=False)

    # =========================================================================
    # NAKŁADKI DEKONWOLUCJI I DEKOMPOZYCJI
    # =========================================================================

    def update_deconv_overlay(
        self,
        active_spec: SpectrumData,
        curves: Optional[Dict[str, Any]],
        peaks: List[Dict[str, Any]],
        show_deriv: bool = False,
        deriv_data: Optional[tuple] = None
    ):
        if (not curves or 'components' not in curves or not peaks) and not show_deriv:
            self.redraw_all()
            return

        self._deconv_cache = {
            'spec': active_spec,
            'curves': curves,
            'peaks': peaks,
            'show_deriv': show_deriv,
            'deriv_data': deriv_data
        }

        prev_xlim = self.ax_main.get_xlim() if self.initial_xlim else None
        prev_ylim = self.ax_main.get_ylim() if self.initial_ylim else None

        self.ax_main.cla()
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

        if active_spec is None:
            self.canvas.draw_idle()
            return

        self.ax_main.plot(active_spec.wavenumbers, active_spec.intensities,
                          label="Eksperyment", color="#ffffff", lw=1.4, zorder=2)

        if curves and 'components' in curves and peaks:
            valid_area_amide = sum(
                p['area'] for p in peaks
                if 1600 <= p['center'] <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]
            )
            y_max_plot = np.max(active_spec.intensities) if len(active_spec.intensities) > 0 else 1.0

            for idx, comp in enumerate(curves['components']):
                p = peaks[idx] if idx < len(peaks) else {}
                color = p.get('color', '#f4a261')
                p_id = p.get('id', comp.get('id', f"p{idx+1}"))

                self.ax_main.plot(active_spec.wavenumbers, comp['y'], lw=1.2, ls="--",
                                  color=color, alpha=0.85, zorder=3)
                self.ax_main.fill_between(active_spec.wavenumbers, comp['y'],
                                          alpha=0.12, color=color, zorder=2.5)

                center_x = p.get('center', 1650.0)
                struct = p.get('structure', 'Other')
                struct_short = struct[:4]

                if 1600 <= center_x <= 1700 and struct not in ["Side Chains", "Other", "Inny"]:
                    perc = (p['area'] / valid_area_amide * 100.0) if valid_area_amide > 0 else 0.0
                    label_text = f"{p_id} {struct_short} ({perc:.1f}%)"
                else:
                    label_text = f"{p_id} {struct_short}"

                peak_h = np.max(comp['y']) if len(comp['y']) > 0 else 0.0
                self.ax_main.text(center_x, peak_h + (0.02 * y_max_plot), label_text,
                                  color=color, fontsize=8, ha='center', va='bottom',
                                  rotation=90, zorder=4)

            if 'total' in curves:
                self.ax_main.plot(active_spec.wavenumbers, curves['total'],
                                  label="Fit Sum", color="#ee6c4d", lw=1.8, zorder=3.5)

            if show_deriv and deriv_data is not None:
                dx, dy = deriv_data
                self.ax_sub.plot(dx, dy, color="#00ffcc", lw=1.1, label="-d²A/dν²")
                self.ax_sub.set_ylabel("-d²A/dν²", color="#dcdcdc", fontsize=9)
            elif 'residuals' in curves:
                self.ax_sub.plot(active_spec.wavenumbers, curves['residuals'],
                                 color="#98c1d9", lw=0.9, label="Residuum")
                self.ax_sub.set_ylabel("Residuum", color="#dcdcdc", fontsize=9)

            self.ax_sub.axhline(0, color="#ee6c4d", lw=0.6, ls=":")

        self.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                            labelcolor='#ffffff', fontsize=8)

        if prev_xlim is not None and prev_ylim is not None and abs(prev_xlim[0] - prev_xlim[1]) > 1:
            self.ax_main.set_xlim(prev_xlim)
            self.ax_main.set_ylim(prev_ylim)
            self.ax_sub.set_xlim(prev_xlim)
        else:
            min_x = np.min(active_spec.wavenumbers)
            max_x = np.max(active_spec.wavenumbers)
            self.ax_main.set_xlim(max_x, min_x)
            self.ax_sub.set_xlim(max_x, min_x)
            margin = (np.max(active_spec.intensities) - np.min(active_spec.intensities)) * 0.1
            self.ax_main.set_ylim(np.min(active_spec.intensities) - margin,
                                  np.max(active_spec.intensities) + margin)

        self.initial_xlim = (np.max(active_spec.wavenumbers), np.min(active_spec.wavenumbers))
        margin = (np.max(active_spec.intensities) - np.min(active_spec.intensities)) * 0.1
        self.initial_ylim = (np.min(active_spec.intensities) - margin,
                             np.max(active_spec.intensities) + margin)
        self.canvas.draw_idle()

    def update_decomposition_overlay(
            self,
            target_spec: SpectrumData,
            fitted_y: np.ndarray,
            components: List[np.ndarray],
            comp_names: List[str],
            peaks_curves: List[np.ndarray],
            excess_residual: np.ndarray,
            fit_residual: np.ndarray
    ):
        self._deconv_cache = None
        self.ax_main.cla()
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

        if target_spec is None:
            self.canvas.draw_idle()
            return

        wn = target_spec.wavenumbers

        self.ax_main.plot(wn, target_spec.intensities, label="W₀ (Eksperyment)",
                          color="#ffffff", lw=1.6, zorder=3)
        self.ax_main.plot(wn, fitted_y, label="Fit Całkowity",
                          color="#ee6c4d", lw=1.4, ls="--", zorder=4)

        palette = ['#2a9d8f', '#f4a261', '#98c1d9', '#e76f51', '#b5e48c', '#d88c9a']
        for idx, comp in enumerate(components):
            name = comp_names[idx] if idx < len(comp_names) else f"W{idx + 1}"
            c = palette[idx % len(palette)]
            self.ax_main.plot(wn, comp, label=f"c·{name}", color=c, lw=1.2, ls=":", alpha=0.9, zorder=2.5)

        for idx, p_curve in enumerate(peaks_curves):
            self.ax_main.plot(wn, p_curve, color="#e0aaff", lw=1.0, ls="-.", alpha=0.8,
                              label=f"Pik GxL {idx + 1}" if idx == 0 else None, zorder=2.4)

        self.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                            labelcolor='#ffffff', fontsize=8)

        self.ax_sub.plot(wn, excess_residual, color="#00ffcc", lw=1.1, label="Residuum nadmiarowe (W₀ - Σ cᵢWᵢ)")
        self.ax_sub.plot(wn, fit_residual, color="#6c757d", lw=0.8, ls=":", label="Residuum dopasowania")
        self.ax_sub.axhline(0, color="#ee6c4d", lw=0.6, ls=":")
        self.ax_sub.set_ylabel("Residua", color="#dcdcdc", fontsize=9)
        self.ax_sub.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                           labelcolor='#ffffff', fontsize=7)

        self.ax_main.set_xlim(np.max(wn), np.min(wn))
        self.ax_sub.set_xlim(np.max(wn), np.min(wn))
        y_span = np.max(target_spec.intensities) - np.min(target_spec.intensities)
        margin = max(y_span * 0.1, 0.01)
        self.ax_main.set_ylim(np.min(target_spec.intensities) - margin,
                              np.max(target_spec.intensities) + margin)

        self.initial_xlim = (np.max(wn), np.min(wn))
        self.initial_ylim = self.ax_main.get_ylim()
        self.canvas.draw_idle()

    def _on_active_table_changed(self, table):
        if session.active_context == "table" and table is not None:
            self.draw_table(table)

    def draw_table(self, table: Optional[DataTable] = None):
        """Renderuje tabelę DataTable (dla profili kinetycznych: wykres X-Y; dla raportu parametrów: czytelna karta)."""
        if table is None:
            table = session.active_table
        if not table or table.n_rows == 0:
            return

        self._deconv_cache = None
        self.kinetic_traces.clear()
        self.last_kinetic_export_data = None
        self.ax_main.cla()
        self.ax_sub.cla()
        self.ax_sub_top.cla()
        self._apply_styling()

        clean_name = table.name.replace("📊", "").strip()

        # BRANCH A: Tabela jest zestawieniem parametrów HDO / Fit (klucz-wartość)
        if table.is_parameter_report:
            row_labels = table.metadata.get("row_labels", [])
            val_col = next(iter(table.y_columns.values())) if table.y_columns else []

            summary_lines = [
                f"TABELA PARAMETRÓW: {clean_name.upper()}",
                "─" * 45
            ]
            for r_idx, lbl in enumerate(row_labels):
                if r_idx < len(val_col):
                    v = val_col[r_idx]
                    if any(k in lbl for k in ['R2', 'RMSE', 'Chi2_red', 'RSS', 'N_multiplier', 'MC_Iterations']):
                        summary_lines.append(f"  • {lbl:<18}: {v:.6g}")

            summary_lines.append("─" * 45)
            summary_lines.append("💡 Kliknij dwukrotnie tabelę w drzewie projektu, aby wyświetlić pełną tabelę wartości.")

            full_card_text = "\n".join(summary_lines)

            self.ax_main.text(
                0.08, 0.5, full_card_text,
                color="#00e5ff", fontfamily="monospace", fontsize=10,
                va='center', ha='left', transform=self.ax_main.transAxes
            )
            self.ax_main.set_xticks([])
            self.ax_main.set_yticks([])
            self.ax_main.set_ylabel("")

            self.ax_sub.set_ylabel("Status", color="#dcdcdc", fontsize=9)
            self.ax_sub.text(
                0.5, 0.5,
                f"Zestawienie Parametrów: {clean_name}  |  {table.n_rows} parametrów  |  PPM: Kopiuj TSV / Zapisz CSV",
                color="#a8dadc", ha='center', va='center',
                transform=self.ax_sub.transAxes, fontsize=10
            )
            self.canvas.draw_idle()
            return

        # BRANCH B: Standardowy profil liczbowy (np. kinetyka, stężenia MCR, Scores)
        x = table.x_values
        palette = ['#00e5ff', '#ee6c4d', '#2a9d8f', '#f4a261', '#e0aaff', '#ffd166', '#ff4d6d']

        for idx, (col_name, y_vals) in enumerate(table.y_columns.items()):
            col = palette[idx % len(palette)]
            self.ax_main.plot(x, y_vals, 'o-', color=col, lw=1.6, markersize=4.5,
                              alpha=0.9, label=col_name)

        self.ax_main.set_xlabel(table.x_label, color="#dcdcdc", fontsize=10)
        self.ax_main.set_ylabel(table.y_label, color="#dcdcdc", fontsize=10)
        self.ax_main.grid(True, linestyle='--', alpha=0.3, color='#888888')

        min_x, max_x = float(np.min(x)), float(np.max(x))
        dx = (max_x - min_x) * 0.04 if max_x > min_x else 0.5
        self.ax_main.set_xlim(min_x - dx, max_x + dx)

        all_y = [v for v in table.y_columns.values() if len(v) > 0]
        if all_y:
            concat_y = np.concatenate(all_y)
            min_y, max_y = float(np.nanmin(concat_y)), float(np.nanmax(concat_y))
            dy = max((max_y - min_y) * 0.08, 0.01)
            self.ax_main.set_ylim(min_y - dy, max_y + dy)

        self.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                            labelcolor='#ffffff', fontsize=8)

        self.initial_xlim = self.ax_main.get_xlim()
        self.initial_ylim = self.ax_main.get_ylim()

        self.ax_sub.set_ylabel("Status", color="#dcdcdc", fontsize=9)
        self.ax_sub.text(
            0.5, 0.5,
            f"Tabela Danych: {clean_name}  |  {table.n_rows} punktów × {table.n_cols} serii",
            color="#a8dadc", ha='center', va='center',
            transform=self.ax_sub.transAxes, fontsize=10
        )
        self.canvas.draw_idle()