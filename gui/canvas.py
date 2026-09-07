"""
gui/canvas.py
Interaktywny dwupanelowy wykres Matplotlib (Góra: Widma, Dół: Residua/Kinetyka).
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Qt, Signal
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import logging
from typing import Optional, List, Dict, Any

from core.session import session
from core.models import SpectrumData

logger = logging.getLogger("SpectroHub")


class SpectroCanvas(QWidget):
    """Zintegrowany wykres z obsługą Smart Zoom, interakcji FTIR i macOS."""

    peak_add_requested = Signal(float, float)
    dft_mode_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 6), facecolor='#1e1e24')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout.addWidget(self.canvas)

        self.ax_main = self.fig.add_subplot(4, 1, (1, 3))
        self.ax_sub = self.fig.add_subplot(4, 1, 4, sharex=self.ax_main)
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.15)

        self.zoom_start = None
        self.zoom_rect = None
        self.initial_xlim = None
        self.initial_ylim = None

        self.interaction_mode = "normal"
        self._deconv_cache: Optional[Dict[str, Any]] = None

        self._apply_styling()
        self._connect_events()

        session.spectrum_added.connect(self.redraw_all)
        session.spectrum_removed.connect(self.redraw_all)
        session.visibility_changed.connect(self.redraw_all)
        session.active_spectrum_changed.connect(self.redraw_all)
        session.session_cleared.connect(self.redraw_all)

    def _apply_styling(self):
        for ax in (self.ax_main, self.ax_sub):
            ax.set_facecolor('#252530')
            ax.tick_params(colors='#dcdcdc', which='both', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#4a4a5a')
            ax.grid(True, linestyle='--', alpha=0.3, color='#888888')

        self.ax_main.set_ylabel("Absorbancja", color="#dcdcdc", fontsize=10)
        self.ax_sub.set_ylabel("Residuum", color="#dcdcdc", fontsize=9)
        self.ax_sub.set_xlabel("Liczba falowa [cm⁻¹]", color="#dcdcdc", fontsize=10)

    def _connect_events(self):
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def prepare_axes(self, shared_x: bool = True):
        """Konfiguruje dolną oś X (współdzielona dla FTIR / niezależna dla profili stężeń)."""
        self._deconv_cache = None
        self.ax_main.clear()
        self.ax_sub.remove()
        if shared_x:
            self.ax_sub = self.fig.add_subplot(4, 1, 4, sharex=self.ax_main)
        else:
            self.ax_sub = self.fig.add_subplot(4, 1, 4)
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.25)
        self._apply_styling()

    def redraw_all(self, *args):
        self.prepare_axes(shared_x=True)
        self._deconv_cache = None
        self.ax_main.clear()
        self.ax_sub.clear()
        self._apply_styling()

        visible_spectra = session.get_visible_spectra()
        active = session.active_spectrum

        if not visible_spectra:
            self.canvas.draw_idle()
            return

        all_wns = []
        palette = ['#ee6c4d', '#98c1d9', '#3d5a80', '#2a9d8f', '#e76f51', '#f4a261']

        for idx, spec in enumerate(visible_spectra):
            is_active = (active and spec.name == active.name)
            color = "#ffffff" if is_active else palette[idx % len(palette)]
            lw = 1.6 if is_active else 1.0
            alpha = 1.0 if is_active else 0.75

            self.ax_main.plot(spec.wavenumbers, spec.intensities, label=spec.name,
                              color=color, lw=lw, alpha=alpha)
            all_wns.append(spec.wavenumbers)

            if hasattr(spec, 'stick_freqs') and spec.stick_freqs is not None and len(spec.stick_freqs) > 0:
                s = getattr(spec, 'scale_factor', 1.0)
                f_scaled = spec.stick_freqs * s
                max_y = np.max(spec.intensities) if len(spec.intensities) > 0 and np.max(spec.intensities) > 0 else 1.0
                max_i = np.max(spec.stick_intens) if len(spec.stick_intens) > 0 and np.max(
                    spec.stick_intens) > 0 else 1.0
                h = (spec.stick_intens / max_i) * max_y * 0.95
                self.ax_main.vlines(f_scaled, 0, h, color=color, alpha=0.75, lw=1.2, ls='-', zorder=2.8)

        if all_wns:
            min_x = min(np.min(w) for w in all_wns)
            max_x = max(np.max(w) for w in all_wns)
            self.ax_main.set_xlim(max_x, min_x)

        self.ax_main.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                            labelcolor='#ffffff', fontsize=8)
        self.ax_sub.axhline(0, color="#ee6c4d", lw=0.8, ls=":")

        self.initial_xlim = self.ax_main.get_xlim()
        self.initial_ylim = self.ax_main.get_ylim()
        self.canvas.draw_idle()

    def update_deconv_overlay(
        self,
        active_spec: SpectrumData,
        curves: Optional[Dict[str, Any]],
        peaks: List[Dict[str, Any]],
        show_deriv: bool = False,
        deriv_data: Optional[tuple] = None
    ):
        self._deconv_cache = {
            'spec': active_spec,
            'curves': curves,
            'peaks': peaks,
            'show_deriv': show_deriv,
            'deriv_data': deriv_data
        }

        prev_xlim = self.ax_main.get_xlim() if self.initial_xlim else None
        prev_ylim = self.ax_main.get_ylim() if self.initial_ylim else None

        self.ax_main.clear()
        self.ax_sub.clear()
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
        else:
            min_x = np.min(active_spec.wavenumbers)
            max_x = np.max(active_spec.wavenumbers)
            self.ax_main.set_xlim(max_x, min_x)
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
        """Wyświetla wyniki dekompozycji Monte Carlo / HDO na wykresie dwupanelowym."""
        self._deconv_cache = None
        self.ax_main.clear()
        self.ax_sub.clear()
        self._apply_styling()

        if target_spec is None:
            self.canvas.draw_idle()
            return

        wn = target_spec.wavenumbers

        # 1. Panel górny: Eksperyment, Fit całkowity oraz składowe
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

        # 2. Panel dolny: Czyste residuum nadmiarowe oraz błąd fitu
        self.ax_sub.plot(wn, excess_residual, color="#00ffcc", lw=1.1, label="Residuum nadmiarowe (W₀ - Σ cᵢWᵢ)")
        self.ax_sub.plot(wn, fit_residual, color="#6c757d", lw=0.8, ls=":", label="Residuum dopasowania")
        self.ax_sub.axhline(0, color="#ee6c4d", lw=0.6, ls=":")
        self.ax_sub.set_ylabel("Residua", color="#dcdcdc", fontsize=9)
        self.ax_sub.legend(loc='upper right', facecolor='#2b2b36', edgecolor='#4a4a5a',
                           labelcolor='#ffffff', fontsize=7)

        # Ustalenie granic osi X (malejąco dla FTIR)
        self.ax_main.set_xlim(np.max(wn), np.min(wn))
        y_span = np.max(target_spec.intensities) - np.min(target_spec.intensities)
        margin = max(y_span * 0.1, 0.01)
        self.ax_main.set_ylim(np.min(target_spec.intensities) - margin,
                              np.max(target_spec.intensities) + margin)

        self.initial_xlim = (np.max(wn), np.min(wn))
        self.initial_ylim = self.ax_main.get_ylim()
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax_main:
            return

        if event.button in [2, 3]:
            if self.initial_xlim and self.initial_ylim:
                self.ax_main.set_xlim(self.initial_xlim)
                self.ax_main.set_ylim(self.initial_ylim)
                self.canvas.draw_idle()
            return

        # Rozpoznanie Shift, Ctrl oraz klawisza Command (⌘ - MetaModifier na macOS)
        modifiers = QApplication.keyboardModifiers()
        is_modifier_held = bool(modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.MetaModifier | Qt.AltModifier))

        if event.button == 1 and (is_modifier_held or self.interaction_mode == "add_peak"):
            if event.xdata is not None and event.ydata is not None:
                self.peak_add_requested.emit(float(event.xdata), float(event.ydata))
            return

        if event.button == 1:
            self.zoom_start = (event.xdata, event.ydata)
            self.zoom_rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                       fill=True, facecolor='#ee6c4d', alpha=0.2,
                                       edgecolor='#ee6c4d', linestyle='--', lw=1)
            self.ax_main.add_patch(self.zoom_rect)
        # Detekcja kliknięcia w pałeczkę drgań DFT dla aktywnego widma
        active = session.active_spectrum
        if event.button == 1 and not is_modifier_held and hasattr(active,
                                                                  'stick_freqs') and active.stick_freqs is not None:
            s = getattr(active, 'scale_factor', 1.0)
            scaled_freqs = active.stick_freqs * s
            dists = np.abs(scaled_freqs - event.xdata)
            closest_idx = int(np.argmin(dists))

            # Tolerancja kliknięcia: 15 cm^-1
            if dists[closest_idx] <= 15.0:
                self.dft_mode_selected.emit(closest_idx)
                return

    def _on_motion(self, event):
        if event.button != 1 or self.zoom_start is None or event.inaxes != self.ax_main:
            return
        x0, y0 = self.zoom_start
        if event.xdata is None or event.ydata is None:
            return

        self.zoom_rect.set_width(event.xdata - x0)
        self.zoom_rect.set_height(event.ydata - y0)
        self.zoom_rect.set_xy((x0, y0))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self.zoom_rect is not None:
            self.zoom_rect.remove()
            self.zoom_rect = None

        if event.button != 1 or self.zoom_start is None:
            self.zoom_start = None
            return

        x0, y0 = self.zoom_start
        x1, y1 = event.xdata, event.ydata
        self.zoom_start = None

        if x1 is None or y1 is None:
            return

        if abs(x1 - x0) > 2.0 and abs(y1 - y0) > 0.001:
            self.ax_main.set_xlim(max(x0, x1), min(x0, x1))
            self.ax_main.set_ylim(min(y0, y1), max(y0, y1))
            self.canvas.draw_idle()