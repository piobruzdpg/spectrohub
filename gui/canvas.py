"""
gui/canvas.py
Interaktywny dwupanelowy wykres Matplotlib (Góra: Widma, Dół: Residua/Kinetyka).
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import logging

from core.session import session
from core.models import SpectrumData

logger = logging.getLogger("SpectroHub")


class SpectroCanvas(QWidget):
    """Zintegrowany wykres z obsługą Smart Zoom i synchronizacją FTIR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Inicjalizacja figury pod ciemny motyw
        self.fig = Figure(figsize=(8, 6), facecolor='#1e1e24')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout.addWidget(self.canvas)

        # Dwa panele: ax_main (75% wysokości), ax_sub (25% wysokości)
        self.ax_main = self.fig.add_subplot(4, 1, (1, 3))
        self.ax_sub = self.fig.add_subplot(4, 1, 4, sharex=self.ax_main)
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.15)

        # Zmienne obsługi myszy (Smart Zoom)
        self.zoom_start = None
        self.zoom_rect = None
        self.initial_xlim = None
        self.initial_ylim = None

        self._apply_styling()
        self._connect_events()

        # Podłączenie pod sygnały sesji
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

    def redraw_all(self, *args):
        """Przelicza i odświeża linie na wykresie na podstawie stanu ProjectSession."""
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

        # Ustalenie domyślnych granic osi FTIR (malejąco: np. 4000 -> 400 cm^-1)
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

    # --- Obsługa Smart Zoom & Reset ---
    def _on_press(self, event):
        if event.inaxes != self.ax_main:
            return

        # Prawy przycisk myszy (PPM) -> Reset widoku
        if event.button in [2, 3]:
            if self.initial_xlim and self.initial_ylim:
                self.ax_main.set_xlim(self.initial_xlim)
                self.ax_main.set_ylim(self.initial_ylim)
                self.canvas.draw_idle()
            return

        # Lewy przycisk myszy (LPM) -> Początek zaznaczenia prostokąta
        if event.button == 1:
            self.zoom_start = (event.xdata, event.ydata)
            self.zoom_rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                       fill=True, facecolor='#ee6c4d', alpha=0.2,
                                       edgecolor='#ee6c4d', linestyle='--', lw=1)
            self.ax_main.add_patch(self.zoom_rect)

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

        # Jeśli zaznaczono realny obszar (a nie pojedyncze kliknięcie)
        if abs(x1 - x0) > 2.0 and abs(y1 - y0) > 0.001:
            # Wymuszenie zachowania malejącej osi FTIR
            self.ax_main.set_xlim(max(x0, x1), min(x0, x1))
            self.ax_main.set_ylim(min(y0, y1), max(y0, y1))
            self.canvas.draw_idle()