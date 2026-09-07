"""
gui/vibration_dialog.py
Pływające okno wizualizacji drgań 3D cząsteczki z silnikiem PyVista (QtInteractor).
Obsługuje animację modów w czasie rzeczywistym, CPK i dynamiczne wiązania.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QWidget
)
from PySide6.QtCore import Qt, QTimer
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from core.i18n import tr
from algorithms.quantum.ped_engine import COVALENT_RADII, ELEMENT_SYMBOLS

CPK_COLORS = {
    1: "#FFFFFF", 6: "#333333", 7: "#3050F8", 8: "#FF0D0D",
    9: "#90E050", 14: "#DAA520", 15: "#FF8000", 16: "#FFFF30",
    17: "#1FF01F", 35: "#A62929", 53: "#940094"
}


class VibrationVisualizerDialog(QDialog):
    """Niezależne okno wizualizacji drgań normalnych w 3D."""

    def __init__(self, parent=None, atomnos=None, coords=None, freqs=None, intens=None, Lx=None, initial_mode: int = 0):
        super().__init__(parent)
        self.setWindowTitle("SpectroHub — Wizualizator Drgań 3D")
        self.resize(750, 600)
        self.setMinimumSize(500, 400)

        # Bezpieczna konwersja tablic NumPy (unikanie 'freqs or []')
        self.atomnos = np.asarray(atomnos if atomnos is not None else [], dtype=int)
        self.coords_orig = np.asarray(coords if coords is not None else np.empty((0, 3)), dtype=float)
        self.freqs = np.asarray(freqs if freqs is not None else [], dtype=float)
        self.intens = np.asarray(intens if intens is not None else [], dtype=float)
        self.Lx = np.asarray(Lx if Lx is not None else np.empty((0, 0)), dtype=float)
        self.current_mode = initial_mode

        self.is_playing = True
        self.frame_step = 0
        self.n_frames = 32
        self.amplitude = 0.8
        self.speed_ms = 35

        self.anim_atom_actors = []
        self.anim_bond_actor = None
        self._calculate_bonds()

        self._init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_animation_tick)
        if self.is_playing:
            self.timer.start(self.speed_ms)

    def _calculate_bonds(self):
        self.bonds = []
        n_atoms = len(self.atomnos)
        for i in range(n_atoms):
            r_i = COVALENT_RADII.get(self.atomnos[i], 0.8)
            for j in range(i + 1, n_atoms):
                r_j = COVALENT_RADII.get(self.atomnos[j], 0.8)
                dist = np.linalg.norm(self.coords_orig[i] - self.coords_orig[j])
                if dist <= (r_i + r_j + 0.35):
                    self.bonds.append((i, j))

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Górny pasek nawigacji modów
        top_bar = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Poprzedni")
        self.btn_prev.clicked.connect(self._prev_mode)

        self.lbl_mode_info = QLabel()
        self.lbl_mode_info.setAlignment(Qt.AlignCenter)
        self.lbl_mode_info.setStyleSheet("font-weight: bold; font-size: 12px; color: #00E5FF;")

        self.btn_next = QPushButton("Następny ▶")
        self.btn_next.clicked.connect(self._next_mode)

        top_bar.addWidget(self.btn_prev)
        top_bar.addWidget(self.lbl_mode_info, 1)
        top_bar.addWidget(self.btn_next)
        layout.addLayout(top_bar)

        # 2. Widok 3D PyVista
        if HAS_PYVISTA and len(self.atomnos) > 0:
            self.plotter = QtInteractor(self)
            self.plotter.set_background('#141418')
            layout.addWidget(self.plotter.interactor, 1)
            self._setup_pyvista_scene()
        else:
            lbl_fallback = QLabel("Wizualizator 3D wymaga biblioteki pyvistaqt lub brakuje geometrii.")
            lbl_fallback.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl_fallback, 1)

        # 3. Dolny pasek kontroli (Play/Pause, Amplituda, Tempo)
        bottom_bar = QHBoxLayout()
        self.btn_play = QPushButton("⏸ Pauza")
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_reset_cam = QPushButton("🔍 Reset Kamery")
        if HAS_PYVISTA and hasattr(self, 'plotter'):
            self.btn_reset_cam.clicked.connect(self.plotter.reset_camera)

        lbl_amp = QLabel("Amplituda:")
        self.slider_amp = QSlider(Qt.Horizontal)
        self.slider_amp.setRange(1, 25)
        self.slider_amp.setValue(int(self.amplitude * 10))
        self.slider_amp.valueChanged.connect(self._on_amp_changed)

        lbl_speed = QLabel("Tempo:")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(15, 80)
        self.slider_speed.setValue(35)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)

        bottom_bar.addWidget(self.btn_play)
        bottom_bar.addWidget(self.btn_reset_cam)
        bottom_bar.addWidget(lbl_amp)
        bottom_bar.addWidget(self.slider_amp)
        bottom_bar.addWidget(lbl_speed)
        bottom_bar.addWidget(self.slider_speed)
        layout.addLayout(bottom_bar)

        self._update_mode_label()

    def _setup_pyvista_scene(self):
        self.plotter.clear()
        self.anim_atom_actors = []

        for idx, (an, coord) in enumerate(zip(self.atomnos, self.coords_orig)):
            sym = ELEMENT_SYMBOLS.get(an, "C")
            color = CPK_COLORS.get(an, "#DD77DD")
            radius = COVALENT_RADII.get(an, 0.7) * 0.38
            sphere = pv.Sphere(radius=radius, center=(0, 0, 0), theta_resolution=16, phi_resolution=16)
            actor = self.plotter.add_mesh(
                sphere, color=color, smooth_shading=True,
                specular=0.5, specular_power=20, pickable=False, render=False
            )
            actor.position = coord
            self.anim_atom_actors.append(actor)

        self._render_bonds(self.coords_orig)
        self.plotter.reset_camera()
        self.plotter.render()

    def _render_bonds(self, current_coords: np.ndarray):
        if hasattr(self, 'plotter') and self.anim_bond_actor is not None:
            self.plotter.remove_actor(self.anim_bond_actor, render=False)
            self.anim_bond_actor = None

        tubes = []
        for a, b in self.bonds:
            tube = pv.Tube(pointa=current_coords[a], pointb=current_coords[b], radius=0.06, resolution=10)
            tubes.append(tube)

        if tubes and hasattr(self, 'plotter'):
            merged = pv.merge(tubes)
            self.anim_bond_actor = self.plotter.add_mesh(
                merged, color='#B0B0B0', opacity=0.85,
                smooth_shading=True, pickable=False, render=False
            )

    def _on_animation_tick(self):
        if not HAS_PYVISTA or not hasattr(self, 'plotter') or self.Lx.size == 0:
            return

        phase = np.sin(2.0 * np.pi * self.frame_step / self.n_frames)
        n_atoms = len(self.atomnos)

        if self.current_mode < self.Lx.shape[1]:
            disp = self.Lx[:, self.current_mode].reshape((n_atoms, 3))
            coords_curr = self.coords_orig + (disp * self.amplitude * phase)

            for idx, coord in enumerate(coords_curr):
                self.anim_atom_actors[idx].position = coord

            self._render_bonds(coords_curr)
            self.plotter.render()

        self.frame_step = (self.frame_step + 1) % self.n_frames

    def _update_mode_label(self):
        if 0 <= self.current_mode < len(self.freqs):
            f = self.freqs[self.current_mode]
            it = self.intens[self.current_mode] if self.current_mode < len(self.intens) else 0.0
            self.lbl_mode_info.setText(
                f"Mod #{self.current_mode + 1} / {len(self.freqs)}  |  ν = {f:.1f} cm⁻¹  |  I = {it:.1f} km/mol"
            )

    def set_mode(self, mode_idx: int):
        if 0 <= mode_idx < len(self.freqs):
            self.current_mode = mode_idx
            self.frame_step = 0
            self._update_mode_label()

    def _prev_mode(self):
        if len(self.freqs) > 0:
            self.set_mode((self.current_mode - 1) % len(self.freqs))

    def _next_mode(self):
        if len(self.freqs) > 0:
            self.set_mode((self.current_mode + 1) % len(self.freqs))

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ Pauza")
            self.timer.start(self.speed_ms)
        else:
            self.btn_play.setText("▶ Odtwarzaj")
            self.timer.stop()

    def _on_amp_changed(self, val: int):
        self.amplitude = val / 10.0

    def _on_speed_changed(self, val: int):
        self.speed_ms = val
        if self.is_playing:
            self.timer.start(self.speed_ms)

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'plotter'):
            self.plotter.close()
        super().closeEvent(event)