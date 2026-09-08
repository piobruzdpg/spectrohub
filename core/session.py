"""
core/session.py
Centralny menedżer stanu danych (Single Source of Truth) dla SpectroHub.
Zarządza Magazynem Danych (Pomiary Eksperymentalne, Wzorce Atmosfery, Serie 2D, DFT)
oraz Niezależnymi Oknami Roboczymi (Workspaces).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack
import logging

from core.models import SpectrumData, SpectralSeries

logger = logging.getLogger("SpectroHub")


def is_atmospheric_name(name: str) -> bool:
    """Rozpoznaje wzorce atmosfery wyłącznie na podstawie fraz atm oraz co2."""
    nl = name.lower()
    return "atm" in nl or "co2" in nl


@dataclass
class WorkspaceWindow:
    """Reprezentuje niezależne okno wykresu/robocze w projekcie."""
    window_id: str
    name: str
    spectra: Dict[str, SpectrumData] = field(default_factory=dict)
    visibility: Dict[str, bool] = field(default_factory=dict)
    active_spectrum_name: Optional[str] = None


class ProjectSession(QObject):
    # Sygnały Magazynu - Pomiary
    spectrum_added = Signal(object)
    spectra_batch_added = Signal(list)
    spectrum_removed = Signal(str)
    active_spectrum_changed = Signal(object)
    visibility_changed = Signal(str, bool)
    session_cleared = Signal()

    # Sygnały Magazynu - Wzorce Atmosfery
    atm_spectrum_added = Signal(object)
    atm_spectra_batch_added = Signal(list)
    atm_spectrum_removed = Signal(str)
    atm_visibility_changed = Signal(str, bool)

    # Sygnały Mutacji Danych
    spectrum_data_changed = Signal(object)
    batch_spectra_changed = Signal(list)

    # Sygnały Serii 2D / 3D
    series_added = Signal(object)
    series_removed = Signal(str)
    series_visibility_changed = Signal(str, bool)
    active_series_changed = Signal(object)
    series_data_changed = Signal(object)

    # Sygnały Okien Roboczych
    window_added = Signal(str, str)
    window_removed = Signal(str)
    active_window_changed = Signal(str)
    window_spectrum_added = Signal(str, object)
    window_spectrum_removed = Signal(str, str)
    window_visibility_changed = Signal(str, str, bool)
    context_changed = Signal(str)  # 'storage' | 'window'

    def __init__(self):
        super().__init__()
        # Magazyn danych (Pomiary próbek)
        self._spectra: Dict[str, SpectrumData] = {}
        self._visibility: Dict[str, bool] = {}

        # Dedykowany magazyn wzorców atmosfery (H2O vapor / CO2)
        self._atm_spectra: Dict[str, SpectrumData] = {}
        self._atm_visibility: Dict[str, bool] = {}

        self._active_spectrum_name: Optional[str] = None
        self.undo_stack = QUndoStack(self)

        self._series: Dict[str, SpectralSeries] = {}
        self._series_visibility: Dict[str, bool] = {}
        self._active_series_name: Optional[str] = None

        # Okna robocze
        self._windows: Dict[str, WorkspaceWindow] = {}
        self._active_window_id: Optional[str] = None
        self._window_counter = 0

        self._active_context: str = "storage"
        self.add_window("Okno 1 (Główne)")

    # --- ZARZĄDZANIE OKNAMI ROBOCZYMI ---
    @property
    def windows(self) -> Dict[str, WorkspaceWindow]:
        return self._windows

    @property
    def active_window(self) -> Optional[WorkspaceWindow]:
        if self._active_window_id and self._active_window_id in self._windows:
            return self._windows[self._active_window_id]
        return None

    @property
    def active_window_id(self) -> Optional[str]:
        return self._active_window_id

    def add_window(self, name: Optional[str] = None) -> str:
        self._window_counter += 1
        window_id = f"win_{self._window_counter}"
        window_name = name or f"Okno {self._window_counter}"

        win = WorkspaceWindow(window_id=window_id, name=window_name)
        self._windows[window_id] = win
        logger.info(f"Utworzono nowe okno robocze: '{window_name}' (ID: {window_id})")
        self.window_added.emit(window_id, window_name)

        if self._active_window_id is None:
            self.set_active_window(window_id)
        return window_id

    def remove_window(self, window_id: str):
        if window_id in self._windows:
            del self._windows[window_id]
            logger.info(f"Usunięto okno robocze: ID {window_id}")
            self.window_removed.emit(window_id)

            if self._active_window_id == window_id:
                new_active = next(iter(self._windows.keys()), None)
                self.set_active_window(new_active)

    def get_display_spectra(self, include_atm: bool = True) -> List[SpectrumData]:
        """Zwraca widma widoczne na wykresie."""
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            return [s for name, s in win.spectra.items() if win.visibility.get(name, True)]

        display = [s for name, s in self._spectra.items() if self._visibility.get(name, True)]
        if include_atm:
            display.extend([s for name, s in self._atm_spectra.items() if self._atm_visibility.get(name, True)])
        return display

    def set_active_window(self, window_id: Optional[str]):
        if window_id is not None and window_id not in self._windows:
            return

        self._active_window_id = window_id
        self._active_context = "window"

        if window_id and window_id in self._windows:
            win = self._windows[window_id]
            if (not win.active_spectrum_name or win.active_spectrum_name not in win.spectra) and win.spectra:
                win.active_spectrum_name = next(iter(win.spectra.keys()))

        logger.debug(f"Aktywne okno robocze: {window_id}")
        self.active_window_changed.emit(window_id)
        self.active_spectrum_changed.emit(self.active_spectrum)

    def set_window_active_spectrum(self, window_id: str, spec_name: str):
        if window_id in self._windows and spec_name in self._windows[window_id].spectra:
            self._active_window_id = window_id
            self._active_context = "window"
            self._windows[window_id].active_spectrum_name = spec_name
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_window_spectrum_visibility(self, window_id: str, spec_name: str, is_visible: bool):
        if window_id in self._windows and spec_name in self._windows[window_id].spectra:
            self._windows[window_id].visibility[spec_name] = is_visible
            self.window_visibility_changed.emit(window_id, spec_name, is_visible)

    def add_spectrum_to_window(self, window_id: str, spectrum: SpectrumData) -> SpectrumData:
        if window_id not in self._windows:
            raise KeyError(f"Nie odnaleziono okna: {window_id}")

        win = self._windows[window_id]
        cloned_spec = spectrum.clone()

        base_name = cloned_spec.name
        unique_name = base_name
        counter = 1
        while unique_name in win.spectra:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        cloned_spec.name = unique_name
        win.spectra[unique_name] = cloned_spec
        win.visibility[unique_name] = True
        win.active_spectrum_name = unique_name

        logger.info(f"Dodano sklonowane widmo '{unique_name}' do okna '{win.name}'")
        self.window_spectrum_added.emit(window_id, cloned_spec)
        return cloned_spec

    def remove_spectrum_from_window(self, window_id: str, name: str):
        if window_id in self._windows:
            win = self._windows[window_id]
            if name in win.spectra:
                del win.spectra[name]
                win.visibility.pop(name, None)
                if win.active_spectrum_name == name:
                    win.active_spectrum_name = next(iter(win.spectra.keys()), None)
                self.window_spectrum_removed.emit(window_id, name)

    # --- KONTEKST ---
    @property
    def active_context(self) -> str:
        return self._active_context

    def set_context(self, context: str):
        if context in ("storage", "window"):
            self._active_context = context
            logger.debug(f"Zmiana kontekstu sesji na: {context}")
            self.context_changed.emit(context)
            self.active_spectrum_changed.emit(self.active_spectrum)

    # --- ZARZĄDZANIE WIDMAMI MAGAZYNU I WZORCAMI ATMOSFERY ---
    @property
    def spectra(self) -> Dict[str, SpectrumData]:
        return self._spectra

    @property
    def atm_spectra(self) -> Dict[str, SpectrumData]:
        return self._atm_spectra

    def get_spectrum(self, name: str) -> Optional[SpectrumData]:
        if name in self._spectra:
            return self._spectra[name]
        if name in self._atm_spectra:
            return self._atm_spectra[name]
        if self.active_window and name in self.active_window.spectra:
            return self.active_window.spectra[name]
        return None

    @property
    def active_spectrum(self) -> Optional[SpectrumData]:
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            if win.active_spectrum_name and win.active_spectrum_name in win.spectra:
                return win.spectra[win.active_spectrum_name]
            if win.spectra:
                first_name = next(iter(win.spectra.keys()))
                win.active_spectrum_name = first_name
                return win.spectra[first_name]
            return None

        if self._active_spectrum_name:
            if self._active_spectrum_name in self._spectra:
                return self._spectra[self._active_spectrum_name]
            if self._active_spectrum_name in self._atm_spectra:
                return self._atm_spectra[self._active_spectrum_name]
        return None

    def add_spectrum(self, spectrum: SpectrumData, is_atm: Optional[bool] = None,
                     make_active: bool = True, emit_signal: bool = True):
        if is_atm is None:
            is_atm = is_atmospheric_name(spectrum.name)

        target_dict = self._atm_spectra if is_atm else self._spectra
        target_vis = self._atm_visibility if is_atm else self._visibility

        base_name = spectrum.name
        unique_name = base_name
        counter = 1
        all_existing = set(self._spectra.keys()) | set(self._atm_spectra.keys())
        while unique_name in all_existing:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        spectrum.name = unique_name
        target_dict[unique_name] = spectrum
        target_vis[unique_name] = True

        if emit_signal:
            if is_atm:
                logger.info(f"Dodano wzorzec atmosfery: '{unique_name}'")
                self.atm_spectrum_added.emit(spectrum)
            else:
                logger.info(f"Dodano widmo pomiarowe: '{unique_name}'")
                self.spectrum_added.emit(spectrum)

            if make_active or self._active_spectrum_name is None:
                self.set_active_spectrum(unique_name)

    def add_spectra_batch(self, spectra_list: List[SpectrumData]):
        if not spectra_list:
            return

        added_exp = []
        added_atm = []

        for spec in spectra_list:
            if is_atmospheric_name(spec.name):
                self.add_spectrum(spec, is_atm=True, make_active=False, emit_signal=False)
                added_atm.append(spec)
            else:
                self.add_spectrum(spec, is_atm=False, make_active=False, emit_signal=False)
                added_exp.append(spec)

        self._active_context = "storage"
        if added_exp:
            logger.info(f"Zarejestrowano wsadowo {len(added_exp)} widm eksperymentalnych.")
            self.spectra_batch_added.emit(added_exp)
        if added_atm:
            logger.info(f"Zarejestrowano wsadowo {len(added_atm)} wzorców atmosfery.")
            self.atm_spectra_batch_added.emit(added_atm)

        all_added = added_exp or added_atm
        if self._active_spectrum_name is None and all_added:
            self.set_active_spectrum(all_added[-1].name)
        else:
            self.context_changed.emit("storage")

    def move_to_atm(self, name: str):
        """Przenosi widmo z pomiarów eksperymentalnych do wzorców atmosfery."""
        if name in self._spectra:
            spec = self._spectra.pop(name)
            vis = self._visibility.pop(name, True)
            self._atm_spectra[name] = spec
            self._atm_visibility[name] = vis
            self.spectrum_removed.emit(name)
            self.atm_spectrum_added.emit(spec)
            logger.info(f"Przeniesiono '{name}' do Wzorców Atmosfery.")

    def move_to_experimental(self, name: str):
        """Przenosi widmo ze wzorców atmosfery do pomiarów eksperymentalnych."""
        if name in self._atm_spectra:
            spec = self._atm_spectra.pop(name)
            vis = self._atm_visibility.pop(name, True)
            self._spectra[name] = spec
            self._visibility[name] = vis
            self.atm_spectrum_removed.emit(name)
            self.spectrum_added.emit(spec)
            logger.info(f"Przeniesiono '{name}' do Pomiarów Eksperymentalnych.")

    def remove_spectrum(self, name: str):
        if name in self._spectra:
            del self._spectra[name]
            self._visibility.pop(name, None)
            logger.info(f"Usunięto widmo z magazynu: '{name}'")
            self.spectrum_removed.emit(name)
            if self._active_spectrum_name == name:
                new_active = next(iter(self._spectra.keys()), next(iter(self._atm_spectra.keys()), None))
                self.set_active_spectrum(new_active)
        elif name in self._atm_spectra:
            del self._atm_spectra[name]
            self._atm_visibility.pop(name, None)
            logger.info(f"Usunięto wzorzec atmosfery: '{name}'")
            self.atm_spectrum_removed.emit(name)
            if self._active_spectrum_name == name:
                new_active = next(iter(self._atm_spectra.keys()), next(iter(self._spectra.keys()), None))
                self.set_active_spectrum(new_active)

    def set_active_spectrum(self, name: Optional[str]):
        if name != self._active_spectrum_name:
            if name is not None and name not in self._spectra and name not in self._atm_spectra:
                return
            self._active_spectrum_name = name
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_visibility(self, name: str, is_visible: bool):
        if name in self._spectra and self._visibility.get(name) != is_visible:
            self._visibility[name] = is_visible
            self.visibility_changed.emit(name, is_visible)
        elif name in self._atm_spectra and self._atm_visibility.get(name) != is_visible:
            self._atm_visibility[name] = is_visible
            self.atm_visibility_changed.emit(name, is_visible)

    def is_visible(self, name: str) -> bool:
        if name in self._spectra:
            return self._visibility.get(name, True)
        if name in self._atm_spectra:
            return self._atm_visibility.get(name, True)
        return True

    def get_visible_spectra(self) -> List[SpectrumData]:
        """Zwraca widoczne widma pomiarowe z magazynu (kompatybilność wsteczna)."""
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    def get_visible_atm_spectra(self) -> List[SpectrumData]:
        """Zwraca widoczne wzorce atmosfery z magazynu."""
        return [s for name, s in self._atm_spectra.items() if self._atm_visibility.get(name, True)]

    def get_atm_matrix(self, target_wavenumbers: np.ndarray, only_visible: bool = True) -> Optional[np.ndarray]:
        """Zwraca macierz (N_pkt x K_widm) wzorców atmosfery interpolowanych do podanej osi X."""
        source_dict = self._atm_spectra
        if only_visible:
            specs = [s for name, s in source_dict.items() if self._atm_visibility.get(name, True)]
        else:
            specs = list(source_dict.values())

        if not specs:
            return None

        cols = []
        for s in specs:
            xw, yw = s.wavenumbers, s.intensities
            if xw[0] > xw[-1]:
                xw_s, yw_s = xw[::-1], yw[::-1]
            else:
                xw_s, yw_s = xw, yw
            cols.append(np.interp(target_wavenumbers, xw_s, yw_s))

        return np.column_stack(cols)

    # --- SERIE I TENSORY ---
    @property
    def series(self) -> Dict[str, SpectralSeries]:
        return self._series

    @property
    def active_series(self) -> Optional[SpectralSeries]:
        if self._active_series_name and self._active_series_name in self._series:
            return self._series[self._active_series_name]
        return None

    def get_series(self, name: str) -> Optional[SpectralSeries]:
        return self._series.get(name)

    def add_series(self, series_obj: SpectralSeries, make_active: bool = True):
        base_name = series_obj.name
        unique_name = base_name
        counter = 1
        while unique_name in self._series:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        series_obj.name = unique_name
        self._series[unique_name] = series_obj
        self._series_visibility[unique_name] = True
        logger.info(f"Dodano serię do magazynu: '{unique_name}'")
        self.series_added.emit(series_obj)
        if make_active or self._active_series_name is None:
            self.set_active_series(unique_name)

    def remove_series(self, name: str):
        if name in self._series:
            del self._series[name]
            self._series_visibility.pop(name, None)
            self.series_removed.emit(name)
            if self._active_series_name == name:
                new_active = next(iter(self._series.keys()), None)
                self.set_active_series(new_active)

    def set_active_series(self, name: Optional[str]):
        if name != self._active_series_name:
            if name is not None and name not in self._series:
                return
            self._active_series_name = name
            self.active_series_changed.emit(self.active_series)

    def set_series_visibility(self, name: str, is_visible: bool):
        if name in self._series and self._series_visibility.get(name) != is_visible:
            self._series_visibility[name] = is_visible
            self.series_visibility_changed.emit(name, is_visible)

    def get_visible_series(self) -> List[SpectralSeries]:
        return [s for name, s in self._series.items() if self._series_visibility.get(name, True)]

    # --- CZYSZCZENIE SESJI ---
    def clear_all(self):
        self._spectra.clear()
        self._visibility.clear()
        self._atm_spectra.clear()
        self._atm_visibility.clear()
        self._active_spectrum_name = None

        self._series.clear()
        self._series_visibility.clear()
        self._active_series_name = None

        self._windows.clear()
        self._active_window_id = None
        self._window_counter = 0

        self.undo_stack.clear()
        logger.info("Wyczyszczono sesję projektu.")
        self.session_cleared.emit()
        self.add_window("Okno 1 (Główne)")


session = ProjectSession()