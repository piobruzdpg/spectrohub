"""
core/session.py
Centralny menedżer stanu danych (Single Source of Truth) dla SpectroHub.
Zarządza Magazynem Danych oraz Niezależnymi Oknami Roboczymi (Workspaces).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack
import logging

from core.models import SpectrumData, SpectralSeries

logger = logging.getLogger("SpectroHub")


@dataclass
class WorkspaceWindow:
    """Reprezentuje niezależne okno wykresu/robocze w projekcie."""
    window_id: str
    name: str
    spectra: Dict[str, SpectrumData] = field(default_factory=dict)
    visibility: Dict[str, bool] = field(default_factory=dict)
    active_spectrum_name: Optional[str] = None


class ProjectSession(QObject):
    # Sygnały Magazynu
    spectrum_added = Signal(object)
    spectra_batch_added = Signal(list)
    spectrum_removed = Signal(str)
    active_spectrum_changed = Signal(object)
    visibility_changed = Signal(str, bool)
    session_cleared = Signal()

    spectrum_data_changed = Signal(object)
    batch_spectra_changed = Signal(list)

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
        # Magazyn danych (surowe / referencyjne)
        self._spectra: Dict[str, SpectrumData] = {}
        self._visibility: Dict[str, bool] = {}
        self._active_spectrum_name: Optional[str] = None
        self.undo_stack = QUndoStack(self)

        self._series: Dict[str, SpectralSeries] = {}
        self._series_visibility: Dict[str, bool] = {}
        self._active_series_name: Optional[str] = None

        # Okna robocze (Przestrzenie robocze)
        self._windows: Dict[str, WorkspaceWindow] = {}
        self._active_window_id: Optional[str] = None
        self._window_counter = 0

        # DOMYŚLNY KONTEKST: Magazyn Danych (od razu widzimy wgrane widma)
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

    def get_display_spectra(self) -> List[SpectrumData]:
        """Zwraca widma widoczne na wykresie w zależności od wybranego kontekstu."""
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            return [s for name, s in win.spectra.items() if win.visibility.get(name, True)]
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

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
        """Klonuje widmo i dodaje je jako niezależną instancję do wskazanego okna."""
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

    # --- KONTEKST I AKTYWNE ŹRÓDŁO WIDOKU ---
    @property
    def active_context(self) -> str:
        return self._active_context

    def set_context(self, context: str):
        """Przełącza widok między Magazynem Danych a Oknem Roboczym."""
        if context in ("storage", "window"):
            self._active_context = context
            logger.debug(f"Zmiana kontekstu sesji na: {context}")
            self.context_changed.emit(context)
            self.active_spectrum_changed.emit(self.active_spectrum)

    # --- ZARZĄDZANIE WIDMAMI MAGAZYNU ---
    @property
    def spectra(self) -> Dict[str, SpectrumData]:
        return self._spectra

    @property
    def active_spectrum(self) -> Optional[SpectrumData]:
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            if win.active_spectrum_name and win.active_spectrum_name in win.spectra:
                return win.spectra[win.active_spectrum_name]
            if win.spectra:
                # Automatycznie wybierz pierwsze widmo z brzegu jako aktywne
                first_name = next(iter(win.spectra.keys()))
                win.active_spectrum_name = first_name
                return win.spectra[first_name]
            return None

        if self._active_spectrum_name and self._active_spectrum_name in self._spectra:
            return self._spectra[self._active_spectrum_name]
        return None

    def add_spectrum(self, spectrum: SpectrumData, make_active: bool = True, emit_signal: bool = True):
        base_name = spectrum.name
        unique_name = base_name
        counter = 1
        while unique_name in self._spectra:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        spectrum.name = unique_name
        self._spectra[unique_name] = spectrum
        self._visibility[unique_name] = True

        if emit_signal:
            logger.info(f"Dodano widmo do magazynu: '{unique_name}' ({spectrum.file_type})")
            self.spectrum_added.emit(spectrum)
            if make_active or self._active_spectrum_name is None:
                self.set_active_spectrum(unique_name)

    def add_spectra_batch(self, spectra_list: List[SpectrumData]):
        if not spectra_list:
            return
        added = []
        for spec in spectra_list:
            self.add_spectrum(spec, make_active=False, emit_signal=False)
            added.append(spec)

        logger.info(f"Zarejestrowano wsadowo w magazynie {len(added)} widm.")
        # Po wgraniu plików zawsze upewniamy się, że patrzymy na magazyn
        self._active_context = "storage"
        self.spectra_batch_added.emit(added)

        if self._active_spectrum_name is None and added:
            self.set_active_spectrum(added[-1].name)
        else:
            self.context_changed.emit("storage")

    def remove_spectrum(self, name: str):
        if name in self._spectra:
            del self._spectra[name]
            self._visibility.pop(name, None)
            logger.info(f"Usunięto widmo z magazynu: '{name}'")
            self.spectrum_removed.emit(name)
            if self._active_spectrum_name == name:
                new_active = next(iter(self._spectra.keys()), None)
                self.set_active_spectrum(new_active)

    def set_active_spectrum(self, name: Optional[str]):
        if name != self._active_spectrum_name:
            if name is not None and name not in self._spectra:
                return
            self._active_spectrum_name = name
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_visibility(self, name: str, is_visible: bool):
        if name in self._spectra and self._visibility.get(name) != is_visible:
            self._visibility[name] = is_visible
            self.visibility_changed.emit(name, is_visible)

    def is_visible(self, name: str) -> bool:
        return self._visibility.get(name, True)

    def get_visible_spectra(self) -> List[SpectrumData]:
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

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