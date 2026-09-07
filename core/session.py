"""
core/session.py
Centralny menedżer stanu danych (Single Source of Truth) dla SpectroHub.
"""

from typing import Dict, List, Optional
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack
import logging

from core.models import SpectrumData, SpectralSeries

logger = logging.getLogger("SpectroHub")


class ProjectSession(QObject):
    """Przechowuje wszystkie załadowane dane i emituje powiadomienia o zmianach."""

    # Bezpieczne sygnały w PySide6 - przekazywanie obiektów Pythona przez object
    spectrum_added = Signal(object)
    spectrum_removed = Signal(str)
    active_spectrum_changed = Signal(object)
    visibility_changed = Signal(str, bool)
    session_cleared = Signal()
    series_added = Signal(object)
    series_removed = Signal(str)

    def __init__(self):
        super().__init__()
        self._spectra: Dict[str, SpectrumData] = {}
        self._visibility: Dict[str, bool] = {}
        self._active_spectrum_name: Optional[str] = None
        self.undo_stack = QUndoStack(self)
        self._series: Dict[str, SpectralSeries] = {}

    @property
    def spectra(self) -> Dict[str, SpectrumData]:
        return self._spectra

    @property
    def active_spectrum(self) -> Optional[SpectrumData]:
        if self._active_spectrum_name and self._active_spectrum_name in self._spectra:
            return self._spectra[self._active_spectrum_name]
        return None

    def add_spectrum(self, spectrum: SpectrumData, make_active: bool = True):
        base_name = spectrum.name
        unique_name = base_name
        counter = 1
        while unique_name in self._spectra:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        spectrum.name = unique_name
        self._spectra[unique_name] = spectrum
        self._visibility[unique_name] = True
        logger.info(f"Dodano widmo do sesji: '{unique_name}' ({spectrum.file_type}, {spectrum.n_points} pkt.)")

        self.spectrum_added.emit(spectrum)

        if make_active or self._active_spectrum_name is None:
            self.set_active_spectrum(unique_name)

    def add_spectra_batch(self, spectra_list: List[SpectrumData]):
        for spec in spectra_list:
            self.add_spectrum(spec, make_active=False)
        if spectra_list and self._active_spectrum_name is None:
            self.set_active_spectrum(spectra_list[-1].name)

    def remove_spectrum(self, name: str):
        if name in self._spectra:
            del self._spectra[name]
            self._visibility.pop(name, None)
            logger.info(f"Usunięto widmo z sesji: '{name}'")
            self.spectrum_removed.emit(name)

            if self._active_spectrum_name == name:
                new_active = next(iter(self._spectra.keys()), None)
                self.set_active_spectrum(new_active)

    @property
    def series(self) -> Dict[str, SpectralSeries]:
        return self._series

    def add_series(self, series_obj: SpectralSeries):
        base_name = series_obj.name
        unique_name = base_name
        counter = 1
        while unique_name in self._series:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        series_obj.name = unique_name
        self._series[unique_name] = series_obj
        logger.info(f"Dodano serię/tensor do sesji: '{unique_name}' (Wymiary: {series_obj.shape})")
        self.series_added.emit(series_obj)

    def remove_series(self, name: str):
        if name in self._series:
            del self._series[name]
            logger.info(f"Usunięto serię z sesji: '{name}'")
            self.series_removed.emit(name)

    def set_active_spectrum(self, name: Optional[str]):
        if name != self._active_spectrum_name:
            if name is not None and name not in self._spectra:
                return
            self._active_spectrum_name = name
            logger.debug(f"Aktywne widmo zmienione na: '{name}'")
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_visibility(self, name: str, is_visible: bool):
        if name in self._spectra and self._visibility.get(name) != is_visible:
            self._visibility[name] = is_visible
            self.visibility_changed.emit(name, is_visible)

    def is_visible(self, name: str) -> bool:
        return self._visibility.get(name, True)

    def get_visible_spectra(self) -> List[SpectrumData]:
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    def clear_all(self):
        self._spectra.clear()
        self._visibility.clear()
        self._active_spectrum_name = None
        self.undo_stack.clear()
        logger.info("Wyczyszczono całą sesję projektu.")
        self.session_cleared.emit()
        self._series.clear()


session = ProjectSession()