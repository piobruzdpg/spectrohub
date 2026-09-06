"""
core/session.py
Centralny menedżer stanu danych (Single Source of Truth) dla SpectroHub.
"""

from typing import Dict, List, Optional
import numpy as np
from PySide6.QtCore import QObject, Signal
import logging

from core.models import SpectrumData
from core.logger import log_algo_exception

logger = logging.getLogger("SpectroHub")


class ProjectSession(QObject):
    """Przechowuje wszystkie załadowane dane i emituje powiadomienia o zmianach."""

    # Sygnały zmian stanu sesji
    spectrum_added = Signal(SpectrumData)
    spectrum_removed = Signal(str)  # id / nazwa
    active_spectrum_changed = Signal(Optional[SpectrumData])
    visibility_changed = Signal(str, bool)  # name, is_visible
    session_cleared = Signal()

    def __init__(self):
        super().__init__()
        self._spectra: Dict[str, SpectrumData] = {}
        self._visibility: Dict[str, bool] = {}
        self._active_spectrum_name: Optional[str] = None

    @property
    def spectra(self) -> Dict[str, SpectrumData]:
        return self._spectra

    @property
    def active_spectrum(self) -> Optional[SpectrumData]:
        if self._active_spectrum_name and self._active_spectrum_name in self._spectra:
            return self._spectra[self._active_spectrum_name]
        return None

    def add_spectrum(self, spectrum: SpectrumData, make_active: bool = True):
        """Dodaje widmo do sesji. Jeśli nazwa istnieje, nadaje unikalny sufiks."""
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
        """Hurtowe dodanie wielu widm (np. po wczytaniu grupy OMNIC .spg)."""
        for spec in spectra_list:
            self.add_spectrum(spec, make_active=False)
        if spectra_list and self._active_spectrum_name is None:
            self.set_active_spectrum(spectra_list[-1].name)

    def remove_spectrum(self, name: str):
        """Usuwa wskazane widmo z pamięci."""
        if name in self._spectra:
            del self._spectra[name]
            self._visibility.pop(name, None)
            logger.info(f"Usunięto widmo z sesji: '{name}'")
            self.spectrum_removed.emit(name)

            if self._active_spectrum_name == name:
                new_active = next(iter(self._spectra.keys()), None)
                self.set_active_spectrum(new_active)

    def set_active_spectrum(self, name: Optional[str]):
        """Zmienia aktywne widmo robocze."""
        if name != self._active_spectrum_name:
            if name is not None and name not in self._spectra:
                return
            self._active_spectrum_name = name
            logger.debug(f"Aktywne widmo zmienione na: '{name}'")
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_visibility(self, name: str, is_visible: bool):
        """Steruje widocznością widma na wykresie."""
        if name in self._spectra and self._visibility.get(name) != is_visible:
            self._visibility[name] = is_visible
            self.visibility_changed.emit(name, is_visible)

    def is_visible(self, name: str) -> bool:
        return self._visibility.get(name, True)

    def get_visible_spectra(self) -> List[SpectrumData]:
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    def clear_all(self):
        """Czyści całą sesję roboczą."""
        self._spectra.clear()
        self._visibility.clear()
        self._active_spectrum_name = None
        logger.info("Wyczyszczono całą sesję projektu.")
        self.session_cleared.emit()


# Globalna instancja sesji
session = ProjectSession()