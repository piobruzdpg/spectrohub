"""
io_hub/readers/base_reader.py
Klasa abstrakcyjna i narzędzia pomocnicze dla parserów spektroskopowych.
"""

from abc import ABC, abstractmethod
import os
import numpy as np
from typing import List
from core.models import SpectrumData
import logging

logger = logging.getLogger("SpectroHub")


class BaseSpectrumReader(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich czytników widm."""

    @abstractmethod
    def read(self, file_path: str) -> List[SpectrumData]:
        """Odczytuje plik i zwraca listę obiektów SpectrumData."""
        pass

    @staticmethod
    def sanitize_and_sort(wavenumbers: np.ndarray, intensities: np.ndarray):
        """
        Usuwa punkty NaN/Inf oraz wymusza rosnący porządek liczb falowych (np. 400 -> 4000 cm^-1).
        """
        wn = np.asarray(wavenumbers, dtype=np.float64)
        intens = np.asarray(intensities, dtype=np.float64)

        # Usunięcie NaN i nieskończoności
        valid = ~(np.isnan(wn) | np.isnan(intens) | np.isinf(wn) | np.isinf(intens))
        wn = wn[valid]
        intens = intens[valid]

        if len(wn) == 0:
            raise ValueError("Widmo po odfiltrowaniu wartości NaN/Inf nie zawiera żadnych punktów.")

        # Wymuszenie monotoniczności rosnącej
        if wn[0] > wn[-1]:
            wn = wn[::-1]
            intens = intens[::-1]

        return wn, intens