"""
core/models.py
Kluczowe struktury danych spektroskopowych dla SpectroHub.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any, List


@dataclass
class SpectrumData:
    """Ustandaryzowany nośnik danych pojedynczego widma."""
    wavenumbers: np.ndarray      # Oś X (float64, uporządkowana monotonicznie)
    intensities: np.ndarray      # Oś Y (float64, absorbancja/transmisja)
    name: str                    # Nazwa widma / identyfikator
    file_path: str               # Oryginalna ścieżka do pliku
    file_type: str               # 'opus', 'spa', 'spg', 'spc', 'csv', 'jcamp', 'gaussian'
    time_str: str = "00:00:00"   # Stempel czasu (HH:MM:SS)
    units_x: str = "cm^-1"
    units_y: str = "absorbance"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Wymuszenie float64 i spójności tablic
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=np.float64)
        self.intensities = np.asarray(self.intensities, dtype=np.float64)

        if len(self.wavenumbers) != len(self.intensities):
            raise ValueError(
                f"Niezgodność wymiarów: X={len(self.wavenumbers)}, Y={len(self.intensities)} w widmie '{self.name}'"
            )

    @property
    def n_points(self) -> int:
        return len(self.wavenumbers)


@dataclass
class GaussianData(SpectrumData):
    """Rozszerzona struktura dla widm teoretycznych DFT z analizą PED."""
    stick_freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    stick_intens: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ped_results: List[Dict[str, Any]] = field(default_factory=list)
    hessian_source: str = "LOG"  # 'FCHK' lub 'LOG'
    scale_factor: float = 1.0