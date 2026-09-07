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
class SpectralSeries:
    """
    Reprezentuje serię ewolucyjną 2D (W x N) lub tensor 3D (W x N x M).
    W = liczba falowa, N = zmienna 1 (np. czas / rampa T), M = zmienna 2 (np. stężenie).
    """
    name: str
    wavenumbers: np.ndarray             # Oś X (W)
    data: np.ndarray                    # Macierz 2D (W x N) lub Tensor 3D (W x N x M)
    axis_n_name: str = "Czas"           # Etykieta wymiaru N
    axis_n_values: Optional[np.ndarray] = None  # Wartości fizyczne (np. sekundy, °C)
    axis_m_name: str = "Stężenie"       # Etykieta wymiaru M (dla 3D)
    axis_m_values: Optional[np.ndarray] = None  # Wartości fizyczne (np. mol/dm³)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_3d(self) -> bool:
        return self.data.ndim == 3

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def get_matrix_2d(self) -> np.ndarray:
        """Zwraca widma spłaszczone do macierzy 2D (W x Próbki) dla metod takich jak PCA/SVD."""
        if self.data.ndim == 2:
            return self.data
        w, n, m = self.data.shape
        return self.data.reshape(w, n * m)

@dataclass
class GaussianData(SpectrumData):
    """Rozszerzona struktura dla widm teoretycznych DFT z analizą PED i wizualizacją 3D."""
    stick_freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    stick_intens: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ped_results: List[Dict[str, Any]] = field(default_factory=list)
    hessian_source: str = "LOG"  # 'FCHK', 'LOG', 'ORCA_OUT'
    scale_factor: float = 1.0
    atomnos: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    Lx: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))