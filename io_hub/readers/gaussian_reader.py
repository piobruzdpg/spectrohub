"""
io_hub/readers/gaussian_reader.py
Czytnik plików wyjściowych Gaussian (.log, .fchk) i ORCA (.out) z generowaniem syntetycznego widma.
"""

import os
import numpy as np
from typing import List

from core.models import GaussianData
from io_hub.readers.base_reader import BaseSpectrumReader
from algorithms.quantum.ped_engine import PEDAnalyzer


class GaussianReader(BaseSpectrumReader):
    """Parsuje wyniki Gaussiana/ORCA i generuje profil widma z pałeczek drgań."""

    def read(self, file_path: str, fwhm: float = 14.0, grid_step: float = 1.0) -> List[GaussianData]:
        atomnos, coords, freqs, intens, Lx, ped_results, hessian_src = PEDAnalyzer.calculate_ped(file_path)

        if not freqs:
            raise ValueError(f"Plik '{os.path.basename(file_path)}' nie zawiera harmonicznych częstości IR.")

        stick_freqs = np.array(freqs, dtype=np.float64)
        stick_intens = np.array(intens, dtype=np.float64)

        # Dopasowanie siatki liczb falowych z marginesem
        min_wn = max(200.0, np.min(stick_freqs) - 150.0)
        max_wn = max(4000.0, np.max(stick_freqs) + 150.0)
        wn_grid = np.arange(min_wn, max_wn + grid_step, grid_step)

        # Generowanie profilu pasm (splot Gaussa o szerokości FWHM)
        c = 4.0 * np.log(2.0)
        spectrum_y = np.zeros_like(wn_grid)
        for f_val, i_val in zip(stick_freqs, stick_intens):
            spectrum_y += i_val * np.exp(-c * ((wn_grid - f_val) / fwhm) ** 2)

        wn_clean, y_clean = self.sanitize_and_sort(wn_grid, spectrum_y)
        filename = os.path.splitext(os.path.basename(file_path))[0]

        return [GaussianData(
            wavenumbers=wn_clean,
            intensities=y_clean,
            name=f"{filename}",
            file_path=file_path,
            file_type="gaussian",
            stick_freqs=stick_freqs,
            stick_intens=stick_intens,
            ped_results=ped_results,
            hessian_source=hessian_src,
            atomnos=atomnos,
            coords=coords,
            Lx=Lx
        )]