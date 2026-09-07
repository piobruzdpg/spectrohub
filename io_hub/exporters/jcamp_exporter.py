"""
io_hub/exporters/jcamp_exporter.py
Zapis widm FTIR do uniwersalnego formatu IUPAC JCAMP-DX 4.24 (.dx, .jdx).
"""

import numpy as np
import logging

from core.models import SpectrumData

logger = logging.getLogger("SpectroHub")


class JCAMPExporter:
    """Natywny eksporter standardu IUPAC JCAMP-DX."""

    @staticmethod
    def export(file_path: str,
               wavenumbers: np.ndarray,
               intensities: np.ndarray,
               title: str = "FTIR Spectrum",
               yunits: str = "ABSORBANCE") -> None:
        if len(wavenumbers) == 0 or len(intensities) == 0:
            raise ValueError("Brak danych widmowych do zapisu JCAMP-DX.")

        first_x = float(wavenumbers[0])
        last_x = float(wavenumbers[-1])
        n_points = len(wavenumbers)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"##TITLE= {title}\n")
            f.write("##JCAMP-DX= 4.24\n")
            f.write("##DATA TYPE= INFRARED SPECTRUM\n")
            f.write("##DATA CLASS= XYDATA\n")
            f.write("##ORIGIN= SpectroHub Integrated Platform\n")
            f.write("##OWNER= Public Domain\n")
            f.write("##XUNITS= 1/CM\n")
            f.write(f"##YUNITS= {yunits.upper()}\n")
            f.write(f"##FIRSTX= {first_x:.4f}\n")
            f.write(f"##LASTX= {last_x:.4f}\n")
            f.write(f"##NPOINTS= {n_points}\n")
            f.write("##XYDATA= (X Y)\n")

            for x_val, y_val in zip(wavenumbers, intensities):
                f.write(f"{x_val:.4f} {y_val:.6e}\n")

            f.write("##END=\n")

        logger.info(f"Pomyślnie wyeksportowano JCAMP-DX: '{file_path}'")

    @classmethod
    def export_spectrum(cls, spectrum: SpectrumData, file_path: str) -> None:
        cls.export(
            file_path=file_path,
            wavenumbers=spectrum.wavenumbers,
            intensities=spectrum.intensities,
            title=spectrum.name,
            yunits=spectrum.units_y
        )