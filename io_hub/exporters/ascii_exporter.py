"""
io_hub/exporters/ascii_exporter.py
Natywny eksporter widm do czystych formatów numerycznych (CSV, DPT, TXT).
Zgodny z programami spektroskopowymi (OMNIC, OPUS, GRAMS, Origin).
"""

import numpy as np
from typing import List
import logging

from core.models import SpectrumData

logger = logging.getLogger("SpectroHub")


class ASCIIExporter:
    """Zapisuje dane widmowe do plików tekstowych CSV / DPT / TXT."""

    @staticmethod
    def export_spectrum(file_path: str,
                        wavenumbers: np.ndarray,
                        intensities: np.ndarray,
                        style: str = "CSV_EU",
                        include_header: bool = False) -> None:
        """
        Zapisuje czyste kolumny numeryczne X i Y.
        style:
          - 'CSV_EU' / 'EU': kropka dziesiętna, separator: ','
          - 'CSV_PL' / 'PL': przecinek dziesiętny, separator: ';'
          - 'DPT_EU' / 'DPT': kropka dziesiętna, separator: '\t' (standard Bruker OPUS DPT)
          - 'DPT_PL': przecinek dziesiętny, separator: '\t'
        """
        if len(wavenumbers) != len(intensities):
            raise ValueError("Niezgodność wymiarów wektorów X i Y.")

        ext = file_path.lower()
        style_u = style.upper()

        if style_u == "DPT_PL":
            sep = "\t"
            dec = ","
        elif style_u in ["DPT_EU", "DPT"]:
            sep = "\t"
            dec = "."
        elif style_u in ["CSV_PL", "PL"]:
            sep = ";"
            dec = ","
        elif style_u in ["CSV_EU", "EU"]:
            sep = ","
            dec = "."
        elif ext.endswith('.dpt'):
            sep = "\t"
            dec = "," if "PL" in style_u else "."
        elif "PL" in style_u:
            sep = ";"
            dec = ","
        else:
            sep = ","
            dec = "."

        with open(file_path, 'w', encoding='utf-8') as f:
            if include_header:
                f.write(f"Wavenumber{sep}Intensity\n")

            for x, y in zip(wavenumbers, intensities):
                if dec == ",":
                    x_str = f"{x:.4f}".replace('.', ',')
                    y_str = f"{y:.6e}".replace('.', ',')
                else:
                    x_str = f"{x:.4f}"
                    y_str = f"{y:.6e}"
                f.write(f"{x_str}{sep}{y_str}\n")

        logger.info(f"Zapisano czyste dane numeryczne ({style}): '{file_path}' ({len(wavenumbers)} pkt.)")

    @classmethod
    def export_spectrum_data(cls, spectrum: SpectrumData, file_path: str, style: str = "CSV_EU", include_header: bool = False) -> None:
        cls.export_spectrum(file_path, spectrum.wavenumbers, spectrum.intensities, style=style, include_header=include_header)

    @staticmethod
    def export_matrix_csv(file_path: str, spectra: List[SpectrumData], style: str = "CSV_EU", include_header: bool = True) -> None:
        """Eksportuje serię widm do macierzy (opcjonalny nagłówek z nazwami widm)."""
        if not spectra:
            return

        ref_wn = spectra[0].wavenumbers
        is_pl = "PL" in style.upper()
        sep = ";" if is_pl else ","
        dec = "," if is_pl else "."

        with open(file_path, 'w', encoding='utf-8') as f:
            if include_header:
                header = ["Wavenumber"] + [s.name for s in spectra]
                f.write(sep.join(header) + "\n")

            for i, wn in enumerate(ref_wn):
                row_vals = [f"{wn:.4f}"]
                for s in spectra:
                    if len(s.wavenumbers) == len(ref_wn) and np.allclose(s.wavenumbers[0], ref_wn[0]):
                        y_val = s.intensities[i]
                    else:
                        y_val = float(np.interp(wn, s.wavenumbers, s.intensities))
                    row_vals.append(f"{y_val:.6e}")

                if dec == ",":
                    row_vals = [v.replace('.', ',') for v in row_vals]

                f.write(sep.join(row_vals) + "\n")

        logger.info(f"Zapisano macierz CSV ({style}): '{file_path}'")