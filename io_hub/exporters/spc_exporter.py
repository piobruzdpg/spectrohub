"""
io_hub/exporters/spc_exporter.py
Natywny generator binarnych plików Thermo Grams SPC (Galactic New SPC 0x4B, float32).
"""

import struct
import numpy as np
from typing import Union
import logging

from core.models import SpectrumData

logger = logging.getLogger("SpectroHub")


class SPCExporter:
    """Zapisuje dane widmowe do binarnego formatu .spc (standard 0x4B float32)."""

    @staticmethod
    def export(file_path: str,
               wavenumbers: np.ndarray,
               intensities: np.ndarray,
               comment: str = "Wygenerowano w SpectroHub",
               source: str = "SpectroHub FTIR Platform") -> None:
        """
        Zapisuje parę wektorów X, Y do pojedynczego pliku .spc.
        Wymusza rosnącą oś X wymaganą przez nagłówek binarny SPC.
        """
        npts = len(wavenumbers)
        if npts != len(intensities):
            raise ValueError(f"Niezgodność wymiarów: X={npts}, Y={len(intensities)}")

        fxmin = float(np.min(wavenumbers))
        fxmax = float(np.max(wavenumbers))

        # Przygotowanie danych float32
        y_data = np.array(intensities, dtype=np.float32)

        # Standard binarny SPC wymaga, aby punkty w pliku były uporządkowane od fxmin do fxmax
        if wavenumbers[0] > wavenumbers[-1]:
            y_data = y_data[::-1]

        # 1. Główny nagłówek SPC (512 bajtów)
        main_hdr = bytearray(512)
        struct.pack_into('<B', main_hdr, 0, 0x00)     # ftflgs: pojedyncze widmo, równy krok X
        struct.pack_into('<B', main_hdr, 1, 0x4B)     # fversn: Nowy format SPC (0x4B)
        struct.pack_into('<B', main_hdr, 2, 1)        # fexper: FTIR / IR
        struct.pack_into('<b', main_hdr, 3, -128)     # fexp: -128 (0x80) -> czyste float32
        struct.pack_into('<I', main_hdr, 4, npts)     # fnpts: liczba punktów
        struct.pack_into('<d', main_hdr, 8, fxmin)    # fxmin: początek zakresu
        struct.pack_into('<d', main_hdr, 16, fxmax)   # fxmax: koniec zakresu
        struct.pack_into('<i', main_hdr, 24, 1)       # fnsub: 1 podwidmo
        struct.pack_into('<B', main_hdr, 28, 1)       # fxtype: 1 = cm^-1
        struct.pack_into('<B', main_hdr, 29, 1)       # fytype: 1 = Absorbancja
        struct.pack_into('<I', main_hdr, 508, 0)      # flogoff: 0 = brak bloku logu

        # Etykiety tekstowe w nagłówku
        b_comment = comment.encode('ascii', errors='replace')[:129]
        b_source = source.encode('ascii', errors='replace')[:29]
        main_hdr[44:44 + len(b_comment)] = b_comment
        main_hdr[174:174 + len(b_source)] = b_source

        # 2. Nagłówek podwidma (32 bajty)
        sub_hdr = bytearray(32)
        struct.pack_into('<B', sub_hdr, 0, 0x00)      # subflgs: 0
        struct.pack_into('<b', sub_hdr, 1, -128)      # subexp: -128 (float32)
        struct.pack_into('<H', sub_hdr, 2, 0)         # subindex: 0
        struct.pack_into('<I', sub_hdr, 16, npts)     # subnpts: liczba punktów
        struct.pack_into('<I', sub_hdr, 20, 1)        # subscan: 1 skan

        # 3. Zapis binarny
        with open(file_path, 'wb') as f:
            f.write(main_hdr)
            f.write(sub_hdr)
            f.write(y_data.tobytes())

        logger.info(f"Wyeksportowano pomyślnie widmo do binarnego SPC: '{file_path}' ({npts} pkt.)")

    @classmethod
    def export_spectrum(cls, spectrum: SpectrumData, file_path: str) -> None:
        """Wygodna metoda eksportująca bezpośrednio obiekt SpectrumData."""
        cls.export(
            file_path=file_path,
            wavenumbers=spectrum.wavenumbers,
            intensities=spectrum.intensities,
            comment=f"Widmo: {spectrum.name}",
            source="SpectroHub FTIR"
        )