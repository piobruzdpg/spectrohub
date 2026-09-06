"""
io_hub/readers/ascii_reader.py
Parser i pisarz plików tekstowych CSV / DPT / TXT.
"""

import os
import re
import numpy as np
from typing import List
import logging

from core.models import SpectrumData
from io_hub.readers.base_reader import BaseSpectrumReader

logger = logging.getLogger("SpectroHub")


class ASCIIReader(BaseSpectrumReader):
    """Obsługuje pliki CSV, DPT, TXT z autodetekcją formatowania."""

    def read(self, file_path: str) -> List[SpectrumData]:
        raw_text = None
        for enc in ('utf-8-sig', 'utf-8', 'cp1250', 'windows-1250', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    raw_text = f.read()
                break
            except Exception:
                continue

        if not raw_text:
            raise ValueError(f"Nie udało się odczytać pliku: {os.path.basename(file_path)}")

        raw_text = raw_text.replace('\xa0', ' ').replace('\ufeff', '')
        lines = raw_text.splitlines()
        parsed_rows = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith(('#', '!', '//')):
                continue

            if ';' in line:
                parts = [p.strip() for p in line.split(';') if p.strip()]
            elif '\t' in line:
                parts = [p.strip() for p in line.split('\t') if p.strip()]
            elif ',' in line:
                if 'e' in line.lower() or line.count(',') > 1:
                    tokens = re.findall(r'[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?', line)
                    parts = tokens
                else:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                parts = re.split(r'\s+', line)

            floats = []
            for p in parts:
                try:
                    floats.append(float(p.replace(',', '.')))
                except ValueError:
                    continue

            if len(floats) >= 2:
                parsed_rows.append((floats[0], floats[1]))

        if not parsed_rows:
            raise ValueError(f"Plik '{os.path.basename(file_path)}' nie zawiera danych numerycznych.")

        arr = np.array(parsed_rows, dtype=float)
        x_clean, y_clean = self.sanitize_and_sort(arr[:, 0], arr[:, 1])
        filename = os.path.splitext(os.path.basename(file_path))[0]

        return [SpectrumData(
            wavenumbers=x_clean,
            intensities=y_clean,
            name=filename,
            file_path=file_path,
            file_type="csv"
        )]


def write_csv_file(file_path: str, wavenumbers: np.ndarray, intensities: np.ndarray, style: str = 'EU', include_header: bool = False):
    """Zapisuje widmo do pliku CSV w formacie międzynarodowym lub polskim."""
    is_pl = style.upper() in ['PL', 'POLISH']
    sep = ';' if is_pl else ','

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        if include_header:
            header = f"Liczba_falowa [cm-1]{sep}Intensywnosc\n" if is_pl else f"Wavenumber [cm-1]{sep}Intensity\n"
            f.write(header)

        for wn, val in zip(wavenumbers, intensities):
            s_wn = f"{wn:.4f}".replace('.', ',') if is_pl else f"{wn:.4f}"
            s_val = f"{val:.6f}".replace('.', ',') if is_pl else f"{val:.6f}"
            f.write(f"{s_wn}{sep}{s_val}\n")