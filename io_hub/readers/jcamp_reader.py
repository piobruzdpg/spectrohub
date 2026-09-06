"""
io_hub/readers/jcamp_reader.py
Czytnik i eksporter formatu IUPAC JCAMP-DX (.dx, .jdx).
"""

import os
import re
import numpy as np
from typing import List

from core.models import SpectrumData
from io_hub.readers.base_reader import BaseSpectrumReader


class JCAMPReader(BaseSpectrumReader):
    """Odczyt plików JCAMP-DX."""

    def read(self, file_path: str) -> List[SpectrumData]:
        wavenumbers, intensities = [], []
        in_data = False

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('$$'):
                    continue

                if line_str.upper().startswith('##XYDATA='):
                    in_data = True
                    continue

                if in_data:
                    if line_str.startswith('##END=') or (line_str.startswith('##') and not line_str.upper().startswith('##XYDATA=')):
                        break
                    parts = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line_str)
                    if len(parts) >= 2:
                        if len(parts) % 2 == 0:
                            for i in range(0, len(parts), 2):
                                wavenumbers.append(float(parts[i]))
                                intensities.append(float(parts[i + 1]))
                        else:
                            wavenumbers.append(float(parts[0]))
                            intensities.append(float(parts[1]))

        if not wavenumbers:
            raise ValueError(f"Nie znaleziono danych w pliku JCAMP: {os.path.basename(file_path)}")

        x_clean, y_clean = self.sanitize_and_sort(np.array(wavenumbers), np.array(intensities))
        filename = os.path.splitext(os.path.basename(file_path))[0]

        return [SpectrumData(
            wavenumbers=x_clean,
            intensities=y_clean,
            name=filename,
            file_path=file_path,
            file_type="jcamp"
        )]