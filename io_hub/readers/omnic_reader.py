"""
io_hub/readers/omnic_reader.py
Natywny dekoder binarnych formatów Thermo OMNIC (.spa i .spg).
"""

import os
import struct
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List
import logging

from core.models import SpectrumData
from io_hub.readers.base_reader import BaseSpectrumReader

logger = logging.getLogger("SpectroHub")


class OMNICReader(BaseSpectrumReader):
    """Czytnik plików .spa oraz .spg pakietu Thermo OMNIC."""

    def read(self, file_path: str) -> List[SpectrumData]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.spa':
            return self._read_spa(file_path)
        elif ext == '.spg':
            return self._read_spg(file_path)
        else:
            raise ValueError(f"Nieobsługiwane rozszerzenie OMNIC: {ext}")

    def _read_spa(self, file_path: str) -> List[SpectrumData]:
        with open(file_path, 'rb') as f:
            content = f.read()

        if len(content) < 300:
            raise ValueError(f"Plik .spa jest uszkodzony: {os.path.basename(file_path)}")

        # Stempel czasu (sekundy od 31.12.1899)
        time_str = "00:00:00"
        try:
            ts_sec = struct.unpack('<I', content[296:300])[0]
            if ts_sec > 0:
                dt_base = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc)
                dt_acq = dt_base + timedelta(seconds=int(ts_sec))
                time_str = dt_acq.strftime("%H:%M:%S")
        except Exception:
            pass

        # Skanowanie katalogu
        pos = 304
        info, y_raw = None, None
        while pos + 10 <= len(content):
            key = content[pos]
            if key in (0, 1) and pos > 304:
                break
            if key == 2:
                pos_header = struct.unpack('<I', content[pos + 2 : pos + 6])[0]
                if 0 < pos_header < len(content):
                    info = self._parse_header(content, pos_header)
            elif key == 3:
                pos_y = struct.unpack('<I', content[pos + 2 : pos + 6])[0]
                size_y = struct.unpack('<I', content[pos + 6 : pos + 10])[0]
                if 0 < pos_y < len(content) and size_y > 0:
                    raw_chunk = content[pos_y : min(len(content), pos_y + size_y)]
                    n_floats = len(raw_chunk) // 4
                    y_raw = np.frombuffer(raw_chunk[:n_floats * 4], dtype=np.float32).astype(float)
            pos += 16

        if y_raw is None:
            raise ValueError(f"Brak danych Y w pliku .spa: {os.path.basename(file_path)}")

        first_x = info["firstx"] if info else 4000.0
        last_x = info["lastx"] if info else 400.0
        x_raw = np.linspace(first_x, last_x, len(y_raw), dtype=float)

        x_clean, y_clean = self.sanitize_and_sort(x_raw, y_raw)
        filename = os.path.basename(file_path)

        return [SpectrumData(
            wavenumbers=x_clean,
            intensities=y_clean,
            name=os.path.splitext(filename)[0],
            file_path=file_path,
            file_type="spa",
            time_str=time_str,
            units_y=info.get("units", "absorbance") if info else "absorbance"
        )]

    def _read_spg(self, file_path: str) -> List[SpectrumData]:
        with open(file_path, 'rb') as f:
            content = f.read()

        if len(content) < 304:
            raise ValueError(f"Plik .spg jest zbyt krótki: {os.path.basename(file_path)}")

        nlines = struct.unpack('<H', content[294:296])[0] if len(content) >= 296 else 0
        if nlines == 0:
            nlines = (len(content) - 304) // 16

        keys, pos = [], 304
        for _ in range(nlines):
            if pos + 10 > len(content): break
            keys.append(content[pos])
            pos += 16

        idx_02 = [i for i, k in enumerate(keys) if k == 2]
        idx_03 = [i for i, k in enumerate(keys) if k == 3]
        idx_6b = [i for i, k in enumerate(keys) if k == 107]
        nspec = min(len(idx_02), len(idx_03))

        spectra = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dt_base = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc)

        for i in range(nspec):
            try:
                pos_02 = 304 + 16 * idx_02[i]
                pos_header = struct.unpack('<I', content[pos_02 + 2 : pos_02 + 6])[0]
                info = self._parse_header(content, pos_header)

                pos_03 = 304 + 16 * idx_03[i]
                pos_y = struct.unpack('<I', content[pos_03 + 2 : pos_03 + 6])[0]
                size_y = struct.unpack('<I', content[pos_03 + 6 : pos_03 + 10])[0]

                raw_chunk = content[pos_y : pos_y + size_y]
                y_raw = np.frombuffer(raw_chunk[:(len(raw_chunk) // 4) * 4], dtype=np.float32).astype(float)

                time_str = "00:00:00"
                sub_name = f"{base_name}_{i + 1:02d}"

                if i < len(idx_6b):
                    pos_6b = 304 + 16 * idx_6b[i]
                    title_pos = struct.unpack('<I', content[pos_6b + 2 : pos_6b + 6])[0]
                    if 0 < title_pos < len(content):
                        if title_pos + 260 <= len(content):
                            ts_sec = struct.unpack('<I', content[title_pos + 256 : title_pos + 260])[0]
                            if ts_sec > 0:
                                dt_acq = dt_base + timedelta(seconds=int(ts_sec))
                                time_str = dt_acq.strftime("%H:%M:%S")

                x_raw = np.linspace(info["firstx"], info["lastx"], len(y_raw), dtype=float)
                x_clean, y_clean = self.sanitize_and_sort(x_raw, y_raw)

                spectra.append(SpectrumData(
                    wavenumbers=x_clean,
                    intensities=y_clean,
                    name=sub_name,
                    file_path=file_path,
                    file_type="spg",
                    time_str=time_str,
                    units_y=info.get("units", "absorbance")
                ))
            except Exception as e:
                logger.debug(f"Pominięto pod-widmo {i} w .spg z powodu błędu: {e}")

        return spectra

    def _parse_header(self, content, pos):
        out = {"nx": 0, "units": "absorbance", "firstx": 4000.0, "lastx": 400.0}
        if pos + 208 <= len(content):
            out["nx"] = struct.unpack('<I', content[pos + 4 : pos + 8])[0]
            y_key = content[pos + 12]
            out["units"] = "transmittance" if y_key == 16 else "absorbance"
            out["firstx"] = float(struct.unpack('<f', content[pos + 16 : pos + 20])[0])
            out["lastx"] = float(struct.unpack('<f', content[pos + 20 : pos + 24])[0])
        return out