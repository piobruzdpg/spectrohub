"""
io_hub/readers/opus_reader.py
Natywny dekoder binarnego formatu Bruker OPUS bez zewnętrznych zależności.
"""

import os
import re
import struct
import datetime
import numpy as np
from typing import List, Optional, Tuple
import logging

from core.models import SpectrumData
from io_hub.readers.base_reader import BaseSpectrumReader

logger = logging.getLogger("SpectroHub")

# [Kody bloków CODE_0..CODE_5, STRUCT_3D_INFO_BLOCK zachowane z Twojego pliku]
CODE_0 = {0: '', 1: 'Real Part of Complex Data', 2: 'Imaginary Part of Complex Data', 3: ''}
CODE_1 = {0: '', 1: 'Sample', 2: 'Reference', 3: ''}
CODE_2 = {
    0: '', 1: 'Data Parameters', 2: 'Instrument Parameters', 3: 'Acquisition Parameters',
    4: 'Fourier Transform Parameters', 5: 'Plot and Display Parameters', 6: 'Optical Parameters',
    7: 'GC Parameters', 8: 'Library Search Parameters', 9: 'Communication Parameters',
    10: 'Sample Origin Parameters', 11: 'Lab and Process Parameters',
}
CODE_3 = {
    0: '', 1: 'Spectrum', 2: 'Interferogram', 3: 'Phase', 4: 'Absorbance', 5: 'Transmittance',
    6: 'Kubelka-Munk', 7: 'Trace (Intensity over time)', 8: 'gc File, Series of Interferograms',
    9: 'gc File, Series of Spectra', 10: 'Raman', 11: 'Emisson', 12: 'Reflectance',
    13: 'Directory', 14: 'Power', 15: 'log Reflectance', 16: 'ATR', 17: 'Photoacoustic',
    18: 'Result of Arithmatics, looks like Transmittance', 19: 'Result of Arithmatics, looks like Absorbance',
    22: 'Match',
}
CODE_4 = {0: '', 1: 'First Derivative', 2: 'Second Derivative', 3: 'n-th Derivative'}
CODE_5 = {0: '', 1: 'Compound Information', 2: '(Series)', 3: 'Molecular Structure', 4: '(Compact)', 5: 'History/Report'}

CODE_3_ABR = {
    0: '', 1: '', 2: 'ig', 3: 'ph', 4: 'a', 5: 't', 6: 'km', 7: 'tr',
    8: 'gcig', 9: 'gcsc', 10: 'ra', 11: 'e', 12: 'r', 13: 'dir', 14: 'pw',
    15: 'logr', 16: 'atr', 17: 'pas', 18: 'arit', 19: 'aria', 22: 'match',
}

TYPE_CODE_LABELS = [CODE_0, CODE_1, CODE_2, CODE_3, CODE_4, CODE_5]


class BlockType(tuple):
    def get_label(self):
        labels = [self._get_sub_type_label(idx) for idx in range(len(self)) if self[idx] > 0 and self._get_sub_type_label(idx) != '']
        return ' '.join(labels) if labels else 'Undefined'

    def _get_sub_type_label(self, pos_idx: int):
        if pos_idx == 3:
            channels, type_idx = divmod(self[3], 32)
        else:
            channels, type_idx = 0, self[pos_idx]
        try:
            label = TYPE_CODE_LABELS[pos_idx][type_idx]
        except KeyError:
            label = f'Unknown {pos_idx} {type_idx}'
        if channels > 0:
            label += f' {channels + 1}-Channel'
        return label


class FileBlock:
    __slots__ = ('type', 'size', 'start', 'bytes', 'data', 'parser', 'parse_error', 'keys', 'debug')

    def __init__(self, filebytes: bytes, block_type: tuple, size: int, start: int, debug=False):
        self.bytes = filebytes[start: start + size]
        self.type = BlockType(block_type)
        self.size = size
        self.start = start
        self.debug = debug
        self.data = None
        self.parser = None
        self.parse_error = None

    def is_data_status(self): return self.type[2] == 1
    def is_rf_param(self): return self.type[2] > 1 and self.type[1] == 2
    def is_param(self): return self.type[2] > 0 or self.type == (0, 0, 0, 0, 0, 1)
    def is_sm_param(self): return self.is_param() and not self.is_data_status() and not self.is_rf_param()
    def is_directory(self): return self.type == (0, 0, 0, 13, 0, 0)
    def is_file_log(self): return self.type == (0, 0, 0, 0, 0, 5)
    def is_data(self): return self.type[2] == 0 and self.type[3] not in [0, 13] and self.type[5] not in [2, 5]
    def is_compact_data(self): return self.is_data() and self.type[5] == 4

    def get_data_key(self):
        if self.is_data():
            channels = self.type[3] // 32 + 1
            type_idx = self.type[3] % 32
            key = CODE_3_ABR.get(type_idx, f'_{self.type[3]}')
            if self.type[1] == 1: key += 'sm'
            elif self.type[1] == 2: key += 'rf'
            elif self.type[1] > 3: key += f'_{self.type[1]}'
            if channels > 1: key += f'_{channels}ch'
            if self.type[5] == 4: key += '_c'
            return key
        return None

    def parse(self):
        try:
            if self.is_directory():
                loc, blocks = 0, []
                while loc < len(self.bytes):
                    t_int, s_int, start = struct.unpack_from('<3i', self.bytes, loc)
                    loc += 12
                    if start > 0:
                        type_bit_str = format(t_int, '#034b')
                        b_type = (
                            int(type_bit_str[-2:], 2), int(type_bit_str[-4:-2], 2),
                            int(type_bit_str[-10:-4], 2), int(type_bit_str[-17:-10], 2),
                            int(type_bit_str[-19:-17], 2), int(type_bit_str[-22:-19], 2)
                        )
                        blocks.append((b_type, s_int * 4, start))
                    else:
                        break
                self.data = blocks
            elif self.is_param():
                loc, params = 0, {}
                while loc < len(self.bytes):
                    key = self.bytes[loc:loc + 3].decode('utf-8', errors='ignore')
                    if key == 'END':
                        break
                    dtype_code, val_size = struct.unpack_from('<2h', self.bytes, loc + 4)
                    val_size *= 2
                    if dtype_code == 0:
                        val = struct.unpack_from('<i', self.bytes, loc + 8)[0]
                    elif dtype_code == 1:
                        val = struct.unpack_from('<d', self.bytes, loc + 8)[0]
                    else:
                        fmt = f"<{val_size}s"
                        raw_s = struct.unpack_from(fmt, self.bytes, loc + 8)[0]
                        null_idx = raw_s.find(b'\x00')
                        val = raw_s[:null_idx].decode('latin-1') if null_idx != -1 else raw_s.decode('latin-1')
                    params[key.lower()] = val
                    loc += val_size + 8
                self.data = params
            elif self.is_data():
                count = round(len(self.bytes) / 4)
                self.data = np.frombuffer(self.bytes, dtype=np.float32, count=count)
        except Exception as e:
            self.parse_error = str(e)
            logger.debug(f"Błąd parsowania bloku OPUS: {e}")


class Parameters:
    def __init__(self, blocks: list):
        self._params = {}
        for block in blocks:
            if isinstance(block.data, dict):
                self._params.update(block.data)

    def __getattr__(self, name):
        name_lower = name.lower()
        if name_lower in self._params:
            return self._params[name_lower]
        raise AttributeError(f"Parametr '{name}' nie istnieje w metadanych OPUS.")

    def get(self, key, default=None):
        return self._params.get(key.lower(), default)

    def extract_datetime(self) -> Optional[datetime.datetime]:
        if 'dat' in self._params and 'tim' in self._params:
            try:
                date_str = str(self._params['dat']).strip()
                time_str = str(self._params['tim']).strip().split(' ')[0]
                dt_str = f"{date_str}-{time_str}"
                for fmt in ('%d/%m/%Y-%H:%M:%S.%f', '%d/%m/%Y-%H:%M:%S', '%Y/%m/%d-%H:%M:%S.%f', '%Y/%m/%d-%H:%M:%S'):
                    try:
                        return datetime.datetime.strptime(dt_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
        return None


class OPUSReader(BaseSpectrumReader):
    """Główny czytnik plików Bruker OPUS zwracający instancję SpectrumData."""

    def read(self, file_path: str) -> List[SpectrumData]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Nie znaleziono pliku OPUS: {file_path}")

        with open(file_path, 'rb') as f:
            filebytes = f.read()

        if len(filebytes) < 500 or filebytes[:4] != b'\n\n\xfe\xfe':
            raise ValueError(f"Plik '{os.path.basename(file_path)}' nie jest poprawnym plikiem binarnej struktury Bruker OPUS.")

        version, dir_start, max_blocks, num_blocks = struct.unpack_from('<di2i', filebytes, 4)
        dir_bytes = filebytes[dir_start : dir_start + max_blocks * 12]

        # Parsowanie bloków
        root_block = FileBlock(filebytes, (0, 0, 0, 13, 0, 0), len(dir_bytes), dir_start)
        root_block.parse()
        raw_blocks_info = root_block.data or []

        blocks = []
        for b_type, b_size, b_start in raw_blocks_info:
            fb = FileBlock(filebytes, b_type, b_size, b_start)
            fb.parse()
            blocks.append(fb)

        # Łączenie parametrów
        param_blocks = [b for b in blocks if b.is_sm_param() and isinstance(b.data, dict)]
        params = Parameters(param_blocks)

        # Znalezienie dopasowań Data <-> Status
        data_status = [b for b in blocks if b.is_data_status() and isinstance(b.data, dict)]
        data_blocks = [b for b in blocks if b.is_data() and b.data is not None]

        matched_data = {}
        for d in data_blocks:
            status_match = [s for s in data_status if s.type[:2] == d.type[:2] and s.type[3:] == d.type[3:]]
            if status_match:
                key = d.get_data_key()
                matched_data[key] = (d, Parameters([status_match[0]]))

        preferred_keys = ['a', 'ab', 'atr', 'tr', 're', 'rf', 'sc', 'sm', 'a_2', 'atr_2', 'tr_2', 'r', 'km']
        chosen_key = next((k for k in preferred_keys if k in matched_data), None)

        if not chosen_key:
            chosen_key = next(iter(matched_data.keys()), None)

        if not chosen_key:
            raise ValueError(f"Nie odnaleziono bloku danych widmowych w pliku Bruker: {os.path.basename(file_path)}")

        d_block, s_params = matched_data[chosen_key]
        npt = s_params.get('npt', len(d_block.data))
        fxv = s_params.get('fxv', 4000.0)
        lxv = s_params.get('lxv', 400.0)
        csf = s_params.get('csf', 1.0)

        y_raw = csf * d_block.data[:npt]
        x_raw = np.linspace(fxv, lxv, npt)

        x_clean, y_clean = self.sanitize_and_sort(x_raw, y_raw)

        # Czas pomiaru
        dt = params.extract_datetime()
        time_str = dt.strftime("%H:%M:%S") if dt else "00:00:00"

        filename = os.path.basename(file_path)
        metadata = {
            "resolution": params.get("res", "N/A"),
            "sample_scans": params.get("nss", "N/A"),
            "sample_name": params.get("snm", filename),
            "date": dt.strftime("%Y-%m-%d") if dt else "N/A",
            "block_key": chosen_key
        }

        return [SpectrumData(
            wavenumbers=x_clean,
            intensities=y_clean,
            name=filename,
            file_path=file_path,
            file_type="opus",
            time_str=time_str,
            units_x="cm^-1",
            units_y="absorbance" if "a" in chosen_key else "intensity",
            metadata=metadata
        )]