"""
io_hub/readers/spc_reader.py
Natywny dekoder binarnych plików Thermo Grams SPC (.spc).
Obsługuje format New (0x4B), Old (0x4D), sub-widma oraz dane stało- i zmiennoprzecinkowe.
"""

import os
import struct
import numpy as np
from typing import List
import logging

from core.models import SpectrumData
from io_hub.readers.base_reader import BaseSpectrumReader

logger = logging.getLogger("SpectroHub")


class SPCReader(BaseSpectrumReader):
    """Natywny dekoder plików binarnych Thermo Grams SPC."""

    SUPPORTED_EXTENSIONS = ['.spc', '.SPC']

    def read(self, file_path: str) -> List[SpectrumData]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Plik nie istnieje: {file_path}")

        with open(file_path, 'rb') as f:
            content = f.read()

        if len(content) < 512:
            raise ValueError(f"Plik SPC za mały na poprawny nagłówek ({len(content)} B)")

        # Odczyt wersji: bajt 1
        ftflgs, fversn = struct.unpack('<BB', content[:2])

        if fversn == 0x4B:
            return self._parse_new_spc(content, file_path, ftflgs)
        elif fversn == 0x4D:
            return self._parse_old_spc(content, file_path, ftflgs)
        else:
            raise NotImplementedError(f"Nieobsługiwany format SPC: 0x{fversn:02X}")

    def _parse_new_spc(self, content: bytes, file_path: str, ftflgs: int) -> List[SpectrumData]:
        """Dekodowanie nowego formatu SPC (0x4B / LSB)."""
        # Rozpakowanie nagłówka 512 bajtów
        head_str = "<BBbBiddicccci9s9sh32s130s30siicchf48sfifc187s"
        unpacked = struct.unpack(head_str, content[:512])

        fexp = unpacked[2]          # b: signed char (wykładnik skali z nagłówka głównego)
        fnpts = unpacked[4]         # i: liczba punktów
        ffirst = unpacked[5]        # d: początek osi X
        flast = unpacked[6]         # d: koniec osi X
        fnsub = unpacked[7]         # i: liczba podwidm
        ffactor = unpacked[24]      # f: offset 260, opcjonalny mnożnik (0.0 oznacza brak/1.0)
        multiplier = float(ffactor) if ffactor != 0.0 else 1.0

        # Flagi binarne
        tsprec = bool(ftflgs & 0x01)  # 16-bit integer if true
        txyxys = bool(ftflgs & 0x40)  # each subfile has own x
        txvals = bool(ftflgs & 0x80)  # global x preceeds y

        # Budowanie osi X
        sub_pos = 512
        if not txyxys:
            if txvals:
                x_end = 512 + (4 * fnpts)
                wavenumbers = np.frombuffer(content[512:x_end], dtype=np.float32).astype(np.float64)
                sub_pos = x_end
            else:
                wavenumbers = np.linspace(ffirst, flast, num=fnpts, dtype=np.float64)
        else:
            wavenumbers = None

        # Odczyt sub-widm
        spectra_list = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for sub_idx in range(fnsub):
            if sub_pos + 32 > len(content):
                break

            # Subheader 32 bajty
            sub_hdr = struct.unpack("<BBhfffiif4s", content[sub_pos:sub_pos + 32])
            sub_exp = struct.unpack('<b', bytes([sub_hdr[1]]))[0]
            sub_npts = sub_hdr[6] if sub_hdr[6] > 0 else fnpts
            dat_pos = sub_pos + 32

            # Wybór właściwego wykładnika skali:
            # sub_exp ma priorytet (nawet przy fnsub == 1). fexp stanowi fallback, jeśli sub_exp == 0.
            is_float = (sub_exp in (-128, 128)) or (fexp in (-128, 128))
            exp_val = sub_exp if sub_exp != 0 else fexp

            # Oś X dla txyxys
            cur_x = wavenumbers
            if txyxys:
                x_bytes = 4 * sub_npts
                raw_x = np.frombuffer(content[dat_pos:dat_pos + x_bytes], dtype=np.int32)
                cur_x = (2.0 ** (exp_val - 32)) * raw_x.astype(np.float64)
                dat_pos += x_bytes

            # Oś Y
            if is_float:
                # Dane zmiennoprzecinkowe float32
                y_bytes = 4 * sub_npts
                cur_y = np.frombuffer(content[dat_pos:dat_pos + y_bytes], dtype=np.float32).astype(np.float64)
                dat_pos += y_bytes
            else:
                # Dane całkowite skalowane potęgą 2
                if tsprec:
                    y_bytes = 2 * sub_npts
                    raw_y = np.frombuffer(content[dat_pos:dat_pos + y_bytes], dtype=np.int16)
                    cur_y = (2.0 ** (exp_val - 16)) * raw_y.astype(np.float64)
                else:
                    y_bytes = 4 * sub_npts
                    raw_y = np.frombuffer(content[dat_pos:dat_pos + y_bytes], dtype=np.int32)
                    cur_y = (2.0 ** (exp_val - 32)) * raw_y.astype(np.float64)
                dat_pos += y_bytes

            # Zastosowanie ewentualnego mnożnika nagłówka głównego
            if multiplier != 1.0:
                cur_y = cur_y * multiplier

            sub_pos = dat_pos
            spec_name = base_name if fnsub == 1 else f"{base_name}_sub{sub_idx + 1}"

            # Sanityzacja osi (uporządkowanie monotoniczne malejąco dla FTIR: np. 4000 -> 400 cm^-1)
            if cur_x[0] < cur_x[-1]:
                cur_x = cur_x[::-1]
                cur_y = cur_y[::-1]

            spectra_list.append(SpectrumData(
                wavenumbers=cur_x,
                intensities=cur_y,
                name=spec_name,
                file_path=file_path,
                file_type="spc",
                metadata={"fnsub": fnsub, "sub_index": sub_idx, "fversn": "0x4B", "exp_val": exp_val}
            ))

        return spectra_list

    def _parse_old_spc(self, content: bytes, file_path: str, ftflgs: int) -> List[SpectrumData]:
        """Dekodowanie starszego formatu SPC (0x4D)."""
        old_hdr = struct.unpack("<BBhfffcchcccc8shh28s130s30s32s", content[:256])
        oexp = struct.unpack('<b', bytes([old_hdr[2]]))[0]
        onpts = old_hdr[3]
        ofirst = old_hdr[4]
        olast = old_hdr[5]

        wavenumbers = np.linspace(ofirst, olast, num=onpts, dtype=np.float64)
        sub_pos = 256
        spectra_list = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        sub_idx = 0

        while sub_pos + 32 < len(content):
            sub_hdr = struct.unpack("<BBhfffiif4s", content[sub_pos:sub_pos + 32])
            sub_exp = struct.unpack('<b', bytes([sub_hdr[1]]))[0]
            pts = sub_hdr[6] if sub_hdr[6] > 0 else onpts

            dat_pos = sub_pos + 32
            exp = sub_exp if (sub_exp != 0 and sub_exp not in (-128, 128)) else oexp

            if sub_exp in [128, -128]:
                y_bytes = 4 * pts
                cur_y = np.frombuffer(content[dat_pos:dat_pos + y_bytes], dtype=np.float32).astype(np.float64)
            else:
                y_bytes = 4 * pts
                raw_bytes = np.frombuffer(content[dat_pos:dat_pos + y_bytes], dtype=np.uint8)
                y_int = []
                for i in range(0, len(raw_bytes), 4):
                    val = (raw_bytes[i + 1] * 16777216 + raw_bytes[i] * 65536 +
                           raw_bytes[i + 3] * 256 + raw_bytes[i + 2])
                    y_int.append(val)
                cur_y = np.int32(y_int) / (2.0 ** (32 - exp))

            sub_pos = dat_pos + y_bytes
            sub_idx += 1
            spec_name = base_name if sub_pos >= len(content) and sub_idx == 1 else f"{base_name}_sub{sub_idx}"

            cur_x = wavenumbers.copy()
            if cur_x[0] < cur_x[-1]:
                cur_x = cur_x[::-1]
                cur_y = cur_y[::-1]

            spectra_list.append(SpectrumData(
                wavenumbers=cur_x,
                intensities=cur_y,
                name=spec_name,
                file_path=file_path,
                file_type="spc",
                metadata={"sub_index": sub_idx, "fversn": "0x4D", "exp_val": exp}
            ))

        return spectra_list