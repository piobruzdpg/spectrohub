"""
io_hub/readers/spc_reader.py
Hybrydowy czytnik i pisarz binarnego formatu Thermo Grams SPC.
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
    """Obsługa odczytu plików Thermo/Galactic Grams .spc."""

    head_str = "cccciddicccci9s9sh32s130s30siicchf48sfifc187s"
    old_head_str = "<cchfffcchcccc8shh28s130s30s32s"

    def read(self, file_path: str) -> List[SpectrumData]:
        with open(file_path, "rb") as fin:
            content = fin.read()

        if len(content) < 256:
            raise ValueError(f"Plik SPC jest zbyt krótki: {os.path.basename(file_path)}")

        ftflg, fversn = struct.unpack('cc', content[:2])
        endian = ">" if fversn == b'\x4c' else "<"

        sub_spectra = []
        time_str = "00:00:00"

        if fversn in (b'\x4b', b'\x4c'):
            fmt = endian + self.head_str
            unpacked = struct.unpack(fmt.encode('ascii'), content[:512])

            ftflg = ord(unpacked[0])
            fexp = ord(unpacked[3])
            fnpts = int(unpacked[4])
            ffirst = float(unpacked[5])
            flast = float(unpacked[6])
            fnsub = int(unpacked[7])

            tsprec = bool(ftflg & 0x01)
            tmulti = bool(ftflg & 0x04)
            txyxys = bool(ftflg & 0x40)
            txvals = bool(ftflg & 0x80)

            # Czas
            d = unpacked[12]
            if d > 0:
                hour = (d >> 6) & 0x1F
                minute = d & 0x3F
                time_str = f"{hour:02d}:{minute:02d}:00"

            # Oś X
            if not txyxys:
                if txvals:
                    x_bytes = content[512 : 512 + (4 * fnpts)]
                    x_raw = np.frombuffer(x_bytes, dtype=endian + "f4").astype(float)
                    sub_pos = 512 + (4 * fnpts)
                else:
                    x_raw = np.linspace(ffirst, flast, num=fnpts, dtype=float)
                    sub_pos = 512
            else:
                sub_pos = 512
                x_raw = None

            # Podpliki Y
            for sub_idx in range(fnsub):
                if sub_pos + 32 > len(content): break
                subexp = struct.unpack(endian + 'b', content[sub_pos + 1 : sub_pos + 2])[0]
                exp = subexp if tmulti else fexp
                if not (-128 < exp <= 128): exp = 0

                y_pos = sub_pos + 32
                if exp in (128, -128):
                    y_raw = np.frombuffer(content[y_pos : y_pos + 4 * fnpts], dtype=endian + "f4").astype(float)
                elif tsprec:
                    raw_int = np.frombuffer(content[y_pos : y_pos + 2 * fnpts], dtype=endian + "i2")
                    y_raw = (2.0 ** (exp - 16)) * raw_int.astype(float)
                else:
                    raw_int = np.frombuffer(content[y_pos : y_pos + 4 * fnpts], dtype=endian + "i4")
                    y_raw = (2.0 ** (exp - 32)) * raw_int.astype(float)

                sub_spectra.append(y_raw)
                sub_pos += (8 * fnpts + 32) if txyxys else (4 * fnpts + 32)

        elif fversn == b'\x4d':  # Stary format
            unpacked = struct.unpack(self.old_head_str.encode('ascii'), content[:256])
            onpts = int(unpacked[3])
            x_raw = np.linspace(float(unpacked[4]), float(unpacked[5]), num=onpts, dtype=float)
            ohour = int(unpacked[11]) if isinstance(unpacked[11], int) else ord(unpacked[11])
            ominute = int(unpacked[12]) if isinstance(unpacked[12], int) else ord(unpacked[12])
            time_str = f"{ohour:02d}:{ominute:02d}:00"

            sub_pos = 224
            if sub_pos + 32 + (4 * onpts) <= len(content):
                y_raw = np.frombuffer(content[sub_pos + 32 : sub_pos + 32 + 4 * onpts], dtype='<f4').astype(float)
                sub_spectra.append(y_raw)
        else:
            raise ValueError(f"Nieznana sygnatura formatu SPC: {fversn}")

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        results = []

        for idx, y_arr in enumerate(sub_spectra):
            sub_name = f"{base_name}_{idx + 1}" if len(sub_spectra) > 1 else base_name
            x_clean, y_clean = self.sanitize_and_sort(x_raw, y_arr)
            results.append(SpectrumData(
                wavenumbers=x_clean,
                intensities=y_clean,
                name=sub_name,
                file_path=file_path,
                file_type="spc",
                time_str=time_str
            ))

        return results


def write_spc_file(file_path: str, wavenumbers: np.ndarray, intensities: np.ndarray, title: str = "IR Spectrum"):
    """Zapisuje pojedyncze widmo do binarnego pliku Galactic SPC (nowy format 0x4B)[cite: 14]."""
    npts = len(wavenumbers)
    if npts == 0 or len(intensities) == 0:
        raise ValueError("Wektory widmowe nie mogą być puste.")

    fxmin = float(np.min(wavenumbers))
    fxmax = float(np.max(wavenumbers))
    y_data = np.array(intensities, dtype=np.float32)

    # SPC wymaga danych uporządkowanych zgodnie z fxmin -> fxmax
    if wavenumbers[0] > wavenumbers[-1]:
        y_data = y_data[::-1]

    main_hdr = bytearray(512)
    struct.pack_into('<B', main_hdr, 0, 0x00)
    struct.pack_into('<B', main_hdr, 1, 0x4B)
    struct.pack_into('<B', main_hdr, 2, 1)
    struct.pack_into('<b', main_hdr, 3, -128)
    struct.pack_into('<I', main_hdr, 4, npts)
    struct.pack_into('<d', main_hdr, 8, fxmin)
    struct.pack_into('<d', main_hdr, 16, fxmax)
    struct.pack_into('<i', main_hdr, 24, 1)

    comment_bytes = title.encode('utf-8', errors='ignore')[:30]
    main_hdr[44 : 44 + len(comment_bytes)] = comment_bytes
    main_hdr[174 : 174 + len(b"SpectroHub")] = b"SpectroHub"

    sub_hdr = bytearray(32)
    struct.pack_into('<B', sub_hdr, 0, 0x00)
    struct.pack_into('<b', sub_hdr, 1, -128)
    struct.pack_into('<I', sub_hdr, 16, npts)

    with open(file_path, 'wb') as f:
        f.write(main_hdr)
        f.write(sub_hdr)
        f.write(y_data.tobytes())