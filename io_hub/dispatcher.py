"""
io_hub/dispatcher.py
Uniwersalny menedżer importu plików spektroskopowych z autodetekcją formatu.
"""

import os
import re
from typing import List
import logging

from core.models import SpectrumData
from core.logger import log_algo_exception
from io_hub.readers.opus_reader import OPUSReader
from io_hub.readers.omnic_reader import OMNICReader
from io_hub.readers.spc_reader import SPCReader
from io_hub.readers.ascii_reader import ASCIIReader
from io_hub.readers.jcamp_reader import JCAMPReader

logger = logging.getLogger("SpectroHub")


class SpectrumImportDispatcher:
    """Fabryka parserów spektroskopowych."""

    _READERS = {
        'spc': SPCReader(),
        'spa': OMNICReader(),
        'spg': OMNICReader(),
        'dx': JCAMPReader(),
        'jdx': JCAMPReader(),
    }
    _ASCII_READER = ASCIIReader()
    _OPUS_READER = OPUSReader()

    @classmethod
    def load_file(cls, file_path: str) -> List[SpectrumData]:
        """
        Główna metoda importu. Bezpiecznie wczytuje dowolny obsługiwany format widma.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Plik nie istnieje: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        try:
            # 1. Sprawdzenie czy to Bruker OPUS (.0, .1, .opus lub brak rozszerzenia z sygnaturą)
            if re.match(r'^\.\d+$', ext) or ext in ['.opus']:
                logger.info(f"Wczytywanie Bruker OPUS: {filename}")
                return cls._OPUS_READER.read(file_path)

            # 2. Dedykowane czytniki po rozszerzeniu
            ext_clean = ext.lstrip('.')
            if ext_clean in cls._READERS:
                logger.info(f"Wczytywanie [{ext_clean.upper()}]: {filename}")
                return cls._READERS[ext_clean].read(file_path)

            # 3. Sprawdzenie sygnatury binarnej (gdy plik OPUS nie ma typowego rozszerzenia)
            with open(file_path, 'rb') as f:
                header_magic = f.read(4)
            if header_magic == b'\n\n\xfe\xfe':
                logger.info(f"Wykryto sygnaturę OPUS w pliku: {filename}")
                return cls._OPUS_READER.read(file_path)

            # 4. Fallback na pliki tekstowe (CSV / DPT / TXT)
            logger.info(f"Wczytywanie formatu tekstowego (ASCII/CSV): {filename}")
            return cls._ASCII_READER.read(file_path)

        except Exception as e:
            log_algo_exception(logger, f"load_file: {filename}", params={"path": file_path})
            raise IOError(f"Błąd odczytu pliku '{filename}': {str(e)}") from e