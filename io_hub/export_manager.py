"""
io_hub/export_manager.py
Centralny silnik eksportu widm pojedynczych, wsadowych oraz paczek ZIP.
Zapamiętuje ostatnio używany format w QSettings oraz automatycznie odwzorowuje hierarchię folderów/okien.
"""

import os
import re
import tempfile
import zipfile
import logging
from typing import List, Tuple, Dict, Any
from PySide6.QtCore import QSettings

from core.models import SpectrumData
from io_hub.exporters.ascii_exporter import ASCIIExporter
from io_hub.exporters.spc_exporter import SPCExporter
from io_hub.exporters.jcamp_exporter import JCAMPExporter

logger = logging.getLogger("SpectroHub")

EXPORT_FORMATS: Dict[str, Dict[str, Any]] = {
    "csv_pl": {
        "id": "csv_pl",
        "name": "CSV (Polski: ';' oraz przecinek dziesiętny)",
        "short_name": "CSV (PL)",
        "ext": ".csv",
        "filter": "Pliki CSV - Format polski (*.csv)"
    },
    "csv_eu": {
        "id": "csv_eu",
        "name": "CSV (Międzynarodowy: ',' oraz kropka dziesiętna)",
        "short_name": "CSV (EU/EN)",
        "ext": ".csv",
        "filter": "Pliki CSV - Format międzynarodowy (*.csv)"
    },
    "dpt_pl": {
        "id": "dpt_pl",
        "name": "DPT (Bruker/ASCII: Tabulator oraz przecinek)",
        "short_name": "DPT (PL)",
        "ext": ".dpt",
        "filter": "Pliki DPT - Format polski (*.dpt)"
    },
    "dpt_eu": {
        "id": "dpt_eu",
        "name": "DPT (Bruker/ASCII: Tabulator oraz kropka)",
        "short_name": "DPT (EU/EN)",
        "ext": ".dpt",
        "filter": "Pliki DPT - Format międzynarodowy (*.dpt)"
    },
    "spc": {
        "id": "spc",
        "name": "Thermo Grams SPC (Galactic 0x4B float32)",
        "short_name": "SPC",
        "ext": ".spc",
        "filter": "Pliki binarne SPC (*.spc)"
    },
    "jcamp": {
        "id": "jcamp",
        "name": "IUPAC JCAMP-DX (Wektorowe XY)",
        "short_name": "JCAMP-DX",
        "ext": ".dx",
        "filter": "Pliki JCAMP-DX (*.dx *.jdx)"
    }
}


class ExportManager:
    """Zarządza zapisem i eksportem pojedynczych i hurtowych zestawów widm."""

    @classmethod
    def sanitize_filename(cls, name: str) -> str:
        """Usuwa niedozwolone w nazwach plików i katalogów znaki systemowe."""
        clean = re.sub(r'[\\/*?:"<>|]', '_', name)
        clean = clean.strip().rstrip('.')
        return clean or "Widmo"

    @classmethod
    def get_last_format(cls) -> str:
        settings = QSettings("SpectroHub", "Export")
        val = settings.value("last_format", "csv_pl")
        return val if val in EXPORT_FORMATS else "csv_pl"

    @classmethod
    def set_last_format(cls, fmt_id: str):
        if fmt_id in EXPORT_FORMATS:
            settings = QSettings("SpectroHub", "Export")
            settings.setValue("last_format", fmt_id)

    @classmethod
    def get_format_info(cls, fmt_id: str) -> Dict[str, Any]:
        return EXPORT_FORMATS.get(fmt_id, EXPORT_FORMATS["csv_pl"])

    @classmethod
    def get_all_formats(cls) -> List[Dict[str, Any]]:
        return list(EXPORT_FORMATS.values())

    @classmethod
    def export_single_spectrum(cls, spec: SpectrumData, file_path: str, fmt_id: str):
        """Eksportuje pojedyncze widmo do wybranego pliku."""
        cls.set_last_format(fmt_id)

        if fmt_id == "csv_pl":
            ASCIIExporter.export_spectrum_data(spec, file_path, style="CSV_PL")
        elif fmt_id == "csv_eu":
            ASCIIExporter.export_spectrum_data(spec, file_path, style="CSV_EU")
        elif fmt_id == "dpt_pl":
            ASCIIExporter.export_spectrum_data(spec, file_path, style="DPT_PL")
        elif fmt_id == "dpt_eu":
            ASCIIExporter.export_spectrum_data(spec, file_path, style="DPT_EU")
        elif fmt_id == "spc":
            SPCExporter.export_spectrum(spec, file_path)
        elif fmt_id == "jcamp":
            JCAMPExporter.export_spectrum(spec, file_path)
        else:
            ASCIIExporter.export_spectrum_data(spec, file_path, style="CSV_EU")

    @classmethod
    def export_batch_to_directory(cls, targets: List[Tuple[SpectrumData, str]], dest_dir: str, fmt_id: str) -> int:
        """
        Zapisuje zbiór widm do katalogu na dysku, odtwarzając strukturę podfolderów.
        targets: lista krotek (obiekt_SpectrumData, względna_ścieżka_katalogu)
        """
        cls.set_last_format(fmt_id)
        fmt = cls.get_format_info(fmt_id)
        ext = fmt["ext"]

        count = 0
        for spec, rel_dir in targets:
            target_folder = os.path.join(dest_dir, rel_dir) if rel_dir else dest_dir
            os.makedirs(target_folder, exist_ok=True)

            base_name = cls.sanitize_filename(spec.name)
            target_path = os.path.join(target_folder, f"{base_name}{ext}")

            # Zabezpieczenie przed nadpisywaniem widm o tej samej nazwie w jednym oknie
            ctr = 1
            while os.path.exists(target_path):
                target_path = os.path.join(target_folder, f"{base_name}_{ctr}{ext}")
                ctr += 1

            cls.export_single_spectrum(spec, target_path, fmt_id)
            count += 1

        return count

    @classmethod
    def export_batch_to_zip(cls, targets: List[Tuple[SpectrumData, str]], zip_path: str, fmt_id: str) -> int:
        """
        Zapisuje zbiór widm do pojedynczego pliku ZIP z zachowaniem hierarchii katalogów wewnątrz archiwum.
        """
        cls.set_last_format(fmt_id)
        fmt = cls.get_format_info(fmt_id)
        ext = fmt["ext"]

        count = 0
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_f:
            with tempfile.TemporaryDirectory() as temp_dir:
                for spec, rel_dir in targets:
                    base_name = cls.sanitize_filename(spec.name)
                    temp_file = os.path.join(temp_dir, f"{base_name}_{count}{ext}")

                    cls.export_single_spectrum(spec, temp_file, fmt_id)

                    arc_name = os.path.join(rel_dir, f"{base_name}{ext}") if rel_dir else f"{base_name}{ext}"
                    # W archiwach ZIP separatorem ścieżek zawsze musi być '/'
                    arc_name = arc_name.replace('\\', '/')

                    zip_f.write(temp_file, arcname=arc_name)
                    count += 1

        return count