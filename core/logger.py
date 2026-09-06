"""
Moduł zaawansowanego logowania diagnostycznego dla SpectroHub.
Zapewnia automatyczny zrzut środowiska oraz formatowanie błędów pod kątem analizy deweloperskiej/LLM.
"""

import os
import sys
import logging
import platform
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class LLMDiagnosticFormatter(logging.Formatter):
    """Kompaktowy format ułatwiający czytanie przez człowieka i analizę przez model AI."""

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname.ljust(8)
        module = f"{record.module}:{record.lineno}".ljust(22)
        message = record.getMessage()

        base_log = f"[{timestamp}] | {level} | {module} | {message}"
        if record.exc_info:
            base_log += f"\n[TRACEBACK START]\n{self.formatException(record.exc_info)}\n[TRACEBACK END]"
        return base_log


def setup_logger(log_file: str = "spectrohub_debug.log", log_level: int = logging.DEBUG) -> logging.Logger:
    """Inicjalizuje globalny rejestrator zdarzeń z wyjściem do konsoli i rotującego pliku."""
    logger = logging.getLogger("SpectroHub")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    # 1. Strumień konsoli (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(LLMDiagnosticFormatter())
    logger.addHandler(console_handler)

    # 2. Rotujący plik logowania (max 10 MB na plik, 3 kopie zapasowe)
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LLMDiagnosticFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Nie udało się zainicjalizować pliku logu: {e}", file=sys.stderr)

    _log_environment_snapshot(logger)
    return logger


def _log_environment_snapshot(logger: logging.Logger):
    """Generuje pełny raport środowiskowy na początku sesji."""

    def get_version(pkg_name: str) -> str:
        try:
            return __import__(pkg_name).__version__
        except Exception:
            return "N/A"

    ram_info = "N/A"
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        ram_info = f"{mem.total / (1024 ** 3):.1f} GB (Dostępne: {mem.available / (1024 ** 3):.1f} GB)"

    cpu_count = os.cpu_count() or 1
    snapshot = (
            "\n" + "=" * 80 + "\n"
                              f"=== SPECTROHUB ENVIRONMENT SNAPSHOT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                              f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
                              f"Python: {platform.python_version()} ({platform.python_implementation()})\n"
                              f"Qt / PySide6: {get_version('PySide6')} | Matplotlib: {get_version('matplotlib')}\n"
                              f"NumPy: {get_version('numpy')} | SciPy: {get_version('scipy')} | Pandas: {get_version('pandas')}\n"
                              f"CPU Cores: {cpu_count} (Rekomendowana pula workerów: {max(1, cpu_count - 1)})\n"
                              f"RAM: {ram_info}\n"
                              f"Katalog roboczy: {os.getcwd()}\n"
            + "=" * 80
    )
    logger.info(snapshot)


def log_algo_exception(logger: logging.Logger, operation_name: str, data=None, params: dict = None):
    """
    Formatuje błąd numeryczny wraz ze strukturą danych wejściowych,
    umożliwiając natychmiastowe zdiagnozowanie problemu w kodzie.
    """
    shape_info = getattr(data, 'shape', 'Brak atrybutu shape')
    dtype_info = getattr(data, 'dtype', type(data))

    # Sprawdzenie obecności NaN / Inf dla tablic numpy
    nan_info = "Nie dotyczy"
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            nan_info = f"NaNs: {nan_count}, Infs: {inf_count}"
    except Exception:
        pass

    msg = (
        f"[ALGO_FAILURE] Wyjątek podczas wykonywania: '{operation_name}'\n"
        f"  ├─ Typ danych: {dtype_info}\n"
        f"  ├─ Kształt (Shape): {shape_info}\n"
        f"  ├─ Spójność macierzy: {nan_info}\n"
        f"  ├─ Parametry wejściowe: {params or {} }\n"
        f"  └─ Traceback:\n{traceback.format_exc().strip()}"
    )
    logger.error(msg)