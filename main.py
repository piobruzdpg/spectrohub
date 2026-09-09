"""
main.py
Punkt startowy aplikacji SpectroHub.
"""

import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, qInstallMessageHandler, QtMsgType

from core.logger import setup_logger
from gui.main_window import MainWindow

# 1. Reguła z gwiazdką dla podkategorii dostępności
os.environ["QT_LOGGING_RULES"] = "qt.qpa.accessibility*=false;qt.accessibility*=false"

def qt_message_filter(mode, context, message):
    """Przechwytuje i wycisza fałszywe alarmy Cocoa Accessibility na macOS."""
    if "Called attribute on invalid object" in message:
        return

    # Wszystkie pozostałe ostrzeżenia i błędy Qt wypisujemy standardowo
    if mode == QtMsgType.QtWarningMsg:
        sys.stderr.write(f"[Qt Warning] {message}\n")
    elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        sys.stderr.write(f"[Qt Error] {message}\n")


def main():
    # Podpięcie filtra komunikatów Qt
    qInstallMessageHandler(qt_message_filter)

    # 1. Inicjalizacja loggera środowiskowego
    logger = setup_logger()
    logger.info("Uruchamianie aplikacji SpectroHub...")

    # 2. Utworzenie instancji Qt
    app = QApplication(sys.argv)
    app.setApplicationName("SpectroHub")
    app.setOrganizationName("GUT_PhysicalChemistry")

    # 3. Obsługa SIGINT (Płynne zamykanie przez Ctrl+C / terminal)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(500)

    # 4. Główne okno
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()