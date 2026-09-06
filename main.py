"""
main.py
Punkt startowy aplikacji SpectroHub.
"""

import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.logger import setup_logger
from gui.main_window import MainWindow


def main():
    # 1. Inicjalizacja loggera środowiskowego
    logger = setup_logger()
    logger.info("Uruchamianie aplikacji SpectroHub...")

    # 2. Utworzenie instancji Qt
    app = QApplication(sys.argv)
    app.setApplicationName("SpectroHub")
    app.setOrganizationName("GUT_PhysicalChemistry")

    # 3. Obsługa SIGINT (Płynne zamykanie przez Ctrl+C / PyCharm)
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