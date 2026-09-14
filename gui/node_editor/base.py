"""
gui/node_editor/base.py
Klasy bazowe dla modułowego silnika węzłów potoku DAG.
Każdy węzeł w systemie dziedziczy po BaseNode i definiuje swoje porty,
logikę obliczeniową, parametry oraz opcjonalny formularz konfiguracji.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QColor, QPainter
from PySide6.QtCore import QRectF


@dataclass
class PortDef:
    """Definicja pojedynczego portu wejściowego lub wyjściowego węzła."""
    port_id: str
    label: str
    is_output: bool
    color: QColor
    allow_multiple: bool = False
    pos_y: Optional[float] = None  # Jeśli None, NodeItem rozmieści porty automatycznie


class BaseNode:
    """
    Abstrakcyjna klasa bazowa węzła potoku.

    Każda podklasa musi zdefiniować:
    - node_type: unikalny identyfikator tekstowy (np. 'asls', 'math_op')
    - title: czytelna nazwa wyświetlana na belce tytułowej
    - category: kategoria w drzewie biblioteki
    """
    node_type: str = ""
    title: str = ""
    category: str = "Inne"
    header_gradient: Tuple[str, str] = ("#1e3d59", "#17b978")

    # Domyślne wymiary kafelka
    width: int = 220
    height: int = 150

    def __init__(self, step_data: Optional[Dict[str, Any]] = None):
        self.step_data: Dict[str, Any] = step_data or self.default_step_data()

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        """Zwraca domyślny słownik parametrów dla nowo tworzonego węzła."""
        return {
            "type": cls.node_type,
            "name": cls.title
        }

    def get_ports(self) -> List[PortDef]:
        """
        Zwraca listę definicji portów (wejść i wyjść).
        Domyślnie: 1 wejście 'in' i 1 wyjście 'out'.
        """
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703")),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#48cae4"))
        ]

    def compute(
            self,
            inputs: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
            input_names: Dict[str, List[str]],
            is_execution: bool = False,
            progress_callback: Optional[Any] = None
    ) -> Tuple[
        Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        Dict[str, List[str]]
    ]:
        """
        Główna metoda obliczeniowa węzła.

        Parametry:
            inputs: {port_id: (x_array, y_array)}
            input_names: {port_id: [nazwa_1, nazwa_2, ...]}
            is_execution: True jeśli kliknięto 'Uruchom Potok', False dla szybkiego podglądu (preview)
            progress_callback: opcjonalny callback do raportowania postępu

        Zwraca:
            (port_results, port_names):
                port_results: {port_id: (x_array, y_array)}
                port_names:   {port_id: [nazwa_1, nazwa_2, ...]}
        """
        raise NotImplementedError(f"Węzeł '{self.node_type}' musi implementować metodę compute().")

    def create_config_dialog(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        """
        Zwraca dedykowany dialog konfiguracji parametrów węzła (QDialog).
        Jeśli zwróci None, edytor użyje standardowego okna lub StepParamDialog.
        """
        return None

    def draw_custom_overlay(self, painter: QPainter, box_rect: QRectF):
        """
        Opcjonalny hook do rysowania niestandardowych elementów na kafelku
        (np. mini-raport HDO, liczba iteracji MC, itp.).
        """
        pass