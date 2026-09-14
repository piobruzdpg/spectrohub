"""
gui/node_editor/registry.py
Centralny rejestr węzłów potoku DAG.
Obsługuje automatyczną rejestrację klas za pomocą dekoratora @register_node
oraz dynamiczne odkrywanie plików w katalogu nodes/.
"""

import os
import importlib
import logging
from typing import Dict, Type, List, Optional, Tuple, Any
from collections import defaultdict

from gui.node_editor.base import BaseNode

logger = logging.getLogger("SpectroHub")

# Główny rejestr: identyfikator typu -> klasa węzła
_NODE_CLASSES: Dict[str, Type[BaseNode]] = {}


def register_node(cls: Type[BaseNode]) -> Type[BaseNode]:
    """Dekorator rejestrujący klasę węzła w systemie."""
    if not cls.node_type:
        raise ValueError(f"Klasa {cls.__name__} nie posiada zdefiniowanego unikalnego 'node_type'!")

    if cls.node_type in _NODE_CLASSES:
        logger.warning(f"Nadpisywanie istniejącego węzła: '{cls.node_type}' przez {cls.__name__}")

    _NODE_CLASSES[cls.node_type] = cls
    return cls


def get_node_class(node_type: str) -> Optional[Type[BaseNode]]:
    """Zwraca zarejestrowaną klasę węzła dla danego typu."""
    _ensure_discovered()
    return _NODE_CLASSES.get(node_type)


def get_all_node_classes() -> Dict[str, Type[BaseNode]]:
    """Zwraca wszystkie zarejestrowane klasy węzłów."""
    _ensure_discovered()
    return dict(_NODE_CLASSES)


def get_categories_dict() -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Buduje skategoryzowany słownik dla lewego doku (biblioteki węzłów).
    Format kompatybilny z NodeLibraryWidget:
    { "Kategoria": [("Tytuł węzła", step_data_dict), ...] }
    """
    _ensure_discovered()
    cat_map = defaultdict(list)

    for node_type, cls in _NODE_CLASSES.items():
        step_data = cls.default_step_data()
        cat_map[cls.category].append((cls.title, step_data))

    return dict(cat_map)


_DISCOVERED = False


def _ensure_discovered():
    """Zapewnia jednokrotne załadowanie wszystkich modułów z katalogu nodes/."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
    if not os.path.isdir(nodes_dir):
        return

    for filename in os.listdir(nodes_dir):
        if filename.startswith("node_") and filename.endswith(".py"):
            mod_name = f"gui.node_editor.nodes.{filename[:-3]}"
            try:
                importlib.import_module(mod_name)
                logger.debug(f"Pomyślnie załadowano moduł węzłów: {mod_name}")
            except Exception as e:
                logger.error(f"Błąd podczas importu węzła {mod_name}: {e}", exc_info=True)