"""
gui/node_editor/palette.py
Paleta i drzewo biblioteki węzłów dla edytora DAG (obsługa Drag & Drop oraz wyszukiwania).
"""

import json
from typing import Dict, Any, Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QAbstractItemView
)
from PySide6.QtCore import Qt, QMimeData, QByteArray
from PySide6.QtGui import QDrag, QFont, QColor

MIME_NODE_TYPE = "application/x-spectrohub-node"


class NodeLibraryTree(QTreeWidget):
    """Drzewo węzłów z obsługą przeciągania na scenę grafu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setAnimated(True)
        self.setIndentation(16)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #1a1a24;
                border: 1px solid #2d2d3a;
                border-radius: 4px;
                color: #e0e0e0;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 4px 6px;
                border-radius: 3px;
            }
            QTreeWidget::item:hover {
                background-color: #2b2b3d;
            }
            QTreeWidget::item:selected {
                background-color: #3d5a80;
                color: #ffffff;
            }
        """)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item or not item.data(0, Qt.UserRole):
            return

        payload = item.data(0, Qt.UserRole)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(MIME_NODE_TYPE, QByteArray(json.dumps(payload).encode("utf-8")))
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class NodeLibraryWidget(QWidget):
    """Panel lewego doku: wyszukiwarka + skategoryzowane drzewo węzłów."""

    def __init__(self, categories_dict: Dict[str, list],
                 on_node_double_clicked: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 parent=None):
        super().__init__(parent)
        self.categories_dict = categories_dict
        self.on_node_double_clicked = on_node_double_clicked

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Wyszukiwarka na żywo
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Szukaj węzła (np. asls, svd, crop)...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._filter_tree)
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #22222f;
                border: 1px solid #3d3d4e;
                border-radius: 4px;
                padding: 5px 8px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #48cae4;
            }
        """)
        layout.addWidget(self.search_box)

        # Drzewo węzłów
        self.tree = NodeLibraryTree(self)
        self.tree.itemDoubleClicked.connect(self._handle_double_click)
        layout.addWidget(self.tree)

        self._build_tree()

    def _build_tree(self):
        self.tree.clear()
        for cat_name, items in self.categories_dict.items():
            cat_item = QTreeWidgetItem(self.tree)
            cat_item.setText(0, cat_name)
            font = QFont()
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setForeground(0, QColor("#a8dadc"))
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsDragEnabled)

            for label, step_data in items:
                child = QTreeWidgetItem(cat_item)
                child.setText(0, label)
                child.setData(0, Qt.UserRole, {"title": label, "step_data": step_data})
                child.setToolTip(0, f"Dodaj bloczek: {label}\nPrzeciągnij na siatkę grafu lub kliknij dwukrotnie.")

            cat_item.setExpanded(True)

    def _filter_tree(self, text: str):
        query = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            any_child_visible = False
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                matches = query in child.text(0).lower()
                child.setHidden(not matches)
                if matches:
                    any_child_visible = True
            cat_item.setHidden(not any_child_visible)
            if query and any_child_visible:
                cat_item.setExpanded(True)

    def _handle_double_click(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole)
        if data and self.on_node_double_clicked:
            self.on_node_double_clicked(data["title"], data["step_data"])