"""
gui/project_metadata_widget.py
Panel edycji i podglądu metadanych projektu (umieszczony pod drzewem projektu).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QGroupBox, QFrame
)
from PySide6.QtCore import Qt

from core.session import session
from core.models import ProjectMetadata


class ProjectMetadataWidget(QWidget):
    """Kompaktowy panel zarządzania parametrami projektu badawczego."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_updating_ui = False
        self._init_ui()
        self._connect_signals()
        self.refresh_from_session()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        box = QGroupBox("📋 Metadane Projektu")
        box.setStyleSheet("QGroupBox { font-weight: bold; color: #a8dadc; }")
        l_box = QVBoxLayout(box)
        l_box.setContentsMargins(6, 8, 6, 6)
        l_box.setSpacing(6)

        # 1. Tytuł projektu
        row_title = QHBoxLayout()
        row_title.addWidget(QLabel("Tytuł:"))
        self.ent_title = QLineEdit()
        self.ent_title.setPlaceholderText("Nazwa serii / projektu...")
        row_title.addWidget(self.ent_title)
        l_box.addLayout(row_title)

        # 2. Osoba opracowująca (Autor)
        row_author = QHBoxLayout()
        row_author.addWidget(QLabel("Autor:"))
        self.ent_author = QLineEdit()
        self.ent_author.setPlaceholderText("Imię i nazwisko / ID...")
        row_author.addWidget(self.ent_author)
        l_box.addLayout(row_author)

        # 3. Wariant opracowania
        row_variant = QHBoxLayout()
        row_variant.addWidget(QLabel("Wariant:"))
        self.combo_variant = QComboBox()
        self.combo_variant.setEditable(True)
        self.combo_variant.addItems([
            "Standardowy",
            "Surowe pomiary",
            "Po kompensacji H₂O/CO₂",
            "Po korekcji linii bazowej",
            "Po dekonwolucji Amidu I",
            "Przygotowanie do publikacji",
            "Archiwalny"
        ])
        row_variant.addWidget(self.combo_variant, 1)
        l_box.addLayout(row_variant)

        # 4. Daty utworzenia / modyfikacji
        self.frame_dates = QFrame()
        self.frame_dates.setStyleSheet("background-color: #1a1a22; border-radius: 3px; padding: 4px;")
        l_dates = QVBoxLayout(self.frame_dates)
        l_dates.setContentsMargins(4, 4, 4, 4)
        l_dates.setSpacing(2)

        self.lbl_created = QLabel("Utworzono: —")
        self.lbl_created.setStyleSheet("color: #8d99ae; font-size: 10px;")
        self.lbl_modified = QLabel("Zmodyfikowano: —")
        self.lbl_modified.setStyleSheet("color: #8d99ae; font-size: 10px;")

        l_dates.addWidget(self.lbl_created)
        l_dates.addWidget(self.lbl_modified)
        l_box.addWidget(self.frame_dates)

        # 5. Opis i notatki eksperymentalne
        l_box.addWidget(QLabel("Opis / Notatki:"))
        self.txt_notes = QTextEdit()
        self.txt_notes.setPlaceholderText("Cel eksperymentu, parametry spektrometru, uwagi do serii...")
        self.txt_notes.setMaximumHeight(85)
        self.txt_notes.setStyleSheet("font-size: 11px;")
        l_box.addWidget(self.txt_notes)

        layout.addWidget(box)

    def _connect_signals(self):
        self.ent_title.editingFinished.connect(self._sync_to_session)
        self.ent_author.editingFinished.connect(self._sync_to_session)
        self.combo_variant.currentTextChanged.connect(self._sync_to_session)
        self.txt_notes.textChanged.connect(self._sync_to_session)

        session.metadata_changed.connect(self._on_metadata_changed)

    def _sync_to_session(self):
        if self._is_updating_ui:
            return

        session.metadata.title = self.ent_title.text().strip() or "Projekt SpectroHub"
        session.metadata.author = self.ent_author.text().strip()
        session.metadata.variant = self.combo_variant.currentText().strip()
        session.metadata.notes = self.txt_notes.toPlainText()
        session.metadata.modified_at = session.metadata.modified_at  # zachowujemy lub aktualizujemy przy zapisie

    def _on_metadata_changed(self, meta: ProjectMetadata):
        self.refresh_from_session()

    def refresh_from_session(self):
        self._is_updating_ui = True
        meta = session.metadata

        self.ent_title.setText(meta.title)
        self.ent_author.setText(meta.author)

        idx = self.combo_variant.findText(meta.variant)
        if idx >= 0:
            self.combo_variant.setCurrentIndex(idx)
        else:
            self.combo_variant.setEditText(meta.variant)

        self.lbl_created.setText(f"Utworzono: {meta.created_at or '—'}")
        self.lbl_modified.setText(f"Zmodyfikowano: {meta.modified_at or '—'}")
        self.txt_notes.setPlainText(meta.notes or meta.description)
        self._is_updating_ui = False