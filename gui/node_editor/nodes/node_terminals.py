"""
gui/node_editor/nodes/node_terminals.py
Węzły końcowe potoku z autonomicznymi dialogami: Output, OutputWindow, OutputTable.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QLineEdit, QLabel
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from core.session import session


class SpectrumOutputDialog(QDialog):
    def __init__(self, node: BaseNode, is_window: bool = False, parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.is_window = is_window
        self.setWindowTitle(f"Parametry zapisu: {node.title}")
        self.resize(460, 320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        if self.is_window:
            self.ent_win_name = QLineEdit(self.step.get('window_name', 'Skorygowane'))
            form.addRow("Nazwa docelowego okna:", self.ent_win_name)

            self.combo_mode = QComboBox()
            self.combo_mode.addItem("Utwórz nowe okno robocze", "create_new")
            self.combo_mode.addItem("Dodaj do istniejącego (o tej nazwie)", "append_existing")
            idx = self.combo_mode.findData(self.step.get('mode', 'create_new'))
            if idx >= 0:
                self.combo_mode.setCurrentIndex(idx)
            form.addRow("Tryb okna:", self.combo_mode)

        self.combo_naming_mode = QComboBox()
        self.combo_naming_mode.addItem("Prefiks / Sufiks do nazwy pierwotnej", "source_affix")
        self.combo_naming_mode.addItem("Własna nazwa bazowa", "custom_base")
        self.combo_naming_mode.addItem("Nazwa operacji / węzła", "node_title")

        cur_n_mode = self.step.get('naming_mode', 'source_affix')
        idx_n = self.combo_naming_mode.findData(cur_n_mode)
        if idx_n >= 0:
            self.combo_naming_mode.setCurrentIndex(idx_n)
        form.addRow("Format nazewnictwa widm:", self.combo_naming_mode)

        self.ent_prefix = QLineEdit(self.step.get('name_prefix', 'proc_'))
        self.ent_prefix.setPlaceholderText("np. proc_, corr_")
        form.addRow("Prefiks:", self.ent_prefix)

        self.ent_suffix = QLineEdit(self.step.get('name_suffix', ''))
        self.ent_suffix.setPlaceholderText("np. _asls, _sub (opcjonalnie)")
        form.addRow("Sufiks:", self.ent_suffix)

        self.ent_custom_base = QLineEdit(self.step.get('custom_name', 'Widmo'))
        form.addRow("Własna nazwa bazowa:", self.ent_custom_base)

        self.lbl_name_preview = QLabel()
        self.lbl_name_preview.setStyleSheet("color: #48cae4; font-weight: bold; font-size: 11px;")
        form.addRow("Podgląd formatu:", self.lbl_name_preview)

        def _update_preview():
            mode = self.combo_naming_mode.currentData()
            is_affix = (mode == "source_affix")
            is_custom = (mode == "custom_base")
            self.ent_prefix.setEnabled(is_affix or is_custom)
            self.ent_suffix.setEnabled(is_affix or is_custom)
            self.ent_custom_base.setEnabled(is_custom)

            pfx = self.ent_prefix.text()
            sfx = self.ent_suffix.text()
            c_base = self.ent_custom_base.text().strip() or "Widmo"

            if mode == "source_affix":
                self.lbl_name_preview.setText(f"próbka_01 ➔ {pfx}próbka_01{sfx}")
            elif mode == "custom_base":
                self.lbl_name_preview.setText(f"próbka_01 ➔ {pfx}{c_base}_1{sfx}")
            else:
                self.lbl_name_preview.setText("próbka_01 ➔ [Nazwa węzła]")

        self.combo_naming_mode.currentIndexChanged.connect(_update_preview)
        self.ent_prefix.textChanged.connect(_update_preview)
        self.ent_suffix.textChanged.connect(_update_preview)
        self.ent_custom_base.textChanged.connect(_update_preview)
        _update_preview()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        self.step['naming_mode'] = self.combo_naming_mode.currentData()
        self.step['name_prefix'] = self.ent_prefix.text()
        self.step['name_suffix'] = self.ent_suffix.text()
        self.step['custom_name'] = self.ent_custom_base.text().strip()

        if self.is_window:
            wname = self.ent_win_name.text().strip() or "Skorygowane"
            self.step['window_name'] = wname
            self.step['mode'] = self.combo_mode.currentData()
            self.step['name'] = f"🗂️ Okno: {wname}"
        else:
            self.step['name'] = "💾 Zapis Wyniku"
        self.accept()


class OutputTableDialog(QDialog):
    def __init__(self, node: 'OutputTableNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Konfiguracja tabeli wyjściowej")
        self.resize(440, 240)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.ent_tbl_name = QLineEdit(self.step.get('table_name', 'Profil'))
        form.addRow("Nazwa tabeli:", self.ent_tbl_name)

        self.combo_folder = QComboBox()
        self.combo_folder.addItem("📁 (Poziom główny / bez folderu)", None)
        for fid, f in session.folders.items():
            self.combo_folder.addItem(f"📁 {f.name}", fid)

        cur_fid = self.step.get('folder_id')
        idx_f = self.combo_folder.findData(cur_fid)
        if idx_f >= 0:
            self.combo_folder.setCurrentIndex(idx_f)
        form.addRow("Docelowy folder:", self.combo_folder)

        self.ent_x_lbl = QLineEdit(self.step.get('x_label', 'Czas / Indeks'))
        form.addRow("Etykieta osi X:", self.ent_x_lbl)

        self.ent_y_lbl = QLineEdit(self.step.get('y_label', 'Wartość / Stężenie'))
        form.addRow("Etykieta osi Y:", self.ent_y_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        tname = self.ent_tbl_name.text().strip() or "Profil"
        self.step['table_name'] = tname
        self.step['folder_id'] = self.combo_folder.currentData()
        self.step['x_label'] = self.ent_x_lbl.text().strip()
        self.step['y_label'] = self.ent_y_lbl.text().strip()
        self.step['name'] = f"📊 Tabela: {tname}"
        self.accept()


@register_node
class OutputNode(BaseNode):
    node_type = "output"
    title = "💾 Zapis Wyniku"
    category = "Węzły Końcowe (Terminale)"
    header_gradient = ("#6a040f", "#9d0208")
    preview_port_id = "in"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Zapis standardowy",
            "naming_mode": "source_affix",
            "name_prefix": "proc_",
            "name_suffix": "",
            "custom_name": "Widmo"
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return SpectrumOutputDialog(self, is_window=False, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        return {"in": data_in}, {"in": names_in}


@register_node
class OutputWindowNode(BaseNode):
    node_type = "output_window"
    title = "🗂️ Kopiuj do okna roboczego"
    category = "Węzły Końcowe (Terminale)"
    header_gradient = ("#7209b7", "#b5179e")
    preview_port_id = "in"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "🗂️ Okno: Skorygowane",
            "window_name": "Skorygowane",
            "mode": "create_new",
            "naming_mode": "source_affix",
            "name_prefix": "proc_",
            "name_suffix": "",
            "custom_name": "Widmo"
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), allow_multiple=True, pos_y=90),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#b5179e"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return SpectrumOutputDialog(self, is_window=True, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sources = inputs.get("in_multi", [])
        source_names = input_names.get("in_multi", [])

        if not sources:
            d_in = inputs.get("in", (None, None))
            if d_in[0] is not None:
                sources = [d_in]
                source_names = [input_names.get("in", [])]

        if not sources or sources[0][0] is None:
            return {"in": (None, None), "out": (None, None)}, {"in": [], "out": []}

        if len(sources) == 1:
            return {"in": sources[0], "out": sources[0]}, {"in": source_names[0], "out": source_names[0]}

        ref_x = sources[0][0]
        cols, all_incoming_names = [], []

        for idx, (sx, sy) in enumerate(sources):
            if sx is None or sy is None:
                continue
            names = source_names[idx] if idx < len(source_names) else []
            sx_s = sx[::-1] if sx[0] > sx[-1] else sx
            if sy.ndim == 2:
                for c in range(sy.shape[1]):
                    sy_col = sy[::-1, c] if sx[0] > sx[-1] else sy[:, c]
                    cols.append(np.interp(ref_x, sx_s, sy_col))
                    all_incoming_names.append(names[c] if c < len(names) else f"Widmo_{len(all_incoming_names) + 1}")
            else:
                sy_s = sy[::-1] if sx[0] > sx[-1] else sy
                cols.append(np.interp(ref_x, sx_s, sy_s))
                all_incoming_names.append(names[0] if names else f"Widmo_{len(all_incoming_names) + 1}")

        combined_y = np.column_stack(cols)
        return {"in": (ref_x.copy(), combined_y), "out": (ref_x.copy(), combined_y)}, {
            "in": all_incoming_names, "out": all_incoming_names
        }


@register_node
class OutputTableNode(BaseNode):
    node_type = "output_table"
    title = "📊 Zapisz jako Tabelę / Profil"
    category = "Węzły Końcowe (Terminale)"
    header_gradient = ("#0077b6", "#0096c7")
    preview_port_id = "in"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "📊 Tabela: Profil",
            "table_name": "Profil",
            "x_label": "Czas / Indeks",
            "y_label": "Wartość / Stężenie",
            "folder_id": None
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), allow_multiple=True, pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return OutputTableDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        return {"in": data_in}, {"in": names_in}


# gui/node_editor/nodes/node_terminals.py (od ok. linii 235)

class OutputFolderDialog(QDialog):
    def __init__(self, node: 'OutputFolderNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Zapis do Folderu Roboczego")
        self.resize(440, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.ent_folder_name = QLineEdit(self.step.get('folder_name', ''))
        self.ent_folder_name.setPlaceholderText("Domyślnie z solwera lub auto (np. Folder_1)")
        form.addRow("Własna nazwa folderu:", self.ent_folder_name)

        self.combo_parent = QComboBox()
        self.combo_parent.addItem("📁 (Poziom główny / bez folderu nadrzędnego)", None)
        for fid, f in session.folders.items():
            self.combo_parent.addItem(f"📁 {f.name}", fid)

        cur_fid = self.step.get('parent_folder_id')
        idx = self.combo_parent.findData(cur_fid)
        if idx >= 0:
            self.combo_parent.setCurrentIndex(idx)
        form.addRow("Folder nadrzędny:", self.combo_parent)

        lbl = QLabel(
            "Węzeł ten tworzy dedykowany folder w drzewie projektu.\n"
            "• Jeśli podłączony jest solwer HDO (lub skalar po HDO), zapisze komplet widm i tabelę.\n"
            "• W pozostałych przypadkach utworzy folder i zapisze wszystkie przekazane widma."
        )
        lbl.setStyleSheet("color: #a8dadc; font-size: 11px; padding: 6px;")
        form.addRow(lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        self.step['folder_name'] = self.ent_folder_name.text().strip()
        self.step['parent_folder_id'] = self.combo_parent.currentData()
        self.accept()


@register_node
class OutputFolderNode(BaseNode):
    node_type = "output_folder"
    title = "📁 Zapis do Folderu"
    category = "Węzły Końcowe (Terminale)"
    header_gradient = ("#1d3557", "#457b9d")
    preview_port_id = "in"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "📁 Zapis do Folderu",
            "folder_name": "",
            "parent_folder_id": None
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), allow_multiple=True, pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return OutputFolderDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        return {"in": data_in}, {"in": names_in}