"""
gui/node_editor/nodes/node_io.py
Węzły wejściowe i multiplexery z autonomicznymi dialogami: Input, SpectrumSelect, MergeSpectra.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from core.session import session


class InputNodeDialog(QDialog):
    def __init__(self, node: 'InputNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle(f"Parametry węzła: {node.title}")
        self.resize(460, 240)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_source_mode = QComboBox()
        self.combo_source_mode.addItem("Tylko aktywne widmo (1D)", "active")
        self.combo_source_mode.addItem("Konkretne widmo z projektu (1D)", "specific")
        self.combo_source_mode.addItem("Wszystkie widoczne w oknie / gałęzi (Batch 2D)", "visible")
        self.combo_source_mode.addItem("Aktywna seria kinetyczna / Tensor", "series")

        cur_mode = self.step.get('source_mode', 'active')
        idx_m = self.combo_source_mode.findData(cur_mode)
        if idx_m >= 0:
            self.combo_source_mode.setCurrentIndex(idx_m)
        form.addRow("Zakres danych wejściowych:", self.combo_source_mode)

        self.combo_specific_spec = QComboBox()
        for name in session.spectra.keys():
            self.combo_specific_spec.addItem(f"📈 {name}", name)
        cur_s = self.step.get('specific_spectrum', '')
        idx_s = self.combo_specific_spec.findData(cur_s)
        if idx_s >= 0:
            self.combo_specific_spec.setCurrentIndex(idx_s)
        form.addRow("Wybrane widmo:", self.combo_specific_spec)

        def _toggle_specific(idx):
            mode = self.combo_source_mode.itemData(idx)
            self.combo_specific_spec.setEnabled(mode == "specific")

        self.combo_source_mode.currentIndexChanged.connect(_toggle_specific)
        _toggle_specific(self.combo_source_mode.currentIndex())

        lbl_hint = QLabel(
            "💡 Wybierz 'Tylko aktywne' lub 'Konkretne widmo' do pojedynczych analiz.\n"
            "Wybierz 'Wszystkie widoczne' do hurtowego przetwarzania wsadowego (Batch)."
        )
        lbl_hint.setStyleSheet("color: #a8dadc; font-size: 11px;")
        form.addRow(lbl_hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        mode = self.combo_source_mode.currentData()
        self.step['source_mode'] = mode
        if mode == 'specific':
            spec_name = self.combo_specific_spec.currentData() or ""
            self.step['specific_spectrum'] = spec_name
            short_n = spec_name[:16] + "…" if len(spec_name) > 16 else spec_name
            self.step['name'] = f"Źródło: {short_n}"
        elif mode == 'visible':
            self.step['name'] = "Źródło: Wszystkie widoczne"
        elif mode == 'series':
            self.step['name'] = "Źródło: Aktywna seria"
        else:
            self.step['name'] = "Źródło: Aktywne widmo"
        self.accept()


class SpectrumSelectNodeDialog(QDialog):
    def __init__(self, node: 'SpectrumSelectNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Wybór widm z projektu")
        self.resize(460, 320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.list_specs = QListWidget()
        self.list_specs.setFixedHeight(180)

        current_selected = self.step.get('selected_spectra', [])
        if not current_selected and self.step.get('selected_spectrum'):
            current_selected = [self.step.get('selected_spectrum')]

        if session.atm_spectra:
            sep_atm = QListWidgetItem("── WZORCE ATMOSFERY (ATM / CO₂) ──")
            sep_atm.setFlags(Qt.NoItemFlags)
            self.list_specs.addItem(sep_atm)
            for name in session.atm_spectra.keys():
                item = QListWidgetItem(f"☁️ {name}")
                item.setData(Qt.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if name in current_selected else Qt.Unchecked)
                self.list_specs.addItem(item)

        if session.spectra:
            sep_exp = QListWidgetItem("── POMIARY EKSPERYMENTALNE ──")
            sep_exp.setFlags(Qt.NoItemFlags)
            self.list_specs.addItem(sep_exp)
            for name in session.spectra.keys():
                item = QListWidgetItem(f"📈 {name}")
                item.setData(Qt.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if name in current_selected else Qt.Unchecked)
                self.list_specs.addItem(item)

        row_btns = QHBoxLayout()
        btn_sel_atm = QPushButton("Zaznacz ATM")
        btn_sel_atm.clicked.connect(self._select_all_atm)
        btn_clear = QPushButton("Odznacz wszystkie")
        btn_clear.clicked.connect(self._clear_all)
        row_btns.addWidget(btn_sel_atm)
        row_btns.addWidget(btn_clear)

        form.addRow("Wybierz widma:", self.list_specs)
        form.addRow("", row_btns)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _select_all_atm(self):
        for i in range(self.list_specs.count()):
            item = self.list_specs.item(i)
            name = item.data(Qt.UserRole)
            if name and name in session.atm_spectra:
                item.setCheckState(Qt.Checked)

    def _clear_all(self):
        for i in range(self.list_specs.count()):
            item = self.list_specs.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)

    def _save_and_accept(self):
        selected = []
        for i in range(self.list_specs.count()):
            item = self.list_specs.item(i)
            if (item.flags() & Qt.ItemIsUserCheckable) and item.checkState() == Qt.Checked:
                val = item.data(Qt.UserRole)
                if val:
                    selected.append(val)

        self.step['selected_spectra'] = selected
        if selected:
            self.step['selected_spectrum'] = selected[0]
            if len(selected) == 1:
                icon = "☁️" if selected[0] in session.atm_spectra else "📈"
                self.step['name'] = f"{icon} {selected[0]}"
            else:
                self.step['name'] = f"📌 {len(selected)} widm"
        else:
            self.step['selected_spectrum'] = ""
            self.step['name'] = "Ref: (brak)"
        self.accept()


@register_node
class InputNode(BaseNode):
    node_type = "input"
    title = "📥 ŹRÓDŁO DANYCH"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#1b4332", "#2d6a4f")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Źródło: Aktywne widmo",
            "source_mode": "active",
            "specific_spectrum": ""
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#52b788"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return InputNodeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        source_mode = self.step_data.get('source_mode', 'active')

        if source_mode == 'specific':
            spec_name = self.step_data.get('specific_spectrum', '')
            spec = session.get_spectrum(spec_name)
            if spec:
                return {"out": (spec.wavenumbers.copy(), spec.intensities.copy())}, {"out": [spec.name]}

        elif source_mode == 'visible':
            visible = session.get_pipeline_target_spectra()
            if len(visible) >= 2:
                ref_wn = visible[0].wavenumbers
                cols, names = [], []
                for s in visible:
                    xw, yw = (s.wavenumbers[::-1], s.intensities[::-1]) if s.wavenumbers[0] > s.wavenumbers[-1] else (s.wavenumbers, s.intensities)
                    cols.append(np.interp(ref_wn, xw, yw))
                    names.append(s.name)
                return {"out": (ref_wn.copy(), np.column_stack(cols))}, {"out": names}
            elif len(visible) == 1:
                return {"out": (visible[0].wavenumbers.copy(), visible[0].intensities.copy())}, {"out": [visible[0].name]}

        elif source_mode == 'series':
            if session.active_series:
                mat = session.active_series.get_matrix_2d()
                names = [f"{session.active_series.name}_f{i + 1}" for i in range(mat.shape[1])]
                return {"out": (session.active_series.wavenumbers.copy(), mat.copy())}, {"out": names}

        active = session.active_spectrum
        if active and len(active.wavenumbers) > 2:
            return {"out": (active.wavenumbers.copy(), active.intensities.copy())}, {"out": [active.name]}

        def_x = np.linspace(4000, 400, 1000)
        return {"out": (def_x, np.zeros_like(def_x))}, {"out": ["Brak danych"]}


@register_node
class SpectrumSelectNode(BaseNode):
    node_type = "spectrum_select"
    title = "📌 Widma z Projektu (Multi-Select)"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#0f4c5c", "#e36414")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Ref: (wybierz)",
            "selected_spectra": []
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#48cae4"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return SpectrumSelectNodeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        selected = self.step_data.get('selected_spectra', [])
        if not selected and self.step_data.get('selected_spectrum'):
            selected = [self.step_data['selected_spectrum']]

        valid_specs = [session.get_spectrum(name) for name in selected if session.get_spectrum(name)]

        if valid_specs:
            ref_x = valid_specs[0].wavenumbers.copy()
            cols, names = [], []
            for s in valid_specs:
                sx, sy = s.wavenumbers, s.intensities
                sx_s, sy_s = (sx[::-1], sy[::-1]) if sx[0] > sx[-1] else (sx, sy)
                cols.append(np.interp(ref_x, sx_s, sy_s))
                names.append(s.name)

            y_out = np.column_stack(cols) if len(cols) > 1 else cols[0]
            return {"out": (ref_x, y_out)}, {"out": names}

        def_x = np.linspace(4000, 400, 1000)
        return {"out": (def_x, np.zeros_like(def_x))}, {"out": ["Ref"]}


@register_node
class MergeSpectraNode(BaseNode):
    node_type = "merge_spectra"
    title = "🔀 Zbieracz / Mux (Merge 2D)"
    category = "Źródła Danych & Odniesienia"
    header_gradient = ("#028090", "#00a896")

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), allow_multiple=True, pos_y=90),
            PortDef(port_id="out", label="Mat", is_output=True, color=QColor("#06d6a0"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Zbieracz / Mux")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            "Węzeł Zbieracza (Mux) łączy sygnały z wielu gałęzi i nodów\n"
            "w jedną wspólną macierz dwuwymiarową (W x N).\n\n"
            "Podłącz kable z dowolnych węzłów pod port wejściowy 'In'."
        )
        lbl.setStyleSheet("color: #a8dadc; font-size: 12px; padding: 10px;")
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        return dlg

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        sources = inputs.get("in_multi", [])
        source_names = input_names.get("in_multi", [])

        if not sources:
            d_in = inputs.get("in", (None, None))
            if d_in[0] is not None:
                sources = [d_in]
                source_names = [input_names.get("in", [])]

        if not sources or sources[0][0] is None:
            return {"out": (None, None)}, {"out": []}

        ref_x = sources[0][0]
        cols, collected_names = [], []

        for idx, (sx, sy) in enumerate(sources):
            if sx is None or sy is None:
                continue
            names = source_names[idx] if idx < len(source_names) else []
            sx_s = sx[::-1] if sx[0] > sx[-1] else sx

            if sy.ndim == 2:
                for c in range(sy.shape[1]):
                    col_data = sy[::-1, c] if sx[0] > sx[-1] else sy[:, c]
                    cols.append(np.interp(ref_x, sx_s, col_data))
                    collected_names.append(names[c] if c < len(names) else f"Widmo_{len(collected_names) + 1}")
            else:
                col_data = sy[::-1] if sx[0] > sx[-1] else sy
                cols.append(np.interp(ref_x, sx_s, col_data))
                collected_names.append(names[0] if names else f"Widmo_{len(collected_names) + 1}")

        if cols:
            return {"out": (ref_x.copy(), np.column_stack(cols))}, {"out": collected_names}

        return {"out": (None, None)}, {"out": []}