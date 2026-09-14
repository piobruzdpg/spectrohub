"""
gui/node_editor/nodes/node_chemometrics.py
Węzły chemometrii (SVD/PCA, MCR-ALS, Skalowania) z dedykowanymi dialogami.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QSpinBox, QCheckBox, QComboBox
)

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from algorithms.chemometrics import multivariate as chem
from algorithms import preprocessing as prep
from core.session import session


class SvdDenoiseDialog(QDialog):
    def __init__(self, node: 'SvdDenoiseNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Odszumianie SVD / PCA")
        self.resize(380, 160)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 40)
        self.spin_k.setValue(int(self.step.get('k', 2)))
        form.addRow("Liczba składowych (Rank k):", self.spin_k)

        self.chk_center = QCheckBox("Centruj do średniej (Klasyczne PCA)")
        self.chk_center.setChecked(bool(self.step.get('center', False)))
        form.addRow("Tryb dekompozycji:", self.chk_center)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        k = int(self.spin_k.value())
        center = bool(self.chk_center.isChecked())
        self.step['k'] = k
        self.step['center'] = center
        tag = "PCA" if center else "SVD"
        self.step['name'] = f"{tag} Denoise (k={k})"
        self.accept()


class McrDecomposeDialog(QDialog):
    def __init__(self, node: 'McrDecomposeNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Dekompozycja MCR-ALS / SPEXFA")
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_mcr_method = QComboBox()
        self.combo_mcr_method.addItem("MCR-ALS (Optymalizacja naprzemienna)", "mcr_als")
        self.combo_mcr_method.addItem("SPEXFA (Ekstrakcja kluczowa IKSFA)", "spexfa")
        idx = self.combo_mcr_method.findData(self.step.get('method', 'mcr_als'))
        if idx >= 0:
            self.combo_mcr_method.setCurrentIndex(idx)
        form.addRow("Algorytm dekompozycji:", self.combo_mcr_method)

        self.spin_mcr_k = QSpinBox()
        self.spin_mcr_k.setRange(1, 20)
        self.spin_mcr_k.setValue(int(self.step.get('k', 2)))
        form.addRow("Liczba czystych składników (k):", self.spin_mcr_k)

        self.chk_nnls = QCheckBox("Wymuś nieujemność widm i stężeń (NNLS)")
        self.chk_nnls.setChecked(bool(self.step.get('non_negative', True)))
        form.addRow("Więzy nieujemności:", self.chk_nnls)

        self.chk_closure = QCheckBox("Domknięcie bilansu (Suma stężeń = 1.0)")
        self.chk_closure.setChecked(bool(self.step.get('closure', False)))
        form.addRow("Więzy stężeń:", self.chk_closure)

        self.chk_unimodal = QCheckBox("Jednomodalność profili (Unimodal)")
        self.chk_unimodal.setChecked(bool(self.step.get('unimodal', False)))
        form.addRow("Kształt profilu:", self.chk_unimodal)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(20, 2000)
        self.spin_max_iter.setValue(int(self.step.get('max_iter', 500)))
        form.addRow("Maksymalna liczba iteracji:", self.spin_max_iter)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        method = self.combo_mcr_method.currentData()
        k = int(self.spin_mcr_k.value())
        self.step['method'] = method
        self.step['k'] = k
        self.step['non_negative'] = bool(self.chk_nnls.isChecked())
        self.step['closure'] = bool(self.chk_closure.isChecked())
        self.step['unimodal'] = bool(self.chk_unimodal.isChecked())
        self.step['max_iter'] = int(self.spin_max_iter.value())
        m_label = "MCR-ALS" if method == "mcr_als" else "SPEXFA"
        self.step['name'] = f"{m_label} (k={k})"
        self.accept()


class MatrixScalingDialog(QDialog):
    def __init__(self, node: 'MatrixScalingNode', parent=None):
        super().__init__(parent)
        self.node = node
        self.step = node.step_data
        self.setWindowTitle("Transformacja macierzowa")
        self.resize(380, 140)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Centrowanie do średniej (Mean-center)", "mean_center")
        self.combo_scale.addItem("Autoskalowanie (Unit Variance)", "autoscaling")
        self.combo_scale.addItem("Skalowanie Pareto", "pareto")
        self.combo_scale.addItem("Korekcja rozproszenia (MSC)", "msc")
        idx = self.combo_scale.findData(self.step.get('scale_type', 'mean_center'))
        if idx >= 0:
            self.combo_scale.setCurrentIndex(idx)
        form.addRow("Transformacja:", self.combo_scale)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        st = self.combo_scale.currentData()
        self.step['scale_type'] = st
        names = {'mean_center': "Mean-center", 'autoscaling': "Autoscaling (UV)", 'pareto': "Skalowanie Pareto", 'msc': "Korekcja MSC"}
        self.step['name'] = f"Skalowanie: {names.get(st, st)}"
        self.accept()


@register_node
class SvdDenoiseNode(BaseNode):
    node_type = "svd_denoise"
    title = "📊 Odszumiacz SVD / PCA (Rank-k)"
    category = "Chemometria & Analiza Wielowymiarowa"
    header_gradient = ("#240046", "#5a189a")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "SVD Denoise (k=2)", "k": 2, "center": False}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="D_k", is_output=True, color=QColor("#48cae4"), pos_y=56),
            PortDef(port_id="out_res", label="Res", is_output=True, color=QColor("#e63946"), pos_y=88),
            PortDef(port_id="out_scores", label="Scr", is_output=True, color=QColor("#52b788"), pos_y=120)
        ]

    def create_config_dialog(self, parent=None):
        return SvdDenoiseDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        in_names = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None), "out_res": (None, None), "out_scores": (None, None)}, {
                "out": [], "out_res": [], "out_scores": []
            }

        curr_x, curr_y = data_in
        k = int(self.step_data.get('k', 2))
        center = bool(self.step_data.get('center', False))

        if curr_y.ndim == 2 and curr_y.shape[1] >= 2:
            k_eff = min(k, curr_y.shape[0], curr_y.shape[1])
            res = chem.run_pca(curr_y, n_components=k_eff, center=center)
            scores = res['scores']
            x_samples = np.arange(1, scores.shape[0] + 1, dtype=np.float64)

            return {
                "out": (curr_x.copy(), res['dr']),
                "out_res": (curr_x.copy(), res['residuals']),
                "out_scores": (x_samples, scores)
            }, {
                "out": in_names,
                "out_res": in_names,
                "out_scores": [f"Score_PC{i + 1}" for i in range(scores.shape[1])]
            }
        else:
            ref_mat = None
            if session.active_series:
                ref_mat = session.active_series.get_matrix_2d()
            else:
                vis = session.get_pipeline_target_spectra()
                if len(vis) >= 2:
                    ref_mat = np.column_stack([
                        np.interp(curr_x, s.wavenumbers[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.wavenumbers,
                                  s.intensities[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.intensities)
                        for s in vis
                    ])

            if ref_mat is not None and ref_mat.shape[1] >= 2:
                k_eff = min(k, ref_mat.shape[0], ref_mat.shape[1])
                res = chem.run_pca(ref_mat, n_components=k_eff, center=center)
                U_k = res['loadings']
                y_recon = (res['mean_spectrum'] + (U_k @ (U_k.T @ (curr_y - res['mean_spectrum'])))) if center else (U_k @ (U_k.T @ curr_y))
                return {
                    "out": (curr_x.copy(), y_recon),
                    "out_res": (curr_x.copy(), curr_y - y_recon),
                    "out_scores": (np.array([1.0]), np.array([1.0]))
                }, {"out": in_names, "out_res": in_names, "out_scores": ["Score_PC1"]}

            return {
                "out": (curr_x.copy(), curr_y.copy()),
                "out_res": (curr_x.copy(), np.zeros_like(curr_y)),
                "out_scores": (np.array([1.0]), np.array([1.0]))
            }, {"out": in_names, "out_res": in_names, "out_scores": ["Score_PC1"]}


@register_node
class McrDecomposeNode(BaseNode):
    node_type = "mcr_decompose"
    title = "🔬 Dekompozycja MCR-ALS / SPEXFA"
    category = "Chemometria & Analiza Wielowymiarowa"
    header_gradient = ("#9d0208", "#d90429")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type, "name": "MCR-ALS (k=2)", "method": "mcr_als",
            "k": 2, "non_negative": True, "closure": False, "unimodal": False, "max_iter": 500
        }

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="S^T", is_output=True, color=QColor("#48cae4"), pos_y=44),
            PortDef(port_id="out_conc", label="C", is_output=True, color=QColor("#52b788"), pos_y=68),
            PortDef(port_id="out_recon", label="Rec", is_output=True, color=QColor("#f4a261"), pos_y=94),
            PortDef(port_id="out_res", label="Res", is_output=True, color=QColor("#e63946"), pos_y=120)
        ]

    def create_config_dialog(self, parent=None):
        return McrDecomposeDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        in_names = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None), "out_conc": (None, None), "out_recon": (None, None), "out_res": (None, None)}, {
                "out": [], "out_conc": [], "out_recon": [], "out_res": []
            }

        curr_x, curr_y = data_in
        k = int(self.step_data.get('k', 2))
        method = self.step_data.get('method', 'mcr_als')

        target_mat = curr_y if (curr_y.ndim == 2 and curr_y.shape[1] >= 2) else (
            session.active_series.get_matrix_2d() if session.active_series else None
        )

        if target_mat is None:
            vis = session.get_pipeline_target_spectra()
            if len(vis) >= 2:
                target_mat = np.column_stack([
                    np.interp(curr_x, s.wavenumbers[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.wavenumbers,
                              s.intensities[::-1] if s.wavenumbers[0] > s.wavenumbers[-1] else s.intensities)
                    for s in vis
                ])

        if target_mat is not None and target_mat.shape[1] >= 2:
            k_eff = min(k, target_mat.shape[0], target_mat.shape[1])
            try:
                if method == 'spexfa':
                    res = chem.run_spexfa(target_mat, n=k_eff)
                else:
                    res = chem.run_mcr_als(
                        target_mat, rank=k_eff,
                        non_negative=self.step_data.get('non_negative', True),
                        closure=self.step_data.get('closure', False),
                        unimodal=self.step_data.get('unimodal', False),
                        max_iter=self.step_data.get('max_iter', 500)
                    )
                pure_spec = res['pure_spectra']
                pure_conc = res['pure_concentrations']
                dr = res['dr']
                residuals = res['residuals']
                x_samples = np.arange(1, pure_conc.shape[0] + 1, dtype=np.float64)

                return {
                    "out": (curr_x.copy(), pure_spec),
                    "out_conc": (x_samples, pure_conc),
                    "out_recon": (curr_x.copy(), dr),
                    "out_res": (curr_x.copy(), residuals)
                }, {
                    "out": [f"MCR_Składnik_{i + 1}" for i in range(k_eff)],
                    "out_conc": [f"Stężenie_C{i + 1}" for i in range(k_eff)],
                    "out_recon": in_names,
                    "out_res": in_names
                }
            except Exception:
                pass

        return {
            "out": (curr_x.copy(), curr_y.copy()),
            "out_conc": (np.array([1.0]), np.array([1.0])),
            "out_recon": (curr_x.copy(), curr_y.copy()),
            "out_res": (curr_x.copy(), np.zeros_like(curr_y))
        }, {"out": in_names, "out_conc": ["C1"], "out_recon": in_names, "out_res": in_names}


@register_node
class MatrixScalingNode(BaseNode):
    node_type = "matrix_scaling"
    title = "📐 Skalowania Populacyjne"
    category = "Chemometria & Analiza Wielowymiarowa"
    header_gradient = ("#0077b6", "#0096c7")

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Skalowanie: Mean-center", "scale_type": "mean_center"}

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#48cae4"), pos_y=90)
        ]

    def create_config_dialog(self, parent=None):
        return MatrixScalingDialog(self, parent=parent)

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None)}, {"out": []}

        curr_x, curr_y = data_in
        st = self.step_data.get('scale_type', 'mean_center')

        if curr_y.ndim == 2 and curr_y.shape[1] > 1:
            if st == 'mean_center': scaled_y, _ = prep.apply_mean_center(curr_y)
            elif st == 'autoscaling': scaled_y = prep.apply_autoscaling(curr_y)
            elif st == 'pareto': scaled_y = prep.apply_pareto_scaling(curr_y)
            elif st == 'msc': scaled_y = prep.apply_msc(curr_y)
            else: scaled_y = curr_y
            return {"out": (curr_x.copy(), scaled_y)}, {"out": list(names_in)}

        return {"out": (curr_x.copy(), curr_y.copy())}, {"out": list(names_in)}