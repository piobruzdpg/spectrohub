"""
gui/node_editor/nodes/node_filters.py
Autonomiczne węzły filtrów spektralnych, pochodnych i normalizacji.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from algorithms import preprocessing as prep
from gui.workspaces.ws_preprocessing import StepParamDialog


class BaseFilterNode(BaseNode):
    """Baza dla węzłów transformacji 1-we/1-wy ze wsparciem 1D oraz macierzy 2D."""
    category = "Filtry i Pochodne"
    header_gradient = ("#1e3d59", "#17b978")

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Out", is_output=True, color=QColor("#00f5d4"), pos_y=90)
        ]

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute(self, inputs, input_names, is_execution=False, progress_callback=None):
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])
        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None)}, {"out": []}

        curr_x, curr_y = data_in
        nx, ny = self.apply_filter(curr_x, curr_y)
        return {"out": (nx, ny)}, {"out": list(names_in)}

    def create_config_dialog(self, parent=None):
        return StepParamDialog(self.step_data, parent=parent)


@register_node
class CropNode(BaseFilterNode):
    node_type = "crop"
    title = "Przycięcie (Crop)"
    category = "Przycinanie i Zakres"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Przycięcie [1000–1800]", "min_w": 1000.0, "max_w": 1800.0}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_w = float(self.step_data.get('min_w', 1000.0))
        max_w = float(self.step_data.get('max_w', 1800.0))
        low, high = min(min_w, max_w), max(min_w, max_w)
        mask = (x >= low) & (x <= high)
        if np.any(mask):
            return x[mask], y[mask, :] if y.ndim == 2 else y[mask]
        return x, y


@register_node
class SgSmoothNode(BaseFilterNode):
    node_type = "sg_smooth"
    title = "Wygładzanie SG"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "SG Wygładzanie (okno=11, st=3)", "win": 11, "poly": 3}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = int(self.step_data.get('win', 11))
        p = int(self.step_data.get('poly', 3))
        return x, prep.apply_savgol(y, window=w, poly=p, deriv=0)


@register_node
class Deriv1Node(BaseFilterNode):
    node_type = "deriv1"
    title = "1. Pochodna SG"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "1. Pochodna SG (okno=11)", "win": 11, "poly": 3}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = int(self.step_data.get('win', 11))
        p = int(self.step_data.get('poly', 3))
        return x, prep.apply_savgol(y, window=w, poly=p, deriv=1)


@register_node
class Deriv2Node(BaseFilterNode):
    node_type = "deriv2"
    title = "2. Pochodna (-) SG"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "2. Pochodna SG (-) (okno=11)", "win": 11, "poly": 3}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = int(self.step_data.get('win', 11))
        p = int(self.step_data.get('poly', 3))
        return x, -1.0 * prep.apply_savgol(y, window=w, poly=p, deriv=2)


@register_node
class FsdNode(BaseFilterNode):
    node_type = "fsd"
    title = "FSD (Kauppinen)"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "FSD (FWHM=18.0, K=2.0)", "fwhm": 18.0, "k_factor": 2.0}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fwhm = float(self.step_data.get('fwhm', 18.0))
        k = float(self.step_data.get('k_factor', 2.0))
        if y.ndim == 2:
            out_y = np.zeros_like(y)
            for c in range(y.shape[1]):
                out_y[:, c] = prep.fourier_self_deconvolution(x, y[:, c], fwhm=fwhm, k_factor=k)
            return x, out_y
        return x, prep.fourier_self_deconvolution(x, y, fwhm=fwhm, k_factor=k)


@register_node
class SnvNode(BaseFilterNode):
    node_type = "snv"
    title = "SNV"
    category = "Normalizacje i Skalowania"

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, prep.apply_snv(y)


@register_node
class MinMaxNode(BaseFilterNode):
    node_type = "minmax"
    title = "Min-Max [0, 1]"
    category = "Normalizacje i Skalowania"

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, prep.apply_min_max(y)


@register_node
class L1NormNode(BaseFilterNode):
    node_type = "l1"
    title = "Norma L₁ (Pole)"
    category = "Normalizacje i Skalowania"

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, prep.apply_norm_l1(y)


@register_node
class L2NormNode(BaseFilterNode):
    node_type = "l2"
    title = "Norma L₂"
    category = "Normalizacje i Skalowania"

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, prep.apply_vector_norm_l2(y)


@register_node
class OffsetMinNode(BaseFilterNode):
    node_type = "offset_min"
    title = "Offset (Min ➔ 0.0)"
    category = "Normalizacje i Skalowania"

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if y.ndim == 2:
            return x, y - np.min(y, axis=0, keepdims=True)
        return x, y - np.min(y)


@register_node
class OffsetPointNode(BaseFilterNode):
    node_type = "offset_point"
    title = "Offset w punkcie ν"
    category = "Normalizacje i Skalowania"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {"type": cls.node_type, "name": "Offset w punkcie 2000 cm⁻¹", "target_wn": 2000.0}

    def apply_filter(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t_wn = float(self.step_data.get('target_wn', 2000.0))
        return x, prep.apply_offset(y, x, target_wn=t_wn, target_val=0.0)