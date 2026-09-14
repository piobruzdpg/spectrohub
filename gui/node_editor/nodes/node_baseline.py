"""
gui/node_editor/nodes/node_baseline.py
Autonomiczne węzły korekcji linii bazowej (AsLS, airPLS, Wielomian, Dwa krańce).
Każdy węzeł posiada wejście 'In' oraz dwa wyjścia: 'Syg' (skorygowany) i 'BL' (linia bazowa).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from gui.node_editor.base import BaseNode, PortDef
from gui.node_editor.registry import register_node
from algorithms import preprocessing as prep
from gui.workspaces.ws_preprocessing import StepParamDialog


class BaseBaselineNode(BaseNode):
    """Klasa bazowa dla wszystkich dwuwyjściowych filtrów linii bazowych."""
    category = "Korekcja Linii Bazowej"
    header_gradient = ("#1e3d59", "#17b978")

    def get_ports(self) -> List[PortDef]:
        return [
            PortDef(port_id="in", label="In", is_output=False, color=QColor("#ffb703"), pos_y=90),
            PortDef(port_id="out", label="Syg", is_output=True, color=QColor("#48cae4"), pos_y=70),
            PortDef(port_id="out_bl", label="BL", is_output=True, color=QColor("#9d4edd"), pos_y=112)
        ]

    def _calc_baseline_1d(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Metoda do zaimplementowania w klasach potomnych dla wektora 1D."""
        raise NotImplementedError

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
        data_in = inputs.get("in", (None, None))
        names_in = input_names.get("in", [])

        if data_in[0] is None or data_in[1] is None:
            return {"out": (None, None), "out_bl": (None, None)}, {"out": [], "out_bl": []}

        curr_x, curr_y = data_in
        bl = np.zeros_like(curr_y)

        if curr_y.ndim == 2:
            for c in range(curr_y.shape[1]):
                bl[:, c] = self._calc_baseline_1d(curr_y[:, c], curr_x)
        else:
            bl = self._calc_baseline_1d(curr_y, curr_x)

        corr_y = curr_y - bl

        return {
            "out": (curr_x.copy(), corr_y),
            "out_bl": (curr_x.copy(), bl)
        }, {
            "out": list(names_in),
            "out_bl": list(names_in)
        }

    def create_config_dialog(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        return StepParamDialog(self.step_data, parent=parent)


@register_node
class AslsNode(BaseBaselineNode):
    node_type = "asls"
    title = "AsLS (Syg + Baza)"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "AsLS (λ=1e5, p=0.001)",
            "lam": 1e5,
            "p": 0.001
        }

    def _calc_baseline_1d(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        lam = float(self.step_data.get('lam', 1e5))
        p = float(self.step_data.get('p', 0.001))
        return prep.baseline_asls(y, lam=lam, p=p)


@register_node
class AirPlsNode(BaseBaselineNode):
    node_type = "airpls"
    title = "airPLS / airALS (Syg + Baza)"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "airPLS (λ=1e5)",
            "lam": 1e5,
            "n_iter": 15
        }

    def _calc_baseline_1d(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        lam = float(self.step_data.get('lam', 1e5))
        n_iter = int(self.step_data.get('n_iter', 15))
        return prep.baseline_airpls(y, lam=lam, n_iter=n_iter)


@register_node
class PolyBaselineNode(BaseBaselineNode):
    node_type = "poly_bl"
    title = "Wielomianowa (Syg + Baza)"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Wielomian st. 3",
            "deg": 3
        }

    def _calc_baseline_1d(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        deg = int(self.step_data.get('deg', 3))
        return prep.baseline_polynomial(y, x, deg=deg)


@register_node
class LinearBaselineNode(BaseBaselineNode):
    node_type = "linear_bl"
    title = "Dwa krańce (Syg + Baza)"

    @classmethod
    def default_step_data(cls) -> Dict[str, Any]:
        return {
            "type": cls.node_type,
            "name": "Linia bazowa (krańce)"
        }

    def _calc_baseline_1d(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(x) >= 2:
            return prep.baseline_linear(x, y, x[0], x[-1])
        return np.zeros_like(y)