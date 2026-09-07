"""
algorithms/fitting/deconv_engine.py
Silnik dekonwolucji pasmowej oparty na bibliotece lmfit.
Czysta logika analityczna bez domieszek UI.
Obsługuje tryb dedykowany Amidu I (struktury białkowe) oraz tryb ogólny (Custom).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from lmfit import Parameters, Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel


STRUCTURE_RANGES_H2O = [
    (1610.0, 1625.0, "Beta-Sheet (Aggregated/Strong)"),
    (1625.0, 1640.0, "Beta-Sheet"),
    (1640.0, 1648.0, "Random Coil"),
    (1648.0, 1660.0, "Alpha-Helix"),
    (1660.0, 1685.0, "Beta-Turn"),
    (1685.0, 1698.0, "Beta-Sheet (Anti-parallel)")
]

STRUCTURE_RANGES_D2O = [
    (1610.0, 1620.0, "Beta-Sheet (Aggregated)"),
    (1620.0, 1638.0, "Beta-Sheet"),
    (1638.0, 1648.0, "Random Coil"),
    (1648.0, 1662.0, "Alpha-Helix"),
    (1662.0, 1680.0, "Beta-Turn"),
    (1680.0, 1695.0, "Beta-Sheet (Anti-parallel)")
]


class DeconvolutionEngine:
    """Silnik optymalizacyjny dopasowania sumy pasm do widma eksperymentalnego."""

    MODEL_MAP = {
        'Gaussian': GaussianModel,
        'Lorentzian': LorentzianModel,
        'Voigt': VoigtModel,
        'PseudoVoigt': PseudoVoigtModel
    }

    @staticmethod
    def guess_structure(center: float, solvent: str = "H2O") -> str:
        """Przypisuje strukturę drugorzędową na podstawie położenia piku."""
        ranges = STRUCTURE_RANGES_H2O if solvent == "H2O" else STRUCTURE_RANGES_D2O
        for start, end, name in ranges:
            if start <= center <= end:
                return name
        return "Other / Side Chain"

    @classmethod
    def fit_peaks(cls, x: np.ndarray, y: np.ndarray, peaks: List[Dict[str, Any]],
                  method: str = "leastsq") -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        if len(peaks) == 0 or len(x) == 0:
            raise ValueError("Brak danych lub brak pików do dopasowania.")

        composite_model: Optional[Model] = None
        params = Parameters()

        for p in peaks:
            prefix = f"{p['id']}_"
            model_cls = cls.MODEL_MAP.get(p.get('type', 'Gaussian'), GaussianModel)
            model = model_cls(prefix=prefix)

            if composite_model is None:
                composite_model = model
            else:
                composite_model += model

            # Dodanie parametrów i więzów
            model.set_param_hint(prefix + 'center', value=p['center'], vary=not p.get('lock_center', False))
            model.set_param_hint(prefix + 'sigma', value=p['sigma'], min=1e-4, vary=not p.get('lock_sigma', False))
            model.set_param_hint(prefix + 'amplitude', value=p['area'], min=0.0, vary=not p.get('lock_area', False))

            if p.get('type') == 'Voigt':
                model.set_param_hint(prefix + 'gamma', value=p.get('gamma', p['sigma']), min=1e-4,
                                     vary=not p.get('lock_extra', False))
            elif p.get('type') == 'PseudoVoigt':
                model.set_param_hint(prefix + 'fraction', value=p.get('fraction', 0.5), min=0.0, max=1.0,
                                     vary=not p.get('lock_extra', False))

            params.update(model.make_params())

        # Wykonanie optymalizacji
        result = composite_model.fit(y, params, x=x, method=method)
        if not result.success:
            raise RuntimeError(f"Optymalizacja lmfit nie osiągnęła zbieżności: {result.message}")

        bv = result.best_values
        updated_peaks = [p.copy() for p in peaks]

        for p in updated_peaks:
            prefix = f"{p['id']}_"
            if prefix + 'center' in bv: p['center'] = float(bv[prefix + 'center'])
            if prefix + 'sigma' in bv: p['sigma'] = float(bv[prefix + 'sigma'])
            if prefix + 'amplitude' in bv: p['area'] = float(bv[prefix + 'amplitude'])
            if p.get('type') == 'Voigt' and prefix + 'gamma' in bv:
                p['gamma'] = float(bv[prefix + 'gamma'])
            if p.get('type') == 'PseudoVoigt' and prefix + 'fraction' in bv:
                p['fraction'] = float(bv[prefix + 'fraction'])

        ss_res = np.sum((y - result.best_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y - result.best_fit) ** 2)))

        gof_metrics = {
            'r2': r2,
            'rmse': rmse,
            'chisqr': float(result.chisqr),
            'redchi': float(result.redchi),
            'iterations': result.nfev
        }

        return updated_peaks, gof_metrics

    @classmethod
    def evaluate_model_curves(cls, x: np.ndarray, y: np.ndarray, peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_y = np.zeros_like(x, dtype=np.float64)
        components = []

        for p in peaks:
            model_cls = cls.MODEL_MAP.get(p.get('type', 'Gaussian'), GaussianModel)
            model = model_cls()
            safe_sigma = max(p['sigma'], 1e-4)

            params = model.make_params(amplitude=p['area'], center=p['center'], sigma=safe_sigma)
            if p.get('type') == 'Voigt':
                params.add('gamma', value=p.get('gamma', safe_sigma))
            elif p.get('type') == 'PseudoVoigt':
                params.add('fraction', value=p.get('fraction', 0.5))

            y_comp = model.eval(params, x=x)
            total_y += y_comp
            components.append({'id': p['id'], 'y': y_comp, 'color': p.get('color', '#ee6c4d')})

        residuals = y - total_y
        return {
            'total': total_y,
            'components': components,
            'residuals': residuals
        }