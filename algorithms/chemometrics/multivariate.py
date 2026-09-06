"""
algorithms/chemometrics/multivariate.py
Zaawansowana analiza chemometryczna: PFA/Malinowski, SPEXFA, MCR-ALS, PARAFAC, Tucker3, 2D-COS.
Brak zależności od GUI.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from scipy.linalg import pinv
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
import tensorly as tl
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import Constraint, ConstraintNorm

from core.logger import log_algo_exception

logger = logging.getLogger("SpectroHub")

# Bezpieczny import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# --- RESTRYKCJE MCR-ALS ---

class ConstraintClosure(Constraint):
    """Wymusza sumowanie profili stężeń do jedności (domknięcie bilansu)."""
    def __init__(self, axis=-1, copy=False):
        super().__init__(copy)
        self.axis = axis

    def transform(self, A):
        if self.copy:
            A = A.copy()
        sums = A.sum(axis=self.axis, keepdims=True)
        sums[sums == 0] = 1e-12
        A /= sums
        return A


class ConstraintUnimodal(Constraint):
    """Wymusza jednomodalność profili (brak wtórnych maksimów)."""
    def __init__(self, copy=False):
        super().__init__(copy)

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for j in range(A.shape[1]):
            c = A[:, j]
            max_idx = np.argmax(c)
            # Wymuszenie monotonicznego spadku w lewo
            for i in range(max_idx - 1, -1, -1):
                if c[i] > c[i + 1]:
                    c[i] = c[i + 1]
            # Wymuszenie monotonicznego spadku w prawo
            for i in range(max_idx + 1, len(c)):
                if c[i] > c[i - 1]:
                    c[i] = c[i - 1]
            A[:, j] = c
        return A


# --- 1. ANALIZA CZYNNIKÓW GŁÓWNYCH (PCA / PFA) I METODY MANIFOLD ---

def run_pca_uncentered(X: np.ndarray, n_components: int) -> Dict[str, Any]:
    """
    Wykonuje PCA bez centrowania (TruncatedSVD) dla zachowania fizycznego sensu widm FTIR.
    """
    n_comp = min(n_components, X.shape[0], X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp)
    scores = svd.fit_transform(X)

    return {
        'scores': scores,
        'loadings': svd.components_.T,
        'variance': svd.explained_variance_ratio_,
        'pca_model': svd,
        'X_original': X
    }


def run_pca_reconstruction(pca_results: Dict[str, Any], k: int) -> Dict[str, Any]:
    """Rekonstrukcja sygnału z pierwszych k składowych."""
    svd = pca_results['pca_model']
    X_orig = pca_results['X_original']
    scores = pca_results['scores']

    scores_k = np.copy(scores)
    scores_k[:, k:] = 0.0

    X_recon = svd.inverse_transform(scores_k)
    residuals = X_orig - X_recon

    return {
        'X_reconstructed': X_recon,
        'residuals': residuals,
        'k': k
    }


def run_fa_rank_analysis(X: np.ndarray) -> Dict[str, Any]:
    """
    Analiza rang Malinowskiego: Błąd Rzeczywisty (RE), Wskaźnik IND oraz wartości własne.
    """
    r, c = X.shape
    sm = min(r, c)
    max_k = sm - 1
    if max_k < 1:
        raise ValueError("Zbyt mała liczba widm do przeprowadzenia analizy rang.")

    u, s_vec, vh = np.linalg.svd(X, full_matrices=False)
    ev = s_vec ** 2

    re = np.zeros(max_k)
    ind = np.zeros(max_k)
    sev = np.zeros(sm + 1)

    for k_sev in range(sm - 1, -1, -1):
        sev[k_sev] = sev[k_sev + 1] + ev[k_sev]

    for l in range(max_k):
        matlab_l = l + 1
        current_sev = sev[matlab_l]
        re_val = np.sqrt(current_sev / (r * (c - matlab_l)))
        re[l] = re_val
        ind[l] = re_val / ((c - matlab_l) ** 2)

    return {
        'u': u, 's_vec': s_vec, 'vh': vh,
        'ev': ev[:max_k], 're': re, 'ind': ind, 'max_k': max_k,
        'X_original': X
    }


def run_umap(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Nieliniowa redukcja wymiarowości UMAP."""
    if not UMAP_AVAILABLE:
        raise ImportError("Biblioteka 'umap-learn' nie jest zainstalowana w bieżącym środowisku.")
    model = umap.UMAP(n_components=n_components, random_state=42)
    return model.fit_transform(X)


def run_tsne(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Nieliniowa redukcja wymiarowości t-SNE."""
    perplexity = min(30.0, max(1.0, float(X.shape[0] - 1.0)))
    model = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    return model.fit_transform(X)


# --- 2. DEKOMPOZYCJA TENSORÓW 3D (PARAFAC / TUCKER3) ---

def run_parafac(tensor: np.ndarray, rank: int, non_negative: bool = False) -> Dict[str, Any]:
    """Dekompozycja CP/PARAFAC tensora 3D (np. Długość fali x Stężenie A x Stężenie B)."""
    tensor_clean = np.nan_to_num(tensor, nan=0.0)
    if non_negative:
        weights, factors = tl.decomposition.non_negative_parafac(tensor_clean, rank=rank, init='random', n_iter_max=150)
    else:
        weights, factors = tl.decomposition.parafac(tensor_clean, rank=rank, n_iter_max=150)
    return {'weights': weights, 'factors': factors}


def run_tucker(tensor: np.ndarray, ranks: List[int]) -> Dict[str, Any]:
    """Dekompozycja Tucker3 tensora na rdzeń i macierze czynnikowe."""
    tensor_clean = np.nan_to_num(tensor, nan=0.0)
    core, factors = tl.decomposition.tucker(tensor_clean, rank=ranks)
    return {'core': core, 'factors': factors}


# --- 3. SPEXFA & MCR-ALS ---

def run_spexfa(X: np.ndarray, n_components: int) -> Dict[str, Any]:
    """
    Metoda SPEXFA: izolacja czystych widm i profili stężeń za pomocą selekcji IKSFA.
    """
    D = X.T
    r, c = D.shape
    n = n_components

    u, s_vec, vh = np.linalg.svd(D, full_matrices=False)
    s_mat = np.diag(s_vec)

    u_n = u[:, :n]
    s_n = s_mat[:n, :n]
    vh_n = vh[:n, :]
    dr = u_n @ s_n @ vh_n

    ev = s_vec ** 2
    sev = np.sum(ev[n:])
    re = np.sqrt(sev / (r * (c - n)))
    cutoff = 5.0 * re * np.sqrt(n)

    ubar = u_n @ s_n
    ubar_norm = np.linalg.norm(ubar, axis=1)
    mask = ubar_norm < cutoff
    ubar_masked = np.copy(ubar)
    ubar_masked[mask, :] = np.nan

    valid_indices = ~mask
    if np.sum(valid_indices) == 0:
        raise ValueError("Wszystkie zmienne znajdują się poniżej progu szumu.")

    ubar_norm_clean = np.linalg.norm(ubar_masked[valid_indices, :], axis=1, keepdims=True)
    ubar_masked[valid_indices, :] = (ubar_masked[valid_indices, :] * np.sqrt(c)) / ubar_norm_clean

    key = [int(np.nanargmin(np.abs(ubar_masked[:, 0])))]

    for k_idx in range(1, n):
        w = ubar_masked[:, :k_idx + 1]
        ky = np.zeros((k_idx + 1, k_idx + 1))
        for j in range(k_idx):
            ky[j, :] = w[key[j], :]

        dt = np.zeros(r)
        for i in range(r):
            if np.isnan(ubar_masked[i, 0]):
                dt[i] = -np.inf
            else:
                ky[k_idx, :] = w[i, :]
                dt[i] = np.abs(np.linalg.det(ky))
        key.append(int(np.argmax(dt)))

    conc_transposed = dr[key, :]
    spex = dr @ pinv(conc_transposed)

    return {
        'C': conc_transposed.T,
        'ST': spex,
        'rank': n
    }


def run_mcr_als(data_matrix_or_tensor: np.ndarray, rank: int, non_negative: bool = True,
                norm: bool = False, closure: bool = False, unimodal: bool = False,
                use_spexfa_init: bool = True, max_iter: int = 500, tol_err_change: float = 1e-4,
                st_init: Optional[np.ndarray] = None, st_fix_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Wielowymiarowa rozdzielczość krzywych (MCR-ALS).
    Przyjmuje zarówno macierz 2D (W x C), jak i tensor 3D (W x N x M).
    """
    if data_matrix_or_tensor.ndim == 3:
        w, n_rows, m_cols = data_matrix_or_tensor.shape
        D_matrix_full = data_matrix_or_tensor.transpose(1, 2, 0).reshape((n_rows * m_cols), w)
        is_tensor = True
    elif data_matrix_or_tensor.ndim == 2:
        # Wiersze: liczby falowe, Kolumny: próbki -> transpozycja do (próbki x cechy)
        w, n_samples = data_matrix_or_tensor.shape
        D_matrix_full = data_matrix_or_tensor.T
        n_rows, m_cols = n_samples, 1
        is_tensor = False
    else:
        raise ValueError("Dane dla MCR-ALS muszą być macierzą 2D lub tensorem 3D.")

    valid_mask = ~np.isnan(D_matrix_full).any(axis=1) & ~np.all(D_matrix_full == 0.0, axis=1)
    D_matrix = D_matrix_full[valid_mask]

    if D_matrix.shape[0] == 0:
        raise ValueError("Brak prawidłowych punktów dla procedury MCR-ALS.")

    n_samples, n_features = D_matrix.shape

    c_regr = NNLS() if non_negative else OLS()
    st_regr = NNLS() if non_negative else OLS()

    c_constraints = []
    if norm: c_constraints.append(ConstraintNorm())
    if closure: c_constraints.append(ConstraintClosure(axis=1))
    if unimodal: c_constraints.append(ConstraintUnimodal())

    final_st_init = np.random.rand(rank, n_features)

    # Inicjalizacja przez SPEXFA
    if use_spexfa_init and st_init is None:
        try:
            spex_res = run_spexfa(D_matrix, rank)
            final_st_init = spex_res['ST'].T
        except Exception as e:
            logger.warning(f"Inicjalizacja SPEXFA nie powiodła się: {e}. Fallback do liczb losowych.")

    if st_init is not None and st_fix_indices:
        for i, fix_idx in enumerate(st_fix_indices):
            if fix_idx < rank:
                final_st_init[fix_idx, :] = st_init[i, :]

    mcr = McrAR(max_iter=max_iter, tol_err_change=tol_err_change,
                c_regr=c_regr, st_regr=st_regr,
                c_constraints=c_constraints, st_constraints=[])

    mcr.fit(D_matrix, ST=final_st_init, st_fix=st_fix_indices)

    C_full = np.full((n_rows * m_cols, rank), np.nan)
    C_full[valid_mask, :] = mcr.C_opt_

    return {
        'C': C_full if is_tensor else C_full.reshape(n_samples, rank),
        'ST': mcr.ST_opt_.T,  # Kształt: (liczby falowe x rank)
        'rank': rank,
        'converged': getattr(mcr, 'converged', True)
    }


# --- 4. SPEKTROSKOPIA KORELACYJNA 2D (2D-COS) ---

def calculate_2dcos(D: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Oblicza synchroniczną (Phi) i asynchroniczną (Psi) macierz korelacji 2D.
    D: macierz widm dynamicznych (Wiersze: liczby falowe, Kolumny: zmienna perturbująca).
    """
    n_features, n_samples = D.shape
    if n_samples < 2:
        return None, None

    mean_spec = D.mean(axis=1, keepdims=True)
    Y = D - mean_spec
    Phi = (1.0 / (n_samples - 1)) * (Y @ Y.T)

    # Macierz Hilberta-Nody
    N = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                N[i, j] = 1.0 / (np.pi * (j - i))

    Psi = (1.0 / (n_samples - 1)) * (Y @ N @ Y.T)
    return Phi, Psi