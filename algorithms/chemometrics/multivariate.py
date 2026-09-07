"""
algorithms/chemometrics/multivariate.py
Silnik chemometryczny analizy wielowymiarowej i dekompozycji serii ewolucyjnych.
Czysty rdzeń numeryczny (NumPy / SciPy / scikit-learn / pyMCR) — zero kodu GUI.
"""

import numpy as np
from numpy.linalg import svd, pinv, det
from scipy.stats import f as f_dist
from typing import Dict, Any, List, Tuple, Optional
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS, OLS
from pymcr.constraints import Constraint, ConstraintNorm


# --- WŁASNE OGRANICZENIA DLA MCR-ALS ---

class ConstraintClosure(Constraint):
    """Ograniczenie domknięcia bilansu stężeń (suma stężeń w próbie = 1.0)."""
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
    """Ograniczenie jednomodalności profilu stężeń."""
    def __init__(self, copy=False):
        super().__init__(copy)

    def transform(self, A):
        if self.copy:
            A = A.copy()
        for j in range(A.shape[1]):
            c = A[:, j]
            max_idx = np.argmax(c)
            for i in range(max_idx - 1, -1, -1):
                if c[i] > c[i + 1]:
                    c[i] = c[i + 1]
            for i in range(max_idx + 1, len(c)):
                if c[i] > c[i - 1]:
                    c[i] = c[i - 1]
            A[:, j] = c
        return A


# --- 1. SUBRUTYNY IKSFA I WFAX (MALINOWSKI) ---

def iksfax_py(d: np.ndarray, n: int, typical: int = 1) -> Optional[List[int]]:
    """
    Iterative Key Set Factor Analysis (IKSFA).
    typical: 1 dla wierszy (widma), 2 dla kolumn (stężenia).
    """
    r, c = d.shape
    sm = min(r, c)
    lg = max(r, c)

    u, s_vec, vt = svd(d, full_matrices=False)
    u_scaled = u[:, :n] @ np.diag(s_vec[:n])
    v_scaled = vt.T[:, :n] @ np.diag(s_vec[:n])

    sev = np.sum(s_vec[n:] ** 2) if n < sm else 0.0
    re = np.sqrt(sev / (lg * (sm - n))) if lg * (sm - n) > 0 else 0.0
    cutoff = 5.0 * re * np.sqrt(n)

    target_matrix = u_scaled if typical == 1 else v_scaled
    norms = np.linalg.norm(target_matrix, axis=1)
    valid_indices = np.where(norms >= cutoff)[0]

    if len(valid_indices) < n:
        valid_indices = np.arange(target_matrix.shape[0])

    if len(valid_indices) == 0:
        return None

    ubar = target_matrix.copy()
    ubar[valid_indices, :] = ubar[valid_indices, :] * (np.sqrt(sm) / norms[valid_indices, np.newaxis])

    key = [valid_indices[np.argmin(np.abs(ubar[valid_indices, 0]))]]

    for k in range(1, n):
        current_keys_rows = ubar[key, :k + 1]
        candidate_indices = [idx for idx in valid_indices if idx not in key]
        if not candidate_indices:
            break

        dets = np.zeros(len(candidate_indices))
        for i, idx in enumerate(candidate_indices):
            test_matrix = np.vstack([current_keys_rows, ubar[idx, :k + 1]])
            try:
                dets[i] = np.abs(det(test_matrix))
            except np.linalg.LinAlgError:
                dets[i] = 0.0

        if np.max(dets) > 0:
            key.append(candidate_indices[np.argmax(dets)])
        else:
            break

    # Udoskonalanie kluczy
    for _ in range(20):
        tkey = list(key)
        w = ubar[:, :n]
        for j in range(len(key)):
            current_temp_keys = [k for i, k in enumerate(key) if i != j]
            candidate_indices = [idx for idx in valid_indices if idx not in current_temp_keys]
            if not candidate_indices:
                continue

            dets = np.zeros(len(candidate_indices))
            base_rows = w[current_temp_keys, :]
            for i, idx in enumerate(candidate_indices):
                test_matrix = np.vstack([base_rows, w[idx, :]])
                try:
                    dets[i] = np.linalg.norm(w[idx, :]) if n == 1 else np.abs(det(test_matrix))
                except np.linalg.LinAlgError:
                    dets[i] = 0.0

            if np.max(dets) > 0:
                key[j] = candidate_indices[np.argmax(dets)]

        if key == tkey:
            break

    return key


def wfax_py(d: np.ndarray, n_total: int, w1: int, w2: int) -> Optional[np.ndarray]:
    """Window Factor Analysis (WFAX) z usuwaniem tła instrumentalnego."""
    nr, nc = d.shape
    if n_total <= 0 or n_total > min(nr, nc) or w1 < 0 or w2 >= nc or w1 > w2:
        return None

    u, s_vec, vt = svd(d, full_matrices=False)
    n_eff = min(n_total, len(s_vec))
    dn = u[:, :n_eff] @ np.diag(s_vec[:n_eff]) @ vt[:n_eff, :]

    cols_to_delete = np.arange(max(0, w1), min(nc - 1, w2) + 1)
    db = np.delete(d, cols_to_delete, axis=1) if len(cols_to_delete) > 0 else d.copy()

    n_bg = n_total - 1
    if n_bg <= 0 or db.shape[1] == 0:
        Xn = dn
    else:
        ub, _, _ = svd(db, full_matrices=False)
        n_bg_eff = min(n_bg, ub.shape[1])
        ub_trunc = ub[:, :n_bg_eff]
        Xn = dn - ub_trunc @ (ub_trunc.T @ dn)

    try:
        _, _, vxt = svd(Xn, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if vxt.shape[0] == 0:
        return np.zeros(nc)

    cx = vxt[0, :]
    max_abs = np.max(np.abs(cx))
    return cx / cx[np.argmax(np.abs(cx))] if max_abs > 1e-10 else np.zeros_like(cx)


# --- 2. GŁÓWNE ALGORYTMY ANALIZY EVOCHEM / CHEMTENSOR ---

def run_pfa(d: np.ndarray, n: int = 0) -> Dict[str, Any]:
    """
    Principal Factor Analysis (Malinowski).
    Zwraca wartości własne (EV), błąd rzeczywisty (RE), wskaźnik IND oraz poziom istotności %SL.
    """
    r, c = d.shape
    sm = min(r, c)
    lg = max(r, c)

    u, s_vec, vt = svd(d, full_matrices=False)
    ev = s_vec ** 2

    re = np.full(sm, np.nan)
    ind = np.full(sm, np.nan)
    Msl = np.full(sm, np.nan)

    sev_cumulative = np.concatenate(([0], np.cumsum(ev[::-1])))[::-1]

    for k in range(sm - 1):
        sev_k = sev_cumulative[k + 1]
        denom_re = lg * (sm - (k + 1))
        re[k] = np.sqrt(sev_k / denom_re) if denom_re > 0 and sev_k >= 0 else 0.0

        denom_ind = (sm - (k + 1)) ** 2
        ind[k] = re[k] / denom_ind if denom_ind > 0 and not np.isnan(re[k]) else 0.0

        df1 = (r - k) * (c - k)
        df2 = sum((r - i) * (c - i) for i in range(k + 1, sm))
        if df1 > 0 and df2 > 0 and sev_k > 1e-12:
            f_stat = (df2 * ev[k]) / (df1 * sev_k)
            Msl[k] = f_dist.sf(f_stat, df1, df2) * 100.0

    res = {
        'factors': np.arange(1, sm + 1),
        'ev': ev,
        're': re,
        'ind': ind,
        'msl': Msl,
        'suggested_k': int(np.nanargmin(ind[:max(1, sm - 2)])) + 1 if sm > 2 else 1,
        'X_original': d
    }

    if n > 0:
        n_eff = min(n, sm)
        u_k = u[:, :n_eff]
        s_k = np.diag(s_vec[:n_eff])
        vt_k = vt[:n_eff, :]
        dr = u_k @ s_k @ vt_k
        res.update({
            'dr': dr,
            'residuals': d - dr,
            'scores': u_k @ s_k,
            'loadings': vt_k.T,
            'k': n_eff
        })

    return res


def run_autowfa(d: np.ndarray, n_chem: int, m_total: Optional[int] = None, noise: float = 0.065) -> Dict[str, Any]:
    """Automatyczna analiza czynnikowa z oknem (AUTOWFA)."""
    r, c = d.shape
    if m_total is None:
        m_total = n_chem

    initial_keys = iksfax_py(d, n_chem, typical=2)
    if initial_keys is None or len(initial_keys) < n_chem:
        raise RuntimeError("IKSFA nie zdołała wyznaczyć początkowego zbioru kluczy.")

    u, s_vec, vt = svd(d, full_matrices=False)
    vbar = vt.T[:, :n_chem]

    keys = np.array(initial_keys)
    w1 = np.zeros(n_chem, dtype=int)
    w2 = np.zeros(n_chem, dtype=int)

    for j in range(n_chem):
        ctest = np.zeros((c, 1))
        ctest[keys[j], 0] = 1.0
        cpred = (vbar @ (vbar.T @ ctest)).flatten()
        max_v = np.max(np.abs(cpred))
        cpred = cpred / max_v if max_v > 1e-10 else np.zeros(c)

        mx_idx = int(np.argmax(cpred))
        keys[j] = mx_idx

        # Lewa krawędź
        left = mx_idx
        for k in range(mx_idx, -1, -1):
            if cpred[k] <= 0:
                left = k + 1
                break
            if k == 0: left = 0
        w1[j] = left

        # Prawa krawędź
        right = mx_idx
        for k in range(mx_idx, c):
            if cpred[k] <= 0:
                right = k - 1
                break
            if k == c - 1: right = c - 1
        w2[j] = right

    sort_idx = np.argsort(keys)
    keys, w1, w2 = keys[sort_idx], w1[sort_idx], w2[sort_idx]

    conc = np.zeros((n_chem, c))
    for j in range(n_chem):
        for _ in range(12):
            cpred = wfax_py(d, m_total, w1[j], w2[j])
            if cpred is None:
                break
            conc[j, :] = cpred
            if np.any(cpred):
                mx_idx = int(np.argmax(cpred))
                keys[j] = mx_idx
                new_w1 = 0
                for k in range(mx_idx, -1, -1):
                    if cpred[k] <= noise:
                        new_w1 = k + 1
                        break
                new_w2 = c - 1
                for k in range(mx_idx, c):
                    if cpred[k] <= noise:
                        new_w2 = k - 1
                        break
                w1[j] = min(new_w1, mx_idx)
                w2[j] = max(new_w2, mx_idx)

    return {
        'conc': conc,
        'keys': keys,
        'windows': np.column_stack([keys, w1, w2]),
        'rank': n_chem
    }


def run_spexfa(d: np.ndarray, n: int) -> Dict[str, Any]:
    """
    Spectral Extraction Factor Analysis (SPEXFA).
    Izoluje czyste widma (Pure Spectra), profile stężeń oraz ich niepewności.
    """
    r, c = d.shape
    u, s_vec, vt = svd(d, full_matrices=False)
    dr = u[:, :n] @ np.diag(s_vec[:n]) @ vt[:n, :]

    key_rows = iksfax_py(d, n, typical=1)
    if key_rows is None or len(key_rows) < n:
        raise RuntimeError("IKSFA nie wyznaczyła kluczowych wierszy dla SPEXFA.")

    valid_keys = [k for k in key_rows if k < dr.shape[0]]
    conc_temp = dr[valid_keys, :]
    conc_norm = np.max(conc_temp, axis=1, keepdims=True)
    conc_norm[conc_norm <= 1e-10] = 1.0
    conc_normalized = conc_temp / conc_norm

    spex = dr @ pinv(conc_normalized)
    e = d - dr

    spexerr = np.zeros_like(spex)
    try:
        xx = pinv(conc_normalized @ conc_normalized.T)
        if c > n:
            for j in range(r):
                vj = np.sum(e[j, :] ** 2) / (c - n)
                for k in range(n):
                    spexerr[j, k] = np.sqrt(max(0.0, vj * xx[k, k]))
    except Exception:
        spexerr.fill(np.nan)

    return {
        'pure_spectra': spex,
        'pure_concentrations': conc_normalized.T,
        'spexerr': spexerr,
        'rank': n,
        'dr': dr
    }


def run_mcr_als(d: np.ndarray, rank: int, non_negative: bool = True,
                norm: bool = False, closure: bool = False, unimodal: bool = False,
                use_spexfa_init: bool = True, max_iter: int = 500,
                tol_err_change: float = 1e-4) -> Dict[str, Any]:
    """
    Multivariate Curve Resolution — Alternating Least Squares (MCR-ALS).
    d: Macierz o kształcie (Liczby falowe, Próbki).
    """
    D_samples = d.T  # pyMCR oczekuje macierzy (Próbki, Liczby falowe)
    n_samples, n_features = D_samples.shape

    c_regr = NNLS() if non_negative else OLS()
    st_regr = NNLS() if non_negative else OLS()

    c_constraints = []
    if norm:
        c_constraints.append(ConstraintNorm())
    if closure:
        c_constraints.append(ConstraintClosure(axis=1))
    if unimodal:
        c_constraints.append(ConstraintUnimodal())

    st_init = np.random.rand(rank, n_features)
    if use_spexfa_init:
        try:
            spex_res = run_spexfa(d, rank)
            st_init = spex_res['pure_spectra'].T
        except Exception:
            pass

    mcr = McrAR(max_iter=max_iter, tol_err_change=tol_err_change,
                c_regr=c_regr, st_regr=st_regr,
                c_constraints=c_constraints, st_constraints=[])

    mcr.fit(D_samples, ST=st_init)

    return {
        'pure_concentrations': mcr.C_opt_,      # (Próbki x rank)
        'pure_spectra': mcr.ST_opt_.T,          # (Liczby falowe x rank)
        'rank': rank,
        'iterations': getattr(mcr, 'n_iter_', max_iter)
    }

try:
    import tensorly as tl
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def run_parafac(tensor_3d: np.ndarray, rank: int, non_negative: bool = False) -> Dict[str, Any]:
    """
    Dekompozycja PARAFAC (CANDECOMP) dla tensora 3D (W x N x M).
    Zwraca wyizolowane widma (Mode 0) oraz trendy zmian wzdłuż wymiarów N i M.
    """
    if not HAS_TENSORLY:
        raise ImportError("Brak biblioteki `tensorly`. Zainstaluj ją wpisując: pip install tensorly")

    if tensor_3d.ndim != 3:
        raise ValueError(f"PARAFAC wymaga tensora 3D (otrzymano tablicę o wymiarze {tensor_3d.ndim}D).")

    tensor_clean = np.nan_to_num(tensor_3d, nan=0.0)

    if non_negative:
        weights, factors = tl.decomposition.non_negative_parafac(tensor_clean, rank=rank, init='random', n_iter_max=150)
    else:
        weights, factors = tl.decomposition.parafac(tensor_clean, rank=rank, n_iter_max=150)

    return {
        'pure_spectra': factors[0],      # Wymiar (W x rank)
        'mode1_trends': factors[1],      # Wymiar (N x rank)
        'mode2_trends': factors[2],      # Wymiar (M x rank)
        'weights': weights,
        'rank': rank
    }

def run_fsmwefa(d: np.ndarray, window_size: int = 7) -> Dict[str, Any]:
    """Fixed-Size Moving Window EFA (FSMWEFA)."""
    r, c = d.shape
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= c:
        raise ValueError("Szerokość okna musi być mniejsza od liczby próbek w serii.")

    half_w = window_size // 2
    max_factors = min(r, window_size)
    efa_matrix = np.full((max_factors, c), np.nan)

    for i in range(c - window_size + 1):
        sub = d[:, i:i + window_size]
        _, s_vec, _ = svd(sub, full_matrices=False)
        center_idx = i + half_w
        efa_matrix[:len(s_vec), center_idx] = s_vec

    return {
        'efa_matrix': efa_matrix,
        'max_factors': max_factors,
        'window_size': window_size
    }


def calculate_2dcos(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wyznacza mapę synchroniczną (Phi) i asynchroniczną (Psi) dla 2D-COS."""
    n_features, n_samples = d.shape
    if n_samples < 2:
        raise ValueError("Zbyt mało próbek do 2D-COS (wymagane min. 2).")

    mean_spec = d.mean(axis=1, keepdims=True)
    y = d - mean_spec
    phi = (1.0 / (n_samples - 1)) * (y @ y.T)

    n_mat = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                n_mat[i, j] = 1.0 / (np.pi * (j - i))

    psi = (1.0 / (n_samples - 1)) * (y @ n_mat @ y.T)
    return phi, psi