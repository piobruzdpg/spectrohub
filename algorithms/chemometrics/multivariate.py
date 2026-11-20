"""
algorithms/chemometrics/multivariate.py
Zunifikowany silnik chemometryczny analizy wielowymiarowej i dekompozycji serii ewolucyjnych.
Zawiera: PCA, PFA Malinowskiego, SPEXFA, AUTOWFA, FSMWEFA, HELP, MCR-ALS, PARAFAC, Tucker3, 2D-COS, UMAP/t-SNE.
Czysty rdzeń numeryczny (NumPy / SciPy / scikit-learn / pyMCR / TensorLy) — zero kodu GUI.
"""

import numpy as np
from numpy.linalg import svd, pinv, det
from scipy.stats import f as f_dist
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS, OLS
from pymcr.constraints import Constraint, ConstraintNorm, ConstraintNonneg
from typing import Dict, Any, List, Optional, Tuple

# Opcjonalne biblioteki
try:
    import tensorly as tl
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =====================================================================
# WŁASNE OGRANICZENIA DLA MCR-ALS
# =====================================================================

class ConstraintClosure(Constraint):
    """Ograniczenie domknięcia bilansu stężeń (suma stężeń w danej próbce = 1.0)."""
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
    """Ograniczenie jednomodalności profilu stężeń (pojedyncze maksimum bez oscylacji)."""
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


# =====================================================================
# 1. SUBRUTYNY IKSFA I WFAX (MALINOWSKI)
# =====================================================================

def iksfax_py(d: np.ndarray, n: int, typical: int = 1) -> Optional[List[int]]:
    """Iterative Key Set Factor Analysis (IKSFA). typical: 1 dla wierszy, 2 dla kolumn."""
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


# =====================================================================
# 2. PCA ORAZ PFA MALINOWSKIEGO (Z REKONSTRUKCJĄ)
# =====================================================================

def run_pfa(d: np.ndarray, n: int = 0) -> Dict[str, Any]:
    """
    Principal Factor Analysis (Malinowski).
    d: Macierz (Liczby falowe x Próbki).
    Wyznacza: EV, RE, IND, %SL, oraz dla zadanego n: Scores, Loadings, Rekonstrukcję (dr) i Residua.
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

    suggested_k = int(np.nanargmin(ind[:max(1, sm - 2)])) + 1 if sm > 2 else 1

    res = {
        'factors': np.arange(1, sm + 1),
        'ev': ev,
        're': re,
        'ind': ind,
        'msl': Msl,
        'suggested_k': suggested_k,
        'X_original': d,
        'u': u,
        's_vec': s_vec,
        'vt': vt
    }

    if n > 0:
        n_eff = min(n, sm)
        u_k = u[:, :n_eff]
        s_k = np.diag(s_vec[:n_eff])
        vt_k = vt[:n_eff, :]

        dr = u_k @ s_k @ vt_k
        residuals = d - dr
        res.update({
            'dr': dr,
            'residuals': residuals,
            'scores': vt_k.T @ s_k,       # Profile w dziedzinie próbek (Próbki x k)
            'loadings': u_k,              # Czyste widmowe ładunki (Liczby falowe x k)
            'k': n_eff,
            'residual_std': np.std(residuals)
        })

    return res


def run_pca(d: np.ndarray, n_components: int, center: bool = True) -> Dict[str, Any]:
    """
    Principal Component Analysis (PCA).
    d: Macierz (Liczby falowe x Próbki).
    center=True: klasyczne PCA z centrowaniem widm do średniej.
    center=False: niecentrowane SVD (Truncated SVD).
    """
    X = d.T  # (Próbki x Liczby falowe)
    n_samples, n_features = X.shape
    k = min(n_components, n_samples, n_features)

    if center:
        mean_vector = np.mean(X, axis=0, keepdims=True)
        X_proc = X - mean_vector
    else:
        mean_vector = np.zeros((1, n_features))
        X_proc = X

    u, s, vt = svd(X_proc, full_matrices=False)

    scores = u[:, :k] @ np.diag(s[:k])     # (Próbki x k)
    loadings = vt[:k, :].T                 # (Liczby falowe x k)
    ev = s ** 2
    total_var = np.sum(ev)
    var_exp = (ev[:k] / total_var) * 100.0 if total_var > 0 else np.zeros(k)

    # Rekonstrukcja macierzy
    X_recon = (scores @ loadings.T) + mean_vector
    dr = X_recon.T                         # (Liczby falowe x Próbki)
    residuals = d - dr

    return {
        'scores': scores,
        'loadings': loadings,
        'ev': ev,
        'explained_variance_ratio': var_exp,
        'dr': dr,
        'residuals': residuals,
        'mean_spectrum': mean_vector.flatten(),
        'k': k,
        'residual_std': np.std(residuals)
    }


# =====================================================================
# 3. SPEXFA, AUTOWFA, FSMWEFA, HELP
# =====================================================================

def run_spexfa(d: np.ndarray, n: int) -> Dict[str, Any]:
    """Spectral Extraction Factor Analysis (SPEXFA). Izoluje widma czyste, profile i błędy."""
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
        'pure_spectra': spex,                     # (Liczby falowe x n)
        'pure_concentrations': conc_normalized.T, # (Próbki x n)
        'spexerr': spexerr,
        'rank': n,
        'dr': dr,
        'residuals': e
    }


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

        left = mx_idx
        for k in range(mx_idx, -1, -1):
            if cpred[k] <= 0:
                left = k + 1
                break
            if k == 0: left = 0
        w1[j] = left

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


def run_help_rank_profile(d: np.ndarray, window_size: int = 5, noise_level: float = 0.05) -> Dict[str, Any]:
    """
    Heuristic Evolving Latent Projections (HELP) — Local Rank Profiling.
    Wykrywa strefy czystych składników (Rank 1) oraz strefy ko-elucji/nakładania.
    """
    r, c = d.shape
    if window_size >= c:
        raise ValueError("Okno HELP musi być mniejsze niż liczba widm w serii.")

    half_w = window_size // 2
    rank_profile = np.zeros(c, dtype=int)

    for i in range(c - window_size + 1):
        sub = d[:, i:i + window_size]
        _, s_vec, _ = svd(sub, full_matrices=False)
        local_rank = np.sum(s_vec > noise_level)
        rank_profile[i + half_w] = max(1, local_rank)

    return {
        'rank_profile': rank_profile,
        'window_size': window_size,
        'noise_level': noise_level
    }


# =====================================================================
# 4. MCR-ALS (Z METRYKAMI LOF, R² I FIXED COMPONENTS)
# =====================================================================

def run_mcr_als(d: np.ndarray, rank: int, non_negative: bool = True,
                norm: bool = False, closure: bool = False, unimodal: bool = False,
                use_spexfa_init: bool = True, max_iter: int = 500,
                tol_err_change: float = 1e-6,
                tol_increase: float = 0.05,
                tol_n_increase: int = 25,
                tol_n_above_min: int = 50,
                st_init: Optional[np.ndarray] = None,
                st_fix_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    MCR-ALS ze szczegółową diagnostyką GoF: Lack of Fit (%LOF) oraz R².
    Obsługuje twarde więzy nieujemności oraz widma referencyjne (st_init, st_fix_indices).
    d: Macierz (Liczby falowe x Próbki).
    """
    D_samples = d.T  # (Próbki x Liczby falowe)
    n_samples, n_features = D_samples.shape

    c_regr = NNLS() if non_negative else OLS()
    st_regr = NNLS() if non_negative else OLS()

    c_constraints = []
    st_constraints = []

    # 1. Twarde więzy nieujemności na oba kanały
    if non_negative:
        c_constraints.append(ConstraintNonneg())
        st_constraints.append(ConstraintNonneg())

    if norm:
        st_constraints.append(ConstraintNorm())
    if closure:
        c_constraints.append(ConstraintClosure(axis=1))
    if unimodal:
        c_constraints.append(ConstraintUnimodal())

    # 2. Konstrukcja macierzy startowej ST_init (rank x Liczby falowe)
    final_st_init = np.random.rand(rank, n_features)
    if use_spexfa_init:
        try:
            spex_res = run_spexfa(d, rank)
            final_st_init = spex_res['pure_spectra'].T.copy()
        except Exception:
            pass

    # Jeśli użytkownik przekazał znane widma referencyjne (1 lub więcej)
    if st_init is not None:
        st_arr = np.asarray(st_init, dtype=np.float64)
        if st_arr.ndim == 1:
            st_arr = st_arr[np.newaxis, :]
        m_ref = min(st_arr.shape[0], rank)
        final_st_init[:m_ref, :] = st_arr[:m_ref, :]

    if non_negative:
        final_st_init = np.clip(final_st_init, 0.0, None)

    # 3. Dopasowanie pyMCR z pełną kontrolą kryteriów zatrzymania i blokowaniem widm
    err_change_arg = tol_err_change if (tol_err_change is not None and tol_err_change > 0.0) else None

    mcr = McrAR(
        max_iter=max_iter,
        tol_err_change=err_change_arg,
        tol_increase=tol_increase,
        tol_n_increase=tol_n_increase,
        tol_n_above_min=tol_n_above_min,
        c_regr=c_regr,
        st_regr=st_regr,
        c_constraints=c_constraints,
        st_constraints=st_constraints
    )

    mcr.fit(D_samples, ST=final_st_init, st_fix=st_fix_indices)

    C_opt = mcr.C_opt_          # (Próbki x rank)
    ST_opt = mcr.ST_opt_        # (rank x Liczby falowe)

    if non_negative:
        C_opt = np.clip(C_opt, 0.0, None)
        ST_opt = np.clip(ST_opt, 0.0, None)

    # 4. Rekonstrukcja i statystyki jakości
    D_recon = C_opt @ ST_opt
    residuals = D_samples - D_recon

    ss_res = np.sum(residuals ** 2)
    ss_total = np.sum(D_samples ** 2)
    mean_D = np.mean(D_samples)
    ss_tot_mean = np.sum((D_samples - mean_D) ** 2)

    lof = 100.0 * np.sqrt(ss_res / ss_total) if ss_total > 0 else 0.0
    r2 = 100.0 * (1.0 - (ss_res / ss_tot_mean)) if ss_tot_mean > 0 else 100.0

    return {
        'pure_concentrations': C_opt,          # (Próbki x rank)
        'pure_spectra': ST_opt.T,              # (Liczby falowe x rank)
        'dr': D_recon.T,                       # (Liczby falowe x Próbki)
        'residuals': residuals.T,
        'rank': rank,
        'iterations': int(getattr(mcr, 'n_iter_', max_iter)),
        'converged': bool(getattr(mcr, 'converged', False)),
        'lof': float(lof),
        'r2': float(r2)
    }


# =====================================================================
# 5. METODY TENSOROWE (PARAFAC, TUCKER3) I 2D-COS
# =====================================================================

def run_parafac(tensor_3d: np.ndarray, rank: int, non_negative: bool = False,
                n_iter_max: int = 150) -> Dict[str, Any]:
    """Dekompozycja PARAFAC z rekonstrukcją tensora i residuów."""
    if not HAS_TENSORLY:
        raise ImportError("Brak biblioteki `tensorly`. Zainstaluj ją: pip install tensorly")

    tensor_clean = np.nan_to_num(tensor_3d, nan=0.0)

    if non_negative:
        weights, factors = tl.decomposition.non_negative_parafac(
            tensor_clean, rank=rank, init='random', n_iter_max=n_iter_max
        )
    else:
        weights, factors = tl.decomposition.parafac(
            tensor_clean, rank=rank, n_iter_max=n_iter_max
        )

    recon_tensor = tl.cp_to_tensor((weights, factors))
    residual_tensor = tensor_clean - recon_tensor

    return {
        'pure_spectra': factors[0],      # (W x rank)
        'mode1_trends': factors[1],      # (N x rank)
        'mode2_trends': factors[2],      # (M x rank)
        'weights': weights,
        'rank': rank,
        'recon_tensor': recon_tensor,
        'residual_tensor': residual_tensor
    }


def run_tucker(tensor_3d: np.ndarray, ranks: List[int]) -> Dict[str, Any]:
    """Dekompozycja Tucker3 dla tensora 3D."""
    if not HAS_TENSORLY:
        raise ImportError("Brak biblioteki `tensorly`.")

    tensor_clean = np.nan_to_num(tensor_3d, nan=0.0)
    core, factors = tl.decomposition.tucker(tensor_clean, rank=ranks)
    recon_tensor = tl.tucker_to_tensor((core, factors))

    return {
        'core': core,
        'factors': factors,
        'recon_tensor': recon_tensor,
        'residual_tensor': tensor_clean - recon_tensor
    }


def calculate_2dcos(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wyznacza mapę synchroniczną (Phi) i asynchroniczną (Psi) dla 2D-COS (Noda)."""
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


# =====================================================================
# 6. REDUKCJA NIELINIOWA (UMAP / t-SNE)
# =====================================================================

def run_manifold_reduction(d: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    """Nieliniowe rzutowanie próbek widmowych (d: Liczby falowe x Próbki)."""
    X = d.T  # (Próbki x Liczby falowe)
    if method.lower() == "umap":
        if not HAS_UMAP:
            raise ImportError("Brak biblioteki umap-learn.")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    elif method.lower() == "tsne":
        if not HAS_SKLEARN:
            raise ImportError("Brak biblioteki scikit-learn.")
        perplexity = min(30.0, max(1.0, X.shape[0] - 1.0))
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        return reducer.fit_transform(X)
    raise ValueError(f"Nieobsługiwana metoda rzutowania: {method}")
