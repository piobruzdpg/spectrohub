"""
algorithms/quantum/ped_engine.py
Kompleksowy silnik PED, analizy wibracyjnej oraz efektu izotopowego dla Gaussiana i ORCA 6.1.
Czysty rdzeń numeryczny (NumPy / SciPy).
"""

import re
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import scipy.linalg

ATOMIC_MASSES: Dict[int, float] = {
    1: 1.007825, 3: 7.016004, 5: 11.009305, 6: 12.000000,
    7: 14.003074, 8: 15.994915, 9: 18.998403, 14: 27.976927,
    15: 30.973762, 16: 31.972071, 17: 34.968853, 35: 78.918337, 53: 126.904473
}

COVALENT_RADII: Dict[int, float] = {
    1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

ELEMENT_SYMBOLS: Dict[int, str] = {
    1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

SYMBOL_TO_ATOMIC_NUM: Dict[str, int] = {v.upper(): k for k, v in ELEMENT_SYMBOLS.items()}


class WilsonSVectors:
    """Analityczne wektory S Wilsona dla współrzędnych wewnętrznych."""

    @staticmethod
    def stretch(rA: np.ndarray, rB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vec = rB - rA
        norm = np.linalg.norm(vec)
        u = vec / norm if norm > 1e-8 else np.zeros(3)
        return -u, u

    @staticmethod
    def bend(rA: np.ndarray, rB: np.ndarray, rC: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rBA = rA - rB
        rBC = rC - rB
        dBA = np.linalg.norm(rBA)
        dBC = np.linalg.norm(rBC)

        if dBA < 1e-6 or dBC < 1e-6:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        eBA = rBA / dBA
        eBC = rBC / dBC
        cos_t = np.clip(np.dot(eBA, eBC), -0.999999, 0.999999)
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t ** 2))

        if sin_t < 1e-5:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        sA = (cos_t * eBA - eBC) / (dBA * sin_t)
        sC = (cos_t * eBC - eBA) / (dBC * sin_t)
        sB = -(sA + sC)
        return np.nan_to_num(sA), np.nan_to_num(sB), np.nan_to_num(sC)

    @staticmethod
    def out_of_plane(rA: np.ndarray, rB: np.ndarray, rC: np.ndarray, rD: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rBA = rA - rB
        rBC = rC - rB
        rBD = rD - rB

        dBC, dBD, dBA = np.linalg.norm(rBC), np.linalg.norm(rBD), np.linalg.norm(rBA)
        if dBC < 1e-6 or dBD < 1e-6 or dBA < 1e-6:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        eBC = rBC / dBC
        eBD = rBD / dBD
        n = np.cross(eBC, eBD)
        norm_n = np.linalg.norm(n)
        n = n / norm_n if norm_n > 1e-5 else np.array([0.0, 0.0, 1.0])

        sA = n / dBA
        sC = np.cross(n, eBD) / dBC
        sD = np.cross(eBC, n) / dBD
        sB = -(sA + sC + sD)
        return np.nan_to_num(sA), np.nan_to_num(sB), np.nan_to_num(sC), np.nan_to_num(sD)


class PEDAnalyzer:
    """Parser i analizator kwantowo-chemiczny PED Pulaya."""

    @staticmethod
    def parse_orca_out(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray]:
        """Parsuje plik wyjściowy ORCA 5/6 (.out) z obliczeń wibracyjnych."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 1. Geometria (szukamy ostatniego bloku CARTESIAN COORDINATES (ANGSTROEM))
        coord_matches = list(re.finditer(r'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n[-]+\s*\n((?:[A-Za-z]+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\n)+)', content))
        if not coord_matches:
            # Fallback dla formatu z numerami atomów
            coord_matches = list(re.finditer(r'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n[-]+\s*\n((?:\s*\d+\s+[A-Za-z]+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\n)+)', content))

        if not coord_matches:
            raise ValueError("Nie znaleziono współrzędnych kartezjańskich w pliku wyjściowym ORCA.")

        last_coord_block = coord_matches[-1].group(1).strip().split('\n')
        atomnos_list = []
        coords_list = []

        for line in last_coord_block:
            parts = line.split()
            if not parts:
                continue
            if parts[0].isdigit():
                sym = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            else:
                sym = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

            an = SYMBOL_TO_ATOMIC_NUM.get(sym.upper(), 6)
            atomnos_list.append(an)
            coords_list.append([x, y, z])

        atomnos = np.array(atomnos_list, dtype=int)
        coords = np.array(coords_list, dtype=float)
        n_atoms = len(atomnos)

        # 2. Częstości wibracyjne
        freq_match = re.search(r'VIBRATIONAL FREQUENCIES\s*\n[-]+\s*\n((?:\s*\d+:\s+[-+]?\d+\.\d+\s+cm\*\*-1\s*\n)+)', content)
        if not freq_match:
            raise ValueError("Nie znaleziono sekcji VIBRATIONAL FREQUENCIES w pliku ORCA.")

        raw_freqs = []
        for line in freq_match.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                raw_freqs.append(float(parts[1]))

        # 3. Intensywności IR
        ir_match = re.search(r'IR SPECTRUM\s*\n[-]+\s*\n((?:\s*\d+:[\s\d\.\-+eE]+\n)+)', content)
        intens_dict = {}
        if ir_match:
            for line in ir_match.group(1).strip().split('\n'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        m_idx = int(parts[0].replace(':', ''))
                        intens_dict[m_idx] = float(parts[3])  # Intensity (km/mol)
                    except ValueError:
                        pass

        # 4. Wektory drgań własnych (NORMAL MODES)
        modes_match = re.search(r'NORMAL MODES\s*\n[-]+\s*\n((?:.|\n)*?)(?:\n\s*\n\s*\n|IR SPECTRUM|THERMOCHEMISTRY|\Z)', content)
        if not modes_match:
            raise ValueError("Nie znaleziono sekcji NORMAL MODES w pliku ORCA.")

        modes_text = modes_match.group(1).strip().split('\n')
        total_modes = len(raw_freqs)
        dim = n_atoms * 3
        Lx = np.zeros((dim, total_modes), dtype=float)

        col_modes = []
        row_coord_idx = 0

        for line in modes_text:
            parts = line.split()
            if not parts:
                continue

            # Wiersz z indeksami modów
            if all(p.isdigit() for p in parts):
                col_modes = [int(p) for p in parts]
                continue

            # Wiersz ze składową przemieszczenia (np. '0C  X  0.0012 0.0345 ...' lub '0X 0.0012 ...')
            floats = []
            for p in parts[1:]:
                try:
                    floats.append(float(p))
                except ValueError:
                    pass

            if len(floats) == len(col_modes):
                for c_idx, m_num in enumerate(col_modes):
                    if m_num < total_modes and row_coord_idx < dim:
                        Lx[row_coord_idx, m_num] = floats[c_idx]
                row_coord_idx += 1
                if row_coord_idx == dim:
                    row_coord_idx = 0

        # Filtrujemy tylko rzeczywiste mody drgań (pomijamy 5 lub 6 modów translacji i rotacji o ν ~ 0)
        vib_indices = [i for i, f in enumerate(raw_freqs) if f > 10.0]
        final_freqs = [raw_freqs[i] for i in vib_indices]
        final_intens = [intens_dict.get(i, 0.0) for i in vib_indices]
        final_Lx = Lx[:, vib_indices]

        return atomnos, coords, final_freqs, final_intens, final_Lx

    @staticmethod
    def read_fchk_hessian(fchk_path: str, n_atoms: int) -> np.ndarray:
        with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start_idx = -1
        for idx, line in enumerate(lines):
            if "Cartesian Force Constants" in line:
                start_idx = idx + 1
                break

        if start_idx == -1:
            raise ValueError("Brak sekcji Cartesian Force Constants w pliku .fchk")

        dim = n_atoms * 3
        expected = (dim * (dim + 1)) // 2
        raw = []
        for line in lines[start_idx:]:
            if re.match(r'^[A-Za-z]', line.strip()):
                break
            for t in line.split():
                try:
                    raw.append(float(t))
                except ValueError:
                    pass
            if len(raw) >= expected:
                break

        Fx = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                Fx[i, j] = raw[idx]
                Fx[j, i] = raw[idx]
                idx += 1
        return Fx

    @staticmethod
    def read_fchk_dipole_derivatives(fchk_path: str, n_atoms: int) -> Optional[np.ndarray]:
        """
        Parsuje sekcję 'Dipole Derivatives' z pliku .fchk (Atomic Polar Tensors - APT).
        Zwraca macierz kartezjańskich pochodnych dipolowych o wymiarach (3, 3 * n_atoms).
        """
        if not os.path.exists(fchk_path):
            return None
        try:
            with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            start_idx = -1
            for idx, line in enumerate(lines):
                if "Dipole Derivatives" in line:
                    start_idx = idx + 1
                    break

            if start_idx == -1:
                return None

            dim = 3 * n_atoms
            expected = 3 * dim  # 3 składowe dipola (x, y, z) x 3N współrzędnych
            raw = []
            for line in lines[start_idx:]:
                if re.match(r'^[A-Za-z]', line.strip()):
                    break
                for t in line.split():
                    try:
                        raw.append(float(t))
                    except ValueError:
                        pass
                if len(raw) >= expected:
                    break

            if len(raw) < expected:
                return None

            # Gaussian zapisuje wierszami po 3 składowe: d(mux)/dxi, d(muy)/dxi, d(muz)/dxi
            apt_matrix = np.array(raw[:expected], dtype=np.float64).reshape((dim, 3)).T
            return apt_matrix  # Kształt: (3, 3 * n_atoms)
        except Exception:
            return None

    @staticmethod
    def parse_gaussian_log(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray]:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        blocks = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "orientation" in line.lower():
                atoms = []
                j = i + 1
                dash_count = 0
                while j < len(lines):
                    if "---" in lines[j]:
                        dash_count += 1
                        if dash_count == 2:
                            j += 1
                            break
                    j += 1
                while j < len(lines) and "---" not in lines[j]:
                    parts = lines[j].split()
                    if len(parts) >= 5:
                        try:
                            an = int(parts[1])
                            if an <= 0: an = 1
                            x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                            atoms.append((an, x, y, z))
                        except ValueError:
                            pass
                    j += 1
                if atoms:
                    blocks.append(atoms)
                i = j
            else:
                i += 1

        if not blocks:
            raise ValueError("Nie znaleziono sekcji ze współrzędnymi w pliku .log")

        freqs, intens, displacements = [], [], []
        mode_atom_indices = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Frequencies --" in line:
                f_tokens = line.split("--")[1].split()
                try:
                    block_freqs = [float(x) for x in f_tokens]
                except ValueError:
                    i += 1
                    continue

                n_block = len(block_freqs)
                if n_block == 0:
                    i += 1
                    continue

                block_intens = [0.0] * n_block
                j = i + 1
                disp_found = False

                while j < min(i + 35, len(lines)):
                    if "IR Inten" in lines[j] and "--" in lines[j]:
                        try:
                            block_intens = [float(x) for x in lines[j].split("--")[1].split()]
                        except Exception:
                            pass

                    line_j = lines[j].strip().lower()
                    if any(k in line_j for k in ["atom", "center"]) and any(k in line_j for k in ["an", "type", "x", "coord"]):
                        j += 1
                        disp_found = True
                        break
                    j += 1

                if not disp_found:
                    i = j
                    continue

                block_disps = [[] for _ in range(n_block)]
                curr_atoms = []

                while j < len(lines):
                    row_line = lines[j].strip()
                    if not row_line:
                        if len(block_disps[0]) > 0: break
                        j += 1
                        continue
                    if any(k in row_line for k in ["Frequencies --", "Harmonic frequencies", "---", "Thermochemistry"]):
                        break

                    parts = row_line.split()
                    floats = [float(x) for x in re.findall(r'[-+]?\d+\.\d+', row_line)]

                    if len(floats) >= n_block * 3:
                        try:
                            a_idx = int(parts[0])
                            coords_chunk = floats[-n_block * 3:]
                            for m in range(n_block):
                                block_disps[m].append([coords_chunk[m*3], coords_chunk[m*3+1], coords_chunk[m*3+2]])
                            curr_atoms.append(a_idx)
                        except (ValueError, IndexError):
                            if len(block_disps[0]) > 0: break
                    else:
                        if len(block_disps[0]) > 0: break
                    j += 1

                if len(block_disps[0]) > 0:
                    if not mode_atom_indices:
                        mode_atom_indices = curr_atoms
                    for m in range(n_block):
                        freqs.append(block_freqs[m])
                        intens.append(block_intens[m] if m < len(block_intens) else 0.0)
                        displacements.append(block_disps[m])
                i = j
            else:
                i += 1

        n_modes = len(freqs)
        n_mode_atoms = len(displacements[0])
        matching_blocks = [b for b in blocks if len(b) == n_mode_atoms]
        chosen_block = matching_blocks[-1] if matching_blocks else blocks[-1]

        atomnos = np.array([a[0] for a in chosen_block], dtype=int)
        coords = np.array([[a[1], a[2], a[3]] for a in chosen_block], dtype=float)
        n_atoms = len(atomnos)

        displacements_np = np.array(displacements)
        Lx = displacements_np.reshape(n_modes, n_atoms * 3).T
        return atomnos, coords, freqs, intens, Lx

    @classmethod
    def build_sic_matrix(cls, atomnos: np.ndarray, coords: np.ndarray,
                         mass_replacements: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, List[str]]:
        n_atoms = len(atomnos)
        symbols = []
        for i, num in enumerate(atomnos):
            if mass_replacements and i in mass_replacements:
                symbols.append("D")
            else:
                symbols.append(ELEMENT_SYMBOLS.get(int(num), f"X{num}"))

        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_cov = COVALENT_RADII.get(int(atomnos[i]), 0.8) + COVALENT_RADII.get(int(atomnos[j]), 0.8)
                if np.linalg.norm(coords[i] - coords[j]) <= (r_cov + 0.35):
                    bonds.append((i, j))

        adj = {i: [] for i in range(n_atoms)}
        for a, b in bonds:
            adj[a].append(b)
            adj[b].append(a)

        processed_bonds = set()
        processed_angles = set()
        b_rows = []
        labels = []

        # Grupy XY3 (np. -CH3, -CD3)
        for center in range(n_atoms):
            neighbors = adj[center]
            h_neighbors = [n for n in neighbors if atomnos[n] == 1]
            if len(h_neighbors) == 3:
                syms = [symbols[h] for h in h_neighbors]
                # Tylko jeśli wszystkie 3 atomy to ten sam izotop (same H lub same D)
                if syms[0] == syms[1] == syms[2]:
                    h1, h2, h3 = h_neighbors
                    for h in h_neighbors:
                        processed_bonds.add(tuple(sorted((center, h))))
                    processed_angles.add((min(h2, h3), center, max(h2, h3)))
                    processed_angles.add((min(h1, h3), center, max(h1, h3)))
                    processed_angles.add((min(h1, h2), center, max(h1, h2)))

                    s_label = f"{symbols[center]}{syms[0]}3"
                    sA1, sB1 = WilsonSVectors.stretch(coords[center], coords[h1])
                    sA2, sB2 = WilsonSVectors.stretch(coords[center], coords[h2])
                    sA3, sB3 = WilsonSVectors.stretch(coords[center], coords[h3])

                    row_s = np.zeros(n_atoms * 3)
                    row_s[center * 3:center * 3 + 3] = (sA1 + sA2 + sA3) / np.sqrt(3)
                    row_s[h1 * 3:h1 * 3 + 3] = sB1 / np.sqrt(3)
                    row_s[h2 * 3:h2 * 3 + 3] = sB2 / np.sqrt(3)
                    row_s[h3 * 3:h3 * 3 + 3] = sB3 / np.sqrt(3)
                    b_rows.append(row_s)
                    labels.append(f"ν_s({s_label})")

        # Grupy XY2 (np. -CH2-, -CD2-, -NH2, -ND2)
        for center in range(n_atoms):
            neighbors = adj[center]
            for i_idx in range(len(neighbors)):
                for j_idx in range(i_idx + 1, len(neighbors)):
                    y1, y2 = neighbors[i_idx], neighbors[j_idx]
                    if symbols[y1] == symbols[y2] and (tuple(sorted((center, y1))) not in processed_bonds):
                        processed_bonds.add(tuple(sorted((center, y1))))
                        processed_bonds.add(tuple(sorted((center, y2))))
                        processed_angles.add((min(y1, y2), center, max(y1, y2)))

                        group_tag = f"{symbols[center]}{symbols[y1]}2"
                        sA1, sB1 = WilsonSVectors.stretch(coords[center], coords[y1])
                        sA2, sB2 = WilsonSVectors.stretch(coords[center], coords[y2])

                        row_s = np.zeros(n_atoms * 3)
                        row_s[center * 3:center * 3 + 3] = (sA1 + sA2) / np.sqrt(2)
                        row_s[y1 * 3:y1 * 3 + 3] = sB1 / np.sqrt(2)
                        row_s[y2 * 3:y2 * 3 + 3] = sB2 / np.sqrt(2)
                        b_rows.append(row_s)
                        labels.append(f"ν_s({group_tag})")

                        row_as = np.zeros(n_atoms * 3)
                        row_as[center * 3:center * 3 + 3] = (sA1 - sA2) / np.sqrt(2)
                        row_as[y1 * 3:y1 * 3 + 3] = sB1 / np.sqrt(2)
                        row_as[y2 * 3:y2 * 3 + 3] = -sB2 / np.sqrt(2)
                        b_rows.append(row_as)
                        labels.append(f"ν_as({group_tag})")

        # Pozostałe wiązania (w tym pojedyncze C-D, O-D, S-D itp.)
        for a, b in bonds:
            bond_key = tuple(sorted((a, b)))
            if bond_key not in processed_bonds:
                sA, sB = WilsonSVectors.stretch(coords[a], coords[b])
                row = np.zeros(n_atoms * 3)
                row[a * 3:a * 3 + 3] = sA
                row[b * 3:b * 3 + 3] = sB
                b_rows.append(row)
                labels.append(f"ν({symbols[a]}{a + 1}-{symbols[b]}{b + 1})")

        # Pozostałe kąty
        for b_center in range(n_atoms):
            neighbors = adj[b_center]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, c = neighbors[i], neighbors[j]
                    angle_key = (min(a, c), b_center, max(a, c))
                    if angle_key not in processed_angles:
                        sA, sB, sC = WilsonSVectors.bend(coords[a], coords[b_center], coords[c])
                        row = np.zeros(n_atoms * 3)
                        row[a * 3:a * 3 + 3] = sA
                        row[b_center * 3:b_center * 3 + 3] = sB
                        row[c * 3:c * 3 + 3] = sC
                        b_rows.append(row)
                        labels.append(f"δ({symbols[a]}{a + 1}-{symbols[b_center]}{b_center + 1}-{symbols[c]}{c + 1})")

        if not b_rows:
            return np.zeros((0, n_atoms * 3)), []

        return np.array(b_rows), labels

    @classmethod
    def compute_ped_analysis(cls, atomnos: np.ndarray, coords: np.ndarray,
                             freqs: List[float], intens: List[float], Lx: np.ndarray,
                             masses: np.ndarray, Fx: np.ndarray,
                             mass_replacements: Optional[Dict[int, float]] = None) -> List[Dict[str, Any]]:
        """Czysty silnik wyznaczania macierzy PED Pulaya dla zadanego układu i wektorów drgań."""
        n_atoms = len(atomnos)
        n_modes = len(freqs)

        if n_atoms == 0 or n_modes == 0:
            return []

        M_diag = np.repeat(masses, 3)
        M_inv_diag = 1.0 / M_diag

        scaled_modes = []
        for k in range(n_modes):
            dk = Lx[:, k]
            mu_k = float(np.dot(dk, M_diag * dk))
            norm_factor = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
            scaled_modes.append(dk * norm_factor)

        B_SIC, labels_SIC = cls.build_sic_matrix(atomnos, coords, mass_replacements=mass_replacements)
        n_sic = len(labels_SIC)

        if n_sic == 0 or B_SIC.shape[0] == 0:
            return [{'freq': f, 'inten': it, 'summary': 'Inne / zdelokalizowane', 'full_ped': []}
                    for f, it in zip(freqs, intens)]

        G = np.dot(B_SIC * M_inv_diag, B_SIC.T)
        G_inv = np.linalg.pinv(G, rcond=1e-6)
        B_plus = np.dot(G_inv, np.dot(B_SIC, np.diag(M_inv_diag)))
        Fq = np.dot(B_plus, np.dot(Fx, B_plus.T))

        ped_results = []
        for m in range(n_modes):
            xk = scaled_modes[m]
            Lq_k = np.dot(B_SIC, xk)
            Fq_Lq = np.dot(Fq, Lq_k)
            E_k = Lq_k * Fq_Lq
            total_E = np.sum(np.abs(E_k))

            ped_percentages = (np.abs(E_k) / total_E) * 100.0 if total_E > 1e-8 else np.zeros(n_sic)
            mode_ped = [(lbl, float(pct)) for lbl, pct in zip(labels_SIC, ped_percentages) if pct >= 1.0]
            mode_ped.sort(key=lambda x: x[1], reverse=True)

            top_popup = [f"{pct:.0f}% {lbl}" for lbl, pct in mode_ped if pct >= 8.0][:3]
            summary_str = " | ".join(top_popup) if top_popup else "Inne"

            ped_results.append({
                'freq': freqs[m],
                'inten': intens[m],
                'summary': summary_str,
                'full_ped': mode_ped
            })

        return ped_results

    @classmethod
    def calculate_ped(cls, filepath: str) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], np.ndarray, List[Dict[str, Any]], str]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.out', '.orca']:
            atomnos, coords, freqs, intens, Lx = cls.parse_orca_out(filepath)
            hessian_source = "ORCA_OUT"
        else:
            atomnos, coords, freqs, intens, Lx = cls.parse_gaussian_log(filepath)
            hessian_source = "LOG"

        n_atoms = len(atomnos)
        n_modes = len(freqs)

        if n_atoms == 0 or n_modes == 0:
            return atomnos, coords, freqs, intens, Lx, [], hessian_source

        masses = np.array([ATOMIC_MASSES.get(int(num), 12.0) for num in atomnos], dtype=np.float64)
        M_diag = np.repeat(masses, 3)

        scaled_modes = []
        for k in range(n_modes):
            dk = Lx[:, k]
            mu_k = float(np.dot(dk, M_diag * dk))
            norm_factor = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
            scaled_modes.append(dk * norm_factor)

        Fx = None
        if hessian_source == "LOG":
            fchk_candidate = os.path.splitext(filepath)[0] + ".fchk"
            if os.path.exists(fchk_candidate):
                try:
                    Fx = cls.read_fchk_hessian(fchk_candidate, n_atoms)
                    hessian_source = "FCHK"
                except Exception:
                    Fx = None

        if Fx is None:
            Fx = np.zeros((n_atoms * 3, n_atoms * 3))
            for k in range(n_modes):
                xk = scaled_modes[k]
                Fx += (freqs[k] ** 2) * np.outer(M_diag * xk, M_diag * xk)

        ped_results = cls.compute_ped_analysis(atomnos, coords, freqs, intens, Lx, masses, Fx)
        return atomnos, coords, freqs, intens, Lx, ped_results, hessian_source

    @classmethod
    def identify_labile_protons(cls, atomnos: np.ndarray, coords: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Identyfikuje atomy wodoru w cząsteczce i klasyfikuje je na labilne (O-H, N-H, S-H, P-H itp.)
        oraz nielabilne (głównie C-H) na podstawie analizy sąsiedztwa i promieni kowalencyjnych.
        """
        n_atoms = len(atomnos)
        protons = {}

        for i in range(n_atoms):
            if atomnos[i] != 1:
                continue

            best_partner = -1
            min_dist = 999.0

            # Szukamy bezpośrednio związanego sąsiada
            for j in range(n_atoms):
                if i == j:
                    continue
                d = np.linalg.norm(coords[i] - coords[j])
                r_cov_sum = COVALENT_RADII.get(atomnos[i], 0.31) + COVALENT_RADII.get(atomnos[j], 0.76)
                if d <= (r_cov_sum + 0.45) and d < min_dist:
                    min_dist = d
                    best_partner = j

            if best_partner != -1:
                z_partner = atomnos[best_partner]
                sym_partner = ELEMENT_SYMBOLS.get(z_partner, f"X{z_partner}")

                # Wszystkie heteroatomy (Z != 6 i Z != 1) to protony labilne
                is_labile = (z_partner != 6 and z_partner != 1)

                if z_partner == 8:
                    group = "O–H (hydroksylowy / karboksylowy)"
                elif z_partner == 7:
                    group = "N–H (aminowy / amidowy / iminowy)"
                elif z_partner == 16:
                    group = "S–H (tiolowy)"
                elif z_partner == 15:
                    group = "P–H / P–OH (fosforanowy)"
                elif z_partner == 6:
                    group = "C–H (nielabilny)"
                else:
                    group = f"{sym_partner}–H"
            else:
                z_partner = 0
                sym_partner = "?"
                is_labile = False
                group = "H (izolowany)"
                min_dist = 0.0

            protons[i] = {
                'atom_idx': i,
                'partner_idx': best_partner,
                'partner_sym': sym_partner,
                'partner_z': z_partner,
                'bond_dist': min_dist,
                'is_labile': is_labile,
                'group_name': group
            }

        return protons

    @classmethod
    def simulate_isotopic_shift(cls, atomnos: np.ndarray, coords: np.ndarray,
                                freqs: List[float], intens: List[float], Lx: np.ndarray,
                                mass_replacements: Dict[int, float],
                                fchk_path: Optional[str] = None) -> Tuple[
        List[float], List[float], np.ndarray, str, List[Dict[str, Any]]]:
        """Przelicza mody, intensywności oraz pełną analizę PED dla wybranego izotopomeru."""
        n_atoms = len(atomnos)
        dim = 3 * n_atoms
        AU_TO_CM = 5140.487143

        masses_orig = np.array([ATOMIC_MASSES.get(int(num), 12.0) for num in atomnos], dtype=np.float64)
        masses_iso = masses_orig.copy()
        for atom_idx, new_mass in mass_replacements.items():
            if 0 <= atom_idx < n_atoms:
                masses_iso[atom_idx] = float(new_mass)

        M_orig_diag = np.repeat(masses_orig, 3)
        M_iso_diag = np.repeat(masses_iso, 3)

        Fx = None
        if fchk_path and os.path.exists(fchk_path):
            try:
                Fx = cls.read_fchk_hessian(fchk_path, n_atoms)
            except Exception:
                Fx = None

        if Fx is None:
            n_modes_in = len(freqs)
            Fx = np.zeros((dim, dim), dtype=np.float64)
            for k in range(n_modes_in):
                dk = Lx[:, k]
                mu_k = float(np.dot(dk, M_orig_diag * dk))
                norm = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
                xk = dk * norm
                omega_au = freqs[k] / AU_TO_CM
                Fx += (omega_au ** 2) * np.outer(M_orig_diag * xk, M_orig_diag * xk)

        def _diagonalize_with_tr_projection(F_mat: np.ndarray, masses_vec: np.ndarray):
            coords_bohr = coords * 1.8897261245658
            total_mass = np.sum(masses_vec)
            r_cm = np.sum(coords_bohr * masses_vec[:, np.newaxis], axis=0) / total_mass
            r_rel = coords_bohr - r_cm

            TR_vectors = np.zeros((dim, 6), dtype=np.float64)
            for i in range(n_atoms):
                sq_m = np.sqrt(masses_vec[i])
                TR_vectors[3 * i + 0, 0] = sq_m
                TR_vectors[3 * i + 1, 1] = sq_m
                TR_vectors[3 * i + 2, 2] = sq_m
                rx, ry, rz = r_rel[i]
                TR_vectors[3 * i + 0, 3] = 0.0
                TR_vectors[3 * i + 1, 3] = -rz * sq_m
                TR_vectors[3 * i + 2, 3] = ry * sq_m
                TR_vectors[3 * i + 0, 4] = rz * sq_m
                TR_vectors[3 * i + 1, 4] = 0.0
                TR_vectors[3 * i + 2, 4] = -rx * sq_m
                TR_vectors[3 * i + 0, 5] = -ry * sq_m
                TR_vectors[3 * i + 1, 5] = rx * sq_m
                TR_vectors[3 * i + 2, 5] = 0.0

            U_tr, s_tr, _ = np.linalg.svd(TR_vectors, full_matrices=False)
            n_tr = int(np.sum(s_tr > 1e-4))
            Q_tr = U_tr[:, :n_tr]
            P_vib = np.eye(dim) - (Q_tr @ Q_tr.T)

            m_diag = np.repeat(masses_vec, 3)
            m_inv_sqrt = 1.0 / np.sqrt(m_diag)

            H_mw = (F_mat * m_inv_sqrt).T * m_inv_sqrt
            H_proj = P_vib @ H_mw @ P_vib
            evals, evecs = scipy.linalg.eigh(H_proj)

            vib_evals = evals[n_tr:]
            vib_evecs_mw = evecs[:, n_tr:]
            vib_freqs = AU_TO_CM * np.sqrt(np.maximum(0.0, vib_evals))
            vib_Lx = (vib_evecs_mw.T * m_inv_sqrt).T
            return vib_freqs, vib_Lx, vib_evecs_mw

        iso_freqs, iso_Lx, iso_L_mw = _diagonalize_with_tr_projection(Fx, masses_iso)

        # Intensywności IR
        apt = None
        if fchk_path and os.path.exists(fchk_path):
            apt = cls.read_fchk_dipole_derivatives(fchk_path, n_atoms)

        if apt is not None:
            _, orig_Lx, _ = _diagonalize_with_tr_projection(Fx, masses_orig)
            dmu_orig = apt @ orig_Lx
            norm_sq_orig = np.sum(dmu_orig ** 2, axis=0)

            sum_intens_orig = np.sum(intens)
            sum_norm_orig = np.sum(norm_sq_orig)
            c_scale = (sum_intens_orig / sum_norm_orig) if (
                        sum_norm_orig > 1e-7 and sum_intens_orig > 1e-7) else 974.878

            dmu_iso = apt @ iso_Lx
            norm_sq_iso = np.sum(dmu_iso ** 2, axis=0)
            iso_intens = list(c_scale * norm_sq_iso)
            intensity_source = "APT (FCHK)"
        else:
            _, _, orig_L_mw = _diagonalize_with_tr_projection(Fx, masses_orig)
            J = orig_L_mw.T @ iso_L_mw
            J_sq = J ** 2
            col_sums = np.sum(J_sq, axis=0)
            col_sums[col_sums < 1e-8] = 1.0
            J_norm = J_sq / col_sums

            n_modes_orig = min(len(intens), orig_L_mw.shape[1])
            iso_intens = list(np.dot(np.array(intens[:n_modes_orig]), J_norm))
            intensity_source = "Duschinsky Projection (.LOG)"

        # Wyznaczenie analizy PED dla izotopomeru
        iso_ped = cls.compute_ped_analysis(
            atomnos, coords, list(iso_freqs), iso_intens, iso_Lx,
            masses_iso, Fx, mass_replacements=mass_replacements
        )

        return list(iso_freqs), iso_intens, iso_Lx, intensity_source, iso_ped