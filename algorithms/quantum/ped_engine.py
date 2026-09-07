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
    def build_sic_matrix(cls, atomnos: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        n_atoms = len(atomnos)
        symbols = [ELEMENT_SYMBOLS.get(num, f"X{num}") for num in atomnos]

        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_cov = COVALENT_RADII.get(atomnos[i], 0.8) + COVALENT_RADII.get(atomnos[j], 0.8)
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

        # Grupy XY3 (np. -CH3)
        for center in range(n_atoms):
            neighbors = adj[center]
            h_neighbors = [n for n in neighbors if atomnos[n] == 1]
            if len(h_neighbors) == 3:
                h1, h2, h3 = h_neighbors
                for h in h_neighbors:
                    processed_bonds.add(tuple(sorted((center, h))))
                processed_angles.add((min(h2, h3), center, max(h2, h3)))
                processed_angles.add((min(h1, h3), center, max(h1, h3)))
                processed_angles.add((min(h1, h2), center, max(h1, h2)))

                s_label = f"{symbols[center]}{symbols[h1]}3"
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

        # Grupy XY2 (np. -CH2-, -NH2)
        for center in range(n_atoms):
            neighbors = adj[center]
            for i_idx in range(len(neighbors)):
                for j_idx in range(i_idx + 1, len(neighbors)):
                    y1, y2 = neighbors[i_idx], neighbors[j_idx]
                    if atomnos[y1] == atomnos[y2] and (tuple(sorted((center, y1))) not in processed_bonds):
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

        # Pozostałe wiązania
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
    def calculate_ped(cls, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray, List[Dict[str, Any]], str]:
        """Uniwersalna metoda wyznaczająca PED dla Gaussiana (.log) oraz ORCA (.out)."""
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

        masses = np.array([ATOMIC_MASSES.get(num, 12.0) for num in atomnos])
        M_diag = np.repeat(masses, 3)
        M_inv_diag = 1.0 / M_diag

        scaled_modes = []
        for k in range(n_modes):
            dk = Lx[:, k]
            mu_k = np.dot(dk, M_diag * dk)
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

        B_SIC, labels_SIC = cls.build_sic_matrix(atomnos, coords)
        n_sic = len(labels_SIC)

        if n_sic == 0 or B_SIC.shape[0] == 0:
            ped_fallback = [{'freq': f, 'inten': it, 'summary': 'Inne / zdelokalizowane', 'full_ped': []}
                            for f, it in zip(freqs, intens)]
            return atomnos, coords, freqs, intens, Lx, ped_fallback, hessian_source

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

        return atomnos, coords, freqs, intens, Lx, ped_results, hessian_source

    @classmethod
    def simulate_isotopic_shift(cls, atomnos: np.ndarray, coords: np.ndarray,
                                freqs: List[float], intens: List[float], Lx: np.ndarray,
                                mass_replacements: Dict[int, float]) -> Tuple[List[float], List[float], np.ndarray]:
        """Szybkie przeliczenie zagadnienia własnego (H->D) on-the-fly w ułamku sekundy."""
        n_atoms = len(atomnos)
        n_modes = len(freqs)
        masses_orig = np.array([ATOMIC_MASSES.get(num, 12.0) for num in atomnos])
        masses_iso = masses_orig.copy()

        for atom_idx, new_mass in mass_replacements.items():
            if 0 <= atom_idx < n_atoms:
                masses_iso[atom_idx] = new_mass

        M_orig_diag = np.repeat(masses_orig, 3)
        M_iso_diag = np.repeat(masses_iso, 3)
        M_iso_inv_sqrt = 1.0 / np.sqrt(M_iso_diag)

        # Rekonstrukcja Fx
        Fx = np.zeros((n_atoms * 3, n_atoms * 3))
        for k in range(n_modes):
            dk = Lx[:, k]
            mu_k = np.dot(dk, M_orig_diag * dk)
            norm = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
            xk = dk * norm
            Fx += (freqs[k] ** 2) * np.outer(M_orig_diag * xk, M_orig_diag * xk)

        H_mw_iso = (Fx * M_iso_inv_sqrt).T * M_iso_inv_sqrt
        eigenvalues, eigenvectors = scipy.linalg.eigh(H_mw_iso)

        pos = eigenvalues > 1e-4
        iso_freqs = list(np.sqrt(eigenvalues[pos]))
        iso_Lx = eigenvectors[:, pos]

        return iso_freqs, intens[:len(iso_freqs)], iso_Lx