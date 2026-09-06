"""
algorithms/quantum/ped_engine.py
Analityczna analiza PED (Potential Energy Distribution) w lokalnych współrzędnych symetrii (SIC).
Wsparcie dla macierzy siłowej z .fchk i .log oraz symulacja przesunięć izotopowych.
"""

import os
import re
import logging
import numpy as np
import scipy.linalg
from typing import List, Dict, Tuple, Any, Optional

from core.logger import log_algo_exception

logger = logging.getLogger("SpectroHub")

ATOMIC_MASSES: Dict[int, float] = {
    1: 1.007825, 3: 7.016004, 5: 11.009305, 6: 12.000000, 7: 14.003074, 8: 15.994915,
    9: 18.998403, 14: 27.976927, 15: 30.973762, 16: 31.972071, 17: 34.968853,
    35: 78.918337, 53: 126.904473
}

COVALENT_RADII: Dict[int, float] = {
    1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

ELEMENT_SYMBOLS: Dict[int, str] = {
    1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


class WilsonSVectors:
    """Analityczne wektory S Wilsona ze stabilizacją numeryczną."""

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
        rBA, rBC, rBD = rA - rB, rC - rB, rD - rB
        dBA, dBC, dBD = np.linalg.norm(rBA), np.linalg.norm(rBC), np.linalg.norm(rBD)

        if dBC < 1e-6 or dBD < 1e-6 or dBA < 1e-6:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        eBC, eBD = rBC / dBC, rBD / dBD
        n = np.cross(eBC, eBD)
        norm_n = np.linalg.norm(n)
        n = n / norm_n if norm_n > 1e-5 else np.array([0.0, 0.0, 1.0])

        sA = n / dBA
        sC = np.cross(n, eBD) / dBC
        sD = np.cross(eBC, n) / dBD
        sB = -(sA + sC + sD)
        return np.nan_to_num(sA), np.nan_to_num(sB), np.nan_to_num(sC), np.nan_to_num(sD)


class PEDEngine:
    """Silnik analizy PED metodą Pulaya z macierzą Wilsona."""

    @staticmethod
    def read_fchk_hessian(fchk_path: str, n_atoms: int) -> np.ndarray:
        """Rekonstruuje pełną macierz siłową (3N x 3N) z dolnego trójkąta pliku .fchk."""
        with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start_idx = -1
        for idx, line in enumerate(lines):
            if "Cartesian Force Constants" in line:
                start_idx = idx + 1
                break

        if start_idx == -1:
            raise ValueError("Brak sekcji 'Cartesian Force Constants' w pliku .fchk")

        dim = n_atoms * 3
        expected_elements = (dim * (dim + 1)) // 2
        raw_values = []

        for line in lines[start_idx:]:
            if re.match(r'^[A-Za-z]', line.strip()):
                break
            for t in line.split():
                try:
                    raw_values.append(float(t))
                except ValueError:
                    pass
            if len(raw_values) >= expected_elements:
                break

        if len(raw_values) < expected_elements:
            raise ValueError(f"Oczekiwano {expected_elements} stałych siłowych, odczytano {len(raw_values)}")

        Fx = np.zeros((dim, dim))
        val_idx = 0
        for i in range(dim):
            for j in range(i + 1):
                Fx[i, j] = raw_values[val_idx]
                Fx[j, i] = raw_values[val_idx]
                val_idx += 1
        return Fx

    @classmethod
    def parse_gaussian_log(cls, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], np.ndarray]:
        """Ekstrahuje geometrię, częstości, intensywności oraz wektory drgań z pliku .log."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1. Odczyt geometrii
        blocks = []
        i = 0
        while i < len(lines):
            if "orientation" in lines[i].lower():
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
                            if an <= 0: an = 1  # Link atom w ONIOM
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
            raise ValueError("Nie znaleziono sekcji ze współrzędnymi atomów w pliku log.")

        # 2. Odczyt drgań normalnych
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
                                block_disps[m].append([coords_chunk[m * 3], coords_chunk[m * 3 + 1], coords_chunk[m * 3 + 2]])
                            curr_atoms.append(a_idx)
                        except (ValueError, IndexError):
                            if len(block_disps[0]) > 0: break
                    j += 1

                if len(block_disps[0]) > 0:
                    if not mode_atom_indices: mode_atom_indices = curr_atoms
                    for m in range(n_block):
                        freqs.append(block_freqs[m])
                        intens.append(block_intens[m] if m < len(block_intens) else 0.0)
                        displacements.append(block_disps[m])
                i = j
            else:
                i += 1

        if not freqs or not displacements:
            raise ValueError("Nie znaleziono wektorów drgań własnych w pliku log.")

        n_modes = len(freqs)
        n_mode_atoms = len(displacements[0])

        matching_blocks = [b for b in blocks if len(b) == n_mode_atoms]
        chosen_block = matching_blocks[-1] if matching_blocks else blocks[-1]

        atomnos = np.array([a[0] for a in chosen_block], dtype=int)
        coords = np.array([[a[1], a[2], a[3]] for a in chosen_block], dtype=float)
        displacements_np = np.array(displacements)
        Lx = displacements_np.reshape(n_modes, len(atomnos) * 3).T

        return atomnos, coords, freqs, intens, Lx

    @classmethod
    def build_sic_matrix(cls, atomnos: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Buduje macierz B Wilsona rzutując drgania na współrzędne symetrii (SIC)."""
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

        # 1. Grupy XY3 (np. -CH3, -NH3+)
        for center in range(n_atoms):
            neighbors = adj[center]
            h_neighbors = [n for n in neighbors if atomnos[n] == 1]
            if len(h_neighbors) == 3:
                h1, h2, h3 = h_neighbors
                for h in h_neighbors: processed_bonds.add(tuple(sorted((center, h))))
                processed_angles.add((min(h2, h3), center, max(h2, h3)))
                processed_angles.add((min(h1, h3), center, max(h1, h3)))
                processed_angles.add((min(h1, h2), center, max(h1, h2)))

                s_label = f"{symbols[center]}{symbols[h1]}3"
                sA1, sB1 = WilsonSVectors.stretch(coords[center], coords[h1])
                sA2, sB2 = WilsonSVectors.stretch(coords[center], coords[h2])
                sA3, sB3 = WilsonSVectors.stretch(coords[center], coords[h3])

                row_s = np.zeros(n_atoms * 3)
                row_s[center * 3 : center * 3 + 3] = (sA1 + sA2 + sA3) / np.sqrt(3)
                row_s[h1 * 3 : h1 * 3 + 3] = sB1 / np.sqrt(3)
                row_s[h2 * 3 : h2 * 3 + 3] = sB2 / np.sqrt(3)
                row_s[h3 * 3 : h3 * 3 + 3] = sB3 / np.sqrt(3)
                b_rows.append(row_s)
                labels.append(f"ν_s({s_label})")

        # 2. Grupy XY2 (np. -CH2-, -NH2)
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
                        row_s[center * 3 : center * 3 + 3] = (sA1 + sA2) / np.sqrt(2)
                        row_s[y1 * 3 : y1 * 3 + 3] = sB1 / np.sqrt(2)
                        row_s[y2 * 3 : y2 * 3 + 3] = sB2 / np.sqrt(2)
                        b_rows.append(row_s)
                        labels.append(f"ν_s({group_tag})")

                        row_as = np.zeros(n_atoms * 3)
                        row_as[center * 3 : center * 3 + 3] = (sA1 - sA2) / np.sqrt(2)
                        row_as[y1 * 3 : y1 * 3 + 3] = sB1 / np.sqrt(2)
                        row_as[y2 * 3 : y2 * 3 + 3] = -sB2 / np.sqrt(2)
                        b_rows.append(row_as)
                        labels.append(f"ν_as({group_tag})")

        # 3. Pozostałe pojedyncze wiązania
        for a, b in bonds:
            bond_key = tuple(sorted((a, b)))
            if bond_key not in processed_bonds:
                sA, sB = WilsonSVectors.stretch(coords[a], coords[b])
                row = np.zeros(n_atoms * 3)
                row[a * 3 : a * 3 + 3] = sA
                row[b * 3 : b * 3 + 3] = sB
                b_rows.append(row)
                labels.append(f"ν({symbols[a]}{a + 1}-{symbols[b]}{b + 1})")

        # 4. Pozostałe kąty walencyjne
        for b_center in range(n_atoms):
            neighbors = adj[b_center]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, c = neighbors[i], neighbors[j]
                    angle_key = (min(a, c), b_center, max(a, c))
                    if angle_key not in processed_angles:
                        sA, sB, sC = WilsonSVectors.bend(coords[a], coords[b_center], coords[c])
                        row = np.zeros(n_atoms * 3)
                        row[a * 3 : a * 3 + 3] = sA
                        row[b_center * 3 : b_center * 3 + 3] = sB
                        row[c * 3 : c * 3 + 3] = sC
                        b_rows.append(row)
                        labels.append(f"δ({symbols[a]}{a + 1}-{symbols[b_center]}{b_center + 1}-{symbols[c]}{c + 1})")

        return (np.array(b_rows), labels) if b_rows else (np.zeros((0, n_atoms * 3)), [])

    @classmethod
    def calculate_ped(cls, filepath: str) -> Tuple[List[float], List[float], List[Dict[str, Any]], str]:
        """Oblicza udziały procentowe PED dla każdego modu normalnego drgań."""
        atomnos, coords, freqs, intens, Lx = cls.parse_gaussian_log(filepath)
        n_atoms = len(atomnos)
        n_modes = len(freqs)

        if n_atoms == 0 or n_modes == 0:
            return freqs, intens, [], "LOG"

        masses = np.array([ATOMIC_MASSES.get(num, 12.0) for num in atomnos])
        M_diag = np.repeat(masses, 3)
        M_inv_diag = 1.0 / M_diag

        scaled_modes = []
        for k in range(n_modes):
            dk = Lx[:, k]
            mu_k = np.dot(dk, M_diag * dk)
            norm_factor = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
            scaled_modes.append(dk * norm_factor)

        # Próba odczytu dokładnego Hessianu z .fchk
        fchk_candidate = os.path.splitext(filepath)[0] + ".fchk"
        hessian_source = "LOG"
        Fx = None

        if os.path.exists(fchk_candidate):
            try:
                Fx = cls.read_fchk_hessian(fchk_candidate, n_atoms)
                hessian_source = "FCHK"
            except Exception as e:
                logger.warning(f"Błąd odczytu .fchk: {e} – powrót do rekonstrukcji z .log.")

        if Fx is None:
            Fx = np.zeros((n_atoms * 3, n_atoms * 3))
            for k in range(n_modes):
                xk = scaled_modes[k]
                Fx += (freqs[k] ** 2) * np.outer(M_diag * xk, M_diag * xk)

        B_SIC, labels_SIC = cls.build_sic_matrix(atomnos, coords)
        n_sic = len(labels_SIC)

        if n_sic == 0 or B_SIC.shape[0] == 0:
            return freqs, intens, [], hessian_source

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

            top_popup = [f"  • {pct:.0f}% {lbl}" for lbl, pct in mode_ped if pct >= 8.0][:4]
            summary_str = "\n".join(top_popup) if top_popup else "  • Inne / zdelokalizowane"

            ped_results.append({
                'freq': freqs[m],
                'inten': intens[m],
                'summary': summary_str,
                'full_ped': mode_ped
            })

        return freqs, intens, ped_results, hessian_source

    @classmethod
    def simulate_isotopic_shift(cls, filepath: str, mass_replacements: Dict[int, float]) -> Tuple[
        List[float], List[float], List[Dict[str, Any]], str]:
        """
        Symuluje przesunięcie izotopowe rozwiązując masowo-ważone zagadnienie własne.
        mass_replacements: {indeks_atomu_0_based: nowa_masa_amu}
        """
        atomnos, coords, freqs, intens, Lx = cls.parse_gaussian_log(filepath)
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

        fchk_candidate = os.path.splitext(filepath)[0] + ".fchk"
        Fx = None
        hessian_source = "LOG"
        conversion_factor = 1.0

        if os.path.exists(fchk_candidate):
            try:
                Fx = cls.read_fchk_hessian(fchk_candidate, n_atoms)
                hessian_source = "FCHK"
                conversion_factor = 5140.487  # Przelicznik Hartree/(Bohr^2 * amu) na cm^-1
            except Exception:
                Fx = None

        if Fx is None:
            Fx = np.zeros((n_atoms * 3, n_atoms * 3))
            for k in range(n_modes):
                dk = Lx[:, k]
                mu_k = np.dot(dk, M_orig_diag * dk)
                norm_factor = 1.0 / np.sqrt(mu_k) if mu_k > 1e-12 else 1.0
                xk = dk * norm_factor
                Fx += (freqs[k] ** 2) * np.outer(M_orig_diag * xk, M_orig_diag * xk)

        # Nowe zagadnienie własne dla izotopomeru
        H_mw_iso = (Fx * M_iso_inv_sqrt).T * M_iso_inv_sqrt
        eigenvalues, eigenvectors = scipy.linalg.eigh(H_mw_iso)

        positive_mask = eigenvalues > 1e-5
        iso_freqs_raw = np.sqrt(eigenvalues[positive_mask]) * conversion_factor
        Lx_iso = eigenvectors[:, positive_mask]

        B_SIC, labels_SIC = cls.build_sic_matrix(atomnos, coords)
        n_sic = len(labels_SIC)

        G_iso = np.dot(B_SIC * (1.0 / M_iso_diag), B_SIC.T)
        G_iso_inv = np.linalg.pinv(G_iso, rcond=1e-6)
        B_plus_iso = np.dot(G_iso_inv, np.dot(B_SIC, np.diag(1.0 / M_iso_diag)))
        Fq_iso = np.dot(B_plus_iso, np.dot(Fx, B_plus_iso.T))

        ped_results = []
        n_iso_modes = len(iso_freqs_raw)

        for m in range(n_iso_modes):
            xk = Lx_iso[:, m]
            Lq_k = np.dot(B_SIC, xk)
            Fq_Lq = np.dot(Fq_iso, Lq_k)
            E_k = Lq_k * Fq_Lq
            total_E = np.sum(np.abs(E_k))

            ped_percentages = (np.abs(E_k) / total_E) * 100.0 if total_E > 1e-8 else np.zeros(n_sic)
            mode_ped = [(lbl, float(pct)) for lbl, pct in zip(labels_SIC, ped_percentages) if pct >= 1.0]
            mode_ped.sort(key=lambda x: x[1], reverse=True)

            top_popup = [f"  • {pct:.0f}% {lbl}" for lbl, pct in mode_ped if pct >= 8.0][:4]
            summary_str = "\n".join(top_popup) if top_popup else "  • Inne / zdelokalizowane"
            approx_inten = intens[m] if m < len(intens) else 0.0

            ped_results.append({
                'freq': float(iso_freqs_raw[m]),
                'inten': approx_inten,
                'summary': summary_str,
                'full_ped': mode_ped
            })

        return list(iso_freqs_raw), intens[:n_iso_modes], ped_results, f"{hessian_source}_ISO"