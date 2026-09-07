"""
io_hub/exporters/excel_exporter.py
Wieloarkuszowy generator raportów Excel (.xlsx) oparty na pandas i openpyxl.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from core.models import SpectrumData, GaussianData

logger = logging.getLogger("SpectroHub")


class ExcelProjectExporter:
    """Eksportuje stan projektu lub wybrane widma do wieloarkuszowego pliku Excela."""

    @staticmethod
    def export(file_path: str,
               spectra: List[SpectrumData],
               deconv_peaks: Optional[List[Dict[str, Any]]] = None) -> None:
        if not spectra:
            raise ValueError("Brak widm do wyeksportowania.")

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 1. Arkusz: Matryca danych widmowych (Liczby falowe malejąco)
            dict_data = {}
            for s in spectra:
                s_series = pd.Series(data=s.intensities, index=s.wavenumbers)
                s_series = s_series[~s_series.index.duplicated(keep='first')]
                dict_data[s.name] = s_series

            df_spectra = pd.DataFrame(dict_data)
            df_spectra.index.name = "Liczba_falowa [cm-1]"
            df_spectra.sort_index(ascending=False).to_excel(writer, sheet_name="Widma_Dane")

            # 2. Arkusz: Podsumowanie i Metadane
            meta_rows = []
            for s in spectra:
                row = {
                    "Nazwa Widma": s.name,
                    "Typ Pliku": s.file_type.upper(),
                    "Ścieżka Oryginalna": s.file_path,
                    "Liczba Punktów": len(s.wavenumbers),
                    "ν Min [cm-1]": float(np.min(s.wavenumbers)),
                    "ν Max [cm-1]": float(np.max(s.wavenumbers)),
                    "Jednostka X": s.units_x,
                    "Jednostka Y": s.units_y
                }
                if isinstance(s, GaussianData):
                    row["Źródło Hessianu"] = s.hessian_source
                    row["Wsp. Skalowania (s)"] = s.scale_factor
                    row["Liczba Modów"] = len(s.stick_freqs)
                meta_rows.append(row)

            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Metadane_Projektu", index=False)

            # 3. Arkusz: Piki dekonwolucji (jeśli przekazano)
            if deconv_peaks:
                peaks_rows = []
                valid_amide = sum(
                    p.get('area', 0.0) for p in deconv_peaks
                    if 1600 <= p.get('center', 0.0) <= 1700 and p.get('structure', '') not in ["Side Chains", "Other", "Inny"]
                )
                for idx, p in enumerate(deconv_peaks, 1):
                    c = p.get('center', 0.0)
                    area = p.get('area', 0.0)
                    fwhm = p.get('sigma', 10.0) * (2.0 if p.get('type') == 'Lorentzian' else 2.35482)
                    perc = (area / valid_amide * 100.0) if (valid_amide > 0 and 1600 <= c <= 1700) else 0.0

                    peaks_rows.append({
                        "Pik": p.get('id', f"p{idx}"),
                        "Kształt": p.get('type', 'Gaussian'),
                        "Pozycja [cm-1]": round(c, 2),
                        "FWHM [cm-1]": round(fwhm, 2),
                        "Pole Powierzchni": round(area, 4),
                        "Udział w Amidzie I [%]": round(perc, 1) if perc > 0 else "—",
                        "Przypisanie": p.get('structure', 'Other'),
                        "Lock Center": p.get('lock_center', False),
                        "Lock FWHM": p.get('lock_sigma', False),
                        "Lock Pole": p.get('lock_area', False)
                    })
                pd.DataFrame(peaks_rows).to_excel(writer, sheet_name="Dekonwolucja_Piki", index=False)

            # 4. Arkusz: Drgania DFT & Analiza PED
            dft_rows = []
            for s in spectra:
                if isinstance(s, GaussianData) and len(s.stick_freqs) > 0:
                    scaled = s.stick_freqs * s.scale_factor
                    for idx, (f_raw, f_sc, it) in enumerate(zip(s.stick_freqs, scaled, s.stick_intens)):
                        ped_text = "—"
                        if s.ped_results and idx < len(s.ped_results):
                            full_ped = s.ped_results[idx].get('full_ped', [])
                            if full_ped:
                                ped_text = " + ".join([f"{pct:.0f}% {lbl}" for lbl, pct in full_ped])
                            else:
                                ped_text = s.ped_results[idx].get('summary', 'Inne')

                        dft_rows.append({
                            "Widmo DFT": s.name,
                            "Mod": idx + 1,
                            "Częstość Surowa [cm-1]": round(f_raw, 2),
                            "Częstość Skalowana [cm-1]": round(f_sc, 2),
                            "Intensywność IR [km/mol]": round(it, 2),
                            "Źródło Hessianu": s.hessian_source,
                            "Analiza PED Pulaya": ped_text
                        })

            if dft_rows:
                pd.DataFrame(dft_rows).to_excel(writer, sheet_name="DFT_Drgania_PED", index=False)

        logger.info(f"Pomyślnie wyeksportowano raport Excel: '{file_path}' ({len(spectra)} widm)")