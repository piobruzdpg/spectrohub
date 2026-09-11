"""
io_hub/project_io.py
Silnik zapisu i odczytu stanu projektu SpectroHub w kontenerowym formacie .shub (ZIP Bundle).
Paczka zawiera:
- project.json: manifest hierarchii, folderów, okien roboczych, potoku DAG i metadanych
- arrays.npz: zoptymalizowany binarny zrzut tablic NumPy (float64) bez utraty precyzji
"""

import os
import io
import json
import zipfile
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.session import ProjectSession
from core.models import SpectrumData, SpectralSeries, GaussianData, ProjectMetadata, WindowFolder

logger = logging.getLogger("SpectroHub")

PROJECT_FORMAT_VERSION = "1.1.0"


class ProjectBundleIO:
    """Zarządza serializacją i deserializacją całego środowiska roboczego SpectroHub."""

    @staticmethod
    def save_project(file_path: str, session: ProjectSession,
                     dag_state: Optional[Dict[str, Any]] = None,
                     linear_pipeline: Optional[list] = None) -> bool:
        """Zapisuje całościowy stan aplikacji do pliku .shub."""
        try:
            arrays_dict: Dict[str, np.ndarray] = {}
            manifest: Dict[str, Any] = {
                "format_version": PROJECT_FORMAT_VERSION,
                "metadata": session.metadata.to_dict(),
                "active_context": session.active_context,
                "active_window_id": session.active_window_id,
                "active_spectrum_name": session.active_spectrum.name if session.active_spectrum else None,
                "dag_state": dag_state or {},
                "linear_pipeline": linear_pipeline or [],
                "storage": {
                    "experimental": [],
                    "atm": [],
                    "series": []
                },
                "folders": [f.to_dict() for f in session.folders.values()],
                "windows": []
            }

            # 1. Serializacja widm eksperymentalnych Magazynu
            for idx, (name, spec) in enumerate(session.spectra.items()):
                k_x = f"exp_x_{idx}"
                k_y = f"exp_y_{idx}"
                arrays_dict[k_x] = spec.wavenumbers
                arrays_dict[k_y] = spec.intensities

                item_dict = {
                    "id": idx,
                    "name": spec.name,
                    "file_path": spec.file_path,
                    "file_type": spec.file_type,
                    "time_str": spec.time_str,
                    "units_x": spec.units_x,
                    "units_y": spec.units_y,
                    "is_visible": session.is_visible(name),
                    "metadata": spec.metadata,
                    "is_gaussian": isinstance(spec, GaussianData)
                }

                if isinstance(spec, GaussianData):
                    arrays_dict[f"exp_stk_f_{idx}"] = spec.stick_freqs
                    arrays_dict[f"exp_stk_i_{idx}"] = spec.stick_intens
                    arrays_dict[f"exp_coords_{idx}"] = spec.coords
                    arrays_dict[f"exp_atomnos_{idx}"] = spec.atomnos
                    arrays_dict[f"exp_lx_{idx}"] = spec.Lx

                    item_dict["dft_data"] = {
                        "hessian_source": spec.hessian_source,
                        "scale_factor": spec.scale_factor,
                        "ped_results": spec.ped_results
                    }

                manifest["storage"]["experimental"].append(item_dict)

            # 2. Serializacja wzorców atmosfery Magazynu
            for idx, (name, spec) in enumerate(session.atm_spectra.items()):
                k_x = f"atm_x_{idx}"
                k_y = f"atm_y_{idx}"
                arrays_dict[k_x] = spec.wavenumbers
                arrays_dict[k_y] = spec.intensities

                manifest["storage"]["atm"].append({
                    "id": idx,
                    "name": spec.name,
                    "file_path": spec.file_path,
                    "file_type": spec.file_type,
                    "time_str": spec.time_str,
                    "units_x": spec.units_x,
                    "units_y": spec.units_y,
                    "is_visible": session.is_visible(name),
                    "metadata": spec.metadata
                })

            # 3. Serializacja serii 2D i tensorów 3D
            for idx, (name, s_obj) in enumerate(session.series.items()):
                arrays_dict[f"series_wn_{idx}"] = s_obj.wavenumbers
                arrays_dict[f"series_data_{idx}"] = s_obj.data

                ser_dict = {
                    "id": idx,
                    "name": s_obj.name,
                    "axis_n_name": s_obj.axis_n_name,
                    "axis_m_name": s_obj.axis_m_name,
                    "is_visible": session._series_visibility.get(name, True),
                    "metadata": s_obj.metadata
                }

                if s_obj.axis_n_values is not None:
                    arrays_dict[f"series_n_vals_{idx}"] = s_obj.axis_n_values
                    ser_dict["has_axis_n"] = True

                if s_obj.axis_m_values is not None:
                    arrays_dict[f"series_m_vals_{idx}"] = s_obj.axis_m_values
                    ser_dict["has_axis_m"] = True

                manifest["storage"]["series"].append(ser_dict)

            # 4. Serializacja Okien Roboczych
            for win_id, win in session.windows.items():
                win_dict = {
                    "window_id": win.window_id,
                    "name": win.name,
                    "folder_id": win.folder_id,
                    "active_spectrum_name": win.active_spectrum_name,
                    "spectra": []
                }
                for s_idx, (s_name, spec) in enumerate(win.spectra.items()):
                    k_x = f"win_{win_id}_x_{s_idx}"
                    k_y = f"win_{win_id}_y_{s_idx}"
                    arrays_dict[k_x] = spec.wavenumbers
                    arrays_dict[k_y] = spec.intensities

                    win_dict["spectra"].append({
                        "id": s_idx,
                        "name": spec.name,
                        "file_path": spec.file_path,
                        "file_type": spec.file_type,
                        "time_str": spec.time_str,
                        "units_x": spec.units_x,
                        "units_y": spec.units_y,
                        "is_visible": win.visibility.get(s_name, True),
                        "metadata": spec.metadata
                    })
                manifest["windows"].append(win_dict)

            # 5. Bezpośredni zapis do archiwum ZIP
            with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_f:
                manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
                zip_f.writestr("project.json", manifest_bytes)

                buf_npz = io.BytesIO()
                np.savez_compressed(buf_npz, **arrays_dict)
                buf_npz.seek(0)
                zip_f.writestr("data.npz", buf_npz.read())

            session.project_file_path = file_path
            logger.info(f"Pomyślnie zapisano projekt SpectroHub do: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Błąd zapisu pliku projektu {file_path}: {e}")
            raise e

    @staticmethod
    def load_project(file_path: str, session: ProjectSession) -> Tuple[Dict[str, Any], list]:
        """
        Odtwarza kompletny stan programu z pliku .shub.
        Zwraca tuple: (dag_state, linear_pipeline).
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zip_f:
                if "project.json" not in zip_f.namelist() or "data.npz" not in zip_f.namelist():
                    raise ValueError("Wybrany plik nie jest poprawnym archiwum projektu SpectroHub (.shub)")

                manifest_data = json.loads(zip_f.read("project.json").decode("utf-8"))
                npz_data = np.load(io.BytesIO(zip_f.read("data.npz")))

            # 1. Czyszczenie obecnej sesji
            session.clear_all()

            # 2. Odtworzenie metadanych projektu
            session.metadata = ProjectMetadata.from_dict(manifest_data.get("metadata", {}))
            session.project_file_path = file_path
            session.metadata_changed.emit(session.metadata)

            # 3. Odtworzenie widm eksperymentalnych Magazynu
            for itm in manifest_data["storage"]["experimental"]:
                idx = itm["id"]
                xw = npz_data[f"exp_x_{idx}"]
                yw = npz_data[f"exp_y_{idx}"]

                if itm.get("is_gaussian", False):
                    dft = itm.get("dft_data", {})
                    spec = GaussianData(
                        wavenumbers=xw,
                        intensities=yw,
                        name=itm["name"],
                        file_path=itm["file_path"],
                        file_type=itm["file_type"],
                        time_str=itm.get("time_str", "00:00:00"),
                        units_x=itm.get("units_x", "cm^-1"),
                        units_y=itm.get("units_y", "absorbance"),
                        metadata=itm.get("metadata", {}),
                        stick_freqs=npz_data.get(f"exp_stk_f_{idx}", np.array([])),
                        stick_intens=npz_data.get(f"exp_stk_i_{idx}", np.array([])),
                        coords=npz_data.get(f"exp_coords_{idx}", np.empty((0, 3))),
                        atomnos=npz_data.get(f"exp_atomnos_{idx}", np.array([])),
                        Lx=npz_data.get(f"exp_lx_{idx}", np.empty((0, 0))),
                        hessian_source=dft.get("hessian_source", "LOG"),
                        scale_factor=dft.get("scale_factor", 1.0),
                        ped_results=dft.get("ped_results", [])
                    )
                else:
                    spec = SpectrumData(
                        wavenumbers=xw,
                        intensities=yw,
                        name=itm["name"],
                        file_path=itm["file_path"],
                        file_type=itm["file_type"],
                        time_str=itm.get("time_str", "00:00:00"),
                        units_x=itm.get("units_x", "cm^-1"),
                        units_y=itm.get("units_y", "absorbance"),
                        metadata=itm.get("metadata", {})
                    )

                session.add_spectrum(spec, is_atm=False, make_active=False)
                session.set_visibility(itm["name"], itm.get("is_visible", True))

            # 4. Odtworzenie wzorców atmosfery
            for itm in manifest_data["storage"]["atm"]:
                idx = itm["id"]
                xw = npz_data[f"atm_x_{idx}"]
                yw = npz_data[f"atm_y_{idx}"]

                spec = SpectrumData(
                    wavenumbers=xw,
                    intensities=yw,
                    name=itm["name"],
                    file_path=itm["file_path"],
                    file_type=itm["file_type"],
                    time_str=itm.get("time_str", "00:00:00"),
                    metadata=itm.get("metadata", {})
                )
                session.add_spectrum(spec, is_atm=True, make_active=False)
                session.set_visibility(itm["name"], itm.get("is_visible", True))

            # 5. Odtworzenie serii 2D / 3D
            for itm in manifest_data["storage"]["series"]:
                idx = itm["id"]
                wn = npz_data[f"series_wn_{idx}"]
                data = npz_data[f"series_data_{idx}"]

                n_vals = npz_data[f"series_n_vals_{idx}"] if itm.get("has_axis_n") else None
                m_vals = npz_data[f"series_m_vals_{idx}"] if itm.get("has_axis_m") else None

                s_obj = SpectralSeries(
                    name=itm["name"],
                    wavenumbers=wn,
                    data=data,
                    axis_n_name=itm.get("axis_n_name", "Czas"),
                    axis_n_values=n_vals,
                    axis_m_name=itm.get("axis_m_name", "Stężenie"),
                    axis_m_values=m_vals,
                    metadata=itm.get("metadata", {})
                )
                session.add_series(s_obj, make_active=False)
                session.set_series_visibility(itm["name"], itm.get("is_visible", True))

            # 6. Odtworzenie folderów okien roboczych
            for f_data in manifest_data.get("folders", []):
                f_obj = WindowFolder.from_dict(f_data)
                session._folders[f_obj.folder_id] = f_obj
                session.folder_added.emit(f_obj.folder_id, f_obj.name, f_obj.parent_folder_id)

            # 7. Odtworzenie Okien Roboczych
            if manifest_data.get("windows"):
                for w_id in list(session.windows.keys()):
                    session.remove_window(w_id)

                for w_info in manifest_data["windows"]:
                    w_id = session.add_window(
                        name=w_info["name"],
                        window_id=w_info["window_id"],
                        folder_id=w_info.get("folder_id")
                    )
                    for s_info in w_info["spectra"]:
                        s_idx = s_info["id"]
                        xw = npz_data[f"win_{w_id}_x_{s_idx}"]
                        yw = npz_data[f"win_{w_id}_y_{s_idx}"]

                        spec = SpectrumData(
                            wavenumbers=xw,
                            intensities=yw,
                            name=s_info["name"],
                            file_path=s_info["file_path"],
                            file_type=s_info["file_type"],
                            time_str=s_info.get("time_str", "00:00:00"),
                            metadata=s_info.get("metadata", {})
                        )
                        win_spec = session.add_spectrum_to_window(w_id, spec)
                        session.set_window_spectrum_visibility(w_id, win_spec.name, s_info.get("is_visible", True))

                    if w_info.get("active_spectrum_name"):
                        session.set_window_active_spectrum(w_id, w_info["active_spectrum_name"])

            # 8. Przywrócenie aktywnego kontekstu i aktywnego widma
            act_ctx = manifest_data.get("active_context", "storage")
            session.set_context(act_ctx)

            if act_ctx == "window" and manifest_data.get("active_window_id"):
                session.set_active_window(manifest_data["active_window_id"])
            elif manifest_data.get("active_spectrum_name"):
                session.set_active_spectrum(manifest_data["active_spectrum_name"])

            dag_state = manifest_data.get("dag_state", {})
            linear_pipeline = manifest_data.get("linear_pipeline", [])

            logger.info(f"Pomyślnie wczytano projekt SpectroHub z: {file_path}")
            return dag_state, linear_pipeline

        except Exception as e:
            logger.error(f"Nie udało się załadować projektu {file_path}: {e}")
            raise e