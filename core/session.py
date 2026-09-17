"""
core/session.py
Centralny menedżer stanu danych (Single Source of Truth) dla SpectroHub.
Obsługuje widma, serie 2D, foldery, okna robocze oraz tabele danych (DataTable).
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import datetime
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack
import logging

from core.models import SpectrumData, SpectralSeries, ProjectMetadata, WindowFolder, WorkspaceWindow, DataTable

logger = logging.getLogger("SpectroHub")


def is_atmospheric_name(name: str) -> bool:
    """Rozpoznaje wzorce atmosfery wyłącznie na podstawie fraz atm oraz co2."""
    nl = name.lower()
    return "atm" in nl or "co2" in nl


class ProjectSession(QObject):
    # Sygnały Magazynu - Pomiary
    spectrum_added = Signal(object)
    spectra_batch_added = Signal(list)
    spectrum_removed = Signal(str)
    active_spectrum_changed = Signal(object)
    visibility_changed = Signal(str, bool)
    session_cleared = Signal()

    # Sygnały Magazynu - Wzorce Atmosfery
    atm_spectrum_added = Signal(object)
    atm_spectra_batch_added = Signal(list)
    atm_spectrum_removed = Signal(str)
    atm_visibility_changed = Signal(str, bool)

    # Sygnały Mutacji Danych
    spectrum_data_changed = Signal(object)
    batch_spectra_changed = Signal(list)

    # Sygnały Serii 2D / 3D
    series_added = Signal(object)
    series_removed = Signal(str)
    series_visibility_changed = Signal(str, bool)
    active_series_changed = Signal(object)
    series_data_changed = Signal(object)

    # Sygnały Okien Roboczych i Kontekstu
    window_added = Signal(str, str)
    window_removed = Signal(str)
    active_window_changed = Signal(str)
    window_spectrum_added = Signal(str, object)
    window_spectra_batch_added = Signal(str, list)  # <--- NOWY SYGNAŁ HURTOWY
    window_spectrum_removed = Signal(str, str)
    window_visibility_changed = Signal(str, str, bool)
    context_changed = Signal(str)            # 'storage' | 'window' | 'table'
    storage_branch_changed = Signal(str)     # 'experimental' | 'atm' | 'dft' | 'series'

    # Sygnały Folderów Okien Roboczych
    folder_added = Signal(str, str, object)       # folder_id, name, parent_folder_id
    folder_removed = Signal(str)                  # folder_id
    folder_renamed = Signal(str, str)             # folder_id, new_name
    window_folder_changed = Signal(str, object)   # window_id, new_folder_id

    # Sygnały Tabel Danych (DataTable)
    table_added = Signal(object)                  # DataTable
    table_removed = Signal(str)                   # table_id
    table_renamed = Signal(str, str)              # table_id, new_name
    table_folder_changed = Signal(str, object)    # table_id, new_folder_id
    active_table_changed = Signal(object)         # DataTable

    # Sygnał Metadanych Projektu
    metadata_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.metadata = ProjectMetadata()
        self.project_file_path: Optional[str] = None
        self.dag_state: Dict[str, Any] = {}

        # Magazyn danych (Pomiary próbek)
        self._spectra: Dict[str, SpectrumData] = {}
        self._visibility: Dict[str, bool] = {}

        # Dedykowany magazyn wzorców atmosfery (H2O vapor / CO2)
        self._atm_spectra: Dict[str, SpectrumData] = {}
        self._atm_visibility: Dict[str, bool] = {}

        self._active_spectrum_name: Optional[str] = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(15)  # <--- Bezpieczny bufor zapobiegający wyciekom pamięci

        self._series: Dict[str, SpectralSeries] = {}
        self._series_visibility: Dict[str, bool] = {}
        self._active_series_name: Optional[str] = None

        # Foldery okien roboczych i tabel
        self._folders: Dict[str, WindowFolder] = {}
        self._folder_counter = 0

        # Okna robocze (startujemy z czystym słownikiem, bez wymuszonego Okna 1)
        self._windows: Dict[str, WorkspaceWindow] = {}
        self._active_window_id: Optional[str] = None
        self._window_counter = 0

        # Tabele danych liczbowych (DataTable)
        self._tables: Dict[str, DataTable] = {}
        self._active_table_id: Optional[str] = None
        self._table_counter = 0

        # Kontekst globalny i gałąź magazynu
        self._active_context: str = "storage"
        self._active_storage_branch: str = "experimental"

    # --- METADANE PROJEKTU ---
    def update_metadata(self, title: Optional[str] = None, author: Optional[str] = None,
                        variant: Optional[str] = None, description: Optional[str] = None,
                        notes: Optional[str] = None):
        if title is not None: self.metadata.title = title
        if author is not None: self.metadata.author = author
        if variant is not None: self.metadata.variant = variant
        if description is not None: self.metadata.description = description
        if notes is not None: self.metadata.notes = notes

        self.metadata.modified_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata_changed.emit(self.metadata)

    # --- ZARZĄDZANIE FOLDERAMI ---
    @property
    def folders(self) -> Dict[str, WindowFolder]:
        return self._folders

    def get_folder(self, folder_id: str) -> Optional[WindowFolder]:
        return self._folders.get(folder_id)

    def add_folder(self, name: Optional[str] = None, parent_folder_id: Optional[str] = None,
                   folder_id: Optional[str] = None) -> str:
        self._folder_counter += 1
        fid = folder_id or f"folder_{self._folder_counter}"
        fname = name or f"Folder {self._folder_counter}"

        folder = WindowFolder(folder_id=fid, name=fname, parent_folder_id=parent_folder_id)
        self._folders[fid] = folder
        logger.info(f"Utworzono folder: '{fname}' (ID: {fid})")
        self.folder_added.emit(fid, fname, parent_folder_id)
        return fid

    def remove_folder(self, folder_id: str, delete_windows: bool = False):
        if folder_id not in self._folders:
            return

        target_parent = self._folders[folder_id].parent_folder_id

        # 1. Okna w tym folderze
        affected_windows = [w_id for w_id, win in self._windows.items() if win.folder_id == folder_id]
        for w_id in affected_windows:
            if delete_windows:
                self.remove_window(w_id)
            else:
                self.move_window_to_folder(w_id, target_parent)

        # 2. Tabele w tym folderze
        affected_tables = [t_id for t_id, tbl in self._tables.items() if tbl.folder_id == folder_id]
        for t_id in affected_tables:
            if delete_windows:
                self.remove_table(t_id)
            else:
                self.move_table_to_folder(t_id, target_parent)

        # 3. Podfoldery (przepięcie wyżej)
        child_folders = [fid for fid, f in self._folders.items() if f.parent_folder_id == folder_id]
        for c_fid in child_folders:
            self._folders[c_fid].parent_folder_id = target_parent

        del self._folders[folder_id]
        logger.info(f"Usunięto folder: ID {folder_id} (delete_windows={delete_windows})")
        self.folder_removed.emit(folder_id)

    def rename_folder(self, folder_id: str, new_name: str):
        if folder_id in self._folders and new_name.strip():
            clean_name = new_name.strip()
            self._folders[folder_id].name = clean_name
            self.folder_renamed.emit(folder_id, clean_name)

    def move_window_to_folder(self, window_id: str, folder_id: Optional[str]):
        if window_id in self._windows:
            if folder_id is not None and folder_id not in self._folders:
                folder_id = None
            self._windows[window_id].folder_id = folder_id
            self.window_folder_changed.emit(window_id, folder_id)

    # --- ZARZĄDZANIE TABELAMI DANYCH (DATATABLE) ---
    @property
    def tables(self) -> Dict[str, DataTable]:
        return self._tables

    @property
    def active_table(self) -> Optional[DataTable]:
        if self._active_table_id and self._active_table_id in self._tables:
            return self._tables[self._active_table_id]
        return None

    def get_table(self, table_id: str) -> Optional[DataTable]:
        return self._tables.get(table_id)

    def add_table(self, table: DataTable, make_active: bool = True) -> str:
        if not table.table_id:
            self._table_counter += 1
            table.table_id = f"tbl_{self._table_counter}"

        self._tables[table.table_id] = table
        logger.info(f"Dodano tabelę danych: '{table.name}' (ID: {table.table_id}, Folder: {table.folder_id})")
        self.table_added.emit(table)

        if make_active:
            self.set_active_table(table.table_id)
        return table.table_id

    def remove_table(self, table_id: str):
        if table_id in self._tables:
            del self._tables[table_id]
            logger.info(f"Usunięto tabelę danych: ID {table_id}")
            self.table_removed.emit(table_id)

            if self._active_table_id == table_id:
                self._active_table_id = next(iter(self._tables.keys()), None)
                self.active_table_changed.emit(self.active_table)

    def rename_table(self, table_id: str, new_name: str):
        if table_id in self._tables and new_name.strip():
            clean_name = new_name.strip()
            self._tables[table_id].name = clean_name
            self.table_renamed.emit(table_id, clean_name)

    def move_table_to_folder(self, table_id: str, folder_id: Optional[str]):
        if table_id in self._tables:
            if folder_id is not None and folder_id not in self._folders:
                folder_id = None
            self._tables[table_id].folder_id = folder_id
            self.table_folder_changed.emit(table_id, folder_id)

    def set_active_table(self, table_id: Optional[str]):
        if table_id is not None and table_id not in self._tables:
            return

        self._active_table_id = table_id
        self._active_context = "table"
        logger.debug(f"Aktywna tabela: {table_id}")
        self.context_changed.emit("table")
        self.active_table_changed.emit(self.active_table)

    # --- ZARZĄDZANIE OKNAMI ROBOCZYMI ---
    def add_spectra_batch_to_window(self, window_id: str, spectra_list: List[SpectrumData]) -> List[SpectrumData]:
        """Hurtowo dodaje całą listę widm do okna roboczego z pojedynczą emisją sygnału."""
        if window_id not in self._windows or not spectra_list:
            return []

        win = self._windows[window_id]
        added = []
        existing_names = set(win.spectra.keys())

        for spectrum in spectra_list:
            cloned_spec = spectrum.clone()
            base_name = cloned_spec.name
            unique_name = base_name
            counter = 1
            while unique_name in existing_names:
                unique_name = f"{base_name}_{counter}"
                counter += 1

            existing_names.add(unique_name)
            cloned_spec.name = unique_name
            win.spectra[unique_name] = cloned_spec
            win.visibility[unique_name] = True
            added.append(cloned_spec)

        if added:
            win.active_spectrum_name = added[-1].name
            self.window_spectra_batch_added.emit(window_id, added)

        return added

    @property
    def windows(self) -> Dict[str, WorkspaceWindow]:
        return self._windows

    @property
    def active_window(self) -> Optional[WorkspaceWindow]:
        if self._active_window_id and self._active_window_id in self._windows:
            return self._windows[self._active_window_id]
        return None

    @property
    def active_window_id(self) -> Optional[str]:
        return self._active_window_id

    def add_window(self, name: Optional[str] = None, window_id: Optional[str] = None,
                   folder_id: Optional[str] = None) -> str:
        self._window_counter += 1
        win_id = window_id or f"win_{self._window_counter}"
        win_name = name or f"Okno {self._window_counter}"

        win = WorkspaceWindow(window_id=win_id, name=win_name, folder_id=folder_id)
        self._windows[win_id] = win
        logger.info(f"Utworzono okno robocze: '{win_name}' (ID: {win_id}, Folder: {folder_id})")
        self.window_added.emit(win_id, win_name)

        if self._active_window_id is None:
            self.set_active_window(win_id)
        return win_id

    def remove_window(self, window_id: str):
        if window_id in self._windows:
            del self._windows[window_id]
            logger.info(f"Usunięto okno robocze: ID {window_id}")
            self.window_removed.emit(window_id)

            if self._active_window_id == window_id:
                new_active = next(iter(self._windows.keys()), None)
                self.set_active_window(new_active)
                # Jeśli usunięto ostatnie okno, wracamy do kontekstu Magazynu Danych
                if new_active is None:
                    self.set_context("storage")

    def set_active_window(self, window_id: Optional[str]):
        if window_id is not None and window_id not in self._windows:
            return

        self._active_window_id = window_id
        self._active_context = "window"

        if window_id and window_id in self._windows:
            win = self._windows[window_id]
            if (not win.active_spectrum_name or win.active_spectrum_name not in win.spectra) and win.spectra:
                win.active_spectrum_name = next(iter(win.spectra.keys()))

        self.active_window_changed.emit(window_id)
        self.active_spectrum_changed.emit(self.active_spectrum)

    def set_window_active_spectrum(self, window_id: str, spec_name: str):
        if window_id in self._windows and spec_name in self._windows[window_id].spectra:
            self._active_window_id = window_id
            self._active_context = "window"
            self._windows[window_id].active_spectrum_name = spec_name
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_window_spectrum_visibility(self, window_id: str, spec_name: str, is_visible: bool):
        if window_id in self._windows and spec_name in self._windows[window_id].spectra:
            self._windows[window_id].visibility[spec_name] = is_visible
            self.window_visibility_changed.emit(window_id, spec_name, is_visible)

    def add_spectrum_to_window(self, window_id: str, spectrum: SpectrumData) -> SpectrumData:
        if window_id not in self._windows:
            raise KeyError(f"Nie odnaleziono okna: {window_id}")

        win = self._windows[window_id]
        cloned_spec = spectrum.clone()

        base_name = cloned_spec.name
        unique_name = base_name
        counter = 1
        while unique_name in win.spectra:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        cloned_spec.name = unique_name
        win.spectra[unique_name] = cloned_spec
        win.visibility[unique_name] = True
        win.active_spectrum_name = unique_name

        self.window_spectrum_added.emit(window_id, cloned_spec)
        return cloned_spec

    def remove_spectrum_from_window(self, window_id: str, name: str):
        if window_id in self._windows:
            win = self._windows[window_id]
            if name in win.spectra:
                del win.spectra[name]
                win.visibility.pop(name, None)
                if win.active_spectrum_name == name:
                    win.active_spectrum_name = next(iter(win.spectra.keys()), None)
                self.window_spectrum_removed.emit(window_id, name)

    # --- KONTEKST I GAŁĘZIE MAGAZYNU ---
    @property
    def active_context(self) -> str:
        return self._active_context

    def set_context(self, context: str):
        if context in ("storage", "window", "table") and context != self._active_context:
            self._active_context = context
            logger.debug(f"Zmiana kontekstu sesji na: {context}")
            self.context_changed.emit(context)
            self.active_spectrum_changed.emit(self.active_spectrum)

    @property
    def active_storage_branch(self) -> str:
        return self._active_storage_branch

    def set_storage_branch(self, branch: str):
        if branch in ("experimental", "atm", "dft", "series") and branch != self._active_storage_branch:
            self._active_storage_branch = branch
            logger.debug(f"Zmiana aktywnej gałęzi magazynu na: {branch}")
            self.storage_branch_changed.emit(branch)
            self.active_spectrum_changed.emit(self.active_spectrum)

    # --- SELEKCJA WIDM DLA WYKRESU I POTOKU ---
    def get_display_spectra(self, include_atm: bool = False) -> List[SpectrumData]:
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            return [s for name, s in win.spectra.items() if win.visibility.get(name, True)]

        if self._active_storage_branch == "atm":
            return [s for name, s in self._atm_spectra.items() if self._atm_visibility.get(name, True)]

        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    def get_pipeline_target_spectra(self) -> List[SpectrumData]:
        if self._active_context == "window" and self.active_window:
            win = self.active_window
            return [s for name, s in win.spectra.items() if win.visibility.get(name, True)]

        if self._active_storage_branch == "atm":
            return [s for name, s in self._atm_spectra.items() if self._atm_visibility.get(name, True)]

        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    # --- ZARZĄDZANIE WIDMAMI MAGAZYNU I WZORCAMI ATMOSFERY ---
    @property
    def spectra(self) -> Dict[str, SpectrumData]:
        return self._spectra

    @property
    def atm_spectra(self) -> Dict[str, SpectrumData]:
        return self._atm_spectra

    def get_spectrum(self, name: str) -> Optional[SpectrumData]:
        if name in self._spectra:
            return self._spectra[name]
        if name in self._atm_spectra:
            return self._atm_spectra[name]
        if self.active_window and name in self.active_window.spectra:
            return self.active_window.spectra[name]
        return None

    @property
    def active_spectrum(self) -> Optional[SpectrumData]:
        if self._active_context == "table":
            return None

        if self._active_context == "window" and self.active_window:
            win = self.active_window
            if win.active_spectrum_name and win.active_spectrum_name in win.spectra:
                return win.spectra[win.active_spectrum_name]
            if win.spectra:
                first_name = next(iter(win.spectra.keys()))
                win.active_spectrum_name = first_name
                return win.spectra[first_name]
            return None

        if self._active_storage_branch == "atm":
            if self._active_spectrum_name in self._atm_spectra:
                return self._atm_spectra[self._active_spectrum_name]
            if self._atm_spectra:
                return next(iter(self._atm_spectra.values()))
            return None

        if self._active_spectrum_name in self._spectra:
            return self._spectra[self._active_spectrum_name]
        if self._spectra:
            return next(iter(self._spectra.values()))
        return None

    def add_spectrum(self, spectrum: SpectrumData, is_atm: Optional[bool] = None,
                     make_active: bool = True, emit_signal: bool = True):
        if is_atm is None:
            is_atm = is_atmospheric_name(spectrum.name)

        target_dict = self._atm_spectra if is_atm else self._spectra
        target_vis = self._atm_visibility if is_atm else self._visibility

        base_name = spectrum.name
        unique_name = base_name
        counter = 1
        all_existing = set(self._spectra.keys()) | set(self._atm_spectra.keys())
        while unique_name in all_existing:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        spectrum.name = unique_name
        target_dict[unique_name] = spectrum
        target_vis[unique_name] = True

        if emit_signal:
            if is_atm:
                self.atm_spectrum_added.emit(spectrum)
            else:
                self.spectrum_added.emit(spectrum)

            if make_active or self._active_spectrum_name is None:
                self.set_active_spectrum(unique_name)

    def add_spectra_batch(self, spectra_list: List[SpectrumData]):
        if not spectra_list:
            return

        added_exp = []
        added_atm = []

        for spec in spectra_list:
            if is_atmospheric_name(spec.name):
                self.add_spectrum(spec, is_atm=True, make_active=False, emit_signal=False)
                added_atm.append(spec)
            else:
                self.add_spectrum(spec, is_atm=False, make_active=False, emit_signal=False)
                added_exp.append(spec)

        self._active_context = "storage"
        self._active_storage_branch = "experimental"

        if added_exp:
            self.spectra_batch_added.emit(added_exp)
        if added_atm:
            self.atm_spectra_batch_added.emit(added_atm)

        all_added = added_exp or added_atm
        if self._active_spectrum_name is None and all_added:
            self.set_active_spectrum(all_added[-1].name)
        else:
            self.context_changed.emit("storage")

    def move_to_atm(self, name: str):
        if name in self._spectra:
            spec = self._spectra.pop(name)
            vis = self._visibility.pop(name, True)
            self._atm_spectra[name] = spec
            self._atm_visibility[name] = vis
            self.spectrum_removed.emit(name)
            self.atm_spectrum_added.emit(spec)

    def move_to_experimental(self, name: str):
        if name in self._atm_spectra:
            spec = self._atm_spectra.pop(name)
            vis = self._atm_visibility.pop(name, True)
            self._spectra[name] = spec
            self._visibility[name] = vis
            self.atm_spectrum_removed.emit(name)
            self.spectrum_added.emit(spec)

    def remove_spectrum(self, name: str):
        if name in self._spectra:
            del self._spectra[name]
            self._visibility.pop(name, None)
            self.spectrum_removed.emit(name)
            if self._active_spectrum_name == name:
                new_active = next(iter(self._spectra.keys()), None)
                self.set_active_spectrum(new_active)
        elif name in self._atm_spectra:
            del self._atm_spectra[name]
            self._atm_visibility.pop(name, None)
            self.atm_spectrum_removed.emit(name)
            if self._active_spectrum_name == name:
                new_active = next(iter(self._atm_spectra.keys()), None)
                self.set_active_spectrum(new_active)

    def set_active_spectrum(self, name: Optional[str]):
        if name != self._active_spectrum_name:
            if name is not None and name not in self._spectra and name not in self._atm_spectra:
                return
            self._active_spectrum_name = name
            self.active_spectrum_changed.emit(self.active_spectrum)

    def set_visibility(self, name: str, is_visible: bool):
        if name in self._spectra and self._visibility.get(name) != is_visible:
            self._visibility[name] = is_visible
            self.visibility_changed.emit(name, is_visible)
        elif name in self._atm_spectra and self._atm_visibility.get(name) != is_visible:
            self._atm_visibility[name] = is_visible
            self.atm_visibility_changed.emit(name, is_visible)

    def is_visible(self, name: str) -> bool:
        if name in self._spectra:
            return self._visibility.get(name, True)
        if name in self._atm_spectra:
            return self._atm_visibility.get(name, True)
        return True

    def get_visible_spectra(self) -> List[SpectrumData]:
        return [s for name, s in self._spectra.items() if self._visibility.get(name, True)]

    def get_visible_atm_spectra(self) -> List[SpectrumData]:
        return [s for name, s in self._atm_spectra.items() if self._atm_visibility.get(name, True)]

    def get_atm_matrix(self, target_wavenumbers: np.ndarray, only_visible: bool = True) -> Optional[np.ndarray]:
        source_dict = self._atm_spectra
        if only_visible:
            specs = [s for name, s in source_dict.items() if self._atm_visibility.get(name, True)]
        else:
            specs = list(source_dict.values())

        if not specs:
            return None

        cols = []
        for s in specs:
            xw, yw = s.wavenumbers, s.intensities
            if xw[0] > xw[-1]:
                xw_s, yw_s = xw[::-1], yw[::-1]
            else:
                xw_s, yw_s = xw, yw
            cols.append(np.interp(target_wavenumbers, xw_s, yw_s))

        return np.column_stack(cols)

    # --- SERIE I TENSORY ---
    @property
    def series(self) -> Dict[str, SpectralSeries]:
        return self._series

    @property
    def active_series(self) -> Optional[SpectralSeries]:
        if self._active_series_name and self._active_series_name in self._series:
            return self._series[self._active_series_name]
        return None

    def get_series(self, name: str) -> Optional[SpectralSeries]:
        return self._series.get(name)

    def add_series(self, series_obj: SpectralSeries, make_active: bool = True):
        base_name = series_obj.name
        unique_name = base_name
        counter = 1
        while unique_name in self._series:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        series_obj.name = unique_name
        self._series[unique_name] = series_obj
        self._series_visibility[unique_name] = True
        self.series_added.emit(series_obj)
        if make_active or self._active_series_name is None:
            self.set_active_series(unique_name)

    def remove_series(self, name: str):
        if name in self._series:
            del self._series[name]
            self._series_visibility.pop(name, None)
            self.series_removed.emit(name)
            if self._active_series_name == name:
                new_active = next(iter(self._series.keys()), None)
                self.set_active_series(new_active)

    def set_active_series(self, name: Optional[str]):
        if name != self._active_series_name:
            if name is not None and name not in self._series:
                return
            self._active_series_name = name
            self.active_series_changed.emit(self.active_series)

    def set_series_visibility(self, name: str, is_visible: bool):
        if name in self._series and self._series_visibility.get(name) != is_visible:
            self._series_visibility[name] = is_visible
            self.series_visibility_changed.emit(name, is_visible)

    def get_visible_series(self) -> List[SpectralSeries]:
        return [s for name, s in self._series.items() if self._series_visibility.get(name, True)]

    # --- CZYSZCZENIE SESJI ---
    def clear_all(self):
        self.metadata = ProjectMetadata()
        self.project_file_path = None
        self.dag_state.clear()

        self._spectra.clear()
        self._visibility.clear()
        self._atm_spectra.clear()
        self._atm_visibility.clear()
        self._active_spectrum_name = None

        self._series.clear()
        self._series_visibility.clear()
        self._active_series_name = None

        self._folders.clear()
        self._folder_counter = 0

        self._windows.clear()
        self._active_window_id = None
        self._window_counter = 0

        self._tables.clear()
        self._active_table_id = None
        self._table_counter = 0

        self.undo_stack.clear()
        logger.info("Wyczyszczono sesję projektu.")
        self.session_cleared.emit()
        self.metadata_changed.emit(self.metadata)


session = ProjectSession()