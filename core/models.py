"""
core/models.py
Kluczowe struktury danych spektroskopowych, tabele profili liczbowych oraz metadane projektu dla SpectroHub.
"""

from dataclasses import dataclass, field
import datetime
import numpy as np
from typing import Optional, Dict, Any, List


@dataclass
class WindowFolder:
    """Reprezentuje folder organizacyjny w sekcji Okien Roboczych."""
    folder_id: str
    name: str
    parent_folder_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_id": self.folder_id,
            "name": self.name,
            "parent_folder_id": self.parent_folder_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WindowFolder":
        return cls(
            folder_id=d.get("folder_id", ""),
            name=d.get("name", "Nowy Folder"),
            parent_folder_id=d.get("parent_folder_id", None),
            metadata=d.get("metadata", {})
        )


@dataclass
class ProjectMetadata:
    """Metadane opisowe całego projektu badawczego."""
    title: str = "Nowy Projekt SpectroHub"
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    modified_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    variant: str = "Standardowy"
    description: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "variant": self.variant,
            "description": self.description,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectMetadata":
        return cls(
            title=d.get("title", "Projekt"),
            author=d.get("author", ""),
            created_at=d.get("created_at", ""),
            modified_at=d.get("modified_at", ""),
            variant=d.get("variant", "Standardowy"),
            description=d.get("description", ""),
            notes=d.get("notes", "")
        )


@dataclass
class DataTable:
    """
    Reprezentuje tabelę danych liczbowych (profile kinetyczne A(t), stężenia MCR C(t),
    ładunki Scores PCA/SVD, parametry dekonwolucji / HDO Fit itp.).
    """
    table_id: str
    name: str
    x_values: np.ndarray                 # Wektor osi X (czas, temperatura, nr próbki lub indeks parametru)
    y_columns: Dict[str, np.ndarray]    # Słownik kolumn danych {etykieta: wektor_Y}
    x_label: str = "Czas / Indeks"
    y_label: str = "Wartość / Stężenie"
    folder_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.x_values = np.asarray(self.x_values, dtype=np.float64)
        for k, v in list(self.y_columns.items()):
            self.y_columns[k] = np.asarray(v, dtype=np.float64)

    @property
    def n_rows(self) -> int:
        return len(self.x_values)

    @property
    def n_cols(self) -> int:
        return len(self.y_columns)

    @property
    def is_parameter_report(self) -> bool:
        """Zwraca True, jeśli tabela jest zestawieniem parametrów (klucz-wartość), a nie profilem kinetycznym."""
        return "row_labels" in self.metadata and len(self.metadata["row_labels"]) > 0

    def clone(self, new_name: Optional[str] = None) -> 'DataTable':
        """Tworzy niezależną kopię tabeli w pamięci RAM."""
        import copy
        return DataTable(
            table_id=self.table_id,
            name=new_name if new_name else self.name,
            x_values=self.x_values.copy(),
            y_columns={k: v.copy() for k, v in self.y_columns.items()},
            x_label=self.x_label,
            y_label=self.y_label,
            folder_id=self.folder_id,
            metadata=copy.deepcopy(self.metadata)
        )

    def to_tsv(self, decimal_comma: bool = True) -> str:
        """Generuje ciąg znaków TSV gotowy do wklejenia w OriginLab lub MS Excel."""
        row_labels = self.metadata.get("row_labels", [])
        has_custom_labels = bool(row_labels and len(row_labels) == self.n_rows)

        col_names = list(self.y_columns.keys())
        first_col_header = "Parametr" if has_custom_labels else self.x_label
        header = [first_col_header] + col_names
        lines = ["\t".join(header)]

        for i in range(self.n_rows):
            if has_custom_labels:
                lbl_str = str(row_labels[i])
            else:
                lbl_str = f"{self.x_values[i]:.6g}"
                if decimal_comma:
                    lbl_str = lbl_str.replace('.', ',')

            row = [lbl_str]
            for c in col_names:
                val = self.y_columns[c][i] if i < len(self.y_columns[c]) else np.nan
                v_str = f"{val:.6g}"
                if decimal_comma:
                    v_str = v_str.replace('.', ',')
                row.append(v_str)
            lines.append("\t".join(row))

        return "\n".join(lines)


@dataclass
class SpectrumData:
    """Ustandaryzowany nośnik danych pojedynczego widma."""
    wavenumbers: np.ndarray      # Oś X (float64, uporządkowana monotonicznie)
    intensities: np.ndarray      # Oś Y (float64, absorbancja/transmisja)
    name: str                    # Nazwa widma / identyfikator
    file_path: str               # Oryginalna ścieżka do pliku
    file_type: str               # 'opus', 'spa', 'spg', 'spc', 'csv', 'jcamp', 'gaussian', 'processed'
    time_str: str = "00:00:00"   # Stempel czasu (HH:MM:SS)
    units_x: str = "cm^-1"
    units_y: str = "absorbance"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=np.float64)
        self.intensities = np.asarray(self.intensities, dtype=np.float64)

        if len(self.wavenumbers) != len(self.intensities):
            raise ValueError(
                f"Niezgodność wymiarów: X={len(self.wavenumbers)}, Y={len(self.intensities)} w widmie '{self.name}'"
            )

    @property
    def n_points(self) -> int:
        return len(self.wavenumbers)

    def clone(self, new_name: Optional[str] = None) -> 'SpectrumData':
        """Tworzy niezależną kopię głęboką widma w pamięci RAM."""
        import copy
        return SpectrumData(
            wavenumbers=self.wavenumbers.copy(),
            intensities=self.intensities.copy(),
            name=new_name if new_name else self.name,
            file_path=self.file_path,
            file_type=self.file_type,
            time_str=self.time_str,
            units_x=self.units_x,
            units_y=self.units_y,
            metadata=copy.deepcopy(self.metadata)
        )


@dataclass
class WorkspaceWindow:
    """Reprezentuje niezależne okno robocze/wykres w projekcie."""
    window_id: str
    name: str
    spectra: Dict[str, SpectrumData] = field(default_factory=dict)
    visibility: Dict[str, bool] = field(default_factory=dict)
    active_spectrum_name: Optional[str] = None
    folder_id: Optional[str] = None


@dataclass
class SpectralSeries:
    """
    Reprezentuje serię ewolucyjną 2D (W x N) lub tensor 3D (W x N x M).
    W = liczba falowa, N = zmienna 1 (np. czas / rampa T), M = zmienna 2 (np. stężenie).
    """
    name: str
    wavenumbers: np.ndarray             # Oś X (W)
    data: np.ndarray                    # Macierz 2D (W x N) lub Tensor 3D (W x N x M)
    axis_n_name: str = "Czas"           # Etykieta wymiaru N
    axis_n_values: Optional[np.ndarray] = None  # Wartości fizyczne (np. sekundy, °C)
    axis_m_name: str = "Stężenie"       # Etykieta wymiaru M (dla 3D)
    axis_m_values: Optional[np.ndarray] = None  # Wartości fizyczne (np. mol/dm³)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)

    @property
    def is_3d(self) -> bool:
        return self.data.ndim == 3

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def n_spectra(self) -> int:
        """Zwraca całkowitą liczbę widm składowych serii."""
        if self.is_3d:
            return self.data.shape[1] * self.data.shape[2]
        return self.data.shape[1] if self.data.ndim == 2 else 0

    def get_spectrum_at(self, idx: int) -> SpectrumData:
        """Zwraca konkretną klatkę serii jako samodzielny obiekt SpectrumData."""
        if self.is_3d:
            w, n, m = self.data.shape
            n_idx = idx // m
            m_idx = idx % m
            y = self.data[:, n_idx, m_idx]
            val_n = self.axis_n_values[n_idx] if self.axis_n_values is not None else n_idx
            val_m = self.axis_m_values[m_idx] if self.axis_m_values is not None else m_idx
            frame_name = f"{self.name}_({self.axis_n_name}={val_n}_{self.axis_m_name}={val_m})"
        else:
            y = self.data[:, idx]
            val_n = self.axis_n_values[idx] if self.axis_n_values is not None else idx
            frame_name = f"{self.name}_({self.axis_n_name}={val_n})"

        return SpectrumData(
            wavenumbers=self.wavenumbers.copy(),
            intensities=y.copy(),
            name=frame_name,
            file_path="",
            file_type="series_frame",
            metadata={"parent_series": self.name, "index": idx}
        )

    def get_all_spectra(self) -> List[SpectrumData]:
        """Zwraca listę wszystkich widm wchodzących w skład serii."""
        return [self.get_spectrum_at(i) for i in range(self.n_spectra)]

    def get_matrix_2d(self) -> np.ndarray:
        """Zwraca widma spłaszczone do macierzy 2D (W x Próbki) dla metod takich jak PCA/SVD."""
        if self.data.ndim == 2:
            return self.data
        w, n, m = self.data.shape
        return self.data.reshape(w, n * m)


@dataclass
class GaussianData(SpectrumData):
    """Rozszerzona struktura dla widm teoretycznych DFT z analizą PED i wizualizacją 3D."""
    stick_freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    stick_intens: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ped_results: List[Dict[str, Any]] = field(default_factory=list)
    hessian_source: str = "LOG"
    scale_factor: float = 1.0
    fwhm: float = 14.0
    shape: str = "Gauss"
    atomnos: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    Lx: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))

    def clone(self, new_name: Optional[str] = None) -> 'GaussianData':
        """Tworzy niezależną kopię głęboką danych kwantowych DFT."""
        import copy
        return GaussianData(
            wavenumbers=self.wavenumbers.copy(),
            intensities=self.intensities.copy(),
            name=new_name if new_name else self.name,
            file_path=self.file_path,
            file_type=self.file_type,
            time_str=self.time_str,
            units_x=self.units_x,
            units_y=self.units_y,
            metadata=copy.deepcopy(self.metadata),
            stick_freqs=self.stick_freqs.copy() if self.stick_freqs is not None else np.array([]),
            stick_intens=self.stick_intens.copy() if self.stick_intens is not None else np.array([]),
            ped_results=copy.deepcopy(self.ped_results),
            hessian_source=self.hessian_source,
            scale_factor=self.scale_factor,
            fwhm=getattr(self, 'fwhm', 14.0),
            shape=getattr(self, 'shape', 'Gauss'),
            atomnos=self.atomnos.copy() if self.atomnos is not None else np.array([]),
            coords=self.coords.copy() if self.coords is not None else np.empty((0, 3)),
            Lx=self.Lx.copy() if self.Lx is not None else np.empty((0, 0))
        )