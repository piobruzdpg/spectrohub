"""Słownik polski dla aplikacji SpectroHub."""

TRANSLATIONS = {
    "app": {
        "title": "SpectroHub - Platforma Analizy FTIR & DFT",
        "ready": "Gotowy",
        "busy": "Przetwarzanie...",
        "cancel": "Anuluj",
        "error": "Błąd",
        "warning": "Ostrzeżenie",
        "success": "Sukces",
    },
    "menu": {
        "file": "Plik",
        "edit": "Edycja",
        "view": "Widok",
        "analysis": "Analiza",
        "tools": "Narzędzia",
        "help": "Pomoc",
        "open_spectra": "Dodaj pliki widm...",
        "save_project": "Zapisz projekt",
        "load_project": "Otwórz projekt",
        "export": "Eksportuj...",
        "exit": "Wyjdź",
    },
    "workspaces": {
        "preprocessing": "1. Preprocessing & Czyszczenie",
        "deconvolution": "2. Dekonwolucja & Fit",
        "chemometrics": "3. Chemometria",
        "quantum": "4. DFT & Wibracje 3D",
        "pub_studio": "5. Pub-Studio",
    },
    "project_tree": {
        "header": "Eksplorator Projektu",
        "group_exp": "Pomiary Eksperymentalne",
        "group_series": "Serie Ewolucyjne (2D)",
        "group_tensors": "Matryce Tensorowe (3D)",
        "group_dft": "Obliczenia Kwantowe (DFT)",
    },
    "deconv": {
        "fit_btn": "⚡ Dopasuj Pasma",
        "add_peak": "➕ Dodaj Pasmo w Kursorze",
        "clear_peaks": "Wyczyść Wszystkie",
        "solvent": "Rozpuszczalnik:",
        "table": {
            "peak": "Pik",
            "center": "Pozycja [cm⁻¹]",
            "fwhm": "FWHM [cm⁻¹]",
            "height": "Wysokość",
            "area": "Pole",
            "area_pct": "Udział [%]",
            "structure": "Struktura",
            "locked": "Zablokowany",
        }
    },
    "dialogs": {
        "file_error": "Nie udało się poprawnie zaimportować wybranego pliku.",
        "confirm_delete": "Czy na pewno chcesz usunąć wybrane elementy?",
    }
}