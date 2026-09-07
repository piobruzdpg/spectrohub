# 📋 Product Requirements Document (PRD): SpectroHub

**Projekt:** Zintegrowana Platforma Analizy i Obróbki Widm FTIR oraz Obliczeń Kwantowo-Chemicznych (SpectroHub)

**Wersja dokumentu:** 1.0 (Architecture Baseline)

**Status:** Projekt wdrożeniowy

**Docelowy stos technologiczny:** Python 3.10+, PySide6 (Qt6), Matplotlib, PyVista (`pyvistaqt`), NumPy, SciPy, pandas, lmfit, pyMCR, TensorLy, openpyxl, h5py.

---

## 1. Wizja i Cel Projektu

### 1.1. Problem

Obecnie w pracowni funkcjonuje 5 wyspecjalizowanych, niezależnych narzędzi:

1. **VaporFit** – kompensacja widm pary wodnej i $CO_2$.
2. **ChemTensor** – analiza wielomacierzy i tensora 3D (PARAFAC, Tucker, 2D-COS, MCR-ALS).
3. **EvoChem** – analiza serii ewolucyjnych (PFA Malinowskiego, AUTOWFA, SPEXFA, MCR-ALS).
4. **BandFit / AmideFit** – dekonwolucja pasma Amidu I, analiza struktur drugorzędowych, GoF, interaktywne suwaki.
5. **IR_Inspektor / HDO Fit** – binarne czytniki I/O, analiza PED Pulaya (Exp $\leftrightarrow$ DFT), animacje 3D modów normalnych, fitowanie Monte Carlo 2 widm bazowych z wyciąganiem czystego residuum.

Utrzymywanie tych programów osobno prowadzi do powielania kodu (np. 4 różne czytniki CSV, podwójny kod dla Savitzky-Golay i ALS), problemów z przenoszeniem danych między etapami pracy oraz braku spójnego interfejsu użytkownika.

### 1.2. Rozwiązanie

Stworzenie jednej wieloplatformowej (macOS / Linux / Windows) aplikacji desktopowej **SpectroHub** o architekturze zorientowanej na **kontekstowe perspektywy robocze (Workspaces)**, ze wspólnym modelem danych w pamięci RAM, odpornym na awarie silnikiem wieloprocesowości oraz zaawansowanym rejestrem zdarzeń (telemetrią błędów).

---

## 2. Architektura Równoległości i Wydajności (Concurrency Engine)

Wielowątkowość w środowisku naukowym (szczególnie na macOS z procesorami Apple Silicon i bibliotekami numerycznymi BLAS/LAPACK/OpenMP) bywa źródłem trudnych do zdebugowania błędów `SIGSEGV` lub blokad pętli zdarzeń UI (`deadlock`).

SpectroHub rozdziela wielozadaniowość na dwa niezależne poziomy:

### 2.1. Poziom 1: Reaktywność GUI (Asynchroniczność Qt)

* **Mechanizm:** Pula wątków `QThreadPool` oraz dedykowane obiekty `QRunnable` / `QThread` komunikujące się z głównym oknem wyłącznie poprzez bezpieczne sygnały Qt (`Signal(dict)`).
* **Zadanie:** Zapobieganie zawieszaniu okna podczas operacji I/O (wczytywanie setek plików Brukera), wstępnego przetwarzania (batch AsLS) oraz aktualizacji wykresów.
* **Informacja zwrotna:** Pasek stanu z pulsującym indykatorem, procentowy `QProgressBar` w stopce oraz możliwość bezpiecznego przerwania operacji przyciskiem **„Anuluj”** (`abort_requested = True`).

### 2.2. Poziom 2: Sprzętowo dopasowane przetwarzanie równoległe (Compute Engine)

* **Detekcja środowiska:** Dynamiczne odczytywanie zasobów:

$$\text{N\_WORKERS} = \max(1, \text{os.cpu\_count}() - 1)$$



(pozostawienie 1 rdzenia zawsze dla systemu operacyjnego i wątku GUI).
* **Izolacja procesów (Unikanie konfliktów OpenMP):**
* Zastosowanie `concurrent.futures.ProcessPoolExecutor` z jawnym kontekstem startowym `multiprocessing.get_context("spawn")`.
* Wymuszenie `OMP_NUM_THREADS=1` i `MKL_NUM_THREADS=1` w procesach potomnych, aby biblioteki C/Fortran (SciPy/NumPy) nie konkurowały o zasoby procesora przy jednoczesnym liczeniu w wielu workerach.


* **Gdzie stosujemy wieloprocesowość:**
1. **Monte Carlo w dekompozycji HDO / pasm syntetycznych:** Równoległe próbkowanie losowych wektorów startowych.
2. **Hurtowa korekcja VaporFit:** Jednoczesne dopasowywanie parametrów pary wodnej dla 100+ widm z serii kinetycznej.
3. **Walidacja krzyżowa (Cross-validation) w PCA / PLS.**
4. **Analiza 3D-COS (przekroje wzdłuż modulatora):** Niezależne obliczanie macierzy synchronicznych/asynchronicznych dla kolejnych plastrów tensora.



---

## 3. Diagnostyka, Logowanie i Gotowość na Debugowanie z LLM

Aplikacja posiada wielopoziomowy system logowania zdarzeń, zaprojektowany tak, aby zrzuty błędów były od razu czytelne i kompletne dla asystenta AI (Gemini).

### 3.1. Rejestrator Danych Środowiskowych (Startup Snapshot)

Przy każdym uruchomieniu programu w pliku `spectrohub_debug.log` generowany jest nagłówek środowiska:

```text
=== SPECTROHUB SESSION START: 2026-09-06 18:30:15 CEST ===
Platform: macOS-15.x-arm64 (Apple Silicon M-series)
Python: 3.11.8 | Qt: 6.7.x | PySide6: 6.7.x
NumPy: 1.26.4 | SciPy: 1.12.0 | Matplotlib: 3.8.3 | PyVista: 0.43.3
CPU Cores: 10 (Worker Pool Size: 9) | RAM Total: 32 GB
Working Directory: /Users/piotr/...
==========================================================

```

### 3.2. Poziomy Logowania i Format Wiadomości

Logowanie odbywa się przez moduł `logging` z jednoczesnym strumieniowaniem na `stdout` (dla uruchamiania z konsoli/skryptu) i do rotującego pliku:

* `DEBUG`: Wymiary macierzy przed i po operacjach (np. `Input tensor shape: (1800, 10, 10), NaNs detected: 0`).
* `INFO`: Załadowanie pliku, ukończenie fitowania, zmiana Workspace'a.
* `WARNING`: Niezbieżność MCR w zadanym limicie iteracji, brakujące metadane czasu pomiaru, konieczność interpolacji osi $\tilde{\nu}$.
* `ERROR`: Niepowodzenie parsowania pliku binarnego, błąd algebry liniowej (osobliwa macierz).
* `CRITICAL`: Błędy zagrażające spójności całego projektu.

### 3.3. LLM-Ready Exception Tracer

Każdy blok `try-except` w newralgicznych modułach raportuje do logu pełny kontekst:

```python
logger.error(f"[ALGO_ERROR] Failed during {operation_name}!\n"
             f"Data shape: {getattr(data, 'shape', None)}, dtype: {getattr(data, 'dtype', None)}\n"
             f"Parameters: {param_dict}\n"
             f"Traceback:\n{traceback.format_exc()}")

```

Umożliwia to błyskawiczne wklejenie fragmentu loga do czatu i natychmiastową diagnozę problemu numerycznego lub indeksowania.

### 3.4. Wewnętrzne Okno Konsoli (In-App Log Viewer)

Dostępne pod klawiszem `F12` lub z menu *Pomoc $\rightarrow$ Dziennik Diagnostyczny*. Pozwala użytkownikowi podejrzeć błędy i kliknąć przycisk **„Kopiuj raport do schowka”**.

---

## 4. Architektura Modułowa I/O Hub (Import / Eksport / Zapis)

Koniec z monolitycznymi plikami wczytującymi. Pakiet `io_hub` zostaje podzielony na wyspecjalizowane, ściśle odizolowane moduły:

```text
io_hub/
├── __init__.py                 # Fabryka parserów (Auto-dispatcher wg rozszerzenia i magii pliku)
├── readers/
│   ├── base_reader.py          # Klasa abstrakcyjna BaseSpectrumReader
│   ├── opus_reader.py          # Natywny dekoder Bruker OPUS (bloki AB, ATR, czasy pomiaru)
│   ├── omnic_reader.py         # Formaty binarne Thermo OMNIC (.spa, .spg)
│   ├── spc_reader.py           # Thermo/GRAMS SPC (obsługa sub-widm)
│   ├── ascii_reader.py         # CSV, DPT, TXT (autodetekcja separatorów i przecinków dziesiętnych)
│   ├── jcamp_reader.py         # Format JCAMP-DX (.dx, .jdx)
│   ├── gaussian_reader.py      # Parsowanie modów i geometrii z .log oraz macierzy siłowej z .fchk
│   └── orca_reader.py          # Parsowanie modów z .out / .property.txt
├── exporters/
│   ├── excel_exporter.py       # Wieloarkuszowy generator raportów XLSX (openpyxl)
│   ├── csv_series_exporter.py  # Eksport macierzy kinetycznych (format PL vs EU/US)
│   ├── spc_exporter.py         # Eksport residuów i widm czystych do binarnego SPC
│   └── figure_exporter.py      # Eksport publikacyjny Matplotlib (600 DPI, wektory SVG/PDF)
└── project/
    └── project_serializer.py   # Zapis/odczyt kompletnego stanu sesji (.shproj w standardzie HDF5)

```

### 4.1. Kluczowe Wymogi dla Importerów

* **Bruker OPUS:** Pancerne parsowanie bloków nagłówkowych bez wywrotki na niestandardowych polach aparatów. Automatyczna ekstrakcja parametrów: data/godzina (`TIM`, `DAT`), temperatura akcesoriów, liczba skanów, rozdzielczość.
* **Odporność na format liczb (ASCII):** Automatyczna analiza pierwszych 10 linii pliku tekstowego w celu rozpoznania separatora (`,`, `;`, `\t`, spacja) oraz kropki vs przecinka dziesiętnego.
* **Walidacja spójności osi $X$:** Weryfikacja czy wczytywana seria posiada identyczną siatkę liczb falowych. W przypadku drobnych rozbieżności (np. $\Delta \tilde{\nu} < 0.1 \text{ cm}^{-1}$ wynikający z zaokrągleń): automatyczna, bezpieczna interpolacja liniowa lub Cubic Spline z wpisem do logu `WARNING`.
* **Trwały zapis projektu (`.shproj`):**
Zamiast wielkich plików JSON (które przy seriach 3D puchną do setek megabajtów), zastosowanie kontenera **HDF5** (`h5py`):
* Błyskawiczny odczyt wielowymiarowych tablic NumPy.
* Zapis metadanych i konfiguracji dopasowanych pików w drzewiastej strukturze atrybutów.



---

## 5. Odporność na Błędy i Polityka Bezpieczeństwa (Defensive Coding)

Aplikacja pod żadnym pozorem nie może się zamknąć samoistnie (crash do pulpitu) w wyniku błędu użytkownika, uszkodzonego pliku czy rozbieżności algorytmu:

1. **Izolacja zdarzeń GUI:** Każdy slot obsługujący kliknięcie przycisku lub interakcję myszy jest zabezpieczony dekoratorem `@safe_slot`:
* Przechwytuje wyjątek.
* Zapisuje pełny traceback do pliku logu.
* Wyświetla elegancki, nieblokujący pasek informacyjny (*Toast notification*) lub okienko `QMessageBox.warning` z przyjaznym dla człowieka komunikatem i kodem błędu.


2. **Sanityzacja danych wejściowych:**
* Przed przekazaniem macierzy do SVD, MCR czy dekonwolucji – automatyczny skan:
* Czy w macierzy są wartości `NaN` lub `Inf`? (automatyczne maskowanie lub interpolacja).
* Czy wycinki liczb falowych są monotoniczne i malejące (zgodnie ze standardem FTIR)?
* Czy liczba punktów w oknie Savitzky-Golay jest nieparzysta i większa od stopnia wielomianu?




3. **Graceful Degradation (Łagodne wygaszanie funkcji):**
* Jeśli w systemie nie ma zainstalowanej biblioteki `pyvistaqt` lub wystąpi problem ze sterownikiem OpenGL:
* Aplikacja uruchamia się normalnie.
* Zakładka **DFT & Wibracje 3D** przełącza się w tryb 2D (wykres pałeczkowy modów normalnych + wektory na płaszczyźnie), informując o braku wsparcia 3D bez blokowania reszty programu.





---

## 6. Specyfikacja Funkcjonalna Perspektyw Roboczych (Workspaces)

Główny interfejs opiera się na 5 logicznych perspektywach roboczych. Zmiana perspektywy na górnej belce automatycznie dostosowuje lewe drzewo danych, centralny canvas oraz prawy panel inspektora.

```
+-----------------------------------------------------------------------------------------------+
|  PERSPEKTYWY: [ 1. Data & Prep ] [ 2. Dekonwolucja ] [ 3. Chemometria ] [ 4. DFT & PED ] [ 5. PubStudio ] |
+-----------------------------------------------------------------------------------------------+

```

### 6.1. Workspace 1: Data Manager & Preprocessing

* **Źródło logiki:** `vaporfit_0.1.py`, moduły preprocessingu z `ChemTensor` i `EvoChem`.
* **Funkcjonalności:**
* **Eksplorator:** Drzewo wczytanych pomiarów, grup widm, serii kinetycznych.
* **VaporFit Core:** Dynamiczne odejmowanie widm atmosferycznych ($H_2O_{vap}$, $CO_2$) z automatyczną optymalizacją wag metodą najmniejszych kwadratów i weryfikacją gładkości (indeksy SSI, SDV).
* **Korekcja linii bazowej:**
* Liniowa (dwupunktowa).
* Wielomianowa.
* **AsLS** (Asymmetric Least Squares Smoothing) – szybka implementacja na macierzach rzadkich (`scipy.sparse`).
* **airPLS** (Adaptive Iteratively Reweighted Penalized Least Squares) – niewymagająca zgadywania parametru asymetrii.


* **Filtry i transformacje:** Savitzky-Golay (wygładzanie oraz 1. i 2. pochodna), SNV, Min-Max, MSC, normalizacja wektorowa $L_2$, Offset (kotwiczenie wskazanego punktu w 0.0), przycinanie zakresów (Crop).
* **Pipeline Manager:** Lista kroków obróbki (np. *1. Crop $\rightarrow$ 2. AsLS $\rightarrow$ 3. SNV*) z możliwością przestawiania kolejności (góra/dół), wyłączania podglądu i powrotu do danych surowych jednym kliknięciem.



### 6.2. Workspace 2: Dekonwolucja i Fitowanie Pasma (Amide & HDO)

* **Źródło logiki:** `bandfit.md`, `fitowanie2widm.md`.
* **Funkcjonalności:**
* **Silnik optymalizacji:** `lmfit` (Levenberg-Marquardt, Nelder-Mead, Powell).
* **Kształty pasm:** Gauss, Lorentz, Voigt, Pseudo-Voigt.
* **Więzy (Constraints):** Sztywne blokowanie (Lock) lub zakresy dopuszczalne dla: położenia ($\tilde{\nu}$), szerokości połówkowej (FWHM), amplitudy i pola powierzchni.
* **Wspomaganie decyzji:** Podgląd 2. pochodnej pod wykresem z automatycznym wyszukiwaniem minimów (Auto-Peak Finder).
* **Interaktywne kontrolki:** Przeciąganie parametrów piku bezpośrednio na wykresie myszą lub polami wartości numerycznych z płynną zmianą i podglądem residuum na żywo.
* **Środowisko $H_2O$ vs $D_2O$:** Automatyczne przełączanie słowników pasm struktury drugorzędowej (Amid I vs Amid I') i wyliczanie procentowego udziału struktur ($\alpha$-helisa, $\beta$-harmonijka, zwroty, zwoje).
* **Dekompozycja HDO / Custom Shapes (Monte Carlo):**
* Import eksperymentalnych widm bazowych ($W_1$, $W_2$) z automatyczną interpolacją.
* Próbkowanie Monte Carlo na wielu rdzeniach CPU w celu znalezienia optymalnych wag $c_A$, $c_B$.
* **Izolacja czystego sygnału nadmiarowego (Residuum):**

$$\text{Residuum} = (N \cdot W_0) - (c_A \cdot W_1 + c_B \cdot W_2)$$



Eksport residuum bezpośrednio do binarnego SPC lub przekazanie do modułu chemometrii.


* **Batch / Global Fitting (Seryjne dopasowywanie):**
* Przenoszenie wyfitowanego zestawu pasm jako szablonu na kolejne widma z serii.
* Dopasowywanie globalne serii temperaturowych (pozycje pasm powiązane/stałe, zmienne tylko pola i FWHM).





### 6.3. Workspace 3: Chemometria i Analiza Ewolucyjna

* **Źródło logiki:** `evochem.py`, `chemotensor.py`.
* **Funkcjonalności:**
* **Analiza Rang i Czynników Głównych (PFA / PCA):**
* Statystyki Malinowskiego: Błąd Rzeczywisty (RE), Wskaźnik IND, poziom istotności F (%SL).
* Wykres osypiska (Scree plot), rekonstrukcja sygnału z wybranej liczby $k$ czynników.


* **Ewolucyjna Analiza Czynnikowa:**
* **AUTOWFA:** Automatyczne wyznaczanie okien analitycznych na podstawie wektorów IKSFA i iteracyjne udoskonalanie profili stężeń.
* **SPEXFA:** Izolacja czystych widm i stężeń składników z niepewnościami statystycznymi (`spexerr`, `concerr`).


* **Multivariate Curve Resolution (MCR-ALS):**
* Ograniczenia: Nieujemność (NNLS), domknięcie bilansu (Closure), jednomodalność (Unimodality).
* Inteligentna inicjalizacja za pomocą SPEXFA lub znanych widm eksperymentalnych.
* Blokowanie wybranych profili stężeń/widm (Fixed components).


* **Metody Wielowymiarowe i Tensorowe (3D):**
* **PARAFAC** oraz **Tucker3** dla siatek pomiarowych (np. stężenie A vs stężenie B vs widmo).
* Rekonstrukcja sygnału z tensora.


* **Spektroskopia Korelacyjna 2D (2D-COS):**
* Macierze synchroniczne $\Phi(\nu_1, \nu_2)$ i asynchroniczne $\Psi(\nu_1, \nu_2)$.
* Slicer 3D pozwalający na przeglądanie korelacji wzdłuż wybranej zmiennej ewolucyjnej.
* Redukcja nieliniowa: UMAP i t-SNE dla szybkiej klasyfikacji próbek.





### 6.4. Workspace 4: Obliczenia Kwantowe i Analiza Wibracyjna (DFT & PED)

* **Źródło logiki:** `ir_inspektor.md`, silnik 3D z `neb_aligner`.
* **Funkcjonalności:**
* **Import DFT:** Odczyt wyjść programów Gaussian 09/16 (`.log`, `.out`, `.fchk`) oraz ORCA 5/6 (`.out`, `.property.txt`).
* **Silnik PED (Potential Energy Distribution):**
* Rzutowanie drgań normalnych na lokalne współrzędne symetrii (SIC) metodą Pulaya z wykorzystaniem macierzy $B$ Wilsona.
* Analityczny odczyt macierzy siłowej z `.fchk` lub rekonstrukcja z wektorów $L_x$ z pliku `.log`.
* Tabela udziałów PED dla każdego modu normalnego.


* **Interaktywne dopasowywanie (Peak Matching Exp $\leftrightarrow$ DFT):**
* Łączenie pasm eksperymentalnych z teoretycznymi pałeczkami Gaussiana/ORCA.
* Wyznaczanie współczynnika przeskalowania częstości (Scaling Factor).


* **Symulacja efektu izotopowego "On-the-Fly":**
* Rozwiązywanie masowo-ważonego zagadnienia własnego ($M_{\text{iso}}^{-1/2} F M_{\text{iso}}^{-1/2}$) dla nowych mas (np. deuterowanie $H \rightarrow D$, izotopy $^{13}C$, $^{15}N$, $^{18}O$) w ułamku sekundy, bez konieczności ponownego uruchamiania obliczeń w klastrze.


* **Wizualizator drgań 3D (PyVista/VTK):**
* Renderowanie cząsteczki z bazą promieni Álvareza i kolorystyką CPK.
* Animacja wektorów drgań w czasie rzeczywistym z suwakami amplitudy i szybkości (analogicznie do odtwarzacza w NEBAligner).
* Podwójne kliknięcie na pałeczkę częstości natychmiast odpala animację odpowiadającego modu.





### 6.5. Workspace 5: Pub-Studio & Raportowanie

* **Źródło logiki:** `figure_exporter.py` z IR_Inspektor, moduły eksportu Excel.
* **Funkcjonalności:**
* **Układy publikacyjne:** Gotowe szablony figur jednoszpaltowych (8.5 cm) i dwuszpaltowych (17.5 cm) dla wydawnictw ACS, RSC, Elsevier, Wiley.
* **Rozdzielczość:** Wektorowy eksport do PDF/SVG oraz rastrowy do TIFF/PNG w 600 DPI.
* **Wielopanelowość:** Układanie widoków w siatki wielopanelowe z automatycznymi literami **a)**, **b)**, **c)** i kontrolą marginesów (panele stykające się vs odseparowane).
* **Inwersja kolorystyczna (Smart Print):** Niezależnie od ciemnego motywu interfejsu (Dark Mode), eksport wykresów automatycznie generuje białe tło, zoptymalizowane grubości linii i kontrastową paletę pasm dostosowaną do druku czarno-białego lub kolorowego.
* **Zbiorczy Raport Excel (.xlsx):**
* Arkusz 1: Metadane pomiaru i parametry obróbki.
* Arkusz 2: Macierz widm (surowe, po preprocessingu, dopasowane).
* Arkusz 3: Parametry pasm dekonwolucji (pozycja, FWHM, pole, udział % w pasmie Amidu I, GoF).
* Arkusz 4: Wyniki chemometryczne (profile stężeń MCR, wagi PARAFAC, wartości własne PFA).
* Arkusz 5: Tabela modów DFT z udziałami PED.





---

## 7. Plan Realizacji i Kamienie Milowe (Roadmap)

Projekt proponuję podzielić na 5 logicznych etapów, z których każdy kończy się w pełni działającą, przetestowaną częścią:

```
[Etap 1: Szkielet & I/O Hub] ──> [Etap 2: Preprocessing & Vapor] ──> [Etap 3: Fitowanie & HDO]
                                                                             │
[Etap 5: Integracja & PubStudio] <── [Etap 4: Chemometria & DFT] <───────────┘

```

* **Etap 1: Fundament Architektury, Silnik I/O i Logowanie**
* Utworzenie repozytorium i struktury modułowej.
* Implementacja wielopoziomowego loggera (`spectrohub_debug.log`) i przechwytywacza wyjątków.
* Moduły I/O: osobne parsery dla OPUS, OMNIC, SPC, CSV/TXT z autodetekcją formatu.
* Główne okno PySide6 z systemem dokowania (`QDockWidget`), paskiem perspektyw i drzewem projektu.


* **Etap 2: Preprocessing Suite & VaporFit**
* Integracja algorytmów obróbki: AsLS, airPLS, Savitzky-Golay, normalizacje, Offset.
* Pipeline Manager (zarządzanie krokami).
* Port silnika VaporFit (wieloprocesowe usuwanie par wody i $CO_2$).


* **Etap 3: Dekonwolucja Amidu I i Moduł HDO**
* Silnik `lmfit` z modelami Voigta, Gaussa, Lorentza i ograniczeniami (Lock/Bounds).
* Interaktywny podgląd 2. pochodnej i detekcja minimów.
* Przeniesienie silnika Monte Carlo z `fitowanie2widm` do wieloprocesowego modułu obliczeniowego z wyciąganiem residuum.


* **Etap 4: Chemometria (EvoChem + ChemTensor) oraz Silnik DFT**
* Zunifikowany silnik SVD / PFA / AUTOWFA / SPEXFA / MCR-ALS.
* Obsługa tensora 3D (PARAFAC, Tucker3, 2D-COS).
* Port parsera Gaussiana/ORCA, silnika PED Pulaya oraz podglądu 3D cząsteczek w `pyvistaqt` z animacją drgań.


* **Etap 5: Pub-Studio, Raportowanie Excel i Wersja Portable**
* Generator figur publikacyjnych 8.5 cm / 600 DPI.
* Zbiorczy eksporter wieloarkuszowy `.xlsx`.
* Przygotowanie skryptu startowego, środowiska wirtualnego i paczki portable dla pracowni studenckiej (macOS / Windows).


---

## 8. Dodatkowe założenia

### 1. Architektura Wielojęzyczności (i18n) bez `if lang == 'pl'`

Zamiast pisać warunki w kodzie widoków, stosuje się wzorzec **Centralnego Menedżera Tłumaczeń (Translation Registry)** połączony ze słownikami kluczy i sygnałami Qt.

**Jak to działa w praktyce:**

1. **Słowniki jako moduły Pythona lub JSON:** Tworzymy folder `locales/` z plikami `pl.py` oraz `en.py`. Każdy plik to hierarchiczny słownik (`dict`), w którym klucze są stałymi identyfikatorami.
2. **Globalna funkcja `tr(key)`:** Zamiast wpisywać tekst na sztywno, kontrolki pytają funkcję pomocniczą o aktualne tłumaczenie klucza (np. `tr("deconv.peaks_table.fwhm")`).
3. **Dynamiczny retranslating bez restartu aplikacji:** Po kliknięciu ikonki flagi w prawym górnym rogu, emitowany jest sygnał Qt: `language_changed.emit()`. Każde okno i widget ma jedną małą metodę `retranslate_ui()`, która nasłuchuje tego sygnału i podmienia teksty w locie.

**Struktura słowników (`locales/`):**

```python
# locales/pl.py
TRANSLATIONS = {
    "app": {
        "title": "SpectroHub - Zintegrowana Platforma FTIR",
        "ready": "Gotowy",
        "cancel": "Anuluj",
    },
    "workspaces": {
        "preprocessing": "1. Preprocessing i Oczyszczanie",
        "deconvolution": "2. Dekonwolucja i Fitowanie",
        "chemometrics": "3. Chemometria",
        "quantum": "4. DFT i Wibracje 3D",
        "pub_studio": "5. Pub-Studio",
    },
    "deconv": {
        "fit_btn": "⚡ Dopasuj Pasma",
        "add_peak": "➕ Dodaj pasmo",
        "table": {
            "peak": "Pik",
            "center": "Pozycja [cm⁻¹]",
            "fwhm": "FWHM [cm⁻¹]",
            "area_pct": "Udział [%]",
        }
    }
}

```

```python
# locales/en.py
TRANSLATIONS = {
    "app": {
        "title": "SpectroHub - Integrated FTIR Platform",
        "ready": "Ready",
        "cancel": "Cancel",
    },
    "workspaces": {
        "preprocessing": "1. Preprocessing & Clean",
        "deconvolution": "2. Deconvolution & Fit",
        "chemometrics": "3. Chemometrics",
        "quantum": "4. DFT & 3D Vibrations",
        "pub_studio": "5. Pub-Studio",
    },
    "deconv": {
        "fit_btn": "⚡ Fit Bands",
        "add_peak": "➕ Add Peak",
        "table": {
            "peak": "Peak",
            "center": "Position [cm⁻¹]",
            "fwhm": "FWHM [cm⁻¹]",
            "area_pct": "Area [%]",
        }
    }
}

```

**Menedżer w kodzie (`core/i18n.py`):**

```python
from PySide6.QtCore import QObject, Signal
from locales import pl, en

class I18nManager(QObject):
    language_changed = Signal()

    def __init__(self):
        super().__init__()
        self._current_lang = "pl"
        self._dictionaries = {
            "pl": pl.TRANSLATIONS,
            "en": en.TRANSLATIONS
        }

    @property
    def current_language(self):
        return self._current_lang

    def set_language(self, lang_code: str):
        if lang_code in self._dictionaries and lang_code != self._current_lang:
            self._current_lang = lang_code
            self.language_changed.emit()

    def tr(self, key_path: str, default: str = "") -> str:
        """Pobiera tekst ze zagnieżdżonego słownika po kropkach, np. 'deconv.table.peak'"""
        keys = key_path.split(".")
        data = self._dictionaries.get(self._current_lang, {})
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
            else:
                return default or key_path
        return str(data) if data is not None else (default or key_path)

# Singleton dostępny w całym projekcie
i18n = I18nManager()
tr = i18n.tr

```

**Użycie w widoku (np. `gui/workspaces/ws_deconvolution.py`):**

```python
class DeconvWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.btn_fit = QPushButton()
        self.btn_add = QPushButton()
        
        # Wiążemy sygnał zmiany języka z odświeżaniem tekstów
        i18n.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.btn_fit.setText(tr("deconv.fit_btn"))
        self.btn_add.setText(tr("deconv.add_peak"))
        # Wszystkie labele, nagłówki tabeli i tooltipy odświeżają się w ułamku sekundy

```

W prawym górnym rogu paska menu lub paska narzędzi umieszczamy mały `QToolButton` z ikoną flagi (lub przełącznik `PL | EN`), który wywołuje wyłącznie:
`i18n.set_language("en")` / `i18n.set_language("pl")`. Wybrany język zapamiętujemy w ustawieniach systemowych (`QSettings`), by pamiętać go po restarcie.

---

### 2. Organizacja Repozytorium Git / GitHub od Dnia 1

W projektach spektroskopowych i chemii obliczeniowej największym zagrożeniem dla Gita są **duże pliki binarne** (pliki `.log` i `.fchk` z Gaussiana po kilkaset megabajtów, bazy plików `.opus`, pliki `.h5` czy zrzuty projektów), które przypadkowo dodane potrafią bezpowrotnie napompować repozytorium do wielu gigabajtów.

**Struktura Repo na GitHubie:**

```text
spectrohub/
├── .github/
│   └── workflows/
│       └── tests.yml            # Automatyczne testy (CI) przy pushu/PR
├── .gitignore                   # Pancerny filtr plików
├── README.md                    # Opis projektu, instalacja, screeny
├── LICENSE                      # Np. GNU GPL v3.0
├── pyproject.toml               # Metadane pakietu i zależności (pip install -e .)
├── requirements.txt             # Główne zależności środowiska
├── spectrohub_app.py            # Skrypt startowy (Launcher)
│
├── src/                         # Właściwy pakiet źródłowy
│   └── spectrohub/
│       ├── __init__.py
│       ├── core/                # Modele danych, i18n, menedżer zadań
│       ├── io_hub/              # Wszystkie czytniki i eksporterzy
│       ├── algorithms/          # Czysta matematyka (Zero Qt!)
│       ├── gui/                 # Widoki, doki, canvas
│       └── locales/             # Słowniki językowe
│
└── tests/                       # Testy jednostkowe
    ├── test_readers.py
    ├── test_math.py
    └── sample_data/             # BARDZO MAŁE pliki testowe (<100 KB) do testów jednostkowych

```

**Rygorystyczny `.gitignore` (Podstawa czystego startu):**

```gitignore
# Środowiska wirtualne i cache Pythona
.venv/
venv/
env/
__pycache__/
*.py[cod]
*$py.class
build/
dist/
*.egg-info/

# Pliki środowisk IDE i systemowe
.idea/
.vscode/
.DS_Store
Thumbs.db

# Dzienniki logów programu
*.log
spectrohub_debug.log

# DANE NAUKOWE (Kluczowe: blokujemy zaśmiecanie repo dużymi danymi!)
*.fchk
*.chk
*.log
*.out
*.property.txt
*.interp
*.h5
*.shproj
*.xlsx
*.csv
*.spc
*.spa
*.spg

# Wyjątek: pozwalamy Gitowi śledzić wyłącznie małe pliki wzorcowe w tests/sample_data/
!tests/sample_data/*

```

**Strategia Gałęzi i Wdrożeń:**

* `main` – stabilny kod, który zawsze działa (tylko przetestowane moduły).
* `dev` – główna gałąź robocza, do której scalamy kolejne etapy.
* Gałęzie etapowe (Feature branches): np. `feat/io-readers`, `feat/deconv-engine`, `feat/gui-framework`.
* **Konwencja Commitów (Conventional Commits):**
* `feat: add Bruker OPUS binary reader with metadata parser`
* `fix: prevent divide-by-zero in Savitzky-Golay for small windows`
* `refactor: extract i18n keys to separate dictionaries`
* `i18n: add English translations for chemometrics panel`



---

### 3. Trzy Zasady Architektoniczne Chroniące Przed „Rozgrzebywaniem”

Abyśmy za dwa miesiące nie musieli przepisywać połowy programu, musimy od pierwszej linii kodu trzymać się trzech reguł:

1. **Zasada Czystego Rdzenia (No Qt in Math):**
Moduły w `algorithms/` oraz `io_hub/` **nie mogą importować PySide6 ani PyVista**. Mają operować wyłącznie na czystych tablicach NumPy, strukturach SciPy i wbudowanych typach Pythona. Dzięki temu:
* Matematykę przetestujemy w 2 sekundy prostym skryptem bez uruchamiania okna aplikacji.
* Funkcje liczące można bez przeszkód uruchamiać w procesach potomnych wielowątkowości (`multiprocessing`).


2. **Jedno Źródło Prawdy (Single Source of Truth):**
Widoki GUI nie przechowują danych widmowych w tabelach czy zmiennych okna. Wszystko znajduje się w `ProjectSession` (aktywne widma, serie, dopasowane piki). Jeśli algorytm zmieni dane, powiadamia o tym sygnałem Qt, a widoki jedynie odświeżają swój stan.
3. **Pasywna Konfiguracja (`QSettings`):**
Stan okna (położenie doków, rozmiar, ostatni katalog, wybrany język, liczba wątków) zapisuje się automatycznie przy zamknięciu programu i odtwarza przy starcie.

---
