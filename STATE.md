06.09.2026 = 23.00 Zrobiliśmy dziś potężny kawał solidnej roboty! To idealny moment na commit i zrobienie checkpointa w repozytorium.

---

**Podsumowanie dzisiejszych postępów:**

* **Faza 1 (Dekonwolucja & Live Interakcja):**
* Obsługa kliknięć myszą (`Shift` / `Cmd` / `Ctrl`) stawiająca piki ze skalowaniem pola do realnej absorbancji.
* Renderowanie dzwonów składowych na żywo z pionowymi etykietami struktur (90°).
* Auto-Peak Finder na bazie odwróconej 2. pochodnej.
* Solwery Levenberg-Marquardt, Nelder-Mead oraz Powell.
* Pełne wsparcie historii operacji przez `QUndoStack`.
* 
* **Moduł I/O Thermo Grams SPC:**
* Natywny, bezbiblioteczny czytnik `SPCReader` (formaty `0x4B`, `0x4D`, sub-widma).
* Precyzyjny binarny eksporter `SPCExporter` w standardzie Galactic New SPC (`0x4B`, `float32`).

* **Faza 2 (VaporFit Suite):**
* Czysty silnik `algorithms/atmospheric.py` z analitycznym OLS na składowych ostrych.
* Auto-walidator SDV eliminujący błąd podwajania pików.
* Hurtowy dialog `VaporFitDialog` (Batch mode) z paskiem postępu i autodetekcją okna SG.

---

**Plan na kolejną sesję:**

1. **Uniwersalny moduł dekompozycji / HDO Fit:** implementacja silnika wieloprocesowego Monte Carlo dla $k$ widm czystych ($W_1, \dots, W_k$) oraz $m$ pasm syntetycznych z izolacją czystego residuum.
2. **Rozszerzenie Dekonwolucji o tryb ogólny:** przełącznik `Amid I (Białka) | Ogólny (Custom)`.


07.09.2026 — Checkpoint: HDO, DFT 3D, Export Suite & Chemometrics

Wykonane w tej sesji:
* **HDO & Dekonwolucja:** Uogólniony model Monte Carlo na k widm bazowych i pasma GxL z wyciąganiem czystego residuum oraz tryb ogólny (Custom) w dekonwolucji z elastycznymi etykietami.
* **DFT & PED 3D:** Parser Gaussiana i ORCA 6.1 (.out), naprawiony dispatcher i kategoryzacja w drzewie, pałeczki drgań na canvasie z synchronizacją tabeli PED, regulacja FWHM/kształtu pasma oraz pływające okno 3D (PyVista) z płynną animacją modów.
* **Pakiet Eksportowy:** Wieloarkuszowy Excel (.xlsx), czyste dwukolumnowe ASCII/CSV/DPT bez nagłówków (kompatybilne z OMNIC/OPUS), SPC i JCAMP-DX w menu i pod PPM.
* **Serie 2D i Tensory 3D:** Model `SpectralSeries`, kreator `TensorBuilderDialog` z ekstrakcją rzeczywistych czasów pomiaru (OPUS TIM, regex z nazw, mtime) oraz zasilaniem siatek 3D z uzupełnianiem braków.
* **Chemometria (Workspace 3):** Zunifikowany silnik łączący EvoChem i ChemTensor (MCR-ALS, SPEXFA, PFA Malinowskiego, AUTOWFA, FSMWEFA, PARAFAC, 2D-COS) z rozłączaną osią X na wykresie i wstrzykiwaniem czystych widm do drzewa.

Plan na kolejną sesję:
1. Workspace 1: Wpięcie hurtowego potoku czyszczenia i narzędzi korekcyjnych (Pipeline Manager).
2. Doszlifowanie UX wykresu głównego dla serii ewolucyjnych (przekroje kinetyczne 2D w stylu SpectraPlot z EvoChem).
3. Moduł Pub-Studio (elastyczne panele publikacyjne 8.5/17.5 cm, 600 DPI).


08.09.2026 stan pracy

**1. Wydajność importu i stabilność renderingu (eliminacja SIGSEGV)**
* **Hurtowy import wsadowy:** Naprawiliśmy wąskie gardło sekwencyjnego ładowania plików w `gui/main_window.py` i `project_tree.py`. Zamiast dodawać pliki po jednym, worker w tle agreguje setki widm i przekazuje je do sesji w jednym strzale.
* **Eliminacja crashy pamięci (`SIGSEGV`):** Zlikwidowaliśmy dynamiczne usuwanie i odtwarzanie podwykresów Matplotliba w pętli zdarzeń Qt (`prepare_axes`), co powodowało utratę wskaźników w backendzie C++.
* **Akceleracja `LineCollection`:** Przy zbiorach powyżej 15 widm wykres rysuje dane jako pojedynczą kolekcję wierzchołków GPU w palecie `viridis`, z wyróżnieniem aktywnego widma na biało. Wyświetlenie 256 widm skróciło się do 1–2 sekund bez zacinania interfejsu.

**2. Naprawa interakcji między panelami a wykresem**
* **Odzyskanie pełnego widoku projektu:** Wyeliminowaliśmy konflikt z `ws_deconvolution.py`, gdzie niewidoczny w tle panel przechwytywał sygnał zmiany aktywnego widma i nadpisywał wykres pojedynczą krzywą.
* Zabezpieczyliśmy kliknięcia w gałęzie nadrzędne drzewa projektu, by nie wysyłały pustych stanów do sesji.

**3. Zintegrowany Pipeline Manager (Workspace 1)**
* **Kolejkowanie etapów potoku:** Wprowadziliśmy panel potoku z listą kroków (*Crop*, *AsLS*, *Linia krańcowa*, *SG*, *Pochodne*, *SNV*, *Min-Max*, *L₂*, *Offset*) oraz przyciskami zmiany kolejności (`▲` / `▼`).
* **Edycja parametrów:** Dodaliśmy okno modalne (`StepParamDialog`) otwierane dwuklikiem na dodany krok, co pozwala precyzyjnie korygować parametry (np. $\lambda$, $p$, okno SG) w locie.
* **Elastyczny zakres:** Potok może działać na aktywnym widmie, wsadowo na wszystkich widocznych plikach lub wzdłuż wymiaru liczb falowych serii 2D/3D (w miejscu lub tworząc obiekty `proc_*`).
* **Offset w punkcie:** Dodaliśmy zerowanie absorbancji przy wybranej liczbie falowej $A(\nu) \rightarrow 0.0$ (zarówno jako przycisk, jak i krok potoku).
* **Uzupełnienie biblioteki `algorithms/preprocessing.py`:** Dodaliśmy `get_first_derivative`, obsłużyliśmy brakujące aliasy (`min_max_scale`, `snv`, itp.) oraz wyeliminowaliśmy błędy `AttributeError`.

**4. Przekroje kinetyczne 2D (SpectraPlot z EvoChem)**
* **Interaktywny kursor analityczny:** Kliknięcie LPM na wybrane pasmo górnego wykresu rzuca pionowy wskaźnik i wykreśla profil ewolucyjny absorbancji na dolnym panelu.
* **Multi-Trace:** `Shift` + LPM pozwala zablokować do 5 pasm naraz w różnych kolorach z automatyczną legendą do bezpośredniego porównywania kinetyk.
* **Rozróżnianie typów danych na osi X:**
* Dla plików eksperymentalnych: czyste indeksy $1, 2, \dots, N$ i etykieta *„Numer widma”* (bez nazw plików).
* Dla serii 2D: rzeczywista oś czasu (np. w minutach) na dole oraz zsynchronizowana górna oś numerów widm (`twiny`).



**5. Ergonomia myszy i pełne Undo / Redo**

* **Rozdzielenie ról przycisków myszy:**
* **Przeciągnięcie LPM:** Niezależny prostokątny zoom na obu panelach (filtr EvoChem rozróżnia kliknięcie analityczne od przeciągnięcia).
* **Dwuklik LPM:** Natychmiastowy reset zoomu do pełnego widoku.
* **PPM:** Dedykowane menu kontekstowe (kopiowanie danych kinetycznych do schowka TSV pod Origin/Excel, eksport do CSV, kopiowanie obrazu wykresu, czyszczenie wskaźników).
* **Globalne skróty cofania:** Wpięliśmy `session.undo_stack` pod systemowe `⌘+Z` / `Ctrl+Z` oraz `⌘+Shift+Z` / `Ctrl+Y` w oknie głównym, z pełną obsługą cofania operacji na widmach, wsadach i seriach 2D.


**6. Decyzje strategiczne do `STATE.md**`

* Odłożyliśmy moduł rycin publikacyjnych (Pub-Studio) na później jako niekrytyczny na tym etapie.
* Zaplanowaliśmy przyszłą ewolucję Pipeline Managera w kierunku wizualnych węzłów / kafelków (styl Orange / n8n / ComfyUI).
* Zanotowaliśmy koncepcję architektury Multi-Canvas (wielu niezależnych przestrzeni kreślenia z obsługą Drag & Drop).


### 08.09.2026 — Checkpoint: Architektura Magazynu i Okien Roboczych oraz Naprawa Pamięci Historii

**1. Pancerne Undo / Redo i Eliminacja Rozjazdów Pamięci RAM**

* **Sygnały modyfikacji in-place:** Do `core/session.py` dodano sygnały `spectrum_data_changed`, `batch_spectra_changed` oraz `series_data_changed`, emitowane po każdej mutacji wektorów $X/Y$ w pamięci.
* **Eliminacja podwójnej mutacji:** W `ws_preprocessing.py` zlikwidowano przedwczesne nadpisywanie tablic w pętli przed wykonaniem `undo_stack.push()`, co zapobiega dublowaniu operacji przez automatyczne wywołanie `redo()`.
* **Izolacja nazw dla widm generowanych (`proc_*`):** Zabezpieczono `AddBatchSpectraCommand` przed rozjazdem identyfikatorów po nadaniu unikalnych nazw w sesji, zapewniając bezbłędne usuwanie wygenerowanych widm przy cofaniu.
* **Reaktywność chemometrii:** `ws_chemometrics.py` podpięto pod nowe sygnały sesji — macierze 2D/3D przeliczają się natychmiast po cofnięciu lub ponowieniu potoku.

**2. Rozdział Architektoniczny: Magazyn Danych vs. Okna Robocze**

* **Głębokie klonowanie (`models.py`):** Wdrożono metody `.clone()` dla `SpectrumData` i `GaussianData`, gwarantujące fizyczną izolację tablic NumPy przy przenoszeniu danych.
* **Wielokontekstowość w `session.py`:** Wprowadzono strukturę `WorkspaceWindow` oraz przełącznik kontekstu (`active_context`: `"storage"` vs `"window"`). Metoda `get_display_spectra()` dynamicznie zasila wykres danymi z aktywnego widoku.
* **Nowa hierarchia `ProjectTreeWidget`:**
* **📦 MAGAZYN DANYCH (Referencja):** Przechowuje surowe/bazowe pliki pomiarowe, serie i DFT. Zabezpieczony przed przypadkową edycją pojedynczych widm, z zachowaniem pełnego dostępu do potoków wsadowych (Batch Preprocessing, Batch VaporFit).
* **🖥️ OKNA WYKRESÓW / ROBOCZE:** Niezależne przestrzenie robocze (piaskownice) z możliwością dodawania, usuwania i zmiany nazw okien.


* **Ergonomia Drag & Drop:** Obsłużono przeciąganie pojedynczych widm, wielokrotnych selekcji oraz całych gałęzi (np. *Pomiary Eksperymentalne*) z automatycznym rozwijaniem poddrzew i bezszumnym klonowaniem do wybranego okna.

**3. Szlifowanie Canvasu i Interakcji Drzewa**

* **Niezależna kinetyka:** Zaktualizowano `SpectroCanvas._get_evolutionary_dataset()`, opierając pobieranie danych na `get_display_spectra()`. Wyeliminowało to konflikt zakresów liczb falowych między przyciętym magazynem a nienaruszonym oknem roboczym.
* **Pełna kontrola widoczności:** Odblokowano flagę `ItemIsUserCheckable` dla nagłówków okien roboczych w drzewie, przywracając hurtowe włączanie/wyłączanie widoczności całych zestawów.
* **Domyślny start:** Ustawiono domyślny kontekst na Magazyn Danych z automatycznym podglądem wczytywanych widm tuż po imporcie.



### 08.09.2026 — Checkpoint: Domknięcie Silnika Chemometrii, Preprocessingu Macierzowego i Selekcji Źródeł

**1. Architektura Wyboru Danych w Chemometrii (`gui/workspaces/ws_chemometrics.py`)**
* **Inteligentny dyspozytor źródeł:** Zastąpiono sztywne listy dynamicznym selektorem ze słownikowym payloadem (`currentData()`).
* **Kontekst domyślny:** Automatyczna detekcja aktywnego środowiska — domyślnie analizowane jest aktywne Okno Robocze lub widoczne widma Magazynu Danych.
* **Wybór wieloźródłowy:** Możliwość skierowania do obliczeń:
  * Dowolnego Okna Roboczego (widoczne z checkboxami lub wszystkie przypisane),
  * Sekcji Magazynu Danych (wszystkie FTIR, widoczne, obliczenia DFT),
  * Pojedynczych Serii 2D oraz Tensorów 3D.

**2. Rozszerzenie Silnika Chemometrycznego (`algorithms/chemometrics/multivariate.py`)**
* **SVD vs. PCA:** Rozdzielenie niecentrowanego SVD (ładunki $V^T$ zachowują fizyczny kształt pasm absorbancji bez ujemnych oscylacji) od klasycznego PCA z centrowaniem do średniej.
* **Trzy tryby inspekcji dla SVD/PCA/PFA:**
  1. *Czyste składowe:* widmowe ładunki (Loadings) i profile w serii (Scores).
  2. *Scree Plot & Diagnostyka:* wartości własne (EV), błąd rzeczywisty (RE), wskaźnik IND Malinowskiego oraz wariancja [%].
  3. *Rekonstrukcja ($D_k$) & Residua ($E$):* wizualizacja odszumionego sygnału na tle oryginału z residuami szumu.
* **Integracja rekonstrukcji:** Przycisk zrzucający odszumioną macierz $D_k$ z $k$ czynników bezpośrednio do drzewa projektu jako nowy obiekt `SpectralSeries`.
* **Klasyfikacja i rzutowanie 2D:** Moduł rzutowania próbek na płaszczyznę (PC1 vs PC2, UMAP, t-SNE) z wizualizacją trajektorii czasowej/indeksowej.
* **Precyzyjny MCR-ALS:** Dodano regulację `max_iter` i `tol_err_change`, obsługę jednomodalności i domknięcia bilansu oraz raportowanie metryk dopasowania %LOF (*Lack of Fit*) i $R^2$.
* **HELP (Local Rank Profiling):** Wdrożono detekcję stref czystych składników (Rank-1) w oknie przesuwnym obok AUTOWFA i FSMWEFA.

**3. Preprocessing Macierzowy i Skalowania Chemometryczne (`algorithms/preprocessing.py`, `ws_preprocessing.py`)**
* **Silnik macierzowy potoku:** Rozdzielono przetwarzanie na `_apply_pipeline_to_matrix` ($W \times N$) oraz `_apply_pipeline_to_vectors`, co zapewnia wektorową wydajność BLAS/NumPy i umożliwia operacje na całych zbiorach.
* **Nowe transformacje:** Centrowanie do średniej (*Mean-centering*), autoskalowanie (*Unit Variance*), skalowanie Pareto oraz wielomacierzyste MSC.
* **Rozszerzenie biblioteki:** Integracja airPLS, Fourier Self-Deconvolution (FSD Kauppinena/OMNIC), normalizacji polowej $L_1$ oraz metryk gładkości (SSI, SDV).

**Plan na kolejną sesję:**
1. **Moduł Matematyki Spektralnej (Spectral Math):** Silnik arytmetyki widm ($S_1 \pm S_2$, $S \pm c$, $S \cdot c$, $S / c$, $S_1 / S_2$) dla pojedynczych widm i wsadowo z podglądem na żywo.
2. Przygotowanie dialogu narzędziowego w PySide6 z pełnym wsparciem historii `QUndoStack`.

### 08.09.2026 — Checkpoint: Nowa Perspektywa Matematyki Spektralnej (Spectral Math) & Decyzja o Korekcji ATR

**1. Nowa Perspektywa: Matematyka Spektralna (`gui/workspaces/ws_spectral_math.py`, `algorithms/spectral_math.py`)**
* **Korektor Parametryczny k (Audio Equalizer Style):** Wdrożono rządek 6 suwaków centrowanych w zerze odpowiadających rzędom wielkości: $\pm 10$, $\pm 1$, $\pm 0.1$, $\pm 0.01$, $\pm 0.001$, $\pm 0.0001$, z osobnym zerowaniem każdego pasma i synchronizacją z polem numerycznym. Umożliwia to precyzyjny dobór współczynnika odejmowania wody/rozpuszczalnika w czasie rzeczywistym (60 FPS).
* **Arytmetyka widmowa:** Dodawanie, odejmowanie, mnożenie i dzielenie widm ($S_1 \pm k \cdot S_2$, $S_1 \cdot S_2$, $S_1 / S_2$) z bezpieczną interpolacją do wspólnej siatki $X$.
* **Operacje ze skalarem:** $S \pm c$, $S \cdot c$, $S / c$.
* **Przeliczenia fotometryczne:** Konwersje $A \leftrightarrow T \leftrightarrow \%T$, transformacja Kubelki-Munka $f(R_\infty)$ dla DRIFTS oraz normalizacja na punkt odniesienia.
* **Wsparcie trybu wsadowego (Batch):** Hurtowe odejmowanie wybranego widma tła od wszystkich widocznych pomiarów.
* **Pełna historia operacji:** Wszystkie zatwierdzenia działają w oparciu o `QUndoStack`.

**2. Decyzja Strategiczna: Zaawansowana Korekcja ATR (Anomalous Dispersion / Kramers-Kronig)**
* Wykluczono prostą korekcję głębokości wnikania bazującą wyłącznie na intensywności ($\propto \lambda$).
* Zaawansowana korekcja z modelowaniem współczynnika załamania $n(\tilde{\nu})$ i relacjami Kramersa-Kroniga zostanie zrealizowana w dedykowanym kroku/perspektywie, z uwzględnieniem pojawiających się otwartych bibliotek algorytmicznych.

### Zaktualizowany wpis do `STATE.md`

```markdown
### 08.09.2026 — Checkpoint: Preprocessing Suite, Chemometria & Nowa Perspektywa Matematyki Spektralnej (Equalizer k)

**1. Nowa Perspektywa Robocza: Matematyka Spektralna (`ws_spectral_math.py`, `spectral_math.py`)**
* **Korektor Parametryczny k (Audio Equalizer Style):** Wdrożono rządek 6 suwaków centrowanych w zerze dla rzędów wielkości: $\pm 10$, $\pm 1$, $\pm 0.1$, $\pm 0.01$, $\pm 0.001$, $\pm 0.0001$, z osobnym zerowaniem każdego pasma, synchronizacją z polem numerycznym oraz domyślną wartością startową $0.0$.
* **Arytmetyka dwuwidmowa:** Dodawanie, odejmowanie, mnożenie i dzielenie ($S_1 \pm k \cdot S_2$, $S_1 \cdot S_2$, $S_1 / S_2$) z automatyczną interpolacją widma odniesienia $S_2$ do siatki $X$ widma badanego.
* **Arytmetyka ze skalarem:** Operacje $S \pm c$, $S \cdot c$, $S / c$.
* **Przeliczenia fotometryczne:** Konwersje $A \leftrightarrow T \leftrightarrow \%T$, transformacja Kubelki-Munka $f(R_\infty)$ dla spektroskopii odbiciowej DRIFTS oraz normalizacja do zadanego pasma odniesienia.
* **Tryb wsadowy (Batch):** Przetwarzanie całej serii widocznej w oknie roboczym lub magazynie ze wsparciem dla stanu wyjściowego $t_0$ (odjęcie widma odniesienia od samego siebie daje bazową linię $\Delta A \equiv 0$).
* **Izolacja interakcji i rendering:** Podgląd na żywo (60 FPS) działa wyłącznie, gdy perspektywa jest aktywna na ekranie; brak fałszywych linii na wykresie po imporcie plików i pełne wsparcie historii `QUndoStack` (zarówno dla nowych widm, jak i modyfikacji in-place).

**2. Domknięcie Silnika Chemometrii (`ws_chemometrics.py`, `multivariate.py`)**
* **Granularne trasowanie danych:** Dynamiczny dyspozytor źródeł z domyślnym wyborem aktywnego kontekstu (Okno Robocze vs Magazyn Danych vs Serie 2D / Tensory 3D).
* **Rozdzielenie SVD i PCA:** Czyste SVD zachowujące fizyczny kształt pasm absorbancji obok klasycznego PCA z centrowaniem do średniej.
* **Trzy podwidoki analityczne:** (1) Ładunki i Wyniki, (2) Scree Plot i wskaźniki Malinowskiego (EV, RE, IND, %SL), (3) Rekonstrukcja macierzy ($D_k$) z residuami szumu ($E$).
* **Zrzut odszumionych serii:** Przycisk wstrzykujący odszumioną macierz rekonstrukcji $D_k$ z $k$ czynników bezpośrednio do drzewa jako `SpectralSeries`.
* **Klasyfikacja i rzutowanie 2D:** Rzutowanie serii na płaszczyznę (PC1 vs PC2, UMAP, t-SNE) z wizualizacją trajektorii ewolucyjnej próbek.
* **MCR-ALS & EFA:** Dodano regulację zbieżności (`max_iter`, `tol_err_change`), więzy unimodalności i closure, wskaźniki dopasowania (%LOF, $R^2$) oraz profilowanie rzędu lokalnego HELP obok AUTOWFA i FSMWEFA.

**3. Preprocessing Macierzowy i Skalowania Populacyjne (`preprocessing.py`, `ws_preprocessing.py`)**
* **Silnik macierzowy potoku:** Rozdzielenie logiki na `_apply_pipeline_to_matrix` ($W \times N$) oraz `_apply_pipeline_to_vectors`, co zapewnia wektorową wydajność BLAS/NumPy.
* **Skalowania chemometryczne w potoku:** Centrowanie do średniej, autoskalowanie (Unit Variance), skalowanie Pareto oraz wielomacierzyste MSC.
* **Uzupełnienie biblioteki filtrów:** Integracja linii bazowej airPLS (airALS), dekonwolucji Fouriera (FSD Kauppinena/OMNIC), normalizacji polowej $L_1$ oraz metryk gładkości (SSI, SDV).

**4. Decyzje strategiczne:**
* Odłożenie zaawansowanej korekcji ATR (modelowanie współczynnika załamania $n(\tilde{\nu})$ i relacje Kramersa-Kroniga) na osobną, dedykowaną perspektywę w oparciu o powstające otwarte biblioteki algorytmiczne.

**Plan na kolejną sesję:**
1. Workspace Dekonwolucji (Workspace 3): audyt i dopracowanie dopasowania pasm syntetycznych (Amide I / Custom) oraz globalnego fitowania serii temperaturowych/kinetycznych.
2. Przygotowanie modułu eksportu i szablonów rycin (Pub-Studio).

```

---

### Zaktualizowane drzewo architektury SpectroHub


---

Oto gotowy wpis podsumowujący dzisiejszą sesję do wklejenia na koniec pliku `STATE.md`:


### 09.09.2026 — Checkpoint: Silnik Potoku Węzłowego (DAG Beta) & Dedykowana Baza Atmosfery

**1. Wizualny Silnik Potoku Węzłowego (`gui/node_editor.py`, DAG Beta)**
* **Architektura DAG (Directed Acyclic Graph):** Oparta o `QGraphicsScene` i `QGraphicsView` z sortowaniem topologicznym (algorytm Kahna), obsługą rozgałęzień, fuzji strumieni oraz zabezpieczeniem przed cyklami.
* **Mini-podglądy (Inline Sparklines):** Każdy bloczek rysuje w czasie rzeczywistym miniaturową krzywą absorbancji zorientowaną zgodnie ze standardem FTIR ($\tilde{\nu}$ maleje w prawo).
* **Live Preview na SpectroCanvas:** Zaznaczenie dowolnego węzła rzuca pomarańczową, przerywaną linię podglądu na żywo na główny wykres z dynamiczną aktualizacją przy zmianie parametrów.
* **Ergonomia grafu:**
  * Poszerzona strefa kliku kabli (14 px hit-box), menu kontekstowe PPM do rozłączania.
  * Usuwanie zaznaczonych węzłów/kabli klawiszem `Delete` / `Backspace`.
  * Duplikowanie bloczków skrótem `⌘D` / `Ctrl+D`, przez menu PPM oraz przeciąganie z wciśniętym `⌘` / `Ctrl`.
  * Menu kontekstowe dodawania węzłów pod PPM na pustym tle sceny.
* **Węzły wielowejściowe i matematyczne:**
  * **Operacja Dwuwidmowa (`math_op`):** Dwa porty wejściowe (`In A`, `In B`) dla operacji $A \pm k \cdot B$, $A \cdot B$, $A / B$ z automatyczną interpolacją w locie.
  * **Operacje ze skalarem (`scalar_math`):** $S \pm c$, $S \cdot c$, $S / c$ z walidacją dzielenia przez zero.
  * **Widmo Odniesienia z Projektu (`spectrum_select`):** Pobieranie dowolnego widma z bazy pomiarowej lub atmosferycznej.
  * **Węzeł VaporFit (`vaporfit`):** Porty `Syg` oraz `Atm` (obsługujący multi-kabel lub auto-pobieranie z bazy atmosfery), analityczny OLS, walidacja SDV i okno auto-tuningu SG.
* **Integracja z QUndoStack:** Obsługa wykonania potoku dla pojedynczego widma, hurtowo (Batch) oraz na seriach 2D / tensorach 3D z pełnym cofaniem operacji.

**2. Dedykowana Baza Wzorców Atmosfery & Stabilizacja Drzewa (`core/session.py`, `gui/project_tree.py`)**
* **Izolacja magazynowa:** Rozdzielenie pomiarów eksperymentalnych od widm odniesienia pary wodnej i $\text{CO}_2$ (`_atm_spectra`).
* **Selektywny auto-routing przy imporcie:** Rozpoznawanie wzorców wyłącznie po znacznikach `atm`, `ATM`, `co2`, `CO2` (wykluczenie `h2o` zapobiega błędnej kategoryzacji roztworów wodnych i hydratów).
* **Nowa gałąź w drzewie:** `☁️ Wzorce Atmosfery (ATM / CO₂)` obok `📈 Pomiary Eksperymentalne`.
* **Przenoszenie w locie (PPM):** Menu kontekstowe umożliwiające przenoszenie pojedynczych lub wielu widm między gałęziami Magazynu.
* **Pancerne znaczniki (Checkboxy):** Wyeliminowano błędy propagacji stanu zaznaczeń; nagłówki kategorii poprawnie kontrolują widoczność dzieci bez gubienia stanu pośredniego.
* **Kompatybilność wsteczna:** Przywrócono metody `get_visible_spectra()` oraz dodano `get_atm_matrix()` interpolujące wzorce wprost do żądanej siatki.

**3. Odświeżenie Dialogu Hurtowego VaporFit (`ws_preprocessing.py`)**
* Dostosowano okno dialogowe do nowej architektury bazy atmosfery: lewa lista pobiera wyłącznie próbki badane, a prawa ładuje wzorce z nowej gałęzi (zaznaczone domyślnie), eliminując potrzebę ręcznego odznaczania.

---

### Zaktualizowane drzewo architektury SpectroHub


```text
spectrohub/
├── main.py                          # Punkt wejścia (Qt logging filter, obsługa SIGINT)
├── STATE.md                         # Checkpoint stanu projektu i pamięć między sesjami
├── PRD.md                           # Wymagania i specyfikacja architektury platformy
│
├── core/                            # Rdzeń stanu, modeli i infrastruktury
│   ├── __init__.py
│   ├── models.py                    # SpectrumData, GaussianData, SpectralSeries, DeconvPeak
│   ├── session.py                   # ProjectSession (Magazyn, Wzorce Atm, Okna Robocze, sygnały, Undo)
│   ├── i18n.py                      # System wielojęzyczności (PL / EN)
│   └── logger.py                    # Konfiguracja logowania systemowego
│
├── algorithms/                      # Czyste moduły matematyczne i analityczne
│   ├── __init__.py
│   ├── preprocessing.py             # AsLS, airPLS, FSD, SavGol, SNV, Min-Max, L1, L2, Pareto, MSC
│   ├── spectral_math.py             # Arytmetyka widm, skalary, fotometria (A/T/%T/KM)
│   ├── atmospheric.py               # VaporFit Core (analityczny OLS, SDV, SSI, okna SG)
│   ├── deconvolution/               # Silnik dopasowania pasm (Amide I, Custom, HDO Monte Carlo)
│   ├── quantum/                     # Chemia obliczeniowa (PED, parsery Gaussian / ORCA)
│   └── chemometrics/                # SVD, PCA, PFA, AUTOWFA, SPEXFA, MCR-ALS, HELP, PARAFAC, 2D-COS
│
├── io_hub/                          # Warstwa Wejścia / Wyjścia (I/O)
│   ├── __init__.py
│   ├── dispatcher.py                # SpectrumImportDispatcher (automatyczne trasowanie formatów)
│   ├── readers/                     # Czytniki (OPUS, OMNIC, SPC, CSV/DPT, JCAMP-DX)
│   └── exporters/                   # Eksporterzy (Excel wieloarkuszowy, ASCII, SPC, JCAMP-DX)
│
├── gui/                             # Warstwa prezentacji PySide6
│   ├── __init__.py
│   ├── main_window.py               # Główne okno, doki, toolbar perspektyw (1-5), menu główne
│   ├── canvas.py                    # SpectroCanvas (Matplotlib, Smart Zoom, LineCollection, kinetyka)
│   ├── project_tree.py              # ProjectTreeWidget (Magazyn, Wzorce Atm, Okna Robocze)
│   ├── node_editor.py               # Wizualny edytor potoku DAG (Bézier, sparklines, multi-input, Undo)
│   ├── vibration_dialog.py          # Pływające okno 3D animacji modów drgań (PyVista)
│   ├── tensor_builder_dialog.py     # Kreator serii kinetycznych 2D i tensora 3D
│   └── workspaces/                  # Dedykowane panele robocze
│       ├── ws_preprocessing.py      # Workspace 1: Pipeline Manager, przycisk DAG Beta, Batch VaporFit
│       ├── ws_spectral_math.py      # Workspace 2: Equalizer k, operacje arytmetyczne, fotometria
│       ├── ws_deconvolution.py      # Workspace 3: Fitowanie Amidu I, tryb Custom, dekompozycja HDO
│       ├── ws_chemometrics.py       # Workspace 4: MCR-ALS, SPEXFA, SVD/PCA, rekonstrukcje D_k
│       └── ws_quantum.py            # Workspace 5: Inspekcja DFT, PED, splot pasm i podgląd 3D
│
└── locales/                         # Słowniki językowe
    ├── pl.py
    └── en.py

```

---

**Plan na kolejną sesję:**

1. **Zapis i odczyt szablonów potoku węzłowego:** Serializacja grafu do formatu JSON (pozycje nodów, parametry, połączenia) oraz import gotowych presetów.
2. **Workspace Dekonwolucji (Workspace 3):** Audyt dopasowania pasm syntetycznych (Amide I / Custom) oraz globalnego fitowania serii temperaturowych/kinetycznych.
3. **Moduł rycin publikacyjnych (Pub-Studio):** Generowanie paneli 8.5 / 17.5 cm pod wymogi wydawnictw (600 DPI).

### 09.09.2026 — Checkpoint: Format Projektowy .shub, Trwały Stan DAG i Modularyzacja Silnika Węzłowego

**1. Architektura Projektowa SpectroHub (.shub Bundle)**
* **Format kontenerowy ZIP (.shub):** Bezpieczna separacja struktury (`project.json`) oraz danych numerycznych w pełnej precyzji `float64` (`data.npz` skompresowane w locie).
* **Model Metadanych (`ProjectMetadata`):** Autor, daty utworzenia i modyfikacji, wariant opracowania oraz wielolinijkowy dziennik/notatki eksperymentalne zintegrowane w `session.py` i odzwierciedlone w tytule okna głównego.
* **Inspektor Metadanych (`ProjectMetadataWidget`):** Nowy, zintegrowany widget umieszczony pod drzewem projektu w pionowym splitterze lewego doku.

**2. Pakietyzacja i Rozbudowa Silnika Węzłowego (`gui/node_editor/`)**
* **Refaktoryzacja modularna:** Rozbicie monolitu na pakiet `gui/node_editor/` (`items.py`, `dialogs.py`, `evaluator.py`, `pipeline_dialog.py`).
* **Trwałość stanu grafu (Persistence):** Stan potoku jest na stałe zsynchronizowany z `session.dag_state` — zamknięcie okna krzyżykiem nie kasuje układu nodów, a otwarcie projektu `.shub` wznawia pracę z zachowaną geometrią połączeń.
* **Architektura hybrydowa i brak modalności:** Okna narzędziowe (`VaporFitDialog`, `NodePipelineDialog`) działają w trybie niemodalnym (`Qt.Window`), umożliwiając interaktywne badanie wykresu na `SpectroCanvas` podczas konfiguracji potoku.
* **Nowe węzły i architektura Multi-Output:**
  * **Wielowyjściowe linie bazowe (AsLS, airPLS, Poly):** Wyjścia `Syg` (skorygowany) oraz `BL` (wyestymowane tło).
  * **Wielowyjściowy VaporFit:** Wyjścia `Syg` oraz `Atm` (dopasowany profil par H₂O/CO₂).
  * **Konwerter Fotometryczny (`photometry`):** Przeliczenia $A \leftrightarrow T \leftrightarrow \%T$ oraz formalizm Kubelki-Munka DRIFTS $f(R_\infty)$.
  * **Splot widm DFT (`dft_broadener`):** Generowanie gładkich profili z pałeczek drgań obliczeń kwantowych (Gauss/Lorentz, regulacja $\sigma$ i FWHM) do bezpośredniego porównywania i odejmowania od eksperymentu.
  * **Terminal Nowego Okna (`output_window`):** Kierowanie wyników potoku bezpośrednio do nowo utworzonych lub istniejących Okien Roboczych.
* **Menedżer Szablonów (Presets Manager):** Wbudowane procedury analityczne (Kompensacja ATM + AsLS, Amid I, 2. pochodna, spektroskopia różnicowa, porównanie z DFT) oraz eksport/import szablonów grafu do zewnętrznych plików `.json`.


### 09.09.2026 — Checkpoint: Format Projektowy .shub, Architektura Hybrydowa DAG, Chemometria i Punkty Kontrolne

**1. Natywny Format Projektu SpectroHub (`.shub` Bundle)**

* **Architektura kontenerowa:** Bezpieczny pakiet ZIP zawierający manifest struktury, parametrów i grafu (`project.json`) oraz zoptymalizowany, binarny zrzut tablic NumPy w pełnej precyzji `float64` (`data.npz`).
* **Zarządzanie Metadanymi (`ProjectMetadata`):** Rejestracja tytułu projektu, autora, wariantu opracowania (np. *Surowe*, *Kompensacja ATM*, *Publikacja*), dat utworzenia/modyfikacji oraz notatek eksperymentalnych zintegrowana w `core/session.py`.
* **Nowy panel metadanych (`ProjectMetadataWidget`):** Kompaktowy, zintegrowany widget umieszczony w pionowym splitterze lewego doku pod drzewem projektu.
* **Integracja z oknem głównym:** Skróty klawiszowe (`⌘S` / `Ctrl+S`, `⌘Shift+S`, `⌘O`, `⌘N`) oraz automatyczna aktualizacja paska tytułu programu.

**2. Pakietyzacja i Architektura Hybrydowa Silnika Węzłowego (`gui/node_editor/`)**

* **Modularyzacja kodu:** Rozbicie monolitycznego pliku na pakiet `gui/node_editor/` (`items.py`, `dialogs.py`, `evaluator.py`, `pipeline_dialog.py`).
* **Trwałość stanu grafu (Persistence):** Pełna synchronizacja geometrii, parametrów i połączeń z `session.dag_state`. Zamknięcie okna edytora nie kasuje schematu, a zapis projektu utrwala potok w pliku `.shub`.
* **Tryb niemodalny (`Qt.Window`):** Okna narzędziowe (`NodePipelineDialog`, `VaporFitDialog`) nie blokują okna głównego, umożliwiając jednoczesną nawigację, powiększanie pasm i smart zoom na `SpectroCanvas`.


* **Reaktywny Live Preview:** Dynamiczne przeliczanie miniatur w bloczkach oraz podgląd na żywo na głównym wykresie podczas modyfikacji suwaków parametrów lub zmiany aktywnego widma w drzewie.


* **System Szablonów Potoków (Presets Manager):** Zestaw wbudowanych procedur laboratoryjnych (m.in. Kompensacja ATM + AsLS, Amid I, 2. Pochodna, Różnicowa spektroskopia wodna, porównanie z DFT) oraz eksport/import potoków do plików `.json`.

**3. Chemometria i Węzły Wielowyjściowe (Multi-Output & Taps)**

* **Węzeł Odszumiania SVD / PCA Rank-$k$ (`svd_denoise`):** Dwa wyjścia – odszumiony sygnał rekonstrukcji ($D_k$) oraz czyste residuum szumu ($E$) wyprowadzane na osobny kabel.


* **Węzeł Dekompozycji MCR-ALS / SPEXFA (`mcr_decompose`):** Trzy wyjścia – wyizolowane widma czystych składników ($S^T$), profile stężeń/rekonstrukcja oraz residua dopasowania z obsługą więzów (NNLS, closure, unimodal).


* **Węzeł Skalowań Populacyjnych (`matrix_scaling`):** Centrowanie do średniej, autoskalowanie (UV), skalowanie Pareto oraz wielomacierzyste MSC.
* **Węzeł Splotu DFT (`dft_broadener`):** Generowanie gładkich profili z pałeczek drgań obliczeń kwantowych (Gauss/Lorentz, regulacja $\sigma$ i FWHM) do bezpośredniego odejmowania od eksperymentu na żywo na ekranie.
* **Węzeł Konwersji Fotometrycznych (`photometry`):** Przeliczenia $A \leftrightarrow T \leftrightarrow \%T$ oraz formalizm Kubelki-Munka DRIFTS $f(R_\infty)$.
* **Terminal jako Punkt Kontrolny (`output_window`):** Kierowanie wyników do wskazanego lub nowo tworzonego Okna Roboczego z zachowaniem portu wyjściowego `Out` (Pass-through / Tap) do dalszego potokowego przetwarzania danych.

**4. Ścisła Izolacja Gałęzi Magazynu Danych**

* **Izolacja kontekstowa (`session.get_pipeline_target_spectra()`):** Rozdzielenie pomiarów eksperymentalnych od wzorców atmosfery. Wybór opcji wsadowej przetwarza wyłącznie widma z aktywnego okna roboczego lub aktualnie wybranej gałęzi magazynu (eliminacja przypadkowego mieszania wzorców z próbkami).


* **Domyślny cel potoku:** Ustawiono domyślny zakres na *„Wszystkie widoczne w aktywnym oknie / gałęzi”* (indeks 0).

---

### 09.09.2026 — Checkpoint: Dokowalna Biblioteka Węzłów, Drag & Drop i Ergonomia Grafu DAG

**1. Architektura Dokowalna Edytora Węzłów (`QMainWindow` + `QDockWidget`)**
* **Migracja edytora DAG:** Zastąpiono `QDialog` pełnoprawną strukturą `QMainWindow` z widokiem `NodeView` jako centralnym widgetem i obsługą niemodalną (`Qt.Window`)[cite: 1, 6].
* **Dedykowany moduł palety (`gui/node_editor/palette.py`):** Wyodrębniono komponenty `NodeLibraryWidget` oraz `NodeLibraryTree`, eliminując duplikację kodu i porządkując przestrzeń nazw pakietu `node_editor`[cite: 1, 2].
* **Lewy dok biblioteki węzłów (`DockNodeLibrary`):**
  * Hierarchiczne, zwijane drzewo wszystkich dostępnych operacji (Chemometria, DFT, Fotometria, Filtry, Baza, Matematyka)[cite: 6].
  * Wyszukiwarka na żywo (`QLineEdit`) z dynamicznym filtrowaniem i automatycznym rozwijaniem pasujących gałęzi.
  * Przycisk przełączania widoczności doku na pasku narzędzi (`toggleViewAction`).

**2. Płynna Ergonomia Myszy, Drag & Drop i Nawigacja Grafu**
* **Obsługa Drag & Drop:** Zaimplementowano natywne przeciąganie bloczków z drzewa biblioteki bezpośrednio na siatkę grafu w oparciu o format MIME (`application/x-spectrohub-node`), wstawiające węzeł dokładnie pod kursorem myszy.
* **Dwuklik w drzewie:** Natychmiastowe wstawienie wybranego węzła w widocznym centrum viewportu.
* **Automatyczne kadrowanie (Zoom to Fit):**
  * Wdrożono metodę `fit_to_content()` z dynamicznym marginesem (60 px) oraz zabezpieczeniem przed nadmiernym zoomem przy małej liczbie węzłów.
  * Domyślny start potoku (`showEvent` z `QTimer`) automatycznie centruje i kadruje wszystkie startowe bloczki, eliminując problem obciętego widoku.
  * Skrót klawiszowy `F` oraz dedykowany przycisk na toolbarze (`⛶ Dopasuj widok (F)`) centrujące graf w dowolnym momencie.
  * Automatyczne dopasowanie widoku po zresetowaniu grafu lub załadowaniu szablonu `.json`[cite: 6].
* **Panoramowanie widoku (Pan):** Płynne przesuwanie kamery środkowym przyciskiem myszy (ŚPM) lub kombinacją `Alt + LPM` (zoptymalizowane pod gładziki i laptopy).

**3. Trwałość Układu Ekranu i Okna (`QSettings`)**
* **Pamięć geometrii okna:** Zintegrowano zapis i odczyt `saveGeometry()` oraz `saveState()` w rejestrze systemowym (`QSettings("SpectroHub", "NodeEditor")`).
* **Rozdzielenie stanu:**
  * `.shub` / JSON: przechowuje logikę analityczną, współrzędne nodów, połączenia i parametry fizykochemiczne[cite: 1, 6].
  * `QSettings`: zapamiętuje wymiary okna, pozycję na monitorze oraz szerokość/stan zadokowania lewego panelu.

---

### Zaktualizowane drzewo architektury SpectroHub

```text
spectrohub/
├── main.py                          # Punkt wejścia (Qt logging filter, obsługa SIGINT, wyciszenie dostępności macOS)
├── STATE.md                         # Checkpoint stanu projektu i pamięć między sesjami
├── PRD.md                           # Wymagania i specyfikacja architektury platformy[cite: 1]
│
├── core/                            # Rdzeń stanu, modeli i infrastruktury[cite: 1]
│   ├── __init__.py[cite: 1]
│   ├── models.py                    # SpectrumData, GaussianData, SpectralSeries, DeconvPeak, ProjectMetadata[cite: 1]
│   ├── session.py                   # ProjectSession (Magazyn, Izolacja Gałęzi, Okna, dag_state, Undo)[cite: 1, 7]
│   ├── i18n.py                      # System wielojęzyczności (PL / EN)[cite: 1]
│   └── logger.py                    # Konfiguracja logowania systemowego[cite: 1]
│
├── algorithms/                      # Czyste moduły matematyczne i analityczne[cite: 1]
│   ├── __init__.py[cite: 1]
│   ├── preprocessing.py             # AsLS, airPLS, FSD, SavGol, SNV, Min-Max, L1, L2, Pareto, MSC[cite: 1, 4]
│   ├── spectral_math.py             # Arytmetyka widm, skalary, fotometria (A/T/%T/KM)[cite: 1]
│   ├── atmospheric.py               # VaporFit Core (analityczny OLS, SDV, SSI, okna SG)[cite: 1, 3, 4]
│   ├── deconvolution/               # Silnik dopasowania pasm (Amide I, Custom, HDO Monte Carlo)[cite: 1]
│   ├── quantum/                     # Chemia obliczeniowa (PED, parsery Gaussian / ORCA)[cite: 1]
│   └── chemometrics/                # multivariate.py (PCA, SVD, PFA, SPEXFA, AUTOWFA, FSMWEFA, HELP, MCR-ALS, PARAFAC, 2D-COS)[cite: 1, 3, 4]
│
├── io_hub/                          # Warstwa Wejścia / Wyjścia (I/O)[cite: 1]
│   ├── __init__.py[cite: 1]
│   ├── dispatcher.py                # SpectrumImportDispatcher (automatyczne trasowanie formatów)[cite: 1]
│   ├── project_io.py                # ProjectBundleIO (Silnik zapisu i odczytu kontenerów .shub / ZIP)[cite: 1]
│   ├── readers/                     # Czytniki (OPUS, OMNIC, SPC, CSV/DPT, JCAMP-DX)[cite: 1]
│   └── exporters/                   # Eksporterzy (Excel wieloarkuszowy, ASCII, SPC, JCAMP-DX)[cite: 1]
│
├── gui/                             # Warstwa prezentacji PySide6[cite: 1]
│   ├── __init__.py[cite: 1]
│   ├── main_window.py               # Główne okno, doki ze splitterem, toolbar perspektyw (1-5), obsługa .shub[cite: 1]
│   ├── canvas.py                    # SpectroCanvas (Matplotlib, Smart Zoom, LineCollection, kinetyka)[cite: 1]
│   ├── project_tree.py              # ProjectTreeWidget (Gałęzie Magazynu, Wzorce Atm, Okna Robocze)[cite: 1]
│   ├── project_metadata_widget.py   # ProjectMetadataWidget (Inspektor metadanych, autor, daty, notatki)[cite: 1]
│   ├── vibration_dialog.py          # Pływające okno 3D animacji modów drgań (PyVista)[cite: 1]
│   ├── tensor_builder_dialog.py     # Kreator serii kinetycznych 2D i tensora 3D[cite: 1]
│   │
│   ├── node_editor/                 # Pakiet Wizualnego Silnika Potoku Węzłowego (DAG)[cite: 1]
│   │   ├── __init__.py              # Re-eksport NodePipelineDialog[cite: 1, 2]
│   │   ├── palette.py               # NodeLibraryWidget, NodeLibraryTree, filtrowanie, Drag & Drop
│   │   ├── items.py                 # Grafika Qt: PortItem, ConnectionItem, NodeItem (Sparklines, Hit-box)[cite: 1, 5]
│   │   ├── dialogs.py               # NodeParamDialog (Fotometria, DFT, Chemometria, Okna, VaporFit, Math)[cite: 1, 3]
│   │   ├── evaluator.py             # DAGEvaluator (Ewaluacja DAG 1D/2D, routing wieloportowy, SVD, MCR)[cite: 1, 4]
│   │   └── pipeline_dialog.py       # NodePipelineDialog (QMainWindow, doki, QSettings, Fit-to-Content, Presety)[cite: 1, 6]
│   │
│   └── workspaces/                  # Dedykowane panele robocze[cite: 1]
│       ├── ws_preprocessing.py      # Workspace 1: Pipeline Manager, wywołanie edytora DAG, Batch VaporFit[cite: 1]
│       ├── ws_spectral_math.py      # Workspace 2: Equalizer k, operacje arytmetyczne, fotometria[cite: 1]
│       ├── ws_deconvolution.py      # Workspace 3: Fitowanie Amidu I, tryb Custom, dekompozycja HDO[cite: 1]
│       ├── ws_chemometrics.py       # Workspace 4: MCR-ALS, SPEXFA, SVD/PCA, rekonstrukcje D_k[cite: 1]
│       └── ws_quantum.py            # Workspace 5: Inspekcja DFT, PED, splot pasm i podgląd 3D[cite: 1]
│
└── locales/                         # Słowniki językowe[cite: 1]
    ├── pl.py[cite: 1]
    └── en.py[cite: 1]
```

### 09.09.2026 — Checkpoint: Determinizm VaporFit, Wieloźródłowy Węzeł Odniesienia i Wielozakładkowy Silnik DAG (Multi-Tab & Floating)

**1. Naprawa Silnika Kompensacji Atmosferycznej (VaporFit Core & Batch Dialog)**
* **Determinizm algebry liniowej OLS:** Wyeliminowano z `algorithms/atmospheric.py` fałszywą heurystykę auto-detekcji znaku SDV (`cand_plus`), która w obecności silnych pasm próbki sztucznie odwracała znak korekty i podwajała oscylacje pary wodnej.
* **Czyste odejmowanie $S_{\text{corr}} = S - A \cdot c$:** Przywrócono logikę oryginalnego VaporFit — wektor współczynników OLS zachowuje właściwy znak algebraiczny (nadmiar pary wodnej w absorbancji daje $c > 0$ i odejmuje pasma, deficyt daje $c < 0$ i kompensuje dołki).
* **Pełna kontrola zakresu:** Zachowano w 100% manualny dobór granic liczb falowych w GUI (użytkownik decyduje o oknie dopasowania: rejon rozciągający OH, zginający OH lub pasmo $\text{CO}_2$ przy $2350\text{ cm}^{-1}$).
* **Porządki w `VaporFitDialog`:** Usunięto mylący przełącznik kierunku; parametry SG i okna są bezpośrednio przekazywane do silnika obliczeniowego.

**2. Wieloźródłowy Węzeł Odniesienia (`spectrum_select`) i Routing Macierzowy**
* **Multi-selekcja w grafie DAG:** Zastąpiono pojedynczy `QComboBox` w `NodeParamDialog` listą `QListWidget` z checkboxami oraz przyciskami szybkiego wyboru (*Zaznacz wszystkie ATM* / *Odznacz*).
* **Macierz 2D jednym kablem:** Węzeł `spectrum_select` pakuje wybrane wzorce do wspólnej siatki i przesyła je jako macierz $W \times K$ przez pojedyncze połączenie.
* **Aktualizacja `evaluator.py`:** Węzeł `vaporfit` obsługuje wejście wielokolumnowe z kabla, zachowując bezprzewodowy tryb domyślny (*Zero-Wire*) – przy braku podłączenia pod `Atm` automatycznie pobiera zaznaczone wzorce z dedykowanej bazy atmosfery.
* **Korekta geometrii dialogu parametrów:** Zoptymalizowano wymiary okna `NodeParamDialog` (460x380 px), gwarantując pełną widoczność przycisków `QDialogButtonBox` (OK / Anuluj).

**3. Wielozakładkowy Edytor Potoku DAG (Multi-Tab & Floating Windows)**
* **Architektura kart (Multi-Tab):** Zastąpiono pojedynczy widok strukturą `QTabWidget`, umożliwiając jednoczesną pracę nad wieloma niezależnymi potokami (np. referencyjny VaporFit, Amid I, brudnopis testowy).
* **Wyrównanie do lewej (macOS Fix):** Zastosowano `LeftAlignedTabStyle` (`QProxyStyle` opakowujący styl arkusza i nadpisujący `SH_TabBar_Alignment`), co trwale eliminuje wymuszone przez macOS centrowanie kart.
* **Wydzielanie do pływających okien (Detach / Tear-off):** Wdrożono `DetachableTabBar` umożliwiający odrywanie zakładek przez przeciągnięcie poza pasek lub kliknięcie w menu kontekstowym PPM; pływające okno (`FloatingPipelineWindow`) posiada własny toolbar i przycisk przypięcia z powrotem.
* **Menu kontekstowe zakładek:** Zmiana nazwy (dwuklik / PPM), duplikacja potoku, wydzielenie oraz zamykanie.
* **Nieniszczące szablony (Presets):** Wbudowane procedury analityczne oraz import plików `.json` otwierają się w nowych zakładkach bez nadpisywania bieżącego schematu.
* **Trwałość stanu v2:** Rozbudowano `export_graph_state` i `import_graph_state` o strukturę wielozakładkową w `.shub` i `session.dag_state` z pełną kompatybilnością wsteczną dla starych projektów jednografowych.

---

**Plan na kolejną sesję:**
1. **Workspace Dekonwolucji (Workspace 3):** Audyt dopasowania pasm syntetycznych (Amide I / Custom) oraz globalnego fitowania serii temperaturowych/kinetycznych.
2. **Moduł rycin publikacyjnych (Pub-Studio):** Projekt generatora paneli jedno- i dwukolumnowych (8.5 / 17.5 cm, 600 DPI) z wektorowym eksportem PDF/SVG/TIFF.


### 10.09.2026 — Checkpoint: Architektura Multi-Input w DAG, Profile Stężeń Chemometrii i Nowe Nody Analityczne

**1. Wielowidmowe Mini-Podglądy (Inline Sparklines 1–3 Widm)**
* **Reprezentacja serii w grafie:** Bloczek węzła dynamicznie rysuje 1, 2 lub maksymalnie 3 krzywe FTIR (początek serii $t_0$, środek $t_{\text{mid}}$ oraz koniec $t_{\text{end}}$)[cite: 3].
* **Wspólne skalowanie intensywności:** Wszystkie krzywe w danym węźle są znormalizowane do wspólnego zakresu $[y_{\min}, y_{\max}]$, co zapewnia poprawną relację amplitud[cite: 3].
* **Trójstopniowa paleta kolorystyczna:** Cyjan (`#48cae4`), żółty (`#ffd166`) i koralowy (`#ff4d6d`) z dyskretną etykietą liczności serii (np. `3/50` lub `2 widma`)[cite: 3].
* **Wieloliniowy Live Preview na Canvasie:** Zaznaczony węzeł rzuca komplet 1–3 przerywanych linii na główny wykres z tą samą paletą[cite: 5].

**2. Węzły Końcowe i Architektura Fan-In (Multi-Input)**
* **Kopiuj do okna roboczego (`output_window`):** Port wejściowy `In` zyskał flagę `allow_multiple=True` — przyjmuje kable z wielu gałęzi jednocześnie[cite: 3]. Podczas wykonania scala wszystkie gałęzie na jednym wykresie[cite: 5].
* **Deduplikacja nazw widm:** Wprowadzono automatyczne nadawanie sufiksów gałęzi (np. `Próbka [AsLS]`, `Próbka [airPLS]`) oraz liczników unikalności `(2)`, eliminując konflikty identyfikatorów w oknach roboczych[cite: 5].
* **Swobodny węzeł „Zapis wyniku” (`output`):** Odblokowano usuwanie klawiszem `Delete`, powielanie (`⌘D` / `Ctrl+D`) oraz menu kontekstowe — węzeł nadpisywania jest w pełni opcjonalny[cite: 3, 5].

**3. Nowe Węzły i Profile Chemometryczne w DAG**
* **Profile stężeniowe $C$ (MCR-ALS):** Nowy port `out_conc` w węźle `mcr_decompose` wyprowadza macierz stężeń $C$ próbek $1 \dots N$[cite: 7].
* **Ładunki próbek (Scores $T$ w PCA/SVD):** Nowy port `out_scores` w węźle `svd_denoise` przekazuje wektor lub macierz rzutów na składowe główne[cite: 7].
* **Węzeł Mux / Zbieracz widm (`merge_spectra`):** Łączy sygnały z wielu gałęzi i źródeł w jedną macierz dwuwymiarową ($W \times N$) jednym kablem zasilającym[cite: 3].
* **Wewnętrzny Standard (`internal_standard`):** Dwa wejścia (`Syg` i `IS`) z automatyczną normalizacją do wybranego pasma wzorca wewnętrznego (wysokość lub pole piku)[cite: 3].
* **Całkowanie Pasma / Kinetyka (`band_integrate`):** Całkowanie w zadanym oknie liczb falowych z lokalną korektą tła (krańcową lub offsetem) — dla serii generuje gotowy profil kinetyczny[cite: 8].
* **Reorganizacja Biblioteki:** Utworzono nadrzędną kategorię *„Źródła Danych & Odniesienia”* dla węzła wieloselekcji widm projektu i zbieracza Mux[cite: 4].


### 10.09.2026 — Checkpoint: Gałąź Dekompozycji HDO / MC w DAG, Stabilizacja NumPy 2.x i Architektura Data-Flow

**1. Migracja do NumPy 2.x i Porządki w Sygnałach Qt**
* **Eliminacja błędu `np.trapz`:** Zastąpiono wycofaną funkcję uniwersalnym aliasem `_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))` w `gui/node_editor/evaluator.py`, przywracając stabilność zdarzeń myszy i klawiatury w Qt na Pythonie 3.13 / NumPy 2.x.
* **Uciszenie ostrzeżeń paska narzędzi:** Nadano unikalne `setObjectName` dla toolbarów głównego i pływającego w edytorze nodów, eliminując komunikaty `QMainWindow::saveState()`.

**2. Dedykowana Gałąź Dekompozycji HDO & Pasma Syntetyczne w DAG (`gui/node_editor/`)**
* **Architektura dedykowanych węzłów:**
  * **`hdo_decomposition` (Solwer HDO / MC):** Wieloportowy solwer łączący widmo badane $W_0$ (z mnożnikiem $N$), widma bazowe $W_k$, piki syntetyczne oraz konfigurację MC. Wyprowadza osobno: całkowity fit (`out_fit`), czyste residuum nadmiarowe (`out_res`), residuum dopasowania (`out_fit_res`), macierz dopasowanych składowych (`out_comps`) oraz profile pasm (`out_peaks`).
  * **`hdo_spectrum_scaled` (Widmo Składowe $W_k \cdot c$):** Dedykowany bloczek łączący wybór widma z projektu, skalowanie oraz granice dopuszczalnych wag w optymalizacji (`w_min`, `w_max`).
  * **`synthetic_peak` (Generator Pasma):** Obsługa profili `GxL` (Gauss × Lorentz), `Gaussian`, `Lorentzian`, `Voigt` oraz `Pseudo-Voigt` z regulacją położenia $\nu_0$, amplitudy, szerokości oraz parametru kształtu $\eta$.
  * **`mc_params` (Parametry Monte Carlo):** Dedykowany węzeł sterujący liczbą iteracji, współczynnikiem asymetrycznej kary za przekraczanie widma badanego oraz kwadratową linią bazową.
* **Eliminacja redundancji nastaw:** Usunięto zduplikowane parametry MC z solwera – węzeł `hdo_decomposition` czerpie nastawy optymalizacji wprost z podłączonego klocka `Parametry MC`.
* **Gotowy szablon laboratoryjny:** Dodano pozycję *„🧪 Dekompozycja HDO / Monte Carlo (W₀ - k·W₁ + Pasma GxL)”* do menu szablonów potoków, budującą pełny schemat jednym kliknięciem.

**3. Wydajność Podglądu na Żywo i Ochrona Pętli Zdarzeń Qt (60 FPS)**
* **Dwutrybowy silnik ewaluacji (`DAGEvaluator`):**
  * *Tryb Podglądu (`preview`):* Zastąpiono ciężki `ProcessPoolExecutor` błyskawicznym, jednowątkowym dopasowaniem OLS w ułamku milisekundy (`max_nfev=35`). Wyeliminowano dławienie pętli zdarzeń Qt i zacinanie UI przy przesuwaniu węzłów na macOS.
  * *Tryb Wykonania (`execute`):* Pełna pula procesów Monte Carlo (`HDOFitEngine.run_monte_carlo`) startuje wyłącznie po fizycznym kliknięciu przycisku uruchomienia.
* **Bezpieczeństwo numeryczne (`hdo_engine.py`):** Zabezpieczono funkcję `gxl_peak` przed przepełnieniem zmiennoprzecinkowym w `np.exp` (`np.clip(..., -100.0, 0.0)`), eliminując `RuntimeWarning: overflow encountered in exp`.
* **Pasek postępu (`QProgressDialog`):** Wdrożono modalne okno postępu z dynamicznym raportem procentowym, śledzeniem najlepszego wskaźnika RSS oraz możliwością bezpiecznego przerwania obliczeń przyciskiem *Anuluj*.
* **Domyślny stan podglądu:** Opcja *„Live Preview na Canvasie”* jest domyślnie odznaczona, co zapobiega zaśmiecaniu wykresu głównego podczas edycji grafu.

**4. Przejście na Pełny Model Data-Flow (Konfigurowalny Węzeł Wejściowy)**
* **Usunięcie sztywnego ComboBoxa z toolbara:** Zlikwidowano globalny selektor „Cel potoku” i checkbox „In-place” na rzecz sterowania bezpośrednio w źródle danych.
* **Węzeł Startowy (`input`):** Dwuklik na klocek źródła otwiera okno wyboru trybu wprowadzania danych:
  1. *Tylko aktywne widmo (1D)* – domyślne dla analiz pojedynczych próbek (HDO, dekonwolucja).
  2. *Konkretne widmo z projektu (1D)* – sztywne przypisanie wskazanego pomiaru.
  3. *Wszystkie widoczne w oknie / gałęzi (Batch 2D)* – tryb wsadowy.
  4. *Aktywna seria kinetyczna / Tensor*.

**5. Czytelne Raportowanie Wyników Fitowania**
* **Kodowanie wag w nazwach widm:** Wygenerowane składowe i residua otrzymują automatycznie w nazwach wyznaczone współczynniki i jakość dopasowania, np. `Woda (c=0.8412)`, `Fit HDO (R²=0.9984)`, `Residuum HDO (R²=0.9984)`.
* **Mini-raport na kafelku solwera:** Bloczek `hdo_decomposition` w edytorze wyświetla w oknie podglądu wartości $R^2$ oraz współczynniki $c_1, c_2, \dots$ w czasie rzeczywistym.
* **Metadane obiektów:** Wszystkie parametry dopasowania trafiają do słownika `metadata` widm, zapewniając ich trwałość w plikach `.shub`.


### 11.09.2026 — Checkpoint: Hierarchia Folderów, Eksport z Drzewa i Elastyczne Nazewnictwo w Potoku DAG

**1. Hierarchiczna Organizacja Okien Roboczych & Eksport z Drzewa (`project_tree.py`, `session.py`, `models.py`)**
* **Model `WindowFolder`:** Reprezentacja wielopoziomowej struktury katalogów na okna robocze z pełnym wsparciem drzewiastym (`parent_folder_id`) i głęboką serializacją w projekcie.
* **Zarządzanie strukturą w `ProjectSession`:** Metody `add_folder()`, `remove_folder()` (z flagą `delete_windows` decydującą o usunięciu lub przeniesieniu okien wyżej), `rename_folder()` oraz `move_window_to_folder()`.
* **Ergonomia Drag & Drop w eksploratorze:**
* Przeciąganie okien między folderami oraz na poziom nadrzędny.
* Zagnieżdżanie podfolderów wewnątrz folderów.
* Przeciąganie pojedynczych widm i zaznaczeń hurtowych bezpośrednio na foldery (z automatycznym routingiem do pierwszego okna lub utworzeniem nowego).
* **Eksport i Zapis z Menu Kontekstowego (`ExportManager`):**
* Szybki zapis pojedynczego widma lub całej grupy w ostatnio wybranym formacie.
* Eksport wsadowy do katalogu na dysku z pełnym odwzorowaniem hierarchii folderów i podfolderów (`_get_window_relative_path`, `_get_folder_relative_path`).
* Pakowanie zaznaczonych widm, okien lub folderów wprost do spakowanego archiwum ZIP z zachowaniem ścieżek relatywnych.

**2. Elastyczny Silnik Nazewnictwa w DAG (`dialogs.py`, `evaluator.py`, `pipeline_dialog.py`, `items.py`)**
* **Przepływ metadanych nazw (`port_names` w `DAGEvaluator`):**
* Węzeł startowy (`input`) pobiera i rejestruje oryginalne nazwy próbek zarówno w trybie pojedynczym, jak i przy wsadowym przetwarzaniu macierzowym (Batch 2D).
* Wszystkie węzły pośrednie i filtry (Crop, AsLS, airPLS, SG, fotometria, skalowania) bezstratnie propagują nazwy pierwotne wzdłuż połączeń grafu.
* **Trzy tryby nazewnictwa w terminalach (`output` oraz `output_window`):**
1. *Prefiks / Sufiks do nazwy pierwotnej* (`source_affix`): domyślna konwencja z ustawionym prefiksem `proc_` oraz opcjonalnym sufiksem (np. `corr_próbka_01_asls`).
2. *Własna nazwa bazowa z indeksem* (`custom_base`): generowanie sekwencji typu `pfx_Baza_1_sfx`.
3. *Nazwa węzła / operacji* (`node_title`): zachowanie kompatybilności wstecznej.
* **Dynamiczny podgląd na żywo:** Dialog parametrów węzła (`NodeParamDialog`) w czasie rzeczywistym wyświetla podgląd wygenerowanej nazwy dla przykładowego widma.
* **Integracja z edycją węzła `output`:** Dwuklik i menu PPM na klocku zapisu standardowego otwiera pełne okno konfiguracji nazewnictwa.
* **Stabilizacja odświeżania podglądu:** Naprawiono obsługę dwuelementowej krotki `(port_results, port_names)` w `recalculate_graph_preview()`, eliminując błąd `AttributeError` podczas edycji potoku.

### 11.09.2026 — Checkpoint: Obiekt DataTable, Obsługa Folderowa Profili i Terminal output_table

**1. Nowa Struktura Danych: Tabele i Profile Liczbowe (`core/models.py`, `core/session.py`)**
* **Model `DataTable`:** Pierwszorzędna reprezentacja danych niewidmowych: profili kinetycznych $A(t)$, ramp temperaturowych $A(T)$, stężeń MCR-ALS $C(t)$, ładunków Scores PCA/SVD oraz całek pasm.
* **Wbudowany format TSV:** Metoda `.to_tsv()` generująca sformatowany ciąg z uwzględnieniem przecinka dziesiętnego, gotowy do natychmiastowego wklejenia w OriginLab lub MS Excel.
* **Integracja z sesją:** Nowy kontekst `'table'`, rejestracja w `session.tables` oraz zestaw sygnałów (`table_added`, `table_removed`, `table_folder_changed`, `active_table_changed`).

**2. Dedykowany Węzeł Terminalowy DAG (`output_table`)**
* **Przechwytywanie strumieni numerycznych:** Węzeł `output_table` („📊 Zapisz jako Tabelę / Profil”) automatycznie zapisuje dane z portów `Area`, `out_conc` czy `out_scores` bez sztucznego rzutowania ich na format widma.
* **Konfiguracja w `NodeParamDialog`:** Elastyczny dobór nazwy tabeli, docelowego folderu w drzewie projektu oraz etykiet osi X i Y.
* **Rejestracja w palecie:** Dodanie węzła do sekcji *Węzły Końcowe (Terminale)* z obsługą kabli multi-input.

**3. Prezentacja na Wykresie i Integracja z Eksploratorem (`canvas.py`, `project_tree.py`)**
* **Automatyczne przełączanie osi (`draw_table`):** Wybór tabeli w drzewie generuje na `SpectroCanvas` klasyczny wykres punktowo-liniowy $X-Y$ z normalną (nieodwróconą) osią X, paletą wieloserijną i dolnym paskiem informacyjnym.
* **Architektura folderowa (Drag & Drop):** Tabele funkcjonują w tych samych folderach co okna robocze z pełnym wsparciem przeciągania myszą między gałęziami i na poziom główny.
* **Menu kontekstowe PPM:** Bezpośrednie kopiowanie danych TSV do schowka systemowego pod Origin/Excel, eksport do pliku CSV, zmiana nazwy i relokacja w strukturze folderów.
* **Stabilizacja fontów Matplotliba:** Usunięcie glifów emoji z adnotacji na wykresie, eliminujące błędy `UserWarning: Glyph missing from font DejaVu Sans`.

**4. Trwałość Stanu w Kontenerze .shub (`io_hub/project_io.py`)**
* **Format projektu v1.2.0:** Pełna serializacja manifestu tabel w `project.json` oraz bezstratny binarny zrzut wektorów X i kolumn Y do archiwum `data.npz` (`float64`).
* **Bezpieczeństwo sesji:** Prawidłowe odtwarzanie powiązań tabel z folderami przy wczytywaniu projektu oraz czyszczeniu sesji.


### 14.09.2026 — Checkpoint: Architektura Wtyczkowa Węzłów DAG (Plugin Architecture & Node Registry)

**1. Modularny Rejestr Węzłów (`gui/node_editor/base.py`, `registry.py`)**
* **Abstrakcja `BaseNode` & `PortDef`:** Każdy węzeł definiuje swoje porty, kolorystykę, domyślne parametry, formularz konfiguracyjny (`create_config_dialog`) oraz czystą logikę obliczeniową (`compute`).
* **Auto-Discovery i Dekorator `@register_node`:** Dynamiczne skanowanie pakietu `gui/node_editor/nodes/` i rejestracja klas w locie bez konieczności ręcznego importowania w rdzeniu silnika.
* **Dynamiczna Paleta:** Drzewo biblioteki węzłów w lewym doku (`palette.py`, `pipeline_dialog.py`) zasila się automatycznie z metadanych zarejestrowanych klas.

**2. Domknięcie Izolacji Domenowej (`gui/node_editor/nodes/`)**
* **Rozbicie na 10 autonomicznych modułów:**
  * `node_io.py`: Węzeł startowy źródła (1D/2D/Seria), selekcja z projektu (ATM/Exp) oraz Mux/Zbieracz.
  * `node_terminals.py`: Terminale zapisu standardowego, okna roboczego oraz tabeli danych (`DataTable`).
  * `node_baseline.py`: AsLS, airPLS/airALS, wielomianowa oraz krańcowa.
  * `node_filters.py`: Savitzky-Golay, 1. i 2. pochodna, FSD Kauppinena, Crop, SNV, Min-Max, normy $L_1$/$L_2$ i offsety.
  * `node_math.py`: Operacje dwuwidmowe $A \pm k\cdot B$, mnożenie/dzielenie przez skalar, standard wewnętrzny (IS) oraz całkowanie pasma/kinetyka.
  * `node_photometry.py`: Przeliczenia fotometryczne $A \leftrightarrow T \leftrightarrow \%T$ oraz formalizm Kubelki-Munka DRIFTS.
  * `node_atmospheric.py`: VaporFit Suite z analitycznym OLS, autotuningiem SG i metrykami gładkości SDV/SSI.
  * `node_quantum.py`: Splot drgań DFT z dopasowaniem FWHM, kształtu (Gauss/Lorentz) i skalowaniem częstości $\sigma$.
  * `node_chemometrics.py`: Odszumiacz SVD/PCA Rank-$k$, dekompozycja MCR-ALS/SPEXFA oraz skalowania macierzowe.
  * `node_hdo.py`: Wieloprocesowy solwer HDO/Monte Carlo, pasma syntetyczne GxL, widma składowe z wagami i parametry MC.

**3. Likwidacja Długu Technologicznego i Monolitów**
* **Czysty `items.py`:** Zlikwidowano setki linii drabinek `if...elif...elif`. `NodeItem` pobiera porty, wymiary, nagłówki i niestandardowe nakładki (np. raporty HDO) bezpośrednio z obiektu `node_obj`.
* **Czysty `evaluator.py`:** Silnik ogranicza się do sortowania topologicznego grafu (algorytm Kahna) oraz propagacji strumieni numerycznych do `node.compute()`.
* **Usunięcie `dialogs.py`:** Skasowano monolityczny plik z uniwersalnym oknem parametrów — każdy węzeł posiada dedykowany, sprofilowany formularz w swoim pliku domenowym.


### 14.09.2026 — Checkpoint: Architektura Wtyczkowa Węzłów DAG, Ergonomia Drzewa i Reorganizacja HDO Suite

**1. Architektura Wtyczkowa i Auto-Discovery Węzłów (`gui/node_editor/`)**

* **Abstrakcja `BaseNode` i `PortDef`:** Każdy węzeł definiuje swoje porty, kolorystykę, domyślne parametry, formularz konfiguracyjny (`create_config_dialog`) oraz logikę numeryczną (`compute`).
* **Auto-Discovery i Rejestr (`registry.py`):** Dynamiczne wykrywanie węzłów z katalogu `gui/node_editor/nodes/` za pomocą dekoratora `@register_node` z automatycznym zasilaniem palety biblioteki w lewym doku.
* **Likwidacja monolitów i długu technicznego:**
* Usunięto zbędny plik `dialogs.py` — każdy węzeł posiada dedykowany formularz w swoim module domenowym lub korzysta z `StepParamDialog`.
* Drastycznie odchudzono `items.py` (usunięto setki linii drabinek `if...elif`; `NodeItem` odpytuje instancję `node_obj`).
* Uproszczono `evaluator.py` do sortowania topologicznego (algorytm Kahna) i czystej delegacji obliczeń do węzłów.


* **Rozbicie na 10 autonomicznych modułów domenowych:** `node_baseline.py`, `node_io.py`, `node_terminals.py`, `node_filters.py`, `node_math.py`, `node_photometry.py`, `node_atmospheric.py`, `node_quantum.py`, `node_chemometrics.py`, `node_hdo.py`.

**2. Ergonomia Drzewa Projektu i Nowe Ikony Okien (`project_tree.py`, `session.py`)**

* **Czysty start obszarów roboczych:** Zlikwidowano wymuszone tworzenie domyślnego „Okno 1 (Główne)” na starcie programu oraz podczas `clear_all()`.
* **Pełna swoboda usuwania okien:** Odblokowano możliwość usunięcia ostatniego okna roboczego z bezpiecznym powrotem do kontekstu Magazynu Danych (`storage`).
* **Czytelna separacja wizualna:** Zastąpiono mylącą ikonkę kart indeksowych (`🗂️`) jednoznaczną ikoną okna (`🪟`), eliminując pomyłki z żółtymi folderami (`📁`).

**3. Reorganizacja Suity HDO i Terminal Folderowy (`node_hdo.py`, `node_terminals.py`)**

* **Węzeł Widma Źródłowego (`HdoSourceSpectrumNode`):** Dedykowany bloczek dla $W_0$ z własnym wyborem widma z projektu i mnożnikiem $N$.
* **Naturalna skala podglądu (Sparklines):** Wyeliminowano sztuczne dopełnianie zerami do 4000–400 cm⁻¹ w węzłach $W_0$, $W_k$ oraz pikach syntetycznych — wykresy w bloczkach idealnie wypełniają szerokość kafelków w swoim rzeczywistym paśmie.
* **Węzeł Konfiguracji HDO (`HdoConfigNode`):** Scalono parametry nazwy bazowej, prefiksów, sufiksów, iteracji MC, kary oraz linii bazowej. Wyłączono fałszywy komunikat „Brak sygnału” (`has_preview = False`).
* **Trójmetodowy Solwer HDO:** Zredukowano 6 portów wyjściowych do 3 portów metod (`out_x`, `out_y`, `out_z`).
* **Terminal Zapisu do Folderu (`OutputFolderNode`):** Odbiera wyniki z wybranej metody i tworzy w drzewie folder `{prefix}{base_name}{suffix}_{n_peaks}p_{metoda}`, a w nim:
* Okno robocze `🪟` ze wszystkimi krzywymi ($W_0\cdot N$, fit, residua, składowe $W_k$, piki).
* Tabelę danych `📊` z kompletem parametrów dopasowania.
* Spójne dziedziczenie nazwy po folderze nadrzędnym.



**4. Bezpieczna Serializacja JSON i Pełna Kontrola Grafu (`pipeline_dialog.py`, `items.py`)**

* **Eliminacja błędu serializacji:** Rozdzielono parametry potoku od dużych tablic numerycznych (`runtime_bundles`), zintegrowano koder `NpEncoder` oraz filtr `clean_for_json`, umożliwiając bezbłędny zapis potoków do plików `.json`.
* **Swoboda edycji grafu:** Odblokowano usuwanie i duplikowanie węzła startowego `input` (klawisz `Delete`, menu kontekstowe, `⌘D` / `Ctrl+D`).

---

