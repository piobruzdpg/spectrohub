import os
import re
from io_hub.readers.ascii_reader import ASCIIReader

EXPORT_DIR = "/Volumes/Dysk_2TB/PYTHON_PROJEKTY/spectrohub/Widma_wysuszanie_TMAO0.25_H2O_po_korekcji_i_przyciete/Widma_2000_4500/Widma_2000_4500"

files = sorted([f for f in os.listdir(EXPORT_DIR) if f.lower().endswith('.csv')])
print(f"Liczba plików na dysku: {len(files)}")

# 1. Test unikalności nazw
names = [os.path.splitext(f)[0] for f in files]
unique_names = set(names)
if len(unique_names) != len(files):
    print(f"⚠️ ZNALEZIONO KOLIZJĘ NAZW! Plików: {len(files)}, unikalnych nazw: {len(unique_names)}")
    duplicates = [n for n in names if names.count(n) > 1]
    print(f"Przykładowe powtórzenia: {set(duplicates[:5])}")

# 2. Test auto-routingu ATM/CO2
atm_co2_matches = [f for f in files if re.search(r'(atm|co2)', f, re.IGNORECASE)]
if atm_co2_matches:
    print(f"⚠️ ZNALEZIONO WZORCE ATMOSFERY: {len(atm_co2_matches)} plików zawiera 'atm'/'co2'!")
    print(f"Przykłady: {atm_co2_matches[:5]}")

# 3. Test parsowania przez ASCIIReader
reader = ASCIIReader()
failed_files = []
for f in files:
    full_p = os.path.join(EXPORT_DIR, f)
    try:
        specs = reader.read(full_p)
        if not specs or len(specs[0].wavenumbers) == 0:
            failed_files.append((f, "Puste widmo"))
    except Exception as e:
        failed_files.append((f, str(e)))

if failed_files:
    print(f"⚠️ BŁĘDY PARSOWANIA: {len(failed_files)} plików nie udało się odczytać!")
    for fn, err in failed_files[:5]:
        print(f"  - {fn}: {err}")
else:
    print("✅ Wszystkie 598 plików parsuje się poprawnie przez ASCIIReader.")