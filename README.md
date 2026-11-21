# Fake-Sniffer

Pakiet działa pod uv, wystarczy zainstalować sync-ować i działać

```
uv sync
uv run main.py --download-models

uv run main.py --text "Rząd wprowadza podatek od deszczu" --out report.json
uv run main.py --image reklama.png --faces-dir referencje/
uv run main.py --url https://example.pl --entity "BRAND=Baltic Pipeline"
uv run main.py --video reklama.mp4 --gpu --no-vectors
uv run main.py --audio nagranie.mp3 --speech-language pl --spectrograms-dir spektrogramy
```

## Twarze

Aby z twarzy porównykania korzystać można, należy katalog przygotować dodatkowy, jego nazwę w parametrze `faces-dir` do programu przekazać, a wnętrze wypełnić jego folderami z nazwą osoby której twarz zawierać podfoldery mają po czym do podfolderów próbki twarzy fotograii uzupełnic należy!

## Modele

Każdy ekstraktor pobiera swój model przy pierwszym użyciu i zapisuje go w katalogu `models/` w repozytorium (podkatalogi: easyocr, huggingface, insightface, whisper, playwright). Kolejne uruchomienia korzystają z dysku. Chromium dla Playwrighta doinstaluje się samo, jeśli go brakuje (na Linuksie mogą być potrzebne biblioteki systemowe: `uv run playwright install-deps chromium`).

```
uv run main.py --download-models
uv run main.py --models-dir /dane/modele --text "..."
```

`--download-models` pobiera wszystko naraz, np. przy budowaniu obrazu kontenera. Katalog można zmienić flagą `--models-dir` albo zmienną `FAKE_SNIFFER_MODELS_DIR`. InsightFace (buffalo_l) ma licencję niekomercyjną.

## Budowanie bazy ze zebranych materiałów

Katalog z incydentami, w którym każdy podfolder to jeden incydent (zrzuty ekranu, filmy, zapisane strony HTML i pliki HAR), zamienia na bazę `build_base.py`. Pracuje offline na plikach i niczego nie pobiera ze stron.

```
uv run build_base.py E:/dane --base base.json --reports-dir reports
uv run cluster.py --base base.json --out clusters.json
```

Z HAR bierze łańcuch przekierowań głównego dokumentu, domenę końcową i adresy IP serwerów (trackery i zasoby zewnętrzne pomija). Z zapisanego HTML bierze odcisk szablonu DOM i widoczny tekst, a obrazy i filmy przechodzą przez ekstraktory jak w `main.py`. Foldery `*_files` i `resources` są pomijane. Raport każdego incydentu trafia do `reports/<folder>.json`, więc kolejne uruchomienia nie liczą go od nowa, chyba że dasz `--force`. Przydatne flagi: `--only 1 10`, `--no-images`, `--no-videos`, `--faces-dir`, `--gpu`.

## Porównanie z bazą znanych incydentów

To osobny proces, który czyta raporty z `main.py`. Baza to jeden plik JSON (`base.json`), a wynik klasteryzacji leży w osobnym pliku (`clusters.json`).

```
#Dodawaine wyniku do bazy
uv run match.py report.json --base base.json --add --label "Orlen, pasywny dochód"
# Test wyniku
uv run match.py nowy.json --base base.json --out wynik.json
# Klastrowanie bazy
uv run cluster.py --base base.json --out clusters.json
# Do którego klastra należy nowy raport
uv run query.py nowy.json --base base.json --clusters clusters.json
```

`match.py` porównuje raport z każdym incydentem w bazie. Wynik to jeden z trzech werdyktów: `KNOWN_CAMPAIGN` (podobne do znanej kampanii), `WEAK_SIMILARITY` (słabe podobieństwo) albo `NEW_CAMPAIGN_CANDIDATE` (nowa kampania). Do tego podaje trzy cechy o największym wkładzie, powody po polsku (wspólna domena, ta sama osoba, podobny obraz, wspólne nazwy własne) i najbliższe incydenty. Z `--add` raport trafia do bazy, ale nie do żadnego klastra, dopóki nie uruchomisz `cluster.py`.

`cluster.py` liczy grupy dla całej bazy i zapisuje je w `clusters.json`, nie zmieniając bazy. Kampania to spójna składowa grafu przy progu `--tight` (0.85), rodzina kampanii to społeczność Louvain przy progu `--threshold` (0.65).

`query.py` odpowiada, do której kampanii i rodziny należy nowy raport, na podstawie klastrów najbliższych incydentów (z rozmiarem grupy). Nie zmienia ani bazy, ani klastrów. Gdy po klasteryzacji do bazy dopisano nowe incydenty, ostrzega, że wynik może być nieaktualny.

Podobieństwo to średnia ważona po modalnościach dostępnych w obu raportach: infrastruktura 0.35, twarz 0.25, obraz 0.20, tekst 0.20. Brakująca modalność nie obniża wyniku. Wspólna domena (poza skracaczami linków), szablon DOM albo certyfikat TLS to twardy ślad i łączy incydenty niezależnie od wyniku. Para z odległością pHash do 4 i cosinusem tekstu powyżej 0.95 dostaje flagę duplikatu. Progi i podłogi skalowania cosinusów (`correlation/similarity.py`) trzeba skalibrować na prawdziwych danych.

---

Zalecamy odpalanie na izolowanych kontenerach! System odpytuje zewnętrzne serwery - nie przesadzać z automatycznymi wywołaniami

Shipped as is, no warrianty, no responsibility! You use at your own risk!s