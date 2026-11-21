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

Zalecamy odpalanie na izolowanych kontenerach! System odpytuje zewnętrzne serwery - nie przesadzać z automatycznymi wywołaniami

Shipped as is, no warrianty, no responsibility!