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

`match.py` porównuje raport z każdym incydentem w bazie. Wynik to jeden z trzech werdyktów: `KNOWN_CAMPAIGN` (podobne do znanej kampanii), `WEAK_SIMILARITY` (słabe podobieństwo) albo `NO_MATCH` (brak dopasowania do znanych incydentów). `NO_MATCH` nie znaczy ani „oszustwo", ani „strona uczciwa": baza zna tylko swoje incydenty, więc np. `google.com` dostanie brak dopasowania. Do tego podaje trzy cechy o największym wkładzie, powody po polsku (wspólna domena, ta sama osoba, podobny obraz, wspólne nazwy własne) i najbliższe incydenty. Z `--add` raport trafia do bazy, ale nie do żadnego klastra, dopóki nie uruchomisz `cluster.py`.

`cluster.py` liczy grupy dla całej bazy i zapisuje je w `clusters.json`, nie zmieniając bazy. Kampania to spójna składowa grafu przy progu `--tight` (0.85), rodzina kampanii to społeczność Louvain przy progu `--threshold` (0.65).

`query.py` odpowiada, do której kampanii i rodziny należy nowy raport, na podstawie klastrów najbliższych incydentów (z rozmiarem grupy). Nie zmienia ani bazy, ani klastrów. Gdy po klasteryzacji do bazy dopisano nowe incydenty, ostrzega, że wynik może być nieaktualny.

Podobieństwo to średnia ważona po modalnościach dostępnych w obu raportach: infrastruktura 0.35, twarz 0.25, obraz 0.20, tekst 0.20. Brakująca modalność nie obniża wyniku. Wspólna domena (poza skracaczami linków), szablon DOM albo certyfikat TLS to twardy ślad i łączy incydenty niezależnie od wyniku. Para z odległością pHash do 4 i cosinusem tekstu powyżej 0.95 dostaje flagę duplikatu. Progi i podłogi skalowania cosinusów (`correlation/similarity.py`) trzeba skalibrować na prawdziwych danych.

## Deepfake

Ekstraktor `feature_extractor/deepfake_extractor.py` ocenia gotowymi klasyfikatorami z Hugging Face, czy obraz, klatki filmu i głos są syntetyczne: obraz i klatki to ViT `prithivMLmods/Deep-Fake-Detector-v2-Model`, głos to wav2vec2 `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`. Wynik to prawdopodobieństwo fałszerstwa (dla klatek i okien dźwięku średnia oraz maksimum), czyli wskaźnik, a nie dowód.

```
uv run main.py --image reklama.png --deepfake
uv run main.py --video reklama.mp4 --deepfake
uv run main.py --download-models --deepfake
```

W `main.py` jest to opcja (`--deepfake`), bo modele ważą razem ponad 1 GB. W GUI analiza deepfake jest włączona, wyłączysz ją flagą `--no-deepfake`. Modele trafiają do wspólnego katalogu `models/huggingface`. Jeśli model głosu ma etykiety typu `LABEL_0`, kod nie wie, która oznacza fałsz, i wtedy podaje same prawdopodobieństwa. Etykietę można wskazać w `OrchestratorConfig(deepfake_audio_fake_label=...)`.


---

Zalecamy odpalanie na izolowanych kontenerach! System odpytuje zewnętrzne serwery - nie przesadzać z automatycznymi wywołaniami

Shipped as is, no warrianty, no responsibility! You use at your own risk!s

## Kiedy detektory się uruchamiają

Detektory uczono na zdjęciach, więc na logach, ikonach, odznakach i ilustracjach dają przypadkowe wyniki. Dlatego deepfake działa tylko na obrazach i klatkach, na których wykryto twarz, a detektor obrazów AI tylko na fotografiach. Grafikę rozpoznaje `feature_extractor/image_kind.py` po pikselach: przezroczyste tło, skrajne proporcje (paski i banery), kilka płaskich kolorów lub duże jednolite plamy. Pominięty krok zostaje w raporcie ze statusem `skipped` i powodem (w GUI wiersz „pominięto"). Progi są heurystyką, więc zdjęcie produktu na białym tle bywa uznane za grafikę, a mocno zaszumiona grafika za zdjęcie.

## Obrazy generowane przez AI

`feature_extractor/ai_image_extractor.py` ocenia gotowym klasyfikatorem z Hugging Face, czy obraz lub klatki filmu wygenerował model (Stable Diffusion, Midjourney, DALL-E). To inne zadanie niż deepfake, który wykrywa podmienione twarze. Domyślny model to `Ateeqq/ai-vs-human-image-detector` (SigLIP, licencja Apache-2.0, etykiety `ai` i `hum`). Wynik to prawdopodobieństwo, że obraz jest z AI, i jest wskaźnikiem, a nie dowodem: detektory słabo uogólniają na generatory spoza zbioru uczącego.

```
uv run main.py --image reklama.png --ai-image
uv run main.py --video reklama.mp4 --ai-image
uv run main.py --download-models --ai-image
```

W GUI detektor jest włączony, wyłączysz go flagą `--no-ai-image`. Inny model ustawisz w `OrchestratorConfig(ai_image_model=...)`. Gdy ma etykiety `LABEL_0` i `LABEL_1`, wskaż fałsz przez `ai_image_fake_label`. Uwaga na licencje: popularny `Organika/sdxl-detector` ma CC-BY-NC, więc nadaje się tylko do użytku niekomercyjnego.

## Media na stronie

Przy `--url` ekstraktor strony znajduje na niej obrazy (`img`, `srcset`, leniwe `data-src`, tła z CSS, `og:image`), filmy (`video`, `source`, strumienie HLS, osadzenia YouTube i podobne) oraz emotikony w tekście. Zapisuje je w kroku `web_infrastructure` (pole `media`) razem z liczbą i gęstością emotikon. Z flagą `--page-media` pobiera obrazy i filmy do `page_media_out/` i puszcza je przez te same ekstraktory co wgrane pliki (OCR, hashe, twarze, deepfake, obrazy AI, klatki wideo). Kroki mają źródło `page_image` i `page_video`.

```
uv run main.py --url https://przyklad.pl/oferta --page-media --ai-image
```

Pobieranie ma limity: do 8 obrazów (największe), 2 filmy, 15 MB na obraz i 60 MB na film, obrazy mniejsze niż 120 px i duplikaty są pomijane. HLS: pobierane jest ok. 30 s pierwszego wariantu, bez szyfrowania i bez fMP4. Osadzone filmy (YouTube) nie są pobierane. Każdy adres, także po przekierowaniu, musi wskazywać na adres publiczny (ochrona przed SSRF). W GUI pobieranie jest włączone, wyłączysz je flagą `--no-page-media`.

Media wyciągane są także z zapisanego HTML-a (plik wgrany w GUI albo w folderze incydentu dla `build_base.py`). Względne linki rozwijane są względem adresu z HAR-a, a gdy go nie ma, z `link rel=canonical` albo `og:url`. Bez takiego adresu względne linki zostaną pominięte. Pobieranie i ocena mediów z zapisanego HTML-a działa tylko przy włączonym `page_media` (w GUI domyślnie tak, w `build_base.py` nie, żeby budowa bazy nie pobierała setek obrazów z sieci).

Gdy przeglądarka zawiedzie, ekstraktor nie traci wyniku: zachowuje częściowo załadowaną stronę, a bez przeglądarki pobiera ją zwykłym HTTP, idąc za przekierowaniami (30x i meta refresh) do ostatniej strony. Taka strona nie wykona JavaScriptu.

## Ocena ryzyka: czy to wygląda na fake

Werdykt z bazy mówi tylko, czy raport przypomina znane incydenty. Przy `NO_MATCH` pomaga osobna ocena `correlation/risk.py`, niezależna od bazy. Używa jawnych reguł z wagami, a każdy sygnał ma powód po polsku, więc wynik da się sprawdzić i poprawić. Sygnały podnoszące ryzyko: świeża domena (RDAP), przekierowania przez wiele domen lub skracacz linków, często nadużywana strefa (`.top`, `.xyz`), brak HTTPS, język nacisku (presja czasu, obietnica zysku, wyłudzanie danych, nagrody), osoba publiczna obok obietnicy zysku, dużo emotikon oraz obrazy AI, deepfake i syntetyczny głos. Sygnał obniżający ryzyko: wiekowa domena. Wynik to `Raczej nie fake` (do 0.25), `Podejrzane` (do 0.5) albo `Prawdopodobnie fake`, a bez danych `Za mało danych`. Ocena jest w GUI (karta „Czy to wygląda na fake?") oraz w wyjściu `match.py` i `query.py` (pole `risk` w `--out`).

Wagi dobrano ręcznie i nie były kalibrowane na prawdziwych danych, a sam tekst nie wystarczy, żeby dać `Prawdopodobnie fake` (frazy mają ograniczony wpływ). To wskaźnik, a nie dowód.

# Przygotowanie do pracy po kolei

Sklonuj repo, a nastepnie wypakuj dane przykładowe do katalogu `dane`. Folderpowinien mieć strukturę następującą: foldery a w każdym wszystkie materiały dotyczące danego incydentu/kampanii. Utwórz folder na podobizny ludzi znanych których często dotyczą kampanie, nazwijmy go `ludzie`. W folderze `ludzie` powinniśmy mieć foldery których nazwami są imiona/nazwiska osób, a w tych katalogach winny znajdować się ich zdjęcia.

```
uv sync
uv run main.py --download-models
uv run build_base.py ./dane --base base.json --reports-dir reports
uv run cluster.py --base base.json --out clusters.json
```


Aby dokonać przygotowania pojedynczych źródeł można wykonac poniższe komendy

```
uv run main.py --text "Rząd wprowadza podatek od deszczu" --out report.json
uv run main.py --image reklama.png --faces-dir ludzie/
uv run main.py --url https://example.com --entity "BRAND=Baltic Pipeline"
uv run main.py --video reklama.mp4 --gpu --no-vectors
uv run main.py --audio nagranie.mp3 --speech-language pl --spectrograms-dir spektrogramy
```

Komendy poniżej pokazują jak opracoweany raport cecdh może zostać porównany z bazą

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

# Interfejs graficzny

```
uv run gui.py --base base.json --clusters clusters.json
```

Otwiera w przeglądarce aplikację pod adresem `http://127.0.0.1:8000` z trzema zakładkami:

- **Analysis**: pole na link, przycisk `+` i strefa przeciągania plików (obraz, film, audio, tekst, HTML, HAR). Po `Submit` pojawia się przydział do kampanii, główne informacje (werdykt, co zadecydowało, dlaczego) i tabela użytych ekstraktorów z ich wynikami. Analizę można dopisać do bazy.
- **Campaigns**: każda kampania jako graf incydentów. Kliknięcie węzła pokazuje panel `Szczegóły fake` z podglądem, przyciskiem `Więcej` i `Porównaj z innym fake` (potem wskazujesz drugi węzeł).
- **Database**: lista incydentów z bazy, szczegóły i usuwanie.

Flagi: `--data-dir` (wgrane pliki i podglądy, domyślnie `gui_data`), `--port`, `--faces-dir`, `--models-dir`, `--gpu`, `--no-browser`. Pierwsza analiza pobiera modele, więc trwa dłużej. Interfejs nasłuchuje tylko na `127.0.0.1`, bo analiza linków wykonuje zapytania do zewnętrznych serwerów i nie ma logowania, nie wystawiaj go do sieci.



