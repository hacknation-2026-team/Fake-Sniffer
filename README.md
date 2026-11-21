# Fake-Sniffer

Pakiet działa pod uv, wystarczy zainstalować sync-ować i działać

```
uv sync
uv run playwright install chromium

uv run main.py --text "Rząd wprowadza podatek od deszczu" --out report.json
uv run main.py --image reklama.png --faces-dir referencje/
uv run main.py --url https://example.pl --entity "BRAND=Baltic Pipeline"
uv run main.py --video reklama.mp4 --gpu --no-vectors
```

Zalecamy odpalanie na izolowanych kontenerach! System odpytuje zewnętrzne serwery - nie przesadzać z automatycznymi wywołaniami

Shipped as is, no warrianty, no responsibility!