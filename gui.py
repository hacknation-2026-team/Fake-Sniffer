import argparse
import threading
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from webapp.service import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gui", description="Interfejs przeglądarkowy Fake-Sniffera.")
    parser.add_argument("--base", type=Path, default=Path("base.json"), help="plik bazy incydentów")
    parser.add_argument("--clusters", type=Path, default=Path("clusters.json"), help="wynik cluster.py")
    parser.add_argument("--data-dir", type=Path, default=Path("gui_data"), help="wgrane pliki i podglądy")
    parser.add_argument("--host", default="127.0.0.1", help="adres nasłuchu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--faces-dir", help="katalog z osobami referencyjnymi: Imie_Nazwisko/*.jpg")
    parser.add_argument("--models-dir", help="katalog na pobrane modele")
    parser.add_argument("--gpu", action="store_true", help="użyj GPU (CUDA)")
    parser.add_argument("--no-deepfake", action="store_true", help="wyłącz analizę deepfake (oszczędza modele)")
    parser.add_argument("--no-ai-image", action="store_true", help="wyłącz detektor obrazów generowanych przez AI")
    parser.add_argument("--no-page-media", action="store_true", help="nie pobieraj obrazów i filmów ze stron")
    parser.add_argument("--no-browser", action="store_true", help="nie otwieraj przeglądarki")
    return parser


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        base_path=args.base,
        clusters_path=args.clusters,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        faces_dir=args.faces_dir,
        use_gpu=args.gpu,
        deepfake=not args.no_deepfake,
        ai_image=not args.no_ai_image,
        page_media=not args.no_page_media,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import uvicorn

    from webapp.server import create_app

    args = build_parser().parse_args(argv)
    app = create_app(settings_from(args))
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Fake-Sniffer GUI: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
