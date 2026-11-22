import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from feature_extractor.orchestrator import Orchestrator, OrchestratorConfig, Report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fake-sniffer",
        description="Analiza podejrzanej reklamy: strona, tekst, obraz i wideo można łączyć.",
    )
    parser.add_argument("--url", help="adres strony do przeanalizowania")
    parser.add_argument("--text", help="tekst podany wprost")
    parser.add_argument("--text-file", type=Path, help="plik tekstowy UTF-8 z tekstem do analizy")
    parser.add_argument("--image", help="ścieżka do obrazka (jpg, png, webp...)")
    parser.add_argument("--video", help="ścieżka do filmu")
    parser.add_argument("--audio", help="ścieżka do nagrania (mp3, wav, m4a...)")
    parser.add_argument("--out", type=Path, default=Path("report.json"), help="plik raportu JSON")
    parser.add_argument("--gpu", action="store_true", help="użyj GPU (CUDA)")
    parser.add_argument("--no-vectors", action="store_true", help="pomiń wektory w raporcie")
    parser.add_argument(
        "--models-dir",
        help="katalog na pobrane modele (domyślnie FAKE_SNIFFER_MODELS_DIR albo ./models)",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="pobierz wszystkie modele z góry i zakończ",
    )
    parser.add_argument(
        "--deepfake",
        action="store_true",
        help="oceń obrazy, klatki i głos klasyfikatorami deepfake (pobiera ponad 1 GB modeli)",
    )
    parser.add_argument("--faces-dir", help="katalog z osobami referencyjnymi: Imie_Nazwisko/*.jpg")
    parser.add_argument("--keyframes-dir", default="keyframes_out", help="gdzie zapisać klatki z wideo")
    parser.add_argument(
        "--entity",
        action="append",
        default=[],
        metavar="ETYKIETA=FRAZA",
        help="własna encja, np. BRAND=Baltic Pipeline (można powtarzać)",
    )
    parser.add_argument("--embedder-model", default=OrchestratorConfig().embedder_model)
    parser.add_argument("--whisper-model", help="model Whisper (domyślnie turbo na GPU, small na CPU)")
    parser.add_argument("--speech-language", help="język mowy, np. pl (domyślnie wykrywany)")
    parser.add_argument("--spectrograms-dir", help="gdzie zapisać spektrogramy .png i .npz (domyślnie nigdzie)")
    return parser


def parse_entities(values: Sequence[str]) -> List[Tuple[str, str]]:
    patterns = []
    for value in values:
        label, separator, phrase = value.partition("=")
        if not separator or not label.strip() or not phrase.strip():
            raise ValueError(f"Encja musi mieć postać ETYKIETA=FRAZA, dostałem: {value!r}")
        patterns.append((label.strip(), phrase.strip()))
    return patterns


def collect_text(text: Optional[str], text_file: Optional[Path]) -> Optional[str]:
    parts = [text] if text else []
    if text_file is not None:
        parts.append(text_file.read_text(encoding="utf-8"))
    return "\n\n".join(parts) if parts else None


def print_summary(report: Report, saved_to: Path) -> None:
    for step in report.steps:
        line = f"{step.status:<8}{step.source:<7}{step.name:<22}{step.duration_sec:>7.1f}s"
        print(f"{line}  {step.error}" if step.error else line)
    print(f"Raport zapisany: {saved_to}")


def download_models(orchestrator: Orchestrator) -> int:
    statuses = orchestrator.prefetch()
    for name, status in statuses.items():
        print(f"{name:<20}{status}")
    return 0 if all(status == "ok" for status in statuses.values()) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        patterns = parse_entities(args.entity)
        text = collect_text(args.text, args.text_file)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    config = OrchestratorConfig(
        use_gpu=args.gpu,
        include_vectors=not args.no_vectors,
        reference_faces_dir=args.faces_dir,
        keyframes_dir=args.keyframes_dir,
        entity_patterns=patterns,
        embedder_model=args.embedder_model,
        whisper_model=args.whisper_model,
        speech_language=args.speech_language,
        spectrograms_dir=args.spectrograms_dir,
        models_dir=args.models_dir,
        deepfake=args.deepfake,
    )
    if args.download_models:
        return download_models(Orchestrator(config))

    if not any([args.url, text, args.image, args.video, args.audio]):
        parser.error("podaj co najmniej jedno z: --url, --text, --text-file, --image, --video, --audio")

    report = Orchestrator(config).run(
        url=args.url, text=text, image=args.image, video=args.video, audio=args.audio
    )
    saved_to = report.save_json(args.out)
    print_summary(report, saved_to)
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
