import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from correlation.base import Base
from correlation.dataset import build_case_report, find_cases
from correlation.profile import profile_from_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_base",
        description="Buduje bazę incydentów z folderów ze zrzutami, stronami HTML, HAR i filmami.",
    )
    parser.add_argument("dataset", type=Path, help="katalog, w którym każdy podfolder to jeden incydent")
    parser.add_argument("--base", type=Path, default=Path("base.json"), help="plik bazy incydentów")
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports"), help="gdzie zapisać raporty incydentów"
    )
    parser.add_argument("--only", nargs="+", metavar="FOLDER", help="tylko te foldery")
    parser.add_argument("--force", action="store_true", help="policz raporty od nowa, nawet jeśli istnieją")
    parser.add_argument("--no-images", action="store_true", help="pomiń obrazy")
    parser.add_argument("--no-videos", action="store_true", help="pomiń filmy")
    parser.add_argument("--faces-dir", help="katalog z osobami referencyjnymi: Imie_Nazwisko/*.jpg")
    parser.add_argument("--gpu", action="store_true", help="użyj GPU (CUDA)")
    parser.add_argument("--models-dir", help="katalog na pobrane modele")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dataset.is_dir():
        parser.error(f"{args.dataset} nie jest katalogiem")

    from feature_extractor.orchestrator import Orchestrator, OrchestratorConfig

    orchestrator = Orchestrator(
        OrchestratorConfig(
            use_gpu=args.gpu, reference_faces_dir=args.faces_dir, models_dir=args.models_dir
        )
    )
    base = Base(args.base)
    added, skipped = 0, 0

    for case in find_cases(args.dataset):
        if args.only and case.name not in args.only:
            continue
        if case.is_empty:
            print(f"{case.name}: brak materiałów, pomijam")
            skipped += 1
            continue

        report_path = args.reports_dir / f"{case.name}.json"
        if report_path.exists() and not args.force:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            origin = "z pliku"
        else:
            report = build_case_report(
                case,
                orchestrator,
                keyframes_root=args.reports_dir / "keyframes",
                images=not args.no_images,
                videos=not args.no_videos,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            origin = "policzony"

        profile = profile_from_report(report)
        if profile.is_empty:
            print(f"{case.name}: raport bez cech do porównania, pomijam")
            skipped += 1
            continue

        base.incidents = [i for i in base.incidents if i.source != case.name]
        incident = base.add(profile, case.name, label=case.name)
        errors = sum(1 for s in report["steps"] if s["status"] == "error")
        print(
            f"{case.name}: {incident.id} ({origin}; html {len(case.html)}, har {len(case.har)}, "
            f"obrazy {len(case.images)}, filmy {len(case.videos)}; błędy kroków: {errors})"
        )
        added += 1

    base.save()
    print(f"Dodano {added}, pominięto {skipped}. Baza: {args.base} ({len(base)} incydentów)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
