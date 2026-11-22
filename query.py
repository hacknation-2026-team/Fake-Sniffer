import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from correlation.base import Base
from correlation.clustering import Clusters
from correlation.matcher import classify
from correlation.profile import load_report, profile_from_report
from correlation.risk import assess, describe_risk
from correlation.similarity import THRESHOLD, TIGHT_THRESHOLD, WEAK_THRESHOLD
from correlation.summary import describe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query",
        description="Pyta, do którego klastra z ostatniej klasteryzacji należy nowy raport.",
    )
    parser.add_argument("reports", nargs="+", type=Path, help="raporty JSON z main.py")
    parser.add_argument("--base", type=Path, default=Path("base.json"), help="plik bazy incydentów")
    parser.add_argument(
        "--clusters", type=Path, default=Path("clusters.json"), help="wynik z cluster.py"
    )
    parser.add_argument("--out", type=Path, help="zapisz odpowiedzi jako JSON")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="próg powiązania")
    parser.add_argument("--tight", type=float, default=TIGHT_THRESHOLD, help="próg tej samej kampanii")
    parser.add_argument("--weak", type=float, default=WEAK_THRESHOLD, help="próg słabego podobieństwa")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        clusters = Clusters.load(args.clusters)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Brak wyniku klasteryzacji {args.clusters} ({exc}). Uruchom najpierw cluster.py.", file=sys.stderr)
        return 1

    base = Base(args.base)
    answers: List[Dict[str, Any]] = []
    failed = False

    for path in args.reports:
        try:
            report = load_report(path)
            profile = profile_from_report(report)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"{path}: nie da się wczytać raportu ({exc})", file=sys.stderr)
            failed = True
            continue
        if profile.is_empty:
            print(f"{path}: raport nie zawiera cech do porównania", file=sys.stderr)
            failed = True
            continue

        verdict = classify(profile, base, clusters, args.threshold, args.tight, args.weak)
        risk = assess(report)
        print(describe(path, verdict, clusters))
        print("\n".join(describe_risk(risk)))
        answers.append({"report": str(path), **verdict.to_dict(), "risk": risk.to_dict()})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
