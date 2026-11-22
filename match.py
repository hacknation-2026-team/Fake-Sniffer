import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from correlation.base import Base
from correlation.matcher import classify
from correlation.profile import load_report, profile_from_report
from correlation.risk import assess, describe_risk
from correlation.similarity import THRESHOLD, WEAK_THRESHOLD
from correlation.summary import describe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="match",
        description="Porównuje raporty z bazą znanych incydentów i mówi, czy to znana kampania.",
    )
    parser.add_argument("reports", nargs="+", type=Path, help="raporty JSON z main.py")
    parser.add_argument("--base", type=Path, default=Path("base.json"), help="plik bazy incydentów")
    parser.add_argument("--add", action="store_true", help="dopisz raporty do bazy po porównaniu")
    parser.add_argument("--label", help="opis dopisywanego incydentu, np. nazwa kampanii")
    parser.add_argument("--out", type=Path, help="zapisz wyniki porównania jako JSON")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="próg powiązania")
    parser.add_argument("--weak", type=float, default=WEAK_THRESHOLD, help="próg słabego podobieństwa")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = Base(args.base)
    results: List[Dict[str, Any]] = []
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

        verdict = classify(profile, base, threshold=args.threshold, weak=args.weak)
        risk = assess(report)
        print(describe(path, verdict))
        print("\n".join(describe_risk(risk)))
        entry = {"report": str(path), **verdict.to_dict(), "risk": risk.to_dict()}

        if args.add:
            incident = base.add(profile, str(path), args.label)
            entry["added_as"] = incident.id
        results.append(entry)

    if args.add and results:
        print(f"Baza zapisana: {base.save()} ({len(base)} incydentów)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
