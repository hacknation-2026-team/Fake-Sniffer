import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from correlation.base import Base
from correlation.clustering import cluster
from correlation.similarity import THRESHOLD, TIGHT_THRESHOLD


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster",
        description="Grupuje incydenty z bazy w kampanie i rodziny kampanii.",
    )
    parser.add_argument("--base", type=Path, default=Path("base.json"), help="plik bazy incydentów")
    parser.add_argument(
        "--out", type=Path, default=Path("clusters.json"), help="plik z wynikiem klasteryzacji"
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="próg krawędzi dla rodzin")
    parser.add_argument("--tight", type=float, default=TIGHT_THRESHOLD, help="próg krawędzi dla kampanii")
    parser.add_argument("--seed", type=int, default=0, help="ziarno algorytmu Louvain")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base = Base(args.base)
    if not base.incidents:
        print(f"Baza {args.base} jest pusta lub nie istnieje.", file=sys.stderr)
        return 1

    result = cluster(base.incidents, args.threshold, args.tight, args.seed)
    campaigns, families = result.campaign_members(), result.family_members()
    for name, members in campaigns.items():
        print(f"{name}: {len(members)} ({', '.join(members)})")
    print(f"Kampanie: {len(campaigns)}, rodziny: {len(families)}, zapisano w {result.save(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
