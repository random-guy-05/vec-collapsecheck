from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from .core import analyze_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect obvious population collapse in a VEC H5AD prediction."
    )
    parser.add_argument("file", type=Path)
    parser.add_argument("--max-cells", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    try:
        import anndata as ad

        data = ad.read_h5ad(args.file, backed="r")
        try:
            report = analyze_matrix(
                data.X,
                max_cells=args.max_cells,
                seed=args.seed,
            )
        finally:
            with suppress(Exception):
                data.file.close()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(report.classification)
    for reason in report.reasons:
        print(f"- {reason}")

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    if report.classification == "SEVERE_COLLAPSE":
        return 4
    if report.classification == "LOW_DIVERSITY_WARNING":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
