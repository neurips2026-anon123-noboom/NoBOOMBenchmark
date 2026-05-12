from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional

sys.dont_write_bytecode = True

from common import OptionalFigureSkipped, load_alarm_results, load_dataset_metadata, repo_root, resolve_repo_path
import plot_alarm_diagnostics
import plot_alarm_heatmap
import plot_alarm_timelines
import plot_dataset_regimes
import plot_protocol
import plot_seed_uncertainty
import plot_winner_gap


FigureFunc = Callable[[], List[Path]]

REQUIRED = ["protocol", "heatmap", "winner-gap", "dataset-regimes"]
OPTIONAL = ["diagnostics", "uncertainty", "timelines"]
ALL = REQUIRED + OPTIONAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NoBOOM paper figures.")
    parser.add_argument("--all", action="store_true", help="Generate required figures and available optional figures.")
    parser.add_argument("--only", choices=ALL, help="Generate a single figure group.")
    parser.add_argument("--outdir", default="figures/generated", help="Output directory for PDF and PNG figures.")
    parser.add_argument("--data-dir", default="data/figures", help="Directory for normalized figure data CSVs.")
    parser.add_argument("--results", help="Explicit ALARM result CSV to normalize and use.")
    parser.add_argument("--strict", action="store_true", help="Raise errors instead of skipping missing optional figures.")
    args = parser.parse_args()
    if not args.all and args.only is None:
        parser.error("Use --all or --only <figure>.")
    if args.all and args.only is not None:
        parser.error("Use either --all or --only, not both.")
    return args


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    data_dir = resolve_repo_path(Path(args.data_dir))
    outdir = resolve_repo_path(Path(args.outdir))
    results_path = resolve_repo_path(Path(args.results)) if args.results else None
    return data_dir, outdir, results_path


def _dispatch(data_dir: Path, outdir: Path, results_path: Optional[Path], strict: bool) -> Dict[str, FigureFunc]:
    return {
        "protocol": lambda: plot_protocol.plot(outdir),
        "heatmap": lambda: plot_alarm_heatmap.plot(data_dir, outdir, results_path),
        "winner-gap": lambda: plot_winner_gap.plot(data_dir, outdir, results_path),
        "dataset-regimes": lambda: plot_dataset_regimes.plot(data_dir, outdir),
        "diagnostics": lambda: plot_alarm_diagnostics.plot(data_dir, outdir, results_path, strict),
        "uncertainty": lambda: plot_seed_uncertainty.plot(data_dir, outdir, results_path, strict),
        "timelines": lambda: plot_alarm_timelines.plot(outdir, strict),
    }


def main() -> None:
    args = parse_args()
    root = repo_root()
    data_dir, outdir, results_path = _paths(args)
    print(f"Repository root: {root}")
    print(f"Figure data directory: {data_dir.relative_to(root)}")
    print(f"Output directory: {outdir.relative_to(root)}")

    alarm_df, alarm_source = load_alarm_results(data_dir=data_dir, results_path=results_path)
    metadata_df, metadata_source = load_dataset_metadata(data_dir=data_dir)
    print(f"ALARM data: {alarm_source} ({alarm_df['status'].eq('complete').sum()} complete cells)")
    print(f"Dataset metadata: {metadata_source} ({len(metadata_df)} datasets)")

    names = ALL if args.all else [args.only]
    dispatch = _dispatch(data_dir, outdir, results_path, args.strict)
    generated: List[Path] = []

    for name in names:
        print(f"Generating {name}...")
        try:
            paths = dispatch[name]()
        except OptionalFigureSkipped as exc:
            print(str(exc))
            continue
        for path in paths:
            print(f"  wrote {path.relative_to(root)}")
        generated.extend(paths)

    if not generated:
        print("No figures generated.")
    else:
        print(f"Generated {len(generated)} files.")


if __name__ == "__main__":
    main()
