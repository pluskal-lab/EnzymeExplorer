"""Concatenate per-shard screening CSVs into a single output file.

The screening worker (:mod:`tps_predict_fasta`) writes one CSV per shard
under ``<output-root>/shards/``. This script merges them into a single
table sorted by ``isTPS_score`` (when present) and optionally deletes the
per-shard files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-dir",
        required=True,
        type=Path,
        help="Directory containing the per-shard CSVs.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Path to the combined CSV.",
    )
    parser.add_argument(
        "--delete-shards",
        action="store_true",
        help="Remove per-shard CSVs (and shards-dir if empty) on success.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="isTPS_score",
        help=(
            "Column to sort by, descending. Skipped if absent. Default: "
            "isTPS_score."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    shard_paths = sorted(args.shards_dir.glob("*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No CSV shards found under {args.shards_dir}")
    logger.info("Concatenating %d shard CSV(s)", len(shard_paths))

    frames = [pd.read_csv(p) for p in shard_paths]
    combined = pd.concat(frames, ignore_index=True)
    if args.sort_by in combined.columns:
        combined = combined.sort_values(args.sort_by, ascending=False)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_path, index=False)
    logger.info("Wrote %d rows to %s", len(combined), args.output_path)

    if args.delete_shards:
        for p in shard_paths:
            p.unlink()
        try:
            args.shards_dir.rmdir()
        except OSError:
            pass
        logger.info("Removed %d shard file(s)", len(shard_paths))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
