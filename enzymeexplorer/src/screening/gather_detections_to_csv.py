"""Concatenate per-shard screening CSVs into combined output files.

The screening workers (:mod:`tps_predict_fasta`) write per-shard CSVs
under ``<output-root>/shards/<classifier>/<shard_name>.csv`` with one
sub-directory per classifier:

* ``plm/``                      — PlmRandomForest predictions
* ``plm_domains/``              — PlmDomainsRandomForest predictions
* ``plm_domains_fallback/``     — PLM-only fallback rows produced by
                                  predict_with_structures for proteins
                                  whose structures had no valid
                                  TPS-family domain
* ``no_structure/``             — UniProt IDs missing from AF-DB
                                  (plm_domains only)

This script merges each sub-directory into a single combined CSV at
``<output-root>/<classifier>.csv`` and optionally deletes the per-shard
files on success. By default it discovers every sub-directory under
``<shards-root>`` and processes each independently — pass
``--classifiers`` to restrict.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# Default ordering of classifiers. Each gets its own combined CSV; the
# launcher only writes a sub-dir when its classifier actually ran.
DEFAULT_SUBDIRS = (
    "plm",
    "plm_domains",
    "plm_domains_fallback",
    "no_structure",
)

# Per-classifier preferred sort key. Skipped silently if absent.
DEFAULT_SORT_KEYS = {
    "plm": "isTPS_score",
    "plm_domains": "isTPS_score",
    "plm_domains_fallback": "isTPS_score",
    "no_structure": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        required=True,
        type=Path,
        help=(
            "Directory holding the per-classifier sub-directories of "
            "shard CSVs (i.e. <launcher's output-root>/shards)."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "Where to drop the combined per-classifier CSVs "
            "(<output-root>/<classifier>.csv)."
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        help=(
            "Restrict to these classifier sub-directories. Default: "
            "auto-discover every sub-directory under --shards-root."
        ),
    )
    parser.add_argument(
        "--delete-shards",
        action="store_true",
        help="Remove per-shard CSVs (and the per-classifier sub-dirs if "
             "empty) on success.",
    )
    return parser.parse_args()


def _gather_one(
    subdir: Path, output_path: Path, sort_key: str | None,
) -> int:
    """Concat every ``*.csv`` under ``subdir`` into ``output_path``.

    Returns the number of rows written.
    """
    shard_paths = sorted(subdir.glob("*.csv"))
    if not shard_paths:
        logger.warning("No CSV shards under %s — skipping", subdir)
        return 0

    frames = [pd.read_csv(p) for p in shard_paths]
    combined = pd.concat(frames, ignore_index=True)
    if sort_key and sort_key in combined.columns:
        combined = combined.sort_values(sort_key, ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d rows from %d shards to %s",
        len(combined), len(shard_paths), output_path,
    )
    return len(combined)


def main(args: argparse.Namespace) -> None:
    if args.classifiers:
        subdirs = list(args.classifiers)
    else:
        subdirs = sorted(
            p.name for p in args.shards_root.iterdir() if p.is_dir()
        )
        if not subdirs:
            raise FileNotFoundError(
                f"No classifier sub-directories under {args.shards_root}"
            )

    for name in subdirs:
        subdir = args.shards_root / name
        if not subdir.is_dir():
            logger.warning("Skipping missing sub-dir %s", subdir)
            continue
        sort_key = DEFAULT_SORT_KEYS.get(name)
        _gather_one(
            subdir,
            args.output_root / f"{name}.csv",
            sort_key=sort_key,
        )

        if args.delete_shards:
            for p in subdir.glob("*.csv"):
                p.unlink()
            try:
                subdir.rmdir()
            except OSError:
                pass

    if args.delete_shards:
        try:
            args.shards_root.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
