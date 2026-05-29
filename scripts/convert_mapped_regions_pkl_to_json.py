"""Convert ``dict[str, list[MappedRegion]]`` .pkl files to portable JSON.

The domain-detection and structural-features pipelines pickle a mapping
of ``{seq_id: [MappedRegion, ...]}``. Pickle stores a fully-qualified
import path to ``MappedRegion``, so any program that doesn't have
EnzymeExplorer installed at the same module path fails to deserialize.

The pipelines now ALSO write a ``.json`` sidecar at write time, but any
files produced before this change exist only as .pkl. This script bulk
converts them in-place — one ``.json`` is emitted next to each input
``.pkl``, with the same stem.

Usage examples:

    # Convert one file:
    python scripts/convert_mapped_regions_pkl_to_json.py \\
        data/detected_domains/.../filename_2_regions.pkl

    # Convert every matching .pkl under a directory tree (default pattern):
    python scripts/convert_mapped_regions_pkl_to_json.py --root data/detected_domains

The script skips a .pkl when a same-stem .json already exists unless
``--force`` is set.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

from enzymeexplorer.src.structure_processing.structural_algorithms import MappedRegion
from enzymeexplorer.src.structure_processing.utils import save_seq_to_regions_json

logger = logging.getLogger(__name__)


def _looks_like_seq_to_regions(obj: object) -> bool:
    """Heuristic: is ``obj`` shaped like ``dict[str, list[MappedRegion]]``?"""
    if not isinstance(obj, dict):
        return False
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if v and not isinstance(v[0], MappedRegion):
            return False
        return True
    # Empty dict — still convertible (writes an empty JSON object).
    return True


def convert_one(pkl_path: Path, *, force: bool = False) -> bool:
    json_path = pkl_path.with_suffix(".json")
    if json_path.exists() and not force:
        logger.info("SKIP (json exists): %s", json_path)
        return False
    with pkl_path.open("rb") as fh:
        obj = pickle.load(fh)
    if not _looks_like_seq_to_regions(obj):
        logger.warning(
            "SKIP (not dict[str, list[MappedRegion]]): %s", pkl_path,
        )
        return False
    save_seq_to_regions_json(obj, json_path)
    logger.info("WROTE %s (%d sequences)", json_path, len(obj))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="One or more .pkl files to convert. Mutually exclusive with --root.",
    )
    parser.add_argument(
        "--root", type=Path,
        help="Recursively scan this directory for .pkl files to convert.",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.pkl",
        help="Glob pattern under --root. Default: *.pkl.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing .json sidecar.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    targets: list[Path] = list(args.paths)
    if args.root is not None:
        if not args.root.is_dir():
            logger.error("--root %s is not a directory", args.root)
            return 2
        targets.extend(args.root.rglob(args.pattern))
    if not targets:
        logger.error("No input paths. Pass positional .pkl paths or --root <dir>.")
        return 2

    n_wrote = 0
    for p in targets:
        if not p.is_file() or p.suffix != ".pkl":
            logger.warning("SKIP (not a .pkl file): %s", p)
            continue
        try:
            if convert_one(p, force=args.force):
                n_wrote += 1
        except Exception:  # noqa: BLE001
            logger.exception("FAILED on %s", p)
    logger.info("Converted %d / %d candidates.", n_wrote, len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
