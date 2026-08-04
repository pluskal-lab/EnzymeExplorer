"""One-off column-rename migration for prediction / screening CSVs.

Rewrites headers in-place so pre-Section-8 CSVs stay usable with the
new consumer scripts:

    <class>_score          →   <class>_raw
    <class>_p_calibrated   →   <class>_p

Where ``<class>`` is any of the calibrated TPS/substrate classes
(``TPS``, ``IDS``, ``FPP``, ``GPP``, ``GGPP``, ``EDSQ``, ``CPP``,
``GFPP``, ``2xFPP``, ``2xGGPP``). Non-matching columns are left
untouched.

Usage:
    python scripts/one_offs/rename_pred_columns.py <path> [<path> ...]
                                                  [--recursive] [--dry-run]

Each ``<path>`` may be a single CSV file or a directory; with
``--recursive`` the walker picks up every ``*.csv`` beneath a directory.

The rewrite is atomic (write to ``<file>.tmp`` then ``os.replace`` to
the original name); a SIGKILL never leaves a partial CSV behind.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


CLASSES = (
    "TPS", "IDS", "FPP", "GPP", "GGPP", "EDSQ", "CPP", "GFPP",
    "2xFPP", "2xGGPP",
)

_SCORE_RE = re.compile(
    r"^(" + "|".join(re.escape(c) for c in CLASSES) + r")_score$"
)
_PCAL_RE = re.compile(
    r"^(" + "|".join(re.escape(c) for c in CLASSES) + r")_p_calibrated$"
)


def _rename_column(col: str) -> str:
    m = _PCAL_RE.match(col)
    if m:
        return f"{m.group(1)}_p"
    m = _SCORE_RE.match(col)
    if m:
        return f"{m.group(1)}_raw"
    return col


def _rewrite_file(path: Path, *, dry_run: bool) -> tuple[int, list[str]]:
    df = pd.read_csv(path)
    new_cols = [_rename_column(c) for c in df.columns]
    changed = [
        f"{old!r}->{new!r}"
        for old, new in zip(df.columns, new_cols)
        if old != new
    ]
    if not changed:
        return 0, []
    df.columns = new_cols
    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return len(changed), changed


def _iter_csvs(paths: list[Path], *, recursive: bool):
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            it = p.rglob("*.csv") if recursive else p.glob("*.csv")
            yield from it
        else:
            logger.warning("Skipping %s (not a file or directory)", p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--recursive", action="store_true",
        help="For directories, recurse into every subdirectory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without touching files.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    total_files = 0
    changed_files = 0
    for csv_path in _iter_csvs(args.paths, recursive=args.recursive):
        total_files += 1
        try:
            n_changed, changed = _rewrite_file(csv_path, dry_run=args.dry_run)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s (%s)", csv_path, exc)
            continue
        if n_changed:
            changed_files += 1
            action = "would rename" if args.dry_run else "renamed"
            logger.info("%s %d cols in %s: %s", action, n_changed, csv_path, changed)
    logger.info(
        "Done. %d CSV(s) scanned; %d %s.",
        total_files, changed_files,
        "would be modified" if args.dry_run else "modified",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
