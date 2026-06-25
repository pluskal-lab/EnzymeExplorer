"""Verify every dataset ID is present in every ESM-2 ablation gathered .h5.

Runs after ``extract_esm2_ablation_embeddings.sh`` to catch any sequences
that were dropped during extraction (CUDA OOM, transient errors). For
each of the 15 expected gathered .h5 files (3 base models x 5 layers),
reports:

* total ID count in the .h5
* dataset IDs missing from the .h5 (writes them to a per-variant
  ``data/missing_<variant>.txt`` so they can be re-extracted in a
  follow-up pass with batch_size=1 if needed)
* fail-the-script-style exit code if any variant is short

Usage:
    python scripts/verify_esm2_embeddings_completeness.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

BASE_LAYERS = [
    ("esm-2-t30", [26, 27, 28, 29, 30]),
    ("esm-2-t33", [29, 30, 31, 32, 33]),
    ("esm-2-t36", [32, 33, 34, 35, 36]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/EnzymeExplorer_Dataset.csv"),
    )
    parser.add_argument("--id-col", default="ID")
    parser.add_argument(
        "--gathered-dir", type=Path, default=Path("data"),
    )
    parser.add_argument(
        "--missing-out-dir", type=Path, default=Path("data"),
        help="Per-variant missing-ID files land here (one per failing variant).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(args.dataset, usecols=[args.id_col]).drop_duplicates(args.id_col)
    expected_ids = set(df[args.id_col].astype(str).tolist())
    logger.info("Expected %d unique IDs", len(expected_ids))

    failures: list[str] = []
    for base, layers in BASE_LAYERS:
        for L in layers:
            variant = f"{base}-L{L}"
            h5_path = args.gathered_dir / f"gathered_embs_{variant}_embs_avg.h5"
            if not h5_path.exists():
                failures.append(variant)
                logger.error("MISSING H5: %s", h5_path)
                continue
            try:
                df_h5 = pd.read_hdf(h5_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(variant)
                logger.error("Could not read %s: %s", h5_path, exc)
                continue
            # gathered .h5 schema: columns ['ID', 'Emb'], with a positional
            # integer index (NOT the protein id) — pull IDs from the column.
            have_ids = set(df_h5["ID"].astype(str).tolist())
            missing = expected_ids - have_ids
            extra = have_ids - expected_ids
            status = "OK" if not missing else "INCOMPLETE"
            logger.info(
                "[%s] %s — present=%d / expected=%d (missing %d, extra %d)",
                status, variant, len(have_ids), len(expected_ids),
                len(missing), len(extra),
            )
            if missing:
                failures.append(variant)
                out = args.missing_out_dir / f"missing_{variant}.txt"
                out.write_text("\n".join(sorted(missing)) + "\n")
                logger.error("  -> wrote %d missing IDs to %s", len(missing), out)

    if failures:
        logger.error(
            "INCOMPLETE / MISSING variants: %s", ", ".join(sorted(set(failures))),
        )
        return 1
    logger.info("All %d variants are complete.", sum(len(L) for _, L in BASE_LAYERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
