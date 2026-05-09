"""TPS screening worker — score one FASTA shard and write a CSV.

Designed for large-scale screening (hundreds of millions of sequences).
The :mod:`tps_screening_cluster_launcher` spawns one of these per GPU,
each with a disjoint ``[start_i, end_i)`` slice of the input FASTA.

Internals delegate to :func:`enzymeexplorer.src.prediction.pipeline.predict_sequences_only`
so the screening output matches the rest of the prediction pipeline:
``id``, ``sequence``, and a ``<class>_score`` + ``<class>_p_calibrated``
column per class. By default rows where every calibrated probability
falls below ``--min-p-keep`` are dropped to keep per-shard CSVs small;
pass ``--keep-all`` to retain everything.
"""

from __future__ import annotations

import argparse
import logging
from itertools import islice
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore

from enzymeexplorer.src.prediction.embeddings import load_plm_embedder
from enzymeexplorer.src.prediction.pipeline import (
    DEFAULT_CALIBRATION_CSV,
    DEFAULT_PLM_MODEL,
    DEFAULT_PLM_ONLY_BUNDLE,
    predict_sequences_only,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path", required=True, type=Path,
        help="Input FASTA to screen (the worker reads only [start-i, end-i)).",
    )
    parser.add_argument(
        "--output-csv", required=True, type=Path,
        help="Per-shard CSV output path.",
    )
    parser.add_argument(
        "--start-i", type=int, default=0,
        help="Index of the first sequence (inclusive) to process.",
    )
    parser.add_argument(
        "--end-i", type=int, default=None,
        help="Index of the last sequence (exclusive). Default: end of FASTA.",
    )
    parser.add_argument(
        "--plm-only-bundle", type=Path, default=Path(DEFAULT_PLM_ONLY_BUNDLE),
    )
    parser.add_argument(
        "--calibration-csv", type=Path, default=Path(DEFAULT_CALIBRATION_CSV),
    )
    parser.add_argument(
        "--plm-model", type=str, default=DEFAULT_PLM_MODEL,
    )
    parser.add_argument("--plm-batch-size", type=int, default=4)
    parser.add_argument(
        "--min-p-keep", type=float, default=0.5,
        help=(
            "Drop rows where every <class>_p_calibrated is below this value. "
            "Default 0.5; use --keep-all to disable filtering. Rows whose "
            "classes all have NaN p_calibrated (skipped calibrators) are "
            "always kept."
        ),
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Retain every row regardless of calibrated probability.",
    )
    return parser.parse_args()


def _load_fasta_shard(
    path: Path, start_i: int, end_i: int | None
) -> pd.DataFrame:
    """Read ``[start_i, end_i)`` records from a FASTA into a DataFrame."""
    records = SeqIO.parse(str(path), "fasta")
    sliced = islice(records, start_i, end_i)
    rows = [{"id": r.id, "sequence": str(r.seq)} for r in sliced]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["id", "sequence"]).drop_duplicates(subset="id")
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(" ", "")
    return df.reset_index(drop=True)


def _filter_by_calibrated_probability(
    table: pd.DataFrame, min_p: float,
) -> pd.DataFrame:
    """Drop rows where every ``*_p_calibrated`` column is below ``min_p``.

    NaN cells (classes without a calibrator) do not count against a row;
    a row whose every calibrated cell is NaN is kept.
    """
    p_cols = [c for c in table.columns if c.endswith("_p_calibrated")]
    if not p_cols:
        return table
    arr = table[p_cols].to_numpy(dtype=np.float64)
    # A row passes if any cell is >= min_p, OR every cell is NaN (no claim).
    has_above = np.nansum(arr >= min_p, axis=1) > 0
    all_nan = np.all(np.isnan(arr), axis=1)
    keep = has_above | all_nan
    return table[keep].reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    sequences_df = _load_fasta_shard(args.fasta_path, args.start_i, args.end_i)
    logger.info(
        "Loaded shard [%d, %s) from %s — %d sequences",
        args.start_i,
        args.end_i if args.end_i is not None else "end",
        args.fasta_path,
        len(sequences_df),
    )
    if sequences_df.empty:
        # Still write an empty CSV with the expected schema so the gather step
        # doesn't have to handle missing files.
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["id", "sequence"]).to_csv(args.output_csv, index=False)
        return

    embedder = load_plm_embedder(args.plm_model)
    table = predict_sequences_only(
        sequences_df,
        plm_only_bundle_path=args.plm_only_bundle,
        calibration_csv_path=args.calibration_csv,
        plm_model=args.plm_model,
        embedder=embedder,
        plm_batch_size=args.plm_batch_size,
    )
    if not args.keep_all:
        before = len(table)
        table = _filter_by_calibrated_probability(table, args.min_p_keep)
        logger.info(
            "Dropped %d/%d rows below min_p_keep=%.3f (use --keep-all to retain)",
            before - len(table),
            before,
            args.min_p_keep,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d predictions to %s", len(table), args.output_csv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
