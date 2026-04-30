"""TPS screening worker — score one FASTA shard and write a CSV.

Designed for large-scale screening (hundreds of millions of sequences).
The :mod:`tps_screening_cluster_launcher` spawns one of these per GPU,
each with a disjoint ``[start_i, end_i)`` slice of the input FASTA.

Internals delegate to :func:`enzymeexplorer.src.prediction.pipeline.predict_sequences_only`
so the screening output matches the rest of the prediction pipeline:
``id``, ``sequence``, and a ``<class>_score`` + ``<class>_tier`` column per
class. By default rows where every class lands in the ``Negative`` tier are
dropped to keep the per-shard CSV manageable; pass ``--keep-negatives`` to
disable.
"""

from __future__ import annotations

import argparse
import logging
from itertools import islice
from pathlib import Path

import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore

from enzymeexplorer.src.prediction.embeddings import load_plm_embedder
from enzymeexplorer.src.prediction.pipeline import (
    DEFAULT_PLM_ONLY_BUNDLE,
    DEFAULT_TIERS_CSV,
    predict_sequences_only,
)
from enzymeexplorer.src.prediction.tiers import NEGATIVE_TIER

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
        "--tiers-csv", type=Path, default=Path(DEFAULT_TIERS_CSV),
    )
    parser.add_argument(
        "--plm-model", type=str, default="esm-1v-finetuned-subseq",
    )
    parser.add_argument("--plm-batch-size", type=int, default=4)
    parser.add_argument(
        "--keep-negatives", action="store_true",
        help=(
            "Keep rows where every class is in the Negative tier. By default "
            "such rows are dropped so per-shard CSVs stay small."
        ),
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


def _drop_all_negative_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every ``*_tier`` column equals ``Negative``."""
    tier_cols = [c for c in table.columns if c.endswith("_tier")]
    if not tier_cols:
        return table
    keep = (table[tier_cols] != NEGATIVE_TIER).any(axis=1)
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
        tiers_csv_path=args.tiers_csv,
        plm_model=args.plm_model,
        embedder=embedder,
        plm_batch_size=args.plm_batch_size,
    )
    if not args.keep_negatives:
        before = len(table)
        table = _drop_all_negative_rows(table)
        logger.info(
            "Dropped %d/%d all-Negative rows (use --keep-negatives to retain)",
            before - len(table),
            before,
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
