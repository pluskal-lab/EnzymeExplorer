"""Predict TPS class probabilities from sequences alone (no structures).

Uses the PlmRandomForest fold ensemble. Output is a single CSV with
``<class>_score`` and ``<class>_tier`` columns per protein, with tiers looked
up from the ``PLM`` rows of ``confidence_tiers.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from enzymeexplorer.src.prediction.inputs import (
    DEFAULT_ID_COLUMN,
    DEFAULT_SEQUENCE_COLUMN,
    load_sequences,
)
from enzymeexplorer.src.prediction.logging_setup import (
    DEFAULT_LOG_DIR,
    configure_logging,
)
from enzymeexplorer.src.prediction.pipeline import (
    DEFAULT_PLM_MODEL,
    DEFAULT_PLM_ONLY_BUNDLE,
    DEFAULT_TIERS_CSV,
    predict_sequences_only,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        required=True,
        type=Path,
        help="Path to sequences (.fasta/.fa/.faa or .csv).",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help=f"CSV column name for protein IDs (default: {DEFAULT_ID_COLUMN}).",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help=(
            f"CSV column name for amino-acid sequences "
            f"(default: {DEFAULT_SEQUENCE_COLUMN})."
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plm-only-bundle",
        default=DEFAULT_PLM_ONLY_BUNDLE,
        type=Path,
        help="Pickle bundle of PlmRandomForest fold classifiers.",
    )
    parser.add_argument(
        "--tiers-csv",
        default=DEFAULT_TIERS_CSV,
        type=Path,
        help="confidence_tiers.csv to look up per-class score bands.",
    )
    parser.add_argument(
        "--plm-model",
        default=DEFAULT_PLM_MODEL,
        help=f"PLM model name (default: {DEFAULT_PLM_MODEL}).",
    )
    parser.add_argument("--plm-batch-size", type=int, default=4)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=(
            f"Directory for the timestamped run log "
            f"(default: {DEFAULT_LOG_DIR})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(
        name="predict_sequences_only", log_dir=args.log_dir
    )
    logger.info("Logging this run to %s", log_path)

    sequences_df = load_sequences(
        args.sequences, id_col=args.id_column, seq_col=args.sequence_column
    )
    logger.info("Loaded %d sequences from %s", len(sequences_df), args.sequences)

    table = predict_sequences_only(
        sequences_df,
        plm_only_bundle_path=args.plm_only_bundle,
        tiers_csv_path=args.tiers_csv,
        plm_model=args.plm_model,
        plm_batch_size=args.plm_batch_size,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d predictions to %s", len(table), args.output_csv)


if __name__ == "__main__":
    main()
