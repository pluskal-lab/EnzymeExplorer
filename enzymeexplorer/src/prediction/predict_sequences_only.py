"""Predict TPS class probabilities from sequences alone (no structures).

Uses the PlmRandomForest fold ensemble. Output is a single CSV with
``<class>_score`` and ``<class>_p_calibrated`` columns per protein, with
calibrated probabilities produced from the ``PLM`` rows of
``calibration_fit_summary.csv``.
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
    DEFAULT_CALIBRATION_CSV,
    DEFAULT_PLM_MODEL,
    DEFAULT_PLM_ONLY_BUNDLE,
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
        "--calibration-csv",
        default=DEFAULT_CALIBRATION_CSV,
        type=Path,
        help="Per-class beta-calibration fit_summary CSV.",
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
    from enzymeexplorer.src.utils.project_info import get_default_workdir_parent

    parser.add_argument(
        "--workdir",
        type=Path,
        default=get_default_workdir_parent(),
        help=(
            "Parent directory for the per-invocation scratch dir. "
            "Anything written under it (incl. every tempfile.X call) "
            "is removed on exit. Defaults to <repo>/tmp so the run "
            "leaves no state outside the EnzymeExplorer tree."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Don't delete the per-invocation scratch dir on exit.",
    )
    return parser.parse_args()


def main() -> None:
    from enzymeexplorer.src.utils.managed_workdir import managed_workdir
    from enzymeexplorer.src.utils.signal_handling import graceful_shutdown

    args = parse_args()
    log_path = configure_logging(
        name="predict_sequences_only", log_dir=args.log_dir
    )
    logger.info("Logging this run to %s", log_path)

    with graceful_shutdown(name="predict_sequences_only"), \
            managed_workdir(args.workdir, keep=args.keep_intermediate):
        sequences_df = load_sequences(
            args.sequences, id_col=args.id_column, seq_col=args.sequence_column
        )
        logger.info(
            "Loaded %d sequences from %s", len(sequences_df), args.sequences
        )

        table = predict_sequences_only(
            sequences_df,
            plm_only_bundle_path=args.plm_only_bundle,
            calibration_csv_path=args.calibration_csv,
            plm_model=args.plm_model,
            plm_batch_size=args.plm_batch_size,
        )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        logger.info("Wrote %d predictions to %s", len(table), args.output_csv)


if __name__ == "__main__":
    main()
