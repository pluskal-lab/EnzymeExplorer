"""``enzyme_explorer_main predict`` subcommand.

Unified entry point for the two prediction flavours:

* Default — structure-aware prediction via :func:`predict_with_structures`.
  Requires ``--structures-dir``.
* ``--no-structures`` — sequence-only prediction via
  :func:`predict_sequences_only`. Runs against the PLM-only ensemble
  (no domain detection, no foldseek alignment).

Design: this file only *composes* the subcommand; the underlying
orchestration lives in :mod:`enzymeexplorer.src.prediction.pipeline`.
The dedicated console scripts ``predict_with_structures`` and
``predict_sequences_only`` continue to work standalone.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def add_predict_subparser(subparsers: argparse._SubParsersAction) -> None:
    from enzymeexplorer.src.prediction.inputs import (
        DEFAULT_ID_COLUMN,
        DEFAULT_SEQUENCE_COLUMN,
    )
    from enzymeexplorer.src.prediction.logging_setup import DEFAULT_LOG_DIR
    from enzymeexplorer.src.prediction.pipeline import (
        DEFAULT_CALIBRATION_CSV,
        DEFAULT_PLM_DOMAINS_BUNDLE,
        DEFAULT_PLM_MODEL,
        DEFAULT_PLM_ONLY_BUNDLE,
        DEFAULT_REFERENCE_DOMAINS_PICKLE,
        DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
    )
    from enzymeexplorer.src.utils.project_info import get_default_workdir_parent

    parser = subparsers.add_parser(
        "predict",
        help=(
            "Predict TPS class probabilities from sequences (default: "
            "structure-aware; --no-structures for sequence-only)."
        ),
    )
    parser.set_defaults(cmd="predict")

    parser.add_argument(
        "--sequences", required=True, type=Path,
        help="Path to sequences (.fasta/.fa/.faa or .csv).",
    )
    parser.add_argument(
        "--no-structures", action="store_true",
        help=(
            "Run sequence-only prediction (PLM ensemble, no domain "
            "detection). --structures-dir is ignored in this mode."
        ),
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=None,
        help=(
            "Directory of PDB/CIF structures (one per protein, named "
            "<id>.pdb). Required unless --no-structures is set."
        ),
    )
    parser.add_argument(
        "--id-column", default=DEFAULT_ID_COLUMN,
        help=f"CSV column name for protein IDs (default: {DEFAULT_ID_COLUMN}).",
    )
    parser.add_argument(
        "--sequence-column", default=DEFAULT_SEQUENCE_COLUMN,
        help=(
            f"CSV column name for amino-acid sequences "
            f"(default: {DEFAULT_SEQUENCE_COLUMN})."
        ),
    )

    # Output: --output-dir writes both frames (structure mode) or is used
    # as the parent of a single CSV (sequence-only mode uses --output-csv
    # directly).
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Structure mode: directory for predictions_plm_domains.csv "
            "and predictions_plm_only_fallback.csv. Required unless "
            "--no-structures is set."
        ),
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help=(
            "Sequence-only mode: output CSV path. Required with "
            "--no-structures."
        ),
    )

    parser.add_argument(
        "--plm-domains-bundle", default=DEFAULT_PLM_DOMAINS_BUNDLE, type=Path,
        help="Pickle bundle of PlmDomainsRandomForest fold classifiers.",
    )
    parser.add_argument(
        "--plm-only-bundle", default=DEFAULT_PLM_ONLY_BUNDLE, type=Path,
        help="Pickle bundle of PlmRandomForest fold classifiers.",
    )
    parser.add_argument(
        "--reference-domains-pickle", default=DEFAULT_REFERENCE_DOMAINS_PICKLE,
        type=Path,
        help="Pickled training reference-domain regions for foldseek alignment.",
    )
    parser.add_argument(
        "--reference-domains-structures-dir",
        default=DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR, type=Path,
        help="Directory of training reference-domain PDB structures.",
    )
    parser.add_argument(
        "--calibration-csv", default=DEFAULT_CALIBRATION_CSV, type=Path,
        help="Per-class beta-calibration fit_summary CSV.",
    )
    parser.add_argument(
        "--plm-model", default=DEFAULT_PLM_MODEL,
        help=f"PLM model name for the main pass (default: {DEFAULT_PLM_MODEL}).",
    )
    parser.add_argument(
        "--plm-only-model", default=DEFAULT_PLM_MODEL,
        help=(
            f"PLM model name for the PLM-only fallback pass "
            f"(default: {DEFAULT_PLM_MODEL}). Ignored with --no-structures."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--plm-batch-size", type=int, default=4)

    parser.add_argument(
        "--workdir", type=Path, default=get_default_workdir_parent(),
        help=(
            "Parent directory for the per-invocation scratch dir. "
            "Defaults to <repo>/tmp so the run leaves no state outside "
            "the EnzymeExplorer tree."
        ),
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Don't delete the per-invocation scratch dir on exit.",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR,
        help=(
            f"Directory for the timestamped run log "
            f"(default: {DEFAULT_LOG_DIR})."
        ),
    )

    # Domain-detection knobs — all OFF by default (maximum sensitivity,
    # no heuristic filtering, one domain per iteration). Users opt in
    # explicitly. Silently ignored under --no-structures.
    parser.add_argument(
        "--prefilter-pdbs-by-foldseek", action="store_true",
        help=(
            "Opt-in: skip (query × template) USalign pairs with no "
            "plausible foldseek alignment. Structure mode only."
        ),
    )
    parser.add_argument(
        "--postfilter-domains-by-foldseek", action="store_true",
        help=(
            "Opt-in: drop detected domains whose foldseek e-value to "
            "any reference exceeds the postfilter threshold. Structure "
            "mode only."
        ),
    )
    parser.add_argument(
        "--detect-multiple-domains-in-each-iteration", action="store_true",
        help=(
            "Opt-in: extract every template match per iteration (higher "
            "recall on multi-domain proteins, slower). Structure mode only."
        ),
    )


def _run_predict_with_structures(args: argparse.Namespace) -> None:
    # Import PyMOL first — see the module docstring of
    # ``predict_with_structures.py`` for why (libstdc++ / GLIBCXX).
    from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401

    from enzymeexplorer.src.prediction.inputs import load_sequences
    from enzymeexplorer.src.prediction.pipeline import predict_with_structures
    from enzymeexplorer.src.utils.managed_workdir import managed_workdir
    from enzymeexplorer.src.utils.signal_handling import graceful_shutdown

    with graceful_shutdown(name="predict_with_structures"), \
            managed_workdir(args.workdir, keep=args.keep_intermediate):
        sequences_df = load_sequences(
            args.sequences, id_col=args.id_column, seq_col=args.sequence_column,
        )
        logger.info(
            "Loaded %d sequences from %s", len(sequences_df), args.sequences,
        )
        plm_domains_table, plm_only_fallback = predict_with_structures(
            sequences_df,
            structures_dir=args.structures_dir,
            reference_domains_pickle=args.reference_domains_pickle,
            reference_domains_structures_dir=args.reference_domains_structures_dir,
            plm_domains_bundle_path=args.plm_domains_bundle,
            plm_only_bundle_path=args.plm_only_bundle,
            calibration_csv_path=args.calibration_csv,
            plm_model=args.plm_model,
            plm_only_model=args.plm_only_model,
            n_jobs=args.n_jobs,
            plm_batch_size=args.plm_batch_size,
            workdir=None,
            keep_intermediate=False,
            prefilter_pdbs_by_foldseek=args.prefilter_pdbs_by_foldseek,
            postfilter_domains_by_foldseek=args.postfilter_domains_by_foldseek,
            detect_multiple_domains_in_each_iteration=(
                args.detect_multiple_domains_in_each_iteration
            ),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plm_domains_path = args.output_dir / "predictions_plm_domains.csv"
        fallback_path = args.output_dir / "predictions_plm_only_fallback.csv"
        plm_domains_table.to_csv(plm_domains_path, index=False)
        logger.info(
            "Wrote %d PLM_Domains predictions to %s",
            len(plm_domains_table), plm_domains_path,
        )
        plm_only_fallback.to_csv(fallback_path, index=False)
        logger.info(
            "Wrote %d PLM-only fallback predictions to %s",
            len(plm_only_fallback), fallback_path,
        )


def _run_predict_sequences_only(args: argparse.Namespace) -> None:
    from enzymeexplorer.src.prediction.inputs import load_sequences
    from enzymeexplorer.src.prediction.pipeline import predict_sequences_only
    from enzymeexplorer.src.utils.managed_workdir import managed_workdir
    from enzymeexplorer.src.utils.signal_handling import graceful_shutdown

    with graceful_shutdown(name="predict_sequences_only"), \
            managed_workdir(args.workdir, keep=args.keep_intermediate):
        sequences_df = load_sequences(
            args.sequences, id_col=args.id_column, seq_col=args.sequence_column,
        )
        logger.info(
            "Loaded %d sequences from %s", len(sequences_df), args.sequences,
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


def run_predict(args: argparse.Namespace) -> None:
    """Dispatch on ``--no-structures``."""
    from enzymeexplorer.src.prediction.logging_setup import configure_logging

    if args.no_structures:
        if args.output_csv is None:
            raise ValueError(
                "--output-csv is required in --no-structures mode."
            )
        log_path = configure_logging(
            name="predict_sequences_only", log_dir=args.log_dir,
        )
        logger.info("Logging this run to %s", log_path)
        _run_predict_sequences_only(args)
        return

    if args.structures_dir is None:
        raise ValueError(
            "--structures-dir is required unless --no-structures is set."
        )
    if args.output_dir is None:
        raise ValueError(
            "--output-dir is required for structure-aware prediction."
        )
    log_path = configure_logging(
        name="predict_with_structures", log_dir=args.log_dir,
    )
    logger.info("Logging this run to %s", log_path)
    _run_predict_with_structures(args)
