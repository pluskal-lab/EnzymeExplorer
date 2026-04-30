"""Predict TPS class probabilities using sequences + structures.

Pipeline:
  1. Load sequences (FASTA or CSV) and detect TPS-family domains in the
     supplied PDB/CIF structures.
  2. Foldseek-align detected domains against a reference set (the same
     reference pickle used at training time) to produce per-protein domain
     features.
  3. Embed sequences with the PLM and run the PlmDomainsRandomForest fold
     ensemble; proteins whose domain features are not meaningful fall back to
     the PlmRandomForest (PLM-only) ensemble in a second pass.
  4. Assign per-class confidence tiers from confidence_tiers.csv (rows for
     the matching classifier — PLM_Domains for the structure pass, PLM for
     the fallback pass) and write two CSVs.

The two output CSVs are kept separate because the tier definitions differ
between classifiers; merging them would mean comparing scores across models
that don't share a calibration.
"""

from __future__ import annotations

# --- Environment fixes that must run before any heavy import ---
# Force single-threaded BLAS / OpenMP. The structure pipeline forks a
# multiprocessing.Pool of PyMOL workers; if NumPy/SciPy have already spawned
# OpenMP threads in the parent, fork() leaves the children with a corrupt
# OpenMP runtime (only the forking thread survives) and the first ``np.*``,
# ``cmd.*`` or ``tmalign(...)`` call in a worker deadlocks. This must happen
# before NumPy/SciPy/PyMOL are imported.
import os as _os  # noqa: E402

for _omp_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_omp_var, "1")

# Import PyMOL *first* so the conda env's libstdc++.so.6 (with GLIBCXX_3.4.30,
# required by libvtkm_cont-1.8.so.1) is the one loaded into the process. If
# numpy / pandas / BioPython are imported before PyMOL, the dynamic loader
# resolves libstdc++ to the system copy (older GLIBCXX) and PyMOL's import
# then fails. The standalone ``detect_domains`` CLI works without setting
# LD_LIBRARY_PATH for the same reason — its module imports PyMOL on line 11,
# before anything else.
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

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
    DEFAULT_PLM_DOMAINS_BUNDLE,
    DEFAULT_PLM_ONLY_BUNDLE,
    DEFAULT_REFERENCE_DOMAINS_PICKLE,
    DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
    DEFAULT_TIERS_CSV,
    predict_with_structures,
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
        "--structures-dir",
        required=True,
        type=Path,
        help="Directory of PDB/CIF structures (one per protein, named <id>.pdb).",
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
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Output directory; two CSVs will be written: "
            "predictions_plm_domains.csv and predictions_plm_only_fallback.csv."
        ),
    )
    parser.add_argument(
        "--plm-domains-bundle",
        default=DEFAULT_PLM_DOMAINS_BUNDLE,
        type=Path,
        help="Pickle bundle of PlmDomainsRandomForest fold classifiers.",
    )
    parser.add_argument(
        "--plm-only-bundle",
        default=DEFAULT_PLM_ONLY_BUNDLE,
        type=Path,
        help="Pickle bundle of PlmRandomForest fold classifiers (fallback).",
    )
    parser.add_argument(
        "--reference-domains-pickle",
        default=DEFAULT_REFERENCE_DOMAINS_PICKLE,
        type=Path,
        help="Pickled training reference-domain regions for foldseek alignment.",
    )
    parser.add_argument(
        "--reference-domains-structures-dir",
        default=DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
        type=Path,
        help="Directory of training reference-domain PDB structures.",
    )
    parser.add_argument(
        "--tiers-csv",
        default=DEFAULT_TIERS_CSV,
        type=Path,
        help="confidence_tiers.csv to look up per-class score bands.",
    )
    parser.add_argument(
        "--plm-model",
        default="esm-1v-finetuned-subseq",
        help="PLM model name for the structure pass.",
    )
    parser.add_argument(
        "--plm-only-model",
        default="esm-1v-finetuned-subseq",
        help="PLM model name for the PLM-only fallback pass.",
    )
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--plm-batch-size", type=int, default=4)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Scratch directory for intermediate files (domain detection, "
            "foldseek alignments). Defaults to a temp dir cleaned on exit."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Don't delete the scratch directory on exit.",
    )
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
        name="predict_with_structures", log_dir=args.log_dir
    )
    logger.info("Logging this run to %s", log_path)

    sequences_df = load_sequences(
        args.sequences, id_col=args.id_column, seq_col=args.sequence_column
    )
    logger.info("Loaded %d sequences from %s", len(sequences_df), args.sequences)

    plm_domains_table, plm_only_fallback = predict_with_structures(
        sequences_df,
        structures_dir=args.structures_dir,
        reference_domains_pickle=args.reference_domains_pickle,
        reference_domains_structures_dir=args.reference_domains_structures_dir,
        plm_domains_bundle_path=args.plm_domains_bundle,
        plm_only_bundle_path=args.plm_only_bundle,
        tiers_csv_path=args.tiers_csv,
        plm_model=args.plm_model,
        plm_only_model=args.plm_only_model,
        n_jobs=args.n_jobs,
        plm_batch_size=args.plm_batch_size,
        workdir=args.workdir,
        keep_intermediate=args.keep_intermediate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plm_domains_path = args.output_dir / "predictions_plm_domains.csv"
    fallback_path = args.output_dir / "predictions_plm_only_fallback.csv"

    plm_domains_table.to_csv(plm_domains_path, index=False)
    logger.info(
        "Wrote %d PLM_Domains predictions to %s",
        len(plm_domains_table),
        plm_domains_path,
    )

    plm_only_fallback.to_csv(fallback_path, index=False)
    logger.info(
        "Wrote %d PLM-only fallback predictions to %s",
        len(plm_only_fallback),
        fallback_path,
    )


if __name__ == "__main__":
    main()
