"""High-level prediction entry points.

Two orchestrators:

* :func:`predict_with_structures` — domain-aware (PLM_Domains) for proteins
  whose detected domains pass the foldseek-meaningfulness threshold; PLM-only
  for the rest. Two output frames so each can carry its own
  classifier-specific calibrated probabilities.
* :func:`predict_sequences_only` — PLM-only over all input proteins.

Both return wide-form tables with one row per protein and
``<class>_raw`` / ``<class>_p`` columns; both also accept a
callable ``embedder`` so the FastAPI app can reuse a long-lived embedder
across requests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd  # type: ignore

from enzymeexplorer.src.prediction import calibration as _calibration
from enzymeexplorer.src.prediction import domains as _domains
from enzymeexplorer.src.prediction import ensemble as _ens

# NB: ``embeddings`` is imported lazily inside ``_ensure_embedder`` because
# importing it eagerly pulls in PyTorch at module-load time. The structure
# prediction path forks a multiprocessing.Pool of PyMOL workers; if PyTorch's
# CPU thread pool has been initialised in the parent process before the fork
# (which happens at *import* time, not just at first tensor op), the forked
# children inherit corrupted thread state and deadlock indefinitely. Keeping
# the embeddings import lazy lets the structure pipeline run with a clean,
# torch-free parent process.
if TYPE_CHECKING:
    from enzymeexplorer.src.prediction.embeddings import PLMEmbedder

logger = logging.getLogger(__name__)


# Default data-file paths are resolved against the repo's ``data/``
# directory (see :func:`enzymeexplorer.src.utils.project_info.get_data_root`),
# *not* the current working directory. This is what lets the
# ``predict_sequences_only`` / ``predict_with_structures`` console
# scripts (installed via ``pip install -e .``) run from any directory
# without hitting FileNotFoundError on the bundled checkpoints.
from enzymeexplorer.src.utils.project_info import get_data_root as _get_data_root

_DATA = _get_data_root()
DEFAULT_CALIBRATION_CSV = str(_DATA / "calibration_fit_summary.csv")
DEFAULT_PLM_MODEL = "ankh_large"
DEFAULT_PLM_DOMAINS_BUNDLE = str(_DATA / "enzyme_explorer_checkpoints.pkl")
DEFAULT_PLM_ONLY_BUNDLE = str(_DATA / "enzyme_explorer_plm_checkpoints.pkl")
DEFAULT_REFERENCE_DOMAINS_PICKLE = str(
    _DATA / "detected_domains" / "martsDB_detected_domains"
    / "martsDB_detected_domains.pkl"
)
DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR = str(
    _DATA / "detected_domains" / "martsDB_detected_domains" / "domains"
)

PLM_DOMAINS_CLASSIFIER_NAME = "PLM_Domains"
PLM_ONLY_CLASSIFIER_NAME = "PLM"


def _filter_by_min_tps_p(table: pd.DataFrame, min_tps_p: float) -> pd.DataFrame:
    """Drop rows whose ``TPS_p`` (calibrated probability) is below ``min_tps_p``.

    A missing column or an all-NaN column short-circuits to the input
    table (e.g. fallback frames produced by ``predict_with_structures``
    that didn't run TPS calibration). NaN cells are treated as failing
    the threshold so an unscored row is dropped — calling code can opt
    out by passing ``min_tps_p=None``.
    """
    col = "TPS_p"
    if col not in table.columns:
        return table.reset_index(drop=True)
    # NaN >= x is False, so rows without a TPS calibration drop out.
    vals = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
    keep = vals >= min_tps_p
    return table.loc[keep].reset_index(drop=True)


def _ensure_embedder(
    embedder: "PLMEmbedder | None",
    *,
    model_name: str,
    max_seq_len: int | None = None,
) -> "PLMEmbedder":
    if embedder is not None:
        return embedder
    # Deferred import — see module docstring on why.
    from enzymeexplorer.src.prediction.embeddings import load_plm_embedder

    return load_plm_embedder(model_name, max_seq_len=max_seq_len)


def _score_and_calibrate(
    *,
    per_fold_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    calibration_csv_path: str | Path,
    classifier_name: str,
) -> pd.DataFrame:
    averaged = _ens.average_over_folds(per_fold_df)
    long_with_p = _calibration.apply_calibration_long(
        averaged, calibration_csv_path, classifier_name,
    )
    seq_lookup = dict(zip(sequences_df["id"], sequences_df["sequence"]))
    return _calibration.assemble_output_table(
        long_with_p, sequence_lookup=seq_lookup,
    )


def predict_with_structures(
    sequences_df: pd.DataFrame,
    structures_dir: str | Path,
    *,
    reference_domains_pickle: str | Path = DEFAULT_REFERENCE_DOMAINS_PICKLE,
    reference_domains_structures_dir: str | Path = (
        DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR
    ),
    plm_domains_bundle_path: str | Path = DEFAULT_PLM_DOMAINS_BUNDLE,
    plm_only_bundle_path: str | Path = DEFAULT_PLM_ONLY_BUNDLE,
    calibration_csv_path: str | Path = DEFAULT_CALIBRATION_CSV,
    plm_model: str = DEFAULT_PLM_MODEL,
    plm_only_model: str = DEFAULT_PLM_MODEL,
    embedder: PLMEmbedder | None = None,
    plm_only_embedder: PLMEmbedder | None = None,
    plm_max_seq_len: int | None = None,
    precomputed_embeddings=None,  # np.ndarray | None — aligned with sequences_df rows
    n_jobs: int = 10,
    plm_batch_size: int = 4,
    workdir: str | Path | None = None,
    keep_intermediate: bool = False,
    min_tps_p: float | None = None,
    prefilter_pdbs_by_foldseek: bool = False,
    postfilter_domains_by_foldseek: bool = False,
    detect_multiple_domains_in_each_iteration: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end prediction with structures.

    Returns ``(plm_domains_predictions, plm_only_fallback_predictions)`` —
    two wide-form DataFrames with per-class ``_raw`` and
    ``_p`` columns. The second frame is empty when every
    protein produced a meaningful domain comparison; otherwise it
    contains the proteins that fell back.

    The split between the two output frames is intentional: calibrators
    are classifier-specific, so PLM_Domains and PLM-only scores are
    calibrated against their own ``fit_summary.csv`` rows.
    """
    sequences_df = sequences_df.reset_index(drop=True)
    protein_ids: list[str] = sequences_df["id"].tolist()
    sequences: list[str] = sequences_df["sequence"].tolist()
    logger.info(
        "predict_with_structures: %d input sequences from %s",
        len(protein_ids),
        structures_dir,
    )

    # NB: domain detection runs *before* PLM embeddings on purpose. The
    # detector forks a multiprocessing.Pool of PyMOL workers; if PyTorch's
    # CPU thread pool has already been initialised (which happens the first
    # time we call the embedder), the forked children inherit corrupted
    # thread state and deadlock. Doing the structure work on a torch-free
    # parent process keeps the fork clean.
    logger.info(
        "[step 1/4] Domain detection + foldseek alignment "
        "(structure-pipeline step; runs first to keep multiprocessing.Pool "
        "fork clean of PyTorch state)"
    )
    domain_result = _domains.detect_and_align_domains(
        structures_dir=structures_dir,
        protein_ids=protein_ids,
        reference_domains_pickle=reference_domains_pickle,
        reference_domains_structures_dir=reference_domains_structures_dir,
        workdir=workdir,
        n_jobs=n_jobs,
        keep_intermediate=keep_intermediate,
        prefilter_pdbs_by_foldseek=prefilter_pdbs_by_foldseek,
        postfilter_domains_by_foldseek=postfilter_domains_by_foldseek,
        detect_multiple_domains_in_each_iteration=(
            detect_multiple_domains_in_each_iteration
        ),
    )
    logger.info(
        "[step 1/4] Domain detection done — %d proteins with detections, "
        "structural-features matrix shape %s",
        len(domain_result.query_seq_ids),
        tuple(domain_result.structural_features.shape),
    )

    if precomputed_embeddings is not None:
        # Resumability hook: the screening worker computes embeddings
        # for the FULL FASTA batch once, caches them to disk, and
        # passes them here (sliced to ``sequences_df`` rows) on every
        # subsequent rerun. Skipping the embedder save on the order
        # of an ankh_large forward pass per 40 000 sequences.
        if len(precomputed_embeddings) != len(sequences):
            raise ValueError(
                f"precomputed_embeddings has {len(precomputed_embeddings)} rows "
                f"but sequences_df has {len(sequences)}; they must align 1-to-1."
            )
        embeddings = precomputed_embeddings
        logger.info(
            "[step 2/4] Reusing precomputed PLM embeddings — shape %s",
            tuple(embeddings.shape),
        )
    else:
        logger.info("[step 2/4] Loading PLM embedder (%s)", plm_model)
        embedder = _ensure_embedder(embedder, model_name=plm_model, max_seq_len=plm_max_seq_len)
        logger.info(
            "[step 2/4] Computing PLM embeddings for %d sequences", len(sequences)
        )
        embeddings = embedder.embed(
            sequences, batch_size=plm_batch_size, progress_desc="PLM embeddings"
        )
        logger.info(
            "[step 2/4] PLM embeddings done — shape %s", tuple(embeddings.shape)
        )

    logger.info(
        "[step 3/4] Loading PLM_Domains fold bundle (%s)", plm_domains_bundle_path
    )
    fold_classifiers_dom = _ens.load_fold_bundle(plm_domains_bundle_path)
    logger.info(
        "[step 3/4] Scoring %d proteins through %d folds (PLM_Domains)",
        len(protein_ids),
        len(fold_classifiers_dom),
    )
    per_fold_dom_df, fallback_ids = _ens.predict_with_plm_and_domains(
        embeddings=embeddings,
        protein_ids=protein_ids,
        structural_features=domain_result.structural_features,
        structural_features_ids=domain_result.query_seq_ids,
        fold_classifiers=fold_classifiers_dom,
    )
    logger.info(
        "[step 3/4] PLM_Domains scored %d proteins; %d falling back to PLM-only",
        per_fold_dom_df["id"].nunique() if not per_fold_dom_df.empty else 0,
        len(fallback_ids),
    )

    plm_domains_table = _score_and_calibrate(
        per_fold_df=per_fold_dom_df,
        sequences_df=sequences_df,
        calibration_csv_path=calibration_csv_path,
        classifier_name=PLM_DOMAINS_CLASSIFIER_NAME,
    )

    plm_only_table = pd.DataFrame()
    if fallback_ids:
        logger.info(
            "[step 4/4] Loading PLM-only fold bundle for %d fallback protein(s) (%s)",
            len(fallback_ids),
            plm_only_bundle_path,
        )
        fallback_df = sequences_df[sequences_df["id"].isin(fallback_ids)].reset_index(
            drop=True
        )
        # If the fallback PLM is the same model, reuse the embeddings
        # already computed for the main pass — sliced down to the
        # fallback subset. Crucially we do NOT load a second copy of
        # the embedder model in this branch (production uses
        # ``plm_only_model == plm_model == ankh_large`` and previously
        # this branch paid for an unnecessary multi-GB load).
        if plm_only_model == plm_model:
            id_to_row = {pid: idx for idx, pid in enumerate(protein_ids)}
            row_idx = [id_to_row[pid] for pid in fallback_df["id"]]
            fallback_embeddings = embeddings[row_idx]
            logger.info(
                "[step 4/4] Reusing existing embeddings for fallback "
                "(no embedder reload)"
            )
        else:
            logger.info(
                "[step 4/4] Recomputing PLM embeddings for fallback (%s)",
                plm_only_model,
            )
            plm_only_embedder_inst = _ensure_embedder(
                plm_only_embedder, model_name=plm_only_model, max_seq_len=plm_max_seq_len,
            )
            fallback_embeddings = plm_only_embedder_inst.embed(
                fallback_df["sequence"].tolist(),
                batch_size=plm_batch_size,
                progress_desc="PLM embeddings (fallback)",
            )
        fold_classifiers_plm = _ens.load_fold_bundle(plm_only_bundle_path)
        logger.info(
            "[step 4/4] Scoring %d fallback proteins through %d folds (PLM-only)",
            len(fallback_df),
            len(fold_classifiers_plm),
        )
        per_fold_plm_df = _ens.predict_with_plm_only(
            embeddings=fallback_embeddings,
            protein_ids=fallback_df["id"].tolist(),
            fold_classifiers=fold_classifiers_plm,
        )
        plm_only_table = _score_and_calibrate(
            per_fold_df=per_fold_plm_df,
            sequences_df=fallback_df,
            calibration_csv_path=calibration_csv_path,
            classifier_name=PLM_ONLY_CLASSIFIER_NAME,
        )
    else:
        logger.info("[step 4/4] No fallback proteins; PLM-only step skipped")
    if min_tps_p is not None:
        plm_domains_table = _filter_by_min_tps_p(plm_domains_table, min_tps_p)
        plm_only_table = _filter_by_min_tps_p(plm_only_table, min_tps_p)
        logger.info(
            "predict_with_structures: kept rows with TPS_p >= %.4f — "
            "PLM_Domains=%d, PLM-only=%d",
            min_tps_p, len(plm_domains_table), len(plm_only_table),
        )
    logger.info(
        "predict_with_structures: done — PLM_Domains rows=%d, PLM-only rows=%d",
        len(plm_domains_table),
        len(plm_only_table),
    )
    return plm_domains_table, plm_only_table


def predict_sequences_only(
    sequences_df: pd.DataFrame,
    *,
    plm_only_bundle_path: str | Path = DEFAULT_PLM_ONLY_BUNDLE,
    calibration_csv_path: str | Path = DEFAULT_CALIBRATION_CSV,
    plm_model: str = DEFAULT_PLM_MODEL,
    embedder: PLMEmbedder | None = None,
    plm_max_seq_len: int | None = None,
    precomputed_embeddings=None,  # np.ndarray | None — aligned with sequences_df rows
    plm_batch_size: int = 4,
    min_tps_p: float | None = None,
) -> pd.DataFrame:
    """End-to-end PLM-only prediction. Returns one wide-form DataFrame.

    ``precomputed_embeddings`` (when given) skips the embedder pass —
    used by the screening worker so a rerun can reuse cached
    embeddings instead of paying the ankh_large forward pass twice.
    """
    sequences_df = sequences_df.reset_index(drop=True)
    logger.info("predict_sequences_only: %d input sequences", len(sequences_df))
    if precomputed_embeddings is not None:
        if len(precomputed_embeddings) != len(sequences_df):
            raise ValueError(
                f"precomputed_embeddings has {len(precomputed_embeddings)} rows "
                f"but sequences_df has {len(sequences_df)}; they must align 1-to-1."
            )
        embeddings = precomputed_embeddings
        logger.info(
            "[step 1/3] Reusing precomputed PLM embeddings — shape %s",
            tuple(embeddings.shape),
        )
    else:
        logger.info("[step 1/3] Loading PLM embedder (%s)", plm_model)
        embedder = _ensure_embedder(embedder, model_name=plm_model, max_seq_len=plm_max_seq_len)
        logger.info(
            "[step 1/3] Computing PLM embeddings for %d sequences",
            len(sequences_df),
        )
        embeddings = embedder.embed(
            sequences_df["sequence"].tolist(),
            batch_size=plm_batch_size,
            progress_desc="PLM embeddings",
        )
    logger.info("[step 2/3] Loading PLM-only fold bundle (%s)", plm_only_bundle_path)
    fold_classifiers = _ens.load_fold_bundle(plm_only_bundle_path)
    logger.info(
        "[step 2/3] Scoring %d proteins through %d folds (PLM-only)",
        len(sequences_df),
        len(fold_classifiers),
    )
    per_fold_df = _ens.predict_with_plm_only(
        embeddings=embeddings,
        protein_ids=sequences_df["id"].tolist(),
        fold_classifiers=fold_classifiers,
    )
    logger.info("[step 3/3] Averaging folds + applying calibration")
    table = _score_and_calibrate(
        per_fold_df=per_fold_df,
        sequences_df=sequences_df,
        calibration_csv_path=calibration_csv_path,
        classifier_name=PLM_ONLY_CLASSIFIER_NAME,
    )
    if min_tps_p is not None:
        before = len(table)
        table = _filter_by_min_tps_p(table, min_tps_p)
        logger.info(
            "predict_sequences_only: kept %d/%d rows with TPS_p >= %.4f",
            len(table), before, min_tps_p,
        )
    logger.info("predict_sequences_only: done — %d rows", len(table))
    return table
