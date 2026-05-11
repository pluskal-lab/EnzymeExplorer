"""TPS screening worker — score one FASTA batch.

Designed for large-scale screening (hundreds of millions of sequences).
One worker process handles one disjoint ``[start_i, end_i)`` slice of
the input FASTA on a single GPU. The SLURM-side orchestration lives in
``scripts/tps_screening_manager.sh`` (top-level CPU job that fans out
one per-batch GPU task plus optional per-batch AF-DB download and
cleanup tasks).

Two classifiers are available; the ``--classifier`` flag picks which
one(s) the worker runs:

* ``plm`` (default) — ``PlmRandomForest`` on PLM embeddings only. Pure
  sequence input, scales to UniProt-size FASTAs.
* ``plm_domains`` — ``PlmDomainsRandomForest`` using PLM embeddings +
  structural domain features. Requires a PDB per scored sequence,
  either pre-staged via ``--structures-dir`` or downloaded on-the-fly
  from AlphaFold DB (sequence IDs are then treated as UniProt
  accessions). Sequences with no AF-DB entry are excluded and
  recorded in the ``no_structure`` per-shard CSV.
* ``both`` — runs both pipelines independently; a failure in one does
  not abort the other.

Per-batch output layout under ``--output-dir`` (sub-dir per classifier,
file per batch — the manager passes ``--shard-name batch_<idx>``)::

    plm/batch_<idx>.csv                  — PlmRandomForest predictions
                                            (all sequences in batch)
    plm_domains/batch_<idx>.csv          — PlmDomainsRandomForest (subset
                                            whose structures were available
                                            and had a valid TPS-family domain)
    plm_domains_fallback/batch_<idx>.csv — PLM-only fallback rows produced
                                            by predict_with_structures for
                                            proteins whose structures were
                                            available but yielded no valid
                                            TPS-family domain
    no_structure/batch_<idx>.csv         — IDs requested for plm_domains
                                            scoring but missing from AF-DB

Filtering is disabled by default — every scored row is written.
Pass ``--min-p-keep <float>`` to enable the screening compression
(drop rows where every ``*_p_calibrated`` is below the threshold).
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
    DEFAULT_PLM_DOMAINS_BUNDLE,
    DEFAULT_PLM_MODEL,
    DEFAULT_PLM_ONLY_BUNDLE,
    DEFAULT_REFERENCE_DOMAINS_PICKLE,
    DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
    predict_sequences_only,
    predict_with_structures,
)

logger = logging.getLogger(__name__)

CLASSIFIER_CHOICES = ("plm", "plm_domains", "both")

# Sub-directory names under --output-dir.
SUBDIR_PLM = "plm"
SUBDIR_PLM_DOMAINS = "plm_domains"
SUBDIR_PLM_DOMAINS_FALLBACK = "plm_domains_fallback"
SUBDIR_NO_STRUCTURE = "no_structure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path", required=True, type=Path,
        help="Input FASTA. The worker reads only [start-i, end-i).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help=(
            "Per-shard outputs land at <output-dir>/<classifier>/"
            "<shard-name>.csv. See module docstring for the full layout."
        ),
    )
    parser.add_argument(
        "--shard-name", required=True, type=str,
        help=(
            "Stem for every per-classifier CSV this worker writes "
            "(e.g. 'session_001_gpu_3_000000000_000040000'). The "
            "launcher generates these so all shards from one screening "
            "are co-locatable."
        ),
    )
    parser.add_argument(
        "--classifier", choices=CLASSIFIER_CHOICES, default="plm",
        help=(
            f"Classifier(s) to run. Default 'plm'. Use 'plm_domains' or "
            f"'both' to also run the structure-aware classifier."
        ),
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=None,
        help=(
            "Directory of pre-downloaded <uniprot_id>.pdb files for the "
            "plm_domains classifier. When omitted, missing structures are "
            "downloaded from AlphaFold DB inline (CPU-bound; prefer the "
            "dedicated `tps_download_af_structures` SLURM job to keep "
            "this off the GPU node)."
        ),
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
        "--plm-domains-bundle", type=Path, default=Path(DEFAULT_PLM_DOMAINS_BUNDLE),
    )
    parser.add_argument(
        "--calibration-csv", type=Path, default=Path(DEFAULT_CALIBRATION_CSV),
    )
    parser.add_argument(
        "--reference-domains-pickle",
        type=Path, default=Path(DEFAULT_REFERENCE_DOMAINS_PICKLE),
    )
    parser.add_argument(
        "--reference-domains-structures-dir",
        type=Path, default=Path(DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR),
    )
    parser.add_argument(
        "--plm-model", type=str, default=DEFAULT_PLM_MODEL,
    )
    parser.add_argument("--plm-batch-size", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument(
        "--af-db-workers", type=int, default=16,
        help="Concurrent AF-DB downloads when --structures-dir is omitted.",
    )
    parser.add_argument(
        "--min-p-keep", type=float, default=None,
        help=(
            "Optional filter: drop rows where every <class>_p_calibrated "
            "is below this value. Disabled by default (every scored row "
            "is written). Applies to both plm and plm_domains outputs."
        ),
    )
    parser.add_argument(
        "--workdir", type=Path, default=None,
        help=(
            "Parent directory for the per-invocation scratch dir. "
            "Inline AF-DB downloads also stage under it. Defaults to "
            "the system tmp directory."
        ),
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Don't delete the per-invocation scratch dir on exit.",
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
    has_above = np.nansum(arr >= min_p, axis=1) > 0
    all_nan = np.all(np.isnan(arr), axis=1)
    keep = has_above | all_nan
    return table[keep].reset_index(drop=True)


def _write_csv(table: pd.DataFrame, subdir: Path, shard_name: str) -> Path:
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"{shard_name}.csv"
    table.to_csv(out, index=False)
    return out


def _resolve_structures(
    sequences_df: pd.DataFrame,
    *,
    structures_dir: Path | None,
    download_dir: Path,
    n_workers: int,
) -> tuple[Path, list[str], list[str]]:
    """Return (effective_structures_dir, ids_with_structure, ids_missing).

    If ``structures_dir`` is provided, we just scan it for ``<id>.pdb``
    files. Otherwise we download from AF-DB into ``download_dir``.
    """
    ids = sequences_df["id"].tolist()
    if structures_dir is not None:
        present = [
            uid for uid in ids
            if (structures_dir / f"{uid}.pdb").is_file()
        ]
        missing = [uid for uid in ids if uid not in set(present)]
        return structures_dir, present, missing

    # Inline AF-DB download path.
    from enzymeexplorer.src.screening.af_db import download_many
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded, missing_set = download_many(
        ids, download_dir, n_workers=n_workers,
    )
    present = [uid for uid in ids if uid in downloaded]
    missing = [uid for uid in ids if uid in missing_set]
    return download_dir, present, missing


def _run_plm(
    sequences_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    embedder,
) -> pd.DataFrame:
    """Run the PLM-only classifier on every sequence in the shard."""
    return predict_sequences_only(
        sequences_df,
        plm_only_bundle_path=args.plm_only_bundle,
        calibration_csv_path=args.calibration_csv,
        plm_model=args.plm_model,
        embedder=embedder,
        plm_batch_size=args.plm_batch_size,
    )


def _run_plm_domains(
    sequences_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    structures_dir: Path,
    embedder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the structure-aware classifier on the sequences whose PDBs
    are present in ``structures_dir``. Returns ``(plm_domains_table,
    plm_only_fallback)`` exactly as ``predict_with_structures`` does."""
    return predict_with_structures(
        sequences_df,
        structures_dir=structures_dir,
        reference_domains_pickle=args.reference_domains_pickle,
        reference_domains_structures_dir=args.reference_domains_structures_dir,
        plm_domains_bundle_path=args.plm_domains_bundle,
        plm_only_bundle_path=args.plm_only_bundle,
        calibration_csv_path=args.calibration_csv,
        plm_model=args.plm_model,
        plm_only_model=args.plm_model,
        n_jobs=args.n_jobs,
        plm_batch_size=args.plm_batch_size,
        embedder=embedder,
        # workdir stays None so the inner tempfile.X calls inherit the
        # managed_workdir-swapped tempfile.tempdir.
        workdir=None,
        keep_intermediate=False,
    )


def main(args: argparse.Namespace) -> None:
    from enzymeexplorer.src.utils.managed_workdir import managed_workdir

    with managed_workdir(args.workdir, keep=args.keep_intermediate) as wd:
        sequences_df = _load_fasta_shard(
            args.fasta_path, args.start_i, args.end_i
        )
        logger.info(
            "Loaded shard [%d, %s) from %s — %d sequences",
            args.start_i,
            args.end_i if args.end_i is not None else "end",
            args.fasta_path,
            len(sequences_df),
        )

        # Empty schemas the gather step can rely on for missing files.
        if sequences_df.empty:
            _write_csv(
                pd.DataFrame(columns=["id", "sequence"]),
                args.output_dir / SUBDIR_PLM, args.shard_name,
            ) if args.classifier in ("plm", "both") else None
            _write_csv(
                pd.DataFrame(columns=["id", "sequence"]),
                args.output_dir / SUBDIR_PLM_DOMAINS, args.shard_name,
            ) if args.classifier in ("plm_domains", "both") else None
            return

        # Shared PLM embedder so both classifiers reuse the same loaded
        # checkpoint instead of paying the load cost twice.
        embedder = load_plm_embedder(args.plm_model)

        # ---------------- PLM (sequences only) ----------------------
        if args.classifier in ("plm", "both"):
            try:
                table = _run_plm(sequences_df, args=args, embedder=embedder)
                if args.min_p_keep is not None:
                    before = len(table)
                    table = _filter_by_calibrated_probability(
                        table, args.min_p_keep,
                    )
                    logger.info(
                        "PLM: dropped %d/%d rows below min_p_keep=%.3f",
                        before - len(table), before, args.min_p_keep,
                    )
                out = _write_csv(
                    table, args.output_dir / SUBDIR_PLM, args.shard_name,
                )
                logger.info("PLM: wrote %d predictions to %s", len(table), out)
            except Exception:
                logger.exception(
                    "PLM classifier failed for shard %s — continuing with "
                    "other classifiers", args.shard_name,
                )

        # ---------------- PlmDomains (structure-aware) -------------
        if args.classifier in ("plm_domains", "both"):
            try:
                # Resolve structures dir (provided or downloaded).
                inline_dl_dir = (
                    Path(wd) / "af_db_pdbs" if wd is not None
                    else Path("./.af_db_pdbs")
                )
                eff_dir, ids_present, ids_missing = _resolve_structures(
                    sequences_df,
                    structures_dir=args.structures_dir,
                    download_dir=inline_dl_dir,
                    n_workers=args.af_db_workers,
                )
                logger.info(
                    "plm_domains: %d/%d sequences have structures (missing %d)",
                    len(ids_present), len(sequences_df), len(ids_missing),
                )

                # Always emit the no_structure CSV so the launcher can
                # aggregate without checking for missing files.
                _write_csv(
                    pd.DataFrame({"id": ids_missing}),
                    args.output_dir / SUBDIR_NO_STRUCTURE, args.shard_name,
                )

                if not ids_present:
                    logger.info(
                        "plm_domains: nothing to score (no structures); "
                        "writing empty plm_domains shard",
                    )
                    _write_csv(
                        pd.DataFrame(columns=["id", "sequence"]),
                        args.output_dir / SUBDIR_PLM_DOMAINS, args.shard_name,
                    )
                    _write_csv(
                        pd.DataFrame(columns=["id", "sequence"]),
                        args.output_dir / SUBDIR_PLM_DOMAINS_FALLBACK,
                        args.shard_name,
                    )
                else:
                    subset = sequences_df[
                        sequences_df["id"].isin(set(ids_present))
                    ].reset_index(drop=True)
                    plm_domains_table, fallback_table = _run_plm_domains(
                        subset,
                        args=args,
                        structures_dir=eff_dir,
                        embedder=embedder,
                    )
                    for tab, subdir, label in (
                        (plm_domains_table, SUBDIR_PLM_DOMAINS, "plm_domains"),
                        (fallback_table, SUBDIR_PLM_DOMAINS_FALLBACK,
                         "plm_domains_fallback"),
                    ):
                        if args.min_p_keep is not None:
                            before = len(tab)
                            tab = _filter_by_calibrated_probability(
                                tab, args.min_p_keep,
                            )
                            logger.info(
                                "%s: dropped %d/%d rows below "
                                "min_p_keep=%.3f",
                                label, before - len(tab), before,
                                args.min_p_keep,
                            )
                        out = _write_csv(
                            tab, args.output_dir / subdir, args.shard_name,
                        )
                        logger.info(
                            "%s: wrote %d predictions to %s",
                            label, len(tab), out,
                        )
            except Exception:
                logger.exception(
                    "plm_domains classifier failed for shard %s — "
                    "continuing", args.shard_name,
                )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
