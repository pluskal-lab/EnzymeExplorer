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
from pymol import cmd

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
    parser.add_argument(
        "--max-seq-len", type=int, default=2000,
        help=(
            "Drop sequences longer than this many residues before running "
            "predictions. Sharding/indexing is unaffected — the oversized "
            "rows are simply skipped at predict time. Pass 0 (or any "
            "non-positive value) to disable the cap."
        ),
    )
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
    from enzymeexplorer.src.utils.project_info import get_default_workdir_parent

    parser.add_argument(
        "--workdir", type=Path, default=get_default_workdir_parent(),
        help=(
            "Parent directory for the per-invocation scratch dir. "
            "Inline AF-DB downloads also stage under it. Defaults to "
            "<repo>/tmp so per-batch scratch never escapes the "
            "EnzymeExplorer tree."
        ),
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Don't delete the per-invocation scratch dir on exit.",
    )
    parser.add_argument(
        "--embeddings-cache-dir", type=Path, default=None,
        help=(
            "Directory for the persistent PLM-embedding cache. When "
            "set, the worker writes a ``<shard-name>.npy`` (+ sidecar "
            "``.ids.txt``) here after computing embeddings, and on "
            "subsequent runs with the same shard + model + ID list "
            "the embedder forward pass is skipped entirely. The "
            "screening manager points this at "
            "``<output_root>/embeddings_cache/`` so the cache "
            "survives the gather job and accelerates retries of "
            "failed batches."
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
    """Atomic CSV write: write to ``<name>.csv.tmp`` then ``rename`` to
    ``<name>.csv``. POSIX rename is atomic on the same filesystem, so
    a SIGKILLed worker can never leave a partial CSV behind — a rerun
    sees either the previous file or no file, never a half-written
    one (which the skip-done logic would otherwise mistake for done)."""
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"{shard_name}.csv"
    tmp = out.with_suffix(".csv.tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def _load_embeddings_cache(
    cache_dir: Path,
    shard_name: str,
    expected_ids: list[str],
    model_name: str,
) -> np.ndarray | None:
    """Return cached embeddings for this shard, or ``None`` if the
    cache is missing, stale (different model / IDs), or malformed.

    Cache layout::

        <cache_dir>/<shard_name>.npy          row-aligned embedding matrix
        <cache_dir>/<shard_name>.ids.txt      first line = model name;
                                              remaining lines = IDs in
                                              row order

    Cache invalidates when:
      * the .npy or .ids.txt is missing,
      * the recorded model name differs from the current ``plm_model``
        (different ankh / esm version → different embeddings),
      * the recorded IDs differ from the current FASTA shard rows
        (user re-sliced the FASTA, or shard_size changed),
      * the embedding matrix's first dim doesn't match the ID list.
    """
    npy = cache_dir / f"{shard_name}.npy"
    ids_path = cache_dir / f"{shard_name}.ids.txt"
    if not (npy.is_file() and ids_path.is_file()):
        return None
    try:
        lines = ids_path.read_text().strip().split("\n")
    except OSError:
        return None
    if not lines:
        return None
    cached_model, cached_ids = lines[0], lines[1:]
    if cached_model != model_name:
        logger.warning(
            "embeddings cache: %s was computed with %r, now %r — recomputing",
            shard_name, cached_model, model_name,
        )
        return None
    if cached_ids != expected_ids:
        logger.warning(
            "embeddings cache: %s ID list differs from current shard — recomputing",
            shard_name,
        )
        return None
    try:
        embeddings = np.load(npy)
    except (OSError, ValueError):
        return None
    if len(embeddings) != len(expected_ids):
        logger.warning(
            "embeddings cache: %s shape mismatch (rows %d vs IDs %d) — recomputing",
            shard_name, len(embeddings), len(expected_ids),
        )
        return None
    logger.info(
        "embeddings cache HIT for %s — loaded %s",
        shard_name, tuple(embeddings.shape),
    )
    return embeddings


def _save_embeddings_cache(
    cache_dir: Path,
    shard_name: str,
    ids: list[str],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """Atomic save of ``embeddings`` and its ID list. Both files use
    ``.tmp`` + ``rename`` so a SIGKILL in the middle can never leave a
    corrupt half-file that a rerun would silently load."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{shard_name}.npy"
    ids_path = cache_dir / f"{shard_name}.ids.txt"
    npy_tmp = cache_dir / f"{shard_name}.npy.tmp"
    ids_tmp = cache_dir / f"{shard_name}.ids.txt.tmp"
    # ``np.save`` would auto-append ``.npy`` if the path doesn't end in
    # ``.npy``, which would land the data at ``<name>.npy.tmp.npy`` and
    # the subsequent rename would fail. Write through an explicit
    # file handle so we control the exact destination.
    with npy_tmp.open("wb") as fh:
        np.save(fh, embeddings, allow_pickle=False)
    ids_tmp.write_text(model_name + "\n" + "\n".join(ids))
    npy_tmp.replace(npy)
    ids_tmp.replace(ids_path)
    logger.info(
        "embeddings cache SAVE for %s — %s",
        shard_name, tuple(embeddings.shape),
    )


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
    precomputed_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Run the PLM-only classifier on every sequence in the shard,
    skipping the embedder forward pass since we already have the
    embeddings (computed once at the top of ``main``)."""
    return predict_sequences_only(
        sequences_df,
        plm_only_bundle_path=args.plm_only_bundle,
        calibration_csv_path=args.calibration_csv,
        plm_model=args.plm_model,
        precomputed_embeddings=precomputed_embeddings,
        plm_batch_size=args.plm_batch_size,
    )


def _run_plm_domains(
    sequences_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    structures_dir: Path,
    precomputed_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the structure-aware classifier on the sequences whose PDBs
    are present in ``structures_dir``. Embeddings for the SUBSET (passed
    by the caller, sliced to match ``sequences_df``) skip the embedder."""
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
        precomputed_embeddings=precomputed_embeddings,
        # workdir stays None so the inner tempfile.X calls inherit the
        # managed_workdir-swapped tempfile.tempdir.
        workdir=None,
        keep_intermediate=False,
    )


def main(args: argparse.Namespace) -> None:
    from enzymeexplorer.src.utils.managed_workdir import managed_workdir
    from enzymeexplorer.src.utils.signal_handling import graceful_shutdown

    with graceful_shutdown(name=f"tps_predict_fasta[{args.shard_name}]"), \
            managed_workdir(args.workdir, keep=args.keep_intermediate) as wd:
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

        # Drop oversized sequences before any embedding / prediction so
        # they can't OOM the GPU or the domain-detection pool. Sharding
        # is upstream of this filter — the per-shard ID space is
        # unchanged, the oversized rows are just absent from the
        # written CSVs.
        if not sequences_df.empty and args.max_seq_len > 0:
            seq_lens = sequences_df["sequence"].str.len()
            keep_mask = seq_lens <= args.max_seq_len
            n_dropped = int((~keep_mask).sum())
            if n_dropped:
                logger.info(
                    "shard %s: dropping %d/%d sequences longer than %d residues "
                    "(max observed: %d)",
                    args.shard_name, n_dropped, len(sequences_df),
                    args.max_seq_len, int(seq_lens.max()),
                )
                sequences_df = sequences_df[keep_mask].reset_index(drop=True)

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

        # ---------------- Skip-done detection ----------------------
        # Use the persistent output dir as a cache: if a previous run
        # already wrote a particular shard CSV, skip the matching
        # classifier path on this run. Combined with the embeddings
        # cache below, this lets a partial-failure rerun resume only
        # the parts that actually failed.
        plm_csv          = args.output_dir / SUBDIR_PLM / f"{args.shard_name}.csv"
        plm_dom_csv      = args.output_dir / SUBDIR_PLM_DOMAINS / f"{args.shard_name}.csv"
        plm_dom_fb_csv   = args.output_dir / SUBDIR_PLM_DOMAINS_FALLBACK / f"{args.shard_name}.csv"
        no_structure_csv = args.output_dir / SUBDIR_NO_STRUCTURE / f"{args.shard_name}.csv"

        need_plm = (
            args.classifier in ("plm", "both") and not plm_csv.is_file()
        )
        # plm_domains is "done" only when *all three* of its companion
        # CSVs exist (predictions, plm-only fallback, no_structure list).
        need_pdm = (
            args.classifier in ("plm_domains", "both")
            and not (
                plm_dom_csv.is_file()
                and plm_dom_fb_csv.is_file()
                and no_structure_csv.is_file()
            )
        )

        if not (need_plm or need_pdm):
            logger.info(
                "shard %s: all requested classifier outputs already exist; "
                "nothing to do", args.shard_name,
            )
            return

        # ---------------- Embeddings (cache or compute) ------------
        # Compute embeddings ONCE for the full shard; both classifiers
        # consume slices of this matrix. The matrix is persisted under
        # ``--embeddings-cache-dir`` so a rerun of this shard skips
        # the ankh_large forward pass entirely.
        ids = sequences_df["id"].tolist()
        embeddings: np.ndarray | None = None

        if args.embeddings_cache_dir is not None:
            embeddings = _load_embeddings_cache(
                args.embeddings_cache_dir, args.shard_name,
                expected_ids=ids, model_name=args.plm_model,
            )

        if embeddings is None:
            logger.info(
                "shard %s: computing PLM embeddings (%s) for %d sequences",
                args.shard_name, args.plm_model, len(ids),
            )
            embedder = load_plm_embedder(args.plm_model)
            embeddings = embedder.embed(
                sequences_df["sequence"].tolist(),
                batch_size=args.plm_batch_size,
                progress_desc=f"PLM embeddings [{args.shard_name}]",
            )
            if args.embeddings_cache_dir is not None:
                _save_embeddings_cache(
                    args.embeddings_cache_dir, args.shard_name,
                    ids=ids, embeddings=embeddings, model_name=args.plm_model,
                )

        # ---------------- PLM (sequences only) ----------------------
        if need_plm:
            try:
                table = _run_plm(
                    sequences_df, args=args, precomputed_embeddings=embeddings,
                )
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
        elif args.classifier in ("plm", "both"):
            logger.info(
                "PLM: %s exists, skipping", plm_csv,
            )

        # ---------------- PlmDomains (structure-aware) -------------
        if need_pdm:
            try:
                # Resolve structures dir (provided or downloaded).
                # Inline downloads stage under the managed workdir so
                # they vanish with the rest of the scratch on exit.
                # ``wd is None`` would mean the operator passed
                # ``--workdir=""`` which we don't support here; fall
                # back to a managed-tmp tempdir so we never spill PDBs
                # into the cwd.
                import tempfile
                inline_dl_dir = (
                    Path(wd) / "af_db_pdbs"
                    if wd is not None
                    else Path(tempfile.mkdtemp(prefix="af_db_pdbs_"))
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
                    # Slice the precomputed embedding matrix down to
                    # only the rows whose proteins have a usable PDB.
                    present_set = set(ids_present)
                    subset_idx = [
                        i for i, uid in enumerate(ids) if uid in present_set
                    ]
                    subset = sequences_df.iloc[subset_idx].reset_index(drop=True)
                    subset_embeddings = embeddings[subset_idx]
                    plm_domains_table, fallback_table = _run_plm_domains(
                        subset,
                        args=args,
                        structures_dir=eff_dir,
                        precomputed_embeddings=subset_embeddings,
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
        elif args.classifier in ("plm_domains", "both"):
            logger.info(
                "plm_domains: %s and companions exist, skipping",
                plm_dom_csv,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
