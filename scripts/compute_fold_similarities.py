#!/usr/bin/env python3
"""Compute per-fold train-test sequence similarity using MMseqs2.

Produces a pickle artifact consumed by evaluation.py for
similarity-binned performance analysis.  Parallel to the existing
``blast_identities_per_fold.pkl`` artifact but richer: stores pident,
qcov, evalue and a has_hit flag per test sequence per fold.

Output schema
-------------
``{fold_index: {seq_id: {"pident": float, "qcov": float,
                          "evalue": float, "has_hit": bool}}}``

Pinned MMseqs2 flags (documented here so bins are interpretable across
reruns):

  --alignment-mode 3   Needleman-Wunsch literal aligned-residue identity
  --format-output      query,target,pident,qcov,evalue
  -e inf               no E-value pre-filter
  --max-seqs 300       retrieve top-N, pick best post-hoc

Best-hit selection rule (frozen):

  Highest pident among hits with qcov >= MIN_QCOV_THRESHOLD.
  Ties broken by lowest evalue.
  If all hits below qcov threshold -> has_hit = False.
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from enzymeexplorer.src.utils.msa import get_fasta_seqs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── Pinned MMseqs2 settings ─────────────────────────────────────────
MMSEQS_ALIGNMENT_MODE = 3
MMSEQS_FORMAT_OUTPUT = "query,target,pident,qcov,evalue"
MMSEQS_EVALUE = "inf"
MMSEQS_MAX_SEQS = 300

MIN_QCOV_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-fold train-test MMseqs2 similarity artifact"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the dataset CSV with fold assignments",
    )
    parser.add_argument(
        "--fold-col",
        type=str,
        required=True,
        help="Name of the fold column (e.g. stratified_phylogeny_based_split_with_minor_products)",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="Uniprot ID",
        help="Column containing sequence identifiers",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="Amino acid sequence",
        help="Column containing amino acid sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output pickle file",
    )
    parser.add_argument(
        "--min-qcov",
        type=float,
        default=MIN_QCOV_THRESHOLD,
        help="Minimum query coverage for a hit to be considered (default: 0.5)",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=MMSEQS_MAX_SEQS,
        help="Number of top hits to retrieve from MMseqs (default: 300)",
    )
    return parser.parse_args()


def write_fasta(ids: list[str], seqs: list[str], output_path: str) -> None:
    fasta_str = get_fasta_seqs(seqs, ids)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fasta_str)


def run_mmseqs_search(
    query_fasta: str,
    target_fasta: str,
    output_file: str,
    tmp_dir: str,
    max_seqs: int = MMSEQS_MAX_SEQS,
) -> None:
    cmd = [
        "mmseqs",
        "easy-search",
        query_fasta,
        target_fasta,
        output_file,
        tmp_dir,
        "--alignment-mode",
        str(MMSEQS_ALIGNMENT_MODE),
        "--format-output",
        MMSEQS_FORMAT_OUTPUT,
        "-e",
        MMSEQS_EVALUE,
        "--max-seqs",
        str(max_seqs),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def select_best_hits(
    m8_file: str,
    min_qcov: float,
) -> dict[str, dict]:
    """Parse .m8 output and select best hit per query.

    Best hit = highest pident among hits with qcov >= min_qcov.
    Ties broken by lowest evalue.
    """
    candidates: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    if not os.path.exists(m8_file) or os.path.getsize(m8_file) == 0:
        return {}

    with open(m8_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            query_id = parts[0]
            pident = float(parts[2])
            qcov = float(parts[3]) / 100.0 if float(parts[3]) > 1.0 else float(parts[3])
            evalue = float(parts[4])
            candidates[query_id].append((pident, qcov, evalue))

    best_hits: dict[str, dict] = {}
    for query_id, hits in candidates.items():
        eligible = [(p, q, e) for p, q, e in hits if q >= min_qcov]
        if eligible:
            eligible.sort(key=lambda x: (-x[0], x[2]))
            p, q, e = eligible[0]
            best_hits[query_id] = {
                "pident": p,
                "qcov": q,
                "evalue": e,
                "has_hit": True,
            }
        else:
            best_hits[query_id] = {
                "pident": 0.0,
                "qcov": 0.0,
                "evalue": float("inf"),
                "has_hit": False,
            }
    return best_hits


def get_fold_names(df: pd.DataFrame, fold_col: str) -> list[str]:
    """Return fold values present in *df*, normalised to ``fold_N`` format."""
    raw = df[fold_col].dropna().unique()
    prefixed = sorted(str(f) for f in raw if str(f).startswith("fold_"))
    if prefixed:
        return prefixed
    # Bare non-negative integers (new dataset)
    bare: list[str] = []
    for f in raw:
        try:
            n = int(float(str(f)))
            if n >= 0:
                bare.append(f"fold_{n}")
        except (ValueError, OverflowError):
            continue
    return sorted(bare, key=lambda x: int(x.replace("fold_", "")))


def _normalize_fold_column(df: pd.DataFrame, col: str) -> None:
    """Ensure fold values use ``fold_N`` format in-place."""
    vals = df[col].dropna().astype(str)
    if vals.empty or vals.str.startswith("fold_").any():
        return
    mask = df[col].notna()
    df.loc[mask, col] = "fold_" + df.loc[mask, col].astype(int).astype(str)


def compute_fold_similarities(
    csv_path: str,
    fold_col: str,
    id_col: str,
    seq_col: str,
    min_qcov: float,
    max_seqs: int,
) -> dict:
    df = pd.read_csv(csv_path)
    unique_df = df.drop_duplicates(subset=[id_col])[[id_col, seq_col, fold_col]].copy()
    _normalize_fold_column(unique_df, fold_col)
    logger.info("Loaded %d unique sequences from %s", len(unique_df), csv_path)

    fold_names = get_fold_names(unique_df, fold_col)
    logger.info("Found %d folds: %s", len(fold_names), fold_names)

    fold_2_similarities: dict = {}

    with tempfile.TemporaryDirectory() as tmp_base:
        for fold_name in tqdm(fold_names, desc="Processing folds"):
            fold_idx = fold_name.replace("fold_", "")
            test_df = unique_df[unique_df[fold_col] == fold_name]
            train_df = unique_df[unique_df[fold_col] != fold_name]
            train_df = train_df[train_df[fold_col].isin(fold_names)]

            logger.info(
                "Fold %s: %d test, %d train sequences",
                fold_idx,
                len(test_df),
                len(train_df),
            )

            if len(test_df) == 0 or len(train_df) == 0:
                logger.warning("Skipping fold %s: empty split", fold_idx)
                continue

            test_fasta = os.path.join(tmp_base, f"test_{fold_idx}.fasta")
            train_fasta = os.path.join(tmp_base, f"train_{fold_idx}.fasta")
            output_m8 = os.path.join(tmp_base, f"results_{fold_idx}.m8")
            mmseqs_tmp = os.path.join(tmp_base, f"mmseqs_tmp_{fold_idx}")
            os.makedirs(mmseqs_tmp, exist_ok=True)

            write_fasta(
                test_df[id_col].tolist(),
                test_df[seq_col].tolist(),
                test_fasta,
            )
            write_fasta(
                train_df[id_col].tolist(),
                train_df[seq_col].tolist(),
                train_fasta,
            )

            try:
                run_mmseqs_search(
                    test_fasta, train_fasta, output_m8, mmseqs_tmp, max_seqs
                )
            except subprocess.CalledProcessError as e:
                logger.error("MMseqs2 failed for fold %s: %s", fold_idx, e)
                continue

            best_hits = select_best_hits(output_m8, min_qcov)

            no_result_ids = set(test_df[id_col]) - set(best_hits.keys())
            for seq_id in no_result_ids:
                best_hits[seq_id] = {
                    "pident": 0.0,
                    "qcov": 0.0,
                    "evalue": float("inf"),
                    "has_hit": False,
                }

            fold_2_similarities[int(fold_idx)] = best_hits
            n_hits = sum(1 for v in best_hits.values() if v["has_hit"])
            logger.info(
                "Fold %s: %d/%d test seqs have valid hits",
                fold_idx,
                n_hits,
                len(best_hits),
            )

    return fold_2_similarities


def main() -> None:
    args = parse_args()

    fold_2_similarities = compute_fold_similarities(
        csv_path=args.csv_path,
        fold_col=args.fold_col,
        id_col=args.id_col,
        seq_col=args.seq_col,
        min_qcov=args.min_qcov,
        max_seqs=args.max_seqs,
    )

    metadata = {
        "mmseqs_alignment_mode": MMSEQS_ALIGNMENT_MODE,
        "mmseqs_format_output": MMSEQS_FORMAT_OUTPUT,
        "mmseqs_evalue": MMSEQS_EVALUE,
        "mmseqs_max_seqs": args.max_seqs,
        "min_qcov_threshold": args.min_qcov,
        "best_hit_rule": "highest pident with qcov >= threshold, tiebreak by lowest evalue",
        "csv_path": args.csv_path,
        "fold_col": args.fold_col,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(fold_2_similarities, f)

    metadata_path = output_path.with_suffix(".meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved similarity artifact to %s", output_path)
    logger.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
