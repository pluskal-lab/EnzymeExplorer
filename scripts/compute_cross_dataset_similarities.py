#!/usr/bin/env python3
"""Compute cross-dataset per-fold similarities using MMseqs2.

For cross-dataset tracks (D, E), training and evaluation data come from
different CSVs.  This script computes the sequence identity of each test
protein (from ``--eval-csv``) to its nearest neighbour in the training set
(from ``--train-csv``), per fold.

Output format matches ``compute_fold_similarities.py``::

    {fold_index: {seq_id: {"pident": float, "qcov": float,
                           "evalue": float, "has_hit": bool,
                           "qcov_pass": bool}}}
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

MMSEQS_ALIGNMENT_MODE = 3
MMSEQS_FORMAT_OUTPUT = "query,target,pident,qcov,evalue"
MMSEQS_EVALUE = "inf"
MMSEQS_MAX_SEQS = 300
MIN_QCOV_THRESHOLD = 0.5


def write_fasta(ids, seqs, output_path):
    fasta_str = get_fasta_seqs(seqs, ids)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fasta_str)


def run_mmseqs_search(query_fasta, target_fasta, output_file, tmp_dir, max_seqs):
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


def select_best_hits(m8_file, min_qcov):
    candidates = defaultdict(list)
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

    best_hits = {}
    for query_id, hits in candidates.items():
        all_sorted = sorted(hits, key=lambda x: (-x[0], x[2]))
        eligible = [(p, q, e) for p, q, e in all_sorted if q >= min_qcov]
        if eligible:
            p, q, e = eligible[0]
            best_hits[query_id] = {
                "pident": p, "qcov": q, "evalue": e,
                "has_hit": True, "qcov_pass": True,
            }
        else:
            p, q, e = all_sorted[0]
            best_hits[query_id] = {
                "pident": p, "qcov": q, "evalue": e,
                "has_hit": True, "qcov_pass": False,
            }
    return best_hits


def _normalize_fold(val):
    s = str(val)
    if s.startswith("fold_"):
        return int(s.replace("fold_", ""))
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--train-fold-col", required=True)
    p.add_argument("--train-id-col", default="Uniprot ID")
    p.add_argument("--train-seq-col", default="Amino acid sequence")
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--eval-fold-col", required=True)
    p.add_argument("--eval-id-col", default="ID")
    p.add_argument("--eval-seq-col", default="Aminoacid_sequence")
    p.add_argument("--output", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    train_df["_fold_int"] = train_df[args.train_fold_col].apply(_normalize_fold)
    train_df = train_df.dropna(subset=["_fold_int"])
    train_df["_fold_int"] = train_df["_fold_int"].astype(int)
    train_unique = train_df.drop_duplicates(subset=[args.train_id_col])

    eval_df = pd.read_csv(args.eval_csv)
    eval_df["_fold_int"] = eval_df[args.eval_fold_col].apply(_normalize_fold)
    eval_df = eval_df.dropna(subset=["_fold_int"])
    eval_df["_fold_int"] = eval_df["_fold_int"].astype(int)
    eval_unique = eval_df.drop_duplicates(subset=[args.eval_id_col])

    logger.info(
        "Train: %d unique seqs, Eval: %d unique seqs",
        len(train_unique), len(eval_unique),
    )

    fold_2_similarities = {}

    with tempfile.TemporaryDirectory() as tmp_base:
        for fold_k in tqdm(range(args.n_folds), desc="Folds"):
            train_fold = train_unique[train_unique["_fold_int"] != fold_k]
            test_fold = eval_unique[eval_unique["_fold_int"] == fold_k]

            logger.info(
                "Fold %d: %d train, %d test", fold_k, len(train_fold), len(test_fold)
            )

            if len(train_fold) == 0 or len(test_fold) == 0:
                logger.warning("Skipping fold %d: empty split", fold_k)
                continue

            test_fasta = os.path.join(tmp_base, f"test_{fold_k}.fasta")
            train_fasta = os.path.join(tmp_base, f"train_{fold_k}.fasta")
            output_m8 = os.path.join(tmp_base, f"results_{fold_k}.m8")
            mmseqs_tmp = os.path.join(tmp_base, f"mmseqs_tmp_{fold_k}")
            os.makedirs(mmseqs_tmp, exist_ok=True)

            write_fasta(
                test_fold[args.eval_id_col].tolist(),
                test_fold[args.eval_seq_col].tolist(),
                test_fasta,
            )
            write_fasta(
                train_fold[args.train_id_col].tolist(),
                train_fold[args.train_seq_col].tolist(),
                train_fasta,
            )

            try:
                run_mmseqs_search(
                    test_fasta, train_fasta, output_m8, mmseqs_tmp, MMSEQS_MAX_SEQS
                )
            except subprocess.CalledProcessError as e:
                logger.error("MMseqs2 failed for fold %d: %s", fold_k, e)
                continue

            best_hits = select_best_hits(output_m8, MIN_QCOV_THRESHOLD)

            no_result_ids = set(test_fold[args.eval_id_col]) - set(best_hits.keys())
            for seq_id in no_result_ids:
                best_hits[seq_id] = {
                    "pident": 0.0, "qcov": 0.0, "evalue": float("inf"),
                    "has_hit": False, "qcov_pass": False,
                }

            fold_2_similarities[fold_k] = best_hits
            logger.info(
                "Fold %d: %d hits, %d no-hit",
                fold_k, len(best_hits) - len(no_result_ids), len(no_result_ids),
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(fold_2_similarities, f)
    logger.info("Saved %s", out_path)

    meta = {
        "mmseqs_alignment_mode": MMSEQS_ALIGNMENT_MODE,
        "mmseqs_format_output": MMSEQS_FORMAT_OUTPUT,
        "mmseqs_evalue": MMSEQS_EVALUE,
        "mmseqs_max_seqs": MMSEQS_MAX_SEQS,
        "min_qcov_threshold": MIN_QCOV_THRESHOLD,
        "train_csv": args.train_csv,
        "train_fold_col": args.train_fold_col,
        "eval_csv": args.eval_csv,
        "eval_fold_col": args.eval_fold_col,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata: %s", meta_path)


if __name__ == "__main__":
    main()
