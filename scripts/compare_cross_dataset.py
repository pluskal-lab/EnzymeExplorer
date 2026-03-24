#!/usr/bin/env python3
"""Compare cross-dataset evaluation results between synced-fold and new-dataset models.

Given two evaluation runs that use the same fold assignments for shared TPS
sequences (e.g. synced-fold old-data model vs new-dataset model), this script:

1. Loads fold_k_results.pkl from both evaluations.
2. Identifies shared sequences in each fold's test set (by amino acid sequence).
3. Computes per-fold and aggregate metrics on the shared-positive subset.
4. Reports a comparison table.

Usage::

    python scripts/compare_cross_dataset.py \\
        --source-model-type PlmRandomForest \\
        --source-model-version tps_esm-1v-subseq_synced_folds \\
        --target-model-type PlmRandomForest \\
        --target-model-version tps_esm-1v-subseq_new_dataset \\
        --source-csv data/TPS-Nov19_2023_with_synced_folds.csv \\
        --target-csv data/EnzymeExplorer_Dataset.csv \\
        --n-folds 5
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    roc_auc_score,
)

from enzymeexplorer.src.evaluation.metrics import summary_mccf1
from enzymeexplorer.src.utils.project_info import get_output_root

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_TARGET_COL = "SMILES_substrate_canonical_no_stereo"
_IS_TPS_CLASS = "isTPS"


def _load_fold_results(
    model_type: str,
    model_version: str,
    fold_i: int,
) -> Optional[tuple[np.ndarray, list[str], pd.DataFrame]]:
    """Load fold results pickle for a given model."""
    root = get_output_root() / model_type / model_version
    if not root.exists():
        return None

    # Find the latest run for this fold
    fold_dir = root / str(fold_i) / "all_classes"
    if not fold_dir.exists():
        fold_dir = root / "all_folds" / "all_classes"
        if not fold_dir.exists():
            return None

    try:
        latest = sorted(fold_dir.glob("*"))[-1]
    except IndexError:
        return None

    pkl_path = latest / f"fold_{fold_i}_results.pkl"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _build_seq_to_id_map(csv_path: str, id_col: str, seq_col: str) -> dict[str, str]:
    """Map amino acid sequence -> ID for matching across datasets."""
    df = pd.read_csv(csv_path, usecols=[id_col, seq_col]).drop_duplicates(
        subset=[seq_col]
    )
    return dict(zip(df[seq_col], df[id_col]))


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute AP, ROC-AUC, MCC-F1 for binary classification."""
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    result: dict[str, float] = {"n_pos": float(n_pos), "n_neg": float(n_neg)}
    if n_pos == 0 or n_neg == 0:
        result["ap"] = float("nan")
        result["auc"] = float("nan")
        result["mccf1"] = float("nan")
        return result
    result["ap"] = average_precision_score(y_true, y_pred)
    result["auc"] = roc_auc_score(y_true, y_pred)
    result["mccf1"] = summary_mccf1(pd.Series(y_true), y_pred)["mccf1_metric"]
    return result


def compare_models(
    source_model_type: str,
    source_model_version: str,
    target_model_type: str,
    target_model_version: str,
    source_csv: str,
    target_csv: str,
    source_id_col: str = "Uniprot ID",
    source_seq_col: str = "Amino acid sequence",
    target_id_col: str = "ID",
    target_seq_col: str = "Aminoacid_sequence",
    n_folds: int = 5,
    class_name: str = _IS_TPS_CLASS,
) -> pd.DataFrame:
    """Compare two models on shared TPS sequences across folds.

    Returns a DataFrame with per-fold and aggregate metrics for both models.
    """
    # Build sequence-to-ID maps for matching
    source_seq_to_id = _build_seq_to_id_map(source_csv, source_id_col, source_seq_col)
    target_seq_to_id = _build_seq_to_id_map(target_csv, target_id_col, target_seq_col)

    # Find shared sequences
    shared_seqs = set(source_seq_to_id.keys()) & set(target_seq_to_id.keys())
    logger.info("Shared sequences across datasets: %d", len(shared_seqs))

    # Build ID mapping: source_id -> target_id (via shared sequence)
    source_id_to_target_id: dict[str, str] = {}
    for seq in shared_seqs:
        src_id = source_seq_to_id[seq]
        tgt_id = target_seq_to_id[seq]
        source_id_to_target_id[src_id] = tgt_id

    rows = []

    for fold_i in range(n_folds):
        src_result = _load_fold_results(source_model_type, source_model_version, fold_i)
        tgt_result = _load_fold_results(target_model_type, target_model_version, fold_i)

        if src_result is None:
            logger.warning("No source results for fold %d", fold_i)
            continue
        if tgt_result is None:
            logger.warning("No target results for fold %d", fold_i)
            continue

        src_proba, src_classes, src_test_df = src_result
        tgt_proba, tgt_classes, tgt_test_df = tgt_result

        if class_name not in src_classes or class_name not in tgt_classes:
            logger.warning("Class %s not in fold %d results", class_name, fold_i)
            continue

        src_class_idx = src_classes.index(class_name)
        tgt_class_idx = tgt_classes.index(class_name)

        # Extract predictions for the target class
        src_preds = dict(
            zip(
                src_test_df[source_id_col],
                src_proba[:, src_class_idx],
            )
        )
        tgt_preds = dict(
            zip(
                tgt_test_df[target_id_col],
                tgt_proba[:, tgt_class_idx],
            )
        )

        # Build labels
        src_labels = dict(
            zip(
                src_test_df[source_id_col],
                src_test_df[_TARGET_COL].map(lambda x: class_name in x),
            )
        )
        tgt_labels = dict(
            zip(
                tgt_test_df[target_id_col],
                tgt_test_df[_TARGET_COL].map(lambda x: class_name in x),
            )
        )

        # --- Metrics on full test sets (each model's own) ---
        src_y_true = np.array(list(src_labels.values()), dtype=float)
        src_y_pred = np.array(list(src_preds.values()))
        tgt_y_true = np.array(list(tgt_labels.values()), dtype=float)
        tgt_y_pred = np.array(list(tgt_preds.values()))

        src_full = _compute_binary_metrics(src_y_true, src_y_pred)
        tgt_full = _compute_binary_metrics(tgt_y_true, tgt_y_pred)

        # --- Metrics on shared-positive subset ---
        # Find shared IDs present in both test folds
        shared_src_ids = [
            src_id
            for src_id in src_preds
            if src_id in source_id_to_target_id
            and source_id_to_target_id[src_id] in tgt_preds
        ]
        if not shared_src_ids:
            logger.warning("No shared sequences in fold %d test sets", fold_i)
            continue

        shared_tgt_ids = [source_id_to_target_id[sid] for sid in shared_src_ids]

        shared_src_y_true = np.array([float(src_labels[sid]) for sid in shared_src_ids])
        shared_src_y_pred = np.array([src_preds[sid] for sid in shared_src_ids])
        shared_tgt_y_pred = np.array([tgt_preds[tid] for tid in shared_tgt_ids])

        shared_src_metrics = _compute_binary_metrics(
            shared_src_y_true, shared_src_y_pred
        )
        shared_tgt_metrics = _compute_binary_metrics(
            shared_src_y_true, shared_tgt_y_pred
        )

        rows.append(
            {
                "fold": fold_i,
                # Source model on its full test set
                "source_full_ap": src_full["ap"],
                "source_full_auc": src_full["auc"],
                "source_full_n_pos": src_full["n_pos"],
                "source_full_n_neg": src_full["n_neg"],
                # Target model on its full test set
                "target_full_ap": tgt_full["ap"],
                "target_full_auc": tgt_full["auc"],
                "target_full_n_pos": tgt_full["n_pos"],
                "target_full_n_neg": tgt_full["n_neg"],
                # Source model on shared positives
                "shared_source_ap": shared_src_metrics["ap"],
                "shared_source_auc": shared_src_metrics["auc"],
                "shared_source_mccf1": shared_src_metrics["mccf1"],
                # Target model on shared positives
                "shared_target_ap": shared_tgt_metrics["ap"],
                "shared_target_auc": shared_tgt_metrics["auc"],
                "shared_target_mccf1": shared_tgt_metrics["mccf1"],
                # Shared subset size
                "shared_n_pos": shared_src_metrics["n_pos"],
                "shared_n_neg": shared_src_metrics["n_neg"],
            }
        )

    if not rows:
        logger.error("No valid fold comparisons found. Have both models been trained?")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate row
    agg = {
        "fold": "mean",
        "source_full_ap": "mean",
        "source_full_auc": "mean",
        "target_full_ap": "mean",
        "target_full_auc": "mean",
        "shared_source_ap": "mean",
        "shared_source_auc": "mean",
        "shared_source_mccf1": "mean",
        "shared_target_ap": "mean",
        "shared_target_auc": "mean",
        "shared_target_mccf1": "mean",
    }
    agg_row = {col: df[col].mean() for col in agg}
    agg_row["fold"] = -1  # sentinel for "aggregate"
    agg_row["source_full_n_pos"] = df["source_full_n_pos"].sum()
    agg_row["source_full_n_neg"] = df["source_full_n_neg"].sum()
    agg_row["target_full_n_pos"] = df["target_full_n_pos"].sum()
    agg_row["target_full_n_neg"] = df["target_full_n_neg"].sum()
    agg_row["shared_n_pos"] = df["shared_n_pos"].sum()
    agg_row["shared_n_neg"] = df["shared_n_neg"].sum()

    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-type", required=True)
    parser.add_argument("--source-model-version", required=True)
    parser.add_argument("--target-model-type", required=True)
    parser.add_argument("--target-model-version", required=True)
    parser.add_argument(
        "--source-csv",
        default="data/TPS-Nov19_2023_with_synced_folds.csv",
    )
    parser.add_argument(
        "--target-csv",
        default="data/EnzymeExplorer_Dataset.csv",
    )
    parser.add_argument("--source-id-col", default="Uniprot ID")
    parser.add_argument("--source-seq-col", default="Amino acid sequence")
    parser.add_argument("--target-id-col", default="ID")
    parser.add_argument("--target-seq-col", default="Aminoacid_sequence")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--class-name", default=_IS_TPS_CLASS)
    parser.add_argument(
        "--output-csv",
        default="outputs/evaluation_results/cross_dataset_comparison.csv",
    )
    args = parser.parse_args()

    df = compare_models(
        source_model_type=args.source_model_type,
        source_model_version=args.source_model_version,
        target_model_type=args.target_model_type,
        target_model_version=args.target_model_version,
        source_csv=args.source_csv,
        target_csv=args.target_csv,
        source_id_col=args.source_id_col,
        source_seq_col=args.source_seq_col,
        target_id_col=args.target_id_col,
        target_seq_col=args.target_seq_col,
        n_folds=args.n_folds,
        class_name=args.class_name,
    )

    if df.empty:
        return

    logger.info("Cross-dataset comparison results:")
    print(df.to_string(index=False))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
