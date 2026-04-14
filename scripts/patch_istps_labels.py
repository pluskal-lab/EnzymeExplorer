#!/usr/bin/env python3
"""Patch fold_results pickles: fix isTPS labels for substrate-bearing negatives.

Substrate-bearing negatives (Type=Unknown proteins with real terpenoid
substrates) were incorrectly labelled isTPS=True.  This script corrects
the test_df label sets inside every fold-result pickle without
retraining or re-predicting.

Usage::

    python scripts/patch_istps_labels.py          # dry-run (default)
    python scripts/patch_istps_labels.py --apply   # apply changes in-place
"""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")

_NON_TPS_LABELS = frozenset({"Unknown", "precursor substr"})

_TRACK_TO_MODELS: dict[str, dict[str, str]] = {
    "A (phylo)": {
        "Blastp": "with_minor_reactions_phylo_folds",
        "CLEANEcDetection": "with_minor_reactions_phylo_folds",
        "Foldseek": "with_minor_reactions_phylo_folds",
        "HMM": "with_minor_reactions_phylo_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_with_minor_reactions_global_tuning_domains_subset_phylo_folds",
    },
    "C (synced)": {
        "Blastp": "synced_folds",
        "CLEANEcDetection": "synced_folds",
        "Foldseek": "synced_folds",
        "HMM": "synced_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_synced_folds",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_synced_folds",
    },
    "B (new)": {
        "Blastp": "new_dataset",
        "CLEANEcDetection": "new_dataset",
        "CLEAN": "new_dataset",
        "Foldseek": "new_dataset",
        "HMM": "new_dataset",
        "PlmRandomForest": "tps_esm-1v-subseq_new_dataset",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_new_dataset",
    },
    "D (cross)": {
        "Blastp": "cross_synced_to_new",
        "CLEANEcDetection": "cross_synced_to_new",
        "Foldseek": "cross_synced_to_new",
        "HMM": "cross_synced_to_new",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
    },
    "E (new TPS+old neg)": {
        "Blastp": "cross_new_tps_old_neg",
        "CLEANEcDetection": "cross_new_tps_old_neg",
        "Foldseek": "cross_new_tps_old_neg",
        "HMM": "cross_new_tps_old_neg",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
    },
}

_TRACK_TO_EVAL_CSV: dict[str, str] = {
    "A (phylo)": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "C (synced)": "data/TPS-Nov19_2023_with_synced_folds.csv",
    "B (new)": "data/EnzymeExplorer_Dataset.csv",
    "D (cross)": "data/EnzymeExplorer_Dataset.csv",
    "E (new TPS+old neg)": "data/EnzymeExplorer_Dataset.csv",
}

_TRACK_TO_TYPE_COL: dict[str, str] = {
    "A (phylo)": "Type (mono, sesq, di, \u2026)",
    "C (synced)": "Type (mono, sesq, di, \u2026)",
    "B (new)": "Type",
    "D (cross)": "Type",
    "E (new TPS+old neg)": "Type",
}

N_FOLDS = 5
TARGET_COL = "SMILES_substrate_canonical_no_stereo"


def _load_fold_pkl_path(model_type: str, version: str, fold_i: int) -> Path | None:
    base = OUTPUT_ROOT / model_type / version
    candidate = base / "all_folds" / "all_classes"
    if candidate.exists():
        ts_dirs = sorted(candidate.iterdir(), reverse=True)
        for ts_dir in ts_dirs:
            if (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break
    pkl = base / f"fold_{fold_i}_results.pkl"
    return pkl if pkl.exists() else None


def _get_unknown_ids(csv_path: str, type_col: str) -> frozenset[str]:
    """Return protein IDs with Type=Unknown from a dataset CSV."""
    df = pd.read_csv(csv_path)
    if type_col not in df.columns:
        return frozenset()
    id_col = "ID" if "ID" in df.columns else "Uniprot ID"
    return frozenset(df.loc[df[type_col] == "Unknown", id_col].unique())


def _fix_labels(label_set: set[str]) -> set[str]:
    """Replace the label set for a substrate-bearing negative with {"Unknown"}."""
    return {"Unknown"}


def patch_all(*, apply: bool = False) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)

    unknown_ids_cache: dict[str, frozenset[str]] = {}

    for track_label, models in _TRACK_TO_MODELS.items():
        csv_path = _TRACK_TO_EVAL_CSV.get(track_label, "")
        type_col = _TRACK_TO_TYPE_COL.get(track_label, "Type")
        if not csv_path or not Path(csv_path).exists():
            logger.warning("Eval CSV not found for track %s: %s", track_label, csv_path)
            continue

        cache_key = f"{csv_path}|{type_col}"
        if cache_key not in unknown_ids_cache:
            unknown_ids_cache[cache_key] = _get_unknown_ids(csv_path, type_col)
        unknown_ids = unknown_ids_cache[cache_key]

        if not unknown_ids:
            logger.info("Track %s: no Unknown-type proteins found, skipping", track_label)
            continue

        for model_type, version in models.items():
            for fold_i in range(N_FOLDS):
                pkl_path = _load_fold_pkl_path(model_type, version, fold_i)
                if pkl_path is None:
                    continue

                with open(pkl_path, "rb") as f:
                    val_proba, class_names, test_df = pickle.load(f)

                id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
                n_fixed = 0
                for idx in test_df.index:
                    pid = test_df.at[idx, id_col]
                    if pid in unknown_ids:
                        old_labels = test_df.at[idx, TARGET_COL]
                        if isinstance(old_labels, set) and "isTPS" in old_labels:
                            test_df.at[idx, TARGET_COL] = _fix_labels(old_labels)
                            n_fixed += 1

                if n_fixed > 0:
                    key = f"{track_label}/{model_type}/{version}/fold_{fold_i}"
                    stats[key] = n_fixed
                    logger.info(
                        "%s fold %d: %d substrate-bearing negatives fixed%s",
                        f"{model_type}/{version}",
                        fold_i,
                        n_fixed,
                        "" if apply else " (dry-run)",
                    )
                    if apply:
                        backup = pkl_path.with_suffix(".pkl.bak")
                        if not backup.exists():
                            shutil.copy2(pkl_path, backup)
                        with open(pkl_path, "wb") as f:
                            pickle.dump((val_proba, class_names, test_df), f)

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes in-place (default: dry-run)",
    )
    args = parser.parse_args()

    stats = patch_all(apply=args.apply)

    total = sum(stats.values())
    if total == 0:
        logger.info("No substrate-bearing negatives with isTPS found in any pickle.")
    else:
        logger.info(
            "Total fixes: %d across %d pickle files%s",
            total,
            len(stats),
            " (APPLIED)" if args.apply else " (dry-run, use --apply to write)",
        )


if __name__ == "__main__":
    main()
