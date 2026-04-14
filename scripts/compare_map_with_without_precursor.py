#!/usr/bin/env python3
"""Compare mAP computed WITH vs WITHOUT 'precursor substr' class."""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

OUTPUT_ROOT = Path("outputs")

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

SKIP_CLASSES = {"isTPS"}
SKIP_CLASSES_NO_PRECURSOR = {"isTPS", "precursor substr"}


def _load_fold_results(model_type: str, version: str, n_folds: int = 5):
    base = OUTPUT_ROOT / model_type / version
    candidate = base / "all_folds" / "all_classes"
    if candidate.exists():
        ts_dirs = sorted(candidate.iterdir(), reverse=True)
        for ts_dir in ts_dirs:
            if (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break
    results = []
    for i in range(n_folds):
        pkl = base / f"fold_{i}_results.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results.append((i, data))
    return results


def compute_map(fold_results, skip_classes):
    fold_maps = []
    for _fold_i, (val_proba, class_names, test_df) in fold_results:
        class_list = list(class_names) if not isinstance(class_names, list) else class_names
        per_class_aps = []
        per_class_detail = {}
        for ci, cname in enumerate(class_list):
            if cname in skip_classes:
                continue
            col = "SMILES_substrate_canonical_no_stereo"
            if col not in test_df.columns:
                break
            labels = test_df[col].values
            y_true = np.array(
                [
                    1 if (isinstance(s, set) and cname in s) or s == cname
                    else 0
                    for s in labels
                ]
            )
            n_pos = int(y_true.sum())
            if n_pos < 1 or n_pos == len(y_true):
                continue
            ap = average_precision_score(y_true, val_proba[:, ci])
            per_class_aps.append(ap)
            per_class_detail[cname] = (ap, n_pos)
        if per_class_aps:
            fold_maps.append(float(np.mean(per_class_aps)))
    if not fold_maps:
        return None
    return float(np.mean(fold_maps))


def main():
    print(f"{'Model':<22} {'Track':<18} {'mAP (with precur)':<20} {'mAP (no precur)':<20} {'delta':<10}")
    print("-" * 92)

    for track_label, models in _TRACK_TO_MODELS.items():
        for model_type, version in models.items():
            fold_results = _load_fold_results(model_type, version)
            if not fold_results:
                continue
            map_with = compute_map(fold_results, SKIP_CLASSES)
            map_without = compute_map(fold_results, SKIP_CLASSES_NO_PRECURSOR)
            if map_with is None and map_without is None:
                continue
            w = f"{map_with:.4f}" if map_with else "—"
            wo = f"{map_without:.4f}" if map_without else "—"
            d = ""
            if map_with and map_without:
                d = f"{map_without - map_with:+.4f}"
            print(f"{model_type:<22} {track_label:<18} {w:<20} {wo:<20} {d:<10}")

    # Also show per-class AP for Blastp Track A to explain the drop
    print("\n\n=== Per-class AP detail: Blastp, Track A ===")
    fold_results = _load_fold_results("Blastp", "with_minor_reactions_phylo_folds")
    if fold_results:
        for fold_i, (val_proba, class_names, test_df) in fold_results:
            class_list = list(class_names) if not isinstance(class_names, list) else class_names
            print(f"\n  Fold {fold_i}:")
            col = "SMILES_substrate_canonical_no_stereo"
            for ci, cname in enumerate(class_list):
                if cname == "isTPS":
                    continue
                labels = test_df[col].values
                y_true = np.array(
                    [
                        1 if (isinstance(s, set) and cname in s) or s == cname
                        else 0
                        for s in labels
                    ]
                )
                n_pos = int(y_true.sum())
                if n_pos < 1 or n_pos == len(y_true):
                    print(f"    {cname[:50]:<52} n_pos={n_pos:>5}  SKIPPED")
                    continue
                ap = average_precision_score(y_true, val_proba[:, ci])
                print(f"    {cname[:50]:<52} n_pos={n_pos:>5}  AP={ap:.4f}")


if __name__ == "__main__":
    main()
