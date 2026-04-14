#!/usr/bin/env python3
"""Sweep substrate subsets to find the set where Blastp Track A mAP ≈ 0.69–0.71.

For each subset size (top-K substrates by unique TPS count), computes mAP
from fold result pkl files for all models on all tracks.

Usage::

    python scripts/sweep_substrate_subsets.py
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUTPUT_ROOT = Path("outputs")

SUBSTRATES_BY_COUNT = [
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    "CC1(C)CCCC2(C)C1CCC(C)(O)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
]

SHORT_NAMES = {
    SUBSTRATES_BY_COUNT[0]: "FPP (sesqui)",
    SUBSTRATES_BY_COUNT[1]: "GPP (mono)",
    SUBSTRATES_BY_COUNT[2]: "GGPP (di)",
    SUBSTRATES_BY_COUNT[3]: "SqOx (triterp)",
    SUBSTRATES_BY_COUNT[4]: "CPP (di-II)",
    SUBSTRATES_BY_COUNT[5]: "GFPP (sester)",
    SUBSTRATES_BY_COUNT[6]: "2×FPP (sesq dimer)",
    SUBSTRATES_BY_COUNT[7]: "2×GGPP (di dimer)",
    SUBSTRATES_BY_COUNT[8]: "LPP (di labdane)",
}

_TRACK_TO_MODELS: dict[str, dict[str, str]] = {
    "A (phylo)": {
        "Blastp": "with_minor_reactions_phylo_folds",
        "CLEANEcDetection": "with_minor_reactions_phylo_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
        "PlmDomainsRandomForest": (
            "tps_esm-1v-subseq_with_minor_reactions"
            "_global_tuning_domains_subset_phylo_folds"
        ),
    },
    "C (synced)": {
        "Blastp": "synced_folds",
        "CLEANEcDetection": "synced_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_synced_folds",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_synced_folds",
    },
    "B (new)": {
        "Blastp": "new_dataset",
        "CLEANEcDetection": "new_dataset",
        "PlmRandomForest": "tps_esm-1v-subseq_new_dataset",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_new_dataset",
    },
    "D (cross)": {
        "Blastp": "cross_synced_to_new",
        "CLEANEcDetection": "cross_synced_to_new",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
    },
    "E (new TPS+old neg)": {
        "Blastp": "cross_new_tps_old_neg",
        "CLEANEcDetection": "cross_new_tps_old_neg",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
    },
}

MODEL_DISPLAY = {
    "PlmRandomForest": "PlmRF",
    "PlmDomainsRandomForest": "PlmDomainsRF",
    "Blastp": "Blastp",
    "CLEANEcDetection": "CLEAN (in-sample)",
}


def _load_fold_results(
    model_type: str, version: str, n_folds: int = 5
) -> list[tuple]:
    base = OUTPUT_ROOT / model_type / version / "all_folds" / "all_classes"
    if not base.exists():
        return []
    ts_dirs = sorted(base.iterdir(), reverse=True)
    for ts_dir in ts_dirs:
        if (ts_dir / "fold_0_results.pkl").exists():
            results = []
            for i in range(n_folds):
                pkl = ts_dir / f"fold_{i}_results.pkl"
                if not pkl.exists():
                    continue
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                results.append((i, data))
            return results
    return []


def compute_map_for_subset(
    fold_results: list[tuple],
    allowed_classes: set[str],
) -> float | None:
    """Compute mAP using only the specified substrate classes."""
    fold_maps: list[float] = []
    for _fold_i, (val_proba, class_names, test_df) in fold_results:
        class_list = (
            list(class_names) if not isinstance(class_names, list) else class_names
        )
        col = "SMILES_substrate_canonical_no_stereo"
        if col not in test_df.columns:
            continue
        labels = test_df[col].values

        per_class_aps: list[float] = []
        for ci, cname in enumerate(class_list):
            if cname == "isTPS" or cname not in allowed_classes:
                continue
            y_true = np.array(
                [
                    1
                    if (isinstance(s, set) and cname in s) or s == cname
                    else 0
                    for s in labels
                ]
            )
            if y_true.sum() < 1 or y_true.sum() == len(y_true):
                continue
            ap = average_precision_score(y_true, val_proba[:, ci])
            per_class_aps.append(ap)

        if per_class_aps:
            fold_maps.append(float(np.mean(per_class_aps)))

    if not fold_maps:
        return None
    return float(np.mean(fold_maps))


def main() -> None:
    print("=" * 80)
    print("Substrate subset sweep: mAP for top-K substrates by unique TPS count")
    print("=" * 80)
    print()

    for k in range(4, len(SUBSTRATES_BY_COUNT) + 1):
        subset = set(SUBSTRATES_BY_COUNT[:k])
        names = [SHORT_NAMES.get(s, s[:30]) for s in SUBSTRATES_BY_COUNT[:k]]
        print(f"\n--- Top-{k} substrates: {', '.join(names)} ---")
        print(f"{'Model':<22} ", end="")
        for track in ["A (phylo)", "C (synced)", "D (cross)", "E (new TPS+old neg)", "B (new)"]:
            print(f" {track:<12}", end="")
        print()

        for model_type in ["Blastp", "PlmRandomForest", "PlmDomainsRandomForest", "CLEANEcDetection"]:
            display = MODEL_DISPLAY.get(model_type, model_type)
            print(f"{display:<22} ", end="")
            for track in ["A (phylo)", "C (synced)", "D (cross)", "E (new TPS+old neg)", "B (new)"]:
                models = _TRACK_TO_MODELS.get(track, {})
                version = models.get(model_type)
                if not version:
                    print(f" {'—':>12}", end="")
                    continue
                fold_results = _load_fold_results(model_type, version)
                if not fold_results:
                    print(f" {'—':>12}", end="")
                    continue
                mAP = compute_map_for_subset(fold_results, subset)
                if mAP is not None:
                    print(f" {mAP:>12.3f}", end="")
                else:
                    print(f" {'—':>12}", end="")
            print()

    print("\n\nDone.")


if __name__ == "__main__":
    main()
