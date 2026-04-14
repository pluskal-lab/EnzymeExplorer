#!/usr/bin/env python3
"""Evaluate new model variants and compare against baselines.

Computes per-fold Average Precision (AP) for isTPS detection and
Mean Average Precision (mAP) for substrate prediction across all
tracks, for:

- CLEANBetterDetection (proportion-based P(TPS))
- PlmRandomForestHierarchical (detection × substrate)
- PlmDomainsRandomForestHierarchical (detection × substrate)

Compares against existing CLEAN, PlmRandomForest, PlmDomainsRandomForest.

Usage::

    conda run -n terpene_miner python scripts/evaluate_new_models.py
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")

SUBSTRATE_CLASSES = [
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "precursor substr",
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
]

FRIENDLY_SUBSTRATE = {
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "FPP",
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "GPP",
    "precursor substr": "precursor",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "GGPP",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C": "squalene_epox",
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "copalyl_PP",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "GFPP",
}

TRACK_MAP = {
    "with_minor_reactions_phylo_folds": "A",
    "synced_folds": "C",
    "new_dataset": "B",
    "cross_synced_to_new": "D",
    "cross_new_tps_old_neg": "E",
    "tps_esm-1v-subseq_with_minor_reactions_phylo_folds": "A",
    "tps_esm-1v-subseq_synced_folds": "C",
    "tps_esm-1v-subseq_new_dataset": "B",
    "tps_esm-1v-subseq_cross_synced_to_new": "D",
    "tps_esm-1v-subseq_cross_new_tps_old_neg": "E",
    "tps_esm-1v-subseq_with_minor_reactions_global_tuning_domains_subset_phylo_folds": "A",
    "with_minor_reactions": "A_old",
}

MODEL_PAIRS = [
    ("CLEAN", "CLEANBetterDetection"),
    ("CLEAN", "CLEANEcDetection"),
    ("PlmRandomForest", "PlmRandomForestHierarchical"),
    ("PlmDomainsRandomForest", "PlmDomainsRandomForestHierarchical"),
]

MIN_POSITIVE_FOR_EVAL = 3


def find_latest_fold_dir(model: str, version: str) -> Path | None:
    root = OUTPUT_ROOT / model / version / "all_folds" / "all_classes"
    if not root.exists():
        return None
    ts_dirs = sorted(root.iterdir())
    return ts_dirs[-1] if ts_dirs else None


def load_fold_results(
    fold_dir: Path, n_folds: int = 5
) -> list[tuple[np.ndarray, list[str], pd.DataFrame]]:
    fold_re = re.compile(r"fold_(\d+)_results\.pkl$")
    results = []
    for i in range(n_folds):
        pkl = fold_dir / f"fold_{i}_results.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as fh:
            val_proba_np, class_names, test_df = pickle.load(fh)
        if not isinstance(class_names, list):
            class_names = list(class_names)
        results.append((val_proba_np, class_names, test_df))
    return results


def compute_istps_ap(
    fold_results: list[tuple[np.ndarray, list[str], pd.DataFrame]],
    target_col: str = "SMILES_substrate_canonical_no_stereo",
) -> list[float]:
    aps = []
    for val_proba_np, class_names, test_df in fold_results:
        if "isTPS" not in class_names:
            continue
        idx = class_names.index("isTPS")
        y_pred = val_proba_np[:, idx]
        y_true = test_df[target_col].map(lambda x: "isTPS" in x).astype(int)
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos < MIN_POSITIVE_FOR_EVAL or n_neg < MIN_POSITIVE_FOR_EVAL:
            continue
        ap = average_precision_score(y_true, y_pred)
        aps.append(ap)
    return aps


def compute_substrate_map(
    fold_results: list[tuple[np.ndarray, list[str], pd.DataFrame]],
    target_col: str = "SMILES_substrate_canonical_no_stereo",
) -> tuple[list[float], dict[str, list[float]]]:
    fold_maps = []
    class_aps: dict[str, list[float]] = defaultdict(list)
    for val_proba_np, class_names, test_df in fold_results:
        per_class_aps = []
        for cls in SUBSTRATE_CLASSES:
            if cls not in class_names:
                continue
            cls_idx = class_names.index(cls)
            y_pred = val_proba_np[:, cls_idx]
            y_true = test_df[target_col].map(lambda x: cls in x).astype(int)
            if y_true.sum() < MIN_POSITIVE_FOR_EVAL:
                continue
            ap = average_precision_score(y_true, y_pred)
            per_class_aps.append(ap)
            friendly = FRIENDLY_SUBSTRATE.get(cls, cls[:20])
            class_aps[friendly].append(ap)
        if per_class_aps:
            fold_maps.append(np.mean(per_class_aps))
    return fold_maps, class_aps


def evaluate_all() -> pd.DataFrame:
    rows = []

    all_models_and_versions: dict[str, list[str]] = {}
    for model_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        versions = []
        for v_dir in sorted(model_dir.iterdir()):
            if v_dir.is_dir() and (v_dir / "all_folds").exists():
                versions.append(v_dir.name)
        if versions:
            all_models_and_versions[model_name] = versions

    for base, new in MODEL_PAIRS:
        base_versions = all_models_and_versions.get(base, [])
        new_versions = all_models_and_versions.get(new, [])

        all_versions = set(base_versions) | set(new_versions)
        for version in sorted(all_versions):
            track_label = TRACK_MAP.get(version, version)
            for model_name in [base, new]:
                fold_dir = find_latest_fold_dir(model_name, version)
                if fold_dir is None:
                    continue
                fold_results = load_fold_results(fold_dir)
                if not fold_results:
                    continue

                istps_aps = compute_istps_ap(fold_results)
                substrate_maps, _ = compute_substrate_map(fold_results)

                row = {
                    "Model": model_name,
                    "Version": version,
                    "Track": track_label,
                    "isTPS_AP_mean": np.mean(istps_aps) if istps_aps else np.nan,
                    "isTPS_AP_std": np.std(istps_aps) if istps_aps else np.nan,
                    "mAP_mean": np.mean(substrate_maps) if substrate_maps else np.nan,
                    "mAP_std": np.std(substrate_maps) if substrate_maps else np.nan,
                    "n_folds": len(fold_results),
                }
                rows.append(row)
                logger.info(
                    "  %s / %s (Track %s): isTPS_AP=%.4f±%.4f, mAP=%.4f±%.4f",
                    model_name,
                    version,
                    track_label,
                    row["isTPS_AP_mean"],
                    row["isTPS_AP_std"],
                    row["mAP_mean"],
                    row["mAP_std"],
                )

    return pd.DataFrame(rows)


def main() -> None:
    results_df = evaluate_all()
    out_path = Path("outputs/evaluation_results/new_models_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs New Models")
    print("=" * 80)
    for base, new in MODEL_PAIRS:
        base_df = results_df[results_df["Model"] == base]
        new_df = results_df[results_df["Model"] == new]
        if base_df.empty and new_df.empty:
            continue
        print(f"\n--- {base} vs {new} ---")
        tracks = sorted(
            set(base_df["Track"].tolist() + new_df["Track"].tolist())
        )
        for track in tracks:
            b = base_df[base_df["Track"] == track]
            n = new_df[new_df["Track"] == track]
            b_ap = b["isTPS_AP_mean"].values[0] if len(b) else float("nan")
            n_ap = n["isTPS_AP_mean"].values[0] if len(n) else float("nan")
            b_map = b["mAP_mean"].values[0] if len(b) else float("nan")
            n_map = n["mAP_mean"].values[0] if len(n) else float("nan")
            delta_ap = n_ap - b_ap if not (np.isnan(b_ap) or np.isnan(n_ap)) else float("nan")
            delta_map = n_map - b_map if not (np.isnan(b_map) or np.isnan(n_map)) else float("nan")
            print(
                f"  Track {track:5s}: "
                f"isTPS AP {b_ap:.4f} -> {n_ap:.4f} (Δ={delta_ap:+.4f})  |  "
                f"mAP {b_map:.4f} -> {n_map:.4f} (Δ={delta_map:+.4f})"
            )


if __name__ == "__main__":
    main()
