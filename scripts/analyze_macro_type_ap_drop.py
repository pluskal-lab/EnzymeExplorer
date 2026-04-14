#!/usr/bin/env python3
"""Analyze per-TPS-type macro AP drops across tracks.

Investigates why per-type macro AP drops significantly on Track B
for all methods and for CLEAN on D/E/B.

Usage::

    conda run -n terpene_miner python scripts/analyze_macro_type_ap_drop.py
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUTPUT_ROOT = Path("outputs")
MIN_TYPE_COUNT = 3

TRACKS = {
    "A": {
        "models": {
            "PlmRandomForest": "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
            "Blastp": "with_minor_reactions_phylo_folds",
            "CLEAN": "with_minor_reactions_phylo_folds",
            "HMM": "with_minor_reactions_phylo_folds",
            "Foldseek": "with_minor_reactions_phylo_folds",
        },
        "eval_csv": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
        "type_col": "Type (mono, sesq, di, \u2026)",
    },
    "C": {
        "models": {
            "PlmRandomForest": "tps_esm-1v-subseq_synced_folds",
            "Blastp": "synced_folds",
            "CLEAN": "synced_folds",
            "HMM": "synced_folds",
            "Foldseek": "synced_folds",
        },
        "eval_csv": "data/TPS-Nov19_2023_with_synced_folds.csv",
        "type_col": "Type (mono, sesq, di, \u2026)",
    },
    "B": {
        "models": {
            "PlmRandomForest": "tps_esm-1v-subseq_new_dataset",
            "Blastp": "new_dataset",
            "CLEAN": "new_dataset",
            "HMM": "new_dataset",
            "Foldseek": "new_dataset",
        },
        "eval_csv": "data/EnzymeExplorer_Dataset.csv",
        "type_col": "Type",
    },
    "D": {
        "models": {
            "PlmRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
            "Blastp": "cross_synced_to_new",
            "CLEAN": "cross_synced_to_new",
            "HMM": "cross_synced_to_new",
            "Foldseek": "cross_synced_to_new",
        },
        "eval_csv": "data/EnzymeExplorer_Dataset.csv",
        "type_col": "Type",
    },
    "E": {
        "models": {
            "PlmRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
            "Blastp": "cross_new_tps_old_neg",
            "CLEAN": "cross_new_tps_old_neg",
            "HMM": "cross_new_tps_old_neg",
            "Foldseek": "cross_new_tps_old_neg",
        },
        "eval_csv": "data/EnzymeExplorer_Dataset.csv",
        "type_col": "Type",
    },
}


def load_fold(model: str, version: str, n_folds: int = 5) -> list[tuple]:
    base = OUTPUT_ROOT / model / version / "all_folds" / "all_classes"
    if not base.exists():
        return []
    ts_dirs = sorted(base.iterdir(), reverse=True)
    fold_dir = None
    for td in ts_dirs:
        if (td / "fold_0_results.pkl").exists():
            fold_dir = td
            break
    if not fold_dir:
        return []
    results = []
    for i in range(n_folds):
        pkl = fold_dir / f"fold_{i}_results.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results.append((i, data))
    return results


def analyze_track(track_name: str, cfg: dict) -> dict:
    eval_df = pd.read_csv(cfg["eval_csv"])
    tc = "Type" if "Type" in eval_df.columns else cfg["type_col"]
    ic = "ID" if "ID" in eval_df.columns else "Uniprot ID"
    id_to_type = dict(zip(eval_df[ic], eval_df[tc].fillna("Unknown")))

    type_counts = (
        eval_df[eval_df[tc] != "Unknown"]
        .drop_duplicates(ic)
        .groupby(tc)
        .size()
        .sort_values(ascending=False)
    )
    print(f"\n{'=' * 60}")
    print(f"Track {track_name}")
    print(f"{'=' * 60}")
    print("Type distribution (unique enzymes):")
    for t, c in type_counts.items():
        print(f"  {t:15s}: {c:5d}")
    print(f"  {'Total TPS':15s}: {type_counts.sum():5d}")
    n_neg = eval_df[eval_df[tc] == "Unknown"].drop_duplicates(ic).shape[0]
    print(f"  {'Negatives':15s}: {n_neg:5d}")

    track_results = {}
    for model, version in cfg["models"].items():
        folds = load_fold(model, version)
        if not folds:
            continue
        print(f"\n  --- {model} ---")
        per_type_aps: dict[str, list[float]] = defaultdict(list)
        per_type_scores: dict[str, list[tuple]] = defaultdict(list)

        for fold_i, (proba, cnames, test_df) in folds:
            clist = list(cnames) if not isinstance(cnames, list) else cnames
            if "isTPS" not in clist:
                continue
            tps_idx = clist.index("isTPS")
            y_pred = proba[:, tps_idx]
            id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
            ids = test_df[id_col].values
            types = pd.Series(
                [str(id_to_type.get(p, "Unknown")).lower().strip() for p in ids]
            )
            _non_tps = {"unknown", "negative", "ggpps", "fpps", "gpps",
                        "gfpps", "hsqs", "pt"}
            neg_mask = types.isin({"unknown", "negative"})
            n_neg_fold = int(neg_mask.sum())
            neg_scores = y_pred[neg_mask.values]

            for tps_type in sorted(
                t for t in types.unique() if t not in _non_tps
            ):
                type_mask = types == tps_type
                n_pos = int(type_mask.sum())
                if n_pos < MIN_TYPE_COUNT or n_neg_fold < MIN_TYPE_COUNT:
                    continue
                subset = type_mask | neg_mask
                y_true = type_mask[subset].astype(int).values
                y_p = y_pred[subset.values]
                ap = average_precision_score(y_true, y_p)
                per_type_aps[tps_type].append(ap)

                pos_scores = y_pred[type_mask.values]
                per_type_scores[tps_type].append(
                    (fold_i, n_pos, ap, pos_scores.mean(), np.median(pos_scores),
                     neg_scores.mean(), np.median(neg_scores),
                     (pos_scores == 0).sum(), n_pos)
                )

        if per_type_aps:
            type_means = {}
            for t, aps in sorted(per_type_aps.items()):
                m = float(np.mean(aps))
                type_means[t] = m
                marker = "  <<<" if m < 0.5 else ""
                print(f"    {t:15s}: AP={m:.3f} (n_folds={len(aps)}){marker}")

                if m < 0.5:
                    for info in per_type_scores[t]:
                        fi, np_, ap_, pm, pmed, nm, nmed, nzero, ntot = info
                        print(
                            f"      fold {fi}: AP={ap_:.3f} n={np_:3d} "
                            f"pos_mean={pm:.3f} pos_med={pmed:.3f} "
                            f"neg_mean={nm:.3f} neg_med={nmed:.3f} "
                            f"zero_pos={nzero}/{ntot}"
                        )

            macro = float(np.mean(list(type_means.values())))
            print(f"    {'MACRO AVG':15s}: AP={macro:.3f}")
            track_results[model] = type_means

    return track_results


def compare_type_distributions():
    """Compare TPS type distributions between old and new datasets."""
    old_df = pd.read_csv(
        "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv"
    )
    new_df = pd.read_csv("data/EnzymeExplorer_Dataset.csv")

    old_tc = "Type (mono, sesq, di, \u2026)"
    new_tc = "Type"
    old_ic = "Uniprot ID"
    new_ic = "ID"

    old_types = (
        old_df[old_df[old_tc] != "Unknown"]
        .drop_duplicates(old_ic)[old_tc]
        .str.lower()
        .str.strip()
        .value_counts()
    )
    new_types = (
        new_df[new_df[new_tc] != "Unknown"]
        .drop_duplicates(new_ic)[new_tc]
        .str.lower()
        .str.strip()
        .value_counts()
    )

    all_types = sorted(set(old_types.index) | set(new_types.index))

    print(f"\n{'=' * 60}")
    print("Type Distribution Comparison: Old vs New Dataset")
    print(f"{'=' * 60}")
    print(f"{'Type':15s} {'Old':>6s} {'New':>6s} {'Delta':>6s} {'New types?':>10s}")
    print("-" * 50)
    for t in all_types:
        o = old_types.get(t, 0)
        n = new_types.get(t, 0)
        delta = n - o
        new_flag = "NEW" if o == 0 and n > 0 else ""
        print(f"  {t:15s} {o:5d}  {n:5d}  {delta:+5d}  {new_flag}")

    old_neg = old_df[old_df[old_tc] == "Unknown"].drop_duplicates(old_ic).shape[0]
    new_neg = new_df[new_df[new_tc] == "Unknown"].drop_duplicates(new_ic).shape[0]
    print(f"\n  Negatives: old={old_neg}, new={new_neg}")
    print(f"  Neg/TPS ratio: old={old_neg / old_types.sum():.1f}, "
          f"new={new_neg / new_types.sum():.1f}")


def main():
    compare_type_distributions()

    all_results = {}
    for track, cfg in TRACKS.items():
        all_results[track] = analyze_track(track, cfg)

    print(f"\n{'=' * 60}")
    print("SUMMARY: Per-type macro AP across tracks")
    print(f"{'=' * 60}")
    models = ["PlmRandomForest", "Blastp", "CLEAN", "HMM", "Foldseek"]
    header = f"{'Model':20s}"
    for t in ["A", "C", "D", "E", "B"]:
        header += f" {t:>8s}"
    print(header)
    print("-" * len(header))
    for model in models:
        row = f"  {model:18s}"
        for t in ["A", "C", "D", "E", "B"]:
            type_means = all_results.get(t, {}).get(model, {})
            if type_means:
                macro = float(np.mean(list(type_means.values())))
                row += f" {macro:8.3f}"
            else:
                row += "      ---"
        print(row)


if __name__ == "__main__":
    main()
