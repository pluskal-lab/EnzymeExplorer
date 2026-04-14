#!/usr/bin/env python3
"""Per-TPS-type detection evaluation and visualization.

For each TPS type (mono, sesq, di, ...), computes TPS detection AP
treating {TPS of that type} as positives and {negatives} as negatives.
Reports per-type AP and macro-average across types.

Also generates:
  - Bar chart of per-type AP for each model
  - Pie chart of TPS type proportions

Usage::

    conda run -n terpene_miner python scripts/evaluate_per_type_tps_detection.py \\
        --track B --dataset-csv data/EnzymeExplorer_Dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUTPUT_ROOT = Path("outputs")
EVAL_DIR = Path("outputs/evaluation_results")
FIG_DIR = Path("outputs/figures")

TRACK_CONFIGS = {
    "A": {
        "models": {
            "Blastp": "with_minor_reactions_phylo_folds",
            "CLEAN": "with_minor_reactions_phylo_folds",
            "CLEANEcDetection": "with_minor_reactions_phylo_folds",
            "CLEANBetterDetection": "with_minor_reactions_phylo_folds",
            "Foldseek": "with_minor_reactions_phylo_folds",
            "HMM": "with_minor_reactions_phylo_folds",
            "PlmRandomForest": "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
        },
        "target_col": "SMILES_substrate_canonical_no_stereo",
        "type_col": "Type (mono, sesq, di, …)",
        "id_col": "Uniprot ID",
    },
    "C": {
        "models": {
            "Blastp": "synced_folds",
            "CLEAN": "synced_folds",
            "CLEANEcDetection": "synced_folds",
            "CLEANBetterDetection": "synced_folds",
            "Foldseek": "synced_folds",
            "HMM": "synced_folds",
            "PlmRandomForest": "tps_esm-1v-subseq_synced_folds",
            "PlmDomainsRandomForest": "tps_esm-1v-subseq_synced_folds",
        },
        "target_col": "SMILES_substrate_canonical_no_stereo",
        "type_col": "Type (mono, sesq, di, …)",
        "id_col": "Uniprot ID",
    },
    "B": {
        "models": {
            "Blastp": "new_dataset",
            "CLEAN": "new_dataset",
            "CLEANEcDetection": "new_dataset",
            "CLEANBetterDetection": "new_dataset",
            "Foldseek": "new_dataset",
            "HMM": "new_dataset",
            "PlmRandomForest": "tps_esm-1v-subseq_new_dataset",
            "PlmDomainsRandomForest": "tps_esm-1v-subseq_new_dataset",
        },
        "target_col": "SMILES_substrate_canonical_no_stereo",
        "type_col": "Type",
        "id_col": "ID",
    },
    "D": {
        "models": {
            "Blastp": "cross_synced_to_new",
            "CLEAN": "cross_synced_to_new",
            "CLEANEcDetection": "cross_synced_to_new",
            "CLEANBetterDetection": "cross_synced_to_new",
            "Foldseek": "cross_synced_to_new",
            "HMM": "cross_synced_to_new",
            "PlmRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
            "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
        },
        "target_col": "SMILES_substrate_canonical_no_stereo",
        "type_col": "Type",
        "id_col": "ID",
    },
    "E": {
        "models": {
            "Blastp": "cross_new_tps_old_neg",
            "CLEAN": "cross_new_tps_old_neg",
            "CLEANEcDetection": "cross_new_tps_old_neg",
            "CLEANBetterDetection": "cross_new_tps_old_neg",
            "Foldseek": "cross_new_tps_old_neg",
            "HMM": "cross_new_tps_old_neg",
            "PlmRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
            "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
        },
        "target_col": "SMILES_substrate_canonical_no_stereo",
        "type_col": "Type",
        "id_col": "ID",
    },
}

TYPE_DISPLAY = {
    "mono": "Mono",
    "sesq": "Sesqui",
    "di": "Di",
    "tri": "Tri",
    "sester": "Sester",
    "tetra": "Tetra",
    "hemi": "Hemi",
    "sesquar": "Sesquar",
}

_NON_TPS_TYPES = frozenset(
    {"unknown", "negative", "ggpps", "fpps", "gpps", "gfpps", "hsqs", "pt"}
)

MODEL_DISPLAY = {
    "PlmRandomForest": "PlmRF",
    "PlmDomainsRandomForest": "PlmDomainsRF",
    "Blastp": "Blastp",
    "CLEAN": "CLEAN",
    "CLEANEcDetection": "CLEANEc",
    "CLEANBetterDetection": "CLEANProp",
    "HMM": "HMM",
    "Foldseek": "Foldseek",
}

PALETTE = {
    "PlmRF": "#2563eb",
    "PlmDomainsRF": "#0ea5e9",
    "CLEAN": "#dc2626",
    "CLEANEc": "#ef4444",
    "CLEANProp": "#f87171",
    "Blastp": "#f97316",
    "HMM": "#16a34a",
    "Foldseek": "#8b5cf6",
}

MIN_TYPE_COUNT = 3


def _get_type_for_row(row, type_col: str, target_col: str) -> str:
    """Determine TPS type for a row."""
    t = row.get(type_col, "Unknown")
    if pd.isna(t) or str(t).strip() == "" or str(t) == "Unknown":
        return "Unknown"
    return str(t).lower().strip()


def load_fold_results(
    model_type: str, model_version: str, n_folds: int = 5
) -> list[tuple]:
    """Load (val_proba_np, class_names, test_df) for each fold."""
    results = []
    base = OUTPUT_ROOT / model_type / model_version

    candidate = base / "all_folds" / "all_classes"
    if candidate.exists():
        timestamps = sorted(candidate.iterdir(), reverse=True)
        for ts_dir in timestamps:
            if (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break
    elif (base / "all_folds").exists():
        sub = base / "all_folds"
        timestamps = sorted(sub.iterdir(), reverse=True)
        for ts_dir in timestamps:
            if ts_dir.is_dir() and (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break

    for fold_i in range(n_folds):
        pkl = base / f"fold_{fold_i}_results.pkl"
        if not pkl.exists():
            logger.warning("Missing: %s", pkl)
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results.append((fold_i, data))
    return results


def compute_per_type_ap(
    fold_results: list[tuple],
    type_col: str,
    dataset_csv: str | None = None,
    eval_dataset: pd.DataFrame | None = None,
) -> dict[str, list[float]]:
    """Compute AP for each TPS type across folds.

    For each type T:
      - positives = proteins with type == T
      - negatives = proteins with type == "Unknown"
      - AP = average_precision_score(y_true, y_pred)
        where y_pred is the isTPS probability from the model.
    """
    type_2_aps: dict[str, list[float]] = defaultdict(list)

    id_to_type: dict[str, str] = {}
    if eval_dataset is not None:
        tc = "Type" if "Type" in eval_dataset.columns else type_col
        ic = "ID" if "ID" in eval_dataset.columns else "Uniprot ID"
        id_to_type = dict(zip(eval_dataset[ic], eval_dataset[tc].fillna("Unknown")))
    elif dataset_csv is not None:
        ds = pd.read_csv(dataset_csv)
        tc = "Type" if "Type" in ds.columns else type_col
        ic = "ID" if "ID" in ds.columns else "Uniprot ID"
        id_to_type = dict(zip(ds[ic], ds[tc].fillna("Unknown")))

    for fold_i, (val_proba, class_names, test_df) in fold_results:
        if "isTPS" not in class_names:
            logger.warning("Fold %d: no isTPS class", fold_i)
            continue
        tps_idx = list(class_names).index("isTPS")
        y_pred = val_proba[:, tps_idx]

        id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"

        if id_to_type:
            types = test_df[id_col].map(
                lambda x: str(id_to_type.get(x, "Unknown")).lower().strip()
            )
        elif type_col in test_df.columns:
            types = test_df[type_col].fillna("Unknown").str.lower().str.strip()
        else:
            for c in test_df.columns:
                if "type" in c.lower():
                    types = test_df[c].fillna("Unknown").str.lower().str.strip()
                    break
            else:
                logger.warning("No type column found in fold %d", fold_i)
                continue

        neg_mask = types.isin({"unknown", "negative"})
        all_tps_types = [
            t for t in types.unique() if t not in _NON_TPS_TYPES
        ]

        for tps_type in all_tps_types:
            type_mask = types == tps_type
            n_pos = type_mask.sum()
            n_neg = neg_mask.sum()
            if n_pos < MIN_TYPE_COUNT or n_neg < MIN_TYPE_COUNT:
                continue

            subset_mask = type_mask | neg_mask
            y_true_subset = type_mask[subset_mask].astype(int).values
            y_pred_subset = y_pred[subset_mask.values]

            ap = average_precision_score(y_true_subset, y_pred_subset)
            type_2_aps[tps_type].append(ap)

    return type_2_aps


TRACK_DATASET_CSVS = {
    "A": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "C": "data/TPS-Nov19_2023_with_synced_folds.csv",
    "B": "data/EnzymeExplorer_Dataset.csv",
    "D": "data/EnzymeExplorer_Dataset.csv",
    "E": "data/EnzymeExplorer_Dataset.csv",
}


def evaluate_track(
    track: str, n_folds: int = 5
) -> dict[str, dict[str, tuple[float, float, list[float]]]]:
    """Evaluate per-type TPS detection for all models in a track.

    Returns {model_display: {tps_type: (mean_ap, sem_ap, fold_aps)}}
    """
    cfg = TRACK_CONFIGS[track]
    eval_csv = TRACK_DATASET_CSVS.get(track)
    eval_df = pd.read_csv(eval_csv) if eval_csv else None

    results = {}
    for model_type, model_version in cfg["models"].items():
        display = MODEL_DISPLAY.get(model_type, model_type)
        fold_results = load_fold_results(model_type, model_version, n_folds)
        if not fold_results:
            logger.warning("No fold results for %s/%s", model_type, model_version)
            continue
        type_aps = compute_per_type_ap(
            fold_results, cfg["type_col"], eval_dataset=eval_df
        )
        model_results = {}
        for tps_type, aps in type_aps.items():
            if len(aps) >= 2:
                mean_ap = np.mean(aps)
                sem = np.std(aps, ddof=1) / np.sqrt(len(aps))
                model_results[tps_type] = (mean_ap, sem, aps)
        if model_results:
            macro_aps = [v[0] for v in model_results.values()]
            model_results["macro_avg"] = (
                np.mean(macro_aps),
                np.std(macro_aps, ddof=1) / np.sqrt(len(macro_aps)),
                macro_aps,
            )
        results[display] = model_results
    return results


def plot_per_type_bars(
    track_results: dict[str, dict[str, tuple]],
    track_label: str,
    outdir: Path,
) -> None:
    """Grouped bar chart of per-type TPS detection AP."""
    all_types = sorted(
        {t for m in track_results.values() for t in m if t != "macro_avg"},
        key=lambda t: -max(
            (track_results[m].get(t, (0,))[0] for m in track_results), default=0
        ),
    )
    all_types.append("macro_avg")
    models = [
        m
        for m in [
            "PlmRF",
            "PlmDomainsRF",
            "CLEAN",
            "CLEANEc",
            "CLEANProp",
            "Blastp",
            "HMM",
            "Foldseek",
        ]
        if m in track_results
    ]

    n_types = len(all_types)
    n_models = len(models)
    bar_w = 0.8 / n_models
    x = np.arange(n_types)

    fig, ax = plt.subplots(figsize=(max(12, 1.3 * n_types), 5.5))

    for mi, m in enumerate(models):
        means, sems = [], []
        for t in all_types:
            v = track_results[m].get(t)
            means.append(v[0] if v else np.nan)
            sems.append(v[1] if v else 0.0)
        offset = (mi - (n_models - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means,
            bar_w * 0.88,
            yerr=sems,
            label=m,
            color=PALETTE.get(m, f"C{mi}"),
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            zorder=3,
        )

    type_labels = [
        TYPE_DISPLAY.get(t, t.title()) if t != "macro_avg" else "Macro\nAvg"
        for t in all_types
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=9, fontweight="medium")
    ax.set_ylabel("Average Precision (TPS detection)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.set_title(
        f"TPS Detection by Type — Track {track_label}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax.axvline(n_types - 1.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    plt.tight_layout()
    stem = f"fig_per_type_tps_track_{track_label.lower()}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)
    logger.info("Saved %s", stem)


def plot_type_proportions(dataset_csv: str, outdir: Path) -> None:
    """Pie chart of TPS type proportions (by unique enzymes)."""
    df = pd.read_csv(dataset_csv)
    type_col = "Type" if "Type" in df.columns else "Type (mono, sesq, di, …)"
    tps = df[df[type_col] != "Unknown"]
    id_col = "ID" if "ID" in df.columns else "Uniprot ID"
    type_counts = tps.drop_duplicates(id_col).groupby(type_col).size()
    type_counts = type_counts.sort_values(ascending=False)

    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=[TYPE_DISPLAY.get(t.lower(), t) for t in type_counts.index],
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(type_counts) / 100)})",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(
        "TPS Type Distribution (unique enzymes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"fig_type_proportions.{ext}")
    plt.close(fig)
    logger.info("Saved fig_type_proportions")


def save_results_csv(
    all_track_results: dict[str, dict[str, dict[str, tuple]]],
    outdir: Path,
) -> None:
    """Save per-type results to CSV."""
    rows = []
    for track, track_res in all_track_results.items():
        for model, type_res in track_res.items():
            for tps_type, (mean_ap, sem, _) in type_res.items():
                rows.append(
                    {
                        "Track": track,
                        "Model": model,
                        "TPS_Type": tps_type,
                        "AP_mean": mean_ap,
                        "AP_sem": sem,
                    }
                )
    df = pd.DataFrame(rows)
    out_path = outdir / "per_type_tps_detection.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tracks",
        nargs="+",
        default=["A", "C", "B", "D", "E"],
        help="Tracks to evaluate",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--outdir",
        type=Path,
        default=FIG_DIR,
    )
    p.add_argument(
        "--dataset-csv",
        default="data/EnzymeExplorer_Dataset.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for track in args.tracks:
        if track not in TRACK_CONFIGS:
            logger.warning("Unknown track: %s", track)
            continue
        logger.info("Evaluating Track %s...", track)
        results = evaluate_track(track, args.n_folds)
        all_results[track] = results

        for model, type_res in results.items():
            logger.info("  %s:", model)
            for tps_type in sorted(type_res):
                mean_ap, sem, _ = type_res[tps_type]
                logger.info("    %s: AP=%.3f ± %.3f", tps_type, mean_ap, sem)

        if results:
            plot_per_type_bars(results, track, args.outdir)
        else:
            logger.warning("No results for Track %s, skipping plot", track)

    plot_type_proportions(args.dataset_csv, args.outdir)
    save_results_csv(all_results, args.outdir)


if __name__ == "__main__":
    main()
