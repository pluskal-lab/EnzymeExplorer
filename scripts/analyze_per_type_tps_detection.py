#!/usr/bin/env python3
"""Deep analysis of per-TPS-type detection performance.

Produces:
  1. Per-fold, per-type sample counts & AP for every model/track
  2. Score-distribution diagnostics for anomalies
  3. Heatmap of model × type AP (with counts annotated)
  4. Fold-level swarm/strip plots for high-variance types
  5. Detailed log of anomalies (unexpected drops, zero scores, etc.)

Usage::

    conda run -n terpene_miner python scripts/analyze_per_type_tps_detection.py
"""

from __future__ import annotations

import json
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
FIG_DIR = Path("outputs/figures/per_type_analysis")

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
        "type_col": "Type (mono, sesq, di, \u2026)",
        "id_col": "Uniprot ID",
        "dataset_csv": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
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
        "type_col": "Type (mono, sesq, di, \u2026)",
        "id_col": "Uniprot ID",
        "dataset_csv": "data/TPS-Nov19_2023_with_synced_folds.csv",
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
        "type_col": "Type",
        "id_col": "ID",
        "dataset_csv": "data/EnzymeExplorer_Dataset.csv",
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
        "type_col": "Type",
        "id_col": "ID",
        "dataset_csv": "data/EnzymeExplorer_Dataset.csv",
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
        "type_col": "Type",
        "id_col": "ID",
        "dataset_csv": "data/EnzymeExplorer_Dataset.csv",
    },
}

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

MODEL_ORDER = [
    "PlmRF",
    "PlmDomainsRF",
    "HMM",
    "Foldseek",
    "Blastp",
    "CLEAN",
    "CLEANEc",
    "CLEANProp",
]

TYPE_DISPLAY = {
    "mono": "Mono",
    "sesq": "Sesqui",
    "di": "Di",
    "tri": "Tri",
    "sester": "Sester",
    "tetra": "Tetra",
    "hemi": "Hemi",
    "sesquar": "Sesquar",
    "di-int": "Di-int",
}

_NON_TPS_TYPES = frozenset(
    {"unknown", "negative", "ggpps", "fpps", "gpps", "gfpps", "hsqs", "pt"}
)

MIN_TYPE_COUNT = 3


def load_fold_results(model_type: str, model_version: str, n_folds: int = 5):
    """Load fold results; returns list of (fold_i, (proba, class_names, df))."""
    results = []
    base = OUTPUT_ROOT / model_type / model_version
    candidate = base / "all_folds" / "all_classes"
    if candidate.exists():
        for ts_dir in sorted(candidate.iterdir(), reverse=True):
            if (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break
    elif (base / "all_folds").exists():
        for ts_dir in sorted((base / "all_folds").iterdir(), reverse=True):
            if ts_dir.is_dir() and (ts_dir / "fold_0_results.pkl").exists():
                base = ts_dir
                break
    for fold_i in range(n_folds):
        pkl = base / f"fold_{fold_i}_results.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results.append((fold_i, data))
    return results


def get_types_for_fold(test_df, eval_df, type_col, id_col):
    """Return Series of type labels (lowercased) for test_df rows."""
    ic = "ID" if "ID" in test_df.columns else "Uniprot ID"
    if eval_df is not None:
        tc = "Type" if "Type" in eval_df.columns else type_col
        ec = "ID" if "ID" in eval_df.columns else "Uniprot ID"
        id_to_type = dict(zip(eval_df[ec], eval_df[tc].fillna("Unknown")))
        types = test_df[ic].map(
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
            types = pd.Series(["unknown"] * len(test_df), index=test_df.index)
    return types


# ── Phase 1: Data Inventory ──────────────────────────────────────────────


def phase1_data_inventory():
    """Count TPS types across datasets and per fold."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA INVENTORY — per-type sample counts")
    logger.info("=" * 70)

    all_counts = {}

    for track, cfg in TRACK_CONFIGS.items():
        eval_df = pd.read_csv(cfg["dataset_csv"])
        first_model = next(iter(cfg["models"]))
        first_version = cfg["models"][first_model]
        fold_results = load_fold_results(first_model, first_version)

        if not fold_results:
            logger.warning("Track %s: no fold results found", track)
            continue

        logger.info("\n--- Track %s ---", track)
        track_counts = {}

        for fold_i, (val_proba, class_names, test_df) in fold_results:
            types = get_types_for_fold(
                test_df, eval_df, cfg["type_col"], cfg["id_col"]
            )
            vc = types.value_counts()
            for tps_type, count in vc.items():
                if tps_type not in track_counts:
                    track_counts[tps_type] = {}
                track_counts[tps_type][fold_i] = count

        header = f"{'Type':<12} " + " ".join(
            f"{'fold_' + str(i):>8}" for i in range(5)
        ) + f" {'total':>8} {'mean':>8}"
        logger.info(header)

        for tps_type in sorted(track_counts.keys()):
            fold_vals = [track_counts[tps_type].get(i, 0) for i in range(5)]
            total = sum(fold_vals)
            mean = np.mean(fold_vals)
            row = f"{tps_type:<12} " + " ".join(
                f"{v:>8}" for v in fold_vals
            ) + f" {total:>8} {mean:>8.1f}"
            logger.info(row)

        all_counts[track] = track_counts

    return all_counts


# ── Phase 2: Per-fold detailed AP + score distributions ──────────────────


def phase2_detailed_ap():
    """Compute per-fold AP and score distributions for every model/track/type."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: DETAILED PER-FOLD AP & SCORE DISTRIBUTIONS")
    logger.info("=" * 70)

    all_records = []
    anomalies = []

    for track, cfg in TRACK_CONFIGS.items():
        eval_df = pd.read_csv(cfg["dataset_csv"])

        for model_type, model_version in cfg["models"].items():
            model_display = MODEL_DISPLAY.get(model_type, model_type)
            fold_results = load_fold_results(model_type, model_version)
            if not fold_results:
                continue

            for fold_i, (val_proba, class_names, test_df) in fold_results:
                class_list = list(class_names)
                if "isTPS" not in class_list:
                    continue
                tps_idx = class_list.index("isTPS")
                y_pred = val_proba[:, tps_idx]

                types = get_types_for_fold(
                    test_df, eval_df, cfg["type_col"], cfg["id_col"]
                )
                neg_mask = types.isin({"unknown", "negative"})
                n_neg = neg_mask.sum()

                for tps_type in [
                    t for t in types.unique() if t not in _NON_TPS_TYPES
                ]:
                    type_mask = types == tps_type
                    n_pos = type_mask.sum()
                    if n_pos < MIN_TYPE_COUNT or n_neg < MIN_TYPE_COUNT:
                        continue

                    subset_mask = type_mask | neg_mask
                    y_true_sub = type_mask[subset_mask].astype(int).values
                    y_pred_sub = y_pred[subset_mask.values]

                    ap = average_precision_score(y_true_sub, y_pred_sub)

                    pos_scores = y_pred[type_mask.values]
                    neg_scores = y_pred[neg_mask.values]

                    record = {
                        "track": track,
                        "model": model_display,
                        "tps_type": tps_type,
                        "fold": fold_i,
                        "n_pos": int(n_pos),
                        "n_neg": int(n_neg),
                        "ap": float(ap),
                        "pos_mean": float(pos_scores.mean()),
                        "pos_median": float(np.median(pos_scores)),
                        "pos_min": float(pos_scores.min()),
                        "pos_max": float(pos_scores.max()),
                        "pos_frac_zero": float((pos_scores == 0).mean()),
                        "neg_mean": float(neg_scores.mean()),
                        "neg_median": float(np.median(neg_scores)),
                        "neg_max": float(neg_scores.max()),
                        "neg_frac_nonzero": float((neg_scores > 0).mean()),
                    }
                    all_records.append(record)

                    if ap < 0.10 and n_pos >= 5:
                        anomalies.append(
                            f"  LOW AP: Track {track}, {model_display}, "
                            f"{tps_type}, fold {fold_i}: AP={ap:.3f} "
                            f"(n_pos={n_pos}, pos_mean={pos_scores.mean():.4f}, "
                            f"pos_frac_zero={record['pos_frac_zero']:.2f}, "
                            f"neg_frac_nonzero={record['neg_frac_nonzero']:.2f})"
                        )

                    if record["pos_frac_zero"] > 0.5 and n_pos >= 5:
                        anomalies.append(
                            f"  ZERO SCORES: Track {track}, {model_display}, "
                            f"{tps_type}, fold {fold_i}: "
                            f"{record['pos_frac_zero']:.0%} of {n_pos} positives "
                            f"have score=0"
                        )

    df = pd.DataFrame(all_records)

    logger.info("\n--- ANOMALY REPORT ---")
    if anomalies:
        for a in sorted(set(anomalies)):
            logger.info(a)
    else:
        logger.info("  No anomalies detected.")

    logger.info("\n--- HIGH-VARIANCE TYPES (SEM > 0.10) ---")
    grouped = (
        df.groupby(["track", "model", "tps_type"])
        .agg(
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            n_folds=("ap", "count"),
            mean_n_pos=("n_pos", "mean"),
        )
        .reset_index()
    )
    grouped["ap_sem"] = grouped["ap_std"] / np.sqrt(grouped["n_folds"])
    high_var = grouped[grouped["ap_sem"] > 0.10].sort_values("ap_sem", ascending=False)
    if len(high_var) > 0:
        for _, row in high_var.iterrows():
            logger.info(
                "  Track %s, %s, %s: AP=%.3f ± %.3f (SEM), "
                "mean_n_pos=%.0f",
                row["track"],
                row["model"],
                row["tps_type"],
                row["ap_mean"],
                row["ap_sem"],
                row["mean_n_pos"],
            )
    else:
        logger.info("  None.")

    return df, grouped


# ── Phase 3: Method-specific deep dives ──────────────────────────────────


def phase3_method_deepdives(df: pd.DataFrame):
    """Investigate specific anomalies identified in the previous run."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: METHOD-SPECIFIC DEEP DIVES")
    logger.info("=" * 70)

    # 3a. HMM di-int on Track A (AP=0.489 vs ~0.99 for Blastp/Foldseek)
    logger.info("\n--- 3a. HMM di-int on Track A ---")
    hmm_diint = df[
        (df["track"] == "A")
        & (df["model"] == "HMM")
        & (df["tps_type"] == "di-int")
    ]
    if len(hmm_diint) > 0:
        logger.info("  Per-fold details:")
        for _, r in hmm_diint.iterrows():
            logger.info(
                "    fold %d: AP=%.3f, n_pos=%d, n_neg=%d, "
                "pos_mean=%.4f, pos_median=%.4f, pos_frac_zero=%.2f, "
                "neg_frac_nonzero=%.2f",
                r["fold"], r["ap"], r["n_pos"], r["n_neg"],
                r["pos_mean"], r["pos_median"], r["pos_frac_zero"],
                r["neg_frac_nonzero"],
            )
        others = df[
            (df["track"] == "A")
            & (df["tps_type"] == "di-int")
            & (df["model"] != "HMM")
        ]
        logger.info("  Other models on di-int Track A:")
        for model in others["model"].unique():
            sub = others[others["model"] == model]
            logger.info(
                "    %s: mean AP=%.3f, per-fold: %s",
                model,
                sub["ap"].mean(),
                [f"{v:.3f}" for v in sub["ap"].values],
            )
    else:
        logger.info("  No data found for HMM di-int Track A")

    # 3b. CLEANEc sester collapse
    logger.info("\n--- 3c. CLEANEc sester detection ---")
    for track in ["A", "C"]:
        clean_ses = df[
            (df["track"] == track)
            & (df["model"] == "CLEAN")
            & (df["tps_type"] == "sester")
        ]
        cleanec_ses = df[
            (df["track"] == track)
            & (df["model"] == "CLEANEc")
            & (df["tps_type"] == "sester")
        ]
        if len(clean_ses) > 0 and len(cleanec_ses) > 0:
            logger.info(
                "  Track %s: CLEAN sester mean AP=%.3f, CLEANEc sester mean AP=%.3f",
                track,
                clean_ses["ap"].mean(),
                cleanec_ses["ap"].mean(),
            )
            logger.info(
                "    CLEANEc pos_frac_zero: %s",
                [f"{v:.2f}" for v in cleanec_ses["pos_frac_zero"].values],
            )
            logger.info(
                "    CLEANEc pos_mean: %s",
                [f"{v:.4f}" for v in cleanec_ses["pos_mean"].values],
            )

    # 3d. Sester on Track B — why near-zero for all?
    logger.info("\n--- 3d. Sester detection on Track B ---")
    sester_b = df[(df["track"] == "B") & (df["tps_type"] == "sester")]
    if len(sester_b) > 0:
        for model in MODEL_ORDER:
            sub = sester_b[sester_b["model"] == model]
            if len(sub) > 0:
                logger.info(
                    "  %s: mean AP=%.3f, mean n_pos=%.0f, "
                    "pos_mean_score=%.4f, pos_frac_zero=%.2f",
                    model,
                    sub["ap"].mean(),
                    sub["n_pos"].mean(),
                    sub["pos_mean"].mean(),
                    sub["pos_frac_zero"].mean(),
                )

    # 3e. GGPPS detection — why CLEANEc drops to 0.051 on A
    logger.info("\n--- 3e. GGPPS/FPPS detection ---")
    for track in ["A", "C"]:
        for ttype in ["ggpps", "fpps"]:
            sub = df[(df["track"] == track) & (df["tps_type"] == ttype)]
            if len(sub) == 0:
                continue
            logger.info("  Track %s, type %s:", track, ttype)
            for model in MODEL_ORDER:
                msub = sub[sub["model"] == model]
                if len(msub) > 0:
                    logger.info(
                        "    %s: mean AP=%.3f, pos_frac_zero=%.2f, "
                        "n_pos_mean=%.0f",
                        model,
                        msub["ap"].mean(),
                        msub["pos_frac_zero"].mean(),
                        msub["n_pos"].mean(),
                    )


# ── Phase 4: Check CLEANEc EC coverage for specific types ──────────────


def phase4_clean_ec_coverage():
    """Check which TPS types are well/poorly covered by the extended EC mapping."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: CLEANEc EC COVERAGE BY TPS TYPE")
    logger.info("=" * 70)

    csv_path = Path("/home/samusevich/CLEAN/app/results")
    ec_mapping_path = Path("data/ec_to_substrate_mapping_extended.json")

    if not ec_mapping_path.exists():
        logger.warning("Extended EC mapping not found")
        return
    with open(ec_mapping_path) as f:
        ec_mapping = json.load(f)
    tps_ecs = {f"EC:{ec}" for ec, subs in ec_mapping.items()
               if set(subs) - {"precursor substr"}}
    logger.info("TPS ECs in extended mapping: %d", len(tps_ecs))

    id_to_ec_conf = {}
    for csv_file in sorted(csv_path.glob("*_maxsep.csv")):
        with open(csv_file) as f:
            for line in f:
                parts = line.strip().split(",")
                pid = parts[0]
                if pid in id_to_ec_conf:
                    continue
                id_to_ec_conf[pid] = {}
                for ec_part in parts[1:]:
                    ec_class, conf = ec_part.split("/")
                    id_to_ec_conf[pid][ec_class] = float(conf)

    for dataset_name, dataset_path, type_col, id_col in [
        ("old", "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
         "Type (mono, sesq, di, \u2026)", "Uniprot ID"),
        ("new", "data/EnzymeExplorer_Dataset.csv", "Type", "ID"),
    ]:
        df = pd.read_csv(dataset_path)
        tps_df = df[~df[type_col].isin(["Unknown"])].copy()
        tps_df["type_lower"] = tps_df[type_col].str.lower().str.strip()

        logger.info("\n--- Dataset: %s (%d TPS rows) ---", dataset_name, len(tps_df))

        for tps_type in sorted(tps_df["type_lower"].unique()):
            type_df = tps_df[tps_df["type_lower"] == tps_type]
            n_total = len(type_df)
            ids = type_df[id_col].unique()

            n_has_tps_ec = 0
            n_no_tps_ec = 0
            n_no_prediction = 0
            max_tps_confs = []

            for pid in ids:
                if pid not in id_to_ec_conf:
                    n_no_prediction += 1
                    continue
                ecs = id_to_ec_conf[pid]
                max_tps_conf = max(
                    (c for ec, c in ecs.items() if ec in tps_ecs), default=0.0
                )
                if max_tps_conf > 0:
                    n_has_tps_ec += 1
                    max_tps_confs.append(max_tps_conf)
                else:
                    n_no_tps_ec += 1

            pct_covered = (
                n_has_tps_ec / (n_has_tps_ec + n_no_tps_ec) * 100
                if (n_has_tps_ec + n_no_tps_ec) > 0
                else 0
            )
            mean_conf = np.mean(max_tps_confs) if max_tps_confs else 0.0
            logger.info(
                "  %s: %d unique IDs, %d have TPS EC (%.0f%%), "
                "%d no TPS EC, %d no prediction, mean_tps_conf=%.3f",
                tps_type,
                len(ids),
                n_has_tps_ec,
                pct_covered,
                n_no_tps_ec,
                n_no_prediction,
                mean_conf,
            )


# ── Phase 5: Visualizations ─────────────────────────────────────────────


def phase5_visualizations(df: pd.DataFrame, grouped: pd.DataFrame):
    """Create heatmaps and detailed visualizations."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: VISUALIZATIONS")
    logger.info("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 5a. Heatmap for each track: Model × Type, with sample counts
    for track in ["A", "C", "B", "D", "E"]:
        tdata = grouped[grouped["track"] == track].copy()
        if len(tdata) == 0:
            continue

        types_present = sorted(
            [t for t in tdata["tps_type"].unique() if t != "macro_avg"],
            key=lambda t: -tdata[tdata["tps_type"] == t]["ap_mean"].max(),
        )
        types_present.append("macro_avg")

        models_present = [m for m in MODEL_ORDER if m in tdata["model"].values]

        mat = np.full((len(models_present), len(types_present)), np.nan)
        counts = np.full((len(models_present), len(types_present)), 0.0)

        for mi, model in enumerate(models_present):
            for ti, tps_type in enumerate(types_present):
                row = tdata[
                    (tdata["model"] == model) & (tdata["tps_type"] == tps_type)
                ]
                if len(row) > 0:
                    mat[mi, ti] = row["ap_mean"].values[0]
                    counts[mi, ti] = row["mean_n_pos"].values[0]

        fig, ax = plt.subplots(
            figsize=(max(10, 0.9 * len(types_present)), max(4, 0.55 * len(models_present)))
        )
        im = ax.imshow(
            mat,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        for mi in range(len(models_present)):
            for ti in range(len(types_present)):
                val = mat[mi, ti]
                if np.isnan(val):
                    ax.text(ti, mi, "—", ha="center", va="center", fontsize=8,
                            color="gray")
                else:
                    cnt = counts[mi, ti]
                    color = "white" if val < 0.4 or val > 0.85 else "black"
                    if types_present[ti] == "macro_avg":
                        ax.text(ti, mi, f"{val:.2f}", ha="center", va="center",
                                fontsize=9, fontweight="bold", color=color)
                    else:
                        ax.text(ti, mi, f"{val:.2f}\n(n={cnt:.0f})", ha="center",
                                va="center", fontsize=7, color=color)

        type_labels = [TYPE_DISPLAY.get(t, t.title()) if t != "macro_avg" else "Macro\nAvg"
                       for t in types_present]
        ax.set_xticks(range(len(types_present)))
        ax.set_xticklabels(type_labels, fontsize=9, fontweight="medium")
        ax.set_yticks(range(len(models_present)))
        ax.set_yticklabels(models_present, fontsize=10)

        ax.axvline(len(types_present) - 1.5, color="black", linewidth=1.5,
                   linestyle="--")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Average Precision", fontsize=10)
        ax.set_title(
            f"Per-Type TPS Detection AP — Track {track}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIG_DIR / f"heatmap_track_{track.lower()}.{ext}")
        plt.close(fig)
        logger.info("Saved heatmap_track_%s", track.lower())

    # 5b. Fold-level strip plot for high-variance cases
    high_var_cases = grouped[grouped["ap_sem"] > 0.08].sort_values(
        "ap_sem", ascending=False
    ).head(20)

    if len(high_var_cases) > 0:
        fig, ax = plt.subplots(
            figsize=(12, max(4, 0.4 * len(high_var_cases)))
        )
        y_positions = []
        y_labels = []

        for idx, (_, row) in enumerate(high_var_cases.iterrows()):
            fold_aps = df[
                (df["track"] == row["track"])
                & (df["model"] == row["model"])
                & (df["tps_type"] == row["tps_type"])
            ]["ap"].values

            y_pos = len(high_var_cases) - idx - 1
            y_positions.append(y_pos)
            y_labels.append(
                f"{row['track']}/{row['model']}/{row['tps_type']} "
                f"(n={row['mean_n_pos']:.0f})"
            )

            ax.scatter(
                fold_aps,
                [y_pos] * len(fold_aps),
                alpha=0.7,
                s=40,
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.plot(
                [row["ap_mean"], row["ap_mean"]],
                [y_pos - 0.3, y_pos + 0.3],
                color="red",
                linewidth=2,
                zorder=4,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Average Precision", fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(
            "Highest-Variance Per-Type Results (individual folds)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIG_DIR / f"fold_variance_strip.{ext}")
        plt.close(fig)
        logger.info("Saved fold_variance_strip")

    # 5c. Summary heatmap across tracks (macro-avg only)
    macro_data = grouped[grouped["tps_type"] == "macro_avg"].copy()
    if len(macro_data) > 0:
        tracks_order = ["A", "C", "B", "D", "E"]
        models_in_data = [m for m in MODEL_ORDER if m in macro_data["model"].values]
        mat = np.full((len(models_in_data), len(tracks_order)), np.nan)
        for mi, model in enumerate(models_in_data):
            for ti, track in enumerate(tracks_order):
                row = macro_data[
                    (macro_data["model"] == model) & (macro_data["track"] == track)
                ]
                if len(row) > 0:
                    mat[mi, ti] = row["ap_mean"].values[0]

        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(models_in_data))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for mi in range(len(models_in_data)):
            for ti in range(len(tracks_order)):
                val = mat[mi, ti]
                if not np.isnan(val):
                    color = "white" if val < 0.35 or val > 0.85 else "black"
                    ax.text(ti, mi, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)
                else:
                    ax.text(ti, mi, "—", ha="center", va="center",
                            fontsize=9, color="gray")

        ax.set_xticks(range(len(tracks_order)))
        ax.set_xticklabels([f"Track {t}" for t in tracks_order], fontsize=10)
        ax.set_yticks(range(len(models_in_data)))
        ax.set_yticklabels(models_in_data, fontsize=10)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Macro-Average AP", fontsize=10)
        ax.set_title(
            "Macro-Average Per-Type TPS Detection AP",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIG_DIR / f"macro_avg_heatmap.{ext}")
        plt.close(fig)
        logger.info("Saved macro_avg_heatmap")

    # 5d. Type-distribution comparison: old vs new dataset
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_i, (label, csv_path, type_col, id_col) in enumerate([
        ("Old Dataset (Track A)",
         "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
         "Type (mono, sesq, di, \u2026)", "Uniprot ID"),
        ("New Dataset (Track B)",
         "data/EnzymeExplorer_Dataset.csv", "Type", "ID"),
    ]):
        ds = pd.read_csv(csv_path)
        tps = ds[~ds[type_col].isin(["Unknown"])].drop_duplicates(id_col)
        counts = tps[type_col].str.lower().str.strip().value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        axes[ax_i].barh(
            [TYPE_DISPLAY.get(t, t.title()) for t in counts.index],
            counts.values,
            color=colors,
            edgecolor="white",
        )
        for i, (cnt, t) in enumerate(zip(counts.values, counts.index)):
            axes[ax_i].text(
                cnt + max(counts) * 0.02, i, str(cnt),
                va="center", fontsize=9,
            )
        axes[ax_i].set_title(label, fontsize=12, fontweight="bold")
        axes[ax_i].set_xlabel("Unique enzymes", fontsize=10)
        axes[ax_i].invert_yaxis()

    plt.suptitle(
        "TPS Type Distribution: Old vs New Dataset",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"type_distribution_comparison.{ext}")
    plt.close(fig)
    logger.info("Saved type_distribution_comparison")


# ── Phase 6: Summary JSON ───────────────────────────────────────────────


def phase6_save_results(df: pd.DataFrame, grouped: pd.DataFrame):
    """Save all detailed results."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(FIG_DIR / "per_fold_per_type_detailed.csv", index=False)
    logger.info("Saved per_fold_per_type_detailed.csv")

    grouped.to_csv(FIG_DIR / "per_type_summary.csv", index=False)
    logger.info("Saved per_type_summary.csv")

    summary = {}
    for track in ["A", "C", "B", "D", "E"]:
        tdata = grouped[grouped["track"] == track]
        track_summary = {}
        for model in tdata["model"].unique():
            mdata = tdata[tdata["model"] == model]
            track_summary[model] = {
                row["tps_type"]: {
                    "ap_mean": round(row["ap_mean"], 4),
                    "ap_sem": round(row["ap_sem"], 4),
                    "mean_n_pos": round(row["mean_n_pos"], 1),
                }
                for _, row in mdata.iterrows()
            }
        summary[track] = track_summary

    with open(FIG_DIR / "per_type_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved per_type_analysis_summary.json")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    all_counts = phase1_data_inventory()
    df, grouped = phase2_detailed_ap()
    phase3_method_deepdives(df)
    phase4_clean_ec_coverage()
    phase5_visualizations(df, grouped)
    phase6_save_results(df, grouped)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("Output directory: %s", FIG_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
