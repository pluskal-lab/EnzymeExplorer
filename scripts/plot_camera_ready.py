#!/usr/bin/env python3
"""Camera-ready figures for the local validation protocol.

Generates a complete set of publication-quality figures:

  Figure 1  – Aggregate performance grouped bar charts (mAP and AP, all models × 5 tracks)
  Figure 2  – Heatmap of all metrics × all tracks (substrate + TPS detection)
  Figure 3  – Atomic-change waterfall (A→C→D→E→B) for substrate and TPS detection
  Figure 4a – Per-similarity-bin line plots for substrate prediction (mAP)
  Figure 4b – Per-similarity-bin line plots for TPS detection (AP)
  Figure 5  – PlmDomainsRF vs PlmRF comparison
  Figure 6  – Side-by-side heatmaps: substrate prediction (mAP) & TPS detection (AP)
  Figure 7  – Major-substrate subset heatmaps (FPP, SqOx, GFPP): mAP & AP

Usage::

    python scripts/plot_camera_ready.py            # all defaults
    python scripts/plot_camera_ready.py --outdir outputs/figures
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
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EVAL_DIR = Path("outputs/evaluation_results")
_BIN_SEP = "_||_"

MODEL_DISPLAY = {
    "PlmRandomForest": "PlmRF",
    "PlmDomainsRandomForest": "PlmDomainsRF",
    "Blastp": "Blastp",
    "CLEANEcDetection": "CLEAN (in-sample)",
    "CLEAN": "CLEAN (retrained)",
    "HMM": "HMM",
    "Foldseek": "Foldseek",
}

PALETTE = {
    "PlmRF": "#2563eb",
    "PlmDomainsRF": "#0ea5e9",
    "CLEAN (in-sample)": "#dc2626",
    "CLEAN (retrained)": "#b91c1c",
    "Blastp": "#f97316",
    "HMM": "#16a34a",
    "Foldseek": "#8b5cf6",
}

MARKERS = {
    "PlmRF": "o",
    "PlmDomainsRF": "P",
    "CLEAN (in-sample)": "D",
    "CLEAN (retrained)": "d",
    "Blastp": "s",
    "HMM": "^",
    "Foldseek": "v",
}

MODEL_ORDER = [
    "PlmRF",
    "PlmDomainsRF",
    "Blastp",
    "HMM",
    "Foldseek",
]

TRACK_COLORS = {
    "A (phylo)": "#3b82f6",
    "C (synced)": "#22c55e",
    "D (cross)": "#f59e0b",
    "E (new TPS+old neg)": "#a855f7",
    "B (new)": "#ef4444",
}

TRACK_ORDER = ["A (phylo)", "C (synced)", "D (cross)", "E (new TPS+old neg)", "B (new)"]


def _short_label(raw: str) -> str:
    prefix = raw.split("__")[0]
    return MODEL_DISPLAY.get(prefix, prefix)


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, tuple):
        return data[0]
    return data


def _agg(
    data: dict[str, list[dict[str, float]]],
    target_class: str | None = None,
    bin_label: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Macro-mean +/- SEM across folds."""
    result: dict[str, tuple[float, float]] = {}
    for model, folds in data.items():
        per_class: dict[str, list[float]] = defaultdict(list)
        for fd in folds:
            for key, val in fd.items():
                k_bin, k_cls = (key.split(_BIN_SEP, 1) + [key])[:2]
                if _BIN_SEP in key:
                    k_bin, k_cls = key.split(_BIN_SEP, 1)
                else:
                    k_bin, k_cls = None, key
                if bin_label is not None and k_bin != bin_label:
                    continue
                if bin_label is None and k_bin is not None:
                    continue
                if target_class is not None and k_cls != target_class:
                    continue
                if not np.isnan(val):
                    per_class[k_cls].append(val)
        if not per_class:
            continue
        cls_means = [np.nanmean(v) for v in per_class.values()]
        macro = float(np.nanmean(cls_means))
        sem = float(
            np.nanstd(cls_means, ddof=1) / np.sqrt(len(cls_means))
            if len(cls_means) > 1
            else 0.0
        )
        result[model] = (macro, sem)
    return result


BINS = ["20-30", "30-40", "40-50", "50-60", "60-70"]


def _collect_aggregate_from_folds(
    n_folds: int = 5,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Compute mAP and AP directly from fold result pkl files.

    Fills in metrics for models that lack pre-computed evaluation CSVs.
    Returns {model_display: {track_label: {"mAP": (m,s), "AP": (m,s)}}}.
    """
    results: dict[str, dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for track_label, models in _TRACK_TO_MODELS.items():
        for model_type, version in models.items():
            display = MODEL_DISPLAY.get(model_type, model_type)
            fold_results = _load_fold_results(model_type, version, n_folds)
            if not fold_results:
                continue

            fold_maps: list[float] = []
            fold_aps: list[float] = []
            for fold_i, (val_proba, class_names, test_df) in fold_results:
                class_list = (
                    list(class_names)
                    if not isinstance(class_names, list)
                    else class_names
                )
                # Per-class AP for mAP
                per_class_aps: list[float] = []
                for ci, cname in enumerate(class_list):
                    if cname == "isTPS":
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
                    if y_true.sum() < 1 or y_true.sum() == len(y_true):
                        continue
                    ap = average_precision_score(y_true, val_proba[:, ci])
                    per_class_aps.append(ap)
                if per_class_aps:
                    fold_maps.append(float(np.mean(per_class_aps)))

                # Binary AP for isTPS
                if "isTPS" in class_list:
                    tps_idx = class_list.index("isTPS")
                    col = "SMILES_substrate_canonical_no_stereo"
                    if col in test_df.columns:
                        labels = test_df[col].values
                        y_true = np.array(
                            [
                                1
                                if (isinstance(s, set) and "isTPS" in s)
                                else 0
                                for s in labels
                            ]
                        )
                        if 0 < y_true.sum() < len(y_true):
                            ap = average_precision_score(
                                y_true, val_proba[:, tps_idx]
                            )
                            fold_aps.append(ap)

            if fold_maps:
                m = float(np.mean(fold_maps))
                s = (
                    float(np.std(fold_maps, ddof=1) / np.sqrt(len(fold_maps)))
                    if len(fold_maps) > 1
                    else 0.0
                )
                results[display][track_label]["mAP"] = (m, s)
            if fold_aps:
                m = float(np.mean(fold_aps))
                s = (
                    float(np.std(fold_aps, ddof=1) / np.sqrt(len(fold_aps)))
                    if len(fold_aps) > 1
                    else 0.0
                )
                results[display][track_label]["AP"] = (m, s)

    return dict(results)


def _collect_aggregate(
    substrate_csvs: dict[str, str],
    tps_csvs: dict[str, str],
    extra_substrate_csvs: dict[str, str] | None = None,
    extra_tps_csvs: dict[str, str] | None = None,
) -> dict:
    """Build { model_display: { track_label: { metric: (mean, sem) } } }.

    Reads from evaluation CSVs for exact consistency with the protocol doc.
    """
    records: dict[str, dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    def _ingest_csv(csv_map: dict[str, str], metric_col: str, out_key: str) -> None:
        for track_label, csv_base in csv_map.items():
            path = EVAL_DIR / f"{csv_base}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                model = _short_label(row["Model"])
                val = row[metric_col]
                sem_lo = row.get(f"{metric_col.split('(')[0].strip()} - SEM", np.nan)
                sem_hi = row.get(f"{metric_col.split('(')[0].strip()} + SEM", np.nan)
                sem = 0.0
                if not np.isnan(sem_lo) and not np.isnan(sem_hi) and not np.isnan(val):
                    sem = (sem_hi - sem_lo) / 2.0
                records[model][track_label][out_key] = (val, sem)

    _ingest_csv(substrate_csvs, "Mean Average Precision (mAP)", "mAP")
    _ingest_csv(tps_csvs, "Mean Average Precision (mAP)", "AP")
    if extra_substrate_csvs:
        _ingest_csv(extra_substrate_csvs, "Mean Average Precision (mAP)", "mAP")
    if extra_tps_csvs:
        _ingest_csv(extra_tps_csvs, "Mean Average Precision (mAP)", "AP")
    return records


def _collect_perbin(
    substrate_pkls: dict[str, str],
    tps_pkls: dict[str, str],
) -> dict:
    """Build { model_display: { track_label: { bin: { 'mAP'|'AP': val } } } }."""
    records: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for track_label, pkl_base in substrate_pkls.items():
        data = _load_pkl(EVAL_DIR / f"model_2_class_2_metric_vals_{pkl_base}.pkl")
        for b in BINS:
            agg = _agg(data, bin_label=b)
            for raw, (m, _) in agg.items():
                records[_short_label(raw)][track_label][b]["mAP"] = m

    for track_label, pkl_base in tps_pkls.items():
        data = _load_pkl(EVAL_DIR / f"model_2_class_2_metric_vals_{pkl_base}.pkl")
        for b in BINS:
            agg = _agg(data, target_class="isTPS", bin_label=b)
            for raw, (m, _) in agg.items():
                records[_short_label(raw)][track_label][b]["AP"] = m

    return records


def _collect_bin_counts(
    similarity_pkls: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Count observations per similarity bin per track.

    Returns { track_label: { bin: n_observations } }.
    """
    sim_files = {
        "A (phylo)": "data/mmseqs_similarities_track_a_phylo.pkl",
        "C (synced)": "data/mmseqs_similarities_track_c_synced.pkl",
        "B (new)": "data/mmseqs_similarities_track_b_new.pkl",
        "D (cross)": "data/mmseqs_similarities_track_d_cross.pkl",
        "E (new TPS+old neg)": "data/mmseqs_similarities_track_e_cross.pkl",
    }
    sim_files.update(similarity_pkls)

    counts: dict[str, dict[str, int]] = {}
    for track_label, sim_path in sim_files.items():
        path = Path(sim_path)
        if not path.exists():
            continue
        import pickle as pkl_mod

        with open(path, "rb") as fh:
            sims = pkl_mod.load(fh)

        bin_counts: dict[str, int] = defaultdict(int)
        seen = set()
        for _, fold_dict in sims.items():
            for seq_id, info in fold_dict.items():
                if seq_id in seen:
                    continue
                seen.add(seq_id)
                pident = info.get("pident", 0) if isinstance(info, dict) else info
                has_hit = info.get("has_hit", True) if isinstance(info, dict) else True
                if not has_hit:
                    bin_counts["no_hit"] += 1
                elif pident < 20:
                    bin_counts["0-20"] += 1
                elif pident < 30:
                    bin_counts["20-30"] += 1
                elif pident < 40:
                    bin_counts["30-40"] += 1
                elif pident < 50:
                    bin_counts["40-50"] += 1
                elif pident < 60:
                    bin_counts["50-60"] += 1
                elif pident < 70:
                    bin_counts["60-70"] += 1
                elif pident < 80:
                    bin_counts["70-80"] += 1
                else:
                    bin_counts["80-100"] += 1
        counts[track_label] = dict(bin_counts)
    return counts


OUTPUT_ROOT = Path("outputs")

_TRACK_TO_MODELS: dict[str, dict[str, str]] = {
    "A (phylo)": {
        "Blastp": "with_minor_reactions_phylo_folds",
        "Foldseek": "with_minor_reactions_phylo_folds",
        "HMM": "with_minor_reactions_phylo_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_with_minor_reactions_global_tuning_domains_subset_phylo_folds",
    },
    "C (synced)": {
        "Blastp": "synced_folds",
        "Foldseek": "synced_folds",
        "HMM": "synced_folds",
        "PlmRandomForest": "tps_esm-1v-subseq_synced_folds",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_synced_folds",
    },
    "B (new)": {
        "Blastp": "new_dataset",
        "Foldseek": "new_dataset",
        "HMM": "new_dataset",
        "PlmRandomForest": "tps_esm-1v-subseq_new_dataset",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_new_dataset",
    },
    "D (cross)": {
        "Blastp": "cross_synced_to_new",
        "Foldseek": "cross_synced_to_new",
        "HMM": "cross_synced_to_new",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_synced_to_new",
    },
    "E (new TPS+old neg)": {
        "Blastp": "cross_new_tps_old_neg",
        "Foldseek": "cross_new_tps_old_neg",
        "HMM": "cross_new_tps_old_neg",
        "PlmRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
        "PlmDomainsRandomForest": "tps_esm-1v-subseq_cross_new_tps_old_neg",
    },
}

def _load_fold_results(model_type: str, version: str, n_folds: int = 5) -> list[tuple]:
    """Load fold result pkl files for a model/version."""
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


def _assign_pident_bin(pident: float) -> str | None:
    """Assign a similarity pident to one of the standard BINS."""
    for lo_str, hi_str in (b.split("-") for b in BINS):
        lo, hi = int(lo_str), int(hi_str)
        if lo <= pident < hi:
            return f"{lo}-{hi}"
    return None


# ───────────────────────────────────────────────────────────────
# Figure 1: Grouped bar charts
# ───────────────────────────────────────────────────────────────


def fig1_grouped_bars(agg: dict, outdir: Path) -> None:
    track_order = TRACK_ORDER
    for metric, ylabel, title_suffix in [
        ("mAP", "Mean Average Precision (mAP)", "Substrate Prediction"),
        ("AP", "Average Precision (AP)", "TPS Detection"),
    ]:
        models = [m for m in MODEL_ORDER if m in agg]
        tracks_present = [
            t
            for t in track_order
            if any(t in agg[m] and metric in agg[m].get(t, {}) for m in models)
        ]

        n_m = len(models)
        n_t = len(tracks_present)
        bar_w = 0.8 / n_t
        x = np.arange(n_m)

        fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_m), 5.5))

        for ti, t in enumerate(tracks_present):
            means, sems = [], []
            for m in models:
                v = agg[m].get(t, {}).get(metric)
                means.append(v[0] if v else np.nan)
                sems.append(v[1] if v else 0.0)
            offset = (ti - (n_t - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset,
                means,
                bar_w * 0.88,
                yerr=sems,
                label=t,
                color=TRACK_COLORS.get(t, f"C{ti}"),
                edgecolor="white",
                linewidth=0.5,
                capsize=3,
                zorder=3,
            )
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f".{int(val*1000):03d}" if val < 1 else "1.00",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight="medium")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.legend(
            fontsize=9,
            title="Track",
            title_fontsize=10,
            loc="upper right",
            framealpha=0.9,
        )
        ax.set_title(
            f"Cross-Track Comparison — {title_suffix}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        stem = f"fig1_grouped_bars_{metric.lower()}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(outdir / f"{stem}.{ext}")
        plt.close(fig)
        logger.info("Saved %s", stem)


# ───────────────────────────────────────────────────────────────
# Figure 2: Heatmap
# ───────────────────────────────────────────────────────────────


def fig2_heatmap(agg: dict, outdir: Path) -> None:
    track_order = TRACK_ORDER
    metrics = ["mAP", "AP"]

    models = [m for m in MODEL_ORDER if m in agg]
    col_labels = []
    for t in track_order:
        for met in metrics:
            col_labels.append(f"{t}\n{met}")

    matrix = np.full((len(models), len(col_labels)), np.nan)
    for mi, m in enumerate(models):
        ci = 0
        for t in track_order:
            for met in metrics:
                v = agg[m].get(t, {}).get(met)
                if v is not None:
                    matrix[mi, ci] = v[0]
                ci += 1

    fig, ax = plt.subplots(
        figsize=(2.0 * len(col_labels) + 1.5, 0.65 * len(models) + 1.8)
    )
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(len(models)):
        for j in range(len(col_labels)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                tc = "white" if v > 0.65 else "black"
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=tc,
                    fontweight="bold",
                )

    for ti in range(1, len(track_order)):
        ax.axvline(ti * len(metrics) - 0.5, color="gray", linewidth=1.2)

    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="Score")
    ax.set_title(
        "All Models × All Tracks — mAP (Substrate) and AP (TPS Detection)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"fig2_heatmap.{ext}")
    plt.close(fig)
    logger.info("Saved fig2_heatmap")


# ───────────────────────────────────────────────────────────────
# Figure 3: Atomic-change waterfall A→C→D→B
# ───────────────────────────────────────────────────────────────


def fig3_waterfall(agg: dict, outdir: Path) -> None:
    """Cumulative bridge chart: A (in-distrib) → decomposed recovery → B.

    Bars:
      1. A  (reference, full-height)  — in-distribution baseline
      2. +curation  (A→C delta)       — better TPS labels/folds
      3. cross-gap  (C→D delta)       — drop from cross-dataset eval
      4. +more TPS  (D→E delta)       — adding 339 new TPS enzymes
      5. +better neg (E→B delta)      — homology-leakage-free negatives
      6. B  (reference, full-height)  — final new-dataset performance

    The intermediate bars are floating: each starts where the previous ended.
    """
    delta_steps = [
        ("+curation\n(A→C)", "A (phylo)", "C (synced)"),
        ("cross-dataset\ngap (C→D)", "C (synced)", "D (cross)"),
        ("+more TPS\n(D→E)", "D (cross)", "E (new TPS+old neg)"),
        ("+better neg\n(E→B)", "E (new TPS+old neg)", "B (new)"),
    ]
    ref_tracks = [("A (phylo)", "A\n(baseline)"), ("B (new)", "B\n(retrained)")]

    for metric, metric_label, title_suffix in [
        ("mAP", "mAP", "Substrate Prediction"),
        ("AP", "AP", "TPS Detection"),
    ]:
        models = [
            m
            for m in MODEL_ORDER
            if m in agg
            and metric in agg[m].get("A (phylo)", {})
            and metric in agg[m].get("B (new)", {})
        ]
        if not models:
            continue

        n_models = len(models)
        n_bars = len(delta_steps) + 2
        x = np.arange(n_bars)
        bar_w = 0.82 / n_models

        fig, ax = plt.subplots(figsize=(max(12, 2.0 * n_bars), 5.5))

        for mi, m in enumerate(models):
            offset = (mi - (n_models - 1) / 2) * bar_w
            color = PALETTE.get(m, f"C{mi}")

            v_a = agg[m].get("A (phylo)", {}).get(metric)
            v_b = agg[m].get("B (new)", {}).get(metric)
            if not v_a or not v_b:
                continue
            a_val = v_a[0]
            b_val = v_b[0]

            deltas = []
            for _, t_from, t_to in delta_steps:
                vf = agg[m].get(t_from, {}).get(metric)
                vt = agg[m].get(t_to, {}).get(metric)
                if vf and vt:
                    deltas.append(vt[0] - vf[0])
                else:
                    deltas.append(0.0)

            bottoms = [0.0] * n_bars
            heights = [0.0] * n_bars

            heights[0] = a_val
            bottoms[0] = 0.0

            running = a_val
            for si, d in enumerate(deltas):
                idx = si + 1
                if d >= 0:
                    bottoms[idx] = running
                    heights[idx] = d
                else:
                    bottoms[idx] = running + d
                    heights[idx] = -d
                running += d

            heights[-1] = b_val
            bottoms[-1] = 0.0

            bar_colors = [color] * n_bars
            alphas = [1.0] * n_bars
            for si, d in enumerate(deltas):
                idx = si + 1
                alphas[idx] = 0.65 if d >= 0 else 0.4

            for bi in range(n_bars):
                edge = "white" if bi in (0, n_bars - 1) else "#555555"
                hatch = None
                if 1 <= bi <= len(deltas) and deltas[bi - 1] < 0:
                    hatch = "///"
                ax.bar(
                    x[bi] + offset,
                    heights[bi],
                    bar_w * 0.88,
                    bottom=bottoms[bi],
                    color=bar_colors[bi],
                    alpha=alphas[bi],
                    edgecolor=edge,
                    linewidth=0.6,
                    hatch=hatch,
                    label=m if bi == 0 else None,
                    zorder=3,
                )

            for bi in range(n_bars):
                val = heights[bi] if bi in (0, n_bars - 1) else deltas[bi - 1]
                top = bottoms[bi] + heights[bi]
                if bi in (0, n_bars - 1):
                    txt = f"{val:.3f}"
                    va, y_pos = "bottom", top + 0.008
                else:
                    txt = f"{val:+.3f}"
                    va = "bottom" if val >= 0 else "top"
                    y_pos = top + 0.008 if val >= 0 else bottoms[bi] - 0.008
                ax.text(
                    x[bi] + offset,
                    y_pos,
                    txt,
                    ha="center",
                    va=va,
                    fontsize=6.5,
                    fontweight="bold" if bi in (0, n_bars - 1) else "normal",
                    rotation=90,
                )

            if n_models == 1 or mi == 0:
                for si in range(len(deltas)):
                    idx = si + 1
                    y_conn = bottoms[idx] + heights[idx] if deltas[si] >= 0 else bottoms[idx]
                    x_start = x[idx - 1] + offset + bar_w * 0.44
                    x_end = x[idx] + offset - bar_w * 0.44
                    if abs(x_end - x_start) > 0.01:
                        ax.plot(
                            [x_start, x_end],
                            [y_conn, y_conn],
                            color="gray",
                            linewidth=0.5,
                            linestyle=":",
                            zorder=2,
                            alpha=0.5,
                        )

        bar_labels = [ref_tracks[0][1]]
        bar_labels += [s[0] for s in delta_steps]
        bar_labels += [ref_tracks[1][1]]

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9, fontweight="medium")
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.set_title(
            f"Decomposition of Performance Recovery — {title_suffix}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

        plt.tight_layout()
        stem = f"fig3_waterfall_{metric.lower()}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(outdir / f"{stem}.{ext}")
        plt.close(fig)
        logger.info("Saved %s", stem)


# ───────────────────────────────────────────────────────────────
# Figures 4a/4b: Per-similarity-bin faceted line plots
# ───────────────────────────────────────────────────────────────


def fig4_perbin(
    perbin: dict,
    outdir: Path,
    bin_counts: dict | None = None,
) -> None:
    track_order = TRACK_ORDER

    plot_specs = [
        ("a", "mAP", "mAP (substrate)", "Substrate Prediction"),
        ("b", "AP", "AP (TPS detection)", "TPS Detection"),
    ]

    for fig_letter, metric, ylabel, title_suffix in plot_specs:
        source = perbin

        def _get_bin_val(src: dict, m: str, t: str, b: str, met: str):
            return src.get(m, {}).get(t, {}).get(b, {}).get(met)

        models = [
            m
            for m in MODEL_ORDER
            if m in source
            and any(
                _get_bin_val(source, m, t, b, metric) is not None
                for t in track_order
                for b in BINS
            )
        ]
        tracks_present = [
            t
            for t in track_order
            if any(
                _get_bin_val(source, m, t, b, metric) is not None
                for m in models
                for b in BINS
            )
        ]
        if not models or not tracks_present:
            continue

        n_t = len(tracks_present)
        fig, axes = plt.subplots(
            1, n_t, figsize=(5 * n_t, 4.5), sharey=True, squeeze=False
        )

        for ci, t in enumerate(tracks_present):
            ax = axes[0, ci]
            for m in models:
                xs, ys = [], []
                for bi, b in enumerate(BINS):
                    v = _get_bin_val(source, m, t, b, metric)
                    if v is not None:
                        xs.append(bi)
                        ys.append(v)
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        marker=MARKERS.get(m, "o"),
                        color=PALETTE.get(m, "gray"),
                        label=m if ci == 0 else None,
                        linewidth=2,
                        markersize=6,
                        alpha=0.85,
                    )

            bin_labels = []
            for b in BINS:
                label = b.replace("-", "\u2013") + "%"
                if bin_counts and t in bin_counts:
                    n = bin_counts[t].get(b, 0)
                    if n > 0:
                        label += f"\nn={n}"
                bin_labels.append(label)

            ax.set_xticks(range(len(BINS)))
            ax.set_xticklabels(bin_labels, fontsize=8, rotation=25, ha="right")
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.grid(alpha=0.2)
            ax.set_title(t, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel("Seq. identity to nearest training hit", fontsize=9)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(models), 6),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, 1.03),
        )
        fig.suptitle(
            f"Per-Similarity-Bin — {title_suffix}",
            fontsize=14,
            fontweight="bold",
            y=1.08,
        )
        plt.tight_layout()
        stem = f"fig4_perbin_{metric.lower()}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(outdir / f"{stem}.{ext}")
        plt.close(fig)
        logger.info("Saved %s", stem)


# ───────────────────────────────────────────────────────────────
# Figure 5: PlmDomainsRF vs PlmRF comparison
# ───────────────────────────────────────────────────────────────


def fig5_domain_comparison(agg: dict, outdir: Path) -> None:
    if "PlmDomainsRF" not in agg:
        logger.info("Skipping fig5 — PlmDomainsRF not in data")
        return

    both_tracks = set(agg["PlmRF"].keys()) & set(agg["PlmDomainsRF"].keys())
    tracks_present = [t for t in TRACK_ORDER if t in both_tracks]
    if not tracks_present:
        return

    metrics = ["mAP", "AP"]
    n_groups = len(tracks_present) * len(metrics)
    x = np.arange(n_groups)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_groups), 5))
    labels = []
    plmrf_vals = []
    plmdom_vals = []

    for t in tracks_present:
        for met in metrics:
            labels.append(f"{t}\n{met}")
            vr = agg["PlmRF"].get(t, {}).get(met)
            vd = agg["PlmDomainsRF"].get(t, {}).get(met)
            plmrf_vals.append(vr[0] if vr else np.nan)
            plmdom_vals.append(vd[0] if vd else np.nan)

    ax.bar(
        x - bar_w / 2,
        plmrf_vals,
        bar_w * 0.9,
        label="PlmRF",
        color=PALETTE["PlmRF"],
        edgecolor="white",
        zorder=3,
    )
    ax.bar(
        x + bar_w / 2,
        plmdom_vals,
        bar_w * 0.9,
        label="PlmDomainsRF",
        color=PALETTE["PlmDomainsRF"],
        edgecolor="white",
        zorder=3,
    )

    for xi, (vr, vd) in enumerate(zip(plmrf_vals, plmdom_vals)):
        for v, off in [(vr, -bar_w / 2), (vd, bar_w / 2)]:
            if not np.isnan(v):
                ax.text(
                    xi + off,
                    v + 0.008,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title(
        "Impact of Domain Features: PlmRF vs PlmDomainsRF",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"fig5_domain_comparison.{ext}")
    plt.close(fig)
    logger.info("Saved fig5_domain_comparison")


# ───────────────────────────────────────────────────────────────
# Figure 6: Combined overview – side-by-side mAP & AP heatmaps
# ───────────────────────────────────────────────────────────────


_HEATMAP_EXCLUDE = {"CLEAN (in-sample)", "CLEAN (retrained)"}


def _single_heatmap(
    agg: dict,
    metric_key: str,
    title: str,
    stem: str,
    outdir: Path,
) -> None:
    """Render one heatmap (models × tracks) for a single metric."""
    track_order = TRACK_ORDER
    models = [
        m for m in MODEL_ORDER if m in agg and m not in _HEATMAP_EXCLUDE
    ]
    n_tracks = len(track_order)

    fig, ax = plt.subplots(
        figsize=(2.2 * n_tracks + 2.0, 0.65 * len(models) + 1.8),
    )

    matrix = np.full((len(models), n_tracks), np.nan)
    for mi, m in enumerate(models):
        for ti, t in enumerate(track_order):
            v = agg[m].get(t, {}).get(metric_key)
            if v:
                matrix[mi, ti] = v[0]

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_tracks))
    ax.set_xticklabels(track_order, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(n_tracks):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(
                    j, i, "\u2014", ha="center", va="center",
                    fontsize=8, color="gray",
                )
            else:
                tc = "white" if v > 0.65 else "black"
                ax.text(
                    j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold",
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03, label="Score")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)
    logger.info("Saved %s", stem)


def fig6_combined_heatmap(
    agg: dict,
    outdir: Path,
) -> None:
    """Separate heatmaps: substrate prediction (mAP) and TPS detection (AP)."""
    _single_heatmap(
        agg, "mAP", "Substrate Prediction (mAP)", "fig6a_heatmap_map", outdir,
    )
    _single_heatmap(
        agg, "AP", "TPS Detection (AP)", "fig6b_heatmap_ap", outdir,
    )


# ───────────────────────────────────────────────────────────────
# Figure 7: Major-substrate subset heatmap (mAP + AP)
# ───────────────────────────────────────────────────────────────

MAJOR_SUBSTRATES: set[str] = {
    # FPP – sesquiterpene
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    # GPP – monoterpene
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    # GGPP – diterpene
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    # Squalene oxide – triterpene
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
    # CPP – diterpene (type II cyclase)
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    # GFPP – sesterterpene
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
}


def _collect_major_substrate_agg(
    n_folds: int = 5,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Compute mAP restricted to MAJOR_SUBSTRATES from fold pkl files."""
    results: dict[str, dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    col = "SMILES_substrate_canonical_no_stereo"
    for track_label, models in _TRACK_TO_MODELS.items():
        for model_type, version in models.items():
            display = MODEL_DISPLAY.get(model_type, model_type)
            fold_results = _load_fold_results(model_type, version, n_folds)
            if not fold_results:
                continue
            fold_maps: list[float] = []
            fold_aps: list[float] = []
            for fold_i, (val_proba, class_names, test_df) in fold_results:
                class_list = (
                    list(class_names)
                    if not isinstance(class_names, list)
                    else class_names
                )
                if col not in test_df.columns:
                    continue
                labels = test_df[col].values

                per_class_aps: list[float] = []
                for ci, cname in enumerate(class_list):
                    if cname == "isTPS" or cname not in MAJOR_SUBSTRATES:
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

                if "isTPS" in class_list:
                    tps_idx = class_list.index("isTPS")
                    y_true = np.array(
                        [
                            1
                            if (isinstance(s, set) and "isTPS" in s)
                            else 0
                            for s in labels
                        ]
                    )
                    if 0 < y_true.sum() < len(y_true):
                        ap = average_precision_score(
                            y_true, val_proba[:, tps_idx]
                        )
                        fold_aps.append(ap)

            if fold_maps:
                m = float(np.mean(fold_maps))
                s = (
                    float(np.std(fold_maps, ddof=1) / np.sqrt(len(fold_maps)))
                    if len(fold_maps) > 1
                    else 0.0
                )
                results[display][track_label]["mAP"] = (m, s)
            if fold_aps:
                m = float(np.mean(fold_aps))
                s = (
                    float(np.std(fold_aps, ddof=1) / np.sqrt(len(fold_aps)))
                    if len(fold_aps) > 1
                    else 0.0
                )
                results[display][track_label]["AP"] = (m, s)
    return dict(results)


def fig7_major_substrate_heatmap(
    major_agg: dict,
    outdir: Path,
) -> None:
    """Separate heatmaps for major-substrate mAP and TPS detection AP."""
    _single_heatmap(
        major_agg,
        "mAP",
        "Substrate Prediction \u2014 Major Substrates (mAP)",
        "fig7a_major_heatmap_map",
        outdir,
    )
    _single_heatmap(
        major_agg,
        "AP",
        "TPS Detection \u2014 Major Substrates (AP)",
        "fig7b_major_heatmap_ap",
        outdir,
    )


# ───────────────────────────────────────────────────────────────
# CLI and main
# ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for output figures (default: outputs/figures)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    substrate_csvs = {
        "A (phylo)": "track_a_phylo_folds",
        "C (synced)": "track_c_synced_folds",
        "D (cross)": "track_d_cross_synced",
        "E (new TPS+old neg)": "track_e_results",
        "B (new)": "track_b_new_dataset",
    }
    tps_csvs = {
        "A (phylo)": "track_a_tps_detection",
        "C (synced)": "track_c_tps_detection",
        "D (cross)": "track_d_tps_detection",
        "E (new TPS+old neg)": "track_e_tps_detection",
        "B (new)": "track_b_tps_detection",
    }

    substrate_perbin_pkls = {
        "A (phylo)": "track_a_phylo_folds",
        "C (synced)": "track_c_synced_folds",
        "D (cross)": "track_d_simbins",
        "E (new TPS+old neg)": "track_e_simbins",
        "B (new)": "track_b_new_dataset",
    }
    tps_perbin_pkls = {
        "A (phylo)": "all_results_track_a_simbins",
        "C (synced)": "track_c_tps_simbins",
        "D (cross)": "track_d_tps_simbins",
        "E (new TPS+old neg)": "track_e_tps_simbins",
        "B (new)": "track_b_tps_simbins",
    }

    extra_substrate_csvs: list[dict[str, str]] = [
        {
            "A (phylo)": "track_a_phylo_folds_with_plmdom",
            "C (synced)": "track_c_synced_folds_with_plmdom",
            "B (new)": "track_b_plmdomainsrf_new_dataset",
            "D (cross)": "track_d_plmdomrf",
        },
        {"E (new TPS+old neg)": "track_e_foldseek"},
    ]
    extra_tps_csvs: list[dict[str, str]] = [
        {
            "A (phylo)": "track_a_tps_detection_with_plmdom",
            "C (synced)": "track_c_tps_detection_with_plmdom",
            "B (new)": "track_b_plmdomainsrf_tps_detection",
            "D (cross)": "track_d_plmdomrf_tps",
        },
        {"E (new TPS+old neg)": "track_e_foldseek_tps"},
    ]

    missing = []
    for label, base in {**substrate_csvs, **tps_csvs}.items():
        path = EVAL_DIR / f"{base}.csv"
        if not path.exists():
            missing.append(str(path))
    if missing:
        logger.warning("Missing CSVs (some figures may be incomplete):")
        for p in missing:
            logger.warning("  %s", p)

    merged_extra_sub: dict[str, str] = {}
    for d in extra_substrate_csvs:
        merged_extra_sub.update(d)
    merged_extra_tps: dict[str, str] = {}
    for d in extra_tps_csvs:
        merged_extra_tps.update(d)

    agg = _collect_aggregate(
        substrate_csvs, tps_csvs, merged_extra_sub, merged_extra_tps
    )

    fold_agg = _collect_aggregate_from_folds()
    for model, tracks in fold_agg.items():
        for track, metrics in tracks.items():
            for metric, val in metrics.items():
                agg.setdefault(model, {}).setdefault(track, {})[metric] = val

    perbin = _collect_perbin(substrate_perbin_pkls, tps_perbin_pkls)
    bin_counts = _collect_bin_counts({})

    logger.info(
        "Loaded data for %d models across %d tracks",
        len(agg),
        len(set().union(*(m.keys() for m in agg.values()))),
    )

    fig1_grouped_bars(agg, args.outdir)
    fig2_heatmap(agg, args.outdir)
    fig3_waterfall(agg, args.outdir)
    fig4_perbin(perbin, args.outdir, bin_counts=bin_counts)
    fig5_domain_comparison(agg, args.outdir)
    fig6_combined_heatmap(agg, args.outdir)

    major_agg = _collect_major_substrate_agg()
    fig7_major_substrate_heatmap(major_agg, args.outdir)

    logger.info("All figures saved to %s", args.outdir)


if __name__ == "__main__":
    main()
