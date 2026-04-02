#!/usr/bin/env python3
"""Generate cross-track comparison visualizations.

Produces two families of figures:

1. **Big-picture overview** (``cross_track_overview_*.{png,pdf}``):
   Grouped bar charts with one bar-group per model, one bar per track.
   Separate figures for substrate prediction (mAP) and TPS detection (AP).

2. **Per-similarity-bin degradation** (``simbin_degradation_*.{png,pdf}``):
   Faceted line plots showing how each metric (AP, MCC-F1) changes across
   sequence-identity bins, with one panel column per track and one line
   per model.

Usage::

    python scripts/plot_cross_track_results.py \\
        --tracks track_a_phylo_folds track_b_new_dataset \\
                 track_c_synced_folds \\
        --track-labels "Track A" "Track B" "Track C" \\
        --models PlmRandomForest__tps_esm-1v-subseq_with_minor_reactions_phylo_folds \\
                 PlmRandomForest__tps_esm-1v-subseq_new_dataset \\
                 PlmRandomForest__tps_esm-1v-subseq_synced_folds \\
        --model-labels PlmRF PlmRF PlmRF

    # Or with Track D added:
    python scripts/plot_cross_track_results.py \\
        --tracks track_a_phylo_folds track_c_synced_folds \\
                 track_d_cross_synced track_b_new_dataset \\
        --track-labels "A (phylo)" "C (synced)" \\
                       "D (cross)" "B (new)" \\
        ...
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EVAL_DIR = Path("outputs/evaluation_results")

_BIN_SEPARATOR = "_||_"
_CAT_SEPARATOR = "_|_"

MODEL_COLORS: dict[str, str] = {
    "PlmRF": "#1f77b4",
    "Blastp": "#ff7f0e",
    "HMM": "#2ca02c",
    "CLEAN (in-sample)": "#d62728",
    "Foldseek": "#9467bd",
}

MODEL_MARKERS: dict[str, str] = {
    "PlmRF": "o",
    "Blastp": "s",
    "HMM": "^",
    "CLEAN (in-sample)": "D",
    "Foldseek": "v",
}


def _load_metric_pickles(
    track_name: str,
) -> tuple[dict, dict, dict]:
    """Load the three metric dicts for a track from its pickle."""
    pkl_path = EVAL_DIR / f"model_2_class_2_metric_vals_{track_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing evaluation pickle: {pkl_path}")
    with open(pkl_path, "rb") as fh:
        ap_vals, rocauc_vals, mccf1_vals = pickle.load(fh)
    return ap_vals, rocauc_vals, mccf1_vals


def _aggregate_metric(
    model_2_class_2_vals: dict[str, list[dict[str, float]]],
    target_class: str | None = None,
    bin_label: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute macro-mean +/- SEM for each model.

    If *target_class* is given, only that class is used (for TPS
    detection with ``target_class="isTPS"``).  Otherwise macro-average
    across all non-binned classes.

    If *bin_label* is given (e.g. ``"20-30"``), restrict to keys
    matching that bin prefix.
    """
    result: dict[str, tuple[float, float]] = {}
    for model, fold_dicts in model_2_class_2_vals.items():
        per_class_means: dict[str, list[float]] = defaultdict(list)
        for fold_dict in fold_dicts:
            for key, val in fold_dict.items():
                raw_key = key
                key_bin = None
                if _BIN_SEPARATOR in raw_key:
                    key_bin, raw_key = raw_key.split(_BIN_SEPARATOR, 1)
                if _CAT_SEPARATOR in raw_key:
                    _, raw_key = raw_key.split(_CAT_SEPARATOR, 1)

                if bin_label is not None and key_bin != bin_label:
                    continue
                if bin_label is None and key_bin is not None:
                    continue

                if target_class is not None and raw_key != target_class:
                    continue

                per_class_means[raw_key].append(val)

        if not per_class_means:
            continue

        class_means = [np.nanmean(v) for v in per_class_means.values()]
        macro = float(np.nanmean(class_means))
        sem = float(np.nanstd(class_means, ddof=1) / np.sqrt(len(class_means)))
        result[model] = (macro, sem)
    return result


def _canonical_model_label(raw: str) -> str:
    """Map raw model key ``ModelType__version`` to a short display label."""
    raw_lower = raw.lower()
    if "plmrandomforest" in raw_lower or "plmrf" in raw_lower:
        return "PlmRF"
    if "blastp" in raw_lower:
        return "Blastp"
    if "clean" in raw_lower:
        return "CLEAN (in-sample)"
    if "hmm" in raw_lower:
        return "HMM"
    if "foldseek" in raw_lower:
        return "Foldseek"
    return raw.split("__")[0]


def _discover_bins(
    model_2_class_2_vals: dict[str, list[dict[str, float]]],
) -> list[str]:
    """Return sorted list of similarity-bin labels present in the data."""
    bins: set[str] = set()
    for fold_dicts in model_2_class_2_vals.values():
        for fd in fold_dicts:
            for key in fd:
                if _BIN_SEPARATOR in key:
                    bl = key.split(_BIN_SEPARATOR, 1)[0]
                    bins.add(bl)
    order = {"no_hit": -1}
    return sorted(bins, key=lambda b: order.get(b, int(b.split("-")[0])))


# ── Figure 1: Big-picture grouped bar chart ──────────────────────


def plot_cross_track_overview(
    tracks_data: list[tuple[str, dict, dict, dict]],
    task: str,
    output_stem: str,
) -> None:
    """Grouped bar chart: one group per model, one bar per track.

    *task* is ``"substrate"`` or ``"tps_detection"``.
    """
    target_class = "isTPS" if task == "tps_detection" else None
    metric_label = (
        "AP (TPS detection)" if task == "tps_detection" else "mAP (substrate)"
    )

    all_models_ordered: list[str] = []
    track_labels: list[str] = []
    model_track_values: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

    for track_label, ap_vals, _, _ in tracks_data:
        track_labels.append(track_label)
        agg = _aggregate_metric(ap_vals, target_class=target_class)
        for raw_model, (mean, sem) in agg.items():
            label = _canonical_model_label(raw_model)
            model_track_values[label][track_label] = (mean, sem)
            if label not in all_models_ordered:
                all_models_ordered.append(label)

    preferred_order = ["PlmRF", "Blastp", "HMM", "CLEAN (in-sample)", "Foldseek"]
    models = [m for m in preferred_order if m in all_models_ordered]
    models += [m for m in all_models_ordered if m not in models]

    n_models = len(models)
    n_tracks = len(track_labels)
    bar_width = 0.8 / n_tracks
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(2.5 + 1.8 * n_models, 5))
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("Set2")
    track_colors = [cmap(i / max(n_tracks - 1, 1)) for i in range(n_tracks)]

    for t_i, track_label in enumerate(track_labels):
        means, sems = [], []
        for m in models:
            val = model_track_values[m].get(track_label, (np.nan, 0.0))
            means.append(val[0])
            sems.append(val[1])
        offset = (t_i - (n_tracks - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            yerr=sems,
            label=track_label,
            color=track_colors[t_i],
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, title="Track", title_fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title(f"Cross-Track Comparison — {metric_label}", fontsize=14, pad=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(EVAL_DIR / f"{output_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_stem)


# ── Figure 2: Per-similarity-bin degradation (faceted line plots) ─


def plot_simbin_degradation(
    tracks_data: list[tuple[str, dict, dict, dict]],
    task: str,
    output_stem: str,
) -> None:
    """Faceted line plots: columns = tracks, rows = metrics (AP, MCC-F1).

    One line per model; x-axis = similarity bins.
    """
    target_class = "isTPS" if task == "tps_detection" else None
    n_tracks = len(tracks_data)

    all_bins: list[str] = []
    for _, ap_vals, _, _ in tracks_data:
        b = _discover_bins(ap_vals)
        if len(b) > len(all_bins):
            all_bins = b
    if not all_bins:
        logger.warning("No similarity bins found; skipping %s", output_stem)
        return

    all_models: list[str] = []
    for _, ap_vals, _, _ in tracks_data:
        for raw in ap_vals:
            label = _canonical_model_label(raw)
            if label not in all_models:
                all_models.append(label)
    preferred_order = ["PlmRF", "Blastp", "HMM", "CLEAN (in-sample)", "Foldseek"]
    models = [m for m in preferred_order if m in all_models]
    models += [m for m in all_models if m not in models]

    metric_rows = [
        ("AP", lambda td: td[1]),
        ("MCC-F1", lambda td: td[3]),
    ]

    fig, axes = plt.subplots(
        len(metric_rows),
        n_tracks,
        figsize=(4.5 * n_tracks, 3.5 * len(metric_rows)),
        sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_i, (metric_name, extractor) in enumerate(metric_rows):
        for col_i, td in enumerate(tracks_data):
            ax = axes[row_i, col_i]
            track_label = td[0]
            metric_dict = extractor(td)

            for model_label in models:
                raw_keys = [
                    k for k in metric_dict if _canonical_model_label(k) == model_label
                ]
                if not raw_keys:
                    continue
                raw_key = raw_keys[0]

                ys = []
                xs = []
                for b_i, bl in enumerate(all_bins):
                    agg = _aggregate_metric(
                        {raw_key: metric_dict[raw_key]},
                        target_class=target_class,
                        bin_label=bl,
                    )
                    if raw_key in agg:
                        xs.append(b_i)
                        ys.append(agg[raw_key][0])

                color = MODEL_COLORS.get(model_label, None)
                marker = MODEL_MARKERS.get(model_label, "o")
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    label=model_label if col_i == 0 else None,
                    color=color,
                    linewidth=1.8,
                    markersize=5,
                    alpha=0.85,
                )

            ax.set_xticks(range(len(all_bins)))
            ax.set_xticklabels(
                [b.replace("-", "–") + "%" for b in all_bins],
                fontsize=8,
                rotation=30,
            )
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            if row_i == 0:
                ax.set_title(track_label, fontsize=12, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(metric_name, fontsize=12)
            if row_i == len(metric_rows) - 1:
                ax.set_xlabel("Sequence identity to nearest training hit", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(models),
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
        )

    task_title = "TPS Detection" if task == "tps_detection" else "Substrate Prediction"
    fig.suptitle(
        f"Performance by Sequence-Identity Bin — {task_title}",
        fontsize=14,
        y=1.06,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            EVAL_DIR / f"{output_stem}.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Saved %s", output_stem)


# ── Figure 3: Heatmap of aggregate metrics (models × tracks) ─────


def plot_metric_heatmap(
    tracks_data: list[tuple[str, dict, dict, dict]],
    task: str,
    output_stem: str,
) -> None:
    """Compact heatmap: rows = models, columns = tracks × metrics."""
    target_class = "isTPS" if task == "tps_detection" else None
    metric_names = ["AP" if task == "tps_detection" else "mAP", "ROC-AUC", "MCC-F1"]
    extractors = [
        lambda td: td[1],
        lambda td: td[2],
        lambda td: td[3],
    ]

    all_models: list[str] = []
    for _, ap_vals, _, _ in tracks_data:
        for raw in ap_vals:
            label = _canonical_model_label(raw)
            if label not in all_models:
                all_models.append(label)
    preferred_order = ["PlmRF", "Blastp", "HMM", "CLEAN (in-sample)", "Foldseek"]
    models = [m for m in preferred_order if m in all_models]
    models += [m for m in all_models if m not in models]

    col_labels = []
    matrix_rows: dict[str, list[float]] = {m: [] for m in models}

    for td in tracks_data:
        track_label = td[0]
        for metric_name, extractor in zip(metric_names, extractors):
            col_labels.append(f"{track_label}\n{metric_name}")
            metric_dict = extractor(td)
            for model_label in models:
                raw_keys = [
                    k for k in metric_dict if _canonical_model_label(k) == model_label
                ]
                if raw_keys:
                    agg = _aggregate_metric(
                        {raw_keys[0]: metric_dict[raw_keys[0]]},
                        target_class=target_class,
                    )
                    val = agg[raw_keys[0]][0] if raw_keys[0] in agg else np.nan
                else:
                    val = np.nan
                matrix_rows[model_label].append(val)

    data = np.array([matrix_rows[m] for m in models])

    fig, ax = plt.subplots(figsize=(2.2 * len(col_labels), 0.7 * len(models) + 1.5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.75 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")

    n_metrics = len(metric_names)
    for t_i in range(1, len(tracks_data)):
        ax.axvline(t_i * n_metrics - 0.5, color="gray", linewidth=1.5)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
    task_title = "TPS Detection" if task == "tps_detection" else "Substrate Prediction"
    ax.set_title(f"All Metrics — {task_title}", fontsize=13, pad=10)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(EVAL_DIR / f"{output_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_stem)


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tracks",
        nargs="+",
        required=True,
        help="Pickle basenames for substrate evaluation, "
        "e.g. 'track_a_phylo_folds track_b_new_dataset'",
    )
    p.add_argument(
        "--tps-tracks",
        nargs="+",
        default=None,
        help="Pickle basenames for TPS detection evaluation "
        "(same order as --tracks). If omitted, uses --tracks.",
    )
    p.add_argument(
        "--track-labels",
        nargs="+",
        required=True,
        help="Display labels for each track, same order as --tracks",
    )
    p.add_argument(
        "--output-prefix",
        default="cross_track",
        help="Filename prefix for output figures (default: cross_track)",
    )
    return p.parse_args()


def _load_tracks(
    track_names: list[str], track_labels: list[str]
) -> list[tuple[str, dict, dict, dict]]:
    """Load evaluation pickles for a list of tracks."""
    tracks_data: list[tuple[str, dict, dict, dict]] = []
    for track_name, track_label in zip(track_names, track_labels):
        try:
            ap, rocauc, mccf1 = _load_metric_pickles(track_name)
            tracks_data.append((track_label, ap, rocauc, mccf1))
            logger.info("Loaded %s as '%s'", track_name, track_label)
        except FileNotFoundError:
            logger.warning("Skipping %s — pickle not found", track_name)
    return tracks_data


def main() -> None:
    args = parse_args()
    assert len(args.tracks) == len(
        args.track_labels
    ), "--tracks and --track-labels must have the same length"

    substrate_data = _load_tracks(args.tracks, args.track_labels)

    tps_names = args.tps_tracks if args.tps_tracks else args.tracks
    assert len(tps_names) == len(
        args.track_labels
    ), "--tps-tracks must have the same length as --track-labels"
    tps_data = _load_tracks(tps_names, args.track_labels)

    if not substrate_data and not tps_data:
        logger.error("No tracks loaded; nothing to plot.")
        return

    for task, data in [("substrate", substrate_data), ("tps_detection", tps_data)]:
        if not data:
            continue
        task_suffix = "substrate" if task == "substrate" else "tps_det"
        plot_cross_track_overview(
            data, task, f"{args.output_prefix}_overview_{task_suffix}"
        )
        plot_simbin_degradation(
            data, task, f"{args.output_prefix}_simbin_{task_suffix}"
        )
        plot_metric_heatmap(data, task, f"{args.output_prefix}_heatmap_{task_suffix}")


if __name__ == "__main__":
    main()
