"""Threshold-sweep diagnostic plots.

For HBI baselines whose optimal e-value (or bitscore) differs by class, plot
the per-class metric across the candidate versions to visualise where each
class peaks. The argmax per class is highlighted; the per-class best version
is exactly the one ``selection.pick_best_versions_per_class`` would pick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

from enzymeexplorer.src.evaluation import io as eio
from enzymeexplorer.src.evaluation.classes import SHORT_TO_SMILES
from enzymeexplorer.src.evaluation.plotting import theme


def compute_threshold_sweep(
    model: str,
    candidate_versions: list[str],
    classes: Iterable[str],
    *,
    metric: str = "ap",
    output_root: Path | None = None,
) -> pd.DataFrame:
    """Score each (version, class) on the latest run; return a long DF with
    columns ``version, class, metric, value`` (mean across folds)."""
    classes = list(classes)
    records: list[dict] = []
    for version in candidate_versions:
        try:
            exp = eio.latest_experiment_dir(model, version, output_root=output_root)
            raws = eio.load_pickle_folds(exp)
        except FileNotFoundError:
            continue
        common = set(raws[0][1])
        for _, names, _ in raws[1:]:
            common &= set(names)
        covered = [c for c in classes if SHORT_TO_SMILES[c] in common]
        if not covered:
            continue
        per_fold = eio.folds_to_dfs(raws, classes_subset=covered)
        for cls in covered:
            scores: list[float] = []
            for _, (lab, prd) in per_fold.items():
                y = lab[cls].to_numpy()
                s = prd[cls].to_numpy()
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                if metric == "ap":
                    scores.append(float(average_precision_score(y, s)))
                elif metric == "roc_auc":
                    scores.append(float(roc_auc_score(y, s)))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            if not scores:
                continue
            records.append(
                {
                    "version": version,
                    "class": cls,
                    "metric": metric,
                    "value": float(np.mean(scores)),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    *,
    version_to_x: dict[str, float] | None = None,
    classes: list[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str = "",
    xlabel: str = "Threshold",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 5),
    mark_argmax: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot one line per class across the candidate versions.

    ``version_to_x`` maps a version label (e.g. ``"eval1e-20"``) to a numeric
    x position (e.g. ``20`` for ``-log10`` of the e-value). When omitted,
    versions are placed on a categorical axis in input order.
    """
    if sweep_df.empty:
        raise ValueError("sweep_df is empty; nothing to plot")
    classes = classes if classes is not None else sorted(sweep_df["class"].unique())
    palette = palette if palette is not None else theme.class_palette(classes)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if version_to_x is None:
        version_order = list(dict.fromkeys(sweep_df["version"].tolist()))
        version_to_x = {v: i for i, v in enumerate(version_order)}
    pos_to_label = {pos: ver for ver, pos in version_to_x.items()}

    for cls in classes:
        sub = sweep_df[sweep_df["class"] == cls].copy()
        if sub.empty:
            continue
        sub["x"] = sub["version"].map(version_to_x)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        ys = sub["value"].to_numpy() * 100
        xs = sub["x"].to_numpy()
        color = palette.get(cls, (0.4, 0.4, 0.4))
        ax.plot(xs, ys, marker="o", markersize=4, label=cls, color=color, linewidth=1.4)
        if mark_argmax and len(ys):
            best = int(np.argmax(ys))
            ax.scatter(xs[best], ys[best], color="black", s=80, zorder=5, alpha=0.7)

    if version_to_x:
        sorted_xs = sorted(set(version_to_x.values()))
        ax.set_xticks(sorted_xs)
        ax.set_xticklabels(
            [pos_to_label.get(x, str(x)) for x in sorted_xs],
            rotation=45,
            ha="right",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else "AP (%)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, ncols=1)
    fig.tight_layout()
    return fig


def parse_eval_neglog10(version: str) -> float | None:
    """Parse ``eval1e-20``, ``eval10``, ``eval100`` etc. into the negated
    base-10 exponent. ``eval1e-20`` -> 20; ``eval10`` -> -1; ``eval100`` -> -2;
    ``eval1e0`` -> 0. Returns None when the format isn't recognised."""
    if not version.startswith("eval"):
        return None
    rest = version[4:]
    if rest.startswith("1e"):
        try:
            return -float(rest[2:])
        except ValueError:
            return None
    try:
        v = float(rest)
        if v <= 0:
            return None
        return -float(np.log10(v))
    except ValueError:
        return None
