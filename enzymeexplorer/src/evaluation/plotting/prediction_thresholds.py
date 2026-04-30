"""Heatmap plots for precision-targeted prediction-score thresholds.

Two views per classifier:
  * threshold heatmap (rows=classes, cols=precision targets, cell=score
    threshold) — lookup table for deployment.
  * recall heatmap (same axes, cell=recall achieved at that threshold) —
    shows the precision-recall trade-off implied by each confidence tier.

Both consume the long-form table from
``prediction_thresholds.compute_thresholds_table``.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme


def _pivot(
    df: pd.DataFrame,
    classifier: str,
    value_col: str,
    classes_order: list[str] | None,
) -> pd.DataFrame:
    sub = df[df["classifier"] == classifier]
    if sub.empty:
        raise ValueError(f"No rows for classifier {classifier!r}")
    pivot = sub.pivot(index="class", columns="precision_target", values=value_col)
    if classes_order is not None:
        pivot = pivot.reindex([c for c in classes_order if c in pivot.index])
    # Sort precision targets descending so the strictest tier is leftmost.
    pivot = pivot.reindex(columns=sorted(pivot.columns, reverse=True))
    return pivot


def _draw_heatmap(
    matrix: pd.DataFrame,
    *,
    cmap: str,
    annotate: bool,
    annotation_fmt: str,
    figsize: tuple[float, float] | None,
    title: str,
    cbar_label: str,
    vmin: float | None,
    vmax: float | None,
) -> plt.Figure:
    if figsize is None:
        figsize = (max(5, 0.9 * len(matrix.columns) + 2), max(3, 0.45 * len(matrix.index) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    values = matrix.values.astype(float)
    norm_vmin = float(np.nanmin(values)) if vmin is None else vmin
    norm_vmax = float(np.nanmax(values)) if vmax is None else vmax
    if not np.isfinite(norm_vmin) or not np.isfinite(norm_vmax) or norm_vmax <= norm_vmin:
        norm_vmin, norm_vmax = 0.0, 1.0
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=norm_vmin, vmax=norm_vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(list(matrix.index))
    ax.set_xlabel("Precision target")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    if annotate:
        threshold = (norm_vmin + norm_vmax) / 2
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if not np.isfinite(v):
                    txt = "—"
                    color = "0.4"
                else:
                    txt = annotation_fmt.format(v)
                    color = "white" if v < threshold else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    return fig


def plot_threshold_heatmap(
    threshold_df: pd.DataFrame,
    classifier: str,
    *,
    classes_order: list[str] | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Score thresholds for each (class, precision target)."""
    pivot = _pivot(threshold_df, classifier, "threshold", classes_order)
    return _draw_heatmap(
        pivot,
        cmap=cmap,
        annotate=annotate,
        annotation_fmt="{:.3f}",
        figsize=figsize,
        title=title or f"{theme.display_name(classifier)} — score threshold per precision target",
        cbar_label="Score threshold",
        vmin=vmin,
        vmax=vmax,
    )


def plot_recall_at_thresholds_heatmap(
    threshold_df: pd.DataFrame,
    classifier: str,
    *,
    classes_order: list[str] | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    as_percentage: bool = True,
) -> plt.Figure:
    """Recall achieved at each (class, precision target) — i.e. the fraction
    of true positives captured when applying the corresponding threshold."""
    pivot = _pivot(threshold_df, classifier, "recall", classes_order)
    if as_percentage:
        pivot = pivot * 100.0
        annotation_fmt = "{:.1f}"
        cbar_label = "Recall (%)"
        vmin, vmax = 0.0, 100.0
    else:
        annotation_fmt = "{:.3f}"
        cbar_label = "Recall"
        vmin, vmax = 0.0, 1.0
    return _draw_heatmap(
        pivot,
        cmap=cmap,
        annotate=annotate,
        annotation_fmt=annotation_fmt,
        figsize=figsize,
        title=title or f"{theme.display_name(classifier)} — recall per precision target",
        cbar_label=cbar_label,
        vmin=vmin,
        vmax=vmax,
    )


def plot_n_above_threshold_heatmap(
    threshold_df: pd.DataFrame,
    classifier: str,
    *,
    classes_order: list[str] | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Number of predictions above the threshold per (class, precision target).

    Useful sanity check: at strict targets (99%) and rare classes, the count
    can drop to a handful, signalling unreliable thresholds.
    """
    pivot = _pivot(threshold_df, classifier, "n_above_threshold", classes_order)
    return _draw_heatmap(
        pivot,
        cmap=cmap,
        annotate=annotate,
        annotation_fmt="{:.0f}",
        figsize=figsize,
        title=title or f"{theme.display_name(classifier)} — predictions retained per precision target",
        cbar_label="# predictions ≥ threshold",
        vmin=None,
        vmax=None,
    )
