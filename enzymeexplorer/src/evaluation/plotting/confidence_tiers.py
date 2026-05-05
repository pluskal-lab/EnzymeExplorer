"""Confidence-tier ladder plot.

For each classifier, draws one horizontal bar per class spanning ``[0, 1]``,
segmented and colour-coded by tier. Lower-bound thresholds are annotated
above each segment. Designed to be a drop-in deployment cheat-sheet: read off
a class's row, find the score, look up the tier.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.confidence_tiers import (
    DEFAULT_TIER_DEFINITIONS,
    NEGATIVE_TIER_COLOR,
    NEGATIVE_TIER_NAME,
    TierDefinition,
    tier_intervals_for_class,
)
from enzymeexplorer.src.evaluation.plotting import theme


def _tier_color_map(
    tier_definitions: Iterable[TierDefinition],
) -> dict[str, str]:
    out = {NEGATIVE_TIER_NAME: NEGATIVE_TIER_COLOR}
    for t in tier_definitions:
        out[t.name] = t.color or "#888888"
    return out


def _draw_ladder_panel(
    ax: plt.Axes,
    tier_table: pd.DataFrame,
    classifier: str,
    *,
    classes_order: list[str],
    tier_definitions: Iterable[TierDefinition],
    color_map: dict[str, str],
    annotate_boundaries: bool,
    annotate_tier_names: bool,
    bar_height: float = 0.65,
) -> None:
    """Draw the per-class horizontal ladder for one classifier into ``ax``."""
    n = len(classes_order)
    for i, cls in enumerate(reversed(classes_order)):
        segments = tier_intervals_for_class(
            tier_table, classifier, cls, tier_definitions=tier_definitions,
        )
        for tier_name, lo, hi in segments:
            if hi <= lo:
                continue
            ax.broken_barh(
                [(lo, hi - lo)],
                (i - bar_height / 2, bar_height),
                facecolors=color_map.get(tier_name, "#888888"),
                edgecolor="black",
                linewidth=0.4,
            )
            if annotate_tier_names and (hi - lo) >= 0.06:
                ax.text(
                    (lo + hi) / 2,
                    i,
                    tier_name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        if annotate_boundaries:
            for _, lo, hi in segments[:-1]:
                ax.text(
                    hi,
                    i + bar_height / 2 + 0.05,
                    f"{hi:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(classes_order)))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def _tier_legend_handles(
    tier_definitions: Iterable[TierDefinition],
    color_map: dict[str, str],
) -> tuple[list, list[str]]:
    def _tier_sort_key(t):
        if t.recall_target is not None:
            return (0, -t.recall_target)
        return (1, t.precision_target if t.precision_target is not None else 0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[NEGATIVE_TIER_NAME])]
    labels = [NEGATIVE_TIER_NAME]
    for t in sorted(tier_definitions, key=_tier_sort_key):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[t.name]))
        if t.precision_target is not None:
            labels.append(f"{t.name} (p≥{t.precision_target:.0%})")
        else:
            labels.append(f"{t.name} (r≥{t.recall_target:.0%})")
    return handles, labels


def plot_tier_ladder(
    tier_table: pd.DataFrame,
    classifier: str,
    *,
    classes_order: list[str] | None = None,
    tier_definitions: Iterable[TierDefinition] = DEFAULT_TIER_DEFINITIONS,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    annotate_boundaries: bool = True,
    annotate_tier_names: bool = True,
) -> plt.Figure:
    """One horizontal tier ladder per class for ``classifier``."""
    sub = tier_table[tier_table["classifier"] == classifier]
    if sub.empty:
        raise ValueError(f"No tier rows for classifier {classifier!r}")
    if classes_order is None:
        classes_order = sorted(sub["class"].unique())
    classes_order = [c for c in classes_order if c in set(sub["class"])]
    n = len(classes_order)
    if figsize is None:
        figsize = (14, max(3.0, 0.65 * n + 1.8))

    color_map = _tier_color_map(tier_definitions)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_ladder_panel(
        ax, tier_table, classifier,
        classes_order=classes_order,
        tier_definitions=tier_definitions,
        color_map=color_map,
        annotate_boundaries=annotate_boundaries,
        annotate_tier_names=annotate_tier_names,
    )
    ax.set_xlabel("Prediction score")
    ax.set_title(
        title or f"{theme.display_name(classifier)} — confidence tiers per substrate"
    )
    handles, labels = _tier_legend_handles(tier_definitions, color_map)
    ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Tier (target)",
    )
    fig.tight_layout()
    return fig


def plot_tier_ladder_grid(
    tier_table: pd.DataFrame,
    *,
    classifier_order: list[str],
    classes_order: list[str] | None = None,
    tier_definitions: Iterable[TierDefinition] = DEFAULT_TIER_DEFINITIONS,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    annotate_boundaries: bool = False,
    annotate_tier_names: bool = True,
) -> plt.Figure:
    """Multi-panel tier ladder: one stacked panel per classifier.

    Useful as a publication overview comparing the score-to-tier mapping
    across all classifiers at once. Boundary annotations default off here
    to keep the grid readable; turn back on for a denser deployment view.
    """
    sub = tier_table[tier_table["classifier"].isin(classifier_order)]
    if sub.empty:
        raise ValueError("No tier rows for the requested classifiers")
    classifier_order = [c for c in classifier_order if c in set(sub["classifier"])]
    if classes_order is None:
        classes_order = sorted(sub["class"].unique())

    n = len(classifier_order)
    if figsize is None:
        per_panel_h = max(2.5, 0.45 * len(classes_order) + 1.2)
        figsize = (14, per_panel_h * n + 1.0)

    color_map = _tier_color_map(tier_definitions)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, clf in zip(axes, classifier_order):
        _draw_ladder_panel(
            ax, tier_table, clf,
            classes_order=[c for c in classes_order if c in set(sub[sub["classifier"] == clf]["class"])],
            tier_definitions=tier_definitions,
            color_map=color_map,
            annotate_boundaries=annotate_boundaries,
            annotate_tier_names=annotate_tier_names,
        )
        ax.set_title(theme.display_name(clf), loc="left", fontsize=11)

    axes[-1].set_xlabel("Prediction score")
    handles, labels = _tier_legend_handles(tier_definitions, color_map)
    fig.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Tier (target)",
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.98, 1.0))
    return fig
