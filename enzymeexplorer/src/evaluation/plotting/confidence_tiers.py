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
    bar_height = 0.65

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
    ax.set_xlabel("Prediction score")
    ax.set_title(
        title or f"{theme.display_name(classifier)} — confidence tiers per substrate"
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    def _tier_sort_key(t):
        # Loose → strict: recall-target tiers (lowest lower bound) first,
        # then precision targets ascending.
        if t.recall_target is not None:
            return (0, -t.recall_target)
        return (1, t.precision_target if t.precision_target is not None else 0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[NEGATIVE_TIER_NAME])
    ]
    legend_labels = [NEGATIVE_TIER_NAME]
    for t in sorted(tier_definitions, key=_tier_sort_key):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[t.name]))
        if t.precision_target is not None:
            legend_labels.append(f"{t.name} (p≥{t.precision_target:.0%})")
        else:
            legend_labels.append(f"{t.name} (r≥{t.recall_target:.0%})")
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Tier (target)",
    )
    fig.tight_layout()
    return fig
