"""Bar charts driven by long-form bootstrap draws.

Functions consume the ``bootstrap_long.csv`` dataframe (columns:
``classifier, class, metric, bootstrap_idx, value``) and let seaborn's
``barplot`` compute error bars via ``errorbar=('ci', 95)`` — matching the
convention from the legacy notebook so CI widths are comparable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme

_DEFAULT_CAPSIZE = 0.15
_DEFAULT_BAR_WIDTH = 0.55
_PER_BAR_WIDTH_INCHES = 1.2
_FIGSIZE_HEIGHT = 5.0


def _auto_figsize(n_bars: int) -> tuple[float, float]:
    return (max(4.0, _PER_BAR_WIDTH_INCHES * n_bars + 1.5), _FIGSIZE_HEIGHT)


def _filter_long(
    bootstrap_long: pd.DataFrame,
    *,
    metric: str,
    target_class: str | None = None,
    classes: list[str] | None = None,
    classifier_subset: list[str] | None = None,
) -> pd.DataFrame:
    df = bootstrap_long[bootstrap_long["metric"] == metric].copy()
    if target_class is not None:
        df = df[df["class"] == target_class]
    if classes is not None:
        df = df[df["class"].isin(classes)]
    if classifier_subset is not None:
        df = df[df["classifier"].isin(classifier_subset)]
    df = df.dropna(subset=["value"])
    return df


def bar_classifier(
    bootstrap_long: pd.DataFrame,
    target_class: str,
    *,
    metric: str = "ap",
    classifier_order: list[str] | None = None,
    classifier_subset: list[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str = "",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    value_label_fmt: str | None = "{:.2f}",
    capsize: float = _DEFAULT_CAPSIZE,
    errorbar: tuple = ("ci", 95),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One bar per classifier for a single (class, metric).

    Bars are drawn via ``sns.barplot``; error bars come from seaborn's
    built-in CI estimation (default ``errorbar=('ci', 95)`` — bootstrapped
    95% CI of the mean over the long-form draws). Mirrors the legacy
    notebook convention.
    """
    df = _filter_long(
        bootstrap_long, metric=metric, target_class=target_class,
        classifier_subset=classifier_subset,
    )
    if df.empty:
        raise ValueError(
            f"No bootstrap rows for class={target_class!r} metric={metric!r}"
        )
    if classifier_order is None:
        means = df.groupby("classifier")["value"].mean().sort_values()
        classifier_order = means.index.tolist()
    else:
        df = df[df["classifier"].isin(classifier_order)]
        classifier_order = [c for c in classifier_order if c in set(df["classifier"])]

    palette = palette if palette is not None else theme.comparison_palette(classifier_order)

    df = df.copy()
    df["value_pct"] = df["value"] * 100.0
    df["display"] = df["classifier"].map(theme.display_name)
    display_order = [theme.display_name(c) for c in classifier_order]
    palette_by_display = {theme.display_name(c): palette[c] for c in classifier_order}

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize if figsize is not None else _auto_figsize(len(classifier_order))
        )
    else:
        fig = ax.figure

    sns.barplot(
        data=df,
        x="display",
        y="value_pct",
        order=display_order,
        hue="display",
        hue_order=display_order,
        palette=palette_by_display,
        legend=False,
        capsize=capsize,
        errorbar=errorbar,
        err_kws={"linewidth": 1.0, "color": "black"},
        edgecolor="black",
        linewidth=0.5,
        width=_DEFAULT_BAR_WIDTH,
        ax=ax,
    )

    y_min = ylim[0] if ylim is not None else None
    y_max = ylim[1] if ylim is not None else None
    if value_label_fmt:
        means_pct = df.groupby("classifier")["value_pct"].mean().to_dict()
        for i, clf in enumerate(classifier_order):
            mean_val = means_pct.get(clf, np.nan)
            if not np.isfinite(mean_val):
                continue
            if y_min is not None and mean_val < y_min:
                ax.text(
                    i,
                    y_min + 0.02 * ((y_max - y_min) if y_max is not None else 1.0),
                    f"↓ {value_label_fmt.format(mean_val)}",
                    ha="center", va="bottom", fontsize=8, color="0.3",
                )
                continue
            offset = 0.5 if y_max is None else 0.012 * ((y_max - y_min) if y_min is not None else 1.0)
            label_y = mean_val + offset
            if y_max is not None and label_y > y_max:
                label_y = y_max - 0.02 * ((y_max - y_min) if y_min is not None else 1.0)
            ax.text(
                i, label_y, value_label_fmt.format(mean_val),
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticklabels(display_order, rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if ylabel is not None else f"{metric.upper()} (%)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def bar_per_class(
    bootstrap_long: pd.DataFrame,
    *,
    classes: list[str],
    classifier_order: list[str],
    metric: str = "ap",
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str = "",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    legend_loc: str = "upper left",
    legend_bbox: tuple[float, float] = (1.02, 1.0),
    capsize: float = 0.05,
    errorbar: tuple = ("ci", 95),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Clustered bars: x = ``classes``, hue = ``classifier``.

    Error bars are computed by seaborn from the long-form bootstrap draws.
    Classifiers with no coverage of any requested class are dropped.
    """
    df = _filter_long(
        bootstrap_long, metric=metric, classes=classes,
        classifier_subset=classifier_order,
    )
    if df.empty:
        raise ValueError("No bootstrap rows match the requested classes/classifiers")
    present = set(df["classifier"].astype(str))
    classifier_order = [c for c in classifier_order if c in present]
    palette = palette if palette is not None else theme.comparison_palette(classifier_order)

    df = df.copy()
    df["value_pct"] = df["value"] * 100.0
    df["display"] = df["classifier"].map(theme.display_name)
    df["class"] = pd.Categorical(df["class"], categories=classes, ordered=True)
    display_order = [theme.display_name(c) for c in classifier_order]
    palette_by_display = {theme.display_name(c): palette[c] for c in classifier_order}

    if ax is None:
        if figsize is None:
            cluster_w = max(1.0, 0.35 * len(classifier_order) + 0.5)
            figsize = (max(6.0, cluster_w * len(classes) + 1.5), _FIGSIZE_HEIGHT)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(
        data=df,
        x="class",
        y="value_pct",
        hue="display",
        order=classes,
        hue_order=display_order,
        palette=palette_by_display,
        capsize=capsize,
        errorbar=errorbar,
        err_kws={"linewidth": 1.0, "color": "black"},
        edgecolor="black",
        linewidth=0.4,
        width=_DEFAULT_BAR_WIDTH,
        ax=ax,
    )
    ax.set_xticklabels(classes, rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if ylabel is not None else f"{metric.upper()} (%)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(
        loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False, title="Classifier"
    )
    fig.tight_layout()
    return fig
