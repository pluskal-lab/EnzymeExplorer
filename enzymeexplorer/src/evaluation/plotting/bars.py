"""Bar charts driven by long-form bootstrap draws.

Functions consume the ``bootstrap_long.csv`` dataframe (columns:
``classifier, class, metric, bootstrap_idx, value``) and let seaborn's
``barplot`` compute error bars via ``errorbar=('ci', 95)`` — matching the
convention from the legacy notebook so CI widths are comparable.

``bar_classifier_zoom`` decides on the fly whether a zoomed companion plot
adds value: it picks the top-performing classifiers (within ``delta_pct``
of the maximum mean) and, if at least one CI pair overlaps inside that
group, returns a tightly-bounded second figure restricted to the group.
``NoZoomNeeded`` signals that no companion plot is warranted.
"""

from __future__ import annotations

import math
from typing import Mapping

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme

_DEFAULT_CAPSIZE = 0.15
_DEFAULT_BAR_WIDTH = 0.55
_PER_BAR_WIDTH_INCHES = 1.2
_FIGSIZE_HEIGHT = 5.0


class NoZoomNeeded(Exception):
    """Raised when the auto-zoom heuristic decides not to emit a companion."""


def compute_full_zoom_ylim(
    means_pct: list[float],
) -> tuple[float, float]:
    """Compute (ymin, ymax) for a "full" zoom that keeps every classifier on
    the canvas while removing dead space outside the perf band.

    Heuristic: take ``[perf_min, perf_max]`` of the bar means and grow it to
    ``[max(0, 2*pmin - pmax), min(100, 2*pmax - pmin)]``. Then floor to the
    largest multiple of 5 ≤ low (or 0 if negative) and ceil to the smallest
    multiple of 5 ≥ high (or 100 if >100). Always returns a non-empty range.
    """
    if not means_pct:
        return 0.0, 100.0
    pmin = float(min(means_pct))
    pmax = float(max(means_pct))
    low = max(0.0, 2 * pmin - pmax)
    high = min(100.0, 2 * pmax - pmin)
    floor = max(0.0, math.floor(low / 5.0) * 5.0)
    ceil = min(100.0, math.ceil(high / 5.0) * 5.0)
    if ceil <= floor:
        ceil = min(100.0, floor + 5.0)
    return floor, ceil


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


def _resolve_xtick_label(clf: str, overrides: Mapping[str, str] | None) -> str:
    if overrides and clf in overrides:
        return overrides[clf]
    return theme.display_name(clf)


def bar_classifier(
    bootstrap_long: pd.DataFrame,
    target_class: str,
    *,
    metric: str = "ap",
    classifier_order: list[str] | None = None,
    classifier_subset: list[str] | None = None,
    pin_last: str | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str = "",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    value_label_fmt: str | None = "{:.2f}",
    capsize: float = _DEFAULT_CAPSIZE,
    errorbar: tuple = ("ci", 95),
    xtick_overrides: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One bar per classifier for a single (class, metric).

    Bars are drawn via ``sns.barplot``; error bars come from seaborn's
    built-in CI estimation (default ``errorbar=('ci', 95)`` — bootstrapped
    95% CI of the mean over the long-form draws). ``xtick_overrides``
    replaces the default per-classifier display name with a custom label
    (used by ablation plots that want to show only the varied dimension).
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
    if pin_last and pin_last in classifier_order:
        classifier_order = [c for c in classifier_order if c != pin_last] + [pin_last]

    palette = palette if palette is not None else theme.comparison_palette(classifier_order)

    df = df.copy()
    df["value_pct"] = df["value"] * 100.0
    label_for = {c: _resolve_xtick_label(c, xtick_overrides) for c in classifier_order}
    df["display"] = df["classifier"].map(label_for)
    display_order = [label_for[c] for c in classifier_order]
    palette_by_display = {label_for[c]: palette[c] for c in classifier_order}

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


def _select_zoom_subset(
    df_long: pd.DataFrame,
    *,
    delta_pct: float,
) -> tuple[list[str], float, float]:
    """Pick top-performing classifiers and decide if zoom is needed.

    Returns ``(subset, ymin, ymax)`` or raises ``NoZoomNeeded``. A subset
    is emitted when (a) at least two classifiers fall within ``delta_pct``
    of the leader, and (b) at least one pair of 95% CIs in that subset
    overlaps — i.e. a zoomed view actually changes the visual story.
    """
    means = df_long.groupby("classifier")["value"].mean() * 100.0
    if means.empty:
        raise NoZoomNeeded("no rows")
    m_max = float(means.max())
    top = means[means >= m_max - delta_pct].sort_values()
    if len(top) < 2:
        raise NoZoomNeeded("only one classifier in top group")

    # 95% CI per classifier inside the top group.
    ci_lo: dict[str, float] = {}
    ci_hi: dict[str, float] = {}
    for clf in top.index:
        vals = (
            df_long[df_long["classifier"] == clf]["value"].dropna().to_numpy() * 100.0
        )
        if vals.size == 0:
            continue
        ci_lo[clf] = float(np.nanquantile(vals, 0.025))
        ci_hi[clf] = float(np.nanquantile(vals, 0.975))
    overlap = False
    keys = list(ci_lo)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if ci_lo[a] <= ci_hi[b] and ci_lo[b] <= ci_hi[a]:
                overlap = True
                break
        if overlap:
            break
    if not overlap:
        raise NoZoomNeeded("top group is well-separated")

    min_top = float(top.min())
    pad = max(0.5, m_max - min_top) * 1.5
    floor = math.floor((min_top - 0.5 * pad) * 2.0) / 2.0
    floor = max(0.0, floor)
    ymax = 100.0
    if ymax - floor < 0.5:
        floor = max(0.0, ymax - 0.5)
    return list(top.index), floor, ymax


def bar_classifier_zoom(
    bootstrap_long: pd.DataFrame,
    target_class: str,
    *,
    metric: str = "ap",
    classifier_subset: list[str] | None = None,
    forced_subset: list[str] | None = None,
    delta_pct: float = 2.0,
    pin_last: str | None = None,
    palette: dict[str, tuple[float, float, float] | str] | None = None,
    xtick_overrides: Mapping[str, str] | None = None,
    base_title: str | None = None,
    fixed_order: list[str] | None = None,
) -> plt.Figure:
    """Auto-zoom companion bar plot.

    If ``forced_subset`` is provided, those classifiers are used directly
    and zoom emission is unconditional (subject to ``len >= 2``).
    Otherwise the heuristic in ``_select_zoom_subset`` decides whether to
    emit; ``NoZoomNeeded`` is raised when not. The y-axis bounds are
    derived from the chosen subset.
    """
    df = _filter_long(
        bootstrap_long, metric=metric, target_class=target_class,
        classifier_subset=classifier_subset,
    )
    if df.empty:
        raise NoZoomNeeded("no bootstrap rows")
    if forced_subset:
        subset = [c for c in forced_subset if c in set(df["classifier"])]
        if len(subset) < 2:
            raise NoZoomNeeded("forced subset too small after filtering")
        sub_means = df[df["classifier"].isin(subset)].groupby("classifier")["value"].mean() * 100.0
        m_max = float(sub_means.max())
        m_min = float(sub_means.min())
        pad = max(0.5, m_max - m_min) * 1.5
        floor = math.floor((m_min - 0.5 * pad) * 2.0) / 2.0
        floor = max(0.0, floor)
        ymin, ymax = floor, 100.0
    else:
        subset, ymin, ymax = _select_zoom_subset(df, delta_pct=delta_pct)

    # If a fixed order is provided (ablations), preserve it; else sort by mean
    # and apply pin_last (all-methods).
    if fixed_order:
        ordered = [c for c in fixed_order if c in subset]
    else:
        sub_means = df[df["classifier"].isin(subset)].groupby("classifier")["value"].mean()
        ordered = sub_means.sort_values().index.tolist()
        if pin_last and pin_last in ordered:
            ordered = [c for c in ordered if c != pin_last] + [pin_last]

    title = f"{base_title} (zoomed)" if base_title else f"{target_class} {metric.upper()} (zoomed)"
    return bar_classifier(
        bootstrap_long, target_class=target_class, metric=metric,
        classifier_subset=subset,
        classifier_order=ordered,
        palette=palette,
        title=title,
        ylim=(ymin, ymax),
        xtick_overrides=xtick_overrides,
    )


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
    xtick_overrides: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Clustered bars: x = ``classes``, hue = ``classifier``.

    Error bars are computed by seaborn from the long-form bootstrap draws.
    Classifiers with no coverage of any requested class are dropped.
    ``xtick_overrides`` is applied to the legend hue labels.
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
    label_for = {c: _resolve_xtick_label(c, xtick_overrides) for c in classifier_order}
    df["display"] = df["classifier"].map(label_for)
    df["class"] = pd.Categorical(df["class"], categories=classes, ordered=True)
    display_order = [label_for[c] for c in classifier_order]
    palette_by_display = {label_for[c]: palette[c] for c in classifier_order}

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
