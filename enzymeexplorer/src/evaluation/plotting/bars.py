"""Bar charts for the evaluation pipeline.

The bar plots consume a *summary* DataFrame (the output of
``compute_cis``) with columns ``classifier, class, metric, point,
ci_low, ci_high``. Bar height is the saved point estimate (the canonical
fold-mean AP) and error bars are precomputed asymmetric distances around
that point — guaranteeing the bar height equals what is saved in
``summary.csv`` and that the CI describes uncertainty of that exact
statistic. We use ``matplotlib.bar`` + ``ax.errorbar`` directly to skip
seaborn's internal CI bootstrap, which would otherwise re-bootstrap our
already-bootstrapped draws.

``bar_classifier_zoom`` decides on the fly whether a zoomed companion
plot adds value: it picks the top-performing classifiers (within
``delta_pct`` of the maximum point) and emits a tightly-bounded second
figure if any pair of CIs in that group overlap. ``NoZoomNeeded``
signals that no companion plot is warranted.
"""

from __future__ import annotations

import math
from typing import Mapping

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme

# Single-classifier bar plots (TPS detection AP, Substrate prediction
# mAP, TPS_IDS_mAP). Bars sit at custom x positions ``i * _BAR_X_STEP``
# so adjacent bars are close together (small inter-bar gap = step minus
# width) and the figure stays narrow. Per-bar horizontal space is sized
# to fit a horizontal multi-line label (e.g. "EnzymeExplorer\nDomains")
# without rotation.
_DEFAULT_BAR_WIDTH = 0.62
_BAR_X_STEP = 0.95
_PER_BAR_WIDTH_INCHES = 0.7
_FIGSIZE_HEIGHT = 2.7


class NoZoomNeeded(Exception):
    """Raised when the auto-zoom heuristic decides not to emit a companion."""


def compute_full_zoom_ylim(
    means_pct: list[float],
) -> tuple[float, float]:
    """Compute (ymin, ymax) for a "full" zoom that keeps every classifier on
    the canvas while removing dead space outside the perf band.

    Kept for backward compat with old callers; new callers should pass
    CI bounds directly via :func:`compute_ci_zoom_ylim`.
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


_NICE_STEPS: tuple[float, ...] = (
    0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
)


def _nice_step(span: float) -> float:
    """Pick a "nice" round step so the panel ends up with ~5–10 grid lines.

    Returns the largest value in ``_NICE_STEPS`` that is ≤ ``span / 6``
    so the resulting axis has at least six visible major ticks. For
    very wide spans the step plateaus at 20.
    """
    if span <= 0:
        return 0.05
    target = span / 6.0
    chosen = _NICE_STEPS[0]
    for s in _NICE_STEPS:
        if s <= target:
            chosen = s
        else:
            break
    return chosen


def compute_ci_zoom_ylim(
    ci_low_pct: list[float],
    ci_high_pct: list[float],
) -> tuple[float, float]:
    """Compute a tight (ymin, ymax) bracketing every CI in the panel.

    Strategy:
      1. Compute the *raw* CI envelope ``[min(ci_low), max(ci_high)]``,
         clipped to ``[0, 100]``.
      2. Pick a "nice" snap step proportional to the envelope span so
         tight envelopes (e.g. 99.4–100) zoom to a 0.1- or 0.2-unit
         grid and wide envelopes (e.g. 30–95) snap to the 5-unit grid
         used previously.
      3. Floor/ceil to that step. The resulting axis is just wide
         enough to display every CI with naked-eye-comparable spacing.

    Falls back to ``(0, 100)`` when no CIs are finite.
    """
    los = [v for v in ci_low_pct if np.isfinite(v)]
    his = [v for v in ci_high_pct if np.isfinite(v)]
    if not los or not his:
        return 0.0, 100.0
    lo = max(0.0, min(los))
    hi = min(100.0, max(his))
    span = max(hi - lo, 0.05)
    step = _nice_step(span)
    floor = max(0.0, math.floor(lo / step) * step)
    ceil = min(100.0, math.ceil(hi / step) * step)
    if ceil <= floor:
        ceil = min(100.0, floor + step)
    return floor, ceil


def _auto_figsize(n_bars: int) -> tuple[float, float]:
    return theme.scale_figsize(
        max(3.0, _PER_BAR_WIDTH_INCHES * n_bars + 1.2), _FIGSIZE_HEIGHT,
    )


def _resolve_xtick_label(clf: str, overrides: Mapping[str, str] | None) -> str:
    if overrides and clf in overrides:
        return overrides[clf]
    return theme.display_name(clf)


def _default_metric_label(metric: str, classes: list[str]) -> str:
    """Auto-pick AP vs mAP label based on whether every class is an aggregate.

    Aggregates are named ``*_mAP`` (e.g. ``Substrate_mAP``,
    ``TPS_IDS_mAP``, ``Overall_mAP``). When every class on the panel
    is an aggregate, the y-axis is labelled ``mAP``; otherwise plain
    ``AP``. ``metric`` switches the label between AP / ROC-AUC.
    """
    base = metric.upper() if metric != "ap" else "AP"
    if metric == "ap" and classes and all(str(c).endswith("_mAP") for c in classes):
        base = "mAP"
    return f"{base} (%)"


def _filter_summary(
    summary: pd.DataFrame,
    *,
    metric: str,
    target_class: str | None = None,
    classes: list[str] | None = None,
    classifier_subset: list[str] | None = None,
) -> pd.DataFrame:
    df = summary[summary["metric"] == metric].copy()
    if target_class is not None:
        df = df[df["class"] == target_class]
    if classes is not None:
        df = df[df["class"].isin(classes)]
    if classifier_subset is not None:
        df = df[df["classifier"].isin(classifier_subset)]
    df = df.dropna(subset=["point"])
    return df


def _yerr_pct(point: pd.Series, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    """Return a (2, N) array of asymmetric error distances IN PERCENT.

    ``ax.errorbar`` expects ``yerr=[lower_dist, upper_dist]`` measured
    *from the bar height*. CI ends are clipped to ``[0, 100]`` so the
    drawn whiskers never extend past the [0, 100] axis range — a normal
    (Wald) CI near the boundary can mathematically extend past 1.0;
    clipping at display time keeps the artefact off the figure without
    altering the saved ``ci_low`` / ``ci_high`` values in
    ``summary.csv``.
    """
    p = point.to_numpy(dtype=np.float64) * 100.0
    l = lo.to_numpy(dtype=np.float64) * 100.0
    h = hi.to_numpy(dtype=np.float64) * 100.0
    l_clipped = np.clip(l, 0.0, 100.0)
    h_clipped = np.clip(h, 0.0, 100.0)
    lower = np.maximum(0.0, p - l_clipped)
    upper = np.maximum(0.0, h_clipped - p)
    return np.vstack([lower, upper])


def bar_classifier(
    summary: pd.DataFrame,
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
    value_label_fmt: str | None = None,
    xtick_overrides: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One bar per classifier for a single (class, metric).

    Bars are matplotlib ``ax.bar`` rectangles drawn at ``summary['point']
    * 100``; error bars are ``ax.errorbar`` segments using
    ``summary['ci_low'..'ci_high']`` directly (no further bootstrap).
    """
    df = _filter_summary(
        summary, metric=metric, target_class=target_class,
        classifier_subset=classifier_subset,
    )
    if df.empty:
        raise ValueError(
            f"No summary rows for class={target_class!r} metric={metric!r}"
        )
    if classifier_order is None:
        ordered = df.sort_values("point")["classifier"].tolist()
    else:
        keep = set(df["classifier"])
        ordered = [c for c in classifier_order if c in keep]
    if pin_last and pin_last in ordered:
        ordered = [c for c in ordered if c != pin_last] + [pin_last]

    palette = palette if palette is not None else theme.comparison_palette(ordered)

    df = df.set_index("classifier").loc[ordered].reset_index()
    label_for = {c: _resolve_xtick_label(c, xtick_overrides) for c in ordered}
    display_labels = [label_for[c] for c in ordered]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize if figsize is not None else _auto_figsize(len(ordered))
        )
    else:
        fig = ax.figure

    # Bars at x = i * _BAR_X_STEP. With _BAR_X_STEP=0.6 and bar width
    # 0.45 the gap between adjacent bar edges is only 0.15 (data units),
    # tightly grouping the columns in a narrow figure.
    x_positions = np.arange(len(ordered), dtype=np.float64) * _BAR_X_STEP
    point_pct = df["point"].to_numpy(dtype=np.float64) * 100.0
    yerr = _yerr_pct(df["point"], df["ci_low"], df["ci_high"])
    bar_colors = [palette[c] for c in ordered]

    ax.bar(
        x_positions, point_pct, width=_DEFAULT_BAR_WIDTH,
        color=bar_colors, edgecolor="none", linewidth=0,
    )
    ax.errorbar(
        x_positions, point_pct,
        yerr=yerr, fmt="none",
        ecolor="0.15",
        elinewidth=theme.scale_stroke(0.7),
        capsize=theme.scale_stroke(2.0),
        capthick=theme.scale_stroke(0.7),
    )
    ax.yaxis.grid(
        True, color=theme.grid_color(),
        linewidth=theme.scale_stroke(0.5), zorder=0,
    )
    ax.set_axisbelow(True)

    y_min = ylim[0] if ylim is not None else None
    y_max = ylim[1] if ylim is not None else None
    # Each AP label sits above the upper whisker of its error bar (or
    # above the bar top if the upper whisker is shorter), so the value
    # is never visually overlapped by its own CI. If a label would
    # otherwise be above ``y_max``, we extend ``y_max`` to make room
    # rather than clamp the label down on top of the whisker.
    upper_whisker_pct = point_pct + yerr[1]
    if value_label_fmt:
        offset_base = (
            (y_max - y_min) if (y_min is not None and y_max is not None) else 1.0
        )
        offset = 0.5 if y_max is None else 0.018 * offset_base
        max_label_y = float("-inf")
        for xi, mean_val, top_val in zip(x_positions, point_pct, upper_whisker_pct):
            if not np.isfinite(mean_val):
                continue
            if y_min is not None and mean_val < y_min:
                ax.text(
                    xi,
                    y_min + 0.02 * offset_base,
                    f"↓ {value_label_fmt.format(mean_val)}",
                    ha="center", va="bottom",
                    fontsize=theme.scale_fontsize(8), color="0.3",
                )
                continue
            label_y = max(mean_val, top_val) + offset
            max_label_y = max(max_label_y, label_y)
            ax.text(
                xi, label_y, value_label_fmt.format(mean_val),
                ha="center", va="bottom",
                fontsize=theme.scale_fontsize(9), clip_on=False,
            )
        # Make room above the user-requested y_max if labels would
        # otherwise overshoot it (an extra ~3% header).
        if y_max is not None and np.isfinite(max_label_y) and max_label_y > y_max:
            extra = max_label_y - y_max + 0.03 * offset_base
            ylim = (y_min, y_max + extra) if y_min is not None else (None, y_max + extra)
    ax.set_xticks(x_positions)
    # Multi-line display names (e.g. "EnzymeExplorer\nDomains") render
    # vertically stacked and un-rotated, so the per-bar slot needs to
    # be wide enough to hold the longest single line — that's why
    # _PER_BAR_WIDTH_INCHES was bumped. Poster mode rotates instead so
    # long single-line names ("EnzymeExplorer") still fit in the
    # smaller 15 cm × 8 cm tile.
    rot, ha = theme.xtick_rotation(len(display_labels))
    ax.set_xticklabels(display_labels, rotation=rot, ha=ha)
    if ylim is None:
        # Default unzoomed range. ymax=102 leaves a 2-pp gap above the
        # CI clip at 100 so whiskers fit cleanly inside the axis frame.
        ax.set_ylim(0.0, 102.0)
    ax.set_xlabel("")
    ax.set_ylabel(
        ylabel if ylabel is not None
        else _default_metric_label(metric, list(df["class"].astype(str).unique()))
    )
    ax.set_title(title, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def _select_zoom_subset(
    df: pd.DataFrame,
    *,
    delta_pct: float,
) -> tuple[list[str], float, float]:
    """Pick top-performing classifiers from a *summary* and decide if zoom adds value.

    Returns ``(subset, ymin, ymax)`` or raises ``NoZoomNeeded``. A subset
    is emitted when (a) at least two classifiers fall within ``delta_pct``
    of the leader, and (b) at least one CI pair in that group overlaps.
    """
    means = df.set_index("classifier")["point"] * 100.0
    if means.empty:
        raise NoZoomNeeded("no rows")
    m_max = float(means.max())
    top = means[means >= m_max - delta_pct].sort_values()
    if len(top) < 2:
        raise NoZoomNeeded("only one classifier in top group")

    sub = df[df["classifier"].isin(top.index)]
    ci_lo = (sub.set_index("classifier")["ci_low"] * 100.0).to_dict()
    ci_hi = (sub.set_index("classifier")["ci_high"] * 100.0).to_dict()
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
    summary: pd.DataFrame,
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
    emit; ``NoZoomNeeded`` is raised when not.
    """
    df = _filter_summary(
        summary, metric=metric, target_class=target_class,
        classifier_subset=classifier_subset,
    )
    if df.empty:
        raise NoZoomNeeded("no summary rows")
    if forced_subset:
        keep = set(df["classifier"])
        subset = [c for c in forced_subset if c in keep]
        if len(subset) < 2:
            raise NoZoomNeeded("forced subset too small after filtering")
        sub = df[df["classifier"].isin(subset)].set_index("classifier")["point"] * 100.0
        m_max = float(sub.max())
        m_min = float(sub.min())
        pad = max(0.5, m_max - m_min) * 1.5
        floor = math.floor((m_min - 0.5 * pad) * 2.0) / 2.0
        floor = max(0.0, floor)
        ymin, ymax = floor, 100.0
    else:
        subset, ymin, ymax = _select_zoom_subset(df, delta_pct=delta_pct)

    if fixed_order:
        ordered = [c for c in fixed_order if c in subset]
    else:
        sub_means = df[df["classifier"].isin(subset)].set_index("classifier")["point"]
        ordered = sub_means.sort_values().index.tolist()
        if pin_last and pin_last in ordered:
            ordered = [c for c in ordered if c != pin_last] + [pin_last]

    title = base_title if base_title else f"{target_class} {metric.upper()}"
    return bar_classifier(
        summary, target_class=target_class, metric=metric,
        classifier_subset=subset,
        classifier_order=ordered,
        palette=palette,
        title=title,
        ylim=(ymin, ymax),
        xtick_overrides=xtick_overrides,
    )


def bar_per_class(
    summary: pd.DataFrame,
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
    xtick_overrides: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Clustered bars: x = ``classes``, hue = ``classifier``.

    Each bar is the saved point estimate; error bars are explicit
    ``[ci_low, ci_high]`` distances. Classifiers absent from
    ``summary`` for every requested class are dropped silently.
    """
    df = _filter_summary(
        summary, metric=metric, classes=classes,
        classifier_subset=classifier_order,
    )
    if df.empty:
        raise ValueError("No summary rows match the requested classes/classifiers")
    present = set(df["classifier"].astype(str))
    classifier_order = [c for c in classifier_order if c in present]
    palette = palette if palette is not None else theme.comparison_palette(classifier_order)

    label_for = {c: _resolve_xtick_label(c, xtick_overrides) for c in classifier_order}

    if ax is None:
        if figsize is None:
            cluster_w = max(0.55, 0.18 * len(classifier_order) + 0.25)
            figsize = theme.scale_figsize(
                max(4.5, cluster_w * len(classes) + 1.4), _FIGSIZE_HEIGHT,
            )
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_clf = len(classifier_order)
    cluster_width = 0.8
    bar_width = cluster_width / n_clf if n_clf > 0 else cluster_width
    x_centers = np.arange(len(classes))

    # Index df once for fast lookup.
    by_clf_class = df.set_index(["classifier", "class"])

    # Track the highest whisker top across all bars so we can extend the
    # user-supplied ylim if needed (whiskers clipped at 100 must sit
    # visibly below the axis frame, otherwise the cap line falls on the
    # frame and looks like an overflow).
    max_whisker_top = float("-inf")

    for i, clf in enumerate(classifier_order):
        offset = (i - (n_clf - 1) / 2.0) * bar_width
        points = []
        ci_los = []
        ci_his = []
        for cls in classes:
            try:
                row = by_clf_class.loc[(clf, cls)]
            except KeyError:
                points.append(np.nan)
                ci_los.append(np.nan)
                ci_his.append(np.nan)
                continue
            points.append(float(row["point"]))
            ci_los.append(float(row["ci_low"]))
            ci_his.append(float(row["ci_high"]))
        points_pct = np.asarray(points) * 100.0
        yerr = _yerr_pct(
            pd.Series(points), pd.Series(ci_los), pd.Series(ci_his),
        )
        valid = ~np.isnan(points_pct)
        upper_tops = points_pct[valid] + yerr[1, valid]
        if upper_tops.size:
            max_whisker_top = max(max_whisker_top, float(upper_tops.max()))
        ax.bar(
            x_centers[valid] + offset,
            points_pct[valid],
            width=bar_width * 0.92,
            color=palette[clf],
            edgecolor="none",
            linewidth=0,
            label=label_for[clf],
        )
        ax.errorbar(
            x_centers[valid] + offset,
            points_pct[valid],
            yerr=yerr[:, valid],
            fmt="none",
            ecolor="0.15",
            elinewidth=theme.scale_stroke(0.6),
            capsize=theme.scale_stroke(1.5),
            capthick=theme.scale_stroke(0.6),
        )
    ax.yaxis.grid(
        True, color=theme.grid_color(),
        linewidth=theme.scale_stroke(0.5), zorder=0,
    )
    ax.set_axisbelow(True)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(classes, rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel(
        ylabel if ylabel is not None
        else _default_metric_label(metric, list(df["class"].astype(str).unique()))
    )
    ax.set_title(title, loc="left")
    if ylim is not None:
        # If whiskers (already clipped to ≤100) reach the requested
        # ymax, extend it by ~3% so the cap line sits visibly inside
        # the axis frame instead of being painted on the edge.
        y_min, y_max = ylim
        if np.isfinite(max_whisker_top) and max_whisker_top >= y_max - 1e-6:
            y_max = max_whisker_top + 0.03 * max(1.0, y_max - y_min)
        ax.set_ylim(y_min, y_max)
    else:
        # Default unzoomed range for clustered per-class plots.
        ax.set_ylim(0.0, 102.0)
    ax.legend(
        loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False, title="Classifier"
    )
    fig.tight_layout()
    return fig
