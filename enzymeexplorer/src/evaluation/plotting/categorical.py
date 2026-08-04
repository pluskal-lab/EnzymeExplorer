"""Categorical (per-Kingdom / per-TPS-type) plots driven by a CI summary
that includes a ``category`` column.

Two views on the same data:
  * ``plot_category_boxplot`` — one box per classifier; the box's data
    points are the per-category point estimates (e.g. one AP per kingdom).
    Classifiers are ordered by increasing mean performance across
    categories. Each box gets a distinct colorblind-safe colour, with
    individual category points overlaid as a strip for transparency.
  * ``plot_category_heatmap`` — rows=categories, cols=classifiers, cell
    colour = point estimate. Best for systematic at-a-glance scans.

Both consume the dataframe returned by ``bootstrap.compute_cis`` (which
must include ``category`` in its columns).
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme


def _scale_pct(values: pd.Series) -> pd.Series:
    return values * 100.0


def _filter(
    cis_df: pd.DataFrame,
    target_class: str,
    metric: str,
    *,
    categories: Iterable[str] | None,
    classifier_subset: Iterable[str] | None,
    skip_nan: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if "category" not in cis_df.columns:
        raise ValueError("cis_df must include a 'category' column")
    df = cis_df[
        (cis_df["class"] == target_class) & (cis_df["metric"] == metric)
    ].copy()
    if df.empty:
        raise ValueError(
            f"No rows for class={target_class!r} metric={metric!r}"
        )
    if classifier_subset is not None:
        df = df[df["classifier"].isin(list(classifier_subset))]
    if categories is not None:
        df = df[df["category"].isin(list(categories))]
    if skip_nan:
        df = df.dropna(subset=["point"])
    cats: list[str]
    if categories is None:
        cats = sorted(df["category"].unique())
    else:
        cats = [c for c in categories if c in set(df["category"])]
    return df, cats


def plot_category_boxplot(
    bootstrap_long: pd.DataFrame,
    target_class: str,
    *,
    metric: str = "ap",
    categories: Iterable[str] | None = None,
    classifier_subset: Iterable[str] | None = None,
    classifier_order: list[str] | None = None,
    pin_last: str | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str = "",
    ylabel: str | None = None,
    xlabel: str = "",
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    showfliers: bool = True,
    box_width: float | None = None,      # cluster width per classifier (0..1)
    linewidth: float = 0.5,              # median / whisker / cap
    edge_linewidth: float = 0.45,        # per-box edge stroke
    flier_size: float = 1.5,
    xtick_fontsize: float | None = None,
    ytick_fontsize: float | None = None,
    title_fontsize: float | None = None,
    intra_box_gap: float = 0.0,          # seaborn ``gap`` — space between
                                         # hue-boxes inside a single cluster
                                         # (fraction of a box's width)
) -> plt.Figure:
    """Grouped boxplot: one column per classifier, one box per category
    inside each column. Box colour is by category (Kingdom / TPS type), so
    the same colour identifies the same category across classifiers.

    Each box draws the bootstrap distribution for its (classifier, category)
    cell — whisker IQR + outliers reflect the actual spread of metric values
    over all bootstrap draws. Classifiers are ordered by ascending mean
    performance across categories.

    Pass the ``bootstrap_long.csv`` dataframe (with columns
    ``classifier, class, metric, category, bootstrap_idx, value``).
    """
    if "category" not in bootstrap_long.columns:
        raise ValueError("Expected `bootstrap_long` with a 'category' column")
    df = bootstrap_long[
        (bootstrap_long["class"] == target_class)
        & (bootstrap_long["metric"] == metric)
    ].copy()
    if classifier_subset is not None:
        df = df[df["classifier"].isin(list(classifier_subset))]
    if categories is not None:
        df = df[df["category"].isin(list(categories))]
    df = df.dropna(subset=["value"])
    if df.empty:
        raise ValueError(
            f"No bootstrap rows for class={target_class!r} metric={metric!r}"
        )

    cats: list[str]
    if categories is None:
        cats = sorted(df["category"].unique())
    else:
        cats = [c for c in categories if c in set(df["category"])]

    # Macro-average mean AP across categories, then sort ascending.
    if classifier_order is None:
        classifier_order = list(df["classifier"].unique())
    classifier_order = [c for c in classifier_order if c in set(df["classifier"])]
    macro_means = (
        df.groupby(["classifier", "category"])["value"].mean()
        .groupby("classifier").mean()
    )
    classifier_order = sorted(
        classifier_order, key=lambda c: macro_means.get(c, float("-inf"))
    )
    if pin_last and pin_last in classifier_order:
        classifier_order = [c for c in classifier_order if c != pin_last] + [pin_last]

    if palette is None:
        all_cats = sorted(
            bootstrap_long["category"].dropna().astype(str).unique()
        )
        full_palette = theme.categorical_palette(all_cats)
        palette = {c: full_palette[c] for c in cats}

    df["value_pct"] = df["value"] * 100.0
    df["classifier"] = pd.Categorical(
        df["classifier"], categories=classifier_order, ordered=True
    )
    df["category"] = pd.Categorical(df["category"], categories=cats, ordered=True)

    target_box_inches = 0.10
    cluster_width = 0.66 if box_width is None else float(box_width)
    if figsize is None:
        figsize = (
            max(
                4.6,
                target_box_inches * len(classifier_order) * len(cats) / cluster_width
                + 1.8,
            ),
            3.0,
        )
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x="classifier",
        y="value_pct",
        hue="category",
        order=classifier_order,
        hue_order=cats,
        palette=palette,
        showfliers=showfliers,
        flierprops=dict(
            marker="o", markersize=flier_size,
            markerfacecolor="0.7", markeredgecolor="none",
            linestyle="none", alpha=0.6,
        ),
        whis=(2.5, 97.5),
        linewidth=linewidth,
        width=cluster_width,
        gap=intra_box_gap,
        ax=ax,
    )
    # Box edges should be subtle.
    for patch in ax.patches:
        patch.set_edgecolor("0.2")
        patch.set_linewidth(edge_linewidth)
    ax.margins(x=0.03)
    ax.yaxis.grid(True, color="0.92", linewidth=0.4)
    ax.set_axisbelow(True)

    ax.set_xticks(range(len(classifier_order)))
    ax.set_xticklabels(
        [theme.display_name(clf) for clf in classifier_order],
        rotation=0, ha="center",
        fontsize=xtick_fontsize if xtick_fontsize is not None else None,
    )
    if ytick_fontsize is not None:
        ax.tick_params(axis="y", labelsize=float(ytick_fontsize))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else f"{metric.upper()} (%)")
    ax.set_title(title, loc="left",
                 fontsize=title_fontsize if title_fontsize is not None else None)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncols=min(len(cats), 6),
        frameon=False,
        title=None,
        handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
    )
    fig.tight_layout()
    return fig


def plot_category_heatmap(
    cis_df: pd.DataFrame,
    target_class: str,
    *,
    metric: str = "ap",
    categories: Iterable[str] | None = None,
    classifier_subset: Iterable[str] | None = None,
    classifier_order: list[str] | None = None,
    pin_last: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    annotation_fmt: str = "{:.1f}",
    figsize: tuple[float, float] | None = None,
    title: str = "",
    title_fontsize: float | None = None,
    skip_nan_categories: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Heatmap: rows=categories, cols=classifiers (ordered by mean perf)."""
    df, cats = _filter(
        cis_df, target_class, metric,
        categories=categories,
        classifier_subset=classifier_subset,
        skip_nan=skip_nan_categories,
    )
    if df.empty:
        raise ValueError("No rows match the requested filters")

    # Macro-average and sort, dropping classifiers with no data for this target
    # (e.g. PFAM/SUPFAM on substrate or TPS_IDS classes).
    present = set(df["classifier"])
    if classifier_order is None:
        classifier_order = list(present)
    else:
        classifier_order = [c for c in classifier_order if c in present]
    means = df.groupby("classifier")["point"].mean()
    classifier_order = sorted(
        classifier_order, key=lambda c: means.get(c, float("-inf"))
    )
    if pin_last and pin_last in classifier_order:
        classifier_order = [c for c in classifier_order if c != pin_last] + [pin_last]
    df = df[df["classifier"].isin(classifier_order)]

    matrix = (
        df.pivot(index="category", columns="classifier", values="point")
        .reindex(index=cats, columns=classifier_order)
        * 100.0
    )
    if figsize is None:
        figsize = (
            max(4.5, 0.55 * len(classifier_order) + 1.6),
            max(2.4, 0.42 * len(cats) + 1.2),
        )
    fig, ax = plt.subplots(figsize=figsize)
    norm_vmin = float(np.nanmin(matrix.values)) if vmin is None else vmin
    norm_vmax = float(np.nanmax(matrix.values)) if vmax is None else vmax
    if not np.isfinite(norm_vmin) or not np.isfinite(norm_vmax):
        norm_vmin, norm_vmax = 0.0, 100.0
    im = ax.imshow(
        matrix.values,
        cmap=cmap,
        aspect="auto",
        vmin=norm_vmin,
        vmax=norm_vmax,
    )
    ax.set_xticks(np.arange(len(classifier_order)))
    ax.set_xticklabels(
        [theme.display_name(c).replace("\n", " ") for c in classifier_order],
        rotation=0, ha="center",
    )
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title(title, loc="left",
                 fontsize=title_fontsize if title_fontsize is not None else None)
    # Hairline spines + faint cell borders so adjacent cells read as a grid.
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.015)
    cbar.set_label(f"{metric.upper()} (%)")
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2.5, width=0.5)

    if annotate:
        # Switch annotation colour at ~55 % through viridis; the split
        # avoids white-on-yellow at the top of the colormap.
        threshold = norm_vmin + 0.55 * (norm_vmax - norm_vmin)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix.values[i, j]
                if not np.isfinite(v):
                    txt = "—"
                    color = "0.4"
                else:
                    txt = annotation_fmt.format(v)
                    color = "white" if v < threshold else "0.1"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7.5)
    fig.tight_layout()
    return fig
