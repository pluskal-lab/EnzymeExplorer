"""Delta-AP forest plot and p-value heatmap for v4 paired bootstrap output.

* **Box** spans the inner 25–75 % percentile range of the bootstrap
  distribution.
* **Whiskers** end at the chosen CI bounds (normal / percentile / BCa).
* **Median line** inside the box marks the median of the distribution.
* No point markers — the box and whiskers carry all the information.

Layouts:
  * Single class (TPS, Substrate_mAP, …) — one panel, one box per pair.
  * Multi-class scenarios — single panel with method pairs on the
    x-axis and substrate classes as the hue (one box per (pair, class)).

The p-value heatmap shows ``-log10(p_adjusted)`` on a colour scale that
flags the α=0.05 threshold.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme


_DEFAULT_FIG_HEIGHT = 4.5


# ---------------------------------------------------------------------------
# Delta forest plot
# ---------------------------------------------------------------------------


def _short_method(c: str, xtick_overrides: Mapping[str, str] | None = None) -> str:
    if xtick_overrides and c in xtick_overrides:
        return xtick_overrides[c]
    return theme.display_name(c)


def _pair_label(
    a: str, b: str,
    xtick_overrides: Mapping[str, str] | None = None,
    shared_a: bool = False,
) -> str:
    """Multi-line tick label.

    When every pair on the axis shares the same ``classifier_a`` (i.e.
    target-model mode) we drop it from each tick label and the caller
    puts a ``vs <target>`` annotation outside the data area instead.
    Words are split onto separate lines so long labels stack
    vertically and never collide horizontally with neighbouring ticks.
    """
    if shared_a:
        one_line = _short_method(b, xtick_overrides).replace("\n", " ")
    else:
        one_line = (
            f"{_short_method(a, xtick_overrides)} − "
            f"{_short_method(b, xtick_overrides)}"
        ).replace("\n", " ")
    return "\n".join(one_line.split())


def _pair_palette(
    pairs: list[tuple[str, str]],
    *,
    shared_a: bool = False,
) -> dict[tuple[str, str], tuple]:
    """Per-pair colour. With ``shared_a`` (target-mode), each pair is
    coloured by its baseline so the same baseline keeps its identity
    colour across plots; otherwise pairs get distinct tab10 hues."""
    if shared_a:
        out: dict[tuple[str, str], tuple] = {}
        cmap = plt.get_cmap("tab10")
        fallback_idx = 0
        for p in pairs:
            baseline = p[1]
            colour = theme.UNIVERSAL_PALETTE.get(baseline)
            if colour is None:
                colour = cmap(fallback_idx % 10)
                fallback_idx += 1
            else:
                colour = mpl.colors.to_rgba(colour)
            out[p] = colour
        return out
    cmap = plt.get_cmap("tab10")
    return {p: cmap(i % 10) for i, p in enumerate(pairs)}


def _bxp_stat_from_distribution(
    d: np.ndarray, ci_low: float, ci_high: float
) -> dict:
    """Produce a ``ax.bxp`` stat dict where the box spans Q25–Q75 of the
    bootstrap distribution and the whiskers extend to the chosen
    CI bounds. All values are in percentage units."""
    return {
        "med": float(np.median(d)) * 100,
        "q1": float(np.quantile(d, 0.25)) * 100,
        "q3": float(np.quantile(d, 0.75)) * 100,
        "whislo": float(ci_low) * 100,
        "whishi": float(ci_high) * 100,
        "fliers": [],
        "label": "",
    }


def _style_box(bxp, colours: list, alpha: float = 0.75) -> None:
    for patch, c in zip(bxp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
        patch.set_alpha(alpha)
    for whisk in bxp["whiskers"]:
        whisk.set_color("black")
    for cap in bxp["caps"]:
        cap.set_color("black")
    for med in bxp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.2)


def _delta_metric_label(metric: str, classes: list[str]) -> str:
    """Pick the label suffix for the y-axis. ``mAP`` if every class is a
    per-class aggregate (suffix ``_mAP``), otherwise ``AP`` / ``ROC AUC``
    based on the metric column."""
    if metric == "ap" and classes and all(c.endswith("_mAP") for c in classes):
        return "mAP"
    if metric == "ap":
        return "AP"
    if metric == "roc_auc":
        return "ROC AUC"
    return metric.upper()


def _short_class_label(cls: str) -> str:
    """Strip the ``Substrate_`` / trailing ``_mAP`` decoration so the
    legend stays compact (``Substrate_mAP`` → ``Substrate``)."""
    if cls.endswith("_mAP"):
        return cls[: -len("_mAP")].replace("Substrate_", "")
    return cls


def plot_delta_forest(
    long_delta: pd.DataFrame,
    summary_delta: pd.DataFrame,
    *,
    classes: list[str],
    metric: str,
    ap_type: str,
    ci_method: str,
    title: str | None = None,
    xtick_overrides: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    grouped: bool = False,
    pair_order: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    """Box-and-whisker plot of paired delta distributions.

    All pairs are always shown — no transitive-reduction pruning. Box
    spans the Q25–Q75 percentile of the bootstrap distribution;
    whiskers stretch to the (``ap_type``, ``ci_method``) CI ends from
    ``summary_delta``; median line inside box; horizontal zero
    reference line.

    Layout:
      * ``grouped=False`` (default): one panel per class for multi-
        class scenarios; single panel for single-class scenarios. One
        box per pair on the x-axis, pair name as x-tick label.
      * ``grouped=True``: a single panel with method pairs on the
        x-axis and one box per class within each pair-group, classes
        encoded as the legend hue. Recommended for the
        ``substrate_per_class`` scenario where all classes can fit on
        one canvas.
    """
    long_delta = long_delta[
        (long_delta["metric"] == metric)
        & (long_delta["ap_type"] == ap_type)
        & (long_delta["class"].isin(classes))
    ].copy()
    summary_delta = summary_delta[
        (summary_delta["metric"] == metric)
        & (summary_delta["ap_type"] == ap_type)
        & (summary_delta["method"] == ci_method)
        & (summary_delta["class"].isin(classes))
    ].copy()

    discovered_pairs = {
        (a, b) for a, b in summary_delta[
            ["classifier_a", "classifier_b"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    if pair_order is not None:
        all_pairs = [tuple(p) for p in pair_order if tuple(p) in discovered_pairs]
        # Append any pairs the caller forgot to rank, in alphabetical order.
        all_pairs.extend(sorted(discovered_pairs - set(all_pairs)))
    else:
        all_pairs = sorted(discovered_pairs)
    # Detect "every pair has the same classifier_a" (target-model mode):
    # when present, hoist it into a subtitle so each tick can show only
    # the baseline name. Avoids the EnzymeExplorer / − / EnzymeExplorer
    # / − pile-up that wrecks the axis at 5+ pairs.
    shared_a = (
        len({p[0] for p in all_pairs}) == 1 and len(all_pairs) > 1
    )
    target_label = (
        _short_method(all_pairs[0][0], xtick_overrides) if shared_a else None
    )
    pair_labels = [
        _pair_label(a, b, xtick_overrides, shared_a=shared_a) for a, b in all_pairs
    ]
    n_pairs = len(all_pairs)
    n_classes = len(classes)

    # ----- Grouped layout: one panel, x = pair, hue = class -----
    if grouped and n_classes > 1:
        cmap = plt.get_cmap("tab10")
        class_colours = {cls: cmap(i % 10) for i, cls in enumerate(classes)}
        # Per-pair cluster spans 0.88 of the unit interval (gap = 0.12
        # between consecutive pair groups). Each class gets ``bar_w``
        # within the cluster; the rendered box is half of that so
        # boxes are slim and adjacent classes leave a visible gap.
        bar_w = 0.88 / max(n_classes, 1)
        bxp_stats: list[dict] = []
        bxp_positions: list[float] = []
        bxp_colours: list = []
        for pi, pair in enumerate(all_pairs):
            a, b = pair
            for ci, cls in enumerate(classes):
                cls_long = long_delta[
                    (long_delta["class"] == cls)
                    & (long_delta["classifier_a"] == a)
                    & (long_delta["classifier_b"] == b)
                ]
                cls_sum = summary_delta[
                    (summary_delta["class"] == cls)
                    & (summary_delta["classifier_a"] == a)
                    & (summary_delta["classifier_b"] == b)
                ]
                if cls_long.empty or cls_sum.empty:
                    continue
                d = cls_long["value"].dropna().to_numpy(dtype=np.float64)
                if d.size == 0:
                    continue
                srow = cls_sum.iloc[0]
                bxp_stats.append(
                    _bxp_stat_from_distribution(d, srow["ci_low"], srow["ci_high"])
                )
                # Centre the cluster of n_classes boxes around pair index pi.
                offset = (ci - (n_classes - 1) / 2.0) * bar_w
                bxp_positions.append(pi + offset)
                bxp_colours.append(class_colours[cls])

        if figsize is None:
            per_pair = max(0.9, 0.32 * n_classes)
            figsize = (max(3.6, per_pair * n_pairs + 1.6), 3.2)
        fig, ax = plt.subplots(figsize=figsize)
        if bxp_stats:
            bxp = ax.bxp(
                bxp_stats, positions=bxp_positions,
                widths=bar_w * 0.675,
                patch_artist=True, showfliers=False,
            )
            _style_box(bxp, bxp_colours, alpha=0.85)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=0.6)
        ax.set_xticks(list(range(n_pairs)))
        ax.set_xticklabels(pair_labels, rotation=0, ha="center")
        ax.set_xlim(-0.5, n_pairs - 0.5)
        metric_label = _delta_metric_label(metric, classes)
        ax.set_ylabel(f"Δ {metric_label} (%)")
        ax.yaxis.grid(True, color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)
        full_title = title or f"Δ {metric_label}"
        if target_label:
            ax.set_title(
                f"{full_title}   ({target_label} − baseline)",
                fontsize=mpl.rcParams["axes.titlesize"], loc="left",
            )
        else:
            ax.set_title(full_title, loc="left")
        # Class legend on the right.
        legend_handles = [
            mpatches.Patch(
                facecolor=class_colours[cls], edgecolor="black",
                linewidth=0.5, alpha=0.85, label=_short_class_label(cls),
            )
            for cls in classes
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
            frameon=False, title="Class",
        )
        fig.tight_layout(rect=(0, 0, 0.92, 1.0))
        return fig

    # ----- Faceted layout: one panel per class (default) -----
    if n_classes <= 1:
        ncols, nrows = 1, 1
    elif n_classes <= 4:
        ncols, nrows = n_classes, 1
    else:
        ncols = min(4, n_classes)
        nrows = int(np.ceil(n_classes / ncols))

    if figsize is None:
        per_panel_w = max(2.6, 0.55 * n_pairs + 1.0)
        figsize = (per_panel_w * ncols, 2.7 * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, sharey=True,
    )
    flat_axes = axes.ravel()
    fixed_positions = list(range(n_pairs))
    palette = _pair_palette(all_pairs, shared_a=shared_a)

    for cls_i, cls in enumerate(classes):
        ax = flat_axes[cls_i]
        bxp_stats = []
        bxp_positions = []
        colours = []
        for i, pair in enumerate(all_pairs):
            a, b = pair
            cls_long = long_delta[
                (long_delta["class"] == cls)
                & (long_delta["classifier_a"] == a)
                & (long_delta["classifier_b"] == b)
            ]
            cls_sum = summary_delta[
                (summary_delta["class"] == cls)
                & (summary_delta["classifier_a"] == a)
                & (summary_delta["classifier_b"] == b)
            ]
            if cls_long.empty or cls_sum.empty:
                continue
            d = cls_long["value"].dropna().to_numpy(dtype=np.float64)
            if d.size == 0:
                continue
            srow = cls_sum.iloc[0]
            bxp_stats.append(
                _bxp_stat_from_distribution(d, srow["ci_low"], srow["ci_high"])
            )
            bxp_positions.append(i)
            colours.append(palette[pair])
        if bxp_stats:
            bxp = ax.bxp(
                bxp_stats, positions=bxp_positions, widths=0.30,
                patch_artist=True, showfliers=False,
            )
            _style_box(bxp, colours)

        ax.axhline(0, color="0.4", linestyle="--", linewidth=0.6)
        ax.set_xticks(fixed_positions)
        ax.set_xticklabels(pair_labels, rotation=0, ha="center")
        ax.set_xlim(-0.6, n_pairs - 0.4)
        ax.set_title(cls if n_classes > 1 else "", loc="left")
        ax.yaxis.grid(True, color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)

    for k in range(n_classes, len(flat_axes)):
        flat_axes[k].set_visible(False)

    metric_label = _delta_metric_label(metric, classes)
    full_title = title or f"Δ {metric_label}"
    if target_label:
        full_title = f"{full_title}   ({target_label} − baseline)"
    fig.suptitle(full_title, x=0.02, ha="left",
                 fontsize=mpl.rcParams["axes.titlesize"])
    fig.supylabel(f"Δ {metric_label} (%)")
    fig.tight_layout(rect=(0, 0, 1.0, 0.94))
    return fig


# ---------------------------------------------------------------------------
# P-value heatmap (class-faceted)
# ---------------------------------------------------------------------------


def _plot_pvalue_lollipop(
    pv: pd.DataFrame,
    *,
    classes: list[str],
    target: str,
    metric: str,
    ap_type: str,
    title: str | None,
    xtick_overrides: Mapping[str, str] | None,
    figsize: tuple[float, float] | None,
) -> plt.Figure:
    """Lollipop strip — one panel per class, baselines on x, points
    coloured by significance band (>0.05 grey, ≤0.05 light, ≤0.01 mid,
    ≤0.001 dark). One Holm-adjusted p-value per (class, baseline)."""
    classes = [c for c in classes if c in set(pv["class"])]
    n_classes = len(classes)
    if n_classes == 0:
        fig, ax = plt.subplots(figsize=(3.0, 1.5))
        ax.axis("off")
        return fig

    # Order baselines globally by mean p-value across classes (most
    # significant on the right, "easiest" on the left).
    baseline_score = (
        pv.groupby("classifier_b")["p_adjusted"].mean().sort_values(ascending=False)
    )
    baselines = list(baseline_score.index)
    short_b = [_short_method(c, xtick_overrides).replace("\n", " ") for c in baselines]

    # Layout: stack vertically when ≤3 classes, otherwise reflow to a
    # 2-row grid so a tall panel set (e.g. 8 substrate classes) doesn't
    # become an awkwardly tall strip.
    if n_classes <= 3:
        ncols, nrows = 1, n_classes
    else:
        ncols = int(np.ceil(n_classes / 2))
        nrows = 2
    if figsize is None:
        per_panel_w = max(2.4, 0.45 * len(baselines) + 1.0)
        per_panel_h = 1.2
        figsize = (per_panel_w * ncols, 0.6 + per_panel_h * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()

    bands = [
        (0.001, "#67000D"),  # ***
        (0.01,  "#CB181D"),  # **
        (0.05,  "#FB6A4A"),  # *
        (1.01,  "0.65"),     # n.s.
    ]

    def _band_colour(p: float) -> str:
        for thr, col in bands:
            if p <= thr:
                return col
        return "0.65"

    target_pretty = _short_method(target, xtick_overrides).replace("\n", " ")
    for ax, cls in zip(flat, classes):
        sub = pv[pv["class"] == cls].set_index("classifier_b")
        ps = [float(sub["p_adjusted"].get(b, np.nan)) for b in baselines]
        scores = [-np.log10(max(p, 1e-300)) if np.isfinite(p) else np.nan for p in ps]
        x = np.arange(len(baselines), dtype=float)
        for xi, p, s in zip(x, ps, scores):
            if not np.isfinite(s):
                continue
            ax.vlines(xi, 0, s, colors=_band_colour(p), linewidth=1.4)
            ax.scatter(xi, s, s=22, color=_band_colour(p), zorder=3,
                       edgecolor="0.15", linewidth=0.4)
        ax.axhline(-np.log10(0.05), color="0.6", linestyle=":", linewidth=0.6)
        ax.axhline(-np.log10(0.01), color="0.4", linestyle=":", linewidth=0.6)
        ax.set_xlim(-0.5, len(baselines) - 0.5)
        finite_scores = [s for s in scores if np.isfinite(s)]
        if finite_scores:
            ax.set_ylim(0, max(2.5, max(finite_scores) * 1.15))
        else:
            ax.set_ylim(0, 2.5)
        ax.yaxis.grid(True, color="0.92", linewidth=0.4)
        ax.set_axisbelow(True)
        if n_classes > 1:
            ax.set_title(cls, loc="left",
                         fontsize=mpl.rcParams["axes.labelsize"])

    # Hide unused panels in the reflowed grid.
    for k in range(n_classes, len(flat)):
        flat[k].set_visible(False)
    # Only the leftmost column gets the y-axis label.
    for r in range(nrows):
        for c in range(ncols):
            ax_rc = axes[r, c]
            if c == 0:
                ax_rc.set_ylabel("−log₁₀ p", labelpad=2)
            else:
                ax_rc.set_ylabel("")
    # x-tick labels on the bottom row of every visible column.
    for c in range(ncols):
        # Find lowest visible axis in column.
        for r in range(nrows - 1, -1, -1):
            ax_rc = axes[r, c]
            if ax_rc.get_visible():
                ax_rc.set_xticks(np.arange(len(baselines)))
                ax_rc.set_xticklabels(short_b, rotation=0, ha="center")
                break

    full_title = title or "Holm-adjusted paired p-values"
    fig.suptitle(
        f"{full_title}   ({target_pretty} vs baseline)",
        x=0.02, ha="left",
        fontsize=mpl.rcParams["axes.titlesize"],
    )
    fig.tight_layout(rect=(0, 0, 1.0, 0.94))
    return fig


def plot_pvalue_heatmap(
    pvalues: pd.DataFrame,
    *,
    classes: list[str],
    classifiers: list[str],
    metric: str,
    ap_type: str,
    title: str | None = None,
    xtick_overrides: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Per-class heatmap of -log10(p_adjusted) over the method-pair grid.

    Every pair is shown — no transitive-reduction pruning. When the
    pairs collapse to a single target classifier (target-model mode)
    we render a 1-D lollipop strip per class instead of a mostly-empty
    2-D heatmap.
    """
    pv = pvalues[
        (pvalues["metric"] == metric)
        & (pvalues["ap_type"] == ap_type)
        & (pvalues["class"].isin(classes))
    ].copy()

    pair_classifiers = pd.concat([pv["classifier_a"], pv["classifier_b"]])
    a_only = set(pv["classifier_a"].unique())
    if len(a_only) == 1 and not pv.empty:
        return _plot_pvalue_lollipop(
            pv,
            classes=classes,
            target=next(iter(a_only)),
            metric=metric,
            ap_type=ap_type,
            title=title,
            xtick_overrides=xtick_overrides,
            figsize=figsize,
        )

    n_classes = len(classes)
    if n_classes <= 1:
        ncols, nrows = 1, 1
    elif n_classes <= 4:
        ncols, nrows = n_classes, 1
    else:
        ncols = min(4, n_classes)
        nrows = int(np.ceil(n_classes / ncols))
    n_clf = len(classifiers)
    if figsize is None:
        per_panel = max(4.0, 0.55 * n_clf + 1.5)
        figsize = (per_panel * ncols, per_panel * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    def _short(c: str) -> str:
        if xtick_overrides and c in xtick_overrides:
            return xtick_overrides[c]
        return theme.display_name(c)

    short_labels = [_short(c) for c in classifiers]
    clf_index = {c: i for i, c in enumerate(classifiers)}

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("0.85")

    # Range for -log10(p_adj). 1.3 ≈ 0.05; 2 = 0.01; 3 = 0.001.
    vmin, vmax = 0.0, 4.0

    last_im = None
    for cls_i, cls in enumerate(classes):
        ax = flat[cls_i]
        mat = np.full((n_clf, n_clf), np.nan, dtype=np.float64)
        for _, row in pv[pv["class"] == cls].iterrows():
            a = row["classifier_a"]; b = row["classifier_b"]
            if a not in clf_index or b not in clf_index:
                continue
            ia, ib = clf_index[a], clf_index[b]
            score = -np.log10(max(row["p_adjusted"], 1e-300))
            mat[ia, ib] = score
            mat[ib, ia] = score  # symmetric

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        last_im = im
        ax.set_xticks(range(n_clf))
        ax.set_yticks(range(n_clf))
        ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(cls if n_classes > 1 else "", fontsize=11)
        # Annotate finite cells.
        for ii in range(n_clf):
            for jj in range(n_clf):
                v = mat[ii, jj]
                if np.isfinite(v):
                    ax.text(
                        jj, ii, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if v > 2 else "black",
                    )

    for k in range(n_classes, len(flat)):
        flat[k].set_visible(False)

    fig.suptitle(title or "p-values", fontsize=12)
    # Tight-layout the panel grid first, then add a dedicated axis for
    # the colorbar on the right-hand strip so it never overlaps the
    # subplots' xtick labels.
    fig.tight_layout(rect=(0, 0, 0.90, 0.96))
    if last_im is not None:
        cax = fig.add_axes((0.92, 0.18, 0.018, 0.66))
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("-log10(p_adjusted)")
        # Threshold lines at 0.05 and 0.01 references.
        cbar.ax.axhline(-np.log10(0.05), color="0.3", linewidth=0.8)
        cbar.ax.axhline(-np.log10(0.01), color="0.3", linewidth=0.8, linestyle="--")
    return fig
