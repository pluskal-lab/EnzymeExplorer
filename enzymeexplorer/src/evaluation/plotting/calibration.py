"""Diagnostic plots for the per-class beta calibration.

All plots accept the long-form DataFrames emitted by
``calibration.fit_calibration_table`` (or any subset filtered to one
``(classifier, target_class)`` pair) and use ``theme.apply_theme`` so
figures land in Nature Chem Bio formatting with no restyling.

Six plot types:

* :func:`plot_reliability_diagram`        — empirical vs. predicted, Wilson bars
* :func:`plot_calibration_curve_with_ribbon` — p̂(s) with cluster-bootstrap CI
* :func:`plot_score_distribution`         — OOF score histogram by label
* :func:`plot_per_fold_params`            — (a, b, c) across LOFO holdouts
* :func:`plot_hard_errors`                — top-K FP / bottom-K FN on p̂ axis
* :func:`plot_calibration_metrics_grid`   — Δ log-loss/Brier/ECE summary heatmap
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.plotting import theme


_POS_COLOR = "#d9544d"
_NEG_COLOR = "#6f6f6f"
_FIT_COLOR = "#1a78b4"
_BAND_COLOR = "#1a78b4"


# ---------------------------------------------------------------------------
# 1. Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    reliability_df: pd.DataFrame,
    *,
    classifier: str,
    target_class: str,
    n_pos: int | None = None,
    n_total: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (4.2, 4.2),
) -> plt.Figure:
    """Per-bin empirical positive rate vs. mean predicted probability.

    ``reliability_df`` should be the rows for one (clf, class). Each bin
    becomes a marker sized by ``n``; vertical error bars show binomial
    Wilson 95%. Diagonal reference line for perfect calibration.
    """
    theme.apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    if reliability_df is None or reliability_df.empty:
        ax.text(0.5, 0.5, "no LOFO predictions", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df = reliability_df.copy()
    x = df["p_pred_mean"].to_numpy()
    y = df["p_obs"].to_numpy()
    yerr_lo = np.clip(y - df["wilson_lo"].to_numpy(), 0.0, None)
    yerr_hi = np.clip(df["wilson_hi"].to_numpy() - y, 0.0, None)
    n_max = max(int(df["n"].max()), 1)
    sizes = (df["n"].to_numpy() / n_max) * 90.0 + 14.0

    ax.plot([0.0, 1.0], [0.0, 1.0],
            color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi],
                fmt="none", color="0.3", alpha=0.7,
                elinewidth=0.8, capsize=2.0, zorder=2)
    ax.scatter(x, y, s=sizes, color=_FIT_COLOR,
               edgecolor="0.15", linewidth=0.5, zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Empirical positive rate")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.6)

    if title is None:
        title = f"{theme.display_name(classifier)} — {target_class}: reliability"
        if n_pos is not None and n_total is not None:
            title += f"  (n={n_total}, n+={n_pos})"
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Calibration curve with cluster-bootstrap CI ribbon
# ---------------------------------------------------------------------------

def plot_calibration_curve_with_ribbon(
    ribbon_df: pd.DataFrame,
    oof_df: pd.DataFrame | None,
    *,
    classifier: str,
    target_class: str,
    title: str | None = None,
    figsize: tuple[float, float] = (5.6, 4.4),
) -> plt.Figure:
    """Calibrated probability vs. raw score: deployment fit + bootstrap ribbon.

    Adds rugs at the top/bottom showing OOF score density for positives /
    negatives so the reader sees where the model has earned its claim.
    """
    theme.apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    if ribbon_df is None or ribbon_df.empty:
        ax.text(0.5, 0.5, "no calibration ribbon", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    rib = ribbon_df.sort_values("score")
    s = rib["score"].to_numpy()
    p_lo = rib["p_lo"].to_numpy()
    p_hi = rib["p_hi"].to_numpy()
    p_hat = rib["p_hat"].to_numpy()

    if not np.all(np.isnan(p_lo)):
        ax.fill_between(s, p_lo, p_hi, color=_BAND_COLOR, alpha=0.18,
                        linewidth=0, label="95% CI (cluster bootstrap)")
    ax.plot(s, p_hat, color=_FIT_COLOR, linewidth=1.5,
            zorder=2, label="deployment fit p̂(s)")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--",
            linewidth=0.7, label="identity (raw=calibrated)")

    if oof_df is not None and not oof_df.empty:
        pos = oof_df.loc[oof_df["label"] == 1, "score"].to_numpy()
        neg = oof_df.loc[oof_df["label"] == 0, "score"].to_numpy()
        ax.scatter(pos, np.full_like(pos, 1.03), marker="|",
                   color=_POS_COLOR, s=18, alpha=0.7, clip_on=False,
                   linewidths=0.7)
        ax.scatter(neg, np.full_like(neg, -0.03), marker="|",
                   color=_NEG_COLOR, s=14, alpha=0.45, clip_on=False,
                   linewidths=0.5)

    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("Raw score s")
    ax.set_ylabel("Calibrated probability p̂(s)")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.6)

    if title is None:
        title = (
            f"{theme.display_name(classifier)} — {target_class}: "
            "calibration curve"
        )
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Score distribution
# ---------------------------------------------------------------------------

def plot_score_distribution(
    oof_df: pd.DataFrame,
    *,
    classifier: str,
    target_class: str,
    n_bins: int = 40,
    title: str | None = None,
    figsize: tuple[float, float] = (5.4, 3.2),
) -> plt.Figure:
    """Two-color histogram of OOF scores by label.

    Makes the RF edge pile-up visible. Pairs with the calibration curve:
    sparse score regions in this histogram should align with wide ribbon
    regions on the curve.
    """
    theme.apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    if oof_df is None or oof_df.empty:
        ax.text(0.5, 0.5, "no OOF data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    pos = oof_df.loc[oof_df["label"] == 1, "score"].to_numpy()
    neg = oof_df.loc[oof_df["label"] == 0, "score"].to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ax.hist(neg, bins=bins, color=_NEG_COLOR, alpha=0.65,
            label=f"negatives (n={len(neg)})", edgecolor="none")
    ax.hist(pos, bins=bins, color=_POS_COLOR, alpha=0.75,
            label=f"positives (n={len(pos)})", edgecolor="none")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(-0.005, 1.005)
    ax.set_xlabel("Raw score s")
    ax.set_ylabel("Count (symlog)")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    if title is None:
        title = (
            f"{theme.display_name(classifier)} — {target_class}: "
            "OOF score distribution"
        )
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Per-fold params (drift)
# ---------------------------------------------------------------------------

_FAMILY_PARAM_LABELS: dict[str, list[tuple[str, str, str]]] = {
    # family -> [(column, display_label, color), ...]
    "beta": [
        ("a", "a (log s)",       "#1a78b4"),
        ("b", "b (-log(1-s))",   "#1ca964"),
        ("c", "c (intercept)",   "#d9544d"),
    ],
    "logit_platt": [
        ("a", "a (logit s)",     "#1a78b4"),
        ("b", "b (intercept)",   "#d9544d"),
    ],
    "temperature": [
        ("T", "T (temperature)", "#1a78b4"),
    ],
    "identity": [],
}


def plot_per_fold_params(
    per_fold_df: pd.DataFrame,
    *,
    classifier: str,
    target_class: str,
    family: str,
    drift_flagged: bool = False,
    spreads: dict[str, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (5.4, 2.8),
) -> plt.Figure:
    """Calibrator parameters across LOFO holdouts (family-aware).

    Bars for the family's relevant params per held-out fold. If
    ``drift_flagged``, the title is annotated with the spread metric.
    For the identity family there are no params; the figure carries an
    explanatory note instead.
    """
    theme.apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    spec = _FAMILY_PARAM_LABELS.get(family, [])
    if not spec:
        ax.text(0.5, 0.5,
                f"family={family!r} has no per-fold parameters",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    if per_fold_df is None or per_fold_df.empty:
        ax.text(0.5, 0.5, "no per-fold parameters",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df = per_fold_df.sort_values("held_out_fold")
    folds = df["held_out_fold"].astype(int).tolist()
    x = np.arange(len(folds))
    n = len(spec)
    total_width = 0.85
    width = total_width / n
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for (col, lbl, color), off in zip(spec, offsets):
        if col not in df.columns:
            continue
        ax.bar(x + off, df[col].to_numpy(), width=width, label=lbl, color=color)
    ax.axhline(0, color="0.4", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"holdout={f}" for f in folds])
    ax.set_ylabel(f"{family} calibrator parameter")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    if title is None:
        title = (
            f"{theme.display_name(classifier)} — {target_class}: "
            f"per-fold params [{family}]"
        )
        if drift_flagged and spreads:
            sp = ", ".join(
                f"{k}={v:.2f}" for k, v in spreads.items()
                if v is not None and not np.isnan(v)
            )
            title += f"  ⚠ DRIFT ({sp})"
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=max(1, n))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Hard errors on the calibrated probability axis
# ---------------------------------------------------------------------------

def plot_hard_errors(
    hard_df: pd.DataFrame,
    *,
    classifier: str,
    target_class: str,
    title: str | None = None,
    figsize: tuple[float, float] = (6.4, 3.2),
) -> plt.Figure:
    """Top-K FPs and bottom-K FNs marked on the calibrated probability axis."""
    theme.apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    if hard_df is None or hard_df.empty:
        ax.text(0.5, 0.5, "no hard errors", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    fps = hard_df[hard_df["kind"] == "FP"]
    fns = hard_df[hard_df["kind"] == "FN"]
    ax.scatter(fps["p_lofo"], np.full(len(fps), 1.0), marker="v",
               s=24, color=_POS_COLOR, edgecolor="0.15", linewidth=0.4,
               zorder=2, label=f"top-{len(fps)} FPs (label=0, high p̂)")
    ax.scatter(fns["p_lofo"], np.full(len(fns), 0.0), marker="^",
               s=24, color=_FIT_COLOR, edgecolor="0.15", linewidth=0.4,
               zorder=2, label=f"bottom-{len(fns)} FNs (label=1, low p̂)")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["FN (label=1)", "FP (label=0)"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-0.005, 1.005)
    ax.set_xlabel("Calibrated probability p̂(s)")
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)

    if title is None:
        title = (
            f"{theme.display_name(classifier)} — {target_class}: "
            "hardest errors on calibrated axis"
        )
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.34),
              ncol=2, borderaxespad=0, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Calibration metrics summary heatmap
# ---------------------------------------------------------------------------

def plot_calibration_metrics_grid(
    metrics_df: pd.DataFrame,
    *,
    class_order: Sequence[str] | None = None,
    classifier_order: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of Δ log-loss / Δ Brier / Δ ECE per (clf, class).

    Δ = raw − calibrated. Positive (green) means calibration helped.
    """
    theme.apply_theme()
    if metrics_df is None or metrics_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.0))
        ax.text(0.5, 0.5, "no metrics", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df = metrics_df.copy()
    df["d_log_loss"] = df["log_loss_raw"] - df["log_loss_cal"]
    df["d_brier"] = df["brier_raw"] - df["brier_cal"]
    df["d_ece"] = df["ece_raw"] - df["ece_cal"]

    classes = list(class_order) if class_order is not None else (
        sorted(df["target_class"].unique())
    )
    clfs = list(classifier_order) if classifier_order is not None else (
        sorted(df["classifier"].unique())
    )

    def _pivot(col: str) -> pd.DataFrame:
        return (
            df.pivot(index="target_class", columns="classifier", values=col)
            .reindex(index=classes, columns=clfs)
        )

    panels = [
        ("Δ log-loss", _pivot("d_log_loss")),
        ("Δ Brier",    _pivot("d_brier")),
        ("Δ ECE",      _pivot("d_ece")),
    ]
    if figsize is None:
        figsize = (3.0 + 1.1 * len(clfs) * 3, 1.6 + 0.4 * len(classes))
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (label, pivot) in zip(axes, panels):
        vals = pivot.to_numpy()
        if np.all(np.isnan(vals)):
            ax.text(0.5, 0.5, f"no {label} data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        im = ax.imshow(vals, cmap="RdYlGn", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [theme.display_name(c) for c in pivot.columns],
            rotation=30, ha="right",
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(label, loc="left")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.3f}",
                        ha="center", va="center", fontsize=6.5,
                        color="0.1")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    if title:
        fig.suptitle(title, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)
    return fig
