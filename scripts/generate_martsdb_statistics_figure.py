"""Regenerate the MARTS-DB overview figure with current data.

Panel A: stacked bar of TPS counts per functional class, kingdoms stacked,
         with a kingdom-share pie inset.
Panel B: per-class sequence-length histograms, kingdoms stacked (bars stacked bottom-up).

Multi-type enzymes contribute one count to every class they participate
in — i.e. rows are unique ``(Enzyme_marts_ID, OriginalType)`` pairs, not
enzymes.

Outputs: ``outputs/martsdb/stats/martsdb_{tps_counts_bar,kingdom_pie,length_hist}.svg``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("martsdb_stats")


# ---------------------------------------------------------------------------
# Palettes and column layout
# ---------------------------------------------------------------------------

# Same RGB tuples the user supplied for the phylo-tree ring; kept
# consistent across all MARTS-DB figures.
_KINGDOM_RGB: dict[str, tuple[float, float, float]] = {
    "Plants":   (0.00784313725490196, 0.6196078431372619, 0.45098039215686275),
    "Fungi":    (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    "Bacteria": (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    "Animals":  (0.8352941176470589, 0.3686274509803922, 0.0),
    "Protists": (0.8, 0.47058823529411764, 0.7372549019607844),
    "Viruses":  (0.9254901960784314, 0.8823529411764706, 0.2),
    "Archaea":  (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
}
KINGDOM_COLORS: dict[str, str] = {
    k: mpl.colors.to_hex(v) for k, v in _KINGDOM_RGB.items()
}

# Kingdom row order for stacking (bottom → top).
KINGDOM_ORDER = ["Plants", "Fungi", "Bacteria", "Animals", "Protists", "Archaea"]

# Column layout: (raw OriginalType code, pretty label used on axes).
CLASS_ORDER: list[tuple[str, str]] = [
    ("hemi",    "hemi (C5)"),
    ("mono",    "mono (C10)"),
    ("sesq",    "sesqui (C15)"),
    ("di",      "di (C20)"),
    ("sester",  "sester (C25)"),
    ("tri",     "tri (C30)"),
    ("sesquar", "sesquar (C35)"),
    ("tetra",   "tetra (C40)"),
    ("sqs",     "squalene synthase"),
    ("psy",     "phytoene synthase"),
    ("pt",      "IDS"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs() -> pd.DataFrame:
    """Return unique ``(marts_id, OriginalType, kingdom, seq_len)`` rows.

    Multi-type enzymes appear once per (enzyme × type) so they are counted
    in every bar/histogram they participate in.
    """
    df = pd.read_csv(
        "data/martsDB_reactions_2026_02_22_preprocessed.csv",
        low_memory=False,
        usecols=["Enzyme_marts_ID", "Kingdom", "OriginalType",
                 "Aminoacid_sequence"],
    )
    df["Kingdom"] = df["Kingdom"].fillna("Unknown").str.capitalize()
    df["seq_len"] = df["Aminoacid_sequence"].str.len()
    pairs = (
        df.dropna(subset=["OriginalType"])
          .drop_duplicates(subset=["Enzyme_marts_ID", "OriginalType"])
          .loc[:, ["Enzyme_marts_ID", "OriginalType", "Kingdom", "seq_len"]]
    )
    logger.info("Total (enzyme × type) pairs: %d across %d enzymes",
                len(pairs), pairs["Enzyme_marts_ID"].nunique())
    return pairs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _stacked_bar_counts(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    """Panel A left: stacked bar of TPS counts per class × kingdom."""
    keep_kingdoms = [k for k in KINGDOM_ORDER
                     if (pairs["Kingdom"] == k).any()]
    counts = (
        pairs.groupby(["OriginalType", "Kingdom"]).size().unstack(fill_value=0)
    )

    x = np.arange(len(CLASS_ORDER))
    bottoms = np.zeros(len(CLASS_ORDER))
    codes  = [c for c, _ in CLASS_ORDER]
    labels = [lbl for _, lbl in CLASS_ORDER]

    for kingdom in keep_kingdoms:
        vals = np.array([int(counts.loc[c, kingdom])
                         if c in counts.index and kingdom in counts.columns
                         else 0 for c in codes])
        ax.bar(
            x, vals, bottom=bottoms, width=0.72,
            color=KINGDOM_COLORS[kingdom], label=kingdom,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        bottoms += vals

    # Annotate small columns.
    for xi, total in zip(x, bottoms.astype(int)):
        if 0 < total <= 10:
            ax.text(xi, total + max(bottoms) * 0.012, str(total),
                    ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_ylabel("TPS count", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8, width=0.6, length=3)
    ax.tick_params(axis="x", width=0.6, length=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5,
                  color="#cccccc", alpha=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.78, 0.98),
        frameon=True, fontsize=8, title_fontsize=8,
        edgecolor="#cccccc", framealpha=0.95,
    )


def _kingdom_pie(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    """Panel A inset: overall kingdom-share pie (one count per enzyme,
    NOT per pair — mirrors the old figure)."""
    per_enzyme = pairs.drop_duplicates("Enzyme_marts_ID")
    counts = per_enzyme["Kingdom"].value_counts()
    ordered = [k for k in KINGDOM_ORDER if k in counts.index]
    values = [int(counts[k]) for k in ordered]
    colors = [KINGDOM_COLORS[k] for k in ordered]
    total = sum(values)

    def _fmt(v: float) -> str:
        pct = v
        return f"{pct:.0f}%" if pct >= 3.0 else ""

    wedges, texts, autotexts = ax.pie(
        values, colors=colors,
        autopct=_fmt, pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=0.6),
        startangle=90, counterclock=False,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#222222")
        at.set_fontweight("bold")
    ax.set_aspect("equal")


def _stacked_length_hist(pairs: pd.DataFrame, out_stem: Path) -> None:
    """Length histograms per class, kingdoms stacked bottom-up — one row per class."""
    n_rows = len(CLASS_ORDER)
    fig = plt.figure(figsize=(6.8, 0.55 * n_rows + 1.0))

    x0, w = 0.20, 0.75
    y0, h = 0.10, 0.85
    row_h = h / n_rows

    max_len = int(min(1200, np.nanpercentile(pairs["seq_len"], 99.5)))
    bins = np.linspace(50, max_len, 61)

    keep_kingdoms = [k for k in KINGDOM_ORDER
                     if (pairs["Kingdom"] == k).any()]

    for i, (code, label) in enumerate(CLASS_ORDER):
        ax = fig.add_axes([x0, y0 + h - (i + 1) * row_h, w, row_h])
        sub = pairs[pairs["OriginalType"] == code]
        if len(sub):
            per_kingdom_data = [
                sub.loc[sub["Kingdom"] == k, "seq_len"].to_numpy()
                for k in keep_kingdoms
            ]
            colors = [KINGDOM_COLORS[k] for k in keep_kingdoms]
            ax.hist(
                per_kingdom_data, bins=bins,
                color=colors, label=keep_kingdoms,
                stacked=True, histtype="stepfilled",
                edgecolor="white", linewidth=0.3,
            )
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.set_ylabel(
            label, rotation=0, ha="right", va="center",
            fontsize=8, labelpad=6,
        )
        if i < n_rows - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Length (aa)", fontsize=9)
            ax.tick_params(axis="x", labelsize=8, width=0.6, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    legend_ax = fig.add_axes([x0, 0.005, w, 0.05])
    legend_ax.axis("off")
    handles = [
        mpl.patches.Patch(facecolor=KINGDOM_COLORS[k], label=k,
                          edgecolor="white", linewidth=0.4)
        for k in keep_kingdoms
    ]
    legend_ax.legend(
        handles=handles, loc="center", ncol=len(handles), frameon=False,
        fontsize=9, handlelength=1.4, handleheight=1.0,
        borderpad=0.2, columnspacing=1.8,
    )
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved %s.svg", out_stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/martsdb/stats",
        help="Output directory for the SVGs (default: outputs/martsdb/stats).",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       9,
        "svg.fonttype":    "none",
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()

    # ---- Panel A: stacked bar --------------------------------------
    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.6))
    _stacked_bar_counts(ax_bar, pairs)
    stem = out_dir / "martsdb_tps_counts_bar"
    fig_bar.savefig(stem.with_suffix(".svg"),
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_bar)
    logger.info("Saved %s.svg", stem)

    # ---- Kingdom pie ------------------------------------------------
    fig_pie, ax_pie = plt.subplots(figsize=(2.8, 2.8))
    _kingdom_pie(ax_pie, pairs)
    stem = out_dir / "martsdb_kingdom_pie"
    fig_pie.savefig(stem.with_suffix(".svg"),
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_pie)
    logger.info("Saved %s.svg", stem)

    # ---- Length histograms (kingdoms stacked) -----------------------
    _stacked_length_hist(pairs, out_dir / "martsdb_length_hist")


if __name__ == "__main__":
    main()
