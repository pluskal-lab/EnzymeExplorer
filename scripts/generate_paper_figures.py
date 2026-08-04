"""Publication-quality figure regeneration for the domain-clustering paper.

Rebuilds four plot families in both PNG and SVG under
``outputs/domain_clustering/``:

1. ``dendrogram_d0_m3``          — HAC dendrogram, single strip renamed
                                   "Domain Subtype", one-column legend,
                                   dashed background gridlines, thick tree
                                   lines. No kingdom / canonical-type strips.
2. ``metrics_d0_m3``             — 6-panel per-clade metrics.
3. ``reactions_per_clade_d0_m3`` — stacked-bar reaction-type composition
                                   (bug fixed: reaction_labels are now
                                   parsed as proper lists, not iterated as
                                   char strings).
4. ``sweep_overview``            — DTC sweep heatmap (rebuilt from the
                                   existing sweep_summary.csv).

Uses a shared colorblind-safe palette (Okabe-Ito main-type hues +
sequential subtype tones) exposed as ``SUBTYPE_PALETTE`` so every plot
uses the same subtype colors.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # type: ignore  # noqa: E402
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
import seaborn as sns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enzymeexplorer.src.domain_clustering import analysis  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("paper_figures")


# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",  # keep text as text in SVG so editors can restyle
})


# ---------------------------------------------------------------------------
# Shared subtype palette
# ---------------------------------------------------------------------------

# Colorblind-safe main-type hues loaded from the shared palette JSON so
# every Section-3 figure uses the same colors.
def _load_main_type_palette() -> dict[str, str]:
    import json as _json
    with open("data/domain_main_type_palette.json") as _fh:
        raw = _json.load(_fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


MAIN_TYPE_COLORS: dict[str, str] = _load_main_type_palette()

_FAMILY_CMAP: dict[str, str] = {
    "alpha":   "Blues",
    "beta":    "Greens",
    "gamma":   "Oranges",
    "delta":   "RdPu",
    "epsilon": "Reds",
    "zeta":    "Greys",
}


def _main_type_of(subtype: str) -> str | None:
    for main in MAIN_TYPE_COLORS:
        if subtype.startswith(main):
            return main
    return None


# Greek-letter display form: alpha1A → α1A, delta1 → δ1, etc.
# The internal ASCII form ("alpha1A") is the canonical identifier used in
# pkls, JSON, CSV; the Greek form is applied only for plot rendering.
_GREEK_MAP: dict[str, str] = {
    "alpha":   "α",
    "beta":    "β",
    "gamma":   "γ",
    "delta":   "δ",
    "epsilon": "ε",
    "zeta":    "ζ",
}


def to_greek(label: str) -> str:
    """Convert an ASCII subtype label to its Greek-letter display form."""
    if not isinstance(label, str):
        return label
    for prefix, greek in _GREEK_MAP.items():
        if label.startswith(prefix):
            return greek + label[len(prefix):]
    return label


def greek_map(labels: "list[str] | dict[str, str]") -> "list[str] | dict[str, str]":
    """Apply :func:`to_greek` element-wise to a list or the *values* of a dict."""
    if isinstance(labels, dict):
        return {k: to_greek(v) for k, v in labels.items()}
    return [to_greek(x) for x in labels]


def build_subtype_palette(
    subtypes: list[str],
    *,
    order: list[str] | None = None,
) -> dict[str, str]:
    """Return ``{subtype: hex_color}`` for the given subtype list.

    Rules:
      * Subtypes are grouped by main type via prefix match (alpha/beta/…).
      * If a family has a single subtype, it gets the canonical Okabe-Ito
        main-type color (``MAIN_TYPE_COLORS[family]``).
      * If a family has multiple subtypes, they get progressively darker
        tones of the family's sequential colormap (positions 0.35..0.95).

    ``order`` (when supplied) is used to determine tone assignment WITHIN
    each family: the first-appearing member of the family gets the
    lightest tone, the last-appearing gets the darkest. This keeps the
    dendrogram's left-to-right sub-clade progression aligned with the
    colormap. When ``order`` is ``None``, members are sorted
    lexicographically.
    """
    ordering = order if order is not None else sorted(subtypes)
    families: dict[str, list[str]] = {m: [] for m in MAIN_TYPE_COLORS}
    for st in ordering:
        m = _main_type_of(st)
        if m is not None and st not in families[m]:
            families[m].append(st)

    palette: dict[str, str] = {}
    for family, members in families.items():
        if not members:
            continue
        if len(members) == 1:
            palette[members[0]] = MAIN_TYPE_COLORS[family]
            continue
        cmap = mpl.colormaps[_FAMILY_CMAP[family]]
        positions = np.linspace(0.35, 0.95, len(members))
        for st, pos in zip(members, positions):
            palette[st] = mpl.colors.to_hex(cmap(pos))
    return palette


def dendrogram_leaf_subtype_order(
    linkage_matrix: np.ndarray, member_ids: list[str],
    subtype_map: dict[str, str],
) -> list[str]:
    """Return unique subtypes in the order they first appear in the
    dendrogram (left-to-right leaf display order)."""
    from scipy.cluster.hierarchy import dendrogram

    fig_tmp = plt.figure()
    ddata = dendrogram(linkage_matrix, no_labels=True, no_plot=False,
                       above_threshold_color="k")
    plt.close(fig_tmp)
    leaves = ddata["leaves"]
    order: list[str] = []
    seen: set[str] = set()
    for i in leaves:
        st = subtype_map.get(member_ids[i])
        if st is not None and st not in seen:
            seen.add(st)
            order.append(st)
    return order


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, stem: Path) -> None:
    """Write PNG + SVG under the same stem."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    logger.info("Saved %s.{png,svg}", stem)


# ---------------------------------------------------------------------------
# 1. Dendrogram with Domain Subtype strip
# ---------------------------------------------------------------------------

def plot_dendrogram(
    linkage_matrix: np.ndarray, member_ids: list[str],
    subtype_map: dict[str, str], clusters: dict[str, list[str]],
    label_map: dict[str, str], palette: dict[str, str], out_stem: Path,
    *, tree_linewidth: float = 1.2,
) -> None:
    from scipy.cluster.hierarchy import dendrogram

    n_leaves = len(member_ids)
    leaf_subtype = [subtype_map.get(mid) for mid in member_ids]

    # Branch-color function: monophyletic subtype-color, mixed = grey.
    Z = np.asarray(linkage_matrix)
    dcache: dict[int, list[int]] = {}
    def _descendants(node_id: int) -> list[int]:
        if node_id in dcache:
            return dcache[node_id]
        if node_id < n_leaves:
            dcache[node_id] = [node_id]
            return dcache[node_id]
        row = node_id - n_leaves
        a, b = int(Z[row, 0]), int(Z[row, 1])
        dcache[node_id] = _descendants(a) + _descendants(b)
        return dcache[node_id]

    MIXED = "#c8c8c8"
    def _link_color(node_id: int) -> str:
        sts = {leaf_subtype[i] for i in _descendants(int(node_id))}
        sts.discard(None)
        return palette[next(iter(sts))] if len(sts) == 1 else MIXED

    # Figure layout: tall main dendrogram + narrow strip below + legend to right.
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[14, 3.2], height_ratios=[10, 0.5],
        hspace=0.05, wspace=0.03,
    )
    ax_d = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_l = fig.add_subplot(gs[:, 1]); ax_l.axis("off")

    ddata = dendrogram(
        linkage_matrix, ax=ax_d, no_labels=True,
        link_color_func=_link_color,
    )
    # Thicken every tree line so it survives paper reduction.
    for coll in ax_d.collections:
        coll.set_linewidth(tree_linewidth)
    for line in ax_d.get_lines():
        line.set_linewidth(tree_linewidth)

    leaves = ddata["leaves"]
    n_ord = len(leaves)
    leaf_x = (np.arange(n_ord) + 0.5) * 10

    ax_d.set_ylabel("Structural similarity")
    ax_d.set_ylim(0, 1.02)
    # Data axis is 1 − TM (distance); relabel so tick text shows similarity,
    # i.e. 1.0 at the bottom (leaves) and 0.0 at the top (root).
    _ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax_d.set_yticks(_ticks)
    ax_d.set_yticklabels([f"{1 - t:.1f}" for t in _ticks])
    ax_d.set_xticks([])
    # Dashed horizontal background gridlines.
    ax_d.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.7, zorder=0)
    ax_d.set_axisbelow(True)
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)

    # Domain-subtype strip below the dendrogram.
    for i, leaf_idx in enumerate(leaves):
        st = leaf_subtype[leaf_idx]
        color = palette.get(st, MIXED)
        ax_s.barh(y=0, width=10.0, left=leaf_x[i] - 5, height=1,
                  color=color, edgecolor="none")
    ax_s.set_xlim(0, n_ord * 10)
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([0.5])
    ax_s.set_yticklabels(["Domain Subtype"], fontsize=11)
    ax_s.set_xticks([])
    ax_s.tick_params(left=False)
    for side in ("top", "right", "bottom", "left"):
        ax_s.spines[side].set_visible(False)

    # Legend — one column. Order matches the dendrogram's left-to-right
    # leaf sequence: whichever subtype appears first as you scan the
    # dendrogram from left is the first legend entry, and so on.
    subtype_counts = Counter(leaf_subtype)
    order: list[str] = []
    seen: set[str] = set()
    for i in leaves:
        st = leaf_subtype[i]
        if st is not None and st not in seen:
            seen.add(st)
            order.append(st)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[s], edgecolor="black", linewidth=0.5) for s in order]
    labels = [f"{to_greek(s)} (n = {subtype_counts.get(s, 0)})" for s in order]
    ax_l.legend(
        handles, labels, title="Domain Subtype",
        loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
        ncol=1, fontsize=11, title_fontsize=13,
        handlelength=1.6, handleheight=1.2, borderaxespad=0,
        labelspacing=0.55,
    )

    ax_d.set_title("Hierarchical clustering of martsDB structural domains", pad=10)
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# 2. Metrics panel (6 subplots)
# ---------------------------------------------------------------------------

def plot_metrics(
    clade_table: pd.DataFrame, label_map_dtc_to_subtype: dict[str, str],
    palette: dict[str, str], out_stem: Path,
    *, bootstrap_threshold: float = 0.85,
) -> None:
    df = clade_table.copy()
    if "is_unassigned" in df.columns:
        df = df[~df["is_unassigned"]]
    # Rename clade_ids to subtype labels.
    df["subtype"] = df["clade_id"].map(lambda c: label_map_dtc_to_subtype.get(c, c))
    # Columns ordered by subtype label alphabetically (not by size).
    df = df.sort_values("subtype", ascending=True).reset_index(drop=True)
    if df.empty:
        return

    x = np.arange(len(df))
    subtypes = df["subtype"].tolist()
    subtypes_greek = [to_greek(s) for s in subtypes]
    colors = [palette.get(s, "#888888") for s in subtypes]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))

    def _bars(ax, values, ylabel, title, ylim=None, yscale="linear"):
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        if yscale != "linear": ax.set_yscale(yscale)
        ax.set_xticks(x)
        ax.set_xticklabels(subtypes_greek, rotation=90, fontsize=9)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # 1) Cluster sizes (log)
    _bars(axes[0, 0], df["n"], "Domains per subtype", "Subtype size", yscale="log")

    # 2) Compactness (intra vs inter)
    ax = axes[0, 1]; bw = 0.4
    ax.bar(x - bw/2, df["mean_intra_tm"], bw, color=colors, edgecolor="black", linewidth=0.4, label="Intra-TM")
    ax.bar(x + bw/2, df["mean_inter_tm"], bw, color="#bbbbbb", edgecolor="black", linewidth=0.4, label="Inter-TM")
    ax.set_ylabel("Mean TM-score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Compactness")
    ax.set_xticks(x); ax.set_xticklabels(subtypes_greek, rotation=90, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

    # 3) Kingdom purity
    _bars(axes[0, 2], df["top_kingdom_frac"], "Top-kingdom fraction",
          "Taxonomic purity", ylim=(0, 1.05))

    # 4) Reaction purity
    _bars(axes[1, 0], df["top_reaction_label_frac"], "Top-reaction fraction",
          "Reaction-class purity", ylim=(0, 1.05))

    # 5) Transitional fraction
    trans_max = float(df["transitional_frac"].max()) if "transitional_frac" in df.columns else 0.0
    _bars(axes[1, 1], df.get("transitional_frac", pd.Series([0.0] * len(df))),
          "Boundary fraction",
          "Transitional-domain fraction",
          ylim=(0, max(0.1, trans_max * 1.15)))

    # 6) Bootstrap
    ax = axes[1, 2]
    ax.bar(x, df["bootstrap_support"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Bootstrap support")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cluster reproducibility")
    ax.set_xticks(x); ax.set_xticklabels(subtypes_greek, rotation=90, fontsize=9)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

    fig.suptitle("Structural-subtype quality metrics", fontsize=14, y=1.00, weight="bold")
    fig.tight_layout()
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# 3. Reactions per clade (bug fixed via load_domain_metadata)
# ---------------------------------------------------------------------------

def _stacked_composition(
    stats_df: pd.DataFrame, label_map_dtc_to_subtype: dict[str, str],
    dist_col: str, y_label: str, title: str, legend_title: str,
    out_stem: Path,
) -> None:
    """Shared implementation for the reactions-per-clade and kingdoms-per-clade
    stacked bar plots. Columns are ordered by subtype label (alphabetical)
    for cross-figure consistency with the metrics panel.
    """
    df = stats_df.copy()
    df["subtype"] = df["foldseek_rep"].map(lambda c: label_map_dtc_to_subtype.get(c, c))
    # Alphabetical subtype ordering — consistent with the metrics panel.
    df = df.sort_values("subtype", ascending=True).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        dist = r[dist_col] or {}
        total = sum(dist.values())
        if total == 0:
            continue
        for cat, cnt in dist.items():
            rows.append({"subtype": r["subtype"], "n": int(r["n"]),
                         "category": str(cat), "fraction": cnt / total})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        logger.warning("No %s data — skipping plot %s", dist_col, out_stem)
        return

    subtype_order = df["subtype"].tolist()
    cat_order = sorted(long_df["category"].unique())
    pivot = (
        long_df.pivot_table(index="subtype", columns="category",
                            values="fraction", aggfunc="sum", fill_value=0)
        .reindex(subtype_order)[cat_order]
    )
    cat_palette = sns.color_palette("colorblind", n_colors=max(len(cat_order), 8))

    fig, ax = plt.subplots(figsize=(max(10, len(subtype_order) * 0.7 + 3), 5.5))
    bottom = np.zeros(len(subtype_order))
    for i, cat in enumerate(cat_order):
        vals = pivot[cat].to_numpy()
        ax.bar(range(len(subtype_order)), vals, bottom=bottom,
               color=cat_palette[i % len(cat_palette)], label=cat,
               edgecolor="white", linewidth=0.5)
        bottom += vals

    ns = [int(r["n"]) for _, r in df.iterrows()]
    ax.set_xticks(range(len(subtype_order)))
    ax.set_xticklabels([f"{to_greek(s)}\nn = {n}" for s, n in zip(subtype_order, ns)],
                       rotation=0, fontsize=9)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              title=legend_title, fontsize=10, title_fontsize=11)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, out_stem)


def plot_reactions(
    stats_df: pd.DataFrame, label_map_dtc_to_subtype: dict[str, str],
    palette: dict[str, str], out_stem: Path,
) -> None:
    """Stacked bar of reaction-type composition per subtype.

    ``stats_df`` must have a ``reaction_label_distribution`` column that is
    an actual dict (Counter). See ``main()`` — we call
    ``analysis.load_domain_metadata`` (not read the CSV) to get real
    list-typed ``reaction_labels`` upstream so this dict is real too.
    """
    _stacked_composition(
        stats_df, label_map_dtc_to_subtype,
        dist_col="reaction_label_distribution",
        y_label="Reaction-type fraction",
        title="Reaction-class composition per structural subtype",
        legend_title="Reaction class",
        out_stem=out_stem,
    )


def plot_kingdoms(
    stats_df: pd.DataFrame, label_map_dtc_to_subtype: dict[str, str],
    palette: dict[str, str], out_stem: Path,
) -> None:
    """Stacked bar of kingdom composition per subtype.

    Sibling of :func:`plot_reactions`; columns ordered by subtype label.
    """
    _stacked_composition(
        stats_df, label_map_dtc_to_subtype,
        dist_col="kingdom_distribution",
        y_label="Kingdom fraction",
        title="Kingdom composition per structural subtype",
        legend_title="Kingdom",
        out_stem=out_stem,
    )


# ---------------------------------------------------------------------------
# 4. Transitional-domain drift heatmap
# ---------------------------------------------------------------------------

def plot_transitional_drift(
    subtype_map: dict[str, str],
    member_ids: list[str],
    distance_matrix: np.ndarray,
    out_stem: Path,
    *, margin_threshold: float = 0.05,
) -> None:
    """Heatmap of transitional membership per home subtype.

    For each domain m with home subtype X:
      * compute TM(m, home_medoid)         → ``home_tm``
      * compute TM(m, best other medoid Y) → ``other_tm`` for the closest
        non-X subtype Y
      * if ``home_tm − other_tm >= margin_threshold`` (default 0.05),
        the domain is confidently in X → contributes to the **diagonal**
        cell (X, X).
      * otherwise the domain is transitional → contributes to the
        off-diagonal cell (X, Y).

    Rows are normalised by the total number of members in the home
    subtype, so each row sums to 1. Diagonal = confidently-home fraction;
    off-diagonal = fraction of X's members that sit within ``margin``
    TM of subtype Y's medoid.

    Row/column axes are sorted alphabetically by subtype label.
    """
    mid_to_idx = {m: i for i, m in enumerate(member_ids)}

    members_by_subtype: dict[str, list[str]] = {}
    for m, st in subtype_map.items():
        members_by_subtype.setdefault(st, []).append(m)
    subtypes = sorted(members_by_subtype)

    medoids: dict[str, str] = {}
    for st, members in members_by_subtype.items():
        if len(members) == 1:
            medoids[st] = members[0]
            continue
        idx = np.array([mid_to_idx[m] for m in members], dtype=int)
        sub = distance_matrix[np.ix_(idx, idx)]
        medoids[st] = members[int(np.argmin(sub.mean(axis=1)))]

    medoid_idxs = np.array([mid_to_idx[medoids[st]] for st in subtypes], dtype=int)
    st_to_row = {st: i for i, st in enumerate(subtypes)}

    counts = np.zeros((len(subtypes), len(subtypes)), dtype=float)
    for m, home in subtype_map.items():
        if home not in st_to_row:
            continue
        i_home = st_to_row[home]
        tms = 1.0 - distance_matrix[mid_to_idx[m], medoid_idxs]  # TM to every medoid
        home_tm = tms[i_home]
        # Find best non-home medoid.
        tms_other = tms.copy()
        tms_other[i_home] = -np.inf
        j_best_other = int(np.argmax(tms_other))
        best_other_tm = tms_other[j_best_other]
        # Diagonal for confidently-home, off-diagonal for transitional.
        if home_tm - best_other_tm >= margin_threshold:
            counts[i_home, i_home] += 1
        else:
            counts[i_home, j_best_other] += 1

    row_totals = counts.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1.0
    matrix = counts / row_totals

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        annot=False,
        cbar_kws={"label": "Drift fraction"},
        linewidths=0.6, linecolor="white",
        xticklabels=[to_greek(s) for s in subtypes], yticklabels=[to_greek(s) for s in subtypes],
    )
    ax.set_xlabel("Nearest-medoid subtype")
    ax.set_ylabel("Home subtype")
    ax.set_title("Structural drift by nearest medoid")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# 5. Sweep overview (DTC hyper-parameter heatmap)
# ---------------------------------------------------------------------------

def plot_sweep_overview(sweep_df: pd.DataFrame, out_stem: Path) -> None:
    # (column, subplot title, colorbar label, cmap, annot fmt).
    # Colorbar label differs from title only where a distinct axis unit is
    # more informative than the subplot's high-level topic (e.g. the
    # Compactness panel's colorbar shows Mean Intra-TM).
    metrics = [
        ("n_clusters",                 "Number of clades",             "Number of clades",       "viridis", ".0f"),
        ("frac_clades_boot_ge_0.7",    "Clades with bootstrap ≥ 0.7",   "Clades with bootstrap ≥ 0.7", "Greens",  ".2f"),
        ("kingdom_purity_weighted",    "Kingdom purity",                "Kingdom purity",         "Blues",   ".2f"),
        ("reaction_purity_weighted",   "Reaction purity",               "Reaction purity",        "Oranges", ".2f"),
        ("transitional_frac_weighted", "Transitional fraction",         "Transitional fraction",  "Reds",    ".3f"),
        ("mean_intra_tm_weighted",     "Compactness",                   "Mean Intra-TM",          "Purples", ".3f"),
    ]
    deep_splits = sorted(sweep_df["deep_split"].unique())
    min_sizes = sorted(sweep_df["min_cluster_size"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for idx, (col, title, cbar_label, cmap, fmt) in enumerate(metrics):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        pivot = sweep_df.pivot(
            index="deep_split", columns="min_cluster_size", values=col,
        ).reindex(index=deep_splits, columns=min_sizes)
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt,
            cbar_kws={"label": cbar_label}, linewidths=0.6, linecolor="white",
            annot_kws={"fontsize": 10},
        )
        ax.set_xlabel("Minimum cluster size")
        ax.set_ylabel("Deep-split parameter")
        ax.set_title(title)
    fig.suptitle("DynamicTreeCut hyper-parameter sweep", fontsize=14, y=1.00, weight="bold")
    fig.tight_layout()
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    hac_dir = Path("outputs/domain_clustering")
    dtc_dir = hac_dir / "dtc_sweep"

    # Inputs.
    linkage_matrix = np.load(hac_dir / "intermediate/linkage_matrix.npy")

    # DISPLAY-ONLY: flip specific internal nodes so structural outliers
    # (delta3 at leaf 1947, zeta pair at node 4619) sit on the RIGHT of
    # their large siblings rather than on the LEFT. This is a rotation
    # of children at a single internal node — the tree topology and every
    # merge height are preserved, only the drawing order changes.
    # Display-only flips (tree topology preserved):
    #   2409 — delta1 to the left of delta2
    #   2419 — alpha4 sub-clade to the right of the alpha family
    #   2422 — delta3 outlier to the right of delta1/delta2
    #   2424 — zeta pair to the right end of the dendrogram
    LEAF_FLIP_ROWS = [2409, 2419, 2422, 2424]
    linkage_matrix = linkage_matrix.copy()
    for r in LEAF_FLIP_ROWS:
        linkage_matrix[r, 0], linkage_matrix[r, 1] = (
            linkage_matrix[r, 1], linkage_matrix[r, 0],
        )
    member_ids = pickle.load(open(hac_dir / "intermediate/member_ids.pkl", "rb"))
    subtype_map = pickle.load(open("data/domain_module_id_2_domain_subtype.pkl", "rb"))
    clades = json.load(open(dtc_dir / "clades_d0_m3.json"))

    # ---- Build shared palette (subtype → hex), keyed by dendrogram order
    #      so within-family tones progress light→dark following the
    #      left-to-right leaf sequence of the dendrogram.
    subtypes = sorted(set(subtype_map.values()))
    dend_order = dendrogram_leaf_subtype_order(linkage_matrix, member_ids, subtype_map)
    palette = build_subtype_palette(subtypes, order=dend_order)
    Path("data/domain_subtype_palette.json").write_text(json.dumps(palette, indent=2))
    logger.info("Dendrogram subtype order: %s", dend_order)
    logger.info("Palette: %s", palette)

    # ---- Build dtc_clade_id → subtype label map (majority vote per clade).
    label_map: dict[str, str] = {}
    for cid, members in clades.items():
        labs = [subtype_map.get(m) for m in members if subtype_map.get(m)]
        label_map[cid] = Counter(labs).most_common(1)[0][0] if labs else cid

    # ---- Load real metadata (bug fix: load_domain_metadata builds
    #      reaction_labels as ACTUAL Python lists, not stringified lists as
    #      the domain_metadata.csv round-trip produced).
    metadata_df = analysis.load_domain_metadata(
        "data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl",
        "data/martsDB_reactions_2026_02_22_preprocessed.csv",
    )
    pairwise_tm = analysis.load_or_compute_pairwise_tm(
        aln_dir=hac_dir / "all_vs_all", pdb_dir=hac_dir / "all_vs_all", threads=1,
    )
    stats_df = analysis.cluster_stats(clades, pairwise_tm, metadata_df)

    # ---- Clade table (with existing labels; we override via label_map).
    clade_table = pd.read_csv(dtc_dir / "clade_table_d0_m3.csv")
    # The clade table's built-in ``label`` column reflects the automatic
    # DTC labels (e.g. "α11"); the paper figures rely on ``label_map``
    # (built from the canonical pkl above) for display, so we don't need
    # to rewrite the clade table's label column here.

    # ---- Sweep summary (may or may not exist depending on prior DTC sweep).
    sweep_summary_path = dtc_dir / "sweep_summary.csv"
    sweep_df = pd.read_csv(sweep_summary_path) if sweep_summary_path.exists() else None

    # ---- Emit all four figure families.
    plot_dendrogram(
        linkage_matrix, member_ids, subtype_map, clades, label_map, palette,
        hac_dir / "dendrogram_d0_m3",
    )
    plot_metrics(
        clade_table, label_map, palette,
        hac_dir / "metrics_d0_m3",
    )
    plot_reactions(
        stats_df, label_map, palette,
        hac_dir / "reactions_per_clade_d0_m3",
    )
    plot_kingdoms(
        stats_df, label_map, palette,
        hac_dir / "kingdoms_per_clade_d0_m3",
    )
    distance_matrix = np.load(hac_dir / "intermediate/distance_matrix.npy")
    plot_transitional_drift(
        subtype_map, member_ids, distance_matrix,
        hac_dir / "transitional_drift_d0_m3",
    )
    if sweep_df is not None:
        plot_sweep_overview(sweep_df, hac_dir / "sweep_overview")
    else:
        logger.warning("No sweep_summary.csv at %s — skipping sweep_overview", sweep_summary_path)


if __name__ == "__main__":
    main()
