"""Visualisations for the Foldseek TM-threshold sweep (seaborn-based).

Two layers of plots:
  * per-T diagnostics: cluster-size histogram, intra-cluster TM vs size
    scatter, stacked-bar of kingdom and canonical-domain-type composition
    of the top-N clusters, per-cluster top-kingdom-fraction histogram,
  * cross-T sweep overview: count of clusters, singleton fraction, mean
    intra-cluster TM, weighted kingdom purity, weighted domain-type purity.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
import seaborn as sns  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(context="notebook", style="whitegrid", palette="muted")


def plot_sweep_overview(sweep_df: pd.DataFrame, output_path: str | Path) -> None:
    """Eight-panel overview of how cluster structure changes with T."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="n_clusters",
        marker="o", ax=axes[0, 0], label="all clusters",
    )
    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="n_clusters_n_ge_2",
        marker="s", ax=axes[0, 0], label="n ≥ 2 only",
    )
    axes[0, 0].set(xlabel="TM threshold", ylabel="# clusters", title="Cluster count")

    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="singleton_frac",
        marker="o", color="tab:red", ax=axes[0, 1],
    )
    axes[0, 1].set(
        xlabel="TM threshold", ylabel="singleton fraction",
        title="Singleton fraction", ylim=(0, 1),
    )

    long_size = sweep_df.melt(
        id_vars="tmscore_threshold",
        value_vars=["max_cluster_size", "mean_cluster_size", "median_cluster_size"],
        var_name="metric", value_name="size",
    )
    long_size["metric"] = long_size["metric"].str.replace("_cluster_size", "")
    sns.lineplot(
        data=long_size, x="tmscore_threshold", y="size", hue="metric",
        marker="o", ax=axes[0, 2],
    )
    axes[0, 2].set(
        xlabel="TM threshold", ylabel="cluster size", title="Cluster size",
        yscale="log",
    )

    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="mean_intra_tm_weighted",
        marker="o", ax=axes[1, 0], color="tab:blue", label="intra (compactness)",
    )
    if "mean_inter_tm_weighted" in sweep_df.columns:
        sns.lineplot(
            data=sweep_df, x="tmscore_threshold", y="mean_inter_tm_weighted",
            marker="s", ax=axes[1, 0], color="tab:orange", label="inter (separation)",
        )
    axes[1, 0].plot(
        sweep_df["tmscore_threshold"], sweep_df["tmscore_threshold"],
        "k--", alpha=0.4, label="y = T",
    )
    axes[1, 0].set(
        xlabel="TM threshold", ylabel="weighted mean TM",
        title="Compactness vs separation (intra/inter cluster TM)",
        ylim=(0, 1),
    )
    axes[1, 0].legend(fontsize=8)

    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="kingdom_purity_weighted",
        marker="o", color="tab:green", ax=axes[1, 1],
    )
    axes[1, 1].set(
        xlabel="TM threshold", ylabel="weighted top-kingdom fraction",
        title="Kingdom purity (n ≥ 2, weighted by size)", ylim=(0, 1.05),
    )

    sns.lineplot(
        data=sweep_df, x="tmscore_threshold", y="domain_type_purity_weighted",
        marker="o", color="tab:purple", ax=axes[1, 2],
    )
    axes[1, 2].set(
        xlabel="TM threshold", ylabel="weighted top-domain-type fraction",
        title="Canonical-domain-type purity (n ≥ 2)", ylim=(0, 1.05),
    )

    if "reaction_type_purity_weighted" in sweep_df.columns:
        sns.lineplot(
            data=sweep_df, x="tmscore_threshold", y="reaction_type_purity_weighted",
            marker="o", color="tab:orange", ax=axes[2, 0],
        )
    axes[2, 0].set(
        xlabel="TM threshold", ylabel="weighted top-reaction-type fraction",
        title="Reaction-type purity (all reactions per parent, n ≥ 2)", ylim=(0, 1.05),
    )

    # Reaction-type effective diversity: exp(Shannon entropy of the
    # reaction-tag distribution) per cluster, then size-weighted across
    # n ≥ 2 clusters. Direct complement to top-label purity:
    #   * 1.0   ⇒ a single reaction type per cluster (perfect monomorphism)
    #   * 2.0   ⇒ effectively two equally-sized reaction types per cluster
    #   * higher ⇒ clusters mix more reaction types
    # While "purity" reports only the dominant tag, perplexity captures the
    # whole distribution: a cluster split 60/40 has the same purity as one
    # split 60/30/10, but a higher perplexity (more types are mixed in).
    if "reaction_label_perplexity_weighted" in sweep_df.columns:
        sns.lineplot(
            data=sweep_df, x="tmscore_threshold",
            y="reaction_label_perplexity_weighted",
            marker="o", color="tab:brown", ax=axes[2, 1],
        )
        axes[2, 1].axhline(1.0, color="grey", linestyle=":", alpha=0.6)
    axes[2, 1].set(
        xlabel="TM threshold",
        ylabel="weighted effective # reaction types",
        title="Reaction-type diversity per cluster (1 = monomorphic)",
    )

    if "transitional_frac" in sweep_df.columns:
        sns.lineplot(
            data=sweep_df, x="tmscore_threshold", y="transitional_frac",
            marker="o", color="tab:red", ax=axes[2, 2],
        )
        axes[2, 2].set(
            xlabel="TM threshold",
            ylabel="fraction of transitional domains",
            title="Transitional fraction (smaller = more confident assignments)",
            ylim=(0, 1.05),
        )
    else:
        axes[2, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_cluster_size_distribution(
    stats_df: pd.DataFrame, T: float, output_path: str | Path
) -> None:
    """Log-y histogram of cluster sizes for a single T."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = stats_df["n"].values
    if len(sizes) == 0:
        plt.close(fig)
        return
    max_size = int(sizes.max())
    log_x = max_size > 30
    if log_x:
        bins = np.logspace(0, np.log10(max_size + 1), 25)
    else:
        bins = np.arange(0.5, max_size + 1.5, 1)
    sns.histplot(sizes, bins=bins, ax=ax, color="steelblue", edgecolor="black")
    ax.set(
        xlabel="cluster size", ylabel="# clusters",
        title=(
            f"Cluster size distribution (T={T:.2f})  "
            f"n_clusters={len(stats_df)}, "
            f"singletons={(sizes == 1).sum()}, max={max_size}"
        ),
        yscale="log",
    )
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_intra_tm_vs_size(
    stats_df: pd.DataFrame, T: float, output_path: str | Path
) -> None:
    """Scatter of mean intra-cluster TM vs cluster size, color = std."""
    multi = stats_df[stats_df["n"] >= 2].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if multi.empty:
        ax.text(0.5, 0.5, "no n≥2 clusters", ha="center", transform=ax.transAxes)
    else:
        sns.scatterplot(
            data=multi, x="n", y="mean_intra_tm",
            hue="std_intra_tm", palette="viridis",
            s=35, alpha=0.8, edgecolor="black", linewidth=0.3, ax=ax,
            legend=False,
        )
        norm = plt.Normalize(
            multi["std_intra_tm"].min(), multi["std_intra_tm"].max()
        )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("std intra-TM")
        ax.axhline(T, color="red", linestyle="--", alpha=0.5, label=f"T = {T:.2f}")
        ax.legend()
        ax.set_xscale("log")
    ax.set(
        xlabel="cluster size", ylabel="mean intra-cluster TM",
        title=f"Intra-cluster TM vs size (T={T:.2f})", ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_top_n_stacked_bar(
    stats_df: pd.DataFrame,
    distribution_col: str,
    T: float,
    output_path: str | Path,
    title: str,
    palette_name: str,
    max_clusters: int = 30,
    *,
    subtitle: str | None = None,
    label_map: dict[str, str] | None = None,
    label_key_col: str = "foldseek_rep",
) -> None:
    """Stacked-bar plot of a categorical distribution across the top-N largest clusters.

    ``label_map`` (optional) maps the cluster ID held in ``label_key_col``
    (default ``foldseek_rep``) to a human-readable display label. When
    supplied, the bar x-axis shows ``"<label>\\nn=<size>"`` instead of
    ``"<medoid>\\nn=<size>"``; ``label_key_col`` may be set to ``"medoid"``
    when ``label_map`` is keyed on medoid IDs instead.
    """
    top = stats_df.nlargest(max_clusters, "n")
    if top.empty:
        return

    rows = []
    for medoid, n, dist in zip(top["medoid"], top["n"], top[distribution_col]):
        total = sum(dist.values()) if dist else 0
        if total == 0:
            continue
        for category, count in dist.items():
            rows.append(
                {
                    "medoid": medoid,
                    "n": int(n),
                    "category": str(category),
                    "fraction": count / total,
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return

    cluster_order = top["medoid"].tolist()
    cat_order = sorted(long_df["category"].unique())
    pivot = (
        long_df.pivot_table(
            index="medoid", columns="category", values="fraction",
            aggfunc="sum", fill_value=0,
        )
        .reindex(cluster_order)
        [cat_order]
    )

    fig_w = max(10, len(cluster_order) * 0.34)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    palette = sns.color_palette(palette_name, n_colors=max(len(cat_order), 3))
    bottom = np.zeros(len(cluster_order))
    for i, cat in enumerate(cat_order):
        ax.bar(
            range(len(cluster_order)), pivot[cat].values, bottom=bottom,
            color=palette[i % len(palette)], label=cat, edgecolor="white",
            linewidth=0.4,
        )
        bottom += pivot[cat].values

    if label_map is not None and label_key_col in top.columns:
        x_labels = [
            f"{label_map.get(key, str(key))}\nn={n}"
            for key, n in zip(top[label_key_col], top["n"])
        ]
    else:
        x_labels = [
            f"{med[:18]}\nn={n}"
            for med, n in zip(top["medoid"], top["n"])
        ]
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    if subtitle is None:
        subtitle = f"T={T:.2f}, top {len(cluster_order)} clusters"
    ax.set(
        ylabel="fraction",
        title=f"{title} ({subtitle})",
        ylim=(0, 1.0),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kingdom_distribution(
    stats_df: pd.DataFrame, T: float, output_path: str | Path,
    max_clusters: int = 30,
    *,
    subtitle: str | None = None,
    label_map: dict[str, str] | None = None,
    label_key_col: str = "foldseek_rep",
) -> None:
    _plot_top_n_stacked_bar(
        stats_df, "kingdom_distribution", T, output_path,
        "Kingdom composition of largest clusters", "tab10", max_clusters,
        subtitle=subtitle, label_map=label_map, label_key_col=label_key_col,
    )


def plot_canonical_domain_type_distribution(
    stats_df: pd.DataFrame, T: float, output_path: str | Path,
    max_clusters: int = 30,
) -> None:
    _plot_top_n_stacked_bar(
        stats_df, "canonical_domain_type_distribution", T, output_path,
        "Canonical-domain-type composition of largest clusters", "Set2",
        max_clusters,
    )


def plot_reaction_label_distribution(
    stats_df: pd.DataFrame, T: float, output_path: str | Path,
    max_clusters: int = 30,
    *,
    subtitle: str | None = None,
    label_map: dict[str, str] | None = None,
    label_key_col: str = "foldseek_rep",
) -> None:
    """Stacked-bar of reaction-type composition for top clusters.

    Domain-frequency weighting: each domain contributes one count per
    reaction tag of its parent enzyme (every reaction counted regardless
    of catalytic class). The "irrelevant" sentinel appears as its own
    stack section only when a cluster contains domains whose parent
    sequence has no reactions in MartsDB at all.
    """
    if "reaction_label_distribution" not in stats_df.columns:
        return
    _plot_top_n_stacked_bar(
        stats_df, "reaction_label_distribution", T, output_path,
        "Reaction-type composition of largest clusters",
        "tab20", max_clusters,
        subtitle=subtitle, label_map=label_map, label_key_col=label_key_col,
    )


def plot_frac_irrelevant_per_cluster(
    stats_df: pd.DataFrame, T: float, output_path: str | Path,
    max_clusters: int = 60,
) -> None:
    """Bar chart of frac_irrelevant_domains per cluster.

    A high bar means most of the cluster's members come from sequences
    whose recorded reactions don't match the domain's catalytic role —
    i.e. the reaction-type purity number for that cluster is sitting on
    thin evidence. Surfaced separately so the user can judge purity
    reliability cluster-by-cluster.
    """
    if "frac_irrelevant_domains" not in stats_df.columns:
        return
    top = stats_df.nlargest(max_clusters, "n").reset_index(drop=True)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(top) * 0.32), 4.5))
    bars = ax.bar(
        np.arange(len(top)), top["frac_irrelevant_domains"],
        color=[
            "#d62728" if v >= 0.5 else "#ff7f0e" if v >= 0.2 else "#1f77b4"
            for v in top["frac_irrelevant_domains"]
        ],
        edgecolor="white", linewidth=0.4,
    )
    labels = [f"{med[:18]}\nn={n}" for med, n in zip(top["medoid"], top["n"])]
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set(
        ylabel="fraction of domains with only 'irrelevant' reaction label",
        ylim=(0, 1.0),
        title=(
            f"frac_irrelevant per cluster (T={T:.2f}, top {len(top)})\n"
            f"weighted mean = {(top['frac_irrelevant_domains'] * top['n']).sum() / max(top['n'].sum(), 1):.3f}"
        ),
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.2, color="black", linestyle=":", linewidth=0.5, alpha=0.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transitional_scatter(
    transitional_df: pd.DataFrame,
    T: float,
    output_path: str | Path,
    margin_threshold: float = 0.05,
) -> None:
    """Scatter of home_mean_tm vs alt_mean_tm.

    Each point is one detected domain. Points on the y=x diagonal are
    perfectly torn between home and alternative clusters; points well
    above the diagonal sit deep inside their home cluster. The
    ``margin_threshold`` band along the diagonal flags transitional
    members (``home_mean - alt_mean ≤ margin_threshold``).
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if transitional_df.empty:
        ax.text(0.5, 0.5, "no non-singleton clusters",
                ha="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    sns.scatterplot(
        data=transitional_df,
        x="alt_mean_tm", y="home_mean_tm",
        hue="is_transitional",
        palette={True: "#d62728", False: "#1f77b4"},
        s=20, alpha=0.7, edgecolor="none", ax=ax,
    )
    lim = (-0.02, 1.02)
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x (equally torn)")
    # Margin band: y = x + margin_threshold (above this, we consider it
    # confidently inside home).
    xs = np.linspace(0, 1, 100)
    ax.fill_between(
        xs, xs, xs + margin_threshold,
        alpha=0.15, color="orange",
        label=f"transitional band (margin ≤ {margin_threshold:.2f})",
    )
    n_transitional = int(transitional_df["is_transitional"].sum())
    n_total = len(transitional_df)
    ax.set(
        xlabel="mean TM to best alternative cluster",
        ylabel="mean TM to home cluster (excluding self)",
        xlim=lim, ylim=lim,
        title=(
            f"Transitional domain map (T={T:.2f})  "
            f"{n_transitional}/{n_total} flagged "
            f"({100.0 * n_transitional / max(n_total, 1):.1f}%)"
        ),
        aspect="equal",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transitional_margin_histogram(
    transitional_df: pd.DataFrame,
    T: float,
    output_path: str | Path,
    margin_threshold: float = 0.05,
) -> None:
    """Histogram of (home_mean_tm − alt_mean_tm) margins."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if transitional_df.empty:
        ax.text(0.5, 0.5, "no non-singleton clusters",
                ha="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    sns.histplot(
        transitional_df["margin"], bins=40, ax=ax,
        color="steelblue", edgecolor="black",
    )
    ax.axvline(
        margin_threshold, color="red", linestyle="--", alpha=0.7,
        label=f"margin = {margin_threshold:.2f}",
    )
    ax.axvline(0, color="black", linestyle=":", alpha=0.5, label="margin = 0")
    n_transitional = int(transitional_df["is_transitional"].sum())
    median = float(transitional_df["margin"].median())
    ax.set(
        xlabel="margin = home mean TM − alt mean TM",
        ylabel="# domains",
        title=(
            f"Transitional-domain margin distribution (T={T:.2f})\n"
            f"median = {median:.3f},  transitional = {n_transitional}"
        ),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_transitional_domains(
    transitional_df: pd.DataFrame,
    T: float,
    output_path: str | Path,
    top_n: int = 30,
) -> None:
    """Horizontal bar of the top-N most transitional domains: paired (home, alt) means."""
    if transitional_df.empty:
        return
    top = transitional_df.nsmallest(top_n, "margin").iloc[::-1].copy()
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
    y = np.arange(len(top))
    ax.barh(
        y - 0.18, top["home_mean_tm"], 0.36,
        color="#1f77b4", label="home (assigned) cluster",
    )
    ax.barh(
        y + 0.18, top["alt_mean_tm"], 0.36,
        color="#d62728", label="best alternative cluster",
    )
    labels = [
        f"{mid[:24]}\nhome: {hr[:14]} (n={hs})  vs  alt: {ar[:14]} (n={als})"
        for mid, hr, hs, ar, als in zip(
            top["module_id"], top["home_rep"], top["home_size"],
            top["alt_rep"], top["alt_size"],
        )
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set(
        xlabel="mean TM to cluster",
        title=f"Top-{len(top)} most transitional domains (T={T:.2f}, smallest margins)",
        xlim=(0, 1),
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transitional_flow(
    transitional_df: pd.DataFrame,
    T: float,
    output_path: str | Path,
    top_pairs: int = 20,
) -> None:
    """Heatmap of (home_cluster × alt_cluster) transitional-domain counts.

    Highlights which cluster pairs share boundary domains — a candidate
    "merge" relation if the user later wants to consolidate.
    """
    flagged = transitional_df[transitional_df["is_transitional"]]
    if flagged.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no transitional domains",
                ha="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    pair_counts = (
        flagged.groupby(["home_rep", "alt_rep"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_pairs)
    )
    if pair_counts.empty:
        return

    pivot = pair_counts.pivot_table(
        index="home_rep", columns="alt_rep", values="count", fill_value=0
    ).astype(int)
    fig, ax = plt.subplots(
        figsize=(max(6, 0.45 * pivot.shape[1] + 4),
                 max(4, 0.45 * pivot.shape[0] + 2))
    )
    sns.heatmap(
        pivot, ax=ax, cmap="Reds", annot=True, fmt="d",
        cbar_kws={"label": "# transitional domains"},
    )
    ax.set(
        xlabel="best alternative cluster (rep)",
        ylabel="home cluster (rep)",
        title=(
            f"Transitional-domain flow between clusters (T={T:.2f}, "
            f"top {len(pair_counts)} pairs)"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dendrogram_truncated(
    linkage_matrix: np.ndarray,
    output_path: str | Path,
    *,
    tm_thresholds: list[float] | None = None,
    p: int = 50,
    truncate_mode: str = "lastp",
) -> None:
    """Top-N-merges dendrogram with horizontal lines at each TM threshold.

    Renders the top ``p`` merges of the hierarchy (default 50) — full
    2000+-leaf dendrograms are illegible. Each TM threshold is drawn as a
    horizontal line at distance ``1 − T`` so it's visually obvious which
    cuts give which cluster counts.
    """
    from scipy.cluster.hierarchy import dendrogram  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0.5,
        above_threshold_color="grey",
    )
    if tm_thresholds is not None:
        cmap = sns.color_palette("rocket_r", n_colors=len(tm_thresholds))
        for i, T in enumerate(sorted(tm_thresholds)):
            ax.axhline(
                1.0 - T, color=cmap[i], linestyle="--", linewidth=1.0,
                alpha=0.85, label=f"T = {T:.2f}",
            )
        ax.legend(loc="upper right", fontsize=8)
    ax.set(
        xlabel="cluster index" if truncate_mode == "lastp" else "domain (truncated)",
        ylabel="distance (1 − TM)",
        title=f"HAC dendrogram (top {p} merges, average linkage)",
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_dendrogram_full(
    linkage_matrix: np.ndarray,
    output_path: str | Path,
    tm_thresholds: list[float] | None = None,
) -> None:
    """Full dendrogram with no leaf labels — saved as PDF for inspection.

    Renders thousands of leaves; readable only when zoomed in a viewer.
    """
    from scipy.cluster.hierarchy import dendrogram  # type: ignore

    fig, ax = plt.subplots(figsize=(20, 8))
    dendrogram(
        linkage_matrix, ax=ax, no_labels=True,
        color_threshold=0.5, above_threshold_color="grey",
    )
    if tm_thresholds is not None:
        cmap = sns.color_palette("rocket_r", n_colors=len(tm_thresholds))
        for i, T in enumerate(sorted(tm_thresholds)):
            ax.axhline(
                1.0 - T, color=cmap[i], linestyle="--", linewidth=0.8,
                alpha=0.8, label=f"T = {T:.2f}",
            )
        ax.legend(loc="upper right", fontsize=8)
    ax.set(
        xlabel="domains (no labels at this scale)",
        ylabel="distance (1 − TM)",
        title="HAC dendrogram (full, average linkage)",
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_dendrogram_with_leaf_categories(
    linkage_matrix: np.ndarray,
    member_ids: list[str],
    metadata_df: pd.DataFrame,
    output_path: str | Path,
    *,
    tm_thresholds: list[float] | None = None,
) -> None:
    """Dendrogram + horizontal stripes coloring each leaf by kingdom and domain type.

    Branch colors are matched to the canonical domain-type stripe palette:
    a clade is drawn in its domain-type color when **all** leaves under it
    share one canonical domain type, otherwise the branch is grey
    (mixed-type clade). This makes it visually obvious where the tree
    becomes type-mixed. Kingdom and domain-type stripes underneath show
    the per-leaf metadata in dendrogram order.

    All three legends (dendrogram branch colors, kingdom, domain type) sit
    in the figure's right margin without overlapping.
    """
    from scipy.cluster.hierarchy import dendrogram  # type: ignore

    metadata_index = metadata_df.set_index("module_id")
    n = len(member_ids)
    n_leaves = n  # in scipy linkage convention

    # ---- Build per-leaf metadata vectors aligned to member_ids order ----
    leaf_kingdom = [
        metadata_index.loc[mid, "kingdom"] if mid in metadata_index.index else None
        for mid in member_ids
    ]
    leaf_dtype = [
        metadata_index.loc[mid, "canonical_domain_type"]
        if mid in metadata_index.index else None
        for mid in member_ids
    ]

    kingdom_categories = sorted(
        {v for v in leaf_kingdom if v is not None and not pd.isna(v)}
    )
    dtype_categories = sorted(
        {v for v in leaf_dtype if v is not None and not pd.isna(v)}
    )
    kingdom_palette = sns.color_palette("tab10", n_colors=max(len(kingdom_categories), 3))
    dtype_palette = sns.color_palette("Set2", n_colors=max(len(dtype_categories), 3))
    kingdom_color = {c: kingdom_palette[i] for i, c in enumerate(kingdom_categories)}
    dtype_color = {c: dtype_palette[i] for i, c in enumerate(dtype_categories)}

    # ---- Memoised "leaves under linkage_id" + "monophyletic dtype" ----
    Z = np.asarray(linkage_matrix)
    descendants_cache: dict[int, list[int]] = {}

    def _descendants(node_id: int) -> list[int]:
        if node_id in descendants_cache:
            return descendants_cache[node_id]
        if node_id < n_leaves:
            descendants_cache[node_id] = [node_id]
            return descendants_cache[node_id]
        row = node_id - n_leaves
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        d = _descendants(left) + _descendants(right)
        descendants_cache[node_id] = d
        return d

    MIXED_BRANCH_COLOR = "#bfbfbf"

    def _link_color_func(node_id: int) -> str:
        leaves = _descendants(int(node_id))
        types = {leaf_dtype[i] for i in leaves}
        types.discard(None)
        if len(types) == 1:
            return mpl_color_hex(dtype_color[next(iter(types))])
        return MIXED_BRANCH_COLOR

    # ---- Layout: dendrogram on top, two stripes underneath; legend column on right ----
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(
        nrows=4, ncols=2, width_ratios=[18, 4],
        height_ratios=[10, 0.5, 0.5, 0.4],
        hspace=0.07, wspace=0.02,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_kingdom = fig.add_subplot(gs[1, 0], sharex=ax_dend)
    ax_dtype = fig.add_subplot(gs[2, 0], sharex=ax_dend)
    ax_legend = fig.add_subplot(gs[:, 1])
    ax_legend.axis("off")

    ddata = dendrogram(
        linkage_matrix, ax=ax_dend, no_labels=True,
        link_color_func=_link_color_func,
    )
    leaves = ddata["leaves"]
    n_leaves_ordered = len(leaves)
    leaf_x = (np.arange(n_leaves_ordered) + 0.5) * 10  # scipy positions: 5, 15, 25, ...

    if tm_thresholds is not None:
        cmap = sns.color_palette("rocket_r", n_colors=len(tm_thresholds))
        for i, T in enumerate(sorted(tm_thresholds)):
            ax_dend.axhline(
                1.0 - T, color=cmap[i], linestyle="--", linewidth=0.8,
                alpha=0.85, label=f"T = {T:.2f}",
            )
        ax_dend.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)
    ax_dend.set(
        ylabel="distance (1 − TM)",
        title=(
            "HAC dendrogram with leaf annotations  —  "
            "branch colors match canonical-domain-type stripe (grey = mixed clade)"
        ),
        ylim=(0, 1.02),
    )
    ax_dend.set_xticks([])

    # ---- Stripe builder (no per-axis legend; legends go to ax_legend) ----
    def _build_stripe(values: list, color_map: dict, axis, axis_label: str) -> None:
        for i, leaf_idx in enumerate(leaves):
            v = values[leaf_idx]
            color = color_map.get(v, (0.85, 0.85, 0.85))
            axis.barh(
                y=0, width=10.0, left=leaf_x[i] - 5, height=1,
                color=color, edgecolor="none",
            )
        axis.set_xlim(0, n_leaves_ordered * 10)
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5])
        axis.set_yticklabels([axis_label], fontsize=9)
        axis.set_xticks([])
        axis.tick_params(left=False)
        for s in ("top", "right", "bottom", "left"):
            axis.spines[s].set_visible(False)

    _build_stripe(leaf_kingdom, kingdom_color, ax_kingdom, "Kingdom")
    _build_stripe(leaf_dtype, dtype_color, ax_dtype, "Domain type")

    # ---- Three stacked legends in the right column ----
    def _add_legend(ax, handles, labels, title: str, anchor_y: float) -> None:
        leg = ax.legend(
            handles, labels, title=title,
            loc="upper left", bbox_to_anchor=(0.0, anchor_y),
            fontsize=8, title_fontsize=9, frameon=True,
            handlelength=1.2, borderpad=0.6,
        )
        ax.add_artist(leg)

    branch_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=dtype_color[c]) for c in dtype_categories
    ] + [plt.Rectangle((0, 0), 1, 1, facecolor=MIXED_BRANCH_COLOR)]
    branch_labels = list(dtype_categories) + ["mixed clade"]
    _add_legend(
        ax_legend, branch_handles, branch_labels,
        "Dendrogram branch color\n(per-clade canonical domain type)",
        1.00,
    )

    kingdom_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=kingdom_color[c]) for c in kingdom_categories
    ]
    _add_legend(
        ax_legend, kingdom_handles, list(kingdom_categories),
        "Kingdom (stripe)", 0.55,
    )

    dtype_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=dtype_color[c]) for c in dtype_categories
    ]
    _add_legend(
        ax_legend, dtype_handles, list(dtype_categories),
        "Canonical domain type (stripe)", 0.20,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_dendrogram_proportional(
    linkage_matrix: np.ndarray,
    member_ids: list[str],
    metadata_df: pd.DataFrame,
    output_path: str | Path,
    *,
    collapse_at_tm: float = 0.6,
    tm_thresholds: list[float] | None = None,
    label_top_k: int = 30,
) -> None:
    """Alternative dendrogram with **proportional clade widths**.

    Standard dendrograms place every truncated clade at equal horizontal
    spacing — a clade of 5 leaves looks the same width as a clade of 1500.
    This view fixes that: every clade below ``collapse_at_tm`` is drawn as
    a rectangle whose **width is proportional to the number of domains** it
    contains. The merge structure above the cut is drawn normally, with
    each subtree's vertical line landing at the size-weighted center of
    the clades below it. Branch colors match the canonical-domain-type
    palette (grey when a clade spans multiple types).

    Bottom stripe (below y=0) shows kingdom composition for each clade as
    a stacked bar inside the clade's horizontal extent.
    """
    from scipy.cluster.hierarchy import dendrogram, fcluster  # type: ignore

    Z = np.asarray(linkage_matrix)
    n = len(member_ids)
    cut_distance = 1.0 - collapse_at_tm

    metadata_index = metadata_df.set_index("module_id")
    leaf_dtype = [
        metadata_index.loc[mid, "canonical_domain_type"]
        if mid in metadata_index.index else None
        for mid in member_ids
    ]
    dtype_categories = sorted({v for v in leaf_dtype if v is not None and not pd.isna(v)})
    kingdom_categories = sorted(
        {v for v in metadata_index["kingdom"].dropna().unique()}
    )
    dtype_palette = sns.color_palette("Set2", n_colors=max(len(dtype_categories), 3))
    kingdom_palette = sns.color_palette("tab10", n_colors=max(len(kingdom_categories), 3))
    dtype_color = {c: dtype_palette[i] for i, c in enumerate(dtype_categories)}
    kingdom_color = {c: kingdom_palette[i] for i, c in enumerate(kingdom_categories)}
    MIXED_COLOR = "#bfbfbf"

    # ---- Cut into clades ----
    cluster_ids = fcluster(Z, t=cut_distance, criterion="distance")
    # Order clades by their first occurrence in the standard leaf order so
    # the proportional view preserves the dendrogram's own left-to-right
    # ordering (helps visual comparison with the equal-width version).
    standard = dendrogram(Z, no_plot=True)
    cluster_first_pos: dict[int, int] = {}
    for pos, leaf in enumerate(standard["leaves"]):
        cid = int(cluster_ids[leaf])
        if cid not in cluster_first_pos:
            cluster_first_pos[cid] = pos
    cluster_order = sorted(cluster_first_pos, key=cluster_first_pos.get)

    # ---- Per-clade width / colour / kingdom-distribution ----
    cluster_size = {cid: int((cluster_ids == cid).sum()) for cid in cluster_order}
    total_size = sum(cluster_size.values())
    cluster_width = {cid: cluster_size[cid] / total_size for cid in cluster_order}
    cluster_left = {}
    cum = 0.0
    for cid in cluster_order:
        cluster_left[cid] = cum
        cum += cluster_width[cid]
    cluster_center = {
        cid: cluster_left[cid] + cluster_width[cid] / 2 for cid in cluster_order
    }

    cluster_dom_dtype: dict[int, str | None] = {}
    cluster_dom_dtype_frac: dict[int, float] = {}
    cluster_kingdom_dist: dict[int, dict[str, int]] = {}
    for cid in cluster_order:
        members = [member_ids[i] for i in np.where(cluster_ids == cid)[0]]
        meta = metadata_index.reindex(members)
        dtypes = meta["canonical_domain_type"].dropna().tolist()
        kingdoms = meta["kingdom"].dropna().tolist()
        from collections import Counter as _Counter

        dt_counts = _Counter(dtypes)
        kg_counts = _Counter(kingdoms)
        if dt_counts:
            top, top_n = dt_counts.most_common(1)[0]
            cluster_dom_dtype[cid] = top
            cluster_dom_dtype_frac[cid] = top_n / len(dtypes)
        else:
            cluster_dom_dtype[cid] = None
            cluster_dom_dtype_frac[cid] = 0.0
        cluster_kingdom_dist[cid] = dict(kg_counts)

    # ---- Branch coloring (per dtype monophyly under each internal node) ----
    descendants_cache: dict[int, list[int]] = {}

    def _descendants(node_id: int) -> list[int]:
        if node_id in descendants_cache:
            return descendants_cache[node_id]
        if node_id < n:
            descendants_cache[node_id] = [node_id]
            return descendants_cache[node_id]
        row = node_id - n
        d = _descendants(int(Z[row, 0])) + _descendants(int(Z[row, 1]))
        descendants_cache[node_id] = d
        return d

    def _branch_color(node_id: int) -> str:
        leaves = _descendants(int(node_id))
        types = {leaf_dtype[i] for i in leaves}
        types.discard(None)
        if len(types) == 1:
            return mpl_color_hex(dtype_color[next(iter(types))])
        return MIXED_COLOR

    # ---- Walk the linkage matrix and draw merges ABOVE the cut ----
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[18, 4], wspace=0.02,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    stripe_h = 0.05  # height (in distance units) of the kingdom stripe under y=0
    # Draw clade rectangles (colored by dominant domain type).
    for cid in cluster_order:
        left = cluster_left[cid]
        w = cluster_width[cid]
        dom = cluster_dom_dtype[cid]
        color = dtype_color[dom] if dom else (0.85, 0.85, 0.85)
        rect = plt.Rectangle(
            (left, 0), w, cut_distance,
            facecolor=color, alpha=0.55, edgecolor="white", linewidth=0.6,
            zorder=2,
        )
        ax.add_patch(rect)

    # Kingdom stripe under y=0 — stacked composition per clade.
    for cid in cluster_order:
        left = cluster_left[cid]
        w = cluster_width[cid]
        dist = cluster_kingdom_dist[cid]
        total_k = sum(dist.values())
        if total_k == 0:
            continue
        x = left
        for k in kingdom_categories:
            frac = dist.get(k, 0) / total_k
            if frac == 0:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (x, -stripe_h), w * frac, stripe_h,
                    color=kingdom_color[k], edgecolor="none", zorder=2,
                )
            )
            x += w * frac

    # Annotate top-K largest clades inside their rectangles where width
    # allows. Width threshold scales with figure size so labels never
    # overlap.
    sorted_by_size = sorted(cluster_order, key=lambda c: -cluster_size[c])
    min_width_for_label = 0.012  # ~1.2% of total width
    for rank, cid in enumerate(sorted_by_size[:label_top_k]):
        w = cluster_width[cid]
        if w < min_width_for_label:
            continue
        cx = cluster_center[cid]
        cy = cut_distance / 2
        dom = cluster_dom_dtype[cid] or "?"
        ax.text(
            cx, cy,
            f"#{rank + 1}\nn={cluster_size[cid]}\n{dom}",
            ha="center", va="center", fontsize=8, fontweight="bold",
            zorder=5,
        )

    # Walk linkage matrix: draw merges above the cut.
    node_center: dict[int, float] = {}
    node_top_y: dict[int, float] = {}
    node_size: dict[int, int] = {}
    for i in range(n):
        cid = int(cluster_ids[i])
        node_center[i] = cluster_center[cid]
        node_top_y[i] = cut_distance
        node_size[i] = 1

    for row in range(Z.shape[0]):
        node_id = n + row
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        distance = float(Z[row, 2])
        if distance < cut_distance:
            # Below cut: collapsed inside its clade.
            node_center[node_id] = node_center[left]
            node_top_y[node_id] = cut_distance
            node_size[node_id] = node_size[left] + node_size[right]
            continue
        lx = node_center[left]
        rx = node_center[right]
        ly = node_top_y[left]
        ry = node_top_y[right]
        my = distance
        color = _branch_color(node_id)
        ax.plot([lx, lx], [ly, my], color=color, linewidth=0.9, zorder=3)
        ax.plot([rx, rx], [ry, my], color=color, linewidth=0.9, zorder=3)
        ax.plot([lx, rx], [my, my], color=color, linewidth=0.9, zorder=3)
        sl = node_size[left]
        sr = node_size[right]
        node_center[node_id] = (sl * lx + sr * rx) / (sl + sr)
        node_top_y[node_id] = my
        node_size[node_id] = sl + sr

    # Cut line + threshold annotations.
    ax.axhline(
        cut_distance, color="black", linestyle="-", linewidth=0.7, alpha=0.5,
    )
    if tm_thresholds is not None:
        cmap = sns.color_palette("rocket_r", n_colors=len(tm_thresholds))
        for i, T in enumerate(sorted(tm_thresholds)):
            d = 1.0 - T
            if d < cut_distance - 1e-9:
                continue  # below the collapse cut: not visible above it
            ax.axhline(
                d, color=cmap[i], linestyle="--", linewidth=0.8,
                alpha=0.85, label=f"T = {T:.2f}",
            )
        ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)

    # Axes cosmetics.
    y_min = -stripe_h * 1.6
    y_max = max(1.02, float(Z[:, 2].max()) * 1.02)
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min, y_max)
    ax.set(
        xlabel="proportional position (rectangle width = fraction of all domains in that clade)",
        ylabel="distance (1 − TM)",
        title=(
            "HAC dendrogram — proportional clade widths  "
            f"(collapsed at TM = {collapse_at_tm:.2f})\n"
            f"{len(cluster_order)} clades shown as rectangles, "
            "colored by dominant canonical domain type; "
            "kingdom composition stripe below"
        ),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legends in the right column.
    def _add_legend(handles, labels, title: str, anchor_y: float) -> None:
        leg = ax_legend.legend(
            handles, labels, title=title,
            loc="upper left", bbox_to_anchor=(0.0, anchor_y),
            fontsize=8, title_fontsize=9, frameon=True,
            handlelength=1.2, borderpad=0.6,
        )
        ax_legend.add_artist(leg)

    branch_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=dtype_color[c]) for c in dtype_categories
    ] + [plt.Rectangle((0, 0), 1, 1, facecolor=MIXED_COLOR)]
    branch_labels = list(dtype_categories) + ["mixed clade"]
    _add_legend(
        branch_handles, branch_labels,
        "Branch / clade-rectangle color\n(canonical domain type)",
        1.00,
    )
    kingdom_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=kingdom_color[c]) for c in kingdom_categories
    ]
    _add_legend(kingdom_handles, list(kingdom_categories), "Kingdom (stripe)", 0.55)

    # A legend block listing the top-K largest clades with their stats.
    top_text_lines = []
    for rank, cid in enumerate(sorted_by_size[:label_top_k]):
        dom = cluster_dom_dtype[cid] or "?"
        purity = cluster_dom_dtype_frac[cid]
        top_text_lines.append(
            f"#{rank + 1:>2}  n={cluster_size[cid]:>4}  "
            f"{dom:<8} (purity {purity * 100:.0f}%)"
        )
    ax_legend.text(
        0.0, 0.30,
        "Top clades by size:\n" + "\n".join(top_text_lines),
        fontsize=7, family="monospace", verticalalignment="top",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _categorical_palette(n: int) -> list:
    """A categorical palette wide enough for arbitrary cluster counts."""
    base = (
        sns.color_palette("tab20", 20)
        + sns.color_palette("tab20b", 20)
        + sns.color_palette("tab20c", 20)
    )
    if n <= len(base):
        return base[:n]
    # Fall back to looping the palette with hue-shifted copies.
    extra = sns.color_palette("Set3", n - len(base))
    return base + list(extra)


def plot_dendrogram_full_with_clade_categories(
    linkage_matrix: np.ndarray,
    member_ids: list[str],
    metadata_df: pd.DataFrame,
    clusters: dict[str, list[str]],
    output_path: str | Path,
    *,
    title: str = "",
    label_map: dict[str, str] | None = None,
    line_width: float = 0.6,
) -> None:
    """Full dendrogram + three leaf-aligned stripes (clade, kingdom, domain
    type), with branch colours driven by clade membership.

    Mirrors :func:`plot_dendrogram_with_leaf_categories` so the two figures
    are visually comparable; adds a third stripe for the user's clade
    assignment and recolours the dendrogram branches accordingly. A clade
    is drawn in its assigned colour when **all** leaves below it belong
    to that clade; otherwise the branch is drawn in grey.
    """
    from scipy.cluster.hierarchy import dendrogram  # type: ignore

    metadata_index = metadata_df.set_index("module_id")
    n = len(member_ids)
    n_leaves = n

    leaf_kingdom = [
        metadata_index.loc[mid, "kingdom"] if mid in metadata_index.index else None
        for mid in member_ids
    ]
    leaf_dtype = [
        metadata_index.loc[mid, "canonical_domain_type"]
        if mid in metadata_index.index else None
        for mid in member_ids
    ]
    member_to_clade: dict[str, str] = {
        m: cid for cid, members in clusters.items() for m in members
    }
    leaf_clade = [member_to_clade.get(mid) for mid in member_ids]

    kingdom_categories = sorted(
        {v for v in leaf_kingdom if v is not None and not pd.isna(v)}
    )
    dtype_categories = sorted(
        {v for v in leaf_dtype if v is not None and not pd.isna(v)}
    )
    clade_categories = sorted(
        clusters.keys(), key=lambda c: (-len(clusters[c]), c),
    )
    kingdom_palette = sns.color_palette("tab10", n_colors=max(len(kingdom_categories), 3))
    dtype_palette = sns.color_palette("Set2", n_colors=max(len(dtype_categories), 3))
    clade_palette = _categorical_palette(len(clade_categories))
    kingdom_color = {c: kingdom_palette[i] for i, c in enumerate(kingdom_categories)}
    dtype_color = {c: dtype_palette[i] for i, c in enumerate(dtype_categories)}
    clade_color = {c: clade_palette[i] for i, c in enumerate(clade_categories)}

    Z = np.asarray(linkage_matrix)
    descendants_cache: dict[int, list[int]] = {}

    def _descendants(node_id: int) -> list[int]:
        if node_id in descendants_cache:
            return descendants_cache[node_id]
        if node_id < n_leaves:
            descendants_cache[node_id] = [node_id]
            return descendants_cache[node_id]
        row = node_id - n_leaves
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        d = _descendants(left) + _descendants(right)
        descendants_cache[node_id] = d
        return d

    MIXED_BRANCH_COLOR = "#bfbfbf"

    def _link_color_func(node_id: int) -> str:
        leaves = _descendants(int(node_id))
        clades = {leaf_clade[i] for i in leaves}
        clades.discard(None)
        if len(clades) == 1:
            return mpl_color_hex(clade_color[next(iter(clades))])
        return MIXED_BRANCH_COLOR

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        nrows=5, ncols=2, width_ratios=[18, 4],
        height_ratios=[10, 0.5, 0.5, 0.5, 0.4],
        hspace=0.07, wspace=0.02,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_clade = fig.add_subplot(gs[1, 0], sharex=ax_dend)
    ax_kingdom = fig.add_subplot(gs[2, 0], sharex=ax_dend)
    ax_dtype = fig.add_subplot(gs[3, 0], sharex=ax_dend)
    ax_legend = fig.add_subplot(gs[:, 1])
    ax_legend.axis("off")

    ddata = dendrogram(
        linkage_matrix, ax=ax_dend, no_labels=True,
        link_color_func=_link_color_func,
    )
    # scipy draws each H/V link as a line. Reduce its width to avoid
    # adjacent leaves' verticals merging into a solid block.
    for coll in ax_dend.collections:
        try:
            coll.set_linewidth(line_width)
        except Exception:
            pass
    for line in ax_dend.get_lines():
        try:
            line.set_linewidth(line_width)
        except Exception:
            pass
    leaves = ddata["leaves"]
    n_leaves_ordered = len(leaves)
    leaf_x = (np.arange(n_leaves_ordered) + 0.5) * 10

    ax_dend.set(
        ylabel="distance (1 − TM)",
        title=(f"HAC dendrogram with clade colouring  —  {title}".strip(" —")
               + "\n(branch colour = clade when monophyletic; grey otherwise)"),
        ylim=(0, 1.02),
    )
    ax_dend.set_xticks([])

    def _build_stripe(values: list, color_map: dict, axis, axis_label: str) -> None:
        for i, leaf_idx in enumerate(leaves):
            v = values[leaf_idx]
            color = color_map.get(v, (0.85, 0.85, 0.85))
            axis.barh(
                y=0, width=10.0, left=leaf_x[i] - 5, height=1,
                color=color, edgecolor="none",
            )
        axis.set_xlim(0, n_leaves_ordered * 10)
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5])
        axis.set_yticklabels([axis_label], fontsize=9)
        axis.set_xticks([])
        axis.tick_params(left=False)
        for s in ("top", "right", "bottom", "left"):
            axis.spines[s].set_visible(False)

    _build_stripe(leaf_clade, clade_color, ax_clade, "Clade")
    _build_stripe(leaf_kingdom, kingdom_color, ax_kingdom, "Kingdom")
    _build_stripe(leaf_dtype, dtype_color, ax_dtype, "Domain type")

    def _add_legend(ax, handles, labels, title_: str, anchor_y: float,
                    ncol: int = 1) -> None:
        leg = ax.legend(
            handles, labels, title=title_,
            loc="upper left", bbox_to_anchor=(0.0, anchor_y),
            fontsize=7, title_fontsize=9, frameon=True,
            handlelength=1.2, borderpad=0.5, ncol=ncol,
        )
        ax.add_artist(leg)

    clade_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=clade_color[c]) for c in clade_categories
    ]
    if label_map is not None:
        clade_labels = [
            f"{label_map.get(c, c)} (n={len(clusters[c])})"
            for c in clade_categories
        ]
    else:
        clade_labels = [
            f"{c} (n={len(clusters[c])})" for c in clade_categories
        ]
    # Show clade legend at top of right column with two columns to fit ~30 entries.
    _add_legend(
        ax_legend, clade_handles, clade_labels,
        "Clade (stripe + branch colour)", 1.00,
        ncol=1 if len(clade_categories) <= 16 else 2,
    )

    kingdom_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=kingdom_color[c]) for c in kingdom_categories
    ]
    _add_legend(
        ax_legend, kingdom_handles, list(kingdom_categories),
        "Kingdom (stripe)", 0.45,
    )

    dtype_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=dtype_color[c]) for c in dtype_categories
    ]
    _add_legend(
        ax_legend, dtype_handles, list(dtype_categories),
        "Canonical domain type (stripe)", 0.18,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_clade_metrics_panel(
    clade_table: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    bootstrap_threshold: float = 0.7,
    label_map: dict[str, str] | None = None,
    representative_col: str | None = None,
) -> None:
    """Six-panel per-clade metric overview for one DTC configuration.

    ``clade_table`` is expected to have one row per clade with columns:
      ``clade_id``, ``n``, ``mean_intra_tm``, ``mean_inter_tm``,
      ``top_kingdom_frac``, ``top_reaction_label_frac``,
      ``transitional_frac``, ``bootstrap_support``.
    """
    df = clade_table.copy()
    if "is_unassigned" in df.columns:
        df = df[~df["is_unassigned"]]
    df = df.sort_values("n", ascending=False).reset_index(drop=True)
    if df.empty:
        return
    palette = _categorical_palette(len(df))
    x = np.arange(len(df))
    if label_map is not None:
        labels = [label_map.get(c, str(c)) for c in df["clade_id"]]
    else:
        labels = [str(c) for c in df["clade_id"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # 1. Cluster sizes
    ax = axes[0, 0]
    ax.bar(x, df["n"].values, color=palette, edgecolor="black", linewidth=0.4)
    ax.set(yscale="log", ylabel="# domains (log)", title="Cluster sizes")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    sns.despine(ax=ax)

    # 2. Compactness vs separation
    ax = axes[0, 1]
    bw = 0.4
    ax.bar(x - bw/2, df["mean_intra_tm"].values, bw,
           color="tab:blue", label="intra-TM (compact)", edgecolor="black", linewidth=0.4)
    ax.bar(x + bw/2, df["mean_inter_tm"].values, bw,
           color="tab:orange", label="inter-TM (separation)", edgecolor="black", linewidth=0.4)
    ax.set(ylabel="mean TM", ylim=(0, 1.02),
           title="Compactness vs separation")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # 3. Kingdom purity
    ax = axes[0, 2]
    ax.bar(x, df["top_kingdom_frac"].values, color=palette,
           edgecolor="black", linewidth=0.4)
    ax.set(ylabel="top-kingdom fraction", ylim=(0, 1.05),
           title="Kingdom purity")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    sns.despine(ax=ax)

    # 4. Reaction-type purity
    ax = axes[1, 0]
    ax.bar(x, df["top_reaction_label_frac"].values, color=palette,
           edgecolor="black", linewidth=0.4)
    ax.set(ylabel="top-reaction fraction", ylim=(0, 1.05),
           title="Reaction-type purity (all reactions per parent)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    sns.despine(ax=ax)

    # 5. Transitional fraction
    ax = axes[1, 1]
    if "transitional_frac" in df.columns:
        ax.bar(x, df["transitional_frac"].values, color=palette,
               edgecolor="black", linewidth=0.4)
    ax.set(ylabel="fraction of borderline domains",
           ylim=(0, max(0.01, float(df.get("transitional_frac", pd.Series([0])).max() * 1.1))),
           title="Transitional fraction (smaller = more confident)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    sns.despine(ax=ax)

    # 6. Bootstrap support
    ax = axes[1, 2]
    if "bootstrap_support" in df.columns:
        bars = ax.bar(x, df["bootstrap_support"].values, color=palette,
                     edgecolor="black", linewidth=0.4)
        ax.axhline(bootstrap_threshold, linestyle="--",
                   color="grey", alpha=0.7,
                   label=f"threshold {bootstrap_threshold:.2f}")
        ax.legend(fontsize=8)
    ax.set(ylabel="bootstrap support fraction",
           ylim=(0, 1.05),
           title="Bootstrap reproducibility")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    sns.despine(ax=ax)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def mpl_color_hex(rgba) -> str:
    """Convert a matplotlib RGB(A) tuple/array to a #rrggbb string for scipy."""
    r, g, b = rgba[:3]
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def plot_n_clusters_vs_height(
    n_vs_T: pd.DataFrame,
    output_path: str | Path,
    tm_thresholds_marked: list[float] | None = None,
) -> None:
    """Curve of # clusters vs TM cut threshold — plateaus = stable cuts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=n_vs_T, x="tmscore_threshold", y="n_clusters",
        marker="o", ax=ax,
    )
    if tm_thresholds_marked is not None:
        for T in tm_thresholds_marked:
            ax.axvline(T, color="red", linestyle="--", alpha=0.4)
    ax.set(
        xlabel="TM threshold (cut at 1 − T)",
        ylabel="# clusters", yscale="log",
        title="HAC: number of clusters vs cut threshold (plateaus = stable cuts)",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _cluster_color_palette(n: int) -> list:
    """Return ``n`` perceptually-distinct colors.

    Uses seaborn's ``hls`` palette for arbitrary n. For n ≤ 20 we prefer
    ``tab20`` because it's qualitative and easier to tell adjacent colors
    apart, but for typical TPS-domain clusterings (10-160 clusters) ``hls``
    with even hue spacing wins.
    """
    if n <= 20:
        return list(sns.color_palette("tab20", n_colors=n))
    return list(sns.color_palette("hls", n_colors=n))


def plot_embedding_scatter_by_clusters(
    embedding: np.ndarray,
    member_ids: list[str],
    clusters: dict[str, list[str]],
    medoid_lookup: dict[str, str],
    output_path: str | Path,
    *,
    T: float | None = None,
    method_label: str = "",
    max_legend_clusters: int = 30,
    label_top_k: int | None = None,
    singleton_color: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 0.55),
) -> None:
    """Scatter of every domain in 2D, colored by its cluster.

    Each multi-member cluster gets a distinct color (size-descending
    palette assignment). Cluster identity is shown via a numbered marker
    placed at the cluster centroid: a small filled circle in the cluster
    color with the cluster index drawn inside in white. The legend on the
    right resolves ``[N]`` → medoid + size. This avoids the original
    in-plot text labels that hid points beneath them.

    Singletons are drawn in light grey beneath everything else.
    """
    if label_top_k is None:
        label_top_k = max_legend_clusters
    member_index = {mid: i for i, mid in enumerate(member_ids)}
    sized_clusters = sorted(
        clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )
    multi_clusters = [(rep, m) for rep, m in sized_clusters if len(m) >= 2]
    singletons = [(rep, m) for rep, m in sized_clusters if len(m) == 1]
    palette = _cluster_color_palette(max(len(multi_clusters), 1))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Singletons sit underneath.
    if singletons:
        sx, sy = [], []
        for _, members in singletons:
            i = member_index.get(members[0])
            if i is None:
                continue
            sx.append(embedding[i, 0])
            sy.append(embedding[i, 1])
        if sx:
            ax.scatter(
                sx, sy, c=[singleton_color], s=10, alpha=0.55,
                edgecolors="none", zorder=1,
                label=f"singletons (n={len(sx)})",
            )

    # Multi-member clusters: small markers, low alpha → less occlusion.
    cluster_centroids = []  # (rank, color, cx, cy, medoid, size)
    for rank, (rep, members) in enumerate(multi_clusters):
        idxs = [member_index[m] for m in members if m in member_index]
        if not idxs:
            continue
        pts = embedding[idxs]
        color = palette[rank % len(palette)]
        medoid = medoid_lookup.get(rep, rep)
        size = len(members)
        if rank < max_legend_clusters:
            label = f"[{rank + 1:>3}] {medoid[:34]} (n={size})"
        else:
            label = None
        ax.scatter(
            pts[:, 0], pts[:, 1], c=[color], s=10, alpha=0.55,
            edgecolors="none", zorder=2,
            label=label,
        )
        cluster_centroids.append(
            (rank, color, float(pts[:, 0].mean()), float(pts[:, 1].mean()),
             medoid, size)
        )

    # Centroid markers ON TOP: bordered colored circle with the cluster
    # index drawn inside. Doesn't hide points (only one marker per cluster).
    for rank, color, cx, cy, _, _ in cluster_centroids[:label_top_k]:
        ax.scatter(
            [cx], [cy], s=260, c=[color], edgecolors="white",
            linewidths=1.6, zorder=20,
        )
        ax.text(
            cx, cy, str(rank + 1),
            ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=21,
        )

    title_bits = [method_label, f"T = {T:.2f}" if T is not None else ""]
    title = " — ".join(b for b in title_bits if b) or "Cluster scatter"
    title += (
        f"\n{len(multi_clusters)} multi-member clusters, "
        f"{len(singletons)} singletons   "
        f"(numbered markers = cluster centroids; legend keys to medoid)"
    )
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title=title)

    # Legend on the right — numbers match the centroid markers.
    if multi_clusters or singletons:
        # Use 2 columns once the legend grows past a sensible single-column length.
        n_legend = min(len(multi_clusters), max_legend_clusters) + (
            1 if singletons else 0
        )
        ncol = 2 if n_legend > 24 else 1
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=7, ncol=ncol, frameon=False,
            handlelength=1.0, columnspacing=0.8,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_scatter_by_metadata(
    embedding: np.ndarray,
    member_ids: list[str],
    metadata_df: pd.DataFrame,
    color_by: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    palette_name: str = "tab10",
) -> None:
    """Reference scatter colored by a metadata column (kingdom or domain type).

    Provides an "intrinsic" view of the embedding — what the *biology*
    looks like before any clustering is applied. Useful to overlay with
    cluster scatters when picking a threshold.
    """
    metadata_index = metadata_df.set_index("module_id")
    fig, ax = plt.subplots(figsize=(10, 7))
    if color_by not in metadata_index.columns:
        ax.text(0.5, 0.5, f"missing {color_by}",
                ha="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    aligned = metadata_index.reindex(member_ids)[color_by].fillna("unknown")
    categories = sorted(aligned.unique())
    palette = sns.color_palette(palette_name, n_colors=max(len(categories), 3))
    cat_to_color = {c: palette[i % len(palette)] for i, c in enumerate(categories)}
    cat_to_color["unknown"] = (0.7, 0.7, 0.7, 0.5)

    for cat in categories:
        mask = (aligned == cat).values
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cat_to_color[cat]], s=18, alpha=0.85,
            label=f"{cat} (n={int(mask.sum())})",
            edgecolors="white", linewidths=0.25,
        )

    ax.set(
        xlabel="UMAP-1", ylabel="UMAP-2",
        title=title or f"UMAP embedding colored by {color_by}",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_kingdom_frac_distribution(
    stats_df: pd.DataFrame, T: float, output_path: str | Path
) -> None:
    """Histogram of per-cluster top-kingdom fraction (n ≥ 2 clusters only)."""
    multi = stats_df[stats_df["n"] >= 2]
    fig, ax = plt.subplots(figsize=(7, 4))
    if multi.empty:
        ax.text(0.5, 0.5, "no n≥2 clusters", ha="center", transform=ax.transAxes)
    else:
        sns.histplot(
            multi["top_kingdom_frac"], bins=np.linspace(0, 1, 21),
            ax=ax, color="seagreen", edgecolor="black",
        )
        median_val = float(multi["top_kingdom_frac"].median())
        ax.axvline(
            median_val, color="black", linestyle="--", alpha=0.6,
            label=f"median = {median_val:.2f}",
        )
        ax.legend()
        ax.set(
            xlabel="top-kingdom fraction in cluster",
            ylabel="# clusters (n ≥ 2)",
            title=f"Per-cluster kingdom purity (T={T:.2f})",
            xlim=(0, 1),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
