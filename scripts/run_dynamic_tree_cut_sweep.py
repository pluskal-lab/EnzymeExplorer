"""Sweep dynamicTreeCut over (deepSplit × minClusterSize) hyperparameters.

Runs every (deepSplit, minClusterSize) combination on the cached HAC
linkage (produced by ``run_hac_domain_clustering.py``) and emits, per
configuration:

  * ``dendrogram_d{D}_m{M}.png``       — full dendrogram with branches
                                         coloured by clade and three
                                         leaf stripes (clade / kingdom /
                                         canonical domain type)
  * ``metrics_d{D}_m{M}.png``          — 6-panel per-clade overview:
                                         cluster sizes, compactness vs
                                         separation, kingdom purity,
                                         reaction-type purity,
                                         transitional fraction,
                                         bootstrap support
  * ``kingdoms_per_clade_d{D}_m{M}.png`` and
    ``reactions_per_clade_d{D}_m{M}.png`` — stacked-bar composition plots
  * ``clade_table_d{D}_m{M}.csv``      — long-form per-clade metric table,
                                         including ``label`` (semantic
                                         majority-domain-type label) and
                                         ``representative`` (medoid)
  * ``sweep_summary.csv``              — aggregate metrics per config
  * ``sweep_overview.png``             — across-config heatmap comparison

The subsampling-jackknife bootstrap trees are precomputed once and reused
across all configs so a 100-iteration support estimate costs the same
regardless of how many configs are swept.
"""
from __future__ import annotations

import os as _os  # noqa: E402
for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
import seaborn as sns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enzymeexplorer.src.domain_clustering import (  # noqa: E402
    analysis,
    clade_detection,
    hac,
    plots,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dtc_sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--hac-dir",
        default="data/domain_clustering/martsDB_hac_sweep",
    )
    p.add_argument(
        "--detected-domains-pkl",
        default="data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl",
    )
    p.add_argument(
        "--martsdb-csv",
        default="data/martsDB_reactions_2026_02_22_preprocessed.csv",
    )
    p.add_argument("--output-subdir", default="dtc_sweep")
    p.add_argument(
        "--deep-splits", type=int, nargs="+", default=[0, 1, 2],
    )
    p.add_argument(
        "--min-cluster-sizes", type=int, nargs="+",
        default=[5, 10, 15, 20],
    )
    p.add_argument("--bootstrap-iter", type=int, default=100)
    p.add_argument("--bootstrap-keep-frac", type=float, default=0.80)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--linkage-method", default="average")
    p.add_argument("--margin-threshold", type=float, default=0.05,
                   help="Threshold below which a domain is called transitional.")
    p.add_argument(
        "--parent-config-id", default=None,
        help=(
            "Optional config_id (e.g. 'd0_m20') treated as the level-1 "
            "(coarse) parent grouping. Children of the same parent get "
            "letter suffixes (alpha1A, alpha1B). Without this flag, every "
            "clade is labelled at level 1 only (alpha1, alpha2, …)."
        ),
    )
    return p.parse_args()


def _per_clade_table(
    clusters: dict[str, list[str]],
    pairwise_tm: dict[tuple[str, str], float],
    distance_matrix: np.ndarray,
    member_ids: list[str],
    metadata_df: pd.DataFrame,
    boot_support: dict[str, dict],
    margin_threshold: float,
    stats: pd.DataFrame | None = None,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-clade metric table for one config.

    ``stats`` may be passed in pre-computed (so callers that also need the
    raw stats_df for stacked-bar plotting can avoid recomputing it).
    """
    if stats is None:
        stats = analysis.cluster_stats(clusters, pairwise_tm, metadata_df)
    # Transitional fraction per clade.
    trans = analysis.find_transitional_domains(
        clusters, pairwise_tm, margin_threshold=margin_threshold,
    )
    if not trans.empty and "home_rep" in trans.columns:
        per_cluster_trans = (
            trans.groupby("home_rep")["is_transitional"].mean().to_dict()
        )
    else:
        per_cluster_trans = {}

    rows = []
    for _, r in stats.iterrows():
        cid = r["foldseek_rep"]
        rows.append({
            "clade_id": cid,
            "label": (label_map or {}).get(cid, cid),
            "representative": r["medoid"],
            "n": int(r["n"]),
            "mean_intra_tm": float(r["mean_intra_tm"])
                if not pd.isna(r["mean_intra_tm"]) else float("nan"),
            "mean_inter_tm": float(r.get("mean_inter_tm", float("nan")))
                if not pd.isna(r.get("mean_inter_tm", float("nan"))) else float("nan"),
            "top_kingdom": r["top_kingdom"],
            "top_kingdom_frac": float(r["top_kingdom_frac"]),
            "top_reaction_label": r["top_reaction_label"],
            "top_reaction_label_frac": float(r["top_reaction_label_frac"]),
            "reaction_label_perplexity": float(
                r.get("reaction_label_perplexity", float("nan"))
            ),
            "transitional_frac": float(per_cluster_trans.get(cid, 0.0)),
            "bootstrap_support": float(boot_support.get(cid, {}).get(
                "support_frac", float("nan")
            )),
            "is_unassigned": cid == "dtc_unassigned",
        })
    return pd.DataFrame(rows).sort_values(
        ["is_unassigned", "n"], ascending=[True, False]
    ).reset_index(drop=True)


def _aggregate_summary(
    config_id: str, deep_split: int, min_size: int,
    clusters: dict[str, list[str]],
    table: pd.DataFrame,
) -> dict:
    """Aggregate metrics over all (n ≥ 2, non-unassigned) clades for one config."""
    multi = table[(~table["is_unassigned"]) & (table["n"] >= 2)]
    n_total = sum(len(m) for m in clusters.values())
    if multi.empty:
        return {
            "config_id": config_id, "deep_split": deep_split,
            "min_cluster_size": min_size,
            "n_clusters": len(clusters), "n_clusters_n_ge_2": 0,
            "n_unassigned": int(table.loc[table["is_unassigned"], "n"].sum()
                                if table["is_unassigned"].any() else 0),
            "n_total_domains": n_total,
        }
    return {
        "config_id": config_id,
        "deep_split": deep_split,
        "min_cluster_size": min_size,
        "n_clusters": len(clusters),
        "n_clusters_n_ge_2": int(len(multi)),
        "n_unassigned": int(table.loc[table["is_unassigned"], "n"].sum()
                            if table["is_unassigned"].any() else 0),
        "n_total_domains": n_total,
        "mean_intra_tm_weighted": float(
            (multi["mean_intra_tm"] * multi["n"]).sum() / max(multi["n"].sum(), 1)
        ),
        "mean_inter_tm_weighted": float(
            (multi["mean_inter_tm"] * multi["n"]).sum() / max(multi["n"].sum(), 1)
        ),
        "kingdom_purity_weighted": float(
            (multi["top_kingdom_frac"] * multi["n"]).sum() / max(multi["n"].sum(), 1)
        ),
        "reaction_purity_weighted": float(
            (multi["top_reaction_label_frac"] * multi["n"]).sum()
            / max(multi["n"].sum(), 1)
        ),
        "transitional_frac_weighted": float(
            (multi["transitional_frac"] * multi["n"]).sum() / max(multi["n"].sum(), 1)
        ),
        "bootstrap_support_weighted": float(
            (multi["bootstrap_support"].fillna(0) * multi["n"]).sum()
            / max(multi["n"].sum(), 1)
        ),
        "frac_clades_boot_ge_0.7": float((multi["bootstrap_support"] >= 0.7).mean()),
    }


def _plot_sweep_overview(
    sweep_df: pd.DataFrame, output_path: Path,
) -> None:
    """Heatmaps of aggregate metrics across (deepSplit, minClusterSize)."""
    metrics = [
        ("n_clusters",                 "# clades",                       "viridis"),
        ("frac_clades_boot_ge_0.7",    "frac clades w/ boot ≥ 0.7",      "Greens"),
        ("kingdom_purity_weighted",    "kingdom purity (weighted)",      "Blues"),
        ("reaction_purity_weighted",   "reaction purity (weighted)",     "Oranges"),
        ("transitional_frac_weighted", "transitional frac (weighted)",   "Reds"),
        ("mean_intra_tm_weighted",     "intra-TM (compactness)",         "Purples"),
    ]
    deep_splits = sorted(sweep_df["deep_split"].unique())
    min_sizes = sorted(sweep_df["min_cluster_size"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, (col, label, cmap) in enumerate(metrics):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        pivot = sweep_df.pivot(
            index="deep_split", columns="min_cluster_size", values=col,
        ).reindex(index=deep_splits, columns=min_sizes)
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, annot=True,
            fmt=".0f" if col == "n_clusters" else ".3f",
            cbar_kws={"label": label}, linewidths=0.5, linecolor="white",
        )
        ax.set(xlabel="minClusterSize", ylabel="deepSplit", title=label)
    fig.suptitle("dynamicTreeCut sweep — aggregate metrics", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    hac_dir = Path(args.hac_dir).resolve()
    out_dir = hac_dir / args.output_subdir
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cached = hac.load_intermediate(hac_dir / "intermediate")
    if cached is None:
        raise SystemExit(f"No cached HAC intermediate at {hac_dir}")
    member_ids, distance_matrix, linkage_matrix = cached
    logger.info("Loaded cached linkage: %d members", len(member_ids))

    metadata_df = analysis.load_domain_metadata(
        args.detected_domains_pkl, args.martsdb_csv,
    )

    # Reconstruct pairwise_tm dict from the densified distance matrix
    # (we need it for analysis.cluster_stats and find_transitional_domains).
    logger.info("Rehydrating pairwise_tm dict from distance matrix")
    pairwise_tm: dict[tuple[str, str], float] = {}
    n = len(member_ids)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_matrix[i, j])
            if d < 1.0 - 1e-9:  # skip "missing" entries
                a, b = member_ids[i], member_ids[j]
                if a > b:
                    a, b = b, a
                pairwise_tm[(a, b)] = 1.0 - d
    logger.info("  %d observed pairs", len(pairwise_tm))

    # Member-id → canonical-domain-type lookup (for clade-label majority vote).
    md_idx = metadata_df.set_index("module_id")
    member_to_canonical = (
        md_idx["canonical_domain_type"].to_dict() if "canonical_domain_type" in md_idx.columns else {}
    )

    # Optional level-1 parent grouping for level-2 letter suffixes.
    parent_clades: dict[str, list[str]] | None = None
    if args.parent_config_id:
        parent_path = out_dir / f"clades_{args.parent_config_id}.json"
        if parent_path.exists():
            with open(parent_path) as f:
                parent_clades = json.load(f)
            logger.info(
                "Using parent config '%s' (%d parent clades) for level-2 labels",
                args.parent_config_id, len(parent_clades) - (1 if "dtc_unassigned" in parent_clades else 0),
            )
        else:
            logger.warning("Parent config %s not found at %s — proceeding with level-1 labels only",
                           args.parent_config_id, parent_path)

    # Precompute bootstrap trees once, reuse across all 15 configs.
    boot_trees = clade_detection.precompute_bootstrap_trees(
        distance_matrix,
        n_iter=args.bootstrap_iter,
        leaf_keep_frac=args.bootstrap_keep_frac,
        linkage_method=args.linkage_method,
        seed=args.bootstrap_seed,
    )

    summary_rows: list[dict] = []
    for ds in args.deep_splits:
        for ms in args.min_cluster_sizes:
            config_id = f"d{ds}_m{ms}"
            logger.info("================ deepSplit=%d minSize=%d ================", ds, ms)
            clades = clade_detection.dynamic_tree_cut(
                linkage_matrix, distance_matrix, member_ids,
                min_cluster_size=ms, deep_split=ds, pam_stage=True,
            )
            with open(out_dir / f"clades_{config_id}.json", "w") as f:
                json.dump(clades, f)

            # Bootstrap support against the precomputed trees.
            clades_for_boot = {
                cid: members for cid, members in clades.items()
                if cid != "dtc_unassigned"
            }
            boot_support = clade_detection.bootstrap_support_from_trees(
                boot_trees, member_ids, clades_for_boot,
            )

            stats_df = analysis.cluster_stats(clades, pairwise_tm, metadata_df)
            label_map = clade_detection.compute_clade_labels(
                clades, member_to_canonical, parent_clusters=parent_clades,
            )
            table = _per_clade_table(
                clades, pairwise_tm, distance_matrix, member_ids,
                metadata_df, boot_support, args.margin_threshold,
                stats=stats_df, label_map=label_map,
            )
            table.to_csv(out_dir / f"clade_table_{config_id}.csv", index=False)

            subtitle = (
                f"deepSplit={ds}, minClusterSize={ms}, "
                f"{len(clades_for_boot)} clades"
                + (f" + {len(clades) - len(clades_for_boot)} unassigned"
                   if len(clades) > len(clades_for_boot) else "")
            )
            plots.plot_dendrogram_full_with_clade_categories(
                linkage_matrix, member_ids, metadata_df, clades,
                plots_dir / f"dendrogram_{config_id}.png",
                title=f"deepSplit={ds}, minClusterSize={ms} ({len(clades_for_boot)} clades)",
                label_map=label_map,
            )
            plots.plot_clade_metrics_panel(
                table, plots_dir / f"metrics_{config_id}.png",
                title=f"DTC metrics — deepSplit={ds}, minClusterSize={ms}",
                label_map=label_map, representative_col="representative",
            )
            plots.plot_kingdom_distribution(
                stats_df, T=0.0,
                output_path=plots_dir / f"kingdoms_per_clade_{config_id}.png",
                subtitle=subtitle,
                label_map=label_map, label_key_col="foldseek_rep",
            )
            plots.plot_reaction_label_distribution(
                stats_df, T=0.0,
                output_path=plots_dir / f"reactions_per_clade_{config_id}.png",
                subtitle=subtitle,
                label_map=label_map, label_key_col="foldseek_rep",
            )

            summary_rows.append(
                _aggregate_summary(config_id, ds, ms, clades_for_boot, table)
            )

    sweep_df = pd.DataFrame(summary_rows)
    sweep_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    _plot_sweep_overview(sweep_df, plots_dir / "sweep_overview.png")
    logger.info("================ done ================")
    logger.info("Sweep summary:\n%s", sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
