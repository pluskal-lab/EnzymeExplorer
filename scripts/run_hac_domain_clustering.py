"""Hierarchical Agglomerative Clustering (average linkage) on (1 − TM) distances.

Sweeps a list of TM cut thresholds on the same dendrogram. Cuts nest
**by construction** because the dendrogram is a single tree, and the
dendrogram itself is a publishable artefact for a structural-nomenclature
paper.

Reuses the cached pairwise-TM lookup at ``--shared-aln-dir`` (a
``pairwise_tm.pkl`` or ``alignment_usalign.tsv`` written by an earlier
USalign all-vs-all run); when that cache is missing the analysis layer
runs USalign over ``--pdb-dir`` to populate it.

Outputs (under ``--output-dir``):

  domain_metadata.csv                           — one row per detected domain
  intermediate/distance_matrix.npy              — densified (1 − TM)
  intermediate/linkage_matrix.npy               — scipy linkage
  intermediate/member_ids.pkl                   — order of rows in the matrix
  intermediate/linkage_meta.json                — method + cophenetic correlation
  clusters/T<NN>_clusters.json                  — {cluster_id: [members]}
  analysis/cluster_stats_T<NN>.csv              — per-cluster summary
  analysis/transitional_domains_T<NN>.csv       — borderline-domain detection
  analysis/sweep_summary.csv                    — per-T summary
  analysis/plots/dendrogram_*.png               — dendrograms
  analysis/plots/n_clusters_vs_height.png
  analysis/plots/sweep_overview.png
  analysis/plots/<per-T diagnostic plots>.png
"""
from __future__ import annotations

# Same import-order dance: PyMOL must come before numpy.
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
from pathlib import Path  # noqa: E402

import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enzymeexplorer.src.domain_clustering import (  # noqa: E402
    analysis,
    embedding as embedding_mod,
    hac,
    plots,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hac_domain_clustering")

DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
# → [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pdb-dir",
        default="data/detected_domains/martsDB_detected_domains/domains",
        help="Used only as a fallback if the alignment cache is missing.",
    )
    p.add_argument(
        "--detected-domains-pkl",
        default="data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl",
    )
    p.add_argument(
        "--martsdb-csv",
        default="data/martsDB_reactions_2026_02_22_preprocessed.csv",
    )
    p.add_argument(
        "--output-dir",
        default="data/domain_clustering/martsDB_hac_sweep",
    )
    p.add_argument(
        "--shared-aln-dir",
        default="data/domain_clustering/martsDB_hac_sweep/all_vs_all",
        help=(
            "Directory holding the all-vs-all pairwise-TM cache "
            "(``pairwise_tm.pkl`` / ``alignment_usalign.tsv``). When the "
            "cache is absent USalign is run over ``--pdb-dir`` to populate "
            "it."
        ),
    )
    p.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help="TM cut thresholds (HAC cuts at 1 − T).",
    )
    p.add_argument(
        "--linkage-method", default="average",
        choices=("average", "single", "complete", "weighted"),
        help="HAC linkage method. Average (UPGMA) is the default.",
    )
    p.add_argument(
        "--missing-distance", type=float, default=1.0,
        help="Distance used for pairs absent from the TM lookup (filtered "
             "out by --c 0.8 coverage at search time). 1.0 = maximally "
             "dissimilar; matches Foldseek's set-cover convention.",
    )
    p.add_argument(
        "--margin-threshold", type=float, default=0.05,
        help="Transitional-domain margin threshold per cut.",
    )
    p.add_argument(
        "--top-transitional-domains", type=int, default=30,
    )
    p.add_argument(
        "--max-clusters-in-stacked-plots", type=int, default=30,
    )
    p.add_argument(
        "--threads", type=int, default=8,
        help="Used only if the alignment cache needs to be rebuilt.",
    )
    p.add_argument(
        "--force-recompute-linkage", action="store_true",
        help="Rebuild the distance matrix and linkage even if cached.",
    )
    p.add_argument("--scatter-n-neighbors", type=int, default=30)
    p.add_argument("--scatter-min-dist", type=float, default=0.0)
    p.add_argument("--scatter-random-state", type=int, default=42)
    p.add_argument("--force-embedding", action="store_true")
    p.add_argument(
        "--force-regenerate-plots", action="store_true",
        help="Re-run cluster-stats + transitional analysis + plots for "
             "thresholds that already have outputs on disk. By default a "
             "T value with an existing cluster_stats_T<NN>.csv is skipped, "
             "so re-runs only do the work for newly-added thresholds.",
    )
    return p.parse_args()


def _save_clusters_json(clusters: dict[str, list[str]], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(clusters, f)


def _save_stats_csv(stats_df: pd.DataFrame, path: Path) -> None:
    df = stats_df.copy()
    for col in (
        "kingdom_distribution",
        "canonical_domain_type_distribution",
        "reaction_label_distribution",
        "members",
    ):
        if col in df.columns:
            df[col] = df[col].apply(json.dumps)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)

    # 1) Domain metadata.
    logger.info("Loading domain metadata")
    metadata_df = analysis.load_domain_metadata(
        args.detected_domains_pkl, args.martsdb_csv
    )
    metadata_df.to_csv(output_dir / "domain_metadata.csv", index=False)

    # 2) Pairwise TM lookup (uses shared cache).
    pairwise_tm = analysis.load_or_compute_pairwise_tm(
        aln_dir=args.shared_aln_dir,
        pdb_dir=args.pdb_dir,
        threads=args.threads,
    )

    # 3) Distance matrix + linkage. Cache the heavy step.
    cached = (
        None if args.force_recompute_linkage
        else hac.load_intermediate(intermediate_dir)
    )
    if cached is not None:
        member_ids, distance_matrix, linkage_matrix = cached
        cophenetic_correlation = None
    else:
        member_ids = sorted(metadata_df["module_id"].unique())
        logger.info(
            "Building distance matrix for %d members (missing → %.2f)",
            len(member_ids), args.missing_distance,
        )
        distance_matrix = hac.build_distance_matrix(
            member_ids, pairwise_tm, missing_distance=args.missing_distance,
        )
        linkage_matrix = hac.compute_linkage(
            distance_matrix, method=args.linkage_method,
        )
        cophenetic_correlation = hac.compute_cophenetic_correlation(
            linkage_matrix, distance_matrix,
        )
        logger.info("Cophenetic correlation: %.4f", cophenetic_correlation)
        hac.save_intermediate(
            intermediate_dir,
            member_ids=member_ids,
            distance_matrix=distance_matrix,
            linkage_matrix=linkage_matrix,
            cophenetic_correlation=cophenetic_correlation,
            method=args.linkage_method,
        )

    # 4a) UMAP embedding (cached under the same ``--shared-aln-dir``).
    embedding, embedding_member_ids = embedding_mod.load_or_compute_embedding(
        cache_dir=Path(args.shared_aln_dir) / "embedding",
        member_ids=member_ids,
        pairwise_tm=pairwise_tm,
        n_neighbors=args.scatter_n_neighbors,
        min_dist=args.scatter_min_dist,
        random_state=args.scatter_random_state,
        force=args.force_embedding,
    )
    plots.plot_embedding_scatter_by_metadata(
        embedding, embedding_member_ids, metadata_df,
        color_by="kingdom",
        output_path=plots_dir / "scatter_by_kingdom.png",
        title="HAC sweep — UMAP layout colored by kingdom",
        palette_name="tab10",
    )
    plots.plot_embedding_scatter_by_metadata(
        embedding, embedding_member_ids, metadata_df,
        color_by="canonical_domain_type",
        output_path=plots_dir / "scatter_by_domain_type.png",
        title="HAC sweep — UMAP layout colored by canonical domain type",
        palette_name="Set2",
    )

    # 4b) Dendrogram plots — once, not per-T.
    plots.plot_dendrogram_truncated(
        linkage_matrix, plots_dir / "dendrogram_truncated.png",
        tm_thresholds=args.thresholds, p=50,
    )
    plots.plot_dendrogram_full(
        linkage_matrix, plots_dir / "dendrogram_full.pdf",
        tm_thresholds=args.thresholds,
    )
    plots.plot_dendrogram_with_leaf_categories(
        linkage_matrix, member_ids, metadata_df,
        plots_dir / "dendrogram_with_leaf_annotations.png",
        tm_thresholds=args.thresholds,
    )
    # Alternative dendrogram: clade widths proportional to leaf counts.
    # Two collapse points so the user can compare granularity.
    for collapse_at_tm in (0.5, 0.6, 0.7):
        plots.plot_dendrogram_proportional(
            linkage_matrix, member_ids, metadata_df,
            plots_dir / f"dendrogram_proportional_collapsed_at_TM{collapse_at_tm:.2f}.png",
            collapse_at_tm=collapse_at_tm,
            tm_thresholds=args.thresholds,
        )

    # 5) #clusters vs height curve — sample more densely than the cut sweep.
    fine_grid = np.arange(0.30, 0.95 + 1e-9, 0.01)
    n_vs_T = hac.n_clusters_vs_threshold(linkage_matrix, fine_grid)
    n_vs_T.to_csv(output_dir / "analysis" / "n_clusters_vs_T.csv", index=False)
    plots.plot_n_clusters_vs_height(
        n_vs_T, plots_dir / "n_clusters_vs_height.png",
        tm_thresholds_marked=args.thresholds,
    )

    # 6) Per-T sweep: cut, analyse, plot.
    for T in args.thresholds:
        T_str = f"{T:.2f}"
        logger.info("================ T = %s ================", T_str)

        stats_csv_existing = (
            output_dir / "analysis" / f"cluster_stats_T{T_str}.csv"
        )
        if stats_csv_existing.exists() and not args.force_regenerate_plots:
            logger.info(
                "T=%s: existing results found — skipping "
                "(--force-regenerate-plots to redo)",
                T_str,
            )
            continue

        clusters = hac.cut_at_threshold(linkage_matrix, member_ids, T)
        _save_clusters_json(
            clusters, output_dir / "clusters" / f"T{T_str}_clusters.json"
        )
        logger.info("T=%s → %d clusters", T_str, len(clusters))

        stats_df = analysis.cluster_stats(clusters, pairwise_tm, metadata_df)
        _save_stats_csv(
            stats_df,
            output_dir / "analysis" / f"cluster_stats_T{T_str}.csv",
        )

        plots.plot_cluster_size_distribution(
            stats_df, T, plots_dir / f"cluster_sizes_T{T_str}.png"
        )
        plots.plot_intra_tm_vs_size(
            stats_df, T, plots_dir / f"intra_tm_vs_size_T{T_str}.png"
        )
        plots.plot_kingdom_distribution(
            stats_df, T, plots_dir / f"kingdom_per_cluster_T{T_str}.png",
            max_clusters=args.max_clusters_in_stacked_plots,
        )
        plots.plot_canonical_domain_type_distribution(
            stats_df, T, plots_dir / f"domain_type_per_cluster_T{T_str}.png",
            max_clusters=args.max_clusters_in_stacked_plots,
        )
        plots.plot_reaction_label_distribution(
            stats_df, T, plots_dir / f"reaction_type_per_cluster_T{T_str}.png",
            max_clusters=args.max_clusters_in_stacked_plots,
        )
        plots.plot_frac_irrelevant_per_cluster(
            stats_df, T, plots_dir / f"frac_irrelevant_per_cluster_T{T_str}.png",
        )
        plots.plot_top_kingdom_frac_distribution(
            stats_df, T, plots_dir / f"kingdom_purity_hist_T{T_str}.png",
        )

        transitional_df = analysis.find_transitional_domains(
            clusters, pairwise_tm, margin_threshold=args.margin_threshold,
        )
        transitional_df.to_csv(
            output_dir / "analysis" / f"transitional_domains_T{T_str}.csv",
            index=False,
        )
        plots.plot_transitional_scatter(
            transitional_df, T,
            plots_dir / f"transitional_scatter_T{T_str}.png",
            margin_threshold=args.margin_threshold,
        )
        plots.plot_transitional_margin_histogram(
            transitional_df, T,
            plots_dir / f"transitional_margin_hist_T{T_str}.png",
            margin_threshold=args.margin_threshold,
        )
        plots.plot_top_transitional_domains(
            transitional_df, T,
            plots_dir / f"transitional_top_T{T_str}.png",
            top_n=args.top_transitional_domains,
        )
        plots.plot_transitional_flow(
            transitional_df, T,
            plots_dir / f"transitional_flow_T{T_str}.png",
        )

        medoid_lookup = {
            row["foldseek_rep"]: row["medoid"]
            for _, row in stats_df.iterrows()
        }
        plots.plot_embedding_scatter_by_clusters(
            embedding, embedding_member_ids, clusters, medoid_lookup,
            output_path=plots_dir / f"scatter_T{T_str}.png",
            T=T, method_label="HAC",
        )

    # Sweep summary built from every cluster_stats CSV on disk — keeps
    # the summary in sync with the full persisted T set even when this
    # run only processed a subset.
    sweep_df = analysis.rebuild_sweep_summary_from_disk(
        output_dir / "analysis", metadata_df,
    )
    sweep_csv = output_dir / "analysis" / "sweep_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    plots.plot_sweep_overview(sweep_df, plots_dir / "sweep_overview.png")
    logger.info("Sweep summary: %s", sweep_csv)

    logger.info("================ HAC done ================")


if __name__ == "__main__":
    main()
