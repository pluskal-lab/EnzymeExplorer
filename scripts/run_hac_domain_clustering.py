"""Hierarchical Agglomerative Clustering (average linkage) on (1 − TM) distances.

Produces the linkage matrix + supporting intermediates consumed by
``run_dynamic_tree_cut_sweep.py`` and ``run_domain_subtype_labeling.py``.
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
  n_clusters_vs_T.csv                           — #clusters at each height
  plots/dendrogram_truncated.png                — overview dendrogram
  plots/dendrogram_full.pdf                     — full dendrogram
  plots/n_clusters_vs_height.png                — cluster-count curve
  plots/scatter_by_kingdom.png                  — UMAP by kingdom
  plots/scatter_by_domain_type.png              — UMAP by canonical domain type
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
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # type: ignore  # noqa: E402

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

# Reference TM heights annotated on the dendrogram plots (informational).
DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]


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
        default="outputs/domain_clustering",
    )
    p.add_argument(
        "--shared-aln-dir",
        default="data/domain_clustering/all_vs_all",
        help=(
            "Directory holding the all-vs-all pairwise-TM cache "
            "(``pairwise_tm.pkl`` / ``alignment_usalign.tsv``). When the "
            "cache is absent USalign is run over ``--pdb-dir`` to populate "
            "it."
        ),
    )
    p.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help="TM heights to annotate on the dendrogram plots (informational).",
    )
    p.add_argument(
        "--linkage-method", default="average",
        choices=("average", "single", "complete", "weighted"),
        help="HAC linkage method. Average (UPGMA) is the default.",
    )
    p.add_argument(
        "--missing-distance", type=float, default=1.0,
        help="Distance used for pairs absent from the TM lookup (1.0 = "
             "maximally dissimilar).",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
        title="HAC — UMAP layout colored by kingdom",
        palette_name="tab10",
    )
    plots.plot_embedding_scatter_by_metadata(
        embedding, embedding_member_ids, metadata_df,
        color_by="canonical_domain_type",
        output_path=plots_dir / "scatter_by_domain_type.png",
        title="HAC — UMAP layout colored by canonical domain type",
        palette_name="Set2",
    )

    # 4b) Dendrogram plots.
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
    for collapse_at_tm in (0.5, 0.6, 0.7):
        plots.plot_dendrogram_proportional(
            linkage_matrix, member_ids, metadata_df,
            plots_dir / f"dendrogram_proportional_collapsed_at_TM{collapse_at_tm:.2f}.png",
            collapse_at_tm=collapse_at_tm,
            tm_thresholds=args.thresholds,
        )

    # 5) #clusters vs height curve.
    fine_grid = np.arange(0.30, 0.95 + 1e-9, 0.01)
    n_vs_T = hac.n_clusters_vs_threshold(linkage_matrix, fine_grid)
    n_vs_T.to_csv(output_dir / "n_clusters_vs_T.csv", index=False)
    plots.plot_n_clusters_vs_height(
        n_vs_T, plots_dir / "n_clusters_vs_height.png",
        tm_thresholds_marked=args.thresholds,
    )

    logger.info("================ HAC done ================")


if __name__ == "__main__":
    main()
