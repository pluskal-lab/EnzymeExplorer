"""Test which TPS domain types drive each reaction class.

H1 (Class 1):  Class-1 reactions are determined by the **alpha** domain
                alone; beta / gamma / delta / epsilon contribute nothing.

H2 (Class 2):  Class-2 reactions are determined by beta / gamma / delta /
                epsilon equally; alpha contributes nothing.

Preprocessing applied uniformly:
  * zeta detected domains are dropped.
  * alpha + ids merged under canonical "alpha".
  * MartsDB OriginalType ∈ {pt, psy, sqs} dropped.
  * Multi-substrate reactions (substrate SMILES contains ".") dropped.

For every (hypothesis, configuration) we compute pair features over a
restricted subset of sequences sharing the configuration's exact domain
set, then report — per axis — marginal Pearson r and *partial* Pearson r
(with the other axes held constant) against three similarity targets:

    reaction_jaccard
    substrate_max_tanimoto
    product_max_tanimoto_shared_ot   (only over shared OriginalType pairs)

Configurations tested:
  H1.1  alpha+beta+gamma  vs Class-1 reactions   (axes: alpha, beta, gamma)
  H1.2  alpha+beta        vs Class-1 reactions   (axes: alpha, beta)
  H2.1  delta+epsilon     vs Class-2 reactions   (axes: delta, epsilon)
  H2.2  alpha+beta+gamma  vs Class-2 reactions   (axes: alpha, beta, gamma —
                                                   bifunctional subset; tests
                                                   alpha is irrelevant for Class 2)

Outputs (under ``--output-dir``):
  reactions_filtered.csv             — preprocessed MartsDB rows
  sequence_summary.csv               — per-sequence metadata aggregate
  <hypothesis>/<config>/pairs.csv    — full pair-feature table
  <hypothesis>/<config>/correlations.csv
  plots/partial_correlations_overview.png
  plots/<hypothesis>_<config>_<target>.png — scatter for each axis vs each target
  plots/<hypothesis>_<config>_partial_bars.png
"""
from __future__ import annotations

import os as _os  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enzymeexplorer.src.domain_clustering import (  # noqa: E402
    analysis,
    catalytic_role,
    plots,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analyze_catalytic_role")


HYPOTHESES = [
    {
        "name": "H1_class1_alpha_drives",
        "target_class": 1,
        "configs": [
            {"name": "alpha_beta_gamma", "axes": ("alpha", "beta", "gamma")},
            {"name": "alpha_beta",       "axes": ("alpha", "beta")},
        ],
    },
    {
        "name": "H2_class2_non_alpha_drives",
        "target_class": 2,
        "configs": [
            {"name": "delta_epsilon",    "axes": ("delta", "epsilon")},
            {"name": "alpha_beta_gamma", "axes": ("alpha", "beta", "gamma")},
        ],
    },
]


# Per-target weight column emitted by ``catalytic_role.attach_pair_weight_columns``.
TARGET_WEIGHT_COL = {
    "reaction_jaccard": "w_target_reaction_jaccard",
    "substrate_mcs_sim": "w_target_substrate_mcs_sim",
    "product_avg_tanimoto_shared_substrate":
        "w_target_product_avg_tanimoto_shared_substrate",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdb-dir",
                   default="data/detected_domains/martsDB_detected_domains/domains")
    p.add_argument("--detected-domains-pkl",
                   default="data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl")
    p.add_argument("--martsdb-csv",
                   default="data/martsDB_reactions_2026_02_22_preprocessed.csv")
    p.add_argument("--shared-aln-dir",
                   default="data/domain_clustering/martsDB_hac_sweep/all_vs_all")
    p.add_argument("--output-dir",
                   default="data/domain_clustering/catalytic_role_analysis")
    p.add_argument(
        "--seq-cluster-tsv",
        default="data/domain_clustering/martsDB_hac_sweep/seq_clusters/T0.90_cluster.tsv",
        help=(
            "Two-column ``rep<TAB>member`` TSV defining sequence clusters used "
            "for redundancy-aware weighting in the correlation analysis. The "
            "default file collapses paralogs at TM ≥ 0.90 (one representative "
            "per close-paralog group) so structural redundancy doesn't double-"
            "count in the correlation statistics."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_w_dir = output_dir / "plots_weighted"
    plots_w_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading metadata + pairwise TM lookup")
    metadata_df = analysis.load_domain_metadata(
        args.detected_domains_pkl, args.martsdb_csv,
    )
    pairwise_tm = analysis.load_or_compute_pairwise_tm(
        aln_dir=args.shared_aln_dir, pdb_dir=args.pdb_dir,
    )

    logger.info("Filtering reactions (drop pt/psy/sqs + multi-substrate)")
    filtered_rxns = catalytic_role.build_filtered_reactions(args.martsdb_csv)
    filtered_rxns.to_csv(output_dir / "reactions_filtered.csv", index=False)

    logger.info("Loading sequence-cluster map from %s", args.seq_cluster_tsv)
    domain_cluster_map = catalytic_role.load_domain_cluster_map(args.seq_cluster_tsv)
    logger.info("  %d module IDs in cluster map", len(domain_cluster_map))

    logger.info("Building per-sequence summary")
    seq_df = catalytic_role.build_sequence_summary(metadata_df, filtered_rxns)
    seq_df_to_save = seq_df.copy()
    # Stringify dict columns for round-trippable CSV.
    for cls in (1, 2):
        col = f"products_by_ot_class{cls}"
        seq_df_to_save[col] = seq_df_to_save[col].apply(
            lambda d: {k: list(v) for k, v in d.items()}
        )
    seq_df_to_save.to_csv(output_dir / "sequence_summary.csv", index=False)

    # Build substrate similarity matrices ONCE (shared across hypotheses).
    logger.info("Building substrate similarity matrices over unique SMILES")
    unique_smis = catalytic_role.collect_unique_substrate_smiles(seq_df)
    smiles_to_idx = {s: i for i, s in enumerate(unique_smis)}
    sim_matrices = catalytic_role.build_substrate_similarity_matrices(unique_smis)
    # Persist matrices + the SMILES index so users can audit.
    pd.Series(unique_smis, name="smiles").to_csv(
        output_dir / "substrate_smiles_index.csv", index_label="idx",
    )
    for method, M in sim_matrices.items():
        pd.DataFrame(M, index=unique_smis, columns=unique_smis).to_csv(
            output_dir / f"substrate_similarity_matrix_{method}.csv",
        )

    overview_corrs: dict[str, pd.DataFrame] = {}
    overview_corrs_w: dict[str, pd.DataFrame] = {}
    overview_axis_labels: dict[str, tuple[str, ...]] = {}
    overview_pair_dfs: dict[str, pd.DataFrame] = {}

    for hypo in HYPOTHESES:
        h_name = hypo["name"]
        target_class = hypo["target_class"]
        h_dir = output_dir / h_name
        h_dir.mkdir(parents=True, exist_ok=True)
        logger.info("================ %s (target Class %d) ================",
                    h_name, target_class)

        logger.info(
            "Precomputing per-sequence substrate-indices + product FPs (class %d)",
            target_class,
        )
        substrate_indices = catalytic_role.precompute_substrate_indices(
            seq_df, target_class, smiles_to_idx,
        )
        product_fps = catalytic_role.precompute_product_fps_by_substrate(
            seq_df, target_class,
        )

        seq_to_ot = dict(zip(
            seq_df["seq_id"], seq_df[f"originaltypes_class{target_class}"],
        ))

        for cfg in hypo["configs"]:
            cfg_name = cfg["name"]
            axes = cfg["axes"]
            cfg_dir = h_dir / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)
            logger.info("---- %s / %s   axes=%s ----", h_name, cfg_name, axes)

            sub = catalytic_role.filter_to_subset(
                seq_df, require_axes=axes,
                require_class_reactions=target_class,
            )
            logger.info(
                "  %d sequences  → %d unordered pairs",
                len(sub), len(sub) * (len(sub) - 1) // 2,
            )
            if len(sub) < 5:
                logger.info("  too few sequences; skipping")
                continue

            # Composite cluster across the config's domain axes — paralog
            # over-representation in any single axis is deweighted only when
            # we cluster jointly over all relevant axes (otherwise e.g. for
            # H1.1 the γ-paralog redundancy survives an α-only clustering).
            seq_to_cluster = catalytic_role.build_seq_to_cluster(
                seq_df, domain_cluster_map, cluster_axes=axes,
            )
            sub_seqs = set(sub["seq_id"])
            n_clusters = len({seq_to_cluster[s] for s in sub_seqs
                              if s in seq_to_cluster})
            logger.info(
                "  composite cluster axes = %s   →   %d distinct cluster IDs "
                "over %d sequences in this config",
                axes, n_clusters, len(sub_seqs),
            )

            pair_df = catalytic_role.compute_pair_features(
                sub, pairwise_tm,
                target_class=target_class,
                domain_axes=axes,
                substrate_indices_per_seq=substrate_indices,
                substrate_similarity_matrices=sim_matrices,
                product_fps_by_substrate=product_fps,
            )

            # Attach cluster / OT-pair / substrate-bin columns + per-target
            # weights for redundancy-aware analysis. The weighting strategy
            # per target is documented in
            # ``catalytic_role.attach_pair_weight_columns``.
            pair_df = catalytic_role.attach_pair_weight_columns(
                pair_df, seq_to_cluster=seq_to_cluster, seq_to_ot=seq_to_ot,
            )
            pair_df.to_csv(cfg_dir / "pairs.csv", index=False)

            corr = catalytic_role.correlation_summary(pair_df, axes=axes)
            corr.to_csv(cfg_dir / "correlations.csv", index=False)
            logger.info("\n%s", corr.to_string(index=False))

            corr_w = catalytic_role.correlation_summary(
                pair_df, axes=axes,
                weights_per_target=TARGET_WEIGHT_COL,
            )
            corr_w.to_csv(cfg_dir / "correlations_weighted.csv", index=False)
            logger.info("\n[weighted]\n%s", corr_w.to_string(index=False))

            tag = f"{h_name}__{cfg_name}"
            overview_corrs[tag] = corr
            overview_corrs_w[tag] = corr_w
            overview_axis_labels[tag] = axes
            overview_pair_dfs[tag] = pair_df

            # Scatter plots: one figure per (config, target) showing all axes.
            for target_col, target_label in catalytic_role.SIMILARITY_TARGETS:
                plots.plot_role_axes_scatter(
                    pair_df, axes=axes,
                    target_col=target_col, target_label=target_label,
                    output_path=plots_dir / f"scatter_{tag}_{target_col}.png",
                    title_suffix=f"({h_name} / {cfg_name})",
                )
                plots.plot_role_axes_scatter(
                    pair_df, axes=axes,
                    target_col=target_col, target_label=target_label,
                    output_path=plots_w_dir / f"scatter_{tag}_{target_col}.png",
                    title_suffix=f"({h_name} / {cfg_name}, weighted)",
                    weight_col=TARGET_WEIGHT_COL.get(target_col),
                )

            plots.plot_partial_corr_per_axis(
                corr, axes=axes,
                output_path=plots_dir / f"partial_bars_{tag}.png",
                title=f"Partial correlations — {h_name} / {cfg_name}",
            )
            plots.plot_partial_corr_per_axis(
                corr_w, axes=axes,
                output_path=plots_w_dir / f"partial_bars_{tag}.png",
                title=f"Partial correlations (weighted) — {h_name} / {cfg_name}",
            )

            # ---- new diagnostics: variance partition + cross-feature heatmap +
            # target × target relationships + stratified-by-substrate-MCS r ----
            partitions = {}
            partitions_w = {}
            for target_col, target_label in catalytic_role.SIMILARITY_TARGETS:
                if target_col not in pair_df.columns:
                    continue
                partitions[target_label] = catalytic_role.variance_partition(
                    pair_df, axes, target_col,
                )
                partitions_w[target_label] = catalytic_role.variance_partition(
                    pair_df, axes, target_col,
                    weights_col=TARGET_WEIGHT_COL.get(target_col),
                )
            pd.DataFrame(partitions).T.to_csv(
                cfg_dir / "variance_partition.csv",
            )
            pd.DataFrame(partitions_w).T.to_csv(
                cfg_dir / "variance_partition_weighted.csv",
            )
            plots.plot_variance_partition(
                partitions, axes,
                plots_dir / f"variance_partition_{tag}.png",
                title=f"{h_name} / {cfg_name}",
            )
            plots.plot_variance_partition(
                partitions_w, axes,
                plots_w_dir / f"variance_partition_{tag}.png",
                title=f"{h_name} / {cfg_name}  (weighted)",
            )

            plots.plot_weighted_vs_unweighted_comparison(
                corr, corr_w, partitions, partitions_w, axes,
                plots_w_dir / f"weighted_vs_unweighted_{tag}.png",
                title=f"{h_name} / {cfg_name}",
            )

            plots.plot_target_target_matrix(
                pair_df,
                targets=list(catalytic_role.SIMILARITY_TARGETS),
                output_path=plots_dir / f"target_target_matrix_{tag}.png",
                title=f"{h_name} / {cfg_name}",
            )
            plots.plot_target_target_matrix(
                pair_df,
                targets=list(catalytic_role.SIMILARITY_TARGETS),
                output_path=plots_w_dir / f"target_target_matrix_{tag}.png",
                title=f"{h_name} / {cfg_name}",
                weights_per_target=TARGET_WEIGHT_COL,
            )

            plots.plot_feature_correlation_heatmap(
                pair_df, axes,
                targets=list(catalytic_role.SIMILARITY_TARGETS),
                output_path=plots_dir / f"feature_heatmap_{tag}.png",
                title=f"{h_name} / {cfg_name}",
            )
            plots.plot_feature_correlation_heatmap(
                pair_df, axes,
                targets=list(catalytic_role.SIMILARITY_TARGETS),
                output_path=plots_w_dir / f"feature_heatmap_{tag}.png",
                title=f"{h_name} / {cfg_name}",
                weights_per_target=TARGET_WEIGHT_COL,
                default_weight_col="w_cluster",
            )

            # Stratify by substrate-MCS bin, computing TM-vs-target r within
            # each bin for ALL three targets. The bin [1.0] result (pairs
            # with the SAME substrate) is the cleanest catalytic test for
            # each target — substrate-class confounding is removed.
            if "substrate_mcs_sim" in pair_df.columns:
                stratified_targets = [
                    ("reaction_jaccard", "reaction-tag Jaccard", "rxn"),
                    # Don't stratify substrate_mcs_sim — it IS the bin axis.
                    ("product_avg_tanimoto_shared_substrate",
                     "product avg-Tan. (shared substrate)", "prod"),
                ]
                for tcol, tlbl, tslug in stratified_targets:
                    if tcol not in pair_df.columns:
                        continue
                    strat = catalytic_role.stratified_correlations(
                        pair_df, axes, target_col=tcol,
                    )
                    strat.to_csv(
                        cfg_dir / f"stratified_correlations_{tslug}.csv",
                        index=False,
                    )
                    plots.plot_stratified_correlation(
                        strat, axes,
                        plots_dir / f"stratified_{tag}_{tslug}.png",
                        title=f"{h_name} / {cfg_name}  —  bin = substrate-MCS",
                        target_label=tlbl,
                    )

                    strat_w = catalytic_role.stratified_correlations(
                        pair_df, axes, target_col=tcol,
                        weights_col=TARGET_WEIGHT_COL.get(tcol),
                    )
                    strat_w.to_csv(
                        cfg_dir / f"stratified_correlations_{tslug}_weighted.csv",
                        index=False,
                    )
                    plots.plot_stratified_correlation(
                        strat_w, axes,
                        plots_w_dir / f"stratified_{tag}_{tslug}.png",
                        title=(f"{h_name} / {cfg_name}  —  bin = substrate-MCS"
                               "  (weighted)"),
                        target_label=tlbl,
                    )

    if overview_corrs:
        plots.plot_partial_corr_overview(
            overview_corrs, overview_axis_labels,
            plots_dir / "partial_correlations_overview.png",
        )
        plots.plot_partial_corr_overview(
            overview_corrs_w, overview_axis_labels,
            plots_w_dir / "partial_correlations_overview.png",
        )
        plots.plot_target_distributions_overview(
            overview_pair_dfs,
            targets=list(catalytic_role.SIMILARITY_TARGETS),
            output_path=plots_dir / "target_distributions_overview.png",
        )
        plots.plot_target_distributions_overview(
            overview_pair_dfs,
            targets=list(catalytic_role.SIMILARITY_TARGETS),
            output_path=plots_w_dir / "target_distributions_overview.png",
            weights_per_target=TARGET_WEIGHT_COL,
        )

    logger.info("================ done ================")


if __name__ == "__main__":
    main()
