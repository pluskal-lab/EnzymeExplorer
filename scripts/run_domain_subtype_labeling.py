"""Produce the canonical domain-subtype labeling for d0_m5.

The dynamicTreeCut sweep (``run_dynamic_tree_cut_sweep.py``) is the
exploratory tool that emits clades + auto-labels for many (deepSplit,
minClusterSize) configs. This script is the production entry point: it
runs a SINGLE config (default ``deepSplit=0, minClusterSize=5``),
computes the auto-labels, optionally applies a manually curated
``clade_id -> label`` override JSON, and writes the canonical artefacts:

  * ``data/domain_module_id_2_domain_subtype.pkl`` —
    ``{module_id: final_label}`` (overwritten by default)
  * ``<dtc_dir>/clade_table_<config_id>_final.csv`` — long-form per-clade
    table with both ``auto_label`` and ``final_label`` columns so the
    override audit trail stays on disk.

Reproducibility: the linkage matrix and bootstrap trees are loaded from
the existing HAC cache, dynamicTreeCut is deterministic given fixed
inputs, and ``compute_clade_labels`` tie-breaks by ``clade_id`` so the
auto-label is identical across re-runs.

Usage::

    # Default config, no overrides — recomputes auto-labels and overwrites pickle.
    python scripts/run_domain_subtype_labeling.py

    # With a manually curated override map (clade_id -> label).
    python scripts/run_domain_subtype_labeling.py \\
        --label-overrides data/domain_subtype_label_overrides.json
"""
from __future__ import annotations

import os as _os
for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402

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
logger = logging.getLogger("subtype_labeling")


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
    p.add_argument("--dtc-subdir", default="dtc_sweep")
    p.add_argument("--deep-split", type=int, default=0)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--bootstrap-iter", type=int, default=100)
    p.add_argument("--bootstrap-keep-frac", type=float, default=0.80)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--linkage-method", default="average")
    p.add_argument(
        "--margin-threshold", type=float, default=0.05,
        help="Threshold below which a domain is called transitional.",
    )
    p.add_argument(
        "--label-overrides",
        default=None,
        help=(
            "Optional path to a JSON file mapping ``clade_id -> label`` "
            "(e.g. ``{\"dtc_1\": \"alpha2D\"}``). Overrides the auto label "
            "for the listed clade_ids; all other clades keep the auto label."
        ),
    )
    p.add_argument(
        "--output-pickle",
        default="data/domain_module_id_2_domain_subtype.pkl",
        help="Where to write the {module_id: final_label} pickle.",
    )
    p.add_argument(
        "--output-table",
        default=None,
        help=(
            "Where to write the canonical clade_table CSV. Defaults to "
            "``<hac-dir>/<dtc-subdir>/clade_table_d{D}_m{M}_final.csv``."
        ),
    )
    p.add_argument(
        "--skip-bootstrap", action="store_true",
        help=(
            "Skip bootstrap-support computation (faster, but the metrics "
            "panel will show NaN for bootstrap support). Off by default."
        ),
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help=(
            "Skip the four diagnostic plots (dendrogram, metrics, kingdoms, "
            "reactions). By default all are produced under "
            "``<hac-dir>/<dtc-subdir>/plots/`` with the ``_final`` suffix."
        ),
    )
    return p.parse_args()


def _per_clade_table(
    clusters: dict[str, list[str]],
    pairwise_tm: dict[tuple[str, str], float],
    metadata_df: pd.DataFrame,
    boot_support: dict[str, dict],
    margin_threshold: float,
    auto_label_map: dict[str, str],
    final_label_map: dict[str, str],
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-clade metric table with both auto and final labels."""
    if stats is None:
        stats = analysis.cluster_stats(clusters, pairwise_tm, metadata_df)
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
            "auto_label": auto_label_map.get(cid, cid),
            "final_label": final_label_map.get(cid, cid),
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


def main() -> None:
    args = parse_args()
    hac_dir = Path(args.hac_dir).resolve()
    dtc_dir = hac_dir / args.dtc_subdir
    dtc_dir.mkdir(parents=True, exist_ok=True)
    config_id = f"d{args.deep_split}_m{args.min_cluster_size}"

    cached = hac.load_intermediate(hac_dir / "intermediate")
    if cached is None:
        raise SystemExit(
            f"No cached HAC intermediate at {hac_dir}. Run "
            f"run_hac_domain_clustering.py first."
        )
    member_ids, distance_matrix, linkage_matrix = cached
    logger.info("Loaded cached linkage: %d members", len(member_ids))

    metadata_df = analysis.load_domain_metadata(
        args.detected_domains_pkl, args.martsdb_csv,
    )

    # Rehydrate pairwise_tm dict from densified distance matrix for stats.
    logger.info("Rehydrating pairwise_tm dict from distance matrix")
    pairwise_tm: dict[tuple[str, str], float] = {}
    n = len(member_ids)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_matrix[i, j])
            if d < 1.0 - 1e-9:
                a, b = member_ids[i], member_ids[j]
                if a > b:
                    a, b = b, a
                pairwise_tm[(a, b)] = 1.0 - d
    logger.info("  %d observed pairs", len(pairwise_tm))

    md_idx = metadata_df.set_index("module_id")
    member_to_canonical = (
        md_idx["canonical_domain_type"].to_dict()
        if "canonical_domain_type" in md_idx.columns else {}
    )

    logger.info(
        "Running dynamicTreeCut (deepSplit=%d, minClusterSize=%d)",
        args.deep_split, args.min_cluster_size,
    )
    clades = clade_detection.dynamic_tree_cut(
        linkage_matrix, distance_matrix, member_ids,
        min_cluster_size=args.min_cluster_size,
        deep_split=args.deep_split,
        pam_stage=True,
    )
    with open(dtc_dir / f"clades_{config_id}.json", "w") as fh:
        json.dump(clades, fh, indent=2, sort_keys=True)
    logger.info("Wrote clades JSON to %s", dtc_dir / f"clades_{config_id}.json")

    auto_label_map = clade_detection.compute_clade_labels(
        clades, member_to_canonical,
    )

    # Apply overrides if provided.
    overrides: dict[str, str] = {}
    if args.label_overrides:
        ov_path = Path(args.label_overrides)
        with open(ov_path, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        unknown = sorted(set(overrides) - set(clades))
        if unknown:
            logger.warning(
                "%d override key(s) don't match any clade_id and will be "
                "ignored: %s",
                len(unknown), unknown,
            )
            overrides = {k: v for k, v in overrides.items() if k in clades}
        logger.info(
            "Loaded %d label overrides from %s", len(overrides), ov_path,
        )
    final_label_map = {**auto_label_map, **overrides}

    # Bootstrap support for the table (optional).
    if args.skip_bootstrap:
        logger.info("Skipping bootstrap-support computation.")
        boot_support: dict[str, dict] = {}
    else:
        boot_trees = clade_detection.precompute_bootstrap_trees(
            distance_matrix,
            n_iter=args.bootstrap_iter,
            leaf_keep_frac=args.bootstrap_keep_frac,
            linkage_method=args.linkage_method,
            seed=args.bootstrap_seed,
        )
        clades_for_boot = {
            cid: members for cid, members in clades.items()
            if cid != "dtc_unassigned"
        }
        boot_support = clade_detection.bootstrap_support_from_trees(
            boot_trees, member_ids, clades_for_boot,
        )

    stats_df = analysis.cluster_stats(clades, pairwise_tm, metadata_df)
    table = _per_clade_table(
        clades, pairwise_tm, metadata_df, boot_support,
        args.margin_threshold, auto_label_map, final_label_map,
        stats=stats_df,
    )
    table_path = (
        Path(args.output_table) if args.output_table
        else dtc_dir / f"clade_table_{config_id}_final.csv"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    logger.info("Wrote final clade table to %s", table_path)

    # Build module_id -> final_label map. Sort keys deterministically before
    # pickling so the on-disk artefact is byte-stable across runs.
    module_to_label: dict[str, str] = {}
    for cid in sorted(clades):
        label = final_label_map.get(cid, cid)
        for mid in sorted(clades[cid]):
            module_to_label[mid] = label

    # Pickle preserves insertion order in cpython but we sort the dict by
    # module_id for portability.
    pickle_path = Path(args.output_pickle)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as fh:
        pickle.dump(dict(sorted(module_to_label.items())), fh)
    logger.info(
        "Wrote module_id -> final_label pickle (%d entries) to %s",
        len(module_to_label), pickle_path,
    )

    # Quick distribution summary.
    from collections import Counter
    dist = Counter(module_to_label.values())
    logger.info("Final label distribution:")
    for lab, cnt in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
        logger.info("  %s: %d", lab, cnt)

    if not args.no_plots:
        plots_dir = dtc_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        n_real = sum(1 for cid in clades if cid != "dtc_unassigned")
        n_unassigned = len(clades.get("dtc_unassigned", []))
        subtitle = (
            f"deepSplit={args.deep_split}, minClusterSize={args.min_cluster_size}, "
            f"{n_real} clades"
            + (f" + {n_unassigned} unassigned" if n_unassigned else "")
        )

        dendro_path = plots_dir / f"dendrogram_{config_id}_final.png"
        plots.plot_dendrogram_full_with_clade_categories(
            linkage_matrix, member_ids, metadata_df, clades,
            dendro_path,
            title=(
                f"Domain subtype dendrogram — {config_id} "
                f"({n_real} clades, final labels)"
            ),
            label_map=final_label_map,
        )
        logger.info("Wrote dendrogram to %s", dendro_path)

        metrics_path = plots_dir / f"metrics_{config_id}_final.png"
        plots.plot_clade_metrics_panel(
            table, metrics_path,
            title=f"DTC metrics — {config_id} (final labels)",
            label_map=final_label_map, representative_col="representative",
        )
        logger.info("Wrote metrics panel to %s", metrics_path)

        kingdoms_path = plots_dir / f"kingdoms_per_clade_{config_id}_final.png"
        plots.plot_kingdom_distribution(
            stats_df, T=0.0,
            output_path=kingdoms_path,
            subtitle=subtitle,
            label_map=final_label_map, label_key_col="foldseek_rep",
        )
        logger.info("Wrote kingdoms-per-clade plot to %s", kingdoms_path)

        reactions_path = plots_dir / f"reactions_per_clade_{config_id}_final.png"
        plots.plot_reaction_label_distribution(
            stats_df, T=0.0,
            output_path=reactions_path,
            subtitle=subtitle,
            label_map=final_label_map, label_key_col="foldseek_rep",
        )
        logger.info("Wrote reactions-per-clade plot to %s", reactions_path)


if __name__ == "__main__":
    main()
