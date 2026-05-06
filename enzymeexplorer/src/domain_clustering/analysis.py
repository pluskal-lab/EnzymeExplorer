"""Cluster-level analysis: metadata join, medoid swap, intra-TM stats, kingdom enrichment.

The pairwise-TM lookup is built once from the all-vs-all Foldseek search
output and reused across every threshold in the sweep. Per cluster we:
  * pick the medoid (member with highest mean intra-cluster TM) — this is
    the "medoid swap" that replaces Foldseek's set-cover representative,
  * compute mean and std of pairwise TM among cluster members,
  * report kingdom and canonical-domain-type composition.
"""
from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


# Domain-type consolidation: ``ids`` and ``zeta`` are alpha-fold variants
# detected against alternate templates; for analysis they're all "alpha".
DOMAIN_TYPE_CANONICAL = {
    "alpha": "alpha",
    "ids": "alpha",
    "zeta": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
}


# OriginalType values that are non-cyclase chain-extension / head-group
# enzymes. Verified empirically against the MartsDB CSV: each of these
# appears only with Class 1 and is functionally distinct from the
# Class-1 cyclases. We keep their label as bare OriginalType (no
# ``Class N`` suffix) so they get their own bucket in purity calculations.
PT_LIKE_ORIGINAL_TYPES = {"pt", "psy", "sqs"}

# Sentinel used when a domain's parent sequence has no reactions in
# MartsDB at all (e.g. detected on a sequence not present in the CSV).
# Every domain gets at least one reaction-type label.
NO_REACTIONS_LABEL = "irrelevant"


def _reaction_label(original_type: str, class_value) -> str:
    """Build the combined reaction-type label.

    pt / psy / sqs → bare OriginalType (no class suffix).
    Otherwise: ``"<OriginalType> Class <N>"`` if Class is present,
    else just ``OriginalType``.
    """
    if pd.isna(original_type):
        return NO_REACTIONS_LABEL
    ot = str(original_type)
    if ot in PT_LIKE_ORIGINAL_TYPES:
        return ot
    if class_value is None or pd.isna(class_value):
        return ot
    try:
        return f"{ot} Class {int(class_value)}"
    except (TypeError, ValueError):
        return ot


def load_domain_metadata(
    detected_domains_pkl: str | Path,
    martsdb_csv: str | Path,
) -> pd.DataFrame:
    """Build per-domain metadata: module_id → seq_id, domain_type, kingdom.

    Columns: module_id, seq_id, domain_type, canonical_domain_type, kingdom.
    Domains whose seq_id is absent from the martsDB CSV get kingdom=NaN.
    """
    # Lazy import so module-level ``import enzymeexplorer.src.domain_clustering.analysis``
    # doesn't pull in PyMOL (which requires a specific libstdc++ ordering vs
    # numpy). Unpickling needs the class at this exact path.
    from enzymeexplorer.src.structure_processing.structural_algorithms import (  # noqa: F401
        MappedRegion,
    )

    with open(detected_domains_pkl, "rb") as f:
        seq_id_to_regions = pickle.load(f)

    rows = []
    for seq_id, regions in seq_id_to_regions.items():
        for region in regions:
            rows.append(
                {
                    "module_id": region.module_id,
                    "seq_id": seq_id,
                    "domain_type": region.domain,
                    "canonical_domain_type": DOMAIN_TYPE_CANONICAL.get(
                        region.domain, region.domain
                    ),
                }
            )
    domains_df = pd.DataFrame(rows)
    logger.info(
        "Loaded %d detected domains across %d sequences",
        len(domains_df), domains_df["seq_id"].nunique(),
    )

    martsdb = pd.read_csv(martsdb_csv)
    required_cols = ("Enzyme_marts_ID", "Kingdom", "OriginalType", "Class")
    missing = [c for c in required_cols if c not in martsdb.columns]
    if missing:
        raise ValueError(
            f"martsDB CSV missing columns {missing}; "
            f"got {list(martsdb.columns)[:10]}"
        )
    seq_meta = (
        martsdb[["Enzyme_marts_ID", "Kingdom"]]
        .drop_duplicates(subset=["Enzyme_marts_ID"])
        .rename(columns={"Enzyme_marts_ID": "seq_id", "Kingdom": "kingdom"})
    )
    domains_df = domains_df.merge(seq_meta, on="seq_id", how="left")

    # Build per-sequence list of (OriginalType, Class) reaction tags.
    # Many sequences have multiple rows in MartsDB (one per substrate /
    # product); we keep only the unique (OriginalType, Class) pairs per
    # sequence so a promiscuous synthase that makes 7 sesquiterpenes
    # contributes one "sesq Class 1" tag, not seven.
    seq_to_reactions: dict[str, list[tuple[str, object]]] = {}
    for _, row in martsdb.iterrows():
        seq_id = row["Enzyme_marts_ID"]
        ot = row["OriginalType"]
        cls = row["Class"]
        # Use a tuple-set per sequence to dedupe.
        seq_to_reactions.setdefault(seq_id, []).append((ot, cls))
    seq_to_reactions = {
        s: sorted(set(tuple(r) for r in rs), key=lambda x: (str(x[0]), str(x[1])))
        for s, rs in seq_to_reactions.items()
    }

    # Each domain inherits ALL reaction-type labels of its parent sequence,
    # regardless of catalytic class. The earlier role-based filter (alpha →
    # Class 1 only, beta/γ/δ/ε → Class 2 only) was dropped because the
    # catalytic-role analysis did not support that strict mapping; we now
    # let every domain count every reaction its enzyme performs. Domains
    # whose parent sequence has no reactions in MartsDB get the
    # ``NO_REACTIONS_LABEL`` sentinel so every row has at least one label.
    reaction_labels_per_domain: list[list[str]] = []
    for _, row in domains_df.iterrows():
        seq_id = row["seq_id"]
        labels: set[str] = set()
        for ot, cls in seq_to_reactions.get(seq_id, []):
            labels.add(_reaction_label(ot, cls))
        if not labels:
            labels = {NO_REACTIONS_LABEL}
        reaction_labels_per_domain.append(sorted(labels))

    domains_df["reaction_labels"] = reaction_labels_per_domain
    domains_df["has_no_reactions"] = [
        labels == [NO_REACTIONS_LABEL]
        for labels in reaction_labels_per_domain
    ]
    n_no_rxn = int(domains_df["has_no_reactions"].sum())
    logger.info(
        "Reaction labels assigned: %d / %d domains have only the "
        "'%s' label (parent sequence has no reactions in MartsDB)",
        n_no_rxn, len(domains_df), NO_REACTIONS_LABEL,
    )
    return domains_df


def load_or_compute_pairwise_tm(
    aln_dir: str | Path,
    pdb_dir: str | Path,
    *,
    force_search: bool = False,
    threads: int = 8,
) -> dict[tuple[str, str], float]:
    """Resolve the pairwise-TM lookup from the cache directory.

    Resolution order:
      1. ``pairwise_tm.pkl`` exists and ``force_search=False`` → load.
      2. ``alignment_usalign.tsv`` exists → parse via USalign parser
         and persist the lookup as a pickle for subsequent runs.
      3. otherwise → run all-vs-all USalign on the PDBs in ``pdb_dir``.
    """
    import pickle

    from enzymeexplorer.src.domain_clustering import usalign_runner

    aln_dir = Path(aln_dir).resolve()
    aln_dir.mkdir(parents=True, exist_ok=True)
    aln_pkl = aln_dir / "pairwise_tm.pkl"
    aln_tsv_usalign = aln_dir / "alignment_usalign.tsv"

    if not force_search and aln_pkl.exists():
        logger.info("Reusing cached pairwise-TM lookup %s", aln_pkl)
        with open(aln_pkl, "rb") as f:
            return pickle.load(f)

    if not force_search and aln_tsv_usalign.exists():
        logger.info("Parsing USalign alignment table %s", aln_tsv_usalign)
        pairwise_tm = usalign_runner.parse_alignment_tsv(aln_tsv_usalign)
        with open(aln_pkl, "wb") as f:
            pickle.dump(pairwise_tm, f)
        logger.info("Cached parsed lookup → %s", aln_pkl)
        return pairwise_tm

    logger.info(
        "No cache — running all-vs-all USalign over %s", pdb_dir,
    )
    tsv_path = usalign_runner.run_all_vs_all(
        pdb_dir=pdb_dir, output_dir=aln_dir, n_jobs=threads,
    )
    pairwise_tm = usalign_runner.parse_alignment_tsv(tsv_path)
    with open(aln_pkl, "wb") as f:
        pickle.dump(pairwise_tm, f)
    logger.info("Cached parsed lookup → %s", aln_pkl)
    return pairwise_tm


def medoid_and_intra_stats(
    members: list[str],
    pairwise_tm: dict[tuple[str, str], float],
) -> tuple[str, float, float, int]:
    """Pick the medoid and compute intra-cluster TM statistics.

    Returns ``(medoid, mean_intra_tm, std_intra_tm, n_observed_pairs)``.
    Pairs missing from ``pairwise_tm`` (filtered out by coverage at search
    time) are treated as 0.0 — same convention Foldseek's clustering uses.
    """
    n = len(members)
    if n == 1:
        return members[0], float("nan"), float("nan"), 0

    per_member: dict[str, list[float]] = {m: [] for m in members}
    pair_scores: list[float] = []
    n_observed = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = members[i], members[j]
            a, b = (mi, mj) if mi <= mj else (mj, mi)
            score = pairwise_tm.get((a, b))
            if score is None:
                score = 0.0
            else:
                n_observed += 1
            per_member[mi].append(score)
            per_member[mj].append(score)
            pair_scores.append(score)

    medoid = max(per_member, key=lambda m: float(np.mean(per_member[m])))
    arr = np.asarray(pair_scores, dtype=np.float64)
    return medoid, float(arr.mean()), float(arr.std()), n_observed


def cluster_stats(
    clusters: dict[str, list[str]],
    pairwise_tm: dict[tuple[str, str], float],
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cluster summary.

    Adds three reaction-type fields on top of the kingdom + domain-type
    purity layer:
      * ``top_reaction_label`` / ``top_reaction_label_frac`` — most-common
        reaction tag and its share of all tag occurrences in the cluster.
        Each domain contributes ALL reaction labels of its parent sequence
        (no role-based filtering); a domain whose parent makes N reactions
        contributes N tag occurrences.
      * ``reaction_label_distribution`` — full Counter dict.
      * ``frac_irrelevant_domains`` — fraction of cluster members whose
        only label is the ``irrelevant`` sentinel (parent sequence has no
        reactions in MartsDB at all). Surfaced separately so reviewers can
        judge the reliability of the reaction-purity number.
    """
    metadata_index = metadata_df.set_index("module_id")
    has_reaction_labels = "reaction_labels" in metadata_index.columns
    has_irrelevant_flag = "has_no_reactions" in metadata_index.columns

    member_to_cluster = {
        m: rep for rep, members in clusters.items() for m in members
    }
    all_members = set(member_to_cluster)

    rows = []
    for rep, members in clusters.items():
        medoid, mean_tm, std_tm, n_obs = medoid_and_intra_stats(members, pairwise_tm)

        # Inter-cluster mean TM: average TM from this cluster's members to
        # all members of OTHER clusters, over the pairs actually observed in
        # the pairwise_tm lookup. Complements intra-cluster TM: low values
        # mean clusters are well-separated structurally, so a clustering
        # that maximises intra (compact) and minimises inter (separated)
        # is the goal.
        members_set = set(members)
        other_members = all_members - members_set
        inter_tms: list[float] = []
        for a in members:
            for b in other_members:
                key = (a, b) if a <= b else (b, a)
                tm = pairwise_tm.get(key)
                if tm is not None:
                    inter_tms.append(tm)
        mean_inter_tm = float(np.mean(inter_tms)) if inter_tms else float("nan")
        n_observed_inter_pairs = len(inter_tms)

        meta = metadata_index.reindex(members)
        kingdoms = meta["kingdom"].dropna().tolist()
        domain_types = meta["canonical_domain_type"].dropna().tolist()

        kingdom_counts = Counter(kingdoms)
        if kingdom_counts:
            top_kingdom, top_kingdom_n = kingdom_counts.most_common(1)[0]
            kingdom_frac = top_kingdom_n / len(kingdoms)
        else:
            top_kingdom, kingdom_frac = "", 0.0

        dt_counts = Counter(domain_types)
        if dt_counts:
            top_dt, top_dt_n = dt_counts.most_common(1)[0]
            dt_frac = top_dt_n / len(domain_types)
        else:
            top_dt, dt_frac = "", 0.0

        # Reaction-type purity (domain-frequency weighting). Every domain
        # contributes the full set of its parent sequence's reaction
        # labels, irrespective of catalytic class.
        rxn_counts: Counter = Counter()
        n_irrelevant = 0
        if has_reaction_labels:
            for labels in meta["reaction_labels"]:
                if labels is None:
                    continue
                # ``labels`` is a list[str]; in the worst case [irrelevant].
                rxn_counts.update(labels)
            if has_irrelevant_flag:
                n_irrelevant = int(meta["has_no_reactions"].sum())
            else:
                n_irrelevant = sum(
                    1 for labels in meta["reaction_labels"]
                    if labels == [NO_REACTIONS_LABEL]
                )
        if rxn_counts:
            top_rxn, top_rxn_n = rxn_counts.most_common(1)[0]
            total_rxn_tags = sum(rxn_counts.values())
            rxn_purity = top_rxn_n / total_rxn_tags
            # Effective number of distinct reaction types (perplexity =
            # exp Shannon entropy). Complements top-label purity: 1.0 means
            # the cluster is reaction-monomorphic; higher values mean a
            # mix of multiple reaction types.
            probs = np.array(
                [c / total_rxn_tags for c in rxn_counts.values()],
                dtype=float,
            )
            entropy = float(-(probs * np.log(probs)).sum())
            rxn_perplexity = float(np.exp(entropy))
        else:
            top_rxn = NO_REACTIONS_LABEL
            rxn_purity = 0.0
            rxn_perplexity = float("nan")

        n = len(members)
        n_total_pairs = n * (n - 1) // 2
        rows.append(
            {
                "foldseek_rep": rep,
                "medoid": medoid,
                "n": n,
                "mean_intra_tm": mean_tm,
                "std_intra_tm": std_tm,
                "n_observed_pairs": n_obs,
                "n_total_pairs": n_total_pairs,
                "mean_inter_tm": mean_inter_tm,
                "n_observed_inter_pairs": n_observed_inter_pairs,
                "top_kingdom": top_kingdom,
                "top_kingdom_frac": kingdom_frac,
                "kingdom_distribution": dict(kingdom_counts),
                "top_canonical_domain_type": top_dt,
                "top_canonical_domain_type_frac": dt_frac,
                "canonical_domain_type_distribution": dict(dt_counts),
                "top_reaction_label": top_rxn,
                "top_reaction_label_frac": rxn_purity,
                "reaction_label_distribution": dict(rxn_counts),
                "reaction_label_perplexity": rxn_perplexity,
                "frac_irrelevant_domains": n_irrelevant / max(n, 1),
                "members": list(members),
            }
        )

    df = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return df


def find_transitional_domains(
    clusters: dict[str, list[str]],
    pairwise_tm: dict[tuple[str, str], float],
    *,
    margin_threshold: float = 0.05,
    min_alt_tm: float = 0.0,
) -> pd.DataFrame:
    """Find domains that sit on the border between two clusters.

    For every member ``d``, computes the **mean TM** between ``d`` and the
    OTHER members of its assigned (home) cluster, then the mean TM between
    ``d`` and every other cluster's full membership. The "best alternative"
    is the non-home cluster with the highest such mean TM.

    The ``margin = home_mean_tm - alt_mean_tm`` measures how confidently
    ``d`` belongs to its home cluster — small / negative margins mark
    transitional domains that could plausibly be reassigned.

    Args:
      clusters: ``{rep: [member, ...]}`` from one Foldseek clustering.
      pairwise_tm: symmetrised TM lookup keyed by sorted ``(a, b)``.
      margin_threshold: a domain is *flagged* transitional when
        ``margin <= margin_threshold``. Default 0.05.
      min_alt_tm: only consider an alternative cluster if its mean TM to
        ``d`` is at least this value. Avoids spurious "best alternatives"
        with vanishingly small TM-scores.

    Returns a DataFrame sorted by ascending margin (most-transitional first).
    Columns: module_id, home_rep, home_size, home_mean_tm, n_home_observed,
    alt_rep, alt_size, alt_mean_tm, n_alt_observed, margin, is_transitional.
    Domains in singleton clusters (no home neighbors) are excluded — there's
    no "home mean" to compare against.
    """
    # Pre-compute member → home_rep so each domain knows its assignment.
    member_to_rep: dict[str, str] = {}
    for rep, members in clusters.items():
        for m in members:
            member_to_rep[m] = rep

    cluster_sizes = {rep: len(ms) for rep, ms in clusters.items()}

    def _mean_tm_to_set(d: str, others: list[str]) -> tuple[float, int]:
        """Mean TM between d and each member of `others` (excluding d itself)."""
        total = 0.0
        n_obs = 0
        n_total = 0
        for m in others:
            if m == d:
                continue
            n_total += 1
            a, b = (d, m) if d <= m else (m, d)
            score = pairwise_tm.get((a, b))
            if score is not None:
                total += score
                n_obs += 1
            # Missing pair counted as 0.0 in the denominator → consistent
            # with cluster_stats and Foldseek's set-cover convention.
        if n_total == 0:
            return float("nan"), 0
        return total / n_total, n_obs

    rows = []
    for d, home_rep in member_to_rep.items():
        home_members = clusters[home_rep]
        if len(home_members) < 2:
            continue  # singleton: no home neighbors → skip
        home_mean, home_obs = _mean_tm_to_set(d, home_members)

        # Best alternative: the non-home cluster with the highest mean TM.
        best_alt_rep = ""
        best_alt_mean = float("-inf")
        best_alt_obs = 0
        for rep, members in clusters.items():
            if rep == home_rep:
                continue
            alt_mean, alt_obs = _mean_tm_to_set(d, members)
            if alt_mean is None or pd.isna(alt_mean):
                continue
            if alt_mean < min_alt_tm:
                continue
            if alt_mean > best_alt_mean:
                best_alt_mean = alt_mean
                best_alt_rep = rep
                best_alt_obs = alt_obs

        if best_alt_rep == "":
            best_alt_mean_for_row = 0.0
        else:
            best_alt_mean_for_row = best_alt_mean
        margin = home_mean - best_alt_mean_for_row

        rows.append(
            {
                "module_id": d,
                "home_rep": home_rep,
                "home_size": cluster_sizes[home_rep],
                "home_mean_tm": home_mean,
                "n_home_observed": home_obs,
                "alt_rep": best_alt_rep,
                "alt_size": cluster_sizes.get(best_alt_rep, 0),
                "alt_mean_tm": best_alt_mean_for_row,
                "n_alt_observed": best_alt_obs,
                "margin": margin,
                "is_transitional": bool(margin <= margin_threshold),
            }
        )

    df = pd.DataFrame(rows).sort_values("margin", ascending=True).reset_index(drop=True)
    return df


def load_cluster_stats(csv_path: str | Path) -> pd.DataFrame:
    """Load a cluster_stats CSV and rehydrate JSON-encoded columns."""
    import json

    df = pd.read_csv(csv_path)
    for col in (
        "kingdom_distribution",
        "canonical_domain_type_distribution",
        "reaction_label_distribution",
        "members",
    ):
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df


def rebuild_sweep_summary_from_disk(
    analysis_dir: str | Path,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Walk every ``cluster_stats_T<NN>.csv`` in ``analysis_dir`` and build
    the cross-T summary. Augments each row with transitional metrics if the
    matching ``transitional_domains_T<NN>.csv`` is present.

    Used so re-runs that process only a subset of thresholds still emit a
    sweep_summary.csv reflecting **every** threshold persisted on disk.
    """
    import re

    analysis_dir = Path(analysis_dir).resolve()
    pattern = re.compile(r"cluster_stats_T(\d+\.\d+)\.csv$")
    rows: list[dict] = []
    for csv_path in sorted(analysis_dir.glob("cluster_stats_T*.csv")):
        m = pattern.search(csv_path.name)
        if not m:
            continue
        T = float(m.group(1))
        stats_df = load_cluster_stats(csv_path)
        row = sweep_summary_row(T, stats_df, metadata_df)

        trans_path = analysis_dir / f"transitional_domains_T{T:.2f}.csv"
        if trans_path.exists():
            trans_df = pd.read_csv(trans_path)
            if not trans_df.empty:
                row["transitional_frac"] = float(trans_df["is_transitional"].mean())
                row["transitional_count"] = int(trans_df["is_transitional"].sum())
                row["median_margin"] = float(trans_df["margin"].median())
            else:
                row["transitional_frac"] = float("nan")
                row["transitional_count"] = 0
                row["median_margin"] = float("nan")
        else:
            row["transitional_frac"] = float("nan")
            row["transitional_count"] = 0
            row["median_margin"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tmscore_threshold").reset_index(drop=True)


def sweep_summary_row(
    tmscore_threshold: float,
    stats_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict:
    """One-line summary statistics for a single TM threshold."""
    n_clusters = len(stats_df)
    n_singletons = int((stats_df["n"] == 1).sum())
    multi = stats_df[stats_df["n"] >= 2]
    has_reaction = "top_reaction_label_frac" in stats_df.columns
    has_perplexity = "reaction_label_perplexity" in stats_df.columns
    has_inter = "mean_inter_tm" in stats_df.columns
    if not multi.empty:
        n_pairs = (multi["n"] * (multi["n"] - 1) / 2).astype(float)
        mean_intra_tm_weighted = float(
            (multi["mean_intra_tm"] * n_pairs).sum() / max(n_pairs.sum(), 1.0)
        )
        kingdom_purity_weighted = float(
            (multi["top_kingdom_frac"] * multi["n"]).sum()
            / max(multi["n"].sum(), 1.0)
        )
        domain_type_purity_weighted = float(
            (multi["top_canonical_domain_type_frac"] * multi["n"]).sum()
            / max(multi["n"].sum(), 1.0)
        )
        if has_reaction:
            reaction_type_purity_weighted = float(
                (multi["top_reaction_label_frac"] * multi["n"]).sum()
                / max(multi["n"].sum(), 1.0)
            )
        else:
            reaction_type_purity_weighted = float("nan")
        if has_perplexity:
            valid = multi["reaction_label_perplexity"].notna()
            denom = float(multi.loc[valid, "n"].sum()) or 1.0
            reaction_label_perplexity_weighted = float(
                (multi.loc[valid, "reaction_label_perplexity"]
                 * multi.loc[valid, "n"]).sum() / denom
            )
        else:
            reaction_label_perplexity_weighted = float("nan")
    else:
        mean_intra_tm_weighted = float("nan")
        kingdom_purity_weighted = float("nan")
        domain_type_purity_weighted = float("nan")
        reaction_type_purity_weighted = float("nan")
        reaction_label_perplexity_weighted = float("nan")

    # Inter-cluster mean TM is computed across ALL clusters (singletons
    # included — a singleton still has cross-cluster pairs to weigh in).
    # Weight each cluster's mean_inter_tm by its observed cross-cluster
    # pair count: each distinct cross-cluster pair contributes to both
    # endpoint clusters, so the double-counting in numerator and
    # denominator cancels and the weighted mean equals the global mean
    # over distinct observed cross-cluster pairs.
    if has_inter:
        valid = stats_df["mean_inter_tm"].notna() & (
            stats_df["n_observed_inter_pairs"] > 0
        )
        if valid.any():
            w = stats_df.loc[valid, "n_observed_inter_pairs"].astype(float)
            mean_inter_tm_weighted = float(
                (stats_df.loc[valid, "mean_inter_tm"] * w).sum() / max(w.sum(), 1.0)
            )
        else:
            mean_inter_tm_weighted = float("nan")
    else:
        mean_inter_tm_weighted = float("nan")

    n_total_domains = int(stats_df["n"].sum())
    return {
        "tmscore_threshold": tmscore_threshold,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "singleton_frac": n_singletons / max(n_clusters, 1),
        "n_clusters_n_ge_2": int((stats_df["n"] >= 2).sum()),
        "n_total_domains": n_total_domains,
        "max_cluster_size": int(stats_df["n"].max()) if len(stats_df) else 0,
        "mean_cluster_size": (
            float(stats_df["n"].mean()) if len(stats_df) else float("nan")
        ),
        "median_cluster_size": (
            float(stats_df["n"].median()) if len(stats_df) else float("nan")
        ),
        "mean_intra_tm_weighted": mean_intra_tm_weighted,
        "mean_inter_tm_weighted": mean_inter_tm_weighted,
        "kingdom_purity_weighted": kingdom_purity_weighted,
        "domain_type_purity_weighted": domain_type_purity_weighted,
        "reaction_type_purity_weighted": reaction_type_purity_weighted,
        "reaction_label_perplexity_weighted": reaction_label_perplexity_weighted,
    }
