"""Hierarchical Agglomerative Clustering on (1 − TM) distances.

HAC is the natural complement to Foldseek's set-cover clustering for a
nomenclature pass: cuts at multiple thresholds nest **by construction**,
the dendrogram itself is publishable as the structural-subtype tree, and
the cut threshold is a literal quantitative definition.

The pairwise TM lookup produced by Foldseek's all-vs-all search is sparse
(pairs filtered out by ``-c 0.8`` coverage are absent). For HAC we densify
into ``D[i, j] = 1 − TM(i, j)``, with missing pairs set to a maximum
distance of 1.0 — i.e. "no significant alignment found" is treated as
"maximally dissimilar," which matches Foldseek's set-cover convention.

Average linkage (UPGMA) is the default — it's the standard for
TM-score-based phylogenetic-style trees and gives interpretable
cluster heights (the height *is* the mean (1 − TM) of the merged
groups). ``ward`` requires Euclidean distances and is not appropriate
here.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.cluster.hierarchy import (  # type: ignore
    cophenet, fcluster, linkage,
)
from scipy.spatial.distance import squareform  # type: ignore

logger = logging.getLogger(__name__)


def build_distance_matrix(
    member_ids: list[str],
    pairwise_tm: dict[tuple[str, str], float],
    *,
    missing_distance: float = 1.0,
) -> np.ndarray:
    """Densify the sparse TM lookup into a symmetric (n, n) distance matrix.

    ``D[i, j] = 1 − TM(i, j)``; pairs absent from the lookup get
    ``missing_distance`` (default 1.0). Diagonal is 0. The output is
    clipped to [0, 1] and forced symmetric.
    """
    n = len(member_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    idx = {mid: i for i, mid in enumerate(member_ids)}
    D = np.full((n, n), missing_distance, dtype=np.float32)
    np.fill_diagonal(D, 0.0)

    n_filled = 0
    for (a, b), tm in pairwise_tm.items():
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            continue
        d = 1.0 - float(tm)
        if d < D[ia, ib]:
            D[ia, ib] = d
            D[ib, ia] = d
            n_filled += 1
    np.clip(D, 0.0, 1.0, out=D)
    total_pairs = n * (n - 1) // 2
    logger.info(
        "Distance matrix: %dx%d, filled %d / %d off-diagonal pairs (%.2f%% coverage)",
        n, n, n_filled, total_pairs, 100.0 * n_filled / max(total_pairs, 1),
    )
    return D


def compute_linkage(
    distance_matrix: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """Run scipy linkage on the condensed distance vector."""
    n = distance_matrix.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 members for linkage, got {n}")
    logger.info("Running %s-linkage on %d members", method, n)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=method)
    logger.info("Linkage matrix: shape=%s", Z.shape)
    return Z


def compute_cophenetic_correlation(
    linkage_matrix: np.ndarray, distance_matrix: np.ndarray
) -> float:
    """Cophenetic correlation: how faithfully the tree preserves pairwise distances.

    Values close to 1.0 mean the tree is a good summary of the
    pairwise (1 − TM) distances; close to 0 means the linkage choice
    distorts the original geometry. Useful diagnostic for the
    nomenclature paper.
    """
    condensed = squareform(distance_matrix, checks=False)
    coph_corr, _ = cophenet(linkage_matrix, condensed)
    return float(coph_corr)


def cut_at_threshold(
    linkage_matrix: np.ndarray,
    member_ids: list[str],
    tm_threshold: float,
) -> dict[str, list[str]]:
    """Cut the dendrogram at distance ``1 − tm_threshold``.

    Returns ``{cluster_id: [member_id, ...]}``. Cluster IDs are synthetic
    strings ``hac_<int>`` since HAC has no natural representative; the
    medoid-swap step (downstream, in ``analysis.cluster_stats``) replaces
    the synthetic id with the actual archetypal member.
    """
    cut_distance = 1.0 - tm_threshold
    labels = fcluster(linkage_matrix, t=cut_distance, criterion="distance")
    clusters: dict[str, list[str]] = {}
    for label, mid in zip(labels, member_ids):
        clusters.setdefault(f"hac_{int(label)}", []).append(mid)
    return clusters


def n_clusters_vs_threshold(
    linkage_matrix: np.ndarray,
    tm_thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """For diagnostic plotting: # clusters at each cut threshold."""
    rows = []
    for T in tm_thresholds:
        labels = fcluster(linkage_matrix, t=1.0 - float(T), criterion="distance")
        rows.append({"tmscore_threshold": float(T), "n_clusters": int(labels.max())})
    return pd.DataFrame(rows)


def save_intermediate(
    output_dir: str | Path,
    *,
    member_ids: list[str],
    distance_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    cophenetic_correlation: float | None = None,
    method: str = "average",
) -> None:
    """Persist the distance matrix + linkage so cuts can be redone cheaply."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "member_ids.pkl", "wb") as f:
        pickle.dump(member_ids, f)
    np.save(output_dir / "distance_matrix.npy", distance_matrix)
    np.save(output_dir / "linkage_matrix.npy", linkage_matrix)
    meta = {"method": method, "n_members": len(member_ids)}
    if cophenetic_correlation is not None:
        meta["cophenetic_correlation"] = cophenetic_correlation
    pd.Series(meta).to_json(output_dir / "linkage_meta.json", indent=2)
    logger.info("Saved HAC intermediates → %s", output_dir)


def load_intermediate(
    output_dir: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """Reload a previously cached (member_ids, distance_matrix, linkage_matrix).

    Returns None if the cache is absent.
    """
    output_dir = Path(output_dir).resolve()
    member_ids_path = output_dir / "member_ids.pkl"
    distance_matrix_path = output_dir / "distance_matrix.npy"
    linkage_path = output_dir / "linkage_matrix.npy"
    if not (
        member_ids_path.exists()
        and distance_matrix_path.exists()
        and linkage_path.exists()
    ):
        return None
    with open(member_ids_path, "rb") as f:
        member_ids = pickle.load(f)
    distance_matrix = np.load(distance_matrix_path)
    linkage_matrix = np.load(linkage_path)
    logger.info(
        "Reusing HAC intermediates from %s  (n=%d)", output_dir, len(member_ids),
    )
    return member_ids, distance_matrix, linkage_matrix
