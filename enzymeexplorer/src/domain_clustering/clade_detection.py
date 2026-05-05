"""Adaptive clade detection on a HAC dendrogram.

Two complementary methods, each per-branch adaptive (no single global
TM threshold):

  * :func:`dynamic_tree_cut` — wrapper around the Langfelder-Horvath
    dynamicTreeCut "hybrid" method (WGCNA). Top-down adaptive cuts plus a
    PAM-like reassignment of borderline leaves.

  * :func:`bootstrap_clade_support` (and the precompute / from-trees
    helpers it builds on) — subsampling jackknife. Repeatedly drop a
    random fraction of leaves, recompute linkage on the survivors, and
    ask whether each reference clade's surviving members reappear as a
    monophyletic group in the bootstrap tree. The fraction of bootstrap
    iterations where this holds is the clade's support.

Use :func:`compute_clade_labels` to assign semantic labels (e.g. ``alpha1``,
``delta2``) on top of the raw clade dictionary returned by
``dynamic_tree_cut``.
"""
from __future__ import annotations

import logging

import numpy as np  # type: ignore
from scipy.cluster.hierarchy import linkage  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic tree cut (Langfelder & Horvath, WGCNA)
# ---------------------------------------------------------------------------

def dynamic_tree_cut(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    member_ids: list[str],
    *,
    min_cluster_size: int = 20,
    deep_split: int = 1,
    pam_stage: bool = True,
) -> dict[str, list[str]]:
    """Adaptive clade detection via the dynamicTreeCut hybrid algorithm.

    ``deep_split`` ∈ {0..4}: 0 = fewest, deepest clades (more permissive
    to large clades); 4 = many shallow clades. Default 1 is a moderate
    setting recommended by Langfelder & Horvath for ~thousand-leaf trees.
    """
    from dynamicTreeCut import cutreeHybrid  # type: ignore

    n = len(member_ids)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} != ({n},{n})"
        )

    result = cutreeHybrid(
        link=linkage_matrix,
        distM=distance_matrix.astype(float),
        minClusterSize=min_cluster_size,
        deepSplit=deep_split,
        pamStage=pam_stage,
        verbose=0,
    )
    labels = np.asarray(result["labels"], dtype=int)

    clusters: dict[str, list[str]] = {}
    for label, mid in zip(labels, member_ids):
        # Label 0 in dynamicTreeCut means "unassigned"; keep it as a
        # dedicated bucket so the user can decide what to do with it.
        cid = f"dtc_{int(label)}" if label > 0 else "dtc_unassigned"
        clusters.setdefault(cid, []).append(mid)
    n_real = sum(1 for c in clusters if c != "dtc_unassigned")
    n_unassigned = len(clusters.get("dtc_unassigned", []))
    logger.info(
        "dynamicTreeCut: %d clades + %d unassigned leaves "
        "(minClusterSize=%d, deepSplit=%d)",
        n_real, n_unassigned, min_cluster_size, deep_split,
    )
    return clusters


# ---------------------------------------------------------------------------
# Method 4a — uniform-compactness top-down descent
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Subsampling-jackknife bootstrap support
# ---------------------------------------------------------------------------

def _all_subtree_leaf_sets(
    linkage_matrix: np.ndarray, n_leaves: int,
) -> list[frozenset]:
    """All subtree leaf sets in a linkage matrix, including singletons."""
    children: list[list[int]] = [[i] for i in range(n_leaves)]
    sets: list[frozenset] = [frozenset([i]) for i in range(n_leaves)]
    for a, b, _, _ in linkage_matrix:
        a_i, b_i = int(a), int(b)
        merged = children[a_i] + children[b_i]
        children.append(merged)
        sets.append(frozenset(merged))
    return sets


def precompute_bootstrap_trees(
    distance_matrix: np.ndarray,
    *,
    n_iter: int = 100,
    leaf_keep_frac: float = 0.80,
    linkage_method: str = "average",
    seed: int = 42,
) -> list[tuple[frozenset, frozenset]]:
    """Precompute the bootstrap dendrograms once so multiple reference
    clade-sets can be evaluated against the same trees.

    Returns a list of ``(keep_set, subtree_orig_sets)`` tuples — one per
    iteration. ``keep_set`` is the set of original-leaf indices retained
    in this iteration; ``subtree_orig_sets`` is a frozenset of frozensets
    (one per subtree in the bootstrap tree, with leaf indices remapped to
    original-index space). Sharing this cache across clade-sets removes
    the need to re-link the distance submatrix per reference config.
    """
    n = distance_matrix.shape[0]
    rng = np.random.default_rng(seed)
    keep_n = int(round(leaf_keep_frac * n))
    if keep_n < 4:
        raise ValueError(
            f"leaf_keep_frac={leaf_keep_frac} on n={n} leaves leaves too few"
        )
    logger.info(
        "Bootstrap: precomputing %d trees × %d / %d leaves kept (linkage=%s)",
        n_iter, keep_n, n, linkage_method,
    )

    trees: list[tuple[frozenset, frozenset]] = []
    for it in range(n_iter):
        keep_idx = np.sort(rng.choice(n, size=keep_n, replace=False))
        sub_d = distance_matrix[np.ix_(keep_idx, keep_idx)]
        condensed = squareform(sub_d, checks=False)
        Z_b = linkage(condensed, method=linkage_method)

        boot_subtree_sets = _all_subtree_leaf_sets(Z_b, len(keep_idx))
        local_to_orig = {i: int(keep_idx[i]) for i in range(len(keep_idx))}
        boot_subtree_orig_sets = frozenset(
            frozenset(local_to_orig[x] for x in s) for s in boot_subtree_sets
        )
        trees.append((frozenset(int(i) for i in keep_idx), boot_subtree_orig_sets))

        if (it + 1) % 10 == 0:
            logger.info("  bootstrap precomputed %d / %d", it + 1, n_iter)
    return trees


def bootstrap_support_from_trees(
    trees: list[tuple[frozenset, frozenset]],
    member_ids: list[str],
    reference_clades: dict[str, list[str]],
) -> dict[str, dict]:
    """Score one reference clade-set against precomputed bootstrap trees."""
    id_to_idx = {mid: i for i, mid in enumerate(member_ids)}
    ref_idx_sets = {
        cid: frozenset(id_to_idx[m] for m in members if m in id_to_idx)
        for cid, members in reference_clades.items()
    }
    n_supported = {cid: 0 for cid in reference_clades}
    n_evaluated = {cid: 0 for cid in reference_clades}

    for keep_set, boot_subtree_orig_sets in trees:
        for cid, ref_set in ref_idx_sets.items():
            survivors = ref_set & keep_set
            if len(survivors) < 2:
                continue
            n_evaluated[cid] += 1
            if survivors in boot_subtree_orig_sets:
                n_supported[cid] += 1

    out: dict[str, dict] = {}
    for cid in reference_clades:
        ne = n_evaluated[cid]
        ns = n_supported[cid]
        out[cid] = {
            "n_iter_evaluated": ne,
            "n_iter_supported": ns,
            "support_frac": (ns / ne) if ne > 0 else float("nan"),
            "n_members": len(reference_clades[cid]),
        }
    return out


def bootstrap_clade_support(
    distance_matrix: np.ndarray,
    member_ids: list[str],
    reference_clades: dict[str, list[str]],
    *,
    n_iter: int = 100,
    leaf_keep_frac: float = 0.80,
    linkage_method: str = "average",
    seed: int = 42,
) -> dict[str, dict]:
    """Convenience wrapper: precompute bootstrap trees + score one config."""
    trees = precompute_bootstrap_trees(
        distance_matrix, n_iter=n_iter, leaf_keep_frac=leaf_keep_frac,
        linkage_method=linkage_method, seed=seed,
    )
    return bootstrap_support_from_trees(trees, member_ids, reference_clades)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compute_clade_labels(
    clusters: dict[str, list[str]],
    member_to_canonical_type: dict[str, str],
    *,
    parent_clusters: dict[str, list[str]] | None = None,
    unassigned_id: str = "dtc_unassigned",
) -> dict[str, str]:
    """Assign semantic labels to DTC clades based on majority canonical type.

    Two-level labelling scheme:

      * **Level 1 — numeric**: ``<type><N>`` (e.g. ``alpha1``, ``delta2``).
        ``N`` is sequential within each canonical-type group, sorted by
        descending clade size. Used when a clade has no siblings under
        a coarser parent grouping.

      * **Level 2 — letter suffix**: ``<type><N><L>`` (e.g. ``alpha1A``,
        ``alpha2B``). Generated only when ``parent_clusters`` is supplied
        and the clade is one of MULTIPLE children of the same parent —
        the parent gets the level-1 label, children get letter suffixes
        in size order (largest = A, second largest = B, …).

    Without ``parent_clusters`` every clade gets a level-1 label only.
    The ``unassigned_id`` clade (if present) is mapped to ``"unassigned"``
    so it stays distinguishable in plots.
    """
    from collections import Counter

    def _dom_type(members: list[str]) -> str:
        c = Counter(member_to_canonical_type.get(m) for m in members)
        c = Counter({k: v for k, v in c.items() if k})
        if not c:
            return "unknown"
        return c.most_common(1)[0][0]

    real = {cid: m for cid, m in clusters.items() if cid != unassigned_id}
    dom = {cid: _dom_type(members) for cid, members in real.items()}
    sizes = {cid: len(members) for cid, members in real.items()}

    labels: dict[str, str] = {}
    if parent_clusters is None:
        # Level-1 only: number sequentially within each canonical type
        # (ordered by descending size).
        by_type: dict[str, list[str]] = {}
        for cid in real:
            by_type.setdefault(dom[cid], []).append(cid)
        for t, cids in by_type.items():
            cids.sort(key=lambda c: (-sizes[c], c))
            for i, cid in enumerate(cids, start=1):
                labels[cid] = f"{t}{i}"
    else:
        # Each child clade is matched to its parent via majority overlap.
        parent_real = {
            pid: m for pid, m in parent_clusters.items() if pid != unassigned_id
        }
        parent_sets = {pid: set(m) for pid, m in parent_real.items()}
        child_parent: dict[str, str] = {}
        for cid, members in real.items():
            child_set = set(members)
            best_pid, best_overlap = None, 0
            for pid, pset in parent_sets.items():
                k = len(child_set & pset)
                if k > best_overlap:
                    best_overlap = k
                    best_pid = pid
            child_parent[cid] = best_pid or "unassigned_parent"

        # Label parents first (level-1 numbering, sorted by size within type).
        parent_dom = {pid: _dom_type(m) for pid, m in parent_real.items()}
        parent_sizes = {pid: len(m) for pid, m in parent_real.items()}
        parent_label: dict[str, str] = {}
        by_type_p: dict[str, list[str]] = {}
        for pid in parent_real:
            by_type_p.setdefault(parent_dom[pid], []).append(pid)
        for t, pids in by_type_p.items():
            pids.sort(key=lambda p: (-parent_sizes[p], p))
            for i, pid in enumerate(pids, start=1):
                parent_label[pid] = f"{t}{i}"

        # Group children by parent; assign letter suffixes when ≥ 2 children.
        children_by_parent: dict[str, list[str]] = {}
        for cid, pid in child_parent.items():
            children_by_parent.setdefault(pid, []).append(cid)
        for pid, cids in children_by_parent.items():
            cids.sort(key=lambda c: (-sizes[c], c))
            base = parent_label.get(pid, f"orphan_{pid}")
            if len(cids) == 1:
                labels[cids[0]] = base
            else:
                for i, cid in enumerate(cids):
                    suffix = chr(ord("A") + i) if i < 26 else f"A{i - 25}"
                    labels[cid] = f"{base}{suffix}"

    if unassigned_id in clusters:
        labels[unassigned_id] = "unassigned"
    return labels


