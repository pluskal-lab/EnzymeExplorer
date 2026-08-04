"""Compute the phylogenetic distance from each dark-putative leaf to the
nearest MARTS-DB leaf on the combined tree.

Distance = sum of branch lengths along the unique path between the two
leaves (i.e. the tree's induced metric — undirected, unrooted).

Approach: two-pass on the unrooted tree.
  1. Post-order + pre-order pass to compute ``dist_to_root`` for every
     node and a parent map (O(N)).
  2. For each dark leaf: walk up to the root while tracking cumulative
     branch length. For every ancestor a, record ``dist_dark_to_a``.
  3. For each MARTS-DB leaf: walk up while tracking cumulative branch
     length. At each ancestor, if we've seen it as a dark ancestor, the
     path distance is ``dist_dark_to_a + dist_marts_to_a`` — the LCA
     gives the minimum. Keep the running min per dark leaf.

This is O((N_dark + N_marts) * depth) with the Biopython ``Tree.trace``
under the hood. For the ~2.3k-leaf tree here it takes seconds.

Outputs
-------
* ``data/dark_proteome_screening/candidate_selection/phylo_tree/dark_distances.csv``  — sorted
  ascending by distance, one row per dark putative.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

REPO = Path(__file__).resolve().parents[2]
SEL_DIR = REPO / "data" / "dark_proteome_screening" / "candidate_selection" / "phylo_tree"
TREE_NWK = SEL_DIR / "tree.nwk"
META_CSV = SEL_DIR / "metadata.csv"
OUT_CSV = SEL_DIR / "dark_distances.csv"


def _build_parent_map(tree) -> dict:
    parent = {}
    for clade in tree.find_clades(order="level"):
        for c in clade.clades:
            parent[c] = clade
    return parent


def _ancestor_walk(leaf, parent) -> list[tuple[object, float]]:
    """Return [(node, cumulative_branch_len_from_leaf), …] from the leaf
    up to (and including) the root."""
    out = []
    cum = 0.0
    node = leaf
    while node is not None:
        out.append((node, cum))
        p = parent.get(node)
        if p is None:
            break
        # branch length is on the CHILD in Biopython.
        bl = node.branch_length if node.branch_length is not None else 0.0
        cum += bl
        node = p
    return out


def main() -> None:
    print(f"loading {TREE_NWK}")
    tree = Phylo.read(str(TREE_NWK), "newick")
    parent = _build_parent_map(tree)
    meta = pd.read_csv(META_CSV).set_index("leaf_id")

    leaves = list(tree.get_terminals())
    marts_leaves = [t for t in leaves if t.name and t.name.startswith("marts_E")]
    dark_leaves = [t for t in leaves if t.name and not t.name.startswith("marts_E")]
    print(f"leaves: dark={len(dark_leaves)}, marts={len(marts_leaves)}")

    # Pre-compute each MARTS leaf's ancestor->distance map for fast lookup.
    marts_anc_dist: list[dict] = []
    for ml in marts_leaves:
        anc = {node: d for node, d in _ancestor_walk(ml, parent)}
        marts_anc_dist.append(anc)

    rows = []
    for i, dl in enumerate(dark_leaves):
        if i and i % 100 == 0:
            print(f"  processing dark leaf {i}/{len(dark_leaves)}")
        dark_anc = _ancestor_walk(dl, parent)
        # dark_anc is [(node, cum_dark)], convert to dict for O(1) lookup.
        dark_map = {node: d for node, d in dark_anc}

        best_dist = float("inf")
        best_marts = None
        for ml, anc in zip(marts_leaves, marts_anc_dist):
            # find LCA-based min: iterate MARTS ancestors, pick the one
            # also present in dark_map with the minimum sum.
            local_min = float("inf")
            for node, d_m in anc.items():
                d_d = dark_map.get(node)
                if d_d is None:
                    continue
                s = d_d + d_m
                if s < local_min:
                    local_min = s
            if local_min < best_dist:
                best_dist = local_min
                best_marts = ml.name

        rows.append({
            "dark_id": dl.name,
            "closest_marts_id": best_marts,
            "closest_marts_kingdom": meta.at[best_marts, "kingdom"] if best_marts in meta.index else "",
            "distance": best_dist,
        })

    out = pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(out)} rows)")
    print("furthest 5:")
    print(out.head(5).to_string(index=False))
    print("closest 5:")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
