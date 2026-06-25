"""Extract candidate-only subtrees from the dark-candidates phylogeny.

Reads ``data/dark_candidates/tree.nwk`` plus the metadata sidecar, then
finds every MAXIMAL subtree whose leaves are all candidate sequences
(``source == "candidate"`` in metadata). For each such subtree we record:

* ``representative``   — the candidate leaf with the highest mean tip
                         distance to its sibling leaves inside the
                         subtree (i.e. the centroid-ish leaf). Falls
                         back to alphabetical order if the subtree is a
                         single leaf.
* ``n_members``        — number of candidate leaves in the subtree.
* ``nearest_martsdb``  — the closest martsDB leaf in the WHOLE tree as
                         measured by patristic distance from the
                         subtree's root node. This is the "how far from
                         anything we already know" score.
* ``distance``         — that patristic distance, with branch lengths.
* ``kingdom_<X>``      — one column per martsDB kingdom giving the
                         candidate-leaf count in the subtree (handy for
                         the manual prioritisation: subtrees of e.g.
                         Archaea-only candidates jump out).

Two CSVs land under ``data/dark_candidates/``:

* ``distant_clades.csv``  — one row per subtree, sorted by ``distance``
                            descending.
* ``subtree_members.csv`` — long-form mapping ``representative, member``
                           (one row per candidate leaf), so the operator
                           can pull every member of a picked subtree.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

logger = logging.getLogger(__name__)


def _leaf_distance_matrix_row(tree, leaf, others) -> dict[str, float]:
    """Distance from ``leaf`` to every leaf in ``others`` using Bio.Phylo's
    ``distance`` traversal. O(N) per pair, used sparingly."""
    return {o.name: tree.distance(leaf, o) for o in others}


def _candidate_only_subtrees(tree, is_candidate: dict[str, bool]):
    """Yield maximal clades whose every terminal is a candidate.

    A clade qualifies if all of its terminals are candidates AND its
    parent clade contains at least one non-candidate terminal (so we
    don't emit nested duplicates). The root is treated as if it has a
    virtual non-candidate parent — i.e. if the WHOLE tree is candidates
    (it isn't, in our case) we'd emit the root.
    """
    # Pre-compute, for every internal clade, whether every terminal under
    # it is a candidate. Walk leaf→root (postorder) using a child→parent
    # map so we don't recompute subtree leaf sets.
    parent_of = {}
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            parent_of[id(child)] = clade

    all_cand: dict[int, bool] = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            all_cand[id(clade)] = bool(is_candidate.get(clade.name, False))
        else:
            all_cand[id(clade)] = all(all_cand[id(c)] for c in clade.clades)

    seen: set[int] = set()
    for clade in tree.find_clades(order="preorder"):
        if id(clade) in seen:
            continue
        if not all_cand[id(clade)]:
            continue
        parent = parent_of.get(id(clade))
        if parent is None or not all_cand[id(parent)]:
            # Maximal candidate-only clade — mark every descendant so we
            # don't re-yield a nested one.
            for desc in clade.find_clades():
                seen.add(id(desc))
            yield clade


def _pick_representative(tree, clade) -> str:
    """Pick the candidate leaf with the smallest mean distance to its
    siblings in the clade. Tie-breaks alphabetically. Falls back to the
    single terminal for singleton clades."""
    leaves = list(clade.get_terminals())
    if len(leaves) == 1:
        return leaves[0].name
    # Mean intra-clade distance per leaf. Use the cached distance helper.
    names = [l.name for l in leaves]
    mean_d = {}
    for i, l in enumerate(leaves):
        d_sum = 0.0
        for j, other in enumerate(leaves):
            if i == j:
                continue
            d_sum += tree.distance(l, other)
        mean_d[l.name] = d_sum / (len(leaves) - 1)
    # Smallest mean distance = most central leaf.
    return min(sorted(names), key=lambda n: mean_d[n])


def _nearest_martsdb(tree, clade, martsdb_leaves) -> tuple[str, float]:
    """Return (id, distance) of the martsDB terminal closest to ``clade``.

    Distance is measured from the clade's root node (or its single leaf
    for singletons). Iterates over all martsDB leaves — O(M) per
    subtree; for a few thousand subtrees and a few thousand martsDB
    leaves this stays well under a minute.
    """
    anchor = clade if not clade.is_terminal() else clade
    best_name = None
    best_d = float("inf")
    for m in martsdb_leaves:
        d = tree.distance(anchor, m)
        if d < best_d:
            best_d = d
            best_name = m.name
    return best_name, best_d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree", type=Path,
        default=Path("data/dark_candidates/tree.nwk"),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("data/dark_candidates/metadata.csv"),
    )
    parser.add_argument(
        "--clades-out", type=Path,
        default=Path("data/dark_candidates/distant_clades.csv"),
    )
    parser.add_argument(
        "--members-out", type=Path,
        default=Path("data/dark_candidates/subtree_members.csv"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    meta = pd.read_csv(args.metadata)
    is_candidate = dict(zip(meta["id"], meta["source"] == "candidate"))
    id_to_kingdom = dict(zip(meta["id"], meta["kingdom"].fillna("Unknown")))

    logger.info("Loading tree from %s", args.tree)
    tree = Phylo.read(str(args.tree), "newick")
    n_leaves = sum(1 for _ in tree.get_terminals())
    logger.info("Tree leaves: %d", n_leaves)

    martsdb_leaves = [l for l in tree.get_terminals() if not is_candidate.get(l.name)]
    logger.info("martsDB leaves: %d", len(martsdb_leaves))

    rows = []
    member_rows = []
    for i, clade in enumerate(_candidate_only_subtrees(tree, is_candidate), 1):
        leaves = list(clade.get_terminals())
        members = [l.name for l in leaves]
        rep = _pick_representative(tree, clade)
        near_id, near_d = _nearest_martsdb(tree, clade, martsdb_leaves)
        kingdom_counts = Counter(id_to_kingdom.get(m, "Unknown") for m in members)
        row = {
            "representative": rep,
            "n_members": len(members),
            "nearest_martsdb": near_id,
            "distance": near_d,
        }
        for k, v in kingdom_counts.items():
            row[f"kingdom_{k}"] = v
        rows.append(row)
        for m in members:
            member_rows.append({"representative": rep, "member": m,
                                "kingdom": id_to_kingdom.get(m, "Unknown")})
        if i % 50 == 0:
            logger.info("Processed %d candidate-only subtrees", i)

    clades_df = pd.DataFrame(rows).fillna(0)
    # Sort largest-distance first so the manual review starts with the
    # rows most worth a second look.
    clades_df = clades_df.sort_values("distance", ascending=False).reset_index(drop=True)
    # Make kingdom columns int-typed for clean display.
    for col in clades_df.columns:
        if col.startswith("kingdom_"):
            clades_df[col] = clades_df[col].astype(int)
    args.clades_out.parent.mkdir(parents=True, exist_ok=True)
    clades_df.to_csv(args.clades_out, index=False)
    pd.DataFrame(member_rows).to_csv(args.members_out, index=False)
    logger.info(
        "Wrote %d subtrees to %s and %d member rows to %s",
        len(clades_df), args.clades_out, len(member_rows), args.members_out,
    )


if __name__ == "__main__":
    main()
