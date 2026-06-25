"""Generate iTOL v7 annotation files for the dark-candidates phylogeny.

Three drop-in iTOL datasets, layered together:

1. ``itol_kingdom_tree.txt`` — every kingdom-monophyletic subtree is
   coloured by its kingdom palette (``TREE_COLORS`` clade + branch
   rows). The tree backbone — branches with mixed-kingdom descendants
   — stays default-grey.

2. ``itol_source_band.txt`` — the single per-leaf strip band used to
   tell apart martsDB / dark candidate / hard-candidate clade leaves
   (``DATASET_COLORSTRIP``).

3. ``itol_hard_candidates_marker.txt`` — ``DATASET_SYMBOL`` ring of
   purple stars marking the hard-candidate leaves. Symbol size is
   scaled by the leaf's patristic distance to the nearest martsDB
   leaf, so bigger stars = farther from anything we already know.
   Replaces the earlier ``DATASET_TEXT`` ring, which crowded the
   outer ring badly.

Inputs (under ``data/dark_candidates/``):
    metadata.csv          — id, source, kingdom, kingdom_detailed, seq_len
    distant_clades.csv    — candidate-only subtrees (used to widen the
                            "hard candidate" colour to the full clade
                            in the source band)
    subtree_members.csv   — long-form (representative, member, kingdom)

Outputs land beside the inputs.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

logger = logging.getLogger(__name__)

KINGDOM_COLORS: dict[str, str] = {
    # Bacteria was matplotlib-blue, which now collides with the
    # candidate blues below; bumped to a teal-green that stays far
    # from Plants' #2ca02c and from any blue.
    "Bacteria": "#16A085",
    "Archaea":  "#ff7f0e",
    "Animals":  "#d62728",
    "Plants":   "#2ca02c",
    "Fungi":    "#9467bd",
    "Protists": "#8c564b",
    "Viruses":  "#e377c2",
    "Unknown":  "#bdbdbd",
}
MARTSDB_YELLOW = "#FFD400"
DARK_CAND_LIGHTBLUE = "#A6CEE3"   # dark candidates (the broad pool)
HARD_CAND_DARKBLUE = "#08306B"    # hard candidates (the curated picks)


def _maximal_monophyletic_subtrees(tree, group_of: dict[str, str | None]):
    """Yield (group_label, clade) for every MAXIMAL clade whose leaves
    all share the same non-None group label.

    Used twice: once to recolour kingdom-monophyletic clades; the same
    walk is reused for the source-based annotations elsewhere by
    passing a different ``group_of`` mapping.
    """
    parent_of = {}
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            parent_of[id(child)] = clade

    # For every node, ``mono_label`` = the single shared group label of
    # every descendant terminal, or ``None`` if the subtree mixes groups
    # (or contains a leaf with group ``None``).
    mono_label: dict[int, str | None] = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            mono_label[id(clade)] = group_of.get(clade.name)
        else:
            labels = {mono_label[id(c)] for c in clade.clades}
            mono_label[id(clade)] = next(iter(labels)) if len(labels) == 1 and None not in labels else None

    seen: set[int] = set()
    for clade in tree.find_clades(order="preorder"):
        if id(clade) in seen:
            continue
        lbl = mono_label[id(clade)]
        if lbl is None:
            continue
        parent = parent_of.get(id(clade))
        # Maximal = parent doesn't share the same monophyletic label.
        if parent is None or mono_label[id(parent)] != lbl:
            for desc in clade.find_clades():
                seen.add(id(desc))
            yield lbl, clade


def _write_treecolors_kingdom(
    tree, meta: pd.DataFrame, out_path: Path,
) -> None:
    """Colour every kingdom-monophyletic subtree end-to-end.

    Rule: if every leaf in a subtree shares the same kingdom, the
    whole subtree (interior branches + leading edge into the MRCA)
    gets the kingdom's palette colour. Mixed-kingdom backbone
    branches stay default. ``Unknown`` (unresolved dark proteins) is
    treated as "no kingdom" — those leaves block monophyly and stay
    default-grey, matching the user's request.

    For each maximal monophyletic clade we emit:
      * multi-leaf -> ``clade`` (paints interior) + ``branch`` (paints
        the edge from the parent into the MRCA, so no thin gap
        between adjacent kingdom subtrees) on the MRCA;
      * singleton -> ``branch`` on the leaf's terminal edge.
    """
    # Unknown -> None so monophyly is broken by an unresolved leaf and
    # those branches stay default-grey (per the user's request).
    kingdom_of_or_none = {
        rid: (k if k != "Unknown" else None)
        for rid, k in zip(meta["id"], meta["kingdom"])
    }

    # Use the maximal-monophyletic walk: one ``clade`` row per maximal
    # subtree. iTOL's ``clade`` rule reliably paints every interior
    # branch inside the clade, which is what the user wants. Singletons
    # use ``branch`` on the leaf name (the one place ``branch`` is
    # guaranteed to work). Pipe-MRCA + ``branch`` is unreliable for
    # internal nodes in iTOL, so we don't use it here.
    lines = [
        "TREE_COLORS",
        "SEPARATOR TAB",
        "DATA",
    ]
    n_clades = 0
    per_kingdom: dict[str, int] = {}
    n_leaves_covered = 0
    for kingdom, clade in _maximal_monophyletic_subtrees(tree, kingdom_of_or_none):
        color = KINGDOM_COLORS[kingdom]
        leaves = [l.name for l in clade.get_terminals()]
        if len(leaves) == 1:
            lines.append(f"{leaves[0]}\tbranch\t{color}\tnormal\t2")
        else:
            # iTOL resolves a pipe MRCA as the LOWEST internal node
            # whose descendants include BOTH named leaves. Picking the
            # alphabetical min/max can land both leaves in the same
            # child subclade — then the MRCA is that subclade's root,
            # not the maximal monophyletic clade's root, and ``clade``
            # paints only one half. Anchoring on one leaf from EACH
            # child of the clade root guarantees the MRCA == this
            # clade's root. Children with only internal-node terminals
            # are impossible (the tree is fully binary at the leaf
            # level), so ``children[i].get_terminals()`` is always
            # non-empty.
            child_leaves = [c.get_terminals()[0].name for c in clade.clades]
            node_id = f"{child_leaves[0]}|{child_leaves[-1]}"
            lines.append(f"{node_id}\tclade\t{color}\tnormal\t2")
        n_clades += 1
        per_kingdom[kingdom] = per_kingdom.get(kingdom, 0) + 1
        n_leaves_covered += len(leaves)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (%d maximal monophyletic clades covering %d/%d leaves; "
        "per-kingdom: %s)",
        out_path, n_clades, n_leaves_covered, len(meta),
        ", ".join(f"{k}={v}" for k, v in sorted(per_kingdom.items())),
    )


def _write_kingdom_legend(meta: pd.DataFrame, out_path: Path) -> None:
    """A 1-pixel-wide kingdom colorstrip used purely as a legend carrier.

    iTOL renders the strip on the tree (effectively invisible at
    width=1) but the legend it declares shows up in the iTOL Datasets
    panel, giving the user a "kingdom → colour" cheat-sheet without
    adding a second visible band.
    """
    present = [k for k in KINGDOM_COLORS if (meta["kingdom"] == k).any()]
    legend_colors = [KINGDOM_COLORS[k] for k in present]
    legend_shapes = ["1"] * len(present)
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "DATASET_LABEL\tKingdom (legend)",
        "COLOR\t#808080",
        "COLOR_BRANCHES\t0",
        "STRIP_WIDTH\t1",  # ~invisible strip, the legend is what matters
        "MARGIN\t0",
        "BORDER_WIDTH\t0",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tKingdom",
        "LEGEND_SHAPES\t" + "\t".join(legend_shapes),
        "LEGEND_COLORS\t" + "\t".join(legend_colors),
        "LEGEND_LABELS\t" + "\t".join(present),
        "DATA",
    ]
    lines = list(header)
    for r in meta.itertuples(index=False):
        kingdom = r.kingdom if r.kingdom in KINGDOM_COLORS else "Unknown"
        lines.append(f"{r.id}\t{KINGDOM_COLORS[kingdom]}\t{kingdom}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (legend-only carrier strip)", out_path)


def _write_source_colorstrip(
    meta: pd.DataFrame,
    hard_clade_members: set[str],
    out_path: Path,
) -> None:
    """Per-leaf strip band: yellow=martsDB, red=dark candidate, purple=hard clade."""
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "DATASET_LABEL\tSource",
        "COLOR\t#808080",
        "COLOR_BRANCHES\t0",
        "STRIP_WIDTH\t25",
        "MARGIN\t5",
        "BORDER_WIDTH\t0",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tSource",
        "LEGEND_SHAPES\t1\t1\t1",
        f"LEGEND_COLORS\t{MARTSDB_YELLOW}\t{DARK_CAND_LIGHTBLUE}\t{HARD_CAND_DARKBLUE}",
        "LEGEND_LABELS\tmartsDB\tdark candidate\thard candidate clade",
        "DATA",
    ]
    lines = list(header)
    for r in meta.itertuples(index=False):
        if r.id in hard_clade_members:
            color, label = HARD_CAND_DARKBLUE, "hard candidate clade"
        elif r.source == "martsDB":
            color, label = MARTSDB_YELLOW, "martsDB"
        else:
            color, label = DARK_CAND_LIGHTBLUE, "dark candidate"
        lines.append(f"{r.id}\t{color}\t{label}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d leaves)", out_path, len(meta))


def _write_hard_candidate_symbols(
    tree, hard_ids: list[str], martsdb_ids: set[str], out_path: Path,
) -> None:
    """Outer-ring ``DATASET_SYMBOL`` star per hard candidate, sized by
    nearest-martsDB distance.

    Symbols are far less crowded than text labels at the outer ring,
    and the size encoding doubles as a "how distant is this hard
    candidate" cue: bigger star = farther from any martsDB leaf.

    Header tuned for "outermost ring":
      * ``MAXIMUM_SIZE 60`` caps the largest star at 60 px.
      * Symbol ``SIZE`` is the leaf's distance to nearest martsDB
        (raw float — iTOL rescales relative to ``MAXIMUM_SIZE``
        across the dataset).
      * ``GRADIENT_FILL 0`` and ``FILL 1`` keep the stars solid.
    """
    martsdb_leaves = [l for l in tree.get_terminals() if l.name in martsdb_ids]
    leaf_by_name = {l.name: l for l in tree.get_terminals()}

    rows: list[tuple[str, float]] = []
    for hid in hard_ids:
        leaf = leaf_by_name.get(hid)
        if leaf is None:
            continue
        best_d = min(tree.distance(leaf, m) for m in martsdb_leaves)
        rows.append((hid, best_d))

    header = [
        "DATASET_SYMBOL",
        "SEPARATOR TAB",
        "DATASET_LABEL\tHard candidates",
        f"COLOR\t{HARD_CAND_DARKBLUE}",
        "MAXIMUM_SIZE\t60",
        "GRADIENT_FILL\t0",
        "BORDER_WIDTH\t1",
        "BORDER_COLOR\t#000000",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tHard candidate (size = distance to nearest martsDB)",
        "LEGEND_SHAPES\t3",
        f"LEGEND_COLORS\t{HARD_CAND_DARKBLUE}",
        "LEGEND_LABELS\thard candidate",
        "DATA",
    ]
    lines = list(header)
    for hid, d in rows:
        # ID  SYMBOL  SIZE  COLOR  FILL  POSITION
        # SYMBOL=3 → star. POSITION=1 → at the leaf's outer edge
        # (anchors the marker to the leaf; the WHOLE dataset still
        # gets drawn outside any band that sits above it in the iTOL
        # Datasets panel — so drag this dataset to the BOTTOM of the
        # panel to keep the stars on the outermost ring).
        lines.append(f"{hid}\t3\t{d:.4f}\t{HARD_CAND_DARKBLUE}\t1\t1")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (%d hard-candidate stars, distance range %.2f–%.2f)",
        out_path, len(rows),
        min((d for _, d in rows), default=0),
        max((d for _, d in rows), default=0),
    )


def _read_fasta_ids(path: Path) -> list[str]:
    """Return the list of FASTA header IDs (first whitespace-delimited
    token after the leading ``>``)."""
    ids: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


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
        "--members", type=Path,
        default=Path("data/dark_candidates/subtree_members.csv"),
    )
    parser.add_argument(
        "--hard-candidates", type=Path,
        default=Path("data/hard_candidates/candidates.fasta"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/dark_candidates"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    meta = pd.read_csv(args.metadata)
    meta["kingdom"] = meta["kingdom"].fillna("Unknown")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree = Phylo.read(str(args.tree), "newick")

    martsdb_ids = set(meta.loc[meta["source"] == "martsDB", "id"])

    # Hard-candidate IDs that actually appear in the tree.
    hard_ids_in_tree: list[str] = []
    if args.hard_candidates and args.hard_candidates.exists():
        hard_all = _read_fasta_ids(args.hard_candidates)
        hard_ids_in_tree = [h for h in hard_all if (meta["id"] == h).any()]
        absent = sorted(set(hard_all) - set(hard_ids_in_tree))
        if absent:
            logger.info(
                "Hard candidates not in the tree (skipped): %s",
                ", ".join(absent),
            )

    # Members of every candidate-only subtree that contains at least
    # one hard candidate — these leaves get the "hard candidate clade"
    # purple in the source band.
    hard_clade_members: set[str] = set()
    if args.members.exists() and hard_ids_in_tree:
        members = pd.read_csv(args.members)
        rep_by_member = dict(zip(members["member"], members["representative"]))
        hard_reps = {rep_by_member[h] for h in hard_ids_in_tree if h in rep_by_member}
        hard_clade_members = set(
            members.loc[members["representative"].isin(hard_reps), "member"]
        )

    _write_treecolors_kingdom(
        tree, meta, args.output_dir / "itol_kingdom_tree.txt",
    )
    _write_kingdom_legend(
        meta, args.output_dir / "itol_kingdom_legend.txt",
    )
    _write_source_colorstrip(
        meta, hard_clade_members, args.output_dir / "itol_source_band.txt",
    )
    if hard_ids_in_tree:
        _write_hard_candidate_symbols(
            tree, hard_ids_in_tree, martsdb_ids,
            args.output_dir / "itol_hard_candidates_marker.txt",
        )


if __name__ == "__main__":
    main()
