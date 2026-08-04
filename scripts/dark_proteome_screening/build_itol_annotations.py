"""Generate iTOL annotation files for the hard-candidates selection tree.

All 2315 leaves now have a Kingdom assignment — MARTS-DB kingdoms come
from ``martsDB_reactions_2026_02_22.csv`` (normalised to palette keys),
dark-putative kingdoms come from a prior colorstrip stashed at
``data/dark_proteome_screening/candidate_selection/phylo_tree/dark_putative_kingdoms_source.txt``.
Clade coloring treats every leaf uniformly (no MARTS-vs-dark distinction).

Emits under ``data/dark_proteome_screening/candidate_selection/phylo_tree/``:

* ``itol_kingdom_treecolors.txt``     — TREE_COLORS. Maximal
  Kingdom-monophyletic subclades painted in that Kingdom's color; every
  subtree is monophyletic in exactly one Kingdom or MIXED (no
  "monophyletic-ignoring-darks" heuristic anymore).
* ``itol_kingdom_colorstrip.txt``     — outer ring: Kingdom color per leaf.
* ``itol_source_colorstrip.txt``      — inner-ring: MARTS-DB (yellow) vs
  dark putative (grey).
* ``itol_selected_triangles.txt``     — DATASET_SYMBOL, green triangles
  pointing at the 12 selected hard-candidate leaves. Triangle size is
  scaled by phylogenetic distance to the closest MARTS-DB sequence.
* ``itol_bar_distance.txt``           — SIMPLE_BAR: distance to closest
  MARTS-DB (0–8 scale, dark leaves only).
* ``itol_bar_tps_p.txt``   — SIMPLE_BAR: TPS_p
  (0.95–1.00 window shifted so a bar of 0 = 0.95, dark only).
* ``itol_bar_fpp_p.txt``   — SIMPLE_BAR: FPP_p
  (0..observed-max, dark only).
* ``itol_bar_ggpp_p.txt``  — SIMPLE_BAR: GGPP_p
  (0..observed-max, dark only).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

REPO = Path(__file__).resolve().parents[2]
SEL_DIR = REPO / "data" / "dark_proteome_screening" / "candidate_selection" / "phylo_tree"
TREE_NWK = SEL_DIR / "tree.nwk"
META_CSV = SEL_DIR / "metadata.csv"
DISTANCES_CSV = SEL_DIR / "dark_distances.csv"
DARK_PUTATIVES_CSV = REPO / "data" / "dark_proteome_screening" / "dark_putatives.csv"
CANDIDATES_FASTA = REPO / "data" / "dark_candidates" / "candidates.fasta"
KINGDOM_PALETTE = REPO / "data" / "kingdom_palette.csv"

OUT_TREECOLORS = SEL_DIR / "itol_kingdom_treecolors.txt"
OUT_KINGDOM_STRIP = SEL_DIR / "itol_kingdom_colorstrip.txt"
OUT_SOURCE_STRIP = SEL_DIR / "itol_source_colorstrip.txt"
OUT_SELECTED_TRI = SEL_DIR / "itol_selected_triangles.txt"

# --- Config ----------------------------------------------------------------
SOURCE_MARTSDB_COLOR = "#f6c700"  # yellow
SOURCE_DARK_COLOR = "#8f8f8f"    # grey
SELECTED_TRIANGLE_COLOR = "#009e73"  # NCB green
TRIANGLE_MIN_SIZE = 15
TRIANGLE_MAX_SIZE = 60
DISTANCE_SCALE_MAX = 8.0  # user-specified for the bar

BAR_METRICS = [
    # (out_file_stem, source_column, label, color, scale_min, scale_max_or_None)
    ("distance",         "distance",           "Phylo distance to closest MARTS-DB", "#4d4d4d", 0.0, DISTANCE_SCALE_MAX),
    ("tps_p", "TPS_p",   "TPS_p (0.95–1.00)",       "#009e73", 0.95, 1.0),
    ("fpp_p", "FPP_p",   "FPP_p",                     "#0173b2", 0.0, None),
    ("ggpp_p","GGPP_p",  "GGPP_p",                    "#de8f05", 0.0, None),
]


# --- Helpers ---------------------------------------------------------------
def _parse_fasta_ids(path: Path) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].split()[0].strip())
    return ids


def _build_parent_map(tree) -> dict:
    parent = {}
    for clade in tree.find_clades(order="level"):
        for c in clade.clades:
            parent[c] = clade
    return parent


def _kingdom_of_subtree(node, leaf_to_kingdom: dict) -> str | None:
    """Single Kingdom if all leaves share it, else 'MIXED', or None if
    no kingdom-annotated leaves under this node."""
    kings = set()
    for t in node.get_terminals():
        k = leaf_to_kingdom.get(t.name)
        if k:
            kings.add(k)
    if not kings:
        return None
    if len(kings) > 1:
        return "MIXED"
    return next(iter(kings))


def _find_maximal_monophyletic_clades(tree, leaf_to_kingdom: dict):
    parent = _build_parent_map(tree)
    node_kingdom: dict = {}
    for node in tree.find_clades(order="postorder"):
        node_kingdom[node] = _kingdom_of_subtree(node, leaf_to_kingdom)
    for node, king in node_kingdom.items():
        if king in (None, "MIXED"):
            continue
        p = parent.get(node)
        if p is not None and node_kingdom.get(p) == king:
            continue
        yield node, king


def _leaf_pair_for_clade(node) -> tuple[str, str] | None:
    terms = list(node.get_terminals())
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0].name, terms[0].name
    return terms[0].name, terms[-1].name


# --- Writers ---------------------------------------------------------------
def _write_tree_colors(tree, leaf_to_kingdom: dict, palette: dict) -> None:
    lines = ["TREE_COLORS", "SEPARATOR TAB", "DATA"]
    for node, king in _find_maximal_monophyletic_clades(tree, leaf_to_kingdom):
        color = palette.get(king)
        if not color:
            continue
        pair = _leaf_pair_for_clade(node)
        if pair is None:
            continue
        a, b = pair
        if a == b:
            lines.append(f"{a}\tbranch\t{color}\tnormal\t2")
        else:
            lines.append(f"{a}|{b}\tclade\t{color}\tnormal\t2")
    OUT_TREECOLORS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TREECOLORS}: {len(lines) - 3} clade directives")


def _write_kingdom_colorstrip(leaf_to_kingdom: dict, palette: dict) -> None:
    kingdoms_present = sorted({k for k in leaf_to_kingdom.values() if k})
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "DATASET_LABEL\tKingdom (outer ring)",
        "COLOR\t#000000",
        "STRIP_WIDTH\t35",
        "MARGIN\t2",
        "BORDER_WIDTH\t0.25",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tKingdom",
        f"LEGEND_SHAPES\t{chr(9).join(['1'] * len(kingdoms_present))}",
        f"LEGEND_COLORS\t{chr(9).join(palette[k] for k in kingdoms_present)}",
        f"LEGEND_LABELS\t{chr(9).join(kingdoms_present)}",
        f"LEGEND_SHAPE_SCALES\t{chr(9).join(['1'] * len(kingdoms_present))}",
        "LEGEND_GRADIENT\t0",
        "DATA",
    ]
    lines = list(header)
    for leaf, king in leaf_to_kingdom.items():
        color = palette.get(king)
        if not color:
            continue
        lines.append(f"{leaf}\t{color}")
    OUT_KINGDOM_STRIP.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_KINGDOM_STRIP}")


def _write_source_colorstrip(meta: pd.DataFrame) -> None:
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "DATASET_LABEL\tSource",
        "COLOR\t#000000",
        "STRIP_WIDTH\t20",
        "MARGIN\t2",
        "BORDER_WIDTH\t0.25",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tSource",
        "LEGEND_SHAPES\t1\t1",
        f"LEGEND_COLORS\t{SOURCE_MARTSDB_COLOR}\t{SOURCE_DARK_COLOR}",
        "LEGEND_LABELS\tMARTS-DB\tDark putative",
        "LEGEND_SHAPE_SCALES\t1\t1",
        "LEGEND_GRADIENT\t0",
        "DATA",
    ]
    lines = list(header)
    for r in meta.itertuples(index=False):
        color = SOURCE_MARTSDB_COLOR if r.source == "martsdb" else SOURCE_DARK_COLOR
        lines.append(f"{r.leaf_id}\t{color}")
    OUT_SOURCE_STRIP.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_SOURCE_STRIP}")


def _write_selected_triangles(selected_ids: list[str], distances: pd.DataFrame) -> None:
    """DATASET_SYMBOL: green triangles pointing at the leaf tip.

    In iTOL's DATASET_SYMBOL, ``4`` = right-pointing triangle. Rendered
    on a leaf in a circular tree, that triangle points toward the leaf
    tip (i.e. outward). Size is scaled linearly by tree distance to the
    closest MARTS-DB leaf: min distance → TRIANGLE_MIN_SIZE, max
    observed distance among selected → TRIANGLE_MAX_SIZE.
    """
    d = distances.set_index("dark_id")
    # Normalize triangle sizes against the full dark-putative distance
    # distribution so selected sizes are comparable to what a viewer sees
    # in the surrounding distance bar.
    all_vals = d["distance"].dropna()
    if len(all_vals) == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(all_vals.min()), float(all_vals.max())
    def _size(v: float) -> float:
        if hi == lo:
            return TRIANGLE_MAX_SIZE
        r = (v - lo) / (hi - lo)
        return TRIANGLE_MIN_SIZE + r * (TRIANGLE_MAX_SIZE - TRIANGLE_MIN_SIZE)

    header = [
        "DATASET_SYMBOL",
        "SEPARATOR TAB",
        f"DATASET_LABEL\tSelected candidates (size ∝ distance to MARTS-DB)",
        "COLOR\t" + SELECTED_TRIANGLE_COLOR,
        "MAXIMUM_SIZE\t" + str(TRIANGLE_MAX_SIZE),
        "GRADIENT_FILL\t0",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tSelected candidate",
        "LEGEND_SHAPES\t4",
        f"LEGEND_COLORS\t{SELECTED_TRIANGLE_COLOR}",
        "LEGEND_LABELS\tSelected",
        "LEGEND_SHAPE_SCALES\t1",
        "DATA",
        "# ID  SYMBOL  SIZE  COLOR  FILL  POSITION",
    ]
    lines = list(header)
    for sid in selected_ids:
        v = d.at[sid, "distance"] if sid in d.index else None
        size = _size(float(v)) if v is not None else TRIANGLE_MIN_SIZE
        # position 0 = at leaf tip; symbol 4 = right-pointing triangle.
        lines.append(f"{sid}\t4\t{size:.2f}\t{SELECTED_TRIANGLE_COLOR}\t1\t0")
    OUT_SELECTED_TRI.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT_SELECTED_TRI} ({len(selected_ids)} triangles; "
        f"distance range {lo:.2f}–{hi:.2f} → size {TRIANGLE_MIN_SIZE}–{TRIANGLE_MAX_SIZE})"
    )


def _write_bar_dataset(
    stem: str,
    values: dict[str, float],
    label: str,
    color: str,
    scale_min: float,
    scale_max: float | None,
) -> None:
    out = SEL_DIR / f"itol_bar_{stem}.txt"
    # For SIMPLE_BAR, iTOL scales by the actual data max unless
    # DATASET_SCALE overrides. We emit values shifted so that
    # scale_min → 0 (bars only visible above scale_min).
    header = [
        "DATASET_SIMPLEBAR",
        "SEPARATOR TAB",
        f"DATASET_LABEL\t{label}",
        f"COLOR\t{color}",
        f"WIDTH\t60",
        f"MARGIN\t2",
        "SHOW_INTERNAL\t0",
        f"BAR_ZERO\t{scale_min:g}",
    ]
    if scale_max is not None:
        # DATASET_SCALE syntax: value|label|color|width|dashed|width
        # We add three ticks: min, mid, max.
        mid = (scale_min + scale_max) / 2.0
        header.append(
            f"DATASET_SCALE\t"
            f"{scale_min:g}-{scale_min:g}-#808080-1-0-8\t"
            f"{mid:g}-{mid:g}-#808080-1-1-8\t"
            f"{scale_max:g}-{scale_max:g}-#808080-1-0-8"
        )
    header += [
        "MAXIMUM_WIDTH\t60",
        "HEIGHT_FACTOR\t1",
        "DATA",
    ]
    lines = list(header)
    for leaf, v in values.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        lines.append(f"{leaf}\t{float(v):.6g}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(values)} bars)")


# --- Main ------------------------------------------------------------------
def main() -> None:
    palette = pd.read_csv(KINGDOM_PALETTE).set_index("kingdom")["color"].to_dict()
    meta = pd.read_csv(META_CSV)
    leaf_to_kingdom = {
        r.leaf_id: r.kingdom for r in meta.itertuples(index=False)
        if pd.notna(r.kingdom) and r.kingdom
    }

    tree = Phylo.read(str(TREE_NWK), "newick")
    _write_tree_colors(tree, leaf_to_kingdom, palette)
    _write_kingdom_colorstrip(leaf_to_kingdom, palette)
    _write_source_colorstrip(meta)

    distances = pd.read_csv(DISTANCES_CSV)
    selected = _parse_fasta_ids(CANDIDATES_FASTA)
    print(f"selected candidates from FASTA: {len(selected)}")
    _write_selected_triangles(selected, distances)

    # Per-dark-leaf bar datasets.
    dark_ids = set(meta.loc[meta["source"] == "dark", "leaf_id"])
    dark_df = pd.read_csv(DARK_PUTATIVES_CSV).set_index("id")
    dist_map = distances.set_index("dark_id")["distance"].to_dict()

    for stem, src_col, label, color, smin, smax in BAR_METRICS:
        if src_col == "distance":
            values = {leaf: dist_map.get(leaf) for leaf in dark_ids}
            resolved_max = smax
        else:
            values = {
                leaf: float(dark_df.at[leaf, src_col])
                for leaf in dark_ids
                if leaf in dark_df.index and pd.notna(dark_df.at[leaf, src_col])
            }
            if smax is None:
                resolved_max = max(values.values()) if values else 1.0
                print(f"  {stem}: auto-max = {resolved_max:.4g}")
            else:
                resolved_max = smax
        _write_bar_dataset(stem, values, label, color, smin, resolved_max)


if __name__ == "__main__":
    main()
