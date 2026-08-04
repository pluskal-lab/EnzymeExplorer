"""Annotate the MARTS-DB phylogeny with kingdom + domain-configuration strips for iTOL.

Consumes the tree produced by ``scripts/build_martsdb_tree.sh`` (MAFFT
``--auto`` → IQ-TREE ``-fast -m LG+G4``, all 1,374 MARTS-DB sequences
keyed by ``marts_E…`` identifiers). Cross-references MARTS-DB metadata
+ the per-sequence detected-domain composition (from Section 2 output +
Section 3 subtype labeling) to emit the following annotation files:

* ``metadata.csv``                     — one row per leaf: marts_id, uniprot,
                                         kingdom, domain_configuration
* ``domain_configuration_palette.csv`` — configuration → colour
* ``kingdom_palette.csv``              — kingdom → colour
* ``itol_kingdom_colorstrip.txt``      — outer ring (kingdom)
* ``itol_domain_config_colorstrip.txt``      — inner ring (domain configuration)
* ``itol_domain_config_treecolors.txt``      — branch colours by monophyletic
                                                domain-configuration clade

All I/O paths are argparse-driven and default to the standard
``outputs/martsdb/phylogeny/`` layout the driver script uses.
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phylo")


_KINGDOM_RGB: dict[str, tuple[float, float, float]] = {
    "Bacteria": (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    "Fungi":    (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    "Plants":   (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    "Animals":  (0.8352941176470589, 0.3686274509803922, 0.0),
    "Protists": (0.8, 0.47058823529411764, 0.7372549019607844),
    "Viruses":  (0.9254901960784314, 0.8823529411764706, 0.2),
    "Archaea":  (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
}
KINGDOM_COLORS: dict[str, str] = {
    k: mpl.colors.to_hex(v) for k, v in _KINGDOM_RGB.items()
}
KINGDOM_COLORS["Unknown"] = "#BDBDBD"

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ",
         "delta": "δ", "epsilon": "ε", "zeta": "ζ"}


def to_greek(label: str) -> str:
    for k, v in GREEK.items():
        if label.startswith(k):
            return v + label[len(k):]
    return label


def _family_rank(subtype: str) -> int:
    """Canonical presentation order: γ/ε first, then β/δ, then α, ζ last.

    Collapses configurations that differ only in N→C domain order
    (e.g. ``α1D + β + γ`` and ``γ + β + α1D``) into a single string,
    so the strip legend does not carry duplicate categories.
    """
    if subtype.startswith(("gamma", "epsilon")):
        return 0
    if subtype.startswith(("beta", "delta")):
        return 1
    if subtype.startswith("alpha"):
        return 2
    return 3


def _domain_config(mapped_regions, subtype_bridge: dict[str, str]) -> str:
    """Return the canonically-ordered configuration string.

    Detections are first ordered N→C by the smallest query-residue
    position covered; that N→C order is then re-bucketed by family
    rank (γ/ε → β/δ → α → ζ) so different residue orderings of the
    same domain set collapse to one string.
    """
    ordered = []
    for reg in mapped_regions:
        pos = [v for v in reg.residues_mapping.values() if v > 0]
        if not pos:
            continue
        subtype = subtype_bridge.get(reg.module_id, reg.domain)
        ordered.append((min(pos), subtype))
    ordered.sort()
    ranked = sorted(
        enumerate(ordered),
        key=lambda kv: (_family_rank(kv[1][1]), kv[0]),
    )
    parts = [to_greek(subtype) for _, (_, subtype) in ranked]
    return " + ".join(parts) if parts else ""


def _domain_config_palette(configs: list[str]) -> dict[str, str]:
    """Assign a distinct hex colour to each unique configuration.

    Cycles through tab20 → tab20b → tab20c → Set3 (up to 68 slots),
    which is enough to give even the long-tail configurations their
    own visually distinct colour.
    """
    palettes = ["tab20", "tab20b", "tab20c", "Set3"]
    slots: list[str] = []
    for name in palettes:
        cmap = mpl.colormaps[name]
        for i in range(cmap.N):
            slots.append(mpl.colors.to_hex(cmap(i)))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    slots = [h for h in slots if not (h in seen or seen.add(h))]
    return {cfg: slots[i % len(slots)] for i, cfg in enumerate(configs)}


def _maximal_monophyletic_subtrees(tree, group_of: dict[str, str | None]):
    """Yield ``(label, clade)`` for every MAXIMAL clade whose leaves all
    share the same non-``None`` group label. Leaves mapped to ``None``
    block monophyly."""
    parent_of: dict[int, object] = {}
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            parent_of[id(child)] = clade

    mono_label: dict[int, str | None] = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            mono_label[id(clade)] = group_of.get(clade.name)
        else:
            labels = {mono_label[id(c)] for c in clade.clades}
            mono_label[id(clade)] = (
                next(iter(labels))
                if len(labels) == 1 and None not in labels
                else None
            )

    seen: set[int] = set()
    for clade in tree.find_clades(order="preorder"):
        if id(clade) in seen:
            continue
        lbl = mono_label[id(clade)]
        if lbl is None:
            continue
        parent = parent_of.get(id(clade))
        if parent is None or mono_label[id(parent)] != lbl:
            for desc in clade.find_clades():
                seen.add(id(desc))
            yield lbl, clade


def _write_treecolors(
    tree, group_of: dict[str, str | None],
    colour_map: dict[str, str], out_path: Path,
) -> None:
    """Colour every maximal monophyletic subtree by its group label."""
    lines = ["TREE_COLORS", "SEPARATOR TAB", "DATA"]
    n_clades = 0
    n_leaves = 0
    for label, clade in _maximal_monophyletic_subtrees(tree, group_of):
        colour = colour_map.get(label)
        if colour is None:
            continue
        leaves = [l.name for l in clade.get_terminals()]
        if len(leaves) == 1:
            lines.append(f"{leaves[0]}\tbranch\t{colour}\tnormal\t2")
        else:
            # Anchor the MRCA on one leaf from each child so iTOL's pipe
            # syntax always resolves to this exact clade root — see the
            # dark-candidates annotation script for the reasoning.
            child_leaves = [c.get_terminals()[0].name for c in clade.clades]
            node_id = f"{child_leaves[0]}|{child_leaves[-1]}"
            lines.append(f"{node_id}\tclade\t{colour}\tnormal\t2")
        n_clades += 1
        n_leaves += len(leaves)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (%d monophyletic clades covering %d leaves)",
        out_path, n_clades, n_leaves,
    )


def _itol_colorstrip(
    path: Path, dataset_label: str,
    colour_map: dict[str, str],
    leaf_to_key: dict[str, str],
    *,
    colour: str = "#000000",
    strip_width: int = 35,
    legend_shape: int = 1,          # 1=square, 2=circle, 3=star, 4=right-tri, 5=left-tri, 6=check
    legend_shape_scale: float = 1.0,
    legend_gradient: bool = False,
) -> None:
    """Write a minimal iTOL v7 ``DATASET_COLORSTRIP`` file.

    ``leaf_to_key`` maps every tree leaf to the categorical key used
    by ``colour_map``. Missing leaves are omitted (iTOL leaves them
    blank in the ring, which is what we want here).
    """
    # iTOL parses LEGEND_* according to the file's SEPARATOR; here that
    # is TAB, so every shape/colour/label must be one tab-delimited field.
    n = len(colour_map)
    legend_shapes = "\t".join([str(legend_shape)] * n)
    legend_colors = "\t".join(colour_map.values())
    legend_labels = "\t".join(colour_map.keys())
    legend_scales = "\t".join([f"{legend_shape_scale:g}"] * n)
    with path.open("w") as f:
        f.write("DATASET_COLORSTRIP\n")
        f.write("SEPARATOR TAB\n")
        f.write(f"DATASET_LABEL\t{dataset_label}\n")
        f.write(f"COLOR\t{colour}\n")
        f.write(f"STRIP_WIDTH\t{strip_width}\n")
        f.write("MARGIN\t2\n")
        f.write("BORDER_WIDTH\t0.25\n")
        f.write("SHOW_INTERNAL\t0\n")
        f.write(f"LEGEND_TITLE\t{dataset_label}\n")
        f.write(f"LEGEND_SHAPES\t{legend_shapes}\n")
        f.write(f"LEGEND_COLORS\t{legend_colors}\n")
        f.write(f"LEGEND_LABELS\t{legend_labels}\n")
        f.write(f"LEGEND_SHAPE_SCALES\t{legend_scales}\n")
        f.write(f"LEGEND_GRADIENT\t{1 if legend_gradient else 0}\n")
        f.write("DATA\n")
        for leaf, key in leaf_to_key.items():
            if key not in colour_map:
                continue
            f.write(f"{leaf}\t{colour_map[key]}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tree", default="outputs/martsdb/phylogeny/tree.nwk",
        help="Newick tree produced by scripts/build_martsdb_tree.sh.",
    )
    p.add_argument(
        "--martsdb-csv",
        default="data/martsDB_reactions_2026_02_22_preprocessed.csv",
    )
    p.add_argument(
        "--detected-domains-pkl",
        default="data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl",
    )
    p.add_argument(
        "--subtype-pkl",
        default="data/domain_module_id_2_domain_subtype.pkl",
    )
    p.add_argument(
        "--output-dir", default="outputs/martsdb/phylogeny",
        help="Directory to write metadata + iTOL annotation files into.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inputs ---------------------------------------------------------
    logger.info("Loading source tree from %s", args.tree)
    tree = Phylo.read(args.tree, "newick")
    leaf_names = {c.name for c in tree.get_terminals()}
    logger.info("Source tree: %d leaves", len(leaf_names))

    df = pd.read_csv(
        args.martsdb_csv,
        low_memory=False,
        usecols=["Enzyme_marts_ID", "Uniprot_ID", "Kingdom"],
    )
    meta = df.drop_duplicates(subset=["Enzyme_marts_ID"])
    meta = meta.rename(columns={
        "Enzyme_marts_ID": "marts_id",
        "Uniprot_ID":      "uniprot",
        "Kingdom":         "kingdom",
    })
    meta["kingdom"] = meta["kingdom"].fillna("Unknown").str.capitalize()
    logger.info("MARTS-DB entries: %d (with UniProt: %d)",
                len(meta), meta["uniprot"].notna().sum())

    detections = pickle.load(open(args.detected_domains_pkl, "rb"))
    subtype_bridge = pickle.load(open(args.subtype_pkl, "rb"))

    # ---- Domain configuration per entry --------------------------------
    logger.info("Computing domain configurations …")
    meta["domain_configuration"] = meta["marts_id"].map(
        lambda m: _domain_config(detections.get(m, []), subtype_bridge)
    )

    # ---- Verify overlap ------------------------------------------------
    meta = meta[meta["marts_id"].isin(leaf_names)].reset_index(drop=True)
    logger.info("MARTS-DB entries kept (in tree): %d", len(meta))
    dropped = 1374 - len(meta)
    if dropped:
        logger.warning("Dropped %d MARTS-DB entries not found in the tree.", dropped)

    # ---- Metadata + palettes ------------------------------------------
    meta = meta[["marts_id", "uniprot", "kingdom", "domain_configuration"]]
    meta.to_csv(out_dir / "metadata.csv", index=False)

    # Domain-config colour assignment: order by frequency so large
    # groups get the earliest (most distinct) palette slots.
    freq = meta["domain_configuration"].value_counts()
    freq = freq[freq.index != ""]
    config_palette = _domain_config_palette(list(freq.index))
    pd.DataFrame(
        {"domain_configuration": list(config_palette),
         "n_leaves": [int(freq[k]) for k in config_palette],
         "color":   list(config_palette.values())}
    ).to_csv(out_dir / "domain_configuration_palette.csv", index=False)

    pd.DataFrame(
        {"kingdom": list(KINGDOM_COLORS),
         "color":   list(KINGDOM_COLORS.values())}
    ).to_csv(out_dir / "kingdom_palette.csv", index=False)

    # ---- iTOL colorstrips ----------------------------------------------
    leaf_to_kingdom = dict(zip(meta["marts_id"], meta["kingdom"]))
    leaf_to_config  = dict(zip(meta["marts_id"], meta["domain_configuration"]))

    _itol_colorstrip(
        out_dir / "itol_kingdom_colorstrip.txt",
        "Kingdom (outer ring)",
        KINGDOM_COLORS, leaf_to_kingdom,
    )
    _itol_colorstrip(
        out_dir / "itol_domain_config_colorstrip.txt",
        "Domain configuration (inner ring)",
        config_palette, leaf_to_config,
        strip_width=70,
        legend_shape=2,          # circle
        legend_shape_scale=0.9,
        legend_gradient=True,
    )

    # Branch colouring: recolour every maximal monophyletic
    # domain-configuration subtree — reproduces the coloured clades of
    # the earlier figure.
    config_of_or_none: dict[str, str | None] = {
        rid: (c if c else None) for rid, c in leaf_to_config.items()
    }
    _write_treecolors(
        tree, config_of_or_none, config_palette,
        out_dir / "itol_domain_config_treecolors.txt",
    )

    logger.info("Done. Outputs under %s", out_dir)
    logger.info("Unique domain configurations: %d", len(config_palette))
    logger.info("Top 10 configurations:")
    for cfg, n in freq.head(10).items():
        logger.info("  %5d  %s", n, cfg)


if __name__ == "__main__":
    main()
