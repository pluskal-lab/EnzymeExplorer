"""Minimal poster dendrogram from the final HAC.

Loads the cached linkage matrix and member ids from
``data/domain_clustering/martsDB_hac_sweep/intermediate/`` plus the
``domain_metadata.csv`` next to it, and renders one dendrogram with no
title, no legend, no grid, no stripe annotations. Only the clades that
are monophyletic in one of {alpha, beta, gamma, delta, epsilon} are
colored; everything else is drawn black.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import squareform


DOMAIN_COLORS = {
    "alpha":   "#00441B",  # very dark green
    "epsilon":    "#1B7B3F",  # dark green
    "gamma":   "#41AB5D",  # medium green
    "delta":   "#74C476",  # light green
    "beta": "#BAE4B3",  # pale green
}
OTHER_COLOR = "#013333"


IGNORE_SUBTYPES = {"alpha_cls2"}
DROP_SUBTYPES = {"delta1", "zeta"}


def _bucket(subtype: str | None) -> str | None:
    """Map a fine-grained subtype label (e.g. ``alpha1A``) to its base type.

    Returns ``None`` for ignored subtypes (``zeta``/``alpha_cls2``) and for
    any label that doesn't start with one of the canonical base types.
    """
    if not isinstance(subtype, str) or subtype in IGNORE_SUBTYPES:
        return None
    for base in DOMAIN_COLORS:
        if subtype.startswith(base):
            return base
    return None


def build_link_color_func(
    linkage_matrix: np.ndarray, leaf_dtype: list[str | None],
):
    n_leaves = len(leaf_dtype)
    Z = np.asarray(linkage_matrix)
    cache: dict[int, set[str | None]] = {}

    def descendants_types(node_id: int) -> set[str | None]:
        if node_id in cache:
            return cache[node_id]
        if node_id < n_leaves:
            cache[node_id] = {leaf_dtype[node_id]}
            return cache[node_id]
        row = node_id - n_leaves
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        types = descendants_types(left) | descendants_types(right)
        cache[node_id] = types
        return types

    def link_color_func(node_id: int) -> str:
        types = {t for t in descendants_types(int(node_id)) if t is not None}
        if len(types) == 1:
            t = next(iter(types))
            if t in DOMAIN_COLORS:
                return DOMAIN_COLORS[t]
        return OTHER_COLOR

    return link_color_func


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hac-dir", type=Path,
        default=Path("data/domain_clustering/martsDB_hac_sweep"),
    )
    parser.add_argument(
        "--subtype-pkl", type=Path,
        default=Path("data/domain_module_id_2_domain_subtype.pkl"),
        help="Pickled dict {module_id: subtype} (e.g. 'alpha1A', 'beta', "
             "'gamma'); subtypes are bucketed to {alpha, beta, gamma, delta, "
             "epsilon} by prefix match.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/domain_clustering/martsDB_hac_sweep/poster_dendrogram.png"),
    )
    parser.add_argument(
        "--balance-clades", action="store_true", default=True,
        help="Subsample each canonical-type group down to the smallest "
             "group's size so the colored clades occupy roughly equal "
             "horizontal extent. On by default; pass --no-balance-clades "
             "to disable.",
    )
    parser.add_argument("--no-balance-clades", dest="balance_clades", action="store_false")
    parser.add_argument(
        "--balance-target", type=int, default=None,
        help="Target leaves per canonical type after balancing. "
             "Default: size of the smallest non-empty type.",
    )
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--linewidth", type=float, default=4.0)
    parser.add_argument("--ylabel-fontsize", type=float, default=42.0)
    parser.add_argument("--ytick-fontsize", type=float, default=32.0)
    parser.add_argument("--axis-linewidth", type=float, default=3.5)
    parser.add_argument("--width", type=float, default=9.0)
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with (args.hac_dir / "intermediate/member_ids.pkl").open("rb") as fh:
        all_member_ids: list[str] = pickle.load(fh)
    with args.subtype_pkl.open("rb") as fh:
        subtype_map: dict = pickle.load(fh)

    keep = [
        i for i, mid in enumerate(all_member_ids)
        if subtype_map.get(mid) not in DROP_SUBTYPES
    ]
    n_dropped = len(all_member_ids) - len(keep)

    if args.balance_clades:
        rng = np.random.default_rng(args.balance_seed)
        by_base: dict[str | None, list[int]] = {}
        for i in keep:
            base = _bucket(subtype_map.get(all_member_ids[i]))
            by_base.setdefault(base, []).append(i)
        typed_sizes = [len(v) for k, v in by_base.items() if k is not None]
        if typed_sizes:
            target = args.balance_target or min(typed_sizes)
            balanced: list[int] = []
            for base, idxs in by_base.items():
                if base is None or len(idxs) <= target:
                    balanced.extend(idxs)
                    continue
                balanced.extend(rng.choice(idxs, size=target, replace=False).tolist())
            balanced.sort()
            sizes = {k: min(len(v), target) for k, v in by_base.items() if k is not None}
            print(f"balanced per-clade sizes (target={target}): {sizes}")
            keep = balanced

    if n_dropped or args.balance_clades:
        D = np.load(args.hac_dir / "intermediate/distance_matrix.npy")
        D = D[np.ix_(keep, keep)]
        member_ids = [all_member_ids[i] for i in keep]
        linkage_matrix = linkage(squareform(D, checks=False), method="average")
        print(f"re-ran linkage on {len(member_ids)} leaves "
              f"(dropped {len(all_member_ids) - len(keep)} total)")
    else:
        linkage_matrix = np.load(args.hac_dir / "intermediate/linkage_matrix.npy")
        member_ids = all_member_ids

    leaf_dtype: list[str | None] = [_bucket(subtype_map.get(mid)) for mid in member_ids]

    link_color_func = build_link_color_func(linkage_matrix, leaf_dtype)
    set_link_color_palette(None)  # ensure our colors aren't overridden by defaults

    with plt.rc_context({"lines.linewidth": args.linewidth}):
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        dendrogram(
            linkage_matrix,
            ax=ax,
            no_labels=True,
            link_color_func=link_color_func,
            above_threshold_color=OTHER_COLOR,
        )

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title("")
    ax.set_ylabel("1 − TMScore", fontsize=args.ylabel_fontsize)
    ax.tick_params(
        axis="y", labelsize=args.ytick_fontsize,
        width=args.axis_linewidth, length=args.axis_linewidth * 4,
    )
    ax.tick_params(axis="x", which="both", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(args.axis_linewidth)
    ax.grid(False)
    ax.set_ylim(0, 1.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}  +  {pdf_path}")


if __name__ == "__main__":
    main()
