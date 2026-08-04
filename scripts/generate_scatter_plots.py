"""Publication-quality scatter plots of martsDB structural domains.

Emits 18 scatter plots × {PNG, SVG} under
``outputs/domain_clustering/scatter/``:

For each subset in {all, alpha, nonalpha}:
    For each method in {pca, tsne, umap}:
        * ``…_<subset>_<method>_subtype.{png,svg}``  — colored by canonical
          subtype (α1A..I, α2A, α3A..B, α4A..B, β, γ, δ1..3, ε, ζ).
          Uses the palette from ``data/domain_subtype_palette.json`` so
          colors match the dendrogram exactly.
        * ``…_<subset>_<method>_group.{png,svg}``    — colored by the
          "group" scheme:
             * subset=all      → main types {α, β, γ, δ, ε, ζ}
             * subset=alpha    → alpha parent groups {α1, α2, α3, α4}
             * subset=nonalpha → main types {β, γ, δ, ε} (ζ excluded per user)
          Colors come from ``MAIN_TYPE_COLORS`` (shared with the
          dendrogram's branch coloring).

Inputs consumed: the cached (1 − TM) distance matrix, member id list, and
the canonical subtype pkl. Distances are used precomputed for tSNE / UMAP;
for PCA each domain's distance-row is used as a 2427-dim feature vector.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # type: ignore  # noqa: E402
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("scatter")


# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

# Nature Chemical Biology figure specs:
#   * Sans-serif (Arial / Helvetica), embedded/editable text in SVG.
#   * Column width 89 mm / double 183 mm; min in-print label ~5-7 pt.
#   * 300 dpi minimum for raster; prefer vector.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "regular",
    "axes.labelsize": 7,
    "axes.labelpad": 3.0,
    "axes.linewidth": 0.6,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.0,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Shared palette (from the paper figures)
# ---------------------------------------------------------------------------

def _load_main_type_palette() -> dict[str, str]:
    import json as _json
    with open("data/domain_main_type_palette.json") as _fh:
        raw = _json.load(_fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


MAIN_TYPE_COLORS: dict[str, str] = _load_main_type_palette()


def _main_type_of(subtype: str) -> str | None:
    for main in MAIN_TYPE_COLORS:
        if subtype.startswith(main):
            return main
    return None


# Greek-letter display form
_GREEK_MAP: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ",
    "delta": "δ", "epsilon": "ε", "zeta": "ζ",
}
def to_greek(label: str) -> str:
    if not isinstance(label, str):
        return label
    for prefix, greek in _GREEK_MAP.items():
        if label.startswith(prefix):
            return greek + label[len(prefix):]
    return label


def alpha_parent_of(subtype: str) -> str | None:
    """Alpha parent group: alpha1, alpha2, alpha3, alpha4 (returns None for non-alpha)."""
    if subtype.startswith("alpha") and len(subtype) >= 6 and subtype[5].isdigit():
        return f"alpha{subtype[5]}"
    return None


def alpha_parent_palette() -> dict[str, str]:
    """Four blue tones from the alpha family's Blues ramp.

    Ordered α1→α4 (light-to-dark), matching both the alphabetical order
    and the dendrogram's left-to-right traversal of the alpha family
    (α4 sub-clade is displayed on the right via a linkage flip so the two
    orderings agree).
    """
    cmap = mpl.colormaps["Blues"]
    tones = [0.35, 0.55, 0.75, 0.95]
    return {f"alpha{i+1}": mpl.colors.to_hex(cmap(t)) for i, t in enumerate(tones)}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_embedding(distance_matrix: np.ndarray, method: str,
                      *, seed: int = 42) -> np.ndarray:
    """Return an (n, 2) embedding of ``distance_matrix`` under ``method``.

    * ``pca`` — sklearn PCA on the distance-matrix rows (each row is a
      len(n)-dim feature vector). Fast, deterministic, linear.
    * ``tsne`` — sklearn TSNE with metric='precomputed'.
    * ``umap`` — umap-learn UMAP with metric='precomputed'.
    """
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(distance_matrix)
    if method == "tsne":
        from sklearn.manifold import TSNE
        n = distance_matrix.shape[0]
        perplexity = min(30, max(5, (n - 1) // 4))
        return TSNE(
            n_components=2, metric="precomputed", init="random",
            random_state=seed, perplexity=perplexity, learning_rate="auto",
            max_iter=1500,
        ).fit_transform(distance_matrix)
    if method == "umap":
        import umap
        n = distance_matrix.shape[0]
        n_neighbors = min(30, max(5, n // 20))
        return umap.UMAP(
            n_components=2, metric="precomputed",
            n_neighbors=n_neighbors, min_dist=0.15,
            random_state=seed,
        ).fit_transform(distance_matrix)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    logger.info("Saved %s.{png,svg}", stem)


def scatter(
    embedding: np.ndarray, labels: np.ndarray,
    label_order: list[str], palette: dict[str, str],
    title: str, xlabel: str, ylabel: str, legend_title: str,
    out_stem: Path,
    *, point_size: float = 9.0, alpha: float = 0.78,
) -> None:
    """Render a legend-outside scatter with one color per label group.

    Sized to a Nature Chemical Biology single-column panel (~89 mm wide
    including the outside legend). Points are rendered largest-group-first
    so that small subtypes are painted on top and remain visible.
    """
    fig, ax = plt.subplots(figsize=(3.9, 3.15))

    counts = {lbl: int((labels == lbl).sum()) for lbl in label_order}
    draw_order = sorted(label_order, key=lambda l: -counts[l])

    for lbl in draw_order:
        mask = labels == lbl
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=palette.get(lbl, "#888888"),
            s=point_size, alpha=alpha,
            edgecolors="white", linewidths=0.25,
            rasterized=True,
        )

    ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.04)

    handles = [
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=palette.get(lbl, "#888888"),
            markeredgecolor="white", markeredgewidth=0.25, markersize=4.5,
            label=to_greek(lbl),
        )
        for lbl in label_order if counts[lbl] > 0
    ]
    ax.legend(
        handles=handles,
        loc="upper left", bbox_to_anchor=(1.015, 1.0),
        title=legend_title, ncol=1,
    )
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    hac_dir = Path("outputs/domain_clustering")
    out_dir = hac_dir / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    D = np.load(hac_dir / "intermediate/distance_matrix.npy")
    member_ids = pickle.load(open(hac_dir / "intermediate/member_ids.pkl", "rb"))
    subtype_map = pickle.load(open("data/domain_module_id_2_domain_subtype.pkl", "rb"))
    subtype_palette = json.loads(Path("data/domain_subtype_palette.json").read_text())

    all_subtypes = np.array([subtype_map[m] for m in member_ids])
    all_main = np.array([_main_type_of(s) for s in all_subtypes])
    all_alpha_parent = np.array([alpha_parent_of(s) for s in all_subtypes])

    def _idx_mask(mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0]

    # Subset definitions.
    subsets = {
        "all":       _idx_mask(np.ones(len(member_ids), dtype=bool)),
        "alpha":     _idx_mask(all_main == "alpha"),
        "nonalpha":  _idx_mask(all_main != "alpha"),
    }

    # Group coloring per subset.
    ap_pal = alpha_parent_palette()
    group_scheme = {
        "all": {
            "labels":     lambda idx: all_main[idx],
            "order":      ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
            "palette":    MAIN_TYPE_COLORS,
            "kind":       "main type",
            "legend":     "Domain type",
        },
        "alpha": {
            "labels":     lambda idx: all_alpha_parent[idx],
            "order":      ["alpha1", "alpha2", "alpha3", "alpha4"],
            "palette":    ap_pal,
            "kind":       "parent group",
            "legend":     "Parent group",
        },
        "nonalpha": {
            "labels":     lambda idx: all_main[idx],
            "order":      ["beta", "gamma", "delta", "epsilon"],
            "palette":    MAIN_TYPE_COLORS,
            "kind":       "main type",
            "legend":     "Domain type",
        },
    }

    # Subtype ordering per subset — same order as the dendrogram legend
    # (light-to-dark within each family).
    def _subtype_order(idx: np.ndarray) -> list[str]:
        present = sorted(set(all_subtypes[idx]))
        # Order matching the paper palette's dendrogram order.
        order_hint = list(subtype_palette.keys())
        return [s for s in order_hint if s in present]

    method_labels = {
        "pca":  ("PCA-1", "PCA-2"),
        "tsne": ("t-SNE-1", "t-SNE-2"),
        "umap": ("UMAP-1", "UMAP-2"),
    }
    method_titles = {
        "pca": "PCA", "tsne": "t-SNE", "umap": "UMAP",
    }
    subset_descriptor = {
        "all":      "MARTS-DB structural landscape",
        "alpha":    "alpha-family structural landscape",
        "nonalpha": "non-alpha structural landscape",
    }

    # For each subset × method: compute embedding once, then plot twice
    # (subtype coloring + group coloring).
    for subset_name, idx in subsets.items():
        D_sub = D[np.ix_(idx, idx)]
        logger.info("Subset %s: %d domains", subset_name, len(idx))

        for method in ("pca", "tsne", "umap"):
            logger.info("  computing %s embedding …", method)
            emb = compute_embedding(D_sub, method)

            desc = subset_descriptor[subset_name]
            method_title = method_titles[method]

            # 1) subtype-colored
            sub_labels = all_subtypes[idx]
            sub_order = _subtype_order(idx)
            scatter(
                emb, sub_labels, sub_order, subtype_palette,
                title=f"{method_title} of the {desc} resolved by subtype",
                xlabel=method_labels[method][0], ylabel=method_labels[method][1],
                legend_title="Domain subtype",
                out_stem=out_dir / f"scatter_{subset_name}_{method}_subtype",
            )

            # 2) group-colored
            gcfg = group_scheme[subset_name]
            group_labels = gcfg["labels"](idx)
            mask_group = np.array([lbl in gcfg["order"] for lbl in group_labels])
            emb_g = emb[mask_group]
            group_labels_f = group_labels[mask_group]
            scatter(
                emb_g, group_labels_f, gcfg["order"], gcfg["palette"],
                title=f"{method_title} of the {desc} resolved by {gcfg['kind']}",
                xlabel=method_labels[method][0], ylabel=method_labels[method][1],
                legend_title=gcfg["legend"],
                out_stem=out_dir / f"scatter_{subset_name}_{method}_group",
            )


def alpha_plus_zeta() -> None:
    """Extra 3-plot set: alpha (α1–α4) + zeta domains, colored by parent group / ζ."""
    hac_dir = Path("outputs/domain_clustering")
    out_dir = hac_dir / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    D = np.load(hac_dir / "intermediate/distance_matrix.npy")
    member_ids = pickle.load(open(hac_dir / "intermediate/member_ids.pkl", "rb"))
    subtype_map = pickle.load(open("data/domain_module_id_2_domain_subtype.pkl", "rb"))

    subtypes = np.array([subtype_map[m] for m in member_ids])
    main_types = np.array([_main_type_of(s) for s in subtypes])
    parents = np.array([alpha_parent_of(s) for s in subtypes])

    keep = (main_types == "alpha") | (main_types == "zeta")
    idx = np.where(keep)[0]
    D_sub = D[np.ix_(idx, idx)]

    labels = np.where(main_types[idx] == "zeta", "zeta", parents[idx])
    order = ["alpha1", "alpha2", "alpha3", "alpha4", "zeta"]
    ap_pal = alpha_parent_palette()
    palette = {**ap_pal, "zeta": MAIN_TYPE_COLORS["zeta"]}

    method_labels = {
        "pca":  ("PCA-1", "PCA-2"),
        "tsne": ("t-SNE-1", "t-SNE-2"),
        "umap": ("UMAP-1", "UMAP-2"),
    }
    method_titles = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    desc = "alpha-family and zeta structural landscape"

    logger.info("Subset alpha+zeta: %d domains", len(idx))
    for method in ("pca", "tsne", "umap"):
        logger.info("  computing %s embedding …", method)
        emb = compute_embedding(D_sub, method)
        scatter(
            emb, labels, order, palette,
            title=f"{method_titles[method]} of the {desc} resolved by parent group",
            xlabel=method_labels[method][0], ylabel=method_labels[method][1],
            legend_title="Parent group",
            out_stem=out_dir / f"scatter_alpha_zeta_{method}_group",
        )


def nonalpha_subtypes() -> None:
    """Extra 3-plot set: non-alpha subtypes (β, γ, ε, δ1, δ2, δ3), subtype-colored."""
    hac_dir = Path("outputs/domain_clustering")
    out_dir = hac_dir / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    D = np.load(hac_dir / "intermediate/distance_matrix.npy")
    member_ids = pickle.load(open(hac_dir / "intermediate/member_ids.pkl", "rb"))
    subtype_map = pickle.load(open("data/domain_module_id_2_domain_subtype.pkl", "rb"))
    subtype_palette = json.loads(Path("data/domain_subtype_palette.json").read_text())

    subtypes = np.array([subtype_map[m] for m in member_ids])
    order = ["beta", "gamma", "delta1", "delta2", "delta3", "epsilon"]
    keep = np.isin(subtypes, order)
    idx = np.where(keep)[0]
    D_sub = D[np.ix_(idx, idx)]
    labels = subtypes[idx]

    palette = {s: subtype_palette[s] for s in order}

    method_labels = {
        "pca":  ("PCA-1", "PCA-2"),
        "tsne": ("t-SNE-1", "t-SNE-2"),
        "umap": ("UMAP-1", "UMAP-2"),
    }
    method_titles = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    desc = "non-alpha structural landscape"

    logger.info("Subset non-alpha subtypes: %d domains", len(idx))
    for method in ("pca", "tsne", "umap"):
        logger.info("  computing %s embedding …", method)
        emb = compute_embedding(D_sub, method)
        scatter(
            emb, labels, order, palette,
            title=f"{method_titles[method]} of the {desc} resolved by subtype",
            xlabel=method_labels[method][0], ylabel=method_labels[method][1],
            legend_title="Domain subtype",
            out_stem=out_dir / f"scatter_nonalpha_sub_{method}_subtype",
        )


def nonalpha_subtypes_with_zeta() -> None:
    """Extra 3-plot set: non-alpha subtypes β, γ, δ1, δ2, δ3, ε, ζ."""
    hac_dir = Path("outputs/domain_clustering")
    out_dir = hac_dir / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    D = np.load(hac_dir / "intermediate/distance_matrix.npy")
    member_ids = pickle.load(open(hac_dir / "intermediate/member_ids.pkl", "rb"))
    subtype_map = pickle.load(open("data/domain_module_id_2_domain_subtype.pkl", "rb"))
    subtype_palette = json.loads(Path("data/domain_subtype_palette.json").read_text())

    subtypes = np.array([subtype_map[m] for m in member_ids])
    order = ["beta", "gamma", "delta1", "delta2", "delta3", "epsilon", "zeta"]
    keep = np.isin(subtypes, order)
    idx = np.where(keep)[0]
    D_sub = D[np.ix_(idx, idx)]
    labels = subtypes[idx]
    palette = {s: subtype_palette[s] for s in order}

    method_labels = {
        "pca":  ("PCA-1", "PCA-2"),
        "tsne": ("t-SNE-1", "t-SNE-2"),
        "umap": ("UMAP-1", "UMAP-2"),
    }
    method_titles = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    desc = "non-alpha structural landscape"

    logger.info("Subset non-alpha subtypes + zeta: %d domains", len(idx))
    for method in ("pca", "tsne", "umap"):
        logger.info("  computing %s embedding …", method)
        emb = compute_embedding(D_sub, method)
        scatter(
            emb, labels, order, palette,
            title=f"{method_titles[method]} of the {desc} resolved by subtype",
            xlabel=method_labels[method][0], ylabel=method_labels[method][1],
            legend_title="Domain subtype",
            out_stem=out_dir / f"scatter_nonalpha_subz_{method}_subtype",
        )


if __name__ == "__main__":
    main()
    alpha_plus_zeta()
    nonalpha_subtypes()
    nonalpha_subtypes_with_zeta()
