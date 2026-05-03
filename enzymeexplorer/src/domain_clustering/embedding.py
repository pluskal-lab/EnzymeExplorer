"""2D embedding of detected domains for cluster-scatter visualisations.

UMAP (with ``metric='precomputed'``) on the dense ``(1 − TM)`` distance
matrix. Computed **once** per (pdb-set, parameter-set) and cached so every
clustering — Foldseek and HAC, every threshold — colors points on the
identical layout. Same domain → same (x, y), with only the colors changing
across plots, which makes side-by-side comparison meaningful.

Why UMAP rather than PCA / MDS / t-SNE: PCA on a TM-distance matrix is
linear and badly distorts neighbour relationships. t-SNE is slower and
preserves only local structure. UMAP preserves local *and* global
structure, scales to thousands of points in seconds, and is deterministic
when seeded.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np  # type: ignore

from enzymeexplorer.src.domain_clustering import hac as _hac

logger = logging.getLogger(__name__)


def _build_distance_matrix(
    member_ids: list[str],
    pairwise_tm: dict[tuple[str, str], float],
    missing_distance: float = 1.0,
) -> np.ndarray:
    """Reuse HAC's densifier (same convention: missing pair → max distance)."""
    return _hac.build_distance_matrix(
        member_ids, pairwise_tm, missing_distance=missing_distance,
    )


def compute_umap_embedding(
    member_ids: list[str],
    pairwise_tm: dict[tuple[str, str], float],
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    random_state: int = 42,
    missing_distance: float = 1.0,
    n_components: int = 2,
) -> np.ndarray:
    """Fit UMAP on the precomputed (1 − TM) distance matrix.

    Returns an ``(n_members, n_components)`` array, indexed by ``member_ids``.
    """
    import umap  # type: ignore  # local import — heavy

    D = _build_distance_matrix(member_ids, pairwise_tm, missing_distance)
    logger.info(
        "Fitting UMAP: n=%d, n_neighbors=%d, min_dist=%.2f, seed=%d",
        len(member_ids), n_neighbors, min_dist, random_state,
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(D)
    logger.info("UMAP embedding shape: %s", embedding.shape)
    return embedding


def load_or_compute_embedding(
    cache_dir: str | Path,
    *,
    member_ids: list[str] | None = None,
    pairwise_tm: dict[tuple[str, str], float] | None = None,
    method: str = "umap",
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    random_state: int = 42,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Load a cached embedding, or compute it and cache it.

    The cache uses a deterministic filename keyed by ``method`` and the
    main hyper-parameters so re-runs with different settings don't
    overwrite each other. Member-ID order is saved alongside; downstream
    plots index into the embedding via that order.
    """
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"embedding_{method}_n{n_neighbors}"
        f"_md{min_dist:.2f}_seed{random_state}"
    )
    embedding_path = cache_dir / f"{base}.npy"
    member_path = cache_dir / f"{base}.member_ids.pkl"

    if not force and embedding_path.exists() and member_path.exists():
        with open(member_path, "rb") as f:
            cached_ids = pickle.load(f)
        embedding = np.load(embedding_path)
        logger.info(
            "Reusing %s embedding from %s (n=%d)",
            method, embedding_path, len(cached_ids),
        )
        return embedding, cached_ids

    if member_ids is None or pairwise_tm is None:
        raise FileNotFoundError(
            f"No cached embedding at {embedding_path}; pass member_ids "
            f"and pairwise_tm to compute it."
        )
    if method != "umap":
        raise ValueError(f"Unsupported embedding method: {method}")

    embedding = compute_umap_embedding(
        member_ids, pairwise_tm,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    np.save(embedding_path, embedding)
    with open(member_path, "wb") as f:
        pickle.dump(list(member_ids), f)
    logger.info("Saved embedding → %s", embedding_path)
    return embedding, list(member_ids)
