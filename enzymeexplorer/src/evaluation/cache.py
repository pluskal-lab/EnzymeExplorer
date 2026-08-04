"""v4 paired-bootstrap cache.

The v4 bootstrap runs *jointly* over all methods being compared so the
draw indices can be shared (paired). That changes the cache key from
"one entry per classifier" to "one entry per (classifier set, classes,
metrics, n_bootstraps, seed, ap_types, target_model)". Two
``evaluate`` invocations with the same set of methods + classes + RNG
parameters reuse the exact same cache entry; one with a superset of
methods is a fresh entry.

Cache layout::

    outputs/evaluation_results/_bootstrap_cache/<hash>/
        bootstrap_long_ap.csv
        point_estimates_ap.csv
        bootstrap_long_delta.csv
        point_estimates_delta.csv
        meta.json

The classifier identity in the hash is built from each classifier's
``(model, version_spec, fold_sizes, per-class experiment_timestamps)``
so retraining a classifier (new timestamp under the same version label)
busts the cache automatically — we never silently reuse stale draws.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation import bootstrap as bs
from enzymeexplorer.src.evaluation.io import FoldDfs
from enzymeexplorer.src.utils.project_info import get_evaluations_output

logger = logging.getLogger(__name__)

CACHE_DIRNAME = "_bootstrap_cache"

# Bumped whenever the bootstrap algorithm semantically changes.
# v4: paired bootstrap with shared draws across methods. Two AP types
#     (pooled_oof, fold_mean) computed in one call. Per-draw deltas
#     between method pairs.
# v5: cluster-block bootstrap by default — resampling unit is a
#     50%-seq-identity mmseqs cluster instead of an individual row.
#     Cache key incorporates the cluster-map hash + bootstrap_unit so
#     switching mapping versions (or falling back to row bootstrap)
#     lands in a fresh cache dir.
BOOTSTRAP_ALGO_VERSION = "v5-cluster-block"


def _classifier_signature(
    label: str,
    model: str,
    version_spec: str | Mapping[str, str],
    class_to_fold_dfs: dict[str, dict[int, FoldDfs]],
    timestamps: Mapping[str, str],
) -> dict:
    """Stable JSON-friendly fingerprint of one classifier."""
    if isinstance(version_spec, Mapping):
        version_norm: object = {k: version_spec[k] for k in sorted(version_spec)}
    else:
        version_norm = version_spec
    fold_sizes: list[list[int]] = []
    for cls in sorted(class_to_fold_dfs):
        fd = class_to_fold_dfs[cls]
        for f in sorted(fd):
            fold_sizes.append([cls.__hash__(), f, len(fd[f][0])])
    return {
        "label": label,
        "model": model,
        "version_spec": version_norm,
        "classes": sorted(class_to_fold_dfs),
        "fold_sizes": fold_sizes,
        "timestamps": [[k, timestamps[k]] for k in sorted(timestamps)],
    }


@dataclass(frozen=True)
class CacheKey:
    """Hashable identifier for a paired-bootstrap call.

    ``classifier_signatures`` is stored as a tuple of canonical
    JSON-string fingerprints (one per classifier), which is the most
    robust way to make the cache key both stable and inspectable.

    ``bootstrap_unit`` and ``cluster_map_hash`` were added in v5 so a
    switch between row / cluster resampling — or a rebuild of the
    cluster TSV — automatically routes to a fresh cache dir.
    """

    classifier_signatures: tuple[str, ...]
    n_bootstraps: int
    seed: int
    ap_types: tuple[str, ...]
    metrics: tuple[str, ...]
    target_model: str | None
    bootstrap_unit: str = "clusters"
    cluster_map_hash: str | None = None
    algo_version: str = BOOTSTRAP_ALGO_VERSION

    def to_dict(self) -> dict:
        return {
            "classifier_signatures": [json.loads(s) for s in self.classifier_signatures],
            "n_bootstraps": self.n_bootstraps,
            "seed": self.seed,
            "ap_types": list(self.ap_types),
            "metrics": list(self.metrics),
            "target_model": self.target_model,
            "bootstrap_unit": self.bootstrap_unit,
            "cluster_map_hash": self.cluster_map_hash,
            "algo_version": self.algo_version,
        }

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]


def cache_root(custom_root: Path | None = None) -> Path:
    base = custom_root or get_evaluations_output()
    return base / CACHE_DIRNAME


def cache_dir_for(key_hash: str, *, custom_root: Path | None = None) -> Path:
    return cache_root(custom_root) / key_hash


def load_cached(cache_path: Path) -> bs.BootstrapResult | None:
    if not cache_path.exists():
        return None
    # jackknife_ap.csv + jackknife_delta.csv are also expected — required
    # for BCa CIs. Cache entries lacking them are treated as incomplete
    # (regenerated on next run) so we never silently drop the jackknife.
    needed = [
        "bootstrap_long_ap.csv",
        "point_estimates_ap.csv",
        "bootstrap_long_delta.csv",
        "point_estimates_delta.csv",
        "jackknife_ap.csv",
        "jackknife_delta.csv",
        "meta.json",
    ]
    if not all((cache_path / fname).exists() for fname in needed):
        return None
    return bs.BootstrapResult.load(cache_path)


def save_cached(
    cache_path: Path, result: bs.BootstrapResult, key: CacheKey
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    result.save(cache_path)
    with open(cache_path / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            key.to_dict() | {"hash": key.hash()},
            fh, indent=2, sort_keys=True, default=str,
        )


def paired_bootstrap_with_cache(
    classifier_to_class_to_fold_dfs: bs.ClassifierClassFoldDfs,
    classifier_metadata: dict[str, dict],
    *,
    metrics: tuple[str, ...],
    ap_types: tuple[str, ...],
    n_bootstraps: int,
    seed: int,
    target_model: str | None,
    force: bool = False,
    custom_root: Path | None = None,
    bootstrap_unit: str = "clusters",
    cluster_map: dict[str, str] | None = None,
    cluster_map_hash: str | None = None,
) -> tuple[bs.BootstrapResult, bool]:
    """Run or reuse a paired bootstrap.

    ``classifier_metadata`` is ``{label: {model, version_spec,
    timestamps}}`` as produced by the CLI's classifier resolver — used
    to build the cache key so retraining busts it.

    Pass ``bootstrap_unit="clusters"`` with a ``cluster_map`` (and its
    precomputed ``cluster_map_hash``) to route to the v5 cluster-block
    path; ``bootstrap_unit="rows"`` restores the legacy v4 behavior.
    """
    sigs = []
    for label in sorted(classifier_to_class_to_fold_dfs):
        meta = classifier_metadata[label]
        sigs.append(
            _classifier_signature(
                label=label,
                model=meta["model"],
                version_spec=meta["version_spec"],
                class_to_fold_dfs=classifier_to_class_to_fold_dfs[label],
                timestamps=meta["timestamps"],
            )
        )
    sigs_tuple = tuple(json.dumps(s, sort_keys=True, default=str) for s in sigs)

    key = CacheKey(
        classifier_signatures=sigs_tuple,
        n_bootstraps=n_bootstraps,
        seed=seed,
        ap_types=tuple(ap_types),
        metrics=tuple(metrics),
        target_model=target_model,
        bootstrap_unit=bootstrap_unit,
        cluster_map_hash=cluster_map_hash,
    )
    cache_path = cache_dir_for(key.hash(), custom_root=custom_root)
    if not force:
        cached = load_cached(cache_path)
        if cached is not None:
            logger.info("Paired-bootstrap cache hit @ %s", cache_path)
            return cached, True
    logger.info(
        "Paired-bootstrap cache miss @ %s — running %s bootstrap "
        "(unit=%s)", cache_path, BOOTSTRAP_ALGO_VERSION, bootstrap_unit,
    )
    result = bs.paired_bootstrap_metric_cis(
        classifier_to_class_to_fold_dfs,
        metrics=metrics,
        ap_types=cast(tuple[bs.ApType, ...], tuple(ap_types)),
        n_bootstraps=n_bootstraps,
        seed=seed,
        target_model=target_model,
        bootstrap_unit=cast(bs.BootstrapUnit, bootstrap_unit),
        cluster_map=cluster_map,
    )
    save_cached(cache_path, result, key)
    return result, False
