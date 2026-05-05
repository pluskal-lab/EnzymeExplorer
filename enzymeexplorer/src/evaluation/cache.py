"""Per-classifier bootstrap cache.

Each classifier's bootstrap draws are cached to disk under
``outputs/evaluation_results/_bootstrap_cache/<classifier_label>/<hash>/``
so that repeated evaluations across the four "purpose" output dirs
(all-methods comparison, ablation, confidence tiers, sweeps) reuse the
same draws without re-running bootstrap. Hash key combines the resolved
version spec, fold sizes, and bootstrap settings (n_bootstraps, seed,
mode) — distractor universe falls out of the version spec automatically
because ``_no_distractors`` is part of the version string.

Categorical bootstrap is not cached here (it depends on a category mask
that varies per output dir and is cheap enough to re-run).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation import bootstrap as bs
from enzymeexplorer.src.evaluation.io import FoldDfs
from enzymeexplorer.src.utils.project_info import get_evaluations_output

logger = logging.getLogger(__name__)

CACHE_DIRNAME = "_bootstrap_cache"


@dataclass(frozen=True)
class CacheKey:
    """Inputs that uniquely identify a cached bootstrap result.

    ``timestamps`` is the per-class experiment-timestamp map captured at load
    time — including it ensures that a re-trained classifier (new timestamp
    under the same version label) busts the cache instead of silently reusing
    stale draws.
    """

    classifier: str
    model: str
    version_spec: str | dict[str, str]
    classes: tuple[str, ...]
    fold_sizes: tuple[tuple[int, int], ...]  # (fold_idx, n_rows) per class shared
    timestamps: tuple[tuple[str, str], ...]  # (class, experiment_dir_name)
    n_bootstraps: int
    seed: int
    mode: str
    metrics: tuple[str, ...]

    def to_dict(self) -> dict:
        version: object = self.version_spec
        if isinstance(version, Mapping):
            version = {k: version[k] for k in sorted(version)}
        return {
            "classifier": self.classifier,
            "model": self.model,
            "version_spec": version,
            "classes": list(self.classes),
            "fold_sizes": [list(t) for t in self.fold_sizes],
            "timestamps": [list(t) for t in self.timestamps],
            "n_bootstraps": self.n_bootstraps,
            "seed": self.seed,
            "mode": self.mode,
            "metrics": list(self.metrics),
        }

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]


def _fold_sizes(class_to_fold_dfs: dict[str, dict[int, FoldDfs]]) -> tuple[tuple[int, int], ...]:
    """Per-fold row counts. Folds share row counts across classes when the
    classifier was loaded from a single experiment dir; for HBI per-class
    specs different classes can come from different experiments, so we hash
    the union of fold sizes seen across (class, fold)."""
    seen: set[tuple[str, int, int]] = set()
    for cls, fold_dfs in class_to_fold_dfs.items():
        for fold_idx, (lab, _) in fold_dfs.items():
            seen.add((cls, fold_idx, len(lab)))
    # Aggregate to (fold_idx, n_rows) deduped — preserves per-fold distinctions
    # while keeping the key compact.
    by_fold: dict[int, int] = {}
    for _, fold_idx, n in seen:
        # If two classes report the same fold but different row counts
        # (HBI per-class spec), bake them both into the hash via a synthetic
        # offset so different layouts don't collide.
        prev = by_fold.get(fold_idx)
        if prev is None or n == prev:
            by_fold[fold_idx] = n
        else:
            by_fold[fold_idx] = max(prev, n)
    return tuple(sorted(by_fold.items()))


def cache_root(custom_root: Path | None = None) -> Path:
    """Return the cache root directory."""
    base = custom_root or get_evaluations_output()
    return base / CACHE_DIRNAME


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)


def cache_dir_for(
    classifier: str,
    key_hash: str,
    *,
    custom_root: Path | None = None,
) -> Path:
    """Return ``<cache_root>/<safe(classifier)>/<hash>/``."""
    return cache_root(custom_root) / _safe_label(classifier) / key_hash


def load_cached(
    cache_path: Path,
) -> bs.BootstrapResult | None:
    """Return a ``BootstrapResult`` from ``cache_path`` or ``None`` if absent
    or incomplete."""
    if not cache_path.exists():
        return None
    needed = ["bootstrap_long.csv", "point_estimates.csv", "jackknife.csv", "meta.json"]
    if not all((cache_path / fname).exists() for fname in needed):
        return None
    return bs.BootstrapResult(
        long_df=pd.read_csv(cache_path / "bootstrap_long.csv"),
        point_estimates=pd.read_csv(cache_path / "point_estimates.csv"),
        jackknife=pd.read_csv(cache_path / "jackknife.csv"),
    )


def save_cached(
    cache_path: Path,
    result: bs.BootstrapResult,
    key: CacheKey,
) -> None:
    """Write a ``BootstrapResult`` and its meta.json into ``cache_path``."""
    cache_path.mkdir(parents=True, exist_ok=True)
    result.save(cache_path)
    with open(cache_path / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(key.to_dict() | {"hash": key.hash()}, fh, indent=2, sort_keys=True)


def bootstrap_with_cache(
    classifier: str,
    model: str,
    version_spec: str | dict[str, str],
    class_to_fold_dfs: dict[str, dict[int, FoldDfs]],
    *,
    timestamps: dict[str, str],
    metrics: tuple[str, ...],
    n_bootstraps: int,
    seed: int,
    mode: str,
    force: bool = False,
    custom_root: Path | None = None,
) -> tuple[bs.BootstrapResult, bool]:
    """Return a cached or freshly-computed ``BootstrapResult`` for one
    classifier. The second item is ``True`` when the cache was reused.

    ``timestamps`` is the ``{class_short: experiment_dir_name}`` map captured
    by ``io.load_classifier_class_fold_dfs`` — included in the cache hash so
    that a re-trained classifier doesn't silently reuse stale draws.
    """
    key = CacheKey(
        classifier=classifier,
        model=model,
        version_spec=version_spec,
        classes=tuple(sorted(class_to_fold_dfs)),
        fold_sizes=_fold_sizes(class_to_fold_dfs),
        timestamps=tuple(sorted(timestamps.items())),
        n_bootstraps=n_bootstraps,
        seed=seed,
        mode=mode,
        metrics=tuple(metrics),
    )
    cache_path = cache_dir_for(classifier, key.hash(), custom_root=custom_root)
    if not force:
        cached = load_cached(cache_path)
        if cached is not None:
            logger.info("Bootstrap cache hit for %s @ %s", classifier, cache_path)
            return cached, True
    logger.info("Bootstrap cache miss for %s — running bootstrap", classifier)
    result = bs.bootstrap_metric_cis(
        {classifier: class_to_fold_dfs},
        metrics=metrics,
        n_bootstraps=n_bootstraps,
        seed=seed,
        mode=mode,
    )
    save_cached(cache_path, result, key)
    return result, False


def merge_results(parts: list[bs.BootstrapResult]) -> bs.BootstrapResult:
    """Concatenate per-classifier ``BootstrapResult`` tables."""
    if not parts:
        return bs.BootstrapResult(
            long_df=pd.DataFrame(),
            point_estimates=pd.DataFrame(),
            jackknife=pd.DataFrame(),
        )
    return bs.BootstrapResult(
        long_df=pd.concat([p.long_df for p in parts], ignore_index=True),
        point_estimates=pd.concat([p.point_estimates for p in parts], ignore_index=True),
        jackknife=pd.concat([p.jackknife for p in parts], ignore_index=True),
    )
