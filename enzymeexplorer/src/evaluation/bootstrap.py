"""Bootstrap utilities for the evaluation pipeline.

Implements two-stage cluster bootstrap (default) and row-only bootstrap over
per-fold (labels, predictions) pairs. Produces a long-form DataFrame of
metric draws together with point estimates and a leave-one-fold-out
jackknife table; downstream code derives percentile or BCa confidence
intervals from these artefacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs

logger = logging.getLogger(__name__)

BootstrapMode = Literal["cluster", "rows"]
CiMethod = Literal["percentile", "bca"]

ClassifierClassFoldDfs = dict[str, dict[str, dict[int, FoldDfs]]]


@dataclass
class BootstrapResult:
    """Long-form bootstrap draws plus point-estimate and jackknife tables."""

    long_df: pd.DataFrame
    point_estimates: pd.DataFrame
    jackknife: pd.DataFrame

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.long_df.to_csv(out_dir / "bootstrap_long.csv", index=False)
        self.point_estimates.to_csv(out_dir / "point_estimates.csv", index=False)
        self.jackknife.to_csv(out_dir / "jackknife.csv", index=False)

    @classmethod
    def load(cls, in_dir: Path) -> "BootstrapResult":
        return cls(
            long_df=pd.read_csv(in_dir / "bootstrap_long.csv"),
            point_estimates=pd.read_csv(in_dir / "point_estimates.csv"),
            jackknife=pd.read_csv(in_dir / "jackknife.csv"),
        )


def _safe_metric(name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # AP is 0/1 in degenerate cases; ROC-AUC undefined. Treat both as NaN
        # so they propagate cleanly through quantile aggregation.
        return float("nan")
    if name == "ap":
        return float(average_precision_score(y_true, y_score))
    if name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    raise ValueError(f"Unsupported metric: {name}")


def _gather_arrays(
    fold_dfs: dict[int, FoldDfs], cls: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        f: (
            np.asarray(lp[0][cls].to_numpy(), dtype=np.int8),
            np.asarray(lp[1][cls].to_numpy(), dtype=np.float64),
        )
        for f, lp in fold_dfs.items()
    }


def bootstrap_metric_cis(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    *,
    metrics: Iterable[str] = ("ap", "roc_auc"),
    n_bootstraps: int = 1000,
    seed: int = 42,
    mode: BootstrapMode = "rows",
    progress: bool = True,
) -> BootstrapResult:
    """Run bootstrap over (classifier × class × metric).

    ``mode='cluster'``: per draw, resample fold IDs with replacement
    (size = n_folds), then resample rows within each picked fold,
    concatenate, score once.

    ``mode='rows'``: per fold, draw bootstrap row samples with a fresh
    seeded RNG, score the metric per fold, then average across folds for
    each draw index. Matches the layout of the legacy hbi notebook.
    """
    metrics = tuple(metrics)
    long_records: list[dict] = []
    point_records: list[dict] = []
    jk_records: list[dict] = []

    items = [
        (clf, cls, fold_dfs)
        for clf, cls_map in classifier_to_class_to_fold_dfs.items()
        for cls, fold_dfs in cls_map.items()
    ]
    iterator = tqdm(items, desc="Bootstrap", unit="(clf,cls)") if progress else items

    for clf, cls, fold_dfs in iterator:
        arrays = _gather_arrays(fold_dfs, cls)
        fold_ids = sorted(arrays)
        n_folds = len(fold_ids)

        full_y = np.concatenate([arrays[f][0] for f in fold_ids])
        full_s = np.concatenate([arrays[f][1] for f in fold_ids])
        for m in metrics:
            point_records.append(
                {
                    "classifier": clf,
                    "class": cls,
                    "metric": m,
                    "value": _safe_metric(m, full_y, full_s),
                }
            )

        for f_left in fold_ids:
            kept = [f for f in fold_ids if f != f_left]
            jk_y = np.concatenate([arrays[f][0] for f in kept])
            jk_s = np.concatenate([arrays[f][1] for f in kept])
            for m in metrics:
                jk_records.append(
                    {
                        "classifier": clf,
                        "class": cls,
                        "metric": m,
                        "fold_left_out": f_left,
                        "value": _safe_metric(m, jk_y, jk_s),
                    }
                )

        if mode == "cluster":
            rng = np.random.default_rng(seed)
            for b in range(n_bootstraps):
                picks = rng.choice(fold_ids, size=n_folds, replace=True)
                ys, ss = [], []
                for f in picks:
                    y, s = arrays[f]
                    idx = rng.integers(0, len(y), size=len(y))
                    ys.append(y[idx])
                    ss.append(s[idx])
                yy = np.concatenate(ys)
                sc = np.concatenate(ss)
                for m in metrics:
                    long_records.append(
                        {
                            "classifier": clf,
                            "class": cls,
                            "metric": m,
                            "bootstrap_idx": b,
                            "value": _safe_metric(m, yy, sc),
                        }
                    )
        elif mode == "rows":
            per_fold_draws: dict[int, dict[str, list[float]]] = {
                f: {m: [] for m in metrics} for f in fold_ids
            }
            for f in fold_ids:
                rng_f = np.random.default_rng(seed)
                y, s = arrays[f]
                n = len(y)
                for _ in range(n_bootstraps):
                    idx = rng_f.integers(0, n, size=n)
                    yb = y[idx]
                    sb = s[idx]
                    for m in metrics:
                        per_fold_draws[f][m].append(_safe_metric(m, yb, sb))
            for b in range(n_bootstraps):
                for m in metrics:
                    vals = [per_fold_draws[f][m][b] for f in fold_ids]
                    long_records.append(
                        {
                            "classifier": clf,
                            "class": cls,
                            "metric": m,
                            "bootstrap_idx": b,
                            "value": float(np.nanmean(vals)),
                        }
                    )
        else:
            raise ValueError(f"Unsupported bootstrap mode: {mode}")

    return BootstrapResult(
        long_df=pd.DataFrame.from_records(long_records),
        point_estimates=pd.DataFrame.from_records(point_records),
        jackknife=pd.DataFrame.from_records(jk_records),
    )


def bootstrap_categorical_metric_cis(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    id_to_category: dict[str, str],
    *,
    category_name: str,
    categories: list[str] | None = None,
    negative_label: str = "Unknown",
    metrics: Iterable[str] = ("ap", "roc_auc"),
    n_bootstraps: int = 1000,
    seed: int = 42,
    mode: BootstrapMode = "cluster",
    progress: bool = True,
) -> BootstrapResult:
    """Per-category bootstrap.

    For each category ``C``, the metric is computed on the slice where
    ``id_to_category[ID] == C`` — i.e. only sequences that actually belong
    to that category. Sequences mapped to ``negative_label`` (and any other
    category) are excluded. ``negative_label`` is retained only as the
    sentinel that filters the category list when ``categories`` is left
    unspecified. When the category slice contains only positives or only
    negatives for a class, ``_safe_metric`` returns NaN and the cell is
    skipped at plotting time.
    """
    metrics = tuple(metrics)
    long_records: list[dict] = []
    point_records: list[dict] = []
    jk_records: list[dict] = []

    items = [
        (clf, cls, fold_dfs)
        for clf, cls_map in classifier_to_class_to_fold_dfs.items()
        for cls, fold_dfs in cls_map.items()
    ]
    iterator = (
        tqdm(items, desc=f"Bootstrap[{category_name}]", unit="(clf,cls)")
        if progress
        else items
    )

    for clf, cls, fold_dfs in iterator:
        arrays = _gather_arrays(fold_dfs, cls)
        fold_ids = sorted(arrays)
        n_folds = len(fold_ids)
        # Per-fold per-row category labels, aligned with arrays[f].
        fold_to_cats: dict[int, np.ndarray] = {}
        for f, (lab, _) in fold_dfs.items():
            ids = lab["ID"].astype(str).to_numpy()
            fold_to_cats[f] = np.array(
                [id_to_category.get(i, negative_label) for i in ids],
                dtype=object,
            )

        # Determine the effective category set.
        if categories is None:
            seen = sorted(
                {
                    c
                    for cats in fold_to_cats.values()
                    for c in np.unique(cats)
                    if c != negative_label
                }
            )
            cats_to_eval = seen
        else:
            cats_to_eval = list(categories)

        # Point estimates and jackknife on the full (or fold-leave-out) pool,
        # masked to each category subset.
        full_y = np.concatenate([arrays[f][0] for f in fold_ids])
        full_s = np.concatenate([arrays[f][1] for f in fold_ids])
        full_c = np.concatenate([fold_to_cats[f] for f in fold_ids])
        for category in cats_to_eval:
            mask = full_c == category
            for m in metrics:
                point_records.append(
                    {
                        "classifier": clf,
                        "class": cls,
                        "metric": m,
                        "category": category,
                        "value": _safe_metric(m, full_y[mask], full_s[mask])
                        if mask.any()
                        else float("nan"),
                    }
                )
        for f_left in fold_ids:
            kept = [f for f in fold_ids if f != f_left]
            jk_y = np.concatenate([arrays[f][0] for f in kept])
            jk_s = np.concatenate([arrays[f][1] for f in kept])
            jk_c = np.concatenate([fold_to_cats[f] for f in kept])
            for category in cats_to_eval:
                mask = jk_c == category
                for m in metrics:
                    jk_records.append(
                        {
                            "classifier": clf,
                            "class": cls,
                            "metric": m,
                            "category": category,
                            "fold_left_out": f_left,
                            "value": _safe_metric(m, jk_y[mask], jk_s[mask])
                            if mask.any()
                            else float("nan"),
                        }
                    )

        if mode == "cluster":
            rng = np.random.default_rng(seed)
            for b in range(n_bootstraps):
                picks = rng.choice(fold_ids, size=n_folds, replace=True)
                ys, ss, cs = [], [], []
                for f in picks:
                    y, s = arrays[f]
                    cats = fold_to_cats[f]
                    idx = rng.integers(0, len(y), size=len(y))
                    ys.append(y[idx])
                    ss.append(s[idx])
                    cs.append(cats[idx])
                yy = np.concatenate(ys)
                sc = np.concatenate(ss)
                cc = np.concatenate(cs)
                for category in cats_to_eval:
                    mask = cc == category
                    for m in metrics:
                        long_records.append(
                            {
                                "classifier": clf,
                                "class": cls,
                                "metric": m,
                                "category": category,
                                "bootstrap_idx": b,
                                "value": _safe_metric(m, yy[mask], sc[mask])
                                if mask.any()
                                else float("nan"),
                            }
                        )
        elif mode == "rows":
            per_fold_draws: dict[int, dict[str, dict[str, list[float]]]] = {
                f: {category: {m: [] for m in metrics} for category in cats_to_eval}
                for f in fold_ids
            }
            for f in fold_ids:
                rng_f = np.random.default_rng(seed)
                y, s = arrays[f]
                cats = fold_to_cats[f]
                n = len(y)
                for _ in range(n_bootstraps):
                    idx = rng_f.integers(0, n, size=n)
                    yb = y[idx]
                    sb = s[idx]
                    cb = cats[idx]
                    for category in cats_to_eval:
                        mask = cb == category
                        for m in metrics:
                            per_fold_draws[f][category][m].append(
                                _safe_metric(m, yb[mask], sb[mask])
                                if mask.any()
                                else float("nan")
                            )
            for b in range(n_bootstraps):
                for category in cats_to_eval:
                    for m in metrics:
                        vals = [per_fold_draws[f][category][m][b] for f in fold_ids]
                        long_records.append(
                            {
                                "classifier": clf,
                                "class": cls,
                                "metric": m,
                                "category": category,
                                "bootstrap_idx": b,
                                "value": float(np.nanmean(vals)),
                            }
                        )
        else:
            raise ValueError(f"Unsupported bootstrap mode: {mode}")

    return BootstrapResult(
        long_df=pd.DataFrame.from_records(long_records),
        point_estimates=pd.DataFrame.from_records(point_records),
        jackknife=pd.DataFrame.from_records(jk_records),
    )


def compute_cis(
    result: BootstrapResult,
    *,
    method: CiMethod = "percentile",
    ci: float = 0.95,
) -> pd.DataFrame:
    """Return point estimate + CI per (classifier, class, metric).

    ``percentile``: empirical alpha/2 and 1-alpha/2 quantiles of the
    bootstrap distribution.

    ``bca``: bias-corrected and accelerated. Bias term ``z0`` from the
    proportion of draws below the point estimate; acceleration ``a`` from
    the leave-one-fold-out jackknife.
    """
    alpha = 1 - ci
    base_keys = ["classifier", "class", "metric"]
    has_category = "category" in result.long_df.columns
    keys = [*base_keys, "category"] if has_category else base_keys
    points = result.point_estimates.set_index(keys)["value"]
    grouped = result.long_df.groupby(keys)["value"]
    out_records: list[dict] = []

    def _row(key, theta, lo, hi, method_label):
        record = dict(zip(keys, key)) if isinstance(key, tuple) else {keys[0]: key}
        record.update(
            {"point": theta, "ci_low": lo, "ci_high": hi, "method": method_label}
        )
        return record

    if method == "percentile":
        for key, draws in grouped:
            theta = float(points.loc[key])
            lo = float(np.nanquantile(draws, alpha / 2))
            hi = float(np.nanquantile(draws, 1 - alpha / 2))
            out_records.append(_row(key, theta, lo, hi, "percentile"))
        return pd.DataFrame.from_records(out_records)

    if method == "bca":
        from scipy.stats import norm  # type: ignore

        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)
        jk = result.jackknife
        for key, draws in grouped:
            theta = float(points.loc[key])
            b = np.asarray(draws.to_numpy(), dtype=np.float64)
            b = b[~np.isnan(b)]
            if b.size == 0:
                out_records.append(_row(key, theta, float("nan"), float("nan"), "bca"))
                continue
            prop_below = float(np.sum(b < theta)) / b.size
            prop_below = min(max(prop_below, 1e-9), 1 - 1e-9)
            z0 = norm.ppf(prop_below)
            mask = pd.Series(True, index=jk.index)
            for k_name, k_val in zip(keys, key if isinstance(key, tuple) else (key,)):
                mask &= jk[k_name] == k_val
            jk_vals = jk.loc[mask, "value"].to_numpy(dtype=np.float64)
            jk_vals = jk_vals[~np.isnan(jk_vals)]
            if jk_vals.size < 2:
                a = 0.0
            else:
                jk_mean = jk_vals.mean()
                num = float(np.sum((jk_mean - jk_vals) ** 3))
                den = 6.0 * float(np.sum((jk_mean - jk_vals) ** 2)) ** 1.5
                a = num / den if den > 0 else 0.0
            a1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
            a2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
            lo = float(np.nanquantile(b, a1))
            hi = float(np.nanquantile(b, a2))
            out_records.append(_row(key, theta, lo, hi, "bca"))
        return pd.DataFrame.from_records(out_records)

    raise ValueError(f"Unsupported CI method: {method}")
