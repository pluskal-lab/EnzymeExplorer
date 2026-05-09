"""Paired bootstrap utilities for the evaluation pipeline (v4).

Two AP statistics are produced from one shared pool of bootstrap draws:

* **Pooled OOF AP** — non-stratified row bootstrap of the concatenated
  out-of-fold predictions. One set of row indices per draw is shared
  across **every method and every class**. AP for class C / method m is
  computed on the same resampled rows as class D / method m'. Drop-and-
  redraw kicks in when the resample is degenerate (zero positives or
  zero negatives) for any requested class.
* **Mean fold AP** — fixed-fold stratified bootstrap. Per class C, per
  fold f, rows are resampled separately within
  (positives_for_C, negatives_for_C) strata. Per-fold AP is computed
  per method, then averaged across folds. Within a single (class, draw)
  cell the resampled indices are identical across methods; different
  classes use different strata so their resamples are independent.

In both cases the **point estimate** is the AP / fold-mean-AP computed
on the original (un-resampled) data — bootstrap is used only to
quantify uncertainty (CIs) and pairwise significance (deltas + p-values).

Per draw the function also computes pairwise deltas
``AP_a[draw] - AP_b[draw]`` for every method pair (or every
(target, other) pair when ``target_model`` is given), so the deltas are
exactly paired and downstream CIs collapse the within-pair variance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs

logger = logging.getLogger(__name__)

ApType = Literal["pooled_oof", "fold_mean"]
CiMethod = Literal["normal", "percentile", "bca"]
PvalueAdjustment = Literal["holm", "bonferroni", "none"]

ClassifierClassFoldDfs = dict[str, dict[str, dict[int, FoldDfs]]]

_AP_TYPES_DEFAULT: tuple[ApType, ...] = ("pooled_oof", "fold_mean")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Long-form bootstrap draws + point estimates + jackknife for AP/deltas.

    All five DataFrames carry the ``ap_type`` column (``pooled_oof`` or
    ``fold_mean``) so a single result holds both bootstrap variants. The
    jackknife tables (one per AP / delta) hold leave-one-fold-out
    re-computations of the *point* statistic and feed BCa CI computation
    in :func:`compute_cis`.
    """

    long_ap: pd.DataFrame  # classifier, class, metric, ap_type, bootstrap_idx, value
    point_ap: pd.DataFrame  # classifier, class, metric, ap_type, value
    long_delta: pd.DataFrame
    point_delta: pd.DataFrame
    jackknife_ap: pd.DataFrame  # classifier, class, metric, ap_type, fold_left_out, value
    jackknife_delta: pd.DataFrame  # classifier_a, classifier_b, ..., fold_left_out, value

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.long_ap.to_csv(out_dir / "bootstrap_long_ap.csv", index=False)
        self.point_ap.to_csv(out_dir / "point_estimates_ap.csv", index=False)
        self.long_delta.to_csv(out_dir / "bootstrap_long_delta.csv", index=False)
        self.point_delta.to_csv(out_dir / "point_estimates_delta.csv", index=False)
        self.jackknife_ap.to_csv(out_dir / "jackknife_ap.csv", index=False)
        self.jackknife_delta.to_csv(out_dir / "jackknife_delta.csv", index=False)

    @classmethod
    def load(cls, in_dir: Path) -> "BootstrapResult":
        return cls(
            long_ap=pd.read_csv(in_dir / "bootstrap_long_ap.csv"),
            point_ap=pd.read_csv(in_dir / "point_estimates_ap.csv"),
            long_delta=pd.read_csv(in_dir / "bootstrap_long_delta.csv"),
            point_delta=pd.read_csv(in_dir / "point_estimates_delta.csv"),
            jackknife_ap=pd.read_csv(in_dir / "jackknife_ap.csv")
                if (in_dir / "jackknife_ap.csv").exists() else pd.DataFrame(),
            jackknife_delta=pd.read_csv(in_dir / "jackknife_delta.csv")
                if (in_dir / "jackknife_delta.csv").exists() else pd.DataFrame(),
        )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _safe_metric(name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ``name`` on (y_true, y_score) returning NaN on degenerate input.

    AP and ROC-AUC are both undefined when ``y_true`` has no positives or
    no negatives. NaN is the "undefined" sentinel.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    if name == "ap":
        return float(average_precision_score(y_true, y_score))
    if name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    raise ValueError(f"Unsupported metric: {name}")


def _gather_class_arrays(
    fold_dfs: dict[int, FoldDfs], cls: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per fold: (labels[cls] as int8, preds[cls] as float64)."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f, (lab, pred) in fold_dfs.items():
        out[f] = (
            np.asarray(lab[cls].to_numpy(), dtype=np.int8),
            np.asarray(pred[cls].to_numpy(), dtype=np.float64),
        )
    return out


def _pooled_arrays(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    classes: list[str],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Concatenate every fold's labels / preds per (classifier, class).

    Returns ``{classifier: {class: (y, s)}}`` where ``y`` and ``s`` are
    1-D arrays indexed identically to the OOF row order. Folds are
    concatenated in sorted order so the row index space is consistent.
    Labels are taken from the first classifier seen (they're identical
    across classifiers for a given fold by construction); the fold
    sizes are also asserted equal across classifiers.
    """
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fold_sizes_canon: list[int] | None = None
    for clf, class_to_dfs in classifier_to_class_to_fold_dfs.items():
        out[clf] = {}
        sizes_this = None
        for cls in classes:
            if cls not in class_to_dfs:
                continue
            fd = class_to_dfs[cls]
            ys = []
            ss = []
            sizes = []
            for f in sorted(fd):
                lab, pred = fd[f]
                ys.append(np.asarray(lab[cls].to_numpy(), dtype=np.int8))
                ss.append(np.asarray(pred[cls].to_numpy(), dtype=np.float64))
                sizes.append(len(lab))
            out[clf][cls] = (np.concatenate(ys), np.concatenate(ss))
            if sizes_this is None:
                sizes_this = sizes
            elif sizes_this != sizes:
                raise ValueError(
                    f"Fold sizes for {clf}/{cls} differ from earlier classes: "
                    f"{sizes_this} vs {sizes}. Pooled OOF needs aligned rows."
                )
        if fold_sizes_canon is None:
            fold_sizes_canon = sizes_this
        elif sizes_this is not None and fold_sizes_canon != sizes_this:
            raise ValueError(
                f"Fold sizes for classifier {clf} differ from the canonical "
                f"sizes {fold_sizes_canon}: {sizes_this}. Pooled OOF requires "
                f"all methods to share the same fold/row layout."
            )
    return out


def _strata_per_class(
    fold_dfs: dict[int, FoldDfs], cls: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per fold: (positive_idx, negative_idx) for class ``cls``."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f, (lab, _) in fold_dfs.items():
        y = lab[cls].to_numpy().astype(bool)
        out[f] = (np.flatnonzero(y), np.flatnonzero(~y))
    return out


def _stratified_resample(
    pos_idx: np.ndarray, neg_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap one fold for one class — strata sampled independently."""
    parts = []
    if pos_idx.size:
        parts.append(rng.choice(pos_idx, size=pos_idx.size, replace=True))
    if neg_idx.size:
        parts.append(rng.choice(neg_idx, size=neg_idx.size, replace=True))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Pooled OOF bootstrap (one shared bootstrap across all classes & methods)
# ---------------------------------------------------------------------------


def _bootstrap_pooled_oof(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    classifiers: list[str],
    classes: list[str],
    metrics: tuple[str, ...],
    n_bootstraps: int,
    seed: int,
    progress: bool,
) -> tuple[
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], float],
]:
    """Pooled OOF bootstrap with shared row indices across methods & classes.

    Returns (per_method_class_metric_draws, per_method_class_metric_point).
    """
    pooled = _pooled_arrays(classifier_to_class_to_fold_dfs, classes)
    n_total = len(next(iter(next(iter(pooled.values())).values()))[0])

    # Determine which (classifier, class) cells are populated.
    populated_cells: dict[str, list[str]] = {
        clf: [c for c in classes if c in pooled[clf]] for clf in classifiers
    }

    # POINT estimates: AP / ROC-AUC on the un-resampled pool.
    point: dict[tuple[str, str, str], float] = {}
    for clf in classifiers:
        for cls in populated_cells[clf]:
            y, s = pooled[clf][cls]
            for m in metrics:
                point[(clf, cls, m)] = _safe_metric(m, y, s)

    # Bootstrap loop.
    rng = np.random.default_rng(seed)
    # Storage: for each (classifier, class, metric), a vector of size n_bootstraps.
    draws: dict[tuple[str, str, str], np.ndarray] = {}
    for clf in classifiers:
        for cls in populated_cells[clf]:
            for m in metrics:
                draws[(clf, cls, m)] = np.empty(n_bootstraps, dtype=np.float64)

    # We need a degeneracy check: any class with 0 pos or 0 neg in the
    # resample → drop-and-redraw. We only need to check ONE method's
    # labels per class (labels are shared across methods).
    canon_clf = classifiers[0]

    survived = 0
    attempts = 0
    iterator = (
        tqdm(total=n_bootstraps, desc="bootstrap[pooled_oof]", unit="draw")
        if progress
        else None
    )
    while survived < n_bootstraps:
        attempts += 1
        idx = rng.integers(0, n_total, size=n_total)
        # Degeneracy check.
        bad = False
        for cls in classes:
            if cls not in pooled[canon_clf]:
                continue
            y_b = pooled[canon_clf][cls][0][idx]
            s_pos = int(y_b.sum())
            if s_pos == 0 or s_pos == len(y_b):
                bad = True
                break
        if bad:
            if attempts > n_bootstraps * 50:
                raise RuntimeError(
                    "pooled-OOF bootstrap exhausted retries (>50× attempts); "
                    "some class has zero positives or zero negatives in too "
                    "many resamples."
                )
            continue
        # Score every (classifier, class, metric) on the same resampled rows.
        for clf in classifiers:
            for cls in populated_cells[clf]:
                y_clf, s_clf = pooled[clf][cls]
                yb = y_clf[idx]
                sb = s_clf[idx]
                for m in metrics:
                    draws[(clf, cls, m)][survived] = _safe_metric(m, yb, sb)
        survived += 1
        if iterator is not None:
            iterator.update(1)
    if iterator is not None:
        iterator.close()
    if attempts > survived:
        logger.info(
            "pooled_oof: %d / %d draws survived (%.1f%% drop)",
            survived, attempts, 100 * (attempts - survived) / attempts,
        )
    return draws, point


# ---------------------------------------------------------------------------
# Mean-fold bootstrap (per-class stratified, shared draws across methods)
# ---------------------------------------------------------------------------


def _bootstrap_mean_fold(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    classifiers: list[str],
    classes: list[str],
    metrics: tuple[str, ...],
    n_bootstraps: int,
    seed: int,
    progress: bool,
) -> tuple[
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], float],
]:
    """Per-class fold-mean bootstrap. Strata: positives_for_C vs negatives_for_C.

    The same draw index ``b`` produces the *same* per-fold resample
    indices for every method (so deltas are paired), but a *different*
    resample for every class (since strata depend on the class label).
    """
    # Pre-fetch arrays + strata.
    fold_arrays: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
    fold_strata: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    populated_cells: dict[str, list[str]] = {clf: [] for clf in classifiers}
    for clf, class_to_dfs in classifier_to_class_to_fold_dfs.items():
        fold_arrays[clf] = {}
        for cls in classes:
            if cls not in class_to_dfs:
                continue
            fold_arrays[clf][cls] = _gather_class_arrays(class_to_dfs[cls], cls)
            populated_cells[clf].append(cls)

    # The strata are method-independent (they depend on labels only,
    # which all methods share for a given fold), so build them once.
    canon_clf = classifiers[0]
    for cls in classes:
        if cls not in classifier_to_class_to_fold_dfs[canon_clf]:
            continue
        fold_strata[cls] = _strata_per_class(
            classifier_to_class_to_fold_dfs[canon_clf][cls], cls
        )

    # POINT estimates: fold-mean AP on un-resampled per-fold arrays.
    point: dict[tuple[str, str, str], float] = {}
    for clf in classifiers:
        for cls in populated_cells[clf]:
            for m in metrics:
                fold_vals = []
                for f, (y, s) in fold_arrays[clf][cls].items():
                    fold_vals.append(_safe_metric(m, y, s))
                point[(clf, cls, m)] = float(np.nanmean(fold_vals))

    # Bootstrap loop: per class, share a single seeded RNG so every
    # method sees the same fold-level resample indices.
    draws: dict[tuple[str, str, str], np.ndarray] = {}
    for clf in classifiers:
        for cls in populated_cells[clf]:
            for m in metrics:
                draws[(clf, cls, m)] = np.empty(n_bootstraps, dtype=np.float64)

    cls_iter = (
        tqdm(classes, desc="bootstrap[fold_mean]", unit="class")
        if progress
        else classes
    )
    for cls in cls_iter:
        if cls not in fold_strata:
            continue
        rng = np.random.default_rng(seed)  # per-class deterministic RNG
        fold_ids = sorted(fold_strata[cls])
        # For each draw, draw all folds' indices once, score every method.
        for b in range(n_bootstraps):
            fold_idx_resamples = {
                f: _stratified_resample(*fold_strata[cls][f], rng) for f in fold_ids
            }
            for clf in classifiers:
                if cls not in fold_arrays[clf]:
                    continue
                for m in metrics:
                    fold_metric_vals = []
                    for f in fold_ids:
                        y_clf, s_clf = fold_arrays[clf][cls][f]
                        idx = fold_idx_resamples[f]
                        fold_metric_vals.append(
                            _safe_metric(m, y_clf[idx], s_clf[idx])
                        )
                    draws[(clf, cls, m)][b] = float(np.nanmean(fold_metric_vals))
    return draws, point


# ---------------------------------------------------------------------------
# Long-form assembly + delta tables
# ---------------------------------------------------------------------------


def _draws_to_long(
    draws: dict[tuple[str, str, str], np.ndarray],
    *,
    ap_type: ApType,
) -> pd.DataFrame:
    rows = []
    for (clf, cls, metric), values in draws.items():
        for b, v in enumerate(values):
            rows.append(
                {
                    "classifier": clf,
                    "class": cls,
                    "metric": metric,
                    "ap_type": ap_type,
                    "bootstrap_idx": b,
                    "value": float(v),
                }
            )
    return pd.DataFrame.from_records(rows)


def _point_to_df(
    point: dict[tuple[str, str, str], float],
    *,
    ap_type: ApType,
) -> pd.DataFrame:
    rows = []
    for (clf, cls, metric), v in point.items():
        rows.append(
            {
                "classifier": clf,
                "class": cls,
                "metric": metric,
                "ap_type": ap_type,
                "value": float(v),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_delta_tables(
    draws_by_ap: dict[ApType, dict[tuple[str, str, str], np.ndarray]],
    point_by_ap: dict[ApType, dict[tuple[str, str, str], float]],
    classifiers: list[str],
    classes: list[str],
    metrics: tuple[str, ...],
    target_model: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form and point delta tables (paired across methods).

    With ``target_model`` set, only ``(target, other)`` pairs are
    produced (one comparison per non-target method per class per metric)
    and the target is always ``classifier_a`` so deltas read
    "target − other".

    Without ``target_model``, every C(N, 2) pair is enumerated and
    orientation is fixed **globally per pair** (``classifier_a`` =
    classifier with the higher fold-mean ``Overall`` AP averaged across
    classes; ties broken alphabetically). One global orientation per
    unordered pair is critical for the aggregate-row build downstream:
    if a pair flipped orientation per class, the
    ``_classifiers_covering`` check in ``aggregate.py`` would reject
    it because no single ordered pair would cover every member class
    — the practical symptom is "Substrate_mAP only shows one pair".
    """
    if target_model is not None and target_model not in classifiers:
        raise ValueError(
            f"target_model={target_model!r} is not among classifiers "
            f"{classifiers}"
        )

    # Decide pair list and a→b ordering.
    pairs: list[tuple[str, str]] = []
    if target_model is not None:
        for other in classifiers:
            if other == target_model:
                continue
            pairs.append((target_model, other))
    else:
        # Compute a global "score" per classifier for orientation:
        # mean of point AP across (classes, metrics, ap_types) using the
        # canonical metric (``ap``) only, restricted to the primary
        # ap_type in ``draws_by_ap`` to keep the choice deterministic.
        primary_ap_type = next(iter(draws_by_ap))
        clf_score: dict[str, float] = {}
        for clf in classifiers:
            vals = [
                point_by_ap[primary_ap_type][(clf, cls, "ap")]
                for cls in classes
                if (clf, cls, "ap") in point_by_ap[primary_ap_type]
            ]
            clf_score[clf] = float(np.nanmean(vals)) if vals else 0.0
        for a, b in combinations(classifiers, 2):
            # Orient so classifier_a has the higher overall score; tie-
            # break alphabetically to keep orientation deterministic.
            if (clf_score[a], a) >= (clf_score[b], b):
                pairs.append((a, b))
            else:
                pairs.append((b, a))

    long_rows: list[dict] = []
    point_rows: list[dict] = []
    for ap_type, draws in draws_by_ap.items():
        for cls in classes:
            for m in metrics:
                for a, b in pairs:
                    if (a, cls, m) not in draws or (b, cls, m) not in draws:
                        continue
                    da = draws[(a, cls, m)]
                    db = draws[(b, cls, m)]
                    delta = da - db
                    pa = point_by_ap[ap_type].get((a, cls, m))
                    pb = point_by_ap[ap_type].get((b, cls, m))
                    if pa is None or pb is None:
                        continue
                    point_rows.append(
                        {
                            "classifier_a": a,
                            "classifier_b": b,
                            "class": cls,
                            "metric": m,
                            "ap_type": ap_type,
                            "value": float(pa - pb),
                        }
                    )
                    for bidx, v in enumerate(delta):
                        long_rows.append(
                            {
                                "classifier_a": a,
                                "classifier_b": b,
                                "class": cls,
                                "metric": m,
                                "ap_type": ap_type,
                                "bootstrap_idx": bidx,
                                "value": float(v),
                            }
                        )
    return (
        pd.DataFrame.from_records(long_rows),
        pd.DataFrame.from_records(point_rows),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _compute_jackknife_ap(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    classifiers: list[str],
    classes: list[str],
    metrics: tuple[str, ...],
    ap_types: tuple[ApType, ...],
) -> pd.DataFrame:
    """Leave-one-fold-out jackknife of the AP point statistic.

    Returns a long-form ``classifier, class, metric, ap_type,
    fold_left_out, value`` table. For ``fold_mean`` the value is the
    mean of per-fold APs over the kept folds; for ``pooled_oof`` it is
    AP on the row-pool excluding the left-out fold.
    """
    rows: list[dict] = []
    for clf in classifiers:
        class_to_dfs = classifier_to_class_to_fold_dfs[clf]
        for cls in classes:
            if cls not in class_to_dfs:
                continue
            fold_dfs = class_to_dfs[cls]
            fold_ids = sorted(fold_dfs)
            arrays = _gather_class_arrays(fold_dfs, cls)
            for f_left in fold_ids:
                kept = [f for f in fold_ids if f != f_left]
                # pooled OOF on kept folds
                if "pooled_oof" in ap_types:
                    yk = np.concatenate([arrays[f][0] for f in kept])
                    sk = np.concatenate([arrays[f][1] for f in kept])
                    for m in metrics:
                        rows.append({
                            "classifier": clf, "class": cls, "metric": m,
                            "ap_type": "pooled_oof", "fold_left_out": f_left,
                            "value": _safe_metric(m, yk, sk),
                        })
                if "fold_mean" in ap_types:
                    for m in metrics:
                        per = [_safe_metric(m, *arrays[f]) for f in kept]
                        rows.append({
                            "classifier": clf, "class": cls, "metric": m,
                            "ap_type": "fold_mean", "fold_left_out": f_left,
                            "value": float(np.nanmean(per)),
                        })
    return pd.DataFrame.from_records(rows)


def _build_jackknife_delta(
    jk_ap: pd.DataFrame, point_delta: pd.DataFrame,
) -> pd.DataFrame:
    """Per (pair, class, metric, ap_type, fold_left_out): difference of
    the leave-one-out AP values."""
    if jk_ap.empty or point_delta.empty:
        return pd.DataFrame()
    pairs = point_delta[["classifier_a", "classifier_b"]].drop_duplicates()
    rows: list[dict] = []
    for (cls, metric, ap_type), grp in jk_ap.groupby(["class", "metric", "ap_type"]):
        # Map (classifier, fold_left_out) → value for this family.
        cmap = {(r["classifier"], r["fold_left_out"]): r["value"] for _, r in grp.iterrows()}
        for _, r in pairs.iterrows():
            a, b = r["classifier_a"], r["classifier_b"]
            for f in sorted({k[1] for k in cmap}):
                if (a, f) not in cmap or (b, f) not in cmap:
                    continue
                rows.append({
                    "classifier_a": a, "classifier_b": b,
                    "class": cls, "metric": metric, "ap_type": ap_type,
                    "fold_left_out": f,
                    "value": float(cmap[(a, f)] - cmap[(b, f)]),
                })
    return pd.DataFrame.from_records(rows)


def paired_bootstrap_metric_cis(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    *,
    metrics: Iterable[str] = ("ap", "roc_auc"),
    ap_types: Iterable[ApType] = _AP_TYPES_DEFAULT,
    n_bootstraps: int = 1000,
    seed: int = 42,
    target_model: str | None = None,
    progress: bool = True,
) -> BootstrapResult:
    """Run the v4 paired bootstrap (pooled OOF + mean-fold) in one pass.

    Same draw indices are applied to every method for a given AP type
    so the returned ``long_delta`` table contains exactly-paired
    differences (within-pair variance is collapsed).
    """
    classifiers = list(classifier_to_class_to_fold_dfs)
    classes = sorted(
        {
            cls
            for class_to_dfs in classifier_to_class_to_fold_dfs.values()
            for cls in class_to_dfs
        }
    )
    metrics = tuple(metrics)
    ap_types = tuple(ap_types)

    draws_by_ap: dict[ApType, dict[tuple[str, str, str], np.ndarray]] = {}
    point_by_ap: dict[ApType, dict[tuple[str, str, str], float]] = {}

    if "pooled_oof" in ap_types:
        d, p = _bootstrap_pooled_oof(
            classifier_to_class_to_fold_dfs, classifiers, classes, metrics,
            n_bootstraps, seed, progress,
        )
        draws_by_ap["pooled_oof"] = d
        point_by_ap["pooled_oof"] = p
    if "fold_mean" in ap_types:
        d, p = _bootstrap_mean_fold(
            classifier_to_class_to_fold_dfs, classifiers, classes, metrics,
            n_bootstraps, seed, progress,
        )
        draws_by_ap["fold_mean"] = d
        point_by_ap["fold_mean"] = p

    long_ap = pd.concat(
        [_draws_to_long(d, ap_type=at) for at, d in draws_by_ap.items()],
        ignore_index=True,
    ) if draws_by_ap else pd.DataFrame()
    point_ap = pd.concat(
        [_point_to_df(p, ap_type=at) for at, p in point_by_ap.items()],
        ignore_index=True,
    ) if point_by_ap else pd.DataFrame()

    long_delta, point_delta = _build_delta_tables(
        draws_by_ap, point_by_ap, classifiers, classes, metrics, target_model,
    )

    jackknife_ap = _compute_jackknife_ap(
        classifier_to_class_to_fold_dfs, classifiers, classes, metrics, ap_types,
    )
    jackknife_delta = _build_jackknife_delta(jackknife_ap, point_delta)

    return BootstrapResult(
        long_ap=long_ap,
        point_ap=point_ap,
        long_delta=long_delta,
        point_delta=point_delta,
        jackknife_ap=jackknife_ap,
        jackknife_delta=jackknife_delta,
    )


# ---------------------------------------------------------------------------
# Confidence intervals (works on either AP or delta long tables)
# ---------------------------------------------------------------------------


_AP_GROUP_KEYS = ("classifier", "class", "metric", "ap_type")
_DELTA_GROUP_KEYS = ("classifier_a", "classifier_b", "class", "metric", "ap_type")


def compute_cis(
    long_df: pd.DataFrame,
    point_df: pd.DataFrame,
    *,
    method: CiMethod = "normal",
    ci: float = 0.95,
    jackknife: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute (point, ci_low, ci_high) per group.

    ``long_df`` is a long-form bootstrap table with a ``value`` column
    and either AP-style group keys (``classifier, class, metric,
    ap_type``) or delta-style group keys (``classifier_a,
    classifier_b, class, metric, ap_type``). The function auto-detects
    the schema.

    Method options:

    * ``normal``: ``point ± z · SD(draws)`` (Wald-style).
    * ``percentile``: empirical α/2 and 1-α/2 quantiles of draws.
    * ``bca``: bias-corrected and accelerated; requires ``jackknife``.
    """
    is_delta = "classifier_a" in long_df.columns
    keys = list(_DELTA_GROUP_KEYS if is_delta else _AP_GROUP_KEYS)
    points = point_df.set_index(keys)["value"]
    grouped = long_df.groupby(list(keys))["value"]

    alpha = 1 - ci
    out: list[dict] = []

    def _row(key, theta, lo, hi, method_label):
        record = dict(zip(keys, key)) if isinstance(key, tuple) else {keys[0]: key}
        record.update(
            {
                "point": float(theta),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "method": method_label,
            }
        )
        return record

    if method == "normal":
        from scipy.stats import norm  # type: ignore

        z = float(norm.ppf(1 - alpha / 2))
        for key, draws in grouped:
            theta = float(points.loc[key])
            b = np.asarray(draws.to_numpy(), dtype=np.float64)
            b = b[~np.isnan(b)]
            if b.size < 2:
                out.append(_row(key, theta, float("nan"), float("nan"), "normal"))
                continue
            sd = float(np.std(b, ddof=1))
            out.append(_row(key, theta, theta - z * sd, theta + z * sd, "normal"))
        return pd.DataFrame.from_records(out)

    if method == "percentile":
        for key, draws in grouped:
            theta = float(points.loc[key])
            b = draws.dropna().to_numpy()
            if b.size == 0:
                out.append(_row(key, theta, float("nan"), float("nan"), "percentile"))
                continue
            lo = float(np.quantile(b, alpha / 2))
            hi = float(np.quantile(b, 1 - alpha / 2))
            out.append(_row(key, theta, lo, hi, "percentile"))
        return pd.DataFrame.from_records(out)

    if method == "bca":
        from scipy.stats import norm  # type: ignore

        z_lo = float(norm.ppf(alpha / 2))
        z_hi = float(norm.ppf(1 - alpha / 2))
        if jackknife is None or jackknife.empty:
            raise ValueError(
                "BCa CI requires a non-empty jackknife table. The v4 "
                "bootstrap doesn't emit one because the canonical "
                "fold-mean point estimate is leave-one-out reproducible "
                "without a separate jackknife pass; pass a jackknife "
                "DataFrame produced by callers if BCa is needed."
            )
        for key, draws in grouped:
            theta = float(points.loc[key])
            b = draws.dropna().to_numpy()
            if b.size == 0:
                out.append(_row(key, theta, float("nan"), float("nan"), "bca"))
                continue
            prop_below = float(np.sum(b < theta)) / b.size
            prop_below = min(max(prop_below, 1e-9), 1 - 1e-9)
            z0 = float(norm.ppf(prop_below))
            mask = pd.Series(True, index=jackknife.index)
            for k_name, k_val in zip(keys, key if isinstance(key, tuple) else (key,)):
                mask &= jackknife[k_name] == k_val
            jk_vals = jackknife.loc[mask, "value"].to_numpy(dtype=np.float64)
            jk_vals = jk_vals[~np.isnan(jk_vals)]
            if jk_vals.size < 2:
                a = 0.0
            else:
                jk_mean = jk_vals.mean()
                num = float(np.sum((jk_mean - jk_vals) ** 3))
                den = 6.0 * float(np.sum((jk_mean - jk_vals) ** 2)) ** 1.5
                a = num / den if den > 0 else 0.0
            a1 = float(norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
            a2 = float(norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))
            lo = float(np.quantile(b, a1))
            hi = float(np.quantile(b, a2))
            out.append(_row(key, theta, lo, hi, "bca"))
        return pd.DataFrame.from_records(out)

    raise ValueError(f"Unsupported CI method: {method}")
