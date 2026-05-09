"""Per-category AP computation that re-uses the v4 paired bootstrap.

Two paths, both producing the same long-form schema
(``classifier, class, metric, ap_type, category_name, category,
bootstrap_idx, value``) so the existing
:func:`plotting.categorical.plot_category_boxplot` can render them
without changes:

* :func:`compute_type_aggregated_ap` — Type-grouped substrate mAP. No
  bootstrap is rerun: it aggregates the per-substrate draws that already
  live in ``bootstrap_long_ap.csv`` by mapping each substrate to its
  Type and averaging APs within each ``(classifier, ap_type, draw,
  Type)`` cell.
* :func:`compute_masked_ap_from_seed` — row-masked AP (e.g. per-Kingdom
  TPS / Substrate). This *does* re-run the AP scoring, but it replays
  the same RNG sequence used by :func:`bootstrap.paired_bootstrap_metric_cis`
  so the resampled row indices are bit-identical to the cached main
  draws — no second bootstrap, only a second AP-evaluation pass under
  the per-category mask.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from enzymeexplorer.src.evaluation.bootstrap import (
    ApType,
    ClassifierClassFoldDfs,
    _gather_class_arrays,
    _pooled_arrays,
    _safe_metric,
    _strata_per_class,
    _stratified_resample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-grouped aggregation (post-process existing draws)
# ---------------------------------------------------------------------------


def compute_type_aggregated_ap(
    long_ap: pd.DataFrame,
    point_ap: pd.DataFrame,
    *,
    type_groupings: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-class AP draws into per-Type mAP draws.

    ``type_groupings`` maps each category name (e.g. ``"TPS_Type"``) to a
    ``{class_short: type_label}`` mapping. The output long table has
    ``class = "_type_grouped"`` and one row per
    ``(classifier, ap_type, metric, category_name, category, bootstrap_idx)``,
    with ``value`` = mean AP across the substrates of that Type for
    that draw. The point table is the analogous mean of point estimates.

    If ``long_ap`` is empty the returned long table is empty too;
    likewise for ``point_ap``.
    """
    long_rows: list[pd.DataFrame] = []
    point_rows: list[pd.DataFrame] = []

    for cat_name, class_to_type in type_groupings.items():
        if not class_to_type:
            continue
        if not long_ap.empty:
            sub = long_ap[long_ap["class"].isin(class_to_type)].copy()
            if not sub.empty:
                sub["category"] = sub["class"].map(class_to_type)
                agg = sub.groupby(
                    ["classifier", "ap_type", "metric", "category", "bootstrap_idx"],
                    as_index=False,
                )["value"].mean()
                agg["class"] = "_type_grouped"
                agg["category_name"] = cat_name
                long_rows.append(agg)
        if not point_ap.empty:
            sub = point_ap[point_ap["class"].isin(class_to_type)].copy()
            if not sub.empty:
                sub["category"] = sub["class"].map(class_to_type)
                agg = sub.groupby(
                    ["classifier", "ap_type", "metric", "category"],
                    as_index=False,
                )["value"].mean()
                agg["class"] = "_type_grouped"
                agg["category_name"] = cat_name
                point_rows.append(agg)

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=[
            "classifier", "class", "metric", "ap_type",
            "category_name", "category", "bootstrap_idx", "value",
        ]
    )
    point_df = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame(
        columns=[
            "classifier", "class", "metric", "ap_type",
            "category_name", "category", "value",
        ]
    )
    return long_df, point_df


# ---------------------------------------------------------------------------
# Row-masked replay (re-evaluate the metric on cached resampled rows)
# ---------------------------------------------------------------------------


def _replay_pooled_oof_indices(
    canon_y_per_class: dict[str, np.ndarray],
    classes: list[str],
    n_total: int,
    n_bootstraps: int,
    seed: int,
) -> list[np.ndarray]:
    """Re-derive the same row indices `_bootstrap_pooled_oof` produced.

    The drop-and-redraw rule **and** the iteration order
    (``for cls in classes``) must match the original implementation
    byte-for-byte; otherwise the replay drifts off the cached draws.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    survived = 0
    attempts = 0
    while survived < n_bootstraps:
        attempts += 1
        idx = rng.integers(0, n_total, size=n_total)
        bad = False
        for cls in classes:
            if cls not in canon_y_per_class:
                continue
            yb = canon_y_per_class[cls][idx]
            s_pos = int(yb.sum())
            if s_pos == 0 or s_pos == len(yb):
                bad = True
                break
        if bad:
            if attempts > n_bootstraps * 50:
                raise RuntimeError(
                    "pooled-OOF replay exhausted retries; the seed/data "
                    "must match the main bootstrap exactly."
                )
            continue
        out.append(idx)
        survived += 1
    return out


def _row_category_assignment(
    ids: np.ndarray, id_to_cat: dict[str, str], negative_label: str = "Unknown",
) -> np.ndarray:
    """Map an ID array to its category labels, defaulting to
    ``negative_label`` for IDs absent from the lookup."""
    return np.array(
        [id_to_cat.get(str(i), negative_label) for i in ids],
        dtype=object,
    )


def compute_masked_ap_from_seed(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
    *,
    masked_categories: dict[str, dict[str, str]],
    metrics: Iterable[str] = ("ap",),
    ap_types: Iterable[ApType] = ("pooled_oof", "fold_mean"),
    n_bootstraps: int = 1000,
    seed: int = 42,
    id_column: str = "ID",
    negative_labels: dict[str, str] | None = None,
    min_rows: int = 5,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-(category, classifier, class) AP draws by replaying
    the v4 paired-bootstrap RNG and applying a row mask.

    ``masked_categories`` maps category-name → ``{ID: category_value}``.
    For each ``cat_name``, every draw produced by the main bootstrap is
    re-scored on the subset of resampled rows whose ID maps to a
    given non-negative category. Resamples whose masked subset has
    fewer than ``min_rows`` valid rows or no positives are skipped (NaN
    is *not* recorded — the cell is just omitted).

    Returns two long-form dataframes with the schema
    ``classifier, class, metric, ap_type, category_name, category,
    bootstrap_idx, value`` (and the point variant without
    ``bootstrap_idx``). The point estimate is the metric on the
    *un-resampled* category-masked rows — same convention as the main
    bootstrap.
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
    negative_labels = negative_labels or {}
    canon_clf = classifiers[0]

    long_rows: list[dict] = []
    point_rows: list[dict] = []

    # ----- Pooled OOF replay -----
    if "pooled_oof" in ap_types:
        pooled = _pooled_arrays(classifier_to_class_to_fold_dfs, classes)
        # Per (clf, cls) IDs in the same row order as labels/preds.
        pooled_ids: dict[str, dict[str, np.ndarray]] = {}
        for clf, class_to_dfs in classifier_to_class_to_fold_dfs.items():
            pooled_ids[clf] = {}
            for cls in classes:
                if cls not in class_to_dfs:
                    continue
                fd = class_to_dfs[cls]
                ids = []
                for f in sorted(fd):
                    lab, _ = fd[f]
                    ids.append(lab[id_column].astype(str).to_numpy())
                pooled_ids[clf][cls] = np.concatenate(ids)

        n_total = len(next(iter(pooled[canon_clf].values()))[0])
        canon_y_per_class = {cls: pooled[canon_clf][cls][0] for cls in pooled[canon_clf]}
        idx_per_draw = _replay_pooled_oof_indices(
            canon_y_per_class, classes, n_total, n_bootstraps, seed,
        )

        for cat_name, id_to_cat in masked_categories.items():
            neg = negative_labels.get(cat_name, "Unknown")
            # IDs are aligned across (clf, cls) within pooled_oof, so the
            # category vector can be computed once from the canonical pool.
            canon_cls = next(iter(pooled_ids[canon_clf]))
            canon_ids = pooled_ids[canon_clf][canon_cls]
            row_cat = _row_category_assignment(canon_ids, id_to_cat, neg)
            cat_values = sorted(set(row_cat) - {neg})

            iterator = (
                tqdm(idx_per_draw, desc=f"pooled_oof[{cat_name}]", unit="draw")
                if progress else idx_per_draw
            )
            for b, idx in enumerate(iterator):
                row_cat_b = row_cat[idx]
                for cat_v in cat_values:
                    mask = row_cat_b == cat_v
                    if mask.sum() < min_rows:
                        continue
                    for clf in classifiers:
                        for cls in pooled[clf]:
                            y, s = pooled[clf][cls]
                            yb = y[idx][mask]
                            sb = s[idx][mask]
                            for m in metrics:
                                v = _safe_metric(m, yb, sb)
                                if not np.isnan(v):
                                    long_rows.append({
                                        "classifier": clf, "class": cls,
                                        "metric": m, "ap_type": "pooled_oof",
                                        "category_name": cat_name,
                                        "category": cat_v,
                                        "bootstrap_idx": b,
                                        "value": float(v),
                                    })

            # POINT — un-resampled, masked.
            for clf in classifiers:
                for cls in pooled[clf]:
                    y, s = pooled[clf][cls]
                    ids = pooled_ids[clf][cls]
                    rcat = _row_category_assignment(ids, id_to_cat, neg)
                    for cat_v in cat_values:
                        mask = rcat == cat_v
                        if mask.sum() < min_rows:
                            continue
                        for m in metrics:
                            v = _safe_metric(m, y[mask], s[mask])
                            if not np.isnan(v):
                                point_rows.append({
                                    "classifier": clf, "class": cls,
                                    "metric": m, "ap_type": "pooled_oof",
                                    "category_name": cat_name,
                                    "category": cat_v,
                                    "value": float(v),
                                })

    # ----- Fold-mean replay -----
    if "fold_mean" in ap_types:
        # Per-class strata (same as `_bootstrap_mean_fold` derives).
        for cls in classes:
            if cls not in classifier_to_class_to_fold_dfs[canon_clf]:
                continue
            fold_strata = _strata_per_class(
                classifier_to_class_to_fold_dfs[canon_clf][cls], cls
            )
            fold_ids_sorted = sorted(fold_strata)

            # Per-fold (y, s) and IDs per classifier.
            fold_arrays: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
            fold_ids_array: dict[int, np.ndarray] = {}
            for clf in classifiers:
                if cls not in classifier_to_class_to_fold_dfs[clf]:
                    continue
                fd = classifier_to_class_to_fold_dfs[clf][cls]
                fold_arrays[clf] = _gather_class_arrays(fd, cls)
                if clf == canon_clf:
                    for f, (lab, _) in fd.items():
                        fold_ids_array[f] = lab[id_column].astype(str).to_numpy()

            for cat_name, id_to_cat in masked_categories.items():
                neg = negative_labels.get(cat_name, "Unknown")
                row_cat_per_fold = {
                    f: _row_category_assignment(fold_ids_array[f], id_to_cat, neg)
                    for f in fold_ids_sorted
                }
                cat_values = sorted(
                    {v for arr in row_cat_per_fold.values() for v in arr if v != neg}
                )

                rng = np.random.default_rng(seed)
                iterator = (
                    tqdm(
                        range(n_bootstraps),
                        desc=f"fold_mean[{cls}/{cat_name}]",
                        unit="draw",
                    ) if progress else range(n_bootstraps)
                )
                for b in iterator:
                    fold_idx_resamples = {
                        f: _stratified_resample(*fold_strata[f], rng)
                        for f in fold_ids_sorted
                    }
                    for cat_v in cat_values:
                        for clf in fold_arrays:
                            fold_metric_vals = {m: [] for m in metrics}
                            for f in fold_ids_sorted:
                                y, s = fold_arrays[clf][f]
                                idx = fold_idx_resamples[f]
                                rc = row_cat_per_fold[f][idx]
                                mask = rc == cat_v
                                if mask.sum() < min_rows:
                                    continue
                                for m in metrics:
                                    v = _safe_metric(m, y[idx][mask], s[idx][mask])
                                    if not np.isnan(v):
                                        fold_metric_vals[m].append(v)
                            for m in metrics:
                                if fold_metric_vals[m]:
                                    long_rows.append({
                                        "classifier": clf, "class": cls,
                                        "metric": m, "ap_type": "fold_mean",
                                        "category_name": cat_name,
                                        "category": cat_v,
                                        "bootstrap_idx": b,
                                        "value": float(np.mean(fold_metric_vals[m])),
                                    })

                # POINT — fold mean of un-resampled masked AP.
                for cat_v in cat_values:
                    for clf in fold_arrays:
                        fold_metric_vals = {m: [] for m in metrics}
                        for f in fold_ids_sorted:
                            y, s = fold_arrays[clf][f]
                            mask = row_cat_per_fold[f] == cat_v
                            if mask.sum() < min_rows:
                                continue
                            for m in metrics:
                                v = _safe_metric(m, y[mask], s[mask])
                                if not np.isnan(v):
                                    fold_metric_vals[m].append(v)
                        for m in metrics:
                            if fold_metric_vals[m]:
                                point_rows.append({
                                    "classifier": clf, "class": cls,
                                    "metric": m, "ap_type": "fold_mean",
                                    "category_name": cat_name,
                                    "category": cat_v,
                                    "value": float(np.mean(fold_metric_vals[m])),
                                })

    long_df = pd.DataFrame.from_records(long_rows) if long_rows else pd.DataFrame(
        columns=[
            "classifier", "class", "metric", "ap_type",
            "category_name", "category", "bootstrap_idx", "value",
        ]
    )
    point_df = pd.DataFrame.from_records(point_rows) if point_rows else pd.DataFrame(
        columns=[
            "classifier", "class", "metric", "ap_type",
            "category_name", "category", "value",
        ]
    )
    return long_df, point_df


def add_substrate_map_aggregate(
    long_df: pd.DataFrame, point_df: pd.DataFrame, *,
    substrate_classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append ``class="Substrate_mAP"`` rows that average AP across the
    listed substrates within each
    ``(classifier, category_name, category, ap_type, metric, draw)``
    cell. Cells where no substrates have data are dropped."""
    if long_df.empty and point_df.empty:
        return long_df, point_df

    keys = ["classifier", "metric", "ap_type", "category_name",
            "category", "bootstrap_idx"]
    point_keys = ["classifier", "metric", "ap_type", "category_name", "category"]

    long_pieces = [long_df]
    point_pieces = [point_df]
    if not long_df.empty:
        sub = long_df[long_df["class"].isin(substrate_classes)]
        if not sub.empty:
            agg = sub.groupby(keys, as_index=False)["value"].mean()
            agg["class"] = "Substrate_mAP"
            long_pieces.append(agg)
    if not point_df.empty:
        sub = point_df[point_df["class"].isin(substrate_classes)]
        if not sub.empty:
            agg = sub.groupby(point_keys, as_index=False)["value"].mean()
            agg["class"] = "Substrate_mAP"
            point_pieces.append(agg)
    return (
        pd.concat(long_pieces, ignore_index=True),
        pd.concat(point_pieces, ignore_index=True),
    )
