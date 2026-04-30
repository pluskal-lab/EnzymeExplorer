"""Precision-targeted prediction-score thresholds.

For an ML model that emits a continuous probability per (sequence, class), this
module computes the smallest score threshold T at which precision among
``{predictions: score >= T}`` first meets a target value (e.g. T@p=0.99 for a
"high-confidence" tier). The thresholds are derived from pooled out-of-fold
predictions — the cleanest signal for deployment calibration since each
sequence's score comes from a model that didn't see it during training.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs

logger = logging.getLogger(__name__)


def compute_precision_targeted_thresholds(
    labels_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    target_class: str,
    precisions: Iterable[float],
) -> pd.DataFrame:
    """For each precision target ``p``, find the smallest threshold ``T``
    such that precision of ``{score >= T}`` is at least ``p``.

    Returns a frame with one row per target containing ``threshold``,
    ``achieved_precision``, ``recall``, and ``n_above_threshold``. NaN rows
    are emitted when a target is unreachable (e.g. there are zero positives,
    or the maximum achievable precision is below ``p``).
    """
    precisions = list(precisions)
    y = labels_df[target_class].to_numpy()
    s = preds_df[target_class].to_numpy()

    if y.sum() == 0:
        return pd.DataFrame(
            [
                {
                    "precision_target": p,
                    "threshold": float("nan"),
                    "achieved_precision": float("nan"),
                    "recall": float("nan"),
                    "n_above_threshold": 0,
                }
                for p in precisions
            ]
        )

    pr, rc, th = precision_recall_curve(y, s)
    # ``precision_recall_curve`` appends a sentinel (precision=1, recall=0)
    # at the end without a corresponding threshold; ignore it when scanning
    # for the target, since it has no actionable threshold value.
    pr_scan = pr[:-1]
    rc_scan = rc[:-1]
    rows: list[dict] = []
    for p_target in precisions:
        valid = pr_scan >= p_target
        if not valid.any():
            rows.append(
                {
                    "precision_target": float(p_target),
                    "threshold": float("nan"),
                    "achieved_precision": float("nan"),
                    "recall": float("nan"),
                    "n_above_threshold": 0,
                }
            )
            continue
        # Smallest threshold achieving the target = first index from the
        # left of the precision_recall_curve output that meets the target.
        i = int(np.argmax(valid))
        rows.append(
            {
                "precision_target": float(p_target),
                "threshold": float(th[i]),
                "achieved_precision": float(pr_scan[i]),
                "recall": float(rc_scan[i]),
                "n_above_threshold": int((s >= th[i]).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_recall_targeted_threshold(
    labels_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    target_class: str,
    recall_floor: float,
) -> dict | None:
    """Largest threshold ``T`` such that recall of ``{score >= T}`` is at
    least ``recall_floor``.

    Used by the confidence-tier system to cap the Negative band so that no
    more than ``1 - recall_floor`` of true positives are mis-labelled as
    Negative. Returns ``None`` when there are no positives or the floor is
    unreachable for any threshold.
    """
    y = labels_df[target_class].to_numpy()
    s = preds_df[target_class].to_numpy()
    if y.sum() == 0:
        return None
    pr, rc, th = precision_recall_curve(y, s)
    # rc decreases (mostly) monotonically with t; find the largest threshold
    # whose recall still meets the floor.
    rc_scan = rc[:-1]
    pr_scan = pr[:-1]
    valid = rc_scan >= recall_floor
    if not valid.any():
        return None
    i = int(np.where(valid)[0][-1])
    return {
        "threshold": float(th[i]),
        "achieved_recall": float(rc_scan[i]),
        "achieved_precision": float(pr_scan[i]),
        "n_above_threshold": int((s >= th[i]).sum()),
    }


def compute_thresholds_table(
    pooled: dict[str, dict[str, FoldDfs]],
    classifiers: Iterable[str],
    target_classes: Iterable[str],
    precisions: Iterable[float],
) -> pd.DataFrame:
    """Long-form table over ``(classifier, class, precision_target)``.

    Skips ``(classifier, class)`` pairs that don't exist in ``pooled``.
    """
    classifiers = list(classifiers)
    target_classes = list(target_classes)
    precisions = list(precisions)
    pieces: list[pd.DataFrame] = []
    for clf in classifiers:
        if clf not in pooled:
            logger.warning("Classifier %s not in pooled dfs; skipping", clf)
            continue
        for cls in target_classes:
            if cls not in pooled[clf]:
                logger.warning(
                    "Class %s not available for %s; skipping", cls, clf
                )
                continue
            lab, prd = pooled[clf][cls]
            sub = compute_precision_targeted_thresholds(lab, prd, cls, precisions)
            sub.insert(0, "class", cls)
            sub.insert(0, "classifier", clf)
            pieces.append(sub)
    if not pieces:
        return pd.DataFrame(
            columns=[
                "classifier",
                "class",
                "precision_target",
                "threshold",
                "achieved_precision",
                "recall",
                "n_above_threshold",
            ]
        )
    return pd.concat(pieces, ignore_index=True)
