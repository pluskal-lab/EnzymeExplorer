"""Confidence-tier mapping for prediction scores.

Each tier is defined by a precision target on held-out (pooled out-of-fold)
data; the tier's score lower-bound is the smallest score at which precision
of ``{predictions: score >= T}`` first meets the target. A score below the
loosest tier's lower-bound is labelled "Negative". When a tier's precision
target is unreachable for a class, it collapses into the next-lower reachable
tier — we never assign a tier we couldn't validate.

This module produces:
  * a frozen lookup table (CSV-friendly) the future inference module will use
    to translate raw scores into tier labels at deploy time, and
  * a runtime helper (``apply_tiers_to_predictions``) for that translation.

Nothing here touches the experiment pipeline; it's a layer over the
``prediction_thresholds`` analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs
from enzymeexplorer.src.evaluation.prediction_thresholds import (
    compute_precision_targeted_thresholds,
    compute_recall_targeted_threshold,
)

NEGATIVE_TIER_NAME = "Negative"


@dataclass(frozen=True)
class TierDefinition:
    """One confidence tier.

    Exactly one of ``precision_target`` or ``recall_target`` must be set.
    A precision-target tier's lower bound is the smallest score at which
    precision of ``{score >= T}`` first meets the target. A recall-target
    tier's lower bound is the largest score at which recall of
    ``{score >= T}`` still meets the target — useful for the loosest tier
    (e.g. ``recall_target=1.0`` gives the score at which every true
    positive is still captured).

    ``classes`` restricts the tier to a subset of target classes; ``None``
    (default) means the tier applies to every class. Used by the
    "Ultra-High Confidence" (p≥0.9999) tier which is only meaningful for
    the TPS detection task.
    """

    name: str
    precision_target: float | None = None
    recall_target: float | None = None
    color: str = ""
    classes: tuple[str, ...] | None = None


DEFAULT_TIER_DEFINITIONS: tuple[TierDefinition, ...] = (
    # TPS-only "near-certain" band: only emitted when the precision target is
    # actually reachable — otherwise it silently collapses into Very High.
    TierDefinition(
        "Ultra-High Confidence", precision_target=0.9999, color="#006837",
        classes=("TPS",),
    ),
    TierDefinition("Very High Confidence", precision_target=0.99, color="#1a9850"),
    TierDefinition("High Confidence", precision_target=0.95, color="#66bd63"),
    TierDefinition("Mid Confidence", precision_target=0.80, color="#fee08b"),
    TierDefinition("Low Confidence", precision_target=0.50, color="#fdae61"),
    TierDefinition("Very Low Confidence", recall_target=1.0, color="#f46d43"),
)
NEGATIVE_TIER_COLOR = "#cccccc"


def _validate(tier: TierDefinition) -> None:
    if (tier.precision_target is None) == (tier.recall_target is None):
        raise ValueError(
            f"Tier {tier.name!r} must set exactly one of "
            "precision_target or recall_target"
        )


def tier_definitions_from_config(
    raw: Iterable[Mapping] | None,
) -> tuple[TierDefinition, ...]:
    """Turn YAML-style dicts into ``TierDefinition`` objects.

    Each entry must set exactly one of ``precision_target`` or
    ``recall_target``. Returned in the order the YAML lists them; downstream
    logic sorts by computed lower bound at render time.
    """
    if not raw:
        return DEFAULT_TIER_DEFINITIONS
    out: list[TierDefinition] = []
    for item in raw:
        cls_list = item.get("classes")
        tier = TierDefinition(
            name=item["name"],
            precision_target=(
                float(item["precision_target"])
                if "precision_target" in item and item["precision_target"] is not None
                else None
            ),
            recall_target=(
                float(item["recall_target"])
                if "recall_target" in item and item["recall_target"] is not None
                else None
            ),
            color=item.get("color", ""),
            classes=tuple(cls_list) if cls_list else None,
        )
        _validate(tier)
        out.append(tier)
    return tuple(out)


def _tiers_for_class(
    tier_definitions: Iterable[TierDefinition], target_class: str,
) -> list[TierDefinition]:
    """Filter tier_definitions down to those that apply to ``target_class``."""
    return [
        t for t in tier_definitions
        if t.classes is None or target_class in t.classes
    ]


def compute_tier_table(
    pooled: dict[str, dict[str, FoldDfs]],
    classifiers: Iterable[str],
    target_classes: Iterable[str],
    tier_definitions: Iterable[TierDefinition] = DEFAULT_TIER_DEFINITIONS,
) -> pd.DataFrame:
    """For each ``(classifier, class, tier)`` derive the score lower-bound.

    Returns long-form columns:
    ``classifier, class, tier, target_type, target_value, lower_bound,
     achieved_precision, recall, n_above_lower, reachable``.

    ``target_type`` is ``"precision"`` (smallest T meeting the target) or
    ``"recall"`` (largest T still meeting the target). Unreachable tiers
    carry NaN bounds and ``reachable=False``.
    """
    classifiers = list(classifiers)
    target_classes = list(target_classes)
    tiers = list(tier_definitions)
    for tier in tiers:
        _validate(tier)

    rows: list[dict] = []
    for clf in classifiers:
        if clf not in pooled:
            continue
        for cls in target_classes:
            if cls not in pooled[clf]:
                continue
            cls_tiers = _tiers_for_class(tiers, cls)
            precisions = sorted(
                {t.precision_target for t in cls_tiers if t.precision_target is not None}
            )
            recall_targets = sorted(
                {t.recall_target for t in cls_tiers if t.recall_target is not None}
            )
            lab, prd = pooled[clf][cls]
            prec_table = (
                compute_precision_targeted_thresholds(lab, prd, cls, precisions)
                .set_index("precision_target")
                if precisions
                else None
            )
            recall_results = {
                rt: compute_recall_targeted_threshold(lab, prd, cls, rt)
                for rt in recall_targets
            }

            for tier in cls_tiers:
                if tier.precision_target is not None:
                    row = prec_table.loc[tier.precision_target]
                    lb = float(row["threshold"])
                    rows.append(
                        {
                            "classifier": clf,
                            "class": cls,
                            "tier": tier.name,
                            "target_type": "precision",
                            "target_value": tier.precision_target,
                            "lower_bound": lb,
                            "achieved_precision": float(row["achieved_precision"]),
                            "recall": float(row["recall"]),
                            "n_above_lower": int(row["n_above_threshold"]),
                            "reachable": bool(np.isfinite(lb)),
                        }
                    )
                else:
                    info = recall_results.get(tier.recall_target)
                    if info is None:
                        rows.append(
                            {
                                "classifier": clf,
                                "class": cls,
                                "tier": tier.name,
                                "target_type": "recall",
                                "target_value": tier.recall_target,
                                "lower_bound": float("nan"),
                                "achieved_precision": float("nan"),
                                "recall": float("nan"),
                                "n_above_lower": 0,
                                "reachable": False,
                            }
                        )
                    else:
                        rows.append(
                            {
                                "classifier": clf,
                                "class": cls,
                                "tier": tier.name,
                                "target_type": "recall",
                                "target_value": tier.recall_target,
                                "lower_bound": float(info["threshold"]),
                                "achieved_precision": float(info["achieved_precision"]),
                                "recall": float(info["achieved_recall"]),
                                "n_above_lower": int(info["n_above_threshold"]),
                                "reachable": True,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def _compute_bands_from_rows(
    rows: pd.DataFrame,
    *,
    score_max: float = 1.0,
    negative_label: str = NEGATIVE_TIER_NAME,
) -> list[tuple[str, float, float]]:
    """Shared band-construction logic used by ``assign_tiers`` and
    ``tier_intervals_for_class``. ``rows`` is the per-class slice of the
    tier table (rows with ``lower_bound==NaN`` already dropped)."""
    if rows.empty:
        return [(negative_label, 0.0, score_max)]

    recall_rows = rows[rows["target_type"] == "recall"]
    precision_rows = rows[rows["target_type"] == "precision"].sort_values(
        "target_value"
    )
    p_targets = precision_rows["target_value"].tolist()
    p_lbs = [float(x) for x in precision_rows["lower_bound"].tolist()]
    p_names = precision_rows["tier"].tolist()

    def _walk(start_running: float, start_prev: str) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        running = start_running
        prev_tier = start_prev
        for lb, name in zip(p_lbs, p_names):
            if lb <= running:
                if lb == running:
                    prev_tier = name
                continue
            out.append((prev_tier, running, lb))
            running = lb
            prev_tier = name
        out.append((prev_tier, running, score_max))
        return out

    if recall_rows.empty:
        return _walk(0.0, negative_label)

    recall_row = recall_rows.iloc[0]
    t_recall = float(recall_row["lower_bound"])
    p_at_recall = float(recall_row["achieved_precision"])
    very_low_name = recall_row["tier"]

    bands: list[tuple[str, float, float]] = []
    if t_recall > 0:
        bands.append((negative_label, 0.0, t_recall))

    if not p_lbs:
        if t_recall < score_max:
            bands.append((very_low_name, t_recall, score_max))
        return bands

    if p_at_recall < p_targets[0]:
        # P(T_recall) below the loosest precision target — emit a Very Low
        # band, then the full precision sequence.
        if p_lbs[0] > t_recall:
            bands.append((very_low_name, t_recall, p_lbs[0]))
        for piece in _walk(p_lbs[0], p_names[0]):
            if piece[2] > piece[1]:
                bands.append(piece)
        return bands

    # P(T_recall) already meets the loosest precision target. The strictest
    # precision target whose value is <= P(T_recall) takes ownership at
    # T_recall; all looser precision tiers are absorbed (their precision
    # thresholds sit below T_recall, in Negative territory).
    start_idx = 0
    for i, p in enumerate(p_targets):
        if p <= p_at_recall:
            start_idx = i
        else:
            break

    running = t_recall
    prev_tier = p_names[start_idx]
    for i in range(start_idx + 1, len(p_lbs)):
        lb = p_lbs[i]
        if lb <= running:
            if lb == running:
                prev_tier = p_names[i]
            continue
        bands.append((prev_tier, running, lb))
        running = lb
        prev_tier = p_names[i]
    bands.append((prev_tier, running, score_max))
    return bands


def assign_tiers(
    scores: np.ndarray,
    tier_table_for_class: pd.DataFrame,
    *,
    negative_label: str = NEGATIVE_TIER_NAME,
) -> np.ndarray:
    """Map raw scores to tier names for a single ``(classifier, class)``.

    Computes the same bands the visualisation draws and assigns each score
    to the band whose ``[lower, upper)`` interval contains it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    out = np.full(scores.shape, negative_label, dtype=object)
    rows = tier_table_for_class.dropna(subset=["lower_bound"])
    bands = _compute_bands_from_rows(rows, score_max=1.0, negative_label=negative_label)
    # Walk strictest first; first matching band wins.
    for name, lo, _ in reversed(bands):
        if name == negative_label:
            break
        mask = (scores >= lo) & (out == negative_label)
        out[mask] = name
    return out


def apply_tiers_to_predictions(
    preds_df: pd.DataFrame,
    tier_table: pd.DataFrame,
    classifier: str,
    *,
    classes: Iterable[str] | None = None,
    id_col: str = "ID",
    negative_label: str = NEGATIVE_TIER_NAME,
) -> pd.DataFrame:
    """Vectorised tier assignment for a multi-class score frame.

    Returns a DataFrame with columns ``ID, <class>_score, <class>_tier`` for
    each requested class. Designed to be called by the future inference
    module after raw scores are computed.
    """
    if classes is None:
        classes = [c for c in preds_df.columns if c != id_col]
    classes = list(classes)
    sub = tier_table[tier_table["classifier"] == classifier]
    if sub.empty:
        raise ValueError(f"No tier rows for classifier {classifier!r}")
    out = preds_df[[id_col]].copy()
    for cls in classes:
        if cls not in preds_df.columns:
            continue
        cls_rows = sub[sub["class"] == cls]
        if cls_rows.empty:
            continue
        out[f"{cls}_score"] = preds_df[cls].to_numpy()
        out[f"{cls}_tier"] = assign_tiers(
            preds_df[cls].to_numpy(),
            cls_rows,
            negative_label=negative_label,
        )
    return out


def tier_intervals_for_class(
    tier_table: pd.DataFrame,
    classifier: str,
    target_class: str,
    tier_definitions: Iterable[TierDefinition] = DEFAULT_TIER_DEFINITIONS,
    *,
    score_max: float = 1.0,
    negative_label: str = NEGATIVE_TIER_NAME,
) -> list[tuple[str, float, float]]:
    """Return the contiguous ``(tier, lo, hi)`` segments covering ``[0, 1]``.

    Walks tiers from loosest precision target to strictest. Unreachable
    tiers collapse into the next-lower reachable tier (i.e. their segment
    is absorbed). The first segment is the Negative band.
    """
    rows = tier_table[
        (tier_table["classifier"] == classifier)
        & (tier_table["class"] == target_class)
    ].dropna(subset=["lower_bound"])
    return _compute_bands_from_rows(
        rows, score_max=score_max, negative_label=negative_label
    )
