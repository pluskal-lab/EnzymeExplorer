"""Aggregate "virtual" classes for the v4 evaluation pipeline.

Adds rows like ``Substrate_mAP`` and ``TPS_IDS_mAP`` by averaging metric
values across a configured set of classes. The aggregate is computed
**per draw** for the bootstrap distributions and **on the point
estimates** for the AP/delta point tables, so downstream CIs and
p-values come out coherent with the per-class draws.

Both ``long_ap`` / ``point_ap`` (one row per classifier × class) and
``long_delta`` / ``point_delta`` (one row per classifier-pair × class)
get aggregate "classes" appended.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.bootstrap import BootstrapResult
from enzymeexplorer.src.evaluation.classes import (
    ALL_CLASSES,
    DETECTION_CLASSES,
    SUBSTRATE_CLASSES,
)

DEFAULT_AGGREGATES: dict[str, list[str]] = {
    "Substrate_mAP": SUBSTRATE_CLASSES,
    "TPS_IDS_mAP": DETECTION_CLASSES,
    "Overall_mAP": ALL_CLASSES,
}


def _classifiers_covering(df: pd.DataFrame, members: list[str]) -> set[str]:
    """Classifiers (or pairs) that have entries for every member class.

    Works on both AP-style frames (``classifier`` column) and
    delta-style frames (``classifier_a``/``classifier_b`` columns).
    """
    sub = df[df["class"].isin(members)]
    if "classifier" in df.columns:
        coverage = sub.groupby(["classifier"])["class"].nunique()
        return set(coverage[coverage == len(set(members))].index)
    coverage = sub.groupby(["classifier_a", "classifier_b"])["class"].nunique()
    pairs = coverage[coverage == len(set(members))].index
    return set(pairs)


def _aggregate(
    df: pd.DataFrame,
    aggregates: Mapping[str, list[str]],
    *,
    extra_keys: list[str],
) -> pd.DataFrame:
    """Per-aggregate: simple mean of member-class values, indexed the same way as the source."""
    if df.empty:
        return df
    pieces: list[pd.DataFrame] = [df]
    is_delta = "classifier_a" in df.columns
    id_cols = (
        ["classifier_a", "classifier_b"] if is_delta else ["classifier"]
    )
    grouping_extras = list(extra_keys)
    for agg_name, members in aggregates.items():
        eligible = _classifiers_covering(df, members)
        if not eligible:
            continue
        if is_delta:
            mask = pd.Series(False, index=df.index)
            for a, b in eligible:
                mask |= (df["classifier_a"] == a) & (df["classifier_b"] == b)
            sub = df[mask & df["class"].isin(members)]
        else:
            sub = df[df["classifier"].isin(eligible) & df["class"].isin(members)]
        group = [*id_cols, "metric", *grouping_extras]
        if "ap_type" in df.columns and "ap_type" not in group:
            group.append("ap_type")
        agg = sub.groupby(group, as_index=False)["value"].mean()
        agg["class"] = agg_name
        out_cols = [*id_cols, "class", "metric", *grouping_extras]
        if "ap_type" in df.columns and "ap_type" not in out_cols:
            out_cols.append("ap_type")
        out_cols.append("value")
        pieces.append(agg[out_cols])
    return pd.concat(pieces, ignore_index=True)


def add_aggregates(
    result: BootstrapResult,
    aggregates: Mapping[str, list[str]] | None = None,
) -> BootstrapResult:
    """Return a new ``BootstrapResult`` with aggregate "classes" appended.

    Aggregates are computed for both AP and delta tables. Eligibility is
    determined per table — a classifier (or pair) appears as an
    aggregate row only if it has all member-class entries for that
    metric × ap_type cell.
    """
    aggregates = dict(aggregates if aggregates is not None else DEFAULT_AGGREGATES)
    return BootstrapResult(
        long_ap=_aggregate(
            result.long_ap, aggregates, extra_keys=["bootstrap_idx"],
        ),
        point_ap=_aggregate(result.point_ap, aggregates, extra_keys=[]),
        long_delta=_aggregate(
            result.long_delta, aggregates, extra_keys=["bootstrap_idx"],
        ),
        point_delta=_aggregate(result.point_delta, aggregates, extra_keys=[]),
        jackknife_ap=_aggregate(
            result.jackknife_ap, aggregates, extra_keys=["fold_left_out"],
        ) if not result.jackknife_ap.empty else result.jackknife_ap,
        jackknife_delta=_aggregate(
            result.jackknife_delta, aggregates, extra_keys=["fold_left_out"],
        ) if not result.jackknife_delta.empty else result.jackknife_delta,
    )
