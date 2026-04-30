"""Aggregate "virtual" classes for the evaluation pipeline.

Adds rows like ``Substrate_mAP`` and ``TPS_IDS_mAP`` by averaging metric values
across a configured set of classes, while preserving the
``bootstrap_idx``/``fold_left_out`` indexing so percentile and BCa CIs work
downstream without changes.
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
    sub = df[df["class"].isin(members)]
    coverage = sub.groupby(["classifier"])["class"].nunique()
    return set(coverage[coverage == len(set(members))].index)


def _aggregate_with_index(
    df: pd.DataFrame,
    aggregates: Mapping[str, list[str]],
    *,
    extra_keys: list[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = [df]
    grouping_extras = list(extra_keys)
    if "category" in df.columns and "category" not in grouping_extras:
        grouping_extras.append("category")
    for agg_name, members in aggregates.items():
        eligible = _classifiers_covering(df, members)
        if not eligible:
            continue
        sub = df[df["classifier"].isin(eligible) & df["class"].isin(members)]
        group = ["classifier", "metric", *grouping_extras]
        agg = sub.groupby(group, as_index=False)["value"].mean()
        agg["class"] = agg_name
        pieces.append(
            agg[["classifier", "class", "metric", *grouping_extras, "value"]]
        )
    return pd.concat(pieces, ignore_index=True)


def add_aggregates(
    result: BootstrapResult,
    aggregates: Mapping[str, list[str]] | None = None,
) -> BootstrapResult:
    """Return a new ``BootstrapResult`` with aggregate-class rows appended to
    ``long_df``, ``point_estimates`` and ``jackknife``.

    Per (classifier, metric[, draw / fold]), each aggregate is the simple
    mean of its member classes' metric values. Classifiers that don't cover
    every member of an aggregate are excluded from that aggregate row.
    """
    aggregates = dict(aggregates if aggregates is not None else DEFAULT_AGGREGATES)
    return BootstrapResult(
        long_df=_aggregate_with_index(
            result.long_df, aggregates, extra_keys=["bootstrap_idx"]
        ),
        point_estimates=_aggregate_with_index(
            result.point_estimates, aggregates, extra_keys=[]
        ),
        jackknife=_aggregate_with_index(
            result.jackknife, aggregates, extra_keys=["fold_left_out"]
        ),
    )
