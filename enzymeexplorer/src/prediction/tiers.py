"""Confidence-tier assignment from ``confidence_tiers.csv``.

The CSV (produced by the evaluation pipeline at
``outputs/evaluation_results/single_hierarchy_final/confidence_tiers.csv``)
defines, per (classifier, class), an ordered set of score thresholds with
named tiers:

    classifier,class,tier,target_type,target_value,lower_bound,...,reachable
    PLM_Domains,TPS,Very High Confidence,precision,0.99,0.5039,...,True
    PLM_Domains,TPS,High Confidence,precision,0.95,0.135,...,True
    ...

For a given (classifier, class, score) the tier is the strictest band whose
``lower_bound`` ≤ score (i.e. the band with the highest target_value among
those the score reaches). Scores below every ``reachable`` band's lower bound
fall into a synthesised ``"Negative"`` bucket so every row in the output
always has a tier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore


NEGATIVE_TIER = "Negative"


def load_tier_table(
    csv_path: str | Path, classifier_name: str
) -> dict[str, list[tuple[str, float]]]:
    """Return ``{class_name: [(tier_name, lower_bound), ...]}`` sorted by
    decreasing strictness — the first tier whose ``lower_bound`` ≤ score wins.
    Unreachable tiers are dropped.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["classifier"] == classifier_name) & (df["reachable"])]
    if df.empty:
        raise ValueError(
            f"No reachable tiers found for classifier={classifier_name!r} "
            f"in {csv_path}. Available classifiers: "
            f"{sorted(pd.read_csv(csv_path)['classifier'].unique())}"
        )
    # For precision tiers, higher target_value = stricter; for recall tiers
    # (the Very Low band, target_type='recall'), lower lower_bound = looser.
    # Sorting by lower_bound descending gives the right "first match wins" order.
    out: dict[str, list[tuple[str, float]]] = {}
    for class_name, sub in df.groupby("class"):
        ordered = (
            sub.sort_values("lower_bound", ascending=False)[["tier", "lower_bound"]]
            .itertuples(index=False, name=None)
        )
        out[class_name] = [(t, float(b)) for t, b in ordered]
    return out


def assign_tier(score: float, ordered_tiers: list[tuple[str, float]]) -> str:
    """Pick the first tier whose ``lower_bound`` ≤ score; fallback Negative."""
    for tier_name, lower_bound in ordered_tiers:
        if score >= lower_bound:
            return tier_name
    return NEGATIVE_TIER


def assign_tiers_long(
    long_df: pd.DataFrame,
    csv_path: str | Path,
    classifier_name: str,
) -> pd.DataFrame:
    """Long-form (``[id, class, score]``) → adds a ``tier`` column."""
    if long_df.empty:
        return long_df.assign(tier=pd.Series(dtype=str))
    table = load_tier_table(csv_path, classifier_name)
    tiers = []
    for _, row in long_df.iterrows():
        ordered = table.get(row["class"], [])
        tiers.append(assign_tier(float(row["score"]), ordered))
    return long_df.assign(tier=tiers)


def assemble_output_table(
    long_with_tiers: pd.DataFrame,
    *,
    sequence_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pivot long ``[id, class, score, tier]`` to a wide per-protein table.

    Output columns: ``id``, optionally ``sequence``, then for each class
    ``<class>_score`` and ``<class>_tier``.
    """
    if long_with_tiers.empty:
        cols = ["id"] + (["sequence"] if sequence_lookup else [])
        return pd.DataFrame(columns=cols)

    score_wide = long_with_tiers.pivot(
        index="id", columns="class", values="score"
    )
    score_wide.columns = [f"{c}_score" for c in score_wide.columns]
    tier_wide = long_with_tiers.pivot(
        index="id", columns="class", values="tier"
    )
    tier_wide.columns = [f"{c}_tier" for c in tier_wide.columns]

    out = pd.concat([score_wide, tier_wide], axis=1).reset_index()
    if sequence_lookup is not None:
        out.insert(1, "sequence", out["id"].map(sequence_lookup))
    return out
