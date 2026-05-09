"""Paired-bootstrap p-values + Holm adjustment + transitive reduction.

The bootstrap delta table produced by
:func:`enzymeexplorer.src.evaluation.bootstrap.paired_bootstrap_metric_cis`
provides, for every method pair × class × metric × AP type, a
distribution of paired differences ``delta_b = AP_a[b] - AP_b[b]``.

Two-sided p-value:

.. math:: p_{\\text{raw}} = 2 \\cdot \\min(\\Pr(\\delta_b \\ge 0),\\ \\Pr(\\delta_b \\le 0))

Adjustment is Holm-Bonferroni across all method pairs within a single
``(class, metric, ap_type)`` family.

Significance for transitive reduction is **CI-based**, not p-value-based:
``A`` is significantly better than ``B`` iff the CI of
``AP_A − AP_B`` is entirely > 0; significantly worse iff entirely < 0.
The transitive reduction prunes pairs that are *implied* by a chain of
significant edges.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


PvalueAdjustment = Literal["holm", "bonferroni", "none"]


# ---------------------------------------------------------------------------
# Paired-bootstrap p-values
# ---------------------------------------------------------------------------


def _two_sided_pvalue(deltas: np.ndarray) -> float:
    """Two-sided p-value of ``mean(deltas) ≠ 0`` from the bootstrap distribution."""
    b = deltas[~np.isnan(deltas)]
    if b.size == 0:
        return float("nan")
    p_pos_or_eq = float(np.mean(b >= 0))
    p_neg_or_eq = float(np.mean(b <= 0))
    raw = 2.0 * min(p_pos_or_eq, p_neg_or_eq)
    # Bound to (0, 1] — at least 1 / B if no draws fall on the wrong side.
    return float(min(1.0, max(raw, 1.0 / b.size)))


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment.

    Sort p-values ascending, multiply each by ``(n - i)`` where ``i`` is
    the rank, then take the cumulative max so adjusted p-values stay
    monotone non-decreasing in rank. Restore the original ordering.
    """
    n = p_values.size
    if n == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.minimum(1.0, sorted_p * (n - np.arange(n)))
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    out = np.empty(n, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


def _bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p_values * p_values.size)


def compute_pvalues(
    long_delta: pd.DataFrame,
    *,
    adjustment: PvalueAdjustment = "holm",
) -> pd.DataFrame:
    """Per (class, metric, ap_type, pair) raw + adjusted p-values.

    Returns a DataFrame with columns
    ``classifier_a, classifier_b, class, metric, ap_type,
    p_raw, p_adjusted, adjustment``.
    """
    if long_delta.empty:
        return pd.DataFrame(
            columns=[
                "classifier_a", "classifier_b", "class", "metric", "ap_type",
                "p_raw", "p_adjusted", "adjustment",
            ]
        )

    rows = []
    family_keys = ["class", "metric", "ap_type"]
    for family, family_df in long_delta.groupby(family_keys):
        # Per family, collect per-pair raw p-values
        pair_p_raw: list[tuple[tuple[str, str], float]] = []
        for (a, b), pair_df in family_df.groupby(["classifier_a", "classifier_b"]):
            deltas = pair_df["value"].to_numpy(dtype=np.float64)
            pair_p_raw.append(((a, b), _two_sided_pvalue(deltas)))
        if not pair_p_raw:
            continue
        p_raw_arr = np.asarray([p for _, p in pair_p_raw], dtype=np.float64)
        if adjustment == "holm":
            p_adj = _holm_adjust(p_raw_arr)
            adj_label = "holm"
        elif adjustment == "bonferroni":
            p_adj = _bonferroni_adjust(p_raw_arr)
            adj_label = "bonferroni"
        elif adjustment == "none":
            p_adj = p_raw_arr.copy()
            adj_label = "none"
        else:
            raise ValueError(f"Unsupported adjustment: {adjustment!r}")
        family_dict = dict(zip(family_keys, family))
        for ((a, b), _p_raw), p_a in zip(pair_p_raw, p_adj):
            rows.append(
                {
                    "classifier_a": a,
                    "classifier_b": b,
                    **family_dict,
                    "p_raw": float(_p_raw),
                    "p_adjusted": float(p_a),
                    "adjustment": adj_label,
                }
            )
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Transitive reduction of the CI-based significance graph
# ---------------------------------------------------------------------------


def _ci_significance(ci_low: float, ci_high: float) -> int:
    """Sign of significant difference based on a CI: +1, -1, or 0."""
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return 0
    if ci_low > 0:
        return +1
    if ci_high < 0:
        return -1
    return 0


def _transitive_reduce(edges: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """Compute the transitive reduction of a directed graph.

    Implementation: for each edge ``a → b``, the edge is *redundant*
    iff there exists an alternate path ``a → ... → b`` of length > 1
    using the other edges. That alternate-path test is run on the
    graph minus the candidate edge. Bare-bones BFS, no extra deps.
    """
    edge_set = {(a, b) for a, b in edges}
    nodes = {n for ab in edge_set for n in ab}
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edge_set:
        adj[a].add(b)

    redundant: set[tuple[str, str]] = set()
    for a, b in edge_set:
        # BFS from a, ignoring the direct edge a → b.
        visited = {a}
        stack = [n for n in adj[a] if n != b]
        found = False
        while stack:
            cur = stack.pop()
            if cur == b:
                found = True
                break
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    stack.append(nxt)
        if found:
            redundant.add((a, b))
    return edge_set - redundant


def transitive_reduction_pairs(
    summary_delta: pd.DataFrame,
    *,
    ap_type: str,
    ci_method: str,
) -> dict[tuple[str, str, str], set[tuple[str, str]]]:
    """For each (class, metric) family in ``ap_type/ci_method``, return
    the set of *surviving* method pairs after transitive reduction.

    Surviving pairs = (significant edges that are NOT implied by a
    chain) ∪ (non-significant edges, which always survive — they're
    the "no clear winner" comparisons).

    Returned mapping: ``{(class, metric, ap_type): {(a, b), ...}}``.
    Each pair appears exactly once with the orientation found in
    ``summary_delta``.
    """
    out: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    sub = summary_delta[
        (summary_delta["ap_type"] == ap_type)
        & (summary_delta["method"] == ci_method)
    ]
    for (cls, metric, ap), grp in sub.groupby(["class", "metric", "ap_type"]):
        sig_edges: list[tuple[str, str]] = []
        all_pairs: list[tuple[str, str]] = []
        for _, r in grp.iterrows():
            a, b = r["classifier_a"], r["classifier_b"]
            all_pairs.append((a, b))
            sig = _ci_significance(r["ci_low"], r["ci_high"])
            if sig == +1:
                sig_edges.append((a, b))
            elif sig == -1:
                sig_edges.append((b, a))
        if not sig_edges:
            out[(cls, metric, ap)] = set(all_pairs)
            continue
        survivors = _transitive_reduce(sig_edges)  # significant survivors only
        # Build final pair set: keep an unordered pair if its
        # significant-edge survives OR if it's non-significant.
        kept: set[tuple[str, str]] = set()
        survivor_unordered = {tuple(sorted(p)) for p in survivors}
        for a, b in all_pairs:
            sig = _ci_significance(
                grp[(grp["classifier_a"] == a) & (grp["classifier_b"] == b)]["ci_low"].iloc[0],
                grp[(grp["classifier_a"] == a) & (grp["classifier_b"] == b)]["ci_high"].iloc[0],
            )
            if sig == 0:
                kept.add((a, b))
            elif tuple(sorted((a, b))) in survivor_unordered:
                kept.add((a, b))
        out[(cls, metric, ap)] = kept
    return out
