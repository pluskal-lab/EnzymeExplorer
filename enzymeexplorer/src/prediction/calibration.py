"""Inference-time application of per-class beta calibration.

Loads ``calibration/fit_summary.csv`` (emitted by the calibration pipeline),
applies the deployment-fit beta calibrator to per-class scores, and
surfaces the calibrated probability directly. No tier labels — the
output table carries a raw ``<class>_raw`` score and a calibrated
``<class>_p`` per class.

Classes whose calibration was skipped at training time (``status``
starting with anything other than ``fit``) get ``NaN`` in the
``_p`` column so downstream consumers can detect "no calibrated
claim available".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.calibration import (
    calibrators_from_fit_summary,
)

logger = logging.getLogger(__name__)


# Canonical column order for prediction outputs. Wide-form tables are
# emitted as ``id`` → per-class calibrated probabilities (`<class>_p`) →
# per-class raw scores (`<class>_raw`) → optional ``sequence``. The class
# order below is fixed for paper-facing reproducibility (TPS first,
# monoterpenes/diterpenes/etc. grouped by carbon count, IDS last).
_OUTPUT_CLASS_ORDER: tuple[str, ...] = (
    "TPS", "GPP", "FPP", "GGPP", "GFPP",
    "CPP", "EDSQ", "2xFPP", "2xGGPP", "IDS",
)


_REQUIRED_COLUMNS = {
    "classifier", "target_class", "family",
    "a", "b", "c", "T", "eps", "status",
}


def _read_fit_summary(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Calibration fit_summary at {csv_path} missing required "
            f"columns: {sorted(missing)}"
        )
    return df


def apply_calibration_long(
    long_df: pd.DataFrame,
    fit_summary_csv: str | Path,
    classifier_name: str,
) -> pd.DataFrame:
    """Long ``[id, class, score]`` → adds ``p`` (calibrated probability).

    Classes without a deployable calibrator (status not starting with
    ``fit``) get ``NaN`` in ``p``. The classifier must be present in
    the fit summary; otherwise raises ``ValueError`` to make
    misconfiguration loud.
    """
    if long_df.empty:
        return long_df.assign(p=pd.Series(dtype=float))
    fit_summary = _read_fit_summary(fit_summary_csv)
    sub = fit_summary[fit_summary["classifier"] == classifier_name]
    if sub.empty:
        avail = sorted(fit_summary["classifier"].unique())
        raise ValueError(
            f"No calibration rows for classifier {classifier_name!r} in "
            f"{fit_summary_csv}. Available: {avail}"
        )
    calibrators = calibrators_from_fit_summary(fit_summary, classifier_name)

    p_cal = np.full(len(long_df), np.nan, dtype=np.float64)
    for cls, bc in calibrators.items():
        mask = (long_df["class"].to_numpy() == cls)
        if not mask.any():
            continue
        p_cal[mask] = bc.predict_proba(long_df.loc[mask, "score"].to_numpy())
    return long_df.assign(p=p_cal)


def assemble_output_table(
    long_with_calibration: pd.DataFrame,
    *,
    sequence_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pivot long ``[id, class, score, p]`` → wide per-protein table.

    Column order (fixed): ``id`` → per-class calibrated ``<class>_p`` in
    :data:`_OUTPUT_CLASS_ORDER` → per-class raw ``<class>_raw`` in the
    same order → optional ``sequence``. Classes present in the long
    frame but not in the canonical order are appended in sorted order
    after the canonical block (each side, raw + p). Classes in the
    canonical order but absent from the data are skipped silently.
    """
    if long_with_calibration.empty:
        cols = ["id"] + (["sequence"] if sequence_lookup else [])
        return pd.DataFrame(columns=cols)
    score_wide = long_with_calibration.pivot(
        index="id", columns="class", values="score",
    )
    p_wide = long_with_calibration.pivot(
        index="id", columns="class", values="p",
    )

    present = set(score_wide.columns) & set(p_wide.columns)
    canonical = [c for c in _OUTPUT_CLASS_ORDER if c in present]
    extras = sorted(present - set(canonical))
    ordered_classes = canonical + extras

    p_cols = [f"{c}_p" for c in ordered_classes]
    raw_cols = [f"{c}_raw" for c in ordered_classes]
    p_wide = p_wide[ordered_classes].copy()
    p_wide.columns = p_cols
    score_wide = score_wide[ordered_classes].copy()
    score_wide.columns = raw_cols

    out = pd.concat([p_wide, score_wide], axis=1).reset_index()
    if sequence_lookup is not None:
        out["sequence"] = out["id"].map(sequence_lookup)
    return out
