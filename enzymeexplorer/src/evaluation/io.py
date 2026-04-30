"""I/O for the evaluation pipeline.

Loads per-fold prediction pickles produced by ``experiment_runner.run_experiment``
and assembles them into per-fold ``(labels_df, preds_df)`` pairs keyed by the
short class names defined in ``classes.py``. Also resolves the latest experiment
timestamp on disk and builds the per-classifier × per-class nested mapping
consumed by the bootstrap module.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.classes import (
    ALL_CLASSES,
    SHORT_TO_SMILES,
    SMILES_TO_SHORT,
)
from enzymeexplorer.src.utils.project_info import get_output_root

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")
_TARGET_COL = "SMILES_substrate_canonical_no_stereo"

FoldRaw = tuple[np.ndarray, list[str], pd.DataFrame]
FoldDfs = tuple[pd.DataFrame, pd.DataFrame]


def latest_experiment_dir(
    model: str,
    version: str,
    *,
    output_root: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Return the timestamp directory for a (model, version) experiment.

    Looks under ``<output_root>/<model>/<version>/all_folds/all_classes/``.
    If ``timestamp`` is given, returns that subdirectory directly; otherwise
    picks the lexicographically (and therefore chronologically) latest entry.
    """
    root = (
        (output_root or get_output_root())
        / model
        / version
        / "all_folds"
        / "all_classes"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment results dir not found: {root}")
    if timestamp is not None:
        target = root / timestamp
        if not target.is_dir():
            raise FileNotFoundError(f"Timestamped experiment dir not found: {target}")
        return target
    candidates = sorted(
        p for p in root.iterdir() if p.is_dir() and _TIMESTAMP_RE.match(p.name)
    )
    if not candidates:
        raise FileNotFoundError(f"No timestamped experiment runs under {root}")
    return candidates[-1]


def load_pickle_folds(experiment_dir: Path, n_folds: int = 5) -> list[FoldRaw]:
    """Load ``fold_{i}_results.pkl`` tuples from a flat experiment directory."""
    raws: list[FoldRaw] = []
    for fold_i in range(n_folds):
        path = experiment_dir / f"fold_{fold_i}_results.pkl"
        with open(path, "rb") as fh:
            val_proba_np, class_names, test_df = pickle.load(fh)
        raws.append((np.asarray(val_proba_np), list(class_names), test_df))
    return raws


def folds_to_dfs(
    raws: list[FoldRaw],
    classes_subset: Iterable[str] | None = None,
    seq_ids: set[str] | None = None,
) -> dict[int, FoldDfs]:
    """Convert raw per-fold tuples to per-fold ``(labels_df, preds_df)``.

    Each DataFrame has an ``ID`` column followed by one column per requested
    short class (in input order). Predictions are clipped to ``[0, 1]``.
    Raises ``KeyError`` if any requested class is absent from the pickled
    ``class_names`` of any fold.
    """
    requested = list(classes_subset) if classes_subset is not None else ALL_CLASSES
    requested_smiles = [SHORT_TO_SMILES[c] for c in requested]

    out: dict[int, FoldDfs] = {}
    for fold_idx, (val_proba_np, class_names, test_df) in enumerate(raws):
        smiles_to_col = {smiles: idx for idx, smiles in enumerate(class_names)}
        missing = [
            short
            for short, smiles in zip(requested, requested_smiles)
            if smiles not in smiles_to_col
        ]
        if missing:
            raise KeyError(
                f"Fold {fold_idx} pickle missing requested classes: {missing}"
            )
        ordered_cols = [smiles_to_col[s] for s in requested_smiles]

        df = test_df.reset_index(drop=True)
        proba = np.asarray(val_proba_np)
        if seq_ids is not None:
            mask = df["ID"].isin(seq_ids).to_numpy()
            df = df.loc[mask].reset_index(drop=True)
            proba = proba[mask]

        target_sets = df[_TARGET_COL]
        labels_cols: dict[str, np.ndarray] = {}
        for short, smiles in zip(requested, requested_smiles):
            labels_cols[short] = target_sets.map(
                lambda s, sm=smiles: int(sm in s)
            ).to_numpy(dtype=np.int8)
        preds_arr = np.minimum(proba[:, ordered_cols], 1.0)
        preds_cols = {short: preds_arr[:, j] for j, short in enumerate(requested)}

        labels_df = pd.DataFrame({"ID": df["ID"].to_numpy(), **labels_cols})
        preds_df = pd.DataFrame({"ID": df["ID"].to_numpy(), **preds_cols})
        out[fold_idx] = (labels_df, preds_df)
    return out


def load_id_metadata(
    csv_path: Path | str,
    columns: list[str],
    *,
    id_col: str = "ID",
) -> dict[str, dict[str, str]]:
    """Return ``{column: {ID: value}}`` for each requested metadata column.

    Reads the cleaned dataset CSV once. Duplicate IDs are collapsed to their
    first non-null value per column.
    """
    df = pd.read_csv(csv_path, usecols=[id_col, *columns])
    df = df.drop_duplicates(subset=[id_col])
    out: dict[str, dict[str, str]] = {}
    for col in columns:
        out[col] = (
            df[[id_col, col]].dropna().set_index(id_col)[col].astype(str).to_dict()
        )
    return out


def load_classifier_class_fold_dfs(
    model: str,
    version_spec: str | Mapping[str, str],
    *,
    classes: Iterable[str] | None = None,
    seq_ids: set[str] | None = None,
    output_root: Path | None = None,
    n_folds: int = 5,
    timestamp: str | None = None,
) -> dict[str, dict[int, FoldDfs]]:
    """Build the ``{class_short: {fold_idx: (labels_df, preds_df)}}`` mapping
    for a single classifier.

    ``version_spec`` is either a single version string (one experiment, all
    classes share the same per-fold DFs) or a ``{class_short: version_str}``
    mapping (HBI per-class optimal: each class loads from its own experiment).
    The single-version case shares DataFrame references across class keys.
    """
    selected = list(classes) if classes is not None else ALL_CLASSES
    out: dict[str, dict[int, FoldDfs]] = {}

    if isinstance(version_spec, str):
        exp_dir = latest_experiment_dir(
            model, version_spec, output_root=output_root, timestamp=timestamp
        )
        raws = load_pickle_folds(exp_dir, n_folds=n_folds)
        per_fold = folds_to_dfs(raws, classes_subset=selected, seq_ids=seq_ids)
        for short in selected:
            out[short] = per_fold
        return out

    missing_classes = set(selected) - set(version_spec)
    if missing_classes:
        raise KeyError(
            f"version_spec missing classes: {sorted(missing_classes)}"
        )
    for short in selected:
        version = version_spec[short]
        exp_dir = latest_experiment_dir(model, version, output_root=output_root)
        raws = load_pickle_folds(exp_dir, n_folds=n_folds)
        out[short] = folds_to_dfs(raws, classes_subset=[short], seq_ids=seq_ids)
    return out
