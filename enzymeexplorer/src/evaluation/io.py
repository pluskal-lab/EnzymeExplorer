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
from enzymeexplorer.src.utils.project_info import get_models_output_root

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")
_TARGET_COL = "SMILES_substrate_canonical_no_stereo"

FoldRaw = tuple[np.ndarray, list[str], pd.DataFrame]
FoldDfs = tuple[pd.DataFrame, pd.DataFrame]


def _has_complete_folds(experiment_dir: Path, n_folds: int) -> bool:
    return all(
        (experiment_dir / f"fold_{i}_results.pkl").exists() for i in range(n_folds)
    )


def latest_experiment_dir(
    model: str,
    version: str,
    *,
    output_root: Path | None = None,
    timestamp: str | None = None,
    n_folds: int = 5,
) -> Path:
    """Return the timestamp directory for a (model, version) experiment.

    Looks under ``<output_root>/<model>/<version>/all_folds/all_classes/``.
    If ``timestamp`` is given, returns that subdirectory directly; otherwise
    picks the most recent run with all ``n_folds`` ``fold_N_results.pkl``
    pickles present. If only the latest run is incomplete (training in
    progress) the previous complete run is used and a warning is logged;
    if no run has all folds, the lexicographically latest is returned and
    a warning is logged so downstream loaders raise the precise missing
    fold error.
    """
    root = (
        (output_root or get_models_output_root())
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
    complete = [p for p in candidates if _has_complete_folds(p, n_folds)]
    if complete:
        if complete[-1] != candidates[-1]:
            logger.warning(
                "%s/%s: latest run %s is incomplete; falling back to %s",
                model, version, candidates[-1].name, complete[-1].name,
            )
        return complete[-1]
    logger.warning(
        "%s/%s: no run has all %d folds; returning latest (%s) for diagnosis",
        model, version, n_folds, candidates[-1].name,
    )
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
    kingdom_cache: Path | str | None = None,
    kingdom_col: str = "Kingdom",
) -> dict[str, dict[str, str]]:
    """Return ``{column: {ID: value}}`` for each requested metadata column.

    Reads the cleaned dataset CSV once. Duplicate IDs are collapsed to their
    first non-null value per column.

    When ``kingdom_cache`` points to a JSON file produced by
    ``scripts/fetch_kingdom_for_uniprot.py`` (``{accession: kingdom}``), its
    entries override CSV ``Kingdom=Unknown`` rows so distractor sequences
    contribute to the per-Kingdom evaluation. Distractor accessions absent
    from the cache stay ``Unknown`` and are filtered out by the
    in-category-only bootstrap mask.
    """
    df = pd.read_csv(csv_path, usecols=[id_col, *columns])
    df = df.drop_duplicates(subset=[id_col])
    out: dict[str, dict[str, str]] = {}
    for col in columns:
        out[col] = (
            df[[id_col, col]].dropna().set_index(id_col)[col].astype(str).to_dict()
        )
    if kingdom_cache and kingdom_col in out:
        cache_path = Path(kingdom_cache)
        if cache_path.exists():
            import json
            with open(cache_path, "r", encoding="utf-8") as fh:
                resolved: dict[str, str] = json.load(fh)
            kingdom_map = out[kingdom_col]
            n_added = 0
            n_overridden = 0
            for acc, k in resolved.items():
                if k == "Unknown":
                    continue
                cur = kingdom_map.get(acc)
                if cur is None:
                    n_added += 1
                elif cur == "Unknown":
                    n_overridden += 1
                else:
                    continue
                kingdom_map[acc] = k
            logger.info(
                "Kingdom cache merged: +%d new, %d overridden from Unknown",
                n_added, n_overridden,
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
    timestamps_per_class: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[int, FoldDfs]], dict[str, str]]:
    """Build the ``{class_short: {fold_idx: (labels_df, preds_df)}}`` mapping
    for a single classifier, plus a ``{class_short: experiment_timestamp}``
    map captured at load time.

    ``version_spec`` is either a single version string (one experiment, all
    classes share the same per-fold DFs) or a ``{class_short: version_str}``
    mapping (homology per-class optimal: each class loads from its own experiment).
    The single-version case shares DataFrame references across class keys.

    Pinning to specific timestamps:

    * ``timestamp``: only meaningful for the single-version case; selects
      that exact timestamped run for every class.
    * ``timestamps_per_class``: ``{class_short: timestamp_dir_name}`` —
      used for both single- and per-class version specs to pin individual
      timestamps. Required when the latest training run is incomplete and
      a previous run's resolved_versions.yaml should be reproduced
      exactly. If a class has no entry, falls back to the
      newest-complete run.
    """
    selected = list(classes) if classes is not None else ALL_CLASSES
    out: dict[str, dict[int, FoldDfs]] = {}
    timestamps: dict[str, str] = {}
    ts_map = dict(timestamps_per_class) if timestamps_per_class else {}

    if isinstance(version_spec, str):
        # Single-version case. If timestamps_per_class is given and pins
        # all selected classes to the same timestamp, honour it; otherwise
        # use the explicit ``timestamp`` arg, otherwise let
        # ``latest_experiment_dir`` pick the newest complete run.
        pinned_ts = timestamp
        per_class_pins = {ts_map[c] for c in selected if c in ts_map}
        if pinned_ts is None and per_class_pins:
            if len(per_class_pins) > 1:
                raise ValueError(
                    f"Single-version spec for {model}/{version_spec} but "
                    f"timestamps_per_class disagrees across classes: "
                    f"{sorted(per_class_pins)}"
                )
            pinned_ts = next(iter(per_class_pins))
        exp_dir = latest_experiment_dir(
            model, version_spec, output_root=output_root, timestamp=pinned_ts
        )
        raws = load_pickle_folds(exp_dir, n_folds=n_folds)
        per_fold = folds_to_dfs(raws, classes_subset=selected, seq_ids=seq_ids)
        for short in selected:
            out[short] = per_fold
            timestamps[short] = exp_dir.name
        return out, timestamps

    missing_classes = set(selected) - set(version_spec)
    if missing_classes:
        raise KeyError(
            f"version_spec missing classes: {sorted(missing_classes)}"
        )
    for short in selected:
        version = version_spec[short]
        exp_dir = latest_experiment_dir(
            model, version, output_root=output_root,
            timestamp=ts_map.get(short),
        )
        raws = load_pickle_folds(exp_dir, n_folds=n_folds)
        out[short] = folds_to_dfs(raws, classes_subset=[short], seq_ids=seq_ids)
        timestamps[short] = exp_dir.name
    return out, timestamps
