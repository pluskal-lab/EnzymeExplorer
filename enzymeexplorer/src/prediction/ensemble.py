"""Fold-ensemble prediction.

Both flavours load a list of fitted classifiers from a
``data/enzyme_explorer*_checkpoints.pkl`` bundle (see
``scripts/bundle_fold_checkpoints.py``) and call each classifier's native
``predict_proba(val_df)``.

The trained models look up features by protein ID from their own internal
stores rather than accepting raw feature arrays:

* ``PlmRandomForest`` reads from ``self.features_df`` (rows
  ``[id_col_name, "Emb"]`` where ``Emb`` is a 1-D PLM embedding).
* ``PlmDomainsRandomForest`` additionally reads from ``self.feats_dom_dists``
  (a list of 1-D ``1 - tmscore`` rows), ``self.all_ids_list_dom`` (their IDs),
  and ``self.allowed_feat_indices`` (column subset selected at training time).

To score new proteins we temporarily replace those stores with the prediction
data, call ``predict_proba``, and restore them — so the loaded classifiers
remain reusable across calls. ``predict_proba`` returns a 2-D array of shape
``[n_val, n_classes]`` aligned to ``classifier.config.class_names``.
"""

from __future__ import annotations

import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from enzymeexplorer.src.evaluation.classes import SMILES_TO_SHORT

logger = logging.getLogger(__name__)


def _short_class_name(raw_name: str) -> str:
    """Translate model output class names (raw SMILES, ``"isTPS"``,
    ``"precursor substr"``) into the plot/tier-friendly short names
    (``"FPP"``, ``"TPS"``, ``"IDS"``, …) used everywhere downstream.

    Pass-through if the name is already short or unmapped — this keeps the
    function safe for class names we haven't catalogued yet.
    """
    return SMILES_TO_SHORT.get(raw_name, raw_name)


def load_fold_bundle(path: str | Path) -> list[Any]:
    """Load the pickled list of fold classifiers."""
    with Path(path).open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, list):
        raise ValueError(
            f"Bundle at {path} is not a list (got {type(bundle).__name__}). "
            f"Expected the list-of-classifiers format produced by "
            f"scripts/bundle_fold_checkpoints.py."
        )
    return bundle


def _build_plm_features_df(
    embeddings: np.ndarray,
    protein_ids: list[str],
    id_col_name: str,
) -> pd.DataFrame:
    """Turn an embeddings matrix + IDs into the ``[id_col, Emb]`` frame the
    models look up by ID."""
    return pd.DataFrame(
        {id_col_name: protein_ids, "Emb": list(embeddings)}
    )


@contextmanager
def _swapped_attributes(obj: Any, **overrides: Any) -> Iterator[None]:
    """Temporarily set attributes on ``obj``, restoring them on exit."""
    sentinel = object()
    saved: dict[str, Any] = {}
    try:
        for name, value in overrides.items():
            saved[name] = getattr(obj, name, sentinel)
            setattr(obj, name, value)
        yield
    finally:
        for name, prev in saved.items():
            if prev is sentinel:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, prev)


def _proba_to_long(
    proba: np.ndarray,
    class_names: list[str],
    ids: list[str],
    fold_index: int,
) -> pd.DataFrame:
    """``[N, C]`` proba → long-form ``[id, fold, class, score]``."""
    if proba.ndim != 2 or proba.shape != (len(ids), len(class_names)):
        raise ValueError(
            f"Unexpected predict_proba shape {proba.shape}; "
            f"expected ({len(ids)}, {len(class_names)})."
        )
    rows = []
    for class_i, class_name in enumerate(class_names):
        short_name = _short_class_name(class_name)
        scores = proba[:, class_i]
        for sample_i, score in enumerate(scores):
            rows.append(
                {
                    "id": ids[sample_i],
                    "fold": fold_index,
                    "class": short_name,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def predict_with_plm_and_domains(
    *,
    embeddings: np.ndarray,
    protein_ids: list[str],
    structural_features: np.ndarray,
    structural_features_ids: list[str],
    fold_classifiers: list[Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Score with PLM + domain features.

    Routing rule: proteins present in ``structural_features_ids`` (i.e. ones
    that produced at least one detected TPS-family domain) are scored by
    every fold classifier via its native ``predict_proba(val_df)``.
    Proteins not in ``structural_features_ids`` — those for which the domain
    detector returned nothing — are returned in ``fallback_ids`` so the caller
    can re-score them with a PLM-only ensemble.
    """
    if embeddings.shape[0] != len(protein_ids):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != protein_ids "
            f"({len(protein_ids)})"
        )

    sf_id_to_row = {pid: idx for idx, pid in enumerate(structural_features_ids)}

    keep_ids: list[str] = []
    fallback_ids: list[str] = []
    keep_dom_rows: list[np.ndarray] = []
    for pid in protein_ids:
        sf_idx = sf_id_to_row.get(pid)
        if sf_idx is None:
            fallback_ids.append(pid)
            continue
        keep_ids.append(pid)
        keep_dom_rows.append(structural_features[sf_idx])

    if not keep_ids:
        return (
            pd.DataFrame(columns=["id", "fold", "class", "score"]),
            fallback_ids,
        )

    id_to_row = {pid: idx for idx, pid in enumerate(protein_ids)}
    plm_keep = embeddings[[id_to_row[pid] for pid in keep_ids]]

    frames: list[pd.DataFrame] = []
    val_df_template_cache: dict[str, pd.DataFrame] = {}
    for fold_i, clf in enumerate(
        tqdm(fold_classifiers, desc="Predicting (PLM+domains, per fold)")
    ):
        id_col = clf.config.id_col_name
        class_names = list(clf.config.class_names)

        plm_features_df = _build_plm_features_df(plm_keep, keep_ids, id_col)
        val_df = val_df_template_cache.setdefault(
            id_col, pd.DataFrame({id_col: keep_ids})
        )

        with _swapped_attributes(
            clf,
            features_df_plm=plm_features_df,
            feats_dom_dists=keep_dom_rows,
            all_ids_list_dom=list(keep_ids),
        ):
            proba = clf.predict_proba(val_df)

        frames.append(_proba_to_long(proba, class_names, keep_ids, fold_i))
    return pd.concat(frames, ignore_index=True), fallback_ids


def predict_with_plm_only(
    *,
    embeddings: np.ndarray,
    protein_ids: list[str],
    fold_classifiers: list[Any],
) -> pd.DataFrame:
    """Score with PLM features only via each fold's native
    ``predict_proba(val_df)``."""
    if embeddings.shape[0] != len(protein_ids):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != protein_ids "
            f"({len(protein_ids)})"
        )
    if not protein_ids:
        return pd.DataFrame(columns=["id", "fold", "class", "score"])

    frames: list[pd.DataFrame] = []
    val_df_cache: dict[str, pd.DataFrame] = {}
    for fold_i, clf in enumerate(
        tqdm(fold_classifiers, desc="Predicting (PLM-only, per fold)")
    ):
        id_col = clf.config.id_col_name
        class_names = list(clf.config.class_names)

        features_df = _build_plm_features_df(embeddings, protein_ids, id_col)
        val_df = val_df_cache.setdefault(
            id_col, pd.DataFrame({id_col: protein_ids})
        )

        with _swapped_attributes(clf, features_df=features_df):
            proba = clf.predict_proba(val_df)
        frames.append(_proba_to_long(proba, class_names, protein_ids, fold_i))
    return pd.concat(frames, ignore_index=True)


def average_over_folds(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """Average per-fold scores → long-form ``[id, class, score]``."""
    if per_fold_df.empty:
        return pd.DataFrame(columns=["id", "class", "score"])
    return (
        per_fold_df.groupby(["id", "class"], as_index=False)["score"].mean()
    )  # type: ignore


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot ``[id, class, score]`` → ``id × class`` wide form."""
    if long_df.empty:
        return pd.DataFrame()
    return long_df.pivot(index="id", columns="class", values="score").reset_index()
