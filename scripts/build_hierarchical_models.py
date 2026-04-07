#!/usr/bin/env python3
"""Build hierarchical PlmRF / PlmDomainsRF substrate prediction models.

The hierarchical approach uses two stages:

    Stage 1 (detector):  A classifier trained on ALL data (TPS + negatives)
        to estimate P(TPS).  This is the existing model's isTPS output.

    Stage 2 (substrate): A new Random Forest trained on TPS-ONLY data to
        estimate P(substrate | TPS).

    Combined:  P(substrate) = P(TPS) × P(substrate | TPS)

For each existing PlmRF / PlmDomainsRF experiment the script:

1. Loads the existing fold results to obtain P(TPS) from Stage 1.
2. Retrains a TPS-only substrate RF using the same embeddings.
3. Multiplies the two probability arrays.
4. Saves the result under ``outputs/{Model}Hierarchical/{version}/...``.

Usage::

    conda run -n terpene_miner python scripts/build_hierarchical_models.py

    # Specific base model and tracks:
    conda run -n terpene_miner python scripts/build_hierarchical_models.py \\
        --base-model PlmRandomForest \\
        --tracks tps_esm-1v-subseq_new_dataset tps_esm-1v-subseq_synced_folds
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs")
CONFIG_ROOT = Path("enzymeexplorer/configs")

_NON_TPS_LABELS = frozenset({"Unknown", "precursor substr"})
_PRECURSOR_TYPES = frozenset({"ggpps", "fpps", "gpps", "gfpps", "hsqs"})
_DEFAULT_TYPE_COL = "Type (mono, sesq, di, …)"


@dataclass
class TrackSpec:
    """Specification for a single experiment track."""

    base_model: str
    version: str
    train_csv: str
    train_emb_h5: str
    id_col: str
    split_col: str
    target_col: str
    class_names: list[str]
    type_col: str
    eval_csv: Optional[str] = None
    eval_emb_h5: Optional[str] = None
    eval_id_col: Optional[str] = None
    eval_split_col: Optional[str] = None


def _resolve_config(base_model: str, version: str) -> dict:
    """Load and resolve a config YAML (with include support)."""
    cfg_path = CONFIG_ROOT / base_model / version / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    if "include" in cfg:
        parent_path = (cfg_path.parent / cfg.pop("include")).resolve()
        with open(parent_path, "r") as fh:
            parent = yaml.safe_load(fh) or {}
        parent.update(cfg)
        cfg = parent
    return cfg


def build_track_spec(base_model: str, version: str) -> TrackSpec:
    """Create a TrackSpec from a model config."""
    cfg = _resolve_config(base_model, version)
    return TrackSpec(
        base_model=base_model,
        version=version,
        train_csv=cfg["tps_cleaned_csv_path"],
        train_emb_h5=cfg["representations_path"],
        id_col=cfg.get("id_col_name", "Uniprot ID"),
        split_col=cfg.get("split_col_name", "Fold"),
        target_col=cfg.get(
            "target_col_name", "SMILES_substrate_canonical_no_stereo"
        ),
        class_names=cfg["class_names"],
        type_col=cfg.get("type_col_name", _DEFAULT_TYPE_COL),
        eval_csv=cfg.get("eval_csv_path"),
        eval_emb_h5=cfg.get("eval_representations_path"),
        eval_id_col=cfg.get("eval_id_col_name"),
        eval_split_col=cfg.get("eval_split_col_name"),
    )


def _normalize_fold_column(df: pd.DataFrame, col: str) -> None:
    """Ensure fold column uses ``fold_N`` format."""
    vals = df[col].dropna().astype(str)
    if vals.empty or vals.str.startswith("fold_").any():
        return
    mask = df[col].notna()
    df[col] = df[col].astype(object)
    df.loc[mask, col] = "fold_" + df.loc[mask, col].astype(int).astype(str)


def _assign_is_tps(label_set: set[str]) -> set[str]:
    if label_set.issubset(_NON_TPS_LABELS):
        return label_set
    return label_set | {"isTPS"}


def _get_latest_fold_dir(base_model: str, version: str) -> Path:
    """Find the latest timestamp directory for a model/version."""
    root = OUTPUT_ROOT / base_model / version / "all_folds" / "all_classes"
    if not root.exists():
        raise FileNotFoundError(f"No results found at {root}")
    ts_dirs = sorted(root.iterdir())
    if not ts_dirs:
        raise FileNotFoundError(f"No timestamp dirs in {root}")
    return ts_dirs[-1]


def load_embeddings(h5_path: str, id_col: str) -> pd.DataFrame:
    """Load embeddings from HDF5 and normalize columns."""
    emb_df = pd.read_hdf(h5_path)
    emb_df.columns = [id_col, "Emb"]
    emb_df.drop_duplicates(subset=[id_col], inplace=True)
    return emb_df


def train_tps_only_substrate_rf(
    train_ids: set[str],
    data_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    spec: TrackSpec,
    substrate_classes: list[str],
    random_state: int = 0,
) -> tuple[RandomForestClassifier, MultiLabelBinarizer]:
    """Train a RF on TPS-only data for substrate prediction.

    Returns the trained classifier and the label binarizer.
    """
    trn_df = data_df[data_df[spec.id_col].isin(train_ids)].copy()

    trn_df = (
        trn_df.groupby(spec.id_col)[spec.target_col].agg(set).reset_index()
    )
    trn_df[spec.target_col] = trn_df[spec.target_col].map(_assign_is_tps)

    tps_mask = trn_df[spec.target_col].map(
        lambda x: not x.issubset(_NON_TPS_LABELS)
    )
    trn_tps = trn_df[tps_mask].copy()

    trn_tps = trn_tps.merge(emb_df, on=spec.id_col, how="inner")
    if len(trn_tps) == 0:
        raise ValueError("No TPS training samples after merging with embeddings")

    binarizer = MultiLabelBinarizer(classes=substrate_classes)
    y_train = binarizer.fit_transform(trn_tps[spec.target_col].values)

    x_train = np.stack(trn_tps["Emb"].values)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=1000,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)
    logger.info(
        "  Trained TPS-only substrate RF: %d samples, %d features, %d classes",
        x_train.shape[0],
        x_train.shape[1],
        len(substrate_classes),
    )
    return clf, binarizer


def predict_substrate_proba(
    clf: RandomForestClassifier,
    test_ids: list[str],
    emb_df: pd.DataFrame,
    id_col: str,
    n_classes: int,
) -> np.ndarray:
    """Predict substrate probabilities for test proteins."""
    test_emb_df = pd.DataFrame({id_col: test_ids}).merge(
        emb_df, on=id_col, how="left"
    )
    average_emb = np.stack(emb_df["Emb"].values).mean(axis=0)
    test_emb_df["Emb"] = test_emb_df["Emb"].map(
        lambda x: x if isinstance(x, np.ndarray) else average_emb
    )
    test_emb_df = test_emb_df.set_index(id_col).loc[test_ids]
    x_test = np.stack(test_emb_df["Emb"].values)

    y_pred = clf.predict_proba(x_test)
    proba = np.zeros((len(test_ids), n_classes))
    for ci in range(n_classes):
        if isinstance(y_pred, list):
            proba[:, ci] = y_pred[ci][:, -1]
        else:
            proba[:, ci] = y_pred[:, ci]
    return proba


def process_track(spec: TrackSpec, n_folds: int = 5) -> int:
    """Process one track: build hierarchical predictions and save.

    Returns number of folds processed.
    """
    src_fold_dir = _get_latest_fold_dir(spec.base_model, spec.version)
    hier_model_name = f"{spec.base_model}Hierarchical"
    dst_root = (
        OUTPUT_ROOT
        / hier_model_name
        / spec.version
        / "all_folds"
        / "all_classes"
        / src_fold_dir.name
    )
    dst_root.mkdir(parents=True, exist_ok=True)

    data_df = pd.read_csv(spec.train_csv)
    _normalize_fold_column(data_df, spec.split_col)

    if spec.type_col in data_df.columns:
        data_df.loc[
            data_df[spec.type_col].isin(_PRECURSOR_TYPES),
            "SMILES_substrate_canonical_no_stereo",
        ] = "precursor substr"

    is_cross = bool(spec.eval_csv)

    eval_data_df = None
    if is_cross:
        eval_data_df = pd.read_csv(spec.eval_csv)
        renames = {}
        if spec.eval_id_col and spec.eval_id_col != spec.id_col:
            renames[spec.eval_id_col] = spec.id_col
        if spec.eval_split_col and spec.eval_split_col != spec.split_col:
            renames[spec.eval_split_col] = spec.split_col
        if renames:
            eval_data_df.rename(columns=renames, inplace=True)
        _normalize_fold_column(eval_data_df, spec.split_col)
        if spec.type_col in eval_data_df.columns:
            eval_data_df.loc[
                eval_data_df[spec.type_col].isin(_PRECURSOR_TYPES),
                "SMILES_substrate_canonical_no_stereo",
            ] = "precursor substr"

    train_emb_df = load_embeddings(spec.train_emb_h5, spec.id_col)
    eval_emb_df = (
        load_embeddings(spec.eval_emb_h5, spec.id_col)
        if is_cross and spec.eval_emb_h5
        else train_emb_df
    )

    substrate_classes = [c for c in spec.class_names if c != "isTPS"]
    if not isinstance(spec.class_names, list):
        spec.class_names = list(spec.class_names)
    istps_idx = (
        spec.class_names.index("isTPS") if "isTPS" in spec.class_names else -1
    )
    if istps_idx < 0:
        logger.error("No isTPS in class_names for %s", spec.version)
        return 0

    fold_re = re.compile(r"fold_(\d+)_results\.pkl$")
    folds = sorted(
        {
            m.group(1)
            for f in src_fold_dir.glob("fold_*_results.pkl")
            if (m := fold_re.search(f.name))
        }
    )

    processed = 0
    for fold_str in folds:
        pkl_name = f"fold_{fold_str}_results.pkl"
        src_path = src_fold_dir / pkl_name

        with open(src_path, "rb") as fh:
            val_proba_np, class_names_arr, test_df = pickle.load(fh)

        class_names_list = (
            list(class_names_arr)
            if not isinstance(class_names_arr, list)
            else class_names_arr
        )
        p_tps = val_proba_np[:, class_names_list.index("isTPS")]

        all_folds = sorted(
            data_df[spec.split_col].dropna().unique(), key=str
        )
        train_folds = [f for f in all_folds if f != f"fold_{fold_str}"]
        train_mask = data_df[spec.split_col].isin(set(train_folds))
        train_ids = set(data_df.loc[train_mask, spec.id_col].values)

        clf, binarizer = train_tps_only_substrate_rf(
            train_ids=train_ids,
            data_df=data_df,
            emb_df=train_emb_df,
            spec=spec,
            substrate_classes=substrate_classes,
            random_state=0,
        )

        id_col_test = (
            "ID" if "ID" in test_df.columns else spec.id_col
        )
        test_ids = list(test_df[id_col_test].values)

        p_substrate = predict_substrate_proba(
            clf, test_ids, eval_emb_df, spec.id_col, len(substrate_classes)
        )

        combined = np.zeros_like(val_proba_np)
        sub_idx = 0
        for ci, cn in enumerate(class_names_list):
            if cn == "isTPS":
                combined[:, ci] = p_tps
            else:
                combined[:, ci] = p_tps * p_substrate[:, sub_idx]
                sub_idx += 1

        dst_path = dst_root / pkl_name
        with open(dst_path, "wb") as fh:
            pickle.dump((combined, class_names_arr, test_df), fh)

        logger.info(
            "  fold %s: P(TPS) mean=%.3f, P(substrate) max_mean=%.3f -> combined max_mean=%.3f",
            fold_str,
            p_tps.mean(),
            p_substrate.max(axis=1).mean(),
            combined[:, :istps_idx].max(axis=1).mean(),
        )
        processed += 1

    return processed


def create_configs(base_model: str, tracks: list[str]) -> None:
    """Copy base model configs to hierarchical model config dirs."""
    hier_model = f"{base_model}Hierarchical"
    for version in tracks:
        src = CONFIG_ROOT / base_model / version / "config.yaml"
        if not src.exists():
            continue
        dst_dir = CONFIG_ROOT / hier_model / version
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "config.yaml"
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.info("Created config: %s", dst)


def discover_tracks(base_model: str) -> list[str]:
    """Find all tracks with existing fold results for a base model."""
    model_out = OUTPUT_ROOT / base_model
    if not model_out.exists():
        return []
    tracks = []
    for d in sorted(model_out.iterdir()):
        if d.is_dir() and (d / "all_folds").exists():
            tracks.append(d.name)
    return tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchical PlmRF/PlmDomainsRF models"
    )
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=["PlmRandomForest", "PlmDomainsRandomForest"],
        help="Base model types to build hierarchical versions for",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="Specific track versions to process (default: auto-discover)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for base_model in args.base_models:
        tracks = args.tracks or discover_tracks(base_model)
        if not tracks:
            logger.warning("No tracks found for %s", base_model)
            continue

        logger.info("=== %s: %d tracks ===", base_model, len(tracks))
        for version in tracks:
            logger.info("--- %s / %s ---", base_model, version)
            try:
                spec = build_track_spec(base_model, version)
            except FileNotFoundError as exc:
                logger.warning("  Skipping: %s", exc)
                continue

            try:
                n = process_track(spec, n_folds=args.n_folds)
                logger.info("  Processed %d folds", n)
            except Exception as exc:
                logger.error("  Failed: %s", exc, exc_info=True)
                continue

        create_configs(base_model, tracks)

    logger.info("Done.")


if __name__ == "__main__":
    main()
