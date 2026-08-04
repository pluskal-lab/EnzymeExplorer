"""Fit a calibration table for a specific PLM (e.g. esm-2-t36-L36) using the latest
PlmRandomForest + PlmDomainsRandomForest trainings.

Writes a single ``fit_summary.csv`` whose schema matches
``data/calibration_fit_summary.csv`` (consumed by the prediction
pipeline as ``--calibration-csv``). Both classifiers (``PLM`` and
``PLM_Domains``) appear in the same file so prediction can pick the
right row by ``classifier`` name regardless of which PLM was used.

Usage:
    python scripts/fit_calibration_for_plm.py \\
        --plm esm-2-t36-L36 \\
        --output data/dark_candidates/bundles/calibration_esm-2-t36-L36.csv \\
        --plm-rf-dir outputs/models/PlmRandomForest/esm-2-t36-L36/all_folds/all_classes/<ts> \\
        --plm-dom-rf-dir outputs/models/PlmDomainsRandomForest/esm-2-t36-L36/all_folds/all_classes/<ts>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore

from enzymeexplorer.src.evaluation.calibration import (
    build_oof_frame,
    fit_calibration_table,
)
from enzymeexplorer.src.evaluation.io import (
    folds_to_dfs,
    load_pickle_folds,
)
from enzymeexplorer.src.evaluation.clusters import load_cluster_map

logger = logging.getLogger(__name__)

CLASSES = ["TPS", "IDS", "FPP", "GPP", "GGPP", "EDSQ", "CPP", "GFPP"]


def _build_oof_map(experiment_dir: Path, classifier_label: str):
    raws = load_pickle_folds(experiment_dir, n_folds=5)
    fold_dfs = folds_to_dfs(raws, classes_subset=CLASSES)
    return {
        cls: build_oof_frame(fold_dfs, cls, classifier_label) for cls in CLASSES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plm", required=True)
    parser.add_argument("--plm-rf-dir", type=Path, required=True)
    parser.add_argument("--plm-dom-rf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=Path("data/EnzymeExplorer_Dataset_clusters_50.tsv"),
        help="Cluster TSV from scripts/one_offs/build_eval_clusters.py — "
             "used for the cluster-block bootstrap ribbon.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    oof_per_clf = {
        "PLM_Domains": _build_oof_map(args.plm_dom_rf_dir, "PLM_Domains"),
        "PLM":         _build_oof_map(args.plm_rf_dir, "PLM"),
    }
    cluster_map = load_cluster_map(args.cluster_map)
    artefacts = fit_calibration_table(
        oof_per_clf, bootstrap_unit="cluster", cluster_map=cluster_map,
    )
    fit_df = artefacts.fit_summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(args.output, index=False)
    logger.info(
        "Wrote %s (%d fit rows, %d non-skip)",
        args.output, len(fit_df),
        int((fit_df["status"].str.startswith("fit", na=False)).sum()),
    )


if __name__ == "__main__":
    main()
