#!/usr/bin/env python3
"""Post-process CLEAN results to compute proportion-based P(TPS).

Instead of using max-confidence across TPS-matched EC predictions (the
original approach), this script recomputes isTPS as:

    P(TPS) = sum(conf_i * is_tps_ec_i) / sum(conf_i)

where ``is_tps_ec_i`` is 1 when the predicted EC number maps to a known
TPS substrate (determined by CLEAN's own Rhea-based EC→substrate lookup,
using Indigo canonicalization to match substrate SMILES), and ``conf_i``
is the maxsep distance/confidence returned by CLEAN.

The new fold results are saved under
``outputs/CLEANBetterDetection/{version}/...`` so that the existing
evaluation pipeline can pick them up as a separate model.

Usage::

    conda run -n terpene_miner python scripts/postprocess_clean_tps_detection.py

To only process a specific track::

    conda run -n terpene_miner python scripts/postprocess_clean_tps_detection.py \\
        --tracks new_dataset synced_folds
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import MolToSmiles, rdChemReactions  # type: ignore

from enzymeexplorer.src.utils.data import get_canonical_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLEAN_RESULTS_DIR = Path("/home/samusevich/CLEAN/app/results")
CLEAN_OUTPUT_ROOT = Path("outputs/CLEAN")
NEW_MODEL_NAME = "CLEANBetterDetection"
NEW_OUTPUT_ROOT = Path("outputs") / NEW_MODEL_NAME

CLEAN_WORKING_DIR = Path("_clean_working_dir")

ALL_TRACKS = [
    "with_minor_reactions_phylo_folds",
    "with_minor_reactions",
    "synced_folds",
    "new_dataset",
    "cross_synced_to_new",
    "cross_new_tps_old_neg",
]

DATASET_PATHS = {
    "with_minor_reactions_phylo_folds": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "with_minor_reactions": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "synced_folds": "data/TPS-Nov19_2023_with_synced_folds.csv",
    "new_dataset": "data/EnzymeExplorer_Dataset.csv",
    "cross_synced_to_new": "data/EnzymeExplorer_Dataset.csv",
    "cross_new_tps_old_neg": "data/EnzymeExplorer_Dataset.csv",
}

TARGET_COL = "SMILES_substrate_canonical_no_stereo"


def _build_ec_2_substrates(
    working_dir: Path,
) -> dict[str, set[str]]:
    """Replicate CLEAN's Rhea-based EC→substrate mapping.

    Uses Indigo canonicalization (via ``get_canonical_smiles``) to match
    CLEAN's internal logic exactly.
    """
    rhea2ec_df = pd.read_csv(working_dir / "rhea2ec.tsv", sep="\t")
    ec_2_rheaids = rhea2ec_df.groupby("ID")["RHEA_ID"].agg(set).to_dict()

    rhea2smiles_df = pd.read_csv(
        working_dir / "rhea-reaction-smiles.tsv", sep="\t", header=None
    )
    rheaid_2_rxn = rhea2smiles_df[[0, 1]].set_index(0)[1].to_dict()

    rhea2directed_df = pd.read_csv(working_dir / "rhea-directions.tsv", sep="\t")
    master_rhea_2_directed = (
        rhea2directed_df[["RHEA_ID_MASTER", "RHEA_ID_LR"]]
        .set_index("RHEA_ID_MASTER")["RHEA_ID_LR"]
        .to_dict()
    )

    ec_2_substrates: dict[str, set[str]] = defaultdict(set)
    for ec_code in ec_2_rheaids:
        ec_class = f"EC:{ec_code}"
        for rhea_id in ec_2_rheaids[ec_code]:
            if rhea_id in master_rhea_2_directed:
                rhea_id = master_rhea_2_directed[rhea_id]
            if rhea_id in rheaid_2_rxn:
                rxn_smiles = rheaid_2_rxn[rhea_id]
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(
                        rxn_smiles, useSmiles=True
                    )
                    trxn = rdChemReactions.ChemicalReaction(rxn)
                    substrates = trxn.GetReactants()
                    for substr in substrates:
                        canon = get_canonical_smiles(MolToSmiles(substr))
                        ec_2_substrates[ec_class].add(canon)
                except Exception:
                    pass

    logger.info("Built ec_2_substrates: %d ECs total", len(ec_2_substrates))
    return dict(ec_2_substrates)


def build_tps_ec_set(
    working_dir: Path,
    dataset_path: str,
) -> set[str]:
    """Build TPS EC set using CLEAN's own Rhea-based substrate matching.

    An EC is TPS-relevant if its Rhea reaction substrates (Indigo-
    canonicalized) overlap with the TPS substrate SMILES from the dataset.
    This replicates exactly how CLEAN internally decides which EC
    predictions count toward isTPS.
    """
    ec_2_substrates = _build_ec_2_substrates(working_dir)

    data_df = pd.read_csv(dataset_path)
    tps_substrate_smiles = {
        s
        for s in data_df[TARGET_COL].values
        if s not in {"Unknown", "Negative"}
    }
    logger.info(
        "TPS substrate SMILES from %s: %d unique",
        dataset_path,
        len(tps_substrate_smiles),
    )

    tps_ecs: set[str] = set()
    for ec, subs in ec_2_substrates.items():
        if subs.intersection(tps_substrate_smiles):
            tps_ecs.add(ec)

    logger.info(
        "TPS EC set (Rhea-based, Indigo canonical): %d ECs", len(tps_ecs)
    )
    return tps_ecs


def load_all_raw_clean_predictions(
    results_dir: Path,
) -> dict[str, dict[str, float]]:
    """Parse all CLEAN maxsep CSV files into a single map.

    Returns ``{protein_id: {ec_class: confidence, ...}, ...}``.
    Since CLEAN is pre-trained, the same protein always gets the same
    predictions regardless of fold, so we can merge all files.
    """
    id_2_ec_conf: dict[str, dict[str, float]] = {}
    csv_files = sorted(results_dir.glob("*_maxsep.csv"))
    logger.info("Found %d raw CLEAN maxsep CSV files", len(csv_files))

    for csv_path in csv_files:
        with open(csv_path, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                pid = parts[0]
                if pid in id_2_ec_conf:
                    continue
                ec_confs: dict[str, float] = {}
                for ec_part in parts[1:]:
                    ec_class, conf_str = ec_part.split("/")
                    ec_confs[ec_class] = float(conf_str)
                id_2_ec_conf[pid] = ec_confs

    logger.info(
        "Loaded raw EC predictions for %d unique proteins", len(id_2_ec_conf)
    )
    return id_2_ec_conf


def compute_ptps_proportion(
    ec_confs: dict[str, float],
    tps_ecs: set[str],
) -> float:
    """Compute P(TPS) = sum(conf for TPS ECs) / sum(conf for all ECs)."""
    if not ec_confs:
        return 0.0
    total_conf = sum(ec_confs.values())
    if total_conf == 0:
        return 0.0
    tps_conf = sum(c for ec, c in ec_confs.items() if ec in tps_ecs)
    return tps_conf / total_conf


def process_track(
    track: str,
    tps_ecs: set[str],
    id_2_ec_conf: dict[str, dict[str, float]],
    n_folds: int = 5,
) -> int:
    """Post-process CLEAN fold results for one track.

    Returns the number of folds successfully processed.
    """
    track_dir = CLEAN_OUTPUT_ROOT / track / "all_folds" / "all_classes"
    if not track_dir.exists():
        logger.warning("Track dir does not exist: %s", track_dir)
        return 0

    ts_dirs = sorted(track_dir.iterdir())
    if not ts_dirs:
        logger.warning("No timestamp dirs in %s", track_dir)
        return 0
    latest_ts = ts_dirs[-1]

    out_dir = NEW_OUTPUT_ROOT / track / "all_folds" / "all_classes" / latest_ts.name
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for fold_i in range(n_folds):
        pkl_name = f"fold_{fold_i}_results.pkl"
        src_path = latest_ts / pkl_name
        if not src_path.exists():
            logger.warning("Missing %s", src_path)
            continue

        with open(src_path, "rb") as fh:
            val_proba_np, class_names, test_df = pickle.load(fh)

        if not isinstance(class_names, list):
            class_names = list(class_names)

        if "isTPS" not in class_names:
            logger.warning("No isTPS class in %s", src_path)
            continue
        istps_idx = class_names.index("isTPS")

        id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
        ids = test_df[id_col].values

        new_istps = np.zeros(len(ids), dtype=np.float64)
        missing = 0
        for i, pid in enumerate(ids):
            if pid in id_2_ec_conf:
                new_istps[i] = compute_ptps_proportion(
                    id_2_ec_conf[pid], tps_ecs
                )
            else:
                missing += 1

        if missing > 0:
            logger.warning(
                "  fold %d: %d/%d proteins had no raw EC predictions",
                fold_i,
                missing,
                len(ids),
            )

        new_proba = val_proba_np.copy()
        new_proba[:, istps_idx] = new_istps

        dst_path = out_dir / pkl_name
        with open(dst_path, "wb") as fh:
            pickle.dump((new_proba, class_names, test_df), fh)

        old_ap_mean = val_proba_np[:, istps_idx].mean()
        new_ap_mean = new_istps.mean()
        logger.info(
            "  fold %d: mean isTPS %.4f -> %.4f (%d proteins, %d missing)",
            fold_i,
            old_ap_mean,
            new_ap_mean,
            len(ids),
            missing,
        )
        processed += 1

    return processed


def create_configs(tracks: list[str]) -> None:
    """Copy CLEAN configs to CLEANBetterDetection so the eval pipeline finds them."""
    from enzymeexplorer.src.utils.project_info import get_config_root

    config_root = get_config_root()
    for track in tracks:
        src_cfg = config_root / "CLEAN" / track / "config.yaml"
        if not src_cfg.exists():
            continue
        dst_dir = config_root / NEW_MODEL_NAME / track
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_cfg = dst_dir / "config.yaml"
        if not dst_cfg.exists():
            import shutil

            shutil.copy2(src_cfg, dst_cfg)
            logger.info("Copied config: %s -> %s", src_cfg, dst_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process CLEAN results with proportion-based P(TPS)"
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=ALL_TRACKS,
        help="CLEAN track names to process (default: all)",
    )
    parser.add_argument(
        "--clean-results-dir",
        type=str,
        default=str(CLEAN_RESULTS_DIR),
        help="Directory containing raw CLEAN *_maxsep.csv files",
    )
    parser.add_argument(
        "--clean-working-dir",
        type=str,
        default=str(CLEAN_WORKING_DIR),
        help="Directory containing Rhea TSV files (rhea2ec.tsv, etc.)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--create-configs",
        action="store_true",
        default=True,
        help="Copy CLEAN configs to CLEANBetterDetection for eval pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    id_2_ec_conf = load_all_raw_clean_predictions(Path(args.clean_results_dir))

    tps_ecs_cache: dict[str, set[str]] = {}

    total_processed = 0
    for track in args.tracks:
        logger.info("=== Processing track: %s ===", track)

        dataset_path = DATASET_PATHS.get(track)
        if dataset_path is None:
            logger.warning("No dataset path configured for track %s", track)
            continue

        if dataset_path not in tps_ecs_cache:
            tps_ecs_cache[dataset_path] = build_tps_ec_set(
                Path(args.clean_working_dir), dataset_path
            )
        tps_ecs = tps_ecs_cache[dataset_path]

        n = process_track(track, tps_ecs, id_2_ec_conf, n_folds=args.n_folds)
        total_processed += n
        logger.info("  Processed %d folds for %s", n, track)

    if args.create_configs:
        create_configs(args.tracks)

    logger.info(
        "Done. Total folds processed: %d. Results saved under %s",
        total_processed,
        NEW_OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
