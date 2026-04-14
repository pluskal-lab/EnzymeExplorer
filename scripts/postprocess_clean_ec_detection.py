#!/usr/bin/env python3
"""Post-process CLEAN results to compute EC-based isTPS (PR #34 approach).

Instead of using the original Rhea-substrate-overlap approach for isTPS,
this script checks whether CLEAN's predicted EC numbers are in the curated
EC-to-substrate mapping.  A protein is flagged as TPS if any returned EC
is in the mapping *and* has at least one non-precursor substrate.  The
isTPS confidence equals the max confidence among matching ECs.

By default the **extended** mapping (``ec_to_substrate_mapping_extended.json``)
is used.  This mapping augments the original PR #34 mapping (292 ECs) with
64 additional ECs whose Rhea substrates (Indigo-canonicalized) overlap with
TPS substrate SMILES from both old and new datasets, totalling 356 ECs.
To regenerate, run ``scripts/extend_ec_mapping.py``.

This replicates the logic from PR #34's modified CLEAN.predict_proba:

    if ec_num in self.ec_2_substrates:
        if len(self.ec_2_substrates[ec_num] - {"precursor substr"}):
            id_2_substr_2_conf[uni_id]["isTPS"] = max(conf, ...)

The new fold results are saved under
``outputs/CLEANEcDetection/{version}/...``.

Usage::

    conda run -n terpene_miner python scripts/postprocess_clean_ec_detection.py
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLEAN_RESULTS_DIR = Path("/home/samusevich/CLEAN/app/results")
CLEAN_OUTPUT_ROOT = Path("outputs/CLEAN")
NEW_MODEL_NAME = "CLEANEcDetection"
NEW_OUTPUT_ROOT = Path("outputs") / NEW_MODEL_NAME

EC_MAPPING_PATH = Path("data/ec_to_substrate_mapping_extended.json")

ALL_TRACKS = [
    "with_minor_reactions_phylo_folds",
    "with_minor_reactions",
    "synced_folds",
    "new_dataset",
    "cross_synced_to_new",
    "cross_new_tps_old_neg",
]


def build_tps_ec_set(ec_mapping_path: Path) -> set[str]:
    """Build TPS EC set from the PR #34 curated mapping.

    An EC is TPS-relevant if it appears in the JSON mapping AND has at
    least one non-precursor substrate (matching PR #34 logic:
    ``if len(self.ec_2_substrates[ec_num] - {"precursor substr"})``).

    Returns ECs with ``EC:`` prefix to match maxsep CSV format.
    """
    with open(ec_mapping_path, "r") as fh:
        ec2sub: dict[str, list[str]] = json.load(fh)

    tps_ecs: set[str] = set()
    for ec, substrates in ec2sub.items():
        if set(substrates) - {"precursor substr"}:
            tps_ecs.add(f"EC:{ec}")

    logger.info(
        "TPS EC set from %s: %d ECs (with non-precursor substrates)",
        ec_mapping_path,
        len(tps_ecs),
    )
    return tps_ecs


def load_all_raw_clean_predictions(
    results_dir: Path,
) -> dict[str, dict[str, float]]:
    """Parse all CLEAN maxsep CSV files into {protein_id: {ec: conf}}."""
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


def compute_ec_based_istps(
    ec_confs: dict[str, float],
    tps_ecs: set[str],
) -> float:
    """Compute isTPS = max confidence among ECs in the TPS set.

    Returns 0.0 if no predicted EC is in ``tps_ecs``.
    """
    max_conf = 0.0
    for ec, conf in ec_confs.items():
        if ec in tps_ecs:
            max_conf = max(max_conf, conf)
    return max_conf


def process_track(
    track: str,
    tps_ecs: set[str],
    id_2_ec_conf: dict[str, dict[str, float]],
    n_folds: int = 5,
) -> int:
    """Post-process CLEAN fold results for one track."""
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
                new_istps[i] = compute_ec_based_istps(
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

        old_mean = val_proba_np[:, istps_idx].mean()
        new_mean = new_istps.mean()
        logger.info(
            "  fold %d: mean isTPS %.4f -> %.4f (%d proteins, %d missing)",
            fold_i,
            old_mean,
            new_mean,
            len(ids),
            missing,
        )
        processed += 1

    return processed


def create_configs(tracks: list[str]) -> None:
    """Copy CLEAN configs to CLEANEcDetection for eval pipeline."""
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
            shutil.copy2(src_cfg, dst_cfg)
            logger.info("Copied config: %s -> %s", src_cfg, dst_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process CLEAN results with EC-based isTPS (PR #34)"
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=ALL_TRACKS,
        help="CLEAN track names to process (default: all)",
    )
    parser.add_argument(
        "--ec-mapping",
        type=str,
        default=str(EC_MAPPING_PATH),
        help="Path to ec_to_substrate_mapping JSON (from PR #34)",
    )
    parser.add_argument(
        "--clean-results-dir",
        type=str,
        default=str(CLEAN_RESULTS_DIR),
        help="Directory containing raw CLEAN *_maxsep.csv files",
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
        help="Copy CLEAN configs for eval pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tps_ecs = build_tps_ec_set(Path(args.ec_mapping))
    id_2_ec_conf = load_all_raw_clean_predictions(Path(args.clean_results_dir))

    total_processed = 0
    for track in args.tracks:
        logger.info("=== Processing track: %s ===", track)
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
