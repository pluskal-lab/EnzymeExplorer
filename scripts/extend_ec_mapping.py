#!/usr/bin/env python3
"""Extend the PR #34 EC-to-substrate mapping to cover the new dataset.

Augments ``data/ec_to_substrate_mapping_2026_03_14.json`` with additional
ECs whose Rhea reaction substrates (Indigo-canonicalized) overlap with
TPS substrate SMILES from both the old and new datasets.  This closes
the coverage gap that causes CLEANEcDetection to fail on new-dataset
tracks.

Usage::

    conda run -n terpene_miner python scripts/extend_ec_mapping.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit.Chem import MolToSmiles, rdChemReactions  # type: ignore

from enzymeexplorer.src.utils.data import get_canonical_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXISTING_MAPPING = Path("data/ec_to_substrate_mapping_2026_03_14.json")
EXTENDED_MAPPING = Path("data/ec_to_substrate_mapping_extended.json")
RHEA_DIR = Path("_clean_working_dir")

DATASET_CSVS = [
    "data/EnzymeExplorer_Dataset.csv",
    "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "data/TPS-Nov19_2023_with_synced_folds.csv",
]

TARGET_COL = "SMILES_substrate_canonical_no_stereo"


def build_rhea_ec_2_substrates(rhea_dir: Path) -> dict[str, set[str]]:
    """Build EC-to-substrates map from Rhea with Indigo canonicalization."""
    rhea2ec_df = pd.read_csv(rhea_dir / "rhea2ec.tsv", sep="\t")
    ec_2_rheaids = rhea2ec_df.groupby("ID")["RHEA_ID"].agg(set).to_dict()

    rhea2smiles_df = pd.read_csv(
        rhea_dir / "rhea-reaction-smiles.tsv", sep="\t", header=None
    )
    rheaid_2_rxn = rhea2smiles_df[[0, 1]].set_index(0)[1].to_dict()

    rhea2dir_df = pd.read_csv(rhea_dir / "rhea-directions.tsv", sep="\t")
    master_2_lr = (
        rhea2dir_df[["RHEA_ID_MASTER", "RHEA_ID_LR"]]
        .set_index("RHEA_ID_MASTER")["RHEA_ID_LR"]
        .to_dict()
    )

    ec_2_subs: dict[str, set[str]] = defaultdict(set)
    for ec_code, rhea_ids in ec_2_rheaids.items():
        for rhea_id in rhea_ids:
            if rhea_id in master_2_lr:
                rhea_id = master_2_lr[rhea_id]
            if rhea_id not in rheaid_2_rxn:
                continue
            try:
                rxn = rdChemReactions.ReactionFromSmarts(
                    rheaid_2_rxn[rhea_id], useSmiles=True
                )
                trxn = rdChemReactions.ChemicalReaction(rxn)
                for mol in trxn.GetReactants():
                    ec_2_subs[ec_code].add(
                        get_canonical_smiles(MolToSmiles(mol))
                    )
            except Exception:
                pass

    logger.info("Rhea ec_2_substrates: %d ECs", len(ec_2_subs))
    return dict(ec_2_subs)


def collect_tps_substrates(csv_paths: list[str]) -> set[str]:
    """Collect all TPS substrate SMILES across datasets."""
    substrates: set[str] = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        substrates |= {
            s for s in df[TARGET_COL].values if s not in {"Unknown", "Negative"}
        }
    logger.info("TPS substrate SMILES (all datasets): %d", len(substrates))
    return substrates


def main() -> None:
    with open(EXISTING_MAPPING) as fh:
        existing: dict[str, list[str]] = json.load(fh)
    logger.info("Existing mapping: %d ECs", len(existing))

    ec_2_subs = build_rhea_ec_2_substrates(RHEA_DIR)
    tps_substrates = collect_tps_substrates(DATASET_CSVS)

    new_ecs: dict[str, list[str]] = {}
    for ec, subs in ec_2_subs.items():
        overlap = subs & tps_substrates
        if overlap and ec not in existing:
            new_ecs[ec] = sorted(overlap)

    logger.info("New ECs to add: %d", len(new_ecs))
    for ec in sorted(new_ecs):
        logger.info("  %s -> %s", ec, new_ecs[ec][:2])

    extended = dict(existing)
    extended.update(new_ecs)
    logger.info(
        "Extended mapping: %d ECs (was %d, added %d)",
        len(extended),
        len(existing),
        len(new_ecs),
    )

    with open(EXTENDED_MAPPING, "w") as fh:
        json.dump(extended, fh, indent=2, sort_keys=True)
    logger.info("Saved to %s", EXTENDED_MAPPING)


if __name__ == "__main__":
    main()
