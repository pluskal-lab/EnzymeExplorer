import argparse
import pandas as pd
from enzymeexplorer.src.data_preparation.constants import (
    TPS_ECS_TO_SUBSTRATES_BASE,
    NON_TPS_ECS,
)
from enzymeexplorer.src.utils.data import (
    get_canonical_smiles,
)
from collections import defaultdict, Counter
import json
from rdkit.Chem import MolToSmiles, rdChemReactions  # type: ignore
from rdkit import Chem
from indigo import Indigo
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)

indigo = Indigo()


def parse_args():
    parser = argparse.ArgumentParser(description="Get EC to substrate mapping")
    parser.add_argument(
        "--rhea-reaction-smiles-tsv-path",
        type=str,
        help="Path to the Rhea reaction smiles TSV file",
        default="./data/rhea-reaction-smiles_2026_03_14.tsv",
    )
    parser.add_argument(
        "--rhea-directions-tsv-path",
        type=str,
        help="Path to Rhea reaction directions TSV file",
        default="./data/rhea-directions_2026_03_14.tsv",
    )
    parser.add_argument(
        "--rhea-ec-tsv-path",
        type=str,
        help="Path to the Rhea-EC number mapping TSV file",
        default="./data/rhea2ec_2026_03_14.tsv",
    )
    parser.add_argument(
        "--martsDB-csv-path",
        type=str,
        help="Path to the raw MartsDB CSV file",
        default="./data/martsDB_reactions_2026_02_22.csv",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        help="Path to the output JSON file",
        default="./data/ec_to_substrate_mapping_2026_03_14.json",
    )
    return parser.parse_args()


def get_canonical_substrates_no_stereo(rxn_smiles: str):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    substrates = trxn.GetReactants()
    return [get_canonical_smiles(MolToSmiles(substr)) for substr in substrates]


def get_canonical_products_no_stereo(rxn_smiles: str):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    products = trxn.GetProducts()
    return [get_canonical_smiles(MolToSmiles(product)) for product in products]


def main():
    args = parse_args()
    rhea_reaction_smiles = pd.read_csv(args.rhea_reaction_smiles_tsv_path, sep="\t", names=["rhea_id", "reaction_smiles"])
    rhea_directions = pd.read_csv(args.rhea_directions_tsv_path, sep="\t")
    rhea_ec_mapping = pd.read_csv(args.rhea_ec_tsv_path, sep="\t")
    martsDB = pd.read_csv(args.martsDB_csv_path)
    logger.info("Data loaded successfully.")

    rhea_smiles = rhea_reaction_smiles.copy()
    rhea_smiles["canonical_substrates_no_stereo"] = rhea_smiles[
        "reaction_smiles"
    ].apply(get_canonical_substrates_no_stereo)
    rhea_smiles["canonical_products_no_stereo"] = rhea_smiles["reaction_smiles"].apply(
        get_canonical_products_no_stereo
    )

    logger.info(
        "Canonical substrates and products extracted from Rhea reaction smiles."
    )

    marts_reaction_smiles = martsDB[["Enzyme_marts_ID", "Type"]].copy()
    marts_reaction_smiles["SMILES_substrate_canonical_no_stereo"] = martsDB[
        "Substrate_smiles"
    ].map(
        lambda smiles: ".".join(
            [
                get_canonical_smiles(smiles, without_stereo=True).upper()
                for smiles in smiles.split(";")
            ]
        )
    )
    marts_reaction_smiles["SMILES_product_canonical_no_stereo"] = martsDB[
        "Product_smiles"
    ].map(
        lambda smiles: ".".join(
            [
                get_canonical_smiles(smiles, without_stereo=True).upper()
                for smiles in smiles.split(";")
            ]
        )
    )
    marts_reaction_smiles["canonical_substrates_no_stereo"] = marts_reaction_smiles[
        "SMILES_substrate_canonical_no_stereo"
    ].map(lambda x: x.split("."))
    marts_reaction_smiles["canonical_products_no_stereo"] = marts_reaction_smiles[
        "SMILES_product_canonical_no_stereo"
    ].map(lambda x: x.split("."))

    logger.info(
        "Canonical substrates and products extracted from MartsDB reaction smiles."
    )

    marts_substrate_set = set(
        marts_reaction_smiles["canonical_substrates_no_stereo"].explode().to_list()
    )
    marts_products_set = set(
        marts_reaction_smiles["canonical_products_no_stereo"].explode().to_list()
    )
    rhea_smiles["canonical_substrates_no_stereo"] = rhea_smiles[
        "canonical_substrates_no_stereo"
    ].map(lambda x: [s.upper() for s in x if s.upper() in marts_substrate_set])
    rhea_smiles["canonical_products_no_stereo"] = rhea_smiles[
        "canonical_products_no_stereo"
    ].map(lambda x: [s.upper() for s in x if s.upper() in marts_products_set])
    rhea_smiles = rhea_smiles[
        (rhea_smiles.canonical_products_no_stereo.map(len) > 0)
        & (rhea_smiles.canonical_substrates_no_stereo.map(len) > 0)
    ]

    logger.info(
        "Filtered Rhea reactions to those with substrates and products in MartsDB."
    )

    martsdb_matched_rhea_ids = []
    for _, row_marts in marts_reaction_smiles.iterrows():
        matched_rhea_ids = set()
        for _, row_rhea in rhea_smiles.iterrows():
            substrate_ctr_rhea = Counter(row_rhea.canonical_substrates_no_stereo)
            product_ctr_rhea = Counter(row_rhea.canonical_products_no_stereo)
            substrate_ctr_marts = Counter(row_marts.canonical_substrates_no_stereo)
            product_ctr_marts = Counter(row_marts.canonical_products_no_stereo)
            substrate_ctr_rhea.subtract(substrate_ctr_marts)
            product_ctr_rhea.subtract(product_ctr_marts)
            if all(value >= 0 for value in product_ctr_rhea.values()) and all(
                value >= 0 for value in substrate_ctr_rhea.values()
            ):
                matched_rhea_ids.add(row_rhea.rhea_id)
        martsdb_matched_rhea_ids.append(matched_rhea_ids)
    marts_reaction_smiles["rhea_ids"] = martsdb_matched_rhea_ids

    logger.info(
        "Matched MartsDB reactions to Rhea reactions based on substrates and products."
    )

    marts_id_to_rhea_id = (
        marts_reaction_smiles[
            [
                "Enzyme_marts_ID",
                "Type",
                "SMILES_substrate_canonical_no_stereo",
                "rhea_ids",
            ]
        ]
        .explode("rhea_ids")
        .merge(
            rhea_directions[["RHEA_ID_LR", "RHEA_ID_MASTER"]],
            left_on="rhea_ids",
            right_on="RHEA_ID_LR",
            how="left",
        )[
            [
                "Enzyme_marts_ID",
                "Type",
                "SMILES_substrate_canonical_no_stereo",
                "RHEA_ID_MASTER",
                "rhea_ids",
            ]
        ]
        .merge(
            rhea_directions[["RHEA_ID_RL", "RHEA_ID_MASTER"]],
            left_on="rhea_ids",
            right_on="RHEA_ID_RL",
            how="left",
            suffixes=("_LR", "_RL"),
        )[
            [
                "Enzyme_marts_ID",
                "Type",
                "SMILES_substrate_canonical_no_stereo",
                "RHEA_ID_MASTER_LR",
                "RHEA_ID_MASTER_RL",
            ]
        ]
    )
    marts_id_to_rhea_id["RHEA_ID"] = marts_id_to_rhea_id[
        "RHEA_ID_MASTER_LR"
    ].combine_first(marts_id_to_rhea_id["RHEA_ID_MASTER_RL"])
    marts_id_to_rhea_id = (
        marts_id_to_rhea_id[
            [
                "Enzyme_marts_ID",
                "Type",
                "SMILES_substrate_canonical_no_stereo",
                "RHEA_ID",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )
    marts_id_to_rhea_id["RHEA_ID"] = marts_id_to_rhea_id["RHEA_ID"].astype(int)
    marts2ec = marts_id_to_rhea_id.merge(
        rhea_ec_mapping[["MASTER_ID", "ID"]],
        left_on="RHEA_ID",
        right_on="MASTER_ID",
        how="inner",
    )

    logger.info("Mapped MartsDB reactions to EC numbers via Rhea IDs.")

    ec_to_substrate_mapping = defaultdict(set)
    for _, row in marts2ec.iterrows():
        if row.ID not in NON_TPS_ECS:
            if row.Type == "pt":
                ec_to_substrate_mapping[row.ID].add("precursor substr")
            else:
                for substrate in row.SMILES_substrate_canonical_no_stereo.split("."):
                    ec_to_substrate_mapping[row.ID].add(substrate)
    for ec, substrate in TPS_ECS_TO_SUBSTRATES_BASE.items():
        if ec not in NON_TPS_ECS:
            ec_to_substrate_mapping[ec].add(substrate)

    with open(args.output_json_path, "w") as f:
        json.dump({
            ec: list(substrates)
            for ec, substrates in sorted(ec_to_substrate_mapping.items(), key=lambda x: list(map(int, x[0].split("."))))
            }, f, indent=4)
    logger.info("EC to substrate mapping saved to JSON file.")


if __name__ == "__main__":
    main()