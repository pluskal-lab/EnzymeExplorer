import pandas as pd
from enzymeexplorer.src.utils.data import (
    get_canonical_smiles,
)
from enzymeexplorer.src.data_preparation.constants import (
    TPS_ECS_TO_SUBSTRATES_BASE,
    NON_TPS_ECS,
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


def extract_canonical_tps_smiles_from_rhea_and_marts(rhea_reaction_smiles: pd.DataFrame, martsDB: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rhea_smiles = rhea_reaction_smiles.copy()
    rhea_smiles["canonical_substrates_no_stereo"] = rhea_smiles[
        "reaction_smiles"
    ].apply(get_canonical_substrates_no_stereo)
    rhea_smiles["canonical_products_no_stereo"] = rhea_smiles["reaction_smiles"].apply(
        get_canonical_products_no_stereo
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
    return rhea_smiles, marts_reaction_smiles

def get_matched_rhea_ids_for_marts_reactions(rhea_smiles: pd.DataFrame, marts_reaction_smiles: pd.DataFrame) -> list[set[str]]:
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
    return martsdb_matched_rhea_ids

def map_marts_reactions_to_ec_numbers_via_rhea_ids(marts_reaction_smiles: pd.DataFrame, rhea_directions: pd.DataFrame, rhea_ec_mapping: pd.DataFrame) -> pd.DataFrame:
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
    return marts2ec

def get_ec_to_substrate_mapping(marts2ec: pd.DataFrame) -> dict[str, set[str]]:
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
    return ec_to_substrate_mapping