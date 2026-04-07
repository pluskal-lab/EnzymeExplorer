import argparse
import pandas as pd
import logging
import json
from collections import defaultdict
from enzymeexplorer.src.data_preparation.ec_utils import (
    extract_canonical_tps_smiles_from_rhea_and_marts,
    get_matched_rhea_ids_for_marts_reactions,
    map_marts_reactions_to_ec_numbers_via_rhea_ids,
    get_ec_to_substrate_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)

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
        "--marts2ec-output-csv-path",
        type=str,
        help="Path to the output CSV file for MartsDB to EC mapping",
        default="./data/marts2ec_2026_03_14.csv",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        help="Path to the output JSON file",
        default="./data/ec_to_substrate_mapping_2026_03_14.json",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    rhea_reaction_smiles = pd.read_csv(args.rhea_reaction_smiles_tsv_path, sep="\t", names=["rhea_id", "reaction_smiles"])
    rhea_directions = pd.read_csv(args.rhea_directions_tsv_path, sep="\t")
    rhea_ec_mapping = pd.read_csv(args.rhea_ec_tsv_path, sep="\t")
    martsDB = pd.read_csv(args.martsDB_csv_path)
    logger.info("Data loaded successfully.")

    rhea_smiles, marts_reaction_smiles = extract_canonical_tps_smiles_from_rhea_and_marts(rhea_reaction_smiles, martsDB)

    logger.info(
        "Canonical substrates and products extracted from reaction smiles."
    )
    rhea_smiles = rhea_smiles[
        (rhea_smiles.canonical_products_no_stereo.map(len) > 0)
        & (rhea_smiles.canonical_substrates_no_stereo.map(len) > 0)
    ]

    logger.info(
        "Filtered Rhea reactions to those with substrates and products in MartsDB."
    )

    marts_reaction_smiles["rhea_ids"] = get_matched_rhea_ids_for_marts_reactions(rhea_smiles, marts_reaction_smiles)

    logger.info(
        "Matched MartsDB reactions to Rhea reactions based on substrates and products."
    )

    marts2ec = map_marts_reactions_to_ec_numbers_via_rhea_ids(marts_reaction_smiles, rhea_directions, rhea_ec_mapping)
    
    if args.marts2ec_output_csv_path is not None:
        marts2ec.to_csv(args.marts2ec_output_csv_path, index=False)
        logger.info(f"MartsDB to EC mapping saved to {args.marts2ec_output_csv_path}.")

    logger.info("Mapped MartsDB reactions to EC numbers via Rhea IDs.")

    ec_to_substrate_mapping = get_ec_to_substrate_mapping(marts2ec)

    with open(args.output_json_path, "w") as f:
        json.dump({
            ec: sorted(list(substrates))
            for ec, substrates in sorted(ec_to_substrate_mapping.items(), key=lambda x: list(map(int, x[0].split("."))))
            }, f, indent=4)
    logger.info("EC to substrate mapping saved to JSON file.")


if __name__ == "__main__":
    main()