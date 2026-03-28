import argparse
import pandas as pd
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)

def parse_args():
    parser = argparse.ArgumentParser(description="Get EC to substrate mapping")
    parser.add_argument(
        "--marts2ec-csv-path",
        type=str,
        help="Path to the Marts2EC CSV file",
        default="./data/marts2ec_2026_03_14.csv",
    )
    parser.add_argument(
        "--swissprot-csv-path",
        type=str,
        help="Path to the SwissProt CSV file",
        default="./data/swissprot_2026_03_14.tsv",
    )
    parser.add_argument(
        "--enzyme-explorer-dataset-csv-path",
        type=str,
        help="Path to the EnzymeExplorer dataset CSV file",
        default="./data/EnzymeExplorer_Dataset.csv",
    )
    parser.add_argument(
        "--martsDB-csv-path",
        type=str,
        help="Path to the raw MartsDB CSV file",
        default="./data/martsDB_reactions_2026_02_22.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory",
        default="./data/clean_datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    marts2ec = pd.read_csv(args.marts2ec_csv_path)
    martsDB = pd.read_csv(args.martsDB_csv_path)
    dataset = pd.read_csv(args.enzyme_explorer_dataset_csv_path)
    swissprot = pd.read_csv(args.swissprot_csv_path, sep="\t")
    swissprot = swissprot[["Entry", "EC number", "Sequence"]].dropna()
    swissprot = swissprot[~swissprot["EC number"].str.contains("-")]
    swissprot["EC number"] = swissprot["EC number"].str.split(";").apply(lambda x: [ec.strip() for ec in x])
    swissprot = swissprot.explode("EC number").reset_index(drop=True)

    marts_id_to_ec = marts2ec[["Enzyme_marts_ID", "ID"]].groupby("Enzyme_marts_ID")["ID"].apply(set).to_dict()
    swissprot_to_ec = swissprot[["Entry", "EC number"]].groupby("Entry")["EC number"].apply(set).to_dict()
    prot_id_to_ec = {**marts_id_to_ec, **swissprot_to_ec}

    dataset["ECs"] = dataset["ID"].map(prot_id_to_ec)
    dataset = dataset[dataset["ECs"].notna()]

    for i in range(5):
        fold_i_train = dataset[dataset["Fold"] != i]
        fold_i_train = fold_i_train[["ID", "ECs", "Aminoacid_sequence"]].drop_duplicates("ID")
        fold_i_train["ECs"] = fold_i_train["ECs"].apply(lambda x: ";".join(x))
        fold_i_train.columns = ["Entry", "EC number", "Sequence"]
        fold_i_train.to_csv(output_dir / f"clean_enzexp_tps_{i}_train.csv", index=False, sep="\t")
        
        fold_i_test = dataset[dataset["Fold"] == i]
        fold_i_test = fold_i_test[["ID", "Aminoacid_sequence"]].drop_duplicates("ID")
        with open(output_dir / f"clean_enzexp_tps_{i}_test.fasta", "w") as f:
            for _, row in fold_i_test.iterrows():
                f.write(f">{row['ID']}\n{row['Aminoacid_sequence']}\n")

    combined = dataset[["ID", "ECs", "Aminoacid_sequence"]].drop_duplicates("ID")
    combined["ECs"] = combined["ECs"].apply(lambda x: ";".join(x))
    combined.columns = ["Entry", "EC number", "Sequence"]
    combined.to_csv(output_dir / "combined_data.csv", index=False, sep="\t")