import argparse
import logging
import pickle
from pathlib import Path
import json

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from enzymeexplorer.src.data_preparation.mmseqs import MMSeqs2Wrapper
from enzymeexplorer.src.data_preparation.hmmer import HMMerWrapper
from enzymeexplorer.src.data_preparation.utils import (
    cluster_dataset,
    get_hard_negative_cluster_ids,
    preprocess_martsdb,
    proprocess_negatives,
    get_stratified_group_kfold_splits,
    get_is_splittable,
    download_af_structure,
)
from enzymeexplorer.src.data_preparation.constants import (
    MAJOR_CLASSES,
)
from goatools.obo_parser import GODag

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
np.random.seed(42)
tqdm.pandas()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate sequence clusters using mmseqs easy-cluster"
    )
    parser.add_argument(
        "--marts-db-csv-path",
        type=str,
        default="data/martsDB_reactions_2026_02_22.csv",
        help="Path to the MartsDB CSV file",
    )
    parser.add_argument(
        "--id2kingdom-output-path",
        type=str,
        default="data/id_2_kingdom_dataset.pkl",
        help="Path to save the TPS Kingdom data",
    )
    parser.add_argument(
        "--substrate2tps-type-output-path",
        type=str,
        default="data/substrate_2_tps_type.pkl",
        help="Path to save the substrate to TPS type mapping",
    )
    parser.add_argument(
        "--swissprot-tsv-path",
        type=str,
        default="data/swissprot.tsv",
        help="Path to the SwissProt TSV file containing EC and GO annotations",
    )
    parser.add_argument(
        "--go-dag-path",
        type=str,
        default="data/go-basic.obo",
        help="Path to the Gene Ontology DAG file",
    )
    parser.add_argument(
        "--pfam-models-dir",
        type=str,
        default="data/pfam_models",
        help="Directory containing Pfam HMM models for filtering negatives",
    )
    parser.add_argument(
        "--supfam-models-dir",
        type=str,
        default="data/supfam_models",
        help="Directory containing Supfam HMM models for filtering negatives",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--number-of-negatives",
        type=int,
        default=10000,
        help="Number of negative examples to pull from SwissProt",
    )
    parser.add_argument(
        "--dataset-output-path",
        type=str,
        default="data/EnzymeExplorer_Dataset.csv",
        help="Path to save the final data CSV file",
    )
    parser.add_argument(
        "--structures-root",
        type=str,
        default="data/enzyme_explorer_pdbs",
        help="Root directory to save downloaded AlphaFold structures",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()

    martsDB = pd.read_csv(cli_args.marts_db_csv_path)
    martsDB, marts_duplicates = preprocess_martsdb(martsDB)

    logger.info(f"Preprocessed MartsDB dataset size: {len(martsDB)}")

    martsDB.to_csv(
        cli_args.marts_db_csv_path.split(".csv")[0] + "_preprocessed.csv", index=False
    )
    json.dump(
        marts_duplicates,
        open(cli_args.marts_db_csv_path.split(".csv")[0] + "_duplicates.json", "w"),
        indent=4,
    )

    logger.info(
        f"Saved preprocessed MartsDB dataset to {cli_args.marts_db_csv_path.split('.csv')[0] + '_preprocessed.csv'}"
    )

    # computing categories per kingdom
    id_2_kingdom_dataset = (
        martsDB[["Enzyme_marts_ID", "Kingdom"]]
        .set_index("Enzyme_marts_ID")
        .to_dict()["Kingdom"]
    )
    with open(cli_args.id2kingdom_output_path, "wb") as file:
        pickle.dump(id_2_kingdom_dataset, file)

    logger.info(f"Saved ID to Kingdom mapping to {cli_args.id2kingdom_output_path}")

    substrate_2_tps_type = (
        martsDB[["SMILES_substrate_canonical_no_stereo", "Type"]]
        .drop_duplicates()
        .groupby("SMILES_substrate_canonical_no_stereo")["Type"]
        .apply(set)
        .reset_index()
        .set_index("SMILES_substrate_canonical_no_stereo")
        .to_dict()["Type"]
    )
    with open(cli_args.substrate2tps_type_output_path, "wb") as file:
        pickle.dump(substrate_2_tps_type, file)

    logger.info(
        f"Saved substrate to TPS type mapping to {cli_args.substrate2tps_type_output_path}"
    )

    swissprot = pd.read_csv(cli_args.swissprot_tsv_path, sep="\t")
    logger.info(f"Loaded SwissProt dataset size: {len(swissprot)}")

    swissprot = swissprot[swissprot["AF_structure_available"] == True]

    logger.info(
        f"Filtered SwissProt dataset to entries with available AlphaFold structures. Remaining size: {len(swissprot)}"
    )

    mmseqs = MMSeqs2Wrapper(threads=8)
    go_dag = GODag(cli_args.go_dag_path)
    hmmer = HMMerWrapper(threads=8)
    nontps_swissprot = proprocess_negatives(
        swissprot,
        martsDB.Aminoacid_sequence.unique().tolist(),
        cli_args.pfam_models_dir,
        cli_args.supfam_models_dir,
        go_dag,
        mmseqs,
        hmmer,
    )

    logger.info(f"Preprocessed non-TPS SwissProt dataset size: {len(nontps_swissprot)}")

    martsDB_clusters_df, martsDB_representatives_df = cluster_dataset(
        martsDB,
        id_column="Enzyme_marts_ID",
        seq_column="Aminoacid_sequence",
        mmseqs=mmseqs,
    )

    logger.info(
        f"Clustered MartsDB sequences into {martsDB_clusters_df['Representative'].nunique()} clusters"
    )

    martsDB_with_clusters = martsDB.merge(
        martsDB_clusters_df,
        left_on="Enzyme_marts_ID",
        right_on="Member",
        how="left",
    )
    assert (
        martsDB_with_clusters[martsDB_with_clusters["Representative"].isna()].shape[0]
        == 0
    ), "Some sequences are not assigned to any cluster."

    is_martsdb_splittable, unsplittable_target_values = get_is_splittable(
        martsDB_with_clusters,
        id_column="Enzyme_marts_ID",
        cluster_id_column="Representative",
        target_col="SMILES_substrate_canonical_no_stereo",
        classes=MAJOR_CLASSES,
        n_folds=cli_args.n_folds,
    )
    if not is_martsdb_splittable:
        raise ValueError(
            "The dataset cannot be split without leakage based on the provided target classes."
        )
    logger.info("MartsDB dataset is splittable without leakage.")

    positives_folds = get_stratified_group_kfold_splits(
        martsDB_with_clusters,
        id_column="Enzyme_marts_ID",
        cluster_id_column="Representative",
        optimize_distribution=True,
        target_col="SMILES_substrate_canonical_no_stereo",
        classes=MAJOR_CLASSES,
        n_folds=cli_args.n_folds,
    )

    logger.info("Generated stratified group K-Fold splits for positives.")

    # Generate hard and easy negatives
    nontps_swissprot_clusters_df, nontps_swissprot_representatives_df = cluster_dataset(
        nontps_swissprot,
        id_column="Entry",
        seq_column="Sequence",
        mmseqs=mmseqs,
    )

    logger.info(
        f"Clustered non-TPS SwissProt sequences into {nontps_swissprot_clusters_df['Representative'].nunique()} clusters"
    )

    hard_negative_cluster_ids = get_hard_negative_cluster_ids(
        nontps_swissprot_representatives_df, martsDB, mmseqs
    )

    hard_negative_clusters = nontps_swissprot_clusters_df[
        nontps_swissprot_clusters_df["Representative"].isin(hard_negative_cluster_ids)
    ].copy()
    hard_negative_clusters["Type"] = "Hard"

    logger.info(f"Identified {len(hard_negative_clusters)} hard negative samples.")

    easy_negative_cluster_ids = nontps_swissprot_clusters_df[
        ~nontps_swissprot_clusters_df["Representative"].isin(hard_negative_cluster_ids)
    ]["Representative"].unique()
    easy_negative_cluster_ids = np.random.choice(
        easy_negative_cluster_ids,
        size=cli_args.number_of_negatives - len(hard_negative_clusters),
        replace=False,
    )

    logger.info(f"Chosen {len(easy_negative_cluster_ids)} easy negative clusters.")

    easy_negative_clusters = nontps_swissprot_clusters_df[
        nontps_swissprot_clusters_df["Representative"].isin(easy_negative_cluster_ids)
    ].copy()
    easy_negative_clusters["Type"] = "Easy"
    easy_negative_clusters.drop_duplicates("Representative", inplace=True)

    logger.info(f"Selected {len(easy_negative_clusters)} easy negative samples.")

    negative_clusters = pd.concat([hard_negative_clusters, easy_negative_clusters])
    negatives_folds = get_stratified_group_kfold_splits(
        negative_clusters,
        id_column="Member",
        cluster_id_column="Representative",
        optimize_distribution=False,
        target_col="Type",
        classes=["Easy", "Hard"],
        n_folds=cli_args.n_folds,
    )

    logger.info("Generated stratified group K-Fold splits for negatives.")

    positives_data = martsDB[
        [
            "Enzyme_marts_ID",
            "Aminoacid_sequence",
            "SMILES_substrate_canonical_no_stereo",
            "SMILES_product_canonical_no_stereo",
            "Type",
            "OriginalType",
            "Kingdom",
            "Class",
        ]
    ].rename(columns={"Enzyme_marts_ID": "ID"})
    positives_data["Fold"] = None
    for fold_idx, val_ids in enumerate(positives_folds):
        positives_data.loc[positives_data["ID"].isin(val_ids), "Fold"] = fold_idx

    logger.info("Assigned folds to positive samples.")

    negative_ids = set(negative_clusters.Member.unique())
    negatives_data = nontps_swissprot[nontps_swissprot.Entry.isin(negative_ids)][
        ["Entry", "Sequence"]
    ].rename(columns={"Entry": "ID", "Sequence": "Aminoacid_sequence"})

    # TODO: Filter negatives by TPS PFAM & SUPFAM hits

    negatives_data["SMILES_substrate_canonical_no_stereo"] = "Unknown"
    negatives_data["SMILES_product_canonical_no_stereo"] = "Unknown"
    negatives_data["Kingdom"] = "Unknown"
    negatives_data["Class"] = "Unknown"
    negatives_data["Type"] = "Unknown"
    negatives_data["OriginalType"] = "Unknown"
    negatives_data["Fold"] = None
    for fold_idx, val_ids in enumerate(negatives_folds):
        negatives_data.loc[negatives_data["ID"].isin(val_ids), "Fold"] = fold_idx

    logger.info("Assigned folds to negative samples.")
    logger.info(
        "Downloading AlphaFold structures for negative samples. This may take some time..."
    )
    download_results = negatives_data["ID"].progress_apply(
        lambda uniprot_id: download_af_structure(uniprot_id, cli_args.structures_root)
    )

    negatives_data = negatives_data[download_results]

    logger.info(
        f"Successfully downloaded AlphaFold structures for {len(negatives_data)} negative samples. Removed {len(download_results) - len(negatives_data)} samples without available structures."
    )

    final_dataset = pd.concat([positives_data, negatives_data]).reset_index(drop=True)

    assert (
        final_dataset["Fold"].isna().sum() == 0
    ), "Some samples are not assigned to any fold."

    final_dataset["Fold_ignore_in_eval"] = None
    final_dataset.loc[
        final_dataset["SMILES_substrate_canonical_no_stereo"].isin(
            unsplittable_target_values
        ),
        "Fold_ignore_in_eval",
    ] = 1

    final_dataset.to_csv(cli_args.dataset_output_path, index=False)
    logger.info(
        f"Final dataset of size {len(final_dataset)} saved to {cli_args.dataset_output_path}"
    )

    final_positives = final_dataset[final_dataset["Type"] != "Unknown"]
    final_positives.to_csv(
        Path(cli_args.dataset_output_path).with_name(
            Path(cli_args.dataset_output_path).stem + "_TPS.csv"
        ),
        index=False,
    )
    logger.info(
        f"Final positives subset of size {len(final_positives)} saved to {Path(cli_args.dataset_output_path).with_name(Path(cli_args.dataset_output_path).stem + '_TPS.csv')}"
    )



if __name__ == "__main__":
    main()