import configargparse
import logging
from pathlib import Path
import json

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
from enzymeexplorer.src.data_preparation.ec_utils import (
    extract_canonical_tps_smiles_from_rhea_and_marts,
    get_matched_rhea_ids_for_marts_reactions,
)
from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import MMSeqs2Wrapper
from enzymeexplorer.src.data_preparation.hmmer_wrapper import HMMerWrapper
from enzymeexplorer.src.data_preparation.positives_utils import (
    preprocess_martsdb,
    prepare_positives_set,
)
from enzymeexplorer.src.data_preparation.common_utils import (
    get_rhea_id_to_master_id_mappings,
    get_non_tps_rhea_ids_with_tps_substrates,
)
from enzymeexplorer.src.data_preparation.negatives_utils import (
    preprocess_negatives,
    mmseqs_based_negative_sampling,
    randomised_negative_sampling,
    prepare_negatives_set,
)
from enzymeexplorer.src.data_preparation.structural_data_utils import (
    download_af_structure,
)
from goatools.obo_parser import GODag

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
np.random.seed(42)
tqdm.pandas()


def parse_args() -> configargparse.Namespace:
    """
    Parse command line arguments.

    :return: Parsed arguments namespace
    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="enzymeexplorer/configs/dataprep_config.yaml",
    )
    parser.add_argument(
        "--presplit-martsdb-clusters-csv-path",
        type=str,
        default="data/EnzymeExplorer_Dataset_TPS.csv",
        help="Path to save/load the pre-split MartsDB clusters CSV file. If the file exists, it will be loaded and the clustering step will be skipped. If it does not exist, the clustering step will be performed and the results will be saved to this path for future use.",
    )
    parser.add_argument(
        "--marts-db-csv-path",
        type=str,
        default="data/martsDB_reactions_2026_02_22.csv",
        help="Path to the MartsDB CSV file",
    )
    parser.add_argument(
        "--sample-negatives-randomly",
        action="store_true",
        help="Whether to sample negative examples randomly from SwissProt instead of using MMSeqs2-based clustering to select hard negatives",
    )
    parser.add_argument(
        "--do-not-filter-negatives-by-putative-tpss",
        action="store_true",
        help="Whether to sample negative examples randomly from SwissProt instead of using MMSeqs2-based clustering to select hard negatives",
    )
    parser.add_argument("--pos_seq_id", type=float, default=0.5)
    parser.add_argument("--pos_coverage", type=float, default=0.8)
    parser.add_argument("--neg_seq_id", type=float, default=0.5)
    parser.add_argument("--neg_coverage", type=float, default=0.8)
    parser.add_argument(
        "--swissprot-tsv-path",
        type=str,
        default="data/swissprot_with_af_2026_03_14.tsv",
        help="Path to the SwissProt TSV file containing EC and GO annotations",
    )
    parser.add_argument(
        "--preprocessed-swissprot-save-path",
        type=str,
        default=None,
        help="Path to save the preprocessed SwissProt dataset after filtering for putative TPSs and redundancy reduction. If not provided, the preprocessed dataset will not be saved.",
    )
    parser.add_argument(
        "--preprocessed-swissprot-load-path",
        type=str,
        default=None,
        help="Path to load the preprocessed SwissProt dataset after filtering for putative TPSs and redundancy reduction. If not provided, the preprocessed dataset will not be loaded.",
    )
    parser.add_argument(
        "--go-dag-path",
        type=str,
        default="data/go-basic_2026_03_14.obo",
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
        "--rhea-to-swissprot-tsv-path",
        type=str,
        default="data/rhea2uniprot_sprot_2026_03_14.tsv",
        help="Path to the Rhea to SwissProt mapping TSV file for determining reaction directionality",
    )
    parser.add_argument(
        "--rhea-directions-tsv-path",
        type=str,
        default="data/rhea-directions_2026_03_14.tsv",
        help="Path to the Rhea directions TSV file for determining reaction directionality",
    )
    parser.add_argument(
        "--rhea-reaction-smiles-tsv-path",
        type=str,
        default="data/rhea-reaction-smiles_2026_03_14.tsv",
        help="Path to the Rhea reaction SMILES TSV file for determining reaction directionality",
    )
    parser.add_argument(
        "--martsDB-rhea-mapping-csv-path",
        type=str,
        default="data/martsDB_reactions_2026_02_22_rhea.csv",
        help="Path to the MartsDB to Rhea mapping CSV",
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
        default=None,
        help="Root directory to save downloaded AlphaFold structures",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()

    mmseqs = MMSeqs2Wrapper(threads=8)
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

    if Path(cli_args.presplit_martsdb_clusters_csv_path).exists():
        logger.info(
            f"Loading pre-split MartsDB clusters from {cli_args.presplit_martsdb_clusters_csv_path}"
        )
        positives_data = pd.read_csv(cli_args.presplit_martsdb_clusters_csv_path)
    else:
        logger.info(
            f"Pre-split MartsDB clusters file not found at {cli_args.presplit_martsdb_clusters_csv_path}. Performing clustering and saving results to this path for future use."
        )
        logger.info(
            "Clustering MartsDB sequences and preparing positives set. This may take some time..."
        )
        
        positives_data = prepare_positives_set(
            martsDB,
            mmseqs,
            cli_args.pos_seq_id,
            cli_args.pos_coverage,
            cli_args.n_folds,
            cli_args.presplit_martsdb_clusters_csv_path,
        )

    logger.info("Starting preparation of negatives set...")
    swissprot = pd.read_csv(cli_args.swissprot_tsv_path, sep="\t")
    logger.info(f"Loaded SwissProt dataset size: {len(swissprot)}")

    swissprot = swissprot[swissprot["AF_structure_available"] == True]
    rhea_reaction_smiles = pd.read_csv(
        cli_args.rhea_reaction_smiles_tsv_path,
        sep="\t",
        names=["rhea_id", "reaction_smiles"],
    )
    rhea_directions = pd.read_csv(cli_args.rhea_directions_tsv_path, sep="\t")

    logger.info(
        f"Filtered SwissProt dataset to entries with available AlphaFold structures. Remaining size: {len(swissprot)}"
    )

    if (
        cli_args.preprocessed_swissprot_load_path
        and Path(cli_args.preprocessed_swissprot_load_path).exists()
    ):
        logger.info(
            f"Loading preprocessed SwissProt dataset from {cli_args.preprocessed_swissprot_load_path}"
        )
        nontps_swissprot = pd.read_csv(
            cli_args.preprocessed_swissprot_load_path, sep="\t"
        )
    else:
        logger.info(
            f"Preprocessed SwissProt dataset file not found at {cli_args.preprocessed_swissprot_load_path}. Performing preprocessing and saving results to this path for future use."
        )
        go_dag = GODag(cli_args.go_dag_path)
        hmmer = HMMerWrapper(threads=8)
        if (
            cli_args.martsDB_rhea_mapping_csv_path
            and Path(cli_args.martsDB_rhea_mapping_csv_path).exists()
        ):
            logger.info(
                f"Loading MartsDB to Rhea mapping from {cli_args.martsDB_rhea_mapping_csv_path}"
            )
            marts_reaction_smiles = pd.read_csv(cli_args.martsDB_rhea_mapping_csv_path)
            marts_reaction_smiles["rhea_ids"] = marts_reaction_smiles['rhea_ids'].map(eval)
        else:
            rhea_smiles_canonical, marts_reaction_smiles = (
                extract_canonical_tps_smiles_from_rhea_and_marts(
                    rhea_reaction_smiles, martsDB
                )
            )
            rhea_smiles_canonical = rhea_smiles_canonical[
                (rhea_smiles_canonical.canonical_products_no_stereo.map(len) > 0)
                & (rhea_smiles_canonical.canonical_substrates_no_stereo.map(len) > 0)
            ]
            marts_reaction_smiles["rhea_ids"] = (
                get_matched_rhea_ids_for_marts_reactions(
                    rhea_smiles_canonical, marts_reaction_smiles
                )
            )
            logger.info(
                f"Extracted canonical reaction SMILES for Rhea reactions and matched them to MartsDB reactions. Sample of matched Rhea IDs for MartsDB reactions: {marts_reaction_smiles['rhea_ids'].explode().dropna().unique()[:10]}"
            )
            marts_reaction_smiles.to_csv(
                cli_args.martsDB_rhea_mapping_csv_path, index=False
            )
            logger.info(
                f"Saved MartsDB to Rhea mapping to {cli_args.martsDB_rhea_mapping_csv_path}"
            )

        marts_rhea_ids = set(
            marts_reaction_smiles["rhea_ids"].explode().dropna().unique()
        )
        rhea_id_to_master_id = get_rhea_id_to_master_id_mappings(rhea_directions)
        
        hard_negative_rhea_ids = get_non_tps_rhea_ids_with_tps_substrates(marts_rhea_ids, rhea_id_to_master_id, rhea_reaction_smiles, martsDB)
        
        nontps_swissprot = preprocess_negatives(
            swissprot,
            martsDB.Aminoacid_sequence.unique().tolist(),
            cli_args.pfam_models_dir,
            cli_args.supfam_models_dir,
            go_dag,
            mmseqs,
            hmmer,
            hard_negative_rhea_ids,
            not cli_args.do_not_filter_negatives_by_putative_tpss,
        )

    logger.info(f"Preprocessed non-TPS SwissProt dataset size: {len(nontps_swissprot)}")

    if cli_args.preprocessed_swissprot_save_path:
        nontps_swissprot.to_csv(
            cli_args.preprocessed_swissprot_save_path, sep="\t", index=False
        )
        logger.info(
            f"Saved preprocessed SwissProt dataset to {cli_args.preprocessed_swissprot_save_path}"
        )

    if cli_args.sample_negatives_randomly:
        logger.info(
            "Sampling negative examples randomly from SwissProt without MMSeqs2-based clustering."
        )
        negative_ids, negatives_folds, negatives_to_accepted_tps_substrates = (
            randomised_negative_sampling(
                nontps_swissprot,
                cli_args.number_of_negatives,
                cli_args.n_folds,
            )
        )
    else:
        logger.info(
            "Preparing negatives data using MMSeqs2-based clustering to identify hard negatives."
        )
        # Generate hard and easy negatives

        rhea_to_swissprot = pd.read_csv(cli_args.rhea_to_swissprot_tsv_path, sep="\t")

        negative_ids, negatives_folds, negatives_to_accepted_tps_substrates = (
            mmseqs_based_negative_sampling(
                nontps_swissprot,
                martsDB,
                mmseqs,
                cli_args.n_folds,
                cli_args.number_of_negatives,
                cli_args.neg_seq_id,
                cli_args.neg_coverage,
                rhea_to_swissprot,
                rhea_reaction_smiles,
                rhea_directions
            )
        )

    negatives_data = prepare_negatives_set(
        nontps_swissprot,
        negative_ids,
        negatives_to_accepted_tps_substrates,
        negatives_folds,
    )

    logger.info(f"Prepared negatives dataset size: {len(negatives_data)}")

    if cli_args.structures_root:
        os.makedirs(cli_args.structures_root, exist_ok=True)
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

    final_dataset.to_csv(cli_args.dataset_output_path, index=False)
    logger.info(
        f"Final dataset of size {len(final_dataset)} saved to {cli_args.dataset_output_path}"
    )


if __name__ == "__main__":
    main()
