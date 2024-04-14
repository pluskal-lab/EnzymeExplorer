""" This is the hello-world script iterating over the folds """

import argparse
import pickle

from utils.data import (
    get_folds,
    get_str,
    get_tps_df,
    get_train_val_per_fold,
    get_unsplittable_targets,
)
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=" TPS folds iteration to store the splits-info into CSVs"
    )
    parser.add_argument(
        "--tps-cleaned-csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions.csv",
    )
    parser.add_argument(
        "--negative-samples-path", type=str, default="data/sampled_id_2_seq.pkl"
    )
    parser.add_argument("--kfolds-path", type=str, default="data/tps_folds_nov2023.h5")
    parser.add_argument(
        "--split-description",
        help="A name for the 5-fold data split",
        type=str,
        default="training_folds_stratified_product_based_split",
        choices=[
            "stratified_phylogeny_based_split_with_minor_products",
            "stratified_phylogeny_based_split",
        ],
    )
    parser.add_argument(
        "--class-column",
        help="A column containing TPS type",
        type=str,
        default="SMILES_substrate_canonical_no_stereo",
    )
    parser.add_argument(
        "--classes",
        help="A list of supported TPS classes",
        type=str,
        nargs="+",
        default=[
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "C(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.C(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
            "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
        ],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli_args = parse_args()

    for is_with_negs in [True, False]:
        tps_df = get_tps_df(
            path_to_file=cli_args.tps_cleaned_csv_path,
            path_to_sampled_negatives=cli_args.negative_samples_path,
        )
        tps_df["Kingdom (plant, fungi, bacteria)"] = tps_df[
            "Kingdom (plant, fungi, bacteria)"
        ].map(lambda x: x.replace(" ", "_"))
        logger.info("Loaded TPS dataset")

        logger.info("Iterating over validation folds..")
        kfold_ids = []
        with logging_redirect_tqdm():
            for order_number, fold_i in enumerate(
                tqdm(
                    get_folds(
                        split_desc=cli_args.split_description,
                        path=cli_args.kfolds_path,
                    ),
                    desc=f"Iterating over validation folds per {cli_args.split_description}..",
                )
            ):
                logger.info("Fold: %s", fold_i)
                train_idx, val_idx = get_train_val_per_fold(
                    fold_i=fold_i,
                    split_desc=cli_args.split_description,
                    path=cli_args.kfolds_path,
                )
                train_ids = {get_str(el[0]) for el in train_idx}
                val_ids = {get_str(el[0]) for el in val_idx}

                kfold_ids.append([train_ids, val_ids])
                logger.info("Checking that there is no leakage between folds..")
                assert (
                    len(train_ids.intersection(val_ids)) == 0
                ), "Error in validation: shared ids between folds"

                train_df = tps_df[tps_df["Uniprot ID"].isin(train_ids)].reset_index(
                    drop=True
                )
                val_df = tps_df[tps_df["Uniprot ID"].isin(val_ids)].reset_index(
                    drop=True
                )
                logger.info(
                    f"The number of training samples is {len(train_df)}, and the number of validation datapoints is {len(val_df)}"
                )
            with open(f"{cli_args.split_description}", "wb") as file:
                pickle.dump(kfold_ids, file)

        unsplittable_targets = get_unsplittable_targets(
            split_desc=cli_args.split_description, path=cli_args.kfolds_path
        )
        tps_df[cli_args.split_description] = "-1"
        tps_df[f"{cli_args.split_description}_ignore_in_eval"] = ""
        for fold_iter, (_, val_ids) in enumerate(kfold_ids):
            tps_df.loc[
                tps_df["Uniprot ID"].isin(val_ids),
                cli_args.split_description,
            ] = f"fold_{fold_iter}"
            tps_df.loc[
                (tps_df[cli_args.class_column].isin(unsplittable_targets)),
                f"{cli_args.split_description}_ignore_in_eval",
            ] = "1"

        tps_df.to_csv(
            f"{cli_args.tps_cleaned_csv_path.replace('.csv', '')}{'_with_neg' if is_with_negs else ''}_with_folds.csv",
            index=False,
        )
