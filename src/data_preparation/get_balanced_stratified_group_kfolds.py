"""This module prepares validation schema"""

import argparse
import json
import logging
import pickle
import warnings
from typing import Optional

warnings.simplefilter("ignore", UserWarning)
import uuid
from collections import defaultdict

import h5py  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore
from sklearn.model_selection import (  # type: ignore
    StratifiedGroupKFold,
    StratifiedKFold,
)

from src.utils.data import get_major_classes_distribution, get_tps_df, triplets_dtype

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tps-cleaned-csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions.csv",
    )
    parser.add_argument(
        "--negative-samples-path", type=str, default="data/sampled_id_2_seq.pkl"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--split-description", type=str, default="stratified_phylogeny_based_split"
    )
    args = parser.parse_args()
    return args


def stratified_kfold_phylogeny_based(
    tps_df: pd.DataFrame,
    args: argparse.Namespace,
    target_col_name: str,
    major_classes: list[set[str]],
    max_allowed_proportion_of_class_in_cc: float = 0.7,
    phylogenetic_clusters_path: str = "data/phylogenetic_clusters.pkl",
    desc: Optional[str] = None,
):
    """
    This function computes stratified group kfold split with sequence groups being defined based on claded in phylogenetic tree
    :param tps_df: pre-processed terpene synthases dataset as a pandas dataframe
    :param args: current argparse.Namespace
    :param desc: validation schema name
    """

    kfold_neg = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)

    with open(phylogenetic_clusters_path, "rb") as f:
        id_2_group, _ = pickle.load(f)

    tps_df.loc[
        tps_df["Uniprot ID"].map(lambda x: x not in id_2_group), target_col_name
    ] = "Unknown"
    tps_df_neg = tps_df[tps_df[target_col_name] == "Unknown"]
    tps_df_pos = tps_df[tps_df[target_col_name] != "Unknown"]

    group_target_2_count: defaultdict[tuple, int] = defaultdict(int)
    id_2_targets = tps_df_pos.groupby("Uniprot ID")[target_col_name].agg(set)
    id_2_group = {
        uni_id: group
        for uni_id, group in id_2_group.items()
        if uni_id in set(id_2_targets.index)
    }
    for node_i, partition_number in id_2_group.items():
        target_values = id_2_targets.loc[node_i]
        for target_value in target_values:
            group_target_2_count[(partition_number, target_value)] += 1
    unsplittable_target_values = set()
    target_val_2_total_count = tps_df_pos[target_col_name].value_counts()

    for (_, target_val), count_in_single_cc in group_target_2_count.items():
        total_target_val_occurrences = target_val_2_total_count.loc[target_val]
        if (
            count_in_single_cc / total_target_val_occurrences
            > max_allowed_proportion_of_class_in_cc
            or total_target_val_occurrences < args.n_folds
        ):
            unsplittable_target_values.add(target_val)

    logger.info("Number of target categories originally: %d", len(major_classes))
    for unsplittable_target_value in unsplittable_target_values:
        try:
            major_classes.remove({unsplittable_target_value})
        except ValueError:
            # not a major class
            continue
    logger.info(
        "Number of target categories after removing categories covered exclusively by the same clade of the phylogenetic tree: %d",
        len(major_classes),
    )

    id_2_targets_df = (
        tps_df_pos.groupby("Uniprot ID")[[target_col_name]].agg(set).reset_index()
    )
    logger.info(
        "Number of positive protein IDs entering K-Fold computation: %d",
        len(id_2_targets_df),
    )

    total_classes_distribution = get_major_classes_distribution(
        id_2_targets_df,
        target_col=target_col_name,
        major_classes=major_classes,
    )
    best_random_state = None
    worst_random_state = None
    min_max_jensenshannon_val = float("inf")
    max_avg_jensenshannon_val = -float("inf")
    major_classes_set = {val.copy().pop() for val in major_classes}

    id_2_targets_df["cc_group"] = id_2_targets_df["Uniprot ID"].map(
        lambda x: str(id_2_group[x])
        if x in id_2_group and len(id_2_targets.loc[x].intersection(major_classes_set))
        else (x if isinstance(x, str) else str(uuid.uuid4()))
    )

    id_2_targets_df[target_col_name] = id_2_targets_df[target_col_name].map(
        lambda x: x if isinstance(x, str) else "missing"
    )

    id_2_targets_df[f"{target_col_name}_sorted_set"] = id_2_targets_df[
        target_col_name
    ].map(lambda targets: str(sorted(targets)))

    for random_state in range(500):
        _t = []
        kfold = StratifiedGroupKFold(
            n_splits=args.n_folds, shuffle=True, random_state=random_state
        )

        folds = list(
            kfold.split(
                id_2_targets_df,
                id_2_targets_df[f"{target_col_name}_sorted_set"],
                id_2_targets_df["cc_group"],
            )
        )
        for trn_idx, val_idx in folds:
            val_df = id_2_targets_df.iloc[val_idx]
            fold_classes_distribution = get_major_classes_distribution(
                val_df,
                target_col=target_col_name,
                major_classes=major_classes,
            )
            _t.append(
                jensenshannon(total_classes_distribution, fold_classes_distribution)
            )
        max_jensenshannon = np.mean(_t)
        if max_jensenshannon < min_max_jensenshannon_val:
            min_max_jensenshannon_val = max_jensenshannon
            logger.info(
                "Fond a better split with min_max_jensenshannon_val of %s",
                min_max_jensenshannon_val,
            )
            best_random_state = random_state
        if max_jensenshannon > max_avg_jensenshannon_val:
            max_avg_jensenshannon_val = max_jensenshannon
            worst_random_state = random_state

    with open(
        "data/stratified_phylogeny_based_best_random_state.json",
        "w",
    ) as file:
        json.dump(best_random_state, file)

    kfold = StratifiedGroupKFold(
        n_splits=args.n_folds, shuffle=True, random_state=best_random_state
    )
    folds = list(
        kfold.split(
            id_2_targets_df,
            id_2_targets_df[target_col_name].map(lambda x: str(sorted(x))),
            id_2_targets_df["cc_group"],
        )
    )
    id_folds_pos = [
        set(id_2_targets_df["Uniprot ID"].values[val_idx]) for _, val_idx in folds
    ]

    best_folds_distributions = []
    for _, val_idx in folds:
        val_df = id_2_targets_df.iloc[val_idx]
        fold_classes_distribution = get_major_classes_distribution(
            val_df,
            target_col=target_col_name,
            major_classes=major_classes,
        )
        best_folds_distributions.append(fold_classes_distribution)

    # for visualization of stratification balancing
    kfold_worst = StratifiedGroupKFold(
        n_splits=args.n_folds, shuffle=True, random_state=worst_random_state
    )
    folds_worst = list(
        kfold_worst.split(
            id_2_targets_df,
            id_2_targets_df[target_col_name].map(lambda x: str(sorted(x))),
            id_2_targets_df["cc_group"],
        )
    )
    worst_folds_distributions = []
    for _, val_idx in folds_worst:
        val_df = id_2_targets_df.iloc[val_idx]
        fold_classes_distribution = get_major_classes_distribution(
            val_df,
            target_col=target_col_name,
            major_classes=major_classes,
        )
        worst_folds_distributions.append(fold_classes_distribution)
    with open("outputs/logs/worst_phylogeny_folds_distributions.pkl", "wb") as file:
        pickle.dump(worst_folds_distributions, file)

    with open("outputs/logs/best_phylogeny_folds_distributions.pkl", "wb") as file:
        pickle.dump(best_folds_distributions, file)

    id_folds_neg = [
        set(tps_df_neg["Uniprot ID"].values[val_idx])
        for _, val_idx in kfold_neg.split(tps_df_neg, tps_df_neg[target_col_name])
    ]

    if desc is None:
        desc = args.split_desc

    tuple_ids_all = (
        tps_df[
            [
                "Uniprot ID",
                "Amino acid sequence",
                "SMILES_substrate_canonical_no_stereo",
                "SMILES_product_canonical_no_stereo",
            ]
        ]
        .apply(
            lambda row: (
                row["Uniprot ID"],
                row["Amino acid sequence"],
                row["SMILES_substrate_canonical_no_stereo"],
                row["SMILES_product_canonical_no_stereo"],
            ),
            axis=1,
        )
        .values
    )

    with h5py.File("data/tps_folds_nov2023.h5", "a") as h5_file:
        group = h5_file.create_group(desc)

        if len(unsplittable_target_values):
            group.create_dataset(
                name="unsplittable_target_values",
                shape=(len(unsplittable_target_values), 1),
                dtype="S105",
                data=[n.encode("ascii", "ignore") for n in unsplittable_target_values],
            )

        for fold_i in range(args.n_folds):
            val_indices = tuple_ids_all[
                tps_df["Uniprot ID"].isin(
                    id_folds_pos[fold_i].union(id_folds_neg[fold_i])
                )
            ].astype(triplets_dtype)
            group.create_dataset(
                f"{fold_i}",
                data=val_indices,
            )


if __name__ == "__main__":
    cli_args = parse_args()
    logger.info("Loading data...")
    terpene_synthases_df = get_tps_df(
        path_to_file=cli_args.tps_cleaned_csv_path,
        path_to_sampled_negatives=cli_args.negative_samples_path,
    )
    terpene_synthases_df = terpene_synthases_df.drop_duplicates("Uniprot ID")
    terpene_synthases_df = terpene_synthases_df[
        terpene_synthases_df["is_substrate_predicted"].isin({1, "Unknown"})
    ]
    logger.info(
        "Data were loaded! There are %s loaded records in total.",
        len(terpene_synthases_df),
    )

    logger.info(
        "Computing a balanced StratifiedGroupKFold for split named %s",
        cli_args.split_description,
    )
    stratified_kfold_phylogeny_based(
        terpene_synthases_df,
        cli_args,
        target_col_name="SMILES_substrate_canonical_no_stereo",
        major_classes=[
            {"CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {"CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {"CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {"CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C"},
            {"CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O"},
            {
                "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O"
            },
            {
                "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O"
            },
            {"CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {"CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {
                "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O"
            },
            {"CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC=C(C)C"},
            {"Negative"},
            {
                "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O"
            },
            {"CC1(C)CCCC2(C)C1CCC(C)(O)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {
                "CC(CCC=C(C)CCC=C(C)C)=CC1C(COP([O-])(=O)OP([O-])([O-])=O)C1(C)CCC=C(C)CCC=C(C)C"
            },
            {"CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"},
            {
                "CC(CCC=C(C)CCC=C(C)CCC=C(C)C)=CC1C(COP(O)(=O)OP(O)(O)=O)C1(C)CCC=C(C)CCC=C(C)CCC=C(C)C"
            },
            {"CC(C)=CCCC(C)=CCOP(O)(=O)OP(O)(O)=O"},
            {"CC(C)=CCCC(C)=C(C)COP([O-])(=O)OP([O-])([O-])=O"},
        ],
        desc=cli_args.split_description,
    )
