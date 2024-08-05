# pylint: disable=R0801
"""The helper script to obtain feature importances for the domains and select the most important ones.
Usage: python -m src.models.plm_domain_faster.get_domains_feature_importances
"""

import argparse
import pickle

import pandas as pd  # type: ignore

from src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
)
from src.utils.project_info import ExperimentInfo, get_config_root, get_output_root


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to gather classifier checkpoints from an output directory"
    )
    parser.add_argument(
        "--top-most-important-domain-features-per-model",
        help="A number of top features computed from domains to take from each model (for 5-fold CV, there are 5 models)",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--output-path",
        help="A file to save the selected domains",
        type=str,
        default="data/domains_subset.pkl",
    )
    parser.add_argument(
        "--domain-features-path",
        help="A file with precomputed domain features",
        type=str,
        default="data/clustering__domain_dist_based_features.pkl",
    )
    parser.add_argument(
        "--n-folds", help="A number of folds used in CV", type=int, default=5
    )
    parser.add_argument(
        "--tps-file-path",
        help="A path to the TPS file",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    config_root_path = get_config_root()
    experiment_kwargs = collect_single_experiment_arguments(config_root_path)
    experiment_info = ExperimentInfo(**experiment_kwargs)
    args = parse_args()

    with open(args.domain_features_path, "rb") as file:
        (
            feats_dom_dists,
            all_ids_list_dom,
            uniid_2_column_ids,
            domain_module_id_2_dist_matrix_index,
        ) = pickle.load(file)

    idx_2_domain_id = {}
    for domain_id, indices in domain_module_id_2_dist_matrix_index.items():
        for i in indices:
            idx_2_domain_id[i] = domain_id

    n_folds = args.n_folds
    experiment_output_folder_root = (
            get_output_root() / experiment_info.model_type / experiment_info.model_version
    )
    assert (
        experiment_output_folder_root.exists()
    ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"
    model_version_fold_folders = {
        x.stem for x in experiment_output_folder_root.glob("*")
    }
    if (
            len(model_version_fold_folders.intersection(set(map(str, range(n_folds)))))
            == n_folds
    ):
        fold_2_root_dir = {
            fold_i: experiment_output_folder_root / f"{fold_i}"
            for fold_i in range(n_folds)
        }
    elif "all_folds" in model_version_fold_folders:
        fold_2_root_dir = {
            fold_i: experiment_output_folder_root / "all_folds"
            for fold_i in range(n_folds)
        }
    else:
        raise NotImplementedError(
            f"Not all fold outputs found. Please run corresponding experiments ({experiment_info}) before evaluation"
        )

    domains_subset: set = set()
    feat_indices_subset: set = set()
    for fold_i, fold_root_dir in fold_2_root_dir.items():
        fold_class_path = fold_root_dir / "all_classes"
        assert fold_class_path.exists(), "Only all_classes are supported"
        try:
            fold_class_latest_path = sorted(fold_class_path.glob("*"))[-1]
        except IndexError as index_error:
            raise NotImplementedError(
                f"Please run corresponding experiments ({experiment_info}) before evaluation"
            ) from index_error
        with open(fold_class_latest_path / f"model_fold_{fold_i}.pkl", "rb") as file:
            model = pickle.load(file)
        importances = model.classifier.feature_importances_
        number_of_domain_comparisons = len(model.allowed_feat_indices)
        plm_embedding_size = len(importances) - number_of_domain_comparisons
        feature_names = [f"tps_{i}" for i in range(plm_embedding_size)] + [
            idx_2_domain_id[feat_i] for feat_i in model.allowed_feat_indices
        ]
        forest_importances = pd.Series(importances, index=feature_names)
        forest_importances_domains = pd.Series(
            importances[plm_embedding_size:],
            index=[idx_2_domain_id[feat_i] for feat_i in model.allowed_feat_indices],
        )
        forest_importances_indices = pd.Series(
            importances[plm_embedding_size:],
            index=model.allowed_feat_indices,
        )
        domains_subset = domains_subset.union(
            set(
                forest_importances_domains.sort_values(ascending=False)
                .iloc[: args.top_most_important_domain_features_per_model]
                .index
            )
        )
        feat_indices_subset = feat_indices_subset.union(
            set(
                forest_importances_indices.sort_values(ascending=False)
                .iloc[: args.top_most_important_domain_features_per_model]
                .index
            )
        )

    terpene_synthases_df = pd.read_csv(args.tps_file_path)
    ids_rare_set = set(
        terpene_synthases_df.loc[
            terpene_synthases_df["Type (mono, sesq, di, …)"].isin({"tetra", "sester"}),
            "Uniprot ID",
        ].unique()
    )
    for domain_id in domain_module_id_2_dist_matrix_index.keys():
        uni_id = domain_id.split("_")[0]
        if uni_id in ids_rare_set:
            domains_subset.add(domain_id)

    with open(args.output_path, "wb") as file_write:
        pickle.dump((domains_subset, feat_indices_subset), file_write)
