# pylint: disable=R0801
"""The helper script to obtain feature importances for the domains and select the most important ones.
Usage: python -m src.models.plm_domain_faster.get_domains_feature_importances
"""

import argparse
import pickle

import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore

from terpeneminer.src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
)
from terpeneminer.src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_output_root,
)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to gather classifier checkpoints from an output directory"
    )
    parser.add_argument(
        "--top-most-important-plm-features-per-model",
        help="A number of top features to take from each model (for 5-fold CV, there are 5 models)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-path",
        help="A file to save the selected domains",
        type=str,
        default="data/plm_feats_subset.pkl",
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
    parser.add_argument(
        "--use-all-folds",
        help="A flag to use all folds instead of individual fold checkpoints",
        action="store_true",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    config_root_path = get_config_root()
    args = parse_args()
    if args.model is None or args.model_version is None:
        experiment_kwargs = collect_single_experiment_arguments(config_root_path)
    else:
        experiment_kwargs = {
            "model_type": args.model,
            "model_version": args.model_version,
        }
    experiment_info = ExperimentInfo(**experiment_kwargs)

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
    if not args.use_all_folds and (
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
        if isinstance(model.classifier, RandomForestClassifier):
            classifiers_fold = [model.classifier]
        elif isinstance(model.classifier, MultiOutputClassifier):
            classifiers_fold = []
            mo_estimators = model.classifier.estimators_
            for mo_estimator in mo_estimators:
                if isinstance(mo_estimator, RandomForestClassifier):
                    classifiers_fold.append(mo_estimator)
                elif isinstance(mo_estimator, CalibratedClassifierCV):
                    for calibrated_classifier in mo_estimator.calibrated_classifiers_:
                        if isinstance(calibrated_classifier.estimator, RandomForestClassifier):
                            classifiers_fold.append(calibrated_classifier.estimator)
        for classifier_ in classifiers_fold:
            importances = classifier_.feature_importances_
            try:
                number_of_domain_comparisons = len(model.allowed_feat_indices)
            except AttributeError:
                number_of_domain_comparisons = 0
            plm_embedding_size = len(importances) - number_of_domain_comparisons
            feature_names = [f"tps_{i}" for i in range(plm_embedding_size)]
            if number_of_domain_comparisons:
                feature_names += [
                    idx_2_domain_id[feat_i] for feat_i in model.allowed_feat_indices
                ]
            forest_importances = pd.Series(importances, index=feature_names)
            forest_importances_indices = pd.Series(
                importances[:plm_embedding_size],
                index=list(range(plm_embedding_size)),
            )
            feat_indices_subset = feat_indices_subset.union(
                set(
                    forest_importances_indices.sort_values(ascending=False)
                    .iloc[: args.top_most_important_plm_features_per_model]
                    .index
                )
            )
    print('feat_indices_subset size: ', len(feat_indices_subset))
    with open(args.output_path, "wb") as file_write:
        pickle.dump(feat_indices_subset, file_write)
