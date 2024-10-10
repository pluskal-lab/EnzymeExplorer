# pylint: disable=R0801
"""A helper script to gather classifier checkpoints from an output directory"""
import argparse
import pickle

# pylint: disable=unused-import
import scipy.stats  # type: ignore
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
        "--n-folds", help="A number of folds used in CV", type=int, default=5
    )
    parser.add_argument(
        "--output-path",
        help="A file to save the model checkpoints to",
        type=str,
        default="data/classifier_checkpoints.pkl",
    )
    parser.add_argument(
        "--use-all-folds",
        help="A flag to use all folds instead of individual fold checkpoints",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    config_root_path = get_config_root()
    experiment_kwargs = collect_single_experiment_arguments(config_root_path)
    experiment_info = ExperimentInfo(**experiment_kwargs)
    args = parse_args()
    n_folds = args.n_folds
    experiment_output_folder_root = (
        get_output_root() / experiment_info.model_type / experiment_info.model_version
    )
    # retrieve the model class
    assert (
        experiment_output_folder_root.exists()
    ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"
    # discover available fold results
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

    classifiers = []
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
        model.classifier.classes_ = model.config.class_names
        if hasattr(model, "allowed_feat_indices"):
            with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
                domain_module_id_2_dist_matrix_index = pickle.load(file)[-1]

            with open("data/domains_subset.pkl", "rb") as file:
                feat_indices_subset = pickle.load(file)[-1]
            domain_module_id_2_dist_matrix_index_subset = {domain_id: [i for i in indices if i in feat_indices_subset]
                                                           for domain_id, indices in
                                                           domain_module_id_2_dist_matrix_index.items()
                                                           if len([i for i in indices if i in feat_indices_subset])}
            feat_idx_2_module_id = {}
            for module_id, feat_indices in domain_module_id_2_dist_matrix_index_subset.items():
                for feat_idx in feat_indices:
                    feat_idx_2_module_id[feat_idx] = module_id
            order_of_domain_modules = [feat_idx_2_module_id[feat_i] for feat_i in model.allowed_feat_indices]
            model.classifier.order_of_domain_modules = order_of_domain_modules
        classifiers.append(model.classifier)

    with open(args.output_path, "wb") as file_writer:
        pickle.dump(classifiers, file_writer)
