"""This file contains an experiment runner
which is capable of gathering all the required pieces of information for a particular experiment
and consequently performing the computational experiment, i.e. instantiating, training and scoring the selected model"""

import inspect
import logging
import os.path
import pickle

import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

from terpeneminer.src import models
from terpeneminer.src.models.ifaces import BaseConfig, BaseModel
from terpeneminer.src.utils.data import get_folds, get_tps_df
from terpeneminer.src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_output_root,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def run_experiment(experiment_info: ExperimentInfo, load_hyperparameters: bool = False):
    """
    This function gathers all the required pieces of information for a particular experiment
    and consequently runs the experiment, i.e. instantiating, training and scoring the selected model
    """
    # retrieve the model class
    try:
        model_class = getattr(models, experiment_info.model_type)
    except AttributeError as ex:
        raise NotImplementedError(
            f"Configured model {experiment_info.model_type} not found. The available models are "
            f"""{','.join([model_type
                  for model_type, model_class in inspect.getmembers(models, inspect.isclass)
                  if issubclass(model_class, BaseModel)])}"""
        ) from ex

    if not issubclass(model_class, BaseModel):
        raise ValueError(
            f'Model class must be a child class of "BaseModel".\n{experiment_info.model_type} is not'
        )
    logger.info(
        "The model class for %s has been successfully retrieved",
        experiment_info.model_type,
    )

    # retrieve the corresponding config
    config_class = model_class.config_class()
    if not issubclass(config_class, BaseConfig):
        raise ValueError(
            f'Config class must be a child class of "BaseConfig".\n{type(config_class)} is not'
        )
    config_path = (
        get_config_root()
        / experiment_info.model_type
        / experiment_info.model_version
        / "config.yaml"
    )
    config_dict = BaseConfig.load(config_path)
    config_dict.update({"experiment_info": experiment_info})
    config = config_class(**config_dict)
    logger.info(
        "The config for %s has been loaded and instantiated",
        experiment_info.model_type,
    )

    # accessing the configured class name, if present
    if experiment_info.class_name != "all_classes":
        config.class_name = experiment_info.class_name

    if hasattr(config, "gpu_id"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    # instantiating model
    model = model_class(config)
    if load_hyperparameters:
        try:
            per_class_optimization = model.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        if per_class_optimization:
            raise NotImplementedError(
                "Please implement loading of outputs for per-class optimization, it wasn't needed before"
            )
        # in the future, refactor hyperparameters loading as a common routine used here and in hyperparameter tuning
        # pylint: disable=R0801
        logger.info("Looking for hyperparameters optimization results...")
        n_folds = len(
            get_folds(
                split_desc=config.split_col_name,
            )
        )
        experiment_output_folder_root = (
            get_output_root()
            / experiment_info.model_type
            / experiment_info.model_version
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
            logger.info("Found %d fold results for %s", n_folds, str(experiment_info))
            fold_2_root_dir = {
                f"{fold_i}": experiment_output_folder_root / f"{fold_i}"
                for fold_i in range(n_folds)
            }
        elif "all_folds" in model_version_fold_folders:
            logger.info("Found all_folds results for %s", f"{experiment_info}")
            fold_2_root_dir = {
                str(fold_i): experiment_output_folder_root / "all_folds"
                for fold_i in range(n_folds)
            }
        else:
            raise NotImplementedError(
                f"Not all fold outputs found. Please run corresponding experiments ({experiment_info}) before evaluation"
            )

    logger.info(
        "Instantiated the model %s", model.config.experiment_info.get_experiment_name()
    )

    tps_df = pd.read_csv(config.tps_cleaned_csv_path)
    tps_df.loc[
        tps_df["Type (mono, sesq, di, …)"].isin(
            {"ggpps", "fpps", "gpps", "gfpps", "hsqs"}
        ),
        "SMILES_substrate_canonical_no_stereo",
    ] = "precursor substr"

    try:
        save_trained_model = config.save_trained_model
    except AttributeError:
        save_trained_model = False
    # iterating over folds
    with logging_redirect_tqdm([logger]):
        # pylint: disable=too-many-nested-blocks
        for test_fold in tqdm(
            get_folds(
                split_desc=config.split_col_name,
            ),
            desc=f"Iterating over validation folds per {config.split_col_name}..",
        ):
            # selecting a single fold to run if specified
            if experiment_info.fold in {"all_folds", test_fold}:
                logger.info("Fold: %s", test_fold)
                fold_needs_resetting = experiment_info.fold == "all_folds"
                model.config.experiment_info.fold = test_fold
                trn_folds = [
                    f"fold_{fold_trn}"
                    for fold_trn in get_folds(
                        split_desc=config.split_col_name,
                    )
                    if fold_trn != test_fold
                ]
                trn_df = tps_df[tps_df[config.split_col_name].isin(set(trn_folds))]
                trn_df.loc[
                    trn_df[f"{config.split_col_name}_ignore_in_eval"] == 1,
                    config.target_col_name,
                ] = "other"
                trn_df = (
                    trn_df.groupby(config.id_col_name)[config.target_col_name]
                    .agg(set)
                    .reset_index()
                )
                trn_df[config.target_col_name] = trn_df[config.target_col_name].map(
                    lambda x: x
                    if len(x.intersection({"Unknown", "precursor substr"}))
                    else x.union({"isTPS"})
                )

                if config.run_against_wetlab:
                    test_df_raw = get_tps_df(
                        path_to_file="data/df_wetlab_long_clean.csv",
                        path_to_sampled_negatives="data/sampled_id_2_seq_experimental.pkl",
                        id_col_name="ID",
                        remove_fragments=False,
                    )
                    test_id_column_name = "ID"
                    raw_dataset_id_colunm_name = config.id_col_name
                    trn_df[test_id_column_name] = trn_df[raw_dataset_id_colunm_name]
                    tps_df[test_id_column_name] = tps_df[raw_dataset_id_colunm_name]
                    model.config.id_col_name = test_id_column_name
                else:
                    test_df_raw = tps_df[
                        tps_df[config.split_col_name] == f"fold_{test_fold}"
                    ]
                    test_df_raw.loc[
                        test_df_raw[f"{config.split_col_name}_ignore_in_eval"] == 1,
                        config.target_col_name,
                    ] = "other"
                    test_id_column_name = config.id_col_name
                    model.config.id_col_name = test_id_column_name
                test_df = (
                    test_df_raw.groupby(test_id_column_name)[config.target_col_name]
                    .agg(set)
                    .reset_index()
                )
                test_df[config.target_col_name] = test_df[config.target_col_name].map(
                    lambda x: x
                    if len(x.intersection({"Unknown", "precursor substr"}))
                    else x.union({"isTPS"})
                )

                # checking if the model requires an amino acid sequence or a group (kingdom) column
                for optional_column_attribute in ["seq_col_name", "group_column_name"]:
                    if (
                        hasattr(config, optional_column_attribute)
                        and getattr(config, optional_column_attribute) is not None
                    ):
                        id_seq_df = tps_df[
                            [
                                raw_dataset_id_colunm_name,
                                getattr(config, optional_column_attribute),
                            ]
                        ].drop_duplicates(raw_dataset_id_colunm_name)
                        trn_df = trn_df.merge(
                            id_seq_df,
                            on=raw_dataset_id_colunm_name,
                        )
                        test_id_seq_df = test_df_raw[
                            [
                                test_id_column_name,
                                getattr(config, optional_column_attribute),
                            ]
                        ].drop_duplicates(test_id_column_name)
                        test_df = test_df.merge(
                            test_id_seq_df,
                            on=test_id_column_name,
                        )

                # retrieving hyperparameters
                if load_hyperparameters:
                    fold_root_dir = fold_2_root_dir[test_fold]
                    logger.info(
                        "Loading hyperparameters for fold %s with root dir %s",
                        test_fold,
                        str(fold_root_dir),
                    )
                    # in the future, refactor hyperparameters loading as a common routine used here and in hyperparameter tuning
                    # pylint: disable=R0801
                    if per_class_optimization:
                        class_names = [
                            model.config.class_names
                            if not hasattr(model.config, "class_name")
                            else [model.config.class_name]
                        ]
                    else:
                        class_names = ["all_classes"]
                    for class_name in class_names:
                        if class_name not in {"Unknown", "other"}:
                            if (fold_root_dir / f"{class_name}").exists():
                                fold_class_path = fold_root_dir / f"{class_name}"
                            elif (fold_root_dir / "all_classes").exists():
                                fold_class_path = fold_root_dir / "all_classes"
                            else:
                                raise ValueError("No fold_class_path found")
                            previous_results = list(
                                fold_class_path.glob(
                                    "*/hyperparameters_optimization/optimization_results_detailed_*.pkl"
                                )
                            )
                            if previous_results:
                                logger.info(
                                    "Found previous results for class %s: %s",
                                    class_name,
                                    previous_results,
                                )
                                with open(previous_results[0], "rb") as file:
                                    best_params, _, _ = pickle.load(file)
                                model.set_params(**best_params)
                                logger.info("Loaded previous best hyperparameters")

                # fitting the model
                model.fit(trn_df)
                logger.info(
                    "Trained model %s (%s), fold %s",
                    experiment_info.model_type,
                    experiment_info.model_version,
                    test_fold,
                )
                if save_trained_model:
                    model.save()

                # scoring the model
                val_proba_np = model.predict_proba(test_df)
                with open(
                    model.output_root / f"fold_{test_fold}_results.pkl", "wb"
                ) as file:
                    pickle.dump((val_proba_np, model.config.class_names, test_df), file)
                if fold_needs_resetting:
                    experiment_info.fold = "all_folds"
