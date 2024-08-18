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
from terpeneminer.src.utils.project_info import ExperimentInfo, get_config_root

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def run_experiment(experiment_info: ExperimentInfo):
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
