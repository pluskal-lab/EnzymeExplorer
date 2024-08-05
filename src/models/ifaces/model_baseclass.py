"""This module defines an abstract class for models"""
import inspect
import json
import logging
import os
import pickle
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn.base  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from skopt import gp_minimize  # type: ignore
from skopt.space import Categorical, Integer, Real  # type: ignore
from skopt.utils import use_named_args  # type: ignore

from src.models.ifaces.config_baseclasses import BaseConfig
from src.utils.project_info import get_output_root

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class BaseModel(ABC, BaseEstimator):
    """Base model class"""

    def __init__(
        self,
        config: BaseConfig,
    ):
        self.config = config
        assert isinstance(self.config.experiment_info._timestamp, datetime)
        self.output_root = (
            get_output_root()
            / config.experiment_info.model_type
            / config.experiment_info.model_version
            / config.experiment_info.fold
            / config.experiment_info.class_name
            / config.experiment_info._timestamp.strftime("%Y%m%d-%H%M%S")
        )
        self.output_root.mkdir(exist_ok=True, parents=True)
        self.classifier_class: sklearn.base.BaseEstimator

    @abstractmethod
    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """
        Function for training model instance
        :param train_df: pandas dataframe containing training data
        :param class_name: name of a class for the separate model fitting for the class
        """
        raise NotImplementedError

    def save(self, experiment_output_folder: Optional[Path | str] = None):
        """
        Function for model persistence
        """
        if experiment_output_folder is None:
            experiment_output_folder = self.output_root
        experiment_output_folder = Path(experiment_output_folder)
        with open(
            experiment_output_folder
            / f"model_fold_{self.config.experiment_info.fold}.pkl",
            "wb",
        ) as file:
            pickle.dump(self, file)

    def fit(self, train_df: pd.DataFrame):
        """
        Fit function
        :param train_df: pandas dataframe containing training data
        """
        try:
            per_class_optimization = self.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        if self.config.optimize_hyperparams:
            try:
                reuse_existing_partial_results = (
                    self.config.reuse_existing_partial_results
                )
            except AttributeError:
                reuse_existing_partial_results = False

            self.optimize_hyperparameters(
                train_df,
                n_calls=self.config.n_calls_hyperparams_opt,
                per_class_optimization=per_class_optimization,
                n_fold_splits=5,
                reuse_existing_partial_results=reuse_existing_partial_results,
                phylogenetic_clusters_path="data/phylogenetic_clusters.pkl",
                **self.config.hyperparam_dimensions,
            )
        try:
            load_per_class_params_from = self.config.load_per_class_params_from
        except AttributeError:
            load_per_class_params_from = None
        if load_per_class_params_from:
            previous_results = list(
                Path(load_per_class_params_from).glob(
                    "*/hyperparameters_optimization/best_params_*.json"
                )
            )
            assert previous_results, (
                f"Requested to load per-class parameters from {load_per_class_params_from},"
                "but no parameters in json are found"
            )
            logger.info("Loading hyper parameters from: %s", previous_results[0])
            with open(previous_results[0], "r", encoding="utf-8") as file:
                best_params = json.load(file)
            self.set_params(**best_params)

        if (
            self.config.optimize_hyperparams or load_per_class_params_from
        ) and per_class_optimization:
            class_names = (
                self.config.class_names
                if not hasattr(self.config, "class_name")
                else [self.config.class_name]
            )
            for class_name in class_names:
                self.fit_core(train_df, class_name=class_name)
        else:
            self.fit_core(train_df)

    @abstractmethod
    def predict_proba(
        self,
        val_df: pd.DataFrame | np.ndarray,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:  # pylint: disable=R0801
        """
        It's a function returning predicted probabilities per either all classes or only for the selected class
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        It's a generic function setting values of all parameters in the kwargs
        """
        for attribute_name, value in kwargs.items():
            if attribute_name not in {"class_name", "per_class"}:
                setattr(self, attribute_name, value if value != "None" else None)

    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        n_calls: int,
        per_class_optimization: bool,
        n_fold_splits: int,
        reuse_existing_partial_results: bool,
        phylogenetic_clusters_path: str,
        **dimension_params,
    ):
        """
        This function performed hyperparameter tuning using algorithms based on gaussian process regression
        """
        logger.info("Starting hyperparameter optimization...")
        if per_class_optimization:
            class_names = (
                self.config.class_names
                if not hasattr(self.config, "class_name")
                else [self.config.class_name]
            )
        else:
            class_names = ["all_classes"]
        for class_name in class_names:
            logger.info("Optimization of hyperparameters for %s", class_name)
            prefix = "" if class_name == "all_classes" else f"{class_name}_"
            if reuse_existing_partial_results:
                previous_results = list(
                    self.output_root.glob(
                        f"../*/hyperparameters_optimization/optimization_results_detailed_{prefix}*.pkl"
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
                    self.set_params(**best_params)
                    logger.info("Restored previous results")
                    continue

            type_2_skopt_class = {
                "categorical": Categorical,
                "float": Real,
                "int": Integer,
            }
            dimensions = []
            # x0 -> to enforce evaluation of the default parameters
            initial_instance_parameters = self.__dict__
            if hasattr(self, "classifier_class"):
                classifier_attributes = inspect.getfullargspec(
                    self.classifier_class
                ).kwonlydefaults
                if classifier_attributes is not None:
                    classifier_attributes.update(initial_instance_parameters)
                    initial_instance_parameters = classifier_attributes
            hyperparam_combinations = []
            for name, characteristics in dimension_params.items():
                if characteristics["type"] != "categorical":
                    next_dims = type_2_skopt_class[characteristics["type"]](
                        *characteristics["args"], name=name
                    )
                else:
                    next_dims = type_2_skopt_class[characteristics["type"]](
                        characteristics["args"], name=name
                    )
                dimensions.append(next_dims)
                assert (
                    name in initial_instance_parameters
                ), f"Hyperparameter {name} does not seem to be a model attribute"
                val = initial_instance_parameters[name]
                hyperparam_combinations.append(val if val is not None else "None")
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            params_output_root = self.output_root / "hyperparameters_optimization"
            logger.info(
                "Hyperparam optimization results will be stored in %s",
                str(params_output_root),
            )
            if not params_output_root.exists():
                params_output_root.mkdir(exist_ok=True, parents=True)

            # The objective function to be minimized
            def make_objective(train_df, space, cross_validation):
                # This decorator converts your objective function with named arguments into one that
                # accepts a list as argument, while doing the conversion automatically
                @use_named_args(space)
                def objective_value(**params):
                    if class_name == "all_classes":
                        prefix = ""
                    else:
                        prefix = f"{class_name}_"
                    params = {
                        f"{prefix}{param_name}": param_value
                        if param_value != "None"
                        else None
                        for param_name, param_value in params.items()
                    }
                    self.set_params(**params)
                    map_scores = []
                    available_ids = train_df[self.config.id_col_name]
                    id_2_classes = train_df[
                        [self.config.id_col_name, self.config.target_col_name]
                    ].set_index(self.config.id_col_name)
                    for trn_idx, val_idx in cross_validation.split(
                        available_ids,
                        y=available_ids.map(
                            lambda uniprot_id: str(sorted(id_2_classes.loc[uniprot_id]))
                        )
                        if not per_class_optimization
                        else available_ids.map(
                            lambda uniprot_id: class_name
                            in id_2_classes.loc[uniprot_id]
                        ).astype(int),
                        groups=train_df["seq_group"],
                    ):
                        trn_ids = set(available_ids.iloc[trn_idx].values)
                        val_ids = set(available_ids.iloc[val_idx].values)
                        trn_df = train_df[
                            train_df[self.config.id_col_name].isin(trn_ids)
                        ]
                        vl_df = train_df[
                            train_df[self.config.id_col_name].isin(val_ids)
                        ]
                        selected_class_name = (
                            class_name if per_class_optimization else None
                        )
                        self.fit_core(
                            trn_df,
                            class_name=selected_class_name,
                        )
                        map_scores.append(
                            eval_model_mean_average_precision_neg(
                                self,
                                vl_df,
                                selected_class_name=selected_class_name,
                            )
                        )
                    score = np.nanmean(map_scores)

                    ckpts = list(params_output_root.glob("*_params.json"))
                    if len(ckpts) > 0:
                        past_performances = sorted(
                            [
                                float(str(ckpt_name.stem).split("_", maxsplit=1)[0])
                                for ckpt_name in ckpts
                            ]
                        )
                    if len(ckpts) == 0 or past_performances[-1] > score:
                        for ckpt in ckpts:
                            os.remove(ckpt)
                        with open(
                            params_output_root
                            / f"{100 - 100*score:.0f}_params_{run_timestamp}.json",
                            "w",
                            encoding="utf8",
                        ) as file:
                            json.dump(
                                {
                                    key: (
                                        val
                                        if not isinstance(val, np.integer)
                                        else int(val)
                                    )
                                    for key, val in params.items()
                                },
                                file,
                            )

                    return score

                return objective_value

            k_fold = StratifiedGroupKFold(
                n_splits=n_fold_splits, shuffle=True, random_state=42
            )
            with open(phylogenetic_clusters_path, "rb") as file:
                id_2_group, _ = pickle.load(file)
            train_df["seq_group"] = train_df[self.config.id_col_name].map(
                lambda x: str(id_2_group[x])
                if x in id_2_group
                else (x if isinstance(x, str) else str(uuid.uuid4()))
            )

            objective = make_objective(
                train_df, space=dimensions, cross_validation=k_fold
            )

            gp_round = gp_minimize(
                func=objective,
                dimensions=dimensions,
                acq_func="gp_hedge",
                n_calls=n_calls,
                n_initial_points=min(10, n_calls // 5),
                random_state=42,
                verbose=True,
                x0=hyperparam_combinations,
            )
            best_params = {
                f"{prefix}{dimensions[i].name}": param_value
                for i, param_value in enumerate(gp_round.x)
            }
            self.set_params(**best_params)

            with open(
                params_output_root
                / f"optimization_results_detailed_{prefix}{run_timestamp}.pkl",
                "wb",
            ) as file_writer:
                pickle.dump(
                    (best_params, gp_round.x_iters, gp_round.func_vals), file_writer
                )

        def _jsonify_value(value):
            if isinstance(value, np.int64):
                return int(value)
            # if isinstance(value, np.float): # should not be a problem in the latest versions
            #     return float(value)
            return value

        # just in case all hyperparameters for all classes were already pre-computed
        if not (self.output_root / "hyperparameters_optimization").exists():
            (self.output_root / "hyperparameters_optimization").mkdir(exist_ok=True)
        with open(
            self.output_root
            / "hyperparameters_optimization"
            / f"best_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
            "w",
            encoding="utf-8",
        ) as file_text_io:
            json.dump(
                {
                    param: _jsonify_value(val)
                    for param, val in self.get_model_specific_params().items()
                },
                file_text_io,
            )

        logger.info("Hyperparameter optimization finished!")

    @classmethod
    @abstractmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of a config class
        """
        raise NotImplementedError

    def get_params(self, deep: bool = True):
        """
        Function retrieving parameters
        """
        return {
            "config": (
                deepcopy(self.__dict__["config"]) if deep else self.__dict__["config"]
            )
        }

    def get_model_specific_params(self, class_name: str = None):
        """
        method inspecting parameters and retrieving ones related to the model object
        """
        initial_instance_parameters = self.__dict__
        try:
            classifier_args = inspect.getfullargspec(self.classifier_class)
            if isinstance(classifier_args.kwonlydefaults, dict):
                classifier_attributes = set(classifier_args.kwonlydefaults.keys())
                classifier_attributes.update(
                    {arg for arg in classifier_args.args if arg != "self"}
                )
            else:
                raise AttributeError("No kwonlydefaults")
        except AttributeError:  # sometimes everything is being hidden in **kwargs
            # (e.g. it's the case for sklearn wrapper of xgboost),
            # then fallback to default constructor
            instance = self.classifier_class()
            classifier_attributes = instance.__dict__
        return {
            key: val
            for key, val in initial_instance_parameters.items()
            if key in classifier_attributes
            or (
                "_".join(key.split("_")[1:]) in classifier_attributes
                and (class_name is None or key.split("_", maxsplit=1)[0] == class_name)
            )
        }


def eval_model_mean_average_precision_neg(
    model: BaseModel,
    val_df: pd.DataFrame,
    selected_class_name: str = None,
    min_number_of_positive_cases: int = 2,
):
    """
    Helper function for model evaluation
    """
    y_pred = model.predict_proba(val_df, selected_class_name=selected_class_name)

    average_precisions = []
    for class_i, class_name in enumerate(model.config.class_names):
        if class_name != model.config.neg_val and (
            selected_class_name is None or class_name == selected_class_name
        ):
            y_true = val_df[model.config.target_col_name].map(lambda x: class_name in x)
            if sum(y_true) >= min_number_of_positive_cases:
                average_precision = average_precision_score(y_true, y_pred[:, class_i])
                average_precisions.append(average_precision)
                print(
                    selected_class_name,
                    class_name,
                    " !!!: ",
                    np.mean(y_true),
                    np.mean(y_pred[:, class_i]),
                    average_precision,
                )

    return 1 - np.mean(average_precisions)
