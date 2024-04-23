"""This module defines an abstract class for models"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml  # type: ignore

from src.utils.project_info import ExperimentInfo


@dataclass
class BaseConfig:
    """
    A data class to store model attributes
    """

    experiment_info: ExperimentInfo
    id_col_name: str
    target_col_name: str
    split_col_name: str
    class_names: list[str]
    optimize_hyperparams: bool
    n_calls_hyperparams_opt: int
    hyperparam_dimensions: dict[
        str,
    ]
    neg_val: str
    negatives_sample_path: str
    tps_cleaned_csv_path: str
    random_state: int

    @classmethod
    def load(cls, path_to_config: Union[str, Path]) -> dict:
        """
        This class function loads config from a configs folder
        :param path_to_config:
        :return: a dictionary loaded from the config yaml
        """
        with open(path_to_config, encoding="utf-8") as file:
            configs_dict = yaml.load(file, Loader=yaml.FullLoader)
        return configs_dict


@dataclass
class SklearnBaseConfig(BaseConfig):
    """
    A data class to store scikit-learn downstream models leveraging precomputed embeddings
    """

    max_train_negs_proportion: float
    save_trained_model: bool
    representations_path: str
