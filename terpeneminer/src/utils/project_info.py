"""This script contains routines for working with project info"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json  # type: ignore


@dataclass_json
@dataclass
class ExperimentInfo:
    """A dataclass to store information about a particular experiment scenario"""

    model_type: str
    model_version: str

    def __post_init__(self):
        """Setting up an experiment timestamp and fold info"""
        self._timestamp = datetime.datetime.now()
        self._fold = "all_folds"
        self._class_name = "all_classes"

    @property
    def fold(self):
        """Fold variable getter"""
        return self._fold

    @fold.setter
    def fold(self, value: str):
        """Fold variable setter"""
        self._fold = value

    @property
    def class_name(self):
        """Class name variable getter"""
        return self._class_name

    @class_name.setter
    def class_name(self, value: str):
        """Class name variable setter"""
        self._class_name = value

    def get_experiment_name(self):
        """Detailed experiment name getter"""
        experiment_name = (
            f"{self.model_type}_{self.model_version}_fold_{self.fold}_{self.class_name}_"
            f'{self._timestamp.strftime("%Y%m%d-%H%M%S")}'
        )
        return experiment_name

    def __lt__(self, other):
        if self.model_type != other.model_type:
            return self.model_type < other.model_type
        if self.model_version != other.model_version:
            return self.model_version < other.model_version
        if self.fold != other.fold:
            return self.fold < other.fold
        if self.class_name != other.class_name:
            return self.class_name < other.class_name
        return False

    @property
    def timestamp(self):
        """Fold variable getter"""
        return self._timestamp

    def __str__(self):
        return f"Experiment: {self.model_type}/{self.model_version}/fold {self.fold}/class {self.class_name}"


def get_project_root() -> Path:
    """
    Returns: absolute path to the project root directory
    """
    return Path(__file__).parent.parent.parent


def get_output_root() -> Path:
    """
    Returns: absolute path to the output directory
    """
    return get_project_root() / "../outputs"


def get_experiments_output() -> Path:
    """
    Returns: absolute path to the experiments directory
    """
    return get_output_root() / "experiment_results"


def get_evaluations_output() -> Path:
    """
    Returns: absolute path to the output of evaluation
    """
    return get_output_root() / "evaluation_results"


def get_config_root() -> Path:
    """
    Returns: absolute path to the config directory
    """
    return Path(get_project_root()) / "configs"
