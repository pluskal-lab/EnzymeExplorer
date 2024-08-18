"""This module defines an dataclasses storing model configs"""
from dataclasses import dataclass

from terpeneminer.src.models.ifaces import SklearnBaseConfig
from terpeneminer.src.models.ifaces.config_baseclasses import EmbSklearnBaseConfig


@dataclass
class FeaturesXGbConfig(SklearnBaseConfig):
    """
    A data class to store some Xgb model attributes
    """

    booster: str
    n_jobs: int
    objective: str
    reg_lambda: float
    gamma: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    scale_pos_weight: float
    min_child_weight: int
    n_estimators: int


@dataclass
class FeaturesRandomForestConfig(SklearnBaseConfig):
    """
    A data class to store model attributes
    """

    n_estimators: int
    n_jobs: int
    class_weight: str
    max_depth: int
    per_class_with_multilabel_regularization: int


@dataclass
class EmbRandomForestConfig(EmbSklearnBaseConfig, FeaturesRandomForestConfig):
    """
    A data class to store the corresponding model attributes
    """


@dataclass
class EmbsXGbConfig(EmbSklearnBaseConfig, FeaturesXGbConfig):
    """
    A data class to store the corresponding model attributes
    """


@dataclass
class EmbMLPConfig(EmbSklearnBaseConfig, SklearnBaseConfig):
    """
    A data class for MLP config
    """

    hidden_layer_sizes: int
    alpha: float
    max_iter: int


@dataclass
class EmbLogisticRegressionConfig(EmbSklearnBaseConfig, SklearnBaseConfig):
    """
    A data class for Logistic Regression config
    """

    penalty: str
    tol: float
    C: float  # pylint: disable=C0103
    max_iter: int
    solver: str
    requires_multioutputwrapper_for_multilabel: bool
