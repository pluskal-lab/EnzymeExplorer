"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
from typing import Type

from sklearn.ensemble import RandomForestClassifier  # type: ignore

from terpeneminer.src.models.config_classes import (
    EmbRandomForestConfig,
    EmbMLPConfig,
    EmbLogisticRegressionConfig,
)
from terpeneminer.src.models.ifaces import EmbsSklearnModel, BaseConfig


# pylint: disable=R0903
class PlmRandomForest(EmbsSklearnModel):
    """
    Random Forest on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbRandomForestConfig | EmbMLPConfig | EmbLogisticRegressionConfig,
    ):
        super().__init__(
            config=config,
        )
        self.classifier_class = RandomForestClassifier

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbRandomForestConfig
