"""A class for Logistic regression predictive models on top of protein language model (PLM) embeddings with
comparative domain features"""
from typing import Type

from sklearn.neural_network import MLPClassifier  # type: ignore

from enzymeexplorer.src.models.config_classes import EmbMLPConfig
from enzymeexplorer.src.models.plm_domain_comparison_randomforest import (
    PlmRandomForest,
)

# pylint: disable=R0903, R0901
class PlmMLP(PlmRandomForest):
    """
    Multi-layer Perceptron on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbMLPConfig,
    ):
        super().__init__(
            config=config,
        )
        self.classifier_class = MLPClassifier

    @classmethod
    def config_class(cls) -> Type[EmbMLPConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbMLPConfig
