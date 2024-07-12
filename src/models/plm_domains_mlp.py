"""A class for Logistic regression predictive models on top of protein language model (PLM) embeddings with
comparative domain features"""
from typing import Type

from sklearn.neural_network import MLPClassifier  # type: ignore

from src.models.config_classes import EmbMLPConfig
from src.models import PlmDomainsRandomForest


# pylint: disable=R0903
class PlmDomainsMLP(PlmDomainsRandomForest):
    """
    Random Forest on top of protein language model (PLM) embeddings
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
