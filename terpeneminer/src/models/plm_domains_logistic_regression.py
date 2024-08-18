"""A class for Logistic regression predictive models on top of protein language model (PLM) embeddings with
comparative domain features"""
from typing import Type

from sklearn.linear_model import LogisticRegression  # type: ignore

from terpeneminer.src.models.config_classes import EmbLogisticRegressionConfig
from terpeneminer.src.models import PlmDomainsRandomForest


# pylint: disable=R0903, R0901
class PlmDomainsLogisticRegression(PlmDomainsRandomForest):
    """
    Random Forest on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbLogisticRegressionConfig,
    ):
        super().__init__(
            config=config,
        )
        self.classifier_class = LogisticRegression

    @classmethod
    def config_class(cls) -> Type[EmbLogisticRegressionConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbLogisticRegressionConfig
