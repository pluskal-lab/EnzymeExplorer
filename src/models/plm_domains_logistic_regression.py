"""A class for Logistic regression predictive models on top of protein language model (PLM) embeddings with
comparative domain features"""
from typing import Type

from sklearn.linear_model import LogisticRegression  # type: ignore

from src.models.ifaces import EmbLogisticRegressionConfig
from src.models import PlmDomainsRandomForest


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
        return EmbLogisticRegressionConfig
