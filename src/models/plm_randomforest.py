"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
from typing import Type

from sklearn.ensemble import RandomForestClassifier  # type: ignore

from src.models.ifaces import EmbRandomForestConfig, EmbsSklearnModel


class PlmRandomForest(EmbsSklearnModel):
    """
    Random Forest on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbRandomForestConfig,
    ):
        super().__init__(
            config=config,
        )
        self.classifier_class = RandomForestClassifier

    @classmethod
    def config_class(cls) -> Type[EmbRandomForestConfig]:
        return EmbRandomForestConfig
