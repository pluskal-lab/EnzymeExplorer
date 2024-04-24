"""A class for XGBoost predictive models on top of protein language model (PLM) embeddings"""
from typing import Type

from xgboost import XGBClassifier  # type: ignore

from src.models.ifaces import EmbsSklearnModel, EmbsXGbConfig


class PlmXgb(EmbsSklearnModel):
    """
    Random Forest on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbsXGbConfig,
    ):

        super().__init__(
            config=config,
        )
        self.classifier_class = XGBClassifier

    @classmethod
    def config_class(cls) -> Type[EmbsXGbConfig]:
        return EmbsXGbConfig
