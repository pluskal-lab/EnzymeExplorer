"""A class for XGBoost predictive models on top of protein language model (PLM) embeddings"""
from typing import Type

from xgboost import XGBClassifier  # type: ignore

from src.models.config_classes import EmbsXGbConfig
from src.models.ifaces import EmbsSklearnModel


# pylint: disable=R0903
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
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbsXGbConfig
