"""A class for Random Forest predictive models working with comparisons between structural domains"""
import logging
from typing import Type

from sklearn.ensemble import RandomForestClassifier  # type: ignore

from src.models.ifaces import DomainsSklearnModel
from src.models.config_classes import FeaturesRandomForestConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


# pylint: disable=R0903
class DomainsRandomForest(DomainsSklearnModel):
    """Random Forest on top of comparisons between structural domains"""

    def __init__(
        self,
        config: FeaturesRandomForestConfig,
    ):
        super().__init__(config=config)
        self.classifier_class = RandomForestClassifier

    @classmethod
    def config_class(cls) -> Type[FeaturesRandomForestConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return FeaturesRandomForestConfig
