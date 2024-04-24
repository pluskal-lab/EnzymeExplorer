"""A class for Random Forest predictive models working with comparisons between structural domains"""
import logging
from typing import Type

from sklearn.ensemble import RandomForestClassifier  # type: ignore

from src.models.ifaces import DomainsSklearnModel, FeaturesRandomForestConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


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
        return FeaturesRandomForestConfig
