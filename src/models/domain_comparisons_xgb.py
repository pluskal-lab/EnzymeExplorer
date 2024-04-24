"""A class for XGBoost-based predictive models working with comparisons between structural domains"""
import logging
from typing import Type

from xgboost import XGBClassifier  # type: ignore

from src.models.ifaces import DomainsSklearnModel, FeaturesXGbConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class DomainsXgb(DomainsSklearnModel):
    """
    XGBClassifier on top of comparisons between structural domains
    """

    def __init__(
        self,
        config: FeaturesXGbConfig,
    ):

        super().__init__(config=config)
        self.classifier_class = XGBClassifier

    @classmethod
    def config_class(cls) -> Type[FeaturesXGbConfig]:
        return FeaturesXGbConfig
