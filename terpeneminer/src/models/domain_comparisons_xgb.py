"""A class for XGBoost-based predictive models working with comparisons between structural domains"""
import logging
from typing import Type

from xgboost import XGBClassifier  # type: ignore

from terpeneminer.src.models.ifaces import DomainsSklearnModel
from terpeneminer.src.models.config_classes import FeaturesXGbConfig

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# pylint: disable=R0903
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
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return FeaturesXGbConfig
