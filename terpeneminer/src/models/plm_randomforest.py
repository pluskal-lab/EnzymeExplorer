"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
from typing import Type
import pickle
from sklearn.ensemble import RandomForestClassifier  # type: ignore

from terpeneminer.src.models.config_classes import (
    EmbRandomForestConfig,
    EmbMLPConfig,
    EmbLogisticRegressionConfig,
)
from terpeneminer.src.models.ifaces import EmbsSklearnModel, BaseConfig


# pylint: disable=R0903
class PlmRandomForest(EmbsSklearnModel):
    """
    Random Forest on top of protein language model (PLM) embeddings
    """

    def __init__(
        self,
        config: EmbRandomForestConfig | EmbMLPConfig | EmbLogisticRegressionConfig,
    ):
        super().__init__(
            config=config,
        )
        self.classifier_class = RandomForestClassifier
        if "plm_subset" in self.config.experiment_info.model_version:
            # to obtain the subset of domain features, run the following code:
            # python -m src.models.plm_domain_faster.get_domains_feature_importances
            with open("data/plm_feats_subset.pkl", "rb") as file:
                self.plm_feat_indices_subset = sorted(pickle.load(file))
        else:
            self.plm_feat_indices_subset = None
        if self.plm_feat_indices_subset is not None:
            self.features_df["Emb"] = self.features_df["Emb"].apply(
                lambda x: x[self.plm_feat_indices_subset]
            )

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbRandomForestConfig
