"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
import pickle
from typing import Type

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from terpeneminer.src.models.config_classes import (
    EmbRandomForestConfig,
    EmbMLPConfig,
    EmbLogisticRegressionConfig,
)
from terpeneminer.src.models import PlmRandomForest
from terpeneminer.src.models.ifaces import BaseConfig
from terpeneminer.src.models.ifaces.domains_sklearn_model import (
    compare_domains_to_known_instances,
)


# pylint: disable=R0903
class PlmDomainsRandomForest(PlmRandomForest):
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
        # pylint: disable=R0801
        with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
            (
                self.feats_dom_dists,
                self.all_ids_list_dom,
                self.uniid_2_column_ids,
                _,
            ) = pickle.load(file)
        self.allowed_feat_indices: list[int] = None  # type: ignore
        self.features_df_plm = self.features_df.copy()
        self.features_df = None
        # to experiment with the domain features subset
        if "domains_subset" in self.config.experiment_info.model_version:
            # to obtain the subset of domain features, run the following code:
            # python -m src.models.plm_domain_faster.get_domains_feature_importances
            with open("data/domains_subset.pkl", "rb") as file:
                _, self.feat_indices_subset = pickle.load(file)
        else:
            self.feat_indices_subset = set()

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """
        Function for training model instance
        :param train_df: pandas dataframe containing training data
        :param class_name: name of a class for the separate model fitting for the class
        """
        # comparisons to domains of trn proteins only to avoid leakage
        (
            self.allowed_feat_indices,
            dom_features_df,
        ) = compare_domains_to_known_instances(train_df, self, self.feat_indices_subset)
        dom_features_df["Emb_dom"] = dom_features_df["Emb"]

        self.features_df = self.features_df_plm.merge(
            dom_features_df[[self.config.id_col_name, "Emb_dom"]],
            on=self.config.id_col_name,
            how="left",
        )
        missing_dist_feats_bool_idx = self.features_df["Emb_dom"].isnull()
        self.features_df.loc[missing_dist_feats_bool_idx, "Emb_dom"] = pd.Series(
            [
                np.zeros(len(self.allowed_feat_indices))
                for _ in range(sum(missing_dist_feats_bool_idx))
            ],
            index=self.features_df.loc[missing_dist_feats_bool_idx].index,
        )

        self.features_df["Emb"] = self.features_df.apply(
            lambda row: np.concatenate((row["Emb"], row["Emb_dom"])), axis=1
        )
        self.features_df.drop("Emb_dom", axis=1, inplace=True)

        super().fit_core(train_df, class_name)

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return EmbRandomForestConfig
