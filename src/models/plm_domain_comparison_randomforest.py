"""A class for Random Forest predictive models on top of protein language model (PLM) embeddings"""
import pickle
from typing import Type

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore

from src.models.ifaces import (
    EmbRandomForestConfig,
    EmbMLPConfig,
    EmbLogisticRegressionConfig,
)
from src.models import PlmRandomForest


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
        self.classifier_class = RandomForestClassifier
        with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
            (
                self.feats_dom_dists,
                self.all_ids_list_dom,
                self.uniid_2_column_ids,
                _,
            ) = pickle.load(file)
        self.allowed_feat_indices = None
        self.features_df_plm = self.features_df.copy()

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        # comparisons to domains of trn proteins only to avoid leakage
        trn_uni_ids = set(train_df["Uniprot ID"].values)
        self.allowed_feat_indices = []
        for trn_id in trn_uni_ids:
            self.allowed_feat_indices.extend(self.uniid_2_column_ids[trn_id])
        dom_features_df = pd.DataFrame(
            {
                "Uniprot ID": self.all_ids_list_dom,
                "Emb_dom": [
                    self.feats_dom_dists[i][self.allowed_feat_indices]
                    for i in range(len(self.feats_dom_dists))
                ],
            }
        )

        self.features_df = self.features_df_plm.merge(
            dom_features_df, on="Uniprot ID", how="left"
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
    def config_class(cls) -> Type[EmbRandomForestConfig]:
        return EmbRandomForestConfig
