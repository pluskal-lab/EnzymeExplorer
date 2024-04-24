"""Abstract class for scikit-learn compatible models build on top of structural-domains comparisons"""
import pickle

import pandas as pd  # type: ignore

from .config_baseclasses import SklearnBaseConfig
from .features_sklearn_model import FeaturesSklearnModel


class DomainsSklearnModel(FeaturesSklearnModel):
    """
    Interface of scikit-learn compatible models build on top of structural-domains comparisons
    """

    def __init__(self, config: SklearnBaseConfig):
        super().__init__(config=config)
        for param, value in config.__dict__.items():
            setattr(self, param, value)
        self.config = config
        with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
            (
                self.feats_dom_dists,
                self.all_ids_list_dom,
                self.uniid_2_column_ids,
                _,
            ) = pickle.load(file)
        self.features_df = None

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        # comparisons to domains of trn proteins only to avoid leakage
        trn_uni_ids = set(train_df["Uniprot ID"].values)
        allowed_feat_indices = []
        for trn_id in trn_uni_ids:
            allowed_feat_indices.extend(self.uniid_2_column_ids[trn_id])
        self.features_df = pd.DataFrame(
            {
                "Uniprot ID": self.all_ids_list_dom,
                "Emb": [
                    self.feats_dom_dists[i][allowed_feat_indices]
                    for i in range(len(self.feats_dom_dists))
                ],
            }
        )
        super().fit_core(train_df, class_name)
