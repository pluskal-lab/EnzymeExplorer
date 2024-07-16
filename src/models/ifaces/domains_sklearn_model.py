"""Abstract class for scikit-learn compatible models build on top of structural-domains comparisons"""
import pickle
from typing import Optional, Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .config_baseclasses import SklearnBaseConfig
from .features_sklearn_model import FeaturesSklearnModel


def compare_domains_to_known_instances(
    train_df: pd.DataFrame, model: Any, domain_indices_subset: Optional[set[int]] = None
) -> tuple[list[int], pd.DataFrame]:
    """
    A function storing comparisons to domains of trn proteins only to avoid leakage
    :param train_df: a training data
    :param model: predictive model
    :param domain_indices_subset: a subset of domain indices to consider
    :return: a list of training data domains and a dataframe with comparisons to the selected training domains
    """
    trn_uni_ids = set(train_df[model.config.id_col_name].values)
    allowed_feat_indices = []
    for required_model_attribute in [
        "uniid_2_column_ids",
        "all_ids_list_dom",
        "feats_dom_dists",
    ]:
        assert hasattr(
            model, required_model_attribute
        ), f"Model {model} has no attribute '{required_model_attribute}'"
    for trn_id in trn_uni_ids:
        allowed_feat_indices.extend(model.uniid_2_column_ids[trn_id])
    if domain_indices_subset is not None:
        allowed_feat_indices = list(set(allowed_feat_indices) & domain_indices_subset)
    features_df_domain_detections = pd.DataFrame(
        {
            model.config.id_col_name: model.all_ids_list_dom,
            "Emb": [
                model.feats_dom_dists[i][allowed_feat_indices]
                for i in range(len(model.feats_dom_dists))
            ],
        }
    )
    return allowed_feat_indices, features_df_domain_detections


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
        self.allowed_feat_indices: list[int] = None  # type: ignore
        self.features_df_domain_detections = None

    def _setup_features_df_for_current_data(self, input_df: pd.DataFrame):
        ids_with_domain_detections = set(self.all_ids_list_dom)
        ids_without_domain_detections = [
            uni_id
            for uni_id in input_df[self.config.id_col_name]
            if uni_id not in ids_with_domain_detections
        ]
        features_df_no_detected_domains = pd.DataFrame(
            {
                "Uniprot ID": ids_without_domain_detections,
                "Emb": [
                    np.zeros(len(self.allowed_feat_indices))
                    for _ in range(len(ids_without_domain_detections))
                ],
            }
        )
        self.features_df = pd.concat(
            (self.features_df_domain_detections, features_df_no_detected_domains)
        )

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        (
            self.allowed_feat_indices,
            self.features_df_domain_detections,
        ) = compare_domains_to_known_instances(train_df, self)
        self._setup_features_df_for_current_data(train_df)
        super().fit_core(train_df, class_name)

    def predict_proba(
        self, val_df: pd.DataFrame, selected_class_name: Optional[str] = None
    ) -> np.ndarray:
        self._setup_features_df_for_current_data(val_df)
        return super().predict_proba(val_df, selected_class_name)
