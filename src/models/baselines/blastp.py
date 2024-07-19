""" This is a wrapper to use the BLASTp for substrate prediction. """
from dataclasses import dataclass
from typing import Type, Optional

import numpy as np
import pandas as pd
from profun.models import BlastMatching, BlastConfig as BlastConfigPackage

from src.models.ifaces import BaseModel, BaseConfig


@dataclass
class BlastConfig(BaseConfig):
    """
    A data class to store Blast-model attributes
    """

    n_neighbours: int
    e_threshold: float
    seq_col_name: str


class Blastp(BaseModel):
    """
    BLASTp-matching, see https://github.com/SamusRam/ProFun/blob/main/profun/models/blast_model.py for details
    """

    def __init__(self, config):
        super().__init__(config=config)
        config.experiment_info.validation_schema = (
            "kfold"  # for compatibility with the profun package
        )
        self.blast_matcher = BlastMatching(config=config)

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        self.blast_matcher.fit_core(train_df, class_name)

    def predict_proba(
        self,
        val_df: pd.DataFrame | np.ndarray,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        assert (
            selected_class_name is None
        ), "This model does not support class selection."
        return self.blast_matcher.predict_proba(val_df)

    @classmethod
    def config_class(cls) -> Type[BlastConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return BlastConfig
