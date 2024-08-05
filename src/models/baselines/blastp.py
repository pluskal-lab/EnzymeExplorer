""" This is a wrapper to use the BLASTp for substrate prediction. """
from dataclasses import dataclass
from typing import Type, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from profun.models import BlastMatching  # type: ignore

from src.models.ifaces import BaseModel, BaseConfig


@dataclass
class BlastConfig(BaseConfig):
    """
    A data class to store Blast-model attributes
    """

    n_neighbours: int
    e_threshold: float
    seq_col_name: str
    pred_batch_size: Optional[int] = 32
    n_jobs: Optional[int] = 64


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
        """
        Function for training the core components of the model.

        :param train_df: A pandas DataFrame containing the training data.
        :param class_name: The name of a class for separate model fitting for the class. Defaults to None.
        """

        self.blast_matcher.fit_core(train_df, class_name)

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Function to predict the class probabilities for the given validation data.

        :param val_df: A pandas DataFrame containing the validation data.
        :param selected_class_name: An optional parameter for selecting a class. Defaults to None.
                                    Note: This model does not support class selection.

        :return: A numpy ndarray containing the predicted class probabilities.
        """

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
