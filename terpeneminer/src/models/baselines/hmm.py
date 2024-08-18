""" This is a wrapper to use the BLASTp for substrate prediction. """
from dataclasses import dataclass
from typing import Type, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from profun.models import ProfileHMM  # type: ignore

from terpeneminer.src.models.ifaces import BaseModel, BaseConfig


@dataclass
class HmmConfig(BaseConfig):
    """
    A config class for profile HMM
    """

    search_e_threshold: float
    zero_conf_level: float
    seq_col_name: str
    group_column_name: Optional[str] = None
    n_jobs: Optional[int] = 56
    pred_batch_size: Optional[int] = 10000


class HMM(BaseModel):
    """
    Profile Hidden Markov Model, see https://github.com/SamusRam/ProFun/blob/main/profun/models/hmm for details
    """

    def __init__(self, config):
        super().__init__(config=config)
        config.experiment_info.validation_schema = (
            "kfold"  # for compatibility with the profun package
        )
        self.hmm = ProfileHMM(config=config)

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """
        Function to train the core components of the model using the HMM (Hidden Markov Model).

        :param train_df: A pandas DataFrame containing the training data.
        :param class_name: The name of a class for separate model fitting for the class. Defaults to None.
        """

        self.hmm.fit_core(train_df, class_name)

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Function to predict class probabilities for the given validation data using the Hidden Markov Model (HMM).

        :param val_df: A pandas DataFrame containing the validation data.
        :param selected_class_name: An optional parameter for selecting a class. Defaults to None.
                                    Note: This model does not support class selection and will raise an assertion error if a class name is provided.

        :return: A numpy ndarray containing the predicted class probabilities.
        """

        assert (
            selected_class_name is None
        ), "This model does not support class selection."
        return self.hmm.predict_proba(val_df)

    @classmethod
    def config_class(cls) -> Type[HmmConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return HmmConfig
