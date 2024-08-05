""" This is a wrapper to use the BLASTp for substrate prediction. """
from dataclasses import dataclass
from typing import Type, Optional
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from profun.models import FoldseekMatching  # type: ignore

from src.models.ifaces import BaseModel
from src.models.baselines.blastp import BlastConfig


@dataclass
class FoldseekConfig(BlastConfig):
    """
    A data class to store Blast-model attributes
    """

    local_pdb_storage_path: Optional[str | Path] = None


class Foldseek(BaseModel):
    """
    Folsdeek-matching, see https://github.com/SamusRam/ProFun/blob/main/profun/models/foldseek_model.py for details
    """

    def __init__(self, config):
        super().__init__(config=config)
        config.experiment_info.validation_schema = (
            "kfold"  # for compatibility with the profun package
        )
        self.foldseek_matcher = FoldseekMatching(config=config)

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """
        Function to train the core components of the model using the foldseek matcher.

        :param train_df: A pandas DataFrame containing the training data.
        :param class_name: The name of a class for separate model fitting for the class. Defaults to None.
        """

        self.foldseek_matcher.fit_core(train_df, class_name)

    def predict_proba(
            self,
            val_df: pd.DataFrame,
            selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Function to predict class probabilities for the given validation data using the foldseek matcher.

        :param val_df: A pandas DataFrame containing the validation data.
        :param selected_class_name: An optional parameter for selecting a class. Defaults to None.
                                    Note: This model does not support class selection and will raise an assertion error if a class name is provided.

        :return: A numpy ndarray containing the predicted class probabilities.
        """

        assert (
                selected_class_name is None
        ), "This model does not support class selection."
        return self.foldseek_matcher.predict_proba(val_df)

    @classmethod
    def config_class(cls) -> Type[FoldseekConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return FoldseekConfig
