"""An abstract class inheriting from FeaturesSklearnModel, where features are protein language model (PLM) embeddings"""
import os.path

import pandas as pd  # type: ignore

from .config_baseclasses import EmbSklearnBaseConfig
from .features_sklearn_model import FeaturesSklearnModel


class EmbsSklearnModel(FeaturesSklearnModel):
    """
    An abstract class for sklearn-compatible models on top of protein language model (PLM) embeddings
    """

    def __init__(self, config: EmbSklearnBaseConfig):
        super().__init__(config=config)
        for param, value in config.__dict__.items():
            setattr(self, param, value)
        representations_path: str = config.representations_path
        if not os.path.exists(representations_path):
            raise FileNotFoundError(
                f"File with pre-computed embeddings {representations_path} does not exist"
            )

        self.features_df: pd.DataFrame = pd.read_hdf(representations_path)
        assert isinstance(self.features_df, pd.DataFrame), (
            "Expected embeddings to be stored as pandas Dataframe"
            f"But {type(self.features_df)} found at {representations_path}"
        )
        self.features_df.drop_duplicates(subset=[self.config.id_col_name], inplace=True)
        self.features_df.columns = [self.config.id_col_name, "Emb"]
