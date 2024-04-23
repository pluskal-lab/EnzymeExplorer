import logging
import pickle
from dataclasses import dataclass
from typing import Type

from xgboost import XGBClassifier

from src.models.ifaces import EmbsSklearnModel, SklearnBaseConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PlmXgbConfig(SklearnBaseConfig):
    """
    A data class to store Xgb model attributes
    """

    n_estimators: int
    learning_rate: float
    reg_lambda: float
    reg_alpha: float
    gamma: float
    max_depth: int
    max_leaves: int
    booster: str
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    colsample_bynode: float
    scale_pos_weight: float
    min_child_weight: int
    max_delta_step: int
    tree_method: str
    n_jobs: int
    gpu_id: int
    predictor: str
    validate_parameters: bool
    objective: str
    verbose: bool
    per_class_optimization: bool
    fold_i: int
    reuse_existing_partial_results: bool


class PlmXgb(EmbsSklearnModel):
    def __init__(
        self,
        config: PlmXgbConfig,
    ):
        super().__init__(
            config=config,
        )
        for param, value in config.__dict__.items():
            setattr(self, param, value)
        self.config = config
        self.classifier_class = XGBClassifier

    def save(self):
        with open(
            self.output_root / f"model_fold_{self.config.experiment_info.fold}.pkl",
            "wb",
        ) as file:
            pickle.dump(self, file)

    @classmethod
    def config_class(cls) -> Type[PlmXgbConfig]:
        return PlmXgbConfig
