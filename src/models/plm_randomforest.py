import logging
import pickle
from dataclasses import dataclass
from typing import Type, Optional

from sklearn.ensemble import RandomForestClassifier

from src.models.ifaces import EmbsSklearnModel, SklearnBaseConfig

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PlmRandomForestConfig(SklearnBaseConfig):
    """
    A data class to store model attributes
    """

    n_estimators: int
    n_jobs: int
    class_weight: str
    max_depth: int
    fold_i: str
    per_class_optimization: bool
    reuse_existing_partial_results: bool
    load_per_class_params_from: Optional[str] = None


class PlmRandomForest(EmbsSklearnModel):
    def __init__(
        self,
        config: PlmRandomForestConfig,
    ):
        super().__init__(
            config=config,
        )
        for param, value in config.__dict__.items():
            setattr(self, param, value)
        self.config = config
        self.classifier_class = RandomForestClassifier

    def save(self):
        with open(
            self.output_root / f"model_fold_{self.config.experiment_info.fold}.pkl",
            "wb",
        ) as file:
            pickle.dump(self, file)

    @classmethod
    def config_class(cls) -> Type[PlmRandomForestConfig]:
        return PlmRandomForestConfig
